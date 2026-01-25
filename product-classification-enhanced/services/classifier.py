import json
import os
from typing import Dict, List, Optional
from PIL import Image
import logging
from pathlib import Path

from models.embeddings import EmbeddingGenerator
from core.config import Config
from core.constants import CONFIDENCE_LEVELS, DECISIONS, DEFAULT_CATEGORIES
from models.clip_manager import clip_manager

logger = logging.getLogger(__name__)

class ProductClassifier:
    """Enhanced product classifier"""
    
    def __init__(self, categories_file: str = None):
        self.config = Config()
        self.clip_manager = clip_manager # Pass the global instance or initialize one
        self.embedding_generator = EmbeddingGenerator(self.clip_manager)
        
        # Load or create categories
        self.categories_file = categories_file or self.config.CATEGORIES_FILE
        self.categories = self._load_categories()
        
        # Precompute category embeddings
        self.category_embeddings = self._precompute_embeddings()
    
    def _load_categories(self) -> Dict:
        """Load categories from file or create default"""
        if os.path.exists(self.categories_file):
            try:
                with open(self.categories_file, 'r') as f:
                    categories = json.load(f)
                logger.info(f"Loaded categories from {self.categories_file}")
                return categories
            except Exception as e:
                logger.warning(f"Failed to load categories: {e}. Creating default.")
        
        # Create default categories
        default_categories = {
            category: [f"{category}_item_{i}" for i in range(1, 6)]
            for category in DEFAULT_CATEGORIES
        }
        
        # Save default categories
        self._save_categories(default_categories)
        return default_categories
    
    def _save_categories(self, categories: Dict):
        """Save categories to file"""
        try:
            os.makedirs(os.path.dirname(self.categories_file), exist_ok=True)
            with open(self.categories_file, 'w') as f:
                json.dump(categories, f, indent=2)
            logger.info(f"Saved categories to {self.categories_file}")
        except Exception as e:
            logger.error(f"Failed to save categories: {e}")
    
    def _precompute_embeddings(self) -> Dict:
        """Precompute embeddings for all categories"""
        embeddings = {}
        
        for category, subcategories in self.categories.items():
            # Create descriptive text for the category
            category_text = f"{category} products including {', '.join(subcategories[:3])}"
            
            # Generate embedding
            try:
                emb = self.embedding_generator.get_text_embedding(category_text)
                embeddings[category] = emb
                logger.debug(f"Precomputed embedding for category: {category}")
            except Exception as e:
                logger.error(f"Failed to precompute embedding for {category}: {e}")
        
        logger.info(f"Precomputed embeddings for {len(embeddings)} categories")
        return embeddings
    
    def classify(self, image: Image.Image, top_k: int = 3) -> List[Dict]:
        """Classify a product image"""
        try:
            # Generate image embedding
            image_emb = self.embedding_generator.get_image_embedding(image)
            
            # Calculate similarity with all categories
            results = []
            for category, category_emb in self.category_embeddings.items():
                similarity = self.embedding_generator.get_similarity(image_emb, category_emb)
                
                # Determine confidence level
                confidence = self._get_confidence_level(similarity)
                
                results.append({
                    'category': category,
                    'similarity': round(similarity, 4),
                    'confidence': confidence,
                    'description': f"Likely a {category} product"
                })
            
            # Sort by similarity (highest first)
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Return top K results
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise
    
    def verify_domain(self, image: Image.Image, domain_text: str, company_name: str = None) -> Dict:
        """Verify if product belongs to a specific domain"""
        try:
            # Generate image embedding
            image_emb = self.embedding_generator.get_image_embedding(image)
            
            # Create multiple prompts for better matching
            prompts = self._create_domain_prompts(domain_text, company_name)
            
            # Calculate similarity with all prompts
            similarities = []
            for prompt in prompts:
                text_emb = self.embedding_generator.get_text_embedding(prompt)
                similarity = self.embedding_generator.get_similarity(image_emb, text_emb)
                similarities.append(similarity)
            
            # Get the best match
            best_similarity = max(similarities) if similarities else 0.0
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            
            # Make decision
            decision, confidence = self._make_verification_decision(best_similarity)
            
            return {
                'best_similarity': round(best_similarity, 4),
                'average_similarity': round(avg_similarity, 4),
                'decision': decision,
                'confidence': confidence,
                'explanation': DECISIONS.get(decision, 'Unknown'),
                'prompts_used': len(prompts)
            }
            
        except Exception as e:
            logger.error(f"Domain verification failed: {e}")
            raise
    
    def _create_domain_prompts(self, domain_text: str, company_name: str = None) -> List[str]:
        """Create multiple prompts for domain verification"""
        prompts = []
        
        # Basic domain text
        prompts.append(domain_text)
        
        # If company name is provided, add company-specific prompts
        if company_name:
            prompts.extend([
                f"{company_name} products: {domain_text}",
                f"Products sold by {company_name}: {domain_text}",
                f"{company_name}'s {domain_text}"
            ])
        
        # Extract main terms from domain text
        domain_terms = [term.strip() for term in domain_text.split(',')]
        if domain_terms:
            main_term = domain_terms[0]
            
            # Add descriptive prompts
            prompts.extend([
                f"a photo of {main_term}",
                f"product image: {main_term}",
                f"commercial product: {main_term}",
                f"{main_term} for sale",
                f"photo showing {main_term}"
            ])
        
        # Add combination prompts
        if len(domain_terms) > 1:
            prompts.append(f"{domain_terms[0]} and {domain_terms[1]}")
        
        # Remove duplicates
        return list(set(prompts))
    
    def _get_confidence_level(self, similarity: float) -> str:
        """Get confidence level based on similarity score"""
        if similarity >= CONFIDENCE_LEVELS['VERY_HIGH']:
            return 'VERY_HIGH'
        elif similarity >= CONFIDENCE_LEVELS['HIGH']:
            return 'HIGH'
        elif similarity >= CONFIDENCE_LEVELS['MEDIUM']:
            return 'MEDIUM'
        elif similarity >= CONFIDENCE_LEVELS['LOW']:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def _make_verification_decision(self, similarity: float) -> tuple:
        """Make verification decision based on similarity"""
        if similarity >= self.config.IN_DOMAIN_THRESHOLD:
            decision = 'IN_DOMAIN'
        elif similarity >= self.config.NEAR_DOMAIN_THRESHOLD:
            decision = 'NEAR_DOMAIN'
        else:
            decision = 'OUT_OF_DOMAIN'
        
        confidence = self._get_confidence_level(similarity)
        return decision, confidence
    
    def add_category(self, category_name: str, subcategories: List[str]):
        """Add a new category"""
        self.categories[category_name] = subcategories
        self._save_categories(self.categories)
        
        # Update embeddings
        category_text = f"{category_name} products including {', '.join(subcategories[:3])}"
        emb = self.embedding_generator.get_text_embedding(category_text)
        self.category_embeddings[category_name] = emb
        logger.info(f"Added new category: {category_name}")


# Global instance
classifier = ProductClassifier()
