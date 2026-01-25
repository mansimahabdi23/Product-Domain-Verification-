import json
import os
from typing import Dict, List, Optional, Tuple
from PIL import Image
import logging
from pathlib import Path
import torch
import torch.nn.functional as F

from models.clip_manager import clip_manager
from core.config import Config
from core.constants import CONFIDENCE_LEVELS, DECISIONS, DEFAULT_CATEGORIES

logger = logging.getLogger(__name__)

class ProductClassifier:
    """Enhanced product classifier"""
    
    def __init__(self, categories_file: str = None):
        self.config = Config()
        
        # Initialize model
        self.model, self.processor = clip_manager.get_model()
        self.device = clip_manager.device
        
        # Load or create categories
        self.categories_file = categories_file or self.config.CATEGORIES_FILE
        self.categories = self._load_categories()
        
        # Precompute category embeddings
        self.category_embeddings = {}
        self._precompute_embeddings()
    
    def _load_categories(self) -> Dict:
        """Load categories from file or create default"""
        if os.path.exists(self.categories_file):
            try:
                with open(self.categories_file, 'r') as f:
                    categories = json.load(f)
                logger.info(f"✅ Loaded {len(categories)} categories from {self.categories_file}")
                return categories
            except Exception as e:
                logger.warning(f"❌ Failed to load categories: {e}. Creating default.")
        
        # Create default categories
        default_categories = {
            "electronics": [
                "television", "smartphone", "laptop", "tablet", "camera",
                "headphones", "speakers", "smartwatch", "gaming console"
            ],
            "furniture": [
                "sofa", "bed", "table", "chair", "desk", "wardrobe",
                "shelf", "cabinet", "ottoman"
            ],
            "clothing": [
                "t-shirt", "jeans", "dress", "jacket", "shirt", "pants",
                "shorts", "skirt", "sweater", "hoodie"
            ],
            "kitchen_appliances": [
                "refrigerator", "oven", "microwave", "blender", "toaster",
                "coffee maker", "dishwasher", "mixer"
            ],
            "home_decor": [
                "lamp", "rug", "curtain", "painting", "vase", "mirror",
                "clock", "candle", "pillow"
            ],
            "sports": [
                "basketball", "football", "tennis racket", "golf clubs",
                "bicycle", "treadmill", "yoga mat", "dumbbells"
            ]
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
            logger.info(f"✅ Saved {len(categories)} categories to {self.categories_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save categories: {e}")
    
    def _precompute_embeddings(self):
        """Precompute embeddings for all categories"""
        logger.info(f"🔄 Precomputing embeddings for {len(self.categories)} categories...")
        
        for category, subcategories in self.categories.items():
            # Create descriptive text for the category
            if subcategories and len(subcategories) > 0:
                # Use first 3 subcategories for description
                description = f"{category} products including {', '.join(subcategories[:3])}"
            else:
                description = f"{category} products"
            
            # Generate embedding
            try:
                inputs = self.processor(text=[description], return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    emb = self.model.get_text_features(**inputs)
                    emb = F.normalize(emb, dim=-1)
                
                self.category_embeddings[category] = emb.cpu()
                logger.debug(f"  ✓ Precomputed embedding for: {category}")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to precompute embedding for {category}: {e}")
                # Create a zero embedding as fallback
                self.category_embeddings[category] = torch.zeros(1, 512)
        
        logger.info(f"✅ Precomputed embeddings for {len(self.category_embeddings)} categories")
    
    def _get_image_embedding(self, image: Image.Image) -> torch.Tensor:
        """Generate embedding for a single image"""
        try:
            # Process image
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs)
                embedding = F.normalize(embedding, dim=-1)
            
            return embedding.cpu()  # Move to CPU for consistency
            
        except Exception as e:
            logger.error(f"❌ Failed to generate image embedding: {e}")
            # Return zero embedding as fallback
            return torch.zeros(1, 512)
    
    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Generate embedding for text"""
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.get_text_features(**inputs)
                embedding = F.normalize(embedding, dim=-1)
            
            return embedding.cpu()
            
        except Exception as e:
            logger.error(f"❌ Failed to generate text embedding: {e}")
            return torch.zeros(1, 512)
    
    def classify(self, image: Image.Image, top_k: int = 3, min_score: float = 0.1) -> List[Dict]:
        """Classify a product image"""
        logger.info(f"🔍 Starting classification (top_k={top_k})")
        
        try:
            # Generate image embedding
            image_emb = self._get_image_embedding(image)
            
            # Calculate similarity with all categories
            results = []
            for category, category_emb in self.category_embeddings.items():
                # Calculate cosine similarity
                similarity = F.cosine_similarity(image_emb, category_emb).item()
                
                # Skip if below minimum score
                if similarity < min_score:
                    continue
                
                # Determine confidence level
                confidence = self._get_confidence_level(similarity)
                
                results.append({
                    'category': category,
                    'similarity': round(similarity, 4),
                    'confidence': confidence,
                    'description': f"Likely a {category} product",
                    'score_percent': round(similarity * 100, 1)
                })
            
            # Sort by similarity (highest first)
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Log results
            if results:
                logger.info(f"✅ Classification successful. Top result: {results[0]['category']} ({results[0]['score_percent']}%)")
                for i, result in enumerate(results[:top_k]):
                    logger.info(f"  {i+1}. {result['category']}: {result['similarity']:.3f} ({result['confidence']})")
            else:
                logger.warning("⚠️ No categories matched with sufficient similarity")
                # Return default "unknown" result
                results.append({
                    'category': 'unknown',
                    'similarity': 0.0,
                    'confidence': 'VERY_LOW',
                    'description': 'Cannot determine product category',
                    'score_percent': 0.0
                })
            
            # Return top K results
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Classification failed: {e}")
            raise
    
    def verify_domain(self, image: Image.Image, domain_text: str, company_name: str = None) -> Dict:
        """Verify if product belongs to a specific domain"""
        logger.info(f"🔍 Starting domain verification")
        
        try:
            # Generate image embedding
            image_emb = self._get_image_embedding(image)
            
            # Create multiple prompts for better matching
            prompts = self._create_domain_prompts(domain_text, company_name)
            logger.debug(f"Generated {len(prompts)} prompts for domain verification")
            
            # Calculate similarity with all prompts
            similarities = []
            for i, prompt in enumerate(prompts):
                text_emb = self._get_text_embedding(prompt)
                similarity = F.cosine_similarity(image_emb, text_emb).item()
                similarities.append(similarity)
                logger.debug(f"  Prompt {i+1}: {similarity:.3f} - '{prompt[:50]}...'")
            
            # Get statistics
            if similarities:
                best_similarity = max(similarities)
                avg_similarity = sum(similarities) / len(similarities)
                
                # Make decision based on thresholds
                decision, confidence = self._make_verification_decision(best_similarity)
                
                logger.info(f"✅ Domain verification: {decision} (best: {best_similarity:.3f}, avg: {avg_similarity:.3f})")
                
                return {
                    'best_similarity': round(best_similarity, 4),
                    'average_similarity': round(avg_similarity, 4),
                    'decision': decision,
                    'confidence': confidence,
                    'explanation': DECISIONS.get(decision, 'Unknown'),
                    'prompts_used': len(prompts),
                    'score_percent': round(best_similarity * 100, 1)
                }
            else:
                logger.warning("⚠️ No prompts generated for domain verification")
                return {
                    'best_similarity': 0.0,
                    'average_similarity': 0.0,
                    'decision': 'OUT_OF_DOMAIN',
                    'confidence': 'VERY_LOW',
                    'explanation': 'No valid prompts generated',
                    'prompts_used': 0,
                    'score_percent': 0.0
                }
            
        except Exception as e:
            logger.error(f"❌ Domain verification failed: {e}")
            raise
    
    def _create_domain_prompts(self, domain_text: str, company_name: str = None) -> List[str]:
        """Create multiple prompts for domain verification"""
        prompts = []
        
        # Clean the domain text
        domain_text = domain_text.strip()
        if not domain_text:
            logger.warning("Domain text is empty")
            return prompts
        
        # Basic domain text variations
        prompts.append(domain_text)
        prompts.append(f"products: {domain_text}")
        prompts.append(f"a photo of {domain_text}")
        prompts.append(f"product image: {domain_text}")
        
        # If company name is provided, add company-specific prompts
        if company_name and company_name.strip():
            company_name = company_name.strip()
            prompts.append(f"{company_name} products: {domain_text}")
            prompts.append(f"products sold by {company_name}: {domain_text}")
            prompts.append(f"{company_name}'s {domain_text}")
        
        # Extract main terms from domain text
        domain_terms = [term.strip() for term in domain_text.split(',') if term.strip()]
        if domain_terms:
            main_term = domain_terms[0]
            
            # Add more specific prompts
            prompts.extend([
                f"a {main_term} product",
                f"commercial {main_term}",
                f"{main_term} for sale",
                f"photo showing {main_term}"
            ])
        
        # Add combination prompts if multiple terms
        if len(domain_terms) > 1:
            prompts.append(f"{domain_terms[0]} and {domain_terms[1]}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_prompts = []
        for prompt in prompts:
            if prompt not in seen:
                seen.add(prompt)
                unique_prompts.append(prompt)
        
        return unique_prompts
    
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
    
    def _make_verification_decision(self, similarity: float) -> Tuple[str, str]:
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
        if subcategories:
            category_text = f"{category_name} products including {', '.join(subcategories[:3])}"
        else:
            category_text = f"{category_name} products"
            
        emb = self._get_text_embedding(category_text)
        self.category_embeddings[category_name] = emb
        logger.info(f"✅ Added new category: {category_name}")


# Global instance
classifier = ProductClassifier()
