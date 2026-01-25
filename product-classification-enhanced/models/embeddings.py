import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Handles embedding generation for images and text"""
    
    def __init__(self, clip_manager):
        self.clip_manager = clip_manager
        self.model, self.processor = clip_manager.get_model()
        self.device = clip_manager.device
    
    def get_image_embedding(self, image: Image.Image) -> torch.Tensor:
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
            logger.error(f"Failed to generate image embedding: {e}")
            raise
    
    def get_text_embedding(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Generate embedding for text"""
        try:
            # Ensure text is a list
            if isinstance(text, str):
                text = [text]
            
            # Process text
            inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model.get_text_features(**inputs)
                embedding = F.normalize(embedding, dim=-1)
            
            return embedding.cpu()  # Move to CPU for consistency
            
        except Exception as e:
            logger.error(f"Failed to generate text embedding: {e}")
            raise
    
    def get_similarity(self, image_embedding: torch.Tensor, text_embedding: torch.Tensor) -> float:
        """Calculate cosine similarity between image and text embeddings"""
        try:
            similarity = F.cosine_similarity(image_embedding, text_embedding)
            return similarity.item()
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def batch_process_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Process multiple images in batch"""
        try:
            # Process all images
            inputs = self.processor(
                images=images, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model.get_image_features(**inputs)
                embeddings = F.normalize(embeddings, dim=-1)
            
            return embeddings.cpu()
            
        except Exception as e:
            logger.error(f"Failed to batch process images: {e}")
            raise
