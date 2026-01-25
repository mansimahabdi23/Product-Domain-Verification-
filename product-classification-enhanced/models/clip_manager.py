import torch
from transformers import CLIPModel, CLIPProcessor
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class CLIPManager:
    """Manages CLIP model loading and inference"""
    
    _instance = None
    _model = None
    _processor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CLIPManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        if not hasattr(self, 'initialized'):
            self.model_name = model_name
            self.device = self._get_device()
            self.initialized = False
    
    def _get_device(self):
        """Get the best available device (GPU/CPU)"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():  # For Apple Silicon
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def load_model(self):
        """Load CLIP model and processor"""
        if self._model is not None and self._processor is not None:
            logger.info("Model already loaded, returning cached instance")
            return self._model, self._processor
        
        logger.info(f"Loading CLIP model: {self.model_name}")
        start_time = time.time()
        
        try:
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            
            # Move model to appropriate device
            self._model = self._model.to(self.device)
            self._model.eval()  # Set to evaluation mode
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds on {self.device}")
            
            self.initialized = True
            return self._model, self._processor
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def get_model(self):
        """Get model instance, loading if necessary"""
        if self._model is None:
            return self.load_model()
        return self._model, self._processor
    
    def warmup(self):
        """Warm up the model for faster inference"""
        logger.info("Warming up model...")
        
        model, processor = self.get_model()
        
        # Create dummy inputs
        dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
        dummy_text = ["a product photo"]
        
        # Run inference
        with torch.no_grad():
            _ = model.get_image_features(dummy_image)
            _ = model.get_text_features(
                processor(text=dummy_text, return_tensors="pt", padding=True).to(self.device)
            )
        
        logger.info("Model warmup complete")


# Global instance
clip_manager = CLIPManager()
