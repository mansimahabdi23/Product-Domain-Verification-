import torch
from transformers import CLIPModel, CLIPProcessor
import logging
from pathlib import Path
import time
import traceback

logger = logging.getLogger(__name__)

class CLIPManager:
    """Manages CLIP model loading and inference"""
    
    _instance = None
    _model = None
    _processor = None
    _model_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CLIPManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        if not hasattr(self, 'initialized'):
            self.model_name = model_name
            self.device = self._get_device()
            self.initialized = False
            self._model_loaded = False
    
    def _get_device(self):
        """Get the best available device (GPU/CPU)"""
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using Apple Metal Performance Shaders")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU")
            return device
        except Exception as e:
            logger.warning(f"Failed to detect device, using CPU: {e}")
            return torch.device("cpu")
    
    def load_model(self):
        """Load CLIP model and processor"""
        if self._model_loaded and self._model is not None and self._processor is not None:
            logger.info("Model already loaded, returning cached instance")
            return self._model, self._processor
        
        logger.info(f"Loading CLIP model: {self.model_name}")
        start_time = time.time()
        
        try:
            # Load model with error handling
            logger.info("Loading model...")
            self._model = CLIPModel.from_pretrained(self.model_name)
            logger.info("Loading processor...")
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            
            # Move model to appropriate device
            logger.info(f"Moving model to {self.device}...")
            self._model = self._model.to(self.device)
            self._model.eval()  # Set to evaluation mode
            
            load_time = time.time() - start_time
            self._model_loaded = True
            logger.info(f"✅ Model loaded successfully in {load_time:.2f} seconds on {self.device}")
            
            return self._model, self._processor
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._model_loaded = False
            raise
    
    def get_model(self):
        """Get model instance, loading if necessary"""
        if not self._model_loaded:
            return self.load_model()
        return self._model, self._processor
    
    def warmup(self):
        """Warm up the model for faster inference"""
        if not self._model_loaded:
            logger.warning("Model not loaded, skipping warmup")
            return
        
        logger.info("Warming up model...")
        try:
            model, processor = self.get_model()
            
            # Create dummy inputs - use values in [0, 1] range to avoid processor errors
            # Or simplified: just use a small random PIL image
            from PIL import Image
            import numpy as np
            
            # Create a localized dummy image (3 channels, 224x224)
            # Use random uint8 array to simulate real image content
            dummy_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            dummy_image = Image.fromarray(dummy_array)
            
            dummy_text = ["a product photo"]
            
            # Run inference
            with torch.no_grad():
                # Process image inputs through processor (handles resizing, normalization)
                image_inputs = processor(images=dummy_image, return_tensors="pt").to(self.device)
                _ = model.get_image_features(**image_inputs)
                
                # Test text features
                text_inputs = processor(text=dummy_text, return_tensors="pt", padding=True).to(self.device)
                _ = model.get_text_features(**text_inputs)
            
            logger.info("✅ Model warmup complete")
            
        except Exception as e:
            logger.error(f"❌ Model warmup failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")


# Global instance
clip_manager = CLIPManager()
