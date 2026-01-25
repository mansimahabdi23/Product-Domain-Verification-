from PIL import Image, ImageOps
import io
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image processing and validation"""
    
    def __init__(self, max_size_mb: int = 10, allowed_extensions: set = None):
        self.max_size_mb = max_size_mb
        self.allowed_extensions = allowed_extensions or {'png', 'jpg', 'jpeg', 'gif', 'webp'}
        self.max_pixels = 2000 * 2000  # Maximum resolution
    
    def validate_image(self, image_data: bytes) -> Tuple[bool, Optional[str]]:
        """Validate image data"""
        try:
            # Check size
            size_mb = len(image_data) / (1024 * 1024)
            if size_mb > self.max_size_mb:
                return False, f"Image too large ({size_mb:.1f}MB > {self.max_size_mb}MB)"
            
            # Try to open image
            image = Image.open(io.BytesIO(image_data))
            image.verify()  # Verify it's a valid image
            
            # Check dimensions
            width, height = image.size
            if width * height > self.max_pixels:
                return False, f"Image resolution too high ({width}x{height})"
            
            return True, None
            
        except Exception as e:
            return False, f"Invalid image: {str(e)}"
    
    def process_image(self, image_data: bytes, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """Process image for model input"""
        try:
            # Open and convert to RGB
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Maintain aspect ratio while resizing
            image = ImageOps.fit(image, target_size, method=Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            raise
    
    def resize_image(self, image: Image.Image, max_dimension: int = 800) -> Image.Image:
        """Resize image for display"""
        width, height = image.size
        
        if max(width, height) <= max_dimension:
            return image
        
        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def save_image_for_display(self, image: Image.Image, output_path: str, quality: int = 85):
        """Save image for web display"""
        try:
            # Resize if too large
            display_image = self.resize_image(image, max_dimension=800)
            
            # Save with compression
            display_image.save(output_path, 'JPEG', quality=quality, optimize=True)
            logger.info(f"Saved display image to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save display image: {e}")


# Global instance
image_processor = ImageProcessor(max_size_mb=10)
