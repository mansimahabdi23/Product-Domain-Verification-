import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Model settings
    MODEL_NAME = os.getenv('MODEL_NAME', 'openai/clip-vit-base-patch32')
    MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', './cache')
    
    # Application settings
    MAX_IMAGE_SIZE_MB = int(os.getenv('MAX_IMAGE_SIZE_MB', 10))
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # Thresholds for classification
    IN_DOMAIN_THRESHOLD = float(os.getenv('IN_DOMAIN_THRESHOLD', 0.25))
    NEAR_DOMAIN_THRESHOLD = float(os.getenv('NEAR_DOMAIN_THRESHOLD', 0.20))
    
    # File paths
    CATEGORIES_FILE = os.getenv('CATEGORIES_FILE', 'data/categories.json')
    DOMAINS_FILE = os.getenv('DOMAINS_FILE', 'data/domains.json')
    
    # Cache settings
    ENABLE_CACHE = os.getenv('ENABLE_CACHE', 'True').lower() == 'true'
    CACHE_TTL = int(os.getenv('CACHE_TTL', 3600))  # 1 hour
    
    @staticmethod
    def init_app():
        """Initialize application"""
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('cache', exist_ok=True)
        os.makedirs('data', exist_ok=True)
