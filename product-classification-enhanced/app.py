from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import io
import logging
from datetime import datetime
import os

from core.config import Config
from models.clip_manager import clip_manager
from services.classifier import classifier
from services.cache import cache
from utils.image_processor import image_processor
from utils.validation import validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize application
Config.init_app()

def startup():
    """Initialize on first request"""
    logger.info("Starting up application...")
    try:
        clip_manager.load_model()
        clip_manager.warmup()
        logger.info("Model loaded and warmed up successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

# Run startup initialization
with app.app_context():
    startup()

@app.route('/', methods=['GET'])
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    """Classify product image"""
    start_time = datetime.now()
    
    try:
        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image data
        image_data = file.read()
        
        # Validate image
        is_valid, error_msg = image_processor.validate_image(image_data)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Process image
        image = image_processor.process_image(image_data)
        
        # Get parameters
        mode = request.form.get('mode', 'classify')
        top_k = int(request.form.get('top_k', 3))
        
        # Generate cache key
        cache_key = f"{mode}_{hash(image_data)}_{top_k}"
        
        # Check cache first
        if cache.get(cache_key):
            logger.info("Cache hit for classification")
            result = cache.get(cache_key)
            result['cached'] = True
            return jsonify(result)
        
        result = {}
        
        if mode == 'classify':
            # Perform classification
            classifications = classifier.classify(image, top_k=top_k)
            result = {
                'mode': 'classification',
                'classifications': classifications,
                'timestamp': datetime.now().isoformat()
            }
        
        elif mode == 'verify':
            # Get domain text
            domain_text = request.form.get('domain_text', '').strip()
            company_name = request.form.get('company_name', '').strip()
            
            # Validate inputs
            is_valid, error_msg = validator.validate_domain_text(domain_text)
            if not is_valid:
                return jsonify({'error': error_msg}), 400
            
            is_valid, error_msg = validator.validate_company_name(company_name)
            if not is_valid:
                return jsonify({'error': error_msg}), 400
            
            # Perform verification
            verification = classifier.verify_domain(
                image, 
                domain_text, 
                company_name
            )
            
            result = {
                'mode': 'verification',
                'verification': verification,
                'timestamp': datetime.now().isoformat()
            }
        
        else:
            return jsonify({'error': 'Invalid mode. Use "classify" or "verify".'}), 400
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        result['processing_time_seconds'] = round(processing_time, 3)
        result['cached'] = False
        
        # Cache the result
        cache.set(cache_key, result)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Classification error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all available categories"""
    try:
        return jsonify({
            'categories': list(classifier.categories.keys()),
            'count': len(classifier.categories)
        })
    except Exception as e:
        logger.error(f"Failed to get categories: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check model status
        model_loaded = clip_manager._model is not None
        
        # Get cache stats
        cache_stats = cache.get_stats()
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': model_loaded,
            'cache_stats': cache_stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear cache"""
    try:
        cache.clear()
        return jsonify({'message': 'Cache cleared successfully'})
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=Config.DEBUG,
        threaded=True
    )
