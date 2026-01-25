# Product Classification System

An AI-powered product classification and domain verification system using CLIP vision-language model.

## Features

- 🎯 **Product Classification**: Automatically classify product images into categories
- 🔍 **Domain Verification**: Check if products belong to specific company domains
- ⚡ **Fast Processing**: Caching and optimized model loading
- 🎨 **Modern UI**: Clean, responsive web interface
- 🔧 **Easy Configuration**: Environment-based configuration
- 📊 **Results Visualization**: Clear, visual representation of results

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd product-classification-enhanced

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs cache data static/css static/js templates

# Copy environment file
cp .env.example .env
# Edit .env file with your settings
```

### 2. Run the Application

```bash
# Start the Flask server
python app.py

# Open browser and navigate to:
# http://localhost:5000
```

### 3. Usage

#### Web Interface
1. Open `http://localhost:5000` in your browser
2. Choose between "Classify Product" or "Verify Domain" mode
3. Upload a product image
4. View the analysis results

#### API Endpoints

- `POST /classify` - Classify or verify product image
- `GET /api/categories` - Get all available categories
- `GET /api/health` - Health check endpoint
- `POST /api/cache/clear` - Clear cache

### 4. Testing

```bash
# Run the test script
python test_classifier.py

# Or run individual tests
python -c "from test_classifier import test_classification; test_classification()"
```

## Project Structure

```
product-classification-enhanced/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── .env                     # Environment variables
├── README.md               # This file
│
├── core/                   # Core configuration
│   ├── config.py          # Application settings
│   └── constants.py       # Constants and thresholds
│
├── models/                # Model management
│   ├── clip_manager.py   # CLIP model handling
│   └── embeddings.py     # Embedding generation
│
├── services/             # Business logic
│   ├── classifier.py     # Classification service
│   ├── verifier.py       # Verification service
│   └── cache.py         # Caching service
│
├── utils/               # Utility functions
│   ├── image_processor.py
│   └── validation.py
│
├── static/             # Frontend assets
│   ├── css/style.css
│   └── js/main.js
│
├── templates/          # HTML templates
│   ├── base.html
│   └── index.html
│
├── data/              # Data files
│   ├── categories.json
│   └── domains.json
│
└── logs/             # Application logs
```

## Configuration

Edit the `.env` file to customize:

```env
# Model Settings
MODEL_NAME=openai/clip-vit-base-patch32

# Thresholds
IN_DOMAIN_THRESHOLD=0.25
NEAR_DOMAIN_THRESHOLD=0.20

# File Settings
MAX_IMAGE_SIZE_MB=10
```

## API Usage Examples

### Classify Product
```bash
curl -X POST http://localhost:5000/classify \
  -F "image=@product.jpg" \
  -F "mode=classify" \
  -F "top_k=3"
```

### Verify Domain
```bash
curl -X POST http://localhost:5000/classify \
  -F "image=@product.jpg" \
  -F "mode=verify" \
  -F "domain_text=electronics,gadgets,devices" \
  -F "company_name=TechCorp"
```

## Adding Custom Categories

Edit `data/categories.json` to add or modify product categories:

```json
{
  "your_category": [
    "subcategory1",
    "subcategory2",
    "subcategory3"
  ]
}
```

## Performance Tips

1. **Enable Caching**: Set `ENABLE_CACHE=True` in `.env`
2. **Use GPU**: Install CUDA-enabled PyTorch for faster inference
3. **Resize Images**: Upload images at reasonable sizes (under 10MB)
4. **Batch Processing**: For multiple images, consider implementing batch API

## Troubleshooting

### Model Loading Issues
```bash
# Clear model cache
rm -rf cache/*

# Check internet connection for model download
```

### Memory Issues
- Reduce `MAX_IMAGE_SIZE_MB` in `.env`
- Close other applications using GPU memory
- Consider using a smaller CLIP model variant

### API Errors
- Check logs in `logs/app.log`
- Ensure all dependencies are installed
- Verify port 5000 is available

## License

MIT License

## Support

For issues and questions, please check:
1. Application logs: `logs/app.log`
2. Model cache: `cache/` directory
3. Open an issue on GitHub repository
