#!/usr/bin/env python3
"""
Test script for Product Classifier
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageDraw
import requests
from io import BytesIO

def create_test_image(text="Test Product", size=(224, 224)):
    """Create a test image"""
    image = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), text, fill='black')
    return image

def test_classification():
    """Test classification functionality"""
    print("Testing Product Classifier...")
    print("=" * 50)
    
    # Create test images
    test_images = {
        "electronics": create_test_image("Smartphone"),
        "furniture": create_test_image("Sofa"),
        "clothing": create_test_image("T-Shirt")
    }
    
    from services.classifier import classifier
    
    for label, image in test_images.items():
        print(f"\nTesting {label} image:")
        print("-" * 30)
        
        try:
            # Test classification
            results = classifier.classify(image, top_k=3)
            
            print(f"Top 3 classifications:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['category']}: {result['similarity']:.3f} ({result['confidence']})")
        
        except Exception as e:
            print(f"Error: {e}")

def test_verification():
    """Test domain verification"""
    print("\n" + "=" * 50)
    print("Testing Domain Verification...")
    print("=" * 50)
    
    from services.classifier import classifier
    
    # Create test image
    test_image = create_test_image("Smartphone")
    
    # Test domains
    test_domains = [
        ("electronics, gadgets, devices", "Electronics Corp", "Should match"),
        ("furniture, chairs, tables", "Furniture Co", "Should not match"),
        ("clothing, apparel, fashion", "Fashion Inc", "Should not match")
    ]
    
    for domain_text, company_name, expected in test_domains:
        print(f"\nTesting: {expected}")
        print(f"Domain: {domain_text}")
        print(f"Company: {company_name}")
        print("-" * 30)
        
        try:
            result = classifier.verify_domain(test_image, domain_text, company_name)
            
            print(f"Decision: {result['decision']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Best Similarity: {result['best_similarity']:.3f}")
            print(f"Explanation: {result['explanation']}")
        
        except Exception as e:
            print(f"Error: {e}")

def test_api():
    """Test the Flask API"""
    print("\n" + "=" * 50)
    print("Testing Flask API...")
    print("=" * 50)
    
    import json
    # Use app_context to avoid startup issues if any remaining
    from app import app
    
    # Create test client
    client = app.test_client()
    
    # Test health endpoint
    print("\nTesting /api/health:")
    response = client.get('/api/health')
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json}")
    
    # Test categories endpoint
    print("\nTesting /api/categories:")
    response = client.get('/api/categories')
    print(f"Status: {response.status_code}")
    data = response.json
    print(f"Number of categories: {data.get('count', 0)}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    # Initialize environment
    from core.config import Config
    Config.init_app()
    
    # Run tests
    test_classification()
    test_verification()
    test_api()
