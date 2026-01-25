#!/usr/bin/env python3
"""
Debug script to test CLIP model and classification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from PIL import Image, ImageDraw
import numpy as np

def test_clip_model():
    """Test if CLIP model is working correctly"""
    print("=" * 60)
    print("Testing CLIP Model")
    print("=" * 60)
    
    from models.clip_manager import clip_manager
    
    try:
        # Test model loading
        print("\n1. Testing model loading...")
        model, processor = clip_manager.load_model()
        print(f"   ✅ Model loaded successfully")
        print(f"   Device: {clip_manager.device}")
        print(f"   Model type: {type(model).__name__}")
        
        # Test image embedding
        print("\n2. Testing image embedding...")
        test_image = Image.new('RGB', (224, 224), color='red')
        draw = ImageDraw.Draw(test_image)
        draw.text((10, 10), "Test", fill='white')
        
        inputs = processor(images=test_image, return_tensors="pt").to(clip_manager.device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        print(f"   ✅ Image embedding shape: {image_features.shape}")
        print(f"   Embedding sample: {image_features[0, :5].cpu().numpy()}")
        
        # Test text embedding
        print("\n3. Testing text embedding...")
        test_text = ["a red product"]
        inputs = processor(text=test_text, return_tensors="pt", padding=True).to(clip_manager.device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        print(f"   ✅ Text embedding shape: {text_features.shape}")
        print(f"   Embedding sample: {text_features[0, :5].cpu().numpy()}")
        
        # Test similarity calculation
        print("\n4. Testing similarity calculation...")
        similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
        print(f"   ✅ Similarity between red image and 'a red product': {similarity.item():.4f}")
        
        # Test with different text
        print("\n5. Testing with different categories...")
        categories = ["electronics", "furniture", "clothing", "food"]
        
        for category in categories:
            inputs = processor(text=[f"a {category} product"], return_tensors="pt", padding=True).to(clip_manager.device)
            with torch.no_grad():
                text_features = model.get_text_features(**inputs)
            similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
            print(f"   '{category}': {similarity.item():.4f}")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed! Model is working correctly.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_classification():
    """Test classification functionality"""
    print("\n" + "=" * 60)
    print("Testing Classification")
    print("=" * 60)
    
    from services.classifier import classifier
    
    try:
        # Create test images
        print("\n1. Creating test images...")
        test_images = {}
        
        # Red square (generic)
        img = Image.new('RGB', (224, 224), color='red')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Test", fill='white')
        test_images['generic'] = img
        
        # Create image that looks like electronics
        img = Image.new('RGB', (224, 224), color='black')
        draw = ImageDraw.Draw(img)
        # Draw a rectangle that looks like a screen
        draw.rectangle([50, 50, 174, 174], outline='white', width=2)
        draw.text((80, 100), "TV", fill='white')
        test_images['electronics'] = img
        
        # Create image that looks like furniture
        img = Image.new('RGB', (224, 224), color='brown')
        draw = ImageDraw.Draw(img)
        # Draw a rectangle that looks like a sofa
        draw.rectangle([30, 100, 194, 150], fill='darkred')
        draw.text((90, 120), "Sofa", fill='white')
        test_images['furniture'] = img
        
        print(f"   Created {len(test_images)} test images")
        
        # Test classification
        print("\n2. Testing classification...")
        for label, image in test_images.items():
            print(f"\n   Testing '{label}' image:")
            print("   " + "-" * 40)
            
            results = classifier.classify(image, top_k=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"   {i}. {result['category']}: {result['similarity']:.4f} ({result['confidence']})")
                
                top_result = results[0]
                if top_result['similarity'] > 0.2:
                    print(f"   ✅ Good match: {top_result['category']}")
                else:
                    print(f"   ⚠️ Weak match: {top_result['category']} ({top_result['similarity']:.4f})")
            else:
                print("   ❌ No classification results")
        
        # Test domain verification
        print("\n3. Testing domain verification...")
        test_image = test_images['electronics']
        
        test_cases = [
            ("electronics, gadgets, devices", "Electronics Corp", "Should match"),
            ("furniture, sofa, chair", "Furniture Store", "Should not match"),
            ("clothing, apparel, fashion", "Fashion Co", "Should not match")
        ]
        
        for domain_text, company, expected in test_cases:
            print(f"\n   Domain: {domain_text}")
            print(f"   Expected: {expected}")
            
            result = classifier.verify_domain(test_image, domain_text, company)
            
            result_decision = result.get('decision', 'UNKNOWN')
            print(f"   Result: {result_decision} (score: {result['best_similarity']:.4f})")
            print(f"   Confidence: {result['confidence']}")
            
            if expected == "Should match" and result_decision == "IN_DOMAIN":
                print("   ✅ Correct!")
            elif expected == "Should not match" and result_decision != "IN_DOMAIN":
                print("   ✅ Correct!")
            else:
                print("   ❌ Unexpected result")
        
        print("\n" + "=" * 60)
        print("✅ Classification tests completed!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Classification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_thresholds():
    """Check if thresholds are appropriate"""
    print("\n" + "=" * 60)
    print("Threshold Analysis")
    print("=" * 60)
    
    print("\nCurrent thresholds in .env:")
    print(f"   IN_DOMAIN_THRESHOLD: {os.getenv('IN_DOMAIN_THRESHOLD', '0.25')}")
    print(f"   NEAR_DOMAIN_THRESHOLD: {os.getenv('NEAR_DOMAIN_THRESHOLD', '0.20')}")
    
    print("\nTypical CLIP similarity scores:")
    print("   - Very similar: 0.30-0.35+")
    print("   - Similar: 0.25-0.30")
    print("   - Somewhat similar: 0.20-0.25")
    print("   - Not similar: 0.15-0.20")
    print("   - Very different: < 0.15")
    
    print("\nRecommendation:")
    print("   If getting too many 'IN_DOMAIN' results, try increasing thresholds:")
    print("   IN_DOMAIN_THRESHOLD=0.30")
    print("   NEAR_DOMAIN_THRESHOLD=0.25")
    
    return True

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Product Classifier Debug Tool")
    print("=" * 60)
    
    # Run tests
    model_ok = test_clip_model()
    
    if model_ok:
        classification_ok = test_classification()
    
    check_thresholds()
    
    print("\nDebug complete!")
