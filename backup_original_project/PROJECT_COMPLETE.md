# Product Classification Project - Complete Documentation

## Directory Structure

```
product-classification/
├── app.py
├── README.md
├── requirements.txt
├── PROJECT_DOCUMENTATION.txt
├── PROJECT_COMPLETE.md (this file)
├── .gitignore
│
├── config/
│   ├── company_domains.py
│   ├── all_categories.json
│   └── __pycache__/
│
├── data/
│   ├── jeans/
│   ├── sofa/
│   ├── tshirt/
│   ├── tv/
│   ├── external/
│   │   ├── industry_data/
│   │   └── regulatory_info/
│   ├── training/
│   │   ├── company_profiles.json
│   │   ├── labeled_pairs.csv
│   │   └── product_listings.json
│   └── validation/
│       └── test_cases.json
│
├── demo/
│   ├── run_demo.py
│   ├── evaluate_dataset.py
│   └── __pycache__/
│
├── models/
│   ├── clip_model.py
│   └── __pycache__/
│
├── output/
│   ├── evaluation_results.json
│   └── results.json
│
├── pipelines/
│   ├── universal_classifier.py
│   ├── verification_pipeline.py
│   └── __pycache__/
│
├── static/
│   ├── style.css
│   └── style_enhanced.css
│
├── templates/
│   ├── index.html
│   └── index_enhanced.html
│
└── utils/
    ├── embedding_utils.py
    ├── image_utils.py
    ├── scoring_utils.py
    ├── domain_generator.py
    └── __pycache__/
```

---

## File Contents

### 1. app.py
```python
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import json

from models.clip_model import load_clip
from pipelines.universal_classifier import UniversalProductClassifier

app = Flask(__name__)

# Load enhanced classifier
classifier = UniversalProductClassifier()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    classification = None
    error = None

    if request.method == "POST":
        try:
            image_file = request.files["image"]
            
            if not image_file:
                raise ValueError("Image is required.")
            
            image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
            
            # Get domain text (if provided)
            domain_text = request.form.get("domain", "").strip()
            
            if domain_text:
                # Domain verification mode
                result = classifier.verify_domain(image, domain_text)
            else:
                # Universal classification mode
                classification = classifier.classify_product(image, top_k=5)
                
        except Exception as e:
            error = str(e)
    
    return render_template("index_enhanced.html", 
                         result=result, 
                         classification=classification, 
                         error=error)

@app.route("/categories", methods=["GET"])
def get_categories():
    """API endpoint to get all available categories"""
    categories = list(classifier.categories.keys())
    return jsonify({"categories": categories})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

### 2. requirements.txt
```
torch
torchvision
transformers
Pillow
numpy
scikit-learn
Flask
```

### 3. README.md
```markdown
# Product Domain Verification (Image-Based)

This project verifies whether a product image belongs to a company's domain
using a pretrained CLIP vision-language model.

No training, no manual category rules, fully semantic.
```

### 4. .gitignore
```
data/
```

### 5. config/company_domains.py
```python
COMPANY_DOMAINS = {
    "fashion_company": "clothing, apparel, fashion garments, t shirts, pants, jeans, shorts, tops, dresses, skirts, jackets, coats, sweaters, hoodies, activewear, sportswear, casual wear, formal wear",
    
    "furniture_company": "sofa, couch, sectional, loveseat, chaise lounge, furniture, home seating products, living room furniture, upholstery, recliner, futon, settee, bench, ottoman, home decor, interior design",
    
    "tv_company": "television, smart tv, led tv, lcd tv, oled tv, 4k tv, 8k tv, home entertainment electronics, display screen, monitor, home theater, streaming device, media player",
    
    # Add more domains
    "electronics_company": "smartphone, tablet, laptop, computer, electronics, gadgets, devices, mobile phone, iphone, android phone, smartwatch, headphones, earphones, charger, cable",
    
    "kitchen_appliances": "refrigerator, oven, microwave, blender, mixer, toaster, coffee maker, dishwasher, stove, cooktop, kitchen appliance, home appliance, cooking device",
    
    "sports_equipment": "basketball, football, soccer ball, tennis racket, golf club, baseball bat, sports gear, athletic equipment, exercise machine, gym equipment, yoga mat, running shoes",
}
```

### 6. config/all_categories.json
```json
{
  "electronics": [
    "television",
    "smartphone",
    "laptop",
    "tablet",
    "camera",
    "headphones",
    "speakers",
    "smartwatch",
    "gaming console"
  ],
  "furniture": [
    "sofa",
    "bed",
    "table",
    "chair",
    "desk",
    "wardrobe",
    "shelf",
    "cabinet",
    "ottoman"
  ],
  "clothing": [
    "t-shirt",
    "jeans",
    "dress",
    "jacket",
    "shirt",
    "pants",
    "shorts",
    "skirt",
    "sweater",
    "hoodie",
    "coat",
    "suit"
  ],
  "kitchen_appliances": [
    "refrigerator",
    "oven",
    "microwave",
    "blender",
    "toaster",
    "coffee maker",
    "dishwasher",
    "mixer"
  ],
  "home_decor": [
    "lamp",
    "rug",
    "curtain",
    "painting",
    "vase",
    "mirror",
    "clock",
    "candle",
    "pillow"
  ],
  "sports": [
    "basketball",
    "football",
    "tennis racket",
    "golf clubs",
    "bicycle",
    "treadmill",
    "yoga mat",
    "dumbbells"
  ],
  "books": [
    "novel",
    "textbook",
    "magazine",
    "comic",
    "ebook",
    "audiobook"
  ],
  "toys": [
    "doll",
    "action figure",
    "lego",
    "puzzle",
    "board game",
    "car toy"
  ],
  "beauty": [
    "perfume",
    "makeup",
    "skincare",
    "shampoo",
    "cosmetics"
  ],
  "automotive": [
    "car",
    "tire",
    "battery",
    "tool",
    "accessory"
  ]
}
```

### 7. models/clip_model.py
```python
from transformers import CLIPProcessor, CLIPModel

def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor
```

### 8. utils/embedding_utils.py
```python
import torch

def get_image_embedding(model, processor, image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb / emb.norm(dim=-1, keepdim=True)

def get_text_embedding(model, processor, text):
    inputs = processor(text=text, return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    return emb / emb.norm(dim=-1, keepdim=True)
```

### 9. utils/image_utils.py
```python
from PIL import Image
import os

def load_images_by_category(base_dir):
    data = {}
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if not os.path.isdir(category_path):
            continue

        images = []
        for file in os.listdir(category_path):
            if file.lower().endswith(".jpg"):
                img_path = os.path.join(category_path, file)
                images.append(Image.open(img_path).convert("RGB"))

        data[category] = images

    return data
```

### 10. utils/scoring_utils.py
```python
import torch.nn.functional as F

IN_DOMAIN_THRESHOLD = 0.25
NEAR_DOMAIN_THRESHOLD = 0.20

def cosine_similarity(img_emb, text_emb):
    return F.cosine_similarity(img_emb, text_emb).item()

def classify_score(score):
    if score >= IN_DOMAIN_THRESHOLD:
        return "IN_DOMAIN"
    elif score >= NEAR_DOMAIN_THRESHOLD:
        return "NEAR_DOMAIN"
    else:
        return "OUT_OF_DOMAIN"
```

### 11. pipelines/verification_pipeline.py
```python
from utils.embedding_utils import get_image_embedding, get_text_embedding
from utils.scoring_utils import cosine_similarity, classify_score
import numpy as np

def verify_image_enhanced(
    model,
    processor,
    image,
    company_domain_text,
    use_multiple_prompts=True
):
    image_emb = get_image_embedding(model, processor, image)
    
    if use_multiple_prompts:
        # Generate multiple prompts for the same domain
        prompts = generate_multiple_prompts(company_domain_text)
        
        scores = []
        for prompt in prompts:
            domain_emb = get_text_embedding(model, processor, prompt)
            score = cosine_similarity(image_emb, domain_emb)
            scores.append(score)
        
        # Use maximum score (most relevant prompt)
        score = max(scores)
    else:
        # Original single prompt
        domain_emb = get_text_embedding(model, processor, company_domain_text)
        score = cosine_similarity(image_emb, domain_emb)
    
    decision = classify_score(score)
    
    return {
        "similarity_score": round(score, 4),
        "decision": decision,
        "confidence": get_confidence_level(score)
    }

def generate_multiple_prompts(domain_text: str):
    """Generate multiple text prompts for better matching"""
    base_terms = domain_text.split(", ")
    
    prompts = [
        # Original
        domain_text,
        # As a photo description
        f"a photo of {base_terms[0]}",
        # As product description
        f"product: {domain_text}",
        # As commercial item
        f"commercial product: {', '.join(base_terms[:3])}",
        # Negative prompts (for contrast)
        f"this is {base_terms[0]}",
        # Simple version
        ", ".join(base_terms[:5])
    ]
    
    return prompts

def get_confidence_level(score: float) -> str:
    """Add confidence levels to results"""
    if score >= 0.35:
        return "HIGH"
    elif score >= 0.25:
        return "MEDIUM"
    elif score >= 0.15:
        return "LOW"
    else:
        return "VERY_LOW"
```

### 12. pipelines/universal_classifier.py
```python
import json
import os
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np

from models.clip_model import load_clip
from utils.embedding_utils import get_image_embedding, get_text_embedding
from utils.scoring_utils import cosine_similarity

class UniversalProductClassifier:
    def __init__(self, categories_file: str = "config/all_categories.json"):
        self.model, self.processor = load_clip()
        self.categories = self.load_categories(categories_file)
        self.category_embeddings = self.precompute_category_embeddings()
    
    def load_categories(self, file_path: str) -> Dict:
        """Load all product categories"""
        # Create a comprehensive category list
        all_categories = {
            "electronics": ["television", "smartphone", "laptop", "tablet", "camera", 
                           "headphones", "speakers", "smartwatch", "gaming console"],
            "furniture": ["sofa", "bed", "table", "chair", "desk", "wardrobe", 
                         "shelf", "cabinet", "ottoman"],
            "clothing": ["t-shirt", "jeans", "dress", "jacket", "shirt", "pants",
                        "shorts", "skirt", "sweater", "hoodie", "coat", "suit"],
            "kitchen_appliances": ["refrigerator", "oven", "microwave", "blender",
                                  "toaster", "coffee maker", "dishwasher", "mixer"],
            "home_decor": ["lamp", "rug", "curtain", "painting", "vase", "mirror",
                          "clock", "candle", "pillow"],
            "sports": ["basketball", "football", "tennis racket", "golf clubs",
                      "bicycle", "treadmill", "yoga mat", "dumbbells"],
            "books": ["novel", "textbook", "magazine", "comic", "ebook", "audiobook"],
            "toys": ["doll", "action figure", "lego", "puzzle", "board game", "car toy"],
            "beauty": ["perfume", "makeup", "skincare", "shampoo", "cosmetics"],
            "automotive": ["car", "tire", "battery", "tool", "accessory"]
        }
        
        # Save to file for future use
        os.makedirs("config", exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(all_categories, f, indent=2)
        
        return all_categories
    
    def precompute_category_embeddings(self) -> Dict:
        """Precompute embeddings for all categories for faster inference"""
        category_embeddings = {}
        
        for main_category, subcategories in self.categories.items():
            # Create rich text for each category
            category_text = f"{main_category} products: {', '.join(subcategories[:10])}"
            
            # Generate embedding
            emb = get_text_embedding(self.model, self.processor, category_text)
            category_embeddings[main_category] = emb
            
            # Also embed each subcategory individually
            for subcat in subcategories[:5]:  # Limit to avoid too many embeddings
                subcat_text = f"a {subcat}, {main_category} product"
                sub_emb = get_text_embedding(self.model, self.processor, subcat_text)
                category_embeddings[f"{main_category}_{subcat}"] = sub_emb
        
        return category_embeddings
    
    def classify_product(self, image: Image.Image, top_k: int = 3) -> List[Dict]:
        """Classify a product image into multiple categories"""
        # Get image embedding
        img_emb = get_image_embedding(self.model, self.processor, image)
        
        # Calculate similarity with all categories
        results = []
        
        for category_name, cat_emb in self.category_embeddings.items():
            score = cosine_similarity(img_emb, cat_emb)
            
            # Parse category hierarchy
            if "_" in category_name:
                main_cat, sub_cat = category_name.split("_", 1)
            else:
                main_cat = category_name
                sub_cat = None
            
            results.append({
                "category": main_cat,
                "subcategory": sub_cat,
                "score": round(score, 4),
                "full_category": category_name
            })
        
        # Sort by score and return top K
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def verify_domain(self, image: Image.Image, domain_text: str) -> Dict:
        """Verify if product belongs to a specific domain"""
        from pipelines.verification_pipeline import verify_image_enhanced
        
        return verify_image_enhanced(
            self.model,
            self.processor,
            image,
            domain_text,
            use_multiple_prompts=True
        )
```

### 13. demo/run_demo.py
```python
import json
import os

from models.clip_model import load_clip
from pipelines.verification_pipeline import verify_image
from utils.image_utils import load_images_by_category
from config.company_domains import COMPANY_DOMAINS

OUTPUT_PATH = "output/results.json"

def run_demo():
    model, processor = load_clip()

    company = "tv_company"
    company_domain = COMPANY_DOMAINS[company]

    test_images = {
        "tv": "data/samples/tv/1.jpg",
        "tshirt": "data/samples/tshirt/1.jpg",
        "sofa": "data/samples/sofa/1.jpg"
    }

    results = {}

    for label, path in test_images.items():
        image = load_images_by_category(path)
        result = verify_image(
            model,
            processor,
            image,
            company_domain
        )
        results[label] = result

    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_demo()
```

### 14. demo/evaluate_dataset.py
```python
import json
from collections import defaultdict

from models.clip_model import load_clip
from pipelines.verification_pipeline import verify_image
from utils.image_utils import load_images_by_category
from config.company_domains import COMPANY_DOMAINS

DATA_DIR = "data"
OUTPUT_FILE = "output/evaluation_results.json"

def main():
    model, processor = load_clip()
    company_domain = COMPANY_DOMAINS["furniture_company"]

    dataset = load_images_by_category(DATA_DIR)

    results = defaultdict(list)

    for category, images in dataset.items():
        print(f"Processing category: {category} ({len(images)} images)")
        for img in images:
            result = verify_image(
                model=model,
                processor=processor,
                image=img,
                company_domain_text=company_domain
            )
            results[category].append(result)

    summary = {}

    for category, entries in results.items():
        in_count = sum(1 for e in entries if e["decision"] == "IN_DOMAIN")
        near_count = sum(1 for e in entries if e["decision"] == "NEAR_DOMAIN")
        out_count = sum(1 for e in entries if e["decision"] == "OUT_OF_DOMAIN")

        summary[category] = {
            "total_images": len(entries),
            "IN_DOMAIN": in_count,
            "NEAR_DOMAIN": near_count,
            "OUT_OF_DOMAIN": out_count
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nEvaluation Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

### 15. static/style.css
```css
body {
    font-family: Arial, sans-serif;
    background: #f5f5f5;
}

.container {
    max-width: 600px;
    margin: 40px auto;
    padding: 20px;
    background: white;
    border-radius: 6px;
}

h1 {
    text-align: center;
}

label {
    display: block;
    margin-top: 15px;
    font-weight: bold;
}

textarea {
    width: 100%;
    height: 80px;
    margin-top: 5px;
}

input[type="file"] {
    margin-top: 5px;
}

button {
    margin-top: 20px;
    padding: 10px;
    width: 100%;
    font-size: 16px;
    cursor: pointer;
}

.result {
    margin-top: 30px;
    padding: 15px;
    background: #e6f7ff;
    border-radius: 4px;
}

.error {
    margin-top: 20px;
    color: red;
}
```

### 16. static/style_enhanced.css
```css
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    margin: 0;
    padding: 20px;
}

.container {
    max-width: 800px;
    margin: 40px auto;
    padding: 30px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}

h1 {
    text-align: center;
    color: #333;
    margin-bottom: 30px;
}

.mode-selector {
    display: flex;
    gap: 10px;
    margin-bottom: 30px;
}

.mode-selector button {
    flex: 1;
    padding: 12px;
    background: #f0f0f0;
    border: 2px solid #ddd;
    border-radius: 6px;
    cursor: pointer;
    font-size: 16px;
    transition: all 0.3s;
}

.mode-selector button:hover {
    background: #e0e0e0;
}

.mode-selector button.active {
    background: #667eea;
    color: white;
    border-color: #667eea;
}

label {
    display: block;
    margin-top: 20px;
    font-weight: bold;
    color: #555;
}

textarea {
    width: 100%;
    height: 80px;
    margin-top: 5px;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 14px;
    resize: vertical;
}

input[type="file"] {
    margin-top: 5px;
    padding: 10px;
    border: 2px dashed #ddd;
    border-radius: 4px;
    width: 100%;
    cursor: pointer;
}

button[type="submit"] {
    margin-top: 30px;
    padding: 15px;
    width: 100%;
    font-size: 18px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    transition: transform 0.3s;
}

button[type="submit"]:hover {
    transform: translateY(-2px);
}

.result {
    margin-top: 40px;
    padding: 25px;
    border-radius: 8px;
    animation: fadeIn 0.5s;
}

.verification {
    background: linear-gradient(135deg, #e6f7ff 0%, #b3e0ff 100%);
    border-left: 5px solid #1890ff;
}

.classification {
    background: linear-gradient(135deg, #f0fff4 0%, #d9f7be 100%);
    border-left: 5px solid #52c41a;
}

.score-display {
    text-align: center;
    padding: 20px;
}

.score-circle {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    background: #1890ff;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    font-weight: bold;
    margin: 0 auto 20px;
    animation: pulse 2s infinite;
}

.decision {
    font-size: 28px;
    font-weight: bold;
    margin: 10px 0;
}

.decision.in_domain {
    color: #52c41a;
}

.decision.near_domain {
    color: #faad14;
}

.decision.out_of_domain {
    color: #ff4d4f;
}

.confidence {
    font-size: 16px;
    color: #666;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
}

th {
    background: #fafafa;
    padding: 12px;
    text-align: left;
    border-bottom: 2px solid #ddd;
}

td {
    padding: 12px;
    border-bottom: 1px solid #eee;
}

.score-cell {
    position: relative;
    width: 200px;
}

.score-bar {
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    background: rgba(82, 196, 26, 0.2);
    z-index: 1;
}

.error {
    margin-top: 20px;
    padding: 15px;
    background: #fff2f0;
    border: 1px solid #ffccc7;
    border-radius: 4px;
    color: #ff4d4f;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}
```

### 17. templates/index.html
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Product Domain Verification</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>

<div class="container">
    <h1>Product Domain Verification</h1>

    <form method="POST" enctype="multipart/form-data">
        <label>Company Domain (text)</label>
        <textarea name="domain" placeholder="e.g. television, smart tv, home entertainment electronics" required></textarea>

        <label>Upload Product Image</label>
        <input type="file" name="image" accept="image/*" required>

        <button type="submit">Verify Product</button>
    </form>

    {% if result %}
        <div class="result">
            <h2>Result</h2>
            <p><strong>Similarity Score:</strong> {{ result.similarity_score }}</p>
            <p><strong>Decision:</strong> {{ result.decision }}</p>
        </div>
    {% endif %}

    {% if error %}
        <div class="error">
            <p>{{ error }}</p>
        </div>
    {% endif %}
</div>

</body>
</html>
```

### 18. templates/index_enhanced.html
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Universal Product Classifier</title>
    <link rel="stylesheet" href="/static/style_enhanced.css">
</head>
<body>

<div class="container">
    <h1>Universal Product Classifier</h1>
    
    <div class="mode-selector">
        <button onclick="setMode('classify')">Classify Product</button>
        <button onclick="setMode('verify')">Verify Domain</button>
    </div>
    
    <form method="POST" enctype="multipart/form-data" id="mainForm">
        
        <div id="verifySection" style="display:none;">
            <label>Company Domain (Optional)</label>
            <textarea name="domain" placeholder="e.g., furniture, sofa, home decor. Leave empty for auto-classification"></textarea>
            <small>If provided, verifies if product matches this domain. If empty, auto-classifies into categories.</small>
        </div>
        
        <label>Upload Product Image</label>
        <input type="file" name="image" accept="image/*" required>
        
        <button type="submit">Analyze Product</button>
    </form>
    
    {% if classification %}
        <div class="result classification">
            <h2>Classification Results</h2>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Subcategory</th>
                    <th>Match Score</th>
                </tr>
                {% for item in classification %}
                <tr>
                    <td>{{ item.category }}</td>
                    <td>{{ item.subcategory or 'N/A' }}</td>
                    <td class="score-cell">
                        <div class="score-bar" style="width: {{ item.score * 100 }}%"></div>
                        {{ item.score }}
                    </td>
                </tr>
                {% endfor %}
            </table>
        </div>
    {% endif %}
    
    {% if result %}
        <div class="result verification">
            <h2>Domain Verification Result</h2>
            <div class="score-display">
                <div class="score-circle" data-score="{{ result.similarity_score }}">
                    {{ result.similarity_score }}
                </div>
                <div class="decision {{ result.decision|lower }}">
                    {{ result.decision }}
                </div>
                <div class="confidence">
                    Confidence: {{ result.confidence }}
                </div>
            </div>
        </div>
    {% endif %}
    
    {% if error %}
        <div class="error">
            <p>{{ error }}</p>
        </div>
    {% endif %}
</div>

<script>
function setMode(mode) {
    const verifySection = document.getElementById('verifySection');
    if (mode === 'verify') {
        verifySection.style.display = 'block';
    } else {
        verifySection.style.display = 'none';
    }
}
</script>

</body>
</html>
```

---

## Project Summary

**Project Name:** Product Domain Verification & Universal Product Classifier

**Purpose:** Uses a pretrained CLIP vision-language model to verify whether product images belong to specific company domains or classify them into multiple product categories.

**Key Features:**
- ✓ No training required
- ✓ No manual category rules
- ✓ Fully semantic approach using CLIP embeddings
- ✓ Dual-mode operation: domain verification and universal classification
- ✓ Enhanced UI with modern styling
- ✓ Multiple prompt generation for better matching
- ✓ Confidence levels and detailed scoring

**Technology Stack:**
- **Framework:** Flask (Python web framework)
- **ML Model:** OpenAI CLIP (Vision-Language Model)
- **Image Processing:** PIL/Pillow
- **Embeddings:** PyTorch + Transformers
- **Frontend:** HTML + CSS + JavaScript

**How It Works:**
1. Load CLIP model and precompute category embeddings
2. Accept product image and optional domain text
3. Generate image embedding from the product photo
4. Calculate cosine similarity with text embeddings
5. Classify based on thresholds (IN_DOMAIN, NEAR_DOMAIN, OUT_OF_DOMAIN)
6. Return similarity score, decision, and confidence level

**Endpoints:**
- `GET /` or `POST /` - Main interface
- `GET /categories` - List all available product categories
