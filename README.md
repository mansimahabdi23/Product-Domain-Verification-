# PS-03 -- Product Domain Verification and Classification Pipeline
# 📌 Overview

This project implements an end-to-end Machine Learning pipeline to classify products into their respective business / industry domains (e.g., FinTech, AgriTech, Cybersecurity) using product names and descriptions.

# 🎯 Problem Statement
E-commerce and enterprise product catalogs often contain noisy or ambiguous product information. Accurately identifying the correct domain of a product is critical for:
-Search and recommendation systems

-Product analytics

-Catalog organization

-Automated decision systems

This project solves the problem by training a text-based domain classifier using curated and generated product data.

# 🧠 Solution Approach
The pipeline follows a structured ML workflow:
1. Data Generation & Collection
Synthetic and real-world product data are combined to improve domain coverage.

2. Data Cleaning & Preparation
Missing values are handled and relevant textual features are extracted.

3. Text Vectorization
Product descriptions are converted into numerical representations suitable for ML models.

4. Model Training
A supervised classification model is trained to predict product domains.

5. Model Persistence
The trained model is serialized and stored for reuse in inference pipelines.

# 🗂️ Project Structure
PS-03/

│

├── .venv/ # Python virtual environment (ignored in Git)

│

├── data/ # Datasets

│ ├── combined_products.csv

│ ├── flipkart_cleaned_data.csv

│ └── generated_products.csv

│

├── model/ # Trained ML models

│ └── flipkart_domain_classifier.pkl

│

├── notebooks/ # Experiments & analysis

│ └── ml_pipeline.ipynb

│

├── src/ # Source code

│ ├── concatenate.py

│ ├── data_generation.py

│ ├── domain_name_classifier.py

│ 

├── requirements.txt # Python dependencies

├── .gitignore

├── .gitattributes

└── README.md

# ⚙️ Environment Setup
1️⃣ Create virtual environment

python -m venv .venv

2️⃣ Activate environment

# Windows

.venv\Scripts\activate

# macOS / Linux

source .venv/bin/activate

3️⃣ Install dependencies

pip install -r requirements.txt

# 🚀 Training the Model

Run the training pipeline from the project root:

python src/train_classifier.py

This will:

-Load and preprocess training data

-Train the domain classification model

-Save the trained model to the model/ directory

#🔍 Using the Trained Model (Inference)

import joblib

model = joblib.load("model/flipkart_domain_classifier.pkl")

prediction = model.predict(["Biometric fingerprint authentication device"])

print(prediction)

⚠️ Ensure the same preprocessing logic and library versions are used during inference.

# 📦 Dependencies

Key libraries used:

Python 3.x

pandas

scikit-learn

joblib

sentence-transformers 

All dependencies are pinned in requirements.txt for reproducibility.

# 🧪 Experiments

Exploratory analysis and rapid experimentation are performed in Jupyter notebooks located in the notebooks/ directory.
Notebooks are not considered production code.

# 🛡️ Reproducibility & Best Practices

One virtual environment per project

Frozen dependencies using requirements.txt

Modular pipeline design

Separation of data, code, and models

# 📈 Future Improvements

Add deep learning–based text embeddings

Introduce model versioning

Add evaluation metrics logging

Expose inference via REST API
