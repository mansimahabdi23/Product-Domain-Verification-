import pandas as pd
import random

DOMAINS = {
    "Agriculture > Fertilizers > Organic": [
        "Organic neem fertilizer for crops",
        "Bio compost fertilizer for soil health",
        "Natural organic manure for farming"
    ],
    "Services > Healthcare > Mental Health": [
        "Online mental health counseling session",
        "Stress management therapy program",
        "Psychological wellness consultation"
    ],
    "Software > Developer Tools": [
        "Cloud-based code deployment platform",
        "API monitoring and debugging software",
        "DevOps automation tool"
    ]
}

rows = []

for category, samples in DOMAINS.items():
    for s in samples:
        rows.append({
            "product_name": s,
            "description": f"{s} designed for professional and commercial use.",
            "product_category_tree": category
        })

df = pd.DataFrame(rows)
df.to_csv("data/generated_products.csv", index=False)
print("Generated synthetic domain products.")
