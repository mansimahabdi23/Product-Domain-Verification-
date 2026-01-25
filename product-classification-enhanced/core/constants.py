"""Application constants"""

# Confidence levels based on similarity score
CONFIDENCE_LEVELS = {
    'VERY_HIGH': 0.35,    # 35%+ similarity
    'HIGH': 0.30,         # 30-34% similarity
    'MEDIUM': 0.25,       # 25-29% similarity
    'LOW': 0.20,          # 20-24% similarity
    'VERY_LOW': 0.15      # 15-19% similarity
}

# Decision categories with explanations
DECISIONS = {
    'IN_DOMAIN': 'The product is clearly within the company\'s domain',
    'NEAR_DOMAIN': 'The product is somewhat related to the company\'s domain',
    'OUT_OF_DOMAIN': 'The product is not related to the company\'s domain',
    'UNKNOWN': 'Unable to determine if product is in domain'
}

# Default categories for initialization
DEFAULT_CATEGORIES = [
    'electronics', 'furniture', 'clothing', 'kitchen_appliances',
    'home_decor', 'sports', 'books', 'toys', 'beauty', 'automotive'
]

# Classification thresholds (CLIP similarity scores are typically low)
CLASSIFICATION_THRESHOLDS = {
    'STRONG_MATCH': 0.25,      # 25%+ similarity = strong match
    'MODERATE_MATCH': 0.20,    # 20-24% similarity = moderate match
    'WEAK_MATCH': 0.15,        # 15-19% similarity = weak match
    'NO_MATCH': 0.10           # Below 10% = no match
}
