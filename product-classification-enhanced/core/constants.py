"""Application constants"""

# Confidence levels
CONFIDENCE_LEVELS = {
    'VERY_HIGH': 0.35,
    'HIGH': 0.30,
    'MEDIUM': 0.25,
    'LOW': 0.20,
    'VERY_LOW': 0.15
}

# Decision categories
DECISIONS = {
    'IN_DOMAIN': 'Product is within the company domain',
    'NEAR_DOMAIN': 'Product is related to the company domain',
    'OUT_OF_DOMAIN': 'Product is not in the company domain'
}

# Default categories
DEFAULT_CATEGORIES = [
    'electronics', 'furniture', 'clothing', 'kitchen_appliances',
    'home_decor', 'sports', 'books', 'toys', 'beauty', 'automotive'
]
