import torch.nn.functional as F

IN_DOMAIN_THRESHOLD = 0.30
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
