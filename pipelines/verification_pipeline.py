from utils.embedding_utils import get_image_embedding, get_text_embedding
from utils.scoring_utils import cosine_similarity, classify_score

def verify_image(
    model,
    processor,
    image,
    company_domain_text
):
    image_emb = get_image_embedding(model, processor, image)
    domain_emb = get_text_embedding(model, processor, company_domain_text)

    score = cosine_similarity(image_emb, domain_emb)
    decision = classify_score(score)

    return {
        "similarity_score": round(score, 4),
        "decision": decision
    }
