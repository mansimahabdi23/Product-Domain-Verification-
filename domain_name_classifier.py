import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = joblib.load("flipkart_domain_classifier.pkl")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
def clean_text(text):
    return str(text).lower().strip()
def compute_similarity(a, b):
    emb = embedder.encode([a, b], normalize_embeddings=True)
    return cosine_similarity(
        [emb[0]], [emb[1]]
    )[0][0]

def verify_product(title, description, category_tree):
    product_text = clean_text(title + " " + description)
    categories = [c.strip() for c in category_tree.split(">")]

    category_text = clean_text(" ".join(categories))
    similarity = compute_similarity(product_text, category_text)

    depth = len(categories)
    sim_threshold = 0.45 if depth == 1 else 0.40 if depth == 2 else 0.35

    decision = "ACCEPT" if similarity >= sim_threshold else "REJECT"

    return {
        "similarity": round(float(similarity), 3),
        "confidence": round(float(model.predict_proba([[similarity]])[0][1]), 3),
        "decision": decision
    }

if __name__ == "__main__":
    print("\n--- Enter Product Details for Verification ---")
    title = input("Product Title: ").strip()
    description = input("Product Description: ").strip()
    category_tree = input("Category Tree (use '>' to separate levels): ").strip()
    result = verify_product(title, description, category_tree)
    print("\n--- Verification Result ---")
    print(f"Similarity Score: {result['similarity']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Decision: {result['decision']}")    


