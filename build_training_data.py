import pandas as pd
import random 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def similarity(a, b):
    emb = embedder.encode([a, b], normalize_embeddings=True)
    return cosine_similarity(
        [emb[0]], [emb[1]]
    )[0][0]

flipkart_data = pd.read_csv("data/flipkart_cleaned_data.csv")
generated_data = pd.read_csv("data/generated_products.csv")

pairs = []

all_categories = generated_data["product_category_tree"].unique().tolist()
for _, row in generated_data.iterrows():
    product_text = f"{row['product_name']} {row['description']}"
    pos_sim = similarity(product_text, row["product_category_tree"])
    pairs.append([pos_sim, 1])

    wrong_cat = random.choice(
        [c for c in all_categories if c != row["product_category_tree"]]
    )

    neg_sim = similarity(product_text, wrong_cat)
    pairs.append([neg_sim, 0])

df_pairs = pd.DataFrame(pairs, columns=["similarity", "label"])
df_pairs.to_csv("data/training_pairs.csv", index=False)

print("Built training data with similarity scores.")