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
