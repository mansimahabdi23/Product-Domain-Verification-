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
