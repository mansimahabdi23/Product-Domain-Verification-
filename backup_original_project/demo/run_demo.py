import json
import os

from models.clip_model import load_clip
from pipelines.verification_pipeline import verify_image
from utils.image_utils import load_images_by_category
from config.company_domains import COMPANY_DOMAINS

OUTPUT_PATH = "output/results.json"

def run_demo():
    model, processor = load_clip()

    company = "tv_company"
    company_domain = COMPANY_DOMAINS[company]

    test_images = {
        "tv": "data/samples/tv/1.jpg",
        "tshirt": "data/samples/tshirt/1.jpg",
        "sofa": "data/samples/sofa/1.jpg"
    }

    results = {}

    for label, path in test_images.items():
        image = load_images_by_category(path)
        result = verify_image(
            model,
            processor,
            image,
            company_domain
        )
        results[label] = result

    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_demo()
