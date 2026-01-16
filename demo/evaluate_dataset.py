import json
from collections import defaultdict

from models.clip_model import load_clip
from pipelines.verification_pipeline import verify_image
from utils.image_utils import load_images_by_category
from config.company_domains import COMPANY_DOMAINS

DATA_DIR = "data"
OUTPUT_FILE = "output/evaluation_results.json"

def main():
    model, processor = load_clip()
    company_domain = COMPANY_DOMAINS["furniture_company"]

    dataset = load_images_by_category(DATA_DIR)

    results = defaultdict(list)

    for category, images in dataset.items():
        print(f"Processing category: {category} ({len(images)} images)")
        for img in images:
            result = verify_image(
                model=model,
                processor=processor,
                image=img,
                company_domain_text=company_domain
            )
            results[category].append(result)

    summary = {}

    for category, entries in results.items():
        in_count = sum(1 for e in entries if e["decision"] == "IN_DOMAIN")
        near_count = sum(1 for e in entries if e["decision"] == "NEAR_DOMAIN")
        out_count = sum(1 for e in entries if e["decision"] == "OUT_OF_DOMAIN")

        summary[category] = {
            "total_images": len(entries),
            "IN_DOMAIN": in_count,
            "NEAR_DOMAIN": near_count,
            "OUT_OF_DOMAIN": out_count
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nEvaluation Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
