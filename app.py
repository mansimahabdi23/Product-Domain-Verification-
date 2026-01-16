from flask import Flask, render_template, request
from PIL import Image
import io

from models.clip_model import load_clip
from pipelines.verification_pipeline import verify_image

app = Flask(__name__)

# Load model ONCE at startup
model, processor = load_clip()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        try:
            domain_text = request.form["domain"]
            image_file = request.files["image"]

            if not domain_text or not image_file:
                raise ValueError("Domain text and image are required.")

            image = Image.open(io.BytesIO(image_file.read())).convert("RGB")

            result = verify_image(
                model=model,
                processor=processor,
                image=image,
                company_domain_text=domain_text
            )

        except Exception as e:
            error = str(e)

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
