from flask import Flask, request, jsonify
from dataclasses import dataclass
from typing import List, Optional
from services.gpt_service import generate_image_prompt
from services.fal_service import generate_image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@dataclass
class AdRequest:
    product_name: str
    theme: str
    extra_input: str
    promotional_offer: str
    image_size: str = "landscape_4_3"
    num_inference_steps: int = 28
    seed: Optional[int] = None
    loras: Optional[List[dict]] = None
    guidance_scale: float = 3.5
    num_images: int = 1
    enable_safety_checker: bool = True
    output_format: str = "jpeg"

@app.route("/generate-ad", methods=["POST"])
def generate_ad():
    data = request.json
    ad_request = AdRequest(**data)

    # Generate prompt using GPT-4
    prompt = generate_image_prompt(
        ad_request.product_name,
        ad_request.theme,
        ad_request.extra_input,
        ad_request.promotional_offer
    )

    # Generate image using FAL AI
    result = generate_image(
        prompt=prompt,
        image_size=ad_request.image_size,
        num_inference_steps=ad_request.num_inference_steps,
        seed=ad_request.seed,
        loras=ad_request.loras,  # This is already a list of dictionaries
        guidance_scale=ad_request.guidance_scale,
        num_images=ad_request.num_images,
        enable_safety_checker=ad_request.enable_safety_checker,
        output_format=ad_request.output_format
    )

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
