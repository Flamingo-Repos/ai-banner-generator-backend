from flask import Flask, request, jsonify
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from services.gpt_service import generate_image_prompt
from services.fal_service import generate_image
from flask_cors import CORS
import os
from pprint import pprint

app = Flask(__name__)
CORS(app)


# add hello world route
@app.route("/")
def hello_world():
    return "Hello, World!"

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
    enable_safety_checker: bool = True
    output_format: str = "jpeg"
    num_images: int = field(default=1)  # Add this line

@app.route("/generate-ad", methods=["POST"])
def generate_ad():
    data = request.json
    ad_request = AdRequest(**data)

    layout_types = ["center", "right", "left", "stylized"]
    results = []

    for layout_type in layout_types:
        # Generate prompt using GPT-4
        prompt = generate_image_prompt(
            ad_request.product_name,
            ad_request.theme,
            ad_request.extra_input,
            ad_request.promotional_offer,
            layout_type
        )

        # Generate image using FAL AI
        result = generate_image(
            prompt=prompt,
            image_size=ad_request.image_size,
            num_inference_steps=ad_request.num_inference_steps,
            seed=ad_request.seed,
            loras=ad_request.loras,
            guidance_scale=ad_request.guidance_scale,
            num_images=1,
            enable_safety_checker=ad_request.enable_safety_checker,
            output_format=ad_request.output_format
        )

        results.append({
            "layout_type": layout_type,
            "prompt": prompt,
            "image_result": result
        })

        # Pretty print the result structure
        print(f"Result structure for {layout_type}:")
        pprint(result)

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
