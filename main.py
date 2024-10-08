from flask import Flask, request, jsonify
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from services.gpt_service import generate_image_prompt
from services.fal_service import generate_image
from flask_cors import CORS
import os
from pprint import pprint
import asyncio
import aiohttp

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
    num_images: int = field(default=1)

async def generate_prompt_and_image(ad_request, layout_type, session):
    prompt = await generate_image_prompt(
        ad_request.product_name,
        ad_request.theme,
        ad_request.extra_input,
        ad_request.promotional_offer,
        layout_type,
        session
    )

    result = await generate_image(
        session,
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

    return {
        "layout_type": layout_type,
        "prompt": prompt,
        "url": result['images'][0]['url'],
        "content_type": result['images'][0]['content_type'],
    }

async def async_generate_ad(data):
    ad_request = AdRequest(**data)
    layout_types = ["center", "right", "left", "stylized"]

    async with aiohttp.ClientSession() as session:
        tasks = [generate_prompt_and_image(ad_request, layout_type, session) for layout_type in layout_types]
        results = await asyncio.gather(*tasks)

    for result in results:
        print(f"Result structure for {result['layout_type']}:")
        pprint(result['image_result'])

    return results

@app.route("/generate-ad", methods=["POST"])
def generate_ad():
    data = request.json
    results = asyncio.run(async_generate_ad(data))
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
