from flask import Flask, request, jsonify, Request
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from services.gpt_service import generate_image_prompt
from services.gpt_background_service import generate_background_prompt
from services.fal_service import generate_image
from services.overlay_service import overlay_images
from flask_cors import CORS
import asyncio
import aiohttp
from services.text_svg_generation_service import generate_text_svg_code

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
    image_size: str = "landscape_4_3"  # Changed default to an accepted value
    num_inference_steps: int = 28
    seed: Optional[int] = None
    loras: Optional[List[dict]] = None
    guidance_scale: float = 3.5
    enable_safety_checker: bool = True
    output_format: str = "jpeg"
    num_images: int = field(default=1)
    flow_type: Literal["product_marketing", "banner_creation"] = "product_marketing"
    banner_types: List[str] = field(default_factory=lambda: ['default'])
    text_overlay: str = "summer sale bonanza 50% off"  # Default text for testing
    text_overlay_position: Optional[str] = None


async def generate_product_marketing(ad_request, layout_type, session):
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
        product_name=ad_request.product_name,
        prompt=prompt,
        image_size=ad_request.image_size,
        num_inference_steps=ad_request.num_inference_steps,
        seed=ad_request.seed,
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

async def generate_banner(session, ad_request, product_name, banner_type):
    try:
        # Generate background prompt
        background_prompt: str = await generate_background_prompt(session, ad_request.theme)
        print(f"Generated background prompt: {background_prompt}")

        # Generate background image
        background_result = await generate_image(
            session,
            product_name=ad_request.product_name,
            prompt=background_prompt,
            image_size=ad_request.image_size,
            num_inference_steps=ad_request.num_inference_steps,
            seed=ad_request.seed,
            guidance_scale=ad_request.guidance_scale,
            num_images=1,
            enable_safety_checker=ad_request.enable_safety_checker,
            output_format=ad_request.output_format
        )
        print(f"Full background result: {background_result}")

        if 'error' in background_result:
            raise ValueError(f"Error in image generation: {background_result['error']}")

        if 'images' not in background_result or not background_result['images']:
            raise ValueError(f"No images generated. Full response: {background_result}")

        background_image_base64 = background_result['images'][0]['content']

        # Generate text overlay
        properties = await generate_text_svg_code(
            session,
            background_prompt,  # Us the background prompt as the image description
            ad_request.text_overlay
            # background_image.size
        )
        if ad_request.text_overlay_position:
            properties["placement"] = ad_request.text_overlay_position

        try:
            # overlay text on background
            final_banner_base64 = overlay_images(background_image_base64, properties)
        except Exception as e:
            print(f"Error decoding background image: {str(e)}")
            return {"error": f"Error decoding background image: {str(e)}"}

        return {
            "prompt": background_prompt,
            "url": final_banner_base64,
            # "text_overlay_properties": text_properties,
            # "combined_image": combined_image_base64,
            # "saved_image_path": file_path
        }

    except Exception as e:
        print(f"Error in generate_banner: {str(e)}")
        return {"error": str(e)}

async def async_generate_ad(data):
    async with aiohttp.ClientSession() as session:
        tasks = []
        # Check if 'banner_types' exists in the data, if not, use a default value
        banner_types = data.get('banner_types', ['default'])
        for banner_type in banner_types:
            ad_request = AdRequest(**data)
            tasks.append(generate_banner(session, ad_request, ad_request.product_name, banner_type))
        results = await asyncio.gather(*tasks)
    return results

@app.route("/generate-ad", methods=["POST"])
def generate_ad():
    try:
        data = request.json
        if 'text_overlay' not in data:
            data['text_overlay'] = "summer sale bonanza 50% off"  # Default text if not provided
        results = asyncio.run(async_generate_ad(data))
        return jsonify(results)
    except Exception as e:
        print(f"Error in generate_ad: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.post("/test-text-overlay")
async def test_text_overlay(request: Request):
    data = await request.json()
    text_content = data.get("text_content", "Sample Text")
    image_description = data.get("image_description", "A blank canvas")
    image_size = tuple(map(int, data.get("image_size", "800x600").split("x")))

    async with aiohttp.ClientSession() as session:
        text_overlay = await generate_text_overlay(session, image_description, text_content, image_size)

    return {"text_overlay": text_overlay}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
