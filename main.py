from flask import Flask, request, jsonify, Request
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from services.gpt_service import generate_image_prompt
from services.gpt_background_service import generate_background_prompt
from services.fal_service import generate_image
from flask_cors import CORS
import os
from pprint import pprint
import asyncio
import aiohttp
from services.text_generation_service import generate_text_overlay
import json
from PIL import Image
import io
import base64
import time
from werkzeug.utils import secure_filename
from background.service import generate_background

app = Flask(__name__)
CORS(app)

# Configure upload folder for temporary file storage
UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
        background_prompt = await generate_background_prompt(session, ad_request.theme)
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

        try:
            # Decode the base64 image
            background_image_data = base64.b64decode(background_image_base64)
            background_image = Image.open(io.BytesIO(background_image_data))
            print(f"Background image decoded successfully. Size: {background_image.size}, Mode: {background_image.mode}")
        except Exception as e:
            print(f"Error decoding background image: {str(e)}")
            return {"error": f"Error decoding background image: {str(e)}"}

        # Generate text overlay
        text_overlay, text_properties = await generate_text_overlay(
            session,
            background_prompt,  # Use the background prompt as the image description
            ad_request.text_overlay,
            background_image.size
        )

        try:
            # Create text overlay image
            text_overlay_image = Image.open(io.BytesIO(base64.b64decode(text_overlay)))
            print("Text overlay image created successfully")
        except Exception as e:
            print(f"Error creating text overlay image: {str(e)}")
            raise

        try:
            # Overlay text on background
            background_image.paste(text_overlay_image, (0, 0), text_overlay_image)
            print("Text overlaid on background successfully")
        except Exception as e:
            print(f"Error overlaying text on background: {str(e)}")
            raise

        try:
            # Save the combined image to a file
            output_dir = "generated_banners"
            os.makedirs(output_dir, exist_ok=True)
            file_name = f"banner_{product_name}_{int(time.time())}.png"
            file_path = os.path.join(output_dir, file_name)
            background_image.save(file_path, format="PNG")
            print(f"Combined image saved successfully to {file_path}")

            # Also create the base64 string
            buffered = io.BytesIO()
            background_image.save(buffered, format="PNG")
            combined_image_base64 = base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            print(f"Error saving combined image: {str(e)}")
            raise

        return {
            "prompt": background_prompt,
            "background_image": background_image_base64,
            "text_overlay_properties": text_properties,
            "combined_image": combined_image_base64,
            "saved_image_path": file_path
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


@app.route('/generate-background', methods=['POST'])
def generate_banner_api():
    try:
        # Check if guidelines file is included in request
        if 'guidelines_file' not in request.files:
            return jsonify({"error": "Guidelines file is required"}), 400

        # Get form data
        company_context = request.form.get('company_context')
        event_context = request.form.get('event_context')

        if not company_context or not event_context:
            return jsonify({"error": "Both company_context and event_context are required"}), 400

        # Save the uploaded file temporarily
        file = request.files['guidelines_file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Generate banner using the existing function
            generated_banners = generate_background(
                guidelines_file_path=filepath,
                company_context=company_context,
                event_context=event_context
            )

            # Clean up the temporary file
            os.remove(filepath)

            # Extract URLs and format response
            banner_urls = []
            top_urls = []
            for banner in generated_banners:
                if 'image' in banner and 'images' in banner['image']:
                    banner_urls.append({
                        "prompt": banner['background_prompt'],
                        "urls": [img['url'] for img in banner['image']['images'] if 'url' in img],
                        "text_specifications": banner.get('text_specifications', {
                            "content": {},
                            "typography": {},
                            "colors": {},
                            "layout": {}
                        })
                    })
                    #pick the last urls from the banner_urls
                    top_urls.append(banner_urls[-1]['urls'][-1])

            if not banner_urls:
                return jsonify({"error": "No valid images were generated"}), 500

            return jsonify({
                "banners": banner_urls,
                "status": "success",
                "count": len(banner_urls),
                "urls": top_urls
            }), 200

        except Exception as e:
            # Clean up the temporary file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e

    except Exception as e:
        print(f"Error in generate_banner_api: {str(e)}")  # Add logging
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

