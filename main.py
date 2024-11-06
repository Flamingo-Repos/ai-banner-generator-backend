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
from werkzeug.utils import secure_filename

from background.service import generate_banner

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

async def async_generate_ad(data):
    ad_request = AdRequest(**data)
    layout_types = ["center", "right", "left", "stylized"]

    async with aiohttp.ClientSession() as session:
        tasks = [generate_prompt_and_image(ad_request, layout_type, session) for layout_type in layout_types]
        results = await asyncio.gather(*tasks)

    return results

@app.route("/generate-ad", methods=["POST"])
def generate_ad():
    data = request.json
    results = asyncio.run(async_generate_ad(data))
    return jsonify(results)

@app.route('/generate-banner', methods=['POST'])
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
            generated_banners = generate_banner(
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
