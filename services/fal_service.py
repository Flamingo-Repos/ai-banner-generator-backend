import os
from dotenv import load_dotenv
from typing import List, Optional
import aiohttp

# Load environment variables from .env file
load_dotenv()

# Get the FAL API key from environment variables
FAL_KEY = os.getenv("FAL_KEY")

# Add this dictionary at the beginning of the file, after the imports and environment variable loading

PRODUCT_MODELS = {
    "Coca Cola": {
        "base_model": "fal-ai/flux/dev",
        "loras": None
    },
    "Nike": {
        "base_model": "fal-ai/flux/dev",
        "loras": None
    },
    "Cadbury": {
        "base_model": "fal-ai/flux/dev",
        "loras": None
    },
    "MyWoodCup": {
        "base_model": "fal-ai/flux-lora",
        "loras": [
            {
                "path": "https://storage.googleapis.com/fal-flux-lora/e9bc640224d24ceeb577d4c356ee5b22_lora.safetensors",
                "scale": 1
            }
        ]
    }
}

async def generate_image(
    session: aiohttp.ClientSession,
    product_name: str,
    prompt: str,
    image_size: str = "landscape_4_3",
    num_inference_steps: int = 28,
    seed: Optional[int] = None,
    guidance_scale: float = 3.5,
    num_images: int = 1,
    enable_safety_checker: bool = True,
    output_format: str = "jpeg"
) -> dict:
    # Map custom sizes to FAL API accepted values
    size_mapping = {
        "1024x768": "landscape_4_3",
        "768x1024": "portrait_4_3",
        "1024x1024": "square_hd",
        # Add more mappings as needed
    }

    # Use the mapped size if available, otherwise use the original input
    fal_image_size = size_mapping.get(image_size, image_size)

    arguments = {
        "prompt": prompt,
        "image_size": fal_image_size,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images": num_images,
        "enable_safety_checker": enable_safety_checker,
        "output_format": output_format,
        "sync_mode": True
    }

    product_config = PRODUCT_MODELS.get(product_name, {
        "base_model": "fal-ai/flux/dev",
        "loras": None
    })

    modelName = product_config["base_model"]
    if seed is not None:
        arguments["seed"] = seed
    if product_config["loras"]:
        arguments["loras"] = product_config["loras"]

    try:
        async with session.post(
            f"https://fal.run/{modelName}",
            headers={"Authorization": f"Key {FAL_KEY}"},
            json=arguments
        ) as response:
            result = await response.json()
            print(f"FAL API response status: {response.status}")

            if response.status != 200:
                error_message = result.get('detail', 'Unknown error occurred')
                return {"error": f"FAL API returned status {response.status}: {error_message}"}

            # Check the structure of the result
            if 'images' not in result:
                return {"error": f"FAL API response does not contain 'images' key. Full response: {result}"}

            if not result['images']:
                return {"error": f"FAL API returned empty 'images' list. Full response: {result}"}

            image_data = result['images'][0].get('url', '')
            if image_data.startswith('data:image/'):
                # Extract the base64 part from the data URI
                base64_data = image_data.split(',', 1)[1]
                result['images'][0]['content'] = base64_data
            else:
                return {"error": f"FAL API response does not contain valid image data. Full response: {result}"}

            return result
    except Exception as e:
        print(f"Error in generate_image: {str(e)}")
        return {"error": str(e)}
