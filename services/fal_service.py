import os
from dotenv import load_dotenv
from typing import List, Optional
import aiohttp

# Load environment variables from .env file
load_dotenv()

# Get the FAL API key from environment variables
FAL_KEY = os.getenv("FAL_KEY")

async def generate_image(
    session: aiohttp.ClientSession,
    prompt: str,
    image_size: str = "landscape_4_3",
    num_inference_steps: int = 28,
    seed: Optional[int] = None,
    loras: Optional[List[dict]] = None,
    guidance_scale: float = 3.5,
    num_images: int = 1,
    enable_safety_checker: bool = True,
    output_format: str = "jpeg"
) -> str:
    arguments = {
        "prompt": prompt,
        "image_size": image_size,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images": num_images,
        "enable_safety_checker": enable_safety_checker,
        "output_format": output_format,
        "sync_mode": True
    }

    modelName = "fal-ai/flux/dev"
    if seed is not None:
        arguments["seed"] = seed
    if loras:
        arguments["loras"] = loras
        modelName = "fal-ai/flux-lora"

    async with session.post(
        f"https://fal.run/{modelName}",
        headers={"Authorization": f"Key {FAL_KEY}"},
        json=arguments
    ) as response:
        result = await response.json()
        return result
