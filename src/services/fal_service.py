import os
import fal_client
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables from .env file
load_dotenv()

# Get the FAL API key from environment variables
FAL_KEY = os.getenv("FAL_KEY")

def generate_image(
    prompt: str,
    image_size: str = "landscape_4_3",
    num_inference_steps: int = 28,
    seed: Optional[int] = None,
    loras: Optional[List[dict]] = None,
    guidance_scale: float = 3.5,
    sync_mode: bool = True,
    num_images: int = 1,
    enable_safety_checker: bool = True,
    output_format: str = "jpeg"
) -> str:
    # Prepare the arguments for the FAL AI API
    arguments = {
        "prompt": prompt,
        "image_size": image_size,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images": num_images,
        "enable_safety_checker": enable_safety_checker,
        "output_format": output_format,
        "sync_mode": sync_mode
    }

    modelName = "fal-ai/flux/dev",
    # Add optional arguments if provided
    if seed is not None:
        arguments["seed"] = seed
    if loras:
        arguments["loras"] = loras
        modelName = "fal-ai/flux-lora"

    # Submit the request to FAL AI
    handler = fal_client.submit(
        modelName,
        arguments=arguments
    )

    # Get the result
    result = handler.get()

    # Return the image URL
    return result
