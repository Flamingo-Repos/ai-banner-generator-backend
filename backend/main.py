from typing import Optional
import ell

import fal_client


@ell.simple("gpt-4o-mini")
def generate_image_prompt(
    product_name: str, theme: str, extra_input: str, promotional_offer: str
) -> str:
    """
    You are an expert at creating image prompts for advertisements.
    """
    return f"""
    Create an image prompt for an advertisement with the following details:
    Product: {product_name}
    Theme: {theme}
    Additional Information: {extra_input}
    Promotional Offer: {promotional_offer}
    """


def generate_image(
    prompt: str,
    image_size: str = "landscape_4_3",
    num_inference_steps: int = 28,
    seed: Optional[int] = None,
    loras: Optional[list[dict]] = None,
    guidance_scale: float = 3.5,
    sync_mode: bool = True,
    num_images: int = 1,
    enable_safety_checker: bool = False,
    output_format: str = "jpeg",
):
    """
    Generate an image based on the given prompt.
    """
    arguments = {
        "prompt": prompt,
        "image_size": image_size,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images": num_images,
        "enable_safety_checker": enable_safety_checker,
        "output_format": output_format,
        "sync_mode": sync_mode,
    }

    modelName = "fal-ai/flux/dev",
    # modelName = "fal-ai/fast-sdxl"

    # Add optional arguments if provided
    if seed is not None:
        arguments["seed"] = seed

    if loras:
        arguments["loras"] = loras
        modelName = "fal-ai/flux-lora"

    handler = fal_client.submit(modelName, arguments=arguments)

    result = handler.get()
    return result



if __name__ == "__main__":
    print("testing image prompt...")
    prompt = generate_image_prompt(
        "Chai",
        "Road side shop aka tapri",
        "A man is pouring chai",
        "Buy one get one free",
    )

    print(prompt)

    print("testing image generation...")
    result = generate_image(
        prompt=prompt,
    )
    print(result)
