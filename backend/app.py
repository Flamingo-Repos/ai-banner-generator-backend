import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from backend.main import generate_image, generate_image_prompt

app = FastAPI()


class AdRequest(BaseModel):
    product_name: str
    theme: str
    extra_input: str
    promotional_offer: str
    image_size: str = "landscape_4_3"
    num_inference_steps: int = 28
    seed: Optional[int] = None
    loras: Optional[list[dict]] = None
    guidance_scale: float = 3.5
    num_images: int = 1
    enable_safety_checker: bool = True
    output_format: str = "jpeg"


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/generate-ad")
def generate_ad(ad_request: AdRequest):
    # Generate prompt using GPT-4
    prompt = generate_image_prompt(
        ad_request.product_name,
        ad_request.theme,
        ad_request.extra_input,
        ad_request.promotional_offer,
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
        output_format=ad_request.output_format,
    )

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
