import os
from dotenv import load_dotenv
from typing import Literal
import aiohttp

# Load environment variables from .env file
load_dotenv()

async def generate_image_prompt(
    product_name: str,
    theme: str,
    extra_input: str,
    promotional_offer: str,
    prompt_type: Literal["center", "right", "left", "stylized"],
    session: aiohttp.ClientSession
) -> str:
    openai_api_key = os.getenv("OPENAI_API_KEY")

    system_message = """You are a specialist in creating detailed prompts for AI image generation, specifically for product posters with a graphic design focus. Your task is to create comprehensive prompts for an AI image generator to create product posters. Follow this structure strictly:

    1. Start by emphasizing the text overlay. The promotional offer MUST be a prominent text element in the image. Describe its size, font style, color, and exact placement in detail.
    2. Describe the main product in photorealistic detail, including its position in the image.
    3. Detail the surrounding elements that relate to the theme, focusing on graphic design elements rather than photorealistic backgrounds. Think in terms of patterns, shapes, and stylized representations.
    4. Specify the style references (e.g., flat design, vector art, graphic poster art).
    5. Describe the overall color scheme and how it relates to the brand and theme.
    6. Include graphic elements to create visual interest and flow.
    7. Use technical design terms (e.g., composition, layout, visual hierarchy).
    8. Incorporate descriptive adjectives for mood and impact throughout the prompt.

    Ensure the prompt is cohesive and creates a vivid mental image of a graphic poster. Aim for a length of 150-200 words."""

    user_message = f"""Create a comprehensive prompt for an AI image generator to create a product poster based on the following inputs:

    Product: {product_name}
    Theme: {theme}
    Promotional Offer: {promotional_offer}
    Additional Information: {extra_input}

    Layout Type: {prompt_type}

    For the layout, follow these guidelines:
    - If "center": Place the product in the center of the image with the text overlay at the top.
    - If "right": Position the product on the right side of the image with the text overlay on the left.
    - If "left": Position the product on the left side of the image with the text overlay on the right.
    - If "stylized": Create a unique, stylized representation of the product, integrating it creatively with the overall design.

    Now, create a detailed prompt based on the provided inputs and specified layout, ensuring a strong emphasis on text display and graphic design elements."""

    async with session.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {openai_api_key}"},
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        }
    ) as response:
        result = await response.json()
        return result['choices'][0]['message']['content']
