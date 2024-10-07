import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def generate_image_prompt(product_name: str, theme: str, extra_input: str, promotional_offer: str) -> str:
    # Initialize OpenAI client with the API key from environment variable
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Construct the message for GPT-4 with the new instructions
    message = f"""You are a specialist in creating detailed prompts for AI image generation, specifically for product posters with a graphic design focus. You will receive three inputs:
Product: {product_name}
Theme: {theme}
Promotional Offer: {promotional_offer}
Your task is to create a comprehensive prompt for an AI image generator to create a product poster. Follow this structure strictly:

Start by emphasizing the text overlay. The promotional offer MUST be the most prominent text element in the image. Describe its size, font style, color, and exact placement in detail.
Describe the main product in photorealistic detail, including its position in the image.
Detail the surrounding elements that relate to the theme, focusing on graphic design elements rather than photorealistic backgrounds. Think in terms of patterns, shapes, and stylized representations.
Specify the style references (e.g., flat design, vector art, graphic poster art). Avoid references to photorealistic styles.
Describe the overall color scheme and how it relates to the brand and theme.
Include graphic elements to create visual interest and flow.
Use technical design terms (e.g., composition, layout, visual hierarchy).
Incorporate descriptive adjectives for mood and impact throughout the prompt.

Ensure the prompt is cohesive and creates a vivid mental image of a graphic poster. Aim for a length of 150-200 words.
Here's an example prompt to guide you:
"Create a graphic poster for Coca-Cola's Holi celebration offer. The text '25% OFF' must dominate the top third of the image in large, bold, white font with a red outline, ensuring high contrast and readability. Below, place a photorealistic Coca-Cola bottle at a 30-degree angle, water droplets glistening on its surface. The background should be a flat, geometric pattern inspired by Holi, using vibrant purples, pinks, and yellows in abstract, splattered shapes. Include stylized, vector-art hands throwing colorful powder from the bottom corners, creating a sense of motion. The Coca-Cola logo should be prominently displayed at the bottom center. Use a visual hierarchy that draws the eye from the '25% OFF' text to the bottle, then to the dynamic elements. The overall style should evoke modern graphic design with a nod to traditional Holi elements. Employ a vibrant, high-contrast color palette to capture the festive spirit of Holi while maintaining brand recognition."
Now, create a similar prompt based on the provided inputs, ensuring a strong emphasis on text display and graphic design elements."""

    # Call GPT-4 to generate the prompt
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert at creating detailed image prompts for product advertisements."},
            {"role": "user", "content": message}
        ]
    )

    return response.choices[0].message.content
