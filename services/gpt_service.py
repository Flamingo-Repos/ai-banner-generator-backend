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
"Text Overlay: "25% OFF" must dominate the top third of the image in bold, glowing golden letters, resembling illuminated Diwali lights. The text should occupy 30% of the poster's area, with a slight 3D effect for depth. Position it centrally, arching slightly to follow the curvature of a Coca-Cola bottle cap.
Main Product: A photorealistic Coca-Cola bottle takes center stage, positioned slightly off-center to the right. The bottle should appear chilled, with small water droplets condensing on its surface. The iconic Coca-Cola logo must be clearly visible and gleaming, as if reflecting festive lights.
Surrounding Elements: Frame the bottle with a stylized rangoli pattern, using geometric shapes and swirls in vibrant colors. Incorporate simplified, vector-style diyas (oil lamps) arranged in a semi-circle at the bottom of the poster. Add abstract firework bursts in the upper corners using radial line patterns.
Style References: Blend modern flat design with traditional Indian art motifs. Use clean lines and bold shapes reminiscent of contemporary graphic poster art, while incorporating elements of intricate Diwali patterns.
Color Scheme: Dominate with deep reds and golds, Coca-Cola's signature colors, accented with purples, oranges, and whites to evoke the richness of Diwali celebrations. Use a gradient background transitioning from deep red at the bottom to a warm golden glow at the top.
Graphic Elements: Include stylized, vector light streams emanating from the diyas, intersecting with the Coca-Cola bottle to create a sense of radiance and flow. Add subtle, geometric sparkle effects around the "25% OFF" text to enhance its prominence.
Technical Terms: Employ a radial composition centered on the Coca-Cola bottle. Utilize negative space around the product to enhance visual hierarchy. Create depth through layering of graphic elements.
Mood and Impact: Craft a vibrant, jubilant atmosphere that exudes warmth and celebration. The poster should feel festive, inviting, and dynamically energetic, capturing the spirit of Diwali while highlighting Coca-Cola as an integral part of the celebration."
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
