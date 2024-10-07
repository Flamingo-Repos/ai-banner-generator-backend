import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def generate_image_prompt(product_name: str, theme: str, extra_input: str, promotional_offer: str) -> str:
    # Initialize OpenAI client with the API key from environment variable
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Construct the message for GPT-4 with the new instructions
    message = f"""You are a specialist in creating detailed prompts for AI image generation, particularly for product posters. You will receive three inputs:
Product: {product_name}
Theme: {theme}
Promotional Offer: {promotional_offer}

Your task is to create a comprehensive prompt for an AI image generator to create a product poster. Follow this structure strictly:

1. Start with a brief overview of the poster concept.
2. Describe the main product in photorealistic detail, including its position in the image.
3. Detail the surrounding elements that relate to the theme.
4. Specify the style references (e.g., high-speed photography, graphic poster art).
5. Describe atmospheric details like lighting and color contrasts.
6. Include action elements to create dynamism.
7. Emphasize the text overlay, especially the promotional offer, describing its appearance and position. This should be a priority in the image.
8. Use technical photography terms.
9. Incorporate descriptive adjectives for mood and impact throughout the prompt.

Ensure the prompt is cohesive and creates a vivid mental image. Aim for a length of 150-200 words.

Here's an example prompt to guide you:
"A vibrant and dynamic product poster for Coca-Cola celebrating Holi. Photorealistic Coca-Cola bottle in the center, drops of water glistening on its surface, logo crisp and clear. Exploding around the bottle, a kaleidoscope of colorful powder in mid-air - vibrant pinks, electric blues, sunny yellows, and lush greens. Background transitions from pure white to a gradient of Holi colors. Freeze-frame action of hands throwing powder, creating a sense of motion and joy. Lighting is bright and energetic, emphasizing the contrast between the dark cola and the colorful powders. Text overlay in a modern, bold font: 'Add Color to Your Celebration' with '25% OFF' prominently displayed at the top. Style inspired by high-speed photography and graphic poster art. 30mm shot, professional, bold colors, awe-inspiring, cinematic feel. The overall composition is balanced but dynamic, drawing the eye to the Coca-Cola bottle while celebrating the explosive energy of Holi."

Now, create a similar prompt based on the provided inputs."""

    # Call GPT-4 to generate the prompt
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert at creating detailed image prompts for product advertisements."},
            {"role": "user", "content": message}
        ]
    )

    return response.choices[0].message.content
