import openai
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure the API key is set
openai.api_key = os.getenv("OPENAI_API_KEY")

async def generate_text_properties(session, image_description, text_content):
    if not openai.api_key:
        raise ValueError("OpenAI API key is not set. Please check your .env file.")

    prompt = f"""
    Given an image with the following description: "{image_description}"
    and the text to be added: "{text_content}",
    suggest the appropriate placement, size, color, and font for the text.

    Respond in the following JSON format:
    {{
        "placement": "top-left|top-center|top-right|center-left|center|center-right|bottom-left|bottom-center|bottom-right",
        "size": "small|medium|large",
        "color": "hex_color_code",
        "font": "font_name"
    }}
    """

    response = await session.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openai.api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    response.raise_for_status()
    response_json = await response.json()
    return json.loads(response_json['choices'][0]['message']['content'])

def create_text_image(text, properties, image_size):
    image = Image.new('RGBA', image_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    font_size = {'small': 24, 'medium': 36, 'large': 48}[properties['size']]

    # Use a default font
    try:
        # Try to use Arial font (common on Windows)
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            # Try to use DejaVu Sans font (common on Linux)
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except IOError:
            # If both fail, use the default font
            font = ImageFont.load_default()

    # Use getbbox instead of textsize (which is deprecated)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    position = {
        'top-left': (10, 10),
        'top-center': ((image_size[0] - text_width) // 2, 10),
        'top-right': (image_size[0] - text_width - 10, 10),
        'center-left': (10, (image_size[1] - text_height) // 2),
        'center': ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2),
        'center-right': (image_size[0] - text_width - 10, (image_size[1] - text_height) // 2),
        'bottom-left': (10, image_size[1] - text_height - 10),
        'bottom-center': ((image_size[0] - text_width) // 2, image_size[1] - text_height - 10),
        'bottom-right': (image_size[0] - text_width - 10, image_size[1] - text_height - 10),
    }[properties['placement']]

    draw.text(position, text, font=font, fill=properties['color'])

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

async def generate_text_overlay(session, image_description, text_content, image_size):
    try:
        properties = await generate_text_properties(session, image_description, text_content)
        print(f"Generated text properties: {properties}")
    except Exception as e:
        print(f"Error generating text properties: {str(e)}")
        raise

    try:
        text_image = create_text_image(text_content, properties, image_size)
        print("Text image created successfully")
        return text_image, properties
    except Exception as e:
        print(f"Error creating text image: {str(e)}")
        raise
