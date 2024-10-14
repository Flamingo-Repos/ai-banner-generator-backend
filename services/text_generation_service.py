import openai
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import io
import base64
import json
import os
from dotenv import load_dotenv
from typing import Tuple
import logging

# Load environment variables from .env file
load_dotenv()

# Ensure the API key is set
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def generate_text_properties(session, image_description, text_content):
    if not openai.api_key:
        raise ValueError("OpenAI API key is not set. Please check your .env file.")

    prompt = f"""
    Given an image with the following description: "{image_description}"
    and the text to be added: "{text_content}",
    suggest the appropriate placement, size (in pixels), color, and font for the text.

    Use only the following options for placement:
    center, center top, center bottom, left, right, left top, left bottom, right top, right bottom

    The size should be a number between 12 and 120 pixels.

    Respond in the following JSON format:
    {{
    "placement": "center",
    "size": 48,
    "color": "#FFFFFF",
    "font": "impact",
    "effects": {{
        "outline": {{
            "color": "#000000",
            "width": 2
        }},
        "shadow": {{
            "color": "#00000080",
            "offset": [2, 2]
        }},
        "gradient": {{
            "colors": ["#FF0000", "#00FF00", "#0000FF"],
            "direction": "horizontal"
        }}
        }}
    }}"""

    try:
        response = await session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openai.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4",  # Make sure this is the correct model name
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        response_json = await response.json()
        logger.debug(f"API Response: {response_json}")

        if 'choices' not in response_json or len(response_json['choices']) == 0:
            raise ValueError("Unexpected API response format")

        content = response_json['choices'][0]['message']['content']
        logger.debug(f"Content: {content}")

        properties = json.loads(content)

        # Ensure shadow offset is a list
        if 'effects' in properties and 'shadow' in properties['effects']:
            properties['effects']['shadow']['offset'] = list(properties['effects']['shadow']['offset'])

        logger.debug(f"Parsed properties: {properties}")
        return properties
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Content that failed to decode: {content}")
        raise
    except Exception as e:
        logger.error(f"Error in generate_text_properties: {str(e)}")
        raise

def create_text_image(text, properties, image_size):
    image = Image.new('RGBA', image_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    font_size = properties['size']

    # Expanded font options
    font_paths = {
        'arial': "arial.ttf",
        'arial bold': "arialbd.ttf",
        'times': "times.ttf",
        'times new roman': "times.ttf",
        'verdana': "verdana.ttf",
        'comic': "comic.ttf",
        'impact': "impact.ttf",
        'georgia': "georgia.ttf",
    }

    try:
        font_name = properties['font'].lower()
        font = ImageFont.truetype(font_paths.get(font_name, "arial.ttf"), font_size)
    except IOError:
        logger.warning(f"Font {properties['font']} not found. Using default font.")
        try:
            # Try to use a default TrueType font
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            # If that fails, use the default bitmap font
            font = ImageFont.load_default()
            logger.warning("Default TrueType font not found. Using bitmap font.")

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    position = calculate_position(properties['placement'], image_size, text_width, text_height)

    # Apply text effects
    if 'effects' in properties:
        if 'outline' in properties['effects']:
            draw_outline_text(draw, position, text, font, properties)
        if 'shadow' in properties['effects']:
            draw_shadow_text(draw, position, text, font, properties)
        if 'gradient' in properties['effects']:
            draw_gradient_text(draw, position, text, font, properties)
    else:
        draw.text(position, text, font=font, fill=properties['color'])

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def calculate_position(placement, image_size, text_width, text_height) -> Tuple[int, int]:
    positions = {
        'center': ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2),
        'center top': ((image_size[0] - text_width) // 2, 10),
        'center bottom': ((image_size[0] - text_width) // 2, image_size[1] - text_height - 10),
        'left': (10, (image_size[1] - text_height) // 2),
        'right': (image_size[0] - text_width - 10, (image_size[1] - text_height) // 2),
        'left top': (10, 10),
        'left bottom': (10, image_size[1] - text_height - 10),
        'right top': (image_size[0] - text_width - 10, 10),
        'right bottom': (image_size[0] - text_width - 10, image_size[1] - text_height - 10),
    }

    # Convert placement to lowercase and replace underscore with space
    normalized_placement = placement.lower().replace('_', ' ')

    if normalized_placement in positions:
        return positions[normalized_placement]
    else:
        # Default to center if an invalid placement is provided
        logger.warning(f"Invalid placement '{placement}'. Defaulting to center.")
        return positions['center']

def draw_outline_text(draw, position, text, font, properties):
    outline_color = properties['effects']['outline']['color']
    outline_width = properties['effects']['outline']['width']
    for offset_x in range(-outline_width, outline_width + 1):
        for offset_y in range(-outline_width, outline_width + 1):
            draw.text((position[0] + offset_x, position[1] + offset_y), text, font=font, fill=outline_color)
    draw.text(position, text, font=font, fill=properties['color'])

def draw_shadow_text(draw, position, text, font, properties):
    shadow_color = properties['effects']['shadow']['color']
    shadow_offset = tuple(properties['effects']['shadow']['offset'])  # Convert list to tuple
    shadow_position = (position[0] + shadow_offset[0], position[1] + shadow_offset[1])
    draw.text(shadow_position, text, font=font, fill=shadow_color)
    draw.text(position, text, font=font, fill=properties['color'])

def draw_gradient_text(draw, position, text, font, properties):
    gradient_colors = properties['effects']['gradient']['colors']
    gradient_direction = properties['effects']['gradient']['direction']

    text_layer = Image.new('RGBA', draw.im.size, (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_layer)
    text_draw.text(position, text, font=font, fill=(255, 255, 255, 255))

    gradient = Image.new('RGBA', draw.im.size, (255, 255, 255, 0))
    gradient_draw = ImageDraw.Draw(gradient)

    if gradient_direction == 'vertical':
        gradient_draw.rectangle([0, 0, gradient.width, gradient.height], fill=gradient_colors[0])
        for i, color in enumerate(gradient_colors[1:], 1):
            y = int(i * gradient.height / len(gradient_colors))
            gradient_draw.rectangle([0, y, gradient.width, gradient.height], fill=color)
    else:  # horizontal
        gradient_draw.rectangle([0, 0, gradient.width, gradient.height], fill=gradient_colors[0])
        for i, color in enumerate(gradient_colors[1:], 1):
            x = int(i * gradient.width / len(gradient_colors))
            gradient_draw.rectangle([x, 0, gradient.width, gradient.height], fill=color)

    gradient_text = Image.composite(gradient, Image.new('RGBA', draw.im.size, (255, 255, 255, 0)), text_layer)
    draw.im.paste(gradient_text, (0, 0), gradient_text)

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
