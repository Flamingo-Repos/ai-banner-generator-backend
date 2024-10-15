import openai
import json
import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Ensure the API key is set
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def generate_text_svg_code(session, image_description, text_content):
    if not openai.api_key:
        raise ValueError("OpenAI API key is not set. Please check your .env file.")

    prompt = f"""
    Generate a production-grade SVG code for a banner ad with the Text Content: "{text_content}"
    Requirements:
    1. Choose appropriate size, color, and font for the text to ensure readability and aesthetic appeal.
    Guidelines:
    2. The SVG should only contain the text with a transparent background.
    3. Make sure entire text is visible and not cropped.
    4. Use multi line text wherever it make sense. 
    
    
    Also, Determine the optimal placement based on the description of background image: {image_description}
    The placement should be one of the following: center, center top, center bottom, left, right, left top, left bottom, right top, right bottom

    Respond strictly in the following JSON format:
    {{
        "placement": "chosen_placement",
        "svg_code": "<svg>...full SVG code here...</svg>"
    }}

    Important Note: The response should contain only the JSON string. No additional text or explanations should be included, Make sure the string starts exactly with the opening curly brace. Ensure the JSON isn't wrapped in any extra characters, like backticks or quotes.
    
    Sample SVG Code for "Buy 1 Get 1" text:
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200">
      <defs>
        <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" style="stop-color:#ff6b6b;stop-opacity:1" />
          <stop offset="100%" style="stop-color:#4ecdc4;stop-opacity:1" />
        </linearGradient>
      </defs>
      
      <text x="200" y="70" font-family="Arial, sans-serif" font-size="48" font-weight="bold" text-anchor="middle" fill="url(#grad1)">
        BUY 1
      </text>
      
      <text x="200" y="130" font-family="Arial, sans-serif" font-size="48" font-weight="bold" text-anchor="middle" fill="url(#grad1)">
        GET 1
      </text>
      
      <text x="200" y="180" font-family="Arial, sans-serif" font-size="24" font-style="italic" text-anchor="middle" fill="#333">
        Limited Time Offer!
      </text>
    </svg>
    
    """

    try:
        response = await session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openai.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o",
                # "type": "json_object",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response_json = await response.json()
        logger.debug(f"API Response: {response_json}")

        if 'choices' not in response_json or len(response_json['choices']) == 0:
            raise ValueError("Unexpected API response format")

        # fetch svg code and placement
        content = response_json['choices'][0]['message']['content']
        content = content.strip().lstrip('```json').rstrip('```').strip()
        properties = json.loads(content)

        return properties

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Content that failed to decode: {content}")
        raise
    except Exception as e:
        logger.error(f"Error in generate_text_properties: {str(e)}")
        raise
