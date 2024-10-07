import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def generate_image_prompt(product_name: str, theme: str, extra_input: str, promotional_offer: str) -> str:
    # Initialize OpenAI client with the API key from environment variable
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Construct the message for GPT-4
    message = f"Create an image prompt for an advertisement with the following details:\n"
    message += f"Product: {product_name}\n"
    message += f"Theme: {theme}\n"
    # message += f"Additional Information: {extra_input}\n"
    message += f"Promotional Offer: {promotional_offer}\n"

    # Call GPT-4 to generate the prompt
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert at creating image prompts for advertisements."},
            {"role": "user", "content": message}
        ]
    )

    return response.choices[0].message.content
