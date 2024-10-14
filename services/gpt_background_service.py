import os
from dotenv import load_dotenv
import aiohttp

# Load environment variables from .env file
load_dotenv()

async def generate_background_prompt(
    session: aiohttp.ClientSession,
    theme: str
) -> str:
    openai_api_key = os.getenv("OPENAI_API_KEY")

    system_message = """You are an AI assistant specialized in creating simple, abstract image prompts based on a given theme. Your task is to generate clear, concise, and visually descriptive prompts that emphasize simplicity and minimalism while incorporating the given theme.

Follow these guidelines:
1. Always use words like "simple", "abstract", "plain", or "minimalist" to set the tone.
2. Start with a basic background, usually a single color that complements the theme.
3. Use the given theme to inform the choice of simple visual elements or symbols.
4. Limit the design to 2-3 key elements to maintain simplicity.
5. Use directional terms (e.g., top, bottom, left, right, center) for precise positioning.
6. Mention any empty or blank spaces explicitly.
7. Describe elements using basic shapes and minimal details when possible.
8. Ensure the overall composition supports the theme.
9. Do not include any text or mention of text in the image prompt.

Example prompt:
"a simple plain background in red, with diwali diyas on the bottom left and right both corners. all other space is blank"

Your prompts should follow this structure and level of simplicity."""

    user_message = f"""Please generate a simple, abstract image prompt based on the following theme:

Theme or purpose of the image: {theme}

Create a concise prompt that incorporates the theme in a minimalist style, suitable for a marketing banner background. Remember, do not include any text elements in the prompt."""

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
        print(result)
        return result['choices'][0]['message']['content']
