import asyncio
import aiohttp
from services.text_svg_generation_service import generate_text_overlay
import base64
from PIL import Image
import io
import traceback

async def test_text_generation():
    async with aiohttp.ClientSession() as session:
        image_description = "A serene beach scene with palm trees and a sunset"
        text_content = "Tropical Paradise Getaway"
        image_size = (800, 600)

        try:
            text_overlay, properties = await generate_text_overlay(session, image_description, text_content, image_size)

            print("Generated text properties:")
            print(properties)

            # Save the text overlay image
            image_data = base64.b64decode(text_overlay)
            image = Image.open(io.BytesIO(image_data))
            image.save("text_overlay.png")
            print("Text overlay image saved as 'text_overlay.png'")

            # Create a colored background to better visualize the text
            background = Image.new('RGB', image_size, (200, 200, 200))
            background.paste(image, (0, 0), image)
            background.save("text_overlay_with_background.png")
            print("Text overlay with background saved as 'text_overlay_with_background.png'")

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print("Traceback:")
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_text_generation())
