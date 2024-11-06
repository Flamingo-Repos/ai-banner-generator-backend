import fal_client
import json
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ImageGenerator:
    def __init__(self):
        # Ensure FAL_KEY is set in environment variables
        if not os.getenv('FAL_KEY'):
            raise ValueError("FAL_KEY environment variable is not set")

    def _on_queue_update(self, update):
        """Handle queue updates during image generation"""
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(f"Progress: {log['message']}")

    def generate_images_from_prompts(self, prompts_json: str, image_size: str = "landscape_16_9") -> List[Dict[str, Any]]:
        """
        Generate images from a JSON string containing prompts

        Args:
            prompts_json: JSON string containing prompts
            image_size: Size specification for the generated images

        Returns:
            List of generated image data
        """
        try:
            # Add debug logging for input
            print(f"\nReceived prompts_json:\n{prompts_json}\n")

            prompts_data = json.loads(prompts_json) if isinstance(prompts_json, str) else prompts_json
            generated_images = []

            # Ensure we're handling the prompts array correctly
            prompts = prompts_data.get("prompts", [])
            if not prompts:
                raise ValueError("No prompts found in input data")

            print(f"\nProcessing {len(prompts)} prompts")

            for prompt_data in prompts:
                try:
                    # Extract background prompt, ensuring it exists
                    background_prompt = prompt_data.get("background", "")
                    if not background_prompt or len(background_prompt.strip()) <= 5:
                        print(f"Skipping invalid prompt: {prompt_data}")
                        continue

                    print(f"\nProcessing prompt for Fal:\n{background_prompt}\n")

                    result = fal_client.subscribe(
                        "fal-ai/flux-pro/v1.1",
                        arguments={
                            "prompt": background_prompt,
                            "image_size": image_size,
                            "num_images": 1,
                            "enable_safety_checker": True,
                            "safety_tolerance": "4"
                        },
                        with_logs=True,
                        on_queue_update=self._on_queue_update,
                    )

                    # Add debug logging
                    print(f"\nFal API Response:\n{json.dumps(result, indent=2)}\n")

                    if result and 'images' in result and result['images']:
                        image_data = {
                            "prompt": background_prompt,
                            "images": result.get("images", []),
                            "seed": result.get("seed"),
                        }
                        generated_images.append(image_data)
                        print(f"Successfully generated image with URL: {result['images'][0].get('url', 'No URL found')}")
                    else:
                        print(f"Warning: No valid image generated for prompt: {background_prompt[:100]}...")

                except Exception as e:
                    print(f"Error generating image for prompt: {str(e)}")
                    continue

            if not generated_images:
                raise ValueError("No valid images were generated from any of the prompts")

            return generated_images

        except Exception as e:
            print(f"Error processing prompts: {str(e)}")
            raise
