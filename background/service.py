from dotenv import load_dotenv
import os
import json
import time
from openai import OpenAI
from typing_extensions import override
from openai import AssistantEventHandler
from .image_generator import ImageGenerator
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI()

class FileReaderEventHandler(AssistantEventHandler):
    def __init__(self):
        super().__init__()
        self.client = OpenAI()

    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot) -> None:
        print(delta.value, end="", flush=True)

    @override
    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)

class PromptCollectorEventHandler(FileReaderEventHandler):
    def __init__(self):
        super().__init__()
        self.current_response = ""
        self.json_started = False
        self.background_json = {"prompts": []}
        self.text_specs_json = {"prompts": []}
        self.image_prompts = []
        self.text_specs = []
        self.client = OpenAI()

    @override
    def on_text_delta(self, delta, snapshot) -> None:
        # Only start collecting when we see the opening brace
        if "{" in delta.value and not self.json_started:
            self.json_started = True
            self.current_response = "{"
        elif self.json_started:
            self.current_response += delta.value
        print(delta.value, end="", flush=True)

    @override
    def on_message_done(self, message) -> None:
        if self.current_response:
            try:
                # Clean up the JSON string
                json_str = self.current_response.strip()

                # Find the last closing brace
                last_brace_index = json_str.rindex('}')
                json_str = json_str[:last_brace_index + 1]

                print(f"\nAttempting to parse JSON...")

                # Parse the response as JSON
                response_data = json.loads(json_str)

                # Split the JSON into background and text specifications
                self.split_json_response(response_data)

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Problematic JSON string: {self.current_response[:200]}...")
            except Exception as e:
                print(f"Unexpected error: {str(e)}")

            self.current_response = ""
            self.json_started = False

    def split_json_response(self, data):
        """Split the JSON response into background and text specifications"""
        try:
            # Clear existing data
            self.background_json["prompts"] = []
            self.text_specs_json["prompts"] = []
            self.image_prompts = []
            self.text_specs = []

            for prompt in data.get("prompts", []):
                if "background" in prompt:
                    # Store the complete background object as is
                    self.background_json["prompts"].append(prompt["background"])
                    # Add to image_prompts without formatting
                    self.image_prompts.append(prompt["background"])

                if "text_specifications" in prompt:
                    self.text_specs_json["prompts"].append(prompt["text_specifications"])
                    self.text_specs = self.text_specs_json["prompts"]

            print(f"\nSplit JSON into {len(self.background_json['prompts'])} backgrounds "
                  f"and {len(self.text_specs_json['prompts'])} text specifications")

            # Save to files for debugging
            with open('backgrounds.json', 'w') as f:
                json.dump(self.background_json, f, indent=2)

            with open('text_specs.json', 'w') as f:
                json.dump(self.text_specs_json, f, indent=2)

            # Debug output
            if self.background_json["prompts"]:
                print("\nBackground JSON sample:")
                print(json.dumps(self.background_json["prompts"][0], indent=2)[:200])

            if self.text_specs_json["prompts"]:
                print("\nText Specs JSON sample:")
                print(json.dumps(self.text_specs_json["prompts"][0], indent=2)[:200])

        except Exception as e:
            print(f"Error splitting JSON: {str(e)}")
            raise

    def _format_background_prompt(self, background: Dict[str, Any]) -> Dict[str, Any]:
        """Return the background object as is"""
        return background

def _extract_text_specs(text_specs_json):
    """Extract text specifications from JSON format"""
    try:
        if not text_specs_json:
            return {}

        return {
            "content": text_specs_json.get("content", {}),
            "typography": text_specs_json.get("typography", {}),
            "colors": text_specs_json.get("colors", {}),
            "layout": text_specs_json.get("layout", {})
        }
    except Exception as e:
        print(f"Error extracting text specifications: {e}")
        return {
            "content": {},
            "typography": {},
            "colors": {},
            "layout": {}
        }

def _format_prompt_to_paragraph(prompt: Dict[str, Any]) -> str:
    """Convert a structured prompt into a natural paragraph"""
    try:
        style = prompt['style']
        comp = prompt['composition']
        tech = prompt['technical']

        paragraph = (
            f"{prompt['main_premise']} "
            f"The composition features {comp['primary_negative_space']} negative space, with "
            f"{comp['element_placement']}. {comp['depth_arrangement']}. "
            f"The design incorporates {style['colors']['primary']}, {style['colors']['secondary']}, "
            f"and {style['colors']['accent']} colors against a {style['colors']['background']} background. "
            f"The image has a {style['texture']} with {style['lighting']}, creating a {style['mood']} atmosphere. "
            f"{tech['elements']} are arranged using a {tech['grid']}, with {tech['margins']}."
        )

        return paragraph
    except KeyError as e:
        print(f"Error formatting prompt: Missing key {e}")
        return str(prompt)

def generate_banner(guidelines_file_path, company_context, event_context):
    """Generate banner images based on guidelines and context"""
    try:
        # Create assistant with modified instructions to enforce JSON structure
        assistant = client.beta.assistants.create(
            name="Image Prompt Generator",
            instructions="""You are a brand-focused image prompt generator. Your output must be ONLY valid JSON with no additional text, following this exact structure:

{
    "prompts": [
        {
            "background": {
                "main_premise": "string",
                "composition": {
                    "primary_negative_space": "string",
                    "element_placement": "string",
                    "depth_arrangement": "string",
                    "transitions": "string"
                },
                "style": {
                    "colors": {
                        "primary": "string (hex)",
                        "secondary": "string (hex)",
                        "accent": "string (hex)",
                        "background": "string (hex)"
                    },
                    "texture": "string",
                    "lighting": "string",
                    "mood": "string"
                },
                "technical": {
                    "resolution": "string",
                    "elements": "string",
                    "margins": "string",
                    "grid": "string"
                }
            },
            "text_specifications": {
                "content": {
                    "headline": "string",
                    "subheading": "string",
                    "cta": "string"
                },
                "typography": {
                    "primary": "string",
                    "secondary": "string",
                    "cta": "string"
                },
                "colors": {
                    "primary_text": "string (hex)",
                    "secondary_text": "string (hex)",
                    "cta_text": "string (hex)"
                },
                "layout": {
                    "headline_position": "string",
                    "subheading_position": "string",
                    "cta_position": "string"
                }
            }
        }
    ]
}

Important: Output ONLY the JSON structure above with no additional text or explanations.


1. **Brand Guidelines Analysis**:
   First, thoroughly analyze the provided brand guidelines document and create a structured summary of:
   - Brand Colors (primary, secondary, accent with exact hex codes)
   - Typography (font families, weights, sizes)
   - Visual Style Elements
   - Layout Principles
   - Design Do's and Don'ts

2. **Guidelines Application**:
   Generate prompts that strictly adhere to the extracted brand guidelines:
   - Use ONLY the colors specified in the guidelines
   - Follow typography rules exactly
   - Maintain brand-specific visual language
   - Respect layout principles

2. **Synthesize Information**:
   - Combine brand and event elements into a cohesive visual concept with minimum 40% negative space.

3. **Create Image Prompts**:
   Each prompt must include these essential components and phrases:

   A. COMPOSITION STRUCTURE:
   - Specify primary negative space location (minimum 40% of total area)
   - Define element placement relative to negative space
   - Detail depth arrangement with clear layering instructions
   - Ensure smooth transitions between elements and empty spaces

   B. REQUIRED NEGATIVE SPACE PHRASES:
   Include at least two of these phrases in each prompt:
   - "with large empty areas in [location]"
   - "with minimal elements"
   - "with clear space in the [center/left/right]"
   - "minimalist and creative composition"
   - "simple design with breathing room"
   - "suitable for text overlay"
   - "designed for content placement"
   - "with ample negative space"

   C. BACKGROUND TYPE SPECIFICATION:
   Use at least one of these terms to ensure text compatibility:
   - "corporate background"
   - "abstract background"
   - "presentation background"
   - "professional backdrop"
   - "communication-focused background"

   D. VISUAL ELEMENTS:
   - Define foreground, midground, and background elements
   - Specify exact hex colors for all elements
   - Include depth and lighting instructions
   - Maintain clear hierarchy between elements

   E. TECHNICAL SPECIFICATIONS:
   - Optimize for 2100:600 aspect ratio
   - Specify focus areas and element clustering
   - Define texture and pattern density
   - Include quality parameters for resolution and detail

""",  # Add your full instructions here
            model="gpt-4o",
            tools=[{"type": "file_search"}]
        )

        # Upload guidelines file
        with open(guidelines_file_path, "rb") as file:
            uploaded_file = client.files.create(file=file, purpose="assistants")

        # Create thread for the entire conversation
        thread = client.beta.threads.create(
            messages=[{
                "role": "user",
                "content": """Analyze the brand guidelines document and provide a structured summary with:
                1. COLORS (hex codes, usage rules)
                2. TYPOGRAPHY (fonts, weights, sizes)
                3. VISUAL ELEMENTS (patterns, textures, icons)
                4. LAYOUT PRINCIPLES (spacing, alignment, composition)""",
                "attachments": [{"file_id": uploaded_file.id, "tools": [{"type": "file_search"}]}]
            }]
        )

        # Run guidelines analysis
        print("\n=== Analyzing Brand Guidelines ===\n")
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            event_handler=FileReaderEventHandler()
        ) as stream:
            stream.until_done()

        # Generate prompts using the same thread
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"""Using the brand guidelines analysis above, generate four unique banner background prompts for:
            Event Context: {event_context}
            Company Context: {company_context}

            Follow the brand guidelines strictly.
            Output must be in the specified JSON format with both background and text specifications for each prompt.
            Each prompt must include all required fields as specified in the JSON structure."""
        )

        # Stream prompt generation
        prompt_handler = PromptCollectorEventHandler()
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            event_handler=prompt_handler
        ) as stream:
            stream.until_done()

        # After collecting prompts, convert them to paragraphs
        image_prompts_paragraphs = []
        for prompt in prompt_handler.image_prompts:
            paragraph = _format_prompt_to_paragraph(prompt)
            image_prompts_paragraphs.append(paragraph)

        # Generate images using paragraphs
        image_generator = ImageGenerator()
        generated_images = []

        for paragraph in image_prompts_paragraphs:
            prompt_data = {
                "prompts": [{"background": paragraph}]
            }
            result = image_generator.generate_images_from_prompts(prompt_data)
            if result:
                generated_images.extend(result)

        # Combine results as before
        complete_banners = []
        for i, image_data in enumerate(generated_images):
            if i < len(prompt_handler.image_prompts):
                banner_data = {
                    "image": {
                        "images": image_data["images"],
                        "seed": image_data.get("seed")
                    },
                    "background_prompt": prompt_handler.image_prompts[i],
                    "text_specifications": _extract_text_specs(prompt_handler.text_specs[i])
                }
                complete_banners.append(banner_data)

        return complete_banners

    except Exception as e:
        print(f"Error in generate_banner: {str(e)}")
        raise

if __name__ == "__main__":
    guidelines_file = "path/to/your/guidelines.pdf"
    company_context = "a free tool that shows how frequently a search term is entered into Google's search engine"
    event_context = "AI agent competition"

    generated_banners = generate_banner(guidelines_file, company_context, event_context)

    for banner in generated_banners:
        print(f"\nPrompt: {banner['background_prompt']}")
        print(f"Image URLs: {[img['url'] for img in banner['image']['images']]}")
