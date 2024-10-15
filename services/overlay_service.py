from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict
import io
import base64
import cairosvg


def convert_svg_to_png(svg_content: str, output_file_path: str):
    """
    Convert SVG content to PNG and save to the specified file path.

    :param svg_content: SVG content as a string
    :param output_file_path: Path to save the output PNG file
    """
    try:
        cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=output_file_path)
        print(f"SVG successfully converted to PNG and saved to {output_file_path}")
    except Exception as e:
        print(f"Error converting SVG to PNG: {e}")


def get_default_font():
    """Return a default system font path."""
    try:
        # Try to use a common system font
        return ImageFont.truetype("Arial.ttf", 12).path
    except IOError:
        # If Arial is not available, use the default bitmap font
        return ImageFont.load_default().path


def calculate_position(background: Image.Image, position: str, element_size: Tuple[int, int]) -> Tuple[int, int]:
    """Calculate the absolute position based on relative position string."""
    bg_width, bg_height = background.size
    elem_width, elem_height = element_size

    positions = position.lower().split()
    x, y = 0, 0

    if "left" in positions:
        x = 0
    elif "center" in positions or "middle" in positions:
        x = (bg_width - elem_width) // 2
    elif "right" in positions:
        x = bg_width - elem_width

    if "top" in positions:
        y = 0
    elif "center" in positions or "middle" in positions:
        y = (bg_height - elem_height) // 2
    elif "bottom" in positions:
        y = bg_height - elem_height

    return x, y


def calculate_size(background: Image.Image, original_size: Tuple[int, int], size: float) -> Tuple[int, int]:
    """Calculate the absolute size based on relative size, maintaining aspect ratio."""
    bg_width, bg_height = background.size
    orig_width, orig_height = original_size
    aspect_ratio = orig_width / orig_height

    new_height = int(bg_height * size)
    new_width = int(new_height * aspect_ratio)

    if new_width > bg_width * size:
        new_width = int(bg_width * size)
        new_height = int(new_width / aspect_ratio)

    return new_width, new_height


def add_image(background: Image.Image, image_path: str, position: str, size: float) -> Image.Image:
    """Add an image to the background at the specified relative position and size, maintaining aspect ratio."""
    try:
        with Image.open(image_path) as img:
            original_size = img.size
            abs_size = calculate_size(background, original_size, size)
            img = img.resize(abs_size, Image.LANCZOS)
            abs_position = calculate_position(background, position, abs_size)
            background.paste(img, abs_position, img if img.mode == 'RGBA' else None)
    except IOError:
        print(f"Warning: Could not open image file {image_path}. Skipping this image.")
    return background


def create_banner(background: Image,
                  additional_images: List[Dict[str, any]]) -> Image.Image:
    """Create a banner with additional images and texts using relative positioning and sizing."""
    try:
        # Add additional images
        for img_data in additional_images:
            background = add_image(background,
                                   img_data['path'],
                                   img_data['position'],
                                   img_data['size'])
        if background:
            background.save("output_banner.png")
            print("Banner created successfully!")
        else:
            print("Failed to create banner.")

        return background
    except IOError:
        print(f"Error: Could not open background image")
        return None


# Example usage
def overlay_images(background_image_base64: str, svg_properties) -> Image.Image:
    background_image_data = base64.b64decode(background_image_base64)
    background_image = Image.open(io.BytesIO(background_image_data))

    # svg_properties.svg_code is a svg code, convert it to png
    svg_code = svg_properties["svg_code"]
    convert_svg_to_png(svg_code, "svg_image1.png")
    position1 = svg_properties["placement"]

    banner = create_banner(
        background=background_image,
        additional_images=[
            {"path": "svg_image1.png", "position": position1, "size": 0.5},
        ]
    )

    if banner:
        banner.save("output_banner.png")
        # Also create the base64 string
        buffered = io.BytesIO()
        background_image.save(buffered, format="PNG")
        combined_image_base64 = base64.b64encode(buffered.getvalue()).decode()
        combined_image_base64 = "data:image/png;base64," + combined_image_base64
        return combined_image_base64
    else:
        return None
        print("Failed to create banner.")
