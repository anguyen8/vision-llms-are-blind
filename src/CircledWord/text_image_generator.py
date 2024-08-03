import matplotlib.pyplot as plt
import matplotlib.patches as patches
import freetype
from PIL import Image, ImageFilter
import numpy as np
import uuid
import os
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_font(font_path, char_size=36 * 64, pixel_height=96):
    face = freetype.Face(font_path)
    face.set_char_size(char_size)  # Set the font size in 1/64th points
    face.set_pixel_sizes(0, pixel_height)  # Set pixel sizes, specifying height
    return face


def draw_glyph(ax, face, char, position):
    face.load_char(char, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
    bitmap = face.glyph.bitmap
    top = face.glyph.bitmap_top
    left = face.glyph.bitmap_left

    x = position[0] + left
    y = 200 - (position[1] - top)

    buffer_array = np.array(bitmap.buffer, dtype=np.uint8)
    if bitmap.width > 0 and bitmap.rows > 0:
        buffer_reshaped = buffer_array.reshape(bitmap.rows, bitmap.width)
        buffer_inverted = 255 - buffer_reshaped
        ax.imshow(
            buffer_inverted,
            cmap="gray",
            interpolation="bilinear",
            extent=(x, x + bitmap.width, y - bitmap.rows, y),
        )

    center_x = x + bitmap.width / 2
    center_y = y - bitmap.rows / 2
    return position[0] + face.glyph.advance.x // 64, position[1], center_x, center_y


def draw_ellipse_on_char(ax, centers, index, thickness, radius_x, radius_y):
    if 0 <= index < len(centers):
        center_x, center_y = centers[index]
        ellipse = patches.Ellipse(
            (center_x, center_y),
            width=2 * radius_x,  # Width of the ellipse (total width across the x-axis)
            height=2
            * radius_y,  # Height of the ellipse (total height across the y-axis)
            fill=False,
            edgecolor="red",
            linewidth=thickness,
        )
        ax.add_patch(ellipse)


def create_text_image(
    text,
    font_path,
    circle_index,
    thickness,
    scale_factor=1.0,  # Default scale factor is 1.0 (no scaling)
    padding=50,
    x_offset=0,
    y_offset=0,
    canvas_width=10,  # in inches
    canvas_height=2,  # in inches
    final_width=256,  # in pixels
    final_height=256,  # in pixels
    output_folder="output",
):
    try:
        face = load_font(font_path)
        total_width = 0
        max_height = 0
        glyph_sizes = []
        for char in text:
            face.load_char(
                char, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL
            )
            glyph_width = face.glyph.advance.x // 64
            glyph_height = (
                face.glyph.bitmap_top + face.glyph.bitmap.rows - face.glyph.bitmap_top
            )
            glyph_sizes.append((glyph_width, glyph_height))
            total_width += glyph_width
            if glyph_height > max_height:
                max_height = glyph_height

        fig, ax = plt.subplots(figsize=(canvas_width, canvas_height), dpi=1200)
        ax.set_frame_on(False)
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_xlim(0, total_width + 2 * padding)
        ax.set_ylim(0, 200)

        vertical_center = 100 + (max_height // 2)
        x, y = (padding, vertical_center)
        centers = []
        for i, (char, (glyph_width, glyph_height)) in enumerate(zip(text, glyph_sizes)):
            x, y, center_x, center_y = draw_glyph(ax, face, char, (x, y))
            centers.append((center_x, center_y))
            if i == circle_index:
                radius_x = (glyph_width / 2) * scale_factor
                radius_y = (glyph_height / 2) * scale_factor
                draw_ellipse_on_char(ax, centers, i, thickness, radius_x, radius_y)

        ax.set_xticks([])
        ax.set_yticks([])

        # Generate a unique high-resolution filename
        high_res_image_path = f"./tmp/ultra_high_res_text_{uuid.uuid4()}.png"
        fig.savefig(high_res_image_path, dpi=1200, bbox_inches="tight", pad_inches=0)

        # # close the figure
        plt.close(fig)

        # Open the high-resolution image, apply blur
        image = Image.open(high_res_image_path)
        blurred_image = image.copy()

        # Calculate the scaling factor while maintaining the aspect ratio
        original_width, original_height = blurred_image.size
        aspect_ratio = original_width / original_height
        if final_width / final_height > aspect_ratio:
            new_height = final_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = final_width
            new_height = int(new_width / aspect_ratio)

        resized_image = blurred_image.resize((new_width, new_height))

        # Create a new image with white background
        final_image = Image.new("RGB", (final_width, final_height), "white")
        paste_x = (final_width - new_width) // 2 + x_offset
        paste_y = (final_height - new_height) // 2 + y_offset
        final_image.paste(resized_image, (paste_x, paste_y))

        # Generate a unique filename for the final image
        unique_id = uuid.uuid4()
        final_image_path = os.path.join(output_folder, f"text_image_{unique_id}.png")
        final_image.save(final_image_path)

        # Clean up the high-resolution image
        # os.remove(high_res_image_path)

        return final_image_path
    except Exception as e:
        logging.error(f"Error creating text image: {e}")
        return None
