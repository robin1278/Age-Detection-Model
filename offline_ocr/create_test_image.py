import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_degraded_image(text, file_path, size=(500, 200), background_color=(128, 128, 128)):
    """
    Create a synthetic image with text and apply degradation effects.
    """
    # Create a base image with a solid color background
    image = Image.new('RGB', size, color=background_color)

    # Use a simple, bold font as a proxy for stenciled text
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    # Draw the text onto the image
    draw = ImageDraw.Draw(image)
    draw.text((10, 50), text, font=font, fill=(220, 220, 220))

    # Convert to OpenCV format
    cv_image = np.array(image)
    cv_image = cv_image[:, :, ::-1].copy()  # Convert RGB to BGR

    # Add Gaussian noise to simulate a grainy surface
    noise = np.random.normal(0, 25, cv_image.shape).astype('uint8')
    noisy_image = cv2.add(cv_image, noise)

    # Apply a slight blur to simulate faded paint
    blurred_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)

    # Save the final image
    cv2.imwrite(file_path, blurred_image)
    print(f"Created test image at {file_path}")

if __name__ == "__main__":
    create_degraded_image("WARNING 123-ABC", "offline_ocr/test_image.png")
