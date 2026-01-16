import argparse
import cv2
import pytesseract
import json

def preprocess_image(image_path):
    """
    Load an image and apply preprocessing steps to improve OCR accuracy on degraded images.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None

    # Convert to grayscale to reduce the complexity of the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # A bilateral filter is effective at noise removal while keeping edges sharp.
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Apply Otsu's thresholding. This is a global thresholding method that is
    # effective when the image histogram has two peaks (e.g., text and background).
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary

def format_output(ocr_data):
    """
    Format the Tesseract OCR data into a structured JSON output.
    """
    output_data = []
    for i in range(len(ocr_data['text'])):
        # Filter out empty strings and low-confidence detections
        text = ocr_data['text'][i].strip()
        if text and int(ocr_data['conf'][i]) > 40:
            conf = float(ocr_data['conf'][i])
            x, y, w, h = (
                int(ocr_data['left'][i]),
                int(ocr_data['top'][i]),
                int(ocr_data['width'][i]),
                int(ocr_data['height'][i]),
            )
            bounding_box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

            output_data.append({
                "text": text,
                "confidence": conf,
                "bounding_box": bounding_box
            })

    return json.dumps(output_data, indent=4)


def main():
    """
    Main function to handle command-line arguments and process the image.
    """
    parser = argparse.ArgumentParser(
        description="Extract stenciled or painted text from images of industrial or military-style boxes."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image file."
    )
    args = parser.parse_args()

    preprocessed_image = preprocess_image(args.image)

    if preprocessed_image is not None:
        # Perform OCR using Tesseract
        ocr_data = pytesseract.image_to_data(preprocessed_image, output_type=pytesseract.Output.DICT)

        # Format and print the results as JSON
        json_output = format_output(ocr_data)
        print(json_output)

if __name__ == "__main__":
    main()
