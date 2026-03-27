
import cv2
import pytesseract
import json
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract text from components in an image using OCR.')
    parser.add_argument('--image', required=True, help='Path to the input image file.')
    parser.add_argument('--json', required=True, help='Path to the input JSON file with component coordinates.')
    args = parser.parse_args()

    # Load the image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read image at {args.image}")
        return

    # Load the JSON file
    try:
        with open(args.json, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {args.json}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON file at {args.json}")
        return

    # Iterate through the shapes in the JSON file
    for shape in data['shapes']:
        if shape['label'] in ['component2', 'component15']:
            # Get the bounding box coordinates
            points = shape['points']
            # Ensure points are correctly ordered (top-left, bottom-right)
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x1 = int(min(x_coords))
            y1 = int(min(y_coords))
            x2 = int(max(x_coords))
            y2 = int(max(y_coords))

            # Crop the component from the image
            component_image = image[y1:y2, x1:x2]

            # Use pytesseract to extract text from the component
            text = pytesseract.image_to_string(component_image)

            # Print the extracted text
            print(f"Component: {shape['label']} at ({x1},{y1}) to ({x2},{y2})")
            print(f"Extracted Text: {text.strip()}")
            print("-" * 20)

if __name__ == '__main__':
    main()
