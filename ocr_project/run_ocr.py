
import cv2
import json
import numpy as np
import pytesseract
import os

def preprocess_and_extract_text(image, shape, output_dir):
    label = shape['label']
    points = np.array(shape['points'], dtype=np.int32)

    # For polygon shapes, find the bounding box
    x, y, w, h = cv2.boundingRect(points)

    # --- Handle Rotated Text on PCB ---
    # Check if the polygon is a tall and thin rectangle (likely vertical text)
    is_vertical = h > w * 2
    if is_vertical:
        # Rotate the entire image to make the text horizontal
        center = (image.shape[1] // 2, image.shape[0] // 2)
        # Assuming text is rotated 90 degrees counter-clockwise
        M = cv2.getRotationMatrix2D(center, 90, 1.0)
        rotated_image = cv2.warpAffine(image, M, (image.shape[0], image.shape[1]))
        # Recalculate points for the rotated image
        rotated_points = cv2.transform(np.array([points]), M)[0]
        x, y, w, h = cv2.boundingRect(rotated_points)
        crop = rotated_image[y:y+h, x:x+w]
    else:
        crop = image[y:y+h, x:x+w]

    if crop.size == 0:
        return ""

    # --- Component-Specific Preprocessing ---
    # Black ICs with light text
    if label in ['component1', 'component2', 'component7']:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        _, binary = cv2.threshold(inverted, 190, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.dilate(binary, kernel, iterations=1)
        config = '--psm 7'
    # Blue potentiometers
    elif label in ['component5', 'component13']:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 50, 255]))
        processed = cv2.bitwise_not(mask) # Invert to make text black
        config = '--psm 7 -c tessedit_char_whitelist=0123456789'
    # Green PCB text & others with dark text
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        config = '--psm 6'

    # Save the processed image for debugging
    filename = os.path.join(output_dir, f"{label}_{np.random.randint(1000)}.png")
    cv2.imwrite(filename, processed)

    text = pytesseract.image_to_string(processed, config=config)
    return text.strip()

def main():
    json_path = 'ocr_project/PCB_AI_TASK_page-0001.json'
    image_path = 'ocr_project/PCB_AI_TASK_page-0001.jpg'
    output_dir = 'ocr_project/output'
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Special handling for text on the PCB that isn't in a labeled component
    # This is a manual definition based on visual inspection
    pcb_text_areas = {
        "Accuracy_Text": [[[275, 115], [275, 235], [295, 235], [295, 115]]],
        "LAC_www": [[[250, 480], [250, 680], [270, 680], [270, 480]]],
        "Yellou_Text": [[[280, 750], [280, 830], [300, 830], [300, 750]]],
        "Speed_Text": [[[790, 850], [790, 930], [810, 930], [810, 850]]]
    }

    for label, points in pcb_text_areas.items():
        shape = {'label': label, 'points': points[0]}
        text = preprocess_and_extract_text(image, shape, output_dir)
        print(f"Area: {label}")
        print(f"Extracted Text: {text}")
        print("-" * 20)


    for shape in data['shapes']:
        # Only process labeled components
        if 'component' in shape['label']:
            text = preprocess_and_extract_text(image, shape, output_dir)
            if text:
                print(f"Component: {shape['label']}")
                print(f"Extracted Text: {text}")
                print("-" * 20)

if __name__ == "__main__":
    main()
