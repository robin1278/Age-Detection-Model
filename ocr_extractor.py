
import cv2
import json
import numpy as np
import pytesseract
from PIL import Image

def get_all_components_info(json_data, component_label):
    components = []
    for shape in json_data['shapes']:
        if shape['label'] == component_label:
            components.append(shape)
    return components

def enhance_and_ocr(image, component_info, component_label, index):
    if component_info['shape_type'] == 'polygon':
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        points = np.array(component_info['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        x, y, w, h = cv2.boundingRect(points)
        cropped_image = cv2.bitwise_and(image, image, mask=mask)[y:y+h, x:x+w]

    elif component_info['shape_type'] == 'circle':
        center = tuple(np.int_(component_info['points'][0]))
        radius_point = tuple(np.int_(component_info['points'][1]))
        radius = int(np.sqrt((center[0] - radius_point[0])**2 + (center[1] - radius_point[1])**2))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        x, y, w, h = cv2.boundingRect(np.array([[[center[0]-radius, center[1]-radius]], [[center[0]+radius, center[1]+radius]]]))
        cropped_image = cv2.bitwise_and(image, image, mask=mask)[y:y+h, x:x+w]

    else:
        return "Unsupported shape type", 0

    cv2.imwrite(f"cropped_{component_label}_{index}.png", cropped_image)

    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    denoised_image = cv2.medianBlur(gray_image, 3)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_image = clahe.apply(denoised_image)

    scale_factor = 4
    width = int(contrast_image.shape[1] * scale_factor)
    height = int(contrast_image.shape[0] * scale_factor)
    resized_image = cv2.resize(contrast_image, (width, height), interpolation=cv2.INTER_LANCZOS4)

    _, preprocessed_image = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert image only for component15
    if component_label == "component15":
        preprocessed_image = cv2.bitwise_not(preprocessed_image)

    cv2.imwrite(f"preprocessed_{component_label}_{index}.png", preprocessed_image)

    pil_img = Image.fromarray(preprocessed_image)

    # Try different PSM values
    custom_config = r'--oem 3 --psm 7' # Treat the image as a single text line.
    if component_label == "component15":
        custom_config = r'--oem 3 --psm 8' # Treat the image as a single word.


    ocr_data = pytesseract.image_to_data(pil_img, config=custom_config, output_type=pytesseract.Output.DICT)

    text = ""
    total_confidence = 0
    word_count = 0
    for i in range(len(ocr_data['text'])):
        if int(ocr_data['conf'][i]) > 0 and ocr_data['text'][i].strip() != "":
            text += ocr_data['text'][i] + " "
            total_confidence += int(ocr_data['conf'][i])
            word_count += 1

    avg_confidence = total_confidence / word_count if word_count > 0 else 0
    return text.strip(), avg_confidence

def main():
    with open('PCB_AI_TASK_page-0001.json', 'r') as f:
        json_data = json.load(f)

    image = cv2.imread('PCB_AI_TASK_page-0001.jpg')

    component2_infos = get_all_components_info(json_data, 'component2')
    if component2_infos:
        for i, info in enumerate(component2_infos):
            text, confidence = enhance_and_ocr(image, info, 'component2', i)
            print(f"Component 2 (instance {i+1}) OCR Result: '{text}' with average confidence: {confidence:.2f}%")
    else:
        print("Component 2 not found in JSON.")

    component15_infos = get_all_components_info(json_data, 'component15')
    if component15_infos:
        for i, info in enumerate(component15_infos):
            text, confidence = enhance_and_ocr(image, info, 'component15', i)
            print(f"Component 15 (instance {i+1}) OCR Result: '{text}' with average confidence: {confidence:.2f}%")
    else:
        print("Component 15 not found in JSON.")

if __name__ == "__main__":
    main()
