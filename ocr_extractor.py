
import cv2
import json
import numpy as np
import easyocr
import math

def get_all_components_info(json_data, component_label):
    components = []
    for shape in json_data['shapes']:
        if shape['label'] == component_label:
            components.append(shape)
    return components

def straighten_and_ocr(image, component_info, component_label, index, reader):
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

    # For component15, detect the angle of the text and rotate the image to make it horizontal
    if component_label == 'component15':
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = cropped_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cropped_image = cv2.warpAffine(cropped_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


    # Perform OCR using easyocr
    results = reader.readtext(cropped_image)

    text = ""
    total_confidence = 0
    if results:
        for (bbox, t, prob) in results:
            text += t + " "
            total_confidence += prob
        avg_confidence = total_confidence / len(results)
    else:
        avg_confidence = 0

    return text.strip(), avg_confidence * 100

def main():
    reader = easyocr.Reader(['en'])
    with open('PCB_AI_TASK_page-0001.json', 'r') as f:
        json_data = json.load(f)

    image = cv2.imread('PCB_AI_TASK_page-0001.jpg')

    component2_infos = get_all_components_info(json_data, 'component2')
    if component2_infos:
        for i, info in enumerate(component2_infos):
            text, confidence = straighten_and_ocr(image, info, 'component2', i, reader)
            print(f"Component 2 (instance {i+1}) OCR Result: '{text}' with average confidence: {confidence:.2f}%")
    else:
        print("Component 2 not found in JSON.")

    component15_infos = get_all_components_info(json_data, 'component15')
    if component15_infos:
        for i, info in enumerate(component15_infos):
            text, confidence = straighten_and_ocr(image, info, 'component15', i, reader)
            print(f"Component 15 (instance {i+1}) OCR Result: '{text}' with average confidence: {confidence:.2f}%")
    else:
        print("Component 15 not found in JSON.")

if __name__ == "__main__":
    main()
