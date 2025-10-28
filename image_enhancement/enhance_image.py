#!/usr/bin/env python3
import torch
from PIL import Image
import numpy as np
from RealESRGAN.model import RealESRGAN
import argparse
import matplotlib.pyplot as plt
import os
import requests

def download_file(url, local_filename):
    """Downloads a file from a URL to a local file."""
    if not os.path.exists(local_filename):
        print(f"Downloading {url} to {local_filename}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return local_filename

def main():
    parser = argparse.ArgumentParser(description="Enhance an image using Real-ESRGAN.")
    parser.add_argument('image_path', type=str, nargs='?', default='placeholder.jpg', help='Path to the input image. Defaults to placeholder.jpg')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=4)

    weights_dir = 'weights'
    os.makedirs(weights_dir, exist_ok=True)

    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    model_path = os.path.join(weights_dir, "RealESRGAN_x4.pth")
    download_file(model_url, model_path)
    model.load_weights(model_path, download=False)

    image = Image.open(args.image_path).convert('RGB')
    sr_image = model.predict(image)

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    base_name = os.path.basename(args.image_path)
    file_name, file_ext = os.path.splitext(base_name)
    output_path = os.path.join(results_dir, f"{file_name}_enhanced{file_ext}")
    sr_image.save(output_path)
    print(f"Enhanced image saved to {output_path}")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sr_image)
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()
