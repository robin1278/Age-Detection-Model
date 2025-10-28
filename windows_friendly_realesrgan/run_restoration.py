import argparse
import cv2
import glob
import os
import requests
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import torch

def download_model(model_path):
    """Downloads the pre-trained model if it doesn't exist."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.isfile(model_path):
        print("Model not found. Downloading...")
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'

        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()

            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            exit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input image path')
    parser.add_argument('--output', type=str, default='output', help='Output folder')
    parser.add_argument('--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision instead of fp16 (for CPU)')
    args = parser.parse_args()

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.fp32 else 'cpu')

    # Model setup
    model_name = 'RealESRGAN_x4plus'
    model_path = os.path.join('weights', model_name + '.pth')

    # Download model if necessary
    download_model(model_path)

    # Use RRDBNet model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=not args.fp32,
        device=device
    )

    os.makedirs(args.output, exist_ok=True)

    img_path = args.input
    img_name = os.path.basename(img_path)
    print(f'Processing {img_name}...')

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Cannot read image {img_path}")
        return

    try:
        output, _ = upsampler.enhance(img, outscale=args.outscale)
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out-of-memory, try using a smaller tile size.')
        return

    save_path = os.path.join(args.output, f'{os.path.splitext(img_name)[0]}_restored.png')
    cv2.imwrite(save_path, output)
    print(f'Restored image saved to {save_path}')

if __name__ == '__main__':
    main()
