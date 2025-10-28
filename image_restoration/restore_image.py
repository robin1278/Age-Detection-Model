import argparse
import cv2
import os
import torch
import tempfile
import shutil
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
import matplotlib.pyplot as plt

def main():
    """
    Restores a single image using a pre-trained Real-ESRGAN model.
    """
    parser = argparse.ArgumentParser(description='Restores a single image using Real-ESRGAN.')
    parser.add_argument('input_image', type=str, help='Path to the input image.')
    args = parser.parse_args()

    # --- Set up default arguments ---
    model_name = 'RealESRGAN_x4plus'
    output_dir = 'results'
    outscale = 4
    tile = 0
    tile_pad = 10
    pre_pad = 0
    fp32 = not torch.cuda.is_available() # Use fp16 for CUDA, fp32 for CPU
    gpu_id = None

    # --- Use a temporary directory for the model ---
    temp_model_dir = tempfile.mkdtemp()

    try:
        # --- Determine model and download to a temporary directory if necessary ---
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']

        model_path = os.path.join(temp_model_dir, model_name + '.pth')
        if not os.path.isfile(model_path):
            for url in file_url:
                model_path = load_file_from_url(
                    url=url, model_dir=temp_model_dir, progress=True, file_name=None)

        # --- Set up the restorer ---
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=not fp32,
            gpu_id=gpu_id)

        # --- Create output directory ---
        os.makedirs(output_dir, exist_ok=True)

        # --- Read and process the image ---
        image_path = args.input_image
        img_name = os.path.basename(image_path)
        print(f'Processing {img_name}...')
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return

        output, _ = upsampler.enhance(img, outscale=outscale)

    except RuntimeError as error:
        print('Error:', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        return
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_model_dir)


    # --- Save the restored image ---
    save_path = os.path.join(output_dir, f'{os.path.splitext(img_name)[0]}_restored.png')
    cv2.imwrite(save_path, output)
    print(f'Restored image saved to {save_path}')

    # --- Display original and restored images ---
    original_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    restored_img_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(original_img_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(restored_img_rgb)
    axes[1].set_title('Restored Image')
    axes[1].axis('off')

    plt.tight_layout()
    # Save the figure instead of showing it directly to be compatible with the environment
    comparison_save_path = os.path.join(output_dir, f'{os.path.splitext(img_name)[0]}_comparison.png')
    plt.savefig(comparison_save_path)
    print(f'Comparison image saved to {comparison_save_path}')


if __name__ == '__main__':
    main()
