# Image Restoration with Real-ESRGAN

This project uses the Real-ESRGAN model to restore and deblur images. The script will take a low-quality image as input and produce a high-quality, restored version.

## Setup

1.  **Install Dependencies:**
    Make sure you have Python installed. Then, install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To restore an image, run the `restore_image.py` script from the command line, providing the path to your input image:

```bash
python restore_image.py /path/to/your/image.jpg
```

For example:
```bash
python restore_image.py my_blurry_photo.png
```

## Output

The restored image and a side-by-side comparison image will be saved in the `results/` directory. The script will create this directory if it doesn't exist.
