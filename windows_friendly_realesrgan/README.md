# Windows-Friendly Real-ESRGAN Project

This project is a self-contained, Windows-compatible version of the Real-ESRGAN image restoration tool. It has been specifically designed to avoid common installation issues on Windows by including the necessary library code directly within the project ("vendoring").

## How It Works

Instead of relying on `pip` to install the `basicsr` and `realesrgan` libraries (which can fail on Windows), this project includes the source code for these libraries directly. The main script is configured to use this local code.

The pre-trained model (`RealESRGAN_x4plus.pth`) is **downloaded automatically** on the first run and saved in the `weights` folder. On subsequent runs, the script will use the local copy.

## Setup and Usage

### 1. Install Dependencies

First, install the required Python packages. These are all standard packages that are known to be compatible with Windows.

Open a terminal or command prompt, navigate to this project's directory (`windows_friendly_realesrgan`), and run:

```bash
pip install -r requirements.txt
```
**Note for GPU Users:** If you have an NVIDIA GPU, you can get a significant speedup by installing the CUDA-enabled version of PyTorch. Please follow the official instructions at [pytorch.org](https://pytorch.org/get-started/locally/) before running the command above.

### 2. Run the Restoration

Place the image you want to restore in the main project directory. Then, run the `run_restoration.py` script and provide the path to your image.

For example, if you have an image named `my_blurry_photo.png`, you would run:

```bash
python run_restoration.py my_blurry_photo.png
```

### 3. Check the Output

The restored image will be saved in a new `output` folder by default. For the example above, the output file would be `output/my_blurry_photo_restored.png`.

You can change the output folder by using the `--output` flag:

```bash
python run_restoration.py my_blurry_photo.png --output my_results
```

This will save the result to `my_results/my_blurry_photo_restored.png`.
