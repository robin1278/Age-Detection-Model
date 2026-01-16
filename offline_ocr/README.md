# Offline OCR for Industrial Boxes

This project provides a command-line tool to extract stenciled or painted text from images of industrial or military-style boxes. It is designed to work completely offline and is optimized for handling degraded images with faded paint, low contrast, and surface damage.

## Features

-   **Offline OCR:** No internet connection or cloud APIs are required.
-   **Robust Preprocessing:** Includes image preprocessing steps to improve accuracy on noisy and low-quality images.
-   **Structured Output:** Produces a structured JSON output with the extracted text, confidence score, and bounding box coordinates for each detection.

## Setup and Installation

### 1. System Dependencies

This tool relies on the Tesseract OCR engine, which must be installed on your system.

**For Debian/Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

**For macOS (using Homebrew):**
```bash
brew install tesseract
```

**For Windows:**
Download and install the Tesseract installer from the [official Tesseract repository](https://github.com/tesseract-ocr/tessdoc). Make sure to add the Tesseract installation directory to your system's `PATH` environment variable.

### 2. Python Dependencies

The necessary Python libraries are listed in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

## Usage

To run the OCR tool, use the `run_ocr.py` script with the `--image` argument, providing the path to your input image.

```bash
python run_ocr.py --image <path_to_your_image.jpg>
```

### Example

If you have an image named `sample_box.png` in the same directory, you would run:

```bash
python run_ocr.py --image sample_box.png
```

The script will print the structured JSON output to the console.
