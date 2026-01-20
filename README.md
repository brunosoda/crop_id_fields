# Digital CNH Document Cropping Tool

A Python-based computer vision toolkit for detecting, cropping, and processing digital CNH (Brazilian driver's license) documents. This project provides automated document boundary detection, intelligent cropping, and image quality comparison capabilities.

## 📋 Features

- **Intelligent Document Detection**: Multi-pipeline approach to detect document boundaries using thresholding, edge detection, and contour analysis
- **Automated Cropping**: Batch processing of images with proportional coordinate-based cropping
- **Image Quality Comparison**: SSIM (Structural Similarity Index) based image comparison for quality assessment
- **Debugging Tools**: Visualization utilities to inspect detection results
- **Flexible Fallback System**: Multiple detection methods ensure robust document detection across various image conditions

## 🏗️ Project Structure

```
crop app/
├── crop_digital_cnh.py           # Main cropping script with reusable crop_image() function
├── batch_process_images.py       # AWS S3 batch processing pipeline
├── coordenadas.py                # Advanced document detection with multiple pipelines
├── find_red_rectangle.py         # Simple proportional coordinate finder
├── compare_ssim.py               # Image quality comparison tool
├── applications for debug/       # Debugging utilities
│   ├── coordenadas.py            # Document detection (debug version)
│   ├── debug_edges.py            # Visualization tool for bounding boxes
│   └── find_red_rectangle.py     # Coordinate finder (debug version)
├── images/                       # Input images directory
├── masks/                        # Mask images directory
├── temp/                         # Temporary input directory for batch processing
└── cropped images/               # Output directory for cropped images
```

## 🔧 Requirements

- Python 3.6+
- OpenCV (`cv2`)
- NumPy
- scikit-image (for SSIM comparison)

Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install opencv-python numpy scikit-image boto3 requests
```

## 📖 Usage

### 1. Document Detection (`coordenadas.py`)

Detects document boundaries in an image using multiple detection pipelines. Returns JSON with bounding box coordinates and detection method.

```bash
python coordenadas.py <image_path>
```

**Output Format:**
```json
{
  "points": [
    {"x": 100, "y": 50},
    {"x": 500, "y": 50},
    {"x": 500, "y": 700},
    {"x": 100, "y": 700}
  ],
  "bbox": {
    "x": 100,
    "y": 50,
    "width": 400,
    "height": 650
  },
  "method": "threshold_close_largest_non_square_quad"
}
```

**Detection Methods:**
- `threshold_close_largest_non_square_quad`: Primary method using Otsu thresholding and morphological closing
- `canny_largest_non_square_quad`: Fallback using Canny edge detection
- `fallback_page_like_bbox`: Page-like bounding box detection
- `fallback_largest_contour`: Last resort using largest contour

### 2. Batch Cropping (`crop_digital_cnh.py`)

Processes all JPG files in the `temp/` directory and crops them using proportional coordinates optimized for digital CNH documents.

```bash
python crop_digital_cnh.py
```

**Crop Region:**
- Width: 12% to 45.5% of image width
- Height: 18% to 31% of image height

Output files are saved to `cropped/` directory as `digital_cnh_cropped_1.jpg`, `digital_cnh_cropped_2.jpg`, etc.

**Note:** The script also exports a reusable `crop_image(input_path, output_path)` function that can be imported by other scripts.

### 2b. AWS S3 Batch Processing (`batch_process_images.py`)

Processes images from a JSON input file, downloads them, crops them, and uploads to AWS S3.

**Prerequisites:**
- AWS credentials configured (via `~/.aws/credentials` or environment variables)
- `input.json` file in the project root

**Input JSON Format:**
```json
[
  {
    "conference_uuid": "uuid-here",
    "file_url": "https://example.com/image.jpg"
  }
]
```

**Usage:**
```bash
python batch_process_images.py
```

**Configuration:**
Edit the constants at the top of `batch_process_images.py`:
- `BUCKET_NAME`: S3 bucket name
- `PREFIX`: S3 prefix for uploaded files
- `MAX_ROWS`: Maximum number of images to process
- `USE_PRESIGNED_URLS`: Generate presigned URLs for output

**Output:**
- Cropped images uploaded to S3
- `output.json` with results containing `conference_uuid` and `cropped_file_url`

### 3. Image Comparison (`compare_ssim.py`)

Compares an image against a reference image using Structural Similarity Index (SSIM).

```bash
python compare_ssim.py <other_image_path> [--resize] [--color]
```

**Options:**
- `--resize`: Automatically resize images if dimensions differ
- `--color`: Use color SSIM (default: grayscale)

**Example:**
```bash
python compare_ssim.py cropped/digital_cnh_cropped_1.jpg --resize
```

### 4. Simple Coordinate Finder (`find_red_rectangle.py`)

Returns proportional bounding box coordinates (20% to 80% of image dimensions).

```bash
python find_red_rectangle.py [image_path]
```

If no path is provided, the script will prompt for input.

### 6. Debug Visualization (`applications for debug/debug_edges.py`)

Visualizes bounding boxes on images for debugging detection results.

**Usage:**
1. Edit `applications for debug/debug_edges.py` and paste the JSON output from `coordenadas.py` into `RAW_JSON`
2. Update `IMG_PATH` if needed
3. Run:
```bash
python applications for debug/debug_edges.py
```

The output image with the bounding box drawn will be saved to `debug_out.jpg`.

## 🔍 Technical Details

### Document Detection Pipeline

The `coordenadas.py` script implements a sophisticated multi-stage detection system:

1. **Preprocessing**: Gaussian blur for noise reduction
2. **Pipeline 1 - Threshold + Morphological Closing**:
   - Otsu binarization (inverted)
   - Morphological closing to join document regions
   - Contour detection and filtering
3. **Pipeline 2 - Canny Edge Detection** (fallback):
   - Edge detection with dilation/erosion
   - Contour analysis
4. **Pipeline 3 - Page-like BBox** (fallback):
   - Bounding box detection ignoring square shapes
5. **Final Fallback**: Largest contour detection

**Filtering Criteria:**
- Minimum area: 25% of image area (configurable)
- Aspect ratio: Excludes quasi-square shapes (0.90-1.10 ratio)
- Margin: 3% padding around detected regions

### Cropping Parameters

The `crop_digital_cnh.py` script uses fixed proportional coordinates optimized for digital CNH documents:
- Left: 12% of width
- Right: 45.5% of width
- Top: 18% of height
- Bottom: 31% of height

## 📝 Notes

- Ensure the `temp/` directory exists and contains JPG files before running `crop_digital_cnh.py`
- The `cropped/` directory will be created automatically if it doesn't exist
- All scripts handle common edge cases and provide informative error messages
- The detection algorithms are optimized for document-like images with clear boundaries

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

