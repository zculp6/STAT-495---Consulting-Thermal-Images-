# Thermal Image Cleaning & Crosshair Detection

An automated pipeline for detecting crosshair markers (red, cyan, and green) and cleaning thermal images by removing UI elements, scales, text, and visual artifacts using masking and Navier-Stokes inpainting.

## Features

- **Automatic Crosshair Detection**
  - Red crosshair (selects best single point)
  - Cyan crosshair detection
  - Green crosshair detection
  
- **Image Cleaning**
  - UI and artifact removal using color thresholding (BGR + HSV)
  - Edge detection and morphological filtering
  - Connected component filtering
  - Navier-Stokes image inpainting for smooth reconstruction

- **Batch Processing**
  - Process entire folders of thermal images
  - Individual image processing support
  
- **Automated Outputs**
  - Cleaned thermal images
  - CSV files with detected crosshair coordinates
  - Optional analysis visualizations

## Project Structure

```
project-folder/
│
├── Cuneo_Hall/                 # Input images (your raw thermal images)
├── Cuneo_Hall_cleaned/         # Output folder (auto-created)
│   ├── image1_cleaned.png
│   ├── image1_coordinates.csv
│   ├── image1_analysis.png
│   └── ...
│
├── thermal_cleaning.py         # Main Python script
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.7 or higher

### Install Required Packages

```bash
pip install opencv-python numpy matplotlib pandas
```

Or install from a requirements file:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
pandas>=1.1.0
```

## Usage

### User Configuration

Edit the following variables at the top of `thermal_cleaning.py`:

```python
INPUT_FOLDER = "Cuneo_Hall"           # Folder containing input images
OUTPUT_FOLDER = "Cuneo_Hall_cleaned"  # Output folder (auto-created)
VISUALIZE = True                       # Set to False to skip visualization
```

### Running the Script

1. Place your thermal images in the input folder (e.g., `Cuneo_Hall/`)
2. Configure the input/output folder names in the script
3. Run the script:

```bash
python thermal_cleaning.py
```

### Output Files

For each processed image, the script generates:

- `{image_name}_cleaned.png` - Cleaned thermal image with artifacts removed
- `{image_name}_coordinates.csv` - Detected crosshair coordinates
- `{image_name}_analysis.png` - Visualization of detection (if `VISUALIZE=True`)

### CSV Format

The output CSV contains the following columns:

```csv
crosshair_type,x,y
red,245,189
cyan,512,384
green,128,256
```

## How It Works

1. **Color Detection**: Identifies crosshair markers using BGR and HSV color space thresholds
2. **Artifact Masking**: Detects UI elements, text, and visual artifacts through edge detection and morphological operations
3. **Inpainting**: Uses Navier-Stokes inpainting algorithm to reconstruct masked regions seamlessly
4. **Coordinate Extraction**: Calculates centroids of detected crosshair regions
5. **Batch Processing**: Iterates through all images in the input folder

## Example

**Before:**
- Raw thermal image with UI overlays, temperature scales, and crosshair markers

**After:**
- Clean thermal data only
- Crosshair coordinates saved to CSV
- Optional analysis visualization showing detected markers

## Troubleshooting

### No crosshairs detected
- Verify crosshair colors match expected RGB/HSV ranges
- Check image quality and resolution
- Adjust detection thresholds in the script

### Poor inpainting results
- Increase inpainting radius parameter
- Check mask quality in visualization output
- Ensure artifacts are properly detected

### Import errors
- Verify all packages are installed: `pip list`
- Try reinstalling opencv-python: `pip install --upgrade opencv-python`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgments

- OpenCV library for image processing capabilities
- Navier-Stokes inpainting algorithm for seamless artifact removal

## Contact

For questions or issues, please open an issue on GitHub or contact the project maintainer.

---

**Course:** STAT-495 Consulting Project  
**Topic:** Thermal Image Analysis and Processing
