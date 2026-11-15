# Streamlit Image Transformation App

An interactive Streamlit web app that applies classic geometric and intensity transformations to an uploaded image, then shows the original and transformed images side by side with PSNR/SSIM metrics, grayscale histograms and a results summary table. A second page has an experimental "medical image analysis" demo that flags regions of high local variance with a heuristic. The CNN defined in the code is never trained or used.

Built as a group coursework project for an Image Processing course at Alamein International University.

## Key features

**Image Enhancement Toolkit (Home page)**

- Upload a `jpg`, `jpeg`, `png`, `bmp` or `gif` image.
- Intensity transformations (each converts the image to grayscale first):
  - **Linear Transform (negative):** `s = 255 - r`
  - **Log Transform:** `s = (255 / log(1 + max(r))) * log(1 + r)`
  - **Threshold Enhancement:** fixed threshold `T = 125` (`s = 255` if `r > T`, else `0`)
  - **Power Law (gamma):** `s = 255 * (r / 255)^gamma`, with a gamma slider from 0.1 to 2.5 (default 0.5)
- Geometric transformations (applied in colour with Pillow):
  - **Rotate:** angle slider from 0 to 360 degrees (default 90), with the canvas expanded to fit
  - **Flip horizontal / flip vertical**
  - **Resize:** scale-factor slider from 0.1 to 2.0 (default 0.5)
- **All Enhancements** mode runs all eight techniques at once (rotation fixed at 90 degrees, resize fixed at 0.5x, gamma from a slider).
- Each result shows:
  - **PSNR**, computed on grayscale versions as `20 * log10(255 / sqrt(MSE))`, reported as 100 when the images are identical
  - **SSIM**, from `skimage.metrics.structural_similarity` with `data_range=255`
  - Original and transformed 256-bin grayscale histograms, drawn with `st.line_chart`
  - A results summary table (technique, purpose, before, after, observation)
- Before computing metrics, the transformed image is resized back to the original size if needed (after rotation or resizing).
- Download buttons for each output, and an option to download every output as a ZIP of PNG files.

**Medical Image Analysis page (experimental demo)**

- Upload an X-ray, MRI or CT image (`jpg`, `jpeg`, `png`, `bmp`, `tif`, `tiff`; the uploader also lists `dcm`).
- The image is converted to grayscale and resized to 224x224.
- A local-variance map (15x15 window) is rendered as a JET-colour heatmap. Fixed thresholds on intensity statistics and heatmap zone ratios then label the image "Anomaly Detected" or "Normal".
- The page shows a heuristic "confidence" value, a score, mean/std intensity and anomaly-zone percentages.
- It also shows the processed image, the heatmap and an adjustable-opacity overlay, and each can be downloaded as a PNG.
- The page displays a warning that it is **not** a diagnostic tool.

**About page:** a short description of the toolkit.

## Tech stack

| Area | Library |
|---|---|
| Web UI | Streamlit |
| Image I/O and geometric transforms | Pillow |
| Array maths, histograms, PSNR | NumPy |
| Filtering, colour maps, PNG encoding (medical page) | OpenCV (`opencv-python`) |
| SSIM | scikit-image |
| Optional CNN definition (not used at runtime, see below) | TensorFlow / Keras |

## Project structure

```text
.
├── Image Processing Project/
│   ├── app.py                  # Streamlit app: transformations, metrics, histograms, pages
│   ├── medical_analysis.py     # MedicalImageAnalyzer: heuristic analysis + unused CNN definition
│   ├── requirements.txt        # Python dependencies (scikit-image is missing, see below)
│   ├── .vscode/settings.json   # Editor settings
│   ├── __pycache__/            # Committed Python bytecode
│   └── env/                    # Committed local Windows virtualenv (Python 3.9); not project code, ignore it
├── ImageTransformationSystem (1).pptx   # Project presentation
├── image report.docx2.docx              # Project report
├── ~$age report.docx2.docx              # Leftover Office lock file
└── README.md
```

## Getting started

### Prerequisites

- Python 3 with `pip`. The pinned `numpy<2.0.0` and `opencv-python<4.9.0` need a Python version that has wheels for them.

### Installation

```bash
git clone https://github.com/Rewanooo22/streamlit-image-transformation-app.git
cd streamlit-image-transformation-app/"Image Processing Project"

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt scikit-image
```

`app.py` imports `skimage.metrics.structural_similarity`, but `scikit-image` is not listed in `requirements.txt`, so the command above installs it explicitly. `requirements.txt` also lists `tensorflow`. The app never imports it in the code path it runs (see Limitations), so you can remove that line if the install is too heavy or fails.

Create a fresh virtual environment as shown above. Do not use the committed `env/` folder: it is a Windows virtualenv tied to the original author's machine.

### Running

From the repository root:

```bash
cd "Image Processing Project"
streamlit run app.py
```

Streamlit prints a local URL (by default `http://localhost:8501`). Use the sidebar **Navigation** menu to switch between **Home**, **Medical Image Analysis** and **About**.

## Project status and limitations

This is coursework, built as an educational demo.

- **The medical analysis is heuristic only.** `MedicalImageAnalyzer.build_model()` defines a four-block Keras CNN, but the repository has no training code and no saved weights, and nothing ever sets `is_trained` to `True`. The app also resets `is_trained` to `False` before every analysis. Every result therefore comes from `_heuristic_analysis()`, which uses hand-set thresholds on intensity statistics and local variance. Its "confidence" value is a heuristic score, not a calibrated probability, and the method has not been validated on any dataset. Do not use it for diagnosis.
- **TensorFlow is effectively unused.** It is imported lazily only on the CNN path, which never runs.
- **Per-image download buttons** send `PIL.Image.tobytes()` (raw pixel data) with a `.png` file name, so those files are not valid PNGs. The "Download All as ZIP" option saves real PNG files.
- **PSNR** is computed on `uint8` arrays without casting to a wider type, so the squared-difference step can overflow and the reported values may be inaccurate.
- The threshold is fixed at 125 (no adaptive or Otsu thresholding). Resizing uses Pillow's default resampling. Intensity transforms always output grayscale.
- The medical uploader accepts `.dcm`, but images are opened with Pillow, which does not read DICOM files.
- The repository commits a virtualenv (`env/`), `__pycache__/` and an Office lock file, and has no `.gitignore`.

## Future improvements (ideas)

- Add `scikit-image` to `requirements.txt`, and remove `tensorflow` or make it an optional extra.
- Encode the per-image downloads as PNG (for example with `io.BytesIO` and `Image.save`).
- Cast to `float` before computing MSE for PSNR.
- Add adaptive or Otsu thresholding, and selectable interpolation for rotation and resizing.
- Remove `env/`, `__pycache__/` and the lock file from version control and add a `.gitignore`.

## Context

Group project for an Image Processing course at Alamein International University. The project report (`image report.docx2.docx`) and presentation (`ImageTransformationSystem (1).pptx`) are included in the repository.
