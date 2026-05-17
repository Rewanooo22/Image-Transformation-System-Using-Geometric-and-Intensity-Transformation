# Image Transformation System Using Geometric and Intensity Transformations

An interactive image processing system built using **Python** and **Streamlit** that applies various geometric and intensity transformations to images while providing quantitative evaluation metrics and histogram analysis.

---

## Overview

Image processing is widely used in computer vision, medical imaging, multimedia systems, and surveillance applications.

This project demonstrates a complete image transformation system that supports:

* Geometric transformations
* Intensity transformations
* Histogram visualization
* PSNR and SSIM evaluation metrics
* Interactive web interface

The system allows users to upload images, apply transformations in real time, compare outputs, and download processed results.

---

## Features

* Upload and process images interactively
* Apply geometric transformations
* Apply intensity transformations
* Compare original and transformed images
* Display histograms
* Calculate PSNR and SSIM metrics
* Download processed images
* Real-time visualization using Streamlit

---

# Geometric Transformations

These transformations modify the spatial arrangement of image pixels.

## 1. Rotation

Rotate images by a specified angle.

### Formula

[
x' = x \cos\theta - y \sin\theta
]

[
y' = x \sin\theta + y \cos\theta
]

### Effects

* Changes image orientation
* Preserves pixel intensity distribution

---

## 2. Flipping

* Horizontal Flip
* Vertical Flip

### Effects

* Mirrors image content
* Histogram remains unchanged

---

## 3. Resizing

Changes image dimensions using scaling.

### Effects

* Alters image resolution
* May affect quality due to interpolation

---

# Intensity Transformations

These transformations modify pixel intensity values.

---

## 1. Negative Transformation

### Formula

[
s = 255 - r
]

### Effects

* Inverts image colors
* Bright becomes dark and vice versa

---

## 2. Log Transformation

### Formula

[
s = c \cdot \log(1 + r)
]

### Effects

* Enhances dark regions
* Compresses bright regions
* Improves shadow details

---

## 3. Thresholding

### Formula

[
s =
\begin{cases}
255 & r > T \
0 & r \leq T
\end{cases}
]

### Effects

* Converts image into binary form
* Useful for segmentation

---

## 4. Gamma Correction (Power Law)

### Formula

[
s = c \cdot r^\gamma
]

### Effects

* γ < 1 → brighter image
* γ > 1 → darker image
* Non-linear brightness control

---

# Evaluation Metrics

## PSNR (Peak Signal-to-Noise Ratio)

### Formula

[
PSNR = 20 \log_{10}\left(\frac{255}{\sqrt{MSE}}\right)
]

### Interpretation

* High PSNR → better image quality
* Low PSNR → greater transformation effect

---

## SSIM (Structural Similarity Index)

Measures similarity between images based on:

* Structure
* Contrast
* Luminance

### Range

* 0 → completely different
* 1 → identical images

---

# Histogram Analysis

Histograms visualize pixel intensity distributions.

### Observations

* Bright images → histogram shifts right
* Dark images → histogram shifts left
* Thresholding → binary peaks
* Gamma correction → depends on γ value

---

# Tech Stack

* Python
* Streamlit
* NumPy
* OpenCV
* PIL (Pillow)
* Scikit-image
* Matplotlib

---

# Installation

Clone the repository:

```bash id="9kt4wd"
git clone https://github.com/Rewanooo22/Short-movie-clip-generation.git
cd Short-movie-clip-generation
```

Install dependencies:

```bash id="8aw2z4"
pip install -r requirements.txt
```

---

# Run the Application

Start the Streamlit app:

```bash id="0mydja"
streamlit run app.py
```

---

# Project Structure

```text id="gcq9sv"
Image-Transformation-System/
│
├── app.py
├── transformations/
├── metrics/
├── histograms/
├── outputs/
├── images/
├── requirements.txt
└── README.md
```

---

# System Features

* Image upload
* Transformation selection
* Real-time visualization
* Metrics computation
* Histogram comparison
* Download transformed results
* Summary table generation

---

# Results Summary

| Technique             | Purpose            | Observation         |
| --------------------- | ------------------ | ------------------- |
| Linear Transformation | Image inversion    | Bright ↔ Dark       |
| Log Transformation    | Enhance dark areas | Improved details    |
| Thresholding          | Segmentation       | Binary output       |
| Gamma Correction      | Brightness control | Depends on γ        |
| Rotation              | Orientation change | No intensity change |
| Flipping              | Mirror image       | Histogram unchanged |
| Resizing              | Scale image        | Resolution changes  |

---

# Limitations

* No adaptive thresholding
* Limited interpolation methods
* No deep learning models
* Heuristic-based analysis only

---

# Applications

* Educational image processing demonstrations
* Basic computer vision preprocessing
* Medical image enhancement
* Multimedia applications
* Histogram analysis experiments

---
---

# Conclusion

This project successfully demonstrates both geometric and intensity image transformations in an interactive system. The integration of evaluation metrics and histogram analysis provides both qualitative and quantitative insights into image processing techniques.

The system serves as an effective educational and visualization tool for understanding image transformation operations.

---

# License

This project is for educational purposes only.

