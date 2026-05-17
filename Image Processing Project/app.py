# -*- coding: utf-8 -*-
"""
Image Enhancement Toolkit
Developed by students at Alamein International University
"""

from skimage.metrics import structural_similarity as ssim
import streamlit as st
from PIL import Image
import numpy as np
import zipfile
import os

try:
    import cv2
except ImportError as e:
    if "numpy.core.multiarray" in str(e):
        st.error(
            "NumPy/OpenCV compatibility issue detected. "
            "Please run: pip install --upgrade --force-reinstall numpy opencv-python"
        )
        raise ImportError(
            "NumPy/OpenCV compatibility issue. Please reinstall: "
            "pip install --upgrade --force-reinstall numpy opencv-python"
        ) from e
    raise

from medical_analysis import MedicalImageAnalyzer

st.set_page_config(page_title="Image Enhancer", layout="centered")


def rotate_image(image, angle=90):
    return image.rotate(angle, expand=True)


def flip_horizontal(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def flip_vertical(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)


def resize_image(image, scale=0.5):
    width, height = image.size
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size)


def linear_transform_enhancement(image):
    np_image = np.array(image.convert('L'))
    enhanced_image = 255 - np_image
    return Image.fromarray(enhanced_image)


def log_transform_enhancement(image):
    np_image = np.array(image.convert('L')).astype(np.float32)
    max_val = np.max(np_image) or 1
    log_image = np.log1p(np_image)
    enhanced_image = (255 / np.log1p(max_val)) * log_image
    enhanced_image = np.clip(enhanced_image, 0, 255)
    return Image.fromarray(enhanced_image.astype(np.uint8))


def threshold_enhancement(image):
    np_image = np.array(image.convert('L'))
    threshold_value = 125
    enhanced_image = np.where(np_image > threshold_value, 255, 0)
    return Image.fromarray(enhanced_image.astype(np.uint8))


def power_law_transform(image, gamma=0.5):
    np_image = np.array(image.convert('L')).astype(np.float32) / 255.0
    transformed = np.power(np_image, gamma)
    output = np.uint8(transformed * 255)
    return Image.fromarray(output)


def prepare_for_metrics(original, enhanced):
    original_gray = original.convert('L')
    enhanced_gray = enhanced.convert('L')

    if original_gray.size != enhanced_gray.size:
        enhanced_gray = enhanced_gray.resize(original_gray.size)

    original_array = np.array(original_gray)
    enhanced_array = np.array(enhanced_gray)

    return original_array, enhanced_array


def calculate_psnr(original, enhanced):
    original_array, enhanced_array = prepare_for_metrics(original, enhanced)
    mse = np.mean((original_array - enhanced_array) ** 2)

    if mse == 0:
        return 100

    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr


def calculate_ssim(original, enhanced):
    original_array, enhanced_array = prepare_for_metrics(original, enhanced)
    score = ssim(original_array, enhanced_array, data_range=255)
    return score


def show_metrics(original, enhanced):
    psnr_value = calculate_psnr(original, enhanced)
    ssim_value = calculate_ssim(original, enhanced)

    st.markdown("### 📊 Evaluation Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("PSNR", f"{psnr_value:.2f} dB")

    with col2:
        st.metric("SSIM", f"{ssim_value:.3f}")


def plot_histogram(image, title):
    img = np.array(image.convert('L'))
    hist = np.histogram(img.flatten(), bins=256, range=[0, 256])[0]

    st.subheader(title)
    st.line_chart(hist)


def show_histograms(original, enhanced):
    col1, col2 = st.columns(2)

    with col1:
        plot_histogram(original, "Original Histogram")

    with col2:
        plot_histogram(enhanced, "Enhanced Histogram")


technique_explanations = {
    "Linear Transform": "Inverts image intensity. Bright areas become dark and dark areas become bright.",
    "Log Transform": "Enhances dark regions by expanding low intensity values.",
    "Threshold Enhancement": "Converts the image into black and white using a fixed threshold.",
    "Power Law Transform": "Changes brightness non-linearly using gamma correction.",
    "Rotate Image": "Rotates the image by a selected angle.",
    "Flip Horizontal": "Mirrors the image from left to right.",
    "Flip Vertical": "Mirrors the image from top to bottom.",
    "Resize Image": "Changes the image dimensions using a scale factor."
}


technique_observations = {
    "Linear Transform": "The image becomes a negative version of the original.",
    "Log Transform": "Dark details become more visible.",
    "Threshold Enhancement": "The image is simplified into black and white regions.",
    "Power Law Transform": "Brightness changes depending on the gamma value.",
    "Rotate Image": "The image orientation changes.",
    "Flip Horizontal": "The left and right sides are reversed.",
    "Flip Vertical": "The top and bottom sides are reversed.",
    "Resize Image": "The image size changes while keeping the same content."
}


def show_result_summary(results):
    st.markdown("### 📋 Results Summary Table")

    header = st.columns([1.4, 2.4, 2, 2, 2.4])
    header[0].markdown("**Technique**")
    header[1].markdown("**Purpose**")
    header[2].markdown("**Before Image**")
    header[3].markdown("**After Image**")
    header[4].markdown("**Observation**")

    for result in results:
        row = st.columns([1.4, 2.4, 2, 2, 2.4])
        row[0].write(result["Technique"])
        row[1].write(result["Purpose"])
        row[2].image(result["Before"], use_container_width=True)
        row[3].image(result["After"], use_container_width=True)
        row[4].write(result["Observation"])


def save_output(images):
    zip_path = "enhanced_images.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for name, img in images:
            image_path = f"{name}.png"
            img.save(image_path)
            zipf.write(image_path, os.path.basename(image_path))
            os.remove(image_path)
    return zip_path


def main():
    menu = st.sidebar.selectbox("Navigation", ["Home", "Medical Image Analysis", "About"])

    if menu == "Medical Image Analysis":
        medical_analysis_page()
        return

    if menu == "About":
        st.title("About This App")
        st.markdown("""
        This is an **Image Enhancement Web Toolkit** developed by students at **Alamein International University**.

        Supported techniques:
        - Linear Transform (Image Negative)
        - Log Transform
        - Threshold Enhancement
        - Power Law (Gamma Correction)
        - Rotate Image
        - Flip Horizontal
        - Flip Vertical
        - Resize Image
        - **Medical Image Analysis** (CNN-based Anomaly Detection)

        Upload an image, apply an enhancement, and download the results.
        """)
        st.info("Part of a university image processing project.")
        return

    st.title("🔧 Image Enhancement Toolkit")

    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png", "bmp", "gif"])

    if uploaded_file is None:
        st.warning("Please upload an image to begin.")
        return

    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)

    enhancement_type = st.selectbox("Choose Enhancement Type", [
        "Linear Transform",
        "Log Transform",
        "Threshold Enhancement",
        "Power Law Transform",
        "Rotate Image",
        "Flip Horizontal",
        "Flip Vertical",
        "Resize Image",
        "All Enhancements"
    ])

    if enhancement_type != "All Enhancements":
        st.info(technique_explanations[enhancement_type])

    enhanced_images = []
    results_summary = []

    if enhancement_type == "Linear Transform":
        enhanced = linear_transform_enhancement(image)
        enhanced_images.append(("linear_transform", enhanced))

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        with col2:
            st.image(enhanced, caption="Linear Transform")

        show_metrics(image, enhanced)
        show_histograms(image, enhanced)

        results_summary.append({
            "Technique": "Linear Transform",
            "Purpose": technique_explanations["Linear Transform"],
            "Before": image,
            "After": enhanced,
            "Observation": technique_observations["Linear Transform"]
        })

    elif enhancement_type == "Log Transform":
        enhanced = log_transform_enhancement(image)
        enhanced_images.append(("log_transform", enhanced))

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        with col2:
            st.image(enhanced, caption="Log Transform")

        show_metrics(image, enhanced)
        show_histograms(image, enhanced)

        results_summary.append({
            "Technique": "Log Transform",
            "Purpose": technique_explanations["Log Transform"],
            "Before": image,
            "After": enhanced,
            "Observation": technique_observations["Log Transform"]
        })

    elif enhancement_type == "Threshold Enhancement":
        enhanced = threshold_enhancement(image)
        enhanced_images.append(("threshold_enhancement", enhanced))

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        with col2:
            st.image(enhanced, caption="Threshold Enhancement")

        show_metrics(image, enhanced)
        show_histograms(image, enhanced)

        results_summary.append({
            "Technique": "Threshold Enhancement",
            "Purpose": technique_explanations["Threshold Enhancement"],
            "Before": image,
            "After": enhanced,
            "Observation": technique_observations["Threshold Enhancement"]
        })

    elif enhancement_type == "Power Law Transform":
        gamma = st.slider("Gamma Value", min_value=0.1, max_value=2.5, value=0.5, step=0.1)
        enhanced = power_law_transform(image, gamma)
        enhanced_images.append(("power_law", enhanced))

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        with col2:
            st.image(enhanced, caption=f"Power Law (Gamma={gamma})")

        show_metrics(image, enhanced)
        show_histograms(image, enhanced)

        results_summary.append({
            "Technique": "Power Law Transform",
            "Purpose": technique_explanations["Power Law Transform"],
            "Before": image,
            "After": enhanced,
            "Observation": f"{technique_observations['Power Law Transform']} Gamma = {gamma}."
        })

    elif enhancement_type == "Rotate Image":
        angle = st.slider("Rotation Angle", 0, 360, 90)
        enhanced = rotate_image(image, angle)
        enhanced_images.append(("rotate", enhanced))

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        with col2:
            st.image(enhanced, caption=f"Rotated {angle}°")

        show_metrics(image, enhanced)
        show_histograms(image, enhanced)

        results_summary.append({
            "Technique": "Rotate Image",
            "Purpose": technique_explanations["Rotate Image"],
            "Before": image,
            "After": enhanced,
            "Observation": f"The image was rotated by {angle} degrees."
        })

    elif enhancement_type == "Flip Horizontal":
        enhanced = flip_horizontal(image)
        enhanced_images.append(("flip_horizontal", enhanced))

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        with col2:
            st.image(enhanced, caption="Flipped Horizontal")

        show_metrics(image, enhanced)
        show_histograms(image, enhanced)

        results_summary.append({
            "Technique": "Flip Horizontal",
            "Purpose": technique_explanations["Flip Horizontal"],
            "Before": image,
            "After": enhanced,
            "Observation": technique_observations["Flip Horizontal"]
        })

    elif enhancement_type == "Flip Vertical":
        enhanced = flip_vertical(image)
        enhanced_images.append(("flip_vertical", enhanced))

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        with col2:
            st.image(enhanced, caption="Flipped Vertical")

        show_metrics(image, enhanced)
        show_histograms(image, enhanced)

        results_summary.append({
            "Technique": "Flip Vertical",
            "Purpose": technique_explanations["Flip Vertical"],
            "Before": image,
            "After": enhanced,
            "Observation": technique_observations["Flip Vertical"]
        })

    elif enhancement_type == "Resize Image":
        scale = st.slider("Scale Factor", 0.1, 2.0, 0.5)
        enhanced = resize_image(image, scale)
        enhanced_images.append(("resize", enhanced))

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        with col2:
            st.image(enhanced, caption=f"Scaled {scale}x")

        show_metrics(image, enhanced)
        show_histograms(image, enhanced)

        results_summary.append({
            "Technique": "Resize Image",
            "Purpose": technique_explanations["Resize Image"],
            "Before": image,
            "After": enhanced,
            "Observation": f"The image was resized using scale factor {scale}."
        })

    elif enhancement_type == "All Enhancements":
        gamma = st.slider("Gamma Value (Power Law)", min_value=0.1, max_value=2.5, value=0.5, step=0.1)

        linear = linear_transform_enhancement(image)
        log = log_transform_enhancement(image)
        thresh = threshold_enhancement(image)
        power = power_law_transform(image, gamma)
        rotate = rotate_image(image, 90)
        flip_h = flip_horizontal(image)
        flip_v = flip_vertical(image)
        resize = resize_image(image, 0.5)

        enhanced_images = [
            ("linear_transform", linear),
            ("log_transform", log),
            ("threshold_enhancement", thresh),
            ("power_law", power),
            ("rotate", rotate),
            ("flip_horizontal", flip_h),
            ("flip_vertical", flip_v),
            ("resize", resize)
        ]

        techniques = [
            ("Linear Transform", linear),
            ("Log Transform", log),
            ("Threshold Enhancement", thresh),
            (f"Power Law (Gamma={gamma})", power),
            ("Rotate 90°", rotate),
            ("Flip Horizontal", flip_h),
            ("Flip Vertical", flip_v),
            ("Resize 0.5x", resize)
        ]

        for name, img in techniques:
            st.subheader(name)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image")
            with col2:
                st.image(img, caption="Enhanced Image")

            show_metrics(image, img)
            show_histograms(image, img)

        results_summary = [
            {
                "Technique": "Linear Transform",
                "Purpose": technique_explanations["Linear Transform"],
                "Before": image,
                "After": linear,
                "Observation": technique_observations["Linear Transform"]
            },
            {
                "Technique": "Log Transform",
                "Purpose": technique_explanations["Log Transform"],
                "Before": image,
                "After": log,
                "Observation": technique_observations["Log Transform"]
            },
            {
                "Technique": "Threshold Enhancement",
                "Purpose": technique_explanations["Threshold Enhancement"],
                "Before": image,
                "After": thresh,
                "Observation": technique_observations["Threshold Enhancement"]
            },
            {
                "Technique": "Power Law Transform",
                "Purpose": technique_explanations["Power Law Transform"],
                "Before": image,
                "After": power,
                "Observation": f"{technique_observations['Power Law Transform']} Gamma = {gamma}."
            },
            {
                "Technique": "Rotate Image",
                "Purpose": technique_explanations["Rotate Image"],
                "Before": image,
                "After": rotate,
                "Observation": "The image was rotated by 90 degrees."
            },
            {
                "Technique": "Flip Horizontal",
                "Purpose": technique_explanations["Flip Horizontal"],
                "Before": image,
                "After": flip_h,
                "Observation": technique_observations["Flip Horizontal"]
            },
            {
                "Technique": "Flip Vertical",
                "Purpose": technique_explanations["Flip Vertical"],
                "Before": image,
                "After": flip_v,
                "Observation": technique_observations["Flip Vertical"]
            },
            {
                "Technique": "Resize Image",
                "Purpose": technique_explanations["Resize Image"],
                "Before": image,
                "After": resize,
                "Observation": "The image was resized using scale factor 0.5."
            }
        ]

    if results_summary:
        show_result_summary(results_summary)

    if enhanced_images:
        st.markdown("### 📥 Download Enhanced Images")

        for name, img in enhanced_images:
            st.download_button(
                label=f"Download {name}",
                data=img.tobytes(),
                file_name=f"{name}.png",
                mime="image/png"
            )

        if st.button("Download All as ZIP"):
            zip_file = save_output(enhanced_images)
            with open(zip_file, "rb") as f:
                st.download_button(
                    "Download ZIP",
                    f,
                    file_name="enhanced_images.zip",
                    mime="application/zip"
                )


def medical_analysis_page():
    st.title("🏥 Medical Image Analysis")
    st.markdown("### CNN-based Anomaly Detection for X-rays and MRI Scans")

    st.warning("""
    **⚠️ Important:** This tool uses heuristic analysis for demonstration purposes only.
    It is NOT a medical diagnostic tool and should NOT be used for actual medical diagnosis.
    Always consult with qualified medical professionals for actual diagnosis.
    """)

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = MedicalImageAnalyzer()

    analyzer = st.session_state.analyzer

    uploaded_file = st.file_uploader(
        "📤 Upload Medical Image (X-ray, MRI, CT Scan)",
        type=["jpg", "jpeg", "png", "bmp", "dcm", "tif", "tiff"]
    )

    if uploaded_file is None:
        st.warning("Please upload a medical image to begin analysis.")
        return

    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Medical Image", use_container_width=True)
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return

    col1, col2 = st.columns(2)
    with col1:
        show_heatmap = st.checkbox("Show Anomaly Heatmap", value=True)
    with col2:
        overlay_alpha = st.slider("Heatmap Overlay Opacity", 0.0, 1.0, 0.4, 0.1)

    if st.button("🔍 Analyze Image", type="primary"):
        try:
            if analyzer.is_trained:
                analyzer.is_trained = False

            results = analyzer.detect_anomalies(image)

            st.markdown("---")
            st.markdown("### 📊 Analysis Results")
            st.info("ℹ️ Analysis Mode: Heuristic-based statistical analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                if results['is_anomaly']:
                    st.error("⚠️ Anomaly Detected")
                else:
                    st.success("✅ Normal")

            with col2:
                st.metric("Confidence", f"{results['confidence']:.1f}%")

            with col3:
                st.metric("Prediction Score", f"{results['prediction_score']:.3f}")

            if 'mean_intensity' in results:
                st.markdown("#### Image Statistics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Mean Intensity", f"{results['mean_intensity']:.1f}")
                with col2:
                    st.metric("Std Deviation", f"{results['std_intensity']:.1f}")
                with col3:
                    st.metric("Anomaly Zones", f"{results['anomaly_zone_ratio']:.1f}%")
                with col4:
                    st.metric("Strong Anomalies", f"{results['strong_anomaly_ratio']:.1f}%")

            st.markdown("---")
            st.markdown("### 🔬 Visual Analysis")

            processed_img = results['processed_image']
            heatmap = results['heatmap']

            if show_heatmap:
                overlay = analyzer.create_overlay(processed_img, heatmap, overlay_alpha)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(processed_img, caption="Processed Image", use_container_width=True)
                with col2:
                    st.image(heatmap, caption="Anomaly Heatmap", use_container_width=True, channels="BGR")
                with col3:
                    st.image(overlay, caption="Overlay Visualization", use_container_width=True, channels="BGR")
            else:
                st.image(processed_img, caption="Processed Image", use_container_width=True)

            st.markdown("---")
            st.markdown("### 📥 Download Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.download_button(
                    label="Download Processed Image",
                    data=cv2.imencode('.png', processed_img)[1].tobytes(),
                    file_name="processed_medical_image.png",
                    mime="image/png"
                )

            with col2:
                if show_heatmap:
                    st.download_button(
                        label="Download Heatmap",
                        data=cv2.imencode('.png', heatmap)[1].tobytes(),
                        file_name="anomaly_heatmap.png",
                        mime="image/png"
                    )

            with col3:
                if show_heatmap:
                    st.download_button(
                        label="Download Overlay",
                        data=cv2.imencode('.png', overlay)[1].tobytes(),
                        file_name="anomaly_overlay.png",
                        mime="image/png"
                    )

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()