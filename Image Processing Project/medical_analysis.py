# -*- coding: utf-8 -*-
"""
Medical Image Analysis Module
CNN-based Anomaly Detection for X-rays and MRI Scans
"""

import numpy as np
from PIL import Image

# OpenCV import with error handling
try:
    import cv2
except ImportError as e:
    if "numpy.core.multiarray" in str(e):
        raise ImportError(
            "NumPy/OpenCV compatibility issue. Please reinstall: "
            "pip install --upgrade --force-reinstall numpy opencv-python"
        ) from e
    raise

# TensorFlow/Keras will be imported lazily when needed
# This allows the module to load even if TensorFlow has compatibility issues

class MedicalImageAnalyzer:
    """CNN-based analyzer for detecting anomalies in medical images"""
    
    def __init__(self):
        self.model = None
        self.input_size = (224, 224)
        self.is_trained = False
        
    def _get_keras(self):
        """Lazy import of TensorFlow/Keras"""
        try:
            import tensorflow as tf
            return tf.keras, tf.keras.layers
        except (ImportError, TypeError, AttributeError) as e:
            error_msg = str(e)
            if "unhashable type" in error_msg or "list" in error_msg.lower():
                raise ImportError(
                    "TensorFlow compatibility issue detected. This may be due to Python 3.9 and TensorFlow 2.13+ compatibility. "
                    "The heuristic analysis mode will work without TensorFlow. "
                    "To use CNN models, consider using Python 3.10+ or TensorFlow 2.12 or earlier."
                ) from e
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow") from e
    
    def build_model(self):
        """Build a CNN model for binary classification (Normal/Anomaly)"""
        keras, layers = self._get_keras()
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        # Note: keras is from _get_keras() method above
        
        self.model = model
        return model
    
    def preprocess_image(self, image):
        """Preprocess medical image for CNN input"""
        # Convert to grayscale if needed
        if isinstance(image, Image.Image):
            if image.mode != 'L':
                image = image.convert('L')
            img_array = np.array(image)
        else:
            img_array = image
        
        # Resize to model input size
        img_resized = cv2.resize(img_array, self.input_size)
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add channel dimension and batch dimension
        img_processed = np.expand_dims(np.expand_dims(img_normalized, axis=-1), axis=0)
        
        return img_processed, img_resized
    
    def detect_anomalies(self, image):
        """
        Detect anomalies in medical image
        
        Returns:
            dict: Analysis results with prediction, confidence, and heatmap
        """
        # Preprocess image
        img_processed, img_resized = self.preprocess_image(image)
        
        # For demo purposes, if model is not trained, use a heuristic approach
        # This avoids TensorFlow import issues
        if not self.is_trained:
            return self._heuristic_analysis(img_resized)
        
        # Only build model if we're using trained model
        if self.model is None:
            self.build_model()
        
        # Make prediction
        prediction = self.model.predict(img_processed, verbose=0)[0][0]
        
        # Determine result
        is_anomaly = prediction > 0.5
        confidence = prediction if is_anomaly else (1 - prediction)
        
        # Generate heatmap (Grad-CAM style visualization)
        heatmap = self._generate_heatmap(img_processed)
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': float(confidence * 100),
            'prediction_score': float(prediction),
            'heatmap': heatmap,
            'processed_image': img_resized
        }
    
    def _heuristic_analysis(self, img_array):
        """
        Heuristic-based analysis when model is not trained
        Uses image statistics to detect potential anomalies
        More conservative approach to reduce false positives
        """
        # Calculate image statistics
        mean_intensity = np.mean(img_array)
        std_intensity = np.std(img_array)
        
        # Calculate local variance to detect unusual regions
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(img_array.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((img_array.astype(np.float32) - local_mean) ** 2, -1, kernel)
        max_local_var = np.max(local_var)
        
        # More conservative thresholds for chest X-rays
        # Normal chest X-rays have high variance due to lungs/heart/bones contrast
        variance_threshold = 60  # Increased from 40 - normal X-rays often have std > 50
        extreme_variance_threshold = 80  # Only flag if variance is very high
        intensity_threshold_low = 20  # Very dark images might indicate issues
        intensity_threshold_high = 230  # Very bright images might indicate issues
        local_var_threshold = np.percentile(local_var, 95)  # Top 5% of local variance
        
        has_anomaly = False
        confidence = 0.3  # Start with lower confidence (more conservative)
        
        # Only flag as anomaly if multiple indicators suggest it
        anomaly_indicators = 0
        
        # Check 1: Extreme variance (very high contrast might indicate issues)
        if std_intensity > extreme_variance_threshold:
            anomaly_indicators += 1
            confidence = min(0.75, 0.4 + (std_intensity - extreme_variance_threshold) / 150)
        
        # Check 2: Very unusual intensity patterns (too dark or too bright)
        if mean_intensity < intensity_threshold_low or mean_intensity > intensity_threshold_high:
            anomaly_indicators += 1
            confidence = max(confidence, 0.65)
        
        # Check 3: Localized high variance regions (potential lesions/abnormalities)
        # Analyze the local variance distribution to detect significant anomaly regions
        high_var_threshold = local_var_threshold * 1.3  # Threshold for high variance regions
        very_high_var_threshold = local_var_threshold * 2.0  # Threshold for very high variance (red zones)
        
        high_var_regions = np.sum(local_var > high_var_threshold)
        very_high_var_regions = np.sum(local_var > very_high_var_threshold)
        total_pixels = local_var.size
        
        high_var_ratio = high_var_regions / total_pixels
        very_high_var_ratio = very_high_var_regions / total_pixels
        
        # Generate heatmap first to analyze it
        heatmap = self._generate_variance_heatmap(img_array)
        
        # Analyze heatmap to detect significant anomaly regions (red/yellow zones)
        # Convert heatmap to grayscale to analyze intensity
        heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        
        # Use both percentile-based and absolute thresholds for better detection
        # Red/yellow zones in heatmap correspond to high values
        percentile_75 = np.percentile(heatmap_gray, 75)
        percentile_90 = np.percentile(heatmap_gray, 90)
        percentile_95 = np.percentile(heatmap_gray, 95)
        
        # Absolute thresholds for visible red/yellow zones in heatmap
        # In JET colormap: blue=low, green=medium, yellow=high, red=very high
        # Grayscale values: low=0-85, medium=85-170, high=170-255
        absolute_yellow_red = 170  # Yellow and red zones
        absolute_red = 220  # Strong red zones
        
        # Use percentile-based detection (relative to image)
        anomaly_zones_percentile = np.sum(heatmap_gray > percentile_75)  # Top 25%
        strong_anomaly_zones_percentile = np.sum(heatmap_gray > percentile_90)  # Top 10%
        very_strong_anomaly_zones_percentile = np.sum(heatmap_gray > percentile_95)  # Top 5%
        
        # Use absolute threshold detection (visible yellow/red zones)
        anomaly_zones_absolute = np.sum(heatmap_gray > absolute_yellow_red)  # Visible yellow/red
        strong_anomaly_zones_absolute = np.sum(heatmap_gray > absolute_red)  # Visible red
        
        # Combine both methods - use the higher count
        anomaly_zones = max(anomaly_zones_percentile, anomaly_zones_absolute)
        strong_anomaly_zones = max(strong_anomaly_zones_percentile, strong_anomaly_zones_absolute)
        very_strong_anomaly_zones = very_strong_anomaly_zones_percentile  # Only percentile for very strong
        
        anomaly_zone_ratio = anomaly_zones / heatmap_gray.size
        strong_anomaly_ratio = strong_anomaly_zones / heatmap_gray.size
        very_strong_anomaly_ratio = very_strong_anomaly_zones / heatmap_gray.size
        
        # Check 3a: Significant high-variance regions
        if high_var_ratio > 0.10:  # More than 10% has high local variance
            anomaly_indicators += 1
            confidence = max(confidence, 0.50 + min(0.30, high_var_ratio * 4))
        
        # Check 3b: Very high variance regions (strong anomalies)
        if very_high_var_ratio > 0.03:  # More than 3% has very high variance
            anomaly_indicators += 1
            confidence = max(confidence, 0.60 + min(0.25, very_high_var_ratio * 5))
        
        # Check 3c: Analyze heatmap - top 25% intensity zones (yellow/orange/red)
        if anomaly_zone_ratio > 0.20:  # More than 20% shows elevated zones
            anomaly_indicators += 1
            confidence = max(confidence, 0.55 + min(0.30, (anomaly_zone_ratio - 0.20) * 2))
        
        # Check 3d: Strong anomaly zones (top 10% - orange/red)
        if strong_anomaly_ratio > 0.05:  # More than 5% shows strong anomalies
            anomaly_indicators += 1
            confidence = max(confidence, 0.65 + min(0.25, (strong_anomaly_ratio - 0.05) * 4))
        
        # Check 3e: Very strong anomaly zones (top 5% - bright red)
        if very_strong_anomaly_ratio > 0.02:  # More than 2% shows very strong anomalies
            anomaly_indicators += 1
            confidence = max(confidence, 0.70 + min(0.20, very_strong_anomaly_ratio * 10))
        
        # Simplified classification logic - prioritize heatmap analysis
        # Use heatmap as primary indicator - if it shows anomalies, flag it
        # Lower thresholds to be more sensitive to visible anomalies
        
        # Primary check: If heatmap shows significant anomaly zones, flag it
        if very_strong_anomaly_ratio > 0.01:  # More than 1% shows very strong red zones
            has_anomaly = True
            confidence = max(confidence, 0.70)
        elif strong_anomaly_ratio > 0.05:  # More than 5% shows strong orange/red zones
            has_anomaly = True
            confidence = max(confidence, 0.65)
        elif anomaly_zone_ratio > 0.20:  # More than 20% shows elevated zones (yellow/orange/red)
            has_anomaly = True
            confidence = max(confidence, 0.60)
        # Secondary check: Multiple statistical indicators
        elif anomaly_indicators >= 2:
            has_anomaly = True
            confidence = max(confidence, 0.55)
        # Tertiary check: Single indicator with heatmap support
        elif anomaly_indicators == 1:
            if strong_anomaly_ratio > 0.03 or anomaly_zone_ratio > 0.15:
                has_anomaly = True
                confidence = max(confidence, 0.50)
            else:
                has_anomaly = False
                confidence = 0.35
        # Final check: Heatmap shows some anomalies even without indicators
        else:
            if strong_anomaly_ratio > 0.04 or anomaly_zone_ratio > 0.18:
                has_anomaly = True
                confidence = 0.55
            else:
                has_anomaly = False
                confidence = 0.30
        
        return {
            'is_anomaly': has_anomaly,
            'confidence': float(confidence * 100),
            'prediction_score': float(confidence),
            'heatmap': heatmap,
            'processed_image': img_array,
            'mean_intensity': float(mean_intensity),
            'std_intensity': float(std_intensity),
            'anomaly_indicators': anomaly_indicators,
            'anomaly_zone_ratio': float(anomaly_zone_ratio * 100),
            'strong_anomaly_ratio': float(strong_anomaly_ratio * 100),
            'very_strong_anomaly_ratio': float(very_strong_anomaly_ratio * 100)
        }
    
    def _generate_variance_heatmap(self, img_array):
        """Generate a heatmap based on local variance"""
        # Calculate local variance using a sliding window
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        # Calculate local mean
        local_mean = cv2.filter2D(img_array.astype(np.float32), -1, kernel)
        
        # Calculate local variance
        local_var = cv2.filter2D((img_array.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        # Normalize heatmap
        if np.max(local_var) > 0:
            heatmap = (local_var / np.max(local_var) * 255).astype(np.uint8)
        else:
            heatmap = np.zeros_like(img_array, dtype=np.uint8)
        
        # Apply colormap for visualization
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap_colored
    
    def _generate_heatmap(self, img_processed):
        """Generate Grad-CAM style heatmap (placeholder for trained model)"""
        # This would use Grad-CAM for a trained model
        # For now, return variance-based heatmap
        img_array = (img_processed[0, :, :, 0] * 255).astype(np.uint8)
        return self._generate_variance_heatmap(img_array)
    
    def create_overlay(self, original, heatmap, alpha=0.4):
        """Create overlay of heatmap on original image"""
        # Resize heatmap to match original if needed
        if heatmap.shape[:2] != original.shape[:2]:
            heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        
        # Convert original to BGR if grayscale
        if len(original.shape) == 2:
            original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            original_bgr = original.copy()
        
        # Blend images
        overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap, alpha, 0)
        
        return overlay

