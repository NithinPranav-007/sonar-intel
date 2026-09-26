"""
Acoustic Image Quality Evaluation.

Computes interpretable quality indicators:
- Signal-to-Noise Ratio (SNR) proxy based on backscatter histogram
- Dynamic range utilization
- Blur / focus index (Laplacian variance)
- Overall normalized quality score [0.0 - 1.0]
"""

from typing import Dict
import numpy as np
import cv2


def compute_image_quality(image: np.ndarray) -> Dict[str, float]:
    """
    Computes an acoustic quality score and metrics dictionary.
    Returns:
        {
            "quality_score": float (0.0 to 1.0),
            "snr_db": float,
            "dynamic_range": float,
            "blur_metric": float,
            "is_usable": bool
        }
    """
    if image is None or image.size == 0:
        return {
            "quality_score": 0.0,
            "snr_db": 0.0,
            "dynamic_range": 0.0,
            "blur_metric": 0.0,
            "is_usable": False
        }

    if len(image.shape) == 3 and image.shape[2] >= 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        created_gray = True
    elif len(image.shape) == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
        created_gray = False
    else:
        gray = image
        created_gray = False

    # 1. Dynamic range (fraction of 256 levels occupied)
    min_val, max_val = float(np.min(gray)), float(np.max(gray))
    dr_fraction = min(1.0, (max_val - min_val) / 255.0)

    # 2. Blur index via variance of Laplacian (use zero-copy strided view for large images and CV_32F)
    h, w = gray.shape[:2]
    if h > 1024 or w > 1024:
        # 2x strided view allocates 0 extra RAM
        lap_input = gray[::2, ::2]
    else:
        lap_input = gray

    laplacian = cv2.Laplacian(lap_input, cv2.CV_32F)
    laplacian_var = float(laplacian.var())
    del laplacian

    # Normalize laplacian variance (typical clear sonar is > 150)
    blur_score = min(1.0, float(laplacian_var) / 250.0)

    # 3. Acoustic contrast / SNR proxy
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    snr_proxy = (std_val / (mean_val + 1e-6))
    snr_score = min(1.0, snr_proxy / 0.8)

    if created_gray:
        del gray

    # Combined composite quality score [0.0 - 1.0]
    composite_quality = (0.35 * dr_fraction) + (0.35 * blur_score) + (0.30 * snr_score)
    composite_quality = round(max(0.05, min(0.99, composite_quality)), 2)

    return {
        "quality_score": composite_quality,
        "snr_db": round(20 * np.log10(max(1.0, std_val)), 1),
        "dynamic_range": round(dr_fraction, 2),
        "blur_metric": round(float(laplacian_var), 1),
        "is_usable": composite_quality >= 0.25
    }
