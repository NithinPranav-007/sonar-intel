"""
DRISHTI Preprocessing Pipeline Reproduction.

Reproduces the preprocessing chain specified in the DRISHTI model documentation:
Input Image -> Validation -> Grayscale & Dynamic Range Normalization ->
Lee Speckle Filtering -> CLAHE -> 3-channel BGR format for YOLOv8s.

Preprocess Version: drishti-prep-v1
CRITICAL RULE: Never modify the original raw image array.
"""

from typing import Dict, Any, Tuple, Optional, Union
import cv2
import numpy as np
from ml.preprocessing.filters import apply_lee_filter


PREPROCESSING_VERSION = "drishti-prep-v1"


def drishti_preprocess(
    image: np.ndarray,
    speckle_filter: bool = True,
    window_size: int = 5,
    noise_var: float = 0.04,
    apply_clahe_enhancement: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: Tuple[int, int] = (8, 8)
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Executes the versioned DRISHTI preprocessing chain on side-scan sonar imagery.

    Args:
        image: Numpy array of the raw image (2D grayscale or 3D BGR).
        speckle_filter: Whether to apply the Lee speckle noise filter.
        window_size: Neighborhood window size for the Lee filter (default 5).
        noise_var: Acoustic noise variance parameter for Lee filter (default 0.04).
        apply_clahe_enhancement: Whether to apply CLAHE.
        clahe_clip_limit: Threshold for contrast limiting in CLAHE (default 2.0).
        clahe_tile_grid: Grid size for local histogram equalization (default (8, 8)).

    Returns:
        Tuple containing:
            - preprocessed_bgr: 3-channel uint8 image ready for YOLOv8s inference.
            - metadata: Audit dictionary documenting stages, parameters, and version.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot preprocess empty or null image.")

    # 1. Ensure working copy to avoid mutating source image
    working = image.copy()

    # 2. Convert to single-channel grayscale if multi-channel
    if len(working.shape) == 3:
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    else:
        gray = working

    # 3. Dynamic Range Normalization (Percentile stretching 1% to 99%)
    p_min, p_max = np.percentile(gray, (1.0, 99.0))
    if p_max <= p_min:
        normalized = np.zeros_like(gray, dtype=np.uint8)
    else:
        clipped = np.clip(gray, p_min, p_max)
        normalized = ((clipped - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)

    # 4. Lee Speckle Filtering
    if speckle_filter:
        speckle_cleaned = apply_lee_filter(
            normalized,
            window_size=window_size,
            noise_var=noise_var
        )
    else:
        speckle_cleaned = normalized

    # 5. Contrast-Limited Adaptive Histogram Equalization (CLAHE)
    if apply_clahe_enhancement:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
        enhanced = clahe.apply(speckle_cleaned)
    else:
        enhanced = speckle_cleaned

    # 6. Format as 3-channel BGR for standard YOLOv8s inference
    preprocessed_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    metadata = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "input_shape": image.shape,
        "output_shape": preprocessed_bgr.shape,
        "speckle_filter": "lee" if speckle_filter else "none",
        "lee_window_size": window_size if speckle_filter else None,
        "clahe_applied": apply_clahe_enhancement,
        "clahe_clip_limit": clahe_clip_limit if apply_clahe_enhancement else None,
        "clahe_tile_grid": clahe_tile_grid if apply_clahe_enhancement else None
    }

    return preprocessed_bgr, metadata
