"""
Side-Scan Sonar (SSS) Image Normalization and Enhancement.

Responsibilities:
- Dynamic range stretching and 8-bit normalization
- Nadir / water-column blanking/masking
- Contrast-Limited Adaptive Histogram Equalization (CLAHE)
"""

from typing import Optional, Tuple
import numpy as np
import cv2


def normalize_sonar_intensity(image: np.ndarray) -> np.ndarray:
    """
    Min-max stretching of raw acoustic intensity to 8-bit grayscale [0, 255].
    Preserves acoustic backscatter dynamics without saturation.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot normalize an empty or null image.")

    if len(image.shape) == 3:
        # Convert to single-channel acoustic backscatter if multi-channel
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Percentile clipping (1st and 99th percentile) to reject sensor artifacts/spikes
    p_min, p_max = np.percentile(gray, (1.0, 99.0))
    if p_max <= p_min:
        return np.zeros_like(gray, dtype=np.uint8)

    clipped = np.clip(gray, p_min, p_max)
    stretched = ((clipped - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
    return stretched


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE)
    to reveal faint acoustic shadows and low-reflectivity seabed boundaries.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def handle_water_column(
    image: np.ndarray,
    nadir_width_ratio: float = 0.08,
    blank_nadir: bool = False
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Detects or blanks the central nadir blind zone / water column return.
    In standard side-scan sonar, the central strip corresponds to the two-way travel
    time through the water column before the first bottom return (FBR).
    
    Returns:
        Processed image and nadir horizontal bounds (nadir_start_x, nadir_end_x).
    """
    h, w = image.shape[:2]
    mid_x = w // 2
    half_nadir = int((w * nadir_width_ratio) / 2)
    nadir_start_x = max(0, mid_x - half_nadir)
    nadir_end_x = min(w, mid_x + half_nadir)

    processed = image.copy()
    if blank_nadir:
        # Zero out nadir blind zone if desired
        if len(processed.shape) == 3:
            processed[:, nadir_start_x:nadir_end_x, :] = 0
        else:
            processed[:, nadir_start_x:nadir_end_x] = 0

    return processed, (nadir_start_x, nadir_end_x)
