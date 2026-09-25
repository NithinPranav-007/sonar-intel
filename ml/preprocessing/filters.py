"""
Acoustic Image Filtering Module for Side-Scan Sonar.

Implements the Lee speckle noise filter for sonar backscatter imagery.
"""

from typing import Union
import cv2
import numpy as np


def apply_lee_filter(
    image: np.ndarray,
    window_size: int = 5,
    noise_var: float = 0.04
) -> np.ndarray:
    """
    Applies the vectorized Lee filter to suppress acoustic speckle noise
    while preserving structural boundaries and acoustic highlight-shadow interfaces.

    Formula:
        mean = E[I]
        variance = E[I^2] - (E[I])^2
        weights = max(0, variance - noise_var) / (variance + eps)
        filtered = mean + weights * (I - mean)

    Args:
        image: Grayscale (2D) or BGR (3D) uint8 or float32 image.
        window_size: Size of local neighborhood window (must be odd, e.g., 3, 5, 7).
        noise_var: Estimated relative noise variance of acoustic backscatter.

    Returns:
        Filtered image as uint8 [0, 255] with the same spatial dimensions.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot filter empty or null image.")

    if window_size % 2 == 0 or window_size < 3:
        raise ValueError(f"window_size must be an odd integer >= 3, got {window_size}")

    # Convert to single-channel float32 in [0, 1]
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    img_float = gray.astype(np.float32) / 255.0

    # Vectorized local mean and squared mean using uniform spatial box filter
    ksize = (window_size, window_size)
    local_mean = cv2.boxFilter(img_float, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT)
    local_sq_mean = cv2.boxFilter(img_float ** 2, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT)

    # Local variance: Var(I) = E[I^2] - (E[I])^2
    local_var = np.maximum(local_sq_mean - (local_mean ** 2), 0.0)

    # Adaptive Lee weighting factor
    weights = np.maximum(0.0, local_var - noise_var) / (local_var + 1e-6)
    weights = np.clip(weights, 0.0, 1.0)

    # Linear minimum mean square error estimate
    filtered = local_mean + weights * (img_float - local_mean)
    filtered = np.clip(filtered, 0.0, 1.0)

    return (filtered * 255.0).astype(np.uint8)
