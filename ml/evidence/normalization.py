"""
Feature Normalization Utilities for SONAR-INTEL Evidence.

Provides deterministic mathematical mapping of raw acoustic and morphological
measurements into standardized continuous intervals [0.0, 1.0].

CRITICAL SEPARATION:
- RAW FEATURES: Physical measurements (e.g. dB intensity, pixels, degrees).
- NORMALIZED FEATURES: Scaled continuous representation for numerical stability.
- SCORES / WEIGHTS: Future fusion logic (NOT IMPLEMENTED HERE).
"""

import numpy as np


def normalize_highlight_contrast(contrast_ratio: float, min_val: float = 1.0, max_val: float = 3.0) -> float:
    """
    Normalizes highlight-to-ambient contrast ratio.
    Typical SSS contrast ratios range from 1.0 (no contrast) to 3.0+ (bright metallic echo).
    """
    if contrast_ratio <= min_val:
        return 0.0
    val = (contrast_ratio - min_val) / (max_val - min_val)
    return float(min(1.0, max(0.0, val)))


def normalize_shadow_deficit(shadow_deficit: float, min_deficit: float = 0.0, max_deficit: float = 60.0) -> float:
    """
    Normalizes acoustic shadow intensity deficit in 8-bit scale [0, 255].
    Typical clear shadows exhibit 20-60 intensity levels deficit below ambient.
    """
    if shadow_deficit <= min_deficit:
        return 0.0
    val = (shadow_deficit - min_deficit) / (max_deficit - min_deficit)
    return float(min(1.0, max(0.0, val)))


def normalize_entropy(entropy: float, min_entropy: float = 2.0, max_entropy: float = 7.5) -> float:
    """
    Normalizes Shannon entropy (bits).
    Structured targets exhibit higher texture complexity than flat mud.
    """
    if entropy <= min_entropy:
        return 0.0
    val = (entropy - min_entropy) / (max_entropy - min_entropy)
    return float(min(1.0, max(0.0, val)))


def normalize_gradient_magnitude(grad_mag: float, min_mag: float = 5.0, max_mag: float = 50.0) -> float:
    """
    Normalizes mean Sobel edge gradient magnitude.
    Sharp anthropogenic edges exhibit higher gradient magnitude than smooth seabed.
    """
    if grad_mag <= min_mag:
        return 0.0
    val = (grad_mag - min_mag) / (max_mag - min_mag)
    return float(min(1.0, max(0.0, val)))
