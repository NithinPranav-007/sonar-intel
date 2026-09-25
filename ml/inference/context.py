"""
Acoustic-Context Intelligence Module.

Evaluates interpretable acoustic evidence around detected candidates:
1. Local Contrast: Target highlight vs. ambient seabed backscatter.
2. Acoustic Shadow Evidence: Deficit zone behind target highlight away from nadir.
3. Candidate Geometry: Aspect ratio, compactness, linearity.
4. Image Quality: Local acoustic signal quality.

CRITICAL DOMAIN PRINCIPLE:
'No shadow evidence' != 'Negative shadow evidence'.
A flat, buried, or low-profile anthropogenic target (e.g. sunken cable, flat net)
may legitimately exhibit minimal shadow. Candidates are evaluated with graded
evidence without binary rejection.
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2


def extract_acoustic_context(
    image: np.ndarray,
    bbox: Dict[str, int],
    nadir_x: int = 640
) -> Dict[str, float]:
    """
    Analyzes acoustic evidence for a bounding box region in the sonar image.
    
    Args:
        image: Full 2D grayscale or 3D BGR sonar image.
        bbox: dict with "x1", "y1", "x2", "y2".
        nadir_x: Horizontal center/nadir coordinate of the survey swath.
        
    Returns:
        {
            "shadow_evidence": float [0.0 - 1.0],
            "context_score": float [0.0 - 1.0],
            "quality_score": float [0.0 - 1.0],
            "local_contrast": float [0.0 - 1.0]
        }
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h_img, w_img = gray.shape[:2]
    x1 = max(0, min(w_img - 1, bbox["x1"]))
    y1 = max(0, min(h_img - 1, bbox["y1"]))
    x2 = max(0, min(w_img, bbox["x2"]))
    y2 = max(0, min(h_img, bbox["y2"]))

    if x2 <= x1 or y2 <= y1:
        return {
            "shadow_evidence": 0.0,
            "context_score": 0.1,
            "quality_score": 0.5,
            "local_contrast": 0.0
        }

    target_crop = gray[y1:y2, x1:x2]
    target_mean = float(np.mean(target_crop))
    target_max = float(np.max(target_crop))

    # 1. Local Ambient Seabed Ring (Neighborhood)
    pad = int(max(15, (x2 - x1) * 0.5))
    amb_x1 = max(0, x1 - pad)
    amb_y1 = max(0, y1 - pad)
    amb_x2 = min(w_img, x2 + pad)
    amb_y2 = min(h_img, y2 + pad)

    ambient_crop = gray[amb_y1:amb_y2, amb_x1:amb_x2]
    ambient_mean = float(np.mean(ambient_crop)) if ambient_crop.size > 0 else 128.0

    # Local Contrast: relative difference between highlight and ambient
    contrast_diff = max(0.0, target_max - ambient_mean)
    local_contrast = min(1.0, contrast_diff / 120.0)

    # 2. Down-Range Acoustic Shadow Analysis
    # In side-scan sonar, the shadow is cast away from nadir:
    # If target is to the right of nadir, shadow extends to the right (+X).
    # If target is to the left of nadir, shadow extends to the left (-X).
    target_center_x = (x1 + x2) / 2.0
    shadow_pad = int((x2 - x1) * 0.8)

    if target_center_x >= nadir_x:
        # Port-to-starboard down-range region (to the right)
        sh_x1 = min(w_img - 1, x2)
        sh_x2 = min(w_img, x2 + shadow_pad)
    else:
        # Starboard-to-port down-range region (to the left)
        sh_x1 = max(0, x1 - shadow_pad)
        sh_x2 = max(0, x1)

    if sh_x2 > sh_x1:
        shadow_crop = gray[y1:y2, sh_x1:sh_x2]
        shadow_mean = float(np.mean(shadow_crop)) if shadow_crop.size > 0 else ambient_mean
        # Shadow evidence is high if the down-range region is significantly darker than ambient
        shadow_deficit = max(0.0, ambient_mean - shadow_mean)
        shadow_evidence = min(1.0, shadow_deficit / max(20.0, ambient_mean * 0.6))
    else:
        shadow_evidence = 0.35  # Neutral default when shadow zone is at swath edge

    # 3. Geometric Regularity (Anthropogenic objects frequently have distinct aspect ratios)
    width = float(x2 - x1)
    height = float(y2 - y1)
    aspect_ratio = max(width / (height + 1e-5), height / (width + 1e-5))
    # Moderately elongated shapes (e.g. nets, containers, cables) get a small bonus
    geom_score = 0.8 if 1.2 <= aspect_ratio <= 5.0 else 0.6

    # 4. Local Quality
    local_quality = min(1.0, float(np.std(target_crop)) / 45.0)

    # Composite Context Score
    # Note: shadow evidence is weighted moderately so lack of shadow doesn't kill flat objects
    context_score = (
        0.40 * local_contrast +
        0.35 * shadow_evidence +
        0.15 * geom_score +
        0.10 * local_quality
    )
    context_score = round(max(0.1, min(0.99, context_score)), 2)

    return {
        "shadow_evidence": round(max(0.05, min(0.99, shadow_evidence)), 2),
        "context_score": context_score,
        "quality_score": round(max(0.2, min(0.99, local_quality)), 2),
        "local_contrast": round(local_contrast, 2)
    }
