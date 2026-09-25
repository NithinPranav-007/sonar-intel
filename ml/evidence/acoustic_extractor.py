"""
Acoustic and Backscatter Evidence Extractor for Side-Scan Sonar Imagery.

Extracts measurable, physics-based acoustic properties around candidate detections:
1. Highlight backscatter strength & local contrast relative to ambient seabed ring
2. Nadir-aware down-range acoustic shadow analysis (present, absent, uncertain, out-of-bounds)
3. Acoustic texture metrics (variance, Shannon entropy, Sobel gradient magnitude, Canny edge density)
4. Directional backscatter consistency and dominant gradient orientation

CRITICAL DOMAIN RULES:
- Never assume every target casts an acoustic shadow (flat/buried targets legitimately lack shadows).
- Shadow absence is explicitly represented ('ABSENT' or 'UNCERTAIN') without forcing a negative decision.
- Strictly preserves raw feature measurements without hardcoded thresholding or final scoring.
"""

from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np

from ml.evidence.schemas import AcousticEvidence, EvidenceStatus
from ml.verification.roi_extractor import ExtractedROI


def compute_shannon_entropy(patch: np.ndarray) -> float:
    """Computes Shannon entropy of an 8-bit grayscale patch in bits [0.0, 8.0]."""
    if patch is None or patch.size == 0:
        return 0.0
    hist, _ = np.histogram(patch.ravel(), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    entropy = -float(np.sum(hist * np.log2(hist)))
    return round(entropy, 3)


def compute_directional_consistency(
    patch: np.ndarray
) -> Tuple[float, float]:
    """
    Computes dominant gradient orientation (degrees) and angular coherence [0.0, 1.0]
    using 2D Sobel gradient vectors.
    """
    if patch is None or patch.size < 9:
        return 0.0, 0.0

    gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)

    mag = np.sqrt(gx ** 2 + gy ** 2)
    angles = np.arctan2(gy, gx)  # [-pi, pi]

    # Weight angles by gradient magnitude
    weight_sum = np.sum(mag)
    if weight_sum < 1e-5:
        return 0.0, 0.0

    # Project into axial distribution [-pi/2, pi/2] using double-angle representation
    cos_2theta = np.sum(mag * np.cos(2.0 * angles)) / weight_sum
    sin_2theta = np.sum(mag * np.sin(2.0 * angles)) / weight_sum

    coherence = float(np.sqrt(cos_2theta ** 2 + sin_2theta ** 2))
    coherence = min(1.0, max(0.0, coherence))

    dominant_angle_rad = 0.5 * np.arctan2(sin_2theta, cos_2theta)
    dominant_angle_deg = float(np.degrees(dominant_angle_rad))

    return round(dominant_angle_deg, 2), round(coherence, 3)


def extract_acoustic_evidence(
    image: np.ndarray,
    bbox: List[int],
    nadir_x: Optional[int] = None,
    roi_context: Optional[ExtractedROI] = None
) -> Tuple[AcousticEvidence, List[str]]:
    """
    Extracts raw acoustic highlight, shadow, texture, and directional features.

    Args:
        image: Full 2D/3D sonar image array.
        bbox: Target bounding box in image coordinates [x1, y1, x2, y2].
        nadir_x: Horizontal coordinate of acoustic nadir (defaults to image width / 2).
        roi_context: Optional pre-extracted ROI object from ROIExtractor.

    Returns:
        Tuple of (AcousticEvidence, warnings_list).
    """
    warnings: List[str] = []

    if image is None or image.size == 0:
        warnings.append("EMPTY_IMAGE_SUPPLIED")
        return _empty_acoustic_evidence("INVALID_IMAGE"), warnings

    if len(image.shape) == 3:
        if image.shape[2] == 1:
            gray = image[:, :, 0]
        elif image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image

    img_h, img_w = gray.shape[:2]
    if nadir_x is None:
        nadir_x = img_w // 2

    # 1. Coordinate Validation & Clamping
    bx1, by1, bx2, by2 = bbox
    bx1 = max(0, min(img_w - 1, bx1))
    by1 = max(0, min(img_h - 1, by1))
    bx2 = max(0, min(img_w, bx2))
    by2 = max(0, min(img_h, by2))

    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    target_area = bw * bh

    if bx2 <= bx1 or by2 <= by1 or target_area < 4:
        warnings.append("DEGENERATE_BOUNDING_BOX")
        return _empty_acoustic_evidence("DEGENERATE_BBOX"), warnings

    # Boundary check
    is_at_boundary = (bx1 == 0 or by1 == 0 or bx2 == img_w or by2 == img_h)
    if is_at_boundary:
        warnings.append("TARGET_TOUCHING_IMAGE_BOUNDARY")

    # 2. Extract Target Crop & Highlight Statistics
    target_crop = gray[by1:by2, bx1:bx2]
    highlight_mean = float(np.mean(target_crop))
    highlight_max = float(np.max(target_crop))
    local_variance = float(np.var(target_crop))

    # 3. Extract Ambient Context Ring (Local Background)
    pad_x = int(max(16, bw * 0.5))
    pad_y = int(max(16, bh * 0.5))

    bg_x1 = max(0, bx1 - pad_x)
    bg_y1 = max(0, by1 - pad_y)
    bg_x2 = min(img_w, bx2 + pad_x)
    bg_y2 = min(img_h, by2 + pad_y)

    bg_crop = gray[bg_y1:bg_y2, bg_x1:bg_x2].copy()

    # Create mask excluding the candidate target to isolate pure background
    bg_mask = np.ones(bg_crop.shape, dtype=bool)
    local_tx1 = bx1 - bg_x1
    local_ty1 = by1 - bg_y1
    local_tx2 = local_tx1 + bw
    local_ty2 = local_ty1 + bh
    bg_mask[local_ty1:local_ty2, local_tx1:local_tx2] = False

    ambient_pixels = bg_crop[bg_mask]
    if ambient_pixels.size > 0:
        local_background_mean = float(np.mean(ambient_pixels))
        local_background_std = float(np.std(ambient_pixels))
    else:
        local_background_mean = float(np.mean(gray))
        local_background_std = float(np.std(gray))
        warnings.append("INSUFFICIENT_AMBIENT_RING_PIXELS")

    # Highlight metrics
    local_contrast_ratio = float(highlight_mean / max(1.0, local_background_mean))
    normalized_highlight_strength = float((highlight_mean - local_background_mean) / 255.0)

    if abs(highlight_mean - local_background_mean) < 10.0:
        warnings.append("LOW_ACOUSTIC_CONTRAST")

    # 4. Down-Range Acoustic Shadow Analysis (Nadir-Aware)
    # Target center relative to nadir
    target_center_x = (bx1 + bx2) / 2.0
    shadow_pad = int(max(bw * 1.0, 32))

    if target_center_x >= nadir_x:
        # Starboard Channel (sound propagates right -> shadow casts to +X)
        sh_x1 = min(img_w - 1, bx2)
        sh_x2 = min(img_w, bx2 + shadow_pad)
    else:
        # Port Channel (sound propagates left -> shadow casts to -X)
        sh_x1 = max(0, bx1 - shadow_pad)
        sh_x2 = max(0, bx1)

    # Shadow vertical coverage matches target vertical span
    sh_y1 = by1
    sh_y2 = by2

    sh_w = sh_x2 - sh_x1
    sh_h = sh_y2 - sh_y1

    if sh_w < 4 or sh_h < 4:
        shadow_status = "OUT_OF_BOUNDS"
        shadow_mean = local_background_mean
        shadow_contrast_ratio = 1.0
        shadow_deficit = 0.0
        shadow_candidate_area_px = 0
        shadow_to_highlight_area_ratio = 0.0
        shadow_length_px = 0.0
        warnings.append("SHADOW_REGION_TRUNCATED_AT_EDGE")
    else:
        shadow_crop = gray[sh_y1:sh_y2, sh_x1:sh_x2]
        shadow_mean = float(np.mean(shadow_crop))
        shadow_deficit = max(0.0, local_background_mean - shadow_mean)
        shadow_contrast_ratio = float(local_background_mean / max(1.0, shadow_mean))

        # Identify dark deficit pixels below ambient threshold (mean - 1.0 * std)
        shadow_threshold = max(5.0, local_background_mean - max(8.0, local_background_std))
        dark_pixels_mask = shadow_crop < shadow_threshold
        shadow_candidate_area_px = int(np.count_nonzero(dark_pixels_mask))
        shadow_to_highlight_area_ratio = float(shadow_candidate_area_px / max(1, target_area))

        # Estimate shadow length along range (horizontal axis)
        col_dark_counts = np.sum(dark_pixels_mask, axis=0)
        dark_cols = np.where(col_dark_counts > (sh_h * 0.2))[0]
        shadow_length_px = float(len(dark_cols))

        # Determine shadow status
        if shadow_deficit >= 15.0 and shadow_candidate_area_px >= max(8, int(target_area * 0.15)):
            shadow_status = "PRESENT"
        elif shadow_deficit < 5.0 and shadow_candidate_area_px < max(4, int(target_area * 0.05)):
            shadow_status = "ABSENT"
        else:
            shadow_status = "UNCERTAIN"

    # 5. Acoustic Texture & Gradients
    entropy = compute_shannon_entropy(target_crop)

    # Mean Sobel gradient magnitude
    gx = cv2.Sobel(target_crop, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(target_crop, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    gradient_magnitude_mean = float(np.mean(grad_mag))

    # Canny edge density
    canny = cv2.Canny(target_crop, 50, 150)
    edge_density = float(np.count_nonzero(canny) / max(1, target_crop.size))

    # 6. Directional Backscatter
    dom_angle, dir_coherence = compute_directional_consistency(target_crop)

    evidence = AcousticEvidence(
        highlight_mean=round(highlight_mean, 2),
        highlight_max=round(highlight_max, 2),
        local_background_mean=round(local_background_mean, 2),
        local_background_std=round(local_background_std, 2),
        local_contrast_ratio=round(local_contrast_ratio, 3),
        normalized_highlight_strength=round(normalized_highlight_strength, 3),
        shadow_status=shadow_status,
        shadow_mean=round(shadow_mean, 2),
        shadow_contrast_ratio=round(shadow_contrast_ratio, 3),
        shadow_deficit=round(shadow_deficit, 2),
        shadow_candidate_area_px=shadow_candidate_area_px,
        shadow_to_highlight_area_ratio=round(shadow_to_highlight_area_ratio, 3),
        shadow_length_px=round(shadow_length_px, 1),
        local_variance=round(local_variance, 2),
        entropy=entropy,
        gradient_magnitude_mean=round(gradient_magnitude_mean, 2),
        edge_density=round(edge_density, 3),
        dominant_gradient_orientation_deg=dom_angle,
        directional_consistency=dir_coherence
    )

    return evidence, warnings


def _empty_acoustic_evidence(status_str: str) -> AcousticEvidence:
    """Returns fallback acoustic evidence for invalid regions."""
    return AcousticEvidence(
        highlight_mean=0.0,
        highlight_max=0.0,
        local_background_mean=0.0,
        local_background_std=0.0,
        local_contrast_ratio=1.0,
        normalized_highlight_strength=0.0,
        shadow_status="OUT_OF_BOUNDS",
        shadow_mean=0.0,
        shadow_contrast_ratio=1.0,
        shadow_deficit=0.0,
        shadow_candidate_area_px=0,
        shadow_to_highlight_area_ratio=0.0,
        shadow_length_px=0.0,
        local_variance=0.0,
        entropy=0.0,
        gradient_magnitude_mean=0.0,
        edge_density=0.0,
        dominant_gradient_orientation_deg=0.0,
        directional_consistency=0.0
    )
