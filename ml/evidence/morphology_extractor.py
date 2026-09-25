"""
Morphological Physical Priors Extractor.

Extracts geometric shape descriptors from 2D binary segmentation masks
or candidate bounding-box geometries:
- Area, Perimeter, Compactness (Isoperimetric quotient)
- Principal Axis Orientation, Elongation, Eccentricity
- Solidity (Convexity ratio)
- Aspect Ratio

Strictly preserves raw measurements without assigning arbitrary confidence scores.
"""

from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np

from ml.evidence.schemas import MorphologyEvidence


def extract_morphology_evidence(
    mask: Optional[np.ndarray] = None,
    bbox: Optional[List[int]] = None,
    source_type: str = "auto"
) -> MorphologyEvidence:
    """
    Extracts raw morphological shape descriptors.

    Args:
        mask: Optional 2D uint8 binary mask array where >0 indicates target.
        bbox: Optional candidate bounding box [x1, y1, x2, y2].
        source_type: "segmentation_mask", "heuristic_contour", "bbox_geometry", or "auto".

    Returns:
        MorphologyEvidence dataclass containing measurable geometric features.
    """
    # 1. Evaluate mask if available
    has_valid_mask = mask is not None and mask.size > 0 and np.count_nonzero(mask) > 0

    if has_valid_mask:
        bin_mask = (mask > 0).astype(np.uint8)
        area = float(np.sum(bin_mask))
        source = "segmentation_mask" if source_type in ("auto", "segmentation_mask") else source_type

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or area < 2.0:
            return _fallback_bbox_morphology(bbox, source="degenerate_mask")

        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(largest_contour, True))

        # Compactness: 4 * pi * Area / Perimeter^2 (Circle = 1.0, fractal/disjoint < 0.1)
        if perimeter > 0:
            compactness = float((4.0 * np.pi * area) / (perimeter ** 2))
            compactness = min(1.0, max(0.0, compactness))
        else:
            compactness = 0.0

        # Convex Hull and Solidity: Area / ConvexHullArea
        hull = cv2.convexHull(largest_contour)
        hull_area = float(cv2.contourArea(hull))
        hull_perimeter = float(cv2.arcLength(hull, True))
        if hull_area > 0:
            solidity = float(min(1.0, max(0.0, area / hull_area)))
        else:
            solidity = 1.0

        if perimeter > 0:
            convexity = float(min(1.0, max(0.0, hull_perimeter / perimeter)))
        else:
            convexity = 1.0

        # Bounding box & Aspect Ratio
        bx, by, bw, bh = cv2.boundingRect(largest_contour)
        bw = max(1, bw)
        bh = max(1, bh)
        aspect_ratio = float(max(bw, bh) / min(bw, bh))

        # Central Moments for Orientation, Elongation, and Eccentricity
        moments = cv2.moments(largest_contour)
        if moments["m00"] > 0:
            mu20 = moments["mu20"] / moments["m00"]
            mu02 = moments["mu02"] / moments["m00"]
            mu11 = moments["mu11"] / moments["m00"]

            # Orientation angle in degrees [-90.0, +90.0]
            theta = 0.5 * np.arctan2(2.0 * mu11, (mu20 - mu02))
            orientation_deg = float(np.degrees(theta))

            # Eigenvalues of second-order moment inertia matrix
            common = np.sqrt(max(0.0, 4.0 * (mu11 ** 2) + ((mu20 - mu02) ** 2)))
            major_axis = np.sqrt(max(1e-6, 2.0 * (mu20 + mu02 + common)))
            minor_axis = np.sqrt(max(1e-6, 2.0 * (mu20 + mu02 - common)))

            elongation = float(max(1.0, major_axis / max(1e-6, minor_axis)))
            # Eccentricity sqrt(1 - (b/a)^2)
            ratio_sq = (minor_axis / max(1e-6, major_axis)) ** 2
            eccentricity = float(np.sqrt(max(0.0, min(1.0, 1.0 - ratio_sq))))
        else:
            orientation_deg = 0.0
            elongation = aspect_ratio
            eccentricity = float(np.sqrt(max(0.0, 1.0 - (1.0 / (aspect_ratio ** 2)))))

        return MorphologyEvidence(
            source=source,
            area_px=round(area, 1),
            width_px=float(bw),
            height_px=float(bh),
            aspect_ratio=round(aspect_ratio, 2),
            perimeter_px=round(perimeter, 1),
            compactness=round(compactness, 4),
            eccentricity=round(eccentricity, 4),
            elongation=round(elongation, 2),
            orientation_deg=round(orientation_deg, 2),
            solidity=round(solidity, 4),
            convexity=round(convexity, 4),
            status="VALID"
        )

    # 2. Bounding Box Geometry Fallback
    return _fallback_bbox_morphology(bbox, source="bbox_geometry")


def _fallback_bbox_morphology(
    bbox: Optional[List[int]],
    source: str = "bbox_geometry"
) -> MorphologyEvidence:
    """Computes geometric properties from bounding box when no segmentation mask is available."""
    if not bbox or len(bbox) != 4:
        return MorphologyEvidence(
            source="unavailable",
            area_px=0.0,
            width_px=0.0,
            height_px=0.0,
            aspect_ratio=1.0,
            perimeter_px=0.0,
            compactness=0.0,
            eccentricity=0.0,
            elongation=1.0,
            orientation_deg=0.0,
            solidity=1.0,
            status="DEGENERATE"
        )

    x1, y1, x2, y2 = bbox
    w = max(1, abs(x2 - x1))
    h = max(1, abs(y2 - y1))
    area = float(w * h)
    perimeter = float(2 * (w + h))
    aspect_ratio = float(max(w, h) / min(w, h))

    # Rectangle compactness: 4 * pi * (w*h) / (2*(w+h))^2 = pi * w*h / (w+h)^2 <= pi/4 (~0.785)
    compactness = float((4.0 * np.pi * area) / (perimeter ** 2)) if perimeter > 0 else 0.0
    compactness = min(1.0, max(0.0, compactness))

    # Rectangle eccentricity
    a = max(w, h) / 2.0
    b = min(w, h) / 2.0
    eccentricity = float(np.sqrt(max(0.0, 1.0 - ((b / a) ** 2)))) if a > 0 else 0.0

    # Bounding box orientation (axis-aligned is 0.0 or 90.0 depending on dominant dimension)
    orientation_deg = 0.0 if w >= h else 90.0

    return MorphologyEvidence(
        source=source,
        area_px=round(area, 1),
        width_px=float(w),
        height_px=float(h),
        aspect_ratio=round(aspect_ratio, 2),
        perimeter_px=round(perimeter, 1),
        compactness=round(compactness, 4),
        eccentricity=round(eccentricity, 4),
        elongation=round(aspect_ratio, 2),
        orientation_deg=round(orientation_deg, 2),
        solidity=1.0,  # Bounding box is convex
        status="ESTIMATED_FROM_BBOX"
    )
