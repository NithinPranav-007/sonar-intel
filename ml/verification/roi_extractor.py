"""
Candidate Region of Interest (ROI) Extractor for Second-Stage Verification.

Extracts localized sonar image crops corresponding to YOLO candidate bounding boxes
with configurable contextual margins, spatial boundary clamping, and coordinate mapping.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, asdict
import cv2
import numpy as np


@dataclass
class ExtractedROI:
    """Represents an extracted region of interest for downstream verification."""
    roi_image: np.ndarray             # Cropped image array (H_crop, W_crop, C) or (H_crop, W_crop)
    bbox_local: List[int]             # Target bounding box in ROI local coordinates [x1, y1, x2, y2]
    bbox_global: List[int]            # Original bounding box in tile/parent coordinates [x1, y1, x2, y2]
    crop_offset_x: int                # Top-left x-coordinate of the crop in the source image
    crop_offset_y: int                # Top-left y-coordinate of the crop in the source image
    crop_width: int                   # Width of the extracted crop
    crop_height: int                  # Height of the extracted crop
    source_width: int                 # Source image width
    source_height: int                # Source image height
    context_margin: float             # Fractional padding margin applied around bbox
    metadata: Dict[str, Any]          # Upstream telemetry/tile provenance metadata

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Exclude raw numpy array from dict representation for lightweight serialization
        d.pop("roi_image", None)
        return d


class ROIExtractor:
    """
    Extracts candidate bounding box crops with acoustic context margins
    for second-stage verification networks (e.g., SW-Net).
    """

    def __init__(
        self,
        context_margin: float = 0.15,
        min_crop_size: int = 32,
        target_size: Optional[Tuple[int, int]] = None
    ):
        """
        Args:
            context_margin: Fractional margin expanded around the bounding box (e.g. 0.15 = 15% margin)
                            to capture surrounding acoustic shadow and seafloor background.
            min_crop_size: Minimum pixel dimension for extracted ROI crop.
            target_size: Optional fixed (width, height) to resize ROI for network input.
        """
        self.context_margin = max(0.0, context_margin)
        self.min_crop_size = min_crop_size
        self.target_size = target_size

    def extract_roi(
        self,
        image: np.ndarray,
        bbox: Union[List[int], Tuple[int, int, int, int], Dict[str, int]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExtractedROI:
        """
        Extracts a single bounding-box ROI from a sonar image or tile.

        Args:
            image: Source sonar image array (H, W) or (H, W, C).
            bbox: Bounding box as [x1, y1, x2, y2] or dict {"x1": ..., "y1": ..., "x2": ..., "y2": ...}.
            metadata: Optional dictionary with upstream candidate details (class_name, confidence, etc.).

        Returns:
            ExtractedROI object containing the cropped array and coordinate offsets.
        """
        if image is None or image.size == 0:
            raise ValueError("Cannot extract ROI from empty or null image.")

        img_h, img_w = image.shape[:2]

        # Normalize bbox format to [x1, y1, x2, y2]
        if isinstance(bbox, dict):
            bx1 = int(bbox.get("x1", 0))
            by1 = int(bbox.get("y1", 0))
            bx2 = int(bbox.get("x2", 0))
            by2 = int(bbox.get("y2", 0))
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bx1, by1, bx2, by2 = map(int, bbox)
        else:
            raise ValueError(f"Unsupported bbox format: {bbox}. Expected [x1, y1, x2, y2] or dict.")

        # Ensure box ordering
        bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
        by1, by2 = min(by1, by2), max(by1, by2)

        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)

        # Compute context margin
        pad_x = int(bw * self.context_margin)
        pad_y = int(bh * self.context_margin)

        # Clamped crop bounds in source image coordinates
        crop_x1 = max(0, bx1 - pad_x)
        crop_y1 = max(0, by1 - pad_y)
        crop_x2 = min(img_w, bx2 + pad_x)
        crop_y2 = min(img_h, by2 + pad_y)

        # Ensure minimum crop size
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        if crop_w < self.min_crop_size:
            diff = self.min_crop_size - crop_w
            crop_x1 = max(0, crop_x1 - diff // 2)
            crop_x2 = min(img_w, crop_x1 + self.min_crop_size)
            if crop_x2 - crop_x1 < self.min_crop_size:
                crop_x1 = max(0, crop_x2 - self.min_crop_size)

        if crop_h < self.min_crop_size:
            diff = self.min_crop_size - crop_h
            crop_y1 = max(0, crop_y1 - diff // 2)
            crop_y2 = min(img_h, crop_y1 + self.min_crop_size)
            if crop_y2 - crop_y1 < self.min_crop_size:
                crop_y1 = max(0, crop_y2 - self.min_crop_size)

        # Slice crop
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        # Compute local bounding box within the extracted crop
        local_x1 = max(0, bx1 - crop_x1)
        local_y1 = max(0, by1 - crop_y1)
        local_x2 = min(crop.shape[1], bx2 - crop_x1)
        local_y2 = min(crop.shape[0], by2 - crop_y1)

        return ExtractedROI(
            roi_image=crop,
            bbox_local=[local_x1, local_y1, local_x2, local_y2],
            bbox_global=[bx1, by1, bx2, by2],
            crop_offset_x=crop_x1,
            crop_offset_y=crop_y1,
            crop_width=crop.shape[1],
            crop_height=crop.shape[0],
            source_width=img_w,
            source_height=img_h,
            context_margin=self.context_margin,
            metadata=metadata or {}
        )
