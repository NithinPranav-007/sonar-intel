"""
End-to-End Sonar Preprocessing Pipeline.

Flow:
load image -> quality check -> normalization -> optional CLAHE -> nadir handling -> tiling -> model input.

CRITICAL REQUIREMENT:
The original raw image MUST NEVER be overwritten.
Always preserve raw image, processed image, and metadata.
"""

from typing import Dict, Any, List, Optional
import os
import cv2
import numpy as np

from ml.preprocessing.normalize import (
    normalize_sonar_intensity,
    apply_clahe,
    handle_water_column
)
from ml.preprocessing.quality import compute_image_quality
from ml.preprocessing.tiling import generate_tiles


class SonarPreprocessingPipeline:
    def __init__(
        self,
        tile_size: int = 640,
        tile_overlap: float = 0.20,
        apply_clahe_enhancement: bool = False,
        blank_nadir: bool = False
    ):
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.apply_clahe_enhancement = apply_clahe_enhancement
        self.blank_nadir = blank_nadir

    def run(
        self,
        image_input: Any,
        output_processed_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the preprocessing workflow.
        
        Args:
            image_input: File path (str) or loaded numpy array (np.ndarray).
            output_processed_path: Optional destination to persist the normalized image.
            
        Returns:
            Dict containing:
                - "raw_image": np.ndarray
                - "processed_image": np.ndarray
                - "quality_metrics": dict
                - "tiles": list of tile dicts ready for YOLO
                - "image_shape": (height, width)
        """
        # 1. Load image without modifying source
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Sonar image not found: {image_input}")
            raw_image = cv2.imread(image_input)
            if raw_image is None:
                raise ValueError(f"Failed to decode image file: {image_input}")
        elif isinstance(image_input, np.ndarray):
            raw_image = image_input.copy()
        else:
            raise TypeError("image_input must be a file path string or numpy ndarray.")

        # 2. Quality Check
        quality_metrics = compute_image_quality(raw_image)

        # 3. Normalization (Min-max acoustic stretching)
        normalized = normalize_sonar_intensity(raw_image)

        # 4. Optional CLAHE enhancement
        if self.apply_clahe_enhancement:
            enhanced = apply_clahe(normalized)
        else:
            enhanced = normalized

        # 5. Nadir / Water-column handling
        processed, nadir_bounds = handle_water_column(
            enhanced,
            nadir_width_ratio=0.08,
            blank_nadir=self.blank_nadir
        )

        # Ensure 3-channel BGR representation for standard YOLO input
        if len(processed.shape) == 2:
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        else:
            processed_bgr = processed

        # Save processed copy if output path is provided
        if output_processed_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_processed_path)), exist_ok=True)
            cv2.imwrite(output_processed_path, processed_bgr)

        # 6. Slicing into overlapping tiles
        tiles = generate_tiles(
            processed_bgr,
            tile_size=self.tile_size,
            overlap=self.tile_overlap
        )

        return {
            "raw_image": raw_image,
            "processed_image": processed_bgr,
            "quality_metrics": quality_metrics,
            "nadir_bounds": nadir_bounds,
            "tiles": tiles,
            "image_shape": (raw_image.shape[0], raw_image.shape[1])
        }
