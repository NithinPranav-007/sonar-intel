"""
Memory-Safe End-to-End Sonar Preprocessing Pipeline.

Flow:
    load image
        -> quality check
        -> normalization
        -> optional CLAHE
        -> nadir / water-column handling
        -> processed image
        -> optional tiling for inference

IMPORTANT:
- Original raw image is never modified.
- Upload/preview processing does NOT generate YOLO tiles.
- Tiles are generated only when explicitly requested.
"""

from typing import Dict, Any, Optional
import os

import cv2
import numpy as np

from ml.preprocessing.normalize import (
    normalize_sonar_intensity,
    apply_clahe,
    handle_water_column,
)
from ml.preprocessing.quality import compute_image_quality
from ml.preprocessing.tiling import generate_tiles


class SonarPreprocessingPipeline:
    def __init__(
        self,
        tile_size: int = 640,
        tile_overlap: float = 0.20,
        apply_clahe_enhancement: bool = False,
        blank_nadir: bool = False,
    ):
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.apply_clahe_enhancement = apply_clahe_enhancement
        self.blank_nadir = blank_nadir

    def run(
        self,
        image_input: Any,
        output_processed_path: Optional[str] = None,
        generate_tile_data: bool = False,
    ) -> Dict[str, Any]:
        """
        Run sonar preprocessing.

        Args:
            image_input:
                Either:
                - image file path
                - numpy ndarray

            output_processed_path:
                Optional path for saving processed preview.

            generate_tile_data:
                False by default.

                Set True only when tiles are actually required
                for inference.

        Returns:
            Dictionary containing:
                - raw_image
                - processed_image
                - quality_metrics
                - nadir_bounds
                - tiles
                - image_shape
        """

        # =========================================================
        # 1. LOAD IMAGE
        # =========================================================

        if isinstance(image_input, str):

            if not os.path.exists(image_input):
                raise FileNotFoundError(
                    f"Sonar image not found: {image_input}"
                )

            raw_image = cv2.imread(
                image_input,
                cv2.IMREAD_UNCHANGED,
            )

            if raw_image is None:
                raise ValueError(
                    f"Failed to decode image file: {image_input}"
                )

        elif isinstance(image_input, np.ndarray):

            # IMPORTANT:
            # Do NOT unnecessarily copy the input array.
            raw_image = image_input

        else:

            raise TypeError(
                "image_input must be a file path string "
                "or numpy ndarray."
            )

        # =========================================================
        # 2. BASIC VALIDATION
        # =========================================================

        if raw_image.size == 0:
            raise ValueError(
                "Image contains no pixel data."
            )

        image_shape = (
            raw_image.shape[0],
            raw_image.shape[1],
        )

        # =========================================================
        # 3. QUALITY CHECK
        # =========================================================

        quality_metrics = compute_image_quality(
            raw_image
        )

        # =========================================================
        # 4. NORMALIZATION
        # =========================================================

        normalized = normalize_sonar_intensity(
            raw_image
        )

        # =========================================================
        # 5. OPTIONAL CLAHE
        # =========================================================

        if self.apply_clahe_enhancement:

            enhanced = apply_clahe(
                normalized
            )

            del normalized

        else:

            enhanced = normalized

        # =========================================================
        # 6. NADIR / WATER COLUMN HANDLING
        # =========================================================

        processed, nadir_bounds = handle_water_column(
            enhanced,
            nadir_width_ratio=0.08,
            blank_nadir=self.blank_nadir,
        )

        del enhanced

        # =========================================================
        # 7. ENSURE 3-CHANNEL BGR
        # =========================================================

        if len(processed.shape) == 2:

            processed_bgr = cv2.cvtColor(
                processed,
                cv2.COLOR_GRAY2BGR,
            )

            del processed

        else:

            processed_bgr = processed

        # =========================================================
        # 8. SAVE PROCESSED PREVIEW
        # =========================================================

        if output_processed_path:

            output_dir = os.path.dirname(
                os.path.abspath(
                    output_processed_path
                )
            )

            os.makedirs(
                output_dir,
                exist_ok=True,
            )

            success = cv2.imwrite(
                output_processed_path,
                processed_bgr,
            )

            if not success:
                raise IOError(
                    "Failed to save processed image: "
                    f"{output_processed_path}"
                )

        # =========================================================
        # 9. GENERATE TILES ONLY WHEN REQUIRED
        # =========================================================

        tiles = []

        if generate_tile_data:

            tiles = generate_tiles(
                processed_bgr,
                tile_size=self.tile_size,
                overlap=self.tile_overlap,
            )

        # =========================================================
        # 10. RETURN
        # =========================================================

        return {
            "raw_image": raw_image,
            "processed_image": processed_bgr,
            "quality_metrics": quality_metrics,
            "nadir_bounds": nadir_bounds,
            "tiles": tiles,
            "image_shape": image_shape,
        }