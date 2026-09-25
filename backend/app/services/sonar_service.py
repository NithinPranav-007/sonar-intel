"""
Memory-Safe Sonar Storage and Image Processing Service.

Responsibilities:
    1. Preserve original sonar image.
    2. Validate uploaded image.
    3. Calculate dimensions and quality.
    4. Generate processed preview.
    5. Avoid generating inference tiles during upload.
    6. Release large image buffers after processing.

The original raw sonar file is NEVER modified.
"""

import os
import gc

import cv2
import numpy as np

from typing import Tuple, Dict, Any, Optional

from ml.preprocessing.pipeline import (
    SonarPreprocessingPipeline
)

from ml.preprocessing.quality import (
    compute_image_quality
)

from backend.app.core.config import settings


class SonarService:

    def __init__(
        self,
        raw_storage_dir: Optional[str] = None,
        processed_storage_dir: Optional[str] = None,
    ):
        """
        Initialize sonar storage service.

        IMPORTANT:
        This service does NOT load a YOLO model.
        The preprocessing pipeline is lightweight.
        """

        # =========================================================
        # STORAGE DIRECTORIES
        # =========================================================

        self.raw_dir = (
            raw_storage_dir
            or settings.STORAGE_RAW_DIR
        )

        self.processed_dir = (
            processed_storage_dir
            or settings.STORAGE_PROCESSED_DIR
        )

        os.makedirs(
            self.raw_dir,
            exist_ok=True,
        )

        os.makedirs(
            self.processed_dir,
            exist_ok=True,
        )

        # =========================================================
        # PREPROCESSING PIPELINE
        # =========================================================

        self.pipeline = SonarPreprocessingPipeline()

    # =============================================================
    # STORE RAW UPLOAD
    # =============================================================

    def store_raw_upload(
        self,
        file_bytes: bytes,
        survey_id: str,
        original_filename: str,
    ) -> Tuple[
        str,
        int,
        int,
        Dict[str, Any],
    ]:
        """
        Store uploaded sonar image.

        The raw file is preserved exactly as uploaded.

        Returns:
            (
                raw_file_path,
                image_width,
                image_height,
                quality_metrics
            )
        """

        # =========================================================
        # 1. VALIDATE FILE
        # =========================================================

        if not file_bytes:
            raise ValueError(
                "Uploaded sonar file is empty."
            )

        # =========================================================
        # 2. DETERMINE EXTENSION
        # =========================================================

        ext = (
            os.path.splitext(
                original_filename
            )[1]
            or ".png"
        )

        raw_filename = (
            f"{survey_id}_raw{ext}"
        )

        target_path = os.path.join(
            self.raw_dir,
            raw_filename,
        )

        # =========================================================
        # 3. SAVE RAW FILE
        # =========================================================

        with open(
            target_path,
            "wb",
        ) as f:

            f.write(file_bytes)

        # =========================================================
        # 4. DECODE IMAGE
        # =========================================================

        nparr = np.frombuffer(
            file_bytes,
            dtype=np.uint8,
        )

        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_UNCHANGED,
        )

        # nparr is no longer needed.
        del nparr

        if img is None:

            raise ValueError(
                "Uploaded file is not a valid "
                "or supported sonar image."
            )

        # =========================================================
        # 5. IMAGE DIMENSIONS
        # =========================================================

        height, width = img.shape[:2]

        # =========================================================
        # 6. QUALITY METRICS
        # =========================================================

        quality = compute_image_quality(
            img
        )

        # =========================================================
        # 7. GENERATE PROCESSED PREVIEW
        # =========================================================

        processed_filename = (
            f"{survey_id}_processed.png"
        )

        processed_path = os.path.join(
            self.processed_dir,
            processed_filename,
        )

        self.pipeline.run(
            img,
            output_processed_path=processed_path,
            generate_tile_data=False,
        )

        # =========================================================
        # 8. RELEASE LARGE IMAGE BUFFER
        # =========================================================

        del img

        gc.collect()

        # =========================================================
        # 9. RETURN
        # =========================================================

        return (
            target_path,
            width,
            height,
            quality,
        )

    # =============================================================
    # PROCESSED IMAGE PATH
    # =============================================================

    def get_processed_path(
        self,
        survey_id: str,
    ) -> str:

        return os.path.join(
            self.processed_dir,
            f"{survey_id}_processed.png",
        )