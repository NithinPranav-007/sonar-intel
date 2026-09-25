"""
Sonar Storage and Image Processing Management Service.

Ensures that the raw sonar image is preserved unconditionally and generates
enhanced preview counterparts for side-by-side inspection in the viewer.
"""

import os
import shutil
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

from ml.preprocessing.pipeline import SonarPreprocessingPipeline
from ml.preprocessing.quality import compute_image_quality
from backend.app.core.config import settings


class SonarService:
    def __init__(
        self,
        raw_storage_dir: Optional[str] = None,
        processed_storage_dir: Optional[str] = None
    ):
        self.raw_dir = raw_storage_dir or settings.STORAGE_RAW_DIR
        self.processed_dir = processed_storage_dir or settings.STORAGE_PROCESSED_DIR
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.pipeline = SonarPreprocessingPipeline()

    def store_raw_upload(
        self,
        file_bytes: bytes,
        survey_id: str,
        original_filename: str
    ) -> Tuple[str, int, int, Dict[str, Any]]:
        """
        Stores uploaded raw image byte stream.
        NEVER mutates or compresses the raw file.
        
        Returns:
            Tuple of (saved_file_path, width, height, quality_metrics)
        """
        ext = os.path.splitext(original_filename)[1] or ".png"
        raw_filename = f"{survey_id}_raw{ext}"
        target_path = os.path.join(self.raw_dir, raw_filename)

        with open(target_path, "wb") as f:
            f.write(file_bytes)

        # Decode for dimension and quality metrics check
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Uploaded file is not a valid or supported sonar image.")

        h, w = img.shape[:2]
        quality = compute_image_quality(img)

        # Also pre-generate CLAHE processed version for side-by-side viewer
        processed_filename = f"{survey_id}_processed.png"
        processed_path = os.path.join(self.processed_dir, processed_filename)
        self.pipeline.run(img, output_processed_path=processed_path)

        return target_path, w, h, quality

    def get_processed_path(self, survey_id: str) -> str:
        return os.path.join(self.processed_dir, f"{survey_id}_processed.png")
