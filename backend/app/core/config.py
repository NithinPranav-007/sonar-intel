"""
Core Configuration Settings for SONAR-INTEL.

Defines model provenance, inference parameters, preprocessing hyperparameters,
and product-level filtering policies.
"""

import os
from typing import List, Tuple


class Settings:
    # ------------------------------------------------------------------
    # Model Provenance & Artifacts
    # ------------------------------------------------------------------
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        os.path.join("ml", "models", "dristri", "best_detector.pt")
    )
    MODEL_NAME: str = os.getenv("MODEL_NAME", "DRISHTI-YOLOv8s")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "baseline-v1")
    MODEL_SHA256: str = os.getenv(
        "MODEL_SHA256",
        "2f55eec5d8fe6b4737706392e259c02660a8542cddbcbd603f96d606c54cb927"
    )
    MODEL_SOURCE: str = os.getenv(
        "MODEL_SOURCE",
        "https://huggingface.co/rehan9599/drishti-detector"
    )

    # ------------------------------------------------------------------
    # Inference Hyperparameters
    # ------------------------------------------------------------------
    IMAGE_SIZE: int = int(os.getenv("IMAGE_SIZE", "640"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
    DEVICE: str = os.getenv("DEVICE", "")  # Empty string triggers auto CUDA/CPU detection

    # ------------------------------------------------------------------
    # Preprocessing Configurations (DRISHTI Specification)
    # ------------------------------------------------------------------
    PREPROCESSING_VERSION: str = os.getenv("PREPROCESSING_VERSION", "drishti-prep-v1")
    PREPROCESSING_SPECKLE_FILTER: str = os.getenv("PREPROCESSING_SPECKLE_FILTER", "lee")
    PREPROCESSING_CLAHE: bool = os.getenv("PREPROCESSING_CLAHE", "true").lower() in ("true", "1", "yes")
    LEE_WINDOW_SIZE: int = int(os.getenv("LEE_WINDOW_SIZE", "5"))
    LEE_NOISE_VAR: float = float(os.getenv("LEE_NOISE_VAR", "0.04"))
    CLAHE_CLIP_LIMIT: float = float(os.getenv("CLAHE_CLIP_LIMIT", "2.0"))
    CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (
        int(os.getenv("CLAHE_GRID_X", "8")),
        int(os.getenv("CLAHE_GRID_Y", "8"))
    )

    # ------------------------------------------------------------------
    # Class Mappings & Product Policy
    # ------------------------------------------------------------------
    RAW_CLASSES: List[str] = [
        "crab_pot",
        "submarine_pipeline",
        "shipwreck",
        "ghost_net",
        "mine_cylinder"
    ]
    # crab_pot has unusable performance per DRISHTI docs; filtered downstream from Contact generation
    FILTERED_CLASSES: List[str] = os.getenv("FILTERED_CLASSES", "crab_pot").split(",")
    FILTERED_CLASSES = [c.strip() for c in FILTERED_CLASSES if c.strip()]


settings = Settings()
