"""
Core Configuration Settings for SONAR-INTEL.

Defines model provenance, inference parameters, preprocessing hyperparameters,
and product-level filtering policies.
"""

import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

# Discover project repo root reliably
_core_dir = Path(__file__).resolve().parent
_repo_root = _core_dir.parent.parent.parent

class Settings:
    # ------------------------------------------------------------------
    # Storage and Base Paths
    # ------------------------------------------------------------------
    REPO_ROOT: Path = _repo_root
    STORAGE_RAW_DIR: str = os.getenv("STORAGE_RAW_DIR", str(_repo_root / "data" / "raw"))
    STORAGE_PROCESSED_DIR: str = os.getenv("STORAGE_PROCESSED_DIR", str(_repo_root / "data" / "processed"))
    STORAGE_OUTPUTS_DIR: str = os.getenv("STORAGE_OUTPUTS_DIR", str(_repo_root / "outputs"))
    DEMO_DATA_DIR: str = os.getenv("DEMO_DATA_DIR", str(_repo_root / "data" / "demo"))

    # ------------------------------------------------------------------
    # Model Provenance & Artifacts
    # ------------------------------------------------------------------
    _distilled_model = _repo_root / "ml" / "models" / "best_distilled_yolo11n.pt"
    _baseline_model = _repo_root / "ml" / "models" / "dristri" / "best_detector.pt"
    _default_model = str(_distilled_model if _distilled_model.exists() else _baseline_model)

    MODEL_PATH: str = os.getenv("MODEL_PATH", _default_model)
    MODEL_NAME: str = os.getenv("MODEL_NAME", "DRISHTI-YOLO11n" if "yolo11" in _default_model else "DRISHTI-YOLOv8s")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "distilled-v1" if "yolo11" in _default_model else "baseline-v1")
    MODEL_SHA256: str = os.getenv(
        "MODEL_SHA256",
        "2f55eec5d8fe6b4737706392e259c02660a8542cddbcbd603f96d606c54cb927"
    )
    MODEL_SOURCE: str = os.getenv(
        "MODEL_SOURCE",
        "https://huggingface.co/rehan9599/drishti-detector"
    )

    # ------------------------------------------------------------------
    # Hugging Face & Hybrid Inference Architecture
    # ------------------------------------------------------------------
    INFERENCE_PROVIDER: str = os.getenv("INFERENCE_PROVIDER", "hybrid")
    HF_SPACE: str = os.getenv("HF_SPACE", "awzsxde/marine-sonar-ai")
    HF_API_ENDPOINT: str = os.getenv("HF_API_ENDPOINT", "/predict")
    HF_API_ENABLED: bool = os.getenv("HF_API_ENABLED", "true").lower() in ("true", "1", "yes")
    HF_API_TIMEOUT: int = int(os.getenv("HF_API_TIMEOUT", "60"))
    HF_MODEL_ID: str = os.getenv("HF_MODEL_ID", "Samyukta31/sonar_yolo")
    HF_MODEL_FILE: str = os.getenv("HF_MODEL_FILE", "best_distilled_yolo11n.pt")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    def is_hf_configured(self) -> bool:
        """Returns True only if HF inference is enabled and a valid token is present."""
        return bool(self.HF_API_ENABLED and self.HF_TOKEN and self.HF_TOKEN.strip())

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
    # Deployment Guardrails & Low-Memory Constraints (Render 512MB)
    # ------------------------------------------------------------------
    MAX_SONAR_FILE_SIZE: int = int(os.getenv("MAX_SONAR_FILE_SIZE", str(25 * 1024 * 1024)))  # 25 MB
    MAX_NAV_FILE_SIZE: int = int(os.getenv("MAX_NAV_FILE_SIZE", str(5 * 1024 * 1024)))       # 5 MB
    MAX_IMAGE_PIXELS: int = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))                     # 25 Megapixels
    INFERENCE_BATCH_SIZE: int = int(os.getenv("INFERENCE_BATCH_SIZE", "1"))

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

