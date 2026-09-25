"""
DRISHTI YOLOv8s Candidate Anomaly Detector.

Responsibilities:
- Load YOLOv8s model once (singleton cache per application process)
- Preprocess sonar imagery using the versioned DRISHTI Lee + CLAHE pipeline
- Execute YOLO inference
- Decode detections and apply confidence and NMS thresholds
- Tag product-level filtered classes (e.g., crab_pot)
- Return pure, model-independent DrishtiDetection internal schemas

NO DATABASE, NO GIS, NO API SIDE EFFECTS.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import os
import gc
import cv2
import numpy as np
import torch

from ml.preprocessing.drishti_preprocess import drishti_preprocess, PREPROCESSING_VERSION
from backend.app.core.config import settings


@dataclass
class DrishtiDetection:
    """Model-independent internal detection representation."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    image_width: int
    image_height: int
    tile_id: Optional[str] = None
    model_name: str = "DRISHTI-YOLOv8s"
    model_version: str = "baseline-v1"
    is_filtered: bool = False
    filter_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DrishtiDetector:
    """
    Dedicated detector service for the pretrained DRISHTI YOLOv8s model.
    """
    _model_cache: Dict[str, Any] = {}

    def _resolve_model_path(self, raw_path: str) -> str:
        """Resolves model path locally or via Hugging Face hub."""
        if os.path.exists(raw_path):
            return raw_path

        # Check relative to REPO_ROOT
        repo_candidate = os.path.join(str(settings.REPO_ROOT), raw_path)
        if os.path.exists(repo_candidate):
            return repo_candidate

        # If model path does not exist, try downloading from Hugging Face if configured
        if settings.HF_MODEL_ID and settings.HF_MODEL_FILE:
            try:
                print(f"[DrishtiDetector] Attempting download of {settings.HF_MODEL_FILE} from HF repo {settings.HF_MODEL_ID}...")
                from huggingface_hub import hf_hub_download
                token = settings.HF_TOKEN if settings.HF_TOKEN else None
                downloaded_file = hf_hub_download(
                    repo_id=settings.HF_MODEL_ID,
                    filename=settings.HF_MODEL_FILE,
                    token=token
                )
                if downloaded_file and os.path.exists(downloaded_file):
                    print(f"[DrishtiDetector] Successfully retrieved model from HF: {downloaded_file}")
                    return downloaded_file
            except Exception as hf_err:
                print(f"[DrishtiDetector] HF download notice ({hf_err}). Checking local fallback paths...")

        # If still not found, check known local model fallbacks
        fallback_candidates = [
            os.path.join(str(settings.REPO_ROOT), "ml", "models", "dristri", "best_detector.pt"),
            os.path.join(str(settings.REPO_ROOT), "ml", "models", "best_distilled_yolo11n.pt"),
            os.path.join("ml", "models", "dristri", "best_detector.pt"),
            os.path.join("ml", "models", "best_distilled_yolo11n.pt"),
            os.path.join("backend", "ml", "models", "dristri", "best_detector.pt")
        ]
        for candidate in fallback_candidates:
            if os.path.exists(candidate):
                print(f"[DrishtiDetector] Using fallback checkpoint at: {candidate}")
                return candidate

        return raw_path

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        image_size: Optional[int] = None,
        device: Optional[str] = None,
        filtered_classes: Optional[List[str]] = None
    ):
        raw_path = model_path or settings.MODEL_PATH
        self.model_path = raw_path
        self.model_name = model_name or settings.MODEL_NAME
        self.model_version = model_version or settings.MODEL_VERSION
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else settings.CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold if iou_threshold is not None else settings.IOU_THRESHOLD
        self.image_size = image_size or settings.IMAGE_SIZE
        self.filtered_classes = set(filtered_classes if filtered_classes is not None else settings.FILTERED_CLASSES)

        if device:
            self.device = device
        elif settings.DEVICE:
            self.device = settings.DEVICE
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Resilient model loading: load if available, or defer to first inference call
        self.model = None
        self.class_names: Dict[int, str] = {
            0: "crab_pot",
            1: "submarine_pipeline",
            2: "shipwreck",
            3: "ghost_net",
            4: "mine_cylinder"
        }
        try:
            self.model = self._get_or_load_model()
            if self.model and hasattr(self.model, "names"):
                self.class_names = self.model.names
        except Exception as e:
            print(f"[DrishtiDetector] Startup load notice: {e}. Model will load on demand.")

    def _get_or_load_model(self):
        """Loads and caches the YOLOv8s/YOLO11n model once per process with low-memory constraints."""
        self.model_path = self._resolve_model_path(self.model_path)
        cache_key = f"{self.model_path}_{self.device}"
        if cache_key in DrishtiDetector._model_cache:
            return DrishtiDetector._model_cache[cache_key]

        # Enforce CPU single-thread limits to prevent thread-pool memory explosion
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        try:
            torch.set_grad_enabled(False)
        except Exception:
            pass

        from ultralytics import YOLO

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Sonar AI model checkpoint not found at: {self.model_path}")

        print(f"[DrishtiDetector] Loading model from {self.model_path} onto {self.device} (low-memory mode)...")
        model = YOLO(self.model_path)
        # Fuse model layers if available to save RAM and computation
        if hasattr(model, "fuse"):
            try:
                model.fuse()
            except Exception:
                pass

        DrishtiDetector._model_cache[cache_key] = model
        gc.collect()
        return model

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Applies Lee speckle filtering and CLAHE consistent with DRISHTI."""
        return drishti_preprocess(
            image=image,
            speckle_filter=settings.PREPROCESSING_SPECKLE_FILTER.lower() == "lee",
            window_size=settings.LEE_WINDOW_SIZE,
            noise_var=settings.LEE_NOISE_VAR,
            apply_clahe_enhancement=settings.PREPROCESSING_CLAHE,
            clahe_clip_limit=settings.CLAHE_CLIP_LIMIT,
            clahe_tile_grid=settings.CLAHE_TILE_GRID_SIZE
        )

    def predict(
        self,
        image: np.ndarray,
        tile_id: Optional[str] = None,
        offset_x: int = 0,
        offset_y: int = 0
    ) -> List[DrishtiDetection]:
        """
        Executes DRISHTI detection on a single image or tile.
        Preprocesses input, executes inference, and decodes detections under torch.inference_mode().

        Args:
            image: 2D or 3D numpy image array.
            tile_id: Optional identifier if running on swath tiles.
            offset_x: Horizontal coordinate offset in parent swath.
            offset_y: Vertical coordinate offset in parent swath.

        Returns:
            List of DrishtiDetection objects.
        """
        if image is None or image.size == 0:
            return []

        if self.model is None:
            self.model = self._get_or_load_model()
            if self.model and hasattr(self.model, "names"):
                self.class_names = self.model.names

        h, w = image.shape[:2]

        # 1. Apply model-specific preprocessing (Lee + CLAHE)
        preprocessed_bgr, _ = self.preprocess(image)

        # 2. Run Ultralytics YOLO inference under torch.inference_mode()
        with torch.inference_mode():
            results = self.model(
                preprocessed_bgr,
                imgsz=min(self.image_size, 640),
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                max_det=50,
                verbose=False
            )

        del preprocessed_bgr

        # 3. Decode detections
        detections = self.decode(
            results=results,
            image_width=w,
            image_height=h,
            tile_id=tile_id,
            offset_x=offset_x,
            offset_y=offset_y
        )

        del results
        return detections

    def decode(
        self,
        results: Any,
        image_width: int,
        image_height: int,
        tile_id: Optional[str] = None,
        offset_x: int = 0,
        offset_y: int = 0
    ) -> List[DrishtiDetection]:
        """Decodes Ultralytics YOLO results into internal DrishtiDetection schema."""
        detections: List[DrishtiDetection] = []

        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                raw_xyxy = box.xyxy[0]
                if hasattr(raw_xyxy, "detach"):
                    raw_xyxy = raw_xyxy.detach().cpu().numpy()
                xyxy = np.asarray(raw_xyxy).reshape(-1)

                raw_conf = box.conf
                if hasattr(raw_conf, "detach"):
                    raw_conf = raw_conf.detach().cpu().numpy()
                conf = float(np.asarray(raw_conf).reshape(-1)[0])

                raw_cls = box.cls
                if hasattr(raw_cls, "detach"):
                    raw_cls = raw_cls.detach().cpu().numpy()
                cls_id = int(np.asarray(raw_cls).reshape(-1)[0])
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

                bx1, by1, bx2, by2 = map(int, xyxy[:4])

                # Clamp bounding box coordinates to image dimensions
                bx1 = max(0, min(image_width, bx1))
                by1 = max(0, min(image_height, by1))
                bx2 = max(0, min(image_width, bx2))
                by2 = max(0, min(image_height, by2))

                # Shift by tile offset if running in tiled mode
                global_x1 = bx1 + offset_x
                global_y1 = by1 + offset_y
                global_x2 = bx2 + offset_x
                global_y2 = by2 + offset_y

                # Apply product-level class policy: crab_pot is filtered downstream
                is_filtered = cls_name in self.filtered_classes
                filter_reason = (
                    f"Filtered per product policy: '{cls_name}' performance not suitable for production triage"
                    if is_filtered else None
                )

                detections.append(DrishtiDetection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=round(conf, 4),
                    bbox=[global_x1, global_y1, global_x2, global_y2],
                    image_width=image_width,
                    image_height=image_height,
                    tile_id=tile_id,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    is_filtered=is_filtered,
                    filter_reason=filter_reason
                ))

        return detections
