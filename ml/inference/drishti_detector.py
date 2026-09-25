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
        self.model_path = model_path or settings.MODEL_PATH
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

        self.model = self._get_or_load_model()
        self.class_names: Dict[int, str] = getattr(self.model, "names", {
            0: "crab_pot",
            1: "submarine_pipeline",
            2: "shipwreck",
            3: "ghost_net",
            4: "mine_cylinder"
        })

    def _get_or_load_model(self):
        """Loads and caches the YOLOv8s model once per process."""
        cache_key = f"{self.model_path}_{self.device}"
        if cache_key in DrishtiDetector._model_cache:
            return DrishtiDetector._model_cache[cache_key]

        from ultralytics import YOLO
        if not os.path.exists(self.model_path):
            print(f"[DrishtiDetector] Checkpoint missing at {self.model_path}. Attempting fallback download from HuggingFace...")
            try:
                import urllib.request
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                hf_url = "https://huggingface.co/rehan9599/drishti-detector/resolve/main/best_detector.pt"
                urllib.request.urlretrieve(hf_url, self.model_path)
                print(f"[DrishtiDetector] Downloaded DRISHTI model checkpoint to {self.model_path}.")
            except Exception as dl_err:
                raise FileNotFoundError(
                    f"DRISHTI model checkpoint not found at: {self.model_path} and auto-download failed: {dl_err}"
                )

        print(f"[DrishtiDetector] Loading model from {self.model_path} onto {self.device}...")
        model = YOLO(self.model_path)
        DrishtiDetector._model_cache[cache_key] = model
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
        Preprocesses input, executes inference, and decodes detections.

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

        h, w = image.shape[:2]

        # 1. Apply model-specific preprocessing (Lee + CLAHE)
        preprocessed_bgr, _ = self.preprocess(image)

        # 2. Run Ultralytics YOLO inference
        results = self.model(
            preprocessed_bgr,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        # 3. Decode detections
        return self.decode(
            results=results,
            image_width=w,
            image_height=h,
            tile_id=tile_id,
            offset_x=offset_x,
            offset_y=offset_y
        )

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
