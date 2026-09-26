"""
Production-Ready Hybrid Inference Orchestration Engine.

Prioritizes remote Hugging Face Space API as PRIMARY inference backend,
and seamlessly falls back to the LOCAL DRISHTI detector model upon any failure.

Execution Flow:
User uploads sonar image
        ↓
SONAR-INTEL backend
        ↓
Try Hugging Face API
        ↓
 ┌───────────────┐
 │ API succeeds? │
 └───────┬───────┘
       YES│       NO
          │        │
          ▼        ▼
   HF inference   Log failure
          │        │
          │        ▼
          │    Local ML model
          │        │
          └───┬────┘
              ▼
       Common detection format
              ↓
       Georeferencing pipeline
              ↓
          Frontend
"""

import os
import time
import logging
import cv2
import numpy as np
from typing import List, Optional, Tuple, Any

from backend.app.core.config import settings
from ml.inference.common import CommonDetection, InferenceResult
from ml.inference.hf_client import (
    HuggingFaceInferenceClient,
    HFInferenceError,
    HFTokenMissingError,
    HFAuthError,
    HFQuotaExceededError,
    HFSpaceUnavailableError,
    HFTimeoutError,
    HFInvalidResponseError,
    _sanitize_error_message
)
from ml.inference.drishti_detector import DrishtiDetector, DrishtiDetection

logger = logging.getLogger("sonar_intel.hybrid_inference")


class HybridInferenceEngine:
    """
    Hybrid inference manager coordinating remote Hugging Face Space API and
    local PyTorch/Ultralytics detector fallback.
    """

    def __init__(
        self,
        local_detector: Optional[DrishtiDetector] = None,
        hf_client: Optional[HuggingFaceInferenceClient] = None
    ):
        self.local_detector = local_detector
        self.hf_client = hf_client or HuggingFaceInferenceClient()

    def _get_local_detector(self) -> DrishtiDetector:
        """Lazily initializes the local detector."""
        if self.local_detector is None:
            self.local_detector = DrishtiDetector()
        return self.local_detector

    def run_local_inference(
        self,
        image_path: str,
        image: Optional[np.ndarray] = None,
        confidence_threshold: Optional[float] = None
    ) -> List[CommonDetection]:
        """
        Executes inference using the existing local ML implementation in ml/inference/drishti_detector.py.
        """
        if image is None:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Local image file not found: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to decode image file for local inference: {image_path}")

        detector = self._get_local_detector()
        orig_thresh = detector.confidence_threshold
        if confidence_threshold is not None:
            detector.confidence_threshold = confidence_threshold

        try:
            raw_detections: List[DrishtiDetection] = detector.predict(image)
        finally:
            detector.confidence_threshold = orig_thresh

        # Map internal DrishtiDetection to CommonDetection
        common_dets: List[CommonDetection] = []
        for d in raw_detections:
            common_dets.append(CommonDetection(
                class_name=d.class_name,
                confidence=d.confidence,
                bbox=d.bbox,
                class_id=d.class_id,
                tile_id=d.tile_id,
                model_name=d.model_name,
                model_version=d.model_version,
                is_filtered=d.is_filtered,
                filter_reason=d.filter_reason,
                source_backend="local"
            ))

        return common_dets

    def run_inference(
        self,
        image_path: str,
        image: Optional[np.ndarray] = None,
        confidence_threshold: Optional[float] = None
    ) -> InferenceResult:
        """
        Executes hybrid inference on a sonar image.
        Attempts Hugging Face Space first; falls back to local detector if HF fails.
        """
        start_time = time.time()
        print("[INFERENCE] Starting inference")

        # Load image metadata for coordinate validation
        if image is None and os.path.exists(image_path):
            img_probe = cv2.imread(image_path)
            if img_probe is not None:
                img_h, img_w = img_probe.shape[:2]
            else:
                img_h, img_w = 640, 640
        elif image is not None:
            img_h, img_w = image.shape[:2]
        else:
            img_h, img_w = 640, 640

        # Case A: HF API is explicitly disabled in environment
        if not settings.HF_API_ENABLED:
            print("[HF] Disabled by configuration")
            print("[ML] Using local model")
            print("[ML] Starting local inference")
            try:
                local_dets = self.run_local_inference(image_path, image, confidence_threshold)
                print("[ML] Local inference completed")
                print("[INFERENCE] Backend selected: LOCAL")
                exec_ms = round((time.time() - start_time) * 1000.0, 1)
                return InferenceResult(
                    success=True,
                    backend="local",
                    detections=local_dets,
                    fallback_used=False,
                    model_name=self._get_local_detector().model_name,
                    model_version=self._get_local_detector().model_version,
                    execution_time_ms=exec_ms
                )
            except Exception as e:
                print(f"[ML] Local inference failed: {e}")
                exec_ms = round((time.time() - start_time) * 1000.0, 1)
                return InferenceResult(
                    success=False,
                    backend="none",
                    detections=[],
                    error=f"Local inference failed: {str(e)}",
                    execution_time_ms=exec_ms
                )

        # Case B: HF Token is missing or empty
        if not settings.HF_TOKEN or not settings.HF_TOKEN.strip():
            print("[HF] HF_TOKEN not configured")
            print("[HF] Skipping Hugging Face inference")
            print("[ML] Falling back to local model")
            print("[ML] Starting local inference")
            try:
                local_dets = self.run_local_inference(image_path, image, confidence_threshold)
                print("[ML] Local inference completed")
                print("[INFERENCE] Backend selected: LOCAL")
                exec_ms = round((time.time() - start_time) * 1000.0, 1)
                return InferenceResult(
                    success=True,
                    backend="local",
                    detections=local_dets,
                    fallback_used=True,
                    fallback_reason="HF_TOKEN not configured in environment",
                    model_name=self._get_local_detector().model_name,
                    model_version=self._get_local_detector().model_version,
                    execution_time_ms=exec_ms
                )
            except Exception as e:
                print(f"[ML] Local inference failed: {e}")
                exec_ms = round((time.time() - start_time) * 1000.0, 1)
                return InferenceResult(
                    success=False,
                    backend="none",
                    detections=[],
                    error=f"Local fallback failed: {str(e)}",
                    execution_time_ms=exec_ms
                )

        # Case C: HF Token is present -> Attempt Hugging Face Space
        print("[HF] Attempting Hugging Face inference")
        print("[HF] Authentication configured: YES")
        print(f"[HF] Space: {settings.HF_SPACE}")
        print(f"[HF] Endpoint: {settings.HF_API_ENDPOINT}")
        print("[HF] API call started")

        hf_success = False
        hf_error_reason: Optional[str] = None
        hf_detections: List[CommonDetection] = []
        annotated_image: Optional[str] = None

        try:
            annotated_image, raw_det_data = self.hf_client.call_api(image_path)
            hf_detections = self.hf_client.parse_detection_data(
                raw_det_data,
                image_width=img_w,
                image_height=img_h
            )
            # Filter by confidence threshold if provided
            if confidence_threshold is not None:
                hf_detections = [d for d in hf_detections if d.confidence >= confidence_threshold]

            hf_success = True
            print("[HF] API call completed successfully")
            print("[INFERENCE] Backend selected: HUGGINGFACE")

            exec_ms = round((time.time() - start_time) * 1000.0, 1)
            return InferenceResult(
                success=True,
                backend="huggingface",
                detections=hf_detections,
                annotated_image=annotated_image,
                fallback_used=False,
                model_name=f"HuggingFace Space ({settings.HF_SPACE})",
                model_version="hf-space-v1",
                execution_time_ms=exec_ms
            )

        except HFInvalidResponseError as e:
            hf_error_reason = f"API responded but detection data is invalid: {_sanitize_error_message(str(e), settings.HF_TOKEN)}"
            print(f"[HF] API responded but detection data is invalid: {_sanitize_error_message(str(e), settings.HF_TOKEN)}")
        except HFQuotaExceededError as e:
            hf_error_reason = "ZeroGPU quota exceeded"
            print("[HF] Inference failed")
            print("[HF] Reason: ZeroGPU quota exceeded")
        except HFTimeoutError as e:
            hf_error_reason = f"Request timeout ({settings.HF_API_TIMEOUT}s)"
            print("[HF] Inference failed")
            print(f"[HF] Reason: Request timeout ({settings.HF_API_TIMEOUT}s)")
        except HFAuthError as e:
            hf_error_reason = "Authentication failure"
            print("[HF] Inference failed")
            print("[HF] Reason: Authentication failure")
        except HFSpaceUnavailableError as e:
            clean_reason = _sanitize_error_message(str(e), settings.HF_TOKEN)
            hf_error_reason = f"Space unavailable ({clean_reason})"
            print("[HF] Inference failed")
            print(f"[HF] Reason: {hf_error_reason}")
        except Exception as e:
            clean_reason = _sanitize_error_message(str(e), settings.HF_TOKEN)
            hf_error_reason = f"Remote inference error ({clean_reason})"
            print("[HF] Inference failed")
            print(f"[HF] Reason: {clean_reason}")

        # HF Failed -> Fall back to local model
        print("[ML] Falling back to local inference...")
        print("[ML] Starting local inference")

        try:
            local_dets = self.run_local_inference(image_path, image, confidence_threshold)
            print("[ML] Local inference completed")
            print("[INFERENCE] Backend selected: LOCAL")

            exec_ms = round((time.time() - start_time) * 1000.0, 1)
            return InferenceResult(
                success=True,
                backend="local",
                detections=local_dets,
                fallback_used=True,
                fallback_reason=hf_error_reason or "Hugging Face Space API unavailable",
                model_name=self._get_local_detector().model_name,
                model_version=self._get_local_detector().model_version,
                execution_time_ms=exec_ms
            )
        except Exception as local_err:
            print(f"[ML] Local inference failed: {local_err}")
            exec_ms = round((time.time() - start_time) * 1000.0, 1)
            return InferenceResult(
                success=False,
                backend="none",
                detections=[],
                fallback_used=True,
                fallback_reason=hf_error_reason,
                error=f"Both Hugging Face ({hf_error_reason}) and Local ML model ({str(local_err)}) failed.",
                execution_time_ms=exec_ms
            )
