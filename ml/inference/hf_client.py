"""
Hugging Face Space Gradio Client for Remote Sonar Anomaly Inference.

Primary inference backend communicating with the Hugging Face Space
(e.g., awzsxde/marine-sonar-ai) via `gradio_client`.

Handles:
- Secure token authentication via environment variables
- Strict zero-leak security: Never logs or returns authentication tokens
- Robust exception classification (ZeroGPU quota, authentication, timeout, offline space, malformed data)
- Structured detection schema parsing and boundary validation
"""

import os
import json
import time
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple

from backend.app.core.config import settings
from ml.inference.common import CommonDetection


class HFInferenceError(Exception):
    """Base exception for Hugging Face inference errors."""
    pass


class HFTokenMissingError(HFInferenceError):
    """Raised when HF_TOKEN is required but not configured."""
    pass


class HFAuthError(HFInferenceError):
    """Raised when HF authentication fails (e.g. invalid token)."""
    pass


class HFQuotaExceededError(HFInferenceError):
    """Raised when ZeroGPU or HF API quota has been exceeded."""
    pass


class HFSpaceUnavailableError(HFInferenceError):
    """Raised when the HF Space is sleeping, paused, building, or unreachable."""
    pass


class HFTimeoutError(HFInferenceError):
    """Raised when the HF Space API call times out."""
    pass


class HFInvalidResponseError(HFInferenceError):
    """Raised when HF API returns an unexpected or unparseable response structure."""
    pass


def _sanitize_error_message(msg: str, token: Optional[str] = None) -> str:
    """Removes any token occurrence from log/exception messages to prevent credential leaks."""
    if not msg:
        return ""
    sanitized = str(msg)
    if token and token.strip():
        sanitized = sanitized.replace(token.strip(), "[REDACTED_HF_TOKEN]")
    # Check for general hf_ token patterns
    import re
    sanitized = re.sub(r"hf_[A-Za-z0-9]{15,}", "[REDACTED_HF_TOKEN]", sanitized)
    return sanitized


class HuggingFaceInferenceClient:
    """
    Client for interacting with the remote Hugging Face Space inference endpoint.
    """

    def __init__(
        self,
        space_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        self.space_name = space_name or settings.HF_SPACE
        self.endpoint = endpoint or settings.HF_API_ENDPOINT
        self.token = token if token is not None else settings.HF_TOKEN
        self.timeout = timeout or settings.HF_API_TIMEOUT
        self._client_instance = None

    def _get_client(self):
        """Initializes or retrieves cached Gradio Client."""
        if self._client_instance is not None:
            return self._client_instance

        try:
            from gradio_client import Client
            hf_token = self.token if (self.token and self.token.strip()) else None
            try:
                self._client_instance = Client(self.space_name, token=hf_token)
            except TypeError:
                self._client_instance = Client(self.space_name, hf_token=hf_token)
            return self._client_instance
        except Exception as e:
            raw_msg = str(e)
            clean_msg = _sanitize_error_message(raw_msg, self.token)
            lower_msg = clean_msg.lower()

            if "unauthorized" in lower_msg or "401" in lower_msg or "invalid token" in lower_msg:
                raise HFAuthError(f"Authentication failure connecting to Space '{self.space_name}': {clean_msg}")
            if "quota" in lower_msg or "zerogpu" in lower_msg:
                raise HFQuotaExceededError(f"ZeroGPU quota exceeded: {clean_msg}")
            if "not found" in lower_msg or "404" in lower_msg or "paused" in lower_msg or "sleeping" in lower_msg or "offline" in lower_msg:
                raise HFSpaceUnavailableError(f"Space unavailable ({self.space_name}): {clean_msg}")

            raise HFSpaceUnavailableError(f"Failed to connect to Hugging Face Space '{self.space_name}': {clean_msg}")

    def call_api(self, image_path: str, timeout: Optional[int] = None) -> Tuple[Optional[str], Any]:
        """
        Calls the Hugging Face Space endpoint with timeout handling and error classification.

        Returns:
            Tuple of (annotated_image_path, detection_data)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Sonar image not found for HF inference: {image_path}")

        req_timeout = timeout or self.timeout

        from gradio_client import handle_file

        def _execute_predict():
            client = self._get_client()
            return client.predict(
                image_path=handle_file(image_path),
                api_name=self.endpoint
            )

        # Run with strict timeout guard
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_execute_predict)
        try:
            result = future.result(timeout=req_timeout)
        except concurrent.futures.TimeoutError:
            executor.shutdown(wait=False, cancel_futures=True)
            raise HFTimeoutError(f"Hugging Face API request timed out after {req_timeout}s")
        except Exception as e:
            executor.shutdown(wait=False, cancel_futures=True)
            raw_msg = str(e)
            clean_msg = _sanitize_error_message(raw_msg, self.token)
            lower_msg = clean_msg.lower()

            # Classify known error patterns
            if "zerogpu" in lower_msg or "quota" in lower_msg or "limit" in lower_msg and "runs" in lower_msg:
                raise HFQuotaExceededError(f"ZeroGPU quota exceeded: {clean_msg}")
            if "401" in lower_msg or "unauthorized" in lower_msg or "authentication" in lower_msg or "invalid token" in lower_msg:
                raise HFAuthError(f"Authentication failure: {clean_msg}")
            if "404" in lower_msg or "not found" in lower_msg or "sleeping" in lower_msg or "paused" in lower_msg or "building" in lower_msg:
                raise HFSpaceUnavailableError(f"Space unavailable: {clean_msg}")
            if "timeout" in lower_msg or "timed out" in lower_msg:
                raise HFTimeoutError(f"Request timeout: {clean_msg}")
            
            # General HF API error
            raise HFInferenceError(f"Hugging Face API error: {clean_msg}")
        finally:
            executor.shutdown(wait=False)

        # Validate response container structure
        if result is None:
            raise HFInvalidResponseError("HF API returned null response")

        annotated_image: Optional[str] = None
        detection_data: Any = None

        if isinstance(result, (tuple, list)):
            if len(result) >= 2:
                annotated_image = result[0] if isinstance(result[0], str) else None
                detection_data = result[1]
            elif len(result) == 1:
                detection_data = result[0]
            else:
                raise HFInvalidResponseError("HF API returned empty list/tuple")
        elif isinstance(result, dict):
            # Dict with keys
            annotated_image = result.get("annotated_image") or result.get("image")
            detection_data = result.get("detections") or result.get("detection_data") or result
        else:
            detection_data = result

        return annotated_image, detection_data

    def parse_detection_data(
        self,
        detection_data: Any,
        image_width: int,
        image_height: int
    ) -> List[CommonDetection]:
        """
        Parses and strictly validates raw detection data into standardized CommonDetection objects.
        """
        if detection_data is None:
            return []

        # Parse string JSON if returned as a JSON-encoded string
        if isinstance(detection_data, str):
            trimmed = detection_data.strip()
            if not trimmed:
                return []
            try:
                detection_data = json.loads(trimmed)
            except Exception as e:
                raise HFInvalidResponseError(f"Failed to parse JSON string detection data: {e}")

        raw_items: List[Any] = []
        if isinstance(detection_data, list):
            raw_items = detection_data
        elif isinstance(detection_data, dict):
            # Check for standard nested keys
            for key in ["detections", "objects", "candidates", "results", "boxes", "items"]:
                if key in detection_data and isinstance(detection_data[key], list):
                    raw_items = detection_data[key]
                    break
            else:
                # If dict doesn't contain a known list key, maybe it's a single detection or empty dict
                if "class" in detection_data or "class_name" in detection_data or "bbox" in detection_data:
                    raw_items = [detection_data]
                else:
                    # Could be metadata or empty result
                    raw_items = []
        else:
            raise HFInvalidResponseError(f"Unexpected detection data type: {type(detection_data).__name__}")

        parsed_detections: List[CommonDetection] = []

        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                raise HFInvalidResponseError(f"Detection item at index {idx} is not a dictionary: {type(item).__name__}")

            # Extract class name
            cls_name = (
                item.get("type") or
                item.get("class") or
                item.get("class_name") or
                item.get("name") or
                item.get("label") or
                item.get("category") or
                "artificial_anomaly"
            )
            cls_name = str(cls_name).strip()

            # Extract confidence
            conf_val = item.get("confidence")
            if conf_val is None:
                conf_val = item.get("score")
            if conf_val is None:
                conf_val = item.get("conf")
            if conf_val is None:
                conf_val = 0.5  # default if not explicitly given

            try:
                conf = float(conf_val)
                conf = max(0.0, min(1.0, conf))
            except (ValueError, TypeError):
                raise HFInvalidResponseError(f"Invalid confidence score '{conf_val}' in detection item {idx}")

            # Extract and validate bounding box
            bbox_raw = item.get("bbox") or item.get("box") or item.get("bounding_box")
            if bbox_raw is None:
                # Check for discrete coordinates
                if all(k in item for k in ["x1", "y1", "x2", "y2"]):
                    bbox_raw = [item["x1"], item["y1"], item["x2"], item["y2"]]
                elif all(k in item for k in ["xmin", "ymin", "xmax", "ymax"]):
                    bbox_raw = [item["xmin"], item["ymin"], item["xmax"], item["ymax"]]
                elif all(k in item for k in ["x", "y", "w", "h"]):
                    bbox_raw = [item["x"], item["y"], item["x"] + item["w"], item["y"] + item["h"]]
                else:
                    raise HFInvalidResponseError(f"Missing bounding box in detection item {idx}")

            # Parse bbox into [x1, y1, x2, y2]
            if isinstance(bbox_raw, dict):
                x1 = bbox_raw.get("x1", bbox_raw.get("xmin", bbox_raw.get("left", 0)))
                y1 = bbox_raw.get("y1", bbox_raw.get("ymin", bbox_raw.get("top", 0)))
                x2 = bbox_raw.get("x2", bbox_raw.get("xmax", bbox_raw.get("right", 0)))
                y2 = bbox_raw.get("y2", bbox_raw.get("ymax", bbox_raw.get("bottom", 0)))
                bbox_list = [x1, y1, x2, y2]
            elif isinstance(bbox_raw, (list, tuple)):
                if len(bbox_raw) != 4:
                    raise HFInvalidResponseError(f"Bounding box must contain 4 values, got {len(bbox_raw)}")
                bbox_list = list(bbox_raw)
            else:
                raise HFInvalidResponseError(f"Invalid bounding box type '{type(bbox_raw).__name__}'")

            try:
                x1, y1, x2, y2 = [float(v) for v in bbox_list]
            except (ValueError, TypeError):
                raise HFInvalidResponseError(f"Non-numeric bounding box values: {bbox_list}")

            # Handle normalized coordinates (0.0 - 1.0)
            if 0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= y2 <= 1.0 and max(image_width, image_height) > 1:
                x1 = int(round(x1 * image_width))
                x2 = int(round(x2 * image_width))
                y1 = int(round(y1 * image_height))
                y2 = int(round(y2 * image_height))
            else:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Ensure valid ordering x1 <= x2, y1 <= y2
            bx1, bx2 = min(x1, x2), max(x1, x2)
            by1, by2 = min(y1, y2), max(y1, y2)

            # Clamp coordinates to image boundaries
            bx1 = max(0, min(image_width, bx1))
            bx2 = max(0, min(image_width, bx2))
            by1 = max(0, min(image_height, by1))
            by2 = max(0, min(image_height, by2))

            # Apply product-level class policy (e.g., crab_pot filtering)
            is_filtered = cls_name in settings.FILTERED_CLASSES
            filter_reason = (
                f"Filtered per product policy: '{cls_name}' performance not suitable for production triage"
                if is_filtered else None
            )

            parsed_detections.append(CommonDetection(
                class_name=cls_name,
                confidence=round(conf, 4),
                bbox=[bx1, by1, bx2, by2],
                class_id=item.get("class_id"),
                tile_id=item.get("tile_id"),
                model_name=f"HuggingFace Space ({self.space_name})",
                model_version="hf-space-v1",
                is_filtered=is_filtered,
                filter_reason=filter_reason,
                source_backend="huggingface"
            ))

        return parsed_detections
