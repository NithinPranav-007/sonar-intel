"""
Comprehensive Unit and Integration Tests for Production-Ready Hybrid Inference Architecture.

Tests all required test matrix cases:
1. Test 1 — HF success (HF called, succeeds, local model NOT called, backend = huggingface)
2. Test 2 — HF quota exceeded (HF called, fails with ZeroGPU quota error, local model called, backend = local)
3. Test 3 — HF token missing (HF skipped, local model called, backend = local)
4. Test 4 — HF timeout (HF times out, local model called, backend = local)
5. Test 5 — Malformed HF response (HF response invalid, local model called, backend = local)
6. Test 6 — Local model failure (HF fails, local model fails, clean error returned)
7. Test 7 — Geospatial & Georeferencing Pipeline Compatibility with either backend
8. Test 8 — Zero-leak security: HF token never exposed in responses or logs
"""

import os
import io
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

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
from ml.inference.hybrid_engine import HybridInferenceEngine
from ml.inference.drishti_detector import DrishtiDetector, DrishtiDetection
from backend.app.services.inference_service import InferenceService
from backend.app.services.geolocation_service import GeolocationService
from backend.app.services.transformer import transform_drishti_detections_to_contacts


@pytest.fixture
def sample_sonar_image_path(tmp_path):
    """Creates a temporary valid 640x640 sonar test image on disk."""
    img_path = str(tmp_path / "test_swath.png")
    img = np.full((640, 640, 3), 120, dtype=np.uint8)
    # Synthetic highlight and shadow
    img[200:250, 200:300] = 240
    img[200:250, 300:400] = 20
    cv2.imwrite(img_path, img)
    return img_path


@pytest.fixture
def mock_local_detector():
    """Mock local detector that returns deterministic candidates."""
    detector = MagicMock(spec=DrishtiDetector)
    detector.model_name = "DRISHTI-YOLO11n"
    detector.model_version = "distilled-v1"
    detector.confidence_threshold = 0.25
    detector.iou_threshold = 0.45
    detector.predict.return_value = [
        DrishtiDetection(
            class_id=2,
            class_name="shipwreck",
            confidence=0.88,
            bbox=[200, 200, 350, 280],
            image_width=640,
            image_height=640,
            tile_id="LOCAL_T001",
            model_name="DRISHTI-YOLO11n",
            model_version="distilled-v1",
            is_filtered=False
        )
    ]
    return detector


class TestHybridInference:
    """Core hybrid inference architectural tests."""

    def test_1_hf_success(self, sample_sonar_image_path, mock_local_detector):
        """
        Test 1 — HF success:
        HF is called, HF succeeds, local model is NOT called, backend = huggingface.
        """
        mock_hf_client = MagicMock(spec=HuggingFaceInferenceClient)
        mock_hf_client.call_api.return_value = (
            "/tmp/annotated.png",
            [
                {"class": "shipwreck", "confidence": 0.94, "bbox": [150, 180, 320, 260]}
            ]
        )
        mock_hf_client.parse_detection_data.return_value = [
            CommonDetection(
                class_name="shipwreck",
                confidence=0.94,
                bbox=[150, 180, 320, 260],
                class_id=2,
                model_name="HuggingFace Space (awzsxde/marine-sonar-ai)",
                model_version="hf-space-v1",
                source_backend="huggingface"
            )
        ]

        engine = HybridInferenceEngine(
            local_detector=mock_local_detector,
            hf_client=mock_hf_client
        )

        with patch.object(settings, "HF_API_ENABLED", True), \
             patch.object(settings, "HF_TOKEN", "hf_valid_test_token_12345"):

            result = engine.run_inference(sample_sonar_image_path)

            # Assertions
            assert result.success is True
            assert result.backend == "huggingface"
            assert result.fallback_used is False
            assert len(result.detections) == 1
            assert result.detections[0].class_name == "shipwreck"
            assert result.detections[0].confidence == 0.94
            assert result.detections[0].bbox == [150, 180, 320, 260]

            # Verify HF was called and local model was NOT called
            mock_hf_client.call_api.assert_called_once_with(sample_sonar_image_path)
            mock_local_detector.predict.assert_not_called()

    def test_2_hf_quota_exceeded(self, sample_sonar_image_path, mock_local_detector):
        """
        Test 2 — HF quota exceeded:
        HF called, ZeroGPU quota exceeded exception raised, local model called, backend = local.
        """
        mock_hf_client = MagicMock(spec=HuggingFaceInferenceClient)
        mock_hf_client.call_api.side_effect = HFQuotaExceededError("ZeroGPU runs limit exceeded")

        engine = HybridInferenceEngine(
            local_detector=mock_local_detector,
            hf_client=mock_hf_client
        )

        with patch.object(settings, "HF_API_ENABLED", True), \
             patch.object(settings, "HF_TOKEN", "hf_valid_test_token_12345"):

            result = engine.run_inference(sample_sonar_image_path)

            # Assertions
            assert result.success is True
            assert result.backend == "local"
            assert result.fallback_used is True
            assert "quota" in result.fallback_reason.lower()
            assert len(result.detections) == 1
            assert result.detections[0].source_backend == "local"

            # Verify HF was called AND local model fallback was invoked
            mock_hf_client.call_api.assert_called_once_with(sample_sonar_image_path)
            mock_local_detector.predict.assert_called_once()

    def test_3_hf_token_missing(self, sample_sonar_image_path, mock_local_detector):
        """
        Test 3 — HF token missing:
        HF is skipped directly without calling remote API, local model called, backend = local.
        """
        mock_hf_client = MagicMock(spec=HuggingFaceInferenceClient)

        engine = HybridInferenceEngine(
            local_detector=mock_local_detector,
            hf_client=mock_hf_client
        )

        with patch.object(settings, "HF_API_ENABLED", True), \
             patch.object(settings, "HF_TOKEN", ""):

            result = engine.run_inference(sample_sonar_image_path)

            # Assertions
            assert result.success is True
            assert result.backend == "local"
            assert result.fallback_used is True
            assert "HF_TOKEN not configured" in result.fallback_reason
            assert len(result.detections) == 1

            # Verify remote HF was NOT called at all
            mock_hf_client.call_api.assert_not_called()
            mock_local_detector.predict.assert_called_once()

    def test_4_hf_timeout(self, sample_sonar_image_path, mock_local_detector):
        """
        Test 4 — HF timeout:
        HF times out after timeout limit, local model called, backend = local.
        """
        mock_hf_client = MagicMock(spec=HuggingFaceInferenceClient)
        mock_hf_client.call_api.side_effect = HFTimeoutError("Request timed out after 60s")

        engine = HybridInferenceEngine(
            local_detector=mock_local_detector,
            hf_client=mock_hf_client
        )

        with patch.object(settings, "HF_API_ENABLED", True), \
             patch.object(settings, "HF_TOKEN", "hf_valid_test_token_12345"):

            result = engine.run_inference(sample_sonar_image_path)

            # Assertions
            assert result.success is True
            assert result.backend == "local"
            assert result.fallback_used is True
            assert "timeout" in result.fallback_reason.lower()
            assert len(result.detections) == 1

            # Verify remote HF was called and local model took over
            mock_hf_client.call_api.assert_called_once_with(sample_sonar_image_path)
            mock_local_detector.predict.assert_called_once()

    def test_5_malformed_hf_response(self, sample_sonar_image_path, mock_local_detector):
        """
        Test 5 — Malformed HF response:
        HF returns unparseable or corrupted detection payload, local model called, backend = local.
        """
        mock_hf_client = MagicMock(spec=HuggingFaceInferenceClient)
        # Call succeeds at network level but parse_detection_data raises HFInvalidResponseError
        mock_hf_client.call_api.return_value = (None, "Corrupted Non-JSON string {{{")
        mock_hf_client.parse_detection_data.side_effect = HFInvalidResponseError("Corrupted non-JSON detection data")

        engine = HybridInferenceEngine(
            local_detector=mock_local_detector,
            hf_client=mock_hf_client
        )

        with patch.object(settings, "HF_API_ENABLED", True), \
             patch.object(settings, "HF_TOKEN", "hf_valid_test_token_12345"):

            result = engine.run_inference(sample_sonar_image_path)

            # Assertions
            assert result.success is True
            assert result.backend == "local"
            assert result.fallback_used is True
            assert "invalid" in result.fallback_reason.lower()
            assert len(result.detections) == 1

            mock_local_detector.predict.assert_called_once()

    def test_6_local_model_failure(self, sample_sonar_image_path):
        """
        Test 6 — Local model failure:
        Both HF and local model fail -> clean failure result returned without crashing.
        """
        mock_hf_client = MagicMock(spec=HuggingFaceInferenceClient)
        mock_hf_client.call_api.side_effect = HFSpaceUnavailableError("Space sleeping")

        failing_local_detector = MagicMock(spec=DrishtiDetector)
        failing_local_detector.predict.side_effect = RuntimeError("Out of memory on local device")

        engine = HybridInferenceEngine(
            local_detector=failing_local_detector,
            hf_client=mock_hf_client
        )

        with patch.object(settings, "HF_API_ENABLED", True), \
             patch.object(settings, "HF_TOKEN", "hf_valid_test_token_12345"):

            result = engine.run_inference(sample_sonar_image_path)

            # Assertions
            assert result.success is False
            assert result.backend == "none"
            assert result.error is not None
            assert "failed" in result.error.lower()

    def test_7_georeferencing_compatibility(self, sample_sonar_image_path, tmp_path):
        """
        Test 7 — Georeferencing pipeline works identically with HF detections or Local detections.
        """
        # Create a synthetic navigation track CSV
        nav_path = str(tmp_path / "nav.csv")
        with open(nav_path, "w", encoding="utf-8") as f:
            f.write("ping_id,timestamp,latitude,longitude,heading,altitude,range\n")
            f.write("1,2026-08-31T14:00:00Z,13.0827,80.2707,45.0,12.0,50.0\n")
            f.write("2,2026-08-31T14:00:01Z,13.0828,80.2708,45.0,12.0,50.0\n")

        geo_service = GeolocationService(nav_file_path=nav_path)

        # 1. Detections from HF
        hf_detection = DrishtiDetection(
            class_id=2,
            class_name="shipwreck",
            confidence=0.92,
            bbox=[200, 200, 300, 300],
            image_width=640,
            image_height=640,
            tile_id="SURV_HF",
            model_name="HuggingFace Space (awzsxde/marine-sonar-ai)",
            model_version="hf-space-v1"
        )
        hf_contacts = transform_drishti_detections_to_contacts(
            detections=[hf_detection],
            survey_id="SURV_GEO_TEST",
            geo_service=geo_service,
            image_width=640,
            image_height=640
        )
        assert len(hf_contacts) == 1
        assert hf_contacts[0].latitude is not None
        assert hf_contacts[0].longitude is not None
        assert hf_contacts[0].localization_status == "ESTIMATED"
        assert hf_contacts[0].model_name == "HuggingFace Space (awzsxde/marine-sonar-ai)"

        # 2. Detections from Local
        local_detection = DrishtiDetection(
            class_id=2,
            class_name="shipwreck",
            confidence=0.85,
            bbox=[200, 200, 300, 300],
            image_width=640,
            image_height=640,
            tile_id="SURV_LOCAL",
            model_name="DRISHTI-YOLO11n",
            model_version="distilled-v1"
        )
        local_contacts = transform_drishti_detections_to_contacts(
            detections=[local_detection],
            survey_id="SURV_GEO_TEST",
            geo_service=geo_service,
            image_width=640,
            image_height=640
        )
        assert len(local_contacts) == 1
        assert local_contacts[0].latitude is not None
        assert local_contacts[0].longitude is not None
        assert local_contacts[0].localization_status == "ESTIMATED"
        assert local_contacts[0].model_name == "DRISHTI-YOLO11n"

    def test_8_zero_leak_security(self):
        """
        Test 8 — HF token is never leaked in error messages, logs, or string representations.
        """
        secret_token = "hf_SUPERSECRETTOKEN123456789"
        raw_error = f"Unauthorized access using token {secret_token} to private endpoint"
        sanitized = _sanitize_error_message(raw_error, token=secret_token)
        assert secret_token not in sanitized
        assert "[REDACTED_HF_TOKEN]" in sanitized
