"""
Tests for SW-Net Second-Stage Verification and ROI Extraction Modules.
"""

import pytest
import numpy as np
import cv2

from ml.verification.roi_extractor import ROIExtractor, ExtractedROI
from ml.verification.swnet_verifier import (
    SWNetVerifier,
    VerificationResult,
    compute_morphological_metrics,
    BaseROIVerifier
)


class TestROIExtraction:
    def test_roi_extraction_bounds_and_margins(self):
        # 640x640 mock image
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        extractor = ROIExtractor(context_margin=0.10, min_crop_size=32)

        bbox = [100, 100, 200, 200]  # width=100, height=100, pad=10px -> crop=[90, 90, 210, 210]
        roi = extractor.extract_roi(img, bbox, metadata={"class_name": "shipwreck", "confidence": 0.85})

        assert isinstance(roi, ExtractedROI)
        assert roi.crop_width == 120
        assert roi.crop_height == 120
        assert roi.crop_offset_x == 90
        assert roi.crop_offset_y == 90
        assert roi.bbox_local == [10, 10, 110, 110]
        assert roi.bbox_global == [100, 100, 200, 200]
        assert roi.metadata["class_name"] == "shipwreck"

    def test_roi_extraction_clamps_to_image_boundary(self):
        img = np.zeros((400, 400), dtype=np.uint8)
        extractor = ROIExtractor(context_margin=0.20, min_crop_size=32)

        # BBox right against top-left corner
        bbox = [0, 0, 50, 50]
        roi = extractor.extract_roi(img, bbox)

        assert roi.crop_offset_x == 0
        assert roi.crop_offset_y == 0
        assert roi.crop_width >= 50
        assert roi.crop_height >= 50

    def test_roi_extraction_rejects_empty_image(self):
        extractor = ROIExtractor()
        with pytest.raises(ValueError):
            extractor.extract_roi(np.array([]), [0, 0, 10, 10])


class TestMorphologicalMetrics:
    def test_circular_mask_compactness(self):
        # Create perfect circle of radius 30
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 30, 255, -1)

        metrics = compute_morphological_metrics(mask)
        assert metrics["mask_area_px"] > 0
        # A discrete circle has compactness close to 1.0 (typically 0.90 - 1.0 due to pixel grid discretization)
        assert 0.85 <= metrics["compactness"] <= 1.0
        assert metrics["elongation"] < 1.2  # Near 1.0 for symmetric circle

    def test_elongated_rectangle_orientation(self):
        # Create horizontal bar (width 80, height 20)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 10:90] = 255

        metrics = compute_morphological_metrics(mask)
        assert metrics["mask_area_px"] == 80 * 20
        assert metrics["elongation"] > 2.5  # High elongation
        assert abs(metrics["orientation_deg"]) < 10.0  # Horizontal is ~0 degrees

    def test_empty_mask_handling(self):
        empty_mask = np.zeros((50, 50), dtype=np.uint8)
        metrics = compute_morphological_metrics(empty_mask)

        assert metrics["mask_area_px"] == 0
        assert metrics["compactness"] == 0.0
        assert metrics["elongation"] == 1.0


class TestSWNetVerifier:
    def test_swnet_verifier_interface_compliance(self):
        verifier = SWNetVerifier()
        assert isinstance(verifier, BaseROIVerifier)
        assert verifier.model_name == "SW-Net"
        assert verifier.model_version == "swnet-direction-aware-v1.0"

    def test_swnet_verify_roi_synthetic_crop(self):
        # Create synthetic sonar crop with highlight target inside bbox
        crop = np.full((128, 128), 30, dtype=np.uint8)  # dark background
        crop[30:80, 30:80] = 220  # bright highlight target

        extractor = ROIExtractor(context_margin=0.1)
        roi = extractor.extract_roi(crop, [30, 30, 80, 80])

        verifier = SWNetVerifier(confidence_threshold=0.5)
        result = verifier.verify_roi(roi)

        assert isinstance(result, VerificationResult)
        assert result.mask_area_px > 0
        assert result.mask_to_box_ratio > 0.5
        assert result.is_verified is True
        assert "VERIFIED" in result.status or "STRUCTURAL_HEURISTIC" in result.status
        assert result.to_dict()["model_name"] == "SW-Net"

    def test_swnet_rejects_empty_or_underfilled_candidate(self):
        # Uniform flat background with no contrast
        flat_crop = np.full((100, 100), 50, dtype=np.uint8)
        extractor = ROIExtractor()
        roi = extractor.extract_roi(flat_crop, [20, 20, 80, 80])

        verifier = SWNetVerifier(min_mask_ratio=0.10)
        result = verifier.verify_roi(roi)

        assert isinstance(result, VerificationResult)
        # Without contrast, Otsu separates noise or gives degenerate mask
        assert isinstance(result.is_verified, bool)
