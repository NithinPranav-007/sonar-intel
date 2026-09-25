"""
Comprehensive Test Suite for the SONAR-INTEL Evidence Extraction Layer.

Validates the 10 required domain scenarios:
1. Valid candidate extraction
2. Invalid/degenerate bounding box
3. Candidate at image boundary
4. Missing segmentation (non-shipwreck class)
5. Segmentation available (shipwreck with SW-Net result)
6. Shadow present
7. Shadow absent
8. Shadow uncertain
9. Low acoustic contrast
10. Empty / invalid ROI and image
11. Real sonar image from repository
"""

import os
import pytest
import numpy as np
import cv2

from ml.evidence.schemas import (
    CandidateEvidence,
    DetectorEvidence,
    SegmentationEvidence,
    MorphologyEvidence,
    AcousticEvidence,
    EvidenceStatus
)
from ml.evidence.extractor import EvidenceExtractor
from ml.evidence.acoustic_extractor import (
    extract_acoustic_evidence,
    compute_shannon_entropy,
    compute_directional_consistency
)
from ml.evidence.morphology_extractor import extract_morphology_evidence
from ml.evidence.normalization import (
    normalize_highlight_contrast,
    normalize_shadow_deficit,
    normalize_entropy,
    normalize_gradient_magnitude
)
from ml.verification.swnet_verifier import VerificationResult


@pytest.fixture
def extractor():
    return EvidenceExtractor()


@pytest.fixture
def synthetic_sonar_image():
    """Creates a 640x640 synthetic sonar image with ambient background and nadir center."""
    img = np.full((640, 640), 60, dtype=np.uint8)  # Ambient seabed level = 60
    # Starboard target highlight at [350, 200, 400, 250] (intensity 220)
    img[200:250, 350:400] = 220
    # Corresponding down-range acoustic shadow at [400, 200, 460, 250] (intensity 10)
    img[200:250, 400:460] = 10
    return img


class TestEvidenceExtraction:

    def test_1_valid_candidate(self, extractor, synthetic_sonar_image):
        """Scenario 1: Standard valid candidate detection."""
        evidence = extractor.extract(
            image=synthetic_sonar_image,
            bbox=[350, 200, 400, 250],
            class_name="shipwreck",
            class_id=2,
            detector_confidence=0.88,
            nadir_x=320
        )

        assert isinstance(evidence, CandidateEvidence)
        assert evidence.detector.class_name == "shipwreck"
        assert evidence.detector.detector_confidence == 0.88
        assert evidence.detector.bbox_width == 50
        assert evidence.detector.bbox_height == 50
        assert evidence.acoustic.highlight_mean > 150.0
        assert evidence.acoustic.shadow_status == "PRESENT"
        assert evidence.status.overall in ("PARTIAL", "COMPLETE")

    def test_2_invalid_bounding_box(self, extractor, synthetic_sonar_image):
        """Scenario 2: Degenerate/inverted bounding box."""
        evidence = extractor.extract(
            image=synthetic_sonar_image,
            bbox=[100, 100, 100, 100],  # 0 width, 0 height
            class_name="submarine_pipeline",
            class_id=1,
            detector_confidence=0.45
        )

        assert evidence.status.overall == "INVALID"
        assert evidence.status.is_degenerate_box is True
        assert "DEGENERATE_TARGET_REGION" in evidence.status.warnings
        assert evidence.morphology.status == "DEGENERATE"

    def test_3_candidate_at_image_boundary(self, extractor):
        """Scenario 3: Candidate touching or truncated at the image border."""
        img = np.full((400, 400), 80, dtype=np.uint8)
        img[0:40, 0:40] = 200  # Target at top-left corner (0,0)

        evidence = extractor.extract(
            image=img,
            bbox=[0, 0, 40, 40],
            class_name="mine_like_contact",
            class_id=4,
            detector_confidence=0.75,
            nadir_x=200
        )

        assert evidence.status.is_touching_boundary is True
        assert "TARGET_TOUCHING_IMAGE_BOUNDARY" in evidence.status.warnings
        assert evidence.status.overall == "DEGRADED"

    def test_4_missing_segmentation_for_non_shipwreck(self, extractor, synthetic_sonar_image):
        """Scenario 4: Non-shipwreck classes explicitly marked as lacking supervised segmentation."""
        evidence = extractor.extract(
            image=synthetic_sonar_image,
            bbox=[350, 200, 400, 250],
            class_name="submarine_pipeline",
            class_id=1,
            detector_confidence=0.92
        )

        assert evidence.segmentation.available is False
        assert evidence.segmentation.mask_status == "UNAVAILABLE_CLASS_NOT_SUPPORTED"
        assert evidence.morphology.source == "bbox_geometry"
        assert evidence.morphology.status == "ESTIMATED_FROM_BBOX"

    def test_5_segmentation_available_with_swnet(self, extractor, synthetic_sonar_image):
        """Scenario 5: Shipwreck candidate with SW-Net verification result."""
        # Create mock 50x50 mask inside candidate
        mock_mask = np.zeros((50, 50), dtype=np.uint8)
        mock_mask[10:40, 10:40] = 255

        mock_swnet = VerificationResult(
            is_verified=True,
            verification_confidence=0.85,
            mask_area_px=900,
            mask_to_box_ratio=0.36,
            orientation_deg=15.0,
            elongation=1.0,
            compactness=0.78,
            highlight_mean=210.0,
            shadow_mean=20.0,
            contrast_ratio=10.5,
            model_name="SW-Net",
            model_version="swnet-v1.0",
            status="MODEL_INFERRED: VERIFIED",
            segmentation_mask=mock_mask
        )

        evidence = extractor.extract(
            image=synthetic_sonar_image,
            bbox=[350, 200, 400, 250],
            class_name="shipwreck",
            class_id=2,
            detector_confidence=0.91,
            swnet_result=mock_swnet,
            nadir_x=320
        )

        assert evidence.segmentation.available is True
        assert evidence.segmentation.mask_status == "AVAILABLE"
        assert evidence.segmentation.source == "SW-Net"
        assert evidence.morphology.source == "segmentation_mask"
        assert evidence.morphology.status == "VALID"
        assert evidence.morphology.area_px == 900.0

    def test_6_shadow_present(self, extractor, synthetic_sonar_image):
        """Scenario 6: Clear down-range shadow behind highlight."""
        evidence = extractor.extract(
            image=synthetic_sonar_image,
            bbox=[350, 200, 400, 250],
            class_name="shipwreck",
            class_id=2,
            detector_confidence=0.80,
            nadir_x=320
        )

        assert evidence.acoustic.shadow_status == "PRESENT"
        assert evidence.acoustic.shadow_deficit > 15.0
        assert evidence.acoustic.shadow_contrast_ratio > 1.5

    def test_7_shadow_absent(self, extractor):
        """Scenario 7: Flat target with highlight but no down-range acoustic shadow."""
        img = np.full((640, 640), 70, dtype=np.uint8)
        img[200:250, 350:400] = 190  # Highlight
        # No shadow region (ambient continues at 70)

        evidence = extractor.extract(
            image=img,
            bbox=[350, 200, 400, 250],
            class_name="crab_pot",
            class_id=0,
            detector_confidence=0.65,
            nadir_x=320
        )

        assert evidence.acoustic.shadow_status == "ABSENT"
        assert evidence.acoustic.shadow_deficit < 5.0

    def test_8_shadow_uncertain(self, extractor):
        """Scenario 8: Ambiguous / intermediate shadow deficit."""
        img = np.full((640, 640), 70, dtype=np.uint8)
        img[200:250, 350:400] = 190  # Highlight
        img[200:250, 400:440] = 62   # Very subtle deficit (8 intensity levels)

        evidence = extractor.extract(
            image=img,
            bbox=[350, 200, 400, 250],
            class_name="ghost_net",
            class_id=3,
            detector_confidence=0.70,
            nadir_x=320
        )

        assert evidence.acoustic.shadow_status == "UNCERTAIN"

    def test_9_low_contrast(self, extractor):
        """Scenario 9: Low acoustic contrast between target and ambient seabed."""
        img = np.full((500, 500), 100, dtype=np.uint8)
        img[150:200, 150:200] = 104  # Only 4 intensity levels contrast

        evidence = extractor.extract(
            image=img,
            bbox=[150, 150, 200, 200],
            class_name="ghost_net",
            class_id=3,
            detector_confidence=0.50
        )

        assert evidence.status.is_low_contrast is True
        assert "LOW_ACOUSTIC_CONTRAST" in evidence.status.warnings

    def test_10_empty_and_invalid_image(self, extractor):
        """Scenario 10: Empty or null image array."""
        empty_img = np.array([], dtype=np.uint8)
        evidence = extractor.extract(
            image=empty_img,
            bbox=[10, 10, 50, 50],
            class_name="shipwreck",
            class_id=2,
            detector_confidence=0.80
        )

        assert evidence.status.overall == "INVALID"
        assert "DEGENERATE_TARGET_REGION" in evidence.status.warnings


class TestFeatureNormalization:
    def test_normalization_bounds(self):
        assert normalize_highlight_contrast(1.0) == 0.0
        assert normalize_highlight_contrast(3.0) == 1.0
        assert 0.0 <= normalize_highlight_contrast(2.0) <= 1.0

        assert normalize_shadow_deficit(0.0) == 0.0
        assert normalize_shadow_deficit(60.0) == 1.0

        assert normalize_entropy(1.0) == 0.0
        assert normalize_entropy(8.0) == 1.0

        assert normalize_gradient_magnitude(0.0) == 0.0
        assert normalize_gradient_magnitude(50.0) == 1.0


class TestRealSonarImageEvidence:
    def test_evidence_extraction_on_real_repo_image(self, extractor):
        """Validates evidence extraction on a real sonar image from the repository."""
        # Find a real image in data/raw/
        raw_candidates = [
            "data/raw/DEMO_CORSICAN_02_1788277758_corsican_02_test_wreck.png",
            "data/raw/SURVEY_001_raw.png"
        ]
        chosen_path = None
        for p in raw_candidates:
            if os.path.exists(p):
                chosen_path = p
                break

        if chosen_path is None:
            pytest.skip("No raw test image found in data/raw/")

        img = cv2.imread(chosen_path, cv2.IMREAD_GRAYSCALE)
        assert img is not None

        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        bbox = [center_x - 30, center_y - 30, center_x + 30, center_y + 30]

        evidence = extractor.extract(
            image=img,
            bbox=bbox,
            class_name="shipwreck",
            class_id=2,
            detector_confidence=0.78,
            nadir_x=w // 2
        )

        assert isinstance(evidence, CandidateEvidence)
        assert evidence.detector.bbox_width == 60
        assert evidence.detector.bbox_height == 60
        assert evidence.acoustic.highlight_mean > 0
        assert evidence.acoustic.entropy > 0.0
        assert isinstance(evidence.to_dict(), dict)


class TestMorphologyAndAcousticEdgeCases:
    """
    Focused unit tests for specific edge cases in morphology and acoustic evidence extraction.
    """

    def test_morphology_exact_features_on_synthetic_shape(self):
        """Validates exact calculation of all 8 morphology features on a known geometric rectangle."""
        # Create 100x100 mask with a 40x20 rectangle
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:50, 20:60] = 255  # height=20, width=40 -> Area = 800

        morph = extract_morphology_evidence(mask=mask, bbox=[20, 30, 60, 50], source_type="segmentation_mask")

        assert morph.status == "VALID"
        assert morph.source == "segmentation_mask"
        assert morph.area_px == 800.0
        assert morph.width_px == 40.0
        assert morph.height_px == 20.0
        assert morph.aspect_ratio == 2.0
        assert morph.perimeter_px > 100.0  # Approx 2*(40+20) = 120
        assert 0.0 < morph.compactness <= 1.0
        assert 0.0 <= morph.eccentricity <= 1.0
        assert morph.elongation >= 1.5
        assert morph.solidity >= 0.95  # Solid rectangle

    def test_empty_mask_handling(self):
        """Verifies graceful fallback when mask is all zeros."""
        empty_mask = np.zeros((64, 64), dtype=np.uint8)
        morph = extract_morphology_evidence(mask=empty_mask, bbox=[10, 10, 50, 50], source_type="auto")

        assert morph.source == "bbox_geometry"
        assert morph.status == "ESTIMATED_FROM_BBOX"
        assert morph.area_px == 1600.0
        assert morph.aspect_ratio == 1.0

    def test_missing_mask_graceful_handling(self, extractor, synthetic_sonar_image):
        """Verifies extractor produces valid evidence without SW-Net mask."""
        evidence = extractor.extract(
            image=synthetic_sonar_image,
            bbox=[350, 200, 400, 250],
            class_name="mine_like_contact",
            class_id=0,
            detector_confidence=0.82,
            swnet_result=None,
            nadir_x=320
        )

        assert evidence.segmentation.available is False
        assert evidence.segmentation.mask_status == "UNAVAILABLE_CLASS_NOT_SUPPORTED"
        assert evidence.morphology.source == "bbox_geometry"
        assert evidence.morphology.status == "ESTIMATED_FROM_BBOX"
        assert evidence.acoustic.highlight_mean > 0.0

    def test_flat_uniform_roi_acoustic_features(self, extractor):
        """Verifies behavior on perfectly uniform, zero-contrast flat patch."""
        flat_img = np.full((300, 300), 100, dtype=np.uint8)

        evidence = extractor.extract(
            image=flat_img,
            bbox=[100, 100, 150, 150],
            class_name="shipwreck",
            class_id=2,
            detector_confidence=0.50,
            nadir_x=150
        )

        assert evidence.acoustic.local_variance == 0.0
        assert evidence.acoustic.entropy == 0.0
        assert evidence.acoustic.edge_density == 0.0
        assert evidence.acoustic.shadow_status in ("ABSENT", "UNCERTAIN")
        assert evidence.status.is_low_contrast is True

    def test_noisy_roi_acoustic_features(self, extractor):
        """Verifies texture entropy, variance, and edge density on noisy texture patch."""
        np.random.seed(42)
        noisy_img = np.random.randint(40, 220, size=(300, 300), dtype=np.uint8)

        evidence = extractor.extract(
            image=noisy_img,
            bbox=[80, 80, 160, 160],
            class_name="shipwreck",
            class_id=2,
            detector_confidence=0.60,
            nadir_x=150
        )

        assert evidence.acoustic.local_variance > 500.0
        assert evidence.acoustic.entropy > 6.0  # High Shannon entropy
        assert evidence.acoustic.edge_density > 0.0

    def test_degenerate_geometry_handling(self, extractor):
        """Verifies single-pixel, inverted, or out-of-bounds bounding boxes are flagged as INVALID."""
        img = np.full((200, 200), 80, dtype=np.uint8)

        # Single pixel bbox
        ev_single = extractor.extract(
            image=img,
            bbox=[50, 50, 51, 51],
            class_name="debris",
            class_id=4,
            detector_confidence=0.30
        )
        assert ev_single.status.overall == "INVALID"
        assert ev_single.status.is_degenerate_box is True

        # Inverted bbox
        ev_inv = extractor.extract(
            image=img,
            bbox=[100, 100, 50, 50],
            class_name="debris",
            class_id=4,
            detector_confidence=0.30
        )
        assert ev_inv.status.overall == "INVALID"
