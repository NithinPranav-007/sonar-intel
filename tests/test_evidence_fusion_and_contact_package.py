"""
Integration Test Suite for Evidence Fusion, Calibration, and Contact Package Generation.

Validates the Person-3 output contract and error-handling resilience:
1. Normal contact with full evidence
2. Low-confidence candidate handling
3. Missing SW-Net output (3-modality fallback)
4. Missing acoustic shadow / flat target handling
5. Missing geolocation (zero coordinate fabrication rule)
6. Malformed detector input handling
7. Invalid ROI handling
8. Calibration transformation interface
9. Operator breakdown indicators
"""

import pytest
import numpy as np

from ml.evidence.schemas import (
    CandidateEvidence,
    DetectorEvidence,
    SegmentationEvidence,
    MorphologyEvidence,
    AcousticEvidence,
    EvidenceStatus
)
from ml.evidence.extractor import EvidenceExtractor
from ml.fusion.evidence_fusion import EvidenceFusionEngine, FusedEvidenceResult
from ml.fusion.calibration import PlattCalibrator, TemperatureCalibrator, IdentityCalibrator
from ml.fusion.contact_package import (
    ContactPackage,
    ContactPackageBuilder,
    ContactLocation,
    ContactAssets
)
from ml.verification.swnet_verifier import VerificationResult


@pytest.fixture
def extractor():
    return EvidenceExtractor()


@pytest.fixture
def fusion_engine():
    return EvidenceFusionEngine()


@pytest.fixture
def package_builder():
    return ContactPackageBuilder()


@pytest.fixture
def mock_evidence():
    """Generates standard mock CandidateEvidence for testing."""
    return CandidateEvidence(
        detector=DetectorEvidence(
            class_name="mine_like_contact",
            class_id=4,
            detector_confidence=0.88,
            bbox=[100, 100, 150, 150],
            bbox_width=50,
            bbox_height=50,
            bbox_area=2500,
            aspect_ratio=1.0,
            class_probability_margin=0.35,
            class_probabilities={"mine_like_contact": 0.88, "crab_pot": 0.05, "shipwreck": 0.04, "submarine_pipeline": 0.02, "ghost_net": 0.01},
            tile_relative_coords={"center_x": 0.1953, "center_y": 0.1953, "width": 0.0781, "height": 0.0781}
        ),
        segmentation=SegmentationEvidence(
            available=False,
            source=None,
            confidence=None,
            mask_status="UNAVAILABLE_CLASS_NOT_SUPPORTED"
        ),
        morphology=MorphologyEvidence(
            source="bbox_geometry",
            area_px=2500.0,
            width_px=50.0,
            height_px=50.0,
            aspect_ratio=1.0,
            perimeter_px=200.0,
            compactness=0.7854,
            eccentricity=0.0,
            elongation=1.0,
            orientation_deg=0.0,
            solidity=1.0,
            convexity=1.0,
            status="ESTIMATED_FROM_BBOX"
        ),
        acoustic=AcousticEvidence(
            highlight_mean=215.0,
            highlight_max=245.0,
            local_background_mean=60.0,
            local_background_std=10.0,
            local_contrast_ratio=3.583,
            normalized_highlight_strength=0.608,
            shadow_status="PRESENT",
            shadow_mean=10.0,
            shadow_contrast_ratio=6.0,
            shadow_deficit=50.0,
            shadow_candidate_area_px=1500,
            shadow_to_highlight_area_ratio=0.60,
            shadow_length_px=30.0,
            local_variance=380.0,
            entropy=5.60,
            gradient_magnitude_mean=32.0,
            edge_density=0.12,
            dominant_gradient_orientation_deg=10.0,
            directional_consistency=0.72
        ),
        status=EvidenceStatus(overall="PARTIAL", warnings=[])
    )


class TestEvidenceFusion:

    def test_1_normal_contact_package_generation(self, fusion_engine, package_builder, mock_evidence):
        """Scenario 1: Full contact package build with valid telemetry."""
        fused = fusion_engine.fuse(mock_evidence)

        assert isinstance(fused, FusedEvidenceResult)
        assert 0.0 <= fused.fused_score <= 1.0
        assert fused.fusion_status == "DEGRADED_3_MODALITIES"
        assert fused.modality_scores["detector"] == 0.88

        # Build contact package with geo-coordinates
        pkg = package_builder.build_package(
            contact_id="CNT-000001",
            survey_id="SURV-2026-09-01",
            evidence=mock_evidence,
            fused_result=fused,
            geo_location=(45.062123, -83.312456, 18.5, "ESTIMATED")
        )

        assert isinstance(pkg, ContactPackage)
        assert pkg.contact_id == "CNT-000001"
        assert pkg.class_name == "mine_like_contact"
        assert pkg.location.latitude == 45.062123
        assert pkg.location.longitude == -83.312456
        assert pkg.location.depth_m == 18.5
        assert pkg.location.status == "ESTIMATED"
        assert pkg.final_confidence > 0.0
        assert pkg.priority in ("HIGH", "MEDIUM", "LOW")
        assert "detector" in pkg.modality_breakdown
        assert "acoustic" in pkg.modality_breakdown
        assert "morphology" in pkg.modality_breakdown

    def test_2_low_confidence_contact(self, fusion_engine, package_builder, mock_evidence):
        """Scenario 2: Low-confidence candidate assigns LOW priority."""
        mock_evidence.detector.detector_confidence = 0.20
        mock_evidence.acoustic.highlight_mean = 70.0
        mock_evidence.acoustic.local_contrast_ratio = 1.15
        mock_evidence.acoustic.shadow_status = "ABSENT"
        mock_evidence.acoustic.shadow_deficit = 0.0

        fused = fusion_engine.fuse(mock_evidence)
        pkg = package_builder.build_package(
            contact_id="CNT-000002",
            survey_id="SURV-01",
            evidence=mock_evidence,
            fused_result=fused
        )

        assert pkg.final_confidence < 0.40
        assert pkg.priority == "LOW"

    def test_3_missing_swnet_fallback(self, fusion_engine, mock_evidence):
        """Scenario 3: Missing SW-Net output does not crash fusion, rebalances to 3 modalities."""
        assert mock_evidence.segmentation.available is False

        fused = fusion_engine.fuse(mock_evidence)
        assert fused.fusion_status == "DEGRADED_3_MODALITIES"
        assert fused.modality_scores["segmentation"] is None
        assert fused.fused_score > 0.50

    def test_4_four_modality_shipwreck_fusion(self, fusion_engine, package_builder, mock_evidence):
        """Scenario 4: Shipwreck with active SW-Net segmentation mask uses 4-modality fusion."""
        mock_evidence.detector.class_name = "shipwreck"
        mock_evidence.detector.class_id = 2
        mock_evidence.segmentation.available = True
        mock_evidence.segmentation.source = "SW-Net"
        mock_evidence.segmentation.confidence = 0.85
        mock_evidence.segmentation.mask_status = "AVAILABLE"

        fused = fusion_engine.fuse(mock_evidence)
        assert fused.fusion_status == "COMPLETE_4_MODALITIES"
        assert fused.modality_scores["segmentation"] == 0.85
        assert fused.weights_applied["segmentation"] == 0.25

    def test_5_missing_geolocation_zero_fabrication(self, fusion_engine, package_builder, mock_evidence):
        """Scenario 5: Navigation unavailable explicitly returns UNAVAILABLE without fake GPS."""
        fused = fusion_engine.fuse(mock_evidence)
        pkg = package_builder.build_package(
            contact_id="CNT-000005",
            survey_id="SURV-01",
            evidence=mock_evidence,
            fused_result=fused,
            geo_location=None  # Navigation absent
        )

        assert pkg.location.status == "UNAVAILABLE"
        assert pkg.location.latitude is None
        assert pkg.location.longitude is None
        assert pkg.location.depth_m is None

    def test_6_malformed_detector_output(self, extractor, fusion_engine, package_builder):
        """Scenario 6: Malformed/inverted bounding box does not crash pipeline."""
        img = np.full((300, 300), 50, dtype=np.uint8)
        evidence = extractor.extract(
            image=img,
            bbox=[200, 200, 100, 100],  # Inverted
            class_name="ghost_net",
            class_id=3,
            detector_confidence=0.60
        )

        fused = fusion_engine.fuse(evidence)
        pkg = package_builder.build_package(
            contact_id="CNT-000006",
            survey_id="SURV-01",
            evidence=evidence,
            fused_result=fused
        )

        assert evidence.status.overall == "INVALID"
        assert pkg.fused_score == 0.0
        assert pkg.priority == "LOW"

    def test_7_calibration_interface(self):
        """Scenario 7: Validates calibration transformation curves."""
        platt = PlattCalibrator(a=-4.5, b=2.25)
        temp = TemperatureCalibrator(temperature=1.2)
        ident = IdentityCalibrator()

        assert 0.0 <= platt.calibrate(0.5) <= 1.0
        assert 0.0 <= temp.calibrate(0.8) <= 1.0
        assert ident.calibrate(0.75) == 0.75
        assert platt.get_metadata()["calibrator_type"] == "PlattScaling"

    def test_8_json_serialization(self, fusion_engine, package_builder, mock_evidence):
        """Scenario 8: Contact package serializes cleanly to JSON for Person 3."""
        fused = fusion_engine.fuse(mock_evidence)
        pkg = package_builder.build_package(
            contact_id="CNT-000008",
            survey_id="SURV-01",
            evidence=mock_evidence,
            fused_result=fused,
            geo_location=(45.0, -83.0, 12.0, "ESTIMATED")
        )

        json_str = pkg.to_json()
        assert isinstance(json_str, str)
        assert "CNT-000008" in json_str
        assert "mine_like_contact" in json_str

    def test_9_inference_service_full_pipeline_contact_packages(self):
        """Scenario 9: Full end-to-end InferenceService run_survey_contact_packages on real image."""
        import os
        from backend.app.services.inference_service import InferenceService

        service = InferenceService()
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

        packages = service.run_survey_contact_packages(
            survey_id="SURV_TEST_01",
            raw_image_path=chosen_path,
            confidence_threshold=0.25
        )

        assert isinstance(packages, list)
        for pkg in packages:
            assert isinstance(pkg, ContactPackage)
            assert pkg.survey_id == "SURV_TEST_01"
            assert pkg.final_confidence >= 0.0
            assert "detector" in pkg.modality_breakdown
            assert pkg.location.status in ("ESTIMATED", "VERIFIED", "UNCERTAIN", "UNAVAILABLE")

