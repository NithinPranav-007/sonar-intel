"""
Machine-Readable Contact Package Generation and Output Contract.

Generates self-contained, fully auditable Contact Package objects for
Person 3 / Frontend Dashboard / PostGIS ingestion.

Enforces:
- Strict preservation of raw detector score, fused score, and calibrated confidence
- Zero coordinate fabrication (returns location.status = 'UNAVAILABLE' when navigation is absent)
- Modality-level score breakdown for operator presentation (Detector %, Segmentation %, Acoustic %, Morphology %)
- Explicit representation of missing or unavailable evidence sources
"""

from typing import Dict, Any, Optional, List, Union, Literal, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json

from ml.evidence.schemas import CandidateEvidence
from ml.fusion.evidence_fusion import FusedEvidenceResult
from ml.fusion.calibration import BaseCalibrator, PlattCalibrator, IdentityCalibrator


@dataclass
class ContactLocation:
    """Georeferenced location metadata for the contact."""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth_m: Optional[float] = None
    uncertainty_m: Optional[float] = None
    status: Literal["ESTIMATED", "VERIFIED", "UNCERTAIN", "UNAVAILABLE"] = "UNAVAILABLE"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContactAssets:
    """Pointers and bounding metadata for cropped visual artifacts."""
    roi_bounds: List[int]                     # [x1, y1, x2, y2] in swath coordinates
    crop_offset: List[int]                    # [offset_x, offset_y]
    crop_size: List[int]                      # [crop_width, crop_height]
    roi_ref: Optional[str] = None             # Optional URI/filepath to ROI image artifact
    mask_ref: Optional[str] = None            # Optional URI/filepath to segmentation mask artifact

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelProvenance:
    """Full architectural provenance and model versioning."""
    detector_name: str = "DRISHTI-YOLOv8s"
    detector_version: str = "baseline-v1"
    swnet_version: Optional[str] = "swnet-direction-aware-v1.0"
    fusion_version: str = "baseline_adaptive_linear_v1"
    calibration_version: str = "platt-mvp-v1.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OperatorIndicators:
    """Summary indicators tailored for hydrographer/operator UI presentation."""
    acoustic: Dict[str, Any]
    morphological: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContactPackage:
    """
    Authoritative machine-readable Contact Package Contract.
    The primary handoff payload delivered to Person 3 (Frontend / API).
    """
    contact_id: str                           # Unique contact identifier, e.g. "CNT-000001"
    survey_id: str                            # Parent survey swath identifier
    timestamp: str                            # ISO-8601 UTC detection timestamp
    class_name: str                           # Canonical class name (e.g. "mine_like_contact", "shipwreck")
    class_id: int                             # Canonical class ID (0 to 4)
    raw_detector_confidence: float            # Raw YOLO model score [0.0, 1.0]
    fused_score: float                        # Uncalibrated multi-modal fusion score [0.0, 1.0]
    final_confidence: float                   # Post-calibration confidence probability [0.0, 1.0]
    bbox: List[int]                           # Bounding box [x1, y1, x2, y2]
    location: ContactLocation
    evidence: Dict[str, Any]                  # Deep evidence breakdown
    modality_breakdown: Dict[str, Optional[float]] # Operator percentage breakdown (Detector %, Seg %, Acoustic %, Morph %)
    indicators: OperatorIndicators
    assets: ContactAssets
    model: ModelProvenance
    priority: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    review_status: Literal["AI_CANDIDATE", "CONFIRMED", "FALSE_POSITIVE", "UNCERTAIN"] = "AI_CANDIDATE"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class ContactPackageBuilder:
    """
    Constructs ContactPackage objects from fused candidate evidence and georeferencing services.
    """

    def __init__(
        self,
        calibrator: Optional[BaseCalibrator] = None,
        model_provenance: Optional[ModelProvenance] = None
    ):
        self.calibrator = calibrator or PlattCalibrator()
        self.model_provenance = model_provenance or ModelProvenance(
            calibration_version=self.calibrator.get_metadata().get("version", "platt-mvp-v1.0")
        )

    def build_package(
        self,
        contact_id: str,
        survey_id: str,
        evidence: CandidateEvidence,
        fused_result: FusedEvidenceResult,
        geo_location: Optional[Tuple[Optional[float], Optional[float], Optional[float], str]] = None,
        roi_assets: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None
    ) -> ContactPackage:
        """
        Builds a comprehensive ContactPackage.

        Args:
            contact_id: Unique contact identifier string.
            survey_id: Parent survey swath identifier.
            evidence: CandidateEvidence from master EvidenceExtractor.
            fused_result: FusedEvidenceResult from EvidenceFusionEngine.
            geo_location: Tuple of (latitude, longitude, depth_m, status_str).
            roi_assets: Optional asset reference paths/bounds.
            timestamp: Optional ISO-8601 timestamp string.

        Returns:
            Fully populated, verified ContactPackage.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        # 1. Calibrate Fused Score
        calibrated_conf = self.calibrator.calibrate(fused_result.fused_score)

        # 2. Geolocation Resolution (Zero Coordinate Fabrication Rule)
        lat, lon, depth, loc_status = None, None, None, "UNAVAILABLE"
        if geo_location is not None and len(geo_location) >= 4:
            g_lat, g_lon, g_depth, g_status = geo_location
            if g_lat is not None and g_lon is not None:
                lat = round(float(g_lat), 6)
                lon = round(float(g_lon), 6)
                depth = round(float(g_depth), 2) if g_depth is not None else None
                loc_status = g_status if g_status in ("ESTIMATED", "VERIFIED", "UNCERTAIN") else "ESTIMATED"
            else:
                loc_status = "UNAVAILABLE"

        location_obj = ContactLocation(
            latitude=lat,
            longitude=lon,
            depth_m=depth,
            uncertainty_m=1.5 if lat is not None else None,
            status=loc_status
        )

        # 3. Resolve Assets
        bx1, by1, bx2, by2 = evidence.detector.bbox
        if roi_assets is not None:
            assets_obj = ContactAssets(
                roi_bounds=roi_assets.get("roi_bounds", [bx1, by1, bx2, by2]),
                crop_offset=roi_assets.get("crop_offset", [bx1, by1]),
                crop_size=roi_assets.get("crop_size", [bx2 - bx1, by2 - by1]),
                roi_ref=roi_assets.get("roi_ref"),
                mask_ref=roi_assets.get("mask_ref")
            )
        else:
            assets_obj = ContactAssets(
                roi_bounds=[bx1, by1, bx2, by2],
                crop_offset=[bx1, by1],
                crop_size=[max(1, bx2 - bx1), max(1, by2 - by1)]
            )

        # 4. Operator Presentation Indicators
        indicators = OperatorIndicators(
            acoustic={
                "highlight_mean": evidence.acoustic.highlight_mean,
                "local_contrast_ratio": evidence.acoustic.local_contrast_ratio,
                "shadow_status": evidence.acoustic.shadow_status,
                "shadow_deficit": evidence.acoustic.shadow_deficit,
                "texture_entropy": evidence.acoustic.entropy,
                "edge_density": evidence.acoustic.edge_density
            },
            morphological={
                "aspect_ratio": evidence.morphology.aspect_ratio,
                "elongation": evidence.morphology.elongation,
                "orientation_deg": evidence.morphology.orientation_deg,
                "compactness": evidence.morphology.compactness,
                "solidity": evidence.morphology.solidity,
                "source": evidence.morphology.source
            }
        )

        # 5. Operational Triage Priority Assignment
        if calibrated_conf >= 0.70 or fused_result.fused_score >= 0.75:
            priority = "HIGH"
        elif calibrated_conf >= 0.40 or fused_result.fused_score >= 0.45:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        # 6. Structured Evidence Payload
        evidence_dict = {
            "detector": evidence.detector.to_dict(),
            "segmentation": evidence.segmentation.to_dict(),
            "morphology": evidence.morphology.to_dict(),
            "acoustic": evidence.acoustic.to_dict(),
            "fused": fused_result.to_dict(),
            "calibrated_confidence": round(calibrated_conf, 3),
            "status": evidence.status.to_dict()
        }

        return ContactPackage(
            contact_id=contact_id,
            survey_id=survey_id,
            timestamp=timestamp,
            class_name=evidence.detector.class_name,
            class_id=evidence.detector.class_id,
            raw_detector_confidence=round(evidence.detector.detector_confidence, 3),
            fused_score=round(fused_result.fused_score, 3),
            final_confidence=round(calibrated_conf, 3),
            bbox=evidence.detector.bbox,
            location=location_obj,
            evidence=evidence_dict,
            modality_breakdown=fused_result.modality_scores,
            indicators=indicators,
            assets=assets_obj,
            model=self.model_provenance,
            priority=priority,
            review_status="AI_CANDIDATE"
        )
