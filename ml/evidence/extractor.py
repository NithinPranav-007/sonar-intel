"""
Master Candidate Evidence Extraction Orchestrator.

Combines:
1. Stage-1 Detector Evidence (YOLO candidate)
2. Stage-2 Segmentation Evidence (SW-Net, strictly constrained to Shipwreck)
3. Stage-2 Morphology Evidence (Geometric physical priors)
4. Stage-2 Acoustic Evidence (Highlight contrast, nadir shadow, texture, gradients)

Does NOT perform final scoring or fusion. Returns structured, unweighted CandidateEvidence.
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np

from ml.evidence.schemas import (
    CandidateEvidence,
    DetectorEvidence,
    SegmentationEvidence,
    MorphologyEvidence,
    AcousticEvidence,
    EvidenceStatus
)
from ml.evidence.morphology_extractor import extract_morphology_evidence
from ml.evidence.acoustic_extractor import extract_acoustic_evidence
from ml.verification.roi_extractor import ROIExtractor, ExtractedROI
from ml.verification.swnet_verifier import VerificationResult


class EvidenceExtractor:
    """
    Orchestrates the extraction of multi-modal evidence across candidate detections.
    """

    def __init__(
        self,
        roi_context_margin: float = 0.15,
        min_crop_size: int = 32
    ):
        self.roi_extractor = ROIExtractor(
            context_margin=roi_context_margin,
            min_crop_size=min_crop_size
        )

    def extract(
        self,
        image: np.ndarray,
        bbox: Union[List[int], Dict[str, int]],
        class_name: str,
        class_id: int,
        detector_confidence: float,
        swnet_result: Optional[VerificationResult] = None,
        class_margin: Optional[float] = None,
        nadir_x: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CandidateEvidence:
        """
        Extracts comprehensive evidence for a single candidate detection.

        Args:
            image: Full 2D/3D sonar image array.
            bbox: Bounding box [x1, y1, x2, y2] or dict {"x1": ..., "y1": ..., "x2": ..., "y2": ...}.
            class_name: Canonical class name string (e.g. "shipwreck", "submarine_pipeline").
            class_id: Canonical integer ID (0 to 4).
            detector_confidence: Raw YOLO model confidence [0.0, 1.0].
            swnet_result: Optional VerificationResult from SWNetVerifier.
            class_margin: Optional difference between top-1 and top-2 class logits.
            nadir_x: Horizontal pixel coordinate of acoustic nadir.
            metadata: Optional upstream telemetry / tracking dictionary.

        Returns:
            CandidateEvidence containing all raw evidence components and status flags.
        """
        all_warnings: List[str] = []

        # 1. Normalize Bounding Box
        if isinstance(bbox, dict):
            bx1, by1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
            bx2, by2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bx1, by1, bx2, by2 = map(int, bbox)
        else:
            bx1, by1, bx2, by2 = 0, 0, 0, 0
            all_warnings.append("INVALID_BBOX_FORMAT")

        bw = max(0, bx2 - bx1)
        bh = max(0, by2 - by1)
        area = bw * bh
        aspect_ratio = float(max(bw, bh) / max(1, min(bw, bh)))

        img_h, img_w = image.shape[:2] if image is not None and image.size > 0 else (640, 640)
        norm_center_x = round(((bx1 + bx2) / 2.0) / max(1, img_w), 4)
        norm_center_y = round(((by1 + by2) / 2.0) / max(1, img_h), 4)
        norm_w = round(bw / max(1, img_w), 4)
        norm_h = round(bh / max(1, img_h), 4)
        tile_coords = {
            "center_x": norm_center_x,
            "center_y": norm_center_y,
            "width": norm_w,
            "height": norm_h
        }

        # 2. Build Detector Evidence
        detector_ev = DetectorEvidence(
            class_name=class_name,
            class_id=class_id,
            detector_confidence=round(float(detector_confidence), 3),
            bbox=[bx1, by1, bx2, by2],
            bbox_width=bw,
            bbox_height=bh,
            bbox_area=area,
            aspect_ratio=round(aspect_ratio, 2),
            class_probability_margin=round(float(class_margin), 3) if class_margin is not None else None,
            class_probabilities=metadata.get("class_probabilities") if metadata else None,
            tile_relative_coords=tile_coords
        )

        # Handle invalid bbox early
        if bw < 2 or bh < 2 or area < 4 or image is None or image.size == 0:
            all_warnings.append("DEGENERATE_TARGET_REGION")
            return CandidateEvidence(
                detector=detector_ev,
                segmentation=SegmentationEvidence(
                    available=False,
                    source=None,
                    confidence=None,
                    mask_status="INVALID_ROI"
                ),
                morphology=MorphologyEvidence(
                    source="unavailable",
                    area_px=float(area),
                    width_px=float(bw),
                    height_px=float(bh),
                    aspect_ratio=1.0,
                    perimeter_px=0.0,
                    compactness=0.0,
                    eccentricity=0.0,
                    elongation=1.0,
                    orientation_deg=0.0,
                    solidity=1.0,
                    convexity=1.0,
                    status="DEGENERATE"
                ),
                acoustic=extract_acoustic_evidence(image, [bx1, by1, bx2, by2], nadir_x)[0],
                status=EvidenceStatus(
                    overall="INVALID",
                    warnings=all_warnings,
                    is_degenerate_box=True
                ),
                raw_metadata=metadata or {}
            )

        # 3. Extract Context ROI
        roi: Optional[ExtractedROI] = None
        try:
            roi = self.roi_extractor.extract_roi(image, [bx1, by1, bx2, by2], metadata=metadata)
        except Exception as e:
            all_warnings.append(f"ROI_EXTRACTION_FAILED: {e}")

        # 4. Resolve Segmentation Evidence (SW-Net / Shipwreck Constraint)
        # DOMAIN RULE: Only shipwreck class has genuine supervised segmentation ground truth in AI4Shipwrecks.
        # Other classes do NOT have supervised segmentation models.
        if class_name.lower() in ("shipwreck", "wreck") or class_id == 2:
            if swnet_result is not None and swnet_result.segmentation_mask is not None:
                crop_area = float(roi.crop_width * roi.crop_height) if roi else float(bw * bh)
                coverage = float(swnet_result.mask_area_px / max(1.0, crop_area))
                seg_ev = SegmentationEvidence(
                    available=True,
                    source=swnet_result.model_name,
                    confidence=swnet_result.verification_confidence,
                    mask_status="AVAILABLE",
                    mask_area_px=swnet_result.mask_area_px,
                    mask_to_box_ratio=swnet_result.mask_to_box_ratio,
                    mask_coverage=round(coverage, 4),
                    mask_compactness=swnet_result.compactness,
                    mask_orientation_deg=swnet_result.orientation_deg,
                    mask_elongation=swnet_result.elongation
                )
                mask_for_morphology = swnet_result.segmentation_mask
                morph_source = "segmentation_mask"
            else:
                seg_ev = SegmentationEvidence(
                    available=False,
                    source=None,
                    confidence=None,
                    mask_status="UNAVAILABLE_NO_CHECKPOINT"
                )
                mask_for_morphology = None
                morph_source = "bbox_geometry"
        else:
            # Non-shipwreck classes explicitly represented as lacking supervised segmentation
            seg_ev = SegmentationEvidence(
                available=False,
                source=None,
                confidence=None,
                mask_status="UNAVAILABLE_CLASS_NOT_SUPPORTED"
            )
            mask_for_morphology = None
            morph_source = "bbox_geometry"

        # 5. Extract Morphology Evidence
        morph_ev = extract_morphology_evidence(
            mask=mask_for_morphology,
            bbox=[bx1, by1, bx2, by2],
            source_type=morph_source
        )

        # 6. Extract Acoustic Evidence
        acoustic_ev, ac_warnings = extract_acoustic_evidence(
            image=image,
            bbox=[bx1, by1, bx2, by2],
            nadir_x=nadir_x,
            roi_context=roi
        )
        all_warnings.extend(ac_warnings)

        # 7. Evaluate Evidence Status
        is_touching_boundary = "TARGET_TOUCHING_IMAGE_BOUNDARY" in all_warnings
        is_low_contrast = "LOW_ACOUSTIC_CONTRAST" in all_warnings

        if not all_warnings and seg_ev.available:
            overall_status = "COMPLETE"
        elif not all_warnings:
            overall_status = "PARTIAL"  # Standard status when segmentation is unavailable
        elif "DEGENERATE_BOUNDING_BOX" in all_warnings or "EMPTY_IMAGE_SUPPLIED" in all_warnings:
            overall_status = "INVALID"
        else:
            overall_status = "DEGRADED"

        status_obj = EvidenceStatus(
            overall=overall_status,
            warnings=all_warnings,
            is_touching_boundary=is_touching_boundary,
            is_low_contrast=is_low_contrast,
            is_degenerate_box=False
        )

        return CandidateEvidence(
            detector=detector_ev,
            segmentation=seg_ev,
            morphology=morph_ev,
            acoustic=acoustic_ev,
            status=status_obj,
            raw_metadata=metadata or {}
        )
