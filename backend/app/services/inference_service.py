"""
Inference Orchestration Service.

Orchestrates the full end-to-end processing pipeline:
SSS Input -> Data Quality -> DRISHTI Preprocessing -> DRISHTI Detector ->
Tile Deduplication -> Acoustic Context Analysis -> Geolocation -> Priority Scoring ->
Canonical Contact Transformation.

Pluggable and configuration-driven.
"""

from typing import List, Optional, Dict, Any
import os
import cv2
import numpy as np

from backend.app.schemas.contact import Contact, BoundingBox
from backend.app.core.config import settings
from ml.preprocessing.tiling import generate_tiles
from ml.preprocessing.quality import compute_image_quality
from ml.inference.drishti_detector import DrishtiDetector, DrishtiDetection
from ml.inference.postprocess import deduplicate_detections
from ml.inference.context import extract_acoustic_context
from backend.app.services.scoring_service import calculate_contact_priority
from backend.app.services.geolocation_service import GeolocationService
from backend.app.services.transformer import transform_drishti_detections_to_contacts
from ml.verification.swnet_verifier import SWNetVerifier
from ml.evidence.extractor import EvidenceExtractor
from ml.fusion.evidence_fusion import EvidenceFusionEngine
from ml.fusion.contact_package import ContactPackage, ContactPackageBuilder


class InferenceService:
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None
    ):
        self.detector = DrishtiDetector(
            model_path=model_path or settings.MODEL_PATH,
            model_name=settings.MODEL_NAME,
            model_version=settings.MODEL_VERSION,
            confidence_threshold=confidence_threshold if confidence_threshold is not None else settings.CONFIDENCE_THRESHOLD,
            iou_threshold=iou_threshold if iou_threshold is not None else settings.IOU_THRESHOLD,
            device=device or settings.DEVICE
        )

    def run_survey_analysis(
        self,
        survey_id: str,
        raw_image_path: str,
        nav_file_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None
    ) -> List[Contact]:
        """
        Executes the full anomaly detection pipeline on an SSS survey swath.
        Returns a list of Canonical Contact objects.
        """
        if confidence_threshold is not None:
            self.detector.confidence_threshold = confidence_threshold

        if not os.path.exists(raw_image_path):
            raise FileNotFoundError(f"Sonar image not found: {raw_image_path}")

        raw_image = cv2.imread(raw_image_path)
        if raw_image is None:
            raise ValueError(f"Failed to decode image file: {raw_image_path}")

        img_h, img_w = raw_image.shape[:2]

        # 1. Compute Data Quality
        quality_metrics = compute_image_quality(raw_image)
        quality_score = quality_metrics.get("quality_score", 1.0)

        # 2. Tiling for side-scan sonar waterfall swaths
        # If image dimensions fit directly in 640x640, run directly
        if img_w <= settings.IMAGE_SIZE and img_h <= settings.IMAGE_SIZE:
            raw_detections = self.detector.predict(
                image=raw_image,
                tile_id=f"{survey_id}_FULL",
                offset_x=0,
                offset_y=0
            )
        else:
            # Generate deterministic overlapping 640x640 tiles
            tiles = generate_tiles(
                raw_image,
                tile_size=settings.IMAGE_SIZE,
                overlap=0.20
            )
            raw_detections: List[DrishtiDetection] = []
            for tile in tiles:
                tile_img = tile.get("tile_image", tile.get("image"))
                offset_x = tile.get("offset_x", tile.get("x_offset", 0))
                offset_y = tile.get("offset_y", tile.get("y_offset", 0))
                tile_dets = self.detector.predict(
                    image=tile_img,
                    tile_id=f"{survey_id}_T{tile['tile_id']:03d}",
                    offset_x=offset_x,
                    offset_y=offset_y
                )
                raw_detections.extend(tile_dets)

        # 3. Deduplicate detections across overlapping tile boundaries
        # Adapt to dict format for existing deduplicate_detections
        det_dicts = [
            {
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox": {"x1": d.bbox[0], "y1": d.bbox[1], "x2": d.bbox[2], "y2": d.bbox[3]},
                "tile_id": d.tile_id,
                "_original_obj": d
            }
            for d in raw_detections
        ]
        filtered_dicts = deduplicate_detections(det_dicts, iou_threshold=self.detector.iou_threshold)
        deduped_detections = [item["_original_obj"] for item in filtered_dicts]

        # 4. Geolocation service initialization
        geo_service = GeolocationService(nav_file_path=nav_file_path)

        # 5. Acoustic context evaluator callback
        def context_eval(bbox_coords: List[int]) -> Dict[str, float]:
            return extract_acoustic_context(
                image=raw_image,
                bbox={"x1": bbox_coords[0], "y1": bbox_coords[1], "x2": bbox_coords[2], "y2": bbox_coords[3]},
                nadir_x=img_w // 2
            )

        # 6. Transform internal DrishtiDetection -> Canonical Contact schema
        contacts = transform_drishti_detections_to_contacts(
            detections=deduped_detections,
            survey_id=survey_id,
            data_quality=quality_score,
            context_evaluator=context_eval,
            geo_service=geo_service,
            image_width=img_w,
            image_height=img_h
        )

        # Sort descending: HIGH priority first, then confidence
        priority_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        contacts.sort(
            key=lambda c: (priority_rank.get(c.priority, 1), c.confidence),
            reverse=True
        )

        # Re-number sorted contacts C001, C002...
        for idx, contact in enumerate(contacts):
            contact.contact_id = f"C{idx+1:03d}"

        return contacts

    def run_survey_contact_packages(
        self,
        survey_id: str,
        raw_image_path: str,
        nav_file_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None
    ) -> List[ContactPackage]:
        """
        Executes the complete multi-modal pipeline on an SSS survey swath:
        YOLO Candidate Search -> SW-Net Verification -> Evidence Extraction ->
        Evidence Fusion -> Confidence Calibration -> Contact Package Generation.

        Returns a list of machine-readable ContactPackage objects for Person 3.
        """
        if confidence_threshold is not None:
            self.detector.confidence_threshold = confidence_threshold

        if not os.path.exists(raw_image_path):
            raise FileNotFoundError(f"Sonar image not found: {raw_image_path}")

        raw_image = cv2.imread(raw_image_path)
        if raw_image is None:
            raise ValueError(f"Failed to decode image file: {raw_image_path}")

        img_h, img_w = raw_image.shape[:2]

        # 1. Tile-based YOLO Detection
        if img_w <= settings.IMAGE_SIZE and img_h <= settings.IMAGE_SIZE:
            raw_detections = self.detector.predict(
                image=raw_image,
                tile_id=f"{survey_id}_FULL",
                offset_x=0,
                offset_y=0
            )
        else:
            tiles = generate_tiles(raw_image, tile_size=settings.IMAGE_SIZE, overlap=0.20)
            raw_detections: List[DrishtiDetection] = []
            for tile in tiles:
                tile_img = tile.get("tile_image", tile.get("image"))
                offset_x = tile.get("offset_x", tile.get("x_offset", 0))
                offset_y = tile.get("offset_y", tile.get("y_offset", 0))
                tile_dets = self.detector.predict(
                    image=tile_img,
                    tile_id=f"{survey_id}_T{tile['tile_id']:03d}",
                    offset_x=offset_x,
                    offset_y=offset_y
                )
                raw_detections.extend(tile_dets)

        # 2. Deduplicate detections
        det_dicts = [
            {
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox": {"x1": d.bbox[0], "y1": d.bbox[1], "x2": d.bbox[2], "y2": d.bbox[3]},
                "tile_id": d.tile_id,
                "_original_obj": d
            }
            for d in raw_detections
        ]
        filtered_dicts = deduplicate_detections(det_dicts, iou_threshold=self.detector.iou_threshold)
        deduped_detections = [item["_original_obj"] for item in filtered_dicts if not item["_original_obj"].is_filtered]

        # 3. Initialize Verification, Evidence, Fusion, and Geolocation engines
        swnet_verifier = SWNetVerifier()
        evidence_extractor = EvidenceExtractor()
        fusion_engine = EvidenceFusionEngine()
        package_builder = ContactPackageBuilder()
        geo_service = GeolocationService(nav_file_path=nav_file_path) if nav_file_path else None

        contact_packages: List[ContactPackage] = []

        for idx, det in enumerate(deduped_detections):
            bx1, by1, bx2, by2 = det.bbox

            # Stage-2: SW-Net verification
            swnet_result = None
            if det.class_name.lower() in ("shipwreck", "wreck"):
                try:
                    swnet_result = swnet_verifier.verify_roi(raw_image, bbox=[bx1, by1, bx2, by2])
                except Exception:
                    swnet_result = None

            # Stage-2.5: Extract full multi-modal evidence
            evidence = evidence_extractor.extract(
                image=raw_image,
                bbox=[bx1, by1, bx2, by2],
                class_name=det.class_name,
                class_id=det.class_id,
                detector_confidence=det.confidence,
                swnet_result=swnet_result,
                nadir_x=img_w // 2,
                metadata={"tile_id": det.tile_id}
            )

            # Stage-3: Multi-Evidence Fusion
            fused_result = fusion_engine.fuse(evidence)

            # Geolocation (Zero Coordinate Fabrication)
            geo_tuple = None
            if geo_service is not None:
                try:
                    center_x = (bx1 + bx2) // 2
                    center_y = (by1 + by2) // 2
                    lat, lon, loc_status = geo_service.estimate_contact_location(
                        bbox_center_x=center_x,
                        bbox_center_y=center_y,
                        image_width=img_w,
                        image_height=img_h
                    )
                    geo_tuple = (lat, lon, None, loc_status)
                except Exception:
                    geo_tuple = None

            # Build Contact Package
            contact_id = f"CNT-{idx+1:06d}"
            pkg = package_builder.build_package(
                contact_id=contact_id,
                survey_id=survey_id,
                evidence=evidence,
                fused_result=fused_result,
                geo_location=geo_tuple
            )
            contact_packages.append(pkg)

        # Sort descending by priority and confidence
        priority_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        contact_packages.sort(
            key=lambda p: (priority_rank.get(p.priority, 1), p.final_confidence),
            reverse=True
        )

        return contact_packages

