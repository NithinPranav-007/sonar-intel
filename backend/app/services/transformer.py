"""
Detection to Canonical Contact Transformation Service.

Transforms internal model-specific detection schemas (e.g. DrishtiDetection)
into the authoritative project-wide Contact schema.

Enforces:
- Downstream product class filtering (filtered detections never become Contacts)
- Separation of model_score, calibrated_confidence, and acoustic context
- Initial review_status as AI_CANDIDATE
- Zero geographic coordinate fabrication (None when navigation is unavailable)
"""

from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone

from ml.inference.drishti_detector import DrishtiDetection
from backend.app.schemas.contact import Contact, BoundingBox


def transform_drishti_detections_to_contacts(
    detections: List[DrishtiDetection],
    survey_id: str,
    data_quality: float = 1.0,
    context_evaluator: Optional[Any] = None,
    geo_service: Optional[Any] = None,
    image_width: int = 640,
    image_height: int = 640
) -> List[Contact]:
    """
    Converts a list of DrishtiDetection objects into canonical Contact objects.

    Args:
        detections: Raw internal detections from DrishtiDetector.
        survey_id: Identifier of the parent survey swath.
        data_quality: Signal quality score of the acoustic swath.
        context_evaluator: Optional callable/evaluator for acoustic shadow physics.
        geo_service: Optional GeolocationService instance for navigation interpolation.
        image_width: Swath image width for relative positioning.
        image_height: Swath image height for relative positioning.

    Returns:
        List of production-eligible Canonical Contact objects.
    """
    contacts: List[Contact] = []
    contact_counter = 1

    for det in detections:
        # Rule 1: Filtered detections (e.g., crab_pot per product policy) do not become Contacts
        if det.is_filtered:
            continue

        x1, y1, x2, y2 = det.bbox

        # Calculate acoustic context if evaluator provided
        shadow_evidence = 0.0
        context_score = 0.0
        if context_evaluator is not None:
            try:
                ctx = context_evaluator(det.bbox)
                shadow_evidence = ctx.get("shadow_evidence", 0.0)
                context_score = ctx.get("context_score", 0.0)
            except Exception:
                pass

        # Estimate geolocation only when valid navigation service is available
        lat: Optional[float] = None
        lon: Optional[float] = None
        loc_status = "UNAVAILABLE"
        loc_uncertainty: Optional[float] = None

        if geo_service is not None:
            try:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                lat, lon, loc_status = geo_service.estimate_contact_location(
                    bbox_center_x=center_x,
                    bbox_center_y=center_y,
                    image_width=image_width,
                    image_height=image_height
                )
                if lat is not None and lon is not None:
                    loc_uncertainty = 1.5  # Typical along-track dead-reckoning uncertainty in meters
            except Exception:
                lat, lon, loc_status = None, None, "UNAVAILABLE"

        # Determine operational priority based on confidence and physics context
        if det.confidence >= 0.70 or context_score >= 0.60:
            priority = "HIGH"
        elif det.confidence >= 0.40 or context_score >= 0.35:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        contact = Contact(
            contact_id=f"C{contact_counter:03d}",
            survey_id=survey_id,
            class_name=det.class_name,
            confidence=det.confidence,
            model_score=det.confidence,
            calibrated_confidence=None,  # Uncalibrated baseline
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            source_tile=det.tile_id,
            source_ping=None,
            detection_timestamp=datetime.now(timezone.utc).isoformat(),
            data_quality=data_quality,
            shadow_evidence=shadow_evidence,
            context_score=context_score,
            priority=priority,
            latitude=lat,
            longitude=lon,
            location_uncertainty=loc_uncertainty,
            localization_status=loc_status,
            review_status="AI_CANDIDATE",  # Never auto-confirmed
            review_note=None,
            model_name=det.model_name,
            model_version=det.model_version
        )

        contacts.append(contact)
        contact_counter += 1

    return contacts
