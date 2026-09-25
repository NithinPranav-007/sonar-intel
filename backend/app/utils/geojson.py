"""
GeoJSON and CSV Serializers for Contacts and Surveys.
"""

from typing import List, Dict, Any
from backend.app.schemas.contact import Contact
import io
import csv


def contacts_to_geojson(contacts: List[Contact]) -> Dict[str, Any]:
    """Converts a list of Canonical Contacts to a GeoJSON FeatureCollection."""
    features = []
    for c in contacts:
        if c.latitude is not None and c.longitude is not None:
            geometry = {
                "type": "Point",
                "coordinates": [round(c.longitude, 6), round(c.latitude, 6)]
            }
        else:
            geometry = None

        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "contact_id": c.contact_id,
                "survey_id": c.survey_id,
                "class_name": c.class_name,
                "confidence": c.confidence,
                "priority": c.priority,
                "shadow_evidence": c.shadow_evidence,
                "context_score": c.context_score,
                "data_quality": c.data_quality,
                "localization_status": c.localization_status,
                "review_status": c.review_status,
                "review_note": c.review_note,
                "model_version": c.model_version,
                "bbox": {
                    "x1": c.bbox.x1,
                    "y1": c.bbox.y1,
                    "x2": c.bbox.x2,
                    "y2": c.bbox.y2
                }
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features
    }


def contacts_to_csv_string(contacts: List[Contact]) -> str:
    """Exports contacts to standard CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "contact_id",
        "survey_id",
        "class_name",
        "confidence",
        "priority",
        "latitude",
        "longitude",
        "localization_status",
        "shadow_evidence",
        "context_score",
        "data_quality",
        "review_status",
        "review_note",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "model_version"
    ])
    for c in contacts:
        writer.writerow([
            c.contact_id,
            c.survey_id,
            c.class_name,
            c.confidence,
            c.priority,
            c.latitude if c.latitude is not None else "",
            c.longitude if c.longitude is not None else "",
            c.localization_status,
            c.shadow_evidence,
            c.context_score,
            c.data_quality,
            c.review_status,
            c.review_note or "",
            c.bbox.x1,
            c.bbox.y1,
            c.bbox.x2,
            c.bbox.y2,
            c.model_version
        ])
    return output.getvalue()
