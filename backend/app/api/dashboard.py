"""
Dashboard Aggregation API.

Provides operational statistics, triage summaries, and recent activity logs for Screen 1.
"""

from typing import Dict, Any, List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.app.database.connection import get_db
from backend.app.database.models import SurveyModel, ContactModel, ReviewModel

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("/stats")
def get_dashboard_stats(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Aggregates real operational metrics across all ingested surveys and triage reviews."""
    total_surveys = db.query(func.count(SurveyModel.survey_id)).scalar() or 0
    total_contacts = db.query(func.count(ContactModel.contact_id)).scalar() or 0

    # Triage review statuses
    confirmed_count = db.query(func.count(ContactModel.contact_id))\
        .filter(ContactModel.review_status == "CONFIRMED").scalar() or 0
    false_positive_count = db.query(func.count(ContactModel.contact_id))\
        .filter(ContactModel.review_status == "FALSE_POSITIVE").scalar() or 0
    uncertain_count = db.query(func.count(ContactModel.contact_id))\
        .filter(ContactModel.review_status == "UNCERTAIN").scalar() or 0
    pending_ai_candidates = db.query(func.count(ContactModel.contact_id))\
        .filter(ContactModel.review_status == "AI_CANDIDATE").scalar() or 0

    # Priority breakdown
    high_priority = db.query(func.count(ContactModel.contact_id))\
        .filter(ContactModel.priority == "HIGH").scalar() or 0
    medium_priority = db.query(func.count(ContactModel.contact_id))\
        .filter(ContactModel.priority == "MEDIUM").scalar() or 0
    low_priority = db.query(func.count(ContactModel.contact_id))\
        .filter(ContactModel.priority == "LOW").scalar() or 0

    # Recent platform activity log (last 10 reviews and recent surveys)
    recent_reviews = db.query(ReviewModel)\
        .order_by(ReviewModel.reviewed_at.desc()).limit(10).all()

    activity_log: List[Dict[str, Any]] = []
    for r in recent_reviews:
        activity_log.append({
            "type": "VERIFICATION",
            "timestamp": r.reviewed_at.strftime("%H:%M UTC") if r.reviewed_at else "Recent",
            "date": r.reviewed_at.strftime("%d %b %Y") if r.reviewed_at else "Today",
            "contact_id": r.contact_id,
            "survey_id": r.survey_id,
            "status": r.review_status,
            "note": r.review_note or f"Contact {r.contact_id} marked as {r.review_status} by operator.",
            "priority": "HIGH" if r.review_status == "CONFIRMED" else "LOW"
        })

    # Add survey ingestion events if activity log has space
    if len(activity_log) < 5:
        recent_surveys = db.query(SurveyModel)\
            .order_by(SurveyModel.created_at.desc()).limit(5).all()
        for s in recent_surveys:
            activity_log.append({
                "type": "DETECTION",
                "timestamp": s.created_at.strftime("%H:%M UTC") if s.created_at else "Recent",
                "date": s.created_at.strftime("%d %b %Y") if s.created_at else "Today",
                "contact_id": "NEW SWATH",
                "survey_id": s.survey_id,
                "status": "INGESTED",
                "note": f"Survey {s.filename} processed ({s.image_width}x{s.image_height} px).",
                "priority": "MEDIUM"
            })

    # Contact coordinates summary for mini map (only contacts with real estimated coordinates)
    contacts_with_geo = db.query(
        ContactModel.contact_id,
        ContactModel.latitude,
        ContactModel.longitude,
        ContactModel.priority,
        ContactModel.review_status
    ).filter(ContactModel.latitude.isnot(None), ContactModel.longitude.isnot(None)).limit(50).all()

    geo_points = [
        {
            "id": c[0],
            "lat": c[1],
            "lon": c[2],
            "priority": c[3],
            "status": c[4]
        }
        for c in contacts_with_geo
    ]

    return {
        "total_surveys": total_surveys,
        "total_detections": total_contacts,
        "confirmed_contacts": confirmed_count,
        "false_positives": false_positive_count,
        "uncertain_reviews": uncertain_count,
        "pending_ai_candidates": pending_ai_candidates,
        "priority_distribution": {
            "high": high_priority,
            "medium": medium_priority,
            "low": low_priority
        },
        "activity_log": activity_log,
        "geo_points": geo_points
    }
