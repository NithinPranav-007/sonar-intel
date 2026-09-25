"""
Reporting and Export API Endpoints.

GET /api/surveys/{survey_id}/geojson
GET /api/surveys/{survey_id}/summary
GET /api/surveys/{survey_id}/csv
GET /api/surveys/{survey_id}/track
"""

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.repository import ContactRepository, SurveyRepository
from backend.app.schemas.survey import SurveySummary
from backend.app.utils.geojson import contacts_to_geojson, contacts_to_csv_string
from backend.app.services.geolocation_service import GeolocationService

router = APIRouter(prefix="/api/surveys", tags=["Reports & Export"])


@router.get("/{survey_id}/geojson")
async def export_survey_geojson(survey_id: str, db: Session = Depends(get_db)):
    """Exports contacts as standard GeoJSON FeatureCollection."""
    survey = SurveyRepository(db).get_survey(survey_id)
    if not survey:
        raise HTTPException(status_code=404, detail=f"Survey '{survey_id}' not found.")

    contacts = ContactRepository(db).get_contacts_by_survey(survey_id)
    geojson_data = contacts_to_geojson(contacts)
    return geojson_data


@router.get("/{survey_id}/csv")
async def export_survey_csv(survey_id: str, db: Session = Depends(get_db)):
    """Exports contacts as tabular CSV report."""
    survey = SurveyRepository(db).get_survey(survey_id)
    if not survey:
        raise HTTPException(status_code=404, detail=f"Survey '{survey_id}' not found.")

    contacts = ContactRepository(db).get_contacts_by_survey(survey_id)
    csv_content = contacts_to_csv_string(contacts)

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={survey_id}_contacts.csv"}
    )


@router.get("/{survey_id}/summary", response_model=SurveySummary)
async def get_survey_summary(survey_id: str, db: Session = Depends(get_db)):
    """Computes mission-level tally of contacts by priority and review state."""
    survey = SurveyRepository(db).get_survey(survey_id)
    if not survey:
        raise HTTPException(status_code=404, detail=f"Survey '{survey_id}' not found.")

    contacts = ContactRepository(db).get_contacts_by_survey(survey_id)
    total = len(contacts)
    high_p = sum(1 for c in contacts if c.priority == "HIGH")
    med_p = sum(1 for c in contacts if c.priority == "MEDIUM")
    low_p = sum(1 for c in contacts if c.priority == "LOW")

    reviewed = sum(1 for c in contacts if c.review_status != "AI_CANDIDATE")
    pending = total - reviewed

    return SurveySummary(
        survey_id=survey.survey_id,
        filename=survey.filename,
        total_contacts=total,
        high_priority=high_p,
        medium_priority=med_p,
        low_priority=low_p,
        reviewed_count=reviewed,
        pending_count=pending,
        data_quality_avg=survey.data_quality
    )


@router.get("/{survey_id}/track")
async def get_survey_track(survey_id: str, db: Session = Depends(get_db)):
    """Returns the vessel/towfish trajectory waypoints for map rendering."""
    survey = SurveyRepository(db).get_survey(survey_id)
    if not survey:
        raise HTTPException(status_code=404, detail=f"Survey '{survey_id}' not found.")

    if not survey.nav_file_path:
        return []

    geo_service = GeolocationService(nav_file_path=survey.nav_file_path)
    return geo_service.get_survey_track()
