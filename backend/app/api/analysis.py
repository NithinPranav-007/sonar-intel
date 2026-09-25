"""
Analysis Trigger API Endpoint.

POST /api/surveys/{survey_id}/analyze
"""

import time
import os
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.repository import SurveyRepository, ContactRepository
from backend.app.schemas.survey import AnalysisRequest, AnalysisResponse
from backend.app.services.inference_service import InferenceService

router = APIRouter(prefix="/api/surveys", tags=["Analysis"])
inference_service = InferenceService()


@router.post("/{survey_id}/analyze", response_model=AnalysisResponse)
async def analyze_survey(
    survey_id: str,
    request: Optional[AnalysisRequest] = None,
    db: Session = Depends(get_db)
):
    """
    Triggers the end-to-end normalization, candidate detection, acoustic context,
    and priority scoring pipeline for a given survey.
    """
    survey_repo = SurveyRepository(db)
    survey = survey_repo.get_survey(survey_id)
    if not survey:
        raise HTTPException(status_code=404, detail=f"Survey '{survey_id}' not found.")

    start_time = time.time()
    conf_thresh = request.confidence_threshold if request else 0.25

    try:
        contacts = inference_service.run_survey_analysis(
            survey_id=survey.survey_id,
            raw_image_path=survey.raw_image_path,
            nav_file_path=survey.nav_file_path,
            confidence_threshold=conf_thresh
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(e)}")

    # Persist contacts into database
    contact_repo = ContactRepository(db)
    saved_contacts = contact_repo.save_contacts(contacts)

    exec_time = round((time.time() - start_time) * 1000.0, 1)

    return AnalysisResponse(
        survey_id=survey_id,
        contacts_count=len(saved_contacts),
        contacts=saved_contacts,
        execution_time_ms=exec_time
    )


@router.get("/{survey_id}/image/raw")
async def get_raw_image(survey_id: str, db: Session = Depends(get_db)):
    """Serves the raw sonar image file."""
    survey = SurveyRepository(db).get_survey(survey_id)
    if not survey or not os.path.exists(survey.raw_image_path):
        raise HTTPException(status_code=404, detail="Raw image not found.")
    return FileResponse(survey.raw_image_path)


@router.get("/{survey_id}/image/processed")
async def get_processed_image(survey_id: str, db: Session = Depends(get_db)):
    """Serves the CLAHE normalized sonar image file."""
    survey = SurveyRepository(db).get_survey(survey_id)
    if not survey or not survey.processed_image_path or not os.path.exists(survey.processed_image_path):
        # Fallback to raw if processed not yet created
        if survey and os.path.exists(survey.raw_image_path):
            return FileResponse(survey.raw_image_path)
        raise HTTPException(status_code=404, detail="Processed image not found.")
    return FileResponse(survey.processed_image_path)
