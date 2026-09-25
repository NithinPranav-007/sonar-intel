"""
Survey Ingestion / Upload API Endpoint.

POST /api/surveys/upload
"""

import os
import time
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.repository import SurveyRepository
from backend.app.services.sonar_service import SonarService
from backend.app.schemas.survey import SurveyUploadResponse

router = APIRouter(prefix="/api/surveys", tags=["Surveys"])
sonar_service = SonarService()


@router.post("/upload", response_model=SurveyUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_survey(
    sonar_file: UploadFile = File(..., description="Raw side-scan sonar waterfall image"),
    nav_file: Optional[UploadFile] = File(None, description="Optional navigation track CSV"),
    survey_id_override: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Ingests, validates, and stores a side-scan sonar swath and optional navigation log.
    The raw image is preserved unconditionally.
    """
    if not sonar_file.filename:
        raise HTTPException(status_code=400, detail="Missing sonar image file.")

    # Generate or use survey ID
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    survey_id = survey_id_override or f"SURV_{timestamp_str}"

    # Read image contents
    file_bytes = await sonar_file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded sonar file is empty.")

    try:
        raw_path, width, height, quality = sonar_service.store_raw_upload(
            file_bytes=file_bytes,
            survey_id=survey_id,
            original_filename=sonar_file.filename
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image validation failed: {str(e)}")

    # Handle optional navigation CSV
    nav_path = None
    if nav_file and nav_file.filename:
        nav_bytes = await nav_file.read()
        if len(nav_bytes) > 0:
            nav_dir = "data/raw"
            os.makedirs(nav_dir, exist_ok=True)
            nav_path = os.path.join(nav_dir, f"{survey_id}_nav.csv")
            with open(nav_path, "wb") as f:
                f.write(nav_bytes)

    # Save to database
    repo = SurveyRepository(db)
    processed_path = sonar_service.get_processed_path(survey_id)
    repo.save_survey(
        survey_id=survey_id,
        filename=sonar_file.filename,
        raw_image_path=raw_path,
        image_width=width,
        image_height=height,
        data_quality=quality["quality_score"],
        nav_file_path=nav_path,
        processed_image_path=processed_path if os.path.exists(processed_path) else None
    )

    return SurveyUploadResponse(
        survey_id=survey_id,
        filename=sonar_file.filename,
        image_width=width,
        image_height=height,
        data_quality=quality["quality_score"],
        has_navigation=bool(nav_path),
        raw_image_url=f"/api/surveys/{survey_id}/image/raw",
        processed_image_url=f"/api/surveys/{survey_id}/image/processed",
        message="Survey uploaded and validated successfully."
    )
