"""
Curated Demo Samples API.

Provides access to real held-out test sonar samples and operational reference swaths
for reproducible, controlled demonstration without fabrication.
"""

from typing import List, Dict, Any, Optional
import os
import shutil
import time
import cv2
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.models import SurveyModel
from backend.app.database.repository import SurveyRepository, ContactRepository
from backend.app.schemas.survey import SurveyUploadResponse
from backend.app.services.sonar_service import SonarService
from backend.app.services.inference_service import InferenceService

router = APIRouter(prefix="/api/demo", tags=["Demo"])
sonar_service = SonarService()
inference_service = InferenceService()

DEMO_SAMPLES = {
    "viator_04": {
        "id": "viator_04",
        "title": "Viator-04 (Held-out Test Shipwreck — True Positive)",
        "description": "Held-out test set sonar swath containing prominent shipwreck hull with strong acoustic highlight and down-range shadow.",
        "category": "TRUE_POSITIVE_BENCHMARK",
        "image_path": "data/demo/sonar/viator_04_test_wreck.png",
        "nav_path": "data/demo/navigation/viator_04_nav.csv",
        "filename": "viator_04_test_wreck.png"
    },
    "artificial_reef_02": {
        "id": "artificial_reef_02",
        "title": "Artificial Reef-02 (Held-out Test Clutter — Operator Triage Demo)",
        "description": "Held-out test set sonar swath with geological ridges and reef structures demonstrating operator false-alarm rejection.",
        "category": "CLUTTER_TRIAGE_DEMO",
        "image_path": "data/demo/sonar/artificial_reef_02_test_clutter.png",
        "nav_path": "data/demo/navigation/artificial_reef_02_nav.csv",
        "filename": "artificial_reef_02_test_clutter.png"
    },
    "corsican_02": {
        "id": "corsican_02",
        "title": "Corsican-02 (Held-out Test Shipwreck — Verified Anomaly)",
        "description": "Held-out test set sonar swath containing verified shipwreck target matching ground-truth YOLO annotation.",
        "category": "TRUE_POSITIVE_BENCHMARK",
        "image_path": "data/demo/sonar/corsican_02_test_wreck.png",
        "nav_path": "data/demo/navigation/corsican_02_nav.csv",
        "filename": "corsican_02_test_wreck.png"
    },
    "survey_001": {
        "id": "survey_001",
        "title": "Survey-001 (Operational Reference Swath with Towfish Nav)",
        "description": "Operational reference swath with full towfish heading and GPS navigation log for spatial estimation.",
        "category": "NAV_INTEGRATED_REFERENCE",
        "image_path": "data/demo/sonar/survey_001_raw.png",
        "nav_path": "data/demo/navigation/survey_001_nav.csv",
        "filename": "survey_001_raw.png"
    }
}


@router.get("/samples")
def get_demo_samples() -> List[Dict[str, Any]]:
    """Returns catalog of curated demo samples."""
    return [
        {
            "id": s["id"],
            "title": s["title"],
            "description": s["description"],
            "category": s["category"],
            "has_navigation": s["nav_path"] is not None
        }
        for s in DEMO_SAMPLES.values()
    ]


@router.post("/load/{sample_id}", response_model=Dict[str, Any])
def load_demo_sample(sample_id: str, db: Session = Depends(get_db)):
    """
    Ingests and executes the full inference pipeline on a curated demo sample.
    Saves the survey and contacts to the database and returns complete results.
    """
    if sample_id not in DEMO_SAMPLES:
        raise HTTPException(status_code=404, detail=f"Demo sample '{sample_id}' not found.")

    sample = DEMO_SAMPLES[sample_id]
    image_path = sample["image_path"]
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Demo file '{image_path}' missing from disk.")

    survey_id = f"DEMO_{sample_id.upper()}_{int(time.time() * 1000)}"
    raw_dest = os.path.join(sonar_service.raw_dir, f"{survey_id}_{sample['filename']}")
    shutil.copyfile(image_path, raw_dest)

    nav_dest = None
    if sample["nav_path"] and os.path.exists(sample["nav_path"]):
        nav_dest = os.path.join(sonar_service.raw_dir, f"{survey_id}_nav.csv")
        shutil.copyfile(sample["nav_path"], nav_dest)

    # 1. Quality & Preprocessing
    img = cv2.imread(raw_dest)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to load copied demo image.")

    h, w = img.shape[:2]
    from ml.preprocessing.quality import compute_image_quality
    quality = compute_image_quality(img)

    processed_path = sonar_service.get_processed_path(survey_id)
    sonar_service.pipeline.run(img, output_processed_path=processed_path)

    SurveyRepository(db).save_survey(
        survey_id=survey_id,
        filename=sample["filename"],
        raw_image_path=raw_dest,
        processed_image_path=processed_path,
        nav_file_path=nav_dest,
        image_width=w,
        image_height=h,
        data_quality=quality["quality_score"]
    )

    # 2. Real Inference
    contacts = inference_service.run_survey_analysis(
        survey_id=survey_id,
        raw_image_path=raw_dest,
        nav_file_path=nav_dest,
        confidence_threshold=0.20
    )
    ContactRepository(db).save_contacts(contacts)

    survey_dto = SurveyUploadResponse(
        survey_id=survey_id,
        filename=sample["filename"],
        image_width=w,
        image_height=h,
        data_quality=quality["quality_score"],
        has_navigation=nav_dest is not None,
        raw_image_url=f"/api/surveys/{survey_id}/image/raw",
        processed_image_url=f"/api/surveys/{survey_id}/image/processed",
        message=f"Curated demo sample '{sample['title']}' loaded and analyzed successfully."
    )

    return {
        "survey": survey_dto.dict(),
        "contacts": [c.dict() for c in contacts],
        "sample_info": sample
    }
