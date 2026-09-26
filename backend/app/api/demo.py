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
from backend.app.services.inference_service import get_inference_service

from backend.app.core.config import settings

router = APIRouter(prefix="/api/demo", tags=["Demo"])
sonar_service = SonarService()
inference_service = get_inference_service()

def resolve_demo_file(rel_path: Optional[str]) -> Optional[str]:
    if not rel_path:
        return None
    if os.path.exists(rel_path):
        return rel_path
    # Check relative to REPO_ROOT
    candidate = os.path.join(str(settings.REPO_ROOT), rel_path)
    if os.path.exists(candidate):
        return candidate
    # Check relative to DEMO_DATA_DIR
    normalized = rel_path.replace("data/demo/", "").replace("data\\demo\\", "")
    candidate2 = os.path.join(str(settings.DEMO_DATA_DIR), normalized)
    if os.path.exists(candidate2):
        return candidate2
    return rel_path

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
async def load_demo_sample(sample_id: str, db: Session = Depends(get_db)):
    """
    Ingests and executes the full inference pipeline on a curated demo sample.
    Saves the survey and contacts to the database and returns complete results.
    """
    if sample_id not in DEMO_SAMPLES:
        raise HTTPException(status_code=404, detail=f"Demo sample '{sample_id}' not found.")

    sample = DEMO_SAMPLES[sample_id]
    image_path = resolve_demo_file(sample["image_path"])
    if not image_path or not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Demo file '{sample['image_path']}' missing from disk.")

    survey_id = f"DEMO_{sample_id.upper()}_{int(time.time() * 1000)}"
    raw_dest = os.path.join(sonar_service.raw_dir, f"{survey_id}_{sample['filename']}")
    shutil.copyfile(image_path, raw_dest)

    nav_dest = None
    nav_file = resolve_demo_file(sample["nav_path"])
    if nav_file and os.path.exists(nav_file):
        nav_dest = os.path.join(sonar_service.raw_dir, f"{survey_id}_nav.csv")
        shutil.copyfile(nav_file, nav_dest)

    # 1. Inspect image dimensions and quality without keeping duplicate buffers in RAM
    img = cv2.imread(raw_dest, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to load copied demo image.")

    h, w = img.shape[:2]
    if (h * w) > settings.MAX_IMAGE_PIXELS:
        del img
        raise HTTPException(status_code=413, detail="Sonar image dimensions are too large for this deployment.")

    from ml.preprocessing.quality import compute_image_quality
    quality = compute_image_quality(img)
    del img
    import gc
    gc.collect()

    processed_path = sonar_service.get_processed_path(survey_id)

    SurveyRepository(db).save_survey(
        survey_id=survey_id,
        filename=sample["filename"],
        raw_image_path=raw_dest,
        processed_image_path=None,  # Generated on-demand when requested by viewer
        nav_file_path=nav_dest,
        image_width=w,
        image_height=h,
        data_quality=quality["quality_score"]
    )

    # 2. Real Inference using streamed tile processing in threadpool
    from starlette.concurrency import run_in_threadpool
    contacts = await run_in_threadpool(
        inference_service.run_survey_analysis,
        survey_id=survey_id,
        raw_image_path=raw_dest,
        nav_file_path=nav_dest,
        confidence_threshold=0.20
    )
    ContactRepository(db).save_contacts(contacts)
    gc.collect()

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
        "survey": survey_dto.model_dump(),
        "contacts": [c.model_dump() for c in contacts],
        "sample_info": sample
    }
