"""
Survey Ingestion / Upload API Endpoint.

POST /api/surveys/upload

Memory-safe upload implementation.
"""

import os
import time
import gc
from typing import Optional

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    Depends,
    HTTPException,
    status,
)

from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.repository import SurveyRepository
from backend.app.services.sonar_service import SonarService
from backend.app.schemas.survey import SurveyUploadResponse


router = APIRouter(
    prefix="/api/surveys",
    tags=["Surveys"],
)


# ================================================================
# CONFIGURATION
# ================================================================

# Render Free has approximately 512 MB RAM.
#
# Limit individual sonar uploads to avoid a huge image
# crashing the entire FastAPI process.
#
# Default: 25 MB
MAX_SONAR_FILE_SIZE = int(
    os.getenv(
        "MAX_SONAR_FILE_SIZE",
        str(25 * 1024 * 1024),
    )
)


MAX_NAV_FILE_SIZE = int(
    os.getenv(
        "MAX_NAV_FILE_SIZE",
        str(5 * 1024 * 1024),
    )
)


# Create lightweight service.
sonar_service = SonarService()


# ================================================================
# UPLOAD ENDPOINT
# ================================================================

@router.post(
    "/upload",
    response_model=SurveyUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_survey(
    sonar_file: UploadFile = File(
        ...,
        description="Raw side-scan sonar waterfall image",
    ),
    nav_file: Optional[UploadFile] = File(
        None,
        description="Optional navigation track CSV",
    ),
    survey_id_override: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Ingest, validate, and store a side-scan sonar swath.

    Processing performed during upload:

        upload
          ↓
        size validation
          ↓
        raw file storage
          ↓
        OpenCV validation
          ↓
        quality calculation
          ↓
        processed preview
          ↓
        database record

    YOLO tiles are NOT generated during upload.
    """

    # ============================================================
    # 1. VALIDATE FILENAME
    # ============================================================

    if not sonar_file.filename:

        raise HTTPException(
            status_code=400,
            detail="Missing sonar image file.",
        )

    # ============================================================
    # 2. GENERATE SURVEY ID
    # ============================================================

    timestamp_str = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    survey_id = (
        survey_id_override
        or f"SURV_{timestamp_str}"
    )

    # ============================================================
    # 3. READ SONAR FILE
    # ============================================================

    file_bytes = await sonar_file.read()

    # ============================================================
    # 4. CHECK FILE SIZE
    # ============================================================

    file_size = len(file_bytes)

    if file_size == 0:

        del file_bytes

        raise HTTPException(
            status_code=400,
            detail="Uploaded sonar file is empty.",
        )

    if file_size > MAX_SONAR_FILE_SIZE:

        del file_bytes

        max_mb = (
            MAX_SONAR_FILE_SIZE
            / (1024 * 1024)
        )

        raise HTTPException(
            status_code=413,
            detail=(
                f"Sonar file is too large. "
                f"Maximum allowed size is "
                f"{max_mb:.1f} MB."
            ),
        )

    # ============================================================
    # 5. PROCESS SONAR IMAGE
    # ============================================================

    try:

        (
            raw_path,
            width,
            height,
            quality,
        ) = sonar_service.store_raw_upload(
            file_bytes=file_bytes,
            survey_id=survey_id,
            original_filename=sonar_file.filename,
        )

    except ValueError as e:

        raise HTTPException(
            status_code=422,
            detail=(
                f"Image validation failed: {str(e)}"
            ),
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to process sonar image: "
                f"{str(e)}"
            ),
        )

    finally:

        # Release uploaded byte buffer.
        del file_bytes

        gc.collect()

    # ============================================================
    # 6. OPTIONAL NAVIGATION FILE
    # ============================================================

    nav_path = None

    if nav_file and nav_file.filename:

        nav_bytes = await nav_file.read()

        nav_size = len(nav_bytes)

        if nav_size > MAX_NAV_FILE_SIZE:

            del nav_bytes

            max_nav_mb = (
                MAX_NAV_FILE_SIZE
                / (1024 * 1024)
            )

            raise HTTPException(
                status_code=413,
                detail=(
                    "Navigation file is too large. "
                    f"Maximum allowed size is "
                    f"{max_nav_mb:.1f} MB."
                ),
            )

        if nav_size > 0:

            nav_dir = sonar_service.raw_dir

            os.makedirs(
                nav_dir,
                exist_ok=True,
            )

            nav_path = os.path.join(
                nav_dir,
                f"{survey_id}_nav.csv",
            )

            with open(
                nav_path,
                "wb",
            ) as f:

                f.write(nav_bytes)

        del nav_bytes

        gc.collect()

    # ============================================================
    # 7. SAVE DATABASE RECORD
    # ============================================================

    try:

        repo = SurveyRepository(db)

        processed_path = (
            sonar_service.get_processed_path(
                survey_id
            )
        )

        repo.save_survey(
            survey_id=survey_id,
            filename=sonar_file.filename,
            raw_image_path=raw_path,
            image_width=width,
            image_height=height,
            data_quality=quality[
                "quality_score"
            ],
            nav_file_path=nav_path,
            processed_image_path=(
                processed_path
                if os.path.exists(
                    processed_path
                )
                else None
            ),
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Survey was processed but "
                "database save failed: "
                f"{str(e)}"
            ),
        )

    # ============================================================
    # 8. RESPONSE
    # ============================================================

    return SurveyUploadResponse(
        survey_id=survey_id,
        filename=sonar_file.filename,
        image_width=width,
        image_height=height,
        data_quality=quality[
            "quality_score"
        ],
        has_navigation=bool(nav_path),
        raw_image_url=(
            f"/api/surveys/"
            f"{survey_id}/image/raw"
        ),
        processed_image_url=(
            f"/api/surveys/"
            f"{survey_id}/image/processed"
        ),
        message=(
            "Survey uploaded and validated "
            "successfully."
        ),
    )