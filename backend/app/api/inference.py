"""
Standalone Sonar Anomaly Inference API Endpoint.

POST /api/inference/detect
Accepts a sonar image via multipart/form-data and returns standardized detection JSON.
Decoupled from database persistence or frontend-specific views.
"""

from typing import List, Optional
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from pydantic import BaseModel, Field

from ml.inference.drishti_detector import DrishtiDetector
from backend.app.core.config import settings

router = APIRouter(prefix="/api/inference", tags=["Inference"])

# Singleton detector instance for API endpoints
detector = DrishtiDetector()


class DetectionItem(BaseModel):
    class_name: str = Field(..., description="Detected anomaly class name")
    confidence: float = Field(..., description="Detector model confidence score")
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    review_status: str = Field(default="AI_CANDIDATE", description="Initial review status")


class ImageMetadata(BaseModel):
    width: int
    height: int


class InferenceResponse(BaseModel):
    model_name: str
    model_version: str
    image: ImageMetadata
    detections: List[DetectionItem]
    filtered_detections_count: int = Field(default=0, description="Count of detections filtered by product policy (e.g. crab_pot)")


@router.post("/detect", response_model=InferenceResponse)
async def detect_sonar_anomalies(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Query(None, ge=0.0, le=1.0)
):
    """
    Executes DRISHTI YOLOv8s anomaly candidate proposal on a single uploaded sonar image.
    Applies Lee speckle filtering + CLAHE preprocessing internally.
    Returns standardized detection schema with AI_CANDIDATE review status.
    """
    # 1. Validate MIME type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Must be an image (PNG, JPEG, TIFF)."
        )

    # 2. Read bytes
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 3. Decode into numpy BGR array
    np_buf = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Failed to decode uploaded image. Invalid or corrupt image file.")

    h, w = image.shape[:2]

    # 4. Set optional override threshold
    orig_conf = detector.confidence_threshold
    if confidence_threshold is not None:
        detector.confidence_threshold = confidence_threshold

    try:
        raw_detections = detector.predict(image)
    finally:
        detector.confidence_threshold = orig_conf

    # 5. Separate eligible detections from product-filtered detections (e.g. crab_pot)
    eligible_detections: List[DetectionItem] = []
    filtered_count = 0

    for det in raw_detections:
        if det.is_filtered:
            filtered_count += 1
            continue
        eligible_detections.append(DetectionItem(
            class_name=det.class_name,
            confidence=det.confidence,
            bbox=det.bbox,
            review_status="AI_CANDIDATE"
        ))

    return InferenceResponse(
        model_name=detector.model_name,
        model_version=detector.model_version,
        image=ImageMetadata(width=w, height=h),
        detections=eligible_detections,
        filtered_detections_count=filtered_count
    )
