"""
Standalone Sonar Anomaly Inference API Endpoint.

POST /api/inference/detect
Accepts a sonar image via multipart/form-data and returns standardized detection JSON.
Decoupled from database persistence or frontend-specific views.
"""

from typing import List, Optional
import os
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from pydantic import BaseModel, Field

from ml.inference.drishti_detector import DrishtiDetector
from backend.app.core.config import settings
from backend.app.services.inference_service import get_inference_service

router = APIRouter(prefix="/api/inference", tags=["Inference"])

def _get_detector() -> DrishtiDetector:
    return get_inference_service().detector


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
    backend: str = Field(default="local", description="Inference backend: 'huggingface' or 'local'")
    fallback_used: bool = Field(default=False, description="Whether local fallback was triggered after HF failure")
    fallback_reason: Optional[str] = Field(default=None, description="Reason for fallback if HF failed")
    annotated_image: Optional[str] = Field(default=None, description="Annotated image path or URL if returned by remote API")
    image: ImageMetadata
    detections: List[DetectionItem]
    filtered_detections_count: int = Field(default=0, description="Count of detections filtered by product policy (e.g. crab_pot)")

    model_config = {
        "protected_namespaces": ()
    }


@router.post("/detect", response_model=InferenceResponse)
async def detect_sonar_anomalies(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Query(None, ge=0.0, le=1.0)
):
    """
    Executes hybrid anomaly candidate detection on a single uploaded sonar image.
    Prioritizes Hugging Face Space API as primary, falling back to local DRISHTI detector.
    Returns standardized detection schema with AI_CANDIDATE review status and backend provenance.
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

    # Save to a temporary file for remote HF upload
    import tempfile
    suffix = ".png" if not file.filename else os.path.splitext(file.filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        tmp_file.write(contents)
        tmp_path = tmp_file.name

    try:
        inference_service = get_inference_service()
        hybrid_engine = inference_service.hybrid_engine
        result = hybrid_engine.run_inference(
            image_path=tmp_path,
            image=image,
            confidence_threshold=confidence_threshold
        )
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    if not result.success and result.error:
        raise HTTPException(status_code=500, detail=result.error)

    # 5. Separate eligible detections from product-filtered detections (e.g. crab_pot)
    eligible_detections: List[DetectionItem] = []
    filtered_count = 0

    for det in result.detections:
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
        model_name=result.model_name or settings.MODEL_NAME,
        model_version=result.model_version or settings.MODEL_VERSION,
        backend=result.backend,
        fallback_used=result.fallback_used,
        fallback_reason=result.fallback_reason,
        annotated_image=result.annotated_image,
        image=ImageMetadata(width=w, height=h),
        detections=eligible_detections,
        filtered_detections_count=filtered_count
    )

