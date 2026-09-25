"""
Survey Pydantic Schemas.
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from backend.app.schemas.contact import Contact


class SurveyUploadResponse(BaseModel):
    survey_id: str
    filename: str
    image_width: int
    image_height: int
    data_quality: float
    has_navigation: bool
    raw_image_url: str
    processed_image_url: Optional[str] = None
    message: str


class AnalysisRequest(BaseModel):
    clahe: bool = Field(default=True, description="Enable CLAHE acoustic normalization")
    confidence_threshold: float = Field(default=0.25, ge=0.05, le=0.95)


class AnalysisResponse(BaseModel):
    survey_id: str
    contacts_count: int
    contacts: List[Contact]
    execution_time_ms: float


class SurveySummary(BaseModel):
    survey_id: str
    filename: str
    total_contacts: int
    high_priority: int
    medium_priority: int
    low_priority: int
    reviewed_count: int
    pending_count: int
    data_quality_avg: float
