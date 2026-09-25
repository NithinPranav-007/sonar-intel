"""
Human Review Schemas.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class ReviewSubmission(BaseModel):
    review_status: Literal["CONFIRMED", "FALSE_POSITIVE", "UNCERTAIN"] = Field(
        ...,
        description="Reviewer triage verdict"
    )
    review_note: Optional[str] = Field(
        default=None,
        description="Detailed analyst observations (e.g. shadow length, target dimensions)"
    )
