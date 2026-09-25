"""
Human-in-the-Loop Review API Endpoint.

POST /api/contacts/{contact_id}/review
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.repository import ContactRepository
from backend.app.schemas.contact import Contact
from backend.app.schemas.review import ReviewSubmission

router = APIRouter(prefix="/api/contacts", tags=["Human Review"])


@router.post("/{contact_id}/review", response_model=Contact)
async def submit_contact_review(
    contact_id: str,
    submission: ReviewSubmission,
    db: Session = Depends(get_db)
):
    """
    Submits a marine surveyor triage judgment (CONFIRM, FALSE_POSITIVE, UNCERTAIN).
    Updates contact status and records the decision to the review audit trail.
    """
    repo = ContactRepository(db)
    updated_contact = repo.update_review(
        contact_id=contact_id,
        review_status=submission.review_status,
        review_note=submission.review_note
    )

    if not updated_contact:
        raise HTTPException(status_code=404, detail=f"Contact '{contact_id}' not found.")

    return updated_contact
