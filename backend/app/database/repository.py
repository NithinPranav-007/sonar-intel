"""
Data Repository Layer for SONAR-INTEL.

Encapsulates SQL operations and transforms database records into Canonical Contact objects.
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from backend.app.database.models import SurveyModel, ContactModel, ReviewModel
from backend.app.schemas.contact import Contact, BoundingBox


def to_canonical_contact(model: ContactModel) -> Contact:
    """Transforms a database ContactModel into the Canonical Contact Pydantic schema."""
    return Contact(
        contact_id=model.contact_id,
        survey_id=model.survey_id,
        class_name=model.class_name,
        confidence=model.confidence,
        bbox=BoundingBox(
            x1=model.bbox_x1,
            y1=model.bbox_y1,
            x2=model.bbox_x2,
            y2=model.bbox_y2
        ),
        data_quality=model.data_quality,
        shadow_evidence=model.shadow_evidence,
        context_score=model.context_score,
        priority=model.priority,
        latitude=model.latitude,
        longitude=model.longitude,
        localization_status=model.localization_status,
        review_status=model.review_status,
        review_note=model.review_note,
        model_version=model.model_version
    )


class SurveyRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_survey(
        self,
        survey_id: str,
        filename: str,
        raw_image_path: str,
        image_width: int,
        image_height: int,
        data_quality: float,
        nav_file_path: Optional[str] = None,
        processed_image_path: Optional[str] = None
    ) -> SurveyModel:
        survey = SurveyModel(
            survey_id=survey_id,
            filename=filename,
            raw_image_path=raw_image_path,
            processed_image_path=processed_image_path,
            nav_file_path=nav_file_path,
            image_width=image_width,
            image_height=image_height,
            data_quality=data_quality,
            has_navigation=bool(nav_file_path)
        )
        self.db.merge(survey)
        self.db.commit()
        return survey

    def get_survey(self, survey_id: str) -> Optional[SurveyModel]:
        return self.db.query(SurveyModel).filter(SurveyModel.survey_id == survey_id).first()

    def update_processed_image(self, survey_id: str, processed_path: str):
        survey = self.get_survey(survey_id)
        if survey:
            survey.processed_image_path = processed_path
            self.db.commit()


class ContactRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_contacts(self, contacts: List[Contact]) -> List[Contact]:
        """Saves or updates a batch of canonical contacts."""
        for c in contacts:
            model = ContactModel(
                contact_id=c.contact_id,
                survey_id=c.survey_id,
                class_name=c.class_name,
                confidence=c.confidence,
                bbox_x1=c.bbox.x1,
                bbox_y1=c.bbox.y1,
                bbox_x2=c.bbox.x2,
                bbox_y2=c.bbox.y2,
                data_quality=c.data_quality,
                shadow_evidence=c.shadow_evidence,
                context_score=c.context_score,
                priority=c.priority,
                latitude=c.latitude,
                longitude=c.longitude,
                localization_status=c.localization_status,
                review_status=c.review_status,
                review_note=c.review_note,
                model_version=c.model_version
            )
            self.db.merge(model)
        self.db.commit()
        return contacts

    def get_contacts_by_survey(self, survey_id: str) -> List[Contact]:
        records = self.db.query(ContactModel).filter(ContactModel.survey_id == survey_id).all()
        return [to_canonical_contact(r) for r in records]

    def get_contact_by_id(self, contact_id: str) -> Optional[Contact]:
        record = self.db.query(ContactModel).filter(ContactModel.contact_id == contact_id).first()
        return to_canonical_contact(record) if record else None

    def update_review(
        self,
        contact_id: str,
        review_status: str,
        review_note: Optional[str]
    ) -> Optional[Contact]:
        contact = self.db.query(ContactModel).filter(ContactModel.contact_id == contact_id).first()
        if not contact:
            return None

        contact.review_status = review_status
        contact.review_note = review_note

        # Log review to audit trail
        audit = ReviewModel(
            contact_id=contact.contact_id,
            survey_id=contact.survey_id,
            review_status=review_status,
            review_note=review_note,
            model_version=contact.model_version
        )
        self.db.add(audit)
        self.db.commit()
        self.db.refresh(contact)
        return to_canonical_contact(contact)

    def search_contacts(
        self,
        q: Optional[str] = None,
        priority: Optional[str] = None,
        review_status: Optional[str] = None,
        survey_id: Optional[str] = None
    ) -> List[Contact]:
        """Search contacts across IDs, survey, priority, and review state."""
        query = self.db.query(ContactModel)
        if survey_id:
            query = query.filter(ContactModel.survey_id == survey_id)
        if priority:
            query = query.filter(ContactModel.priority == priority.upper())
        if review_status:
            query = query.filter(ContactModel.review_status == review_status.upper())
        if q:
            term = f"%{q.strip()}%"
            query = query.filter(
                (ContactModel.contact_id.ilike(term)) |
                (ContactModel.survey_id.ilike(term)) |
                (ContactModel.class_name.ilike(term)) |
                (ContactModel.review_status.ilike(term)) |
                (ContactModel.priority.ilike(term))
            )
        records = query.all()
        return [to_canonical_contact(r) for r in records]
