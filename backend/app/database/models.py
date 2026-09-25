"""
SQLAlchemy Database Models for SONAR-INTEL.

Tables:
- SurveyModel
- ContactModel
- ReviewModel
"""

import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Text
)
from sqlalchemy.orm import relationship
from backend.app.database.connection import Base


class SurveyModel(Base):
    __tablename__ = "surveys"

    survey_id = Column(String(64), primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    raw_image_path = Column(String(512), nullable=False)
    processed_image_path = Column(String(512), nullable=True)
    nav_file_path = Column(String(512), nullable=True)
    image_width = Column(Integer, default=0)
    image_height = Column(Integer, default=0)
    data_quality = Column(Float, default=1.0)
    has_navigation = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    contacts = relationship("ContactModel", back_populates="survey", cascade="all, delete-orphan")


class ContactModel(Base):
    __tablename__ = "contacts"

    contact_id = Column(String(64), primary_key=True, index=True)
    survey_id = Column(String(64), ForeignKey("surveys.survey_id", ondelete="CASCADE"), nullable=False, index=True)
    class_name = Column(String(64), default="artificial_anomaly", nullable=False)
    confidence = Column(Float, nullable=False)

    # Pixel coordinates in survey swath
    bbox_x1 = Column(Integer, nullable=False)
    bbox_y1 = Column(Integer, nullable=False)
    bbox_x2 = Column(Integer, nullable=False)
    bbox_y2 = Column(Integer, nullable=False)

    # Acoustic metrics
    data_quality = Column(Float, default=1.0)
    shadow_evidence = Column(Float, default=0.0)
    context_score = Column(Float, default=0.0)
    priority = Column(String(16), default="MEDIUM", index=True)  # HIGH, MEDIUM, LOW

    # Geospatial
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    localization_status = Column(String(24), default="UNAVAILABLE")  # ESTIMATED, VERIFIED, UNCERTAIN, UNAVAILABLE

    # Human-in-the-loop triage
    review_status = Column(String(24), default="AI_CANDIDATE", index=True)  # AI_CANDIDATE, CONFIRMED, FALSE_POSITIVE, UNCERTAIN
    review_note = Column(Text, nullable=True)

    model_version = Column(String(32), default="yolov8n-sonar-baseline")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    survey = relationship("SurveyModel", back_populates="contacts")
    reviews = relationship("ReviewModel", back_populates="contact", cascade="all, delete-orphan")


class ReviewModel(Base):
    __tablename__ = "reviews"

    review_id = Column(Integer, primary_key=True, autoincrement=True)
    contact_id = Column(String(64), ForeignKey("contacts.contact_id", ondelete="CASCADE"), nullable=False, index=True)
    survey_id = Column(String(64), nullable=False)
    review_status = Column(String(24), nullable=False)
    review_note = Column(Text, nullable=True)
    model_version = Column(String(32), nullable=False)
    reviewed_at = Column(DateTime, default=datetime.datetime.utcnow)

    contact = relationship("ContactModel", back_populates="reviews")
