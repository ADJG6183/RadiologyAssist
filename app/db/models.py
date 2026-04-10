from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Integer, String, Text, Date, DateTime, Float,
    ForeignKey, func, CheckConstraint, Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"

    patient_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mrn: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    date_of_birth: Mapped[datetime] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    studies: Mapped[List["Study"]] = relationship(back_populates="patient")


class Study(Base):
    __tablename__ = "studies"
    __table_args__ = (
        # IMPROVEMENT: Indexes on frequently queried columns
        # These make queries like "find studies for patient X" or "order studies by date" fast
        Index("ix_studies_patient_id", "patient_id"),
        Index("ix_studies_created_at", "created_at"),
    )

    study_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.patient_id"), nullable=False)
    study_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    modality: Mapped[str] = mapped_column(String(50), nullable=False)
    institution: Mapped[Optional[str]] = mapped_column(String(200))
    # DICOM image path — None if no image was uploaded for this study.
    # The pipeline's ANALYZE_IMAGE stage skips cleanly when this is None.
    dicom_uri: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    # Pre-analyzed DICOM findings — populated by POST /studies/{id}/analyze.
    # The pipeline reads this rather than re-running analysis on every /run call.
    image_findings: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    patient: Mapped["Patient"] = relationship(back_populates="studies")
    report_inputs: Mapped[List["ReportInput"]] = relationship(back_populates="study")
    report_drafts: Mapped[List["ReportDraft"]] = relationship(back_populates="study")
    agent_events: Mapped[List["AgentEvent"]] = relationship(back_populates="study")


class ReportInput(Base):
    __tablename__ = "report_inputs"
    __table_args__ = (
        # IMPROVEMENT: Index on study_id so we can quickly find inputs for a study
        Index("ix_report_inputs_study_id", "study_id"),
    )

    input_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("studies.study_id"), nullable=False)
    transcript_text: Mapped[Optional[str]] = mapped_column(Text)
    audio_uri: Mapped[Optional[str]] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    study: Mapped["Study"] = relationship(back_populates="report_inputs")


class ReportDraft(Base):
    __tablename__ = "report_drafts"
    __table_args__ = (
        # Database-level guard — prevents any value other than these three
        # from ever being written to status, even via direct DB access.
        CheckConstraint("status IN ('draft', 'approved', 'rejected')", name="ck_report_drafts_status"),
        # IMPROVEMENT: Indexes for common query patterns
        # list_studies() needs to find the latest draft per study (study_id + order by created_at)
        # The pipeline queries drafts by status, so that's indexed too
        Index("ix_report_drafts_study_id", "study_id"),
        Index("ix_report_drafts_status", "status"),
        Index("ix_report_drafts_created_at", "created_at"),
    )

    draft_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("studies.study_id"), nullable=False)
    draft_text: Mapped[str] = mapped_column(Text, nullable=False)
    structured_json: Mapped[Optional[str]] = mapped_column(Text)
    model_name: Mapped[Optional[str]] = mapped_column(String(100))
    version: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Sign-off fields — populated when a radiologist approves or rejects the draft.
    # status defaults to "draft"; valid values: "draft" | "approved" | "rejected"
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="draft")
    actioned_by: Mapped[Optional[str]] = mapped_column(String(200))   # who approved or rejected
    actioned_at: Mapped[Optional[datetime]] = mapped_column(DateTime)  # when they did it
    rejection_reason: Mapped[Optional[str]] = mapped_column(Text)      # only set on rejection

    # Quality scoring — populated by stage_safety via the enhanced prompt.
    # nullable=True so existing draft rows are unaffected by this schema change.
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quality_breakdown: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string

    # DICOM image analysis findings — populated by stage_analyze_image.
    # None when no DICOM was uploaded or pixel decode failed.
    image_findings: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    study: Mapped["Study"] = relationship(back_populates="report_drafts")


class AgentEvent(Base):
    __tablename__ = "agent_events"
    __table_args__ = (
        # IMPROVEMENT: Index on study_id so we can quickly load audit trail for a study
        Index("ix_agent_events_study_id", "study_id"),
    )

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("studies.study_id"), nullable=False)
    step: Mapped[Optional[str]] = mapped_column(String(50))
    tool_name: Mapped[Optional[str]] = mapped_column(String(100))
    output_summary: Mapped[Optional[str]] = mapped_column(Text)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    study: Mapped["Study"] = relationship(back_populates="agent_events")
