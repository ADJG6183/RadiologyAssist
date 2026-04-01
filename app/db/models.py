from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Integer, String, Text, Date, DateTime,
    ForeignKey, func,
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

    study_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.patient_id"), nullable=False)
    study_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    modality: Mapped[str] = mapped_column(String(50), nullable=False)
    institution: Mapped[Optional[str]] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    patient: Mapped["Patient"] = relationship(back_populates="studies")
    report_inputs: Mapped[List["ReportInput"]] = relationship(back_populates="study")
    report_drafts: Mapped[List["ReportDraft"]] = relationship(back_populates="study")
    agent_events: Mapped[List["AgentEvent"]] = relationship(back_populates="study")


class ReportInput(Base):
    __tablename__ = "report_inputs"

    input_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("studies.study_id"), nullable=False)
    transcript_text: Mapped[Optional[str]] = mapped_column(Text)
    audio_uri: Mapped[Optional[str]] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    study: Mapped["Study"] = relationship(back_populates="report_inputs")


class ReportDraft(Base):
    __tablename__ = "report_drafts"

    draft_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("studies.study_id"), nullable=False)
    draft_text: Mapped[str] = mapped_column(Text, nullable=False)
    structured_json: Mapped[Optional[str]] = mapped_column(Text)
    model_name: Mapped[Optional[str]] = mapped_column(String(100))
    version: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    study: Mapped["Study"] = relationship(back_populates="report_drafts")


class AgentEvent(Base):
    __tablename__ = "agent_events"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("studies.study_id"), nullable=False)
    step: Mapped[Optional[str]] = mapped_column(String(50))
    tool_name: Mapped[Optional[str]] = mapped_column(String(100))
    output_summary: Mapped[Optional[str]] = mapped_column(Text)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    study: Mapped["Study"] = relationship(back_populates="agent_events")
