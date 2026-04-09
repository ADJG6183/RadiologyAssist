"""
Pydantic schemas for API request and response bodies.

These are separate from the SQLAlchemy ORM models in app/db/models.py.
The ORM models represent database rows; schemas represent what the API
accepts and returns over HTTP. Keeping them separate lets us control
exactly which fields are exposed and how they're shaped.
"""

import json
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# IMPROVEMENT: Enums for constrained fields
# These ensure only valid values enter the system right at the API gateway.
# ---------------------------------------------------------------------------

class Modality(str, Enum):
    """Valid radiology imaging modalities.
    
    By inheriting from str, these enums work seamlessly in Pydantic and APIs.
    Pydantic automatically validates: if the API gets modality="Toaster",
    it returns a 422 error before any business logic runs.
    """
    CT = "CT"
    MRI = "MRI"
    XRAY = "X-Ray"
    ULTRASOUND = "Ultrasound"
    PET = "PET"
    FLUOROSCOPY = "Fluoroscopy"
    NUCLEAR_MEDICINE = "Nuclear Medicine"


class DraftStatus(str, Enum):
    """Valid states for a report draft.
    
    A draft starts as 'draft', then moves to either 'approved' or 'rejected'.
    Once in approved/rejected, it cannot change.
    """
    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Request schemas (inbound)
# ---------------------------------------------------------------------------

class PatientIn(BaseModel):
    mrn: str
    first_name: str
    last_name: str
    date_of_birth: date


class StudyIn(BaseModel):
    patient: PatientIn
    study_date: date
    modality: Modality  # IMPROVEMENT: Now type-safe! Only valid values allowed.
    institution: Optional[str] = None


class DictationJSON(BaseModel):
    transcript_text: str


class ApproveIn(BaseModel):
    actioned_by: str   # radiologist name or ID — will become a real user ref once auth is added


class RejectIn(BaseModel):
    actioned_by: str       # who is rejecting
    rejection_reason: str  # required — radiologist must explain what needs fixing


# ---------------------------------------------------------------------------
# Response schemas (outbound)
# ---------------------------------------------------------------------------

class StudyRead(BaseModel):
    study_id: int
    patient_id: int
    modality: str
    study_date: date
    institution: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class RunResult(BaseModel):
    study_id: int
    draft_id: int
    message: str


class ReportDraftRead(BaseModel):
    draft_id: int
    study_id: int
    draft_text: str
    structured_json: Any          # stored as Text in DB; we parse it before returning
    model_name: Optional[str] = None
    created_at: datetime
    # Sign-off fields
    status: DraftStatus = DraftStatus.DRAFT  # IMPROVEMENT: Type-safe status
    actioned_by: Optional[str] = None
    actioned_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    # Quality scoring fields — None if the pipeline ran before this feature was added
    quality_score: Optional[float] = None
    quality_breakdown: Optional[Any] = None  # parsed from JSON string, like structured_json

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm_with_parsed_json(cls, draft) -> "ReportDraftRead":
        """
        ORM → schema, converting the stored JSON strings to real dicts.
        SQLAlchemy stores structured_json and quality_breakdown as plain Text
        so the column stays compatible with MS SQL. We parse both here so the
        API response contains objects, not raw strings.
        """
        raw = draft.structured_json
        parsed = json.loads(raw) if raw else None
        raw_qb = draft.quality_breakdown
        parsed_qb = json.loads(raw_qb) if raw_qb else None
        return cls(
            draft_id=draft.draft_id,
            study_id=draft.study_id,
            draft_text=draft.draft_text,
            structured_json=parsed,
            model_name=draft.model_name,
            created_at=draft.created_at,
            status=draft.status,
            actioned_by=draft.actioned_by,
            actioned_at=draft.actioned_at,
            rejection_reason=draft.rejection_reason,
            quality_score=draft.quality_score,
            quality_breakdown=parsed_qb,
        )


class AgentEventRead(BaseModel):
    event_id: int
    study_id: int
    step: Optional[str] = None
    tool_name: Optional[str] = None
    output_summary: Optional[str] = None
    latency_ms: Optional[int] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# UI list / detail schemas — richer than StudyRead; include patient info
# and the latest draft status so the studies table can show everything in
# one request.
# ---------------------------------------------------------------------------

class PatientRead(BaseModel):
    patient_id: int
    mrn: str
    first_name: str
    last_name: str
    date_of_birth: date
    created_at: datetime

    model_config = {"from_attributes": True}


class StudyListItem(BaseModel):
    """One row in the studies table — study + patient name + draft status."""
    study_id: int
    patient_id: int
    patient_name: str       # "First Last" — computed in the endpoint
    mrn: str
    modality: str
    study_date: date
    institution: Optional[str] = None
    created_at: datetime
    latest_draft_status: Optional[str] = None   # null if no draft yet


class StudyDetailRead(BaseModel):
    """Full study card shown in the detail panel."""
    study_id: int
    modality: str
    study_date: date
    institution: Optional[str] = None
    created_at: datetime
    patient: PatientRead
    latest_draft_status: Optional[str] = None
