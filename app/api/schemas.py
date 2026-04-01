"""
Pydantic schemas for API request and response bodies.

These are separate from the SQLAlchemy ORM models in app/db/models.py.
The ORM models represent database rows; schemas represent what the API
accepts and returns over HTTP. Keeping them separate lets us control
exactly which fields are exposed and how they're shaped.
"""

import json
from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel


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
    modality: str
    institution: Optional[str] = None


class DictationJSON(BaseModel):
    transcript_text: str


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

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm_with_parsed_json(cls, draft) -> "ReportDraftRead":
        """
        ORM → schema, converting the stored JSON string to a real dict.
        SQLAlchemy stores structured_json as plain Text so the column
        stays compatible with MS SQL. We parse it here so the API
        response contains an object, not a raw string.
        """
        raw = draft.structured_json
        parsed = json.loads(raw) if raw else None
        return cls(
            draft_id=draft.draft_id,
            study_id=draft.study_id,
            draft_text=draft.draft_text,
            structured_json=parsed,
            model_name=draft.model_name,
            created_at=draft.created_at,
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
