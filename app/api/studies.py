"""
REST endpoints for studies.

All routes are registered on a router mounted at /api/v1 in main.py.

Flow for a typical request:
  1. Client creates a study (POST /studies)
  2. Client submits a transcript (POST /studies/{id}/dictation/json)
  3. Client triggers the pipeline (POST /studies/{id}/run)
  4. Client retrieves the draft (GET /studies/{id}/draft)
  5. Client inspects the audit trail (GET /studies/{id}/events)
"""

import os
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.agents.pipeline import run_pipeline
from app.api.schemas import (
    AgentEventRead,
    DictationJSON,
    ReportDraftRead,
    RunResult,
    StudyIn,
    StudyRead,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.db.connection import get_db
from app.db.models import AgentEvent, Patient, ReportDraft, ReportInput, Study

router = APIRouter()
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# POST /studies
# Create a new patient (or reuse existing by MRN) and attach a study.
# ---------------------------------------------------------------------------

@router.post("/studies", response_model=StudyRead, status_code=201)
def create_study(body: StudyIn, db: Session = Depends(get_db)):
    """
    Create a study.  If a patient with the supplied MRN already exists
    the existing record is reused; otherwise a new patient row is inserted.
    This mirrors typical RIS behaviour where the patient is looked up by MRN.
    """
    # Upsert patient by MRN
    patient = db.query(Patient).filter(Patient.mrn == body.patient.mrn).first()
    if patient is None:
        patient = Patient(
            mrn=body.patient.mrn,
            first_name=body.patient.first_name,
            last_name=body.patient.last_name,
            date_of_birth=body.patient.date_of_birth,
        )
        db.add(patient)
        db.flush()  # assigns patient_id without committing the transaction yet

    study = Study(
        patient_id=patient.patient_id,
        study_date=body.study_date,
        modality=body.modality,
        institution=body.institution,
    )
    db.add(study)
    db.commit()
    db.refresh(study)

    log.info("api.create_study", study_id=study.study_id, patient_id=patient.patient_id)
    return study


# ---------------------------------------------------------------------------
# POST /studies/{study_id}/dictation
# Multipart upload: either a text field called "transcript" or an audio file.
# ---------------------------------------------------------------------------

@router.post("/studies/{study_id}/dictation", status_code=201)
def submit_dictation_multipart(
    study_id: int,
    transcript: Optional[str] = Form(default=None),
    audio: Optional[UploadFile] = File(default=None),
    db: Session = Depends(get_db),
):
    """
    Accept a transcript via multipart form data.

    Two modes:
    - `transcript` form field: plain text dictation (no audio transcription needed)
    - `audio` file upload: saved to disk; a future transcription service would
      convert it to text.  For now we store the URI and leave transcript_text null.

    One of the two must be provided.
    """
    _require_study(db, study_id)

    if transcript is None and audio is None:
        raise HTTPException(status_code=422, detail="Provide either 'transcript' text or an 'audio' file.")

    audio_uri: Optional[str] = None

    if audio is not None:
        # Save the file to the configured upload directory.
        upload_dir = settings.audio_upload_dir
        os.makedirs(upload_dir, exist_ok=True)
        dest = os.path.join(upload_dir, f"study_{study_id}_{audio.filename}")
        with open(dest, "wb") as fh:
            fh.write(audio.file.read())
        audio_uri = dest
        log.info("api.audio_saved", study_id=study_id, path=dest)

    report_input = ReportInput(
        study_id=study_id,
        transcript_text=transcript,
        audio_uri=audio_uri,
    )
    db.add(report_input)
    db.commit()

    return {"input_id": report_input.input_id, "study_id": study_id}


# ---------------------------------------------------------------------------
# POST /studies/{study_id}/dictation/json
# JSON body variant — simpler for programmatic clients.
# ---------------------------------------------------------------------------

@router.post("/studies/{study_id}/dictation/json", status_code=201)
def submit_dictation_json(
    study_id: int,
    body: DictationJSON,
    db: Session = Depends(get_db),
):
    """
    Submit a transcript as a JSON body.  Equivalent to the multipart endpoint
    but easier to call from test clients or curl.
    """
    _require_study(db, study_id)

    report_input = ReportInput(
        study_id=study_id,
        transcript_text=body.transcript_text,
    )
    db.add(report_input)
    db.commit()

    log.info("api.dictation_json", study_id=study_id, chars=len(body.transcript_text))
    return {"input_id": report_input.input_id, "study_id": study_id}


# ---------------------------------------------------------------------------
# POST /studies/{study_id}/run
# Trigger the 6-stage pipeline for the most recent transcript.
# ---------------------------------------------------------------------------

@router.post("/studies/{study_id}/run", response_model=RunResult)
def run_study_pipeline(study_id: int, db: Session = Depends(get_db)):
    """
    Run the full 6-stage pipeline (TRANSCRIBE → RETRIEVE → EXTRACT →
    DRAFT → SAFETY → SAVE) using the study's most recent transcript.

    Returns the generated draft_id on success.
    Raises 422 if no transcript exists yet.
    Raises 500 if the safety check rejects the draft.
    """
    _require_study(db, study_id)

    # Grab the latest transcript for this study
    latest_input = (
        db.query(ReportInput)
        .filter(ReportInput.study_id == study_id)
        .filter(ReportInput.transcript_text.isnot(None))
        .order_by(ReportInput.created_at.desc())
        .first()
    )
    if latest_input is None:
        raise HTTPException(
            status_code=422,
            detail="No transcript found for this study. Submit a dictation first.",
        )

    try:
        ctx = run_pipeline(study_id, latest_input.transcript_text, db)
    except ValueError as exc:
        # Pipeline raises ValueError for safety failures and validation errors.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    log.info("api.pipeline_complete", study_id=study_id, draft_id=ctx.draft_id)
    return RunResult(
        study_id=study_id,
        draft_id=ctx.draft_id,
        message="Pipeline completed successfully.",
    )


# ---------------------------------------------------------------------------
# GET /studies/{study_id}/draft
# Return the most recently generated report draft.
# ---------------------------------------------------------------------------

@router.get("/studies/{study_id}/draft", response_model=ReportDraftRead)
def get_draft(study_id: int, db: Session = Depends(get_db)):
    """Return the latest report draft for a study."""
    _require_study(db, study_id)

    draft = (
        db.query(ReportDraft)
        .filter(ReportDraft.study_id == study_id)
        .order_by(ReportDraft.created_at.desc())
        .first()
    )
    if draft is None:
        raise HTTPException(status_code=404, detail="No draft found. Run the pipeline first.")

    return ReportDraftRead.from_orm_with_parsed_json(draft)


# ---------------------------------------------------------------------------
# GET /studies/{study_id}/events
# Return the full audit trail for a study.
# ---------------------------------------------------------------------------

@router.get("/studies/{study_id}/events", response_model=list[AgentEventRead])
def get_events(study_id: int, db: Session = Depends(get_db)):
    """Return all pipeline audit events for a study, oldest first."""
    _require_study(db, study_id)

    events = (
        db.query(AgentEvent)
        .filter(AgentEvent.study_id == study_id)
        .order_by(AgentEvent.created_at.asc())
        .all()
    )
    return events


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _require_study(db: Session, study_id: int) -> Study:
    """Raise 404 if the study doesn't exist. Reused across endpoints."""
    study = db.get(Study, study_id)
    if study is None:
        raise HTTPException(status_code=404, detail=f"Study {study_id} not found.")
    return study
