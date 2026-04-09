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
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.agents.pipeline import run_pipeline
from app.services.transcription import get_transcription_client
from sqlalchemy import select as sa_select

from app.api.schemas import (
    AgentEventRead,
    ApproveIn,
    DictationJSON,
    PatientRead,
    RejectIn,
    ReportDraftRead,
    RunResult,
    StudyDetailRead,
    StudyIn,
    StudyListItem,
    StudyRead,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.db.connection import get_db
from app.db.models import AgentEvent, Patient, ReportDraft, ReportInput, Study

router = APIRouter()
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# GET /studies
# Return all studies with patient name and latest draft status in one shot.
# The UI studies table needs all three pieces — doing it in one query
# avoids the classic "N+1" problem (one query per study to fetch patient).
# ---------------------------------------------------------------------------

@router.get("/studies", response_model=list[StudyListItem])
def list_studies(db: Session = Depends(get_db)):
    """
    List all studies, newest first.

    Uses a correlated scalar subquery to fetch each study's latest draft
    status without a separate round-trip per row.
    A correlated subquery is like a formula in a spreadsheet cell — it
    runs once *per row* and looks sideways at data in a related table.
    """
    # This subquery asks: "for THIS study row, what is the status of the
    # most-recently-created draft?" — correlate(Study) tells SQLAlchemy
    # the Study.study_id reference should look at the outer query's row.
    latest_status_sq = (
        sa_select(ReportDraft.status)
        .where(ReportDraft.study_id == Study.study_id)
        .order_by(ReportDraft.created_at.desc())
        .limit(1)
        .correlate(Study)
        .scalar_subquery()
    )

    rows = (
        db.query(Study, Patient, latest_status_sq)
        .join(Patient, Study.patient_id == Patient.patient_id)
        .order_by(Study.created_at.desc())
        .all()
    )

    return [
        StudyListItem(
            study_id=study.study_id,
            patient_id=patient.patient_id,
            patient_name=f"{patient.first_name} {patient.last_name}",
            mrn=patient.mrn,
            modality=study.modality,
            study_date=study.study_date,
            institution=study.institution,
            created_at=study.created_at,
            latest_draft_status=status,
        )
        for study, patient, status in rows
    ]


# ---------------------------------------------------------------------------
# GET /studies/{study_id}
# Return a single study with full patient info — used by the detail panel.
# ---------------------------------------------------------------------------

@router.get("/studies/{study_id}", response_model=StudyDetailRead)
def get_study(study_id: int, db: Session = Depends(get_db)):
    """Return one study with its patient record embedded."""
    study = _require_study(db, study_id)

    latest_draft = (
        db.query(ReportDraft)
        .filter(ReportDraft.study_id == study_id)
        .order_by(ReportDraft.created_at.desc())
        .first()
    )

    return StudyDetailRead(
        study_id=study.study_id,
        modality=study.modality,
        study_date=study.study_date,
        institution=study.institution,
        created_at=study.created_at,
        patient=PatientRead.model_validate(study.patient),
        latest_draft_status=latest_draft.status if latest_draft else None,
    )


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
        # Step 1 — Save the raw audio file to disk first.
        # We do this before transcription so the file is never lost,
        # even if the transcription API call fails afterward.
        upload_dir = settings.audio_upload_dir
        os.makedirs(upload_dir, exist_ok=True)
        # Include a timestamp so re-uploads never overwrite existing files.
        # e.g. study_1_1712620800_recording.wav
        ts = int(time.time())
        dest = os.path.join(upload_dir, f"study_{study_id}_{ts}_{audio.filename}")
        with open(dest, "wb") as fh:
            fh.write(audio.file.read())
        audio_uri = dest
        log.info("api.audio_saved", study_id=study_id, path=dest)

        # Step 2 — Write the DB record immediately so the audio URI is always
        # tracked, even if transcription fails. This means the file on disk
        # always has a corresponding row in the database.
        report_input = ReportInput(
            study_id=study_id,
            transcript_text=None,  # filled in below if transcription succeeds
            audio_uri=audio_uri,
        )
        db.add(report_input)
        db.flush()  # get input_id without committing yet

        # Step 3 — Transcribe. If this fails, we commit the row with
        # transcript_text=NULL so the audio is still tracked, then raise
        # so the caller knows transcription didn't succeed.
        try:
            transcriber = get_transcription_client()
            transcript = transcriber.transcribe(dest)
            report_input.transcript_text = transcript
            log.info("api.transcription_complete", study_id=study_id, chars=len(transcript))
        except ValueError as exc:
            # Pre-flight failure (bad format, file too large) — commit the
            # audio record, then surface the error.
            db.commit()
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            # API failure (network, quota, etc.) — same: save the record first.
            db.commit()
            log.error("api.transcription_failed", study_id=study_id, error=str(exc))
            raise HTTPException(
                status_code=502,
                detail=f"Transcription service error: {exc}",
            ) from exc

        db.commit()
        return {"input_id": report_input.input_id, "study_id": study_id}

    # Text-only path (no audio) — write the record normally
    report_input = ReportInput(
        study_id=study_id,
        transcript_text=transcript,
        audio_uri=None,
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

    # Block re-run if the study already has an approved draft.
    # An approved report is final — the radiologist must reject it first
    # if a new version is needed.
    existing_draft = (
        db.query(ReportDraft)
        .filter(ReportDraft.study_id == study_id)
        .order_by(ReportDraft.created_at.desc())
        .first()
    )
    if existing_draft and existing_draft.status == "approved":
        raise HTTPException(
            status_code=409,
            detail="This study already has an approved report. Reject it first to generate a new draft.",
        )

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
    draft = _require_draft(db, study_id)
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
# POST /studies/{study_id}/draft/approve
# Radiologist signs off — marks the draft as approved.
# ---------------------------------------------------------------------------

@router.post("/studies/{study_id}/draft/approve", response_model=ReportDraftRead)
def approve_draft(study_id: int, body: ApproveIn, db: Session = Depends(get_db)):
    """
    Approve the latest draft for a study.

    Records who approved it and when. Only a draft with status="draft"
    can be approved — attempting to approve an already-actioned draft
    returns 409 Conflict.

    409 Conflict means: "your request is valid, but the current state of
    the resource prevents it." It's the right code for state machine violations.
    """
    _require_study(db, study_id)
    draft = _require_draft(db, study_id)

    if draft.status != "draft":
        raise HTTPException(
            status_code=409,
            detail=f"Draft is already '{draft.status}' and cannot be approved.",
        )

    draft.status = "approved"
    draft.actioned_by = body.actioned_by
    draft.actioned_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(draft)

    log.info("api.draft_approved", study_id=study_id, draft_id=draft.draft_id, by=body.actioned_by)
    return ReportDraftRead.from_orm_with_parsed_json(draft)


# ---------------------------------------------------------------------------
# POST /studies/{study_id}/draft/reject
# Radiologist sends it back — marks the draft as rejected with a reason.
# ---------------------------------------------------------------------------

@router.post("/studies/{study_id}/draft/reject", response_model=ReportDraftRead)
def reject_draft(study_id: int, body: RejectIn, db: Session = Depends(get_db)):
    """
    Reject the latest draft for a study.

    Requires a rejection_reason so there is always a clear record of
    what needs to be fixed. Only a draft with status="draft" can be
    rejected — attempting to reject an already-actioned draft returns 409.
    """
    _require_study(db, study_id)
    draft = _require_draft(db, study_id)

    if draft.status != "draft":
        raise HTTPException(
            status_code=409,
            detail=f"Draft is already '{draft.status}' and cannot be rejected.",
        )

    draft.status = "rejected"
    draft.actioned_by = body.actioned_by
    draft.actioned_at = datetime.now(timezone.utc)
    draft.rejection_reason = body.rejection_reason
    db.commit()
    db.refresh(draft)

    log.info(
        "api.draft_rejected",
        study_id=study_id,
        draft_id=draft.draft_id,
        by=body.actioned_by,
        reason=body.rejection_reason,
    )
    return ReportDraftRead.from_orm_with_parsed_json(draft)


# ---------------------------------------------------------------------------
# POST /studies/{study_id}/dicom
# Upload a DICOM file and attach it to the study.  The pipeline's
# ANALYZE_IMAGE stage will use it on the next /run call.
# ---------------------------------------------------------------------------

@router.post("/studies/{study_id}/dicom", status_code=201)
def upload_dicom(
    study_id: int,
    dicom_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a DICOM file for a study.

    Saves the file to dicom_upload_dir and records the path in study.dicom_uri.
    On the next /run call, ANALYZE_IMAGE will load this file and pass windowed
    PNG slices to Claude's vision API before the EXTRACT stage runs.

    Calling this endpoint again with a new file overwrites the previous URI —
    only the most-recently-uploaded DICOM is used by the pipeline.
    """
    study = _require_study(db, study_id)

    upload_dir = settings.dicom_upload_dir
    os.makedirs(upload_dir, exist_ok=True)
    ts = int(time.time())
    dest = os.path.join(upload_dir, f"study_{study_id}_{ts}_{dicom_file.filename}")
    with open(dest, "wb") as fh:
        fh.write(dicom_file.file.read())

    study.dicom_uri = dest
    db.commit()

    log.info("api.dicom_uploaded", study_id=study_id, path=dest)
    return {"study_id": study_id, "dicom_uri": dest}


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _require_study(db: Session, study_id: int) -> Study:
    """Raise 404 if the study doesn't exist. Reused across endpoints."""
    study = db.get(Study, study_id)
    if study is None:
        raise HTTPException(status_code=404, detail=f"Study {study_id} not found.")
    return study


def _require_draft(db: Session, study_id: int) -> ReportDraft:
    """Raise 404 if no draft exists for this study. Reused by sign-off endpoints."""
    draft = (
        db.query(ReportDraft)
        .filter(ReportDraft.study_id == study_id)
        .order_by(ReportDraft.created_at.desc())
        .first()
    )
    if draft is None:
        raise HTTPException(status_code=404, detail="No draft found. Run the pipeline first.")
    return draft
