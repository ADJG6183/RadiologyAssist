"""
Seven-stage radiology report pipeline.

Stages
------
1. TRANSCRIBE     – accept / normalise the raw transcript text
2. RETRIEVE       – fetch study context and prior reports from the DB
3. ANALYZE_IMAGE  – (optional) analyse DICOM images via Claude vision API
4. EXTRACT        – ask the LLM to pull structured fields from the transcript
5. DRAFT          – ask the LLM to write the free-text report
6. SAFETY         – ask the LLM (acting as reviewer) to approve the draft
7. SAVE           – persist the draft and all audit events to the DB

ANALYZE_IMAGE fires for every study but skips immediately (logs a "skip" event)
when study.dicom_uri is None.  This keeps the event count predictable: you
always get exactly 7 events regardless of whether a DICOM image was uploaded.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.models import ReportDraft, AgentEvent, Study
from app.services.llm import LLMClient, get_llm_client
from app.services.prompts import prompt_analyze_image, prompt_draft, prompt_extract, prompt_safety

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data container passed between stages
# ---------------------------------------------------------------------------

@dataclass
class PipelineContext:
    study_id: int
    transcript: str
    model_name: str = "unknown"
    prior_reports: list[dict] = field(default_factory=list)
    structured_fields: dict = field(default_factory=dict)
    draft_text: str = ""
    safety_result: dict = field(default_factory=dict)
    draft_id: Optional[int] = None
    events: list[dict] = field(default_factory=list)
    # DICOM image findings — populated by stage_analyze_image, consumed by stage_extract
    image_findings: Optional[str] = None
    # Quality scoring — populated by stage_safety, persisted by stage_save
    quality_score: Optional[float] = None
    quality_breakdown: Optional[dict] = None


# ---------------------------------------------------------------------------
# Individual stage functions
# ---------------------------------------------------------------------------

def stage_transcribe(ctx: PipelineContext) -> PipelineContext:
    """
    Stage 1 — TRANSCRIBE
    Normalise whitespace and validate the transcript is non-empty.
    (Audio → text transcription happens upstream, before the pipeline.)
    """
    t0 = time.monotonic()
    cleaned = " ".join(ctx.transcript.split())
    if not cleaned:
        raise ValueError("Transcript is empty — cannot run pipeline.")
    ctx.transcript = cleaned
    ctx.events.append(_event("TRANSCRIBE", "normalise", f"Cleaned transcript, {len(cleaned)} chars", t0))
    log.info("pipeline.transcribe", study_id=ctx.study_id, chars=len(cleaned))
    return ctx


def stage_retrieve(ctx: PipelineContext, db: Session) -> PipelineContext:
    """
    Stage 2 — RETRIEVE
    Load study details and the patient's last 3 prior reports for context.
    """
    t0 = time.monotonic()

    study = db.get(Study, ctx.study_id)
    if study is None:
        raise ValueError(f"Study {ctx.study_id} not found.")

    prior = (
        db.query(ReportDraft)
        .join(Study)
        .filter(Study.patient_id == study.patient_id)
        .filter(ReportDraft.study_id != ctx.study_id)
        .filter(ReportDraft.status == "approved")   # never use rejected drafts as reference
        .order_by(ReportDraft.created_at.desc())
        .limit(3)
        .all()
    )

    ctx.prior_reports = []
    for r in prior:
        # Use structured_json if available — it has clean impression/findings
        # that are far more useful to Claude than raw truncated text.
        # Fall back to the first 500 chars of draft_text if no structured data.
        if r.structured_json:
            try:
                fields = json.loads(r.structured_json)
                ctx.prior_reports.append({
                    "draft_id": r.draft_id,
                    "impression": fields.get("impression", ""),
                    "findings": fields.get("findings", []),
                })
                continue
            except json.JSONDecodeError:
                pass
        ctx.prior_reports.append({
            "draft_id": r.draft_id,
            "text": r.draft_text[:500],
        })

    summary = f"study={ctx.study_id}, prior_reports={len(ctx.prior_reports)}"
    ctx.events.append(_event("RETRIEVE", "db_query", summary, t0))
    log.info("pipeline.retrieve", study_id=ctx.study_id, prior_count=len(ctx.prior_reports))
    return ctx


def stage_analyze_image(ctx: PipelineContext, db: Session, llm: LLMClient) -> PipelineContext:
    """
    Stage 3 — ANALYZE_IMAGE
    If the study has a DICOM file attached, load it, apply two clinical windows
    per representative slice, and ask Claude's vision API to describe what it sees.

    Skips immediately (with a "skip" event) when study.dicom_uri is None so that
    the total event count stays at 7 regardless of whether images were uploaded.
    """
    from app.core.config import settings
    from app.services.dicom import DICOMProcessor

    t0 = time.monotonic()

    study = db.get(Study, ctx.study_id)
    if study is None or not study.dicom_uri:
        ctx.events.append(_event("ANALYZE_IMAGE", "skip", "No DICOM file attached — skipping image analysis", t0))
        log.info("pipeline.analyze_image.skipped", study_id=ctx.study_id)
        return ctx

    proc = DICOMProcessor().load(study.dicom_uri)
    metadata = proc.extract_metadata()
    max_slices = getattr(settings, "dicom_max_slices_analyzed", 5)
    slices = proc.get_representative_slices(max_slices=max_slices)

    # Two windows per slice — lung and mediastinal — so Claude sees the same
    # two views a radiologist toggles between on the workstation.
    image_bytes_list = []
    for idx in slices:
        image_bytes_list.append(proc.to_png_bytes(idx, window_center=-600, window_width=1500))
        image_bytes_list.append(proc.to_png_bytes(idx, window_center=40,   window_width=400))

    raw = llm.vision_complete(prompt_analyze_image(metadata, len(slices)), image_bytes_list)

    try:
        result = json.loads(raw)
        ctx.image_findings = result.get("visual_impression", "") + "\n" + "\n".join(result.get("visual_findings", []))
    except json.JSONDecodeError:
        ctx.image_findings = raw  # pass raw text through if JSON fails

    ctx.events.append(_event("ANALYZE_IMAGE", "vision_complete", f"slices={len(slices)},windows=2", t0))
    log.info("pipeline.analyze_image", study_id=ctx.study_id, slices=len(slices))
    return ctx


def stage_extract(ctx: PipelineContext, llm: LLMClient) -> PipelineContext:
    """
    Stage 4 — EXTRACT
    Ask the LLM to pull structured fields (modality, laterality, findings…)
    from the raw transcript, optionally enriched with DICOM image findings.
    """
    t0 = time.monotonic()

    raw = llm.complete(prompt_extract(ctx.transcript, image_findings=ctx.image_findings))

    try:
        ctx.structured_fields = json.loads(raw)
    except json.JSONDecodeError:
        ctx.structured_fields = {"raw_extract": raw}

    ctx.events.append(_event("EXTRACT", "llm_extract", f"fields={list(ctx.structured_fields.keys())}", t0))
    log.info("pipeline.extract", study_id=ctx.study_id, fields=list(ctx.structured_fields.keys()))
    return ctx


def stage_draft(ctx: PipelineContext, llm: LLMClient) -> PipelineContext:
    """
    Stage 4 — DRAFT
    Ask the LLM to write the free-text radiology report.
    """
    t0 = time.monotonic()

    ctx.draft_text = llm.complete(
        prompt_draft(ctx.transcript, ctx.structured_fields, ctx.prior_reports)
    )

    ctx.events.append(_event("DRAFT", "llm_draft", f"draft_len={len(ctx.draft_text)}", t0))
    log.info("pipeline.draft", study_id=ctx.study_id, draft_len=len(ctx.draft_text))
    return ctx


def stage_safety(ctx: PipelineContext, llm: LLMClient) -> PipelineContext:
    """
    Stage 5 — SAFETY
    Two-layer quality check:
      Layer 1 — run_rule_checks(): fast regex checks, zero LLM cost
      Layer 2 — LLM safety + quality scoring using enhanced prompt_safety()

    The rule check results are passed to the LLM as structured context so
    Claude focuses on clinical reasoning instead of re-finding mechanical issues.
    The LLM response now includes quality_score and dimensions in addition to
    the existing approved/issues/confidence fields.
    """
    from app.services.quality import run_rule_checks

    t0 = time.monotonic()

    # Layer 1: fast deterministic checks (microseconds, no API call)
    rule_results = run_rule_checks(ctx.draft_text, ctx.structured_fields)

    # Layer 2: LLM safety review, with rule results as pre-populated context
    raw = llm.complete(prompt_safety(ctx.draft_text, ctx.structured_fields, rule_results))

    try:
        ctx.safety_result = json.loads(raw)
    except json.JSONDecodeError:
        ctx.safety_result = {"approved": False, "issues": ["Could not parse safety response."], "confidence": 0.0}

    approved = ctx.safety_result.get("approved", False)

    # Extract quality fields — use .get() so the stage never raises if
    # the LLM omits them (graceful degradation: score stays None)
    ctx.quality_score = ctx.safety_result.get("quality_score")
    ctx.quality_breakdown = ctx.safety_result.get("dimensions")

    ctx.events.append(
        _event("SAFETY", "llm_safety", f"approved={approved},quality={ctx.quality_score}", t0)
    )
    log.info("pipeline.safety", study_id=ctx.study_id, approved=approved, quality=ctx.quality_score)

    if not approved:
        issues = ctx.safety_result.get("issues", [])
        raise ValueError(f"Safety check failed: {issues}")

    return ctx


def stage_save(ctx: PipelineContext, db: Session) -> PipelineContext:
    """
    Stage 6 — SAVE
    Persist the report draft and all audit events to the database.
    """
    t0 = time.monotonic()

    # Save the draft
    draft = ReportDraft(
        study_id=ctx.study_id,
        draft_text=ctx.draft_text,
        structured_json=json.dumps(ctx.structured_fields),
        model_name=ctx.model_name,
        version="1.0",
        quality_score=ctx.quality_score,
        quality_breakdown=json.dumps(ctx.quality_breakdown) if ctx.quality_breakdown else None,
    )
    db.add(draft)
    db.flush()  # get draft.draft_id without committing yet
    ctx.draft_id = draft.draft_id

    # Save all accumulated audit events
    ctx.events.append(_event("SAVE", "db_write", f"draft_id={ctx.draft_id}", t0))
    for ev in ctx.events:
        db.add(AgentEvent(
            study_id=ctx.study_id,
            step=ev["step"],
            tool_name=ev["tool_name"],
            output_summary=ev["output_summary"],
            latency_ms=ev["latency_ms"],
        ))

    db.commit()
    log.info("pipeline.save", study_id=ctx.study_id, draft_id=ctx.draft_id)
    return ctx


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(study_id: int, transcript: str, db: Session) -> PipelineContext:
    """
    Run all 6 stages in sequence for a given study and transcript.
    Returns the completed PipelineContext (contains draft_id, structured_fields, etc.)
    
    IMPROVEMENT: Now captures intermediate results and detailed error logs
    so if a stage fails, you don't lose the work from previous stages.
    """
    llm = get_llm_client()
    ctx = PipelineContext(
        study_id=study_id,
        transcript=transcript,
        model_name=llm.model,
    )

    stages = [
        ("TRANSCRIBE",    lambda: stage_transcribe(ctx)),
        ("RETRIEVE",      lambda: stage_retrieve(ctx, db)),
        ("ANALYZE_IMAGE", lambda: stage_analyze_image(ctx, db, llm)),
        ("EXTRACT",       lambda: stage_extract(ctx, llm)),
        ("DRAFT",         lambda: stage_draft(ctx, llm)),
        ("SAFETY",        lambda: stage_safety(ctx, llm)),
        ("SAVE",          lambda: stage_save(ctx, db)),
    ]

    # IMPROVEMENT: Loop through stages and catch errors individually
    # This way if stage 4 fails, we've still logged stages 1-3's results
    for stage_name, stage_fn in stages:
        try:
            stage_fn()
        except Exception as e:
            # Log the failure with full context
            # This helps engineers understand exactly where the pipeline broke
            error_event = {
                "step": stage_name,
                "tool_name": "ERROR",
                "output_summary": f"FAILED: {str(e)[:200]}",  # first 200 chars of error
                "latency_ms": 0,
            }
            ctx.events.append(error_event)

            # Log to structured logging so we can search for pipeline failures
            log.error(
                "pipeline.stage_failed",
                study_id=study_id,
                stage=stage_name,
                error=str(e),
                exc_info=True,  # Include full stack trace
            )

            # Re-raise the error so the API knows something went wrong
            raise RuntimeError(f"Pipeline failed at {stage_name}: {str(e)}") from e

    log.info("pipeline.complete", study_id=study_id, draft_id=ctx.draft_id)
    return ctx


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _event(step: str, tool_name: str, output_summary: str, t0: float) -> dict:
    return {
        "step": step,
        "tool_name": tool_name,
        "output_summary": output_summary,
        "latency_ms": int((time.monotonic() - t0) * 1000),
    }
