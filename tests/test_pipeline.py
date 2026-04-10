"""
Pipeline stage unit tests.

These tests call pipeline stage functions directly — no HTTP involved.
This lets us test the kitchen independently of the front door.

Why test stages directly?
  - Faster feedback: a stage test runs in milliseconds
  - Precise failure messages: if EXTRACT breaks, only EXTRACT tests fail
  - Easier to test edge cases (empty transcript, bad LLM output, etc.)
"""

import json
import pytest

from app.agents.pipeline import (
    PipelineContext,
    run_pipeline,
    stage_analyze_image,
    stage_draft,
    stage_extract,
    stage_retrieve,
    stage_safety,
    stage_transcribe,
)
from app.db.models import Patient, ReportDraft, Study
from app.services.llm import MockLLMClient
from datetime import date


# ---------------------------------------------------------------------------
# Helpers — build minimal DB records for tests that need them
# ---------------------------------------------------------------------------

def _seed_study(db):
    """Insert one patient and one study, return the study_id."""
    patient = Patient(
        mrn="MRN-TEST",
        first_name="Test",
        last_name="Patient",
        date_of_birth=date(1990, 1, 1),
    )
    db.add(patient)
    db.flush()

    study = Study(
        patient_id=patient.patient_id,
        study_date=date(2026, 4, 1),
        modality="CT",
    )
    db.add(study)
    db.commit()
    return study.study_id


def _ctx(study_id=1, transcript="CT chest normal."):
    """Build a minimal PipelineContext for testing a single stage."""
    return PipelineContext(study_id=study_id, transcript=transcript)


# ---------------------------------------------------------------------------
# stage_transcribe
# ---------------------------------------------------------------------------

def test_transcribe_cleans_whitespace():
    """
    Extra spaces, tabs, and newlines should be collapsed to single spaces.
    The pipeline receives raw dictation which often has messy spacing.
    """
    ctx = _ctx(transcript="  CT   chest\n\tnormal.  ")
    result = stage_transcribe(ctx)
    assert result.transcript == "CT chest normal."


def test_transcribe_records_event():
    """Every stage must log an event — the audit trail depends on this."""
    ctx = _ctx()
    result = stage_transcribe(ctx)
    assert len(result.events) == 1
    assert result.events[0]["step"] == "TRANSCRIBE"


def test_transcribe_allows_empty_transcript():
    """
    An empty transcript is now allowed — the radiologist may be running the
    pipeline from DICOM image analysis alone (no dictation recorded).
    The API endpoint validates that at least one content source exists before
    calling run_pipeline(), so stage_transcribe no longer needs to guard this.
    The stage should complete cleanly and record a descriptive event summary.
    """
    ctx = _ctx(transcript="   ")
    result = stage_transcribe(ctx)
    assert len(result.events) == 1
    assert result.events[0]["step"] == "TRANSCRIBE"
    assert result.transcript == ""   # whitespace collapsed to empty string


# ---------------------------------------------------------------------------
# stage_retrieve
# ---------------------------------------------------------------------------

def test_retrieve_loads_study(db):
    """
    RETRIEVE should find the study in the database and record zero prior
    reports for a brand-new patient.
    """
    study_id = _seed_study(db)
    ctx = _ctx(study_id=study_id)
    result = stage_retrieve(ctx, db)
    assert result.prior_reports == []
    assert result.events[0]["step"] == "RETRIEVE"


def test_retrieve_raises_on_missing_study(db):
    """If the study_id doesn't exist in the DB, the pipeline should fail clearly."""
    ctx = _ctx(study_id=99999)
    with pytest.raises(ValueError, match="not found"):
        stage_retrieve(ctx, db)


def test_retrieve_loads_prior_reports(db):
    """
    If the patient had previous studies with drafts, RETRIEVE should load
    up to 3 of them for context.
    """
    study_id = _seed_study(db)

    # Add a prior APPROVED draft — rejected drafts are excluded from context (fix #4)
    prior = ReportDraft(
        study_id=study_id,
        draft_text="Prior report text.",
        model_name="MockLLMClient",
        version="1.0",
        status="approved",
    )
    db.add(prior)
    db.commit()

    # Create a second study for the same patient to retrieve against
    study = db.get(Study, study_id)
    study2 = Study(
        patient_id=study.patient_id,
        study_date=date(2026, 4, 2),
        modality="MRI",
    )
    db.add(study2)
    db.commit()

    ctx = _ctx(study_id=study2.study_id)
    result = stage_retrieve(ctx, db)
    assert len(result.prior_reports) == 1
    assert "Prior report" in result.prior_reports[0]["text"]


# ---------------------------------------------------------------------------
# stage_analyze_image
# ---------------------------------------------------------------------------

def test_analyze_image_skips_when_no_dicom_uri(db):
    """
    When study.dicom_uri is None, ANALYZE_IMAGE must skip cleanly and log
    a 'skip' event.  ctx.image_findings stays None so stage_extract gets
    no image context — that is the correct degraded behaviour.
    """
    study_id = _seed_study(db)
    ctx = _ctx(study_id=study_id)
    llm = MockLLMClient()

    result = stage_analyze_image(ctx, db, llm)

    # image_findings is not populated — no DICOM was available
    assert result.image_findings is None
    # But an event was still logged (keeps event count predictable)
    assert len(result.events) == 1
    assert result.events[0]["step"] == "ANALYZE_IMAGE"
    assert "skip" in result.events[0]["tool_name"]


# ---------------------------------------------------------------------------
# stage_extract
# ---------------------------------------------------------------------------

def test_extract_returns_structured_fields():
    """
    EXTRACT should call the LLM and parse the JSON response into a dict
    stored on the context.
    """
    llm = MockLLMClient()
    ctx = _ctx()
    result = stage_extract(ctx, llm)
    assert isinstance(result.structured_fields, dict)
    assert "modality" in result.structured_fields
    assert "findings" in result.structured_fields


def test_prompt_extract_includes_image_findings_when_present():
    """
    When image_findings is set on the context, prompt_extract() should include
    an IMAGE ANALYSIS FINDINGS section in the prompt so the LLM can reconcile
    the dictation with what was actually seen on the images.
    """
    from app.services.prompts import prompt_extract
    prompt = prompt_extract("CT chest normal.", image_findings="No pleural effusion on lung window.")
    assert "IMAGE ANALYSIS FINDINGS" in prompt
    assert "No pleural effusion" in prompt


def test_extract_handles_invalid_json_gracefully():
    """
    If the LLM returns something that isn't valid JSON, the stage should
    not crash — it should store the raw text under 'raw_extract' instead.
    This is defensive programming: the AI can misbehave.
    """
    class BrokenLLM(MockLLMClient):
        def complete(self, prompt):
            return "This is not JSON at all."

    ctx = _ctx()
    result = stage_extract(ctx, BrokenLLM())
    assert "raw_extract" in result.structured_fields


# ---------------------------------------------------------------------------
# stage_draft
# ---------------------------------------------------------------------------

def test_draft_produces_non_empty_text():
    """DRAFT should return a non-empty string from the LLM."""
    llm = MockLLMClient()
    ctx = _ctx()
    ctx.structured_fields = {"modality": "CT", "findings": ["clear"]}
    result = stage_draft(ctx, llm)
    assert isinstance(result.draft_text, str)
    assert len(result.draft_text) > 0


# ---------------------------------------------------------------------------
# stage_safety
# ---------------------------------------------------------------------------

def test_safety_approves_valid_draft():
    """The mock LLM always approves — safety should pass without raising."""
    llm = MockLLMClient()
    ctx = _ctx()
    ctx.draft_text = "RADIOLOGY REPORT\nFindings: normal."
    result = stage_safety(ctx, llm)
    assert result.safety_result["approved"] is True


def test_safety_stores_quality_score():
    """
    The enhanced safety stage should extract quality_score and dimensions
    from the LLM response and store them on the context.
    These fields power the quality scoring feature.
    """
    llm = MockLLMClient()
    ctx = _ctx()
    ctx.draft_text = "RADIOLOGY REPORT\nFindings: normal."
    result = stage_safety(ctx, llm)
    assert result.quality_score == 0.92
    assert isinstance(result.quality_breakdown, dict)
    assert "completeness" in result.quality_breakdown
    assert result.quality_breakdown["completeness"] == 0.95


def test_safety_quality_score_none_on_missing_fields():
    """
    If the LLM omits quality fields (e.g. older model, malformed response),
    quality_score stays None — no crash, no validation error.
    This is called graceful degradation: the pipeline keeps working even
    when optional enrichment is missing.
    """
    class NoQualityLLM(MockLLMClient):
        def complete(self, prompt):
            import json
            return json.dumps({
                "approved": True,
                "issues": [],
                "confidence": 0.9,
                # deliberately omitting quality_score and dimensions
            })

    ctx = _ctx()
    ctx.draft_text = "RADIOLOGY REPORT\nFindings: normal."
    result = stage_safety(ctx, NoQualityLLM())
    assert result.quality_score is None
    assert result.quality_breakdown is None


def test_safety_raises_when_not_approved():
    """
    If the safety reviewer rejects the draft, the pipeline must stop.
    We simulate a rejecting LLM to test this branch.
    """
    class RejectingLLM(MockLLMClient):
        def complete(self, prompt):
            return json.dumps({
                "approved": False,
                "issues": ["Laterality conflict detected."],
                "confidence": 0.3,
            })

    ctx = _ctx()
    ctx.draft_text = "Bad draft."
    with pytest.raises(ValueError, match="Safety check failed"):
        stage_safety(ctx, RejectingLLM())


# ---------------------------------------------------------------------------
# run_pipeline (full orchestrator)
# ---------------------------------------------------------------------------

def test_run_pipeline_end_to_end(db):
    """
    Run the full 6-stage pipeline through the orchestrator.
    This is the closest thing to an integration test at the pipeline level.
    """
    study_id = _seed_study(db)
    ctx = run_pipeline(study_id, "CT chest without contrast. Lungs clear.", db)

    # A draft was saved to the database
    assert ctx.draft_id is not None

    # 7 events were recorded (one per stage, including ANALYZE_IMAGE skip)
    assert len(ctx.events) == 7
    assert ctx.events[2]["step"] == "ANALYZE_IMAGE"

    # The model name was captured
    assert ctx.model_name == "MockLLMClient"

    # The draft actually exists in the DB
    draft = db.get(ReportDraft, ctx.draft_id)
    assert draft is not None
    assert len(draft.draft_text) > 0

    # Quality score was persisted
    assert draft.quality_score == 0.92
    assert draft.quality_breakdown is not None  # stored as JSON string
