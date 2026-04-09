"""
API endpoint tests.

Each test uses the `client` fixture from conftest.py, which gives us:
  - A fresh in-memory SQLite database
  - A TestClient that talks to our FastAPI app

Reading the assertions:
  assert response.status_code == 201   → the server returned HTTP 201 Created
  assert data["study_id"] == 1         → the JSON body had study_id equal to 1
"""

import pytest


# ---------------------------------------------------------------------------
# Helpers — reusable request bodies so we don't repeat ourselves
# ---------------------------------------------------------------------------

PATIENT = {
    "mrn": "MRN001",
    "first_name": "Jane",
    "last_name": "Doe",
    "date_of_birth": "1980-05-12",
}

STUDY_BODY = {
    "patient": PATIENT,
    "study_date": "2026-04-01",
    "modality": "CT",
    "institution": "General Hospital",
}

TRANSCRIPT = "CT chest without contrast. Lungs clear bilaterally. No pleural effusion. Heart normal."


def _create_study(client):
    """Helper: create a study and return the parsed JSON response."""
    r = client.post("/api/v1/studies", json=STUDY_BODY)
    assert r.status_code == 201
    return r.json()


def _submit_transcript(client, study_id, text=TRANSCRIPT):
    """Helper: submit a transcript and return the parsed JSON response."""
    r = client.post(
        f"/api/v1/studies/{study_id}/dictation/json",
        json={"transcript_text": text},
    )
    assert r.status_code == 201
    return r.json()


# ---------------------------------------------------------------------------
# POST /studies
# ---------------------------------------------------------------------------

def test_create_study_returns_201(client):
    """Creating a study should return 201 and the study data."""
    r = client.post("/api/v1/studies", json=STUDY_BODY)
    assert r.status_code == 201
    data = r.json()
    assert data["study_id"] == 1
    assert data["patient_id"] == 1
    assert data["modality"] == "CT"
    assert data["institution"] == "General Hospital"


def test_create_study_reuses_existing_patient(client):
    """
    If the same MRN is submitted twice, the second study should attach to
    the existing patient row rather than creating a duplicate.
    """
    study1 = _create_study(client)
    study2 = _create_study(client)
    # Two studies, but both should share the same patient_id
    assert study1["patient_id"] == study2["patient_id"]
    # They get different study IDs
    assert study1["study_id"] != study2["study_id"]


def test_create_study_without_institution(client):
    """Institution is optional — omitting it should still work."""
    body = {**STUDY_BODY, "institution": None}
    r = client.post("/api/v1/studies", json=body)
    assert r.status_code == 201
    assert r.json()["institution"] is None


# ---------------------------------------------------------------------------
# POST /studies/{id}/dictation/json
# ---------------------------------------------------------------------------

def test_submit_dictation_json(client):
    """Submitting a transcript should return 201 with an input_id."""
    study = _create_study(client)
    r = client.post(
        f"/api/v1/studies/{study['study_id']}/dictation/json",
        json={"transcript_text": TRANSCRIPT},
    )
    assert r.status_code == 201
    data = r.json()
    assert data["input_id"] == 1
    assert data["study_id"] == study["study_id"]


def test_submit_dictation_to_missing_study_returns_404(client):
    """Submitting a transcript to a study that doesn't exist should 404."""
    r = client.post(
        "/api/v1/studies/999/dictation/json",
        json={"transcript_text": TRANSCRIPT},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /studies/{id}/run
# ---------------------------------------------------------------------------

def test_run_pipeline_returns_draft_id(client):
    """Running the pipeline should return a draft_id."""
    study = _create_study(client)
    _submit_transcript(client, study["study_id"])

    r = client.post(f"/api/v1/studies/{study['study_id']}/run")
    assert r.status_code == 200
    data = r.json()
    assert data["draft_id"] == 1
    assert data["study_id"] == study["study_id"]
    assert "Pipeline completed" in data["message"]


def test_run_pipeline_without_transcript_returns_422(client):
    """
    Running the pipeline before submitting a transcript should return 422
    (Unprocessable Entity — the request is valid JSON but we can't act on it).
    """
    study = _create_study(client)
    r = client.post(f"/api/v1/studies/{study['study_id']}/run")
    assert r.status_code == 422


def test_run_pipeline_on_missing_study_returns_404(client):
    r = client.post("/api/v1/studies/999/run")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /studies/{id}/draft
# ---------------------------------------------------------------------------

def test_get_draft_returns_report(client):
    """After running the pipeline, the draft endpoint should return a report."""
    study = _create_study(client)
    _submit_transcript(client, study["study_id"])
    client.post(f"/api/v1/studies/{study['study_id']}/run")

    r = client.get(f"/api/v1/studies/{study['study_id']}/draft")
    assert r.status_code == 200
    data = r.json()

    # The draft text should be a non-empty string
    assert isinstance(data["draft_text"], str)
    assert len(data["draft_text"]) > 0

    # structured_json should come back as a real object, not a string
    assert isinstance(data["structured_json"], dict)
    assert "modality" in data["structured_json"]

    # The model name should be recorded
    assert data["model_name"] == "MockLLMClient"

    # Quality score should be present — the enhanced safety stage populates it
    assert data["quality_score"] == 0.92
    assert isinstance(data["quality_breakdown"], dict)
    assert "completeness" in data["quality_breakdown"]


def test_get_draft_before_pipeline_returns_404(client):
    """Asking for a draft before the pipeline has run should 404."""
    study = _create_study(client)
    r = client.get(f"/api/v1/studies/{study['study_id']}/draft")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /studies/{id}/events
# ---------------------------------------------------------------------------

def test_get_events_returns_seven_stages(client):
    """
    After running the pipeline, the events endpoint should return exactly
    7 audit rows — one per pipeline stage (ANALYZE_IMAGE logs a skip event
    when no DICOM is attached, keeping the count predictable).
    """
    study = _create_study(client)
    _submit_transcript(client, study["study_id"])
    client.post(f"/api/v1/studies/{study['study_id']}/run")

    r = client.get(f"/api/v1/studies/{study['study_id']}/events")
    assert r.status_code == 200
    events = r.json()

    assert len(events) == 7
    stage_names = [e["step"] for e in events]
    assert stage_names == ["TRANSCRIBE", "RETRIEVE", "ANALYZE_IMAGE", "EXTRACT", "DRAFT", "SAFETY", "SAVE"]


def test_upload_dicom_returns_201(client, tmp_path):
    """
    Uploading a DICOM file should return 201 and record the path in dicom_uri.
    We send a dummy .dcm file — the endpoint only saves bytes and records the
    path; it doesn't parse the DICOM until the pipeline runs.
    """
    study = _create_study(client)
    fake_dcm = tmp_path / "test.dcm"
    fake_dcm.write_bytes(b"DICM" + b"\x00" * 128)  # minimal valid-looking DICOM header

    with open(fake_dcm, "rb") as f:
        r = client.post(
            f"/api/v1/studies/{study['study_id']}/dicom",
            files={"dicom_file": ("test.dcm", f, "application/octet-stream")},
        )
    assert r.status_code == 201
    data = r.json()
    assert data["study_id"] == study["study_id"]
    assert "dicom_uri" in data
    assert data["dicom_uri"].endswith("test.dcm")


def test_get_events_before_pipeline_returns_empty_list(client):
    """No pipeline run means no events — should return an empty list, not a 404."""
    study = _create_study(client)
    r = client.get(f"/api/v1/studies/{study['study_id']}/events")
    assert r.status_code == 200
    assert r.json() == []


# ---------------------------------------------------------------------------
# Full end-to-end flow
# ---------------------------------------------------------------------------

def test_full_flow(client):
    """
    Run the complete happy path in one test:
    create study → submit transcript → run pipeline → get draft → get events.
    This mirrors exactly what you tested manually with curl.
    """
    # 1. Create study
    study = _create_study(client)
    study_id = study["study_id"]

    # 2. Submit transcript
    inp = _submit_transcript(client, study_id)
    assert inp["input_id"] is not None

    # 3. Run pipeline
    run = client.post(f"/api/v1/studies/{study_id}/run").json()
    assert run["draft_id"] is not None

    # 4. Get draft
    draft = client.get(f"/api/v1/studies/{study_id}/draft").json()
    assert len(draft["draft_text"]) > 0
    assert draft["structured_json"]["modality"] == "CT"

    # 5. Get events — all 7 stages logged (ANALYZE_IMAGE logs skip event when no DICOM)
    events = client.get(f"/api/v1/studies/{study_id}/events").json()
    assert len(events) == 7
