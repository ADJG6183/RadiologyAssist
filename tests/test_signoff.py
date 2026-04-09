"""
Sign-off endpoint tests.

We test:
1. Approve happy path — status flips, who/when recorded
2. Reject happy path — status flips, reason recorded
3. State machine — can't approve/reject an already-actioned draft
4. GET /draft now includes status fields
5. 404 before pipeline has run
"""

import os
os.environ["TEST_MODE"] = "1"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["TRANSCRIPTION_PROVIDER"] = "mock"

STUDY_BODY = {
    "patient": {
        "mrn": "MRN-SIGNOFF",
        "first_name": "Sign",
        "last_name": "Off",
        "date_of_birth": "1975-03-20",
    },
    "study_date": "2026-04-09",
    "modality": "MRI",
}

TRANSCRIPT = "MRI brain without contrast. No acute intracranial abnormality."


def _full_setup(client):
    """Create study, submit transcript, run pipeline. Returns study_id."""
    study = client.post("/api/v1/studies", json=STUDY_BODY).json()
    study_id = study["study_id"]
    client.post(
        f"/api/v1/studies/{study_id}/dictation/json",
        json={"transcript_text": TRANSCRIPT},
    )
    client.post(f"/api/v1/studies/{study_id}/run")
    return study_id


# ---------------------------------------------------------------------------
# Approve
# ---------------------------------------------------------------------------

def test_approve_draft_returns_approved_status(client):
    """
    Approving a draft should flip status to 'approved' and record
    who approved it and when.
    """
    study_id = _full_setup(client)

    r = client.post(
        f"/api/v1/studies/{study_id}/draft/approve",
        json={"actioned_by": "Dr. Smith"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "approved"
    assert data["actioned_by"] == "Dr. Smith"
    assert data["actioned_at"] is not None
    assert data["rejection_reason"] is None


def test_approve_updates_get_draft(client):
    """
    After approval, GET /draft should reflect the new status —
    not return stale 'draft' data.
    """
    study_id = _full_setup(client)
    client.post(
        f"/api/v1/studies/{study_id}/draft/approve",
        json={"actioned_by": "Dr. Jones"},
    )

    draft = client.get(f"/api/v1/studies/{study_id}/draft").json()
    assert draft["status"] == "approved"
    assert draft["actioned_by"] == "Dr. Jones"


# ---------------------------------------------------------------------------
# Reject
# ---------------------------------------------------------------------------

def test_reject_draft_returns_rejected_status(client):
    """
    Rejecting a draft should flip status to 'rejected' and record
    the reason — so there is always a clear audit trail.
    """
    study_id = _full_setup(client)

    r = client.post(
        f"/api/v1/studies/{study_id}/draft/reject",
        json={
            "actioned_by": "Dr. Chen",
            "rejection_reason": "Laterality incorrect — says left, should be right.",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "rejected"
    assert data["actioned_by"] == "Dr. Chen"
    assert data["rejection_reason"] == "Laterality incorrect — says left, should be right."
    assert data["actioned_at"] is not None


# ---------------------------------------------------------------------------
# State machine — these tests protect the integrity of the sign-off record
# ---------------------------------------------------------------------------

def test_cannot_approve_already_approved_draft(client):
    """
    Approving twice should return 409 Conflict.
    A signed report must not be silently overwritten.
    """
    study_id = _full_setup(client)
    client.post(
        f"/api/v1/studies/{study_id}/draft/approve",
        json={"actioned_by": "Dr. Smith"},
    )

    r = client.post(
        f"/api/v1/studies/{study_id}/draft/approve",
        json={"actioned_by": "Dr. Other"},
    )
    assert r.status_code == 409
    assert "approved" in r.json()["detail"]


def test_cannot_reject_already_approved_draft(client):
    """
    You can't reject a draft that's already been approved — 409 Conflict.
    """
    study_id = _full_setup(client)
    client.post(
        f"/api/v1/studies/{study_id}/draft/approve",
        json={"actioned_by": "Dr. Smith"},
    )

    r = client.post(
        f"/api/v1/studies/{study_id}/draft/reject",
        json={"actioned_by": "Dr. Other", "rejection_reason": "Too late."},
    )
    assert r.status_code == 409


def test_cannot_approve_already_rejected_draft(client):
    """
    You can't approve a draft that's already been rejected — 409 Conflict.
    The radiologist must run the pipeline again to get a new draft.
    """
    study_id = _full_setup(client)
    client.post(
        f"/api/v1/studies/{study_id}/draft/reject",
        json={"actioned_by": "Dr. Smith", "rejection_reason": "Needs revision."},
    )

    r = client.post(
        f"/api/v1/studies/{study_id}/draft/approve",
        json={"actioned_by": "Dr. Smith"},
    )
    assert r.status_code == 409


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_approve_before_pipeline_returns_404(client):
    """Can't approve a draft that doesn't exist yet."""
    study = client.post("/api/v1/studies", json={
        **STUDY_BODY,
        "patient": {**STUDY_BODY["patient"], "mrn": "MRN-NODRAFT"},
    }).json()
    r = client.post(
        f"/api/v1/studies/{study['study_id']}/draft/approve",
        json={"actioned_by": "Dr. Smith"},
    )
    assert r.status_code == 404


def test_new_draft_status_is_draft(client):
    """
    Immediately after the pipeline runs, the draft status should be 'draft'
    — not approved or rejected.
    """
    study_id = _full_setup(client)
    draft = client.get(f"/api/v1/studies/{study_id}/draft").json()
    assert draft["status"] == "draft"
    assert draft["actioned_by"] is None
    assert draft["actioned_at"] is None
    assert draft["rejection_reason"] is None
