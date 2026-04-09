"""
Transcription service tests.

We test three things:
1. MockTranscriptionClient — returns canned text, no file needed
2. OpenAITranscriptionClient pre-flight checks — bad format and file too large
   (we test our guard code, not the Whisper API itself)
3. The multipart dictation endpoint — audio upload now stores transcript text

For test #2 we use tmp_path, a built-in pytest fixture that creates a
temporary directory that is automatically cleaned up after each test.
Think of it as a scratch pad — write files there, pytest deletes them.
"""

import os
import pytest

os.environ["TEST_MODE"] = "1"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["TRANSCRIPTION_PROVIDER"] = "mock"

from app.services.transcription import (
    MAX_FILE_BYTES,
    MockTranscriptionClient,
    OpenAITranscriptionClient,
)


# ---------------------------------------------------------------------------
# MockTranscriptionClient
# ---------------------------------------------------------------------------

def test_mock_transcription_returns_string():
    """Mock client should return a non-empty string without touching any file."""
    client = MockTranscriptionClient()
    result = client.transcribe("/fake/path/audio.wav")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# OpenAITranscriptionClient — pre-flight validation
# We instantiate the class without an API key by patching the openai import.
# We only care about the format/size checks, which run before any API call.
# ---------------------------------------------------------------------------

def test_openai_rejects_unsupported_format(tmp_path):
    """
    Uploading a .flac file (not supported by Whisper) should raise
    ValueError before any API call is made.

    tmp_path is a pytest built-in fixture — it gives us a real temporary
    directory we can write files into. The file doesn't need real audio
    content; we just need it to exist so stat() doesn't fail.
    """
    bad_file = tmp_path / "recording.flac"
    bad_file.write_bytes(b"fake audio content")

    # We can't instantiate OpenAITranscriptionClient without a key,
    # so we test the validation logic directly via the class method.
    # Bypass __init__ by creating the object without calling it.
    client = OpenAITranscriptionClient.__new__(OpenAITranscriptionClient)

    with pytest.raises(ValueError, match="Unsupported audio format"):
        client.transcribe(str(bad_file))


def test_openai_rejects_oversized_file(tmp_path):
    """
    A file over 25 MB should raise ValueError before any API call.
    We fake a large file by writing a small file then monkey-patching
    pathlib's stat() result — writing 25 MB of real data would be slow.
    """
    audio_file = tmp_path / "big_recording.wav"
    audio_file.write_bytes(b"fake audio")

    client = OpenAITranscriptionClient.__new__(OpenAITranscriptionClient)

    # Temporarily make the file appear oversized by subclassing Path.
    # This is a common pattern: test the branch without the cost.
    from unittest.mock import patch
    oversized_stat = os.stat_result((
        0o100644, 0, 0, 1, 0, 0,
        MAX_FILE_BYTES + 1,  # st_size — one byte over the limit
        0, 0, 0,
    ))
    with patch("pathlib.Path.stat", return_value=oversized_stat):
        with pytest.raises(ValueError, match="exceeds the Whisper API limit"):
            client.transcribe(str(audio_file))


def test_openai_accepts_valid_wav_format(tmp_path):
    """
    A valid .wav file under the size limit should pass pre-flight checks
    and proceed to the API call. We verify it doesn't raise on format/size.
    We stop before the actual API call by giving the client a fake _client
    that raises a recognizable error.
    """
    audio_file = tmp_path / "dictation.wav"
    audio_file.write_bytes(b"fake audio content")

    client = OpenAITranscriptionClient.__new__(OpenAITranscriptionClient)

    class FakeOpenAI:
        class audio:
            class transcriptions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("reached_api_call")

    client._client = FakeOpenAI()

    # Pre-flight checks pass, then we hit the fake API and get our marker error
    with pytest.raises(RuntimeError, match="reached_api_call"):
        client.transcribe(str(audio_file))


# ---------------------------------------------------------------------------
# Dictation endpoint — audio upload now transcribes
# ---------------------------------------------------------------------------

STUDY_BODY = {
    "patient": {
        "mrn": "MRN-AUDIO",
        "first_name": "Audio",
        "last_name": "Test",
        "date_of_birth": "1990-01-01",
    },
    "study_date": "2026-04-09",
    "modality": "CT",
}


def test_audio_upload_stores_transcript(client, tmp_path):
    """
    Uploading an audio file via the multipart endpoint should:
    1. Save the file to disk
    2. Call the transcription service (mock returns canned text)
    3. Store the transcript_text in the DB (not NULL)

    We verify #3 by immediately running the pipeline — which would fail
    with a 422 if transcript_text were still NULL.
    """
    # Create study
    study = client.post("/api/v1/studies", json=STUDY_BODY).json()
    study_id = study["study_id"]

    # Create a fake audio file
    fake_audio = tmp_path / "dictation.wav"
    fake_audio.write_bytes(b"fake audio bytes")

    # Upload as multipart
    with open(fake_audio, "rb") as f:
        r = client.post(
            f"/api/v1/studies/{study_id}/dictation",
            files={"audio": ("dictation.wav", f, "audio/wav")},
        )

    assert r.status_code == 201
    data = r.json()
    assert data["study_id"] == study_id

    # If transcript_text was stored, the pipeline can run.
    # A NULL transcript_text would return 422 "No transcript found".
    run = client.post(f"/api/v1/studies/{study_id}/run")
    assert run.status_code == 200, f"Pipeline failed — transcript probably NULL: {run.json()}"


def test_audio_upload_and_json_both_work(client, tmp_path):
    """
    Both dictation paths (audio upload and JSON) should produce a runnable
    pipeline. This confirms the two code paths are equivalent in outcome.
    """
    # Audio path
    study_a = client.post("/api/v1/studies", json={**STUDY_BODY, "patient": {**STUDY_BODY["patient"], "mrn": "MRN-A"}}).json()
    fake_audio = tmp_path / "audio.wav"
    fake_audio.write_bytes(b"fake")
    with open(fake_audio, "rb") as f:
        client.post(f"/api/v1/studies/{study_a['study_id']}/dictation", files={"audio": ("audio.wav", f, "audio/wav")})
    run_a = client.post(f"/api/v1/studies/{study_a['study_id']}/run")
    assert run_a.status_code == 200

    # JSON path
    study_b = client.post("/api/v1/studies", json={**STUDY_BODY, "patient": {**STUDY_BODY["patient"], "mrn": "MRN-B"}}).json()
    client.post(f"/api/v1/studies/{study_b['study_id']}/dictation/json", json={"transcript_text": "CT chest normal."})
    run_b = client.post(f"/api/v1/studies/{study_b['study_id']}/run")
    assert run_b.status_code == 200
