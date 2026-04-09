"""
Audio transcription service.

Mirrors the structure of app/services/llm.py — an abstract base class
with a single method, one mock implementation for tests, and one real
implementation per provider.

Providers
---------
MockTranscriptionClient  – returns canned text, no API call, used in tests
OpenAITranscriptionClient – calls the Whisper API via the openai SDK

The factory function get_transcription_client() reads TRANSCRIPTION_PROVIDER
from config, so swapping providers requires only a .env change.

Constraints
-----------
- Whisper API accepts: mp3, mp4, mpeg, mpga, m4a, wav, webm
- Whisper API max file size: 25 MB
- Requires OPENAI_API_KEY when provider = "openai"
"""

from abc import ABC, abstractmethod
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

# Formats the Whisper API accepts.
SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}

# 25 MB in bytes — the Whisper API hard limit.
MAX_FILE_BYTES = 25 * 1024 * 1024


class TranscriptionClient(ABC):
    """
    Abstract base — every provider must implement transcribe().
    Callers pass a file path; the client returns the transcript as a string.
    """

    @abstractmethod
    def transcribe(self, file_path: str) -> str:
        """Read an audio file and return the transcribed text."""


class MockTranscriptionClient(TranscriptionClient):
    """
    Deterministic mock for tests — no API call, no file reading.
    Returns a realistic dictation string so the pipeline has something to work with.
    """

    def transcribe(self, file_path: str) -> str:
        log.info("transcription.mock", file_path=file_path)
        return (
            "CT chest without contrast. Lungs are clear bilaterally. "
            "No pneumothorax. No pleural effusion. "
            "Cardiac silhouette normal in size. "
            "Mediastinum unremarkable. No acute osseous abnormality."
        )


class OpenAITranscriptionClient(TranscriptionClient):
    """
    Whisper via the OpenAI SDK.

    The openai SDK is imported lazily so the rest of the app still loads
    cleanly when the package isn't installed (e.g. if someone only wants
    to run in mock mode).

    Raises ValueError before making any API call if:
    - The file format is not supported by Whisper
    - The file exceeds the 25 MB API limit
    """

    def __init__(self):
        import openai  # noqa: PLC0415 — intentional lazy import
        self._client = openai.OpenAI(api_key=settings.openai_api_key)

    def transcribe(self, file_path: str) -> str:
        path = Path(file_path)

        # --- Pre-flight checks before spending money on an API call ---

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format '{suffix}'. "
                f"Whisper accepts: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )

        file_size = path.stat().st_size
        if file_size > MAX_FILE_BYTES:
            raise ValueError(
                f"Audio file is {file_size / 1024 / 1024:.1f} MB — "
                f"exceeds the Whisper API limit of 25 MB."
            )

        log.info("transcription.start", file_path=file_path, size_bytes=file_size)

        with open(file_path, "rb") as audio_file:
            response = self._client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                # Hint to Whisper that this is medical dictation.
                # This improves accuracy for clinical terminology.
                prompt="Radiology dictation. Medical terminology.",
            )

        log.info("transcription.complete", chars=len(response.text))
        return response.text


def get_transcription_client() -> TranscriptionClient:
    """
    Factory — returns the right client based on TRANSCRIPTION_PROVIDER in .env.
    """
    provider = settings.transcription_provider.lower()

    if provider == "mock":
        return MockTranscriptionClient()

    if provider == "openai":
        return OpenAITranscriptionClient()

    raise ValueError(
        f"Unknown TRANSCRIPTION_PROVIDER: {provider!r}. "
        f"Valid values: mock, openai"
    )
