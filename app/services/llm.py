"""
LLM service layer.

LLMClient is the abstract interface. All pipeline code calls .complete()
and .vision_complete() and never cares which provider is underneath.

Providers
---------
MockLLMClient       – deterministic, no API key, used in dev and all tests
AnthropicLLMClient  – Claude via the Anthropic SDK (LLM_PROVIDER=anthropic)
"""

import base64
import json
from abc import ABC, abstractmethod

from app.core.config import settings


class LLMClient(ABC):
    """
    Abstract base: every provider must implement complete() and vision_complete().

    The `model` class attribute is used by the pipeline to record
    which model produced a given draft.  Subclasses should override it.
    """

    model: str = "unknown"

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send a prompt, return the model's text response."""

    @abstractmethod
    def vision_complete(self, prompt: str, image_bytes_list: list) -> str:
        """
        Send a prompt alongside one or more images (PNG bytes), return text.

        image_bytes_list: list of bytes objects, each a PNG-encoded image.
        The images are sent as base64-encoded data in the request.
        """


class MockLLMClient(LLMClient):
    """
    Deterministic mock — no network calls, no API key.
    Inspects the prompt to decide which canned response to return.
    """

    model = "MockLLMClient"

    def complete(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        # EXTRACT stage: match the unique phrase from prompt_extract()
        if "clinical data extraction system" in prompt_lower:
            return json.dumps({
                "modality": "CT",
                "body_part": "Chest",
                "laterality": "bilateral",
                "technique": "without contrast",
                "clinical_history": "",
                "findings": [
                    "No acute cardiopulmonary process.",
                    "Heart size within normal limits.",
                    "No pleural effusion.",
                ],
                "impression": "Unremarkable CT chest.",
                "critical_findings": [],
                "recommendations": [],
            })

        # SAFETY stage: match the unique phrase from prompt_safety()
        if "checks to perform" in prompt_lower:
            return json.dumps({
                "approved": True,
                "issues": [],
                "confidence": 0.97,
                "quality_score": 0.92,
                "dimensions": {
                    "completeness": 0.95,
                    "consistency": 0.90,
                    "clinical_accuracy": 0.93,
                    "format_compliance": 0.90,
                },
            })

        # DRAFT stage: generate the free-text report
        return (
            "RADIOLOGY REPORT\n"
            "----------------\n"
            "Technique: CT of the chest without contrast.\n\n"
            "Findings:\n"
            "  Lungs: Clear. No consolidation, effusion, or pneumothorax.\n"
            "  Heart: Normal size and contour.\n"
            "  Mediastinum: Unremarkable.\n\n"
            "Impression:\n"
            "  No acute cardiopulmonary process.\n"
        )

    def vision_complete(self, prompt: str, image_bytes_list: list) -> str:
        """Return canned image analysis JSON regardless of image content."""
        return json.dumps({
            "visual_findings": [
                "Lungs appear clear bilaterally on lung window.",
                "No pleural effusion or pneumothorax identified.",
                "Heart size within normal limits on mediastinal window.",
            ],
            "visual_impression": "No acute findings identified on DICOM image review.",
            "image_quality": "Diagnostic",
            "windows_reviewed": ["lung", "mediastinal"],
        })


class AnthropicLLMClient(LLMClient):
    """
    Claude via the Anthropic SDK.

    Set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY=sk-ant-... in .env.

    We import `anthropic` lazily (inside __init__) so that the rest of the
    app still imports cleanly in TEST_MODE even if the SDK isn't installed.
    """

    model = "claude-sonnet-4-6"

    def __init__(self):
        import anthropic  # noqa: PLC0415 — intentionally lazy import
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def complete(self, prompt: str) -> str:
        """
        Send a single-turn message and return the text content.

        max_tokens=2048 is generous for radiology reports (~300-600 words).
        We strip markdown code fences before returning because Claude
        sometimes wraps JSON in ```json ... ``` despite being told not to.
        """
        message = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return _strip_code_fences(message.content[0].text)

    def vision_complete(self, prompt: str, image_bytes_list: list) -> str:
        """
        Send PNG images + prompt to Claude's vision API, return text.

        Each item in image_bytes_list is base64-encoded and sent as an
        image/png content block before the text prompt. Claude processes
        all images in a single request — no N round-trips needed.
        """
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(img).decode(),
                },
            }
            for img in image_bytes_list
        ]
        content.append({"type": "text", "text": prompt})

        message = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        return _strip_code_fences(message.content[0].text)


def _strip_code_fences(text: str) -> str:
    """
    Remove markdown code fences if present.
    Handles ```json ... ```, ``` ... ```, and leading/trailing whitespace.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence line and the closing fence
        lines = stripped.splitlines()
        # Remove first line (```json or ```) and last line (```)
        inner = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        return "\n".join(inner).strip()
    return stripped


def get_llm_client() -> LLMClient:
    """
    Factory — returns the right client based on LLM_PROVIDER in .env.
    All pipeline code calls this instead of instantiating a client directly.
    """
    provider = settings.llm_provider.lower()

    if provider == "mock":
        return MockLLMClient()

    if provider == "anthropic":
        return AnthropicLLMClient()

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Valid values: mock, anthropic")
