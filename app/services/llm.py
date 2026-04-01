"""
LLM service layer.

LLMClient is the abstract interface. All pipeline code calls .complete()
and never cares which provider is underneath.

Providers
---------
MockLLMClient       – deterministic, no API key, used in dev and all tests
AnthropicLLMClient  – Claude via the Anthropic SDK (LLM_PROVIDER=anthropic)
"""

import json
from abc import ABC, abstractmethod

from app.core.config import settings


class LLMClient(ABC):
    """
    Abstract base: every provider must implement complete().

    The `model` class attribute is used by the pipeline to record
    which model produced a given draft.  Subclasses should override it.
    """

    model: str = "unknown"

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send a prompt, return the model's text response."""


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
