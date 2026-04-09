# CLAUDE.md — Radiology AI Assistant

This file is read automatically by Claude Code at the start of every session.
It tells Claude how to behave in this project.

---

## Teaching Mode

The owner of this project is actively learning software engineering and ML engineering
concepts while building. Claude must follow these rules in every session:

- **Explain every non-trivial decision** before writing code — what it does, why this
  approach was chosen over alternatives.
- **Use analogies and plain language** first, then introduce the technical term.
  Target level: someone who has never heard the concept before.
  Example: "A correlated subquery is like a formula in a spreadsheet that looks sideways
  at a related table — it runs once per row."
- **Flag constraints proactively** — Python version limitations, API rate limits, DB
  compatibility issues, performance trade-offs. Don't wait to be asked.
- **Deliberate like an engineer** — when multiple approaches exist, briefly explain
  the trade-offs and state which one was chosen and why. Don't just pick one silently.
- **Point out what was wrong in previous drafts** — if the user brings a draft plan
  or code, critique it honestly before improving it.
- **Teach from failures** — when a test fails or a bug is found, explain the root cause
  in plain terms before fixing it. The fix is less valuable than understanding why.
- **After every significant change**, summarize: what changed, why it was designed this
  way, and what a developer should remember about it going forward.

---

## Project Overview

Production-style AI backend: radiologist dictation → structured radiology report.

**Tech stack:** Python 3.9, FastAPI, SQLAlchemy 2.0, Pydantic v2, structlog, Claude (Anthropic SDK), Whisper (OpenAI SDK), SQLite (dev/test) / MS SQL (prod).

**Pipeline (currently 6 stages):**
```
TRANSCRIBE → RETRIEVE → EXTRACT → DRAFT → SAFETY → SAVE
```
DICOM image analysis (ANALYZE_IMAGE stage) is planned — will insert between RETRIEVE and EXTRACT.

---

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Run in mock mode (no API keys needed — all LLM calls use MockLLMClient)
TEST_MODE=1 uvicorn main:app --reload

# Run with real Claude (requires ANTHROPIC_API_KEY in .env)
LLM_PROVIDER=anthropic TEST_MODE=1 uvicorn main:app --reload

# Run all tests (always use TEST_MODE=1 — never hits real APIs)
TEST_MODE=1 pytest -v

# Open interactive API docs
# http://localhost:8000/docs

# Open the UI
# http://localhost:8000
```

---

## Key Architecture Decisions

### Abstract Base Class pattern for services
`LLMClient` and `TranscriptionClient` are ABCs in `app/services/llm.py` and
`app/services/transcription.py`. Pipeline code calls `.complete()` and never
knows which provider is underneath. Swap providers by changing `LLM_PROVIDER`
in `.env` — no code changes required.

### TEST_MODE=1 → SQLite with StaticPool
`app/db/connection.py` switches to SQLite in-memory when `TEST_MODE=1`.
`StaticPool` ensures all connections share the same in-memory database — without
it, every new connection gets a blank DB and the tables created at startup disappear.

### MockLLMClient keyword routing
`MockLLMClient.complete()` inspects the prompt for unique phrases to return the
right canned response without any external calls:
- `"clinical data extraction system"` → EXTRACT response (JSON fields)
- `"checks to perform"` → SAFETY response (approved + quality score)
- anything else → DRAFT response (free-text report)

If you change a prompt, update the keyword too — otherwise mock tests silently
return the wrong canned response.

### Quality scoring: two-layer approach
`stage_safety` runs `run_rule_checks()` (fast regex, zero LLM cost) first, then
passes violations to `prompt_safety` as structured context. Claude scores four
dimensions: completeness, consistency, clinical_accuracy, format_compliance.
Rule checks: `app/services/quality.py`. Prompt: `app/services/prompts.py`.

### Structured JSON stored as Text
`ReportDraft.structured_json` and `quality_breakdown` are stored as plain `Text`
(JSON strings) for MS SQL compatibility. Both are parsed with `json.loads()` in
`ReportDraftRead.from_orm_with_parsed_json()` before being returned by the API.

### State machine on ReportDraft.status
Valid transitions: `draft` → `approved` or `draft` → `rejected`. Already-actioned
drafts return 409 Conflict. Enforced at two levels: application (FastAPI returns 409)
and database (CHECK constraint prevents invalid values via direct DB access).

---

## File Map

```
main.py                         FastAPI entry point, lifespan creates DB tables
app/
  core/
    config.py                   Pydantic Settings — all env vars live here
    logging.py                  structlog; console in dev, JSON in prod
  db/
    models.py                   5 ORM models: Patient, Study, ReportInput, ReportDraft, AgentEvent
    connection.py               Engine factory; TEST_MODE switches to SQLite + StaticPool
  agents/
    pipeline.py                 PipelineContext dataclass + 6 stage functions + run_pipeline()
  api/
    studies.py                  10 FastAPI endpoints
    schemas.py                  All Pydantic request/response schemas (includes Modality/DraftStatus enums)
  services/
    llm.py                      LLMClient ABC + MockLLMClient + AnthropicLLMClient
    transcription.py            TranscriptionClient ABC + Mock + OpenAITranscriptionClient
    prompts.py                  3 clinical prompt functions (extract, draft, safety)
    quality.py                  run_rule_checks() — fast pre-flight report quality checks
static/
  index.html                    Single-page UI (Alpine.js + Tailwind CSS, no build step)
tests/
  conftest.py                   Fixtures: SQLite DB session, TestClient with dependency override
  test_api.py                   HTTP endpoint tests (13 tests)
  test_pipeline.py              Pipeline stage unit tests (15 tests)
  test_quality.py               Rule-based quality check unit tests (7 tests)
  test_signoff.py               Approve/reject state machine tests (8 tests)
  test_transcription.py         Transcription service tests (6 tests)
```

---

## Planned Features (do not implement without being asked)

1. **DICOM Image Analysis** — `app/services/dicom.py` (`DICOMProcessor` class),
   `vision_complete()` method on `LLMClient` ABC, new `ANALYZE_IMAGE` pipeline stage,
   `POST /studies/{id}/dicom` endpoint. Dependencies: `pydicom`, `Pillow`. No PyTorch.

2. **Authentication** — `actioned_by` is currently a free-text string. Eventual plan
   is to replace it with a real user reference once an auth layer is added.

---

## Constraints to Always Remember

- **Python 3.9** — no `str | None` union syntax, no `list[dict]` as type hints in
  dataclasses. Use `Optional[str]` from `typing` and `List[dict]` or untyped.
- **No PyTorch / heavy ML libraries** — use Claude's vision API for image analysis,
  not computer vision models. The repo stays pip-installable on any laptop.
- **Synchronous pipeline** — `run_pipeline()` blocks the HTTP thread. Acceptable for
  a demo; a production version would use background tasks or a task queue (Celery/Redis).
- **SQLite in tests** — `CHECK` constraints work; `server_default` values are not
  applied until the next DB read. Re-fetch objects after insert if you need defaults.
- **MockLLMClient is the test contract** — if you change what the real API returns,
  update the mock to match. Tests that pass against a wrong mock give false confidence.
