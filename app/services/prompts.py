"""
Clinical prompts for the three LLM stages of the pipeline.

Why a separate file?
  Prompts are the most frequently iterated part of an AI system.
  Keeping them here means you can improve clinical quality without
  touching any pipeline logic. Each prompt is a function so it can
  receive the specific context it needs.

Prompt design principles used here:
  1. Role first   — tell Claude exactly who it is before the task
  2. Schema       — for JSON stages, show the exact shape expected
  3. Rules        — explicit clinical constraints Claude must follow
  4. Output only  — tell Claude not to explain itself, just return the output
"""


def prompt_extract(transcript: str) -> str:
    """
    Stage 3 — EXTRACT
    Ask Claude to pull structured fields out of the raw dictation.
    We specify the exact JSON schema so the output is always consistent.
    """
    return f"""You are a clinical data extraction system for a radiology department.

Extract structured fields from the radiology dictation below.
Return ONLY valid JSON — no explanation, no markdown, no code fences.

Required JSON schema:
{{
  "modality": string,           // e.g. "CT", "MRI", "X-Ray", "Ultrasound", "PET"
  "body_part": string,          // e.g. "Chest", "Abdomen", "Brain", "Spine"
  "laterality": string,         // "left", "right", "bilateral", or "N/A"
  "technique": string,          // e.g. "without contrast", "with and without contrast"
  "clinical_history": string,   // reason for the exam if mentioned, else ""
  "findings": [string],         // list of individual finding statements
  "impression": string,         // overall conclusion
  "critical_findings": [string],// any urgent findings (pneumothorax, PE, mass), else []
  "recommendations": [string]   // follow-up suggestions, else []
}}

Rules:
- If a field is not mentioned in the dictation, use an empty string "" or empty list [].
- Do not invent findings that are not in the dictation.
- Separate each distinct finding into its own list item.
- critical_findings must only contain findings that are clinically urgent.

Dictation:
{transcript}"""


def prompt_draft(
    transcript: str,
    structured_fields: dict,
    prior_reports: list,
) -> str:
    """
    Stage 4 — DRAFT
    Ask Claude to write a formal radiology report.
    We provide the exact section structure expected.
    """
    import json

    prior_section = ""
    if prior_reports:
        prior_section = "\n\nPRIOR REPORTS FOR CONTEXT (do not copy, use only for comparison):\n"
        for i, r in enumerate(prior_reports, 1):
            prior_section += f"Prior {i}: {r['text']}\n"

    return f"""You are an attending radiologist writing a formal diagnostic radiology report.

Using the dictation and extracted fields below, write a complete radiology report.

STRICT FORMAT — use exactly these section headers in this order:
TECHNIQUE:
CLINICAL HISTORY:
FINDINGS:
IMPRESSION:
RECOMMENDATIONS: (omit this section entirely if there are none)

Rules:
- Write in formal medical prose, third person.
- Each finding should be a complete sentence.
- The IMPRESSION must summarize the most clinically significant findings.
- If a critical finding exists, it must appear first in the IMPRESSION.
- Do not add any text before TECHNIQUE: or after the last section.
- Do not include phrases like "I reviewed" or "the patient was seen".

DICTATION:
{transcript}

EXTRACTED FIELDS:
{json.dumps(structured_fields, indent=2)}
{prior_section}"""


def prompt_safety(draft_text: str, structured_fields: dict) -> str:
    """
    Stage 5 — SAFETY
    Ask Claude to review the draft for clinical and factual errors.
    We give it specific rules to check so it doesn't just rubber-stamp the report.
    """
    import json

    return f"""You are a senior radiologist performing a quality review of a radiology report draft.

Review the draft below against the extracted fields and check for the following issues:

CHECKS TO PERFORM:
1. Laterality conflict — does the report mention left/right inconsistently with the extracted fields?
2. Modality mismatch — does the report describe a technique inconsistent with the modality?
3. Missing critical findings — are any critical findings from the extracted fields absent from the impression?
4. Invented findings — does the report mention findings not present in the extracted fields?
5. Incomplete impression — does the impression fail to address the most significant findings?
6. Format violations — are any required sections (TECHNIQUE, FINDINGS, IMPRESSION) missing?

Return ONLY valid JSON — no explanation, no markdown, no code fences.

Required JSON schema:
{{
  "approved": boolean,       // true only if no issues found
  "issues": [string],        // list of specific issue descriptions, empty if approved
  "confidence": float        // your confidence in this review, 0.0 to 1.0
}}

EXTRACTED FIELDS:
{json.dumps(structured_fields, indent=2)}

DRAFT REPORT:
{draft_text}"""
