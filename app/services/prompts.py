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

from typing import Optional


def prompt_extract(transcript: str, image_findings: Optional[str] = None) -> str:
    """
    Stage 3 — EXTRACT
    Ask Claude to pull structured fields out of the raw dictation.
    We specify the exact JSON schema so the output is always consistent.

    The optional image_findings parameter carries the output of the ANALYZE_IMAGE
    stage.  When present, it is included as additional visual context so the
    extraction can reconcile dictation with what was actually seen on the images.
    """
    # When no dictation was provided the radiologist is relying entirely on
    # the DICOM image analysis (image_findings).  Tell Claude explicitly so
    # it knows to derive all fields from the image section rather than the
    # empty dictation block.
    if transcript.strip():
        dictation_block = f"Dictation:\n{transcript}"
        image_label = "IMAGE ANALYSIS FINDINGS (from DICOM review — use to supplement, not replace, dictation):"
    else:
        dictation_block = "Dictation:\n(No dictation provided — derive all fields from the image analysis findings below)"
        image_label = "IMAGE ANALYSIS FINDINGS (primary source — no dictation was recorded):"

    image_section = ""
    if image_findings:
        image_section = f"\n\n{image_label}\n{image_findings}\n"

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
- If a field is not mentioned in the source material, use an empty string "" or empty list [].
- Do not invent findings not present in the source material.
- Separate each distinct finding into its own list item.
- critical_findings must only contain findings that are clinically urgent.
{image_section}
{dictation_block}"""


def prompt_analyze_image(metadata: dict, num_slices: int) -> str:
    """
    Stage 3a — ANALYZE_IMAGE
    Ask Claude to describe what it sees in the provided DICOM PNG images.

    Called with the windowed PNG bytes via llm.vision_complete().
    The metadata dict (from DICOMProcessor.extract_metadata()) gives Claude
    clinical context: which body part, which modality, slice thickness, etc.

    Returns JSON with visual_findings, visual_impression, image_quality,
    and windows_reviewed so the downstream EXTRACT stage can use it.
    """
    import json

    return f"""You are an expert radiologist reviewing DICOM images from a {metadata.get('modality', 'CT')} examination.

The images shown are windowed PNG representations of {num_slices} representative slice(s).
Each slice is shown twice: once with lung windowing (centre=-600 HU, width=1500) and once
with mediastinal windowing (centre=+40 HU, width=400).

Scan metadata:
{json.dumps(metadata, indent=2)}

Review all provided images carefully and return a structured visual analysis.
Return ONLY valid JSON — no explanation, no markdown, no code fences.

Required JSON schema:
{{
  "visual_findings": [string],    // list of specific visual observations per structure
  "visual_impression": string,    // single-sentence overall impression from images alone
  "image_quality": string,        // "Diagnostic", "Suboptimal", or "Non-diagnostic"
  "windows_reviewed": [string]    // windows used: ["lung", "mediastinal"]
}}

Rules:
- Describe only what you can see — do not invent findings.
- Separate findings by anatomical structure (lungs, heart, mediastinum, etc.).
- If image quality prevents confident assessment, state that in image_quality."""


def prompt_analyze_metadata(metadata: dict) -> str:
    """
    Fallback for ANALYZE_IMAGE when pixel data is unavailable.

    When a DICOM file loads but has no decodable pixel data (SR, RT, PR files,
    or corrupt CT exports) we still have the header tags — modality, body part,
    scan parameters. This prompt asks Claude to infer clinical context from
    those tags alone, using complete() instead of vision_complete().
    """
    import json

    return f"""You are an expert radiologist reviewing DICOM file metadata from a {metadata.get('modality', 'CT')} examination.

No image pixels are available, but the DICOM header tags are provided below.
Based on the scan parameters, infer what you can about the study type and technique.

DICOM metadata:
{json.dumps(metadata, indent=2)}

Return ONLY valid JSON — no explanation, no markdown, no code fences.

Required JSON schema:
{{
  "visual_findings": [],
  "visual_impression": string,  // inferred from metadata only — state this clearly
  "image_quality": "Non-diagnostic",
  "windows_reviewed": []
}}

Rules:
- You have no images — do not invent findings. visual_findings must be empty.
- visual_impression should summarise the scan type/technique from metadata only.
- Always prefix visual_impression with "Metadata only (no pixel data available): "."""


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
        prior_section = "\n\nPRIOR APPROVED REPORTS FOR CONTEXT (do not copy, use only for comparison):\n"
        for i, r in enumerate(prior_reports, 1):
            if "impression" in r:
                # Structured form — most useful to Claude
                findings_str = "; ".join(r["findings"]) if r["findings"] else "none recorded"
                prior_section += (
                    f"Prior {i} — Impression: {r['impression']} | "
                    f"Key findings: {findings_str}\n"
                )
            else:
                prior_section += f"Prior {i}: {r['text']}\n"

    dictation_section = transcript if transcript.strip() else "(No dictation — report derived from image analysis and extracted fields)"

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
{dictation_section}

EXTRACTED FIELDS:
{json.dumps(structured_fields, indent=2)}
{prior_section}"""


def prompt_safety(
    draft_text: str,
    structured_fields: dict,
    rule_check_results: Optional[dict] = None,
) -> str:
    """
    Stage 5 — SAFETY
    Ask Claude to review the draft for clinical and factual errors.
    We give it specific rules to check so it doesn't just rubber-stamp the report.

    The optional rule_check_results parameter carries the output of the fast
    deterministic pre-checks from app/services/quality.py.  When present, we
    include those findings as a structured context block so Claude can focus
    on clinical reasoning rather than re-checking things a regex already caught.

    The JSON schema now includes quality_score and dimensions so that every
    safety review simultaneously produces a quality rating — one LLM call,
    two outputs.
    """
    import json

    # Build the automated pre-checks section if rule results were passed in.
    # This is like handing a checklist to the reviewer before they start reading —
    # they know upfront which mechanical issues were already flagged.
    pre_checks_section = ""
    if rule_check_results:
        violations = rule_check_results.get("rule_violations", [])
        if violations:
            pre_checks_section = (
                "\n\nAUTOMATED PRE-CHECKS (already identified by rule engine — "
                "factor these into your scores):\n"
                + "\n".join(f"  - {v}" for v in violations)
            )
        else:
            pre_checks_section = (
                "\n\nAUTOMATED PRE-CHECKS: All mechanical checks passed "
                "(headers present, laterality consistent, word count adequate)."
            )

    return f"""You are a senior radiologist performing a quality review of a radiology report draft.

Review the draft below against the extracted fields and check for the following issues:

CHECKS TO PERFORM:
1. Laterality conflict — does the report mention left/right inconsistently with the extracted fields?
2. Modality mismatch — does the report describe a technique inconsistent with the modality?
3. Missing critical findings — are any critical findings from the extracted fields absent from the impression?
4. Invented findings — does the report mention findings not present in the extracted fields?
5. Incomplete impression — does the impression fail to address the most significant findings?
6. Format violations — are any required sections (TECHNIQUE, FINDINGS, IMPRESSION) missing?{pre_checks_section}

Return ONLY valid JSON — no explanation, no markdown, no code fences.

Required JSON schema:
{{
  "approved": boolean,       // true only if no significant issues found
  "issues": [string],        // list of specific issue descriptions, empty if approved
  "confidence": float,       // your confidence in this review, 0.0 to 1.0
  "quality_score": float,    // overall quality 0.0–1.0, mean of the four dimensions below
  "dimensions": {{
    "completeness": float,      // 0–1: all required clinical info present
    "consistency": float,       // 0–1: no contradictions between fields and draft
    "clinical_accuracy": float, // 0–1: findings are clinically coherent and plausible
    "format_compliance": float  // 0–1: all required sections present and correctly labeled
  }}
}}

EXTRACTED FIELDS:
{json.dumps(structured_fields, indent=2)}

DRAFT REPORT:
{draft_text}"""
