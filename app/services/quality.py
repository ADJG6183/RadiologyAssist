"""
Fast, deterministic pre-flight quality checks for report drafts.

Why have this at all when Claude already reviews the report in stage_safety?
  Because LLM calls cost tokens (money + latency) and some failures are
  purely mechanical. A missing "IMPRESSION:" header requires no AI to spot —
  regex finds it in microseconds. We run these checks first and pass the
  results to Claude as structured context, so Claude focuses on *clinical*
  reasoning rather than re-discovering things a grep would catch in 0.001s.

  Think of it like a pilot's pre-flight checklist before a complex flight:
  you don't need a senior engineer to verify the fuel cap is on — you check
  that mechanically so the engineer can focus on the hard questions.
"""

import re
from typing import Optional

# Section headers the pipeline prompts Claude to include (from prompt_draft)
_REQUIRED_HEADERS = ["TECHNIQUE:", "FINDINGS:", "IMPRESSION:"]


def run_rule_checks(draft_text: str, structured_fields: dict) -> dict:
    """
    Run fast, zero-dependency rule checks against a report draft.

    Returns a dict with:
      - one bool per check name (True = passed)
      - "rule_violations": list of human-readable strings for each failure
      - "all_passed": True only if every check passed

    All checks use stdlib only (re, str). No LLM. No external libraries.
    Typical runtime: < 1ms.
    """
    violations = []

    # ------------------------------------------------------------------
    # Check 1: Required section headers present
    # re.MULTILINE makes ^ match at the start of each line, not just the
    # start of the whole string.  Without it, "^TECHNIQUE:" would only
    # match if TECHNIQUE was the very first word in the document.
    # ------------------------------------------------------------------
    headers_ok = all(
        re.search(rf"^{re.escape(h)}", draft_text, re.MULTILINE)
        for h in _REQUIRED_HEADERS
    )
    if not headers_ok:
        missing = [
            h for h in _REQUIRED_HEADERS
            if not re.search(rf"^{re.escape(h)}", draft_text, re.MULTILINE)
        ]
        violations.append(f"section_headers_present: missing {missing}")

    # ------------------------------------------------------------------
    # Check 2: Required structured fields are populated
    # modality and impression must be non-empty strings — these are the
    # two fields the DRAFT prompt always uses.
    # ------------------------------------------------------------------
    fields_ok = bool(
        structured_fields.get("modality")
        and structured_fields.get("impression")
    )
    if not fields_ok:
        empty = [f for f in ("modality", "impression") if not structured_fields.get(f)]
        violations.append(f"required_fields_populated: empty fields {empty}")

    # ------------------------------------------------------------------
    # Check 3: Laterality consistency
    # If the extraction says "left", the word "right" should not appear
    # in the draft without "left" also appearing nearby.
    # This is a *heuristic*, not a perfect check — "the right ventricle"
    # is clinically normal even in a "left" laterality scan.  But if only
    # "right" appears with no "left" counterpart, it's very likely wrong.
    # ------------------------------------------------------------------
    laterality = str(structured_fields.get("laterality", "")).lower()
    laterality_ok = True
    if laterality in ("left", "right"):
        opposite = "right" if laterality == "left" else "left"
        correct  = laterality
        text_lower = draft_text.lower()
        # Flag if the OPPOSITE side word appears but the CORRECT one does not
        opposite_present = bool(re.search(rf"\b{opposite}\b", text_lower))
        correct_present  = bool(re.search(rf"\b{correct}\b",  text_lower))
        if opposite_present and not correct_present:
            laterality_ok = False
            violations.append(
                f"laterality_consistent: extracted='{laterality}' but draft "
                f"mentions '{opposite}' without '{correct}'"
            )

    # ------------------------------------------------------------------
    # Check 4: Critical findings appear in the IMPRESSION section
    # If the EXTRACT stage flagged urgent findings, they MUST appear in
    # the impression — that's the section radiologists and clinicians read
    # first when there's an emergency.
    # ------------------------------------------------------------------
    critical = structured_fields.get("critical_findings", [])
    critical_ok = True
    if critical:
        # Extract text after the IMPRESSION: header
        impression_match = re.search(
            r"^IMPRESSION:(.*?)(?=^[A-Z]+:|$)",
            draft_text,
            re.MULTILINE | re.DOTALL,
        )
        if impression_match:
            impression_text = impression_match.group(1).lower()
            # Check that at least one word from the first critical finding
            # appears in the impression text
            first_finding_tokens = set(critical[0].lower().split())
            significant_tokens = {t for t in first_finding_tokens if len(t) > 3}
            if not any(t in impression_text for t in significant_tokens):
                critical_ok = False
                violations.append(
                    f"critical_findings_in_impression: '{critical[0]}' "
                    "not reflected in IMPRESSION section"
                )
        else:
            # No IMPRESSION section found at all
            critical_ok = False
            violations.append(
                "critical_findings_in_impression: IMPRESSION section not found"
            )

    # ------------------------------------------------------------------
    # Check 5: Minimum word count
    # A real radiology report has at minimum a TECHNIQUE, FINDINGS, and
    # IMPRESSION section with at least a sentence each.  Fewer than 50
    # words almost certainly means truncation or a blank response.
    # ------------------------------------------------------------------
    word_count = len(draft_text.split())
    length_ok = word_count >= 50
    if not length_ok:
        violations.append(
            f"word_count_adequate: {word_count} words (minimum 50)"
        )

    return {
        "section_headers_present":        headers_ok,
        "required_fields_populated":      fields_ok,
        "laterality_consistent":          laterality_ok,
        "critical_findings_in_impression": critical_ok,
        "word_count_adequate":            length_ok,
        "rule_violations":                violations,
        "all_passed":                     len(violations) == 0,
    }
