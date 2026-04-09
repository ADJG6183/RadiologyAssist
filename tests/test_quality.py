"""
Unit tests for the rule-based quality checker.

These tests have ZERO external dependencies — no DB, no LLM, no HTTP.
They run in pure Python against hand-crafted inputs so every edge case
can be provoked deterministically.

Why test this separately from the pipeline?
  Because run_rule_checks() is pure logic with no side effects.
  A unit test here runs in ~1ms and gives you a precise failure message:
  "laterality_consistent failed because 'right' appeared without 'left'".
  An integration test through the full pipeline would take seconds and
  the failure message would be much less useful.
"""

import pytest
from app.services.quality import run_rule_checks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_draft():
    """Minimal report that should pass all rule checks (including word count ≥ 50)."""
    return (
        "TECHNIQUE: CT of the chest without contrast was performed.\n"
        "CLINICAL HISTORY: Patient presents with cough and fever for three days.\n"
        "FINDINGS: Lungs are clear bilaterally with no focal consolidation. "
        "No pleural effusion or pneumothorax identified. "
        "Heart size is within normal limits. "
        "The mediastinum and hila are unremarkable. "
        "Visualised bony structures are intact with no acute fracture.\n"
        "IMPRESSION: No acute cardiopulmonary process identified. "
        "Unremarkable CT examination of the chest without contrast."
    )


def _valid_fields():
    """Minimal structured fields that should pass all rule checks."""
    return {
        "modality": "CT",
        "laterality": "bilateral",
        "impression": "Unremarkable CT chest.",
        "critical_findings": [],
    }


# ---------------------------------------------------------------------------
# Check 1: section headers
# ---------------------------------------------------------------------------

def test_rule_checks_passes_on_valid_draft():
    """A well-formed report with all headers and fields should pass everything."""
    result = run_rule_checks(_valid_draft(), _valid_fields())
    assert result["all_passed"] is True
    assert result["rule_violations"] == []
    assert result["section_headers_present"] is True
    assert result["required_fields_populated"] is True
    assert result["laterality_consistent"] is True
    assert result["critical_findings_in_impression"] is True
    assert result["word_count_adequate"] is True


def test_rule_checks_detects_missing_impression_header():
    """
    If the IMPRESSION: header is absent, section_headers_present must fail.
    This is the most common format violation — Claude sometimes writes
    'Impression:' (lowercase) which breaks the section regex.
    """
    draft = _valid_draft().replace("IMPRESSION:", "Impression:")
    result = run_rule_checks(draft, _valid_fields())
    assert result["section_headers_present"] is False
    assert any("section_headers_present" in v for v in result["rule_violations"])
    assert result["all_passed"] is False


def test_rule_checks_detects_missing_findings_header():
    draft = _valid_draft().replace("FINDINGS:", "Findings:")
    result = run_rule_checks(draft, _valid_fields())
    assert result["section_headers_present"] is False


# ---------------------------------------------------------------------------
# Check 2: required fields populated
# ---------------------------------------------------------------------------

def test_rule_checks_detects_empty_modality():
    """
    If the EXTRACT stage failed to identify the modality, the report is
    missing a required field and the check must flag it.
    """
    fields = _valid_fields()
    fields["modality"] = ""
    result = run_rule_checks(_valid_draft(), fields)
    assert result["required_fields_populated"] is False
    assert any("required_fields_populated" in v for v in result["rule_violations"])


# ---------------------------------------------------------------------------
# Check 3: laterality consistency
# ---------------------------------------------------------------------------

def test_rule_checks_detects_laterality_conflict():
    """
    Extracted laterality is 'left' but the draft exclusively says 'right'
    with no mention of 'left' at all.
    This is a serious clinical error — the report describes the wrong side.

    Note: the heuristic flags conflict when the OPPOSITE side appears
    without the CORRECT side appearing at all. A draft saying 'right'
    AND 'left' would NOT trigger this (bilateral mention is often fine).
    """
    draft = (
        "TECHNIQUE: CT chest without contrast was performed.\n"
        "FINDINGS: Right lung consolidation noted in the right lower lobe. "
        "Right pleural effusion present. Right hilar adenopathy observed. "
        "Heart size normal. Mediastinum unremarkable. Bones intact.\n"
        "IMPRESSION: Right lower lobe pneumonia with right-sided effusion. "
        "Recommend follow-up CT of the right chest in six weeks.\n"
    )
    # This draft has no mention of 'left' at all — only 'right' throughout
    fields = _valid_fields()
    fields["laterality"] = "left"   # extracted as left, but report says right everywhere

    result = run_rule_checks(draft, fields)
    assert result["laterality_consistent"] is False
    assert any("laterality_consistent" in v for v in result["rule_violations"])


def test_rule_checks_passes_bilateral_laterality():
    """
    When laterality is 'bilateral' (not left or right), the laterality
    check should be skipped — no conflict is possible.
    """
    result = run_rule_checks(_valid_draft(), _valid_fields())
    assert result["laterality_consistent"] is True


# ---------------------------------------------------------------------------
# Check 4: critical findings in impression
# ---------------------------------------------------------------------------

def test_rule_checks_detects_critical_finding_absent_from_impression():
    """
    If a critical finding like 'pneumothorax' was extracted but the word
    doesn't appear in the IMPRESSION section, the check must flag it.
    Radiologists read the impression first in emergencies — it must be there.
    """
    draft = (
        "TECHNIQUE: CT chest without contrast.\n"
        "FINDINGS: Large right-sided pneumothorax identified. "
        "Lungs otherwise clear. Heart normal.\n"
        "IMPRESSION: Normal chest. No acute findings.\n"  # ← missing pneumothorax!
    )
    draft += "Additional detail. " * 10  # pad word count
    fields = _valid_fields()
    fields["critical_findings"] = ["large pneumothorax"]

    result = run_rule_checks(draft, fields)
    assert result["critical_findings_in_impression"] is False
    assert any("critical_findings_in_impression" in v for v in result["rule_violations"])


# ---------------------------------------------------------------------------
# Check 5: word count
# ---------------------------------------------------------------------------

def test_rule_checks_detects_short_report():
    """
    A 10-word report is clearly a placeholder or a truncated response.
    The check should flag it immediately without needing an LLM.
    """
    draft = "TECHNIQUE: CT. FINDINGS: Normal. IMPRESSION: Normal."
    result = run_rule_checks(draft, _valid_fields())
    assert result["word_count_adequate"] is False
    assert any("word_count_adequate" in v for v in result["rule_violations"])
