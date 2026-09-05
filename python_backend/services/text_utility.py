"""Measure how much clinical meaning survived text de-identification.

Privacy validation asks whether anything identifying is left. This asks the
other half of the question, and the two are kept separate on purpose: a
document that fails privacy is blocked outright and never reaches here, and a
document that passes privacy but fails utility is sent to review rather than
reported as a successful research artifact.

Measured against the original, not against an abstract ideal:

* **clinical_term_preservation** - of the clinical vocabulary present in the
  input (diagnoses, medications, labs, units, anatomy, modalities), what share
  is still present in the output? This is the number Phase 9 was failing.
* **content_token_preservation** - of the ordinary non-PHI word tokens in the
  input, what share survives? Catches wholesale destruction that a vocabulary
  measure alone would miss.
* **numeric_preservation** - clinical numbers (doses, lab values, vitals) that
  were not identifiers must survive; losing them silently destroys the
  measurements the research depends on.

Nothing here returns a matched value. Counts and ratios only.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Set

from services.clinical_vocabulary import (
    ANATOMY_TERMS,
    CLINICAL_EPONYMS,
    CLINICAL_MEASUREMENT_TERMS,
    MEDICATION_TERMS,
    MODALITY_TERMS,
)

#: Every term whose survival is evidence that clinical meaning was preserved.
CLINICAL_UTILITY_TERMS: frozenset[str] = frozenset(
    MODALITY_TERMS
    | CLINICAL_MEASUREMENT_TERMS
    | ANATOMY_TERMS
    | MEDICATION_TERMS
    | CLINICAL_EPONYMS
)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

#: Our own replacement tokens. Present in the output but not in the input, so
#: they must not be counted as either preserved or lost content.
_OUTPUT_TOKENS = re.compile(
    r"<REDACTED_[A-Z0-9_]+>|"
    r"\b(?:PATIENT|PROVIDER|FACILITY|ORG|PLACE|ADDRESS|RECORD|PATIENTID|PLAN|"
    r"ACCESSION|DEVICE|IDENTIFIER|USER)_\d{3,}|"
    r"90\+"
)


def _terms(text: str) -> List[str]:
    return [match.group(0).casefold() for match in _WORD.finditer(text or "")]


def _clinical_terms(text: str) -> List[str]:
    return [term for term in _terms(text) if term.strip("'-") in CLINICAL_UTILITY_TERMS]


def _numbers(text: str) -> List[str]:
    return [match.group(0) for match in _NUMBER.finditer(text or "")]


def measure_text_utility(
    original: str,
    anonymized: str,
    redacted_values: Sequence[str] = (),
) -> Dict[str, float]:
    """Compare an anonymized document against its input.

    ``redacted_values`` are the original spans that were deliberately removed.
    Their tokens are excluded from the content measure, because removing PHI
    is the goal rather than a utility loss.
    """
    removed_tokens: Set[str] = set()
    removed_numbers: Set[str] = set()
    for value in redacted_values:
        removed_tokens.update(_terms(value))
        removed_numbers.update(_numbers(value))

    cleaned_output = _OUTPUT_TOKENS.sub(" ", anonymized or "")

    original_clinical = _clinical_terms(original)
    output_clinical = _clinical_terms(cleaned_output)
    clinical_preserved = _multiset_overlap(original_clinical, output_clinical)

    original_content = [t for t in _terms(original) if t not in removed_tokens]
    output_content = _terms(cleaned_output)
    content_preserved = _multiset_overlap(original_content, output_content)

    original_numbers = [n for n in _numbers(original) if n not in removed_numbers]
    output_numbers = _numbers(cleaned_output)
    numbers_preserved = _multiset_overlap(original_numbers, output_numbers)

    return {
        "clinical_terms_in_input": len(original_clinical),
        "clinical_terms_preserved": clinical_preserved,
        "clinical_term_preservation": _ratio(clinical_preserved, len(original_clinical)),
        "content_tokens_in_input": len(original_content),
        "content_tokens_preserved": content_preserved,
        "content_token_preservation": _ratio(content_preserved, len(original_content)),
        "clinical_numbers_in_input": len(original_numbers),
        "clinical_numbers_preserved": numbers_preserved,
        "numeric_preservation": _ratio(numbers_preserved, len(original_numbers)),
    }


def _multiset_overlap(left: Iterable[str], right: Iterable[str]) -> int:
    """How many of ``left`` are still present in ``right``, counting repeats."""
    from collections import Counter

    remaining = Counter(right)
    kept = 0
    for item in left:
        if remaining.get(item, 0) > 0:
            remaining[item] -= 1
            kept += 1
    return kept


def _ratio(numerator: int, denominator: int) -> float:
    # An input with no clinical content cannot lose any, so the vacuous case is
    # a pass rather than a division error.
    if denominator <= 0:
        return 1.0
    return round(numerator / denominator, 4)
