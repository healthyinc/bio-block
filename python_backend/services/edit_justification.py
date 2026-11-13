"""Why each text-changing edit was made, and whether it may release automatically.

Every edit the anonymizer applies carries a typed justification. A file is
released automatically only when every applied edit is justified by
deterministic identifier evidence, by explicit contextual evidence (a title,
"seen by", an organisation designator, an address), by corroboration that is
more than detectors agreeing (a person-name shape, a relationship word, a
proper noun away from a sentence opening), or by a documented format
transformation. An edit resting on a model's label alone is still applied -
privacy first - and the file is held with the value-free reason
``ambiguous_model_edit`` for a person to decide.

This is what protects clinical content no vocabulary lists: an unknown
analyte, sequence or stain a model mislabels is either kept by the shared
clinical-preservation policy's structural rules, or its removal holds the
file. It is never silently removed from a released file.

Nothing here returns a value. Records are counted by class and reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Tuple

POLICY_VERSION = "edit-justification-v1"
#: The review reason for an applied edit nothing but a model's label supports.
AMBIGUOUS_MODEL_EDIT = "ambiguous_model_edit"
#: A short number no label or clinical structure explains: replaced in the
#: review candidate, and the file is held.
AMBIGUOUS_SHORT_NUMBER = "ambiguous_short_number"
# Compatibility name for callers; new decisions emit the policy term above.
AMBIGUOUS_NUMERIC_VALUE = AMBIGUOUS_SHORT_NUMBER


class EditClass(str, Enum):
    DETERMINISTIC_PHI = "deterministic_phi"
    CONTEXTUAL_PHI = "contextual_phi"
    CORROBORATED_PHI = "corroborated_phi"
    PROTECTED_CLINICAL = "protected_clinical"
    SANITIZER_TOKEN = "sanitizer_token"
    REMOVED_SURFACE = "removed_surface"
    AMBIGUOUS_MODEL_EDIT = "ambiguous_model_edit"
    UNSUPPORTED_OR_UNVERIFIABLE = "unsupported_or_unverifiable"


#: Rationales that are explicit context rather than corroboration.
_CONTEXTUAL = frozenset(
    {
        "person_context_present",
        "facility_context_present",
        "organisation_designator_present",
        "address_structure",
        "relationship_context",
        "dicom_person_name_structure",
        "unique_identifier_syntax_under_structured_claim",
        "category_not_confusable_with_clinical_text",
        "announced_identifier_value",
    }
)
_CORROBORATED = frozenset(
    {
        "compound_surname_shape",
        "multi_token_name_shape",
        "corroborated_proper_noun",
    }
)


def classify(
    evidence: str, action: str, rationale: str, justified: bool = True
) -> EditClass:
    """The class of one assessed finding, from its evidence, action and rationale."""
    if action == "preserve":
        return EditClass.PROTECTED_CLINICAL
    if action == "review":
        return EditClass.UNSUPPORTED_OR_UNVERIFIABLE
    if not justified:
        return EditClass.AMBIGUOUS_MODEL_EDIT
    if evidence == "deterministic" or rationale == "exact_structured_match":
        return EditClass.DETERMINISTIC_PHI
    if rationale in _CORROBORATED:
        return EditClass.CORROBORATED_PHI
    return EditClass.CONTEXTUAL_PHI


@dataclass(frozen=True)
class EditRecord:
    """One applied or declined edit. Offsets and the token never leave the process."""

    source_start: int
    source_end: int
    replacement: str
    surface: str
    detectors: Tuple[str, ...]
    categories: Tuple[str, ...]
    deterministic: bool
    structural_context: str
    clinical_evidence: str
    token_evidence: str
    decision: EditClass
    reason: str
    policy_version: str = POLICY_VERSION

    def public_dict(self) -> Dict[str, object]:
        return {
            "surface": self.surface,
            "detectors": list(self.detectors),
            "categories": list(self.categories),
            "deterministic": self.deterministic,
            "structural_context": self.structural_context,
            "clinical_evidence": self.clinical_evidence,
            "token_evidence": self.token_evidence,
            "decision": self.decision.value,
            "reason": self.reason,
            "policy_version": self.policy_version,
        }


def summary(records: Iterable[EditRecord]) -> Dict[str, object]:
    """Counts by class and reason. Value-free."""
    by_class: Dict[str, int] = {}
    by_reason: Dict[str, int] = {}
    for record in records:
        by_class[record.decision.value] = by_class.get(record.decision.value, 0) + 1
        by_reason[record.reason] = by_reason.get(record.reason, 0) + 1
    return {
        "policy_version": POLICY_VERSION,
        "by_class": by_class,
        "by_reason": by_reason,
    }


def releasable(records: Iterable[EditRecord]) -> Tuple[bool, Optional[str]]:
    """Whether a set of applied edits allows automatic release, and why not."""
    for record in records:
        if record.decision is EditClass.AMBIGUOUS_MODEL_EDIT:
            return False, AMBIGUOUS_MODEL_EDIT
        if record.decision is EditClass.UNSUPPORTED_OR_UNVERIFIABLE:
            return False, "unverifiable_edit"
    return True, None
