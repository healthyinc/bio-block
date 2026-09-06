"""Second-pass validator: regressions for every pattern that used to block.

Before Phase 11 the validator blanked each generated token with spaces and
re-scanned the result. That rewrote the sentence it was checking - "Dr.
PROVIDER_001 at FACILITY_001" became "Dr.<12 spaces>at<13 spaces>" - and the
detectors reliably predicted that a name belonged in the hole. Sixty per cent
of documents with nothing surviving were held for manual review on the
strength of artefacts the masking itself had created.

Each test below is one of those shapes. They assert the two halves of the
correction together: the artefact no longer blocks, *and* the thing the
artefact was hiding - a real identifier next to a surrogate, a deterministic
match, a mangled replacement - still does.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.text_anonymization import (  # noqa: E402
    CLASSIFICATION_EXACT_SURROGATE,
    CLASSIFICATION_GENUINE_PHI,
    CLASSIFICATION_MALFORMED,
    CLASSIFICATION_PLACEHOLDER,
    RESIDUAL_CLASSIFICATIONS,
    anonymize_clinical_text,
    residual_findings,
    residual_phi_categories,
)
from services.transformation_provenance import (  # noqa: E402
    KIND_SURROGATE,
    ReplacementRegion,
    TransformationProvenance,
)

FINDING_FIELDS = {
    "detector",
    "category",
    "evidence_type",
    "location_type",
    "overlaps_generated_region",
    "classification",
    "blocking",
}


def _anonymized(text):
    return anonymize_clinical_text(text, profile="strict", study_salt="study-a")


# -- the shapes that used to block -----------------------------------------


PREVIOUSLY_BLOCKING = [
    pytest.param(
        "Care was provided by Dr. Priyanka Venkataraman at "
        "Saint Corwin Memorial Institute.",
        id="provider_and_facility_surrogates_adjacent",
    ),
    pytest.param(
        "Patient Rukmini Balasubramanian was accompanied by "
        "Deepa Krishnamurthy, the patient's mother.",
        id="two_person_surrogates_in_one_sentence",
    ),
    pytest.param(
        "Rukmini Balasubramanian returned for review.",
        id="surrogate_opening_the_sentence",
    ),
    pytest.param(
        "Referred by Dr. Anand Rajagopalan to Dr. Meera Subramanian.",
        id="consecutive_provider_surrogates",
    ),
    pytest.param(
        "Rukmini Balasubramanian, Deepa Krishnamurthy and Anand Rajagopalan "
        "attended.",
        id="three_surrogates_in_a_list",
    ),
    pytest.param(
        "Metformin 500 mg nightly for Rukmini Balasubramanian; HbA1c 7.2.",
        id="surrogate_beside_preserved_clinical_content",
    ),
    pytest.param(
        "Admitted to Saint Corwin Memorial Institute on 14 March 2021.",
        id="facility_surrogate_beside_a_shifted_date",
    ),
    pytest.param(
        "The patient is 94 years old and remains independent.",
        id="age_generalised_to_the_safe_harbor_aggregate",
    ),
]


@pytest.mark.parametrize("text", PREVIOUSLY_BLOCKING)
def test_previously_blocking_patterns_no_longer_block(text):
    result = _anonymized(text)

    assert result["residual_phi_categories"] == {}


@pytest.mark.parametrize("text", PREVIOUSLY_BLOCKING)
def test_findings_on_generated_tokens_are_attributed_not_ignored(text):
    """Discounted findings are still recorded, with a reason.

    Silence would be indistinguishable from a validator that had stopped
    looking, which is the failure mode this whole phase exists to avoid.
    """
    result = _anonymized(text)
    findings = residual_findings(result["anonymized_text"])

    for finding in findings:
        assert finding["classification"] in RESIDUAL_CLASSIFICATIONS


def test_finding_records_carry_only_the_permitted_fields():
    result = _anonymized(
        "Patient Rukmini Balasubramanian was seen by Dr. Anand Rajagopalan."
    )
    findings = residual_findings(result["anonymized_text"])

    for finding in findings:
        assert set(finding) == FINDING_FIELDS


def test_finding_records_never_carry_a_value_or_offsets():
    text = "Patient Rukmini Balasubramanian has MRN 44821903."
    result = _anonymized(text)
    findings = residual_findings(result["anonymized_text"])

    serialized = repr(findings)
    assert "Rukmini" not in serialized
    assert "44821903" not in serialized
    assert "start" not in serialized
    assert "end" not in serialized


# -- what must still block --------------------------------------------------


def test_deterministic_finding_in_original_text_always_blocks():
    """A regex match on identifier syntax outranks every other consideration."""
    findings = residual_findings("Contact synthetic.person@example.org for notes.")

    blocking = [f for f in findings if f["blocking"]]
    assert blocking
    assert blocking[0]["classification"] == CLASSIFICATION_GENUINE_PHI
    assert blocking[0]["evidence_type"] == "deterministic"


def test_deterministic_finding_blocks_even_beside_a_surrogate():
    text = "PATIENT_001 can be reached on synthetic.person@example.org."
    provenance = TransformationProvenance(
        regions=(
            ReplacementRegion(0, len("PATIENT_001"), KIND_SURROGATE, "PERSON"),
        )
    )

    assert residual_phi_categories(text, provenance)


def test_partial_overlap_with_a_surrogate_does_not_excuse_a_finding():
    """"Wholly inside" is the rule; touching a surrogate is not enough.

    The uncovered half of a straddling prediction is text the sanitizer did
    not write, and a missed identifier sitting immediately beside a surrogate
    is precisely the case that must not be waved through.
    """
    text = "Reviewed by PATIENT_001 Balasubramanian on the ward."
    start = text.index("PATIENT_001")
    provenance = TransformationProvenance(
        regions=(
            ReplacementRegion(
                start, start + len("PATIENT_001"), KIND_SURROGATE, "PERSON"
            ),
        )
    )

    findings = residual_findings(text, provenance)
    straddling = [
        f
        for f in findings
        if f["location_type"] == "spans_generated_and_original"
    ]

    for finding in straddling:
        assert finding["classification"] != CLASSIFICATION_EXACT_SURROGATE
        assert finding["classification"] != CLASSIFICATION_PLACEHOLDER


def test_truncated_placeholder_is_malformed_output_and_blocks():
    findings = residual_findings("Value <REDACTED_PERSON noted on the chart.")

    assert any(f["classification"] == CLASSIFICATION_MALFORMED for f in findings)
    assert any(f["blocking"] for f in findings)


def test_dangling_surrogate_stem_is_malformed_output_and_blocks():
    findings = residual_findings("Seen by PATIENT_ at the clinic.")

    assert any(f["classification"] == CLASSIFICATION_MALFORMED for f in findings)


def test_recorded_region_that_no_longer_holds_its_token_is_malformed():
    """Provenance claiming a region the text does not contain is unsound.

    If the two disagree, every attribution built on that map is unreliable, so
    the document blocks rather than being judged against a map we know to be
    wrong.
    """
    text = "Reviewed by the attending physician."
    provenance = TransformationProvenance(
        regions=(ReplacementRegion(0, 8, KIND_SURROGATE, "PERSON"),)
    )

    findings = residual_findings(text, provenance)

    assert any(f["classification"] == CLASSIFICATION_MALFORMED for f in findings)
    assert residual_phi_categories(text, provenance)


def test_well_formed_placeholders_are_not_malformed():
    findings = residual_findings("Value <REDACTED_DATE> recorded at intake.")

    assert not [
        f for f in findings if f["classification"] == CLASSIFICATION_MALFORMED
    ]


# -- the validator is still connected --------------------------------------


def test_anonymizer_reports_its_own_residual_verdict():
    """The scan runs where the provenance map lives, not in each caller.

    Handing the map out instead would put generated-region offsets into every
    serialized response, and any caller that forgot to pass it back would
    silently re-scan surrogates as surviving text.
    """
    result = _anonymized("Patient Rukmini Balasubramanian attended clinic.")

    assert "residual_phi_categories" in result
    assert "provenance" not in result


# -- straddling predictions: the remainder is examined, not ignored ---------


def test_a_prediction_reaching_one_word_past_a_surrogate_is_not_a_block():
    """Detectors routinely propose one word more than the token they saw.

    "Dr. PROVIDER_001" is a prediction about our own surrogate plus a title.
    The title is examined rather than ignored - it reads as context, not
    identity - so the finding is an artefact of the boundary.
    """
    text = "Care was provided by Dr. PROVIDER_001 at the clinic."
    start = text.index("PROVIDER_001")
    provenance = TransformationProvenance(
        regions=(
            ReplacementRegion(
                start, start + len("PROVIDER_001"), KIND_SURROGATE, "PERSON"
            ),
        )
    )

    assert residual_phi_categories(text, provenance) == {}


def test_a_name_beside_a_surrogate_still_blocks():
    """The remainder is real text, so the finding survives examination."""
    text = "Reviewed by PATIENT_001 Balasubramanian on the ward."
    start = text.index("PATIENT_001")
    provenance = TransformationProvenance(
        regions=(
            ReplacementRegion(
                start, start + len("PATIENT_001"), KIND_SURROGATE, "PERSON"
            ),
        )
    )

    assert residual_phi_categories(text, provenance)


def test_uncovered_fragments_are_computed_exactly():
    provenance = TransformationProvenance(
        regions=(
            ReplacementRegion(4, 8, KIND_SURROGATE, "PERSON"),
            ReplacementRegion(12, 16, KIND_SURROGATE, "PERSON"),
        )
    )

    assert provenance.uncovered(0, 20) == [(0, 4), (8, 12), (16, 20)]
    assert provenance.uncovered(4, 8) == []
    assert provenance.uncovered(6, 14) == [(8, 12)]
