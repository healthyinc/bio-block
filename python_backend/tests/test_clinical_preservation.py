"""The shared clinical-preservation policy, and the three defect classes it closes.

Found on the frozen delta corpus v3, whose result stands unchanged:

- a released PDF lost a unit: the proper-noun fallback read "mU" in
  "3.1 mU/L" as a name, and the redaction took the whole word;
- a clinician letter was held: removing an email left "Queries to" alone on
  its line, and a model read "Queries" as a person;
- a pathology slide was held: a model read "PAS-D stain" as a passport number.

Each is fixed as a class - unit grammar after a result, authentic tokens
written where a value was removed, stains in pathology context - and each
fix is refused wherever privacy evidence touches the span. Every value is
invented.
"""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIOBLOCK_STUDY_SALT", "clinical-preservation-test-salt")

from services import clinical_preservation as cp  # noqa: E402
from services.detection_evidence import assess_finding  # noqa: E402

fitz = pytest.importorskip("fitz", reason="PyMuPDF is required for PDF tests")


@pytest.fixture(autouse=True)
def _legacy_models(monkeypatch):
    monkeypatch.setenv("PHI_MODEL_MODE", "legacy_test")


def _action(text, span, category="PERSON", source="ner", agreeing=1, occurrence=0):
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(span, start + 1)
    return assess_finding(
        text, start, start + len(span), category, source, agreeing_detectors=agreeing
    ).action


# -- units -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit",
    [
        "mU/L",
        "mmol/L",
        "mmol/mol",
        "g/dL",
        "µmol/L",
        "μmol/L",
        "umol/L",
        "ng/mL",
        "IU/L",
        "mL/min/1.73m2",
        "mL/min/1.73m²",
        "x10^9/L",
        "×10⁹/L",
        "10*9/L",
        "%",
        "mmHg",
        "mg/kg",
        "U/L",
        "pg/mL",
        "mEq/L",
        "kPa",
    ],
)
def test_simple_and_compound_units_parse_as_whole_expressions(unit):
    assert cp.is_unit_expression(unit)


@pytest.mark.parametrize("text", ["10^9", "Smith", "mU/Lx", "MRN", "", "L/"])
def test_things_that_are_not_units_do_not_parse(text):
    assert not cp.is_unit_expression(text)


@pytest.mark.parametrize(
    "text,span",
    [
        ("TSH 3.1 mU/L on the current dose.", "mU"),
        ("TSH 3.1 mU/L on the current dose.", "mU/L"),
        ("TSH 3.1\nmU/L on the current dose.", "mU/L"),  # split across text runs
        ("Ferritin 14 µg/L, low.", "µg/L"),
        ("eGFR 48 mL/min/1.73m² and stable.", "mL/min/1.73m²"),
        ("WCC 7.2 ×10⁹/L this morning.", "×10⁹/L"),
        ("Creatinine <44 umol/L.", "umol/L"),
    ],
)
def test_a_unit_after_a_result_is_kept_against_the_proper_noun_fallback(text, span):
    assert _action(text, span, "PERSON", "strict_proper_noun") == "preserve"
    assert _action(text, span, "ORGANIZATION", "ner", agreeing=2) == "preserve"


@pytest.mark.parametrize(
    "text,span,category",
    [
        ("MRN 3.1 mU/L", "mU/L", "ORGANIZATION"),  # identifier label first
        ("Call 0161 496 0558 mL now", "0161 496 0558", "PHONE_NUMBER"),
        ("Email mg.lo@example.org", "mg.lo@example.org", "EMAIL_ADDRESS"),
        ("Glucose 5.4 mmol/L Smith attended.", "Smith", "PERSON"),
        ("Dr 3 L Harrow", "Harrow", "PERSON"),
    ],
)
def test_the_same_characters_in_identifiers_and_names_are_still_removed(
    text, span, category
):
    source = (
        "structured_pattern" if category in ("PHONE_NUMBER", "EMAIL_ADDRESS") else "ner"
    )
    assert _action(text, span, category, source) == "redact"


@pytest.mark.parametrize(
    "text,span",
    [
        ("Licence 20 mg", "mg"),
        ("MRN 3.1 mU/L", "mU/L"),
        ("Tel 0161 496 0558 mL", "mL"),
        ("Passport 44 IU", "IU"),
        ("Dr 5 mg Harrow", "mg"),
    ],
)
def test_the_policy_refuses_units_after_identifier_labels_titles_and_phone_digits(
    text, span
):
    start = text.index(span)
    assert not cp.assess_clinical_span(text, start, start + len(span)).preserve


def test_units_are_found_only_after_a_result_and_never_after_an_identifier_label():
    text = "Hb 128 g/L, 98% sats, WCC 7.2 x10^9/L, ref 20 mg, phone 0161 496 0558 mL"
    found = [text[start:end] for start, end, _unit in cp.measurements(text)]

    assert found == ["128 g/L", "98%", "7.2 x10^9/L"]


# -- stains and imaging sequences ------------------------------------------------------


@pytest.mark.parametrize(
    "text,span,category",
    [
        (
            "PAS-D stain|Objective = 20x|Pixel size = 0.25 um",
            "PAS-D stain",
            "PASSPORT_NUMBER",
        ),
        ("Ki-67 immunohistochemistry of the biopsy.", "Ki-67", "ORGANIZATION"),
        ("Gram stain of the biopsy was negative.", "Gram", "PERSON"),
        ("Masson trichrome staining of the section.", "Masson trichrome", "PERSON"),
        ("CD20 positive on the biopsy section.", "CD20", "ORGANIZATION"),
        (
            "Congo red stain|Objective = 20x|Pixel size = 0.25 um",
            "Congo red stain",
            "ORGANIZATION",
        ),
        ("Perls stain on the marrow biopsy.", "Perls", "PERSON"),
    ],
)
def test_stains_in_pathology_context_survive_two_agreeing_models(text, span, category):
    assert _action(text, span, category, "gliner_multi_pii", agreeing=2) == "preserve"


@pytest.mark.parametrize(
    "text,span,category",
    [
        ("Passport: PAS-D stain", "PAS-D stain", "PASSPORT_NUMBER"),
        (
            "Registration number PAS-D stain on the biopsy",
            "PAS-D stain",
            "PASSPORT_NUMBER",
        ),
        ("Seen by Dr Gram about the biopsy.", "Gram", "PERSON"),
        (
            "Gram reviewed the biopsy today.",
            "Gram",
            "PERSON",
        ),  # an eponym alone is a name
        ("<PERSON_1> Ki-67 staining of the biopsy.", "Ki-67", "ORGANIZATION"),
        ("PATIENT_004 PAS-D stain of the biopsy.", "PAS-D stain", "PASSPORT_NUMBER"),
        ("PAS-D7731 on the slide", "PAS-D7731", "PASSPORT_NUMBER"),
        ("PAS-D stain", "PAS-D stain", "PASSPORT_NUMBER"),  # no pathology context
        (
            "Congo attended the biopsy clinic.",
            "Congo",
            "PERSON",
        ),  # a colour dye's first word
        ("Perls reviewed the biopsy.", "Perls", "PERSON"),
    ],
)
def test_stain_like_identifiers_and_names_stay_protected(text, span, category):
    assert _action(text, span, category, "gliner_multi_pii") == "redact"


@pytest.mark.parametrize(
    "text,span",
    [
        ("Sagittal STIR shows marrow oedema.", "STIR"),
        ("Axial SWI shows no blooming.", "SWI"),
        ("MRI brain: DWI shows no restricted diffusion.", "DWI"),
    ],
)
def test_sequences_in_imaging_context_are_kept(text, span):
    assert _action(text, span, "ORGANIZATION", "ner", agreeing=2) == "preserve"


@pytest.mark.parametrize(
    "text,span",
    [
        ("Mr STIR Hollis attended the MRI.", "STIR"),
        ("Seen by Dr Flair for the MRI.", "Flair"),
        ("Reference STIR44 for the MRI scan.", "STIR44"),
    ],
)
def test_sequence_like_names_and_identifiers_stay_protected(text, span):
    assert _action(text, span, "PERSON", "ner") == "redact"


def test_deterministic_evidence_is_decided_before_the_policy():
    text = "Passport number 123456789 for the biopsy stain."
    assert (
        _action(text, "123456789", "PASSPORT_NUMBER", "structured_pattern") == "redact"
    )


# -- policy data, decisions and versioning ---------------------------------------------


def test_the_policy_validates_itself_and_is_versioned():
    assert cp.POLICY_VERSION == "clinical-preservation-v2"
    decision = cp.assess_clinical_span("TSH 3.1 mU/L today", 8, 12)
    assert decision.preserve and decision.category is cp.TermCategory.UNIT
    public = decision.public_dict()
    assert public["policy_version"] == cp.POLICY_VERSION
    assert "term" not in public and "mU" not in repr(public)


def test_invalid_policy_data_fails_at_validation(monkeypatch):
    monkeypatch.setattr(cp, "_STAIN_TERMS", cp._STAIN_TERMS + ("pas",))
    with pytest.raises(cp.PolicyError):
        cp._validate()
    monkeypatch.setattr(cp, "_STAIN_TERMS", ("Bad Entry!",))
    with pytest.raises(cp.PolicyError):
        cp._validate()
    monkeypatch.setattr(cp, "_STAIN_TERMS", ())
    with pytest.raises(cp.PolicyError):
        cp._validate()


def test_decisions_are_deterministic():
    text = "PAS-D stain|Objective = 20x|Pixel size = 0.25 um. TSH 3.1 mU/L."
    assert cp.clinical_invariants(text) == cp.clinical_invariants(text)
    assert cp.invariant_keys(text) == cp.invariant_keys(text)


def test_the_utility_gate_names_only_unapproved_losses():
    text = "TSH 3.1 mU/L and MRN 60318842 on PAS-D stain of the biopsy."
    unit = text.index("mU/L")
    assert cp.unattributed_losses(text, [(unit, unit + 2, False)]) == (
        cp.CLINICAL_UNIT_CHANGED,
    )
    stain = text.index("PAS-D")
    assert cp.unattributed_losses(text, [(stain, stain + 5, False)]) == (
        cp.CLINICAL_TERM_CHANGED,
    )
    record = text.index("60318842")
    assert cp.unattributed_losses(text, [(record, record + 8, True)]) == ()
    # Removing the unit also breaks the analyte-result-unit relation.
    assert cp.CLINICAL_UNIT_CHANGED in cp.lost_invariants(
        text, text.replace("mU/L", "")
    )


def test_the_anonymizer_holds_a_note_whose_unit_a_heuristic_removed(monkeypatch):
    from services import text_anonymization as ta
    from services.phi_detection import PhiEntity

    text = "TSH 3.1 mU/L on the current dose."
    start = text.index("mU")
    forced = [PhiEntity("PERSON", start, start + 2, "strict_proper_noun")]
    monkeypatch.setattr(ta, "_detect_entities", lambda *args, **kwargs: (forced, []))
    result = ta.anonymize_clinical_text(text)
    assert cp.CLINICAL_UNIT_CHANGED in result["review_required_reasons"]


# -- PDF -------------------------------------------------------------------------------


# -- TIFF / WSI ------------------------------------------------------------------------


# -- CSV -------------------------------------------------------------------------------
