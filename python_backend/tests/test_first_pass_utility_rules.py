"""First-pass utility hardening: false positives that destroyed clinical content.

Phase 12G measured, against gold labels, what the first pass removed from
documents that carry no identifier at all: a blood pressure read as an
organisation, "Ward staff" read as a person, "the review clinic" read as a
named body. Each fix here is a grammatical or closed-class rule rather than a
list of the sentences that failed, so every section tests unseen variations,
and each also tests the name that must still be removed from the same place.

The development corpus and the acceptance-v3 templates are not quoted here
beyond the original failing shapes. All values are invented.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIOBLOCK_STUDY_SALT", "first-pass-utility-test-salt")

from services.detection_evidence import assess_finding  # noqa: E402


@pytest.fixture(autouse=True)
def _legacy_models(monkeypatch):
    monkeypatch.setenv("PHI_MODEL_MODE", "legacy_test")


def _assess(text, span, category, source="ner", **kwargs):
    start = text.index(span)
    return assess_finding(text, start, start + len(span), category, source, **kwargs)


def _anonymized(text):
    from services import text_anonymization as ta

    ta._build_detectors.cache_clear()
    return ta.anonymize_clinical_text(text, profile="strict")["anonymized_text"]


# -- extracted image text keeps its fail-closed default ------------------------
#
# classify_extracted_text decides whether a burned-in OCR box on an image is
# kept, and it reads the same evidence model as the first pass. Only
# EXTRACTED_NON_PHI keeps a box. The rules above may turn a false positive into
# "uncertain" - which the caller still redacts - but must never let a name or
# an identifier come back as safe to keep. No test covered this path before.


def _classify(text):
    from services import text_anonymization as ta

    ta._build_detectors.cache_clear()
    return ta.classify_extracted_text(text)


def test_encoding_alias_label_is_preserved_but_the_alias_name_is_not():
    text = "Clinical note; Unicode alias: Zoë Fictional was seen."
    assert (
        _assess(text, "Unicode", "PERSON", contains_proper_noun=True).action
        == "preserve"
    )
    assert (
        _assess(text, "Zoë Fictional", "PERSON", contains_proper_noun=True).action
        == "redact"
    )
    assert (
        _assess(
            "Patient Unicode attended.", "Unicode", "PERSON", contains_proper_noun=True
        ).action
        == "redact"
    )


def test_companion_name_and_attending_subject_have_contextual_justification():
    from services.detection_evidence import phi_justification

    text = "Aster Vale-001 attended with SyntheticRelative-001."
    for span in ("Aster", "Vale", "SyntheticRelative-001"):
        start = text.index(span)
        assert phi_justification(text, start, start + len(span), "PERSON", 1, False)
    unicode_text = "Zoë Fictional-059 attended with Noël Invented-059."
    start = unicode_text.index("Zoë")
    assert (
        phi_justification(unicode_text, start, start + len("Zoë"), "PERSON", 1, False)
        == "person_context_present"
    )
    decoy = "HbA1c attended with serum 4417."
    assert (
        phi_justification(
            decoy, decoy.index("serum"), decoy.index("serum") + 5, "PERSON", 1, True
        )
        is None
    )


def test_relationship_noun_and_demographic_label_stay_but_names_do_not():
    text = "Patient attended with a housemate Rowan Example; Asian MRN SYN-MRN-410000."
    assert (
        _assess(text, "housemate", "PERSON", contains_proper_noun=True).action
        == "preserve"
    )
    assert (
        _assess(text, "Asian", "PERSON", contains_proper_noun=True).action == "preserve"
    )
    assert (
        _assess(text, "Rowan Example", "PERSON", contains_proper_noun=True).action
        == "redact"
    )
    relationship = "Patient attended with a housemate Housemate Fiction-012."
    assert (
        _assess(
            relationship,
            "housemate",
            "LOCATION",
            agreeing_detectors=2,
            contains_proper_noun=True,
        ).action
        == "preserve"
    )
    assert (
        _assess(
            relationship, "Housemate Fiction-012", "PERSON", contains_proper_noun=True
        ).action
        == "redact"
    )
    compound = "Patient attended with housemate Émile Synthetic-059."
    assert (
        _assess(
            compound,
            "housemate Émile Synthetic-059",
            "ORGANIZATION",
            contains_proper_noun=False,
        ).action
        == "redact"
    )
    assert (
        _assess(
            "Dr Asian was seen.", "Asian", "PERSON", contains_proper_noun=True
        ).action
        == "redact"
    )


def test_a_model_span_of_category_word_and_record_label_holds_no_name():
    """GLiNER proposed 'Asian MRN' (beside a replaced record number) as a PERSON."""
    text = "Occupation: Nurse <PERSON_1>; Asian MRN <RECORD_ID_1>; extension 4417."
    kwargs = {
        "agreeing_detectors": 2,
        "contains_proper_noun": True,
        "source": "gliner_multi_pii",
    }
    decision = _assess(text, "Asian MRN", "PERSON", **kwargs)
    assert decision.action == "preserve" and decision.justified
    # The exemption is exhaustive and whole-label: any added word, a surname-like
    # race word, a naming context, or an accented name keeps the span redacted.
    for added in ("Asian MRN Jordan", "Asian Noël Example", "Black MRN", "White MRN"):
        sample = f"Seen; {added} <RECORD_ID_1>."
        assert _assess(sample, added, "PERSON", **kwargs).action != "preserve", added
    assert (
        _assess("Dr Asian MRN was seen.", "Asian MRN", "PERSON", **kwargs).action
        != "preserve"
    )
    assert (
        _assess(
            "Patient attended with Asian Noël Example.",
            "Asian Noël Example",
            "PERSON",
            **kwargs,
        ).action
        != "preserve"
    )


@pytest.mark.parametrize(
    "text,span,category",
    [
        (
            "Seen; fingerprint template FP-SYN-6152; stable.",
            "fingerprint template FP-SYN-6152",
            "ORGANIZATION",
        ),
        ("Seen; voiceprint VP-SYN-6153; stable.", "voiceprint VP-SYN-6153", "FACILITY"),
        (
            "Seen; vehicle registration YN52-ABC42; stable.",
            "vehicle registration YN52-ABC42",
            "PERSON",
        ),
        ("Seen; postal code 89111 noted; stable.", "postal code 89111", "ORGANIZATION"),
        (
            "Seen; zip code 02110-1010 recorded; stable.",
            "zip code 02110-1010",
            "LOCATION",
        ),
    ],
)
def test_a_label_and_a_value_of_that_fields_shape_is_a_justified_edit(
    text, span, category
):
    decision = _assess(
        text, span, category, source="gliner_multi_pii", contains_proper_noun=True
    )
    assert (decision.action, decision.justified, decision.rationale) == (
        "redact",
        True,
        "announced_identifier_value",
    )


@pytest.mark.parametrize(
    "text,span,category",
    [
        (
            "Seen; iris scan Jordan Fictional; stable.",
            "iris scan Jordan Fictional",
            "PERSON",
        ),
        ("Seen; postal code HbA1c; stable.", "postal code HbA1c", "ORGANIZATION"),
        ("Seen; record 89111 noted; stable.", "record 89111", "ORGANIZATION"),
        ("Seen; K80.20 noted; stable.", "K80.20", "ORGANIZATION"),
        ("Seen; Zoë Fictional-12 noted; stable.", "Zoë Fictional-12", "ORGANIZATION"),
    ],
)
def test_other_label_value_shapes_are_not_justified_by_that_rule(text, span, category):
    decision = _assess(
        text, span, category, source="gliner_multi_pii", contains_proper_noun=True
    )
    assert decision.rationale != "announced_identifier_value", (text, decision)


def test_a_code_after_its_field_label_outside_the_span_is_justified():
    kept = _assess(
        "Seen; iris scan IR-SYN-6154; stable.",
        "IR-SYN-6154",
        "ORGANIZATION",
        source="ner",
        contains_proper_noun=True,
    )
    link = _assess(
        "Seen; link http://records.example.org/chart?id=48; stable.",
        "link http://records.example.org/chart?id=48",
        "PERSON",
        source="ner",
        contains_proper_noun=True,
    )
    assert (kept.action, kept.justified) == (
        "redact",
        True,
    ) or kept.action == "preserve"
    assert (link.action, link.justified, link.rationale) == (
        "redact",
        True,
        "announced_identifier_value",
    )
    # A name or ordinary word before the code, or a field break between them,
    # is not a label.
    for text in (
        "Seen; Jordan IR-SYN-6154; stable.",
        "Seen; iris scan. IR-SYN-6154 stable.",
        "Seen; HbA1c IR-SYN-6154; stable.",
        "Seen; scan 7.2 stable.",
    ):
        value = "IR-SYN-6154" if "IR-SYN" in text else "7.2"
        decision = _assess(
            text, value, "ORGANIZATION", source="ner", contains_proper_noun=True
        )
        assert decision.rationale != "announced_identifier_value", (text, decision)


@pytest.mark.parametrize(
    "label", ["SSN", "iris scan", "postal code", "vehicle registration"]
)
def test_a_bare_identifier_field_label_is_kept_not_replaced(label):
    text = f"Seen; {label} <RECORD_ID_1>; stable."
    decision = _assess(
        text,
        label,
        "ORGANIZATION",
        source="gliner_multi_pii",
        contains_proper_noun=False,
    )
    assert decision.action == "preserve", (label, decision)
    # A name that merely follows a label is never swept in with it.
    named = _assess(
        f"Seen; {label} Jordan Fictional; stable.",
        f"{label} Jordan Fictional",
        "PERSON",
        source="gliner_multi_pii",
        contains_proper_noun=True,
    )
    assert (
        named.action != "preserve" or named.rationale != "span_contains_no_name_token"
    )


@pytest.mark.parametrize(
    "text",
    [
        "Zelquin Vantreske",
        "Dr Quenbarrow",
        "Rilvendale serum",
        "left Ashvelin",
        "Ravensholt team",
    ],
)
def test_an_extracted_name_is_never_kept(text):
    from services.text_anonymization import EXTRACTED_NON_PHI

    assert _classify(text) != EXTRACTED_NON_PHI


@pytest.mark.parametrize(
    "text",
    [
        "zvantreske@example.invalid",
        "961-22-4410",
        "198.51.100.77",
    ],
)
def test_an_extracted_identifier_is_never_kept(text):
    from services.text_anonymization import EXTRACTED_NON_PHI

    assert _classify(text) != EXTRACTED_NON_PHI


def test_a_role_phrase_box_is_not_kept_without_a_confirmed_marker():
    """Preserving it in prose does not make an image box safe to keep."""
    from services.text_anonymization import EXTRACTED_NON_PHI

    assert _classify("Ward staff") != EXTRACTED_NON_PHI


def test_a_burned_in_measurement_marker_is_still_kept():
    from services.text_anonymization import EXTRACTED_NON_PHI

    assert _classify("12 mm") == EXTRACTED_NON_PHI


# -- A: measurements ---------------------------------------------------------


@pytest.mark.parametrize(
    "text,span,category",
    [
        ("BP 128/76, HbA1c 7.2 percent.", "BP 128/76", "ORGANIZATION"),
        ("Obs recorded BP 142/90 at triage.", "BP 142/90", "ORGANIZATION"),
        ("HR 104, RR 22 overnight.", "HR 104, RR 22", "ORGANIZATION"),
        ("Sats SpO2 94% on air.", "SpO2 94%", "PERSON"),
        ("Temp 38.4 C at handover.", "Temp 38.4 C", "ORGANIZATION"),
        ("Sodium was 139 mmol/L on repeat.", "139 mmol/L", "ORGANIZATION"),
        ("GCS 14 on arrival.", "GCS 14", "FACILITY"),
    ],
)
def test_a_measurement_is_not_a_name(text, span, category):
    assessment = _assess(text, span, category, contains_proper_noun=True)

    assert assessment.action == "preserve"
    assert assessment.rationale == "measurement_value_grammar"


def test_a_vital_label_before_its_value_is_not_a_name():
    assessment = _assess(
        "BP 142/90 and temperature 37.9 C.",
        "BP",
        "PERSON",
        source="strict_proper_noun",
        contains_proper_noun=True,
    )

    assert assessment.action == "preserve"


def test_many_agreeing_detectors_do_not_make_a_measurement_a_name():
    assessment = _assess(
        "BP 128/76 on arrival.",
        "BP 128/76",
        "ORGANIZATION",
        agreeing_detectors=3,
        contains_proper_noun=True,
    )

    assert assessment.action == "preserve"


def test_a_bare_ratio_stays_redacted_because_it_could_be_a_date():
    assessment = _assess(
        "Seen again on 12/03 for review.",
        "12/03",
        "ORGANIZATION",
        source="gliner_multi_pii",
        contains_proper_noun=True,
    )

    assert assessment.action == "redact"


def test_a_name_beside_a_number_is_still_a_name():
    assessment = _assess(
        "Ellworth 72 attended.",
        "Ellworth 72",
        "PERSON",
        source="gliner_multi_pii",
        contains_proper_noun=True,
    )

    assert assessment.action == "redact"


def test_naming_context_outranks_measurement_grammar():
    assessment = _assess(
        "Reviewed by Dr HR 88 today.",
        "HR 88",
        "PERSON",
        source="gliner_multi_pii",
        contains_proper_noun=True,
    )

    assert assessment.action == "redact"


def test_places_are_not_given_measurement_protection():
    assessment = _assess(
        "Moved to unit 97% complete.",
        "97%",
        "LOCATION",
        source="gliner_multi_pii",
        contains_proper_noun=False,
    )

    assert assessment.rationale != "measurement_value_grammar"


def test_a_measurement_survives_and_the_address_beside_it_does_not():
    text = "Levothyroxine 75 mcg each morning.\nBP 128/76, 198.51.100.77, HbA1c 7.2 percent."

    output = _anonymized(text)

    assert "BP 128/76" in output
    assert "198.51.100.77" not in output


# -- C: role and service phrases ---------------------------------------------


from services.clinical_vocabulary import is_non_name_label  # noqa: E402


@pytest.mark.parametrize(
    "phrase",
    [
        "Ward staff",
        "Theatre staff",
        "Pharmacy team",
        "nursing team",
        "surgical registrar",
        "Community midwives",
        "ambulance crew",
    ],
)
def test_a_role_or_service_phrase_is_not_a_name(phrase):
    text = f"{phrase} documented no overnight events."

    assessment = _assess(
        text, phrase, "PERSON", source="gliner_multi_pii", contains_proper_noun=True
    )

    assert assessment.action == "preserve"


def test_two_detectors_on_a_structural_role_phrase_keep_it():
    """A role named by its service or shift is proven by structure: two models
    agreeing on a PERSON reading share one misreading. A name in front of the
    role is still a name (next test)."""
    text = "Ward staff documented no overnight events."

    assessment = _assess(
        text,
        "Ward staff",
        "PERSON",
        source="gliner_multi_pii",
        agreeing_detectors=2,
        contains_proper_noun=True,
    )

    assert (assessment.action, assessment.rationale) == (
        "preserve",
        "staff_role_phrase",
    )


def test_a_name_in_front_of_a_role_noun_is_still_a_name():
    text = "Ravensholt team documented no overnight events."

    assessment = _assess(
        text,
        "Ravensholt team",
        "PERSON",
        source="gliner_multi_pii",
        contains_proper_noun=True,
    )

    assert assessment.action == "redact"


def test_a_named_nurse_is_still_redacted():
    text = "Staff nurse Ashvelin Thackerwild documented the dose."

    assessment = _assess(
        text,
        "Ashvelin Thackerwild",
        "PERSON",
        source="gliner_multi_pii",
        contains_proper_noun=True,
    )

    assert assessment.action == "redact"


@pytest.mark.parametrize(
    "surname",
    [
        "porter",
        "day",
        "night",
        "carter",
        "taylor",
        "cook",
        "mason",
        "bishop",
        "marshal",
    ],
)
def test_common_surnames_are_not_role_labels(surname):
    assert not is_non_name_label(surname)
    assert not is_non_name_label(surname.capitalize())


# -- D: organisation designators ---------------------------------------------


from services.detection_evidence import clinical_reading_strength  # noqa: E402


@pytest.mark.parametrize(
    "text,span",
    [
        (
            "Patient PATIENT_001 attended the review clinic with PROVIDER_001.",
            "review clinic",
        ),
        ("She was reviewed in the fracture clinic.", "fracture clinic"),
        ("Referred to a diabetes clinic next month.", "diabetes clinic"),
        ("Review clinic appointment was booked.", "Review clinic"),
        ("Samples went to the pathology laboratory.", "pathology laboratory"),
    ],
)
def test_a_generic_service_is_not_a_named_organisation(text, span):
    start = text.index(span)

    _reading, rationale = clinical_reading_strength(text, start, start + len(span))

    assert rationale != "organisation_designator_present"


def test_the_service_survives_end_to_end_when_no_proper_noun_is_present():
    text = "Patient PATIENT_001 attended the review clinic with PROVIDER_001."

    assessment = _assess(
        text,
        "review clinic",
        "ORGANIZATION",
        source="gliner_multi_pii",
        contains_proper_noun=False,
    )

    assert assessment.action == "preserve"


@pytest.mark.parametrize(
    "text,span",
    [
        ("Letter sent to Marlowbridge Clinic yesterday.", "Marlowbridge Clinic"),
        (
            "Samples were sent to Verrando Pressure Diagnostics Ltd.",
            "Verrando Pressure Diagnostics Ltd",
        ),
        (
            "Transferred from the Netherquay Infirmary overnight.",
            "the Netherquay Infirmary",
        ),
        ("Invoice from 3 Rivers Laboratory attached.", "3 Rivers Laboratory"),
    ],
)
def test_a_named_organisation_still_carries_the_designator(text, span):
    start = text.index(span)

    reading, rationale = clinical_reading_strength(text, start, start + len(span))

    assert (reading, rationale) == ("identifier", "organisation_designator_present")


def test_a_named_clinic_opening_a_sentence_is_still_redacted():
    """Sentence case removes the designator evidence, not the proper-noun evidence."""
    text = "Marlowbridge Clinic wrote back on Monday."

    assessment = _assess(
        text,
        "Marlowbridge Clinic",
        "ORGANIZATION",
        source="gliner_multi_pii",
        contains_proper_noun=True,
    )

    assert assessment.action == "redact"


# -- B: specimen qualifiers ---------------------------------------------------


@pytest.mark.parametrize(
    "text,span",
    [
        ("Serum potassium 4.1 mmol/L and sodium 139 mmol/L.", "Serum"),
        ("Venous lactate 2.3 mmol/L on repeat.", "Venous"),
        ("Plasma glucose 6.1 mmol/L after the meal.", "Plasma"),
        ("Capillary ketones 0.4 mmol/L at handover.", "Capillary"),
        ("Urine sodium 42 mmol/L was sent.", "Urine"),
        ("Arterial lactate 1.8 mmol/L improved.", "Arterial"),
    ],
)
def test_a_specimen_qualifier_before_a_measurement_is_not_a_name(text, span):
    assessment = _assess(text, span, "ORGANIZATION", contains_proper_noun=True)

    assert assessment.action == "preserve"
    assert assessment.rationale == "specimen_measurement_grammar"


def test_a_surname_in_front_of_an_analyte_is_still_a_name():
    text = "Quenbarrow potassium 4.1 mmol/L noted."

    assessment = _assess(
        text,
        "Quenbarrow",
        "PERSON",
        source="gliner_multi_pii",
        contains_proper_noun=True,
    )

    assert assessment.action == "redact"


def test_a_name_spanning_into_the_qualifier_is_still_a_name():
    text = "Rilvendale serum sodium 141 mmol/L."

    assessment = _assess(
        text,
        "Rilvendale serum",
        "PERSON",
        source="gliner_multi_pii",
        contains_proper_noun=True,
    )

    assert assessment.action == "redact"


def test_a_qualifier_without_a_measurement_gets_no_protection():
    text = "Serum was sent to the laboratory."

    start = text.index("Serum")
    _reading, rationale = clinical_reading_strength(text, start, start + len("Serum"))

    assert rationale != "specimen_measurement_grammar"


def test_naming_context_outranks_a_qualifier():
    text = "Reviewed by Dr Plasma potassium 4.1 mmol/L."

    assessment = _assess(
        text, "Plasma", "PERSON", source="gliner_multi_pii", contains_proper_noun=True
    )

    assert assessment.action == "redact"


def test_a_specimen_survives_end_to_end():
    output = _anonymized(
        "Serum potassium 4.1 mmol/L and sodium 139 mmol/L on repeat sampling."
    )

    assert "Serum potassium 4.1 mmol/L" in output


def test_an_unseen_specimen_survives_end_to_end():
    output = _anonymized(
        "Venous lactate 2.3 mmol/L and plasma glucose 6.1 mmol/L were repeated."
    )

    assert "Venous lactate 2.3 mmol/L" in output


# -- E: anatomical position ---------------------------------------------------


@pytest.mark.parametrize(
    "text,span",
    [
        ("The left iliac crest was tender to palpation.", "crest"),
        ("The right femoral head was tender on rotation.", "head"),
        ("The posterior tibial pulse was palpable.", "tibial pulse"),
        ("Swelling over the medial malleolus persisted.", "malleolus"),
        ("Bruising on both lower limbs.", "limbs"),
    ],
)
def test_a_site_governed_by_laterality_is_not_a_name(text, span):
    assessment = _assess(
        text, span, "PERSON", source="strict_proper_noun", contains_proper_noun=True
    )

    assert assessment.action == "preserve"
    assert assessment.rationale == "anatomical_position_grammar"


def test_left_as_a_verb_does_not_protect_a_lowercase_name():
    text = "He left rukmini with the discharge letter."

    assessment = _assess(
        text,
        "rukmini",
        "PERSON",
        source="strict_proper_noun",
        contains_proper_noun=True,
    )

    assert assessment.action == "redact"


def test_a_capitalised_name_after_laterality_is_still_a_name():
    text = "The left Ashvelin file was archived."

    assessment = _assess(
        text,
        "Ashvelin",
        "PERSON",
        source="strict_proper_noun",
        contains_proper_noun=True,
    )

    assert assessment.action == "redact"


def test_a_sentence_boundary_ends_the_construction():
    text = "Patient went left. rukmini called later."

    assessment = _assess(
        text,
        "rukmini",
        "PERSON",
        source="strict_proper_noun",
        contains_proper_noun=True,
    )

    assert assessment.action == "redact"


def test_the_anatomical_site_survives_end_to_end():
    output = _anonymized(
        "The left iliac crest was tender to palpation without guarding."
    )

    assert "left iliac crest" in output


def test_a_lower_case_name_without_laterality_is_still_removed_end_to_end():
    output = _anonymized(
        "CASE ESCALATED BY rukmini balasubramanian and reviewed later."
    )

    assert "balasubramanian" not in output


# -- second round: what acceptance-v3 showed the first round missed --------------
#
# acceptance-v3 is development evidence now, not an acceptance partition. The
# cases below are invented variations of its failure classes, never its text.


@pytest.mark.parametrize(
    "text,span",
    [
        (
            "Tenderness over the left anatomical snuffbox persisted.",
            "anatomical snuffbox",
        ),
        ("Pain in her right thenar eminence settled.", "thenar eminence"),
        ("The left snuffbox was tender.", "snuffbox"),
        ("Swelling of the right fifth metacarpal was noted.", "fifth metacarpal"),
    ],
)
def test_an_article_or_possessive_makes_laterality_an_adjective(text, span):
    assessment = _assess(
        text, span, "PERSON", source="strict_proper_noun", contains_proper_noun=True
    )

    assert assessment.rationale == "anatomical_position_grammar"


@pytest.mark.parametrize(
    "text,span",
    [
        ("The nurse that left rukmini devi at the desk returned.", "rukmini devi"),
        ("Mother and left rukmini devi alone.", "rukmini devi"),
        ("He left rukmini devi with the letter.", "rukmini devi"),
    ],
)
def test_left_as_a_verb_still_does_not_protect_a_lowercase_name(text, span):
    assessment = _assess(
        text, span, "PERSON", source="strict_proper_noun", contains_proper_noun=True
    )

    assert assessment.action == "redact"


@pytest.mark.parametrize(
    "text,span",
    [
        ("Urine microscopy showed no white cells.", "Urine"),
        ("Sputum culture grew no organisms.", "Sputum"),
        ("Stool microscopy was negative.", "Stool microscopy"),
        ("Blood cultures were sent at triage.", "Blood"),
        ("Arterial gas was repeated.", "Arterial"),
    ],
)
def test_a_specimen_before_a_laboratory_test_is_not_a_name(text, span):
    assessment = _assess(text, span, "PERSON", contains_proper_noun=True)

    assert assessment.rationale == "specimen_measurement_grammar"


@pytest.mark.parametrize(
    "text,span",
    [
        ("Quenbarrow microscopy was reviewed.", "Quenbarrow"),
        ("Urine Ashvelin collected the sample.", "Ashvelin"),
    ],
)
def test_a_name_beside_a_laboratory_test_is_still_a_name(text, span):
    assessment = _assess(
        text, span, "PERSON", source="gliner_multi_pii", contains_proper_noun=True
    )

    assert assessment.action == "redact"


@pytest.mark.parametrize(
    "phrase",
    [
        "Night nursing team",
        "Day staff",
        "weekend registrars",
        "Early nurses",
        "Overnight team",
    ],
)
def test_a_shift_word_before_a_staff_group_is_not_a_name(phrase):
    text = f"{phrase} reported two episodes of vomiting."

    assessment = _assess(
        text,
        phrase,
        "ORGANIZATION",
        source="gliner_multi_pii",
        contains_proper_noun=True,
    )

    assert assessment.action == "preserve"


@pytest.mark.parametrize(
    "phrase", ["Night Ravensholt team", "Taylor team", "Night Quenbarrow"]
)
def test_a_name_among_shift_words_is_still_a_name(phrase):
    text = f"{phrase} reported two episodes of vomiting."

    assessment = _assess(
        text,
        phrase,
        "ORGANIZATION",
        source="gliner_multi_pii",
        contains_proper_noun=True,
    )

    assert assessment.action == "redact"


def test_a_shift_word_on_its_own_is_still_not_a_label():
    assert not is_non_name_label("Night")
    assert not is_non_name_label("day")
    assert is_non_name_label("Night nursing team")


@pytest.mark.parametrize(
    "text,span,category",
    [
        ("BP 132/84 and HR 90 were recorded.", "BP 132/84", "PHONE_NUMBER"),
        ("HR 90 at rest.", "HR 90", "HEALTH_PLAN_ID"),
        ("RR 22 on arrival.", "RR 22", "IDENTIFIER"),
        ("BP 132/84 recorded at triage.", "132/84", "DATE_TIME"),
        ("Obs: BP 118/72, HR 64.", "BP 118/72, HR 64", "PHONE_NUMBER"),
        ("MAP 65 was maintained.", "MAP 65", "ADDRESS"),
    ],
)
def test_a_labelled_reading_is_not_an_identifier_whatever_a_model_calls_it(
    text, span, category
):
    assessment = _assess(
        text,
        span,
        category,
        source="gliner_multi_pii",
        agreeing_detectors=2,
        contains_proper_noun=True,
    )

    assert (assessment.action, assessment.rationale) == (
        "preserve",
        "measurement_value_grammar",
    )


def test_unitless_laboratory_result_is_preserved_without_shielding_same_digits_as_contact():
    text = "extension 4417; laboratory result 4417."
    result_start = text.index("laboratory result 4417")
    result = assess_finding(
        text,
        result_start,
        result_start + len("laboratory result 4417"),
        "HEALTH_PLAN_ID",
        "gliner_multi_pii",
        agreeing_detectors=2,
    )
    assert (result.action, result.rationale) == (
        "preserve",
        "measurement_value_grammar",
    )
    contact = _assess(
        text,
        "extension 4417",
        "HEALTH_PLAN_ID",
        source="gliner_multi_pii",
        agreeing_detectors=2,
    )
    assert contact.action != "preserve"
    mixed = _assess(
        "laboratory result 4417 and Jordan Fictional",
        "laboratory result 4417 and Jordan Fictional",
        "HEALTH_PLAN_ID",
        source="gliner_multi_pii",
        agreeing_detectors=2,
    )
    assert mixed.action != "preserve"


def test_unicode_names_with_invented_ordinals_keep_contextual_person_evidence():
    named = _assess(
        "Noël Invented-002 attended the visit.",
        "Noël Invented-002",
        "PERSON",
        source="gliner_multi_pii",
        agreeing_detectors=2,
        contains_proper_noun=True,
    )
    assert named.action == "redact" and named.justified
    related = _assess(
        "Seen with housemate Émile Synthetic-002.",
        "Émile Synthetic-002",
        "PERSON",
        source="gliner_multi_pii",
        contains_proper_noun=False,
    )
    assert related.action == "redact" and related.justified
    clinical = _assess(
        "Testosterone in Serum or Plasma was measured.",
        "Serum or Plasma",
        "PERSON",
        source="gliner_multi_pii",
        contains_proper_noun=True,
    )
    assert clinical.action == "preserve"


def test_synthetic_encounter_provenance_is_not_a_global_person_exemption():
    marker = _assess(
        "SYNTHETIC endocrine encounter; follow-up planned.",
        "SYNTHETIC",
        "PERSON",
        source="gliner_multi_pii",
        contains_proper_noun=True,
    )
    assert (marker.action, marker.rationale) == (
        "preserve",
        "synthetic_provenance_marker",
    )
    named = _assess(
        "Patient SYNTHETIC attended the visit.",
        "SYNTHETIC",
        "PERSON",
        source="gliner_multi_pii",
        contains_proper_noun=True,
    )
    assert named.action == "redact"
    for opening in (
        "SYNTHETIC record. Patient seen.",
        "SYNTHETIC clinical note; stable.",
        "SYNTHETIC record: follow-up.",
    ):
        kept = _assess(
            opening,
            "SYNTHETIC",
            "PERSON",
            source="gliner_multi_pii",
            contains_proper_noun=True,
        )
        assert (kept.action, kept.rationale) == (
            "preserve",
            "synthetic_provenance_marker",
        )
    # Not the opening marker of a described document: a name, a number, other
    # position, other case or a capitalized continuation is still a name.
    for text in (
        "SYNTHETIC Jordan Fictional was seen.",
        "SYNTHETIC 4417 record.",
        "Seen. SYNTHETIC record.",
        "Synthetic record. Seen.",
        "SYNTHETIC Noël record.",
        "SYNTHETIC, Jordan Fictional seen.",
    ):
        span = "Synthetic" if text.startswith("Synthetic") else "SYNTHETIC"
        result = _assess(
            text, span, "PERSON", source="gliner_multi_pii", contains_proper_noun=True
        )
        assert result.rationale != "synthetic_provenance_marker", text


@pytest.mark.parametrize(
    "text,span,category",
    [
        ("Reviewed with 555.0187 today.", "555.0187", "PHONE_NUMBER"),
        ("Seen again on 12/03 for review.", "12/03", "DATE_TIME"),
        ("HR 90123456 appears on the claim form.", "HR 90123456", "HEALTH_PLAN_ID"),
        ("Moved to unit 97% complete.", "97%", "LOCATION"),
    ],
)
def test_a_number_without_a_reading_label_gets_no_reading_protection(
    text, span, category
):
    assessment = _assess(
        text, span, category, source="gliner_multi_pii", contains_proper_noun=False
    )

    assert assessment.rationale != "measurement_value_grammar"


def test_a_unit_after_its_number_is_not_a_name():
    assessment = _assess(
        "Timed cortisol 452 nmol/L was within range.",
        "nmol",
        "PERSON",
        source="strict_proper_noun",
        contains_proper_noun=True,
    )

    assert assessment.action == "preserve"


def test_a_unit_shaped_word_with_no_number_gets_no_protection():
    assessment = _assess(
        "Nmol attended the clinic.",
        "Nmol",
        "PERSON",
        source="strict_proper_noun",
        contains_proper_noun=True,
    )

    assert assessment.action == "redact"


@pytest.mark.parametrize(
    "text,span,category,source",
    [
        ("Bay 4 remained stable overnight.", "Bay 4", "DATE_TIME", "ner"),
        ("| bay 7 | stable |", "bay 7", "ADDRESS", "gliner_multi_pii"),
        ("Moved to bed 12 after review.", "bed 12", "ORGANIZATION", "gliner_multi_pii"),
        ("Ward 7B handover was complete.", "Ward 7B", "FACILITY", "gliner_multi_pii"),
        ("Theatre 3 list overran.", "Theatre 3", "LOCATION", "gliner_multi_pii"),
        (
            "Isolated in side room 2 overnight.",
            "side room 2",
            "LOCATION",
            "gliner_multi_pii",
        ),
        ("Bay 4 remained stable overnight.", "Bay", "PERSON", "strict_proper_noun"),
    ],
)
def test_a_bed_or_bay_number_is_not_an_identifier(text, span, category, source):
    assessment = _assess(text, span, category, source=source, contains_proper_noun=True)

    assert (assessment.action, assessment.rationale) == (
        "preserve",
        "clinical_designator_grammar",
    )


@pytest.mark.parametrize(
    "text,span,category",
    [
        ("Lives at Bay 4 Harbour Road.", "Bay 4", "ADDRESS"),
        ("Flat above Ward 12 Street shops.", "Ward 12", "ADDRESS"),
        ("Reviewed by Dr Bay 4 today.", "Bay 4", "PERSON"),
    ],
)
def test_a_designator_inside_an_address_or_a_name_gets_no_protection(
    text, span, category
):
    assessment = _assess(
        text, span, category, source="gliner_multi_pii", contains_proper_noun=True
    )

    assert assessment.rationale != "clinical_designator_grammar"


@pytest.mark.parametrize(
    "text,span,category",
    [
        ("NEWS2 score 5 prompted escalation.", "NEWS2", "IDENTIFIER"),
        ("qSOFA 2 on arrival.", "qSOFA 2", "ORGANIZATION"),
        ("CURB-65 score of 3 was calculated.", "CURB-65", "IDENTIFIER"),
        ("Wells score 4 was documented.", "Wells score 4", "PERSON"),
        ("PHQ-9 of 12 was recorded.", "PHQ-9", "IDENTIFIER"),
    ],
)
def test_a_named_clinical_score_is_not_an_identifier(text, span, category):
    assessment = _assess(
        text, span, category, source="stanford_deidentifier", contains_proper_noun=True
    )

    assert (assessment.action, assessment.rationale) == (
        "preserve",
        "clinical_score_grammar",
    )


@pytest.mark.parametrize(
    "text,span",
    [
        ("Wells attended with his daughter.", "Wells"),
        ("Mr Wells score 4 on the form.", "Wells"),
    ],
)
def test_a_surname_that_names_a_score_is_still_a_name(text, span):
    assessment = _assess(
        text, span, "PERSON", source="gliner_multi_pii", contains_proper_noun=True
    )

    assert assessment.action == "redact"


def test_a_structured_match_is_never_preserved_by_clinical_grammar():
    text = "Bay 4 contact 555-0142."
    start = text.index("555-0142")

    assessment = assess_finding(
        text, start, start + 8, "PHONE_NUMBER", "structured_pattern"
    )

    assert assessment.action == "redact"


@pytest.mark.parametrize(
    "text,span",
    [
        ("Ward staff reported no overnight events.", "Ward staff"),
        ("Night nursing team called the registrar.", "Night nursing team"),
    ],
)
def test_a_reporting_verb_after_a_staff_phrase_is_not_naming_context(text, span):
    """Teams report and call; a verb staff use is not evidence of a personal name."""
    start = text.index(span)

    reading, rationale = clinical_reading_strength(text, start, start + len(span))

    assert (reading, rationale) == ("clinical", "staff_role_phrase")


@pytest.mark.parametrize(
    "text,span",
    [
        ("Ashvelin reported no overnight events.", "Ashvelin"),
        ("Dr Ward staff reported the change.", "Ward staff"),
    ],
)
def test_a_reporting_verb_still_marks_a_name_and_a_title_still_wins(text, span):
    start = text.index(span)

    reading, _rationale = clinical_reading_strength(text, start, start + len(span))

    assert reading == "identifier"


# -- trimming a model span that runs across a line -------------------------------------


from services.detection_evidence import trim_clinical_edges  # noqa: E402


@pytest.mark.parametrize(
    "text,span,category,remainder",
    [
        (
            "BP 132/84 LIC-4471220 HR 90.",
            "BP 132/84 LIC-4471220 HR 90",
            "HEALTH_PLAN_ID",
            "LIC-4471220",
        ),
        (
            "From: ZX-99172\nHR 104, RR 22",
            "ZX-99172\nHR 104, RR 22",
            "PHONE_NUMBER",
            "ZX-99172",
        ),
        ("Ellworth 12 mg daily", "Ellworth 12 mg", "PERSON", "Ellworth"),
        ("Bay 4 Quenbarrow reviewed", "Bay 4 Quenbarrow", "PERSON", "Quenbarrow"),
        (
            "NEWS2 score 5 AC-771 escalated",
            "NEWS2 score 5 AC-771",
            "IDENTIFIER",
            "AC-771",
        ),
    ],
)
def test_readings_at_the_edges_of_a_model_span_are_trimmed(
    text, span, category, remainder
):
    start = text.index(span)

    new_start, new_end = trim_clinical_edges(text, start, start + len(span), category)

    assert text[new_start:new_end] == remainder


def test_a_bare_reading_is_trimmed_when_its_label_sits_before_the_span():
    text = "BP 132/84 LIC-4471220 HR 90."
    start = text.index("132/84")

    new_start, new_end = trim_clinical_edges(
        text, start, text.index(" HR") + 6, "DATE_TIME"
    )

    assert text[new_start:new_end] == "LIC-4471220"


def test_a_wholly_clinical_span_is_not_trimmed_to_nothing():
    text = "HR 104, RR 22, SpO2 94% on air."
    span = "HR 104, RR 22, SpO2 94%"

    assert trim_clinical_edges(text, 0, len(span), "PHONE_NUMBER") == (0, len(span))


def test_a_place_never_loses_a_bare_unit_value():
    text = "Lives at 12 High Street 40 mm"
    span = "12 High Street 40 mm"
    start = text.index(span)

    new_start, new_end = trim_clinical_edges(text, start, start + len(span), "ADDRESS")

    assert text[new_start:new_end] == span


def test_readings_survive_and_the_identifier_between_them_does_not(monkeypatch):
    from services import text_anonymization as ta
    from services.phi_detection import StructuredPatternDetector

    text = "Obs: BP 132/84 LIC-4471220 HR 90 at triage."
    line = "BP 132/84 LIC-4471220 HR 90"

    class _LineModel:
        source = "gliner_multi_pii"

        def detect(self, value):
            from services.phi_detection import DetectedEntity

            if line not in value:
                return []
            start = value.index(line)
            return [
                DetectedEntity(
                    entity_type="HEALTH_PLAN_ID",
                    start=start,
                    end=start + len(line),
                    source=self.source,
                    score=0.9,
                )
            ]

    monkeypatch.setattr(
        ta,
        "_detectors",
        lambda *_args, **_kwargs: (StructuredPatternDetector(), _LineModel()),
    )

    output = ta.anonymize_clinical_text(text, profile="strict")["anonymized_text"]

    assert "BP 132/84" in output
    assert "HR 90" in output
    assert "4471220" not in output


def test_unseen_scaffold_lines_survive_end_to_end():
    text = (
        "Bay 6 remained settled overnight.\n"
        "Urine dipstick was negative for nitrites.\n"
        "NEWS2 score 3 was recorded at handover.\n"
        "Timed cortisol 380 nmol/L was noted.\n"
        "Tenderness over the right anatomical snuffbox eased.\n"
        "Evening nursing team completed the checks."
    )

    output = _anonymized(text)

    for phrase in (
        "Bay 6",
        "Urine dipstick",
        "NEWS2 score 3",
        "380 nmol/L",
        "right anatomical snuffbox",
        "Evening nursing team",
    ):
        assert phrase in output


# -- a model span that crosses a line break ---------------------------------------------


class _FixedSpanModel:
    """A stand-in model that proposes one fixed span wherever it appears."""

    def __init__(self, source, category, span):
        self.source, self.category, self.span = source, category, span

    def detect(self, value):
        from services.phi_detection import DetectedEntity

        if self.span not in value:
            return []
        start = value.index(self.span)
        return [
            DetectedEntity(
                entity_type=self.category,
                start=start,
                end=start + len(self.span),
                source=self.source,
                score=0.9,
            )
        ]


def _anonymized_with(monkeypatch, text, model):
    from services import text_anonymization as ta
    from services.phi_detection import StructuredPatternDetector

    monkeypatch.setattr(
        ta, "_detectors", lambda *_args, **_kwargs: (StructuredPatternDetector(), model)
    )
    return ta.anonymize_clinical_text(text, profile="strict")["anonymized_text"]


def test_a_score_on_the_next_line_survives_an_identifier_span_that_reaches_it(
    monkeypatch,
):
    text = "Letter filed under REF-004.\nNEWS2 score 4 prompted review."
    model = _FixedSpanModel("stanford_deidentifier", "IDENTIFIER", "004.\nNEWS2")

    output = _anonymized_with(monkeypatch, text, model)

    assert "NEWS2 score 4" in output
    assert "004" not in output


def test_a_specimen_on_the_next_line_survives_a_name_span_that_reaches_it(monkeypatch):
    text = "Sodium 139 mmol/L.\nUrine dipstick was negative."
    model = _FixedSpanModel("ner", "PERSON", "L.\nUrine")

    output = _anonymized_with(monkeypatch, text, model)

    assert "mmol/L" in output
    assert "Urine dipstick" in output


def test_a_single_letter_before_a_full_stop_carries_no_identity():
    assessment = _assess(
        "Sodium 139 mmol/L.\nUrine dipstick was negative.", "L.", "PERSON"
    )

    assert (assessment.action, assessment.rationale) == (
        "preserve",
        "single_character_span_carries_no_identity",
    )


def test_an_initial_in_a_naming_construction_is_still_removed():
    assessment = _assess("Patient R. attended the clinic.", "R.", "PERSON")

    assert assessment.action == "redact"


@pytest.mark.parametrize(
    "category",
    [
        "US_SSN",
        "PHONE_NUMBER",
        "ACCOUNT_NUMBER",
        "EMAIL_ADDRESS",
        "DEVICE_ID",
        "PERSON",
    ],
)
def test_a_bay_number_is_not_an_identifier_whatever_category_a_model_gives_it(category):
    assessment = _assess(
        "| bay 7 | stable |",
        "bay 7",
        category,
        source="gliner_multi_pii",
        contains_proper_noun=True,
    )

    assert (assessment.action, assessment.rationale) == (
        "preserve",
        "clinical_designator_grammar",
    )


def test_a_model_span_across_table_cells_is_assessed_cell_by_cell(monkeypatch):
    text = "| VIN QX7Z4411 | bay 7 | stable |"
    model = _FixedSpanModel("gliner_multi_pii", "ADDRESS", "QX7Z4411 | bay 7 | stable")

    output = _anonymized_with(monkeypatch, text, model)

    assert "bay 7" in output
    assert "stable" in output
    assert "QX7Z4411" not in output


def test_a_tight_pipe_inside_an_identifier_is_not_a_cell_boundary():
    from services import text_anonymization as ta
    from services.phi_detection import DetectedEntity

    text = "Code K42||7315||QD filed."
    start = text.index("K42")
    entity = DetectedEntity(
        entity_type="IDENTIFIER",
        start=start,
        end=start + 13,
        source="stanford_deidentifier",
    )

    assert ta._line_parts(entity, text) == [entity]


# -- durations are lengths of time, not dates --------------------------------------------


@pytest.mark.parametrize(
    "text,span",
    [
        ("Co-amoxiclav 625 mg three times daily for 7 days.", "7 days"),
        ("Co-amoxiclav 625 mg three times daily for 7 days.", "for 7 days"),
        ("Prednisolone 30 mg daily x 5 days then stop.", "5 days"),
        ("Headache over the past 3 weeks.", "3 weeks"),
        ("Continue for 10 to 14 days.", "10 to 14 days"),
        ("Paracetamol every 6 hours as required.", "6 hours"),
        ("Symptoms lasting 2 months.", "2 months"),
        ("Smoked for 20 years.", "20 years"),
    ],
)
def test_a_cued_quantity_of_time_is_a_duration(text, span):
    assessment = _assess(
        text, span, "DATE_TIME", source="ner", contains_proper_noun=False
    )

    assert (assessment.action, assessment.rationale) == (
        "preserve",
        "duration_of_time_not_a_date",
    )


def test_treatment_followup_interval_is_not_a_calendar_date():
    text = "Continue treatment and return in 3 months."
    assessment = _assess(
        text, "3 months", "DATE_TIME", source="ner", contains_proper_noun=False
    )
    assert (assessment.action, assessment.rationale) == (
        "preserve",
        "treatment_followup_interval_not_calendar_date",
    )


def test_a_follow_up_interval_after_an_event_is_still_not_the_follow_up_rule():
    """The treatment-follow-up rule itself does not cover an interval anchored to
    an event; the relative-offset rule does, and says so under its own name."""
    text = "Continue treatment and return in 3 months after discharge."
    assessment = _assess(
        text, "3 months", "DATE_TIME", source="ner", contains_proper_noun=False
    )
    assert assessment.rationale != "treatment_followup_interval_not_calendar_date"
    assert (assessment.action, assessment.rationale) == (
        "preserve",
        "relative_time_offset_not_a_date",
    )


@pytest.mark.parametrize(
    "text,span",
    [
        ("Admitted 5 days ago with chest pain.", "5 days"),
        ("Review in 5 days in clinic.", "5 days"),
        ("Discharged 3 days after admission.", "3 days"),
        ("Wound checked for 5 days after surgery.", "5 days"),
        ("Seen within 2 weeks of the referral.", "2 weeks"),
        ("5 days", "5 days"),
    ],
)
def test_an_offset_from_an_unnamed_moment_is_not_a_date(text, span):
    """Phase 2 contract: a counted offset has no calendar element in it, so it
    cannot place an event on a date, and removing it removes clinical sequencing
    for nothing. It is kept under its own rationale, not as a duration."""
    assessment = _assess(
        text, span, "DATE_TIME", source="ner", contains_proper_noun=False
    )

    assert assessment.rationale != "duration_of_time_not_a_date"
    assert (assessment.action, assessment.rationale) == (
        "preserve",
        "relative_time_offset_not_a_date",
    )


@pytest.mark.parametrize(
    "text,span",
    [
        ("Symptoms began 14 March 2024.", "14 March 2024"),
        ("Clinic on 12/03/2024 confirmed it.", "12/03/2024"),
        ("Lived there for 95 years.", "95 years"),
        ("Patient is 94 years old.", "94 years"),
    ],
)
def test_a_calendar_date_or_an_age_over_89_is_not_exempted(text, span):
    assessment = _assess(
        text, span, "DATE_TIME", source="ner", contains_proper_noun=False
    )

    assert assessment.rationale != "duration_of_time_not_a_date"
    assert assessment.action != "preserve"


def test_a_structured_date_is_never_exempted_as_a_duration():
    text = "Treated for 5 days from 2024-03-05."
    start = text.index("2024-03-05")

    assessment = assess_finding(
        text, start, start + 10, "DATE_TIME", "structured_pattern"
    )

    assert assessment.action == "redact"


def test_a_duration_survives_and_the_date_beside_it_does_not(monkeypatch):
    text = "Amoxicillin 500 mg for 5 days, started 2024-03-05 after admission on 2024-03-04."
    model = _FixedSpanModel("ner", "DATE_TIME", "5 days")

    output = _anonymized_with(monkeypatch, text, model)

    assert "for 5 days" in output
    assert "2024-03-05" not in output
    assert "2024-03-04" not in output


def test_a_relative_offset_span_is_kept_and_the_calendar_date_beside_it_is_not(
    monkeypatch,
):
    text = "Admitted 5 days ago with chest pain; seen on 2024-03-05."
    model = _FixedSpanModel("ner", "DATE_TIME", "5 days")

    output = _anonymized_with(monkeypatch, text, model)

    assert "5 days ago" in output
    assert "2024-03-05" not in output


# -- workforce groups, headings and documentation labels ----------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "Physiotherapy team",
        "Occupational therapy team",
        "Cardiology registrars",
        "Palliative care team",
        "Dietetics staff",
        "Stroke nurses",
        "Radiotherapy team",
    ],
)
def test_a_staff_group_named_by_its_service_is_not_an_organisation(phrase):
    text = f"{phrase} reviewed the plan with her."

    assessment = _assess(
        text,
        phrase,
        "ORGANIZATION",
        source="gliner_multi_pii",
        contains_proper_noun=True,
    )

    assert assessment.action == "preserve"


@pytest.mark.parametrize(
    "phrase",
    [
        "Brightwater Physiotherapy team",
        "Castellowe cardiology team",
        "Ravensholt registrars",
    ],
)
def test_a_named_body_in_front_of_a_service_group_is_still_removed(phrase):
    text = f"{phrase} reviewed the plan with her."

    assessment = _assess(
        text,
        phrase,
        "ORGANIZATION",
        source="gliner_multi_pii",
        contains_proper_noun=True,
    )

    assert assessment.action == "redact"


@pytest.mark.parametrize(
    "text,span,category,remainder",
    [
        (
            "Referral from FACILITY_003 filed.",
            "Referral from FACILITY_003",
            "PERSON",
            "FACILITY_003",
        ),
        (
            "Referral from Dr Ashvelin Thackerwild today.",
            "Referral from Dr Ashvelin Thackerwild",
            "PERSON",
            "Dr Ashvelin Thackerwild",
        ),
        (
            "Referral from Brightwater Clinic arrived.",
            "Referral from Brightwater Clinic",
            "ORGANIZATION",
            "Brightwater Clinic",
        ),
        ("ALERT QX7Z4411 FLAGGED", "ALERT QX7Z4411 FLAGGED", "PERSON", "QX7Z4411"),
        (
            "Pt  ref :   LK4471290  ;  noted",
            "Pt  ref :   LK4471290",
            "DRIVER_LICENSE",
            "LK4471290",
        ),
        (
            "contact/id: QX7Z4411 / pharmacy",
            "QX7Z4411 / pharmacy",
            "ADDRESS",
            "QX7Z4411",
        ),
        ("Castellowe BP 118/76 HR 72.", "Castellowe BP", "ORGANIZATION", "Castellowe"),
    ],
)
def test_heading_and_label_words_are_trimmed_off_a_span(
    text, span, category, remainder
):
    start = text.index(span)

    new_start, new_end = trim_clinical_edges(text, start, start + len(span), category)

    assert text[new_start:new_end] == remainder


@pytest.mark.parametrize(
    "text,span,category",
    [
        (
            "Referred to Community Health Partners today.",
            "Community Health Partners",
            "ORGANIZATION",
        ),
        ("Lives on Pharmacy Lane, Wrenhollow.", "Pharmacy Lane, Wrenhollow", "ADDRESS"),
        ("ALERT CASTELLOWE FLAGGED", "ALERT CASTELLOWE FLAGGED", "PERSON"),
        ("Castellowe Pending review.", "Castellowe Pending", "PERSON"),
        ("Sister Castellowe reviewed.", "Sister Castellowe", "PERSON"),
    ],
)
def test_label_words_that_may_belong_to_the_name_are_not_trimmed(text, span, category):
    start = text.index(span)

    assert trim_clinical_edges(text, start, start + len(span), category) == (
        start,
        start + len(span),
    )


def test_a_heading_before_a_surrogate_is_kept_and_a_named_person_after_it_is_not(
    monkeypatch,
):
    kept = _anonymized_with(
        monkeypatch,
        "Referral from FACILITY_003 filed under RECORD_005.",
        _FixedSpanModel("gliner_multi_pii", "PERSON", "Referral from FACILITY_003"),
    )
    removed = _anonymized_with(
        monkeypatch,
        "Referral from Ashvelin Thackerwild was filed.",
        _FixedSpanModel(
            "gliner_multi_pii", "PERSON", "Referral from Ashvelin Thackerwild"
        ),
    )

    assert "Referral from" in kept
    assert "Referral from" in removed
    assert "Thackerwild" not in removed


@pytest.mark.parametrize(
    "text,span,category,source",
    [
        ("WRISTBAND TAG: QX7Z4411", "WRISTBAND TAG", "PERSON", "strict_proper_noun"),
        ("WRISTBAND TAG: QX7Z4411", "WRISTBAND", "PERSON", "ner"),
        ("ALERT QX7Z4411 FLAGGED", "FLAGGED", "PERSON", "strict_proper_noun"),
        ("ALERT QX7Z4411 FLAGGED", "ALERT", "ORGANIZATION", "ner"),
        (
            'Label reads "QX7Z4411" on the band.',
            "Label",
            "PERSON",
            "strict_proper_noun",
        ),
        ("Pt  ref :   QX7Z4411  ;  noted", "Pt", "PERSON", "strict_proper_noun"),
        ("contact/id: QX7Z4411 / pharmacy", "id", "PERSON", "strict_proper_noun"),
        (
            "Confirmed on callback: QX7Z4411",
            "Confirmed",
            "PERSON",
            "strict_proper_noun",
        ),
    ],
)
def test_documentation_words_are_not_names(text, span, category, source):
    assessment = _assess(text, span, category, source=source, contains_proper_noun=True)

    assert assessment.action in ("preserve", "review")


@pytest.mark.parametrize(
    "text,span",
    [
        ("Castellowe: follow-up arranged.", "Castellowe"),
        ("reviewed with castellowe dravenhurst today.", "castellowe dravenhurst"),
        ("NOTE: CASTELLOWE REVIEWED", "CASTELLOWE"),
        ("Dr Field reviewed the wound.", "Field"),
        ("Mark attended with his sister.", "Mark"),
    ],
)
def test_a_name_in_a_label_like_place_is_still_removed(text, span):
    assessment = _assess(
        text, span, "PERSON", source="strict_proper_noun", contains_proper_noun=True
    )

    assert assessment.action == "redact"


def test_a_list_number_is_a_list_marker_not_a_postcode():
    text = "2) Castellowe - awaiting callback"

    assessment = _assess(
        text, "2", "POSTAL_CODE", source="gliner_multi_pii", contains_proper_noun=False
    )

    assert (assessment.action, assessment.rationale) == (
        "preserve",
        "list_marker_grammar",
    )


def test_a_number_inside_an_address_is_not_a_list_marker():
    text = "Flat 2, Castle Court, Wrenhollow"

    assessment = _assess(
        text, "2", "POSTAL_CODE", source="gliner_multi_pii", contains_proper_noun=False
    )

    assert assessment.rationale != "list_marker_grammar"


def test_a_name_broken_across_two_lines_is_still_removed_from_both(monkeypatch):
    text = "Seen by Ashvelin\nThackerwild on the ward."
    model = _FixedSpanModel("gliner_multi_pii", "PERSON", "Ashvelin\nThackerwild")

    output = _anonymized_with(monkeypatch, text, model)

    assert "Ashvelin" not in output
    assert "Thackerwild" not in output
