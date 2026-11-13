"""Reading a line by its grammar, and the identifiers that must not pass.

The raster classifier answers about a span. Asked about a single OCR word it
cannot answer at all - "96" is a number, "g/L." is letters and a stop - and
asked that way it called forty of forty-two boxes on a clinic letter
identifying and filled them, taking the diagnosis, the code, the analyte, the
value, the unit and the dose. Asked about the line the printer put there it
does much better, and what it still cannot place is a reading: it returned
"uncertain" for ``Result: haemoglobin 96 g/L.`` and for ``FEV1 2.4 L``.

The recognizer under test rescues exactly those, on the evidence of their
shape. Every test is paired: a structure that must be recognized, and an
identifier that must not be, including identifiers placed deliberately beside
clinical words.
"""

from __future__ import annotations

import pytest
from services.clinical_measurement import (
    EVIDENCE_ACUITY,
    EVIDENCE_DIAGNOSIS_CODE,
    EVIDENCE_DOSE,
    EVIDENCE_QUANTITY,
    EVIDENCE_SCORE,
    carries_structured_identifier,
    line_reads_as_measurement,
    measurement_evidence,
)

# -- the structures the brief names --------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "FEV1 2.4 L",
        "SpO2 98%",
        "37 degrees Celsius",
        "37 °C",
        "5 umol/L",
        "5 µmol/L",
        "bilirubin 28 umol/L",
        "Result: haemoglobin 96 g/L.",
        "Plan: ferrous fumarate 210 mg daily.",
        "latanoprost 50 micrograms/mL",
        "salbutamol 100 micrograms",
        "blood pressure 128/82 mmHg",
        "GAD-7 score 12",
        "PHQ-9 score of 4",
        "Diagnosis: D64.9 - iron deficiency anaemia.",
        "Impression: J45.909",
        "visual acuity 6/12 Snellen metres",
    ],
)
def test_a_clinical_reading_is_recognised(line):
    assert line_reads_as_measurement(line), measurement_evidence(line)


@pytest.mark.parametrize(
    "line,kind",
    [
        ("haemoglobin 96 g/L", EVIDENCE_QUANTITY),
        ("Diagnosis: D64.9", EVIDENCE_DIAGNOSIS_CODE),
        ("GAD-7 score 12", EVIDENCE_SCORE),
        ("ferrous fumarate 210 mg", EVIDENCE_DOSE),
        ("acuity 6/12 Snellen metres", EVIDENCE_ACUITY),
    ],
)
def test_the_evidence_is_named(line, kind):
    assert kind in measurement_evidence(line)


# -- the identifiers that must not pass -----------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "Record: RT4420518",
        "MRN 4420518",
        "Account: ACR520563",
        "Telephone: 555-0121",
        "Email: isolde.w@example.invalid",
        "Accession ACC-2024-00123",
        "Patient: Isolde Warrenbeck",
        "Reviewed by: Dr Nadia Oyelowo",
        "Address: 18 Harrowgate Lane, Pentlemoor",
        "I.W. 47",
        "Dr N.O.",
        "NHS 943 476 5919",
    ],
)
def test_an_identifier_is_not_a_reading(line):
    assert not line_reads_as_measurement(line)


@pytest.mark.parametrize(
    "line",
    [
        "haemoglobin 96 g/L RT4420518",
        "bilirubin 28 umol/L for ACR520563",
        "Diagnosis D64.9 telephone 555-0121",
        "FEV1 2.4 L, contact isolde.w@example.invalid",
    ],
)
def test_an_identifier_beside_a_clinical_word_still_blocks_the_line(line):
    """The counterexample the brief asks for by name.

    A quantity in the line does not make the rest of it safe. Grammar is
    evidence about the reading; it is not permission for what sits beside it.
    """
    assert measurement_evidence(line), "the reading grammar is present"
    assert carries_structured_identifier(line)
    assert not line_reads_as_measurement(line)


def test_a_bare_number_is_not_a_reading():
    """A number with nothing to give it meaning is not a measurement, which
    is what keeps "I.W. 47" and a stray page number out."""
    assert not line_reads_as_measurement("47")
    assert not line_reads_as_measurement("Patient 47")


def test_a_bare_unit_is_not_a_reading():
    assert not line_reads_as_measurement("mg")
    assert not line_reads_as_measurement("umol/L")


@pytest.mark.parametrize("line", ["04/03", "12/05", "Date 04/03/1971", "ratio 1/2"])
def test_a_bare_ratio_is_not_a_reading(line):
    """A ratio with no unit is a date as often as it is a measurement, so the
    unit is required and one word may stand between them - enough for
    "6/12 Snellen metres", not enough for a date."""
    assert not line_reads_as_measurement(line)


# -- the diagnosis-code exception is narrow -------------------------------------------------


def test_a_diagnosis_code_is_not_treated_as_an_identifier():
    """ "J45.909" reaches the deterministic scan as a letter and five digits
    and is reported; "D64.9" is not. That is an accident of length, and only
    a span that is wholly a well-formed code is set aside."""
    assert carries_structured_identifier("J45.909") is False
    assert not carries_structured_identifier("Impression: J45.909")


def test_a_record_number_is_not_rescued_by_that_exception():
    assert carries_structured_identifier("RT4420518")
    assert carries_structured_identifier("J45.909 MRN RT4420518")


# -- the ordering that makes it safe --------------------------------------------------------


def test_the_recognizer_does_not_classify_anything():
    """It answers about shape only. The caller asks it solely about a line
    the model already declined to call PHI, so a line the model did call PHI
    is never reconsidered - that ordering, not this function, is what stops a
    name being rescued by a number next to it."""
    assert line_reads_as_measurement("Isolde Warrenbeck 96 g/L") is True
    # ...which is exactly why the caller never asks about a PHI line. The
    # production path is covered in test_ocr_evidence.


# -- narrative that carries no quantity -----------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "Procedure: spirometry arranged.",
        "Procedure: abdominal ultrasound arranged.",
        "Procedure: Ultrasound arranged.",
        "Note: No evidence of active bleeding.",
        "Note: Reversibility not demonstrated.",
        "Findings: no mass seen",
        "Impression: consistent with anaemia of unspecified cause",
    ],
)
def test_clinical_narrative_is_recognised(line):
    """Some of what a record says carries no quantity at all. The procedure
    and the negation are clinical facts, and the second states one by what it
    rules out."""
    from services.clinical_measurement import line_reads_as_clinical_narrative

    assert line_reads_as_clinical_narrative(line)


@pytest.mark.parametrize(
    "line",
    [
        "Patient: Isolde Warrenbeck",
        "Record: RT4420518",
        "Telephone: 555-0121",
        "Address: 18 Harrowgate Lane",
        "Reviewed by: Dr Nadia Oyelowo",
    ],
)
def test_an_administrative_head_is_never_narrative(line):
    """ "Patient:", "Record:" and "Telephone:" are followed by who and which.
    Nothing under those heads is narrative worth keeping, so they cannot be
    read this way at all."""
    from services.clinical_measurement import line_reads_as_clinical_narrative

    assert not line_reads_as_clinical_narrative(line)


@pytest.mark.parametrize(
    "line",
    [
        "Procedure: performed by Dr Achebe",
        "Note: reviewed with Achebe",
        "Findings: discussed with Marisol Vanterpool",
        "Note: contact RT4420518",
        "Procedure: booked, ring 555-0121",
    ],
)
def test_a_name_or_identifier_under_a_clinical_head_declines(line):
    """The counterexamples. A clinical heading is not a licence for whatever
    follows it: an honorific, a word capitalised mid-sentence, or anything
    the deterministic scan calls an identifier ends it."""
    from services.clinical_measurement import line_reads_as_clinical_narrative

    assert not line_reads_as_clinical_narrative(line)


def test_a_headless_line_is_not_narrative():
    from services.clinical_measurement import line_reads_as_clinical_narrative

    assert not line_reads_as_clinical_narrative("spirometry arranged")
    assert not line_reads_as_clinical_narrative("Procedure:")
