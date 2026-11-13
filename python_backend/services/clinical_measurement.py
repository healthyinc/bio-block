"""Whether a line of text is a reading, by its grammar rather than its words.

The raster classifier answers PHI / clinical / uncertain, and on a clinic
letter read line by line it answered "uncertain" for most of what matters:
``Diagnosis: D64.9 - iron deficiency anaemia.``, ``Result: haemoglobin 96
g/L.``, ``FEV1 2.4 L``, ``37 degrees Celsius``, ``GAD-7 score 12``. Uncertain
means filled, and filling those empties the document of everything a clinician
would open it for.

Extending a word list would not fix it, and is the wrong shape of answer: the
next assay is always missing, and a list of permitted tokens is a list of ways
in. What these lines have in common is not vocabulary but **grammar** - a
quantity written next to the thing that gives it meaning:

* a number joined to a unit - ``96 g/L``, ``210 mg``, ``98%``, ``2.4 L``;
* a diagnosis code in its own well-formed shape - ``D64.9``, ``J45.909``;
* a score named and then given - ``GAD-7 score 12``.

So this recognizes the shape. It is deliberately not a classifier and never
promotes anything on its own: the caller asks it only about a line the model
already declined to call PHI, and a line the model *did* call PHI is never
reconsidered here. That ordering is the safety property - this can rescue a
reading the model was unsure of, and it can never overrule the model saying
"this names somebody".

The second guard is exclusion: a line carrying anything the deterministic
scan calls an identifier is not a reading, whatever else is in it. A record
number beside the word "haemoglobin" is still a record number.
"""

from __future__ import annotations

import re
from typing import List, Tuple

#: Units written as a word, a symbol, or a quotient. Deliberately a shape -
#: letters, optionally over more letters - rather than an enumeration, so an
#: unfamiliar unit still reads as one.
_UNIT = (
    r"(?:%|°[CF]|"
    r"(?:m|µ|u|n|k|d|c)?(?:g|L|l|m|mol|mols|Hz|Pa|Gy|Sv|J|W|V|A|K|s|min|h)"
    r"(?:/(?:m|µ|u|n|k|d|c)?(?:g|L|l|m|mol|s|min|h|day|dL|kg|m2|m²))?|"
    r"mmHg|cmH2O|bpm|kPa|IU|iu|units?|micrograms?(?:/m[lL])?|milligrams?|"
    r"millilitres?|milliliters?|kilograms?|centimetres?|centimeters?|"
    r"metres?|meters?|inches|degrees?|celsius|fahrenheit|kelvin"
    r")"
)
#: A number as a measurement is written: optional sign, digits, optional
#: decimal, and nothing that turns it into a serial number.
_NUMBER = r"[+-]?\d{1,4}(?:\.\d{1,4})?"

#: A quantity: the number and its unit, adjacent or separated by one space.
#: This is the spatial relationship the evidence rests on - a number on its
#: own is not a reading, and a unit on its own is not either.
QUANTITY = re.compile(rf"(?<![\w/-]){_NUMBER}\s?{_UNIT}(?![\w/])", re.IGNORECASE)

#: A ratio carrying a unit, as a blood pressure or a visual acuity is
#: written. The unit is required, and one word may stand between them so
#: "6/12 Snellen metres" reads while a bare "04/03" - which is a date as
#: often as it is anything - does not.
ACUITY = re.compile(
    rf"(?<![\w/-])\d{{1,3}}\s?/\s?\d{{1,3}}\s+(?:[A-Za-z]+\s+)?{_UNIT}(?![\w/])",
    re.IGNORECASE,
)

#: ICD-10 shape: a letter, two digits, optionally a point and up to four more.
DIAGNOSIS_CODE = re.compile(r"(?<![\w.-])[A-TV-Z]\d{2}(?:\.\d{1,4})?(?![\w-])")

#: A named score followed by its value: "GAD-7 score 12", "PHQ-9 of 4". The
#: stem is short and the value small, which is what keeps an accession number
#: like ACC-2024-00123 out of it.
SCORE = re.compile(
    r"(?<![\w-])[A-Za-z]{2,6}-?\d{0,2}\s*(?:score|scale|index)\s*(?:of\s*)?:?\s*\d{1,3}"
    r"(?![\w-])",
    re.IGNORECASE,
)

#: A quantity whose unit is attached to the following word, as a dose is
#: written: "210 mg daily", "100 micrograms".
DOSE = re.compile(
    rf"(?<![\w-]){_NUMBER}\s?(?:mg|mcg|µg|ug|g|kg|mL|ml|L|units?|iu|IU|"
    r"micrograms?|milligrams?)(?![\w/])",
    re.IGNORECASE,
)

EVIDENCE_QUANTITY = "quantity_with_unit"
EVIDENCE_DIAGNOSIS_CODE = "diagnosis_code_grammar"
EVIDENCE_SCORE = "named_score_with_value"
EVIDENCE_DOSE = "dose_with_unit"
EVIDENCE_ACUITY = "ratio_reading"


def measurement_evidence(line: str) -> Tuple[str, ...]:
    """Every kind of reading grammar this line exhibits, in a stable order."""
    text = line or ""
    found: List[str] = []
    if QUANTITY.search(text):
        found.append(EVIDENCE_QUANTITY)
    if DIAGNOSIS_CODE.search(text):
        found.append(EVIDENCE_DIAGNOSIS_CODE)
    if SCORE.search(text):
        found.append(EVIDENCE_SCORE)
    if DOSE.search(text):
        found.append(EVIDENCE_DOSE)
    if ACUITY.search(text):
        found.append(EVIDENCE_ACUITY)
    return tuple(found)


def carries_structured_identifier(line: str) -> bool:
    """Whether the deterministic scan finds an identifier anywhere in the line.

    The exclusion that stops grammar becoming a way in. "haemoglobin 96 g/L"
    is a reading; "haemoglobin 96 g/L RT4420518" contains a record number, and
    the presence of a quantity does not make the rest of the line safe.
    """
    from services.residual_identifier_scan import scan as scan_structured

    text = line or ""
    try:
        hits = scan_structured(text)
    except Exception:
        # A scan that cannot answer is not permission.
        return True
    for hit in hits:
        start = getattr(hit, "start", None)
        end = getattr(hit, "end", None)
        if start is None or end is None:
            return True
        span = text[start:end]
        # A diagnosis code is identifier-shaped and is not an identifier.
        # "J45.909" reaches the scan as a letter and five digits and is
        # reported; "D64.9" is not, which is an accident of length rather
        # than a distinction worth honouring. Only a span that is wholly a
        # well-formed code is set aside, so "RT4420518" beside a reading is
        # untouched by this.
        if DIAGNOSIS_CODE.fullmatch(span.strip()):
            continue
        return True
    return False


def line_reads_as_measurement(line: str) -> bool:
    """Whether this line is a clinical reading on the evidence of its shape.

    Both halves are required: some reading grammar has to be present, and
    nothing the deterministic scan calls an identifier may be. Neither is
    sufficient alone - a bare number is not a reading, and the absence of an
    identifier says nothing at all.
    """
    text = (line or "").strip()
    if not text:
        return False
    if not measurement_evidence(text):
        return False
    return not carries_structured_identifier(text)


#: A word that is capitalised where a sentence would not capitalise it. The
#: only signal separating "abdominal ultrasound arranged" from "reviewed with
#: Achebe" once the model has declined to place either.
_MID_SENTENCE_CAPITAL = re.compile(r"(?<=\S\s)[A-Z][a-z]{2,}")
_HONORIFIC = re.compile(
    r"\b(?:dr|mr|mrs|ms|miss|prof|professor|sr|jr)\b\.?", re.IGNORECASE
)


def line_reads_as_clinical_narrative(line: str) -> bool:
    """Whether this line is a clinical note rather than a party to the record.

    Some of what a record says carries no quantity at all. "Procedure:
    spirometry arranged." and "Note: No evidence of active bleeding." are the
    finding and the negation, and filling them empties the document as surely
    as losing the value would - the second one is a clinical fact stated by
    what it rules out.

    Four things are required, and the first is the strongest:

    * the line is headed by a field that introduces narrative - "Procedure",
      "Findings", "Note" - and not by one that introduces a party.
      "Patient:", "Record:" and "Telephone:" can never be read this way;
    * nothing the deterministic scan calls an identifier is in it;
    * no honorific, which is how a clinician is named;
    * no word capitalised mid-sentence, which is how anyone else is. That is
      a weak signal on its own and it is not on its own: it applies only to a
      line already headed clinically, already free of identifiers, and
      already declined by the model as PHI - "Procedure: performed by Dr
      Achebe" never reaches here, because the model calls it PHI and the
      caller does not ask.
    """
    from services.clinical_vocabulary import is_clinical_narrative_head

    text = (line or "").strip()
    if ":" not in text:
        return False
    head, _, body = text.partition(":")
    if not is_clinical_narrative_head(head.strip()):
        return False
    body = body.strip()
    if not body:
        return False
    if _HONORIFIC.search(body):
        return False
    if _MID_SENTENCE_CAPITAL.search(body):
        return False
    return not carries_structured_identifier(text)
