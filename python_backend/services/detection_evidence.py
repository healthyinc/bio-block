"""Layered evidence model for detection findings.

Phase 10 protected clinical content with a closed vocabulary: if a word was in
the list it survived, otherwise it was redacted. That preserves the words
somebody thought of and destroys every unseen medication, condition or
eponym - and worse, it protects a term unconditionally once listed, so a
patient genuinely surnamed Wilson could be shielded by a rule meant for
Wilson's disease.

This module replaces "is it in the list?" with a layered decision in which the
vocabulary is one evidence source rather than the verdict:

1. **Exact high-confidence structured PHI always wins.** A regex that matched
   an SSN, an email or an MRN is not overridden by anything.
2. **Multiple agreeing detectors normally win.** Two independent models
   proposing the same span is strong evidence of an identifier.
3. **A clinical reading may block a weak, heuristic-only redaction.** This is
   where the vocabulary and the surrounding grammar are consulted.
4. **A clinical term never overrides strong contextual evidence that it is
   being used as a person, facility or location.** "Dr Parkinson" is a person
   even though "Parkinson" is a recorded eponym.
5. **Unknown and ambiguous terms escalate.** Neither deleted on suspicion nor
   preserved on hope: they go to a human.

Evidence strength, not word membership, decides. Nothing here inspects a
candidate against any expected-value list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from services import numeric_context
from services import structured_identifier_spec as identifier_spec
from services.clinical_preservation import (
    STRUCTURAL_REASONS,
    assess_clinical_span,
    clinical_service_region,
    modality_before,
)
from services.clinical_vocabulary import (
    CLINICAL_DOMAIN_NOUNS,
    CLINICAL_HEAD_NOUNS,
    CLINICAL_MEASUREMENT_TERMS,
    CLINICAL_SERVICE_TERMS,
    NAMED_SCALE_MODIFIERS,
    contains_distinctive_clinical_token,
    is_bare_role_noun_phrase,
    is_clinical_eponym_use,
    is_clinical_field_label,
    is_clinical_finding_term,
    is_clinical_product_term,
    is_identifier_label_text,
    is_label_word,
    is_non_name_label,
    is_race_ethnicity_category,
    is_staff_role_phrase,
    is_unconditional_clinical_term,
    is_unit_eponym,
    is_unit_head_word,
)
from services.phi_detection import (
    SOURCE_AGE_RULE,
    SOURCE_CONTEXT_RULE,
    SOURCE_NER,
    SOURCE_REPEAT_PROPAGATION,
    SOURCE_RESIDUAL_IDENTIFIER_SCAN,
    SOURCE_STRICT_PROPER_NOUN,
    SOURCE_STRUCTURED_PATTERN,
    is_unique_identifier_value,
)

#: Bumped when the rules move, not when the file is edited. A frozen
#: configuration is only meaningful if this string moves with the behaviour.
#:
#: v2 and v3 were both written from what the real chain did to the calibration
#: partition: a diagnosis is never an identifier; a single character carries no
#: identity; a span the tagger finds no proper noun in is not a name; Layer 1
#: belongs to the regex rather than to a model's choice of category label; and
#: a time reference with no calendar content is not a date element.
#:
#: v5 reads readings, bed designators and clinical scores before Layer 1,
#: whatever category a model gave them, and trims them off a model span's
#: edges; a model's phone-number claim needs seven digits.
#:
#: v5.1: a model's structured claim that fails its own syntax check fails
#: closed - a code-shaped span is removed as OTHER_UNIQUE_ID, other digits
#: go to review - instead of reaching the name-only proper-noun exemption.
#: Bed numbers and clinical scores are protected whatever category a
#: model gives them.
#:
#: v5.2: a quantity of time governed by a duration cue ("for 5 days") is a
#: length of time, not a date element, and is kept; anchored or bare
#: quantities of time keep today's handling.
#:
#: v5.3: heading and label words at a model span's edges are trimmed when a
#: boundary separates them from the rest - label punctuation, a line break,
#: a heading preposition or an identifier beside them - as is a vital-sign
#: label cut off from its value; a list number is a list marker.
#:
#: v6 is written from the attributed failures of the Phase 15 final corpus,
#: where four rules between them destroyed 1,164 clinical tokens and one
#: released a biometric identifier twice:
#:
#: * agreement now means a *compatible identifier claim*. A detector reading a
#:   span as a medical condition was counted as a second vote that the span
#:   was a name, so the prediction that argued for keeping the text became the
#:   corroboration for removing it;
#: * a clinical reading is consulted for every model claim, not only for the
#:   categories a name is confused with. spaCy labelled "HbA1c 58" and
#:   "ramipril 5" DATE_TIME, and the category alone decided. A span carrying
#:   calendar content is still a date element whatever else is in it;
#: * a word, a value and a unit reads as a laboratory measurement without the
#:   analyte having to be in any list, which is what stops the vocabulary from
#:   deciding which results survive;
#: * the denominator of a compound unit - the "min" of "L/min", the "mol" of
#:   "mmol/mol" - is part of a measurement, not a name; and
#: * a "medical condition" prediction no longer shields a code written inside
#:   it. The span keeps its clinical words and the identifier is removed.
EVIDENCE_MODEL_VERSION = "detection-evidence-v6"

# -- evidence strength ------------------------------------------------------

EVIDENCE_DETERMINISTIC = "deterministic"  # a structured pattern or age rule
EVIDENCE_MODEL = "model"  # a trained model proposed it
EVIDENCE_CONTEXT = "context_rule"  # an explicit contextual construction
EVIDENCE_HEURISTIC = "heuristic"  # capitalisation shape only

#: Sources whose findings are exact matches on identifier syntax. The repeat
#: source only ever re-finds a value a structured match already confirmed, and
#: the residual scan is the validator's own structured rule.
_DETERMINISTIC_SOURCES = frozenset(
    {
        SOURCE_STRUCTURED_PATTERN,
        SOURCE_AGE_RULE,
        SOURCE_REPEAT_PROPAGATION,
        SOURCE_RESIDUAL_IDENTIFIER_SCAN,
    }
)
_MODEL_SOURCES = frozenset({SOURCE_NER, "stanford_deidentifier", "gliner_multi_pii"})
#: Sources that find a name by its capital letters.
_CASE_DRIVEN_SOURCES = frozenset({SOURCE_NER, SOURCE_STRICT_PROPER_NOUN})

#: Words a title keeps in lower case between its capitalised ones.
_TITLE_CONNECTORS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


def line_carries_no_case_evidence(line: str) -> bool:
    """Whether every word of a line is capitalised, so case says nothing.

    An all-capitals line of two words or more, or a title-case line of three or
    more (small connectors aside), is a banner or heading: its capitals come
    from the layout, not from the words being names. A two-word title-case line
    is how a name is written, so it is not treated this way, and neither is a
    "Label: value" line, whose value is not a heading.
    """
    # A line with a colon is a field, one with a digit is data, and one with a
    # naming title ("Dr 3 L Harrow") is about a person: none is a heading.
    if (
        ":" in line
        or any(character.isdigit() for character in line)
        or _NAMING_TITLE_IN_LINE.search(line)
    ):
        return False
    # A hyphenated or apostrophised word is one word: "Nakashima-Holt".
    words = re.findall(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", line)
    letters = [character for character in line if character.isalpha()]
    if not letters or not words:
        return False
    # How the line ends says it is a heading: a kind of document, or a section.
    # Without that a short capitalised line is how a name is written.
    last = words[-1].casefold()
    labelled = last in _DOCUMENT_NOUNS or last in _SECTION_NOUNS
    if len(words) >= 2 and not any(character.islower() for character in letters):
        # Single letters other than "A" and "I" are initials or stain and
        # annotation marks ("H&E"), which put a name beside them.
        return all(len(word) >= 2 or word in ("A", "I") for word in words) and (
            len(words) >= 3 or labelled
        )
    content = [word for word in words if word.lower() not in _TITLE_CONNECTORS]
    return (
        len(words) >= 3
        and len(content) >= 2
        and all(word[0].isupper() for word in content)
        and all(len(word) >= 3 or word.lower() in _TITLE_CONNECTORS for word in words)
        # "Chart for Jordan Fictional" is a title with a person in it; only a
        # title that ends in a kind of document or section is a heading.
        and labelled
    )


_SERVICE_NOUNS = frozenset(
    {
        "clinic",
        "clinics",
        "department",
        "dept",
        "ward",
        "unit",
        "service",
        "services",
        "team",
        "clinic",
        "laboratory",
        "lab",
        "theatre",
    }
)
#: "Bed 14, Cubicle 3, Ward 5": one or more word-number pairs.
_DESIGNATOR_RUN = re.compile(
    r"(?:[^\W\d_]+\.?[ \t]+\d{1,4}[A-Za-z]?(?:[ \t]*[,;/&-][ \t]*|[ \t]+and[ \t]+)?)+"
)
#: "Clinic 2 Room 5": places written one after another with only a space between.
#: Without a separator the words have to be places themselves, or "Alice 55 Bob 60"
#: would read as a list of rooms.
_PLACE_WORDS = (
    r"(?:bay|bed|room|rm|ward|clinic|cubicle|theatre|theater|floor|level|suite|"
    r"unit|pod|cot|chair|trolley|resus|area|zone|block|wing|building|bldg|"
    r"consulting|treatment|side)"
)
_DESIGNATOR_WORD_RUN = re.compile(
    rf"(?:{_PLACE_WORDS}\.?[ \t]+\d{{1,4}}[A-Za-z]?(?:[ \t]*[,;/&-][ \t]*|[ \t]+and[ \t]+|[ \t]+)?)+",
    re.IGNORECASE,
)


_DOCUMENT_NOUNS = frozenset(
    {
        "form",
        "forms",
        "report",
        "reports",
        "summary",
        "summaries",
        "letter",
        "letters",
        "note",
        "notes",
        "record",
        "records",
        "sheet",
        "chart",
        "referral",
        "certificate",
        "statement",
        "review",
        "assessment",
        "history",
        "plan",
        "results",
        "protocol",
        "consent",
        "request",
        "order",
        "prescription",
        "handover",
        "bundle",
        "addendum",
        "narrative",
        "invoice",
        "checklist",
        "checklists",
        "slip",
        "worksheet",
        "questionnaire",
        "leaflet",
        "agenda",
        "schedule",
        "timetable",
        "log",
        "diary",
        "overview",
        "synopsis",
    }
)


def _ends_in_document_noun(span: str) -> bool:
    words = span.split()
    return (
        2 <= len(words) <= 5 and words[-1].strip(".,;:()").casefold() in _DOCUMENT_NOUNS
    )


_PAGE_LABELS = frozenset({"page", "pg", "pp", "p"})
_PAGE_NUMBER_AFTER = re.compile(
    r"[ \t]*\.?[ \t]*\d{1,4}(?:[ \t]*(?:of|/)[ \t]*\d{1,4})?\b"
)
_SERVICE_NOUN_AFTER = re.compile(
    r"^[ \t]+(?:" + "|".join(sorted(_SERVICE_NOUNS)) + r")\b", re.IGNORECASE
)


def _service_unit_phrase(text: str, start: int, end: int) -> bool:
    """A span that is, or is the subject words of, a clinic, unit or service."""
    words = text[start:end].split()
    if 2 <= len(words) <= 3 and words[-1].strip(".,;:()").casefold() in _SERVICE_NOUNS:
        return True
    return 1 <= len(words) <= 2 and bool(
        _SERVICE_NOUN_AFTER.match(text[end : end + 24])
    )


_TITLE_CASE_WORD = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*")


def _is_whole_line_title_case(text: str, start: int, end: int) -> bool:
    """The span is a whole line of two to six capitalised words, with no colon or digit."""
    line = _enclosing_line(text, start, end).strip()
    if text[start:end].strip() != line or ":" in line or any(c.isdigit() for c in line):
        return False
    words = _TITLE_CASE_WORD.findall(line)
    content = [w for w in words if w.casefold() not in _TITLE_CONNECTORS]
    return (
        2 <= len(words) <= 6
        and len(content) >= 2
        and all(word[0].isupper() for word in content)
    )


def _is_clinical_domain_noun_opening(text: str, start: int, end: int) -> bool:
    """One word of the closed clinical-domain list that opens a sentence or a line.

    The next word has to be lower case (or a colon follow it): a capitalised
    word after it makes the pair a possible name, and that is not excused.
    """
    word = text[start:end].strip()
    if (
        not re.fullmatch(r"[^\W\d_]+", word)
        or word.casefold() not in CLINICAL_DOMAIN_NOUNS
        or not word[0].isupper()
    ):
        return False
    if start > 0 and not _OPENING.search(text[max(0, start - 8) : start]):
        # A text layer or an OCR read often loses the space after a full stop:
        # "specimen.Mobility". The stop directly after a lower-case letter still
        # ends a sentence; one after a capital or a digit ("Dr.Smith", "4.2") does not.
        if not re.search(r"[a-z][.!?]$", text[max(0, start - 2) : start]):
            return False
    return bool(re.match(r"[ \t]*:|[ \t]+[a-z]", text[end : end + 40]))


#: Single capital letters written one at a time: "G M C", "N.H.S.", "C T".
_SPELLED_LETTERS = re.compile(r"(?:[A-Z]\.?[ \t]+){2,5}[A-Z]\.?|(?:[A-Z]\.){3,6}")
_OPENING = re.compile(r"(?:^|[.!?][\"')\]]?[ \t]+|\n[ \t]*)$")


def capitals_only_at_openings(text: str, start: int, end: int) -> bool:
    """Whether every capitalised word in the span opens a line or a sentence.

    Where a capital is the only reason a word looks like a name and that
    capital is explained by position, case carries no evidence. A capital in
    the middle of a sentence is never explained this way.
    """
    span = text[start:end]
    seen = False
    for match in re.finditer(r"[^\W\d_]+", span):
        word = match.group(0)
        if not word[0].isupper():
            continue
        if len(word) > 1 and word.isupper():
            return False
        seen = True
        if not _OPENING.search(
            text[max(0, start + match.start() - 8) : start + match.start()]
        ):
            return False
    return seen


_NAMING_TITLE_IN_LINE = re.compile(
    r"\b(?:dr|mr|mrs|ms|miss|mx|prof|professor|sir|dame)\b\.?", re.IGNORECASE
)
#: Words that end a section heading ("Past Medical History", "Review Of Systems").
_SECTION_NOUNS = frozenset(
    {
        "history",
        "examination",
        "course",
        "medications",
        "allergies",
        "diagnoses",
        "diagnosis",
        "findings",
        "investigations",
        "impression",
        "conclusion",
        "details",
        "information",
        "overview",
        "systems",
        "background",
        "discussion",
        "recommendations",
        "management",
        "procedures",
        "outcome",
        "results",
        "assessment",
        "plan",
        "bundle",
    }
)
#: Words that end a title only when everything before them is heading vocabulary.
#: "Page" is a surname as well as a part of a document, so it is never enough
#: alone ("Hope Page"), and it is not a document noun for the looser rules.
_HEADING_FINAL_NOUNS = (
    _DOCUMENT_NOUNS
    | _SECTION_NOUNS
    | frozenset(
        {
            "page",
            "pages",
            "cover",
            "index",
            "appendix",
            "annex",
            "attachment",
            "enclosure",
            "label",
            "sticker",
            "banner",
            "header",
            "footer",
            "heading",
            "title",
        }
    )
)
#: Words that describe a kind of document without naming anybody.
_HEADING_MODIFIERS = frozenset(
    {
        "discharge",
        "admission",
        "referral",
        "clinic",
        "clinical",
        "outpatient",
        "inpatient",
        "scanned",
        "typed",
        "handwritten",
        "transfer",
        "progress",
        "operative",
        "procedure",
        "surgical",
        "medication",
        "laboratory",
        "pathology",
        "radiology",
        "imaging",
        "nursing",
        "ward",
        "follow-up",
        "followup",
        "annual",
        "interim",
        "final",
        "draft",
        "patient",
        "medical",
        "care",
        "health",
        "hospital",
        "cover",
        "title",
        "front",
        "back",
        "signature",
        "continuation",
        "biopsy",
        "specimen",
        "emergency",
        "triage",
        "weekly",
        "daily",
        "monthly",
        "quarterly",
        "long-stay",
        "short-stay",
        "long",
        "short",
        "stay",
        "supported",
        "surfaces",
        "immunisation",
        "immunization",
        "vaccination",
        "allergy",
        "investigation",
        "test",
        "tests",
        "blood",
        "urine",
        "pre-operative",
        "post-operative",
        "preoperative",
        "postoperative",
        "ward-round",
        "round",
        "case",
        "day",
        "night",
        "second",
        "first",
        "new",
        "sample",
        "samples",
        "result",
        "screening",
        "consultation",
        "intake",
        "exit",
        "initial",
        "repeat",
        "updated",
        "revised",
        "copy",
        "original",
    }
)
_HEADING_SPECIALTY = re.compile(
    r"[a-z]{3,}(?:ology|ologies|iatry|iatrics|paedics|pedics|ics)"
)
_NON_TITLE_CHARACTERS = re.compile(r"[\d:]")


def _is_heading_vocabulary_word(word: str) -> bool:
    """A word that can stand in a title without being somebody's name."""
    folded = word.strip(".,;:()").casefold()
    return bool(folded) and (
        folded in _DOCUMENT_NOUNS
        or folded in _SECTION_NOUNS
        or folded in _HEADING_MODIFIERS
        or folded in _TITLE_CONNECTORS
        or folded in CLINICAL_SERVICE_TERMS
        or folded in CLINICAL_DOMAIN_NOUNS
        or bool(_HEADING_SPECIALTY.fullmatch(folded))
    )


def _is_document_title(span: str) -> bool:
    """A whole title built only from heading vocabulary and ending in a document word.

    "Discharge Checklist", "Scanned Referral Page", "Outpatient Discharge
    Summary". Every word must be heading vocabulary, so a person's name inside
    a title ("Jordan Fictional Report") makes it something else.
    """
    words = [w for w in re.findall(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", span)]
    if (
        not 2 <= len(words) <= 6
        or _NON_TITLE_CHARACTERS.search(span)
        or _NAMING_TITLE_IN_LINE.search(span)
    ):
        return False
    if words[-1].casefold() not in _HEADING_FINAL_NOUNS:
        return False
    return all(_is_heading_vocabulary_word(word) for word in words[:-1])


def _enclosing_line(text: str, start: int, end: int) -> str:
    """The whole line a span sits on."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    return text[line_start : len(text) if line_end < 0 else line_end]


#: Categories whose syntax is self-evidencing. A regex hit on one of these is
#: never overridden by a clinical reading.
EXACT_STRUCTURED_CATEGORIES = frozenset(
    {
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "SSN",
        "URL",
        "IP_ADDRESS",
        "MEDICAL_RECORD_NUMBER",
        "PATIENT_ID",
        "HEALTH_PLAN_ID",
        "INSURANCE_ID",
        "ACCESSION_NUMBER",
        "ACCOUNT_NUMBER",
        "DEVICE_ID",
        "OTHER_UNIQUE_ID",
        "AGE_OVER_89",
    }
)

#: Fewest digits a *model's* span needs before its category claim earns Layer
#: 1 authority. A telephone number has at least seven; "132/84" labelled a
#: phone number by a model is a blood pressure, not a number to dial.
_STRUCTURAL_MIN_DIGITS = {"PHONE_NUMBER": 7}

#: An ICD-10-CM diagnosis token in an explicit diagnosis field. The shape and
#: the field words come from the shared grammar, because the residual
#: validator has to exempt exactly the same thing: a code this pass keeps and
#: that pass still calls an identifier leaves the document anonymized
#: correctly and unreleasable. See ``services.structured_identifier_spec``.
_ICD_DIAGNOSIS_CODE = re.compile(
    f"^{identifier_spec.DIAGNOSIS_CODE_PATTERN}$", re.IGNORECASE
)
_DIAGNOSIS_FIELD_BEFORE = re.compile(
    r"(?:"
    + "|".join(re.escape(head) for head in identifier_spec.DIAGNOSIS_FIELD_HEADS)
    + r")\s*[:#]?\s*$",
    re.IGNORECASE,
)
#: Separators that join one token to the next inside a code or an identifier.
#: A trailing one is only a continuation when something alphanumeric follows,
#: which is what separates the sentence in "Diagnosis: J45.909." from the
#: longer identifier in "Diagnosis: J45.909-22".
_TOKEN_JOINERS = ".-/_+"

#: A whole diagnosis field: an approved head, an optional qualifier before it,
#: an approved separator and exactly one complete ICD token, with nothing else
#: inside the span.
#:
#: A model routinely proposes the label together with the value - GLiNER
#: called "Dx: J45.909" a health plan identifier - and that span is not an ICD
#: token, so reading only the value let Layer 1 remove the label and the code
#: together. Anchoring at both ends is what keeps this narrow: a span carrying
#: any further text, a second code or another identifier does not match, and
#: an identifier's own head is not in the alternation at all.
_GOVERNED_DIAGNOSIS_FIELD = re.compile(
    r"^[\s\W]{0,4}?(?:(?:"
    + "|".join(re.escape(word) for word in identifier_spec.DIAGNOSIS_FIELD_QUALIFIERS)
    + r")\s+)?(?:"
    + "|".join(re.escape(head) for head in identifier_spec.DIAGNOSIS_FIELD_HEADS)
    + r")\s*["
    + re.escape(identifier_spec.DIAGNOSIS_FIELD_SEPARATORS)
    + r"]?\s*(?P<code>"
    + identifier_spec.DIAGNOSIS_CODE_PATTERN
    + r")\s*[.,;)\]]?\s*$",
    re.IGNORECASE,
)


#: A field that names equipment, at the start of its line or cell.
_EQUIPMENT_FIELD_BEFORE = re.compile(
    r"(?:^|[\n\t|])[ \t]*(?:model|scanner(?:[ \t]+model)?|device[ \t]+model|make|manufacturer|"
    r"software(?:[ \t]+version)?|firmware(?:[ \t]+version)?)[ \t]*[:=][ \t]*$",
    re.IGNORECASE,
)
#: A product designation: a short letter prefix and a short number ("SL-40",
#: "HX 9", "AT2"). A serial number is longer and never this shape.
_PRODUCT_CODE = re.compile(r"[A-Z][A-Za-z]{0,5}[- ]?\d{1,4}[A-Za-z]?")


def _fills_equipment_field(text: str, start: int, end: int) -> bool:
    """The span is exactly a product code filling an equipment field to its end."""
    span = text[start:end]
    if not _PRODUCT_CODE.fullmatch(span):
        return False
    line_start = max(
        text.rfind(chr(10), 0, start),
        text.rfind(chr(9), 0, start),
        text.rfind("|", 0, start),
    )
    before = text[line_start + 1 if line_start >= 0 else 0 : start]
    return bool(_EQUIPMENT_FIELD_BEFORE.search(chr(10) + before)) and bool(
        _CLINICAL_FIELD_AFTER.match(text, end)
    )


def _fills_clinical_field(text: str, start: int, end: int) -> bool:
    """The span is the whole value of a field that names a clinical term.

    "analyte: Cortisol" on its own line or in its own cell: the field label
    says what the value is, and a person's name is never an analyte. Only a
    short value that fills the field to its end qualifies.
    """
    span = text[start:end]
    if not span.strip() or len(span.split()) > 4 or len(span) > 48:
        return False
    line_start = max(
        text.rfind(chr(10), 0, start),
        text.rfind(chr(9), 0, start),
        text.rfind("|", 0, start),
    )
    before = text[line_start + 1 if line_start >= 0 else 0 : start]
    return bool(_CLINICAL_FIELD_BEFORE.search(chr(10) + before)) and bool(
        _CLINICAL_FIELD_AFTER.match(text, end)
    )


def _governed_diagnosis_field(text: str, start: int, end: int) -> bool:
    """The span is exactly one diagnosis field governing one complete code.

    Judged on the structure rather than on the detector's boundaries, because
    a model draws its span where it likes: the same field arrives as the code
    alone, as a fragment of it, or with the label attached.
    """
    span = text[start:end]
    if len(span) > 96:
        return False
    match = _GOVERNED_DIAGNOSIS_FIELD.match(span)
    if match is None:
        return False
    # The code has to be a whole token in the document too, so a span ending
    # inside "J45.909X7" cannot qualify by stopping early.
    code_end = start + match.end("code")
    following = text[code_end : code_end + 2]
    if following and (
        following[0].isalnum()
        or (
            following[0] in _TOKEN_JOINERS
            and len(following) > 1
            and following[1].isalnum()
        )
    ):
        return False
    return True


def _enclosing_token(text: str, start: int, end: int) -> Tuple[int, int]:
    """The whole code- or identifier-shaped token a span sits inside.

    A model does not draw its span where a reader would. Stanford's
    deidentifier proposed ``.909`` inside ``G43.909`` and called it an
    identifier, which left ``G43`` joined to a surrogate and destroyed the
    diagnosis - the exemption declined to protect a fragment, correctly, and
    the fragment was all it was shown.

    Judging the enclosing token instead keeps both readings right: the token
    around ``.909`` is a whole ICD code and is kept, while the token around
    the ``J45.909`` of ``J45.909X7`` is the account number it belongs to and
    is not. A separator continues the token only when something alphanumeric
    follows it, so a sentence's full stop is not part of the code.
    """
    while start > 0 and (
        text[start - 1].isalnum()
        or (
            text[start - 1] in _TOKEN_JOINERS
            and start > 1
            and text[start - 2].isalnum()
        )
    ):
        start -= 1
    while end < len(text) and (
        text[end].isalnum()
        or (
            text[end] in _TOKEN_JOINERS
            and end + 1 < len(text)
            and text[end + 1].isalnum()
        )
    ):
        end += 1
    return start, end


#: Categories a clinical term can be misassigned to.
NAME_SHAPED_CATEGORIES = frozenset(
    {"PERSON", "ORGANIZATION", "FACILITY", "LOCATION", "ADDRESS", "MEDICAL_CONDITION"}
)

#: Categories that are never Safe Harbor identifiers, whichever detector
#: proposes them.
#:
#: A diagnosis is not one of the eighteen. GLiNER is asked for "medical
#: condition" because naming the label stops it reaching for PERSON when it
#: sees an eponym - the label earns its place as a *discriminator*. Treating
#: what it returns as PHI inverts the purpose of the pipeline: it redacted
#: "Vital signs remained stable" and "Fluid balance was maintained" and left
#: the note medically empty. The prediction is kept as evidence that the span
#: reads clinically; it is never grounds to remove it.
NON_IDENTIFIER_CATEGORIES = frozenset({"MEDICAL_CONDITION"})


#: Months and weekdays. A date element that Safe Harbor cares about names one
#: of these or carries digits; "the morning round" and "overnight" name
#: neither and are not date elements at all.
_CALENDAR_WORDS = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b",
    re.IGNORECASE,
)

#: Categories that are only themselves when the span carries the syntax.
_STRUCTURAL_HALLMARK = {
    "EMAIL_ADDRESS": lambda span: "@" in span,
    "URL": lambda span: "://" in span or "www." in span.lower(),
    "IP_ADDRESS": lambda span: span.count(".") >= 3 or ":" in span,
}


#: A role phrase, and a label word immediately followed by the code it labels
#: ("HCPC PA44213"): proven by structure, so model agreement does not
#: overrule them. The code itself is still replaced by its own detector.
REASON_STAFF_ROLE_PHRASE = "staff_role_phrase"
REASON_LABEL_OF_CODE = "label_of_following_code"
REASON_CLINICAL_FIELD_VALUE = "value_of_clinical_field"
REASON_CLINICAL_RESPONSE = "clinical_finding_response_grammar"
_STRUCTURAL_LABEL_READINGS = frozenset(
    {
        REASON_STAFF_ROLE_PHRASE,
        REASON_LABEL_OF_CODE,
        REASON_CLINICAL_FIELD_VALUE,
        REASON_CLINICAL_RESPONSE,
        "synthetic_provenance_marker",
        "relationship_role_noun",
    }
)
#: A registering or issuing body written as an acronym: four to six letters,
#: at least three capitals ("GPhC", "HCPC"). Initials are shorter.
_CODE_BODY_ACRONYM = re.compile(r"(?=(?:[^A-Z]*[A-Z]){3})[A-Z][A-Za-z]{3,5}")
#: The code a label governed, already replaced by the sanitizer's token.
_TOKEN_AFTER = re.compile(r"[ \t]*[:#]?[ \t]*<[A-Z][A-Z_]*_\d+>")
#: Fields whose value is a clinical term and never a person: "analyte:
#: Cortisol", "Stain = Grocott", "Organism: Nocardia".
_CLINICAL_FIELD_BEFORE = re.compile(
    r"(?:^|[\n\t|])[ \t]*(?:analyte|test(?:[ \t]+name)?|investigation|assay|parameter|marker|"
    r"measurement|drug|medication|medicine|stain|sequence|organism|specimen[ \t]+type|"
    r"sample[ \t]+type|procedure|modality)[ \t]*[:=][ \t]*$",
    re.IGNORECASE,
)
#: What may end such a value: the line, the cell, or a full stop before either.
_CLINICAL_FIELD_AFTER = re.compile(r"[ \t]*[.;,]?[ \t]*(?:$|[\n\r\t|])")
_LABELLED_CODE_AFTER = re.compile(r"[ \t]*[:#]?[ \t]*[A-Za-z]{0,4}[-/]?\d")

#: A span that is nothing but one number.
_NUMBER_ONLY = re.compile(r"\d{1,9}(?:[.,]\d+)?")
#: A character that joins a number to a code: "E11.909", "12-909", "A909".
_JOINED_TO_CODE = re.compile(r"[\w\-/.:#]")
#: After the number a full stop joins only when a digit follows: "2946." ends
#: a sentence, "909.1" continues a code.
_JOINED_TO_CODE_AFTER = re.compile(r"[\w\-/:#]|[.,]\w")
#: Numeric-context reasons that are readings. Part of a code, and a room,
#: page or sequence number, are not: a fragment of an identifier must never
#: be kept because of where it sits.
_NUMERIC_READINGS = frozenset(
    {
        "value_with_unit",
        "count_or_duration",
        "imaging_or_dose_parameter",
        "year",
        "clock_time",
        "score_or_measured_quantity",
    }
)


def _has_structural_syntax(category: str, span: str) -> bool:
    """Whether a span actually looks like the category it was labelled.

    Layer 1 exists for a regex that matched identifier syntax. Extending its
    authority to a *model* saying "PHONE_NUMBER" gives a prediction the
    standing of a match: GLiNER labelled the bare word "Fax" a phone number,
    and Layer 1 then blocked the document with no further examination. A model
    finding has to look like the thing before it gets that authority.
    """
    hallmark = _STRUCTURAL_HALLMARK.get(category)
    if hallmark is not None:
        return bool(hallmark(span))
    digits = sum(character.isdigit() for character in span)
    return digits >= _STRUCTURAL_MIN_DIGITS.get(category, 2)


def is_relative_time_expression(category: str, span: str) -> bool:
    """A time reference with no calendar content is not a date element.

    Safe Harbor removes elements of dates. "the morning round", "overnight"
    and "the next appointment" carry no date, and replacing them with
    <REDACTED_TIME> removes clinical sequencing for no privacy gain.
    """
    if category not in {"DATE_TIME", "DATE", "TIME"}:
        return False
    if any(character.isdigit() for character in span):
        return False
    return not _CALENDAR_WORDS.search(span)


#: A year, or a numeric date written with separators. Used to keep a date
#: element out of the clinical-reading exemption: "5 Jan", "12/03/2024" and
#: "2024-01-05" carry calendar content and are removed whatever else the span
#: reads as, while "HbA1c 58" and "ramipril 5 mg" carry none.
_CALENDAR_FIGURE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b\d{1,4}[-/.]\d{1,2}(?:[-/.]\d{1,4})?\b"
)


def _carries_calendar_content(category: str, span: str) -> bool:
    if category not in _TEMPORAL_CATEGORIES:
        return False
    return bool(_CALENDAR_WORDS.search(span) or _CALENDAR_FIGURE.search(span))


def evidence_type(source: str) -> str:
    if source in _DETERMINISTIC_SOURCES:
        return EVIDENCE_DETERMINISTIC
    if source == SOURCE_CONTEXT_RULE:
        return EVIDENCE_CONTEXT
    if source == SOURCE_STRICT_PROPER_NOUN:
        return EVIDENCE_HEURISTIC
    if source in _MODEL_SOURCES:
        return EVIDENCE_MODEL
    return EVIDENCE_MODEL


# -- contextual evidence that a term is being used as an identifier ---------

#: Clinical actions that take an agent: "given by", "reviewed by". Verbs, never
#: part of a name - "Given PCV15" is an action and a vaccine code.
_CLINICAL_ACTION_VERBS = (
    "seen",
    "treated",
    "reviewed",
    "referred",
    "examined",
    "admitted",
    "discharged",
    "signed",
    "countersigned",
    "accompanied",
    "administered",
    "given",
    "prescribed",
    "performed",
    "vaccinated",
)
#: Titles and constructions that mark the following words as a person, whatever
#: the dictionary says about them.
_PERSON_CONTEXT_BEFORE = re.compile(
    r"(?:\b(?:dr|dr\.|doctor|prof|prof\.|professor|mr|mr\.|mrs|mrs\.|ms|ms\.|"
    r"miss|sir|dame|nurse|sister|consultant|attending|surgeon|registrar)\s*|"
    r"\b(?:patient(?:\s+name)?|attending\s+physician)\s*[:\-\t]\s*|"
    r"\b(?:patient|resident|client|subject|donor|recipient)\s+|"
    r"\b(?:" + "|".join(_CLINICAL_ACTION_VERBS) + r")\s+by\s+|"
    r"\bname\s*[:\-]\s*)$",
    re.IGNORECASE,
)
_PERSON_CONTEXT_AFTER = re.compile(
    r"^\s*(?:\b(?:attended|visited|called|arrived|consented|reported|stated|"
    r"complained|presented|was\s+admitted|was\s+discharged|was\s+seen|"
    r"signed|telephoned|is\s+the\s+(?:patient|next\s+of\s+kin))\b)",
    re.IGNORECASE,
)
_FACILITY_CONTEXT_BEFORE = re.compile(
    r"(?:\bat\s+|\badmitted\s+to\s+|\btransferred\s+to\s+|\breferred\s+to\s+)$",
    re.IGNORECASE,
)
#: A possessive immediately after the term, with a clinical head noun soon
#: after, is the classic eponym construction.
_SCALE_HEADS = (
    "chart",
    "charts",
    "scale",
    "scales",
    "score",
    "scores",
    "index",
    "status",
    "rating",
    "classification",
    "criteria",
)
_EPONYM_AFTER = re.compile(
    r"^['’]?s?\s+(?:(?:"
    + "|".join(sorted(NAMED_SCALE_MODIFIERS))
    + r")\s+(?:"
    + "|".join(_SCALE_HEADS)
    + r")|(?:"
    + "|".join(sorted(CLINICAL_HEAD_NOUNS))
    + r"))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvidenceAssessment:
    """Why a finding was kept, dropped or escalated. Carries no value."""

    category: str
    source: str
    evidence: str
    #: "identifier" | "clinical" | "ambiguous"
    reading: str
    #: "redact" | "preserve" | "review"
    action: str
    rationale: str
    agreeing_detectors: int = 1
    #: False for a redaction proposed by a model or heuristic with no
    #: identifier evidence behind it. It is still applied - privacy first -
    #: but the file it is in cannot release automatically.
    justified: bool = True

    def to_public_dict(self) -> Dict[str, object]:
        return {
            "category": self.category,
            "detector": self.source,
            "evidence_type": self.evidence,
            "reading": self.reading,
            "action": self.action,
            "rationale": self.rationale,
            "agreeing_detectors": self.agreeing_detectors,
            "justified": self.justified,
        }


def _starts_a_new_labelled_field(text: str, start: int, end: int) -> bool:
    """Whether this span opens a field of its own.

    True for the "Dx" in "Dx: K80.20", false for the "Dr Achebe" that follows
    "Reviewed by:". The distinction decides whether a label on the line above
    is still speaking.
    """
    span = text[start:end].lstrip()
    head = re.split(r"[\s:;,]+", span, maxsplit=1)[0] if span else ""
    if not head or not is_clinical_field_label(head):
        return False
    rest = span[len(head) :].lstrip()
    return rest.startswith(":")


def _context_says_person(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 48) : start]
    after = text[end : end + 48]
    if chr(10) in before and _starts_a_new_labelled_field(text, start, end):
        # A field label introduces the value on its own line. "Patient:" with
        # its value removed does not reach across the break to claim the next
        # field, and letting it was enough to make "Dx: K80.20" read as a
        # person and hold the document. A span that is not itself a field
        # head - the "Dr Achebe" under "Reviewed by:" - keeps the whole of
        # its context, so a name that wrapped onto its own line is still
        # named.
        before = before.rsplit(chr(10), 1)[-1]
    return bool(
        _PERSON_CONTEXT_BEFORE.search(before) or _PERSON_CONTEXT_AFTER.match(after)
    )


def _context_says_facility(text: str, start: int, end: int) -> bool:
    return bool(_FACILITY_CONTEXT_BEFORE.search(text[max(0, start - 48) : start]))


#: Characters that end a sentence or a labelled line. `rstrip` removes the
#: newline itself, so a line break shows up as the punctuation before it or as
#: an empty prefix.
_SENTENCE_ENDINGS = frozenset(".!?:;")


def _is_sentence_initial(text: str, start: int) -> bool:
    """True when a span opens a sentence.

    English capitalises the first word of every sentence, so a capitalised
    token there is not evidence of a name. Without this the proper-noun
    heuristic flags ordinary openers - "Care was provided", "Assessment
    follows" - and either destroys them or floods manual review.
    """
    before = text[:start].rstrip()
    if not before:
        return True
    return before[-1] in _SENTENCE_ENDINGS


def _is_single_token(text: str, start: int, end: int) -> bool:
    return not re.search(r"\s", text[start:end].strip())


def _context_says_clinical(text: str, start: int, end: int) -> bool:
    """A clinical construction around the span: possessive plus a head noun.

    Also a capitalised phrase that itself ends in a head noun ("Glasgow Coma
    Scale", "Mowat-Wilson syndrome"): a person's name does not end that way.
    """
    if _EPONYM_AFTER.match(text[end : end + 48]):
        return True
    words = text[start:end].split()
    # The first word has to be an ordinary capitalised word: a code such as
    # "PAS-D" after "Passport:" is not an eponym however the phrase ends.
    return (
        len(words) >= 2
        and re.fullmatch(r"[A-Z][a-z]+(?:[-'][A-Za-z]+)*", words[0]) is not None
        and words[-1].strip(".,;:()").casefold() in CLINICAL_HEAD_NOUNS
    )


#: Words that exist to mark the thing before them as an organisation. A span
#: ending in one is a named body, whatever clinical words it contains: "Blood
#: Pressure Diagnostics Ltd" is a company, not a vital sign. This is a
#: grammatical designator test, not a directory of known organisations, so an
#: institution nobody listed is caught the same way.
_ORG_DESIGNATOR = re.compile(
    r"\b(?:ltd|limited|inc|incorporated|llc|llp|plc|pvt|gmbh|bv|nv|ag|sa|"
    r"corp|corporation|company|holdings|ventures|enterprises|"
    r"hospital|hospitals|clinic|clinics|infirmary|institute|institution|"
    r"laborator(?:y|ies)|labs?|diagnostics|imaging|radiology|pathology|"
    r"healthcare|health(?:care)?\s+system|trust|foundation|charity|"
    r"university|college|academy|"
    r"hospice|dispensary|polyclinic|nursing\s+home|"
    r"associates|partners|practice|centre|center)\.?$",
    re.IGNORECASE,
)

#: A quantity with a unit immediately after a term is prescribing grammar.
#: "Ranolazatib 250 mg twice daily" reads as a medication even though no
#: vocabulary contains that invented name - which is the point: the evidence
#: is the construction, so an unseen drug survives on the same footing as a
#: familiar one.
_DOSAGE_AFTER = re.compile(
    r"^[\s,:-]*\d+(?:[.,]\d+)?\s*"
    r"(?:mg|mcg|µg|ug|g|kg|ml|l|iu|units?|mmol|meq|mg/kg|mg/dl|mmhg|%)\b",
    re.IGNORECASE,
)
#: A finding followed by a response and a dose-bearing treatment. Terminology
#: alone is not release authority; this grammar is the structural evidence.
_CLINICAL_RESPONSE_AFTER = re.compile(
    r"^\s+(?:improved|worsened|resolved|recurred|persisted|settled|responded)\b"
    r"[^.!?\n]{0,48}\b\d+(?:[.,]\d+)?\s*"
    r"(?:mg|mcg|µg|ug|g|kg|ml|l|iu|units?|mmol|meq|mg/kg|mg/dl|mmhg|%)\b",
    re.IGNORECASE,
)


#: Words that can open a noun phrase without naming anything.
_DETERMINERS = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "our",
        "their",
        "his",
        "her",
        "its",
        "my",
        "your",
        "each",
        "every",
        "any",
        "another",
    }
)


def _has_organisation_designator(span: str, sentence_initial: bool = False) -> bool:
    """True when the span ends in a word that declares it an organisation.

    The designator only declares a *named* body. "Marlowbridge Clinic" is one;
    "the review clinic" and "a diabetes clinic" are services, and treating
    them as organisations replaced ordinary words with surrogates. So some
    word before the designator has to be capitalised - and not merely because
    it opens the sentence, which is ordinary sentence case and not a name.
    """
    cleaned = span.strip().rstrip(".,;:")
    if not re.search(r"\s", cleaned):
        # A single word is its own designator only by coincidence; requiring
        # a modifier before it keeps "Practice" or "Centre" standing alone in
        # ordinary prose from reading as a company name.
        return False
    match = _ORG_DESIGNATOR.search(cleaned)
    if not match:
        return False
    modifiers = [
        word
        for word in re.findall(r"[^\s]+", cleaned[: match.start()])
        if word.casefold().strip(".,'") not in _DETERMINERS
    ]
    if sentence_initial and modifiers and cleaned.split()[0] == modifiers[0]:
        modifiers = modifiers[1:]
    return any(word[:1].isupper() or word[:1].isdigit() for word in modifiers)


def _context_says_dosage(text: str, start: int, end: int) -> bool:
    return bool(_DOSAGE_AFTER.match(text[end : end + 32]))


# -- readings, bed designators and clinical scores ---------------------------
#
# None of these can identify anybody, and models label all of them as
# identifiers: GLiNER called "BP 132/84" a phone number and "HR 90" a
# health-plan number, spaCy called "Bay 4" a date, and Stanford called
# "NEWS2" an identifier. Each is recognised by its construction, never by the
# sentence it appeared in.

#: Vital-sign labels and units, from the canonical identifier spec, so the
#: first pass's phone guard and this model cannot disagree about a reading.
_VITAL_SIGN_LABELS = identifier_spec.VITAL_SIGN_LABELS
_MEASUREMENT_UNIT = (
    "(?:"
    + "|".join(
        re.escape(unit)
        for unit in sorted(identifier_spec.MEASUREMENT_UNITS, key=len, reverse=True)
    )
    + ")(?![a-z])"
)
_VITAL_LABEL = "(?:" + "|".join(sorted(_VITAL_SIGN_LABELS, key=len, reverse=True)) + ")"
#: A bare ratio is never protected on its own: 12/03 reads as a blood
#: pressure or as a day and month, and a date element is an identifier. With a
#: vital-sign label in front of it the reading is no longer ambiguous.
_LABELLED_VALUE = (
    _VITAL_LABEL + r"\s*[:=]?\s*\d{1,3}(?:[.,]\d+)?(?:\s*/\s*\d{1,3})?"
    r"(?:\s*" + _MEASUREMENT_UNIT + ")?"
)
_UNIT_VALUE = r"\d{1,4}(?:[.,]\d+)?\s*" + _MEASUREMENT_UNIT
_MEASUREMENT_ITEM = "(?:" + _LABELLED_VALUE + "|" + _UNIT_VALUE + ")"
_MEASUREMENT_EXPRESSION = re.compile(
    "^" + _MEASUREMENT_ITEM + r"(?:\s*[,;]\s*" + _MEASUREMENT_ITEM + ")*$",
    re.IGNORECASE,
)
_LABELLED_EXPRESSION = re.compile(
    "^" + _LABELLED_VALUE + r"(?:\s*[,;]\s*" + _LABELLED_VALUE + ")*$",
    re.IGNORECASE,
)
# A unitless value is still a reading when its *own* phrase says it is a
# laboratory result. This does not borrow a label across a semicolon: the
# same digits in an adjacent extension remain a contact identifier.
_LAB_RESULT_EXPRESSION = re.compile(
    r"^(?:laboratory|lab)\s+(?:result|value|reading)\s*[:=]?\s*"
    r"\d{1,6}(?:[.,]\d+)?$",
    re.IGNORECASE,
)
_VALUE_AFTER_VITAL_LABEL = re.compile(
    r"^\s*[:=]?\s*\d{1,3}(?:[.,]\d+)?(?:\s*/\s*\d{1,3})?", re.IGNORECASE
)
#: A reading with no label of its own, and the label written just before it:
#: "132/84" in "BP 132/84".
_BARE_READING = re.compile(
    r"^\d{1,3}(?:[.,]\d+)?(?:\s*/\s*\d{1,3})?(?:\s*" + _MEASUREMENT_UNIT + r")?$",
    re.IGNORECASE,
)
_BARE_LEADING_READING = re.compile(
    r"^\s*\d{1,3}(?:[.,]\d+)?(?:\s*/\s*\d{1,3})?(?:\s*"
    + _MEASUREMENT_UNIT
    + r")?(?=[\s,;]|$)",
    re.IGNORECASE,
)
#: A vital-sign label left at the end of a span with its value after it: the
#: "BP" of "<name> BP" when "118/76" follows.
_TRAILING_VITAL_LABEL = re.compile(
    r"(?:^|(?<=[\s,;:|]))" + _VITAL_LABEL + r"[\s:=]*$", re.IGNORECASE
)
_VITAL_LABEL_JUST_BEFORE = re.compile(
    r"(?<![A-Za-z0-9])" + _VITAL_LABEL + r"\s*[:=]?\s*$", re.IGNORECASE
)
#: A unit standing after its number: "nmol/L" in "452 nmol/L". Single-letter
#: units are left to the single-character rule.
_LONE_UNIT = re.compile(
    r"^(?:"
    + "|".join(
        re.escape(unit)
        for unit in sorted(identifier_spec.MEASUREMENT_UNITS, key=len, reverse=True)
        if len(unit) > 1
    )
    + r")(?:/[a-z]{1,3})?$",
    re.IGNORECASE,
)
_NUMBER_JUST_BEFORE = re.compile(r"\d(?:[.,]\d+)?\s*$")

#: A unit written with a solidus: "g/L", "mmol/mol", "mL/min/1.73m2". The
#: recorded units cover what somebody listed; this covers the shape. No
#: identifier is written as a word divided by a solidus followed by a word,
#: and a laboratory result routinely is.
_COMPOUND_UNIT = r"[A-Za-zµ]{1,6}(?:/[A-Za-z0-9.]{1,10}){1,2}"
#: The compound form is tried first so "g/L" is read whole. Read the recorded
#: unit first and the match stops at the "g", leaving "/L" outside it, and the
#: coverage test below would then see a measurement that does not account for
#: its own span.
_ANALYTE_UNIT = "(?:" + _COMPOUND_UNIT + "|" + _MEASUREMENT_UNIT + ")"
#: An analyte, its value and its unit. The analyte does not have to be a word
#: anybody listed - that is the point. A closed vocabulary protects the
#: analytes somebody thought of and destroys every other laboratory result,
#: and "Haemoglobin 118 g/L" was destroyed 104 times by exactly that gap.
_ANALYTE_MEASUREMENT = re.compile(
    r"^([A-Za-z][A-Za-z0-9'\-]{2,})\s*(?:[:=]|of|was|is|at)?\s*"
    r"\d{1,4}(?:[.,]\d+)?\s*" + _ANALYTE_UNIT,
    re.IGNORECASE,
)
#: Characters that end a token: whitespace, list punctuation, and the
#: brackets a unit expression can sit inside.
_TOKEN_BREAK = frozenset(" \t\r\n,;:|()[]")
#: How far past a span's end the unit that completes a measurement may sit.
#: Long enough for " mL/min/1.73m2", short enough that it cannot reach into
#: the next clause.
_MEASUREMENT_LOOKAHEAD = 16


def _is_analyte_measurement(text: str, start: int, end: int) -> bool:
    """Whether the span is an analyte with its value and unit.

    The leading word has to be one a surname is unlikely to be mistaken for:
    lower case, carrying a digit, or opening a sentence. "ferritin 14 ug/L",
    "HbA1c 58 mmol/mol" and a sentence-initial "Haemoglobin 118 g/L" qualify;
    a capitalised surname in the middle of a sentence with a dose after it
    does not, and is left to the rules that already judge it.

    The unit is looked for a little past the span's end, because a model does
    not draw its span where a reader would: spaCy proposed "Haemoglobin 118"
    without the "g/L" that makes it a result, 106 times. And the measurement
    has to account for the *whole* span, not merely open it - "Haemoglobin 118
    g/L and Orsolyne Marchbanke" is a measurement followed by a name, and a
    rule that read only the beginning of it would release the name.
    """
    window = text[start : min(len(text), end + _MEASUREMENT_LOOKAHEAD)]
    leading = len(window) - len(window.lstrip())
    match = _ANALYTE_MEASUREMENT.match(window[leading:])
    if match is None:
        return False
    if leading + match.end() < len(text[start:end].rstrip(" .,;:")):
        return False
    word = match.group(1)
    return (
        word.islower()
        or any(character.isdigit() for character in word)
        or _is_sentence_initial(text, start)
    )


def _is_measurement_unit(text: str, start: int, end: int) -> bool:
    """The span is a unit being used to measure something.

    Two losses in the same family made this necessary. A temperature written
    "38.2 degrees Celsius" came back as "38.2 degrees <PERSON_2>", and an
    acuity written "12 Snellen metres" lost the scale it was measured on. The
    tagger is not wrong about the shape - Anders Celsius and Herman Snellen
    were people, and a patient may be called either - so being in the lexicon
    protects nothing by itself.

    What protects it is the company it keeps: a number or a unit word on one
    side, or a unit head noun on the other. "Reviewed by Celsius" has neither
    and stays a name, which is the case this rule is written around rather
    than against.
    """
    span = text[start:end].strip()
    if not span:
        return False
    if not (
        is_unit_eponym(span) or span.casefold() in identifier_spec.MEASUREMENT_UNITS
    ):
        return False

    before = text[max(0, start - 24) : start]
    after = text[end : end + 24]
    previous = before.rsplit(None, 1)[-1] if before.strip() else ""
    following = after.split(None, 1)[0] if after.strip() else ""

    # "38.2 degrees Celsius" / "12 Snellen metres" / "umol/L" / "28 umol"
    if previous and (is_unit_head_word(previous) or previous[-1:].isdigit()):
        return True
    if following and is_unit_head_word(following):
        return True
    # Part of a compound unit: "umol/L" reads as one token to a clinician and
    # as a word beside a solidus to a tagger.
    if text[end : end + 1] == "/" or text[max(0, start - 1) : start] == "/":
        return True
    return False


def _is_unit_fragment(text: str, start: int, end: int) -> bool:
    """Whether the span is the denominator of a compound unit.

    "mmol/mol" and "L/min" are one token each, but a tagger sees a lower-case
    word after a solidus and the proper-noun fallback proposed "mol" and "min"
    as names 250 times. A fragment only counts when what precedes the solidus
    is itself a value or a recorded unit, so "Marchbanke/Nettlecombe" is not
    turned into a measurement by this.
    """
    if start == 0 or text[start - 1] != "/":
        return False
    left = start - 1
    while left > 0 and text[left - 1] not in _TOKEN_BREAK:
        left -= 1
    head = text[left : start - 1].split("/")[0]
    if not head:
        return False
    return bool(
        head[0].isdigit()
        or head.casefold() in identifier_spec.MEASUREMENT_UNITS
        or _LONE_UNIT.match(head)
    )


#: The categories any measurement gets misread as, unit-only readings
#: included. Places are deliberately absent: a number standing on its own can
#: be part of an address.
_MEASUREMENT_MISREAD_CATEGORIES = frozenset({"PERSON", "ORGANIZATION", "FACILITY"})
#: Categories a *labelled* reading gets misread as. A label is required here
#: because these categories do carry numbers; "HR 90" cannot be a health-plan
#: number, but "90" alone could be part of one.
_READING_MISREAD_CATEGORIES = frozenset(
    {
        "PHONE_NUMBER",
        "HEALTH_PLAN_ID",
        "INSURANCE_ID",
        "MEDICAL_RECORD_NUMBER",
        "PATIENT_ID",
        "ACCESSION_NUMBER",
        "ACCOUNT_NUMBER",
        "DEVICE_ID",
        "US_SSN",
        "SSN",
        "IDENTIFIER",
        "PASSPORT_NUMBER",
        "DRIVER_LICENSE",
        "USERNAME",
        "DATE_TIME",
        "DATE",
        "TIME",
        "ADDRESS",
        "POSTAL_CODE",
        "LOCATION",
    }
)
#: Categories whose spans are never widened to swallow a unit-only reading.
_PLACE_CATEGORIES = frozenset({"LOCATION", "ADDRESS", "POSTAL_CODE", "GPE"})

_DESIGNATOR_WORD = (
    r"(?:bay|bed|bedspace|bed\s+space|cubicle|side\s+room|room|rm|theatre|theater|pod|"
    r"resus|trolley|chair|ward|cot|incubator)"
)
_DESIGNATOR_ITEM = _DESIGNATOR_WORD + r"\s*(?:no\.?\s*|#\s*)?\d{1,3}[A-Za-z]?"
#: A bed, bay or theatre with its number: "Bay 4", "bed 12", "Ward 7B".
_CLINICAL_DESIGNATOR = re.compile(
    r"(?<![A-Za-z0-9])" + _DESIGNATOR_ITEM + r"(?![\w%/\-]|[.,]\d)",
    re.IGNORECASE,
)
#: Words after a designator that turn it into part of an address.
_STREET_WORDS = frozenset(
    {
        "street",
        "st",
        "road",
        "rd",
        "avenue",
        "ave",
        "lane",
        "ln",
        "drive",
        "close",
        "way",
        "court",
        "ct",
        "crescent",
        "terrace",
        "place",
        "square",
        "gardens",
        "grove",
        "hill",
        "park",
        "row",
        "walk",
        "mews",
    }
)

#: Scoring systems whose names are not words anybody is called.
_DISTINCTIVE_SCORES = (
    "news2",
    "mews",
    "pews",
    "apgar",
    "curb-65",
    "crb-65",
    "curb65",
    "qsofa",
    "sofa",
    "cha2ds2-vasc",
    "cha2ds2vasc",
    "has-bled",
    "ecog",
    "nyha",
    "timi",
    "meld",
    "child-pugh",
    "phq-9",
    "gad-7",
    "mmse",
    "amts",
    "4at",
    "audit-c",
    "abcd2",
    "nihss",
    "das28",
    "rass",
    "cpot",
    "flacc",
    "epds",
    "hads",
)
#: Scoring systems named after people or ordinary words. Only ever a score
#: with the word "score" written after them: "Wells score 4", never "Wells 4".
_WORDED_SCORES = (
    "news",
    "wells",
    "braden",
    "waterlow",
    "rockwood",
    "barthel",
    "alvarado",
    "centor",
    "ranson",
    "grace",
    "must",
    "bishop",
    "glasgow",
)


def _alternation(words) -> str:
    """Words as one alternation, longest first so a prefix never wins."""
    return "|".join(re.escape(word) for word in sorted(words, key=len, reverse=True))


_SCORE_VALUE = r"(?:\s*(?:of\s+|[:=]\s*)?\d{1,2}(?:\s*/\s*\d{1,2})?)"
_DISTINCTIVE_SCORE = re.compile(
    r"(?<![A-Za-z0-9])(?P<name>"
    + _alternation(_DISTINCTIVE_SCORES)
    + r")(?P<rest>(?:\s+(?:score|grade|class))?"
    + _SCORE_VALUE
    + r"?)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_WORDED_SCORE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    + _alternation(_WORDED_SCORES)
    + r")\s+score"
    + _SCORE_VALUE
    + r"?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_SCORE_ITEM = (
    r"(?:(?:"
    + _alternation(_DISTINCTIVE_SCORES)
    + r")(?:\s+(?:score|grade|class))?"
    + _SCORE_VALUE
    + r"|(?:"
    + _alternation(_WORDED_SCORES)
    + r")\s+score"
    + _SCORE_VALUE
    + ")"
)


# -- durations ---------------------------------------------------------------
#
# "Amoxicillin for 5 days" states how long, not when. spaCy labels the "5
# days" a date and it was replaced, taking a treatment course out of the note.
# A date element anchors something to a moment; a duration does not. So a
# quantity of time is kept only when a duration cue governs it, and anything
# that anchors it to a moment - "5 days ago", "3 days after admission",
# "within 5 days" - or leaves it bare keeps the handling it had before.

_TEMPORAL_CATEGORIES = frozenset({"DATE", "DATE_TIME", "TIME"})
_DURATION_UNIT = (
    r"(?:minutes?|mins?|hours?|hrs?|days?|nights?|weeks?|wks?|months?|years?|yrs?)"
)
_DURATION_COUNT = (
    r"(?:\d{1,3}(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|fourteen|twenty|thirty|several|few)"
)
#: A counted quantity of time: "5 days", "10 to 14 days", "3-4 weeks".
_DURATION_QUANTITY = re.compile(
    r"^" + _DURATION_COUNT + r"(?:\s*(?:-|to|or)\s*" + _DURATION_COUNT + r")?"
    r"[\s-]*" + _DURATION_UNIT + r"$",
    re.IGNORECASE,
)
#: A cue written inside the span itself: "for 5 days".
_DURATION_CUE_INSIDE = re.compile(r"^(?:for|over|lasting|x|\u00d7)\s*", re.IGNORECASE)
#: Words just before a quantity that make it a length of time.
_DURATION_CUE_BEFORE = re.compile(
    r"(?:\bfor|\bover|\blasting|\bx|\u00d7|\bevery|\bduration\s+of|\bcourse\s+of|"
    r"\bperiod\s+of|\b(?:the\s+)?(?:past|last|next|previous|further|another))\s*$",
    re.IGNORECASE,
)
#: Words just after a quantity that anchor it to a moment or make it an age.
_ANCHOR_AFTER = re.compile(
    r"^\s*(?:ago|later|earlier|before|after|prior|previously|since|from|until|"
    r"post[- ]?op\w*|of\s+(?:admission|discharge|age|life)|old|of\s+age)\b",
    re.IGNORECASE,
)
#: A years count this large reads as an age, and ages over 89 are identifiers.
_AGE_SIZED_YEARS = 89


def _is_duration(text: str, start: int, end: int) -> bool:
    """A quantity of time that a duration cue governs and nothing anchors."""
    start, end = _stripped_bounds(text, start, end)
    span = text[start:end]
    cue = _DURATION_CUE_INSIDE.match(span)
    quantity = span[cue.end() :] if cue else span
    if not _DURATION_QUANTITY.match(quantity.strip()):
        return False
    if _ANCHOR_AFTER.match(text[end : end + 32]):
        return False
    if re.search(r"\b(?:years?|yrs?)\b", quantity, re.IGNORECASE) and any(
        int(number) > _AGE_SIZED_YEARS for number in re.findall(r"\d+", quantity)
    ):
        return False
    return bool(cue) or bool(
        _DURATION_CUE_BEFORE.search(text[max(0, start - 24) : start])
    )


#: A relative offset from an unnamed moment: "6 weeks", "2 days ago", "in 3
#: months", "day 5", "post-operative day 5". Nothing in it is a calendar
#: element, so nothing in it can place the event on a date.
_RELATIVE_OFFSET = re.compile(
    r"^(?:(?:in|within|after|for|over|every)\s+)?"
    + _DURATION_COUNT
    + r"(?:\s*(?:-|to|or)\s*"
    + _DURATION_COUNT
    + r")?[\s-]*"
    + _DURATION_UNIT
    + r"(?:\s+(?:ago|later|earlier|before|after|prior))?$",
    re.IGNORECASE,
)
_DAY_COUNT = re.compile(
    r"^(?:post[- ]?op(?:erative)?\s+|hospital\s+|illness\s+|treatment\s+|admission\s+)?"
    r"(?:day|week)\s+\d{1,3}$",
    re.IGNORECASE,
)
#: What turns an offset back into an age or a bound to a date.
_OFFSET_NOT_RELATIVE_AFTER = re.compile(
    r"^\s*(?:old\b|of\s+age\b|since\b|until\b|from\s+\d)", re.IGNORECASE
)


def _is_relative_offset(text: str, start: int, end: int) -> bool:
    """A counted offset or day number with no calendar content in it."""
    start, end = _stripped_bounds(text, start, end)
    span = text[start:end].strip(" .,;:")
    if not (_RELATIVE_OFFSET.match(span) or _DAY_COUNT.match(span)):
        return False
    if _OFFSET_NOT_RELATIVE_AFTER.match(text[end : end + 24]):
        return False
    if re.search(r"\b(?:years?|yrs?)\b", span, re.IGNORECASE) and any(
        int(number) > _AGE_SIZED_YEARS for number in re.findall(r"\d+", span)
    ):
        return False
    return True


_CLOCK_TIME = re.compile(r"^(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\s?[AaPp][Mm])?$")
#: A calendar element on the same line: numeric dates, month names, a bare year
#: beside a month, or a sanitizer date placeholder already standing there.
_DATE_ON_LINE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b|"
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b|"
    r"<REDACTED_DATE>|T\d{2}:\d{2}",
    re.IGNORECASE,
)


def is_undated_clock_time(text: str, start: int, end: int) -> bool:
    """A time of day with no date beside it: not an element of any date."""
    span = text[start:end].strip()
    if not _CLOCK_TIME.match(span):
        return False
    return not _DATE_ON_LINE.search(_enclosing_line(text, start, end))


_TREATMENT_FOLLOWUP = re.compile(
    r"\b(?:continue|complete|maintain|finish|resume)\b.{0,48}"
    r"\b(?:return|follow[- ]?up|review)\s+in\s*$",
    re.IGNORECASE,
)


def _is_treatment_followup_interval(text: str, start: int, end: int) -> bool:
    """A treatment-plan interval, not a calendar date or historical anchor."""
    start, end = _stripped_bounds(text, start, end)
    quantity = re.sub(r"^in\s+", "", text[start:end], flags=re.IGNORECASE)
    if not _DURATION_QUANTITY.fullmatch(quantity.strip()):
        return False
    if _ANCHOR_AFTER.match(text[end : end + 32]):
        return False
    return bool(_TREATMENT_FOLLOWUP.search(text[max(0, start - 80) : start]))


def _stripped_bounds(text: str, start: int, end: int) -> Tuple[int, int]:
    segment = text[start:end]
    left = start + (len(segment) - len(segment.lstrip()))
    right = end - (len(segment) - len(segment.rstrip()))
    return left, max(left, right)


def _is_labelled_reading(text: str, start: int, end: int) -> bool:
    """A reading whose vital-sign label is in the span or just before it."""
    start, end = _stripped_bounds(text, start, end)
    span = text[start:end].rstrip(".,;")
    if not span:
        return False
    if _LAB_RESULT_EXPRESSION.fullmatch(span):
        number = re.search(r"\d{1,6}(?:[.,]\d+)?$", span)
        if (
            number is not None
            and numeric_context.classify(span, number.start(), number.end()).is_clinical
        ):
            return True
    if _LABELLED_EXPRESSION.match(span):
        return True
    if span.casefold() in _VITAL_SIGN_LABELS and _VALUE_AFTER_VITAL_LABEL.match(
        text[end : end + 24]
    ):
        return True
    return bool(
        _BARE_READING.match(span)
        and _VITAL_LABEL_JUST_BEFORE.search(text[max(0, start - 16) : start])
    )


def _is_lone_unit(text: str, start: int, end: int) -> bool:
    start, end = _stripped_bounds(text, start, end)
    return bool(
        _LONE_UNIT.match(text[start:end])
        and _NUMBER_JUST_BEFORE.search(text[max(0, start - 16) : start])
    )


def _is_measurement(text: str, start: int, end: int) -> bool:
    """A labelled or unit-bearing value, or a vital-sign label before its value.

    Grammar, not a list of known readings: "BP 128/76", "SpO2 94%" and
    "139 mmol/L" all read as measurements whether or not anyone has seen that
    value before, and none of them can name a person.
    """
    span = text[start:end].strip().rstrip(".")
    if _MEASUREMENT_EXPRESSION.match(span):
        return True
    return _is_labelled_reading(text, start, end) or _is_lone_unit(text, start, end)


def _inside_match(
    pattern: "re.Pattern[str]", text: str, start: int, end: int
) -> Optional["re.Match[str]"]:
    """The match of ``pattern`` that wholly contains [start, end), if any."""
    start, end = _stripped_bounds(text, start, end)
    if start >= end:
        return None
    for match in pattern.finditer(text, max(0, start - 32), min(len(text), end + 24)):
        if match.start() <= start and end <= match.end():
            return match
    return None


def _designator_continues_into_an_address(text: str, end: int) -> bool:
    """A street word, or a capitalised name, directly after a designator and its number.

    "Bay 4 Harbour Road" is an address fragment and "Ward 12 Street shops" is a
    street, whatever the designator before them is; they get no protection.
    """
    following = re.match(r"[ \t]+([A-Za-z][A-Za-z'\-]*)", text[end : end + 32])
    if not following:
        return False
    word = following.group(1)
    # Another place word ("Clinic 2 Room 5") continues the list of places; it is not
    # the beginning of an address.
    if re.fullmatch(_PLACE_WORDS, word, re.IGNORECASE):
        return False
    return word.casefold() in _STREET_WORDS or (
        len(word) > 1 and word[0].isupper() and word[1:].islower()
    )


def _is_clinical_designator(text: str, start: int, end: int) -> bool:
    """A bed, bay or theatre number, not an address.

    "Bay 4", "bed 12" and "Ward 7B" identify a place in a ward, which is not a
    Safe Harbor identifier. A street word or a capitalised name straight after
    the number makes it an address fragment instead, and it gets no
    protection.
    """
    match = _inside_match(_CLINICAL_DESIGNATOR, text, start, end)
    if match is None:
        return False
    # The same guard as every other designator reading: a street word or a name
    # after the number is an address, a following place word ("Bay 4 Room 11") is
    # the next place in a list.
    return not _designator_continues_into_an_address(text, match.end())


def _is_clinical_score(text: str, start: int, end: int) -> bool:
    """A named clinical score: "NEWS2 score 5", "qSOFA 2", "Wells score 4"."""
    distinctive = _inside_match(_DISTINCTIVE_SCORE, text, start, end)
    if distinctive is not None and distinctive.group("rest").strip():
        return True
    return _inside_match(_WORDED_SCORE, text, start, end) is not None


_PRESSURE_PAIR = re.compile(r"\d{2,3}[ \t]*/[ \t]*\d{2,3}")
_PRESSURE_UNIT_AFTER = re.compile(
    r"[ \t]*(?:mm[ ]?Hg|kPa|cmH2?O|cmH₂O)(?![A-Za-z])", re.IGNORECASE
)
_PRESSURE_LABEL_BEFORE = re.compile(
    r"(?i)(?:blood[ \t]+pressure|(?<![A-Za-z])b/?p|(?<![A-Za-z])bp)[ \t]*[:=]?[ \t]*$"
)


def _is_pressure_pair(text: str, start: int, end: int) -> bool:
    """A systolic/diastolic pair: its pressure unit follows it, or its label precedes it.

    The unit or the label has to be there; a bare "12/03" is a day and a month as
    readily as a pressure and is left to the date rules.
    """
    start, end = _stripped_bounds(text, start, end)
    if not _PRESSURE_PAIR.fullmatch(text[start:end]):
        return False
    return bool(
        _PRESSURE_UNIT_AFTER.match(text, end)
        or _PRESSURE_LABEL_BEFORE.search(text[max(0, start - 24) : start])
    )


def _clinical_quantity_reading(
    text: str, start: int, end: int, category: str
) -> Optional[str]:
    """The rationale when a span is a reading, designator or score."""
    facility = _context_says_facility(text, start, end)
    # "118/74 mmHg" is a pair no calendar contains, in front of a pressure unit.
    if category in _READING_MISREAD_CATEGORIES and _is_pressure_pair(text, start, end):
        return "measurement_value_grammar"
    if category in _MEASUREMENT_MISREAD_CATEGORIES:
        if _is_measurement(text, start, end) and not facility:
            return "measurement_value_grammar"
    elif category in _READING_MISREAD_CATEGORIES:
        if _is_labelled_reading(text, start, end) and not facility:
            return "measurement_value_grammar"
    # A bed number or a named score identifies nobody, whatever category a
    # model assigned: GLiNER called "bay 7" a social security number.
    if _is_clinical_designator(text, start, end):
        return "clinical_designator_grammar"
    # "Bay 6, Room 12": places in a building, read by a model as one identifier.
    # Every word has to be a place word, so a name with a number is not one.
    if _DESIGNATOR_WORD_RUN.fullmatch(
        text[start:end].strip().rstrip(".,;")
    ) and not _designator_continues_into_an_address(text, end):
        return "clinical_designator_grammar"
    if _is_clinical_score(text, start, end):
        return "clinical_score_grammar"
    if category not in _LIST_NUMBER_EXCLUDED and _is_list_marker(text, start, end):
        return "list_marker_grammar"
    return None


#: A number that opens a line and is closed by a bracket or a full stop: "2) ...".
_LIST_MARKER_AFTER = re.compile(r"^[.)][ \t]")
#: A day of the month or an age can open a line too; those are never list numbers.
_LIST_NUMBER_EXCLUDED = frozenset({"DATE", "DATE_TIME", "TIME", "AGE_OVER_89"})


def _is_list_marker(text: str, start: int, end: int) -> bool:
    """A list number, not a postcode or a count: "2) awaiting callback"."""
    start, end = _stripped_bounds(text, start, end)
    if not re.fullmatch(r"\d{1,2}", text[start:end]):
        return False
    line_start = text.rfind("\n", 0, start) + 1
    return not text[line_start:start].strip() and bool(
        _LIST_MARKER_AFTER.match(text[end : end + 2])
    )


def _edge_patterns(
    include_unit_values: bool,
) -> Tuple["re.Pattern[str]", "re.Pattern[str]"]:
    analyte = (
        "(?:" + _alternation(CLINICAL_MEASUREMENT_TERMS | _SPECIMEN_QUALIFIERS) + ")"
    )
    items = [
        _LABELLED_VALUE,
        r"(?:"
        + analyte
        + r"\s+){1,3}[:=]?\s*\d{1,4}(?:[.,]\d+)?(?:\s*"
        + _MEASUREMENT_UNIT
        + ")?",
        _DESIGNATOR_ITEM,
        _SCORE_ITEM,
    ]
    if include_unit_values:
        items.append(_UNIT_VALUE)
    item = "(?:" + "|".join(items) + ")"
    leading = re.compile(r"^[\s,;:|]*" + item + r"(?=[\s,;:|]|$)", re.IGNORECASE)
    trailing = re.compile(r"(?:^|(?<=[\s,;:|]))" + item + r"[\s,;:.|]*$", re.IGNORECASE)
    return leading, trailing


_EDGE_SEPARATORS = " \t\r\n,;:|"

#: A whole token, or a label punctuation mark, at a span's edge.
_EDGE_TOKEN = re.compile(r"[A-Za-z0-9_][\w'\-]*|[:/|;]")
_LABEL_PUNCTUATION = frozenset(":/|;")
#: Words that announce a person. Never trimmed off a span: separating a title
#: from the name it introduces would leave the title's evidence behind.
_TITLE_WORDS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "miss",
        "mx",
        "prof",
        "professor",
        "doctor",
        "nurse",
        "sister",
        "consultant",
        "attending",
        "resident",
        "registrar",
        "patient",
        "provider",
        "surgeon",
        "physician",
    }
)
#: Prepositions that end a heading: "Referral from", "Copy to".
_HEADING_PREPOSITIONS = frozenset({"from", "to", "by", "for", "via", "per", "re"})


def _is_trimmable_label(token: str) -> bool:
    if token in _LABEL_PUNCTUATION:
        return True
    folded = token.casefold()
    return token.isalpha() and folded not in _TITLE_WORDS and is_label_word(folded)


def _is_identifier_token(token: str) -> bool:
    """A token no label word can be part of: "QX7Z4411", "FACILITY_003", "44172290"."""
    digits = sum(character.isdigit() for character in token)
    return digits > 0 and (digits >= 4 or any(c.isalpha() or c == "_" for c in token))


def _is_hard_break(gap: str) -> bool:
    return "\n" in gap or "\t" in gap


def _trim_label_words(text: str, start: int, end: int) -> Tuple[int, int]:
    """Heading and label words cut off a span's edges across a boundary.

    GLiNER drew "Referral from FACILITY_003" as one person, and spans across
    "ALERT <identifier> FLAGGED" or "<identifier> / pharmacy" took the label
    words with the value. A run of label words at an edge is cut off only
    where something separates it from the rest: label punctuation, a tab or
    line break, a heading preposition, or an identifier beside it. Without
    such a boundary the words may be part of the name - "Community Health
    Partners", "Pharmacy Lane" - and the span is left whole. Titles are
    never cut, what is left is still assessed and removed, and a span made
    only of labels is left for the evidence model to judge as a label.
    """
    if is_non_name_label(text[start:end]):
        return start, end
    tokens = list(_EDGE_TOKEN.finditer(text, start, end))
    new_start = start
    has_label_word = False
    for index, token in enumerate(tokens[:-1]):
        word = token.group(0)
        if not _is_trimmable_label(word):
            break
        has_label_word = has_label_word or is_non_name_label(word)
        following = tokens[index + 1]
        if (
            word in _LABEL_PUNCTUATION
            or _is_hard_break(text[token.end() : following.start()])
            or (has_label_word and word.casefold() in _HEADING_PREPOSITIONS)
            or _is_identifier_token(following.group(0))
        ):
            new_start = following.start()
    remaining = [token for token in tokens if token.start() >= new_start]
    new_end = end
    for index in range(len(remaining) - 1, 0, -1):
        token = remaining[index]
        if not _is_trimmable_label(token.group(0)):
            break
        preceding = remaining[index - 1]
        if (
            token.group(0) in _LABEL_PUNCTUATION
            or _is_hard_break(text[preceding.end() : token.start()])
            or _is_identifier_token(preceding.group(0))
        ):
            new_end = preceding.end()
    if not text[new_start:new_end].strip(" \t\r\n:/|;,"):
        return start, end
    return new_start, new_end


def trim_clinical_edges(
    text: str, start: int, end: int, category: str
) -> Tuple[int, int]:
    """A model span without the readings, scores or bed numbers at its edges.

    Models draw one span across a whole line - "BP 132/84 <identifier> HR
    90" - and replacing it removes the vital signs with the identifier. The
    readings at either edge are cut off and the remainder is what gets
    assessed and, if it is an identifier, replaced. The span is never trimmed
    to nothing here: a span that is entirely clinical is left whole for the
    evidence model to preserve or not. Places never lose a unit-only value,
    because a bare number can belong to an address.
    """
    if not 0 <= start < end <= len(text):
        # Not a span of this text. Left exactly as it is, so the overlap
        # resolver drops it instead of this function repairing it into one.
        return start, end
    original = (start, end)
    leading, trailing = _EDGE_PLACE if category in _PLACE_CATEGORIES else _EDGE_ANY
    changed = True
    while changed and start < end:
        changed = False
        segment = text[start:end]
        content = len(segment.rstrip())
        match = leading.match(segment)
        if match and match.end() < content:
            start += match.end()
            changed = True
            continue
        if _VITAL_LABEL_JUST_BEFORE.search(text[max(0, start - 16) : start]):
            match = _BARE_LEADING_READING.match(segment)
            if match and match.end() < content:
                start += match.end()
                changed = True
                continue
        match = trailing.search(segment)
        if match and match.start() > len(segment) - len(segment.lstrip()):
            end = start + match.start()
            changed = True
            continue
        label = _TRAILING_VITAL_LABEL.search(segment)
        if (
            label
            and label.start() > len(segment) - len(segment.lstrip())
            and _VALUE_AFTER_VITAL_LABEL.match(text[end : end + 24])
        ):
            end = start + label.start()
            changed = True
    if start < end:
        start, end = _trim_label_words(text, start, end)
    while start < end and text[start] in _EDGE_SEPARATORS:
        start += 1
    while end > start and text[end - 1] in _EDGE_SEPARATORS:
        end -= 1
    if start >= end:
        # Every part was clinical. The evidence model decides the whole span.
        return original
    return start, end


#: Markers burned into a medical image that carry clinical meaning: laterality,
#: patient orientation, pathology stains/receptors, magnification, scale, and
#: measurements with units. These are the things a blanket "black out every
#: OCR box" destroys, and losing one is not a small loss.
_CLINICAL_MARKER = re.compile(
    r"^(?:"
    r"[LR]|LT|RT|LEFT|RIGHT|AP|PA|LAT|OBL|"
    r"SUPINE|PRONE|ERECT|UPRIGHT|DECUBITUS|PORTABLE|"
    r"H\s*&\s*E|KI-?67|HER2|ER|PR|PD-L1|"
    r"\d+(?:[.,]\d+)?\s*(?:x|×)|"
    r"\d+(?:[.,]\d+)?\s*(?:um|µm|mm|cm|m|kg|g|mg|ml|s|ms|mgy|kvp|mas|hu|t|mt)|"
    r"\d+(?:[.,]\d+)?\s*(?:x|×)\s*\d+(?:[.,]\d+)?\s*(?:mm|cm)?"
    r")$",
    re.IGNORECASE,
)


def is_confirmed_clinical_marker(text: str) -> bool:
    """Whether a short extracted string is recognisably clinical, not a name.

    Deliberately narrow. Anything this does not recognise is *uncertain*, not
    safe: the caller redacts uncertain text and records that it did, so an
    unrecognised marker costs a little utility rather than leaking a value.
    """
    candidate = text.strip().strip(":;,.")
    if not candidate:
        return False
    if _CLINICAL_MARKER.match(candidate):
        return True
    return is_unconditional_clinical_term(
        candidate
    ) or contains_distinctive_clinical_token(candidate)


#: Specimen and sampling qualifiers. Only ever evidence together with the
#: measurement grammar that follows them, never on their own.
_SPECIMEN_QUALIFIERS = frozenset(
    {
        "serum",
        "plasma",
        "venous",
        "arterial",
        "capillary",
        "blood",
        "whole",
        "urine",
        "urinary",
        "fasting",
        "random",
        "spot",
        "peak",
        "trough",
        "cord",
        "sputum",
        "stool",
        "csf",
        "cerebrospinal",
        "synovial",
        "pleural",
        "peritoneal",
        "ascitic",
        "baseline",
        "repeat",
        "salivary",
    }
)
_WORD_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")
#: Up to two words, then a value and a unit: "potassium 4.1 mmol/L".
_VALUE_WITHIN_TWO_WORDS = re.compile(
    r"^\s+(?:[A-Za-z][\w\-/]*\s+){0,2}[:=]?\s*\d+(?:[.,]\d+)?\s*"
    r"(?:%|[a-zµ]+(?:/[a-z]+)?)",
    re.IGNORECASE,
)

_EDGE_ANY = _edge_patterns(include_unit_values=True)
_EDGE_PLACE = _edge_patterns(include_unit_values=False)


def _context_says_specimen(text: str, start: int, end: int) -> bool:
    """A specimen qualifier heading a measurement or a test.

    "Serum potassium 4.1 mmol/L" and "Urine microscopy showed no casts". The
    qualifier list is closed but never decides alone: the word after it has
    to be a known analyte or a laboratory test, or a value with a unit has to
    follow, so a surname standing in front of an analyte is not a qualifier
    and is not protected.
    """
    from services.clinical_vocabulary import LABORATORY_TEST_NAMES

    words = [word.casefold() for word in _WORD_TOKEN.findall(text[start:end])]
    if not words or not any(word in _SPECIMEN_QUALIFIERS for word in words):
        return False
    if not all(
        word in _SPECIMEN_QUALIFIERS
        or word in CLINICAL_MEASUREMENT_TERMS
        or word in LABORATORY_TEST_NAMES
        for word in words
    ):
        return False
    if any(word in LABORATORY_TEST_NAMES for word in words):
        return True
    after = text[end : end + 48]
    following = _WORD_TOKEN.match(after.lstrip())
    if following and (
        following.group(0).casefold() in CLINICAL_MEASUREMENT_TERMS
        or following.group(0).casefold() in LABORATORY_TEST_NAMES
    ):
        return True
    return bool(_VALUE_WITHIN_TWO_WORDS.match(after))


_TOKEN_SHAPE = re.compile(
    r"<[A-Z][A-Z_]*_[1-9][0-9]*>|\b[A-Z][A-Z]*(?:_[A-Z]+)*_\d{3,}\b"
)
_TITLE_WORD = re.compile(
    r"^(?:dr|mr|mrs|ms|miss|mx|prof|professor|doctor|sir|dame|nurse|"
    r"sister|rev|fr)\.?$",
    re.IGNORECASE,
)


def _name_beside(text: str, start: int, end: int) -> bool:
    """Whether the word directly before or after the span could be part of a name.

    Name-like: a replacement-token shape (whoever wrote it: "<PERSON_1>
    Inhaler" may be "Alice Inhaler" with the given name already replaced), a
    title, or a capitalised word that is not itself clinical vocabulary.
    Sentence punctuation, line breaks and a clinical action verb before the
    span ("Given PCV15") end the neighbourhood.
    """
    before = re.search(r"(\S+)([ \t]+)$", text[max(0, start - 48) : start])
    after = re.match(r"([ \t]+)(\S+)", text[end : end + 48])
    words = []
    if (
        before
        and not before.group(1).endswith((".", ":", ";", "!", "?", ",", ")"))
        and before.group(1).casefold() not in _CLINICAL_ACTION_VERBS
    ):
        words.append(before.group(1))
    if after:
        words.append(after.group(2).rstrip(".,;:!?)"))
    for word in words:
        if _TOKEN_SHAPE.search(word) or _TITLE_WORD.match(word):
            return True
        if word[:1].isupper() and not (
            is_clinical_product_term(word)
            or is_unconditional_clinical_term(word)
            or is_non_name_label(word)
        ):
            return True
    return False


def clinical_reading_strength(text: str, start: int, end: int) -> Tuple[str, str]:
    """How strongly the surrounding text reads as clinical rather than naming.

    Returns (reading, rationale). The vocabulary contributes evidence here; it
    does not decide alone, and it is checked *after* the naming context so a
    listed eponym used as a surname is not shielded.
    """
    span = text[start:end]
    if (
        start == 0
        and span == "SYNTHETIC"
        and (
            re.match(r"^\s+[a-z][a-z -]{2,48}\s+encounter\s*[;:,.]", text[end:])
            or re.match(
                r"^\s+(?:[a-z]+\s+){0,2}(?:record|note|visit|case|summary|chart"
                r"|report|entry|document|dataset|data)s?\s*[;:,.]",
                text[end:],
            )
        )
    ):
        # Only the opening provenance marker of a note: the word as written
        # in capitals at offset 0, then a lowercase description of the
        # document and its terminator ("SYNTHETIC record."). The same word
        # following "Patient", before a name or inside an identifier is not
        # exempt, and nothing else in the note is.
        return "clinical", "synthetic_provenance_marker"
    if (
        re.fullmatch(r"(?i:unicode|ascii|utf-?8|latin-?1)", span.strip())
        and re.match(r"\s+alias\s*:", text[end:], re.IGNORECASE)
        and re.search(r"(?:^|[;\n])\s*$", text[:start])
    ):
        # Encoding labels in a note describe how the following alias is
        # written. The alias itself is still assessed as a person's name.
        return "clinical", "encoding_alias_field_label"
    if re.search(r"(?:^|[;\n])\s*$", text[:start]) and re.match(
        r"\s+(?:MRN|medical\s+record\s+number)\b", text[end:], re.IGNORECASE
    ):
        from services.clinical_vocabulary import is_demographic_category_label

        if is_demographic_category_label(span.strip()):
            # A demographic category immediately governing a record-number
            # label is not the holder's name. A person's name elsewhere in
            # the sentence still receives normal identifier scrutiny.
            return "clinical", "demographic_category_before_record_label"
    if span == "housemate" and re.match(r"\s+[A-ZÀ-ÖØ-Þ]", text[end:]):
        # The lower-case relationship noun introduces the following name;
        # a capitalized Housemate inside that name gets no such exemption.
        return "clinical", "relationship_role_noun"

    # Layer 4 first: naming context beats the dictionary. A reporting verb
    # after a role or staff-group phrase is what staff do, not evidence of a
    # personal name - "Night nursing team reported" names nobody. A title or
    # "seen by" in front of the span still marks a person, whatever it says.
    before = text[max(0, start - 48) : start]
    after = text[end : end + 48]
    # A field label alone on the line after "Patient:" is what a redacted
    # page leaves when the value between them was cut out ("Patient:\nNHS").
    # On the same line, or with any other word in the span, context wins.
    orphaned_label = before.rstrip(" \t").endswith(("\n", "\r")) and is_non_name_label(
        span
    )
    # "MR KNEE": upper-case MR before upper-case anatomy is the modality.
    titled = _PERSON_CONTEXT_BEFORE.search(before) and not modality_before(text, start)
    if (titled and not orphaned_label) or (
        _PERSON_CONTEXT_AFTER.match(after) and not is_non_name_label(span)
    ):
        return "identifier", "person_context_present"
    if _RELATIONSHIP_BEFORE.search(before) and _person_name_shaped(span):
        # A named household member or relative is a person even when a
        # clinical-domain tagger gives a Unicode name no PROPN token. The
        # name shape keeps "relative risk" and other ordinary phrases out.
        return "identifier", "relationship_context"
    if _context_says_facility(text, start, end):
        return "identifier", "facility_context_present"
    # "Thyroid clinic": a service noun after clinical subject words names a
    # department, not an organisation; the shared policy decides it below.
    if (
        _has_organisation_designator(
            span, sentence_initial=_is_sentence_initial(text, start)
        )
        and clinical_service_region(text, start, end) is None
    ):
        return "identifier", "organisation_designator_present"

    # A lab measurement's specimen. Checked after naming context, so a title
    # or a facility construction in front of the word still wins.
    # A name-shaped span ("John Smith", "Tarleton-Achebe") is never kept by a
    # clinical grammar because a dose or a result happens to follow it.
    if _context_says_specimen(text, start, end) and not _person_name_shaped(span):
        return "clinical", "specimen_measurement_grammar"

    # An analyte, its value and its unit, with no dependence on the analyte
    # being a word anybody listed. Also after naming context, for the same
    # reason: a title in front of the span still marks a person.
    if _is_analyte_measurement(text, start, end) and not _person_name_shaped(span):
        return "clinical", "analyte_measurement_grammar"

    # The shared clinical-preservation policy: a unit after a result, a stain
    # in pathology context, a sequence in imaging context - each refused when
    # a label, title, token or name-like word touches it. The audit reaches
    # this same function, so it re-proves the decision on the artifact.
    structural = assess_clinical_span(text, start, end)
    if structural.preserve:
        return "clinical", structural.reason

    # A vaccine, vaccine code or clinical device on the closed list, with no
    # name beside it. After naming context, so "Dr. Inhaler" or "administered
    # by Influenza Smith" still read as a person.
    if is_clinical_product_term(span) and not _name_beside(text, start, end):
        return "clinical", "clinical_product_term_without_name_beside_it"

    # Vocabulary and grammar as evidence. Role phrases and code labels first:
    # their structure is the evidence, and a clinical word inside one
    # ("Respiratory physiotherapist") must not demote it to a word reading.
    if is_staff_role_phrase(span):
        return "clinical", REASON_STAFF_ROLE_PHRASE
    if is_clinical_finding_term(span) and _CLINICAL_RESPONSE_AFTER.match(after):
        return "clinical", REASON_CLINICAL_RESPONSE
    if (is_label_word(span) or _CODE_BODY_ACRONYM.fullmatch(span)) and (
        _LABELLED_CODE_AFTER.match(text, end) or _TOKEN_AFTER.match(text, end)
    ):
        return "clinical", REASON_LABEL_OF_CODE
    if _fills_clinical_field(text, start, end):
        return "clinical", REASON_CLINICAL_FIELD_VALUE
    if is_unconditional_clinical_term(span):
        return "clinical", "term_never_a_name"
    if _context_says_anatomy(text, start, end):
        return "clinical", "anatomical_position_grammar"
    if _context_says_dosage(text, start, end) and not _person_name_shaped(span):
        # Checked before the vocabulary so an unseen medication is protected
        # by the construction rather than by having been listed.
        return "clinical", "dosage_construction_grammar"
    if contains_distinctive_clinical_token(span):
        return "clinical", "contains_drug_or_analyte_token"
    if is_non_name_label(span):
        return "clinical", "field_label_or_role_noun"
    if _context_says_clinical(text, start, end) and not _token_beside(text, start, end):
        # A token-shaped word beside the span may stand for a replaced given
        # name ("<PERSON_1> Movat stain"), so the construction proves nothing.
        # Grammar alone, no dictionary needed: "<Word>'s syndrome" is a
        # condition whether or not the eponym is one we have heard of. This is
        # what removes the dependence on the closed list.
        return "clinical", "clinical_construction_grammar"
    if is_clinical_eponym_use(span, re.findall(r"\S+", text[end : end + 64])[:4]):
        return "clinical", "recorded_eponym_in_clinical_use"

    return "ambiguous", "no_decisive_evidence"


#: Laterality and position words. A word they govern reads as an anatomical
#: site: names do not take laterality ("the left Smith").
_ANATOMICAL_POSITION = frozenset(
    {
        "left",
        "right",
        "bilateral",
        "both",
        "upper",
        "lower",
        "medial",
        "lateral",
        "anterior",
        "posterior",
        "proximal",
        "distal",
        "superior",
        "inferior",
        "dorsal",
        "ventral",
        "palmar",
        "plantar",
    }
)
#: "left" and "right" are also verbs and "both" a pronoun, so they govern a
#: site only across at least one lowercase modifier - "left iliac crest",
#: never "left rukmini" - or when an article or possessive shows they are
#: adjectives: "over the left snuffbox", "her right knee".
_POSITION_NEEDS_MODIFIER = frozenset({"left", "right", "both"})
#: Words that make the following "left" or "right" an adjective. Demonstratives
#: are absent: "the nurse that left rukmini" uses "that" as a relative pronoun.
_ADJECTIVE_MAKERS = frozenset(
    {"the", "a", "an", "his", "her", "their", "its", "my", "your", "our"}
)
_WORDS_BEFORE = re.compile(r"((?:[A-Za-z][A-Za-z\-]*[ \t]+){1,4})$")


def _context_says_anatomy(text: str, start: int, end: int) -> bool:
    """A lowercase span governed by a laterality or position word.

    Grammar rather than an anatomy list, so an unlisted structure - a crest,
    a malleolus, a snuffbox - is protected by its construction. Restricted to
    lowercase spans, and blocked by any punctuation or line break between the
    position word and the span, because a name is capitalised and a sentence
    boundary ends the construction.
    """
    span = text[start:end].strip()
    if not span or span != span.lower():
        return False
    tail = _WORDS_BEFORE.search(text[max(0, start - 80) : start])
    if not tail:
        return False
    words = tail.group(1).split()
    for index, word in enumerate(words):
        lowered = word.casefold()
        if lowered not in _ANATOMICAL_POSITION:
            continue
        between = words[index + 1 :]
        if not all(other.islower() for other in between):
            continue
        if lowered in _POSITION_NEEDS_MODIFIER and not between:
            adjective = index > 0 and words[index - 1].casefold() in _ADJECTIVE_MAKERS
            if not adjective:
                continue
        return True
    return False


def _span_is_labels_and_codes(text: str, start: int, end: int) -> bool:
    """Whether a name-shaped span contains nothing that could be a name.

    Preserving the "Dx" label stopped it being replaced, and the model then
    proposed the wider span "Dx: K80.20" as a person instead. Nothing in that
    span is a person: one token heads a field, the other is a well-formed
    diagnosis code, and there is no third thing.

    So the test is exhaustive rather than indicative - every token has to be
    accounted for as a label, a clinical code, a number or punctuation. One
    unexplained word and the rule declines, which is what keeps "Dr Baker:
    K80.20" and "Isolde: 12" out of it.
    """
    span = text[start:end].strip()
    if not span:
        return False
    tokens = [t for t in re.split(r"[\s:;,()\[\]/]+", span) if t]
    if not tokens:
        return False
    accounted = 0
    for token in tokens:
        bare = token.strip(".,;:-")
        if not bare:
            accounted += 1
            continue
        if is_clinical_field_label(bare):
            accounted += 1
            continue
        # A standard race/ethnicity category word is a category, not a name,
        # exactly like a field label: "Asian MRN" holds no name. The category
        # must be a whole closed-vocabulary label; any other word declines.
        if is_race_ethnicity_category(bare):
            accounted += 1
            continue
        # A word that only labels an identifier field ("SSN", "iris scan",
        # "postal code") carries no identity. The value it announces is a
        # separate span; a code or an unlisted word declines the whole span.
        if is_identifier_label_text(bare):
            accounted += 1
            continue
        if _ICD_DIAGNOSIS_CODE.fullmatch(bare):
            accounted += 1
            continue
        # Nothing else is accounted for - and in particular not "a token with
        # a digit in it". That was the first version of this rule, and the
        # suite caught it: "Record: RT4420518" and "Account: ACR520563" are a
        # label followed by a record number, every token has a digit, and the
        # rule would have declared them free of anything worth redacting.
        # Only a diagnosis code earns the exemption, because only a diagnosis
        # code has a grammar narrow enough to be sure of.
        return False
    return accounted == len(tokens)


#: The last word of a field that holds a contact detail or a code: "Workstation
#: address:", "GMC reg:", "Reference no.:".
_LABEL_NOUNS = frozenset(
    {
        "address",
        "number",
        "no",
        "id",
        "identifier",
        "phone",
        "telephone",
        "tel",
        "fax",
        "email",
        "reg",
        "registration",
        "licence",
        "license",
        "reference",
        "ref",
        "code",
        "pin",
        "extension",
        "ext",
        "bleep",
        "pager",
        "mobile",
    }
)


def _collapse_spaced_letters(span: str) -> str:
    """Letters written one at a time are one word: "N M C  r e g" reads "NMC reg".

    Single letters separated by one space join; a gap of two or more spaces
    separates words, as an OCR engine or a typist spacing out a label leaves it.
    """
    words = []
    for part in re.split(r"[ \t]{2,}", span.strip()):
        merged: List[str] = []
        run: List[str] = []
        for token in part.split():
            bare = token.rstrip(".")
            if len(bare) == 1 and bare.isalpha():
                run.append(bare)
                continue
            if run:
                merged.append("".join(run))
                run = []
            merged.append(token)
        if run:
            merged.append("".join(run))
        words.append(" ".join(merged))
    return " ".join(words)


def _is_label_noun_phrase(text: str, start: int, span: str) -> bool:
    """A short phrase that opens its line and ends in a label noun.

    The key of a key-value line, not its value: a model reads "Workstation
    address" as an address because of the word, though the value follows the
    colon. It has to begin the line and be letters only, so it is the key.
    """
    span = _collapse_spaced_letters(span)
    if not re.fullmatch(r"[^\W\d_]+\.?(?:[ \t]+[^\W\d_]+\.?){0,3}", span):
        return False
    if span.split()[-1].rstrip(".").casefold() not in _LABEL_NOUNS:
        return False
    line_start = text.rfind("\n", 0, start) + 1
    return not text[line_start:start].strip(" \t")


def _identifier_label_words() -> frozenset:
    """The words that end the label of a keyed identifier, from the one specification."""
    words = set(_LABEL_NOUNS) | {
        word.rstrip(".").casefold() for word in identifier_spec.QUALIFIERS
    }
    for keyed in identifier_spec.KEYED_IDENTIFIERS:
        words.update(head.split()[-1].casefold() for head in keyed.heads)
    for label in identifier_spec.PROFESSIONAL_REGISTRATION_LABELS:
        words.add(label.split()[-1].casefold())
    return frozenset(words)


def _identifier_label_vocabulary() -> frozenset:
    """Every word a field label may be made of: the words that end one, and their modifiers."""
    words = set(_identifier_label_words()) | _LABEL_MODIFIERS
    for keyed in identifier_spec.KEYED_IDENTIFIERS:
        for head in keyed.heads:
            words.update(word.casefold() for word in re.findall(r"[^\W\d_]+", head))
    for label in identifier_spec.PROFESSIONAL_REGISTRATION_LABELS:
        words.update(word.casefold() for word in label.split())
    return frozenset(words)


#: Words that qualify a label's last word without being an identifier head: "driver
#: licence", "pacemaker serial". Closed on purpose, so a surname before "MRN" is not one.
_LABEL_MODIFIERS = frozenset(
    {
        "driver",
        "driving",
        "pacemaker",
        "vehicle",
        "passport",
        "social",
        "security",
        "national",
        "tax",
        "bank",
        "credit",
        "card",
        "certificate",
        "plate",
        "employee",
        "student",
        "voter",
        "insurance",
        "group",
        "health",
        "identity",
        "identification",
        "medical",
        "patient",
        "hospital",
        "device",
        "implant",
        "pump",
        "stimulator",
        "lead",
    }
)
_IDENTIFIER_LABEL_WORDS = _identifier_label_words()
_IDENTIFIER_LABEL_VOCABULARY = _identifier_label_vocabulary()
_LABEL_BEFORE_TOKEN = re.compile(
    r"(?P<label>[^\W\d_]+\.?(?:[ \t]+[^\W\d_]+\.?){0,3})[ \t]*(?:#|:)?", re.IGNORECASE
)
_GENERATED_TOKEN_AT_START = re.compile(r"<[A-Z][A-Z_]*_[1-9][0-9]*>")
_TOKEN_AFTER_LABEL = re.compile(r"[ \t]*(?:#|:)?[ \t]*<[A-Z][A-Z_]*_[1-9][0-9]*>")


def _label_of_a_replaced_value(text: str, start: int, end: int) -> bool:
    """Whether the span is only the label in front of a value that was replaced.

    The keyed-identifier rules replace the value and leave its label, so the
    output reads "Account # <ACCOUNT_ID_1>". A model reads that pair as a name
    or a licence because of the label's words. The label says what field it
    was and identifies nobody: a short, letters-only phrase that ends in one
    of the words the specification uses to announce a value, directly followed
    by a generated token, and the token itself when the span reaches it.
    """
    span = text[start:end]
    token = _GENERATED_TOKEN_AT_START.search(span)
    if token:
        if span[token.end() :].strip():
            return False
        label = span[: token.start()]
    else:
        label = span
        if not _TOKEN_AFTER_LABEL.match(text[end : end + 40]):
            return False
    return is_identifier_label_phrase(label)


def is_identifier_label_phrase(phrase: str) -> bool:
    """A short letters-only phrase whose words announce an identifier ("Medical record number").

    Every word comes from the closed label vocabulary and the last one is a word
    the specification uses to announce a value; a name before it ("Black MRN")
    makes it something else. Used where a label is kept beside a replaced value,
    and where the same label has to be covered along with that value in pixels.
    """
    match = _LABEL_BEFORE_TOKEN.fullmatch((phrase or "").strip())
    if match is None:
        return False
    words = [word.rstrip(".").casefold() for word in match.group("label").split()]
    return words[-1] in _IDENTIFIER_LABEL_WORDS and all(
        word in _IDENTIFIER_LABEL_VOCABULARY for word in words
    )


def _inside_label_key(text: str, start: int, end: int) -> bool:
    """Whether the span lies within the key of a "Key: value" line.

    A model may mark only part of the key ("GMC" in "GMC reg:"). The key as a
    whole is what is judged: a short, letters-only phrase that opens the line,
    ends in a label noun and is followed by its colon.
    """
    # A model may draw the span with the key's own colon in it ("Pager no.:").
    end = min(end, len(text))
    while end > start and text[end - 1] in " \t:":
        end -= 1
    line_start = text.rfind("\n", 0, start) + 1
    colon = text.find(":", end)
    if colon < 0 or colon - line_start > 48 or "\n" in text[line_start:colon]:
        return False
    key = text[line_start:colon]
    key_start = line_start + len(key) - len(key.lstrip(" \t"))
    return (
        key_start <= start
        and end <= colon
        and _is_label_noun_phrase(text, key_start, key.strip())
    )


def _is_clinical_field_label(text: str, start: int, end: int) -> bool:
    """Whether this span is a field heading rather than somebody's name.

    Two halves of evidence, and neither is enough alone. The word has to be
    one that heads a field in a clinical record, and it has to be standing
    where a heading stands - immediately before a colon, with nothing but
    space between. "Baker" is a surname; "Baker:" at the head of a line is a
    label, and only the second form reaches this.

    It matters because the label carries meaning the values depend on.
    Replacing "Dx" with a person surrogate did not only lose the word: it
    removed the evidence that the ICD code beside it was a diagnosis, and the
    code was then read as an account number and blocked the document.
    """
    span = text[start:end].strip()
    if span and _inside_label_key(text, start, end):
        return True
    if not span or not (
        is_clinical_field_label(span) or _is_label_noun_phrase(text, start, span)
    ):
        return False
    # Label position: a colon follows, allowing only spaces between.
    after = text[end : end + 4]
    if not after.lstrip(" ").startswith(":"):
        return False
    # Naming context is already excluded by the branch that calls this, so a
    # label introduced by "Dr" or "Mr" never reaches here and needs no second
    # test of its own.
    return True


def assess_finding(
    text: str,
    start: int,
    end: int,
    category: str,
    source: str,
    agreeing_detectors: int = 1,
    clinical_support: bool = False,
    contains_proper_noun: Optional[bool] = None,
    opening_modifier: bool = False,
    opening_unconfirmed: bool = False,
    occurrence_support: Optional[Callable[[], Optional[Tuple[int, int]]]] = None,
) -> EvidenceAssessment:
    """Decide what to do with one proposed span, by evidence strength.

    ``occurrence_support`` is asked only when nothing else justifies replacing the
    span: it returns how many of the span's occurrences in the text were claimed.
    """
    evidence = evidence_type(source)

    # Layer 0: some categories are not identifiers at all. No amount of model
    # confidence turns a diagnosis into one of the eighteen.
    if category in NON_IDENTIFIER_CATEGORIES:
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "clinical",
            "preserve",
            "category_is_not_a_safe_harbor_identifier",
            agreeing_detectors,
        )

    # A single character carries no identity on its own - the "C" in "37.2°C"
    # was being replaced with a patient surrogate. An initial in a naming
    # construction ("Patient R.") is still caught, because the context check
    # below runs on it like any other span. Sentence punctuation is not part of
    # the character: the "L." ending "mmol/L." is still one letter.
    if (
        category in NAME_SHAPED_CATEGORIES
        and len(text[start:end].strip().strip(".,;:")) <= 1
        and not _context_says_person(text, start, end)
    ):
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "clinical",
            "preserve",
            "single_character_span_carries_no_identity",
            agreeing_detectors,
        )

    # Readings, bed designators and clinical scores, whatever a model called
    # them. Checked before Layer 1 because a model's category claim is not a
    # regex match: "HR 90" labelled a health-plan number carries digits, but
    # it is a heart rate, and however many detectors agree it stays one. A
    # deterministic match never reaches this, and naming context still wins.
    if evidence != EVIDENCE_DETERMINISTIC and not _context_says_person(
        text, start, end
    ):
        if _is_clinical_field_label(text, start, end):
            return EvidenceAssessment(
                category,
                source,
                evidence,
                "clinical",
                "preserve",
                "clinical_field_label",
                agreeing_detectors,
            )
        if category in NAME_SHAPED_CATEGORIES and _span_is_labels_and_codes(
            text, start, end
        ):
            return EvidenceAssessment(
                category,
                source,
                evidence,
                "clinical",
                "preserve",
                "span_contains_no_name_token",
                agreeing_detectors,
            )
        if _is_unit_fragment(text, start, end):
            return EvidenceAssessment(
                category,
                source,
                evidence,
                "clinical",
                "preserve",
                "unit_expression_fragment",
                agreeing_detectors,
            )
        if _is_measurement_unit(text, start, end):
            return EvidenceAssessment(
                category,
                source,
                evidence,
                "clinical",
                "preserve",
                "measurement_unit_in_use",
                agreeing_detectors,
            )
        if category not in NAME_SHAPED_CATEGORIES and _fills_equipment_field(
            text, start, end
        ):
            # "Model: SL-40" names a product, not a device: an identifier
            # there would be a serial number, which this shape is not.
            return EvidenceAssessment(
                category,
                source,
                evidence,
                "clinical",
                "preserve",
                "equipment_model_designation",
                agreeing_detectors,
            )
        grammar = _clinical_quantity_reading(text, start, end, category)
        if grammar is not None:
            return EvidenceAssessment(
                category,
                source,
                evidence,
                "clinical",
                "preserve",
                grammar,
                agreeing_detectors,
            )
        # A bare number judged where it stands: a unit, an imaging parameter,
        # a dose, a score, a year or a time in its own sentence makes it a
        # reading, whatever label a model gave it and whatever the same
        # digits are elsewhere in the file. Only this occurrence is read.
        if (
            _NUMBER_ONLY.fullmatch(text[start:end])
            and not _JOINED_TO_CODE.search(text[max(0, start - 1) : start])
            and not _JOINED_TO_CODE_AFTER.match(text, end)
            and numeric_context.classify(text, start, end).reason in _NUMERIC_READINGS
        ):
            return EvidenceAssessment(
                category,
                source,
                evidence,
                "clinical",
                "preserve",
                "numeric_context_clinical",
                agreeing_detectors,
            )
        if category in _TEMPORAL_CATEGORIES and _is_duration(text, start, end):
            return EvidenceAssessment(
                category,
                source,
                evidence,
                "clinical",
                "preserve",
                "duration_of_time_not_a_date",
                agreeing_detectors,
            )
        if category in _TEMPORAL_CATEGORIES and _is_treatment_followup_interval(
            text, start, end
        ):
            return EvidenceAssessment(
                category,
                source,
                evidence,
                "clinical",
                "preserve",
                "treatment_followup_interval_not_calendar_date",
                agreeing_detectors,
            )
        if category in _TEMPORAL_CATEGORIES and _is_relative_offset(text, start, end):
            return EvidenceAssessment(
                category,
                source,
                evidence,
                "clinical",
                "preserve",
                "relative_time_offset_not_a_date",
                agreeing_detectors,
            )
        # A model can call an ICD diagnosis an ``IDENTIFIER``. Preserve only
        # the complete ICD-shaped token immediately governed by an explicit
        # diagnosis field. Deterministic identifier matches never reach this
        # exemption, and an ICD-looking value after MRN/account/patient labels
        # receives no protection.
        token_start, token_end = _enclosing_token(text, start, end)
        candidate = text[token_start:token_end].strip().strip(",;.")
        before = text[max(0, token_start - 32) : token_start]
        if _governed_diagnosis_field(text, start, end) or (
            _ICD_DIAGNOSIS_CODE.fullmatch(candidate)
            and _DIAGNOSIS_FIELD_BEFORE.search(before)
        ):
            return EvidenceAssessment(
                category,
                source,
                evidence,
                "clinical",
                "preserve",
                "explicit_diagnosis_code_field",
                agreeing_detectors,
            )

    # A capitalisation-driven reading of a heading. The spaCy tagger and the
    # proper-noun fallback find names by their capitals, and in a banner or a
    # title every word has them, so neither has evidence there. Naming context
    # still wins, and the pinned models, which do not read case alone, are
    # assessed on their own claims.
    if (
        source in _CASE_DRIVEN_SOURCES
        and category in NAME_SHAPED_CATEGORIES
        and not _context_says_person(text, start, end)
        and line_carries_no_case_evidence(_enclosing_line(text, start, end))
    ):
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "clinical",
            "preserve",
            "heading_capitalisation_carries_no_case_evidence",
            agreeing_detectors,
        )

    # "Page 5", "Pg 2 of 7": the word that labels a page number is structure
    # whatever stands beside it, including a replacement token before it.
    if (
        evidence != EVIDENCE_DETERMINISTIC
        and category in NAME_SHAPED_CATEGORIES
        and text[start:end].strip().casefold() in _PAGE_LABELS
        and _PAGE_NUMBER_AFTER.match(text, end)
    ):
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "clinical",
            "preserve",
            "page_number_label",
            agreeing_detectors,
        )

    # Letters spelled out one by one ("G M C", "N.H.S.") are an acronym, as OCR
    # and typists write it, not a name. Naming context still wins.
    if (
        evidence != EVIDENCE_DETERMINISTIC
        and category in NAME_SHAPED_CATEGORIES
        and _SPELLED_LETTERS.fullmatch(text[start:end].strip())
        and not _context_says_person(text, start, end)
    ):
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "clinical",
            "preserve",
            "spelled_out_acronym_not_a_name",
            agreeing_detectors,
        )

    # A title that ends in the name of a kind of document ("Referral Form",
    # "Discharge Summary", "Source Bundle"): nobody's name ends that way, and
    # whichever detector read it as one was reading the typography of a
    # heading. Naming context still wins.
    if (
        evidence != EVIDENCE_DETERMINISTIC
        and category in NAME_SHAPED_CATEGORIES
        # A pinned model's claim is excused only for a title built wholly from
        # heading vocabulary: "Jordan Fictional Report" ends in a document word
        # too, and has a person in it. A capitalisation-driven source has no
        # evidence about the other words, so the document word alone is enough.
        and (
            _is_document_title(text[start:end])
            or (
                source in _CASE_DRIVEN_SOURCES
                and _ends_in_document_noun(text[start:end])
            )
        )
        # The whole line, so it is a heading: in running prose a name span that
        # happens to take the next word ("Vantreske Plan") is still a name.
        and text[start:end].strip() == _enclosing_line(text, start, end).strip()
        and not _context_says_person(text, start, end)
    ):
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "clinical",
            "preserve",
            "document_title_noun_not_a_name",
            agreeing_detectors,
        )

    # A whole line of capitalised words that only a capitalisation-driven source
    # calls a name, while both pinned models read the text and neither did. The
    # shape of two or three capitalised words is how a name is written and how a
    # heading is written, and nothing here says which, so it is replaced and the
    # file goes to a person instead of releasing an edit that rests on shape.
    if (
        opening_unconfirmed
        and evidence != EVIDENCE_DETERMINISTIC
        and source in _CASE_DRIVEN_SOURCES
        and category in NAME_SHAPED_CATEGORIES
        and agreeing_detectors < 2
        and _is_whole_line_title_case(text, start, end)
        and not _context_says_person(text, start, end)
    ):
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "ambiguous",
            "redact",
            "whole_line_title_case_unconfirmed_by_pinned_models",
            agreeing_detectors,
            justified=False,
        )

    # A heading that is itself the claimed organisation or place. A model's
    # reading of a whole title-case line as a body has no designator in it
    # ("Ltd", "Hospital", "Clinic" at the end) and no naming construction; the
    # only evidence offered is that its words are capitalised, which a heading's
    # always are.
    if (
        evidence != EVIDENCE_DETERMINISTIC
        and category in ("ORGANIZATION", "FACILITY", "LOCATION", "ADDRESS")
        and line_carries_no_case_evidence(_enclosing_line(text, start, end))
        and text[start:end].strip() == _enclosing_line(text, start, end).strip()
        and not _has_organisation_designator(text[start:end], sentence_initial=True)
        and not _context_says_person(text, start, end)
        and not _context_says_facility(text, start, end)
        and not _ADDRESS_STRUCTURE.search(text[start:end])
    ):
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "clinical",
            "preserve",
            "heading_line_without_designator",
            agreeing_detectors,
        )

    # A word that opens a line or a sentence and modifies a common noun that
    # follows it ("Metabolic screen raised"), claimed only by the capitalisation
    # detectors with no pinned model agreeing. Its capital is explained by its
    # position and the tagger reads a modifier-plus-noun construction, so the
    # claim has no evidence in it. A pinned model reading the same span still
    # decides it, and a word followed by a verb, a colon or a capital is never
    # an opening modifier (the caller proved the construction from the parse).
    # The same holds, whatever follows the word, when both pinned models read
    # the text and neither named this span: ``opening_unconfirmed``. With no
    # pinned model to ask - the legacy test chain - nothing is excused.
    if (
        (opening_modifier or opening_unconfirmed)
        and source in _CASE_DRIVEN_SOURCES
        and agreeing_detectors < 2
        and category in NAME_SHAPED_CATEGORIES
        and capitals_only_at_openings(text, start, end)
        and not _context_says_person(text, start, end)
    ):
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "clinical",
            "preserve",
            "opening_word_claimed_by_capitalisation_alone",
            agreeing_detectors,
        )

    # A clinic, unit or service named by subject words the tagger reads as
    # ordinary nouns ("Genetics clinic") is a department, not an organisation.
    # A proper noun in it ("Rivermead clinic") is the opposite and is replaced.
    if (
        category in ("ORGANIZATION", "FACILITY")
        and evidence != EVIDENCE_DETERMINISTIC
        and contains_proper_noun is False
        and _service_unit_phrase(text, start, end)
        and not _context_says_person(text, start, end)
        and not _context_says_facility(text, start, end)
    ):
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "clinical",
            "preserve",
            "service_unit_without_proper_noun",
            agreeing_detectors,
        )

    # A run of designator-and-number pairs ("Bed 14, Cubicle 3, Ward 5") is a
    # list of places in a ward, not a name; names do not take such numbers.
    if (
        evidence != EVIDENCE_DETERMINISTIC
        and category in NAME_SHAPED_CATEGORIES
        and (
            _DESIGNATOR_RUN.fullmatch(text[start:end].strip())
            or _DESIGNATOR_WORD_RUN.fullmatch(text[start:end].strip())
        )
        and not _designator_continues_into_an_address(text, end)
        and not _context_says_person(text, start, end)
    ):
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "clinical",
            "preserve",
            "designator_number_run_not_a_name",
            agreeing_detectors,
        )

    # A time of day with no date on its line is not an element of a date, so
    # it is clinical sequencing ("glucose at 14:00"), whichever detector found
    # it. A time beside a date keeps the handling it had.
    if category in _TEMPORAL_CATEGORIES and is_undated_clock_time(text, start, end):
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "clinical",
            "preserve",
            "clock_time_without_a_date_is_not_a_date_element",
            agreeing_detectors,
        )

    # The label left in front of a value that was replaced is a field name, not
    # an identifier, whichever category a model gave the pair.
    if evidence != EVIDENCE_DETERMINISTIC and _label_of_a_replaced_value(
        text, start, end
    ):
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "clinical",
            "preserve",
            "label_of_a_replaced_value",
            agreeing_detectors,
        )

    # Layer 1: exact structured identifiers always win. A deterministic source
    # is the regex itself and needs no further proof; a model claiming one of
    # these categories has to produce a span that looks like one.
    span = text[start:end]
    if evidence == EVIDENCE_DETERMINISTIC or (
        category in EXACT_STRUCTURED_CATEGORIES
        and _has_structural_syntax(category, span)
    ):
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "identifier",
            "redact",
            "exact_structured_match",
            agreeing_detectors,
        )

    if (
        evidence != EVIDENCE_DETERMINISTIC
        and category in EXACT_STRUCTURED_CATEGORIES
        and _SOURCE_TOKEN_SHAPE.fullmatch(span.strip())
    ):
        # Input text shaped like one of the sanitizer's own tokens is
        # untrusted: it is never accepted as already removed, and replacing it
        # costs nothing clinical. A digit or two in it is no reason to escalate.
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "identifier",
            "redact",
            "token_shaped_source_text",
            agreeing_detectors,
        )

    if evidence != EVIDENCE_DETERMINISTIC and category in EXACT_STRUCTURED_CATEGORIES:
        # A model's structured claim whose span failed that category's
        # syntax check. It still says "identifier", and the rules below
        # exist for words that might be names: a tagger that finds no
        # proper noun in "K42;;7315;;QD" is no reason to keep it. So the
        # claim fails closed. A span shaped like a unique identifier is
        # removed as one - not as the phone number it is too short to be -
        # and any other span carrying digits goes to a human. Readings,
        # bed numbers and scores were already kept above.
        if is_unique_identifier_value(span):
            return EvidenceAssessment(
                "OTHER_UNIQUE_ID",
                source,
                evidence,
                "identifier",
                "redact",
                "unique_identifier_syntax_under_structured_claim",
                agreeing_detectors,
            )
        if sum(character.isdigit() for character in span) >= 2:
            return EvidenceAssessment(
                category,
                source,
                evidence,
                "ambiguous",
                "review",
                "digits_under_structured_claim_without_its_syntax",
                agreeing_detectors,
            )

    if is_relative_time_expression(category, span):
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "clinical",
            "preserve",
            "relative_time_expression_carries_no_date",
            agreeing_detectors,
        )

    reading, rationale = clinical_reading_strength(text, start, end)

    if category not in NAME_SHAPED_CATEGORIES and category not in (
        EXACT_STRUCTURED_CATEGORIES
    ):
        # A category a clinical term is not normally confused with. That was
        # once enough to remove the span, because the category was read as if
        # it were a regex match - but only a deterministic source is one, and
        # those returned under Layer 1 above. What is left here is a model's
        # guess, and spaCy guessed DATE_TIME over "HbA1c 58" and "ramipril 5"
        # in 172 of 216 documents. The claim still stands unless the text
        # reads clinically on its own grammar; a span carrying a month, a
        # weekday or a calendar figure is a date element whatever surrounds it.
        if reading == "clinical" and not _carries_calendar_content(category, span):
            return EvidenceAssessment(
                category,
                source,
                evidence,
                "clinical",
                "preserve",
                rationale,
                agreeing_detectors,
            )
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "identifier",
            "redact",
            "category_not_confusable_with_clinical_text",
            agreeing_detectors,
        )

    # Only where no specific clinical grammar decided the span above. A single
    # common noun that opens a sentence of a clinical note ("Mobility had improved",
    # "Imaging was reported by Dr ..."): its capital is explained by its position;
    # the word is on a closed list of assessed functions, systems and tests that no
    # one is called; no naming context is beside it (that returned "identifier"
    # above); and a lower-case word follows, so it is not the first of two names. A
    # pinned model reading the same span does not change that, because the models
    # read capitalisation too.
    if (
        reading == "ambiguous"
        and evidence != EVIDENCE_DETERMINISTIC
        and category in NAME_SHAPED_CATEGORIES
        and _is_clinical_domain_noun_opening(text, start, end)
    ):
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "clinical",
            "preserve",
            "clinical_domain_noun_at_opening",
            agreeing_detectors,
        )

    # A lower-case role noun ("assistant", "one assistant") is not a name. It has
    # no capital letter for a tagger to have read, and naming context still wins
    # (it returned "identifier" above).
    if (
        reading == "ambiguous"
        and evidence != EVIDENCE_DETERMINISTIC
        and category in NAME_SHAPED_CATEGORIES
        and is_bare_role_noun_phrase(text[start:end])
    ):
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "clinical",
            "preserve",
            "role_noun_not_a_name",
            agreeing_detectors,
        )

    if reading == "ambiguous" and clinical_support:
        # Another detector read this same span as a medical condition. That is
        # exactly the corroboration the vocabulary cannot give for a term
        # nobody listed, and it arrives from a model rather than a list.
        # Naming context still wins: it returns "identifier" above.
        reading = "clinical"
        rationale = "another_detector_read_the_span_as_clinical"

    if (
        reading != "identifier"
        and contains_proper_noun is False
        and agreeing_detectors < 2
    ):
        # A tagger found no proper noun anywhere in this span. "nursing
        # staff", "multidisciplinary team" and "Enrolment" are not names, and
        # a single model calling one a PERSON is not enough to replace it with
        # a patient surrogate.
        #
        # Three guards keep this from becoming a hole. Naming context still
        # wins, because it returns "identifier" above. Two agreeing detectors
        # still win, because a name the tagger mis-labels is exactly what
        # corroboration is for. And an unavailable parse arrives as None, not
        # False, so a tagger that failed exempts nothing.
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "clinical",
            "preserve",
            "no_proper_noun_token_in_span",
            agreeing_detectors,
        )

    if reading == "identifier":
        # Layer 4: contextual naming evidence beats a clinical reading.
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "identifier",
            "redact",
            rationale,
            agreeing_detectors,
        )

    if reading == "clinical":
        # Layer 3: a clinical reading blocks weak redaction, but does not
        # overrule several detectors agreeing - unless the reading is
        # structural. Two models can share one domain-specific misreading
        # ("PAS-D stain" as a passport number); a stain the policy proved in
        # pathology context with nothing identifying beside it is not made a
        # name by a second model saying so.
        # A lower-case common noun that is a role or a field word ("family") is not
        # a name the models agreeing could have found: nothing typed in lower case
        # and listed as a label is somebody's name, and naming context returned above.
        lower_case_label_word = (
            rationale == "field_label_or_role_noun"
            and text[start:end].strip().islower()
        )
        if (
            agreeing_detectors >= 2
            and evidence == EVIDENCE_MODEL
            and rationale not in STRUCTURAL_REASONS
            and rationale not in _STRUCTURAL_LABEL_READINGS
            and rationale not in _MEASUREMENT_GRAMMARS
            and not lower_case_label_word
        ):
            return EvidenceAssessment(
                category,
                source,
                evidence,
                "ambiguous",
                "review",
                f"{rationale}_but_multiple_detectors_agree",
                agreeing_detectors,
            )
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "clinical",
            "preserve",
            rationale,
            agreeing_detectors,
        )

    # Ambiguous.
    if evidence == EVIDENCE_HEURISTIC:
        if agreeing_detectors >= 2:
            # Another detector saw the same span: no longer heuristic-only.
            return EvidenceAssessment(
                category,
                source,
                evidence,
                "identifier",
                "redact",
                "heuristic_supported_by_another_detector",
                agreeing_detectors,
            )
        # Layer 5, resolved fail-closed. An unexplained name-shaped span is
        # replaced by a surrogate rather than escalated-and-left-in-place: the
        # proper-noun fallback exists precisely to catch names the models miss,
        # and leaving one in the released text to await review would be a
        # privacy regression. Replacement is also the recoverable direction -
        # a surrogate costs one token of context, a leaked name costs the
        # patient.
        #
        # What stops this from destroying ordinary prose is upstream: the
        # detector no longer proposes a capitalised span whose proper-noun
        # reading is an artifact of sentence position (see
        # ner_phi_detector._is_case_artifact), so spans reaching here are
        # name-shaped in a way sentence capitalisation does not explain.
        #
        # Still replaced, but only released when the span looks like a name
        # by its own grammar; otherwise the file is held for a person, so an
        # unknown clinical word is never silently lost in a released file.
        if (
            agreeing_detectors < 2
            and _ACRONYM_WORD.match(text[start:end].strip(".,;:()"))
            and len(text[start:end].strip(".,;:()")) <= 5
        ):
            # A short all-capitals token is not the shape this fallback exists
            # to catch - "GFR", "ECG" - and naming context already returned
            # above. A model's label on it still goes through the gate.
            return EvidenceAssessment(
                category,
                source,
                evidence,
                "clinical",
                "preserve",
                "acronym_not_name_shaped",
                agreeing_detectors,
            )
        justification = phi_justification(
            text, start, end, category, agreeing_detectors, contains_proper_noun
        )
        if (
            justification is None
            and occurrence_support is not None
            and _claimed_in_a_minority(occurrence_support(), category)
        ):
            return EvidenceAssessment(
                category,
                source,
                evidence,
                "clinical",
                "preserve",
                "claimed_in_a_minority_of_its_occurrences",
                agreeing_detectors,
            )
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "identifier",
            "redact",
            justification or "heuristic_edit_without_phi_justification",
            agreeing_detectors,
            justified=justification is not None,
        )

    # Layer 2: a model or context rule with no clinical reading. Replaced
    # either way. Released only when something beyond the model's label says
    # identifier - name grammar, a relationship, corroboration a sentence
    # position does not explain, address structure. Otherwise the edit is
    # ambiguous and the file goes to a person with the replacement applied.
    justification = phi_justification(
        text, start, end, category, agreeing_detectors, contains_proper_noun
    )
    # A claim on a small minority of a word's occurrences, with nothing to justify
    # it, is a model's noise on a repeated word: the other occurrences were read by
    # the same models and not claimed, so replacing this one protects nothing and
    # only leaves the text partly changed.
    if (
        justification is None
        and occurrence_support is not None
        and _claimed_in_a_minority(occurrence_support(), category)
    ):
        return EvidenceAssessment(
            category,
            source,
            evidence,
            "clinical",
            "preserve",
            "claimed_in_a_minority_of_its_occurrences",
            agreeing_detectors,
        )
    return EvidenceAssessment(
        category,
        source,
        evidence,
        "identifier",
        "redact",
        justification or "model_edit_without_phi_justification",
        agreeing_detectors,
        justified=justification is not None,
    )


#: A word shaped like part of a name: capitalised, or a hyphenated
#: capitalised compound, or an upper-case name part longer than an acronym.
_NAME_PART = re.compile(
    r"^(?:[A-Z][a-z'’]+(?:-[A-Z][a-z'’]+)*|[A-Z]{2,}(?:-[A-Z]{2,})+)$"
)
_COMPOUND_NAME = re.compile(
    r"^(?:[A-Z][a-z'’]+(?:-[A-Z][a-z'’]+)+|[A-Z]{2,}-[A-Z]{2,})$"
)
_ACRONYM_WORD = re.compile(r"^[A-Z0-9]{2,6}$")


def _name_base(word: str) -> str:
    # Synthetic records often attach an ordinal to an otherwise ordinary
    # invented name. The ordinal is not evidence that the name is clinical.
    return re.sub(r"-\d{1,6}$", "", word)


def _name_part(word: str) -> bool:
    base = _name_base(word)
    if _NAME_PART.fullmatch(base):
        return True
    # str.isupper/islower recognize Unicode titlecase names that ASCII [A-Z]
    # misses, while requiring every hyphenated atom to have name-like case.
    atoms = base.split("-")
    return bool(atoms) and all(
        len(atom) >= 2
        and atom[0].isupper()
        and all(char.islower() or char in "'’" for char in atom[1:])
        for atom in atoms
    )


_RELATIONSHIP_BEFORE = re.compile(
    r"(?i)\b(?:son|daughter|wife|husband|partner|spouse|mother|father|mum|dad|brother|"
    r"nephew|niece|aunt|uncle|cousin|grandson|granddaughter|grandmother|grandfather|"
    r"carer|caregiver|guardian|friend|neighbou?r|visitor|colleague|dear|attn|attention|cc|"
    r"named|called|known\s+as|next\s+of\s+kin|nok|contact\s+person|interpreter|"
    r"relative|housemate|household\s+member|attended\s+with|arrived\s+with|"
    r"accompanied\s+by)"
    r"\s*[:,]?\s*(?:(?:is|was|:)\s*)?$"
)
_ADDRESS_STRUCTURE = re.compile(
    r"(?i)\b\d{1,5}[A-Za-z]?\s+(?:[A-Za-z'’-]+\s+){1,3}(?:road|rd|street|st|lane|ln|avenue|"
    r"ave|row|close|drive|way|court|ct|place|pl|terrace|crescent|gardens|square|hill|park|"
    r"grove|mews|walk|boulevard|blvd)\b"
)
_CAMEL_PERSON_TOKEN = re.compile(r"[A-Z][a-z]+(?:[A-Z][a-z]+)+(?:-\d{1,6})?")
_ATTENDING_SUBJECT = re.compile(
    r"\b[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ'’]+\s+"
    r"[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ'’]+(?:-\d{1,6})?\s+attended\b"
)


#: Established clinical grammars that are structure, not vocabulary: a value
#: and its unit around the span. Two models agreeing do not overrule them.
_MEASUREMENT_GRAMMARS = frozenset(
    {"analyte_measurement_grammar", "specimen_measurement_grammar"}
)


def _token_beside(text: str, start: int, end: int) -> bool:
    """A replacement-token shape immediately before or after the span, on its line."""
    before = text[max(0, start - 24) : start].split(chr(10))[-1].rstrip()
    after = text[end : end + 24].split(chr(10))[0].lstrip()
    return bool(
        _TOKEN_SHAPE.search(before[-20:])
        and before.endswith((">",))
        or re.match(r"<[A-Z][A-Z_]*_[1-9][0-9]*>", after)
        or re.search(r"\b[A-Z][A-Z]*(?:_[A-Z]+)*_\d{3,}$", before)
    )


def _person_name_shaped(span: str) -> bool:
    """A hyphenated capitalised compound, or two or more capitalised non-clinical words."""
    words = [w.strip(".,;:()[]\"'") for w in span.split()]
    parts = [
        w
        for w in words
        if w
        and _name_part(w)
        and not _ACRONYM_WORD.match(w)
        and not _clinical_word(_name_base(w))
    ]
    return any(_COMPOUND_NAME.match(w) for w in parts) or len(parts) >= 2


def _clinical_word(word: str) -> bool:
    folded = word.strip(".,;:()[]").casefold()
    return bool(folded) and (
        is_unconditional_clinical_term(folded)
        or is_non_name_label(folded)
        or is_clinical_product_term(folded)
        or contains_distinctive_clinical_token(folded)
    )


_SOURCE_TOKEN_SHAPE = re.compile(r"<?[A-Z][A-Z]*(?:_[A-Z]+)*_\d{1,6}>?")


_POSTAL_VALUE = re.compile(r"\d{5}(?:-\d{4})?")
_ANNOUNCED_CODE = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")


def _announced_identifier_value(span: str) -> Optional[str]:
    """Identifier-field label words, then one value of that field's own shape.

    "iris scan IR-SYN-6154" and "postal code 89111": the structure, not a
    model's label, says what the last token is. Every earlier word must be a
    field label from the closed list, so a name or an ordinary noun before the
    value declines; so does a bare clinical-looking code, because the label has
    to be in the span. A postal value is accepted only after a postal label.
    """
    words = span.split()
    if len(words) < 2:
        return None
    value = words[-1].strip(".,;:()[]\"'")
    labels = " ".join(words[:-1])
    if not value or not is_identifier_label_text(labels):
        return None
    if re.fullmatch(r"(?i)(?:https?://|www\.)\S+", value):
        return "announced_identifier_value"
    if _POSTAL_VALUE.fullmatch(value):
        return (
            "announced_identifier_value"
            if re.search(r"(?i)\b(?:zip|postal|postcode)\b", labels)
            else None
        )
    # The announcing label is in the span, so a shorter code is enough than for
    # an unannounced one: letters and at least three digits in one token.
    announced_code = bool(
        _ANNOUNCED_CODE.fullmatch(value)
        and sum(character.isdigit() for character in value) >= 3
        and any(character.isalpha() for character in value)
    )
    return (
        "announced_identifier_value"
        if announced_code or is_unique_identifier_value(value)
        else None
    )


def _labelled_just_before(text: str, start: int, span: str) -> Optional[str]:
    """A code-shaped span whose field label sits immediately before it, outside it.

    "iris scan" is kept as a label while a model proposes only "IR-SYN-6154".
    The words before the span, in the same field (no ; : . or line break
    between), must all be identifier-field labels from the closed list, and
    the span itself must be one announced-code token. A name or any other
    word before the span declines.
    """
    value = span.strip(".,;:()[]\"'")
    if not value or " " in value or not _ANNOUNCED_CODE.fullmatch(value):
        return None
    if not (
        sum(character.isdigit() for character in value) >= 3
        and any(character.isalpha() for character in value)
    ):
        return None
    field = re.split(r"[;:.\n\r]", text[max(0, start - 48) : start])[-1]
    words = field.split()[-3:]
    if words and is_identifier_label_text(" ".join(words)):
        return "announced_identifier_value"
    return None


def phi_justification(
    text: str,
    start: int,
    end: int,
    category: str,
    agreeing_detectors: int,
    contains_proper_noun: Optional[bool],
) -> Optional[str]:
    """Why a model's proposed redaction is an identifier, or None when nothing says so.

    Detector count alone is not corroboration: two models share the same
    domain-specific misreadings. What counts is structure a clinical term
    does not have - a person-name shape, a relationship word before it,
    corroboration away from a sentence opening, an address.
    """
    span = text[start:end]
    if re.search(r"[A-Za-z]\^[A-Za-z]", span):
        return "dicom_person_name_structure"
    # Source text shaped like a replacement token is unverifiable: removing
    # it costs nothing clinical, and it is never trusted as already removed.
    if _SOURCE_TOKEN_SHAPE.fullmatch(span.strip()):
        return "token_shaped_source_text"
    if _ADDRESS_STRUCTURE.search(text[max(0, start - 12) : end + 24]) and category in (
        "ADDRESS",
        "LOCATION",
        "STREET_ADDRESS",
        "GPE",
        "FACILITY",
    ):
        return "address_structure"
    announced = _announced_identifier_value(span) or _labelled_just_before(
        text, start, span
    )
    if announced:
        return announced
    words = [w.strip(".,;:()[]\"'") for w in span.split()]
    name_parts = [
        w
        for w in words
        if w
        and _name_part(w)
        and not _ACRONYM_WORD.match(w)
        and not _clinical_word(_name_base(w))
    ]
    if any(_COMPOUND_NAME.match(w) for w in name_parts):
        return "compound_surname_shape"
    if len(name_parts) >= 2 and contains_proper_noun is not False:
        return "multi_token_name_shape"
    if _RELATIONSHIP_BEFORE.search(text[max(0, start - 48) : start]) and (
        name_parts or _CAMEL_PERSON_TOKEN.fullmatch(span)
    ):
        return "relationship_context"
    role_prefix = re.match(r"(?i)^(?:housemate|relative|spouse|partner)\s+(.+)$", span)
    if role_prefix and _person_name_shaped(role_prefix.group(1)):
        return "relationship_context"
    if name_parts and category == "PERSON":
        # A model can offer only one half of a full subject name (often with
        # the ordinal outside its span). The adjacent reporting predicate is
        # independent evidence of the name, not a vote from another model.
        sentence = text[max(0, start - 48) : min(len(text), end + 48)]
        for match in _ATTENDING_SUBJECT.finditer(sentence):
            absolute_start = max(0, start - 48) + match.start()
            absolute_end = max(0, start - 48) + match.end()
            if absolute_start <= start and end <= absolute_end:
                return "person_context_present"
    if (
        name_parts
        and agreeing_detectors >= 2
        and contains_proper_noun
        and not _is_sentence_initial(text, start)
    ):
        return "corroborated_proper_noun"
    return None


def _same_reading(category: str, other: str) -> bool:
    """Whether two labels are two detectors making the same identifier claim.

    A prediction in a category that is not an identifier at all is not a claim
    that the span is one, so it is not agreement with one. GLiNER is asked for
    "medical condition" precisely so that a clinical span gets a clinical
    label; counting that label as a second vote for PERSON made the detector
    arguing for preservation the corroboration for removal, and "Haemoglobin
    118 g/L" went to manual review on the strength of it. The condition label
    keeps its own role in :func:`has_clinical_support`, where it is read as
    evidence that the span is clinical.
    """
    if category in NON_IDENTIFIER_CATEGORIES or other in NON_IDENTIFIER_CATEGORIES:
        return False
    if category == other:
        return True
    return category in NAME_SHAPED_CATEGORIES and other in NAME_SHAPED_CATEGORIES


def contains_proper_noun_token(
    proper_nouns,
    start: int,
    end: int,
):
    """Whether any proper-noun token overlaps the span.

    ``None`` in, ``None`` out: an unavailable parse must stay distinguishable
    from a parse that found nothing.
    """
    if proper_nouns is None:
        return None
    return any(
        token_start < end and start < token_end
        for token_start, token_end in proper_nouns
    )


#: A token, for looking inside a span a model handed over whole.
_SPAN_TOKEN = re.compile(r"[^\s,;:|()\[\]]+")


def identifier_token_spans(text: str, start: int, end: int) -> List[Tuple[int, int]]:
    """Offsets of every token inside the span that is written as an identifier.

    A model calling a span a medical condition says the span is not one of the
    eighteen. It says nothing about a code sitting inside it: GLiNER read
    "palm-vein template PVT123456" as a condition, and the biometric
    identifier in it was released twice on the strength of that reading.

    The syntax test is the one a model's own structured claim has to pass - a
    letter and at least four digits - so a reading, a ratio, a date, an
    ICD-10 code and an ordinary word all fail it, and the clinical words
    around the code are left where they are.
    """
    spans: List[Tuple[int, int]] = []
    for match in _SPAN_TOKEN.finditer(text[start:end]):
        raw = match.group(0)
        token = raw.strip(".,;:")
        if not token or not is_unique_identifier_value(token):
            continue
        offset = start + match.start() + raw.index(token)
        spans.append((offset, offset + len(token)))
    return spans


def has_clinical_support(
    entities: Sequence[object],
    start: int,
    end: int,
) -> bool:
    """Whether some detector read an overlapping span as a medical condition."""
    return any(
        getattr(e, "entity_type", "") in NON_IDENTIFIER_CATEGORIES
        and getattr(e, "start", -1) < end
        and start < getattr(e, "end", -1)
        for e in entities
    )


def count_agreeing(
    entities: Sequence[object],
    start: int,
    end: int,
    category: str,
    exclude_source: Optional[str] = None,
) -> int:
    """How many distinct detectors made the *same* claim about this span.

    Agreement means two detectors independently reading the span the same
    way - both calling it a name, both calling it a facility. Bare overlap is
    not agreement: a structured-pattern hit on "MRN: 123456" overlaps a model
    calling the label "MRN" an organisation, but the two are describing
    different things, and counting them as corroboration escalated every
    ordinary field label to manual review.

    Deterministic sources are excluded for the same reason. A regex is
    authoritative about its own category - it decides on its own under Layer 1
    - and is silent about whether a neighbouring word is a name.
    """
    sources = set()
    for entity in entities:
        source = getattr(entity, "source", None)
        if source is None or source == exclude_source:
            continue
        if source in _DETERMINISTIC_SOURCES:
            continue
        if getattr(entity, "start", -1) >= end or start >= getattr(entity, "end", -1):
            continue
        if not _same_reading(category, getattr(entity, "entity_type", "")):
            continue
        sources.add(source)
    return len(sources) + (1 if exclude_source else 0)


#: A word has to occur at least this often in a text before how many of its
#: occurrences were claimed says anything about the claim.
MINORITY_MIN_OCCURRENCES = 8
#: A claim on at most this share of a word's occurrences is a minority.
MINORITY_MAX_SHARE = 0.15
_CLAIMABLE_SURFACE = re.compile(r"[A-Za-z][A-Za-z'’\-]*(?: [A-Za-z][A-Za-z'’\-]*)?")


def occurrence_support(
    entities: Sequence[object],
    text: str,
    start: int,
    end: int,
) -> Optional[Tuple[int, int]]:
    """How many of a span's occurrences in this text a detector claimed as an identifier.

    Returns ``(claimed, total)`` over the whole-word, same-case occurrences of the
    span's text, or ``None`` when the span is not a word or two that could recur or
    the text has too few occurrences to say anything. An occurrence counts as
    claimed when any non-deterministic detector proposed an identifier category
    over it.
    """
    surface = text[start:end].strip()
    if not 3 <= len(surface) <= 40 or not _CLAIMABLE_SURFACE.fullmatch(surface):
        return None
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(surface)}(?![A-Za-z0-9])")
    occurrences = [match.start() for match in pattern.finditer(text)]
    if len(occurrences) < MINORITY_MIN_OCCURRENCES:
        return None
    claims = [
        (getattr(entity, "start", -1), getattr(entity, "end", -1))
        for entity in entities
        if getattr(entity, "source", None) not in _DETERMINISTIC_SOURCES
        and getattr(entity, "entity_type", "") not in NON_IDENTIFIER_CATEGORIES
    ]
    claimed = sum(
        1
        for position in occurrences
        if any(
            low < position + len(surface) and position < high for low, high in claims
        )
    )
    return claimed, len(occurrences)


#: Only a place-like claim can be set aside this way. A person's name that the
#: models claim now and then is exactly what a person has to look at.
_MINORITY_CATEGORIES = frozenset({"FACILITY", "ORGANIZATION", "LOCATION"})


def _claimed_in_a_minority(support: Optional[Tuple[int, int]], category: str) -> bool:
    if support is None or category not in _MINORITY_CATEGORIES:
        return False
    claimed, total = support
    return total >= MINORITY_MIN_OCCURRENCES and claimed <= total * MINORITY_MAX_SHARE
