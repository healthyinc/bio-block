"""What one occurrence of a number is, from its own context only.

The same digits can be a bleep in one sentence and a laboratory result in the
next. Every decision here is about one occurrence: the text around *this*
span, bounded by its sentence, line, table cell or metadata field. Nothing is
looked up by the digits themselves, and nothing is remembered between calls.

Order of evidence (the first that applies wins):

1. a contact label before or after the number (bleep, pager, ext, direct
   line, switchboard, contact number ...), tolerant of OCR spacing
   ("E x t .", "Ext .", "EXT:") and matched on the original characters, so
   offsets never move;
2. clinical structure - a result with a unit, a dose, an imaging parameter,
   a score or scale, a count, a year, a clock time, a room or bed, a page,
   part of a code, date or ratio;
3. otherwise the occurrence is unresolved: the review candidate replaces it
   and the file is held. It is never released automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

POLICY_VERSION = "numeric-context-v2"

# Public occurrence classes.  The aliases at the end preserve the small API
# used by older callers while making the release decision explicit.
CONFIRMED_IDENTIFIER = "confirmed_identifier"
CONFIRMED_CONTACT = "confirmed_contact"
CONFIRMED_CLINICAL_VALUE = "confirmed_clinical_value"
CONFIRMED_DATE_OR_TIME = "confirmed_date_or_time"
CONFIRMED_CODE = "confirmed_code"
CONFIRMED_DOCUMENT_MARKER = "confirmed_document_marker"
AMBIGUOUS_IDENTIFIER_CANDIDATE = "ambiguous_identifier_candidate"

CONTACT = CONFIRMED_CONTACT
CLINICAL = CONFIRMED_CLINICAL_VALUE
UNRESOLVED = AMBIGUOUS_IDENTIFIER_CANDIDATE

#: Contact labels, as written. Multi-word labels allow any run of spaces.
CONTACT_LABELS: Tuple[str, ...] = (
    "bleep",
    "bleeper",
    "pager",
    "ext",
    "extension",
    "switchboard",
    "callback",
    "call back",
    "call",
    "ring",
    "dial",
    "contact number",
    "contact no",
    "internal number",
    "internal extension",
    "on-call number",
    "on call number",
    "direct line",
    "direct dial",
    "dect",
    "phone ext",
    "tel ext",
    "page on",
    "page via",
    "page at",
)


def _ocr_tolerant(label: str) -> str:
    """A label as OCR writes it: letters optionally split by single spaces.

    "ext" matches "ext", "Ext", "EXT" and "E x t"; the spaced form must space
    every letter, so no ordinary word ("next", "text") is mistaken for one.
    """
    words = []
    for word in label.replace("-", " ").split():
        letters = [re.escape(ch) for ch in word]
        words.append("(?:" + "".join(letters) + "|" + r"[ ]".join(letters) + ")")
    return r"[\s\-]+".join(words)


_LABEL = "|".join(
    _ocr_tolerant(label) for label in sorted(CONTACT_LABELS, key=len, reverse=True)
)
#: The label, then optional OCR-spaced punctuation, then the number - on one line.
LABEL_BEFORE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    + _LABEL
    + r")(?![A-Za-z])[ \t]?[.:#=]?[ \t]*(?:[.:#=][ \t]*)?"
    r"(?:(?:on|via|at|number|no\.?|#)[ \t]+)?\(?[ \t]*$",
    re.IGNORECASE,
)
LABEL_AFTER = re.compile(
    r"^[ \t]*\([ \t]*(?:" + _LABEL + r")[ \t]*\.?[ \t]*\)", re.IGNORECASE
)

#: A contact number: 3 to 6 digits, or two groups split by one space or hyphen.
CONTACT_VALUE = re.compile(r"\d{2,3}[ \-]\d{1,4}|\d{3,6}")

_UNIT_AFTER = re.compile(
    r"^[ \t]*(?:%|x\s*10|×\s*10|10\^|[µu]?mol|n?mol|pmol|[mµunpk]?g\b|[mµunpk]?g/|"
    r"[mµ]?l\b|[mµ]?L\b|dl\b|dL\b|iu\b|IU\b|[mµ]?IU|[mk]?U/|U\b|units?\b|mEq|ms\b|msec|"
    r"s\b|sec|min\b|mins\b|mm\b|cm\b|m\b|km\b|mmHg|cmH2O|kPa|bpm|/min|beats|breaths|"
    r"kg\b|lb\b|Hz|kHz|MHz|T\b|tesla|°|deg|mSv|Gy|cGy|mGy|MBq|kBq|fL\b|pg\b|cells|/L|"
    r"/l\b|/µL|/uL|/mm|mOsm|ng|mcg|mg|ml|mL|g/|L/|mm/|kcal|cal\b|J\b|W\b|Pa\b|"
    r"per\b|/day|/d\b|/h\b|/hr|/wk|/week|puffs?|drops?|tabs?|tablets?|capsules?|doses?|"
    r"sachets?|vials?|ampoules?|patches?|sprays?|micrograms?|milligrams?|nanograms?|"
    r"kilograms?|litres?|liters?|millilitres?|milliliters?)",
    re.IGNORECASE,
)
_COUNTED_AFTER = re.compile(
    r"^[ \t]*(?:x\b|×|times\b|days?\b|weeks?\b|months?\b|years?\b|yrs?\b|hours?\b|hrs?\b|"
    r"minutes?\b|seconds?\b|episodes?\b|occasions?\b|attempts?\b|steps?\b|metres?\b|"
    r"meters?\b|feet\b|ft\b|miles?\b|of\s+\d|/\s*\d|\.\d|,\d{3}\b)",
    re.IGNORECASE,
)
_IMAGING_BEFORE = re.compile(
    r"(?:\b(?:TR|TE|TI|TA|FA|NEX|ETL|FOV|b|b-value|slice|slices|matrix|bandwidth|"
    r"flip angle|echo time|repetition time|inversion time|thickness|gap|kVp|mAs|mA|CTDI|"
    r"DLP|dose|acquisition parameter|scan parameter|protocol parameter)\s*[:=]?\s*)$",
    re.IGNORECASE,
)
_SCORE_BEFORE = re.compile(
    r"(?:\b(?:score|scale|grade|stage|class|GCS|NEWS2?|APGAR|MMSE|MoCA|PHQ-?9|GAD-?7|"
    r"Wells|CURB-?65|CHA2DS2-?VASc|HAS-?BLED|Waterlow|Braden|Barthel|ASA|NYHA|MRC|"
    r"pain|VAS|NRS|level|count|total|result|reading|value|dose|volume|rate|weight|"
    r"height|temp(?:erature)?|BP|HR|RR|SpO2|sats?|pulse|glucose|BM|CBG|"
    # Common laboratory and observation abbreviations, pinned: a bare number
    # after an unknown word stays unresolved and holds the file.
    r"Hb|Hgb|WCC|WBC|RBC|Plt|PLT|platelets|Na|K|Cl|Ca|Mg|PO4|Cr|Creat|creatinine|Ur|"
    r"urea|eGFR|GFR|ALT|AST|ALP|GGT|Bili|bilirubin|Alb|albumin|CRP|ESR|PCT|INR|APTT|PT|"
    r"TSH|T3|T4|FT4|HbA1c|LDL|HDL|TG|chol|cholesterol|CK|LDH|troponin|trop|BNP|"
    r"NT-?proBNP|ferritin|B12|folate|lactate|pH|pCO2|pO2|HCO3|BE|osmolality|amylase|"
    r"lipase|cortisol|prolactin|PSA|CEA|CA-?125|CA19-?9|AFP|IgE|IgG|IgA|IgM)"
    r"\s*(?:of|was|is|=|:)?\s*)$",
    re.IGNORECASE,
)
_YEAR_BEFORE = re.compile(
    r"(?:\b(?:since|in|from|until|till|during|year|yr|born|diagnosed|dx|onset|circa|c\.|"
    r"before|after|by|of|summer|winter|spring|autumn|fall|early|late|mid|"
    r"january|february|march|april|may|june|july|august|september|october|november|"
    r"december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s*)$",
    re.IGNORECASE,
)
_TIME_BEFORE = re.compile(
    r"(?:\b(?:at|by|from|until|till|before|after|around|approx|"
    r"@|time|timed|hrs?|h)\s*)$",
    re.IGNORECASE,
)
_ROOM_BEFORE = re.compile(
    r"(?:\b(?:room|rm|bed|bay|ward|floor|level|theatre|theater|cubicle|side room|suite|"
    r"clinic|page|pages|p|pp|section|chapter|item|question|q|step|cycle|day|week|visit|"
    r"dose|lot|batch|version|v|model|series|image|frame|slice|table|figure|fig|line|row|"
    r"column|col|cell|sheet|box|study|protocol|trial|cohort|arm)\.?"
    r"(?:\s+(?:number|no\.?|num\.?))?\s*[:#]?\s*)$",
    re.IGNORECASE,
)
_CODE_EDGE = re.compile(r"[A-Za-z0-9][\-/.:]$|^[\-/.:][A-Za-z0-9]")


@dataclass(frozen=True)
class NumericContext:
    kind: str  # one of the seven occurrence classes above
    reason: str  # value-free: label_before, unit_after, imaging_parameter, ...

    @property
    def is_contact(self) -> bool:
        return self.kind == CONTACT

    @property
    def is_clinical(self) -> bool:
        return self.kind in {
            CONFIRMED_CLINICAL_VALUE,
            CONFIRMED_DATE_OR_TIME,
            CONFIRMED_CODE,
            CONFIRMED_DOCUMENT_MARKER,
        }


def segment(text: str, start: int, end: int, limit: int = 120) -> Tuple[int, int]:
    """The sentence, line, cell or field around a span: context never crosses it."""
    low = max(0, start - limit)
    left = text[low:start]
    cut = max(
        left.rfind("\n"),
        left.rfind("\r"),
        left.rfind("\t"),
        left.rfind("|"),
        _last_sentence_end(left),
    )
    seg_start = low + cut + 1 if cut >= 0 else low
    right = text[end : end + limit]
    stops = [
        i
        for i in (
            right.find("\n"),
            right.find("\r"),
            right.find("\t"),
            right.find("|"),
            _first_sentence_end(right),
        )
        if i >= 0
    ]
    seg_end = end + (min(stops) if stops else len(right))
    return seg_start, seg_end


def _last_sentence_end(text: str) -> int:
    found = -1
    for match in re.finditer(r"[.;!?](?=\s+(?:[A-Z0-9(<]|$))", text):
        # "Ext ." and "No." are labels, not sentence ends.
        before = text[: match.start()].rstrip()
        if re.search(
            r"(?i)(?:\b(?:ext|no|tel|dr|mr|mrs|ms|st|approx|ref)|\b[A-Za-z])$", before
        ):
            continue
        found = match.start()
    return found


def _first_sentence_end(text: str) -> int:
    for match in re.finditer(r"[.;!?](?=\s|$)", text):
        before = text[: match.start()]
        if re.search(r"\d$", before) and re.match(
            r"\d", text[match.end() : match.end() + 1]
        ):
            continue
        return match.start()
    return -1


def classify(text: str, start: int, end: int) -> NumericContext:
    """Contact, clinical or unresolved, for the number at ``text[start:end]``."""
    seg_start, seg_end = segment(text, start, end)
    before = text[seg_start:start]
    after = text[end:seg_end]
    # 1. a contact label, on the same line, in the same sentence
    if LABEL_BEFORE.search(before[-48:]) or LABEL_AFTER.match(after):
        if not (_UNIT_AFTER.match(after) or _COUNTED_AFTER.match(after)):
            return NumericContext(CONTACT, "contact_label")
    # 2. clinical structure
    if _CODE_EDGE.search(before[-2:] if before else "") or _CODE_EDGE.search(after[:2]):
        return NumericContext(CONFIRMED_CODE, "part_of_code_date_or_ratio")
    if _UNIT_AFTER.match(after):
        return NumericContext(CLINICAL, "value_with_unit")
    if _COUNTED_AFTER.match(after):
        return NumericContext(CLINICAL, "count_or_duration")
    if _IMAGING_BEFORE.search(before[-40:]):
        return NumericContext(CLINICAL, "imaging_or_dose_parameter")
    value = text[start:end]
    if value.isdigit() and len(value) == 4:
        number = int(value)
        if 1900 <= number <= 2100 and (
            _YEAR_BEFORE.search(before[-24:]) or not before.strip()
        ):
            return NumericContext(CONFIRMED_DATE_OR_TIME, "year")
        if number <= 2359 and number % 100 < 60 and _TIME_BEFORE.search(before[-16:]):
            return NumericContext(CONFIRMED_DATE_OR_TIME, "clock_time")
    if _ROOM_BEFORE.search(before[-24:]):
        return NumericContext(CONFIRMED_DOCUMENT_MARKER, "room_page_or_sequence_number")
    if _SCORE_BEFORE.search(before[-32:]) and not re.search(
        r"(?i)\b(?:tried|rang|ring|call|" r"phone|dial|contact)\b", before[-32:]
    ):
        return NumericContext(CLINICAL, "score_or_measured_quantity")
    return NumericContext(UNRESOLVED, "no_label_and_no_clinical_structure")


#: Standalone integers of 4-6 digits, not part of a longer number, decimal,
#: code, date, time, ratio, token or identifier.
_STANDALONE = re.compile(r"(?<![\w.,:/\-<>#@+])\d{4,6}(?![\w,:/\-<>@]|\.\d)")


def unresolved_numbers(text: str):
    """(start, end) of every standalone short number that is not resolved."""
    for match in _STANDALONE.finditer(text or ""):
        context = classify(text, match.start(), match.end())
        if context.kind == UNRESOLVED:
            yield match.start(), match.end()


def contact_value_at(text: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    """The contact number at a span, when its own context labels it as one."""
    if classify(text, start, end).is_contact:
        return start, end
    return None
