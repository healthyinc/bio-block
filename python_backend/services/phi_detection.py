import re
from dataclasses import dataclass
from functools import lru_cache
from typing import (
    Callable,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Pattern,
    Sequence,
    Tuple,
    Union,
)

from services import structured_identifier_spec as spec
from services.privacy_contracts import PhiDetector as SharedPhiDetector
from services.privacy_contracts import PhiEntity

SOURCE_NER = "ner"
SOURCE_STRUCTURED_PATTERN = "structured_pattern"
SOURCE_AGE_RULE = "age_rule"
SOURCE_CONTEXT_RULE = "context_rule"
SOURCE_STRICT_PROPER_NOUN = "strict_proper_noun"
#: Another occurrence of a structured value the first pass already confirmed.
SOURCE_REPEAT_PROPAGATION = "structured_repeat"
#: The residual validator's own structured scan. Declared here with the other
#: source names; implemented separately in services.residual_identifier_scan.
SOURCE_RESIDUAL_IDENTIFIER_SCAN = "residual_identifier_scan"


DetectedEntity = PhiEntity
PhiDetector = SharedPhiDetector

Span = Tuple[int, int]


@dataclass(frozen=True)
class _PatternDefinition:
    name: str
    entity_type: str
    regex: Pattern[str]
    #: When set, the detected span is this capture group rather than the whole
    #: match, so surrounding clinical words are preserved. "94 years old"
    #: yields a span over "94" alone, which redacts to "90+ years old".
    group: Union[int, str] = 0
    #: Decides whether a match is an identifier and where its span lies.
    #: ``None`` from it rejects the match, and the search resumes one
    #: character later, so a rejected head cannot hide a real one behind it:
    #: "implant serial IMP-7000001" is found from "serial" after "implant"
    #: fails to announce a value with digits.
    accept: Optional[Callable[[str, "re.Match[str]"], Optional[Span]]] = None

    def span_for(self, text: str, match: "re.Match[str]") -> Optional[Span]:
        if self.accept is not None:
            return self.accept(text, match)
        return match.start(self.group), match.end(self.group)


def _compile(pattern: str) -> Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def _digits(value: str) -> int:
    return sum(character.isdigit() for character in value)


def _letters(value: str) -> int:
    return sum(character.isalpha() for character in value)


# -- telephone numbers and measurement guards -----------------------------------

#: A unit after a number makes it a measurement. Single-letter units are left
#: out of the phone guard: "555-0142 C" is far likelier a number than a
#: temperature, and missing a phone number costs more than a false alarm.
_UNIT_AFTER = re.compile(
    r"\s*(?:"
    + "|".join(
        re.escape(unit)
        for unit in sorted(spec.MEASUREMENT_UNITS, key=len, reverse=True)
        if len(unit) > 1 or unit == "%"
    )
    + r")(?![A-Z])",
    re.IGNORECASE,
)
_WORD_BEFORE = re.compile(r"([A-Za-z][A-Za-z0-9]*)(\s*[:=])?\s*$")
#: Two or more numbers joined by a dash or slash and nothing else: a range, a
#: ratio or a date, never a reference.
_NUMERIC_RANGE = re.compile(r"\d+(?:[.,]\d+)?(?:\s*[-/]\s*\d+(?:[.,]\d+)?)+")


@lru_cache(maxsize=1)
def _measurement_words() -> FrozenSet[str]:
    from services.clinical_vocabulary import CLINICAL_MEASUREMENT_TERMS

    return frozenset(spec.VITAL_SIGN_LABELS | CLINICAL_MEASUREMENT_TERMS)


# -- identifiers announced by a word, compiled from the canonical spec --------


def _phrase(words: str) -> str:
    return r"\s+".join(re.escape(word) for word in words.split())


_QUALIFIER_EXPRESSION = "|".join(
    re.escape(qualifier) for qualifier in sorted(spec.QUALIFIERS, key=len, reverse=True)
)


_GROUP_AFTER = re.compile(r"[ ](\d{1,%d})(?![\w@/.:%%+-])" % spec.MAX_GROUP_DIGITS)
#: What may follow the last group of a value written in groups: the end of the
#: line, or the punctuation that ends a field. A word, a unit or another number
#: after it means the group was something else ("12345 6 mg", "12345 14 March").
_FIELD_END = re.compile(r"[ \t]*(?:\r?\n|$|[.,;)])")


def grouped_digits_end(text: str, start: int, end: int) -> int:
    """The end of a digit-group value ("2951 66420 1") that starts at ``start``.

    Only a value whose first group is all digits is extended, by up to
    ``MAX_GROUPED_EXTRA_GROUPS`` further digit-only groups separated by single
    spaces, and only to the longest run that ends its field. A run that does not
    end its field is not a continuation, so the value keeps its own end.
    """
    if not text[start:end].isdigit():
        return end
    cursor = end
    ends = []
    for _ in range(spec.MAX_GROUPED_EXTRA_GROUPS):
        group = _GROUP_AFTER.match(text, cursor)
        if group is None:
            break
        cursor = group.end()
        ends.append(cursor)
    for candidate in reversed(ends):
        # The unit check reads the same line only: the next line's first word is
        # never a unit of this value.
        same_line = text[candidate : candidate + 16].split("\n")[0]
        if _FIELD_END.match(text, candidate) and not _UNIT_AFTER.match(same_line):
            return candidate
    return end


def _keyed_definition(identifier: spec.KeyedIdentifier) -> _PatternDefinition:
    heads = "|".join(
        _phrase(head) for head in sorted(identifier.heads, key=len, reverse=True)
    )
    unit = "[A-Z0-9" + re.escape(identifier.extra_characters) + "]"
    joiner = "-"
    if identifier.inner_delimiters:
        joiner = "[-" + re.escape(identifier.inner_delimiters) + "]{1,2}"
    expression = _compile(
        r"(?<![A-Z0-9])(?:" + heads + r")(?![A-Z])"
        r"(?:\s*(?:" + _QUALIFIER_EXPRESSION + r")(?![A-Z]))*"
        r"\.?\s*[:#-]?\s*"
        r"(?P<value>" + unit + r"(?:" + unit + r"|" + joiner + r"(?=" + unit + r"))*)"
    )

    def accept(text: str, match: "re.Match[str]") -> Optional[Span]:
        start, end = match.span("value")
        # A bracket belongs to the value only when the value opened it.
        while (
            end > start
            and text[end - 1] == ")"
            and (text.count("(", start, end) < text.count(")", start, end))
        ):
            end -= 1
        while end > start and text[end - 1] == "+":
            end -= 1
        end = _without_run_on_sentence(text, start, end)
        if identifier.grouped_digits:
            end = grouped_digits_end(text, start, end)
        value = text[start:end]
        if len(value) < identifier.min_length or _digits(value) < identifier.min_digits:
            return None
        if identifier.digits_without_letters is not None and not _letters(value):
            # A value with no letter needs more digits, and a range or a
            # number before a unit is a reading, not a reference.
            if (
                _digits(value) < identifier.digits_without_letters
                or _NUMERIC_RANGE.fullmatch(value)
                or _UNIT_AFTER.match(text, end)
            ):
                return None
        return (match.start() if identifier.span_includes_head else start), end

    return _PatternDefinition(
        identifier.category.lower(), identifier.category, expression, accept=accept
    )


def _ocr_label(label: str) -> str:
    """A registration label with optional OCR spacing inside acronyms."""
    words = []
    for word in label.split():
        escaped = re.escape(word)
        if len(word) <= 4 and word.isalpha():
            spaced = r"[ .]*".join(re.escape(character) for character in word)
            words.append("(?:" + escaped + "|" + spaced + r"\.?)")
        else:
            words.append(escaped)
    return r"[ \t]+".join(words)


_REGISTRATION_LABEL = "|".join(
    _ocr_label(label)
    for label in sorted(spec.PROFESSIONAL_REGISTRATION_LABELS, key=len, reverse=True)
)
#: A qualifier may be spaced out the way the label is ("r e g", "n o").
_REGISTRATION_QUALIFIER = "|".join(
    _ocr_label(item)
    for item in sorted(spec.PROFESSIONAL_REGISTRATION_QUALIFIERS, key=len, reverse=True)
)
_REGISTRATION_VALUE_PLAIN = r"[A-Z0-9](?:[A-Z0-9]|[-/](?=[A-Z0-9])){3,23}"
#: A value an OCR engine or a typist spaced out one character at a time
#: ("9 8 B 1 2 7 6 F"); only ever read after a registration label.
_REGISTRATION_VALUE_SPACED = r"[A-Z0-9](?:[ ][A-Z0-9]){5,23}"
_REGISTRATION_VALUE = (
    r"(?:" + _REGISTRATION_VALUE_SPACED + r"|" + _REGISTRATION_VALUE_PLAIN + r")"
)
_REGISTRATION_AFTER_LABEL = _compile(
    r"(?<![A-Z0-9])(?:" + _REGISTRATION_LABEL + r")(?![A-Z])"
    r"(?:[ \t]+(?:" + _REGISTRATION_QUALIFIER + r")(?![A-Z]))?"
    r"[ \t]*[.:#=\-]?[ \t]*(?P<value>" + _REGISTRATION_VALUE + r")(?![A-Z0-9])"
)
_REGISTRATION_BEFORE_LABEL = _compile(
    r"(?<![A-Z0-9])(?P<value>" + _REGISTRATION_VALUE + r")(?![A-Z0-9])"
    r"[ \t]*(?:\([ \t]*)?(?:" + _REGISTRATION_LABEL + r")(?![A-Z])(?:[ \t]*\))?"
)


def npi_checksum_valid(value: str) -> bool:
    """Whether a ten-digit NPI satisfies its ISO/IEC 7812 Luhn check digit."""
    if not re.fullmatch(r"\d{10}", value or ""):
        return False
    digits = [int(character) for character in "80840" + value]
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _accept_professional_registration(
    text: str, match: "re.Match[str]"
) -> Optional[Span]:
    start, end = match.span("value")
    value = text[start:end]
    if sum(character.isdigit() for character in value) < 3:
        return None
    # An immediate unit proves the token is a measurement.  A failed NPI
    # checksum does not prove benignity under a strong NPI label, so it is
    # still transformed; npi_checksum_valid is reported/tested separately.
    if _UNIT_AFTER.match(text, end):
        return None
    return start, end


_PROFESSIONAL_REGISTRATION_DEFINITIONS = (
    _PatternDefinition(
        "professional_registration_after_label",
        "LICENSE_NUMBER",
        _REGISTRATION_AFTER_LABEL,
        group="value",
        accept=_accept_professional_registration,
    ),
    _PatternDefinition(
        "professional_registration_before_label",
        "LICENSE_NUMBER",
        _REGISTRATION_BEFORE_LABEL,
        group="value",
        accept=_accept_professional_registration,
    ),
)


# -- unannounced unique identifiers joined by distinctive delimiters ----------

_UNIQUE_JOIN = "[" + re.escape(spec.UNIQUE_DELIMITERS) + "]"
#: Parts joined by a single tight delimiter or a doubled one, which OCR may
#: space out: "HT::5820-J4::MV", "K42 ;; 7315 ;; QD". A single delimiter with
#: spaces around it is a table cell or a list, not part of an identifier.
_DELIMITED_IDENTIFIER = _compile(
    r"(?<![A-Z0-9" + re.escape(spec.UNIQUE_DELIMITERS) + r"])"
    r"(?P<value>[A-Z0-9]{1,16}(?:(?:[ ]?("
    + _UNIQUE_JOIN
    + r")[ ]?\2[ ]?|"
    + _UNIQUE_JOIN
    + r")[A-Z0-9]{1,16}){"
    + str(spec.UNIQUE_MIN_PARTS_UNANNOUNCED - 1)
    + r",7})"
    r"(?!" + r"[A-Z0-9]|" + _UNIQUE_JOIN + r"[A-Z0-9])"
)
_PARTS = re.compile(r"[A-Za-z0-9]+")
_DELIMITER_RUNS = re.compile(_UNIQUE_JOIN + "+")


def _parts_carry_digits(parts: Sequence[str]) -> bool:
    return any(_digits(part) >= spec.UNIQUE_MIN_PART_DIGITS for part in parts)


def _accept_delimited_identifier(text: str, match: "re.Match[str]") -> Optional[Span]:
    """A delimited code, unless it could be a date, a ratio or clinical shorthand.

    It needs a letter, enough digits, some part with several digits - which
    spinal levels and gene alleles do not have - and a doubled delimiter or
    two different ones, which dates and ranges do not use.
    """
    start, end = match.span("value")
    value = text[start:end]
    parts = _PARTS.findall(value)
    runs = _DELIMITER_RUNS.findall(value.replace(" ", ""))
    if not _letters(value) or _digits(value) < spec.UNIQUE_MIN_DIGITS:
        return None
    if len(parts) < spec.UNIQUE_MIN_PARTS_UNANNOUNCED or not _parts_carry_digits(parts):
        return None
    doubled = any(len(run) >= 2 and len(set(run)) == 1 for run in runs)
    kinds = {character for run in runs for character in run}
    if not doubled and len(kinds) < 2:
        return None
    return start, end


#: A letter-and-digit value, optionally joined by delimiters, as a whole span.
_IDENTIFIER_SHAPE = re.compile(
    r"[A-Za-z0-9]+(?:(?:[ ]?("
    + _UNIQUE_JOIN
    + r")[ ]?\1[ ]?|"
    + _UNIQUE_JOIN
    + r")[A-Za-z0-9]+)*"
)


def is_unique_identifier_value(value: str) -> bool:
    """Whether a whole span has the shape of a unique identifier.

    Used where a model has already claimed the span is an identifier and the
    question is whether the text agrees: a letter, at least four digits, and
    either one long token ("QL4827") or parts of which one carries several
    digits ("K42;;7315;;QD"). A reading, a ratio, a date or a word never
    qualifies, because none has both a letter and those digits.
    """
    value = value.strip()
    if not value or not _IDENTIFIER_SHAPE.fullmatch(value):
        return False
    if not _letters(value) or _digits(value) < spec.UNIQUE_MIN_DIGITS:
        return False
    parts = _PARTS.findall(value)
    if len(parts) == 1:
        return len(value) >= spec.UNIQUE_MIN_LENGTH_UNDELIMITED
    return _parts_carry_digits(parts)


def _accept_local_phone(text: str, match: "re.Match[str]") -> Optional[Span]:
    """A short local number, unless the grammar around it says measurement.

    "555.0142" with nothing around it is a phone number. The same digits after
    a vital-sign label or an analyte, inside a range, or before a unit are a
    reading, and a reading is never replaced.
    """
    start, end = match.span("value")
    before = _WORD_BEFORE.search(text[max(0, start - 32) : start])
    if before:
        word = before.group(1).casefold()
        if word in _measurement_words():
            return None
        # "From:" opens a header; a range word only governs the bare number.
        if word in spec.RANGE_WORDS and not before.group(2):
            return None
    if _UNIT_AFTER.match(text, end):
        return None
    return start, end


_LOCAL_PHONE = _compile(
    r"(?<![\w.\-/:+])(?P<value>\(\d{3}\)[ ]?\d{4}|\d{3}["
    + re.escape(spec.PHONE.contextless_separators)
    + r"]\d{4})(?![\w%]|[.\-/:]\d)"
)
#: A number written with its trunk zero, or with a plus and a country code,
#: and grouped by spaces, dots or hyphens. Digit counts come from the spec.
_NATIONAL_PHONE = _compile(
    r"(?<![\w.\-/:+])(?P<value>\+\d{1,3}[ .\-]?\(?\d{1,5}\)?(?:[ .\-]?\d{2,4}){1,4}"
    r"|0\d{1,4}[ .\-]\d{3,4}[ .\-]\d{3,4}|0\d{3,4}[ .\-]\d{6})(?![\w%]|[.\-/:]\d)"
)


def _accept_national_phone(text: str, match: "re.Match[str]") -> Optional[Span]:
    start, end = match.span("value")
    value = text[start:end]
    digits = _digits(value)
    low, high = (
        spec.INTERNATIONAL_PHONE_DIGITS
        if value.startswith("+")
        else spec.TRUNK_PHONE_DIGITS
    )
    if not low <= digits <= high:
        return None
    return _accept_local_phone(text, match)


_CONTEXT_PHONE = _compile(
    r"(?<![A-Z0-9])(?:"
    + "|".join(
        re.escape(word)
        for word in sorted(spec.PHONE.context_words, key=len, reverse=True)
    )
    + r")(?![A-Z])\.?(?:\s*(?:number|no|#)(?![A-Z])\.?)?\s*[:#-]?\s*"
    r"(?P<value>\(?\d{3}\)?[\s.\-]?\d{4}|\d{7})(?![A-Z0-9]|[.\-/:]\d)"
)


#: Labels that make a short number a way to reach a person: an extension, a
#: bleep or pager, a switchboard or direct line. The numbers are short enough
#: that no phone grammar sees them ("bleep 4417"), and a model may or may not
#: tag them, so the label is the evidence. "page" is a label only as a verb
#: with a target ("page on 2231"), never as "Page 3 of 5".
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
)
_CONTACT_LABEL = "|".join(
    re.escape(label).replace(r"\ ", r"[\s-]+")
    for label in sorted(CONTACT_LABELS, key=len, reverse=True)
)
#: A short contact number: 3 to 6 digits, or two groups split by one space or
#: hyphen as OCR and typists write them ("44 17", "44-17").
_CONTACT_VALUE = r"(?P<value>\d{2,3}[ \-]\d{1,4}|\d{3,6})"
#: Words that make the number something counted or measured, not dialled.
_COUNTED_AFTER = re.compile(
    r"\s*(?:x\b|×|times\b|days?\b|weeks?\b|months?\b|years?\b|hours?\b|hrs?\b|minutes?\b|"
    r"mins?\b|seconds?\b|secs?\b|doses?\b|tablets?\b|capsules?\b|puffs?\b|drops?\b|"
    r"units?\b|degrees?\b|°|of\s+\d|/\s*\d|\.\d|%)",
    re.IGNORECASE,
)
_CONTACT_AFTER_LABEL = _compile(
    r"(?<![A-Z0-9])(?:" + _CONTACT_LABEL + r"|page(?=\s+(?:on|via|at)\b))"
    r"(?![A-Z])\.?(?:\s+(?:on|via|at|number|no\.?|#))?\s*[:=#\-]?\s*\(?\s*"
    + _CONTACT_VALUE
    + r"(?![\w%]|[.\-/:]\d)"
)
_CONTACT_BEFORE_LABEL = _compile(
    r"(?<![\w.\-/:+])"
    + _CONTACT_VALUE
    + r"\s*\(\s*(?:bleep|bleeper|pager|ext\.?|extension|switchboard|direct\s+line)\s*\)"
)


def _accept_labelled_contact(text: str, match: "re.Match[str]") -> Optional[Span]:
    """A short number whose own sentence labels it a contact."""
    from services import numeric_context

    start, end = match.span("value")
    if not numeric_context.classify(text, start, end).is_contact:
        return None
    if _UNIT_AFTER.match(text, end) or _COUNTED_AFTER.match(text, end):
        return None
    digits = re.sub(r"\D", "", text[start:end])
    # "call 2025 review" is a year after a verb; "extension 1994" is a line.
    if (
        len(digits) == 4
        and 1900 <= int(digits) <= 2100
        and re.search(r"(?i)\bcall(?:\s*back)?\s*$", text[max(0, start - 16) : start])
    ):
        return None
    return start, end


def _accept_contact_number(text: str, match: "re.Match[str]") -> Optional[Span]:
    """A labelled short number, unless what follows makes it counted or measured."""
    start, end = match.span("value")
    if _UNIT_AFTER.match(text, end) or _COUNTED_AFTER.match(text, end):
        return None
    digits = re.sub(r"\D", "", text[start:end])
    # A four-digit year after "call" ("call 2025 review") is a date, not a line.
    if (
        len(digits) == 4
        and 1900 <= int(digits) <= 2100
        and "-" not in text[start:end]
        and " " not in text[start:end]
    ):
        return None
    return start, end


#: A capitalised word joined to the end of a match by a full stop.
_RUN_ON_SENTENCE = re.compile(r"\.([A-Z][a-z]+)$")


def _without_run_on_sentence(text: str, start: int, end: int) -> int:
    """End a match before a sentence that follows it with no space.

    "Follow-up via x@clinic-archive.invalid.Next review booked." is an address,
    a full stop and a new sentence. The expressions allow dots inside a value,
    so they took "Next" as part of it. A capitalised word after the last dot,
    followed by a space and a lowercase word, is the start of the next
    sentence; the match ends before it.
    """
    match = _RUN_ON_SENTENCE.search(text[start:end])
    if match is None:
        return end
    following = text[end : end + 2]
    if len(following) == 2 and following[0] in " \t" and following[1].islower():
        return start + match.start()
    return end


def _accept_email(text: str, match: "re.Match[str]") -> Optional[Span]:
    start = match.start()
    end = _without_run_on_sentence(text, start, match.end())
    domain = text[start:end].rpartition("@")[2]
    return (start, end) if "." in domain else (start, match.end())


def _accept_url(text: str, match: "re.Match[str]") -> Optional[Span]:
    start = match.start()
    end = _without_run_on_sentence(text, start, match.end())
    # What is left must still be an address: a path, or a host with a dot.
    host = text[start:end].split("://", 1)[-1]
    if host.casefold().startswith("www."):
        host = host[4:]
    return (start, end) if ("/" in host or "." in host) else (start, match.end())


# Context is part of identifier patterns so ordinary clinical numbers are not
# classified as identifiers. These expressions are compiled once at import.
STRUCTURED_PATTERNS: Sequence[_PatternDefinition] = (
    *_PROFESSIONAL_REGISTRATION_DEFINITIONS,
    *(
        _keyed_definition(identifier)
        for identifier in spec.KEYED_IDENTIFIERS
        if identifier.category != "LICENSE_NUMBER"
    ),
    _PatternDefinition(
        "delimited_unique_identifier",
        "OTHER_UNIQUE_ID",
        _DELIMITED_IDENTIFIER,
        group="value",
        accept=_accept_delimited_identifier,
    ),
    _PatternDefinition(
        # Input text written like the sanitizer's own token family ("PROVIDER_00A1",
        # "RECORD_7731X"): a category word, an underscore and a code with a digit.
        # It is never trusted as already removed - the ledger holds only what this
        # run wrote - so it is replaced as unverifiable source text. Upper case
        # only, and only the sanitizer's own category words, so "ICD_10" and
        # "HLA_B27" are untouched.
        "forged_token_shape",
        "OTHER_UNIQUE_ID",
        re.compile(
            r"(?<![A-Za-z0-9_<])(?:PERSON|PROVIDER|PATIENT|FACILITY|LOCATION|ORGANIZATION|"
            r"PHONE|EMAIL|RECORD|ACCOUNT|LICENSE|VEHICLE|DEVICE|URL|ADDRESS|DATE|PLAN|"
            r"REDACTED|IDENTIFIER|ACCESSION|USER|ORG|PLACE|PATIENTID|HEALTH_PLAN)"
            r"(?:_[A-Z]+)*_[A-Z0-9]*\d[A-Z0-9]*(?![A-Za-z0-9_>])"
        ),
    ),
    _PatternDefinition(
        "email_address",
        "EMAIL_ADDRESS",
        _compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
        accept=_accept_email,
    ),
    _PatternDefinition(
        "phone_number",
        "PHONE_NUMBER",
        _compile(
            r"(?<!\w)(?:\+?1[\s.-]?)?" r"(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
        ),
    ),
    _PatternDefinition(
        "national_phone_number",
        "PHONE_NUMBER",
        _NATIONAL_PHONE,
        group="value",
        accept=_accept_national_phone,
    ),
    _PatternDefinition(
        "local_phone_number",
        "PHONE_NUMBER",
        _LOCAL_PHONE,
        group="value",
        accept=_accept_local_phone,
    ),
    _PatternDefinition(
        "announced_phone_number",
        "PHONE_NUMBER",
        _CONTEXT_PHONE,
        group="value",
    ),
    _PatternDefinition(
        "contact_extension_number",
        "PHONE_NUMBER",
        _CONTACT_AFTER_LABEL,
        group="value",
        accept=_accept_contact_number,
    ),
    _PatternDefinition(
        "contact_extension_number_before_label",
        "PHONE_NUMBER",
        _CONTACT_BEFORE_LABEL,
        group="value",
        accept=_accept_contact_number,
    ),
    _PatternDefinition(
        # OCR-spaced and punctuated labels ("E x t .", "Ext .", "EXT:"),
        # decided per occurrence by the shared numeric-context classifier on
        # the original characters, so offsets never move.
        "contact_number_ocr_tolerant_label",
        "PHONE_NUMBER",
        _compile(
            r"(?<![\w.,/\-])(?P<value>\d{2,3}[ \-]\d{1,4}|\d{3,6})(?![\w/\-]|[.,]\d)"
        ),
        group="value",
        accept=_accept_labelled_contact,
    ),
    _PatternDefinition(
        "us_ssn",
        "US_SSN",
        _compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    _PatternDefinition(
        "url",
        "URL",
        _compile(
            # A trailing ; , . : ! ? or quote ends the sentence or field, not
            # the address: dropping it keeps "see <URL_1>; MRI ..." delimited.
            r"\b(?:https?://|www\.)[A-Z0-9.-]+(?:/[A-Z0-9._~:/?#\[\]@!$&'()*+,;=%-]*)?"
            r"(?<![;,.:!?'])"
        ),
        accept=_accept_url,
    ),
    _PatternDefinition(
        "ip_address",
        "IP_ADDRESS",
        # The boundaries reject a longer dotted number without rejecting a
        # trailing sentence period: "accessed from 203.0.113.42." is the
        # common phrasing, and a plain (?![\d.]) never matches it.
        _compile(
            r"(?<!\d)(?<!\d\.)(?:25[0-5]|2[0-4]\d|1?\d?\d)"
            r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?!\d)(?!\.\d)"
        ),
    ),
    _PatternDefinition(
        "numeric_date",
        "DATE_TIME",
        _compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b"),
    ),
    _PatternDefinition(
        "clock_time",
        "TIME",
        _compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?:\s?[AP]M)?\b"),
    ),
)


def _matches(definition: _PatternDefinition, text: str):
    """Every accepted match of one definition, as (match, span)."""
    position = 0
    while position <= len(text):
        match = definition.regex.search(text, position)
        if match is None:
            return
        span = definition.span_for(text, match)
        if span is None:
            position = match.start() + 1
            continue
        yield match, span
        position = max(match.end(), match.start() + 1)


#: Characters that are a hyphen to a reader and to nobody's regular expression:
#: a soft hyphen (what many PDF fonts extract a hyphen as), the Unicode hyphens,
#: dashes and the minus sign. Folded one character for one, so every offset in
#: the folded text is an offset in the original.
_DASH_FOLD = {ord(c): "-" for c in "­‐‑‒–—―−﹘﹣－"}
#: Characters a font's character map extracts in place of ordinary punctuation.
#: U+037E, the Greek question mark, is what several PDF fonts give for ";" and is
#: canonically equivalent to it.
_DASH_FOLD[0x037E] = ";"


def fold_dashes(text: str) -> str:
    """The text with dash variants as '-' and lookalike punctuation as itself.

    Same length, same offsets: the folded text is only for reading.
    """
    return text.translate(_DASH_FOLD) if text else text


class StructuredPatternDetector:
    def detect(self, text: str) -> List[DetectedEntity]:
        text = fold_dashes(text)
        entities: List[DetectedEntity] = []
        for definition in STRUCTURED_PATTERNS:
            for _match, (start, end) in _matches(definition, text):
                entities.append(
                    DetectedEntity(
                        entity_type=definition.entity_type,
                        start=start,
                        end=end,
                        source=SOURCE_STRUCTURED_PATTERN,
                        original_label=definition.name,
                    )
                )
        return entities


# Safe Harbor requires ages over 89 to be aggregated. An age at or below 89 is
# not an identifier on its own and is ordinary clinical fact, so it is left
# alone. A bare number is never treated as an age: an explicit age cue must be
# present, or "the patient received 94 mg" would be destroyed.
_AGE_WITH_CUE = _compile(r"(?:\bage[ds]?\b\s*[:\-]?\s*|\baged\s+)(\d{1,3})\b")
_AGE_WITH_UNIT = _compile(r"\b(\d{1,3})\s*(?:-\s*)?(?:years?|yrs?)\s*(?:-\s*)?old\b")
_AGE_YO = _compile(r"\b(\d{1,3})\s*(?:y/?o|yo)\b")

_AGE_PATTERNS = (_AGE_WITH_CUE, _AGE_WITH_UNIT, _AGE_YO)

#: Age references that cannot be resolved to a number. Safe Harbor turns on
#: whether the age exceeds 89, so an unresolvable reference is escalated
#: rather than guessed at in either direction.
_UNCERTAIN_AGE = _compile(
    r"\b(?:nonagenarian|centenarian|in\s+(?:his|her|their)\s+(?:nineties|90s|"
    r"late\s+eighties)|over\s+(?:ninety|90)|aged\s+over\s+\d{1,3}|"
    r"more\s+than\s+\d{1,3}\s+years\s+old)\b"
)

AGE_SAFE_HARBOR_THRESHOLD = 89


class AgeOverThresholdDetector:
    """Deterministic detector for ages that Safe Harbor requires aggregating.

    Emits ``AGE_OVER_89`` for a resolvable age above 89, spanning the number
    only so the surrounding clinical phrasing survives. Emits
    ``AGE_UNCERTAIN`` for an age reference that cannot be resolved to a
    number, which routes the document to review instead of guessing.
    """

    def detect(self, text: str) -> List[DetectedEntity]:
        entities: List[DetectedEntity] = []
        for pattern in _AGE_PATTERNS:
            for match in pattern.finditer(text):
                try:
                    value = int(match.group(1))
                except (TypeError, ValueError):
                    continue
                if value <= AGE_SAFE_HARBOR_THRESHOLD or value > 130:
                    # At or below the threshold it is ordinary clinical fact;
                    # above 130 it is not a human age and some other rule owns
                    # the number.
                    continue
                entities.append(
                    DetectedEntity(
                        entity_type="AGE_OVER_89",
                        start=match.start(1),
                        end=match.end(1),
                        source=SOURCE_AGE_RULE,
                        original_label="age_over_threshold",
                    )
                )
        for match in _UNCERTAIN_AGE.finditer(text):
            entities.append(
                DetectedEntity(
                    entity_type="AGE_UNCERTAIN",
                    start=match.start(),
                    end=match.end(),
                    source=SOURCE_AGE_RULE,
                    original_label="uncertain_age_reference",
                )
            )
        return entities


_EXACT_STRUCTURED_TYPES = {
    "MEDICAL_RECORD_NUMBER",
    "PATIENT_ID",
    # A professional registration found by its label is as exact as an account
    # number: without this a model span that took the label along ("AHPRA
    # MED0001234567") was wider, outranked the value alone, and removed the label.
    "LICENSE_NUMBER",
    "HEALTH_PLAN_ID",
    "INSURANCE_ID",
    "ACCESSION_NUMBER",
    "ACCOUNT_NUMBER",
    "DEVICE_ID",
    "OTHER_UNIQUE_ID",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "SSN",
    "URL",
    "IP_ADDRESS",
}
_BROAD_TEMPORAL_TYPES = {"DATE", "DATE_TIME", "TIME"}
#: Rule-based sources whose exact identifiers outrank every model span.
_STRUCTURED_SOURCES = frozenset({SOURCE_STRUCTURED_PATTERN, SOURCE_REPEAT_PROPAGATION})
_SOURCE_ORDER = {
    SOURCE_STRUCTURED_PATTERN: 0,
    SOURCE_REPEAT_PROPAGATION: 0,
    SOURCE_CONTEXT_RULE: 1,
    SOURCE_NER: 2,
    SOURCE_STRICT_PROPER_NOUN: 3,
}


def _priority(entity: DetectedEntity) -> int:
    if (
        entity.source in _STRUCTURED_SOURCES
        and entity.entity_type in _EXACT_STRUCTURED_TYPES
    ):
        return 400
    if entity.source == SOURCE_CONTEXT_RULE:
        return 300
    if entity.entity_type not in _BROAD_TEMPORAL_TYPES:
        return 200
    return 100


def resolve_overlaps(
    entities: Iterable[DetectedEntity],
    text_length: int,
) -> List[DetectedEntity]:
    """Select valid, deterministic, non-overlapping spans."""

    valid = {
        entity for entity in entities if 0 <= entity.start < entity.end <= text_length
    }
    # The key must totally order the candidates: set iteration order is not
    # stable across runs, so any tie left here would make redaction
    # non-deterministic. Rule-based findings carry no score and rank above
    # model candidates.
    ordered = sorted(
        valid,
        key=lambda entity: (
            -_priority(entity),
            -(entity.end - entity.start),
            -(1.0 if entity.score is None else entity.score),
            entity.start,
            _SOURCE_ORDER.get(entity.source, 99),
            entity.entity_type,
            entity.original_label or "",
            entity.source,
        ),
    )

    selected: List[DetectedEntity] = []
    for entity in ordered:
        if any(
            entity.start < accepted.end and accepted.start < entity.end
            for accepted in selected
        ):
            continue
        selected.append(entity)

    return sorted(
        selected,
        key=lambda entity: (entity.start, entity.end, entity.entity_type),
    )


# -- completing and repeating confirmed structured identifiers ----------------

#: Categories whose span a model can cut short inside an identifier token.
#: Names, places, dates and organisations are absent: their boundaries are
#: words, and widening them to a neighbouring token would remove clinical text.
IDENTIFIER_LIKE_CATEGORIES: FrozenSet[str] = frozenset(
    (_EXACT_STRUCTURED_TYPES - {"EMAIL_ADDRESS", "URL"})
    | {
        "IDENTIFIER",
        "PASSPORT_NUMBER",
        "DRIVER_LICENSE",
        "LICENSE_NUMBER",
        "CERTIFICATE_NUMBER",
        "VEHICLE_ID",
        "BIOMETRIC_ID",
    }
)

#: A run of letters and digits joined by tight delimiters: a widening candidate.
_IDENTIFIER_TOKEN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9]|" + _UNIQUE_JOIN + r"{1,2}(?=[A-Za-z0-9]))*"
)
#: Six or more digits and nothing else: an identifier-shaped number.
_LONG_DIGIT_RUN = re.compile(r"\d{6,}")


#: How far before a span its announcing words may start.
_HEAD_WINDOW = 64


def _core_span(
    definition: _PatternDefinition, match: "re.Match[str]", span: Span
) -> Span:
    """The value inside a structured match, without the words announcing it."""
    if "value" in definition.regex.groupindex:
        return max(match.start("value"), span[0]), span[1]
    return span


def structured_value(text: str, start: int, end: int) -> Optional[Tuple[str, int, int]]:
    """(category, value start, value end) when a span is one whole identifier.

    Only categories that may be propagated are considered, and the span must
    be matched from its first character to its last - a span that merely
    contains an identifier somewhere is not a confirmed value.
    """
    segment = text[start:end]
    stripped = segment.strip()
    if not stripped:
        return None
    base = start + (len(segment) - len(segment.lstrip()))
    stop = base + len(stripped)
    for definition in STRUCTURED_PATTERNS:
        if definition.entity_type not in spec.REPEATABLE_CATEGORIES:
            continue
        for match, span in _matches(definition, stripped):
            # Only the first match can start at the span's first character.
            if match.start() == 0 and span[1] == len(stripped):
                core_start, core_end = _core_span(definition, match, span)
                return definition.entity_type, base + core_start, base + core_end
            break
        # A value whose announcing words stay in the text sits just after
        # them: "Study ID" then "RX-418273". It is confirmed only when that
        # definition's accepted span is exactly this span.
        window_start = max(0, base - _HEAD_WINDOW)
        for _match, span in _matches(definition, text[window_start:stop]):
            if (window_start + span[0], window_start + span[1]) == (base, stop):
                return definition.entity_type, base, stop
    return None


def propagate_repeated_identifiers(
    entities: Sequence[DetectedEntity],
    text: str,
) -> List[DetectedEntity]:
    """Every other boundary-safe occurrence of a confirmed structured value.

    Once one occurrence of a phone number, account, device identifier, SSN,
    URL, IP address, record, plan or other unique identifier has been
    confirmed as a whole identifier, the same value elsewhere in the document
    is the same identifier, whatever the detectors made of that occurrence.
    Only the spec's repeatable categories are propagated, and only values long
    enough and carrying enough digits that a coincidental match is not
    plausible.

    The value is used only to search this document and is not stored,
    returned or logged; the result carries categories and offsets.
    """
    from services import numeric_context

    found: List[DetectedEntity] = []
    searched = set()
    for entity in entities:
        confirmed = structured_value(text, entity.start, entity.end)
        if confirmed is None:
            continue
        category, core_start, core_end = confirmed
        length = core_end - core_start
        digits = _digits(text[core_start:core_end])
        required_digits = spec.REPEAT_MIN_DIGITS.get(category, length + 1)
        if length < spec.REPEAT_MIN_LENGTH or digits < required_digits:
            continue
        key = (category, text[core_start:core_end].casefold())
        if key in searched:
            continue
        searched.add(key)
        occurrence = re.compile(
            r"(?<![A-Za-z0-9])(?<![A-Za-z0-9][.\-/])"
            + re.escape(text[core_start:core_end])
            + r"(?![A-Za-z0-9])(?![.\-/][A-Za-z0-9])",
            re.IGNORECASE,
        )
        for match in occurrence.finditer(text):
            if any(e.start <= match.start() and match.end() <= e.end for e in entities):
                continue
            # Each occurrence is judged by its own context, never by the value
            # alone: the same digits as a result, dose, year or imaging
            # parameter stay where they are.
            if numeric_context.classify(text, match.start(), match.end()).is_clinical:
                continue
            found.append(
                DetectedEntity(
                    entity_type=category,
                    start=match.start(),
                    end=match.end(),
                    source=SOURCE_REPEAT_PROPAGATION,
                    original_label="repeat_of_structured_identifier",
                )
            )
    return found


def expand_partial_identifiers(
    entities: Sequence[DetectedEntity],
    text: str,
) -> List[DetectedEntity]:
    """Widen a model's identifier span that stops inside a validated identifier.

    A model can label "DEV70000" inside "DEV7000001", or part of a delimited
    code, and replacing only that part leaves the rest behind. A span in an
    identifier category is widened only to what the identifier grammar
    validates: a structured match it overlaps, or a token it cuts through that
    has the shape of a unique identifier or is a long run of digits. Names,
    places and dates are never widened, and a reading, ratio or measurement is
    never a validated identifier.
    """
    structured: Optional[List[Span]] = None
    widened: List[DetectedEntity] = []
    for entity in entities:
        if (
            entity.source in _STRUCTURED_SOURCES
            or entity.source == SOURCE_AGE_RULE
            or entity.entity_type not in IDENTIFIER_LIKE_CATEGORIES
        ):
            widened.append(entity)
            continue
        if structured is None:
            structured = [
                (e.start, e.end) for e in StructuredPatternDetector().detect(text)
            ]
        start, end = entity.start, entity.end
        for other_start, other_end in structured:
            if other_start < end and start < other_end:
                start, end = min(start, other_start), max(end, other_end)
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        line_end = len(text) if line_end < 0 else line_end
        for token in _IDENTIFIER_TOKEN.finditer(text, line_start, line_end):
            cuts = (
                token.start() < start < token.end() or token.start() < end < token.end()
            )
            if not cuts:
                continue
            value = token.group(0)
            validated = is_unique_identifier_value(value) or _LONG_DIGIT_RUN.fullmatch(
                value
            )
            if not validated or _UNIT_AFTER.match(text, token.end()):
                continue
            start, end = min(start, token.start()), max(end, token.end())
        if (start, end) == (entity.start, entity.end):
            widened.append(entity)
        else:
            widened.append(
                DetectedEntity(
                    entity_type=entity.entity_type,
                    start=start,
                    end=end,
                    source=entity.source,
                    score=entity.score,
                    original_label=entity.original_label,
                )
            )
    return widened
