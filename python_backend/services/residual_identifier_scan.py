"""The residual validator's own scan for structured identifiers.

The first pass finds structured identifiers with expressions compiled from
``services.structured_identifier_spec``. If the validator reused those
expressions, one mistake in them would carry an identifier through both
passes unseen - the failure a validator exists to prevent. So this module
reads the same description with different code: it walks whitespace-delimited
tokens, looks back along the line for an announcing word, and judges a token
by its digit layout instead of by a compiled expression.

Its thresholds are looser than the first pass's on purpose. It needs one digit
after an announcing word where the first pass needs several, it has no
analyte or range guard, it flags long unannounced letter-and-digit tokens, and
it joins delimited parts however OCR spaced them. A finding blocks release
unless the token is wholly the sanitizer's own output. A false alarm costs a
human review; a miss releases an identifier.

Three holds cover layouts no first-pass rule describes: a value announced as a
vehicle identifier, a value after a bare record label, and a day left beside a
date the sanitizer replaced in pieces. They only hold a document for review;
none removes anything.

Returns categories and offsets only. Never a value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from services import structured_identifier_spec as spec
from services.phi_detection import (
    SOURCE_RESIDUAL_IDENTIFIER_SCAN,
    DetectedEntity,
    fold_dashes,
)
from services.transformation_provenance import KIND_SHIFTED_DATE

#: A replaced value kept as one token by the audit's placeholder filler; it is
#: the value a label announced (see ``services.search.extraction``).
PLACEHOLDER_FILLER = "█"

#: How many tokens may sit between an announcing word and its value on the
#: same line: "account number on file 4417-22901".
LOOK_BACK_TOKENS = 4
#: Fewest characters of a token after an announcing word.
ANNOUNCED_MIN_LENGTH = 4
#: An unannounced token with letters and at least this many digits reads as an
#: identifier on its own: "DEV7000001", "NCT01234567".
UNANNOUNCED_MIXED_DIGITS = 5
UNANNOUNCED_MIXED_LENGTH = 6
#: An unannounced run of digits alone needs more: "250000" is a count.
UNANNOUNCED_DIGITS = 9

#: Words that announce a vehicle identifier. No first-pass rule covers the
#: category - the models do - so a value announced this way that reaches the
#: output holds the document for review: "reg plate AB12CDE".
VEHICLE_HEADS = frozenset({"plate", "reg", "vin", "vehicle"})

#: A bare record label: "Record 483271", "record no. 48-27131". The first pass
#: keys record numbers on "MRN", "medical record" and "chart number"; a value
#: after "record" alone was released when the detectors missed it. Only an
#: identifier-like value holds - five or more digits, or letters with at least
#: four digits - so "record 120 mg" and "record day 3" stay instructions.
RECORD_HEADS = frozenset({"record", "records"})
RECORD_QUALIFIERS = frozenset({"no", "num", "nr", "number", "id"})
RECORD_MIN_DIGITS = 5
RECORD_MIN_MIXED_DIGITS = 4
_RECORD_VALUE = re.compile(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*")

_EDGE = "\"'`,;:!?[]{}<>#*|="
#: Characters a telephone layout may use: digits and its separators.
_LAYOUT = frozenset(".-()+") | {str(digit) for digit in range(10)}
_SCHEMES = ("http://", "https://", "ftp://", "www.")
_SSN_CONTEXT = ("ssn", "social")
_UNITS = frozenset(
    unit for unit in spec.MEASUREMENT_UNITS if len(unit) > 1 or unit == "%"
)
#: Punctuation that glues words into one whitespace token: "vin-1HGCM...",
#: "AB12CDE,seen,stable".
_GLUE = re.compile(r"[-,;:/|_#=]+")

_JOIN_CHARACTERS = "[" + re.escape(spec.UNIQUE_DELIMITERS) + "]"
#: Alphanumeric parts joined by a tight delimiter, or by a doubled or tripled
#: one with any horizontal spacing around it: however OCR laid it out.
_JOINED_PARTS = re.compile(
    r"[A-Za-z0-9]+(?:(?:[ \t]*("
    + _JOIN_CHARACTERS
    + r")[ \t]*\1(?:[ \t]*\1)?[ \t]*|"
    + _JOIN_CHARACTERS
    + r")[A-Za-z0-9]+){2,8}"
)

#: A number that could be a day of the month. A time, decimal, range, fraction
#: or longer number is not one.
_DAY = re.compile(r"(?<![\w.:/-])(?:3[01]|[12]\d|[1-9])(?:st|nd|rd|th)?(?!\w|[.:/-]\d)")
#: What may sit between the parts of one written date on a single line.
_DATE_GAP = " \t,./-"
_DATE_TYPES = frozenset({"DATE", "DATE_TIME"})


def _residual_ocr_label(label: str) -> str:
    words = []
    for word in label.split():
        if len(word) <= 4 and word.isalpha():
            words.append(
                r"[ .]*".join(re.escape(character) for character in word) + r"\.?"
            )
        else:
            words.append(re.escape(word))
    return r"[ \t]+".join(words)


_REGISTRATION_LABEL = "|".join(
    _residual_ocr_label(label)
    for label in sorted(spec.PROFESSIONAL_REGISTRATION_LABELS, key=len, reverse=True)
)
_REGISTRATION_QUALIFIER = "|".join(
    _residual_ocr_label(item)
    for item in sorted(spec.PROFESSIONAL_REGISTRATION_QUALIFIERS, key=len, reverse=True)
)
#: Plain, or spaced out one character at a time as OCR and typists write it.
_REGISTRATION_VALUE = (
    r"(?:[A-Za-z0-9](?:[ ][A-Za-z0-9]){5,23}"
    r"|[A-Za-z0-9](?:[A-Za-z0-9]|[-/](?=[A-Za-z0-9])){3,23})"
)
_REGISTRATION_AFTER = re.compile(
    r"(?<![A-Za-z0-9])(?:" + _REGISTRATION_LABEL + r")(?![A-Za-z])"
    r"(?:[ \t]+(?:" + _REGISTRATION_QUALIFIER + r")(?![A-Za-z]))?"
    r"[ \t]*[.:#=\-]?[ \t]*(?P<value>" + _REGISTRATION_VALUE + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_REGISTRATION_BEFORE = re.compile(
    r"(?<![A-Za-z0-9])(?P<value>" + _REGISTRATION_VALUE + r")(?![A-Za-z0-9])"
    r"[ \t]*(?:\([ \t]*)?(?:" + _REGISTRATION_LABEL + r")(?![A-Za-z])(?:[ \t]*\))?",
    re.IGNORECASE,
)


def _professional_registrations(text: str) -> List[DetectedEntity]:
    """Line-bounded registration values, found independently of first-pass regexes."""
    found: List[DetectedEntity] = []
    occupied = set()
    for expression in (_REGISTRATION_AFTER, _REGISTRATION_BEFORE):
        for match in expression.finditer(text or ""):
            start, end = match.span("value")
            value = text[start:end]
            if sum(character.isdigit() for character in value) < 3:
                continue
            if _followed_by_unit(
                _tokens(text),
                next(
                    (
                        index
                        for index, token in enumerate(_tokens(text))
                        if token.start <= start and end <= token.end
                    ),
                    -1,
                ),
            ):
                continue
            if (start, end) not in occupied:
                occupied.add((start, end))
                found.append(
                    _entity("LICENSE_NUMBER", start, end, "professional_registration")
                )
    return found


@dataclass(frozen=True)
class _Token:
    start: int
    end: int
    word: str
    line: int
    #: The token ends a sentence: its stripped tail carried . ; ! or ?, and
    #: it is not an abbreviation ("No.", "Ref.", an initial).
    stops: bool = False
    #: The original spelling has a lower-case letter: an analyte ("HbA1c",
    #: "SpO2") is written that way; a record value rarely is.
    lower: bool = False


def _tokens(text: str) -> List[_Token]:
    tokens: List[_Token] = []
    line = 0
    cursor = 0
    for chunk in re.finditer(r"\S+", text):
        line += text.count("\n", cursor, chunk.start())
        cursor = chunk.start()
        start, end = chunk.start(), chunk.end()
        changed = True
        while changed and start < end:
            changed = False
            if text[start] in _EDGE:
                start += 1
                changed = True
            elif text[end - 1] in _EDGE or text[end - 1] == ".":
                end -= 1
                changed = True
            elif text[end - 1] == ")" and text.count("(", start, end) < text.count(
                ")", start, end
            ):
                end -= 1
                changed = True
            elif text[start] == "(" and text.count("(", start, end) > text.count(
                ")", start, end
            ):
                start += 1
                changed = True
            elif text[start:end].casefold().endswith(("'s", "’s")) and end - start > 2:
                end -= 2
                changed = True
        if start < end:
            word = text[start:end].casefold()
            stops = bool(
                re.search(r"[.;!?]", text[end : chunk.end()])
            ) and not _ABBREVIATION.fullmatch(word + ".")
            tokens.append(
                _Token(
                    start,
                    end,
                    word,
                    line,
                    stops,
                    any(c.islower() for c in text[start:end]),
                )
            )
    return tokens


def _digit_groups(word: str) -> List[int]:
    return [len(group) for group in re.findall(r"\d+", word)]


def _is_ip_address(word: str) -> bool:
    parts = word.split(".")
    return len(parts) == 4 and all(
        part.isdigit() and len(part) <= 3 and int(part) <= 255 for part in parts
    )


def _is_ssn(word: str) -> bool:
    return bool(re.fullmatch(r"\d{3}([-.])\d{2}\1\d{4}", word))


def _is_phone_layout(word: str) -> bool:
    if not set(word) <= _LAYOUT:
        return False
    groups = _digit_groups(word)
    if groups == list(spec.PHONE.local_groups):
        return True
    digits = sum(groups)
    national = spec.PHONE.national_digits
    return digits in (national, national + 1) and len(groups) >= 2


def _followed_by_unit(tokens: Sequence[_Token], index: int) -> bool:
    if (
        index < 0
        or index + 1 >= len(tokens)
        or tokens[index + 1].line != tokens[index].line
    ):
        return False
    return tokens[index + 1].word.rstrip(".,;") in _UNITS


#: A word ending in a full stop that does not end a sentence: "No.", "Ref.",
#: a single initial, a dotted abbreviation ("e.g.").
_ABBREVIATION = re.compile(
    r"(?i)(?:no|nos|ref|reg|tel|ext|dr|mr|mrs|ms|st|[a-z]|(?:[a-z]\.)+[a-z])\."
)


def _identifier_shaped(word: str) -> bool:
    """A token whose own digit layout reads as an identifier, announced or not."""
    digits = sum(_digit_groups(word))
    letters = sum(character.isalpha() for character in word)
    return digits >= 6 or bool(letters and digits >= UNANNOUNCED_MIXED_DIGITS)


def _head_consumed(tokens: Sequence[_Token], head_position: int, index: int) -> bool:
    """Whether the head's own value already closed its field before this token.

    A label announces one value. "Chart no. M-443003; HbA1c 7.2%": the record
    number completes the field with its terminator, so the analyte reading
    after it belongs to a different field and is not announced by "chart".
    A token that is identifier-shaped on its own is never excused this way,
    so a second record number after the first is still found.
    """
    if _identifier_shaped(tokens[index].word):
        return False
    line = tokens[index].line
    for position in range(head_position + 1, index):
        token = tokens[position]
        if (
            token.line == line
            and token.stops
            and token.word not in _UNITS
            and (
                token.word == PLACEHOLDER_FILLER
                or any(character.isdigit() for character in token.word)
            )
        ):
            return True
    return False


def _head_distance(
    tokens: Sequence[_Token], index: int, heads: Sequence[str]
) -> Optional[int]:
    """Tokens between the nearest of ``heads`` and this token, on this line."""
    line = tokens[index].line
    first = max(0, index - LOOK_BACK_TOKENS - 3)
    # A label announces a mixed-case word only in its own sentence: "MRN
    # 51820374, bleep 4417. HbA1c 61" does not make the analyte a record
    # number. Digit and upper-case values still reach back, privacy first.
    for position in range(index - 1, first - 1, -1) if tokens[index].lower else ():
        if tokens[position].stops and tokens[position].word not in heads:
            first = position + 1
            break
    words = [
        (position, tokens[position].word.strip(".:#-"))
        for position in range(first, index)
        if tokens[position].line == line
    ]
    nearest = None
    for head in heads:
        parts = head.split()
        for offset in range(len(words) - len(parts) + 1):
            if [
                word for _position, word in words[offset : offset + len(parts)]
            ] != parts:
                continue
            head_position = words[offset + len(parts) - 1][0]
            between = index - head_position - 1
            if between <= LOOK_BACK_TOKENS and (nearest is None or between < nearest):
                if between and _head_consumed(tokens, head_position, index):
                    continue
                nearest = between
    return nearest


_DIAGNOSIS_CODE = re.compile(spec.DIAGNOSIS_CODE_PATTERN)


_SEPARATOR_STRIP = "." + spec.DIAGNOSIS_FIELD_SEPARATORS


def _is_governed_diagnosis_code(tokens: Sequence[_Token], index: int) -> bool:
    """This whole token is an ICD code in an explicit diagnosis field.

    The first pass exempts the same thing on the same terms, from the same
    grammar; see ``services.structured_identifier_spec``. Tokens here have
    already had their edge punctuation stripped, so matching the whole word
    is what "not the head of a longer string" means at this level.

    The head has to be the nearest word before the code on the same line,
    with nothing between them but separators - the full-width colon and the
    dash family included, because scanned text uses them. A sentence that
    merely mentions the word governs nothing, and an identifier announced by
    its own head is caught earlier and never reaches this.
    """
    if not _DIAGNOSIS_CODE.fullmatch(tokens[index].word):
        return False
    line = tokens[index].line
    position = index - 1
    # Step back over separator-only tokens: "Dx - J45.909" writes the dash on
    # its own, and so does OCR of a printed field.
    while position >= 0 and tokens[position].line == line:
        word = tokens[position].word.strip(_SEPARATOR_STRIP)
        if word:
            break
        position -= 1
    if position < 0 or tokens[position].line != line:
        return False
    head = tokens[position].word.strip(_SEPARATOR_STRIP)
    if head in spec.DIAGNOSIS_FIELD_HEADS:
        return True
    # "Primary diagnosis: J45.909" - one approved qualifier may precede.
    if head in spec.DIAGNOSIS_FIELD_QUALIFIERS and position + 1 < index:
        following = tokens[position + 1].word.strip(_SEPARATOR_STRIP)
        return following in spec.DIAGNOSIS_FIELD_HEADS
    return False


def _attached_head(word: str, heads: Sequence[str]) -> Optional[int]:
    """Offset of the value when a head is written onto it: "mrn-12345"."""
    for head in heads:
        if " " in head:
            continue
        if word.startswith(head) and len(word) > len(head) and word[len(head)] in ":#-":
            return len(head) + 1
    return None


def _entity(category: str, start: int, end: int, rule: str) -> DetectedEntity:
    return DetectedEntity(
        entity_type=category,
        start=start,
        end=end,
        source=SOURCE_RESIDUAL_IDENTIFIER_SCAN,
        original_label=rule,
    )


def _phone(tokens: Sequence[_Token], index: int) -> Optional[Tuple[int, str]]:
    """(last token index, rule) when a phone number starts at this token."""
    word = tokens[index].word
    context = _head_distance(tokens, index, spec.PHONE.context_words) is not None
    attached = _attached_head(word, spec.PHONE.context_words)
    if attached is not None and set(word[attached:]) <= _LAYOUT:
        rest = word[attached:]
        if _is_phone_layout(rest) or (rest.isdigit() and len(rest) >= 7):
            return index, "announced_phone"
    if _is_phone_layout(word) and not _followed_by_unit(tokens, index):
        return index, "phone_layout"
    if word.isdigit() and len(word) == sum(spec.PHONE.local_groups) and context:
        return index, "announced_phone"
    if (
        index + 1 < len(tokens)
        and tokens[index + 1].line == tokens[index].line
        and set(word) <= _LAYOUT
        and _digit_groups(word) == [spec.PHONE.local_groups[0]]
        and set(tokens[index + 1].word) <= _LAYOUT
        and _digit_groups(tokens[index + 1].word) == [spec.PHONE.local_groups[1]]
        and (word.startswith("(") or context)
    ):
        return index + 1, "split_phone_layout"
    spaced = _spaced_national_phone(tokens, index)
    if spaced is not None:
        return spaced, "spaced_national_phone"
    return None


def _spaced_national_phone(tokens: Sequence[_Token], index: int) -> Optional[int]:
    """Last token of a trunk-zero or plus-prefixed number written in spaced groups.

    "0161 496 0731", "07700 900412" and "+44 20 7946 0873" arrive as two to
    five tokens of digits on one line. The digit total decides, from the spec's
    ranges, so a list of small counts is not taken for a number.
    """
    word = tokens[index].word
    international = (
        word.startswith("+") and word[1:].isdigit() and 1 <= len(word) - 1 <= 3
    )
    trunk = word.isdigit() and word.startswith("0") and 3 <= len(word) <= 5
    if not (international or trunk):
        return None
    digits = len(word.lstrip("+"))
    last = index
    for follower in range(index + 1, min(index + 5, len(tokens))):
        token = tokens[follower]
        if (
            token.line != tokens[index].line
            or not token.word.isdigit()
            or not 2 <= len(token.word) <= 6
        ):
            break
        digits += len(token.word)
        last = follower
        low, high = (
            spec.INTERNATIONAL_PHONE_DIGITS
            if international
            else spec.TRUNK_PHONE_DIGITS
        )
        if (
            low <= digits <= high
            and last > index
            and not _followed_by_unit(tokens, last)
        ):
            return last
        if digits > high:
            break
    return None


#: A sanitizer token with its brackets stripped and case folded:
#: "vehicle_id_2", "person_485", "redacted_postal_code".
_SANITIZER_TOKEN_WORD = re.compile(r"(?:[a-z]+_)+\d{1,6}|redacted(?:_[a-z]+)+")


def _vehicle_value(tokens: Sequence[_Token], index: int) -> bool:
    """A value announced as a vehicle identifier: "reg plate AB12CDE".

    Words glued together by punctuation count separately, so a sentence run on
    with commas or a label joined by a hyphen still announces the value.
    """
    word = tokens[index].word
    parts = [part for part in _GLUE.split(word) if part]
    for position, part in enumerate(parts):
        if part in VEHICLE_HEADS and any(
            any(character.isdigit() for character in later)
            for later in parts[position + 1 :]
        ):
            return True
    if len(word) < ANNOUNCED_MIN_LENGTH:
        return False
    line = tokens[index].line
    for position in range(max(0, index - LOOK_BACK_TOKENS - 1), index):
        if _SANITIZER_TOKEN_WORD.fullmatch(tokens[position].word):
            # "<VEHICLE_ID_2>" names a category the sanitizer already replaced;
            # the word inside it announces nothing about its neighbours.
            continue
        if tokens[position].line == line and any(
            part.strip(".") in VEHICLE_HEADS
            for part in _GLUE.split(tokens[position].word)
        ):
            if index - position > 1 and _head_consumed(tokens, position, index):
                continue
            return True
    return False


def _record_value_like(value: str) -> bool:
    value = value.strip(".:#-")
    if not _RECORD_VALUE.fullmatch(value):
        return False
    digits = sum(character.isdigit() for character in value)
    if any(character.isalpha() for character in value):
        return digits >= RECORD_MIN_MIXED_DIGITS
    return digits >= RECORD_MIN_DIGITS


def _record_label_value(tokens: Sequence[_Token], index: int) -> bool:
    """An identifier-like value directly after a bare record label.

    "Record 483271", "record no: 483271", "Record #AB-48213", "record-483271".
    The label must sit immediately before the value, optionally with "no" or
    "number" between them, and a value followed by a unit is a measurement.
    """
    word = tokens[index].word
    attached = _attached_head(word, tuple(RECORD_HEADS))
    if attached is not None:
        return _record_value_like(word[attached:])
    if not _record_value_like(word) or _followed_by_unit(tokens, index):
        return False
    line = tokens[index].line
    previous = index - 1
    if (
        previous >= 0
        and tokens[previous].line == line
        and tokens[previous].word.strip(".:#-") in RECORD_QUALIFIERS
    ):
        previous -= 1
    return (
        previous >= 0
        and tokens[previous].line == line
        and tokens[previous].word.strip(".:#-") in RECORD_HEADS
    )


def _joined_identifiers(text: str) -> List[DetectedEntity]:
    """Delimited codes, spaced or not: "K42;;7315;;QD", "K42 ;; 7315 ;; QD".

    Looser than the first pass: any doubled delimiter or two kinds, and a part
    that mixes letters and digits is enough without a part of several digits.
    """
    found: List[DetectedEntity] = []
    for match in _JOINED_PARTS.finditer(text):
        value = match.group(0)
        digits = sum(character.isdigit() for character in value)
        if digits < spec.UNIQUE_MIN_DIGITS or not any(c.isalpha() for c in value):
            continue
        runs = re.findall(_JOIN_CHARACTERS + "+", re.sub(r"[ \t]", "", value))
        doubled = any(len(run) >= 2 for run in runs)
        kinds = {character for run in runs for character in run}
        if not doubled and len(kinds) < 2:
            continue
        parts = re.findall(r"[A-Za-z0-9]+", value)
        mixed = any(
            any(c.isalpha() for c in part) and any(c.isdigit() for c in part)
            for part in parts
        )
        several = any(
            sum(c.isdigit() for c in part) >= spec.UNIQUE_MIN_PART_DIGITS
            for part in parts
        )
        if mixed or several:
            found.append(
                _entity(
                    "OTHER_UNIQUE_ID", match.start(), match.end(), "delimited_parts"
                )
            )
    return found


_PLACEHOLDER_SHAPE = re.compile(r"<REDACTED_[A-Z0-9_]+>")


def foreign_token_entities(text: str) -> List[DetectedEntity]:
    """Tokens written like the sanitizer's own, found in text it has not yet written.

    On input, nothing in a document that has not been through
    this run can be a token this run wrote, so ``<ACCOUNT_ID_2>`` or
    ``<REDACTED_DATE>`` in the input is somebody else's text that happens to look
    like ours. Left in place it would be indistinguishable from a real
    replacement, and whether a model happened to claim it decided whether it was
    replaced. It is replaced like any other unverifiable identifier-shaped value.
    This is not part of ``scan``. On written output, callers must classify
    these findings against the current-run provenance; a shape alone is never
    evidence that the sanitizer wrote it.
    """
    from services.surrogates import (
        HASH_SURROGATE_PATTERN,
        LEGACY_SURROGATE_PATTERN,
        SURROGATE_PATTERN,
    )

    found = [
        _entity("OTHER_UNIQUE_ID", match.start(), match.end(), "foreign_token_shape")
        for pattern in (
            SURROGATE_PATTERN,
            LEGACY_SURROGATE_PATTERN,
            HASH_SURROGATE_PATTERN,
            _PLACEHOLDER_SHAPE,
        )
        for match in pattern.finditer(text)
    ]
    return sorted(found, key=lambda entity: entity.start)


def scan(text: str) -> List[DetectedEntity]:
    """Structured identifiers in ``text``, by category and offset."""
    text = fold_dashes(text)
    found: List[DetectedEntity] = _professional_registrations(
        text
    ) + _joined_identifiers(text)
    joined = [(entity.start, entity.end) for entity in found]
    tokens = _tokens(text)
    consumed = -1
    for index, token in enumerate(tokens):
        if index <= consumed:
            continue
        if any(start <= token.start and token.end <= end for start, end in joined):
            continue
        word = token.word
        digits = sum(_digit_groups(word))

        if word.startswith(_SCHEMES):
            found.append(_entity("URL", token.start, token.end, "url_scheme"))
            continue
        if "@" in word and "." in word.split("@")[-1]:
            found.append(
                _entity("EMAIL_ADDRESS", token.start, token.end, "email_shape")
            )
            continue
        if digits == 0:
            continue
        if _is_ip_address(word):
            found.append(_entity("IP_ADDRESS", token.start, token.end, "dotted_quad"))
            continue
        if _is_ssn(word) or (
            word.isdigit()
            and len(word) == 9
            and _head_distance(tokens, index, _SSN_CONTEXT) is not None
        ):
            found.append(_entity("US_SSN", token.start, token.end, "ssn_layout"))
            continue
        phone = _phone(tokens, index)
        if phone is not None:
            last, rule = phone
            found.append(_entity("PHONE_NUMBER", token.start, tokens[last].end, rule))
            consumed = last
            continue
        if _vehicle_value(tokens, index):
            found.append(
                _entity("VEHICLE_ID", token.start, token.end, "announced_vehicle_value")
            )
            continue
        if _record_label_value(tokens, index):
            found.append(
                _entity(
                    "MEDICAL_RECORD_NUMBER",
                    token.start,
                    token.end,
                    "bare_record_label_value",
                )
            )
            continue

        keyed = _keyed_category(tokens, index)
        if keyed is not None:
            category, offset = keyed
            found.append(
                _entity(category, token.start + offset, token.end, "announced_value")
            )
            continue

        if _is_governed_diagnosis_code(tokens, index):
            continue

        letters = sum(character.isalpha() for character in word)
        if (
            letters
            and digits >= UNANNOUNCED_MIXED_DIGITS
            and len(word) >= UNANNOUNCED_MIXED_LENGTH
        ):
            found.append(
                _entity("IDENTIFIER", token.start, token.end, "dense_mixed_token")
            )
        elif word.isdigit() and len(word) >= UNANNOUNCED_DIGITS:
            found.append(
                _entity("IDENTIFIER", token.start, token.end, "long_digit_run")
            )
    return [_whole_bracketed_token(text, entity) for entity in found]


def _whole_bracketed_token(text: str, entity: DetectedEntity) -> DetectedEntity:
    """A span inside "<...>" covers the whole token, brackets included.

    Tokens strip their edge punctuation, so the value inside an existing
    placeholder ("<RECORD_ID_1>") is found without its brackets. Replacing only
    the inside would leave "<" + new token + ">" behind.
    """
    start, end = entity.start, entity.end
    if (
        start > 0
        and end < len(text)
        and text[start - 1] == "<"
        and text[end] == ">"
        and not any(character in text[start:end] for character in "<> \t\n")
    ):
        return DetectedEntity(
            entity_type=entity.entity_type,
            start=start - 1,
            end=end + 1,
            source=entity.source,
            score=entity.score,
            original_label=entity.original_label,
        )
    return entity


def _keyed_category(tokens: Sequence[_Token], index: int) -> Optional[Tuple[str, int]]:
    """The category of the nearest announcing word, and the value's offset."""
    word = tokens[index].word
    for identifier in spec.KEYED_IDENTIFIERS:
        if identifier.category == "LICENSE_NUMBER":
            continue
        offset = _attached_head(word, identifier.heads)
        if offset is not None and any(
            character.isdigit() for character in word[offset:]
        ):
            return identifier.category, offset
    if len(word) < ANNOUNCED_MIN_LENGTH:
        return None
    if word in spec.QUALIFIERS or any(
        word in identifier.heads for identifier in spec.KEYED_IDENTIFIERS
    ):
        return None
    nearest: Optional[Tuple[int, str]] = None
    for identifier in spec.KEYED_IDENTIFIERS:
        if identifier.category == "LICENSE_NUMBER":
            continue
        distance = _head_distance(tokens, index, identifier.heads)
        if distance is not None and (nearest is None or distance < nearest[0]):
            nearest = (distance, identifier.category)
    return (nearest[1], 0) if nearest else None


def _reaches_replaced_date(
    text: str, regions: Sequence, cursor: int, forward: bool
) -> bool:
    """Whether generated regions, joined only by date punctuation, lead to a date."""
    while True:
        if forward:
            while cursor < len(text) and text[cursor] in _DATE_GAP:
                cursor += 1
            region = next((r for r in regions if r.start == cursor), None)
        else:
            while cursor > 0 and text[cursor - 1] in _DATE_GAP:
                cursor -= 1
            region = next((r for r in regions if r.end == cursor), None)
        if region is None:
            return False
        if region.entity_type in _DATE_TYPES or region.kind == KIND_SHIFTED_DATE:
            return True
        cursor = region.end if forward else region.start


def date_fragments(text: str, regions: Sequence) -> List[DetectedEntity]:
    """Days left beside a date the sanitizer replaced in pieces.

    "discharged on 14 PATIENT_001 <REDACTED_DATE>": the month was read as a
    name and the year as a date, and the day was covered by neither. The
    validator never holds the value, so it judges the layout - a number that
    could be a day, joined to a replaced date by nothing but spaces, date
    punctuation and other generated tokens. A number before a unit is a
    measurement, not a day.
    """
    found: List[DetectedEntity] = []
    if not regions:
        return found
    for match in _DAY.finditer(text):
        start, end = match.start(), match.end()
        if any(region.start <= start and end <= region.end for region in regions):
            continue
        following = re.match(r"[ \t]*([^\s\d]+)", text[end:])
        if following and following.group(1).casefold().rstrip(".,;") in _UNITS:
            continue
        if _reaches_replaced_date(text, regions, end, True) or _reaches_replaced_date(
            text, regions, start, False
        ):
            found.append(_entity("DATE", start, end, "day_beside_replaced_date"))
    return found
