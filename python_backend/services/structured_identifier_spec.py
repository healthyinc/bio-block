"""One description of structured-identifier grammar, read by two passes.

The first pass and the residual validator both have to recognise a phone
number, an account number and a device identifier. When each pass kept its
own prefixes they drifted: the Safe Harbor table knew "UDI", the text first
pass did not, and acceptance-v3 released a document through that gap.

This module is the grammar as data - the words that announce an identifier,
the digits its value must carry, and the measurement vocabulary that must
never be read as one. It contains no detector. The first pass compiles
precise expressions from it (``services.phi_detection``). The residual
validator walks tokens against it with its own code and looser thresholds
(``services.residual_identifier_scan``). A mistake in one implementation is
caught by the other; only a mistake in this description is shared, which is
why it is small and pinned by tests.

Nothing here holds, produces or logs a value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Mapping, Optional, Tuple

#: Bumped whenever the grammar below changes meaning.
#:
#: v2 adds unusual unique identifiers: codes after an announcing word, and
#: unannounced codes joined by distinctive delimiters ("HT::5820-J4::MV").
#: v3 adds the diagnosis-code fields below, the first entry here that says a
#: shape is *not* an identifier in a given context rather than that it is.
#: v4 describes the whole governed field - head, separator and code - because
#: a model proposes the label together with the value often enough that
#: reading only the value left the code destroyed whenever it did.
SPEC_VERSION = "structured-identifier-spec-v6"


@dataclass(frozen=True)
class KeyedIdentifier:
    """An identifier announced by a word in front of it: "acct 4417-22901"."""

    category: str
    #: Words or phrases that announce the value, lower case. Words inside a
    #: phrase match any run of whitespace.
    heads: Tuple[str, ...]
    #: Fewest characters in the value itself, the head excluded.
    min_length: int
    #: Fewest digits the value must carry. A word such as "reviewed" after
    #: "device" is not a serial number.
    min_digits: int
    #: Characters a value may use anywhere besides letters, digits and single
    #: hyphens.
    extra_characters: str = ""
    #: Punctuation allowed only *between* a value's characters, singly or
    #: doubled: "K42;;7315;;QD". Never at either end.
    inner_delimiters: str = ""
    #: When set, a value without a letter needs at least this many digits and
    #: must not be a numeric range or a number before a unit: "reference
    #: 135-145 mmol/L" is a reference range, not a reference.
    digits_without_letters: Optional[int] = None
    #: Whether the replaced span includes the announcing words. Off by
    #: default: "Medical record number:" says what the field is and identifies
    #: nobody, so only the value is replaced and the label stays, in the text
    #: and in the PDF box that covers it. A rule that must consume its head can
    #: still ask for it.
    span_includes_head: bool = False
    #: Whether a value written as digit groups separated by single spaces
    #: ("2951 66420 1") is one value. Only a run of digit-only groups that ends
    #: its field counts, so "12345 6 mg" and "12345 14 March" are not extended.
    grouped_digits: bool = False


#: Fewest and most digit groups after the first that a grouped value may carry,
#: and the most digits in one of them.
MAX_GROUPED_EXTRA_GROUPS = 3
MAX_GROUP_DIGITS = 6


#: Words that may sit between a head and its value: "account number",
#: "serial no.", "MRN #".
QUALIFIERS: Tuple[str, ...] = ("number", "num", "no", "nr", "#", "id", "identifier")

# Professional identifiers have layouts that vary by regulator, while their
# label is strong deterministic evidence.  These labels are consumed on the
# same line by the first pass and the independently implemented residual
# scanner.  Acronyms may be OCR-spaced by either implementation.
PROFESSIONAL_REGISTRATION_LABELS: Tuple[str, ...] = (
    "medical council registration",
    "medical council registration number",
    "provider registration",
    "provider registration number",
    "professional registration",
    "professional registration number",
    "registration",
    "licence",
    "license",
    "licence number",
    "license number",
    "prescriber number",
    "provider number",
    "gmc registration",
    "nmc pin",
    "gmc",
    "nmc",
    "hcpc",
    "gdc",
    "gphc",
    "npi",
    "dea",
    # Regulators outside the UK and US whose registers are written the same way.
    "ahpra",
    "mcnz",
    "cpso",
    "hpcsa",
    "rcvs",
    "goc",
    "gosc",
)
PROFESSIONAL_REGISTRATION_QUALIFIERS: Tuple[str, ...] = (
    "number",
    "num",
    "no",
    "nr",
    "pin",
    "id",
    "identifier",
    "registration",
    "reg",
)

KEYED_IDENTIFIERS: Tuple[KeyedIdentifier, ...] = (
    KeyedIdentifier(
        "MEDICAL_RECORD_NUMBER",
        heads=(
            "mrn",
            "medical record",
            "hospital number",
            "hospital no",
            "chart number",
            "chart no",
        ),
        min_length=5,
        min_digits=1,
    ),
    KeyedIdentifier(
        "PATIENT_ID",
        heads=("patient id", "patient identifier", "patient number", "pt id"),
        min_length=4,
        min_digits=1,
    ),
    KeyedIdentifier(
        "HEALTH_PLAN_ID",
        heads=(
            "health plan",
            "health plan beneficiary",
            "beneficiary id",
            "insurance id",
            "insurance number",
            "policy id",
            "policy number",
            "member id",
            "subscriber id",
            # National schemes whose card number is written in digit groups.
            "medicare",
            "medicaid",
            "health card",
            "nhs",
        ),
        min_length=6,
        min_digits=1,
        grouped_digits=True,
    ),
    KeyedIdentifier(
        "ACCESSION_NUMBER",
        heads=("accession", "acc no"),
        min_length=5,
        min_digits=1,
    ),
    KeyedIdentifier(
        "ACCOUNT_NUMBER",
        heads=("account", "acct", "acc't", "accnt", "a/c"),
        min_length=5,
        min_digits=5,
    ),
    KeyedIdentifier(
        "DEVICE_ID",
        heads=(
            "device",
            "implant",
            "equipment",
            "serial",
            "s/n",
            "sn",
            "udi",
            "udi-di",
            "udi-pi",
            "unique device identifier",
        ),
        min_length=6,
        min_digits=2,
        # GS1 and HIBCC device identifiers carry application identifiers in
        # parentheses and a leading plus.
        extra_characters="()+",
    ),
    KeyedIdentifier(
        # A professional registration or licence number after its label: the
        # label stays, the number is an identifier however short its shape.
        "LICENSE_NUMBER",
        heads=(
            "gmc",
            "gmc registration",
            "nmc",
            "nmc pin",
            "hcpc",
            "gdc",
            "gphc",
            "registration",
            "professional registration",
            "licence",
            "license",
            "licence number",
            "license number",
            "prescriber number",
            "provider number",
            "npi",
            "dea",
            "ahpra",
            "mcnz",
            "cpso",
            "hpcsa",
            "rcvs",
            "goc",
            "gosc",
        ),
        min_length=5,
        min_digits=4,
        inner_delimiters="-/",
        span_includes_head=False,
    ),
    KeyedIdentifier(
        "OTHER_UNIQUE_ID",
        heads=(
            "reference",
            "ref",
            "local reference",
            "study code",
            "legacy code",
            "study id",
            "subject id",
            "participant id",
            "enrolment id",
            "enrollment id",
            "case id",
            "case number",
            "tracking number",
            "record locator",
            "unique id",
            "unique identifier",
            "sample id",
            "specimen id",
            "barcode",
        ),
        min_length=6,
        min_digits=4,
        inner_delimiters=";:/|_.",
        digits_without_letters=6,
        span_includes_head=False,
    ),
)

#: Punctuation that joins the parts of an unannounced delimited identifier.
UNIQUE_DELIMITERS = ";:/|_-"
#: Fewest digits in any unique identifier.
UNIQUE_MIN_DIGITS = 4
#: Some part of a delimited identifier carries at least this many digits.
#: Spinal levels ("C3-C4/C5-C6") and similar clinical shorthand never do.
UNIQUE_MIN_PART_DIGITS = 3
#: An unannounced delimited identifier has at least this many parts, and a
#: doubled delimiter or two different delimiters.
UNIQUE_MIN_PARTS_UNANNOUNCED = 3
#: A single letter-and-digit token needs this length to read as an identifier.
UNIQUE_MIN_LENGTH_UNDELIMITED = 6


@dataclass(frozen=True)
class PhoneGrammar:
    """Telephone numbers, including short local forms with no area code."""

    #: Words that announce a number. With one present, a space or no
    #: separator at all is accepted between the digit groups.
    context_words: Tuple[str, ...]
    #: Separators accepted between a local number's groups with no context
    #: word: "555.0142" and "555-0142", never "555 0142" or "5550142".
    contextless_separators: str
    #: A local number is exactly three digits then four.
    local_groups: Tuple[int, int]
    #: A full national number, area code included.
    national_digits: int


PHONE = PhoneGrammar(
    context_words=(
        "phone",
        "telephone",
        "tel",
        "ph",
        "mobile",
        "cell",
        "landline",
        "contact",
        "call",
        "ring",
        "dial",
        "fax",
        "pager",
    ),
    contextless_separators=".-",
    local_groups=(3, 4),
    national_digits=10,
)

#: Numbers that are not North American. A national number written with its
#: trunk zero ("0161 496 0731", "07700 900412") has 10 or 11 digits, and an
#: international one ("+44 20 7946 0873") 8 to 15 after its plus sign. Both
#: first pass and residual scan read these ranges from here.
TRUNK_PHONE_DIGITS: Tuple[int, int] = (10, 11)
INTERNATIONAL_PHONE_DIGITS: Tuple[int, int] = (8, 15)

#: Diagnosis-code fields. A complete ICD-shaped token governed by one of
#: these is a diagnosis, not an identifier.
#:
#: The shapes genuinely collide: "J45.909" carries a letter and five digits,
#: exactly as an account number can, and nothing in the token itself settles
#: which it is. Only the field in front of it does. This lives here rather
#: than in one pass because the two passes have to agree about it - while the
#: first pass kept such a code and the residual validator still called it an
#: identifier, the document was anonymized correctly and then never released.
#:
#: The exemption is deliberately narrow, and both passes apply it the same
#: way: an explicit field immediately in front, not the word "diagnosis"
#: somewhere in the sentence, and a whole token, never the ICD-looking head
#: of a longer string. An identifier announced by MRN, account or patient
#: keeps its own head and is unaffected.
DIAGNOSIS_FIELD_HEADS: Tuple[str, ...] = (
    "diagnosis",
    "diagnoses",
    "dx",
    "icd",
    "icd10",
    "icd-10",
    "icd10cm",
    "icd-10-cm",
)

#: An ICD-10-CM code: a letter, a digit, one further alphanumeric, and an
#: optional fraction of up to four. Narrower than the search tokenizer's
#: "code-shaped" syntax, which a patient or account identifier also matches.
DIAGNOSIS_CODE_PATTERN = r"[A-Za-z][0-9][A-Za-z0-9](?:\.[A-Za-z0-9]{1,4})?"

#: Characters that may stand between a diagnosis field head and its code.
#: The full-width colon and the dash family are here because scanned text and
#: OCR output use them where a typist would use a colon or a hyphen; a span
#: that carries any other punctuation between head and value is not a plain
#: diagnosis field and gets no exemption.
DIAGNOSIS_FIELD_SEPARATORS = ":#=：﹕∶-‐‑‒–—−"

#: Words that may sit between the head and the separator, as "primary" does
#: in "Primary diagnosis:". Deliberately short: an unknown word between the
#: two means the span was not simply a diagnosis field.
DIAGNOSIS_FIELD_QUALIFIERS: Tuple[str, ...] = (
    "primary",
    "secondary",
    "principal",
    "admitting",
    "discharge",
    "working",
    "final",
    "provisional",
    "clinical",
    "coded",
    "confirmed",
)

#: Vital-sign labels. A number after one is a reading, not an identifier.
VITAL_SIGN_LABELS: FrozenSet[str] = frozenset(
    {
        "bp",
        "hr",
        "rr",
        "spo2",
        "sao2",
        "sats",
        "temp",
        "gcs",
        "map",
        "cvp",
        "etco2",
        "pulse",
        "o2",
    }
)

#: Units that turn a number into a measurement.
MEASUREMENT_UNITS: Tuple[str, ...] = (
    "%",
    "mmhg",
    "kpa",
    "bpm",
    "mg/dl",
    "mg/l",
    "mmol/l",
    "nmol/l",
    "umol/l",
    "miu/l",
    "iu/l",
    "mmol",
    "nmol",
    "umol",
    "mcg",
    "ug",
    "mg",
    "kg",
    "g",
    "ml",
    "l",
    "iu",
    "units",
    "unit",
    "c",
    "f",
    "cm",
    "mm",
)

#: Words that put the number after them inside a range. "From" and "to" are
#: deliberately absent: "From:" opens a header and "results to" precedes a
#: number that is being contacted.
RANGE_WORDS: FrozenSet[str] = frozenset(
    {
        "range",
        "between",
        "ref",
        "reference",
        "normal",
        "approx",
        "approximately",
        "about",
    }
)

#: Categories whose one confirmed value may be removed wherever it recurs.
#: Names, organisations, dates, places, role nouns and short tokens are
#: deliberately absent: a surname or a date recurs for reasons that have
#: nothing to do with the same identifier.
REPEATABLE_CATEGORIES: FrozenSet[str] = frozenset(
    {
        "PHONE_NUMBER",
        "ACCOUNT_NUMBER",
        "DEVICE_ID",
        "US_SSN",
        "SSN",
        "URL",
        "IP_ADDRESS",
        "MEDICAL_RECORD_NUMBER",
        "HEALTH_PLAN_ID",
        "INSURANCE_ID",
        "OTHER_UNIQUE_ID",
    }
)

#: Fewest digits a value needs before its repeats are searched for. Stricter
#: than detection, so an ambiguous short value is never propagated.
REPEAT_MIN_DIGITS: Mapping[str, int] = {
    "PHONE_NUMBER": 7,
    "ACCOUNT_NUMBER": 5,
    "DEVICE_ID": 4,
    "US_SSN": 9,
    "SSN": 9,
    "URL": 0,
    "IP_ADDRESS": 4,
    "MEDICAL_RECORD_NUMBER": 5,
    "HEALTH_PLAN_ID": 5,
    "INSURANCE_ID": 5,
    "OTHER_UNIQUE_ID": 4,
}

#: Fewest characters a propagated value may have.
REPEAT_MIN_LENGTH = 6
