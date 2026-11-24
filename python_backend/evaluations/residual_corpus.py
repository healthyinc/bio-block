"""Synthetic corpus for testing the residual validator, not the first pass.

The second pass exists to catch PHI that survived transformation or reappeared
during serialization, reconstruction or metadata handling. Measuring it with
ordinary notes tests the wrong thing: those are cleaned by the first pass, so
the validator mostly sees its own surrogates.

So each document here is built to look like finished output - surrogates and
placeholders already in place - and then a synthetic identifier is injected
*after* that, simulating a value that escaped. The injected span is recorded
by offset and category, never by value, which is what makes recall
measurable per category.

Two partitions, disjoint by construction:

* ``calibration`` - for choosing anything that needs choosing;
* ``validation``  - opened once, never used for tuning.

A third corpus of safe clinical text carries no identifiers at all, and
exists to measure how often a candidate sends a clean document to manual
review.

Every value is invented: nonsense name stems, the reserved .invalid domain,
the 555-01xx fiction range, documentation IP blocks, and identifier formats
that are shaped like the real thing without being anyone's.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

CALIBRATION = "calibration"
VALIDATION = "validation"
SAFE_CLINICAL = "safe_clinical"
PARTITIONS = (CALIBRATION, VALIDATION, SAFE_CLINICAL)

DOCUMENTS_PER_PARTITION = 120

#: Safe Harbor identifier families this corpus can represent in text. Full-face
#: photographs are category 17 and have no textual form, so they are named here
#: as explicitly out of scope rather than silently missing.
CATEGORIES: Tuple[str, ...] = (
    "NAME",
    "GEOGRAPHY",
    "DATE",
    "AGE_OVER_89",
    "PHONE_NUMBER",
    "FAX_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "MEDICAL_RECORD_NUMBER",
    "HEALTH_PLAN_ID",
    "ACCOUNT_NUMBER",
    "LICENSE_NUMBER",
    "VEHICLE_ID",
    "DEVICE_ID",
    "URL",
    "IP_ADDRESS",
    "BIOMETRIC_ID",
    "OTHER_UNIQUE_ID",
)

UNREPRESENTABLE_CATEGORIES = ("FULL_FACE_PHOTOGRAPH",)

_STEMS = (
    "Zelquin",
    "Marbrint",
    "Voskaley",
    "Trennadi",
    "Quillorst",
    "Fendrose",
    "Oplavine",
    "Yarrimund",
    "Kestreloe",
    "Wandrily",
)
_SURNAMES = (
    "Vantreske",
    "Morrowgale",
    "Pellinquist",
    "Danforlay",
    "Ashgrovine",
    "Bellinshaw",
    "Corrivault",
    "Endermoss",
    "Fallowbrae",
    "Grimwarden",
)
_PLACES = ("Duskwater", "Thornmere", "Ravensholt", "BrackenfelI", "Wyndemoor")

#: Text that already looks transformed, so the validator sees realistic output.
_TRANSFORMED_LINES = (
    "Patient PATIENT_001 was reviewed by Dr. PROVIDER_001 at FACILITY_001.",
    "Metformin 500 mg twice daily was continued without adjustment.",
    "Blood pressure remained stable; HbA1c measured 7.2 percent.",
    "Contact details were removed: <REDACTED_EMAIL> and <REDACTED_PHONE>.",
    "The encounter date is recorded as <REDACTED_DATE>.",
    "Record RECORD_001 was updated after the multidisciplinary review.",
    "No adverse reaction was recorded by nursing staff overnight.",
    "Parkinson's disease was considered and thought unlikely.",
)

_SAFE_CLINICAL_LINES = (
    "Respiratory examination was unremarkable on repeat auscultation.",
    "Creatinine 0.9 mg/dL with an estimated GFR above 90.",
    "The scan showed a small lesion in the left temporal lobe.",
    "Wilson disease was excluded by serum copper studies.",
    "Atorvastatin 40 mg nightly is tolerated with no reported myalgia.",
    "Bell palsy resolved fully without antiviral treatment.",
    "The multidisciplinary team agreed to continue current management.",
    "Blood pressure remained stable throughout the observation period.",
)


def _value(category: str, rng: random.Random) -> str:
    """One synthetic identifier of the requested family."""
    if category == "NAME":
        return f"{rng.choice(_STEMS)} {rng.choice(_SURNAMES)}"
    if category == "GEOGRAPHY":
        return f"{rng.randint(1, 99)} {rng.choice(_PLACES)} Lane, {rng.choice(_PLACES)}"
    if category == "DATE":
        return f"{rng.randint(1, 28)} March {rng.randint(2015, 2022)}"
    if category == "AGE_OVER_89":
        return f"{rng.randint(90, 104)} years old"
    if category == "PHONE_NUMBER":
        return f"555-01{rng.randint(10, 99)}"
    if category == "FAX_NUMBER":
        return f"fax 555-01{rng.randint(10, 99)}"
    if category == "EMAIL_ADDRESS":
        return f"{rng.choice(_STEMS).lower()}.{rng.choice(_SURNAMES).lower()}@example.invalid"
    if category == "US_SSN":
        return f"9{rng.randint(10, 89)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"
    if category == "MEDICAL_RECORD_NUMBER":
        return f"MRN {rng.randint(10_000_000, 99_999_999)}"
    if category == "HEALTH_PLAN_ID":
        return f"SYN-PLAN-{rng.randint(1_000_000, 9_999_999)}"
    if category == "ACCOUNT_NUMBER":
        return f"account {rng.randint(100_000_000, 999_999_999)}"
    if category == "LICENSE_NUMBER":
        return f"licence SYN-LIC-{rng.randint(100_000, 999_999)}"
    if category == "VEHICLE_ID":
        return f"vehicle SYNVIN{rng.randint(10_000_000, 99_999_999)}"
    if category == "DEVICE_ID":
        return f"device SYN-DEV-{rng.randint(100_000, 999_999)}"
    if category == "URL":
        return f"https://records.example.invalid/{rng.choice(_STEMS).lower()}"
    if category == "IP_ADDRESS":
        return f"203.0.113.{rng.randint(2, 250)}"
    if category == "BIOMETRIC_ID":
        return f"fingerprint template FP-{rng.randint(100_000, 999_999)}"
    if category == "OTHER_UNIQUE_ID":
        return f"study code SYN-{rng.randint(100_000, 999_999)}"
    raise ValueError(f"unknown category: {category}")


#: Placements that have historically been where a validator loses a span.
SHAPES: Tuple[str, ...] = (
    "start_of_text",
    "end_of_text",
    "after_line_break",
    "after_crlf",
    "inside_punctuation",
    "possessive",
    "table_row",
    "ocr_spacing",
    "unicode_context",
    "repeated",
    "adjacent_to_surrogate",
    "long_line",
)


def _place(
    shape: str, value: str, body: List[str], rng: random.Random
) -> Tuple[str, int, int]:
    """Return (text, start, end) with the value placed per the shape."""
    filler = "\n".join(body)
    if shape == "start_of_text":
        text = f"{value} was recorded before review.\n{filler}"
        return text, 0, len(value)
    if shape == "end_of_text":
        prefix = f"{filler}\nThe remaining entry reads "
        return prefix + value, len(prefix), len(prefix) + len(value)
    if shape == "after_line_break":
        prefix = f"{filler}\n"
        return (
            prefix + value + "\nEnd of record.",
            len(prefix),
            len(prefix) + len(value),
        )
    if shape == "after_crlf":
        prefix = f"{filler}\r\n"
        return (
            prefix + value + "\r\nEnd of record.",
            len(prefix),
            len(prefix) + len(value),
        )
    if shape == "inside_punctuation":
        prefix = f"{filler}\nEntry ("
        return prefix + value + ") was appended.", len(prefix), len(prefix) + len(value)
    if shape == "possessive":
        prefix = f"{filler}\nThe record belonging to "
        return (
            prefix + value + "'s file was reopened.",
            len(prefix),
            len(prefix) + len(value),
        )
    if shape == "table_row":
        prefix = f"{filler}\nfield\tvalue\nresidual\t"
        return (
            prefix + value + "\nstatus\treviewed",
            len(prefix),
            len(prefix) + len(value),
        )
    if shape == "ocr_spacing":
        prefix = f"{filler}\nScanned  entry :   "
        return prefix + value + "  .", len(prefix), len(prefix) + len(value)
    if shape == "unicode_context":
        prefix = f"{filler}\nNotes café / naïve / 日本語: "
        return prefix + value + " recorded.", len(prefix), len(prefix) + len(value)
    if shape == "repeated":
        prefix = f"{filler}\nSeen as {value}, and again as "
        return (
            prefix + value + " in the appendix.",
            len(prefix),
            len(prefix) + len(value),
        )
    if shape == "adjacent_to_surrogate":
        prefix = f"{filler}\nPATIENT_001 "
        return prefix + value + " FACILITY_001", len(prefix), len(prefix) + len(value)
    if shape == "long_line":
        prefix = f"{filler}\n" + (
            "continuous observation text without punctuation " * 40
        )
        return prefix + value + " end", len(prefix), len(prefix) + len(value)
    raise ValueError(f"unknown shape: {shape}")


@dataclass(frozen=True)
class ResidualSpan:
    """One injected identifier. Offsets and category only."""

    start: int
    end: int
    category: str
    shape: str


@dataclass(frozen=True)
class ResidualDocument:
    doc_id: str
    text: str
    spans: Tuple[ResidualSpan, ...] = ()

    @property
    def has_residual(self) -> bool:
        return bool(self.spans)

    def value(self, span: ResidualSpan) -> str:
        """The injected text. For assertions only; never reported."""
        return self.text[span.start : span.end]


def _seed_for(partition: str, index: int) -> random.Random:
    # Derived only from partition and index, so the corpus is identical
    # everywhere and the two partitions never share a value pool.
    return random.Random(f"bioblock-residual-v1:{partition}:{index}")


def build_document(partition: str, index: int) -> ResidualDocument:
    rng = _seed_for(partition, index)

    if partition == SAFE_CLINICAL:
        lines = [rng.choice(_SAFE_CLINICAL_LINES) for _ in range(rng.randint(4, 10))]
        lines.insert(0, rng.choice(_TRANSFORMED_LINES))
        return ResidualDocument(f"{partition}-{index:04d}", "\n".join(lines))

    body = [rng.choice(_TRANSFORMED_LINES) for _ in range(rng.randint(3, 7))]
    # Both indexed directly on the document number. 18 categories and 12
    # shapes share a factor, so the pair repeats every 36 documents and each
    # partition covers every category and every shape. Deriving the shape from
    # index // len(CATEGORIES) instead left the last five shapes - including
    # the Unicode, OCR-spacing and long-line cases - never generated at all.
    category = CATEGORIES[index % len(CATEGORIES)]
    shape = SHAPES[index % len(SHAPES)]
    value = _value(category, rng)
    text, start, end = _place(shape, value, body, rng)

    # A second identifier in some documents, so multi-category coverage is
    # tested rather than assumed.
    spans = [ResidualSpan(start, end, category, shape)]
    if index % 4 == 0:
        extra_category = CATEGORIES[(index + 7) % len(CATEGORIES)]
        extra_value = _value(extra_category, rng)
        suffix = f"\nAlso present: {extra_value}."
        extra_start = len(text) + len("\nAlso present: ")
        text = text + suffix
        spans.append(
            ResidualSpan(
                extra_start, extra_start + len(extra_value), extra_category, "appended"
            )
        )

    return ResidualDocument(f"{partition}-{index:04d}", text, tuple(spans))


def partition_documents(
    partition: str, count: int = DOCUMENTS_PER_PARTITION
) -> List[ResidualDocument]:
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition: {partition}")
    return [build_document(partition, index) for index in range(count)]


def partition_digest(partition: str, count: int = DOCUMENTS_PER_PARTITION) -> str:
    """Content hash, so a cache entry can be tied to the exact corpus."""
    digest = hashlib.sha256()
    for document in partition_documents(partition, count):
        digest.update(document.doc_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.text.encode("utf-8"))
        digest.update(b"\0")
        for span in document.spans:
            digest.update(f"{span.start}:{span.end}:{span.category}\0".encode())
    return digest.hexdigest()


def category_coverage(
    partition: str, count: int = DOCUMENTS_PER_PARTITION
) -> Dict[str, int]:
    counts: Dict[str, int] = {category: 0 for category in CATEGORIES}
    for document in partition_documents(partition, count):
        for span in document.spans:
            counts[span.category] = counts.get(span.category, 0) + 1
    return counts
