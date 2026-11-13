"""Deterministic synthetic documents for latency benchmarking.

Generated from a fixed seed so a run is reproducible and nothing has to be
committed. Every identifier is unmistakably invented: names built from
nonsense stems, the reserved .invalid domain, the 555-01xx fiction telephone
range, TEST-NET documentation addresses, and social-security numbers in the
900 block that was never issued. Nothing here resembles a real person, and no
real medical text is used.

These exist to exercise timing, not detection quality. The corpus that
measures privacy lives in evaluations/ and is unrelated.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

SEED = 12_012_024

#: The text route rejects uploads above this, so the largest bucket stops
#: below it. A 1 MB bucket is not measurable against the current limit.
TEXT_LIMIT_BYTES = 262_144

SIZE_TARGETS: Tuple[Tuple[str, int], ...] = (
    ("tiny", 1_024),
    ("small", 10_240),
    ("medium", 102_400),
    ("large", 200_000),
)

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

#: Clinical filler with no identifiers. Real vocabulary, invented sentences.
_CLINICAL_LINES = (
    "Blood pressure remained stable throughout the observation period.",
    "Metformin 500 mg twice daily was continued without adjustment.",
    "Atorvastatin 40 mg nightly is tolerated with no reported myalgia.",
    "HbA1c measured 7.2 percent, improved from the prior review.",
    "The scan showed a small lesion in the left temporal lobe.",
    "Parkinson's disease was considered and thought unlikely.",
    "Wilson disease was excluded by serum copper studies.",
    "Bell palsy resolved fully without antiviral treatment.",
    "Creatinine 0.9 mg/dL with an estimated GFR above 90.",
    "No adverse reaction was recorded by nursing staff overnight.",
    "Respiratory examination was unremarkable on repeat auscultation.",
    "The multidisciplinary team reviewed the plan and agreed to continue.",
)


def _person(rng: random.Random) -> str:
    return f"{rng.choice(_STEMS)} {rng.choice(_SURNAMES)}"


def _structured_block(rng: random.Random) -> List[str]:
    """Obvious structured identifiers, every one synthetic."""
    return [
        f"Patient {_person(rng)} was admitted for review.",
        f"MRN {rng.randint(10_000_000, 99_999_999)} recorded at intake.",
        f"Contact {rng.choice(_STEMS).lower()}@example.invalid for records.",
        f"Telephone 555-01{rng.randint(10, 99)} listed as primary.",
        f"Social security number 9{rng.randint(10, 89)}-{rng.randint(10, 99)}"
        f"-{rng.randint(1000, 9999)} on file.",
        f"Terminal address 203.0.113.{rng.randint(2, 250)} used for upload.",
    ]


def _date_and_age_block(rng: random.Random) -> List[str]:
    """Dates and an age above the Safe Harbor threshold."""
    return [
        f"Reviewed on {rng.randint(1, 28)} March {rng.randint(2015, 2022)}.",
        f"The patient is {rng.randint(90, 103)} years old and independent.",
        f"Follow-up booked for {rng.randint(1, 28)}/0{rng.randint(1, 9)}/2021.",
    ]


def _repeated_identifier_block(rng: random.Random) -> List[str]:
    """The same person several times, to exercise surrogate reuse."""
    who = _person(rng)
    return [
        f"{who} attended the morning clinic.",
        f"{who} reported no new symptoms.",
        f"Discharge instructions were given to {who}.",
    ]


def _ambiguous_block(rng: random.Random) -> List[str]:
    """Shapes that tend to end in manual review."""
    return [
        f"Samples were sent to {rng.choice(_SURNAMES)} Diagnostics Ltd.",
        f"Referred by Dr {rng.choice(_SURNAMES)} to the regional service.",
        f"{rng.choice(_SURNAMES)} syndrome was considered unlikely.",
    ]


CONTENT_KINDS: Tuple[str, ...] = (
    "no_phi",
    "structured_phi",
    "mixed_clinical",
    "repeated_identifiers",
    "eponyms_and_medications",
    "dates_and_ages",
    "expected_pass",
    "expected_manual_review",
)


def _seed_lines(kind: str, rng: random.Random) -> List[str]:
    if kind == "no_phi":
        return list(_CLINICAL_LINES)
    if kind == "structured_phi":
        return _structured_block(rng)
    if kind == "mixed_clinical":
        return [*_structured_block(rng), *_CLINICAL_LINES[:6]]
    if kind == "repeated_identifiers":
        return _repeated_identifier_block(rng)
    if kind == "eponyms_and_medications":
        return [
            l
            for l in _CLINICAL_LINES
            if any(
                t in l
                for t in ("Parkinson", "Wilson", "Bell", "Metformin", "Atorvastatin")
            )
        ]
    if kind == "dates_and_ages":
        return _date_and_age_block(rng)
    if kind == "expected_pass":
        return list(_CLINICAL_LINES)
    if kind == "expected_manual_review":
        return _ambiguous_block(rng)
    raise ValueError(f"unknown content kind: {kind}")


def build_document(kind: str, target_bytes: int, seed: int = SEED) -> bytes:
    """A synthetic document of roughly ``target_bytes``, reproducible per seed."""
    if kind not in CONTENT_KINDS:
        raise ValueError(f"unknown content kind: {kind}")
    if target_bytes > TEXT_LIMIT_BYTES:
        raise ValueError(
            f"target {target_bytes} exceeds the {TEXT_LIMIT_BYTES} byte text limit"
        )

    rng = random.Random(f"{seed}:{kind}:{target_bytes}")
    seed_lines = _seed_lines(kind, rng)

    lines: List[str] = ["Synthetic benchmark note. All values are invented."]
    length = len(lines[0]) + 1
    index = 0
    while length < target_bytes:
        line = seed_lines[index % len(seed_lines)]
        index += 1
        # Vary the filler so the text is not one repeated string, which would
        # let caching flatter the measurement.
        if index % 5 == 0:
            line = f"{line} Entry {index}."
        lines.append(line)
        length += len(line) + 1

    return ("\n".join(lines)).encode("utf-8")


def size_targets(include_large: bool = True) -> Tuple[Tuple[str, int], ...]:
    if include_large:
        return SIZE_TARGETS
    return tuple(t for t in SIZE_TARGETS if t[0] != "large")


def build_suite(seed: int = SEED) -> Dict[str, bytes]:
    """One document per size bucket, mixed content, for the size sweep."""
    return {
        name: build_document("mixed_clinical", target, seed)
        for name, target in SIZE_TARGETS
    }
