"""Where the sanitizer wrote into the output, in final-output coordinates.

The Phase 10 residual validator worked by blanking every generated token with
spaces and re-scanning the result. That destroyed the sentence it was meant to
check: "Dr. PROVIDER_001 at FACILITY_001" became "Dr.<12 spaces>at<13 spaces>",
and both models reliably predicted that a name belonged in the hole. Sixty per
cent of clean documents were blocked by findings the masking itself created.

The fix is to stop distorting the text. The validator now scans the exact
serialized output, and this module records precisely which character ranges of
that output the sanitizer wrote, so a second-pass finding can be attributed:

* wholly inside a generated region  -> the detector is reading our own token;
* anywhere else                     -> real surviving text, and it counts.

The rule is deliberately strict about "wholly". A prediction that merely
touches a surrogate still covers text we did not generate, and that
surrounding text is exactly where a missed identifier would sit. Partial
overlap is never a reason to ignore a finding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import AbstractSet, List, Optional, Tuple

#: What the sanitizer wrote at a given range of the output.
KIND_SURROGATE = "surrogate"
KIND_PLACEHOLDER = "placeholder"
KIND_GENERALIZED = "generalized"
KIND_SHIFTED_DATE = "shifted_date"

GENERATED_KINDS = frozenset(
    {KIND_SURROGATE, KIND_PLACEHOLDER, KIND_GENERALIZED, KIND_SHIFTED_DATE}
)


@dataclass(frozen=True)
class ReplacementRegion:
    """One span the sanitizer wrote, in **final output** coordinates."""

    start: int
    end: int
    kind: str
    #: The internal category that produced it. Never the original value.
    entity_type: str

    def contains(self, start: int, end: int) -> bool:
        """True only when [start, end) lies wholly inside this region."""
        return self.start <= start and end <= self.end

    def overlaps(self, start: int, end: int) -> bool:
        return start < self.end and self.start < end


@dataclass(frozen=True)
class TransformationProvenance:
    """Every generated region in one anonymized document.

    ``emitted`` is the exact set of replacement tokens this document's
    transformation wrote. Offsets do not survive a file being written and read
    back, so a reopened document recovers its regions by matching token
    shapes - and a shape is not evidence. A document that arrives already
    containing ``PERSON_DEADBEEF`` would be handed a region saying the
    sanitizer wrote it, and the residual scan would skip exactly the span an
    attacker chose. Matching against what was actually emitted is what turns
    a shape back into evidence.

    The tokens are the sanitizer's own output and hold no original value, so
    keeping them costs nothing. They never leave the process: no public
    response carries this.
    """

    regions: Tuple[ReplacementRegion, ...] = ()
    emitted: frozenset = frozenset()

    def covering(self, start: int, end: int) -> Optional[ReplacementRegion]:
        """The region wholly containing this span, if any."""
        for region in self.regions:
            if region.contains(start, end):
                return region
        return None

    def touching(self, start: int, end: int) -> List[ReplacementRegion]:
        return [r for r in self.regions if r.overlaps(start, end)]

    def is_generated(self, start: int, end: int) -> bool:
        """Whether a span is entirely sanitizer-generated.

        Partial overlap returns False on purpose: the uncovered part is text
        we did not write, and a missed identifier adjacent to a surrogate is
        precisely the case that must not be waved through.
        """
        return self.covering(start, end) is not None

    def location_type(self, start: int, end: int) -> str:
        """Classify where a span sits relative to generated regions."""
        if self.is_generated(start, end):
            return "inside_generated_region"
        touching = self.touching(start, end)
        if touching:
            return "spans_generated_and_original"
        return "original_text"

    def uncovered(self, start: int, end: int) -> List[Tuple[int, int]]:
        """The parts of [start, end) that no generated region covers.

        This is the text a straddling prediction is actually making a claim
        about, over and above the token we wrote. Returning it lets the caller
        examine that remainder rather than either ignoring the finding or
        blocking on it wholesale.
        """
        gaps: List[Tuple[int, int]] = []
        cursor = start
        for region in sorted(self.regions, key=lambda r: r.start):
            if region.end <= cursor or region.start >= end:
                continue
            if region.start > cursor:
                gaps.append((cursor, min(region.start, end)))
            cursor = max(cursor, region.end)
            if cursor >= end:
                break
        if cursor < end:
            gaps.append((cursor, end))
        return gaps

    def counts(self) -> dict:
        summary: dict = {}
        for region in self.regions:
            summary[region.kind] = summary.get(region.kind, 0) + 1
        return dict(sorted(summary.items()))

    def total_generated_characters(self) -> int:
        return sum(region.end - region.start for region in self.regions)


class ProvenanceBuilder:
    """Accumulates regions while replacements are applied back-to-front.

    Replacements run from the end of the document so earlier offsets stay
    valid, which means a region's final coordinates are already correct when
    it is written: nothing before it has moved yet.
    """

    def __init__(self) -> None:
        self._regions: List[ReplacementRegion] = []
        self._emitted: set = set()

    def record(
        self,
        start: int,
        length: int,
        kind: str,
        entity_type: str,
        token: Optional[str] = None,
    ) -> None:
        """Record one written span, and the token written there.

        ``token`` is what the sanitizer put in the output. It is kept so that
        a later pass over a reopened file can tell our own writing from text
        that merely looks like it.
        """
        if length <= 0:
            return
        self._regions.append(
            ReplacementRegion(
                start=start, end=start + length, kind=kind, entity_type=entity_type
            )
        )
        if token:
            self._emitted.add(token)

    def build(self) -> TransformationProvenance:
        return TransformationProvenance(
            regions=tuple(sorted(self._regions, key=lambda r: (r.start, r.end))),
            emitted=frozenset(self._emitted),
        )


def kind_for_replacement(entity_type: str, replacement: str) -> str:
    """Classify what kind of generated token a replacement is."""
    if entity_type == "AGE_OVER_89":
        return KIND_GENERALIZED
    if replacement.startswith("<REDACTED_"):
        return KIND_PLACEHOLDER
    if entity_type in {"DATE", "DATE_TIME"} and not replacement.startswith("<"):
        return KIND_SHIFTED_DATE
    return KIND_SURROGATE


# Written-text provenance is shared policy; it does not depend on a format handler.
from services.surrogates import HASH_SURROGATE_PATTERN as _HASH_SURROGATE
from services.surrogates import LEGACY_SURROGATE_PATTERN, SURROGATE_PATTERN

_PLACEHOLDER = re.compile(r"<REDACTED_[A-Z0-9_]+>")

_AGE_AGGREGATE = re.compile(r"(?<!\d)90\+")


def generated_token_provenance(text: str, emitted: AbstractSet[str]):
    """Where the anonymizer's own tokens sit in text re-extracted from a file.

    The offsets recorded during the transformation do not survive a file being
    written and read back, so the spans are found again by matching the token
    shapes the sanitizer emits.

    A shape on its own is not evidence, and trusting one was a real hole. A
    document that arrives already containing ``PERSON_DEADBEEF`` or
    ``<REDACTED_PERSON>`` was handed a region saying the sanitizer wrote it,
    and the residual pass then skipped precisely the span whoever wrote the
    document had chosen. Passing ``emitted`` - the tokens this transformation
    actually wrote - closes that: a shape is discounted only when it is one of
    them, and anything else is inspected like any other text.

    The ledger is mandatory. Passing an empty set fails closed: no token shape
    is trusted merely because it resembles sanitizer output.
    """
    builder = ProvenanceBuilder()
    taken: List[Tuple[int, int]] = []
    for pattern, kind, category in (
        (_PLACEHOLDER, KIND_PLACEHOLDER, "PLACEHOLDER"),
        (SURROGATE_PATTERN, KIND_SURROGATE, "SURROGATE"),
        (LEGACY_SURROGATE_PATTERN, KIND_SURROGATE, "SURROGATE"),
        (_HASH_SURROGATE, KIND_SURROGATE, "SURROGATE"),
        (_AGE_AGGREGATE, KIND_GENERALIZED, "AGE_OVER_89"),
    ):
        for match in pattern.finditer(text):
            if match.group(0) not in emitted:
                # Shaped like our writing, but we did not write it.
                continue
            # A current token altered by prefix/suffix text or nested inside
            # another token is no longer the occurrence we emitted.
            before = text[match.start() - 1 : match.start()]
            after = text[match.end() : match.end() + 1]
            if before and (before.isalnum() or before in "_<"):
                continue
            if after and (after.isalnum() or after in "_>"):
                continue
            if any(match.start() < end and start < match.end() for start, end in taken):
                continue
            taken.append((match.start(), match.end()))
            builder.record(
                match.start(),
                match.end() - match.start(),
                kind,
                category,
                token=match.group(0),
            )
    return builder.build()
