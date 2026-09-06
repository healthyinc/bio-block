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

from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    """Every generated region in one anonymized document."""

    regions: Tuple[ReplacementRegion, ...] = ()

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

    def record(self, start: int, length: int, kind: str, entity_type: str) -> None:
        if length <= 0:
            return
        self._regions.append(
            ReplacementRegion(
                start=start, end=start + length, kind=kind, entity_type=entity_type
            )
        )

    def build(self) -> TransformationProvenance:
        return TransformationProvenance(
            regions=tuple(sorted(self._regions, key=lambda r: (r.start, r.end)))
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
