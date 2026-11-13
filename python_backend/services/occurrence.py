"""One decision per occurrence: what was found where, and what was done with it.

Every candidate span the first pass considers ends as exactly one decision:
``replace`` (removed and given a token), ``retain`` (kept, with the clinical
or contextual evidence that kept it) or ``hold`` (the file goes to a person).
A decision is addressed by surface and offsets, never by the value it covers,
and carries no text: it can be counted, compared and logged.

Detection and replacement consistency are separate: a decision says *this
occurrence* is a contact; only then does the allocator give it the token its
classified identity already has.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

POLICY_VERSION = "occurrence-decisions-v1"

REPLACE = "replace"
RETAIN = "retain"
HOLD = "hold"


@dataclass(frozen=True)
class OccurrenceDecision:
    surface: str  # text, csv_cell, pdf_page, dicom_field, ...
    location: str  # page/frame/sheet/cell/field/window - no values
    start: int
    end: int
    entity_type: str
    detector: str
    deterministic: bool
    neural_score: Optional[float]
    clinical_evidence: str  # the structural reason that kept it, or ""
    context: Tuple[int, int]  # the sentence/line/cell bounds that were read
    conflict: str  # value-free description of opposing evidence
    action: str  # replace | retain | hold
    reason: str
    replacement_family: str
    provenance: str  # the pass and stage that decided it
    agreeing_detectors: int = 1

    def public_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class DecisionLog:
    """The decisions of one pass. Bounded; offsets and codes only."""

    limit: int = 20_000
    decisions: List[OccurrenceDecision] = field(default_factory=list)
    overflow: int = 0

    def add(self, decision: OccurrenceDecision) -> None:
        if len(self.decisions) < self.limit:
            self.decisions.append(decision)
        else:
            self.overflow += 1

    def summary(self) -> Dict[str, object]:
        by_action: Dict[str, int] = {}
        by_reason: Dict[str, int] = {}
        for decision in self.decisions:
            by_action[decision.action] = by_action.get(decision.action, 0) + 1
            key = f"{decision.action}:{decision.reason}"
            by_reason[key] = by_reason.get(key, 0) + 1
        return {
            "policy": POLICY_VERSION,
            "decisions": len(self.decisions),
            "overflow": self.overflow,
            "by_action": dict(sorted(by_action.items())),
            "by_reason": dict(sorted(by_reason.items())),
        }


def merge_summaries(total: Dict[str, object], part: Dict[str, object]) -> None:
    """Add one window's summary to a streamed note's running total.

    Windows overlap, so an occurrence in an overlap can be decided twice: the
    total counts decisions made, window by window, and says so.
    """
    if not part:
        return
    total["policy"] = part.get("policy", POLICY_VERSION)
    total["counted_per_window"] = True
    total["windows"] = int(total.get("windows", 0)) + 1
    for key in ("decisions", "overflow"):
        total[key] = int(total.get(key, 0)) + int(part.get(key, 0) or 0)
    for key in ("by_action", "by_reason"):
        merged = dict(total.get(key) or {})
        for name, count in (part.get(key) or {}).items():
            merged[name] = merged.get(name, 0) + int(count)
        total[key] = dict(sorted(merged.items()))


def spans(
    decisions: Iterable[OccurrenceDecision], action: str
) -> List[Tuple[int, int]]:
    return sorted((d.start, d.end) for d in decisions if d.action == action)
