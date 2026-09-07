"""Span matching and metrics for the labelled PHI corpus.

Scoring is span-level, and two different questions are answered separately:

* **span recall** - was the gold span overlapped by *any* detection, so the
  value would in fact be redacted? This is the privacy-critical figure, and it
  is the one false negatives are counted against.
* **typed recall** - was it overlapped by a detection whose category is an
  accepted alias of the gold category? A value that is removed but labelled
  differently is a reporting problem, not a leak, so the two are never merged.

False positives are counted against the negative controls the corpus records
(eponymous conditions, modality abbreviations, ordinary numbers) plus any
detection that overlaps no gold span at all. Over-redaction is a usability
cost, not a privacy failure, so it is measured but never allowed to outrank a
false negative when a threshold is chosen.

Nothing in this module returns a matched value. Reports carry offsets,
categories, counts and rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from evaluations.labelled_corpus import CATEGORY_ALIASES, LabelledDocument

#: False negatives are weighted this much more heavily than false positives in
#: the composite score used to rank candidate thresholds. A missed identifier
#: is a disclosure; an over-redacted word is an inconvenience.
FN_WEIGHT = 10.0


@dataclass(frozen=True)
class Detection:
    """One proposed span, normalized away from any detector-specific shape."""

    start: int
    end: int
    category: str
    source: str
    score: float | None = None


@dataclass
class DocumentScore:
    doc_id: str
    gold_total: int = 0
    span_hits: int = 0
    typed_hits: int = 0
    missed: List[Tuple[str, Tuple[str, ...]]] = field(default_factory=list)
    false_positives: int = 0
    negatives_total: int = 0
    negatives_redacted: int = 0
    detections: int = 0

    @property
    def leaked(self) -> bool:
        """True if any gold span in this document went unredacted."""
        return self.span_hits < self.gold_total


def _accepted(gold_category: str) -> Tuple[str, ...]:
    return CATEGORY_ALIASES.get(gold_category, (gold_category,))


def score_document(
    document: LabelledDocument,
    detections: Sequence[Detection],
) -> DocumentScore:
    """Score one document's detections against its gold spans."""
    result = DocumentScore(doc_id=document.doc_id)
    result.gold_total = len(document.spans)
    result.detections = len(detections)
    result.negatives_total = len(document.negatives)

    covered_by_any: List[bool] = []
    for span in document.spans:
        overlapping = [d for d in detections if span.overlaps(d.start, d.end)]
        covered = bool(overlapping)
        covered_by_any.append(covered)
        if covered:
            result.span_hits += 1
            accepted = _accepted(span.category)
            if any(d.category in accepted for d in overlapping):
                result.typed_hits += 1
        else:
            result.missed.append((span.category, span.tags))

    # A detection that overlaps no gold span is a false positive.
    for detection in detections:
        if not any(
            span.overlaps(detection.start, detection.end) for span in document.spans
        ):
            result.false_positives += 1

    # Negative controls that were nonetheless covered by a detection.
    for start, end, _label in document.negatives:
        if any(d.start < end and start < d.end for d in detections):
            result.negatives_redacted += 1

    return result


@dataclass
class CategoryScore:
    gold: int = 0
    span_hits: int = 0
    typed_hits: int = 0
    missed_tags: Dict[str, int] = field(default_factory=dict)

    @property
    def span_recall(self) -> float:
        return self.span_hits / self.gold if self.gold else 0.0

    @property
    def typed_recall(self) -> float:
        return self.typed_hits / self.gold if self.gold else 0.0


def aggregate(
    documents: Sequence[LabelledDocument],
    detections_by_doc: Mapping[str, Sequence[Detection]],
) -> Dict[str, Any]:
    """Aggregate per-document scores into the full report body."""
    per_doc: List[DocumentScore] = []
    by_category: Dict[str, CategoryScore] = {}

    for document in documents:
        detections = detections_by_doc.get(document.doc_id, ())
        score = score_document(document, detections)
        per_doc.append(score)

        for span in document.spans:
            bucket = by_category.setdefault(span.category, CategoryScore())
            bucket.gold += 1
            overlapping = [d for d in detections if span.overlaps(d.start, d.end)]
            if overlapping:
                bucket.span_hits += 1
                if any(d.category in _accepted(span.category) for d in overlapping):
                    bucket.typed_hits += 1
            else:
                for tag in span.tags or ("untagged",):
                    bucket.missed_tags[tag] = bucket.missed_tags.get(tag, 0) + 1

    gold_total = sum(s.gold_total for s in per_doc)
    span_hits = sum(s.span_hits for s in per_doc)
    typed_hits = sum(s.typed_hits for s in per_doc)
    false_positives = sum(s.false_positives for s in per_doc)
    detections_total = sum(s.detections for s in per_doc)
    negatives_total = sum(s.negatives_total for s in per_doc)
    negatives_redacted = sum(s.negatives_redacted for s in per_doc)
    leaking_docs = [s.doc_id for s in per_doc if s.leaked]

    false_negatives = gold_total - span_hits
    true_positives = span_hits
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    recall = true_positives / gold_total if gold_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "documents": len(per_doc),
        "gold_spans": gold_total,
        "detections": detections_total,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "precision": round(precision, 4),
        "span_recall": round(recall, 4),
        "typed_recall": round(typed_hits / gold_total, 4) if gold_total else 0.0,
        "f1": round(f1, 4),
        # A document leaks if any single gold span in it survived.
        "document_leakage_rate": round(len(leaking_docs) / len(per_doc), 4)
        if per_doc
        else 0.0,
        "leaking_documents": sorted(leaking_docs),
        # Useful-text preservation: the share of recorded non-PHI terms that
        # were NOT swept up by a detection.
        "negative_terms": negatives_total,
        "negative_terms_redacted": negatives_redacted,
        "useful_text_preservation": round(
            1 - (negatives_redacted / negatives_total), 4
        )
        if negatives_total
        else 1.0,
        "composite_cost": round(FN_WEIGHT * false_negatives + false_positives, 2),
        "by_category": {
            name: {
                "gold": bucket.gold,
                "span_recall": round(bucket.span_recall, 4),
                "typed_recall": round(bucket.typed_recall, 4),
                "missed": bucket.gold - bucket.span_hits,
                "missed_tags": dict(sorted(bucket.missed_tags.items())),
            }
            for name, bucket in sorted(by_category.items())
        },
    }


def missed_categories(report: Mapping[str, Any]) -> List[str]:
    """Categories with zero span recall: nothing detected them at all."""
    return sorted(
        name
        for name, stats in report.get("by_category", {}).items()
        if stats["gold"] and stats["span_recall"] == 0.0
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile. Avoids a numpy dependency in the report path."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered) + 0.5) - 1))
    return round(ordered[index], 2)
