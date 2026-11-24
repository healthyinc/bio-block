"""Shadow evaluation of residual-validator candidates.

The second pass costs about as much as the first, and GLiNER is roughly 78%
of detector time, so the obvious question is whether the second GLiNER run
earns its cost. This answers that offline. Nothing here is wired to
production: candidates are composed from recorded detector output and
compared, and none of them can authorize a release.

Each detector runs once per document and its findings are cached. A cache
entry is only reused when the model revisions, checksums, thresholds,
detector configuration and corpus digest all match, because a cached span
from a different configuration would silently answer a question nobody asked.

    <model-venv-python> evaluations/residual_shadow.py --partition calibration
    <model-venv-python> evaluations/residual_shadow.py --partition validation

Results go to evaluations/diagnostics/, which is git-ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

DIAGNOSTICS_DIR = BACKEND_ROOT / "evaluations" / "diagnostics"
CACHE_DIR = DIAGNOSTICS_DIR / "detector_cache"

CACHE_VERSION = 1

#: Detector keys used throughout. "structured" and "age" are deterministic
#: rules; the rest are models.
D_STRUCTURED = "structured"
D_AGE = "age"
D_SPACY = "spacy"
D_STANFORD = "stanford"
D_GLINER = "gliner"

ALL_DETECTORS = (D_STRUCTURED, D_AGE, D_SPACY, D_STANFORD, D_GLINER)

#: The candidates under evaluation. The control is what production runs now.
CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "control": (D_STRUCTURED, D_AGE, D_SPACY, D_STANFORD, D_GLINER),
    "A_no_gliner": (D_STRUCTURED, D_AGE, D_SPACY, D_STANFORD),
    "B_stanford_only": (D_STRUCTURED, D_AGE, D_STANFORD),
    "C_no_stanford": (D_STRUCTURED, D_AGE, D_SPACY, D_GLINER),
    # D is a cascade: rules and Stanford over everything, GLiNER only over
    # regions the cascade itself nominates as high risk. Composed below.
    "D_cascade": (D_STRUCTURED, D_AGE, D_SPACY, D_STANFORD),
}

#: Candidate D nominates a region as high risk when a cheap signal fires but
#: no model has claimed the span. Deliberately broad: the cascade is only
#: acceptable if it never misses, so it prefers nominating too much.
HIGH_RISK_WINDOW = 240


@dataclass(frozen=True)
class Finding:
    start: int
    end: int
    category: str
    detector: str
    score: Optional[float] = None

    def overlaps(self, start: int, end: int) -> bool:
        return self.start < end and start < self.end


# -- detector execution, cached --------------------------------------------


def _configuration_fingerprint() -> Dict[str, Any]:
    """Everything that would change a detector's answer."""
    from services.local_model_detectors import (
        SOURCE_GLINER,
        SOURCE_STANFORD,
        calibrated_config_for,
        load_model_manifest,
    )

    manifest = load_model_manifest()
    models = {
        name: {"revision": spec.revision, "weight_sha256": spec.weight_sha256}
        for name, spec in sorted(manifest.items())
    }
    configs = {}
    for key, source in ((D_STANFORD, SOURCE_STANFORD), (D_GLINER, SOURCE_GLINER)):
        config = calibrated_config_for(source)
        configs[key] = {
            "candidate_threshold": config.candidate_threshold,
            "redaction_threshold": config.redaction_threshold,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "batch_size": config.batch_size,
        }
    return {"cache_version": CACHE_VERSION, "models": models, "detectors": configs}


def _fingerprint_digest(fingerprint: Dict[str, Any], corpus_digest: str) -> str:
    payload = json.dumps(
        {"fingerprint": fingerprint, "corpus": corpus_digest},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _detectors_for_text():
    """Construct each detector once; they are reused for every document."""
    from services.local_model_detectors import (
        SOURCE_GLINER,
        SOURCE_STANFORD,
        GlinerPiiDetector,
        StanfordClinicalDetector,
        calibrated_config_for,
    )
    from services.ner_phi_detector import SpacyNerPhiDetector, configured_model_name
    from services.phi_detection import (
        AgeOverThresholdDetector,
        StructuredPatternDetector,
    )

    return {
        D_STRUCTURED: StructuredPatternDetector(),
        D_AGE: AgeOverThresholdDetector(),
        D_SPACY: SpacyNerPhiDetector(
            configured_model_name(), high_recall_proper_nouns=True
        ),
        D_STANFORD: StanfordClinicalDetector(calibrated_config_for(SOURCE_STANFORD)),
        D_GLINER: GlinerPiiDetector(calibrated_config_for(SOURCE_GLINER)),
    }


def run_detectors(partition: str, count: int, use_cache: bool = True) -> Dict[str, Any]:
    """Run every detector once per document, caching by exact configuration."""
    from evaluations.residual_corpus import partition_digest, partition_documents

    documents = partition_documents(partition, count)
    corpus_digest = partition_digest(partition, count)
    fingerprint = _configuration_fingerprint()
    key = _fingerprint_digest(fingerprint, corpus_digest)
    cache_path = CACHE_DIR / f"{partition}-{count}-{key}.json"

    if use_cache and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if _cache_is_usable(cached, key, len(documents)):
            cached["from_cache"] = True
            return cached
        # Stale or incomplete entries are discarded rather than repaired.

    detectors = _detectors_for_text()
    per_document: List[Dict[str, Any]] = []
    timings: Dict[str, List[float]] = {name: [] for name in ALL_DETECTORS}

    for document in documents:
        findings: Dict[str, List[List[Any]]] = {}
        for name, detector in detectors.items():
            started = time.perf_counter()
            produced = detector.detect(document.text)
            timings[name].append((time.perf_counter() - started) * 1000)
            rows = [
                [
                    int(e.start),
                    int(e.end),
                    e.entity_type,
                    None if e.score is None else round(float(e.score), 6),
                ]
                for e in produced
            ]
            rows.sort(key=lambda r: (r[0], r[1], r[2]))
            findings[name] = rows
        per_document.append({"doc_id": document.doc_id, "findings": findings})

    payload = {
        "cache_key": key,
        "partition": partition,
        "count": len(documents),
        "corpus_digest": corpus_digest,
        "fingerprint": fingerprint,
        "documents": per_document,
        "detector_ms": {
            name: round(statistics.fmean(values), 3) if values else 0.0
            for name, values in timings.items()
        },
        "from_cache": False,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _cache_is_usable(cached: Dict[str, Any], key: str, expected_documents: int) -> bool:
    """Reject anything that does not match exactly, rather than adapting."""
    if cached.get("cache_key") != key:
        return False
    if cached.get("count") != expected_documents:
        return False
    documents = cached.get("documents")
    if not isinstance(documents, list) or len(documents) != expected_documents:
        return False
    for entry in documents:
        findings = entry.get("findings")
        if not isinstance(findings, dict):
            return False
        # Every detector must be present. A partial record would silently
        # understate a candidate that depends on the missing one.
        if set(findings) != set(ALL_DETECTORS):
            return False
    return True


# -- candidate composition --------------------------------------------------


def _findings(entry: Dict[str, Any], detector: str) -> List[Finding]:
    return [
        Finding(row[0], row[1], row[2], detector, row[3])
        for row in entry["findings"][detector]
    ]


def compose(entry: Dict[str, Any], candidate: str) -> List[Finding]:
    """The findings a candidate would report for one document."""
    if candidate == "D_cascade":
        return _compose_cascade(entry)
    detectors = CANDIDATES[candidate]
    combined: List[Finding] = []
    for detector in detectors:
        combined.extend(_findings(entry, detector))
    combined.sort(key=lambda f: (f.start, f.end, f.category, f.detector))
    return combined


def _compose_cascade(entry: Dict[str, Any]) -> List[Finding]:
    """Rules and Stanford everywhere; GLiNER only over nominated regions.

    The nomination is deliberately generous - a window around every cheap
    signal - because the cascade is only worth considering if it never loses
    a span. Its cost saving comes from the documents where nothing is
    nominated at all.
    """
    base: List[Finding] = []
    for detector in (D_STRUCTURED, D_AGE, D_SPACY, D_STANFORD):
        base.extend(_findings(entry, detector))

    regions = high_risk_regions(entry)
    for finding in _findings(entry, D_GLINER):
        if any(finding.overlaps(start, end) for start, end in regions):
            base.append(finding)

    base.sort(key=lambda f: (f.start, f.end, f.category, f.detector))
    return base


def high_risk_regions(entry: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Where the cascade would ask GLiNER to look.

    Any span proposed by the cheap detectors, widened by a fixed window. If
    nothing cheap fires anywhere, the document is not nominated and GLiNER is
    not run over it at all - which is exactly the case this candidate has to
    justify.
    """
    anchors: List[Tuple[int, int]] = []
    for detector in (D_STRUCTURED, D_AGE, D_SPACY, D_STANFORD):
        for finding in _findings(entry, detector):
            anchors.append(
                (
                    max(0, finding.start - HIGH_RISK_WINDOW),
                    finding.end + HIGH_RISK_WINDOW,
                )
            )
    if not anchors:
        return []
    anchors.sort()
    merged = [anchors[0]]
    for start, end in anchors[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


# -- scoring ----------------------------------------------------------------


def evaluate_candidate(
    cached: Dict[str, Any], candidate: str, partition: str, count: int
) -> Dict[str, Any]:
    from evaluations.residual_corpus import partition_documents

    documents = partition_documents(partition, count)
    by_id = {d.doc_id: d for d in documents}

    injected = 0
    caught = 0
    missed_categories: Dict[str, int] = {}
    caught_categories: Dict[str, int] = {}
    leaked_documents = 0
    flagged_documents = 0
    unique_credit: Dict[str, int] = {name: 0 for name in ALL_DETECTORS}

    for entry in cached["documents"]:
        document = by_id[entry["doc_id"]]
        findings = compose(entry, candidate)
        if findings:
            flagged_documents += 1

        document_leaked = False
        for span in document.spans:
            injected += 1
            hit = [f for f in findings if f.overlaps(span.start, span.end)]
            if hit:
                caught += 1
                caught_categories[span.category] = (
                    caught_categories.get(span.category, 0) + 1
                )
                # Attribute the catch when exactly one detector family found it.
                detectors = {f.detector for f in hit}
                if len(detectors) == 1:
                    only = next(iter(detectors))
                    unique_credit[only] = unique_credit.get(only, 0) + 1
            else:
                document_leaked = True
                missed_categories[span.category] = (
                    missed_categories.get(span.category, 0) + 1
                )
        if document_leaked:
            leaked_documents += 1

    total_documents = len(cached["documents"])
    detector_ms = cached["detector_ms"]
    if candidate == "D_cascade":
        # GLiNER runs only over nominated documents, so its cost is prorated.
        nominated = sum(1 for entry in cached["documents"] if high_risk_regions(entry))
        share = nominated / total_documents if total_documents else 0.0
        modelled_ms = (
            sum(detector_ms[d] for d in (D_STRUCTURED, D_AGE, D_SPACY, D_STANFORD))
            + detector_ms[D_GLINER] * share
        )
    else:
        nominated = None
        share = None
        modelled_ms = sum(detector_ms[d] for d in CANDIDATES[candidate])

    return {
        "candidate": candidate,
        "detectors": list(CANDIDATES[candidate])
        + (["gliner (nominated regions only)"] if candidate == "D_cascade" else []),
        "documents": total_documents,
        "injected_spans": injected,
        "caught_spans": caught,
        "recall": round(caught / injected, 6) if injected else None,
        "false_negatives": injected - caught,
        "missed_categories": dict(sorted(missed_categories.items())),
        "caught_categories": dict(sorted(caught_categories.items())),
        "document_leakage_rate": (
            round(leaked_documents / total_documents, 6) if total_documents else 0.0
        ),
        "flagged_documents": flagged_documents,
        "flagged_rate": (
            round(flagged_documents / total_documents, 6) if total_documents else 0.0
        ),
        "unique_detector_credit": dict(sorted(unique_credit.items())),
        "modelled_detector_ms": round(modelled_ms, 2),
        "cascade_nominated_documents": nominated,
        "cascade_nominated_share": None if share is None else round(share, 4),
    }


def run(partition: str, count: int, use_cache: bool) -> Dict[str, Any]:
    cached = run_detectors(partition, count, use_cache=use_cache)
    results = {
        name: evaluate_candidate(cached, name, partition, count) for name in CANDIDATES
    }
    control_ms = results["control"]["modelled_detector_ms"] or 1.0
    for result in results.values():
        result["latency_improvement_pct"] = round(
            (control_ms - result["modelled_detector_ms"]) / control_ms * 100, 2
        )
    return {
        "partition": partition,
        "count": count,
        "from_cache": cached.get("from_cache", False),
        "corpus_digest": cached["corpus_digest"],
        "cache_key": cached["cache_key"],
        "detector_ms": cached["detector_ms"],
        "candidates": results,
    }


def print_summary(report: Dict[str, Any]) -> None:
    print()
    print(
        f"partition {report['partition']}  documents={report['count']}  "
        f"cached={report['from_cache']}"
    )
    print(f"detector mean ms/doc: {report['detector_ms']}")
    print()
    header = (
        f"{'candidate':<18}{'recall':>10}{'FN':>5}{'leak':>8}"
        f"{'flagged':>9}{'ms/doc':>9}{'faster%':>9}  missed"
    )
    print(header)
    print("-" * (len(header) + 12))
    for name, r in report["candidates"].items():
        recall = "n/a" if r["recall"] is None else f"{r['recall']:.4f}"
        print(
            f"{name:<18}{recall:>10}{r['false_negatives']:>5}"
            f"{r['document_leakage_rate']:>8.4f}{r['flagged_rate']:>9.4f}"
            f"{r['modelled_detector_ms']:>9.1f}{r['latency_improvement_pct']:>9.1f}"
            f"  {sorted(r['missed_categories']) or '-'}"
        )


def main() -> int:
    from evaluations.residual_corpus import DOCUMENTS_PER_PARTITION, PARTITIONS

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--partition", default="calibration", choices=PARTITIONS)
    parser.add_argument("--count", type=int, default=DOCUMENTS_PER_PARTITION)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    report = run(args.partition, args.count, use_cache=not args.no_cache)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    label = args.label or args.partition
    path = DIAGNOSTICS_DIR / f"residual-shadow-{label}.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print_summary(report)
    print()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
