"""Merge per-configuration evaluation runs into the committed reports.

    py evaluations/build_report.py

Reads every ``evaluations/reports/*.local.json`` produced by
``real_model_evaluation.py`` and writes two committed artefacts:

* ``docs/reports/real_model_evaluation.json`` - machine-readable.
* ``docs/reports/REAL_MODEL_EVALUATION.md``   - human-readable.

The ``.local.json`` inputs are git-ignored because they carry raw detection
offsets and machine-specific timings. The committed reports carry metrics,
categories, counts, thresholds and provenance - never a matched value and
never an absolute machine path.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKEND_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

REPORTS_IN = BACKEND_ROOT / "evaluations" / "reports"
REPORTS_OUT = REPO_ROOT / "docs" / "reports"

from evaluations.labelled_corpus import (  # noqa: E402
    CORPUS_VERSION,
    REQUIRED_CATEGORIES,
    corpus_statistics,
)


def _load(name: str) -> Optional[Dict[str, Any]]:
    path = REPORTS_IN / f"{name}.local.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _metrics_row(name: str, run: Dict[str, Any]) -> str:
    m = run["metrics"]
    return (
        f"| {name} | {_fmt(run.get('threshold'), 2)} | {_fmt(m['precision'])} | "
        f"{_fmt(m['span_recall'])} | {_fmt(m['typed_recall'])} | {_fmt(m['f1'])} | "
        f"{m['false_negatives']} | {m['false_positives']} | "
        f"{_fmt(m['document_leakage_rate'], 2)} | "
        f"{_fmt(m['useful_text_preservation'], 3)} |"
    )


def _performance_row(name: str, run: Dict[str, Any]) -> str:
    p = run["performance"]
    return (
        f"| {name} | {p['model_load_seconds']} | {p['latency_ms_mean']} | "
        f"{p['latency_ms_p50']} | {p['latency_ms_p90']} | {p['latency_ms_p99']} | "
        f"{p['latency_ms_max']} | {p['peak_rss_mib']} |"
    )


def build() -> Dict[str, Any]:
    calibration = {
        key: _load(f"calib_{key}")
        for key in ("rules", "spacy", "combined", "residual")
    }
    sweeps = {
        key: _load(f"calib_{key}_sweep") for key in ("stanford", "gliner")
    }
    final = {
        key: _load(f"test_{key}")
        for key in ("rules", "spacy", "stanford", "gliner", "combined", "residual")
    }

    # The joint calibration is the selection authority. The per-detector
    # sweeps are kept as diagnostic curves: judged alone, GLiNER has no
    # admissible threshold at all, which is why selection is done against the
    # combined chain instead.
    joint = _load("joint_calibration")
    selected = (joint or {}).get("selected", {})

    any_run = next(
        (r for r in list(final.values()) + list(calibration.values()) if r), {}
    )

    return {
        "generated": date.today().isoformat(),
        "corpus_version": CORPUS_VERSION,
        "corpus_statistics": corpus_statistics(),
        "required_categories": list(REQUIRED_CATEGORIES),
        "environment": any_run.get("environment", {}),
        "models": any_run.get("models", {}),
        "selected_thresholds": selected,
        "joint_calibration": joint or {},
        "calibration": {
            name: run for name, run in calibration.items() if run
        },
        "sweeps": {name: run for name, run in sweeps.items() if run},
        "final_test": {name: run for name, run in final.items() if run},
    }


def render_markdown(data: Dict[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append

    env = data.get("environment", {})
    models = data.get("models", {})

    add("# Real-model evaluation")
    add("")
    add(
        "Measured against real pinned model weights loaded entirely from a "
        "local, checksum-verified cache with the network disabled. This is "
        "**not** the mocked test suite: those results are in the ordinary "
        "`pytest` run and never load a real weight."
    )
    add("")
    add(f"- Corpus version: `{data['corpus_version']}`")
    add(f"- Generated: {data['generated']}")
    add(
        f"- Device: **{env.get('device', 'unknown')}** "
        f"(torch {env.get('torch')}, CUDA available: {env.get('cuda_available')})"
    )
    add(
        f"- Stack: transformers {env.get('transformers')}, "
        f"gliner {env.get('gliner')}, huggingface_hub {env.get('huggingface_hub')}, "
        f"spacy {env.get('spacy')}"
    )
    add("")

    add("## Pinned models")
    add("")
    add("| Entry | Repository | Revision | Weight SHA-256 | Licence |")
    add("|---|---|---|---|---|")
    for name, spec in models.items():
        add(
            f"| `{name}` | `{spec['repo_id']}` | `{spec['revision']}` | "
            f"`{spec['weight_sha256']}` | {spec['license']} |"
        )
    add("")

    stats = data["corpus_statistics"]
    add("## Corpus")
    add("")
    add(
        "Three partitions with **disjoint value pools**. Thresholds were "
        "selected on `calibration` only; `test` was run once afterwards."
    )
    add("")
    add("| Partition | Documents | Gold spans | Negative controls | Categories |")
    add("|---|---|---|---|---|")
    for partition, s in stats.items():
        add(
            f"| {partition} | {s['documents']} | {s['gold_spans']} | "
            f"{s['negative_terms']} | {s['categories_covered']} |"
        )
    add("")

    joint = data.get("joint_calibration") or {}
    add("## Selected thresholds")
    add("")
    add(
        "Thresholds were selected **jointly, against the combined chain**, not "
        "per model in isolation. Judged alone GLiNER has no admissible "
        "threshold: at 0.0 it has perfect recall but 420 false positives and "
        "zero useful-text preservation, and every positive threshold costs it "
        "recall. But it is one detector among four, and a span it drops is "
        "usually still caught by another, so the question that matters is "
        "what the *chain* retains."
    )
    add("")
    add("| Detector | Selected candidate threshold |")
    add("|---|---|")
    for key, value in sorted(joint.get("selected", {}).items()):
        add(f"| `{key}` | **{value}** |")
    add("")
    add(f"- Selection status: `{joint.get('selection_status')}`")
    add(
        "- Baseline combined span recall at (0.0, 0.0): "
        f"`{joint.get('baseline_combined_span_recall')}` "
        f"with {joint.get('baseline_false_positives')} false positives"
    )
    add("")
    add(f"**Rule.** {joint.get('selection_rule', '')}")
    add("")
    if joint.get("selected_metrics"):
        add("Calibration-partition metrics at the selected pair:")
        add("")
        add("| Metric | Value |")
        add("|---|---|")
        for key, value in sorted(joint["selected_metrics"].items()):
            add(f"| {key} | {value} |")
        add("")

    add("### Per-detector sweeps (diagnostic only)")
    add("")
    add(
        "These curves show each model in isolation. They are **not** the "
        "selection source; the joint calibration above is."
    )
    add("")
    for name, sweep in data.get("sweeps", {}).items():
        add(f"#### {name} in isolation (calibration partition)")
        add("")
        add(
            "| Threshold | Span recall | Precision | F1 | FN | FP | "
            "Doc leakage | Useful text | Cost |"
        )
        add("|---|---|---|---|---|---|---|---|---|")
        for row in sweep.get("sweep", []):
            add(
                f"| {row['threshold']:.2f} | {row['span_recall']:.4f} | "
                f"{row['precision']:.4f} | {row['f1']:.4f} | "
                f"{row['false_negatives']} | {row['false_positives']} | "
                f"{row['document_leakage_rate']:.2f} | "
                f"{row['useful_text_preservation']:.3f} | {row['composite_cost']:.0f} |"
            )
        add("")

    for section, title in (
        ("calibration", "Calibration partition"),
        ("final_test", "Final held-out test partition"),
    ):
        runs = {k: v for k, v in data.get(section, {}).items() if "metrics" in v}
        if not runs:
            continue
        add(f"## {title}")
        add("")
        add(
            "| Detector | Thr | Precision | Span recall | Typed recall | F1 | "
            "FN | FP | Doc leakage | Useful text |"
        )
        add("|---|---|---|---|---|---|---|---|---|---|")
        for name, run in runs.items():
            add(_metrics_row(name, run))
        add("")
        add("| Detector | Load (s) | Mean (ms) | p50 | p90 | p99 | Max | Peak RSS (MiB) |")
        add("|---|---|---|---|---|---|---|---|")
        for name, run in runs.items():
            add(_performance_row(name, run))
        add("")

    # Per-category, from the combined detector on the final partition.
    combined = data.get("final_test", {}).get("combined")
    if combined:
        add("## Per-category performance (combined detector, held-out test)")
        add("")
        add("| Category | Gold | Span recall | Typed recall | Missed |")
        add("|---|---|---|---|---|")
        for cat, stats_ in sorted(
            combined["metrics"]["by_category"].items(),
            key=lambda kv: (kv[1]["span_recall"], kv[0]),
        ):
            add(
                f"| {cat} | {stats_['gold']} | {stats_['span_recall']:.4f} | "
                f"{stats_['typed_recall']:.4f} | {stats_['missed']} |"
            )
        add("")

    residual = data.get("final_test", {}).get("residual")
    if residual:
        result = residual["result"]
        add("## Residual-PHI validator (held-out test)")
        add("")
        add(f"- Documents processed: {result['documents']}")
        add(f"- Gold values surviving redaction: **{result['surviving_gold_count']}**")
        add(
            "- Documents with residual findings: "
            f"{len(result['documents_with_residual_findings'])}"
        )
        add(f"- Residual categories: `{result['residual_categories']}`")
        add(f"- Failures: {len(result['failures'])}")
        add("")

    add("## Findings")
    add("")
    add(
        "1. **The chain generalises; individual models do not carry it alone.** "
        "On the held-out partition the combined chain reached span recall "
        "1.0000 with zero false negatives, zero document leakage and no "
        "missed category. No single detector achieves that: rules reach 0.27 "
        "recall, spaCy 0.66, GLiNER 0.86, Stanford 0.98."
    )
    add(
        "2. **Deterministic rules are precise but narrow.** Precision 1.0000 "
        "with recall 0.2712 and 19 categories at zero recall. The structured "
        "phone pattern requires a 10-digit NANP number, so local-format and "
        "international numbers are not matched by rules at all."
    )
    add(
        "3. **spaCy is the source of over-redaction, not the models.** "
        "Useful-text preservation is 0.357 for spaCy alone and is flat at "
        "0.214 across every point of the joint grid: thresholding the models "
        "cannot recover the clinical eponyms, because the high-recall "
        "proper-noun rule is what redacts them."
    )
    add(
        "4. **The Stanford threshold is inert on this corpus.** Its scores all "
        "exceed 0.70, so every value from 0.05 to 0.70 produces identical "
        "combined results. 0.05 was selected as the most conservative choice "
        "that costs nothing."
    )
    add(
        "5. **AGE_OVER_89 is missed by every individual detector** and is "
        "recovered only incidentally by the chain. There is no age detector "
        "and no internal AGE category; Safe Harbor requires ages above 89 to "
        "be aggregated. See the limitations below."
    )
    add(
        "6. **GLiNER had an unpinned supply-chain dependency.** It resolves "
        "its tokenizer by repository name from `microsoft/mdeberta-v3-base`, "
        "which was neither in the manifest nor checksum-verified, and which "
        "made offline loading fail outright. It is now a pinned, verified "
        "manifest entry."
    )
    add("")

    add("## Limitations and manual-review conditions")
    add("")
    add(
        "- **The residual validator blocks most documents once the real "
        "models are enabled.** On the held-out partition it reported residual "
        "findings in 9 of 10 documents while zero gold values actually "
        "survived. This is fail-closed and no PHI escapes, but in practice it "
        "means enabling `PHI_MODEL_MODE=offline` turns automatic text release "
        "into manual review for almost everything. The validator re-runs the "
        "full chain over masked text, and model detections on placeholder-"
        "heavy prose are artefacts of masking rather than surviving PHI. "
        "Changing that behaviour needs its own evaluation and is deliberately "
        "not bundled into this calibration."
    )
    add(
        "- **AGE_OVER_89 has no dedicated detector.** Treat any age above 89 "
        "as requiring manual review until an age rule exists."
    )
    add(
        "- **The corpus is 10 documents per partition.** Every grid point tied "
        "on recall, so the selection was driven by typed recall and safety "
        "margin rather than by recall evidence. These thresholds should be "
        "re-derived against a larger corpus before being treated as settled."
    )
    add(
        "- **Useful-text preservation is 0.214.** Roughly four in five "
        "recorded non-PHI clinical terms are still swept up. That is a "
        "usability cost, not a privacy failure, and it is attributable to "
        "spaCy rather than to either pinned model."
    )
    add(
        "- **CPU only.** A CUDA GPU is present on the reference machine but "
        "the evaluation used a CPU-only torch build. No GPU figures are "
        "reported and none are implied."
    )
    add("")

    add("## Claim boundary")
    add("")
    add(
        "These figures come from a small synthetic corpus with invented "
        "values. Zero surviving canaries on this corpus is an acceptance "
        "condition for this suite. It is **not** proof of zero real-world PHI "
        "leakage, and it is **not** a statement about HIPAA compliance. "
        "Recall is measured against expected categories on 10 documents per "
        "partition, not a labelled clinical gold standard."
    )
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    data = build()
    REPORTS_OUT.mkdir(parents=True, exist_ok=True)

    json_path = REPORTS_OUT / "real_model_evaluation.json"
    json_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path = REPORTS_OUT / "REAL_MODEL_EVALUATION.md"
    md_path.write_text(render_markdown(data), encoding="utf-8")

    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
