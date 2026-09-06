# Phase 11 evaluation — corpus v4.0, held-out partition `heldout_v3`

Run once, on the configuration pinned in
[`config/evaluation_freeze.json`](../../python_backend/config/evaluation_freeze.json),
with the pinned Stanford and GLiNER models loaded offline from the local
cache. Two hundred synthetic documents. Every value in the corpus is invented.

    <model-venv-python> evaluations/phase11_evaluation.py --partition heldout_v3

## Result

### Privacy

| Metric | Phase 10 held-out (10 docs) | Phase 11 held-out (200 docs) |
|---|---|---|
| PHI recall (span) | 1.0000 | **1.0000** |
| False negatives | 0 | **0** |
| Residual canaries | 0 | **0** |
| Document leakage rate | 0.00 | **0.00** |
| PHI precision | 0.7284 | **0.8009** |
| Typed recall | — | **0.9101** |
| F1 | — | **0.8894** |
| Missed categories | none | **none** |

### Utility

| Metric | Phase 9 | Phase 10 | Phase 11 |
|---|---|---|---|
| Useful-text preservation | 0.2143 | 0.9286 | **0.9898** |
| Clinical-term preservation | not measured | 1.0000 | **1.0000** |
| Content-token preservation | — | — | **0.9856** |
| Numeric preservation | — | — | **1.0000** |
| Negative terms wrongly redacted | — | — | **2 of 196** |

### Release

| Metric | Value |
|---|---|
| Documents | 200 |
| Manual-review rate | **0.485** |
| Blocked by residual findings | 0.325 |
| Held for review only | 0.160 |
| Failures | 0 |

Second-pass findings by classification: 210 exact generated surrogate, 85
additional plausible PHI, 56 anonymizer placeholder, 35 detector artefact from
modified context, 1 useful clinical content.

### Performance

Mean 3,793 ms per document, maximum 44,282 ms, on CPU with both models
resident in one process.

## The manual-review target was not met

The engineering target was 0.20. The measured rate is **0.485**, so automatic
text release with the real model chain is **not yet practical**, and the
conservative block stays in place. This is the outcome the phase brief
anticipated: keep the block and say so rather than relax a privacy check to
reach a number.

What moved during calibration, in order, each fixing a defect the real chain
exposed on the calibration partition:

| Correction | Calibration manual review |
|---|---|
| Phase 10 baseline (10 documents) | 0.60 |
| Provenance map replaces masking (200 documents) | 0.760 |
| Diagnoses are never identifiers; straddling remainders examined | 0.645 |
| A span with no proper-noun token is not a name | 0.580 |
| Layer 1 needs structural syntax; relative time is not a date | 0.530 |

The first row and the rest are not comparable — ten documents cannot express
a rate finer than ten per cent, which is why the corpus grew. Within the
200-document measurements the direction is consistent and every step came
from a named defect with a regression test, not from a threshold nudge.

What remains is not one defect class. Roughly a third of held documents carry
a second-pass finding on text outside every generated region, and a sixth are
Layer 5 escalations where two detectors agree a span is a name while a
clinical reading also applies. Both are the machinery working as designed;
reducing them further means improving first-pass precision, not loosening the
validator.

## What these numbers are not

- **Not evidence of HIPAA compliance.** Nothing here certifies anything. The
  status vocabulary stays `safe_harbor_technical_checks_passed`,
  `residual_phi_not_detected`, `privacy_requirements_not_met`,
  `manual_review_required`, `expert_determination_required`,
  `unsupported_or_unscannable`.
- **Not evidence of zero real-world leakage.** Zero residual canaries means
  no *invented* value survived on a *synthetic* corpus. Real notes carry
  spellings, abbreviations and formats this corpus does not.
- **Not a legal threshold.** 0.20 is a project engineering target, recorded
  as such in the freeze file.

## Partition discipline

`heldout_v3` was run once, after the configuration was frozen. It had not
informed any decision before that run, and this report is its result whatever
it says.

Two partitions are spent and are diagnostic data from here on: `test` was
inspected during Phase 10, and `heldout_v2` was reported before the clinical
vocabulary was extended. `tests/test_evaluation_freeze.py` fails if any frozen
value — corpus digest, model revision, threshold, rule version, label map —
moves, because a held-out number describes one configuration and nothing else.

Any further tuning requires a corpus v5 with a fresh held-out partition.
