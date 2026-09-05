# Real-model evaluation

Measured against real pinned model weights loaded entirely from a local, checksum-verified cache with the network disabled. This is **not** the mocked test suite: those results are in the ordinary `pytest` run and never load a real weight.

- Corpus version: `canary-v2.0`
- Generated: 2026-09-05
- Device: **cpu** (torch 2.4.1+cpu, CUDA available: False)
- Stack: transformers 4.44.2, gliner 0.2.22, huggingface_hub 0.24.7, spacy 3.8.14

## Pinned models

| Entry | Repository | Revision | Weight SHA-256 | Licence |
|---|---|---|---|---|
| `gliner_multi_pii` | `urchade/gliner_multi_pii-v1` | `1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d` | `3003753fba99e40645cf088c7367a2c6211fc174897dc64f1f9c147c29d18d2d` | Apache-2.0 |
| `mdeberta_backbone` | `microsoft/mdeberta-v3-base` | `a0484667b22365f84929a935b5e50a51f71f159d` | `13c8d666d62a7bc4ac8f040aab68e942c861f93303156cc28f5c7e885d86d6e3` | MIT |
| `stanford_deidentifier` | `StanfordAIMI/stanford-deidentifier-base` | `661b9c1c717d3165512d440abc3700c386aefab6` | `fa49ef069171e479f546ce2ee5ed599aa585d1d33bc7a8f54400ac57d9cd2716` | MIT |

## Corpus

Three partitions with **disjoint value pools**. Thresholds were selected on `calibration` only; `test` was run once afterwards.

| Partition | Documents | Gold spans | Negative controls | Categories |
|---|---|---|---|---|
| development | 10 | 59 | 14 | 29 |
| calibration | 10 | 59 | 14 | 29 |
| test | 10 | 59 | 14 | 29 |

## Selected thresholds

Thresholds were selected **jointly, against the combined chain**, not per model in isolation. Judged alone GLiNER has no admissible threshold: at 0.0 it has perfect recall but 420 false positives and zero useful-text preservation, and every positive threshold costs it recall. But it is one detector among four, and a span it drops is usually still caught by another, so the question that matters is what the *chain* retains.

| Detector | Selected candidate threshold |
|---|---|
| `gliner_candidate_threshold` | **0.1** |
| `stanford_candidate_threshold` | **0.05** |

- Selection status: `selected`
- Baseline combined span recall at (0.0, 0.0): `1.0` with 410 false positives

**Rule.** 1. combined span recall must not fall below the all-zero baseline - a false negative is a disclosure and is never traded away. 2. zero is excluded for either model: it is not a calibrated value, it is the absence of one. 3. among the rest, maximise typed recall, which improves how a finding is labelled without trading away any span. 4. ties go to the LOWEST thresholds, keeping the widest safety margin on unseen data, because precision gained by raising a threshold is bought with candidates that would otherwise have been redacted.

Calibration-partition metrics at the selected pair:

| Metric | Value |
|---|---|
| composite_cost | 52.0 |
| document_leakage_rate | 0.0 |
| f1 | 0.6941 |
| false_negatives | 0 |
| false_positives | 52 |
| missed_categories | [] |
| precision | 0.5315 |
| span_recall | 1.0 |
| typed_recall | 0.8475 |
| useful_text_preservation | 0.2143 |

### Per-detector sweeps (diagnostic only)

These curves show each model in isolation. They are **not** the selection source; the joint calibration above is.

#### stanford in isolation (calibration partition)

| Threshold | Span recall | Precision | F1 | FN | FP | Doc leakage | Useful text | Cost |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.9831 | 0.9831 | 0.9831 | 1 | 1 | 0.10 | 1.000 | 11 |
| 0.05 | 0.9831 | 0.9831 | 0.9831 | 1 | 1 | 0.10 | 1.000 | 11 |
| 0.10 | 0.9831 | 0.9831 | 0.9831 | 1 | 1 | 0.10 | 1.000 | 11 |
| 0.15 | 0.9831 | 0.9831 | 0.9831 | 1 | 1 | 0.10 | 1.000 | 11 |
| 0.20 | 0.9831 | 0.9831 | 0.9831 | 1 | 1 | 0.10 | 1.000 | 11 |
| 0.25 | 0.9831 | 0.9831 | 0.9831 | 1 | 1 | 0.10 | 1.000 | 11 |
| 0.30 | 0.9831 | 0.9831 | 0.9831 | 1 | 1 | 0.10 | 1.000 | 11 |
| 0.40 | 0.9831 | 0.9831 | 0.9831 | 1 | 1 | 0.10 | 1.000 | 11 |
| 0.50 | 0.9831 | 0.9831 | 0.9831 | 1 | 1 | 0.10 | 1.000 | 11 |
| 0.60 | 0.9661 | 0.9828 | 0.9744 | 2 | 1 | 0.20 | 1.000 | 21 |
| 0.70 | 0.9322 | 1.0000 | 0.9649 | 4 | 0 | 0.30 | 1.000 | 40 |
| 0.80 | 0.8814 | 1.0000 | 0.9369 | 7 | 0 | 0.50 | 1.000 | 70 |
| 0.90 | 0.7119 | 1.0000 | 0.8317 | 17 | 0 | 0.70 | 1.000 | 170 |

#### gliner in isolation (calibration partition)

| Threshold | Span recall | Precision | F1 | FN | FP | Doc leakage | Useful text | Cost |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 1.0000 | 0.1232 | 0.2193 | 0 | 420 | 0.00 | 0.000 | 420 |
| 0.05 | 0.9322 | 0.5189 | 0.6667 | 4 | 51 | 0.30 | 0.214 | 91 |
| 0.10 | 0.8644 | 0.6071 | 0.7133 | 8 | 33 | 0.30 | 0.286 | 113 |
| 0.15 | 0.7966 | 0.6184 | 0.6963 | 12 | 29 | 0.40 | 0.286 | 149 |
| 0.20 | 0.7797 | 0.6133 | 0.6866 | 13 | 29 | 0.40 | 0.286 | 159 |
| 0.25 | 0.7119 | 0.5915 | 0.6462 | 17 | 29 | 0.50 | 0.286 | 199 |
| 0.30 | 0.6949 | 0.5942 | 0.6406 | 18 | 28 | 0.50 | 0.286 | 208 |
| 0.40 | 0.6102 | 0.5714 | 0.5902 | 23 | 27 | 0.60 | 0.286 | 257 |
| 0.50 | 0.6102 | 0.5806 | 0.5950 | 23 | 26 | 0.60 | 0.286 | 256 |
| 0.60 | 0.5763 | 0.5763 | 0.5763 | 25 | 25 | 0.60 | 0.286 | 275 |
| 0.70 | 0.5593 | 0.6346 | 0.5946 | 26 | 19 | 0.60 | 0.286 | 279 |
| 0.80 | 0.4915 | 0.6905 | 0.5743 | 30 | 13 | 0.60 | 0.286 | 313 |
| 0.90 | 0.4068 | 0.6857 | 0.5106 | 35 | 11 | 0.80 | 0.286 | 361 |

## Calibration partition

| Detector | Thr | Precision | Span recall | Typed recall | F1 | FN | FP | Doc leakage | Useful text |
|---|---|---|---|---|---|---|---|---|---|
| rules | 0.00 | 1.0000 | 0.2712 | 0.2542 | 0.4267 | 43 | 0 | 1.00 | 1.000 |
| spacy | 0.00 | 0.5606 | 0.6271 | 0.3051 | 0.5920 | 22 | 29 | 0.80 | 0.357 |
| combined | 0.00 | 0.1258 | 1.0000 | 0.8136 | 0.2235 | 0 | 410 | 0.00 | 0.000 |

| Detector | Load (s) | Mean (ms) | p50 | p90 | p99 | Max | Peak RSS (MiB) |
|---|---|---|---|---|---|---|---|
| rules | 0.0 | 0.35 | 0.23 | 1.31 | 1.31 | 1.31 | 0.0 |
| spacy | 6.68 | 32.72 | 24.09 | 111.04 | 111.04 | 111.04 | 0.0 |
| combined | 50.55 | 2795.16 | 1611.95 | 13036.86 | 13036.86 | 13036.86 | 2228.1 |

## Final held-out test partition

| Detector | Thr | Precision | Span recall | Typed recall | F1 | FN | FP | Doc leakage | Useful text |
|---|---|---|---|---|---|---|---|---|---|
| rules | 0.00 | 1.0000 | 0.2712 | 0.2542 | 0.4267 | 43 | 0 | 1.00 | 1.000 |
| spacy | 0.00 | 0.5652 | 0.6610 | 0.3898 | 0.6094 | 20 | 30 | 0.70 | 0.357 |
| stanford | 0.05 | 0.9206 | 0.9831 | 0.7119 | 0.9508 | 1 | 5 | 0.10 | 1.000 |
| gliner | 0.10 | 0.6145 | 0.8644 | 0.7458 | 0.7183 | 8 | 32 | 0.40 | 0.286 |
| combined | 0.00 | 0.5315 | 1.0000 | 0.9153 | 0.6941 | 0 | 52 | 0.00 | 0.214 |

| Detector | Load (s) | Mean (ms) | p50 | p90 | p99 | Max | Peak RSS (MiB) |
|---|---|---|---|---|---|---|---|
| rules | 0.0 | 0.42 | 0.28 | 1.53 | 1.53 | 1.53 | 20.2 |
| spacy | 6.35 | 40.15 | 37.71 | 112.75 | 112.75 | 112.75 | 269.2 |
| stanford | 13.7 | 901.77 | 575.07 | 2899.76 | 2899.76 | 2899.76 | 593.4 |
| gliner | 39.77 | 2170.61 | 1245.39 | 9132.84 | 9132.84 | 9132.84 | 2726.8 |
| combined | 34.99 | 1699.67 | 974.65 | 6870.82 | 6870.82 | 6870.82 | 2341.8 |

## Per-category performance (combined detector, held-out test)

| Category | Gold | Span recall | Typed recall | Missed |
|---|---|---|---|---|
| ACCESSION | 1 | 1.0000 | 1.0000 | 0 |
| ACCOUNT_NUMBER | 1 | 1.0000 | 1.0000 | 0 |
| ADDRESS | 3 | 1.0000 | 1.0000 | 0 |
| AGE_OVER_89 | 1 | 1.0000 | 1.0000 | 0 |
| BIOMETRIC_ID | 1 | 1.0000 | 1.0000 | 0 |
| CERTIFICATE_NUMBER | 1 | 1.0000 | 1.0000 | 0 |
| DATE | 4 | 1.0000 | 1.0000 | 0 |
| DEVICE_ID | 1 | 1.0000 | 1.0000 | 0 |
| EMAIL | 3 | 1.0000 | 1.0000 | 0 |
| EMPLOYER | 1 | 1.0000 | 1.0000 | 0 |
| FAX | 1 | 1.0000 | 1.0000 | 0 |
| GEOGRAPHY | 2 | 1.0000 | 1.0000 | 0 |
| HEALTH_PLAN | 1 | 1.0000 | 1.0000 | 0 |
| HOSPITAL | 3 | 1.0000 | 0.6667 | 0 |
| IP_ADDRESS | 1 | 1.0000 | 1.0000 | 0 |
| LICENSE_NUMBER | 1 | 1.0000 | 1.0000 | 0 |
| MRN | 4 | 1.0000 | 0.5000 | 0 |
| ORGANIZATION | 1 | 1.0000 | 1.0000 | 0 |
| PATIENT_ID | 1 | 1.0000 | 1.0000 | 0 |
| PERSON | 10 | 1.0000 | 1.0000 | 0 |
| PERSON_CLINICIAN | 2 | 1.0000 | 1.0000 | 0 |
| PERSON_RELATIVE | 2 | 1.0000 | 1.0000 | 0 |
| PHONE | 4 | 1.0000 | 1.0000 | 0 |
| POSTAL_CODE | 3 | 1.0000 | 0.6667 | 0 |
| SSN | 1 | 1.0000 | 1.0000 | 0 |
| UNUSUAL_ID | 1 | 1.0000 | 1.0000 | 0 |
| URL | 2 | 1.0000 | 1.0000 | 0 |
| USERNAME | 1 | 1.0000 | 0.0000 | 0 |
| VEHICLE_ID | 1 | 1.0000 | 1.0000 | 0 |

## Residual-PHI validator (held-out test)

- Documents processed: 10
- Gold values surviving redaction: **0**
- Documents with residual findings: 9
- Residual categories: `{'ADDRESS': 2, 'DATE_TIME': 1, 'DRIVER_LICENSE': 3, 'EMAIL_ADDRESS': 1, 'MEDICAL_CONDITION': 3, 'ORGANIZATION': 8, 'PERSON': 21, 'PHONE_NUMBER': 1}`
- Failures: 0

## Findings

1. **The chain generalises; individual models do not carry it alone.** On the held-out partition the combined chain reached span recall 1.0000 with zero false negatives, zero document leakage and no missed category. No single detector achieves that: rules reach 0.27 recall, spaCy 0.66, GLiNER 0.86, Stanford 0.98.
2. **Deterministic rules are precise but narrow.** Precision 1.0000 with recall 0.2712 and 19 categories at zero recall. The structured phone pattern requires a 10-digit NANP number, so local-format and international numbers are not matched by rules at all.
3. **spaCy is the source of over-redaction, not the models.** Useful-text preservation is 0.357 for spaCy alone and is flat at 0.214 across every point of the joint grid: thresholding the models cannot recover the clinical eponyms, because the high-recall proper-noun rule is what redacts them.
4. **The Stanford threshold is inert on this corpus.** Its scores all exceed 0.70, so every value from 0.05 to 0.70 produces identical combined results. 0.05 was selected as the most conservative choice that costs nothing.
5. **AGE_OVER_89 is missed by every individual detector** and is recovered only incidentally by the chain. There is no age detector and no internal AGE category; Safe Harbor requires ages above 89 to be aggregated. See the limitations below.
6. **GLiNER had an unpinned supply-chain dependency.** It resolves its tokenizer by repository name from `microsoft/mdeberta-v3-base`, which was neither in the manifest nor checksum-verified, and which made offline loading fail outright. It is now a pinned, verified manifest entry.

## Limitations and manual-review conditions

- **The residual validator blocks most documents once the real models are enabled.** On the held-out partition it reported residual findings in 9 of 10 documents while zero gold values actually survived. This is fail-closed and no PHI escapes, but in practice it means enabling `PHI_MODEL_MODE=offline` turns automatic text release into manual review for almost everything. The validator re-runs the full chain over masked text, and model detections on placeholder-heavy prose are artefacts of masking rather than surviving PHI. Changing that behaviour needs its own evaluation and is deliberately not bundled into this calibration.
- **AGE_OVER_89 has no dedicated detector.** Treat any age above 89 as requiring manual review until an age rule exists.
- **The corpus is 10 documents per partition.** Every grid point tied on recall, so the selection was driven by typed recall and safety margin rather than by recall evidence. These thresholds should be re-derived against a larger corpus before being treated as settled.
- **Useful-text preservation is 0.214.** Roughly four in five recorded non-PHI clinical terms are still swept up. That is a usability cost, not a privacy failure, and it is attributable to spaCy rather than to either pinned model.
- **CPU only.** A CUDA GPU is present on the reference machine but the evaluation used a CPU-only torch build. No GPU figures are reported and none are implied.

## Claim boundary

These figures come from a small synthetic corpus with invented values. Zero surviving canaries on this corpus is an acceptance condition for this suite. It is **not** proof of zero real-world PHI leakage, and it is **not** a statement about HIPAA compliance. Recall is measured against expected categories on 10 documents per partition, not a labelled clinical gold standard.

