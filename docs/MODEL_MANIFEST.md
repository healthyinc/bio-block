# Local model manifest

Model artifacts are provisioned separately and are never committed. Runtime
loading is offline-only (`local_files_only=True`), lazy, and bound to the exact
revision and weight SHA-256 in `python_backend/config/model_manifest.json`.

| Role | Repository | Revision | Pinned file SHA-256 | Declared license |
|---|---|---|---|---|
| Clinical text detector | `StanfordAIMI/stanford-deidentifier-base` | `661b9c1c717d3165512d440abc3700c386aefab6` | `fa49ef069171e479f546ce2ee5ed599aa585d1d33bc7a8f54400ac57d9cd2716` | MIT |
| Open-ended PII detector | `urchade/gliner_multi_pii-v1` | `1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d` | `3003753fba99e40645cf088c7367a2c6211fc174897dc64f1f9c147c29d18d2d` | Apache-2.0 |
| GLiNER tokenizer backbone | `microsoft/mdeberta-v3-base` | `a0484667b22365f84929a935b5e50a51f71f159d` | `13c8d666d62a7bc4ac8f040aab68e942c861f93303156cc28f5c7e885d86d6e3` (`spm.model`) | MIT |

`config/model_files.lock.json` extends this to a SHA-256 for **every** file in
each snapshot, so drift in a file the manifest does not pin is still
detectable. Setup is documented in [MODEL_SETUP.md](MODEL_SETUP.md).

### The tokenizer backbone is a real supply-chain input

GLiNER's snapshot ships only its own weights and config. At construction it
resolves a tokenizer **by repository name** from `config.model_name`. Until
Phase 9 that repository was neither pinned nor verified, and the consequence
was concrete: GLiNER could not load offline at all. It is now a manifest entry
with its own revision and checksum, restricted by `allow_patterns` to the three
tokenizer files so its ~1 GB of encoder weights are never fetched.
`load_gliner_model` verifies it before building the model.

The license entries record upstream model-card declarations and are not legal
advice. Optional adapter dependencies are pinned in `requirements-models.txt`.
Ordinary unit tests use mocks and do not download or load these weights.

## Role of the models

Both adapters are **detectors only**. They propose candidate spans that are
normalized into the internal PHI taxonomy and then handed to the Week 6 Safe
Harbor logic, which remains the sole authority for transformation, validation,
and the release decision. No model output can make an artifact releasable.

## Loading contract

- Loaders are lazy and `lru_cache`d, so each worker resolves and loads a model
  at most once.
- `snapshot_download(..., local_files_only=True)` resolves a locally cached
  snapshot. `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` are also forced on, so
  no transitive helper can reach the network.
- The declared weight file is digested with SHA-256 (streamed, 1 MiB at a time)
  and compared to the manifest **before** the model is constructed. A mismatch
  raises `model_checksum_mismatch` and blocks; it is never a warning.

## Chunking and overlap resolution

Long text is split into overlapping windows (`chunk_size` / `chunk_overlap`) so
an entity that straddles a window boundary is still wholly contained in at
least one window, provided it is no longer than `chunk_overlap` characters. A
zero overlap is rejected at configuration time.

Because windows overlap, the same entity is routinely proposed twice.
`merge_chunk_entities` resolves this deterministically, per detector and
category: identical spans collapse to the highest-scoring copy, and overlapping
spans collapse to the longest, then highest-scoring, then earliest copy. The
result is independent of the order predictions arrive in. Agreement between
*different* detectors is deliberately preserved and left to the pipeline-level
`resolve_overlaps`, whose ordering key is total so redaction cannot vary
between runs.

Inference is batch-aware: windows are submitted in batches of `batch_size`,
using each model's native batch API where one exists.

## Fail-closed error codes

Every one of these blocks the artifact. None of them ever falls back to
returning original or partially redacted content, and none carries a matched
value — only a code.

| Code | Raised when |
|---|---|
| `invalid_model_configuration` | Unparseable/contradictory config, or an unknown `PHI_MODEL_MODE` |
| `invalid_model_manifest` | The manifest is missing, malformed, or has unexpected fields |
| `local_models_disabled` | A model load was attempted outside `offline` mode |
| `model_files_unavailable` | The pinned snapshot or its weight file is not cached locally |
| `model_checksum_mismatch` | The local weight file does not match the pinned digest |
| `stanford_model_unavailable` / `gliner_model_unavailable` | Construction of the loaded model failed |
| `stanford_inference_failed` / `gliner_inference_failed` | Inference raised |
| `model_inference_timeout` | The wall-clock inference budget was exhausted |
| `model_output_malformed` | Prediction count mismatch, missing offsets, or offsets outside the window |

Out-of-range offsets fail closed rather than being silently dropped: dropping a
candidate would be a fail-open path.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `PHI_MODEL_MODE` | `offline` | `offline` loads the pinned local models; `legacy_test` runs the rule-based and spaCy detectors only. Any other value fails closed. |
| `PHI_CANDIDATE_THRESHOLD` | *(locked file)* | Overrides the calibrated threshold for both models. Set only to tighten or loosen one deployment |
| `PHI_REDACTION_THRESHOLD` | *(locked file)* | As above |
| `PHI_TEXT_CHUNK_SIZE` | `2000` | Window size in characters |
| `PHI_TEXT_CHUNK_OVERLAP` | `200` | Window overlap in characters; must be ≥ 1 and < `chunk_size` |
| `PHI_MODEL_BATCH_SIZE` | `8` | Windows submitted per batch |
| `PHI_MODEL_TIMEOUT_SECONDS` | `120` | Wall-clock budget for one `detect()` call |
| `PHI_THRESHOLDS_CALIBRATED` | `0` | Legacy flag. `calibrated` is now derived from whether the locked file supplied the value |

## Calibrated thresholds

Thresholds are no longer zero. `config/detection_thresholds.json` holds the
locked, calibrated values:

| Detector | Candidate threshold |
|---|---|
| `stanford_deidentifier` | **0.05** |
| `gliner_multi_pii` | **0.10** |

They were selected **jointly against the combined chain**, on the calibration
partition only, and the held-out partition was run once afterwards. Precedence
at load time is: environment variable, then the locked file, then the
conservative zero default. A missing or unreadable calibration therefore
over-redacts rather than under-redacts.

Full method, per-category results and limitations:
[reports/REAL_MODEL_EVALUATION.md](reports/REAL_MODEL_EVALUATION.md).

Zero residual synthetic canaries remains a test acceptance condition, not proof
of zero PHI leakage.

## Running real-model checks

The ordinary suite never touches real weights: `tests/conftest.py` pins
`PHI_MODEL_MODE=legacy_test`, and every adapter test monkeypatches the loaders.
Real-model checks are opt-in and require the pinned snapshots to already be in
the local cache:

```
PHI_RUN_REAL_MODEL_EVAL=1 py -3.11 -m pytest tests/test_real_model_evaluation.py -m real_models
PHI_RUN_REAL_MODEL_EVAL=1 py -3.11 evaluations/real_model_smoke.py
```

Both use synthetic canary text only and report categories and counts, never
matched values.
