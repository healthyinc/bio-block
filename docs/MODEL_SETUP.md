# Provisioning the local models

The manifest said model artifacts are "provisioned separately" without saying
how. This is the procedure.

Nothing here is required for ordinary development: the default `pytest` run
pins `PHI_MODEL_MODE=legacy_test`, mocks every loader, and downloads nothing.
Follow this only to run the real-model evaluation.

## 1. Create an isolated environment

The model stack is deliberately **not** installed into the main environment.
`transformers==4.44.2` pins `huggingface_hub<1.0` and `tokenizers<0.20`, while
the main environment runs `huggingface_hub 1.16.1` and `tokenizers 0.23.1`
underneath ChromaDB. Installing the model stack in place would downgrade both
and put the 465-test suite at risk.

```powershell
cd python_backend
py -3.11 -m venv .venv-models
.\.venv-models\Scripts\python.exe -m pip install --upgrade pip

# CPU build. See "GPU" below before choosing.
.\.venv-models\Scripts\python.exe -m pip install `
    --index-url https://download.pytorch.org/whl/cpu "torch==2.4.1"

.\.venv-models\Scripts\python.exe -m pip install -r requirements-models.txt
.\.venv-models\Scripts\python.exe -m pip install "spacy==3.8.14" click pytest
.\.venv-models\Scripts\python.exe -m spacy download en_core_web_sm
```

`.venv-models/` and `.model-cache/` are both git-ignored.

## 2. Download and verify the weights

```powershell
.\.venv-models\Scripts\python.exe evaluations\provision_models.py --write-lock
```

This fetches each repository **at the exact pinned revision**, never at HEAD,
into `python_backend/.model-cache/`. It then digests the declared weight file
and compares it to `config/model_manifest.json`. A mismatch, a missing file, or
an unresolvable revision exits non-zero and refuses to report success.

`--write-lock` records a SHA-256 for *every* file in each snapshot into
`config/model_files.lock.json`, so drift in a file the manifest does not pin is
still detectable.

Measured on the reference machine (12-core AMD, CPU-only):

| Entry | Files | Size | Download + verify |
|---|---|---|---|
| `stanford_deidentifier` | 7 | 418.0 MiB | 39.5 s |
| `gliner_multi_pii` | 4 | 1.1 GiB | 104.7 s |
| `mdeberta_backbone` | 3 | 4.1 MiB | 0.3 s |
| **Total** | 14 | **1.5 GiB** | **147.9 s** |

Cache location is `python_backend/.model-cache/` relative to the repository.
No absolute path is recorded in any committed file.

Only anonymous file downloads from the Hugging Face CDN are performed. **No
hosted or paid inference API is contacted at any point**, during setup or at
inference time.

## 3. Verify offline, without network

```powershell
.\.venv-models\Scripts\python.exe evaluations\provision_models.py --verify
```

Sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, resolves every snapshot
from the local cache only, and re-checksums. Completes in about 5 seconds and
fails closed if anything is missing or altered.

## The GLiNER tokenizer backbone

`urchade/gliner_multi_pii-v1` ships only `gliner_config.json` and
`pytorch_model.bin`. At construction it resolves a tokenizer **by repository
name** from `config.model_name`, which is `microsoft/mdeberta-v3-base`. That is
a second supply-chain input, and until Phase 9 it was neither pinned nor
verified: GLiNER simply failed to load offline, and online it would have picked
up whatever HEAD happened to be.

It is now a manifest entry in its own right:

- pinned to revision `a0484667b22365f84929a935b5e50a51f71f159d`
- `spm.model` checksummed against the manifest before GLiNER is constructed
- `allow_patterns` restricts the download to `config.json`, `spm.model` and
  `tokenizer_config.json`, so its own ~1 GB of encoder weights are never
  fetched — GLiNER uses `encoder_from_pretrained=False` and needs only the
  tokenizer
- `alias_ref: main` makes the provisioner write a branch ref pointing at the
  pinned commit, because a backbone is resolved by name rather than by path and
  offline resolution needs one. The content behind that ref is the pinned
  revision and is checksum-verified, so this makes the pin usable rather than
  loosening it.

`load_gliner_model` verifies every declared backbone before building the model.

## GPU

The reference machine has an NVIDIA GTX 1650 Ti (4 GiB VRAM), but the
evaluation was run with a **CPU-only** torch build and all reported figures are
CPU figures. The machine has 7.4 GiB of RAM, and the CUDA build plus a 4 GiB
VRAM budget was not a configuration these numbers could be trusted on.

To evaluate on GPU, install a CUDA torch build instead of the CPU one above and
re-run. `real_model_evaluation.py` records `device` and `cuda_available` in
every report, so GPU and CPU results cannot be confused. **GPU results are not
required**: CPU results stand on their own.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `model_files_unavailable` | Snapshot not cached. Run step 2. |
| `model_checksum_mismatch` | Local weight differs from the manifest. Delete the cache entry and re-download; do not load it. |
| `gliner_model_unavailable` | Usually the backbone is missing. Re-run step 2, which provisions it. |
| `local_models_disabled` | `PHI_MODEL_MODE` is not `offline`. |
| `invalid_model_configuration` | Unknown `PHI_MODEL_MODE`, or contradictory thresholds. |
| `ImportError: DLL load failed` on torch | The install had not finished. Wait for pip to complete. |
