# Clinical Text PHI Detection Setup

Clinical `.txt` ingestion uses a trained spaCy NER pipeline together with local
structured-identifier patterns and contextual person-name rules. Microsoft
Presidio remains installed for image anonymization, but it is not used by the
clinical-text service in `services/text_anonymization.py`.

## Install

Install all backend dependencies from the pinned requirements file:

```bash
cd python_backend
python -m pip install -r requirements.txt
```

The requirements file installs the CPU-compatible `en_core_web_sm` 3.8.0 wheel
with its SHA-256 checksum. That model supports spaCy `>=3.8.0,<3.9.0`; no model
is downloaded while an API request is running.

To install only the trained model in an existing spaCy 3.8 environment, use the
same pinned artifact:

```bash
python -m pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl#sha256=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
```

Verify that the trained NER component loads before starting the API:

```bash
python -c "import spacy; assert 'ner' in spacy.load('en_core_web_sm').pipe_names"
```

## Configuration

The default model package is `en_core_web_sm`. Deployments can select another
installed package by setting:

```text
PHI_NER_MODEL=en_core_web_sm
```

Only installed Python package names are accepted. Local filesystem paths are
rejected so paths cannot leak through response metadata or errors.

`BIOBLOCK_STUDY_SALT` must also be configured for deterministic surrogates. Do
not commit the real salt.

If the configured trained model is absent or cannot load, text anonymization
fails closed with `ner_model_unavailable`. It never reports unmodified input as
success when NER did not run.
