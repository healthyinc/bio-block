import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.local_model_detectors import MODE_LEGACY_TEST, MODEL_MODE_ENV_VAR  # noqa: E402


# Ordinary unit tests run against deterministic local doubles and the already
# pinned small spaCy fixture model. The 438 MB / 1.16 GB pinned snapshots are
# never fetched or loaded here; real-model checks are a separate opt-in command
# (see tests/test_real_model_evaluation.py and evaluations/real_model_smoke.py).
os.environ[MODEL_MODE_ENV_VAR] = MODE_LEGACY_TEST


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_models: opt-in checks that load locally provisioned pinned weights; "
        "skipped unless PHI_RUN_REAL_MODEL_EVAL=1.",
    )
