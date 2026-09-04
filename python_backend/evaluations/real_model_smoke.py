"""Opt-in smoke check for locally pre-fetched real model weights.

Run only after separately provisioning the pinned manifest snapshots:

    PHI_RUN_REAL_MODEL_EVAL=1 py -3.11 evaluations/real_model_smoke.py

Loading is offline-only and checksum-verified. Nothing is downloaded. Only
counts, categories, and detector names are printed: never a matched value.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.local_model_detectors import (  # noqa: E402
    MODE_OFFLINE,
    MODEL_MODE_ENV_VAR,
    GlinerPiiDetector,
    LocalModelError,
    StanfordClinicalDetector,
)

# Synthetic canaries only. Never put real clinical text in this file.
SYNTHETIC_NOTE = "Patient Synthetic Person has identifier SYN-12345."


def main() -> int:
    if os.getenv("PHI_RUN_REAL_MODEL_EVAL") != "1":
        print("Real-model evaluation skipped; set PHI_RUN_REAL_MODEL_EVAL=1.")
        return 0

    os.environ[MODEL_MODE_ENV_VAR] = MODE_OFFLINE
    result = {}
    for detector in (StanfordClinicalDetector(), GlinerPiiDetector()):
        name = detector.__class__.__name__
        try:
            entities = detector.detect(SYNTHETIC_NOTE)
        except LocalModelError as exc:
            # Fail closed and report the code only.
            result[name] = {"status": "blocked", "error_code": exc.error_code}
            continue
        categories = {}
        for entity in entities:
            categories[entity.entity_type] = categories.get(entity.entity_type, 0) + 1
        result[name] = {
            "status": "completed",
            "candidate_count": len(entities),
            "categories": categories,
        }

    print(json.dumps(result, sort_keys=True, indent=2))
    completed = [
        item
        for item in result.values()
        if item.get("status") == "completed" and item.get("candidate_count")
    ]
    return 0 if len(completed) == len(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
