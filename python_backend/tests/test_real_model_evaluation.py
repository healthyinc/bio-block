"""Opt-in evaluation against the real, locally provisioned model weights.

These tests are skipped by default and are NOT part of the ordinary suite. They
never download anything: they require the pinned manifest snapshots to already
be present in the local Hugging Face cache, and they fail closed if a snapshot
is missing or its checksum does not match.

Run with:

    PHI_RUN_REAL_MODEL_EVAL=1 py -3.11 -m pytest tests/test_real_model_evaluation.py -m real_models

Fixture text below is synthetic. It contains no real patient information.
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import local_model_detectors as models  # noqa: E402

pytestmark = pytest.mark.real_models

REQUIRED = pytest.mark.skipif(
    os.getenv("PHI_RUN_REAL_MODEL_EVAL") != "1",
    reason="real-model evaluation is opt-in; set PHI_RUN_REAL_MODEL_EVAL=1",
)

# Synthetic canaries only. Never add real clinical text to this file.
SYNTHETIC_NOTE = (
    "Patient Synthetic Person was seen by Dr. Fictional Clinician at "
    "Nowhere General Hospital on 2019-04-02. MRN: SYN-4820193. "
    "Contact 555-0100 or synthetic.person@example.invalid."
)


@pytest.fixture(scope="module")
def offline_mode():
    previous = os.environ.get(models.MODEL_MODE_ENV_VAR)
    os.environ[models.MODEL_MODE_ENV_VAR] = models.MODE_OFFLINE
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(models.MODEL_MODE_ENV_VAR, None)
        else:
            os.environ[models.MODEL_MODE_ENV_VAR] = previous


@REQUIRED
def test_pinned_snapshots_are_present_and_checksum_clean(offline_mode):
    for spec in models.load_model_manifest().values():
        # Raises LocalModelError if the snapshot is absent or tampered with.
        models._offline_snapshot(spec)


@REQUIRED
def test_stanford_detector_finds_synthetic_canaries(offline_mode):
    entities = models.StanfordClinicalDetector().detect(SYNTHETIC_NOTE)

    categories = {entity.entity_type for entity in entities}
    assert entities, "clinical de-identifier returned no candidates"
    assert "PERSON" in categories
    assert all(entity.source == models.SOURCE_STANFORD for entity in entities)


@REQUIRED
def test_gliner_detector_finds_synthetic_canaries(offline_mode):
    entities = models.GlinerPiiDetector().detect(SYNTHETIC_NOTE)

    categories = {entity.entity_type for entity in entities}
    assert entities, "open-ended PII model returned no candidates"
    assert categories & {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"}
    assert all(entity.source == models.SOURCE_GLINER for entity in entities)


@REQUIRED
def test_long_input_is_chunked_without_losing_boundary_entities(offline_mode):
    filler = "The patient tolerated the procedure well. " * 60
    text = filler + SYNTHETIC_NOTE + filler
    config = models.DetectorConfig(chunk_size=512, chunk_overlap=128)

    entities = models.StanfordClinicalDetector(config).detect(text)

    assert entities
    # Overlapping windows must not produce duplicate spans.
    spans = [(entity.start, entity.end, entity.entity_type) for entity in entities]
    assert len(spans) == len(set(spans))


@REQUIRED
def test_repeated_detection_is_deterministic(offline_mode):
    detector = models.StanfordClinicalDetector()

    first = detector.detect(SYNTHETIC_NOTE)
    second = detector.detect(SYNTHETIC_NOTE)

    assert first == second
