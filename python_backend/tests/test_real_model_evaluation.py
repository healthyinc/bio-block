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


# ---------------------------------------------------------------------------
# Phase 9: offline-only loading and fail-closed cache handling
# ---------------------------------------------------------------------------


@REQUIRED
def test_pinned_backbone_is_declared_and_verified(offline_mode):
    """GLiNER resolves a tokenizer backbone by repository name.

    That backbone is a second supply-chain input. It must be pinned and
    checksum-verified like any other, or an unpinned tokenizer could be picked
    up silently from whatever happens to be cached.
    """
    backbones = models.backbone_specs_for("gliner_multi_pii")

    assert backbones, "GLiNER's tokenizer backbone is not declared in the manifest"
    backbone = backbones[0]
    assert backbone.repo_id == "microsoft/mdeberta-v3-base"
    assert len(backbone.revision) == 40
    assert len(backbone.weight_sha256) == 64
    assert backbone.role != "detector"
    # Verifying it must not require the network.
    models._offline_snapshot(backbone)


@REQUIRED
def test_detector_specs_exclude_transitive_backbones():
    detectors = models.detector_specs()

    assert set(detectors) == {"stanford_deidentifier", "gliner_multi_pii"}
    assert all(spec.role == "detector" for spec in detectors.values())


@REQUIRED
def test_missing_cache_entry_fails_closed(offline_mode, monkeypatch):
    """A snapshot that is not cached must block, never fall back to a fetch."""
    absent = models.ModelSpec(
        name="absent",
        repo_id="synthetic/not-provisioned",
        revision="b" * 40,
        license="MIT",
        weight_file="pytorch_model.bin",
        weight_sha256="0" * 64,
    )

    with pytest.raises(models.LocalModelError) as exc:
        models._offline_snapshot(absent)

    assert exc.value.error_code == "model_files_unavailable"


@REQUIRED
def test_corrupted_cache_fails_closed(offline_mode, tmp_path):
    """A weight whose bytes changed must not load, even one byte different."""
    spec = models.load_model_manifest()["stanford_deidentifier"]
    snapshot = models._offline_snapshot(spec)

    # Copy the real weight and flip a single byte.
    import shutil

    corrupted_dir = tmp_path / "corrupted"
    corrupted_dir.mkdir()
    source = os.path.join(snapshot, spec.weight_file)
    target = corrupted_dir / spec.weight_file
    shutil.copyfile(source, target)
    with open(target, "r+b") as handle:
        handle.seek(0)
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 0x01]))

    with pytest.raises(models.LocalModelError) as exc:
        models._verify_weight(str(corrupted_dir), spec)

    assert exc.value.error_code == "model_checksum_mismatch"


@REQUIRED
def test_truncated_weight_fails_closed(offline_mode, tmp_path):
    spec = models.load_model_manifest()["mdeberta_backbone"]
    snapshot = models._offline_snapshot(spec)

    truncated_dir = tmp_path / "truncated"
    truncated_dir.mkdir()
    source = os.path.join(snapshot, spec.weight_file)
    with open(source, "rb") as handle:
        head = handle.read(1024)
    (truncated_dir / spec.weight_file).write_bytes(head)

    with pytest.raises(models.LocalModelError) as exc:
        models._verify_weight(str(truncated_dir), spec)

    assert exc.value.error_code == "model_checksum_mismatch"


@REQUIRED
def test_offline_flags_are_set_by_loading(offline_mode):
    spec = models.load_model_manifest()["stanford_deidentifier"]
    models._offline_snapshot(spec)

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


@REQUIRED
def test_real_inference_never_puts_phi_in_an_exception(offline_mode, monkeypatch):
    """A failure mid-inference must carry a code, not the text being scanned."""
    secret = "Ananya Krishnamurthy"

    detector = models.StanfordClinicalDetector()

    def exploding(_texts):
        raise RuntimeError(f"backend died while scoring {secret}")

    monkeypatch.setattr(detector, "_infer", exploding)

    with pytest.raises(models.LocalModelError) as exc:
        detector.detect(f"Patient {secret} attended clinic.")

    assert secret not in str(exc.value)
    assert str(exc.value) == "stanford_inference_failed"


@REQUIRED
def test_real_detections_carry_no_matched_text(offline_mode):
    text = "Patient Ananya Krishnamurthy, MRN SYN-4820193, called 555-0142."

    entities = models.StanfordClinicalDetector().detect(text)

    assert entities
    for entity in entities:
        serialized = repr(entity)
        assert "Ananya" not in serialized
        assert "SYN-4820193" not in serialized
        assert set(vars(entity)) == {
            "entity_type",
            "start",
            "end",
            "source",
            "score",
            "original_label",
        }
