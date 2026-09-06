"""Mocked unit tests for the local model adapters.

Nothing here downloads or loads real weights: every loader is monkeypatched.
Real-model behaviour is covered by the opt-in suite in
tests/test_real_model_evaluation.py.
"""

import json
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import local_model_detectors as models  # noqa: E402
from services.privacy_contracts import PhiEntity  # noqa: E402


# --------------------------------------------------------------------------
# Manifest pinning and configuration
# --------------------------------------------------------------------------


def test_manifest_pins_immutable_revisions_checksums_and_licenses():
    manifest = models.load_model_manifest()

    assert manifest["stanford_deidentifier"].revision == (
        "661b9c1c717d3165512d440abc3700c386aefab6"
    )
    assert manifest["stanford_deidentifier"].license == "MIT"
    assert manifest["gliner_multi_pii"].revision == (
        "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d"
    )
    assert manifest["gliner_multi_pii"].license == "Apache-2.0"
    assert all(len(spec.weight_sha256) == 64 for spec in manifest.values())


def test_uncalibrated_defaults_redact_every_model_candidate():
    config = models.DetectorConfig()

    assert config.calibrated is False
    assert config.candidate_threshold == 0.0
    assert config.redaction_threshold == 0.0


def test_thresholds_are_configurable_but_not_claimed_calibrated(monkeypatch):
    monkeypatch.setenv("PHI_CANDIDATE_THRESHOLD", "0.2")
    monkeypatch.setenv("PHI_REDACTION_THRESHOLD", "0.8")
    monkeypatch.delenv("PHI_THRESHOLDS_CALIBRATED", raising=False)

    config = models.detector_config_from_env()

    assert config.candidate_threshold == 0.2
    assert config.redaction_threshold == 0.8
    assert config.calibrated is False


def test_invalid_threshold_order_fails_closed(monkeypatch):
    monkeypatch.setenv("PHI_CANDIDATE_THRESHOLD", "0.9")
    monkeypatch.setenv("PHI_REDACTION_THRESHOLD", "0.2")

    with pytest.raises(models.LocalModelError) as exc:
        models.detector_config_from_env()

    assert exc.value.error_code == "invalid_model_configuration"


def test_zero_chunk_overlap_is_rejected_so_boundaries_stay_covered(monkeypatch):
    monkeypatch.setenv("PHI_TEXT_CHUNK_OVERLAP", "0")

    with pytest.raises(models.LocalModelError) as exc:
        models.detector_config_from_env()

    assert exc.value.error_code == "invalid_model_configuration"


def test_non_positive_inference_budget_is_rejected(monkeypatch):
    monkeypatch.setenv("PHI_MODEL_TIMEOUT_SECONDS", "0")

    with pytest.raises(models.LocalModelError) as exc:
        models.detector_config_from_env()

    assert exc.value.error_code == "invalid_model_configuration"


def test_unknown_model_mode_fails_closed(monkeypatch):
    monkeypatch.setenv(models.MODEL_MODE_ENV_VAR, "online")

    with pytest.raises(models.LocalModelError) as exc:
        models.resolve_model_mode()

    assert exc.value.error_code == "invalid_model_configuration"


def test_default_model_mode_is_offline(monkeypatch):
    monkeypatch.delenv(models.MODEL_MODE_ENV_VAR, raising=False)

    assert models.resolve_model_mode() == models.MODE_OFFLINE
    assert models.local_models_enabled() is True


# --------------------------------------------------------------------------
# Chunking and deterministic overlap resolution
# --------------------------------------------------------------------------


def test_overlapping_chunks_cover_boundaries_without_unbounded_input():
    chunks = models.overlapping_chunks("abcdefghij", chunk_size=5, overlap=1)

    assert chunks == [(0, "abcde"), (4, "efghi"), (8, "ij")]


def test_overlapping_chunks_reject_invalid_windows():
    with pytest.raises(models.LocalModelError) as exc:
        models.overlapping_chunks("abcdef", chunk_size=3, overlap=3)

    assert exc.value.error_code == "invalid_model_configuration"


def test_entity_straddling_a_chunk_boundary_is_still_detected():
    text = "x" * 8 + "NAME" + "y" * 8
    chunks = models.overlapping_chunks(text, chunk_size=10, overlap=6)

    assert any(
        offset <= 8 and 12 <= offset + len(chunk) for offset, chunk in chunks
    ), "no window wholly contains the boundary-straddling span"


def _entity(start, end, score, source="stanford_deidentifier", label="PATIENT"):
    return PhiEntity(
        entity_type="PERSON",
        start=start,
        end=end,
        source=source,
        score=score,
        original_label=label,
    )


def test_duplicate_span_from_two_chunks_collapses_to_highest_score():
    merged = models.merge_chunk_entities(
        [_entity(4, 9, 0.51), _entity(4, 9, 0.93)]
    )

    assert len(merged) == 1
    assert merged[0].score == 0.93


def test_partially_overlapping_duplicates_keep_the_longest_span():
    merged = models.merge_chunk_entities(
        [_entity(4, 9, 0.9), _entity(4, 12, 0.6)]
    )

    assert [(item.start, item.end) for item in merged] == [(4, 12)]


def test_merge_is_order_independent_and_deterministic():
    candidates = [
        _entity(4, 9, 0.5),
        _entity(4, 12, 0.5),
        _entity(20, 24, 0.7),
        _entity(4, 9, 0.8),
    ]

    forward = models.merge_chunk_entities(candidates)
    backward = models.merge_chunk_entities(list(reversed(candidates)))

    assert forward == backward
    assert [(item.start, item.end) for item in forward] == [(4, 12), (20, 24)]


def test_merge_preserves_agreement_between_different_detectors():
    merged = models.merge_chunk_entities(
        [
            _entity(4, 9, 0.9, source="stanford_deidentifier"),
            _entity(4, 9, 0.8, source="gliner_multi_pii", label="person"),
        ]
    )

    assert len(merged) == 2
    assert {item.source for item in merged} == {
        "stanford_deidentifier",
        "gliner_multi_pii",
    }


# --------------------------------------------------------------------------
# Label normalization and batched inference
# --------------------------------------------------------------------------


def test_stanford_detector_batches_chunks_and_normalizes_labels(monkeypatch):
    calls = []

    def fake_pipeline(chunks, batch_size):
        calls.append((list(chunks), batch_size))
        return [
            [{"entity_group": "PATIENT", "start": 1, "end": 3, "score": 0.4}]
            if len(chunk) >= 3
            else []
            for chunk in chunks
        ]

    monkeypatch.setattr(models, "load_stanford_pipeline", lambda: fake_pipeline)
    detector = models.StanfordClinicalDetector(
        models.DetectorConfig(
            candidate_threshold=0.2,
            redaction_threshold=0.8,
            chunk_size=5,
            chunk_overlap=1,
            batch_size=2,
        )
    )

    entities = detector.detect("abcdefghij")

    assert calls == [(["abcde", "efghi"], 2), (["ij"], 2)]
    assert [(entity.start, entity.end) for entity in entities] == [(1, 3), (5, 7)]
    assert {entity.entity_type for entity in entities} == {"PERSON"}
    assert all(entity.source == "stanford_deidentifier" for entity in entities)
    assert all(entity.score == 0.4 for entity in entities)


def test_stanford_labels_map_onto_the_internal_taxonomy():
    assert models.STANFORD_LABELS["HCW"] == "PERSON"
    assert models.STANFORD_LABELS["HOSPITAL"] == "FACILITY"
    assert models.STANFORD_LABELS["ID"] == "IDENTIFIER"
    assert models.STANFORD_LABELS["DATE"] == "DATE_TIME"
    assert models.STANFORD_LABELS["PHONE"] == "PHONE_NUMBER"


def test_gliner_labels_map_onto_the_internal_taxonomy():
    assert models.GLINER_LABELS["mobile phone number"] == "PHONE_NUMBER"
    assert models.GLINER_LABELS["social security number"] == "US_SSN"
    assert models.GLINER_LABELS["health insurance id number"] == "HEALTH_PLAN_ID"
    assert models.GLINER_LABELS["date of birth"] == "DATE_TIME"
    assert set(models.GLINER_REQUESTED_LABELS) == set(models.GLINER_LABELS)


def test_unmapped_native_label_is_ignored_not_guessed(monkeypatch):
    monkeypatch.setattr(
        models,
        "load_stanford_pipeline",
        lambda: (lambda chunks, batch_size: [[
            {"entity_group": "NOT_A_REAL_LABEL", "start": 0, "end": 2, "score": 0.9}
        ] for _ in chunks]),
    )
    detector = models.StanfordClinicalDetector(
        models.DetectorConfig(chunk_size=100, chunk_overlap=10)
    )

    assert detector.detect("abcdef") == []


def test_gliner_detector_uses_batch_api_and_global_offsets(monkeypatch):
    class FakeGliner:
        def batch_predict_entities(self, chunks, labels, threshold, batch_size):
            assert threshold == 0.0
            assert batch_size == 4
            assert "person" in labels
            return [
                [] if index == 0 else [
                    {"label": "person", "start": 1, "end": 3, "score": 0.7}
                ]
                for index, _ in enumerate(chunks)
            ]

    monkeypatch.setattr(models, "load_gliner_model", lambda: FakeGliner())
    detector = models.GlinerPiiDetector(
        models.DetectorConfig(chunk_size=6, chunk_overlap=2, batch_size=4)
    )

    entities = detector.detect("abcdefghij")

    assert len(entities) == 1
    assert (entities[0].start, entities[0].end) == (5, 7)
    assert entities[0].entity_type == "PERSON"
    assert entities[0].source == "gliner_multi_pii"


def test_gliner_falls_back_to_single_prediction_api(monkeypatch):
    class LegacyGliner:
        def predict_entities(self, text, labels, threshold):
            return [{"label": "email", "start": 0, "end": 2, "score": 0.6}]

    monkeypatch.setattr(models, "load_gliner_model", lambda: LegacyGliner())
    detector = models.GlinerPiiDetector(
        models.DetectorConfig(chunk_size=100, chunk_overlap=10)
    )

    entities = detector.detect("abcdef")

    assert [entity.entity_type for entity in entities] == ["EMAIL_ADDRESS"]


def test_empty_text_never_invokes_a_model(monkeypatch):
    def explode():
        raise AssertionError("model must not be loaded for empty input")

    monkeypatch.setattr(models, "load_stanford_pipeline", explode)
    detector = models.StanfordClinicalDetector(models.DetectorConfig())

    assert detector.detect("") == []


# --------------------------------------------------------------------------
# Fail-closed behaviour
# --------------------------------------------------------------------------


def test_checksum_mismatch_blocks_model_loading(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"not-the-approved-model")
    spec = models.ModelSpec(
        name="synthetic",
        repo_id="synthetic/model",
        revision="a" * 40,
        license="MIT",
        weight_file="weights.bin",
        weight_sha256="0" * 64,
    )

    with pytest.raises(models.LocalModelError) as exc:
        models._verify_weight(str(tmp_path), spec)

    assert exc.value.error_code == "model_checksum_mismatch"


def test_matching_checksum_accepts_the_weight(tmp_path):
    import hashlib

    payload = b"approved-synthetic-weights"
    (tmp_path / "weights.bin").write_bytes(payload)
    spec = models.ModelSpec(
        name="synthetic",
        repo_id="synthetic/model",
        revision="a" * 40,
        license="MIT",
        weight_file="weights.bin",
        weight_sha256=hashlib.sha256(payload).hexdigest(),
    )

    models._verify_weight(str(tmp_path), spec)


def test_missing_weight_file_fails_closed(tmp_path):
    spec = models.ModelSpec(
        name="synthetic",
        repo_id="synthetic/model",
        revision="a" * 40,
        license="MIT",
        weight_file="absent.bin",
        weight_sha256="0" * 64,
    )

    with pytest.raises(models.LocalModelError) as exc:
        models._verify_weight(str(tmp_path), spec)

    assert exc.value.error_code == "model_files_unavailable"


def test_snapshot_resolution_is_offline_only_and_verifies_checksum(monkeypatch):
    import huggingface_hub

    recorded = {}

    def fake_snapshot_download(repo_id, revision, local_files_only):
        recorded.update(
            repo_id=repo_id, revision=revision, local_files_only=local_files_only
        )
        return "snapshot-dir"

    verified = {}
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", fake_snapshot_download, raising=False
    )
    monkeypatch.setattr(
        models,
        "_verify_weight",
        lambda path, spec: verified.update(path=path, spec=spec.name),
    )
    monkeypatch.setenv(models.MODEL_MODE_ENV_VAR, models.MODE_OFFLINE)
    spec = models.load_model_manifest()["stanford_deidentifier"]

    assert models._offline_snapshot(spec) == "snapshot-dir"
    assert recorded["local_files_only"] is True
    assert recorded["revision"] == spec.revision
    assert verified == {"path": "snapshot-dir", "spec": "stanford_deidentifier"}
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_snapshot_resolution_is_blocked_in_legacy_test_mode(monkeypatch):
    monkeypatch.setenv(models.MODEL_MODE_ENV_VAR, models.MODE_LEGACY_TEST)
    spec = models.load_model_manifest()["gliner_multi_pii"]

    with pytest.raises(models.LocalModelError) as exc:
        models._offline_snapshot(spec)

    assert exc.value.error_code == "local_models_disabled"


def test_unavailable_snapshot_fails_closed(monkeypatch):
    import huggingface_hub

    def fake_snapshot_download(**_kwargs):
        raise OSError("not cached locally")

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", fake_snapshot_download, raising=False
    )
    monkeypatch.setenv(models.MODEL_MODE_ENV_VAR, models.MODE_OFFLINE)
    spec = models.load_model_manifest()["stanford_deidentifier"]

    with pytest.raises(models.LocalModelError) as exc:
        models._offline_snapshot(spec)

    assert exc.value.error_code == "model_files_unavailable"


def test_inference_failure_never_returns_partial_results(monkeypatch):
    def failing_pipeline(chunks, batch_size):
        raise RuntimeError("tensor allocation failed")

    monkeypatch.setattr(models, "load_stanford_pipeline", lambda: failing_pipeline)
    detector = models.StanfordClinicalDetector(models.DetectorConfig())

    with pytest.raises(models.LocalModelError) as exc:
        detector.detect("Patient text")

    assert exc.value.error_code == "stanford_inference_failed"
    assert exc.value.status_code == 500


def test_gliner_inference_failure_fails_closed(monkeypatch):
    class FailingGliner:
        def batch_predict_entities(self, *_args, **_kwargs):
            raise RuntimeError("backend crashed")

    monkeypatch.setattr(models, "load_gliner_model", lambda: FailingGliner())
    detector = models.GlinerPiiDetector(models.DetectorConfig())

    with pytest.raises(models.LocalModelError) as exc:
        detector.detect("Patient text")

    assert exc.value.error_code == "gliner_inference_failed"


def test_exhausted_time_budget_blocks_instead_of_returning_text(monkeypatch):
    clock = iter([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    monkeypatch.setattr(models.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        models,
        "load_stanford_pipeline",
        lambda: (lambda chunks, batch_size: [[] for _ in chunks]),
    )
    detector = models.StanfordClinicalDetector(
        models.DetectorConfig(
            chunk_size=4, chunk_overlap=1, batch_size=1, inference_budget_seconds=1.0
        )
    )

    with pytest.raises(models.LocalModelError) as exc:
        detector.detect("abcdefghijkl")

    assert exc.value.error_code == "model_inference_timeout"


def test_prediction_count_mismatch_fails_closed(monkeypatch):
    monkeypatch.setattr(
        models,
        "load_stanford_pipeline",
        lambda: (lambda chunks, batch_size: [[], []]),
    )
    detector = models.StanfordClinicalDetector(
        models.DetectorConfig(chunk_size=4, chunk_overlap=1, batch_size=8)
    )

    with pytest.raises(models.LocalModelError) as exc:
        detector.detect("abcdefghijkl")

    assert exc.value.error_code == "model_output_malformed"


def test_out_of_range_offsets_fail_closed_instead_of_being_dropped(monkeypatch):
    monkeypatch.setattr(
        models,
        "load_stanford_pipeline",
        lambda: (lambda chunks, batch_size: [
            [{"entity_group": "PATIENT", "start": 0, "end": 999, "score": 0.9}]
            for _ in chunks
        ]),
    )
    detector = models.StanfordClinicalDetector(
        models.DetectorConfig(chunk_size=100, chunk_overlap=10)
    )

    with pytest.raises(models.LocalModelError) as exc:
        detector.detect("abcdef")

    assert exc.value.error_code == "model_output_malformed"


def test_missing_offsets_fail_closed(monkeypatch):
    monkeypatch.setattr(
        models,
        "load_stanford_pipeline",
        lambda: (lambda chunks, batch_size: [
            [{"entity_group": "PATIENT", "score": 0.9}] for _ in chunks
        ]),
    )
    detector = models.StanfordClinicalDetector(
        models.DetectorConfig(chunk_size=100, chunk_overlap=10)
    )

    with pytest.raises(models.LocalModelError) as exc:
        detector.detect("abcdef")

    assert exc.value.error_code == "model_output_malformed"


def test_candidates_below_the_redaction_threshold_are_still_redacted(monkeypatch):
    monkeypatch.setattr(
        models,
        "load_stanford_pipeline",
        lambda: (lambda chunks, batch_size: [
            [{"entity_group": "PATIENT", "start": 0, "end": 3, "score": 0.25}]
            for _ in chunks
        ]),
    )
    detector = models.StanfordClinicalDetector(
        models.DetectorConfig(
            candidate_threshold=0.1,
            redaction_threshold=0.9,
            chunk_size=100,
            chunk_overlap=10,
        )
    )

    entities = detector.detect("abcdef")

    assert len(entities) == 1, "a low-confidence candidate must not be dropped"
    assert entities[0].score == 0.25


def test_sub_candidate_scores_are_excluded(monkeypatch):
    monkeypatch.setattr(
        models,
        "load_stanford_pipeline",
        lambda: (lambda chunks, batch_size: [
            [{"entity_group": "PATIENT", "start": 0, "end": 3, "score": 0.05}]
            for _ in chunks
        ]),
    )
    detector = models.StanfordClinicalDetector(
        models.DetectorConfig(
            candidate_threshold=0.5,
            redaction_threshold=0.5,
            chunk_size=100,
            chunk_overlap=10,
        )
    )

    assert detector.detect("abcdef") == []


# --------------------------------------------------------------------------
# Disclosure hygiene
# --------------------------------------------------------------------------


def test_manifest_contains_no_local_paths_or_credentials():
    raw = models.MANIFEST_PATH.read_text(encoding="utf-8")
    parsed = json.loads(raw)

    # Credential-shaped *keys*, not a bare substring: "tokenizer_backbone" is
    # legitimate vocabulary and a substring check on "token" flags it.
    credential_keys = {
        "token",
        "auth_token",
        "access_token",
        "hf_token",
        "api_key",
        "apikey",
        "password",
        "secret",
        "credential",
    }
    for entry in parsed.values():
        assert not credential_keys & {key.casefold() for key in entry}

    # No bearer-style literal anywhere, and no absolute machine path.
    assert "hf_" not in raw.replace("hf_token", "").casefold()
    assert "c:\\" not in raw.casefold()
    assert "/users/" not in raw.casefold()
    assert "/home/" not in raw.casefold()
    assert all("revision" in entry for entry in parsed.values())


def test_manifest_declares_the_gliner_tokenizer_backbone():
    """GLiNER resolves a tokenizer by repository name at construction time.

    That is a second supply-chain input, so it is pinned and checksummed like
    any other rather than being picked up from whatever happens to be cached.
    """
    manifest = models.load_model_manifest()

    assert "mdeberta_backbone" in manifest
    backbone = manifest["mdeberta_backbone"]
    assert backbone.repo_id == "microsoft/mdeberta-v3-base"
    assert len(backbone.revision) == 40
    assert len(backbone.weight_sha256) == 64
    assert backbone.role == "tokenizer_backbone"
    assert backbone.required_by == "gliner_multi_pii"
    # Only the tokenizer files, not the backbone's own 1 GB of weights.
    assert "spm.model" in backbone.allow_patterns
    assert "pytorch_model.bin" not in backbone.allow_patterns


def test_backbones_are_excluded_from_the_detector_set():
    assert set(models.detector_specs()) == {
        "stanford_deidentifier",
        "gliner_multi_pii",
    }
    assert [s.repo_id for s in models.backbone_specs_for("gliner_multi_pii")] == [
        "microsoft/mdeberta-v3-base"
    ]


def test_locked_thresholds_are_calibrated_and_non_zero():
    """Zero is the uncalibrated default and must not survive as production."""
    locked = models.load_locked_thresholds()

    assert locked, "no calibrated thresholds are locked"
    for name in ("stanford_deidentifier", "gliner_multi_pii"):
        assert locked[name]["candidate_threshold"] > 0.0
        config = models.calibrated_config_for(name)
        assert config.calibrated is True
        assert config.candidate_threshold > 0.0


def test_environment_override_beats_the_locked_calibration(monkeypatch):
    monkeypatch.setenv("PHI_CANDIDATE_THRESHOLD", "0.42")
    monkeypatch.setenv("PHI_REDACTION_THRESHOLD", "0.42")

    config = models.calibrated_config_for("gliner_multi_pii")

    assert config.candidate_threshold == 0.42
    # An operator override is not the committed calibration.
    assert config.calibrated is False


def test_missing_calibration_falls_back_to_redacting_everything(monkeypatch):
    monkeypatch.setattr(models, "load_locked_thresholds", lambda: {})
    monkeypatch.delenv("PHI_CANDIDATE_THRESHOLD", raising=False)
    monkeypatch.delenv("PHI_REDACTION_THRESHOLD", raising=False)

    config = models.calibrated_config_for("gliner_multi_pii")

    # Absent calibration over-redacts rather than under-redacts.
    assert config.candidate_threshold == 0.0
    assert config.calibrated is False


def test_model_errors_never_carry_detected_values(monkeypatch):
    secret = "Jane Q Patient"

    def failing_pipeline(chunks, batch_size):
        raise RuntimeError(f"crashed while scoring {secret}")

    monkeypatch.setattr(models, "load_stanford_pipeline", lambda: failing_pipeline)
    detector = models.StanfordClinicalDetector(models.DetectorConfig())

    with pytest.raises(models.LocalModelError) as exc:
        detector.detect(f"Patient {secret} was seen today.")

    assert secret not in str(exc.value)
    assert str(exc.value) == "stanford_inference_failed"


def test_entities_carry_offsets_and_categories_not_text(monkeypatch):
    monkeypatch.setattr(
        models,
        "load_stanford_pipeline",
        lambda: (lambda chunks, batch_size: [
            [{"entity_group": "PATIENT", "start": 8, "end": 12, "score": 0.9}]
            for _ in chunks
        ]),
    )
    detector = models.StanfordClinicalDetector(
        models.DetectorConfig(chunk_size=200, chunk_overlap=20)
    )

    entities = detector.detect("Patient Jane was seen today.")

    assert len(entities) == 1
    assert set(vars(entities[0])) == {
        "entity_type",
        "start",
        "end",
        "source",
        "score",
        "original_label",
    }
    assert "Jane" not in repr(entities[0])
