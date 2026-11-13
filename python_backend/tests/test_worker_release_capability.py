"""Runtime availability is required in addition to sanitized bytes."""

import json
from types import SimpleNamespace

import pytest
from services import detector_capability as capability
from services import local_model_detectors as models
from services import model_client, model_lifecycle
from services.privacy_contracts import SanitizedArtifact, issue_release


def artifact():
    return SanitizedArtifact(
        b"synthetic sanitized bytes", "text/plain", "safe.txt", ("test",)
    )


@pytest.mark.parametrize("phase", ["uninitialized", "preparing", "failed"])
def test_offline_configuration_without_completed_warmup_cannot_release(
    monkeypatch, phase
):
    monkeypatch.setenv(models.MODEL_MODE_ENV_VAR, models.MODE_OFFLINE)
    monkeypatch.setattr(
        model_lifecycle, "_STATE", model_lifecycle.LifecycleState(state=phase)
    )
    monkeypatch.setattr(
        model_client, "current_client", lambda: SimpleNamespace(is_ready=lambda: True)
    )
    decision = issue_release(artifact())
    assert not decision.releasable
    assert decision.artifact_sha256 is None
    assert capability.REASON_UNAVAILABLE in decision.reason_codes


@pytest.mark.parametrize("client", [None, SimpleNamespace(is_ready=lambda: False)])
def test_missing_or_replacing_worker_invalidates_a_previously_ready_runtime(
    monkeypatch, client
):
    monkeypatch.setenv(models.MODEL_MODE_ENV_VAR, models.MODE_OFFLINE)
    monkeypatch.setattr(
        model_lifecycle, "_STATE", model_lifecycle.LifecycleState(state="ready")
    )
    monkeypatch.setattr(model_client, "current_client", lambda: client)
    assert not issue_release(artifact()).releasable


def test_a_ready_production_runtime_can_bind_verified_bytes(monkeypatch):
    monkeypatch.setenv(models.MODEL_MODE_ENV_VAR, models.MODE_OFFLINE)
    monkeypatch.setattr(
        model_lifecycle, "_STATE", model_lifecycle.LifecycleState(state="ready")
    )
    monkeypatch.setattr(
        model_client, "current_client", lambda: SimpleNamespace(is_ready=lambda: True)
    )
    sanitized = artifact()
    decision = issue_release(sanitized)
    assert decision.releasable
    assert decision.artifact_sha256 == sanitized.sha256


def test_acknowledged_test_detectors_still_cannot_authorize_release():
    decision = issue_release(artifact())
    assert not decision.releasable
    assert capability.REASON_NON_PRODUCTION in decision.reason_codes


def test_index_and_preview_hold_without_exposing_content():
    from services.release_gate import sanitize_for_index, sanitized_preview

    index = sanitize_for_index({"text": "Patient Private Synthetic MRN 123"})
    preview = sanitized_preview(b"unvalidated pixels", "image.png", "image/png")
    assert not index.cleared and not index.fields
    assert not preview.released and preview.content is None
    assert capability.REASON_NON_PRODUCTION in index.reason_codes
    assert capability.REASON_NON_PRODUCTION in preview.reason_codes
    assert "Private" not in json.dumps(index.safe_summary())


def test_raster_endpoint_response_does_not_stream_without_production_capability():
    import main

    # A completed pixel verifier alone is insufficient to authorize release.
    response = main._raster_redaction_response(
        SimpleNamespace(released=True), "private.png", "redacted"
    )
    assert response.status_code == 422
    assert b"non_production_detector_mode" in response.body
    assert b"private.png" not in response.body


def test_ordinary_ingestion_reports_the_nonproduction_hold():
    from services.ingestion import route_for_ingestion

    result = route_for_ingestion(
        "safe.txt",
        "text/plain",
        b"Routine observation.",
        profile="strict",
        text_content=b"Routine observation.",
    )
    decision = result["release_decision"]
    assert not decision["releasable"] and decision["artifact_sha256"] is None
    assert capability.REASON_NON_PRODUCTION in decision["reason_codes"]
    assert set(result["downstream"].values()) == {"blocked"}


def test_spacy_load_failure_is_not_successful_warmup(monkeypatch):
    from services import ner_phi_detector

    def unavailable(*args, **kwargs):
        raise RuntimeError("private path and note must not escape")

    monkeypatch.setattr(ner_phi_detector, "load_spacy_pipeline", unavailable)
    monkeypatch.setenv(model_lifecycle.PREWARM_ENV_VAR, "0")
    monkeypatch.setattr(model_lifecycle, "_STATE", model_lifecycle.LifecycleState())
    state = model_lifecycle.prewarm()
    assert not state.is_ready()
    assert state.error_code == "spacy_prewarm_failed"
    assert "private" not in json.dumps(state.to_public_dict())
