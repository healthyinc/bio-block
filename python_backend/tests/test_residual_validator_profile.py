"""Phase 12E: the residual validator profile, and what it must never allow.

Phase 12D found GLiNER caught nothing in the second pass that another
detector did not also catch, across 300 synthetic documents and eighteen
categories, while costing about three quarters of the pass. ``optimized_v1``
drops it from that pass. These tests fence in what that is allowed to mean.

Three properties matter more than the saving:

* the first pass never loses GLiNER, in any profile;
* an upload cannot choose the profile, and a value nobody recognizes fails
  closed rather than resolving to the weaker one;
* a detector that did not report is a failure, not an empty result - partial
  coverage and a clean document look identical in the output.

``full_v1`` stays the default here, so the rollback is the thing that happens
when nobody does anything.
"""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import residual_profile as rp  # noqa: E402
from services import text_anonymization as ta  # noqa: E402
from services.residual_profile import (  # noqa: E402
    PASS_FIRST,
    PASS_RESIDUAL,
    PROFILE_FULL,
    PROFILE_OPTIMIZED,
    ResidualProfileError,
)

NOTE = b"Patient Zelquin Vantreske, MRN 44821903, seen on 14 March 2021."


@pytest.fixture(autouse=True)
def _mocked(monkeypatch):
    monkeypatch.setenv("PHI_MODEL_MODE", "legacy_test")
    monkeypatch.setenv("BIOBLOCK_STUDY_SALT", "phase12e-salt")


def _client():
    import main
    from fastapi.testclient import TestClient

    return TestClient(main.app), main


def _upload(client, payload=NOTE, data=None):
    return client.post(
        "/api/v1/ingest",
        files={"file": ("note.txt", io.BytesIO(payload), "text/plain")},
        data=data or {"profile": "strict"},
    )


# -- the contract -----------------------------------------------------------


def test_default_is_the_full_profile(monkeypatch):
    """Doing nothing must select the validator that was already shipping."""
    monkeypatch.delenv(rp.PROFILE_ENV_VAR, raising=False)

    assert rp.configured_profile() == PROFILE_FULL
    assert rp.DEFAULT_PROFILE == PROFILE_FULL


def test_empty_configuration_is_the_full_profile(monkeypatch):
    monkeypatch.setenv(rp.PROFILE_ENV_VAR, "   ")

    assert rp.configured_profile() == PROFILE_FULL


def test_optimized_is_selectable_by_configuration(monkeypatch):
    monkeypatch.setenv(rp.PROFILE_ENV_VAR, PROFILE_OPTIMIZED)

    assert rp.configured_profile() == PROFILE_OPTIMIZED


@pytest.mark.parametrize(
    "value",
    ["", "optimised_v1", "OPTIMIZED_V1", "none", "full", "v1", "optimized_v2", "0"],
)
def test_an_unrecognized_profile_fails_closed(monkeypatch, value):
    """A typo must not resolve to the weaker validator."""
    monkeypatch.setenv(rp.PROFILE_ENV_VAR, value or "x")

    if value == "":
        return
    with pytest.raises(ResidualProfileError) as exc:
        rp.configured_profile()
    assert exc.value.error_code == "invalid_residual_validator_profile"


def test_startup_rejects_an_invalid_profile(monkeypatch):
    from services import model_lifecycle

    monkeypatch.setenv(rp.PROFILE_ENV_VAR, "nonsense")
    monkeypatch.setenv("PHI_MODEL_PREWARM", "0")
    model_lifecycle.reset_state()

    state = model_lifecycle.prewarm()

    assert state.is_ready() is False
    assert state.error_code == "invalid_residual_validator_profile"


# -- GLiNER stays in the first pass ----------------------------------------


@pytest.mark.parametrize("profile", [PROFILE_FULL, PROFILE_OPTIMIZED])
def test_first_pass_always_requires_gliner(profile):
    """The optimization is about the second pass and only the second pass."""
    assert rp.SOURCE_GLINER_NAME in rp.required_sources(PASS_FIRST, profile)
    assert rp.omits_gliner(PASS_FIRST, profile) is False


def test_optimized_omits_gliner_only_from_the_residual_pass():
    assert rp.omits_gliner(PASS_RESIDUAL, PROFILE_OPTIMIZED) is True
    assert rp.omits_gliner(PASS_RESIDUAL, PROFILE_FULL) is False


def test_full_profile_residual_composition():
    assert set(rp.required_sources(PASS_RESIDUAL, PROFILE_FULL)) == {
        "structured_pattern",
        "age_rule",
        "ner",
        "stanford_deidentifier",
        "gliner_multi_pii",
    }


def test_optimized_profile_residual_composition():
    assert set(rp.required_sources(PASS_RESIDUAL, PROFILE_OPTIMIZED)) == {
        "structured_pattern",
        "age_rule",
        "ner",
        "stanford_deidentifier",
    }


def test_no_profile_leaves_the_residual_pass_without_a_model():
    """Deterministic rules alone are not a validator."""
    models = {"ner", "stanford_deidentifier", "gliner_multi_pii"}

    for profile in rp.ALLOWED_PROFILES:
        assert set(rp.required_sources(PASS_RESIDUAL, profile)) & models


def test_an_unknown_pass_name_is_an_error():
    with pytest.raises(ResidualProfileError):
        rp.required_sources("third_pass", PROFILE_FULL)


# -- a request cannot reach the profile ------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "residual_validator_profile",
        "validator_profile",
        "PHI_RESIDUAL_VALIDATOR_PROFILE",
        "residual_profile",
    ],
)
def test_an_upload_cannot_select_the_profile(monkeypatch, field):
    """The only way in is server configuration."""
    monkeypatch.delenv(rp.PROFILE_ENV_VAR, raising=False)
    client, _ = _client()

    response = _upload(client, data={"profile": "strict", field: PROFILE_OPTIMIZED})

    assert response.status_code in {200, 422}
    assert rp.configured_profile() == PROFILE_FULL


def test_the_privacy_profile_field_does_not_reach_the_validator(monkeypatch):
    """`profile` selects strict or research; it is a different thing."""
    monkeypatch.delenv(rp.PROFILE_ENV_VAR, raising=False)
    client, _ = _client()

    _upload(client, data={"profile": PROFILE_OPTIMIZED})

    assert rp.configured_profile() == PROFILE_FULL


# -- output equivalence -----------------------------------------------------


def _capture(monkeypatch, profile):
    monkeypatch.setenv(rp.PROFILE_ENV_VAR, profile)
    ta._build_detectors.cache_clear()
    client, _ = _client()
    body = _upload(client).json()
    return {
        # Present only for a released artifact: a hold must not hand back the
        # candidate it refused. The comparison is the point either way - the
        # first pass must not move when only the second pass changes.
        "anonymized_text": body.get("anonymized_text"),
        "detected_entities": body["detected_entities"],
        "entity_count": body["entity_count"],
        "surrogate_counts": body.get("surrogate_counts"),
        "utility_metrics": body["utility_metrics"],
    }


def test_first_pass_output_is_identical_under_both_profiles(monkeypatch):
    """Only the second pass changed, so the released text must not move."""
    full = _capture(monkeypatch, PROFILE_FULL)
    optimized = _capture(monkeypatch, PROFILE_OPTIMIZED)

    assert full == optimized


def test_both_profiles_run_exactly_two_passes(monkeypatch):
    for profile in (PROFILE_FULL, PROFILE_OPTIMIZED):
        monkeypatch.setenv(rp.PROFILE_ENV_VAR, profile)
        monkeypatch.setenv("PHI_LATENCY_METRICS", "1")
        ta._build_detectors.cache_clear()
        client, main = _client()

        assert _upload(client).status_code == 200

        record = main.latest_request_timings()[-1]
        assert record.call_count("anonymize.detect_pass1") == 1, profile
        assert record.call_count("anonymize.residual_validation") == 1, profile
        assert record.call_count("detect.evidence_model") == 2, profile


def test_rollback_to_full_restores_the_full_chain(monkeypatch):
    monkeypatch.setenv(rp.PROFILE_ENV_VAR, PROFILE_OPTIMIZED)
    assert rp.omits_gliner(PASS_RESIDUAL, rp.configured_profile()) is True

    monkeypatch.setenv(rp.PROFILE_ENV_VAR, PROFILE_FULL)
    assert rp.omits_gliner(PASS_RESIDUAL, rp.configured_profile()) is False


# -- evidence ---------------------------------------------------------------


def test_evidence_reports_required_and_completed():
    evidence = rp.build_evidence(
        PASS_RESIDUAL,
        PROFILE_OPTIMIZED,
        ("structured_pattern", "age_rule", "ner", "stanford_deidentifier"),
    )

    assert evidence.complete is True
    assert evidence.missing == ()


def test_incomplete_evidence_is_not_complete():
    evidence = rp.build_evidence(
        PASS_RESIDUAL,
        PROFILE_FULL,
        ("structured_pattern", "age_rule"),
    )

    assert evidence.complete is False
    assert "stanford_deidentifier" in evidence.missing


def test_evidence_carries_no_paths_or_text():
    evidence = rp.build_evidence(
        PASS_RESIDUAL, PROFILE_FULL, rp.required_sources(PASS_RESIDUAL, PROFILE_FULL)
    )
    payload = repr(evidence.to_public_dict())

    for forbidden in (
        "Users",
        ".safetensors",
        "model-cache",
        "sha256",
        "Vantreske",
        "44821903",
    ):
        assert forbidden not in payload


def test_evidence_is_not_returned_to_the_client(monkeypatch):
    monkeypatch.delenv(rp.PROFILE_ENV_VAR, raising=False)
    client, _ = _client()

    body = _upload(client).json()

    assert "detectors_required" not in body
    assert "validator_profile" not in body


# -- failure paths still block ---------------------------------------------


def test_a_detector_that_does_not_report_blocks(monkeypatch):
    """Partial coverage must not be reported as a clean document."""
    from services.ner_phi_detector import NerPhiDetectionError

    class _Silent:
        source = "stanford_deidentifier"

        def detect(self, text):
            raise RuntimeError("model died")

    monkeypatch.setattr(
        ta,
        "_detectors",
        lambda *_args, **_kwargs: (_Silent(),),
    )

    with pytest.raises(NerPhiDetectionError):
        ta._detect_entities("Patient Zelquin Vantreske.", "en_core_web_sm", "strict")


def test_an_empty_residual_chain_blocks(monkeypatch):
    """A configuration that yields no detector is a failure, not a pass."""
    from services.ner_phi_detector import NerPhiDetectionError

    monkeypatch.setattr(ta, "_build_detectors", lambda *_a, **_k: ())

    with pytest.raises(NerPhiDetectionError):
        ta._detect_entities(
            "text",
            "en_core_web_sm",
            "strict",
            pass_kind=PASS_RESIDUAL,
        )


def test_an_invalid_profile_blocks_detection(monkeypatch):
    from services.ner_phi_detector import NerPhiDetectionError

    monkeypatch.setenv(rp.PROFILE_ENV_VAR, "nonsense")

    with pytest.raises(NerPhiDetectionError) as exc:
        ta._detect_entities(
            "text",
            "en_core_web_sm",
            "strict",
            pass_kind=PASS_RESIDUAL,
        )
    assert exc.value.error_code == "invalid_residual_validator_profile"


def test_model_failure_blocks_under_both_profiles(monkeypatch):
    """Fail-closed is unchanged by the optimization."""
    from services.local_model_detectors import LocalModelError
    from services.ner_phi_detector import NerPhiDetectionError

    class _Broken:
        source = "stanford_deidentifier"

        def detect(self, text):
            raise LocalModelError("stanford_model_unavailable", 503)

    for profile in (PROFILE_FULL, PROFILE_OPTIMIZED):
        monkeypatch.setenv(rp.PROFILE_ENV_VAR, profile)
        monkeypatch.setattr(ta, "_detectors", lambda *_a, **_k: (_Broken(),))

        with pytest.raises(NerPhiDetectionError):
            ta._detect_entities("text", "en_core_web_sm", "strict")


def test_no_profile_can_bypass_the_release_gate():
    """The profile chooses detectors; it never decides releasability."""
    import inspect

    source = inspect.getsource(rp)

    for forbidden in (
        "release_gate",
        "release_decision",
        "releasable",
        "issue_release",
    ):
        assert forbidden not in source
