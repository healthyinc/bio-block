"""Phase 12A: the instrumentation must measure without changing anything.

Two obligations are tested here. The first is that timing records cannot
carry PHI - a record that could hold a name or a filename would be a new way
to leak one, and it would be an easy one to miss because nobody reads timing
output looking for identifiers. The second is that switching collection on
changes no anonymized output, no release decision and no status code; an
instrument that alters what it measures is worse than no instrument.

The detector-pass count is pinned here too. Production runs two passes - the
first over the upload, the second over the redacted output - and the
evaluation harness's third, metrics-only pass must never reach the API.
"""

import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import fixtures  # noqa: E402
from benchmarks.latency_benchmark import (  # noqa: E402
    OUTCOME_FAILED,
    OUTCOME_MANUAL_REVIEW,
    OUTCOME_REJECTED,
    OUTCOME_SUCCESS,
    OUTCOME_TIMEOUT,
    _assemble,
    classify,
    percentile,
    summarize,
)
from services import latency_metrics  # noqa: E402

SYNTHETIC_NOTE = (
    b"Patient Zelquin Vantreske, MRN 44821903, was seen by "
    b"Dr Marbrint Morrowgale on 14 March 2021. Metformin 500 mg nightly."
)


@pytest.fixture
def collecting(monkeypatch):
    monkeypatch.setenv(latency_metrics.ENABLE_ENV_VAR, "1")
    yield
    latency_metrics.finish_request()


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("PHI_MODEL_MODE", "legacy_test")
    monkeypatch.setenv("BIOBLOCK_STUDY_SALT", "latency-test-salt")
    import main
    from fastapi.testclient import TestClient

    return TestClient(main.app), main


def _upload(client, payload=SYNTHETIC_NOTE, name="note.txt"):
    return client.post(
        "/api/v1/ingest",
        files={"file": (name, io.BytesIO(payload), "text/plain")},
        data={"profile": "strict"},
    )


# -- collection is off unless asked for -------------------------------------


def test_collection_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(latency_metrics.ENABLE_ENV_VAR, raising=False)

    assert latency_metrics.metrics_enabled() is False
    assert latency_metrics.start_request() is None
    with latency_metrics.stage("anything"):
        pass
    assert latency_metrics.current_request() is None


def test_stage_is_a_no_op_when_nothing_collects(monkeypatch):
    """The disabled path must not build objects or raise."""
    monkeypatch.delenv(latency_metrics.ENABLE_ENV_VAR, raising=False)

    with latency_metrics.stage("a"), latency_metrics.stage("b"):
        pass


def test_collection_records_nested_depth(collecting):
    latency_metrics.start_request()
    with latency_metrics.stage("outer"):
        with latency_metrics.stage("inner"):
            pass
    record = latency_metrics.finish_request()

    depths = {s.name: s.depth for s in record.stages}
    assert depths["outer"] == 0
    assert depths["inner"] == 1
    assert all(s.duration_ns >= 0 for s in record.stages)


def test_a_failing_stage_is_still_recorded(collecting):
    """Time spent before a failure is still time the user waited."""
    latency_metrics.start_request()
    with pytest.raises(ValueError):
        with latency_metrics.stage("explodes"):
            raise ValueError("boom")
    record = latency_metrics.finish_request()

    assert record.call_count("explodes") == 1


# -- timing records must not carry PHI --------------------------------------


def test_timing_records_hold_no_document_content(collecting, api_client):
    client, main = api_client
    response = _upload(client)
    assert response.status_code == 200

    serialized = json.dumps([r.to_public_dict() for r in main.latest_request_timings()])

    for secret in (
        "Zelquin",
        "Vantreske",
        "Marbrint",
        "Morrowgale",
        "44821903",
        "Metformin",
        "14 March 2021",
    ):
        assert secret not in serialized


def test_timing_records_hold_no_filename(collecting, api_client):
    client, main = api_client
    _upload(client, name="Vantreske-discharge-summary.txt")

    serialized = json.dumps([r.to_public_dict() for r in main.latest_request_timings()])

    assert "Vantreske" not in serialized
    assert ".txt" not in serialized


def test_public_record_exposes_only_safe_fields(collecting):
    latency_metrics.start_request(modality="text", size_bucket_name="tiny")
    with latency_metrics.stage("s"):
        pass
    record = latency_metrics.finish_request().to_public_dict()

    assert set(record) == {
        "request_id",
        "modality",
        "size_bucket",
        "concurrency",
        "model_ready",
        "outcome",
        "release_category",
        "stages",
    }
    assert set(record["stages"][0]) == {"stage", "depth", "duration_ms"}


def test_size_is_reported_as_a_bucket_not_an_exact_length():
    """An exact byte count is a weak identifier for a known document."""
    assert latency_metrics.size_bucket(900) == "tiny"
    assert latency_metrics.size_bucket(10_240) == "small"
    assert latency_metrics.size_bucket(102_400) == "medium"
    assert latency_metrics.size_bucket(262_144) == "large"
    assert latency_metrics.size_bucket(10_000_000) == "oversize"


# -- how many detector passes a production request runs ---------------------


def test_production_request_runs_exactly_two_detector_passes(collecting, api_client):
    """One pass over the upload, one over the redacted output. No more.

    A third pass exists in the evaluation harness, which calls the detector
    again to score metrics. That pass must never run inside the API: it would
    add its full cost to every user request while changing no output.
    """
    client, main = api_client
    assert _upload(client).status_code == 200

    record = main.latest_request_timings()[-1]

    assert record.call_count("anonymize.detect_pass1") == 1
    assert record.call_count("anonymize.residual_validation") == 1
    # The evidence model runs once per detector chain pass, so it counts them.
    assert record.call_count("detect.evidence_model") == 2


def test_no_evaluation_only_pass_reaches_the_api(collecting, api_client):
    """The evaluation entry point must not be on the request path."""
    client, main = api_client
    calls = []

    from services import text_anonymization as ta

    original = ta._detect_entities

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    ta._detect_entities = counting
    try:
        assert _upload(client).status_code == 200
    finally:
        ta._detect_entities = original

    assert len(calls) == 2, f"expected 2 detector passes, saw {len(calls)}"


# -- instrumentation must not change behaviour ------------------------------


def _capture(client):
    response = _upload(client)
    body = response.json()
    return {
        "status": response.status_code,
        # Present only for a released artifact: a hold must not hand back the
        # candidate it refused. Either way the comparison below is what this
        # test is for - instrumentation may not change the answer.
        "anonymized_text": body.get("anonymized_text"),
        "detected_entities": body["detected_entities"],
        "residual": body["residual_phi_categories"],
        "utility": body["utility_metrics"],
        "verdict": body["utility_verdict"],
        "release": body["release_decision"],
        "review": body.get("review_required_reasons"),
        "surrogates": body.get("surrogate_counts"),
    }


def test_output_is_identical_with_and_without_instrumentation(monkeypatch, api_client):
    """The measurement must not move the thing being measured."""
    client, _ = api_client

    monkeypatch.delenv(latency_metrics.ENABLE_ENV_VAR, raising=False)
    without = _capture(client)

    monkeypatch.setenv(latency_metrics.ENABLE_ENV_VAR, "1")
    with_timing = _capture(client)
    latency_metrics.finish_request()

    assert without == with_timing


def test_failure_behaviour_is_identical_with_instrumentation(monkeypatch, api_client):
    client, _ = api_client

    monkeypatch.delenv(latency_metrics.ENABLE_ENV_VAR, raising=False)
    plain = client.post(
        "/api/v1/ingest",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        data={"profile": "strict"},
    )

    monkeypatch.setenv(latency_metrics.ENABLE_ENV_VAR, "1")
    timed = client.post(
        "/api/v1/ingest",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        data={"profile": "strict"},
    )
    latency_metrics.finish_request()

    assert plain.status_code == timed.status_code
    assert plain.json() == timed.json()


def test_timing_details_are_not_returned_to_the_client(collecting, api_client):
    """Stage timings stay in-process; they are not part of the response."""
    client, _ = api_client
    body = _upload(client).json()

    assert "stages" not in body
    assert "timings" not in body
    assert "latency_ms" not in body


# -- benchmark statistics ---------------------------------------------------


def test_percentiles_use_nearest_rank():
    values = [float(n) for n in range(1, 101)]

    assert percentile(values, 0.50) == 50.0
    assert percentile(values, 0.90) == 90.0
    assert percentile(values, 0.95) == 95.0
    assert percentile(values, 0.99) == 99.0
    assert percentile([], 0.5) == 0.0
    assert percentile([7.0], 0.99) == 7.0


def test_summary_reports_mean_min_and_max():
    stats = summarize([10.0, 20.0, 30.0])

    assert stats["count"] == 3
    assert stats["mean"] == 20.0
    assert stats["min"] == 10.0
    assert stats["max"] == 30.0


def test_outcomes_are_counted_separately():
    """A fast rejection is not a fast success and must not be pooled with one."""
    results = [
        {"outcome": OUTCOME_SUCCESS, "latency_ms": 10.0},
        {"outcome": OUTCOME_MANUAL_REVIEW, "latency_ms": 20.0},
        {"outcome": OUTCOME_REJECTED, "latency_ms": 5.0},
        {"outcome": OUTCOME_FAILED, "latency_ms": 30.0},
        {"outcome": OUTCOME_TIMEOUT, "latency_ms": 900.0},
    ]
    assembled = _assemble(results, wall_seconds=1.0)

    assert assembled["outcomes"][OUTCOME_SUCCESS] == 1
    assert assembled["outcomes"][OUTCOME_TIMEOUT] == 1
    # A timeout has no meaningful latency and must not distort the tail.
    assert assembled["latency_ms"]["count"] == 4
    assert assembled["latency_ms"]["max"] == 30.0


def test_classification_separates_release_from_manual_review():
    assert classify(200, {"release_decision": {"releasable": True}}) == OUTCOME_SUCCESS
    assert (
        classify(200, {"release_decision": {"releasable": False}})
        == OUTCOME_MANUAL_REVIEW
    )
    assert classify(422, None) == OUTCOME_MANUAL_REVIEW
    assert classify(413, None) == OUTCOME_REJECTED
    assert classify(500, None) == OUTCOME_FAILED


def test_report_carries_the_fields_the_schema_promises(collecting, api_client):
    from benchmarks.latency_benchmark import execution_mode

    mode = execution_mode()
    assert set(mode) >= {"device", "torch", "cuda_available"}
    assert mode["device"] in {"cpu", "cuda"}


# -- fixtures ---------------------------------------------------------------


def test_fixtures_are_deterministic_for_a_seed():
    first = fixtures.build_document("mixed_clinical", 4_096)
    second = fixtures.build_document("mixed_clinical", 4_096)

    assert first == second


def test_fixtures_differ_between_content_kinds():
    a = fixtures.build_document("no_phi", 2_048)
    b = fixtures.build_document("structured_phi", 2_048)

    assert a != b


def test_fixtures_respect_the_text_upload_limit():
    with pytest.raises(ValueError):
        fixtures.build_document("no_phi", fixtures.TEXT_LIMIT_BYTES + 1)


def test_every_declared_content_kind_builds():
    for kind in fixtures.CONTENT_KINDS:
        assert fixtures.build_document(kind, 1_024)


def test_fixture_documents_reach_their_size_bucket():
    for name, target in fixtures.SIZE_TARGETS:
        document = fixtures.build_document("mixed_clinical", target)
        assert len(document) >= target, name


def test_benchmark_results_directory_is_ignored():
    """Generated results must never be committed."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ignore_file = os.path.join(root, ".gitignore")
    with open(ignore_file, encoding="utf-8") as handle:
        assert "python_backend/benchmarks/results/" in handle.read()
