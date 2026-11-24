"""Phase 12F: one deadline for the whole request, and what it must guarantee.

One acceptance request took 2,846 seconds without a single timeout firing.
Nothing was broken in the sense of raising: every budget in the path was a
*per-call* budget, and each one was satisfied. ``_run_batched`` started a
fresh window on every ``detect()``, a request makes four of those, and the
worker client's read timeout reset per call as well. Four conforming calls
still add up to an unbounded request.

These tests fence in the fix. Three properties matter more than the number
itself:

* the deadline is created once, from server configuration, and nothing
  downstream may extend it - not a detector, chunk, batch or pass;
* a worker blocked inside an inference call is killed by the parent, because
  the worker's own between-batch check cannot interrupt it;
* running out of time produces no artifact, and no response from a worker
  that was replaced can be mistaken for an answer to the live request.

The slow worker here is mocked. A real blocked model would make these tests
take as long as the bug did.
"""

import io
import json
import os
import subprocess
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import request_deadline as rd  # noqa: E402
from services.request_deadline import (  # noqa: E402
    DeadlineConfigurationError,
    RequestDeadlineExceeded,
)

NOTE = b"Patient Zelquin Vantreske, MRN 44821903, seen on 14 March 2021."


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setenv("PHI_MODEL_MODE", "legacy_test")
    monkeypatch.setenv("BIOBLOCK_STUDY_SALT", "phase12f-salt")
    monkeypatch.delenv(rd.DEADLINE_ENV_VAR, raising=False)
    rd.clear()
    yield
    rd.clear()


def _client():
    import main
    from fastapi.testclient import TestClient

    return TestClient(main.app), main


# -- configuration ----------------------------------------------------------


def test_default_deadline_is_used_when_nothing_is_configured():
    assert rd.configured_seconds() == rd.DEFAULT_DEADLINE_SECONDS


def test_default_covers_the_largest_upload_the_route_accepts():
    """A 200 KB document measured ~212 s in 12A; the default must clear it."""
    assert rd.DEFAULT_DEADLINE_SECONDS > 212.0


def test_configured_deadline_is_honoured(monkeypatch):
    monkeypatch.setenv(rd.DEADLINE_ENV_VAR, "90")

    assert rd.configured_seconds() == 90.0


@pytest.mark.parametrize(
    "value", ["abc", "-1", "0", "4.9", "1801", "1e9", "None", "60s"]
)
def test_an_unusable_deadline_is_a_configuration_error(monkeypatch, value):
    """A bad value fails loudly. It must never resolve to 'no deadline'."""
    monkeypatch.setenv(rd.DEADLINE_ENV_VAR, value)

    with pytest.raises(DeadlineConfigurationError) as exc:
        rd.configured_seconds()
    assert exc.value.error_code == "invalid_request_deadline"


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_deadline_means_unset(monkeypatch, value):
    monkeypatch.setenv(rd.DEADLINE_ENV_VAR, value)

    assert rd.configured_seconds() == rd.DEFAULT_DEADLINE_SECONDS


def test_startup_rejects_an_unusable_deadline(monkeypatch):
    from services import model_lifecycle

    monkeypatch.setenv(rd.DEADLINE_ENV_VAR, "nonsense")
    monkeypatch.setenv("PHI_MODEL_PREWARM", "0")
    model_lifecycle.reset_state()

    state = model_lifecycle.prewarm()

    assert state.is_ready() is False
    assert state.error_code == "invalid_request_deadline"


@pytest.mark.parametrize(
    "field",
    [
        "deadline",
        "timeout",
        "request_deadline_seconds",
        "PHI_REQUEST_DEADLINE_SECONDS",
        "max_seconds",
    ],
)
def test_an_upload_cannot_set_the_deadline(monkeypatch, field):
    """Server configuration is the only way in."""
    monkeypatch.setenv(rd.DEADLINE_ENV_VAR, "60")
    client, _ = _client()

    response = client.post(
        "/api/v1/ingest",
        files={"file": ("note.txt", io.BytesIO(NOTE), "text/plain")},
        data={"profile": "strict", field: "9999"},
    )

    assert response.status_code in {200, 422}
    assert rd.configured_seconds() == 60.0


# -- one deadline, never extended -------------------------------------------


def test_the_deadline_is_absolute_and_monotonic():
    deadline = rd.start(10.0)
    first = deadline.remaining()
    time.sleep(0.05)

    assert deadline.remaining() < first
    assert deadline.total_seconds == 10.0


def test_a_sub_limit_never_exceeds_what_is_left():
    rd.start(2.0)

    assert rd.budget_for("stage", 120.0) <= 2.0
    assert rd.budget_for("stage", 0.5) == pytest.approx(0.5, abs=0.01)


def test_repeated_sub_limits_do_not_reset_the_clock():
    """The defect, stated directly: four calls of 120 s each used to be legal."""
    rd.start(1.0)
    budgets = []
    for _ in range(4):
        budgets.append(rd.budget_for("model_inference", 120.0))
        time.sleep(0.05)

    assert all(b <= 1.0 for b in budgets)
    assert budgets == sorted(budgets, reverse=True), "budget must only shrink"


def test_check_raises_once_the_deadline_has_passed():
    rd.start(0.05)
    time.sleep(0.1)

    with pytest.raises(RequestDeadlineExceeded) as exc:
        rd.check("detect.stanford_deidentifier")
    assert exc.value.error_code == "request_deadline_exceeded"
    assert exc.value.stage == "detect.stanford_deidentifier"
    assert exc.value.status_code == 503


def test_budget_for_raises_rather_than_returning_a_negative_budget():
    rd.start(0.05)
    time.sleep(0.1)

    with pytest.raises(RequestDeadlineExceeded):
        rd.budget_for("model_worker", 120.0)


def test_no_active_deadline_leaves_sub_limits_alone():
    """Offline evaluation runs outside a request and keeps its own budgets."""
    rd.clear()

    assert rd.remaining() is None
    assert rd.budget_for("model_inference", 120.0) == 120.0
    rd.check("anything")


def test_the_deadline_does_not_leak_between_requests():
    client, _ = _client()

    client.post(
        "/api/v1/ingest",
        files={"file": ("note.txt", io.BytesIO(NOTE), "text/plain")},
        data={"profile": "strict"},
    )

    assert rd.current() is None


def test_concurrent_requests_do_not_share_a_deadline():
    """A ContextVar, so one request's clock is invisible to another."""
    seen = {}

    def worker(name, seconds):
        rd.start(seconds)
        time.sleep(0.05)
        seen[name] = rd.current().total_seconds

    threads = [
        threading.Thread(target=worker, args=("a", 10.0)),
        threading.Thread(target=worker, args=("b", 20.0)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert seen == {"a": 10.0, "b": 20.0}
    assert rd.current() is None


# -- the detector path ------------------------------------------------------


def test_batched_inference_is_bounded_by_the_request_not_its_own_budget():
    from services.local_model_detectors import (
        DetectorConfig,
        LocalModelError,
        _run_batched,
    )

    config = DetectorConfig(inference_budget_seconds=120.0, batch_size=1)
    chunks = [(i, "chunk %d" % i) for i in range(4)]

    def slow(texts):
        time.sleep(0.08)
        return [[] for _ in texts]

    rd.start(0.15)
    with pytest.raises(LocalModelError) as exc:
        _run_batched(chunks, config, slow, "model_inference_failed")

    assert "timeout" in exc.value.error_code or "deadline" in exc.value.error_code


def test_batched_inference_keeps_its_own_budget_when_it_is_stricter():
    from services.local_model_detectors import (
        DetectorConfig,
        LocalModelError,
        _run_batched,
    )

    config = DetectorConfig(inference_budget_seconds=0.1, batch_size=1)
    chunks = [(i, "chunk %d" % i) for i in range(4)]

    def slow(texts):
        time.sleep(0.08)
        return [[] for _ in texts]

    rd.start(600.0)
    with pytest.raises(LocalModelError):
        _run_batched(chunks, config, slow, "model_inference_failed")


def test_a_detection_pass_stops_between_detectors_when_time_runs_out():
    from services import text_anonymization as ta
    from services.ner_phi_detector import NerPhiDetectionError

    calls = []

    class _Slow:
        source = "stanford_deidentifier"

        def detect(self, text):
            calls.append(1)
            time.sleep(0.12)
            return []

    original = ta._detectors
    ta._detectors = lambda *_a, **_k: (_Slow(), _Slow(), _Slow())
    try:
        rd.start(0.2)
        with pytest.raises(NerPhiDetectionError) as exc:
            ta._detect_entities("Patient note.", "en_core_web_sm", "strict")
    finally:
        ta._detectors = original

    assert exc.value.error_code == "request_deadline_exceeded"
    assert len(calls) < 3, "the chain must stop, not run to completion"


def test_running_out_of_time_blocks_rather_than_returning_findings():
    """Fail-closed: a partial detection is not a clean document."""
    from services import text_anonymization as ta
    from services.ner_phi_detector import NerPhiDetectionError

    class _Expired:
        source = "stanford_deidentifier"

        def detect(self, text):
            raise RequestDeadlineExceeded(stage="detect.stanford_deidentifier")

    original = ta._detectors
    ta._detectors = lambda *_a, **_k: (_Expired(),)
    try:
        rd.start(60.0)
        with pytest.raises(NerPhiDetectionError):
            ta._detect_entities("Patient note.", "en_core_web_sm", "strict")
    finally:
        ta._detectors = original


# -- the worker boundary ----------------------------------------------------


class _FakeProcess:
    """A worker that answers each request on demand, or never.

    It answers only after a request has been written, as a real worker does,
    so the client's single reader never sees an answer before its question.
    A scripted response without ``v``/``gen``/``id`` gets the current ones
    filled in unless the test set them.
    """

    def __init__(self, responses=None, hang=False):
        self._responses = list(responses or [])
        self._hang = hang
        self.stdout = self
        self.stdin = self
        #: What the parent wrote, kept after close() so a test can inspect it.
        self.written = []
        self.terminated = False
        self.killed = False
        self._exited = False
        self._answered = 0
        self._changed = threading.Condition()

    # stdout half
    def readline(self):
        with self._changed:
            while True:
                if self._exited:
                    return ""
                requests = [json.loads(line) for line in self.written]
                requests = [r for r in requests if r.get("op") != "shutdown"]
                if (
                    not self._hang
                    and self._answered < len(requests)
                    and self._answered < len(self._responses)
                ):
                    request = requests[self._answered]
                    response = dict(self._responses[self._answered])
                    self._answered += 1
                    response.setdefault("v", request.get("v"))
                    response.setdefault("gen", request.get("gen"))
                    if response.get("id", "") == "":
                        response["id"] = request.get("id")
                    return json.dumps(response) + "\n"
                self._changed.wait(0.05)

    def close(self):
        pass

    # stdin half
    def write(self, payload):
        with self._changed:
            self.written.append(payload)
            self._changed.notify_all()
        return len(payload)

    # process half
    def poll(self):
        return 0 if self._exited else None

    def wait(self, timeout=None):
        if self._exited:
            return 0
        raise subprocess.TimeoutExpired("worker", timeout or 0)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        with self._changed:
            self._exited = True
            self._changed.notify_all()

    def flush(self):
        pass


def _worker_client(process, ready=True):
    """A client adopting ``process`` as its generation-1 worker."""
    from services.model_client import ModelWorkerClient

    client = ModelWorkerClient(python_executable=None)
    client._generation = 1
    client._process = process
    client._ready = ready
    client._reader = threading.Thread(
        target=client._read_loop, args=(process, 1), daemon=True
    )
    client._reader.start()
    client._ensure_watchdog()
    return client


def test_a_caller_timeout_abandons_the_request_without_killing_the_worker():
    from services.local_model_detectors import LocalModelError

    process = _FakeProcess(hang=True)
    client = _worker_client(process)
    try:
        rd.start(60.0)
        started = time.monotonic()
        with pytest.raises(LocalModelError) as exc:
            client._request({"op": "ping"}, timeout=0.1)
        elapsed = time.monotonic() - started

        assert exc.value.error_code == rd.ERROR_WORKER_TIMEOUT
        assert elapsed < 5
        assert not process.terminated and not process.killed
        assert client.status()["abandoned"] == 1
    finally:
        client.stop()


def test_a_wedged_worker_is_terminated_after_its_grace_not_waited_on(monkeypatch):
    from services import model_client as mc
    from services.local_model_detectors import LocalModelError

    monkeypatch.setenv(mc.GRACE_ENV_VAR, "0.2")
    process = _FakeProcess(hang=True)
    client = _worker_client(process)
    try:
        with pytest.raises(LocalModelError):
            client._request({"op": "ping"}, timeout=0.1)
        waited = time.monotonic()
        while not process.killed and time.monotonic() - waited < 30:
            time.sleep(0.05)

        assert process.terminated and process.killed
        assert client.status()["restarts"] == 1
        assert client.is_ready() is False  # no interpreter: it cannot restart
    finally:
        client.stop()


def test_the_worker_read_timeout_shrinks_to_the_remaining_request_time():
    from services import model_client as mc

    seen = {}

    class _Recording(mc.ModelWorkerClient):
        def _request(self, payload, timeout, **_kwargs):
            seen["timeout"] = timeout
            return {"id": 1, "ok": True, "entities": []}

    rd.start(3.0)
    _Recording(python_executable=None).detect("stanford", "text")

    assert seen["timeout"] <= 3.0
    assert seen["timeout"] < mc.REQUEST_TIMEOUT_SECONDS


def test_a_response_from_another_generation_is_rejected():
    """The contamination case: an old process answering into a live request."""
    from services.local_model_detectors import LocalModelError

    process = _FakeProcess(responses=[{"gen": 99, "ok": True, "entities": []}])
    client = _worker_client(process)
    try:
        with pytest.raises(LocalModelError) as exc:
            client._request({"op": "ping"}, timeout=5.0)
        assert exc.value.error_code == "model_worker_generation_mismatch"
    finally:
        client.stop()


def test_a_response_for_a_different_request_is_rejected():
    from services.local_model_detectors import LocalModelError

    process = _FakeProcess(responses=[{"id": 99, "ok": True, "entities": []}])
    client = _worker_client(process)
    try:
        with pytest.raises(LocalModelError) as exc:
            client._request({"op": "ping"}, timeout=5.0)
        assert exc.value.error_code == "model_response_id_mismatch"
    finally:
        client.stop()


def test_a_mismatched_response_replaces_the_worker():
    """The stream cannot be trusted; reusing it could misanswer a later request."""
    from services.local_model_detectors import LocalModelError

    process = _FakeProcess(responses=[{"id": 99, "ok": True, "entities": []}])
    client = _worker_client(process)
    try:
        with pytest.raises(LocalModelError):
            client._request({"op": "ping"}, timeout=5.0)
        waited = time.monotonic()
        while (
            not (process.terminated or process.killed)
            and time.monotonic() - waited < 30
        ):
            time.sleep(0.05)

        assert client.is_ready() is False
        assert process.terminated or process.killed
    finally:
        client.stop()


def test_a_response_without_an_identifier_is_rejected():
    from services.local_model_detectors import LocalModelError

    process = _FakeProcess(responses=[{"id": None, "ok": True, "entities": []}])
    client = _worker_client(process)
    try:
        with pytest.raises(LocalModelError) as exc:
            client._request({"op": "ping"}, timeout=5.0)
        assert exc.value.error_code == "model_response_missing_id"
    finally:
        client.stop()


def test_request_identifiers_do_not_repeat_within_a_client():
    process = _FakeProcess(responses=[{"ok": True}, {"ok": True}])
    client = _worker_client(process)
    try:
        client._request({"op": "ping"}, timeout=5.0)
        client._request({"op": "ping"}, timeout=5.0)

        assert client._request_id == 2
        ids = [json.loads(line)["id"] for line in process.written]
        assert ids == [1, 2]
    finally:
        client.stop()


def test_every_request_carries_version_generation_and_id():
    process = _FakeProcess(responses=[{"ok": True}])
    client = _worker_client(process)
    try:
        client._request({"op": "ping"}, timeout=5.0)
        sent = json.loads(process.written[0])
        assert (sent["v"], sent["gen"], sent["id"]) == (2, 1, 1)
    finally:
        client.stop()


def test_a_worker_that_cannot_be_restarted_fails_closed():
    """No worker means no detection, and no detection means no artifact."""
    from services.local_model_detectors import LocalModelError

    client = _worker_client(_FakeProcess(hang=True), ready=False)
    client._process = None
    client._python = None
    try:
        with pytest.raises(LocalModelError) as exc:
            client._request({"op": "ping"}, timeout=1.0)
        assert exc.value.error_code == "model_worker_python_unavailable"
    finally:
        client.stop()


def test_shutdown_at_exit_is_still_polite():
    """An idle worker is asked to stop; a busy one would not be reading."""
    process = _FakeProcess(responses=[])
    client = _worker_client(process)

    client.stop()

    assert any("shutdown" in line for line in process.written)


# -- the response the client gets -------------------------------------------


def _probe_response(path, stage):
    import main
    from fastapi.testclient import TestClient

    async def _probe():
        raise RequestDeadlineExceeded(stage=stage)

    main.app.get(path)(_probe)
    return TestClient(main.app, raise_server_exceptions=False).get(path)


def test_the_timeout_response_carries_no_document_content():
    response = _probe_response("/_deadline_probe", "detect.gliner_multi_pii")

    assert response.status_code == 503
    for forbidden in ("Vantreske", "44821903", "note.txt", "Users", "Patient"):
        assert forbidden not in response.text


def test_the_timeout_response_uses_the_allowed_status_vocabulary():
    body = _probe_response(
        "/_deadline_probe_status", "anonymize.residual_validation"
    ).json()

    assert body["anonymization_status"] == "privacy_requirements_not_met"
    assert body["error_code"] == "request_deadline_exceeded"
    assert "hipaa" not in json.dumps(body).lower()


def test_the_timeout_response_contains_no_artifact():
    body = _probe_response("/_deadline_probe_artifact", "anonymize.detect_pass1").json()

    for forbidden in (
        "anonymized_text",
        "download_url",
        "artifact_id",
        "preview",
        "cid",
        "ipfs_hash",
    ):
        assert forbidden not in body


def test_the_deadline_module_never_decides_releasability():
    import inspect

    source = inspect.getsource(rd)

    for forbidden in (
        "release_gate",
        "release_decision",
        "releasable",
        "issue_release",
        "anonymized_text",
    ):
        assert forbidden not in source
