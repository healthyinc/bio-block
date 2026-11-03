"""The out-of-process worker's failure paths, without loading any weights.

The worker is the one component whose failures are invisible from inside the
API process: it dies, hangs or answers with rubbish in another process, and
the only thing the pipeline sees is a missing response. Every one of those
outcomes must block. A worker that cannot answer must never be read as a
worker that answered "no PHI found" - that is the exact shape of a fail-open
bug, and it would release the original text.

These drive the real client against a scripted stand-in child process, so they
run in the ordinary suite: no model weights, no downloads, and each failure
mode is produced on demand rather than waited for. The tests that genuinely
need the pinned models - real inference, checksum verification, offline
enforcement - live in ``test_production_model_path`` behind its opt-in flag.
"""

import json
import os
import subprocess
import sys
import textwrap
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.local_model_detectors import LocalModelError  # noqa: E402
from services.model_client import ModelWorkerClient  # noqa: E402

#: A stand-in worker. `behaviour` decides how it misbehaves after the banner.
FAKE_WORKER = textwrap.dedent("""
    import json, os, sys, time

    behaviour = sys.argv[1]
    GEN = int(os.environ.get("BIOBLOCK_WORKER_GENERATION", "0"))

    def say(message):
        sys.stdout.write(json.dumps({"v": 2, "gen": GEN, **message}) + "\\n")
        sys.stdout.flush()

    if behaviour == "no_banner":
        time.sleep(30)
        sys.exit(0)
    if behaviour == "banner_error":
        say({"id": 0, "ready": False, "error_code": "model_checksum_mismatch"})
        time.sleep(30)
        sys.exit(0)

    say({"id": 0, "ready": True, "models": ["fake"], "pid": os.getpid(), "load_count": 1})
    served = 0

    for line in sys.stdin:
        try:
            request = json.loads(line)
        except Exception:
            continue
        op = request.get("op")
        if op == "shutdown":
            sys.exit(0)
        served += 1
        if behaviour == "stderr_trace":
            # A traceback quoting the note: only its class may ever be counted.
            sys.stderr.write("Traceback (most recent call last):\\n")
            sys.stderr.write('  File "worker.py", line 9, in detect\\n')
            sys.stderr.write("ValueError: note mentions Oswin Treadgold 0161 555 0144\\n")
            sys.stderr.flush()
        if behaviour == "hang":
            time.sleep(30)
            continue
        if behaviour == "slow_first" and served == 1:
            # Answers the first request late, after its caller gave up.
            time.sleep(3)
        if behaviour == "wrong_id":
            say({"id": "someone-else", "ok": True, "entities": []})
            continue
        if behaviour == "wrong_gen":
            sys.stdout.write(json.dumps({"v": 2, "gen": GEN + 7, "id": request.get("id"),
                                         "ok": True, "entities": []}) + "\\n")
            sys.stdout.flush()
            continue
        if behaviour == "crash":
            sys.exit(9)
        if behaviour == "malformed":
            sys.stdout.write("this is not json\\n")
            sys.stdout.flush()
            continue
        if behaviour == "error":
            say({"id": request.get("id"), "ok": False, "error_code": "inference_failed"})
            continue
        if behaviour == "raw_error":
            say({"id": request.get("id"), "ok": False, "error_code": "Patient Private MRN 123"})
            continue
        if behaviour == "missing_entities":
            say({"id": request.get("id"), "ok": True})
            continue
        if behaviour == "bad_entities":
            say({"id": request.get("id"), "ok": True, "entities": {}})
            continue
        if behaviour == "invalid_id":
            say({"id": {}, "ok": True, "entities": []})
            continue
        if behaviour == "slow_once":
            time.sleep(2)
        if request.get("op") == "ping":
            say({"id": request.get("id"), "ok": True, "ready": True, "pid": os.getpid(),
                 "load_count": 1})
            continue
        say({
            "id": request.get("id"),
            "ok": True,
            "entities": [{
                "entity_type": "PERSON", "start": len(request.get("text", "")) % 3,
                "end": min(4, len(request.get("text", ""))), "source": "fake_detector", "score": 0.9,
            }],
        })
    """).strip()


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    """Return a factory that builds a client backed by the scripted worker."""
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")

    def build(behaviour: str) -> ModelWorkerClient:
        from services import model_client as module

        real_popen = subprocess.Popen

        def fake_popen(argv, **kwargs):
            return real_popen([sys.executable, str(script), behaviour], **kwargs)

        monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
        return ModelWorkerClient(python_executable=script)

    return build


# ---------------------------------------------------------------------------
# Startup, readiness, shutdown
# ---------------------------------------------------------------------------


def test_worker_starts_and_reports_ready(fake_worker):
    client = fake_worker("ok")
    try:
        client.start()
        assert client.is_ready() is True
    finally:
        client.stop()


def test_ready_worker_answers_a_detect_request(fake_worker):
    client = fake_worker("ok")
    try:
        entities = client.detect("fake", "Synthetic note text.")
        assert len(entities) == 1
        assert entities[0].entity_type == "PERSON"
    finally:
        client.stop()


def test_shutdown_leaves_no_running_process(fake_worker):
    client = fake_worker("ok")
    client.start()
    process = client._process

    client.stop()

    assert client.is_ready() is False
    assert process.poll() is not None


def test_a_worker_that_never_signals_ready_blocks(fake_worker, monkeypatch):
    from services import model_client as module

    monkeypatch.setattr(module, "STARTUP_TIMEOUT_SECONDS", 2.0)
    client = fake_worker("no_banner")
    try:
        with pytest.raises(LocalModelError) as excinfo:
            client.start()
        assert excinfo.value.error_code == "model_worker_not_ready"
    finally:
        client.stop()


def test_a_checksum_failure_at_startup_is_carried_through_not_flattened(fake_worker):
    """The banner's error code survives. "It did not start" is not a diagnosis."""
    client = fake_worker("banner_error")
    try:
        with pytest.raises(LocalModelError) as excinfo:
            client.start()
        assert excinfo.value.error_code == "model_checksum_mismatch"
    finally:
        client.stop()


# ---------------------------------------------------------------------------
# Failure modes, all of which must block
# ---------------------------------------------------------------------------


def test_timeout_blocks_and_does_not_return_an_empty_finding_list(
    fake_worker, monkeypatch
):
    from services import model_client as module

    monkeypatch.setattr(module, "REQUEST_TIMEOUT_SECONDS", 2.0)
    client = fake_worker("hang")
    try:
        with pytest.raises(LocalModelError) as excinfo:
            client.detect("fake", "Synthetic note text.")
        assert excinfo.value.error_code == "model_worker_timeout"
    finally:
        client.stop()


def test_worker_timeout_uses_the_bounded_configured_model_budget(monkeypatch):
    from services import model_client as module

    monkeypatch.setenv("PHI_MODEL_TIMEOUT_SECONDS", "420")
    assert module.request_timeout_seconds() == 420.0

    for invalid in ("0", "1801", "not-a-number"):
        monkeypatch.setenv("PHI_MODEL_TIMEOUT_SECONDS", invalid)
        with pytest.raises(LocalModelError) as caught:
            module.request_timeout_seconds()
        assert caught.value.error_code == "invalid_model_configuration"


def test_a_wedged_worker_is_replaced_once_after_its_grace_period(
    fake_worker, monkeypatch
):
    """Unanswered past its limit and the grace: one controlled replacement."""
    import time

    from services import model_client as module

    monkeypatch.setattr(module, "REQUEST_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setenv(module.GRACE_ENV_VAR, "0.5")
    client = fake_worker("hang")
    try:
        client.start()
        first = client.status()["generation"]
        with pytest.raises(LocalModelError) as caught:
            client.detect("fake", "Synthetic note text.")
        assert caught.value.error_code == "model_worker_timeout"
        waited = time.monotonic()
        while client.status()["restarts"] == 0 and time.monotonic() - waited < 20:
            time.sleep(0.1)
        time.sleep(1.0)
        status = client.status()
        assert status["restarts"] == 1
        assert status["generation"] > first
        assert client.is_ready() is True  # the successor started and is ready
    finally:
        client.stop()


def test_worker_crash_mid_request_blocks(fake_worker):
    client = fake_worker("crash")
    try:
        with pytest.raises(LocalModelError):
            client.detect("fake", "Synthetic note text.")
    finally:
        client.stop()


def test_a_malformed_response_blocks_instead_of_being_parsed_as_no_findings(
    fake_worker, monkeypatch
):
    from services import model_client as module

    monkeypatch.setattr(module, "REQUEST_TIMEOUT_SECONDS", 3.0)
    client = fake_worker("malformed")
    try:
        with pytest.raises(LocalModelError):
            client.detect("fake", "Synthetic note text.")
    finally:
        client.stop()


def test_an_error_response_blocks_with_its_own_code(fake_worker):
    client = fake_worker("error")
    try:
        with pytest.raises(LocalModelError) as excinfo:
            client.detect("fake", "Synthetic note text.")
        assert excinfo.value.error_code == "inference_failed"
    finally:
        client.stop()


def test_the_client_restarts_a_dead_worker_on_the_next_request(fake_worker):
    """Recovery is allowed; silently answering without a worker is not."""
    client = fake_worker("ok")
    try:
        client.start()
        client._process.kill()
        client._process.wait(timeout=10)

        entities = client.detect("fake", "Synthetic note text.")

        assert client.is_ready() is True
        assert len(entities) == 1
    finally:
        client.stop()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_requests_are_serialised_to_one_in_flight_at_a_time(fake_worker):
    """The worker holds two models on a memory-tight machine.

    Overlapping writes would interleave on one pipe and each reader would take
    the other's response. The lock is what keeps a request's answer its own.
    """
    from services import model_client as module

    assert module.MAX_CONCURRENT_REQUESTS == 1

    client = fake_worker("slow_once")
    results = []
    errors = []

    def run():
        try:
            results.append(client.detect("fake", "Synthetic note text."))
        except Exception as exc:  # pragma: no cover - failure detail only
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(3)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert not errors
        assert len(results) == 3
        assert all(len(entities) == 1 for entities in results)
    finally:
        client.stop()


# ---------------------------------------------------------------------------
# Diagnostics carry no content
# ---------------------------------------------------------------------------


def test_worker_responses_carry_codes_not_text():
    """An error path must not echo the text it was given.

    A worker's diagnostics are the least-watched output in the system, and
    text arriving there would be the original note, unredacted.
    """
    from services import model_worker

    worker = model_worker._Worker()
    worker._load_error = "model_files_unavailable"

    note = "Patient Rukmini Balasubramanian has MRN SYN-6610284."
    response = worker.handle(
        {
            "v": 2,
            "gen": worker.generation,
            "id": 1,
            "op": "detect",
            "detector": "x",
            "text": note,
        }
    )

    serialized = json.dumps(response)
    assert response["ok"] is False
    assert "Rukmini" not in serialized
    assert "SYN-6610284" not in serialized
    assert response["error_code"] == "model_files_unavailable"


def test_worker_stderr_reaches_the_client_only_as_error_class_counts(fake_worker):
    """A traceback quoting the note is drained and reduced to class names:
    no message, path or value is kept, and the stream never blocks the worker."""
    import time as _time

    client = fake_worker("stderr_trace")
    try:
        client.start()
        client.detect("fake", "abc")
        deadline = _time.monotonic() + 5
        while (
            not client.status().get("worker_diagnostics")
            and _time.monotonic() < deadline
        ):
            _time.sleep(0.05)
        diagnostics = client.status().get("worker_diagnostics")
        assert diagnostics == {"ValueError": 1}
        assert "Treadgold" not in repr(client.status())
    finally:
        client.stop()


def test_a_nearly_spent_deadline_is_a_deadline_error_and_sends_nothing(fake_worker):
    from services import request_deadline

    client = fake_worker("ok")
    try:
        client.start()
        before = client.status()["detect_requests"]
        request_deadline.start(1.0)
        try:
            with pytest.raises(LocalModelError) as caught:
                client.detect("fake", "abc")
        finally:
            request_deadline.clear()
        assert caught.value.error_code == "request_deadline_exceeded"
        status = client.status()
        assert status["restarts"] == 0 and status.get("deadline_refusals") == 1
        assert status["detect_requests"] == before + 1 and client.is_ready()
    finally:
        client.stop()


def test_a_caller_deadline_never_gets_a_healthy_worker_replaced(
    fake_worker, monkeypatch
):
    """The caller's deadline runs out while the worker is merely busy: the
    caller gets a deadline error, and the watchdog - judging the worker by
    the model budget, not the caller's clock - leaves the process alone."""
    import time as _time

    from services import model_client as module
    from services import request_deadline

    monkeypatch.setattr(module, "REQUEST_TIMEOUT_SECONDS", 20.0)
    monkeypatch.setattr(module, "WATCHDOG_INTERVAL_SECONDS", 0.2)
    monkeypatch.setenv(module.GRACE_ENV_VAR, "1")
    client = fake_worker("slow_first")
    try:
        client.start()
        pid = client.runtime["pid"]
        request_deadline.start(2.5)
        try:
            with pytest.raises(LocalModelError) as caught:
                client.detect("fake", "abc")
        finally:
            request_deadline.clear()
        assert caught.value.error_code == "request_deadline_exceeded"
        _time.sleep(2.0)
        assert client.status()["restarts"] == 0 and client.ping()["pid"] == pid
    finally:
        client.stop()


# ---------------------------------------------------------------------------
# Protocol v2: correlation by id, abandonment, generations
# ---------------------------------------------------------------------------


def test_an_abandoned_request_does_not_kill_a_healthy_worker_or_answer_the_next(
    fake_worker, monkeypatch
):
    """The caller gives up; the worker was only busy. Its late answer is
    dropped by id, and the next request gets its own answer from the same
    process."""
    from services import model_client as module

    monkeypatch.setattr(module, "REQUEST_TIMEOUT_SECONDS", 1.0)
    client = fake_worker("slow_first")
    try:
        client.start()
        pid = client.runtime["pid"]
        with pytest.raises(LocalModelError) as caught:
            client.detect("fake", "abc")
        assert caught.value.error_code == "model_worker_timeout"
        assert client.is_ready() is True

        monkeypatch.setattr(module, "REQUEST_TIMEOUT_SECONDS", 10.0)
        entities = client.detect("fake", "abcd")  # len % 3 == 1: its own answer

        assert entities[0].start == 1
        status = client.status()
        assert status["late_discarded"] == 1 and status["abandoned"] == 1
        assert status["restarts"] == 0 and client.ping()["pid"] == pid
    finally:
        client.stop()


def test_an_answer_for_another_request_fails_closed_and_replaces_the_worker(
    fake_worker,
):
    client = fake_worker("wrong_id")
    try:
        with pytest.raises(LocalModelError) as caught:
            client.detect("fake", "Synthetic note text.")
        assert caught.value.error_code == "model_response_id_mismatch"
    finally:
        client.stop()


def test_an_answer_from_another_generation_fails_closed(fake_worker):
    client = fake_worker("wrong_gen")
    try:
        with pytest.raises(LocalModelError) as caught:
            client.detect("fake", "Synthetic note text.")
        assert caught.value.error_code == "model_worker_generation_mismatch"
    finally:
        client.stop()


def test_concurrent_restart_requests_start_one_worker(fake_worker):
    client = fake_worker("ok")
    try:
        threads = [threading.Thread(target=client.start) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert client.status()["starts"] == 1

        generation = client.status()["generation"]
        replacers = [
            threading.Thread(
                target=client._replace, args=(generation, "model_worker_unresponsive")
            )
            for _ in range(5)
        ]
        for thread in replacers:
            thread.start()
        for thread in replacers:
            thread.join(timeout=60)
        status = client.status()
        assert status["restarts"] == 1 and status["starts"] == 2
        assert client.is_ready()
    finally:
        client.stop()


def test_waiting_callers_are_bounded(fake_worker, monkeypatch):
    from services import model_client as module

    monkeypatch.setattr(module, "MAX_PENDING_REQUESTS", 1)
    monkeypatch.setattr(module, "REQUEST_TIMEOUT_SECONDS", 5.0)
    client = fake_worker("slow_once")
    errors = []

    def run():
        try:
            client.detect("fake", "Synthetic note text.")
        except LocalModelError as exc:
            errors.append(exc.error_code)

    try:
        client.start()
        threads = [threading.Thread(target=run) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert module.ERROR_WORKER_BUSY in errors
    finally:
        client.stop()


def test_a_crash_fails_the_request_in_flight_with_its_own_code(fake_worker):
    client = fake_worker("crash")
    try:
        with pytest.raises(LocalModelError) as caught:
            client.detect("fake", "Synthetic note text.")
        assert caught.value.error_code == "model_worker_crashed"
    finally:
        client.stop()


def test_the_worker_refuses_another_generation_or_protocol():
    from services import model_worker

    worker = model_worker._Worker()
    wrong_gen = worker.handle(
        {"v": 2, "gen": worker.generation + 1, "id": 5, "op": "ping"}
    )
    old_protocol = worker.handle({"id": 6, "op": "ping"})

    assert wrong_gen == {
        "id": 5,
        "ok": False,
        "error_code": "worker_generation_mismatch",
    }
    assert old_protocol["error_code"] == "protocol_version_mismatch"


@pytest.mark.parametrize(
    "behaviour,code",
    [
        ("missing_entities", "model_output_malformed"),
        ("bad_entities", "model_output_malformed"),
        ("invalid_id", "model_response_invalid_id"),
        ("raw_error", "model_worker_error"),
    ],
)
def test_invalid_success_or_error_payloads_fail_closed(fake_worker, behaviour, code):
    client = fake_worker(behaviour)
    try:
        client.start()
        with pytest.raises(LocalModelError) as raised:
            client.detect("fake", "Patient Private MRN 123")
        assert raised.value.error_code == code
        assert "Private" not in str(raised.value)
    finally:
        client.stop()


def test_protocol_failure_revokes_readiness_before_supervisor_runs(monkeypatch):
    from types import SimpleNamespace

    client = ModelWorkerClient()
    client._generation = 7
    client._ready = True
    client._process = SimpleNamespace(poll=lambda: None)
    # Deliberately leave replacement unscheduled: readiness cannot depend on
    # how quickly a background thread gets CPU time.
    monkeypatch.setattr(client, "_replace", lambda *args: None)
    assert client.is_ready()
    assert client._protocol_failure(7, "model_response_malformed") is False
    assert not client.is_ready()
    client._process = None
