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
FAKE_WORKER = textwrap.dedent(
    '''
    import json, sys, time

    behaviour = sys.argv[1]

    if behaviour == "no_banner":
        time.sleep(30)
        sys.exit(0)
    if behaviour == "banner_error":
        sys.stdout.write(json.dumps({"ready": False, "error_code": "model_checksum_mismatch"}) + "\\n")
        sys.stdout.flush()
        time.sleep(30)
        sys.exit(0)

    sys.stdout.write(json.dumps({"ready": True, "models": ["fake"]}) + "\\n")
    sys.stdout.flush()

    for line in sys.stdin:
        try:
            request = json.loads(line)
        except Exception:
            continue
        op = request.get("op")
        if op == "shutdown":
            sys.exit(0)
        if behaviour == "hang":
            time.sleep(30)
            continue
        if behaviour == "crash":
            sys.exit(9)
        if behaviour == "malformed":
            sys.stdout.write("this is not json\\n")
            sys.stdout.flush()
            continue
        if behaviour == "error":
            sys.stdout.write(json.dumps({"id": request.get("id"), "ok": False, "error_code": "inference_failed"}) + "\\n")
            sys.stdout.flush()
            continue
        if behaviour == "slow_once":
            time.sleep(2)
        sys.stdout.write(json.dumps({
            "id": request.get("id"),
            "ok": True,
            "entities": [{
                "entity_type": "PERSON", "start": 0, "end": 4,
                "source": "fake_detector", "score": 0.9,
            }],
        }) + "\\n")
        sys.stdout.flush()
    '''
).strip()


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    """Return a factory that builds a client backed by the scripted worker."""
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")

    def build(behaviour: str) -> ModelWorkerClient:
        from services import model_client as module

        real_popen = subprocess.Popen

        def fake_popen(argv, **kwargs):
            return real_popen(
                [sys.executable, str(script), behaviour], **kwargs
            )

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


def test_a_wedged_worker_is_killed_rather_than_left_desynchronised(
    fake_worker, monkeypatch
):
    """Half a response left in the pipe would corrupt every later request."""
    from services import model_client as module

    monkeypatch.setattr(module, "REQUEST_TIMEOUT_SECONDS", 2.0)
    client = fake_worker("hang")
    try:
        with pytest.raises(LocalModelError):
            client.detect("fake", "Synthetic note text.")
        assert client.is_ready() is False
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
    response = worker.handle({"id": 1, "op": "detect", "detector": "x", "text": note})

    serialized = json.dumps(response)
    assert response["ok"] is False
    assert "Rukmini" not in serialized
    assert "SYN-6610284" not in serialized
    assert response["error_code"] == "model_files_unavailable"


def test_client_never_inherits_the_worker_stderr_stream():
    """stderr is discarded, so a traceback quoting the note cannot surface."""
    from services import model_client

    source = model_client.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()

    assert "stderr=subprocess.DEVNULL" in text
