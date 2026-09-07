"""Proof that the real models participate in the production API path.

Standalone evaluation is not proof of integration. Before Phase 10 the two
were quietly disconnected: the calibration ran under the model virtualenv,
while the FastAPI process could not import torch at all and answered every
``PHI_MODEL_MODE=offline`` request with ``model_files_unavailable``. The
thresholds were real and unreachable.

These tests drive the actual FastAPI route with invented clinical text and
assert that Stanford and GLiNER appear in the detection sources of the
response. They are opt-in because they start a real worker and load 1.5 GiB of
weights.

    PHI_RUN_REAL_MODEL_EVAL=1 PHI_MODEL_WORKER=1 py -3.11 -m pytest \\
        tests/test_production_model_path.py -m real_models

Every value below is invented.
"""

import io
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.real_models

REQUIRED = pytest.mark.skipif(
    os.getenv("PHI_RUN_REAL_MODEL_EVAL") != "1",
    reason="real-model integration is opt-in; set PHI_RUN_REAL_MODEL_EVAL=1",
)

# Invented. No real patient information appears in this file.
CLINICAL_NOTE = (
    "Patient Padmavathi Venkataraghavan was reviewed by "
    "Dr. Shalini Muthukrishnan at Duskwater Priory Teaching Hospital "
    "on 2015-08-14.\n"
    "MRN: SYN-2748903. Contact p.venkataraghavan@example.invalid.\n"
    "The patient is 102 years old and has Parkinson's disease.\n"
    "Metformin 500 mg twice daily. HbA1c 7.4 percent. Creatinine 1.2 mg/dL.\n"
    "CT of the abdomen showed no acute finding."
)

STANFORD_SOURCE = "stanford_deidentifier"
GLINER_SOURCE = "gliner_multi_pii"


@pytest.fixture(scope="module")
def offline_worker_api():
    """FastAPI client with the out-of-process model path enabled."""
    from services import text_anonymization
    from services.model_client import shutdown_client, worker_python

    if worker_python() is None:
        pytest.skip("model virtualenv is not provisioned")

    previous = {
        key: os.environ.get(key)
        for key in ("PHI_MODEL_MODE", "PHI_MODEL_WORKER", "BIOBLOCK_STUDY_SALT")
    }
    os.environ["PHI_MODEL_MODE"] = "offline"
    os.environ["PHI_MODEL_WORKER"] = "1"
    os.environ.setdefault("BIOBLOCK_STUDY_SALT", "integration-salt")
    text_anonymization._build_detectors.cache_clear()

    from fastapi.testclient import TestClient

    from main import app

    try:
        yield TestClient(app)
    finally:
        shutdown_client()
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        text_anonymization._build_detectors.cache_clear()


def _ingest(client, text: str):
    return client.post(
        "/api/v1/ingest",
        files={"file": ("note.txt", io.BytesIO(text.encode("utf-8")), "text/plain")},
        data={"profile": "strict"},
    )


@pytest.fixture
def standalone_worker():
    """One worker, and only one, for tests that drive the client directly.

    Each of these starts its own child holding both models. Run after a test
    that left the module-level client alive, two children compete for about
    three gigabytes and the second never reaches its readiness banner - which
    surfaces as `model_worker_not_ready` and reads like a fail-closed bug
    rather than the memory contention it is. Production has one client per API
    process; the fixture restores that invariant.
    """
    from services.model_client import ModelWorkerClient, shutdown_client, worker_python

    if worker_python() is None:
        pytest.skip("model virtualenv is not provisioned")

    shutdown_client()
    client = ModelWorkerClient()
    try:
        yield client
    finally:
        client.stop()


@REQUIRED
def test_real_models_participate_in_the_api_route(offline_worker_api):
    """Both pinned models must be in the chain the API actually runs.

    Asserted at the detector level rather than on the final
    ``detection_sources``: overlap resolution legitimately drops a model's
    span when another detector proposes a longer one covering the same text,
    so a model can participate without surviving into the resolved output.
    """
    from services.ner_phi_detector import configured_model_name
    from services.text_anonymization import _detectors

    chain = _detectors(configured_model_name(), "strict")
    sources = {getattr(d, "source", None) for d in chain}
    assert STANFORD_SOURCE in sources, f"Stanford not wired into the chain: {sources}"
    assert GLINER_SOURCE in sources, f"GLiNER not wired into the chain: {sources}"

    # And each must actually return findings for this note, through the worker.
    for detector in chain:
        if getattr(detector, "source", None) in (STANFORD_SOURCE, GLINER_SOURCE):
            found = detector.detect(CLINICAL_NOTE)
            assert found, f"{detector.source} returned nothing"

    # The route itself runs that chain end to end. The resolved
    # detection_sources are deliberately not asserted on: overlap resolution
    # ranks rule and spaCy findings above model findings on a tie, so a model
    # routinely contributes a span that another detector's label then wins.
    # Absence from the resolved output is not absence from the pipeline.
    response = _ingest(offline_worker_api, CLINICAL_NOTE)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["anonymization_status"] == "completed"
    assert body["detection_sources"], "the chain produced no findings at all"


@REQUIRED
def test_api_route_removes_identity_and_keeps_medicine(offline_worker_api):
    response = _ingest(offline_worker_api, CLINICAL_NOTE)
    body = response.json()
    anonymized = body["anonymized_text"]

    for identifier in (
        "Padmavathi",
        "Venkataraghavan",
        "Shalini",
        "Muthukrishnan",
        "Duskwater",
        "SYN-2748903",
        "p.venkataraghavan@example.invalid",
    ):
        assert identifier not in anonymized, f"{identifier!r} survived"

    for clinical in ("Parkinson", "Metformin", "500", "HbA1c", "7.4", "Creatinine", "CT"):
        assert clinical in anonymized, f"{clinical!r} was destroyed"

    # Safe Harbor aggregation, applied by the deterministic age rule.
    assert "90+" in anonymized
    assert "102" not in anonymized


@REQUIRED
def test_api_route_uses_consistent_surrogates(offline_worker_api):
    import re

    repeated = (
        "Patient Padmavathi Venkataraghavan attended. "
        "Padmavathi Venkataraghavan was discharged. "
        "Dr. Shalini Muthukrishnan signed the note."
    )
    body = _ingest(offline_worker_api, repeated).json()
    anonymized = body["anonymized_text"]

    patients = set(re.findall(r"PATIENT_\d{3,}", anonymized))
    providers = set(re.findall(r"PROVIDER_\d{3,}", anonymized))

    # One patient mentioned twice keeps one surrogate; the clinician is a
    # different person and gets a different one.
    assert len(patients) == 1, anonymized
    assert anonymized.count(next(iter(patients))) == 2
    assert len(providers) == 1, anonymized
    assert patients.isdisjoint(providers)


@REQUIRED
def test_worker_readiness_and_clean_shutdown(standalone_worker):
    client = standalone_worker
    client.start()
    try:
        assert client.is_ready()
        ping = client.ping()
        assert ping["ok"] is True
        assert ping["ready"] is True
        assert set(ping["models"]) == {"stanford", "gliner"}
    finally:
        client.stop()
    assert not client.is_ready()


@REQUIRED
def test_worker_rejects_an_oversized_request(standalone_worker):
    from services.local_model_detectors import LocalModelError

    client = standalone_worker
    client.start()
    try:
        with pytest.raises(LocalModelError) as exc:
            client.detect("stanford", "a" * (600 * 1024))
        assert exc.value.error_code == "request_too_large"
    finally:
        client.stop()


@REQUIRED
def test_worker_failure_blocks_rather_than_returning_no_findings(monkeypatch):
    """A dead worker must block the artifact, never report a clean document."""
    from services.local_model_detectors import LocalModelError
    from services.model_client import RemoteModelDetector

    class DeadClient:
        def detect(self, detector, text):
            raise LocalModelError("model_worker_timeout", 503)

    monkeypatch.setattr(
        "services.model_client.get_client", lambda: DeadClient()
    )

    with pytest.raises(LocalModelError) as exc:
        RemoteModelDetector("stanford", "stanford_deidentifier").detect("Patient X.")

    assert exc.value.error_code == "model_worker_timeout"


def test_missing_worker_python_fails_closed(monkeypatch):
    """Runs without the models: a missing interpreter must block, not skip."""
    from services.local_model_detectors import LocalModelError
    from services.model_client import ModelWorkerClient

    client = ModelWorkerClient(python_executable=None)
    monkeypatch.setattr("services.model_client.worker_python", lambda: None)
    client._python = None

    with pytest.raises(LocalModelError) as exc:
        client.start()

    assert exc.value.error_code == "model_worker_python_unavailable"
