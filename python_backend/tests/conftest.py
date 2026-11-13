import os
import socket
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.local_model_detectors import (  # noqa: E402
    MODE_LEGACY_TEST,
    MODEL_MODE_ENV_VAR,
    NON_PRODUCTION_ACK_ENV_VAR,
)

# Ordinary unit tests run against deterministic local doubles and the already
# pinned small spaCy fixture model. The 438 MB / 1.16 GB pinned snapshots are
# never fetched or loaded here; real-model checks are a separate opt-in command
# (see tests/test_real_model_evaluation.py and evaluations/real_model_smoke.py).
os.environ[MODEL_MODE_ENV_VAR] = MODE_LEGACY_TEST
os.environ[NON_PRODUCTION_ACK_ENV_VAR] = "1"

# Ordinary tests must not acquire weights or use a developer's persistent
# application index. Real-model checks remain explicitly opt-in below.
_ordinary = os.environ.get("PHI_RUN_REAL_MODEL_EVAL") != "1"
_isolation = pytest.MonkeyPatch()
_test_storage = None
if _ordinary:
    _isolation.setenv("HF_HUB_OFFLINE", "1")
    _isolation.setenv("TRANSFORMERS_OFFLINE", "1")
    _isolation.setenv("ANONYMIZED_TELEMETRY", "False")

    _socket_connect = socket.socket.connect
    _socket_connect_ex = socket.socket.connect_ex

    def _no_network(sock, address):
        # Windows asyncio implements its event-loop wakeup pipe with a
        # loopback socket pair. ASGI requests still use no external transport.
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return _socket_connect(sock, address)
        raise AssertionError("ordinary tests cannot connect to external services")

    def _no_network_ex(sock, address):
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return _socket_connect_ex(sock, address)
        raise AssertionError("ordinary tests cannot connect to external services")

    _isolation.setattr(socket.socket, "connect", _no_network)
    _isolation.setattr(socket.socket, "connect_ex", _no_network_ex)

    import chromadb
    from chromadb.api.types import DefaultEmbeddingFunction

    def _test_embeddings(self, input):
        # A deterministic storage double; these tests do not qualify retrieval
        # quality. No Chroma ONNX model is loaded or downloaded.
        return [[1.0] + [0.0] * 15 for _ in input]

    _isolation.setattr(DefaultEmbeddingFunction, "__call__", _test_embeddings)
    _persistent_client = chromadb.PersistentClient
    _test_storage = tempfile.TemporaryDirectory(
        prefix="bioblock-core-tests-", ignore_cleanup_errors=True
    )

    def _isolated_client(path=None, **kwargs):
        return _persistent_client(path=_test_storage.name, **kwargs)

    _isolation.setattr(chromadb, "PersistentClient", _isolated_client)


def pytest_unconfigure(config):
    _isolation.undo()
    if _test_storage is not None:
        _test_storage.cleanup()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_models: opt-in checks that load locally provisioned pinned weights; "
        "skipped unless PHI_RUN_REAL_MODEL_EVAL=1.",
    )


@pytest.fixture
def simulated_production_detectors(monkeypatch):
    """Exercise release mechanics as if the pinned models had run.

    Detection is still the lightweight test chain. A test using this shows what
    the release path does with a verified artifact - storing it, serving it,
    refusing tampered bytes - not that the models would have found everything.
    It exists only in tests: nothing a deployment can configure has this effect.
    """
    from services import detector_capability

    monkeypatch.setattr(
        detector_capability,
        "current_capability",
        lambda: detector_capability.CAPABILITY_PRODUCTION,
    )
