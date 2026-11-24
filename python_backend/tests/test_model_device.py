"""Device selection and CUDA out-of-memory recovery, without a GPU or torch.

A stand-in for torch plays each state a machine can be in - no CUDA build, a
build without a device, a device that fails its first kernel, a healthy one -
so every branch of the selection runs in the ordinary suite. The real CUDA
path is exercised by the opt-in parity evaluation (docs/GPU_RUNTIME.md).
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import local_model_detectors as models  # noqa: E402
from services import model_device as md  # noqa: E402
from services.local_model_detectors import DetectorConfig, LocalModelError  # noqa: E402


class _Tensor:
    def __matmul__(self, other):
        return self

    def sum(self):
        return self

    def item(self):
        return 64.0


class _Parameter:
    def __init__(self, device):
        self.device = SimpleNamespace(type=device)


class _Module:
    def __init__(self, *devices):
        self._devices = devices

    def parameters(self):
        return [_Parameter(device) for device in self._devices]


def _fake_torch(cuda_build=True, available=True, kernel_fails=False):
    touched = {"cuda": False}

    def ones(shape, device):
        touched["cuda"] = True
        if kernel_fails:
            raise RuntimeError("CUDA error: no kernel image is available")
        return _Tensor()

    cuda = SimpleNamespace(
        is_available=lambda: touched.__setitem__("cuda", True) or available,
        device_count=lambda: 1 if available else 0,
        get_device_properties=lambda index: SimpleNamespace(
            name="Test GPU", total_memory=4 * 2**30
        ),
        synchronize=lambda: None,
        OutOfMemoryError=type("OutOfMemoryError", (RuntimeError,), {}),
    )
    backends = SimpleNamespace(
        cudnn=SimpleNamespace(benchmark=True, deterministic=False, allow_tf32=True),
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
    )
    return (
        SimpleNamespace(
            __version__="2.4.1+cu121" if cuda_build else "2.4.1+cpu",
            version=SimpleNamespace(cuda="12.1" if cuda_build else None),
            cuda=cuda,
            ones=ones,
            backends=backends,
        ),
        touched,
    )


@pytest.fixture
def torch_state(monkeypatch):
    def install(**kwargs):
        torch, touched = _fake_torch(**kwargs)
        monkeypatch.setattr(md, "_torch", lambda: torch)
        return torch, touched

    return install


# -- selection -----------------------------------------------------------------------


def test_the_default_is_cpu_until_parity_has_been_shown(monkeypatch):
    monkeypatch.delenv(md.DEVICE_ENV_VAR, raising=False)
    assert md.requested_device() == md.DEVICE_CPU


def test_an_unknown_device_is_a_configuration_error(monkeypatch):
    monkeypatch.setenv(md.DEVICE_ENV_VAR, "tpu")
    with pytest.raises(md.DeviceSelectionError) as caught:
        md.requested_device()
    assert caught.value.error_code == md.ERROR_INVALID_DEVICE


def test_cpu_never_initializes_cuda(torch_state):
    _torch, touched = torch_state()
    state = md.probe(md.DEVICE_CPU)

    assert state.selected == md.DEVICE_CPU and not state.acceleration_active
    assert touched["cuda"] is False and state.reason == "cpu_requested"


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"cuda_build": False}, md.ERROR_CUDA_BUILD),
        ({"available": False}, md.ERROR_CUDA_UNAVAILABLE),
        ({"kernel_fails": True}, md.ERROR_CUDA_INIT),
    ],
)
def test_cuda_requested_fails_clearly_and_never_selects_cpu(torch_state, kwargs, code):
    torch_state(**kwargs)
    with pytest.raises(md.DeviceSelectionError) as caught:
        md.probe(md.DEVICE_CUDA)
    assert caught.value.error_code == code


def test_auto_selects_a_healthy_cuda_device_with_fp32_and_deterministic_kernels(
    torch_state,
):
    torch, _touched = torch_state()
    state = md.probe(md.DEVICE_AUTO)

    assert state.selected == md.DEVICE_CUDA and state.gpu_name == "Test GPU"
    assert state.total_vram_bytes == 4 * 2**30 and state.precision == "fp32"
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.deterministic is True


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"cuda_build": False}, md.ERROR_CUDA_BUILD),
        ({"available": False}, md.ERROR_CUDA_UNAVAILABLE),
        ({"kernel_fails": True}, md.ERROR_CUDA_INIT),
    ],
)
def test_auto_falls_back_to_cpu_before_any_job_and_says_why(torch_state, kwargs, code):
    torch_state(**kwargs)
    state = md.probe(md.DEVICE_AUTO)

    assert state.selected == md.DEVICE_CPU and state.reason == code
    assert state.acceleration_active is False


def test_placement_is_verified_parameter_by_parameter():
    assert md.verify_placement({"a": _Module("cuda", "cuda")}, "cuda") == {"a": "cuda"}
    with pytest.raises(md.DeviceSelectionError) as caught:
        md.verify_placement({"a": _Module("cuda", "cpu")}, "cuda")
    assert caught.value.error_code == md.ERROR_CUDA_PLACEMENT


def test_cpu_fallback_after_startup_is_off_unless_configured(monkeypatch):
    monkeypatch.delenv(md.CPU_FALLBACK_ENV_VAR, raising=False)
    assert md.cpu_fallback_allowed() is False
    monkeypatch.setenv(md.CPU_FALLBACK_ENV_VAR, "true")
    assert md.cpu_fallback_allowed() is True


def test_the_device_state_publishes_no_path():
    state = md.DeviceState(requested="auto", selected="cpu", torch_build="2.4.1+cpu")
    assert all(
        "\\" not in str(value) and "/" not in str(value)
        for value in state.to_dict().values()
    )


# -- the worker loads each model once and verifies where it is --------------------


def test_the_worker_selects_loads_once_verifies_and_warms_up(monkeypatch, torch_state):
    from services import model_worker

    torch_state()
    monkeypatch.setenv(md.DEVICE_ENV_VAR, "cpu")
    loads = {"stanford": 0, "gliner": 0}
    calls = []

    class _Detector:
        def __init__(self, config):
            pass

        def detect(self, text):
            calls.append(len(text))
            return []

    def stanford():
        loads["stanford"] += 1
        return SimpleNamespace(model=_Module("cpu"))

    def gliner():
        loads["gliner"] += 1
        return _Module("cpu")

    stanford.cache_clear = gliner.cache_clear = lambda: None
    monkeypatch.setattr(models, "load_stanford_pipeline", stanford)
    monkeypatch.setattr(models, "load_gliner_model", gliner)
    monkeypatch.setattr(models, "StanfordClinicalDetector", _Detector)
    monkeypatch.setattr(models, "GlinerPiiDetector", _Detector)
    monkeypatch.setattr(models, "calibrated_config_for", lambda name: None)

    worker = model_worker._Worker()
    worker.warm()
    status = worker.status()

    assert status["ready"] and status["load_count"] == 1
    assert loads == {"stanford": 1, "gliner": 1}
    assert status["device"]["model_devices"] == {"stanford": "cpu", "gliner": "cpu"}
    assert len(calls) == 2  # one synthetic warm-up per model
    assert status["device"]["acceleration_active"] is False


def test_cuda_requested_but_unavailable_leaves_the_worker_not_ready(
    monkeypatch, torch_state
):
    from services import model_worker

    torch_state(available=False)
    monkeypatch.setenv(md.DEVICE_ENV_VAR, "cuda")
    worker = model_worker._Worker()
    worker.warm()

    assert worker.status()["ready"] is False
    assert worker.status()["error_code"] == md.ERROR_CUDA_UNAVAILABLE


# -- out of memory ---------------------------------------------------------------------


def _oom():
    return RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


def test_an_oom_batch_is_split_in_half_and_answers_stay_in_order(monkeypatch):
    monkeypatch.setattr(md, "release_cuda_cache", lambda: None)
    sizes = []

    def runner(texts):
        sizes.append(len(texts))
        if len(texts) > 2:
            raise _oom()
        return [[{"text": text}] for text in texts]

    chunks = [(index * 10, f"chunk-{index}") for index in range(8)]
    answers = models._run_batched(chunks, DetectorConfig(batch_size=8), runner, "probe")

    assert [answer[0]["text"] for answer in answers] == [text for _o, text in chunks]
    assert sizes[0] == 8 and max(sizes[1:]) <= 4 and sizes.count(2) == 4


def test_oom_retries_are_bounded_and_end_in_an_error_not_a_result(monkeypatch):
    monkeypatch.setattr(md, "release_cuda_cache", lambda: None)
    monkeypatch.setenv(md.OOM_RETRIES_ENV_VAR, "2")
    attempts = []

    def runner(texts):
        attempts.append(len(texts))
        raise _oom()

    with pytest.raises(LocalModelError) as caught:
        models._run_batched(
            [(i, f"c{i}") for i in range(8)],
            DetectorConfig(batch_size=8),
            runner,
            "probe",
        )
    assert caught.value.error_code == md.ERROR_CUDA_OOM
    assert len(attempts) <= 1 + 2 + 1


def test_oom_at_the_minimum_batch_is_an_error(monkeypatch):
    monkeypatch.setattr(md, "release_cuda_cache", lambda: None)
    with pytest.raises(LocalModelError) as caught:
        models._run_batched(
            [(0, "only")],
            DetectorConfig(batch_size=1),
            lambda texts: (_ for _ in ()).throw(_oom()),
            "probe",
        )
    assert caught.value.error_code == md.ERROR_CUDA_OOM


def test_an_ordinary_failure_is_not_mistaken_for_oom():
    def runner(texts):
        raise ValueError("shape mismatch")

    with pytest.raises(LocalModelError) as caught:
        models._run_batched(
            [(0, "a"), (5, "b")],
            DetectorConfig(batch_size=2),
            runner,
            "stanford_inference_failed",
        )
    assert caught.value.error_code == "stanford_inference_failed"


def test_the_oom_record_holds_sizes_only(monkeypatch):
    monkeypatch.setattr(md, "release_cuda_cache", lambda: None)
    models.OOM_EVENTS.clear()

    def runner(texts):
        if len(texts) > 1:
            raise _oom()
        return [[] for _ in texts]

    models._run_batched(
        [(0, "Patient Zelquin"), (9, "MRN 4821")],
        DetectorConfig(batch_size=2),
        runner,
        "probe",
    )
    allowed = {"batch_texts", "allocated", "reserved"}
    assert models.OOM_EVENTS and set(models.OOM_EVENTS[0]) <= allowed
    assert "Zelquin" not in repr(models.OOM_EVENTS)


def test_a_worker_oom_without_fallback_fails_the_request_and_keeps_the_device(
    monkeypatch,
):
    from services import model_worker

    monkeypatch.delenv(md.CPU_FALLBACK_ENV_VAR, raising=False)
    worker = model_worker._Worker()
    worker._ready = True
    worker.device = md.DeviceState(
        requested="cuda", selected="cuda", acceleration_active=True
    )

    class _Oom:
        def detect(self, text):
            raise LocalModelError(md.ERROR_CUDA_OOM, 503)

    worker._detectors = {"stanford": _Oom()}
    response = worker.handle(
        {
            "v": 2,
            "gen": worker.generation,
            "id": 3,
            "op": "detect",
            "detector": "stanford",
            "text": "synthetic",
        }
    )

    assert response == {"id": 3, "ok": False, "error_code": md.ERROR_CUDA_OOM}
    assert worker.device.selected == "cuda" and worker.device.oom_events == 1


def test_a_worker_oom_with_fallback_reruns_the_whole_request_on_cpu(monkeypatch):
    from services import model_worker

    monkeypatch.setenv(md.CPU_FALLBACK_ENV_VAR, "true")
    worker = model_worker._Worker()
    worker._ready = True
    worker.device = md.DeviceState(
        requested="cuda", selected="cuda", acceleration_active=True
    )

    class _Oom:
        def detect(self, text):
            raise LocalModelError(md.ERROR_CUDA_OOM, 503)

    class _Cpu:
        def detect(self, text):
            return []

    worker._detectors = {"stanford": _Oom()}

    def reload(device):
        assert device == "cpu"
        worker._detectors = {"stanford": _Cpu()}

    monkeypatch.setattr(worker, "_reset_models", lambda: None)
    monkeypatch.setattr(worker, "_load", reload)
    response = worker.handle(
        {
            "v": 2,
            "gen": worker.generation,
            "id": 4,
            "op": "detect",
            "detector": "stanford",
            "text": "synthetic",
        }
    )

    assert response["ok"] is True and response["degraded"] is True
    assert response["device"] == "cpu"
    assert worker.device.degraded and worker.device.reason == md.ERROR_CUDA_OOM
    assert (
        worker.device.acceleration_active is False and worker.device.cpu_fallbacks == 1
    )


# -- readiness and status ---------------------------------------------------------------


def _client(ready, device=None):
    return SimpleNamespace(
        is_ready=lambda: ready,
        runtime={
            "pid": 7,
            "load_count": 1,
            "model_batch_size": 8,
            "device": device or {},
        },
        status=lambda: {"generation": 3, "restarts": 1},
    )


def test_readiness_follows_the_live_worker_not_just_startup(monkeypatch):
    from services import model_lifecycle as ml

    monkeypatch.setattr(ml, "current_client", lambda: _client(False))
    assert ml.LifecycleState(state=ml.STATE_READY).is_ready() is False
    monkeypatch.setattr(ml, "current_client", lambda: _client(True))
    assert ml.LifecycleState(state=ml.STATE_READY).is_ready() is True
    assert ml.LifecycleState(state=ml.STATE_PREPARING).is_ready() is False


def test_status_separates_build_visibility_and_active_acceleration(monkeypatch):
    from services import model_lifecycle as ml

    built_not_used = {
        "requested": "cpu",
        "selected": "cpu",
        "cuda_build": True,
        "gpu_visible": False,
        "model_devices": {"stanford": "cpu", "gliner": "cpu"},
    }
    monkeypatch.setattr(ml, "current_client", lambda: _client(True, built_not_used))
    status = ml.inference_status()
    assert status["cuda_build"] is True and status["acceleration_active"] is False
    assert status["status"] == ml.STATUS_STANDARD
    assert (status["worker_generation"], status["model_load_count"]) == (3, 1)

    active = {
        **built_not_used,
        "requested": "cuda",
        "selected": "cuda",
        "gpu_visible": True,
        "acceleration_active": True,
        "model_devices": {"stanford": "cuda", "gliner": "cuda"},
    }
    monkeypatch.setattr(ml, "current_client", lambda: _client(True, active))
    status = ml.inference_status()
    assert status["status"] == ml.STATUS_ACCELERATED
    assert (status["stanford_device"], status["gliner_device"]) == ("cuda", "cuda")

    degraded = {
        **active,
        "selected": "cpu",
        "acceleration_active": False,
        "degraded": True,
        "reason": md.ERROR_CUDA_OOM,
    }
    monkeypatch.setattr(ml, "current_client", lambda: _client(True, degraded))
    assert ml.inference_status()["status"] == ml.STATUS_DEGRADED


def test_every_answer_carries_current_vram_so_the_report_never_goes_stale(monkeypatch):
    from services import model_worker

    worker = model_worker._Worker()
    worker._ready = True
    worker.device = md.DeviceState(
        requested="cuda", selected="cuda", acceleration_active=True
    )

    def refresh(state):
        state.allocated_vram_bytes, state.peak_allocated_vram_bytes = 7, 9
        return state

    monkeypatch.setattr(md, "refresh_memory", refresh)
    worker._detectors = {"stanford": SimpleNamespace(detect=lambda text: [])}
    response = worker.handle(
        {
            "v": 2,
            "gen": worker.generation,
            "id": 8,
            "op": "detect",
            "detector": "stanford",
            "text": "synthetic",
        }
    )

    assert response["memory"]["allocated_vram_bytes"] == 7
    assert response["memory"]["peak_allocated_vram_bytes"] == 9
