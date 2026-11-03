"""Which device the model worker runs the neural detectors on, and proof of it.

Selected once, in the worker, before either model is loaded:

``cpu``   (the default) never initializes CUDA.
``cuda``  requires a CUDA build of torch, a visible device, and both models
          verified on it after warm-up; otherwise the worker is not ready.
          Never silently CPU.
``auto``  CUDA when it initializes and both models place and warm up on it,
          else CPU with the reason recorded - decided before any job is
          accepted, and never described as acceleration.

Only placement changes. Tokenization, rules, thresholds and everything outside
the two neural models stay where they were, and the models run in FP32: no
half precision, mixed precision, compilation or quantization.

Nothing here reads, logs or returns model input.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional

DEVICE_ENV_VAR = "BIOBLOCK_MODEL_DEVICE"
CPU_FALLBACK_ENV_VAR = "BIOBLOCK_CUDA_ALLOW_CPU_FALLBACK"
MIN_BATCH_ENV_VAR = "BIOBLOCK_CUDA_MIN_BATCH_SIZE"
OOM_RETRIES_ENV_VAR = "BIOBLOCK_CUDA_OOM_MAX_RETRIES"

DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_AUTO = "auto"
DEVICES = (DEVICE_AUTO, DEVICE_CPU, DEVICE_CUDA)
#: CPU until the privacy-parity gate has passed on the deployment's own GPU.
DEFAULT_DEVICE = DEVICE_CPU
#: Texts per forward pass on CUDA unless PHI_MODEL_BATCH_SIZE says otherwise.
#: Measured on a 4 GB GTX 1650 Ti: batches of 1, 2, 4 and 8 ran within a few
#: per cent of each other and produced byte-identical outputs, but 8 let the
#: caching allocator reserve the whole card (0 MB free) where 2 left 1.1 GB.
CUDA_DEFAULT_BATCH_SIZE = 2

ERROR_INVALID_DEVICE = "invalid_model_device"
ERROR_CUDA_BUILD = "cuda_torch_build_unavailable"
ERROR_CUDA_UNAVAILABLE = "cuda_device_unavailable"
ERROR_CUDA_INIT = "cuda_initialization_failed"
ERROR_CUDA_PLACEMENT = "cuda_model_placement_failed"
ERROR_CUDA_OOM = "cuda_out_of_memory"


class DeviceSelectionError(RuntimeError):
    def __init__(self, error_code: str):
        super().__init__(error_code)
        self.error_code = error_code


def requested_device() -> str:
    value = (os.getenv(DEVICE_ENV_VAR) or DEFAULT_DEVICE).strip().lower()
    if value not in DEVICES:
        raise DeviceSelectionError(ERROR_INVALID_DEVICE)
    return value


def cpu_fallback_allowed() -> bool:
    """Whether a CUDA failure after startup may rerun work on CPU. Off by default."""
    return os.getenv(CPU_FALLBACK_ENV_VAR, "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _bounded_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise DeviceSelectionError(ERROR_INVALID_DEVICE) from exc
    return max(low, min(high, value))


def min_batch_size() -> int:
    return _bounded_int(MIN_BATCH_ENV_VAR, 1, 1, 64)


def max_oom_retries() -> int:
    return _bounded_int(OOM_RETRIES_ENV_VAR, 4, 0, 8)


@dataclass
class DeviceState:
    """What the worker reports about placement. Safe to publish: no paths."""

    requested: str = DEFAULT_DEVICE
    selected: str = DEVICE_CPU
    torch_build: str = ""
    cuda_build: bool = False
    cuda_runtime: Optional[str] = None
    gpu_visible: bool = False
    gpu_name: Optional[str] = None
    total_vram_bytes: Optional[int] = None
    free_vram_bytes: Optional[int] = None
    allocated_vram_bytes: Optional[int] = None
    reserved_vram_bytes: Optional[int] = None
    peak_allocated_vram_bytes: Optional[int] = None
    model_devices: Dict[str, str] = field(default_factory=dict)
    acceleration_active: bool = False
    degraded: bool = False
    #: Why CUDA is not in use, or why the runtime is degraded. A code.
    reason: Optional[str] = None
    precision: str = "fp32"
    oom_events: int = 0
    cpu_fallbacks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _torch():
    import torch

    return torch


def probe(requested: str) -> DeviceState:
    """Choose the device, touching CUDA only when it may be used."""
    torch = _torch()
    state = DeviceState(requested=requested, torch_build=str(torch.__version__))
    state.cuda_build = torch.version.cuda is not None
    state.cuda_runtime = torch.version.cuda
    if requested == DEVICE_CPU:
        state.reason = "cpu_requested"
        return state
    failure: Optional[str] = None
    if not state.cuda_build:
        failure = ERROR_CUDA_BUILD
    else:
        try:
            state.gpu_visible = (
                bool(torch.cuda.is_available()) and torch.cuda.device_count() > 0
            )
            if not state.gpu_visible:
                failure = ERROR_CUDA_UNAVAILABLE
            else:
                properties = torch.cuda.get_device_properties(0)
                state.gpu_name = str(properties.name)
                state.total_vram_bytes = int(properties.total_memory)
                # A real kernel, synchronized: initialization is not proven
                # by is_available() alone.
                sample = torch.ones((8, 8), device=DEVICE_CUDA)
                (sample @ sample).sum().item()
                torch.cuda.synchronize()
                del sample
        except Exception:
            failure = ERROR_CUDA_INIT
    if failure is None:
        state.selected = DEVICE_CUDA
        _deterministic(torch)
        return state
    if requested == DEVICE_CUDA:
        raise DeviceSelectionError(failure)
    state.reason = failure
    return state


def _deterministic(torch) -> None:
    """Deterministic kernels where supported, without refusing unsupported ones."""
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass


def parameter_devices(module: Any) -> set:
    return {parameter.device.type for parameter in module.parameters()}


def verify_placement(modules: Dict[str, Any], device: str) -> Dict[str, str]:
    """Every parameter of every model on ``device``, or an error."""
    placed: Dict[str, str] = {}
    for name, module in modules.items():
        devices = parameter_devices(module)
        if devices != {device}:
            raise DeviceSelectionError(
                ERROR_CUDA_PLACEMENT
                if device == DEVICE_CUDA
                else "cpu_model_placement_failed"
            )
        placed[name] = device
    return placed


def refresh_memory(state: DeviceState) -> DeviceState:
    if state.selected != DEVICE_CUDA:
        for key in (
            "free_vram_bytes",
            "allocated_vram_bytes",
            "reserved_vram_bytes",
            "peak_allocated_vram_bytes",
        ):
            setattr(state, key, None)
        return state
    torch = _torch()
    try:
        free, _total = torch.cuda.mem_get_info()
        state.free_vram_bytes = int(free)
        state.allocated_vram_bytes = int(torch.cuda.memory_allocated())
        state.reserved_vram_bytes = int(torch.cuda.memory_reserved())
        state.peak_allocated_vram_bytes = int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return state


def is_out_of_memory(exc: BaseException) -> bool:
    """A CUDA out-of-memory error, told apart from every other failure."""
    seen: Iterable[BaseException] = _chain(exc)
    try:
        oom_type = _torch().cuda.OutOfMemoryError
    except Exception:
        oom_type = ()
    for item in seen:
        if oom_type and isinstance(item, oom_type):
            return True
        if isinstance(item, RuntimeError) and "CUDA out of memory" in str(item)[:200]:
            return True
    return False


def _chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def release_cuda_cache() -> None:
    try:
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
