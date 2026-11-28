"""Register Intel Iris Xe as a PyTorch device."""

from contextlib import contextmanager
from importlib.metadata import version as _distribution_version
import sys
from types import SimpleNamespace

import torch

__version__ = _distribution_version("torch-iris")

if torch.__version__.split("+", 1)[0] != "2.12.1":
    raise ImportError(f"torch-iris requires PyTorch 2.12.1, found {torch.__version__}")

backend = torch._C._get_privateuse1_backend_name()
if backend not in {"privateuseone", "iris"} or "torch_directml" in sys.modules:
    raise ImportError("torch-iris requires the PrivateUse1 slot. Use DirectML in a separate process.")

from . import _C
from . import amp, ops
from .amp import GradScaler

to_dlpack = _C.to_dlpack
from_dlpack = _C.from_dlpack


def is_available():
    """Return whether the OpenCL runtime reports an Intel Iris Xe GPU."""
    return _C.is_available()


def device_count():
    return int(is_available())


def current_device():
    return 0


@contextmanager
def device(index):
    """Select the single Iris device for PyTorch storage transfers."""
    _device(index)
    yield


def _device(index=None):
    selected = torch.device("iris:0") if index is None else (
        torch.device("iris", index) if isinstance(index, int) else torch.device(index)
    )
    if selected.type != "iris" or selected.index not in (None, 0):
        raise ValueError(f"Expected iris:0, got {selected}")
    return torch.device("iris:0")


def set_device(device):
    _device(device)


def device_name():
    """Return the OpenCL name of the selected GPU."""
    return _C.device_name()


def get_device_name(device=None):
    _device(device)
    return device_name()


def get_device_properties(device=None):
    _device(device)
    return SimpleNamespace(**_C._device_properties())


def get_device_capability(device=None):
    """Return the OpenCL device version as (major, minor)."""
    opencl = get_device_properties(device).opencl_version.split()[1]
    return tuple(map(int, opencl.split(".")))


def synchronize(device=None):
    """Wait for all Iris streams to finish."""
    if device is not None:
        _device(device)
    _C.synchronize()


def empty_cache():
    """Return unused device buffers to the OpenCL driver."""
    _C.empty_cache()


def memory_stats(device=None):
    """Return live and cached device bytes and buffer allocation counters."""
    if device is not None:
        _device(device)
    return _C.memory_stats()


def memory_allocated(device=None):
    return memory_stats(device)["allocated_bytes"]


def memory_reserved(device=None):
    return memory_stats(device)["reserved_bytes"]


def max_memory_allocated(device=None):
    return memory_stats(device)["peak_allocated_bytes"]


def max_memory_reserved(device=None):
    return memory_stats(device)["peak_reserved_bytes"]


def reset_peak_memory_stats(device=None):
    _device(device)
    _C._reset_peak_memory_stats()


def is_initialized():
    return _C._is_initialized()


def init():
    _C._init()


def Stream(device=None, priority=0):
    return torch.Stream(device=_device(device), priority=priority)


def Event(enable_timing=False):
    return torch.Event(device=_device(), enable_timing=enable_timing)


def current_stream(device=None):
    _device(device)
    return torch.accelerator.current_stream(0)


def default_stream(device=None):
    _device(device)
    return torch.Stream(stream_id=0, device_index=0, device_type=current_stream().device_type)


def stream(selected):
    if selected is None:
        return device(None)
    _device(selected.device)
    return selected


def _is_in_bad_fork():
    # The backend is Windows-only, where multiprocessing starts a fresh process.
    return False


def manual_seed_all(seed):
    """Reset the Philox seed and counter for the Iris device."""
    _C._manual_seed(seed % 2**64)


manual_seed = manual_seed_all


def seed():
    return _C._default_generator().seed()


def initial_seed():
    return _C._default_generator().initial_seed()


def get_rng_state(device=None):
    _device(device)
    return _C._default_generator().get_state()


def set_rng_state(new_state, device=None):
    _device(device)
    _C._default_generator().set_state(new_state)


def get_amp_supported_dtype():
    """Return the autocast dtypes supported by the OpenCL device."""
    return [torch.float16] if _C._has_fp16() else []


def is_bf16_supported(including_emulation=True):
    """Report BF16 storage and FP32-based arithmetic emulation availability."""
    return including_emulation and is_available()


def fallback_stats():
    """Return a dict mapping ATen operator names to CPU fallback invocation counts."""
    return dict(_C.fallback_stats())


def reset_fallback_stats():
    """Reset CPU fallback counters and once-per-operator warning tracking."""
    _C.reset_fallback_stats()


def is_cpu_fallback_enabled():
    """Return whether automatic CPU fallback is enabled for unimplemented ops."""
    return bool(_C.is_cpu_fallback_enabled())


def set_cpu_fallback_enabled(enabled: bool):
    """Enable or disable automatic CPU fallback (False enables strict GPU-only mode)."""
    _C.set_cpu_fallback_enabled(bool(enabled))


def is_fallback_warning_enabled():
    """Return whether once-per-operator CPU fallback warnings are enabled."""
    return bool(_C.is_fallback_warning_enabled())


def set_fallback_warning_enabled(enabled: bool):
    """Enable or disable once-per-operator CPU fallback warnings."""
    _C.set_fallback_warning_enabled(bool(enabled))


if backend != "iris":
    torch.utils.rename_privateuse1_backend("iris")
    torch._register_device_module("iris", sys.modules[__name__])

    from . import (
        _compile, _ctc, _detection, _dispatcher_ops, _factorizations, _grid, _linalg, _model_ops, _random_ops,
        _reductions, _sparse, _spatial_ops, _special, _spectral, _tensor_ops,
    )

    _aten_library = torch.library.Library("aten", "IMPL", "PrivateUse1")
    _model_ops.register(_aten_library)
    _linalg.register(_aten_library)
    _linalg_autograd_library = _factorizations.register(_aten_library)
    _spectral.register(_aten_library)
    _tensor_ops.register(_aten_library)
    _dispatcher_ops.register(_aten_library)
    _random_ops.register(_aten_library)
    _scan_autograd_library = _reductions.register(_aten_library)
    _spatial_ops.register(_aten_library)
    _special.register(_aten_library)
    _ctc.register(_aten_library)
    _grid_autograd_library = _grid.register(_aten_library)
    _vision_libraries = _detection.register()
    _sparse_library = _sparse.register()
    _compile.register()

    from torch import _tensor_str

    def _formatter_dtype(tensor, original=_tensor_str.tensor_totype):
        # PyTorch already uses FP32 for formatting devices without FP64.
        return tensor.float() if tensor.device.type == "iris" else original(tensor)

    _tensor_str.tensor_totype = _formatter_dtype
