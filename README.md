# torch-iris

`torch-iris` is an OpenCL backend for PyTorch on Intel Iris Xe graphics. Import it to register `iris:0` and the `torch.iris` device APIs.

## Requirements

- **OS:** 64-bit Windows (Windows 10 / Windows 11)
- **GPU & Driver:** Intel Iris Xe Graphics with an Intel graphics driver exposing OpenCL
- **Python:** Python `3.12` (`>=3.12, <3.13`)
- **PyTorch:** `torch==2.12.1` (CPU build)

> **Note:** `torch-iris` claims PyTorch's `PrivateUse1` backend slot. It cannot be loaded in the same Python process as `torch-directml` (which also uses `PrivateUse1` and targets an older PyTorch version).

## Installation

Install the CPU build of PyTorch first, then install `torch-iris` from PyPI.

```powershell
py -3.12 -m venv .venv-iris
$iris = '.\.venv-iris\Scripts\python.exe'
& $iris -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
& $iris -m pip install torch-iris
```

You can also download a Windows x64 wheel from [GitHub Releases](https://github.com/adityaanilraut/torch-iris/releases) and pass its path to `pip install --no-deps`.

### Build from source

Requires **Visual Studio 2022 Build Tools** with the **Desktop development with C++** workload and a **Windows 10/11 SDK**. The repository bundles the Khronos OpenCL C headers (`third_party/opencl`) and loads the OpenCL runtime dynamically from your installed Intel graphics driver.

With the environment above and the x64 MSVC toolchain loaded, build from the repository root.

```powershell
& $iris -m pip install setuptools wheel
& $iris -m pip wheel --no-build-isolation --no-deps --wheel-dir dist .
& $iris -m pip install --force-reinstall --no-deps dist\torch_iris-0.18.0-cp312-cp312-win_amd64.whl
```

## Quick Start

Always `import torch_iris` **before** creating `torch.device("iris:0")` or moving tensors and modules to the device:

```python
import torch
import torch_iris

assert torch_iris.is_available(), "Intel Iris Xe OpenCL device unavailable"
device = torch.device("iris:0")
print("Using:", torch_iris.device_name())

model = torch.nn.Sequential(
    torch.nn.Linear(32, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 8),
).to(device)

inputs = torch.randn(64, 32, device=device)
targets = torch.randn(64, 8, device=device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer.zero_grad(set_to_none=True)
loss = torch.nn.functional.mse_loss(model(inputs), targets)
loss.backward()
optimizer.step()
print("Loss:", loss.item())
```

For a complete end-to-end CNN training script on MNIST (with `AdamW(fused=True)`, checkpointing, and inference), see [`examples/mnist_iris.py`](examples/mnist_iris.py):

```powershell
& $iris -m pip install torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu
& $iris examples\mnist_iris.py
```

The example saves the latest training state to `last_model.pth` and keeps `best_model.pth` for inference. Pass `--resume last_model.pth --epochs 5` to continue through epoch five, and `--report mnist.json` to record per-epoch loss and accuracy, peak allocator memory, and CPU fallback counts. The resume checkpoint includes the CPU, Iris, and minibatch shuffle RNG states.

## Key Features & Device APIs

Once imported, `torch_iris` registers `torch.iris` as a PyTorch device module and exposes:

- **Device inspection:** `is_available()`, `device_count()`, `device_name()`, `get_device_properties()`, `get_device_capability()`
- **Synchronization & streams:** `synchronize()`, `Stream()`, `Event()`, `current_stream()`, `default_stream()`
- **Allocator & memory tracking:** `memory_stats()`, `memory_allocated()`, `memory_reserved()`, `max_memory_allocated()`, `max_memory_reserved()`, `reset_peak_memory_stats()`, `empty_cache()`
- **RNG state:** `manual_seed()`, `manual_seed_all()`, `get_rng_state()`, `set_rng_state()`
- **Inference & async transfers:** Standard `model.eval()` and `torch.inference_mode()` work as expected. Pinned contiguous CPU tensors support asynchronous copies with `non_blocking=True` (call `torch_iris.synchronize()` before reading a CPU destination or measuring wall time).

### Mixed Precision (FP16 AMP)

Check `torch.float16 in torch_iris.get_amp_supported_dtype()` before enabling FP16 autocast. BF16 rounding, unary math, and selected matrix products use FP32 arithmetic on the GPU. `get_device_properties()` reports the driver's subgroup widths, the enabled subgroup kernel and FP16 capabilities, packed signed INT8 dot-product acceleration, and the per-buffer allocation limit in bytes.

Use PyTorch's autocast context with **`torch_iris.GradScaler()`**:

```python
scaler = torch_iris.GradScaler()

with torch.autocast("iris", dtype=torch.float16):
    logits = model(inputs)
    loss = criterion(logits, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### INT8 Matrix Multiplication

`torch._int_mm` accepts two signed INT8 matrices on `iris:0` and returns an INT32 matrix on the GPU. The backend uses packed DP4a dot products for compatible layouts on drivers that expose them and a strided OpenCL kernel for other supported layouts. The `out=` variant accepts a contiguous INT32 Iris tensor with the result shape.

```python
import torch
import torch_iris

torch_iris.set_cpu_fallback_enabled(False)
a = torch.randint(-128, 128, (64, 128), dtype=torch.int8).to("iris:0")
b = torch.randint(-128, 128, (128, 96), dtype=torch.int8).to("iris:0")
c = torch._int_mm(a, b)
assert c.dtype == torch.int32 and c.device.type == "iris"
```

`torch.mm`, `torch.bmm`, `torch.addmm`, and `torch.baddbmm` also accept `out_dtype=torch.float32` with FP16 or BF16 inputs. The backend converts the operands on the GPU before multiplying.

### Zero-Copy DLPack Round Trips

Use the package helpers for same-context Iris tensors:

```python
capsule = torch_iris.to_dlpack(tensor)
restored = torch_iris.from_dlpack(capsule)
```

## Reliability & CPU Fallback

The backend can run unimplemented ATen operators on CPU and records each fallback call.

- **Once-per-operator warning:** The first time an unimplemented operator falls back to CPU, `torch-iris` emits a `UserWarning` naming the operator so host/device transfers are never hidden. Set `TORCH_IRIS_FALLBACK_WARN=0` (or call `torch_iris.set_fallback_warning_enabled(False)`) to silence warnings.
- **Inspect fallback usage:** Call `torch_iris.fallback_stats()` to see a `{op_name: count}` dictionary of every operator that fell back to CPU, and `torch_iris.reset_fallback_stats()` to clear counters:

```python
torch_iris.reset_fallback_stats()
# ... run your model or training loop ...
print(torch_iris.fallback_stats())  # {} when 100% native on Iris Xe
```

- **Strict GPU-only mode:** Disable CPU fallback via the environment variable `TORCH_IRIS_CPU_FALLBACK=0` or at runtime with `torch_iris.set_cpu_fallback_enabled(False)` (check with `torch_iris.is_cpu_fallback_enabled()`). In strict mode, unimplemented operators raise `NotImplementedError` immediately.
- **Defensive OpenCL memory handling:** Allocations exceeding `max_mem_alloc_size` raise a clear `torch.OutOfMemoryError`, and out-of-memory allocations automatically flush and reclaim cached device buffers before failing.

## Known Limitations & Performance Notes

- **Storage limits.** `float64` and `complex128` allocations raise an error.
  - Use `torch_iris.GradScaler()` instead of `torch.amp.GradScaler("iris")` (upstream PyTorch 2.12.1 casts the scale reciprocal to `float64`).
  - For legacy `torch.jit.trace` with Iris inputs, pass `check_trace=False` (PyTorch's built-in trace checker casts outputs to `float64`) and validate outputs separately.
- **DLPack interop:** PyTorch 2.12.1's generic `torch.from_dlpack()` does not map the `iris` device type; use `torch_iris.to_dlpack()` and `torch_iris.from_dlpack()`.
- **`mem_get_info()` unavailable:** Windows OpenCL drivers do not expose a reliable free-memory query. Use `torch_iris.get_device_properties()` for driver capacity/per-buffer limits and `torch_iris.memory_stats()` for bytes tracked by the `torch-iris` allocator.
- **Performance guidance:** Large device-resident matrix multiplications and vision workloads benefit most from Iris Xe acceleration. Small tensors, recurrent layers, matrix factorizations, or transfer-heavy loops can be slower than CPU due to OpenCL launch and host/device copy overhead—always time the full step with `torch_iris.synchronize()`.

## Testing & Verification

Install `pytest` (and `torchvision` for vision tests) in your environment and run the unit test and CPU-reference parity suites:

```powershell
& $iris -m pip install pytest
& $iris -m pip install torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu

# Run the unit and parity test suite
& $iris -m pytest -p no:cacheprovider -q tests

# Run standalone parity scopes against the built wheel
$wheel = 'dist\torch_iris-0.18.0-cp312-cp312-win_amd64.whl'
foreach ($scope in 'forward', 'training', 'operators') {
    & $iris -m parity.run iris --scope $scope --wheel $wheel --output "parity-$scope.json"
    if ($LASTEXITCODE -ne 0) { throw "Parity failed for $scope" }
}
```

The hosted Windows job builds, installs, and imports the wheel and publishes its SHA-256. Physical GPU verification runs separately. Parity reports count per-case CPU fallback calls and distinguish native kernels, on-device emulation, CPU fallback, blocked cases, and failures. If FP16 is unavailable, its cases are marked `hardware_blocked` and the scope exits incomplete. Set `$env:TORCH_IRIS_CPU_FALLBACK = '0'` for GPU-only verification. Set `$env:TORCH_IRIS_MASK_OPTIONAL_FEATURES = '1'` before starting a separate process to verify portable kernels with FP16, subgroup kernels, and packed INT8 acceleration disabled. Supply a CUDA reference only when captured on a physical NVIDIA GPU.

Run `& $iris -m parity.benchmark --cpu-threads 4 --output benchmark.json` for synchronized FP32 measurements of matrix multiplication, MLP and CNN training steps, causal attention, and a matrix multiplication with input and output copies included.

| Workload | Exercised scope | Executable coverage |
| --- | --- | --- |
| MLP and matrix products | FP32 broadcast and strided inputs, empty dimensions, gradients, fused bias, `out=`; FP16 and BF16 products with FP32 output; signed INT8 to INT32 | [Forward parity](parity/scope-v1.json), [matrix tests](tests/test_reliability_fallback.py) |
| CNN and optimization | FP32 grouped/dilated convolution, pooling, normalization, training and gradients; SGD, Adam and AdamW state and updates; FP16 autocast | [Training parity](parity/scope-training-v1.json), [MNIST checkpoint test](tests/test_mnist_checkpoint.py) |
| Attention | FP32 causal and explicit masks, strided inputs, grouped-query attention, forward/backward and decoding; FP16 where available | [Operator parity](parity/scope-operators-v1.json), [attention tests](tests/test_iris.py) |
| Typed pointwise and linalg | BF16 and complex64 unary values, predicates, gradients and applicable output variants; BF16 rounding and batched solve/inverse regressions | [Unary tests](tests/test_typed_unary.py), [linalg tests](tests/test_capability_fixes.py) |

To generate the PyTorch operator inventory for developing new parity cases, install `expecttest` and run:

```powershell
& $iris -m pip install expecttest
& $iris -m parity.inventory --output parity/inventory-v1.json
```

## License & Legal

`torch-iris` is licensed under the **Apache License, Version 2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE) for full details.

- **Third-Party Software:** This repository bundles the Khronos OpenCL C headers (`third_party/opencl/CL/`), Copyright (c) 2008–2020 The Khronos Group Inc., licensed under the Apache License, Version 2.0 ([`third_party/opencl/LICENSE`](third_party/opencl/LICENSE)).
- **Trademarks & Non-Affiliation:** `torch-iris` is an independent open-source project and is not affiliated with, endorsed by, or sponsored by Intel Corporation, Meta Platforms, Inc., the PyTorch Foundation, The Linux Foundation, The Khronos Group Inc., or Microsoft Corporation. PyTorch is a trademark of The Linux Foundation; Intel and Intel Iris Xe are trademarks of Intel Corporation; OpenCL is a trademark of Apple Inc. used by permission by Khronos; Microsoft, Windows, and DirectML are trademarks of Microsoft Corporation.
