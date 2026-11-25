# torch-iris

**An OpenCL backend extension for PyTorch that enables Intel Iris Xe GPU acceleration on 64-bit Windows.**

`torch-iris` plugs into PyTorch's `PrivateUse1` backend slot and registers an `iris:0` device (along with `torch.iris` utilities). It is a **PyTorch extension**, not a standalone replacement for PyTorch—you install standard CPU PyTorch (`2.12.1`), add `torch-iris`, import both, and run standard PyTorch models and training loops on `torch.device("iris:0")`.

---

## Requirements

- **OS:** 64-bit Windows (Windows 10 / Windows 11)
- **GPU & Driver:** Intel Iris Xe Graphics with an Intel graphics driver exposing OpenCL
- **Python:** Python `3.12` (`>=3.12, <3.13`)
- **PyTorch:** `torch==2.12.1` (CPU build)

> **Note:** `torch-iris` claims PyTorch's `PrivateUse1` backend slot. It cannot be loaded in the same Python process as `torch-directml` (which also uses `PrivateUse1` and targets an older PyTorch version).

---

## Installation

### Option 1: Install from PyPI (Recommended)

On **64-bit Windows with Python 3.12**, install directly via `pip` (this automatically installs `torch==2.12.1` alongside `torch-iris`):

```powershell
py -3.12 -m venv .venv-iris
.\.venv-iris\Scripts\python.exe -m pip install torch-iris
```

### Option 2: Install the prebuilt wheel from GitHub Releases

Download `torch_iris-0.16.0-cp312-cp312-win_amd64.whl` from the [GitHub Releases](https://github.com/adityaanilraut/torch-iris/releases/tag/v0.16.0) page, then run:

```powershell
py -3.12 -m venv .venv-iris
$iris = '.\.venv-iris\Scripts\python.exe'

# 1. Install CPU PyTorch 2.12.1
& $iris -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

# 2. Install the torch-iris extension wheel
& $iris -m pip install --no-deps .\torch_iris-0.16.0-cp312-cp312-win_amd64.whl
```

### Option 3: Build from source

Requires **Visual Studio 2022 Build Tools** with the **Desktop development with C++** workload and a **Windows 10/11 SDK**. The repository bundles the Khronos OpenCL C headers (`third_party/opencl`) and loads the OpenCL runtime dynamically from your installed Intel graphics driver.

Run from an **x64 Native Tools Command Prompt for VS 2022** (or a shell with MSVC + Windows SDK configured):

```powershell
py -3.12 -m venv .venv-iris
$iris = '.\.venv-iris\Scripts\python.exe'

& $iris -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
& $iris -m pip install setuptools wheel
& $iris -m pip wheel --no-build-isolation --no-deps --wheel-dir dist .
& $iris -m pip install --force-reinstall --no-deps dist\torch_iris-0.16.0-cp312-cp312-win_amd64.whl
```

---

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

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, foreach=False)
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

---

## Key Features & Device APIs

Once imported, `torch_iris` registers `torch.iris` as a PyTorch device module and exposes:

- **Device inspection:** `is_available()`, `device_count()`, `device_name()`, `get_device_properties()`, `get_device_capability()`
- **Synchronization & streams:** `synchronize()`, `Stream()`, `Event()`, `current_stream()`, `default_stream()`
- **Allocator & memory tracking:** `memory_stats()`, `memory_allocated()`, `memory_reserved()`, `max_memory_allocated()`, `max_memory_reserved()`, `reset_peak_memory_stats()`, `empty_cache()`
- **RNG state:** `manual_seed()`, `manual_seed_all()`, `get_rng_state()`, `set_rng_state()`
- **Inference & async transfers:** Standard `model.eval()` and `torch.inference_mode()` work as expected. Pinned contiguous CPU tensors support asynchronous copies with `non_blocking=True` (call `torch_iris.synchronize()` before reading a CPU destination or measuring wall time).

### Mixed Precision (FP16 AMP)

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

### Zero-Copy DLPack Round Trips

Use the package helpers for same-context Iris tensors:

```python
capsule = torch_iris.to_dlpack(tensor)
restored = torch_iris.from_dlpack(capsule)
```

---

## Known Limitations & Performance Notes

- **No FP64 (`float64`) storage:** Intel Iris Xe OpenCL drivers do not expose `cl_khr_fp64`. Operations requiring `float64` storage are intentionally unsupported rather than silently downcast.
  - Use `torch_iris.GradScaler()` instead of `torch.amp.GradScaler("iris")` (upstream PyTorch 2.12.1 casts the scale reciprocal to `float64`).
  - For legacy `torch.jit.trace` with Iris inputs, pass `check_trace=False` (PyTorch's built-in trace checker casts outputs to `float64`) and validate outputs separately.
- **DLPack interop:** PyTorch 2.12.1's generic `torch.from_dlpack()` does not map the `iris` device type; use `torch_iris.to_dlpack()` and `torch_iris.from_dlpack()`.
- **`mem_get_info()` unavailable:** Windows OpenCL drivers do not expose a reliable free-memory query. Use `torch_iris.get_device_properties()` for driver capacity/per-buffer limits and `torch_iris.memory_stats()` for bytes tracked by the `torch-iris` allocator.
- **Performance guidance:** Large device-resident matrix multiplications and vision workloads benefit most from Iris Xe acceleration. Small tensors, recurrent layers, matrix factorizations, or transfer-heavy loops can be slower than CPU due to OpenCL launch and host/device copy overhead—always time the full step with `torch_iris.synchronize()`.

---

## Testing & Verification

Install `pytest` (and `torchvision` for vision tests) in your environment and run the unit test and CPU-reference parity suites:

```powershell
& $iris -m pip install pytest
& $iris -m pip install torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu

# Run the unit and parity test suite
& $iris -m pytest -p no:cacheprovider -q tests

# Run standalone parity scopes against the built wheel
$wheel = 'dist\torch_iris-0.16.0-cp312-cp312-win_amd64.whl'
foreach ($scope in 'forward', 'training', 'operators') {
    & $iris -m parity.run iris --scope $scope --wheel $wheel --output "parity-$scope.json"
}
```

---

## Project Structure

```text
torch-iris/
├── csrc/                  # C++20 / OpenCL backend, runtime, and kernel sources
├── examples/              # End-to-end usage examples (e.g., MNIST CNN training)
├── parity/                # CPU-reference parity runner and scope fixtures
├── src/torch_iris/        # Python device registration, AMP, and op dispatchers
├── tests/                 # Unit, capability, and parity tests
├── third_party/opencl/    # Bundled Khronos OpenCL C headers + Apache-2.0 license
├── LICENSE                # Apache License 2.0
├── NOTICE                 # Copyright, third-party attributions, and trademark notices
├── pyproject.toml         # Package metadata and build configuration
└── setup.py               # PyTorch CppExtension build script
```

---

## License & Legal

`torch-iris` is licensed under the **Apache License, Version 2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE) for full details.

- **Third-Party Software:** This repository bundles the Khronos OpenCL C headers (`third_party/opencl/CL/`), Copyright (c) 2008–2020 The Khronos Group Inc., licensed under the Apache License, Version 2.0 ([`third_party/opencl/LICENSE`](third_party/opencl/LICENSE)).
- **Trademarks & Non-Affiliation:** `torch-iris` is an independent open-source project and is not affiliated with, endorsed by, or sponsored by Intel Corporation, Meta Platforms, Inc., the PyTorch Foundation, The Linux Foundation, The Khronos Group Inc., or Microsoft Corporation. PyTorch is a trademark of The Linux Foundation; Intel and Intel Iris Xe are trademarks of Intel Corporation; OpenCL is a trademark of Apple Inc. used by permission by Khronos; Microsoft, Windows, and DirectML are trademarks of Microsoft Corporation.
