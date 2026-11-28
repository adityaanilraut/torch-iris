"""Time FP32 workloads on the local CPU and physical Iris Xe."""

import argparse
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import torch
import torch.nn.functional as F
import torch_iris

from examples.mnist_iris import training_state


WARMUPS = 5
REPETITIONS = 15


def time_step(step, synchronize):
    for _ in range(WARMUPS):
        step()
    synchronize()
    times = []
    for _ in range(REPETITIONS):
        start = perf_counter()
        step()
        synchronize()
        times.append(1000 * (perf_counter() - start))
    return {"median_ms": median(times), "samples_ms": times}


def training_step(model, optimizer, inputs, targets, criterion):
    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

    return step


def measure(device):
    # Generate equivalent inputs and model parameters on the CPU for both runs.
    torch.manual_seed(2026)
    a, b = (torch.randn(1024, 1024).to(device) for _ in range(2))
    x = torch.randn(128, 256).to(device)
    mlp_targets = torch.randn(128, 64).to(device)
    mlp = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.ReLU(),
                              torch.nn.Linear(512, 64)).to(device)
    sgd = torch.optim.SGD(mlp.parameters(), lr=.01)
    cnn, adamw, _ = training_state(device)
    # Use the same objective for the CPU and Iris steps without backend-specific masks.
    cnn.classifier[3].p = 0.
    images = torch.randn(32, 1, 28, 28).to(device)
    labels = torch.randint(0, 10, (32,)).to(device)
    query, key, value = (torch.randn(2, 4, 128, 64).to(device) for _ in range(3))
    host_a, host_b = (torch.randn(384, 384) for _ in range(2))
    synchronize = torch_iris.synchronize if device == "iris:0" else lambda: None
    return {
        "matmul": time_step(lambda: torch.mm(a, b), synchronize),
        "mlp_training": time_step(training_step(mlp, sgd, x, mlp_targets, F.mse_loss), synchronize),
        "cnn_training": time_step(training_step(cnn, adamw, images, labels, F.cross_entropy), synchronize),
        "attention": time_step(lambda: F.scaled_dot_product_attention(query, key, value, is_causal=True),
                               synchronize),
        "matmul_with_transfers": time_step(lambda: torch.mm(host_a.to(device), host_b.to(device)).cpu(),
                                           synchronize),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-threads", type=int, default=torch.get_num_threads())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if not torch_iris.is_available():
        raise SystemExit("The CPU/Iris comparison requires a physical Intel Iris Xe OpenCL GPU.")
    torch.set_num_threads(args.cpu_threads)
    torch_iris.set_cpu_fallback_enabled(False)
    torch_iris.reset_fallback_stats()
    cpu, iris = measure("cpu"), measure("iris:0")
    result = json.dumps({
        "torch": str(torch.__version__),
        "torch_iris": torch_iris.__version__,
        "package": torch_iris.__file__,
        "extension": torch_iris._C.__file__,
        "processor": platform.processor(),
        "cpu_threads": torch.get_num_threads(),
        "gpu": vars(torch_iris.get_device_properties()),
        "warmup_steps": WARMUPS, "measured_steps": REPETITIONS,
        "dtype": "float32",
        "workloads": {
            "matmul": {"inputs": [[1024, 1024], [1024, 1024]]},
            "mlp_training": {"input": [128, 256], "widths": [256, 512, 64], "optimizer": "SGD"},
            "cnn_training": {"input": [32, 1, 28, 28], "labels_dtype": "int64",
                             "optimizer": "fused AdamW", "dropout": 0.0},
            "attention": {"q_k_v": [2, 4, 128, 64], "causal": True},
            "matmul_with_transfers": {"inputs": [[384, 384], [384, 384]],
                                      "timed_transfers": "two host-to-device and one device-to-host"},
        },
        "cpu": cpu, "iris": iris,
        "cpu_faster": [name for name in cpu if cpu[name]["median_ms"] < iris[name]["median_ms"]],
        "cpu_fallbacks": torch_iris.fallback_stats(),
    }, indent=2)
    if args.output:
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
