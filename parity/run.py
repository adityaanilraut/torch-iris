"""Run the named parity scope on Iris or collect a reference from physical NVIDIA CUDA."""

import argparse
from collections import Counter
import csv
from hashlib import file_digest, sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from .capabilities import iris_case_blocker
from .operators import OPERATOR_SCOPE, run_operator_case
from .training import TRAINING_SCOPE, run_training_case
from .workloads import SCOPE, compare, run_case


def digest(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def scope_digest(scope):
    root = Path(__file__).parent
    files = ["scope-v1.json", "workloads.py", "capabilities.py"]
    if scope is TRAINING_SCOPE:
        files.extend(["scope-training-v1.json", "training.py"])
    elif scope is OPERATOR_SCOPE:
        files.extend(["scope-operators-v1.json", "operators.py"])
    return sha256(b"".join((root / name).read_text(encoding="utf-8").encode()
                           for name in files)).hexdigest()


def adapters():
    if sys.platform != "win32":
        return []
    command = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,DriverVersion | ConvertTo-Json -Compress -AsArray"
    )
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if not shell:
        return []
    # Windows PowerShell 5.1 does not have ConvertTo-Json -AsArray.
    if Path(shell).stem.lower() == "powershell":
        command = "@(" + command.split(" | ConvertTo-Json")[0] + ") | ConvertTo-Json -Compress"
    info = json.loads(subprocess.check_output([shell, "-NoProfile", "-Command", command], text=True))
    return info if isinstance(info, list) else [info]


def torch_metadata():
    return {
        "torch": str(torch.__version__),
        "git": torch.version.git_version,
        "cuda_build": torch.version.cuda,
        "python": sys.version.split()[0],
    }


def cuda_metadata(device):
    if torch.version.cuda is None or torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("A PyTorch 2.12.1 CUDA build and a physical NVIDIA GPU are required")
    gpu = torch.cuda.get_device_properties(device)
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
        text=True,
    )
    devices = [
        {"uuid": uuid.strip(), "gpu": name.strip(), "driver_version": version.strip()}
        for uuid, name, version in csv.reader(driver.splitlines())
    ]
    if not devices:
        raise RuntimeError("nvidia-smi did not report a physical GPU")
    return torch_metadata() | {
        "gpu": gpu.name,
        "compute_capability": [gpu.major, gpu.minor],
        "gpu_uuid": str(getattr(gpu, "uuid", "")),
        "nvidia_smi": devices,
        "device": str(device),
        "tf32": False,
    }


def capture_cuda(device, output, scope, execute):
    metadata = cuda_metadata(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    values = {}
    results = {}
    for case in scope["cases"]:
        case_id = case["id"]
        try:
            actual = execute(case, device)
            torch.cuda.synchronize(device)
            compare(actual, execute(case, "cpu"), scope["tolerances"])
            values[case_id] = actual
            results[case_id] = {"status": "pass"}
        except Exception as exc:
            results[case_id] = {"status": "fail", "error": str(exc)}
        print(case_id, "CUDA", results[case_id]["status"])
    torch.save({
        "scope_sha256": scope_digest(scope), "metadata": metadata, "values": values, "results": results,
    }, output)
    print(f"CUDA reference written to {output}")
    return int(any(row["status"] != "pass" for row in results.values()))


def run_iris(output, reference_path, wheel, scope, execute):
    import torch_iris

    if not torch_iris.is_available():
        raise RuntimeError("A physical Intel Iris Xe OpenCL GPU is required")
    extension = Path(torch_iris._C.__file__).resolve()
    package = Path(torch_iris.__file__).resolve().parent
    properties = vars(torch_iris.get_device_properties())
    iris = torch_metadata() | {
        "gpu": torch_iris.device_name(),
        "device_properties": properties,
        "verification_mode": "feature_masked" if properties["optional_features_masked"] else "physical_driver",
        "package_version": torch_iris.__version__,
        "package_path": str(package),
        "cpu_fallback_enabled": torch_iris.is_cpu_fallback_enabled(),
        "windows_display_adapters": adapters(),
        "extension_path": str(extension),
        "extension_sha256": digest(extension),
    }
    if wheel:
        from zipfile import ZipFile
        with ZipFile(wheel) as archive:
            wheel_files = {name for name in archive.namelist() if name.startswith("torch_iris/")
                           and name.endswith((".py", ".pyd"))}
            installed = [extension, *package.glob("*.py")]
            if wheel_files != {f"torch_iris/{path.name}" for path in installed}:
                raise RuntimeError("The installed package files do not match the requested wheel")
            for path in installed:
                if sha256(archive.read(f"torch_iris/{path.name}")).hexdigest() != digest(path):
                    raise RuntimeError(f"The installed {path.name} does not match the requested wheel")
        iris.update({"wheel": str(wheel), "wheel_sha256": digest(wheel)})

    reference = None
    if reference_path:
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
        if reference["scope_sha256"] != scope_digest(scope):
            raise RuntimeError("CUDA reference was generated from a different scope or workload revision")
        meta = reference["metadata"]
        if (meta["torch"].split("+", 1)[0] != "2.12.1" or not meta["cuda_build"]
                or not meta["nvidia_smi"]):
            raise RuntimeError("CUDA reference is missing its build or physical GPU driver information")
    nvidia_visible = any("nvidia" in row["Name"].lower() for row in iris["windows_display_adapters"])
    cuda_status = "untested" if nvidia_visible else "hardware_blocked"

    results = []
    for case in scope["cases"]:
        case_id = case["id"]
        row = case | {"cpu_status": "untested", "iris_status": "untested", "cuda_status": cuda_status,
                      "iris_cpu_fallbacks": {}}
        try:
            expected = execute(case, "cpu")
            row["cpu_status"] = "pass"
        except Exception as exc:
            row.update(cpu_status="fail", cpu_error=str(exc))
        blocker = iris_case_blocker(case)
        if blocker:
            row.update(iris_status="hardware_blocked", iris_reason=blocker)
        elif row["cpu_status"] == "pass":
            previous = Counter(torch_iris.fallback_stats())
            try:
                kernels = torch_iris._C._kernel_submissions()
                actual = execute(case, "iris:0")
                compare(actual, expected, scope["tolerances"])
                kernels = torch_iris._C._kernel_submissions() - kernels
                if case_id in {"mlp_forward_2d", "mlp_forward_3d"} and kernels < 4:
                    raise AssertionError("Expected native Iris kernel submissions for the complete MLP")
                row["iris_kernel_submissions"] = kernels
                implementation = case.get("iris_implementation", "native")
                if not properties["subgroup_kernels"]:
                    implementation = case.get("iris_implementation_without_subgroups", implementation)
                row["iris_implementation_used"] = implementation
                row["iris_status"] = f"{implementation}_pass"
            except Exception as exc:
                row.update(iris_status="fail", iris_error=str(exc))
            fallbacks = Counter(torch_iris.fallback_stats()) - previous
            row["iris_cpu_fallbacks"] = dict(fallbacks)
            if fallbacks and row["iris_status"].endswith("_pass"):
                row["iris_status"] = "fallback_pass"
        if reference is not None:
            cuda_result = reference["results"][case_id]
            if cuda_result["status"] == "fail":
                row.update(cuda_status="fail", cuda_error=cuda_result["error"])
            elif row["iris_status"].endswith("_pass"):
                try:
                    cuda = reference["values"][case_id]
                    compare(cuda, expected, scope["tolerances"])
                    compare(actual, cuda, scope["tolerances"])
                    row["cuda_status"] = "verified_pass"
                except Exception as exc:
                    row.update(cuda_status="fail", cuda_error=str(exc))
            else:
                row["cuda_status"] = "reference_pass"
        results.append(row)
        print(case_id, row["iris_status"], row["cuda_status"])

    iris_counts = Counter(row["iris_status"] for row in results)
    cuda_counts = Counter(row["cuda_status"] for row in results)
    count = len(results)
    verified = cuda_counts["verified_pass"]
    functional = iris_counts["native_pass"] + iris_counts["emulated_pass"] + iris_counts["fallback_pass"]
    report = {
        "format_version": 1,
        "scope": scope["scope"],
        "scope_version": scope["version"],
        "scope_sha256": scope_digest(scope),
        "tolerances": scope["tolerances"],
        "iris": iris,
        "cuda_reference": reference["metadata"] | {"artifact_sha256": digest(reference_path)}
        if reference else {"status": cuda_status, "local_cuda_build": torch.version.cuda},
        "coverage": {
            "iris_functional": {"passed": functional, "required": count},
            "iris_native": {"passed": iris_counts["native_pass"], "required": count},
            "cuda_verified_total": {"passed": verified, "required": count},
            "cuda_verified_native": {
                "passed": sum(row["iris_status"] == "native_pass" and
                              row["cuda_status"] == "verified_pass" for row in results),
                "required": count,
            },
            "iris_status_counts": dict(iris_counts),
            "cuda_status_counts": dict(cuda_counts),
        },
        "cases": results,
        "future_workflows": [
            row | {"status": "untested"} for row in scope.get("future_workflows", SCOPE["future_workflows"])
            if scope is SCOPE or row["id"] not in {"mlp_training", "training_host_transfer"}
        ],
        "vendor_capabilities": [
            row | {"iris_status": "hardware_blocked", "cuda_status": "untested"}
            for row in SCOPE["vendor_capabilities"]
        ],
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Iris functional {functional}/{count}, native {iris_counts['native_pass']}/{count}, "
          f"CUDA verified {verified}/{count}. Report written to {output}")
    return int(functional != count or (reference is not None and verified != count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    iris = commands.add_parser("iris")
    iris.add_argument("--output", type=Path, required=True)
    iris.add_argument("--cuda-reference", type=Path)
    iris.add_argument("--wheel", type=Path)
    cuda = commands.add_parser("cuda")
    cuda.add_argument("--device", type=torch.device, default=torch.device("cuda:0"))
    cuda.add_argument("--output", type=Path, required=True)
    scopes = {
        "forward": (SCOPE, run_case),
        "training": (TRAINING_SCOPE, run_training_case),
        "operators": (OPERATOR_SCOPE, run_operator_case),
    }
    for subcommand in (iris, cuda):
        subcommand.add_argument("--scope", choices=scopes, default="forward")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output path")
    if torch.__version__.split("+", 1)[0] != "2.12.1":
        parser.error(f"expected PyTorch 2.12.1, found {torch.__version__}")
    scope, execute = scopes[args.scope]
    if args.mode == "cuda":
        if args.device.type != "cuda":
            parser.error("CUDA reference requires a cuda device")
        return capture_cuda(args.device, args.output, scope, execute)
    else:
        return run_iris(args.output, args.cuda_reference, args.wheel, scope, execute)


if __name__ == "__main__":
    sys.exit(main())
