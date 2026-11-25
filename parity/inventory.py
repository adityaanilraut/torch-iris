"""Generate the PyTorch 2.12.1 reference catalog and named workload inventory."""

import argparse
import json
from pathlib import Path

import torch

from .operators import OPERATOR_SCOPE
from .training import TRAINING_SCOPE
from .workloads import SCOPE


def dtypes(values):
    return sorted(str(dtype) for dtype in (values or ()))


def generator_name(fn):
    return getattr(fn, "__name__", type(fn).__name__) if fn is not None else None


def inventory():
    if torch.__version__.split("+", 1)[0] != "2.12.1":
        raise RuntimeError(f"This inventory requires PyTorch 2.12.1, found {torch.__version__}")
    from torch.testing._internal.common_methods_invocations import op_db

    opinfo = []
    for op in sorted(op_db, key=lambda op: (op.name, op.variant_test_name)):
        opinfo.append({
            "name": op.name,
            "variant": op.variant_test_name,
            "aten_name": op.aten_name,
            "aliases": sorted(alias.name for alias in op.aliases),
            "cpu_dtypes": dtypes(op.supported_dtypes("cpu")),
            "declared_cuda_dtypes": dtypes(op.supported_dtypes("cuda")),
            "declared_cuda_backward_dtypes": dtypes(op.supported_backward_dtypes("cuda")),
            "method": op.method_variant is not None,
            "inplace": op.inplace_variant is not None,
            "out": op.supports_out,
            "autograd": op.supports_autograd,
            "inplace_autograd": op.supports_inplace_autograd,
            "gradgrad": op.supports_gradgrad,
            "forward_ad": op.supports_forward_ad,
            "layouts": {
                layout: getattr(op, f"supports_{layout}")
                for layout in ("sparse", "sparse_csr", "sparse_csc", "sparse_bsr", "sparse_bsc", "njt")
            },
            "sample_generator": generator_name(op.sample_inputs_func),
            "reference_generator": generator_name(op.reference_inputs_func),
            "error_generator": generator_name(op.error_inputs_func),
            "status": "untested",
        })
    schemas = []
    for schema in sorted(torch._C._jit_get_all_schemas(), key=str):
        if not schema.name.startswith("aten::"):
            continue
        schemas.append({
            "operator": schema.name + (f".{schema.overload_name}" if schema.overload_name else ""),
            "schema": str(schema),
            "mutable": schema.is_mutable,
            "status": "untested",
        })
    return {
        "format_version": 1,
        "source": {"torch": str(torch.__version__), "git": torch.version.git_version},
        "catalog_counts": {"opinfo_rows": len(opinfo), "aten_schemas": len(schemas)},
        "opinfo": opinfo,
        "aten_schemas": schemas,
        "named_scope": SCOPE,
        "additional_scopes": [TRAINING_SCOPE, OPERATOR_SCOPE],
        "note": "Catalog rows describe candidate tests. Results for executable cases are in the coverage report.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["catalog_counts"])


if __name__ == "__main__":
    main()
