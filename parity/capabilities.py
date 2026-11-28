"""Capability requirements shared by the parity runner and tests."""

import torch


def iris_case_blocker(case):
    if case.get("requires_fp16"):
        import torch_iris

        if torch.float16 not in torch_iris.get_amp_supported_dtype():
            return "OpenCL FP16 is unavailable"
    return None
