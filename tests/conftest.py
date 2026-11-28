import pytest
import torch
import torch_iris


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_fp16: requires OpenCL FP16 support")


def pytest_collection_modifyitems(items):
    if torch.float16 in torch_iris.get_amp_supported_dtype():
        return
    for item in items:
        if item.get_closest_marker("requires_fp16"):
            item.add_marker(pytest.mark.skip(reason="OpenCL FP16 is unavailable"))
