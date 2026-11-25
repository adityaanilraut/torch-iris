"""Versioned portable workloads using only normal PyTorch entry points."""

import json
from pathlib import Path

import torch


SCOPE = json.loads(Path(__file__).with_name("scope-v1.json").read_text(encoding="utf-8"))
TOLERANCE = SCOPE["tolerances"]


def fixture(*shape, phase=0):
    count = 1
    for size in shape:
        count *= size
    # Dyadic values are identical across PyTorch CPU and CUDA builds.
    return ((torch.arange(count, dtype=torch.float32) + phase) % 31 - 15).reshape(shape) / 32


def snapshot(**tensors):
    return {
        key: {"value": value.detach().cpu().clone(), "stride": list(value.stride())}
        for key, value in tensors.items()
    }


def compare(actual, reference, tolerance=TOLERANCE):
    if actual.keys() != reference.keys():
        raise AssertionError(f"Output keys differ: {actual.keys()} vs {reference.keys()}")
    for key, result in actual.items():
        expected = reference[key]
        if result["stride"] != expected["stride"]:
            raise AssertionError(f"{key} stride {result['stride']} != {expected['stride']}")
        torch.testing.assert_close(
            result["value"], expected["value"], equal_nan=True,
            msg=lambda msg: f"{key}: {msg}", **tolerance,
        )


def make_mlp(device):
    model = torch.nn.Sequential(torch.nn.Linear(7, 13), torch.nn.ReLU(), torch.nn.Linear(13, 5))
    for index, param in enumerate(model.parameters()):
        param.copy_(fixture(*param.shape, phase=7 * index))
    return model.to(device)


def mlp_forward(device, shape):
    model = make_mlp(device)
    captures = {}

    def capture(name):
        def hook(_module, _inputs, output):
            assert output.device == device
            captures[name] = output
        return hook

    for name, layer in zip(("first_linear", "relu", "second_linear"), model, strict=True):
        layer.register_forward_hook(capture(name))
    output = model(fixture(*shape, phase=3).to(device))
    reduction = output.sum()
    assert reduction.device == device
    return snapshot(**captures, reduction=reduction)


def mlp_forward_2d(device):
    return mlp_forward(device, (11, 7))


def mlp_forward_3d(device):
    return mlp_forward(device, (2, 3, 7))


def mlp_forward_empty_batch(device):
    return mlp_forward(device, (0, 7))


def linear_device_init(device):
    def construct():
        torch.manual_seed(7462)
        return torch.nn.Linear(7, 13, device=device)

    layer, repeat = construct(), construct()
    weights = tuple(layer.parameters())
    for value, duplicate in zip(weights, repeat.parameters(), strict=True):
        assert value.device == device and value.dtype == torch.float32
        host = value.cpu()
        assert torch.max(host.abs()) <= 7 ** -0.5
        torch.testing.assert_close(host, duplicate.cpu(), rtol=0, atol=0)
        assert torch.count_nonzero(host) == host.numel()
    host_input = fixture(11, 7, phase=3)
    output = layer(host_input.to(device))
    assert output.device == device
    expected = torch.nn.functional.linear(host_input, *(value.cpu() for value in weights))
    compare(snapshot(output=output), snapshot(output=expected))
    # Seeded RNG streams differ by backend. The exact outputs of the shared-weight
    # workload are compared separately. Record the observable result invariants here.
    return snapshot(output_metadata=torch.tensor([*output.shape], dtype=torch.int64))


def view_alias(device):
    root = fixture(4, 6).to(device)
    view = root.view(2, 2, 6).transpose(0, 2)
    offset = root.as_strided((2, 3), (6, 2), 7)
    transposed = root.t()
    reshaped = transposed.reshape(24)
    result = snapshot(reshaped=reshaped)
    before = root._version
    offset.fill_(2.5)
    assert root._version == view._version == offset._version == before + 1
    assert offset.storage_offset() == 7 and view.storage_offset() == 0
    compare(snapshot(reshaped=reshaped), result, {"rtol": 0, "atol": 0})
    return result | snapshot(root=root, view=view, transposed=transposed, offset=offset)


def expanded_alias(device):
    root = fixture(1, 3).to(device)
    expanded = root.expand(4, 3)
    assert expanded.stride() == (0, 1)
    before = root._version
    expanded.fill_(3.0)
    assert root._version == expanded._version == before + 1
    return snapshot(root=root, expanded=expanded)


def strided_copy(device):
    host = fixture(2, 3).t()
    uploaded = host.to(device)
    assert uploaded.stride() == host.stride()
    target = torch.empty_strided((3, 2), (1, 5), device=device)
    assert target.copy_(uploaded) is target
    before = snapshot(holes=target)
    target.copy_(fixture(1, 2, phase=4).to(device))
    host_target = torch.empty_strided((3, 2), (1, 7))
    host_target.copy_(target)
    return before | snapshot(uploaded=uploaded, target=target, host_target=host_target)


def binary_broadcast(device):
    left = fixture(4, 3).to(device).t().unsqueeze(1)
    right = fixture(2, 1, phase=5).to(device)
    expanded = right.expand(3, 2, 4)
    empty = torch.empty((3, 0, 4), device=device)
    return snapshot(
        added=torch.add(left, right, alpha=-0.75),
        multiplied=left * expanded,
        empty=left + empty,
    )


def initialization(device):
    zeros = torch.zeros((3, 4), device=device)
    filled = torch.full((3, 4), -0.25, device=device)
    view = filled.as_strided((2, 2), (4, 2), 1)
    fill_version = filled._version
    assert view.fill_(1.25) is view
    assert filled._version == fill_version + 1
    cleared = fixture(2, 3).to(device)
    assert cleared.zero_() is cleared
    return snapshot(zeros=zeros, filled=filled, cleared=cleared)


def mm_strided(device):
    left = fixture(5, 7).to(device).as_strided((4, 3), (7, 2), 1)
    right = fixture(6, 5, phase=7).to(device).as_strided((5, 3), (5, 1), 6).t()
    return snapshot(output=torch.mm(left, right))


def addmm_scalars(device):
    left = fixture(5, 7).to(device)
    right = fixture(3, 7, phase=7).to(device).t()
    bias = fixture(3, phase=2).to(device)
    nan = torch.full((3,), float("nan"), device=device)
    empty_left = torch.empty((5, 0), device=device)
    empty_right = torch.empty((0, 3), device=device)
    return snapshot(
        scaled=torch.addmm(bias, left, right, alpha=0.75, beta=-0.25),
        ignored_bias=torch.addmm(nan, left, right, beta=0),
        zero_inner=torch.addmm(bias, empty_left, empty_right, beta=2),
        zero_inner_and_beta=torch.addmm(nan, empty_left, empty_right, beta=0),
    )


def reduce_sum(device):
    view = fixture(5, 13, 9).to(device).transpose(0, 2)
    empty = torch.empty((0, 3), device=device)
    scalar = fixture(phase=3).to(device)
    return snapshot(
        all=view.sum(), dim=view.sum(1), keepdim=view.sum((-1, 0), keepdim=True),
        empty=empty.sum(0), empty_out=empty.sum(1), scalar=scalar.sum(0, keepdim=True),
    )


def nonfinite(device):
    values = torch.tensor([float("nan"), -float("inf"), float("inf"), -0.0]).to(device)
    return snapshot(relu=values.relu(), sum=values.sum())


def expected_errors(device):
    matrix = torch.empty((2, 3), device=device)
    wrong = torch.empty((4, 2), device=device)
    def partial_overlap():
        root = torch.empty((5,), device=device)
        return root.as_strided((3,), (1,), 1).copy_(root.as_strided((3,), (1,), 0))

    for name, op in (
        ("view", lambda: matrix.t().view(6)),
        ("broadcast", lambda: matrix + wrong),
        ("addmm", lambda: torch.addmm(matrix, matrix, wrong)),
        ("reduction", lambda: matrix.sum((1, -1))),
        ("copy into expansion", lambda: matrix.expand(2, 2, 3).copy_(fixture(2, 2, 3).to(device))),
        ("copy into partially overlapping slice", partial_overlap),
    ):
        try:
            op()
        except NotImplementedError as exc:
            raise AssertionError(f"{name} failed due to missing dispatch") from exc
        except RuntimeError:
            continue
        raise AssertionError(f"{name} accepted invalid input")
    return {}


def run_case(case, device):
    with torch.no_grad():
        return globals()[case["id"]](torch.device(device))
