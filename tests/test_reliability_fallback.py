"""Tests for boxed CPU fallback, observability, ergonomics, and OpenCL hardening."""

import copy
import json
import os
import subprocess
import sys
import warnings

import pytest
import torch
import torch.nn.functional as F

import torch_iris
from parity.capabilities import iris_case_blocker
from parity.operators import OPERATOR_SCOPE, run_operator_case
from parity.run import run_iris
from parity.training import TRAINING_SCOPE, run_training_case
from parity.workloads import SCOPE, compare, run_case, snapshot


pytestmark = pytest.mark.skipif(not torch_iris.is_available(), reason="requires Intel Iris Xe")
DEVICE = torch.device("iris:0")


@pytest.fixture(autouse=True)
def _reset_fallback_state():
    enabled = torch_iris.is_cpu_fallback_enabled()
    warning = torch_iris.is_fallback_warning_enabled()
    torch_iris.set_fallback_warning_enabled(True)
    torch_iris.reset_fallback_stats()
    try:
        yield
    finally:
        torch_iris.set_cpu_fallback_enabled(enabled)
        torch_iris.set_fallback_warning_enabled(warning)
        torch_iris.reset_fallback_stats()


def test_unimplemented_op_cpu_fallback_stats_and_once_warning():
    torch_iris.set_cpu_fallback_enabled(True)
    host = torch.tensor([-2.5, -0.5, 0.0, 1.25, 4.0], dtype=torch.float32)
    x = host.to(DEVICE)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out1 = torch.special.i0e(x)
        out2 = torch.special.i0e(x)

    assert out1.device == DEVICE
    torch.testing.assert_close(out1.cpu(), torch.special.i0e(host), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out2.cpu(), torch.special.i0e(host), rtol=1e-5, atol=1e-6)

    stats = torch_iris.fallback_stats()
    assert stats.get("aten::special_i0e.out") == 2

    fallback_warnings = [w for w in caught if "special_i0e" in str(w.message)]
    assert len(fallback_warnings) == 1

    # Reset clears counters and allows warning once again.
    torch_iris.reset_fallback_stats()
    assert torch_iris.fallback_stats() == {}
    with warnings.catch_warnings(record=True) as caught_again:
        warnings.simplefilter("always")
        _ = torch.special.i0e(x)
    assert len([w for w in caught_again if "special_i0e" in str(w.message)]) == 1

    # Silencing warnings still runs fallback and records stats.
    torch_iris.reset_fallback_stats()
    torch_iris.set_fallback_warning_enabled(False)
    assert not torch_iris.is_fallback_warning_enabled()
    with warnings.catch_warnings(record=True) as silenced:
        warnings.simplefilter("always")
        _ = torch.special.i0e(x)
    assert [w for w in silenced if "special_i0e" in str(w.message)] == []
    assert torch_iris.fallback_stats().get("aten::special_i0e.out") == 1


def test_parity_report_counts_actual_fallbacks_per_case(tmp_path):
    torch_iris.set_cpu_fallback_enabled(True)
    scope = SCOPE | {"cases": [{"id": "fallback_1"}, {"id": "native"}, {"id": "fallback_2"}]}

    def execute(case, device):
        x = torch.tensor([.25, .5], device=device)
        value = x + 1 if case["id"] == "native" else torch.special.i0e(x)
        return snapshot(value=value)

    output = tmp_path / "parity.json"
    torch_iris.set_fallback_warning_enabled(False)
    assert run_iris(output, None, None, scope, execute) == 0
    report = json.loads(output.read_text())
    cases = report["cases"]
    assert [row["iris_status"] for row in cases] == ["fallback_pass", "native_pass", "fallback_pass"]
    assert [row["iris_cpu_fallbacks"] for row in cases] == [
        {"aten::special_i0e.out": 1}, {}, {"aten::special_i0e.out": 1},
    ]
    assert report["coverage"]["iris_native"]["passed"] == 1
    assert report["coverage"]["iris_functional"]["passed"] == 3


def test_parity_reports_missing_fp16_without_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(torch_iris, "get_amp_supported_dtype", lambda: [])
    scope = SCOPE | {"cases": [{"id": name, "requires_fp16": True}
                              for name in ("fp16_operators", "autocast_training")]}
    devices = []

    def execute(case, device):
        devices.append(device)
        return {}

    output = tmp_path / "parity.json"
    assert run_iris(output, None, None, scope, execute) == 1
    assert devices == ["cpu", "cpu"]
    report = json.loads(output.read_text())
    assert report["coverage"]["iris_status_counts"] == {"hardware_blocked": 2}
    assert report["coverage"]["iris_functional"] == {"passed": 0, "required": 2}


def test_strict_gpu_only_mode_blocks_fallback():
    x = torch.randn(4, device=DEVICE)
    torch_iris.set_cpu_fallback_enabled(False)
    assert not torch_iris.is_cpu_fallback_enabled()

    with pytest.raises(NotImplementedError, match="CPU fallback is disabled"):
        _ = torch.special.i0e(x)

    assert torch_iris.fallback_stats() == {}

    # Native ops continue to work in strict GPU-only mode.
    y = (x * 2.0 + 1.0).relu().sum()
    assert y.device == DEVICE
    assert torch_iris.fallback_stats() == {}


@pytest.mark.parametrize("rows,inner,cols,strided", [
    (64, 128, 96, False), (11, 17, 9, True), (3, 0, 2, False), (0, 5, 2, False),
])
def test_native_int8_matmul(rows, inner, cols, strided):
    torch_iris.set_cpu_fallback_enabled(False)
    rng = torch.Generator().manual_seed(12)
    step = 2 if strided else 1
    left = torch.randint(-128, 128, (rows, inner * step), dtype=torch.int8, generator=rng)
    right = torch.randint(-128, 128, (inner * step, cols), dtype=torch.int8, generator=rng)
    expected = left[:, ::step].int() @ right[::step].int()
    a, b = left.to(DEVICE)[:, ::step], right.to(DEVICE)[::step]
    result = torch._int_mm(a, b)
    assert result.device == DEVICE
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)

    storage = torch.full((expected.numel() + 2,), -1, device=DEVICE, dtype=torch.int32)
    output = storage[1:-1].view(rows, cols)
    assert torch._int_mm(a, b, out=output) is output
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert storage.cpu()[[0, -1]].tolist() == [-1, -1]
    assert torch_iris.fallback_stats() == {}


def test_native_int8_matmul_checks_out_contract():
    torch_iris.set_cpu_fallback_enabled(False)
    a = torch.ones((3, 5), dtype=torch.int8, device=DEVICE)
    b = torch.ones((5, 2), dtype=torch.int8, device=DEVICE)
    with pytest.raises(RuntimeError, match="int32"):
        torch._int_mm(a, b, out=torch.empty((3, 2), dtype=torch.int64, device=DEVICE))
    with pytest.raises(RuntimeError, match="result shape"):
        torch._int_mm(a, b, out=torch.empty((0,), dtype=torch.int32, device=DEVICE))
    with pytest.raises(RuntimeError, match="contiguous"):
        torch._int_mm(a, b, out=torch.empty((2, 3), dtype=torch.int32, device=DEVICE).T)


@pytest.mark.parametrize("group_size", [3, 8])
@pytest.mark.parametrize("with_bias", [False, True])
def test_weight_only_linear_group_boundaries_and_empty(group_size, with_bias):
    torch_iris.set_cpu_fallback_enabled(False)
    generator = torch.Generator().manual_seed(2026)
    inner = 2 * group_size
    x = torch.randn(2, 3, inner, generator=generator).transpose(0, 1)
    weight = torch.randn(5, inner, generator=generator)
    bias = torch.randn(5, generator=generator) if with_bias else None
    for bits in ([8] if group_size == 3 else [4, 8]):
        quantize = getattr(torch_iris.ops, f"quantize_weight_int{bits}")
        linear = getattr(torch_iris.ops, f"linear_int{bits}")
        packed, scales, zeros = quantize(weight, group_size=group_size, symmetric=False)
        for rows, w, s, z in ((x, packed, scales, zeros), (x[..., :0], packed[:, :0], scales[:, :0], zeros[:, :0]),
                              (x[:0], packed, scales, zeros)):
            args = rows, w, s, z, bias
            expected = linear(*args, group_size=group_size)
            actual = linear(*(value.to(DEVICE) if value is not None else None for value in args),
                            group_size=group_size)
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-5)
    assert torch_iris.fallback_stats() == {}


def test_native_matrix_product_overloads():
    torch_iris.set_cpu_fallback_enabled(False)
    a = torch.arange(-6, 6, dtype=torch.float32).reshape(3, 4) / 4
    b = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4
    a_batch, b_batch = a.expand(2, -1, -1).clone(), b.expand(2, -1, -1).clone()
    product, batch_product = torch.mm(a, b), torch.bmm(a_batch, b_batch)
    x, y, xb, yb = (tensor.to(DEVICE) for tensor in (a, b, a_batch, b_batch))
    for operation, inputs, expected in (
        (torch.mm, (x, y), product),
        (torch.matmul, (x, y), product),
        (torch.bmm, (xb, yb), batch_product),
        (torch.matmul, (xb, yb), batch_product),
        (torch.addmm, (product.to(DEVICE), x, y), 2 * product),
        (torch.dot, (x[0], x[1]), torch.dot(a[0], a[1])),
    ):
        output = torch.empty_like(expected, device=DEVICE)
        assert operation(*inputs, out=output) is output
        torch.testing.assert_close(output.cpu(), expected)

    destination = torch.zeros_like(product, device=DEVICE)
    assert destination.addmm_(x, y) is destination
    torch.testing.assert_close(destination.cpu(), product)
    assert destination.addbmm_(xb, yb) is destination
    torch.testing.assert_close(destination.cpu(), product + batch_product.sum(0))

    activation = torch.ops.aten._addmm_activation
    torch.testing.assert_close(activation(destination, x, y, use_gelu=True).cpu(),
                               activation(destination.cpu(), a, b, use_gelu=True))
    torch.testing.assert_close(torch.ops.aten.linear.out(x, y.T, out=destination).cpu(), product)
    assert torch_iris.fallback_stats() == {}


@pytest.mark.parametrize("dtype", [pytest.param(torch.float16, marks=pytest.mark.requires_fp16), torch.bfloat16])
def test_native_matrix_products_with_float32_output(dtype):
    torch_iris.set_cpu_fallback_enabled(False)
    a = torch.arange(-6, 6, dtype=dtype).reshape(3, 4) / 4
    b = torch.arange(8, dtype=dtype).reshape(4, 2) / 4
    x, y = a.to(DEVICE), b.to(DEVICE)
    bias = torch.ones((3, 2), device=DEVICE, dtype=dtype)
    for operation, inputs, expected in (
        (torch.mm, (x, y), a.float() @ b.float()),
        (torch.bmm, (x[None], y[None]), (a.float() @ b.float())[None]),
        (torch.addmm, (bias, x, y), 1 + a.float() @ b.float()),
        (torch.baddbmm, (bias[None], x[None], y[None]), (1 + a.float() @ b.float())[None]),
    ):
        torch.testing.assert_close(operation(*inputs, out_dtype=torch.float32).cpu(), expected)
        output = torch.empty_like(expected, device=DEVICE)
        assert operation(*inputs, out_dtype=torch.float32, out=output) is output
        torch.testing.assert_close(output.cpu(), expected)
    assert torch_iris.fallback_stats() == {}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int32, torch.complex64])
def test_native_addmm_input_dtypes(dtype):
    torch_iris.set_cpu_fallback_enabled(False)
    a = torch.arange(12).reshape(3, 4).to(dtype)
    b = torch.arange(8).reshape(4, 2).to(dtype)
    bias = torch.ones(2, dtype=dtype)
    expected = torch.addmm(bias, a, b, alpha=2, beta=3)
    actual = torch.addmm(bias.to(DEVICE), a.to(DEVICE), b.to(DEVICE), alpha=2, beta=3)
    torch.testing.assert_close(actual.cpu(), expected)
    assert torch_iris.fallback_stats() == {}


def test_env_var_controls_in_fresh_subprocess():
    env_strict = os.environ.copy()
    env_strict["TORCH_IRIS_CPU_FALLBACK"] = "0"
    env_strict["TORCH_IRIS_FALLBACK_WARN"] = "0"
    script = (
        "import torch, torch_iris\n"
        "assert not torch_iris.is_cpu_fallback_enabled()\n"
        "assert not torch_iris.is_fallback_warning_enabled()\n"
        "x = torch.randn(4, device='iris')\n"
        "try:\n"
        "    torch.special.i0e(x)\n"
        "    raise SystemExit('expected NotImplementedError')\n"
        "except NotImplementedError:\n"
        "    pass\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], env=env_strict, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_fallback_out_inplace_multi_output_strided_and_autograd():
    torch_iris.set_cpu_fallback_enabled(True)
    torch_iris.set_fallback_warning_enabled(False)

    # 1. Multi-tensor output (torch.frexp returns (mantissa, exponent))
    host = torch.tensor([[-3.5, 0.25, 8.0, 1.5], [0.0, -0.125, 16.0, 7.25]], dtype=torch.float32)
    strided_host = host.T
    strided_iris = host.to(DEVICE).T
    m_cpu, e_cpu = torch.frexp(strided_host)
    m_iris, e_iris = torch.frexp(strided_iris)
    assert m_iris.device == DEVICE and e_iris.device == DEVICE
    torch.testing.assert_close(m_iris.cpu(), m_cpu, rtol=0, atol=0)
    torch.testing.assert_close(e_iris.cpu(), e_cpu, rtol=0, atol=0)
    assert torch_iris.fallback_stats().get("aten::frexp.Tensor_out", 0) >= 1

    # 2. out= into a non-contiguous strided view
    storage_host = torch.full((4, 6), -99.0)
    storage_iris = storage_host.to(DEVICE)
    out_host = storage_host[:, ::3]  # shape (4, 2), non-contiguous
    out_iris = storage_iris[:, ::3]
    torch.special.i0e(strided_host, out=out_host)
    returned = torch.special.i0e(strided_iris, out=out_iris)
    assert returned is out_iris
    torch.testing.assert_close(storage_iris.cpu(), storage_host, rtol=1e-5, atol=1e-6)

    # 3. In-place op on a strided view (aten::special_i0e_ or aten::cauchy_? Use aten::cholesky_inverse / linalg / erfcx / etc.)
    base_host = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
    base_iris = base_host.to(DEVICE)
    view_host = base_host[:, ::2]
    view_iris = base_iris[:, ::2]
    # aten::lgamma_ is native; use aten::special_i0e via out=view (mutates view in-place) and aten::hypot?
    # Test a genuine in-place fallback op: aten::igamma_ (not natively implemented on Iris)
    a_host = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    x_host = torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32)
    a_base_host = torch.zeros(2, 4)
    a_base_host[:, ::2] = a_host
    a_base_iris = a_base_host.to(DEVICE)
    ret_inplace = a_base_iris[:, ::2].igamma_(x_host.to(DEVICE))
    a_base_host[:, ::2].igamma_(x_host)
    assert ret_inplace.device == DEVICE
    torch.testing.assert_close(a_base_iris.cpu(), a_base_host, rtol=1e-5, atol=1e-6)
    assert any("igamma" in k for k in torch_iris.fallback_stats())

    # 4. Autograd forward + backward through fallback (cholesky_inverse & special.i0e)
    lower_cpu = torch.tensor([[2.0, 0.0, 0.0], [0.5, 1.5, 0.0], [-0.25, 0.4, 1.2]], requires_grad=True)
    lower_iris = lower_cpu.detach().to(DEVICE).requires_grad_()
    inv_cpu = torch.cholesky_inverse(lower_cpu)
    inv_iris = torch.cholesky_inverse(lower_iris)
    assert inv_iris.device == DEVICE
    torch.testing.assert_close(inv_iris.cpu(), inv_cpu, rtol=1e-4, atol=1e-5)
    inv_cpu.square().sum().backward()
    inv_iris.square().sum().backward()
    assert lower_iris.grad is not None and lower_iris.grad.device == DEVICE
    torch.testing.assert_close(lower_iris.grad.cpu(), lower_cpu.grad, rtol=1e-4, atol=1e-5)

    # Also test elementwise fallback with autograd (special.i0e backward uses special_i1e / fallback)
    z_cpu = torch.tensor([-1.5, 0.25, 2.0], requires_grad=True)
    z_iris = z_cpu.detach().to(DEVICE).requires_grad_()
    y_cpu = torch.special.i0e(z_cpu)
    y_iris = torch.special.i0e(z_iris)
    torch.testing.assert_close(y_iris.cpu(), y_cpu, rtol=1e-5, atol=1e-6)
    y_cpu.sum().backward()
    y_iris.sum().backward()
    assert z_iris.grad is not None and z_iris.grad.device == DEVICE
    torch.testing.assert_close(z_iris.grad.cpu(), z_cpu.grad, rtol=1e-5, atol=1e-6)


def test_fp64_guard_remains_explicit_with_fallback():
    torch_iris.set_cpu_fallback_enabled(True)
    props = torch_iris.get_device_properties()
    if props.double_fp_config == 0:
        with pytest.raises(RuntimeError, match="storage"):
            torch.ones(2, device=DEVICE, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="storage"):
            torch.kaiser_window(5, device=DEVICE, dtype=torch.float64)


def test_boolean_and_integer_reductions_native():
    torch.manual_seed(42)
    torch_iris.reset_fallback_stats()

    logits = torch.randn(64, 10)
    labels = torch.randint(0, 10, (64,))
    expected = (logits.argmax(dim=1) == labels).sum()
    actual = (logits.to(DEVICE).argmax(dim=1) == labels.to(DEVICE)).sum()
    assert actual.device == DEVICE
    assert actual.dtype == expected.dtype == torch.int64
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    # Large bool reduction (> 4096 elements) and strided bool reduction
    big_a = torch.randint(0, 5, (128, 64))
    big_b = torch.randint(0, 5, (128, 64))
    big_mask_cpu = (big_a == big_b)
    big_mask_iris = (big_a.to(DEVICE) == big_b.to(DEVICE))
    torch.testing.assert_close(big_mask_iris.sum().cpu(), big_mask_cpu.sum(), rtol=0, atol=0)
    torch.testing.assert_close(big_mask_iris.T[:, ::2].sum(dim=1).cpu(), big_mask_cpu.T[:, ::2].sum(dim=1), rtol=0, atol=0)
    assert torch_iris.fallback_stats() == {}


@pytest.mark.parametrize(
    "opt_cls,kwargs",
    [
        (torch.optim.SGD, {"lr": 0.05, "momentum": 0.9, "weight_decay": 1e-3}),
        (torch.optim.SGD, {"lr": 0.05, "momentum": 0.9, "weight_decay": 1e-3, "foreach": True}),
        (torch.optim.SGD, {"lr": 0.05, "momentum": 0.9, "weight_decay": 1e-3, "foreach": False}),
        (torch.optim.SGD, {"lr": 0.05, "momentum": 0.9, "weight_decay": 1e-3, "fused": True}),
        (torch.optim.Adam, {"lr": 1e-2, "weight_decay": 1e-3}),
        (torch.optim.Adam, {"lr": 1e-2, "amsgrad": True, "foreach": True}),
        (torch.optim.Adam, {"lr": 1e-2, "foreach": False}),
        (torch.optim.Adam, {"lr": 1e-2, "fused": True}),
        (torch.optim.AdamW, {"lr": 1e-2, "weight_decay": 1e-2}),
        (torch.optim.AdamW, {"lr": 1e-2, "amsgrad": True, "foreach": True}),
        (torch.optim.AdamW, {"lr": 1e-2, "foreach": False}),
        (torch.optim.AdamW, {"lr": 1e-2, "fused": True}),
    ],
)
def test_standard_optimizers_default_foreach_and_fused_native(opt_cls, kwargs):
    torch.manual_seed(123)
    torch_iris.reset_fallback_stats()

    cpu_model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )
    iris_model = copy.deepcopy(cpu_model).to(DEVICE)

    cpu_kwargs = {k: v for k, v in kwargs.items() if k != "fused"}
    cpu_opt = opt_cls(cpu_model.parameters(), **cpu_kwargs)
    iris_opt = opt_cls(iris_model.parameters(), **kwargs)

    for _ in range(3):
        x = torch.randn(4, 8)
        target = torch.randn(4, 4)

        cpu_opt.zero_grad(set_to_none=True)
        F.mse_loss(cpu_model(x), target).backward()
        cpu_opt.step()

        iris_opt.zero_grad(set_to_none=True)
        F.mse_loss(iris_model(x.to(DEVICE)), target.to(DEVICE)).backward()
        iris_opt.step()

    for p_iris, p_cpu in zip(iris_model.parameters(), cpu_model.parameters(), strict=True):
        torch.testing.assert_close(p_iris.cpu(), p_cpu, rtol=1e-4, atol=1e-5)

    assert torch_iris.fallback_stats() == {}


def test_empty_and_zero_size_tensors_hardening():
    torch_iris.reset_fallback_stats()

    # Empty BatchNorm2d train & eval (previously hit float64 accumulate type on CPU composite path)
    bn_eval = torch.nn.BatchNorm2d(4).to(DEVICE).eval()
    out_bn_eval = bn_eval(torch.empty(0, 4, 4, 4, device=DEVICE))
    assert out_bn_eval.shape == (0, 4, 4, 4) and out_bn_eval.device == DEVICE

    bn_train = torch.nn.BatchNorm2d(4).to(DEVICE).train()
    x_bn = torch.empty(0, 4, 4, 4, device=DEVICE, requires_grad=True)
    out_bn_train = bn_train(x_bn)
    assert out_bn_train.shape == (0, 4, 4, 4)
    out_bn_train.sum().backward()
    assert x_bn.grad.shape == (0, 4, 4, 4)
    torch.testing.assert_close(bn_train.weight.grad.cpu(), torch.zeros(4))
    torch.testing.assert_close(bn_train.bias.grad.cpu(), torch.zeros(4))

    # Empty LayerNorm forward & backward
    ln = torch.nn.LayerNorm(4).to(DEVICE)
    x_ln = torch.empty(0, 4, device=DEVICE, requires_grad=True)
    y_ln = ln(x_ln)
    assert y_ln.shape == (0, 4)
    y_ln.sum().backward()
    torch.testing.assert_close(ln.weight.grad.cpu(), torch.zeros(4))
    torch.testing.assert_close(ln.bias.grad.cpu(), torch.zeros(4))

    # Empty GroupNorm, Softmax, Linear, CrossEntropy, Pooling, Reductions
    assert torch.nn.GroupNorm(2, 4).to(DEVICE)(torch.empty(0, 4, 4, 4, device=DEVICE)).shape == (0, 4, 4, 4)
    assert F.softmax(torch.empty(0, 4, device=DEVICE), dim=-1).shape == (0, 4)
    assert F.softmax(torch.empty(3, 0, device=DEVICE), dim=-1).shape == (3, 0)
    assert torch.nn.Linear(4, 8).to(DEVICE)(torch.empty(0, 4, device=DEVICE)).shape == (0, 8)
    assert F.max_pool2d(torch.empty(0, 4, 4, 4, device=DEVICE), 2).shape == (0, 4, 2, 2)
    assert F.avg_pool2d(torch.empty(0, 4, 4, 4, device=DEVICE), 2).shape == (0, 4, 2, 2)
    torch.testing.assert_close(torch.empty(3, 0, device=DEVICE).sum(1).cpu(), torch.zeros(3))
    torch.testing.assert_close(torch.empty(0, dtype=torch.bool, device=DEVICE).sum().cpu(), torch.tensor(0))
    assert not torch.empty(0, dtype=torch.bool, device=DEVICE).any().item()
    assert torch.empty(0, dtype=torch.bool, device=DEVICE).all().item()

    assert torch_iris.fallback_stats() == {}


def test_oversized_allocation_and_cache_reclamation():
    props = torch_iris.get_device_properties()
    if not props.max_allocation:
        pytest.skip("OpenCL per-buffer allocation limit is unavailable")
    torch_iris.empty_cache()
    baseline = torch_iris.memory_allocated()
    oversized_elems = props.max_allocation // 4 + 1024
    with pytest.raises(torch.OutOfMemoryError, match="max_mem_alloc_size"):
        torch.empty((oversized_elems,), dtype=torch.float32, device=DEVICE)

    # Verify allocator remains healthy after failed allocation and empty_cache reclaims pooled buffers.
    torch_iris.empty_cache()
    torch_iris.reset_peak_memory_stats()
    for _ in range(5):
        t = torch.ones((1024, 1024), dtype=torch.float32, device=DEVICE)
        assert t.sum().item() == pytest.approx(1024.0 * 1024.0)
        del t
        torch_iris.synchronize()
        assert torch_iris.memory_allocated() == baseline
    stats_cached = torch_iris.memory_stats()
    assert stats_cached["cached_bytes"] >= 1024 * 1024 * 4
    assert "cache_reclaims" in stats_cached
    torch_iris.empty_cache()
    cleared = torch_iris.memory_stats()
    assert cleared["cached_bytes"] == 0
    assert cleared["reserved_bytes"] == cleared["allocated_bytes"] == baseline
    print(json.dumps({"oversized_request_bytes": oversized_elems * 4, "baseline_bytes": baseline,
                      "after_recovery": stats_cached, "after_empty_cache": cleared}))


def test_parity_scopes_have_zero_cpu_fallbacks():
    torch_iris.reset_fallback_stats()
    torch_iris.set_cpu_fallback_enabled(False)

    for scope, execute in ((SCOPE, run_case), (TRAINING_SCOPE, run_training_case), (OPERATOR_SCOPE, run_operator_case)):
        for case in scope["cases"]:
            if not iris_case_blocker(case):
                compare(execute(case, "iris:0"), execute(case, "cpu"), scope["tolerances"])

    assert torch_iris.fallback_stats() == {}
