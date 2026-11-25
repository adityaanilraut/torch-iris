import pytest
import torch
import torch_iris
from torch.nn import functional as F


pytestmark = pytest.mark.skipif(not torch_iris.is_available(), reason="requires Intel Iris Xe")


def check(cpu, iris, *, check_stride=False, **kwargs):
    assert iris.device.type == "iris"
    if check_stride:
        assert iris.stride() == cpu.stride()
    torch.testing.assert_close(iris.cpu(), cpu, equal_nan=True, **kwargs)


def compare_backward(fn, *inputs, rtol=1e-5, atol=1e-6):
    host = tuple(x.detach().requires_grad_() for x in inputs)
    device = tuple(x.detach().to("iris").requires_grad_() for x in inputs)
    expected, actual = fn(*host), fn(*device)
    check(expected, actual, rtol=rtol, atol=atol)
    seed = torch.randn_like(expected)
    host_grads = torch.autograd.grad(expected, host, seed)
    device_grads = torch.autograd.grad(actual, device, seed.to("iris"))
    for cpu_grad, iris_grad in zip(host_grads, device_grads):
        check(cpu_grad, iris_grad, rtol=rtol, atol=atol)


def test_integer_clamp_and_one_hot_pipeline():
    values = torch.tensor([[2**55 + 7, 2**55 + 9], [2**55 + 2, 2**55 + 4]]).T
    device = values.to("iris")
    lo, hi = 2**55 + 4, 2**55 + 7
    check(values.clamp(lo, hi), device.clamp(lo, hi), check_stride=True)
    lower = torch.tensor(lo)
    check(values.clamp_min(lower), device.clamp_min(lower.to("iris")))
    check(values.clamp(min=lower, max=values), device.clamp(min=lower.to("iris"), max=device))
    check(values.clamp(max=hi), device.clamp(max=hi))
    check(values.clamp(max=3.5), device.clamp(max=3.5))
    alias = device[0]
    assert device.clamp_(lo, hi) is device
    check(values.clamp(lo, hi)[0], alias)
    def one_hot(x):
        return F.one_hot((x.abs() * 3).long().clamp(0, 5).flatten(), 6)
    floats = torch.arange(-24, 24, dtype=torch.float32).reshape(8, 6).T / 9
    check(one_hot(floats), one_hot(floats.to("iris")))
    compare_backward(lambda x: x.clamp(-1, 2), floats)


@pytest.mark.parametrize("mode", ["floor", "trunc"])
@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.float32, torch.float16])
def test_rounded_division(mode, dtype):
    inputs = torch.tensor([[-11, 11, -6, 6], [9, -9, -1, 1]], dtype=dtype).T
    divisors = torch.tensor([-3, 2], dtype=dtype)
    host = torch.div(inputs, divisors, rounding_mode=mode)
    x, y = inputs.to("iris"), divisors.to("iris")
    check(host, torch.div(x, y, rounding_mode=mode), check_stride=True)
    check(torch.div(inputs, -3, rounding_mode=mode), torch.div(x, -3, rounding_mode=mode))
    out = torch.empty_like(x)
    assert torch.div(x, y, rounding_mode=mode, out=out) is out
    check(host, out)
    if dtype.is_floating_point:
        compare_backward(lambda a, b: torch.div(a, b, rounding_mode=mode), inputs, divisors)
    else:
        with pytest.raises(RuntimeError, match="ZeroDivision"):
            torch.div(x, torch.zeros_like(y), rounding_mode=mode)
    if mode == "floor":
        check(host, x // y)
        check(torch.floor_divide(inputs, -3), torch.floor_divide(x, -3))


def test_extrema_and_nansum():
    values = torch.tensor([1, 1, -2, -2, 3, 3, 1, -2.0]).reshape(2, 2, 2).transpose(0, 1)
    for fn in (torch.amax, torch.amin):
        compare_backward(lambda x: fn(x, (2, 0), keepdim=True), values)
        check(fn(values), fn(values.to("iris")))
    values[0, 0, 0] = torch.nan
    for expected, actual in zip(torch.aminmax(values, dim=-1), torch.aminmax(values.to("iris"), dim=-1)):
        check(expected, actual)
    compare_backward(lambda x: x.nansum((0, 2), keepdim=True), values)
    ints = torch.tensor([[2**48, -7], [-9, 2**48]], dtype=torch.int64)
    check(ints.nansum(), ints.to("iris").nansum())
    check(torch.empty(0, 2).amax(1), torch.empty(0, 2, device="iris").amax(1))


def test_index_copy_fill_take_and_put():
    base = torch.arange(12.0).reshape(3, 4).T
    index = torch.tensor([2, 0])
    source = torch.arange(8.0).reshape(2, 4).T
    compare_backward(lambda x, y: x.index_copy(1, index.to(x.device), y), base, source)
    compare_backward(lambda x: x.index_fill(0, torch.tensor([-1, 1], device=x.device), -5), base)
    take_at = torch.tensor([[-1, 6], [6, 0]])
    compare_backward(lambda x: x.take(take_at.to(x.device)), base)
    x, y = base.clone(), base.to("iris")
    alias = y[:, 1]
    values = torch.arange(4.0).reshape(2, 2)
    x.put_(take_at, values, accumulate=True)
    y.put_(take_at.to("iris"), values.to("iris"), accumulate=True)
    check(x, y, check_stride=True)
    check(x[:, 1], alias)


@pytest.mark.parametrize("dtype", [None, torch.int32, torch.int64])
@pytest.mark.parametrize("rows,columns,offset", [
    (5, 5, 0), (17, 5, -3), (3, 12, 2), (7, 4, 9), (4, 7, -4),
    (0, 5, 0), (5, 0, 2), (2, 5, -(2**63)), (2**31 + 3, 2, -(2**31 + 2)),
])
def test_tril_indices_factory(rows, columns, offset, dtype):
    expected = torch.tril_indices(rows, columns, offset, dtype=dtype, device="cpu")
    before = torch_iris._C._kernel_submissions()
    actual = torch.tril_indices(rows, columns, offset, dtype=dtype, device="iris:0")
    assert torch_iris._C._kernel_submissions() - before == bool(expected.numel())
    assert not actual.requires_grad
    check(expected, actual, check_stride=True, rtol=0, atol=0)


def test_tril_indices_out_and_defaults():
    expected = torch.tril_indices(5, 3, -1, dtype=torch.int32)
    storage = torch.full((expected.size(1) + 2, 4), -1, dtype=torch.int32, device="iris")
    out = storage[1:-1, ::2].T
    assert torch.ops.aten.tril_indices.out(5, 3, -1, out=out) is out
    host = torch.full(storage.shape, -1, dtype=storage.dtype)
    host[1:-1, ::2].T.copy_(expected)
    check(host, storage, rtol=0, atol=0)
    empty = torch.empty(0, dtype=torch.int32, device="iris")
    assert torch.ops.aten.tril_indices.out(5, 3, -1, out=empty) is empty
    check(expected, empty, check_stride=True, rtol=0, atol=0)
    with torch.device("iris"):
        assert torch.tril_indices(5, 3).device == torch.device("iris:0")


@pytest.mark.parametrize("rows,columns,kwargs,message", [
    (-1, 3, {}, "non-negative"),
    (3, -1, {}, "non-negative"),
    (3, 3, {"dtype": torch.float32}, "int32 or int64"),
    (3, 3, {"pin_memory": True}, "pinned memory"),
    (3, 3, {"device": "iris:1"}, "iris:0"),
    (2**62, 2**62, {}, "size overflow"),
])
def test_tril_indices_rejects_invalid_inputs(rows, columns, kwargs, message):
    with pytest.raises(RuntimeError, match=message):
        torch.tril_indices(rows, columns, **({"device": "iris"} | kwargs))


@pytest.mark.parametrize("mode", ["sum", "prod", "mean", "amin", "amax"])
@pytest.mark.parametrize("include_self", [False, True])
def test_scatter_reduce_with_duplicates(mode, include_self):
    index = torch.tensor([[1, 0, 1], [0, 0, 1]], dtype=torch.long)
    initial = torch.tensor([[2, -3, 4], [1, 5, 7.0]])
    values = torch.tensor([[2, 3, -2], [1, 1, -5.0]])
    def reduce(x, y):
        return x.scatter_reduce(1, index.to(x.device), y, mode, include_self=include_self)
    compare_backward(reduce, initial, values)
    check(reduce(initial.long(), values.long()), reduce(initial.long().to("iris"), values.long().to("iris")))


@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
def test_embedding_bag(mode):
    ids = torch.tensor([0, 2, 2, 0, 3, 4, 3])
    offsets = torch.tensor([0, 0, 3, 5, 7])
    weights = torch.arange(30.0).view(6, 5).T.contiguous().T
    sample = torch.rand(7)
    def forward(weight, *samples):
        return F.embedding_bag(
            ids.to(weight.device), weight, offsets.to(weight.device), mode=mode,
            include_last_offset=True, padding_idx=0, per_sample_weights=samples[0] if samples else None,
        )
    compare_backward(forward, weights, *((sample,) if mode == "sum" else ()))


def test_adaptive_max_pool_and_fold():
    x = torch.randn(2, 3, 8, 12).transpose(2, 3)
    for op in (lambda t: F.adaptive_max_pool2d(t, (5, 3)),
               lambda t: F.unfold(t, (3, 2), (2, 1), (1, 2), (2, 1))):
        compare_backward(op, x)
    indices = F.adaptive_max_pool2d(x, (5, 3), return_indices=True)[1]
    check(indices, F.adaptive_max_pool2d(x.to("iris"), (5, 3), return_indices=True)[1])
    cols = F.unfold(x, (3, 2), padding=1)
    compare_backward(lambda t: F.fold(t, (12, 8), (3, 2), padding=1), cols)


@pytest.mark.parametrize("mode, antialias", [("bilinear", True), ("bicubic", False), ("bicubic", True)])
@pytest.mark.parametrize("align", [False, True])
def test_resize_forward_backward(mode, antialias, align):
    x = torch.rand(2, 3, 8, 12).transpose(2, 3)
    for size in ((5, 6), (17, 15), (1, 1)):
        compare_backward(
            lambda t: F.interpolate(t, size=size, mode=mode, align_corners=align, antialias=antialias),
            x, rtol=7e-5, atol=5e-6,
        )


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_regression_losses(reduction):
    x = torch.tensor([[0.0, 0.1, 0.3], [0.5, 0.9, 1.0]]).T
    y = torch.tensor([[1.0, 0.2, 0.3], [0.3, 1.0, 0.0]]).T
    compare_backward(lambda t: F.binary_cross_entropy(t, y.to(t.device), reduction=reduction), x)
    compare_backward(lambda a, b: F.smooth_l1_loss(a, b, reduction=reduction, beta=0.4), x, y)
    compare_backward(lambda a, b: F.huber_loss(a, b, reduction=reduction, delta=0.3), x, y)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.bfloat16, torch.complex64])
def test_software_dtypes(dtype):
    x = torch.tensor([[11, -42, 13], [17, -51, 52], [-21, 14, -29]], dtype=torch.float32).T
    if dtype == torch.complex64:
        x = torch.complex(x, x / 7)
    else:
        x = x.to(dtype)
    dx = x.to("iris")
    for fn in (lambda t: t.clone(), lambda t: t + t, lambda t: t * t,
               lambda t: t @ t, lambda t: t.sum(), lambda t: torch.ones_like(t)):
        check(fn(x), fn(dx))
    if not x.is_complex():
        check(x.float(), dx.float())
    if dtype == torch.complex64:
        view = torch.view_as_real(dx)
        assert torch._C._is_alias_of(view, dx)
        view[0, 0, 1] = -3
        assert dx[0, 0].item().imag == -3
        check(x.conj_physical(), x.to("iris").conj_physical())
        real = torch.tensor([0, -0.0, 0])
        only_imaginary = torch.complex(real, torch.tensor([0, 0, 1.0]))
        check(only_imaginary.nonzero(), only_imaginary.to("iris").nonzero())
        with pytest.raises(RuntimeError, match="not supported for complex"):
            dx.amax()


def test_bool_and_mixed_inplace_multiply():
    values = torch.arange(24, dtype=torch.float32).reshape(4, 6).T
    mask = values > 4
    check(values.clone().mul_(mask), values.to("iris").mul_(mask.to("iris")), check_stride=True)
    check(mask.clone().mul_(~mask), mask.to("iris").mul_(~mask.to("iris")))
    x = torch.arange(5)
    check(torch.combinations(x), torch.combinations(x.to("iris")))


@pytest.mark.parametrize("size", [31, 64, 100])
def test_fft_forward_backward(size):
    x = torch.randn(2, size)
    compare_backward(lambda t: torch.fft.rfft(t, norm="ortho").real, x, rtol=1e-4, atol=2e-5)
    check(x, torch.fft.irfft(torch.fft.rfft(x.to("iris")), n=size), rtol=1e-4, atol=2e-5)


def test_fft2_and_stft():
    x = torch.randn(7, 16)
    check(torch.fft.rfft2(x, norm="forward"), torch.fft.rfft2(x.to("iris"), norm="forward"), atol=2e-6, rtol=2e-5)
    def stft(t):
        result = torch.stft(t, 8, hop_length=2, window=torch.hann_window(8, device=t.device), return_complex=True)
        return torch.view_as_real(result)
    compare_backward(stft, x, rtol=2e-4, atol=2e-5)


def test_linalg_factor_orientation_and_solve():
    x = torch.tensor([[2.0, 1, 0], [4, 1, 3], [-1, 2, 1]])
    rhs = torch.randn(3, 2)
    for matrix in (x, x.T):
        compare_backward(torch.linalg.det, matrix)
        compare_backward(lambda t: torch.linalg.slogdet(t).logabsdet, matrix)
        compare_backward(torch.linalg.solve, matrix, rhs)
        compare_backward(lambda t: torch.linalg.cholesky(t.mT @ t), matrix)
        compare_backward(lambda t: torch.linalg.qr(t).R, matrix, rtol=3e-4, atol=1e-5)
    compare_backward(lambda a, b: torch.linalg.solve_triangular(a.triu(), b, upper=True), x, rhs)
    compare_backward(torch.linalg.matrix_exp, torch.stack((x * 0.1, x)), rtol=2e-4, atol=2e-5)


def test_device_state_events_and_compiled_chain():
    torch_iris.init()
    assert torch_iris.is_initialized()
    torch_iris.manual_seed(37)
    snapshot = torch_iris.get_rng_state()
    values = torch.randn(23, device="iris")
    torch_iris.set_rng_state(snapshot)
    check(values.cpu(), torch.randn_like(values), rtol=0, atol=0)
    assert torch_iris.initial_seed() == 37
    torch_iris.reset_peak_memory_stats()
    before = torch_iris.memory_allocated()
    aux = torch.empty(4096, device="iris")
    assert torch_iris.max_memory_allocated() >= before + aux.nbytes
    stream = torch_iris.Stream()
    torch_iris.synchronize()
    with torch_iris.stream(stream):
        assert torch_iris.current_stream() == stream
        compiled = torch.compile(lambda x: x * 2.3 + 1.4, backend="iris", fullgraph=True)
        x = values.detach().requires_grad_()
        before = torch_iris._C._kernel_submissions()
        result = compiled(x)
        assert torch_iris._C._kernel_submissions() - before == 1
        done = torch_iris.Event()
        done.record(stream)
    done.synchronize()
    assert torch_iris.current_stream() == torch_iris.default_stream()
    check(values.cpu() * 2.3 + 1.4, result, rtol=0, atol=0)
    grad = torch.autograd.grad(result.sum(), x)[0]
    check(torch.full((values.numel(),), 2.3), grad)


def test_dlpack_strided_offset_and_lifetime():
    x = torch.arange(42, device="iris", dtype=torch.float32).reshape(6, 7).T[2:6, 1:5]
    capsule = torch_iris.to_dlpack(x)
    other = torch_iris.from_dlpack(capsule)
    assert other.storage_offset() == x.storage_offset()
    assert other.stride() == x.stride()
    check(x.cpu(), other)
    other[0, 1] = -42
    assert x[0, 1].item() == -42
    expected = x.cpu()
    del x, capsule
    check(expected, other)
    empty = torch.empty((0, 3), device="iris")
    check(empty.cpu(), torch_iris.from_dlpack(torch_iris.to_dlpack(empty)), check_stride=True)


def test_torchscript_trace_with_cpu_reference():
    model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2)).eval()
    from copy import deepcopy

    gpu = deepcopy(model).to("iris")
    example = torch.arange(12.0).reshape(3, 4) / 12
    traced = torch.jit.trace(gpu, example.to("iris"), check_trace=False)
    with torch.no_grad():
        for sample in (example, example[:2].neg()):
            check(model(sample), traced(sample.to("iris")))


def test_opencl_memory_and_fp64_properties():
    properties = torch_iris.get_device_properties(0)
    assert 0 < properties.max_allocation <= properties.total_memory
    assert isinstance(properties.host_unified_memory, bool)
    assert properties.double_fp_config >= 0
    if properties.double_fp_config == 0:
        with pytest.raises(RuntimeError, match="storage"):
            torch.ones(2, device="iris", dtype=torch.float64)


def test_torchvision_detection_backward():
    tv = pytest.importorskip("torchvision").ops
    sample = torch.randn(2, 3, 8, 8)
    boxes = torch.tensor([[0, -3, -2, 7, 5], [1, 0, 1, 7, 8]], dtype=torch.float32)
    for aligned in (False, True):
        compare_backward(lambda t: tv.roi_align(t, boxes.to(t.device), (3, 4), sampling_ratio=-1, aligned=aligned),
                         sample, rtol=1e-4, atol=1e-5)
    compare_backward(lambda t: tv.roi_pool(t, boxes.to(t.device), (3, 4)), sample)
    weight = torch.randn(4, 1, 3, 3) * 0.2
    offsets = torch.randn(2, 36, 8, 8) * 0.2
    mask = torch.rand(2, 18, 8, 8)
    compare_backward(lambda x, y, z, m: tv.deform_conv2d(x, y, z, padding=1, mask=m),
                     sample[:, :2], offsets, weight, mask, rtol=3e-4, atol=3e-5)


def test_sparse_coo_and_sparse_adam():
    indices = torch.tensor([[0, 0, 2, 1], [1, 1, 1, 2]])
    values = torch.tensor([1.0, 2, 3, 5])
    def sparse_mm(value, rhs):
        sparse = torch.sparse_coo_tensor(indices.to(value.device), value, (3, 3), check_invariants=False)
        return torch.sparse.mm(sparse, rhs)
    compare_backward(sparse_mm, values, torch.randn(3, 4))
    from copy import deepcopy
    cpu = torch.nn.Embedding(16, 4, sparse=True)
    iris = deepcopy(cpu).to("iris")
    rows = torch.tensor([[1, 4, 1], [7, 3, 7]])
    opt_cpu = torch.optim.SparseAdam(cpu.parameters(), lr=0.01)
    opt_iris = torch.optim.SparseAdam(iris.parameters(), lr=0.01)
    for _ in range(3):
        for model, opt in ((cpu, opt_cpu), (iris, opt_iris)):
            opt.zero_grad(set_to_none=True)
            model(rows.to(model.weight.device)).square().sum().backward()
            model(rows[:, 1:].to(model.weight.device)).sum().backward()
            opt.step()
        check(cpu.weight, iris.weight)
        for key in ("exp_avg", "exp_avg_sq"):
            check(opt_cpu.state[cpu.weight][key], opt_iris.state[iris.weight][key])
