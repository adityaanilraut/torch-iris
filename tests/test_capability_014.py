"""CPU parity for the new device-resident Iris dispatcher paths."""

import pytest
import torch
import torch.nn.functional as F

import torch_iris


pytestmark = pytest.mark.skipif(not torch_iris.is_available(), reason="Iris device unavailable")
DEVICE = torch.device("iris:0")


@pytest.mark.parametrize("stride,padding", [(1, (1, 1)), (2, (0, 1))])
@pytest.mark.parametrize("dtype", [torch.float32, pytest.param(torch.float16, marks=pytest.mark.requires_fp16)])
def test_depthwise_3x3_strided_inputs_and_gradients(stride, padding, dtype):
    g = torch.Generator().manual_seed(772)
    host = [torch.randn(shape, generator=g).to(dtype) / 4 for shape in ((3, 5, 11, 30), (5, 1, 3, 3), (6,))]
    device = [x.to(DEVICE) for x in host]

    def forward_backward(tensors):
        x, w, b = tensors
        x, w, b = (x[1:, :, :, ::2], w, b[1:])
        x, w, b = (t.detach().requires_grad_() for t in (x, w, b))
        y = F.conv2d(x, w, b, stride=stride, padding=padding, groups=5)
        return y, *torch.autograd.grad(y.square().mean(), (x, w, b))

    for actual, expected in zip(forward_backward(device), forward_backward(host), strict=True):
        close(actual, expected, rtol=4e-3 if dtype == torch.float16 else 2e-4,
              atol=4e-4 if dtype == torch.float16 else 2e-6)


def close(got, expected, **kwargs):
    assert got.device == DEVICE
    torch.testing.assert_close(got.cpu(), expected, **kwargs)


def parity(operation, *inputs, rtol=1e-4, atol=2e-6):
    host = [value.detach().requires_grad_() for value in inputs]
    iris = [value.detach().to(DEVICE).requires_grad_() for value in inputs]
    expected, actual = operation(*host), operation(*iris)
    close(actual, expected, atol=atol, rtol=rtol)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for output, reference in zip(iris, host):
        close(output.grad, reference.grad, atol=atol, rtol=rtol)


def test_pointwise_promotion_signed_zero_strides_and_aliases():
    x = torch.tensor([[-0., 3., torch.nan], [4., -2., 0.]]).T
    y = torch.tensor([[0., -4., 1.], [-3., torch.nan, -0.]]).T
    a, b = x.to(DEVICE), y.to(DEVICE)
    for fn in (torch.copysign, torch.fmax, torch.fmin, torch.hypot, torch.logaddexp, torch.xlogy):
        reference = fn(x, y)
        out = torch.empty_strided(x.shape, x.stride(), device=DEVICE)
        assert fn(a, b, out=out) is out
        close(out, reference, equal_nan=True)
        assert out.stride() == x.stride()
    close(torch.signbit(torch.copysign(a, b)), torch.signbit(torch.copysign(x, y)))
    target = torch.empty(0, device=DEVICE).set_(a[1:])
    assert target.stride() == a.stride() and target.storage_offset() == a.stride(0)
    target[0, 1] = 31
    assert a[1, 1].item() == 31
    close(torch.fmax(a, b, out=a), torch.fmax(torch.tensor([[-0., 3., torch.nan], [4., 31., 0.]]).T, y),
          equal_nan=True)
    integers = torch.tensor([-2**63 + 1, -3, 3, 2**62, 7, 9])
    shifts = torch.tensor([63, 2, 1, 1, 64, -1])
    close(integers.to(DEVICE) << shifts.to(DEVICE), integers << shifts)
    close(integers.to(DEVICE) >> shifts.to(DEVICE), integers >> shifts)
    close(torch.hypot(torch.tensor([torch.inf], device=DEVICE), torch.tensor([torch.nan], device=DEVICE)),
          torch.tensor([torch.inf]))
    close(torch.logspace(-2, 2, 9, device=DEVICE), torch.logspace(-2, 2, 9), rtol=0, atol=1e-6)


def test_reductions_values_indices_and_backward():
    data = torch.tensor([[2., 0., -3., 2.], [1., 1., 3., 1.]]).T.contiguous().T
    parity(lambda x: x.cumprod(-1), data)
    parity(lambda x: x.logcumsumexp(-1), data)
    parity(lambda x: x.logsumexp(0), data)
    data[1, 2:] = torch.nan
    resident = data.to(DEVICE)
    for operation in (torch.cummax, torch.cummin):
        expected, actual = operation(data, -1), operation(resident, -1)
        close(actual.values, expected.values, equal_nan=True)
        close(actual.indices, expected.indices)
    values, indices = torch.nanmedian(resident, dim=1)
    close(values, torch.nanmedian(data, dim=1).values)
    close(resident.gather(1, indices[:, None]).flatten(), values.cpu())
    labels = torch.tensor([[2, 2, 0], [1, 0, 1]], dtype=torch.int16)
    operations = (
        lambda t: torch.unique(t, return_inverse=True, return_counts=True),
        lambda t: torch.unique_consecutive(t, dim=1, return_inverse=True, return_counts=True),
    )
    for fn in operations:
        for output, reference in zip(fn(labels.to(DEVICE)), fn(labels)):
            close(output, reference)
    close(torch.bincount(labels.long().flatten().to(DEVICE), minlength=5),
          torch.bincount(labels.long().flatten(), minlength=5))


def test_special_and_distance_gradients():
    x = torch.tensor([-2.75, -0.4, 0.18, 0.49, 1., 2., 6., 10.])
    for fn in (torch.lgamma, torch.digamma, torch.i0):
        parity(fn, x, rtol=2e-4, atol=5e-6)
    parity(torch.erfinv, torch.tensor([-0.9999, -0.8, -0., 0.15, 0.73, 0.98]), rtol=3e-4, atol=4e-6)
    g = torch.Generator().manual_seed(714)
    matrix, other = torch.randn(5, 4, generator=g).T, torch.randn(2, 7, 5, generator=g)
    parity(lambda a, b: torch.cdist(a, b, p=1.5), matrix, other, atol=3e-5)
    parity(lambda a: torch.pdist(a, p=2.5), matrix, atol=2e-5)


def test_condensed_distance_order_empty_shapes_and_gradients():
    data = torch.randn(7, 33, generator=torch.Generator().manual_seed(4)).T
    for values in (data, data[:32], data[:31], data[:1], data[:2], data[:, :0]):
        parity(torch.pdist, values)
    # CPU PyTorch 2.12.1 divides by zero in backward for a zero-row matrix.
    empty = data[:0].to(DEVICE).requires_grad_()
    distances = torch.pdist(empty)
    close(distances, torch.pdist(data[:0]))
    distances.sum().backward()
    close(empty.grad, data[:0])


def test_bucketize_integer_bounds_and_strided_out():
    base = 2**40
    boundaries = torch.tensor([0, base, 0, base + 1, 0, base + 8])[1::2]
    values = torch.tensor([base - 1, base, base + 1, base + 2, base + 8, base + 9]).reshape(2, 3).T
    keys, limits = values.to(DEVICE), boundaries.to(DEVICE)
    for right in (False, True):
        out = torch.empty_strided(values.shape, values.stride(), device=DEVICE, dtype=torch.int32)
        assert torch.bucketize(keys, limits, right=right, out_int32=True, out=out) is out
        close(out, torch.bucketize(values.contiguous(), boundaries.contiguous(), right=right, out_int32=True))
        assert out.stride() == values.stride()
        offset = torch.tensor([0, base, 0, base + 1, 0, base + 8], device=DEVICE)[1::2]
        close(torch.bucketize(base + 1, offset, right=right), torch.bucketize(base + 1, boundaries.contiguous(), right=right))
        close(torch.bucketize(keys, limits[:0], right=right), torch.zeros(values.shape, dtype=torch.long))
    large_mixed = torch.tensor([0, 1, 2147483647], dtype=torch.int32)
    close(torch.bucketize(large_mixed.to(DEVICE), limits), torch.bucketize(large_mixed, boundaries.contiguous()))


def test_spatial_and_margin_gradients():
    gen = torch.Generator().manual_seed(93)
    a = torch.randn(2, 2, 3, 4, 5, generator=gen).transpose(-1, -2)
    grid = torch.randn(2, 2, 3, 3, 3, generator=gen)
    for padding in ("zeros", "border", "reflection"):
        parity(lambda x, g: F.grid_sample(x, g, padding_mode=padding, align_corners=False),
               a, grid, atol=2e-5)
    for op in (
        lambda t: F.pad(t, (1, 2, -1, 1, 1, 0), "reflect"),
        lambda t: F.pad(t, (1, 2, 1, 1, 1, 0), "replicate"),
        lambda t: F.interpolate(t, scale_factor=(1.5, 0.75, 1.25), mode="trilinear", align_corners=False),
        lambda t: F.interpolate(t, size=(5, 3, 6), mode="nearest"),
        lambda t: F.adaptive_avg_pool3d(t, (4, 3, 2)),
    ):
        parity(op, a, atol=2e-5)
    labels = torch.tensor([[1, 1, 3, -1, 0], [2, -1, 0, 0, 0]])
    scores = torch.randn(2, 5, generator=gen)
    parity(lambda x: F.multilabel_margin_loss(x, labels.to(x.device), reduction="none"), scores)


def test_ctc_padded_packed_lengths_and_impossible_alignment():
    p = torch.randn(9, 3, 5, generator=torch.Generator().manual_seed(77)).log_softmax(-1)
    padded = torch.tensor([[1, 1, 2], [3, 0, 0], [1, 1, 0]])
    packed = torch.tensor([1, 1, 2, 3, 1, 1])
    for targets, tensor_lengths in ((padded, False), (packed, True)):
        def loss(x):
            lengths = (torch.tensor([9, 6, 2]), torch.tensor([3, 1, 2])) if tensor_lengths else ([9, 6, 2], [3, 1, 2])
            if tensor_lengths:
                lengths = tuple(length.to(x.device) for length in lengths)
            return F.ctc_loss(x, targets.to(x.device), *lengths, reduction="none", zero_infinity=True)

        parity(loss, p, rtol=4e-4, atol=5e-5)


def test_random_repeatability_and_moments():
    def sample():
        generator = torch.Generator(device=DEVICE).manual_seed(513)
        rates = torch.tensor([0., 2., 12., 80.], device=DEVICE).expand(5000, 4)
        return torch.poisson(rates, generator=generator), torch.randperm(1000, device=DEVICE, generator=generator)

    counts, order = sample()
    repeat_counts, repeat_order = sample()
    close(counts, repeat_counts.cpu())
    close(order, repeat_order.cpu())
    close(order.sort().values, torch.arange(1000))
    assert (counts >= 0).all().item() and (counts == counts.floor()).all().item()
    close(counts.mean(0), torch.tensor([0., 2., 12., 80.]), atol=0.4, rtol=0)


def test_batched_svd_and_eigh_reconstruction_strides_and_gradients():
    generator = torch.Generator().manual_seed(541)
    a = torch.randn(2, 5, 8, generator=generator).mT
    a[1, :, 4] = a[1, :, 0] + a[1, :, 2]
    for matrix in (a, a.mT):
        x = matrix.to(DEVICE)
        u, s, vh = torch.linalg.svd(x, full_matrices=True)
        k = min(x.shape[-2:])
        eye_u = torch.eye(x.size(-2)).expand_as(u.cpu())
        eye_v = torch.eye(x.size(-1)).expand_as(vh.cpu())
        close(u.mT @ u, eye_u, atol=6e-6, rtol=2e-5)
        close(vh @ vh.mT, eye_v, atol=6e-6, rtol=2e-5)
        close((u[..., :k] * s.unsqueeze(-2)) @ vh[..., :k, :], matrix, atol=7e-6, rtol=3e-5)
        close(s, torch.linalg.svdvals(matrix), atol=2e-6, rtol=1e-5)
        assert u.stride(-2) == 1 and vh.stride(-2) == 1
    parity(torch.linalg.svdvals, a[:1])
    base = torch.randn(2, 5, 5, generator=generator)
    symmetric = base + base.mT
    x = symmetric.to(DEVICE)
    w, q = torch.linalg.eigh(x, UPLO="U")
    close(w, torch.linalg.eigvalsh(symmetric), atol=4e-6, rtol=2e-5)
    close((q * w.unsqueeze(-2)) @ q.mT, symmetric, atol=8e-6, rtol=3e-5)
    close(q.mT @ q, torch.eye(5).expand(symmetric.shape), atol=5e-6, rtol=2e-5)
    parity(lambda t: torch.linalg.eigvalsh(t + t.mT), base, atol=3e-5)


def test_pinv_lstsq_and_device_matched_triangular_solve():
    rng = torch.Generator().manual_seed(89)
    a = torch.randn(2, 7, 4, generator=rng)
    a[1, :, 3] = a[1, :, 2]
    b = torch.randn(2, 7, 2, generator=rng)
    x, y = a.to(DEVICE), b.to(DEVICE)
    out = torch.empty_strided((2, 4, 7), (28, 1, 4), device=DEVICE)
    assert torch.linalg.pinv(x, out=out) is out
    close(out, torch.linalg.pinv(a), rtol=2e-4, atol=3e-6)
    timed = torch.linalg.lstsq(x, y, driver="gelsd")
    reference = torch.linalg.lstsq(a, b, driver="gelsd")
    for actual, expected in zip(timed, reference):
        close(actual, expected, atol=5e-6, rtol=2e-4)
    ill = torch.diag_embed(torch.tensor([[2., 0.02, 1e-5], [-3., 0.03, 0.]]))
    close(torch.linalg.pinv(ill.to(DEVICE), hermitian=True, atol=2e-5),
          torch.linalg.pinv(ill, hermitian=True, atol=2e-5), atol=2e-6, rtol=2e-4)
    close(torch.linalg.pinv(ill.to(DEVICE), rtol=torch.tensor([1e-7, 1e-3], device=DEVICE)),
          torch.linalg.pinv(ill, rtol=torch.tensor([1e-7, 1e-3])), atol=2e-6, rtol=2e-4)
    q, _ = torch.linalg.qr(torch.randn(2, 7, 4, generator=rng))
    v, _ = torch.linalg.qr(torch.randn(2, 4, 4, generator=rng))
    scaled = (q * torch.tensor([1., 0.1, 0.001, 1e-5])) @ v.mT
    close(torch.linalg.svdvals(scaled.to(DEVICE)), torch.linalg.svdvals(scaled), atol=2e-7, rtol=2e-4)
    known = torch.randn(2, 4, 2, generator=rng)
    observed = torch.linalg.lstsq(scaled.to(DEVICE), (scaled @ known).to(DEVICE), driver="gelsd")
    close(observed.solution, known, atol=0.01, rtol=0.005)
    parity(torch.linalg.pinv, a[:1], atol=1e-5)
    parity(lambda x, rhs: torch.linalg.lstsq(x, rhs, driver="gelss").solution, a[:1], b[:1], atol=2e-5)
    upper = torch.randn(5, 5, generator=rng).triu(1)
    rhs = torch.randn(5, 2, generator=rng)
    expected = torch.triangular_solve(rhs, upper + torch.eye(5)).solution
    resident = upper.to(DEVICE) + torch.eye(5, device=DEVICE)
    close(torch.triangular_solve(rhs.to(DEVICE), resident).solution, expected)


def test_sparse_degree_normalization_and_accumulated_sparseadam():
    keys = torch.tensor([[0, 0, 0, 1, 2], [0, 0, 1, 2, 3]])
    weights = torch.tensor([1., 3., 2., 4., 5.])
    features = torch.arange(12.).reshape(4, 3)

    def graph(value, dense):
        sparse = torch.sparse_coo_tensor(keys.to(value.device), value, (3, 4), check_invariants=False)
        degree = torch.sparse.sum(sparse, dim=1).to_dense().clamp_min(1)
        return torch.sparse.mm(sparse, dense) / degree.unsqueeze(-1)

    parity(graph, weights, features, atol=3e-5)
    cpu = torch.nn.Embedding(9, 3, sparse=True)
    gpu = torch.nn.Embedding.from_pretrained(cpu.weight.detach().to(DEVICE), freeze=False, sparse=True)
    optimizer = (torch.optim.SparseAdam(cpu.parameters()), torch.optim.SparseAdam(gpu.parameters()))
    for _ in range(2):
        for model, opt in zip((cpu, gpu), optimizer):
            opt.zero_grad()
            for indices in ([0, 2, 2], [2, 5, 8]):
                model(torch.tensor(indices, device=model.weight.device)).sum().backward()
            opt.step()
        close(gpu.weight, cpu.weight, atol=5e-6, rtol=2e-5)
