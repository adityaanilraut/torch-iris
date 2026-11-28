import pytest
import torch
import torch_iris


pytestmark = pytest.mark.skipif(not torch_iris.is_available(), reason="requires Intel Iris Xe")


def assert_values(actual, expected, **kwargs):
    torch.testing.assert_close(actual.cpu(), expected, **kwargs)


def complex_fixture():
    real = torch.arange(1., 31.).view(5, 6)
    return torch.complex(real, real.flip((1,)) - 11)


@pytest.mark.parametrize("view", [torch.conj, lambda x: x.H, torch.adjoint], ids=["conj", "H", "adjoint"])
@pytest.mark.parametrize("strided", [False, True])
def test_lazy_conjugate(view, strided):
    host = complex_fixture()
    device = host.to("iris")
    if strided:
        host, device = host[1:, 1::2].T, device[1:, 1::2].T
    expected, actual = view(host), view(device)
    assert actual.is_conj()
    assert actual.stride() == expected.stride()
    assert actual.storage_offset() == expected.storage_offset()
    assert_values(actual, expected)
    resolved = actual.resolve_conj()
    assert not resolved.is_conj()
    assert_values(resolved, expected)
    assert actual.imag.is_neg()
    assert_values(actual.imag, expected.imag)
    assert_values(actual.imag.square(), expected.imag.square())
    assert_values(actual.imag + 2, expected.imag + 2)
    assert_values(actual * (1 + 2j), expected * (1 + 2j))
    assert_values(actual.conj(), host if view is torch.conj else host.T)


def test_conjugate_copy_into_views_and_across_devices():
    source = complex_fixture()[:1, 1::2]
    for source_device, destination_device in (("cpu", "iris"), ("iris", "cpu"), ("iris", "iris")):
        for source_conj in (False, True):
            for destination_conj in (False, True):
                def copy(src, target):
                    if source_conj:
                        src = src.conj()
                    dst = target[1::2, ::2]
                    if destination_conj:
                        dst = dst.conj()
                    assert dst.copy_(src) is dst
                    return target

                expected = copy(source, torch.zeros((5, 6), dtype=source.dtype))
                actual = copy(source.to(source_device), torch.zeros((5, 6), dtype=source.dtype,
                                                                   device=destination_device))
                assert_values(actual, expected)

    # Rank seven exercises the geometry-buffer route. Check the sign bit of zero.
    x = torch.complex(torch.tensor([0., -0.]), torch.tensor([0., -0.])).reshape(1, 1, 2, 1, 1, 1, 1)
    actual = x.to("iris").conj().resolve_conj().cpu()
    assert_values(torch.signbit(actual.imag), torch.signbit(x.conj().imag))


def test_conj_physical_and_backward():
    host = complex_fixture()
    device = host.to("iris")
    assert_values(device.conj_physical(), host.conj_physical())
    assert_values(device.conj().conj_physical(), host.conj().conj_physical())
    for x in (host, device):
        x[1:, ::2].conj().conj_physical_()
    assert_values(device, host)

    def gradient(x):
        x = x.requires_grad_()
        return torch.autograd.grad((x.H.imag * 2 + x.H.real).sum(), x)[0]

    assert_values(gradient(device), gradient(host))


@pytest.mark.parametrize("dtype", [pytest.param(torch.float16, marks=pytest.mark.requires_fp16),
                                  torch.bfloat16, torch.int64])
def test_negative_views_resolve_and_cast(dtype):
    expected = torch._neg_view(torch.arange(12).to(dtype).view(3, 4).T[1:])
    actual = torch._neg_view(torch.arange(12).to(dtype).to("iris").view(3, 4).T[1:])
    assert actual.is_neg()
    assert_values(actual, expected)
    assert_values(actual.resolve_neg(), expected)
    assert_values(actual.float(), expected.float())


@pytest.mark.parametrize("size", [8, 9])
@pytest.mark.parametrize("norm", ["backward", "forward", "ortho"])
def test_fft_views_and_roundtrips(size, norm):
    host = torch.tensor([1., 2., 4., 3., -1., 6., 5., 0., 3., -4., 1., 0.]).repeat(2, 2).T[1::2].T
    device = torch.empty_strided(host.shape, host.stride(), device="iris").copy_(host)
    for name in ("fft", "ifft", "rfft", "ihfft", "hfft"):
        transform = getattr(torch.fft, name)
        expected = transform(host, n=size, norm=norm)
        actual = transform(device, n=size, norm=norm)
        assert_values(actual, expected, rtol=2e-4, atol=3e-5)
        assert_values(actual.resolve_conj(), expected, rtol=2e-4, atol=3e-5)

    def roundtrip(x):
        complex_input = torch.complex(x, x.flip((-1,))).conj()
        return torch.fft.ifft(torch.fft.fft(complex_input, n=size, norm=norm), norm=norm)

    def hermitian_roundtrip(x):
        return torch.fft.hfft(torch.fft.ihfft(x, n=size, norm=norm), n=size, norm=norm)

    assert_values(roundtrip(device), roundtrip(host), rtol=2e-4, atol=3e-5)
    assert_values(hermitian_roundtrip(device), hermitian_roundtrip(host), rtol=2e-4, atol=3e-5)


@pytest.mark.parametrize("length", [2, 1024, 2048])
def test_fft_multidimensional_out_and_backward(length):
    generator = torch.Generator().manual_seed(502)
    host = torch.randn((length, 8), generator=generator).T[::2]
    device = torch.empty_strided(host.shape, host.stride(), device="iris").copy_(host)
    opts = {"dim": (1, 0), "norm": "ortho"}
    reference = torch.fft.rfftn(host, **opts)
    out = torch.empty((reference.size(1), reference.size(0)), device="iris", dtype=torch.complex64).T
    assert torch.fft.rfftn(device, **opts, out=out) is out
    assert_values(out, reference, rtol=3e-4, atol=4e-5)
    assert_values(torch.fft.irfftn(out, s=(length, host.size(0)), **opts), host, rtol=3e-4, atol=4e-5)

    def gradient(x):
        x = x.requires_grad_()
        y = torch.fft.rfftn(x, **opts)
        return torch.autograd.grad((y.real.square() + y.imag.square()).sum(), x)[0]

    assert_values(gradient(device), gradient(host), rtol=3e-4, atol=4e-5)


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32,
                                  torch.int64, pytest.param(torch.float16, marks=pytest.mark.requires_fp16),
                                  torch.bfloat16, torch.float32])
def test_polygamma_promotion_and_out(order, dtype):
    values = [[1, 2, 3], [4, 5, 0]]
    if dtype.is_floating_point:
        values = [[1., -0.75, 3.], [4., 5., 0.]]
    x = torch.tensor(values, dtype=dtype).T
    iris = x.to("iris")
    expected = torch.special.polygamma(order, x)
    assert_values(torch.special.polygamma(order, iris), expected, equal_nan=True)
    out_types = [torch.float32, torch.bfloat16, torch.complex64, *torch_iris.get_amp_supported_dtype()]
    for out_dtype in out_types:
        target = torch.empty((2, 3), dtype=out_dtype).T
        actual = torch.empty((2, 3), dtype=out_dtype, device="iris").T
        expected_out = torch.polygamma(order, x, out=target)
        assert torch.polygamma(order, iris, out=actual) is actual
        assert_values(actual, expected_out, equal_nan=True)
    invalid = torch.empty(3, dtype=torch.int64, device="iris")
    with pytest.raises(RuntimeError, match="cast"):
        torch.polygamma(order, iris, out=invalid)
    assert invalid.shape == (3,)


def test_polygamma_unsupported_cases():
    with pytest.raises(RuntimeError, match="orders zero and one"):
        torch.polygamma(2, torch.ones(2, device="iris"))
    with pytest.raises(RuntimeError, match="requires real inputs"):
        torch.polygamma(1, torch.ones(2, device="iris", dtype=torch.complex64))


def test_polygamma_order_zero_backward():
    host = torch.tensor([0.25, 1., 2., 6.])

    def gradient(x):
        x = x.requires_grad_()
        return torch.autograd.grad(torch.polygamma(0, x).sum(), x)[0]

    assert_values(gradient(host.to("iris")), gradient(host))
