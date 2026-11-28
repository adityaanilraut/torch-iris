import pytest
import torch
import torch_iris


pytestmark = pytest.mark.skipif(not torch_iris.is_available(), reason="requires Intel Iris Xe")


@pytest.fixture(autouse=True)
def strict_gpu():
    previous = torch_iris.is_cpu_fallback_enabled()
    torch_iris.set_cpu_fallback_enabled(False)
    try:
        yield
    finally:
        torch_iris.set_cpu_fallback_enabled(previous)


def assert_result(actual, expected, *, check_signed_zero=True):
    assert actual.device.type == "iris"
    result = actual.cpu()
    if expected.is_complex():
        result, expected = torch.view_as_real(result), torch.view_as_real(expected)
    torch.testing.assert_close(result, expected, equal_nan=True)
    if check_signed_zero and expected.is_floating_point():
        zeros = expected == 0
        assert torch.equal(result[zeros].signbit(), expected[zeros].signbit())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.complex64])
@pytest.mark.parametrize("name", ["abs", "sgn", "neg", "isnan", "isfinite", "isinf"])
def test_typed_unary_values_views_and_mutation(dtype, name):
    inf, nan = float("inf"), float("nan")
    real = torch.tensor([[0., -0., -2., inf, -inf], [nan, 2., -3., -0., 4.]])
    imag = torch.tensor([[-0., 0., 3., nan, 0.], [1., nan, inf, 2., -0.]])
    data = torch.complex(real, imag) if dtype.is_complex else real.to(dtype)
    cpu_base, iris_base = data.clone(), data.to("iris")
    host, iris = cpu_base.T, iris_base.T
    operation = getattr(torch, name)
    expected = operation(host)
    # CPU SIMD complex neg canonicalizes zero in vectorized blocks. Its scalar path
    # flips both sign bits, which is also the Iris convention for every element.
    signed_zero = not (dtype.is_complex and name == "neg")
    actual = operation(iris)
    assert_result(actual, expected, check_signed_zero=signed_zero)
    if not signed_zero:
        components = torch.view_as_real(host)
        zeros = components == 0
        assert torch.equal(torch.view_as_real(actual.cpu())[zeros].signbit(), ~components[zeros].signbit())
    assert_result(operation(iris[:0]), operation(host[:0]))
    assert_result(operation(iris[1, 0]), operation(host[1, 0]))
    if dtype.is_complex:
        assert_result(operation(iris.conj()), operation(host.conj()), check_signed_zero=signed_zero)
    assert_result(iris_base, cpu_base)
    if name == "isnan":
        host_out = torch.empty((2, 10), dtype=torch.bool)[:, ::2].T
        iris_out = torch.empty((2, 10), dtype=torch.bool, device="iris")[:, ::2].T
        assert torch.ops.aten.isnan.out(host, out=host_out) is host_out
        assert torch.ops.aten.isnan.out(iris, out=iris_out) is iris_out
        assert_result(iris_out, host_out)
    if name in {"isnan", "isfinite", "isinf"}:
        return

    out_types = (torch.float32, torch.bfloat16, torch.complex64) if name == "abs" and dtype.is_complex else (dtype,)
    for out_type in out_types:
        cpu_out = torch.full((2, 10), 19, dtype=out_type)
        iris_out = cpu_out.to("iris")
        cpu_view, iris_view = cpu_out[:, ::2].T, iris_out[:, ::2].T
        assert operation(host, out=cpu_view) is cpu_view
        assert operation(iris, out=iris_view) is iris_view
        assert_result(iris_out, cpu_out)
        empty = torch.empty(0, dtype=out_type, device="iris")
        assert operation(iris, out=empty) is empty
        assert_result(empty, expected.to(out_type), check_signed_zero=signed_zero)

    method = name + "_"
    if name == "abs" and dtype.is_complex:
        with pytest.raises(RuntimeError, match="In-place abs"):
            getattr(iris, method)()
        assert_result(iris_base, cpu_base)
    else:
        cpu_view, iris_view = cpu_base[:, 1::2].T, iris_base[:, 1::2].T
        getattr(cpu_view, method)()
        assert getattr(iris_view, method)() is iris_view
        assert_result(iris_base, cpu_base)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.complex64])
@pytest.mark.parametrize("name", ["abs", "sgn", "neg"])
def test_typed_unary_backward(dtype, name):
    data = torch.tensor([[0., -2., 3.], [4., -5., 6.]])
    if dtype.is_complex:
        data = torch.complex(data, data.flip((0, 1)))
        data[0, 0] = 0
    data = data.to(dtype)
    operation = getattr(torch, name)

    def gradient(base):
        x = base.T.requires_grad_()
        y = operation(x)
        gradient = torch.full_like(y, .5 + 1j if y.is_complex() else .5)
        return torch.autograd.grad(y, x, gradient)[0]

    assert_result(gradient(data.to("iris")), gradient(data))
