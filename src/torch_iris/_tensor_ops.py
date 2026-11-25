"""Shape, indexing, and pointwise ATen compositions for Iris."""

import torch

from ._model_ops import _out


_SEARCHSORTED_INTS = (torch.bool, torch.uint8, torch.int32, torch.int64)


def _trace(input):
    if input.ndim != 2:
        raise RuntimeError("trace expects a matrix")
    return input.diagonal().sum()


def _trace_backward(grad, sizes):
    rows, cols = sizes
    return torch.eye(rows, cols, device=grad.device, dtype=grad.dtype) * grad


def _bucketize(input, boundaries, *, out_int32=False, right=False):
    if boundaries.ndim != 1:
        raise RuntimeError("bucketize expects one-dimensional boundaries")
    if not isinstance(input, torch.Tensor):
        input = torch.scalar_tensor(input, device=boundaries.device, dtype=torch.result_type(boundaries, input))
    if input.device != boundaries.device:
        raise RuntimeError("bucketize operands must be on the same device")
    if input.dtype in _SEARCHSORTED_INTS and boundaries.dtype in _SEARCHSORTED_INTS:
        return torch.searchsorted(boundaries, input, out_int32=out_int32, right=right)
    count = boundaries.numel()
    low = input.new_zeros(input.shape, dtype=torch.long)
    high = torch.full_like(low, count)
    for _ in range(count.bit_length()):
        center = ((low + high) // 2).clamp(max=count - 1)
        value = boundaries.index_select(0, center.flatten()).reshape(center.shape)
        before = input < value if right else input <= value
        active = low < high
        low = torch.where(active & ~before, center + 1, low)
        high = torch.where(active & before, center, high)
    return low.int() if out_int32 else low


def _cross(input, other, *, dim=-1):
    if input.ndim != other.ndim or input.size(dim) != 3 or other.size(dim) != 3:
        raise RuntimeError("linalg.cross expects the same rank and length 3 along dim")
    if input.dtype != other.dtype:
        raise RuntimeError("linalg.cross expects matching input dtypes")
    x, y, z = input.unbind(dim)
    a, b, c = other.unbind(dim)
    return torch.stack((y * c - z * b, z * a - x * c, x * b - y * a), dim)


def _batch_add(reduce):
    def run(input, batch1, batch2, *, beta=1, alpha=1):
        if input.dtype != batch1.dtype or input.dtype != batch2.dtype:
            raise RuntimeError("batched matrix product expects matching input dtypes")
        if (batch1.ndim != 3 or batch2.ndim != 3 or batch1.size(0) != batch2.size(0)
                or batch1.size(2) != batch2.size(1)):
            raise RuntimeError("incompatible batch matrix shapes")
        size = (batch1.size(0), batch1.size(1), batch2.size(2))
        bias = input.expand(size[1:] if reduce else size)
        if alpha == 0:
            return torch.zeros_like(bias, memory_format=torch.contiguous_format) if beta == 0 else (bias * beta).contiguous()
        product = batch1.bmm(batch2)
        if reduce:
            product = product.sum(0, dtype=input.dtype)
        return product * alpha if beta == 0 else (bias * beta).add(product, alpha=alpha).contiguous()

    return run


def _clamp_tensor(input, lower=None, upper=None):
    from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND, elementwise_dtypes

    operands = [t for t in (input, lower, upper) if t is not None]
    if len(operands) == 1:
        raise RuntimeError("clamp requires at least one bound")
    _, dtype = elementwise_dtypes(*operands, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH)
    result = input.to(dtype)
    if lower is not None:
        result = torch.maximum(result, lower.to(dtype))
    if upper is not None:
        result = torch.minimum(result, upper.to(dtype))
    return result


def _clamp_tensor_inplace(input, lower=None, upper=None):
    result = _clamp_tensor(input, lower, upper)
    if result.shape != input.shape or not torch.can_cast(result.dtype, input.dtype):
        raise RuntimeError("in-place clamp cannot change shape or cast the result")
    return input.copy_(result)


def _signbit(input):
    bit_dtype = {torch.float32: torch.int32, torch.float16: torch.int16, torch.bfloat16: torch.int16}.get(input.dtype)
    if bit_dtype is not None:
        return input.contiguous().view(bit_dtype) < 0
    if input.is_complex():
        raise RuntimeError("signbit does not support complex input")
    return input < 0


def _nan_to_num(input, nan=None, posinf=None, neginf=None):
    if input.is_complex():
        values = _nan_to_num(torch.view_as_real(input.resolve_conj()), nan, posinf, neginf)
        return torch.view_as_complex(values)
    if not input.is_floating_point():
        return input.clone()
    dtype = input.dtype
    finite = torch.finfo(dtype)
    value = input.float()
    value = torch.where(value.isnan(), 0 if nan is None else nan, value)
    value = torch.where(input.float() == torch.inf, finite.max if posinf is None else posinf, value)
    value = torch.where(input.float() == -torch.inf, finite.min if neginf is None else neginf, value)
    return value.to(dtype)


def _count(input, dim=None):
    return (input != 0).sum(dim=dim, dtype=torch.long)


def register(library):
    library.impl("trace", _trace)
    library.impl("trace_backward", _trace_backward)
    library.impl("count_nonzero", _count)
    library.impl("count_nonzero.dim_IntList", _count)
    library.impl("clamp.Tensor", _clamp_tensor)
    library.impl("clamp.Tensor_out", _out(_clamp_tensor, "out"))
    library.impl("clamp_.Tensor", _clamp_tensor_inplace)
    for name, overload, fn in (
        ("bucketize.Tensor", "bucketize.Tensor_out", _bucketize),
        ("bucketize.Scalar", "bucketize.Scalar_out", _bucketize),
        ("linalg_cross", "linalg_cross.out", _cross),
        ("baddbmm", "baddbmm.out", _batch_add(False)),
        ("addbmm", "addbmm.out", _batch_add(True)),
        ("signbit", "signbit.out", _signbit),
        ("nan_to_num", "nan_to_num.out", _nan_to_num),
    ):
        library.impl(name, fn)
        library.impl(overload, _out(fn, "out"))
