"""Pointwise, distance, and metadata implementations for the Iris dispatcher."""

import torch
from torch import _refs
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND, elementwise_dtypes
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper

from ._model_ops import _decompose, _out, _register_pairs, _write_out


@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"), type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
)
def _fextreme(a, b, extreme):
    result = extreme(a, b)
    return torch.where(b.isnan(), a, torch.where(a.isnan(), b, result)) if a.is_floating_point() else result


@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"), type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)
def _hypot(a, b):
    if not a.is_floating_point():
        raise RuntimeError("hypot expects floating tensors")
    a, b = a.abs(), b.abs()
    scale = torch.maximum(a, b)
    # This formulation avoids overflow on large inputs and underflow on small ones.
    ratio = torch.minimum(a, b) / torch.where(scale == 0, 1, scale)
    return torch.where(a.isinf() | b.isinf(), torch.inf, scale * (1 + ratio.square()).sqrt())


@elementwise_type_promotion_wrapper(
    type_promoting_args=("input",), type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)
def _logit(input, eps=None):
    x = input if eps is None or eps < 0 else input.clamp(min=eps, max=1 - eps)
    return (x / (1 - x)).log()


def _shift(left, right, *, arithmetic):
    _, dtype = elementwise_dtypes(left, right, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH)
    if dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise RuntimeError("Iris bit shifts require integral operands")
    bits = torch.iinfo(dtype).bits
    if not isinstance(left, torch.Tensor):
        left = right.new_full((), left, dtype=dtype)
    if not isinstance(right, torch.Tensor):
        right = left.new_full((), right, dtype=dtype)
    right = right.to(dtype).long()
    value = left.to(dtype).long()
    if arithmetic:
        bound = bits - int(dtype != torch.uint8)
        amount = torch.where((right < 0) | (right >= bound), bound, right)
    else:
        amount = right
    # Split shifts into powers of two. The largest divisor here fits in int64.
    for bit in (1, 2, 4, 8, 16, 32):
        if bit >= bits:
            break
        factor = 1 << bit
        shifted = value // factor if arithmetic else value * factor
        value = torch.where((amount & bit) != 0, shifted, value)
    if not arithmetic:
        value = torch.where((right < 0) | (right >= bits), 0, value)
    elif dtype == torch.uint8:
        value = torch.where(amount == bits, 0, value)
    return value.to(dtype)


def _logspace(start, end, steps, base=10., *, out):
    if out.is_complex():
        raise RuntimeError("Iris logspace supports real exponents and output")
    exponent = torch.linspace(start, end, steps, device=out.device, dtype=torch.float32)
    result = torch.pow(base, exponent)
    # OpenCL pow can lose an ulp even for small exact integer powers.
    bounded = exponent.abs() <= 32
    integer = torch.where(bounded, exponent, 0).long()
    magnitude = integer.abs()
    power = torch.ones_like(exponent)
    factor = exponent.new_full((), base)
    for bit in (1, 2, 4, 8, 16, 32):
        power = torch.where((magnitude & bit) != 0, power * factor, power)
        if bit != 32:
            factor = factor * factor
    exact_exponent = bounded & (exponent == integer)
    result = torch.where(exact_exponent, torch.where(integer < 0, power.reciprocal(), power), result)
    return _write_out(out, result.to(out.dtype))


def _vdot(left, right):
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape or left.dtype != right.dtype:
        raise RuntimeError("vdot expects equal one-dimensional operands of the same dtype")
    if left.is_complex():
        return (left.conj() * right).sum(dtype=left.dtype)
    return torch.dot(left, right)


def _cdist(x1, x2, p, compute_mode=None):
    if x1.ndim < 2 or x2.ndim < 2 or x1.size(-1) != x2.size(-1) or p < 0:
        raise RuntimeError("cdist expects matrices with matching feature dimensions and nonnegative p")
    return torch.linalg.vector_norm(x1.unsqueeze(-2) - x2.unsqueeze(-3), p, dim=-1)


def _distance_gradient(diff, p, distance):
    if p == 0:
        return torch.zeros_like(diff)
    if p == 1:
        return diff.sign()
    if p == torch.inf:
        return torch.where(diff.abs() == distance, diff.sign(), 0)
    safe = torch.where(distance > 0, distance, 1)
    magnitude = (diff.abs() / safe).pow(p - 1)
    return torch.where((diff != 0) & (distance > 0), diff.sign() * magnitude, 0)


def _cdist_backward(grad, x1, x2, p, cdist):
    diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
    return (grad.unsqueeze(-1) * _distance_gradient(diff, p, cdist.unsqueeze(-1))).sum(-2).sum_to_size(x1.shape)


def _pair_indices(input):
    rows = input.size(0)
    # At small sizes the mask scan costs less than the extra indexing launches.
    if rows < 32:
        positions = torch.arange(rows, device=input.device)
        return (positions[:, None] < positions).nonzero().unbind(1)

    def row_start(row):
        return (row * (2 * rows - 1 - row)) // 2

    # The first row starts at zero. Search the remaining starts for the row number.
    starts = row_start(torch.arange(1, rows, device=input.device))
    offset = torch.arange(rows * (rows - 1) // 2, device=input.device)
    i = torch.searchsorted(starts, offset, right=True)
    return i, offset - row_start(i) + i + 1


def _pdist(input, p=2.):
    if input.ndim != 2 or not input.is_floating_point() or p < 0:
        raise RuntimeError("pdist expects a floating matrix and nonnegative p")
    i, j = _pair_indices(input)
    return torch.linalg.vector_norm(input.index_select(0, i) - input.index_select(0, j), p, dim=-1)


def _pdist_backward(grad, input, p, pdist):
    i, j = _pair_indices(input)
    diff = input.index_select(0, i) - input.index_select(0, j)
    values = grad[:, None] * _distance_gradient(diff, p, pdist[:, None])
    result = torch.zeros_like(input, memory_format=torch.contiguous_format)
    return result.scatter_add_(0, i[:, None].expand(values.shape), values).scatter_add_(
        0, j[:, None].expand(values.shape), -values)


def _set_tensor(input, source):
    if input.dtype != source.dtype or input.device != source.device:
        raise RuntimeError("set_ expects the same dtype and device")
    return input.set_(source.untyped_storage(), source.storage_offset(), source.size(), source.stride())


def register(library):
    pair = _register_pairs(library)

    pair("fmin", lambda a, b: _fextreme(a, b, torch.minimum))
    pair("fmax", lambda a, b: _fextreme(a, b, torch.maximum))
    for name, fn in (
        ("hypot", _hypot), ("copysign", _refs.copysign), ("logaddexp", _refs.logaddexp),
        ("sinc", _refs.sinc), ("logit", _logit), ("vdot", _vdot), ("_cdist_forward", _cdist),
        ("_cdist_backward", _cdist_backward), ("_pdist_forward", _pdist), ("_pdist_backward", _pdist_backward),
    ):
        pair(name, fn)
    pair("logit_backward", _decompose("logit_backward"), "grad_input", overload="grad_input")
    for name in ("Tensor", "Scalar_Self", "Scalar_Other"):
        pair(f"xlogy.{name}", _refs.xlogy, overload=f"Out{name}")
    for name, arithmetic in (("left", False), ("right", True)):
        def shift(a, b, arithmetic=arithmetic):
            return _shift(a, b, arithmetic=arithmetic)

        for suffix in ("Tensor", "Tensor_Scalar", "Scalar_Tensor"):
            pair(f"bitwise_{name}_shift.{suffix}", shift, overload=f"{suffix}_out")
        for suffix in ("Tensor", "Scalar"):
            pair(f"__{name[0]}shift__.{suffix}", shift, overload=f"{suffix}_out")
    for overload in ("out", "dtype_out", "Scalar_out", "ScalarOpt_dtype_out"):
        library.impl(f"norm.{overload}", _out(_refs.norm, "out"))
    library.impl("logspace.out", _logspace)
    library.impl("set_.source_Tensor", _set_tensor)
    library.impl("_efficientzerotensor", lambda size, **kwargs: torch.zeros(size, **kwargs))
