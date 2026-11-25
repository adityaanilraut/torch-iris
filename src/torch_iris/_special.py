"""Real special functions implemented with FP32 OpenCL arithmetic."""

import math

import torch
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper, out_wrapper

from ._model_ops import _out


_EULER = 0.5772156649015329
_ZETA = (1.64493406685, 1.20205690316, 1.08232323371, 1.03692775514,
         1.01734306198, 1.00834927738, 1.00407735620, 1.00200839283,
         1.00099457513, 1.00049418860, 1.00024608655, 1.00012271335,
         1.00006124814, 1.00003058824, 1.00001528226, 1.00000763720,
         1.00000381729, 1.00000190821, 1.00000095396, 1.00000047693,
         1.00000023845, 1.00000011922, 1.00000005961)


def _lgamma_positive(input):
    large = input > 8
    x = torch.where(large, 1., input)
    tiny = x < 0.5
    adjustment = torch.where(tiny, -x.log(), 0.)
    x = torch.where(tiny, x + 1, x)
    for _ in range(7):
        reduce = x > 1.5
        x = torch.where(reduce, x - 1, x)
        adjustment = adjustment + torch.where(reduce, x.log(), 0.)
    # log Γ(1 + z) around z=0. Recurrence keeps |z| <= 1/2.
    z = x - 1
    polynomial = z.new_zeros(())
    for degree, coefficient in reversed(tuple(enumerate(_ZETA, 2))):
        polynomial = polynomial * z + (-1)**degree * coefficient / degree
    series = (polynomial * z - _EULER) * z + adjustment
    inv = input.reciprocal()
    inv2 = inv.square()
    correction = inv * (1 / 12 + inv2 * (-1 / 360 + inv2 * (1 / 1260 - inv2 / 1680)))
    stirling = (input - 0.5) * input.log() - input + 0.9189385332046727 + correction
    return torch.where(large, stirling, series)


def _lgamma(input):
    x = input.float()
    negative = x < 0
    direct = _lgamma_positive(torch.where(negative, 1 - x, x))
    sine = (math.pi * (x - x.round())).sin()
    reflection = math.log(math.pi) - sine.abs().log() - direct
    result = torch.where(negative, reflection, direct)
    poles = (x <= 0) & (x == x.floor())
    return torch.where(poles | x.isinf(), torch.inf, result).to(input.dtype)


def _digamma(input):
    x = input.float()
    negative = x < 0
    y = torch.where(negative, 1 - x, x)
    correction = torch.zeros_like(y)
    for _ in range(8):
        step = y < 8
        correction = correction - torch.where(step, y.reciprocal(), 0.)
        y = torch.where(step, y + 1, y)
    inv = y.reciprocal()
    z = inv.square()
    direct = y.log() - inv * 0.5 + z * (-1 / 12 + z * (1 / 120 + z * (-1 / 252 + z / 240)))
    angle = math.pi * (x - x.round())
    result = direct + correction - torch.where(negative, math.pi * angle.cos() / angle.sin(), 0.)
    result = torch.where(negative & (x == x.floor()), torch.nan, result)
    return torch.where(x == 0, -x.reciprocal(), result).to(input.dtype)


@elementwise_type_promotion_wrapper(
    type_promoting_args=("input",), type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)
def _polygamma(n, input):
    if input.dtype != torch.float32:
        raise RuntimeError("Iris polygamma requires real inputs with float32 computation")
    if n == 0:
        return _digamma(input)
    if n != 1:
        raise RuntimeError("Iris polygamma supports orders zero and one")
    x = input.float()
    negative = x < 0
    y = torch.where(negative, 1 - x, x)
    correction = torch.zeros_like(y)
    for _ in range(8):
        step = y < 8
        correction = correction + torch.where(step, y.reciprocal().square(), 0.)
        y = torch.where(step, y + 1, y)
    inv = y.reciprocal()
    z = inv.square()
    direct = correction + inv + z * (0.5 + inv * (1 / 6 + z * (-1 / 30 + z / 42)))
    reflection = (math.pi / (math.pi * (x - x.round())).sin()).square() - direct
    result = torch.where(negative, reflection, direct)
    return torch.where(x == 0, torch.inf, result)


_write_polygamma_out = out_wrapper()(_polygamma)


def _polygamma_out(n, input, *, out):
    dtype = torch.get_default_dtype() if not input.is_floating_point() and not input.is_complex() else input.dtype
    if out.device != input.device:
        raise RuntimeError("Iris polygamma requires input and out on the same device")
    if not torch.can_cast(dtype, out.dtype):
        raise RuntimeError(f"Iris polygamma cannot cast result dtype {dtype} to {out.dtype}")
    return _write_polygamma_out(n, input, out=out)


def _erfinv(input):
    x = input.float()
    magnitude = x.abs()
    # A closed-form starting point followed by Newton refinement. erfc
    # retains significant digits in the tails where erf rounds toward one.
    log = torch.log1p(-magnitude.square())
    a = 0.147
    t = 2 / (math.pi * a) + log * 0.5
    initial = ((t.square() - log / a).sqrt() - t).clamp_min(0).sqrt()
    y = torch.where(magnitude < 0.5, magnitude * (math.sqrt(math.pi) / 2), initial)
    for _ in range(4):
        error = torch.where(magnitude < 0.75, y.erf() - magnitude, (1 - magnitude) - y.erfc())
        y = y - error * (math.sqrt(math.pi) / 2) * y.square().exp()
    result = torch.copysign(y, x)
    result = torch.where(magnitude == 1, torch.copysign(torch.full_like(x, torch.inf), x), result)
    return torch.where(magnitude > 1, torch.nan, result).to(input.dtype)


def _bessel(input, order):
    x = input.float()
    magnitude = x.abs()
    small = magnitude <= 20
    half = torch.where(small, magnitude * 0.5, 0.)
    square = half.square()
    term = torch.ones_like(half) if order == 0 else half
    series = term
    for k in range(1, 28):
        term = term * square / (k * (k + order))
        series = series + term
    z = torch.where(small, 1., magnitude).reciprocal() / 8
    polynomial = torch.ones_like(z)
    coefficient = 1.
    for k in range(1, 7):
        coefficient *= ((2 * k - 1)**2 - 4 * order**2) / k
        polynomial = polynomial + coefficient * z.pow(k)
    exponential = (magnitude - 0.5 * (magnitude.log() + math.log(2 * math.pi))).exp()
    result = torch.where(small, series, exponential * polynomial)
    result = torch.where(magnitude.isinf(), torch.inf, result)
    return (result if order == 0 else torch.copysign(result, x)).to(input.dtype)


def register(library):
    for name, fn in (
        ("lgamma", _lgamma), ("digamma", _digamma), ("erfinv", _erfinv),
        ("i0", lambda input: _bessel(input, 0)), ("special_i1", lambda input: _bessel(input, 1)),
    ):
        promoted = elementwise_type_promotion_wrapper(
            type_promoting_args=("input",), type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )(fn)
        library.impl(name, promoted)
        library.impl(f"{name}.out", _out(promoted, "out"))
    library.impl("polygamma", _polygamma)
    library.impl("polygamma.out", _polygamma_out)
