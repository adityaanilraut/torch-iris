"""Distributions composed from the Iris generator and on-device transforms."""

import math

import torch

from ._model_ops import _out, _write_out


def _probability(p):
    if not ((p >= 0) & (p <= 1)).all().item():
        raise RuntimeError("Bernoulli expects probabilities in [0, 1]")


def _bernoulli(input, p, *, generator=None):
    if p.device != input.device or not p.is_floating_point():
        raise RuntimeError("Bernoulli expects floating probabilities on the destination device")
    p = p.expand(input.shape).float()
    _probability(p)
    samples = torch.rand(input.shape, dtype=torch.float32, device=input.device, generator=generator)
    return input.copy_(samples < p)


def _random_uniform(input, generator):
    return torch.rand(input.shape, dtype=torch.float32, device=input.device, generator=generator)


def _exponential(input, lambd=1., *, generator=None):
    if not input.is_floating_point() or not math.isfinite(lambd) or lambd <= 0:
        raise RuntimeError("exponential_ expects floating input and a positive finite rate")
    u = _random_uniform(input, generator)
    return input.copy_(-torch.log1p(-u.clamp_min(torch.finfo(torch.float32).eps / 2)) / lambd)


def _log_normal(input, mean=1., std=2., *, generator=None):
    if not input.is_floating_point() or not math.isfinite(std) or std <= 0:
        raise RuntimeError("log_normal_ expects floating input and positive finite std")
    sample = torch.empty_like(input, dtype=torch.float32).normal_(mean, std, generator=generator)
    return input.copy_(sample.exp())


def _cauchy(input, median=0., sigma=1., *, generator=None):
    if not input.is_floating_point() or not math.isfinite(sigma) or sigma <= 0:
        raise RuntimeError("cauchy_ expects floating input and positive finite sigma")
    u = _random_uniform(input, generator).clamp_min(torch.finfo(torch.float32).eps / 2)
    return input.copy_((math.pi * (u - 0.5)).tan() * sigma + median)


def _geometric(input, p, *, generator=None):
    if not 0 < p < 1:
        raise RuntimeError("geometric_ expects 0 < p < 1")
    u = _random_uniform(input, generator)
    return input.copy_((torch.log1p(-u) / math.log1p(-p)).floor() + 1)


def _normal(mean, std, *, generator=None):
    ref = mean if isinstance(mean, torch.Tensor) else std
    if not ref.is_floating_point():
        raise RuntimeError("Iris normal expects a floating tensor")
    if isinstance(std, torch.Tensor):
        if std.device != ref.device or not std.is_floating_point() or not (std >= 0).all().item():
            raise RuntimeError("normal expects nonnegative floating std on the mean device")
    elif not std >= 0:
        raise RuntimeError("normal expects nonnegative std")
    shape = torch.broadcast_shapes(*(v.shape for v in (mean, std) if isinstance(v, torch.Tensor)))
    sample = torch.empty(shape, dtype=torch.float32, device=ref.device).normal_(generator=generator)
    mean = mean.float() if isinstance(mean, torch.Tensor) else mean
    std = std.float() if isinstance(std, torch.Tensor) else std
    return (sample * std + mean).to(ref.dtype)


def _randperm(n, *, generator=None, out):
    if n < 0 or out.dtype == torch.bool or out.is_complex():
        raise RuntimeError("randperm expects nonnegative n and a real numeric output")
    exact_limit = {torch.float16: 2048, torch.bfloat16: 256, torch.float32: 2**24}
    maximum = exact_limit[out.dtype] if out.is_floating_point() else torch.iinfo(out.dtype).max
    if n - 1 > maximum:
        raise RuntimeError("randperm cannot represent the permutation in the output dtype")
    high = out.new_empty((n,), dtype=torch.long)
    low = torch.empty_like(high)
    while True:
        # Separate Philox lanes provide independent 31-bit parts, with no modulus bias.
        high.random_(1 << 31, generator=generator)
        low.random_(1 << 31, generator=generator)
        seq, order = (high * (1 << 31) + low).sort()
        if n < 2 or not (seq[1:] == seq[:-1]).any().item():
            break
    return _write_out(out, order.to(out.dtype))


def _poisson(input, generator=None):
    if not input.is_floating_point():
        raise RuntimeError("Poisson expects floating rates")
    rate = input.float()
    if not ((rate >= 0) & (rate <= 2**24)).all().item():
        raise RuntimeError("Iris Poisson requires finite rates in [0, 2**24]")
    result = torch.zeros_like(rate)
    # The waiting time to the k-th arrival is the negative log of the
    # product of k independent uniforms.
    active = (rate > 0) & (rate < 10)
    product = torch.ones_like(rate)
    stop = (-rate).exp()
    while active.any().item():
        product = product * _random_uniform(rate, generator)
        active = active & (product > stop)
        result = result + active.float()

    active = rate >= 10
    if not active.any().item():
        return result.to(input.dtype)
    # Hörmann's PTRS rejection method for larger means.
    lam = rate.clamp_min(10)
    b = 0.931 + 2.53 * lam.sqrt()
    a = -0.059 + 0.02483 * b
    alpha = 1.1239 + 1.1328 / (b - 3.4)
    squeeze = 0.9277 - 3.6224 / (b - 2)
    log_factorial = rate.new_tensor([math.lgamma(k + 1) for k in range(10)])
    while active.any().item():
        u = _random_uniform(rate, generator) - 0.5
        v = _random_uniform(rate, generator)
        us = (0.5 - u.abs()).clamp_min(torch.finfo(torch.float32).eps / 2)
        k = ((2 * a / us + b) * u + lam + 0.43).floor()
        small = k.clamp(0, 9).long()
        small_log_mass = k * lam.log() - lam - log_factorial[small]
        positive = k.clamp_min(1)
        d = (positive - lam) / lam
        series = d.square() * (0.5 + d * (-1 / 6 + d * (1 / 12 + d * (-1 / 20 + d / 30))))
        deviance = lam * torch.where(d.abs() < 0.025, series, (1 + d) * torch.log1p(d) - d)
        inv = positive.reciprocal()
        log_mass = -deviance - 0.5 * (math.log(2 * math.pi) + positive.log()) - inv * (1 / 12 - inv.square() / 360)
        log_mass = torch.where(k < 10, small_log_mass, log_mass)
        accepted = (k >= 0) & (((us >= 0.07) & (v <= squeeze)) | (
            ((us >= 0.013) | (v <= us)) & ((v * alpha / (a / us.square() + b)).log() <= log_mass)
        ))
        result = torch.where(active & accepted, k, result)
        active = active & ~accepted
    return result.to(input.dtype)


def register(library):
    library.impl("bernoulli_.Tensor", _bernoulli)
    library.impl("exponential_", _exponential)
    library.impl("log_normal_", _log_normal)
    library.impl("cauchy_", _cauchy)
    library.impl("geometric_", _geometric)
    library.impl("randperm.generator_out", _randperm)
    library.impl("poisson", _poisson)
    library.impl("poisson.out", _out(_poisson, "out"))
    for overload in ("Tensor_Tensor", "Tensor_float", "float_Tensor"):
        library.impl(f"normal.{overload}", _normal)
        library.impl(f"normal.{overload}_out", _out(_normal, "out"))
