"""Device-resident scans, order statistics, and histograms."""

import math

import torch

from ._model_ops import _out, _register_pairs, _write_out


def _order_dtype(input):
    if input.dtype in (torch.int8, torch.int16):
        return torch.long
    if input.dtype == torch.bfloat16:
        return torch.float32
    return input.dtype


def _scan(input, dim, combine):
    if input.ndim == 0:
        if dim not in (0, -1):
            raise IndexError("scalar scan dimension out of range")
        return input.clone()
    value = input.movedim(dim, -1)
    if input.numel() == 0:
        return input.clone()
    stride = 1
    while stride < value.size(-1):
        value = torch.cat((value[..., :stride], combine(value[..., :-stride], value[..., stride:])), -1)
        stride *= 2
    return value.movedim(-1, dim).contiguous()


def _cumprod(input, dim, *, dtype=None):
    dtype = dtype or (input.dtype if input.is_floating_point() or input.is_complex() else torch.long)
    compute = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    return _scan(input.to(compute), dim, torch.mul).to(dtype)


def _logcumsumexp(input, dim):
    if input.is_complex() or not input.is_floating_point():
        raise RuntimeError("Iris logcumsumexp requires real floating input")
    return _scan(input.float(), dim, torch.logaddexp).to(input.dtype)


def _cumprod_out(input, dim, *, dtype=None, out):
    return _write_out(out, _cumprod(input, dim, dtype=dtype or out.dtype))


def _cumextreme(largest):
    def run(input, values, indices, dim):
        if input.ndim == 0 or input.numel() == 0:
            values.copy_(input)
            indices.zero_()
            return
        value = input.movedim(dim, -1)
        width = value.size(-1)
        position = torch.arange(width, device=input.device).expand(value.shape)
        stride = 1
        while stride < width:
            a, b = value[..., :-stride], value[..., stride:]
            take_right = b >= a if largest else b <= a
            if value.is_floating_point():
                take_right = take_right | b.isnan()
            position = torch.cat((position[..., :stride],
                                  torch.where(take_right, position[..., stride:], position[..., :-stride])), -1)
            value = torch.cat((value[..., :stride], torch.where(take_right, b, a)), -1)
            stride *= 2
        values.copy_(value.movedim(-1, dim))
        indices.copy_(position.movedim(-1, dim))

    return run


def _kth(input, k, dim=-1, keepdim=False):
    width = input.size(dim) if input.ndim else 1
    if not 1 <= k <= width:
        raise RuntimeError("kthvalue: k out of range")
    if input.ndim == 0:
        return input.clone(), input.new_zeros((), dtype=torch.long)
    values, indices = input.to(_order_dtype(input)).sort(dim=dim)
    result = (values.narrow(dim, k - 1, 1).to(input.dtype), indices.narrow(dim, k - 1, 1))
    return tuple((value if keepdim else value.squeeze(dim)).contiguous() for value in result)


def _median_dim(ignore_nan):
    def run(input, dim, keepdim=False):
        if input.ndim == 0:
            return input.clone(), input.new_zeros((), dtype=torch.long)
        if not input.size(dim):
            raise IndexError("median expects a nonempty reduction dimension")
        values, indices = input.to(_order_dtype(input)).sort(dim=dim, stable=True)
        if input.is_floating_point():
            count = (~values.isnan()).sum(dim, keepdim=True)
            if ignore_nan:
                position = ((count - 1) // 2).clamp(min=0)
            else:
                position = torch.where(count < input.size(dim), count, (input.size(dim) - 1) // 2)
        else:
            position = indices.narrow(dim, 0, 1).new_full(
                indices.narrow(dim, 0, 1).shape, (input.size(dim) - 1) // 2)
        result = (values.gather(dim, position).to(input.dtype), indices.gather(dim, position))
        return result if keepdim else tuple(value.squeeze(dim) for value in result)

    return run


def _median(ignore_nan):
    select = _median_dim(ignore_nan)

    def run(input):
        if not input.numel():
            if not input.is_floating_point():
                value = torch.iinfo(input.dtype).min if input.dtype in (torch.int32, torch.int64) else 0
                return input.new_full((), value)
            return input.new_full((), torch.nan)
        return select(input.flatten(), 0)[0]

    return run


def _mode(input, dim=-1, keepdim=False):
    if input.ndim == 0:
        return input.clone(), input.new_zeros((), dtype=torch.long)
    values, indices = input.to(_order_dtype(input)).movedim(dim, -1).sort(stable=True)
    width = values.size(-1)
    if not width:
        raise IndexError("mode expects a nonempty reduction dimension")
    positions = torch.arange(width, device=input.device)
    equal = values[..., 1:] == values[..., :-1]
    start = torch.cat((torch.ones_like(values[..., :1], dtype=torch.bool), ~equal), -1)
    end = torch.cat((~equal, torch.ones_like(values[..., :1], dtype=torch.bool)), -1)
    begin = torch.where(start, positions, 0).cummax(-1).values
    chosen = torch.where(end, positions - begin + 1, 0).argmax(-1, keepdim=True)
    result = (values.gather(-1, chosen).movedim(-1, dim).to(input.dtype),
              indices.gather(-1, chosen).movedim(-1, dim))
    return result if keepdim else tuple(value.squeeze(dim) for value in result)


def _unique(input, return_inverse, return_counts, dim, sort):
    flat = input.flatten() if dim is None else input.movedim(dim, 0)
    width = flat.size(0)
    empty = input.new_empty((0,), dtype=torch.long)
    if not width:
        inverse = empty.reshape(input.shape) if dim is None else empty
        return flat.clone() if dim is None else input.clone(), inverse if return_inverse else empty, empty
    flat = flat.to(_order_dtype(input))
    seq, order = flat.sort(stable=True) if sort else (flat, None)
    different = seq[1:] != seq[:-1]
    if seq.ndim > 1:
        different = different.flatten(1).any(-1)
    first = torch.cat((torch.ones_like(different.new_empty(1)), different))
    starts = first.nonzero().flatten()
    unique = seq.index_select(0, starts)
    if dim is not None:
        unique = unique.movedim(0, dim)
    inverse = empty
    if return_inverse:
        groups = first.long().cumsum(0) - 1
        inverse = torch.empty_like(groups).scatter_(0, order, groups) if sort else groups
        if dim is None:
            inverse = inverse.reshape(input.shape)
    counts = torch.cat((starts[1:], starts.new_full((1,), width))) - starts if return_counts else empty
    return unique.to(input.dtype), inverse, counts


def _unique_sorted(input, sorted=True, return_inverse=False, return_counts=False):
    # ATen CPU sorts flat unique even when sorted=False.
    return _unique(input, return_inverse, return_counts, None, True)


def _unique_consecutive(input, return_inverse=False, return_counts=False, dim=None):
    return _unique(input, return_inverse, return_counts, dim, False)


def _bincount(input, weights=None, minlength=0):
    if input.ndim != 1 or input.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise RuntimeError("bincount expects a one-dimensional nonnegative integral tensor")
    if minlength < 0:
        raise RuntimeError("bincount expects a nonnegative minlength")
    if weights is not None and (weights.shape != input.shape or weights.device != input.device):
        raise RuntimeError("bincount weights must match the input shape and device")
    width = minlength
    if input.numel():
        low, high = input.long().aminmax()
        if low.item() < 0:
            raise RuntimeError("bincount expects nonnegative values")
        width = max(width, high.item() + 1)
    if weights is None or not input.numel():
        weights = torch.ones_like(input, dtype=torch.long)
    elif weights.dtype != torch.float32:
        raise RuntimeError("Iris weighted bincount requires float32 weights")
    return weights.new_zeros((width,)).scatter_add_(0, input.long(), weights)


def _histc(input, bins=100, min=0, max=0):
    if not input.is_floating_point() or bins <= 0:
        raise RuntimeError("histc expects floating input and positive bins")
    values = input.float().flatten()
    if min == max and input.numel():
        low, high = values.aminmax()
        min, max = low.item(), high.item()
    if min == max:
        min, max = min - 1, max + 1
    if not math.isfinite(min) or not math.isfinite(max) or min >= max:
        raise RuntimeError("histc expects a finite, increasing range")
    inside = (values >= min) & (values <= max)
    bucket = torch.where(inside, (values - min) * bins / (max - min), 0).long().clamp(0, bins - 1)
    return values.new_zeros((bins,)).scatter_add_(0, bucket, inside.float()).to(input.dtype)


def register(library):
    pair = _register_pairs(library)

    library.impl("cumprod", _cumprod)
    library.impl("cumprod.out", _cumprod_out)
    library.impl("_cummax_helper", _cumextreme(True))
    library.impl("_cummin_helper", _cumextreme(False))
    pair("_logcumsumexp", _logcumsumexp)
    pair("kthvalue", _kth, "values", "indices", overload="values")
    pair("mode", _mode, "values", "indices", overload="values")
    for name, ignore_nan in (("median", False), ("nanmedian", True)):
        pair(name, _median(ignore_nan))
        library.impl(f"{name}.dim", _median_dim(ignore_nan))
        library.impl(f"{name}.dim_values", _out(_median_dim(ignore_nan), "values", "indices"))
    pair("_unique2", _unique_sorted, "out0", "out1", "out2")
    pair("unique_consecutive", _unique_consecutive, "out0", "out1", "out2")
    pair("bincount", _bincount)
    pair("histc", _histc)
    library.impl("sum.IntList_out", lambda t, dim, keepdim=False, *, dtype=None, out:
                 _write_out(out, torch.sum(t, dim, keepdim, dtype=dtype or out.dtype)))
    library.impl("sum.out", lambda t, *, dtype=None, out:
                 _write_out(out, torch.sum(t, dtype=dtype or out.dtype)))
    autograd = torch.library.Library("aten", "IMPL", "AutogradPrivateUse1")
    autograd.impl("cumprod", _cumprod)
    return autograd
