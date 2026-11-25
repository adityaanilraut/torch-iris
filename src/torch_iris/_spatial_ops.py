"""Separable Iris compositions for spatial and model operations."""

import torch
from torch._refs.nn import functional as refs

from ._model_ops import _decompose, _pool_windows, _register_pairs, _scatter_axis


def _copy_inplace(fn):
    def run(input, *args, **kwargs):
        return input.copy_(fn(input, *args, **kwargs))

    return run


def _shrink_backward(grad, input, lambd):
    return torch.where((input >= -lambd) & (input <= lambd), 0, grad)


def _rrelu(input, noise, lower=0.125, upper=1 / 3, training=False, generator=None):
    if not training:
        return torch.nn.functional.leaky_relu(input, (lower + upper) / 2)
    if lower > upper:
        raise RuntimeError("RReLU expects lower <= upper")
    slope = torch.empty_like(input).uniform_(lower, upper, generator=generator)
    noise.copy_(torch.where(input <= 0, slope, 1))
    return input * noise


def _threshold(input, threshold, value):
    return torch.where(input <= threshold, input.new_full((), value), input)


def _embedding_renorm(input, indices, max_norm, norm_type):
    keys = torch.unique(indices.flatten().long())
    rows = input.index_select(0, keys)
    norm = torch.linalg.vector_norm(rows.float(), norm_type, dim=1, keepdim=True)
    scaled = rows * torch.where(norm > max_norm, max_norm / (norm + 1e-7), 1).to(rows.dtype)
    return input.index_copy_(0, keys, scaled)


def _weight_norm(v, g, dim=0):
    dim %= v.ndim
    axes = [axis for axis in range(v.ndim) if axis != dim]
    norm = torch.linalg.vector_norm(v.float(), dim=axes, keepdim=True) if axes else v.float().abs()
    return (v.float() * (g.float() / norm)).to(v.dtype), norm


def _weight_norm_backward(grad, v, g, norms, dim):
    dim %= v.ndim
    axes = [axis for axis in range(v.ndim) if axis != dim]
    product = grad.float() * v.float()
    dot = product.sum(axes, keepdim=True) if axes else product
    normalized = v.float() / norms
    grad_g = dot / norms
    grad_v = (grad.float() - normalized * grad_g) * (g.float() / norms)
    return grad_v.to(v.dtype), grad_g.to(g.dtype)


def _pad3d(reflect, backward=False):
    def run(input, *args):
        original, padding = args if backward else (input, args[0])
        if original.ndim not in (4, 5) or len(padding) != 6:
            raise RuntimeError("3D padding expects four or five dimensions and six padding values")
        value = input.float() if backward and input.dtype != torch.float32 else input
        for dim, left, right in zip((-1, -2, -3), padding[::2], padding[1::2]):
            width = original.size(dim)
            if width == 0 or width + left + right < 0 or (reflect and max(left, right) >= width):
                raise RuntimeError("3D padding is outside the input dimension")
            index = torch.arange(-left, width + right, device=input.device)
            if reflect:
                index = width - 1 - (width - 1 - index.abs()).abs()
            else:
                index = index.clamp(0, width - 1)
            if backward:
                value = _scatter_axis(value.movedim(dim, -1), index, width).movedim(-1, dim)
            else:
                value = value.index_select(dim, index)
        return value.to(input.dtype).contiguous()

    return run


def _interpolate(mode, dimensions, backward=False):
    def run(input, output_size, *args, **kwargs):
        original = args[0] if backward else input.shape
        args = args[1:] if backward else args
        linear = mode == "linear"
        align = (args[0] if args else kwargs.get("align_corners", False)) if linear else False
        args = args[1:] if linear else args
        keys = ("scales",) if backward and dimensions == 1 else ("scales_d", "scales_h", "scales_w")[-dimensions:]
        if len(original) != 2 + dimensions or len(output_size) != dimensions or min(*original[1:], *output_size) <= 0:
            raise RuntimeError("Iris interpolation expects positive channel and spatial sizes")
        value = input.float() if linear or backward else input
        for i, (size, out_size, key) in enumerate(zip(original[-dimensions:], output_size, keys)):
            dim = i - dimensions
            scale = args[i] if i < len(args) else kwargs.get(key)
            ratio = ((size - 1) / max(1, out_size - 1)) if align else (1 / scale if scale and scale > 0 else size / out_size)
            position = torch.arange(out_size, device=input.device, dtype=torch.float32)
            if linear:
                position = (position * ratio if align else (position + 0.5) * ratio - 0.5).clamp_min(0)
            elif mode == "exact":
                position = (position + 0.5) * ratio
            elif out_size not in (size, 2 * size):
                position = position * ratio
            elif out_size == 2 * size:
                position = position * 0.5
            base = position.floor().long().clamp(max=size - 1)
            value = value.movedim(dim, -1)
            if linear:
                next_index = (base + 1).clamp(max=size - 1)
                blend = position - base.float()
                if backward:
                    value = (_scatter_axis(value * (1 - blend), base, size)
                             + _scatter_axis(value * blend, next_index, size))
                else:
                    value = value.index_select(-1, base) * (1 - blend) + value.index_select(-1, next_index) * blend
            elif backward:
                value = _scatter_axis(value, base, size)
            else:
                value = value.index_select(-1, base)
            value = value.movedim(-1, dim)
        return value.to(input.dtype).contiguous()

    return run


def _adaptive_avg(input, output_size, backward_shape=None):
    shape = input.shape if backward_shape is None else backward_shape
    if len(shape) not in (4, 5) or len(output_size) != 3 or min(shape[-3:]) <= 0 or min(output_size) < 0:
        raise RuntimeError("adaptive_avg_pool3d expects four or five dimensions and valid spatial sizes")
    value = input.float()
    for dim, size, width in zip((-1, -2, -3), reversed(shape[-3:]), reversed(output_size)):
        value = value.movedim(dim, -1)
        if width == 0:
            value = value[..., :0] if backward_shape is None else value.new_zeros((*value.shape[:-1], size))
        else:
            index, start, end = _pool_windows(size, width, input.device)
            valid = index < end[:, None]
            index = index.clamp(max=size - 1)
            length = end - start
            if backward_shape is None:
                selected = value.index_select(-1, index.flatten()).unflatten(-1, index.shape)
                value = torch.where(valid, selected, 0).sum(-1) / length
            else:
                expanded = torch.where(valid, (value / length).unsqueeze(-1), 0)
                value = _scatter_axis(expanded.flatten(-2), index.flatten(), size)
        value = value.movedim(-1, dim)
    return value.to(input.dtype).contiguous()


def _max_unpool(input, indices, output_size):
    if input.shape != indices.shape or indices.dtype != torch.long or input.ndim not in (3, 4):
        raise RuntimeError("max_unpool2d expects a 3D/4D tensor and matching int64 indices")
    if len(output_size) != 2 or min(output_size) < 0:
        raise RuntimeError("max_unpool2d expects a two-dimensional output size")
    shape = (*input.shape[:-2], *output_size)
    return input.new_zeros(shape).flatten(-2).scatter_(-1, indices.flatten(-2), input.flatten(-2)).view(shape)


def _multi_margin_backward(grad, input, target, p=1, margin=1, weight=None, reduction=1):
    matrix = torch.atleast_2d(input)
    keys = torch.atleast_1d(target).unsqueeze(1)
    diff = margin - matrix.gather(1, keys) + matrix
    active = torch.arange(matrix.size(1), device=input.device) != keys
    contribution = torch.where(active & (diff > 0), 1. if p == 1 else 2 * diff, 0)
    if weight is not None:
        contribution = contribution * weight.index_select(0, keys.flatten()).unsqueeze(1)
    contribution = contribution.scatter_(1, keys, -contribution.sum(1, keepdim=True))
    factor = matrix.numel() if reduction == 1 else matrix.size(1)
    return (contribution * grad.reshape(-1, 1) / factor).reshape(input.shape)


def _multilabel_terms(input, target, is_target=None):
    if input.ndim > 2 or target.shape != input.shape or target.dtype != torch.long:
        raise RuntimeError("multilabel_margin_loss expects a vector or matrix and matching int64 labels")
    matrix = torch.atleast_2d(input)
    labels = torch.atleast_2d(target)
    width = matrix.size(1)
    if not width:
        raise RuntimeError("multilabel_margin_loss expects a nonempty class dimension")
    classes = torch.arange(width, device=input.device)
    stop = torch.where(labels < 0, classes, width).amin(dim=-1, keepdim=True)
    active = classes < stop
    indices = torch.where(active, labels, 0)
    difference = 1 - matrix.gather(1, indices).unsqueeze(-1) + matrix.unsqueeze(1)
    if is_target is None:
        present = (active.unsqueeze(-1) & (indices.unsqueeze(-1) == classes)).any(1)
    else:
        present = torch.atleast_2d(is_target) != 0
    mask = active.unsqueeze(-1) & ~present.unsqueeze(1)
    return difference, mask, indices, present


def _multilabel_forward(input, target, reduction):
    difference, mask, _, present = _multilabel_terms(input, target)
    loss = torch.where(mask, difference.clamp_min(0), 0).sum(dim=(1, 2)) / difference.size(-1)
    if reduction == 1:
        loss = loss.mean()
    elif reduction == 2:
        loss = loss.sum()
    elif input.ndim < 2:
        loss = loss.squeeze(0)
    return loss, present.to(input.dtype).reshape(target.shape)


def _multilabel_backward(grad, input, target, reduction, is_target):
    difference, mask, indices, _ = _multilabel_terms(input, target, is_target)
    increment = (mask & (difference > 0)).to(input.dtype)
    result = increment.sum(1).scatter_add_(1, indices, -increment.sum(2))
    divisor = input.numel() if reduction == 1 else difference.size(-1)
    return (result * grad.reshape(-1, 1) / divisor).reshape(input.shape)


def register(library):
    pair = _register_pairs(library)

    pair("channel_shuffle", refs.channel_shuffle)
    pair("glu", refs.glu)
    pair("glu_backward", _decompose("glu_backward"), "grad_input", overload="grad_input")
    for name, fn in (("hardshrink", refs.hardshrink), ("softshrink", refs.softshrink)):
        pair(name, fn)
        pair(f"{name}_backward", _shrink_backward, "grad_input", overload="grad_input")
    pair("threshold", _threshold)
    library.impl("threshold_", _copy_inplace(_threshold))
    pair("rrelu_with_noise", _rrelu)
    pair("rrelu_with_noise_backward", _decompose("rrelu_with_noise_backward"))
    library.impl("rrelu_with_noise_", _copy_inplace(_rrelu))
    library.impl("embedding_renorm_", _embedding_renorm)
    pair("_weight_norm_interface", _weight_norm, "out0", "out1")
    pair("_weight_norm_interface_backward", _weight_norm_backward, "out0", "out1")
    pair("_adaptive_avg_pool3d", _adaptive_avg)
    pair("_adaptive_avg_pool3d_backward", lambda grad, input: _adaptive_avg(grad, grad.shape[-3:], input.shape))
    for prefix, reflect in (("reflection", True), ("replication", False)):
        pair(f"{prefix}_pad3d", _pad3d(reflect))
        pair(f"{prefix}_pad3d_backward", _pad3d(reflect, True), "grad_input", overload="grad_input")
    for name, mode, dimensions in (
        ("upsample_linear1d", "linear", 1), ("upsample_trilinear3d", "linear", 3),
        ("upsample_nearest3d", "nearest", 3), ("_upsample_nearest_exact2d", "exact", 2),
    ):
        pair(name, _interpolate(mode, dimensions))
        pair(f"{name}_backward", _interpolate(mode, dimensions, True), "grad_input", overload="grad_input")
    pair("max_unpool2d", _max_unpool)
    pair("multi_margin_loss", _decompose("multi_margin_loss"))
    pair("multi_margin_loss_backward", _multi_margin_backward, "grad_input", overload="grad_input")
    pair("multilabel_margin_loss_forward", _multilabel_forward, "output", "is_target", overload="output")
    pair("multilabel_margin_loss_backward", _multilabel_backward, "grad_input", overload="grad_input")
