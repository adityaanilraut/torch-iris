"""ATen model kernels composed from Iris tensor operations."""

import math

import torch


def _decompose(name):
    def run(*args, **kwargs):
        from torch._decomp import decompositions

        return getattr(decompositions, name)(*args, **kwargs)

    return run


def _write_out(destination, value):
    if destination.device != value.device or destination.dtype != value.dtype:
        raise RuntimeError("out tensor must match the result device and dtype")
    if destination.shape != value.shape:
        destination.resize_(value.shape)
    return destination.copy_(value)


def _out(function, *names):
    def run(*args, **kwargs):
        outputs = tuple(kwargs.pop(name) for name in names)
        result = function(*args, **kwargs)
        for destination, value in zip(outputs, (result,) if len(outputs) == 1 else result):
            _write_out(destination, value)
        return outputs[0] if len(outputs) == 1 else outputs

    return run


def _register_pairs(library):
    def pair(name, function, *outputs, overload="out"):
        library.impl(name, function)
        library.impl(f"{name.partition('.')[0]}.{overload}", _out(function, *(outputs or ("out",))))

    return pair


def _col2im(input, output_size, kernel_size, dilation, padding, stride):
    if input.ndim not in (2, 3):
        raise RuntimeError("col2im expects a 2D or 3D input")
    if input.shape[-1] == 0:
        raise RuntimeError("col2im expects a non-empty column dimension")
    if any(
        step <= 0 or size + 2 * pad <= dil * (kernel - 1)
        for size, pad, dil, kernel, step in zip(output_size, padding, dilation, kernel_size, stride)
    ):
        raise RuntimeError("col2im expects a positive number of windows along each axis")
    return _decompose("col2im")(input, list(output_size), list(kernel_size), list(dilation), list(padding), list(stride)).contiguous()


def _bce(input, target, weight=None, reduction=1):
    if input.shape != target.shape or input.dtype != target.dtype:
        raise RuntimeError("binary_cross_entropy expects matching input and target shapes and dtypes")
    for name, value in (("input", input), ("target", target)):
        if not ((value >= 0) & (value <= 1)).all().item():
            raise RuntimeError(f"all elements of {name} should be between 0 and 1")
    return _decompose("binary_cross_entropy")(input, target, weight, reduction)


def _smooth_l1(input, target, reduction=1, beta=1.0):
    if not beta >= 0:
        raise RuntimeError("smooth_l1_loss expects a non-negative beta")
    return _decompose("smooth_l1_loss")(input, target, reduction, beta)


def _huber(input, target, reduction=1, delta=1.0):
    if not delta > 0:
        raise RuntimeError("huber_loss expects a positive delta")
    dtype = torch.result_type(input, target)
    error = (input.float() - target.float()).abs()
    loss = torch.where(error < delta, 0.5 * error.square(), delta * (error - 0.5 * delta))
    if reduction == 1:
        loss = loss.mean()
    elif reduction == 2:
        loss = loss.sum()
    return loss.to(dtype)


def _pool_windows(size, output_size, device):
    positions = torch.arange(output_size, device=device)
    start = (positions * size).div(output_size, rounding_mode="floor")
    end = ((positions + 1) * size + output_size - 1).div(output_size, rounding_mode="floor")
    # An adaptive window can be one cell wider than ceil(size / output_size).
    width = min(size, (size + output_size - 1) // output_size + 1)
    window = start[:, None] + torch.arange(width, device=device)
    return window, start, end


def _pool_axis(size, output_size, device):
    window, _, end = _pool_windows(size, output_size, device)
    return torch.minimum(window, end[:, None] - 1)


def _scatter_axis(value, index, size):
    return value.new_zeros((*value.shape[:-1], size)).scatter_add_(-1, index.expand(value.shape), value)


def _adaptive_max(input, output_size):
    if input.ndim not in (3, 4) or min(input.shape[-3:]) == 0:
        raise RuntimeError("adaptive_max_pool2d expects a non-empty CHW or NCHW input")
    height, width = input.shape[-2:]
    out_h, out_w = output_size
    if min(out_h, out_w) < 0:
        raise RuntimeError("adaptive_max_pool2d output size cannot be negative")
    shape = (*input.shape[:-2], out_h, out_w)
    if not out_h or not out_w or not input.numel():
        return input.new_empty(shape), input.new_empty(shape, dtype=torch.long)
    rows = _pool_axis(height, out_h, input.device)
    cols = _pool_axis(width, out_w, input.device)
    offsets = (rows[:, None, :, None] * width + cols[None, :, None, :]).flatten(-2)
    windows = input.flatten(-2)[..., offsets]
    values, indices = windows.max(dim=-1)
    # ATen picks the last NaN in a window, and the first occurrence for finite ties.
    found_nan = windows.isnan()
    positions = torch.arange(windows.size(-1), device=input.device)
    last_nan = torch.where(found_nan, positions, -1).amax(dim=-1)
    indices = torch.where(last_nan >= 0, last_nan, indices)
    indices = offsets.expand(windows.shape).gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    return values, indices


def _adaptive_max_backward(grad_output, input, indices):
    flat = input.new_zeros((*input.shape[:-2], input.size(-2) * input.size(-1)), dtype=torch.float32)
    flat.scatter_add_(-1, indices.flatten(-2), grad_output.flatten(-2).float())
    return flat.reshape(input.shape).to(input.dtype)


def _bag_add(values, indices, count):
    shape = (count, *values.shape[1:])
    index = indices.view(-1, *([1] * (values.ndim - 1))).long().expand(values.shape)
    return values.new_zeros(shape).scatter_add_(0, index, values)


def _embedding_bag(weight, indices, offsets, scale_grad_by_freq=False, mode=0, sparse=False,
                   per_sample_weights=None, include_last_offset=False, padding_idx=-1):
    if weight.ndim != 2 or indices.ndim != 1 or offsets.ndim != 1 or mode not in (0, 1, 2):
        raise RuntimeError("invalid embedding_bag dimensions or mode")
    if weight.dtype not in (torch.float32, torch.float16) or indices.dtype not in (torch.int32, torch.int64) or offsets.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("invalid embedding_bag weight or index dtype")
    if per_sample_weights is not None and (
        mode != 0 or per_sample_weights.shape != indices.shape or per_sample_weights.dtype != weight.dtype
    ):
        raise RuntimeError("embedding_bag per-sample weights require sum mode and matching shape and dtype")
    dtype = torch.promote_types(indices.dtype, offsets.dtype)
    offsets = offsets.to(dtype)
    indices = indices.to(dtype)
    bags = offsets.numel() - int(include_last_offset)
    if bags < 0 or (bags == 0 and indices.numel()):
        raise RuntimeError("embedding_bag requires offsets for the supplied indices")
    if offsets.numel() and (
        offsets[0].item() != 0
        or offsets[-1].item() > indices.numel()
        or (offsets[1:] < offsets[:-1]).any().item()
    ):
        raise RuntimeError("embedding_bag offsets must start at zero and be sorted within the index range")
    if not indices.numel():
        sizes = offsets.new_zeros((bags,))
        best = offsets.new_zeros((bags, weight.size(1))) if mode == 2 else sizes.clone()
        return weight.new_zeros((bags, weight.size(1))), indices.new_empty((0,)), sizes, best
    positions = torch.arange(indices.numel(), device=indices.device)
    left = torch.zeros_like(positions)
    right = torch.full_like(positions, bags)
    for _ in range(bags.bit_length()):
        mid = (left + right).div(2, rounding_mode="trunc").clamp(max=bags - 1)
        advance = offsets.gather(0, mid) <= positions
        active = left < right
        left = torch.where(active & advance, mid + 1, left)
        right = torch.where(active & ~advance, mid, right)
    offset2bag = left - 1
    valid = (indices != padding_idx)
    size = _bag_add(valid.to(dtype), offset2bag, bags)
    rows = weight.index_select(0, indices).float()
    if mode == 2:
        keys = offset2bag[:, None].expand(rows.shape)
        maximum = rows.new_full((bags, rows.size(1)), -torch.inf)
        maximum.scatter_reduce_(0, keys, torch.where(valid[:, None], rows, -torch.inf), "amax")
        selected = maximum.index_select(0, offset2bag)
        winners = valid[:, None] & ((rows == selected) | (rows.isnan() & selected.isnan()))
        count = indices.numel()
        best = positions.new_full(maximum.shape, count)
        best.scatter_reduce_(0, keys, torch.where(winners, positions[:, None], count), "amin")
        best = indices.gather(0, best.clamp(max=count - 1).flatten()).view(best.shape)
        best = torch.where(size[:, None] != 0, best, 0)
        output = torch.where(size[:, None] != 0, maximum, 0)
    else:
        if per_sample_weights is not None:
            rows = rows * per_sample_weights.float()[:, None]
        output = _bag_add(torch.where(valid[:, None], rows, 0), offset2bag, bags)
        if mode == 1:
            output = output / size.clamp(min=1)[:, None]
        else:
            size = offsets.new_zeros((bags,))
        best = size.clone()
    return output.to(weight.dtype), offset2bag.to(dtype), size, best


def _embedding_bag_dense_backward(grad, indices, offset2bag, bag_size, maximum_indices, num_weights,
                                  scale_grad_by_freq, mode, per_sample_weights, padding_idx=-1):
    values = grad.float()
    if mode == 2:
        output = values.new_zeros((num_weights, grad.size(1)))
        if num_weights == 0:
            return output.to(grad.dtype)
        nonempty = torch.where(bag_size[:, None] != 0, values, 0)
        return output.scatter_add_(0, maximum_indices.long(), nonempty).to(grad.dtype)
    rows = values.index_select(0, offset2bag)
    if mode == 1:
        rows = rows / bag_size.index_select(0, offset2bag).clamp(min=1)[:, None]
    if per_sample_weights is not None:
        rows = rows * per_sample_weights.float()[:, None]
    if scale_grad_by_freq:
        counts = _bag_add(torch.ones_like(indices), indices, num_weights)
        rows = rows / counts.index_select(0, indices)[:, None]
    rows = torch.where((indices != padding_idx)[:, None], rows, 0)
    return _bag_add(rows, indices, num_weights).to(grad.dtype)


def _embedding_bag_backward(grad, indices, offsets, offset2bag, bag_size, maximum_indices,
                            num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights=None, padding_idx=-1):
    if sparse:
        raise RuntimeError("Iris EmbeddingBag backward requires dense gradients")
    return _embedding_bag_dense_backward(
        grad, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq,
        mode, per_sample_weights, padding_idx,
    )


def _embedding_bag_per_sample_weights_backward(grad, weight, indices, offsets, offset2bag, mode, padding_idx=-1):
    if mode != 0:
        raise RuntimeError("embedding_bag per-sample weights require sum mode")
    dtype = grad.dtype
    dots = (grad.index_select(0, offset2bag).float() * weight.index_select(0, indices).float()).sum(-1)
    return torch.where(indices != padding_idx, dots, 0).to(dtype)


def _resize_weights(size, out_size, align_corners, scale_factor, taps, antialias, device):
    scale = ((size - 1) / (out_size - 1) if out_size > 1 else 0.0) if align_corners else (
        1.0 / scale_factor if scale_factor is not None and scale_factor > 0 else size / out_size
    )
    position = torch.arange(out_size, device=device, dtype=torch.float32)
    if antialias:
        stretch = max(1.0, scale)
        support = (taps / 2) * stretch
        center = (position + 0.5) * scale
        start = (center - support + 0.5).floor().long().clamp(min=0)
        span = min(size, math.ceil(2 * support) + 1)
        indices = start[:, None] + torch.arange(span, device=device)
        distance = ((indices.float() + 0.5 - center[:, None]) / stretch).abs()
        valid = indices < size
        coefficient = -0.5
    else:
        center = position * scale if align_corners else (position + 0.5) * scale - 0.5
        start = center.floor().long() - 1
        indices = start[:, None] + torch.arange(4, device=device)
        distance = (indices.float() - center[:, None]).abs()
        valid = None
        coefficient = -0.75
    if taps == 2:
        weights = (1 - distance).clamp(min=0)
    else:
        near = ((coefficient + 2) * distance - (coefficient + 3)) * distance.square() + 1
        far = ((coefficient * distance - 5 * coefficient) * distance + 8 * coefficient) * distance - 4 * coefficient
        weights = torch.where(distance < 1, near, torch.where(distance < 2, far, 0))
    if antialias:
        weights = torch.where(valid, weights, 0)
        total = weights.sum(-1, keepdim=True)
        weights = weights / torch.where(total != 0, total, 1)
    return indices.clamp(0, size - 1), weights, valid


def _resize_axis(input, size, out_size, scale, align_corners, taps, antialias, backward):
    if size == out_size:
        return input
    indices, weights, valid = _resize_weights(size, out_size, align_corners, scale, taps, antialias, input.device)
    if backward:
        contributions = input.unsqueeze(-1) * weights
        if valid is not None:
            contributions = torch.where(valid, contributions, 0)
        return _scatter_axis(contributions.flatten(-2), indices.flatten(), size)
    values = input[..., indices]
    if valid is not None:
        values = torch.where(valid, values, 0)
    return (values * weights).sum(-1)


def _resize(taps, antialias, backward=False):
    def run(input, output_size, *args, **kwargs):
        input_size = args[0] if backward else input.shape
        if backward:
            args = args[1:]
        # The positional case comes from ATen-generated autograd and vec wrappers.
        align_corners = args[0] if args else kwargs.pop("align_corners")
        scales_h = args[1] if len(args) > 1 else kwargs.pop("scales_h", None)
        scales_w = args[2] if len(args) > 2 else kwargs.pop("scales_w", None)
        if len(input_size) != 4 or min(input_size[1:]) <= 0 or min(output_size) <= 0:
            raise RuntimeError("upsample2d expects a 4D input with positive spatial sizes")
        if input.dtype not in (torch.float32, torch.float16, torch.uint8) or (backward and input.dtype == torch.uint8):
            raise RuntimeError("Iris resize supports float32, float16 and uint8 forward")
        height, width = input_size[-2:]
        out_h, out_w = output_size
        value = input.float()
        if backward:
            value = _resize_axis(value.transpose(-1, -2), height, out_h, scales_h, align_corners, taps, antialias, True).transpose(-1, -2)
            value = _resize_axis(value, width, out_w, scales_w, align_corners, taps, antialias, True)
        else:
            value = _resize_axis(value, width, out_w, scales_w, align_corners, taps, antialias, False)
            value = _resize_axis(value.transpose(-1, -2), height, out_h, scales_h, align_corners, taps, antialias, False).transpose(-1, -2)
        if input.dtype == torch.uint8:
            value = value.round().clamp(0, 255)
        return value.to(input.dtype).contiguous().clone() if value is input else value.to(input.dtype).contiguous()

    return run


def register(library):
    pair = _register_pairs(library)

    pair("im2col", _decompose("im2col"))
    pair("col2im", _col2im)
    pair("smooth_l1_loss", _smooth_l1)
    pair("smooth_l1_loss_backward", _decompose("smooth_l1_loss_backward"), "grad_input", overload="grad_input")
    pair("huber_loss", _huber)
    pair("huber_loss_backward", _decompose("huber_loss_backward"), "grad_input")
    pair("binary_cross_entropy", _bce)
    pair("binary_cross_entropy_backward", _decompose("binary_cross_entropy_backward"), "grad_input", overload="grad_input")
    pair("adaptive_max_pool2d", _adaptive_max, "out", "indices")
    pair("adaptive_max_pool2d_backward", _adaptive_max_backward, "grad_input", overload="grad_input")
    outputs = ("out0", "out1", "out2", "out3")
    pair("_embedding_bag", _embedding_bag, *outputs)
    pair("_embedding_bag_forward_only", _embedding_bag, *outputs)
    pair("_embedding_bag_dense_backward", _embedding_bag_dense_backward)
    library.impl("_embedding_bag_backward", _embedding_bag_backward)
    pair("_embedding_bag_per_sample_weights_backward", _embedding_bag_per_sample_weights_backward)
    for name, taps, antialias in (
        ("upsample_bicubic2d", 4, False),
        ("_upsample_bilinear2d_aa", 2, True),
        ("_upsample_bicubic2d_aa", 4, True),
    ):
        pair(name, _resize(taps, antialias))
        pair(f"{name}_backward", _resize(taps, antialias, backward=True), "grad_input", overload="grad_input")
