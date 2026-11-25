"""TorchVision detection operations built out of Iris ATen kernels."""

import torch
from torch.nn import functional as F


def _nms(boxes, scores, iou_threshold):
    if boxes.ndim != 2 or boxes.size(-1) != 4 or scores.shape != (boxes.size(0),):
        raise RuntimeError("nms expects [N, 4] boxes and [N] scores")
    if boxes.dtype != scores.dtype or boxes.dtype not in (torch.float32, torch.float16):
        raise RuntimeError("Iris nms requires matching floating-point boxes and scores")
    order = scores.argsort(descending=True, stable=True)
    if not order.numel():
        return order
    corners = boxes.float()
    top_left, bottom_right = corners[:, :2], corners[:, 2:]
    area = (bottom_right - top_left).prod(-1)
    chosen = []
    while order.numel():
        first, rest = order[:1], order[1:]
        chosen.append(first)
        if not rest.numel():
            break
        lo = torch.maximum(top_left.index_select(0, first), top_left.index_select(0, rest))
        hi = torch.minimum(bottom_right.index_select(0, first), bottom_right.index_select(0, rest))
        common = (hi - lo).clamp(min=0).prod(-1)
        union = area.index_select(0, first) + area.index_select(0, rest) - common
        order = rest[~(common / union > iou_threshold)]
    return torch.cat(chosen)


def _roi_inputs(input, rois, pooled_height, pooled_width):
    if input.ndim != 4 or min(input.shape[2:]) <= 0 or rois.ndim != 2 or rois.size(-1) != 5:
        raise RuntimeError("ROI ops expect an NCHW input and [K, 5] boxes")
    if min(pooled_height, pooled_width) < 0:
        raise RuntimeError("ROI pooling sizes must be non-negative")
    if input.dtype != rois.dtype or input.dtype not in (torch.float32, torch.float16):
        raise RuntimeError("Iris ROI ops expect matching floating-point input and boxes")
    boxes = rois.detach().float()
    return input.float().index_select(0, boxes[:, 0].long()), boxes[:, 1:]


def _roi_align(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
    features, boxes = _roi_inputs(input, rois, pooled_height, pooled_width)
    count, channels, height, width = features.shape
    shape = (count, channels, pooled_height, pooled_width)
    if not count or not pooled_height or not pooled_width:
        return features.reshape(-1)[:0].reshape(shape).to(input.dtype)
    x0, y0, x1, y1 = (boxes * spatial_scale - (0.5 if aligned else 0)).unbind(-1)
    extent_y, extent_x = y1 - y0, x1 - x0
    if not aligned:
        extent_y, extent_x = extent_y.clamp(min=1), extent_x.clamp(min=1)
    step_y, step_x = extent_y / pooled_height, extent_x / pooled_width

    def axis(origin, step, bins, image_size):
        if sampling_ratio > 0:
            size = sampling_ratio
            grid_size = torch.full_like(origin, size)
        else:
            grid_size = step.ceil().clamp(min=0)
            size = max(1, int(grid_size.max().item()))
        pos = torch.arange(size, device=input.device, dtype=torch.float32)
        offset = (pos + 0.5) / grid_size.clamp(min=1)[:, None]
        base = torch.arange(bins, device=input.device, dtype=torch.float32)
        coord = origin[:, None, None] + step[:, None, None] * (base[None, :, None] + offset[:, None, :])
        active = (pos < grid_size[:, None])[:, None, :] & (coord >= -1) & (coord <= image_size)
        return ((coord + 0.5) * (2.0 / image_size) - 1).flatten(1), active.flatten(1), grid_size, size

    ys, valid_y, grids_y, sy = axis(y0, step_y, pooled_height, height)
    xs, valid_x, grids_x, sx = axis(x0, step_x, pooled_width, width)
    gx = xs[:, None, :].expand(count, ys.size(-1), xs.size(-1))
    gy = ys[:, :, None].expand(gx.shape)
    grid = torch.stack((gx, gy), dim=-1)
    sampled = F.grid_sample(features, grid, padding_mode="border", align_corners=False)
    sampled = torch.where(valid_y[:, None, :, None] & valid_x[:, None, None, :], sampled, 0)
    pooled = sampled.reshape(count, channels, pooled_height, sy, pooled_width, sx).sum((3, 5))
    return (pooled / (grids_y * grids_x).clamp(min=1)[:, None, None, None]).to(input.dtype)


def _roi_pool(input, rois, spatial_scale, pooled_height, pooled_width):
    features, boxes = _roi_inputs(input, rois, pooled_height, pooled_width)
    count, channels, height, width = features.shape
    shape = (count, channels, pooled_height, pooled_width)
    if not count or not pooled_height or not pooled_width:
        return features.reshape(-1)[:0].reshape(shape).to(input.dtype), features.new_empty(shape, dtype=torch.int32)
    scaled = boxes * spatial_scale
    rounded = scaled.sign() * (scaled.abs() + 0.5).floor()
    x0, y0, x1, y1 = rounded.unbind(-1)
    h = (y1 - y0 + 1).clamp(min=1) / pooled_height
    w = (x1 - x0 + 1).clamp(min=1) / pooled_width
    rows = torch.arange(height, device=input.device)
    cols = torch.arange(width, device=input.device)
    features = features.flatten(-2)
    features = torch.where(features.isnan(), -torch.inf, features)
    bins, indices = [], []
    for row in range(pooled_height):
        start_y = y0 + (row * h).floor()
        end_y = y0 + ((row + 1) * h).ceil()
        use_y = (rows >= start_y[:, None]) & (rows < end_y[:, None])
        for col in range(pooled_width):
            start_x = x0 + (col * w).floor()
            end_x = x0 + ((col + 1) * w).ceil()
            use_x = (cols >= start_x[:, None]) & (cols < end_x[:, None])
            mask = (use_y[:, :, None] & use_x[:, None, :]).flatten(1)
            empty = ~mask.any(-1)
            region = torch.where(mask[:, None, :], features, -torch.inf)
            value, index = region.max(-1)
            index = torch.where(value > -torch.inf, index, -1)
            bins.append(torch.where(empty[:, None], 0, value))
            indices.append(index)
    return torch.stack(bins, -1).reshape(shape).to(input.dtype), torch.stack(indices, -1).reshape(shape).int()


def _deform_conv(input, weight, offset, mask, bias, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
                 groups, offset_groups, use_mask):
    dtype = input.dtype
    if dtype not in (torch.float32, torch.float16) or any(
        t.dtype != dtype or t.device != input.device for t in (weight, offset, mask, bias)
    ):
        raise RuntimeError("Iris deform_conv2d requires matching float32 or float16 tensors")
    if input.ndim != 4 or weight.ndim != 4 or offset.ndim != 4:
        raise RuntimeError("deform_conv2d expects 4D input, weight, and offset")
    n, channels, h, w = input.shape
    output_channels, group_width, kh, kw = weight.shape
    oh = (h + 2 * pad_h - dilation_h * (kh - 1) - 1) // stride_h + 1
    ow = (w + 2 * pad_w - dilation_w * (kw - 1) - 1) // stride_w + 1
    if (min(oh, ow, h, w, kh, kw, groups, offset_groups, channels, output_channels) <= 0
            or channels % offset_groups or channels != groups * group_width or output_channels % groups
            or offset.shape != (n, 2 * offset_groups * kh * kw, oh, ow)
            or (use_mask and mask.shape != (n, offset_groups * kh * kw, oh, ow))
            or bias.shape != (output_channels,)):
        raise RuntimeError("incompatible deform_conv2d input, weight, offset, mask or bias shape")
    y = torch.arange(oh, device=input.device, dtype=torch.float32)[:, None] * stride_h - pad_h
    x = torch.arange(ow, device=input.device, dtype=torch.float32) * stride_w - pad_w
    offset_width = channels // offset_groups
    taps = kh * kw
    offs = offset.float().reshape(n, offset_groups, taps, 2, oh, ow)
    masks = mask.float().reshape(n, offset_groups, taps, oh, ow) if use_mask else None
    source = input.float().reshape(n * offset_groups, offset_width, h, w)
    cols = []
    for tap in range(taps):
        ty, tx = divmod(tap, kw)
        yy = ((y + (ty * dilation_h + 0.5) + offs[:, :, tap, 0]) * (2.0 / h) - 1)
        xx = ((x + (tx * dilation_w + 0.5) + offs[:, :, tap, 1]) * (2.0 / w) - 1)
        grid = torch.stack((xx, yy), -1).reshape(n * offset_groups, oh, ow, 2)
        sample = F.grid_sample(source, grid, align_corners=False).reshape(n, offset_groups, offset_width, oh, ow)
        if use_mask:
            sample = sample * masks[:, :, tap, None]
        cols.append(sample.reshape(n, channels, oh, ow))
    columns = torch.stack(cols, dim=2).reshape(n, groups, group_width * taps, oh * ow)
    filters = weight.float().reshape(groups, output_channels // groups, group_width * taps)
    output = (filters @ columns).reshape(n, output_channels, oh, ow)
    return (output + bias.float()[None, :, None, None]).to(dtype)


def register():
    device = torch.library.Library("torchvision", "IMPL", "PrivateUse1")
    autograd = torch.library.Library("torchvision", "IMPL", "AutogradPrivateUse1")
    device.impl("nms", _nms)
    for name, fn in (("roi_align", _roi_align), ("roi_pool", _roi_pool), ("deform_conv2d", _deform_conv)):
        device.impl(name, fn)
        autograd.impl(name, fn)
    return device, autograd
