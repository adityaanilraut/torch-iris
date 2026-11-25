"""Volumetric grid sampling using Iris gather and its autograd formula."""

from itertools import product

import torch

from ._model_ops import _out


def _coordinate(value, size, padding, align):
    # Treat invalid grid coordinates as the top/left endpoint before converting.
    value = torch.where(value.isnan(), -1, value)
    value = (value + 1) * ((size - 1) / 2 if align else size / 2) - (0 if align else 0.5)
    if padding == 2:
        minimum = 0 if align else -0.5
        span = size - 1 if align else size
        if not span:
            return value * 0
        offset = (value - minimum).abs()
        position = offset - (offset / (2 * span)).floor() * (2 * span)
        value = minimum + span - (position - span).abs()
    if padding:
        value = torch.where(value <= 0, 0, torch.where(value >= size - 1, size - 1, value))
    return value


def _grid(input, grid, interpolation_mode, padding_mode, align_corners):
    if (input.ndim != 5 or grid.ndim != 5 or input.size(0) != grid.size(0)
            or grid.size(-1) != 3 or min(input.shape[-3:]) <= 0 or input.device != grid.device
            or not input.is_floating_point() or input.dtype != grid.dtype):
        raise RuntimeError("grid_sampler_3d expects compatible floating NCDHW input and NDHW3 grid")
    if interpolation_mode not in (0, 1) or padding_mode not in (0, 1, 2):
        raise RuntimeError("grid_sampler_3d accepts bilinear or nearest interpolation and valid padding")
    shape = grid.shape[1:-1]
    values = input.float().flatten(2)
    axes = [_coordinate(coord, size, padding_mode, align_corners)
            for coord, size in zip(grid.float().unbind(-1), reversed(input.shape[-3:]))]
    depth, height, width = input.shape[-3:]

    def load(x, y, z):
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (z >= 0) & (z < depth)
        flat = torch.where(valid, x + y * width + z * (width * height), 0).long().flatten(1)
        sample = values.gather(2, flat.unsqueeze(1).expand(-1, input.size(1), -1))
        return torch.where(valid.flatten(1).unsqueeze(1), sample, 0.)

    if interpolation_mode == 1:
        result = load(*(coord.round() for coord in axes))
        # Nearest-neighbor grid derivatives are identically zero.
        grid_zero = torch.where(grid.isfinite(), grid, 0).sum(-1).flatten(1) * 0
        result = result + grid_zero.unsqueeze(1)
    else:
        lows = [axis.floor() for axis in axes]
        frac = [axis - low for axis, low in zip(axes, lows)]
        result = values.new_zeros((input.size(0), input.size(1), grid[..., 0].flatten(1).size(1)))
        for dx, dy, dz in product((0, 1), repeat=3):
            weight = ((frac[0] if dx else 1 - frac[0]) * (frac[1] if dy else 1 - frac[1])
                      * (frac[2] if dz else 1 - frac[2]))
            weight = torch.where(weight.isfinite(), weight, 0.)
            result = result + load(lows[0] + dx, lows[1] + dy, lows[2] + dz) * weight.flatten(1).unsqueeze(1)
    return result.reshape(input.size(0), input.size(1), *shape).to(input.dtype)


def register(library):
    library.impl("grid_sampler_3d", _grid)
    library.impl("grid_sampler_3d.out", _out(_grid, "out"))
    autograd = torch.library.Library("aten", "IMPL", "AutogradPrivateUse1")
    autograd.impl("grid_sampler_3d", _grid)
    return autograd
