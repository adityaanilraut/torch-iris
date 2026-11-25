"""Complex64 Fourier transforms using Iris OpenCL kernels."""

import math

import torch


def _radix2(real, imaginary, forward):
    size = real.size(-1)
    if size <= 1024:
        return torch.ops.torch_iris.fft_radix2(real, imaginary, forward).unbind(-1)
    positions = torch.arange(size, device=real.device)
    reverse = torch.zeros_like(positions)
    for bit in range(size.bit_length() - 1):
        reverse = reverse * 2 + ((positions & (1 << bit)) != 0).long()
    real, imaginary = real.index_select(-1, reverse), imaginary.index_select(-1, reverse)
    sign = -1 if forward else 1
    for bit in range(size.bit_length() - 1):
        half = 1 << bit
        width = half * 2
        angle = torch.arange(half, device=real.device, dtype=torch.float32) * (sign * 2 * math.pi / width)
        cos, sin = angle.cos(), angle.sin()
        r = real.unflatten(-1, (size // width, width))
        i = imaginary.unflatten(-1, (size // width, width))
        ar, br = r[..., :half], r[..., half:]
        ai, bi = i[..., :half], i[..., half:]
        vr, vi = br * cos - bi * sin, br * sin + bi * cos
        real = torch.cat((ar + vr, ar - vr), dim=-1).flatten(-2)
        imaginary = torch.cat((ai + vi, ai - vi), dim=-1).flatten(-2)
    return real, imaginary


def _direct(real, imaginary, forward):
    size = real.size(-1)
    if not real.numel():
        return real, imaginary
    positions = torch.arange(size, device=real.device)
    sign = -1 if forward else 1
    # Bound the temporary coefficient matrices even for non-radix-2 lengths.
    tile = max(1, (1 << 19) // size)
    outputs_r, outputs_i = [], []
    for first in range(0, size, tile):
        frequency = positions[first:first + tile]
        angle = (positions[:, None] * frequency % size).float() * (sign * 2 * math.pi / size)
        cos, sin = angle.cos(), angle.sin()
        outputs_r.append(real @ cos - imaginary @ sin)
        outputs_i.append(real @ sin + imaginary @ cos)
    return torch.cat(outputs_r, dim=-1), torch.cat(outputs_i, dim=-1)


def _transform(real, imaginary, dims, normalization, forward):
    if normalization not in (0, 1, 2):
        raise RuntimeError("invalid FFT normalization")
    scale = 1
    for dim in dims:
        size = real.size(dim)
        if size <= 0:
            raise RuntimeError("FFT transform length must be positive")
        scale *= size
        if size == 1 or not real.numel():
            continue
        transform = _radix2 if not (size & (size - 1)) else _direct
        r, i = transform(real.movedim(dim, -1), imaginary.movedim(dim, -1), forward)
        real, imaginary = r.movedim(-1, dim), i.movedim(-1, dim)
    if normalization:
        divisor = math.sqrt(scale) if normalization == 1 else scale
        real, imaginary = real / divisor, imaginary / divisor
    return real, imaginary


def _complex(real, imag):
    if real.dtype != torch.float32 or imag.dtype != torch.float32:
        raise RuntimeError("Iris complex construction requires float32 components")
    return torch.view_as_complex(torch.stack(torch.broadcast_tensors(real, imag), dim=-1))


def _c2c(input, dim, normalization, forward):
    if input.dtype != torch.complex64:
        raise RuntimeError("Iris FFT supports complex64")
    real, imaginary = torch.view_as_real(input.resolve_conj()).unbind(-1)
    return _complex(*_transform(real, imaginary, dim, normalization, forward))


def _r2c(input, dim, normalization, onesided):
    if input.dtype != torch.float32:
        raise RuntimeError("Iris real FFT supports float32")
    real, imaginary = _transform(input, torch.zeros_like(input), dim, normalization, True)
    if onesided:
        length = input.size(dim[-1]) // 2 + 1
        real, imaginary = real.narrow(dim[-1], 0, length), imaginary.narrow(dim[-1], 0, length)
    return _complex(real, imaginary)


def _c2r(input, dim, normalization, last_dim_size):
    if input.dtype != torch.complex64:
        raise RuntimeError("Iris inverse real FFT supports complex64")
    real, imaginary = torch.view_as_real(input.resolve_conj()).unbind(-1)
    real, imaginary = _transform(real, imaginary, dim[:-1], normalization, False)
    axis = dim[-1]
    length = (last_dim_size - 1) // 2
    tail_r = real.narrow(axis, 1, length).flip((axis,))
    tail_i = imaginary.narrow(axis, 1, length).flip((axis,))
    real = torch.cat((real, tail_r), dim=axis)
    imaginary = torch.cat((imaginary, -tail_i), dim=axis)
    return _transform(real, imaginary, (axis,), normalization, False)[0]


def register(library):
    from ._model_ops import _out

    for name, function in (("complex", _complex), ("_fft_c2c", _c2c), ("_fft_r2c", _r2c), ("_fft_c2r", _c2r)):
        library.impl(name, function)
        library.impl(f"{name}.out", _out(function, "out"))
