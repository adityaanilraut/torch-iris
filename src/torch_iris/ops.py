"""Native quantized and fused Intel Iris Xe operators."""

import torch
from . import _C


def quantize_weight_int8(weight: torch.Tensor, group_size: int = 128, symmetric: bool = True):
    """Quantize a 2D [N, K] floating-point weight matrix to group-wise uint8 + scales (+ optional zeros).

    Dequantization formula: w_approx = (qweight.float() - zero) * scale
    where zero = 128.0 when symmetric=True (zeros=None) or per-group zeros when symmetric=False.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight [N, K], got shape {tuple(weight.shape)}")
    n, k = weight.shape
    if group_size <= 0 or k % group_size != 0:
        raise ValueError(f"K ({k}) must be divisible by group_size ({group_size})")
    num_groups = k // group_size
    w = weight.detach().float().reshape(n, num_groups, group_size)
    if symmetric:
        max_abs = w.abs().amax(dim=-1).clamp(min=1e-8)
        scales = max_abs / 127.0
        q = torch.clamp(torch.round(w / scales.unsqueeze(-1)) + 128.0, 0.0, 255.0).to(torch.uint8)
        return q.reshape(n, k), scales.to(weight.dtype), None
    w_min = w.amin(dim=-1)
    w_max = w.amax(dim=-1)
    scales = ((w_max - w_min) / 255.0).clamp(min=1e-8)
    zeros = torch.clamp(torch.round(-w_min / scales), 0.0, 255.0)
    q = torch.clamp(torch.round(w / scales.unsqueeze(-1) + zeros.unsqueeze(-1)), 0.0, 255.0).to(torch.uint8)
    return q.reshape(n, k), scales.to(weight.dtype), zeros.to(weight.dtype)


def quantize_weight_int4(weight: torch.Tensor, group_size: int = 128, symmetric: bool = True):
    """Quantize and pack a 2D [N, K] floating-point weight matrix to group-wise 4-bit uint8 [N, K // 2].

    Low nibble (bits 0..3) stores even K indices; high nibble (bits 4..7) stores odd K indices.
    Dequantization formula: w_approx = (nibble - zero) * scale
    where zero = 8.0 when symmetric=True (zeros=None) or per-group zeros when symmetric=False.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight [N, K], got shape {tuple(weight.shape)}")
    n, k = weight.shape
    if k % 2 != 0:
        raise ValueError(f"K ({k}) must be even for INT4 packing")
    if group_size <= 0 or group_size % 8 != 0 or k % group_size != 0:
        raise ValueError(f"K ({k}) must be divisible by group_size ({group_size}), which must be a multiple of 8")
    num_groups = k // group_size
    w = weight.detach().float().reshape(n, num_groups, group_size)
    if symmetric:
        max_abs = w.abs().amax(dim=-1).clamp(min=1e-8)
        scales = max_abs / 7.0
        q = torch.clamp(torch.round(w / scales.unsqueeze(-1)) + 8.0, 0.0, 15.0).to(torch.uint8)
        zeros = None
    else:
        w_min = w.amin(dim=-1)
        w_max = w.amax(dim=-1)
        scales = ((w_max - w_min) / 15.0).clamp(min=1e-8)
        zeros = torch.clamp(torch.round(-w_min / scales), 0.0, 15.0)
        q = torch.clamp(torch.round(w / scales.unsqueeze(-1) + zeros.unsqueeze(-1)), 0.0, 15.0).to(torch.uint8)
        zeros = zeros.to(weight.dtype)
    q_flat = q.reshape(n, k // 2, 2)
    packed = (q_flat[:, :, 0] | (q_flat[:, :, 1] << 4)).contiguous()
    return packed, scales.to(weight.dtype), zeros


def linear_int8(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    group_size: int = 128,
) -> torch.Tensor:
    """Weight-only INT8 linear layer with group-wise dequantization in registers on Iris Xe."""
    if x.device.type == "iris":
        return _C._linear_int8(x, qweight, scales, zeros, bias, group_size)
    n, k = qweight.shape
    num_groups = k // group_size
    zp = 128.0 if zeros is None else zeros.float().unsqueeze(-1)
    w = (qweight.float().reshape(n, num_groups, group_size) - zp) * scales.float().unsqueeze(-1)
    w = w.reshape(n, k)
    out = torch.nn.functional.linear(x.float(), w, bias.float() if bias is not None else None)
    return out.to(x.dtype)


def linear_int4(
    x: torch.Tensor,
    qweight_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    group_size: int = 128,
) -> torch.Tensor:
    """Weight-only packed INT4 linear layer with group-wise dequantization in registers on Iris Xe."""
    if x.device.type == "iris":
        return _C._linear_int4(x, qweight_packed, scales, zeros, bias, group_size)
    n, k_half = qweight_packed.shape
    k = k_half * 2
    num_groups = k // group_size
    lo = (qweight_packed & 0x0F).float()
    hi = (qweight_packed >> 4).float()
    q = torch.stack((lo, hi), dim=-1).reshape(n, num_groups, group_size)
    zp = 8.0 if zeros is None else zeros.float().unsqueeze(-1)
    w = ((q - zp) * scales.float().unsqueeze(-1)).reshape(n, k)
    out = torch.nn.functional.linear(x.float(), w, bias.float() if bias is not None else None)
    return out.to(x.dtype)
