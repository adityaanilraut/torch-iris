"""Batched real Jacobi factorizations composed from Iris ATen kernels."""

import torch

from ._linalg import _check_matrix, _column_major
from ._model_ops import _register_pairs


_EPS = torch.finfo(torch.float32).eps
_MAX_SWEEPS = 40


def _schedule(width, device):
    """Round-robin matchings. Each sweep rotates every pair exactly once."""
    if width < 2:
        return ()
    players = list(range(width + width % 2))
    positions = torch.arange(width, device=device)
    matchings = []
    for _ in range(len(players) - 1):
        partners = list(range(width))
        for left, right in zip(players[:len(players) // 2], reversed(players[len(players) // 2:])):
            if left < width and right < width:
                partners[left], partners[right] = right, left
        other = torch.tensor(partners, device=device)
        matchings.append((other, torch.where(positions < other, 1., -1.), positions != other))
        players = [players[0], players[-1], *players[1:-1]]
    return tuple(matchings)


def _rotation(diagonal, cross, other, orientation, enabled):
    delta = diagonal.index_select(-1, other) - diagonal
    twice_cross = 2 * cross
    denominator = delta.abs() + (delta.square() + twice_cross.square()).sqrt()
    sign = torch.where(delta == 0, orientation, delta.sign())
    t = torch.where(enabled, sign * twice_cross / torch.where(denominator == 0, 1, denominator), 0)
    c = (1 + t.square()).rsqrt()
    return c, c * t


def _rotate(matrix, other, c, s):
    return c.unsqueeze(-2) * matrix - s.unsqueeze(-2) * matrix.index_select(-1, other)


def _identity(input, width):
    return torch.eye(width, dtype=input.dtype, device=input.device).expand(*input.shape[:-2], width, width)


def _matrix_scale(input):
    magnitude = input.abs().amax(dim=(-2, -1), keepdim=True)
    if not magnitude.isfinite().all().item():
        raise RuntimeError("Iris Jacobi factorization requires finite float32 inputs")
    return torch.where(magnitude == 0, 1, magnitude)


def _eigh(A, UPLO="L", compute_v=True):
    _check_matrix(A, square=True)
    if UPLO not in ("L", "l", "U", "u"):
        raise RuntimeError("linalg.eigh expects UPLO='L' or 'U'")
    n = A.size(-1)
    vectors = _identity(A, n).clone() if compute_v else A.new_empty((0,))
    if not A.numel():
        return A.new_empty(A.shape[:-1]), _column_major(vectors) if compute_v else vectors
    positions = torch.arange(n, device=A.device)
    triangle = positions[:, None] >= positions if UPLO.upper() == "L" else positions[:, None] <= positions
    value = torch.where(triangle, A, A.mT)
    scale = _matrix_scale(value)
    value = value / scale
    off_diagonal = positions[:, None] != positions
    schedule = _schedule(n, A.device)
    for _ in range(_MAX_SWEEPS):
        for other, orientation, enabled in schedule:
            diagonal = value.diagonal(dim1=-2, dim2=-1)
            pairs = other.reshape(*([1] * (value.ndim - 2)), n, 1).expand(*value.shape[:-1], 1)
            cross = value.gather(-1, pairs).squeeze(-1)
            c, s = _rotation(diagonal, cross, other, orientation, enabled & (cross.abs() > _EPS))
            value = _rotate(_rotate(value, other, c, s).mT, other, c, s).mT
            # Suppress the rounding drift between equivalent triangles.
            value = (value + value.mT) * 0.5
            if compute_v:
                vectors = _rotate(vectors, other, c, s)
        if (torch.where(off_diagonal, value.abs(), 0).amax() <= 4 * _EPS).item():
            break
    else:
        raise RuntimeError("Iris symmetric Jacobi eigensolver did not converge")
    values, order = (value.diagonal(dim1=-2, dim2=-1) * scale.squeeze(-1)).sort(dim=-1)
    if compute_v:
        vectors = _column_major(vectors.gather(-1, order.unsqueeze(-2).expand(vectors.shape)))
    return values, vectors


def _complete_left(columns, singular, full_matrices):
    rows, width = columns.shape[-2:]
    output_width = rows if full_matrices else width
    scales = columns.abs().amax(dim=-2)
    scaled = columns / torch.where(scales > 0, scales, 1).unsqueeze(-2)
    length = scaled.square().sum(dim=-2).sqrt()
    scaled = scaled / torch.where(length > 0, length, 1).unsqueeze(-2)
    cutoff = singular[..., :1] * (_EPS * rows)
    reliable = singular > cutoff
    if output_width == width and reliable.all().item():
        return scaled
    indices = torch.arange(rows, device=columns.device)
    vectors = []
    for i in range(output_width):
        if i:
            q = torch.stack(vectors, dim=-1)
            pick = q.square().sum(-1).argmin(-1, keepdim=True)
        else:
            pick = columns.new_zeros(columns.shape[:-2] + (1,), dtype=torch.long)
        basis = (indices == pick).float().unsqueeze(-1)
        if i:
            for _ in range(2):
                basis = basis - q @ (q.mT @ basis)
        basis = basis.squeeze(-1)
        basis = basis / basis.square().sum(-1, keepdim=True).sqrt()
        if i < width:
            actual = scaled[..., :, i]
            usable = length[..., i] > 0
            if i:
                overlap = (q.mT @ actual.unsqueeze(-1)).squeeze(-1).abs().amax(-1)
                usable = usable & (overlap <= 8 * _EPS)
            vector = torch.where((reliable[..., i] | usable)[..., None], actual, basis)
        else:
            vector = basis
        vectors.append(vector)
    return torch.stack(vectors, dim=-1)


def _svd(A, full_matrices=False, compute_uv=True, *, driver=None):
    _check_matrix(A)
    if driver is not None:
        raise RuntimeError("Iris SVD uses its Jacobi implementation and does not accept a CUDA driver")
    m, n = A.shape[-2:]
    width = min(m, n)
    empty = A.new_empty((0,))
    if not A.numel():
        S = A.new_empty((*A.shape[:-2], width))
        if not compute_uv:
            return empty, S, empty.clone()
        U = _identity(A, m)[..., :m if full_matrices else width].clone()
        Vh = _identity(A, n)[..., :n if full_matrices else width, :].clone()
        return _column_major(U), S, _column_major(Vh)
    wide = m < n
    value = A.mT if wide else A
    scale = _matrix_scale(value)
    value = value / scale
    V = _identity(A, width) if compute_uv else empty
    schedule = _schedule(width, A.device)
    tolerance = 4 * _EPS
    for _ in range(_MAX_SWEEPS):
        for other, orientation, enabled in schedule:
            partner = value.index_select(-1, other)
            scale_pair = torch.maximum(value.abs().amax(-2), partner.abs().amax(-2))
            scale_pair = torch.where(scale_pair > 0, scale_pair, 1).unsqueeze(-2)
            left, right = value / scale_pair, partner / scale_pair
            left_sq, right_sq = left.square().sum(-2), right.square().sum(-2)
            cross = (left * right).sum(-2)
            # Pair-local scaling is shared by partners. Their squared norms differ.
            rotate = enabled & (cross.abs() > tolerance * left_sq.sqrt() * right_sq.sqrt())
            c, s = _rotation(left_sq, cross, other, orientation, rotate)
            value = c.unsqueeze(-2) * value - s.unsqueeze(-2) * partner
            if compute_uv:
                V = _rotate(V, other, c, s)
        peak = value.abs().amax(dim=-2, keepdim=True)
        unit = value / torch.where(peak > 0, peak, 1)
        length = unit.square().sum(-2, keepdim=True).sqrt()
        norm = peak * length
        unit = unit / torch.where(length > 0, length, 1)
        significant = (norm > norm.amax(-1, keepdim=True) * (_EPS * max(m, n))).squeeze(-2)
        positions = torch.arange(width, device=A.device)
        check = (positions[:, None] != positions) & significant[..., :, None] & significant[..., None, :]
        if (torch.where(check, (unit.mT @ unit).abs(), 0).amax() <= 8 * _EPS).item():
            break
    else:
        raise RuntimeError("Iris one-sided Jacobi SVD did not converge")
    S, order = (norm.squeeze(-2) * scale.squeeze(-1)).sort(dim=-1, descending=True)
    if not compute_uv:
        return empty, S, empty.clone()
    value = value.gather(-1, order.unsqueeze(-2).expand(value.shape))
    left = _complete_left(value, S, full_matrices)
    right = V.gather(-1, order.unsqueeze(-2).expand(V.shape))
    U, Vh = (right, left.mT) if wide else (left, right.mT)
    return _column_major(U), S, _column_major(Vh)


def _tolerance(A, value):
    if value is None:
        return None
    if value.requires_grad or value.is_complex() or value.device != A.device:
        raise RuntimeError("Iris pinv tolerances must be real Iris tensors without gradients")
    return value.float().unsqueeze(-1)


def _pinv(A, *, atol=None, rtol=None, hermitian=False):
    _check_matrix(A, square=hermitian)
    atol = _tolerance(A, atol)
    rtol = _tolerance(A, rtol)
    default_rtol = max(A.shape[-2:]) * _EPS
    if rtol is None:
        rtol = default_rtol if atol is None else torch.where(atol > 0, 0., default_rtol)
    if hermitian:
        vals, U = torch.linalg.eigh(A)
        V = U
    else:
        U, vals, Vh = torch.linalg.svd(A, full_matrices=False)
        V = Vh.mT
    if vals.size(-1) == 0:
        return _column_major(A.mT.clone())
    magnitude = vals.abs()
    limit = magnitude.amax(-1, keepdim=True) * rtol
    limit = limit.clamp_min(0.) if atol is None else torch.maximum(limit, atol)
    retained = magnitude > limit
    inverse = torch.where(retained, torch.where(retained, vals, 1).reciprocal(), 0.)
    return _column_major((V * inverse.unsqueeze(-2)) @ U.mT)


def _pinv_scalar(A, *, atol=None, rtol=None, hermitian=False):
    # Stock scalar overload constructs FP64 tensors before dispatching to the
    # generated autograd kernel for the tensor-tolerance overload.
    atol = None if atol is None else A.new_full((), atol)
    rtol = None if rtol is None else A.new_full((), rtol)
    return torch.ops.aten.linalg_pinv.atol_rtol_tensor(A, atol=atol, rtol=rtol, hermitian=hermitian)


def _pinv_rcond(A, rcond, hermitian=False):
    return _pinv_scalar(A, rtol=rcond, hermitian=hermitian)


def _pinv_rcond_tensor(A, rcond, hermitian=False):
    return torch.ops.aten.linalg_pinv.atol_rtol_tensor(A, rtol=rcond, hermitian=hermitian)


def _lstsq(A, B, rcond=None, *, driver=None):
    _check_matrix(A)
    if B.dtype != A.dtype or B.device != A.device or not B.ndim:
        raise RuntimeError("linalg.lstsq expects a matching floating Iris right-hand side")
    if driver not in (None, "gels", "gelsd", "gelss"):
        raise RuntimeError("Iris lstsq accepts gels, gelsd or gelss. Pivoted-QR gelsy is unavailable")
    if B.ndim not in (A.ndim, A.ndim - 1):
        raise RuntimeError("linalg.lstsq expects A.ndim - B.ndim to be zero or one")
    m, n = A.shape[-2:]
    vector = B.ndim == 1 or (B.ndim == A.ndim - 1 and B.shape == A.shape[:-1])
    B = B.unsqueeze(-1) if vector else B
    if B.size(-2) != m:
        raise RuntimeError("linalg.lstsq expects the same number of rows in A and B")
    batch = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    empty = A.new_empty((0,))
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    cutoff = (max(m, n) * _EPS) if rcond is None or rcond < 0 else rcond
    if driver == "gels":
        cutoff = _EPS * max(m, n)
    keep = S > S[..., :1] * cutoff
    rank = keep.sum(-1, dtype=torch.long)
    if driver == "gels" and not (rank == min(m, n)).all().item():
        raise RuntimeError("Iris lstsq driver gels requires full column or row rank")
    inverse = torch.where(keep, torch.where(keep, S, 1).reciprocal(), 0.)
    solution = _column_major(Vh.mT @ ((U.mT @ B) * inverse.unsqueeze(-1)))
    residual = empty
    if driver is not None and m > n and (rank == n).all().item():
        residual = (A @ solution - B).square().sum(-2).expand(*batch, B.size(-1))
    singular = S if driver in ("gelsd", "gelss") else empty
    if vector:
        solution = solution.squeeze(-1)
    return solution, residual, rank if driver != "gels" else empty.long(), singular


def register(library):
    pair = _register_pairs(library)
    for name, fn, overload, outputs in (
        ("_linalg_eigh", _eigh, "eigenvalues", ("eigenvalues", "eigenvectors")),
        ("_linalg_svd", _svd, "U", ("U", "S", "Vh")),
        ("linalg_lstsq", _lstsq, "out", ("solution", "residuals", "rank", "singular_values")),
        ("linalg_pinv.atol_rtol_tensor", _pinv, "atol_rtol_tensor_out", ("out",)),
        ("linalg_pinv.atol_rtol_float", _pinv_scalar, "atol_rtol_float_out", ("out",)),
        ("linalg_pinv", _pinv_rcond, "out", ("out",)),
        ("linalg_pinv.rcond_tensor", _pinv_rcond_tensor, "out_rcond_tensor", ("out",)),
    ):
        pair(name, fn, *outputs, overload=overload)
    autograd = torch.library.Library("aten", "IMPL", "AutogradPrivateUse1")
    autograd.impl("linalg_pinv.atol_rtol_float", _pinv_scalar)
    autograd.impl("linalg_pinv", _pinv_rcond)
    autograd.impl("linalg_pinv.rcond_tensor", _pinv_rcond_tensor)
    return autograd
