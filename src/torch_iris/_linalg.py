"""FP32 factorizations for dense Iris tensors."""

import math

import torch


def _check_matrix(input, square=False):
    if input.dtype != torch.float32 or input.ndim < 2:
        raise RuntimeError("Iris dense factorizations expect a float32 matrix or batch")
    if square and input.size(-1) != input.size(-2):
        raise RuntimeError("expected square matrices")


def _column_major(input):
    return input.transpose(-2, -1).contiguous().transpose(-2, -1)


def _swap_rows(input, positions, row, pivot):
    lookup = torch.where(positions == row, pivot[..., None],
                         torch.where(positions == pivot[..., None], row, positions))
    return input.gather(-2, lookup.unsqueeze(-1).expand(input.shape))


def _lu_factor(input, *, pivot=True, check_errors=False):
    _check_matrix(input)
    if not pivot:
        raise RuntimeError("Iris LU factorization requires pivoting")
    m, n = input.shape[-2:]
    lu = input
    positions = torch.arange(m, device=input.device)
    info = input.new_zeros(input.shape[:-2], dtype=torch.int32)
    pivots = []
    for step in range(min(m, n)):
        selected = lu[..., step:, step].abs().argmax(-1) + step
        pivots.append((selected + 1).int())
        lu = _swap_rows(lu, positions, step, selected)
        diagonal = lu[..., step, step]
        singular = diagonal == 0
        info = torch.where((info == 0) & singular, step + 1, info)
        if step + 1 < m:
            multipliers = lu[..., step + 1:, step] / torch.where(singular, 1, diagonal)[..., None]
            trailing = lu[..., step + 1:, step + 1:] - multipliers[..., :, None] * lu[..., step:step + 1, step + 1:]
            bottom = torch.cat((lu[..., step + 1:, :step], multipliers[..., None], trailing), dim=-1)
            lu = torch.cat((lu[..., :step + 1, :], bottom), dim=-2)
    pivots = torch.stack(pivots, dim=-1) if pivots else input.new_empty((*input.shape[:-2], 0), dtype=torch.int32)
    if check_errors:
        torch.ops.aten._linalg_check_errors(info, "torch.linalg.lu_factor_ex", is_matrix=input.ndim == 2)
    return _column_major(lu.clone() if lu is input else lu), pivots, info


def _triangular(input, B, *, upper, left=True, unitriangular=False):
    _check_matrix(input, square=True)
    _check_matrix(B)
    if not left:
        return _triangular(input.mT, B.mT, upper=not upper, unitriangular=unitriangular).mT
    width = input.size(-1)
    if B.size(-2) != width:
        raise RuntimeError("incompatible triangular system shapes")
    batch = torch.broadcast_shapes(input.shape[:-2], B.shape[:-2])
    rhs = B.expand(*batch, *B.shape[-2:]).clone()
    for step in range(width - 1, -1, -1) if upper else range(width):
        row = rhs[..., step, :]
        if not unitriangular:
            row = row / input[..., step, step, None]
        if upper:
            remaining = rhs[..., :step, :] - input[..., :step, step, None] * row.unsqueeze(-2)
            rhs = torch.cat((remaining, row.unsqueeze(-2), rhs[..., step + 1:, :]), dim=-2)
        else:
            remaining = rhs[..., step + 1:, :] - input[..., step + 1:, step, None] * row.unsqueeze(-2)
            rhs = torch.cat((rhs[..., :step, :], row.unsqueeze(-2), remaining), dim=-2)
    return _column_major(rhs)


def _triangular_solve(B, A, upper=True, transpose=False, unitriangular=False):
    coeff = A.mT if transpose else A
    X = _triangular(coeff, B, upper=upper != transpose, unitriangular=unitriangular)
    broadcast = A.expand(*X.shape[:-2], *A.shape[-2:])
    return X, _column_major(broadcast.clone())


def _lu_solve(LU, pivots, B, *, left=True, adjoint=False):
    _check_matrix(LU, square=True)
    _check_matrix(B)
    width = LU.size(-1)
    if pivots.dtype != torch.int32 or pivots.ndim != LU.ndim - 1 or pivots.shape != LU.shape[:-1]:
        raise RuntimeError("LU pivots must be int32 with one entry per row")
    if not left:
        return _lu_solve(LU, pivots, B.mT, left=True, adjoint=not adjoint).mT
    if B.size(-2) != width:
        raise RuntimeError("incompatible LU system shapes")
    batch = torch.broadcast_shapes(LU.shape[:-2], B.shape[:-2])
    rhs = B.expand(*batch, *B.shape[-2:])
    positions = torch.arange(width, device=LU.device)
    if adjoint:
        rhs = _triangular(LU.mT, rhs, upper=False)
        rhs = _triangular(LU.mT, rhs, upper=True, unitriangular=True)
        for step in reversed(range(width)):
            rhs = _swap_rows(rhs, positions, step, pivots[..., step].long() - 1)
    else:
        for step in range(width):
            rhs = _swap_rows(rhs, positions, step, pivots[..., step].long() - 1)
        rhs = _triangular(LU, rhs, upper=False, unitriangular=True)
        rhs = _triangular(LU, rhs, upper=True)
    return _column_major(rhs)


def _solve(A, B, *, left=True, check_errors=False):
    vector = B.ndim == 1
    if vector and not left:
        raise RuntimeError("right solve expects a matrix RHS")
    _check_matrix(A, square=True)
    lu, pivots, info = _lu_factor(A)
    if check_errors:
        torch.ops.aten._linalg_check_errors(info, "torch.linalg.solve_ex", is_matrix=A.ndim == 2)
    result = _lu_solve(lu, pivots, B.unsqueeze(-1) if vector else B, left=left)
    return (result.squeeze(-1) if vector else result), lu, pivots, info


def _inverse(A, *, check_errors=False):
    _check_matrix(A, square=True)
    identity = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
    inverse, _, _, info = _solve(A, identity, check_errors=check_errors)
    return inverse, info


def _det(A):
    _check_matrix(A, square=True)
    # The generated determinant backward uses this same choice of saved factor orientation.
    lu, pivots, _ = _lu_factor(A.mT if A.is_contiguous() else A)
    swap = (pivots != torch.arange(1, A.size(-1) + 1, device=A.device)).sum(-1)
    sign = torch.where(swap % 2 == 0, 1.0, -1.0)
    return lu.diagonal(dim1=-2, dim2=-1).prod(-1) * sign, lu, pivots


def _slogdet(A):
    _check_matrix(A, square=True)
    lu, pivots, _ = _lu_factor(A.mT if A.is_contiguous() else A)
    swap = (pivots != torch.arange(1, A.size(-1) + 1, device=A.device)).sum(-1)
    sign = torch.where(swap % 2 == 0, 1.0, -1.0)
    diagonal = lu.diagonal(dim1=-2, dim2=-1)
    return sign * diagonal.sign().prod(-1), diagonal.abs().log().sum(-1), lu, pivots


def _cholesky(A, *, upper=False, check_errors=False):
    _check_matrix(A, square=True)
    width = A.size(-1)
    schur = A.mT if upper else A
    columns = []
    info = A.new_zeros(A.shape[:-2], dtype=torch.int32)
    for step in range(width):
        diagonal = schur[..., 0, 0]
        bad = ~(diagonal > 0)
        info = torch.where((info == 0) & bad, step + 1, info)
        value = torch.where(bad, diagonal, diagonal.sqrt())
        tail = schur[..., 1:, 0] / value[..., None]
        columns.append(torch.cat((A.new_zeros((*A.shape[:-2], step)), value[..., None], tail), dim=-1))
        schur = schur[..., 1:, 1:] - tail[..., :, None] * tail[..., None, :]
    result = torch.stack(columns, dim=-1) if columns else A.clone()
    if check_errors:
        torch.ops.aten._linalg_check_errors(info, "torch.linalg.cholesky_ex", is_matrix=A.ndim == 2)
    return _column_major(result.mT if upper else result), info


def _qr(A, mode="reduced"):
    _check_matrix(A)
    if mode not in ("reduced", "complete", "r"):
        raise RuntimeError("linalg.qr mode must be reduced, complete or r")
    m, n = A.shape[-2:]
    k = min(m, n)
    q = torch.eye(m, device=A.device, dtype=A.dtype).expand(*A.shape[:-2], m, m) if mode != "r" else A.new_empty(0)
    r = A.clone()
    row = torch.arange(m, device=A.device)
    col = torch.arange(n, device=A.device)
    for step in range(k):
        x = torch.where(row >= step, r[..., :, step], 0)
        # Scale before norm computation to avoid overflow and loss of subnormal values.
        magnitude = x.abs().amax(dim=-1, keepdim=True)
        x = x / torch.where(magnitude > 0, magnitude, 1)
        norm = x.square().sum(-1).sqrt()
        pivot = x[..., step]
        sign = torch.where(pivot < 0, -1.0, 1.0)
        # GEQRF leaves an already triangular column alone.
        active = torch.where(row > step, x.abs(), 0).amax(dim=-1) != 0
        v = x + torch.where(row == step, (sign * norm)[..., None], 0)
        sq_norm = v.square().sum(-1)
        factor = torch.where(active, 2 / torch.where(sq_norm > 0, sq_norm, 1), 0)
        projection = (v.unsqueeze(-2) @ r).squeeze(-2) * factor[..., None]
        r = r - v.unsqueeze(-1) * torch.where(col >= step, projection, 0).unsqueeze(-2)
        if mode != "r":
            projection = (q @ v.unsqueeze(-1)).squeeze(-1) * factor[..., None]
            q = q - projection.unsqueeze(-1) * v.unsqueeze(-2)
    if mode != "complete":
        r = r[..., :k, :]
        if mode == "reduced":
            q = q[..., :k]
    if mode != "r" and q.is_contiguous() and k == 0:
        q = q.clone()
    return _column_major(q) if mode != "r" else q, _column_major(r.triu())


def _matrix_exp(A):
    _check_matrix(A, square=True)
    width = A.size(-1)
    if not A.numel():
        return A.clone()
    if width == 1:
        return A.exp()
    identity = torch.eye(width, device=A.device, dtype=A.dtype).expand(A.shape)
    norm = A.abs().sum(dim=-2).amax(dim=-1)
    steps = norm.clamp(min=1).log2().ceil()
    maximum = steps.amax().item()
    if not math.isfinite(maximum):
        raise RuntimeError("Iris matrix_exp requires a finite FP32 matrix 1-norm")
    x = A * (-steps).exp2()[..., None, None]
    term = identity
    value = identity
    for order in range(1, 15):
        term = (term @ x) / order
        value = value + term
    for order in range(int(maximum)):
        value = torch.where((steps > order)[..., None, None], value @ value, value)
    return value


def register(library):
    from ._model_ops import _out

    for name, fn, overload, outputs in (
        ("linalg_lu_factor_ex", _lu_factor, "out", ("LU", "pivots", "info")),
        ("linalg_lu_solve", _lu_solve, "out", ("out",)),
        ("linalg_solve_triangular", _triangular, "out", ("out",)),
        ("triangular_solve", _triangular_solve, "X", ("X", "M")),
        ("_linalg_solve_ex", _solve, "result", ("result", "LU", "pivots", "info")),
        ("linalg_inv_ex", _inverse, "inverse", ("inverse", "info")),
        ("_linalg_det", _det, "result", ("result", "LU", "pivots")),
        ("_linalg_slogdet", _slogdet, "sign", ("sign", "logabsdet", "LU", "pivots")),
        ("linalg_cholesky_ex", _cholesky, "L", ("L", "info")),
        ("linalg_qr", _qr, "out", ("Q", "R")),
    ):
        library.impl(name, fn)
        library.impl(f"{name}.{overload}", _out(fn, *outputs))
    library.impl("linalg_matrix_exp", _matrix_exp)
