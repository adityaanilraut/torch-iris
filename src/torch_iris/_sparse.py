"""COO operations for sparse-dense matrix products and embedding updates."""

import torch


def _coo(reference, indices, values, *, coalesced=False):
    return torch.sparse_coo_tensor(indices, values, reference.shape, is_coalesced=coalesced)


def _keys(input):
    indices = input._indices()
    linear = indices.new_zeros((input._nnz(),))
    for size, coordinates in zip(input.shape[:input.sparse_dim()], indices.unbind(0)):
        linear = linear * size + coordinates
    return linear


def _scatter_rows(output, index, values):
    index = index.reshape(-1, *([1] * (values.ndim - 1))).expand(values.shape)
    return output.scatter_add_(0, index, values)


def _coalesce(input):
    if input._nnz() < 2:
        return _coo(input, input._indices().clone(), input._values().clone(), coalesced=True)
    keys, perm = _keys(input).sort(stable=True)
    first = torch.cat((keys.new_ones((1,), dtype=torch.bool), keys[1:] != keys[:-1]))
    groups = first.long().cumsum(0) - 1
    selected = perm[first]
    values = input._values().index_select(0, perm)
    combined = _scatter_rows(values.new_zeros((selected.numel(), *values.shape[1:])), groups, values)
    return _coo(input, input._indices().index_select(1, selected), combined, coalesced=True)


def _to_dense(input, dtype=None, masked_grad=None):
    if dtype is not None and dtype != input.dtype:
        raise RuntimeError("sparse COO to_dense expects the input dtype")
    values = input._values()
    if input.dtype == torch.bool:
        values = values.long()
    source = values.new_zeros(input.shape)
    flat = source.view(-1, *input.shape[input.sparse_dim():]) if source.numel() else None
    if flat is not None:
        _scatter_rows(flat, _keys(input), values)
    return source.to(input.dtype)


def _sparse_mask(input, mask):
    if input.shape != mask.shape:
        raise RuntimeError("Iris sparse_mask requires matching shapes")
    indices = mask._indices()
    if input.is_sparse:
        if input.sparse_dim() != mask.sparse_dim():
            raise RuntimeError("Iris sparse_mask requires matching sparse dimensions")
        source = input.coalesce()
        if not source._nnz():
            values = input._values().new_zeros((mask._nnz(), *input.shape[input.sparse_dim():]))
        else:
            keys = _keys(source)
            target = _keys(mask)
            positions = torch.bucketize(target, keys).clamp(max=keys.numel() - 1)
            values = source._values().index_select(0, positions)
            exists = keys.index_select(0, positions) == target
            values = torch.where(exists.reshape(-1, *([1] * (values.ndim - 1))), values, 0)
    elif input.layout != torch.strided:
        raise RuntimeError("Iris sparse_mask expects a dense or sparse COO input")
    elif not mask.sparse_dim():
        values = input.unsqueeze(0).expand(mask._nnz(), *input.shape)
    else:
        values = input[tuple(indices.unbind(0))]
    return _coo(mask, indices.clone(), values, coalesced=mask.is_coalesced())


def _add(input, other, *, alpha=1):
    if input.shape != other.shape or input.dtype != other.dtype:
        raise RuntimeError("Iris sparse addition requires matching shapes and dtypes")
    if input.is_sparse and other.is_sparse:
        if input.sparse_dim() != other.sparse_dim():
            raise RuntimeError("Iris sparse addition expects matching sparse dimensions")
        indices = torch.cat((input._indices(), other._indices()), dim=1)
        values = torch.cat((input._values(), other._values() * alpha), dim=0)
        return _coo(input, indices, values)
    if input.is_sparse:
        raise RuntimeError("Iris sparse addition expects a dense left operand for a mixed-layout result")
    return _add_inplace(input.clone(), other, alpha=alpha)


def _add_inplace(input, other, *, alpha=1):
    if input.shape != other.shape or input.dtype != other.dtype:
        raise RuntimeError("Iris sparse addition requires matching shapes and dtypes")
    if input.is_sparse:
        return input.copy_(_add(input, other, alpha=alpha))
    if not other.sparse_dim():
        input.add_(other._values().sum(dim=0), alpha=alpha)
    elif other._nnz():
        input.index_put_(tuple(other._indices().unbind(0)), other._values() * alpha, accumulate=True)
    return input


def _sparse_addmm(input, mat1, mat2, *, beta=1, alpha=1):
    if (mat1.layout != torch.sparse_coo or mat1.ndim != 2 or mat1.sparse_dim() != 2
            or mat2.layout != torch.strided or mat2.ndim != 2 or mat1.size(1) != mat2.size(0)
            or input.dtype != mat1.dtype or mat1.dtype != mat2.dtype):
        raise RuntimeError("Iris sparse addmm expects a COO matrix and matching dense operands")
    rows, cols = mat1._indices().unbind(0)
    shape = (mat1.size(0), mat2.size(1))
    result = input.new_zeros(shape) if beta == 0 else (input.expand(shape) * beta).contiguous()
    if alpha == 0:
        return result
    values = (mat2.index_select(0, cols) * mat1._values()[:, None]) * alpha
    return _scatter_rows(result, rows, values)


def _sparse_mm(left, right):
    if right.is_sparse:
        if left.is_sparse:
            raise RuntimeError("Iris mm expects a dense operand when multiplying a COO matrix")
        return _sparse_mm(right.t(), left.t()).t().contiguous()
    return _sparse_addmm(right.new_zeros(()), left, right, beta=0)


def _clone(input, *, memory_format=None):
    if memory_format not in (None, torch.preserve_format):
        raise RuntimeError("COO clone only supports preserve_format")
    return _coo(input, input._indices().clone(), input._values().clone(), coalesced=input.is_coalesced())


def _sum_layout(input, dims):
    rank = input.ndim
    if any(not -rank <= dim < rank for dim in dims):
        raise IndexError("sparse sum dimension out of range")
    reduced = sorted({dim % rank for dim in dims})
    if len(reduced) != len(dims):
        raise RuntimeError("sparse sum dimensions must be unique")
    sparse_dims = [i for i in range(input.sparse_dim()) if i not in reduced]
    dense_dims = [i - input.sparse_dim() + 1 for i in reduced if i >= input.sparse_dim()]
    return reduced, sparse_dims, dense_dims


def _sparse_sum(input, dim=None, *, dtype=None):
    if dim is None:
        return input._values().sum(dtype=dtype)
    reduced, sparse_dims, dense_dims = _sum_layout(input, dim)
    values = input._values()
    if not sparse_dims:
        return values.sum([0, *dense_dims], dtype=dtype)
    if dense_dims:
        values = values.sum(dense_dims, dtype=dtype)
    elif dtype is not None:
        values = values.to(dtype)
    indices = input._indices()[sparse_dims].contiguous()
    shape = tuple(size for i, size in enumerate(input.shape) if i not in reduced)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def _sparse_sum_backward(grad, input, dim):
    reduced, sparse_dims, dense_dims = _sum_layout(input, dim)
    source = input.coalesce()
    shape = source._values().shape
    if sparse_dims:
        indices = source.indices()[sparse_dims].contiguous()
        sizes = tuple(size for i, size in enumerate(source.shape) if i not in reduced)
        sparse_mask = torch.sparse_coo_tensor(indices, grad.new_empty(
            (source._nnz(), *sizes[len(sparse_dims):]), layout=torch.strided), sizes)
        values = _sparse_mask(grad, sparse_mask)._values()
    else:
        values = grad.unsqueeze(0)
    for axis in dense_dims:
        values = values.unsqueeze(axis)
    return _coo(source, source.indices().clone(), values.expand(shape).contiguous(), coalesced=True)


def register():
    from ._model_ops import _out

    library = torch.library.Library("aten", "IMPL", "SparsePrivateUse1")
    for name, fn in (
        ("_coalesce", _coalesce), ("_to_dense", _to_dense), ("sparse_mask", _sparse_mask),
        ("add.Tensor", _add), ("add_.Tensor", _add_inplace), ("addmm", _sparse_addmm), ("clone", _clone),
        ("mm", _sparse_mm),
        ("_sparse_sum", _sparse_sum), ("_sparse_sum.dtype", _sparse_sum),
        ("_sparse_sum.dim", _sparse_sum), ("_sparse_sum.dim_dtype", _sparse_sum),
        ("_sparse_sum.dim_out", _out(_sparse_sum, "out")), ("_sparse_sum_backward", _sparse_sum_backward),
        ("neg", lambda t: _coo(t, t._indices().clone(), -t._values(), coalesced=t.is_coalesced())),
    ):
        library.impl(name, fn)
    return library
