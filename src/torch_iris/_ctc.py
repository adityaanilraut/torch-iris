"""Connectionist temporal classification with log-domain Iris dynamic programming."""

import torch

from ._model_ops import _out


def _metadata(lengths):
    if isinstance(lengths, torch.Tensor):
        if lengths.ndim != 1 or lengths.dtype not in (torch.int32, torch.int64):
            raise RuntimeError("CTC lengths must be one-dimensional integer arrays")
        return lengths.cpu().tolist()
    return lengths


def _states(prob, targets, input_lengths, target_lengths, blank):
    if prob.ndim != 3 or prob.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise RuntimeError("Iris CTC expects floating log_probs with shape (time, batch, classes)")
    time, batch, classes = prob.shape
    if (len(input_lengths) != batch or len(target_lengths) != batch or not 0 <= blank < classes
            or any(n < 0 or n > time for n in input_lengths) or any(n < 0 for n in target_lengths)):
        raise RuntimeError("CTC expects valid lengths for each batch entry and a valid blank class")
    if targets.dtype not in (torch.int32, torch.int64) or targets.ndim not in (1, 2):
        raise RuntimeError("CTC expects integer padded or concatenated targets")
    max_target = max(target_lengths, default=0)
    if (targets.ndim == 1 and targets.numel() != sum(target_lengths)
            or targets.ndim == 2 and (targets.size(0) != batch or targets.size(1) < max_target)):
        raise RuntimeError("CTC target shape does not match target_lengths")
    lengths = torch.tensor(target_lengths, dtype=torch.long, device=prob.device)[:, None]
    times = torch.tensor(input_lengths, dtype=torch.long, device=prob.device)[:, None]
    positions = torch.arange(2 * max_target + 1, device=prob.device)
    labels = torch.full((batch, positions.numel()), blank, device=prob.device, dtype=torch.long)
    if max_target:
        padded = targets.to(device=prob.device, dtype=torch.long)
        if targets.ndim == 1:
            starts = lengths.cumsum(0) - lengths
            lookup = starts + torch.arange(max_target, device=prob.device)
            padded = padded.index_select(0, lookup.clamp(max=targets.numel() - 1).flatten()).view(batch, max_target)
        labels[:, 1::2].copy_(torch.where(positions[1::2] // 2 < lengths, padded[:, :max_target], blank))
    valid = positions <= 2 * lengths
    accept = (positions == 2 * lengths) | (positions == 2 * lengths - 1)
    previous = torch.cat((labels[:, :2], labels[:, :-2]), dim=-1)
    skip = (positions >= 2) & (labels != blank) & (labels != previous)
    return labels, times, valid, accept, skip


def _shift(value, width, reverse=False):
    if width >= value.size(-1):
        return torch.full_like(value, -torch.inf)
    fill = torch.full_like(value[..., :width], -torch.inf)
    parts = (value[..., width:], fill) if reverse else (fill, value[..., :-width])
    return torch.cat(parts, -1)


def _ctc(prob, targets, input_lengths, target_lengths, blank=0, zero_infinity=False):
    input_lengths, target_lengths = _metadata(input_lengths), _metadata(target_lengths)
    labels, times, valid, accept, skip = _states(prob, targets, input_lengths, target_lengths, blank)
    p = prob.float()
    start = torch.arange(labels.size(1), device=prob.device) == 0
    state = torch.where(start, p.new_zeros(labels.shape), -torch.inf)
    saved = []
    for step, emission in enumerate(p.unbind(0)):
        forward = torch.logaddexp(state, _shift(state, 1))
        forward = torch.logaddexp(forward, torch.where(skip, _shift(state, 2), -torch.inf))
        forward = torch.where(valid, forward + emission.gather(-1, labels), -torch.inf)
        state = torch.where(step < times, forward, state)
        saved.append(state)
    loss = -torch.where(accept, state, -torch.inf).logsumexp(-1)
    alpha = torch.stack(saved, dim=1) if saved else p.new_empty((labels.size(0), 0, labels.size(1)))
    return loss.to(prob.dtype), alpha


def _ctc_backward(grad, prob, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha,
                  blank, zero_infinity=False):
    input_lengths, target_lengths = _metadata(input_lengths), _metadata(target_lengths)
    labels, times, valid, accept, skip = _states(prob, targets, input_lengths, target_lengths, blank)
    p = prob.float()
    tail = torch.where(accept, p.new_zeros(labels.shape), -torch.inf)
    infinite = neg_log_likelihood.float().isinf()[:, None]
    nll = torch.where(infinite, 0, neg_log_likelihood.float()[:, None])
    result = []
    for step in reversed(range(p.size(0))):
        active = step < times
        expected = (log_alpha[:, step] + tail + nll).exp()
        posterior = torch.zeros_like(p[step]).scatter_add_(-1, labels, expected)
        delta = (p[step].exp() - posterior) * grad.float()[:, None]
        delta = torch.where(infinite, 0 if zero_infinity else torch.nan, delta)
        result.append(torch.where(active, delta, 0))
        next_state = tail + p[step].gather(-1, labels)
        shifted = _shift(torch.where(skip, next_state, -torch.inf), 2, reverse=True)
        backward = torch.logaddexp(next_state, _shift(next_state, 1, reverse=True))
        backward = torch.logaddexp(backward, shifted)
        tail = torch.where(active & valid, backward, tail)
    return torch.stack(list(reversed(result))).to(prob.dtype) if result else torch.empty_like(prob)


def register(library):
    for name in ("_ctc_loss", "_ctc_loss.Tensor"):
        library.impl(name, _ctc)
    library.impl("_ctc_loss.out", _out(_ctc, "out0", "out1"))
    library.impl("_ctc_loss.Tensor_out", _out(_ctc, "out0", "out1"))
    library.impl("_ctc_loss_backward", _ctc_backward)
    library.impl("_ctc_loss_backward.Tensor", _ctc_backward)
    library.impl("_ctc_loss_backward.out", _out(_ctc_backward, "out"))
