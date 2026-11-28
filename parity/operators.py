"""Focused operator expansion cases for portable FP32 eager workloads."""

import json
from pathlib import Path

import torch

from .workloads import fixture, snapshot


OPERATOR_SCOPE = json.loads(
    Path(__file__).with_name("scope-operators-v1.json").read_text(encoding="utf-8")
)


def basic_indexing(device):
    base = fixture(4, 5, 6).to(device)
    view = base[1::2, -2, 1::2]
    before = base._version
    assert view.storage_offset() == 49
    view.fill_(1.75)
    assert base._version == view._version == before + 1
    try:
        base[4]
    except IndexError:
        pass
    else:
        raise AssertionError("Invalid integer index was accepted")
    return snapshot(base=base, view=view)


def basic_indexing_backward(device):
    base = fixture(4, 5, 6).to(device).requires_grad_()
    view = base[1::2, -2, 1::2]
    view.backward(fixture(2, 3, phase=6).to(device))
    return snapshot(view=view, grad=base.grad)


def reduction_mean(device):
    base = fixture(3, 7, 5).to(device).requires_grad_()
    view = base.transpose(1, 2)
    value = view.mean((0, -1), keepdim=True)
    value.backward(fixture(1, 5, 1, phase=7).to(device))
    return snapshot(
        value=value, grad=base.grad, whole=view.mean(),
        empty=torch.empty((0, 3), device=device).mean(0),
    )


def batched_mm(device):
    left = fixture(3, 5, 4).to(device).transpose(1, 2)
    right = fixture(3, 6, 5, phase=5).to(device).transpose(1, 2)
    empty_k = torch.bmm(torch.empty(2, 3, 0, device=device), torch.empty(2, 0, 4, device=device))
    empty_batch = torch.bmm(torch.empty(0, 3, 2, device=device), torch.empty(0, 2, 4, device=device))
    batched = left.unsqueeze(1) @ right[:1].unsqueeze(0).expand(1, 2, 5, 6)
    return snapshot(
        bmm=torch.bmm(left, right), broadcast=batched, empty_k=empty_k, empty_batch=empty_batch,
        matrix_broadcast=left @ right[0],
    )


def batched_mm_backward(device):
    left = fixture(2, 1, 4, 5).to(device).requires_grad_()
    right = fixture(1, 3, 6, 5, phase=5).to(device).requires_grad_()
    result = torch.matmul(left, right.transpose(-1, -2))
    result.backward(fixture(2, 3, 4, 6, phase=9).to(device))
    return snapshot(result=result, left_grad=left.grad, right_grad=right.grad)


def transposed_gemm(device):
    outputs = {}
    for rows, inner, columns in ((35, 47, 19), (257, 129, 259)):
        bias = fixture(columns, rows + 1, phase=9).to(device)[:, 1:].transpose(0, 1)
        for layout in ("NN", "NT", "TN", "TT"):
            left_shape = (inner, rows) if layout[0] == "T" else (rows, inner)
            right_shape = (columns, inner) if layout[1] == "T" else (inner, columns)
            left = fixture(*left_shape).to(device)
            right = fixture(*right_shape, phase=5).to(device)
            if layout[0] == "T":
                left = left.t()
            if layout[1] == "T":
                right = right.t()
            outputs[f"{rows}.{layout}.mm"] = torch.mm(left, right)
            outputs[f"{rows}.{layout}.addmm"] = torch.addmm(bias, left, right, alpha=-0.25, beta=1.5)
    return snapshot(**outputs)


def tiled_batched_mm(device):
    left = fixture(1, 65, 35).to(device).expand(3, -1, -1).transpose(1, 2)
    right = fixture(7, 47, 65, phase=5).to(device)[1::2, 1::2, :].transpose(1, 2)
    return snapshot(result=torch.bmm(left, right))


def typed_storage(device):
    integers = torch.tensor([[2**50, -3, 0, 5], [4, 0, -7, 2**49]], dtype=torch.int64).to(device)
    view = integers[:, 1::2].t()
    mask = view.to(torch.bool)
    cast = view.to(torch.float32)
    exact = view.clone()
    resized = torch.empty(1, dtype=torch.int32, device=device)
    resized.fill_(17)
    resized.resize_(7)
    resized[1:].fill_(-9)
    return snapshot(integers=integers, view=view, mask=mask, cast=cast, exact=exact, resized=resized)


def native_rng(device):
    def generate():
        torch.manual_seed(73)
        a = torch.empty((2, 2000), device=device).t()
        b = torch.empty_like(a)
        c = torch.empty_like(a)
        return a.uniform_(-1, 1), b.normal_(), c.bernoulli_(0.25)

    values = generate()
    for result, repeat in zip(values, generate(), strict=True):
        torch.testing.assert_close(result.cpu(), repeat.cpu(), rtol=0, atol=0)
    assert values[0].cpu().abs().max() <= 1
    assert abs(values[1].cpu().std().item() - 1) < 0.1
    assert abs(values[2].cpu().mean().item() - 0.25) < 0.05
    return snapshot(
        **{f"rng_{index}": torch.empty_like(tensor, device="cpu").zero_()
           for index, tensor in enumerate(values)}
    )


def pointwise_math(device):
    input = (fixture(11, 7, phase=3).t() + 0.65).to(device)
    outputs = {}
    for name in ("ceil", "floor", "trunc", "abs", "sqrt", "rsqrt", "exp", "log", "tanh", "sigmoid", "reciprocal"):
        op = getattr(torch, name)
        outputs[name] = op(input)
        outputs[f"{name}_out"] = op(input, out=torch.empty(0, device=device))
        mutated = input.clone()
        getattr(mutated, f"{name}_")()
        outputs[f"{name}_inplace"] = mutated
    for name, fn in {
        "gelu": torch.nn.functional.gelu,
        "gelu_tanh": lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
        "silu": torch.nn.functional.silu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
    }.items():
        leaf = input.clone().requires_grad_()
        output = fn(leaf)
        output.sum().backward()
        outputs[f"{name}_value"] = output
        outputs[f"{name}_grad"] = leaf.grad
    outputs.update(clamp=torch.clamp(input, 0.4, 0.9), pow_scalar=input.pow(2.5),
                   pow_tensor=input.pow(input), divide=input / (input + 0.25))
    outputs.update(
        clamp_out=torch.clamp(input, 0.4, 0.9, out=torch.empty(0, device=device)),
        pow_out=torch.pow(input, 2.5, out=torch.empty(0, device=device)),
        add_out=torch.add(input, input, out=torch.empty(0, device=device)),
        pow_inplace=input.clone().pow_(2.5),
        clamp_inplace=input.clone().clamp_(0.4, 0.9),
        divide_inplace=input.clone().div_(input + 0.25),
    )
    return snapshot(**outputs)


def formatting_operators(device):
    input = torch.tensor([[1.0, float("nan"), -1.25], [float("inf"), 0.0, 1e-5]]).t().to(device)
    finite = torch.isfinite(input)
    numeric = input.masked_select(finite & input.ne(0))
    return snapshot(
        finite=finite, nan=input.isnan(), inf=input.isinf(),
        any_finite=finite.any(), all_finite=finite.all(), min=numeric.min(), max=numeric.max(),
        eq=input == 1, lt=input < 1, le=input <= 1, gt=input > 1, ge=input >= 1,
        ne=input.ne(input), selected=numeric,
        cat=torch.cat((input, input), dim=1),
    )


def advanced_indexing(device):
    input = fixture(3, 4, 5).to(device)
    first = torch.tensor([[0], [-1]], device=device)
    last = torch.tensor([[4, 1, 0]], device=device)
    mask = torch.tensor([True, False, True], device=device)
    index = torch.tensor([2, 0, 2], dtype=torch.int32, device=device)
    mutated = input.clone()
    mutated[first, :, last] = 3.5
    gathered_input = input[0].clone().requires_grad_()
    positions = torch.tensor([[0, 1, 1], [4, 2, 0], [2, 2, 2], [3, 4, 0]], device=device)
    selected = torch.gather(gathered_input, 1, positions)
    selected.sum().backward()
    return snapshot(
        indexed=input[first, :, last], masked=input[mask], nonzero=mask.nonzero(),
        mutated=mutated, picked=torch.index_select(input, 0, index),
        gather=selected, gather_grad=gathered_input.grad,
        scattered=torch.scatter(input[0], 1, positions, -1.5),
    )


def promotion_and_where(device):
    integer = torch.tensor([[2**53 + 1, -3, 5], [7, -8, 0]], dtype=torch.int64).to(device)
    short = torch.tensor([[4, 3, -2]], dtype=torch.int32).to(device)
    floating = fixture(2, 3).to(device)
    mask = integer > 3
    return snapshot(
        comparison=integer == 2**53 + 1, mask=mask,
        int_add=integer + short, int_sub=integer - short, int_mul=integer * short,
        add=floating + short, div=short / integer, where=torch.where(mask, integer, short),
    )


def dropout_training(device):
    torch.manual_seed(905)
    input = torch.ones((32, 64), device=device, requires_grad=True)
    actual = torch.nn.functional.dropout(input, p=0.25, training=True)
    actual.sum().backward()
    torch.testing.assert_close(input.grad.cpu(), actual.cpu(), rtol=0, atol=0)
    host = actual.cpu()
    assert abs(host.mean().item() - 1.0) < 0.08
    assert torch.all((host == 0) | (host == torch.tensor(4 / 3)))
    return snapshot(inference=torch.nn.functional.dropout(input, p=0.25, training=False))


def convolution_and_pooling(device):
    src = fixture(2, 4, 9, 11, phase=3).to(device).requires_grad_()
    weights = fixture(6, 2, 3, 2).to(device).requires_grad_()
    bias = fixture(6, phase=6).to(device).requires_grad_()
    convolved = torch.nn.functional.conv2d(src, weights, bias, stride=(2, 1),
                                          padding=(2, 0), dilation=(2, 1), groups=2)
    max_pooled, indices = torch.nn.functional.max_pool2d(
        convolved, (2, 2), stride=(2, 2), padding=(1, 0), dilation=(2, 1), ceil_mode=True, return_indices=True,
    )
    averaged = torch.nn.functional.avg_pool2d(convolved, (2, 3), stride=(2, 2), padding=1,
                                              ceil_mode=True, count_include_pad=False)
    custom_divisor = torch.nn.functional.avg_pool2d(convolved, 2, stride=1, divisor_override=5)
    (max_pooled.sum() + averaged.sum() + custom_divisor.sum()).backward()
    return snapshot(convolved=convolved, max_pooled=max_pooled, indices=indices, averaged=averaged,
                    custom_divisor=custom_divisor, input_grad=src.grad, weight_grad=weights.grad,
                    bias_grad=bias.grad)


def normalization_and_attention(device):
    source = fixture(3, 2, 4, 5, phase=4).transpose(0, 1).to(device).requires_grad_()
    weight = (fixture(4, 5) + 1).to(device).requires_grad_()
    bias = fixture(4, 5, phase=6).to(device).requires_grad_()
    normalized, mean, rstd = torch.native_layer_norm(source, (4, 5), weight, bias, 1e-5)
    normalized.backward(fixture(*normalized.shape).to(device))
    captures = snapshot(normalized=normalized, mean=mean, rstd=rstd, input_grad=source.grad,
                        weight_grad=weight.grad, bias_grad=bias.grad)
    mask = torch.tensor([[False, False, False], [True, False, True]], device=device)
    for name, attention_mask, causal in (("masked", mask, False), ("causal", None, True)):
        query = fixture(2, 3, 2, 5).to(device).requires_grad_()
        key = fixture(2, 3, 3, 5, phase=2).to(device).requires_grad_()
        value = fixture(2, 3, 3, 7, phase=4).to(device).requires_grad_()
        attention = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, is_causal=causal,
        )
        attention.backward(fixture(*attention.shape).to(device))
        captures.update(snapshot(**{f"{name}.out": attention, f"{name}.q_grad": query.grad,
                                    f"{name}.k_grad": key.grad, f"{name}.v_grad": value.grad}))
    return captures


def fp16_operators(device):
    left = fixture(37, 31, phase=2).to(device).half()
    right = fixture(29, 31, phase=4).to(device).half().t()
    bias = fixture(29, phase=8).to(device).half()
    result = torch.addmm(bias, left, right, alpha=0.5, beta=-0.25)
    cast = left.float().t().contiguous().half()
    entered = fixture(4, 8).to(device).half()
    return snapshot(
        matrix=result, cast=cast, sum=cast.sum(dtype=torch.float32),
        mean=cast.mean(0), relu=cast.relu(), scale=cast * 0.25, softmax=entered.softmax(-1),
    )


def bfloat16_unary(device):
    x = torch.tensor([-0., -2., 3., float("inf"), float("nan")], dtype=torch.bfloat16, device=device)
    return snapshot(abs=x.abs(), sgn=x.sgn(), neg=x.neg(), isnan=x.isnan(),
                    isfinite=x.isfinite(), isinf=x.isinf())


def complex64_unary(device):
    source = [0j, -2 + 3j, 4 - 5j, complex(float("inf"), 1), complex(1, float("nan"))]
    x = torch.tensor(source, dtype=torch.complex64, device=device)
    differentiable = x[:3].detach().requires_grad_()
    gradient = torch.full_like(differentiable, .5 + 1j)
    grad, = torch.autograd.grad(differentiable.sgn(), differentiable, gradient)
    return snapshot(abs=x.abs(), sgn=torch.view_as_real(x.sgn()), neg=torch.view_as_real(x.neg()),
                    isnan=x.isnan(), isfinite=x.isfinite(), isinf=x.isinf(), grad=torch.view_as_real(grad))


def streamed_transfers(device):
    if device.type == "cpu":
        return snapshot(value=fixture(32, 41) + 0.125)
    source = fixture(32, 41).pin_memory()
    producer = torch.Stream(device=device)
    consumer = torch.Stream(device=device)
    with producer:
        resident = source.to(device, non_blocking=True)
        ready = producer.record_event()
    with consumer:
        consumer.wait_event(ready)
        resident.record_stream(consumer)
        updated = resident + 0.125
        host = torch.empty_like(source, pin_memory=True)
        host.copy_(updated, non_blocking=True)
        done = consumer.record_event()
    done.synchronize()
    assert ready.query() and done.query()
    return snapshot(value=host)


def fused_attention_inference(device):
    q = fixture(2, 13, 3, 7, phase=2).transpose(1, 2).to(device)
    k = fixture(2, 15, 3, 7, phase=4).transpose(1, 2).to(device)
    v = fixture(2, 15, 3, 9, phase=6).transpose(1, 2).to(device)
    return snapshot(
        attention=torch.nn.functional.scaled_dot_product_attention(q, k, v),
        causal=torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=0.25),
    )


def softmax(device):
    base = (fixture(2, 7, 73) * 80 + 1000).to(device).requires_grad_()
    view = base.transpose(0, 2)
    output = torch.softmax(view, dim=0)
    output.backward(fixture(73, 7, 2, phase=7).to(device))
    return snapshot(output=output, grad=base.grad)


def softmax_edge_cases(device):
    inputs = torch.tensor([[1., float("nan")], [1., float("inf")], [1., -float("inf")]]).to(device)
    try:
        torch.softmax(inputs, dim=2)
    except IndexError:
        pass
    else:
        raise AssertionError("Invalid softmax dimension was accepted")
    return snapshot(
        nonfinite=torch.softmax(inputs, dim=-1),
        empty=torch.softmax(torch.empty((2, 0), device=device), dim=-1),
        scalar=torch.softmax(fixture().to(device), dim=0),
    )


def vision_norms_upsample_conv(device):
    x = fixture(2, 8, 6, 7, phase=1).to(device).requires_grad_()
    rm = torch.zeros(8, device=device)
    rv = torch.ones(8, device=device)
    bw = (fixture(8, phase=2) + 1.0).to(device).requires_grad_()
    bb = fixture(8, phase=3).to(device).requires_grad_()
    bn_train = torch.nn.functional.batch_norm(x, rm, rv, bw, bb, training=True, momentum=0.1, eps=1e-5)
    bn_eval = torch.nn.functional.batch_norm(bn_train, rm, rv, bw, bb, training=False, momentum=0.1, eps=1e-5)
    gn = torch.nn.functional.group_norm(bn_eval, 4, bw, bb, eps=1e-5)
    aap = torch.nn.functional.adaptive_avg_pool2d(gn, (2, 3))
    up_near = torch.nn.functional.interpolate(aap, size=(5, 6), mode="nearest")
    up_bilin = torch.nn.functional.interpolate(aap, size=(5, 6), mode="bilinear", align_corners=False)
    ct_w = fixture(8, 4, 3, 3, phase=4).to(device).requires_grad_()
    ct = torch.nn.functional.conv_transpose2d(up_bilin, ct_w, stride=2, padding=1, output_padding=1)
    x1d = fixture(2, 4, 11, phase=5).to(device).requires_grad_()
    w1d = fixture(6, 4, 3, phase=6).to(device).requires_grad_()
    c1d = torch.nn.functional.conv1d(x1d, w1d, stride=2, padding=1)
    (up_near.sum() + ct.sum() + c1d.sum()).backward()
    return snapshot(
        bn_train=bn_train, bn_eval=bn_eval, rm=rm, rv=rv, gn=gn, aap=aap,
        up_near=up_near, up_bilin=up_bilin, ct=ct, c1d=c1d,
        x_grad=x.grad, bw_grad=bw.grad, bb_grad=bb.grad, ct_w_grad=ct_w.grad,
        x1d_grad=x1d.grad, w1d_grad=w1d.grad,
    )


def llm_reductions_sort_rope(device):
    x = fixture(2, 5, 16, phase=2).to(device).requires_grad_()
    w = (fixture(16, phase=4) + 1.0).to(device).requires_grad_()
    rms = torch.nn.functional.rms_norm(x, (16,), w, 1e-6)
    rms.sum().backward()
    t = torch.tensor([[1.0, 3.0, 3.0, -2.0], [float("nan"), 2.0, float("nan"), 1.0]], device=device)
    sv, si = torch.sort(t, dim=1, descending=True, stable=True)
    tv, ti = torch.topk(fixture(2, 48, phase=3).to(device), 7, dim=-1)
    pos = torch.arange(0, 8, device=device, dtype=torch.float32)
    lin = torch.linspace(-1.0, 1.0, 8, device=device)
    bi = torch.tensor([3, 6, 12, 0], dtype=torch.int64, device=device)
    bj = torch.tensor([5, 3, 10, 7], dtype=torch.int64, device=device)
    m1 = torch.tensor([True, True, False, False], device=device)
    m2 = torch.tensor([True, False, True, False], device=device)
    return snapshot(
        rms=rms, rms_dx=x.grad, rms_dw=w.grad,
        argmax=t.argmax(1), argmin=t.argmin(1),
        sort_v=sv, sort_i=si, topk_v=tv, topk_i=ti,
        cumsum=x.detach().cumsum(-1), arange=pos, linspace=lin,
        sin=torch.sin(lin), cos=torch.cos(lin), neg=torch.neg(lin),
        clamp_min=torch.clamp_min(lin, -0.25), clamp_max=torch.clamp_max(lin, 0.25),
        bxor=bi ^ bj, bnot=~bi, land=torch.logical_and(m1, m2), lor=torch.logical_or(m1, m2),
        lnot=torch.logical_not(m1),
    )


def fused_optimizers(device):
    outputs = {}
    for name, make_opt in (
        ("sgd", lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.8, weight_decay=0.01, nesterov=True, fused=True)),
        ("adam", lambda p: torch.optim.Adam(p, lr=0.02, weight_decay=0.01, amsgrad=True, fused=True)),
        ("adamw", lambda p: torch.optim.AdamW(p, lr=0.02, weight_decay=0.05, fused=True)),
    ):
        w1 = fixture(13, 7, phase=1).to(device).requires_grad_()
        b1 = fixture(13, phase=2).to(device).requires_grad_()
        opt = make_opt([w1, b1])
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            x = fixture(5, 7, phase=step * 3).to(device)
            y = fixture(5, 13, phase=step * 4 + 1).to(device)
            loss = torch.nn.functional.mse_loss(torch.nn.functional.linear(x, w1, b1), y)
            loss.backward()
            opt.step()
        outputs.update(snapshot(**{f"{name}.w1": w1, f"{name}.b1": b1, f"{name}.loss": loss}))
    return outputs


def llm_sampling_masking_indexing(device):
    seq = torch.tensor([1.0, 2.5, 2.5, 6.0, 9.0], device=device)
    vals = torch.tensor([0.0, 2.5, 4.0, 9.0, 11.0], device=device)
    ss_l = torch.searchsorted(seq, vals, side="left")
    ss_r = torch.searchsorted(seq, vals, side="right", out_int32=True)
    base = torch.zeros((3, 4), device=device)
    mask = torch.tensor([[True, False, True, False], [False, True, False, True], [True, True, False, False]], device=device)
    ms = base.clone().masked_scatter_(mask, torch.arange(1, 7, dtype=torch.float32, device=device))
    a = torch.tensor([1, 2, 3, 4, 5, 6], device=device)
    b = torch.tensor([2, 4, 6, 8], device=device)
    ia = torch.zeros((4, 3), device=device).index_add_(
        0, torch.tensor([0, 2, 0, 3], device=device), fixture(4, 3, phase=1).to(device), alpha=1.5
    )
    t = fixture(3, 4, phase=2).to(device)
    ri = t.repeat_interleave(torch.tensor([2, 0, 1], device=device), dim=0)
    rl = torch.roll(t, shifts=(-1, 2), dims=(0, 1))
    # Multinomial & explicit device Generator deterministic invariants
    g1 = torch.Generator(device=device).manual_seed(321)
    g2 = torch.Generator(device=device).manual_seed(321)
    probs = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device)
    m1 = torch.multinomial(probs, 4, replacement=True, generator=g1)
    m2 = torch.multinomial(probs, 4, replacement=True, generator=g2)
    return snapshot(
        ss_l=ss_l, ss_r=ss_r, ms=ms, isin_f=torch.isin(a, b), isin_t=torch.isin(a, b, invert=True),
        ia=ia, ri=ri, rl=rl, m_det=m1, m_eq=torch.tensor(bool(torch.equal(m1, m2))),
    )


def modern_vision_pad_act_grid_3d(device):
    x = fixture(2, 4, 6, 6, phase=1).to(device).requires_grad_()
    cp = torch.nn.functional.pad(x, (1, 1, 2, 0), mode="constant", value=0.5)
    rp = torch.nn.functional.pad(x, (1, 2, 1, 2), mode="reflect")
    ep = torch.nn.functional.pad(x, (2, 1, 2, 1), mode="replicate")
    a_lr = torch.nn.functional.leaky_relu(x, 0.2)
    a_el = torch.nn.functional.elu(x, 1.0)
    a_hs = torch.nn.functional.hardswish(x)
    a_hg = torch.nn.functional.hardsigmoid(x)
    a_mi = torch.nn.functional.mish(x)
    a_r6 = torch.nn.functional.relu6(x + 2.0)
    pw = fixture(4, phase=2).to(device).requires_grad_()
    a_pr = torch.nn.functional.prelu(x, pw)
    grid = (fixture(2, 5, 5, 2, phase=3) * 0.9).to(device).requires_grad_()
    gs = torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    (cp.sum() + rp.sum() + ep.sum() + a_lr.sum() + a_el.sum() + a_hs.sum() + a_hg.sum() + a_mi.sum() + a_r6.sum() + a_pr.sum() + gs.sum()).backward()

    x3 = fixture(1, 2, 4, 4, 4, phase=4).to(device).requires_grad_()
    w3 = fixture(3, 2, 3, 3, 3, phase=5).to(device).requires_grad_()
    c3 = torch.nn.functional.conv3d(x3, w3, padding=1)
    mp3 = torch.nn.functional.max_pool3d(c3, 2, 2)
    ap3 = torch.nn.functional.avg_pool3d(c3, 2, 2)
    (mp3.sum() + ap3.sum()).backward()
    return snapshot(
        cp=cp, rp=rp, ep=ep, a_lr=a_lr, a_el=a_el, a_hs=a_hs, a_hg=a_hg, a_mi=a_mi, a_r6=a_r6,
        a_pr=a_pr, gs=gs, x_grad=x.grad, pw_grad=pw.grad, grid_grad=grid.grad,
        c3=c3, mp3=mp3, ap3=ap3, x3_grad=x3.grad, w3_grad=w3.grad,
    )


def math_reductions_and_large_int_sort(device):
    x = fixture(3, 4, 5, phase=1).to(device)
    y = (fixture(3, 4, 5, phase=2).abs() + 0.3).to(device)
    w = (fixture(3, 4, 5, phase=3).abs() * 0.5).to(device)
    vm_v, vm_m = torch.var_mean(x, dim=-1, correction=1)
    big_int = torch.arange(1200, 0, -1, dtype=torch.int64, device=device)
    sv, si = torch.sort(big_int, descending=False)
    return snapshot(
        var=torch.var(x, dim=(0, 2), correction=0, keepdim=True),
        std=torch.std(x, dim=1, correction=1),
        vm_v=vm_v, vm_m=vm_m,
        prod=torch.prod(y, dim=-1),
        vnorm2=torch.linalg.vector_norm(x, ord=2, dim=-1),
        vnorm1=torch.linalg.vector_norm(x, ord=1, dim=-1),
        vnorm_inf=torch.linalg.vector_norm(x, ord=float("inf"), dim=-1),
        atan2=torch.atan2(x, y), erf=torch.erf(x), log1p=torch.log1p(y), expm1=torch.expm1(x),
        fmod=torch.fmod(x, y), rem=torch.remainder(x, y),
        lerp=torch.lerp(x, y, w), addcmul=torch.addcmul(x, y, w, value=2.0), addcdiv=torch.addcdiv(x, x, y, value=1.5),
        big_sv=sv[:16], big_si=si[:16],
    )


def run_operator_case(case, device):
    return globals()[case["id"]](torch.device(device))
