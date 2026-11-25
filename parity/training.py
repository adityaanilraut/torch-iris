"""First-order autograd and SGD workloads for the shared FP32 MLP."""

import json
from itertools import product
from pathlib import Path

import torch

from .workloads import fixture, make_mlp, snapshot


TRAINING_SCOPE = json.loads(
    Path(__file__).with_name("scope-training-v1.json").read_text(encoding="utf-8")
)


def training_loop(device, host_transfers):
    with torch.no_grad():
        model = make_mlp(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.125, momentum=0.75, foreach=False)
    params = dict(model.named_parameters())
    history = {}
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        assert all(param.grad is None for param in params.values())
        host_input = fixture(11, 7, phase=step * 5)
        if host_transfers:
            host_input.requires_grad_()
            inputs = host_input.to(device)
            inputs.retain_grad()
        else:
            inputs = host_input.to(device).requires_grad_()
        target = fixture(11, 5, phase=step * 3 + 7).to(device)
        prediction = model(inputs)
        loss = loss_fn(prediction, target)
        assert prediction.device == loss.device == device
        backward_loss = loss.cpu() if host_transfers else loss
        if host_transfers and device.type != "cpu":
            assert backward_loss.device.type == "cpu"
            assert backward_loss.grad_fn is not loss.grad_fn
            assert not inputs.is_leaf
        backward_loss.backward()

        values = {"prediction": prediction, "loss": backward_loss, "input_grad": inputs.grad}
        assert inputs.grad.device == device
        if host_transfers:
            assert host_input.grad.device.type == "cpu"
            torch.testing.assert_close(host_input.grad, inputs.grad.cpu(), rtol=0, atol=0)
            values["host_input_grad"] = host_input.grad
        for name, param in params.items():
            assert param.grad.device == device
            values[f"{name}.grad"] = param.grad
        history.update({f"{step}.{name}": captured for name, captured in snapshot(**values).items()})

        prior_versions = {name: param._version for name, param in params.items()}
        optimizer.step()
        values = {}
        for name, param in params.items():
            momentum = optimizer.state[param]["momentum_buffer"]
            assert param._version == prior_versions[name] + 1
            assert param.device == momentum.device == device
            values[f"{name}.param"] = param
            values[f"{name}.momentum"] = momentum
        history.update({f"{step}.{name}": captured for name, captured in snapshot(**values).items()})
    return history


def mlp_training(device):
    return training_loop(device, host_transfers=False)


def training_host_transfer(device):
    return training_loop(device, host_transfers=True)


def mse_reductions(device):
    values = {}
    for reduction in ("none", "sum", "mean"):
        inputs = fixture(3, 5).to(device).requires_grad_()
        target = fixture(3, 5, phase=4).to(device).requires_grad_()
        loss = torch.nn.MSELoss(reduction=reduction)(inputs, target)
        grad = None if reduction != "none" else fixture(3, 5, phase=6).to(device)
        loss.backward(grad)
        values[f"{reduction}.loss"] = loss
        values[f"{reduction}.input_grad"] = inputs.grad
        values[f"{reduction}.target_grad"] = target.grad
    empty = torch.empty((0, 2), device=device, requires_grad=True)
    loss = torch.nn.MSELoss()(empty, torch.empty_like(empty))
    loss.backward()
    return snapshot(**values, empty_mean=loss, empty_grad=empty.grad)


def adam_training(device):
    outputs = {}
    for optimizer_class, foreach in product((torch.optim.Adam, torch.optim.AdamW), (False, True)):
        with torch.no_grad():
            model = make_mlp(device)
        optimizer = optimizer_class(
            model.parameters(), lr=0.025, weight_decay=0.01, amsgrad=True, foreach=foreach,
        )
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            inputs = fixture(11, 7, phase=step * 5).to(device)
            target = fixture(11, 5, phase=step * 3 + 7).to(device)
            loss = torch.nn.functional.mse_loss(model(inputs), target)
            loss.backward()
            optimizer.step()
            prefix = f"{optimizer_class.__name__}.foreach={foreach}.{step}"
            outputs.update(snapshot(**{f"{prefix}.loss": loss}))
            for name, param in model.named_parameters():
                state = optimizer.state[param]
                values = {key: state[key] for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")}
                outputs.update(snapshot(**{f"{prefix}.{name}.{key}": v for key, v in values.items()},
                                        **{f"{prefix}.{name}.param": param}))
    return outputs


def embedding_cross_entropy(device):
    indices = torch.tensor([[1, 2, 1], [3, 0, 2]], dtype=torch.int32, device=device)
    targets = torch.tensor([1, 2, -100, 0, 0, 1], dtype=torch.int64, device=device)
    outputs = {}
    for reduction in ("none", "mean", "sum"):
        weights = fixture(6, 3).to(device).requires_grad_()
        embed = torch.nn.functional.embedding(indices, weights, padding_idx=0, scale_grad_by_freq=True)
        output = embed.reshape(6, 3)
        loss = torch.nn.functional.cross_entropy(
            output, targets, weight=torch.tensor([1.5, 0.25, 0.75], device=device), reduction=reduction,
        )
        (loss.sum() if reduction == "none" else loss).backward()
        outputs.update(snapshot(**{f"{reduction}.output": output, f"{reduction}.loss": loss,
                                   f"{reduction}.grad": weights.grad}))
    return outputs


def _optimizer_snapshot(model, input, targets):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, foreach=False)
    prediction = model(input)
    loss = torch.nn.functional.cross_entropy(prediction.reshape(-1, prediction.size(-1)), targets)
    loss.backward()
    values = snapshot(prediction=prediction, loss=loss)
    for name, param in model.named_parameters():
        values.update(snapshot(**{f"{name}.grad": param.grad}))
    optimizer.step()
    for name, param in model.named_parameters():
        values.update(snapshot(**{f"{name}.param": param}))
    return values


def _initialize(model, device):
    with torch.no_grad():
        for index, param in enumerate(model.parameters()):
            param.copy_(fixture(*param.shape, phase=5 * index))
    return model.to(device)


def cnn_training(device):
    model = _initialize(torch.nn.Sequential(
        torch.nn.Conv2d(2, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 4 * 4, 5),
    ), device)
    return _optimizer_snapshot(model, fixture(3, 2, 8, 8).to(device),
                               torch.tensor([0, 3, 1], device=device))


def transformer_training(device):
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(17, 8)
            self.norm = torch.nn.LayerNorm(8)
            self.attention = torch.nn.MultiheadAttention(8, 2, batch_first=True)
            self.mlp = torch.nn.Sequential(torch.nn.Linear(8, 13), torch.nn.GELU(), torch.nn.Linear(13, 17))

        def forward(self, tokens):
            value = self.embedding(tokens)
            normalized = self.norm(value)
            attention, _ = self.attention(normalized, normalized, normalized, need_weights=False)
            return self.mlp(value + attention)

    tokens = torch.tensor([[1, 2, 5, 3], [2, 7, 5, 1]], device=device)
    targets = torch.tensor([2, 5, 3, 7, 7, 5, 1, 4], device=device)
    return _optimizer_snapshot(_initialize(Block(), device), tokens, targets)


def autocast_training(device):
    with torch.no_grad():
        model = make_mlp(device)
    inputs = fixture(11, 7, phase=3).to(device).requires_grad_()
    target = fixture(11, 5, phase=9).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.125, foreach=False)
    with torch.autocast(device.type, dtype=torch.float16):
        prediction = model(inputs)
        loss = torch.nn.functional.mse_loss(prediction, target)
    assert prediction.dtype == torch.float16 and loss.dtype == torch.float32
    loss.backward()
    optimizer.step()
    return snapshot(prediction=prediction, loss=loss, input_grad=inputs.grad,
                    **dict(model.named_parameters()))


def resnet_block_training(device):
    class ResNetBlockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(4, 8, 3, padding=1, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(8)
            self.conv2 = torch.nn.Conv2d(8, 8, 3, padding=1, bias=False)
            self.bn2 = torch.nn.BatchNorm2d(8)
            self.skip = torch.nn.Conv2d(4, 8, 1, bias=False)
            self.skip_bn = torch.nn.BatchNorm2d(8)
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(8, 5)

        def forward(self, x):
            identity = self.skip_bn(self.skip(x))
            out = torch.relu(self.bn1(self.conv1(x)))
            out = torch.relu(self.bn2(self.conv2(out)) + identity)
            out = self.pool(out).flatten(1)
            return self.fc(out)

    outputs = {}
    for opt_name, make_opt in (
        ("sgd", lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, weight_decay=0.01, fused=True)),
        ("adamw", lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.01, fused=True)),
    ):
        model = _initialize(ResNetBlockModel(), device)
        opt = make_opt(model.parameters())
        x = fixture(4, 4, 8, 8, phase=2).to(device)
        targets = torch.tensor([0, 3, 1, 4], device=device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        opt.step()
        outputs.update(snapshot(**{
            f"{opt_name}.logits": logits,
            f"{opt_name}.loss": loss,
            f"{opt_name}.bn1_rm": model.bn1.running_mean,
            f"{opt_name}.bn1_rv": model.bn1.running_var,
        }))
        for name, param in model.named_parameters():
            outputs.update(snapshot(**{f"{opt_name}.{name}.param": param, f"{opt_name}.{name}.grad": param.grad}))
    return outputs


def llama_block_training_and_generation(device):
    class LlamaMini(torch.nn.Module):
        def __init__(self, vocab=32, dim=128, n_heads=2, head_dim=64, hidden=192):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = head_dim
            self.embed = torch.nn.Embedding(vocab, dim)
            self.norm1_w = torch.nn.Parameter(torch.ones(dim))
            self.qkv = torch.nn.Linear(dim, 3 * dim, bias=False)
            self.wo = torch.nn.Linear(dim, dim, bias=False)
            self.norm2_w = torch.nn.Parameter(torch.ones(dim))
            self.w_gate = torch.nn.Linear(dim, hidden, bias=False)
            self.w_up = torch.nn.Linear(dim, hidden, bias=False)
            self.w_down = torch.nn.Linear(hidden, dim, bias=False)
            self.final_norm_w = torch.nn.Parameter(torch.ones(dim))
            self.lm_head = torch.nn.Linear(dim, vocab, bias=False)

        def apply_rope(self, x):
            B, H, L, D = x.shape
            half = D // 2
            pos = torch.arange(L, device=x.device, dtype=torch.float32).unsqueeze(1)
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
            angles = pos * inv_freq.unsqueeze(0)
            cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
            sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

        def forward(self, tokens):
            B, L = tokens.shape
            h = self.embed(tokens)
            hn = torch.nn.functional.rms_norm(h, (h.size(-1),), self.norm1_w, 1e-6)
            qkv = self.qkv(hn).view(B, L, 3, self.n_heads, self.head_dim)
            q = self.apply_rope(qkv[:, :, 0].transpose(1, 2))
            k = self.apply_rope(qkv[:, :, 1].transpose(1, 2))
            v = qkv[:, :, 2].transpose(1, 2)
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            h = h + self.wo(attn.transpose(1, 2).contiguous().view(B, L, -1))
            hn2 = torch.nn.functional.rms_norm(h, (h.size(-1),), self.norm2_w, 1e-6)
            mlp = self.w_down(torch.nn.functional.silu(self.w_gate(hn2)) * self.w_up(hn2))
            h = h + mlp
            h = torch.nn.functional.rms_norm(h, (h.size(-1),), self.final_norm_w, 1e-6)
            return self.lm_head(h)

    model = _initialize(LlamaMini(), device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01, fused=True)
    tokens = torch.tensor([[1, 5, 12, 3, 7, 2], [4, 8, 2, 15, 9, 6]], device=device)
    targets = torch.tensor([5, 12, 3, 7, 2, 9, 8, 2, 15, 9, 6, 1], device=device)
    opt.zero_grad(set_to_none=True)
    logits = model(tokens)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets)
    loss.backward()
    opt.step()

    # Autoregressive top-k + greedy argmax generation (3 steps)
    with torch.no_grad():
        seq = tokens[:1, :4].clone()
        topk_vals_list, topk_idx_list = [], []
        for _ in range(3):
            next_logits = model(seq)[:, -1, :]
            tv, ti = torch.topk(next_logits, 5, dim=-1)
            topk_vals_list.append(tv)
            topk_idx_list.append(ti)
            next_tok = next_logits.argmax(dim=-1, keepdim=True)
            seq = torch.cat((seq, next_tok), dim=1)

    outputs = snapshot(
        logits=logits, loss=loss, generated=seq,
        topk_vals=torch.cat(topk_vals_list, dim=0),
        topk_idxs=torch.cat(topk_idx_list, dim=0),
    )
    for name, param in model.named_parameters():
        outputs.update(snapshot(**{f"{name}.param": param, f"{name}.grad": param.grad}))
    return outputs


def flash_sdpa_backward_and_kv_decode(device):
    q = fixture(1, 4, 24, 64, phase=1).to(device).requires_grad_()
    k = fixture(1, 2, 24, 64, phase=2).to(device).requires_grad_()
    v = fixture(1, 2, 24, 64, phase=3).to(device).requires_grad_()
    go = fixture(1, 4, 24, 64, phase=4).to(device)
    out_gqa = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    out_gqa.backward(go)

    q128 = fixture(1, 2, 16, 128, phase=5).to(device).requires_grad_()
    k128 = fixture(1, 2, 16, 128, phase=6).to(device).requires_grad_()
    v128 = fixture(1, 2, 16, 128, phase=7).to(device).requires_grad_()
    out128 = torch.nn.functional.scaled_dot_product_attention(q128, k128, v128, is_causal=False)
    out128.sum().backward()

    # KV-cached decode (Lq=1, Lk=96)
    with torch.no_grad():
        qd = fixture(1, 4, 1, 64, phase=8).to(device)
        kd = fixture(1, 4, 96, 64, phase=9).to(device)
        vd = fixture(1, 4, 96, 64, phase=10).to(device)
        dec = torch.nn.functional.scaled_dot_product_attention(qd, kd, vd, is_causal=False)
    return snapshot(
        out_gqa=out_gqa, q_grad=q.grad, k_grad=k.grad, v_grad=v.grad,
        out128=out128, q128_grad=q128.grad, k128_grad=k128.grad, v128_grad=v128.grad,
        dec=dec,
    )


def unet_modern_vision_training(device):
    class ModernUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = torch.nn.Conv2d(4, 8, 3, padding=0)
            self.gn1 = torch.nn.GroupNorm(4, 8)
            self.c2 = torch.nn.Conv2d(8, 8, 3, stride=2, padding=1)
            self.up = torch.nn.ConvTranspose2d(8, 8, 2, stride=2)
            self.out = torch.nn.Conv2d(16, 4, 1)

        def forward(self, x):
            xp = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="reflect")
            e1 = torch.nn.functional.mish(self.gn1(self.c1(xp)))
            m = torch.nn.functional.leaky_relu(self.c2(e1), 0.1)
            u = torch.nn.functional.relu6(self.up(m), inplace=True)
            return self.out(torch.cat((u, e1), dim=1))

    model = _initialize(ModernUNet(), device)
    x = fixture(2, 4, 8, 8, phase=1).to(device).requires_grad_()
    grid = (fixture(2, 8, 8, 2, phase=2) * 0.8).to(device).requires_grad_()
    y = model(x)
    warped = torch.nn.functional.grid_sample(y, grid, mode="bilinear", padding_mode="border", align_corners=False)
    loss = warped.pow(2).mean()
    loss.backward()
    outputs = snapshot(y=y, warped=warped, loss=loss, x_grad=x.grad, grid_grad=grid.grad)
    for name, param in model.named_parameters():
        outputs.update(snapshot(**{f"{name}.grad": param.grad}))
    return outputs


def run_training_case(case, device):
    with torch.enable_grad():
        return globals()[case["id"]](torch.device(device))
