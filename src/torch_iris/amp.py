"""Automatic mixed precision utilities for Iris."""

import torch
from torch.amp.grad_scaler import OptState


class GradScaler(torch.amp.GradScaler):
    """Scale losses on Iris using an FP32 reciprocal and native OpenCL kernels."""

    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000, enabled=True):
        super().__init__("iris", init_scale, growth_factor, backoff_factor, growth_interval, enabled)

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        if not self.is_enabled():
            return

        scale, _ = self._check_scale_growth_tracker("unscale_")
        optimizer_state = self._per_optimizer_states[id(optimizer)]
        if optimizer_state["stage"] is OptState.UNSCALED:
            raise RuntimeError("unscale_() has already been called on this optimizer since the last update().")
        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")

        found_inf = torch.zeros((), dtype=torch.float32, device=scale.device)
        optimizer_state["found_inf_per_device"] = self._unscale_grads_(
            optimizer, scale.reciprocal(), found_inf, False
        )
        optimizer_state["stage"] = OptState.UNSCALED
