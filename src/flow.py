import torch
import torch.nn as nn


class LinearFlow(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        noise_scale: float = 1.0,
        t_eps: float = 1e-5,
        condition: bool = False,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.sample_shape = model.sample_shape
        self.noise_scale = noise_scale
        self.t_eps = t_eps
        self.condition = condition
        self.dropout_prob = dropout_prob

    def dropout(self, y: torch.Tensor) -> torch.Tensor:
        if self.dropout_prob <= 0.0 or not self.training:
            return y
        if not 0.0 <= self.dropout_prob <= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1], got {self.dropout_prob}")

        keep_mask = torch.rand(y.shape[0], device=y.device) >= self.dropout_prob
        keep_mask = keep_mask.reshape((y.shape[0],) + (1,) * (y.ndim - 1))
        return y * keep_mask.to(dtype=y.dtype)

    def compute_loss(
        self,
        x1: torch.Tensor,
        y: torch.Tensor | None = None,
        cond_steps: int | None = None,
    ) -> torch.Tensor:
        if self.condition and y is None:
            raise ValueError("Conditioning is enabled, but no condition `y` is provided.")
        if not self.condition and y is not None:
            raise ValueError("Conditioning is disabled, but a condition `y` is provided.")
        
        x0 = torch.randn_like(x1) * self.noise_scale
        t = torch.rand(x1.size(0), device=x1.device, dtype=x1.dtype)
        t = t.clip(self.t_eps, 1.0 - self.t_eps)
        t_expand = t.reshape((-1,) + (x1.ndim - 1) * (1,))

        if cond_steps is not None:
            if not (1 <= cond_steps < x1.shape[1]):
                raise ValueError(
                    f"cond_steps must be in [1, T], got cond_steps={cond_steps}, T={x1.shape[1]}"
                )
            x0[:, :cond_steps] = x1[:, :cond_steps]
        xt = (1 - t_expand) * x0 + t_expand * x1
        if cond_steps is not None:
            xt[:, :cond_steps] = x1[:, :cond_steps]

        if self.condition:
            assert y is not None
            y_cond = self.dropout(y)
            pred = self.model(xt, t, y_cond)
        else:
            pred = self.model(xt, t)

        err = (pred - x1) ** 2
        if cond_steps is None:
            return err.mean()
        return err[:, cond_steps:, :].mean()
    

    @torch.inference_mode()
    def sample(
        self,
        num_steps: int,
        cond_prefix: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if self.condition:
            raise TypeError("use sample_cfg(...) for class-conditional models")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        dtype = next(self.model.parameters()).dtype
        seq_len = int(self.sample_shape[0])
        n, _, d = cond_prefix.shape
        k = cond_prefix.shape[1]
        cond_prefix = cond_prefix.to(device=device, dtype=dtype)
        x_t = torch.randn(n, seq_len, d, device=device, dtype=dtype) * self.noise_scale
        x_t[:, :k] = cond_prefix

        dt = (1.0 - self.t_eps) / num_steps
        for step in range(num_steps):
            t_value = step * dt
            t = torch.full((n,), t_value, device=device, dtype=dtype)
            pred_x1 = self.model(x_t, t)
            denom = (1.0 - t).reshape((-1,) + (x_t.ndim - 1) * (1,)).clamp_min(self.t_eps)
            velocity = (pred_x1 - x_t) / denom
            x_t = x_t + dt * velocity
            x_t[:, :k] = cond_prefix
        return x_t


    @torch.inference_mode()
    def sample_cfg(
        self,
        num_steps: int,
        cond_prefix: torch.Tensor,
        device: torch.device,
        y: torch.Tensor,
        cond_scale: float = 1.0,
    ) -> torch.Tensor:
        if not self.condition:
            raise TypeError("use sample(...) for unconditional models")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        dtype = next(self.model.parameters()).dtype
        seq_len = int(self.sample_shape[0])
        n, _, d = cond_prefix.shape
        k = cond_prefix.shape[1]
        cond_prefix = cond_prefix.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)
        if y.shape[0] != n:
            raise ValueError(f"y batch size must match cond_prefix, got {y.shape[0]} and {n}")

        x_t = torch.randn(n, seq_len, d, device=device, dtype=dtype) * self.noise_scale
        x_t[:, :k] = cond_prefix
        y_null = torch.zeros_like(y)

        dt = (1.0 - self.t_eps) / num_steps
        for step in range(num_steps):
            t_value = step * dt
            t = torch.full((n,), t_value, device=device, dtype=dtype)
            pred_x1_uncond = self.model(x_t, t, y_null)
            pred_x1_cond = self.model(x_t, t, y)
            denom = (1.0 - t).reshape((-1,) + (x_t.ndim - 1) * (1,)).clamp_min(self.t_eps)
            v_uncond = (pred_x1_uncond - x_t) / denom
            v_cond = (pred_x1_cond - x_t) / denom
            velocity = v_uncond + cond_scale * (v_cond - v_uncond)
            x_t = x_t + dt * velocity
            x_t[:, :k] = cond_prefix
        return x_t


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, buffer in model_buffers.items():
        ema_buffers[name].copy_(buffer)
