"""
Velocity-parameterized DDPM used by both DoomDiT backbones.

The network predicts v = sqrt(abar_t) * eps - sqrt(1 - abar_t) * x0 (Salimans and Ho, 2022),
the target MultiGen and most 2024+ video world models train on. Linear beta schedule with 1,000
steps, matching the April recipe and both warm starts' schedules. Sampling is DDIM on the
velocity prediction; `eta=0` is deterministic given the noise, which is what the rollout
metrics need.
"""
import math

import torch


class VDiffusion:
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)
        abar = torch.cumprod(1.0 - betas, dim=0)
        self.num_steps = num_steps
        self.sqrt_abar = abar.sqrt().float().to(device)
        self.sqrt_1m_abar = (1.0 - abar).sqrt().float().to(device)

    def to(self, device):
        self.sqrt_abar = self.sqrt_abar.to(device)
        self.sqrt_1m_abar = self.sqrt_1m_abar.to(device)
        return self

    def _coef(self, t, ndim):
        a = self.sqrt_abar[t].view(-1, *([1] * (ndim - 1)))
        s = self.sqrt_1m_abar[t].view(-1, *([1] * (ndim - 1)))
        return a, s

    def q_sample(self, x0, t, noise):
        a, s = self._coef(t, x0.ndim)
        return a * x0 + s * noise

    def v_target(self, x0, t, noise):
        a, s = self._coef(t, x0.ndim)
        return a * noise - s * x0

    def x0_from_v(self, xt, t, v):
        a, s = self._coef(t, xt.ndim)
        return a * xt - s * v

    def eps_from_v(self, xt, t, v):
        a, s = self._coef(t, xt.ndim)
        return s * xt + a * v

    def training_loss(self, model_fn, x0, noise=None):
        """MSE on v. `model_fn(x_t, t)` returns the v prediction with x0's shape."""
        b = x0.shape[0]
        t = torch.randint(0, self.num_steps, (b,), device=x0.device)
        noise = torch.randn_like(x0) if noise is None else noise
        xt = self.q_sample(x0, t, noise)
        v_pred = model_fn(xt, t)
        return torch.mean((v_pred.float() - self.v_target(x0, t, noise)) ** 2)

    @torch.no_grad()
    def ddim_sample(self, model_fn, shape, steps=50, eta=0.0, noise=None, device=None, clip=None):
        """DDIM over `steps` respaced timesteps. Returns x0 estimate at the final step."""
        device = device or self.sqrt_abar.device
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        ts = torch.linspace(self.num_steps - 1, 0, steps, device=device).round().long()
        for i, t in enumerate(ts):
            tb = t.expand(shape[0])
            v = model_fn(x, tb).float()
            x0 = self.x0_from_v(x, tb, v)
            if clip is not None:
                x0 = x0.clamp(-clip, clip)
            eps = self.eps_from_v(x, tb, v)
            if i == steps - 1:
                x = x0
                break
            t_prev = ts[i + 1].expand(shape[0])
            a_prev, s_prev = self._coef(t_prev, x.ndim)
            a_t, s_t = self._coef(tb, x.ndim)
            sigma = eta * torch.sqrt((s_prev ** 2 / s_t ** 2) * (1 - a_t ** 2 / a_prev ** 2))
            dir_xt = torch.sqrt(torch.clamp(s_prev ** 2 - sigma ** 2, min=0.0)) * eps
            x = a_prev * x0 + dir_xt + sigma * torch.randn_like(x)
        return x


def noise_augment(context, max_level=0.7, buckets=10, generator=None):
    """GameNGen-style context corruption.

    Draws one noise level per sample in [0, max_level] as a fraction of the variance-preserving
    schedule (level 1.0 would be pure noise), corrupts every context latent with that level, and
    returns the discretized bucket id (0 = clean) for conditioning. `context` is (B, L*4, H, W).
    """
    b = context.shape[0]
    level = torch.rand(b, device=context.device, generator=generator) * max_level
    bucket = torch.clamp((level / max_level * buckets).long(), max=buckets - 1)
    bucket = torch.where(level > 0, bucket, torch.zeros_like(bucket))
    a = torch.sqrt(1.0 - level).view(b, 1, 1, 1)
    s = torch.sqrt(level).view(b, 1, 1, 1)
    return a * context + s * torch.randn_like(context), bucket
