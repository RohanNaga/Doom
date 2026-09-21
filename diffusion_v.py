"""
DDPM used by every DoomDiT backbone, in either the velocity or the epsilon parameterization.

By default the network predicts v = sqrt(abar_t) * eps - sqrt(1 - abar_t) * x0 (Salimans and Ho,
2022), the target MultiGen and most 2024+ video world models train on, and the one every finished
row was trained with. `objective="eps"` switches the loss target and the sampler's conversion to
the classic noise prediction, which is one cell of the knob grid; the betas, the timestep
distribution and the sampler itself are untouched, so the cell isolates the parameterization.
Linear beta schedule with 1,000 steps, matching the April recipe and every warm start's schedule.
Sampling is DDIM on the prediction; `eta=0` is deterministic given the noise, which is what the
rollout metrics need.
"""
import torch

OBJECTIVES = ("v", "eps")


def checkpoint_objective(ck, override="auto"):
    """The parameterization a checkpoint's prediction has to be read in.

    `train_wm.py` records it in `ck["args"]["objective"]`. Checkpoints written before the flag
    existed carry no key and are velocity models, which is what the default returns, so the five
    finished rows keep scoring exactly as they did. Anything other than "auto" in `override` wins,
    for the case where a checkpoint's record is wrong and has to be forced by hand.
    """
    if override and override != "auto":
        if override not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES} or 'auto', got {override}")
        return override
    return (ck.get("args") or {}).get("objective") or "v"


class VDiffusion:
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02, device="cpu", objective="v"):
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective}")
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)
        abar = torch.cumprod(1.0 - betas, dim=0)
        self.num_steps = num_steps
        self.objective = objective
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

    def x0_from_eps(self, xt, t, eps):
        a, s = self._coef(t, xt.ndim)
        return (xt - s * eps) / a

    def target(self, x0, t, noise):
        """What the network is trained to output at (x_t, t): the velocity, or the noise itself."""
        return noise if self.objective == "eps" else self.v_target(x0, t, noise)

    def x0_from_pred(self, xt, t, pred):
        """x0 estimate from whatever this parameterization's network predicts."""
        return self.x0_from_eps(xt, t, pred) if self.objective == "eps" else self.x0_from_v(xt, t, pred)

    def eps_from_pred(self, xt, t, pred):
        """Noise estimate from whatever this parameterization's network predicts."""
        return pred if self.objective == "eps" else self.eps_from_v(xt, t, pred)

    def training_loss(self, model_fn, x0, noise=None, t=None, per_sample=False):
        """MSE on this parameterization's target. `model_fn(x_t, t)` returns the prediction with x0's shape.
        `per_sample=True` returns one loss per batch element (for timestep-binned validation)."""
        b = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.num_steps, (b,), device=x0.device)
        noise = torch.randn_like(x0) if noise is None else noise
        xt = self.q_sample(x0, t, noise)
        pred = model_fn(xt, t)
        err = (pred.float() - self.target(x0, t, noise)) ** 2
        return err.flatten(1).mean(1) if per_sample else err.mean()

    @torch.no_grad()
    def ddim_sample(self, model_fn, shape, steps=50, eta=0.0, noise=None, device=None, clip=None,
                    noise_fn=None):
        """DDIM over `steps` respaced timesteps. Returns x0 estimate at the final step.

        `noise_fn(step)` supplies the stochastic term's noise when `eta > 0`; without it that noise
        comes from the global generator, so an eta > 0 comparison is unpaired however carefully the
        initial `noise` was keyed. `None` is the old behaviour exactly.
        """
        device = device or self.sqrt_abar.device
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        ts = torch.linspace(self.num_steps - 1, 0, steps, device=device).round().long()
        for i, t in enumerate(ts):
            tb = t.expand(shape[0])
            pred = model_fn(x, tb).float()
            x0 = self.x0_from_pred(x, tb, pred)
            if clip is not None:
                x0 = x0.clamp(-clip, clip)
            eps = self.eps_from_pred(x, tb, pred)
            if i == steps - 1:
                x = x0
                break
            t_prev = ts[i + 1].expand(shape[0])
            a_prev, s_prev = self._coef(t_prev, x.ndim)
            a_t, s_t = self._coef(tb, x.ndim)
            sigma = eta * torch.sqrt((s_prev ** 2 / s_t ** 2) * (1 - a_t ** 2 / a_prev ** 2))
            dir_xt = torch.sqrt(torch.clamp(s_prev ** 2 - sigma ** 2, min=0.0)) * eps
            extra = torch.randn_like(x) if noise_fn is None else noise_fn(i).to(x.dtype)
            x = a_prev * x0 + dir_xt + sigma * extra
        return x


def noise_augment(context, max_level=0.7, buckets=10, generator=None, level=None, eps=None):
    """GameNGen-style context corruption.

    Draws one noise level per sample in [0, max_level] as a fraction of the variance-preserving
    schedule (level 1.0 would be pure noise), corrupts every context latent with that level, and
    returns the discretized bucket id (0 = clean) for conditioning. `context` is (B, L*4, H, W).
    """
    b = context.shape[0]
    if level is None:
        level = torch.rand(b, device=context.device, generator=generator) * max_level
    bucket = torch.clamp((level / max_level * buckets).long(), max=buckets - 1)
    bucket = torch.where(level > 0, bucket, torch.zeros_like(bucket))
    a = torch.sqrt(1.0 - level).view(b, 1, 1, 1)
    s = torch.sqrt(level).view(b, 1, 1, 1)
    if eps is None:
        eps = torch.randn(context.shape, device=context.device, generator=generator, dtype=context.dtype)
    return a * context + s * eps, bucket
