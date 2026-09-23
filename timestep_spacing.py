"""Which timesteps a few-step DDIM sampler visits, for evaluation only.

`VDiffusion.ddim_sample` (diffusion_v.py) spaces its steps uniformly in t over the trained linear
betas 1e-4..0.02. At 4 steps that is t = [999, 666, 333, 0], at SNR [4e-5, 0.011, 0.47, 9999]: two
of the four network calls are at essentially pure noise, and the whole range from SNR 0.47 to clean
is one jump (docs/REVIEW_2026-09-22.md M1). Every number reported so far, including the 4-step
column of the step sweep, used that `linear` spacing. This module lets the evaluators try others
without retraining and without touching the trained schedule:

  linear    `torch.linspace(T-1, 0, steps).round()`, exactly what `ddim_sample` does (the default,
            and the default path calls `ddim_sample` itself, so it stays bit-identical)
  trailing  the "trailing" spacing of Lin et al. 2024 (diffusers' `timestep_spacing="trailing"`):
            t_i = round(T - i*T/steps) - 1, which starts at T-1 and never samples t = 0
  karras    sigma spaced as in Karras et al. 2022 (rho = 7) between the schedule's largest and
            smallest sigma = sqrt((1 - abar) / abar), each mapped to the nearest trained t; denser
            near clean, sparser near noise

Only the visited indices change; the betas, the model and the update rule are the trained ones.
The sampler loop in `ddim_sample_at` is `VDiffusion.ddim_sample`'s own, over a given index list;
`test_timestep_spacing.py` checks that it reproduces `ddim_sample` bit for bit on the linear list.
"""
import torch

SPACINGS = ("linear", "trailing", "karras")
KARRAS_RHO = 7.0


def timesteps(spacing, steps, num_steps=1000, sqrt_abar=None):
    """The descending trained-timestep indices a `steps`-step sampler visits under `spacing`."""
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if spacing == "linear":
        return torch.linspace(num_steps - 1, 0, steps).round().long()
    if spacing == "trailing":
        return (torch.arange(num_steps, 0, -num_steps / steps).round() - 1).long()[:steps]
    if spacing == "karras":
        if sqrt_abar is None:
            raise ValueError("karras spacing needs the schedule's sqrt(alpha-bar)")
        abar = sqrt_abar.double().cpu() ** 2
        sigma = ((1 - abar) / abar).sqrt()
        smax, smin = float(sigma[-1]), float(sigma[0])
        ramp = torch.linspace(0, 1, steps, dtype=torch.float64)
        target = (smax ** (1 / KARRAS_RHO) + ramp * (smin ** (1 / KARRAS_RHO) - smax ** (1 / KARRAS_RHO))) ** KARRAS_RHO
        ts = (sigma.log()[None, :] - target.log()[:, None]).abs().argmin(dim=1)
        # two targets can land on one trained index near t = 0; keep the list strictly decreasing
        out = [int(ts[0])]
        for t in ts[1:].tolist():
            out.append(max(0, min(int(t), out[-1] - 1)))
        return torch.tensor(out, dtype=torch.long)
    raise ValueError(f"unknown timestep spacing {spacing!r}; one of {SPACINGS}")


@torch.no_grad()
def ddim_sample_at(diffusion, model_fn, shape, ts, eta=0.0, noise=None, device=None, clip=None, noise_fn=None):
    """`VDiffusion.ddim_sample`'s loop over an explicit descending index list `ts`."""
    device = device or diffusion.sqrt_abar.device
    x = torch.randn(shape, device=device) if noise is None else noise.to(device)
    ts = ts.to(device)
    steps = len(ts)
    for i, t in enumerate(ts):
        tb = t.expand(shape[0])
        pred = model_fn(x, tb).float()
        x0 = diffusion.x0_from_pred(x, tb, pred)
        if clip is not None:
            x0 = x0.clamp(-clip, clip)
        eps = diffusion.eps_from_pred(x, tb, pred)
        if i == steps - 1:
            x = x0
            break
        t_prev = ts[i + 1].expand(shape[0])
        a_prev, s_prev = diffusion._coef(t_prev, x.ndim)
        a_t, s_t = diffusion._coef(tb, x.ndim)
        sigma = eta * torch.sqrt((s_prev ** 2 / s_t ** 2) * (1 - a_t ** 2 / a_prev ** 2))
        dir_xt = torch.sqrt(torch.clamp(s_prev ** 2 - sigma ** 2, min=0.0)) * eps
        extra = torch.randn_like(x) if noise_fn is None else noise_fn(i).to(x.dtype)
        x = a_prev * x0 + dir_xt + sigma * extra
    return x


def sample(diffusion, model_fn, shape, steps=50, spacing="linear", **kw):
    """DDIM with the chosen spacing. `linear` IS `diffusion.ddim_sample`, unchanged."""
    if spacing == "linear":
        return diffusion.ddim_sample(model_fn, shape, steps=steps, **kw)
    ts = timesteps(spacing, steps, diffusion.num_steps, diffusion.sqrt_abar)
    return ddim_sample_at(diffusion, model_fn, shape, ts, **kw)
