"""Shared helpers for loading a DoomDiT checkpoint and its VAE."""
import math

import torch
from diffusers.models import AutoencoderKL

from models import DiT_models

VAE_NAME = "stabilityai/sd-vae-ft-mse"
LATENT_SCALE = 0.18215
LATENT_SHIFT = None            # sd-vae-ft-mse has no shift_factor; SD 3.5's autoencoder shifts by 0.0609


def build_doomdit(model_name="DiT-XL/2"):
    """Instantiate the DOOM-shaped DiT: 16x20 padded latent grid, 16 context + 4 target channels."""
    return DiT_models[model_name](input_size=(16, 20), in_channels=20, pred_channels=4, num_classes=18)


def load_doomdit(ckpt_path, use_ema=False, device="cuda", model_name="DiT-XL/2"):
    """Load a checkpoint saved by trainDoom.py. Returns (model, info).

    Defaults to the live weights: the bf16 EMA underflow bug (commit 9936b36)
    once froze the EMA at init, so the live model is the ground truth of learning.
    The model is cast to the checkpoint's storage dtype (bf16 for released weights).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if use_ema and "ema" in ckpt:
        state, source = ckpt["ema"], "ema"
    elif "model" in ckpt:
        state, source = ckpt["model"], "model"
    else:
        state, source = ckpt, "raw"
    dtype = next(v.dtype for v in state.values() if v.is_floating_point())
    model = build_doomdit(model_name)
    model.load_state_dict(state, strict=True)
    model = model.to(device).to(dtype).eval()
    info = {"weights": source, "dtype": dtype, "step": ckpt.get("step", "?"), "loss": ckpt.get("loss", "?")}
    return model, info


def load_vae(device="cpu"):
    vae = AutoencoderKL.from_pretrained(VAE_NAME).to(device).eval()
    vae.requires_grad_(False)
    return vae


def build_vae(vae_id="", subfolder="", device="cpu", cache_dir=None,
              latent_channels=None, scaling_factor=None, shift_factor=None):
    """Load an autoencoder and assert it is the latent space the caller declared.

    With no `vae_id` this is `load_vae`, i.e. sd-vae-ft-mse, unchanged. The three declared
    numbers are optional; each one given is checked against the config, so a typo in a repo id
    fails here instead of silently writing latents under the wrong normalisation.
    """
    if not vae_id:
        vae = load_vae(device)
    else:
        kw = {"subfolder": subfolder} if subfolder else {}
        vae = AutoencoderKL.from_pretrained(vae_id, cache_dir=cache_dir, **kw).to(device).eval()
        vae.requires_grad_(False)
    got = latent_contract(vae)
    want = {"latent_channels": latent_channels, "scaling_factor": scaling_factor, "shift_factor": shift_factor}
    bad = {k: (got[k], v) for k, v in want.items() if v is not None and not _same(got[k], v)}
    if bad:
        raise SystemExit(f"{vae_id or VAE_NAME} latent contract mismatch (config, requested): {bad}")
    return vae


def _same(a, b):
    return a is not None and math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)


def latent_contract(vae):
    """The three numbers that define how latents of this autoencoder are normalised."""
    cfg = vae.config
    shift = getattr(cfg, "shift_factor", None)
    return {"latent_channels": int(cfg.latent_channels), "scaling_factor": float(cfg.scaling_factor),
            "shift_factor": None if shift is None else float(shift)}


def normalize_latents(z, scale=LATENT_SCALE, shift=LATENT_SHIFT):
    """Raw posterior mean -> stored latent, the way every diffusers pipeline does it: (z - shift) * scale."""
    return (z if shift in (None, 0) else z - shift) * scale


def denormalize_latents(z, scale=LATENT_SCALE, shift=LATENT_SHIFT):
    """Stored latent -> the value `vae.decode` expects. Inverse of `normalize_latents`."""
    z = z / scale
    return z if shift in (None, 0) else z + shift


@torch.no_grad()
def decode_latents(vae, latents):
    """(B, 4, 16|15, 20) scaled latents -> (B, 3, 120, 160) images in [0, 1]. Strips the pad row."""
    z = latents[:, :, :15, :].float() / LATENT_SCALE
    img = vae.decode(z.to(next(vae.parameters()).device)).sample
    return (img * 0.5 + 0.5).clamp(0, 1)
