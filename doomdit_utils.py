"""Shared helpers for loading a DoomDiT checkpoint and its VAE."""
import torch
from diffusers.models import AutoencoderKL

from models import DiT_models

VAE_NAME = "stabilityai/sd-vae-ft-mse"
LATENT_SCALE = 0.18215


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


@torch.no_grad()
def decode_latents(vae, latents):
    """(B, 4, 16|15, 20) scaled latents -> (B, 3, 120, 160) images in [0, 1]. Strips the pad row."""
    z = latents[:, :, :15, :].float() / LATENT_SCALE
    img = vae.decode(z.to(next(vae.parameters()).device)).sample
    return (img * 0.5 + 0.5).clamp(0, 1)
