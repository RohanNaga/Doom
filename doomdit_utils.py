"""Shared helpers for loading a DoomDiT checkpoint and its VAE."""
import math
import os

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


def load_world_model_state(model, ck, use_ema=False):
    """Load a `train_wm.py` checkpoint into an already-built backbone. Returns which half was used.

    `ck["model"]` is a full state dict and loads strictly. `ck["ema"]` holds one tensor per
    parameter, so it is applied on top of the live weights and may leave buffers untouched; any
    other gap means the EMA does not belong to this model and is an error rather than a warning.
    For the 4-channel rows, whose models have no persistent buffers, the two paths are identical
    to the strict load they replace.
    """
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()}, strict=True)
    if not use_ema:
        return "model"
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in ck["ema"].items()}, strict=False)
    gap = set(missing) - {n for n, _ in model.named_buffers()}
    if gap or unexpected:
        raise SystemExit(f"EMA weights do not match this model: missing {sorted(gap)}, unexpected {sorted(unexpected)}")
    return "ema"


def load_vae(device="cpu"):
    vae = AutoencoderKL.from_pretrained(VAE_NAME).to(device).eval()
    vae.requires_grad_(False)
    return vae


WEIGHT_FILES = ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin")


def resolve_vae_dir(path):
    """Point a local path at the directory that actually holds the weights.

    A decoder tune writes `<out_dir>/vae`, so both `<out_dir>` and `<out_dir>/vae` name the
    same checkpoint and both are accepted. Repo ids are returned untouched. A local directory
    with no weight file raises here, naming itself, instead of surfacing as diffusers' report
    that it could not find `diffusion_pytorch_model.bin` in it: that is what a half-written
    save looks like, and the difference matters when a gate has to be rerun.
    """
    if not os.path.isdir(path):
        return path
    for cand in (path, os.path.join(path, "vae")):
        if any(os.path.exists(os.path.join(cand, f)) for f in WEIGHT_FILES):
            return cand
    raise SystemExit(f"no autoencoder weights ({' or '.join(WEIGHT_FILES)}) in {path}; "
                     "the fine-tune that was to write them did not finish saving")


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
        if subfolder and os.path.isdir(os.path.join(vae_id, subfolder)):
            vae_id, subfolder = os.path.join(vae_id, subfolder), ""    # a local path, not a repo id
        vae_id = resolve_vae_dir(vae_id)
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
