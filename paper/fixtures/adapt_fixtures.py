"""Shared pieces for the adaptation tests: a tiny next-tic PixArt source snapshot, a one-map corpus, stubs.

The source snapshot is written the way `train_wm.py` writes `snap_*.pt`: bf16 weights under `model`, bf16
EMA tensors by parameter name under `ema`, and the run's args. Its EMA differs from its live weights on
purpose, so a test can tell which of the two a loader used.
"""
import os
import sys
import types

import numpy as np
import torch

from pertic_fixtures import held_actions, write_pertic_episode

CTX = 4                    # context tics of the tiny models
BITS = 19                  # the executed control is the engine's 19 buttons
MAP = 17
EPISODES = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
TINY_PIXART = dict(num_attention_heads=2, attention_head_dim=8, in_channels=4, out_channels=8,
                   num_layers=2, caption_channels=32, sample_size=64, patch_size=2,
                   cross_attention_dim=16, use_additional_conditions=False, norm_num_groups=2)
PIXART_REPO = "PixArt-alpha/PixArt-XL-2-512x512"
# the fixture map has 10 episodes, so its split states the budget: 6 adapt, 4 held out, a step curve on the
# first 4 of the adapt list, and 3 windows from each held-out episode (12 scored windows)
SPLIT_FLAGS = ["--adapt-episodes", "6", "--held-out-episodes", "4", "--ladder", "1,2,4,6", "--step-curve-k", "4",
               "--windows-per-episode", "3"]


def serve_tiny_pixart(monkeypatch):
    """Serve the tiny transformer wherever PixArt would pull the 611M checkpoint."""
    from diffusers import PixArtTransformer2DModel as P
    monkeypatch.setattr(P, "from_pretrained", classmethod(lambda cls, *a, **k: cls(**TINY_PIXART)))
    monkeypatch.setattr(P, "load_config", classmethod(lambda cls, *a, **k: dict(TINY_PIXART)))


def stub_lpips(monkeypatch):
    """A stand-in `lpips` package where the real one is absent, so the scoring path runs rather than skips."""
    import importlib.util
    if importlib.util.find_spec("lpips") is not None:
        return

    class _Stub(torch.nn.Module):
        def __init__(self, net="alex", verbose=False):
            super().__init__()

        def forward(self, a, b):
            return (a - b).abs().flatten(1).mean(1).view(-1, 1, 1, 1)

    module = types.ModuleType("lpips")
    module.LPIPS = _Stub
    monkeypatch.setitem(sys.modules, "lpips", module)


def tiny_vae(path, channels=4):
    """A 4-block AutoencoderKL small enough for the CPU, saved where `build_vae` can read it."""
    from diffusers.models import AutoencoderKL
    AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                  up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(4, 4, 4, 4),
                  layers_per_block=1, norm_num_groups=2, sample_size=256,
                  latent_channels=channels, scaling_factor=0.18215).eval().save_pretrained(str(path))
    return str(path)


def source_args(**over):
    """The args a next-tic PixArt run records, at the tiny sizes."""
    a = {"backbone": "pixart", "latent_channels": 4, "resolved_latent_channels": 4, "context_frames": CTX,
         "num_actions": 3, "noise_buckets": 4, "noise_aug_max": 0.7, "objective": "v", "tic_stride": 1,
         "action_history": CTX, "control_bits": BITS, "resolved_control_bits": BITS, "phase_conditioning": False,
         "phase_buckets": 5, "action_inject": "token", "action_dropout": 0.0, "warm_start": PIXART_REPO,
         "hf_cache": None, "seed": 0, "lr": 5e-5, "clip": 1.0}
    a.update(over)
    return a


def tiny_source_model(seed=0):
    import backbones
    from diffusers import PixArtTransformer2DModel
    torch.manual_seed(seed)
    return backbones.PixArtWorldModel(num_actions=3, context_frames=CTX, noise_buckets=4, action_dropout=0.0,
                                      grad_ckpt=False, action_history=CTX, control_bits=BITS,
                                      transformer=PixArtTransformer2DModel(**TINY_PIXART))


def write_source_snapshot(path, seed=0, ema_shift=0.05, step=200000, args=None):
    """A `snap_*.pt`-shaped source: bf16 live weights, bf16 EMA weights that differ from them, args.

    `args` overrides entries of the recorded run args (`source_args`), e.g. its clip or spike guard.
    """
    m = tiny_source_model(seed)
    g = torch.Generator().manual_seed(seed + 1)
    ema = {n: (p.detach() + ema_shift * torch.randn(p.shape, generator=g)).to(torch.bfloat16)
           for n, p in m.named_parameters()}
    ck = {"model": {k: v.detach().to(torch.bfloat16) for k, v in m.state_dict().items()}, "ema": ema,
          "step": step, "val_loss": 0.1, "args": source_args(**(args or {}))}
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    torch.save(ck, str(path))
    return str(path)


def buttons(rng, n):
    """Random executed controls with plenty of single-turn rows, so turning windows exist."""
    out = []
    for _ in range(n):
        b = ["0"] * BITS
        r = rng.rand()
        if r < 0.35:
            b[2] = "1"                      # TURN_LEFT
        elif r < 0.7:
            b[3] = "1"                      # TURN_RIGHT
        if rng.rand() < 0.5:
            b[0] = "1"                      # forward, allowed in a turning window
        out.append("".join(b))
    return out


def map_corpus(path, episodes=EPISODES, T=24, seed=0, map_id=MAP):
    """Per-tic episodes of one map with textured latents (row t filled with noise seeded by (ep, t)).

    `T` must be a multiple of 4 (the recorded action repeat); action ids stay inside the tiny models' 3.
    """
    rng = np.random.RandomState(seed)
    for ep in episodes:
        write_pertic_episode(str(path), ep, held_actions(np.arange(T // 4) % 3),
                             buttons=buttons(rng, T), map_ids=np.full(T, map_id))
        lat = np.stack([np.random.RandomState(ep * 1000 + t).randn(4, 32, 40) for t in range(T)])
        np.save(os.path.join(str(path), f"ep_{ep:05d}_latents.npy"), lat.astype(np.float16))
    return str(path)
