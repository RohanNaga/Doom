"""
Teacher-forced next-frame metrics for the new stack (velocity or epsilon backbones, stride-4 latents).

For each held-out window: L real context latents and the action, one DDIM sample, decode
through the (optionally fine-tuned) VAE, score against the raw lossless frame from the
parquet recording and against the VAE-decoded ground truth. Reports PSNR, LPIPS, latent
MSE, HUD-crop PSNR, the copy-last-frame baseline, and the VAE ceiling. Per window it also
reports the copy-last LATENT MSE (the last real context latent against the scored target) and
`latent_mse_ratio`, the model's latent MSE over it: a persistence-normalised outcome that no
decoder touches (Sep 26 2026).

    python eval_tf.py --ckpt results/010-dit-l32/best.pt --backbone dit --latents-dir data/latents_arnold \
        --parquet-dir raw_arnold --split data/split_arnold.json --subset val --num-windows 2048 --out-dir eval/dit_val

`--wandb-run <training run>` also appends the raw PSNR, LPIPS and persistence floor to the W&B run
`<training run>-eval` as eval/<live|ema>_h<H>/..., at the step in the checkpoint's filename.

An adaptation checkpoint (`adapt_wm.py`, a LoRA adapter on a frozen snapshot) is scored like any other:
`--ckpt adapter_0000250.pt` rebuilds the frozen source it names and applies the adapter, and `--use-ema`
selects the adapter's EMA. `--windows-file <adaptation split> --windows-key held_out_windows` scores exactly
the windows that split recorded instead of a fresh `--num-windows` draw (`adapt_split.py`); without it the
draw is unchanged. The adapter records its source snapshot's absolute path on the machine it trained on;
`--source-root DIR` (DIR/basename, then DIR/<recorded parent dir name>/basename) or `--source-path FILE` finds
the same file elsewhere, the recorded SHA-256 still has to match, and `adapter_source` in the config says which
file was loaded and whether it was an override.

One pass leaves every per-window quantity a table needs in `per_window.csv` (the cohesion decision,
`.claude/analyses/evaluation-cohesion-decision-2026-09-26.md`, item 3). Every column that existed before
it keeps its name, value and position; the additions follow them:

* `copy_lpips_dec` and `copy_lpips_raw`, the decoded copy-last frame's LPIPS against the decoded and the raw
  truth, beside `lpips_dec` and `lpips_raw`;
* a pixel MSE beside every PSNR (`psnr` replaced by `mse` in the name), so a mean of PSNRs and the PSNR
  of a mean MSE can both be recomputed;
* `scene_*`, every full-frame pixel metric again on rows 0 to 207, the frame without the 32-row HUD that
  the `hud_*` columns score;
* `dup_raw` (the raw target repeats the raw last context frame) and `dup_latent` (copy-last latent MSE is
  exactly 0), counted under `duplicates` in metrics.json, which also carries `<mean>_nodup` beside every
  mean, taken over the windows neither flag marks;
* `start_tic`, `scored_tic` and `context_motion`, the mean absolute latent change between consecutive
  context frames.

`--decoder [name=]path` (repeatable) decodes the SAME predictions with a further decoder, a fine-tuned
one from `finetune_decoder.py` for instance; its pixel columns carry the suffix `_<name>` (`_tuned` when no
name is given). `--vae-path`, the stock decoder by default, is always scored and keeps the unsuffixed
names. Latent and persistence columns are decoder-free and appear once. `--save-latents` writes the
scored predictions in fp16, with the window identities, to `pred_latents.npz`, so a later decoder never
forces a resample.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from backbones import (BACKBONES, PIXART_DEFAULT, SD35_DEFAULT, UNIDIFFUSER_DEFAULT, build_model,
                       resolve_latent_channels)
from diffusion_v import VDiffusion, checkpoint_objective
from doom_data import LatentWindowDataset, load_split
from doomdit_utils import LATENT_SCALE, build_vae, denormalize_latents, load_world_model_state
from timestep_spacing import SPACINGS
from timestep_spacing import sample as sample_spaced
from wandb_log import add_eval_args, log_evaluation

HUD_ROWS = 32
# The per-window columns the frozen results were read from, in their original order. per_window.csv
# and metrics.json keep them first, so every earlier reader, positional or by name, reads the same file.
# The frozen distance-study scores came from 6a33311 and ca03bad, which had no copy-last latent columns;
# 60d99c4 added those two, so they follow the frozen set rather than sit inside it.
LEGACY_COLUMNS = ("psnr_dec", "lpips_dec", "copy_psnr_dec", "latent_mse",
                  "hud_psnr_dec", "psnr_raw", "lpips_raw", "copy_psnr_raw", "vae_psnr", "vae_lpips",
                  "hud_psnr_raw", "hud_vae_psnr", "persist_psnr_raw", "persist_lpips_raw", "persist_hud_psnr_raw",
                  "copy_latent_mse", "latent_mse_ratio")
# columns that identify or flag a window rather than score it; they are never averaged
WINDOW_COLUMNS = ("index", "episode", "map", "start", "action", "tics_since_decision",
                  "start_tic", "scored_tic", "dup_raw", "dup_latent")
DUPLICATE_FLAGS = ("dup_raw", "dup_latent")
LATENTS_FILE = "pred_latents.npz"
DEFAULT_DECODER_NAME = "tuned"
RESERVED_DECODER_NAMES = ("nodup",)       # `_nodup` already suffixes the summary's duplicate-free means


def psnr(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


def pixel_mse(a, b):
    """Per-sample mean squared error of images in [0, 1]: what `psnr` takes the log of, unclamped.

    Kept beside every PSNR so both aggregations can be recomputed from the rows: the mean of per-window
    PSNRs (a mean of logs, the reported one) and the PSNR of the mean MSE (a log of means). An exact
    repeat reads 0 here where `psnr` clamps it to 100 dB.
    """
    return ((a - b) ** 2).flatten(1).mean(1)


def hud_crop(x):
    """The bottom `HUD_ROWS` rows of (B, 3, H, W) frames: the status bar, rows 208 to 239 of 240."""
    return x[:, :, -HUD_ROWS:]


def scene_crop(x):
    """The rows above the HUD, 0 to 207 of 240. With `hud_crop` it tiles the frame exactly."""
    return x[:, :, :-HUD_ROWS]


def lpips_of(lp, a, b):
    """LPIPS of two (1, 3, H, W) images in [0, 1], mapped to [-1, 1] the way every LPIPS column is."""
    return float(lp(a * 2 - 1, b * 2 - 1).flatten())


def pair_metrics(a, b, lp, psnr_key, lpips_key=None, hud_key=None):
    """PSNR, pixel MSE and, with `lpips_key`, LPIPS of image `a` against reference `b`, keyed by column.

    The full frame goes under `psnr_key` and `lpips_key`, the HUD crop's PSNR and MSE under `hud_key` when
    one is given, and every full-frame metric again on the scene crop under a `scene_` prefix. An MSE key
    is its PSNR key with `psnr` replaced by `mse`. `a` and `b` are (1, 3, 240, 320) images in [0, 1].
    """
    out = {psnr_key: float(psnr(a, b)), psnr_key.replace("psnr", "mse"): float(pixel_mse(a, b))}
    if lpips_key:
        out[lpips_key] = lpips_of(lp, a, b)
    if hud_key:
        ha, hb = hud_crop(a), hud_crop(b)
        out[hud_key] = float(psnr(ha, hb))
        out[hud_key.replace("psnr", "mse")] = float(pixel_mse(ha, hb))
    sa, sb = scene_crop(a), scene_crop(b)
    out["scene_" + psnr_key] = float(psnr(sa, sb))
    out["scene_" + psnr_key.replace("psnr", "mse")] = float(pixel_mse(sa, sb))
    if lpips_key:
        out["scene_" + lpips_key] = lpips_of(lp, sa, sb)
    return out


def decoder_metrics(pred, gt, last, lp, raw=None):
    """Every pixel column of one window that depends on the decoder, under its unsuffixed name.

    `pred`, `gt` and `last` are one decoder's decodings of the predicted latent, the true latent and the
    last real context latent; `raw` is the lossless target frame, or None. Each is (1, 3, 240, 320) in
    [0, 1]. The decoded truth is the reference of the `*_dec` columns, so the decoder's floor cancels;
    the raw frame is the reference of the `*_raw` columns; `vae_*` is the decoder's reconstruction of the
    truth against the raw frame, the ceiling. `copy_*` is the decoded last context latent.
    """
    m = pair_metrics(pred, gt, lp, "psnr_dec", "lpips_dec", "hud_psnr_dec")
    m.update(pair_metrics(last, gt, lp, "copy_psnr_dec", "copy_lpips_dec"))
    if raw is not None:
        m.update(pair_metrics(pred, raw, lp, "psnr_raw", "lpips_raw", "hud_psnr_raw"))
        m.update(pair_metrics(last, raw, lp, "copy_psnr_raw", "copy_lpips_raw"))
        m.update(pair_metrics(gt, raw, lp, "vae_psnr", "vae_lpips", "hud_vae_psnr"))
    return m


def persistence_metrics(last_raw, raw, lp):
    """The decoder-free floor of one window: the RAW last context frame against the RAW target."""
    return pair_metrics(last_raw, raw, lp, "persist_psnr_raw", "persist_lpips_raw", "persist_hud_psnr_raw")


def context_motion(ctx, latent_channels):
    """Per window, the mean absolute latent change between consecutive frames of the real context."""
    ctx = ctx.float()
    return (ctx[:, latent_channels:] - ctx[:, :-latent_channels]).abs().flatten(1).mean(1)


def parse_decoders(specs):
    """[(name, path)] from repeated `--decoder [name=]path` values, in the order given.

    The name becomes the column suffix `_<name>`, so it is a plain identifier (letters and digits,
    starting with a letter); a value whose text before `=` is not one is a path, and takes the default
    name `tuned`. Two decoders under one name would write one set of columns twice, so that is refused.
    """
    out, seen = [], set()
    for spec in specs or []:
        name, sep, path = spec.partition("=")
        if not (sep and re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", name)):
            name, path = DEFAULT_DECODER_NAME, spec
        if not path:
            raise SystemExit(f"--decoder {spec!r} names no path")
        if name in RESERVED_DECODER_NAMES:
            raise SystemExit(f"--decoder name {name!r} is reserved (the summary's `_{name}` means use it)")
        if name in seen:
            raise SystemExit(f"--decoder {name!r} given twice; name each decoder as name=path")
        seen.add(name)
        out.append((name, path))
    return out


def window_start_tic(ds, gi, tic_stride):
    """The recorded tic of window `gi`'s first context frame (row times the spacing in old layouts)."""
    slot, start = ds.locate(gi)
    ep = ds.episodes[slot]
    tics = ep[5] if len(ep) > 5 else None
    return int(tics[start]) if tics is not None else start * tic_stride


def duplicate_rule(flags):
    """The one exclusion rule every `*_nodup` mean uses, in words, for the flags this run could set."""
    return f"a window is left out of every *_nodup mean when {' or '.join(flags)} is 1"


def copy_last_latent_mse(ctx, tgt, latent_channels):
    """Per window, the MSE of copying the last real context latent forward onto the scored target."""
    return ((ctx[:, -latent_channels:].float() - tgt.float()) ** 2).flatten(1).mean(1)


def latent_mse_ratio(model_mse, copy_mse):
    """Model latent MSE over copy-last latent MSE; NaN where copying was exact (the ratio is undefined
    there, and a clamped denominator would let one static window dominate the mean)."""
    return torch.where(copy_mse > 0, model_mse / copy_mse.clamp_min(1e-30), torch.full_like(model_mse, float("nan")))


@torch.no_grad()
def decode(vae, z, scale=LATENT_SCALE, shift=None):
    img = vae.decode(denormalize_latents(z.float(), scale, shift)).sample[:, :, :240]
    return (img * 0.5 + 0.5).clamp(0, 1)


def backbone_source(args):
    """Where build_model should pull architecture/weights from for this backbone.

    The DiT is built from local code, so it needs nothing; the diffusers backbones must be
    instantiated from the same repo they were trained from before the checkpoint is loaded.
    """
    return {"dit": None, "unet": args.sd_path, "pixart": args.pixart_path, "unidiffuser": args.unidiffuser_path,
            "sd35": args.sd35_path}[args.backbone]


def checkpoint_interface(ck, args):
    """The conditioning interface the checkpoint was trained with, checked against this invocation.

    Frame spacing, action-history length and phase conditioning all change what the model's inputs
    mean, and none of them is visible in the weights. Reading them from the checkpoint rather than
    from the command line is what makes it impossible to score a next-tic model on decision-spaced
    windows, or to feed a control-history model a single action id, and quietly get a number.
    """
    a = ck.get("args") or {}
    # `resolved_control_bits` is the width the trainer actually built with; `control_bits` is the
    # flag, which is 0 whenever the width came from the corpus. Prefer the resolved one and refuse a
    # zero: a width-0 control embedder loads without complaint and predicts from nothing.
    bits = int(a.get("resolved_control_bits") or a.get("control_bits") or 0)
    trained = {"tic_stride": int(a.get("tic_stride", 4)),
               "action_history": int(a.get("action_history", 0) or 0),
               "phase_buckets": int(a.get("phase_buckets", 0) or 0) if a.get("phase_conditioning") else 0,
               "control_bits": bits}
    # the mismatch complaint comes first: it is the more specific one, and a caller who asked for the
    # wrong spacing should hear about that rather than about a width they never mentioned
    for key, given in (("tic_stride", args.tic_stride), ("action_history", args.action_history)):
        if given is not None and int(given) != trained[key]:
            raise SystemExit(f"--{key.replace('_', '-')} {given} disagrees with the checkpoint's {trained[key]}; "
                             f"{args.ckpt} was trained with {json.dumps(trained)}")
    if trained["action_history"] and not bits:
        raise SystemExit(f"{args.ckpt} was trained with --action-history {trained['action_history']} but records "
                         "no button-vector width, so the control embedder cannot be rebuilt. It was written by "
                         "a trainer that stored the flag rather than the resolved width; re-save it from the "
                         "current trainer.")
    return trained


def load_model(args, device, latent_channels, trained):
    """(model, step, objective). The objective is the checkpoint's own, so an epsilon-trained cell
    is sampled as epsilon without the caller having to remember which it was."""
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # the action table's size depends on the training-time dropout (fast-DiT adds the null row only when it is > 0)
    dropout = ck.get("args", {}).get("action_dropout", 0.1)
    # the adaLN injection cell renames the adaln_single subtree, so the graph has to be rebuilt the way it was trained
    inject = (ck.get("args") or {}).get("action_inject") or "token"
    model = build_model(args.backbone, args.num_actions, args.context_frames, args.noise_buckets, grad_ckpt=False,
                        warm_start=backbone_source(args), cache_dir=args.hf_cache, action_dropout=dropout,
                        latent_channels=latent_channels, action_inject=inject,
                        phase_buckets=trained["phase_buckets"], action_history=trained["action_history"],
                        control_bits=trained["control_bits"])
    from lora import is_adapter_checkpoint, load_adapter_checkpoint
    if is_adapter_checkpoint(ck):
        # a LoRA adaptation (adapt_wm.py): its frozen source snapshot, then the adapter and the parts it
        # trained, live or EMA; `ck["args"]` is the source run's, so the graph above is the source's graph.
        # Which source file was loaded, and whether it was an override, rides on `args` into the config.
        args.adapter_source = load_adapter_checkpoint(model, ck, args.use_ema,
                                                      source_root=getattr(args, "source_root", "") or "",
                                                      source_path=getattr(args, "source_path", "") or "")
        return model.to(device).eval(), ck.get("step", "?"), checkpoint_objective(ck, args.objective)
    if args.use_ema and not ck.get("ema"):
        raise SystemExit(f"--use-ema requested but {args.ckpt} carries no EMA weights (use a recovery checkpoint, not best.pt)")
    load_world_model_state(model, ck, args.use_ema)
    return model.to(device).eval(), ck.get("step", "?"), checkpoint_objective(ck, args.objective)


def wandb_tag(args):
    """This read's W&B series prefix: `live_h<H>` or `ema_h<H>`, the tags the sidecar uses."""
    return f"{'ema' if args.use_ema else 'live'}_h{max(1, int(args.horizon_tics))}"


def decoder_record(args):
    """The decoder a score was decoded through: name, identity and provenance, written beside it.

    From `decoder_provenance.describe`, which reads the decoder's own `provenance.json` or the
    registry (`release/decoder_registry.json`). A decoder tuned on the unseen maps defeats an
    unseen-map claim however the dynamics model was trained, so every number has to name it.
    """
    from decoder_provenance import describe
    from doomdit_utils import VAE_NAME
    path = args.vae_path or VAE_NAME
    if args.vae_subfolder and os.path.isdir(os.path.join(path, args.vae_subfolder)):
        path = os.path.join(path, args.vae_subfolder)
    return describe(path)


def extra_decoder_record(name, path):
    """`decoder_record` for a `--decoder` decoder, with the suffix its columns carry."""
    from decoder_provenance import describe
    return {**describe(path), "path": path, "suffix": f"_{name}"}


class HorizonOne(torch.utils.data.Dataset):
    """A (context, target, action) dataset presented as the one-step case of the horizon contract.

    Lets the stride-4 path and the per-tic path go through one scoring loop: at K = 1 the loop runs
    a single sample and scores it against `targets[:, 0]`, which is exactly what the old loop did.
    """

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ctx, tgt, act = self.ds[i][:3]
        act = act.reshape(1, *act.shape) if act.ndim else act.reshape(1)
        return ctx, tgt.unsqueeze(0), act, torch.zeros(1, dtype=torch.long)


def draw_windows(n_total, num_windows, seed):
    """The sorted dataset indices a run scores: `num_windows` of `n_total`, drawn without replacement.

    One function so the evaluation launcher can name the exact window manifest a score was computed
    on (`score_identity.py windows`) and key its cache by it.
    """
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_total, size=min(num_windows, n_total), replace=False))


def window_seed(purpose, *parts):
    """A 63-bit seed from a purpose tag and a tuple of integers, by hashing rather than arithmetic.

    An arithmetic key collides. `seed*1_000_003 + ep*7919 + start*97 + step` gives 87,206 for both
    (ep 11, start 0, step 97) and (ep 11, start 1, step 0), so two different windows would be
    sampled from the same noise and a paired comparison would silently compare a window with
    itself. Hashing the packed tuple cannot alias for any values we use, and the purpose tag keeps
    the initial noise and the sampler's stochastic noise on separate streams.
    """
    h = hashlib.blake2b(str(purpose).encode() + b"|" + b"|".join(str(int(p)).encode() for p in parts),
                        digest_size=8)
    return int.from_bytes(h.digest(), "big") >> 1        # torch seeds must be non-negative


def window_noise(shape, keys, purpose="init"):
    """Per-window noise, one generator per window, keyed by `keys[i]` (an int or a tuple).

    A single global seed does not give paired samples once two runs consume different numbers of
    random values -- a different step count, a guidance branch, a different batch size -- so each
    window's noise comes from its own generator. Comparisons across checkpoints, step counts and
    samplers are then paired window by window.
    """
    out = []
    for k in keys:
        parts = k if isinstance(k, (tuple, list)) else (k,)
        g = torch.Generator().manual_seed(window_seed(purpose, *parts))
        out.append(torch.randn(shape[1:], generator=g))
    return torch.stack(out)


def eta_noise_fn(shape, keys, device):
    """A per-step, per-window noise source for the sampler's stochastic term.

    With `eta > 0` the sampler adds fresh noise at every step, and `torch.randn_like` takes it from
    the global generator, so an eta > 0 comparison is unpaired however carefully the initial noise
    was keyed. This hands the sampler a callback instead, on its own purpose tag.
    """
    def fn(step):
        return window_noise(shape, [tuple(k) + (step,) for k in keys], purpose="eta").to(device)
    return fn


class RawFrames:
    """Lazy per-episode access to raw frames in the parquet recordings."""

    def __init__(self, parquet_dir):
        self.dir, self.cache = parquet_dir, {}

    def get(self, episode_id, tic):
        """The frame recorded at exactly `tic`, or a refusal.

        `searchsorted` returns an INSERTION POINT, so a tic the recording does not hold silently
        scored the next frame -- or, past the end, the last one. A raw reference that is off by a
        tic is invisible in the metric: it moves the number by a fraction of a dB while looking
        entirely healthy. The hit is therefore checked for equality.
        """
        import pyarrow.parquet as pq
        if episode_id not in self.cache:
            t = pq.read_table(os.path.join(self.dir, f"ep_{episode_id:05d}.parquet"), columns=["tic", "frame"])
            self.cache = {episode_id: (np.array(t["tic"]), t["frame"])}   # keep one episode resident
        tics, frames = self.cache[episode_id]
        i = int(np.searchsorted(tics, tic))
        if i >= len(tics) or int(tics[i]) != int(tic):
            raise KeyError(f"episode {episode_id} has no frame at tic {tic} "
                           f"(recorded tics {int(tics[0])}..{int(tics[-1])}, {len(tics)} rows); the "
                           "sidecar and the recording do not agree, so the raw reference would be "
                           "the wrong frame")
        return np.asarray(Image.open(io.BytesIO(frames[i].as_py())).convert("RGB"), dtype=np.uint8)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    latent_channels = resolve_latent_channels(args.backbone, args.latent_channels)
    ck_peek = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    trained = checkpoint_interface(ck_peek, args)
    del ck_peek
    tic_stride = trained["tic_stride"]
    K = max(1, int(args.horizon_tics))
    if K > 1 and tic_stride != 1:
        raise SystemExit("--horizon-tics > 1 rolls the model forward tic by tic, which only means "
                         "something for a next-tic model (the checkpoint's tic_stride is "
                         f"{tic_stride})")
    extra_specs = parse_decoders(args.decoders)
    model, step, objective = load_model(args, device, latent_channels, trained)
    vae = build_vae(args.vae_path, args.vae_subfolder, device, args.hf_cache, latent_channels=latent_channels,
                    scaling_factor=args.latent_scale, shift_factor=args.latent_shift)
    # a tuned decoder reads the stock encoder's latents, so it must declare the corpus's own contract
    extra_vaes = [(name, build_vae(path, "", device, args.hf_cache, latent_channels=latent_channels,
                                   scaling_factor=args.latent_scale, shift_factor=args.latent_shift))
                  for name, path in extra_specs]
    import lpips
    lp = lpips.LPIPS(net=args.lpips_net, verbose=False).to(device).eval()
    diffusion = VDiffusion(device=device, objective=objective)

    split = load_split(args.split)
    if tic_stride == 1:
        from doom_data import TicWindowDataset
        base = TicWindowDataset(args.latents_dir, split[args.subset], args.context_frames,
                                latent_channels=latent_channels, horizon=K, with_horizon=True,
                                action_history=trained["action_history"])
        ds, windows = base, base
    else:
        base = LatentWindowDataset(args.latents_dir, split[args.subset], args.context_frames,
                                   latent_channels=latent_channels)
        ds, windows = base, HorizonOne(base)
    windows_file = getattr(args, "windows_file", "")
    if windows_file:
        # the windows an adaptation split recorded, as indices of this dataset (refused if any is not one)
        from adapt_split import load_windows, windows_in_dataset
        idx = windows_in_dataset(ds, load_windows(windows_file, args.windows_key))
    else:
        idx = draw_windows(len(ds), args.num_windows, args.seed)
    loader = DataLoader(Subset(windows, idx.tolist()), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    raw = RawFrames(args.parquet_dir) if args.parquet_dir else None
    print(f"{args.subset}: {len(ds.episodes)} episodes, {len(ds):,} windows, evaluating {len(idx)}, step {step}, "
          f"objective {objective}, tic stride {tic_stride}, horizon {K} tic(s) = {K * tic_stride} tic(s) of game time")
    if hasattr(ds, "summary"):
        print(f"window validity: {json.dumps(ds.summary)}")

    def dec(z, with_vae=vae):
        return decode(with_vae, z, args.latent_scale, args.latent_shift)

    rows, saved, t_sample, n = [], [], 0.0, 0
    for b, (ctx, tgts, acts, phases) in enumerate(loader):
        ctx, tgts, acts, phases = ctx.to(device), tgts.to(device), acts.to(device), phases.to(device)
        B = ctx.shape[0]
        gis = [int(idx[b * args.batch_size + i]) for i in range(B)]
        bucket = torch.zeros(B, dtype=torch.long, device=device)
        t0 = time.time()
        # roll K tics forward from REAL context, feeding predictions back; K = 1 is the single
        # teacher-forced step every stride-4 row was scored with
        run = ctx
        for k in range(K):
            run_in = run
            keys = [(args.seed, g, k) for g in gis]
            shape = (B, latent_channels) + tuple(tgts.shape[-2:])
            if args.infer_noise > 0:
                lvl = args.infer_noise
                bucket = torch.full_like(bucket, min(int(lvl / args.train_noise_max * args.noise_buckets), args.noise_buckets - 1))
                # the context corruption is keyed on the same (seed, window, step) tuple as the
                # initial noise, on its own purpose tag. `torch.randn_like` took it from the global
                # generator, so with --infer-noise > 0 a window's context noise depended on the
                # batch size and the sampler step count and the comparison was not paired at all
                ctx_eps = window_noise(run.shape, keys, purpose="context").to(device)
                run_in = (1.0 - lvl) ** 0.5 * run + lvl ** 0.5 * ctx_eps
            act = acts[:, k]
            ph = phases[:, k] if trained["phase_buckets"] else None
            noise = window_noise(shape, keys).to(device)
            nfn = eta_noise_fn(shape, keys, device) if args.eta > 0 else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # `linear` (the default) is diffusion.ddim_sample unchanged; see timestep_spacing.py
                pred = sample_spaced(diffusion, lambda xt, t: model(xt, t, act, run_in, bucket, ph), noise.shape,
                                     steps=args.steps, spacing=args.timestep_spacing, eta=args.eta, noise=noise,
                                     device=device, noise_fn=nfn)
            if k < K - 1:
                run = torch.cat([run[:, latent_channels:], pred.float()], dim=1)
        torch.cuda.synchronize() if device == "cuda" else None
        t_sample += time.time() - t0; n += B
        tgt = tgts[:, K - 1]
        last = ctx[:, -latent_channels:]
        pred_img, gt_img, last_img = dec(pred), dec(tgt), dec(last)
        # the same predictions through every further decoder: one sample, several renderings
        extra_imgs = [(name, (dec(pred, v), dec(tgt, v), dec(last, v))) for name, v in extra_vaes]
        lat_mse = ((pred.float() - tgt.float()) ** 2).flatten(1).mean(1)
        # the decoder-free persistence reference: the last REAL context latent, K tics before the target
        copy_mse = copy_last_latent_mse(ctx, tgt, latent_channels)
        mse_ratio = latent_mse_ratio(lat_mse, copy_mse)
        motion = context_motion(ctx, latent_channels)
        if args.save_latents:
            saved.append(pred.detach().to(torch.float16).cpu())
        for i in range(B):
            gi = gis[i]; slot, start = ds.locate(gi)
            ep_id, map_id = ds.episodes[slot][0], ds.episodes[slot][3]
            r = dict(index=gi, episode=ep_id, map=map_id, start=start,
                     action=int(acts[i, K - 1]) if acts.ndim == 2 else -1,
                     tics_since_decision=int(phases[i, K - 1]))
            tic = ds.target_tic(gi)
            if tic is None:
                tic = (start + args.context_frames) * tic_stride
            scored_tic = tic + (K - 1) * tic_stride
            rf = lf = None
            if raw is not None:
                rf = torch.from_numpy(raw.get(ep_id, scored_tic)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
                # The floor, decoder-independent: the RAW last context frame against the RAW target.
                # `copy_psnr_raw` decodes the last latent first, so it carries the decoder's own
                # reconstruction error and is NOT a persistence reference; it is kept for continuity
                # with the stride-4 rows, and `persist_*_raw` is the floor every next-tic number is
                # read against.
                lf = torch.from_numpy(raw.get(ep_id, scored_tic - K * tic_stride)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
            m = {"latent_mse": float(lat_mse[i]), "copy_latent_mse": float(copy_mse[i]),
                 "latent_mse_ratio": float(mse_ratio[i])}
            m.update(decoder_metrics(pred_img[i:i+1], gt_img[i:i+1], last_img[i:i+1], lp, rf))
            if raw is not None:
                m.update(persistence_metrics(lf, rf, lp))
            for name, imgs in extra_imgs:
                m.update({f"{k}_{name}": v for k, v in decoder_metrics(*(im[i:i+1] for im in imgs), lp, rf).items()})
            r.update({k: m.pop(k) for k in LEGACY_COLUMNS if k in m})
            r.update(start_tic=window_start_tic(ds, gi, tic_stride), scored_tic=scored_tic,
                     context_motion=float(motion[i]), dup_latent=int(float(copy_mse[i]) == 0.0))
            if raw is not None:
                r["dup_raw"] = int(m["persist_mse_raw"] == 0.0)
            r.update(m)
            rows.append(r)
        if b < args.save_images:
            from torchvision.utils import save_image
            save_image(torch.cat([last_img, gt_img, pred_img]).cpu(), os.path.join(args.out_dir, f"batch_{b:03d}_last_gt_pred.png"), nrow=B)
        if (b + 1) % 10 == 0:
            print(f"  {len(rows)}/{len(idx)} psnr_dec={np.mean([r['psnr_dec'] for r in rows]):.2f}", flush=True)

    keys = [k for k in rows[0] if k not in WINDOW_COLUMNS]

    def agg(k, subset=None):
        src = rows if subset is None else subset
        v = np.array([r[k] for r in src], dtype=np.float64); v = v[~np.isnan(v)]
        return {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v))), "n": int(len(v))} if len(v) else None

    # is the first tic after a decision harder than the three that follow it? `tics_since_decision`
    # of the SCORED target, so bucket 0 is a decision tic and the last bucket is off the grid
    phase_keys = [k for k in ("psnr_dec", "lpips_dec", "psnr_raw", "persist_psnr_raw") if k in rows[0]]

    def breakdowns(src):
        """(per-map means, per-`tics_since_decision` aggregates) over the windows in `src`."""
        per_map = {str(m): {k: float(np.mean([r[k] for r in src if r["map"] == m])) for k in ("psnr_dec", "lpips_dec")}
                   for m in sorted(set(r["map"] for r in src))}
        per_phase = {str(p): {**{k: agg(k, [r for r in src if r["tics_since_decision"] == p]) for k in phase_keys},
                              "windows": sum(1 for r in src if r["tics_since_decision"] == p)}
                     for p in sorted({r["tics_since_decision"] for r in src})}
        return per_map, per_phase

    summary = {k: agg(k) for k in keys}
    summary["per_map"], summary["per_tics_since_decision"] = breakdowns(rows)
    # One exclusion rule for every column: a window whose target repeats its last context frame, in raw
    # pixels or in latents, is counted here and left out of every `_nodup` mean. The means above it still
    # average every window, as every frozen result did.
    flags = [f for f in DUPLICATE_FLAGS if f in rows[0]]
    kept = [r for r in rows if not any(r[f] for f in flags)]
    summary["duplicates"] = {"rule": duplicate_rule(flags), "windows": len(rows),
                             **{f: sum(r[f] for r in rows) for f in flags}, "excluded": len(rows) - len(kept)}
    summary.update({f"{k}_nodup": agg(k, kept) for k in keys})
    summary["per_map_nodup"], summary["per_tics_since_decision_nodup"] = breakdowns(kept)
    summary["sampling_frames_per_s"] = n / max(t_sample, 1e-9)
    summary["config"] = {**vars(args), "step": step, "resolved_latent_channels": latent_channels,
                         "resolved_objective": objective, "checkpoint_interface": trained,
                         "horizon_tics": K, "game_time_tics": K * tic_stride,
                         "window_validity": getattr(ds, "summary", None)}
    summary["decoder"] = decoder_record(args)
    summary["extra_decoders"] = {name: extra_decoder_record(name, path) for name, path in extra_specs}
    json.dump(summary, open(os.path.join(args.out_dir, "metrics.json"), "w"), indent=1)
    with open(os.path.join(args.out_dir, "per_window.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if args.save_latents:
        save_latents(os.path.join(args.out_dir, LATENTS_FILE), torch.cat(saved), rows, K)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "per_map", "per_map_nodup")}, indent=1))
    log_evaluation(args, wandb_tag(args), summary, ckpt=args.ckpt, recorded_step=step, out_dir=args.out_dir)


def save_latents(path, preds, rows, horizon):
    """The scored predictions, fp16, with the identities that find each window again, as one npz.

    `pred` is (N, C, H, W) in per_window.csv's row order: at horizon K the Kth tic's prediction, the
    latent every pixel column decoded. Beside it, per window, `index` (the dataset index), `episode`,
    `start` (the context's first row in the episode's latents), `start_tic`, `scored_tic`, `map` and
    `tics_since_decision`, and the scalar `horizon_tics`. The true and last-context latents are the
    corpus rows `start + L + K - 1` and `start + L - 1`, so a later decoder can rescore every column.
    """
    ids = ("index", "episode", "start", "start_tic", "scored_tic", "map", "tics_since_decision")
    np.savez(path, pred=preds.to(torch.float16).numpy(), horizon_tics=np.int64(horizon),
             **{k: np.array([r[k] for r in rows], dtype=np.int64) for k in ids})


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--backbone", choices=list(BACKBONES), required=True)
    p.add_argument("--latent-channels", type=int, default=0, help="0 takes the backbone's own (4 for the SD KL-f8 rows, 16 for sd35)")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--objective", choices=["auto", "v", "eps"], default="auto",
                   help="auto reads the parameterization the checkpoint was trained in (v for every pre-grid row)")
    p.add_argument("--context-frames", type=int, default=32)
    p.add_argument("--num-actions", type=int, default=29)
    p.add_argument("--noise-buckets", type=int, default=10)
    p.add_argument("--infer-noise", type=float, default=0.0, help="context noise level at inference (0 = clean)")
    p.add_argument("--train-noise-max", type=float, default=0.7)
    p.add_argument("--latents-dir", required=True)
    p.add_argument("--parquet-dir", default="", help="raw recordings for lossless-frame scoring")
    p.add_argument("--stride", type=int, default=4, help="deprecated; the frame spacing now comes from the checkpoint")
    p.add_argument("--tic-stride", type=int, choices=[1, 4], default=None,
                   help="assert the checkpoint's frame spacing in tics; omitted, it is read from the checkpoint "
                        "so a next-tic model can never be scored on decision-spaced windows by accident")
    p.add_argument("--action-history", type=int, default=None,
                   help="assert the checkpoint's executed-control history length; read from the checkpoint otherwise")
    p.add_argument("--horizon-tics", type=int, default=1,
                   help="roll the model K tics forward from REAL context, feeding its own outputs back with the "
                        "recorded controls, and score the Kth frame. K=4 on a next-tic model is the same 114 ms "
                        "of game time as one step of a stride-4 model, which is the only way the two compare")
    p.add_argument("--split", required=True)
    p.add_argument("--subset", default="val", choices=["val", "train", "unseen_map"])
    p.add_argument("--num-windows", type=int, default=2048)
    p.add_argument("--windows-file", default="",
                   help="an adaptation split (adapt_split.py): score exactly the [episode, start] windows it records "
                        "under --windows-key instead of drawing --num-windows; empty (default) draws as always")
    p.add_argument("--windows-key", default="held_out_windows",
                   choices=["held_out_windows", "legacy_held_out_windows"],
                   help="held_out_windows: the split's fixed draw from its held-out episodes (score with --split "
                        "<the adaptation split>); legacy_held_out_windows: the distance study's draw restricted to "
                        "them (score with --split <the map's distance-study split> to keep its noise keys)")
    p.add_argument("--source-root", default="",
                   help="adapter checkpoints only: a directory holding the recorded source snapshot on this machine "
                        "(DIR/basename, then DIR/<recorded parent dir name>/basename); its SHA-256 must still match")
    p.add_argument("--source-path", default="",
                   help="adapter checkpoints only: the source snapshot file on this machine; its SHA-256 must match")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader worker processes; 0 loads in this process, which the trainer's periodic reads "
                        "use so that a stopped read leaves no workers behind")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--timestep-spacing", dest="timestep_spacing", choices=SPACINGS, default="linear",
                   help="which trained timesteps the DDIM sampler visits (timestep_spacing.py). linear, the default, "
                        "is uniform in t and is what every reported number, the 4-step sweep included, used; "
                        "trailing and karras are for the few-step spacing sweep (docs/REVIEW_2026-09-22.md M1)")
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--lpips-net", default="alex")
    p.add_argument("--vae-path", default="", help="fine-tuned VAE directory or repo id; default sd-vae-ft-mse")
    p.add_argument("--vae-subfolder", default="", help="subfolder inside --vae-path (e.g. vae for a full pipeline repo)")
    p.add_argument("--latent-scale", type=float, default=LATENT_SCALE, help="scaling_factor the corpus was encoded with")
    p.add_argument("--latent-shift", type=float, default=None, help="shift_factor the corpus was encoded with (SD 3.5: 0.0609)")
    p.add_argument("--decoder", dest="decoders", action="append", default=None, metavar="[NAME=]PATH",
                   help="a further decoder (a finetune_decoder.py output, <out> or <out>/vae) that decodes the same "
                        "predictions; its pixel columns carry the suffix _NAME (_tuned by default). Repeatable. "
                        "--vae-path is always scored and keeps the unsuffixed names; the latent contract "
                        "(--latent-scale, --latent-shift) is checked against each decoder's config")
    p.add_argument("--save-latents", action="store_true",
                   help=f"write the scored predictions (fp16) and the window identities to {LATENTS_FILE} "
                        "beside metrics.json, so a later decoder can rescore without resampling")
    p.add_argument("--sd-path", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--pixart-path", default=PIXART_DEFAULT)
    p.add_argument("--unidiffuser-path", default=UNIDIFFUSER_DEFAULT)
    p.add_argument("--sd35-path", default=SD35_DEFAULT)
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--save-images", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    return add_eval_args(p)


if __name__ == "__main__":
    main(build_parser().parse_args())
