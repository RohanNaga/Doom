"""
Teacher-forced next-frame metrics for the new stack (velocity or epsilon backbones, stride-4 latents).

For each held-out window: L real context latents and the action, one DDIM sample, decode
through the (optionally fine-tuned) VAE, score against the raw lossless frame from the
parquet recording and against the VAE-decoded ground truth. Reports PSNR, LPIPS, latent
MSE, HUD-crop PSNR, the copy-last-frame baseline, and the VAE ceiling.

    python eval_tf.py --ckpt results/010-dit-l32/best.pt --backbone dit --latents-dir data/latents_arnold \
        --parquet-dir raw_arnold --split data/split_arnold.json --subset val --num-windows 2048 --out-dir eval/dit_val

`--wandb-run <training run>` also appends the raw PSNR, LPIPS and persistence floor to the W&B run
`<training run>-eval` as eval/<live|ema>_h<H>/..., at the step in the checkpoint's filename.
"""
import argparse
import csv
import hashlib
import io
import json
import os
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


def psnr(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


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
    model, step, objective = load_model(args, device, latent_channels, trained)
    vae = build_vae(args.vae_path, args.vae_subfolder, device, args.hf_cache, latent_channels=latent_channels,
                    scaling_factor=args.latent_scale, shift_factor=args.latent_shift)
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
    idx = draw_windows(len(ds), args.num_windows, args.seed)
    loader = DataLoader(Subset(windows, idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=2)
    raw = RawFrames(args.parquet_dir) if args.parquet_dir else None
    print(f"{args.subset}: {len(ds.episodes)} episodes, {len(ds):,} windows, evaluating {len(idx)}, step {step}, "
          f"objective {objective}, tic stride {tic_stride}, horizon {K} tic(s) = {K * tic_stride} tic(s) of game time")
    if hasattr(ds, "summary"):
        print(f"window validity: {json.dumps(ds.summary)}")

    def dec(z):
        return decode(vae, z, args.latent_scale, args.latent_shift)

    rows, t_sample, n = [], 0.0, 0
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
        pred_img, gt_img, last_img = dec(pred), dec(tgt), dec(ctx[:, -latent_channels:])
        lat_mse = ((pred.float() - tgt.float()) ** 2).flatten(1).mean(1)
        for i in range(B):
            gi = gis[i]; slot, start = ds.locate(gi)
            ep_id, map_id = ds.episodes[slot][0], ds.episodes[slot][3]
            r = dict(index=gi, episode=ep_id, map=map_id, start=start,
                     action=int(acts[i, K - 1]) if acts.ndim == 2 else -1,
                     tics_since_decision=int(phases[i, K - 1]),
                     psnr_dec=float(psnr(pred_img[i:i+1], gt_img[i:i+1])), lpips_dec=float(lp(pred_img[i:i+1] * 2 - 1, gt_img[i:i+1] * 2 - 1).flatten()),
                     copy_psnr_dec=float(psnr(last_img[i:i+1], gt_img[i:i+1])), latent_mse=float(lat_mse[i]),
                     hud_psnr_dec=float(psnr(pred_img[i:i+1, :, -HUD_ROWS:], gt_img[i:i+1, :, -HUD_ROWS:])))
            if raw is not None:
                tic = ds.target_tic(gi)
                if tic is None:
                    tic = (start + args.context_frames) * tic_stride
                scored_tic = tic + (K - 1) * tic_stride
                rf = torch.from_numpy(raw.get(ep_id, scored_tic)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
                r.update(psnr_raw=float(psnr(pred_img[i:i+1], rf)), lpips_raw=float(lp(pred_img[i:i+1] * 2 - 1, rf * 2 - 1).flatten()),
                         copy_psnr_raw=float(psnr(last_img[i:i+1], rf)), vae_psnr=float(psnr(gt_img[i:i+1], rf)),
                         vae_lpips=float(lp(gt_img[i:i+1] * 2 - 1, rf * 2 - 1).flatten()),
                         hud_psnr_raw=float(psnr(pred_img[i:i+1, :, -HUD_ROWS:], rf[:, :, -HUD_ROWS:])),
                         hud_vae_psnr=float(psnr(gt_img[i:i+1, :, -HUD_ROWS:], rf[:, :, -HUD_ROWS:])))
                # The floor, decoder-independent: the RAW last context frame against the RAW target.
                # `copy_psnr_raw` above decodes the last latent first, so it carries the decoder's own
                # reconstruction error and is NOT a persistence reference; it is kept for continuity
                # with the stride-4 rows, and `persist_*_raw` is the floor every next-tic number is
                # read against.
                lf = torch.from_numpy(raw.get(ep_id, scored_tic - K * tic_stride)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
                r.update(persist_psnr_raw=float(psnr(lf, rf)),
                         persist_lpips_raw=float(lp(lf * 2 - 1, rf * 2 - 1).flatten()),
                         persist_hud_psnr_raw=float(psnr(lf[:, :, -HUD_ROWS:], rf[:, :, -HUD_ROWS:])))
            rows.append(r)
        if b < args.save_images:
            from torchvision.utils import save_image
            save_image(torch.cat([last_img, gt_img, pred_img]).cpu(), os.path.join(args.out_dir, f"batch_{b:03d}_last_gt_pred.png"), nrow=B)
        if (b + 1) % 10 == 0:
            print(f"  {len(rows)}/{len(idx)} psnr_dec={np.mean([r['psnr_dec'] for r in rows]):.2f}", flush=True)

    keys = [k for k in rows[0] if k not in ("index", "episode", "map", "start", "action", "tics_since_decision")]

    def agg(k, subset=None):
        src = rows if subset is None else subset
        v = np.array([r[k] for r in src], dtype=np.float64); v = v[~np.isnan(v)]
        return {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v))), "n": int(len(v))} if len(v) else None
    summary = {k: agg(k) for k in keys}
    summary["per_map"] = {str(m): {k: float(np.mean([r[k] for r in rows if r["map"] == m])) for k in ("psnr_dec", "lpips_dec")}
                          for m in sorted(set(r["map"] for r in rows))}
    # is the first tic after a decision harder than the three that follow it? `tics_since_decision`
    # of the SCORED target, so bucket 0 is a decision tic and the last bucket is off the grid
    phase_keys = [k for k in ("psnr_dec", "lpips_dec", "psnr_raw", "persist_psnr_raw") if k in rows[0]]
    summary["per_tics_since_decision"] = {
        str(p): {**{k: agg(k, [r for r in rows if r["tics_since_decision"] == p]) for k in phase_keys},
                 "windows": sum(1 for r in rows if r["tics_since_decision"] == p)}
        for p in sorted({r["tics_since_decision"] for r in rows})}
    summary["sampling_frames_per_s"] = n / max(t_sample, 1e-9)
    summary["config"] = {**vars(args), "step": step, "resolved_latent_channels": latent_channels,
                         "resolved_objective": objective, "checkpoint_interface": trained,
                         "horizon_tics": K, "game_time_tics": K * tic_stride,
                         "window_validity": getattr(ds, "summary", None)}
    summary["decoder"] = decoder_record(args)
    json.dump(summary, open(os.path.join(args.out_dir, "metrics.json"), "w"), indent=1)
    with open(os.path.join(args.out_dir, "per_window.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "per_map")}, indent=1))
    log_evaluation(args, wandb_tag(args), summary, ckpt=args.ckpt, recorded_step=step, out_dir=args.out_dir)


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
    p.add_argument("--batch-size", type=int, default=16)
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
