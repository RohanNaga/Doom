"""Does the predicted frame turn the way the control says? The directional check of review 7.2.

The smoke probe (`smoke_probe.py`) proves only that flipping the newest control changes the output.
A model can pass it and still turn the wrong way, or learn persistence and barely move. This check
asks the directional question on real validation windows, and measures motion beside it.

**Turning windows.** From the val split, windows whose NEWEST executed control (row `r-1`, the one
applied from the last context frame into the target; `doom_data.TicWindowDataset`) holds exactly one
of TURN_LEFT (bit 2) and TURN_RIGHT (bit 3) and neither strafe bit (MOVE_LEFT 4, MOVE_RIGHT 5). A
strafe translates the view, which mixes a second horizontal motion into the shift being measured.
Forward and backward motion are allowed: their image motion expands about the centre, and its
horizontal part cancels over the symmetric crop the estimator reads. `--windows` of each direction
are drawn with `--seed`. The windows also need the next four tics inside one life, because the
motion ratio below rolls four tics forward on the same windows.

**The swap.** Each window is predicted twice from the same context and the same per-window noise
(`eval_tf.window_noise`, keyed `(seed, window, 0)` exactly as `eval_tf.py` keys its first step):
once with the recorded controls and once with TURN_LEFT and TURN_RIGHT swapped in the newest
control token only. Every other tic of the 32-tic history is unchanged. The sampler is DDIM with
eta 0, so the two samples differ only through the control.

**The shift.** Both predictions, the stored target and the last context frame are decoded through
the given decoder (`eval_tf.decode`), and `horizontal_shift` measures how far each frame's content
moved sideways relative to the decoded last context frame: the argmax of a normalised
cross-correlation over integer shifts on a band of middle rows in a central crop, refined to sub-pixel
precision by a parabola (the justification is in its docstring). Sign convention: shift s means
frame(x) matches last(x - s). A LEFT turn swings the view left, so the scene slides RIGHT and s > 0;
a right turn gives s < 0 (`EXPECTED_SIGN`). One tic of turning is 1.76 to 7 degrees, about 5 to 20
pixels at the 90 degree field of view, so shifts under `SIGN_DEADBAND` pixels count as no motion.

**What is reported**, per direction and pooled, with every per-window value in the JSON:

  recorded     mean and median shift under the recorded control, and the fraction whose sign is
               the one the control implies
  swapped      the same under the swapped control (whose implied sign is the opposite)
  correct_frac the "directional correctness": the fraction of windows whose shift changes sign
               when the control is swapped (both shifts past the deadband, opposite signs)
  reference    the same shift measured on the ground-truth next frame (decoded stored latent), and
               `ref_frac`, the fraction that moves the way the control says. This tests the
               measurement (estimator, crop and sign convention) on real frames, not the model; if
               `ref_frac` is low, the model's numbers cannot be read until that is fixed. With
               `--parquet-dir`, `reference_raw` repeats it on the raw recorded frames, free of the
               decoder.
  persistence  the copy-last-frame baseline: shift 0 on every window by construction, hence
               correct_frac 0 and no window moving the right way

**Motion ratio** (review 7.2 and M4). On the same turning windows, a closed-loop rollout of
`MOTION_HORIZON` = 4 tics from real context with the recorded controls, each prediction fed back as
the newest context frame: the same contract as `eval_tf.py --horizon-tics 4` (its dataset, its
control indexing, its noise keys `(seed, window, k)`; the loop is inline in `eval_tf.main`, so it is
repeated here). Motion is `mean |x_h - x_(h-1)|` over pixels and channels in [0, 1], with x_0 the
decoded last context frame, as `motion_audit.py` defines it; the ratio is prediction over ground
truth, a ratio of means over all windows and the four steps, with the per-step ratios beside it.
The h = 1 entry is the teacher-forced one-step ratio. A model that learned persistence scores 0.

No W&B logging. The summary line a steward greps is

    DIRECTIONAL_CHECK <run> <step> correct_frac=<x> ref_frac=<y> weights=<live|ema>

where <run> is `--run`, else the checkpoint's directory name; name `--out` per weights too.

Spiderman, one free card each (check `nvidia-smi` first), 10k snapshot, live weights; add
`--use-ema` and change `--out` for the EMA read. The 4-channel U-Net row, stock decoder:

    cd /sata2/data/rnagabhi/doom/repo && D=/sata2/data/rnagabhi/doom
    CUDA_VISIBLE_DEVICES=<card> $HOME/miniconda3/envs/doom/bin/python directional_check.py \\
        --ckpt $D/results_spiderman/040-unet-nexttic/snap_0010000.pt --backbone unet --latent-channels 4 \\
        --sd-path CompVis/stable-diffusion-v1-4 --hf-cache $D/hf/hub \\
        --latents-dir $D/latents_arnold_dense_pertic_eval/val \\
        --split $D/latents_arnold_dense_pertic_eval/split_val.json --subset val \\
        --parquet-dir $D/raw_arnold_dense/arenas --windows 128 --steps 10 --seed 0 --device cuda:0 \\
        --out $D/results_spiderman/040-unet-nexttic/directional_0010000_live.json

The SD 3.5 row, with the decoder flags of `scripts/cluster/gates.sh` readback_cmd:

    CUDA_VISIBLE_DEVICES=<card> $HOME/wanenc/bin/python directional_check.py \\
        --ckpt $D/results_spiderman/042-sd35-nexttic/snap_0010000.pt --backbone sd35 --latent-channels 16 \\
        --sd35-path stabilityai/stable-diffusion-3.5-medium --hf-cache $D/hf/hub \\
        --vae-path stabilityai/stable-diffusion-3.5-medium --vae-subfolder vae \\
        --latent-scale 1.5305 --latent-shift 0.0609 \\
        --latents-dir $D/latents_arnold_dense_pertic_eval_sd35/val \\
        --split $D/latents_arnold_dense_pertic_eval_sd35/split_val.json --subset val \\
        --parquet-dir $D/raw_arnold_dense/arenas --windows 128 --steps 10 --seed 0 --device cuda:0 \\
        --out $D/results_spiderman/042-sd35-nexttic/directional_0010000_live.json

Cost per window: five sampler runs of `--steps` model calls (recorded, swapped, three more rollout
steps) and ten decodes, at `--batch-size` windows per call.

A requested-action row (`ACTION_HISTORY=0`, e.g. 044-unet-nexttic-reqaction) is refused: it has no
control token to swap, and mapping an action id to its mirrored id is a separate design.
"""
import argparse
import functools
import json
import os

import numpy as np
import torch

from backbones import BACKBONES, PIXART_DEFAULT, SD35_DEFAULT, UNIDIFFUSER_DEFAULT, resolve_latent_channels
from check_action_alignment import TURN_LEFT, TURN_RIGHT
from diffusion_v import VDiffusion
from doom_data import TicWindowDataset, load_split
from doomdit_utils import LATENT_SCALE, build_vae
from eval_tf import RawFrames, checkpoint_interface, decode, decoder_record, load_model, window_noise

MOVE_LEFT, MOVE_RIGHT = 4, 5          # the strafe bits of the executed control (docs/cards/arnold/buttons.json)
EXPECTED_SIGN = {"left": 1, "right": -1}   # a left turn slides the scene right: positive shift
DIRECTIONS = tuple(EXPECTED_SIGN)
MOTION_HORIZON = 4                    # tics of the motion-ratio rollout, the same game time as one stride-4 step
MAX_SHIFT = 32                        # pixels searched each way; one tic of turning is about 5 to 20
BAND_ROWS = (48, 120)                 # middle rows of the 240-row frame: above the weapon, far from the HUD
SIGN_DEADBAND = 1.0                   # pixels; below the estimator's resolution and far below the smallest turn
FLAT_VARIANCE = 1e-6                  # luminance variance under which a band has no texture to match
EXACT_MATCH = 1.0 - 1e-9              # a correlation this close to 1 is an exact integer shift


def turn_direction(controls):
    """Per control row: +1 for a left turn, -1 for a right turn, 0 for anything else.

    A turning row holds exactly one of TURN_LEFT and TURN_RIGHT and neither strafe bit. The value is
    also the sign of the image shift the turn should produce (`EXPECTED_SIGN`).
    """
    c = np.asarray(controls) > 0.5
    turn = c[..., TURN_LEFT].astype(np.int64) - c[..., TURN_RIGHT].astype(np.int64)
    strafe = c[..., MOVE_LEFT] | c[..., MOVE_RIGHT]
    return np.where(strafe, 0, turn)


def turning_windows(ds, per_direction, seed=0):
    """({direction: sorted dataset indices}, {direction: windows available}) of turning windows.

    A window is classified by its newest control, row `start + L - 1`, the one that carries the last
    context frame into the target. `per_direction` of each direction are drawn without replacement;
    a corpus with fewer is refused rather than scored on a smaller sample.
    """
    if not ds.action_history:
        raise ValueError("turning windows are read from the executed-control history; build the dataset "
                         "with action_history=L")
    pools = {d: [] for d in DIRECTIONS}
    for slot, ep in enumerate(ds.episodes):
        starts, controls = ep[4], ep[7]
        turn = turn_direction(controls[starts + ds.L - 1])
        base = int(ds.offsets[slot])
        for d in DIRECTIONS:
            pools[d].append(base + np.flatnonzero(turn == EXPECTED_SIGN[d]))
    pools = {d: np.concatenate(p).astype(np.int64) if p else np.zeros(0, np.int64) for d, p in pools.items()}
    available = {d: int(len(p)) for d, p in pools.items()}
    short = [d for d in DIRECTIONS if available[d] < per_direction]
    if short:
        raise ValueError(f"too few turning windows: {available} available, {per_direction} per direction "
                         f"asked for; lower --windows or widen the split")
    rng = np.random.RandomState(seed)
    return {d: np.sort(rng.choice(pools[d], size=per_direction, replace=False)) for d in DIRECTIONS}, available


def swap_turns(act):
    """A copy of a (B, L, bits) control history with TURN_LEFT and TURN_RIGHT swapped in the newest row."""
    out = act.clone()
    out[:, -1, TURN_LEFT] = act[:, -1, TURN_RIGHT]
    out[:, -1, TURN_RIGHT] = act[:, -1, TURN_LEFT]
    return out


def luminance(img):
    """(B, 3, H, W) RGB in [0, 1] -> (B, H, W) Rec. 601 luma."""
    w = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32, device=img.device).view(1, 3, 1, 1)
    return (img.float() * w).sum(1)


@torch.no_grad()
def horizontal_shift(a, b, max_shift=MAX_SHIFT, rows=BAND_ROWS):
    """(shift, peak) per image pair: b's content sits `shift` pixels right of a's, b(x) = a(x - shift).

    The method is a normalised cross-correlation search over integer shifts on the luma of a band of
    middle rows, with a parabolic sub-pixel refinement. Why each part:

    * A turn is a yaw rotation, and on a 90 degree perspective view a yaw rotation moves the whole
      image sideways by nearly the same amount (exactly, for the sky). One horizontal translation is
      therefore the right model, and a global search for it is enough.
    * The band of middle rows (`BAND_ROWS`) sits around the horizon, above the weapon sprite and far
      from the HUD. Both of those are fixed to the screen, so including them pulls the peak towards
      zero whatever the view does.
    * The reference is a central crop of `a` (`max_shift` pixels in from each side), and each
      candidate shift compares it with the equally sized window of `b`. Every candidate therefore
      compares pixels that exist in both frames. FFT phase correlation would instead assume the frame
      wraps around, so the scene entering at one edge and the static HUD would add a spurious peak at
      zero unless carefully windowed and masked.
    * Normalising each window (zero mean, unit norm) makes the score insensitive to the brightness
      and contrast a few-step diffusion sample loses, and the peak value is a match confidence in
      [-1, 1].
    * Forward motion expands the scene about the centre; its horizontal part is antisymmetric about
      the centre, so over a symmetric crop it broadens the peak rather than moving it.
    * The parabola through the peak and its neighbours gives sub-pixel precision; a peak that is an
      exact match (correlation 1) is an integer shift and is left as it is.

    A band without texture (a flat wall filling the view) has no shift: both outputs are NaN.
    """
    ya = luminance(a)[:, rows[0]:rows[1]].double()
    yb = luminance(b)[:, rows[0]:rows[1]].double()
    W, m = ya.shape[-1], int(max_shift)
    ref = ya[..., m:W - m]
    ref = ref - ref.mean((1, 2), keepdim=True)
    n = ref[0].numel()
    ref_norm = ref.square().sum((1, 2)).sqrt()
    textured = ref_norm.square() / n > FLAT_VARIANCE
    scores = []
    for s in range(-m, m + 1):
        win = yb[..., m + s:W - m + s]
        win = win - win.mean((1, 2), keepdim=True)
        win_norm = win.square().sum((1, 2)).sqrt()
        ncc = (ref * win).sum((1, 2)) / (ref_norm * win_norm).clamp_min(1e-300)
        ok = textured & (win_norm.square() / n > FLAT_VARIANCE)
        scores.append(torch.where(ok, ncc, torch.full_like(ncc, float("nan"))))
    c = torch.stack(scores, 1)
    defined = ~torch.isnan(c).all(1)
    c = torch.nan_to_num(c, nan=-2.0)
    k = c.argmax(1)
    peak = c.gather(1, k[:, None])[:, 0]
    cm = c.gather(1, (k - 1).clamp(min=0)[:, None])[:, 0]
    cp = c.gather(1, (k + 1).clamp(max=2 * m)[:, None])[:, 0]
    curv = cm - 2 * peak + cp
    refine = (k > 0) & (k < 2 * m) & (curv < 0) & (peak < EXACT_MATCH)
    delta = torch.where(refine, 0.5 * (cm - cp) / torch.where(refine, curv, torch.ones_like(curv)),
                        torch.zeros_like(curv)).clamp(-0.5, 0.5)
    nan = torch.full_like(peak, float("nan"))
    shift = torch.where(defined, (k - m).double() + delta, nan)
    return shift.float().cpu(), torch.where(defined, peak, nan).float().cpu()


def frame_motion(frames):
    """(B, K) mean absolute change between consecutive frames of a list of K + 1 (B, 3, H, W) batches."""
    return torch.stack([(frames[k + 1].float() - frames[k].float()).abs().flatten(1).mean(1)
                        for k in range(len(frames) - 1)], 1).cpu()


def _sample(model, diffusion, context, act, phase, keys, shape, steps, device):
    """One DDIM sample (eta 0) per window from its own keyed noise, as `eval_tf.main` draws it."""
    noise = window_noise(shape, keys).to(device)
    bucket = torch.zeros(shape[0], dtype=torch.long, device=device)
    cuda = str(device).startswith("cuda")
    with torch.autocast("cuda" if cuda else "cpu", dtype=torch.bfloat16, enabled=cuda):
        pred = diffusion.ddim_sample(lambda xt, t: model(xt, t, act, context, bucket, phase), shape,
                                     steps=steps, noise=noise, device=device)
    return pred.float()


@torch.no_grad()
def run_windows(model, decode_fn, ds, windows, *, latent_channels, diffusion, steps=10, seed=0, batch_size=16,
                device="cpu", phase_buckets=0, raw=None, log_every=0):
    """Per-window records: the three shifts, their match confidences and the rollout motion.

    `ds` is a `TicWindowDataset` built with `horizon=MOTION_HORIZON, with_horizon=True` and the
    checkpoint's action history; `windows` is `turning_windows`' selection; `decode_fn` maps latents
    to (B, 3, 240, 320) images in [0, 1]; `raw` is an `eval_tf.RawFrames` or None.
    """
    C = latent_channels
    order = sorted((int(g), d) for d in DIRECTIONS for g in windows[d])   # episode order, for the raw cache
    records = []
    for b0 in range(0, len(order), batch_size):
        chunk = order[b0:b0 + batch_size]
        items = [ds[g] for g, _ in chunk]
        ctx, tgts, acts, phases = (torch.stack([it[j] for it in items]).to(device) for j in range(4))
        H = tgts.shape[1]
        keys = [[(seed, g, k) for g, _ in chunk] for k in range(H)]
        phase = [phases[:, k] if phase_buckets else None for k in range(H)]
        sample = functools.partial(_sample, model, diffusion, shape=(len(chunk), C) + tuple(tgts.shape[-2:]),
                                   steps=steps, device=device)
        # the swap reuses the recorded sample's context and noise, so only the control differs
        preds = [sample(ctx, acts[:, 0], phase[0], keys[0])]
        swapped = sample(ctx, swap_turns(acts[:, 0]), phase[0], keys[0])
        run = ctx
        for k in range(1, H):
            run = torch.cat([run[:, C:], preds[-1]], dim=1)
            preds.append(sample(run, acts[:, k], phase[k], keys[k]))
        last = decode_fn(ctx[:, -C:])
        pred_img = [decode_fn(p) for p in preds]
        real_img = [decode_fn(tgts[:, k]) for k in range(H)]
        s_rec, p_rec = horizontal_shift(last, pred_img[0])
        s_swp, p_swp = horizontal_shift(last, decode_fn(swapped))
        s_ref, p_ref = horizontal_shift(last, real_img[0])
        m_pred, m_real = frame_motion([last] + pred_img), frame_motion([last] + real_img)
        for i, (g, d) in enumerate(chunk):
            slot, start = ds.locate(g)
            ep = ds.episodes[slot]
            tic = ds.target_tic(g)
            r = {"index": g, "episode": int(ep[0]), "map": int(ep[3]), "start": start, "target_tic": tic,
                 "direction": d, "expected_sign": EXPECTED_SIGN[d],
                 "shift_recorded": float(s_rec[i]), "shift_swapped": float(s_swp[i]),
                 "shift_reference": float(s_ref[i]),
                 "peak_recorded": float(p_rec[i]), "peak_swapped": float(p_swp[i]), "peak_reference": float(p_ref[i]),
                 "motion_pred": [float(x) for x in m_pred[i]], "motion_real": [float(x) for x in m_real[i]]}
            if raw is not None:
                # the window rules guarantee consecutive tics, so the last context frame is tic - 1
                lf, rf = (torch.from_numpy(raw.get(int(ep[0]), t)).permute(2, 0, 1).float().div(255).unsqueeze(0)
                          for t in (tic - 1, tic))
                r["shift_reference_raw"] = float(horizontal_shift(lf, rf)[0][0])
            records.append(r)
        if log_every and (b0 // batch_size + 1) % log_every == 0:
            print(f"  {len(records)}/{len(order)} windows", flush=True)
    return records


def _signs(v):
    """Sign of each shift, 0 inside the deadband or where the shift is undefined."""
    v = np.asarray(v, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.where(np.abs(v) >= SIGN_DEADBAND, np.sign(v), 0.0)


def shift_stats(values, expected):
    """Mean and median shift over the defined windows, and the fraction moving the `expected` way."""
    v = np.asarray(values, dtype=np.float64)
    ok = v[~np.isnan(v)]
    return {"mean": float(ok.mean()) if len(ok) else None, "median": float(np.median(ok)) if len(ok) else None,
            "expected_sign_frac": float(np.mean(_signs(v) == expected)) if len(v) else None,
            "undefined": int(np.isnan(v).sum())}


def flip_frac(rec, swp):
    """Fraction of windows whose shift has opposite signs under the recorded and the swapped control."""
    return float(np.mean(_signs(rec) * _signs(swp) < 0)) if len(rec) else None


PERSISTENCE = {"mean": 0.0, "median": 0.0, "expected_sign_frac": 0.0, "correct_frac": 0.0}


def summarize(records):
    """Per-direction and pooled directional numbers, and the motion ratio, from `run_windows` records."""
    has_raw = bool(records) and "shift_reference_raw" in records[0]

    def col(rows, key):
        return np.array([r[key] for r in rows], dtype=np.float64)

    per = {}
    for d in DIRECTIONS:
        rows = [r for r in records if r["direction"] == d]
        e = EXPECTED_SIGN[d]
        rec, swp, ref = col(rows, "shift_recorded"), col(rows, "shift_swapped"), col(rows, "shift_reference")
        per[d] = {"windows": len(rows), "correct_frac": flip_frac(rec, swp),
                  "recorded": shift_stats(rec, e), "swapped": shift_stats(swp, -e), "reference": shift_stats(ref, e),
                  "persistence": dict(PERSISTENCE)}
        per[d]["ref_frac"] = per[d]["reference"]["expected_sign_frac"]
        if has_raw:
            per[d]["reference_raw"] = shift_stats(col(rows, "shift_reference_raw"), e)
    expected = np.array([r["expected_sign"] for r in records], dtype=np.float64)

    def agree(key):
        return float(np.mean(_signs(col(records, key)) == expected)) if records else None

    pred = np.array([r["motion_pred"] for r in records], dtype=np.float64).reshape(len(records), -1)
    real = np.array([r["motion_real"] for r in records], dtype=np.float64).reshape(len(records), -1)

    def ratio(p, q):
        return float(p.sum() / q.sum()) if q.sum() > 0 else None

    return {"windows": len(records), "sign_deadband_px": SIGN_DEADBAND,
            "correct_frac": flip_frac(col(records, "shift_recorded"), col(records, "shift_swapped")),
            "ref_frac": agree("shift_reference"), "ref_raw_frac": agree("shift_reference_raw") if has_raw else None,
            "recorded_expected_sign_frac": agree("shift_recorded"),
            "persistence": dict(PERSISTENCE), "per_direction": per,
            "motion": {"horizon": int(pred.shape[1]), "ratio": ratio(pred, real),
                       "per_horizon": {str(h + 1): ratio(pred[:, h], real[:, h]) for h in range(pred.shape[1])},
                       "mean_pred": float(pred.mean()) if pred.size else None,
                       "mean_real": float(real.mean()) if real.size else None,
                       "persistence_ratio": 0.0}}


def fmt(x):
    return "nan" if x is None else f"{x:.4f}"


def summary_line(run, step, summary, weights="live"):
    """The one line a steward greps; `weights` says which weights (live or ema) it measured."""
    return (f"DIRECTIONAL_CHECK {run} {step} correct_frac={fmt(summary['correct_frac'])} "
            f"ref_frac={fmt(summary['ref_frac'])} weights={weights}")


def model_namespace(args):
    """The argument set `eval_tf.load_model` and `checkpoint_interface` read."""
    return argparse.Namespace(ckpt=args.ckpt, backbone=args.backbone, num_actions=args.num_actions,
                              context_frames=args.context_frames, noise_buckets=args.noise_buckets,
                              hf_cache=args.hf_cache, use_ema=args.use_ema, objective="auto", sd_path=args.sd_path,
                              pixart_path=args.pixart_path, unidiffuser_path=args.unidiffuser_path,
                              sd35_path=args.sd35_path, tic_stride=None, action_history=None)


def main(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    ns = model_namespace(args)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    trained = checkpoint_interface(ck, ns)
    del ck
    if trained["tic_stride"] != 1:
        raise SystemExit(f"{args.ckpt} is not a next-tic model (tic_stride {trained['tic_stride']}); the check swaps "
                         "the control of one tic")
    if not trained["action_history"]:
        raise SystemExit(f"{args.ckpt} conditions on the requested action id, not an executed-control history, so "
                         "there is no newest control token to swap TURN_LEFT and TURN_RIGHT in")
    latent_channels = resolve_latent_channels(args.backbone, args.latent_channels)
    ds = TicWindowDataset(args.latents_dir, load_split(args.split)[args.subset], args.context_frames,
                          latent_channels=latent_channels, horizon=MOTION_HORIZON, with_horizon=True,
                          action_history=trained["action_history"])
    windows, available = turning_windows(ds, args.windows, args.seed)
    model, step, objective = load_model(ns, device, latent_channels, trained)
    vae = build_vae(args.vae_path, args.vae_subfolder, device, args.hf_cache, latent_channels=latent_channels,
                    scaling_factor=args.latent_scale, shift_factor=args.latent_shift)
    run = args.run or os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
    print(f"{run} step {step} ({'ema' if args.use_ema else 'live'}): {args.windows} left and {args.windows} right "
          f"turning windows of {available}, {args.steps} DDIM steps, objective {objective}", flush=True)

    def dec(z):
        return decode(vae, z, args.latent_scale, args.latent_shift)

    records = run_windows(model, dec, ds, windows, latent_channels=latent_channels,
                          diffusion=VDiffusion(device=device, objective=objective), steps=args.steps,
                          seed=args.seed, batch_size=args.batch_size, device=device,
                          phase_buckets=trained["phase_buckets"],
                          raw=RawFrames(args.parquet_dir) if args.parquet_dir else None, log_every=4)
    summary = summarize(records)
    out = {"summary": summary, "windows": records,
           "config": {**vars(args), "run": run, "step": step, "resolved_latent_channels": latent_channels,
                      "resolved_objective": objective, "checkpoint_interface": trained,
                      "available_turning_windows": available, "window_validity": ds.summary,
                      "motion_mode": f"closed-loop rollout of {MOTION_HORIZON} tics from real context with the "
                                     "recorded controls (eval_tf --horizon-tics contract); h1 is teacher-forced",
                      "max_shift_px": MAX_SHIFT, "band_rows": list(BAND_ROWS), "sign_deadband_px": SIGN_DEADBAND},
           "decoder": decoder_record(args)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_direction"}, indent=1, default=float))
    print(summary_line(run, step, summary, "ema" if args.use_ema else "live"), flush=True)
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--ckpt", required=True, help="a next-tic recovery checkpoint or snap_*.pt")
    p.add_argument("--backbone", choices=list(BACKBONES), required=True)
    p.add_argument("--run", default="", help="the run name on the summary line; default the checkpoint's directory")
    p.add_argument("--use-ema", dest="use_ema", action="store_true")
    p.add_argument("--latents-dir", dest="latents_dir", required=True)
    p.add_argument("--split", required=True, help="split JSON whose --subset lists the episode ids")
    p.add_argument("--subset", default="val", choices=["val", "train", "unseen_map"])
    p.add_argument("--parquet-dir", dest="parquet_dir", default="",
                   help="raw recordings; adds the decoder-free reference shift")
    p.add_argument("--windows", type=int, default=128, help="turning windows per direction")
    p.add_argument("--steps", type=int, default=10, help="DDIM sampler steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--latent-channels", dest="latent_channels", type=int, default=0,
                   help="0 takes the backbone's own (4 for the SD KL-f8 rows, 16 for sd35)")
    p.add_argument("--context-frames", dest="context_frames", type=int, default=32)
    p.add_argument("--num-actions", dest="num_actions", type=int, default=29)
    p.add_argument("--noise-buckets", dest="noise_buckets", type=int, default=10)
    p.add_argument("--vae-path", dest="vae_path", default="",
                   help="decoder directory or repo id; default sd-vae-ft-mse")
    p.add_argument("--vae-subfolder", dest="vae_subfolder", default="")
    p.add_argument("--latent-scale", dest="latent_scale", type=float, default=LATENT_SCALE)
    p.add_argument("--latent-shift", dest="latent_shift", type=float, default=None)
    p.add_argument("--sd-path", dest="sd_path", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--pixart-path", dest="pixart_path", default=PIXART_DEFAULT)
    p.add_argument("--unidiffuser-path", dest="unidiffuser_path", default=UNIDIFFUSER_DEFAULT)
    p.add_argument("--sd35-path", dest="sd35_path", default=SD35_DEFAULT)
    p.add_argument("--hf-cache", dest="hf_cache", default=None)
    p.add_argument("--out", required=True, help="the JSON report")
    return p


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
