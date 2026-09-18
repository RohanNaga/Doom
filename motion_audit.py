"""Motion audit of finished rollouts: is a long-horizon PSNR lead stability or staticness?

At horizon 64 the two DiTs hold higher PSNR than the U-Net while every model sits at or below
the copy-seed persistence reference perceptually. Both "more stable" and "more static" predict
that ordering, so PSNR against the ground truth cannot separate them. This script measures how
much the predictions *move*, and compares that with how much the real continuation moves over
the same trajectory.

For each run it decodes `rollouts_seen.npz` exactly the way `rollout_eval.do_score` and
`rollout_audit.decode_sweep` do (latents divided by LATENT_SCALE, decoded, cropped to 240 rows,
mapped to [0, 1]) and computes, per horizon h = 1..64 and per rollout:

a. **Frame-difference energy** `mean |x_h - x_{h-1}|` over pixels and channels, with x_0 the last
   context frame, for the prediction and for the real continuation from the same context.
b. **Distance to the seed**, PSNR(x_h, x_0), for the prediction and for the real frame. A model
   that freezes stays close to the seed for longer than reality does.
c. **Dense optical-flow magnitude** (Farneback on greyscale consecutive frames), mean magnitude
   per frame, for the prediction and the real continuation. Skipped with a note if cv2 is absent.
d. The **fraction of rollouts whose prediction at h = 64 is closer to the seed than the real
   frame is**, i.e. PSNR(pred_64, seed) > PSNR(real_64, seed).

The headline is the prediction/real ratio of (a) and (c) at h = 8, 16, 32, 64, as a ratio of
means with a 95% percentile CI from an episode-level cluster bootstrap (windows inside an
episode are correlated, so the resampling unit is the episode; the draws come from
`rollout_audit.episode_resamples`, and the rollout set is shared across runs so cross-run
differences are paired). Copy-seed is the anchor: its ratio is 0 by construction.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/sata2/data/rnagabhi/doom/repo \
    python motion_audit.py \
        --results-root /sata2/data/rnagabhi/doom/results_spiderman \
        --runs 030-dit-l32-aligned 032-dit-l32-aligned-seed1 031-unet-l32-aligned \
        --vae-path /sata2/data/rnagabhi/doom/vae_decoder_arnold_lpips/vae \
        --summary /sata2/data/rnagabhi/doom/tmp/motion/summary.json
"""
import argparse
import json
import os

import numpy as np
import torch

from eval_tf import decode
from rollout_eval import psnr
from rollout_audit import episode_resamples, load_vae_for_scoring, percentile_ci, run_paths

DEFAULT_REPORT_HORIZONS = (8, 16, 32, 64)

# Per-rollout, per-horizon series this pass produces. `diff` is frame-difference energy in
# [0, 1] units, `psnr_seed` decibels against the last context frame, `flow` pixels per frame.
SERIES = ("diff_pred", "diff_real", "psnr_seed_pred", "psnr_seed_real", "flow_pred", "flow_real")

# Farneback settings: OpenCV's documented defaults, fixed here so the numbers are reproducible.
FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

# Verdict thresholds on the prediction/real motion ratio CI (see `verdict`).
UNDER_MOVING_CEIL = 0.85
OVER_MOVING_FLOOR = 1.15


def import_cv2():
    """OpenCV if the server env has it, else None (the flow section is then skipped)."""
    try:
        import cv2
    except ImportError:
        return None
    return cv2


def to_greyscale_u8(img, cv2):
    """A (B, 3, H, W) batch in [0, 1] as a list of uint8 greyscale arrays for Farneback.

    Farneback wants 8-bit single-channel input, so the quantisation happens here rather than
    inside the flow call; PSNR and the frame differences stay on the float tensors.
    """
    arr = (img.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return [cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) for a in arr]


def flow_magnitude(cv2, prev_grey, cur_grey):
    """Mean dense-flow magnitude in pixels between two greyscale frames."""
    f = cv2.calcOpticalFlowFarneback(prev_grey, cur_grey, None, **FARNEBACK)
    return float(np.hypot(f[..., 0], f[..., 1]).mean())


@torch.no_grad()
def motion_pass(npz, vae, device, batch, cv2, max_rollouts=0, log_every=64):
    """Per-rollout, per-horizon motion series for the prediction and the real continuation.

    Returns a dict of (N, H) float arrays keyed by `SERIES`; the flow entries are all-NaN when
    cv2 is unavailable. Frames are decoded one horizon at a time and only the previous horizon's
    batch is retained, so memory does not grow with the horizon.
    """
    pred, gt, seed = npz["pred"], npz["gt"], npz["seed"]
    n, h_max = pred.shape[:2]
    if max_rollouts:
        n = min(n, max_rollouts)
    out = {k: np.full((n, h_max), np.nan) for k in SERIES}

    def dec(z):
        return decode(vae, torch.from_numpy(np.asarray(z, dtype=np.float32)).to(device))

    for i in range(0, n, batch):
        sl = slice(i, min(i + batch, n))
        # x_0 for both sequences is the same last context frame: the rollout and the real
        # continuation start from an identical history.
        s = dec(seed[sl, -1])
        prev = {"pred": s, "real": s}
        prev_grey = {"pred": to_greyscale_u8(s, cv2), "real": to_greyscale_u8(s, cv2)} if cv2 else None
        for h in range(1, h_max + 1):
            cur = {"pred": dec(pred[sl, h - 1]), "real": dec(gt[sl, h - 1])}
            for tag, x in cur.items():
                out[f"diff_{tag}"][sl, h - 1] = (x - prev[tag]).abs().flatten(1).mean(1).double().cpu().numpy()
                out[f"psnr_seed_{tag}"][sl, h - 1] = psnr(x, s).double().cpu().numpy()
                if cv2:
                    grey = to_greyscale_u8(x, cv2)
                    out[f"flow_{tag}"][sl, h - 1] = [
                        flow_magnitude(cv2, a, b) for a, b in zip(prev_grey[tag], grey)
                    ]
                    prev_grey[tag] = grey
            prev = cur
        del prev, cur, s
        if sl.stop % log_every == 0 or sl.stop == n:
            print(f"    motion pass {sl.stop}/{n} rollouts", flush=True)
    return out


def boot_ratio_ci(num, den, draws):
    """Ratio of means num/den with a 95% CI from precomputed episode-bootstrap index draws.

    The ratio of means, not the mean of per-rollout ratios: a rollout whose real continuation
    barely moves would otherwise dominate the average through a near-zero denominator.
    """
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    samples = np.array([num[d].mean() / den[d].mean() for d in draws])
    lo, hi = percentile_ci(samples)
    return {"ratio": float(num.mean() / den.mean()), "ci95": [lo, hi],
            "boot_sd": float(samples.std(ddof=1)),
            "mean_pred": float(num.mean()), "mean_real": float(den.mean()), "n": int(len(num))}


def boot_mean_ci(values, draws):
    """Mean plus a 95% percentile CI from the same episode-bootstrap draws."""
    values = np.asarray(values, dtype=np.float64)
    samples = np.array([values[d].mean() for d in draws])
    lo, hi = percentile_ci(samples)
    return {"mean": float(values.mean()), "ci95": [lo, hi], "boot_sd": float(samples.std(ddof=1)),
            "n": int(len(values))}


def verdict(ratio_entries):
    """Plain verdict from the motion-ratio CIs at the reported horizons.

    under-moving when every CI lies below `UNDER_MOVING_CEIL`, over-moving when every CI lies
    above `OVER_MOVING_FLOOR`, matched when every CI brackets 1, and mixed otherwise (the label
    then names the horizons that are under or over).
    """
    if not ratio_entries:
        return "no measurement"
    labels = {}
    for h, e in ratio_entries.items():
        lo, hi = e["ci95"]
        if hi < UNDER_MOVING_CEIL:
            labels[h] = "under"
        elif lo > OVER_MOVING_FLOOR:
            labels[h] = "over"
        elif lo <= 1.0 <= hi:
            labels[h] = "matched"
        else:
            labels[h] = "between"
    vals = set(labels.values())
    if vals == {"under"}:
        return "under-moving"
    if vals == {"over"}:
        return "over-moving"
    if vals == {"matched"}:
        return "matched"
    return "mixed (" + ", ".join(f"h{h}:{labels[h]}" for h in sorted(labels)) + ")"


def aggregate(series, draws, report_horizons, has_flow):
    """Curves over every horizon plus bootstrapped ratios at the reported horizons."""
    h_max = series["diff_pred"].shape[1]
    keys = SERIES if has_flow else tuple(k for k in SERIES if not k.startswith("flow_"))
    res = {
        "horizon": list(range(1, h_max + 1)),
        "curves": {k: [float(v) for v in series[k].mean(axis=0)] for k in keys},
        "ratio_diff": {}, "ratio_flow": {},
        "psnr_seed": {},
    }
    for h in report_horizons:
        j = h - 1
        res["ratio_diff"][str(h)] = boot_ratio_ci(series["diff_pred"][:, j], series["diff_real"][:, j], draws)
        if has_flow:
            res["ratio_flow"][str(h)] = boot_ratio_ci(series["flow_pred"][:, j], series["flow_real"][:, j], draws)
        res["psnr_seed"][str(h)] = {
            "pred": boot_mean_ci(series["psnr_seed_pred"][:, j], draws),
            "real": boot_mean_ci(series["psnr_seed_real"][:, j], draws),
            "pred_minus_real_db": float(series["psnr_seed_pred"][:, j].mean()
                                        - series["psnr_seed_real"][:, j].mean()),
        }
    closer = (series["psnr_seed_pred"][:, -1] > series["psnr_seed_real"][:, -1]).astype(np.float64)
    res["closer_to_seed_than_real_h64"] = boot_mean_ci(closer, draws)
    res["verdict_frame_diff"] = verdict(res["ratio_diff"])
    res["verdict_flow"] = verdict(res["ratio_flow"]) if has_flow else "no measurement (cv2 missing)"
    return res


def copy_seed_anchor(series, report_horizons, has_flow):
    """The persistence anchor: copy-seed repeats x_0, so it moves exactly zero.

    Its frame-difference energy and flow magnitude are identically zero, hence ratio 0 at every
    horizon; its PSNR to the seed is infinite and its prediction is closer to the seed than the
    real frame in every rollout. Recorded rather than measured, with the real-motion
    denominators carried over so the anchor and the models share a scale.
    """
    entry = {"ratio_diff": {}, "ratio_flow": {}, "note": "computed analytically, not decoded"}
    for h in report_horizons:
        j = h - 1
        entry["ratio_diff"][str(h)] = {"ratio": 0.0, "ci95": [0.0, 0.0], "mean_pred": 0.0,
                                       "mean_real": float(series["diff_real"][:, j].mean())}
        if has_flow:
            entry["ratio_flow"][str(h)] = {"ratio": 0.0, "ci95": [0.0, 0.0], "mean_pred": 0.0,
                                           "mean_real": float(series["flow_real"][:, j].mean())}
    entry["psnr_seed_pred_db"] = "inf"
    entry["closer_to_seed_than_real_h64"] = {"mean": 1.0, "ci95": [1.0, 1.0]}
    entry["verdict_frame_diff"] = "under-moving (zero motion by construction)"
    entry["verdict_flow"] = entry["verdict_frame_diff"] if has_flow else "no measurement (cv2 missing)"
    return entry


def paired_pred_motion_diffs(per_run, runs, draws, report_horizons):
    """Run-minus-run prediction motion, on the shared rollouts and the shared episode draws.

    The rollout set is identical across runs (checked by `rollout_audit --identity`), so these
    differences are paired and say whether one backbone moves more than another, separately from
    whether either matches reality.
    """
    out = {}
    for a in runs:
        for b in runs:
            if a >= b:
                continue
            entry = {}
            for h in report_horizons:
                j = h - 1
                x = per_run[a]["diff_pred"][:, j].astype(np.float64)
                y = per_run[b]["diff_pred"][:, j].astype(np.float64)
                samples = np.array([x[d].mean() - y[d].mean() for d in draws])
                lo, hi = percentile_ci(samples)
                entry[str(h)] = {"diff": float(x.mean() - y.mean()), "ci95": [lo, hi],
                                 "excludes_zero": bool(lo > 0 or hi < 0)}
            out[f"{a}_minus_{b}"] = entry
    return out


def render_table(summary):
    """One text table, models x horizons, of the frame-difference and flow motion ratios."""
    runs = summary["runs"] + ["copy-seed"]
    hs = [str(h) for h in summary["report_horizons"]]
    lines = []
    for metric, label in (("ratio_diff", "frame-difference energy"), ("ratio_flow", "optical-flow magnitude")):
        if metric == "ratio_flow" and not summary["cv2_available"]:
            lines.append("optical-flow magnitude ratio: skipped (cv2 unavailable)")
            continue
        lines.append(f"{label}: prediction/real ratio [95% episode-bootstrap CI]")
        lines.append("  " + "run".ljust(32) + "".join(f"h={h}".ljust(24) for h in hs))
        for run in runs:
            src = summary["copy_seed"] if run == "copy-seed" else summary["per_run"][run]
            cells = []
            for h in hs:
                e = src[metric].get(h)
                cells.append("n/a".ljust(24) if e is None
                             else f"{e['ratio']:.3f} [{e['ci95'][0]:.3f},{e['ci95'][1]:.3f}]".ljust(24))
            lines.append("  " + run.ljust(32) + "".join(cells))
        lines.append("")
    lines.append("verdicts (frame difference / flow)")
    for run in runs:
        src = summary["copy_seed"] if run == "copy-seed" else summary["per_run"][run]
        lines.append(f"  {run.ljust(32)} {src['verdict_frame_diff']} / {src['verdict_flow']}")
    return "\n".join(lines)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cv2 = None if args.no_flow else import_cv2()
    if cv2 is None:
        print("NOTE: optical flow skipped (cv2 unavailable)" if not args.no_flow
              else "NOTE: optical flow skipped (--no-flow)", flush=True)
    vae = load_vae_for_scoring(args.vae_path, device)

    summary = {"runs": list(args.runs), "results_root": args.results_root,
               "report_horizons": list(args.report_horizons), "vae_path": args.vae_path,
               "device": device, "cv2_available": cv2 is not None,
               "cv2_version": getattr(cv2, "__version__", None), "farneback": FARNEBACK,
               "bootstrap_resamples": args.boot, "bootstrap_seed": args.boot_seed,
               "verdict_thresholds": {"under_moving_ceil": UNDER_MOVING_CEIL,
                                      "over_moving_floor": OVER_MOVING_FLOOR},
               "per_run": {}}

    draws, episode_ids, series_by_run = None, None, {}
    for run in args.runs:
        print(f"== {run}", flush=True)
        p = run_paths(args.results_root, run)
        os.makedirs(p["audit_dir"], exist_ok=True)
        npz = np.load(p["rollouts"])
        if draws is None:
            episode_ids = npz["episode"] if not args.max_rollouts else npz["episode"][:args.max_rollouts]
            draws = episode_resamples(episode_ids, args.boot, args.boot_seed)
        elif not np.array_equal(episode_ids, npz["episode"][:len(episode_ids)]):
            raise SystemExit(f"{run} does not share the rollout episode set; bootstrap pairing would be wrong")
        series = motion_pass(npz, vae, device, args.decode_batch, cv2, args.max_rollouts)
        series_by_run[run] = series
        res = aggregate(series, draws, args.report_horizons, cv2 is not None)
        res["run"] = run
        res["num_rollouts"] = int(series["diff_pred"].shape[0])
        res["num_episodes"] = int(len(np.unique(npz["episode"])))
        json.dump(res, open(os.path.join(p["audit_dir"], "motion.json"), "w"), indent=1)
        np.savez(os.path.join(p["audit_dir"], "motion_per_rollout.npz"), episode=npz["episode"], **series)
        summary["per_run"][run] = res
        print(f"  {run}: {res['verdict_frame_diff']} (frame diff) / {res['verdict_flow']} (flow)", flush=True)

    summary["copy_seed"] = copy_seed_anchor(series_by_run[args.runs[0]], args.report_horizons, cv2 is not None)
    summary["paired_pred_motion"] = paired_pred_motion_diffs(series_by_run, list(args.runs), draws,
                                                             args.report_horizons)
    summary["table"] = render_table(summary)

    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    json.dump(summary, open(args.summary, "w"), indent=1)
    print("\n" + summary["table"] + "\n")
    print("WROTE", args.summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--report-horizons", type=int, nargs="+", default=list(DEFAULT_REPORT_HORIZONS))
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--decode-batch", type=int, default=16)
    p.add_argument("--vae-path", default="")
    p.add_argument("--max-rollouts", type=int, default=0, help="smoke-test knob: audit only the first N rollouts")
    p.add_argument("--no-flow", action="store_true", help="skip the optical-flow section even if cv2 is present")
    p.add_argument("--summary", required=True)
    main(p.parse_args())
