"""
Post-hoc audit of finished rollouts: blur control, late-drop distribution, episode bootstrap.

Reads the artifacts `rollout_eval.py` already wrote (`rollouts_seen.npz`,
`rollout_metrics_seen/drift.json`) and the teacher-forced `per_window.csv` files, and answers
three questions that the averaged drift curve cannot:

1. **Blur control** (`--blur`). Decode the predicted and ground-truth frames at the requested
   horizons exactly the way `rollout_eval.do_score` does, blur *the predictions only* with a
   Gaussian of sigma in {0, 0.5, 1, 2, 4} px, and rescore PSNR and LPIPS against the same
   targets. Copy-seed (the last context frame repeated) runs through the identical sweep as
   the persistence reference. If blurring a model closes a PSNR gap without costing LPIPS,
   that model's PSNR lead is smoothing rather than prediction.

2. **Late drop** (`--late-drop`). Per-rollout PSNR at horizon 32 and 64 and the per-rollout
   change between them, so a mean that is dragged by a handful of collapsed clips is visible
   as a distribution (median, p10, p90) and as trimmed means.

3. **Uncertainty** (`--bootstrap`). Windows and rollouts inside one episode are correlated, so
   the resampling unit is the *episode*, not the window. Bootstrap 95% percentile CIs for the
   teacher-forced and rollout headline numbers, and for the paired U-Net minus DiT differences
   (the window and rollout sets are shared across runs, so the difference is paired).

`--identity` checks that the runs really do share rollout trajectories, context endpoints,
action sequences and teacher-forced window sets.

Decoding, PSNR and the IDM procedure are the functions `rollout_eval.py` and `eval_tf.py` use,
imported rather than reimplemented, so the numbers are comparable to `drift.json`.

    python rollout_audit.py --all \
        --results-root /sata2/data/rnagabhi/doom/results_spiderman \
        --runs 030-dit-l32-aligned 032-dit-l32-aligned-seed1 031-unet-l32-aligned \
        --vae-path /sata2/data/rnagabhi/doom/vae_decoder_arnold_lpips/vae \
        --idm /sata2/data/rnagabhi/doom/results_spiderman/idm_aligned/idm.pt \
        --summary /sata2/data/rnagabhi/doom/tmp/audit/summary.json
"""
import argparse
import csv
import hashlib
import json
import os

import numpy as np
import torch

from eval_tf import decode
from rollout_eval import psnr

# Rollouts are indexed by 1-based horizon everywhere in drift.json; keep that convention.
DEFAULT_HORIZONS = (8, 32, 64)
DEFAULT_SIGMAS = (0.0, 0.5, 1.0, 2.0, 4.0)


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------

def md5_of(array):
    """Stable digest of an array's bytes, for cross-run identity checks."""
    return hashlib.md5(np.ascontiguousarray(array).tobytes()).hexdigest()


def gaussian_blur(img, sigma):
    """Gaussian blur on a (B, 3, H, W) image batch in [0, 1]; sigma 0 is the identity."""
    if sigma <= 0:
        return img
    from torchvision.transforms.functional import gaussian_blur as tv_blur
    k = 2 * int(np.ceil(3 * sigma)) + 1          # 3 sigma each side, odd kernel
    return tv_blur(img, kernel_size=[k, k], sigma=[sigma, sigma]).clamp(0, 1)


def load_vae_for_scoring(vae_path, device):
    """The fine-tuned decoder the rollouts were scored with, or the frozen default."""
    if vae_path:
        from diffusers.models import AutoencoderKL
        return AutoencoderKL.from_pretrained(vae_path).to(device).eval()
    from doomdit_utils import load_vae
    return load_vae(device)


def percentile_ci(samples, alpha=0.05):
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# --------------------------------------------------------------------------------------
# question 1 and 2: one decode pass yields both
# --------------------------------------------------------------------------------------

@torch.no_grad()
def decode_sweep(npz, vae, lpips_model, horizons, sigmas, device, batch=16, log_every=64):
    """Per-rollout PSNR and LPIPS for predictions and copy-seed at each horizon and blur sigma.

    Returns nested dict metric -> horizon -> sigma -> (N,) float64 array. Decoding follows
    `rollout_eval.do_score`: latents are divided by LATENT_SCALE, decoded, cropped to 240 rows
    and mapped to [0, 1]; only the prediction (and, as a control, the copy-seed image) is
    blurred, never the target.
    """
    pred, gt, seed = npz["pred"], npz["gt"], npz["seed"]
    n = pred.shape[0]
    out = {m: {h: {s: np.zeros(n) for s in sigmas} for h in horizons} for m in ("psnr", "lpips", "copy_psnr", "copy_lpips")}
    for i in range(0, n, batch):
        sl = slice(i, min(i + batch, n))
        last = decode(vae, torch.from_numpy(np.asarray(seed[sl, -1], dtype=np.float32)).to(device))
        for h in horizons:
            g = decode(vae, torch.from_numpy(np.asarray(gt[sl, h - 1], dtype=np.float32)).to(device))
            p = decode(vae, torch.from_numpy(np.asarray(pred[sl, h - 1], dtype=np.float32)).to(device))
            for s in sigmas:
                pb, cb = gaussian_blur(p, s), gaussian_blur(last, s)
                out["psnr"][h][s][sl] = psnr(pb, g).double().cpu().numpy()
                out["copy_psnr"][h][s][sl] = psnr(cb, g).double().cpu().numpy()
                out["lpips"][h][s][sl] = lpips_model(pb * 2 - 1, g * 2 - 1).flatten().double().cpu().numpy()
                out["copy_lpips"][h][s][sl] = lpips_model(cb * 2 - 1, g * 2 - 1).flatten().double().cpu().numpy()
            del g, p
        del last
        if (sl.stop) % log_every == 0 or sl.stop == n:
            print(f"    decoded {sl.stop}/{n} rollouts", flush=True)
    return out


def late_drop_stats(psnr32, psnr64, drops=(0.05, 0.10)):
    """Distribution of the per-rollout horizon-32 to horizon-64 PSNR change.

    `trimmed` drops the worst rollouts *by that change* (the clips that collapse late);
    `trimmed_by_psnr64` drops the worst by absolute horizon-64 PSNR. Both say how much of the
    reported mean a small tail owns.
    """
    d = psnr64 - psnr32
    n = len(d)
    stats = {
        "n": int(n),
        "mean_psnr32": float(psnr32.mean()), "mean_psnr64": float(psnr64.mean()),
        "mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
        "p10_delta": float(np.percentile(d, 10)), "p90_delta": float(np.percentile(d, 90)),
        "frac_delta_negative": float((d < 0).mean()),
        "frac_delta_below_-2db": float((d < -2.0).mean()),
    }
    for f in drops:
        k = int(np.floor(f * n))
        keep_delta = np.argsort(d)[k:]                 # drop the k most negative changes
        keep_abs = np.argsort(psnr64)[k:]              # drop the k worst horizon-64 clips
        stats[f"drop_worst_{int(f * 100)}pct"] = {
            "dropped": int(k),
            "mean_delta": float(d[keep_delta].mean()),
            "mean_delta_shift": float(d[keep_delta].mean() - d.mean()),
            "mean_psnr64_trim_by_delta": float(psnr64[keep_delta].mean()),
            "mean_psnr64_trim_by_psnr64": float(psnr64[keep_abs].mean()),
            "mean_psnr64_shift_by_psnr64": float(psnr64[keep_abs].mean() - psnr64.mean()),
        }
    return stats


# --------------------------------------------------------------------------------------
# question 3: IDM per rollout, and the episode bootstrap
# --------------------------------------------------------------------------------------

@torch.no_grad()
def idm_per_rollout(npz, idm_path, device):
    """Per-rollout mean IDM top-1 action agreement over the horizon, and the real-frame reference.

    Same procedure as `rollout_eval.do_score`: the windowed IDM is fed the last K-1 real seed
    latents followed by the rollout, so every judged transition has bidirectional context.
    """
    from train_idm import IDM
    ck = torch.load(idm_path, map_location="cpu", weights_only=False)
    idm = IDM(ck["num_actions"], ck["width"], ck.get("window", 8), ck.get("depth", 4)).to(device).eval()
    idm.load_state_dict(ck["model"])
    k = idm.window
    pred, gt, seed, actions = npz["pred"], npz["gt"], npz["seed"], npz["actions"]
    n, h = pred.shape[:2]
    res = {"idm_top1": np.zeros(n), "idm_real_top1": np.zeros(n)}
    for i in range(n):
        y = torch.from_numpy(actions[i]).to(device)
        for tag, frames in (("idm_top1", pred[i]), ("idm_real_top1", gt[i])):
            seq = torch.from_numpy(np.concatenate([seed[i, -(k - 1):], frames], axis=0).astype(np.float32)).to(device)
            logits = idm.predict_sequence(seq)[-h:]
            res[tag][i] = float((logits.argmax(-1) == y).double().mean())
        if (i + 1) % 64 == 0:
            print(f"    idm {i + 1}/{n}", flush=True)
    return res


def episode_resamples(episode_ids, n_boot, seed):
    """Index lists for a cluster bootstrap that resamples whole episodes with replacement.

    Returned once per unit set and reused across runs and metrics, so every bootstrap
    difference is computed on the same draws and the pairing across runs is exact.
    """
    episode_ids = np.asarray(episode_ids)
    uniq = np.unique(episode_ids)
    members = {e: np.flatnonzero(episode_ids == e) for e in uniq}
    rng = np.random.RandomState(seed)
    draws = []
    for _ in range(n_boot):
        picked = uniq[rng.randint(len(uniq), size=len(uniq))]
        draws.append(np.concatenate([members[e] for e in picked]).astype(np.int32))
    return draws


def boot_mean_ci(values, draws):
    """Mean plus a percentile 95% CI from precomputed cluster-bootstrap index draws."""
    values = np.asarray(values, dtype=np.float64)
    samples = np.array([values[d].mean() for d in draws])
    lo, hi = percentile_ci(samples)
    return {"mean": float(values.mean()), "ci95": [lo, hi], "boot_sd": float(samples.std(ddof=1)), "n": int(len(values))}


def boot_diff_ci(a, b, draws):
    """Paired difference a - b with a CI from the same episode draws applied to both runs."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    samples = np.array([a[d].mean() - b[d].mean() for d in draws])
    lo, hi = percentile_ci(samples)
    return {"diff": float(a.mean() - b.mean()), "ci95": [lo, hi], "boot_sd": float(samples.std(ddof=1)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


# --------------------------------------------------------------------------------------
# teacher-forced per-window tables
# --------------------------------------------------------------------------------------

TF_METRICS = ("psnr_raw", "lpips_raw", "psnr_dec", "lpips_dec", "copy_psnr_raw")


def read_per_window(path):
    """Per-window rows of `eval_tf.py`, as float arrays keyed by column plus the episode id."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {"episode": np.array([int(r["episode"]) for r in rows]),
            "index": np.array([int(r["index"]) for r in rows]),
            "start": np.array([int(r["start"]) for r in rows]),
            "action": np.array([int(r["action"]) for r in rows])}
    for k in TF_METRICS:
        if k in rows[0]:
            cols[k] = np.array([float(r[k]) for r in rows])
    return cols


def tf_window_key(cols):
    """Digest of the window set (index, episode, start, action), for the identity check."""
    stacked = np.stack([cols["index"], cols["episode"], cols["start"], cols["action"]])
    return md5_of(stacked)


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------

def run_paths(root, run):
    return {
        "rollouts": os.path.join(root, run, "rollouts_seen.npz"),
        "drift": os.path.join(root, run, "rollout_metrics_seen", "drift.json"),
        "tf_seen": os.path.join(root, run, "eval_tf_seen", "per_window.csv"),
        "tf_unseen": os.path.join(root, run, "eval_tf_unseen", "per_window.csv"),
        "audit_dir": os.path.join(root, run, "audit"),
    }


def identity_report(root, runs):
    """Confirm the runs share rollout trajectories, context endpoints, actions and TF windows."""
    rep = {"rollout": {}, "tf_seen": {}, "tf_unseen": {}}
    for run in runs:
        p = run_paths(root, run)
        d = np.load(p["rollouts"])
        rep["rollout"][run] = {
            "episode_md5": md5_of(d["episode"]), "start_md5": md5_of(d["start"]),
            "actions_md5": md5_of(d["actions"]), "seed_md5": md5_of(d["seed"]),
            "seed_last_frame_md5": md5_of(d["seed"][:, -1]), "gt_md5": md5_of(d["gt"]),
            "shape": list(d["pred"].shape), "num_episodes": int(len(np.unique(d["episode"]))),
        }
        for tag in ("tf_seen", "tf_unseen"):
            cols = read_per_window(p[tag])
            rep[tag][run] = {"window_set_md5": tf_window_key(cols), "num_windows": int(len(cols["index"])),
                             "num_episodes": int(len(np.unique(cols["episode"])))}
    agree = {}
    for section, fields in (("rollout", ("episode_md5", "start_md5", "actions_md5", "seed_md5", "gt_md5")),
                            ("tf_seen", ("window_set_md5",)), ("tf_unseen", ("window_set_md5",))):
        agree[section] = {f: len({rep[section][r][f] for r in runs}) == 1 for f in fields}
    rep["all_identical"] = agree
    return rep


def audit_run(run, args, device, vae, lpips_model, tf_draws):
    """Everything that needs this run's own artifacts; returns the per-run result dict."""
    p = run_paths(args.results_root, run)
    os.makedirs(p["audit_dir"], exist_ok=True)
    npz = np.load(p["rollouts"])
    res = {"run": run, "num_rollouts": int(npz["pred"].shape[0]), "horizon": int(npz["pred"].shape[1])}
    per_rollout = {}

    if args.blur or args.late_drop or args.bootstrap:
        print(f"  [{run}] decoding horizons {list(args.horizons)} x sigmas {list(args.sigmas)}", flush=True)
        sweep = decode_sweep(npz, vae, lpips_model, args.horizons, args.sigmas, device, batch=args.decode_batch)
        res["blur_sweep"] = {
            str(h): {
                str(s): {
                    "psnr": float(sweep["psnr"][h][s].mean()), "lpips": float(sweep["lpips"][h][s].mean()),
                    "copy_seed_psnr": float(sweep["copy_psnr"][h][s].mean()),
                    "copy_seed_lpips": float(sweep["copy_lpips"][h][s].mean()),
                } for s in args.sigmas
            } for h in args.horizons
        }
        for h in args.horizons:
            for s in args.sigmas:
                per_rollout[f"psnr_h{h}_s{s}"] = sweep["psnr"][h][s]
                per_rollout[f"lpips_h{h}_s{s}"] = sweep["lpips"][h][s]
                per_rollout[f"copy_psnr_h{h}_s{s}"] = sweep["copy_psnr"][h][s]
                per_rollout[f"copy_lpips_h{h}_s{s}"] = sweep["copy_lpips"][h][s]
        if 32 in args.horizons and 64 in args.horizons:
            res["late_drop"] = late_drop_stats(sweep["psnr"][32][0.0], sweep["psnr"][64][0.0])

    if args.bootstrap:
        if args.idm:
            print(f"  [{run}] idm", flush=True)
            per_rollout.update(idm_per_rollout(npz, args.idm, device))
        draws = episode_resamples(npz["episode"], args.boot, args.boot_seed)
        boot = {"rollout": {}}
        for key in ("psnr_h64_s0.0", "lpips_h64_s0.0", "copy_psnr_h64_s0.0", "idm_top1", "idm_real_top1"):
            if key in per_rollout:
                boot["rollout"][key] = boot_mean_ci(per_rollout[key], draws)
        for tag in ("tf_seen", "tf_unseen"):
            cols = read_per_window(p[tag])
            boot[tag] = {k: boot_mean_ci(cols[k], tf_draws[tag]) for k in TF_METRICS if k in cols}
        res["bootstrap"] = boot

    json.dump(res, open(os.path.join(p["audit_dir"], "audit.json"), "w"), indent=1)
    if per_rollout:
        np.savez(os.path.join(p["audit_dir"], "per_rollout.npz"), episode=npz["episode"], **per_rollout)
    return res, per_rollout


def paired_differences(args, per_rollout, tf_draws, rollout_draws):
    """U-Net minus each DiT, on the shared rollouts and the shared teacher-forced windows."""
    out = {}
    for unet in args.unet_runs:
        for dit in [r for r in args.runs if r != unet]:
            key = f"{unet}_minus_{dit}"
            entry = {"rollout": {}, "tf_seen": {}, "tf_unseen": {}}
            for k in ("psnr_h64_s0.0", "lpips_h64_s0.0", "idm_top1"):
                if k in per_rollout.get(unet, {}) and k in per_rollout.get(dit, {}):
                    entry["rollout"][k] = boot_diff_ci(per_rollout[unet][k], per_rollout[dit][k], rollout_draws)
            for tag in ("tf_seen", "tf_unseen"):
                a = read_per_window(run_paths(args.results_root, unet)[tag])
                b = read_per_window(run_paths(args.results_root, dit)[tag])
                assert tf_window_key(a) == tf_window_key(b), f"window sets differ: {unet} vs {dit} ({tag})"
                entry[tag] = {m: boot_diff_ci(a[m], b[m], tf_draws[tag]) for m in TF_METRICS if m in a and m in b}
            out[key] = entry
    return out


def main(args):
    if args.all:
        args.blur = args.late_drop = args.bootstrap = args.identity = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {"runs": list(args.runs), "results_root": args.results_root,
               "horizons": list(args.horizons), "sigmas": list(args.sigmas),
               "bootstrap_resamples": args.boot, "bootstrap_seed": args.boot_seed,
               "vae_path": args.vae_path, "idm": args.idm, "device": device}

    if args.identity:
        print("identity check", flush=True)
        summary["identity"] = identity_report(args.results_root, args.runs)

    tf_draws = {}
    if args.bootstrap:
        for tag in ("tf_seen", "tf_unseen"):
            cols = read_per_window(run_paths(args.results_root, args.runs[0])[tag])
            tf_draws[tag] = episode_resamples(cols["episode"], args.boot, args.boot_seed)

    vae = lpips_model = None
    if args.blur or args.late_drop or args.bootstrap:
        vae = load_vae_for_scoring(args.vae_path, device)
        import lpips as lpips_pkg
        lpips_model = lpips_pkg.LPIPS(net=args.lpips_net, verbose=False).to(device).eval()

    per_run, per_rollout = {}, {}
    for run in args.runs:
        print(f"== {run}", flush=True)
        res, pr = audit_run(run, args, device, vae, lpips_model, tf_draws)
        per_run[run], per_rollout[run] = res, pr
    summary["per_run"] = per_run

    if args.bootstrap and args.unet_runs:
        rollout_draws = episode_resamples(np.load(run_paths(args.results_root, args.runs[0])["rollouts"])["episode"],
                                          args.boot, args.boot_seed)
        summary["paired"] = paired_differences(args, per_rollout, tf_draws, rollout_draws)

    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    json.dump(summary, open(args.summary, "w"), indent=1)
    print("WROTE", args.summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--unet-runs", nargs="*", default=[], help="runs that appear on the left of the paired differences")
    p.add_argument("--all", action="store_true")
    p.add_argument("--blur", action="store_true")
    p.add_argument("--late-drop", action="store_true")
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--identity", action="store_true")
    p.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    p.add_argument("--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMAS))
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--decode-batch", type=int, default=16)
    p.add_argument("--lpips-net", default="alex")
    p.add_argument("--vae-path", default="")
    p.add_argument("--idm", default="")
    p.add_argument("--summary", required=True)
    main(p.parse_args())
