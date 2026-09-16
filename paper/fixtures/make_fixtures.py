"""
Fabricated result directories with the real on-disk schemas, for testing the paper builders.

Writes, under --out, the layout `scripts/spiderman/after_run2.sh`, `rollout_audit.py`,
`train_idm.py` and `train_wm.py` produce on Spiderman:

    <run>/config.json                              train_wm.py (params, backbone, steps, seed, git)
    <run>/log.jsonl                                start / val / end events
    <run>/eval_tf_{seen,unseen,seen_ema}/metrics.json + per_window.csv     eval_tf.py
    <run>/rollout_metrics_seen/drift.json + fvd16.json + fvd32.json        rollout_eval.py, fvd.py
    <run>/audit/audit.json + per_rollout.npz       rollout_audit.py
    tmp/audit/summary.json                         rollout_audit.py --summary
    idm_aligned/metrics.json, idm_aligned_k2/metrics.json                  train_idm.py

Some artifacts are left out on purpose so the builders' "n/a" paths are exercised:
PixArt has no audit, the SkyReels row has no FVD and no audit, the PixArt compute-matched
run has no rollout, the grid-label rows have no EMA evaluation, and the UniDiffuser run
directory does not exist at all. Nothing here is a real measurement; the base values are
only chosen so the rendered figures look like the real ones.

    python paper/fixtures/make_fixtures.py --out /tmp/paper_fixture
"""
import argparse
import csv
import json
import os

import numpy as np

HORIZON = 64
SIGMAS = (0.0, 0.5, 1.0, 2.0, 4.0)
AUDIT_HORIZONS = (8, 32, 64)

# per-run base numbers: (backbone, params, seen psnr, seen lpips, unseen psnr, unseen lpips, psnr@64, lpips@64,
#                        idm, fvd16, fvd32, final val loss, steps, seed)
RUNS = {
    "030-dit-l32-aligned":       ("dit",    675_000_000, 21.06, 0.311, 19.52, 0.450, 18.30, 0.553, 0.492, 232, 481, 0.2139, 90000, 0),
    "032-dit-l32-aligned-seed1": ("dit",    675_000_000, 21.21, 0.307, 19.59, 0.442, 17.68, 0.550, 0.476, 198, 407, 0.2139, 90000, 1),
    "031-unet-l32-aligned":      ("unet",   860_000_000, 21.36, 0.270, 19.14, 0.446, 16.03, 0.566, 0.509, 184, 356, 0.2043, 90000, 0),
    "033-pixart-l32-aligned":    ("pixart", 612_000_000, 21.40, 0.265, 19.30, 0.440, 17.20, 0.560, 0.500, 190, 380, 0.2030, 90000, 0),
    "050-skyreels-l8-flow":      ("skyreels-df-1.3b", 1_435_571_776, 19.10, 0.400, 18.20, 0.520, 15.50, 0.600, 0.300, None, None, 0.9, 10000, 0),
    "060-dit-cm2500":            ("dit",    675_000_000, 19.80, 0.380, 18.60, 0.500, 16.50, 0.600, 0.330, 320, 650, 0.2600, 2500, 0),
    "061-unet-cm2500":           ("unet",   860_000_000, 20.30, 0.340, 18.70, 0.490, 16.20, 0.590, 0.360, 280, 560, 0.2450, 2500, 0),
    "062-pixart-cm2500":         ("pixart", 612_000_000, 20.10, 0.350, 18.65, 0.495, 16.30, 0.595, 0.350, 300, 600, 0.2480, 2500, 0),
    "ablation_gridlabels_dit75k":  ("dit",  675_000_000, 20.18, 0.358, 18.79, 0.472, 16.07, 0.605, 0.344, 329, 690, 0.2300, 75000, 0),
    "ablation_gridlabels_unet20k": ("unet", 860_000_000, 19.82, 0.372, 18.71, 0.479, 15.90, 0.562, 0.326, 265, 531, 0.2500, 20000, 0),
}
NO_AUDIT = {"033-pixart-l32-aligned", "050-skyreels-l8-flow", "060-dit-cm2500", "061-unet-cm2500", "062-pixart-cm2500",
            "ablation_gridlabels_dit75k", "ablation_gridlabels_unet20k"}
NO_FVD = {"050-skyreels-l8-flow"}
NO_ROLLOUT = {"062-pixart-cm2500"}
NO_EMA = {"050-skyreels-l8-flow", "ablation_gridlabels_dit75k", "ablation_gridlabels_unet20k"}
AUDITED = ["030-dit-l32-aligned", "032-dit-l32-aligned-seed1", "031-unet-l32-aligned"]

COPY_LAST = {"seen": 19.41, "unseen": 18.50}
VAE = {"seen": (28.61, 0.051), "unseen": (26.36, 0.070)}
COPY_SEED_64 = (17.86, 0.510)
TF_KEYS = ("psnr_dec", "lpips_dec", "copy_psnr_dec", "latent_mse", "hud_psnr_dec", "psnr_raw", "lpips_raw",
           "copy_psnr_raw", "vae_psnr", "vae_lpips", "hud_psnr_raw", "hud_vae_psnr")


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)


def tf_windows(rng, corpus, n, psnr_mean, lpips_mean):
    """Per-window rows with the eval_tf.py columns; seen maps 1..15, unseen maps 16..17."""
    maps = list(range(1, 16)) if corpus == "seen" else [16, 17]
    episodes = np.arange(1000, 1000 + 4 * len(maps))
    ep_map = {int(e): maps[i % len(maps)] for i, e in enumerate(episodes)}
    rows = []
    for i in range(n):
        ep = int(rng.choice(episodes))
        m = ep_map[ep]
        map_shift = 0.15 * (m - np.mean(maps))                 # a visible per-map spread
        p_raw = psnr_mean + map_shift + rng.normal(0, 2.5)
        l_raw = max(0.02, lpips_mean - 0.004 * map_shift + rng.normal(0, 0.08))
        rows.append(dict(
            index=i, episode=ep, map=m, start=int(rng.randint(0, 400)), action=int(rng.randint(0, 29)),
            psnr_dec=p_raw + 0.6, lpips_dec=l_raw - 0.02, copy_psnr_dec=COPY_LAST[corpus] + 0.5 + rng.normal(0, 2.5),
            latent_mse=float(abs(rng.normal(0.3, 0.05))), hud_psnr_dec=p_raw + 8,
            psnr_raw=p_raw, lpips_raw=l_raw, copy_psnr_raw=COPY_LAST[corpus] + rng.normal(0, 2.5),
            vae_psnr=VAE[corpus][0] + rng.normal(0, 1.5), vae_lpips=VAE[corpus][1] + rng.normal(0, 0.01),
            hud_psnr_raw=p_raw + 7.5, hud_vae_psnr=VAE[corpus][0] + 6))
    return rows


def write_tf(out_dir, rows, step):
    def agg(k):
        v = np.array([r[k] for r in rows], dtype=np.float64)
        return {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v))), "n": int(len(v))}
    summary = {k: agg(k) for k in TF_KEYS}
    summary["per_map"] = {str(m): {k: float(np.mean([r[k] for r in rows if r["map"] == m])) for k in ("psnr_dec", "lpips_dec")}
                          for m in sorted(set(r["map"] for r in rows))}
    summary["sampling_frames_per_s"] = 3.2
    summary["config"] = {"num_windows": len(rows), "steps": 50, "eta": 0.0, "step": step, "use_ema": out_dir.endswith("_ema")}
    dump(os.path.join(out_dir, "metrics.json"), summary)
    with open(os.path.join(out_dir, "per_window.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def drift_curves(rng, psnr64, lpips64, n=256):
    """Per-rollout (n, 64) PSNR and LPIPS whose horizon means decay from a short-horizon value to the base."""
    h = np.arange(1, HORIZON + 1)
    psnr_mean = psnr64 + (22.5 - psnr64) * np.exp(-(h - 1) / 20.0)
    lpips_mean = lpips64 - (lpips64 - 0.25) * np.exp(-(h - 1) / 18.0)
    per_rollout_offset = rng.normal(0, 2.0, size=(n, 1))
    psnr = psnr_mean[None] + per_rollout_offset + rng.normal(0, 0.4, size=(n, HORIZON))
    lpips = np.clip(lpips_mean[None] - 0.02 * per_rollout_offset + rng.normal(0, 0.03, size=(n, HORIZON)), 0.02, 1.0)
    copy = COPY_SEED_64[0] + (24.0 - COPY_SEED_64[0]) * np.exp(-(h - 1) / 12.0)
    return psnr, lpips, copy


def write_rollout(run_dir, rng, spec, episodes):
    _, _, _, _, _, _, psnr64, lpips64, idm, fvd16, fvd32, _, _, _ = spec
    psnr, lpips, copy = drift_curves(rng, psnr64, lpips64)
    n = psnr.shape[0]
    out = {"horizon": list(range(1, HORIZON + 1)), "psnr": psnr.mean(0).tolist(), "lpips": lpips.mean(0).tolist(),
           "copy_seed_psnr": copy.tolist(), "latent_mse": (0.2 + 0.01 * np.arange(HORIZON)).tolist(), "num_rollouts": int(n)}
    for hh in (8, 16, 32, 64):
        out[f"psnr@{hh}"] = float(psnr[:, hh - 1].mean())
        out[f"lpips@{hh}"] = float(lpips[:, hh - 1].mean())
    idm_top1 = np.clip(idm + rng.normal(0, 0.03, size=HORIZON), 0, 1)
    idm_real = np.clip(0.847 + rng.normal(0, 0.02, size=HORIZON), 0, 1)
    for k, v in (("top1", idm_top1), ("movement", idm_top1 + 0.2), ("real_top1", idm_real), ("real_movement", idm_real + 0.03)):
        out[f"idm_{k}"] = v.tolist()
        out[f"idm_{k}_mean"] = float(v.mean())
    out["idm_val_top1"] = 0.829
    out["idm_val_movement"] = 0.849
    out["idm_majority_baseline"] = 0.232
    d = os.path.join(run_dir, "rollout_metrics_seen")
    dump(os.path.join(d, "drift.json"), out)
    if fvd16 is not None:
        dump(os.path.join(d, "fvd16.json"), {"fvd": float(fvd16), "frames": 16, "num_clips": int(n)})
        dump(os.path.join(d, "fvd32.json"), {"fvd": float(fvd32), "frames": 32, "num_clips": int(n)})
    return psnr, lpips, idm_top1


def write_audit(run_dir, run, rng, psnr, lpips, idm_mean, episodes, tf_rows):
    """audit.json and per_rollout.npz with the rollout_audit.py schema; returns the audit dict for summary.json."""
    n = psnr.shape[0]
    res = {"run": run, "num_rollouts": int(n), "horizon": HORIZON}
    per_rollout = {"episode": episodes}
    sweep = {}
    for h in AUDIT_HORIZONS:
        sweep[str(h)] = {}
        for s in SIGMAS:
            p = psnr[:, h - 1] - 0.12 * s + rng.normal(0, 0.05, size=n)          # blur never helps here
            l = lpips[:, h - 1] + 0.02 * s + rng.normal(0, 0.005, size=n)
            cp = (COPY_SEED_64[0] if h == 64 else COPY_SEED_64[0] + 30.0 / h) - 0.1 * s + rng.normal(0, 1.5, size=n)
            cl = (COPY_SEED_64[1] if h == 64 else COPY_SEED_64[1] - 0.6 / h) + 0.015 * s + rng.normal(0, 0.02, size=n)
            sweep[str(h)][str(s)] = {"psnr": float(p.mean()), "lpips": float(l.mean()),
                                     "copy_seed_psnr": float(cp.mean()), "copy_seed_lpips": float(cl.mean())}
            per_rollout[f"psnr_h{h}_s{s}"] = p
            per_rollout[f"lpips_h{h}_s{s}"] = l
            per_rollout[f"copy_psnr_h{h}_s{s}"] = cp
            per_rollout[f"copy_lpips_h{h}_s{s}"] = cl
    res["blur_sweep"] = sweep
    d = per_rollout["psnr_h64_s0.0"] - per_rollout["psnr_h32_s0.0"]
    res["late_drop"] = {"n": int(n), "mean_psnr32": float(per_rollout["psnr_h32_s0.0"].mean()),
                        "mean_psnr64": float(per_rollout["psnr_h64_s0.0"].mean()), "mean_delta": float(d.mean()),
                        "median_delta": float(np.median(d)), "p10_delta": float(np.percentile(d, 10)),
                        "p90_delta": float(np.percentile(d, 90)), "frac_delta_negative": float((d < 0).mean()),
                        "frac_delta_below_-2db": float((d < -2).mean()),
                        "drop_worst_5pct": {"dropped": 12, "mean_delta": float(d.mean()) + 0.3, "mean_delta_shift": 0.3,
                                            "mean_psnr64_trim_by_delta": float(per_rollout["psnr_h64_s0.0"].mean()) + 0.3,
                                            "mean_psnr64_trim_by_psnr64": float(per_rollout["psnr_h64_s0.0"].mean()) + 0.4,
                                            "mean_psnr64_shift_by_psnr64": 0.4},
                        "drop_worst_10pct": {"dropped": 25, "mean_delta": float(d.mean()) + 0.6, "mean_delta_shift": 0.6,
                                             "mean_psnr64_trim_by_delta": float(per_rollout["psnr_h64_s0.0"].mean()) + 0.6,
                                             "mean_psnr64_trim_by_psnr64": float(per_rollout["psnr_h64_s0.0"].mean()) + 0.7,
                                             "mean_psnr64_shift_by_psnr64": 0.7}}
    per_rollout["idm_top1"] = np.clip(idm_mean + rng.normal(0, 0.1, size=n), 0, 1)
    per_rollout["idm_real_top1"] = np.clip(0.847 + rng.normal(0, 0.05, size=n), 0, 1)

    def ci(v):
        v = np.asarray(v, dtype=np.float64)
        sd = v.std(ddof=1) / np.sqrt(len(v)) * 1.3
        return {"mean": float(v.mean()), "ci95": [float(v.mean() - 1.96 * sd), float(v.mean() + 1.96 * sd)],
                "boot_sd": float(sd), "n": int(len(v))}
    boot = {"rollout": {k: ci(per_rollout[k]) for k in ("psnr_h64_s0.0", "lpips_h64_s0.0", "copy_psnr_h64_s0.0", "idm_top1", "idm_real_top1")}}
    for tag in ("tf_seen", "tf_unseen"):
        boot[tag] = {k: ci([r[k] for r in tf_rows[tag]]) for k in ("psnr_raw", "lpips_raw", "psnr_dec", "lpips_dec", "copy_psnr_raw")}
    res["bootstrap"] = boot
    d = os.path.join(run_dir, "audit")
    dump(os.path.join(d, "audit.json"), res)
    np.savez(os.path.join(d, "per_rollout.npz"), **per_rollout)
    return res, per_rollout, tf_rows


def write_idm(out, rng):
    for name, top1, macro, mov, maj, window in (("idm_aligned", 0.829, 0.548, 0.849, 0.232, 8), ("idm_aligned_k2", 0.701, 0.406, 0.727, 0.227, 2)):
        support = rng.randint(50, 5000, size=29).tolist()
        dump(os.path.join(out, name, "metrics.json"),
             {"top1": top1, "macro_recall": macro, "movement": mov, "majority_baseline": maj, "support": support,
              "train_windows": 120000, "val_windows": 8000, "classes": ["idle", "forward", "back", "left", "right", "turn_l", "turn_r"],
              "args": {"window": window, "steps": 8000}})


def build(out, seed=0):
    rng = np.random.RandomState(seed)
    summary = {"runs": AUDITED, "results_root": out, "horizons": list(AUDIT_HORIZONS), "sigmas": list(SIGMAS),
               "bootstrap_resamples": 10000, "bootstrap_seed": 0, "vae_path": "vae", "idm": "idm_aligned/idm.pt",
               "device": "cuda", "identity": {"all_identical": {"rollout": {"episode_md5": True, "start_md5": True, "actions_md5": True,
                                                                              "seed_md5": True, "gt_md5": True},
                                                                  "tf_seen": {"window_set_md5": True}, "tf_unseen": {"window_set_md5": True}}},
               "per_run": {}, "paired": {}}
    per_rollout_all, tf_all = {}, {}
    rollout_episodes = np.repeat(np.arange(2000, 2064), 4)      # 64 episodes x 4 rollouts, shared across runs
    for run, spec in RUNS.items():
        backbone, params, ps, ls, pu, lu, p64, l64, idm, fvd16, fvd32, vloss, steps, seed_ = spec
        r = os.path.join(out, run)
        dump(os.path.join(r, "config.json"), {"backbone": backbone, "params": params, "steps": steps, "seed": seed_,
                                              "git": "fixture0", "context_frames": 32, "global_batch": 32, "lr": 5e-5,
                                              "world_size": 1, "accum": 1, "torch": "2.5.0"})
        with open(os.path.join(r, "log.jsonl"), "w") as f:
            f.write(json.dumps({"event": "start", "backbone": backbone, "params": params, "world": 1, "accum": 1}) + "\n")
            for s in range(1000, steps + 1, max(1000, steps // 10)):
                v = vloss + (0.6 - vloss) * np.exp(-s / (steps / 4))
                f.write(json.dumps({"event": "val", "step": s, "val_loss": float(v), "val_loss_by_t_quartile": [v] * 4, "excursion": False}) + "\n")
            f.write(json.dumps({"event": "val", "step": steps, "val_loss": float(vloss), "val_loss_by_t_quartile": [vloss] * 4, "excursion": False}) + "\n")
            f.write(json.dumps({"event": "end", "step": steps}) + "\n")
        tf_rows = {}
        for corpus, pm, lm in (("seen", ps, ls), ("unseen", pu, lu)):
            rows = tf_windows(np.random.RandomState(seed + hash(corpus) % 1000), corpus, 512, pm, lm)
            # the model-dependent columns get this run's numbers; the shared columns stay identical across runs
            write_tf(os.path.join(r, f"eval_tf_{corpus}"), rows, steps)
            tf_rows[f"tf_{corpus}"] = rows
        if run not in NO_EMA:
            gain = 0.31 if backbone == "unet" else 0.0
            rows = tf_windows(rng, "seen", 512, ps + gain, ls - (0.02 if backbone == "unet" else 0.005))
            write_tf(os.path.join(r, "eval_tf_seen_ema"), rows, steps)
        if run not in NO_ROLLOUT:
            psnr, lpips, idm_top1 = write_rollout(r, rng, spec, rollout_episodes)
            if run not in NO_AUDIT:
                res, pr, _ = write_audit(r, run, rng, psnr, lpips, idm, rollout_episodes, tf_rows)
                summary["per_run"][run] = res
                per_rollout_all[run], tf_all[run] = pr, tf_rows
    for dit in ("030-dit-l32-aligned", "032-dit-l32-aligned-seed1"):
        unet = "031-unet-l32-aligned"
        entry = {"rollout": {}, "tf_seen": {}, "tf_unseen": {}}
        for k in ("psnr_h64_s0.0", "lpips_h64_s0.0", "idm_top1"):
            diff = float(per_rollout_all[unet][k].mean() - per_rollout_all[dit][k].mean())
            entry["rollout"][k] = {"diff": diff, "ci95": [diff - 0.5, diff + 0.5], "boot_sd": 0.25, "excludes_zero": abs(diff) > 0.5}
        for tag in ("tf_seen", "tf_unseen"):
            for m in ("psnr_raw", "lpips_raw"):
                diff = float(np.mean([x[m] for x in tf_all[unet][tag]]) - np.mean([x[m] for x in tf_all[dit][tag]]))
                w = 0.3 if m == "psnr_raw" else 0.01
                entry[tag][m] = {"diff": diff, "ci95": [diff - w, diff + w], "boot_sd": w / 2, "excludes_zero": abs(diff) > w}
        summary["paired"][f"{unet}_minus_{dit}"] = entry
    dump(os.path.join(out, "tmp", "audit", "summary.json"), summary)
    write_idm(out, rng)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print("WROTE", build(a.out, a.seed))
