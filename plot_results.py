"""
Figures for the paper from the evaluation outputs.

    python plot_results.py --runs results/010-dit-l32:DiT-XL/2 results/011-unet-l32:U-Net --out figures/
Reads <run>/rollout_metrics/drift.json (drift curves, IDM per horizon) and <run>/log.jsonl
(train and val loss) and writes drift_psnr.pdf, drift_lpips.pdf, idm_horizon.pdf, loss.pdf.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"DiT-XL/2": "#8B1E1E", "U-Net": "#2B5C8A"}


def main(args):
    os.makedirs(args.out, exist_ok=True)
    runs = [r.split(":") for r in args.runs]
    fig_p, ax_p = plt.subplots(figsize=(4.2, 3)); fig_l, ax_l = plt.subplots(figsize=(4.2, 3)); fig_i, ax_i = plt.subplots(figsize=(4.2, 3))
    fig_t, ax_t = plt.subplots(figsize=(4.2, 3))
    for path, name in runs:
        c = COLORS.get(name)
        d = os.path.join(path, "rollout_metrics", "drift.json")
        if os.path.exists(d):
            d = json.load(open(d)); h = d["horizon"]
            ax_p.plot(h, d["psnr"], label=name, color=c); ax_l.plot(h, d["lpips"], label=name, color=c)
            if "copy_seed_psnr" in d and name == runs[0][1]:
                ax_p.plot(h, d["copy_seed_psnr"], label="copy seed frame", color="#888", ls="--")
            if "idm_top1" in d:
                ax_i.plot(h, d["idm_movement"], label=f"{name} (movement)", color=c)
                ax_i.plot(h, d["idm_top1"], color=c, ls=":", label=f"{name} (top-1)")
                if d.get("idm_ceiling_movement") and name == runs[0][1]:
                    ax_i.axhline(d["idm_ceiling_movement"], color="#888", ls="--", label="IDM ceiling (real)")
        lg = os.path.join(path, "log.jsonl")
        if os.path.exists(lg):
            rows = [json.loads(l) for l in open(lg)]
            tr = [(r["step"], r["loss"]) for r in rows if r.get("event") == "train"]
            va = [(r["step"], r["val_loss"]) for r in rows if r.get("event") == "val"]
            if tr: ax_t.plot(*zip(*tr), color=c, alpha=0.5, lw=0.8, label=f"{name} train")
            if va: ax_t.plot(*zip(*va), color=c, marker="o", ms=3, label=f"{name} val")
    for ax, yl, fig, fn in [(ax_p, "PSNR (dB)", fig_p, "drift_psnr.pdf"), (ax_l, "LPIPS", fig_l, "drift_lpips.pdf"),
                            (ax_i, "action agreement", fig_i, "idm_horizon.pdf"), (ax_t, "v-loss", fig_t, "loss.pdf")]:
        ax.set_xlabel("rollout horizon (decision frames)" if fn != "loss.pdf" else "step"); ax.set_ylabel(yl); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, fn)); fig.savefig(os.path.join(args.out, fn.replace(".pdf", ".png")), dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--out", default="figures")
    main(p.parse_args())
