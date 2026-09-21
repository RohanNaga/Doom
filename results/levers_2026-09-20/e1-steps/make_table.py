"""Table and plot for the DDIM step-count sweep.

Reads the `metrics.json` each `eval_tf.py` run wrote and reports PSNR and LPIPS against the number
of denoising steps, per row and per corpus, next to the copy-last floor and the decoder ceiling
that bracket them. The 50-step column is the reproduction check: it is the same protocol as the
stored 2,048-window number, on a 512-window subsample, so the two must agree inside their combined
standard error before any of the other columns means anything.

    python make_table.py --dir .
"""
import argparse
import glob
import json
import os

ROW_NAME = {"unet": "SD 1.4 U-Net (031-unet-l32-aligned)", "pixart": "PixArt-alpha 512 (033-pixart-l32-aligned)"}

# The stored numbers each row was scored with in after_run2.sh: 2,048 windows, 50 steps, seed 0,
# tuned 4-channel decoder, raw lossless frames as the target.
STORED = {
    ("unet", "seen"): {"psnr": (21.3636, 0.0755), "lpips": (0.2705, 0.0022), "copy": 19.4083, "ceiling": 28.6132},
    ("unet", "unseen"): {"psnr": (19.1348, 0.0481), "lpips": (0.4464, 0.0012), "copy": 18.4945, "ceiling": 26.3556},
    ("pixart", "seen"): {"psnr": (21.3513, 0.0756), "lpips": (0.2723, 0.0022), "copy": 19.4083, "ceiling": 28.6132},
    ("pixart", "unseen"): {"psnr": (19.3921, 0.0503), "lpips": (0.4337, 0.0013), "copy": 18.4945, "ceiling": 26.3556},
}


def load(directory):
    """{(backbone, corpus): {steps: metrics}} from <backbone>_<corpus>_s<steps>/metrics.json."""
    out = {}
    for p in sorted(glob.glob(os.path.join(directory, "*_s*", "metrics.json"))):
        tag = os.path.basename(os.path.dirname(p))
        backbone, corpus, s = tag.rsplit("_", 2)
        out.setdefault((backbone, corpus), {})[int(s[1:])] = json.load(open(p))
    return out


def mean(d, key):
    v = d.get(key)
    return None if v is None else v["mean"]


def sem(d, key):
    v = d.get(key)
    return None if v is None else v["sem"]


def fmt(x, nd=2):
    return "--" if x is None else f"{x:.{nd}f}"


def agrees(a, sa, b, sb):
    """Whether two means agree inside two standard errors of their difference."""
    if None in (a, sa, b, sb):
        return None
    return abs(a - b) <= 2 * (sa ** 2 + sb ** 2) ** 0.5


def table(data):
    lines = ["# How many DDIM steps the teacher-forced numbers need", "",
             "512 windows, seed 0, live weights, tuned 4-channel decoder, scored against the raw",
             "lossless frame. Everything except `--num-windows` and `--steps` is the argument list",
             "`after_run2.sh` scored the row with.", ""]
    for key in sorted(data):
        bb, corpus = key
        runs, ref = data[key], STORED.get(key, {})
        lines += [f"### {ROW_NAME.get(bb, bb)} -- {corpus}", "",
                  "| steps | PSNR | +/- | LPIPS | +/- | HUD PSNR | frames/s | n |",
                  "|---|---|---|---|---|---|---|---|"]
        for s in sorted(runs):
            d = runs[s]
            lines.append(f"| {s} | {fmt(mean(d, 'psnr_raw'))} | {fmt(sem(d, 'psnr_raw'), 3)} "
                         f"| {fmt(mean(d, 'lpips_raw'), 4)} | {fmt(sem(d, 'lpips_raw'), 4)} "
                         f"| {fmt(mean(d, 'hud_psnr_raw'))} | {fmt(d.get('sampling_frames_per_s'), 2)} "
                         f"| {d['psnr_raw']['n'] if d.get('psnr_raw') else '--'} |")
        floor = mean(runs[max(runs)], "copy_psnr_raw") if runs else None
        ceiling = mean(runs[max(runs)], "vae_psnr") if runs else None
        lines += [f"| _copy-last floor_ | {fmt(floor)} | | | | | | |",
                  f"| _decoder ceiling_ | {fmt(ceiling)} | | {fmt(mean(runs[max(runs)], 'vae_lpips'), 4)} | | | | |", ""]

        if 50 in runs and ref:
            got, gs = mean(runs[50], "psnr_raw"), sem(runs[50], "psnr_raw")
            ok = agrees(got, gs, *ref["psnr"])
            lines.append(f"Reproduction at 50 steps: {fmt(got)} +/- {fmt(gs, 3)} dB here against the stored "
                         f"2,048-window {ref['psnr'][0]:.2f} +/- {ref['psnr'][1]:.3f} dB -- "
                         f"{'agrees' if ok else 'DOES NOT AGREE'} inside two standard errors of the difference.")
            best = max(runs, key=lambda s: mean(runs[s], "psnr_raw") or -1e9)
            at50 = mean(runs[50], "psnr_raw")
            for s in (4, 8):
                if s in runs:
                    lines.append(f"At {s} steps: {mean(runs[s], 'psnr_raw') - at50:+.3f} dB and "
                                 f"{mean(runs[s], 'lpips_raw') - mean(runs[50], 'lpips_raw'):+.4f} LPIPS "
                                 f"against 50 steps.")
            lines += [f"Best of the sweep: {best} steps.", ""]
    return "\n".join(lines)


def plot(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted(data)
    fig, axes = plt.subplots(2, len(keys), figsize=(3.6 * len(keys), 6.4), squeeze=False)
    for col, key in enumerate(keys):
        runs = data[key]
        xs = sorted(runs)
        last = runs[max(runs)]
        for r, metric in enumerate(("psnr_raw", "lpips_raw")):
            ax = axes[r][col]
            ax.plot(xs, [mean(runs[s], metric) for s in xs], "o-", lw=1.6)
            if r == 0:
                ax.axhline(mean(last, "copy_psnr_raw"), color="k", ls="--", lw=1.0, label="copy-last floor")
                ax.axhline(mean(last, "vae_psnr"), color="g", ls=":", lw=1.0, label="decoder ceiling")
            else:
                ax.axhline(mean(last, "vae_lpips"), color="g", ls=":", lw=1.0, label="decoder ceiling")
            ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([str(s) for s in xs])
            ax.set_ylabel("PSNR (dB)" if r == 0 else "LPIPS")
            ax.set_xlabel("denoising steps"); ax.grid(alpha=.3)
            ax.legend(fontsize=7, frameon=False)
        axes[0][col].set_title(f"{key[0]} -- {key[1]}", fontsize=9)
    fig.suptitle("Teacher-forced quality against DDIM step count (512 windows)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--out-table", default="TABLE.md")
    p.add_argument("--out-plot", default="psnr_vs_steps.png")
    a = p.parse_args()
    d = load(a.dir)
    if not d:
        raise SystemExit(f"no */metrics.json under {a.dir}")
    t = table(d)
    open(os.path.join(a.dir, a.out_table), "w").write(t + "\n")
    print(t)
    plot(d, os.path.join(a.dir, a.out_plot))
