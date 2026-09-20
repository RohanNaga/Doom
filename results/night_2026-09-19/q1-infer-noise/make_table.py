"""Table and plot for the inference-noise sweep.

Reads the `drift.json` each scored setting produced and writes one markdown table plus one figure of
PSNR against horizon, a line per noise level, a panel per row. Run it from the worktree over the
mirrored metrics directory; it needs nothing from the server.

    python make_table.py --dir . --out-table table.md --out-plot psnr_vs_horizon.png
"""
import argparse
import glob
import json
import os

HORIZONS = (1, 8, 16, 32, 64)
STORED = {"unet": 16.024523723870516, "pixart": 17.18336133658886}   # 256-window runs, infer-noise 0
ROW_NAME = {"unet": "SD 1.4 U-Net (031)", "pixart": "PixArt-alpha 512 (033)"}


def load(directory):
    out = {}
    for p in sorted(glob.glob(os.path.join(directory, "*_n*", "drift.json"))):
        tag = os.path.basename(os.path.dirname(p))
        backbone, noise = tag.rsplit("_n", 1)
        d = json.load(open(p))
        prov = os.path.join(os.path.dirname(p), "provenance.json")
        d["_provenance"] = json.load(open(prov)) if os.path.exists(prov) else {}
        # FVD is only computed where clips were kept (the 0.0 reference and the best level)
        for f in (16, 32):
            fp = os.path.join(os.path.dirname(p), f"fvd{f}.json")
            d[f"fvd{f}"] = json.load(open(fp))["fvd"] if os.path.exists(fp) else None
        out.setdefault(backbone, {})[float(noise)] = d
    return out


def at(d, key, h):
    """`key` at horizon `h`, 1-based, from the per-horizon list."""
    v = d.get(key)
    return v[h - 1] if isinstance(v, list) and len(v) >= h else None


def fmt(x, nd=2):
    return "--" if x is None else f"{x:.{nd}f}"


def table(data):
    lines = []
    for bb in sorted(data):
        rows = data[bb]
        copy = at(next(iter(rows.values())), "copy_seed_psnr", 64)
        lines += [f"### {ROW_NAME.get(bb, bb)}", ""]
        lines.append("| infer-noise | " + " | ".join(f"PSNR@{h}" for h in HORIZONS)
                     + " | " + " | ".join(f"LPIPS@{h}" for h in HORIZONS)
                     + " | IDM top-1 | FVD16 | FVD32 | n |")
        lines.append("|---" * (2 + 2 * len(HORIZONS) + 3) + "|")
        for nz in sorted(rows):
            d = rows[nz]
            lines.append(
                f"| {nz:g} | " + " | ".join(fmt(at(d, "psnr", h)) for h in HORIZONS)
                + " | " + " | ".join(fmt(at(d, "lpips", h), 3) for h in HORIZONS)
                + f" | {fmt(d.get('idm_top1_mean'), 3)} | {fmt(d.get('fvd16'), 1)}"
                + f" | {fmt(d.get('fvd32'), 1)} | {d.get('num_rollouts', '--')} |")
        lines.append("| _copy-seed reference_ | " + " | ".join(
            fmt(at(next(iter(rows.values())), "copy_seed_psnr", h)) for h in HORIZONS)
            + " | " + " | ".join("--" for _ in HORIZONS) + " | -- | -- | -- | -- |")
        best = max(rows, key=lambda k: at(rows[k], "psnr", 64) or -1e9)
        base = at(rows.get(0.0, {}), "psnr", 64)
        top = at(rows[best], "psnr", 64)
        lines += ["", f"Stored 256-window run at 0.0: PSNR@64 {STORED.get(bb, float('nan')):.2f} dB. "
                      f"Copy-seed at 64: {fmt(copy)} dB. "
                      f"Best level here: {best:g} at {fmt(top)} dB"
                      + (f", {top - base:+.2f} dB against 0.0." if base is not None else "."), ""]
    return "\n".join(lines)


def plot(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bbs = sorted(data)
    fig, axes = plt.subplots(1, len(bbs), figsize=(5.2 * len(bbs), 4.0), sharey=True, squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ax, bb in zip(axes[0], bbs):
        rows = data[bb]
        levels = sorted(rows)
        for i, nz in enumerate(levels):
            d = rows[nz]
            ax.plot(d["horizon"], d["psnr"], color=cmap(i / max(len(levels) - 1, 1)),
                    label=f"infer-noise {nz:g}", lw=1.6)
        ref = next(iter(rows.values()))
        ax.plot(ref["horizon"], ref["copy_seed_psnr"], "k--", lw=1.2, label="copy-seed")
        ax.set_title(ROW_NAME.get(bb, bb)); ax.set_xlabel("horizon (decision frames)")
        ax.grid(alpha=.3)
    axes[0][0].set_ylabel("PSNR (dB)")
    axes[0][-1].legend(fontsize=8, frameon=False)
    fig.suptitle("Rollout PSNR against horizon by inference-time context-noise level, seen split")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--out-table", default="table.md")
    p.add_argument("--out-plot", default="psnr_vs_horizon.png")
    a = p.parse_args()
    data = load(a.dir)
    if not data:
        raise SystemExit(f"no */drift.json under {a.dir}")
    t = table(data)
    open(os.path.join(a.dir, a.out_table), "w").write(t + "\n")
    print(t)
    plot(data, os.path.join(a.dir, a.out_plot))
