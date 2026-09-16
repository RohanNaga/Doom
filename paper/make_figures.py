"""
Build every figure of the paper from the result directories (vector PDF, column-width sizing).

    python paper/make_figures.py --results-root paper/results_mirror --out paper/figures [--png]

Figures:
  drift_psnr.pdf, drift_lpips.pdf, drift.pdf   Figure 1: PSNR and LPIPS versus rollout horizon 1..64 for
                                               every row in the "drift" group, copy-seed dashed, episode-bootstrap
                                               bands (continuous when per-horizon per-rollout data exists, whiskers
                                               at the audited horizons 8/32/64 otherwise).
  tf_seen_unseen.pdf                           Figure 2: teacher-forced seen vs unseen PSNR and LPIPS per row with
                                               95% episode-bootstrap CIs from audit.json (falls back to 1.96 SEM,
                                               drawn hollow), plus the per-map scatter from metrics.json per_map.
  blur_control.pdf                             Figure 3 (appendix): blur-control sweep from audit.json.

Every number is read from the files; a missing artifact leaves its row out of the panel and
prints a warning, and the figure is still written so the paper compiles.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperdata as pd  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito, colour-blind safe; copy-seed and references are grey/black.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
GREY = "#555555"
COL_W = 3.25          # inches, one column of a two-column page
FULL_W = 6.75
AUDIT_H = (8, 32, 64)

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 6, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "lines.linewidth": 1.0, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
})


def colour_for(rows):
    return {r.key: PALETTE[i % len(PALETTE)] for i, r in enumerate(rows)}


def save(fig, out, name, png):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    if png:
        fig.savefig(os.path.join(out, name + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", os.path.join(out, name + ".pdf"))


def empty_note(ax, text):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color=GREY)


# ---------------------------------------------------------------------------------------------
# Figure 1: drift curves
# ---------------------------------------------------------------------------------------------

def band(run, metric, n_boot):
    """(h, lo, hi) arrays for a continuous per-horizon bootstrap band, or None."""
    pr = run.per_rollout
    if pr is None or metric not in pr or np.ndim(pr[metric]) != 2 or "episode" not in pr:
        return None
    vals, eps = pr[metric], pr["episode"]
    draws = pd.episode_resamples(eps, n_boot, 0)
    samples = np.array([vals[d].mean(0) for d in draws])
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    return np.arange(1, vals.shape[1] + 1), lo, hi


def draw_drift(ax, rows, metric, colours, n_boot, legend=True):
    drawn = 0
    copy_done = False
    for r in rows:
        curve = r.rollout_curve(metric)
        if curve is None:
            continue
        h = np.arange(1, len(curve) + 1)
        c = colours[r.key]
        ax.plot(h, curve, color=c, label=r.short)
        b = band(r, metric, n_boot)
        if b is not None:
            ax.fill_between(b[0], b[1], b[2], color=c, alpha=0.18, linewidth=0)
        else:
            for hh in AUDIT_H:
                ci = r.rollout_ci(metric, hh, n_boot=n_boot)
                if ci and hh <= len(curve):
                    ax.errorbar([hh], [curve[hh - 1]], yerr=[[curve[hh - 1] - ci[0]], [ci[1] - curve[hh - 1]]],
                                color=c, fmt="o", ms=2, capsize=1.5, elinewidth=0.7)
        if not copy_done:
            if metric == "psnr":
                cs = r.rollout_curve("copy_seed_psnr")
                if cs is not None:
                    ax.plot(np.arange(1, len(cs) + 1), cs, color=GREY, ls="--", label="copy-seed")
                    copy_done = True
            else:
                pts = [(hh, r.copy_seed_lpips(hh)) for hh in AUDIT_H if r.copy_seed_lpips(hh) is not None]
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=GREY, ls="--", marker="s", ms=2, label="copy-seed (audited horizons)")
                    copy_done = True
        drawn += 1
    ax.set_xlabel("rollout horizon (frames)")
    ax.set_ylabel("PSNR (dB)" if metric == "psnr" else "LPIPS")
    ax.set_xlim(1, 64)
    ax.set_xticks([1, 8, 16, 32, 48, 64])
    if drawn == 0:
        empty_note(ax, "no drift.json found")
        pd.warn(f"figure 1 ({metric}): no row has rollout_metrics_seen/drift.json")
    elif legend:
        ax.legend(loc="best", ncol=1, handlelength=1.6)
    return drawn


def figure_drift(rows, out, png, n_boot):
    colours = colour_for(rows)
    for metric in ("psnr", "lpips"):
        fig, ax = plt.subplots(figsize=(COL_W, 2.1))
        draw_drift(ax, rows, metric, colours, n_boot)
        save(fig, out, f"drift_{metric}", png)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.1))
    draw_drift(axes[0], rows, "psnr", colours, n_boot, legend=True)
    draw_drift(axes[1], rows, "lpips", colours, n_boot, legend=False)
    axes[0].set_title("(a) PSNR vs. horizon, seen maps", loc="left")
    axes[1].set_title("(b) LPIPS vs. horizon, seen maps", loc="left")
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "drift", png)


# ---------------------------------------------------------------------------------------------
# Figure 2: teacher-forced seen vs unseen, and per-map scatter
# ---------------------------------------------------------------------------------------------

def tf_point(run, corpus, key):
    """(mean, lo, hi, from_audit) or None."""
    m = run.tf_mean(corpus, key)
    if m is None:
        return None
    ci = run.tf_ci(corpus, key)
    if ci:
        return m, ci[0], ci[1], True
    sem = pd.get(run.tf(corpus), key, "sem")
    if sem is None:
        return m, m, m, False
    return m, m - 1.96 * sem, m + 1.96 * sem, False


def draw_seen_unseen(ax, rows, key, colours):
    labels, drawn = [], 0
    for i, r in enumerate(rows):
        pts = [tf_point(r, c, key) for c in ("seen", "unseen")]
        if all(p is None for p in pts):
            continue
        labels.append((i, r.short))
        for j, (p, marker) in enumerate(zip(pts, ("o", "^"))):
            if p is None:
                continue
            m, lo, hi, from_audit = p
            x = i + (-0.17 if j == 0 else 0.17)
            ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt=marker, ms=3.2, color=colours[r.key],
                        mfc=colours[r.key] if from_audit else "white", capsize=1.5, elinewidth=0.7)
        drawn += 1
    ax.set_xticks([i for i, _ in labels])
    ax.set_xticklabels([s for _, s in labels], rotation=25, ha="right")
    ax.set_ylabel("PSNR (dB)" if "psnr" in key else "LPIPS")
    if drawn == 0:
        empty_note(ax, "no eval_tf metrics found")
    return drawn


def draw_per_map(ax, rows, key, colours):
    drawn = 0
    for r in rows:
        seen, unseen = r.per_map("seen", key), r.per_map("unseen", key)
        if not seen and not unseen:
            continue
        c = colours[r.key]
        if seen:
            ax.scatter(list(seen), list(seen.values()), color=c, marker="o", s=9, label=r.short, zorder=3)
        if unseen:
            ax.scatter(list(unseen), list(unseen.values()), color=c, marker="^", s=12, zorder=3)
        drawn += 1
    if not drawn:
        empty_note(ax, "no per_map in metrics.json")
    ax.set_xlabel("map id")
    ax.set_ylabel("PSNR (dB), decoded GT" if "psnr" in key else "LPIPS, decoded GT")
    return drawn


def figure_tf(rows, out, png):
    colours = colour_for(rows)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.0), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    n_a = draw_seen_unseen(axes[0], rows, "psnr_raw", colours)
    draw_seen_unseen(axes[1], rows, "lpips_raw", colours)
    n_c = draw_per_map(axes[2], rows, "psnr_dec", colours)
    axes[0].set_title("(a) teacher-forced PSNR", loc="left")
    axes[1].set_title("(b) teacher-forced LPIPS", loc="left")
    axes[2].set_title("(c) per-map PSNR", loc="left")
    # one legend for all panels, below them, so no legend covers a data point
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colours[r.key], marker="s", ls="", ms=4, label=r.short)
               for r in rows if r.tf("seen") is not None or r.tf("unseen") is not None]
    if n_a or n_c:
        handles += [Line2D([], [], color=GREY, marker="o", ls="", ms=4, label="seen / training map"),
                    Line2D([], [], color=GREY, marker="^", ls="", ms=4.5, label="unseen map"),
                    Line2D([], [], color=GREY, marker="o", mfc="white", ls="", ms=4, label="hollow: 1.96 SEM, no audit CI")]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), bbox_to_anchor=(0.5, -0.01),
                   handlelength=1.0, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(w_pad=1.2, rect=[0, 0.14, 1, 1])
    save(fig, out, "tf_seen_unseen", png)


# ---------------------------------------------------------------------------------------------
# Figure 3: blur-control sweep
# ---------------------------------------------------------------------------------------------

def figure_blur(rows, out, png):
    colours = colour_for(rows)
    sigmas = None
    for r in rows:
        sweep = pd.get(r.audit, "blur_sweep")
        if sweep:
            sigmas = sorted(float(s) for s in next(iter(sweep.values())).keys())
            break
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 3.4), sharex=True)
    for col, h in enumerate((8, 64)):
        for row_i, (metric, copy_key) in enumerate((("psnr", "copy_seed_psnr"), ("lpips", "copy_seed_lpips"))):
            ax = axes[row_i, col]
            drawn = 0
            if sigmas:
                copy_done = False
                for r in rows:
                    ys = [r.blur(h, s, metric) for s in sigmas]
                    if all(y is None for y in ys):
                        continue
                    ax.plot(sigmas, ys, marker="o", ms=2.5, color=colours[r.key], label=r.short)
                    drawn += 1
                    if not copy_done:
                        cs = [r.blur(h, s, copy_key) for s in sigmas]
                        if not all(c is None for c in cs):
                            ax.plot(sigmas, cs, ls="--", marker="s", ms=2, color=GREY, label="copy-seed")
                            copy_done = True
            if drawn == 0:
                empty_note(ax, "no blur_sweep in audit.json")
            ax.set_title(f"{'PSNR' if metric == 'psnr' else 'LPIPS'} at horizon {h}", loc="left")
            if sigmas:
                ax.set_xticks(sigmas)
            if row_i == 1:
                ax.set_xlabel("Gaussian blur sigma on predictions (px)")
            ax.set_ylabel("PSNR (dB)" if metric == "psnr" else "LPIPS")
    if sigmas:
        axes[0, 0].legend(loc="best", handlelength=1.5)
    else:
        pd.warn("figure 3: no row has audit.json with blur_sweep")
    fig.tight_layout(w_pad=1.2, h_pad=0.8)
    save(fig, out, "blur_control", png)


def main(args):
    reg = pd.load_registry(args.rows)
    figure_drift(pd.load_rows(args.results_root, reg, "drift"), args.out, args.png, args.boot)
    figure_tf(pd.load_rows(args.results_root, reg, "tf"), args.out, args.png)
    figure_blur(pd.load_rows(args.results_root, reg, "blur"), args.out, args.png)
    if pd.warn.seen:
        print(f"[paper] {len(pd.warn.seen)} warning(s); see stderr", file=sys.stderr)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True, help="local mirror of results_spiderman (see sync_results.sh)")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
    p.add_argument("--rows", default=pd.ROWS_JSON)
    p.add_argument("--png", action="store_true", help="also write 300 dpi PNG previews")
    p.add_argument("--boot", type=int, default=2000, help="episode-bootstrap resamples for figure bands")
    main(p.parse_args())
