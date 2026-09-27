"""
One-tic PSNR on the training maps' validation windows against updates, for the three backbones.

    python tools/training_curves.py
    python tools/training_curves.py --curves-dir results/training_curves --out-dir paper/figures

**Input.** One JSON per backbone at `<curves-dir>/<backbone>.json`, backbone in `BACKBONES` order:

    {"backbone": "unet" | "pixart" | "sd35", "label": str,
     "persistence": {"psnr": float, "lpips": float},
     "reads": [{"step": int, "weights": "ema" | "live", "psnr": float, "lpips": float[, "note": str | null]}, ...]}

Other top-level keys (the steward's `source`) are provenance and are not drawn. Other files in the directory,
such as the steward's combined `training_curves_h1.json`, are not read: only the three backbone names are.
A missing backbone file is skipped; a file whose `backbone` disagrees with its name, a read with unknown
weights, a non-finite value, a negative step, or two reads of one step and weight set stop the run.

**Output.** `<out-dir>/fig1_curves.pdf` and `.png`, drawn at `PANEL_SIZE` with `paper/make_adapt_figures.py`'s
`style()`, `new_figure()` and `save()` so it prints at the paper's type sizes, in its backbone colours
(U-Net, PixArt, SD 3.5 as in Figure 2): EMA reads solid, live reads dotted where present, persistence dashed,
x in thousands of updates. A read carrying a `note` (SD 3.5's provisional 170k read from another scorer, for
example) is drawn as an open marker and not joined to its line. The y axis starts just below the lowest live
read and persistence, so the EMA's warm-up reads (the fp32 EMA at 0.9999 still averages the pretrained start
for the first few thousand updates, 7 to 12 dB at 5k) enter from below the frame rather than flattening it.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "paper"))
from make_adapt_figures import INK, MARKERS, MUTED, SECONDARY, SERIES, new_figure, save, style  # noqa: E402

from matplotlib.lines import Line2D  # noqa: E402

BACKBONES = ("unet", "pixart", "sd35")
WEIGHTS = ("ema", "live")
LINESTYLES = {"ema": "-", "live": ":"}
PANEL_SIZE = (1.7, 1.15)
STEM = "fig1_curves"
DEFAULT_CURVES_DIR = os.path.join(REPO, "results", "training_curves")
DEFAULT_OUT_DIR = os.path.join(REPO, "paper", "figures")
FLOOR_MARGIN = 0.35     # dB below the lowest live read or persistence
CEILING_MARGIN = 0.15   # dB above the highest drawn read


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _check_read(path, i, r):
    where = f"{path}: read {i}"
    if r.get("weights") not in WEIGHTS:
        raise SystemExit(f"{where}: weights {r.get('weights')!r} is not one of {WEIGHTS}")
    step = r.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise SystemExit(f"{where}: step {step!r} is not a non-negative integer")
    for key in ("psnr", "lpips"):
        if not _finite(r.get(key)):
            raise SystemExit(f"{where}: {key} {r.get(key)!r} is not a finite number")


def load_curve(path, backbone):
    """One backbone's curve file, checked. Reads are returned sorted by (weights, step)."""
    with open(path) as f:
        body = json.load(f)
    if body.get("backbone") != backbone:
        raise SystemExit(f"{path}: backbone {body.get('backbone')!r} disagrees with the file name {backbone}.json")
    pers = body.get("persistence") or {}
    if not (_finite(pers.get("psnr")) and _finite(pers.get("lpips"))):
        raise SystemExit(f"{path}: persistence needs finite psnr and lpips, got {pers!r}")
    reads = body.get("reads") or []
    seen = set()
    for i, r in enumerate(reads):
        _check_read(path, i, r)
        key = (r["weights"], r["step"])
        if key in seen:
            raise SystemExit(f"{path}: two {r['weights']} reads at step {r['step']}")
        seen.add(key)
    return {"backbone": backbone, "label": body.get("label") or backbone, "path": path,
            "persistence": {"psnr": float(pers["psnr"]), "lpips": float(pers["lpips"])},
            "reads": sorted(reads, key=lambda r: (r["weights"], r["step"]))}


def load_curves(curves_dir):
    """Every present `<backbone>.json` under `curves_dir`, in `BACKBONES` order."""
    curves = []
    for b in BACKBONES:
        path = os.path.join(curves_dir, f"{b}.json")
        if os.path.exists(path):
            curves.append(load_curve(path, b))
        else:
            print(f"training_curves: no {b}.json in {curves_dir}; {b} is not drawn")
    if not curves:
        raise SystemExit(f"no curve file ({', '.join(b + '.json' for b in BACKBONES)}) in {curves_dir}")
    return curves


def series(curve, weights, noted=False):
    """{"k_updates", "psnr"} of one weight set, in step order: the joined reads, or with `noted` the reads
    that carry a note and are drawn apart."""
    rs = [r for r in curve["reads"] if r["weights"] == weights and bool(r.get("note")) == noted]
    return {"k_updates": [r["step"] / 1000.0 for r in rs], "psnr": [float(r["psnr"]) for r in rs]}


def y_limits(curves):
    """(floor, ceiling) in dB: the floor from the live reads and persistence, the ceiling from every read."""
    anchors = [c["persistence"]["psnr"] for c in curves]
    anchors += [r["psnr"] for c in curves for r in c["reads"] if r["weights"] == "live"]
    if all(r["weights"] == "ema" for c in curves for r in c["reads"]):
        # no live reads anywhere: the late half of each EMA curve stands in for them
        for c in curves:
            ema = [r["psnr"] for r in c["reads"] if r["weights"] == "ema"]
            anchors += ema[len(ema) // 2:]
    top = max([r["psnr"] for c in curves for r in c["reads"]] + anchors)
    return math.floor((min(anchors) - FLOOR_MARGIN) * 2) / 2, top + CEILING_MARGIN


def display_label(label):
    """The file's label as printed: Greek alpha for PixArt, as the paper writes it."""
    return label.replace("-alpha", "-\u03b1")


def draw(curves, size=PANEL_SIZE):
    """The panel: (figure, axes). Colours and markers follow the backbone's slot in `BACKBONES`."""
    style()
    fig, (ax,) = new_figure(size)
    lo, hi = y_limits(curves)
    pers = {round(c["persistence"]["psnr"], 6) for c in curves}
    for c in curves:
        slot = BACKBONES.index(c["backbone"])
        colour, marker = SERIES[slot % len(SERIES)], MARKERS[slot % len(MARKERS)]
        for w in WEIGHTS:
            s = series(c, w)
            if s["k_updates"]:
                ax.plot(s["k_updates"], s["psnr"], color=colour, ls=LINESTYLES[w], lw=0.9 if w == "ema" else 0.8,
                        zorder=3 if w == "ema" else 2, label=f"{c['label']} {w.upper() if w == 'ema' else w}")
            apart = series(c, w, noted=True)
            if apart["k_updates"]:
                ax.plot(apart["k_updates"], apart["psnr"], ls="none", marker=marker, ms=2.6, mfc="white",
                        mec=colour, mew=0.6, zorder=4, label=f"{c['label']} {w} (noted)")
        if len(pers) > 1:
            # different window sets: each backbone's own persistence, in its colour
            ax.axhline(c["persistence"]["psnr"], color=colour, lw=0.6, ls="--", zorder=1, label="persistence")
    if len(pers) == 1:
        ax.axhline(curves[0]["persistence"]["psnr"], color=MUTED, lw=0.7, ls="--", zorder=1, label="persistence")
        ax.text(0.98, curves[0]["persistence"]["psnr"] + 0.03, "persistence", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=5.5, color=SECONDARY)
    right = max(r["step"] for c in curves for r in c["reads"]) / 1000.0
    ax.set_xlim(0, right * 1.02)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("updates (thousands)")
    ax.set_ylabel("one-tic PSNR (dB)")
    handles = [Line2D([], [], color=SERIES[BACKBONES.index(c["backbone"]) % len(SERIES)], lw=1.0,
                      label=display_label(c["label"])) for c in curves]
    handles += [Line2D([], [], color=INK, lw=0.9, ls="-", label="EMA")]
    if any(r["weights"] == "live" for c in curves for r in c["reads"]):
        handles += [Line2D([], [], color=INK, lw=0.8, ls=":", label="live")]
    ax.legend(handles=handles, loc="lower right", ncol=2, fontsize=5, handlelength=1.4, columnspacing=0.8,
              handletextpad=0.4, labelspacing=0.2)
    return fig, ax


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--curves-dir", default=DEFAULT_CURVES_DIR, help="directory holding <backbone>.json")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--stem", default=STEM)
    a = p.parse_args(argv)
    curves = load_curves(a.curves_dir)
    fig, _ = draw(curves)
    for path in save(fig, a.out_dir, a.stem):
        print("wrote", path)
    for c in curves:
        for w in WEIGHTS:
            s = series(c, w)
            if s["k_updates"]:
                print(f"{c['backbone']} {w}: {len(s['k_updates'])} reads, {s['k_updates'][0]:g}k to "
                      f"{s['k_updates'][-1]:g}k, last {s['psnr'][-1]:.2f} dB")
        for r in c["reads"]:
            if r.get("note"):
                print(f"{c['backbone']} {r['weights']} {r['step']}: drawn apart ({r['note']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
