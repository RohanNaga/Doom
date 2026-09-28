"""
Training curves of the three backbones: one-tic full-frame PSNR against the raw frame on the training maps'
validation windows, against updates (Rohan, Sep 27 evening: absolute quantities, no persistence).

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

**Quantity.** Each read's PSNR as read: the rendered prediction (stock decoder) against the raw frame, full frame,
one tic ahead, a window mean without an interval. Nothing is subtracted and no persistence line is drawn; the
file's persistence PSNR is kept in the sidecar only. `series` still carries the gain over persistence for the record.

**Outputs** (`paper/FIGURE_STANDARDS.md`, Figure 1b, and round 1 of the figure review), drawn with `figstyle`:

- `fig1_curves.pdf/.png`, `PANEL_SIZE` (1.5 in wide, for a place in Figure 1's row): EMA weights only, one line
  per backbone in its encoding colour with its marker (open: every read is through the stock decoder) at every
  50k updates, labelled at its right end (leaders where the labels had to move apart), the zero line labelled
  "copy-last".
- `figA_training_curves.pdf/.png`, full width: the same with the live weights dotted beside each EMA line.

A read carrying a `note` (SD 3.5's 200k read, scored after the periodic evaluation ended at 155k) is an open marker not joined to its
line. Early EMA reads fall far below copy-last (the fp32 EMA at 0.9999 still averages the pretrained start for the
first few thousand updates, 6.7 to 11.8 dB PSNR at 5k); the axis floor clips them, and every clipped read is
printed and written to `<stem>.json` beside the figure so the caption can disclose it.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "paper"))
import figstyle as fs  # noqa: E402
from matplotlib import ticker  # noqa: E402

BACKBONES = ("unet", "pixart", "sd35")
WEIGHTS = ("ema", "live")
LINESTYLES = {"ema": "-", "live": ":"}
PANEL_SIZE = (1.5, 1.4)
APPENDIX_SIZE = (fs.TEXT_WIDTH, 1.8)
STEM = "fig1_curves"
APPENDIX_STEM = "figA_training_curves"
DEFAULT_CURVES_DIR = os.path.join(REPO, "results", "training_curves")
DEFAULT_OUT_DIR = os.path.join(REPO, "paper", "figures")
# dB: the axis floor sits this far under the lowest live read; reads below it (the EMA's warm-up) are clipped and
# disclosed
FLOOR_BELOW_LIVE = 1.0
CEILING_MARGIN = 0.25   # dB above the highest drawn read
MARK_EVERY = 50000      # updates between markers on a line
# the U-Net's and PixArt's curves nearly coincide: their markers alternate (round 2, Astra 26), each at a measured read
MARK_PHASE = {"unet": MARK_EVERY // 2, "pixart": 0, "sd35": 0}


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
    """{"k_updates", "psnr", "gain"} of one weight set, in step order: the joined reads, or with `noted` the reads
    that carry a note and are drawn apart. The gain is the read's PSNR minus the file's raw copy-last PSNR."""
    rs = [r for r in curve["reads"] if r["weights"] == weights and bool(r.get("note")) == noted]
    base = curve["persistence"]["psnr"]
    return {"k_updates": [r["step"] / 1000.0 for r in rs], "psnr": [float(r["psnr"]) for r in rs],
            "gain": [float(r["psnr"]) - base for r in rs]}


def y_limits(curves, weights=WEIGHTS):
    """(floor, ceiling) of the PSNR axis: `FLOOR_BELOW_LIVE` under the lowest live read of any backbone (the EMA's
    warm-up falls far below it and is clipped), the ceiling from every drawn read."""
    live = [v for c in curves for v in series(c, "live")["psnr"]]
    drawn = [v for c in curves for w in weights for noted in (False, True) for v in series(c, w, noted)["psnr"]]
    floor = (min(live) if live else min(drawn)) - FLOOR_BELOW_LIVE
    return floor, max(drawn) + CEILING_MARGIN


def clipped_reads(curves, weights=WEIGHTS):
    """The reads under the axis floor, for the caption: [{backbone, weights, step, psnr, gain}]."""
    floor = y_limits(curves)[0]
    out = []
    for c in curves:
        for w in weights:
            s = series(c, w)
            out += [{"backbone": c["backbone"], "weights": w, "step": round(k * 1000), "psnr": round(p, 2),
                     "gain": round(g, 2)} for k, p, g in zip(s["k_updates"], s["psnr"], s["gain"]) if p < floor]
    return out


def draw(curves, weights=("ema",), size=PANEL_SIZE):
    """The panel: (figure, axes). Colours and markers are the encoding table's; stock-decoder reads are open."""
    fs.style()
    fig, (ax,) = fs.new_figure(size)
    lo, hi = y_limits(curves, weights)
    ends = []
    for c in curves:
        ent = fs.BACKBONES[c["backbone"]]
        for w in weights:
            s = series(c, w)
            if s["k_updates"]:
                phase = MARK_PHASE.get(c["backbone"], 0)
                every = [i for i, k in enumerate(s["k_updates"])
                         if round(k * 1000) > 0 and (round(k * 1000) - phase) % MARK_EVERY == 0]
                if not every or s["k_updates"][-1] - s["k_updates"][every[-1]] >= MARK_EVERY / 1000:
                    every.append(len(s["k_updates"]) - 1)      # the last read, unless a marker sits close by
                ax.plot(s["k_updates"], s["psnr"], color=ent.colour, ls=LINESTYLES[w],
                        lw=fs.DATA_LW if w == "ema" else 0.8, zorder=3 if w == "ema" else 2,
                        marker=ent.marker if w == "ema" else None, markevery=sorted(set(every)), ms=3.5,
                        mfc="white", mec=ent.colour, mew=0.7, label=f"{c['label']} {w.upper() if w == 'ema' else w}")
                if w == "ema":
                    # the label sits right of the backbone's rightmost mark, a noted read included
                    right_k = max(s["k_updates"] + series(c, w, noted=True)["k_updates"])
                    ends.append((right_k, s["psnr"][-1], ent.label, ent.colour))
            apart = series(c, w, noted=True)
            if apart["k_updates"]:
                ax.plot(apart["k_updates"], apart["psnr"], ls="none", marker=ent.marker, ms=3.5, mfc="white",
                        mec=ent.colour, mew=0.7, zorder=4, label=f"{c['label']} {w} (noted)")
    fs.end_labels(ax, ends, gap=(hi - lo) * 0.13, leaders=True)
    right = max(r["step"] for c in curves for r in c["reads"]) / 1000.0
    ax.set_xlim(0, right * (1.02 if size[0] < 3 else 1.1))
    ax.set_ylim(lo, hi)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100 if size[0] < 3 else 50))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.set_xlabel("updates (thousands)")
    ax.set_ylabel("full-frame\nPSNR (dB)" if size[1] < 1.6 else "full-frame PSNR (dB)")
    if size[0] < 3:
        fs.panel_letter(ax, "b")              # its place in Figure 1 (round 2, Astra 27)
    return fig, ax


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--curves-dir", default=DEFAULT_CURVES_DIR, help="directory holding <backbone>.json")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--stem", default=STEM)
    p.add_argument("--appendix-stem", default=APPENDIX_STEM)
    a = p.parse_args(argv)
    curves = load_curves(a.curves_dir)
    written = []
    for stem, weights, size in ((a.stem, ("ema",), PANEL_SIZE), (a.appendix_stem, WEIGHTS, APPENDIX_SIZE)):
        fig, _ = draw(curves, weights, size)
        written += fs.save(fig, a.out_dir, stem)
        side = os.path.join(a.out_dir, f"{stem}.json")
        with open(side, "w") as f:
            json.dump({"weights": list(weights), "quantity": "full-frame PSNR against the raw frame (dB)",
                       "floor_db": y_limits(curves, weights)[0],
                       "persistence_psnr": {c["backbone"]: c["persistence"]["psnr"] for c in curves},
                       "clipped_reads": clipped_reads(curves, weights),
                       "noted_reads": [{"backbone": c["backbone"], **{k: r[k] for k in ("step", "weights", "psnr")},
                                        "note": r["note"]} for c in curves for r in c["reads"] if r.get("note")],
                       "last_reads": {c["backbone"]: {w: {"step": int(series(c, w)["k_updates"][-1] * 1000),
                                                          "psnr": series(c, w)["psnr"][-1]}
                                                      for w in weights if series(c, w)["k_updates"]}
                                      for c in curves}}, f, indent=1)
            f.write("\n")
        written.append(side)
    for path in written:
        print("wrote", path)
    for r in clipped_reads(curves):
        print(f"clipped below the axis floor: {r['backbone']} {r['weights']} {r['step']}: {r['psnr']} dB PSNR")
    for c in curves:
        for r in c["reads"]:
            if r.get("note"):
                print(f"{c['backbone']} {r['weights']} {r['step']}: drawn apart ({r['note']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
