"""
The persistence-free set (Rohan, 2026-09-27 evening): raw scene PSNR and scene LPIPS as the only quantities, with
the reconstruction ceiling and the in-distribution level as the only references. Nothing here replaces a figure or
table in the paper; the figures carry the `raw_` prefix and sit beside the current ones until Rohan chooses.

    python paper/make_raw_figures.py     # writes paper/figures/raw/ and paper/tables/tuned/{adapt_groups.tex,
                                         # adapt_perarena.tex, results_full.tex (Table 1), results_slim.tex and
                                         # results_slim_caption.tex (the four-page table), raw_summary.json}

**Quantities.** Raw scene PSNR and LPIPS compare the rendered prediction D(z_hat) with the raw ground-truth frame x
on scene rows 0 to 207 (`scene_psnr_raw`, `scene_lpips_raw` in eval_tf's per-window files). A map's reconstruction
upper bound (the ceiling) is the same comparison for the decoder applied to the ground-truth latent, D(z) against x
(`scene_vae_psnr`,
`scene_vae_lpips`). The U-Net and PixArt-alpha render through the fine-tuned SD 1 decoder (the `_tuned` columns),
SD 3.5 through its own. The in-distribution level is the training maps' validation read (512 windows, four maps
pooled). Windows eval_tf flags as duplicates are left out, as `score_adapt.py` leaves them out.

**What exists where.** Raw-frame columns exist only in the fresh-rescore reads: zero-shot per arena, the training
maps' validation, and the U-Net adapter of the 8k-grid runs by step (`adapt<step>g8k_live_tuned`, and
`adapt8000_live_tuned` for 8k; `adapter_rows`), all on the same 256 held-out windows per arena. The adaptation runs'
own per-step files compare the prediction with the decoded ground truth, D(z_hat) against D(z), not with x, so they
are not read here. The merged row's curve panels draw each arena's raw trajectory through the reads that exist
(0 is the zero-shot read), sharing their y axes with the before-and-after panels so each curve ends on its filled
8k mark.

**The budget** (the main session's decision, 2026-09-27). With G = upper bound - PSNR the gap to the reconstruction
upper bound, G0 its zero-shot value and G_train the training maps' own gap (pooled), an arena's budget is the first
adapter read that closes half of its excess gap, G <= (G0 + G_train) / 2 (`budget_threshold`, rule
`half_excess_gap`; reaching counts); an arena that does not by 8k is censored, and a median over arenas ranks a
censored arena above every crossing (`make_adapt_figures.censored_median_entry`). The rules `half_ceiling_gap`
(G <= G0 / 2) and `in_distribution_gap` (G <= G_train) are kept for the record. Until every step of the 8k grid has
a raw read the table prints the budget cells as \\tbd and the summary marks the budgets provisional.

**Groups and colours** (the same decision). The arenas sorted by G0, largest first, split into hard, medium and easy
(4, 5, 4); the merged row colours each arena by its group, three shades of the adapter's blue ramp, dark = hard.
Zero-shot PSNR itself (`--group-by zero_shot_psnr`) tracks persistence's PSNR more than the shift (the paper
owner's check). Table 2's candidate (`adapt_groups.tex`) gives each group's medians and the full fine-tune
column's comparator arenas.

**Outputs.** `<out-dir>/raw_*.pdf` and 600 dpi PNGs at the slot sizes in `SIZES`; `<tables-dir>/adapt_groups.tex`;
`--summary`: every number drawn, the step checks, the counts, the inputs' sha256.
"""
import argparse
import csv
import datetime
import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
import figstyle as fs  # noqa: E402
import make_adapt_figures as maf  # noqa: E402
import raw_adapters as rad  # noqa: E402
from score_adapt import DUPLICATE_FLAGS  # noqa: E402

from matplotlib import ticker  # noqa: E402  (maf has already selected the Agg backend)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# the zero-shot rows when no later SD 3.5 read through its fine-tuned decoder exists (`backbone_rows` picks the rows)
BACKBONES = ("unet200k_ema", "pixart200k_ema", "sd35_170000")
DECODER_SUFFIX = {"unet200k_ema": "_tuned", "pixart200k_ema": "_tuned", "sd35_170000": ""}
SD35_FALLBACK = "sd35_170000"                   # the provisional 170k read through the stock decoder
GRID = (0, 50, 100, 150, 250, 500, 1000, 2000, 4000, 8000)      # the 8k grid's reads
ADAPTER_ROW_RE = re.compile(r"^adapt(\d+)g8k_live_tuned$")
ADAPTER_8K_ROW = "adapt8000_live_tuned"          # item 11's 8k read of the 8k-grid runs (no g8k in its name)
BUDGET_RULE = "half_lpips_rise"                 # the first grid read recovering half of the zero-shot LPIPS rise
BASE_4K_ROW = "adapt4000_live_tuned"            # the base run's 4k adapter, for the base-against-8k-grid check
TRAINING_MAPS = maf.TRAINING_MAPS
PRIMARY = "unet200k_ema"
COMPARATOR_ARENAS = (6, 7, 8, 16)               # the full fine-tune's arenas (Table 2's last column)
GROUP_NAMES = ("hard", "medium", "easy")
TABLE_GROUP_ORDER = ("easy", "medium", "hard")   # the tables' row order (Rohan, Sep 29): easiest first
# Table 2's per-backbone blocks at the headline step, in order: (key in raw_adapters.load_blocks, label, arenas; None
# for every arena); the U-Net LoRA block is matched to the full fine-tune's arenas
BLOCKS = (("unet_lora", "U-Net LoRA", COMPARATOR_ARENAS),
          # every map the full fine-tune has: the four 4k-only originals plus the 8k-grid runs landing on main
          # (Sep 29); the caption's adapter comparison is recomputed over the block's own maps
          ("unet_full", "U-Net full fine-tune", None),
          ("pixart_lora", "PixArt-$\\alpha$ LoRA", None), ("sd35_lora", "SD 3.5 LoRA", None))
BLOCK_STYLE = {"unet_lora": ("unet", "-"), "unet_full": ("full", (0, (3, 2))), "pixart_lora": ("pixart", "-"),
               "sd35_lora": ("sd35", "-")}
PANEL_LABELS = {"unet_lora": "U-Net", "unet_full": "U-Net full", "pixart_lora": "PixArt-\u03b1", "sd35_lora": "SD 3.5"}
HEADLINE_STEP = 4000                            # the paper's budget (Rohan, Sep 27 evening); 8k is the check
ROW_TICKS = (0, 100, 1000, 8000)                # labelled steps in the row's narrow curve panels
GRID_TICKS = (0, 50, 250, 1000, 4000, 8000)     # and in the two-by-two layout's wider ones
SIZES = {"raw_row": (5.5, 1.75), "raw_row_grid": (5.5, 3.0), "raw_figA_adapt_arenas": (5.5, 3.0),
         "raw_backbones": (5.5, 1.6), "raw_backbones_shared": (5.5, 1.6), "raw_fullft": (5.5, 1.6),
         "raw_fig3a_psnr": (2.25, 1.5),
         "raw_fig3a_lpips": (2.25, 1.5),
         # the body candidates (Rohan chooses): the row with the backbones as (e, f) on a line beneath it (1.75 in of
         # row and 1.35 in of backbone panels with their key), and the separate figure
         "raw_row_backbones": (5.5, 3.1), "raw_backbones_body": (5.5, 1.6),
         # the row again at the 1.5 in height the page-4 fit may need (Rohan, Sep 28 evening): same panels and type
         "raw_row_150": (5.5, 1.5),
         # Figure 3 rebuilt per backbone (Rohan, Sep 28 night): (a, b) every backbone per arena, (c, d) the medians
         # against updates; one row of four, and two rows of two when the per-arena panels need the width
         "raw_row_v2": (5.5, 1.75), "raw_row_v2_tall": (5.5, 3.0)}
IN_DISTRIBUTION = "training maps (in distribution)"
BAND_LABEL = "train vs train $d$"               # the training maps' own d range (tools/family_step.py)
PSNR_LABEL, LPIPS_LABEL = "scene PSNR (dB) ↑", "scene LPIPS ↓"


# ---------------------------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------------------------

def read_windows(path):
    """The rows of one eval_tf `per_window.csv`, duplicate windows left out, numbers as floats where they parse."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if any(maf.as_float(r.get(c)) == 1 for c in DUPLICATE_FLAGS):
            continue
        out.append({k: (maf.as_float(v) if maf.as_float(v) is not None else v) for k, v in r.items()})
    return out


def _mean(rows, column):
    v = [r[column] for r in rows if isinstance(r.get(column), float) and math.isfinite(r[column])]
    return float(np.mean(v)) if v else None


def means_by_map(rows, columns):
    """{map: {column: mean over the map's windows, "n": windows}} by the rows' `map` column."""
    groups = {}
    for r in rows:
        groups.setdefault(int(r["map"]), []).append(r)
    return {m: {"n": len(rs), **{c: _mean(rs, c) for c in columns}} for m, rs in groups.items()}


def raw_columns(sfx):
    return {"psnr": f"scene_psnr_raw{sfx}", "lpips": f"scene_lpips_raw{sfx}",
            "ceiling_psnr": f"scene_vae_psnr{sfx}", "ceiling_lpips": f"scene_vae_lpips{sfx}"}


def _renamed(entry, cols):
    return {"n": entry.get("n"), **{k: entry.get(c) for k, c in cols.items()}}


def per_window_files(row_dir):
    """{map: per_window.csv} of a fresh-rescore row: `map<NN>/per_window.csv` or one level below."""
    out = {}
    if not os.path.isdir(row_dir):
        return out
    for d in sorted(os.listdir(row_dir)):
        m = re.fullmatch(r"map0*(\d+)", d)
        if not m:
            continue
        found = sorted(glob.glob(os.path.join(row_dir, d, "per_window.csv")) or
                       glob.glob(os.path.join(row_dir, d, "*", "per_window.csv")))
        if len(found) == 1:
            out[int(m.group(1))] = found[0]
    return out


def row_reads(root, row, sfx):
    """({arena: {psnr, lpips, ceiling_psnr, ceiling_lpips, n}}, files) of one fresh-rescore row."""
    cols = raw_columns(sfx)
    out, files = {}, []
    for arena, path in per_window_files(os.path.join(root, row)).items():
        e = means_by_map(read_windows(path), list(cols.values())).get(arena)
        if e and e[cols["psnr"]] is not None:
            out[arena] = _renamed(e, cols)
            files.append(path)
    return out, files


def home_reads(root, row, sfx, extra=()):
    """The training maps' validation read of one row: per map and pooled (the raw columns, and `extra` pooled)."""
    found = sorted(glob.glob(os.path.join(root, f"home_{row}", "**", "per_window.csv"), recursive=True))
    if len(found) != 1:
        raise SystemExit(f"{root}/home_{row}: expected one per_window.csv, found {len(found)}")
    cols = raw_columns(sfx)
    rows = read_windows(found[0])
    return {"maps": {m: _renamed(e, cols) for m, e in means_by_map(rows, list(cols.values())).items()},
            "pooled": {**{k: _mean(rows, c) for k, c in cols.items()}, **{c: _mean(rows, c) for c in extra}},
            "n": len(rows), "path": found[0]}


def _header(path):
    with open(path, newline="") as f:
        return set(next(csv.reader(f), []))


def _carries_tuned(root, row):
    """Whether a row's arena files and its one training-map file all carry the fine-tuned decoder's raw columns."""
    arena = per_window_files(os.path.join(root, row))
    home = glob.glob(os.path.join(glob.escape(root), f"home_{row}", "**", "per_window.csv"), recursive=True)
    need = {"scene_psnr_raw_tuned", "scene_lpips_raw_tuned", "scene_vae_psnr_tuned"}
    return bool(arena) and len(home) == 1 and all(need <= _header(p) for p in list(arena.values()) + home)


def latest_sd35_row(root, tuned=False):
    """The SD 3.5 zero-shot row with the highest checkpoint under `root` (`sd35_<step>` with per-window files and
    a home read beside it; with `tuned`, both carrying the fine-tuned SD 3.5 decoder's columns), or None: its upper
    bound and persistence are the SD 3.5 blocks' references and its home read their in-distribution gap."""
    rows = [(int(m.group(1)), d) for d in (os.listdir(root) if os.path.isdir(root) else [])
            for m in [re.fullmatch(r"sd35_(\d+)", d)] if m and per_window_files(os.path.join(root, d))
            and os.path.isdir(os.path.join(root, f"home_{d}")) and (not tuned or _carries_tuned(root, d))]
    return max(rows)[1] if rows else None


def backbone_rows(root):
    """{name: (row directory, column suffix)} of the zero-shot reads the figures and summaries use, each backbone
    through its own fine-tuned decoder (Table 1's rule): the U-Net and PixArt-alpha through the fine-tuned SD 1
    decoder, SD 3.5 at its latest checkpoint whose arena and training-map files carry the fine-tuned SD 3.5
    decoder's columns; without one, the provisional 170k read through the stock decoder (the fallback)."""
    rows = {n: (n + DECODER_SUFFIX[n], DECODER_SUFFIX[n]) for n in BACKBONES if not n.startswith("sd35")}
    sd = latest_sd35_row(root, tuned=True)
    rows.update({sd: (sd, "_tuned")} if sd else {SD35_FALLBACK: (SD35_FALLBACK, DECODER_SUFFIX[SD35_FALLBACK])})
    return rows


def training_distances(path):
    """{map: D} of the training maps' validation entries (`val/<map>`), as `tools/family_step.py` reads them."""
    with open(path) as f:
        js = json.load(f)
    out = {int(m["map"]): float(m["D"]) for m in js.get("maps", [])
           if m.get("set") == "val" and int(m["map"]) in TRAINING_MAPS and maf.finite(m.get("D"))}
    if sorted(out) != sorted(TRAINING_MAPS):
        raise SystemExit(f"{path}: no val D for training maps {sorted(set(TRAINING_MAPS) - set(out))}")
    return out


def adapter_rows(root):
    """{step: row} of the 8k-grid adapter's raw-frame reads under `root`: `adapt<step>g8k_live_tuned`, and
    `adapt8000_live_tuned` for 8k when no g8k-named 8k read exists (`adapt4000_live_tuned` is the base run's)."""
    names = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))) if os.path.isdir(root) else []
    out = {int(m.group(1)): d for d in names for m in [ADAPTER_ROW_RE.match(d)] if m}
    if 8000 not in out and ADAPTER_8K_ROW in names:
        out[8000] = ADAPTER_8K_ROW
    return dict(sorted(out.items()))


def distance_floor(path):
    with open(path) as f:
        js = json.load(f)
    floor = (js.get("floor") or {}).get(js.get("primary_arm", "motion"))
    if not floor:
        raise SystemExit(f"{path}: no floor for the primary arm")
    return {"min": floor["min"], "max": floor["max"]}


# ---------------------------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------------------------

def guard_reads(run_dirs):
    """{arena: {directional_0, directional_8k, trainmap_psnr_dec_0, trainmap_psnr_dec_8k, forgetting_dec}} from the
    8k-grid runs' guard rows (live weights, steps 0 and 8000): the directional check, and the training maps' scene
    PSNR against the decoded ground truth through the stock decoder, read from the per-window file each row names
    (found beside the run); forgetting_dec is its change from 0 to 8k. A decoded-reference, stock-decoder number,
    marked as such wherever it is printed, until a raw guard rescore exists."""
    out = {}
    for run in sorted(run_dirs):
        m = re.search(r"_map0*(\d+)_", os.path.basename(run.rstrip("/")))
        path = os.path.join(run, "scores.jsonl")
        if not m or not os.path.exists(path):
            continue
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        guard = {r["step"]: r for r in rows
                 if r.get("weights") == "live" and r.get("directional_correct_frac") is not None}
        if 0 not in guard or 8000 not in guard:
            continue
        psnr = {}
        for step in (0, 8000):
            named = guard[step].get("trainmap_per_window") or ""
            parts = named.replace("\\", "/").split("/")
            local = os.path.join(run, "scores", parts[-3], parts[-2], parts[-1]) if len(parts) >= 3 else ""
            psnr[step] = _mean(read_windows(local), "scene_psnr_dec") if os.path.exists(local) else None
        out[int(m.group(1))] = {
            "directional_0": guard[0]["directional_correct_frac"],
            "directional_8k": guard[8000]["directional_correct_frac"],
            "directional_4k": guard[4000]["directional_correct_frac"] if 4000 in guard else None,
            "trainmap_psnr_dec_0": psnr[0], "trainmap_psnr_dec_8k": psnr[8000],
            "forgetting_dec": psnr[8000] - psnr[0] if None not in psnr.values() else None}
    return out


def perarena_table(arenas, adaptation, zero, guards, budgets_final):
    """The appendix's per-map table: unseen map, its zero-shot LPIPS rise over the in-distribution level, the
    budget (the first grid read recovering half of that rise; \\tbd until every grid step has a raw read), raw
    scene PSNR and LPIPS at 0, 4k and 8k, the recovered shares at 4k, forgetting on the training maps (decoded
    reference, stock decoder, marked) and the directional check at 8k."""
    def num(v, d, signed=False):
        if v is None:
            return "--"
        text = f"{v:+.{d}f}" if signed else f"{v:.{d}f}"
        return text.replace("-", "$-$")

    lines = ["% generated by paper/make_raw_figures.py: raw scene PSNR and LPIPS through the fine-tuned decoder,",
             "% U-Net adapter (non-EMA weights, 8k grid); rise: zero-shot scene LPIPS minus the training maps';",
             "% budget: the first grid read at or below (LPIPS at 0 + the training maps' LPIPS) / 2; recovered:",
             "% shares at 4k, per map;",
             "% forgetting (dagger) is the training maps' scene PSNR against the decoded ground truth through the",
             "% stock decoder, 0 to 8k, until a raw guard rescore exists",
             "\\begin{tabular}{rrrrrrrrrrrr}", "\\toprule",
             "& LPIPS & & \\multicolumn{3}{c}{Scene PSNR (dB)} & \\multicolumn{3}{c}{Scene LPIPS} & "
             "\\multicolumn{2}{c}{Recovered (\\%)} & Forgetting$^\\dagger$ \\\\",
             "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-11}",
             "Unseen map & rise & Budget & 0 & 4k & 8k & 0 & 4k & 8k & LPIPS & PSNR & (dB) / directional \\\\",
             "\\midrule"]
    for m in arenas:
        r, g = adaptation[m], guards.get(m, {})
        budget = ("\\tbd{}" if not budgets_final else
                  ("${>}$8k" if r["budget"] is None else fs.step_label(r["budget"])))
        guard = (f"{num(g.get('forgetting_dec'), 2, signed=True)} / {num(g.get('directional_8k'), 2)}" if g
                 else "\\tbd{} / \\tbd{}")
        lines.append(f"{m} & {num(r['lpips_rise'], 3)} & {budget} & "
                     f"{num(r['zero_shot'], 2)} & {num(r['psnr'].get('4000'), 2)} & {num(r['psnr'].get('8000'), 2)} & "
                     f"{num(zero[m]['lpips'], 3)} & {num(r['lpips'].get('4000'), 3)} & "
                     f"{num(r['lpips'].get('8000'), 3)} & {share_cell(r['lpips_share_4k'])} & "
                     f"{share_cell(r['psnr_share_4k'])} & {guard} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "% dagger: decoded reference (the prediction against the decoded ground truth), stock decoder"]
    return "\n".join(lines) + "\n"


def lpips_threshold(lpips_zero_shot, level_lpips):
    """The scene LPIPS an adapter read must reach to count as recovering half of the map's zero-shot LPIPS rise
    over its backbone's in-distribution level on the training maps (Rohan's budget rule, Sep 29 00:30 EDT, after
    the upper bound left the paper): (LPIPS at 0 + the training maps' LPIPS) / 2."""
    return (lpips_zero_shot + level_lpips) / 2


def budget_step(reads, threshold):
    """The first read (step, LPIPS), in step order, at or below `threshold`; None if none is (censored)."""
    return next((s for s, v in sorted(reads) if v is not None and v <= threshold), None)


def share(recovered, deficit):
    """A recovered share: `recovered` over `deficit`, or None when the deficit is not positive (the map starts at
    or past the in-distribution level, so there is nothing to recover) or either is missing."""
    return recovered / deficit if recovered is not None and deficit is not None and deficit > 0 else None


def bootstrap(values, statistic, draws=10000, seed=0):
    """The 95% percentile interval of `statistic` over arenas resampled with replacement (`draws` draws)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    stats = [statistic(v[rng.integers(0, len(v), len(v))]) for _ in range(draws)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def text_numbers(adaptation, complete, s0, grid_complete, draws=10000):
    """The numbers the text waits on (the paper owner, Sep 27 evening), over the maps with an 8k read: the count
    past the budget's threshold by 4k and by 8k, the median budget (censored as +inf), and the median recovered
    shares at the headline (LPIPS and PSNR, per map then the median), each with its map-bootstrap interval; the mean
    share of each map's 0-to-8k raw gain in place at every raw read; Spearman of the zero-shot latent skill S0 with
    the headline LPIPS share, the 8k raw gain and the budget. Everything that depends on the budget is provisional
    until every grid step has a raw read."""
    rows = [adaptation[m] for m in complete]
    budgets = [math.inf if r["budget"] is None else r["budget"] for r in rows]

    def count_by(step):
        hit = [1.0 if b <= step else 0.0 for b in budgets]
        return {"count": int(sum(hit)), "n": len(hit), "interval": bootstrap(hit, np.sum, draws)}

    def share_entry(key):
        v = [r[key] for r in rows if r[key] is not None]
        return {"value": median(v), "interval": bootstrap(v, np.median, draws) if v else None, "n": len(v)}

    gains = [r["psnr"]["8000"] - r["zero_shot"] for r in rows]
    shares = {}
    for s in sorted({int(k) for r in rows for k in r["psnr"]} - {8000}):
        v = [(r["psnr"][str(s)] - r["zero_shot"]) / (r["psnr"]["8000"] - r["zero_shot"]) for r in rows
             if r["psnr"].get(str(s)) is not None and r["psnr"]["8000"] != r["zero_shot"]]
        shares[str(s)] = float(np.mean(v)) if v else None
    median_budget = maf.median_censored(budgets)
    budget_ci = bootstrap(budgets, np.median, draws)
    x = [s0.get(m) for m in complete]
    return {
        "provisional": not grid_complete,
        "pending": {"gain_step": not grid_complete, "budget": not grid_complete,
                    "forgetting": True},            # a raw rescore of the training maps' guard reads
        "past_threshold_by_4k": count_by(4000), "past_threshold_by_8k": count_by(8000),
        "median_budget": {"value": None if median_budget is None or math.isinf(median_budget) else median_budget,
                          "censored": median_budget is not None and math.isinf(median_budget),
                          "interval": [None if math.isinf(v) else v for v in budget_ci],
                          "censored_count": sum(1 for b in budgets if math.isinf(b))},
        "median_lpips_share_4k": share_entry("lpips_share_4k"),
        "median_psnr_share_4k": share_entry("psnr_share_4k"),
        "gain_share_by_step": shares,
        "spearman_S0": {"lpips_share_4k": maf.spearman(x, [r["lpips_share_4k"] for r in rows]),
                        "gain_8k": maf.spearman(x, gains),
                        "budget": maf.spearman(x, [maf.cost_rank_value(r["budget"], 8000) for r in rows])},
    }


def step_check(train, arenas, higher_is_better=True):
    """Whether every arena lies on the far side of every training map (below in PSNR, above in LPIPS), and the
    margin between the two families (positive when the step holds)."""
    margin = (min(train) - max(arenas)) if higher_is_better else (min(arenas) - max(train))
    return {"holds": margin > 0, "margin": margin}


def median(values):
    v = [x for x in values if x is not None]
    return float(statistics.median(v)) if v else None


def no_ceiling(entry):
    """A read without its reconstruction upper bound columns (`ceiling_psnr`, `ceiling_lpips`): the paper dropped
    the upper bound (Rohan, Sep 28 night), so the summary records only what is drawn or tabled."""
    return {k: v for k, v in entry.items() if not k.startswith("ceiling_")}


def group_arenas(values, higher_is_harder=False):
    """[(name, arenas)] for hard, medium, easy: the arenas sorted hardest first by `values` (lowest first, or highest
    with `higher_is_harder`, as for the gap to the upper bound), split 4, 5, 4 for 13 arenas and in near-thirds (the
    middle largest) otherwise."""
    order = sorted(values, key=lambda a: (-values[a] if higher_is_harder else values[a], a))
    n = len(order)
    edge = 4 if n == 13 else n // 3
    sizes = (edge, n - 2 * edge, edge)
    out, i = [], 0
    for name, k in zip(GROUP_NAMES, sizes):
        out.append((name, sorted(order[i:i + k])))
        i += k
    return out


GROUP_SHADES = {"hard": 0.0, "medium": 0.5, "easy": 1.0}      # positions on the adapter's blue ramp


def group_colours(groups):
    """({arena: colour}, [(group, colour)]): one shade of the adapter's blue ramp per group, dark = hard, so a
    three-swatch key names every mark exactly."""
    shade = {g: fs.ramp_colour(GROUP_SHADES[g], 0.0, 1.0) for g in GROUP_SHADES}
    return ({m: shade[name] for name, members in groups for m in members},
            [(name, shade[name]) for name, members in groups if members])


def group_stats(groups, per):
    """Per group and for all maps: the medians of zero-shot, 4k and 8k raw PSNR and LPIPS, of the recovered shares
    at the headline (per map, then the median), the censored median of the budget and the censored count. `per`
    maps a map to {zero_shot, psnr_4k, psnr_8k, lpips_zero_shot, lpips_4k, lpips_8k, lpips_share_4k,
    psnr_share_4k, budget}."""
    keys = {"psnr_zero_shot": "zero_shot", "psnr_4k": "psnr_4k", "psnr_8k": "psnr_8k",
            "lpips_zero_shot": "lpips_zero_shot", "lpips_4k": "lpips_4k", "lpips_8k": "lpips_8k",
            "lpips_share_4k": "lpips_share_4k", "psnr_share_4k": "psnr_share_4k"}
    out = {}
    for name, arenas in list(groups) + [("all", sorted(per))]:
        rs = [per[a] for a in arenas]
        out[name] = {"arenas": list(arenas), "n": len(rs), **{k: median([r[v] for r in rs]) for k, v in keys.items()},
                     "budget": maf.censored_median_entry(rs, "budget", 8000),
                     "budget_middle": middle_reads([r["budget"] for r in rs]),
                     "censored": sum(1 for r in rs if r["budget"] is None)}
    return out


def middle_reads(budgets):
    """The middle budget read(s) of a group, a censored arena (None) ranked above every crossing: one read for an
    odd group, the two middle reads for an even one (equal pairs collapse to one). Budgets are reads on a log grid,
    so their mean is a value no arena can have; the pair says what is known (the paper owner, Sep 27)."""
    ranked = sorted(budgets, key=lambda b: (b is None, b or 0))
    n = len(ranked)
    if not n:
        return []
    pair = [ranked[n // 2]] if n % 2 else [ranked[n // 2 - 1], ranked[n // 2]]
    return pair[:1] if len(pair) == 2 and pair[0] == pair[1] else pair


def budget_label(reads, last=8000):
    """A budget cell as printed: grid labels, ">8k" (the grid's `last` read) in math mode for censored, a pair joined
    by an en dash."""
    if not reads:
        return "--"
    return "--".join(f"${{>}}${fs.step_label(int(last))}" if r is None else fs.step_label(int(r)) for r in reads)


def coarse_budget_label(reads, headline=HEADLINE_STEP):
    """A budget cell for a block scored only at 0 and the headline, where the rule cannot resolve finer: "${\\le}$4k"
    for a middle read reached by the headline, "${>}$4k" for one not reached (censored), a pair joined by an en dash
    when the two middle reads differ."""
    if not reads:
        return "--"
    cells = [("${>}$" if r is None else "${\\le}$") + fs.step_label(int(headline)) for r in reads]
    return "--".join(dict.fromkeys(cells))


def _budget_cell(g, final=True):
    if not final:
        return "\\tbd{}"
    return budget_label(g["budget_middle"]) + (f"; {g['censored']} of {g['n']} censored" if g["censored"] else "")


def finished(arena_data, headline=HEADLINE_STEP):
    """Whether an arena's run has its headline read, or its grid stops before the headline. A run still training
    has its early reads only; in a block or a curve it would enter the zero-shot median and not the headline's."""
    return headline not in arena_data["grid"] or headline in arena_data["reads"]


def borrow_zero_shot(full, lora):
    """({arena: data}, borrowed arenas): full fine-tune arenas without a step-0 read take the U-Net LoRA's. Before
    its first update a rank-0 run is the checkpoint it starts from, and so is a LoRA (its B matrix starts at zero), so
    on the same held-out windows the two step-0 reads are one measurement. Borrowed only when both runs name the same
    source checkpoint and weights (`raw_adapters.run_source`); the read is marked `borrowed`. The input is not
    changed."""
    out, borrowed = {}, []
    for m, d in full.items():
        mine = lora.get(m) or {}
        zero = (mine.get("reads") or {}).get(0)
        if 0 not in d["reads"] and zero and d.get("source") and d.get("source") == mine.get("source"):
            out[m] = {**d, "reads": {0: {**zero, "borrowed": "unet_lora"}, **d["reads"]}}
            borrowed.append(m)
        else:
            out[m] = d
    return out, sorted(borrowed)


def split_arenas(arenas_data, wanted, headline=HEADLINE_STEP, check=8000):
    """(reported, decoded_only, in_progress) of the `wanted` arenas with a zero-shot (step 0) read. Arenas still
    training (`finished`) wait. Of the rest an arena is raw when its zero-shot, headline and check reads are raw
    wherever they exist. When any arena is raw the block reports the raw ones and lists the rest, which wait for a
    raw rescore: a median across raw reads and reads against the decoded ground truth mixes two quantities. With no
    raw arena it reports them all (against the decoded ground truth)."""
    started = [a for a in wanted if a in arenas_data and 0 in arenas_data[a]["reads"]]
    in_progress = [a for a in started if not finished(arenas_data[a], headline)]
    present = [a for a in started if a not in in_progress]
    raw = [a for a in present if all(r["quantity"] == "raw" for st, r in arenas_data[a]["reads"].items()
                                     if st in (0, headline, check))]
    return ((raw, [a for a in present if a not in raw]) if raw else (present, [])) + (in_progress,)


def recovery_shares(arenas_data, arenas, level, headline=HEADLINE_STEP):
    """{lpips, psnr: {median, min, n}} of what the adapter recovers by the `headline` step, each map's share
    computed on its own reads, then the median and minimum over the maps (fractions, not percent): `lpips`: (LPIPS
    at 0 - LPIPS at the headline) / (LPIPS at 0 - the training maps' LPIPS), the share of the zero-shot LPIPS rise
    over the in-distribution level that is undone; `psnr`: (PSNR at the headline - PSNR at 0) / (the training maps'
    PSNR - PSNR at 0), the share of the lost scene PSNR that is regained (above 1 when the map ends past the training
    maps). `level` is the backbone's training-map (PSNR, LPIPS); a map whose deficit is not positive (it starts at
    or past the training maps) is left out of that share, and `n` says how many maps enter it. Entries are None
    when the block has no level or no map."""
    def summarise(values):
        v = [x for x in values if x is not None]
        return {"median": float(statistics.median(v)), "min": float(min(v)), "n": len(v)} if v else None

    lpips, psnr = [], []
    for a in arenas:
        reads = arenas_data[a]["reads"]
        r0, r4 = reads.get(0), reads.get(headline)
        if not r0 or not r4 or not level:
            continue
        if level[1] is not None:
            lpips.append(share(r0["lpips"] - r4["lpips"], r0["lpips"] - level[1]))
        if level[0] is not None:
            psnr.append(share(r4["psnr"] - r0["psnr"], level[0] - r0["psnr"]))
    return {"lpips": summarise(lpips), "psnr": summarise(psnr)}


def block_summary(arenas_data, wanted, level, headline=HEADLINE_STEP, check=8000):
    """One per-backbone block of Table 2: over the `wanted` maps it reports (`split_arenas`), the medians of scene
    PSNR and LPIPS at 0, the headline and the check step (None when the block's grid stops before it), the
    recovered shares (`recovery_shares`) and the budget (the first grid read recovering half of the map's zero-shot
    LPIPS rise over the backbone's in-distribution `level` (PSNR, LPIPS); final only when every map's grid
    reads are raw and the level exists), the quantity the reads are in (`raw_adapters.step_read`), the maps left out as
    decoded only, the maps whose budget is the first point of their grid (the rule cannot resolve finer there, as
    for SD 3.5's grid starting at 250), and the median GPU-hours to the headline per card (`gpu_hours_by_card`).
    `arenas_data` is {map: {grid, reads: {step: read}, gpu_hours, card}}."""
    present, decoded_only, in_progress = split_arenas(arenas_data, wanted, headline, check)

    def get(a, step, key):
        return (arenas_data[a]["reads"].get(step) or {}).get(key)
    has_check = any(check in arenas_data[a]["grid"] for a in present)
    kinds = {r["quantity"] for a in present for st, r in arenas_data[a]["reads"].items() if st in (0, headline, check)}
    quantity = (kinds.pop() if len(kinds) == 1 else "mixed") if kinds else None
    has_level = bool(level) and level[1] is not None
    final = bool(present) and quantity == "raw" and has_level
    budgets, at_first, incomplete = [], [], []
    grids = [st for a in present for st in arenas_data[a]["grid"] if st > 0]
    for a in present if has_level else ():
        reads, grid = arenas_data[a]["reads"], [st for st in arenas_data[a]["grid"] if st > 0]
        # a run whose grid stops short (map 15's SD 3.5 rerun diverged before 8k) still gives a budget when it
        # crosses on the reads it has; only a map that never crosses on an incomplete grid is undecided
        have = [st for st in grid if st in reads]
        if any(reads[st]["quantity"] != "raw" for st in have) or not have:
            final = False
            continue
        b = budget_step([(st, reads[st]["lpips"]) for st in have], lpips_threshold(reads[0]["lpips"], level[1]))
        if len(have) < len(grid):
            incomplete.append(a)
            if b is None:
                final = False
                continue
        budgets.append(b)
        if b is not None and b == min(grid):
            at_first.append((a, min(grid)))
    # every map's grid starts at the headline (scored at 0 and 4k, or 0, 4k and 8k): a budget is "by the headline"
    # or "beyond it", nothing finer
    coarse = bool(present) and all(min(st for st in arenas_data[a]["grid"] if st > 0) == headline for a in present)
    return {"n": len(present), "n_wanted": len(wanted), "wanted": list(wanted), "arenas": present,
            "quantity": quantity, "decoded_only": decoded_only, "in_progress": in_progress, "headline": headline,
            "coarse_grid": coarse, "last_read": max(grids) if grids else check,
            "first_read": min(grids) if grids else None,
            "psnr_zero_shot": median([get(a, 0, "psnr") for a in present]),
            "psnr_4k": median([get(a, headline, "psnr") for a in present]),
            "psnr_8k": median([get(a, check, "psnr") for a in present]) if has_check else None,
            "lpips_zero_shot": median([get(a, 0, "lpips") for a in present]),
            "lpips_4k": median([get(a, headline, "lpips") for a in present]),
            "lpips_8k": median([get(a, check, "lpips") for a in present]) if has_check else None,
            "n_8k": sum(1 for a in present if get(a, check, "psnr") is not None),
            "maps_8k": [a for a in present if get(a, check, "psnr") is not None],
            "grid_sources": {str(a): arenas_data[a].get("grid_source") for a in present},
            "runs": {str(a): arenas_data[a].get("run") for a in present},
            "guards": {str(a): arenas_data[a].get("guards") for a in present if arenas_data[a].get("guards")},
            "has_check": has_check, "level": list(level) if level else None, "budget_final": final,
            "shares": recovery_shares(arenas_data, present, level, headline),
            "budgets": {str(a): b for a, b in zip(present, budgets)} if final else None,
            "budget_middle": middle_reads(budgets) if final else None,
            "budget_at_first_read": at_first if final else None,
            "incomplete_grid": incomplete,
            "censored": sum(1 for b in budgets if b is None) if final else None,
            "gpu_hours_headline_by_card": gpu_hours_by_card({a: arenas_data[a] for a in present}, headline)}


def gpu_hours_by_card(arenas_data, step):
    """{card: {median, n, arenas}} of the GPU-hours to `step` over the arenas whose log has that step, grouped by the
    card each trained on ("unknown" when the log names none), cards sorted. Never one median over two cards: the SD 3.5
    LoRA took 1.33 h to 4k on an A6000 and 3.33 to 3.38 h on an A4000."""
    groups = {}
    for a, d in sorted(arenas_data.items()):
        h = (d.get("gpu_hours") or {}).get(str(step))
        if h is not None:
            groups.setdefault(d.get("card") or "unknown", []).append((a, h))
    return {card: {"median": median([h for _, h in v]), "n": len(v), "arenas": [a for a, _ in v]}
            for card, v in sorted(groups.items())}


def block_label(name, b, wanted_all):
    """A short row label, since the table sets at \\scriptsize in the body: "U-Net LoRA (6, 7, 8, 16)" for a block on
    a named set that is complete or not started, "(1 of 4)" while partial; "PixArt-$\\alpha$ LoRA (all 13)" or
    "(11 of 13)" for a block on every arena. The arenas scored are in the table's comment lines. A dagger when the
    reads are not raw (against the decoded ground truth)."""
    if b["n"] not in (0, b["n_wanted"]):
        label = f"{name} ({b['n']} of {b['n_wanted']})"
    elif wanted_all:
        label = f"{name} (all {b['n_wanted']})"
    else:
        label = f"{name} ({', '.join(map(str, b['wanted']))})"
    return label + ("$^\\ddagger$" if b["quantity"] not in (None, "raw") else "")


def block_budget_cell(b):
    """A block's budget cell: the middle read(s) (as "${\\le}$4k" when the block is scored only at 0 and the
    headline), the censored count when any, \\tbd until the budget is final."""
    if not b["budget_final"]:
        return "\\tbd{}"
    middle = coarse_budget_label(b["budget_middle"], b.get("headline", HEADLINE_STEP)) if b.get("coarse_grid") \
        else budget_label(b["budget_middle"], b.get("last_read", 8000))
    return middle + (f"; {b['censored']} of {b['n']} censored" if b["censored"] else "")


def slim_label(name, b, every):
    """A slim-table row label: the backbone's name alone when its block covers every arena (`every`), else as
    Table 2 labels its blocks ("(8 of 13)", "(6, 7, 8, 16)")."""
    complete = b["n"] in (0, b["n_wanted"])
    return name + ("$^\\ddagger$" if b["quantity"] not in (None, "raw") else "") if every and complete \
        else block_label(name, b, every)


def share_cell(share):
    """A recovered share as a whole percent: a block's {median, min, n} or one map's fraction ("--" when none)."""
    if share is None or share == {}:
        return "--"
    return f"{100 * (share['median'] if isinstance(share, dict) else share):.0f}"


def slim_table(rows, stamp=""):
    """The slim results table (the four-page layout's replacement for Tables 1 and 2): one row per backbone over the
    unseen arenas, then the U-Net's LoRA and full fine-tune on the comparator arenas. Columns: the backbone's raw scene
    PSNR and LPIPS on the training maps (in domain, for reference; blank for the comparison rows, whose checkpoint is
    the U-Net's and whose own in-domain score after adaptation is not measured), the zero-shot and headline medians
    of the block's own reads (paired: the same maps), then the recovered shares (`recovery_shares`: LPIPS and PSNR;
    medians of per-map shares, in percent). The budget stays in the supplement's table. `rows` is [(name,
    block_summary, (psnr, lpips) on the training maps or None, covers every map)]."""
    def num(v, d):
        return "--" if v is None else f"{v:.{d}f}"

    head = rows[0][1].get("headline", HEADLINE_STEP) if rows else HEADLINE_STEP
    lines = [f"% generated by paper/make_raw_figures.py{stamp}: raw scene reads, each backbone through its fine-tuned",
             "% decoder, non-EMA adapter weights; medians over each row's unseen maps (zero-shot and adapted paired);",
             "% recovered: per-map shares by the headline step, median over the row's unseen maps, in percent",
             "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
             # group headers no wider than their columns, which would otherwise take the excess into the last one
             "& \\multicolumn{2}{c}{In domain} & \\multicolumn{2}{c}{Zero-shot} & "
             f"\\multicolumn{{2}}{{c}}{{{fs.step_label(head)} updates}} & \\multicolumn{{2}}{{c}}{{Recovered (\\%)}} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
             "Model & PSNR & LPIPS & PSNR & LPIPS & PSNR & LPIPS & LPIPS & PSNR \\\\",
             "\\midrule"]
    for i, (name, b, level, every) in enumerate(rows):
        if i and not every and rows[i - 1][3]:
            lines.append("\\midrule")
        train = [num(level[0], 2), num(level[1], 3)] if level else ["", ""]
        shares = b.get("shares") or {}
        if not b["n"]:
            cells = train + ["\\tbd{}"] * 6
        else:
            cells = train + [num(b["psnr_zero_shot"], 2), num(b["lpips_zero_shot"], 3), num(b["psnr_4k"], 2),
                             num(b["lpips_4k"], 3)] + [share_cell(shares.get(k)) for k in ("lpips", "psnr")]
        lines.append(" & ".join([slim_label(name, b, every)] + cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def slim_caption(rows, n_train):
    """The slim table's caption (a `\\caption{...}` line), built from the rows so the arena sets, the step and the
    window count it states are the table's own. `rows` is `slim_table`'s, `n_train` the training-map windows behind
    the in-domain columns. The decoders it names are the ones `raw_summary.json` records under `definitions`."""
    head = rows[0][1].get("headline", HEADLINE_STEP)
    step = fs.step_label(head)
    over = next((b for _, b, _, every in rows if every and b["n"]), None)
    comparison = next((b for _, b, _, every in rows if not every and b["n"]), None)
    n_over = over["n_wanted"] if over else "the"
    four = (", ".join(map(str, comparison["wanted"][:-1])) + " and " + str(comparison["wanted"][-1])) \
        if comparison else "--"                     # Rohan's form: "maps 6, 7, 8 and 16"
    return ("\\caption{Scene PSNR (dB) and LPIPS one tic ahead, in domain, zero-shot on the unseen maps and after "
            f"{step} adapter updates. Every score is on the scene crop (rows 0 to 207) against the raw frame, "
            "each backbone rendered through its fine-tuned decoder (SD~1's for the U-Net and PixArt-$\\alpha$, "
            f"SD~3.5's own for SD~3.5). In domain: the four training maps, pooled over "
            f"{n_train} validation windows, for reference. Zero-shot and {step}: medians over the {n_over} "
            "unseen maps, before and after adapting on eight episodes of each map (rank-16 LoRA, non-EMA weights). "
            "Recovered, shares per map, then the median: the share of the zero-shot LPIPS rise over the "
            "in-distribution level that the adapter undoes (LPIPS) and of the lost scene PSNR that it regains "
            "(PSNR). The last two rows compare the U-Net's "
            f"adapter with a full fine-tune of all its parameters on the four comparator maps {four} only: both "
            "start from the same zero-shot read, and their shares are against the U-Net's in-domain level.}\n")


TABLE_LABELS = {"unet": "SD 1.4 U-Net", "pixart": "PixArt-$\\alpha$", "sd35": "SD 3.5 Medium"}   # Table 1's names
DIRECTIONAL_SETS = (("training", "val"), ("unseen", "arenas13"))    # Table 1's map rows and the sets that score them
DIRECTIONAL_FILE_RE = re.compile(r"directional_map0*(\d+)_([a-z0-9]+)_ema\.json")


def directional_files(roots, name):
    """{(map, set): path} of a backbone's directional-check reads (tools/directional_check.py's
    `directional_map<NN>_<set>_ema.json`) under the first of `roots` holding a directory `<name>` or `<name>_ema`;
    empty when none does."""
    for root in roots:
        for d in (name, f"{name}_ema"):
            found = sorted(glob.glob(os.path.join(glob.escape(os.path.join(root, d)), "directional_map*_ema.json")))
            out = {(int(m.group(1)), m.group(2)): p for p in found
                   for m in [DIRECTIONAL_FILE_RE.fullmatch(os.path.basename(p))] if m}
            if out:
                return out
    return {}


def directional_reads(roots, name):
    """Table 1's directional column for one backbone: {"training": read, "unseen": read, "files": [paths]}, each read
    {correct, reference, maps, windows} pooled over the set's windows from the per-map files (`summary.correct_frac`:
    the fraction of turning windows whose predicted view turns the other way once the newest turn control is
    swapped; `summary.ref_raw_frac`: the same test on the recorded ground-truth frames), or None for a set with no
    file. Every map's file weighs by its window count, so unequal maps pool like one run."""
    files = directional_files(roots, name)
    out = {"files": sorted(files.values())}
    for label, set_name in DIRECTIONAL_SETS:
        reads = []
        for (m, s), path in sorted(files.items()):
            if s != set_name:
                continue
            with open(path) as f:
                summary = json.load(f)["summary"]
            reads.append((m, summary["windows"], summary["correct_frac"], summary.get("ref_raw_frac")))
        n = sum(w for _, w, _, _ in reads)
        out[label] = None if not n else {
            "correct": sum(w * c for _, w, c, _ in reads) / n,
            "reference": sum(w * r for _, w, _, r in reads) / n if all(r is not None for _, _, _, r in reads)
            else None,
            "maps": [m for m, _, _, _ in reads], "windows": n}
    return out


def results_table(names, home, zero, directional, roots, stamp=""):
    """Table 1 as a tabular: per backbone a training row (the validation read pooled over its windows) and an
    unseen row (medians over the unseen maps) of raw scene PSNR, LPIPS and the directional score
    (`directional_reads`; "--" when the backbone has no read). No reconstruction upper bound (Rohan, Sep 28 night).
    The comment lines give each directional read's maps, windows and the ground-truth frames' own reversal rate.
    Provisional marks are the paper owner's, not printed here."""
    def num(v, d):
        return "--" if v is None else f"{v:.{d}f}"

    lines = [f"% generated by paper/make_raw_figures.py{stamp}: raw scene reads one tic ahead, each backbone through",
             "% its fine-tuned decoder; training maps pooled over the validation windows, unseen maps as medians",
             "% over the unseen maps; directional pooled over each set's turning windows (files listed below)",
             "\\begin{tabular}{llccc}", "\\toprule",
             "Model & Maps & PSNR (dB) & LPIPS & Directional \\\\", "\\midrule"]
    notes = []
    for name in names:
        label, arenas = TABLE_LABELS[fs.backbone_of(name)], sorted(zero[name])
        pooled, d = home[name]["pooled"], directional.get(name) or {}
        rows = [("training", pooled["psnr"], pooled["lpips"], d.get("training")),
                ("unseen", median([zero[name][m]["psnr"] for m in arenas]),
                 median([zero[name][m]["lpips"] for m in arenas]), d.get("unseen"))]
        for i, (maps, psnr, lpips, dr) in enumerate(rows):
            cell = num(dr["correct"] if dr else None, 3)
            lines.append(f"{label if i == 0 else ''} & {maps} & {num(psnr, 2)} & {num(lpips, 3)} & {cell} \\\\")
            notes.append(f"% {label} {maps}: " + (
                f"directional over {len(dr['maps'])} maps ({', '.join(map(str, dr['maps']))}), {dr['windows']} "
                f"windows, ground-truth frames {num(dr['reference'], 3)}" if dr else
                f"no directional read of {name} under {', '.join(rel(r) for r in roots)}"))
    lines += ["\\bottomrule", "\\end{tabular}"] + notes
    return "\n".join(lines) + "\n"


def merged_table(names, home, zero, directional, gstats, block_stats, unet_guards, budgets_final, blocks=(),
                 stamp="", with_8k=True, with_budget=True, directional_at=8000):
    """Tables 1 and 2 as one full-width table (Rohan, Sep 29: nothing in the supplement that was in the body, and
    Table 1 used three quarters of the width). Columns: the row's maps; scene PSNR at 0, 4k, 8k; scene LPIPS at 0,
    4k, 8k; the directional score at 0 and 8k; the recovered shares (PSNR, LPIPS, the score columns' order; per map, then the median); the
    budget. Rows per backbone: its training maps (the in-distribution reads in the 0 columns, Table 1's training
    directional), then the unseen maps over all 13 (Table 1's unseen directional at 0, the guards' median at 8k);
    the U-Net adds its three groups and its full fine-tune (`block_stats["unet_full"]`, directional from its own
    guards). Every number Table 1 and Table 2 print appears here; the comment lines carry what Table 2's do.
    `unet_guards` is `guard_reads`' {map: {directional_0, directional_8k}}; the other backbones' guards come from
    their blocks (`guards` per map and step). The groups' rows run easy, medium, hard (`TABLE_GROUP_ORDER`).
    Without `with_8k` the PSNR and LPIPS columns at 8k are left out (the body reports 4k; the per-map table in
    the appendix keeps 8k) and the directional score keeps its two reads, 0 and 8k. Without `with_budget` the
    budget column is left out (Rohan, Sep 29: the text gives the medians, the appendix the budget per map).
    `directional_at` is the adapted directional column's update count (8000, or 4000 since the 4k guard rows of
    Sep 29); at 4000 the unseen rows take both directional columns from the adapter runs' guard rows (the same
    windows and weights at 0 and 4k, medians over maps), and the training row keeps the pooled read."""
    def num(v, d):
        return "--" if v is None else f"{v:.{d}f}"

    def guard_median(block, step, maps=None):
        vals = [g[step]["directional"] for m, g in (block.get("guards") or {}).items()
                if (maps is None or int(m) in maps) and step in g and g[step].get("directional") is not None]
        return median(vals)

    def unet_guard_median(maps, key):
        return median([unet_guards[m][key] for m in maps if m in unet_guards and unet_guards[m].get(key) is not None])

    lines = [f"% generated by paper/make_raw_figures.py{stamp}: Tables 1 and 2 merged; raw scene reads one tic",
             "% ahead through each backbone's fine-tuned decoder; training maps pooled over the validation windows,",
             "% unseen maps as medians (groups: terciles of the zero-shot LPIPS rise); directional at 0 from the",
             f"% pooled directional reads and at {fs.step_label(directional_at)} from the adapter runs' guard rows (medians over maps);",
             "% recovered: shares per map at 4k, then the median; budget: the first grid read recovering half of the",
             "% map's LPIPS rise over its backbone's in-distribution level",
             "\\begin{tabular}{lrrrrrrrrrrr}", "\\toprule",
             "& \\multicolumn{3}{c}{Scene PSNR (dB) $\\uparrow$} & \\multicolumn{3}{c}{Scene LPIPS $\\downarrow$} & "
             "\\multicolumn{2}{c}{Turn resp.\ $\uparrow$} & \\multicolumn{2}{c}{Recovered (\\%)} & Budget to \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}",
             f"Model, maps & 0 & 4k & 8k & 0 & 4k & 8k & 0 & {fs.step_label(directional_at)} & PSNR & LPIPS & "
             "half the rise \\\\"]
    keys = {"unet": "unet_lora", "pixart": "pixart_lora", "sd35": "sd35_lora"}
    unet_key = "directional_8k" if directional_at == 8000 else "directional_4k"
    paired = directional_at != 8000
    for name in names:
        who = fs.backbone_of(name)
        label, pooled, d = TABLE_LABELS[who], home[name]["pooled"], directional.get(name) or {}
        block = block_stats.get(keys[who]) or {}
        lines.append("\\midrule")
        train_dir = num((d.get("training") or {}).get("correct"), 3)
        lines.append(" & ".join([f"{label}, training maps", num(pooled["psnr"], 2), "", "", num(pooled["lpips"], 3),
                                 "", "", train_dir, "", "", "", ""]) + " \\\\")
        unseen_dir = num((d.get("unseen") or {}).get("correct"), 3)
        if who == "unet":
            g = gstats["all"]
            eight = unet_guard_median(g["arenas"], unet_key)
            if paired:
                unseen_dir = num(unet_guard_median(g["arenas"], "directional_0"), 3)
            lines.append(f"unseen maps (all {g['n']}) & {num(g['psnr_zero_shot'], 2)} & {num(g['psnr_4k'], 2)} & "
                         f"{num(g['psnr_8k'], 2)} & {num(g['lpips_zero_shot'], 3)} & {num(g['lpips_4k'], 3)} & "
                         f"{num(g['lpips_8k'], 3)} & {unseen_dir} & {num(eight, 3)} & "
                         f"{share_cell(g['psnr_share_4k'])} & {share_cell(g['lpips_share_4k'])} & "
                         f"{_budget_cell(g, budgets_final)} \\\\")
            for gname in TABLE_GROUP_ORDER:
                g = gstats[gname]
                if not g["n"]:
                    continue
                lines.append(f"unseen {gname} ({', '.join(map(str, g['arenas']))}) & {num(g['psnr_zero_shot'], 2)} & "
                             f"{num(g['psnr_4k'], 2)} & {num(g['psnr_8k'], 2)} & {num(g['lpips_zero_shot'], 3)} & "
                             f"{num(g['lpips_4k'], 3)} & {num(g['lpips_8k'], 3)} & "
                             f"{num(unet_guard_median(g['arenas'], 'directional_0'), 3)} & "
                             f"{num(unet_guard_median(g['arenas'], unet_key), 3)} & "
                             f"{share_cell(g['psnr_share_4k'])} & {share_cell(g['lpips_share_4k'])} & "
                             f"{_budget_cell(g, budgets_final)} \\\\")
            full = block_stats.get("unet_full")
            if full and full["n"]:
                shares = full.get("shares") or {}
                # the full fine-tune starts from the same 200k checkpoint as the LoRA, so its zero-shot directional
                # read is the LoRA runs' step-0 guard over the same maps (its own scoring starts at 4k)
                full_dir0 = guard_median(full, 0)
                if full_dir0 is None:
                    full_dir0 = unet_guard_median(full["arenas"], "directional_0")
                lines.append(f"{block_label('full fine-tune', full, True)} & {num(full['psnr_zero_shot'], 2)} & "
                             f"{num(full['psnr_4k'], 2)} & {num(full['psnr_8k'], 2)} & "
                             f"{num(full['lpips_zero_shot'], 3)} & {num(full['lpips_4k'], 3)} & "
                             f"{num(full['lpips_8k'], 3)} & {num(full_dir0, 3)} & "
                             f"{num(guard_median(full, directional_at), 3)} & {share_cell(shares.get('psnr'))} & "
                             f"{share_cell(shares.get('lpips'))} & {block_budget_cell(full)} \\\\")
        elif block.get("n"):
            shares = block.get("shares") or {}
            if paired and guard_median(block, 0) is not None:
                unseen_dir = num(guard_median(block, 0), 3)
            lines.append(f"{block_label('unseen maps', block, True)} & {num(block['psnr_zero_shot'], 2)} & "
                         f"{num(block['psnr_4k'], 2)} & {num(block['psnr_8k'], 2)} & "
                         f"{num(block['lpips_zero_shot'], 3)} & {num(block['lpips_4k'], 3)} & "
                         f"{num(block['lpips_8k'], 3)} & {unseen_dir} & {num(guard_median(block, directional_at), 3)} & "
                         f"{share_cell(shares.get('psnr'))} & {share_cell(shares.get('lpips'))} & "
                         f"{block_budget_cell(block)} \\\\")
    if not with_8k:
        lines = [_without_8k(line) for line in lines]
    if not with_budget:
        lines = [_without_budget(line) for line in lines]
    lines += ["\\bottomrule", "\\end{tabular}"] + block_comment_lines(blocks)
    return "\n".join(lines) + "\n"


def _without_8k(line):
    """One line of the merged table without its two 8k score columns (cells 3 and 6 of 12); the header lines are
    rewritten for ten columns."""
    if line.startswith("\\begin{tabular}"):
        return "\\begin{tabular}{lrrrrrrrrr}"
    if line.startswith("& \\multicolumn{3}{c}{Scene PSNR"):
        # two-column groups are narrower than their headings: the headings drop "Scene" (the caption says it)
        return (line.replace("\\multicolumn{3}{c}", "\\multicolumn{2}{c}")
                .replace("{Scene PSNR (dB)", "{PSNR (dB)").replace("{Scene LPIPS", "{LPIPS"))
    if line.startswith("\\cmidrule"):
        return "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}"
    if line.startswith("%") or line.count(" & ") != 11:
        return line
    cells = line[:-len(" \\\\")].split(" & ")
    return " & ".join(c for i, c in enumerate(cells) if i not in (3, 6)) + " \\\\"


def _without_budget(line):
    """One line of the merged table without its last column, the budget."""
    if line.startswith("\\begin{tabular}"):
        return line[:-2] + "}" if line.endswith("r}") else line
    if line.startswith("%") or " & " not in line:
        return line
    return line[:line.rindex(" & ")] + " \\\\"


def merged_caption(names, home, directional, gstats, block_stats):
    """The merged table's caption (a `\\caption{...}` line, four lines at \\footnotesize): what is measured, the
    maps each row's medians run over, the shares' per-map-then-median rule, the budget rule, the 8k note and the
    directional references, built from the same records as the table."""
    n_train = home[PRIMARY]["n"]
    refs = directional.get(PRIMARY) or {}
    ref_train = (refs.get("training") or {}).get("reference")
    ref_unseen = (refs.get("unseen") or {}).get("reference")
    full = block_stats.get("unet_full") or {}
    partial_8k = [TABLE_LABELS[fs.backbone_of(n)] for n in names
                  if (b := block_stats.get({"unet": "unet_lora", "pixart": "pixart_lora", "sd35": "sd35_lora"}
                                           [fs.backbone_of(n)]) or {}).get("n") and 0 < b.get("n_8k", 0) < b["n"]]
    if full.get("n") and 0 < full.get("n_8k", 0) < full["n"]:
        partial_8k.append("the full fine-tune")
    # four lines at \footnotesize (the coordinator, Sep 29): what is measured, the medians' maps, the shares'
    # per-map-then-median rule, the budget rule and the directional references; the 8k note in three words
    return ("\\caption{One tic ahead against the raw frame; training maps pooled over "
            f"{n_train} windows, unseen maps as medians over the {gstats['all']['n']} at 0, 4k and 8k updates "
            "(groups: terciles of the zero-shot LPIPS rise"
            + ("; 8k: over the maps scored there" if partial_8k else "") + "). Directional "
            f"(ground truth {ref_train:.3f} / {ref_unseen:.3f}). Recovered: LPIPS rise and lost PSNR regained, "
            "per map then median; PSNR shares above 100: static maps whose adapted PSNR passes the training-map "
            "level. Budget: first grid read recovering half the rise.}\n")


def groups_table(stats, stamp="", budgets_final=True, blocks=()):
    """Table 2: rows hard, medium, easy (their maps listed; terciles of the zero-shot LPIPS rise) and all maps,
    medians of the U-Net's raw scene reads through the fine-tuned decoder at zero-shot, 4k and 8k, the recovered
    shares at 4k (PSNR, LPIPS, the score columns' order; per map, then the median) and the budget (\\tbd cells until
    `budgets_final`); then
    the per-backbone blocks (`blocks`: [(label, block_summary, decoder identity)]), each \\tbd where it has no map
    yet, "--" at 8k where its grid stops at 4k. No upper bound or gap (Rohan, Sep 28 night)."""
    def num(v, d):
        return "--" if v is None else f"{v:.{d}f}"

    lines = [f"% generated by paper/make_raw_figures.py{stamp}",
             "% medians per group of the U-Net's raw scene reads (fine-tuned decoder, non-EMA adapter weights);",
             "% groups: terciles of the zero-shot LPIPS rise over the training maps (hard = largest rise);",
             "% recovered: shares per map at 4k, then the median; budget: the first grid read that recovers half",
             "% of the map's zero-shot LPIPS rise over its backbone's in-distribution level (\\tbd until every",
             "% read exists)",
             "\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
             "& \\multicolumn{3}{c}{Scene PSNR (dB) $\\uparrow$} & \\multicolumn{3}{c}{Scene LPIPS $\\downarrow$} & "
             "\\multicolumn{2}{c}{Recovered (\\%)} & Budget to \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}",
             "Unseen maps (by zero-shot LPIPS rise) & zero-shot & 4k & 8k & zero-shot & 4k & 8k & PSNR & LPIPS & "
             "half the rise \\\\",
             "\\midrule"]
    for name in list(TABLE_GROUP_ORDER) + ["all"]:
        g = stats[name]
        label = f"{name.capitalize()} ({', '.join(map(str, g['arenas']))})" if name != "all" else f"All {g['n']}"
        if name == "all":
            lines.append("\\midrule")
        lines.append(f"{label} & {num(g['psnr_zero_shot'], 2)} & {num(g['psnr_4k'], 2)} & {num(g['psnr_8k'], 2)} & "
                     f"{num(g['lpips_zero_shot'], 3)} & {num(g['lpips_4k'], 3)} & {num(g['lpips_8k'], 3)} & "
                     f"{share_cell(g['psnr_share_4k'])} & {share_cell(g['lpips_share_4k'])} & "
                     f"{_budget_cell(g, budgets_final)} \\\\")
    if blocks:
        lines.append("\\midrule")
    for label, b, _ in blocks:
        if not b["n"]:
            lines.append(f"{label} & " + " & ".join(["\\tbd{}"] * 9) + " \\\\")
            continue
        shares = b.get("shares") or {}
        lines.append(f"{label} & {num(b['psnr_zero_shot'], 2)} & {num(b['psnr_4k'], 2)} & {num(b['psnr_8k'], 2)} & "
                     f"{num(b['lpips_zero_shot'], 3)} & {num(b['lpips_4k'], 3)} & {num(b['lpips_8k'], 3)} & "
                     f"{share_cell(shares.get('psnr'))} & {share_cell(shares.get('lpips'))} & "
                     f"{block_budget_cell(b)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"] + block_comment_lines(blocks)
    return "\n".join(lines) + "\n"


def block_comment_lines(blocks):
    """The comment lines under a block table, one per block (`blocks`: [(label, block_summary, decoder identity)]):
    its maps, decoder and read kind, the first-grid-point maps, the matched adapter numbers (the full fine-tune),
    maps left out, borrowed zero-shot reads, the coarse-grid note, the 8k map list and the full-grid reruns."""
    lines = []
    for label, b, identity in blocks:
        left = b.get("decoded_only") or []
        first = [] if b.get("coarse_grid") else (b.get("budget_at_first_read") or [])   # the coarse note covers it
        lines.append(f"% {label}: maps {', '.join(map(str, b['arenas'])) or '--'}; "
                     f"decoder {identity or '--'}, reads {b['quantity'] or 'none'}, {b['n']} maps"
                     + (", dagger: against the decoded ground truth" if b["quantity"] not in (None, "raw") else "")
                     + ("; budget at the first point of its grid for maps " + " and ".join(
                         f"{', '.join(str(a) for a, s in first if s == step)} ({fs.step_label(int(step))})"
                         for step in sorted({s for _, s in first})) + ", where the rule cannot resolve finer"
                        if first else "")
                     + (f"; against the adapter's {ma['psnr_4k']:.2f} dB / {ma['lpips_4k']:.3f} and "
                        f"{share_cell(ma['shares'].get('lpips'))} percent (LPIPS) / "
                        f"{share_cell(ma['shares'].get('psnr'))} percent (PSNR) recovered on the same maps "
                        f"{', '.join(map(str, ma['arenas']))}, budget {budget_label(ma['budget_middle'])}"
                        if (ma := b.get("matched_adapter")) else "")
                     + (f"; maps {', '.join(map(str, left))} read against the decoded ground truth only, left out "
                        "until rescored raw" if left else "")
                     + (f"; maps {', '.join(map(str, b['in_progress']))} still training (no "
                        f"{budget_label([b.get('headline', HEADLINE_STEP)])} read yet), left out"
                        if b.get("in_progress") else "")
                     + (f"; zero-shot of maps {', '.join(map(str, b['zero_shot_borrowed']))} from the U-Net LoRA's "
                        "step 0 (the same checkpoint and windows)" if b.get("zero_shot_borrowed") else "")
                     + ("; budget grid starts at 4k: ${\\le}$4k means reached by 4k, the rule cannot resolve finer"
                        if b.get("coarse_grid") else "")
                     + (f"; 8k over the {b['n_8k']} maps scored there so far "
                        f"({', '.join(map(str, b['maps_8k']))})" if 0 < b.get("n_8k", 0) < b["n"] else "")
                     + (f"; full grid (every read 0 to 8k) on maps {', '.join(a for a, s in (b.get('grid_sources') or {}).items() if s == 'full')}"
                        if any(s == "full" for s in (b.get("grid_sources") or {}).values()) else "")
                     + (f"; partial rerun on maps {', '.join(a for a, s in (b.get('grid_sources') or {}).items() if s == 'partial')}"
                        if any(s == "partial" for s in (b.get("grid_sources") or {}).values()) else ""))
    return lines


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------

def fig_3a(entries, floor, key, out_dir):
    """3a without persistence (the supplement): zero-shot raw scene PSNR (or LPIPS) against the frame distance d per
    map, three backbones in their encoding colours and markers, the training maps' validation points over the grey
    d floor, each family's median as a segment across its d range (the step drawn as a step)."""
    stem = f"raw_fig3a_{key}"
    fig, (ax,) = fs.new_figure(SIZES[stem])
    fs.training_band(ax, floor["min"], floor["max"])
    handles = []
    for name, rows in entries.items():
        ent = fs.BACKBONES[fs.backbone_of(name)]
        big = fs.backbone_of(name) == "pixart"
        style = {"marker": ent.marker, "ms": 5.2 if big else fs.MARKER_SIZE, "mfc": "white" if big else ent.colour,
                 "mec": ent.colour if big else "white", "mew": 0.8 if big else fs.MARKER_EDGE}
        pts = [(r["D"], r[key], r["role"]) for r in rows.values() if r["D"] is not None and r.get(key) is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls="none", zorder=2.8 if big else 3, **style)
        for role in ("training", "arena"):
            g = [p for p in pts if p[2] == role]
            if g:
                ax.plot([min(p[0] for p in g), max(p[0] for p in g)], [median([p[1] for p in g])] * 2,
                        color=ent.colour, lw=fs.DATA_LW, solid_capstyle="butt", zorder=2)
        handles.append(Line2D([], [], color=ent.colour, lw=fs.DATA_LW, label=ent.label, **style))
    handles.append(Patch(facecolor=fs.TRAINING_BAND, edgecolor="none", label=BAND_LABEL))
    higher = key == "psnr"
    lo, hi = ax.get_ylim()
    span = hi - lo
    # the key sits where the data leave room: above the arenas in PSNR, under them (the axis from 0) in LPIPS
    ax.set_ylim(lo, hi + span * 0.15) if higher else ax.set_ylim(0.0, hi + span * 0.05)
    ax.legend(handles=handles, loc="upper right" if higher else "lower right", frameon=False, fontsize=fs.ANNOT_PT,
              handlelength=1.6,
              handletextpad=0.4, labelspacing=0.25, borderaxespad=0.1)
    ax.set_xlim(0, 0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_xlabel("frame distance $d$")
    ax.set_ylabel("zero-shot scene\nPSNR (dB) ↑" if higher else "zero-shot scene\nLPIPS ↓")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2 if higher else 0.05))
    return fs.save(fig, out_dir, stem)          # a single supplement panel: no panel letter


def _in_distribution(ax, level, label=None, below=False):
    ax.axhline(level, color=fs.TRAINING_LINE, lw=fs.REF_LW, ls=fs.TRAINING_DASH, zorder=1.4, gid="ref")
    if label:
        ax.text(0.01, level, label, transform=ax.get_yaxis_transform(), ha="left", va="top" if below else "bottom",
                fontsize=fs.MIN_PT, color=fs.TRAINING_LINE, gid="decor")


def before_after_panel(ax, arenas, zero, adapted, level, key, colour_of):
    """One unseen map per column in number order: zero-shot open, after the headline updates filled, a thin
    connector between them, the in-distribution level as the dashed line; marks in the map's group colour. No
    upper bound tick (Rohan, Sep 28 night)."""
    pos = {a: i for i, a in enumerate(arenas)}
    _in_distribution(ax, level)
    for a in arenas:
        c = colour_of(a)
        z, d = zero[a][key], adapted.get(a, {}).get(key)
        if d is not None:
            ax.plot([pos[a], pos[a]], [z, d], color=c, lw=0.8, solid_capstyle="butt", zorder=2.7)
            ax.plot([pos[a]], [d], ls="none", marker="o", ms=fs.MARKER_SIZE, mfc=c, mec=c, mew=0.6, zorder=3.2)
        ax.plot([pos[a]], [z], ls="none", marker="o", ms=fs.MARKER_SIZE, mfc="white", mec=c, mew=0.8, zorder=3.1)
    ax.set_xticks([pos[a] for a in arenas], [str(a) for a in arenas])
    ax.tick_params(axis="x", length=0, labelsize=fs.MIN_PT, pad=1.0)
    ax.set_xlim(-0.6, len(arenas) - 0.4)
    ax.set_xlabel("unseen map")


def curve_panel(ax, curves, key, level, colour_of, steps, named=(), ticks=None, xlabel="adapter updates (log)",
                headline=HEADLINE_STEP):
    """One line per arena against adapter updates (log axis, the 0 read at the left), in the arena's colour, a
    small mark at every read and a filled one at the `headline` read (the filled mark of the before-and-after
    panel), the in-distribution level as the dashed line; the arenas in `named` labelled at their right ends.
    `ticks` are the labelled steps (every grid step keeps a tick; with three reads or fewer all are labelled)."""
    labelled = list(steps) if len(steps) <= 3 else ([t for t in ticks if t in steps] if ticks else None)
    z = fs.step_axis(ax, steps, labelled=labelled)
    for a, c in sorted(curves.items()):
        xs = [z if s == 0 else s for s in c["steps"]]
        ax.plot(xs, c[key], color=colour_of(a), lw=0.8, zorder=3, marker="o", ms=1.2)
        if headline in c["steps"]:
            j = c["steps"].index(headline)
            ax.plot([xs[j]], [c[key][j]], ls="none", marker="o", ms=2.6, mfc=colour_of(a), mec="white",
                    mew=0.3, zorder=3.5)
    _in_distribution(ax, level)
    named = [a for a in named if a in curves]
    if named:
        lo, hi = ax.get_ylim()
        ys = fs.declutter([curves[a][key][-1] for a in named], (hi - lo) * 0.07)
        for a, y in zip(named, ys):
            ax.annotate(str(a), (curves[a]["steps"][-1], y), xytext=(3, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=fs.ANNOT_PT, color=colour_of(a), annotation_clip=False)
    ax.set_xlabel(xlabel)


def _text_entry():
    label = Line2D([], [], ls="none")
    label.set_visible(False)                     # a text-only entry names the entries after it
    return label


def row_legends(fig, key_colours, headline=HEADLINE_STEP):
    """The row's two keys: above it, what each mark and line means; below it, the colour of the map groups by
    the zero-shot LPIPS rise over the training maps (`key_colours`: [(group, colour)])."""
    ink = fs.BACKBONES["unet"].colour
    marks = [Line2D([], [], ls="none", marker="o", ms=fs.MARKER_SIZE, mfc="white", mec=ink, mew=0.8),
             Line2D([], [], ls="none", marker="o", ms=fs.MARKER_SIZE, mfc=ink, mec=ink, mew=0.6),
             Line2D([], [], color=fs.TRAINING_LINE, lw=fs.REF_LW, ls=fs.TRAINING_DASH)]
    fig.legend(marks, ["zero-shot", f"after {fs.step_label(headline)} updates", IN_DISTRIBUTION],
               loc="outside upper center",
               ncol=3, handlelength=1.4, columnspacing=1.4, handletextpad=0.4, borderaxespad=0.1)
    swatches = [_text_entry()] + [Patch(facecolor=c, edgecolor="none") for _, c in key_colours]
    fig.legend(swatches, ["zero-shot LPIPS rise over the training maps:"] + [g for g, _ in key_colours],
               loc="outside lower center", ncol=len(swatches), handlelength=1.0, handleheight=0.8,
               columnspacing=1.0, handletextpad=0.35, borderaxespad=0.1)


def backbone_row_key(fig, curves):
    """The key under the backbone panels (e, f): each curve's backbone and the arenas its median covers."""
    handles, names = backbone_handles(curves)
    n = curves[0][3]
    fig.legend([_text_entry()] + handles, [f"(e, f) median over the {n} unseen maps all three have:" if n else
                                           "(e, f) medians:"] + names,
               loc="outside lower center", ncol=len(names) + 1, handlelength=1.6, columnspacing=1.2,
               handletextpad=0.4, borderaxespad=0.1)


def row_backbones_figure(size, row_size):
    """(figure, the row's four axes, the two backbone axes): the Figure 3 row in a top subfigure of exactly
    `row_size` (so its panels sit where the row alone puts them), the backbone panels in the rest below it."""
    fig = fs.plt.figure(figsize=size, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.04)
    top, bottom = fig.subfigures(2, 1, height_ratios=[row_size[1], size[1] - row_size[1]], hspace=0.0)
    row = top.subplots(1, 4, gridspec_kw={"width_ratios": [1.6, 1.6, 0.8, 0.8]})
    return fig, (top, bottom), list(row), list(bottom.subplots(1, 2))


def fig_row(arenas, zero, adapted, level, trajectories, colour_of, named, out_dir, layout="row",
            key_colours=(), headline=HEADLINE_STEP, backbones=None):
    """The merged Figure 3: how much the unseen arenas improve and how fast. (a) before and after `headline`
    updates in raw PSNR (`adapted` holds that step's reads), (b) in LPIPS, (c) and (d) each arena's raw trajectory
    over the adapter reads that exist, sharing its y axis with (a) and (b) so the filled mark is the curve's read at
    the headline step. `layout` "grid" draws two rows of two (PSNR above, LPIPS below); "row_backbones" (a body
    candidate) adds (e) and (f) on a line of their own under the unchanged row (six panels in one 5.5 in row leave
    the 13 arena numbers overlapping): the median raw PSNR and LPIPS of each backbone's adapter over the arenas they
    share, on their own y axes, from `backbones` = (curves, levels) as `fig_backbones` takes them, keyed below."""
    stem = {"row": "raw_row", "row_150": "raw_row_150", "grid": "raw_row_grid",
            "row_backbones": "raw_row_backbones"}[layout]
    extra, subs = (), None
    if layout in ("row", "row_150"):        # the same row; "row_150" only at the 1.5 in slot height
        fig, (pa, la, pc, lc) = fs.new_figure(SIZES[stem], ncols=4, width_ratios=[1.6, 1.6, 0.8, 0.8], wspace=0.02)
    elif layout == "row_backbones":
        fig, subs, (pa, la, pc, lc), extra = row_backbones_figure(SIZES[stem], SIZES["raw_row"])
    else:
        fig, (pa, pc, la, lc) = fs.new_figure(SIZES[stem], ncols=2, nrows=2, width_ratios=[1.5, 1.0], wspace=0.05,
                                              hspace=0.06)
    in_row = layout != "grid"
    pc.sharey(pa)
    lc.sharey(la)
    for ax, key in ((pa, "psnr"), (la, "lpips")):
        before_after_panel(ax, arenas, zero, adapted, level[key], key, colour_of)
    steps = sorted({s for c in trajectories.values() for s in c["steps"]})
    for ax, key in ((pc, "psnr"), (lc, "lpips")):
        curve_panel(ax, trajectories, key, level[key], colour_of, steps, named=named if key == "psnr" else (),
                    xlabel="updates (log)" if in_row else "adapter updates (log)",
                    ticks=ROW_TICKS if in_row else GRID_TICKS, headline=headline)
    pa.set_ylabel(PSNR_LABEL)
    la.set_ylabel(LPIPS_LABEL)
    if layout == "grid":
        for ax in (pc, lc):
            ax.tick_params(axis="y", labelleft=False)
    pa.yaxis.set_major_locator(ticker.MultipleLocator(2))
    la.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    if extra:
        curves, levels = backbones
        b_steps = sorted({st for _, _, pts, _ in curves for st in pts})
        for ax, j, label in ((extra[0], 0, PSNR_LABEL), (extra[1], 1, LPIPS_LABEL)):
            z = fs.step_axis(ax, b_steps, label="adapter updates (log)",
                             labelled=[t for t in GRID_TICKS if t in b_steps])
            backbone_curves(ax, curves, j, levels, z, headline)
            drawn = [levels[k][j] for k, *_ in curves if k in levels and levels[k][j] is not None]
            if drawn:
                ax.text(0.01, max(drawn), "training maps", transform=ax.get_yaxis_transform(), ha="left",
                        va="bottom", fontsize=fs.MIN_PT, color=fs.TRAINING_LINE, gid="decor")
            ax.set_ylabel(label)
        extra[0].yaxis.set_major_locator(ticker.MultipleLocator(1))
        extra[1].yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        backbone_row_key(subs[1], curves)
    for ax, letter in zip([pa, la, pc, lc, *extra], "abcdef"):
        fs.panel_letter(ax, letter)
    if key_colours:
        row_legends(subs[0] if subs else fig, key_colours, headline)
    return fs.save(fig, out_dir, stem)


def shared_arenas(arena_sets):
    """The arenas every LoRA block (one per backbone) has finished, sorted: the like-for-like set of the second
    backbone panel. The full fine-tune runs on a subset by design, so it does not shrink the set."""
    lora = [set(v) for k, v in arena_sets.items() if k.endswith("_lora")]
    return sorted(set.intersection(*lora)) if lora else []


def fullft_arenas(lora_arenas, full_arenas):
    """The arenas of the full fine-tune panel: the comparator arenas the U-Net LoRA has, once the full fine-tune has
    every one of them (sorted); None before then, or when there is none."""
    wanted = sorted(set(COMPARATOR_ARENAS) & set(lora_arenas))
    return wanted if wanted and set(wanted) <= set(full_arenas) else None


def fullft_label(points):
    """The full fine-tune's key entry, named by the reads it has: "full fine-tune (0, 4k)"."""
    return f"full fine-tune ({', '.join(fs.step_label(int(st)) for st in sorted(points))})"


def panel_points(raw, arenas=None):
    """{step: (median PSNR, median LPIPS)} over `arenas` (every arena of `raw` when None) of one block's raw reads
    (`raw` is {arena: {step: read}}), at the steps every one of those maps has (a median over the maps that happen
    to carry a step would mix populations while the full-grid reruns land); None when the block lacks one of
    `arenas`."""
    if arenas is not None and any(m not in raw for m in arenas):
        return None
    pick = {m: raw[m] for m in (raw if arenas is None else arenas)}
    steps = sorted({st for r in pick.values() for st in r} & set.intersection(*(set(r) for r in pick.values()))) \
        if pick else []
    return {st: (median([r[st]["psnr"] for r in pick.values()]),
                 median([r[st]["lpips"] for r in pick.values()])) for st in steps}


def backbone_style(key):
    """(entity, marker, line style) of a Table 2 block's curve: its backbone's colour and marker, the full fine-tune
    black with round points."""
    who, ls = BLOCK_STYLE[key]
    ent = fs.FULL_FINE_TUNE if who == "full" else fs.BACKBONES[who]
    return ent, ("o" if who == "full" else ent.marker), ls


def backbone_curves(ax, curves, j, levels, z, headline=HEADLINE_STEP, marker_size=1.6, headline_size=4.6):
    """Draw each block's median curve of quantity `j` (0 PSNR, 1 LPIPS) on a step axis whose 0 read sits at `z`:
    a line with small marks at the reads (the full fine-tune as bare points), a large filled mark at the headline
    read, and the backbone's training-map level as a thin dashed line in its colour. Returns the curves' right ends
    [(x, y, label, colour)], the label carrying the arena count when there is one."""
    ends = []
    for key, label, pts, n in curves:
        ent, marker, ls = backbone_style(key)
        full = BLOCK_STYLE[key][0] == "full"
        xs = [z if st == 0 else st for st in sorted(pts)]
        ys = [pts[st][j] for st in sorted(pts)]
        # the full fine-tune is scored at its two reads only: points, no connector (a line between them would
        # draw values nobody measured, and it crossed the LoRA curve where neither was read)
        ax.plot(xs, ys, color=ent.colour, ls="none" if full else ls, lw=fs.DATA_LW, marker=marker,
                ms=2.6 if full else marker_size, mfc=ent.colour, mec=ent.colour, zorder=3)
        if headline in pts:
            ax.plot([headline], [pts[headline][j]], ls="none", marker=marker, ms=headline_size, mfc=ent.colour,
                    mec="white", mew=0.5, zorder=3.5)
        ends.append((xs[-1], ys[-1], label if n is None else f"{label} ({n})", ent.colour))
        if key in levels and levels[key][j] is not None and not full:     # the full FT shares the U-Net's
            ax.axhline(levels[key][j], color=ent.colour, lw=fs.MIN_LW, ls=fs.TRAINING_DASH, zorder=1.5, gid="ref")
    return ends


def backbone_handles(curves, marker_size=1.6):
    """Key entries naming each curve's backbone: its line and mark."""
    out = []
    for key, *_ in curves:
        ent, marker, ls = backbone_style(key)
        out.append(Line2D([], [], color=ent.colour, ls=ls, lw=fs.DATA_LW, marker=marker, ms=marker_size + 1.0,
                          mfc=ent.colour, mec=ent.colour))
    return out, [label for _, label, *_ in curves]


def backbone_key(fig, curves, headline=HEADLINE_STEP):
    """The separate body figure's key, above the panels: the backbones, the filled headline mark and the training
    maps' dashed level (drawn in each backbone's colour; keyed in the first one's)."""
    handles, labels = backbone_handles(curves)
    ink = backbone_style(curves[0][0])[0].colour
    handles += [Line2D([], [], ls="none", marker="o", ms=4.6, mfc=ink, mec="white", mew=0.5),
                Line2D([], [], color=ink, lw=fs.MIN_LW, ls=fs.TRAINING_DASH)]
    labels += [f"after {fs.step_label(headline)} updates", IN_DISTRIBUTION]
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), handlelength=1.6, columnspacing=1.4,
               handletextpad=0.4, borderaxespad=0.1)


def fig_backbones(curves, levels, out_dir, headline=HEADLINE_STEP, stem="raw_backbones", keyed=False):
    """The per-backbone adaptation panel: the median over each block's arenas of raw scene PSNR (left) and LPIPS
    (right) against adapter updates (log axis, the 0 read at the left), one curve per block in its backbone's colour
    and marker (the full fine-tune as black points at its reads, no connector), the headline read filled, each
    backbone's training-map level as a short dashed segment at the right edge, each curve labelled at its right end
    with its arena count (none when `n` is None). `curves` is [(key, label, {step: (psnr, lpips)}, n)], `levels`
    {key: (psnr, lpips)}; `stem` names the variant (the shared one draws every curve over the same arenas). With
    `keyed` (the separate body candidate) a key above the panels names the backbones and the marks instead of the end
    labels and the "training maps" text; the arena count goes to the caption."""
    fig, (px, lx) = fs.new_figure(SIZES[stem], ncols=2, wspace=0.08)
    steps = sorted({st for _, _, pts, _ in curves for st in pts})
    for ax, j in ((px, 0), (lx, 1)):
        z = fs.step_axis(ax, steps, labelled=[t for t in GRID_TICKS if t in steps] or None)
        ends = backbone_curves(ax, curves, j, levels, z, headline)
        drawn_levels = [levels[k][j] for k, *_ in curves if k in levels and levels[k][j] is not None]
        if drawn_levels and not keyed:
            top = max(drawn_levels)                  # above the highest level line, clear of the curves
            ax.text(0.01, top, "training maps", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                    fontsize=fs.MIN_PT, color=fs.TRAINING_LINE, gid="decor")
        if j == 0 and not keyed:
            # every label right of the rightmost end: a curve that stops at 4k would put its label on the
            # 4k-to-8k stretch of one that runs on, where the curves end within a tenth of a dB of each other
            x_label = max(x for x, *_ in ends)
            # the declutter gap is one label line with a little air (1.4 times the label's size) in data units at the
            # axis's height after the layout engine has run (constrained layout shrinks the axes at draw time)
            fig.draw_without_rendering()
            height_pt = ax.get_window_extent().height * 72.0 / fig.dpi
            gap = 1.4 * fs.ANNOT_PT * (ax.get_ylim()[1] - ax.get_ylim()[0]) / height_pt
            fs.end_labels(ax, [(x_label, y, text, c) for _, y, text, c in ends], gap=gap)
        ax.set_xlabel("adapter updates (log)")
    px.set_ylabel(PSNR_LABEL)
    lx.set_ylabel(LPIPS_LABEL)
    px.yaxis.set_major_locator(ticker.MultipleLocator(1))
    lx.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    fs.panel_letter(px, "a")
    fs.panel_letter(lx, "b")
    if keyed:
        backbone_key(fig, curves, headline)
    return fs.save(fig, out_dir, stem)


def per_arena_backbone_panel(ax, arenas, backbones, key, levels, marker_size, offset=0.27, headline=HEADLINE_STEP):
    """(a) or (b) of the per-backbone row: one slot per unseen map in number order; inside it each backbone at its
    own small horizontal offset, in its encoding colour and marker: zero-shot open, the `headline` read filled, a
    thin connector between them; each backbone's training-map level as a thin dashed line in its colour. No
    reconstruction upper bound here (Rohan, Sep 28 night: the SD 3.5 ticks at 29 to 33 dB stretched the axis and
    shrank the drops and recoveries; the y axes now fit the reads and the levels). `backbones` is [(block key,
    zero-shot reads {map: {psnr, lpips}}, headline reads {map: {psnr, lpips}})], `levels` {block key: (psnr,
    lpips)}. The paper calls the arenas "unseen maps" in this figure; the tick labels stay the map numbers."""
    pos = {a: i for i, a in enumerate(arenas)}
    j = 0 if key == "psnr" else 1
    for k, (bkey, zero, after) in enumerate(backbones):
        ent, marker, _ = backbone_style(bkey)
        dx = (k - (len(backbones) - 1) / 2) * offset
        if bkey in levels and levels[bkey][j] is not None:
            ax.axhline(levels[bkey][j], color=ent.colour, lw=fs.MIN_LW, ls=fs.TRAINING_DASH, zorder=1.4, gid="ref")
        for a in arenas:
            if a not in zero:
                continue
            x = pos[a] + dx
            z, d = zero[a][key], after.get(a, {}).get(key)
            if d is not None:
                ax.plot([x, x], [z, d], color=ent.colour, lw=0.7, solid_capstyle="butt", zorder=2.7)
                ax.plot([x], [d], ls="none", marker=marker, ms=marker_size, mfc=ent.colour, mec=ent.colour, mew=0.5,
                        zorder=3.2)
            ax.plot([x], [z], ls="none", marker=marker, ms=marker_size, mfc="white", mec=ent.colour, mew=0.7,
                    zorder=3.1)
    ax.set_xticks([pos[a] for a in arenas], [str(a) for a in arenas])
    ax.tick_params(axis="x", length=0, labelsize=fs.MIN_PT, pad=1.0)
    ax.set_xlim(-0.6, len(arenas) - 0.4)
    ax.set_xlabel("unseen map")


def row_v2_keys(fig, backbones, headline=HEADLINE_STEP):
    """The per-backbone row's two keys: above, each backbone's colour and marker, then what open and filled mean;
    below, the training-map level. The panels draw the marks and the level in each backbone's colour; the key
    shows them in the first backbone's (as `backbone_key` does), and its label says so."""
    handles, labels = [], []
    for bkey, *_ in backbones:
        ent, marker, _ = backbone_style(bkey)
        handles.append(Line2D([], [], color=ent.colour, ls="none", marker=marker, ms=fs.MARKER_SIZE,
                              mfc=ent.colour, mec=ent.colour))
        labels.append(ent.label)
    grey = backbone_style(backbones[0][0])[0].colour
    handles += [Line2D([], [], color=grey, ls="none", marker="o", ms=fs.MARKER_SIZE, mfc="white", mec=grey, mew=0.8),
                Line2D([], [], color=grey, ls="none", marker="o", ms=fs.MARKER_SIZE, mfc=grey, mec=grey, mew=0.6)]
    labels += ["zero-shot", f"after {fs.step_label(headline)} updates"]
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), handlelength=1.2, columnspacing=1.2,
               handletextpad=0.4, borderaxespad=0.1)
    fig.legend([Line2D([], [], color=grey, lw=fs.MIN_LW, ls=fs.TRAINING_DASH)],
               [f"{IN_DISTRIBUTION}, per backbone in its color"], loc="outside lower center", ncol=1,
               handlelength=1.4, handletextpad=0.4, borderaxespad=0.1)


def fig_row_v2(arenas, backbones, curves, levels, out_dir, headline=HEADLINE_STEP, tall=False):
    """Figure 3 rebuilt per backbone (Rohan, Sep 28 night): (a) raw scene PSNR and (b) LPIPS per unseen map,
    zero-shot (open) to the `headline` read (filled), for every backbone in its colour at its own offset inside the
    map's slot, with each backbone's training-map level (dashed; no upper bound ticks); (c, d)
    the shared panel's median curves against adapter updates (`curves` and `levels` as `fig_backbones` takes them).
    One row of four at the slot height, or with `tall` two rows of two (a, b above c, d) with the per-arena panels
    at twice the width and full-size marks."""
    stem = "raw_row_v2_tall" if tall else "raw_row_v2"
    if tall:
        fig, (pa, la, pc, lc) = fs.new_figure(SIZES[stem], ncols=2, nrows=2, wspace=0.08, hspace=0.08)
    else:
        fig, (pa, la, pc, lc) = fs.new_figure(SIZES[stem], ncols=4, width_ratios=[1.3, 1.3, 1.1, 1.1], wspace=0.02)
    # three marks per arena slot: full-size marks fit only when the panel is two rows' width
    marker = fs.MARKER_SIZE if tall else 2.8
    for ax, key in ((pa, "psnr"), (la, "lpips")):
        per_arena_backbone_panel(ax, arenas, backbones, key, levels, marker, headline=headline)
    pa.set_ylabel(PSNR_LABEL)
    la.set_ylabel(LPIPS_LABEL)
    if not tall:   # thirteen map numbers share a narrow panel: the smallest allowed size keeps them apart
        for ax in (pa, la):
            ax.tick_params(axis="x", labelsize=fs.MIN_PT, pad=1.5)
    pa.yaxis.set_major_locator(ticker.MultipleLocator(2))
    la.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    b_steps = sorted({st for _, _, pts, _ in curves for st in pts})
    for ax, j, label in ((pc, 0, PSNR_LABEL), (lc, 1, LPIPS_LABEL)):
        z = fs.step_axis(ax, b_steps, label="adapter updates (log)" if tall else "updates (log)",
                         labelled=[t for t in (GRID_TICKS if tall else ROW_TICKS) if t in b_steps])
        backbone_curves(ax, curves, j, levels, z, headline)
        drawn = [levels[k][j] for k, *_ in curves if k in levels and levels[k][j] is not None]
        if drawn:
            ax.text(0.01, max(drawn), "training maps", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                    fontsize=fs.MIN_PT, color=fs.TRAINING_LINE, gid="decor")
        ax.set_ylabel(label)
    pc.yaxis.set_major_locator(ticker.MultipleLocator(1))
    lc.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for ax, letter in zip((pa, la, pc, lc), "abcd"):
        fs.panel_letter(ax, letter)
    row_v2_keys(fig, backbones, headline)
    return fs.save(fig, out_dir, stem)


def fig_arenas_raw(arenas, trajectories, adaptation, level, colour_of, out_dir, headline=HEADLINE_STEP):
    """Appendix: one small panel per unseen map, raw scene LPIPS against adapter updates (log axis, the 0 read at
    the left) in the map's group colour, the headline read filled, the budget's threshold (dotted: half of the
    map's zero-shot LPIPS rise recovered) and the training maps' level (dashed); shared axes in map order, the
    key in the first empty slot (or under the panels when none is empty). LPIPS because the budget and the groups
    are defined on the LPIPS rise (Rohan, Sep 29 00:30); no upper bound."""
    stem = "raw_figA_adapt_arenas"
    recs = [m for m in arenas if m in trajectories]
    ncols = min(5, len(recs))
    nrows = math.ceil(len(recs) / ncols)
    fig, axes = fs.new_figure(SIZES[stem], ncols=ncols, nrows=nrows, sharex=True, sharey=True, wspace=0.02,
                              hspace=0.03)
    steps = sorted({s for m in recs for s in trajectories[m]["steps"]})
    values = [v for m in recs for v in trajectories[m]["lpips"]] + [level]
    ink = fs.BACKBONES["unet"].colour
    key = ([Line2D([], [], color=ink, lw=fs.DATA_LW, marker="o", ms=2.5),
            Line2D([], [], ls="none", marker="o", ms=3.5, mfc=ink, mec=ink),
            Line2D([], [], color=fs.CONTEXT_INK, lw=fs.MIN_LW, ls=(0, (1, 1.2))),
            Line2D([], [], color=fs.TRAINING_LINE, lw=fs.REF_LW, ls=fs.TRAINING_DASH)],
           ["scene LPIPS", f"after {fs.step_label(headline)} updates", "half the LPIPS rise recovered",
            IN_DISTRIBUTION])
    keyed = False
    for i, ax in enumerate(axes):
        if i >= len(recs):
            if keyed:
                ax.remove()
            else:
                ax.set_axis_off()
                ax.set_gid("decor")
                ax.legend(*key, loc="center", frameon=False, fontsize=fs.MIN_PT, handlelength=1.6)
                keyed = True
            continue
        m = recs[i]
        t, r = trajectories[m], adaptation[m]
        z = fs.step_axis(ax, steps, label="")
        xs = [z if s == 0 else s for s in t["steps"]]
        ax.plot(xs, t["lpips"], color=colour_of(m), lw=fs.DATA_LW, marker="o", ms=2.5, zorder=3)
        if headline in t["steps"]:
            j = t["steps"].index(headline)
            ax.plot([xs[j]], [t["lpips"][j]], ls="none", marker="o", ms=3.5, mfc=colour_of(m), mec=colour_of(m),
                    zorder=3.5)
        ax.axhline(r["threshold"], color=fs.CONTEXT_INK, lw=fs.MIN_LW, ls=(0, (1, 1.2)), zorder=1.4, gid="ref")
        ax.axhline(level, color=fs.TRAINING_LINE, lw=fs.REF_LW, ls=fs.TRAINING_DASH, zorder=1.4, gid="ref")
        # the map's name at the top right: the curves fall from the top left and the level line sits at the bottom
        ax.text(0.97, 0.95, f"map {m}", transform=ax.transAxes, ha="right", va="top", fontsize=fs.ANNOT_PT,
                gid="decor")
        ax.set_ylim(min(values) - 0.02, max(values) + 0.02)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        if i % ncols == 0:
            ax.set_ylabel("scene LPIPS")
        ax.tick_params(labelbottom=i + ncols >= len(recs))
    if not keyed:
        fig.legend(*key, loc="outside lower center", ncol=len(key[1]), handlelength=1.4, columnspacing=1.0)
    fig.supxlabel("adapter updates (log scale)", fontsize=fs.LABEL_PT)
    return fs.save(fig, out_dir, stem)


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path):
    return os.path.relpath(path, REPO) if os.path.abspath(path).startswith(REPO) else path


def build_parser():
    p = argparse.ArgumentParser(description="The persistence-free set: raw scene PSNR and LPIPS.")
    p.add_argument("--fresh-root", default=os.path.join(REPO, "results", "fresh_rescore"))
    p.add_argument("--directional-roots", nargs="+",
                   default=[os.path.join(REPO, "results", "directional_fresh"),
                            os.path.join(REPO, "results", "fresh_rescore", "directional_fresh")],
                   help="where each backbone's directional-check reads live (<row>/ or <row>_ema/), for Table 1")
    p.add_argument("--distances", default=os.path.join(REPO, "results", "distance_v2", "distances_sd1.json"))
    p.add_argument("--training-distances",
                   default=os.path.join(REPO, "results", "distance_study", "figure_unet_h1", "stats.json"))
    p.add_argument("--adapt-root", default=os.path.join(REPO, "results", "adapt"),
                   help="every backbone's adaptation runs: Table 2's per-backbone blocks and the backbone panel")
    p.add_argument("--sd35-full-root", default=os.path.join(REPO, "results", "adapt_sd35_full"),
                   help="SD 3.5's full-grid reruns (ten reads 0 to 8k), preferred over the coarse-grid runs")
    p.add_argument("--pixart-full-root", default=os.path.join(REPO, "results", "adapt_pixart_full"),
                   help="PixArt-alpha's full-grid reruns, preferred over the coarse-grid runs")
    p.add_argument("--fullft-g8k-root", default=os.path.join(REPO, "results", "adapt_fullft_g8k"),
                   help="the U-Net full fine-tunes on the 8k grid (map<NN>/), preferred over the 4k-only originals")
    p.add_argument("--adapt-glob",
                   default=os.path.join(REPO, "results", "adapt", "unet200k_arenas13_map??_r16_k8_s0_g8k"),
                   help="the 8k-grid run directories whose guard rows give forgetting and the directional check")
    p.add_argument("--family-step", default=os.path.join(REPO, "results", "family_step", "family_step.json"),
                   help="the zero-shot latent skill S0 per arena (tools/family_step.py), for the Spearman numbers")
    p.add_argument("--headline-step", type=int, default=HEADLINE_STEP,
                   help="the adapter read the filled marks and the medians report (8k stays the check)")
    p.add_argument("--group-by", choices=("lpips_rise", "zero_shot_psnr"), default="lpips_rise",
                   help="the key that groups and colours the unseen maps: terciles of the zero-shot LPIPS rise "
                        "over the training maps, the budget's own quantity (default)")
    p.add_argument("--out-dir", default=os.path.join(HERE, "figures", "raw"))
    p.add_argument("--tables-dir", default=os.path.join(HERE, "tables", "tuned"))
    p.add_argument("--summary", default=os.path.join(HERE, "tables", "tuned", "raw_summary.json"))
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    fs.style()
    arena_d = maf.load_distances(a.distances)
    train_d = training_distances(a.training_distances)
    floor = distance_floor(a.distances)
    inputs = [a.distances, a.training_distances]
    zero, home = {}, {}
    rows_used = backbone_rows(a.fresh_root)
    names = list(rows_used)
    sd35_name = next(n for n in names if n.startswith("sd35"))
    for name in names:
        row, sfx = rows_used[name]
        zero[name], files = row_reads(a.fresh_root, row, sfx)
        home[name] = home_reads(a.fresh_root, row, sfx,
                                extra=("scene_psnr_dec_tuned", "scene_lpips_dec_tuned") if name == PRIMARY else ())
        inputs += files + [home[name]["path"]]
    arenas = sorted(zero[PRIMARY])
    if not arenas:
        raise SystemExit(f"no zero-shot reads of {PRIMARY} under {a.fresh_root}")
    z0 = {m: zero[PRIMARY][m]["psnr"] for m in arenas}

    # the step without persistence (the supplement's 3a)
    entries, step = {}, {}
    for name in names:
        rows = {m: {**zero[name][m], "D": arena_d.get(m), "role": "arena"} for m in zero[name]}
        rows.update({m: {**home[name]["maps"][m], "D": train_d.get(m), "role": "training"}
                     for m in TRAINING_MAPS if m in home[name]["maps"]})
        entries[name] = rows
        tr, ar = [rows[m] for m in TRAINING_MAPS if m in rows], [rows[m] for m in arenas if m in rows]
        step[name] = {"psnr": step_check([r["psnr"] for r in tr], [r["psnr"] for r in ar]),
                      "lpips": step_check([r["lpips"] for r in tr], [r["lpips"] for r in ar], higher_is_better=False),
                      "spearman_d_psnr_arenas": maf.spearman([r["D"] for r in ar], [r["psnr"] for r in ar])["rho"],
                      "spearman_d_lpips_arenas": maf.spearman([r["D"] for r in ar], [r["lpips"] for r in ar])["rho"]}
    written = fig_3a(entries, floor, "psnr", a.out_dir) + fig_3a(entries, floor, "lpips", a.out_dir)

    # Table 1 (the supplement's copy of the body's hand-set table): the same reads plus the directional column
    directional = {name: directional_reads(a.directional_roots, name) for name in names}
    inputs += [p for d in directional.values() for p in d["files"]]
    full = os.path.join(a.tables_dir, "results_full.tex")
    os.makedirs(a.tables_dir, exist_ok=True)
    with open(full, "w") as f:
        f.write(results_table(names, home, zero, directional, a.directional_roots))
    written.append(full)

    # the adapter's raw reads by step, the budget, before and after, the raw trajectories
    rows = adapter_rows(a.fresh_root)
    if 8000 not in rows:
        raise SystemExit(f"no 8k raw read of the 8k-grid adapter under {a.fresh_root}")
    adapted = {}
    for s, row in rows.items():
        adapted[s], files = row_reads(a.fresh_root, row, "_tuned")
        inputs += files
    base4k, files = row_reads(a.fresh_root, BASE_4K_ROW, "_tuned")
    inputs += files
    level = {k: home[PRIMARY]["pooled"][k] for k in ("psnr", "lpips")}      # the U-Net's in-distribution level
    grid_complete = all(s in adapted and all(m in adapted[s] for m in arenas) for s in GRID if s > 0)
    adaptation, trajectories = {}, {}
    for m in arenas:
        l0 = zero[PRIMARY][m]["lpips"]
        psnr_reads = [(s, adapted[s].get(m, {}).get("psnr")) for s in sorted(adapted)]
        lpips_reads = [(s, adapted[s].get(m, {}).get("lpips")) for s in sorted(adapted)]
        threshold = lpips_threshold(l0, level["lpips"])
        p4, l4 = adapted.get(a.headline_step, {}).get(m, {}).get("psnr"), \
            adapted.get(a.headline_step, {}).get(m, {}).get("lpips")
        # the deficits against the in-distribution level: the zero-shot LPIPS rise (the budget's and the groups'
        # key, Rohan Sep 29 00:30) and the PSNR deficit; the recovered shares at the headline are per map, the
        # medians come later
        adaptation[m] = {"zero_shot": z0[m], "lpips_zero_shot": l0,
                         "psnr_deficit": level["psnr"] - z0[m], "lpips_rise": l0 - level["lpips"],
                         "threshold": threshold,
                         "psnr": {str(s): v for s, v in psnr_reads},
                         "lpips": {str(s): v for s, v in lpips_reads},
                         "lpips_share_4k": share(None if l4 is None else l0 - l4, l0 - level["lpips"]),
                         "psnr_share_4k": share(None if p4 is None else p4 - z0[m], level["psnr"] - z0[m]),
                         "budget": budget_step(lpips_reads, threshold)}
        steps = [s for s, v in psnr_reads if v is not None]
        if 8000 in steps:
            trajectories[m] = {"steps": [0] + steps,
                               "psnr": [z0[m]] + [adapted[s][m]["psnr"] for s in steps],
                               "lpips": [l0] + [adapted[s][m]["lpips"] for s in steps]}
    complete = sorted(trajectories)
    budgets = [{"budget": adaptation[m]["budget"]} for m in complete]
    counts = {"n": len(complete), "rule": BUDGET_RULE, "in_distribution": level,
              "grid_complete": grid_complete, "reads": sorted(adapted),
              "past_half_by_4k": sum(1 for b in budgets if b["budget"] is not None and b["budget"] <= 4000),
              "past_half_by_8k": sum(1 for b in budgets if b["budget"] is not None),
              "median_budget": maf.censored_median_entry(budgets, "budget", 8000),
              "censored": [m for m, b in zip(complete, budgets) if b["budget"] is None],
              "budget_at_first_read": [m for m in complete if adaptation[m]["budget"] == min(s for s in GRID if s)]}
    ad8 = adapted[8000]
    if a.headline_step not in adapted:
        raise SystemExit(f"no raw read of the adapter at the headline step {a.headline_step}")
    after = adapted[a.headline_step]
    ba = {}
    for key in ("psnr", "lpips"):
        better = (lambda v, lv: v >= lv) if key == "psnr" else (lambda v, lv: v <= lv)
        z = {m: zero[PRIMARY][m][key] for m in arenas}
        ba[key] = {"in_distribution": level[key], "headline_step": a.headline_step,
                   "median_zero_shot": median(list(z.values())),
                   "median_adapted": median([after[m][key] for m in arenas if m in after]),
                   "median_drop": median([level[key] - z[m] for m in arenas]),
                   "median_recovery": median([after[m][key] - z[m] for m in arenas if m in after]),
                   "median_left_to_in_distribution": median([level[key] - after[m][key] for m in arenas if m in after]),
                   "arenas_at_or_past_in_distribution": sum(1 for m in arenas if m in after
                                                            and better(after[m][key], level[key])),
                   # the 8k check
                   "median_adapted_8k": median([ad8[m][key] for m in arenas if m in ad8]),
                   "median_recovery_8k": median([ad8[m][key] - z[m] for m in arenas if m in ad8]),
                   "arenas_at_or_past_in_distribution_8k": sum(1 for m in arenas if m in ad8
                                                               and better(ad8[m][key], level[key]))}

    # the grouping and colouring key: the zero-shot LPIPS rise over the training maps (larger = harder; terciles
    # hard, medium, easy, the budget's own quantity), or zero-shot PSNR
    if a.group_by == "lpips_rise":
        key_of = {m: adaptation[m]["lpips_rise"] for m in arenas}
        harder, key_name = True, "zero-shot LPIPS rise over the training maps"
    else:
        key_of, harder, key_name = dict(z0), False, "zero-shot raw scene PSNR (dB)"
    ease = {m: (-v if harder else v) for m, v in key_of.items()}          # low = hard
    named = (min(complete, key=lambda m: ease[m]), max(complete, key=lambda m: ease[m])) if complete else ()
    groups = group_arenas({m: key_of[m] for m in complete}, higher_is_harder=harder)
    # one shade per group, so the key's three swatches are exactly the marks' colours (the paper owner's check:
    # a continuous ramp keyed by three swatches made two hard maps read as medium)
    shade_of, key_colours = group_colours(groups)

    def colour_of(m):
        return shade_of.get(m, fs.BACKBONES["adapter"].colour)
    for layout in ("row", "row_150", "grid"):
        written += fig_row(arenas, zero[PRIMARY], after, level, trajectories, colour_of, named, a.out_dir,
                           layout=layout, key_colours=key_colours, headline=a.headline_step)
    written += fig_arenas_raw(arenas, trajectories, adaptation, level["lpips"], colour_of, a.out_dir,
                              a.headline_step)

    # Table 2: the unseen maps grouped by the key
    at = {s: adapted.get(s, {}) for s in (4000, 8000)}
    per = {m: {"zero_shot": z0[m], "psnr_4k": at[4000].get(m, {}).get("psnr"), "psnr_8k": ad8[m]["psnr"],
               "lpips_zero_shot": zero[PRIMARY][m]["lpips"],
               "lpips_4k": at[4000].get(m, {}).get("lpips"), "lpips_8k": ad8[m]["lpips"],
               "lpips_share_4k": adaptation[m]["lpips_share_4k"], "psnr_share_4k": adaptation[m]["psnr_share_4k"],
               "budget": adaptation[m]["budget"]} for m in complete}
    gstats = group_stats(groups, per)
    os.makedirs(a.tables_dir, exist_ok=True)
    table = os.path.join(a.tables_dir, "adapt_groups.tex")
    with open(table, "w") as f:
        f.write(groups_table(gstats, budgets_final=grid_complete))
    written.append(table)

    # Table 2's per-backbone blocks and the per-backbone panel: every backbone's adaptation runs
    # the SD 3.5 blocks need only the latest checkpoint's training-map read for their gap (raw_adapters picks the
    # decoder's columns), so they take the latest SD 3.5 row even before its arena files carry the tuned columns
    sd35_row = latest_sd35_row(a.fresh_root) or SD35_FALLBACK
    zero_rows = {"unet": rows_used[PRIMARY][0], "pixart": rows_used["pixart200k_ema"][0], "sd35": sd35_row}
    refs = {}
    for backbone, row in zero_rows.items():
        files = per_window_files(os.path.join(a.fresh_root, row))
        tuned = row.endswith("_tuned") or _carries_tuned(a.fresh_root, row)
        refs[backbone] = {dec: rad.zero_shot_refs(files, "" if dec == "stock" else f"_{dec}")
                          for dec in (("stock", "tuned") if tuned else ("stock",))}
    loaded = rad.load_blocks(a.adapt_root, refs, full_roots={"sd35": a.sd35_full_root, "pixart": a.pixart_full_root},
                             fullft_root=a.fullft_g8k_root)

    unet_hours = loaded.get("unet_lora", {}).get("arenas", {})
    unet_data = {m: {"grid": list(GRID), "gpu_hours": unet_hours.get(m, {}).get("gpu_hours", {}),
                     "card": unet_hours.get(m, {}).get("card"), "source": unet_hours.get(m, {}).get("source"),
                     "reads": {st: {"quantity": "raw", "psnr": p, "lpips": lp}
                               for st, p, lp in zip(t["steps"], t["psnr"], t["lpips"])}}
                 for m, t in trajectories.items()}
    full_data, full_borrowed = borrow_zero_shot(loaded.get("unet_full", {}).get("arenas", {}), unet_data)
    block_rows, block_stats, levels, curves, not_drawn, panel_arenas, panel_raw = [], {}, {}, [], {}, {}, {}
    train_level = {}
    for key, name, wanted in BLOCKS:
        backbone = key.split("_")[0]
        data = unet_data if key == "unet_lora" else full_data if key == "unet_full" else \
            loaded.get(key, {}).get("arenas", {})
        wanted_list = list(wanted) if wanted else arenas
        if key == "unet_lora":
            decoder, identity = "tuned", loaded.get("unet_lora", {}).get("identity") or "fine-tuned SD 1"
        else:
            # the decoder of the arenas the block reports, not of those it leaves out as decoded only
            reported, _, _ = split_arenas(data, wanted_list, a.headline_step)
            decs = sorted({data[m].get("decoder") for m in reported} - {None})
            decoder = decs[0] if len(decs) == 1 else "mixed" if decs else loaded.get(key, {}).get("decoder")
            identity = next((data[m]["identity"] for m in reported
                             if data[m].get("decoder") == decoder and data[m].get("identity")),
                            loaded.get(key, {}).get("identity"))
        level_decoder = decoder or ("tuned" if backbone != "sd35" else "stock")
        _, block_level = rad.in_distribution_gap(a.fresh_root, zero_rows[backbone], level_decoder)
        train_level[key] = block_level
        b = block_summary(data, wanted_list, block_level, a.headline_step)
        b.update({"decoder": decoder, "decoder_identity": identity, "level_source": zero_rows[backbone],
                  "level_note": None if block_level else
                  f"no training-map read of {zero_rows[backbone]} through the {level_decoder} decoder: "
                  "no in-distribution level, so no budget and no shares",
                  "zero_shot_borrowed": [m for m in full_borrowed if m in b["arenas"]] if key == "unet_full" else []})
        if key == "unet_full" and b["arenas"]:
            # the caption's comparison: the adapter's own numbers on the same maps (Rohan, Sep 29 00:30: the
            # "U-Net LoRA (6, 7, 8, 16)" row leaves Table 2, the full fine-tune stays as the comparison), over
            # whatever maps the full fine-tune has (the 8k-grid runs landing on main widen the set)
            lora = block_summary(unet_data, b["arenas"], block_level, a.headline_step)
            b["matched_adapter"] = {"psnr_4k": lora["psnr_4k"], "lpips_4k": lora["lpips_4k"],
                                    "psnr_8k": lora["psnr_8k"], "lpips_8k": lora["lpips_8k"],
                                    "shares": lora["shares"], "budget_middle": lora["budget_middle"],
                                    "arenas": lora["arenas"]}
        block_stats[key] = b
        if key != "unet_lora":
            block_rows.append((block_label(name, b, wanted is None), b, identity))
        # the panel: raw reads only, every finished arena the block has
        raw = {m: {st: r for st, r in d["reads"].items() if r["quantity"] == "raw"} for m, d in data.items()
               if finished(d, a.headline_step)}
        raw = {m: r for m, r in raw.items() if 0 in r}
        if not data:
            not_drawn[key] = "no runs"
            continue
        if not raw:
            not_drawn[key] = "no raw reads"
            continue
        curves.append((key, PANEL_LABELS[key], panel_points(raw), len(raw)))
        panel_arenas[key], panel_raw[key] = sorted(raw), raw
        levels[key] = block_level if block_level else (None, None)
    with open(table, "w") as f:
        f.write(groups_table(gstats, budgets_final=grid_complete, blocks=block_rows))
    # the slim results table (a candidate to replace Tables 1 and 2 together): one row per backbone over every arena,
    # then the U-Net's LoRA and full fine-tune on the comparator arenas; the groups stay in Table 2's candidate
    unet_all = block_summary(unet_data, arenas, train_level.get("unet_lora"), a.headline_step)
    slim = os.path.join(a.tables_dir, "results_slim.tex")
    slim_rows = [("SD 1.4 U-Net", unet_all, train_level.get("unet_lora"), True),
                 ("PixArt-$\\alpha$", block_stats["pixart_lora"], train_level.get("pixart_lora"), True),
                 ("SD 3.5 Medium", block_stats["sd35_lora"], train_level.get("sd35_lora"), True),
                 # the comparison rows leave the in-domain columns blank: their checkpoint is the U-Net's, and their
                 # own in-domain score after adaptation is not measured (the shares still use the U-Net's level)
                 ("U-Net LoRA", block_stats["unet_lora"], None, False),
                 ("U-Net full fine-tune", block_stats["unet_full"], None, False)]
    with open(slim, "w") as f:
        f.write(slim_table(slim_rows))
    slim_caption_path = os.path.join(a.tables_dir, "results_slim_caption.tex")
    with open(slim_caption_path, "w") as f:
        f.write(slim_caption(slim_rows, home[PRIMARY]["n"]))
    written += [slim, slim_caption_path]
    # the every-arena panel: one LoRA curve per backbone. The full fine-tune runs on a few arenas by design, and its
    # median beside medians over every arena would compare different arenas; it joins the shared panel instead
    every = [c for c in curves if c[0].endswith("_lora")]
    not_drawn.update({c[0]: "a subset of the unseen maps: drawn on the shared panel once it covers the shared maps"
                      for c in curves if not c[0].endswith("_lora")})
    if every:
        written += fig_backbones(every, levels, a.out_dir, a.headline_step)
    # the like-for-like panel: every curve over the arenas all LoRA backbones have finished (the full fine-tune too,
    # once it has all of them)
    shared = shared_arenas(panel_arenas)
    shared_curves = [(key, label, pts, len(shared)) for key, label, *_ in curves
                     for pts in [panel_points(panel_raw[key], shared)] if shared and pts]
    curves = every
    if shared_curves:
        written += fig_backbones(shared_curves, levels, a.out_dir, a.headline_step, stem="raw_backbones_shared")
    # the full fine-tune panel: the U-Net LoRA's whole grid against the full fine-tune's two reads (0 and 4k) on the
    # comparator arenas, drawn once the full fine-tune has every one the LoRA has
    ft_arenas = fullft_arenas(panel_arenas.get("unet_lora", []), panel_arenas.get("unet_full", []))
    ft_points = {key: panel_points(panel_raw[key], ft_arenas) for key in ("unet_lora", "unet_full") if ft_arenas}
    ft_curves = [(key, label, ft_points[key], n) for key, label, n in
                 (("unet_lora", "U-Net LoRA", len(ft_arenas or [])),
                  ("unet_full", fullft_label(ft_points.get("unet_full") or {}), None))
                 if ft_points.get(key)]
    if len(ft_curves) == 2:
        written += fig_backbones(ft_curves, levels, a.out_dir, a.headline_step, stem="raw_fullft")
    # the body candidates: the shared panel's LoRA curves (one per backbone; the full fine-tune has its own panel) as
    # (e, f) of the merged row, and as a separate keyed figure
    body_curves = [c for c in shared_curves if c[0].endswith("_lora")]
    body_stems = []
    if len(body_curves) >= 2:
        written += fig_row(arenas, zero[PRIMARY], after, level, trajectories, colour_of, named, a.out_dir,
                           layout="row_backbones", key_colours=key_colours, headline=a.headline_step,
                           backbones=(body_curves, levels))
        written += fig_backbones(body_curves, levels, a.out_dir, a.headline_step, stem="raw_backbones_body",
                                 keyed=True)
        body_stems = ["raw_row_backbones", "raw_backbones_body"]
    # Figure 3 per backbone (Rohan, Sep 28 night): (a, b) every backbone's per-arena zero-shot and headline reads
    # (the U-Net's from the fresh-rescore adapter row, the others' from their LoRA runs' raw files, the same reads
    # as the medians) with its own upper bound, (c, d) the body candidates' median curves
    zero_of = {"unet_lora": PRIMARY, "pixart_lora": "pixart200k_ema", "sd35_lora": sd35_name}
    heads = {"unet_lora": after}
    heads.update({k: {m: r[a.headline_step] for m, r in panel_raw[k].items() if a.headline_step in r}
                  for k in ("pixart_lora", "sd35_lora") if k in panel_raw})
    v2_backbones = [(c[0], zero[zero_of[c[0]]], heads[c[0]]) for c in body_curves
                    if c[0] in heads and zero_of.get(c[0]) in zero]
    v2_stems = []
    if len(v2_backbones) >= 2:
        for tall in (False, True):
            written += fig_row_v2(arenas, v2_backbones, body_curves, levels, a.out_dir, a.headline_step, tall=tall)
        v2_stems = ["raw_row_v2", "raw_row_v2_tall"]

    s0 = {}
    if os.path.exists(a.family_step):
        with open(a.family_step) as f:
            fsj = json.load(f)
        s0 = {int(m): e.get("S0") for m, e in fsj["backbones"][PRIMARY]["maps"].items()}
        inputs.append(a.family_step)
    numbers = text_numbers(adaptation, complete, s0, grid_complete)
    guards = guard_reads(sorted(glob.glob(a.adapt_glob)))
    # Tables 1 and 2 merged (Rohan, Sep 29): one full-width table and its caption, for the body
    merged = os.path.join(a.tables_dir, "results_merged.tex")
    with open(merged, "w") as f:
        f.write(merged_table(names, home, zero, directional, gstats, block_stats, guards, grid_complete,
                             blocks=block_rows, with_8k=False, with_budget=False, directional_at=4000))
    merged_caption_path = os.path.join(a.tables_dir, "results_merged_caption.tex")
    with open(merged_caption_path, "w") as f:
        f.write(merged_caption(names, home, directional, gstats, block_stats))
    written += [merged, merged_caption_path]
    per_table = os.path.join(a.tables_dir, "adapt_perarena.tex")
    with open(per_table, "w") as f:
        f.write(perarena_table(arenas, adaptation, zero[PRIMARY], guards, grid_complete))
    written.append(per_table)
    per_arena_rows = {str(m): {"psnr_deficit": adaptation[m]["psnr_deficit"], "lpips_rise": adaptation[m]["lpips_rise"],
                               "threshold": adaptation[m]["threshold"],
                               "budget": adaptation[m]["budget"],
                               "lpips_share_4k": adaptation[m]["lpips_share_4k"],
                               "psnr_share_4k": adaptation[m]["psnr_share_4k"],
                               **guards.get(m, {})} for m in arenas}

    def diff(key):
        pairs = [(m, adapted[4000][m][key] - base4k[m][key]) for m in arenas
                 if m in adapted.get(4000, {}) and m in base4k]
        return {"per_arena": {str(m): v for m, v in pairs}, "median": median([v for _, v in pairs]),
                "max_abs": max((abs(v) for _, v in pairs), default=None)}

    summary = {
        "generated_by": "paper/make_raw_figures.py",
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "definitions": {"psnr": "scene PSNR of D(z_hat) against the raw frame, rows 0 to 207",
                        "lpips": "scene LPIPS of D(z_hat) against the raw frame",
                        "in_distribution": "the same backbone on the training maps, pooled over the validation "
                                           "windows (the level every deficit is measured against; no "
                                           "reconstruction upper bound anywhere since Sep 28 night)",
                        "decoders": {n: ("fine-tuned SD 1" if not n.startswith("sd35") else "fine-tuned SD 3.5"
                                         if rows_used[n][1] else "stock SD 3.5 (fallback)") for n in names},
                        "psnr_deficit": "in-distribution scene PSNR minus zero-shot scene PSNR of the same "
                                        "backbone (a map's zero-shot deficit against the training maps)",
                        "lpips_rise": "zero-shot scene LPIPS minus the in-distribution LPIPS of the same backbone",
                        "lpips_share_4k": "(LPIPS at 0 - LPIPS at 4k) / lpips_rise, per map; the tables print the "
                                          "median over maps",
                        "psnr_share_4k": "(PSNR at 4k - PSNR at 0) / psnr_deficit, per map; median over maps",
                        "budget": "the first grid read at or below (LPIPS at 0 + in-distribution LPIPS) / 2, i.e. "
                                  "recovering at least half of the map's zero-shot LPIPS rise (half_lpips_rise, "
                                  "Rohan Sep 29 00:30); None when censored; a map whose budget is the first "
                                  "point of its grid is listed under budget_at_first_read",
                        "groups": f"terciles of the {key_name}, hardest first, split 4, 5, 4 for 13 maps"},
        "group_key": key_name,
        "headline_step": a.headline_step,
        "sd35_row": {"row": rows_used[sd35_name][0], "decoder": "tuned" if rows_used[sd35_name][1] else "stock",
                     "fallback": sd35_name == SD35_FALLBACK and not rows_used[sd35_name][1]},
        # the upper bound is read (the per-window files carry it) but no longer recorded: nothing draws it
        "zero_shot": {n: {str(m): no_ceiling(e) for m, e in zero[n].items()} for n in names},
        "in_distribution": {n: {"pooled": no_ceiling(home[n]["pooled"]),
                                "maps": {str(m): no_ceiling(e) for m, e in home[n]["maps"].items()},
                                "n": home[n]["n"]} for n in names},
        "directional": {n: {k: v for k, v in d.items() if k != "files"} for n, d in directional.items()},
        "d": {"arenas": {str(m): v for m, v in arena_d.items()}, "training": {str(m): v for m, v in train_d.items()},
              "floor": floor},
        "step": step,
        "before_after": ba,
        "adaptation": {"per_arena": {str(m): r for m, r in adaptation.items()}, **counts},
        "groups": gstats,
        "text_numbers": numbers,
        "blocks": block_stats,
        "slim_unet_all": unet_all,                  # the slim table's U-Net row: the LoRA over every arena
        "backbone_panel": {"drawn": [c[0] for c in curves], "not_drawn": not_drawn, "arenas": panel_arenas,
                           "medians": {c[0]: {str(st): v for st, v in c[2].items()} for c in curves}},
        "fullft_panel": {"arenas": ft_arenas or [], "drawn": [c[0] for c in ft_curves] if len(ft_curves) == 2 else [],
                         "labels": [c[1] if c[3] is None else f"{c[1]} ({c[3]})" for c in ft_curves]
                         if len(ft_curves) == 2 else [],
                         "waiting_for": sorted(set(COMPARATOR_ARENAS) & set(panel_arenas.get("unet_lora", []))
                                               - set(panel_arenas.get("unet_full", []))),
                         "medians": {c[0]: {str(st): v for st, v in c[2].items()} for c in ft_curves}
                         if len(ft_curves) == 2 else {}},
        "backbone_panel_shared": {"arenas": shared, "drawn": [c[0] for c in shared_curves],
                                  "medians": {c[0]: {str(st): v for st, v in c[2].items()} for c in shared_curves}},
        "body_candidates": {"arenas": shared if body_stems else [], "drawn": [c[0] for c in body_curves]
                            if body_stems else [], "stems": body_stems},
        "row_v2": {"stems": v2_stems, "backbones": [b[0] for b in v2_backbones],
                   "headline": {b[0]: {str(m): (r["psnr"], r["lpips"]) for m, r in sorted(b[2].items())}
                                for b in v2_backbones},
                   "levels": {b[0]: levels.get(b[0]) for b in v2_backbones}},
        "per_arena_table": per_arena_rows,
        "named_curves": list(named),
        "g8k_4k_minus_base_4k": {"psnr": diff("psnr"), "lpips": diff("lpips")},
        "trajectories": {str(m): t for m, t in trajectories.items()},
        "inputs": [{"path": rel(p), "sha256": sha256(p)} for p in sorted(set(inputs))],
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.summary)), exist_ok=True)
    with open(a.summary, "w") as f:
        json.dump(maf.jsonable(summary), f, indent=1)
        f.write("\n")
    for p in written + [a.summary]:
        print("wrote", rel(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
