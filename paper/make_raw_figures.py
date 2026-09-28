"""
The persistence-free set (Rohan, 2026-09-27 evening): raw scene PSNR and scene LPIPS as the only quantities, with
the reconstruction ceiling and the in-distribution level as the only references. Nothing here replaces a figure or
table in the paper; the figures carry the `raw_` prefix and sit beside the current ones until Rohan chooses.

    python paper/make_raw_figures.py     # writes paper/figures/raw/ and paper/tables/tuned/{adapt_groups.tex,
                                         # adapt_perarena.tex, results_slim.tex, raw_summary.json}

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
BUDGET_RULES = ("half_excess_gap", "half_ceiling_gap", "in_distribution_gap")
BASE_4K_ROW = "adapt4000_live_tuned"            # the base run's 4k adapter, for the base-against-8k-grid check
TRAINING_MAPS = maf.TRAINING_MAPS
PRIMARY = "unet200k_ema"
COMPARATOR_ARENAS = (6, 7, 8, 16)               # the full fine-tune's arenas (Table 2's last column)
GROUP_NAMES = ("hard", "medium", "easy")
# Table 2's per-backbone blocks at the headline step, in order: (key in raw_adapters.load_blocks, label, arenas; None
# for every arena); the U-Net LoRA block is matched to the full fine-tune's arenas
BLOCKS = (("unet_lora", "U-Net LoRA", COMPARATOR_ARENAS),
          ("unet_full", "U-Net full fine-tune", COMPARATOR_ARENAS),
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
         "raw_row_backbones": (5.5, 3.1), "raw_backbones_body": (5.5, 1.6)}
UPPER = "reconstruction upper bound"             # never "ceiling" in a label
IN_DISTRIBUTION = "training maps (in distribution)"
BAND_LABEL = "train vs train $d$"               # the training maps' own d range (tools/family_step.py)
CEILING_INK = fs.BLACK
CEILING_LW = 1.0
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
            "trainmap_psnr_dec_0": psnr[0], "trainmap_psnr_dec_8k": psnr[8000],
            "forgetting_dec": psnr[8000] - psnr[0] if None not in psnr.values() else None}
    return out


def perarena_table(arenas, adaptation, zero, guards, budgets_final):
    """The appendix's per-arena table: arena, zero-shot gap to the upper bound, the budget's threshold gap
    (G0 + G_train) / 2, the budget (\\tbd until every grid step has a raw read), raw scene PSNR and LPIPS at 0, 4k and
    8k, forgetting on the training maps (decoded reference, stock decoder, marked) and the directional check at 8k."""
    def num(v, d, signed=False):
        if v is None:
            return "--"
        text = f"{v:+.{d}f}" if signed else f"{v:.{d}f}"
        return text.replace("-", "$-$")

    lines = ["% generated by paper/make_raw_figures.py: raw scene PSNR and LPIPS through the fine-tuned decoder,",
             "% U-Net adapter (non-EMA weights, 8k grid); forgetting (dagger) is the training maps' scene PSNR against",
             "% the decoded ground truth through the stock decoder, 0 to 8k, until a raw guard rescore exists",
             "\\begin{tabular}{rrrrrrrrrrr}", "\\toprule",
             "& Gap & Threshold & & \\multicolumn{3}{c}{Scene PSNR (dB)} & \\multicolumn{3}{c}{Scene LPIPS} & "
             "Forgetting$^\\dagger$ \\\\",
             "\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
             "Arena & $G_0$ (dB) & gap (dB) & Budget & 0 & 4k & 8k & 0 & 4k & 8k & (dB) / directional \\\\",
             "\\midrule"]
    for m in arenas:
        r, g = adaptation[m], guards.get(m, {})
        budget = ("\\tbd{}" if not budgets_final else
                  ("${>}$8k" if r["budget"] is None else fs.step_label(r["budget"])))
        guard = (f"{num(g.get('forgetting_dec'), 2, signed=True)} / {num(g.get('directional_8k'), 2)}" if g
                 else "\\tbd{} / \\tbd{}")
        lines.append(f"{m} & {num(r['gap_zero_shot'], 2)} & {num(r['threshold_gap'], 2)} & {budget} & "
                     f"{num(r['zero_shot'], 2)} & {num(r['psnr'].get('4000'), 2)} & {num(r['psnr'].get('8000'), 2)} & "
                     f"{num(zero[m]['lpips'], 3)} & {num(r['lpips'].get('4000'), 3)} & "
                     f"{num(r['lpips'].get('8000'), 3)} & {guard} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "% dagger: decoded reference (the prediction against the decoded ground truth), stock decoder"]
    return "\n".join(lines) + "\n"


def half_ceiling_budget(z0, reads, ceiling):
    """The first read (step, value) reaching halfway from z0 to the upper bound (rule half_ceiling_gap)."""
    return budget_step(reads, budget_threshold("half_ceiling_gap", z0, ceiling, 0.0))


def budget_threshold(rule, z0, upper, g_train):
    """The raw PSNR an adapter read must reach under `rule`: half of the excess gap over the training maps' gap
    (`half_excess_gap`, upper - (G0 + g_train) / 2), half of the zero-shot gap (`half_ceiling_gap`), or the
    training maps' gap itself (`in_distribution_gap`, upper - g_train); G0 = upper - z0."""
    g0 = upper - z0
    return {"half_excess_gap": upper - (g0 + g_train) / 2, "half_ceiling_gap": upper - g0 / 2,
            "in_distribution_gap": upper - g_train}[rule]


def budget_step(reads, threshold):
    """The first read (step, value), in step order, that reaches `threshold`; None if none does."""
    return next((s for s, v in sorted(reads) if v is not None and v >= threshold), None)


def bootstrap(values, statistic, draws=10000, seed=0):
    """The 95% percentile interval of `statistic` over arenas resampled with replacement (`draws` draws)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    stats = [statistic(v[rng.integers(0, len(v), len(v))]) for _ in range(draws)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def text_numbers(adaptation, complete, s0, grid_complete, draws=10000):
    """The numbers the text waits on (the paper owner, Sep 27 evening), over the arenas with an 8k read: the count
    past the budget's threshold by 4k and by 8k, the median budget (censored as +inf) and the median 8k gap to the
    upper bound, each with its arena-bootstrap interval; the mean share of each arena's 0-to-8k raw gain in place
    at every raw read; Spearman of the zero-shot latent skill S0 with the 8k gap, the 8k raw gain and the budget.
    Everything that depends on the budget is provisional until every grid step has a raw read."""
    rows = [adaptation[m] for m in complete]
    budgets = [math.inf if r["budget"] is None else r["budget"] for r in rows]

    def count_by(step):
        hit = [1.0 if b <= step else 0.0 for b in budgets]
        return {"count": int(sum(hit)), "n": len(hit), "interval": bootstrap(hit, np.sum, draws)}

    gaps = [r["upper_bound"] - r["psnr"]["8000"] for r in rows]
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
        "median_gap_8k": {"value": median(gaps), "interval": bootstrap(gaps, np.median, draws)},
        "gain_share_by_step": shares,
        "spearman_S0": {"gap_8k": maf.spearman(x, gaps), "gain_8k": maf.spearman(x, gains),
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
    """Per group and for all arenas: the medians of zero-shot, 8k and ceiling raw PSNR and zero-shot and 8k LPIPS,
    the censored median of the budget and the censored count. `per` maps an arena to {zero_shot, psnr_4k,
    psnr_8k, ceiling (the upper bound), lpips_zero_shot, lpips_4k, lpips_8k, budget}."""
    keys = {"psnr_zero_shot": "zero_shot", "psnr_4k": "psnr_4k", "psnr_8k": "psnr_8k", "ceiling": "ceiling",
            "lpips_zero_shot": "lpips_zero_shot", "lpips_4k": "lpips_4k", "lpips_8k": "lpips_8k"}
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


def budget_label(reads):
    """A budget cell as printed: grid labels, ">8k" in math mode for censored, a pair joined by an en dash."""
    if not reads:
        return "--"
    return "--".join("${>}$8k" if r is None else fs.step_label(int(r)) for r in reads)


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


def block_summary(arenas_data, wanted, g_train, rule, headline=HEADLINE_STEP, check=8000):
    """One per-backbone block of Table 2: over the `wanted` arenas it reports (`split_arenas`), the medians of scene
    PSNR and LPIPS at 0, the headline and the check step (None when the block's grid stops before it), the upper
    bound, the budget (final only when every arena's grid reads are raw), the quantity the reads are in
    (`raw_adapters.step_read`), the arenas left out as decoded only, and the median GPU-hours to the headline per card
    (`gpu_hours_by_card`). `arenas_data` is {arena: {grid, reads: {step: read}, gpu_hours, card}}."""
    present, decoded_only, in_progress = split_arenas(arenas_data, wanted, headline, check)

    def get(a, step, key):
        return (arenas_data[a]["reads"].get(step) or {}).get(key)

    def upper_of(a):
        return next((r["upper"] for r in arenas_data[a]["reads"].values() if r.get("upper") is not None), None)
    has_check = any(check in arenas_data[a]["grid"] for a in present)
    kinds = {r["quantity"] for a in present for st, r in arenas_data[a]["reads"].items() if st in (0, headline, check)}
    quantity = (kinds.pop() if len(kinds) == 1 else "mixed") if kinds else None
    final = bool(present) and quantity == "raw" and g_train is not None
    budgets = []
    for a in present if g_train is not None else ():
        reads, grid = arenas_data[a]["reads"], [st for st in arenas_data[a]["grid"] if st > 0]
        if any(st not in reads or reads[st]["quantity"] != "raw" for st in grid) or upper_of(a) is None:
            final = False
            continue
        threshold = budget_threshold(rule, reads[0]["psnr"], upper_of(a), g_train)
        budgets.append(budget_step([(st, reads[st]["psnr"]) for st in grid], threshold))
    # scored only at 0 and the headline: a budget is "by the headline" or "beyond it", nothing finer
    coarse = bool(present) and all({st for st in arenas_data[a]["grid"] if st > 0} == {headline} for a in present)
    return {"n": len(present), "n_wanted": len(wanted), "wanted": list(wanted), "arenas": present,
            "quantity": quantity, "decoded_only": decoded_only, "in_progress": in_progress, "headline": headline,
            "coarse_grid": coarse,
            "psnr_zero_shot": median([get(a, 0, "psnr") for a in present]),
            "psnr_4k": median([get(a, headline, "psnr") for a in present]),
            "psnr_8k": median([get(a, check, "psnr") for a in present]) if has_check else None,
            "ceiling": median([upper_of(a) for a in present]),
            "lpips_zero_shot": median([get(a, 0, "lpips") for a in present]),
            "lpips_4k": median([get(a, headline, "lpips") for a in present]),
            "lpips_8k": median([get(a, check, "lpips") for a in present]) if has_check else None,
            "has_check": has_check, "g_train": g_train, "budget_final": final,
            "budgets": {str(a): b for a, b in zip(present, budgets)} if final else None,
            "budget_middle": middle_reads(budgets) if final else None,
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
        else budget_label(b["budget_middle"])
    return middle + (f"; {b['censored']} of {b['n']} censored" if b["censored"] else "")


def slim_label(name, b, every):
    """A slim-table row label: the backbone's name alone when its block covers every arena (`every`), else as
    Table 2 labels its blocks ("(8 of 13)", "(6, 7, 8, 16)")."""
    complete = b["n"] in (0, b["n_wanted"])
    return name + ("$^\\ddagger$" if b["quantity"] not in (None, "raw") else "") if every and complete \
        else block_label(name, b, every)


def slim_table(rows, stamp=""):
    """The slim results table (a candidate to replace Tables 1 and 2 together): one row per backbone, then the U-Net's
    LoRA and full fine-tune on the comparator arenas. Columns: the training maps' raw scene PSNR and LPIPS (blank for
    the comparison rows, whose source is the U-Net's), the zero-shot and headline medians of the block's own reads
    (paired: the same arenas), the median reconstruction upper bound over them and the budget cell as Table 2 prints
    it. `rows` is [(name, block_summary, (psnr, lpips) on the training maps or None, covers every arena)]."""
    def num(v, d):
        return "--" if v is None else f"{v:.{d}f}"

    head = rows[0][1].get("headline", HEADLINE_STEP) if rows else HEADLINE_STEP
    lines = [f"% generated by paper/make_raw_figures.py{stamp}: raw scene reads, each backbone through its fine-tuned",
             "% decoder, non-EMA adapter weights; medians over each row's arenas (zero-shot and adapted paired)",
             "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
             # group headers no wider than their two columns, which would otherwise take the excess into the
             # second column; the caption says the last two groups are the unseen arenas
             "& \\multicolumn{2}{c}{Training maps} & \\multicolumn{2}{c}{Zero-shot} & "
             f"\\multicolumn{{2}}{{c}}{{{fs.step_label(head)} updates}} & Upper & \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "Model & PSNR & LPIPS & PSNR & LPIPS & PSNR & LPIPS & bound & Budget \\\\",
             "\\midrule"]
    for i, (name, b, level, every) in enumerate(rows):
        if i and not every and rows[i - 1][3]:
            lines.append("\\midrule")
        train = [num(level[0], 2), num(level[1], 3)] if level else ["", ""]
        if not b["n"]:
            cells = train + ["\\tbd{}"] * 6
        else:
            cells = train + [num(b["psnr_zero_shot"], 2), num(b["lpips_zero_shot"], 3), num(b["psnr_4k"], 2),
                             num(b["lpips_4k"], 3), num(b["ceiling"], 2), block_budget_cell(b)]
        lines.append(" & ".join([slim_label(name, b, every)] + cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def groups_table(stats, stamp="", budgets_final=True, blocks=()):
    """Table 2's candidate: rows hard, medium, easy (their arenas listed) and all arenas, medians of the U-Net's raw
    scene reads through the fine-tuned decoder at zero-shot, 4k and 8k, the upper bound and the budget (\\tbd cells
    until `budgets_final`); then the per-backbone blocks (`blocks`: [(label, block_summary, decoder identity)]), each
    \\tbd where it has no arena yet, "--" at 8k where its grid stops at 4k."""
    def num(v, d):
        return "--" if v is None else f"{v:.{d}f}"

    lines = [f"% generated by paper/make_raw_figures.py{stamp}",
             "% medians per group of the U-Net's raw scene reads (fine-tuned decoder, non-EMA adapter weights);",
             "% groups by the zero-shot gap to the reconstruction upper bound; budget: the first grid read that",
             "% closes half of the excess gap over the training maps' own gap (\\tbd until every grid read exists)",
             "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
             "& \\multicolumn{4}{c}{Scene PSNR (dB) $\\uparrow$} & \\multicolumn{3}{c}{Scene LPIPS $\\downarrow$} & "
             "Budget to \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}",
             "Arenas (by zero-shot gap) & zero-shot & 4k & 8k & upper bound & zero-shot & 4k & 8k & half the "
             "excess gap \\\\",
             "\\midrule"]
    for name in list(GROUP_NAMES) + ["all"]:
        g = stats[name]
        label = f"{name.capitalize()} ({', '.join(map(str, g['arenas']))})" if name != "all" else f"All {g['n']}"
        if name == "all":
            lines.append("\\midrule")
        lines.append(f"{label} & {num(g['psnr_zero_shot'], 2)} & {num(g['psnr_4k'], 2)} & {num(g['psnr_8k'], 2)} & "
                     f"{num(g['ceiling'], 2)} & {num(g['lpips_zero_shot'], 3)} & {num(g['lpips_4k'], 3)} & "
                     f"{num(g['lpips_8k'], 3)} & {_budget_cell(g, budgets_final)} \\\\")
    if blocks:
        lines.append("\\midrule")
    for label, b, _ in blocks:
        if not b["n"]:
            lines.append(f"{label} & " + " & ".join(["\\tbd{}"] * 8) + " \\\\")
            continue
        lines.append(f"{label} & {num(b['psnr_zero_shot'], 2)} & {num(b['psnr_4k'], 2)} & {num(b['psnr_8k'], 2)} & "
                     f"{num(b['ceiling'], 2)} & {num(b['lpips_zero_shot'], 3)} & {num(b['lpips_4k'], 3)} & "
                     f"{num(b['lpips_8k'], 3)} & {block_budget_cell(b)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    for label, b, identity in blocks:
        left = b.get("decoded_only") or []
        lines.append(f"% {label}: arenas {', '.join(map(str, b['arenas'])) or '--'}; "
                     f"decoder {identity or '--'}, reads {b['quantity'] or 'none'}, {b['n']} arenas"
                     + (", dagger: against the decoded ground truth" if b["quantity"] not in (None, "raw") else "")
                     + (f"; arenas {', '.join(map(str, left))} read against the decoded ground truth only, left out "
                        "until rescored raw" if left else "")
                     + (f"; arenas {', '.join(map(str, b['in_progress']))} still training (no "
                        f"{budget_label([b.get('headline', HEADLINE_STEP)])} read yet), left out"
                        if b.get("in_progress") else "")
                     + (f"; zero-shot of arenas {', '.join(map(str, b['zero_shot_borrowed']))} from the U-Net LoRA's "
                        "step 0 (the same checkpoint and windows)" if b.get("zero_shot_borrowed") else "")
                     + ("; budget grid 0 and 4k only: ${\\le}$4k means reached by 4k, the rule cannot resolve finer"
                        if b.get("coarse_grid") else ""))
    return "\n".join(lines) + "\n"


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


def _ceiling_ticks(ax, xs, values, half=0.36):
    for x, v in zip(xs, values):
        if v is not None:
            ax.plot([x - half, x + half], [v, v], color=CEILING_INK, lw=CEILING_LW, zorder=2.6,
                    solid_capstyle="butt")


def _in_distribution(ax, level, label=None, below=False):
    ax.axhline(level, color=fs.TRAINING_LINE, lw=fs.REF_LW, ls=fs.TRAINING_DASH, zorder=1.4, gid="ref")
    if label:
        ax.text(0.01, level, label, transform=ax.get_yaxis_transform(), ha="left", va="top" if below else "bottom",
                fontsize=fs.MIN_PT, color=fs.TRAINING_LINE, gid="decor")


def before_after_panel(ax, arenas, zero, adapted, ceilings, level, key, colour_of):
    """One arena per column in number order: the ceiling as a dotted tick, zero-shot open, after 8k updates filled,
    a thin connector between them, the in-distribution level as the dashed line; marks in the arena's colour."""
    pos = {a: i for i, a in enumerate(arenas)}
    _in_distribution(ax, level)
    _ceiling_ticks(ax, [pos[a] for a in arenas], [ceilings[a][f"ceiling_{key}"] for a in arenas])
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
    ax.set_xlabel("unseen arena")


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
    """The row's two keys: above it, what each mark and line means; below it, the colour of the arena groups by
    the zero-shot gap to the upper bound (`key_colours`: [(group, colour)])."""
    ink = fs.BACKBONES["unet"].colour
    marks = [Line2D([], [], ls="none", marker="o", ms=fs.MARKER_SIZE, mfc="white", mec=ink, mew=0.8),
             Line2D([], [], ls="none", marker="o", ms=fs.MARKER_SIZE, mfc=ink, mec=ink, mew=0.6),
             Line2D([], [], color=CEILING_INK, lw=CEILING_LW),
             Line2D([], [], color=fs.TRAINING_LINE, lw=fs.REF_LW, ls=fs.TRAINING_DASH)]
    fig.legend(marks, ["zero-shot", f"after {fs.step_label(headline)} updates", UPPER, IN_DISTRIBUTION],
               loc="outside upper center",
               ncol=4, handlelength=1.4, columnspacing=1.4, handletextpad=0.4, borderaxespad=0.1)
    swatches = [_text_entry()] + [Patch(facecolor=c, edgecolor="none") for _, c in key_colours]
    fig.legend(swatches, ["zero-shot gap to the upper bound:"] + [g for g, _ in key_colours],
               loc="outside lower center", ncol=len(swatches), handlelength=1.0, handleheight=0.8,
               columnspacing=1.0, handletextpad=0.35, borderaxespad=0.1)


def backbone_row_key(fig, curves):
    """The key under the backbone panels (e, f): each curve's backbone and the arenas its median covers."""
    handles, names = backbone_handles(curves)
    n = curves[0][3]
    fig.legend([_text_entry()] + handles, [f"(e, f) median over the {n} arenas all three have:" if n else
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


def fig_row(arenas, zero, adapted, ceilings, level, trajectories, colour_of, named, out_dir, layout="row",
            key_colours=(), headline=HEADLINE_STEP, backbones=None):
    """The merged Figure 3: how much the unseen arenas improve and how fast. (a) before and after `headline`
    updates in raw PSNR (`adapted` holds that step's reads), (b) in LPIPS, (c) and (d) each arena's raw trajectory
    over the adapter reads that exist, sharing its y axis with (a) and (b) so the filled mark is the curve's read at
    the headline step. `layout` "grid" draws two rows of two (PSNR above, LPIPS below); "row_backbones" (a body
    candidate) adds (e) and (f) on a line of their own under the unchanged row (six panels in one 5.5 in row leave
    the 13 arena numbers overlapping): the median raw PSNR and LPIPS of each backbone's adapter over the arenas they
    share, on their own y axes, from `backbones` = (curves, levels) as `fig_backbones` takes them, keyed below."""
    stem = {"row": "raw_row", "grid": "raw_row_grid", "row_backbones": "raw_row_backbones"}[layout]
    extra, subs = (), None
    if layout == "row":
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
        before_after_panel(ax, arenas, zero, adapted, ceilings, level[key], key, colour_of)
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
    (`raw` is {arena: {step: read}}); None when the block lacks one of `arenas`."""
    if arenas is not None and any(m not in raw for m in arenas):
        return None
    pick = {m: raw[m] for m in (raw if arenas is None else arenas)}
    steps = sorted({st for r in pick.values() for st in r})
    return {st: (median([r[st]["psnr"] for r in pick.values() if st in r]),
                 median([r[st]["lpips"] for r in pick.values() if st in r])) for st in steps}


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


def fig_arenas_raw(arenas, trajectories, adaptation, level, colour_of, out_dir, headline=HEADLINE_STEP):
    """Appendix: one small panel per arena, raw scene PSNR against adapter updates (log axis, the 0 read at the
    left) in the arena's group colour, the headline read filled, the budget's threshold (dotted), the arena's
    reconstruction upper bound (black) and the training maps' level (dashed); shared axes in arena order, the
    key in the first empty slot (or under the panels when none is empty)."""
    stem = "raw_figA_adapt_arenas"
    recs = [m for m in arenas if m in trajectories]
    ncols = min(5, len(recs))
    nrows = math.ceil(len(recs) / ncols)
    fig, axes = fs.new_figure(SIZES[stem], ncols=ncols, nrows=nrows, sharex=True, sharey=True, wspace=0.02,
                              hspace=0.03)
    steps = sorted({s for m in recs for s in trajectories[m]["steps"]})
    values = [v for m in recs for v in trajectories[m]["psnr"]] + [adaptation[m]["upper_bound"] for m in recs]
    ink = fs.BACKBONES["unet"].colour
    key = ([Line2D([], [], color=ink, lw=fs.DATA_LW, marker="o", ms=2.5),
            Line2D([], [], ls="none", marker="o", ms=3.5, mfc=ink, mec=ink),
            Line2D([], [], color=fs.CONTEXT_INK, lw=fs.MIN_LW, ls=(0, (1, 1.2))),
            Line2D([], [], color=CEILING_INK, lw=fs.REF_LW),
            Line2D([], [], color=fs.TRAINING_LINE, lw=fs.REF_LW, ls=fs.TRAINING_DASH)],
           ["scene PSNR", f"after {fs.step_label(headline)} updates", "half the excess gap", UPPER, IN_DISTRIBUTION])
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
        ax.plot(xs, t["psnr"], color=colour_of(m), lw=fs.DATA_LW, marker="o", ms=2.5, zorder=3)
        if headline in t["steps"]:
            j = t["steps"].index(headline)
            ax.plot([xs[j]], [t["psnr"][j]], ls="none", marker="o", ms=3.5, mfc=colour_of(m), mec=colour_of(m),
                    zorder=3.5)
        ax.axhline(r["threshold"], color=fs.CONTEXT_INK, lw=fs.MIN_LW, ls=(0, (1, 1.2)), zorder=1.4, gid="ref")
        ax.axhline(r["upper_bound"], color=CEILING_INK, lw=fs.REF_LW, zorder=1.4, gid="ref")
        ax.axhline(level, color=fs.TRAINING_LINE, lw=fs.REF_LW, ls=fs.TRAINING_DASH, zorder=1.4, gid="ref")
        # the arena's name at the bottom edge, under every threshold line (the lowest sits 0.9 dB above the floor)
        ax.text(0.97, 0.02, f"arena {m}", transform=ax.transAxes, ha="right", va="bottom", fontsize=fs.ANNOT_PT,
                gid="decor")
        ax.set_ylim(min(values) - 0.8, max(values) + 0.5)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        if i % ncols == 0:
            ax.set_ylabel("scene PSNR (dB)")
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
    p.add_argument("--distances", default=os.path.join(REPO, "results", "distance_v2", "distances_sd1.json"))
    p.add_argument("--training-distances",
                   default=os.path.join(REPO, "results", "distance_study", "figure_unet_h1", "stats.json"))
    p.add_argument("--adapt-root", default=os.path.join(REPO, "results", "adapt"),
                   help="every backbone's adaptation runs: Table 2's per-backbone blocks and the backbone panel")
    p.add_argument("--adapt-glob",
                   default=os.path.join(REPO, "results", "adapt", "unet200k_arenas13_map??_r16_k8_s0_g8k"),
                   help="the 8k-grid run directories whose guard rows give forgetting and the directional check")
    p.add_argument("--family-step", default=os.path.join(REPO, "results", "family_step", "family_step.json"),
                   help="the zero-shot latent skill S0 per arena (tools/family_step.py), for the Spearman numbers")
    p.add_argument("--budget-rule", choices=BUDGET_RULES, default="half_excess_gap")
    p.add_argument("--headline-step", type=int, default=HEADLINE_STEP,
                   help="the adapter read the filled marks and the medians report (8k stays the check)")
    p.add_argument("--group-by", choices=("gap", "zero_shot_psnr"), default="gap",
                   help="the key that groups and colours the arenas: the zero-shot gap to the upper bound (default)")
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
    ceilings = {m: {k: zero[PRIMARY][m][k] for k in ("ceiling_psnr", "ceiling_lpips")} for m in arenas}

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
    g_train = home[PRIMARY]["pooled"]["ceiling_psnr"] - home[PRIMARY]["pooled"]["psnr"]
    grid_complete = all(s in adapted and all(m in adapted[s] for m in arenas) for s in GRID if s > 0)
    adaptation, trajectories = {}, {}
    for m in arenas:
        upper = ceilings[m]["ceiling_psnr"]
        reads = [(s, adapted[s].get(m, {}).get("psnr")) for s in sorted(adapted)]
        thresholds = {r: budget_threshold(r, z0[m], upper, g_train) for r in BUDGET_RULES}
        adaptation[m] = {"zero_shot": z0[m], "upper_bound": upper, "gap_zero_shot": upper - z0[m],
                         "threshold": thresholds[a.budget_rule],
                         "threshold_gap": upper - thresholds[a.budget_rule],
                         "psnr": {str(s): v for s, v in reads},
                         "lpips": {str(s): adapted[s].get(m, {}).get("lpips") for s in sorted(adapted)},
                         "budget": budget_step(reads, thresholds[a.budget_rule]),
                         "budget_by_rule": {r: budget_step(reads, t) for r, t in thresholds.items()}}
        steps = [s for s, v in reads if v is not None]
        if 8000 in steps:
            trajectories[m] = {"steps": [0] + steps,
                               "psnr": [z0[m]] + [adapted[s][m]["psnr"] for s in steps],
                               "lpips": [zero[PRIMARY][m]["lpips"]] + [adapted[s][m]["lpips"] for s in steps]}
    complete = sorted(trajectories)

    def counts_for(rule):
        budgets = [{"budget": adaptation[m]["budget_by_rule"][rule]} for m in complete]
        return {"past_half_by_4k": sum(1 for b in budgets if b["budget"] is not None and b["budget"] <= 4000),
                "past_half_by_8k": sum(1 for b in budgets if b["budget"] is not None),
                "median_budget": maf.censored_median_entry(budgets, "budget", 8000),
                "censored": [m for m, b in zip(complete, budgets) if b["budget"] is None]}
    counts = {"n": len(complete), "rule": a.budget_rule, "in_distribution_gap": g_train,
              "grid_complete": grid_complete, "reads": sorted(adapted), **counts_for(a.budget_rule),
              "by_rule": {r: counts_for(r) for r in BUDGET_RULES}}
    ad8 = adapted[8000]
    if a.headline_step not in adapted:
        raise SystemExit(f"no raw read of the adapter at the headline step {a.headline_step}")
    after = adapted[a.headline_step]
    level = {k: home[PRIMARY]["pooled"][k] for k in ("psnr", "lpips")}
    ba = {}
    for key in ("psnr", "lpips"):
        better = (lambda v, lv: v >= lv) if key == "psnr" else (lambda v, lv: v <= lv)
        z = {m: zero[PRIMARY][m][key] for m in arenas}
        ba[key] = {"in_distribution": level[key], "headline_step": a.headline_step,
                   "median_zero_shot": median(list(z.values())),
                   "median_adapted": median([after[m][key] for m in arenas if m in after]),
                   "median_upper_bound": median([ceilings[m][f"ceiling_{key}"] for m in arenas]),
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

    # the grouping and colouring key: the zero-shot gap to the upper bound (larger = harder), or zero-shot PSNR
    if a.group_by == "gap":
        key_of = {m: adaptation[m]["gap_zero_shot"] for m in arenas}
        harder, key_name = True, "zero-shot gap to the reconstruction upper bound (dB)"
    else:
        key_of, harder, key_name = dict(z0), False, "zero-shot raw scene PSNR (dB)"
    ease = {m: (-v if harder else v) for m, v in key_of.items()}          # low = hard
    named = (min(complete, key=lambda m: ease[m]), max(complete, key=lambda m: ease[m])) if complete else ()
    groups = group_arenas({m: key_of[m] for m in complete}, higher_is_harder=harder)
    # one shade per group, so the key's three swatches are exactly the marks' colours (the paper owner's check:
    # a continuous ramp keyed by three swatches made two hard arenas read as medium)
    shade_of, key_colours = group_colours(groups)

    def colour_of(m):
        return shade_of.get(m, fs.BACKBONES["adapter"].colour)
    for layout in ("row", "grid"):
        written += fig_row(arenas, zero[PRIMARY], after, ceilings, level, trajectories, colour_of, named, a.out_dir,
                           layout=layout, key_colours=key_colours, headline=a.headline_step)
    written += fig_arenas_raw(arenas, trajectories, adaptation, level["psnr"], colour_of, a.out_dir, a.headline_step)

    # Table 2's candidate: the arenas grouped by the key
    at = {s: adapted.get(s, {}) for s in (4000, 8000)}
    per = {m: {"zero_shot": z0[m], "psnr_4k": at[4000].get(m, {}).get("psnr"), "psnr_8k": ad8[m]["psnr"],
               "ceiling": ceilings[m]["ceiling_psnr"], "lpips_zero_shot": zero[PRIMARY][m]["lpips"],
               "lpips_4k": at[4000].get(m, {}).get("lpips"), "lpips_8k": ad8[m]["lpips"],
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
    loaded = rad.load_blocks(a.adapt_root, refs)

    unet_hours = loaded.get("unet_lora", {}).get("arenas", {})
    unet_data = {m: {"grid": list(GRID), "gpu_hours": unet_hours.get(m, {}).get("gpu_hours", {}),
                     "card": unet_hours.get(m, {}).get("card"), "source": unet_hours.get(m, {}).get("source"),
                     "reads": {st: {"quantity": "raw", "psnr": p, "lpips": lp, "upper": adaptation[m]["upper_bound"]}
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
        gap_decoder = decoder or ("tuned" if backbone != "sd35" else "stock")
        g, block_level = rad.in_distribution_gap(a.fresh_root, zero_rows[backbone], gap_decoder)
        train_level[key] = block_level
        b = block_summary(data, wanted_list, g, a.budget_rule, a.headline_step)
        b.update({"decoder": decoder, "decoder_identity": identity, "g_train_source": zero_rows[backbone],
                  "g_train_note": None if g is not None else
                  f"no training-map read of {zero_rows[backbone]} through the {gap_decoder} decoder: "
                  "no in-distribution gap, so no budget",
                  "zero_shot_borrowed": [m for m in full_borrowed if m in b["arenas"]] if key == "unet_full" else []})
        block_stats[key] = b
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
    unet_all = block_summary(unet_data, arenas, block_stats["unet_lora"]["g_train"], a.budget_rule, a.headline_step)
    slim = os.path.join(a.tables_dir, "results_slim.tex")
    with open(slim, "w") as f:
        f.write(slim_table([("SD 1.4 U-Net", unet_all, train_level.get("unet_lora"), True),
                            ("PixArt-$\\alpha$", block_stats["pixart_lora"], train_level.get("pixart_lora"), True),
                            ("SD 3.5 Medium", block_stats["sd35_lora"], train_level.get("sd35_lora"), True),
                            ("U-Net LoRA", block_stats["unet_lora"], None, False),
                            ("U-Net full fine-tune", block_stats["unet_full"], None, False)]))
    written.append(slim)
    # the every-arena panel: one LoRA curve per backbone. The full fine-tune runs on a few arenas by design, and its
    # median beside medians over every arena would compare different arenas; it joins the shared panel instead
    every = [c for c in curves if c[0].endswith("_lora")]
    not_drawn.update({c[0]: "a subset of arenas: drawn on the shared panel once it covers the shared arenas"
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
        written += fig_row(arenas, zero[PRIMARY], after, ceilings, level, trajectories, colour_of, named, a.out_dir,
                           layout="row_backbones", key_colours=key_colours, headline=a.headline_step,
                           backbones=(body_curves, levels))
        written += fig_backbones(body_curves, levels, a.out_dir, a.headline_step, stem="raw_backbones_body",
                                 keyed=True)
        body_stems = ["raw_row_backbones", "raw_backbones_body"]

    s0 = {}
    if os.path.exists(a.family_step):
        with open(a.family_step) as f:
            fsj = json.load(f)
        s0 = {int(m): e.get("S0") for m, e in fsj["backbones"][PRIMARY]["maps"].items()}
        inputs.append(a.family_step)
    numbers = text_numbers(adaptation, complete, s0, grid_complete)
    guards = guard_reads(sorted(glob.glob(a.adapt_glob)))
    per_table = os.path.join(a.tables_dir, "adapt_perarena.tex")
    with open(per_table, "w") as f:
        f.write(perarena_table(arenas, adaptation, zero[PRIMARY], guards, grid_complete))
    written.append(per_table)
    per_arena_rows = {str(m): {"gap_zero_shot": adaptation[m]["gap_zero_shot"],
                               "threshold_gap": adaptation[m]["threshold_gap"], "budget": adaptation[m]["budget"],
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
                        "upper_bound": "the same for D(z), the decoder on the ground-truth latent",
                        "decoders": {n: ("fine-tuned SD 1" if not n.startswith("sd35") else "fine-tuned SD 3.5"
                                         if rows_used[n][1] else "stock SD 3.5 (fallback)") for n in names},
                        "budget": f"first adapter read reaching the {a.budget_rule} threshold (budget_threshold)",
                        "groups": f"arenas sorted by the {key_name}, hardest first, split 4, 5, 4"},
        "group_key": key_name,
        "headline_step": a.headline_step,
        "sd35_row": {"row": rows_used[sd35_name][0], "decoder": "tuned" if rows_used[sd35_name][1] else "stock",
                     "fallback": sd35_name == SD35_FALLBACK and not rows_used[sd35_name][1]},
        "zero_shot": {n: {str(m): e for m, e in zero[n].items()} for n in names},
        "in_distribution": {n: {"pooled": home[n]["pooled"], "maps": {str(m): e for m, e in home[n]["maps"].items()},
                                "n": home[n]["n"]} for n in names},
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
