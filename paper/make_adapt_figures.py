"""
The adaptation study's figures, tables and summary, from `score_adapt.py` rows and `eval_tf.py` reads.

    python paper/make_adapt_figures.py                                   # stock decoder, scene home 4.138 dB
    python paper/make_adapt_figures.py --decoder tuned --home-json <training maps' eval_tf metrics.json>
    python paper/make_adapt_figures.py --fresh-root results/fresh_rescore --with-raw

**Inputs.** Every `scores.jsonl` that `--runs-glob` matches (default `results/adapt/*/scores.jsonl`), one row
per scored checkpoint, in a run directory named `<source>_<set>_map<NN>_r<rank>_k<k>_s<seed>[_<variant>]`.
The rank-16, eight-episode, seed-0 runs without a variant are the headline set, one per arena; other seeds are
the seed spread, other `k` the episode ladder, and a variant suffix (`lr5e4`, `lr3e4`, `g8k`) a recipe test.
Only `--weights` rows are read (live by default); where a run was scored under more than one evaluation
configuration, the configuration with the most steps is read (the latest on a tie) and the choice is noted.
The frozen per-arena distance D is each arena's primary entry in `--distances` (`distances_sd1.json`).
Zero-shot reads of other backbones are directories of `eval_tf.py` reads keyed by map,
`<fresh-root>/<row>/<name with map<NN>>/metrics.json`; a `_h<K>` suffix other than `_h1` is skipped.

**Quantities**, `score_adapt.py`'s outcomes on the scene crop under `--decoder`: A = `heldout_A_<decoder>`
(decoded advantage over copy-last, dB); M = `heldout_B_<decoder>` (perceptual margin against raw persistence,
present when the raw frames were scored); G = `heldout_C_<decoder>`; S = `heldout_latent_skill` (no decoder);
LPIPS = `heldout_lpips_dec` (full frame against the decoded truth; `lpips_dec_<decoder>` from the per-window
file for another decoder); forgetting = the change of `trainmap_A_<decoder>` from step 0; directional =
`directional_correct_frac`. Intervals are 95% percentile bootstraps over held-out episodes from the per-window
files the rows name (found beside the run when the recorded path is a server path), with the windows
`eval_tf.py` flags as duplicates left out as `score_adapt.py` leaves them out; each file's mean is checked
against its row and a disagreement is noted.

**The cost rule** (`.claude/analyses/cost-target-decision-2026-09-26.md`, `lora-results-decision-2026-09-27.md`).
Per arena the half-gap line is (A0 + home) / 2, where home is the training maps' A on the same crop and decoder:
`--home` (default 4.138 dB, the stock decoder's scene home), or `--home-json`, an eval_tf read of the training
maps, as `scene_psnr_dec - scene_copy_psnr_dec` (duplicate-free means where the read has them). The cost is the
first grid step at or before `--budget` whose A reaches the line (`score_adapt.adaptation_cost`); a curve scored
to the budget that never reaches it is right-censored, and one not yet scored to the budget is incomplete and
left out of every count. The second line is home itself. A median over arenas ranks a censored arena above
every crossing and is censored when the middle lands on one, so it is never a median over crossers alone.
Beside the crossing: A at the grid steps and the area under the curve, the trapezoid mean of A over
[0, budget] in dB. Spearman correlations are scipy's (average ranks for ties) with a censored cost ranked as
twice the budget.

**Outputs.** In `--out-dir`, PDF and PNG, each drawn at the size of its slot in `main.tex` so its fonts print
at their nominal size, with TrueType fonts embedded: `fig3_adaptation_curves` (A against updates, one line per
arena coloured by D, home dashed, each arena's half-gap line faint, crossings filled, censored arenas open at the
right edge), `fig3_adaptation_skill` (S instead of A), `fig3_adaptation_terciles` (arenas averaged in D
terciles, band the range), `fig3_adaptation_seeds` (when other seeds exist), `fig_adapt_ladder` (A at the
budget against episodes, when ladder runs exist), `fig_adapt_recipe` (when recipe runs exist),
`fig2a_advantage_by_distance` (A0 per arena sorted by D, one marker per backbone, each backbone's home) and,
with `--with-raw` and a margin to draw, `fig2b_margin_by_distance`. In `--tables-dir`, under a provenance
header: `adapt_cost.tex` (the per-arena cost table), `adapt_perarena.tex` (the body of the appendix's
`tab:perarena-adapt`), both booktabs tabulars for an existing table float, with `\\tbd` where the rows do not
carry a quantity yet, and `adapt_summary.json`, the numbers the text quotes.
"""
import argparse
import csv
import datetime
import functools
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
sys.path.insert(0, REPO)
from score_adapt import DUPLICATE_FLAGS, OUTCOMES, STOCK, adaptation_cost  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, ticker, transforms  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

STOCK_HOME = 4.138        # training maps' scene A, stock decoder, 512 validation windows (lora-results decision)
TRAINING_MAPS = (2, 3, 4, 5)
BASE_RANK, BASE_K, BASE_SEED = 16, 8, 0
DEFAULT_BUDGET = 4000
RUN_RE = re.compile(r"_map(?P<map>\d+)_r(?P<rank>\d+)_k(?P<k>\d+)_s(?P<seed>\d+)(?:_(?P<variant>\w+))?$")
MAP_DIR_RE = re.compile(r"(?:^|[_-])map0*(?P<map>\d+)(?:_h(?P<h>\d+))?$|^0*(?P<bare>\d+)$")
DECODER_A_RE = re.compile(r"^heldout_A_([A-Za-z][A-Za-z0-9]*)$")
ROW_NAMES = {"unet": "U-Net", "pixart": "PixArt", "sd35": "SD 3.5",
             "unet200k_ema": "U-Net", "pixart200k_ema": "PixArt", "adapt4000_live": "U-Net + LoRA (4k)",
             "sd35_ema": "SD 3.5 (provisional)"}
ROW_ORDER = ("unet", "unet200k_ema", "pixart", "pixart200k_ema", "sd35", "sd35_ema", "adapt4000_live")
RECIPE_ORDER = ("lr3e4", "lr5e4", "g8k")
RECIPE_LABELS = {"lr3e4": "lr 3e-4", "lr5e4": "lr 5e-4", "g8k": "8k grid"}
RECIPE_TICKS = (50, 250, 1000, 4000, 8000)      # labelled steps on the recipe panels' narrow axes

# Sizes of the slots in main.tex (text width 5.5 in): Figure 3 is \linewidth of a 0.33\linewidth minipage by
# 1.15 in, each Figure 2 panel 0.49 of a 0.64\linewidth minipage by 1.15 in, an appendix figure 0.6\linewidth
# by 1.8 in. Drawn at these sizes, \figslot includes them at scale 1.
FIG3_SIZE = (1.8, 1.15)
FIG2_SIZE = (1.7, 1.15)
WIDE_SIZE = (3.3, 1.8)
SMALL_SIZE = (1.8, 1.35)

# Reference palette (dataviz skill): the sequential blue ramp, steps 250 to 700 (the lightest clears 2:1 on white),
# categorical slots 1 to 4 in their validated order, and the neutral inks.
BLUE_RAMP = ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b")
TERCILE_COLOURS = ("#5598e7", "#256abf", "#0d366b")
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MARKERS = ("o", "s", "^", "D")
INK, SECONDARY, MUTED, SHADE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


# ---------------------------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------------------------

def rel(path):
    """`path` relative to the repository when it lies inside it."""
    p = os.path.abspath(path)
    return os.path.relpath(p, REPO) if p.startswith(REPO + os.sep) else p


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def as_float(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


class Run:
    """One adaptation run: its rows by step (one weights, one evaluation configuration) and its name's fields."""

    def __init__(self, name, path, by_step, arena, k, seed, rank, variant):
        self.name, self.path, self.by_step = name, path, by_step
        self.arena, self.k, self.seed, self.rank, self.variant = arena, k, seed, rank, variant
        self.run_dir = os.path.dirname(os.path.abspath(path))
        self.steps = sorted(by_step)

    def rows(self):
        return [self.by_step[s] for s in self.steps]

    def value(self, step, column):
        r = self.by_step.get(step)
        return as_float(r.get(column)) if r else None

    def series(self, column, budget=None):
        """[(step, value)] of the finite values of `column`, up to `budget` when one is given."""
        return [(s, v) for s in self.steps if (budget is None or s <= budget)
                for v in [self.value(s, column)] if v is not None]

    def decoders(self):
        """The decoders whose A the step-0 row carries, stock first."""
        r = self.by_step.get(0) or self.by_step[self.steps[0]]
        names = sorted({m.group(1) for k, v in r.items() for m in [DECODER_A_RE.match(k)] if m and finite(v)})
        return sorted(names, key=lambda d: (d != STOCK, d))


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_run_name(name):
    """{map, rank, k, seed, variant} from a run directory's name, or None."""
    m = RUN_RE.search(name)
    if not m:
        return None
    return {"map": int(m["map"]), "rank": int(m["rank"]), "k": int(m["k"]), "seed": int(m["seed"]),
            "variant": m["variant"] or ""}


def pick_evaluation(rows, column):
    """The rows of one evaluation configuration: among those that carry `column`, the fingerprint with the most
    steps, the latest scored on a tie."""
    groups = {}
    for r in rows:
        groups.setdefault(str(r.get("eval_fingerprint")), []).append(r)
    carrying = [rs for rs in groups.values() if any(finite(r.get(column)) for r in rs)] or list(groups.values())
    best = max(carrying, key=lambda rs: (len({r["step"] for r in rs}), max(str(r.get("scored_at", "")) for r in rs)))
    return best, len(groups)


def load_runs(paths, weights, decoder, notes):
    """Every run the paths name, as `Run`s; runs with nothing to read are noted and skipped."""
    runs = []
    for p in sorted(paths):
        name = os.path.basename(os.path.dirname(os.path.abspath(p)))
        meta = parse_run_name(name)
        if meta is None:
            notes.append(f"{rel(p)}: the directory name does not parse as a run (_map<NN>_r<R>_k<K>_s<S>); skipped")
            continue
        rows = [r for r in read_jsonl(p) if r.get("weights") == weights]
        if not rows:
            notes.append(f"{rel(p)}: no {weights} rows; skipped")
            continue
        rows, n_eval = pick_evaluation(rows, f"heldout_A_{decoder}")
        if n_eval > 1:
            notes.append(f"{name}: {n_eval} evaluation configurations; read {rows[0].get('eval_fingerprint')} "
                         f"({len(rows)} rows)")
        by_step = {}
        for r in sorted(rows, key=lambda r: str(r.get("scored_at", ""))):
            by_step[int(r["step"])] = r
        if len(by_step) < len(rows):
            notes.append(f"{name}: {len(rows) - len(by_step)} repeated steps in one configuration; the latest kept")
        arena = int(rows[0].get("map_id", meta["map"]))
        if arena != meta["map"]:
            notes.append(f"{name}: map_id {arena} differs from the name's map {meta['map']}; map_id used")
        runs.append(Run(name, p, by_step, arena, int(rows[0].get("adapt_episodes_k") or meta["k"]),
                        int(rows[0].get("seed", meta["seed"])), meta["rank"], meta["variant"]))
    return runs


def classify(runs, notes):
    """(headline runs by arena, other seeds, the episode ladder, recipe tests)."""
    base, seeds, ladder, recipe = {}, [], [], []
    for r in runs:
        if r.variant or r.rank != BASE_RANK:
            if r.rank != BASE_RANK:
                r.variant = f"r{r.rank}" + (f"_{r.variant}" if r.variant else "")
            recipe.append(r)
        elif r.k != BASE_K and r.seed == BASE_SEED:
            ladder.append(r)
        elif r.k != BASE_K:
            notes.append(f"{r.name}: a ladder run at seed {r.seed}; not drawn")
        elif r.seed != BASE_SEED:
            seeds.append(r)
        elif r.arena in base:
            raise SystemExit(f"two headline runs for arena {r.arena}: {base[r.arena].name} and {r.name}")
        else:
            base[r.arena] = r
    return base, seeds, ladder, recipe


def load_distances(path):
    """{arena: D} from the primary entries of a `distances_<space>.json`."""
    with open(path) as f:
        js = json.load(f)
    return {int(m["map"]): float(m["D"]) for m in js.get("maps", [])
            if m.get("role") == "primary" and m.get("cluster", "arena") == "arena" and finite(m.get("D"))}


def _mean(v):
    if isinstance(v, dict):
        v = v.get("mean")
    return float(v) if finite(v) else None


def metrics_outcome(metrics, outcome, decoder):
    """One outcome of `score_adapt.OUTCOMES` from an eval_tf summary: model mean minus reference mean.

    The duplicate-free means (`_nodup`) are used when both columns have them, which is `score_adapt.py`'s
    exclusion; otherwise the plain means.
    """
    model, ref, ref_free = OUTCOMES[outcome]
    sfx = "" if decoder == STOCK else f"_{decoder}"
    a, b = model + sfx, ref if ref_free else ref + sfx
    for suffix in ("_nodup", ""):
        x, y = _mean(metrics.get(a + suffix)), _mean(metrics.get(b + suffix))
        if x is not None and y is not None:
            return x - y
    return None


def home_from_json(path, decoder):
    """The training maps' A from an eval_tf `metrics.json`: `scene_psnr_dec - scene_copy_psnr_dec` (per decoder)."""
    with open(path) as f:
        v = metrics_outcome(json.load(f), "A", decoder)
    if v is None:
        raise SystemExit(f"{path} has no scene_psnr_dec / scene_copy_psnr_dec for decoder {decoder}")
    return v


@functools.lru_cache(maxsize=None)
def read_windows(path):
    """The rows of an eval_tf `per_window.csv` that `score_adapt.py` averages (duplicate windows left out)."""
    with open(path) as f:
        reader = csv.DictReader(f)
        header, rows = list(reader.fieldnames or []), list(reader)
    flags = [c for c in DUPLICATE_FLAGS if c in header]
    return tuple(r for r in rows if not any(as_float(r.get(c)) == 1 for c in flags))


def window_outcome(windows, outcome, decoder):
    """(episodes, values) of one outcome per window, finite values only."""
    model, ref, ref_free = OUTCOMES[outcome]
    sfx = "" if decoder == STOCK else f"_{decoder}"
    return window_difference(windows, model + sfx, ref if ref_free else ref + sfx)


def window_difference(windows, a, b=None):
    eps, vals = [], []
    for r in windows:
        x, y = as_float(r.get(a)), (as_float(r.get(b)) if b else 0.0)
        if x is not None and y is not None:
            eps.append(r["episode"])
            vals.append(x - y)
    return eps, vals


def episode_bootstrap(episodes, values, draws, seed):
    """95% percentile interval of the pooled window mean, resampling episodes with replacement; None below 2."""
    if draws <= 0 or not values:
        return None
    uniq, inv = np.unique(np.asarray(episodes), return_inverse=True)
    if len(uniq) < 2:
        return None
    sums = np.bincount(inv, weights=np.asarray(values, float))
    counts = np.bincount(inv).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(uniq), size=(draws, len(uniq)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def local_per_window(run, row):
    """The row's per-window file on this machine: its recorded path, or the same path under the run directory."""
    p = row.get("heldout_per_window")
    if not p:
        return None
    if os.path.exists(p):
        return p
    marker = f"/{run.name}/"
    i = p.find(marker)
    if i < 0:
        return None
    local = os.path.join(run.run_dir, p[i + len(marker):])
    return local if os.path.exists(local) else None


def load_zero_shot_row(row_dir, decoder, draws=0, seed=0):
    """{arena: {A, M, n, A_ci, M_ci, path}} from a directory of eval_tf reads keyed by map (one-tic reads only)."""
    out = {}
    for d in sorted(os.listdir(row_dir)):
        full = os.path.join(row_dir, d)
        m = MAP_DIR_RE.search(d)
        if not os.path.isdir(full) or not m or (m["h"] is not None and int(m["h"]) != 1):
            continue
        found = [os.path.join(full, "metrics.json")] if os.path.exists(os.path.join(full, "metrics.json")) else \
            sorted(glob.glob(os.path.join(full, "*", "metrics.json")))
        if len(found) != 1:
            continue
        with open(found[0]) as f:
            metrics = json.load(f)
        n = (metrics.get("scene_psnr_dec_nodup") or metrics.get("scene_psnr_dec") or {})
        rec = {"A": metrics_outcome(metrics, "A", decoder), "M": metrics_outcome(metrics, "B", decoder),
               "n": n.get("n") if isinstance(n, dict) else None, "A_ci": None, "M_ci": None, "path": found[0]}
        pw = os.path.join(os.path.dirname(found[0]), "per_window.csv")
        if draws and os.path.exists(pw):
            windows = read_windows(pw)
            rec["A_ci"] = episode_bootstrap(*window_outcome(windows, "A", decoder), draws, seed)
            if rec["M"] is not None:
                rec["M_ci"] = episode_bootstrap(*window_outcome(windows, "B", decoder), draws, seed)
        out[int(m["map"] or m["bare"])] = rec
    return out


def pooled_home(maps):
    """The training maps' A pooled over their windows (each map weighted by its window count), or None."""
    pts = [(maps[m]["A"], maps[m]["n"] or 1) for m in TRAINING_MAPS if m in maps and maps[m]["A"] is not None]
    return sum(a * n for a, n in pts) / sum(n for _, n in pts) if pts else None


# ---------------------------------------------------------------------------------------------
# the cost rule and the statistics
# ---------------------------------------------------------------------------------------------

def half_gap_line(a0, home):
    """The line an arena crosses when it has closed half of its step-0 gap to the home advantage."""
    return (a0 + home) / 2


def crossing(rows, column, target, budget):
    """The first scored step at or before `budget` whose `column` reaches `target` (`score_adapt.adaptation_cost`).

    Not reached and scored to the budget: right-censored. Not reached and not yet scored to the budget:
    incomplete, which is neither a crossing nor a censoring. `budget` None reads the whole curve.
    """
    use = [r for r in rows if budget is None or int(r["step"]) <= budget]
    c = adaptation_cost(use, column, target, higher_is_better=True)
    pts = {int(r["step"]): as_float(r.get(column)) for r in use}
    last = c["last_step_scored"]
    incomplete = bool(c["censored"] and budget is not None and (last is None or last < budget))
    return {"target": target, "step": c["step"], "value": c["value"],
            "censored": bool(c["censored"]) and not incomplete, "incomplete": incomplete, "last_step": last,
            "last_value": pts.get(last) if last is not None else None}


def median_censored(values):
    """Median with right-censored entries as +inf: censored (+inf) when the middle lands on one; None if empty."""
    v, n = sorted(values), len(values)
    if not v:
        return None
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def cost_rank_value(cost, budget):
    """A cost for rank statistics: the crossing step, or twice the budget for a censored arena (tied above all)."""
    return 2 * budget if cost is None else cost


def spearman(x, y):
    """scipy's Spearman correlation (average ranks for ties) over the pairs where both values are finite."""
    from scipy.stats import spearmanr
    pairs = [(a, b) for a, b in zip(x, y) if finite(a) and finite(b)]
    out = {"rho": None, "p": None, "n": len(pairs)}
    if len(pairs) < 3 or len({a for a, _ in pairs}) < 2 or len({b for _, b in pairs}) < 2:
        return out
    r = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    out.update(rho=float(r.statistic), p=float(r.pvalue))
    return out


def median_or_none(values):
    v = [x for x in values if finite(x)]
    return float(statistics.median(v)) if v else None


def auc_mean(points, budget):
    """Trapezoid area under A over [0, budget] divided by the budget: the time-averaged A in dB; None if unscored."""
    pts = [(s, v) for s, v in points if s <= budget]
    if not pts or pts[0][0] != 0 or pts[-1][0] != budget or budget <= 0:
        return None
    return sum((s1 - s0) * (v0 + v1) / 2 for (s0, v0), (s1, v1) in zip(pts, pts[1:])) / budget


def run_record(run, decoder, home, budget, D, draws, seed, notes):
    """Every per-arena number of one run under one decoder, or None when its step-0 A is missing."""
    colA = f"heldout_A_{decoder}"
    A = dict(run.series(colA))
    if 0 not in A:
        notes.append(f"{run.name}: no step-0 {colA}; left out")
        return None
    a0, rows = A[0], run.rows()
    line = half_gap_line(a0, home)
    half, homex = crossing(rows, colA, line, budget), crossing(rows, colA, home, budget)
    full = crossing(rows, colA, line, None)
    S = dict(run.series("heldout_latent_skill"))
    tm = dict(run.series(f"trainmap_A_{decoder}"))
    rec = {"arena": run.arena, "run": run.name, "seed": run.seed, "k": run.k, "variant": run.variant,
           "D": D.get(run.arena), "A0": a0, "A": {str(s): v for s, v in sorted(A.items())}, "A_budget": A.get(budget),
           "gain": A[budget] - a0 if budget in A else None, "half_gap_line": line,
           "cost_half_gap": half["step"], "censored_half_gap": half["censored"], "incomplete": half["incomplete"],
           "cost_home": homex["step"], "censored_home": homex["censored"],
           "cost_half_gap_full_curve": full["step"], "last_step": run.steps[-1], "A_last": A.get(max(A)),
           "auc": auc_mean(sorted(A.items()), budget),
           "S0": S.get(0), "S_budget": S.get(budget), "S": {str(s): v for s, v in sorted(S.items())},
           "lpips0": lpips_value(run, 0, decoder), "lpips_budget": lpips_value(run, budget, decoder),
           "M0": run.value(0, f"heldout_B_{decoder}"), "M_budget": run.value(budget, f"heldout_B_{decoder}"),
           "G_budget": run.value(budget, f"heldout_C_{decoder}"),
           "forgetting_budget": tm[budget] - tm[0] if 0 in tm and budget in tm else None,
           "directional_budget": run.value(budget, "directional_correct_frac"),
           "trainmap_S0": run.value(0, "trainmap_latent_skill"),
           "A0_ci": None, "A_budget_ci": None, "M0_ci": None,
           "other_decoders": {d: {"A0": run.value(0, f"heldout_A_{d}"), "A_budget": run.value(budget, f"heldout_A_{d}")}
                              for d in run.decoders() if d != decoder}}
    if draws:
        for step, key in ((0, "A0_ci"), (budget, "A_budget_ci")):
            rec[key] = checked_interval(run, step, "A", decoder, colA, draws, seed, notes)
        if rec["M0"] is not None:
            rec["M0_ci"] = checked_interval(run, 0, "B", decoder, f"heldout_B_{decoder}", draws, seed, notes)
    return rec


def checked_interval(run, step, outcome, decoder, column, draws, seed, notes):
    """The episode-bootstrap interval of one row's outcome, after checking its per-window mean against the row."""
    row = run.by_step.get(step)
    if row is None or as_float(row.get(column)) is None:
        return None
    path = local_per_window(run, row)
    if path is None:
        notes.append(f"{run.name} step {step}: no per_window.csv on this machine; interval left out")
        return None
    eps, vals = window_outcome(read_windows(path), outcome, decoder)
    if not vals:
        notes.append(f"{run.name} step {step}: per_window {rel(path)} has no {outcome} columns for {decoder}")
        return None
    mean = sum(vals) / len(vals)
    if abs(mean - float(row[column])) > 1e-6:
        notes.append(f"arena {run.arena} ({run.name}) step {step}: per_window mean {mean:.4f} differs from the row's "
                     f"{column} {float(row[column]):.4f} ({rel(path)})")
    return episode_bootstrap(eps, vals, draws, seed)


def lpips_value(run, step, decoder):
    """Decoded LPIPS at one step: the row's column, or for another decoder its per-window mean."""
    row = run.by_step.get(step)
    if row is None:
        return None
    if decoder == STOCK:
        return as_float(row.get("heldout_lpips_dec"))
    v = as_float(row.get(f"heldout_lpips_dec_{decoder}"))
    if v is not None:
        return v
    path = local_per_window(run, row)
    if path is None:
        return None
    _, vals = window_difference(read_windows(path), f"lpips_dec_{decoder}")
    return sum(vals) / len(vals) if vals else None


def censored_median_entry(records, key, budget):
    costs = [math.inf if r[key] is None else r[key] for r in records]
    m = median_censored(costs)
    if m is None:
        return {"value": None, "censored": None, "label": None}
    if math.isinf(m):
        return {"value": None, "censored": True, "label": f">{budget}"}
    return {"value": m, "censored": False, "label": f"{m:g}"}


def build_summary(records, budget, home):
    """The counts, medians and correlations the text quotes, over the arenas scored to the budget."""
    done = [r for r in records if not r["incomplete"]]
    steps = sorted({int(s) for r in done for s in r["A"] if int(s) <= budget})
    crossed = [r for r in done if r["cost_half_gap"] is not None]
    positive = [s for s in steps if s > 0]
    below = [s for s in steps if s < budget]
    late_from = max(below) if below else None
    late = [r["A_budget"] - r["A"][str(late_from)] for r in done
            if late_from is not None and r["A_budget"] is not None and str(late_from) in r["A"]]
    first = positive[0] if positive else None
    shares = [(r["A"][str(first)] - r["A0"]) / r["gain"] for r in done
              if first is not None and str(first) in r["A"] and r["gain"] and r["gain"] > 0]
    margins = [r["M_budget"] for r in done if r["M_budget"] is not None]
    cost_rank = [cost_rank_value(r["cost_half_gap"], budget) for r in done]
    col = {k: [r[k] for r in done] for k in ("D", "A0", "A_budget", "gain", "S0")}
    return {
        "n_arenas": len(records), "n_complete": len(done),
        "crossings": {
            "half_gap_by_budget": len(crossed),
            "half_gap_by_500": sum(1 for r in crossed if r["cost_half_gap"] <= 500),
            "half_gap_by_step": {str(s): sum(1 for r in crossed if r["cost_half_gap"] <= s) for s in steps},
            "home_by_budget": sum(1 for r in done if r["cost_home"] is not None),
            "above_home_at_budget": sum(1 for r in done if r["A_budget"] is not None and r["A_budget"] >= home),
            "margin_ties_by_budget": sum(1 for m in margins if m <= 0) if margins else None,
            "crossed_arenas": sorted(r["arena"] for r in crossed),
            "censored_arenas": sorted(r["arena"] for r in done if r["censored_half_gap"]),
            "home_crossed_arenas": sorted(r["arena"] for r in done if r["cost_home"] is not None),
            "incomplete_arenas": sorted(r["arena"] for r in records if r["incomplete"]),
        },
        "medians": {
            **{k: median_or_none([r[k] for r in done]) for k in
               ("A0", "A_budget", "gain", "half_gap_line", "auc", "S0", "S_budget", "lpips0", "lpips_budget",
                "M_budget", "G_budget", "forgetting_budget", "directional_budget")},
            "cost_half_gap": censored_median_entry(done, "cost_half_gap", budget),
            "cost_home": censored_median_entry(done, "cost_home", budget),
        },
        "spearman_cost_censored_as": 2 * budget,
        "spearman": {
            "D_vs_A0": spearman(col["D"], col["A0"]), "D_vs_A_budget": spearman(col["D"], col["A_budget"]),
            "D_vs_gain": spearman(col["D"], col["gain"]), "D_vs_cost": spearman(col["D"], cost_rank),
            "S0_vs_A_budget": spearman(col["S0"], col["A_budget"]), "S0_vs_cost": spearman(col["S0"], cost_rank),
            "A0_vs_cost": spearman(col["A0"], cost_rank),
        },
        "late_gain": {"from": late_from, "to": budget, "mean": sum(late) / len(late) if late else None,
                      "positive": sum(1 for g in late if g > 0), "n": len(late)},
        "gain_share_by_first_step": {"step": first, "mean": sum(shares) / len(shares) if shares else None,
                                     "n": len(shares)},
    }


def seed_entries(base, seeds, records_of, budget):
    """Per arena with more than one seed: A per seed and step, the spread at the budget, each seed's cost."""
    out = []
    for arena in sorted({r.arena for r in seeds}):
        if arena not in base or records_of.get(base[arena].name) is None:
            continue
        recs = [records_of[base[arena].name]] + [records_of[r.name] for r in seeds if r.arena == arena]
        recs = [r for r in recs if r is not None]
        at = {str(r["seed"]): r["A_budget"] for r in recs}
        vals = [v for v in at.values() if v is not None]
        s0 = recs[0]
        diff = {s: {step: r["A"][step] - s0["A"][step] for step in r["A"] if step in s0["A"]}
                for r in recs[1:] for s in [str(r["seed"])]}
        out.append({"arena": arena, "D": s0["D"], "seeds": [r["seed"] for r in recs], "A_budget": at,
                    "spread_budget": max(vals) - min(vals) if len(vals) > 1 else None,
                    "cost_half_gap": {str(r["seed"]): r["cost_half_gap"] for r in recs},
                    "A": {str(r["seed"]): r["A"] for r in recs}, "diff_from_seed0": diff})
    return out


def ladder_entries(base, ladder, records_of):
    """{arena: {k: A at the budget}} for the arenas with ladder runs, the headline k included."""
    out = {}
    for arena in sorted({r.arena for r in ladder}):
        runs = [r for r in ladder if r.arena == arena] + ([base[arena]] if arena in base else [])
        out[str(arena)] = {str(r.k): records_of[r.name]["A_budget"] for r in sorted(runs, key=lambda r: r.k)
                           if records_of.get(r.name)}
    return out


def recipe_entries(base, recipe, records_of, seeds_summary):
    spread = {e["arena"]: e["spread_budget"] for e in seeds_summary}
    out = []
    for r in sorted(recipe, key=lambda r: (r.arena, recipe_rank(r.variant))):
        rec, b = records_of.get(r.name), records_of.get(base[r.arena].name) if r.arena in base else None
        if rec is None:
            continue
        delta = rec["A_budget"] - b["A_budget"] if b and rec["A_budget"] is not None and b["A_budget"] is not None \
            else None
        out.append({"arena": r.arena, "variant": r.variant, "label": RECIPE_LABELS.get(r.variant, r.variant),
                    "run": r.name, "A_budget": rec["A_budget"], "delta_vs_base": delta, "A_last": rec["A_last"],
                    "last_step": rec["last_step"], "cost_half_gap": rec["cost_half_gap"],
                    "cost_half_gap_full_curve": rec["cost_half_gap_full_curve"],
                    "seed_spread_budget": spread.get(r.arena)})
    return out


def recipe_rank(variant):
    return (RECIPE_ORDER.index(variant) if variant in RECIPE_ORDER else len(RECIPE_ORDER), variant)


# ---------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------

def budget_label(budget):
    return f"{budget // 1000}k" if budget >= 1000 and budget % 1000 == 0 else str(budget)


def num(v, digits, signed=False, prov=False):
    """A table number (`$-$` for minus), or `\\tbd` when the rows do not carry it."""
    if not finite(v):
        return "\\tbd"
    s = f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"
    s = s.replace("-", "$-$")
    return f"\\prov{{{s}}}" if prov else s


def cost_cell(step, censored, incomplete, budget, prov=False):
    if incomplete:
        return "\\tbd"
    s = f"${{>}}${budget:,}" if censored else f"{step:,}"
    return f"\\prov{{{s}}}" if prov else s


def median_cost_cell(entry, prov):
    if entry["censored"] is None:
        return "\\tbd"
    s = f"${{>}}${int(entry['label'][1:]):,}" if entry["censored"] else f"{entry['value']:,.0f}"
    return f"\\prov{{{s}}}" if prov else s


def header(files, when, lines):
    first = f"% generated by make_adapt_figures.py from {', '.join(rel(f) for f in files)} at {when}"
    return "\n".join([first] + [f"% {x}" for x in lines]) + "\n"


def tabular(colspec, head, body):
    out = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    out += ["\\midrule" if row == "midrule" else " & ".join(row) + " \\\\" for row in body]
    return "\n".join(out + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def cost_table(records, summary, decoder, budget, prov):
    """The per-arena cost table: arena, D, A0, A at the budget, half-gap line, cost, S and LPIPS at 0 and the budget."""
    b = budget_label(budget)
    others = sorted({d for r in records for d in r["other_decoders"]})
    head = ["Arena", "$D$", "$A_0$", f"$A_{{\\mathrm{{{b}}}}}$", "Half-gap line", "Cost", "$S_0$",
            f"$S_{{\\mathrm{{{b}}}}}$", "LPIPS$_0$", f"LPIPS$_{{\\mathrm{{{b}}}}}$"]
    head += [f"$A_{{\\mathrm{{{b}}}}}$, {d}" for d in others]
    body = []
    for r in records:
        row = [str(r["arena"]), num(r["D"], 3, prov=prov), num(r["A0"], 2, True, prov),
               num(r["A_budget"], 2, True, prov), num(r["half_gap_line"], 2, prov=prov),
               cost_cell(r["cost_half_gap"], r["censored_half_gap"], r["incomplete"], budget, prov),
               num(r["S0"], 2, prov=prov), num(r["S_budget"], 2, prov=prov), num(r["lpips0"], 3, prov=prov),
               num(r["lpips_budget"], 3, prov=prov)]
        row += [num(r["other_decoders"].get(d, {}).get("A_budget"), 2, True, prov) for d in others]
        body.append(row)
    m = summary["medians"]
    other_medians = [median_or_none([r["other_decoders"].get(d, {}).get("A_budget") for r in records
                                     if not r["incomplete"]]) for d in others]
    body += ["midrule", ["Median", "", num(m["A0"], 2, True, prov), num(m["A_budget"], 2, True, prov),
                         num(m["half_gap_line"], 2, prov=prov), median_cost_cell(m["cost_half_gap"], prov),
                         num(m["S0"], 2, prov=prov), num(m["S_budget"], 2, prov=prov), num(m["lpips0"], 3, prov=prov),
                         num(m["lpips_budget"], 3, prov=prov)] + [num(v, 2, True, prov) for v in other_medians]]
    return tabular("r" + "c" * (len(head) - 1), head, body)


def perarena_table(records, budget, prov):
    """The appendix table `tab:perarena-adapt`: the columns appendix.tex declares, one row per arena by D."""
    b = budget_label(budget)
    head = ["Arena", "$D$", "$A_0$", "Half-gap line", "Half gap", "Home line", f"$A$ at {b}", f"$M$ at {b}",
            "Forgetting", "Directional"]
    body = [[str(r["arena"]), num(r["D"], 3, prov=prov), num(r["A0"], 2, True, prov),
             num(r["half_gap_line"], 2, prov=prov),
             cost_cell(r["cost_half_gap"], r["censored_half_gap"], r["incomplete"], budget, prov),
             cost_cell(r["cost_home"], r["censored_home"], r["incomplete"], budget, prov),
             num(r["A_budget"], 2, True, prov), num(r["M_budget"], 3, True, prov),
             num(r["forgetting_budget"], 2, True, prov), num(r["directional_budget"], 2, prov=prov)]
            for r in records]
    return tabular("r" + "c" * (len(head) - 1), head, body)


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------

def style():
    """Print-size type: 6 pt labels, 5.5 pt ticks, TrueType embedding (no Type 3 fonts in the PDF)."""
    have = {f.name for f in font_manager.fontManager.ttflist}
    fam = next((f for f in ("Arial", "Helvetica", "DejaVu Sans") if f in have), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [fam, "DejaVu Sans"], "font.size": 6, "axes.labelsize": 6,
        "axes.titlesize": 6, "legend.fontsize": 5.5, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
        "ytick.major.size": 2, "xtick.minor.size": 0, "ytick.minor.size": 0, "lines.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": SECONDARY, "axes.labelcolor": INK,
        "xtick.color": SECONDARY, "ytick.color": SECONDARY, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "text.color": INK, "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.labelpad": 1.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "mathtext.fontset": "dejavusans" if fam == "DejaVu Sans" else "custom", "mathtext.rm": fam,
        "mathtext.it": f"{fam}:italic", "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def new_figure(size, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=size, layout="constrained", squeeze=False)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.03, hspace=0.02)
    return fig, list(axes[0])


def save(fig, out_dir, stem):
    """PDF (no creation date, so an unchanged figure rewrites byte-identical) and a 600 dpi PNG."""
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, f"{stem}.pdf"), os.path.join(out_dir, f"{stem}.png")]
    fig.savefig(paths[0], metadata={"CreationDate": None, "Creator": None, "Producer": None})
    fig.savefig(paths[1], dpi=600, metadata={"Software": None})
    plt.close(fig)
    return paths


def d_colours(records):
    """(colormap, norm) of D over the drawn arenas: the one-hue blue ramp, light near to dark far."""
    ds = [r["D"] for r in records if r["D"] is not None]
    cmap = LinearSegmentedColormap.from_list("d_blue", BLUE_RAMP)
    return cmap, Normalize(vmin=min(ds) if ds else 0.0, vmax=max(ds) if ds else 1.0)


def colour_of(rec, cmap, norm):
    return cmap(norm(rec["D"])) if rec["D"] is not None else MUTED


def zero_position(steps):
    """Where step 0 sits on the log axis: a factor 2.5 left of the first positive step."""
    pos = sorted(s for s in steps if s > 0)
    return pos[0] / 2.5 if pos else 1.0


def step_label(s):
    return "0" if s == 0 else f"{s // 1000}k" if s >= 1000 and s % 1000 == 0 else str(s)


def step_axis(ax, steps, label="LoRA updates", labelled=None):
    """A log axis of updates with 0 as its first labelled tick, a break mark after it; returns x of step 0.

    Grid steps are labelled left to right while they sit at least a quarter decade apart, or exactly the steps
    in `labelled`; the others get an unlabelled minor tick.
    """
    z = zero_position(steps)
    ax.set_xscale("log")
    grid = sorted(set(steps) | {0})
    if labelled is None:
        ticks, last = [], None
        for s in grid:
            x = z if s == 0 else s
            if last is None or math.log10(x / last) >= 0.25:
                ticks.append(s)
                last = x
    else:
        ticks = [s for s in grid if s in set(labelled) | {0}]
    ax.xaxis.set_major_locator(ticker.FixedLocator([z if s == 0 else s for s in ticks]))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([step_label(s) for s in ticks]))
    ax.xaxis.set_minor_locator(ticker.FixedLocator([s for s in grid if s not in ticks]))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", length=1.2, width=0.4)
    ax.set_xlim(z / 1.4, max(steps) * 1.4)
    ax.set_xlabel(label)
    first = min((s for s in steps if s > 0), default=z * 2.5)
    brk = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(math.sqrt(z * first), 0, "//", transform=brk, ha="center", va="center", fontsize=5, color=SECONDARY,
            clip_on=False, zorder=5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2})
    return z


def xs_of(points, z):
    return [z if s == 0 else s for s, _ in points]


def home_line(ax, value, z, text="home", colour=SECONDARY):
    ax.axhline(value, color=colour, lw=0.7, ls=(0, (3, 2)), zorder=2)
    if text:
        ax.text(z, value, f" {text}", ha="left", va="bottom", fontsize=5, color=SECONDARY, zorder=5)


def colourbar(fig, axes, cmap, norm):
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=0.07, pad=0.02, aspect=16)
    cb.set_label("$D$", labelpad=1)
    cb.ax.tick_params(labelsize=5, width=0.4, length=1.5, pad=1)
    cb.locator = ticker.MaxNLocator(3)
    cb.update_ticks()
    cb.outline.set_linewidth(0.4)
    return cb


def draw_curve(ax, points, z, colour, crossing_step=None, censored=False, ls="-", lw=0.8, marks=True):
    ax.plot(xs_of(points, z), [v for _, v in points], color=colour, lw=lw, ls=ls, zorder=3, solid_capstyle="round")
    if not marks or not points:
        return
    if crossing_step is not None:
        v = dict(points)[crossing_step]
        ax.plot(z if crossing_step == 0 else crossing_step, v, "o", ms=3, mfc=colour, mec="white", mew=0.4, zorder=4)
    elif censored:
        s, v = points[-1]
        ax.plot(s, v, "o", ms=3, mfc="white", mec=colour, mew=0.6, zorder=4)


def fig_curves(records, runs, home, budget, out_dir, column="A"):
    """Figure 3: A (or S) against updates, one line per arena coloured by D, home and half-gap lines, crossings."""
    cmap, norm = d_colours(records)
    fig, (ax,) = new_figure(FIG3_SIZE)
    steps = sorted({s for r in records for s in runs[r["run"]].steps if s <= budget})
    z = step_axis(ax, steps)
    skill = column == "S"
    for r in sorted(records, key=lambda r: r["D"] if r["D"] is not None else -1):
        c = colour_of(r, cmap, norm)
        pts = runs[r["run"]].series("heldout_latent_skill" if skill else f"heldout_A_{r['decoder']}", budget)
        if not skill:
            ax.axhline(r["half_gap_line"], color=c, lw=0.35, alpha=0.25, zorder=1)
        draw_curve(ax, pts, z, c, None if skill else r["cost_half_gap"], not skill and r["censored_half_gap"],
                   marks=not skill)
    ref = median_or_none([r["trainmap_S0"] for r in records]) if skill else home
    if ref is not None:
        home_line(ax, ref, z)
    ax.set_ylabel("$S$ (dB)" if skill else "$A$ (dB)")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    colourbar(fig, ax, cmap, norm)
    return save(fig, out_dir, "fig3_adaptation_skill" if skill else "fig3_adaptation_curves")


def fig_terciles(records, runs, home, budget, out_dir):
    """A averaged over arenas in D terciles (fewer groups below three arenas); the band is the range."""
    ordered = [r for r in sorted(records, key=lambda r: r["D"]) if r["D"] is not None]
    groups = [g for g in np.array_split(np.arange(len(ordered)), min(3, len(ordered))) if len(g)]
    fig, (ax,) = new_figure(WIDE_SIZE)
    steps = sorted({s for r in ordered for s in runs[r["run"]].steps if s <= budget})
    z = step_axis(ax, steps)
    for gi, g in enumerate(groups):
        members = [ordered[i] for i in g]
        common = sorted(set.intersection(*[{s for s, _ in runs[m["run"]].series(f"heldout_A_{m['decoder']}", budget)}
                                            for m in members]))
        vals = np.array([[dict(runs[m["run"]].series(f"heldout_A_{m['decoder']}"))[s] for s in common]
                         for m in members])
        c = TERCILE_COLOURS[gi if len(groups) == 3 else min(2 * gi, 2)]
        x = [z if s == 0 else s for s in common]
        ax.fill_between(x, vals.min(0), vals.max(0), color=c, alpha=0.12, lw=0, zorder=1)
        lo, hi = members[0]["D"], members[-1]["D"]
        ax.plot(x, vals.mean(0), color=c, lw=1.0, marker=MARKERS[gi], ms=2.5, zorder=3,
                label=f"$D$ {lo:.3f}–{hi:.3f} ({len(members)})")
    home_line(ax, home, z)
    ax.set_ylabel("$A$ (dB), mean over arenas")
    ax.legend(loc="lower right")
    return save(fig, out_dir, "fig3_adaptation_terciles")


def arena_styles(arenas, records):
    """Identity encoding for a handful of named arenas: categorical slots and markers in order of D."""
    d = {r["arena"]: r["D"] for r in records}
    order = sorted(arenas, key=lambda a: (d.get(a) is None, d.get(a) or 0, a))
    return {a: (SERIES[i % len(SERIES)], MARKERS[i % len(MARKERS)]) for i, a in enumerate(order)}


def arena_label(arena, records):
    d = next((r["D"] for r in records if r["arena"] == arena), None)
    return f"arena {arena}" + (f" ($D$ {d:.3f})" if d is not None else "")


def fig_seeds(entries, records, base, seeds, decoder, budget, out_dir):
    """Seed spread: each seed's curve (seed 0 solid), and every other seed's difference from seed 0 per step."""
    fig, (ax, ax2) = new_figure((WIDE_SIZE[0], 1.7), ncols=2)
    arenas = [e["arena"] for e in entries]
    look = arena_styles(arenas, records)
    runs = [base[a] for a in arenas] + [r for r in seeds if r.arena in arenas]
    steps = sorted({s for r in runs for s in r.steps if s <= budget})
    z = step_axis(ax, steps)
    z2 = step_axis(ax2, steps)
    styles = ("-", (0, (2.5, 1.5)), (0, (1, 1)), (0, (4, 1, 1, 1)))
    for e in entries:
        c, m = look[e["arena"]]
        for i, seed in enumerate(e["seeds"]):
            run = base[e["arena"]] if seed == BASE_SEED else \
                next(r for r in seeds if r.arena == e["arena"] and r.seed == seed)
            draw_curve(ax, run.series(f"heldout_A_{decoder}", budget), z, c, marks=False, ls=styles[i % 4])
        for i, (seed, d) in enumerate(sorted(e["diff_from_seed0"].items())):
            pts = sorted((int(s), v) for s, v in d.items() if int(s) <= budget)
            ax2.plot(xs_of(pts, z2), [v for _, v in pts], color=c, lw=0.8, ls=styles[i % 4], marker=m, ms=2,
                     label=arena_label(e["arena"], records) if i == 0 else None)
    ax2.axhline(0, color=INK, lw=0.5, zorder=1)
    ax.set_ylabel("$A$ (dB)")
    ax2.set_ylabel("$A$, seed $-$ seed 0 (dB)")
    n_seeds = max(len(e["seeds"]) for e in entries)
    seed_names = next(e["seeds"] for e in entries if len(e["seeds"]) == n_seeds)
    handles = [Line2D([], [], color=SECONDARY, lw=0.8, ls=styles[i % 4]) for i in range(n_seeds)]
    ax.legend(handles, [f"seed {s}" for s in seed_names], loc="lower right")
    fig.legend(*ax2.get_legend_handles_labels(), loc="outside lower center", ncol=min(2, len(entries)))
    return save(fig, out_dir, "fig3_adaptation_seeds")


def fig_ladder(ladder, records, home, budget, out_dir):
    """A at the budget against adaptation episodes, one line per arena; its step-0 A dotted, home dashed."""
    look = arena_styles([int(a) for a in ladder], records)
    fig, (ax,) = new_figure(SMALL_SIZE)
    ks = sorted({int(k) for v in ladder.values() for k in v})
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(ticker.FixedLocator(ks))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(k) for k in ks]))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    for arena, byk in sorted(ladder.items(), key=lambda kv: int(kv[0])):
        rec = next((r for r in records if r["arena"] == int(arena)), None)
        c, m = look[int(arena)]
        pts = sorted((int(k), v) for k, v in byk.items() if v is not None)
        ax.plot([k for k, _ in pts], [v for _, v in pts], color=c, lw=0.8, marker=m, ms=2.5,
                label=arena_label(int(arena), records))
        if rec:
            ax.axhline(rec["A0"], color=c, lw=0.5, ls=(0, (1, 1.5)), zorder=1)
    ax.axhline(home, color=SECONDARY, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.text(ks[0], home, " home", ha="left", va="bottom", fontsize=5, color=SECONDARY)
    ax.set_xlim(ks[0] / 1.3, ks[-1] * 1.3)
    ax.set_xlabel("adaptation episodes")
    ax.set_ylabel(f"$A$ at {budget_label(budget)} (dB)")
    ax.legend(loc="lower right")
    return save(fig, out_dir, "fig_adapt_ladder")


def fig_recipe(recipe, base, seeds, records_of, home, out_dir):
    """The recipe tests: per arena the headline run, its seed-1 run in grey, and each variant's full curve."""
    arenas = sorted({r.arena for r in recipe if r.arena in base})
    fig, axes = new_figure((WIDE_SIZE[0], 1.6), ncols=max(1, len(arenas)))
    handles = {}
    for ax, arena in zip(axes, arenas):
        b = base[arena]
        brec = records_of[b.name]
        lines = [(b, "base (lr 1e-4)", SERIES[0], MARKERS[0], "-")]
        lines += [(r, "seed 1" if r.seed == 1 else f"seed {r.seed}", MUTED, "", (0, (2.5, 1.5)))
                  for r in seeds if r.arena == arena]
        variants = sorted((r for r in recipe if r.arena == arena), key=lambda r: recipe_rank(r.variant))
        lines += [(r, RECIPE_LABELS.get(r.variant, r.variant), SERIES[1 + recipe_rank(r.variant)[0] % 3],
                   MARKERS[1 + recipe_rank(r.variant)[0] % 3], "-") for r in variants]
        steps = sorted({s for r, *_ in lines for s in r.steps})
        z = step_axis(ax, steps, labelled=RECIPE_TICKS)
        for r, label, c, m, ls in lines:
            pts = r.series(f"heldout_A_{brec['decoder']}")
            h, = ax.plot(xs_of(pts, z), [v for _, v in pts], color=c, lw=0.8, ls=ls, marker=m or None, ms=2, zorder=3)
            handles.setdefault(label, h)
        ax.axhline(brec["half_gap_line"], color=INK, lw=0.4, alpha=0.35, zorder=1)
        ax.text(z, brec["half_gap_line"], " half gap", ha="left", va="bottom", fontsize=5, color=SECONDARY)
        home_line(ax, home, z)
        ax.set_title(f"arena {arena} ($D$ {brec['D']:.3f})" if brec["D"] is not None else f"arena {arena}")
        ax.set_ylabel("$A$ (dB)")
    fig.legend(list(handles.values()), list(handles), loc="outside lower center", ncol=min(5, len(handles)))
    return save(fig, out_dir, "fig_adapt_recipe")


def fig_zero_shot(zero_shot, D, key, out_dir, stem, ylabel, zero=False):
    """Figure 2: one value per arena sorted by D, training maps shaded, a marker per backbone, each home dashed.

    A backbone keeps its colour and marker whichever rows are present (U-Net, PixArt, SD 3.5, then others).
    """
    names = [r for r in ROW_ORDER if r in zero_shot] + sorted(r for r in zero_shot if r not in ROW_ORDER)
    slot = {n: i for i, n in enumerate(list(ROW_ORDER) + [n for n in names if n not in ROW_ORDER])}
    rows = [n for n in names if any(v.get(key) is not None for v in zero_shot[n]["maps"].values())]
    arenas = sorted({a for r in rows for a, v in zero_shot[r]["maps"].items() if v.get(key) is not None and a in D},
                    key=lambda a: (D[a], a))
    fig, (ax,) = new_figure(FIG2_SIZE)
    pos = {a: i for i, a in enumerate(arenas)}
    for a in arenas:
        if a in TRAINING_MAPS:
            ax.axvspan(pos[a] - 0.5, pos[a] + 0.5, color=SHADE, alpha=0.6, lw=0, zorder=0)
    width = 0.5 / max(1, len(rows))
    for i, name in enumerate(rows):
        maps = zero_shot[name]["maps"]
        off = (i - (len(rows) - 1) / 2) * width
        pts = [(pos[a] + off, maps[a][key], maps[a].get(f"{key}_ci")) for a in arenas
               if a in maps and maps[a].get(key) is not None]
        c, m = SERIES[slot[name] % len(SERIES)], MARKERS[slot[name] % len(MARKERS)]
        for x, v, ci in pts:
            if ci:
                ax.plot([x, x], ci, color=c, lw=0.6, zorder=2, solid_capstyle="butt")
        ax.plot([x for x, _, _ in pts], [v for _, v, _ in pts], ls="none", marker=m, ms=2.8, mfc=c,
                mec="white", mew=0.3, zorder=3, label=ROW_NAMES.get(name, name))
        home = zero_shot[name].get("home") if key == "A" else zero_shot[name].get("home_M")
        if home is not None:
            ax.axhline(home, color=c, lw=0.7, ls=(0, (3, 2)), zorder=1)
    if zero:
        ax.axhline(0, color=INK, lw=0.5, zorder=1)
    homes = [zero_shot[r].get("home") for r in rows if key == "A" and zero_shot[r].get("home") is not None]
    if homes:
        ax.text(len(arenas) - 0.45, max(homes), "home", ha="right", va="bottom", fontsize=5, color=SECONDARY)
    # two-digit arena numbers need about 0.1 in each: past 13 arenas every other label drops a line
    stagger = len(arenas) > 13
    ax.set_xticks(range(len(arenas)), [("\n" if stagger and i % 2 else "") + str(a) for i, a in enumerate(arenas)])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(arenas) - 0.4)
    ax.set_xlabel("arena, by $D$")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    if len(rows) > 1:
        fig.legend(loc="outside upper center", ncol=len(rows), handletextpad=0.1, columnspacing=0.8,
                   fontsize=5)
    return save(fig, out_dir, stem)


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------

def resolve_home(a):
    if a.home is not None and a.home_json:
        raise SystemExit("give --home or --home-json, not both")
    if a.home_json:
        return home_from_json(a.home_json, a.decoder), a.home_json
    if a.home is not None:
        return a.home, "--home"
    if a.decoder == STOCK:
        return STOCK_HOME, "default: stock decoder, scene crop, training maps' 512 validation windows (2026-09-27)"
    raise SystemExit(f"--decoder {a.decoder} needs its own home: --home <dB> or --home-json <eval_tf metrics.json>")


def zero_shot_rows(a, records, home, draws, notes):
    """{row: {source, home, maps}}: the U-Net from the headline runs' step 0, replaced by a fresh `unet` read."""
    out = {"unet": {"source": "adaptation step 0", "home": home,
                    "maps": {r["arena"]: {"A": r["A0"], "A_ci": r["A0_ci"], "M": r["M0"], "M_ci": r["M0_ci"]}
                             for r in records}}} if records else {}
    dirs = []
    if a.fresh_root and os.path.isdir(a.fresh_root):
        dirs += [(d, os.path.join(a.fresh_root, d)) for d in sorted(os.listdir(a.fresh_root))
                 if os.path.isdir(os.path.join(a.fresh_root, d))]
    for spec in a.zero_shot:
        name, sep, path = spec.partition("=")
        if not sep:
            raise SystemExit(f"--zero-shot {spec!r}: expected NAME=DIR")
        dirs.append((name, path))
    for name, path in dirs:
        maps = load_zero_shot_row(path, a.decoder, draws, a.seed)
        if not maps:
            notes.append(f"{rel(path)}: no one-tic eval_tf reads keyed by map; row {name} skipped")
            continue
        own = pooled_home(maps)
        out[name] = {"source": rel(path), "home": own if own is not None else (home if name == "unet" else None),
                     "home_M": median_or_none([maps[m]["M"] for m in TRAINING_MAPS if m in maps]),
                     "maps": maps}
    return out


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def build_parser():
    p = argparse.ArgumentParser(description="The adaptation study's figures, tables and summary.")
    p.add_argument("--runs-glob", default=os.path.join(REPO, "results", "adapt", "*", "scores.jsonl"))
    p.add_argument("--distances", default=os.path.join(REPO, "results", "distance_study", "distances_sd1.json"),
                   help="distances_<space>.json; each arena's primary D colours and sorts everything")
    p.add_argument("--decoder", default=STOCK, help="stock or tuned (any decoder the rows carry as heldout_A_<name>)")
    p.add_argument("--home", type=float, default=None,
                   help=f"the training maps' scene A for --decoder (default {STOCK_HOME} for the stock decoder)")
    p.add_argument("--home-json", default=None,
                   help="an eval_tf metrics.json of the training maps; home = scene_psnr_dec - scene_copy_psnr_dec")
    p.add_argument("--weights", default="live", choices=("live", "ema"))
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="the fixed budget: A at it, censoring beyond it")
    p.add_argument("--fresh-root", default=os.path.join(REPO, "results", "fresh_rescore"),
                   help="<root>/<row>/<map dir>/metrics.json zero-shot reads; each row a marker in Figure 2")
    p.add_argument("--zero-shot", action="append", default=[], metavar="NAME=DIR",
                   help="one more zero-shot row: a directory of eval_tf reads keyed by map (repeatable)")
    p.add_argument("--with-raw", action="store_true", help="also draw Figure 2b, the margin M, where rows carry it")
    p.add_argument("--prov", action="store_true", help="wrap every table number in \\prov{} (provisional marks)")
    p.add_argument("--bootstrap", type=int, default=2000, help="episode-bootstrap draws for the intervals (0: none)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=os.path.join(HERE, "figures"))
    p.add_argument("--tables-dir", default=os.path.join(HERE, "tables"))
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    notes = []
    home, home_source = resolve_home(a)
    paths = sorted(glob.glob(a.runs_glob))
    if not paths:
        raise SystemExit(f"no scores.jsonl matches {a.runs_glob}")
    D = load_distances(a.distances)
    runs = load_runs(paths, a.weights, a.decoder, notes)
    base, seeds, ladder, recipe = classify(runs, notes)
    if not base:
        raise SystemExit("no headline runs (rank 16, k 8, seed 0, no variant) among " + a.runs_glob)
    for arena in sorted(base):
        if arena not in D:
            notes.append(f"arena {arena}: no primary D in {rel(a.distances)}")
    records_of = {}
    for run in runs:
        draws = a.bootstrap if run is base.get(run.arena) else 0
        rec = run_record(run, a.decoder, home, a.budget, D, draws, a.seed, notes)
        if rec is not None:
            rec["decoder"] = a.decoder
        records_of[run.name] = rec
    records = sorted((records_of[r.name] for r in base.values() if records_of[r.name]),
                     key=lambda r: (r["D"] is None, r["D"] or 0, r["arena"]))
    if not records:
        raise SystemExit(f"no headline run carries heldout_A_{a.decoder} at step 0; notes: " + "; ".join(notes))
    by_name = {r.name: r for r in runs}
    for r in records:
        if r["incomplete"]:
            notes.append(f"arena {r['arena']}: scored to step {r['last_step']} of {a.budget}; left out of the counts")

    summary = build_summary(records, a.budget, home)
    seed_summary = seed_entries(base, seeds, records_of, a.budget)
    ladder_summary = ladder_entries(base, ladder, records_of)
    zero_shot = zero_shot_rows(a, records, home, a.bootstrap, notes)
    when = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    used = [base[a_].path for a_ in sorted(base)]
    inputs = [{"path": rel(p), "sha256": sha256(p)} for p in sorted({r.path for r in runs}) + [a.distances]]
    summary = {"generated_by": "paper/make_adapt_figures.py", "generated_at": when, "decoder": a.decoder,
               "home": home, "home_source": rel(home_source) if os.path.exists(home_source) else home_source,
               "budget": a.budget, "weights": a.weights, **summary,
               "per_arena": records, "seeds": seed_summary, "ladder": ladder_summary,
               "recipe": recipe_entries(base, recipe, records_of, seed_summary),
               "zero_shot": {k: {**v, "maps": {str(m): {x: y for x, y in e.items() if x != "path"}
                                              for m, e in v["maps"].items()}} for k, v in zero_shot.items()},
               "inputs": inputs, "notes": notes}

    os.makedirs(a.tables_dir, exist_ok=True)
    head = header([a.runs_glob, a.distances], when, [
        f"{len(used)} headline runs (rank 16, 8 episodes, seed 0), {a.weights} weights, decoder {a.decoder}, "
        f"home {home:.3f} dB ({rel(home_source) if os.path.exists(home_source) else home_source}), budget {a.budget}",
        "\\tbd marks a quantity these rows do not carry yet; costs are first grid crossings, >budget right-censored",
    ] + [f"  {rel(p)} sha256 {sha256(p)[:16]}" for p in used])
    written = []
    for name, body in (("adapt_cost.tex", cost_table(records, summary, a.decoder, a.budget, a.prov)),
                       ("adapt_perarena.tex", perarena_table(records, a.budget, a.prov))):
        path = os.path.join(a.tables_dir, name)
        with open(path, "w") as f:
            f.write(head + body)
        written.append(path)
    path = os.path.join(a.tables_dir, "adapt_summary.json")
    with open(path, "w") as f:
        json.dump(jsonable(summary), f, indent=1)
        f.write("\n")
    written.append(path)

    style()
    drawn = [r for r in records if r["D"] is not None]
    if drawn:
        written += fig_curves(drawn, by_name, home, a.budget, a.out_dir, "A")
        written += fig_curves(drawn, by_name, home, a.budget, a.out_dir, "S")
        written += fig_terciles(drawn, by_name, home, a.budget, a.out_dir)
    else:
        print("note: no arena has a D; Figure 3 and its variants not drawn")
    if seed_summary:
        written += fig_seeds(seed_summary, records, base, seeds, a.decoder, a.budget, a.out_dir)
    if ladder_summary:
        written += fig_ladder(ladder_summary, records, home, a.budget, a.out_dir)
    if [r for r in recipe if r.arena in base]:
        written += fig_recipe(recipe, base, seeds, records_of, home, a.out_dir)
    if zero_shot:
        written += fig_zero_shot(zero_shot, D, "A", a.out_dir, "fig2a_advantage_by_distance", "$A_0$ (dB)")
        if a.with_raw and any(e.get("M") is not None for v in zero_shot.values() for e in v["maps"].values()):
            written += fig_zero_shot(zero_shot, D, "M", a.out_dir, "fig2b_margin_by_distance", "$M_0$", zero=True)
    for p in written:
        print("wrote", rel(p))
    for n in notes:
        print("note:", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
