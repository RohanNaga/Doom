"""
The adaptation study's figures, tables and summary, from `score_adapt.py` rows and `eval_tf.py` reads.

    python paper/make_adapt_figures.py --with-raw                        # stock decoder, scene home 4.138 dB
    python paper/make_adapt_figures.py --decoder tuned --home-json results/home_unet_tuned/metrics.json \\
        --out-dir paper/figures/tuned --tables-dir paper/tables/tuned --with-raw
    python paper/make_adapt_figures.py ... --headline-variant g8k       # the 8k-grid reruns as the headline set

**Inputs.** Every `scores.jsonl` that `--runs-glob` matches (default `results/adapt/*/scores.jsonl`), one row
per scored checkpoint, in a run directory named `<source>_<set>_map<NN>_r<rank>_k<k>_s<seed>[_<variant>]`.
The rank-16, eight-episode, seed-0 runs of `--headline-variant` (default: no variant) are the headline set, one
per arena; other seeds of that variant are the seed spread, other `k` the episode ladder, and every other variant
(`lr5e4`, `lr3e4`, `g8k`) a recipe test against the base recipe. Seeds and ladder rungs are compared with the
seed-0, eight-episode run of their own variant at the largest step both reached. Only `--weights` rows are read
(live by default); where a run was scored under more than one evaluation configuration, the configuration with the
most steps is read (the latest on a tie) and the choice is noted. The frozen per-arena distance D is each arena's
primary entry in `--distances` (`distances_sd1.json`); it labels tables only and orders or colours nothing.
Zero-shot reads of other backbones are directories of `eval_tf.py` reads keyed by map,
`<fresh-root>/<row>/<name with map<NN>>/metrics.json`; a `_h<K>` suffix other than `_h1` is skipped.

**Quantities**, `score_adapt.py`'s outcomes on the scene crop under `--decoder`: A = `heldout_A_<decoder>`
(decoded advantage over copy-last, dB); M = `heldout_B_<decoder>` (perceptual margin against raw persistence,
present when the raw frames were scored; otherwise read from the adapter's zero-shot row at the budget, noted);
G = `heldout_C_<decoder>`; S = `heldout_latent_skill` (no decoder); LPIPS = `heldout_lpips_dec`; forgetting = the
change of `trainmap_A_<decoder>` from step 0; directional = `directional_correct_frac`. Per-arena intervals are
95% percentile bootstraps over held-out episodes from the per-window files the rows name (found beside the run when
the recorded path is a server path), with the windows `eval_tf.py` flags as duplicates left out as
`score_adapt.py` leaves them out; each file's mean is checked against its row and a disagreement is noted.

**The cost rule** (`.claude/analyses/cost-target-decision-2026-09-26.md`, `lora-results-decision-2026-09-27.md`).
Per arena the half-gap line is (A0 + home) / 2, where home is the training maps' A on the same crop and decoder:
`--home` (default 4.138 dB, the stock decoder's scene home), or `--home-json`, an eval_tf read of the training
maps, as `scene_psnr_dec - scene_copy_psnr_dec` (duplicate-free means where the read has them). The cost is
the first grid step at or before the budget (`--budget`, default the headline grid's last step) whose A reaches the
line; a curve scored to the budget that never reaches it is right-censored, and one not yet scored to the budget is
incomplete and left out of every count. A median over arenas ranks a censored arena above every crossing and is
censored when the middle lands on one. Spearman correlations are scipy's (average ranks) with a censored cost
ranked as twice the budget.

**Outputs.** In `--out-dir`, PDF and 600 dpi PNG through `figstyle.save` (drawn at printed size for the 5.5 in
page, refused when degenerate): `fig3_adaptation_curves` (A per arena, the blue ramp keyed to S0, three arenas
labelled), `fig3_adaptation_skill` (S instead of A), `fig3_adaptation_seeds`, `fig_adapt_ladder` and
`fig_adapt_recipe` (when their runs carry the decoder), `fig2c_outcomes_by_skill` (A at the budget against S0),
`fig2a_advantage_by_distance` and, with `--with-raw`, `fig2b_margin_by_distance` (zero-shot A and M per arena and
backbone). In `--tables-dir`, under a
provenance header: `adapt_cost.tex` (per arena), `adapt_perarena.tex` (the appendix table body) and
`adapt_summary.json`, the numbers the text quotes.
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
sys.path.insert(0, HERE)
from score_adapt import DUPLICATE_FLAGS, OUTCOMES, STOCK, adaptation_cost  # noqa: E402

import figstyle as fs  # noqa: E402
from figstyle import step_axis, xs_of  # noqa: E402
from matplotlib import ticker  # noqa: E402

STOCK_HOME = 4.138        # training maps' scene A, stock decoder, 512 validation windows (lora-results decision)
TRAINING_MAPS = (2, 3, 4, 5)
BASE_RANK, BASE_K, BASE_SEED = 16, 8, 0
FALLBACK_BUDGET = 4000
FIXED_THRESHOLD = 3.5     # dB; a fixed A threshold beside the per-arena half-gap line (statistics decision)
NAMED_ARENAS = (7, 9, 12)  # the far arena, the one at the training maps' line, the censored one (standard, Fig. 4)
RUN_RE = re.compile(r"_map(?P<map>\d+)_r(?P<rank>\d+)_k(?P<k>\d+)_s(?P<seed>\d+)(?:_(?P<variant>\w+))?$")
MAP_DIR_RE = re.compile(r"(?:^|[_-])map0*(?P<map>\d+)(?:_h(?P<h>\d+))?$|^0*(?P<bare>\d+)$")
DECODER_A_RE = re.compile(r"^heldout_A_([A-Za-z][A-Za-z0-9]*)$")
ADAPTER_ROW_RE = re.compile(r"^adapt(?P<step>\d+)_")
ROW_ORDER = ("unet", "unet200k_ema", "unet200k_ema_tuned", "pixart", "pixart200k_ema", "pixart200k_ema_tuned",
             "sd35", "sd35_ema", "sd35_170000", "adapt4000_live", "adapt4000_live_tuned")
RECIPE_ORDER = ("lr3e4", "lr5e4", "g8k")
RECIPE_LABELS = {"lr3e4": "lr 3e-4", "lr5e4": "lr 5e-4", "g8k": "8k grid", "": "base"}

# Printed sizes (in) on the 5.5 in single-column page (FIGURE_STANDARDS section 4); `\figslot` includes at scale 1.
CURVES_SIZE = (3.3, 1.6)
FIG3_SIZE = CURVES_SIZE          # the curves panel's name before the renumbering; its PDF is drawn at this size
HALF_SIZE = (2.7, 1.6)           # an appendix half-width panel (ladder, profiles, endpoint against skill)
WIDE_SIZE = (fs.TEXT_WIDTH, 1.6)  # an appendix full-width pair (seeds, recipe)
ZERO_SHOT_SIZE = (fs.TEXT_WIDTH / 2, 1.8)
ADAPTER = fs.BACKBONES["adapter"]


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

    def __init__(self, name, path, by_step, arena, k, seed, rank, variant, grid=None):
        self.name, self.path, self.by_step = name, path, by_step
        self.arena, self.k, self.seed, self.rank, self.variant = arena, k, seed, rank, variant
        self.run_dir = os.path.dirname(os.path.abspath(path))
        self.steps = sorted(by_step)
        self.grid = sorted(grid) if grid else self.steps

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
        grid = [int(s) for s in rows[0].get("grid") or [] if finite(s)]
        runs.append(Run(name, p, by_step, arena, int(rows[0].get("adapt_episodes_k") or meta["k"]),
                        int(rows[0].get("seed", meta["seed"])), meta["rank"], meta["variant"], grid))
    return runs


def classify(runs, notes, headline_variant=""):
    """(headline runs by arena, other seeds, the episode ladder, recipe tests, anchors).

    The anchors are every variant's seed-0, eight-episode, rank-16 run keyed by (variant, arena): a seed or ladder
    run is compared with its own variant's anchor, a recipe test with the base recipe's ("" variant).
    """
    base, seeds, ladder, recipe, anchors = {}, [], [], [], {}
    for r in runs:
        if r.rank != BASE_RANK:
            r.variant = f"r{r.rank}" + (f"_{r.variant}" if r.variant else "")
            recipe.append(r)
            continue
        if r.k == BASE_K and r.seed == BASE_SEED:
            if (r.variant, r.arena) in anchors:
                raise SystemExit(f"two runs for arena {r.arena} variant {r.variant!r}: "
                                 f"{anchors[(r.variant, r.arena)].name} and {r.name}")
            anchors[(r.variant, r.arena)] = r
            if r.variant == headline_variant:
                base[r.arena] = r
            elif r.variant:
                recipe.append(r)
        elif r.k != BASE_K and r.seed == BASE_SEED:
            ladder.append(r)
        elif r.k != BASE_K:
            notes.append(f"{r.name}: a ladder run at seed {r.seed}; not drawn")
        else:
            seeds.append(r)
    return base, seeds, ladder, recipe, anchors


def headline_budget(base):
    """The headline runs' last planned grid step (the largest shared by all of them), or the fallback."""
    ends = [max(r.grid) for r in base.values() if r.grid]
    return min(ends) if ends else FALLBACK_BUDGET


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


def home_interval(home_json, decoder, draws, seed):
    """The training maps' A with its episode-bootstrap interval, from the per-window file beside `home_json`."""
    if not home_json:
        return None
    pw = os.path.join(os.path.dirname(home_json), "per_window.csv")
    if not os.path.exists(pw):
        return None
    eps, vals = window_outcome(read_windows(pw), "A", decoder)
    ci = episode_bootstrap(eps, vals, draws, seed)
    return {"ci": ci, "episodes": len(set(eps)), "windows": len(vals), "source": rel(pw)} if ci else None


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
# the cost rule and the per-arena statistics
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


def gap_share(a0, a_budget, home):
    """The share of the zero-shot gap to home closed at the budget, (A_budget - A0) / (home - A0)."""
    if a0 is None or a_budget is None or home - a0 <= 0:
        return None
    return (a_budget - a0) / (home - a0)


def run_record(run, decoder, home, budget, D, draws, seed, notes, fixed=FIXED_THRESHOLD):
    """Every per-arena number of one run under one decoder, or None when its step-0 A is missing."""
    colA = f"heldout_A_{decoder}"
    A = dict(run.series(colA))
    if 0 not in A:
        notes.append(f"{run.name}: no step-0 {colA}; left out")
        return None
    a0, rows = A[0], run.rows()
    line = half_gap_line(a0, home)
    half, homex = crossing(rows, colA, line, budget), crossing(rows, colA, home, budget)
    fixedx = crossing(rows, colA, fixed, budget)
    full = crossing(rows, colA, line, None)
    S = dict(run.series("heldout_latent_skill"))
    tm = dict(run.series(f"trainmap_A_{decoder}"))
    rec = {"arena": run.arena, "run": run.name, "seed": run.seed, "k": run.k, "variant": run.variant,
           "D": D.get(run.arena), "A0": a0, "A": {str(s): v for s, v in sorted(A.items())}, "A_budget": A.get(budget),
           "gain": A[budget] - a0 if budget in A else None, "half_gap_line": line,
           "gap_share_budget": gap_share(a0, A.get(budget), home),
           "cost_half_gap": half["step"], "censored_half_gap": half["censored"], "incomplete": half["incomplete"],
           "cost_home": homex["step"], "censored_home": homex["censored"],
           "cost_fixed": fixedx["step"], "censored_fixed": fixedx["censored"],
           "cost_half_gap_full_curve": full["step"], "last_step": run.steps[-1], "A_last": A.get(max(A)),
           "auc": auc_mean(sorted(A.items()), budget),
           "S0": S.get(0), "S_budget": S.get(budget), "S": {str(s): v for s, v in sorted(S.items())},
           "lpips0": lpips_value(run, 0, decoder), "lpips_budget": lpips_value(run, budget, decoder),
           "M0": run.value(0, f"heldout_B_{decoder}"), "M_budget": run.value(budget, f"heldout_B_{decoder}"),
           "M_budget_ci": None, "G_budget": run.value(budget, f"heldout_C_{decoder}"),
           "forgetting_budget": tm[budget] - tm[0] if 0 in tm and budget in tm else None,
           "directional_budget": run.value(budget, "directional_correct_frac"),
           "trainmap_S0": run.value(0, "trainmap_latent_skill"),
           "A0_ci": None, "A_budget_ci": None, "M0_ci": None, "A_ci": {},
           "other_decoders": {d: {"A0": run.value(0, f"heldout_A_{d}"), "A_budget": run.value(budget, f"heldout_A_{d}")}
                              for d in run.decoders() if d != decoder}}
    if draws:
        for step in sorted(A):
            ci = checked_interval(run, step, "A", decoder, colA, draws, seed, notes)
            if ci is not None:
                rec["A_ci"][str(step)] = ci
        rec["A0_ci"], rec["A_budget_ci"] = rec["A_ci"].get("0"), rec["A_ci"].get(str(budget))
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
            "fixed_by_budget": sum(1 for r in done if r["cost_fixed"] is not None),
            "above_home_at_budget": sum(1 for r in done if r["A_budget"] is not None and r["A_budget"] >= home),
            "margin_lower_by_budget": sum(1 for m in margins if m < 0) if margins else None,
            "margin_within_001_by_budget": sum(1 for m in margins if 0 <= m <= 0.01) if margins else None,
            "crossed_arenas": sorted(r["arena"] for r in crossed),
            "censored_arenas": sorted(r["arena"] for r in done if r["censored_half_gap"]),
            "home_crossed_arenas": sorted(r["arena"] for r in done if r["cost_home"] is not None),
            "incomplete_arenas": sorted(r["arena"] for r in records if r["incomplete"]),
        },
        "medians": {
            **{k: median_or_none([r[k] for r in done]) for k in
               ("A0", "A_budget", "gain", "gap_share_budget", "half_gap_line", "auc", "S0", "S_budget", "lpips0",
                "lpips_budget", "M_budget", "G_budget", "forgetting_budget", "directional_budget")},
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


def common_step(records, budget):
    """The largest step every record has scored, at most `budget`."""
    shared = set.intersection(*[{int(s) for s in r["A"]} for r in records]) if records else set()
    shared = [s for s in shared if s <= budget]
    return max(shared) if shared else None


def seed_entries(anchors, seeds, records_of, budget, variant):
    """Per arena of `variant` with more than one scored seed: A per seed and step, the spread and each seed's cost
    at the largest step all its seeds reached."""
    out = []
    for arena in sorted({r.arena for r in seeds if r.variant == variant}):
        anchor = anchors.get((variant, arena))
        if anchor is None or records_of.get(anchor.name) is None:
            continue
        recs = [records_of[anchor.name]] + [records_of[r.name] for r in seeds
                                            if r.arena == arena and r.variant == variant]
        recs = [r for r in recs if r is not None]
        if len(recs) < 2:
            continue
        at = common_step(recs, budget)
        vals = {str(r["seed"]): r["A"].get(str(at)) for r in recs}
        finite_vals = [v for v in vals.values() if v is not None]
        s0 = recs[0]
        diff = {str(r["seed"]): {step: r["A"][step] - s0["A"][step] for step in r["A"] if step in s0["A"]}
                for r in recs[1:]}
        out.append({"arena": arena, "D": s0["D"], "seeds": [r["seed"] for r in recs], "at_step": at, "A_budget": vals,
                    "spread_budget": max(finite_vals) - min(finite_vals) if len(finite_vals) > 1 else None,
                    "cost_half_gap": {str(r["seed"]): r["cost_half_gap"] for r in recs},
                    "A": {str(r["seed"]): r["A"] for r in recs}, "diff_from_seed0": diff})
    return out


def ladder_entries(anchors, ladder, records_of, budget):
    """{arena: {k: A}} for the arenas with ladder runs, the eight-episode anchor included, at the largest step all
    the arena's rungs reached (at most the budget); arenas with fewer than two scored rungs are left out."""
    out = {}
    for arena in sorted({r.arena for r in ladder}):
        rungs = [r for r in ladder if r.arena == arena]
        anchor = anchors.get((rungs[0].variant, arena))
        runs = rungs + ([anchor] if anchor else [])
        recs = [(r.k, records_of.get(r.name)) for r in sorted(runs, key=lambda r: r.k) if records_of.get(r.name)]
        if len(recs) < 2:
            continue
        at = common_step([rec for _, rec in recs], budget)
        out[str(arena)] = {str(k): rec["A"].get(str(at)) for k, rec in recs}
    return out


def recipe_entries(anchors, base, recipe, records_of, seeds_summary):
    """Each recipe test against the base recipe's run of its arena (the headline run when no base run exists)."""
    spread = {e["arena"]: e["spread_budget"] for e in seeds_summary}
    out = []
    for r in sorted(recipe, key=lambda r: (r.arena, recipe_rank(r.variant))):
        rec = records_of.get(r.name)
        anchor = anchors.get(("", r.arena)) or base.get(r.arena)
        b = records_of.get(anchor.name) if anchor else None
        if rec is None:
            continue
        at = common_step([rec, b], math.inf) if b else None
        at_budget = rec["A_budget"] is not None and b is not None and b["A_budget"] is not None
        delta = rec["A_budget"] - b["A_budget"] if at_budget else \
            (rec["A"][str(at)] - b["A"][str(at)] if at is not None else None)
        out.append({"arena": r.arena, "variant": r.variant, "label": RECIPE_LABELS.get(r.variant, r.variant),
                    "run": r.name, "A_budget": rec["A_budget"], "delta_vs_base": delta,
                    "delta_at": None if at_budget else at, "A_last": rec["A_last"],
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
    return tabular("r" * len(head), head, body)


def perarena_table(records, budget, prov):
    """The appendix table `tab:perarena-adapt`: the columns appendix.tex declares, one row per arena."""
    b = budget_label(budget)
    head = ["Arena", "$D$", "$A_0$", "Half-gap line", "Half gap", "Training-maps line", f"$A$ at {b}",
            f"$M$ at {b}",
            "Forgetting", "Directional"]
    body = [[str(r["arena"]), num(r["D"], 3, prov=prov), num(r["A0"], 2, True, prov),
             num(r["half_gap_line"], 2, prov=prov),
             cost_cell(r["cost_half_gap"], r["censored_half_gap"], r["incomplete"], budget, prov),
             cost_cell(r["cost_home"], r["censored_home"], r["incomplete"], budget, prov),
             num(r["A_budget"], 2, True, prov), num(r["M_budget"], 3, True, prov),
             num(r["forgetting_budget"], 2, True, prov), num(r["directional_budget"], 2, prov=prov)]
            for r in records]
    return tabular("r" * len(head), head, body)


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------

def a_axis(ax, lo_data, hi_data, label="$A$ (dB)", step=1.0):
    """The A axis: from 0 (copy-last) to past the data, round 1 dB ticks."""
    top = max(hi_data, 0.0)
    ax.set_ylim(min(0.0, lo_data) - 0.15, top + 0.35)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.set_ylabel(label)


def fig_curves(records, runs, home, budget, out_dir, column="A", band=None):
    """A (or S) against updates, one line per arena in the adapter's blue ramp keyed to the zero-shot skill S0
    (light: high S0), the named arenas labelled at their right ends, the training maps' line and copy-last."""
    s0 = [r["S0"] for r in records if r["S0"] is not None]
    lo, hi = (min(s0), max(s0)) if s0 else (0.0, 1.0)
    fig, (ax,) = fs.new_figure(CURVES_SIZE)
    steps = sorted({s for r in records for s in runs[r["run"]].steps if s <= budget})
    z = step_axis(ax, steps)
    skill = column == "S"
    ends = []
    for r in sorted(records, key=lambda r: -(r["S0"] or 0)):
        pts = runs[r["run"]].series("heldout_latent_skill" if skill else f"heldout_A_{r['decoder']}", budget)
        c = fs.ramp_colour(r["S0"], lo, hi)
        ax.plot(xs_of(pts, z), [v for _, v in pts], color=c, lw=0.8, zorder=3, solid_capstyle="round")
        if r["arena"] in NAMED_ARENAS and pts:
            ends.append((pts[-1][1], r["arena"]))
    ref = median_or_none([r["trainmap_S0"] for r in records]) if skill else home
    # a label within 0.15 dB of the training maps' dashed line moves above it
    ends = [(v if ref is None or abs(v - ref) >= 0.15 else ref + 0.2, arena) for v, arena in ends]
    fs.end_labels(ax, [(steps[-1], v, str(arena)) for v, arena in ends], gap=0.28)
    if ref is not None:
        fs.training_line(ax, ref, band=None if skill else band)
    fs.copy_last_line(ax, where=0.0, align="left")
    vals = [v for r in records for _, v in runs[r["run"]].series(
        "heldout_latent_skill" if skill else f"heldout_A_{r['decoder']}", budget)]
    a_axis(ax, min(vals), max(vals + ([ref] if ref is not None else [])), "$S$ (dB)" if skill else "$A$ (dB)")
    return fs.save(fig, out_dir, "fig3_adaptation_skill" if skill else "fig3_adaptation_curves")


def arena_line_style(i):
    return ("-", (0, (3, 1.5)), (0, (1, 1)), (0, (4, 1, 1, 1)))[i % 4]


def fig_seeds(entries, anchors, seeds, decoder, budget, variant, out_dir, spread=None):
    """Appendix: (a) each seed's curve per arena (dark grey, seed 0 solid, seed 1 dashed, arenas labelled at the
    right ends); (b) every other seed's difference from seed 0 per step as points, the zero line and a grey band at
    the largest spread at the budget."""
    fig, (ax, ax2) = fs.new_figure(WIDE_SIZE, ncols=2, wspace=0.08)
    drawn = {e["arena"] for e in entries}
    runs = [anchors[(variant, a)] for a in drawn] + [r for r in seeds if r.variant == variant and r.arena in drawn]
    steps = sorted({s for r in runs for s in r.steps if s <= budget})
    z, z2 = step_axis(ax, steps), step_axis(ax2, steps)
    ends = []
    for e in entries:
        for i, seed in enumerate(e["seeds"]):
            run = anchors[(variant, e["arena"])] if seed == BASE_SEED else \
                next(r for r in seeds if r.arena == e["arena"] and r.seed == seed and r.variant == variant)
            pts = run.series(f"heldout_A_{decoder}", budget)
            ax.plot(xs_of(pts, z), [v for _, v in pts], color=fs.CONTEXT_INK, lw=0.8, ls=arena_line_style(i))
            if i == 0:
                ends.append((xs_of(pts, z)[-1], pts[-1][1], str(e["arena"])))
        for _seed, dd in sorted(e["diff_from_seed0"].items()):
            pts = sorted((int(s), v) for s, v in dd.items() if int(s) <= budget and int(s) > 0)
            ax2.plot(xs_of(pts, z2), [v for _, v in pts], ls="none", marker="o", ms=2.5, mfc=fs.CONTEXT_INK,
                     mec="white", mew=fs.MARKER_EDGE)
    fs.end_labels(ax, ends, gap=0.16)
    width = spread if spread is not None else max((e["spread_budget"] or 0) for e in entries)
    if width:
        for w in (-width, width):
            ax2.axhline(w, color=fs.CONTEXT_INK, lw=fs.MIN_LW, ls=(0, (1, 1.2)), zorder=1, gid="ref")
        ax2.text(1.0, width, f"seed spread at {budget_label(budget)}", transform=ax2.get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=fs.ANNOT_PT, color=fs.CONTEXT_INK)
    ax2.axhline(0, color=fs.BLACK, lw=fs.REF_LW, gid="ref")
    ax.set_ylabel("$A$ (dB)")
    ax2.set_ylabel("$A$, seed 1 $-$ seed 0 (dB)")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(4))
    fs.panel_letter(ax, "a")
    fs.panel_letter(ax2, "b")
    return fs.save(fig, out_dir, "fig3_adaptation_seeds")


def fig_ladder(ladder, records, anchors_step0, home, out_dir, band=None):
    """Appendix: A against adaptation episodes (log scale) per arena, dark grey with the arena labelled at its
    right end; each arena's step-0 A dotted and labelled 'step 0' once; the training maps' line."""
    fig, (ax,) = fs.new_figure(HALF_SIZE)
    ks = sorted({int(k) for v in ladder.values() for k in v})
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(ticker.FixedLocator(ks))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(k) for k in ks]))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    labelled = False
    vals, ends = [], []
    for i, (arena, byk) in enumerate(sorted(ladder.items(), key=lambda kv: int(kv[0]))):
        pts = sorted((int(k), v) for k, v in byk.items() if v is not None)
        vals += [v for _, v in pts]
        ax.plot([k for k, _ in pts], [v for _, v in pts], color=fs.CONTEXT_INK, lw=0.8, ls=arena_line_style(i),
                marker="o", ms=2.5, mfc=fs.CONTEXT_INK, mec="white", mew=fs.MARKER_EDGE)
        ends.append((pts[-1][0], pts[-1][1], str(arena)))
        a0 = anchors_step0.get(int(arena))
        if a0 is not None:
            vals.append(a0)
            ax.plot([ks[0], ks[-1]], [a0, a0], color=fs.FAINT, lw=fs.MIN_LW, ls=(0, (1, 1.5)), zorder=1, gid="ref")
            if not labelled:
                ax.text(ks[0], a0, "step 0", ha="left", va="bottom", fontsize=fs.ANNOT_PT, color=fs.CONTEXT_INK)
                labelled = True
    fs.end_labels(ax, ends, gap=0.16)
    fs.training_line(ax, home, where=0.0, band=band)
    ax.set_xlim(ks[0] / 1.3, ks[-1] * 1.6)
    ax.set_xlabel("adaptation episodes (log scale)")
    a_axis(ax, min(vals + [0.0]), max(vals + [home]))
    ax.set_ylim(bottom=min(vals) - 0.4)
    return fs.save(fig, out_dir, "fig_adapt_ladder")


def fig_recipe(recipe, anchors, base, seeds, records_of, home, decoder, out_dir, band=None):
    """Appendix: per arena (panels a, b, ...) the base recipe (black), its seed-1 run (light grey) and each recipe
    test (dark grey, one line style each), sharing one A axis; the arena's half-gap line (dotted) and the training
    maps' line; one frameless legend, since the curves overlap too closely for end labels."""
    arenas = sorted({r.arena for r in recipe if (anchors.get(("", r.arena)) or base.get(r.arena))
                     and records_of.get(r.name)})
    fig, axes = fs.new_figure(WIDE_SIZE, ncols=max(1, len(arenas)), sharey=True, wspace=0.06)
    styles = {"lr3e4": (0, (3, 1.5)), "lr5e4": (0, (4, 1, 1, 1)), "g8k": (0, (7, 2))}
    for i, (ax, arena) in enumerate(zip(axes, arenas)):
        b = anchors.get(("", arena)) or base[arena]
        brec = records_of[b.name]
        lines = [(b, "base", fs.INK, "-", fs.DATA_LW)]
        lines += [(r, f"seed {r.seed}", fs.FAINT, "-", fs.DATA_LW) for r in seeds
                  if r.arena == arena and r.variant == b.variant and records_of.get(r.name)]
        lines += [(r, RECIPE_LABELS.get(r.variant, r.variant), fs.CONTEXT_INK, styles.get(r.variant, (0, (2, 2))),
                   0.8) for r in sorted((r for r in recipe if r.arena == arena and records_of.get(r.name)),
                                        key=lambda r: recipe_rank(r.variant))]
        steps = sorted({s for r, *_ in lines for s in r.steps})
        z = step_axis(ax, steps)
        for r, label, c, ls, lw in lines:
            pts = r.series(f"heldout_A_{decoder}")
            ax.plot(xs_of(pts, z), [v for _, v in pts], color=c, lw=lw, ls=ls, zorder=3, label=label)
        ax.axhline(brec["half_gap_line"], color=fs.CONTEXT_INK, lw=fs.MIN_LW, ls=(0, (1, 1.2)), gid="ref")
        if i == len(arenas) - 1:
            ax.text(0.0, brec["half_gap_line"], " half-gap line", transform=ax.get_yaxis_transform(), ha="left",
                    va="bottom", fontsize=fs.ANNOT_PT, color=fs.CONTEXT_INK)
        fs.training_line(ax, home, where=0.0, band=band)
        ax.text(0.98, 0.04, f"arena {arena}", transform=ax.transAxes, ha="right", va="bottom", fontsize=fs.ANNOT_PT)
        ax.set_ylabel("$A$ (dB)" if i == 0 else "")
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        fs.panel_letter(ax, "abcdefgh"[i])
    handles = {}
    for ax in axes:
        for h, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, h)
    order = ["base"] + sorted((k for k in handles if k.startswith("seed")), key=str) + \
        [k for k in handles if k != "base" and not k.startswith("seed")]
    axes[0].legend([handles[k] for k in order], order, loc="upper left", bbox_to_anchor=(0.0, 0.9), ncol=3,
                   handlelength=2.4, columnspacing=1.0)
    return fs.save(fig, out_dir, "fig_adapt_recipe")


# hand-placed label offsets (points) for the endpoint-against-skill scatter, where neighbours collide
SKILL_LABEL_OFFSETS = {7: (3, -6), 13: (3, 3), 11: (-3, 3), 9: (3, -4), 1: (3, 0), 12: (0, -7), 8: (3, 3)}


def fig_skill(records, home, budget, out_dir, band=None):
    """Appendix: A at the budget against the zero-shot latent skill S0, one diamond per arena, every arena
    numbered; the training maps' line."""
    done = [r for r in records if r["S0"] is not None and r["A_budget"] is not None]
    if not done:
        return []
    fig, (ax,) = fs.new_figure(HALF_SIZE)
    ax.set_gid("points")
    for r in done:
        ci = r.get("A_budget_ci")
        if ci:
            ax.plot([r["S0"], r["S0"]], ci, color=ADAPTER.colour, lw=fs.MIN_LW, zorder=3.5)
        ax.plot(r["S0"], r["A_budget"], ls="none", marker=ADAPTER.marker, ms=3.2, mfc=ADAPTER.colour, mec="white",
                mew=fs.MARKER_EDGE, zorder=3)
        dx, dy = SKILL_LABEL_OFFSETS.get(r["arena"], (3, 0))
        fs.direct_label(ax, r["S0"], r["A_budget"], str(r["arena"]), colour=fs.CONTEXT_INK, dx=dx, dy=dy,
                        ha="left" if dx > 0 else "right" if dx < 0 else "center")
    fs.training_line(ax, home, where=0.0, band=band)
    ax.set_xlabel("zero-shot latent skill $S_0$ (dB)")
    lo = min(r["A_budget"] for r in done)
    ax.set_ylim(min(lo, home) - 0.5, max(max(r["A_budget"] for r in done), home) + 0.4)
    ax.set_ylabel(f"$A$ at {budget_label(budget)} (dB)")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    return fs.save(fig, out_dir, "fig2c_outcomes_by_skill")


def fig_zero_shot(zero_shot, order, key, out_dir, stem, ylabel, zero=False):
    """Zero-shot A (or M) per arena, arenas in `order`, training maps first under the grey band; one marker per
    backbone (encoding table), open for its stock decoder and filled for the tuned one; each backbone's home as a
    short dashed rule in its colour; intervals drawn above the markers."""
    names = [r for r in ROW_ORDER if r in zero_shot] + sorted(r for r in zero_shot if r not in ROW_ORDER)
    rows = [n for n in names if any(v.get(key) is not None for v in zero_shot[n]["maps"].values())]
    arenas = [a for a in order if any(zero_shot[r]["maps"].get(a, {}).get(key) is not None for r in rows)]
    if not arenas:
        return []
    fig, (ax,) = fs.new_figure(ZERO_SHOT_SIZE)
    pos = {a: i for i, a in enumerate(arenas)}
    home_maps = [a for a in arenas if a in TRAINING_MAPS]
    if home_maps:
        fs.training_band(ax, min(pos[a] for a in home_maps) - 0.5, max(pos[a] for a in home_maps) + 0.5)
    width = 0.6 / max(1, len(rows))
    for i, name in enumerate(rows):
        maps = zero_shot[name]["maps"]
        ent = fs.BACKBONES[fs.backbone_of(name)]
        tuned = name.endswith("_tuned")
        off = (i - (len(rows) - 1) / 2) * width
        pts = [(pos[a] + off, maps[a][key], maps[a].get(f"{key}_ci")) for a in arenas
               if a in maps and maps[a].get(key) is not None]
        for x, _v, ci in pts:
            if ci:
                ax.plot([x, x], ci, color=ent.colour, lw=0.7, zorder=4, solid_capstyle="butt")
        ax.plot([x for x, _, _ in pts], [v for _, v, _ in pts], ls="none", marker=ent.marker, ms=3.2,
                mfc=ent.colour if tuned else "white", mec=ent.colour, mew=0.7, zorder=3)
        home = zero_shot[name].get("home") if key == "A" else zero_shot[name].get("home_M")
        if home is not None:
            ax.plot([len(arenas) - 0.3, len(arenas) + 0.3], [home, home], color=ent.colour, lw=fs.REF_LW,
                    ls=fs.TRAINING_DASH, zorder=2, clip_on=False)
    if zero:
        fs.copy_last_line(ax, where=0.0, align="left")
    ax.set_xticks(range(len(arenas)), [str(a) for a in arenas])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(arenas) + 0.4)
    ax.set_xlabel("arena, by zero-shot skill $S_0$")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0 if key == "A" else 0.05))
    return fs.save(fig, out_dir, stem)


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
    names = {name for name, _ in dirs}
    for name, path in dirs:
        if a.decoder == STOCK and name.endswith("_tuned"):
            continue
        if a.decoder != STOCK and (name + "_" + a.decoder) in names:
            continue  # its rescored twin carries this decoder's columns
        if name.startswith("home_") or name == "adapters":
            continue
        # SD 3.5 renders through its own 16-channel decoder, so its row keeps the stock columns in every pass
        maps = load_zero_shot_row(path, STOCK if name.startswith("sd35") else a.decoder, draws, a.seed)
        if not maps:
            notes.append(f"{rel(path)}: no one-tic eval_tf reads keyed by map; row {name} skipped")
            continue
        own = pooled_home(maps)
        out[name] = {"source": rel(path), "home": own if own is not None else (home if name == "unet" else None),
                     "home_M": median_or_none([maps[m]["M"] for m in TRAINING_MAPS if m in maps]),
                     "maps": maps}
    if any(n.startswith("unet200k") for n in out) and "unet" in out:
        fresh = next(n for n in out if n.startswith("unet200k"))
        out[fresh]["home"] = out[fresh]["home"] if out[fresh]["home"] is not None else out["unet"]["home"]
        del out["unet"]  # the fresh read is the same model on the same windows; keep one marker per arena
    return out


def margins_from_adapter_row(records, zero_shot, budget, notes):
    """Fill each record's M at the budget from the adapter's zero-shot row at that step when the rows lack it."""
    if any(r["M_budget"] is not None for r in records):
        return
    for name, row in zero_shot.items():
        m = ADAPTER_ROW_RE.match(name)
        if not m or int(m["step"]) != budget:
            continue
        filled = 0
        for r in records:
            e = row["maps"].get(r["arena"])
            if e and e.get("M") is not None:
                r["M_budget"], r["M_budget_ci"] = e["M"], e.get("M_ci")
                filled += 1
        if filled:
            notes.append(f"M at {budget}: the adapter rows carry no heldout_B; read from {row['source']} "
                         f"({filled} arenas)")
        return


def without_paths(maps):
    """A zero-shot row's per-map entries without their local file paths, keyed by map as strings."""
    return {str(m): {x: y for x, y in e.items() if x != "path"} for m, e in maps.items()}


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
    p.add_argument("--distances", default=os.path.join(REPO, "results", "distance_v2", "distances_sd1.json"),
                   help="distances_<space>.json; each arena's primary D, reported in the tables")
    p.add_argument("--decoder", default=STOCK, help="stock or tuned (any decoder the rows carry as heldout_A_<name>)")
    p.add_argument("--home", type=float, default=None,
                   help=f"the training maps' scene A for --decoder (default {STOCK_HOME} for the stock decoder)")
    p.add_argument("--home-json", default=None,
                   help="an eval_tf metrics.json of the training maps; home = scene_psnr_dec - scene_copy_psnr_dec")
    p.add_argument("--weights", default="live", choices=("live", "ema"))
    p.add_argument("--headline-variant", default="",
                   help="the run-name variant whose seed-0, 8-episode runs are the headline set (e.g. g8k)")
    p.add_argument("--budget", type=int, default=None,
                   help="the fixed budget: A at it, censoring beyond it (default: the headline grid's last step)")
    p.add_argument("--fresh-root", default=os.path.join(REPO, "results", "fresh_rescore"),
                   help="<root>/<row>/<map dir>/metrics.json zero-shot reads; each row a marker per arena")
    p.add_argument("--zero-shot", action="append", default=[], metavar="NAME=DIR",
                   help="one more zero-shot row: a directory of eval_tf reads keyed by map (repeatable)")
    p.add_argument("--with-raw", action="store_true", help="also draw the margin M per arena, where rows carry it")
    p.add_argument("--prov", action="store_true", help="wrap every table number in \\prov{} (provisional marks)")
    p.add_argument("--bootstrap", type=int, default=2000, help="episode-bootstrap draws per arena interval (0: none)")
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
    base, seeds, ladder, recipe, anchors = classify(runs, notes, a.headline_variant)
    if not base:
        raise SystemExit(f"no headline runs (rank 16, k 8, seed 0, variant {a.headline_variant!r}) among "
                         + a.runs_glob)
    budget = a.budget if a.budget is not None else headline_budget(base)
    for arena in sorted(base):
        if arena not in D:
            notes.append(f"arena {arena}: no primary D in {rel(a.distances)}")
    records_of = {}
    for run in runs:
        draws = a.bootstrap if run is base.get(run.arena) else 0
        rec = run_record(run, a.decoder, home, budget, D, draws, a.seed, notes, FIXED_THRESHOLD)
        if rec is not None:
            rec["decoder"] = a.decoder
        records_of[run.name] = rec
    # tables list arenas by D, as the appendix table always has; figures order by the zero-shot skill S0
    records = sorted((records_of[r.name] for r in base.values() if records_of[r.name]),
                     key=lambda r: (r["D"] is None, r["D"] or 0, r["arena"]))
    if not records:
        raise SystemExit(f"no headline run carries heldout_A_{a.decoder} at step 0; notes: " + "; ".join(notes))
    by_name = {r.name: r for r in runs}
    for r in records:
        if r["incomplete"]:
            notes.append(f"arena {r['arena']}: scored to step {r['last_step']} of {budget}; left out of the counts")

    zero_shot = zero_shot_rows(a, records, home, a.bootstrap, notes)
    margins_from_adapter_row(records, zero_shot, budget, notes)
    summary = build_summary(records, budget, home)
    seed_summary = seed_entries(anchors, seeds, records_of, budget, a.headline_variant)
    ladder_summary = ladder_entries(anchors, ladder, records_of, budget)
    home_ci = home_interval(a.home_json, a.decoder, max(a.bootstrap, 10000), a.seed)
    when = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    used = [base[a_].path for a_ in sorted(base)]
    inputs = [{"path": rel(p), "sha256": sha256(p)} for p in sorted({r.path for r in runs}) + [a.distances]]
    summary = {"generated_by": "paper/make_adapt_figures.py", "generated_at": when, "decoder": a.decoder,
               "home": home, "home_ci": home_ci,
               "home_source": rel(home_source) if os.path.exists(home_source) else home_source,
               "budget": budget, "headline_variant": a.headline_variant, "weights": a.weights, **summary,
               "per_arena": records, "seeds": seed_summary, "ladder": ladder_summary,
               "recipe": recipe_entries(anchors, base, recipe, records_of, seed_summary),
               "zero_shot": {k: {**v, "maps": without_paths(v["maps"])} for k, v in zero_shot.items()},
               "inputs": inputs, "notes": notes}

    os.makedirs(a.tables_dir, exist_ok=True)
    head = header([a.runs_glob, a.distances], when, [
        f"{len(used)} headline runs (rank 16, 8 episodes, seed 0, variant {a.headline_variant or 'base'}), "
        f"{a.weights} weights, decoder {a.decoder}, "
        f"training maps (in-distribution) A {home:.3f} dB "
        f"({rel(home_source) if os.path.exists(home_source) else home_source}), budget {budget}",
        "half-gap line = (A0 + training maps' A) / 2; the training maps' A is an in-distribution reference, not the "
        "arena's own ceiling",
        "\\tbd marks a quantity these rows do not carry yet; costs are first grid crossings, >budget right-censored",
    ] + [f"  {rel(p)} sha256 {sha256(p)[:16]}" for p in used])
    written = []
    tables = [("adapt_cost.tex", cost_table(records, summary, a.decoder, budget, a.prov)),
              ("adapt_perarena.tex", perarena_table(records, budget, a.prov))]
    for name, body in tables:
        path = os.path.join(a.tables_dir, name)
        with open(path, "w") as f:
            f.write(head + body)
        written.append(path)
    path = os.path.join(a.tables_dir, "adapt_summary.json")
    with open(path, "w") as f:
        json.dump(jsonable(summary), f, indent=1)
        f.write("\n")
    written.append(path)

    fs.style()
    written += draw_figures(a, records, by_name, home, budget, seed_summary, anchors, seeds, ladder_summary,
                            recipe, base, records_of, zero_shot, notes, band=(home_ci or {}).get("ci"))
    for p in written:
        print("wrote", rel(p))
    for n in notes:
        print("note:", n)
    return 0


def draw_figures(a, records, by_name, home, budget, seed_summary, anchors, seeds, ladder_summary, recipe,
                 base, records_of, zero_shot, notes, band=None):
    """Every figure the data support; a figure the data cannot fill is refused by `figstyle.save` and noted.
    `band` is the training maps' 95% episode interval, drawn around their dashed reference line."""
    written = []

    def attempt(fn, *args, **kw):
        try:
            return fn(*args, **kw)
        except fs.DegenerateFigure as e:
            notes.append(f"not drawn: {e}")
            return []

    written += attempt(fig_curves, records, by_name, home, budget, a.out_dir, "A", band=band)
    written += attempt(fig_curves, records, by_name, home, budget, a.out_dir, "S")
    written += attempt(fig_skill, records, home, budget, a.out_dir, band=band)
    if seed_summary:
        written += attempt(fig_seeds, seed_summary, anchors, seeds, a.decoder, budget, a.headline_variant, a.out_dir)
    if ladder_summary:
        step0 = {r["arena"]: r["A0"] for r in records}
        written += attempt(fig_ladder, ladder_summary, records, step0, home, a.out_dir, band=band)
    if [r for r in recipe if records_of.get(r.name)]:
        written += attempt(fig_recipe, recipe, anchors, base, seeds, records_of, home, a.decoder, a.out_dir,
                           band=band)
    if zero_shot:
        order = [r["arena"] for r in sorted(records, key=lambda r: (r["S0"] is None, -(r["S0"] or 0), r["arena"]))]
        order = [m for m in TRAINING_MAPS if any(m in v["maps"] for v in zero_shot.values())] + order
        written += attempt(fig_zero_shot, zero_shot, order, "A", a.out_dir, "fig2a_advantage_by_distance",
                           "$A$ (dB)")
        if a.with_raw and any(e.get("M") is not None for v in zero_shot.values() for e in v["maps"].values()):
            written += attempt(fig_zero_shot, zero_shot, order, "M", a.out_dir, "fig2b_margin_by_distance",
                               "$M$, LPIPS difference (lower is better)", zero=True)
    return written


if __name__ == "__main__":
    sys.exit(main())
