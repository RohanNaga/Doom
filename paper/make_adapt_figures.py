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
G = `heldout_C_<decoder>` (likewise from the adapter's raw-frame read `adapt<budget>_...` when the rows lack it);
S = `heldout_latent_skill` (no decoder); LPIPS = `heldout_lpips_dec`; forgetting = the
change of `trainmap_A_<decoder>` from step 0; directional = `directional_correct_frac`. Per-arena intervals are
95% percentile bootstraps over held-out episodes from the per-window files the rows name (found beside the run when
the recorded path is a server path), with the windows `eval_tf.py` flags as duplicates left out as
`score_adapt.py` leaves them out; each file's mean is checked against its row and a disagreement is noted.

**The cost rule** (`.claude/analyses/cost-target-decision-2026-09-26.md`, `lora-results-decision-2026-09-27.md`).
Per arena the half-gap line is (A0 + home) / 2, where home is the training maps' A on the same crop and decoder:
`--home` (default 4.138 dB, the stock decoder's scene home), or `--home-json`, an eval_tf read of the training
maps, as `scene_psnr_dec - scene_copy_psnr_dec` (duplicate-free means where the read has them); with a
`per_window.csv` beside it, home's own episode interval is reported. The cost is the first grid step at or before
the budget (`--budget`, default the headline grid's last step) whose A reaches the line; a curve scored to the
budget that never reaches it is right-censored, and one not yet scored to the budget is incomplete and left out of
every count. A median over arenas ranks a censored arena above every crossing and is censored when the middle lands
on one. Spearman correlations are scipy's (average ranks) with a censored cost ranked as twice the budget.

**Across arenas** (`.claude/analyses/per-arena-statistics-decision-2026-09-27.md`). The arena is the unit. Every
aggregate carries a 95% percentile interval from a nested bootstrap (`--arena-bootstrap` draws): arenas resampled
with replacement, then each drawn arena's held-out episodes, paired across steps, with each draw's half-gap line
recomputed from its own A0 and home frozen; an arena whose per-window files are missing enters with its point
curve (noted). Reported: the interquartile mean (`scipy.stats.trim_mean(x, 0.25)`) and median of A per step; the
cumulative attainment of the half-gap line, one minus Kaplan-Meier, which equals the fraction crossed because
every censoring falls at the last grid step, with the at-risk counts; the same for the fixed threshold
`--fixed-threshold` (3.5 dB); the share of the gap closed at the budget, (A_budget - A0) / (home - A0); the median
budget with its censored share; the margin M at the budget (lower than persistence, within 0.01); Spearman S0 with
A at the budget; the performance profiles P(A >= tau) at 0, 500 and the budget.

**Outputs.** In `--out-dir`, PDF and 600 dpi PNG through `figstyle.save` (drawn at printed size for the 5.5 in
page, refused when degenerate): `fig4_adaptation` (Figure 4: (a) the interquartile-mean learning curve over the
faint per-arena curves, (b) the attainment curves with the at-risk row and per-arena crossing ticks),
`figA_adapt_arenas` (the per-arena small multiples with episode bands), `figA_adapt_profiles`,
`fig3_adaptation_curves` (A per arena, the blue ramp keyed to S0, three arenas labelled), `fig3_adaptation_skill`
(S instead of A), `fig3_adaptation_seeds`, `fig_adapt_ladder` and `fig_adapt_recipe` (when their runs carry the
decoder), `fig2c_outcomes_by_skill` (A at the budget against S0), `fig3b_zero_shot_paired` (Figure 3b: zero-shot
A and M per arena in S0 order for every backbone under `--fresh-root`, stock decoder open and tuned filled on the
same windows, the training maps' reads in a first column, the 4k adapter on the A row), and the appendix variants
`fig2a_advantage_by_distance` and, with `--with-raw`, `fig2b_margin_by_distance` (one decoder per build, arenas in
S0 order despite the historical names). In `--tables-dir`, under a
provenance header: `adapt_cost.tex` (per arena), `adapt_perarena_A.tex` (the A-based per-arena table; the
appendix's raw table is `paper/make_raw_figures.py`'s `adapt_perarena.tex`), `adapt_table3.tex`
(Table 3's tabular with arena-bootstrap intervals in brackets), and `adapt_summary.json`, the numbers the text
quotes.
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
import matplotlib.patches  # noqa: E402
from figstyle import step_axis, step_label, xs_of  # noqa: E402
from matplotlib import ticker, transforms  # noqa: E402
from matplotlib.path import Path  # noqa: E402

STOCK_HOME = 4.138        # training maps' scene A, stock decoder, 512 validation windows (lora-results decision)
TRAINING_MAPS = (2, 3, 4, 5)
BASE_RANK, BASE_K, BASE_SEED = 16, 8, 0
FALLBACK_BUDGET = 4000
FIXED_THRESHOLD = 3.5     # dB; the second attainment curve (statistics decision, 2026-09-27)
PROFILE_STEPS = (0, 500)  # performance profiles at these steps and at the budget
NAMED_ARENAS = (7, 9, 12)  # the far arena, the one at the training maps' line, the censored one (standard, Fig. 4)
COMPARATOR_ARENA = 7      # Table 3's single-arena column (the full fine-tune's arena)
LORA_PARAMS, FULL_PARAMS = "4.2M (0.49\\%)", "860M (all)"   # lora.parameter_counts (RESEARCH_CONTEXT 09-26 20:10)
RUN_RE = re.compile(r"_map(?P<map>\d+)_r(?P<rank>\d+)_k(?P<k>\d+)_s(?P<seed>\d+)(?:_(?P<variant>\w+))?$")
MAP_DIR_RE = re.compile(r"(?:^|[_-])map0*(?P<map>\d+)(?:_h(?P<h>\d+))?$|^0*(?P<bare>\d+)$")
DECODER_A_RE = re.compile(r"^heldout_A_([A-Za-z][A-Za-z0-9]*)$")
ADAPTER_ROW_RE = re.compile(r"^adapt(?P<step>\d+)_")
ROW_ORDER = ("unet", "unet200k_ema", "unet200k_ema_tuned", "pixart", "pixart200k_ema", "pixart200k_ema_tuned",
             "sd35", "sd35_ema", "sd35_170000", "adapt4000_live", "adapt4000_live_tuned")
RECIPE_ORDER = ("lr3e4", "lr5e4", "g8k")
DECODER_NAMES = {"stock": "stock decoder", "tuned": "fine-tuned decoder"}   # printed names (Rohan, 2026-09-27)
RECIPE_LABELS = {"lr3e4": "lr 3e-4", "lr5e4": "lr 5e-4", "g8k": "8k grid", "": "base"}

# Printed sizes (in) on the 5.5 in single-column page (FIGURE_STANDARDS section 4); `\figslot` includes at scale 1.
FIG4_SIZE = (fs.TEXT_WIDTH, 1.6)
DOTS_SIZE = (fs.TEXT_WIDTH, 1.4)          # Figure 4's body slot, unscaled (paper owner, 2026-09-27)
CURVES_SIZE = (3.3, 1.6)
FIG3_SIZE = CURVES_SIZE          # the curves panel's name before the renumbering; its PDF is drawn at this size
HALF_SIZE = (2.7, 1.6)           # an appendix half-width panel (ladder, profiles, endpoint against skill)
WIDE_SIZE = (fs.TEXT_WIDTH, 1.6)  # an appendix full-width pair (seeds, recipe)
ARENAS_SIZE = (fs.TEXT_WIDTH, 3.0)
ZERO_SHOT_SIZE = (fs.TEXT_WIDTH / 2, 1.8)
ADAPTER = fs.BACKBONES["adapter"]


def row_label(name):
    """A result row's printed name: its backbone's name in the encoding table (provenance stays in the caption)."""
    return fs.BACKBONES[fs.backbone_of(name)].label


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
               "G": metrics_outcome(metrics, "C", decoder),
               "n": n.get("n") if isinstance(n, dict) else None, "A_ci": None, "M_ci": None, "path": found[0]}
        pw = os.path.join(os.path.dirname(found[0]), "per_window.csv")
        if draws and os.path.exists(pw):
            windows = read_windows(pw)
            rec["A_ci"] = episode_bootstrap(*window_outcome(windows, "A", decoder), draws, seed)
            if rec["M"] is not None:
                rec["M_ci"] = episode_bootstrap(*window_outcome(windows, "B", decoder), draws, seed)
        out[int(m["map"] or m["bare"])] = rec
    return out


def home_read(root, row, decoder, draws, seed):
    """{A, M, A_ci, M_ci} of a backbone's training-maps read `<root>/home_<row>/**/metrics.json` under `decoder`."""
    found = sorted(glob.glob(os.path.join(root, f"home_{row}", "**", "metrics.json"), recursive=True))
    if len(found) != 1:
        return None
    with open(found[0]) as f:
        metrics = json.load(f)
    out = {"A": metrics_outcome(metrics, "A", decoder), "M": metrics_outcome(metrics, "B", decoder),
           "A_ci": None, "M_ci": None}
    pw = os.path.join(os.path.dirname(found[0]), "per_window.csv")
    if draws and os.path.exists(pw):
        windows = read_windows(pw)
        out["A_ci"] = episode_bootstrap(*window_outcome(windows, "A", decoder), draws, seed)
        if out["M"] is not None:
            out["M_ci"] = episode_bootstrap(*window_outcome(windows, "B", decoder), draws, seed)
    return out if out["A"] is not None else None


def paired_zero_shot(root, draws, seed, notes):
    """{backbone: {decoder: {"maps": {arena: {A, M, A_ci, M_ci}}, "home": {...}, "source": dir}}} for both decoders.

    A row `<name>` and its rescored twin `<name>_tuned` are one backbone's reads of the same windows; the stock
    columns come from the plain directory (or the twin, which carries them too), the tuned columns from the twin.
    A backbone with more than one row name keeps the first in sorted order and notes the rest.
    """
    if not root or not os.path.isdir(root):
        return {}
    names = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith("home_")
                   and d != "adapters")
    bases = sorted({n[:-len("_tuned")] if n.endswith("_tuned") else n for n in names})
    out = {}
    for base in bases:
        key = fs.backbone_of(base)
        if key in out:
            notes.append(f"Figure 3b: {base} is a second {key} row; {out[key]['name']} is drawn")
            continue
        twin = base + "_tuned" if base + "_tuned" in names else None
        entry = {"name": base}
        for decoder, row in (("stock", base if base in names else twin), ("tuned", twin)):
            if row is None:
                continue
            maps = load_zero_shot_row(os.path.join(root, row), decoder, draws, seed)
            maps = {m: e for m, e in maps.items() if e["A"] is not None}
            if maps:
                entry[decoder] = {"maps": maps, "source": rel(os.path.join(root, row)),
                                  "home": home_read(root, row, decoder, draws, seed)
                                  or home_read(root, base, decoder, draws, seed)}
        if "stock" in entry or "tuned" in entry:
            out[key] = entry
    return out


ABSOLUTE = {"psnr": "scene_psnr_raw", "lpips": "scene_lpips_raw"}
PERSISTENCE = {"psnr": "scene_persist_psnr_raw", "lpips": "scene_persist_lpips_raw"}


def absolute_read(metrics_path, decoder, draws, seed):
    """Absolute scene PSNR and LPIPS of one eval_tf read against the raw true frame under `decoder`, persistence's
    beside them (decoder-free), each with its episode-bootstrap interval: {psnr, lpips, persist_psnr, persist_lpips,
    <key>_ci}."""
    with open(metrics_path) as f:
        metrics = json.load(f)
    sfx = "" if decoder == STOCK else f"_{decoder}"
    cols = {**{k: v + sfx for k, v in ABSOLUTE.items()}, **{f"persist_{k}": v for k, v in PERSISTENCE.items()}}
    out = {}
    for key, col in cols.items():
        out[key] = next((m for m in (_mean(metrics.get(col + "_nodup")), _mean(metrics.get(col))) if m is not None),
                        None)
        out[f"{key}_ci"] = None
    pw = os.path.join(os.path.dirname(metrics_path), "per_window.csv")
    if draws and os.path.exists(pw):
        windows = read_windows(pw)
        for key, col in cols.items():
            if out[key] is not None:
                out[f"{key}_ci"] = episode_bootstrap(*window_difference(windows, col), draws, seed)
    return out if out["psnr"] is not None else None


def absolute_zero_shot(root, draws, seed, notes):
    """{backbone: {decoder: {"maps": {arena: absolute_read}, "home": absolute_read}}} from the same rows as
    `paired_zero_shot` (a row and its `_tuned` twin), for Figure 3b in absolute form."""
    if not root or not os.path.isdir(root):
        return {}
    names = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith("home_")
                   and d != "adapters")
    out = {}
    for base in sorted({n[:-len("_tuned")] if n.endswith("_tuned") else n for n in names}):
        key = fs.backbone_of(base)
        if key in out:
            continue
        twin = base + "_tuned" if base + "_tuned" in names else None
        entry = {}
        for decoder, row in (("stock", base if base in names else twin), ("tuned", twin)):
            if row is None:
                continue
            maps = {}
            for d in sorted(os.listdir(os.path.join(root, row))):
                m = MAP_DIR_RE.search(d)
                mp = os.path.join(root, row, d, "metrics.json")
                if m and os.path.exists(mp) and (m["h"] is None or int(m["h"]) == 1):
                    r = absolute_read(mp, decoder, draws, seed)
                    if r is not None:
                        maps[int(m["map"] or m["bare"])] = r
            home = None
            for hrow in (row, base):
                found = sorted(glob.glob(os.path.join(root, f"home_{hrow}", "**", "metrics.json"), recursive=True))
                if len(found) == 1:
                    home = absolute_read(found[0], decoder, draws, seed)
                    if home is not None:
                        break
            if maps:
                entry[decoder] = {"maps": maps, "home": home, "source": rel(os.path.join(root, row))}
        if entry:
            out[key] = entry
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
# across arenas: the nested bootstrap (arenas, then each arena's held-out episodes)
# ---------------------------------------------------------------------------------------------

def iqm(values, axis=None):
    """The interquartile mean: `scipy.stats.trim_mean(x, 0.25)` (rliable's IQM; a plain mean below four values)."""
    from scipy.stats import trim_mean
    return trim_mean(values, 0.25, axis=axis)


def episode_matrix(run, steps, decoder):
    """(sums [steps x episodes], counts [episodes]) of A per held-out episode at each step, paired across steps,
    or None when a step lacks its local per-window file or the steps disagree on the episodes or window counts."""
    episodes, sums, counts = None, [], None
    for s in steps:
        row = run.by_step.get(s)
        path = local_per_window(run, row) if row else None
        if path is None:
            return None
        eps, vals = window_outcome(read_windows(path), "A", decoder)
        if not vals:
            return None
        uniq, inv = np.unique(np.asarray(eps), return_inverse=True)
        c = np.bincount(inv).astype(float)
        if episodes is None:
            episodes, counts = uniq, c
        elif len(uniq) != len(episodes) or (uniq != episodes).any() or (c != counts).any():
            return None
        sums.append(np.bincount(inv, weights=np.asarray(vals, float)))
    return np.asarray(sums), counts


def nested_bootstrap(point, mats, draws, seed):
    """[draws x arenas x steps] resampled per-arena curves: arenas drawn with replacement, then each drawn arena's
    episodes with replacement, one episode draw shared by all steps; an arena without episodes keeps its point."""
    rng = np.random.default_rng(seed)
    n, S = point.shape
    idx = rng.integers(0, n, (draws, n))
    out = point[idx].copy()
    for a, mat in enumerate(mats):
        if mat is None:
            continue
        sums, counts = mat
        where = np.argwhere(idx == a)
        if not len(where):
            continue
        E = len(counts)
        e = rng.integers(0, E, (len(where), E))
        out[where[:, 0], where[:, 1], :] = (sums[:, e].sum(-1) / counts[e].sum(-1)).T
    return idx, out


def interval(samples, axis=0):
    """The 95% percentile interval of bootstrap samples (NumPy's default interpolation)."""
    lo, hi = np.nanpercentile(samples, [2.5, 97.5], axis=axis)
    return lo, hi


def first_step(curves, targets, steps):
    """The first step whose value reaches the target, per curve along the last axis; +inf when none does."""
    reached = curves >= targets[..., None]
    first = np.where(reached.any(-1), np.asarray(steps, float)[reached.argmax(-1)], np.inf)
    return first


def censored_quantiles(samples):
    """(2.5, 97.5) percentiles of a sample that holds +inf for censored values, without interpolating into inf."""
    return [float(np.percentile(samples, q, method="inverted_cdf")) for q in (2.5, 97.5)]


def at_risk(costs, steps):
    """Arenas still under observation and not yet across before each step (the Kaplan-Meier at-risk row)."""
    return [int(sum(1 for c in costs if c >= s)) for s in steps]


def profile(values, taus):
    """P(A >= tau) over arenas for each threshold: the fraction of arenas at or above it."""
    v = np.asarray(values, float)
    return (v[..., :, None] >= np.asarray(taus)[None, :]).mean(-2)


def across_arenas(records, runs, decoder, home, budget, draws, seed, notes, fixed=FIXED_THRESHOLD):
    """The distribution statements over the arenas scored to the budget, each with its nested-bootstrap interval,
    plus the arrays the figures draw (under the key `_draw`, which the summary drops)."""
    done = [r for r in records if not r["incomplete"]]
    steps = sorted(set.intersection(*[{int(s) for s in r["A"] if int(s) <= budget} for r in done])) if done else []
    if len(done) < 2 or len(steps) < 2 or 0 not in steps:
        return None
    point = np.array([[r["A"][str(s)] for s in steps] for r in done])
    mats = [episode_matrix(runs[r["run"]], steps, decoder) for r in done]
    missing = [r["arena"] for r, m in zip(done, mats) if m is None]
    if missing:
        notes.append(f"across arenas: arenas {missing} lack paired per-window files at every step; they enter the "
                     "bootstrap with their point curves (arena resampling only)")
    idx, boot = nested_bootstrap(point, mats, draws, seed)
    j_budget = steps.index(budget) if budget in steps else len(steps) - 1
    lines_p = (point[:, 0] + home) / 2
    lines_b = (boot[:, :, 0] + home) / 2
    cost_p = first_step(point, lines_p, steps)
    cost_b = first_step(boot, lines_b, steps)
    fixed_p = first_step(point, np.full(len(done), fixed), steps)
    fixed_b = first_step(boot, np.full(boot.shape[:2], fixed), steps)
    att = [float(np.mean(cost_p <= s)) for s in steps]
    att_b = np.stack([(cost_b <= s).mean(1) for s in steps], 1)
    fatt = [float(np.mean(fixed_p <= s)) for s in steps]
    fatt_b = np.stack([(fixed_b <= s).mean(1) for s in steps], 1)
    iqm_p, iqm_b = iqm(point, axis=0), iqm(boot, axis=1)
    med_p, med_b = np.median(point, 0), np.median(boot, 1)
    gain_p, gain_b = point[:, j_budget] - point[:, 0], boot[:, :, j_budget] - boot[:, :, 0]
    share_p = gain_p / (home - point[:, 0])
    share_b = gain_b / (home - boot[:, :, 0])
    medcost_b = np.median(cost_b, 1)
    n = len(done)

    def band(p, b):
        lo, hi = interval(b)
        return {"value": [float(x) for x in np.atleast_1d(p)], "ci": [[float(a), float(c)] for a, c in
                                                                      zip(np.atleast_1d(lo), np.atleast_1d(hi))]}

    def count(p_frac, b_frac):
        lo, hi = interval(b_frac)
        return {"count": int(round(p_frac * n)), "n": n, "fraction": float(p_frac),
                "fraction_ci": [float(lo), float(hi)], "count_ci": [int(round(lo * n)), int(round(hi * n))]}

    out = {
        "method": (f"nested bootstrap, {draws} draws (seed {seed}): arenas with replacement, then each drawn arena's "
                   "held-out episodes with replacement, paired across steps; half-gap lines recomputed per draw from "
                   "its A0 with home frozen; 95% percentile intervals, pointwise"),
        "n": n, "arenas": [r["arena"] for r in done], "steps": steps, "budget": steps[j_budget],
        "episode_resampling_missing": missing,
        "iqm_A": band(iqm_p, iqm_b), "median_A": band(med_p, med_b),
        "attainment_half_gap": {**band(att, att_b), "at_risk": at_risk(cost_p, steps),
                                "crossing_step": {str(r["arena"]): (None if math.isinf(c) else int(c))
                                                  for r, c in zip(done, cost_p)}},
        "attainment_fixed": {**band(fatt, fatt_b), "threshold": fixed, "at_risk": at_risk(fixed_p, steps)},
        "half_gap_by_budget": count(att[j_budget], att_b[:, j_budget]),
        "fixed_by_budget": count(fatt[j_budget], fatt_b[:, j_budget]),
        "iqm_gap_share": band(iqm((point - point[:, :1]) / (home - point[:, :1]), axis=0),
                              iqm((boot - boot[:, :, :1]) / (home - boot[:, :, :1]), axis=1)),
        "median_gain": band(np.median(gain_p), np.median(gain_b, 1)),
        "iqm_gain": band(iqm(gain_p), iqm(gain_b, axis=1)),
        "gap_share": {"per_arena": {str(r["arena"]): float(s) for r, s in zip(done, share_p)},
                      "median": float(np.median(share_p)), "quartiles": [float(q) for q in
                                                                         np.percentile(share_p, [25, 75])],
                      "median_ci": [float(x) for x in interval(np.median(share_b, 1))]},
        "median_budget": {"value": None if math.isinf(np.median(cost_p)) else float(np.median(cost_p)),
                          "ci": [None if math.isinf(x) else x for x in censored_quantiles(medcost_b)],
                          "censored_share": float(np.mean(np.isinf(medcost_b))),
                          "distribution": {("censored" if math.isinf(v) else f"{v:g}"): float(np.mean(medcost_b == v))
                                           for v in np.unique(medcost_b)}},
        "profiles": {}, "_draw": {"point": point, "boot": boot, "att_b": att_b, "fatt_b": fatt_b, "iqm_b": iqm_b,
                                  "cost_p": cost_p, "fixed_p": fixed_p, "idx": idx},
    }
    # an arena crosses by the budget when home <= 2 A_s - A0 at some step s: every arena crosses at or below this home
    out["gap_share"]["home_at_which_all_cross"] = float(np.min(np.max(2 * point - point[:, :1], axis=1)))
    S0 = np.array([r["S0"] if r["S0"] is not None else np.nan for r in done])
    if np.isfinite(S0).sum() >= 3:
        from scipy.stats import spearmanr
        rho = spearmanr(S0, point[:, j_budget]).statistic
        rhos = [spearmanr(S0[i], boot[b, :, j_budget]).statistic for b, i in enumerate(idx[:2000])]
        out["spearman_S0_A_budget"] = {"rho": float(rho), "ci": [float(x) for x in np.nanpercentile(rhos, [2.5, 97.5])],
                                       "draws": min(2000, draws)}
        rho_g = spearmanr(S0, gain_p).statistic
        out["spearman_S0_gain"] = {"rho": float(rho_g)}
    taus = sorted({2.5, 3.0, 3.5, 4.0, 4.5, 5.0, round(home, 3)})
    for s in sorted({*[p for p in PROFILE_STEPS if p in steps], steps[j_budget]}):
        j = steps.index(s)
        p, b = profile(point[:, j], taus), profile(boot[:, :, j], taus)
        lo, hi = interval(b)
        out["profiles"][str(s)] = {"tau": taus, "fraction": [float(x) for x in p],
                                   "ci": [[float(a), float(c)] for a, c in zip(lo, hi)]}
    margins = [(r["arena"], r["M_budget"]) for r in done if r["M_budget"] is not None]
    if len(margins) >= 2:
        m = np.array([v for _, v in margins])
        mi = np.random.default_rng(seed).integers(0, len(m), (draws, len(m)))
        lo, hi = interval((m[mi] < 0).mean(1))
        mlo, mhi = interval(np.median(m[mi], 1))
        out["margin_budget"] = {"lower": int((m < 0).sum()), "within_001": int(((m >= 0) & (m <= 0.01)).sum()),
                                "n": len(m), "lower_fraction_ci": [float(lo), float(hi)],
                                "median": float(np.median(m)), "median_ci": [float(mlo), float(mhi)],
                                "method": "arena bootstrap of the per-arena margins"}
    return out


def public(stats):
    """The across-arena statistics without the arrays the figures draw."""
    return {k: v for k, v in stats.items() if not k.startswith("_")} if stats else None


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


def num_ci(v, ci, digits, prov=False):
    """A number with its interval in brackets, `2.04 [1.70, 2.36]`, or the number alone without an interval."""
    s = num(v, digits)
    if ci and all(finite(x) for x in ci):
        s += f" [{num(ci[0], digits)}, {num(ci[1], digits)}]"
    return f"\\prov{{{s}}}" if prov and finite(v) else s


def cost_cell(step, censored, incomplete, budget, prov=False):
    if incomplete:
        return "\\tbd"
    s = f"${{>}}${budget:,}" if censored else f"{step:,}"
    return f"\\prov{{{s}}}" if prov else s


def median_cost_cell(entry, prov, ci=None, censored_n=None, n=None, budget=None):
    """The median budget: censored when the middle arena is; its arena-bootstrap interval in brackets (an upper end
    past the budget written as >budget) and the count of censored arenas beside it."""
    if entry["censored"] is None:
        return "\\tbd"
    s = f"${{>}}${int(entry['label'][1:]):,}" if entry["censored"] else f"{entry['value']:,.0f}"
    if ci is not None:
        lo, hi = ci
        s += " [" + ("${>}$" + f"{budget:,}" if lo is None else f"{lo:,.0f}") + ", " + \
             ("${>}$" + f"{budget:,}" if hi is None else f"{hi:,.0f}") + "]"
    if censored_n is not None and n:
        s += f"; {censored_n} of {n} censored"
    return f"\\prov{{{s}}}" if prov else s


def header(files, when, lines):
    first = f"% generated by make_adapt_figures.py from {', '.join(rel(f) for f in files)} at {when}"
    return "\n".join([first] + [f"% {x}" for x in lines]) + "\n"


def tabular(colspec, head, body):
    out = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    out += ["\\midrule" if row == "midrule" else " & ".join(row) + " \\\\" for row in body]
    return "\n".join(out + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def cost_table(records, summary, decoder, budget, prov, stats=None):
    """The per-arena cost table: arena, D, A0, A at the budget, half-gap line, cost, S and LPIPS at 0 and the budget;
    a median row and, with the across-arena statistics, an IQM row, both with arena-bootstrap intervals."""
    b = budget_label(budget)
    others = sorted({d for r in records for d in r["other_decoders"]})
    head = ["Unseen map", "$d$", "$A_0$", f"$A_{{\\mathrm{{{b}}}}}$", "Half-gap line", "Budget", "$S_0$",
            f"$S_{{\\mathrm{{{b}}}}}$", "LPIPS$_0$", f"LPIPS$_{{\\mathrm{{{b}}}}}$"]
    head += [f"$A_{{\\mathrm{{{b}}}}}$, {DECODER_NAMES.get(d, d)}" for d in others]
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
    med_ci = {0: None, 1: None}
    cost_ci, censored_n, n = None, None, None
    if stats:
        med_ci = {0: stats["median_A"]["ci"][0], 1: stats["median_A"]["ci"][-1]}
        cost_ci, n = stats["median_budget"]["ci"], stats["n"]
        censored_n = n - stats["half_gap_by_budget"]["count"]
    body += ["midrule", ["Median", "", num_ci(m["A0"], med_ci[0], 2, prov), num_ci(m["A_budget"], med_ci[1], 2, prov),
                         num(m["half_gap_line"], 2, prov=prov),
                         median_cost_cell(m["cost_half_gap"], prov, cost_ci, censored_n, n, budget),
                         num(m["S0"], 2, prov=prov), num(m["S_budget"], 2, prov=prov), num(m["lpips0"], 3, prov=prov),
                         num(m["lpips_budget"], 3, prov=prov)] + [num(v, 2, True, prov) for v in other_medians]]
    if stats:
        v, ci = stats["iqm_A"]["value"], stats["iqm_A"]["ci"]
        body.append(["IQM", "", num_ci(v[0], ci[0], 2, prov), num_ci(v[-1], ci[-1], 2, prov)] +
                    [""] * (len(head) - 4))
    return tabular("r" * len(head), head, body)


def perarena_table(records, budget, prov):
    """The appendix table `tab:perarena-adapt`: the columns appendix.tex declares, one row per arena."""
    b = budget_label(budget)
    head = ["Unseen map", "$d$", "$A_0$", "Half-gap line", "Half gap", "Training-maps line", f"$A$ at {b}",
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


def table3(records, summary, stats, home, home_ci, budget, prov):
    """Table 3 (`tab:cost`): the 13-arena column as medians with arena-bootstrap intervals in brackets, the
    comparator arena with its episode intervals, the full fine-tune pending; the median budget states its censoring."""
    b = budget_label(budget)
    comp = next((r for r in records if r["arena"] == COMPARATOR_ARENA and not r["incomplete"]), None)
    m = summary["medians"]
    n = summary["n_complete"]

    def cnt(entry):
        if not entry:
            return "\\tbd"
        lo, hi = entry["count_ci"]
        s = f"{entry['count']} [{lo}, {hi}]"
        return f"\\prov{{{s}}}" if prov else s

    def cost_of(r, key, cens):
        return "\\tbd" if r is None else cost_cell(r[key], r[cens], r["incomplete"], budget, prov)

    cost_all = median_cost_cell(m["cost_half_gap"], prov, stats["median_budget"]["ci"] if stats else None,
                                n - stats["half_gap_by_budget"]["count"] if stats else None, n, budget)
    home_cost = median_cost_cell(m["cost_home"], prov)
    mb = stats.get("margin_budget") if stats else None
    share = stats["gap_share"] if stats else None
    home_cell = num_ci(home, (home_ci or {}).get("ci"), 2, prov)
    rows = [
        ["Parameters trained; GPU-hours", f"{LORA_PARAMS}; \\tbd{{}}", "4.2M; \\tbd{}", f"{FULL_PARAMS}; \\tbd{{}}"],
        ["$A_\\text{train}$ (dB), training maps (in-distribution)", home_cell, "--", "--"],
        [f"Unseen maps past half gap / $A\\geq{FIXED_THRESHOLD:g}$ dB by {b}",
         f"{cnt(stats and stats['half_gap_by_budget'])} / {cnt(stats and stats['fixed_by_budget'])} of {n}", "--",
         "--"],
        [f"Share of the gap closed at {b}",
         (num_ci(share["median"], share["median_ci"], 2, prov) if share else "\\tbd"),
         num(comp and comp["gap_share_budget"], 2, prov=prov), "\\tbd"],
        ["Budget to half gap / training-maps line (updates)", f"{cost_all} / {home_cost}",
         f"{cost_of(comp, 'cost_half_gap', 'censored_half_gap')} / {cost_of(comp, 'cost_home', 'censored_home')}",
         "\\tbd{} / \\tbd{}"],
        [f"$A$ (dB) / $M$ / $G$ (dB) at {b}",
         f"{num_ci(m['A_budget'], stats and stats['median_A']['ci'][-1], 2, prov)} / "
         f"{num_ci(mb['median'], mb['median_ci'], 3, prov) if mb else num(m['M_budget'], 3, True, prov)} / "
         f"{num(m['G_budget'], 2, prov=prov)}",
         (f"{num_ci(comp['A_budget'], comp['A_budget_ci'], 2, prov)} / "
          f"{num_ci(comp['M_budget'], comp['M_budget_ci'], 3, prov)} / {num(comp['G_budget'], 2, prov=prov)}")
         if comp else "\\tbd{} / \\tbd{} / \\tbd{}", "\\tbd{} / \\tbd{} / \\tbd{}"],
        [f"Forgetting (dB) / directional at {b}",
         f"{num(m['forgetting_budget'], 2, True, prov)} / {num(m['directional_budget'], 2, prov=prov)}",
         (f"{num(comp['forgetting_budget'], 2, True, prov)} / {num(comp['directional_budget'], 2, prov=prov)}"
          if comp else "\\tbd{} / \\tbd{}"), "\\tbd{} / \\tbd{}"],
    ]
    head = ["", f"LoRA, {n} unseen maps, median [95\\% map bootstrap]", f"LoRA, map {COMPARATOR_ARENA}",
            f"Full, map {COMPARATOR_ARENA}"]
    return tabular("lrrr", head, rows)


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------

def a_axis(ax, lo_data, hi_data, label=fs.A_LABEL, step=1.0):
    """The A axis: from 0 (copy-last) to past the data, round 1 dB ticks."""
    top = max(hi_data, 0.0)
    ax.set_ylim(min(0.0, lo_data) - 0.15, top + 0.35)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.set_ylabel(label)


def fig_adaptation(stats, records, home, budget, out_dir, fixed=FIXED_THRESHOLD, band=None):
    """Figure 4. (a) The interquartile mean (IQM) of A across arenas against updates with its nested-bootstrap
    band, the per-arena curves faint behind (arenas 7, 9 and 12 labelled at their ends), the training maps' band
    and copy-last. (b) Cumulative attainment, one minus Kaplan-Meier: the fraction of arenas whose A has reached its
    half-gap line (solid, band), the fixed threshold as a dashed curve with its own thin bounds, one tick per arena
    at its crossing in lanes above the curves (open at the last grid step: censored there), and the arenas at risk
    before each read under the labelled budgets."""
    fig, (ax, bx) = fs.new_figure(FIG4_SIZE, ncols=2, wspace=0.08, h_pad=0.01)
    iqm_panel(ax, stats, home, band)
    attainment_panel(bx, stats, fixed)
    for a in (ax, bx):                      # one shared update label under both panels
        a.set_xlabel("")
    fig.supxlabel("adapter updates (log scale)", fontsize=fs.LABEL_PT)
    return fs.save(fig, out_dir, "fig4_adaptation")


def iqm_panel(ax, stats, home, band, letter="a"):
    """Figure 4a: the IQM of A over the faint per-arena curves."""
    d = stats["_draw"]
    steps, point = stats["steps"], d["point"]
    z = step_axis(ax, steps)
    xs = [z if s == 0 else s for s in steps]
    for row in point:
        ax.plot(xs, row, color=fs.FAINT, lw=fs.MIN_LW, zorder=2)
    val, ci = stats["iqm_A"]["value"], np.array(stats["iqm_A"]["ci"])
    ax.fill_between(xs, ci[:, 0], ci[:, 1], color=ADAPTER.colour, alpha=0.18, lw=0, zorder=2.5)
    ax.plot(xs, val, color=ADAPTER.colour, lw=fs.DATA_LW, marker=ADAPTER.marker, ms=fs.MARKER_SIZE, mec="white",
            mew=fs.MARKER_EDGE, zorder=4)
    # an end within 0.15 dB of the training maps' dashed line keeps its leader on the data; only its text moves up
    ends = [(xs[-1], row[-1], str(a), fs.CONTEXT_INK, None if abs(row[-1] - home) >= 0.15 else home + 0.22)
            for a, row in zip(stats["arenas"], point) if a in NAMED_ARENAS]
    fs.end_labels(ax, ends + [(xs[-1], val[-1], "IQM", ADAPTER.colour)], gap=0.4, leaders=True)
    fs.training_line(ax, home, where=0.0, align="left", band=band)
    fs.copy_last_line(ax, where=1.0, align="right")
    a_axis(ax, 0.0, max(home, point.max()))
    fs.panel_letter(ax, letter)


RUG_LANE = 0.125          # data units of the attainment axis per rug lane (about 6.5 pt at 1.6 in)
RUG_BASE = 1.06           # the lowest rug lane sits just above the fraction 1


def attainment_panel(bx, stats, fixed, letter="b"):
    """Figure 4b: attainment of the half-gap line and of the fixed threshold, the crossing rug, the at-risk row."""
    steps = stats["steps"]
    z = step_axis(bx, steps, labelled=steps)
    xs = [z if s == 0 else s for s in steps]
    att, att_ci = stats["attainment_half_gap"]["value"], np.array(stats["attainment_half_gap"]["ci"])
    fatt, fatt_ci = stats["attainment_fixed"]["value"], np.array(stats["attainment_fixed"]["ci"])
    bx.fill_between(xs, att_ci[:, 0], att_ci[:, 1], step="post", color=ADAPTER.colour, alpha=0.18, lw=0,
                    zorder=2.5)
    for k in (0, 1):
        bx.step(xs, fatt_ci[:, k], where="post", color=fs.CONTEXT_INK, lw=fs.MIN_LW, ls=(0, (1, 1.5)), zorder=3)
    bx.step(xs, att, where="post", color=ADAPTER.colour, lw=fs.DATA_LW, zorder=4)
    # the fixed threshold: neutral dark dashes above the blue curve, so the stretches they share stay visible
    bx.step(xs, fatt, where="post", color=fs.INK, lw=0.8, ls=(0, (3, 2)), zorder=4.5)
    label_attainment(bx, xs, att, fatt, fixed)
    lanes = crossing_rug(bx, stats, z, steps)
    bx.set_ylim(-0.03, RUG_BASE + RUG_LANE * lanes)
    bx.spines["left"].set_bounds(0, 1)
    bx.yaxis.set_major_locator(ticker.FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
    bx.yaxis.set_major_formatter(ticker.FixedFormatter(["0", "", "0.5", "", "1"]))
    bx.set_ylabel("fraction reaching\nthreshold")
    at_risk_row(bx, stats["attainment_half_gap"]["at_risk"], xs)
    fs.panel_letter(bx, letter)


def label_attainment(ax, xs, att, fatt, fixed):
    """Direct labels inside the panel: the half-gap curve left of its first rise; the fixed-threshold curve under
    its longest stretch apart from the half-gap curve, starting just right of that stretch's riser (or under the
    half-gap label when the two curves coincide)."""
    rise = next((i for i, v in enumerate(att) if v > 0), None)
    if rise is not None:
        # under the first flat stretch after the rise: the space left of the riser is the y axis's
        fs.direct_label(ax, xs[rise], att[rise], "half-gap threshold", colour=ADAPTER.colour, dx=2.5, dy=-1.5,
                        ha="left", va="top")
    name = f"$A\\geq{fixed:g}$ dB"
    apart = [i for i in range(len(xs) - 1) if abs(att[i] - fatt[i]) > 1e-9]
    if apart:
        i = max(apart, key=lambda i: xs[i + 1] / xs[i])
        below = fatt[i] < att[i]
        fs.direct_label(ax, xs[i], fatt[i], name, colour=fs.INK, dx=2.5, dy=-1.5 if below else 1.5,
                        ha="left", va="top" if below else "bottom")
    elif rise is not None:
        fs.direct_label(ax, xs[rise], att[rise], "and " + name, colour=fs.INK, dx=-3, dy=-8, ha="right")


OPEN_TICK = Path([(-0.18, -0.5), (0.18, -0.5), (0.18, 0.5), (-0.18, 0.5), (-0.18, -0.5)],
                 [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])


def crossing_rug(ax, stats, z, steps):
    """One tick per arena above the curves, at its crossing step; arenas tied at a step take separate lanes (the
    lowest arena number on top) with their numbers beside the ticks; arenas that never cross are open ticks at the
    last grid step, marked "censored at <step>". Returns the number of lanes."""
    groups = {}
    for arena, step in stats["attainment_half_gap"]["crossing_step"].items():
        groups.setdefault(("open", steps[-1]) if step is None else ("filled", step), []).append(int(arena))
    lanes = max((len(v) for v in groups.values()), default=1)
    top_y = RUG_BASE + RUG_LANE * (lanes - 0.5)
    for (kind, step), arenas in groups.items():
        x = z if step == 0 else step
        for lane, a in enumerate(sorted(arenas)):
            y = top_y - RUG_LANE * lane
            if kind == "open":
                ax.plot([x], [y], marker=OPEN_TICK, ms=5.5, mfc="white", mec=fs.INK, mew=0.5, ls="none",
                        clip_on=False, zorder=5, gid="decor")
            else:
                ax.plot([x, x], [y - RUG_LANE * 0.36, y + RUG_LANE * 0.36], color=fs.INK, lw=1.0,
                        solid_capstyle="butt", clip_on=False, zorder=5, gid="decor")
            ax.annotate(str(a), (x, y), xytext=(2.5, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=fs.MIN_PT, color=fs.CONTEXT_INK, annotation_clip=False)
        if kind == "open":
            y = top_y - RUG_LANE * (len(arenas) - 1)
            ax.annotate(f"censored at {step_label(step)}", (x, y), xytext=(-3, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=fs.MIN_PT, color=fs.CONTEXT_INK, annotation_clip=False)
    return lanes


def at_risk_row(ax, counts, xs, below_pt=12.5):
    """The arenas at risk just before each read, between the tick labels and the axis label, one under each
    labelled budget, the row named at the left."""
    fig = ax.figure
    shift = transforms.ScaledTranslation(0, -below_pt / 72, fig.dpi_scale_trans)
    under = transforms.blended_transform_factory(ax.transData, ax.transAxes) + shift
    for x, c in zip(xs, counts):
        ax.text(x, 0, str(c), transform=under, ha="center", va="top", fontsize=fs.MIN_PT, color=fs.CONTEXT_INK,
                gid="decor")
    ax.text(0.0, 0, "at risk\nbefore read ", transform=ax.transAxes + shift, ha="right", va="top",
            fontsize=fs.MIN_PT, color=fs.CONTEXT_INK, linespacing=1.0, gid="decor")
    ax.xaxis.labelpad = below_pt - 3.0


CENSORED_FILL, CENSORED_EDGE = "#A6A6A6", "#595959"   # grey-filled: open means the stock decoder elsewhere
HALF_GAP_LABEL = "half gap"        # Table 3's words; "half of the gap" is wider than any free stretch at 8k
LABEL_PAD = 0.4                    # arena slots: half a budget label's width and a gutter


def clears_half_gap(share, never):
    """True when an arena's dot and its label leave the band just above the 0.5 line free: a crossing dot high
    enough that its budget sits above the text, or a censored dot under the line with its "never" beneath."""
    return share < 0.47 if never else share > 0.65


def half_gap_x(shares, never, width, pad=LABEL_PAD, lo=-0.8, hi=None):
    """The x centre (in arena slots) of the half-gap line's label, `width` slots wide, set just above the line:
    among the placements inside the panel [lo, hi], the one that meets the fewest arenas that do not clear the
    text (`clears_half_gap`), then the one with the most room to the nearest of them or to the panel's edge."""
    hi = len(shares) - 0.4 if hi is None else hi
    blocked = [i for i, (v, n) in enumerate(zip(shares, never)) if not clears_half_gap(v, n)]
    best = None
    for c in np.arange(lo + width / 2, hi - width / 2 + 1e-9, 0.05):
        hits = sum(1 for i in blocked if abs(i - c) <= width / 2 + pad)
        room = min([abs(i - c) for i in blocked] + [c - lo, hi - c]) - width / 2 - pad
        if best is None or (hits, -room) < best[0]:
            best = ((hits, -room), float(c))
    return best[1] if best else (lo + hi) / 2


def fig_dots(stats, records, home, out_dir, band=None):
    """Figure 4, third variant (Rohan's simpler form). (a) The 13 per-arena curves of A in light grey and their median
    in bold with markers at the reads, the in-distribution band and the persistence line; no bootstrap band (the
    IQM and its interval go to the caption). (b) One dot per arena, arenas in number order as in Figure 3b, at the
    share of its gap closed at the last read, (A - A0) / (A_train - A0); above each dot the budget at which it first
    crossed half of its gap, or "never" under a grey-filled dot (open would mean the stock decoder); lines at 0, 0.5
    and 1."""
    d = stats["_draw"]
    steps, point = stats["steps"], d["point"]
    fig, (ax, bx) = fs.new_figure(DOTS_SIZE, ncols=2, wspace=0.08)
    z = step_axis(ax, steps)
    xs = [z if s == 0 else s for s in steps]
    for row in point:
        ax.plot(xs, row, color=fs.FAINT, lw=fs.MIN_LW, zorder=2)
    med = np.median(point, axis=0)
    ax.plot(xs, med, color=ADAPTER.colour, lw=1.4, marker=ADAPTER.marker, ms=fs.MARKER_SIZE, mec="white",
            mew=fs.MARKER_EDGE, zorder=4)
    fs.direct_label(ax, xs[-1], med[-1], "median", colour=ADAPTER.colour, dx=3)
    fs.training_line(ax, home, where=0.0, align="left", band=band)
    fs.copy_last_line(ax, where=1.0, align="right")
    a_axis(ax, 0.0, max(home, point.max()), label=fs.A_LABEL_SHORT)   # one line overruns a 1 in axis
    fs.panel_letter(ax, "a")

    shares = stats["gap_share"]["per_arena"]
    order = sorted(r["arena"] for r in records if str(r["arena"]) in shares)     # number order, as Figure 3b
    crossing = stats["attainment_half_gap"]["crossing_step"]
    lowered = False
    for i, a in enumerate(order):
        v, step = shares[str(a)], crossing.get(str(a))
        never = step is None
        bx.plot([i], [v], ls="none", marker="o", ms=4.0, mew=0.8, mec=CENSORED_EDGE if never else ADAPTER.colour,
                mfc=CENSORED_FILL if never else ADAPTER.colour, color=CENSORED_FILL if never else ADAPTER.colour,
                zorder=3)
        # a censored arena sits just under the 0.5 line, so its "never" goes beneath the dot, clear of the line;
        # neighbouring "never" labels alternate between two depths so they do not overlap
        prev_never = i > 0 and crossing.get(str(order[i - 1])) is None
        lowered = never and prev_never and not lowered
        dy = (-12 if lowered else -5) if never else 4
        bx.annotate("never" if never else step_label(step), (i, v), xytext=(0, dy),
                    textcoords="offset points", ha="center", va="top" if never else "bottom", fontsize=fs.MIN_PT,
                    color=fs.CONTEXT_INK, annotation_clip=False,
                    bbox={"boxstyle": "square,pad=0.05", "fc": "white", "ec": "none"})   # break a line the text sits on
    bx.set_xlim(-0.8, len(order) - 0.4)          # room for the first arena's "never" inside the spine
    bx.set_ylim(0, 1.1)
    for y, text, style in ((0.0, "zero-shot", "-"), (0.5, HALF_GAP_LABEL, (0, (1, 1.2))),
                           (1.0, "in-distribution", fs.TRAINING_DASH)):
        bx.axhline(y, color=fs.BLACK if y == 0 else (fs.CONTEXT_INK if y == 0.5 else fs.TRAINING_LINE),
                   lw=fs.REF_LW if y != 0.5 else fs.MIN_LW, ls=style, zorder=1.5, gid="ref")
        if y == 0.5:
            # on the line, where no dot or budget label meets the text (`half_gap_x`); its width is measured in
            # arena slots at the printed size
            label = bx.text(0, y, text, fontsize=fs.ANNOT_PT)
            box = label.get_window_extent(fig.canvas.get_renderer())
            label.remove()
            inv = bx.transData.inverted()
            width = inv.transform((box.x1, box.y0))[0] - inv.transform((box.x0, box.y0))[0]
            mid = half_gap_x([shares[str(a)] for a in order], [crossing.get(str(a)) is None for a in order], width)
            bx.annotate(text, (mid, y), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom",
                        fontsize=fs.ANNOT_PT, color=fs.CONTEXT_INK)
            continue
        bx.text(1.0, y, text, transform=bx.get_yaxis_transform(), ha="right", va="bottom", fontsize=fs.ANNOT_PT,
                color=fs.INK if y == 0 else fs.TRAINING_LINE)
    bx.set_xticks(range(len(order)), [str(a) for a in order])
    bx.tick_params(axis="x", length=0)
    bx.yaxis.set_major_locator(ticker.FixedLocator([0, 0.5, 1]))
    bx.yaxis.set_major_formatter(ticker.FixedFormatter(["0", "0.5", "1"]))
    bx.set_xlabel("unseen map")
    bx.set_ylabel("share of gap\nclosed at " + step_label(stats["budget"]))
    fs.panel_letter(bx, "b")
    return fs.save(fig, out_dir, "fig4_dots")


def fig_gapshare(stats, home, out_dir):
    """Figure 4a, variant for the round-2 choice: the share of each arena's gap to the in-distribution reference
    closed, (A - A0) / (A_ref - A0), per arena faint, the IQM with its nested band, the half-gap criterion at 0.5
    and the reference at 1."""
    d = stats["_draw"]
    steps, point, boot = stats["steps"], d["point"], d["boot"]
    share = (point - point[:, :1]) / (home - point[:, :1])
    share_b = (boot - boot[:, :, :1]) / (home - boot[:, :, :1])
    fig, (ax,) = fs.new_figure((FIG4_SIZE[0] / 2, FIG4_SIZE[1]))
    z = step_axis(ax, steps)
    xs = [z if s == 0 else s for s in steps]
    for row in share:
        ax.plot(xs, row, color=fs.FAINT, lw=fs.MIN_LW, zorder=2)
    val = iqm(share, axis=0)
    lo, hi = interval(iqm(share_b, axis=1))
    ax.fill_between(xs, lo, hi, color=ADAPTER.colour, alpha=0.18, lw=0, zorder=2.5)
    ax.plot(xs, val, color=ADAPTER.colour, lw=fs.DATA_LW, marker=ADAPTER.marker, ms=fs.MARKER_SIZE, mec="white",
            mew=fs.MARKER_EDGE, zorder=4)
    fs.direct_label(ax, xs[-1], val[-1], "IQM", colour=ADAPTER.colour, dx=3)
    ax.axhline(0.5, color=fs.CONTEXT_INK, lw=fs.MIN_LW, ls=(0, (1, 1.2)), zorder=1.4, gid="ref")
    ax.text(0.0, 0.5, " half of in-distribution gap", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=fs.ANNOT_PT, color=fs.CONTEXT_INK)
    fs.training_line(ax, 1.0, where=0.0)
    ax.axhline(0, color=fs.BLACK, lw=fs.REF_LW, zorder=1.5, gid="ref")
    ax.text(1.0, 0, "zero-shot", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=fs.ANNOT_PT)
    ax.set_ylim(-0.08, max(1.15, float(share.max()) + 0.1))
    ax.yaxis.set_major_locator(ticker.FixedLocator([0, 0.5, 1]))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(["0", "0.5", "1"]))
    ax.set_ylabel("share of the gap closed,\n$(A - A_0)/(A_\\mathrm{train} - A_0)$")
    fs.panel_letter(ax, "a")
    return fs.save(fig, out_dir, "fig4a_gapshare")


def fig_arenas(records, runs, home, budget, out_dir, band=None):
    """Appendix: one small panel per arena, A against updates with its episode band and diamonds at the measured
    budgets, the half-gap line (dotted), the training maps' band, copy-last at 0 and the crossing ticked; shared
    axes and one shared update label, ordered by arena; a key in the first empty slot."""
    recs = sorted(records, key=lambda r: r["arena"])
    ncols = min(5, len(recs))
    nrows = math.ceil(len(recs) / ncols)
    fig, axes = fs.new_figure((ARENAS_SIZE[0], min(ARENAS_SIZE[1], 0.3 + 0.95 * nrows)), ncols=ncols, nrows=nrows,
                              sharex=True, sharey=True, wspace=0.02, hspace=0.03)
    steps = sorted({int(s) for r in recs for s in r["A"] if int(s) <= budget})
    top = max(max(v for r in recs for v in r["A"].values()), home)
    keyed = False
    for i, ax in enumerate(axes):
        if i >= len(recs):
            if keyed:
                ax.remove()
            else:
                arenas_key(ax, band is not None)
                keyed = True
            continue
        r = recs[i]
        z = step_axis(ax, steps, label="")
        pts = [(s, v) for s, v in sorted((int(s), v) for s, v in r["A"].items()) if s <= budget]
        x = xs_of(pts, z)
        cis = [r["A_ci"].get(str(s)) for s, _ in pts]
        if all(cis):
            ax.fill_between(x, [c[0] for c in cis], [c[1] for c in cis], color=ADAPTER.colour, alpha=0.25, lw=0,
                            zorder=2.5)
        ax.plot(x, [v for _, v in pts], color=ADAPTER.colour, lw=fs.DATA_LW, marker=ADAPTER.marker, ms=3.0,
                mec="white", mew=fs.MARKER_EDGE, zorder=3)
        ax.axhline(r["half_gap_line"], color=fs.CONTEXT_INK, lw=fs.MIN_LW, ls=(0, (1, 1.2)), zorder=1.4, gid="ref")
        fs.training_line(ax, home, label=fs.TRAINING_LABEL if i == 0 and len(recs) % ncols == 0 else None,
                         where=0.0, band=band)
        fs.copy_last_line(ax, label=False)
        if r["cost_half_gap"] is not None:
            cx = z if r["cost_half_gap"] == 0 else r["cost_half_gap"]
            ax.plot([cx, cx], [r["half_gap_line"] - 0.35, r["half_gap_line"] + 0.35], color=fs.INK, lw=0.8,
                    zorder=4, gid="decor")
        ax.text(0.97, 0.07, f"map {r['arena']}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=fs.ANNOT_PT, gid="decor")
        ax.set_ylim(-0.2, top + 0.4)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        if i % ncols == 0:
            ax.set_ylabel(fs.A_LABEL_SHORT)
        # the lowest panel of each column carries the x tick labels, also where the row below is short
        ax.tick_params(labelbottom=i + ncols >= len(recs))
    fig.supxlabel("adapter updates (log scale)", fontsize=fs.LABEL_PT)
    return fs.save(fig, out_dir, "figA_adapt_arenas")


def arenas_key(ax, with_band):
    """The small multiples' key, drawn in the first empty slot: what each reference mark means."""
    ax.set_gid("decor")
    ax.axis("off")
    rows = [("training maps", "(in-distribution)", "train"), ("half-gap line", "", "half"),
            ("first crossing", "", "tick"), ("measured read", "", "read"), (fs.PERSISTENCE_LABEL, "", "zero")]
    for k, (text, sub, kind) in enumerate(rows):
        y = 0.9 - 0.2 * k
        if kind == "train":
            if with_band:
                ax.add_patch(matplotlib.patches.Rectangle((0.02, y - 0.05), 0.22, 0.1, transform=ax.transAxes,
                                                          color=fs.TRAINING_BAND, lw=0))
            ax.plot([0.02, 0.24], [y, y], transform=ax.transAxes, color=fs.TRAINING_LINE, lw=fs.REF_LW,
                    ls=fs.TRAINING_DASH)
        elif kind == "half":
            ax.plot([0.02, 0.24], [y, y], transform=ax.transAxes, color=fs.CONTEXT_INK, lw=fs.MIN_LW, ls=(0, (1, 1.2)))
        elif kind == "tick":
            ax.plot([0.13, 0.13], [y - 0.06, y + 0.06], transform=ax.transAxes, color=fs.INK, lw=0.8)
        elif kind == "read":
            ax.plot([0.13], [y], transform=ax.transAxes, marker=ADAPTER.marker, ms=3.0, color=ADAPTER.colour,
                    mec="white", mew=fs.MARKER_EDGE, ls="none")
        else:
            ax.plot([0.02, 0.24], [y, y], transform=ax.transAxes, color=fs.BLACK, lw=fs.REF_LW)
        ax.text(0.3, y, text + (f"\n{sub}" if sub else ""), transform=ax.transAxes, ha="left", va="center",
                fontsize=fs.ANNOT_PT, color=fs.CONTEXT_INK, linespacing=1.0)


def fig_profiles(stats, home, out_dir, band=None):
    """Appendix: performance profiles, the fraction of arenas with A at or above a threshold, at step 0, 500 and
    the budget, each with its nested-bootstrap band, in the adapter's blue ramp from light (step 0) to dark (the
    budget); the threshold axis from 0 dB (copy-last) with the training maps' band as a vertical reference."""
    d = stats["_draw"]
    steps, point, boot = stats["steps"], d["point"], d["boot"]
    show = sorted({*[s for s in PROFILE_STEPS if s in steps], stats["budget"]})
    lo_t, hi_t = 0.0, float(max(point.max(), home)) + 0.3
    taus = np.linspace(lo_t, hi_t, 400)
    fig, (ax,) = fs.new_figure(HALF_SIZE)
    for n, s in enumerate(show):
        j = steps.index(s)
        colour = fs.ADAPTER_RAMP(0.15 + 0.85 * n / max(1, len(show) - 1))
        p = profile(point[:, j], taus)
        lo, hi = interval(profile(boot[:, :, j], taus))
        ax.fill_between(taus, lo, hi, step="post", color=colour, alpha=0.13, lw=0)
        ax.step(taus, p, where="post", color=colour, lw=fs.DATA_LW)
        # labels alternate sides of their curves at staggered heights so neighbouring curves do not collide
        level = (0.62, 0.9, 0.35, 0.75)[n % 4]
        k = int(np.argmin(np.abs(p - level)))
        left = n % 2 == 1
        fs.direct_label(ax, taus[k], p[k], "zero-shot" if s == 0 else f"{step_label(s)} updates", colour=colour,
                        dx=-3 if left else 3, ha="right" if left else "left")
    fs.training_line(ax, home, orientation="v", label=None, band=band)
    top = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(home, 1.02, fs.TRAINING_LABEL + " ", transform=top, ha="right", va="bottom", fontsize=fs.ANNOT_PT,
            color=fs.TRAINING_LINE)
    ax.axvline(0, color=fs.BLACK, lw=fs.REF_LW, zorder=1.5, gid="ref")
    ax.text(0, 0.02, " " + fs.PERSISTENCE_LABEL, transform=top, ha="left", va="bottom", fontsize=fs.ANNOT_PT)
    ax.set_xlim(lo_t - 0.1, hi_t)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.FixedLocator([0, 0.5, 1]))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(["0", "0.5", "1"]))
    ax.set_xlabel("threshold $\\tau$ on " + fs.A_LABEL)
    ax.set_ylabel("unseen maps at or above $\\tau$\n(fraction)")
    return fs.save(fig, out_dir, "figA_adapt_profiles")


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
    a_axis(ax, min(vals), max(vals + ([ref] if ref is not None else [])), fs.S_LABEL if skill else fs.A_LABEL)
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
    ax.set_ylabel(fs.A_LABEL)
    ax2.set_ylabel("seed 1 $-$ seed 0 (dB)")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(4))
    fs.panel_letter(ax, "a")
    fs.panel_letter(ax2, "b")
    return fs.save(fig, out_dir, "fig3_adaptation_seeds")


def ladder_gains(ladder, anchors_step0):
    """{arena: {k: A_k - A0}}: each ladder rung's gain over the arena's own 0-update read. Copy-last is the same
    on both reads, so this is the rung's PSNR gain against the decoded ground truth."""
    return {str(arena): {str(k): v - anchors_step0[int(arena)] for k, v in byk.items() if v is not None}
            for arena, byk in ladder.items() if anchors_step0.get(int(arena)) is not None}


def fig_ladder(ladder, records, anchors_step0, home, out_dir, band=None):
    """Appendix: the PSNR gain over the arena's own 0-update read (A - A0) against adaptation episodes (log scale)
    per arena, dark grey with the arena labelled at its right end, the zero line labelled "0 updates". Absolute
    reads are not drawn: the ladder runs carry only the decoded-reference quantity through the stock decoder."""
    fig, (ax,) = fs.new_figure(HALF_SIZE)
    gains = ladder_gains(ladder, anchors_step0)
    ks = sorted({int(k) for v in gains.values() for k in v})
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(ticker.FixedLocator(ks))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(k) for k in ks]))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    vals, ends = [0.0], []
    for i, (arena, byk) in enumerate(sorted(gains.items(), key=lambda kv: int(kv[0]))):
        pts = sorted((int(k), v) for k, v in byk.items())
        vals += [v for _, v in pts]
        ax.plot([k for k, _ in pts], [v for _, v in pts], color=fs.CONTEXT_INK, lw=0.8, ls=arena_line_style(i),
                marker="o", ms=2.5, mfc=fs.CONTEXT_INK, mec="white", mew=fs.MARKER_EDGE)
        ends.append((pts[-1][0], pts[-1][1], str(arena)))
    ax.axhline(0.0, color=fs.BLACK, lw=fs.REF_LW, zorder=1.5, gid="ref")
    ax.text(0.0, 0.0, " 0 updates", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=fs.ANNOT_PT, color=fs.INK, gid="decor")
    fs.end_labels(ax, ends, gap=0.16)
    ax.set_xlim(ks[0] / 1.3, ks[-1] * 1.6)
    ax.set_xlabel("adaptation episodes (log scale)")
    ax.set_ylabel("PSNR gain over\n0 updates (dB)")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.set_ylim(min(vals) - 0.3, max(vals) + 0.4)
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
        ax.text(0.98, 0.04, f"map {arena}", transform=ax.transAxes, ha="right", va="bottom", fontsize=fs.ANNOT_PT)
        ax.set_ylabel(fs.A_LABEL if i == 0 else "")
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


def fig_skill(records, home, budget, out_dir, band=None, x="S0", xlabel="zero-shot latent skill $S_0$ (dB)",
              stem="fig2c_outcomes_by_skill", size=None):
    """A at the budget against a per-arena predictor (`x`, default the zero-shot skill S0; the family-step tool
    passes the deficit Delta), one diamond per arena with its episode interval, every arena numbered by
    `figstyle.label_points` (coincident markers share a label); the training maps' band."""
    done = [r for r in records if r.get(x) is not None and r.get("A_budget") is not None]
    if not done:
        return []
    fig, (ax,) = fs.new_figure(size or HALF_SIZE)
    ax.set_gid("points")
    for r in done:
        ci = r.get("A_budget_ci")
        if ci:
            ax.plot([r[x], r[x]], ci, color=ADAPTER.colour, lw=fs.MIN_LW, zorder=3.5)
        ax.plot(r[x], r["A_budget"], ls="none", marker=ADAPTER.marker, ms=fs.MARKER_SIZE, mfc=ADAPTER.colour,
                mec="white", mew=fs.MARKER_EDGE, color=ADAPTER.colour, zorder=3)
    fs.training_line(ax, home, where=0.0, band=band)
    ax.set_xlabel(xlabel)
    lo = min(r["A_budget"] for r in done)
    top = max([r["A_budget"] for r in done] + [home] + ([band[1]] if band else []))
    ax.set_ylim(min(lo, home) - 0.5, top + 0.45)
    ax.set_ylabel(f"\u0394PSNR vs persistence\nat {budget_label(budget)} (dB)")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
    ax.margins(x=0.08)
    # nearby arenas are unequal observations: each keeps its own number, with a leader when it had to move
    fs.label_points(ax, [(r[x], r["A_budget"], str(r["arena"])) for r in done], merge=False, leaders=True)
    return fs.save(fig, out_dir, stem)


ABSOLUTE_SIZE = (3.15, 1.5)                  # beside 3a in one row, set unscaled (paper owner, 2026-09-27)
ABSOLUTE_WIDE_SIZE = (fs.TEXT_WIDTH, 2.1)     # the appendix's stock twin
ABSOLUTE_OFFSETS = {"unet": -0.3, "pixart": 0.0, "sd35": 0.3}    # 3.75 pt marks stay apart in a 0.19 in column
PERSISTENCE_INK = "#8C8C8C"
COLUMN_TINT = "#F4F4F4"


def fig_zero_shot_absolute(absolute, arenas, out_dir, decoder="tuned", stem="fig3b_zero_shot_absolute",
                           size=ABSOLUTE_SIZE, letter=None):
    """Figure 3b, absolute form: per unseen arena (arena-number order) scene PSNR (top, higher is better) and scene
    LPIPS (bottom, lower is better) against the raw true frame, five marks per arena: persistence as a grey bar
    across the column, the U-Net, PixArt-alpha and SD 3.5 (zero-shot) and the U-Net after 4k adapter updates; the
    training maps' pooled read in the first column under the grey band. `decoder` picks the decoder of the U-Net,
    PixArt and adapter marks (filled for the fine-tuned decoder, open for stock); SD 3.5 is always through its own
    stock decoder (open). Alternate columns are tinted so each arena's marks read as one group. The arrows in the
    y labels say which way is better, since a 0.5 in panel has no room for a note."""
    from matplotlib.lines import Line2D
    fig, (ax, lx) = fs.new_figure(size, nrows=2, sharex=True, h_pad=0.01, hspace=0.02)
    cols = ["train"] + list(arenas)
    train_w = 1.6
    xpos = {"train": train_w / 2}
    xpos.update({a: train_w + 0.5 + i for i, a in enumerate(arenas)})
    persistence_shown = False
    for panel, key in ((ax, "psnr"), (lx, "lpips")):
        fs.training_band(panel, 0.0, train_w)
        # the tint starts on the second arena, so the column beside the training maps' grey band stays white
        for i, a in enumerate(arenas):
            if i % 2 == 1:
                panel.axvspan(xpos[a] - 0.5, xpos[a] + 0.5, color=COLUMN_TINT, lw=0, zorder=0, gid="decor")
        for col in cols:
            half = (train_w / 2 if col == "train" else 0.5) - 0.08
            persist = None
            for backbone, entry in absolute.items():
                if backbone not in ABSOLUTE_OFFSETS:     # the adapter's endpoint lives in Figure 4 (round 2)
                    continue
                d = STOCK if backbone == "sd35" else decoder
                src = entry.get(d) or (entry.get(STOCK) if backbone == "sd35" else None)
                if not src or (backbone == "adapter" and col == "train"):
                    continue
                e = src["home"] if col == "train" else src["maps"].get(col)
                if not e or e.get(key) is None:
                    continue
                if persist is None and e.get(f"persist_{key}") is not None:
                    persist = e[f"persist_{key}"]
                ent = fs.BACKBONES[backbone]
                x = xpos[col] + ABSOLUTE_OFFSETS[backbone] * (train_w if col == "train" else 1.0)
                ci = e.get(f"{key}_ci")
                if ci:
                    panel.plot([x, x], ci, color=ent.colour, lw=fs.MIN_LW, zorder=4, solid_capstyle="butt")
                filled = d != STOCK
                panel.plot([x], [e[key]], ls="none", marker=ent.marker, ms=fs.MARKER_SIZE, mew=0.6, mec=ent.colour,
                           mfc=ent.colour if filled else "white", color=ent.colour, zorder=3)
            if persist is not None:
                panel.plot([xpos[col] - half, xpos[col] + half], [persist, persist], color=PERSISTENCE_INK, lw=1.3,
                           solid_capstyle="butt", zorder=2.5)
                persistence_shown = True
        panel.tick_params(axis="x", length=0)
    ax.set_ylabel("PSNR \u2191" if size[1] < 1.8 else "PSNR (dB) \u2191")     # dB in the caption at 0.5 in
    lx.set_ylabel("LPIPS \u2193")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2 if size[1] > 1.8 else 3))
    lx.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    lx.set_xticks([xpos[c] for c in cols], ["training\nmaps"] + [str(a) for a in arenas])
    for t in lx.xaxis.get_ticklabels()[:1]:
        t.set_fontsize(fs.MIN_PT)
        t.set_linespacing(0.9)
    lx.set_xlim(-0.1, xpos[arenas[-1]] + 0.6)
    lx.set_xlabel("unseen map")
    fill = decoder != STOCK
    handles = ([Line2D([], [], color=PERSISTENCE_INK, lw=1.3)] if persistence_shown else []) + [
        Line2D([], [], ls="none", marker=fs.BACKBONES[b].marker, ms=fs.MARKER_SIZE, mew=0.6,
               mec=fs.BACKBONES[b].colour, mfc=fs.BACKBONES[b].colour if (fill and b != "sd35") else "white",
               color=fs.BACKBONES[b].colour)
        for b in ("unet", "pixart", "sd35") if b in absolute]
    labels = (["persistence"] if persistence_shown else []) + [
        {"unet": "U-Net", "pixart": "PixArt-\u03b1", "sd35": "SD 3.5"}[b]      # open: its own decoder (caption)
        for b in ("unet", "pixart", "sd35") if b in absolute]
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), handlelength=1.4, columnspacing=1.1,
               handletextpad=0.4, borderaxespad=0.1)
    if letter:      # in the legend's row at the figure's top left, level with 3a's letter, not a row of its own
        fig.text(0.0, 1.0, letter, ha="left", va="top", fontsize=fs.LETTER_PT, fontweight="bold")
    return fs.save(fig, out_dir, stem)


FIG3B_SIZE = (3.3, 1.9)
PAIRED_OFFSETS = {"unet": -0.27, "pixart": -0.09, "sd35": 0.09, "adapter": 0.27}


def fig_zero_shot_paired(paired, order, out_dir):
    """Figure 3b: zero-shot A (top) and M (bottom) per unseen arena, arenas ordered by the U-Net's zero-shot skill
    S0 and the training maps' reads in a first column under the grey band; one colour and marker per backbone,
    open for the stock decoder and filled for the tuned one on the same windows, a thin grey segment joining each
    pair; episode intervals above the markers; the 4k adapter (tuned decoder) as its own diamond on the A row;
    copy-last as the zero line of both rows."""
    fig, (ax, mx) = fs.new_figure(FIG3B_SIZE, nrows=2, sharex=True, h_pad=0.01, hspace=0.02)
    train_w = 2.0
    xpos = {a: train_w + 0.5 + i for i, a in enumerate(order)}
    xpos["train"] = train_w / 2
    drawn = {"A": [], "M": []}
    for panel, key in ((ax, "A"), (mx, "M")):
        fs.training_band(panel, 0.0, train_w)
        for backbone, entry in paired.items():
            if backbone == "adapter":          # the adapter's endpoint lives in Figure 4 (round 2, Astra 13)
                continue
            ent = fs.BACKBONES[backbone]
            off = PAIRED_OFFSETS[backbone] * (1.6 if backbone != "adapter" else 1.0)
            decoders = ("tuned",) if backbone == "adapter" else ("stock", "tuned")
            for col in ["train"] + list(order):
                vals = {}
                for d in decoders:
                    src = entry.get(d)
                    if not src:
                        continue
                    e = src["home"] if col == "train" else src["maps"].get(col)
                    if e and e.get(key) is not None:
                        v, ci = e[key], e.get(f"{key}_ci")
                        if key == "M":
                            v, ci = fs.m_value(v), fs.m_interval(ci)
                        vals[d] = (v, ci)
                if not vals or (backbone == "adapter" and col == "train"):
                    continue
                x = xpos[col] + (off * (1.8 if col == "train" else 1.0))
                if len(vals) == 2:
                    panel.plot([x, x], [vals["stock"][0], vals["tuned"][0]], color=fs.FAINT, lw=fs.MIN_LW, zorder=2)
                for d, (v, ci) in vals.items():
                    if ci:
                        panel.plot([x, x], ci, color=ent.colour, lw=fs.MIN_LW, zorder=4, solid_capstyle="butt")
                    panel.plot([x], [v], ls="none", marker=ent.marker, ms=fs.MARKER_SIZE, mew=0.6, mec=ent.colour,
                               mfc=ent.colour if d == "tuned" else "white", color=ent.colour, zorder=3)
                    drawn[key].append(v)
        fs.copy_last_line(panel, where=0.0, align="left")
        panel.tick_params(axis="x", length=0)
    ax.set_ylabel(fs.A_LABEL_SHORT)
    mx.set_ylabel("$M$, LPIPS diff.")
    mx.text(0.995, 0.03, "higher is better" if fs.M_POSITIVE_IS_BETTER else "lower is better", transform=mx.transAxes,
            ha="right", va="bottom", fontsize=fs.ANNOT_PT, color=fs.CONTEXT_INK)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    mx.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_ylim(min(0.0, min(drawn["A"])) - 0.4, max(drawn["A"]) + 0.5)
    if drawn["M"]:
        mx.set_ylim(min(0.0, min(drawn["M"])) - 0.03, max(drawn["M"]) + 0.03)
    ticks = [xpos["train"]] + [xpos[a] for a in order]
    mx.set_xticks(ticks, ["training\nmaps"] + [str(a) for a in order])
    for t in mx.xaxis.get_ticklabels()[:1]:
        t.set_fontsize(fs.MIN_PT)
        t.set_linespacing(0.9)
    mx.set_xlim(-0.1, xpos[order[-1]] + 0.6 if order else train_w + 0.5)
    mx.set_xlabel("unseen map, by the U-Net's zero-shot skill $S_0$")
    return fs.save(fig, out_dir, "fig3b_zero_shot_paired")


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
        conv = (fs.m_value, fs.m_interval) if key == "M" else (lambda v: v, lambda c: c)
        pts = [(pos[a] + off, conv[0](maps[a][key]), conv[1](maps[a].get(f"{key}_ci"))) for a in arenas
               if a in maps and maps[a].get(key) is not None]
        for x, _v, ci in pts:
            if ci:
                ax.plot([x, x], ci, color=ent.colour, lw=0.7, zorder=4, solid_capstyle="butt")
        ax.plot([x for x, _, _ in pts], [v for _, v, _ in pts], ls="none", marker=ent.marker, ms=3.2,
                mfc=ent.colour if tuned else "white", mec=ent.colour, mew=0.7, zorder=3)
        home = zero_shot[name].get("home") if key == "A" else fs.m_value(zero_shot[name].get("home_M"))
        if home is not None:
            ax.plot([len(arenas) - 0.3, len(arenas) + 0.3], [home, home], color=ent.colour, lw=fs.REF_LW,
                    ls=fs.TRAINING_DASH, zorder=2, clip_on=False)
    if zero:
        fs.copy_last_line(ax, where=0.0, align="left")
    ax.set_xticks(range(len(arenas)), [str(a) for a in arenas])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(arenas) + 0.4)
    ax.set_xlabel("unseen map, by zero-shot skill $S_0$")
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
    """Fill each record's M and G at the budget from the adapter's raw-frame read at that step
    (`adapt<budget>_...`) when the adaptation rows carry neither (`heldout_B`, `heldout_C`); each is filled
    independently, and only when no record has it."""
    for key, column in (("M", "heldout_B"), ("G", "heldout_C")):
        if any(r[f"{key}_budget"] is not None for r in records):
            continue
        for name, row in zero_shot.items():
            m = ADAPTER_ROW_RE.match(name)
            if not m or int(m["step"]) != budget:
                continue
            filled = 0
            for r in records:
                e = row["maps"].get(r["arena"])
                if e and e.get(key) is not None:
                    r[f"{key}_budget"] = e[key]
                    if key == "M":
                        r["M_budget_ci"] = e.get("M_ci")
                    filled += 1
            if filled:
                notes.append(f"{key} at {budget}: the adapter rows carry no {column}; read from {row['source']} "
                             f"({filled} unseen maps)")
            break


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
    p.add_argument("--fixed-threshold", type=float, default=FIXED_THRESHOLD,
                   help="the fixed A threshold (dB) of the second attainment curve")
    p.add_argument("--fresh-root", default=os.path.join(REPO, "results", "fresh_rescore"),
                   help="<root>/<row>/<map dir>/metrics.json zero-shot reads; each row a marker per arena")
    p.add_argument("--zero-shot", action="append", default=[], metavar="NAME=DIR",
                   help="one more zero-shot row: a directory of eval_tf reads keyed by map (repeatable)")
    p.add_argument("--with-raw", action="store_true", help="also draw the margin M per arena, where rows carry it")
    p.add_argument("--prov", action="store_true", help="wrap every table number in \\prov{} (provisional marks)")
    p.add_argument("--bootstrap", type=int, default=2000, help="episode-bootstrap draws per arena interval (0: none)")
    p.add_argument("--arena-bootstrap", type=int, default=10000,
                   help="nested (arena, then episode) bootstrap draws for the across-arena statistics (0: none)")
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
        rec = run_record(run, a.decoder, home, budget, D, draws, a.seed, notes, a.fixed_threshold)
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
    paired = paired_zero_shot(a.fresh_root, a.bootstrap, a.seed, notes)
    absolute = absolute_zero_shot(a.fresh_root, a.bootstrap, a.seed, notes)
    s0_order = [r["arena"] for r in sorted(records, key=lambda r: (r["S0"] is None, -(r["S0"] or 0), r["arena"]))]
    margins_from_adapter_row(records, zero_shot, budget, notes)
    summary = build_summary(records, budget, home)
    seed_summary = seed_entries(anchors, seeds, records_of, budget, a.headline_variant)
    ladder_summary = ladder_entries(anchors, ladder, records_of, budget)
    stats = across_arenas(records, by_name, a.decoder, home, budget, a.arena_bootstrap, a.seed, notes,
                          a.fixed_threshold) if a.arena_bootstrap else None
    home_ci = home_interval(a.home_json, a.decoder, max(a.bootstrap, 10000), a.seed)
    when = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    used = [base[a_].path for a_ in sorted(base)]
    inputs = [{"path": rel(p), "sha256": sha256(p)} for p in sorted({r.path for r in runs}) + [a.distances]]
    summary = {"generated_by": "paper/make_adapt_figures.py", "generated_at": when, "decoder": a.decoder,
               "home": home, "home_ci": home_ci,
               "home_source": rel(home_source) if os.path.exists(home_source) else home_source,
               "budget": budget, "headline_variant": a.headline_variant, "weights": a.weights, **summary,
               "across_arenas": public(stats),
               "per_arena": records, "seeds": seed_summary, "ladder": ladder_summary,
               "ladder_gain": ladder_gains(ladder_summary, {r["arena"]: r["A0"] for r in records}),
               "recipe": recipe_entries(anchors, base, recipe, records_of, seed_summary),
               "zero_shot": {k: {**v, "maps": without_paths(v["maps"])} for k, v in zero_shot.items()},
               "zero_shot_absolute": {k: {d: {"source": e[d]["source"], "home": e[d]["home"],
                                              "maps": {str(m): v for m, v in e[d]["maps"].items()}}
                                          for d in e} for k, e in absolute.items()},
               "zero_shot_paired": {"order": s0_order, "rows": {
                   k: {d: {"source": e[d]["source"], "home": e[d]["home"], "maps": without_paths(e[d]["maps"])}
                       for d in ("stock", "tuned") if d in e} for k, e in paired.items()}},
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
        "brackets: 95% intervals, nested (arena, then episode) bootstrap for aggregates, episode bootstrap per arena",
    ] + [f"  {rel(p)} sha256 {sha256(p)[:16]}" for p in used])
    written = []
    tables = [("adapt_cost.tex", cost_table(records, summary, a.decoder, budget, a.prov, public(stats))),
              ("adapt_perarena_A.tex", perarena_table(records, budget, a.prov)),
              ("adapt_table3.tex", table3(records, summary, public(stats), home, home_ci, budget, a.prov))]
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
    written += draw_figures(a, stats, records, by_name, home, budget, seed_summary, anchors, seeds, ladder_summary,
                            recipe, base, records_of, zero_shot, notes, band=(home_ci or {}).get("ci"),
                            paired=paired, s0_order=s0_order, absolute=absolute)
    for p in written:
        print("wrote", rel(p))
    for n in notes:
        print("note:", n)
    return 0


def draw_figures(a, stats, records, by_name, home, budget, seed_summary, anchors, seeds, ladder_summary, recipe,
                 base, records_of, zero_shot, notes, band=None, paired=None, s0_order=(), absolute=None):
    """Every figure the data support; a figure the data cannot fill is refused by `figstyle.save` and noted.
    `band` is the training maps' 95% episode interval, drawn around their dashed reference line."""
    written = []

    def attempt(fn, *args, **kw):
        try:
            return fn(*args, **kw)
        except fs.DegenerateFigure as e:
            notes.append(f"not drawn: {e}")
            return []

    if stats:
        written += attempt(fig_adaptation, stats, records, home, budget, a.out_dir, a.fixed_threshold, band=band)
        written += attempt(fig_gapshare, stats, home, a.out_dir)
        written += attempt(fig_dots, stats, records, home, a.out_dir, band=band)
        written += attempt(fig_profiles, stats, home, a.out_dir, band=band)
    complete = [r for r in records if not r["incomplete"]]
    if complete:
        written += attempt(fig_arenas, complete, by_name, home, budget, a.out_dir, band=band)
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
    order = [a_ for a_ in s0_order if any(a_ in e.get(d, {}).get("maps", {}) for e in (paired or {}).values()
                                          for d in ("stock", "tuned"))]
    if order:
        written += attempt(fig_zero_shot_paired, paired, order, a.out_dir)
    numbered = sorted({m for e in (absolute or {}).values() for d in e.values() for m in d["maps"]} -
                      set(TRAINING_MAPS))
    if numbered:
        written += attempt(fig_zero_shot_absolute, absolute, numbered, a.out_dir, "tuned", "fig3b_zero_shot_absolute",
                           ABSOLUTE_SIZE, "b")
        written += attempt(fig_zero_shot_absolute, absolute, numbered, a.out_dir, STOCK,
                           "fig3b_zero_shot_absolute_stock", ABSOLUTE_WIDE_SIZE)
    if zero_shot:
        order = [m for m in TRAINING_MAPS if any(m in v["maps"] for v in zero_shot.values())] + list(s0_order)
        written += attempt(fig_zero_shot, zero_shot, order, "A", a.out_dir, "fig2a_advantage_by_distance",
                           fs.A_LABEL)
        if a.with_raw and any(e.get("M") is not None for v in zero_shot.values() for e in v["maps"].values()):
            written += attempt(fig_zero_shot, zero_shot, order, "M", a.out_dir, "fig2b_margin_by_distance",
                               fs.m_label(), zero=True)
    return written


if __name__ == "__main__":
    sys.exit(main())
