"""
The distance study's statistics, appendix table and figure (memo section 6 item 3).

    python paper/make_distance_figure.py --distances results/distance_study/distances_sd1.json \
        --row unet=$D/results_spiderman/distance_study/040-unet-nexttic/snap_0200000_ema_ddim10 \
        [--row sd35=...] [--space pixels=results/distance_study/distances_pixels.json] \
        [--space sd35=results/distance_study/distances_sd35.json] --out paper/figures/distance

Inputs are the files the pipeline writes: `distances_<space>.json` (and `bootstrap_<space>.npz` beside
it) from `distance_study.py distances`, and one score directory per row from `score_distance_maps.sh`,
holding `<set>_map<NN>_h1/` and `_h4/` with `eval_tf.py`'s `metrics.json` and `per_window.csv`. The
first `--row` is the primary row (U-Net 200k EMA) unless `--primary-row` names another.

Outcome per map: the gain over its own persistence reference at one tic, mean(psnr_raw -
persist_psnr_raw) over the map's 256 windows, which is the log persistence-to-model MSE ratio per
window, so the map's motion level cancels to first order. The motion covariate is the map's mean
persistence PSNR.

The one pre-declared primary test: the partial Spearman correlation of D with the gain, controlling
for persistence PSNR (rank residuals), SD 1.x space, primary row, one tic. Its 95% CI is a case
bootstrap over maps (10,000 draws) with episodes resampled within each drawn map and D taken from
that map's distance-bootstrap draws; its p-value is a Freedman-Lane permutation of the gain's rank
residuals (refitted on the covariate for every permutation since Sep 26 2026; the pre-declared code
permuted the residuals without the refit, Kennedy's variant, while its docstring named Freedman-Lane).
Around it: the leave-one-map-out sign check, the same statistic within each cluster (arenas, campaign
maps), for every row and in every other feature space (with Kendall's tau between the spaces' map
rankings), the raw correlations the "does not support" list needs, and the secondary outcomes (LPIPS
gain, four-tic gain on four-tic persistence at the full bootstrap, the within-map episode correlation).

The "every row" condition needs every row of `--expect-rows` (unet, sd35, pixart); one row supplied is
not every row. Beside the pre-declared verdict, which stays as it was, `amended_assessment` is the dated
reading of Astra's Sep 26 review, specified after the results were seen: it separates the measurement
checks (i, iii, v) from the map assumption (iv) and the subset tolerance (ii), and reports the pooled
association against the structural controls (training-map membership, map family, unseen maps only,
n_eff, bootstrap SD). The amendment draws from its own random stream, so no pre-declared number moves.

Writes `stats.json`, `distance_table.md` (the appendix table) and the figure in two themes, each with
its own validated steps: `distance_gain.{png,svg}` on the light surface and `distance_gain_dark.{png,svg}`
on the dark one. Up to three rows share one panel (the categorical slots that pass every all-pairs
colour check in both themes); more rows are drawn as one panel per row. Training maps are filled
markers, unseen maps open ones, and arena 7 (the same-WAD arena farther than most campaign maps) is named.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from distance_study import _jsonable, kendall_tau, rank_average, spearman  # noqa: E402,F401

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "band": "#e1e0d9",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "band": "#2c2c2a",
             "series": ["#3987e5", "#d95926", "#199e70"]},
}
MARKERS = {"arena": "o", "campaign": "s"}
NAMED_ARENAS = {7: "arena 7"}      # check (iv)'s counterexample: a training-WAD arena farther than most campaign maps
TAU_AGREE = 0.6            # memo section 5: map rankings agree between feature spaces
EXPECTED_ROWS = "unet,sd35,pixart"
ROW_NAMES = {"unet": "U-Net", "sd35": "SD 3.5", "pixart": "PixArt"}
AMENDED_ON = "2026-09-26"
MEASUREMENT_CHECKS = ("i", "iii", "v")


# ---------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------

def _residual(r, z):
    """Residual of ranks r on ranks z, by least squares with an intercept."""
    zc = z - z.mean()
    return r - r.mean() - (np.dot(zc, r - r.mean()) / np.dot(zc, zc)) * zc


def _covariate_ranks(z):
    """Average ranks of each covariate, z of shape (n,) or (n, k); constant columns are no covariate."""
    z = np.asarray(z, dtype=np.float64)
    cols = [rank_average(c) for c in (z[:, None] if z.ndim == 1 else z).T]
    return [c for c in cols if np.ptp(c) > 0]


def _residual_on(r, cols):
    """Residual of ranks r on an intercept and the covariate rank columns, by least squares."""
    if not cols:
        return r - r.mean()
    if len(cols) == 1:
        return _residual(r, cols[0])
    Z = np.column_stack([np.ones(len(r))] + cols)
    return r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]


def _rank_residuals(x, y, z):
    cols = _covariate_ranks(z)
    return _residual_on(rank_average(x), cols), _residual_on(rank_average(y), cols), cols


def partial_spearman(x, y, z):
    """Spearman correlation of x and y with the covariate(s) z partialled out of both ranks.

    z is one covariate (n,) or several (n, k); with every column constant this is the plain Spearman.
    """
    if not _covariate_ranks(z):
        return spearman(x, y)
    ex, ey, _ = _rank_residuals(x, y, z)
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def freedman_lane_null(x, y, z, perms):
    """|Partial Spearman| of x and y given z under each permutation in `perms`, by Freedman and Lane.

    The outcome's rank residuals on the covariates are permuted, added back to the covariates' fitted
    values and the model is refitted, which amounts to residualising the permuted residuals again before
    they are correlated. Without that refit (Kennedy's variant) a permuted outcome keeps a component along
    the covariates, every permutation statistic comes out too small and the test rejects too often; with
    several covariates on few maps the excess is large (test_distance_figure.py plants such a null).
    """
    ex, ey, cols = _rank_residuals(x, y, z)
    return np.array([abs(np.corrcoef(ex, _residual_on(ey[p], cols))[0, 1]) for p in perms])


def permutation_p(x, y, z, n, rng):
    """Two-sided Freedman-Lane permutation p-value of `partial_spearman(x, y, z)` over n permutations."""
    ex, ey, _ = _rank_residuals(x, y, z)
    obs = abs(np.corrcoef(ex, ey)[0, 1])
    null = freedman_lane_null(x, y, z, (rng.permutation(len(ey)) for _ in range(n)))
    return float((1 + np.sum(null >= obs - 1e-12)) / (n + 1))


def case_bootstrap(stat, D, Ddraws, gain, gain_boot, persist, persist_boot, draws, rng, extra=None):
    """Percentile 95% CI of stat(D, gain, persist) over a case bootstrap of maps.

    Each draw resamples maps with replacement; a drawn map takes its gain and persistence from draw b of
    its own episode bootstrap when there is one, and its D from a random draw of its distance bootstrap
    when there is one, so all three sources of error propagate. `extra`, an (n, k) array of further
    per-map covariates, travels with its map and is passed to stat beside persistence.
    """
    n = len(D)
    vals = np.empty(draws)
    for b in range(draws):
        pick = rng.integers(0, n, n)
        d = D[pick] if Ddraws is None else Ddraws[pick, rng.integers(0, Ddraws.shape[1], n)]
        g = gain[pick] if gain_boot is None else gain_boot[pick, b % gain_boot.shape[1]]
        p = persist[pick] if persist_boot is None else persist_boot[pick, b % persist_boot.shape[1]]
        vals[b] = stat(d, g, p if extra is None else np.column_stack([p, extra[pick]]))
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return [None, None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def correlation_block(D, Ddraws, gain, gain_boot, persist, persist_boot, draws, perms, rng, partial=True,
                      extra=None):
    """The partial (or plain) Spearman correlation with its bootstrap CI and permutation p-value.

    `extra` (n, k) adds covariates beside persistence (partial only)."""
    if len(D) < 4:
        return {"n_maps": int(len(D)), "partial_spearman": None, "ci95": [None, None], "permutation_p": None}
    stat = partial_spearman if partial else (lambda d, g, p: spearman(d, g))
    z = (persist if extra is None else np.column_stack([persist, extra])) if partial else np.ones(len(D))
    return {"n_maps": int(len(D)), "partial_spearman": stat(D, gain, z if partial else persist),
            "ci95": case_bootstrap(stat, D, Ddraws, gain, gain_boot, persist, persist_boot, draws, rng,
                                   extra if partial else None),
            "permutation_p": permutation_p(D, gain, z, perms, rng)}


def within_map_episode_rho(pairs, perms, rng):
    """Episode level, within maps: Spearman of map-centred ranks of (D_e, gain_e), permuted within maps."""
    groups = [(np.asarray(d), np.asarray(g)) for d, g in pairs if len(d) >= 3]
    if not groups:
        return None

    def centred(gs):
        xs = np.concatenate([rank_average(d) - rank_average(d).mean() for d, _ in gs])
        ys = np.concatenate([rank_average(g) - rank_average(g).mean() for _, g in gs])
        return float(np.corrcoef(xs, ys)[0, 1])
    obs = centred(groups)
    null = [abs(centred([(d, g[rng.permutation(len(g))]) for d, g in groups])) for _ in range(perms)]
    return {"episodes": int(sum(len(d) for d, _ in groups)), "maps": len(groups), "rho": obs,
            "permutation_p": float((1 + np.sum(np.array(null) >= abs(obs) - 1e-12)) / (perms + 1))}


# ---------------------------------------------------------------------------------------------
# reading scores
# ---------------------------------------------------------------------------------------------

def score_dir(root, point, horizon):
    return os.path.join(root, f"{point['set']}_map{int(point['map']):02d}_h{horizon}")


def read_map_score(d, draws, rng):
    """One map's outcomes from a score directory, with an episode bootstrap when per-window rows exist."""
    mpath = os.path.join(d, "metrics.json")
    if not os.path.isfile(mpath):
        return None
    with open(mpath) as f:
        met = json.load(f)

    def mean(k):
        v = met.get(k)
        return float(v["mean"]) if isinstance(v, dict) and v.get("mean") is not None else None
    raw, pers = mean("psnr_raw"), mean("persist_psnr_raw")
    if raw is None or pers is None:
        return None
    lp, plp = mean("lpips_raw"), mean("persist_lpips_raw")
    out = {"gain": raw - pers, "persist": pers, "raw_psnr": raw,
           "lpips_gain": plp - lp if lp is not None and plp is not None else None,
           "gain_boot": None, "persist_boot": None, "episodes": None}
    wpath = os.path.join(d, "per_window.csv")
    if os.path.isfile(wpath):
        with open(wpath) as f:
            blank = (None, "", "nan")
            rows = [r for r in csv.DictReader(f)
                    if r.get("psnr_raw") not in blank and r.get("persist_psnr_raw") not in blank]
        if rows:
            ep = np.array([int(r["episode"]) for r in rows])
            g = np.array([float(r["psnr_raw"]) - float(r["persist_psnr_raw"]) for r in rows])
            p = np.array([float(r["persist_psnr_raw"]) for r in rows])
            out.update(gain=float(g.mean()), persist=float(p.mean()), raw_psnr=float((g + p).mean()), windows=len(rows))
            if all(r.get("lpips_raw") not in (None, "") and r.get("persist_lpips_raw") not in (None, "") for r in rows):
                out["lpips_gain"] = float(np.mean([float(r["persist_lpips_raw"]) - float(r["lpips_raw"])
                                                   for r in rows]))
            uniq, inv = np.unique(ep, return_inverse=True)
            sg, sp, cnt = (np.bincount(inv, weights=w, minlength=len(uniq)) for w in (g, p, np.ones(len(g))))
            idx = rng.integers(0, len(uniq), size=(draws, len(uniq)))
            out["gain_boot"] = sg[idx].sum(1) / cnt[idx].sum(1)
            out["persist_boot"] = sp[idx].sum(1) / cnt[idx].sum(1)
            out["episodes"] = {int(e): float(sg[i] / cnt[i]) for i, e in enumerate(uniq)}
    return out


# ---------------------------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------------------------

def marker_style(point, colour, theme):
    """Filled markers for training maps and open ones (the surface colour inside a coloured ring) for
    unseen maps, so the reader sees how much of the trend the four training maps carry."""
    if point.get("training"):
        return {"color": colour, "mfc": colour, "mec": theme["surface"], "mew": 0.8}
    return {"color": colour, "mfc": theme["surface"], "mec": colour, "mew": 1.1}


def draw_figure(points, rows, primary_stats, floor, out_dir, theme_name):
    """Gain over persistence against distance, one marker per map, in one theme. Returns the panel count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    t = THEMES[theme_name]
    names = list(rows)
    facets = len(names) > len(t["series"])
    n_pan = len(names) if facets else 1
    ncols = 2 if facets else 1
    nrows = int(np.ceil(n_pan / ncols))
    # the legend band below the plot: the rows (one panel only) beside the keys, which are the two
    # clusters, training and unseen, the floor band when there is one, and persistence
    n_keys = 5 + bool(floor)
    band_in = 0.12 + 0.145 * max(0 if facets else len(names), int(np.ceil(n_keys / (2 if facets else 1))))
    # the plot area stays 2.5 in tall per row of panels whatever the legend needs
    size = (6.8, 2.5 * nrows + max(0.6, band_in)) if facets else (3.4, 2.5 + band_in)
    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "svg.fonttype": "none"})
    fig, axes = plt.subplots(nrows, ncols, figsize=size, sharex=True, sharey=True, squeeze=False,
                             facecolor=t["surface"])
    axes = axes.ravel()
    for ax in axes:
        ax.set_facecolor(t["surface"])
        ax.grid(True, color=t["grid"], lw=0.5, ls="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(t["axis"])
            ax.spines[side].set_linewidth(0.6)
        ax.tick_params(colors=t["muted"], labelcolor=t["secondary"], width=0.6, length=3)
        if floor:
            ax.axvspan(floor[0], floor[1], color=t["band"], lw=0, zorder=0)
        ax.axhline(0.0, color=t["muted"], lw=0.8, ls=(0, (4, 3)), zorder=1)
    for i, name in enumerate(names):
        ax = axes[i if facets else 0]
        colour = t["series"][0 if facets else i]
        for p in points:
            s = p["rows"].get(name)
            if s is None:
                continue
            xerr = None if p.get("D_ci") is None else [[p["D"] - p["D_ci"][0]], [p["D_ci"][1] - p["D"]]]
            yerr = None if s.get("gain_ci") is None else [[s["gain"] - s["gain_ci"][0]], [s["gain_ci"][1] - s["gain"]]]
            if xerr is not None or yerr is not None:
                ax.errorbar(p["D"], s["gain"], xerr=xerr, yerr=yerr, fmt="none", ecolor=colour, elinewidth=0.7,
                            alpha=0.55, capsize=0, zorder=2)
            ax.plot(p["D"], s["gain"], MARKERS[p["cluster"]], ms=6, zorder=3, **marker_style(p, colour, t))
            label = NAMED_ARENAS.get(p["map"]) if p["cluster"] == "arena" else None
            if label:
                ax.annotate(label, (p["D"], s["gain"]), xytext=(5, 4), textcoords="offset points", fontsize=6.5,
                            color=t["secondary"], zorder=4)
        if facets:
            st = primary_stats[name]
            ax.set_title(f"{name}   partial ρ {_fmt(st['partial_spearman'])} {_fmt_ci(st['ci95'])}",
                         color=t["ink"], fontsize=8, loc="left")
    for ax in axes[n_pan:]:
        ax.set_visible(False)
    for ax in axes[:n_pan]:
        if ax in axes[(nrows - 1) * ncols:] or not facets:
            ax.set_xlabel("Distance to the nearest training map (SW₂)", color=t["secondary"])
    for r in range(nrows):
        axes[r * ncols].set_ylabel("Gain over persistence, 1 tic (dB)", color=t["secondary"])
    # two legends below the plot: the rows with their partial rho (one panel only), and the keys
    row_handles = [] if facets else [
        Line2D([], [], marker="o", ls="", color=t["series"][i], mec=t["surface"], ms=6,
               label=f"{name}  partial ρ {_fmt(primary_stats[name]['partial_spearman'])} "
                     f"{_fmt_ci(primary_stats[name]['ci95'])}")
        for i, name in enumerate(names)]
    keys = [Line2D([], [], marker=MARKERS[c], ls="", color=t["muted"], ms=5, label=c) for c in ("arena", "campaign")]
    filled = {**marker_style({"training": True}, t["muted"], t), "mec": t["muted"]}
    keys += [Line2D([], [], marker="o", ls="", ms=5, label="training map", **filled),
             Line2D([], [], marker="o", ls="", ms=5, label="unseen map", **marker_style({}, t["muted"], t))]
    if floor:
        keys.append(Patch(color=t["band"], label="train vs train floor"))
    keys.append(Line2D([], [], color=t["muted"], lw=0.8, ls=(0, (4, 3)), label="persistence"))
    if row_handles:
        fig.legend(handles=row_handles, loc="lower left", frameon=False, fontsize=7, bbox_to_anchor=(0.0, 0.0),
                   labelcolor=t["ink"], handletextpad=0.3)
        fig.legend(handles=keys, loc="lower right", frameon=False, fontsize=7, bbox_to_anchor=(1.0, 0.0),
                   labelcolor=t["ink"], handletextpad=0.3)
    else:
        fig.legend(handles=keys, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0),
                   labelcolor=t["ink"])
    fig.tight_layout(rect=(0, band_in / size[1], 1, 1))
    stem = "distance_gain" + ("" if theme_name == "light" else "_dark")
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), dpi=300, facecolor=t["surface"], bbox_inches="tight",
                    pad_inches=0.03)
    plt.close(fig)
    return n_pan


def _fmt(v, digits=2):
    """A signed value with the typographic minus the axis ticks use."""
    return "n/a" if v is None or not np.isfinite(v) else f"{v:+.{digits}f}".replace("-", "\u2212")


def _fmt_ci(ci):
    return "" if not ci or ci[0] is None else f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------

def _bootstrap_beside(path, name):
    """{map key: draws} from an npz beside the distances file, or None."""
    boot = os.path.join(os.path.dirname(os.path.abspath(path)), name)
    if not os.path.isfile(boot):
        return None
    with np.load(boot) as z:
        return {str(k): row for k, row in zip(z["keys"], z["draws"])}


def load_distances(path):
    with open(path) as f:
        js = json.load(f)
    prim = [m for m in js["maps"] if m.get("role", "primary") == "primary"]
    return js, prim, _bootstrap_beside(path, f"bootstrap_{js.get('space')}.npz")


def load_full_bootstrap(path, space):
    """The bootstrap of the full subset-averaging estimator (`distance_study.py` writes it from Sep 26 2026
    as bootstrap_full_<space>.npz, beside the single-cloud bootstrap it keeps for the frozen files)."""
    return _bootstrap_beside(path, f"bootstrap_full_{space}.npz")


def is_training_map(point, js):
    """Whether the model trained on this map: one of the distances file's reference maps, or, in a file
    that names none, a map of its validation set (the held-out episodes of the training maps)."""
    ref = (js.get("references") or {}).get("maps")
    if ref:
        return int(point["map"]) in {int(m) for m in ref}
    return point["set"] == (js.get("config") or {}).get("validation_set", "val")


def replay_predeclared_h4_draws(rng, s4):
    """Advance the shared stream exactly as the pre-declared run did when it read the four-tic scores.

    That run took a 2-draw episode bootstrap of the four-tic read from the shared stream and never used
    it. Drawing the same numbers keeps every later pre-declared number, the primary CI among them,
    reproducible bit for bit now that the four-tic bootstrap runs at the full count on its own stream.
    """
    if s4 is not None and s4.get("episodes"):
        k = len(s4["episodes"])
        rng.integers(0, k, size=(2, k))


# name: (maps, covariates beside persistence PSNR), the structural controls of Astra's Sep 26 review
STRUCTURAL_CONTROLS = {
    "training_indicator": ("all", ("training",)),
    "training_indicator_family": ("all", ("training", "campaign")),
    "unseen_only_family": ("unseen", ("campaign",)),
    "n_eff": ("all", ("n_eff",)),
    "bootstrap_sd": ("all", ("D_sd",)),
}


def map_covariate(point, name):
    """A map-level covariate: the training-map and campaign-family indicators, or a recorded value."""
    if name == "training":
        return float(point["training"])
    if name == "campaign":
        return float(point["cluster"] == "campaign")
    v = point.get(name)
    return None if v is None else float(v)


def amended_assessment(js, points, primary_row, result, cond, measure):
    """The dated amendment that sits beside the pre-declared verdict, which it never changes.

    Specified on Sep 26 2026 after the results were seen (Astra's review). It reads the distance checks
    in three groups: the measurement checks (i, iii, v), which say whether D measures what it should; the
    map assumption (iv), an expectation about which maps are near, which arena 7 contradicts; and the
    subset tolerance (ii), judged on single 4-episode subsets while D averages ten. It then asks whether
    the pooled association survives the structure of the maps: training-map membership, map family and
    the unseen maps alone. `measure(points, covariates, draws_key)` returns the primary row's correlation
    block on those maps with those covariates beside persistence, from the amendment's own random stream.
    """
    checks = js.get("checks") or {}

    def passed(c):
        v = checks.get(c)
        return v.get("pass") if isinstance(v, dict) else None
    results = {c: passed(c) for c in MEASUREMENT_CHECKS}
    vals = list(results.values())
    meas = False if False in vals else None if None in vals else True
    iv = checks.get("iv") if isinstance(checks.get("iv"), dict) else {}
    far = iv.get("max_same_wad")
    camp = [p for p in points if p["cluster"] == "campaign"]
    map_assumption = {"check": "iv", "pass": passed("iv"), "max_same_wad": far,
                      "campaign_maps_nearer": None if far is None else int(sum(p["D"] < far for p in camp)),
                      "campaign_maps": len(camp),
                      "reading": "an expectation about the maps, not a test that D measures visual distance"}
    subset_tolerance = {"check": "ii", "pass": passed("ii"),
                        "reading": "judged on pairs of single 4-episode subsets; the reported D is the mean of "
                                   "ten, so the tolerance bounds a noisier quantity than the one reported"}

    full = bool(points) and all("_Dfull" in p for p in points)
    dkey = "_Dfull" if full else "_Ddraws"
    controls = {}
    for name, (which, covs) in STRUCTURAL_CONTROLS.items():
        pts = [p for p in points if which == "all" or not p["training"]]
        rec = {"maps": which, "covariates": ["persistence PSNR", *covs]}
        lacking = [c for c in covs if any(map_covariate(p, c) is None for p in pts)]
        controls[name] = {**rec, "partial_spearman": None, "note": f"no {lacking[0]} for some maps"} if lacking \
            else {**rec, **measure(pts, covs, dkey)}
    uci = controls["unseen_only_family"].get("ci95") or [None, None]
    unseen_below = None if uci[1] is None else bool(uci[1] < 0)
    conditions = {k: {"basis": "pre-declared", "value": cond[c]} for k, c in (
        ("primary_ci_excludes_zero", "negative_with_ci_excluding_zero"), ("leave_one_out_sign", "leave_one_out_sign"),
        ("within_both_clusters", "within_both_clusters"), ("pixel_space", "pixel_space"))}
    conditions.update({"measurement_checks_i_iii_v": {"basis": "amended", "value": meas},
                       "unseen_only_family_ci_below_zero": {"basis": "amended", "value": unseen_below},
                       "every_expected_row_negative": {"basis": "amended", "value": cond["every_row"]},
                       "bootstrap_covers_full_estimator": {"basis": "amended", "value": full}})

    who = ROW_NAMES.get(primary_row, primary_row)
    pooled = cond["negative_with_ci_excluding_zero"]
    head = (f"pooled {who} association not computed" if pooled is None
            else f"no evidence for the pooled {who} association" if not pooled
            else f"{'evidence' if unseen_below else 'qualified evidence'} for the pooled {who} association")
    gate = {"does not support": "original gate failed", "supports": "original gate passed"}.get(
        result, "original gate incomplete")
    trend = ("unseen-only trend not computed" if uci[1] is None else "unseen-only trend holds" if uci[1] < 0
             else "unseen-only trend reverses" if uci[0] > 0 else "unseen-only trend inconclusive")
    backbone = "cross-backbone replication " + {True: "holds", False: "fails"}.get(cond["every_row"], "pending")
    parts = [head, gate, trend, backbone]
    if meas is not True:
        parts.insert(0, "a measurement check (i, iii, v) failed" if meas is False
                     else "measurement checks not recorded")
    out = {"date": AMENDED_ON,
           "status": "specified after observing results (Astra's review, Sep 26 2026); the pre-declared "
                     "verdict above is unchanged",
           "original_verdict": result,
           "measurement_checks": {"checks": list(MEASUREMENT_CHECKS), "results": results, "pass": meas},
           "map_assumption": map_assumption, "subset_tolerance": subset_tolerance,
           "conditions": conditions, "structural_controls": controls}
    if full:
        out["primary_with_full_bootstrap"] = measure(points, (), "_Dfull")
    out["summary"] = "; ".join(parts)
    return out


def main(argv=None):
    a = build_parser().parse_args(argv)
    rng = np.random.default_rng(a.seed)
    # the four-tic bootstrap and the amended assessment each draw from their own stream, so neither moves
    # a pre-declared number
    rng_h4 = np.random.default_rng([a.seed, 4])
    rng_amended = np.random.default_rng([a.seed, 20260926])
    js, prim, ddraws = load_distances(a.distances)
    full_draws = load_full_bootstrap(a.distances, js.get("space"))
    rows = dict(r.split("=", 1) for r in a.row)
    if not rows:
        raise SystemExit("name at least one scored row as --row LABEL=DIR")
    primary_row = a.primary_row or next(iter(rows))
    if primary_row not in rows:
        raise SystemExit(f"--primary-row {primary_row} is not one of {list(rows)}")
    os.makedirs(a.out, exist_ok=True)

    # per map, per row: outcomes and their episode bootstraps
    points = []
    for m in prim:
        p = {"key": m["key"], "set": m["set"], "map": int(m["map"]), "cluster": m.get("cluster", "arena"),
             "training": is_training_map(m, js),
             "D": float(m["D"]), "nearest": m.get("nearest"), "n_eff": m.get("n_eff"), "lives": m.get("lives"),
             "valid_fraction": m.get("valid_fraction"), "rows": {}, "_boot": {}, "_boot4": {}}
        if ddraws is not None and m["key"] in ddraws:
            p["_Ddraws"] = np.asarray(ddraws[m["key"]], dtype=np.float64)
            p["D_ci"] = [float(np.percentile(p["_Ddraws"], 2.5)), float(np.percentile(p["_Ddraws"], 97.5))]
            p["D_sd"] = float(np.std(p["_Ddraws"], ddof=1))
        if full_draws is not None and m["key"] in full_draws:
            p["_Dfull"] = np.asarray(full_draws[m["key"]], dtype=np.float64)
        for name, root in rows.items():
            s1 = read_map_score(score_dir(root, m, 1), a.draws, rng)
            if s1 is None:
                continue
            s4 = read_map_score(score_dir(root, m, 4), a.draws, rng_h4)
            replay_predeclared_h4_draws(rng, s4)
            rec = {"gain": s1["gain"], "persist": s1["persist"], "raw_psnr": s1["raw_psnr"],
                   "lpips_gain": s1["lpips_gain"], "gain_h4": s4["gain"] if s4 else None,
                   "persist_h4": s4["persist"] if s4 else None, "windows": s1.get("windows")}
            if s1["gain_boot"] is not None:
                rec["gain_ci"] = [float(np.percentile(s1["gain_boot"], q)) for q in (2.5, 97.5)]
            if s4 and s4["gain_boot"] is not None:
                rec["gain_h4_ci"] = [float(np.percentile(s4["gain_boot"], q)) for q in (2.5, 97.5)]
            p["rows"][name] = rec
            p["_boot"][name] = s1
            p["_boot4"][name] = s4
        points.append(p)

    def arrays(name, pts, key="gain", dkey="_Ddraws"):
        """The maps with this outcome, and its arrays; the four-tic gain goes with four-tic persistence."""
        zkey, bkey = ("persist_h4", "_boot4") if key == "gain_h4" else ("persist", "_boot")
        use = [p for p in pts if name in p["rows"] and p["rows"][name].get(key) is not None]
        D = np.array([p["D"] for p in use])
        y = np.array([p["rows"][name][key] for p in use])
        z = np.array([p["rows"][name][zkey] for p in use])
        Dd = None
        if use and all(dkey in p for p in use):
            k = min(len(p[dkey]) for p in use)
            Dd = np.stack([p[dkey][:k] for p in use])
        gb = pb = None
        if key in ("gain", "gain_h4") and use and all(p[bkey][name]["gain_boot"] is not None for p in use):
            gb = np.stack([p[bkey][name]["gain_boot"] for p in use])
            pb = np.stack([p[bkey][name]["persist_boot"] for p in use])
        return use, D, y, z, Dd, gb, pb

    def block(name, pts, key="gain", partial=True):
        use, D, y, z, Dd, gb, pb = arrays(name, pts, key)
        return correlation_block(D, Dd, y, gb, z, pb, a.draws, a.permutations, rng, partial)

    def measure(pts, covs, dkey):
        """The amendment's correlation block: primary row, extra covariates, its own stream."""
        use, D, y, z, Dd, gb, pb = arrays(primary_row, pts, dkey=dkey)
        extra = np.array([[map_covariate(p, c) for c in covs] for p in use]) if covs else None
        return correlation_block(D, Dd, y, gb, z, pb, a.draws, a.permutations, rng_amended, extra=extra)

    # the primary test and its companions
    use, D, g, z, Dd, gb, pb = arrays(primary_row, points)
    prim_block = block(primary_row, points)
    raw_block = block(primary_row, points, partial=False)
    rawpsnr_block = block(primary_row, points, key="raw_psnr", partial=False)
    loo = [partial_spearman(np.delete(D, i), np.delete(g, i), np.delete(z, i)) for i in range(len(D))]
    rho = prim_block["partial_spearman"]
    sign = np.sign(rho) if rho is not None and np.isfinite(rho) else 0
    finite = [float(v) for v in loo if np.isfinite(v)]
    same = int(sum(np.sign(v) == sign for v in finite)) if sign else 0
    primary = {"row": primary_row, "space": js.get("space"), "arm": js.get("primary_arm"), "horizon_tics": 1,
               "outcome": "mean(psnr_raw - persist_psnr_raw)", "covariate": "mean persist_psnr_raw",
               **prim_block, "draws": a.draws, "permutations": a.permutations,
               "raw_spearman": raw_block["partial_spearman"], "raw_ci95": raw_block["ci95"],
               "raw_psnr_spearman": rawpsnr_block["partial_spearman"], "raw_psnr_ci95": rawpsnr_block["ci95"],
               "leave_one_out": {"n": len(loo), "same_sign": same, "min": min(finite) if finite else None,
                                 "max": max(finite) if finite else None, "values": loo,
                                 "pass": bool(sign != 0 and same == len(loo))}}
    clusters = {c: block(primary_row, [p for p in points if p["cluster"] == c]) for c in ("arena", "campaign")}
    row_stats = {}
    for name in rows:
        b = block(name, points)
        b["missing"] = [p["key"] for p in points if name not in p["rows"]]
        row_stats[name] = b
    # the four-tic gain on four-tic persistence, with its episode bootstrap at the full draw count (the
    # pre-declared code read it with 2 draws, left them out, and controlled for one-tic persistence)
    secondary = {"lpips_gain": block(primary_row, points, key="lpips_gain"),
                 "gain_h4": {**block(primary_row, points, key="gain_h4"),
                             "covariate": "four-tic mean persist_psnr_raw", "draws": a.draws}}
    # episode level, within maps, when the per-episode distances sit beside the distances file
    epi_csv = os.path.join(os.path.dirname(os.path.abspath(a.distances)), f"per_episode_{js.get('space')}.csv")
    secondary["episode_within_map"] = None
    if os.path.isfile(epi_csv):
        with open(epi_csv) as f:
            de = {(r["set"], int(r["map"]), int(r["episode"])): float(r["D"]) for r in csv.DictReader(f)}
        pairs = []
        for p in use:
            eps = p["_boot"][primary_row]["episodes"] or {}
            keys = [e for e in eps if (p["set"], p["map"], e) in de]
            pairs.append(([de[(p["set"], p["map"], e)] for e in keys], [eps[e] for e in keys]))
        secondary["episode_within_map"] = within_map_episode_rho(pairs, a.permutations, rng)

    # the other feature spaces: the same statistic on their distance, and agreement of the rankings
    spaces = {}
    for spec in a.space:
        sname, spath = spec.split("=", 1)
        sjs, sprim, _ = load_distances(spath)
        sD = {m["key"]: float(m["D"]) for m in sprim}
        common = [p for p in use if p["key"] in sD]
        x = np.array([sD[p["key"]] for p in common])
        y = np.array([p["rows"][primary_row]["gain"] for p in common])
        zz = np.array([p["rows"][primary_row]["persist"] for p in common])
        spaces[sname] = {"file": os.path.abspath(spath), "n_maps": len(common),
                         "partial_spearman": partial_spearman(x, y, zz) if len(common) > 3 else None,
                         "kendall_tau_with_primary": kendall_tau(np.array([p["D"] for p in common]), x)
                         if len(common) > 1 else None}

    # the verdict, as memo section 5 lists it
    def neg(v):
        return None if v is None or not np.isfinite(v) else bool(v < 0)
    ci = primary["ci95"]
    pix = spaces.get("pixels")
    # every EXPECTED row: the U-Net alone is not every row, so a missing row leaves the condition pending,
    # while a supplied row that is not negative fails it whatever is missing
    expected = [r for r in a.expect_rows.split(",") if r]
    rows_missing = [r for r in expected if r not in row_stats]
    signs = [neg(r["partial_spearman"]) for r in row_stats.values()]
    every_row = False if False in signs else None if rows_missing or None in signs else True
    cond = {"negative_with_ci_excluding_zero": None if ci[1] is None
            else bool(primary["partial_spearman"] < 0 and ci[1] < 0),
            "leave_one_out_sign": primary["leave_one_out"]["pass"] and sign < 0,
            "within_both_clusters": None if any(clusters[c]["partial_spearman"] is None for c in clusters)
            else all(neg(clusters[c]["partial_spearman"]) for c in clusters),
            "every_row": every_row,
            "pixel_space": None if pix is None or pix["partial_spearman"] is None
            else bool(pix["partial_spearman"] < 0 and (pix["kendall_tau_with_primary"] or 0) >= TAU_AGREE),
            "distance_validation": (js.get("checks") or {}).get("all_pass")}
    raw_sig = raw_block["ci95"][1] is not None and raw_block["ci95"][1] < 0
    diag = {"vanishes_under_motion_control": bool(raw_sig and not cond["negative_with_ci_excluding_zero"]),
            "only_in_raw_psnr": bool(rawpsnr_block["ci95"][1] is not None and rawpsnr_block["ci95"][1] < 0
                                     and not cond["negative_with_ci_excluding_zero"]),
            "only_the_cluster_gap": bool(cond["negative_with_ci_excluding_zero"]
                                         and cond["within_both_clusters"] is False),
            "rows_disagree": cond["every_row"] is False}
    result = "does not support" if any(v is False for v in cond.values()) else \
        "incomplete" if any(v is None for v in cond.values()) else "supports"
    verdict = {"result": result, "conditions": cond, "diagnostics": diag, "rows_expected": expected,
               "rows_missing": rows_missing,
               "never_claimed": "causation, or maps beyond these arenas and curated campaign maps"}
    amended = amended_assessment(js, [p for p in points if primary_row in p["rows"]], primary_row, result, cond,
                                 measure)

    floor = (js.get("floor") or {}).get(js.get("primary_arm") or "motion")
    band = (floor["min"], floor["max"]) if floor else None
    panels = 0
    for theme in THEMES:
        panels = draw_figure(points, rows, row_stats, band, a.out, theme)

    # the appendix table
    names = list(rows)
    head = ["Map", "Cluster", "D", "Nearest", "n_eff", "Lives", "Valid", "Persistence (dB)"] + \
        [f"{n} gain h1 (dB)" for n in names] + [f"{n} LPIPS gain" for n in names] + [f"{n} gain h4 (dB)" for n in names]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for p in sorted(points, key=lambda q: q["D"]):
        r0 = p["rows"].get(primary_row, {})
        dcell = f"{p['D']:.3f}" + (f" ± {p['D_sd']:.3f}" if "D_sd" in p else "")
        cells = [p["key"], p["cluster"], dcell, str(p.get("nearest", "")),
                 f"{p['n_eff']:.0f}" if p.get("n_eff") is not None else "",
                 f"{p['lives']:.1f}" if p.get("lives") is not None else "",
                 f"{p['valid_fraction']:.2f}" if p.get("valid_fraction") is not None else "",
                 f"{r0['persist']:.2f}" if r0 else ""]
        for key, fmt in (("gain", "{:+.2f}"), ("lpips_gain", "{:+.3f}"), ("gain_h4", "{:+.2f}")):
            for n in names:
                s = p["rows"].get(n)
                v = s.get(key) if s else None
                cell = "" if v is None else fmt.format(v)
                if key == "gain" and s and s.get("gain_ci"):
                    cell += f" [{s['gain_ci'][0]:+.2f}, {s['gain_ci'][1]:+.2f}]"
                cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    caption = (f"Distance D (sliced W2 to the nearest training map, {js.get('space')} space, "
               f"{js.get('primary_arm')} weights; ± bootstrap SD), and gain over persistence per map. "
               f"Primary: partial Spearman {_fmt(primary['partial_spearman'])} {_fmt_ci(primary['ci95'])} "
               f"over {primary['n_maps']} maps ({primary_row}, one tic).")
    with open(os.path.join(a.out, "distance_table.md"), "w") as f:
        f.write(caption + "\n\n" + "\n".join(lines) + "\n")

    maps_out = [{k: v for k, v in p.items() if not k.startswith("_")} for p in points]
    stats = {"distances": os.path.abspath(a.distances), "rows": row_stats, "primary": primary,
             "within_cluster": clusters, "secondary": secondary, "spaces": spaces, "verdict": verdict,
             "amended_assessment": amended, "maps": maps_out, "seed": a.seed,
             "figure": {"panels": panels, "files": ["distance_gain.png", "distance_gain.svg", "distance_gain_dark.png",
                                                    "distance_gain_dark.svg"],
                        "palette": {k: v["series"] for k, v in THEMES.items()}}}
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(_jsonable(stats), f, indent=1)
    shown = ("row", "n_maps", "partial_spearman", "ci95", "permutation_p")
    print(json.dumps({"primary": {k: primary[k] for k in shown},
                      "verdict": result, "out": os.path.abspath(a.out)}, default=str))
    print("AMENDED_ASSESSMENT " + amended["summary"])
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="The distance study's statistics, appendix table and figure.")
    p.add_argument("--distances", required=True, help="distances_<space>.json of the primary space (SD 1.x)")
    p.add_argument("--row", action="append", default=[], metavar="LABEL=DIR",
                   help="a scored row: score_distance_maps.sh's output directory; the first is primary")
    p.add_argument("--primary-row", dest="primary_row", default="")
    p.add_argument("--expect-rows", dest="expect_rows", default=EXPECTED_ROWS,
                   help="comma list of the rows the every-row condition needs; one missing leaves it pending")
    p.add_argument("--space", action="append", default=[], metavar="NAME=JSON",
                   help="another feature space's distances (pixels, sd35) for the sign and ranking checks")
    p.add_argument("--out", default=os.path.join(HERE, "figures", "distance"))
    p.add_argument("--draws", type=int, default=10000, help="case-bootstrap draws over maps")
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    sys.exit(main())
