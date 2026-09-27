"""
The family step and the motion-matched skill deficit: the two distance panels of Figure 2 (d, e).

    python tools/family_step.py                       # defaults below; run from anywhere in the repo

**Zero-shot latent skill.** Per window w, s_w = -10 log10(latent_mse_w / copy_latent_mse_w): the model's latent
MSE over the copy-last latent MSE, in dB, higher is better (`score_adapt.py`'s LATENT SKILL). Windows that
`eval_tf.py` flags as duplicates (`dup_raw`, `dup_latent`) are left out, as `score_adapt.py` leaves them out, and
a window whose ratio is not a positive finite number is left out and counted. S0 of a map is the mean of s_w over
its windows: an arena's from `<fresh-root>/<backbone>/map<NN>/per_window.csv` (its 8 held-out episodes), a training
map's from the home read `<fresh-root>/home_<backbone>/val/per_window.csv` grouped by the file's `map` column. The
validation episodes 6000 to 6099 are interleaved, not blocked: the recorder gives episode e the map
(2, 3, 4, 5)[e % 4] (`doom_data.dense_episode_map`), and every home window's `map` must agree with that rule.

**The family step.** x is the frame distance D per map: an arena's primary D in `--distances`
(`distances_sd1.json`, the fresh 24-episode set); a training map's is its validation D in
`--training-distances` (`stats.json` beside `distance_table.md`, rows `val/2` to `val/5`); the D floor is the
band [min, max] of `floor.<primary arm>` in `--distances` (a training map's held-out episodes against its own
training footage). y is S0 per map and backbone. The step holds for a backbone when every arena's S0 lies below
every training map's S0 (and for D when every arena lies above every training map and the floor); the margin is
the gap between the two families. Within the arenas the Spearman of D with S0 says whether there is a slope.

**The motion-matched skill deficit.** Deciles of the copy-last latent error (`copy_latent_mse`, the motion a
window carries) are cut on the reference windows (`np.quantile`, linear interpolation), and each reference
decile carries its mean skill. A target window falls in the decile of its copy error (one outside the
reference range joins the end decile) and its deficit is that decile's mean skill minus its own skill; Delta of
a map is the mean deficit over its windows. For an arena the reference is all four training maps' home windows
of the same backbone. The floor scores each training map against the other three (leave one map out), so a
training map's Delta is what an unseen map with no shift would score. `delta_lomo_mean` is a sensitivity
variant: an arena's Delta averaged over those four three-map references, which matches the floor's reference
size.

**Adaptation outcomes** come from `--adapt-summary` (`paper/tables/tuned/adapt_summary.json`, the U-Net LoRA on
the tuned decoder): A at the budget, the half-gap cost with a censored arena tied at twice the budget
(`make_adapt_figures.cost_rank_value`), and the fraction of the gap closed, gain / (home - A0). Correlations are
scipy's Spearman (average ranks for ties) over the 13 arenas.

**Outputs.** `<out-dir>/family_step.json` (every per-map number, the floors, the step checks, the correlations,
the inputs' sha256) and `family_step.csv` (one row per backbone and map); `<fig-dir>/fig2d_family_step` (D
against S0, training floor shaded, one marker per backbone) and `fig2e_deficit` (A at the budget and the
half-gap budget against the primary backbone's Delta, drawn by `make_adapt_figures.fig_skill`), PDF and PNG.
"""
import argparse
import csv
import datetime
import glob
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "paper"))
import make_adapt_figures as maf  # noqa: E402
from score_adapt import DUPLICATE_FLAGS  # noqa: E402

from matplotlib import ticker  # noqa: E402  (maf has already selected the Agg backend)

TRAINING_MAPS = maf.TRAINING_MAPS
BACKBONES = ("unet200k_ema", "pixart200k_ema", "sd35_170000")
N_BINS = 10
MAP_DIR_RE = re.compile(r"map0*(\d+)$")


# ---------------------------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------------------------

class Windows:
    """Per-window arrays of one per-window file (or a subset): map, episode, copy-last latent MSE, skill."""

    def __init__(self, map_, episode, copy, skill, dropped_duplicates=0, dropped_undefined=0):
        self.map, self.episode = np.asarray(map_, int), np.asarray(episode, int)
        self.copy, self.skill = np.asarray(copy, float), np.asarray(skill, float)
        self.dropped_duplicates, self.dropped_undefined = dropped_duplicates, dropped_undefined

    @classmethod
    def from_rows(cls, rows):
        """From dicts with `map`, `episode`, `copy` and `skill` (duplicates already left out)."""
        return cls([r["map"] for r in rows], [r["episode"] for r in rows], [r["copy"] for r in rows],
                   [r["skill"] for r in rows])

    def __len__(self):
        return len(self.skill)

    def subset(self, mask):
        return Windows(self.map[mask], self.episode[mask], self.copy[mask], self.skill[mask])

    def maps(self):
        return sorted(int(m) for m in np.unique(self.map))


def window_skill(latent_mse, copy_latent_mse):
    """-10 log10 of the model-to-copy-last latent MSE ratio, or None when the ratio is not positive and finite."""
    try:
        ratio = float(latent_mse) / float(copy_latent_mse)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return -10.0 * math.log10(ratio) if ratio > 0 and math.isfinite(ratio) else None


def read_windows(path):
    """The windows of an eval_tf `per_window.csv` that the skill averages: duplicates and undefined ratios out."""
    with open(path) as f:
        reader = csv.DictReader(f)
        header, rows = list(reader.fieldnames or []), list(reader)
    for col in ("map", "episode", "latent_mse", "copy_latent_mse"):
        if col not in header:
            raise SystemExit(f"{path}: no {col} column")
    flags = [c for c in DUPLICATE_FLAGS if c in header]
    kept, dups, undefined = [], 0, 0
    for r in rows:
        if any(maf.as_float(r.get(c)) == 1 for c in flags):
            dups += 1
            continue
        s = window_skill(r["latent_mse"], r["copy_latent_mse"])
        if s is None:
            undefined += 1
            continue
        kept.append({"map": int(r["map"]), "episode": int(r["episode"]), "copy": float(r["copy_latent_mse"]),
                     "skill": s})
    w = Windows.from_rows(kept)
    w.dropped_duplicates, w.dropped_undefined = dups, undefined
    return w


def dense_map(episode, maps=TRAINING_MAPS):
    """The recorder's map for a dense-corpus episode, maps[e % len(maps)] (doom_data.dense_episode_map, which is
    not imported because doom_data pulls in torch)."""
    return int(maps[int(episode) % len(maps)])


def read_home(path):
    """The training maps' validation windows, each window's `map` checked against the recorder's rule."""
    w = read_windows(path)
    bad = [(int(e), int(m)) for e, m in zip(w.episode, w.map) if dense_map(e) != m]
    if bad:
        raise SystemExit(f"{path}: {len(bad)} windows whose map column disagrees with map = (2, 3, 4, 5)[e % 4], "
                         f"first episode {bad[0][0]} labelled map {bad[0][1]}")
    return w


# ---------------------------------------------------------------------------------------------
# the deficit, the floor, the step
# ---------------------------------------------------------------------------------------------

def decile_table(reference, n_bins=N_BINS):
    """(inner edges, mean skill per bin, windows per bin) of the reference's copy-error quantile bins."""
    edges = np.quantile(reference.copy, np.linspace(0, 1, n_bins + 1))[1:-1]
    idx = np.searchsorted(edges, reference.copy, side="right")
    counts = np.bincount(idx, minlength=n_bins)
    if (counts == 0).any():
        raise SystemExit(f"a copy-error bin of the reference is empty (counts {counts.tolist()}): too many ties")
    means = np.bincount(idx, weights=reference.skill, minlength=n_bins) / counts
    return edges, means, counts


def matched_deficit(reference, target, n_bins=N_BINS):
    """Delta of `target` against `reference`: the mean over target windows of the reference's mean skill in the
    window's copy-error decile minus the window's skill."""
    edges, means, _ = decile_table(reference, n_bins)
    k = np.searchsorted(edges, target.copy, side="right")
    per = means[k] - target.skill
    outside = int(((target.copy < reference.copy.min()) | (target.copy > reference.copy.max())).sum())
    return {"delta": float(per.mean()), "per_window": per, "decile": k, "n": int(len(per)),
            "outside_reference": outside}


def leave_one_map_out(home, maps=TRAINING_MAPS, n_bins=N_BINS):
    """The floor: {map: matched_deficit of the map's windows against the other maps' windows}."""
    return {m: matched_deficit(home.subset(home.map != m), home.subset(home.map == m), n_bins) for m in maps}


def leave_one_map_out_average(home, target, maps=TRAINING_MAPS, n_bins=N_BINS):
    """A target's Delta averaged over the floor's references (every training map but one)."""
    return float(np.mean([matched_deficit(home.subset(home.map != m), target, n_bins)["delta"] for m in maps]))


def step_check(training, arenas, higher_is_farther=False):
    """Does every arena sit on the far side of every training map? For skill, far is lower; for D, higher.

    `margin` is the gap between the families (positive when the step holds) and `arenas_on_training_side` the
    arenas that break it.
    """
    t, a = list(training.values()), list(arenas.values())
    if higher_is_farther:
        margin = min(a) - max(t)
        inside = sorted(k for k, v in arenas.items() if v <= max(t))
    else:
        margin = min(t) - max(a)
        inside = sorted(k for k, v in arenas.items() if v >= min(t))
    return {"holds": not inside, "margin": margin, "training_min": min(t), "training_max": max(t),
            "arena_min": min(a), "arena_max": max(a), "arenas_on_training_side": inside}


def value_range(values):
    v = list(values)
    return [min(v), max(v)] if v else None


# ---------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------

def training_distances(path):
    """{map: D} of the training maps' validation entries (`val/<map>`) of a distance-study stats or distances file."""
    with open(path) as f:
        js = json.load(f)
    out = {int(m["map"]): float(m["D"]) for m in js.get("maps", [])
           if m.get("set") == "val" and int(m["map"]) in TRAINING_MAPS and maf.finite(m.get("D"))}
    missing = [m for m in TRAINING_MAPS if m not in out]
    if missing:
        raise SystemExit(f"{path}: no val D for training maps {missing}")
    return out


def distance_floor(path):
    """The D floor of `distances_<space>.json`: its primary arm's floor summary."""
    with open(path) as f:
        js = json.load(f)
    arm = js.get("primary_arm", "motion")
    floor = (js.get("floor") or {}).get(arm)
    if not floor:
        raise SystemExit(f"{path}: no floor.{arm}")
    return {"arm": arm, **{k: floor.get(k) for k in ("n", "min", "max", "mean", "p2.5", "p97.5")}}


def arena_files(row_dir):
    """{arena: per_window.csv} of a directory of eval_tf reads keyed by map."""
    out = {}
    for d in sorted(glob.glob(os.path.join(row_dir, "map*"))):
        m = MAP_DIR_RE.search(os.path.basename(d))
        p = os.path.join(d, "per_window.csv")
        if m and os.path.exists(p):
            out[int(m.group(1))] = p
    if not out:
        raise SystemExit(f"{row_dir}: no map<NN>/per_window.csv")
    return out


def adaptation_records(path):
    """(summary header, {arena: record}) from `make_adapt_figures.py`'s adapt_summary.json."""
    with open(path) as f:
        js = json.load(f)
    home, budget = js["home"], js["budget"]
    recs = {}
    for r in js["per_arena"]:
        if r.get("incomplete"):
            continue
        gap = home - r["A0"] if maf.finite(r.get("A0")) else None
        recs[int(r["arena"])] = {**r, "cost_rank": maf.cost_rank_value(r["cost_half_gap"], budget),
                                 "fraction_closed": r["gain"] / gap if gap and maf.finite(r.get("gain")) else None}
    head = {"source": maf.rel(path), "decoder": js.get("decoder"), "weights": js.get("weights"), "home": home,
            "budget": budget, "cost_censored_as": maf.cost_rank_value(None, budget), "n_arenas": len(recs)}
    return head, recs


# ---------------------------------------------------------------------------------------------
# the study
# ---------------------------------------------------------------------------------------------

def backbone_entry(name, fresh_root, arena_D, train_D, n_bins):
    """Every per-map number of one backbone: S0, Delta (the floor for training maps), the step, D vs S0."""
    home_path = os.path.join(fresh_root, f"home_{name}", "val", "per_window.csv")
    home = read_home(home_path)
    missing = [m for m in TRAINING_MAPS if m not in home.maps()]
    if missing:
        raise SystemExit(f"{home_path}: no windows of training maps {missing}")
    floor = leave_one_map_out(home, TRAINING_MAPS, n_bins)
    maps, files = {}, [home_path]
    for m in TRAINING_MAPS:
        w = home.subset(home.map == m)
        maps[m] = {"role": "training", "D": train_D.get(m), "S0": float(w.skill.mean()), "n": len(w),
                   "episodes": int(len(np.unique(w.episode))), "delta": floor[m]["delta"],
                   "delta_reference": "the other three training maps",
                   "outside_reference": floor[m]["outside_reference"]}
    for a, p in sorted(arena_files(os.path.join(fresh_root, name)).items()):
        w = read_windows(p)
        if w.maps() != [a]:
            raise SystemExit(f"{p}: map column {w.maps()} is not the directory's map {a}")
        d = matched_deficit(home, w, n_bins)
        maps[a] = {"role": "arena", "D": arena_D.get(a), "S0": float(w.skill.mean()), "n": len(w),
                   "episodes": int(len(np.unique(w.episode))), "delta": d["delta"],
                   "delta_reference": "all four training maps",
                   "delta_lomo_mean": leave_one_map_out_average(home, w, TRAINING_MAPS, n_bins),
                   "outside_reference": d["outside_reference"],
                   "dropped_duplicates": w.dropped_duplicates, "dropped_undefined": w.dropped_undefined}
        files.append(p)
    arenas = [m for m in maps if maps[m]["role"] == "arena"]
    s0 = {m: maps[m]["S0"] for m in maps}
    floor_range = value_range(maps[m]["delta"] for m in TRAINING_MAPS)
    arena_range = value_range(maps[m]["delta"] for m in arenas)
    return {
        "label": maf.ROW_NAMES.get(name, name), "home": maf.rel(home_path),
        "home_dropped_duplicates": home.dropped_duplicates, "home_dropped_undefined": home.dropped_undefined,
        "maps": maps,
        "step": step_check({m: s0[m] for m in TRAINING_MAPS}, {m: s0[m] for m in arenas}),
        "spearman_D_vs_S0_arenas": maf.spearman([maps[m]["D"] for m in arenas], [s0[m] for m in arenas]),
        "delta_floor_range": floor_range, "delta_arena_range": arena_range,
        "delta_lomo_arena_range": value_range(maps[m]["delta_lomo_mean"] for m in arenas),
        "delta_passes_floor": arena_range[0] > floor_range[1],
    }, files


def correlations(entries, primary, adapt):
    """The primary backbone's Delta against the adaptation outcomes, D and S0; Delta across backbones."""
    p = entries[primary]["maps"]
    arenas = sorted(m for m in p if p[m]["role"] == "arena" and m in adapt)

    def col(maps, key):
        return [maps[m].get(key) for m in arenas]

    def outcome(key):
        return [adapt[m].get(key) for m in arenas]

    out = {"n_arenas": len(arenas), "arenas": arenas, "backbone": primary}
    for tag, key in (("delta", "delta"), ("delta_lomo_mean", "delta_lomo_mean")):
        x = col(p, key)
        out[f"{tag}_vs_A_budget"] = maf.spearman(x, outcome("A_budget"))
        out[f"{tag}_vs_cost_half_gap"] = maf.spearman(x, outcome("cost_rank"))
        out[f"{tag}_vs_fraction_closed"] = maf.spearman(x, outcome("fraction_closed"))
        out[f"{tag}_vs_D"] = maf.spearman(x, col(p, "D"))
        out[f"{tag}_vs_S0"] = maf.spearman(x, col(p, "S0"))
    out["S0_vs_A_budget"] = maf.spearman(col(p, "S0"), outcome("A_budget"))
    out["S0_vs_cost_half_gap"] = maf.spearman(col(p, "S0"), outcome("cost_rank"))
    for other in entries:
        if other != primary:
            out[f"delta_{primary}_vs_{other}"] = maf.spearman(col(p, "delta"), col(entries[other]["maps"], "delta"))
    return out


# ---------------------------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------------------------

CSV_FIELDS = ["backbone", "label", "map", "role", "D", "S0", "n_windows", "episodes", "delta", "delta_reference",
              "delta_lomo_mean", "outside_reference", "lora_A0", "lora_A_budget", "lora_gain",
              "lora_fraction_closed", "lora_cost_half_gap", "lora_censored_half_gap"]


def write_csv(path, entries, adapt):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for name, e in entries.items():
            for m, r in sorted(e["maps"].items(), key=lambda kv: (kv[1]["role"] != "training", kv[0])):
                a = adapt.get(m, {}) if r["role"] == "arena" else {}
                w.writerow({"backbone": name, "label": e["label"], "map": m, "role": r["role"], "D": r["D"],
                            "S0": r["S0"], "n_windows": r["n"], "episodes": r["episodes"], "delta": r["delta"],
                            "delta_reference": r["delta_reference"], "delta_lomo_mean": r.get("delta_lomo_mean"),
                            "outside_reference": r["outside_reference"], "lora_A0": a.get("A0"),
                            "lora_A_budget": a.get("A_budget"), "lora_gain": a.get("gain"),
                            "lora_fraction_closed": a.get("fraction_closed"),
                            "lora_cost_half_gap": a.get("cost_half_gap"),
                            "lora_censored_half_gap": a.get("censored_half_gap")})


def fig_family_step(entries, floor, out_dir):
    """Figure 2d: S0 against D per map, one marker per backbone, the training floor of D shaded.

    Each backbone's lowest training map is a faint line in its colour across the panel, so the step reads as
    every arena of that colour sitting under its own line.
    """
    fig, (ax,) = maf.new_figure(maf.FIG2_SIZE)
    ax.axvspan(floor["min"], floor["max"], color=maf.SHADE, alpha=0.6, lw=0, zorder=0)
    for name, e in entries.items():
        slot = maf.backbone_slot(name)
        c, mk = maf.SERIES[slot % len(maf.SERIES)], maf.MARKERS[slot % len(maf.MARKERS)]
        pts = sorted((r["D"], r["S0"]) for r in e["maps"].values() if r["D"] is not None)
        ax.plot([d for d, _ in pts], [s for _, s in pts], ls="none", marker=mk, ms=2.6, mfc=c, mec="white", mew=0.3,
                zorder=3, label=e["label"])
        ax.axhline(e["step"]["training_min"], color=c, lw=0.5, ls=(0, (1, 1.5)), alpha=0.8, zorder=1)
    ax.text((floor["min"] + floor["max"]) / 2, 0.02, "training\nfloor", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=4.5, color=maf.SECONDARY, linespacing=0.9)
    ax.set_xlim(0, None)
    ax.set_xlabel("frame distance $D$")
    ax.set_ylabel("zero-shot skill $S_0$ (dB)")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.legend(loc="upper right", handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2, fontsize=4.5)
    return maf.save(fig, out_dir, "fig2d_family_step")


def fig_deficit(entries, primary, adapt, head, out_dir):
    """Figure 2e: A at the budget and the half-gap budget against the primary backbone's Delta (`fig_skill`)."""
    maps = entries[primary]["maps"]
    records = [{**adapt[m], "delta": maps[m]["delta"], "D": maps[m]["D"]} for m in sorted(adapt)
               if m in maps and maps[m]["role"] == "arena"]
    return maf.fig_skill(records, head["home"], head["budget"], out_dir, x="delta",
                         xlabel=f"skill deficit $\\Delta$, {entries[primary]['label']} (dB)", stem="fig2e_deficit")


def build_parser():
    p = argparse.ArgumentParser(description="The family step (D vs zero-shot skill) and the motion-matched deficit.")
    p.add_argument("--fresh-root", default=os.path.join(REPO, "results", "fresh_rescore"),
                   help="<root>/<backbone>/map<NN>/per_window.csv and <root>/home_<backbone>/val/per_window.csv")
    p.add_argument("--backbones", nargs="+", default=list(BACKBONES),
                   help="row names under --fresh-root; the first is the primary backbone of the Delta panel")
    p.add_argument("--distances", default=os.path.join(REPO, "results", "distance_v2", "distances_sd1.json"),
                   help="the arenas' primary D and the D floor")
    p.add_argument("--training-distances",
                   default=os.path.join(REPO, "results", "distance_study", "figure_unet_h1", "stats.json"),
                   help="the training maps' validation D (val/<map>); the source of distance_table.md")
    p.add_argument("--adapt-summary", default=os.path.join(REPO, "paper", "tables", "tuned", "adapt_summary.json"))
    p.add_argument("--bins", type=int, default=N_BINS, help="copy-error quantile bins of the deficit")
    p.add_argument("--out-dir", default=os.path.join(REPO, "results", "family_step"))
    p.add_argument("--fig-dir", default=os.path.join(REPO, "paper", "figures"))
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    arena_D = maf.load_distances(a.distances)
    train_D = training_distances(a.training_distances)
    floor = distance_floor(a.distances)
    head, adapt = adaptation_records(a.adapt_summary)
    entries, files = {}, [a.distances, a.training_distances, a.adapt_summary]
    for name in a.backbones:
        entries[name], used = backbone_entry(name, a.fresh_root, arena_D, train_D, a.bins)
        files += used
    primary = a.backbones[0]
    arenas_D = {m: r["D"] for m, r in entries[primary]["maps"].items() if r["role"] == "arena" and r["D"] is not None}
    d_step = step_check({**train_D, "floor_max": floor["max"]}, arenas_D, higher_is_farther=True)
    result = {
        "generated_by": "tools/family_step.py",
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "definitions": {
            "skill": "s_w = -10 log10(latent_mse / copy_latent_mse); duplicates (dup_raw, dup_latent) left out",
            "S0": "mean s_w over a map's windows (arenas: 8 held-out episodes; training maps: home val windows)",
            "episode_map_rule": "map = (2, 3, 4, 5)[episode % 4] (doom_data.dense_episode_map), checked per window",
            "delta": f"mean over a map's windows of (reference mean skill in the window's copy_latent_mse "
                     f"{a.bins}-quantile bin) - s_w; reference = all training maps' home windows for an arena, "
                     f"the other three training maps for a training map (the floor)",
            "delta_lomo_mean": "an arena's delta averaged over the four three-training-map references",
            "D": "arenas: primary D of --distances; training maps: val/<map> D of --training-distances",
            "D_floor": "floor.<primary arm> of --distances: [min, max]",
            "cost_rank": "half-gap cost with a censored arena tied at twice the budget",
            "fraction_closed": "gain / (home - A0) of the tuned-decoder adaptation summary",
        },
        "bins": a.bins, "primary_backbone": primary,
        "D_floor": floor, "D_training": train_D, "D_step": d_step,
        "adaptation": head, "backbones": entries,
        "correlations": correlations(entries, primary, adapt),
        "inputs": [{"path": maf.rel(p), "sha256": maf.sha256(p)} for p in files],
    }
    os.makedirs(a.out_dir, exist_ok=True)
    written = [os.path.join(a.out_dir, "family_step.json"), os.path.join(a.out_dir, "family_step.csv")]
    with open(written[0], "w") as f:
        json.dump(maf.jsonable(result), f, indent=1)
        f.write("\n")
    write_csv(written[1], entries, adapt)
    maf.style()
    written += fig_family_step(entries, floor, a.fig_dir)
    written += fig_deficit(entries, primary, adapt, head, a.fig_dir)
    for p in written:
        print("wrote", maf.rel(p))
    print_summary(result)
    return 0


def print_summary(result):
    c = result["correlations"]
    print(f"D step: {'holds' if result['D_step']['holds'] else 'fails'}, margin {result['D_step']['margin']:+.3f}")
    for e in result["backbones"].values():
        s, r = e["step"], e["spearman_D_vs_S0_arenas"]
        print(f"{e['label']:28s} S0 training {s['training_min']:.2f}-{s['training_max']:.2f}, arenas "
              f"{s['arena_min']:.2f}-{s['arena_max']:.2f}, step {'holds' if s['holds'] else 'fails'}; "
              f"D vs S0 {r['rho']:+.2f}; Delta floor {e['delta_floor_range'][0]:+.2f} to "
              f"{e['delta_floor_range'][1]:+.2f}, arenas {e['delta_arena_range'][0]:.2f} to "
              f"{e['delta_arena_range'][1]:.2f}")
    for k, v in c.items():
        if isinstance(v, dict) and v.get("rho") is not None:
            print(f"  {k:48s} {v['rho']:+.3f} (p {v['p']:.3g}, n {v['n']})")


if __name__ == "__main__":
    sys.exit(main())
