"""`tools/family_step.py`: the family step (frame distance D against zero-shot latent skill) and the
motion-matched skill deficit Delta, from `eval_tf.py` per-window files.

Every test builds windows by hand. A window is a (copy-last latent MSE, skill) pair written as
`copy_latent_mse` and `latent_mse = copy * 10^(-skill/10)`, so the tool's per-window skill
-10 log10(latent_mse / copy_latent_mse) returns the skill exactly. Two constructions carry the checks:

  decile matching: a reference of 100 windows with copy values 1..100 and skill equal to the decile index
      (0 for copy 1..10, 1 for 11..20, ...), so every decile's mean skill is its index;
  the leave-one-map-out floor: four training maps that share one set of copy values, with skill
      f(copy) + offset(map); a map scored against the other three then has Delta = mean(other offsets) -
      its own offset exactly, because the same copy values fill the reference deciles.

    python -m pytest paper/fixtures/test_family_step.py -q
"""
import csv
import json
import math
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))

pytest.importorskip("matplotlib")
pytest.importorskip("scipy")
import family_step as fs  # noqa: E402

TRAINING = (2, 3, 4, 5)
COPIES = [0.05 + 0.01 * i for i in range(40)]          # shared by every synthetic map, distinct values


def f(copy):
    """A skill that falls with motion, the shape the real windows have."""
    return 3.0 - 10.0 * copy


def window(copy, skill, m=2, episode=0, dup=0):
    return {"map": m, "episode": episode, "copy": copy, "skill": skill, "dup": dup}


def write_per_window(path, windows, episode_of=None):
    """eval_tf's per_window.csv for the given windows (only the columns the tool reads, plus a decoy)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["index", "episode", "map", "start", "latent_mse", "copy_latent_mse", "latent_mse_ratio", "dup_latent",
              "dup_raw", "psnr_dec"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, x in enumerate(windows):
            ep = x["episode"] if episode_of is None else episode_of(x, i)
            w.writerow({"index": i, "episode": ep, "map": x["map"], "start": 10 * i,
                        "latent_mse": x["copy"] * 10 ** (-x["skill"] / 10), "copy_latent_mse": x["copy"],
                        "latent_mse_ratio": 99.0, "dup_latent": x["dup"], "dup_raw": 0, "psnr_dec": 20.0})
    return str(path)


def home_windows(offsets):
    """Validation windows of the four training maps: the shared copy values, skill f(copy) + offset(map).

    Episodes follow the recorder's rule (map = (2, 3, 4, 5)[episode % 4]) from 6000.
    """
    out = []
    for m in TRAINING:
        for j, c in enumerate(COPIES):
            out.append(window(c, f(c) + offsets[m], m=m, episode=6000 + (m - 2) + 4 * (j % 5)))
    return out


def arr(windows):
    return fs.Windows.from_rows(windows)


# ---------------------------------------------------------------------------------------------
# the per-window skill
# ---------------------------------------------------------------------------------------------

def test_the_window_skill_is_minus_ten_log_ratio_and_duplicates_are_left_out(tmp_path):
    p = write_per_window(tmp_path / "pw.csv", [window(0.2, 1.5), window(0.4, -0.5), window(0.3, 9.0, dup=1)])
    w = fs.read_windows(p)
    assert len(w) == 2 and w.dropped_duplicates == 1
    assert w.skill == pytest.approx([1.5, -0.5])
    assert w.copy == pytest.approx([0.2, 0.4])
    assert fs.window_skill(0.1, 0.2) == pytest.approx(-10 * math.log10(0.5))
    assert fs.window_skill(0.1, 0.0) is None and fs.window_skill(0.0, 0.2) is None


def test_the_map_of_a_validation_episode_follows_the_recorders_rule():
    assert [fs.dense_map(e) for e in (6000, 6001, 6002, 6003, 6004, 6099)] == [2, 3, 4, 5, 2, 5]


def test_a_home_file_whose_map_column_breaks_the_recorders_rule_stops(tmp_path):
    p = write_per_window(tmp_path / "home.csv", [window(0.2, 1.0, m=3, episode=6000)])
    with pytest.raises(SystemExit, match="map"):
        fs.read_home(p)


# ---------------------------------------------------------------------------------------------
# decile matching
# ---------------------------------------------------------------------------------------------

def decile_reference():
    return arr([window(float(c), float((c - 1) // 10)) for c in range(1, 101)])


def test_each_reference_decile_carries_its_mean_skill():
    edges, means, counts = fs.decile_table(decile_reference(), 10)
    assert len(edges) == 9
    assert means == pytest.approx(list(range(10)))
    assert list(counts) == [10] * 10


def test_a_window_is_matched_to_the_reference_decile_of_its_copy_error():
    ref = decile_reference()
    # copy 55 sits in the sixth decile (51..60, mean skill 5); its own skill 2 leaves a deficit of 3
    d = fs.matched_deficit(ref, arr([window(55.0, 2.0)]))
    assert d["delta"] == pytest.approx(3.0)
    assert list(d["decile"]) == [5]
    # the deficit of a map is the mean over its windows: deciles 0 and 9, deficits 0 - 1 and 9 - 4
    d = fs.matched_deficit(ref, arr([window(3.0, 1.0), window(97.0, 4.0)]))
    assert d["delta"] == pytest.approx(((0 - 1) + (9 - 4)) / 2)


def test_windows_outside_the_reference_range_join_the_end_deciles():
    ref = decile_reference()
    d = fs.matched_deficit(ref, arr([window(1e-6, 0.0), window(1e6, 0.0)]))
    assert list(d["decile"]) == [0, 9]
    assert d["outside_reference"] == 2
    assert d["delta"] == pytest.approx((0 + 9) / 2)


def test_the_reference_scored_against_itself_has_no_deficit():
    rng = np.random.default_rng(0)
    ref = arr([window(float(c), float(s)) for c, s in zip(rng.lognormal(size=333), rng.normal(size=333))])
    assert fs.matched_deficit(ref, ref)["delta"] == pytest.approx(0.0, abs=1e-12)


def test_the_deficit_matches_on_copy_error_so_a_harder_motion_mix_alone_leaves_no_deficit():
    # the target holds only the fastest fifth of the reference's windows (its top two deciles, four each),
    # each exactly as skilled as the reference at its copy error: a raw skill gap, no matched deficit
    ref = arr([window(c, f(c)) for c in COPIES])
    fast = arr([window(c, f(c)) for c in COPIES[-8:]])
    assert ref.skill.mean() - fast.skill.mean() > 1.0
    d = fs.matched_deficit(ref, fast)
    assert sorted(set(d["decile"])) == [8, 9]
    assert d["delta"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------------------------
# the leave-one-map-out floor
# ---------------------------------------------------------------------------------------------

def test_the_floor_scores_each_training_map_against_the_other_three():
    offsets = {2: 0.0, 3: 0.0, 4: 0.0, 5: -1.0}
    floor = fs.leave_one_map_out(arr(home_windows(offsets)), TRAINING, 10)
    # map 5 against maps 2 to 4 (offset 0): Delta = 0 - (-1) = 1; with itself in the reference it would be 0.75
    assert floor[5]["delta"] == pytest.approx(1.0)
    # map 2 against maps 3, 4, 5: Delta = (0 + 0 - 1) / 3 - 0
    for m in (2, 3, 4):
        assert floor[m]["delta"] == pytest.approx(-1 / 3)
    assert {m: floor[m]["n"] for m in TRAINING} == {m: len(COPIES) for m in TRAINING}


def test_the_three_map_average_is_the_mean_over_the_four_floor_references():
    offsets = {2: 0.4, 3: 0.0, 4: -0.2, 5: -1.0}
    home = arr(home_windows(offsets))
    target = arr([window(c, f(c) - 2.0, m=9) for c in COPIES])
    each = [np.mean([offsets[k] for k in TRAINING if k != m]) + 2.0 for m in TRAINING]
    assert fs.leave_one_map_out_average(home, target, TRAINING, 10) == pytest.approx(np.mean(each))
    assert fs.matched_deficit(home, target)["delta"] == pytest.approx(np.mean(list(offsets.values())) + 2.0)


# ---------------------------------------------------------------------------------------------
# the step check
# ---------------------------------------------------------------------------------------------

def test_the_step_holds_when_every_arena_sits_below_every_training_map():
    s = fs.step_check({2: 2.5, 3: 3.0, 4: 2.8, 5: 2.6}, {1: 1.2, 6: 2.4})
    assert s["holds"] and s["margin"] == pytest.approx(0.1)
    assert s["training_min"] == pytest.approx(2.5) and s["arena_max"] == pytest.approx(2.4)
    assert s["arenas_on_training_side"] == []


def test_one_arena_at_or_above_the_lowest_training_map_breaks_the_step():
    s = fs.step_check({2: 2.5, 3: 3.0, 4: 2.8, 5: 2.6}, {1: 1.2, 6: 2.4, 7: 2.5, 8: 2.9})
    assert not s["holds"] and s["margin"] == pytest.approx(-0.4)
    assert s["arenas_on_training_side"] == [7, 8]


def test_the_distance_step_runs_the_other_way_every_arena_above_every_training_map():
    s = fs.step_check({2: 0.06, 3: 0.03}, {1: 0.12, 6: 0.2}, higher_is_farther=True)
    assert s["holds"] and s["margin"] == pytest.approx(0.06)
    s = fs.step_check({2: 0.06, 3: 0.03}, {1: 0.05, 6: 0.2}, higher_is_farther=True)
    assert not s["holds"] and s["arenas_on_training_side"] == [1]


# ---------------------------------------------------------------------------------------------
# the command line, end to end on a small tree
# ---------------------------------------------------------------------------------------------

BACKBONES = ("unet200k_ema", "pixart200k_ema", "sd35_170000")
ARENA_SHIFT = {1: 1.0, 6: 1.6, 7: 1.3, 8: 2.0}       # Delta of each arena when the home offsets are zero
ARENA_D = {1: 0.17, 6: 0.20, 7: 0.27, 8: 0.12}
TRAIN_D = {2: 0.065, 3: 0.034, 4: 0.069, 5: 0.026}


def build_tree(root):
    fresh = os.path.join(root, "fresh_rescore")
    for bi, bb in enumerate(BACKBONES):
        write_per_window(os.path.join(fresh, f"home_{bb}", "val", "per_window.csv"),
                         home_windows(dict.fromkeys(TRAINING, 0.0)))
        for a, shift in ARENA_SHIFT.items():
            ws = [window(c, f(c) - shift - 0.1 * bi, m=a, episode=40 + (j % 8)) for j, c in enumerate(COPIES)]
            write_per_window(os.path.join(fresh, bb, f"map{a:02d}", "per_window.csv"), ws)
    dist = os.path.join(root, "distances_sd1.json")
    with open(dist, "w") as fh:
        json.dump({"primary_arm": "motion",
                   "floor": {"motion": {"n": 4, "min": 0.022, "max": 0.093, "p2.5": 0.024, "p97.5": 0.092,
                                        "values": [0.022, 0.04, 0.06, 0.093]}},
                   "maps": [{"key": f"arenas13/{a}", "set": "arenas13", "map": a, "role": "primary",
                             "cluster": "arena", "D": d} for a, d in ARENA_D.items()]}, fh)
    stats = os.path.join(root, "stats.json")
    with open(stats, "w") as fh:
        json.dump({"maps": [{"key": f"val/{m}", "set": "val", "map": m, "D": d} for m, d in TRAIN_D.items()]
                   + [{"key": "seen/15", "set": "seen", "map": 15, "D": 0.127}]}, fh)
    summary = os.path.join(root, "adapt_summary.json")
    per = []
    for a, shift in ARENA_SHIFT.items():
        cost = {1: 250, 6: None, 7: 1000, 8: None}[a]
        per.append({"arena": a, "A0": 2.0 - shift, "A_budget": 5.0 - shift, "gain": 3.0,
                    "cost_half_gap": cost, "censored_half_gap": cost is None, "incomplete": False,
                    "A": {"0": 2.0 - shift, "250": 3.0 - shift, "1000": 4.0 - shift, "4000": 5.0 - shift}})
    with open(summary, "w") as fh:
        json.dump({"decoder": "tuned", "home": 5.0, "budget": 4000, "spearman_cost_censored_as": 8000,
                   "per_arena": per}, fh)
    return fresh, dist, stats, summary


def run_cli(tmp_path):
    fresh, dist, stats, summary = build_tree(str(tmp_path))
    out, figs = tmp_path / "out", tmp_path / "figs"
    assert fs.main(["--fresh-root", fresh, "--distances", dist, "--training-distances", stats,
                    "--adapt-summary", summary, "--out-dir", str(out), "--fig-dir", str(figs)]) == 0
    return json.load(open(out / "family_step.json")), out, figs


def test_the_cli_writes_every_per_map_number_the_floor_and_the_correlations(tmp_path):
    js, out, _ = run_cli(tmp_path)
    u = js["backbones"]["unet200k_ema"]
    for a, shift in ARENA_SHIFT.items():
        assert u["maps"][str(a)]["delta"] == pytest.approx(shift)
        assert u["maps"][str(a)]["S0"] == pytest.approx(np.mean([f(c) for c in COPIES]) - shift)
        assert u["maps"][str(a)]["D"] == pytest.approx(ARENA_D[a])
    for m in TRAINING:
        assert u["maps"][str(m)]["role"] == "training"
        assert u["maps"][str(m)]["delta"] == pytest.approx(0.0, abs=1e-12)     # the floor, all offsets zero
        assert u["maps"][str(m)]["D"] == pytest.approx(TRAIN_D[m])
    assert u["step"]["holds"] and u["delta_passes_floor"]
    assert js["D_step"]["holds"] and js["D_floor"]["max"] == pytest.approx(0.093)
    # Delta orders the arenas exactly as A at the budget reverses it
    assert js["correlations"]["delta_vs_A_budget"]["rho"] == pytest.approx(-1.0)
    assert js["correlations"]["delta_unet200k_ema_vs_pixart200k_ema"]["rho"] == pytest.approx(1.0)
    assert "delta_vs_cost_half_gap" in js["correlations"] and "delta_vs_D" in js["correlations"]
    assert u["spearman_D_vs_S0_arenas"]["n"] == len(ARENA_SHIFT)
    rows = list(csv.DictReader(open(out / "family_step.csv")))
    assert len(rows) == len(BACKBONES) * (len(TRAINING) + len(ARENA_SHIFT))
    assert {r["role"] for r in rows} == {"training", "arena"}


def test_the_two_panels_are_written_in_the_builders_style(tmp_path):
    _, _, figs = run_cli(tmp_path)
    # Figure 3a sits beside 3b in one row: the paper owner's slot, 2.25 x 1.5 in, set without scaling
    for stem, size in (("fig2d_family_step", (2.25, 1.5)), ("fig2e_deficit", fs.DEFICIT_SIZE)):
        for ext in ("pdf", "png"):
            assert os.path.getsize(figs / f"{stem}.{ext}") > 1000
        pdf = open(figs / f"{stem}.pdf", "rb").read()
        assert b"/Type3" not in pdf
        m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf)
        assert float(m.group(1)) / 72 == pytest.approx(size[0], abs=0.01)
        assert float(m.group(2)) / 72 == pytest.approx(size[1], abs=0.01)
