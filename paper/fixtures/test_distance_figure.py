"""`paper/make_distance_figure.py`: the distance study's statistics, appendix table and figure.

Every test builds the study's inputs by hand with a known relation between distance and quality: a
`distances_<space>.json` and its bootstrap draws in the layout `distance_study.py distances` writes,
and per-map score directories in the layout `score_distance_maps.sh` writes (`metrics.json` and
`per_window.csv` from `eval_tf.py`). The statistics must then find what was planted:

  * the pre-declared primary statistic, the partial Spearman correlation of distance with the gain
    over persistence controlling for the map's persistence PSNR, negative with a bootstrap CI that
    excludes zero when the gain truly falls with distance, and near zero when the gain tracks motion
    alone (the relation that "vanishes under motion control");
  * the leave-one-map-out sign check, the within-cluster and per-row checks, the pixel-space
    agreement, and the verdict that combines them as the memo's section 5 lists;
  * the appendix table and the figure in both themes.

    python -m pytest paper/fixtures/test_distance_figure.py -q
"""
import csv
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
sys.path.insert(0, PAPER)
sys.path.insert(0, os.path.dirname(PAPER))

pytest.importorskip("matplotlib")
import make_distance_figure as mdf  # noqa: E402

ARENAS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
CAMPAIGN = [18, 19, 20, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32]


def set_of(m):
    return "val" if m in (2, 3, 4, 5) else "arenas_678" if m in (6, 7, 8) else \
        "unseen" if m in (16, 17) else "seen" if m < 18 else "unseen2"


def study(tmp, gain, persist, rows=("unet",), seed=0, with_windows=True, with_boot=True, missing=()):
    """`gain(D, rng)` and `persist(D, rng)` give each window's gain and persistence PSNR for a map at D."""
    rng = np.random.default_rng(seed)
    maps = ARENAS + CAMPAIGN
    D = {m: (0.1 + 0.02 * rng.random()) if m in (2, 3, 4, 5) else
         (0.3 + 0.1 * rng.random()) if m < 18 else (0.8 + 0.6 * rng.random()) for m in maps}
    # arenas far enough apart inside their cluster that a within-cluster relation exists too
    for i, m in enumerate(ARENAS):
        if m not in (2, 3, 4, 5):
            D[m] = 0.2 + 0.03 * i
    points = [{"key": f"{set_of(m)}/{m}", "set": set_of(m), "map": m, "role": "primary",
               "cluster": "campaign" if m >= 18 else "arena", "D": D[m], "nearest": 2, "n_eff": 400.0,
               "lives": 5.0, "valid_fraction": 0.9, "episodes": list(range(4))} for m in maps]
    points.append({"key": "seen/2", "set": "seen", "map": 2, "role": "replication", "cluster": "arena",
                   "D": D[2], "nearest": 2, "n_eff": 400.0, "lives": 5.0, "valid_fraction": 0.9, "episodes": [0]})
    out = tmp / "distances"
    out.mkdir(parents=True, exist_ok=True)
    js = {"space": "sd1", "primary_arm": "motion", "maps": points,
          "floor": {"motion": {"min": 0.09, "max": 0.13}}, "checks": {"all_pass": True}}
    (out / "distances_sd1.json").write_text(json.dumps(js))
    if with_boot:
        draws = np.array([p["D"] + 0.01 * rng.normal(size=50) for p in points])
        np.savez(out / "bootstrap_sd1.npz", keys=np.array([p["key"] for p in points]), draws=draws,
                 arm=np.array("motion"))
    score_dirs = {}
    for r, row in enumerate(rows):
        root = tmp / "scores" / row
        for m in maps:
            if (row, m) in missing:
                continue
            for k in (1, 4):
                d = root / f"{set_of(m)}_map{m:02d}_h{k}"
                d.mkdir(parents=True, exist_ok=True)
                n = 64
                eps = np.repeat(np.arange(8), n // 8)
                p = persist(D[m], rng) + rng.normal(0, 1.0, size=n)
                g = gain(D[m], rng) * (0.8 if k == 4 else 1.0) + rng.normal(0, 0.3, size=n) - 0.05 * r
                lp = 0.3 + rng.normal(0, 0.02, size=n)
                lg = 0.05 * gain(D[m], rng) + rng.normal(0, 0.01, size=n)
                rows_ = [{"episode": int(e), "map": m, "psnr_raw": float(pp + gg), "persist_psnr_raw": float(pp),
                          "lpips_raw": float(ll - lgg), "persist_lpips_raw": float(ll)}
                         for e, pp, gg, ll, lgg in zip(eps, p, g, lp, lg)]
                met = {"psnr_raw": {"mean": float(np.mean(p + g))}, "persist_psnr_raw": {"mean": float(np.mean(p))},
                       "lpips_raw": {"mean": float(np.mean(lp - lg))},
                       "persist_lpips_raw": {"mean": float(np.mean(lp))}}
                (d / "metrics.json").write_text(json.dumps(met))
                if with_windows:
                    with open(d / "per_window.csv", "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(rows_[0]))
                        w.writeheader()
                        w.writerows(rows_)
        score_dirs[row] = root
    return out / "distances_sd1.json", score_dirs, D


def falls(D, rng):
    return 2.0 - 2.0 * D


def flat(D, rng):
    return 0.8


def persist_indep(D, rng):
    return 19.0 + rng.normal(0, 0.8)


def run(tmp, dist, scores, *extra, draws=2000):
    args = ["--distances", str(dist), "--out", str(tmp / "fig"), "--draws", str(draws), "--permutations", "2000"]
    for row, d in scores.items():
        args += ["--row", f"{row}={d}"]
    assert mdf.main(args + [str(a) for a in extra]) == 0
    return json.load(open(tmp / "fig" / "stats.json"))


def test_partial_spearman_removes_what_the_covariate_explains():
    rng = np.random.default_rng(0)
    z = rng.normal(size=300)
    x = z + 0.3 * rng.normal(size=300)
    y = z + 0.3 * rng.normal(size=300)                  # y depends on x only through z
    assert mdf.spearman(x, y) > 0.8
    assert abs(mdf.partial_spearman(x, y, z)) < 0.15
    y2 = -x + 0.3 * rng.normal(size=300)
    assert mdf.partial_spearman(x, y2, z) < -0.5
    # a constant covariate is no covariate
    assert mdf.partial_spearman(x, y2, np.ones(300)) == pytest.approx(mdf.spearman(x, y2), abs=1e-12)


def test_a_gain_that_falls_with_distance_is_found_and_supported(tmp_path):
    dist, scores, D = study(tmp_path, falls, persist_indep, rows=("unet", "sd35"))
    s = run(tmp_path, dist, scores, "--space", f"pixels={dist}")
    p = s["primary"]
    assert p["row"] == "unet" and p["n_maps"] == 30
    assert p["partial_spearman"] < -0.7 and p["ci95"][1] < 0 and p["permutation_p"] < 0.01
    assert p["leave_one_out"]["n"] == 30 and p["leave_one_out"]["same_sign"] == 30 and p["leave_one_out"]["pass"]
    assert s["within_cluster"]["arena"]["partial_spearman"] < 0 and s["within_cluster"]["campaign"]["n_maps"] == 13
    assert all(r["partial_spearman"] < 0 for r in s["rows"].values()) and set(s["rows"]) == {"unet", "sd35"}
    assert s["spaces"]["pixels"]["kendall_tau_with_primary"] == pytest.approx(1.0)
    assert s["verdict"]["result"] == "supports", s["verdict"]
    # the secondaries the memo pre-declares
    assert s["secondary"]["lpips_gain"]["partial_spearman"] < 0 and s["secondary"]["gain_h4"]["partial_spearman"] < 0
    # the replication copy of map 2 is not a point of the study
    assert "seen/2" not in {m["key"] for m in s["maps"]}
    # the table and both themes of the figure
    table = (tmp_path / "fig" / "distance_table.md").read_text()
    assert table.count("\n| ") >= 30 and "unseen2/18" in table
    for name in ("distance_gain.png", "distance_gain.svg", "distance_gain_dark.png", "distance_gain_dark.svg"):
        assert (tmp_path / "fig" / name).stat().st_size > 1000
    assert s["figure"]["panels"] == 1


def test_no_relation_is_not_supported(tmp_path):
    dist, scores, _ = study(tmp_path, flat, persist_indep, seed=1)
    s = run(tmp_path, dist, scores)
    assert s["primary"]["ci95"][0] < 0 < s["primary"]["ci95"][1]
    assert s["verdict"]["result"] == "does not support"


def test_a_relation_carried_by_motion_vanishes_under_motion_control(tmp_path):
    # far maps are calmer (higher persistence PSNR), and the gain depends on persistence alone
    dist, scores, _ = study(tmp_path, lambda D, r: 0.0, lambda D, r: 18.0 + 3.0 * D + 0.1 * r.normal(), seed=2)

    # the gain is a function of the window's persistence (plus window noise), not of distance
    noise = np.random.default_rng(5)
    for d in scores["unet"].glob("*_h1"):
        rows = list(csv.DictReader(open(d / "per_window.csv")))
        for r in rows:
            g = 4.0 - 0.2 * float(r["persist_psnr_raw"]) + noise.normal(0, 0.3)
            r["psnr_raw"] = str(float(r["persist_psnr_raw"]) + g)
        with open(d / "per_window.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    s = run(tmp_path, dist, scores)
    assert s["primary"]["raw_spearman"] < -0.5
    assert abs(s["primary"]["partial_spearman"]) < abs(s["primary"]["raw_spearman"]) / 2
    assert s["verdict"]["result"] == "does not support"
    assert s["verdict"]["conditions"]["negative_with_ci_excluding_zero"] is False


def test_metrics_json_alone_is_enough_and_a_missing_map_is_recorded(tmp_path):
    dist, scores, D = study(tmp_path, falls, persist_indep, with_windows=False, with_boot=False,
                            missing={("unet", 19)})
    s = run(tmp_path, dist, scores, draws=500)
    assert s["primary"]["n_maps"] == 29 and s["rows"]["unet"]["missing"] == ["unseen2/19"]
    m = {x["key"]: x for x in s["maps"]}["val/2"]
    d = scores["unet"] / "val_map02_h1"
    met = json.load(open(d / "metrics.json"))
    assert m["rows"]["unet"]["gain"] == pytest.approx(met["psnr_raw"]["mean"] - met["persist_psnr_raw"]["mean"])
    assert s["primary"]["partial_spearman"] < -0.5
    # without per-window data the pixel space and the episode bootstrap are simply absent
    assert s["verdict"]["conditions"]["pixel_space"] is None and s["verdict"]["result"] == "incomplete"


def test_more_than_three_rows_are_drawn_as_small_multiples(tmp_path):
    dist, scores, _ = study(tmp_path, falls, persist_indep, rows=("a", "b", "c", "d"))
    s = run(tmp_path, dist, scores, draws=300)
    assert s["figure"]["panels"] == 4 and s["figure"]["palette"]["light"][:1] == ["#2a78d6"]
