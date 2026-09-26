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
  * the appendix table and the figure in both themes;
  * the Sep 26 amendments (Astra's review): a Freedman-Lane permutation that refits the covariates,
    the four-tic secondary on four-tic persistence at the full bootstrap, an every-row condition
    that needs the expected rows, the dated amended assessment beside the untouched pre-declared
    verdict, and filled training / open unseen markers with arena 7 named. The pre-declared numbers
    themselves must not move: a planted study pins them to the values the pre-amendment code gave.

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


def study(tmp, gain, persist, rows=("unet",), seed=0, with_windows=True, with_boot=True, missing=(),
          checks=None):
    """`gain(D, rng)` and `persist(D, rng)` give each window's gain and persistence PSNR for a map at D.

    Maps 2 to 5 are the training maps (the references), held out in the `val` set as in the study."""
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
    js = {"space": "sd1", "primary_arm": "motion", "maps": points, "references": {"maps": [2, 3, 4, 5]},
          "floor": {"motion": {"min": 0.09, "max": 0.13}}, "checks": checks or {"all_pass": True}}
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
    s = run(tmp_path, dist, scores, "--space", f"pixels={dist}", "--expect-rows", "unet,sd35")
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


# ---------------------------------------------------------------------------------------------
# the Sep 26 amendments (Astra's review)
# ---------------------------------------------------------------------------------------------

def noisy_falls(D, rng):
    return 1.0 - 1.0 * D + 0.4 * rng.normal()


# make_distance_figure.py at 1302230, before the amendments, on study(noisy_falls, seed=3), draws=500
PREDECLARED = {"partial_spearman": -0.7092693468287935, "ci95": [-0.8834351211796682, -0.41903576878441073],
               "raw_ci95": [-0.8890130671225566, -0.39529434111392736],
               "raw_psnr_ci95": [-0.6099587198576956, 0.045428825801207086], "loo_min": -0.761724348220076,
               "arena_ci95": [-0.9013128964386199, -0.3738005147261807],
               "campaign_ci95": [-0.8699382969600422, -0.051988767968104185],
               "sd35_ci95": [-0.9069127553378579, -0.4333154310146421],
               "lpips_ci95": [-0.8973651631888764, -0.592065933345942]}


def test_the_pre_declared_numbers_do_not_move(tmp_path):
    """Every pre-declared estimate and bootstrap CI reproduces bit for bit: the four-tic bootstrap now
    draws from its own stream, and the amended block from another, so neither shifts the shared one."""
    dist, scores, _ = study(tmp_path, noisy_falls, persist_indep, rows=("unet", "sd35"), seed=3)
    s = run(tmp_path, dist, scores, draws=500)
    p = s["primary"]
    got = {"partial_spearman": p["partial_spearman"], "ci95": p["ci95"], "raw_ci95": p["raw_ci95"],
           "raw_psnr_ci95": p["raw_psnr_ci95"], "loo_min": p["leave_one_out"]["min"],
           "arena_ci95": s["within_cluster"]["arena"]["ci95"],
           "campaign_ci95": s["within_cluster"]["campaign"]["ci95"], "sd35_ci95": s["rows"]["sd35"]["ci95"],
           "lpips_ci95": s["secondary"]["lpips_gain"]["ci95"]}
    for k, v in PREDECLARED.items():
        assert got[k] == pytest.approx(v, abs=1e-12), k


def _refit_residual(v, Z):
    return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]


def test_freedman_lane_refits_the_permuted_outcome():
    """Each permutation statistic is the partial correlation after the covariates' fitted values plus
    the permuted residuals are refitted, here by a plain least-squares refit; the Kennedy-style version
    the pre-declared code ran skipped the refit and so understated every permutation statistic."""
    rng = np.random.default_rng(0)
    n = 15
    z = rng.normal(size=(n, 2))
    x = z @ np.array([1.0, 0.5]) + rng.normal(size=n)
    y = z @ np.array([0.3, 1.0]) + rng.normal(size=n)
    rx, ry = mdf.rank_average(x), mdf.rank_average(y)
    Z = np.column_stack([np.ones(n)] + [mdf.rank_average(c) for c in z.T])
    ex, ey = _refit_residual(rx, Z), _refit_residual(ry, Z)
    perms = [rng.permutation(n) for _ in range(40)]
    want = np.array([abs(np.corrcoef(ex, _refit_residual((ry - ey) + ey[q], Z))[0, 1]) for q in perms])
    assert np.allclose(mdf.freedman_lane_null(x, y, z, perms), want, atol=1e-12)
    kennedy = np.array([abs(np.corrcoef(ex, ey[q])[0, 1]) for q in perms])
    assert np.all(want >= kennedy - 1e-12) and np.max(want - kennedy) > 1e-3
    # the observed statistic is the partial Spearman on both covariates
    assert np.corrcoef(ex, ey)[0, 1] == pytest.approx(mdf.partial_spearman(x, y, z), abs=1e-12)
    # one covariate, given flat or as a column, is the pre-declared statistic
    assert mdf.partial_spearman(x, y, z[:, 0]) == pytest.approx(mdf.partial_spearman(x, y, z[:, :1]), abs=1e-12)


def test_freedman_lane_holds_its_level_where_kennedy_does_not():
    """A planted null: x and y share five covariates and nothing else, on 12 maps. Permuting the
    residuals without the refit rejects it far more often than 5 percent; Freedman-Lane does not."""
    rng, krng = np.random.default_rng(2), np.random.default_rng(3)
    n, k, reps, perms = 12, 5, 200, 99
    fl = kennedy = 0
    for _ in range(reps):
        zs = np.column_stack([rng.permutation(n).astype(float) for _ in range(k)])
        shared = zs.sum(1)
        x = shared + 0.7 * shared.std() * rng.normal(size=n)
        y = shared + 0.7 * shared.std() * rng.normal(size=n)
        fl += mdf.permutation_p(x, y, zs, perms, rng) <= 0.05
        Z = np.column_stack([np.ones(n)] + [mdf.rank_average(c) for c in zs.T])
        ex, ey = _refit_residual(mdf.rank_average(x), Z), _refit_residual(mdf.rank_average(y), Z)
        obs = abs(np.corrcoef(ex, ey)[0, 1])
        null = [abs(np.corrcoef(ex, ey[krng.permutation(n)])[0, 1]) for _ in range(perms)]
        kennedy += (1 + np.sum(np.array(null) >= obs - 1e-12)) / (perms + 1) <= 0.05
    assert fl / reps <= 0.09, fl / reps
    assert kennedy / reps >= 0.13, kennedy / reps


def test_the_four_tic_secondary_uses_four_tic_persistence_and_the_full_bootstrap(tmp_path):
    dist, scores, _ = study(tmp_path, falls, persist_indep, seed=4)
    s = run(tmp_path, dist, scores, draws=400)
    r = [m["rows"]["unet"] for m in s["maps"]]
    D = np.array([m["D"] for m in s["maps"]])
    g4, p4, p1 = (np.array([x[k] for x in r]) for k in ("gain_h4", "persist_h4", "persist"))
    h4 = s["secondary"]["gain_h4"]
    assert h4["partial_spearman"] == pytest.approx(mdf.partial_spearman(D, g4, p4), abs=1e-12)
    assert abs(h4["partial_spearman"] - mdf.partial_spearman(D, g4, p1)) > 1e-6
    assert h4["covariate"] == "four-tic mean persist_psnr_raw" and h4["draws"] == 400
    # the four-tic episode bootstrap runs at the full draw count, so each map's interval is a real one
    assert all(x["gain_h4_ci"][0] < x["gain_h4"] < x["gain_h4_ci"][1] for x in r)


def test_every_row_needs_the_expected_rows(tmp_path):
    """One row supplied is not "every row": the U-Net alone leaves the condition pending."""
    dist, scores, D = study(tmp_path, falls, persist_indep)
    s = run(tmp_path, dist, scores, draws=300)
    v = s["verdict"]
    assert v["conditions"]["every_row"] is None
    assert v["rows_expected"] == ["unet", "sd35", "pixart"] and v["rows_missing"] == ["sd35", "pixart"]
    s = run(tmp_path, dist, scores, "--expect-rows", "unet", draws=300)
    assert s["verdict"]["conditions"]["every_row"] is True and s["verdict"]["rows_missing"] == []
    # a supplied row that rises with distance fails the condition even while another is missing
    dist, scores, D = study(tmp_path / "b", falls, persist_indep, rows=("unet", "sd35"))
    for d in scores["sd35"].glob("*_h1"):
        m = int(d.name.split("_map")[1][:2])
        rows = list(csv.DictReader(open(d / "per_window.csv")))
        for row in rows:
            row["psnr_raw"] = str(float(row["persist_psnr_raw"]) + 2.0 * D[m])
        with open(d / "per_window.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    s = run(tmp_path / "b", dist, scores, draws=300)
    assert s["verdict"]["conditions"]["every_row"] is False and s["verdict"]["rows_missing"] == ["pixart"]


CHECKS_AS_FROZEN = {"i": {"pass": True}, "ii": {"pass": False}, "iii": {"pass": True},
                    "iv": {"pass": False, "max_same_wad": 0.68, "min_campaign": 0.8}, "v": {"pass": True},
                    "all_pass": False}
CHECKS_ALL_PASS = {"i": {"pass": True}, "ii": {"pass": True}, "iii": {"pass": True}, "iv": {"pass": True},
                   "v": {"pass": True}, "all_pass": True}


def by_membership(D, rng):
    """The gain steps down with the map's group (training, unseen arena, campaign) and not with D."""
    return (1.5 if D < 0.15 else 0.5 if D < 0.75 else -0.5) + 0.4 * rng.normal()


def _covariates(maps):
    training = np.array([m["map"] in (2, 3, 4, 5) for m in maps], dtype=float)
    campaign = np.array([m["cluster"] == "campaign" for m in maps], dtype=float)
    return training, campaign


def test_the_amended_assessment_separates_the_pooled_association_from_map_membership(tmp_path, capsys):
    """Astra's finding, planted: the pooled association comes from which group a map is in. The pre-declared
    verdict stays as it was; the dated amendment reports qualified pooled evidence and an inconclusive
    unseen-only trend, and separates the measurement checks from the map assumption and the subset tolerance."""
    dist, scores, _ = study(tmp_path, by_membership, persist_indep, seed=5, checks=CHECKS_AS_FROZEN)
    s = run(tmp_path, dist, scores, draws=1000)
    assert s["verdict"]["result"] == "does not support"
    assert s["primary"]["ci95"][1] < 0
    am = s["amended_assessment"]
    assert am["date"] == "2026-09-26" and "specified after observing results" in am["status"]
    assert am["original_verdict"] == "does not support"
    assert am["measurement_checks"] == {"checks": ["i", "iii", "v"], "results": {"i": True, "iii": True, "v": True},
                                        "pass": True}
    assert am["map_assumption"]["check"] == "iv" and am["map_assumption"]["pass"] is False
    assert am["subset_tolerance"]["check"] == "ii" and am["subset_tolerance"]["pass"] is False
    c = am["conditions"]
    assert c["primary_ci_excludes_zero"] == {"basis": "pre-declared", "value": True}
    assert c["measurement_checks_i_iii_v"] == {"basis": "amended", "value": True}
    assert c["unseen_only_family_ci_below_zero"] == {"basis": "amended", "value": False}
    assert c["every_expected_row_negative"] == {"basis": "amended", "value": None}
    assert c["bootstrap_covers_full_estimator"] == {"basis": "amended", "value": False}
    summary = ("qualified evidence for the pooled U-Net association; original gate failed; "
               "unseen-only trend inconclusive; cross-backbone replication pending")
    assert am["summary"] == summary and summary in capsys.readouterr().out

    # every structural control is the partial Spearman on the covariates rebuilt here
    maps = s["maps"]
    D = np.array([m["D"] for m in maps])
    g = np.array([m["rows"]["unet"]["gain"] for m in maps])
    p = np.array([m["rows"]["unet"]["persist"] for m in maps])
    training, campaign = _covariates(maps)
    neff = np.array([m["n_eff"] for m in maps])
    dsd = np.array([m["D_sd"] for m in maps])
    unseen = training == 0
    want = {"training_indicator": (30, mdf.partial_spearman(D, g, np.column_stack([p, training]))),
            "training_indicator_family": (30, mdf.partial_spearman(D, g, np.column_stack([p, training, campaign]))),
            "unseen_only_family": (26, mdf.partial_spearman(D[unseen], g[unseen],
                                                            np.column_stack([p[unseen], campaign[unseen]]))),
            "n_eff": (30, mdf.partial_spearman(D, g, np.column_stack([p, neff]))),
            "bootstrap_sd": (30, mdf.partial_spearman(D, g, np.column_stack([p, dsd])))}
    sc = am["structural_controls"]
    assert set(sc) == set(want)
    for k, (n, rho) in want.items():
        assert sc[k]["n_maps"] == n and sc[k]["partial_spearman"] == pytest.approx(rho, abs=1e-12), k
    # the planted membership effect is what carries the pooled number
    assert abs(sc["unseen_only_family"]["partial_spearman"]) < abs(s["primary"]["partial_spearman"])
    assert sc["unseen_only_family"]["ci95"][0] < 0 < sc["unseen_only_family"]["ci95"][1]


def test_an_association_that_holds_everywhere_is_reported_as_such(tmp_path):
    dist, scores, _ = study(tmp_path, falls, persist_indep, rows=("unet", "sd35", "pixart"), checks=CHECKS_ALL_PASS)
    s = run(tmp_path, dist, scores, "--space", f"pixels={dist}", draws=500)
    assert s["verdict"]["result"] == "supports"
    am = s["amended_assessment"]
    assert am["summary"] == ("evidence for the pooled U-Net association; original gate passed; "
                             "unseen-only trend holds; cross-backbone replication holds")
    assert am["conditions"]["every_expected_row_negative"]["value"] is True


def test_a_full_estimator_bootstrap_beside_the_distances_is_used(tmp_path):
    """`distance_study.py` writes bootstrap_full_<space>.npz from Sep 26; the amendment then reads D's
    uncertainty from the estimator the study reports rather than from single 4-episode clouds."""
    dist, scores, _ = study(tmp_path, falls, persist_indep, checks=CHECKS_ALL_PASS)
    js = json.load(open(dist))
    rng = np.random.default_rng(9)
    keys = [p["key"] for p in js["maps"]]
    np.savez(dist.parent / "bootstrap_full_sd1.npz", keys=np.array(keys),
             draws=np.array([p["D"] + 0.001 * rng.normal(size=40) for p in js["maps"]]), arm=np.array("motion"))
    s = run(tmp_path, dist, scores, draws=300)
    am = s["amended_assessment"]
    assert am["conditions"]["bootstrap_covers_full_estimator"]["value"] is True
    full = am["primary_with_full_bootstrap"]
    assert full["partial_spearman"] == pytest.approx(s["primary"]["partial_spearman"], abs=1e-12)
    assert full["ci95"][1] < 0


def test_training_maps_are_filled_unseen_maps_open_and_arena_7_is_named(tmp_path):
    t = mdf.THEMES["light"]
    filled = mdf.marker_style({"training": True}, "#2a78d6", t)
    hollow = mdf.marker_style({"training": False}, "#2a78d6", t)
    assert filled["mfc"] == "#2a78d6"
    assert hollow["mfc"] == t["surface"] and hollow["mec"] == "#2a78d6"
    dist, scores, _ = study(tmp_path, falls, persist_indep)
    s = run(tmp_path, dist, scores, draws=200)
    assert {m["key"] for m in s["maps"] if m["training"]} == {"val/2", "val/3", "val/4", "val/5"}
    for name in ("distance_gain.svg", "distance_gain_dark.svg"):
        svg = (tmp_path / "fig" / name).read_text()
        assert "arena 7" in svg and "training map" in svg and "unseen map" in svg
