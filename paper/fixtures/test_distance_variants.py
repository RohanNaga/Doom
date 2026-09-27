"""`tools/distance_variants.py`: the distance-variants table on the frozen clouds.

Rohan asked whether a KL divergence, or another distance, would do better than the sliced Wasserstein
frame distance D. The tool recomputes the table for nine alternatives with D's exact cloud and floor
construction. These tests pin every estimator against a closed form on synthetic clouds: the Frechet
distance and the Gaussian KL divergence on Gaussians, the sliced W2 on shifted point masses, the
unbiased MMD on identical clouds and on two Gaussians, and the Wang, Kulkarni and Verdu kNN KL
estimator on two Gaussians. The batched engine the pipeline runs is checked against the standalone
estimators, and the command line runs end to end on a small synthetic corpus. numpy and scipy only.

    python -m pytest paper/fixtures/test_distance_variants.py -q
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, REPO)

import distance_variants as dv  # noqa: E402
from distance_study import random_directions, save_cloud  # noqa: E402


def gaussian(n, mean, cov, seed):
    return np.random.default_rng(seed).multivariate_normal(mean, cov, size=n)


def random_spd(d, seed):
    a = np.random.default_rng(seed).standard_normal((d, d))
    return a @ a.T / d + 0.5 * np.eye(d)


# ---------------------------------------------------------------------------------------------
# Frechet distance (the FID formula) between Gaussian fits
# ---------------------------------------------------------------------------------------------

def test_frechet_is_the_closed_form_for_diagonal_gaussians():
    # commuting covariances: ||m0 - m1||^2 + sum (sqrt a - sqrt b)^2
    m0, m1 = np.array([0.0, 1.0, -2.0]), np.array([1.0, 1.0, 0.5])
    a, b = np.array([1.0, 4.0, 0.25]), np.array([9.0, 1.0, 0.25])
    want = np.sum((m0 - m1) ** 2) + np.sum((np.sqrt(a) - np.sqrt(b)) ** 2)
    assert dv.frechet_distance(m0, np.diag(a), m1, np.diag(b)) == pytest.approx(want, rel=1e-12)


def test_frechet_matches_the_matrix_square_root_formula_for_full_covariances():
    scipy_linalg = pytest.importorskip("scipy.linalg")
    s0, s1 = random_spd(6, 0), random_spd(6, 1)
    m0, m1 = np.arange(6.0), np.ones(6)
    want = np.sum((m0 - m1) ** 2) + np.trace(s0 + s1 - 2 * np.real(scipy_linalg.sqrtm(s0 @ s1)))
    assert dv.frechet_distance(m0, s0, m1, s1) == pytest.approx(want, rel=1e-9)
    assert dv.frechet_distance(m1, s1, m0, s0) == pytest.approx(want, rel=1e-9)     # symmetric
    assert dv.frechet_distance(m0, s0, m0, s0) == pytest.approx(0.0, abs=1e-10)


def test_frechet_on_gaussian_samples_converges_to_the_closed_form():
    d = 4
    x = gaussian(40_000, np.zeros(d), np.eye(d), 0)
    y = gaussian(40_000, np.full(d, 0.5), 4.0 * np.eye(d), 1)
    want = d * 0.25 + d * (1.0 - 2.0) ** 2                  # ||mu||^2 + sum (1 - 2)^2
    got = dv.frechet_distance(*dv.gaussian_fit(x), *dv.gaussian_fit(y))
    assert got == pytest.approx(want, rel=0.03)


# ---------------------------------------------------------------------------------------------
# Gaussian KL divergence, both directions
# ---------------------------------------------------------------------------------------------

def kl_1d(m0, s0, m1, s1):
    return np.log(s1 / s0) + (s0 ** 2 + (m0 - m1) ** 2) / (2 * s1 ** 2) - 0.5


def test_gaussian_kl_is_the_closed_form_and_is_directed():
    m0, m1 = np.array([0.0, 1.0]), np.array([1.0, -1.0])
    sd0, sd1 = np.array([1.0, 2.0]), np.array([3.0, 0.5])
    want01 = sum(kl_1d(*v) for v in zip(m0, sd0, m1, sd1))
    want10 = sum(kl_1d(*v) for v in zip(m1, sd1, m0, sd0))
    assert dv.gaussian_kl(m0, np.diag(sd0 ** 2), m1, np.diag(sd1 ** 2)) == pytest.approx(want01, rel=1e-12)
    assert dv.gaussian_kl(m1, np.diag(sd1 ** 2), m0, np.diag(sd0 ** 2)) == pytest.approx(want10, rel=1e-12)
    assert want01 != pytest.approx(want10)
    s = random_spd(5, 3)
    assert dv.gaussian_kl(np.ones(5), s, np.ones(5), s) == pytest.approx(0.0, abs=1e-10)


def test_gaussian_kl_on_full_covariances_matches_the_textbook_formula():
    s0, s1 = random_spd(5, 4), random_spd(5, 5)
    m0, m1 = np.zeros(5), np.linspace(-1, 1, 5)
    i1 = np.linalg.inv(s1)
    want = 0.5 * (np.trace(i1 @ s0) + (m1 - m0) @ i1 @ (m1 - m0) - 5
                  + np.linalg.slogdet(s1)[1] - np.linalg.slogdet(s0)[1])
    assert dv.gaussian_kl(m0, s0, m1, s1) == pytest.approx(want, rel=1e-10)


def test_the_ridge_adds_a_scaled_identity():
    s = np.diag([1.0, 3.0])
    assert np.allclose(dv.ridge(s, 0.1), s + 0.1 * 2.0 * np.eye(2))       # 0.1 x mean variance
    assert dv.ridge(s, 0.0) is s


# ---------------------------------------------------------------------------------------------
# sliced W2 (distance_study.distance_table, the engine D runs on) on shifted point masses
# ---------------------------------------------------------------------------------------------

def test_sliced_w2_between_point_masses_is_the_projected_shift():
    d = 8
    a, b = np.zeros(d), np.linspace(0.5, 2.0, d)
    dirs = random_directions(d, 400, seed=0)
    x, y = np.tile(a, (30, 1)), np.tile(b, (50, 1))
    got = dv.distance_table([(x, {"motion": np.random.default_rng(0).random(30), "uniform": None})],
                            {"ref": (y, {"motion": np.random.default_rng(1).random(50), "uniform": None})},
                            dirs, ("motion", "uniform"))
    # every 1-D transport moves all the mass by theta . (b - a), whatever the weights
    want = np.sqrt(np.mean((dirs.T @ (b - a)) ** 2))
    assert got[0, 0, 0] == pytest.approx(want, rel=1e-12)
    assert got[1, 0, 0] == pytest.approx(want, rel=1e-12)
    # and over the uniform sphere E[(theta . u)^2] = |u|^2 / d
    many = dv.distance_table([(x, None)], {"ref": (y, None)}, random_directions(d, 20_000, seed=1), ("uniform",))
    assert many[0, 0, 0] == pytest.approx(np.linalg.norm(b - a) / np.sqrt(d), rel=0.02)


# ---------------------------------------------------------------------------------------------
# unbiased MMD^2 with a Gaussian kernel
# ---------------------------------------------------------------------------------------------

def test_mmd_of_a_cloud_with_itself_is_about_zero():
    x = gaussian(600, np.zeros(3), np.eye(3), 0)
    bw = dv.median_bandwidth(x, max_points=600, seed=0)
    # the unbiased estimator of a cloud against itself is -(2/n)(1 - mean off-diagonal kernel), O(1/n)
    assert abs(dv.mmd2_unbiased(x, x, bw)) < 2.0 / len(x)
    y = gaussian(600, np.zeros(3), np.eye(3), 1)
    assert abs(dv.mmd2_unbiased(x, y, bw)) < 0.01


def test_mmd_matches_the_gaussian_closed_form():
    # E k(X, Y) for X ~ N(m0, S0), Y ~ N(m1, S1), k = exp(-|x - y|^2 / (2 s^2)):
    # det(I + (S0 + S1) / s^2)^(-1/2) exp(-(m0 - m1)' (s^2 I + S0 + S1)^(-1) (m0 - m1) / 2)
    d, s = 2, 1.5
    m0, m1, s0, s1 = np.zeros(d), np.array([1.0, 0.5]), np.eye(d), np.diag([0.5, 2.0])

    def ek(ma, sa, mb, sb):
        c = sa + sb
        dm = ma - mb
        quad = dm @ np.linalg.solve(s ** 2 * np.eye(d) + c, dm)
        return np.linalg.det(np.eye(d) + c / s ** 2) ** -0.5 * np.exp(-0.5 * quad)
    want = ek(m0, s0, m0, s0) + ek(m1, s1, m1, s1) - 2 * ek(m0, s0, m1, s1)
    x, y = gaussian(4000, m0, s0, 2), gaussian(4000, m1, s1, 3)
    assert dv.mmd2_unbiased(x, y, s) == pytest.approx(want, abs=0.006)


def test_the_median_bandwidth_is_the_median_pairwise_distance():
    y = np.array([[0.0], [1.0], [3.0]])           # pairwise distances 1, 3, 2
    assert dv.median_bandwidth(y, max_points=10, seed=0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------------------------
# kNN KL (Wang, Kulkarni and Verdu 2009)
# ---------------------------------------------------------------------------------------------

def test_knn_kl_converges_to_the_gaussian_kl_in_both_directions():
    # The estimator is consistent but biased low at finite n wherever P's tails reach into sparse
    # regions of Q (over five seeds: 3 and 6 percent low at n = 20,000 for this pair, 4 and 9 at 6,000;
    # a pair with a 4:1 variance ratio is still 6 percent low at 60,000). A mild pair pins the formula.
    d = 2
    m0, m1 = np.zeros(d), np.array([1.0, 0.0])
    s0, s1 = np.eye(d), np.diag([1.5, 0.75])
    x, y = gaussian(20_000, m0, s0, 4), gaussian(20_000, m1, s1, 5)
    fwd, rev = dv.gaussian_kl(m0, s0, m1, s1), dv.gaussian_kl(m1, s1, m0, s0)       # 0.392 and 0.566 nats
    assert dv.knn_kl(x, y, k=5) == pytest.approx(fwd, rel=0.10)
    assert dv.knn_kl(y, x, k=5) == pytest.approx(rev, rel=0.10)
    assert dv.knn_kl(y, x, k=5) > dv.knn_kl(x, y, k=5)                              # directed, as KL is
    # the same distribution: about zero, with no sample-size bias left (the log(m / (n - 1)) term)
    z = gaussian(2000, m0, s0, 6)
    assert dv.knn_kl(x, z, k=5) == pytest.approx(0.0, abs=0.05)


def test_knn_kl_excludes_the_point_itself_but_not_its_duplicates():
    x = np.random.default_rng(0).standard_normal((50, 3))
    y = np.random.default_rng(1).standard_normal((70, 3))
    xd = np.vstack([x, x[:1]])                   # a duplicated point: its own nearest neighbour is at 0
    assert np.isfinite(dv.knn_kl(xd, y, k=2))
    with pytest.raises(ValueError, match="zero"):
        dv.knn_kl(np.vstack([x] + [x[:1]] * 2), y, k=2)     # three copies: the 2nd neighbour is at 0


# ---------------------------------------------------------------------------------------------
# the batched engine the pipeline runs equals the standalone estimators
# ---------------------------------------------------------------------------------------------

def test_the_engine_agrees_with_the_standalone_estimators():
    x = gaussian(120, np.zeros(4), random_spd(4, 7), 8)
    y = gaussian(300, np.full(4, 0.3), random_spd(4, 9), 10)
    bw = 1.7
    ref = dv.Reference(y, bandwidth=bw, k=5, ridge_frac=0.0, block=64)
    got = dv.pair_variants(dv.CloudStats(x, bandwidth=bw, k=5, ridge_frac=0.0), ref)
    mx, sx = dv.gaussian_fit(x)
    my, sy = dv.gaussian_fit(y)
    assert got["frechet"] == pytest.approx(dv.frechet_distance(mx, sx, my, sy), rel=1e-9)
    assert got["gauss_kl_ts"] == pytest.approx(dv.gaussian_kl(mx, sx, my, sy), rel=1e-9)
    assert got["gauss_kl_st"] == pytest.approx(dv.gaussian_kl(my, sy, mx, sx), rel=1e-9)
    assert got["mmd2"] == pytest.approx(dv.mmd2_unbiased(x, y, bw), rel=1e-9, abs=1e-12)
    assert got["knn_kl_ts"] == pytest.approx(dv.knn_kl(x, y, k=5), rel=1e-9)
    assert got["knn_kl_st"] == pytest.approx(dv.knn_kl(y, x, k=5), rel=1e-9)


# ---------------------------------------------------------------------------------------------
# the command line, end to end, on a synthetic corpus
# ---------------------------------------------------------------------------------------------

DIM = 6
TRAIN_MAPS = (2, 3, 4, 5)


def map_mean(m):
    v = np.zeros(DIM)
    v[m % DIM] = 1.5
    return v


def write_set(path, name, maps, episodes_per_map, first_episode, shift=None, frames=25, seed=0):
    rng = np.random.default_rng(seed)
    feats, motion, episode, mp = [], [], [], []
    ep_id, ep_map = [], []
    e = first_episode
    for m in maps:
        base = map_mean(3) + (shift or {}).get(m, 0.0) if m not in TRAIN_MAPS else map_mean(m)
        for _ in range(episodes_per_map):
            feats.append(base + rng.standard_normal((frames, DIM)) * 0.5)
            motion.append(rng.random(frames) + 0.05)
            episode.append(np.full(frames, e))
            mp.append(np.full(frames, m))
            ep_id.append(e)
            ep_map.append(m)
            e += 1
    n_ep = len(ep_id)
    save_cloud(path, {"feats": np.concatenate(feats).astype(np.float32), "motion": np.concatenate(motion),
                      "episode": np.concatenate(episode), "map": np.concatenate(mp)},
               {"id": np.array(ep_id), "map": np.array(ep_map), "drawn": np.full(n_ep, frames)},
               {"set": name, "space": "sd1"})


def test_the_command_line_builds_the_table(tmp_path):
    ref, floor = tmp_path / "train_draw1.npz", tmp_path / "train_draw2.npz"
    arenas, val = tmp_path / "arenas.npz", tmp_path / "val.npz"
    write_set(ref, "train_draw1", TRAIN_MAPS, 8, 0, seed=1)
    write_set(floor, "train_draw2", TRAIN_MAPS, 8, 1000, seed=2)
    write_set(val, "val", TRAIN_MAPS, 8, 2000, seed=3)
    shifts = {11: 0.6, 12: 1.2, 13: 2.4}                       # arena 13 is the farthest from map 3
    write_set(arenas, "arenas", list(shifts), 8, 3000, shift=shifts, seed=4)
    adapt = tmp_path / "adapt_summary.json"
    per_arena = [{"arena": m, "A0": 3.0 - s, "A_budget": 4.0 - 0.1 * s, "gain": 1.0 + 0.9 * s, "S0": 2.0 - 0.2 * s,
                  "cost_half_gap": None if m == 11 else 250 * (m - 11), "censored_half_gap": m == 11}
                 for m, s in shifts.items()]
    adapt.write_text(json.dumps({"per_arena": per_arena, "spearman_cost_censored_as": 8000}))
    out = tmp_path / "out"
    rc = dv.main(["--reference", str(ref), "--floor", str(floor), "--target", str(arenas), "--target", str(val),
                  "--arena-set", "arenas", "--validation-set", "val", "--adapt", str(adapt),
                  "--projections", "200", "--bandwidth-points", "400", "--out", str(out)])
    assert rc == 0
    res = json.loads((out / "variants_sd1.json").read_text())
    assert (out / "variants_sd1.csv").is_file() and (out / "VARIANTS.md").is_file()
    assert set(res["variants"]) == set(dv.VARIANTS)
    for name, v in res["variants"].items():
        vals = [v["per_map"][f"arenas/{m}"]["value"] for m in shifts]
        assert vals == sorted(vals), f"{name}: a larger shift must be farther"
        assert len(v["floor"]["values"]) == 40
        assert v["family"]["all_arenas_above_floor"], name
        assert v["spearman"]["D"]["rho"] == pytest.approx(1.0)
        assert v["spearman"]["A0"]["rho"] == pytest.approx(-1.0)
    # the censored arena is tied at the top of the budget
    assert res["outcomes"]["11"]["cost"] == 8000
    md = (out / "VARIANTS.md").read_text()
    assert md.startswith("# ") and "| variant |" in md
    assert "**Reading.** All 10 variants place every one of the 3 unseen arenas above their own training floor" in md
    assert res["reading"] in md
