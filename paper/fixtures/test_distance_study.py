"""`distance_study.py`: the weighted sliced-Wasserstein distance between clouds of frames.

The distance study (`.claude/analyses/distance-study-design-2026-09-24.md`) places every evaluation
map on an axis of distance from the training footage. These tests pin the metric on synthetic clouds
whose answers are known by construction: a cloud against itself, a translated copy (exact, because a
1-D translation moves every quantile by the same amount), independent draws moved further apart,
weights that must behave like repeated points, a seed that must reproduce the value, the exact 1-D
weighted Wasserstein (scipy) and POT's sliced Wasserstein where either is installed, the motion
weights, the latent pooling, and the reason the primary distance is taken to the nearest training
map rather than to the pooled corpus. numpy only; nothing here needs a GPU, torch or real data.

    python -m pytest paper/fixtures/test_distance_study.py -q
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from distance_study import (QUIET_SHARE, latent_motion, motion_weights,  # noqa: E402
                            nearest_reference_distance, pool_latents, pooled_reference,
                            random_directions, sliced_wasserstein)


def cloud(n, d, seed, loc=0.0, scale=1.0):
    return np.random.default_rng(seed).normal(loc, scale, size=(n, d))


# ---------------------------------------------------------------------------------------------
# the four properties the design memo asks for
# ---------------------------------------------------------------------------------------------

def test_identical_clouds_are_at_distance_zero():
    x = cloud(500, 16, 0)
    assert sliced_wasserstein(x, x, seed=0) == 0.0
    # the order of the points is not part of a distribution
    perm = np.random.default_rng(1).permutation(len(x))
    assert sliced_wasserstein(x, x[perm], seed=0) == 0.0
    # nor is the scale of the weights, and a weighted cloud is at zero from itself
    w = np.random.default_rng(2).uniform(0.1, 3.0, size=len(x))
    assert sliced_wasserstein(x, x[perm], x_weights=w, y_weights=w[perm], seed=0) == 0.0
    # w and 7w normalise to cumulative weights that differ in the last bit; the square root of SW_2
    # turns that 1e-18 residue into about 1e-9, far below any distance the study reads
    assert sliced_wasserstein(x, x, x_weights=w, y_weights=7.0 * w, seed=0) == pytest.approx(0.0, abs=1e-8)


def test_a_translated_copy_is_exactly_as_far_as_the_projected_shift():
    """Moving every point by v moves every 1-D quantile by theta.v, so on each direction the
    Wasserstein distance is |theta.v| exactly and SW_2 = sqrt(mean (theta.v)^2): linear in the shift."""
    x = cloud(400, 8, 3)
    v = np.random.default_rng(4).normal(size=8)
    dirs = random_directions(8, 256, seed=0)
    unit = np.sqrt(np.mean((dirs.T @ v) ** 2))
    got = [sliced_wasserstein(x, x + t * v, directions=dirs) for t in (0.25, 0.5, 1.0, 2.0, 4.0)]
    for t, d in zip((0.25, 0.5, 1.0, 2.0, 4.0), got):
        assert d == pytest.approx(t * unit, rel=1e-9)
    assert np.all(np.diff(got) > 0)


def test_the_distance_between_independent_draws_grows_monotonically_with_their_separation():
    """Two independent samples of one law sit at a small finite-sample floor; moving one of them
    away raises the distance at every step and, once the shift dominates, towards the translation
    value sqrt(mean (theta.e)^2) * shift."""
    x, y = cloud(3000, 8, 5), cloud(3000, 8, 6)
    e = np.zeros(8)
    e[0] = 1.0
    dirs = random_directions(8, 512, seed=0)
    shifts = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    got = np.array([sliced_wasserstein(x, y + s * e, directions=dirs) for s in shifts])
    assert np.all(np.diff(got) > 0), got
    assert got[0] < 0.1                                  # the floor of two 3,000-point samples
    unit = np.sqrt(np.mean(dirs[0] ** 2))
    assert got[-1] == pytest.approx(8.0 * unit, rel=0.05)


def test_weights_change_the_distance_and_act_exactly_like_repeated_points():
    rng = np.random.default_rng(7)
    near = rng.normal(0.0, 0.3, size=(300, 4))
    far = rng.normal(3.0, 0.3, size=(300, 4))
    x = np.concatenate([near, far])
    ref = cloud(600, 4, 8, loc=0.0, scale=0.3)           # the reference holds only the near cluster
    # putting more of the cloud's mass on the cluster the reference shares brings it closer
    got = []
    for alpha in (0.5, 0.7, 0.9, 0.99):
        w = np.concatenate([np.full(300, alpha), np.full(300, 1.0 - alpha)])
        got.append(sliced_wasserstein(x, ref, x_weights=w, seed=0))
    assert np.all(np.diff(got) < 0), got
    assert got[0] > 1.5 * got[-1]
    # integer weights are the same distribution as repeating each point that many times
    small = rng.normal(size=(7, 5))
    counts = np.array([1, 2, 3, 1, 1, 4, 2])
    y = rng.normal(0.5, 1.0, size=(9, 5))
    weighted = sliced_wasserstein(small, y, x_weights=counts, seed=3)
    repeated = sliced_wasserstein(np.repeat(small, counts, axis=0), y, seed=3)
    assert weighted == pytest.approx(repeated, rel=1e-12)
    # only relative weights matter
    assert sliced_wasserstein(small, y, x_weights=5.0 * counts, seed=3) == pytest.approx(weighted, rel=1e-12)


def test_the_value_is_reproducible_from_its_seed():
    x, y = cloud(800, 32, 9), cloud(800, 32, 10, loc=0.3)
    a = sliced_wasserstein(x, y, seed=11, n_projections=2000)
    b = sliced_wasserstein(x, y, seed=11, n_projections=2000)
    c = sliced_wasserstein(x, y, seed=12, n_projections=2000)
    assert a == b                                        # bit for bit
    assert a != c                                        # a different set of directions ...
    assert c == pytest.approx(a, rel=0.05)               # ... estimating the same quantity
    np.testing.assert_array_equal(random_directions(32, 50, seed=11), random_directions(32, 50, seed=11))
    dirs = random_directions(32, 50, seed=11)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=0), 1.0, rtol=1e-12)
    # passing the directions explicitly gives the same number as passing the seed that made them
    assert sliced_wasserstein(x, y, directions=random_directions(32, 2000, seed=11)) == a


def test_a_chunked_computation_equals_the_single_pass_value():
    """Projections are processed in chunks to bound memory; the chunk size must not change the answer."""
    x, y = cloud(700, 12, 13), cloud(500, 12, 14, loc=0.2)
    w = np.random.default_rng(15).uniform(size=700)
    one = sliced_wasserstein(x, y, x_weights=w, seed=0, n_projections=300, max_elements=10 ** 9)
    many = sliced_wasserstein(x, y, x_weights=w, seed=0, n_projections=300, max_elements=5000)
    assert many == pytest.approx(one, rel=1e-12)


# ---------------------------------------------------------------------------------------------
# independent references for the 1-D transport
# ---------------------------------------------------------------------------------------------

def test_in_one_dimension_it_is_the_exact_weighted_wasserstein_distance():
    """In 1-D every direction is +1 or -1, so the sliced distance is the exact one; scipy computes W_1
    of weighted samples by an independent method (CDF differences)."""
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(16)
    x, y = rng.normal(size=300), rng.normal(0.4, 1.3, size=200)
    wx, wy = rng.uniform(0.1, 2.0, size=300), rng.uniform(0.1, 2.0, size=200)
    ours = sliced_wasserstein(x[:, None], y[:, None], x_weights=wx, y_weights=wy, p=1, seed=0,
                              n_projections=8)
    assert ours == pytest.approx(stats.wasserstein_distance(x, y, wx, wy), rel=1e-9)


def test_in_one_dimension_w2_of_equal_uniform_samples_is_the_sorted_difference():
    rng = np.random.default_rng(17)
    x, y = rng.normal(size=400), rng.exponential(size=400)
    exact = np.sqrt(np.mean((np.sort(x) - np.sort(y)) ** 2))
    assert sliced_wasserstein(x[:, None], y[:, None], p=2, seed=0, n_projections=4) == pytest.approx(exact, rel=1e-12)


def test_agrees_with_pot_on_the_same_directions_when_pot_is_installed():
    ot = pytest.importorskip("ot")
    rng = np.random.default_rng(18)
    x, y = rng.normal(size=(250, 6)), rng.normal(0.5, 1.0, size=(180, 6))
    wx, wy = rng.uniform(size=250), rng.uniform(size=180)
    dirs = random_directions(6, 100, seed=0)
    ours = sliced_wasserstein(x, y, x_weights=wx, y_weights=wy, directions=dirs)
    theirs = ot.sliced_wasserstein_distance(x, y, a=wx / wx.sum(), b=wy / wy.sum(), projections=dirs, p=2)
    assert ours == pytest.approx(float(theirs), rel=1e-8)


def test_bad_inputs_are_refused():
    x = cloud(10, 3, 19)
    with pytest.raises(ValueError):
        sliced_wasserstein(x, cloud(10, 4, 20))                      # different feature spaces
    with pytest.raises(ValueError):
        sliced_wasserstein(x, x, x_weights=-np.ones(10))             # negative mass
    with pytest.raises(ValueError):
        sliced_wasserstein(x, x, x_weights=np.zeros(10))             # no mass
    with pytest.raises(ValueError):
        sliced_wasserstein(x, x, x_weights=np.ones(9))               # one weight per point
    with pytest.raises(ValueError):
        sliced_wasserstein(x, x, p=0.5)                              # not a metric below p = 1
    with pytest.raises(ValueError):
        sliced_wasserstein(x, x, directions=random_directions(4, 10, seed=0))


# ---------------------------------------------------------------------------------------------
# motion weights: eventful frames carry more weight, the quietest half keeps a fixed share
# ---------------------------------------------------------------------------------------------

def test_the_quietest_half_keeps_the_requested_share_of_the_weight():
    motion = np.random.default_rng(21).exponential(size=1001) ** 2      # heavy-tailed, like real motion
    w = motion_weights(motion)
    assert w.sum() == pytest.approx(1.0, rel=1e-12)
    quiet = np.argsort(motion, kind="stable")[: len(motion) // 2]
    assert w[quiet].sum() == pytest.approx(QUIET_SHARE, rel=1e-9)
    assert QUIET_SHARE == 0.25
    # more motion never means less weight
    order = np.argsort(motion, kind="stable")
    assert np.all(np.diff(w[order]) >= -1e-15)
    # a share of one half is uniform weighting, the sensitivity arm
    np.testing.assert_allclose(motion_weights(motion, quiet_share=0.5), np.full(1001, 1 / 1001), rtol=1e-9)


def test_a_cloud_whose_quiet_half_already_holds_the_share_is_weighted_by_motion_alone():
    """The floor only ever lifts the quiet half; where it already holds the share, weights are motion."""
    motion = 1.0 + 0.01 * np.random.default_rng(22).uniform(size=200)       # nearly constant motion
    np.testing.assert_allclose(motion_weights(motion), motion / motion.sum(), rtol=1e-12)


def test_motion_weights_refuse_what_they_cannot_weight():
    with pytest.raises(ValueError):
        motion_weights(np.array([1.0, -0.1, 2.0]))
    with pytest.raises(ValueError):
        motion_weights(np.array([1.0, np.nan, 2.0]))
    with pytest.raises(ValueError):
        motion_weights(np.array([1.0, 2.0]), quiet_share=0.6)
    with pytest.raises(ValueError):
        motion_weights(np.zeros(4))


def test_latent_motion_is_the_frame_to_frame_change_within_one_life():
    v = np.random.default_rng(23).normal(size=(4, 32, 40))
    segment = np.stack([t * v for t in range(6)])                         # constant velocity
    m = latent_motion(segment)
    assert m.shape == (6,)
    # the first frame has no predecessor in its life and takes the second frame's motion
    np.testing.assert_allclose(m, np.full(6, np.linalg.norm(v)), rtol=1e-12)
    with pytest.raises(ValueError):
        latent_motion(segment[:1])


# ---------------------------------------------------------------------------------------------
# features: pooled VAE latents
# ---------------------------------------------------------------------------------------------

def test_pooled_latents_drop_the_padding_rows_and_average_blocks():
    lat = np.zeros((2, 4, 32, 40), dtype=np.float16)
    lat[:, :, 30:] = 100.0                                                 # the two padding rows
    feats = pool_latents(lat)
    assert feats.shape == (2, 4 * 6 * 8) and feats.dtype == np.float32
    np.testing.assert_array_equal(feats, 0.0)
    lat[:, :, :30] = 3.0
    np.testing.assert_allclose(pool_latents(lat), 3.0)
    one = np.zeros((1, 4, 32, 40))
    one[0, 0, 0, 0] = 25.0                                                 # one hot cell of the first 5x5 block
    feats = pool_latents(one)
    assert feats[0, 0] == pytest.approx(1.0) and np.count_nonzero(feats) == 1
    with pytest.raises(ValueError):
        pool_latents(np.zeros((1, 4, 28, 40)))                             # fewer rows than the visible frame
    with pytest.raises(ValueError):
        pool_latents(np.zeros((1, 4, 32, 40)), block=7)                    # 7 does not tile 30 x 40
    with pytest.raises(ValueError):
        pool_latents(np.zeros((4, 32, 40)))                                # one frame without its batch axis


# ---------------------------------------------------------------------------------------------
# why the primary distance is to the nearest training map, not to the pooled corpus
# ---------------------------------------------------------------------------------------------

def test_the_nearest_training_map_does_not_rank_a_seen_map_beyond_an_unseen_one():
    """Four training maps at 0, 2, 4, 6 along one axis. A held-out draw of the first map is SEEN; a
    map centred at 3 is UNSEEN. Wasserstein to the pooled corpus charges the seen map for not also
    looking like the other three maps (W_2^2 = (0 + 4 + 16 + 36) / 4 against (9 + 1 + 1 + 9) / 4 in
    1-D), so it ranks the seen map as the farther one. Distance to the nearest training map does not."""
    axis = np.zeros(4)
    axis[0] = 1.0

    def blob(centre, seed, n=1500):
        return cloud(n, 4, seed, scale=0.1) + centre * axis

    refs = {m: blob(c, 30 + m) for m, c in ((2, 0.0), (3, 2.0), (4, 4.0), (5, 6.0))}
    seen, unseen = blob(0.0, 40), blob(3.0, 41)
    pooled_x, pooled_w = pooled_reference(refs)
    assert pooled_w.sum() == pytest.approx(1.0) and len(pooled_x) == 6000
    d_pool = {k: sliced_wasserstein(c, pooled_x, y_weights=pooled_w, seed=0) for k, c in
              (("seen", seen), ("unseen", unseen))}
    assert d_pool["seen"] > d_pool["unseen"]                             # the artefact
    near_seen = nearest_reference_distance(seen, refs, seed=0)
    near_unseen = nearest_reference_distance(unseen, refs, seed=0)
    assert near_seen["nearest"] == 2
    assert near_unseen["nearest"] in (3, 4)
    assert near_seen["distance"] < 0.1 < near_unseen["distance"]
    assert near_seen["distance"] == min(near_seen["per_reference"].values())


def test_nearest_reference_takes_weighted_clouds_on_both_sides():
    refs = {"a": (cloud(400, 3, 50), np.ones(400)), "b": cloud(400, 3, 51, loc=2.0)}
    x = cloud(300, 3, 52)
    w = np.random.default_rng(53).uniform(size=300)
    got = nearest_reference_distance(x, refs, x_weights=w, seed=4)
    assert got["per_reference"]["a"] == sliced_wasserstein(x, refs["a"][0], x_weights=w, seed=4)
    assert got["per_reference"]["b"] == sliced_wasserstein(x, refs["b"], x_weights=w, seed=4)
    assert got["nearest"] == "a"
