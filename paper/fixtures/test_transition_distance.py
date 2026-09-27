"""`distance_study.py transitions | coverage | transfer-gap | compare`: the transition-level distances.

Two candidates replace or accompany the frozen per-frame distance D (`transition_distance.py` docstring):
directed coverage of transition windows by the training memory (Astra) and the kNN transfer gap G (Opus). These
tests pin, on tiny synthetic per-tic corpora whose controls drive the latent dynamics:

  * the window features follow the recorder's row convention (row t carries the control leaving frame t) and
    sit on the frame clouds' own frames for the same seed; the cache is deterministic and parallel-safe;
  * the exact search keeps at most one neighbour per memory episode and never one from the query's own;
  * the state-only ablation equals coverage when dynamics, innovation and controls are constant;
  * shuffling controls raises coverage when controls carry information and leaves it unchanged otherwise;
  * G is zero for identical memories, near zero for one law, and positive when the target's dynamics differ;
  * compare writes one CSV and JSON with D, coverage, G, ablations, bootstrap SDs and rank agreement, and
    checks that the training maps sit at the floor of every distance.

numpy, pyarrow-free, CPU only, seconds:

    python -m pytest paper/fixtures/test_transition_distance.py -q
"""
import csv
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import distance_study as ds  # noqa: E402
import transition_distance as td  # noqa: E402

L = 8                       # context frames (32 on the real corpora)
LAGS = "1,2,4,7"            # (1, 4, 16, 31) on the real corpora; 7 reaches the oldest of 8 context frames
FPE = 20                    # windows per episode (250 on the real corpora): two per motion decile
T = 120
TRAIN_MAPS = (2, 3, 4, 5)
PATTERNS = np.random.default_rng(7).normal(size=(2, 4, 30, 40)).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# synthetic corpora: latents whose next change is set by the executed control
# ---------------------------------------------------------------------------------------------

def write_episode(d, ep, map_id=2, law=1.0, loc=0.0, informative=True, constant_controls=False, T=T):
    """One latent/sidecar pair. Decisions every 4 tics pick turn {none, left, right}, forward and attack. Row t's
    control moves frame t into t + 1 (record_arnold's row semantics): left adds `law` x pattern 0, right subtracts
    it, forward adds pattern 1. With `informative=False` the change follows an independent control sequence, so
    the stored controls say nothing about it. Noise of varying size spreads motion over its deciles."""
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(1000 + ep)
    n_dec = T // 4 + 1

    def controls(r):
        return (r.integers(0, 3, n_dec).repeat(4)[:T], r.integers(0, 2, n_dec).repeat(4)[:T],
                r.integers(0, 2, n_dec).repeat(4)[:T])
    turn, fwd, att = controls(rng)
    if constant_controls:
        turn, fwd, att = np.ones(T, int), np.ones(T, int), np.zeros(T, int)
    dturn, dfwd, _ = (turn, fwd, att) if informative else controls(np.random.default_rng(9000 + ep))
    sign = np.where(dturn == 1, 1.0, np.where(dturn == 2, -1.0, 0.0))
    drive = law * sign[:, None, None, None] * PATTERNS[0] + dfwd[:, None, None, None] * PATTERNS[1]
    noise = rng.normal(size=(T, 4, 30, 40)) * rng.uniform(0.05, 1.0, size=(T, 1, 1, 1))
    step = 0.03 * (drive + 0.5 * noise)
    lat = np.zeros((T, 4, 32, 40), np.float32)
    lat[0, :, :30] = loc
    lat[1:, :, :30] = loc + np.cumsum(step[:-1], axis=0)
    bits = np.zeros((T, 9), int)
    bits[:, 0], bits[:, 2], bits[:, 3], bits[:, 6] = fwd, turn == 1, turn == 2, att
    tics = np.arange(T, dtype=np.int64)
    np.save(os.path.join(d, f"ep_{ep:05d}_latents.npy"), lat.astype(np.float16))
    np.savez(os.path.join(d, f"ep_{ep:05d}_meta.npz"), action=np.zeros(T, np.int64),
             buttons=np.array(["".join(map(str, b)) for b in bits]), tic=tics, deaths=np.zeros(T, np.int64),
             map_id=np.full(T, map_id, np.int64), episode_id=np.full(T, ep, np.int64),
             is_decision=(tics % 4) == 0, chain_id=np.where(tics % 4 == 0, 0, -1))


def corpus(root, name, maps, per_map, first_id=0, **kw):
    """Episodes of `maps` in turn by id (`maps[e % len(maps)]`, the recorder's rule)."""
    d = os.path.join(root, name)
    for i in range(per_map * len(maps)):
        m = maps[i % len(maps)]
        write_episode(d, first_id + i, map_id=m, **{k: (v[m] if isinstance(v, dict) else v) for k, v in kw.items()})
    return d


def run(*argv):
    return ds.main([str(a) for a in argv])


def draw(out, name, latents, *extra):
    return run("transitions", "--out", out, "--set", name, "--latents", latents, "--context-frames", L,
               "--lags", LAGS, "--frames-per-episode", FPE, *extra)


def write_splits(latents, out_dir, set_name):
    import adapt_split
    return adapt_split.main(["--latents-dir", latents, "--map", "all", "--set", set_name, "--seed", "0",
                             "--adapt-episodes", "3", "--held-out-episodes", "2", "--ladder", "1,2,3",
                             "--step-curve-k", "2", "--windows-per-episode", "4", "--context-frames", str(L),
                             "--out-dir", out_dir])


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    """Training memory on maps 2 to 5 (law 1, one appearance per map), their validation episodes (same law) and
    two unseen arenas whose turns move the frame twice as far (law 2) under a shifted appearance, with adaptation
    splits for the scored sets."""
    root = str(tmp_path_factory.mktemp("transition_study"))
    locs = {2: 0.0, 3: 0.3, 4: -0.3, 5: 0.6}
    train = corpus(root, "arenas", TRAIN_MAPS, 4, loc=locs)
    val = corpus(root, "val", TRAIN_MAPS, 5, first_id=6000, loc=locs)
    new = corpus(root, "arenas13", (6, 7), 5, first_id=16, law=2.0, loc={6: 0.9, 7: -0.9})
    out = os.path.join(root, "out")
    splits = os.path.join(root, "splits")
    write_splits(val, splits, "val")
    write_splits(new, splits, "arenas13")
    draw(out, "train", train, "--episodes-per-map", 4, "--ids", "0:16")
    draw(out, "val", val)
    draw(out, "arenas13", new, "--splits", splits)
    return {"root": root, "out": out, "splits": splits, "train": train, "val": val, "new": new}


# ---------------------------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------------------------

def test_window_features_follow_the_row_convention():
    rng = np.random.default_rng(0)
    lat = rng.normal(size=(40, 4, 32, 40)).astype(np.float32)
    targets = np.array([10, 17, 39])
    state, dyn, innov = td.window_features(lat, targets, (1, 2, 4, 7))
    pool = ds.pool_latents
    assert state.shape == (3, 192) and dyn.shape == (3, 4, 192) and innov.shape == (3, 192)
    for i, r in enumerate(targets):
        np.testing.assert_allclose(state[i], pool(lat[[r - 1]])[0], atol=1e-6)            # newest context frame
        np.testing.assert_allclose(innov[i], pool(lat[[r]])[0] - pool(lat[[r - 1]])[0], atol=1e-5)
        for j, lag in enumerate((1, 2, 4, 7)):
            np.testing.assert_allclose(dyn[i, j], pool(lat[[r - 1]])[0] - pool(lat[[r - 1 - lag]])[0], atol=1e-5)
    with pytest.raises(ValueError):
        td.window_features(lat, np.array([5]), (1, 7))                                   # the lag leaves the episode


def test_control_features_are_the_newest_control_and_context_rates():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=(50, 19)).astype(np.uint8)
    targets = np.array([L, 21, 49])
    newest, freq, switch = td.control_features(bits, targets, L)
    for i, r in enumerate(targets):
        np.testing.assert_array_equal(newest[i], bits[r - 1])                   # applied from r - 1 into r
        np.testing.assert_allclose(freq[i], bits[r - L:r].mean(axis=0), atol=1e-6)
        np.testing.assert_allclose(switch[i], (bits[r - L + 1:r] != bits[r - L:r - 1]).mean(axis=0), atol=1e-6)
    assert newest.dtype == np.uint8 and freq.dtype == np.float32
    with pytest.raises(ValueError):
        td.control_features(bits, np.array([L - 1]), L)                       # no full control history


def test_control_bits_read_raw_and_normalised_strings():
    raw = np.array(["101000100", "0100000000000000000001", "0000000000000000001"])
    bits = td.control_bits(raw)
    assert bits.shape == (3, 19) and bits.dtype == np.uint8
    np.testing.assert_array_equal(bits[0, :9], [1, 0, 1, 0, 0, 0, 1, 0, 0])
    assert bits[1].tolist() == [0, 1] + [0] * 17                             # the bit past 19 was never executed
    assert bits[2, 18] == 1


def test_the_control_class_is_turn_by_move_by_attack():
    rows = []
    for turn in ((0, 0), (1, 0), (0, 1), (1, 1)):
        for move in ((0, 0), (1, 0), (0, 1), (1, 1)):
            for attack in (0, 1):
                b = np.zeros(19, np.uint8)
                b[td.TURN_LEFT], b[td.TURN_RIGHT] = turn
                b[td.MOVE_FORWARD], b[td.MOVE_BACKWARD] = move
                b[td.ATTACK], b[4] = attack, 1                                 # strafe never changes the class
                rows.append(b)
    cls = td.control_class(np.array(rows))
    assert set(cls.tolist()) == set(range(td.N_CLASSES))
    b = np.zeros((3, 19), np.uint8)
    b[0, td.TURN_LEFT] = 1
    b[1, [td.TURN_LEFT, td.TURN_RIGHT]] = 1                                   # both directions cancel
    b[2, [td.TURN_RIGHT, td.MOVE_BACKWARD, td.ATTACK]] = 1
    assert td.control_class(b).tolist() == [6, 0, 6 * 2 + 2 * 2 + 1]


# ---------------------------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------------------------

def test_the_cache_holds_every_feature_with_its_settings(study):
    d = os.path.join(study["out"], "transitions", "sd1", "arenas13")
    with open(os.path.join(d, "index.json")) as f:
        index = json.load(f)
    assert index["settings"] == {"space": "sd1", "context_frames": L, "lags": [1, 2, 4, 7], "frames_per_episode": FPE,
                                 "bins": 10, "seed": 0}
    assert len(index["episodes"]) == 10 and all(r["drawn"] == FPE for r in index["episodes"])
    with np.load(os.path.join(d, index["episodes"][0]["file"])) as z:
        assert z["state"].shape == (FPE, 192) and z["dyn"].shape == (FPE, 4, 192) and z["innov"].shape == (FPE, 192)
        assert z["newest"].shape == (FPE, 19) and z["freq"].shape == (FPE, 19) and z["switch"].shape == (FPE, 19)
        assert z["klass"].shape == (FPE,) and np.bincount(z["decile"], minlength=10).tolist() == [2] * 10
        meta = json.loads(str(z["meta"]))
    assert meta["settings"] == index["settings"] and meta["identity"]["set"] == "arenas13"
    assert "transition_distance_sha256" in meta["code"]


def test_the_windows_sit_on_the_frame_clouds_frames(study, tmp_path):
    run("clouds", "--space", "sd1", "--out", tmp_path, "--eval", f"val={study['val']}", "--context-frames", L,
        "--frames-per-episode", FPE)
    cloud = ds.load_cloud(tmp_path / "clouds" / "sd1" / "val.npz")["frames"]
    t = td.load_set(study["out"], "sd1", "val")["windows"]
    for ep in (6000, 6007):
        mine, theirs = t["episode"] == ep, cloud["episode"] == ep
        np.testing.assert_array_equal(np.sort(t["row"][mine]), np.sort(cloud["row"][theirs]))
        np.testing.assert_allclose(np.sort(t["motion"][mine]), np.sort(cloud["motion"][theirs]))


def test_the_memory_draw_is_the_frame_reference_draw(study, tmp_path):
    run("clouds", "--space", "sd1", "--out", tmp_path, "--reference", study["train"], "--reference-ids", "0:16",
        "--episodes-per-map", 4, "--context-frames", L, "--frames-per-episode", FPE)
    ref = ds.load_cloud(tmp_path / "clouds" / "sd1" / "train_draw1.npz")
    t = td.load_set(study["out"], "sd1", "train")
    assert sorted(t["episodes"]) == sorted(int(e) for e in ref["episodes"]["id"])


def test_the_cache_is_deterministic_parallel_safe_and_refuses_other_settings(study, tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    draw(a, "val", study["val"])
    draw(b, "val", study["val"], "--workers", 2)
    draw(a, "val", study["val"])                                             # a rerun reuses the cache
    x, y = td.load_set(a, "sd1", "val")["windows"], td.load_set(b, "sd1", "val")["windows"]
    ref = td.load_set(study["out"], "sd1", "val")["windows"]
    for k in td.WINDOW_ARRAYS:
        np.testing.assert_array_equal(x[k], y[k])
        np.testing.assert_array_equal(x[k], ref[k])
    with pytest.raises(SystemExit, match="other settings"):
        draw(a, "val", study["val"], "--seed", 1)
    draw(a, "val", study["val"], "--seed", 1, "--force")
    assert not np.array_equal(td.load_set(a, "sd1", "val")["windows"]["row"], x["row"])


def test_sets_drawn_differently_are_not_scored_together(study, tmp_path):
    out = str(tmp_path)
    draw(out, "train", study["train"], "--episodes-per-map", 2)
    run("transitions", "--out", out, "--set", "val", "--latents", study["val"], "--context-frames", L,
        "--lags", "1,2", "--frames-per-episode", FPE)
    with pytest.raises(SystemExit, match="drawn differently"):
        run("coverage", "--out", out, "--memory", "train", "--targets", "val", "--bootstrap", 0)


# ---------------------------------------------------------------------------------------------
# the exact search
# ---------------------------------------------------------------------------------------------

def brute_knn(q, qk, m, mk, k, distinct):
    out = []
    for i in range(len(q)):
        d = np.linalg.norm(m - q[i], axis=1)
        d[mk == qk[i]] = np.inf
        if distinct:
            best = {}
            for j in np.argsort(d, kind="stable"):
                best.setdefault(mk[j], d[j])
            d = np.array(sorted(best.values()))
        out.append(np.sort(d)[:k])
    return np.array(out)


def test_at_most_one_neighbour_per_memory_episode_and_never_the_query_own():
    m = np.array([[0.1, 0], [0.1, 0.01], [0.1, -0.01], [1.0, 0], [2.0, 0], [3.0, 0]])
    mk = np.array([7, 7, 7, 8, 9, 10])
    q = np.zeros((2, 2))
    d, nb = td.knn(q, np.array([1, 7]), m, mk, 3, distinct=True, neighbours=True)
    np.testing.assert_allclose(d[0], [0.1, 1.0, 2.0])                        # one window of episode 7 counts
    np.testing.assert_allclose(d[1], [1.0, 2.0, 3.0])                        # the query's own episode is out
    assert nb[0].tolist() == [0, 3, 4] and nb[1].tolist() == [3, 4, 5]
    d, _ = td.knn(q, np.array([1, 7]), m, mk, 3, distinct=False)
    np.testing.assert_allclose(d[0], [0.1, np.hypot(0.1, 0.01), np.hypot(0.1, 0.01)])
    np.testing.assert_allclose(d[1], [1.0, 2.0, 3.0])
    cov, _ = td.directed_coverage(q, np.array([1, 7]), m, mk, 3)
    np.testing.assert_allclose(cov, [(0.1 + 1 + 2) / 3, 2.0])
    with pytest.raises(ValueError, match="cannot give"):
        td.knn(q, np.array([1, 7]), m[:4], mk[:4], 2, distinct=True)          # episodes 7 and 8 only


@pytest.mark.parametrize("distinct", [True, False])
def test_the_search_equals_brute_force_whatever_the_block(distinct):
    rng = np.random.default_rng(3)
    m, mk = rng.normal(size=(300, 5)), rng.integers(0, 25, 300)
    q, qk = rng.normal(size=(41, 5)), rng.integers(0, 40, 41)
    want = brute_knn(q, qk, m, mk, 3, distinct)
    for block in (1, 7, 256):
        d, nb = td.knn(q, qk, m, mk, 3, distinct, block=block, neighbours=True)
        np.testing.assert_allclose(d, want, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(m[nb] - q[:, None], axis=2), want, atol=1e-9)
        if distinct:
            assert all(len(set(mk[row].tolist())) == 3 for row in nb)
        assert not np.any(mk[nb] == qk[:, None])


# ---------------------------------------------------------------------------------------------
# candidate 1: the metric and its ablations
# ---------------------------------------------------------------------------------------------

def window_table(n, rng, episodes, informative=True, law=1.0, constant=False, shift=0.0):
    """Array-level windows: the innovation follows the newest turn control (left +law, right -law) when
    `informative`, and an independent draw otherwise."""
    turn = rng.integers(0, 3, n)
    newest = np.zeros((n, 19), np.uint8)
    newest[:, td.TURN_LEFT], newest[:, td.TURN_RIGHT] = turn == 1, turn == 2
    drive_turn = turn if informative else rng.integers(0, 3, n)
    sign = np.where(drive_turn == 1, 1.0, np.where(drive_turn == 2, -1.0, 0.0))
    e = np.zeros(192)
    e[:12] = 1.0
    w = {"state": rng.normal(size=(n, 192)) + shift, "dyn": rng.normal(size=(n, 4, 192)),
         "innov": law * sign[:, None] * e + 0.2 * rng.normal(size=(n, 192)), "newest": newest,
         "freq": rng.uniform(size=(n, 19)).astype(np.float32), "switch": rng.uniform(size=(n, 19)).astype(np.float32)}
    if constant:
        w["dyn"] = np.ones((n, 4, 192))
        w["innov"] = np.full((n, 192), 0.5)
        w["newest"] = np.ones((n, 19), np.uint8)
        w["freq"] = np.full((n, 19), 0.25, np.float32)
        w["switch"] = np.zeros((n, 19), np.float32)
    w["klass"] = td.control_class(w["newest"])
    w["episode"] = w["key"] = rng.choice(episodes, n)
    w["weight"] = np.ones(n)
    return w


def coverage_of(mem, tgt, which=td.BLOCKS, target_blocks=None):
    space = td.CoverageSpace(td.raw_blocks(mem))
    tb = target_blocks or td.raw_blocks(tgt)
    v, _ = td.directed_coverage(space.transform(tb, which), tgt["key"],
                                space.transform(td.raw_blocks(mem), which), mem["key"], 3)
    return float(v.mean()), space


def test_the_blocks_carry_equal_weight_and_unit_rms_pairwise_distance():
    rng = np.random.default_rng(4)
    mem = window_table(400, rng, np.arange(20))
    space = td.CoverageSpace(td.raw_blocks(mem))
    x = space.transform(td.raw_blocks(mem))
    assert x.shape == (400, 32 + 64 + 32 + 57)
    rms = lambda z: np.sqrt(2 * np.mean(np.sum((z - z.mean(0)) ** 2, axis=1)))  # noqa: E731
    assert rms(x) == pytest.approx(1.0)
    for b in td.BLOCKS:
        assert rms(space.transform(td.raw_blocks(mem), (b,))) == pytest.approx(np.sqrt(0.25))
    again = td.CoverageSpace(td.raw_blocks(mem)).transform(td.raw_blocks(mem))
    np.testing.assert_array_equal(x, again)                                  # the projections are seeded


def test_state_only_equals_coverage_on_constant_dynamics():
    rng = np.random.default_rng(5)
    mem = window_table(300, rng, np.arange(30), constant=True)
    tgt = window_table(60, rng, np.arange(100, 110), constant=True, shift=0.5)
    full, space = coverage_of(mem, tgt)
    state, _ = coverage_of(mem, tgt, ("state",))
    assert set(space.degenerate) == {"dynamics", "innovation", "controls"}
    assert state == pytest.approx(full, rel=1e-12)
    rng = np.random.default_rng(5)
    mem, tgt = window_table(300, rng, np.arange(30)), window_table(60, rng, np.arange(100, 110), shift=0.5)
    assert coverage_of(mem, tgt, ("state",))[0] < coverage_of(mem, tgt)[0]   # a share once the rest varies


def shuffled_controls(tgt, seed=0):
    tb = td.raw_blocks(tgt)
    perm = np.random.default_rng(seed).permutation(len(tgt["key"]))
    return dict(tb, controls=tb["controls"][perm])


def test_shuffled_controls_rise_only_when_controls_carry_information():
    rises = {}
    for informative in (True, False):
        rng = np.random.default_rng(6)
        mem = window_table(1500, rng, np.arange(40), informative=informative)
        tgt = window_table(300, rng, np.arange(100, 110), informative=informative)
        base, _ = coverage_of(mem, tgt)
        rises[informative] = coverage_of(mem, tgt, target_blocks=shuffled_controls(tgt))[0] - base
    assert rises[True] > 0.01                                                # measured 0.017 on this draw
    assert abs(rises[False]) < rises[True] / 5                               # measured 0.0008
    rng = np.random.default_rng(6)
    mem = window_table(300, rng, np.arange(30))
    tgt = window_table(60, rng, np.arange(100, 110))
    for w in (mem, tgt):
        w["newest"][:], w["freq"][:], w["switch"][:] = 1, 0.5, 0.0
    assert coverage_of(mem, tgt, target_blocks=shuffled_controls(tgt))[0] == coverage_of(mem, tgt)[0]


def test_coverage_reports_every_ablation_the_floor_and_the_checks(study):
    run("coverage", "--out", study["out"], "--memory", "train", "--targets", "val", "arenas13", "--splits",
        study["splits"], "--bootstrap", 200)
    with open(os.path.join(study["out"], "coverage_sd1.json")) as f:
        cov = json.load(f)
    pts = {p["key"]: p for p in cov["points"]}
    assert sorted(pts) == ["arenas13/6", "arenas13/7", "val/2", "val/3", "val/4", "val/5"]
    for p in pts.values():
        assert p["state_only"] <= p["coverage"] and p["without_state"] <= p["coverage"]
        assert p["bootstrap"]["coverage"]["sd"] > 0 and p["n_episodes"] == 5
        assert p["windows"] == 5 * FPE
        assert sorted(p["by_decile"], key=int) == [str(x) for x in range(10)]
        assert sum(v["n"] for v in p["by_decile"].values()) == sum(v["n"] for v in p["by_class"].values()) == 5 * FPE
    assert cov["checks"]["training_maps_at_floor"]["pass"] is True
    assert cov["checks"]["temporal_sensitivity"]["pass"] is True
    assert cov["checks"]["control_sensitivity"]["pass"] is True
    assert set(cov["checks"]) >= {"state_share", "memory_halves", "validation_inside_floor"}
    assert cov["config"]["k"] == 3 and cov["config"]["distinct_episodes"] is True
    assert cov["memory"]["episodes"] == 16 and cov["memory"]["maps"] == [2, 3, 4, 5]
    assert cov["config"]["window_settings"]["lags"] == [1, 2, 4, 7]
    with np.load(os.path.join(study["out"], "coverage_sd1_windows.npz")) as z:
        assert z["coverage"].shape == (6 * 5 * FPE,) and z["neighbour_episode"].shape == (6 * 5 * FPE, 3)
        assert all(len(set(r.tolist())) == 3 for r in z["neighbour_key"])
    with open(os.path.join(study["out"], "coverage_sd1.csv")) as f:
        assert len(list(csv.DictReader(f))) == 6


def test_the_bootstrap_redraws_whole_episodes():
    v = np.array([1.0, 1.0, 3.0, 3.0])
    ep = np.array([1, 1, 2, 2])
    bt = td.episode_bootstrap([v, v], np.ones(4), ep, 500, np.random.default_rng(0))
    assert set(np.round(bt[:, 0], 9).tolist()) <= {1.0, 2.0, 3.0}            # means of redrawn episodes only
    np.testing.assert_array_equal(bt[:, 0], bt[:, 1])                        # one redraw shared by every column
    assert td.episode_bootstrap([v], np.ones(4), np.ones(4), 50, np.random.default_rng(0)).std() == 0


# ---------------------------------------------------------------------------------------------
# candidate 2: the transfer gap
# ---------------------------------------------------------------------------------------------

def test_the_predictor_searches_within_the_control_class_and_falls_back_when_rare():
    rng = np.random.default_rng(8)
    n = 200
    mem_cls = np.where(np.arange(n) < 100, 1, 2).astype(np.int8)
    innov = np.where(mem_cls[:, None] == 1, 1.0, -1.0) * np.ones((n, 3))
    x = rng.normal(size=(n, 4))
    q = rng.normal(size=(5, 4))
    pred, fb = td.knn_predict(q, np.full(5, 1, np.int8), np.arange(5) + 1000, x, mem_cls, np.arange(n), innov,
                              k=4, min_class=50)
    np.testing.assert_allclose(pred, 1.0)                                    # class 1 memory only
    assert not fb.any()
    pred, fb = td.knn_predict(q, np.full(5, 1, np.int8), np.arange(5) + 1000, x, mem_cls, np.arange(n), innov,
                              k=4, min_class=500)
    assert fb.all()                                                          # 100 < 500: every class searched


def gap_tables(rng, law, n, episodes):
    w = window_table(n, rng, episodes, law=law)
    w["dyn"][:, 0] = rng.normal(size=(n, 192))
    return w


def test_the_transfer_gap_is_zero_for_one_law_and_positive_when_dynamics_differ():
    rng = np.random.default_rng(9)
    same = gap_tables(rng, 1.0, 800, np.arange(10))
    lt, lo, ok, _ = td.transfer_gap(gap_tables(rng, 1.0, 300, np.arange(50, 58)), same, same, k=16, min_class=50)
    assert np.all(lt[ok] == lo[ok])                                          # identical memories: G = 0 exactly
    train, own = gap_tables(rng, 1.0, 800, np.arange(10)), gap_tables(rng, 1.0, 800, np.arange(20, 30))
    target = gap_tables(rng, 1.0, 400, np.arange(50, 58))
    lt, lo, ok, _ = td.transfer_gap(target, own, train, k=16, min_class=50)
    g_same = lt[ok].mean() - lo[ok].mean()
    own_new, target_new = gap_tables(rng, -1.0, 800, np.arange(20, 30)), gap_tables(rng, -1.0, 400, np.arange(50, 58))
    lt, lo, ok, _ = td.transfer_gap(target_new, own_new, train, k=16, min_class=50)
    g_new = lt[ok].mean() - lo[ok].mean()
    assert abs(g_same) < 0.05 and g_new > 0.3
    assert lo[ok].mean() < 0                                                 # the own-map predictor beats copy-last


def test_a_static_window_has_no_copy_last_error_and_is_excluded():
    rng = np.random.default_rng(10)
    train = gap_tables(rng, 1.0, 200, np.arange(10))
    target = gap_tables(rng, 1.0, 20, np.arange(50, 52))
    target["innov"][:3] = 0.0
    _, _, ok, info = td.transfer_gap(target, train, train, k=16, min_class=50)
    assert ok.tolist() == [False] * 3 + [True] * 17 and info["excluded_static"] == 3


def test_transfer_gap_scores_held_out_episodes_against_the_adaptation_episodes(study):
    run("transfer-gap", "--out", study["out"], "--memory", "train", "--targets", "val", "arenas13", "--splits",
        study["splits"], "--min-class", 5, "--bootstrap", 200)
    with open(os.path.join(study["out"], "transfer_gap_sd1.json")) as f:
        gap = json.load(f)
    pts = {p["key"]: p for p in gap["points"]}
    assert sorted(pts) == ["arenas13/6", "arenas13/7", "val/2", "val/3", "val/4", "val/5"]
    for p in pts.values():
        with open(p["split"]) as f:
            split = json.load(f)
        assert p["held_out"] == split["held_out"] and p["own_episodes"] == sorted(split["adapt"])
        assert p["windows"] == 2 * FPE and p["own_memory_windows"] == 3 * FPE == p["train_memory_windows"]
        assert p["G"] == pytest.approx(p["L_train"] - p["L_own"])
        assert p["bootstrap"]["G"]["sd"] >= 0
    assert gap["checks"]["training_maps_at_floor"]["pass"] is True and gap["sets_without_splits"] == []
    assert min(pts[k]["G"] for k in ("arenas13/6", "arenas13/7")) > 0.2
    with open(os.path.join(study["out"], "transfer_gap_sd1.csv")) as f:
        assert len(list(csv.DictReader(f))) == 6
    with pytest.raises(SystemExit, match="--splits"):
        run("transfer-gap", "--out", study["out"], "--targets", "arenas13")


# ---------------------------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------------------------

def frozen_like(path, D):
    json.dump({"primary_arm": "motion", "floor": {"motion": {"max": 0.1}},
               "maps": [{"map": m, "role": "primary", "D": d, "bootstrap": {"sd": 0.01}} for m, d in D.items()]
               + [{"map": 2, "role": "replication", "D": 9.9}]}, open(path, "w"))
    return path


def test_compare_writes_one_csv_and_json_with_the_floor_check(study, tmp_path):
    run("coverage", "--out", study["out"], "--memory", "train", "--targets", "val", "arenas13", "--splits",
        study["splits"], "--bootstrap", 50)
    run("transfer-gap", "--out", study["out"], "--memory", "train", "--targets", "val", "arenas13", "--splits",
        study["splits"], "--min-class", 5, "--bootstrap", 50)
    D = frozen_like(tmp_path / "d.json", {2: 0.06, 3: 0.03, 4: 0.07, 5: 0.03, 6: 0.19, 7: 0.28})
    run("compare", "--out", study["out"], "--frame-distances", D, "--name", "synthetic")
    with open(os.path.join(study["out"], "compare_synthetic_sd1.json")) as f:
        cmp_ = json.load(f)
    with open(os.path.join(study["out"], "compare_synthetic_sd1.csv")) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == td.COMPARE_COLUMNS
        rows = list(reader)
    assert len(rows) == 6 and {r["key"] for r in rows} == {m["key"] for m in cmp_["maps"]}
    assert set(cmp_) >= {"inputs", "spearman", "floor", "floor_all_pass", "maps", "code", "outputs"}
    assert set(cmp_["spearman"]) == {"all", "unseen"}
    assert set(cmp_["spearman"]["all"]) == {f"{x}~{y}" for x, y in td.PAIRS}
    by = {m["key"]: m for m in cmp_["maps"]}
    assert by["val/2"]["D"] == 0.06 and by["val/2"]["training"] and not by["arenas13/6"]["training"]
    assert all(m["coverage_sd"] is not None and m["G_sd"] is not None for m in cmp_["maps"])
    assert cmp_["floor"]["D"]["pass"] and cmp_["floor"]["coverage"]["pass"] and cmp_["floor"]["G"]["pass"]
    assert cmp_["floor"]["D"]["training_inside_frozen_floor"] is True and cmp_["floor_all_pass"] is True
    assert cmp_["inputs"]["frame_distances"]["sha256"] and cmp_["inputs"]["transfer_gap"]["config"]["k"] == 16

    D = frozen_like(tmp_path / "e.json", {2: 0.25, 3: 0.03, 4: 0.07, 5: 0.03, 6: 0.19, 7: 0.28})
    run("compare", "--out", study["out"], "--frame-distances", D, "--name", "broken")
    with open(os.path.join(study["out"], "compare_broken_sd1.json")) as f:
        broken = json.load(f)
    assert broken["floor"]["D"]["pass"] is False and broken["floor_all_pass"] is False


def test_the_frozen_frame_distance_gives_every_arena_and_the_training_maps_from_validation():
    path = os.path.join(REPO, "results", "distance_study", "distances_sd1.json")
    if not os.path.isfile(path):
        pytest.skip("the frozen distances are not in this checkout")
    D, floor, arm = td.frame_distances(path)
    assert set(range(1, 18)) <= set(D)
    assert D[2][0] == pytest.approx(0.0649, abs=1e-4)                        # val/2, not the seen/2 replication
    assert max(D[m][0] for m in TRAIN_MAPS) < min(D[m][0] for m in (1, *range(6, 18)))
    assert arm == "motion" and floor["motion"]["max"] > 0


def test_distance_study_registers_the_transition_subcommands():
    p = ds.build_parser()
    minimal = {"clouds": ["--space", "sd1"], "splits": [], "distances": ["--space", "sd1"],
               "transitions": ["--set", "x", "--latents", "y"], "coverage": [], "transfer-gap": [], "compare": []}
    for cmd, extra in minimal.items():
        assert p.parse_args([cmd, *extra]).cmd == cmd
    a = p.parse_args(["coverage"])
    assert (a.k, a.memory, a.bootstrap, a.weighting, a.projection_seed) == (3, ["train"], 1000, "eligible", 0)
    assert (a.state_dim, a.dynamics_dim, a.innovation_dim, a.controls) == (32, 64, 32, "newest,freq,switch")
    g = p.parse_args(["transfer-gap"])
    assert (g.k, g.min_class, g.gap_memory_size, g.own_episodes_k) == (16, 500, 0, 0)
    t = p.parse_args(["transitions", "--set", "x", "--latents", "y"])
    assert (t.lags, t.frames_per_episode, t.bins, t.context_frames) == ("1,4,16,31", 250, 10, 32)
