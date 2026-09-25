"""`distance_study.py clouds | splits | distances`: the CPU pipeline of the map-distance study.

The metric core (`test_distance_study.py`) is pinned on synthetic clouds. These tests pin what feeds it
and what it writes, on synthetic per-tic corpora in the layout `encode_parquet.py --every-tic` writes:

  * `clouds` cuts every episode into lives exactly as the scored windows are cut (`tic_window_starts`),
    takes each frame's motion within its life, draws the same number of frames from every motion
    decile, draws the reference episodes seeded and map-balanced with a disjoint second draw, and can
    carry one draw into another latent space or into raw pixels by tic;
  * `splits` writes one `make_dense_eval_splits.py`-format file per map that `eval_tf.py` reads;
  * `distances` reproduces the committed sliced Wasserstein through its sorted-reference fast path,
    averages ten 4-episode subsets as five disjoint pairs, finds the best mixture of training maps,
    computes the train-versus-train floor and validation checks (i) to (v), switches to the uniform
    arm when distance tracks the effective sample size, and writes the JSON, the per-episode CSV and
    the bootstrap draws.

numpy, pyarrow and PIL only; nothing touches a GPU or real data.

    python -m pytest paper/fixtures/test_distance_study_pipeline.py -q
"""
import csv
import io
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import distance_study as ds  # noqa: E402
from distance_study import (episode_motion, eligible_rows, kish_neff, latent_motion,  # noqa: E402
                            load_cloud, pool_latents, save_cloud, sliced_wasserstein, spearman,
                            stratified_draw)

L = 8                   # context frames in the synthetic corpora (32 on the real ones)
FPE = 20                # frames per episode (250 on the real ones): two per motion decile
TRAIN_MAPS = (2, 3, 4, 5)


# ---------------------------------------------------------------------------------------------
# synthetic corpora
# ---------------------------------------------------------------------------------------------

def write_episode(d, ep, T=100, map_id=2, channels=4, deaths=None, tics=None, loc=0.0):
    """One latent/sidecar pair. The latents are a random walk whose step size varies by row, so
    motion spans a range and its deciles are distinct; the two padding rows are zero."""
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(1000 + ep)
    step = rng.normal(size=(T, channels, 30, 40)) * rng.uniform(0.1, 3.0, size=(T, 1, 1, 1))
    lat = np.zeros((T, channels, 32, 40), np.float32)
    lat[:, :, :30] = loc + 0.02 * np.cumsum(step, axis=0)
    lat = lat.astype(np.float16)
    tics = np.arange(T, dtype=np.int64) if tics is None else np.asarray(tics, dtype=np.int64)
    deaths = np.zeros(T, np.int64) if deaths is None else np.asarray(deaths, dtype=np.int64)
    np.save(os.path.join(d, f"ep_{ep:05d}_latents.npy"), lat)
    np.savez(os.path.join(d, f"ep_{ep:05d}_meta.npz"), action=np.zeros(T, np.int64),
             buttons=np.array(["100000000"] * T), tic=tics, deaths=deaths,
             map_id=np.full(T, map_id, np.int64), episode_id=np.full(T, ep, np.int64),
             is_decision=(tics % 4) == 0, chain_id=np.where(tics % 4 == 0, 0, -1))
    return lat


def training_corpus(root, per_map=6, channels=4, name="arenas"):
    """Maps 2 to 5 assigned by episode id the way the recorder does (`maps[e % 4]`)."""
    d = os.path.join(root, name)
    for e in range(per_map * 4):
        write_episode(d, e, map_id=TRAIN_MAPS[e % 4], channels=channels)
    return d


def run(*argv):
    return ds.main([str(a) for a in argv])


def small(*extra):
    return ["--context-frames", L, "--frames-per-episode", FPE, *extra]


# ---------------------------------------------------------------------------------------------
# clouds: lives, motion, the stratified draw
# ---------------------------------------------------------------------------------------------

def test_eligible_frames_are_the_targets_of_the_windows_eval_tf_scores(tmp_path):
    from doom_data import tic_window_starts
    tics = np.concatenate([np.arange(0, 40), np.arange(41, 100)])          # tic 40 never rendered
    deaths = np.array([0] * 70 + [1] * 29)                                  # respawn on row 70
    write_episode(str(tmp_path), 0, T=99, tics=tics, deaths=deaths)
    meta = dict(np.load(tmp_path / "ep_00000_meta.npz"))
    rows = eligible_rows(meta, L)
    np.testing.assert_array_equal(rows, tic_window_starts(meta, L, 1) + L)
    # every eligible target has L rows of the same life, with consecutive tics, behind it
    for t in rows:
        assert np.all(np.diff(tics[t - L:t + 1]) == 1)
        assert len(set(deaths[t - L:t + 1].tolist())) == 1
    assert not set(range(40, 40 + L)) & set(rows.tolist())                  # just after the gap
    assert not set(range(70, 70 + L)) & set(rows.tolist())                  # just after the respawn
    assert 40 + L in rows and 70 + L in rows


def test_lives_and_the_valid_window_fraction_are_read_from_the_sidecar(tmp_path):
    deaths = np.array([0] * 30 + [1] * 30 + [2] * 40)
    write_episode(str(tmp_path), 0, T=100, deaths=deaths)
    meta = dict(np.load(tmp_path / "ep_00000_meta.npz"))
    info = ds.episode_info(meta, L)
    assert info["lives"] == 3
    np.testing.assert_array_equal(ds.life_index(deaths)[[0, 29, 30, 59, 60, 99]], [0, 0, 1, 1, 2, 2])
    # candidates are T - L windows of L + 1 rows; each respawn removes the L windows that straddle it
    assert info["eligible"] == 100 - L - 2 * L
    assert info["valid_fraction"] == pytest.approx((100 - L - 2 * L) / (100 - L))


def test_streamed_motion_equals_latent_motion_whatever_the_chunk(tmp_path):
    lat = write_episode(str(tmp_path), 0, T=57)
    mm = np.load(tmp_path / "ep_00000_latents.npy", mmap_mode="r")
    want = latent_motion(lat)
    for chunk in (1, 5, 16, 56, 10_000):
        np.testing.assert_allclose(episode_motion(mm, chunk=chunk), want, rtol=1e-12)
    rows = np.array([3, 9, 40])
    np.testing.assert_allclose(ds.motion_at(mm, rows), want[rows], rtol=1e-12)


def test_the_draw_takes_the_same_number_of_frames_from_every_motion_decile():
    motion = np.random.default_rng(0).exponential(size=503)
    idx, dec = stratified_draw(motion, per_bin=7, bins=10, rng=np.random.default_rng(1))
    assert len(idx) == 70 and len(set(idx.tolist())) == 70 and np.all(np.diff(idx) > 0)
    assert np.bincount(dec, minlength=10).tolist() == [7] * 10
    # the bins are equal-count motion ranks: every frame of a higher bin moved at least as much
    ranks = np.argsort(np.argsort(motion, kind="stable"), kind="stable")
    edges = np.cumsum([len(g) for g in np.array_split(np.arange(503), 10)])
    np.testing.assert_array_equal(np.searchsorted(edges, ranks[idx], side="right"), dec)
    again, _ = stratified_draw(motion, per_bin=7, bins=10, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(idx, again)
    other, _ = stratified_draw(motion, per_bin=7, bins=10, rng=np.random.default_rng(2))
    assert not np.array_equal(idx, other)
    assert stratified_draw(motion[:69], per_bin=7, bins=10, rng=np.random.default_rng(1)) is None


def test_the_reference_draws_are_disjoint_map_balanced_and_reproducible(tmp_path):
    d = training_corpus(str(tmp_path), per_map=6)
    out = tmp_path / "out"
    for draw in (1, 2):
        run("clouds", "--space", "sd1", "--out", out, "--reference", d, "--reference-ids", "0:24",
            "--episodes-per-map", 2, "--draw", draw, *small())
    one = load_cloud(out / "clouds" / "sd1" / "train_draw1.npz")
    two = load_cloud(out / "clouds" / "sd1" / "train_draw2.npz")
    e1, e2 = set(one["frames"]["episode"].tolist()), set(two["frames"]["episode"].tolist())
    assert len(e1) == len(e2) == 8 and not e1 & e2
    for c in (one, two):
        maps = c["frames"]["map"]
        assert sorted(np.unique(maps).tolist()) == list(TRAIN_MAPS)
        assert np.bincount(maps)[list(TRAIN_MAPS)].tolist() == [2 * FPE] * 4
        assert c["meta"]["draw"] in (1, 2) and c["meta"]["kind"] == "reference"
    # the same seed draws the same episodes and frames again
    run("clouds", "--space", "sd1", "--out", tmp_path / "again", "--reference", d, "--reference-ids", "0:24",
        "--episodes-per-map", 2, "--draw", 1, *small())
    again = load_cloud(tmp_path / "again" / "clouds" / "sd1" / "train_draw1.npz")
    for k in ("episode", "row", "feats"):
        np.testing.assert_array_equal(again["frames"][k], one["frames"][k])
    # the id range is honoured: episodes 0:8 hold two per map, so a second draw of two cannot exist
    with pytest.raises(SystemExit):
        run("clouds", "--space", "sd1", "--out", tmp_path / "x", "--reference", d, "--reference-ids", "0:8",
            "--episodes-per-map", 2, "--draw", 2, *small())


def test_cloud_features_are_the_pooled_latents_of_the_drawn_rows(tmp_path):
    d = str(tmp_path / "val")
    lats = {e: write_episode(d, e, map_id=TRAIN_MAPS[e % 4]) for e in range(4)}
    run("clouds", "--space", "sd1", "--out", tmp_path / "out", "--eval", f"val={d}", *small())
    c = load_cloud(tmp_path / "out" / "clouds" / "sd1" / "val.npz")
    f = c["frames"]
    assert f["feats"].shape == (4 * FPE, 192) and f["feats"].dtype == np.float32
    for e, lat in lats.items():
        m = f["episode"] == e
        rows = f["row"][m]
        meta = dict(np.load(os.path.join(d, f"ep_{e:05d}_meta.npz")))
        assert set(rows.tolist()) <= set(eligible_rows(meta, L).tolist())
        np.testing.assert_array_equal(f["feats"][m], pool_latents(lat[rows]))
        np.testing.assert_allclose(f["motion"][m], latent_motion(lat)[rows], rtol=1e-12)
        np.testing.assert_array_equal(f["tic"][m], meta["tic"][rows])
        assert np.all(f["map"][m] == TRAIN_MAPS[e % 4])
        assert np.bincount(f["decile"][m], minlength=10).tolist() == [FPE // 10] * 10
    ep = c["episodes"]
    assert ep["id"].tolist() == [0, 1, 2, 3] and ep["drawn"].tolist() == [FPE] * 4
    assert c["meta"]["space"] == "sd1" and c["meta"]["set"] == "val"


def test_an_episode_without_enough_eligible_frames_is_recorded_and_left_out(tmp_path):
    d = str(tmp_path / "val")
    write_episode(d, 0)
    write_episode(d, 1, T=L + 12)                                           # 12 eligible frames < 20
    run("clouds", "--space", "sd1", "--out", tmp_path / "out", "--eval", f"val={d}", *small())
    c = load_cloud(tmp_path / "out" / "clouds" / "sd1" / "val.npz")
    assert set(c["frames"]["episode"].tolist()) == {0}
    ep = c["episodes"]
    assert ep["id"].tolist() == [0, 1] and ep["drawn"].tolist() == [FPE, 0] and ep["eligible"][1] == 12


def test_limit_caps_the_episodes_per_map(tmp_path):
    d = str(tmp_path / "val")
    for e in range(8):
        write_episode(d, e, map_id=TRAIN_MAPS[e % 2])
    run("clouds", "--space", "sd1", "--out", tmp_path / "out", "--eval", f"val={d}", "--limit", 1, *small())
    c = load_cloud(tmp_path / "out" / "clouds" / "sd1" / "val.npz")
    assert c["episodes"]["id"].tolist() == [0, 1]


def test_eval_sets_are_restricted_to_the_published_split_when_one_exists(tmp_path):
    d = str(tmp_path / "seen")
    for e in range(3):
        write_episode(d, e)
    with open(tmp_path / "split_seen.json", "w") as f:
        json.dump({"val": [0, 2], "meta": {}}, f)
    run("clouds", "--space", "sd1", "--out", tmp_path / "out", "--eval", f"seen={d}", *small())
    c = load_cloud(tmp_path / "out" / "clouds" / "sd1" / "seen.npz")
    assert c["episodes"]["id"].tolist() == [0, 2]
    assert c["meta"]["published_split"].endswith("split_seen.json")


def test_sd35_reuses_the_sd1_frames_by_tic(tmp_path):
    d1, d35 = str(tmp_path / "sd1" / "val"), str(tmp_path / "sd35" / "val")
    lat35 = {}
    for e in range(2):
        write_episode(d1, e)
        lat35[e] = write_episode(d35, e, channels=16)
    out = tmp_path / "out"
    run("clouds", "--space", "sd1", "--out", out, "--eval", f"val={d1}", *small())
    run("clouds", "--space", "sd35", "--out", out, "--eval", f"val={d35}",
        "--frames-from", out / "clouds" / "sd1", *small())
    a = load_cloud(out / "clouds" / "sd1" / "val.npz")["frames"]
    b = load_cloud(out / "clouds" / "sd35" / "val.npz")["frames"]
    np.testing.assert_array_equal(a["episode"], b["episode"])
    np.testing.assert_array_equal(a["tic"], b["tic"])
    assert b["feats"].shape == (2 * FPE, 768)
    for e in range(2):
        m = b["episode"] == e
        np.testing.assert_array_equal(b["feats"][m], pool_latents(lat35[e][b["row"][m]]))
        np.testing.assert_allclose(b["motion"][m], latent_motion(lat35[e])[b["row"][m]], rtol=1e-12)
    # a corpus whose tics do not hold the drawn frame is refused rather than read at the wrong row
    d_bad = str(tmp_path / "bad" / "val")
    for e in range(2):
        write_episode(d_bad, e, channels=16, tics=np.arange(100) + 1000)
    with pytest.raises(SystemExit):
        run("clouds", "--space", "sd35", "--out", tmp_path / "o2", "--eval", f"val={d_bad}",
            "--frames-from", out / "clouds" / "sd1", *small())


def write_raw(parquet_dir, ep, T=100):
    """A raw recording: one PNG per tic, 240 x 320 RGB, and the frames it holds (for checking)."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from PIL import Image
    os.makedirs(parquet_dir, exist_ok=True)
    rng = np.random.default_rng(ep)
    frames = rng.integers(0, 256, size=(T, 240, 320, 3), dtype=np.uint8)
    blobs = []
    for f in frames:
        buf = io.BytesIO()
        Image.fromarray(f).save(buf, format="PNG", compress_level=0)
        blobs.append(buf.getvalue())
    pq.write_table(pa.table({"tic": np.arange(T, dtype=np.int64), "frame": pa.array(blobs, pa.binary())}),
                   os.path.join(parquet_dir, f"ep_{ep:05d}.parquet"))
    return frames


def test_pixel_features_are_the_raw_frames_block_averaged_at_the_drawn_tics(tmp_path):
    d1, raw = str(tmp_path / "lat" / "val"), str(tmp_path / "raw" / "val")
    frames = {}
    for e in range(2):
        write_episode(d1, e)
        frames[e] = write_raw(raw, e)
    out = tmp_path / "out"
    run("clouds", "--space", "sd1", "--out", out, "--eval", f"val={d1}", *small())
    run("clouds", "--space", "pixels", "--out", out, "--eval", f"val={raw}",
        "--frames-from", out / "clouds" / "sd1", *small())
    p = load_cloud(out / "clouds" / "pixels" / "val.npz")["frames"]
    assert p["feats"].shape == (2 * FPE, 900)
    for e in range(2):
        m = p["episode"] == e
        tics = p["tic"][m]
        f = frames[e][tics].astype(np.float64) / 255.0
        want = f.reshape(-1, 15, 16, 20, 16, 3).mean(axis=(2, 4)).transpose(0, 3, 1, 2).reshape(len(tics), -1)
        np.testing.assert_allclose(p["feats"][m], want, atol=1e-6)
        prev = frames[e][tics - 1].astype(np.float64) / 255.0
        np.testing.assert_allclose(p["motion"][m], np.linalg.norm((f - prev).reshape(len(tics), -1), axis=1),
                                   rtol=1e-9)
    # pixels have no latents to rank motion in, so the draw must come from a latent space
    with pytest.raises(SystemExit):
        run("clouds", "--space", "pixels", "--out", tmp_path / "o2", "--eval", f"val={raw}", *small())


# ---------------------------------------------------------------------------------------------
# splits: one eval_tf-readable file per map
# ---------------------------------------------------------------------------------------------

def test_splits_writes_one_eval_tf_readable_file_per_primary_map(tmp_path):
    pytest.importorskip("torch")
    from doom_data import TicWindowDataset, load_split
    val, unseen2 = str(tmp_path / "val"), str(tmp_path / "unseen2")
    for e in range(8):
        write_episode(val, e, map_id=TRAIN_MAPS[e % 4])
    for e in range(4):
        write_episode(unseen2, 100 + e, map_id=18 + e % 2)
    out = tmp_path / "out"
    run("splits", "--out", out, "--eval", f"val={val}", "--eval", f"unseen2={unseen2}", "--context-frames", L)
    names = sorted(os.listdir(out / "splits"))
    assert names == ["index.json"] + [f"split_unseen2_map{m:02d}.json" for m in (18, 19)] + \
        [f"split_val_map{m:02d}.json" for m in TRAIN_MAPS]
    s = load_split(out / "splits" / "split_val_map03.json")
    assert s["val"] == [1, 5]
    assert s["meta"]["subset_key"] == "val" and s["meta"]["num_windows"] == 256
    assert s["meta"]["set"] == "val" and s["meta"]["map"] == 3 and s["meta"]["cluster"] == "arena"
    assert s["meta"]["windows_h1"] == 2 * (100 - L) and s["meta"]["windows_h4"] == 2 * (100 - L - 3)
    assert load_split(out / "splits" / "split_unseen2_map19.json")["meta"]["cluster"] == "campaign"
    # exactly what eval_tf.py builds from `split[args.subset]`: that map's episodes and nothing else
    tw = TicWindowDataset(val, s["val"], L)
    assert sorted(e[0] for e in tw.episodes) == [1, 5] and {e[3] for e in tw.episodes} == {3}
    assert len(tw) == s["meta"]["windows_h1"]
    # a smoke run keeps the first episodes of every map, and counts only their windows
    run("splits", "--out", tmp_path / "lim", "--eval", f"val={val}", "--context-frames", L, "--limit", 1)
    lim = load_split(tmp_path / "lim" / "splits" / "split_val_map03.json")
    assert lim["val"] == [1] and lim["meta"]["windows_h1"] == 100 - L and lim["meta"]["short"]


def test_a_map_in_two_sets_is_scored_once_from_the_set_with_more_episodes(tmp_path):
    val, seen = str(tmp_path / "val"), str(tmp_path / "seen")
    for e in range(6):
        write_episode(val, e, map_id=2)
    for e in range(2):
        write_episode(seen, 50 + e, map_id=2)
        write_episode(seen, 60 + e, map_id=9)
    out = tmp_path / "out"
    run("splits", "--out", out, "--eval", f"seen={seen}", "--eval", f"val={val}", "--context-frames", L)
    assert sorted(os.listdir(out / "splits")) == ["index.json", "split_seen_map09.json", "split_val_map02.json"]
    index = json.load(open(out / "splits" / "index.json"))
    roles = {(r["set"], r["map"]): r["role"] for r in index["maps"]}
    assert roles == {("val", 2): "primary", ("seen", 2): "replication", ("seen", 9): "primary"}
    assert ds.assign_roles({("a", 1): 4, ("b", 1): 4}, ["b", "a"]) == {("b", 1): "primary", ("a", 1): "replication"}


# ---------------------------------------------------------------------------------------------
# distances: the fast path, the mixture, the subsets
# ---------------------------------------------------------------------------------------------

def test_the_sorted_reference_gives_the_committed_sliced_wasserstein():
    rng = np.random.default_rng(0)
    x, y1, y2 = rng.normal(size=(300, 6)), rng.normal(0.3, 1.2, size=(700, 6)), rng.normal(-1, 1, size=(500, 6))
    wx, w1 = rng.uniform(0.1, 3, size=300), rng.exponential(size=700)
    w1[:5] = 0.0                                                             # zero-mass points are allowed
    dirs = ds.random_directions(6, 97, seed=0)
    refs = {"a": (y1, {"m": w1, "u": None}), "b": (y2, {"m": None, "u": None})}
    got = ds.distance_table([(x, {"m": wx}), (lambda: x[:40], {})], refs, dirs, arms=("m", "u"), block=13)
    assert got.shape == (2, 2, 2)
    assert got[0, 0, 0] == pytest.approx(sliced_wasserstein(x, y1, wx, w1, directions=dirs), rel=1e-9)
    assert got[0, 0, 1] == pytest.approx(sliced_wasserstein(x, y2, wx, None, directions=dirs), rel=1e-9)
    assert got[1, 0, 0] == pytest.approx(sliced_wasserstein(x, y1, directions=dirs), rel=1e-9)
    assert got[1, 1, 1] == pytest.approx(sliced_wasserstein(x[:40], y2, directions=dirs), rel=1e-9)
    # one block or many: the same numbers
    one = ds.distance_table([(x, {"m": wx})], refs, dirs, arms=("m",), block=10_000)
    assert one[0, 0, 0] == pytest.approx(got[0, 0, 0], rel=1e-12)
    # a cloud against itself is at zero up to rounding
    self_d = ds.distance_table([(y1, {"m": w1})], {"s": (y1, {"m": w1})}, dirs, arms=("m",))
    assert self_d[0, 0, 0] < 1e-6


def test_a_mixture_of_references_at_fixed_weights_is_the_weighted_pooled_distance():
    rng = np.random.default_rng(1)
    refs = {k: (rng.normal(k, 1.0, size=(200 + 50 * k, 3)), {"m": rng.uniform(size=200 + 50 * k)}) for k in range(3)}
    x, wx = rng.normal(1.0, 1.0, size=(150, 3)), rng.uniform(size=150)
    dirs = ds.random_directions(3, 64, seed=0)
    alpha = np.array([0.2, 0.5, 0.3])
    ys = np.concatenate([refs[k][0] for k in range(3)])
    wy = np.concatenate([alpha[k] * refs[k][1]["m"] / refs[k][1]["m"].sum() for k in range(3)])
    mix = ds.MixtureReference([refs[k] for k in range(3)], dirs, "m")
    xs, wxs, cx = ds.sorted_cloud(x, wx, dirs)
    got = np.sqrt(np.mean(mix.cost(xs, wxs, cx, alpha)))
    assert got == pytest.approx(sliced_wasserstein(x, ys, wx, wy, directions=dirs), rel=1e-9)


def test_the_best_mixture_recovers_the_mixing_weights_and_never_exceeds_the_nearest_map():
    rng = np.random.default_rng(2)
    centres = [np.array([0.0, 0, 0]), np.array([4.0, 0, 0]), np.array([0.0, 4, 0]), np.array([0.0, 0, 4])]
    refs = [(rng.normal(size=(1500, 3)) * 0.5 + c, {"m": None}) for c in centres]
    x = np.concatenate([rng.normal(size=(300, 3)) * 0.5 + centres[0], rng.normal(size=(100, 3)) * 0.5 + centres[1]])
    dirs = ds.random_directions(3, 200, seed=0)
    d_mix, alpha = ds.mixture_distances([(x, {"m": None})], refs, dirs, "m", search_projections=100)
    assert alpha.shape == (1, 4) and alpha[0].sum() == pytest.approx(1.0)
    np.testing.assert_allclose(alpha[0], [0.75, 0.25, 0.0, 0.0], atol=0.07)
    nearest = ds.distance_table([(x, {"m": None})], dict(enumerate(refs)), dirs, arms=("m",)).min()
    assert d_mix[0] < 0.6 * nearest


def test_subsets_average_ten_draws_as_five_disjoint_pairs_or_every_combination_when_fewer():
    rng = np.random.default_rng(3)
    subsets, pairs = ds.episode_subsets(list(range(25)), 4, 10, rng)
    assert len(subsets) == 10 and len(pairs) == 5 and all(len(s) == 4 for s in subsets)
    for i, j in pairs:
        assert not set(subsets[i]) & set(subsets[j])
    assert len({tuple(sorted(s)) for s in subsets}) == 10
    assert ds.episode_subsets([7, 8, 9, 10], 4, 10, rng) == ([[7, 8, 9, 10]], [])
    five, pairs5 = ds.episode_subsets([1, 2, 3, 4, 5], 4, 10, rng)
    assert len(five) == 5 and pairs5 == [] and len({tuple(s) for s in five}) == 5
    short, _ = ds.episode_subsets([1, 2], 4, 10, rng)
    assert short == [[1, 2]]


def test_spearman_kish_and_kendall():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2, 2], [1, 2, 3, 4]) == pytest.approx(np.corrcoef([1.5, 1.5, 3.5, 3.5], [1, 2, 3, 4])[0, 1])
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))
    assert kish_neff(np.ones(40)) == pytest.approx(40.0)
    assert kish_neff(np.array([1.0, 0, 0, 0])) == pytest.approx(1.0)
    assert ds.kendall_tau([1, 2, 3, 4], [1, 2, 4, 3]) == pytest.approx(4 / 6)


# ---------------------------------------------------------------------------------------------
# distances end to end, on cloud files with a known geometry
# ---------------------------------------------------------------------------------------------

DIM = 6
CENTRES = {2: 0, 3: 1, 4: 2, 5: 3}          # the training maps sit on four orthogonal axes
FIRST_ID = {"train_draw1": 0, "train_draw2": 5000, "val": 10_000, "arenas_678": 11_000, "seen": 12_000,
            "unseen2": 13_000}


def centre(axis, r):
    c = np.zeros(DIM)
    c[axis] = r
    return c


def write_set(path, set_name, maps, episodes, rng, kind="eval", heavy=()):
    """A cloud file: `maps` is {map_id: centre}; each episode holds FPE frames around it. Maps in
    `heavy` get heavy-tailed motion, so their motion weights concentrate and n_eff falls."""
    feats, motion, epi, mp, ids, emap = [], [], [], [], [], []
    e = FIRST_ID[set_name]
    for m, c in maps.items():
        for _ in range(episodes):
            feats.append(rng.normal(size=(FPE, DIM)) * 0.5 + c)
            motion.append(rng.pareto(0.7, size=FPE) + 1e-3 if m in heavy else rng.uniform(0.5, 1.5, size=FPE))
            epi.append(np.full(FPE, e))
            mp.append(np.full(FPE, m))
            ids.append(e)
            emap.append(m)
            e += 1
    n = len(ids)
    frames = {"feats": np.concatenate(feats).astype(np.float32), "motion": np.concatenate(motion),
              "episode": np.concatenate(epi), "map": np.concatenate(mp),
              "tic": np.tile(np.arange(FPE), n), "row": np.tile(np.arange(FPE), n),
              "life": np.zeros(n * FPE, np.int64), "decile": np.tile(np.arange(FPE) % 10, n)}
    episodes_tab = {"id": np.array(ids), "map": np.array(emap), "rows": np.full(n, 1000),
                    "lives": np.full(n, 3), "eligible": np.full(n, 900), "valid_fraction": np.full(n, 0.9),
                    "drawn": np.full(n, FPE)}
    save_cloud(path, frames, episodes_tab, {"space": "toy", "set": set_name, "kind": kind,
                                            "draw": 2 if set_name == "train_draw2" else 1})


def toy_study(root, heavy=(), seed=0):
    rng = np.random.default_rng(seed)
    space = os.path.join(root, "clouds", "toy")
    os.makedirs(space, exist_ok=True)
    train = {m: centre(a, 3.0) for m, a in CENTRES.items()}
    write_set(os.path.join(space, "train_draw1.npz"), "train_draw1", train, 8, rng, "reference")
    write_set(os.path.join(space, "train_draw2.npz"), "train_draw2", train, 8, rng, "reference")
    write_set(os.path.join(space, "val.npz"), "val", train, 10, rng)
    near = {m: centre(CENTRES[2], 3.0) + centre(4, 0.8 + 0.2 * i) for i, m in enumerate((6, 7, 8))}
    write_set(os.path.join(space, "arenas_678.npz"), "arenas_678", near, 8, rng)
    write_set(os.path.join(space, "seen.npz"), "seen",
              {1: centre(CENTRES[3], 3.0) + centre(5, 1.2), 2: train[2], 3: train[3]}, 4, rng)
    far = {m: centre(4, 2.5 + 0.5 * i) + centre(5, 2.0) for i, m in enumerate((18, 19, 20))}
    write_set(os.path.join(space, "unseen2.npz"), "unseen2", far, 10, rng, heavy=heavy)
    return space


def distances(root, *extra):
    return run("distances", "--space", "toy", "--out", root, "--projections", 96, "--seed", 0, *extra)


def test_distances_end_to_end_on_clouds_with_a_known_geometry(tmp_path):
    toy_study(str(tmp_path))
    assert distances(tmp_path, "--bootstrap", 20) == 0
    out = json.load(open(tmp_path / "distances_toy.json"))
    maps = {m["key"]: m for m in out["maps"]}
    assert set(maps) == {"val/2", "val/3", "val/4", "val/5", "arenas_678/6", "arenas_678/7", "arenas_678/8",
                         "seen/1", "seen/2", "seen/3", "unseen2/18", "unseen2/19", "unseen2/20"}
    assert maps["seen/2"]["role"] == "replication" and maps["val/2"]["role"] == "primary"
    assert maps["unseen2/18"]["cluster"] == "campaign" and maps["seen/1"]["cluster"] == "arena"
    # each validation map is nearest to itself, and ten subsets were averaged as five disjoint pairs
    for m in TRAIN_MAPS:
        v = maps[f"val/{m}"]
        assert v["nearest"] == m and v["subsets"] == 10 and len(v["disjoint_pairs"]) == 5
        assert set(v["per_reference"]) == {str(k) for k in TRAIN_MAPS}
        assert v["D"] == pytest.approx(v["per_subset"]["motion"]["mean"])
    assert maps["seen/1"]["subsets"] == 1
    # the distance orders the geometry: seen maps < near arenas < far campaign maps
    d = {k: v["D"] for k, v in maps.items()}
    assert max(d[f"val/{m}"] for m in TRAIN_MAPS) < min(d["arenas_678/6"], d["arenas_678/7"], d["arenas_678/8"])
    assert max(d["arenas_678/6"], d["arenas_678/7"], d["arenas_678/8"]) < min(d["unseen2/18"], d["unseen2/19"])
    # the pooled corpus charges a seen map for the other three maps; the mixture can only be nearer
    assert maps["val/2"]["D_pooled"] > 3 * maps["val/2"]["D"]
    for v in maps.values():
        assert v["D_mixture"] <= v["D"] * (1 + 1e-9)
        assert sum(v["mixture_weights"].values()) == pytest.approx(1.0)
    # the floor, the checks and the arm
    floor = out["floor"]["motion"]
    assert floor["n"] == 4 * 10 and floor["min"] <= floor["max"]
    checks = out["checks"]
    assert out["primary_arm"] == "motion" and checks["v"]["pass"]
    assert checks["i"]["pass"], checks["i"]
    assert checks["iii"]["spearman"] >= 0.95 and checks["iii"]["pass"]
    assert checks["iv"]["pass"]
    assert set(checks["ii"]["replication"]) == {"2", "3"}
    assert out["code"]["distance_study_sha256"] and "git_commit" in out["code"]
    assert out["config"]["projections"] == 96
    # the bootstrap: one row of draws per map point, and the attenuation read
    boot = np.load(tmp_path / "bootstrap_toy.npz")
    assert boot["draws"].shape == (13, 20) and list(boot["keys"]) == [m["key"] for m in out["maps"]]
    assert maps["val/2"]["bootstrap"]["n"] == 20 and maps["val/2"]["bootstrap"]["sd"] > 0
    assert out["bootstrap"]["attenuation_ratio"] < 0.1
    # the per-episode table: every episode with frames, its distance to every training map
    rows = list(csv.DictReader(open(tmp_path / "per_episode_toy.csv")))
    assert len(rows) == 4 * 10 + 3 * 8 + 3 * 4 + 3 * 10
    r = rows[0]
    for col in ("set", "map", "episode", "role", "cluster", "frames", "lives", "valid_fraction", "n_eff",
                "D", "nearest", "D_uniform", "D_pooled", "D_pooled_uniform", "d_2", "d_5", "d_uniform_2"):
        assert col in r
    assert float(r["D"]) == pytest.approx(min(float(r[f"d_{k}"]) for k in TRAIN_MAPS))


def test_check_v_makes_the_uniform_arm_primary_when_distance_tracks_n_eff(tmp_path):
    toy_study(str(tmp_path), heavy=(18, 19, 20))
    distances(tmp_path, "--no-mixture")
    out = json.load(open(tmp_path / "distances_toy.json"))
    assert not out["checks"]["v"]["pass"] and abs(out["checks"]["v"]["spearman"]) >= 0.4
    assert out["primary_arm"] == "uniform"
    m = {v["key"]: v for v in out["maps"]}["unseen2/18"]
    assert m["D"] == pytest.approx(m["per_subset"]["uniform"]["mean"])
    assert m["D_motion"] == pytest.approx(m["per_subset"]["motion"]["mean"])
    assert "D_mixture" not in m


def test_limit_scores_only_the_first_map_points(tmp_path):
    toy_study(str(tmp_path))
    distances(tmp_path, "--limit", 3, "--no-mixture")
    out = json.load(open(tmp_path / "distances_toy.json"))
    assert len(out["maps"]) == 3 and out["config"]["limit"] == 3
