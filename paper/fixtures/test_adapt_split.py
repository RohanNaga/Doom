"""CPU gates for the per-map adaptation split (`adapt_split.py`).

The split is the experiment's identity: every adaptation checkpoint records its hash, and every score
of every checkpoint is read on the windows it lists. So:

* **deterministic** in (the map's episodes, seed), whatever order the episodes arrive in;
* **the same budget on every map**, 6 adapt and 4 held out by default, 10 episodes or 20;
* **disjoint**: no held-out episode is ever adapted on, including through the data grid;
* **nested data grid**: the first k of one ordered pool, so k = 1, 2, 4, 6 share one held-out set;
* **held-out windows only from held-out episodes**, each one a real window of the dataset the evaluator
  builds, and the legacy (distance-study) draw is exactly that draw restricted to the held-out episodes.

    python -m pytest paper/fixtures/test_adapt_split.py -q
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from pertic_fixtures import held_actions, write_pertic_episode  # noqa: E402

import adapt_split  # noqa: E402
from doom_data import TicWindowDataset, tic_window_starts  # noqa: E402

CTX = 4


# ---------------------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [10, 20])
def test_the_default_budget_is_six_and_four_on_every_map(n):
    s = adapt_split.make_adapt_split(range(100, 100 + n), seed=0)
    assert len(s["adapt"]) == 6 and len(s["held_out"]) == 4
    assert len(s["adapt_pool"]) == n - 4


@pytest.mark.parametrize("n", [10, 20])
def test_adapt_pool_and_held_out_are_disjoint_and_cover_the_map(n):
    eps = list(range(7, 7 + 3 * n, 3))
    s = adapt_split.make_adapt_split(eps, seed=3)
    assert not set(s["held_out"]) & set(s["adapt_pool"])
    assert sorted(s["held_out"] + s["adapt_pool"]) == sorted(eps)
    assert set(s["adapt"]) <= set(s["adapt_pool"])


def test_the_split_is_deterministic_and_order_free():
    eps = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    a = adapt_split.make_adapt_split(eps, seed=0)
    assert a == adapt_split.make_adapt_split(list(reversed(eps)), seed=0)
    assert a != adapt_split.make_adapt_split(eps, seed=1)


def test_the_ratio_is_a_flag():
    s = adapt_split.make_adapt_split(range(10), seed=0, n_adapt=8, n_held_out=2)
    assert len(s["adapt"]) == 8 and len(s["held_out"]) == 2
    with pytest.raises(ValueError, match="exceed"):
        adapt_split.make_adapt_split(range(10), seed=0, n_adapt=7, n_held_out=4)
    with pytest.raises(ValueError, match="repeats"):
        adapt_split.make_adapt_split([1, 1, 2, 3, 4, 5], seed=0, n_adapt=2, n_held_out=1)


@pytest.mark.parametrize("n", [10, 20])
def test_the_data_grid_is_a_nested_prefix_of_one_pool(n):
    s = adapt_split.make_adapt_split(range(n), seed=0)
    grid = [adapt_split.adapt_episodes(s, k) for k in (1, 2, 4, 6)]
    for small, big in zip(grid, grid[1:]):
        assert set(small) < set(big)
    assert adapt_split.adapt_episodes(s, 6) == s["adapt"] == adapt_split.adapt_episodes(s, 0)
    assert not set(adapt_split.adapt_episodes(s, n - 4)) & set(s["held_out"])
    with pytest.raises(ValueError, match="pool"):
        adapt_split.adapt_episodes(s, n - 3)


def test_a_twenty_episode_map_reaches_sixteen_episodes_through_the_grid_only():
    s = adapt_split.make_adapt_split(range(20), seed=0)
    assert len(s["adapt"]) == 6
    assert len(adapt_split.adapt_episodes(s, 16)) == 16
    assert len(adapt_split.adapt_episodes(s, 12)) == 12


# ---------------------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------------------

def corpus(path, ids=(1, 3, 5, 7, 9, 11, 13, 15, 17, 19), T=40):
    """Ten per-tic episodes of one map; odd episodes die once, so some windows are invalid."""
    for i, ep in enumerate(ids):
        deaths = np.zeros(T, dtype=np.int64)
        if i % 2:
            deaths[T // 2:] = 1
        write_pertic_episode(str(path), ep, held_actions([0] * (T // 4)), deaths=deaths, map_ids=np.full(T, 17))
    return str(path), list(ids)


def test_the_split_writer_draw_is_eval_tfs_draw():
    import eval_tf
    for n, k, seed in ((1000, 256, 0), (100, 256, 3), (5000, 64, 9)):
        assert np.array_equal(adapt_split.draw_windows(n, k, seed), eval_tf.draw_windows(n, k, seed))


def test_held_out_windows_come_only_from_held_out_episodes_and_are_dataset_windows(tmp_path):
    d, eps = corpus(tmp_path / "lat")
    s = adapt_split.build_split(eps, "unseen", 17, d, seed=0, num_windows=40, context_frames=CTX, legacy=False)
    held = set(s["held_out"])
    assert s["val"] == s["held_out"] and s["train"] == s["adapt"]
    pairs = s[adapt_split.WINDOW_KEY]
    assert len(pairs) == 40 and len({tuple(p) for p in pairs}) == 40
    assert {e for e, _ in pairs} <= held
    for e, start in pairs:
        meta = np.load(os.path.join(d, f"ep_{e:05d}_meta.npz"))
        assert start in set(tic_window_starts(meta, CTX, 1).tolist())
    ds = TicWindowDataset(d, s["val"], CTX)
    idx = adapt_split.windows_in_dataset(ds, pairs)
    back = sorted([int(ds.episodes[ds.locate(i)[0]][0]), ds.locate(i)[1]] for i in idx)
    assert back == sorted(pairs)
    # deterministic in the seed, and a different seed draws different windows
    again = adapt_split.build_split(eps, "unseen", 17, d, seed=0, num_windows=40, context_frames=CTX, legacy=False)
    other = adapt_split.build_split(eps, "unseen", 17, d, seed=1, num_windows=40, context_frames=CTX, legacy=False)
    assert again[adapt_split.WINDOW_KEY] == pairs and other[adapt_split.WINDOW_KEY] != pairs


def test_the_legacy_draw_is_the_study_draw_restricted_to_held_out(tmp_path):
    import eval_tf
    d, eps = corpus(tmp_path / "lat")
    s = adapt_split.build_split(eps, "unseen", 17, d, seed=2, num_windows=30, context_frames=CTX,
                                legacy_windows=50, legacy_seed=0)
    full = TicWindowDataset(d, eps, CTX)
    drawn = eval_tf.draw_windows(len(full), 50, 0)
    study = [[int(full.episodes[full.locate(i)[0]][0]), full.locate(i)[1]] for i in drawn]
    want = [p for p in study if p[0] in set(s["held_out"])]
    assert s[adapt_split.LEGACY_KEY] == want and 0 < len(want) < 50
    assert s["meta"]["legacy"]["drawn"] == 50 and s["meta"]["legacy"]["kept_in_held_out"] == len(want)
    # scored against the map's full split, the legacy windows keep the study's own indices (and noise keys)
    assert set(adapt_split.windows_in_dataset(full, want).tolist()) <= set(drawn.tolist())


def test_a_window_the_dataset_does_not_hold_is_refused(tmp_path):
    d, eps = corpus(tmp_path / "lat")
    ds = TicWindowDataset(d, eps[:2], CTX)
    with pytest.raises(ValueError, match="not windows of this dataset"):
        adapt_split.windows_in_dataset(ds, [[eps[5], 0]])
    with pytest.raises(ValueError, match="repeats"):
        adapt_split.windows_in_dataset(ds, [[eps[0], 0], [eps[0], 0]])


def test_the_cli_writes_one_file_per_map_and_seed_and_never_silently_replaces_it(tmp_path, capsys):
    d, eps = corpus(tmp_path / "lat")
    src = tmp_path / "split_unseen_map17.json"
    src.write_text(json.dumps({"val": eps, "meta": {"set": "unseen", "map": 17, "latents_dir": "/elsewhere"}}))
    out = tmp_path / "splits"
    argv = ["--map-split", str(src), "--latents-dir", d, "--seed", "0", "--windows", "32",
            "--context-frames", str(CTX), "--legacy-windows", "64", "--out-dir", str(out)]
    assert adapt_split.main(argv) == 0
    path = out / "split_adapt_unseen_map17_seed0.json"
    s = json.loads(path.read_text())
    assert s["meta"]["kind"] == "adapt_split" and s["meta"]["source_split_sha256"]
    assert len(s["val"]) == 4 and len(s["train"]) == 6
    assert adapt_split.load_windows(str(path)) == s[adapt_split.WINDOW_KEY]
    assert adapt_split.load_windows(str(path), adapt_split.LEGACY_KEY) == s[adapt_split.LEGACY_KEY]
    before = path.read_bytes()
    assert adapt_split.main(argv) == 0 and path.read_bytes() == before     # a rerun is a no-op
    with pytest.raises(SystemExit, match="different split"):
        adapt_split.main(argv[:5] + ["1"] + argv[6:] + ["--out", str(path)])
    assert adapt_split.main(argv[:5] + ["1"] + argv[6:]) == 0
    assert (out / "split_adapt_unseen_map17_seed1.json").exists()
    capsys.readouterr()
