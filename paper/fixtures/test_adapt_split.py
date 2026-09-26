"""CPU gates for the per-map adaptation split (`adapt_split.py`).

The split is the experiment's identity: every adaptation checkpoint records its hash, and every score
of every checkpoint is read on the windows it lists. So:

* **deterministic** in (the map's episodes, seed), whatever order the episodes arrive in;
* **the budget is a flag, 16 adapt and 8 held out by default** on the 24-episode target arenas;
* **disjoint**: no held-out episode is ever adapted on, on any rung of the ladder;
* **nested data ladder**: rungs 1, 2, 4, 8, 16 are the first k of one ordered adapt list, and the step
  curve is the k = 8 rung unless a run asks for another;
* **held-out windows only from held-out episodes**, a fixed number from each, every one a real window of
  the dataset the evaluator builds; the legacy (distance-study) draw is exactly that draw restricted to
  the held-out episodes;
* **episode ids come from the corpus**: its manifest (checked against the directory) or the directory
  listing itself, never from a hard-coded count.

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
SMALL = dict(n_adapt=6, n_held_out=4, ladder=(1, 2, 4, 6), step_curve_k=4)      # a 10-episode map


# ---------------------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------------------

def test_the_default_budget_is_sixteen_and_eight_with_the_ladder_and_the_step_curve_at_eight():
    s = adapt_split.make_adapt_split(range(500, 524), seed=0)
    assert len(s["adapt"]) == 16 and len(s["held_out"]) == 8
    assert s["ladder"] == [1, 2, 4, 8, 16] and s["step_curve_k"] == 8
    assert adapt_split.adapt_episodes(s) == sorted(s["adapt"][:8])
    rungs = [adapt_split.adapt_episodes(s, k) for k in s["ladder"]]
    for small, big in zip(rungs, rungs[1:]):
        assert set(small) < set(big)
    assert set(rungs[-1]) == set(s["adapt"]) and not set(rungs[-1]) & set(s["held_out"])


@pytest.mark.parametrize("n", [10, 20, 24])
def test_adapt_list_and_held_out_are_disjoint_and_cover_the_map(n):
    eps = list(range(7, 7 + 3 * n, 3))
    s = adapt_split.make_adapt_split(eps, seed=3, **SMALL)
    assert not set(s["held_out"]) & set(s["adapt_pool"])
    assert sorted(s["held_out"] + s["adapt_pool"]) == sorted(eps)
    assert s["adapt"] == s["adapt_pool"][:6]


def test_the_split_is_deterministic_and_order_free():
    eps = list(range(1, 49, 2))
    a = adapt_split.make_adapt_split(eps, seed=0)
    assert a == adapt_split.make_adapt_split(list(reversed(eps)), seed=0)
    assert a != adapt_split.make_adapt_split(eps, seed=1)


def test_the_counts_and_the_ladder_are_flags_and_are_checked():
    s = adapt_split.make_adapt_split(range(10), seed=0, n_adapt=8, n_held_out=2, ladder=(2, 8), step_curve_k=2)
    assert len(s["adapt"]) == 8 and len(s["held_out"]) == 2 and s["ladder"] == [2, 8]
    with pytest.raises(ValueError, match="exceed"):
        adapt_split.make_adapt_split(range(20), seed=0)                    # 16 + 8 > 20
    with pytest.raises(ValueError, match="within"):
        adapt_split.make_adapt_split(range(24), seed=0, ladder=(1, 32))
    with pytest.raises(ValueError, match="within"):
        adapt_split.make_adapt_split(range(24), seed=0, step_curve_k=17)
    with pytest.raises(ValueError, match="repeats"):
        adapt_split.make_adapt_split([1, 1, 2, 3, 4, 5], seed=0, n_adapt=2, n_held_out=1, ladder=(1,), step_curve_k=1)


def test_a_rung_beyond_the_adapt_list_is_refused():
    s = adapt_split.make_adapt_split(range(24), seed=0)
    assert len(adapt_split.adapt_episodes(s, 16)) == 16
    with pytest.raises(ValueError, match="holds 16"):
        adapt_split.adapt_episodes(s, 17)


# ---------------------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------------------

def corpus(path, ids=tuple(range(1, 20, 2)), T=40, map_id=17):
    """Per-tic episodes of one map; every other episode dies once, so some windows are invalid."""
    for i, ep in enumerate(ids):
        deaths = np.zeros(T, dtype=np.int64)
        if i % 2:
            deaths[T // 2:] = 1
        write_pertic_episode(str(path), ep, held_actions([0] * (T // 4)), deaths=deaths, map_ids=np.full(T, map_id))
    return str(path), list(ids)


def test_the_split_writer_draw_is_eval_tfs_draw():
    import eval_tf
    for n, k, seed in ((1000, 256, 0), (100, 256, 3), (5000, 64, 9)):
        assert np.array_equal(adapt_split.draw_windows(n, k, seed), eval_tf.draw_windows(n, k, seed))


def test_a_fixed_number_of_windows_from_each_held_out_episode_and_nothing_else(tmp_path):
    d, eps = corpus(tmp_path / "lat")
    s = adapt_split.build_split(eps, "unseen", 17, d, seed=0, windows_per_episode=5, context_frames=CTX, **SMALL)
    held = set(s["held_out"])
    assert s["val"] == s["held_out"] and s["train"] == sorted(s["adapt"][:4])
    pairs = s[adapt_split.WINDOW_KEY]
    assert len(pairs) == 20 and len({tuple(p) for p in pairs}) == 20
    assert {e for e, _ in pairs} == held
    assert all(sum(1 for e, _ in pairs if e == h) == 5 for h in held)
    for e, start in pairs:
        meta = np.load(os.path.join(d, f"ep_{e:05d}_meta.npz"))
        assert start in set(tic_window_starts(meta, CTX, 1).tolist())
    ds = TicWindowDataset(d, s["val"], CTX)
    idx = adapt_split.windows_in_dataset(ds, pairs)
    back = sorted([int(ds.episodes[ds.locate(i)[0]][0]), ds.locate(i)[1]] for i in idx)
    assert back == sorted(pairs)


def test_each_episodes_windows_are_its_own(tmp_path):
    d, eps = corpus(tmp_path / "lat")
    both, _ = adapt_split.draw_held_out_windows(d, [eps[0], eps[1]], 6, seed=2, context_frames=CTX)
    alone, avail = adapt_split.draw_held_out_windows(d, [eps[0]], 6, seed=2, context_frames=CTX)
    assert [p for p in both if p[0] == eps[0]] == alone
    other, _ = adapt_split.draw_held_out_windows(d, [eps[0]], 6, seed=3, context_frames=CTX)
    assert other != alone
    # an episode with fewer valid windows than asked gives all it has
    few, avail = adapt_split.draw_held_out_windows(d, [eps[1]], 10_000, seed=0, context_frames=CTX)
    assert len(few) == avail[eps[1]] < 10_000


def test_the_legacy_draw_is_the_study_draw_restricted_to_held_out(tmp_path):
    import eval_tf
    d, eps = corpus(tmp_path / "lat")
    s = adapt_split.build_split(eps, "unseen", 17, d, seed=2, windows_per_episode=3, context_frames=CTX,
                                legacy=True, legacy_windows=50, legacy_seed=0, **SMALL)
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


# ---------------------------------------------------------------------------------------
# the episode source: manifest or directory listing
# ---------------------------------------------------------------------------------------

def two_maps(path):
    """A 24-episode map 13 and a 24-episode map 14, ids interleaved as a recorder hands them out."""
    d = str(path)
    corpus(d, ids=tuple(range(0, 48, 2)), T=24, map_id=13)
    corpus(d, ids=tuple(range(1, 48, 2)), T=24, map_id=14)
    return d


@pytest.mark.parametrize("shape", ["jsonl", "records", "episodes", "maps", "bare"])
def test_every_manifest_shape_gives_the_same_maps(tmp_path, shape):
    recs = [{"episode_id": e, "map_id": 13 if e % 2 == 0 else 14, "seeds": [e]} for e in range(48)]
    maps = {"13": list(range(0, 48, 2)), "14": list(range(1, 48, 2))}
    path = tmp_path / "manifest"
    if shape == "jsonl":
        path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    else:
        path.write_text(json.dumps({"records": recs, "episodes": {"episodes": recs}, "maps": {"maps": maps},
                                    "bare": maps}[shape]))
    assert adapt_split.manifest_maps(str(path)) == {13: list(range(0, 48, 2)), 14: list(range(1, 48, 2))}


def test_the_directory_listing_groups_episodes_by_their_sidecar_map(tmp_path):
    d = two_maps(tmp_path / "arenas13")
    assert adapt_split.directory_maps(d) == {13: list(range(0, 48, 2)), 14: list(range(1, 48, 2))}


def test_a_manifest_that_disagrees_with_the_directory_is_refused(tmp_path):
    d = two_maps(tmp_path / "arenas13")
    on_disk = adapt_split.directory_maps(d)
    with pytest.raises(SystemExit, match="disagrees"):
        adapt_split.check_against_directory({13: [0, 2, 1]}, on_disk, d)        # episode 1 is on map 14
    with pytest.raises(SystemExit, match="disagrees"):
        adapt_split.check_against_directory({13: [0, 2, 100]}, on_disk, d)      # episode 100 is not there


def test_the_cli_writes_one_split_per_map_from_the_manifest_or_the_listing(tmp_path, capsys):
    d = two_maps(tmp_path / "arenas13")
    man = tmp_path / "manifest.jsonl"
    man.write_text("\n".join(json.dumps({"episode_id": e, "map_id": 13 if e % 2 == 0 else 14}) for e in range(48)))
    out = tmp_path / "splits"
    argv = ["--latents-dir", d, "--manifest", str(man), "--map", "all", "--windows-per-episode", "2",
            "--context-frames", str(CTX), "--out-dir", str(out)]
    assert adapt_split.main(argv) == 0
    for m in (13, 14):
        s = json.loads((out / f"split_adapt_arenas13_map{m}_seed0.json").read_text())
        assert s["meta"]["kind"] == "adapt_split" and s["meta"]["source_kind"] == "manifest"
        assert s["meta"]["set"] == "arenas13" and s["meta"]["map"] == m and s["meta"]["source_sha256"]
        assert len(s["val"]) == 8 and len(s["adapt"]) == 16 and len(s["train"]) == 8
        assert len(s[adapt_split.WINDOW_KEY]) == 16 and adapt_split.LEGACY_KEY not in s
        assert all(e % 2 == m - 13 for e in s["val"] + s["adapt"])
    # the directory listing alone gives the same splits, and a rerun is a no-op
    before = (out / "split_adapt_arenas13_map13_seed0.json").read_bytes()
    assert adapt_split.main(["--latents-dir", d, "--map", "13", "--windows-per-episode", "2", "--context-frames",
                             str(CTX), "--out-dir", str(out)]) == 0
    assert (out / "split_adapt_arenas13_map13_seed0.json").read_bytes() == before
    with pytest.raises(SystemExit, match="different split"):
        adapt_split.main(argv[:-2] + ["--seed", "1", "--map", "13", "--out", str(out / "split_adapt_arenas13_map13_seed0.json")])
    with pytest.raises(SystemExit, match="not in the source"):
        adapt_split.main(["--latents-dir", d, "--map", "15", "--out-dir", str(out)])
    capsys.readouterr()


def test_the_distance_study_split_still_works_and_carries_the_legacy_draw(tmp_path, capsys):
    d, eps = corpus(tmp_path / "lat")
    src = tmp_path / "split_unseen_map17.json"
    src.write_text(json.dumps({"val": eps, "meta": {"set": "unseen", "map": 17, "latents_dir": "/elsewhere"}}))
    path = tmp_path / "split_adapt_unseen_map17.json"
    assert adapt_split.main(["--map-split", str(src), "--latents-dir", d, "--adapt-episodes", "6",
                             "--held-out-episodes", "4", "--ladder", "1,2,4,6", "--step-curve-k", "4",
                             "--windows-per-episode", "3", "--context-frames", str(CTX), "--legacy-windows", "64",
                             "--out", str(path)]) == 0
    s = json.loads(path.read_text())
    assert s["meta"]["source_kind"] == "distance_study_split" and s[adapt_split.LEGACY_KEY]
    assert adapt_split.load_windows(str(path)) == s[adapt_split.WINDOW_KEY]
    assert adapt_split.load_windows(str(path), adapt_split.LEGACY_KEY) == s[adapt_split.LEGACY_KEY]
    capsys.readouterr()
