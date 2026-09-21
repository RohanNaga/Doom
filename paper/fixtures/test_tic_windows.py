"""The per-tic window dataset: which action conditions which target, and which windows are legal.

Everything the next-tic runs rest on is here. Two rules have to be exactly right or the whole
experiment is uninterpretable:

  * the target tic is conditioned on the action *in effect* over the transition into it, which is
    the action stored on the last context row, and
  * a window never spans a tic gap, a death, a map change or an episode boundary.

Both are checked on synthetic episodes whose latents are filled with their own row index, so a test
can read a tensor and name the tic it came from instead of trusting the indexing arithmetic.

    python -m pytest paper/fixtures/test_tic_windows.py -q
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

pytest.importorskip("torch")

from pertic_fixtures import (STRIDE, frame_index_of, held_actions,  # noqa: E402
                             write_pertic_episode, write_stride4_episode)

import doom_data  # noqa: E402
from doom_data import (PHASE_BUCKETS, PHASE_OFF_GRID, LatentWindowDataset, TicWindowDataset,  # noqa: E402
                       assert_disjoint, limit_to_encoded, parse_episode_ids, tic_window_starts,
                       tics_since_decision)


def corpus(tmp_path, name, actions, **kw):
    d = str(tmp_path / name)
    write_pertic_episode(d, 0, actions, **kw)
    return d


# ---------------------------------------------------------------------------------------
# which action conditions which target
# ---------------------------------------------------------------------------------------

def test_the_target_is_conditioned_on_the_action_held_over_its_own_transition(tmp_path):
    """The rule, on a window that both contains a change of action and ends on one.

    Decisions 5, 6, 7, 8 are held four tics each, so rows 0-3 carry 5, rows 4-7 carry 6, rows 8-11
    carry 7. With L = 6 and start = 2 the context is tics 2 to 7 -- the action changes from 5 to 6
    inside it -- and the target is tic 8. The action that carries tic 7 into tic 8 is the one on
    row 7, which is 6. It is neither the action of the decision that opened the context (5) nor the
    value stored on the target row itself (7, the next decision, which begins *at* tic 8).
    """
    acts = held_actions([5, 6, 7, 8])
    ds = TicWindowDataset(corpus(tmp_path, "held", acts), context_frames=6)
    starts = [ds.locate(i)[1] for i in range(len(ds))]
    i = starts.index(2)
    ctx, tgt, a = ds[i]
    assert [frame_index_of(ctx[4 * k:4 * (k + 1)]) for k in range(6)] == [2, 3, 4, 5, 6, 7]
    assert frame_index_of(tgt) == 8
    assert sorted(set(acts[2:8].tolist())) == [5, 6]      # the action really does change mid-window
    assert int(a) == int(acts[7]) == 6
    assert int(a) != int(acts[8]) == 7                    # not the target row's own action
    assert int(a) != int(acts[2]) == 5                    # not the oldest context row's action


def test_every_window_reads_the_action_of_its_last_context_row(tmp_path):
    acts = held_actions([1, 2, 3, 4, 5, 6, 7, 8])
    ds = TicWindowDataset(corpus(tmp_path, "all", acts), context_frames=6)
    for i in range(len(ds)):
        _, start = ds.locate(i)
        _, tgt, a = ds[i]
        assert ds.held_action_row(start) == start + 6 - 1
        assert int(a) == int(acts[start + 6 - 1])
        assert frame_index_of(tgt) == start + 6


def test_the_conditioning_rule_is_the_one_the_stride4_dataset_uses(tmp_path):
    """Same index, so the only difference between a next-tic and a next-decision row is spacing."""
    per = corpus(tmp_path, "per", held_actions([2, 3, 4, 5, 6, 7]))
    dec = str(tmp_path / "dec")
    write_stride4_episode(dec, 0, [2, 3, 4, 5, 6, 7])
    tic_ds = TicWindowDataset(per, context_frames=4)
    dec_ds = LatentWindowDataset(dec, context_frames=4)
    assert int(dec_ds[0][2]) == 5                       # act[start + L - 1] with start 0
    slot_starts = [tic_ds.locate(i)[1] for i in range(len(tic_ds))]
    assert int(tic_ds[slot_starts.index(0)][2]) == int(held_actions([2, 3, 4, 5, 6, 7])[3])


# ---------------------------------------------------------------------------------------
# window validity
# ---------------------------------------------------------------------------------------

def test_a_missing_tic_splits_the_windows(tmp_path):
    """An unrendered tic in the middle means the pair around it is two tics apart, not one."""
    tics = np.concatenate([np.arange(0, 8), np.arange(9, 16)])       # tic 8 never rendered
    d = corpus(tmp_path, "gap", held_actions([1, 2, 3, 4])[:15], tics=tics,
               decisions=(tics % STRIDE) == 0)
    ds = TicWindowDataset(d, context_frames=4)
    tgt_tics = sorted(ds.target_tic(i) for i in range(len(ds)))
    assert 8 not in tgt_tics and 9 not in tgt_tics      # no window may bridge the hole
    assert all(t <= 7 or t >= 13 for t in tgt_tics), tgt_tics


def test_a_death_ends_every_window(tmp_path):
    """`deaths` increments on the respawn row, so the pixels jump there; no window may include it."""
    deaths = np.array([0] * 8 + [1] * 8)
    d = corpus(tmp_path, "death", held_actions([1, 2, 3, 4]), deaths=deaths)
    ds = TicWindowDataset(d, context_frames=4)
    for i in range(len(ds)):
        _, s = ds.locate(i)
        assert len(set(deaths[s:s + 5].tolist())) == 1, f"window at {s} crosses the respawn"
    assert 8 not in {ds.target_tic(i) for i in range(len(ds))}


def test_a_map_change_ends_every_window(tmp_path):
    maps = np.array([7] * 8 + [9] * 8)
    d = corpus(tmp_path, "maps", held_actions([1, 2, 3, 4]), map_ids=maps)
    ds = TicWindowDataset(d, context_frames=4)
    for i in range(len(ds)):
        _, s = ds.locate(i)
        assert len(set(maps[s:s + 5].tolist())) == 1


def test_windows_never_cross_an_episode_boundary(tmp_path):
    d = str(tmp_path / "two")
    write_pertic_episode(d, 0, held_actions([1, 2]))       # 8 tics
    write_pertic_episode(d, 1, held_actions([3, 4]))       # 8 tics
    ds = TicWindowDataset(d, context_frames=4)
    assert len(ds) == 2 * (8 - 4)                          # 4 windows per episode, never 2*8-4
    for i in range(len(ds)):
        slot, s = ds.locate(i)
        assert 0 <= s <= 3, (slot, s)


def test_window_count_is_the_contiguous_run_arithmetic(tmp_path):
    ds = TicWindowDataset(corpus(tmp_path, "count", held_actions([1] * 8)), context_frames=8)
    assert len(ds) == 32 - 8


def test_tic_window_starts_needs_no_chain_id(tmp_path):
    """`chain_id` is -1 off the decision rows, so using it would throw away three tics in four."""
    d = corpus(tmp_path, "chain", held_actions([1, 2, 3, 4]))
    meta = np.load(os.path.join(d, "ep_00000_meta.npz"))
    assert (meta["chain_id"] == -1).sum() == 12
    assert len(tic_window_starts(meta, 4)) == 12


# ---------------------------------------------------------------------------------------
# tics_since_decision
# ---------------------------------------------------------------------------------------

def test_phase_is_the_position_inside_the_held_action_run():
    flag = np.zeros(12, dtype=bool)
    flag[::4] = True
    assert tics_since_decision(flag).tolist() == [0, 1, 2, 3] * 3


def test_phase_uses_the_last_bucket_when_the_decision_grid_has_a_hole():
    """An interrupted decision leaves no `is_decision` row, and phase 4 must not read as phase 3."""
    flag = np.zeros(10, dtype=bool)
    flag[0] = True                      # only one decision row: rows 4+ are off the grid
    ph = tics_since_decision(flag)
    assert ph.tolist() == [0, 1, 2, 3] + [PHASE_OFF_GRID] * 6
    assert PHASE_OFF_GRID == PHASE_BUCKETS - 1


def test_rows_before_the_first_decision_are_off_grid():
    flag = np.zeros(6, dtype=bool)
    flag[3] = True
    assert tics_since_decision(flag).tolist() == [PHASE_OFF_GRID] * 3 + [0, 1, 2]


def test_the_dataset_reports_the_phase_of_the_target(tmp_path):
    d = corpus(tmp_path, "phase", held_actions([1, 2, 3, 4]))
    ds = TicWindowDataset(d, context_frames=4, with_phase=True)
    for i in range(len(ds)):
        _, s = ds.locate(i)
        ctx, tgt, a, ph = ds[i]
        assert int(ph) == (s + 4) % STRIDE == ds.target_phase(i)
    assert sorted({int(ds[i][3]) for i in range(len(ds))}) == [0, 1, 2, 3]


def test_phase_is_absent_unless_asked_for(tmp_path):
    ds = TicWindowDataset(corpus(tmp_path, "nophase", held_actions([1, 2, 3, 4])), context_frames=4)
    assert len(ds[0]) == 3


def test_turning_phase_on_does_not_change_the_window_set(tmp_path):
    """The off-grid bucket exists so the comparison is not confounded by a different dataset."""
    d = corpus(tmp_path, "same", held_actions([1, 2, 3, 4]),
               decisions=np.array([True] + [False] * 15))
    a = TicWindowDataset(d, context_frames=4)
    b = TicWindowDataset(d, context_frames=4, with_phase=True)
    assert len(a) == len(b)
    assert [a.locate(i) for i in range(len(a))] == [b.locate(i) for i in range(len(b))]


# ---------------------------------------------------------------------------------------
# horizon windows, for equal-game-time scoring
# ---------------------------------------------------------------------------------------

def test_a_horizon_window_demands_that_many_contiguous_targets(tmp_path):
    d = corpus(tmp_path, "hor", held_actions([1, 2, 3, 4]))
    assert len(TicWindowDataset(d, context_frames=4, horizon=1)) == 12
    assert len(TicWindowDataset(d, context_frames=4, horizon=4)) == 9


def test_a_horizon_window_stops_at_a_death(tmp_path):
    deaths = np.array([0] * 10 + [1] * 6)
    d = corpus(tmp_path, "hordeath", held_actions([1, 2, 3, 4]), deaths=deaths)
    ds = TicWindowDataset(d, context_frames=4, horizon=4)
    for i in range(len(ds)):
        _, s = ds.locate(i)
        assert len(set(deaths[s:s + 8].tolist())) == 1


def test_with_horizon_returns_every_step_target_action_and_phase(tmp_path):
    acts = held_actions([1, 2, 3, 4, 5, 6])
    d = corpus(tmp_path, "horfull", acts)
    ds = TicWindowDataset(d, context_frames=4, horizon=4, with_horizon=True)
    for i in range(len(ds)):
        _, s = ds.locate(i)
        ctx, tgts, a, ph = ds[i]
        assert tgts.shape[0] == 4 and a.shape == (4,) and ph.shape == (4,)
        assert [frame_index_of(tgts[k]) for k in range(4)] == [s + 4 + k for k in range(4)]
        # step k predicts row s+4+k, conditioned on the action of row s+3+k
        assert a.tolist() == acts[s + 3:s + 7].tolist()
        assert ph.tolist() == [(s + 4 + k) % STRIDE for k in range(4)]


def test_horizon_one_with_horizon_matches_the_training_contract(tmp_path):
    d = corpus(tmp_path, "h1", held_actions([1, 2, 3, 4]))
    plain = TicWindowDataset(d, context_frames=4)
    hor = TicWindowDataset(d, context_frames=4, horizon=1, with_horizon=True)
    assert len(plain) == len(hor)
    for i in range(len(plain)):
        assert frame_index_of(plain[i][1]) == frame_index_of(hor[i][1][0])
        assert int(plain[i][2]) == int(hor[i][2][0])


# ---------------------------------------------------------------------------------------
# episode selection
# ---------------------------------------------------------------------------------------

def test_parse_episode_ids_is_half_open_like_a_python_slice():
    assert parse_episode_ids("0:4") == [0, 1, 2, 3]
    assert parse_episode_ids("7900:8000")[0] == 7900
    assert parse_episode_ids("7900:8000")[-1] == 7999
    assert len(parse_episode_ids("7900:8000")) == 100
    assert parse_episode_ids("3,1,2") == [1, 2, 3]
    assert parse_episode_ids("") == []


def test_parse_episode_ids_refuses_an_empty_range():
    with pytest.raises(ValueError, match="empty"):
        parse_episode_ids("10:10")


def test_assert_disjoint_accepts_adjacent_ranges_and_refuses_an_overlap():
    assert assert_disjoint(parse_episode_ids("0:2000"), parse_episode_ids("7900:8000"))
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(parse_episode_ids("0:2000"), parse_episode_ids("1999:2100"))


def test_limit_to_encoded_truncates_against_the_directory_prefix(tmp_path):
    d = str(tmp_path / "prefix")
    for ep in (0, 1, 2, 5):
        write_pertic_episode(d, ep, held_actions([1, 2]))
    assert limit_to_encoded(d, [0, 1, 2, 5]) == [0, 1, 2, 5]
    assert limit_to_encoded(d, [0, 1, 2, 5], max_episodes=3) == [0, 1, 2]
    assert limit_to_encoded(d, [2, 5], max_episodes=3) == [2]     # requested ids past the prefix drop out
    assert limit_to_encoded(d, [7], max_episodes=0) == []         # an id with no encoded episode


def test_episode_ids_select_only_those_episodes(tmp_path):
    d = str(tmp_path / "sel")
    for ep in range(4):
        write_pertic_episode(d, ep, held_actions([1, 2]))
    ds = TicWindowDataset(d, episode_ids=[1, 3], context_frames=4)
    assert sorted(e[0] for e in ds.episodes) == [1, 3]


# ---------------------------------------------------------------------------------------
# the stride-4 path is untouched
# ---------------------------------------------------------------------------------------

def test_the_tic_dataset_refuses_a_stride4_corpus(tmp_path):
    d = str(tmp_path / "s4")
    write_stride4_episode(d, 0, [1, 2, 3, 4, 5, 6, 7, 8])
    with pytest.raises(ValueError, match="is_decision"):
        TicWindowDataset(d, context_frames=4)


def test_the_stride4_dataset_still_yields_three_tensors_and_the_same_action(tmp_path):
    d = str(tmp_path / "s4b")
    write_stride4_episode(d, 0, [1, 2, 3, 4, 5, 6, 7, 8])
    ds = LatentWindowDataset(d, context_frames=4)
    assert len(ds) == 4
    ctx, tgt, a = ds[0]
    assert ctx.shape == (16, 32, 40) and tgt.shape == (4, 32, 40) and int(a) == 4


def test_the_tic_dataset_keeps_the_stride4_sample_shapes(tmp_path):
    """The model input contract is unchanged, which is why every backbone works unmodified."""
    d = corpus(tmp_path, "shapes", held_actions([1] * 10))
    ctx, tgt, a = TicWindowDataset(d, context_frames=32)[0]
    assert ctx.shape == (4 * 32, 32, 40) and tgt.shape == (4, 32, 40)
    assert a.dtype.__str__() == "torch.int64"


def test_the_module_exports_the_phase_bucket_contract():
    assert doom_data.PHASE_BUCKETS == 5 and doom_data.PHASE_OFF_GRID == 4


# ---------------------------------------------------------------------------------------
# the fixed dense-corpus split
# ---------------------------------------------------------------------------------------

def test_the_dense_split_ranges_are_the_decided_ones():
    s = doom_data.load_dense_split()
    assert s["segments"]["arenas"]["ranges"] == {"train": "0:6000", "val": "6000:7000", "test": "7000:8000"}
    assert s["segments"]["arenas_678"]["ranges"]["unseen"] == "0:60"
    assert s["next_tic_runs"]["train_ids"] == "0:2000"


def test_every_dense_range_is_map_balanced():
    """From the recorder's own id-to-map rule, not from reading the corpus."""
    s = doom_data.load_dense_split()
    for seg, spec in s["segments"].items():
        maps = spec["maps"]
        for name in spec["ranges"]:
            counts = doom_data.dense_map_counts(maps, doom_data.dense_ids(s, seg, name))
            assert sorted(counts) == sorted(maps), (seg, name, counts)
            assert len(set(counts.values())) == 1, f"{seg} {name} is not map-balanced: {counts}"


def test_the_next_tic_evaluation_ranges_are_map_balanced_too():
    s = doom_data.load_dense_split()
    nt = s["next_tic_runs"]
    for key, seg in (("train_ids", "arenas"), ("val_ids", "arenas"),
                     ("test_ids", "arenas"), ("unseen_ids", "arenas_678")):
        maps = s["segments"][seg]["maps"]
        counts = doom_data.dense_map_counts(maps, parse_episode_ids(nt[key]))
        assert len(set(counts.values())) == 1 and sorted(counts) == sorted(maps), (key, counts)
    assert set(doom_data.dense_map_counts(s["segments"]["arenas"]["maps"],
                                          parse_episode_ids(nt["val_ids"])).values()) == {25}
    assert set(doom_data.dense_map_counts(s["segments"]["arenas_678"]["maps"],
                                          parse_episode_ids(nt["unseen_ids"])).values()) == {20}


def test_the_id_to_map_rule_is_the_recorders():
    """`maps[e % len(maps)]`, so episode 0 is map 2 and episode 6000 is map 2 again."""
    assert doom_data.dense_episode_map((2, 3, 4, 5), 0) == 2
    assert [doom_data.dense_episode_map((2, 3, 4, 5), e) for e in range(6)] == [2, 3, 4, 5, 2, 3]
    assert doom_data.dense_episode_map((2, 3, 4, 5), 6000) == 2
    assert [doom_data.dense_episode_map((6, 7, 8), e) for e in range(4)] == [6, 7, 8, 6]


def test_the_split_ranges_do_not_overlap_each_other():
    s = doom_data.load_dense_split()
    tr, va, te = (doom_data.dense_ids(s, "arenas", n) for n in ("train", "val", "test"))
    assert_disjoint(tr, va)
    assert_disjoint(tr, te)
    assert_disjoint(va, te)
    assert len(tr) + len(va) + len(te) == s["segments"]["arenas"]["episodes"]


def test_the_training_ids_of_the_next_tic_runs_pass_the_guard():
    s = doom_data.load_dense_split()
    assert doom_data.check_dense_training_ids(s, "arenas", parse_episode_ids(s["next_tic_runs"]["train_ids"]))


def test_the_guard_refuses_a_training_range_that_reaches_into_validation():
    """The mistyped TRAIN_IDS that would put a headline number on training data."""
    s = doom_data.load_dense_split()
    with pytest.raises(ValueError, match="overlap"):
        doom_data.check_dense_training_ids(s, "arenas", parse_episode_ids("0:6500"))
    with pytest.raises(ValueError, match="overlap"):
        doom_data.check_dense_training_ids(s, "arenas", parse_episode_ids("0:8000"))
