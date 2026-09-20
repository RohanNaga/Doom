"""`persistence_stats.py` measures the floor a corpus sets, so its arithmetic has to be checkable.

These build small parquet corpora whose answers are known by construction -- a still episode, an
episode that changes by a fixed amount, a two-map corpus, a walk over a known number of floor cells,
a per-tic file against one that holds every fourth tic -- and assert the reported numbers. Nothing
here needs a GPU, ViZDoom or torch; the autoencoder ceiling is a separate pass and is not covered.

    python -m pytest paper/fixtures/test_persistence_stats.py -q
"""
import io
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

pytest.importorskip("pyarrow")
pytest.importorskip("PIL")

from persistence_stats import (GAPS, NEAR_STATIC_DB, build_parser, do_stats,  # noqa: E402
                               episode_job, psnr_u8, row_semantics)

H, W = 240, 320


def png(arr):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return b.getvalue()


def write_episode(path, frames, *, map_id=1, tics=None, actions=None, pos=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = len(frames)
    tics = np.arange(n, dtype=np.int32) if tics is None else np.asarray(tics, dtype=np.int32)
    actions = np.zeros(n, dtype=np.int16) if actions is None else np.asarray(actions, dtype=np.int16)
    pos = np.zeros((n, 2), dtype=np.float32) if pos is None else np.asarray(pos, dtype=np.float32)
    pq.write_table(pa.table({
        "episode_id": pa.array([0] * n, pa.int32()), "map_id": pa.array([map_id] * n, pa.int8()),
        "tic": pa.array(tics, pa.int32()), "action": pa.array(actions, pa.int16()),
        "pos_x": pa.array(pos[:, 0], pa.float32()), "pos_y": pa.array(pos[:, 1], pa.float32()),
        "frame": pa.array([png(f) for f in frames], pa.binary())}), path)


def still(n, value=100):
    return [np.full((H, W, 3), value, np.uint8) for _ in range(n)]


def run(tmp_path, **kw):
    out = str(tmp_path / "stats.json")
    args = build_parser().parse_args(["--dir", str(tmp_path), "--out", out, "--workers", "1",
                                      "--episodes", "50", "--anchors", "8"]
                                     + [x for k, v in kw.items() for x in (k, str(v))])
    do_stats(args)
    return json.load(open(out))


def test_a_perfectly_still_corpus_reports_the_ceiling_of_the_psnr_formula(tmp_path):
    ceiling = psnr_u8(np.zeros(1), np.zeros(1))      # the 1e-10 MSE clamp, about 148 dB
    write_episode(str(tmp_path / "ep_00000.parquet"), still(40))
    r = run(tmp_path)
    for g in GAPS:
        assert r["all"]["persistence_psnr"][str(g)] == pytest.approx(ceiling, abs=0.01)
    assert r["all"]["near_static_fraction"] == 1.0


def test_a_known_constant_difference_gives_the_hand_computed_psnr(tmp_path):
    # every frame differs from its neighbour by exactly 4 grey levels, so the MSE is 16 at every gap
    frames = [np.full((H, W, 3), (100 + 4 * i) % 256, np.uint8) for i in range(40)]
    write_episode(str(tmp_path / "ep_00000.parquet"), frames)
    r = run(tmp_path)
    assert r["all"]["persistence_psnr"]["1"] == pytest.approx(psnr_u8(np.zeros(1), np.full(1, 4)), abs=0.01)
    # doubling the gap doubles the difference, which costs exactly 6.02 dB
    assert (r["all"]["persistence_psnr"]["1"] - r["all"]["persistence_psnr"]["2"]) == pytest.approx(6.02, abs=0.05)


def test_near_static_fraction_counts_only_pairs_above_the_threshold(tmp_path):
    """Half the episodes still, half pure noise: the fraction is a count, not an average of dB."""
    noisy = list(np.random.RandomState(0).randint(0, 255, (40, H, W, 3), dtype=np.uint8))
    write_episode(str(tmp_path / "ep_00000.parquet"), still(40), map_id=1)
    write_episode(str(tmp_path / "ep_00001.parquet"), noisy, map_id=2)
    args = build_parser().parse_args(["--dir", str(tmp_path), "--out", str(tmp_path / "s.json"),
                                      "--workers", "1", "--anchors", "8", "--per-map"])
    do_stats(args)
    r = json.load(open(str(tmp_path / "s.json")))
    assert r["all"]["near_static_fraction"] == pytest.approx(0.5, abs=0.01)
    assert r["per_map"]["1"]["near_static_fraction"] == 1.0
    assert r["per_map"]["2"]["near_static_fraction"] == 0.0
    assert r["per_map"]["2"]["persistence_psnr"]["1"] < NEAR_STATIC_DB


def test_per_map_splits_by_map_id_and_the_totals_agree(tmp_path):
    write_episode(str(tmp_path / "ep_00000.parquet"), still(40), map_id=2)
    write_episode(str(tmp_path / "ep_00001.parquet"), still(40), map_id=5)
    write_episode(str(tmp_path / "ep_00002.parquet"), still(40), map_id=5)
    args = build_parser().parse_args(["--dir", str(tmp_path), "--out", str(tmp_path / "m.json"),
                                      "--workers", "1", "--anchors", "8", "--per-map"])
    do_stats(args)
    r = json.load(open(str(tmp_path / "m.json")))
    assert sorted(r["per_map"]) == ["2", "5"]
    assert r["per_map"]["2"]["episodes"] == 1 and r["per_map"]["5"]["episodes"] == 2
    assert sum(m["pairs"] for m in r["per_map"].values()) == r["all"]["pairs"]


def test_explored_cells_counts_distinct_64_unit_floor_cells(tmp_path):
    # a straight walk of 10 cells: 40 rows stepping 16 units, so every fourth row enters a new cell
    pos = np.stack([np.arange(40) * 16.0, np.zeros(40)], axis=1)
    write_episode(str(tmp_path / "ep_00000.parquet"), still(40), pos=pos)
    r = run(tmp_path)
    assert r["all"]["explored_cells_union"] == 10
    assert r["all"]["explored_cells_per_episode_mean"] == 10.0


def test_row_semantics_sees_a_per_tic_file_with_a_four_tic_decision_grid():
    tic = np.arange(40)
    action = np.repeat(np.arange(10), 4)
    s = row_semantics(tic, action)
    assert s["tic_step"][1] == 39 and s["tic_step"][4] == 0
    assert s["run_lengths"][4] == 10
    assert s["run_start_phase"].tolist() == [10, 0, 0, 0]


def test_row_semantics_sees_a_decision_only_file_as_four_tic_steps():
    tic = np.arange(0, 40, 4)
    s = row_semantics(tic, np.arange(10))
    assert s["tic_step"][4] == 9 and s["tic_step"][1] == 0
    assert s["run_lengths"][1] == 10


def test_row_semantics_reports_a_drifting_decision_phase():
    """Ours drifts because deaths and the 40-tic anti-stuck override break the grid."""
    tic = np.arange(40)
    action = np.repeat(np.arange(8), [4, 4, 3, 4, 4, 4, 4, 13])
    s = row_semantics(tic, action)
    assert s["run_start_phase"][0] < 8 and sum(s["run_start_phase"]) == 8


def test_a_short_episode_is_reported_as_an_error_not_silently_dropped(tmp_path):
    write_episode(str(tmp_path / "ep_00000.parquet"), still(40))
    write_episode(str(tmp_path / "ep_00001.parquet"), still(4))
    r = run(tmp_path)
    assert r["n_errors"] == 1 and "only 4 rows" in r["errors"][0]["error"]
    assert r["all"]["episodes"] == 1


def test_a_wrong_sized_frame_is_reported_rather_than_measured(tmp_path):
    write_episode(str(tmp_path / "ep_00000.parquet"), [np.zeros((120, 160, 3), np.uint8)] * 40)
    r = episode_job((str(tmp_path / "ep_00000.parquet"), 4, 0, False))
    assert "frame shape" in r["error"]


def test_cached_frames_match_the_anchors_that_were_measured(tmp_path):
    write_episode(str(tmp_path / "ep_00000.parquet"), still(40, value=77))
    cache = str(tmp_path / "frames.npy")
    args = build_parser().parse_args(["--dir", str(tmp_path), "--out", str(tmp_path / "s.json"),
                                      "--workers", "1", "--anchors", "6", "--cache-frames", cache])
    do_stats(args)
    arr = np.load(cache)
    assert arr.shape == (6, H, W, 3) and np.all(arr == 77)


def test_frames_cap_truncates_the_cache(tmp_path):
    write_episode(str(tmp_path / "ep_00000.parquet"), still(40))
    write_episode(str(tmp_path / "ep_00001.parquet"), still(40))
    cache = str(tmp_path / "frames.npy")
    args = build_parser().parse_args(["--dir", str(tmp_path), "--out", str(tmp_path / "s.json"),
                                      "--workers", "1", "--anchors", "6", "--cache-frames", cache,
                                      "--frames-cap", "7"])
    do_stats(args)
    assert np.load(cache).shape[0] == 7


def test_an_empty_directory_is_refused(tmp_path):
    args = build_parser().parse_args(["--dir", str(tmp_path), "--out", str(tmp_path / "s.json")])
    with pytest.raises(SystemExit):
        do_stats(args)
