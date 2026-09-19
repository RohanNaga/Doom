"""`verify_dense.py` has to go red for the reasons a real dense segment goes wrong.

Each test breaks one property of a good segment and asserts the report names it: a short episode, a
reused episode id, two episodes recorded from the same seeds, a map that should not be there, the
wrong corpus id, a truncated schema, and two row semantics mixed into one directory. The passing
case pins the counts, because a report that says "ok" while counting nothing is worse than a failure.

    python -m pytest paper/fixtures/test_verify_dense.py -q
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
from verify_dense import SCHEMA, md5, verify  # noqa: E402

CORPUS = "arnold-train-dense-v1"


def frame_png(tic):
    from PIL import Image
    a = np.zeros((240, 320, 3), np.uint8)
    a[:, :, 0] = tic % 256
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def write_episode(d, episode_id, map_id, tics=40, corpus_id=CORPUS, seeds=None, stride=None,
                  drop_column=None, deaths_at=None):
    """One episode file in the recording schema, with the metadata record_arnold.py writes."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(d, exist_ok=True)
    deaths = np.zeros(tics, np.int16)
    if deaths_at:
        deaths[deaths_at:] = 1
    cols = {
        "episode_id": pa.array([episode_id] * tics, pa.int32()),
        "map_id": pa.array([map_id] * tics, pa.int8()),
        "tic": pa.array(np.arange(tics), pa.int32()),
        "action": pa.array(np.zeros(tics, np.int16), pa.int16()),
        "buttons": pa.array(["100000000"] * tics, pa.string()),
        "health": pa.array(np.full(tics, 100, np.int16), pa.int16()),
        "ammo": pa.array(np.full(tics, 50, np.int16), pa.int16()),
        "kills": pa.array(np.zeros(tics, np.int16), pa.int16()),
        "deaths": pa.array(deaths, pa.int16()),
        "frags": pa.array(np.zeros(tics, np.int16), pa.int16()),
        "pos_x": pa.array(np.zeros(tics, np.float32()), pa.float32()),
        "pos_y": pa.array(np.zeros(tics, np.float32()), pa.float32()),
        "angle": pa.array(np.zeros(tics, np.float32()), pa.float32()),
        "frame": pa.array([frame_png(i) for i in range(tics)], pa.binary()),
    }
    if drop_column:
        cols.pop(drop_column)
    t = pa.table(cols)
    prov = {"seed_scheme": "doomdit-episode-v1", "corpus_id": corpus_id, "episode_id": episode_id,
            "map_id": map_id, "seeds": seeds if seeds is not None else {"vizdoom": episode_id}}
    if stride:
        prov["stored_tic_stride"] = stride
    t = t.replace_schema_metadata({b"doomdit_episode": json.dumps(prov, sort_keys=True).encode()})
    pq.write_table(t, os.path.join(d, f"ep_{episode_id:05d}.parquet"), compression=None, row_group_size=256)
    return d


def good(tmp_path, name="seg"):
    d = str(tmp_path / name)
    for ep, m in enumerate([2, 3, 4, 5]):
        write_episode(d, ep, m, deaths_at=20)
    return d


def test_a_consistent_segment_passes_and_counts(tmp_path):
    r = verify(good(tmp_path), expect_maps=[2, 3, 4, 5], expect_corpus_id=CORPUS)
    assert r["ok"], r["problems"]
    assert r["episodes"] == 4 and r["tics"]["total"] == 160 and r["tics"]["mean"] == 40
    assert sorted(r["per_map"]) == ["2", "3", "4", "5"]
    assert r["per_map"]["2"] == dict(episodes=1, tics=40, lives=2, bytes=r["per_map"]["2"]["bytes"])
    assert r["corpus_ids"] == [CORPUS] and r["stored_tic_stride"] == [1]


def test_short_episode_is_reported(tmp_path):
    d = good(tmp_path)
    write_episode(d, 9, 2, tics=5)
    r = verify(d, min_tics=40)
    assert not r["ok"] and any("short of the" in p for p in r["problems"])


def test_reused_episode_id_and_repeated_seeds_are_reported(tmp_path):
    d = str(tmp_path / "dup")
    write_episode(d, 0, 2, seeds={"vizdoom": 7})
    write_episode(d, 1, 3, seeds={"vizdoom": 7})      # a different id but the same rollout
    r = verify(d)
    assert not r["ok"] and any("same seeds as" in p for p in r["problems"])


def test_unexpected_map_and_corpus_id_are_reported(tmp_path):
    d = good(tmp_path)
    write_episode(d, 8, 11, corpus_id="arnold-dense-arenas678-v1")
    r = verify(d, expect_maps=[2, 3, 4, 5], expect_corpus_id=CORPUS)
    assert not r["ok"]
    assert any("maps [2, 3, 4, 5, 11]" in p for p in r["problems"])
    assert any("corpus id" in p for p in r["problems"])


def test_truncated_schema_is_reported(tmp_path):
    d = str(tmp_path / "bad")
    write_episode(d, 0, 2, drop_column="angle")
    r = verify(d)
    assert not r["ok"] and any("not the recording schema" in p for p in r["problems"])


def test_mixed_row_semantics_in_one_directory_are_reported(tmp_path):
    """A decision-only file among per-tic files silently changes what a row means."""
    d = good(tmp_path)
    write_episode(d, 7, 2, stride=4)
    r = verify(d)
    assert not r["ok"] and any("mixed row semantics" in p for p in r["problems"])


def test_frame_sampling_decodes_real_pictures(tmp_path):
    r = verify(good(tmp_path), sample_frames=3)
    assert r["ok"], r["problems"]
    assert r["frame_check"]["shapes"] == {"(240, 320, 3)": 6}


def test_manifest_hashes_match_the_files(tmp_path):
    d = good(tmp_path)
    p = os.path.join(d, "ep_00000.parquet")
    assert md5(p) == md5(p) and len(md5(p)) == 32
    assert [c for c, _ in SCHEMA][:3] == ["episode_id", "map_id", "tic"]


def test_empty_directory_is_reported(tmp_path):
    r = verify(str(tmp_path / "nothing"))
    assert not r["ok"] and any("no ep_" in p for p in r["problems"])
