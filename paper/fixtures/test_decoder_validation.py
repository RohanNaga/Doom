"""Where the decoder tune's validation frames come from, and which saved decoder is the run's decoder.

Astra's review of 2026-09-26 (section 5): the MSE tune validated on the cached 2,000 frames of the
17-map corpus (finetune_decoder.py:373), whose maps 1 and 9 to 15 count as unseen for the next-tic
rows, so choosing among its hourly checkpoints on that curve would have been a choice made on unseen
maps. These tests pin that

  * `--val-dir` / `--val-ids` take the validation frames from named held-out episodes; for a dense
    segment the ids must lie inside its validation range (no training id, no sealed test id), a
    segment without one (the unseen arenas_678) is refused, and every named episode must exist;
  * those frames are cached under a name of their own that hashes the exact id list, so neither the
    17-map cache nor another id list can be read back in their place;
  * without `--val-dir` the old path, its frames and its cache name are unchanged;
  * the terminal checkpoint is the selected decoder and the best-on-validation checkpoint is recorded
    beside it, never swapped in for it.

They run without pyarrow: the parquet readers are replaced by fakes that paint each frame's episode
id into its pixels, so where a validation frame came from can be read off the frame itself.

    python -m pytest paper/fixtures/test_decoder_validation.py -q
"""
import argparse
import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import finetune_decoder  # noqa: E402
from finetune_decoder import checkpoint_selection, val_cache_tag, validation_episode_ids  # noqa: E402
from test_decoder_save import tiny_vae  # noqa: E402

VAL_IDS = list(range(6000, 6004))


def fake_frame(episode, row):
    """A 240x320 frame whose first pixel names its episode, with some texture for the loss to fit."""
    f = (np.random.RandomState(episode * 1000 + row).rand(240, 320, 3) * 255).astype(np.uint8)
    f[0, 0, 0], f[0, 0, 1] = episode % 256, episode // 256
    return f


def episode_in(frame):
    return int(frame[0, 0, 0]) + 256 * int(frame[0, 0, 1])


def episodes(dirpath, ids):
    """Empty `ep_XXXXX.parquet` files: the fakes read nothing, the existence check reads the names."""
    os.makedirs(dirpath, exist_ok=True)
    for e in ids:
        open(os.path.join(dirpath, f"ep_{e:05d}.parquet"), "wb").close()
    return dirpath


def fake_readers(monkeypatch):
    """Replace every parquet reader `finetune_decoder.main` touches; return the sample_frames call log."""
    calls = []

    def sample_frames(parquet_dir, episode_ids, n, stride, seed):
        calls.append((parquet_dir, [int(e) for e in episode_ids], n, stride, seed))
        eps = list(episode_ids)
        return [(os.path.join(parquet_dir, f"ep_{eps[i % len(eps)]:05d}.parquet"), (i // len(eps)) * stride)
                for i in range(n)]

    def stream_batches(dataset, batch_size, workers, max_batches):
        for b in range(max_batches):
            yield torch.from_numpy(np.stack([fake_frame(b % 4, i) for i in range(batch_size)]))

    monkeypatch.setattr(finetune_decoder, "build_vae", lambda *a, **kw: tiny_vae())
    monkeypatch.setattr(finetune_decoder, "sample_frames", sample_frames)
    monkeypatch.setattr(finetune_decoder, "load_frames",
                        lambda items: [fake_frame(finetune_decoder.episode_of(p), r) for p, r in items])
    monkeypatch.setattr(finetune_decoder, "training_episodes",
                        lambda d, ids, groups=None: {"dir": d, "segment": None, "ids": sorted(ids), "maps": []})
    monkeypatch.setattr(finetune_decoder, "sample_row_groups",
                        lambda d, n, seed, max_eps=0, ids=None: ([(os.path.join(d, "ep_00000.parquet"), 0)], 256, 1))
    monkeypatch.setattr(finetune_decoder, "stream_batches", stream_batches)
    return calls


def tune(tmp_path, monkeypatch, in_memory=True, **over):
    """`finetune_decoder.main` on fake frames: the cached training sample, or a stream, as asked."""
    calls = fake_readers(monkeypatch)
    argv = ["--out-dir", str(tmp_path / "run")]
    if in_memory:
        train = episodes(str(tmp_path / "train"), [0, 1])
        split = str(tmp_path / "split.json")
        json.dump({"train": [0], "val": [1]}, open(split, "w"))
        argv += ["--in-dir", train, "--split", split]
    args = vars(finetune_decoder.build_parser().parse_args(argv))
    args.update(train_frames=16, val_frames=4, stride=1, epochs=1, batch_size=2, device="cpu")
    args.update(over)
    finetune_decoder.main(argparse.Namespace(**args))
    return json.load(open(os.path.join(args["out_dir"], "metrics.json"))), calls


# ---------------------------------------------------------------------------------------
# the validation ids
# ---------------------------------------------------------------------------------------

def test_dense_validation_ids_must_lie_inside_the_segments_validation_range(tmp_path):
    arenas = str(tmp_path / "arenas")
    assert validation_episode_ids(arenas, "6000:6100") == list(range(6000, 6100))
    for bad in ("5990:6010",      # reaches back into training
                "6990:7010",      # reaches into the sealed test range
                "0:100", "7000:7100"):
        with pytest.raises(SystemExit, match="validation range"):
            validation_episode_ids(arenas, bad)


def test_validation_ids_are_named_never_implied(tmp_path):
    with pytest.raises(SystemExit, match="--val-ids"):
        validation_episode_ids(str(tmp_path / "arenas"), "")


def test_the_unseen_segment_is_never_a_validation_set(tmp_path):
    with pytest.raises(SystemExit, match="no validation range"):
        validation_episode_ids(str(tmp_path / "arenas_678"), "60:120")


def test_a_non_dense_directory_takes_any_ids_but_never_the_training_ones(tmp_path):
    d = str(tmp_path / "corpus")
    assert validation_episode_ids(d, "1:3") == [1, 2]
    assert validation_episode_ids(d, "1:3", str(tmp_path / "elsewhere"), [1, 2]) == [1, 2]
    with pytest.raises(SystemExit, match="overlap"):
        validation_episode_ids(d, "1:3", d, [2, 5])
    with pytest.raises(SystemExit, match="every file"):
        validation_episode_ids(d, "1:3", d, None)


def test_the_cache_name_changes_with_every_id(tmp_path):
    d = str(tmp_path / "arenas")
    a = val_cache_tag(d, [6000, 6001, 6003], 2000, 1)
    assert a.startswith("arenas_valids_6000-6003_3ep_") and a.endswith("_2000_s1")
    assert a != val_cache_tag(d, [6000, 6002, 6003], 2000, 1)        # same bounds, same count
    assert a != val_cache_tag(d, [6000, 6001, 6003], 2000, 4)
    assert val_cache_tag(d, [6003, 6000, 6001], 2000, 1) == a         # order does not matter


# ---------------------------------------------------------------------------------------
# a tune validated on held-out dense episodes
# ---------------------------------------------------------------------------------------

def test_a_tune_validates_on_exactly_the_named_episodes_and_caches_them_apart(tmp_path, monkeypatch):
    arenas = episodes(str(tmp_path / "arenas"), VAL_IDS)
    cache = str(tmp_path / "cache")
    m, calls = tune(tmp_path, monkeypatch, val_dir=arenas, val_ids="6000:6004", frame_cache=cache)

    assert (arenas, VAL_IDS, 4, 1, 1) in calls
    assert not [c for c in calls if c[0] == str(tmp_path / "train") and c[1] == [1]]   # the old val split
    tag = val_cache_tag(arenas, VAL_IDS, 4, 1)
    cached = np.load(os.path.join(cache, tag + ".npy"))
    assert {episode_in(f) for f in cached} == set(VAL_IDS)
    assert not os.path.exists(os.path.join(cache, "train_split.json_val_4_s1.npy"))

    prov = json.load(open(tmp_path / "run" / "provenance.json"))
    assert prov["validation_frames"] == {"dir": arenas, "segment": "arenas", "ids": VAL_IDS, "frames": 4,
                                         "stride": 1, "frame_cache": tag}
    assert m["provenance"]["validation_frames"] == prov["validation_frames"]
    assert m["val_frames"] == 4


def test_a_missing_validation_episode_is_refused(tmp_path, monkeypatch):
    arenas = episodes(str(tmp_path / "arenas"), VAL_IDS[:3])
    with pytest.raises(SystemExit, match="6003"):
        tune(tmp_path, monkeypatch, val_dir=arenas, val_ids="6000:6004")


def test_the_launchers_shape_needs_neither_the_17_map_corpus_nor_its_split(tmp_path, monkeypatch):
    """A stream and a validation directory: `--in-dir` and `--split` have nothing left to supply."""
    arenas = episodes(str(tmp_path / "arenas"), [0] + VAL_IDS)
    m, calls = tune(tmp_path, monkeypatch, in_memory=False, stream_dir=arenas, stream_ids="0:4", max_steps=3,
                    val_dir=arenas, val_ids="6000:6004")
    assert m["steps"] == 3 and m["train_frames"] == 0
    assert [c[:2] for c in calls] == [(arenas, VAL_IDS)]
    assert m["provenance"]["validation_frames"]["ids"] == VAL_IDS


def test_the_stream_and_the_validation_set_may_not_share_an_episode(tmp_path, monkeypatch):
    corpus = episodes(str(tmp_path / "corpus"), range(4))
    with pytest.raises(SystemExit, match="overlap"):
        tune(tmp_path, monkeypatch, in_memory=False, stream_dir=corpus, stream_ids="0:3", max_steps=2,
             val_dir=corpus, val_ids="2:4")


def test_without_a_validation_directory_the_frames_come_from_somewhere(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="--in-dir"):
        tune(tmp_path, monkeypatch, in_memory=False, stream_dir=str(tmp_path / "s"), stream_ids="0:2", max_steps=2)


def test_without_val_dir_the_old_split_and_cache_name_are_used(tmp_path, monkeypatch):
    cache = str(tmp_path / "cache")
    m, calls = tune(tmp_path, monkeypatch, frame_cache=cache)
    assert (str(tmp_path / "train"), [1], 4, 1, 1) in calls
    assert os.path.exists(os.path.join(cache, "train_split.json_val_4_s1.npy"))
    assert m["provenance"]["validation_frames"] == {"dir": str(tmp_path / "train"),
                                                    "split": str(tmp_path / "split.json"), "ids": [1]}


# ---------------------------------------------------------------------------------------
# the terminal checkpoint is the decoder; the best on validation is only recorded
# ---------------------------------------------------------------------------------------

def test_the_best_hourly_checkpoint_is_recorded_not_selected():
    history = [{"step": 10, "psnr": 30.0},
               {"step": 20, "hour_ckpt": 1, "psnr": 31.0, "lpips": 0.05},
               {"step": 40, "hour_ckpt": 2, "psnr": 32.5, "lpips": 0.04}]
    s = checkpoint_selection(history, 60, {"psnr": 32.0, "lpips": 0.045})
    assert s["rule"] == "terminal" and s["selected"] == "vae"
    assert s["terminal"] == {"dir": "vae", "step": 60, "psnr": 32.0, "lpips": 0.045}
    assert s["best_on_validation"] == {"dir": "vae_h2", "step": 40, "hour": 2, "psnr": 32.5, "lpips": 0.04}
    assert [c["dir"] for c in s["candidates"]] == ["vae", "vae_h1", "vae_h2"]


def test_a_tie_with_the_terminal_checkpoint_names_the_terminal_one():
    s = checkpoint_selection([{"step": 60, "hour_ckpt": 4, "psnr": 32.0}], 60, {"psnr": 32.0})
    assert s["best_on_validation"]["dir"] == "vae"


def test_a_tune_records_both_and_keeps_the_terminal_decoder_in_vae(tmp_path, monkeypatch):
    m, _ = tune(tmp_path, monkeypatch, ckpt_every_hours=1e-9)
    s = m["checkpoint_selection"]
    assert s["selected"] == "vae" and s["rule"] == "terminal"
    assert s["terminal"]["step"] == m["steps"] == 8
    assert len(s["candidates"]) == 1 + len([e for e in m["history"] if "hour_ckpt" in e]) > 1
    assert s["best_on_validation"]["psnr"] == max(c["psnr"] for c in s["candidates"])
    for c in s["candidates"]:
        assert os.path.isdir(tmp_path / "run" / c["dir"]), c["dir"]
