"""Streaming the decoder tune off a corpus that does not fit in memory, on a wall-clock budget.

The tunes so far decode a 50k-frame sample into one uint8 array (11.5 GB on disk) and take two
epochs over it. The dense corpus is 1.8 TB, so the MSE-only ceiling run has to read frames as it
goes, and because the recorder writes `row_group_size=256` the only affordable unit of random
access is a whole row group: drawing single random rows would read about 7.7 MB per frame.

What these tests pin is that the streamed sample is exactly the frames of the row groups it
selected -- every one of them, once, in an order that is not the corpus order -- that the budget
flags stop the run and leave an hourly checkpoint that reloads, and that a run without any of the
new flags takes the same path it did before.

    python -m pytest paper/fixtures/test_decoder_stream.py -q
"""
import json
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

pytest.importorskip("pyarrow")
pytest.importorskip("torch")

import torch  # noqa: E402
from diffusers.models import AutoencoderKL  # noqa: E402

import finetune_decoder  # noqa: E402
from finetune_decoder import (PAD_TO, RowGroupFrames, sample_row_groups, save_vae,  # noqa: E402
                              stream_batches, to_tensor)
from test_decision_only import FORWARD, write_parquet  # noqa: E402
from test_decoder_save import tiny_vae  # noqa: E402

ROWS = 260          # two row groups per episode: 256 + 4
OFFSET = 1000       # episode e paints tics e*OFFSET.., so a frame names its episode as well as its tic


def corpus(dirpath, episodes, rows=ROWS):
    """A parquet corpus in the recorder's schema whose frames encode their own tic."""
    os.makedirs(dirpath, exist_ok=True)
    for e in episodes:
        write_parquet(os.path.join(dirpath, f"ep_{e:05d}.parquet"),
                      [(e * OFFSET + t, 0, FORWARD, 0) for t in range(rows)], 1)
    return dirpath


def every_tic(episodes, rows=ROWS):
    return sorted(e * OFFSET + t for e in episodes for t in range(rows))


def tic_of(frame_u8):
    """Recover the tic `frame_png` painted into a frame, so a stream can be checked frame by frame."""
    a = np.asarray(frame_u8)
    return int(a[0, 0, 0]) + 256 * int(a[0, 0, 1])


def drain(dataset, workers=0):
    """Every frame the dataset yields, as tics, through a DataLoader of the given width."""
    from torch.utils.data import DataLoader
    out = []
    for batch in DataLoader(dataset, batch_size=4, num_workers=workers):
        out += [tic_of(f) for f in batch]
    return out


# ---------------------------------------------------------------- the row-group sample

def test_sample_row_groups_takes_whole_groups_up_to_the_frame_target(tmp_path):
    d = corpus(str(tmp_path / "dense"), range(4))
    groups, frames, eps = sample_row_groups(d, 300, seed=0)

    assert frames >= 300                              # the target is met, not undershot
    assert frames < 300 + 256                         # and overshot by at most one group
    assert 2 <= len(groups) and eps <= len(groups)
    assert len(set(groups)) == len(groups)            # no group twice
    assert all(isinstance(g, int) and os.path.exists(p) for p, g in groups)


def test_sample_row_groups_is_seeded_and_covers_the_corpus_when_asked_for_all(tmp_path):
    d = corpus(str(tmp_path / "dense"), range(4))
    a, na, _ = sample_row_groups(d, 400, seed=7)
    b, nb, _ = sample_row_groups(d, 400, seed=7)
    c, _, _ = sample_row_groups(d, 400, seed=8)
    assert a == b and na == nb                         # same seed, same sample
    assert a != c                                      # and the seed is doing something

    everything, n, eps = sample_row_groups(d, 10**9, seed=0)
    assert n == 4 * ROWS and eps == 4                   # asking for more than exists takes it all
    assert len(everything) == 8 and len(set(everything)) == 8


def test_sample_row_groups_caps_the_episodes_it_opens(tmp_path):
    d = corpus(str(tmp_path / "dense"), range(6))
    _, _, eps = sample_row_groups(d, 10**9, seed=0, max_episodes=2)
    assert eps == 2


def test_sample_row_groups_names_an_empty_directory(tmp_path):
    empty = str(tmp_path / "nothing")
    os.makedirs(empty)
    with pytest.raises(SystemExit, match=re.escape(empty)):
        sample_row_groups(empty, 10, seed=0)


# ---------------------------------------------------------------- the stream itself

def test_stream_yields_every_frame_of_its_groups_exactly_once(tmp_path):
    """The property the sample size rests on: no frame is dropped and none is served twice."""
    d = corpus(str(tmp_path / "dense"), range(3))
    groups, frames, _ = sample_row_groups(d, 10**9, seed=1)
    got = drain(RowGroupFrames(groups, buffer=64, seed=1))

    assert len(got) == frames == 3 * ROWS
    assert sorted(got) == every_tic(range(3))


def test_stream_shuffles_across_row_groups(tmp_path):
    """A buffer of several groups' worth of frames is what breaks up 256 consecutive tics."""
    d = corpus(str(tmp_path / "dense"), range(3))
    groups, _, _ = sample_row_groups(d, 10**9, seed=1)
    got = drain(RowGroupFrames(groups, buffer=512, seed=1))

    assert got != sorted(got)
    assert len({t // OFFSET for t in got[:64]}) > 1    # one batch spans more than one episode


def test_stream_workers_partition_the_groups(tmp_path):
    """Two workers must split the sample, not each replay all of it."""
    d = corpus(str(tmp_path / "dense"), range(4))
    groups, frames, _ = sample_row_groups(d, 10**9, seed=2)
    got = drain(RowGroupFrames(groups, buffer=64, seed=2), workers=2)

    assert len(got) == frames
    assert sorted(got) == every_tic(range(4))


def test_stream_reshuffles_on_a_second_pass(tmp_path):
    d = corpus(str(tmp_path / "dense"), range(2))
    groups, _, _ = sample_row_groups(d, 10**9, seed=3)
    ds = RowGroupFrames(groups, buffer=512, seed=3)
    first = drain(ds)
    ds.epoch += 1
    assert drain(ds) != first


def test_stream_batches_delivers_exactly_the_budget_across_passes(tmp_path):
    """The budget is counted in batches, and a pass ending mid-budget starts another one."""
    d = corpus(str(tmp_path / "dense"), range(1))
    groups, frames, _ = sample_row_groups(d, 10**9, seed=4)
    ds = RowGroupFrames(groups, buffer=8, seed=4)
    per_pass = frames // 8

    got = list(stream_batches(ds, batch_size=8, workers=0, max_batches=per_pass + 3))
    assert len(got) == per_pass + 3
    assert all(tuple(b.shape) == (8, 240, 320, 3) and b.dtype == torch.uint8 for b in got)
    assert ds.epoch == 1                              # a second pass was started


# ---------------------------------------------------------------- collation and naming

def test_to_tensor_accepts_a_collated_batch_and_a_list_alike(tmp_path):
    frames = [np.full((240, 320, 3), v, np.uint8) for v in (0, 127, 255)]
    as_list = to_tensor(frames, "cpu")
    as_batch = to_tensor(torch.from_numpy(np.stack(frames)), "cpu")

    assert as_list.shape == (3, 3, PAD_TO, 320)
    assert torch.equal(as_list, as_batch)
    assert as_list[:, :, :240].min() == -1.0 and as_list[:, :, :240].max() >= 1.0


def test_save_vae_under_a_name_leaves_the_main_checkpoint_alone(tmp_path):
    out = str(tmp_path / "run")
    save_vae(tiny_vae(), out)
    main_cfg = open(os.path.join(out, "vae", "config.json")).read()

    path = save_vae(tiny_vae(latent_channels=8), out, name="vae_h1")
    assert path == os.path.join(out, "vae_h1")
    assert AutoencoderKL.from_pretrained(path).config.latent_channels == 8
    assert open(os.path.join(out, "vae", "config.json")).read() == main_cfg


# ---------------------------------------------------------------- the run

def tune(tmp_path, monkeypatch, **over):
    """Run `finetune_decoder.main` on a toy corpus with a two-level autoencoder."""
    monkeypatch.setattr(finetune_decoder, "build_vae", lambda *a, **kw: tiny_vae())
    train = corpus(str(tmp_path / "train"), [0, 1], rows=24)
    split = str(tmp_path / "split.json")
    json.dump({"train": [0], "val": [1]}, open(split, "w"))
    args = dict(in_dir=train, split=split, out_dir=str(tmp_path / "run"), train_frames=16,
                val_frames=4, stride=1, epochs=1, batch_size=2, accum=1, lr=1e-5, lpips_weight=0.0,
                report_lpips=False, val_every=0, channels_last=False, frame_cache="", vae_id="",
                vae_subfolder="", cache_dir=None, latent_channels=None, scaling_factor=None,
                shift_factor=None, device="cpu", stream_dir="", stream_frames=400000,
                stream_episodes=0, stream_buffer=4096, workers=0, max_steps=0, max_hours=0.0,
                ckpt_every_hours=0.0, seed=0)
    args.update(over)
    finetune_decoder.main(argparse_ns(args))
    return json.load(open(os.path.join(args["out_dir"], "metrics.json")))


def argparse_ns(d):
    import argparse
    return argparse.Namespace(**d)


def test_the_in_memory_path_is_unchanged_by_the_new_flags(tmp_path, monkeypatch):
    """No new flag given: one epoch of 16 frames at batch 2 is still eight updates."""
    m = tune(tmp_path, monkeypatch)
    assert m["steps"] == 8 and m["presentations"] == 16
    assert m["stream"] is None and m["stopped"] == ""
    assert m["train_frames"] == 16
    assert AutoencoderKL.from_pretrained(os.path.join(str(tmp_path / "run"), "vae")) is not None


def test_a_streamed_run_trains_off_the_stream_and_records_it(tmp_path, monkeypatch):
    dense = corpus(str(tmp_path / "dense"), range(2))
    m = tune(tmp_path, monkeypatch, stream_dir=dense, stream_frames=300, max_steps=5)

    assert m["steps"] == 5 and m["presentations"] == 10
    assert m["train_frames"] == 0                      # the 11.5 GB cached array is never built
    assert m["stream"]["frames"] >= 300 and m["stream"]["row_groups"] >= 2
    assert m["stream"]["dir"] == dense
    assert m["val_frames"] == 4                        # validation still comes from --split


def test_a_streamed_run_needs_a_step_budget(tmp_path, monkeypatch):
    dense = corpus(str(tmp_path / "dense"), range(1))
    with pytest.raises(SystemExit, match="max-steps"):
        tune(tmp_path, monkeypatch, stream_dir=dense)


def test_max_hours_stops_the_run_and_says_so(tmp_path, monkeypatch):
    dense = corpus(str(tmp_path / "dense"), range(1))
    m = tune(tmp_path, monkeypatch, stream_dir=dense, stream_frames=300, max_steps=10**6,
             max_hours=1e-9)

    assert 0 < m["steps"] < 10**6
    assert "max-hours" in m["stopped"]
    assert AutoencoderKL.from_pretrained(os.path.join(str(tmp_path / "run"), "vae")) is not None


def test_hourly_checkpoints_are_written_and_reload(tmp_path, monkeypatch):
    dense = corpus(str(tmp_path / "dense"), range(1))
    m = tune(tmp_path, monkeypatch, stream_dir=dense, stream_frames=300, max_steps=3,
             ckpt_every_hours=1e-9)

    hours = [h for h in os.listdir(str(tmp_path / "run")) if h.startswith("vae_h")]
    assert hours, os.listdir(str(tmp_path / "run"))
    for h in hours:
        AutoencoderKL.from_pretrained(os.path.join(str(tmp_path / "run"), h))
    assert [e for e in m["history"] if "hour_ckpt" in e]
