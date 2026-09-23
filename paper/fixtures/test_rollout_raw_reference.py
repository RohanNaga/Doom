"""A rollout is scored against the game's frames, every valid IDM window counts, clips stream.

Three defects:

  * every rollout metric compared DECODED prediction against DECODED ground-truth latent, and the
    "real" FVD clips were decoded ground truth. Each autoencoder defines a different decoded target
    distribution, and a blurry decoder improves its own decoded score while getting no closer to the
    game's pixels, so two rows scored that way are not comparable.
  * the IDM fixed its run length to the FIRST accepted rollout's and skipped every rollout of any
    other length: the result depended on which rollout came first, most could be discarded, and zero
    usable runs aborted scoring before the drift curve and the clips were written.
  * 256 rollouts x 256 frames x 3x240x320 uint8 is 15.10 GB per stack, held in a Python list and
    then `np.stack`ed again while saving: a peak over 60 GB.

    python -m pytest paper/fixtures/test_rollout_raw_reference.py -q
"""
import io
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

pytest.importorskip("pyarrow")

import rollout_eval  # noqa: E402
# the lpips stand-in and the tiny autoencoder are the ones the IDM gate already uses: where the real
# package is installed it runs, and where it is not the stub with the same call shape is inserted, so
# these tests exercise the scorer end to end instead of skipping
from test_idm_reencode import lpips_available, tiny_sd_encoder  # noqa: E402,F401

LAT_H, LAT_W = 32, 40
# a seed long enough to hold two decision tics four apart, which is what the IDM's window needs
N, H, L = 3, 12, 12
DEC = [3, 7, 11]      # positions on the four-tic decision grid, in the seed and in the horizon


def _write_recording(path, tics, fill):
    """One recording whose frame at tic t is a flat colour derived from t."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    blobs = []
    for t in tics:
        buf = io.BytesIO()
        Image.new("RGB", (320, 240), (int(fill(t)) % 256, 0, 0)).save(buf, format="PNG")
        blobs.append(buf.getvalue())
    pq.write_table(pa.table({"tic": pa.array([int(t) for t in tics], pa.int32()),
                             "frame": pa.array(blobs, pa.binary())}), path, compression=None)


@pytest.fixture
def rollouts(tmp_path):
    """A rollout npz and a matching raw recording directory, both 4-channel."""
    rng = np.random.RandomState(0)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    episodes = np.arange(N, dtype=np.int64)
    starts = np.full(N, 2, dtype=np.int64)
    seed_tic = np.stack([np.arange(2, 2 + L, dtype=np.int64) for _ in range(N)])
    tic = np.stack([np.arange(2 + L, 2 + L + H, dtype=np.int64) for _ in range(N)])
    for ep in episodes:
        _write_recording(os.path.join(raw_dir, f"ep_{ep:05d}.parquet"), range(40),
                         lambda t, e=int(ep): t * 3 + e)
    path = tmp_path / "rollouts.npz"
    np.savez(path,
             pred=rng.randn(N, H, 4, LAT_H, LAT_W).astype(np.float16) * 0.3,
             gt=rng.randn(N, H, 4, LAT_H, LAT_W).astype(np.float16) * 0.3,
             seed=rng.randn(N, L, 4, LAT_H, LAT_W).astype(np.float16) * 0.3,
             actions=rng.randint(0, 3, (N, H)).astype(np.int64),
             episode=episodes, map=np.full(N, 3, np.int64), start=starts,
             tic=tic, seed_tic=seed_tic,
             config=json.dumps({"tic_stride": 1, "resolved_latent_channels": 4}))
    vae_dir = tmp_path / "vae"
    tiny_sd_encoder().save_pretrained(str(vae_dir))
    return str(path), str(vae_dir), str(raw_dir)


def _score(rollouts, out_dir, **kw):
    path, vae_dir, raw_dir = rollouts
    argv = ["--score", "--rollouts", path, "--vae-path", vae_dir, "--out-dir", str(out_dir),
            "--decode-batch", "2"]
    for k, v in kw.items():
        argv += [k] if v is True else [k, str(v)]
    rollout_eval.do_score(rollout_eval.build_parser().parse_args(argv))
    with open(os.path.join(str(out_dir), "drift.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------
# the raw reference
# ---------------------------------------------------------------------------------------

def test_without_the_recordings_only_the_decoded_keys_are_reported(rollouts, tmp_path):
    out = _score(rollouts, tmp_path / "dec", **{"--save-clips": 0})
    assert out["reference"] == "decoded_gt"
    assert "psnr" in out and "psnr_raw" not in out
    assert "DECODED" in out["decoded_note"]


def test_with_the_recordings_the_raw_keys_appear_alongside(rollouts, tmp_path):
    out = _score(rollouts, tmp_path / "raw", **{"--save-clips": 0, "--parquet-dir": rollouts[2]})
    assert out["reference"] == "raw"
    for k in ("psnr_raw", "lpips_raw", "vae_psnr_raw"):
        assert len(out[k]) == H, k
        assert all(np.isfinite(out[k])), k
    assert out["raw_parquet_dir"] == rollouts[2]
    # the decoded keys are still there, under their original names, for continuity
    assert len(out["psnr"]) == H and len(out["copy_seed_psnr"]) == H


def test_both_raw_persistence_floors_are_reported(rollouts, tmp_path):
    """Copy-seed holds the last SEED frame for the whole horizon; copy-last holds the previous frame
    one step at a time. Neither decodes anything, so neither carries the decoder's own error."""
    out = _score(rollouts, tmp_path / "floors", **{"--save-clips": 0, "--parquet-dir": rollouts[2]})
    seed_floor = out["persist_seed_psnr_raw"]
    last_floor = out["persist_last_psnr_raw"]
    assert len(seed_floor) == len(last_floor) == H
    # the fixture's frames change every tic, so copy-last (one tic back) beats copy-seed (which
    # falls further behind with every step) at every horizon past the first
    assert last_floor[-1] > seed_floor[-1]
    assert all(np.isfinite(seed_floor)) and all(np.isfinite(last_floor))
    assert "persist_seed_lpips_raw" in out and "persist_last_lpips_raw" in out


def test_the_raw_scores_are_reported_at_the_named_horizons(rollouts, tmp_path):
    out = _score(rollouts, tmp_path / "at", **{"--save-clips": 0, "--parquet-dir": rollouts[2],
                                               "--score-at": "1,4"})
    for hh in (1, 4):
        assert abs(out[f"psnr_raw@{hh}"] - out["psnr_raw"][hh - 1]) < 1e-9
        assert f"persist_last_psnr_raw@{hh}" in out


def test_the_scores_are_appended_to_the_runs_wandb_eval_run(rollouts, tmp_path, monkeypatch):
    """The scoring pass is the one with numbers, so it logs, and the step comes from the checkpoint's filename."""
    from wandb_stub import inits, logged, stub_wandb
    wb = stub_wandb()
    monkeypatch.setitem(sys.modules, "wandb", wb)
    out = _score(rollouts, tmp_path / "wb", **{"--save-clips": 0, "--parquet-dir": rollouts[2], "--score-at": "4",
                                               "--ckpt": "results/040/snap_0010000.pt",
                                               "--wandb-run": "040-unet-nexttic"})
    (kw,) = inits(wb)
    assert kw["id"] == "040-unet-nexttic-eval" and kw["group"] == "040-unet-nexttic"
    (row,) = logged(wb)
    assert row["step"] == 10000
    assert row["eval/rollout/psnr_raw@4"] == out["psnr_raw@4"]
    assert row["eval/rollout/persist_last_psnr_raw@4"] == out["persist_last_psnr_raw@4"]


def test_an_old_rollout_without_tics_cannot_be_scored_against_raw_frames(rollouts, tmp_path):
    path, vae_dir, raw_dir = rollouts
    d = dict(np.load(path))
    d.pop("tic")
    np.savez(path, **d)
    with pytest.raises(SystemExit, match="carries no `tic` array"):
        _score((path, vae_dir, raw_dir), tmp_path / "old",
               **{"--save-clips": 0, "--parquet-dir": raw_dir})


def test_the_rollout_writer_stores_the_recorded_tics():
    src = open(os.path.join(REPO, "rollout_eval.py")).read()
    assert 'extra["tic"]' in src and 'extra["seed_tic"]' in src


# ---------------------------------------------------------------------------------------
# the raw FVD reference clips
# ---------------------------------------------------------------------------------------

def test_the_raw_clip_set_holds_the_game_frames(rollouts, tmp_path):
    out_dir = tmp_path / "clips"
    _score(rollouts, out_dir, **{"--save-clips": 2, "--parquet-dir": rollouts[2]})
    with np.load(os.path.join(out_dir, "clips_u8_raw.npz")) as z:
        raw_gt, raw_pred = z["gt"], z["pred"]
    with np.load(os.path.join(out_dir, "clips_u8.npz")) as z:
        dec_gt = z["gt"]
    assert raw_gt.shape == dec_gt.shape == (2, H, 3, 240, 320)
    assert not np.array_equal(raw_gt, dec_gt), "the raw reference is the decoded one again"
    # the raw clip's first frame is the recorded frame at that rollout's first target tic
    assert int(raw_gt[0, 0, 0, 0, 0]) == (2 + L) * 3 % 256
    assert np.array_equal(raw_pred, np.load(os.path.join(out_dir, "clips_u8.npz"))["pred"])


def test_both_clip_sets_get_a_stride_four_companion(rollouts, tmp_path):
    out_dir = tmp_path / "stride"
    _score(rollouts, out_dir, **{"--save-clips": 1, "--parquet-dir": rollouts[2]})
    for base in ("clips_u8", "clips_u8_raw"):
        with np.load(os.path.join(out_dir, f"{base}_stride4.npz")) as z:
            assert z["pred"].shape[1] == len(range(3, H, 4)), base


def test_the_launcher_scores_the_raw_clips_first(tmp_path):
    import subprocess
    e = {**os.environ, "DRY": "1", "DOOM_ROOT": str(tmp_path)}
    src = open(os.path.join(REPO, "scripts", "spiderman", "after_nexttic.sh")).read()
    line = [ln for ln in src.splitlines() if ln.strip().startswith("for CLIPS in")]
    assert line and line[0].index("clips_u8_raw.npz") < line[0].index(" clips_u8.npz"), line
    assert subprocess.run(["bash", "-n", os.path.join(REPO, "scripts", "spiderman",
                                                      "after_nexttic.sh")], env=e).returncode == 0


# ---------------------------------------------------------------------------------------
# clips are streamed, not accumulated
# ---------------------------------------------------------------------------------------

def test_the_clip_store_never_holds_more_than_one_rollout(tmp_path):
    """A row-count proxy for the 60 GB peak: the store's resident objects are memmaps on disk, so
    the process's own allocation does not scale with the number of rollouts."""
    store = rollout_eval.ClipStore(str(tmp_path), n=4, frames=3, shape=(3, 4, 5))
    for n in range(4):
        store.put("pred", n, 0, np.full((3, 3, 4, 5), n + 1, dtype=np.uint8))
        store.put("gt", n, 0, np.zeros((3, 3, 4, 5), dtype=np.uint8))
    assert all(isinstance(a, np.memmap) for a in store.arrays.values())
    written = store.save(str(tmp_path), "clips_u8")
    with np.load(written[0]) as z:
        assert z["pred"].shape == (4, 3, 3, 4, 5)
        assert int(z["pred"][2, 0, 0, 0, 0]) == 3
    store.close()
    assert not os.path.isdir(os.path.join(str(tmp_path), ".clips_tmp")), "the staging files survived"


def test_the_stored_horizon_can_be_capped(tmp_path):
    store = rollout_eval.ClipStore(str(tmp_path), n=2, frames=2, shape=(3, 4, 5))
    store.put("pred", 0, 0, np.ones((6, 3, 4, 5), dtype=np.uint8))
    store.put("gt", 0, 0, np.ones((6, 3, 4, 5), dtype=np.uint8))
    store.put("pred", 0, 4, np.ones((2, 3, 4, 5), dtype=np.uint8))       # past the cap: dropped
    written = store.save(str(tmp_path), "capped")
    with np.load(written[0]) as z:
        assert z["pred"].shape[1] == 2
    store.close()


def test_clip_frames_caps_what_the_scorer_writes(rollouts, tmp_path):
    out_dir = tmp_path / "capped"
    _score(rollouts, out_dir, **{"--save-clips": 2, "--clip-frames": 3, "--parquet-dir": rollouts[2]})
    with np.load(os.path.join(out_dir, "clips_u8_raw.npz")) as z:
        assert z["pred"].shape[1] == 3
    assert not os.path.isdir(os.path.join(str(out_dir), ".clips_tmp"))


def test_no_clip_lists_are_accumulated_any_more():
    src = open(os.path.join(REPO, "rollout_eval.py")).read()
    assert "clips_pred, clips_gt = [], []" not in src
    assert "np.stack(clips_pred)" not in src, "the stacking peak is back"


# ---------------------------------------------------------------------------------------
# the IDM aggregates every valid window
# ---------------------------------------------------------------------------------------

@pytest.fixture
def idm_path(tmp_path):
    from train_idm import IDM
    torch.manual_seed(4)
    idm = IDM(num_actions=3, width=8, window=3, depth=1, heads=2)
    p = tmp_path / "idm.pt"
    torch.save({"model": idm.state_dict(), "num_actions": 3, "width": 8, "window": 3, "depth": 1,
                "act2mov": {0: 0, 1: 1, 2: 1}, "classes": ["none", "move"]}, str(p))
    return str(p)


def _with_decisions(path, masks, chains=None):
    """Add per-tic decision/chain arrays to a rollout npz, as `--rollout` writes them."""
    d = dict(np.load(path))
    n, h = d["pred"].shape[:2]
    lseed = d["seed"].shape[1]
    d["decision"] = np.asarray(masks, dtype=bool)
    d["chain"] = (np.zeros((n, h), dtype=np.int64) if chains is None else np.asarray(chains))
    # a seed whose own decision tics sit on the four-tic grid ending four tics before the first
    # judged frame, which is what `judged` requires
    sd = np.zeros((n, lseed), dtype=bool)
    sd[:, DEC] = True
    d["seed_decision"] = sd
    d["seed_chain"] = np.zeros((n, lseed), dtype=np.int64)
    np.savez(path, **d)


def test_rollouts_of_different_run_lengths_all_count(rollouts, idm_path, tmp_path):
    """The defect: `steps` was fixed to the first accepted rollout's run length, so a rollout with
    a longer verified run was discarded entirely."""
    path, vae_dir, raw_dir = rollouts
    # rollout 0 has two judged decisions (frames 3 and 7 of a 12-frame horizon would be needed for
    # more), rollout 1 has one, rollout 2 has none
    d = dict(np.load(path))
    n, h = d["pred"].shape[:2]
    masks = np.zeros((n, h), dtype=bool)
    masks[0, DEC] = True               # three judged decisions, exactly four tics apart
    masks[1, DEC[:2]] = True           # two
    _with_decisions(path, masks)
    out = _score((path, vae_dir, raw_dir), tmp_path / "idm",
                 **{"--save-clips": 0, "--idm": idm_path})
    assert out["idm_rollouts_scored"] == 2 and out["idm_rollouts_skipped"] == 1
    assert out["idm_windows_scored"] == 5, out["idm_denominator"]
    assert out["idm_denominator"][:3] == [2, 2, 1]
    assert out["idm_run_length_min"] == 2 and out["idm_run_length_max"] == 3


def test_every_position_reports_its_own_denominator(rollouts, idm_path, tmp_path):
    path, vae_dir, raw_dir = rollouts
    d = dict(np.load(path))
    n, h = d["pred"].shape[:2]
    masks = np.zeros((n, h), dtype=bool)
    masks[:, DEC[:2]] = True
    _with_decisions(path, masks)
    out = _score((path, vae_dir, raw_dir), tmp_path / "den",
                 **{"--save-clips": 0, "--idm": idm_path})
    assert out["idm_denominator"][:2] == [3, 3]
    assert out["idm_denominator"][2] == 0
    assert out["idm_top1"][2] is None, "a position nothing scored must not report a rate"
    assert all(0.0 <= v <= 1.0 for v in out["idm_top1"][:2])
    assert 0.0 <= out["idm_top1_mean"] <= 1.0


def test_no_usable_window_still_writes_the_drift_curve(rollouts, idm_path, tmp_path):
    """Zero usable runs used to abort before drift.json and the clips were written, throwing away
    every metric that had already been computed."""
    path, vae_dir, raw_dir = rollouts
    d = dict(np.load(path))
    n, h = d["pred"].shape[:2]
    _with_decisions(path, np.zeros((n, h), dtype=bool))
    out_dir = tmp_path / "noidm"
    out = _score((path, vae_dir, raw_dir), out_dir,
                 **{"--save-clips": 1, "--idm": idm_path, "--parquet-dir": raw_dir})
    assert out["idm_rollouts_scored"] == 0 and "idm_error" in out
    assert "idm_top1" not in out
    assert len(out["psnr_raw"]) == H, "the raw drift curve was thrown away"
    assert os.path.isfile(os.path.join(str(out_dir), "clips_u8_raw.npz"))


def test_a_window_across_a_chain_boundary_is_still_dropped(rollouts, idm_path, tmp_path):
    path, vae_dir, raw_dir = rollouts
    d = dict(np.load(path))
    n, h = d["pred"].shape[:2]
    masks = np.zeros((n, h), dtype=bool)
    masks[:, DEC] = True
    chains = np.zeros((n, h), dtype=np.int64)
    chains[0, DEC[2]] = 1               # a new chain: rollout 0's last step is no longer verified
    _with_decisions(path, masks, chains)
    out = _score((path, vae_dir, raw_dir), tmp_path / "chain",
                 **{"--save-clips": 0, "--idm": idm_path})
    assert out["idm_denominator"][:3] == [3, 3, 2], "the cross-chain step was accepted"
