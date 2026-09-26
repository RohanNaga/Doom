"""CPU gates for the per-window columns the fresh evaluation set is scored with once (Sep 26 2026).

The cohesion decision (`.claude/analyses/evaluation-cohesion-decision-2026-09-26.md`, item 3) asks one
scoring pass to leave every per-window quantity a table needs, so no table is rerun:

* every column the frozen results were read from keeps its name and its value, byte for byte, against
  the scorer that produced them (pinned at `PINNED_SCORER`);
* `copy_lpips_dec` beside `lpips_dec`, pixel MSE beside every PSNR, and `scene_*` on rows 0 to 207;
* duplicate-window flags (raw and latent), counted in the summary, with a `*_nodup` mean beside every mean;
* `--decoder [name=]path`, repeatable, decoding ONE set of predictions with each decoder under a suffix;
* `--save-latents`, the predicted latents in fp16 with the window identities;
* the same columns at four tics as at one.

Raw frames come from an in-memory stand-in for `eval_tf.RawFrames`, so these run without pyarrow.

    python -m pytest paper/fixtures/test_eval_tf_columns.py -q
"""
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

os.environ.setdefault("ACCELERATE_USE_CPU", "1")

from pertic_fixtures import held_actions, write_pertic_episode  # noqa: E402
import test_nexttic_eval as nexttic  # noqa: E402

import backbones  # noqa: E402
import eval_tf  # noqa: E402

# the next-tic eval tests' fixtures and tiny models, shared rather than copied: the autouse LPIPS
# stand-in (where the package is absent) and the 16k-parameter PixArt served in place of the real one
CTX, _tiny_model, _tiny_vae = nexttic.CTX, nexttic._tiny_model, nexttic._tiny_vae
lpips_available, tiny_hub = nexttic.lpips_available, nexttic.tiny_hub

# the last commit whose eval_tf.py produced the frozen *_h1 / *_h4 results
PINNED_SCORER = "60d99c4a39d7647288ecaa9fd3318fa56663cc13"
T = 24                                  # tics per synthetic episode
VAL_EP = 1


# ---------------------------------------------------------------------------------------
# fixtures: a per-tic corpus, raw frames in memory, and the scorer's command line
# ---------------------------------------------------------------------------------------

def _frames(seed=0, dup_tics=()):
    """{(episode, tic): (240, 320, 3) uint8}; each tic in `dup_tics` repeats the frame one tic earlier."""
    rng = np.random.RandomState(seed)
    out = {}
    for ep in (0, VAL_EP):
        for t in range(T):
            out[(ep, t)] = rng.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            if ep == VAL_EP and t in dup_tics:
                out[(ep, t)] = out[(ep, t - 1)].copy()
    return out


def _fake_raw(frames):
    class FakeRaw:
        """`eval_tf.RawFrames` served from a dict, with the same refusal for a tic it does not hold."""

        def __init__(self, parquet_dir):
            self.dir = parquet_dir

        def get(self, episode_id, tic):
            if (int(episode_id), int(tic)) not in frames:
                raise KeyError(f"episode {episode_id} has no frame at tic {tic}")
            return frames[(int(episode_id), int(tic))]
    return FakeRaw


def _corpus(root, latent_dup_rows=()):
    """Two per-tic episodes (latent of row t filled with t); each row in `latent_dup_rows` of the val
    episode repeats the latent of the row before it, so copy-last is exact there."""
    lat_dir = os.path.join(str(root), "latents")
    for ep in (0, VAL_EP):
        write_pertic_episode(lat_dir, ep, held_actions([0, 1, 2, 0, 1, 2]))
    if latent_dup_rows:
        path = os.path.join(lat_dir, f"ep_{VAL_EP:05d}_latents.npy")
        lat = np.load(path)
        for t in latent_dup_rows:
            lat[t] = lat[t - 1]
        np.save(path, lat)
    split = os.path.join(str(root), "split.json")
    json.dump({"train": [0], "val": [VAL_EP]}, open(split, "w"))
    return lat_dir, split


@pytest.fixture
def scorer_inputs(tmp_path):
    """One checkpoint and one decoder, shared by every run in a test so runs compare like for like."""
    model = _tiny_model()
    ck = str(tmp_path / "best.pt")
    torch.save({"model": model.state_dict(), "step": 7,
                "args": {"action_dropout": 0.0, "objective": "v", "tic_stride": 1}}, ck)
    return ck, _tiny_vae(tmp_path / "vae")


def _argv(ck, vae, lat_dir, split, out, horizon=1, windows=6, raw=True, extra=()):
    argv = ["--ckpt", ck, "--backbone", "pixart", "--pixart-path", backbones.PIXART_DEFAULT,
            "--latent-channels", "4", "--vae-path", vae, "--latents-dir", lat_dir, "--split", split,
            "--subset", "val", "--context-frames", str(CTX), "--num-actions", "3", "--noise-buckets", "4",
            "--num-windows", str(windows), "--batch-size", "4", "--steps", "2", "--save-images", "0",
            "--num-workers", "0", "--horizon-tics", str(horizon), "--out-dir", out]
    if raw:
        argv += ["--parquet-dir", os.path.dirname(split)]
    return argv + list(extra)


def _score(module, argv):
    module.main(module.build_parser().parse_args(argv))
    out = argv[argv.index("--out-dir") + 1]
    with open(os.path.join(out, "per_window.csv"), newline="") as f:
        reader = csv.DictReader(f)
        header, rows = list(reader.fieldnames), list(reader)
    return json.load(open(os.path.join(out, "metrics.json"))), header, rows


def _pinned_scorer(tmp_path):
    """The pre-change eval_tf.py, from git, as its own module; skipped where there is no history."""
    try:
        src = subprocess.run(["git", "-C", REPO, "show", f"{PINNED_SCORER}:eval_tf.py"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"no git history to read the pinned scorer from: {e}")
    path = tmp_path / "eval_tf_pinned.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("eval_tf_pinned", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------
# every frozen column is unchanged
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("horizon", [1, 4])
def test_the_existing_summary_and_columns_are_unchanged(tmp_path, tiny_hub, scorer_inputs, monkeypatch, horizon):
    """Against the scorer that produced the frozen results: every summary key it wrote, and every per-window
    column, holds the same value serialised the same way. The new columns only add."""
    pinned = _pinned_scorer(tmp_path)
    frames = _frames(dup_tics=(12,))
    monkeypatch.setattr(pinned, "RawFrames", _fake_raw(frames))
    monkeypatch.setattr(eval_tf, "RawFrames", _fake_raw(frames))
    ck, vae = scorer_inputs
    lat_dir, split = _corpus(tmp_path / "corpus", latent_dup_rows=(12,))
    old, old_header, old_rows = _score(pinned, _argv(ck, vae, lat_dir, split, str(tmp_path / "old"), horizon, 8))
    new, new_header, new_rows = _score(eval_tf, _argv(ck, vae, lat_dir, split, str(tmp_path / "new"), horizon, 8))
    for k, v in old.items():
        if k in ("config", "sampling_frames_per_s"):
            continue
        assert json.dumps(new[k]) == json.dumps(v), k
    old_config = {k: v for k, v in old["config"].items() if k != "out_dir"}
    assert {k: new["config"][k] for k in old_config} == old_config
    assert set(new["config"]) - set(old["config"]) == {"decoders", "save_latents"}
    # the per-window file: the old header is the new header's prefix, and every old cell is identical
    assert new_header[:len(old_header)] == old_header
    assert len(new_rows) == len(old_rows) == 8
    for o, n in zip(old_rows, new_rows):
        assert {k: n[k] for k in old_header} == o


# ---------------------------------------------------------------------------------------
# the new columns
# ---------------------------------------------------------------------------------------

def test_the_scene_and_hud_crops_tile_the_frame():
    x = torch.rand(2, 3, 240, 320)
    scene, hud = eval_tf.scene_crop(x), eval_tf.hud_crop(x)
    assert scene.shape == (2, 3, 208, 320) and hud.shape == (2, 3, eval_tf.HUD_ROWS, 320)
    assert torch.equal(torch.cat([scene, hud], dim=2), x)
    # the full-frame MSE is the row-weighted mean of the two crops'
    full = eval_tf.pixel_mse(x, torch.zeros_like(x))
    parts = (208 * eval_tf.pixel_mse(scene, torch.zeros_like(scene))
             + 32 * eval_tf.pixel_mse(hud, torch.zeros_like(hud))) / 240
    assert torch.allclose(full, parts, rtol=1e-6)


def test_pixel_mse_is_what_psnr_takes_the_log_of():
    a, b = torch.rand(3, 3, 240, 320), torch.rand(3, 3, 240, 320)
    assert torch.allclose(eval_tf.psnr(a, b), 10 * torch.log10(1.0 / eval_tf.pixel_mse(a, b)))
    assert float(eval_tf.pixel_mse(a[:1], a[:1])) == 0.0         # unclamped, where psnr reads 100 dB


def test_copy_lpips_dec_is_the_lpips_of_the_decoded_copy_against_the_decoded_truth():
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).eval()
    g = torch.Generator().manual_seed(0)
    pred, gt, last, raw = (torch.rand(1, 3, 240, 320, generator=g) for _ in range(4))
    m = eval_tf.decoder_metrics(pred, gt, last, lp, raw)
    assert m["copy_lpips_dec"] == float(lp(last * 2 - 1, gt * 2 - 1).flatten())
    assert m["copy_psnr_dec"] == float(eval_tf.psnr(last, gt))           # the same two images
    assert m["lpips_dec"] == float(lp(pred * 2 - 1, gt * 2 - 1).flatten())
    assert m["scene_copy_lpips_dec"] == float(lp(last[:, :, :208] * 2 - 1, gt[:, :, :208] * 2 - 1).flatten())
    assert m["scene_vae_psnr"] == float(eval_tf.psnr(gt[:, :, :208], raw[:, :, :208]))
    assert m["hud_mse_raw"] == float(eval_tf.pixel_mse(pred[:, :, 208:], raw[:, :, 208:]))
    # no raw frame, no raw columns
    assert not any(k.endswith("_raw") or k.startswith(("vae_", "scene_vae_", "hud_vae_"))
                   for k in eval_tf.decoder_metrics(pred, gt, last, lp))


# every decoder-dependent pixel column, as `decoder_metrics` names it with a raw frame
DECODER_COLUMNS = {
    "psnr_dec", "mse_dec", "lpips_dec", "hud_psnr_dec", "hud_mse_dec",
    "scene_psnr_dec", "scene_mse_dec", "scene_lpips_dec",
    "copy_psnr_dec", "copy_mse_dec", "copy_lpips_dec",
    "scene_copy_psnr_dec", "scene_copy_mse_dec", "scene_copy_lpips_dec",
    "psnr_raw", "mse_raw", "lpips_raw", "hud_psnr_raw", "hud_mse_raw",
    "scene_psnr_raw", "scene_mse_raw", "scene_lpips_raw",
    "copy_psnr_raw", "copy_mse_raw", "scene_copy_psnr_raw", "scene_copy_mse_raw",
    "vae_psnr", "vae_mse", "vae_lpips", "hud_vae_psnr", "hud_vae_mse",
    "scene_vae_psnr", "scene_vae_mse", "scene_vae_lpips"}
DECODER_FREE_COLUMNS = {
    "latent_mse", "copy_latent_mse", "latent_mse_ratio",
    "persist_psnr_raw", "persist_mse_raw", "persist_lpips_raw", "persist_hud_psnr_raw", "persist_hud_mse_raw",
    "scene_persist_psnr_raw", "scene_persist_mse_raw", "scene_persist_lpips_raw", "context_motion"}
WINDOW_IDENTITY = {"index", "episode", "map", "start", "action", "tics_since_decision",
                   "start_tic", "scored_tic", "dup_raw", "dup_latent"}


@pytest.mark.parametrize("horizon", [1, 4])
def test_every_window_row_carries_the_new_columns(tmp_path, tiny_hub, scorer_inputs, monkeypatch, horizon):
    """The same column set at one tic and at four; identities and motion read back from the fixture, where
    row t's latent is filled with t, so every consecutive context step changes each element by exactly 1."""
    monkeypatch.setattr(eval_tf, "RawFrames", _fake_raw(_frames()))
    ck, vae = scorer_inputs
    lat_dir, split = _corpus(tmp_path / "corpus")
    m, header, rows = _score(eval_tf, _argv(ck, vae, lat_dir, split, str(tmp_path / "out"), horizon))
    assert set(header) == WINDOW_IDENTITY | DECODER_COLUMNS | DECODER_FREE_COLUMNS
    for r in rows:
        assert int(r["episode"]) == VAL_EP and int(r["map"]) == 7
        assert int(r["start_tic"]) == int(r["start"])                   # tic t is row t in the fixture
        assert int(r["scored_tic"]) == int(r["start"]) + CTX + horizon - 1
        assert float(r["context_motion"]) == pytest.approx(1.0)
        assert r["dup_raw"] == r["dup_latent"] == "0"
        # scene and HUD rows account for the whole frame in every MSE
        for full, scene, hud in (("mse_dec", "scene_mse_dec", "hud_mse_dec"),
                                 ("mse_raw", "scene_mse_raw", "hud_mse_raw"),
                                 ("vae_mse", "scene_vae_mse", "hud_vae_mse"),
                                 ("persist_mse_raw", "scene_persist_mse_raw", "persist_hud_mse_raw")):
            assert float(r[full]) == pytest.approx((208 * float(r[scene]) + 32 * float(r[hud])) / 240, rel=1e-5)
        assert float(r["psnr_dec"]) == pytest.approx(10 * np.log10(1 / float(r["mse_dec"])), rel=1e-5)
    for k in ("copy_lpips_dec", "scene_psnr_dec", "mse_raw", "scene_persist_lpips_raw", "context_motion"):
        assert m[k]["n"] == len(rows)
    assert m["duplicates"]["excluded"] == 0 and m["psnr_dec_nodup"] == m["psnr_dec"]


def test_copy_lpips_dec_scores_the_same_decoded_images_as_copy_psnr_dec(tmp_path, tiny_hub, scorer_inputs):
    """End to end: recompute both copy columns from the corpus and the decoder, outside the scorer. The
    decodes are batched as the scorer batched them (rows in groups of --batch-size), because a CPU
    convolution over a batch of 4 and over a batch of 1 differ by up to 5e-4 per pixel."""
    import lpips
    from doom_data import TicWindowDataset
    from doomdit_utils import build_vae
    ck, vae_dir = scorer_inputs
    lat_dir, split = _corpus(tmp_path / "corpus")
    _, _, rows = _score(eval_tf, _argv(ck, vae_dir, lat_dir, split, str(tmp_path / "out"), raw=False))
    ds = TicWindowDataset(lat_dir, [VAL_EP], CTX, latent_channels=4, horizon=1, with_horizon=True)
    vae = build_vae(vae_dir, "", "cpu")
    lp = lpips.LPIPS(net="alex", verbose=False).eval()
    for b in range(0, len(rows), 4):
        batch = rows[b:b + 4]
        items = [ds[int(r["index"])] for r in batch]
        last = eval_tf.decode(vae, torch.stack([it[0][-4:] for it in items]))
        gt = eval_tf.decode(vae, torch.stack([it[1][0] for it in items]))
        for i, r in enumerate(batch):
            li, gi = last[i:i + 1], gt[i:i + 1]
            assert float(r["copy_lpips_dec"]) == float(lp(li * 2 - 1, gi * 2 - 1).flatten())
            assert float(r["copy_psnr_dec"]) == float(eval_tf.psnr(li, gi))


def test_duplicate_windows_are_flagged_and_left_out_of_every_nodup_mean(tmp_path, tiny_hub, scorer_inputs, monkeypatch):
    """Target tic 10 repeats its last context frame in pixels and in latents, 15 only in latents, 18 only in
    pixels. The flags are separate; the default means still average every window; `*_nodup` drops all three."""
    monkeypatch.setattr(eval_tf, "RawFrames", _fake_raw(_frames(dup_tics=(10, 18))))
    ck, vae = scorer_inputs
    lat_dir, split = _corpus(tmp_path / "corpus", latent_dup_rows=(10, 15))
    m, _, rows = _score(eval_tf, _argv(ck, vae, lat_dir, split, str(tmp_path / "out"), windows=100))
    by_tic = {int(r["scored_tic"]): r for r in rows}
    assert {10, 15, 18} <= set(by_tic)
    assert {t for t, r in by_tic.items() if r["dup_raw"] == "1"} == {10, 18}
    assert {t for t, r in by_tic.items() if r["dup_latent"] == "1"} == {10, 15}
    assert float(by_tic[10]["persist_mse_raw"]) == 0.0 and float(by_tic[15]["copy_latent_mse"]) == 0.0
    assert m["duplicates"] == {"rule": "a window is left out of every *_nodup mean when dup_raw or dup_latent is 1",
                               "windows": len(rows), "dup_raw": 2, "dup_latent": 2, "excluded": 3}
    kept = [r for r in rows if int(r["scored_tic"]) not in (10, 15, 18)]
    for k in ("psnr_dec", "persist_psnr_raw", "scene_lpips_raw", "copy_latent_mse", "mse_raw"):
        assert m[k]["n"] == len(rows)                                  # the default mean is unchanged
        assert m[f"{k}_nodup"]["n"] == len(kept)
        assert m[f"{k}_nodup"]["mean"] == pytest.approx(np.mean([float(r[k]) for r in kept]))
    # the persistence floor's 100 dB clamp on an exact repeat is what the exclusion takes out
    assert m["persist_psnr_raw_nodup"]["mean"] < m["persist_psnr_raw"]["mean"]
    assert set(m["per_map_nodup"]) == set(m["per_map"])
    assert sum(v["windows"] for v in m["per_tics_since_decision_nodup"].values()) == len(kept)


def test_without_raw_frames_the_latent_flag_alone_is_the_rule(tmp_path, tiny_hub, scorer_inputs):
    ck, vae = scorer_inputs
    lat_dir, split = _corpus(tmp_path / "corpus", latent_dup_rows=(10,))
    m, header, rows = _score(eval_tf, _argv(ck, vae, lat_dir, split, str(tmp_path / "out"), windows=100, raw=False))
    assert "dup_raw" not in header and not any(k.endswith("_raw") for k in header)
    assert m["duplicates"]["excluded"] == m["duplicates"]["dup_latent"] == 1
    assert m["duplicates"]["rule"] == "a window is left out of every *_nodup mean when dup_latent is 1"
    assert m["latent_mse_nodup"]["n"] == len(rows) - 1


# ---------------------------------------------------------------------------------------
# --decoder: one set of predictions, every decoder
# ---------------------------------------------------------------------------------------

def test_decoder_specs_parse_to_suffixes():
    assert eval_tf.parse_decoders(None) == []
    assert eval_tf.parse_decoders(["/d/vae_decoder_sd1x_mse/vae"]) == [("tuned", "/d/vae_decoder_sd1x_mse/vae")]
    assert eval_tf.parse_decoders(["mse=/d/a", "lpips2=/d/b"]) == [("mse", "/d/a"), ("lpips2", "/d/b")]
    assert eval_tf.parse_decoders(["/d/run=3/vae"]) == [("tuned", "/d/run=3/vae")]   # not a name, a path
    with pytest.raises(SystemExit, match="twice"):
        eval_tf.parse_decoders(["/d/a", "/d/b"])
    with pytest.raises(SystemExit, match="reserved"):
        eval_tf.parse_decoders(["nodup=/d/a"])
    with pytest.raises(SystemExit, match="no path"):
        eval_tf.parse_decoders(["mse="])


def test_two_decoders_score_one_set_of_predictions(tmp_path, tiny_hub, scorer_inputs, monkeypatch):
    """`same` is a copy of the stock decoder, so its columns must equal the unsuffixed ones exactly, which
    only holds if both decoded the same predictions; `other` has different weights. The sampler runs once
    per batch whatever the number of decoders, and the decoder-free columns appear once."""
    monkeypatch.setattr(eval_tf, "RawFrames", _fake_raw(_frames()))
    ck, vae = scorer_inputs
    same = str(tmp_path / "same")
    shutil.copytree(vae, same)
    torch.manual_seed(123)
    other = _tiny_vae(tmp_path / "other")
    lat_dir, split = _corpus(tmp_path / "corpus")
    calls = []
    real = eval_tf.sample_spaced

    def counted(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(eval_tf, "sample_spaced", counted)
    base, _, base_rows = _score(eval_tf, _argv(ck, vae, lat_dir, split, str(tmp_path / "base"), windows=8))
    calls.clear()
    m, header, rows = _score(eval_tf, _argv(ck, vae, lat_dir, split, str(tmp_path / "two"), windows=8,
                                            extra=["--decoder", f"same={same}", "--decoder", other]))
    assert len(calls) == 2                                     # 8 windows, batch 4: one sample per batch
    suffixed = {c for c in header if c.endswith(("_same", "_tuned"))}
    assert suffixed == {f"{c}_same" for c in DECODER_COLUMNS} | {f"{c}_tuned" for c in DECODER_COLUMNS}
    assert set(header) == WINDOW_IDENTITY | DECODER_COLUMNS | DECODER_FREE_COLUMNS | suffixed
    for r, b in zip(rows, base_rows):
        for c in DECODER_COLUMNS:
            assert r[f"{c}_same"] == r[c], c
        assert r["psnr_dec_tuned"] != r["psnr_dec"] and r["vae_psnr_tuned"] != r["vae_psnr"]
        for c in DECODER_COLUMNS | DECODER_FREE_COLUMNS:          # adding decoders changes no stock column
            assert r[c] == b[c], c
    assert m["psnr_dec_tuned"]["n"] == 8 and "psnr_dec_tuned_nodup" in m
    assert set(m["extra_decoders"]) == {"same", "tuned"}
    assert m["extra_decoders"]["tuned"]["suffix"] == "_tuned" and m["extra_decoders"]["tuned"]["path"] == other
    assert m["decoder"] == base["decoder"]                     # the stock decoder's record is unchanged


def test_a_decoder_with_another_latent_contract_is_refused(tmp_path, tiny_hub, scorer_inputs):
    """A decoder reads the stock encoder's latents, so its scale must be the corpus's."""
    from diffusers.models import AutoencoderKL
    ck, vae = scorer_inputs
    wrong = str(tmp_path / "wrong")
    cfg = dict(AutoencoderKL.load_config(vae))
    cfg["scaling_factor"] = 0.5
    AutoencoderKL.from_config(cfg).save_pretrained(wrong)
    lat_dir, split = _corpus(tmp_path / "corpus")
    with pytest.raises(SystemExit, match="latent contract"):
        _score(eval_tf, _argv(ck, vae, lat_dir, split, str(tmp_path / "out"), raw=False,
                              extra=["--decoder", f"bad={wrong}"]))


# ---------------------------------------------------------------------------------------
# --save-latents
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("horizon", [1, 4])
def test_the_saved_latents_are_the_scored_predictions(tmp_path, tiny_hub, scorer_inputs, monkeypatch, horizon):
    """The npz holds, in per_window.csv's order, the fp16 prediction every pixel column decoded (the Kth
    tic's at horizon K) and the identities that find the window again in the corpus."""
    ck, vae = scorer_inputs
    lat_dir, split = _corpus(tmp_path / "corpus")
    scored = []
    real = eval_tf.sample_spaced

    def recorded(*a, **kw):
        out = real(*a, **kw)
        scored.append(out.detach().clone())
        return out

    monkeypatch.setattr(eval_tf, "sample_spaced", recorded)
    out = str(tmp_path / "out")
    m, _, rows = _score(eval_tf, _argv(ck, vae, lat_dir, split, out, horizon, raw=False,
                                       extra=["--save-latents"]))
    z = np.load(os.path.join(out, eval_tf.LATENTS_FILE))
    last_steps = scored[horizon - 1::horizon]                 # the step each batch's scored frame came from
    want = torch.cat(last_steps).to(torch.float16).numpy()
    assert z["pred"].dtype == np.float16 and z["pred"].shape == want.shape == (len(rows), 4, 32, 40)
    assert np.array_equal(z["pred"], want)
    for k in ("index", "episode", "start", "start_tic", "scored_tic", "map", "tics_since_decision"):
        assert z[k].tolist() == [int(r[k]) for r in rows], k
    assert int(z["horizon_tics"]) == horizon
    assert m["config"]["save_latents"] is True


def test_no_latents_file_unless_asked(tmp_path, tiny_hub, scorer_inputs):
    ck, vae = scorer_inputs
    lat_dir, split = _corpus(tmp_path / "corpus")
    out = str(tmp_path / "out")
    _score(eval_tf, _argv(ck, vae, lat_dir, split, out, raw=False))
    assert not os.path.exists(os.path.join(out, eval_tf.LATENTS_FILE))
    assert os.path.exists(os.path.join(out, "per_window.csv"))
