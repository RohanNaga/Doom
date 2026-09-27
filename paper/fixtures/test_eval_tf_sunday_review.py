"""Offline review of 5b8301e against the actual frozen scorer and Sunday protocol.

The three failing tests assert required behavior; they are deliberately not xfailed.
Only temporary fixtures are written. No downloads, real LPIPS weights, or GPU needed.
Run with PYTHONDONTWRITEBYTECODE=1 and pytest -p no:cacheprovider.
"""
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import adapt_fixtures as F
import adapt_split
import eval_tf
import test_eval_tf_columns as C
from doom_data import TicWindowDataset
from pertic_fixtures import write_pertic_episode

tiny_hub = C.tiny_hub
scorer_inputs = C.scorer_inputs
FROZEN = REPO / "results/distance_study/scores/040-unet-nexttic_snap_0200000_ema_ddim10"


@pytest.fixture(autouse=True)
def offline_cpu(monkeypatch):
    F.stub_lpips(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def frozen_comparison(tmp_path, tiny_hub, scorer_inputs, monkeypatch):
    # Use the committed provenance, not the newer PINNED_SCORER in the existing test.
    path = FROZEN / "val_map02_h1"
    commit = re.search(r"code=(\w+)", (path / "provenance.txt").read_text()).group(1)
    monkeypatch.setattr(C, "PINNED_SCORER", commit)
    old_module = C._pinned_scorer(tmp_path)
    raw = C._fake_raw(C._frames(dup_tics=(12,)))
    monkeypatch.setattr(old_module, "RawFrames", raw)
    monkeypatch.setattr(eval_tf, "RawFrames", raw)
    ck, vae = scorer_inputs
    lat, split = C._corpus(tmp_path / "corpus", latent_dup_rows=(12,))
    old = C._score(old_module, C._argv(ck, vae, lat, split, str(tmp_path / "old"), windows=8))
    new = C._score(eval_tf, C._argv(ck, vae, lat, split, str(tmp_path / "new"), windows=8))
    # Confirm that the historical module actually has all 30 frozen metric schemas.
    old_keys = [k for k, v in old[0].items() if isinstance(v, dict) and "mean" in v]
    files = list(FROZEN.glob("*_h1/metrics.json"))
    assert len(files) == 30
    for f in files:
        m = json.loads(f.read_text())
        assert [k for k, v in m.items() if isinstance(v, dict) and "mean" in v] == old_keys
    return old, new


def test_actual_frozen_scorer_values_are_unchanged(frozen_comparison):
    """Numeric compatibility passes on synthetic data even though column positions fail."""
    (old, header, rows), (new, _, new_rows) = frozen_comparison
    for k, v in old.items():
        if k not in ("config", "sampling_frames_per_s"):
            assert json.dumps(new[k]) == json.dumps(v), k
    for a, b in zip(rows, new_rows):
        assert {k: b[k] for k in header} == a


def test_actual_frozen_csv_header_is_a_prefix(frozen_comparison):
    (_, old_header, _), (_, new_header, _) = frozen_comparison
    assert new_header[:len(old_header)] == old_header


def test_copy_last_raw_lpips_is_complete_for_each_decoder():
    """Decision item 3 requests pixel MSE and LPIPS of each decoded model/copy/reconstruction."""
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False)
    gen = torch.Generator().manual_seed(30)
    pred, gt, last, raw = [torch.rand(1, 3, 240, 320, generator=gen) for _ in range(4)]
    m = eval_tf.decoder_metrics(pred, gt, last, lp, raw)
    assert {"copy_lpips_raw", "scene_copy_lpips_raw"} <= m.keys()
    assert m["copy_lpips_raw"] == eval_tf.lpips_of(lp, last, raw)
    assert m["scene_copy_lpips_raw"] == eval_tf.lpips_of(lp, last[:, :, :208], raw[:, :, :208])


def test_32_window_manifest_is_valid_for_the_required_four_tic_rescore(tmp_path):
    """A real 32-window draw, with ample valid h4 windows, currently picks three h1-only starts."""
    lat = str(tmp_path / "lat")
    write_pertic_episode(lat, 2, np.zeros(128, dtype=np.int64))
    pairs, _ = adapt_split.draw_held_out_windows(lat, [2], per_episode=32, seed=0, context_frames=32)
    assert len(pairs) == 32
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"meta": {"kind": adapt_split.KIND}, "held_out_windows": pairs}))
    ds = TicWindowDataset(lat, [2], 32, horizon=4, with_horizon=True)
    assert len(ds) >= 32
    # These are the exact two helpers eval_tf.main uses at lines 449-450.
    indices = adapt_split.windows_in_dataset(ds, adapt_split.load_windows(str(manifest)))
    assert len(indices) == 32


def test_crop_strips_padding_and_lpips_uses_identical_normalization():
    seen = []

    class VAE:
        def decode(self, z):
            seen.append(z.clone())
            # Distinct scene, HUD and padding values catch either incorrect crop boundary.
            out = torch.full((1, 3, 256, 320), -0.5)
            out[:, :, 208:240] = 0.5
            out[:, :, 240:] = 1.0
            return SimpleNamespace(sample=out)

    z = torch.full((1, 16, 32, 40), 0.125)
    image = eval_tf.decode(VAE(), z, scale=1.5305, shift=0.0609)
    assert torch.equal(seen[0], z / 1.5305 + 0.0609)
    assert image.shape == (1, 3, 240, 320)
    assert torch.all(eval_tf.scene_crop(image) == 0.25)
    assert torch.all(eval_tf.hud_crop(image) == 0.75)
    calls = []

    def lp(a, b):
        calls.append((a.clone(), b.clone()))
        return (a - b).abs().mean().reshape(1)

    eval_tf.pair_metrics(image, torch.zeros_like(image), lp, "psnr", "lpips")
    assert torch.equal(calls[0][0], image * 2 - 1)
    assert torch.equal(calls[1][0], calls[0][0][:, :, :208])
    assert torch.equal(calls[1][1], calls[0][1][:, :, :208])


def test_lpips_call_budget_is_twenty_two_for_two_decoders():
    calls = []

    def lp(a, b):
        calls.append(a.shape)
        return torch.zeros(1)

    x = torch.zeros(1, 3, 240, 320)
    for _ in range(2):
        eval_tf.decoder_metrics(x, x, x, lp, x)
    eval_tf.persistence_metrics(x, x, lp)
    # ten per decoder (finding 3 adds copy_lpips_raw and its scene crop) plus two decoder-free persistence reads
    assert len(calls) == 22


def test_every_nodup_column_uses_the_same_mask_with_a_second_decoder(
        tmp_path, tiny_hub, scorer_inputs, monkeypatch):
    ck, vae = scorer_inputs
    monkeypatch.setattr(eval_tf, "RawFrames", C._fake_raw(C._frames(dup_tics=(10, 18))))
    lat, split = C._corpus(tmp_path / "corpus", latent_dup_rows=(10, 15))
    decodes = []
    real_decode = eval_tf.decode

    def recorded(vae, z, scale, shift):
        decodes.append((z.detach().clone(), scale, shift))
        return real_decode(vae, z, scale, shift)

    monkeypatch.setattr(eval_tf, "decode", recorded)
    m, header, rows = C._score(eval_tf, C._argv(
        ck, vae, lat, split, str(tmp_path / "out"), windows=100,
        extra=["--decoder", f"tuned={vae}"]))
    assert len(decodes) == 6 * ((len(rows) + 3) // 4)
    for i in range(0, len(decodes), 6):
        for j in range(3):
            a, b = decodes[i+j], decodes[i+j+3]
            assert torch.equal(a[0], b[0]) and a[1:] == b[1:]
    excluded = [r for r in rows if int(r["dup_raw"]) or int(r["dup_latent"])]
    assert {int(r["scored_tic"]) for r in excluded} == {10, 15, 18}
    kept = [r for r in rows if r not in excluded]
    for k in header:
        if k in eval_tf.WINDOW_COLUMNS:
            continue
        want = [float(r[k]) for r in kept]
        assert np.isfinite(want).all()
        assert m[k + "_nodup"]["n"] == len(kept), k
        assert m[k + "_nodup"]["mean"] == pytest.approx(np.mean(want)), k
