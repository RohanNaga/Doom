"""The `latent_channels` generalisation must leave the 4-channel rows bit-identical.

Every existing row (DiT, U-Net, PixArt, UniDiffuser) was trained in the SD KL-f8 latent space,
so the whole point of `latent_channels` is that it is invisible to them. These checks pin that:
the shared helpers still produce the 132-channel input and the 4-channel target, the trainer's
synthetic windows and the dataset still hand back (4L, 32, 40) / (4, 32, 40), and the latent
normalisation round-trips to the same numbers `LATENT_SCALE` alone used to give.

The heavy diffusers backbones are not built here (they need network access and gigabytes of
weights); `verify_sd35.py` gates those on CPU.

    python -m pytest paper/fixtures/test_latent_channels.py -q
"""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import backbones  # noqa: E402
import doom_data  # noqa: E402
import doomdit_utils  # noqa: E402
from train_wm import SyntheticWindows  # noqa: E402


def test_stacked_input_channels_match_the_trained_rows():
    # 132 = 32 context latents x 4 + 4, the number every finished row's config.json carries
    assert backbones.stacked_in_channels(4, 32) == 132
    assert backbones.stacked_in_channels(16, 32) == 528


def test_backbone_latent_channels_defaults():
    for backbone in ("dit", "unet", "pixart", "unidiffuser"):
        assert backbones.resolve_latent_channels(backbone) == 4
        assert backbones.resolve_latent_channels(backbone, 0) == 4
    assert backbones.resolve_latent_channels("sd35") == 16
    assert backbones.resolve_latent_channels("dit", 16) == 16   # an explicit request always wins


def test_input_conv_inflation_is_pretrained_on_the_target_and_zero_elsewhere():
    old = torch.nn.Conv2d(4, 8, 2, 2)
    new = backbones.inflate_input_conv(old, 132, 4)
    assert new.weight.shape == (8, 132, 2, 2)
    assert torch.equal(new.weight[:, -4:], old.weight)
    assert new.weight[:, :-4].abs().max().item() == 0.0
    assert torch.equal(new.bias, old.bias)
    # zero context channels must reproduce the pretrained conv exactly
    x = torch.randn(2, 4, 32, 40)
    padded = torch.cat([torch.zeros(2, 128, 32, 40), x], dim=1)
    assert (new(padded) - old(x)).abs().max().item() == 0.0


def test_input_conv_inflation_refuses_the_wrong_latent_space():
    with pytest.raises(ValueError):
        backbones.inflate_input_conv(torch.nn.Conv2d(4, 8, 2, 2), 528, 16)


def test_dit_warm_start_refuses_a_non_four_channel_latent_space(tmp_path):
    m = backbones.DiTWorldModel(num_actions=2, context_frames=1, noise_buckets=2, model_name="DiT-S/2",
                                grad_ckpt=False, latent_channels=16)
    ckpt = tmp_path / "fake.pt"
    torch.save({"model": {}}, ckpt)
    with pytest.raises(ValueError):
        m.load_imagenet_warm_start(str(ckpt))


def test_dit_four_channel_shapes_are_unchanged():
    m = backbones.DiTWorldModel(num_actions=3, context_frames=2, noise_buckets=2, model_name="DiT-S/2",
                                action_dropout=0.0, grad_ckpt=False)
    assert m.dit.x_embedder.proj.weight.shape[1] == 12   # 2 * 4 + 4
    out = m(torch.randn(2, 4, 32, 40), torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
            torch.randn(2, 8, 32, 40), torch.zeros(2, dtype=torch.long))
    assert out.shape == (2, 4, 32, 40)


def test_synthetic_windows_default_to_the_four_channel_layout():
    ds = SyntheticWindows(4, 32, 29)
    ctx, tgt, act = ds[0]
    assert ctx.shape == (128, 32, 40) and tgt.shape == (4, 32, 40)
    # same generator seed, same numbers as before the generalisation
    g = torch.Generator().manual_seed(0)
    assert torch.equal(ctx, torch.randn(128, 32, 40, generator=g))
    assert torch.equal(tgt, torch.randn(4, 32, 40, generator=g))
    assert 0 <= int(act) < 29
    assert SyntheticWindows(4, 32, 29, 16)[0][0].shape == (512, 32, 40)


def _write_episode(d, ep, T, channels):
    lat = np.arange(T * channels * 32 * 40, dtype=np.float16).reshape(T, channels, 32, 40)
    np.save(os.path.join(d, f"ep_{ep:05d}_latents.npy"), lat)
    np.savez(os.path.join(d, f"ep_{ep:05d}_meta.npz"), action=np.zeros(T, np.int64),
             tic=np.arange(T, dtype=np.int64) * 4, map_id=np.full(T, 1, np.int64),
             chain_id=np.zeros(T, np.int64))


def test_latent_window_dataset_defaults_to_four_channels(tmp_path):
    _write_episode(str(tmp_path), 1, 6, 4)
    ds = doom_data.LatentWindowDataset(str(tmp_path), [1], context_frames=4)
    ctx, tgt, act = ds[0]
    assert ctx.shape == (16, 32, 40) and tgt.shape == (4, 32, 40) and int(act) == 0
    assert ds.latent_channels == 4


def test_latent_window_dataset_rejects_a_corpus_in_another_latent_space(tmp_path):
    _write_episode(str(tmp_path), 2, 6, 16)
    with pytest.raises(ValueError):
        doom_data.LatentWindowDataset(str(tmp_path), [2], context_frames=4)
    ds = doom_data.LatentWindowDataset(str(tmp_path), [2], context_frames=4, latent_channels=16)
    assert ds[0][0].shape == (64, 32, 40) and ds[0][1].shape == (16, 32, 40)


def test_latent_normalisation_matches_the_bare_scale_for_the_sd_rows():
    z = torch.randn(3, 4, 32, 40)
    stored = doomdit_utils.normalize_latents(z)
    assert torch.equal(stored, z * doomdit_utils.LATENT_SCALE)
    assert torch.equal(doomdit_utils.denormalize_latents(stored), stored / doomdit_utils.LATENT_SCALE)


def test_latent_normalisation_round_trips_with_a_shift():
    z = torch.randn(3, 16, 32, 40).double()
    scale, shift = 1.5305, 0.0609            # SD 3.5's autoencoder
    stored = doomdit_utils.normalize_latents(z, scale, shift)
    assert torch.equal(stored, (z - shift) * scale)
    assert torch.allclose(doomdit_utils.denormalize_latents(stored, scale, shift), z, atol=1e-12)
