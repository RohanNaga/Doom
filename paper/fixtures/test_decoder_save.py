"""The save half of the decoder tune, which lost a finished two-epoch run on 2026-09-18.

`--channels-last` calls `vae.to(memory_format=torch.channels_last)`, which leaves every 4-D
conv weight strided as NHWC. `safetensors.torch.save_file` refuses a non-contiguous tensor, so
`save_pretrained` wrote `config.json` and then raised, after the tuning GPU hours were already
spent. These checks pin the three things that keep that from recurring: the weights are packed
before the write, the written directory is proved loadable before it replaces the previous
checkpoint, and a directory holding only a config fails with a message that names it.

    python -m pytest paper/fixtures/test_decoder_save.py -q
"""
import json
import os
import re
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from diffusers.models import AutoencoderKL  # noqa: E402

from doomdit_utils import build_vae, resolve_vae_dir  # noqa: E402
from finetune_decoder import save_vae  # noqa: E402


def tiny_vae(latent_channels=4):
    """A two-level AutoencoderKL small enough to save in a test, same class as the real one."""
    return AutoencoderKL(in_channels=3, out_channels=3,
                         down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
                         up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
                         block_out_channels=(8, 8), layers_per_block=1, norm_num_groups=8,
                         latent_channels=latent_channels, sample_size=32)


def test_channels_last_weights_are_not_contiguous():
    """The precondition the bug rests on: without packing, safetensors would refuse these."""
    vae = tiny_vae().to(memory_format=torch.channels_last)
    assert not vae.encoder.conv_in.weight.is_contiguous()


def test_save_vae_round_trips_a_channels_last_model(tmp_path):
    vae = tiny_vae().to(memory_format=torch.channels_last)
    out = str(tmp_path / "run")
    path = save_vae(vae, out)

    assert path == os.path.join(out, "vae")
    assert os.path.exists(os.path.join(path, "diffusion_pytorch_model.safetensors"))
    assert not os.path.exists(os.path.join(out, "vae.saving"))       # scratch directory cleaned up

    reloaded = AutoencoderKL.from_pretrained(path)
    live = vae.state_dict()
    assert set(reloaded.state_dict()) == set(live)
    for k, v in reloaded.state_dict().items():
        assert torch.equal(v, live[k].cpu()), k


def test_save_vae_leaves_the_model_trainable_in_channels_last(tmp_path):
    """The epoch checkpoint happens mid-run, so packing must not change the model or its layout."""
    vae = tiny_vae().to(memory_format=torch.channels_last)
    before = {k: v.clone() for k, v in vae.state_dict().items()}
    save_vae(vae, str(tmp_path / "run"), channels_last=True)

    assert not vae.encoder.conv_in.weight.is_contiguous()             # still NHWC for the next epoch
    for k, v in vae.state_dict().items():
        assert torch.equal(v, before[k]), k
    x = torch.randn(1, 3, 32, 32)
    assert vae.decode(vae.encode(x).latent_dist.mean).sample.shape == x.shape


def test_save_vae_keeps_the_old_checkpoint_when_the_new_one_will_not_load(tmp_path, monkeypatch):
    """A failed save must not destroy the epoch that already succeeded."""
    out = str(tmp_path / "run")
    save_vae(tiny_vae(), out)
    good = open(os.path.join(out, "vae", "config.json")).read()

    def broken(self, *a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(AutoencoderKL, "save_pretrained", broken)
    with pytest.raises(RuntimeError):
        save_vae(tiny_vae(), out)
    assert open(os.path.join(out, "vae", "config.json")).read() == good


def test_save_vae_rejects_a_reload_that_differs(tmp_path, monkeypatch):
    """The verification is the point: a silently wrong write has to fail loudly."""
    def wrong(cls, path, **kw):
        return tiny_vae(latent_channels=8)

    monkeypatch.setattr(AutoencoderKL, "from_pretrained", classmethod(wrong))
    with pytest.raises(SystemExit, match="reload"):
        save_vae(tiny_vae(), str(tmp_path / "run"))


def test_resolve_vae_dir_accepts_both_layouts(tmp_path):
    """The scorer is pointed at `<out>/vae`, but `<out>` itself has to work too."""
    out = tmp_path / "run"
    save_vae(tiny_vae(), str(out))
    assert resolve_vae_dir(str(out / "vae")) == str(out / "vae")
    assert resolve_vae_dir(str(out)) == str(out / "vae")
    assert resolve_vae_dir("Alpha-VLLM/Lumina-Image-2.0") == "Alpha-VLLM/Lumina-Image-2.0"


def test_resolve_vae_dir_accepts_a_pickle_checkpoint(tmp_path):
    """`save_pretrained(safe_serialization=False)` writes a .bin; that is still a real decoder."""
    d = tmp_path / "vae"
    d.mkdir()
    tiny_vae().save_pretrained(str(d), safe_serialization=False)
    assert os.path.exists(d / "diffusion_pytorch_model.bin")
    assert resolve_vae_dir(str(d)) == str(d)


def test_build_vae_names_a_weightless_directory(tmp_path):
    """Exactly what the failed gate left behind: a config and nothing else."""
    d = tmp_path / "vae"
    d.mkdir()
    json.dump({"_class_name": "AutoencoderKL"}, open(d / "config.json", "w"))
    with pytest.raises(SystemExit, match=re.escape(str(d))):
        build_vae(str(d))
