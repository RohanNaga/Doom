"""The IDM judge lives in SD 1.x latent space, so a 16-channel row has to round-trip to reach it.

`train_idm.IDM`'s encoder opens with `nn.Conv2d(4, ...)`, so it can only read 4-channel SD
latents. The SD 3.5 row's rollouts carry 16 channels, and `rollout_eval.py --score` used to hand
the raw rollout latents straight to `IDM.predict_sequence`. These checks pin the round trip that
fixes that: decode the rollout with the row's own decoder, re-encode the decoded frames with the
SD 1.x encoder the IDM was trained on, and only then judge.

Everything here runs on the CPU with random weights and random data, so no number is a
measurement; the shapes, the normalisation and the refusal are what is being gated.

    python -m pytest paper/fixtures/test_idm_reencode.py -q
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import rollout_eval  # noqa: E402
from doomdit_utils import LATENT_SCALE, encode_for_idm  # noqa: E402

LAT_H, LAT_W = 32, 40


@pytest.fixture(autouse=True)
def lpips_available(monkeypatch):
    """`do_score` imports lpips for the drift curves, which are not what this file gates.

    Where the package is installed the real one runs; where it is not (the laptop environment the
    CPU gates run in) a stand-in with the same call shape is inserted, so the IDM path is still
    exercised end to end rather than skipped.
    """
    try:
        import lpips  # noqa: F401
        return
    except ImportError:
        pass
    import types

    class _Stub(torch.nn.Module):
        def __init__(self, net="alex", verbose=False):
            super().__init__()

        def forward(self, a, b):
            return (a - b).abs().flatten(1).mean(1).view(-1, 1, 1, 1)

    module = types.ModuleType("lpips")
    module.LPIPS = _Stub
    monkeypatch.setitem(sys.modules, "lpips", module)


def tiny_sd_encoder():
    """A random 4-channel f8 `AutoencoderKL` standing in for sd-vae-ft-mse."""
    from diffusers.models import AutoencoderKL
    torch.manual_seed(1)
    return AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                         up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(4, 4, 4, 4),
                         layers_per_block=1, norm_num_groups=2, sample_size=256, latent_channels=4,
                         scaling_factor=LATENT_SCALE).eval()


def tiny_sd35_autoencoder():
    """A random 16-channel f8 `AutoencoderKL` carrying SD 3.5's latent contract."""
    from diffusers.models import AutoencoderKL
    torch.manual_seed(2)
    return AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                         up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(4, 4, 4, 4),
                         layers_per_block=1, norm_num_groups=2, sample_size=256, latent_channels=16,
                         scaling_factor=1.5305, shift_factor=0.0609).eval()


# ---- the encoder helper ----------------------------------------------------------------------

def test_encode_for_idm_pads_after_the_normalisation_and_scales_the_posterior_mean():
    sd = tiny_sd_encoder()
    frames = torch.rand(3, 3, 240, 320)
    got = encode_for_idm(sd, frames, "cpu", batch=2)
    # the contract encode_parquet.py writes: [-1, 1] first, then zero rows to 256, then mean * scale
    x = torch.nn.functional.pad(frames * 2 - 1, (0, 0, 0, 16))
    with torch.no_grad():
        want = sd.encode(x).latent_dist.mean * LATENT_SCALE
    assert got.shape == (3, 4, LAT_H, LAT_W)
    assert torch.allclose(got, want, atol=1e-6)


def test_encode_for_idm_pads_with_zeros_not_with_the_decoders_own_rows():
    """Padding before the [-1, 1] shift would write -1 into the pad rows, which is not what the
    IDM's training latents carry there. A mid-grey frame separates the two orders."""
    sd = tiny_sd_encoder()
    grey = torch.full((1, 3, 240, 320), 0.5)
    correct = encode_for_idm(sd, grey, "cpu")
    x = torch.nn.functional.pad(grey, (0, 0, 0, 16)) * 2 - 1      # the wrong order
    with torch.no_grad():
        wrong = sd.encode(x).latent_dist.mean * LATENT_SCALE
    assert not torch.allclose(correct, wrong, atol=1e-4)


def test_the_video_evaluator_and_the_rollout_scorer_share_one_encoder_helper():
    import eval_video
    sd = tiny_sd_encoder()
    frames = torch.rand(2, 3, 240, 320)
    assert torch.equal(eval_video.sd_encode(sd, frames, "cpu", 2), encode_for_idm(sd, frames, "cpu", 2))


# ---- the scorer ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_idm(tmp_path_factory):
    """A 4-channel IDM checkpoint with train_idm.py's own save schema."""
    from train_idm import IDM
    torch.manual_seed(3)
    num_actions, window = 3, 4
    idm = IDM(num_actions=num_actions, width=8, window=window, depth=1, heads=2)
    path = tmp_path_factory.mktemp("idm") / "idm.pt"
    torch.save({"model": idm.state_dict(), "num_actions": num_actions, "width": 8, "window": window,
                "depth": 1, "heads": 2, "act2mov": {0: 0, 1: 1, 2: 1}, "classes": ["none", "move"],
                "val_top1": 0.5, "val_movement": 0.6, "val_metrics": {"majority_baseline": 0.4}}, str(path))
    return str(path)


@pytest.fixture(scope="module")
def sd35_rollouts(tmp_path_factory):
    """A 16-channel rollout file with rollout_eval.do_rollout's keys, and the decoder that reads it."""
    d = tmp_path_factory.mktemp("sd35_roll")
    rng = np.random.RandomState(0)
    n, h, ctx = 2, 3, 4
    np.savez(d / "rollouts.npz",
             pred=rng.randn(n, h, 16, LAT_H, LAT_W).astype(np.float16) * 0.5,
             gt=rng.randn(n, h, 16, LAT_H, LAT_W).astype(np.float16) * 0.5,
             seed=rng.randn(n, ctx, 16, LAT_H, LAT_W).astype(np.float16) * 0.5,
             actions=rng.randint(0, 3, (n, h)).astype(np.int64),
             episode=np.arange(n), map=np.ones(n, np.int64), start=np.zeros(n, np.int64),
             config=json.dumps({"resolved_latent_channels": 16}))
    vae_dir = d / "vae"
    tiny_sd35_autoencoder().save_pretrained(str(vae_dir))
    sd_dir = d / "sd_vae"
    tiny_sd_encoder().save_pretrained(str(sd_dir))
    return str(d / "rollouts.npz"), str(vae_dir), str(sd_dir)


def _score_args(rollouts, vae_dir, out_dir, **kw):
    argv = ["--score", "--rollouts", rollouts, "--vae-path", vae_dir, "--out-dir", out_dir,
            "--latent-scale", "1.5305", "--latent-shift", "0.0609", "--decode-batch", "2",
            "--save-clips", "0"]
    for k, v in kw.items():
        argv += [k] if v is True else [k, str(v)]
    return rollout_eval.build_parser().parse_args(argv)


def test_sixteen_channel_scoring_refuses_the_idm_without_the_round_trip(sd35_rollouts, tiny_idm, tmp_path):
    rollouts, vae_dir, _ = sd35_rollouts
    args = _score_args(rollouts, vae_dir, str(tmp_path / "no_reencode"), **{"--idm": tiny_idm})
    with pytest.raises(SystemExit) as e:
        rollout_eval.do_score(args)
    assert "16" in str(e.value) and "--idm-reencode-vae" in str(e.value)


def test_sixteen_channel_scoring_runs_the_idm_through_the_round_trip(sd35_rollouts, tiny_idm, tmp_path):
    rollouts, vae_dir, sd_dir = sd35_rollouts
    out = str(tmp_path / "reencode")
    knobs = {"--idm": tiny_idm, "--idm-reencode-vae": sd_dir}
    rollout_eval.do_score(_score_args(rollouts, vae_dir, out, **knobs))
    drift = json.load(open(os.path.join(out, "drift.json")))
    assert drift["num_rollouts"] == 2 and len(drift["psnr"]) == 3
    for k in ("idm_top1", "idm_movement", "idm_real_top1", "idm_real_movement"):
        assert len(drift[k]) == 3, k
        assert all(0.0 <= v <= 1.0 for v in drift[k]), k
        assert 0.0 <= drift[f"{k}_mean"] <= 1.0, k
    # the round trip is recorded next to the numbers it produced, so a table never mixes the two
    assert drift["idm_reencode_vae"] == sd_dir


def test_four_channel_scoring_still_reaches_the_idm_directly(tmp_path):
    """The SD rows must be untouched: no round trip, no refusal, same code path as before."""
    from train_idm import IDM
    torch.manual_seed(4)
    idm = IDM(num_actions=3, width=8, window=4, depth=1, heads=2)
    idm_path = tmp_path / "idm.pt"
    torch.save({"model": idm.state_dict(), "num_actions": 3, "width": 8, "window": 4, "depth": 1,
                "act2mov": {0: 0, 1: 1, 2: 1}, "classes": ["none", "move"]}, str(idm_path))
    rng = np.random.RandomState(1)
    n, h, ctx = 2, 3, 4
    roll = tmp_path / "rollouts.npz"
    np.savez(roll, pred=rng.randn(n, h, 4, LAT_H, LAT_W).astype(np.float16) * 0.5,
             gt=rng.randn(n, h, 4, LAT_H, LAT_W).astype(np.float16) * 0.5,
             seed=rng.randn(n, ctx, 4, LAT_H, LAT_W).astype(np.float16) * 0.5,
             actions=rng.randint(0, 3, (n, h)).astype(np.int64))
    vae_dir = tmp_path / "vae"
    tiny_sd_encoder().save_pretrained(str(vae_dir))
    out = str(tmp_path / "metrics")
    rollout_eval.do_score(rollout_eval.build_parser().parse_args(
        ["--score", "--rollouts", str(roll), "--vae-path", str(vae_dir), "--out-dir", out,
         "--idm", str(idm_path), "--decode-batch", "2", "--save-clips", "0"]))
    drift = json.load(open(os.path.join(out, "drift.json")))
    assert drift["idm_reencode_vae"] is None
    assert len(drift["idm_top1"]) == 3
