"""The directional check and the motion ratio of review 7.2 (`directional_check.py`).

Stand-in models run through the real sampler and a toy decoder on a synthetic per-tic corpus whose
frames scroll one latent column (8 decoded pixels) per tic in the direction of the turn bit:

  * a model that scrolls the last context frame the way the newest control's turn bit says must be
    directionally correct on every window, with left turns positive and right turns negative;
  * a model that ignores the controls must score zero, because the swap cannot change its output;
  * an identity (persistence) model must have a motion ratio of zero, and the turning model one;
  * the shift estimator must recover known synthetic shifts, integer and fractional, within a pixel.

The CLI runs end to end on a tiny PixArt on the CPU, as `test_gate_probes.py` does.

    python -m pytest paper/fixtures/test_directional_check.py -q
"""
import json
import os
import re
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import backbones  # noqa: E402
import directional_check as dc  # noqa: E402
from diffusion_v import VDiffusion  # noqa: E402
from doom_data import TicWindowDataset  # noqa: E402
from pertic_fixtures import held_actions, write_pertic_episode  # noqa: E402

L = 4                       # context tics
COLS = 1                    # latent columns scrolled per turning tic: 8 decoded pixels
PX = 8 * COLS
BITS = 19
LEFT = "0010000000000000000"
RIGHT = "0001000000000000000"
FORWARD = "1000000000000000000"
LEFT_FORWARD = "1010000000000000000"      # a turn with forward motion is still a turning window
LEFT_STRAFE = "0010100000000000000"       # a strafe mixes translation into the shift: excluded
BOTH = "0011000000000000000"              # both turn bits cancel: excluded
TINY_PIXART = {"num_attention_heads": 2, "attention_head_dim": 8, "in_channels": 4, "out_channels": 8,
               "num_layers": 2, "caption_channels": 32, "sample_size": 64, "patch_size": 2,
               "cross_attention_dim": 16, "use_additional_conditions": False, "norm_num_groups": 2}


def turn_of(btn):
    """+1 when the left bit is held without the right one, -1 for the reverse, else 0."""
    return int(btn[2] == "1") - int(btn[3] == "1")


def texture(rng, channels):
    z = rng.randn(channels, 32, 40)
    for _ in range(2):
        z = (z + np.roll(z, 1, -1) + np.roll(z, -1, -1) + np.roll(z, 1, -2) + np.roll(z, -1, -2)) / 5
    return z / z.std()


def turning_corpus(path, ids, channels=4, T=96, seed=0):
    """Per-tic episodes whose frame t+1 is frame t scrolled by the turn bit of row t (row semantics)."""
    rng = np.random.RandomState(seed)
    for ep in ids:
        blocks = rng.choice([LEFT, RIGHT, FORWARD, LEFT_FORWARD, LEFT_STRAFE, BOTH], size=T // 4)
        btns = [str(b) for b in blocks for _ in range(4)]
        z, lat = texture(rng, channels), []
        for t in range(T):
            lat.append(z)
            z = np.roll(z, turn_of(btns[t]) * COLS, axis=-1)
        write_pertic_episode(str(path), ep, held_actions([0] * (T // 4)), buttons=btns, channels=channels)
        np.save(os.path.join(str(path), f"ep_{ep:05d}_latents.npy"), np.stack(lat).astype(np.float16))
    return str(path)


def dataset(path, ids, channels=4):
    return TicWindowDataset(path, ids, L, latent_channels=channels, horizon=dc.MOTION_HORIZON,
                            with_horizon=True, action_history=L)


def toy_decode(z):
    """(B, C, 32, 40) latents -> (B, 3, 240, 320) in [0, 1]; a latent column is 8 pixels."""
    img = F.interpolate(z[:, :3].float(), scale_factor=8, mode="bilinear", align_corners=False)
    return torch.sigmoid(img[:, :, :240])


class StandIn(torch.nn.Module):
    """A world model whose sample is a chosen function of the last context latent.

    It returns the velocity that makes the sampler's x0 estimate exactly that latent at every step,
    so the real DDIM loop, the fixed noise and the control plumbing are all exercised.
      follow    scroll by the newest control's turn bit (the ground-truth dynamics of the corpus)
      ignore    always scroll right, whatever the control says
      identity  copy the last context frame (persistence)
    """

    def __init__(self, channels, mode):
        super().__init__()
        self.c, self.mode = channels, mode
        self.diff = VDiffusion(objective="v")
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, xt, t, act, ctx, bucket, phase=None):
        last = ctx[:, -self.c:].float()
        if self.mode == "identity":
            x0 = last
        elif self.mode == "ignore":
            x0 = torch.roll(last, COLS, dims=-1)
        else:
            d = (act[:, -1, dc.TURN_LEFT] - act[:, -1, dc.TURN_RIGHT]).round().long().tolist()
            x0 = torch.stack([torch.roll(last[i], d[i] * COLS, dims=-1) for i in range(len(d))])
        a, s = self.diff._coef(t, xt.ndim)
        return (a * xt - x0) / s


def run(tmp_path, mode, per_direction=8, channels=4):
    d = turning_corpus(tmp_path / "val", (6000, 6001), channels=channels)
    ds = dataset(d, [6000, 6001], channels)
    windows, available = dc.turning_windows(ds, per_direction, seed=0)
    records = dc.run_windows(StandIn(channels, mode), toy_decode, ds, windows, latent_channels=channels,
                             diffusion=VDiffusion(objective="v"), steps=3, seed=0, batch_size=5, device="cpu")
    return dc.summarize(records), records, available


# ---------------------------------------------------------------------------------------
# window selection and the swap
# ---------------------------------------------------------------------------------------

def test_turn_direction_needs_exactly_one_turn_bit_and_no_strafe():
    right_strafe = "0001010000000000000"
    rows = np.array([[float(c) for c in s]
                     for s in (LEFT, RIGHT, FORWARD, LEFT_FORWARD, LEFT_STRAFE, BOTH, right_strafe)])
    assert dc.turn_direction(rows).tolist() == [1, -1, 0, 1, 0, 0, 0]
    assert dc.EXPECTED_SIGN == {"left": 1, "right": -1}


def test_turning_windows_are_chosen_by_the_newest_control(tmp_path):
    d = turning_corpus(tmp_path / "val", (6000, 6001))
    ds = dataset(d, [6000, 6001])
    windows, available = dc.turning_windows(ds, 6, seed=3)
    for direction, sign in (("left", 1), ("right", -1)):
        idx = windows[direction]
        assert len(idx) == 6 == len(set(idx.tolist())) and available[direction] >= 6
        for g in idx.tolist():
            newest = ds[g][2][0][-1].numpy()           # control applied from the last context frame
            assert dc.turn_direction(newest[None])[0] == sign
    again, _ = dc.turning_windows(ds, 6, seed=3)
    assert all(np.array_equal(windows[k], again[k]) for k in windows)
    with pytest.raises(ValueError, match="turning windows"):
        dc.turning_windows(ds, 10_000, seed=0)


def test_the_swap_touches_only_the_newest_turn_bits():
    act = torch.from_numpy(np.random.RandomState(0).randint(0, 2, (3, L, BITS)).astype(np.float32))
    act[:, -1, dc.TURN_LEFT], act[:, -1, dc.TURN_RIGHT] = 1.0, 0.0
    out = dc.swap_turns(act)
    assert out[:, -1, dc.TURN_LEFT].eq(0).all() and out[:, -1, dc.TURN_RIGHT].eq(1).all()
    keep = torch.ones_like(act, dtype=torch.bool)
    keep[:, -1, [dc.TURN_LEFT, dc.TURN_RIGHT]] = False
    assert torch.equal(out[keep], act[keep])
    assert act[:, -1, dc.TURN_LEFT].eq(1).all(), "the swap must not modify its input"


# ---------------------------------------------------------------------------------------
# the shift estimator
# ---------------------------------------------------------------------------------------

def scene(shift, rng_seed=0, H=240, W=320):
    """An analytic textured frame scrolled right by `shift` pixels, with a static HUD and weapon."""
    rng = np.random.RandomState(rng_seed)
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    f = np.zeros((H, W))
    for _ in range(12):
        fx, fy = rng.uniform(1 / 60, 1 / 8), rng.uniform(-1 / 40, 1 / 40)
        f += rng.uniform(0.5, 1.0) * np.sin(2 * np.pi * (fx * (x - shift) + fy * y) + rng.uniform(0, 2 * np.pi))
    img = 0.5 + 0.08 * f
    img[H - 32:] = 0.3 + 0.1 * np.sin(x[H - 32:] / 5)       # the HUD never moves
    img[150:H - 32, 130:190] = 0.9                          # nor does the weapon sprite
    rgb = np.stack([img, 0.9 * img, 0.8 * img])
    return torch.from_numpy(rgb).float().clamp(0, 1).unsqueeze(0)


@pytest.mark.parametrize("shift", [-17, -5.5, 0, 3.25, 9, 12.6, 22])
def test_the_estimator_recovers_a_known_shift_within_a_pixel(shift):
    a = scene(0.0)
    b = scene(shift)
    s, peak = dc.horizontal_shift(a, b)
    assert abs(float(s[0]) - shift) < 1.0 and float(peak[0]) > 0.9
    # a blurred, lower-contrast prediction is what a 10-step sampler hands the estimator
    soft = 0.2 + 0.6 * F.avg_pool2d(b, 5, stride=1, padding=2)
    s2, _ = dc.horizontal_shift(a, soft)
    assert abs(float(s2[0]) - shift) < 1.0


def test_the_estimator_is_zero_on_identical_frames_and_undefined_on_flat_ones():
    a = scene(0.0, rng_seed=4)
    s, _ = dc.horizontal_shift(torch.cat([a, a]), torch.cat([a, a]))
    assert s.abs().max() == 0
    flat = torch.full_like(a, 0.4)
    s, peak = dc.horizontal_shift(flat, a)
    assert torch.isnan(s).all() and torch.isnan(peak).all()


def test_the_estimator_reads_a_decoded_latent_column_as_eight_pixels():
    rng = np.random.RandomState(1)
    z = torch.from_numpy(texture(rng, 4)).float().unsqueeze(0)
    s, _ = dc.horizontal_shift(toy_decode(z), toy_decode(torch.roll(z, 2, dims=-1)))
    assert abs(float(s[0]) - 16) < 1.0


# ---------------------------------------------------------------------------------------
# the check on stand-in models
# ---------------------------------------------------------------------------------------

def test_a_model_that_turns_with_the_control_is_directionally_correct(tmp_path):
    summary, records, _ = run(tmp_path, "follow")
    assert summary["correct_frac"] == 1.0 and summary["ref_frac"] == 1.0
    left, right = summary["per_direction"]["left"], summary["per_direction"]["right"]
    assert abs(left["recorded"]["mean"] - PX) < 1 and abs(left["swapped"]["mean"] + PX) < 1
    assert abs(right["recorded"]["mean"] + PX) < 1 and abs(right["swapped"]["mean"] - PX) < 1
    assert abs(left["reference"]["median"] - PX) < 1 and abs(right["reference"]["median"] + PX) < 1
    for d in (left, right):
        assert d["windows"] == 8 and d["correct_frac"] == 1.0
        assert d["recorded"]["expected_sign_frac"] == 1.0 and d["swapped"]["expected_sign_frac"] == 1.0
        assert d["persistence"] == {"mean": 0.0, "median": 0.0, "expected_sign_frac": 0.0, "correct_frac": 0.0}
    assert len(records) == 16 and {r["direction"] for r in records} == {"left", "right"}
    # the model reproduces the corpus dynamics, so it moves exactly as much as the real frames
    assert summary["motion"]["ratio"] == pytest.approx(1.0, abs=1e-3)
    assert set(summary["motion"]["per_horizon"]) == {"1", "2", "3", "4"}


def test_a_model_that_ignores_controls_scores_zero(tmp_path):
    summary, records, _ = run(tmp_path, "ignore")
    assert summary["correct_frac"] == 0.0
    assert summary["ref_frac"] == 1.0, "the reference measures the data, not the model"
    assert all(r["shift_recorded"] == r["shift_swapped"] for r in records)
    assert summary["per_direction"]["left"]["recorded"]["expected_sign_frac"] == 1.0
    assert summary["per_direction"]["right"]["recorded"]["expected_sign_frac"] == 0.0


def test_an_identity_model_does_not_move(tmp_path):
    summary, records, _ = run(tmp_path, "identity")
    assert summary["motion"]["ratio"] == pytest.approx(0.0, abs=1e-6)
    assert summary["motion"]["persistence_ratio"] == 0.0
    assert summary["motion"]["mean_real"] > 0
    assert summary["correct_frac"] == 0.0
    assert all(r["shift_recorded"] == 0 for r in records)


class FakeRaw:
    """`eval_tf.RawFrames` over the corpus: the uint8 frame of a tic, and a log of the tics asked for."""

    def __init__(self, path):
        self.path, self.asked = path, []

    def get(self, episode_id, tic):
        self.asked.append((episode_id, tic))
        lat = np.load(os.path.join(self.path, f"ep_{episode_id:05d}_latents.npy"))
        img = toy_decode(torch.from_numpy(lat[tic].astype(np.float32))[None])[0]
        return (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def test_the_raw_reference_reads_the_last_context_tic_and_the_target_tic(tmp_path):
    d = turning_corpus(tmp_path / "val", (6000, 6001))
    ds = dataset(d, [6000, 6001])
    windows, _ = dc.turning_windows(ds, 4, seed=0)
    raw = FakeRaw(d)
    records = dc.run_windows(StandIn(4, "follow"), toy_decode, ds, windows, latent_channels=4,
                             diffusion=VDiffusion(objective="v"), steps=2, seed=0, batch_size=8, device="cpu", raw=raw)
    for r in records:
        assert r["target_tic"] == r["start"] + L
        assert (r["episode"], r["target_tic"] - 1) in raw.asked and (r["episode"], r["target_tic"]) in raw.asked
    s = dc.summarize(records)
    assert s["ref_raw_frac"] == 1.0
    assert abs(s["per_direction"]["left"]["reference_raw"]["mean"] - PX) < 1
    assert abs(s["per_direction"]["right"]["reference_raw"]["mean"] + PX) < 1


def test_the_summary_line_is_greppable():
    line = dc.summary_line("040-unet-nexttic", 10000, {"correct_frac": 0.8125, "ref_frac": 0.96875})
    assert line == "DIRECTIONAL_CHECK 040-unet-nexttic 10000 correct_frac=0.8125 ref_frac=0.9688"


# ---------------------------------------------------------------------------------------
# the CLI on a tiny PixArt
# ---------------------------------------------------------------------------------------

class ToyVAE:
    def decode(self, z):
        img = F.interpolate(z[:, :3].float(), scale_factor=8, mode="bilinear", align_corners=False)
        return types.SimpleNamespace(sample=torch.tanh(img))


@pytest.fixture
def tiny_pixart(monkeypatch):
    from diffusers import PixArtTransformer2DModel as P
    monkeypatch.setattr(P, "from_pretrained", classmethod(lambda cls, *a, **k: cls(**TINY_PIXART)))
    monkeypatch.setattr(P, "load_config", classmethod(lambda cls, *a, **k: dict(TINY_PIXART)))
    monkeypatch.setattr(dc, "build_vae", lambda *a, **k: ToyVAE())


def checkpoint(tmp_path, dead_controls=False, **args):
    from diffusers import PixArtTransformer2DModel
    torch.manual_seed(0)
    m = backbones.PixArtWorldModel(num_actions=3, context_frames=L, noise_buckets=4, action_dropout=0.0,
                                   grad_ckpt=False, action_history=L, control_bits=BITS,
                                   transformer=PixArtTransformer2DModel(**TINY_PIXART))
    if dead_controls:
        with torch.no_grad():
            m.control_history.mlp[2].weight.zero_()
            m.control_history.mlp[2].bias.zero_()
    ck = {"model": {k: v.detach().clone() for k, v in m.state_dict().items()},
          "ema": {n: p.detach().clone() for n, p in m.named_parameters()}, "step": 10000,
          "args": {"action_history": L, "objective": "v", "action_dropout": 0.0, "tic_stride": 1,
                   "resolved_control_bits": BITS, **args}}
    run_dir = tmp_path / "040-unet-nexttic"
    run_dir.mkdir(exist_ok=True)
    path = run_dir / "snap_0010000.pt"
    torch.save(ck, str(path))
    return str(path)


def cli(tmp_path, ckpt, *extra):
    d = turning_corpus(tmp_path / "val", (6000, 6001))
    split = tmp_path / "split_val.json"
    split.write_text(json.dumps({"val": [6000, 6001]}))
    out = tmp_path / "directional.json"
    a = dc.build_parser().parse_args(["--ckpt", ckpt, "--backbone", "pixart", "--pixart-path", backbones.PIXART_DEFAULT,
                                      "--latents-dir", d, "--split", str(split), "--subset", "val",
                                      "--latent-channels", "4", "--context-frames", str(L), "--num-actions", "3",
                                      "--noise-buckets", "4", "--windows", "3", "--steps", "2", "--batch-size", "4",
                                      "--device", "cpu", "--out", str(out), *extra])
    return a, out


def test_the_cli_writes_every_window_and_one_summary_line(tmp_path, tiny_pixart, capsys):
    a, out = cli(tmp_path, checkpoint(tmp_path), "--use-ema")
    assert dc.main(a) == 0
    rep = json.load(open(out))
    assert len(rep["windows"]) == 6 and rep["config"]["use_ema"] is True
    assert rep["config"]["step"] == 10000 and rep["config"]["run"] == "040-unet-nexttic"
    assert rep["config"]["motion_mode"].startswith("closed-loop")
    for k in ("shift_recorded", "shift_swapped", "shift_reference", "motion_pred", "motion_real"):
        assert k in rep["windows"][0]
    assert len(rep["windows"][0]["motion_pred"]) == dc.MOTION_HORIZON
    assert "decoder" in rep and "per_direction" in rep["summary"]
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("DIRECTIONAL_CHECK")]
    assert len(lines) == 1
    assert re.fullmatch(r"DIRECTIONAL_CHECK 040-unet-nexttic 10000 correct_frac=\S+ ref_frac=\S+", lines[0])


def test_dead_controls_give_identical_predictions_under_the_same_noise(tmp_path, tiny_pixart):
    """Recorded and swapped samples share their noise, so a model deaf to controls cannot flip."""
    a, out = cli(tmp_path, checkpoint(tmp_path, dead_controls=True))
    assert dc.main(a) == 0
    rep = json.load(open(out))
    rec = np.array([w["shift_recorded"] for w in rep["windows"]], dtype=float)
    swp = np.array([w["shift_swapped"] for w in rep["windows"]], dtype=float)
    assert np.array_equal(rec, swp, equal_nan=True) and rep["summary"]["correct_frac"] == 0.0


@pytest.mark.parametrize("args, message", [({"action_history": 0}, "executed-control history"),
                                           ({"tic_stride": 4}, "next-tic")])
def test_the_cli_refuses_checkpoints_it_cannot_check(tmp_path, tiny_pixart, args, message):
    a, _ = cli(tmp_path, checkpoint(tmp_path, **args))
    with pytest.raises(SystemExit, match=message):
        dc.main(a)


def test_the_documented_spiderman_commands_parse():
    """The GPU commands in the module docstring use only flags the parser accepts."""
    doc = dc.__doc__.replace("\\\n", " ")
    cmds = re.findall(r"CUDA_VISIBLE_DEVICES=\S+ \S+ directional_check\.py ([^\n]*)", doc)
    assert len(cmds) == 2
    seen = set()
    for c in cmds:
        a = dc.build_parser().parse_args(c.replace("$D", "/d").split())
        seen.add(a.backbone)
        assert a.steps == 10 and a.windows == 128 and a.subset == "val"
    assert seen == {"unet", "sd35"}
    sd35 = [c for c in cmds if "--backbone sd35" in c][0]
    for flag in ("--vae-path stabilityai/stable-diffusion-3.5-medium", "--vae-subfolder vae",
                 "--latent-scale 1.5305", "--latent-shift 0.0609", "--latent-channels 16"):
        assert flag in " ".join(sd35.split())
    unet = [c for c in cmds if "--backbone unet" in c][0]
    assert "--sd-path CompVis/stable-diffusion-v1-4" in " ".join(unet.split())
