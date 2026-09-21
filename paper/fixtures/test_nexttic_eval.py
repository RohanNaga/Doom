"""CPU gates for next-tic evaluation: the interface check, the floor, equal game time, rollouts.

Four things have to be right or a next-tic number is unreadable:

* the evaluator takes the frame spacing and the conditioning interface from the CHECKPOINT, so a
  next-tic model can never be scored on decision-spaced windows by accident;
* the floor is the RAW last frame against the RAW target, not the decoded last latent, which
  carries the decoder's own reconstruction error and is therefore not a persistence reference;
* `--horizon-tics 4` rolls four tics forward from real context, which is the only way a next-tic
  model and a next-decision model are compared over the same 114 ms of game time;
* a rollout's frame buffer and control buffer shift together, so step k sees exactly the pairs a
  teacher-forced window at step k would see.

    python -m pytest paper/fixtures/test_nexttic_eval.py -q
"""
import io
import json
import os
import sys
import types

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

os.environ.setdefault("ACCELERATE_USE_CPU", "1")

from pertic_fixtures import held_actions, write_pertic_episode  # noqa: E402

import backbones  # noqa: E402
import eval_tf  # noqa: E402
import rollout_eval  # noqa: E402
from doom_data import TicWindowDataset, control_matrix  # noqa: E402


@pytest.fixture(autouse=True)
def lpips_available(monkeypatch):
    """A stand-in where the package is absent, so the scoring path runs rather than skipping."""
    try:
        import lpips  # noqa: F401
        return
    except ImportError:
        pass

    class _Stub(torch.nn.Module):
        def __init__(self, net="alex", verbose=False):
            super().__init__()

        def forward(self, a, b):
            return (a - b).abs().flatten(1).mean(1).view(-1, 1, 1, 1)

    module = types.ModuleType("lpips")
    module.LPIPS = _Stub
    monkeypatch.setitem(sys.modules, "lpips", module)


# ---------------------------------------------------------------------------------------
# the interface is read from the checkpoint
# ---------------------------------------------------------------------------------------

def _args(**kw):
    ns = types.SimpleNamespace(ckpt="x.pt", tic_stride=None, action_history=None)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_a_checkpoint_without_the_flags_is_a_stride_four_single_action_model():
    """Every finished row predates these flags and must keep scoring exactly as it did."""
    got = eval_tf.checkpoint_interface({"args": {"objective": "v"}}, _args())
    assert got == {"tic_stride": 4, "action_history": 0, "phase_buckets": 0, "control_bits": 0}


def test_the_interface_comes_from_the_checkpoint():
    ck = {"args": {"tic_stride": 1, "action_history": 32, "phase_conditioning": True,
                   "phase_buckets": 5, "control_bits": 9}}
    assert eval_tf.checkpoint_interface(ck, _args()) == {"tic_stride": 1, "action_history": 32,
                                                         "phase_buckets": 5, "control_bits": 9}


def test_phase_buckets_are_zero_when_phase_conditioning_was_off():
    ck = {"args": {"tic_stride": 1, "phase_conditioning": False, "phase_buckets": 5}}
    assert eval_tf.checkpoint_interface(ck, _args())["phase_buckets"] == 0


@pytest.mark.parametrize("flag,value", [("tic_stride", 4), ("action_history", 0)])
def test_an_asserted_interface_that_disagrees_with_the_checkpoint_is_refused(flag, value):
    ck = {"args": {"tic_stride": 1, "action_history": 32}}
    with pytest.raises(SystemExit, match="disagrees with the checkpoint"):
        eval_tf.checkpoint_interface(ck, _args(**{flag: value}))


def test_an_asserted_interface_that_agrees_is_accepted():
    ck = {"args": {"tic_stride": 1, "action_history": 32}}
    assert eval_tf.checkpoint_interface(ck, _args(tic_stride=1, action_history=32))["tic_stride"] == 1


# ---------------------------------------------------------------------------------------
# per-window noise and the horizon-1 shim
# ---------------------------------------------------------------------------------------

def test_window_noise_is_paired_per_window_not_per_run():
    """Two runs that consume different numbers of random values still get the same noise per window."""
    a = eval_tf.window_noise((3, 4, 8, 10), [11, 22, 33])
    b = eval_tf.window_noise((3, 4, 8, 10), [11, 22, 33])
    assert torch.equal(a, b)
    assert not torch.allclose(a[0], a[1])
    # the noise of window 22 does not depend on which windows came with it
    assert torch.equal(eval_tf.window_noise((1, 4, 8, 10), [22])[0], a[1])


def test_the_horizon_one_shim_presents_a_three_tuple_dataset_as_one_step(tmp_path):
    class Three(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return torch.zeros(8, 2, 2), torch.full((4, 2, 2), float(i)), torch.tensor(7)

    ds = eval_tf.HorizonOne(Three())
    ctx, tgts, acts, phases = ds[1]
    assert tgts.shape == (1, 4, 2, 2) and float(tgts[0, 0, 0, 0]) == 1.0
    assert acts.shape == (1,) and int(acts[0]) == 7
    assert phases.shape == (1,) and int(phases[0]) == 0


# ---------------------------------------------------------------------------------------
# eval_tf end to end on a per-tic corpus
# ---------------------------------------------------------------------------------------

CTX, BITS = 4, 9


def _png(rgb):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _write_parquet(path, tics, rng):
    """A recording with just the columns `eval_tf.RawFrames` reads, and real 320x240 PNG frames."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    frames = [_png(rng.randint(0, 255, (240, 320, 3), dtype=np.uint8)) for _ in tics]
    pq.write_table(pa.table({"tic": pa.array(np.asarray(tics), pa.int32()),
                             "frame": pa.array(frames, pa.binary())}), path)


def _tiny_vae(path, channels=4):
    from diffusers.models import AutoencoderKL
    AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                  up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(4, 4, 4, 4),
                  layers_per_block=1, norm_num_groups=2, sample_size=256,
                  latent_channels=channels, scaling_factor=0.18215).eval().save_pretrained(str(path))
    return str(path)


TINY_PIXART = dict(num_attention_heads=2, attention_head_dim=8, in_channels=4, out_channels=8,
                   num_layers=2, caption_channels=32, sample_size=64, patch_size=2,
                   cross_attention_dim=16, use_additional_conditions=False, norm_num_groups=2)


@pytest.fixture
def tiny_hub(monkeypatch):
    """Serve a 16k-parameter transformer wherever PixArt would pull the 611M checkpoint.

    PixArt rather than the DiT because `build_model` has no model-size knob: it would build the real
    DiT-XL/2, which is 673M parameters on the CPU.
    """
    from diffusers import PixArtTransformer2DModel as P
    monkeypatch.setattr(P, "from_pretrained", classmethod(lambda cls, *a, **k: cls(**TINY_PIXART)))
    monkeypatch.setattr(P, "load_config", classmethod(lambda cls, *a, **k: dict(TINY_PIXART)))


def _tiny_model(action_history=0, control_bits=0):
    from diffusers import PixArtTransformer2DModel
    torch.manual_seed(0)
    return backbones.PixArtWorldModel(num_actions=3, context_frames=CTX, noise_buckets=4,
                                      action_dropout=0.0, grad_ckpt=False,
                                      action_history=action_history, control_bits=control_bits,
                                      transformer=PixArtTransformer2DModel(**TINY_PIXART))


@pytest.fixture
def pertic_eval_corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("pertic_eval")
    rng = np.random.RandomState(0)
    T = 24
    lat_dir = str(d / "latents")
    for ep in (0, 1):
        btns = ["".join(str(int(b)) for b in rng.randint(0, 2, BITS)) for _ in range(T)]
        # action ids must stay inside --num-actions 3, or the action table lookup is out of range
        write_pertic_episode(lat_dir, ep, held_actions([0, 1, 2, 0, 1, 2]), buttons=btns)
        _write_parquet(os.path.join(str(d), f"ep_{ep:05d}.parquet"), np.arange(T), rng)
    split = str(d / "split.json")
    json.dump({"train": [0], "val": [1]}, open(split, "w"))
    return lat_dir, str(d), split, str(d)


def _run_eval(tmp_path, corpus, ckpt_args, extra):
    lat_dir, parquet_dir, split, _ = corpus
    model = _tiny_model(ckpt_args.get("action_history", 0), ckpt_args.get("control_bits", 0))
    ck = str(tmp_path / "best.pt")
    torch.save({"model": model.state_dict(), "step": 7,
                "args": {"action_dropout": 0.0, "objective": "v", **ckpt_args}}, ck)
    out = str(tmp_path / "out")
    eval_tf.main(eval_tf.build_parser().parse_args(
        ["--ckpt", ck, "--backbone", "pixart", "--pixart-path", backbones.PIXART_DEFAULT,
         "--latent-channels", "4",
         "--vae-path", _tiny_vae(tmp_path / "vae"),
         "--latents-dir", lat_dir, "--parquet-dir", parquet_dir, "--split", split, "--subset", "val",
         "--context-frames", str(CTX), "--num-actions", "3", "--noise-buckets", "4",
         "--num-windows", "4", "--batch-size", "2", "--steps", "2", "--save-images", "0",
         "--out-dir", out] + extra))
    return json.load(open(os.path.join(out, "metrics.json")))


def test_a_per_tic_checkpoint_is_scored_on_per_tic_windows(pertic_eval_corpus, tiny_hub, tmp_path):
    m = _run_eval(tmp_path, pertic_eval_corpus, {"tic_stride": 1}, [])
    assert m["config"]["checkpoint_interface"]["tic_stride"] == 1
    assert m["config"]["horizon_tics"] == 1 and m["config"]["game_time_tics"] == 1
    assert m["psnr_dec"]["n"] == 4 and np.isfinite(m["psnr_dec"]["mean"])
    assert m["config"]["window_validity"]["excluded_fraction"] == 0.0


def test_the_floor_is_decoder_independent(pertic_eval_corpus, tiny_hub, tmp_path):
    """`persist_psnr_raw` is raw-against-raw; `copy_psnr_raw` decodes first and is kept only for
    continuity with the stride-4 rows."""
    m = _run_eval(tmp_path, pertic_eval_corpus, {"tic_stride": 1}, [])
    assert m["persist_psnr_raw"] is not None and np.isfinite(m["persist_psnr_raw"]["mean"])
    assert m["persist_lpips_raw"] is not None and m["persist_hud_psnr_raw"] is not None
    assert m["copy_psnr_raw"] is not None
    # the two are different quantities: the decoded one passes through the autoencoder
    assert m["persist_psnr_raw"]["mean"] != m["copy_psnr_raw"]["mean"]
    assert m["vae_psnr"] is not None, "the ceiling must still be reported"


def test_the_phase_breakdown_covers_the_held_action_run(pertic_eval_corpus, tiny_hub, tmp_path):
    m = _run_eval(tmp_path, pertic_eval_corpus, {"tic_stride": 1}, ["--num-windows", "12"])
    per = m["per_tics_since_decision"]
    assert per, "no breakdown by tics_since_decision"
    assert all(int(k) in (0, 1, 2, 3, 4) for k in per)
    assert sum(v["windows"] for v in per.values()) == m["psnr_dec"]["n"]
    for v in per.values():
        assert "psnr_dec" in v and "persist_psnr_raw" in v


def test_four_tics_forward_is_scored_four_tics_later(pertic_eval_corpus, tiny_hub, tmp_path):
    """Equal game time against a stride-4 model's single step."""
    m = _run_eval(tmp_path, pertic_eval_corpus, {"tic_stride": 1}, ["--horizon-tics", "4"])
    assert m["config"]["horizon_tics"] == 4 and m["config"]["game_time_tics"] == 4
    assert m["config"]["window_validity"]["horizon"] == 4
    assert np.isfinite(m["psnr_dec"]["mean"]) and np.isfinite(m["persist_psnr_raw"]["mean"])


def test_the_horizon_floor_is_the_gap_k_floor(pertic_eval_corpus, tiny_hub, tmp_path):
    """The floor at gap 4 must be the frame four tics earlier, not the frame one tic earlier."""
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    one = _run_eval(tmp_path / "a", pertic_eval_corpus, {"tic_stride": 1}, [])
    four = _run_eval(tmp_path / "b", pertic_eval_corpus, {"tic_stride": 1}, ["--horizon-tics", "4"])
    assert one["persist_psnr_raw"]["mean"] != four["persist_psnr_raw"]["mean"]


def test_a_horizon_beyond_one_is_refused_for_a_stride_four_checkpoint(pertic_eval_corpus, tiny_hub, tmp_path):
    with pytest.raises(SystemExit, match="only means something for a next-tic model"):
        _run_eval(tmp_path, pertic_eval_corpus, {"tic_stride": 4}, ["--horizon-tics", "4"])


def test_a_control_history_checkpoint_is_scored_with_control_history(pertic_eval_corpus, tiny_hub, tmp_path):
    m = _run_eval(tmp_path, pertic_eval_corpus,
                  {"tic_stride": 1, "action_history": CTX, "control_bits": BITS}, [])
    assert m["config"]["checkpoint_interface"]["action_history"] == CTX
    assert np.isfinite(m["psnr_dec"]["mean"])
    # a control vector has no single id, so the per-window `action` column records -1 rather than
    # a meaningless int() of a tensor
    import csv
    rows = list(csv.DictReader(open(os.path.join(str(tmp_path / "out"), "per_window.csv"))))
    assert all(r["action"] == "-1" for r in rows)


# ---------------------------------------------------------------------------------------
# rollout window selection and the (frame, control) parity
# ---------------------------------------------------------------------------------------

def test_a_per_tic_rollout_window_never_crosses_a_respawn(tmp_path):
    d = str(tmp_path / "lat")
    deaths = np.array([0] * 12 + [1] * 12)
    write_pertic_episode(d, 0, held_actions([1] * 6), deaths=deaths)
    picks = rollout_eval.collect_rollout_windows(d, [0], L=4, H=4, n=20, seed=0, tic_stride=1)
    for _, _, s, _, _ in picks:
        assert len(set(deaths[s:s + 8].tolist())) == 1, f"window at {s} spans the respawn"


def test_a_per_tic_rollout_ignores_chain_ids(tmp_path):
    """Two -1 endpoints satisfy the stride-4 chain test; the per-tic contract does not use it."""
    d = str(tmp_path / "lat2")
    deaths = np.array([0] * 12 + [1] * 12)
    write_pertic_episode(d, 0, held_actions([1] * 6), deaths=deaths,
                         decisions=np.zeros(24, dtype=bool))
    meta = np.load(os.path.join(d, "ep_00000_meta.npz"))
    assert (meta["chain_id"] == -1).all()
    picks = rollout_eval.collect_rollout_windows(d, [0], L=4, H=4, n=20, seed=0, tic_stride=1)
    assert picks and all(len(set(deaths[s:s + 8].tolist())) == 1 for _, _, s, _, _ in picks)


def test_a_per_tic_rollout_refuses_a_stride_four_corpus(tmp_path):
    from pertic_fixtures import write_stride4_episode
    d = str(tmp_path / "s4")
    write_stride4_episode(d, 0, list(range(16)))
    with pytest.raises(ValueError, match="is_decision"):
        rollout_eval.collect_rollout_windows(d, [0], L=4, H=4, n=2, seed=0, tic_stride=1)


def test_the_control_buffer_shifts_with_the_frame_buffer(tmp_path):
    """The parity Astra asked for: step k's (frames, controls) pair equals the teacher-forced one.

    A rollout that shifted only the frame buffer would keep feeding the newest control of step 0
    while the context advanced, which no loss curve would ever reveal.
    """
    L, H, T = 4, 4, 24
    rng = np.random.RandomState(3)
    btns = ["".join(str(int(b)) for b in rng.randint(0, 2, BITS)) for _ in range(T)]
    d = str(tmp_path / "parity")
    write_pertic_episode(d, 0, held_actions([1, 2, 3, 4, 5, 6]), buttons=btns)
    meta = np.load(os.path.join(d, "ep_00000_meta.npz"))
    ctl = control_matrix(meta["buttons"])

    ds = TicWindowDataset(d, [0], context_frames=L, horizon=H, with_horizon=True, action_history=L)
    for i in range(len(ds)):
        _, s = ds.locate(i)
        _, _, tf_controls, _ = ds[i]
        for k in range(H):
            roll = rollout_eval.step_controls(ctl, s, L, k)
            assert np.array_equal(roll, tf_controls[k].numpy()), f"step {k} of window at {s}"
            # and the newest control is the one leaving that step's last context frame
            assert np.array_equal(roll[-1], ctl[s + L + k - 1])


def test_the_rollout_action_index_matches_the_teacher_forced_one(tmp_path):
    """The legacy single-action path: step h reads `action[s + L - 1 + h]`."""
    L, H = 4, 4
    d = str(tmp_path / "actidx")
    acts = held_actions([1, 2, 3, 4, 5, 6])
    write_pertic_episode(d, 0, acts)
    ds = TicWindowDataset(d, [0], context_frames=L, horizon=H, with_horizon=True)
    for i in range(len(ds)):
        _, s = ds.locate(i)
        _, _, tf_acts, _ = ds[i]
        assert tf_acts.tolist() == acts[s + L - 1:s + L - 1 + H].tolist()


def test_the_score_points_line_up_with_the_decision_horizons(tmp_path):
    """4, 32, 64, 128, 256 tics is the same game time as 1, 8, 16, 32, 64 decision steps."""
    assert [h // 4 for h in (4, 32, 64, 128, 256)] == [1, 8, 16, 32, 64]
    p = rollout_eval.build_parser()
    assert p.parse_args(["--score"]).score_at == ""
    assert p.parse_args(["--score", "--score-at", "4,32"]).score_at == "4,32"
    assert p.parse_args(["--rollout"]).horizon == 64
