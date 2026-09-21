"""CPU gates for the next-tic trainer: `--tic-stride`, `--phase-conditioning`, `--init-from`.

Three properties have to hold or the next-tic rows are not comparable with the stride-4 rows they
are meant to be read against:

* **The default path does not move.** `--tic-stride 4` with no phase conditioning must build the
  same model, from the same RNG draws, and call it with the same five arguments as every finished
  row did. Turning phase conditioning on must add a table and change nothing else.
* **The new flags reach the checkpoint.** `tic_stride`, the dataset class and the phase bucket
  count are recorded, so an evaluator can never silently score a next-tic model as a
  next-decision one.
* **Episode selection cannot overlap.** A training range that reaches into the dense corpus's
  val or test range fails at startup, not after 400k updates.

The heavy warm starts are replaced by tiny configs of the same classes, as `test_grid_flags.py`
does; the point is the wiring, not the weights.

    python -m pytest paper/fixtures/test_nexttic_train.py -q
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
sys.path.insert(0, HERE)

# `AcceleratorState` is a process-wide singleton fixed at the first `Accelerator()`, so the choice
# of device has to be made at import time, before any test builds one. Without this the trainer
# tests pick up the laptop's MPS device and every later test in the session inherits it.
os.environ.setdefault("ACCELERATE_USE_CPU", "1")

from pertic_fixtures import held_actions, write_pertic_episode, write_stride4_episode  # noqa: E402

import backbones  # noqa: E402
import train_wm  # noqa: E402
from doom_data import PHASE_BUCKETS  # noqa: E402

TINY_PIXART = dict(num_attention_heads=2, attention_head_dim=8, in_channels=4, out_channels=8,
                   num_layers=2, caption_channels=32, sample_size=64, patch_size=2,
                   cross_attention_dim=16, use_additional_conditions=False, norm_num_groups=2)


# ---------------------------------------------------------------------------------------
# the phase embedding: attached only when asked for, and it changes the prediction
# ---------------------------------------------------------------------------------------

def _dit(phase_buckets=0, seed=0):
    torch.manual_seed(seed)
    return backbones.DiTWorldModel(num_actions=3, context_frames=2, noise_buckets=4, model_name="DiT-S/2",
                                   action_dropout=0.0, grad_ckpt=False, phase_buckets=phase_buckets)


def _inputs(model, n=2, seed=7):
    g = torch.Generator().manual_seed(seed)
    c = model.latent_channels
    return dict(x=torch.randn(n, c, 32, 40, generator=g), t=torch.full((n,), 300, dtype=torch.long),
                action=torch.tensor([1, 2])[:n],
                context=torch.randn(n, c * model.context_frames, 32, 40, generator=g),
                noise_bucket=torch.tensor([0, 3])[:n])


def test_phase_off_attaches_no_parameter_and_draws_no_rng():
    """Bit-identity of the default path: same keys, same values, same init draws."""
    a, b = _dit(0, seed=0), _dit(0, seed=0)
    assert set(a.state_dict()) == set(b.state_dict())
    assert not any("phase_embedder" in k for k in a.state_dict())
    for k in a.state_dict():
        assert torch.equal(a.state_dict()[k], b.state_dict()[k]), k


def test_turning_phase_on_adds_a_table_and_moves_nothing_else():
    """The table is built after every other embedding, so the shared weights are untouched."""
    off, on = _dit(0, seed=0), _dit(PHASE_BUCKETS, seed=0)
    extra = set(on.state_dict()) - set(off.state_dict())
    assert extra == {"phase_embedder.weight"}
    assert on.state_dict()["phase_embedder.weight"].shape == (PHASE_BUCKETS, off.noise_embedder.embedding_dim)
    for k in off.state_dict():
        assert torch.equal(off.state_dict()[k], on.state_dict()[k]), f"{k} moved when phase was enabled"


def _wake(m, seed=0):
    """Small random values in every parameter.

    The DiT zero-initialises its adaLN modulations and its output head, so at initialisation *no*
    conditioning signal can reach the output and a "does the phase matter" test would pass
    vacuously. Waking the parameters up is what makes the check about the wiring.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    return m


@torch.no_grad()
def test_the_phase_changes_the_prediction():
    m = _wake(_dit(PHASE_BUCKETS, seed=1)).eval()
    kw = _inputs(m)
    p0 = m(**kw, phase=torch.zeros(2, dtype=torch.long))
    p1 = m(**kw, phase=torch.ones(2, dtype=torch.long))
    assert torch.isfinite(p0).all() and not torch.allclose(p0, p1)


@torch.no_grad()
def test_a_model_without_a_phase_table_ignores_a_phase_argument():
    """So an evaluator may pass it unconditionally without changing a stride-4 number."""
    m = _wake(_dit(0, seed=1)).eval()
    kw = _inputs(m)
    assert torch.equal(m(**kw), m(**kw, phase=torch.ones(2, dtype=torch.long)))


def test_a_phase_conditioned_model_refuses_to_run_without_a_phase():
    m = _dit(PHASE_BUCKETS, seed=1).eval()
    with pytest.raises(ValueError, match="tics_since_decision"):
        m(**_inputs(m))


# ---------------------------------------------------------------------------------------
# where the phase enters each backbone
# ---------------------------------------------------------------------------------------

def test_the_dit_sums_the_phase_into_adaln():
    m = _dit(PHASE_BUCKETS, seed=2)
    assert m.phase_embedder.embedding_dim == m.noise_embedder.embedding_dim


class TokenSpy(torch.nn.Module):
    """Stands in for a wrapped diffusers model and records the conditioning sequence it was handed.

    A plain function cannot be assigned over a submodule, so the spy has to be a Module itself.
    """

    def __init__(self, out_channels):
        super().__init__()
        self.out_channels, self.seen = out_channels, None

    def forward(self, sample, *a, encoder_hidden_states=None, **k):
        self.seen = tuple(encoder_hidden_states.shape)
        return type("O", (), {"sample": torch.zeros(sample.shape[0], self.out_channels, 32, 40)})()


def test_the_unet_gets_a_second_cross_attention_token(monkeypatch):
    from diffusers import UNet2DConditionModel
    tiny = dict(sample_size=8, block_out_channels=(8, 8), layers_per_block=1, in_channels=4,
                out_channels=4, cross_attention_dim=16, attention_head_dim=2, norm_num_groups=8,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"))
    monkeypatch.setattr(UNet2DConditionModel, "from_pretrained",
                        classmethod(lambda cls, *a, **k: cls(**tiny, num_class_embeds=k.get("num_class_embeds", 4))))
    seen = {}
    for name, buckets, phase in (("plain", 0, None), ("phase", PHASE_BUCKETS, torch.zeros(2, dtype=torch.long))):
        m = backbones.UNetWorldModel(num_actions=3, context_frames=2, noise_buckets=4, action_dropout=0.0,
                                     grad_ckpt=False, phase_buckets=buckets)
        m.unet = TokenSpy(4)
        m(**_inputs(m), phase=phase)
        seen[name] = m.unet.seen
    assert seen["plain"] == (2, 1, 16)
    assert seen["phase"] == (2, 2, 16)      # action token, then phase token


def test_pixart_gets_a_third_caption_token():
    from diffusers import PixArtTransformer2DModel
    seen = {}
    for name, buckets, phase in (("plain", 0, None), ("phase", PHASE_BUCKETS, torch.zeros(2, dtype=torch.long))):
        m = backbones.PixArtWorldModel(num_actions=3, context_frames=2, noise_buckets=4, action_dropout=0.0,
                                       grad_ckpt=False, phase_buckets=buckets,
                                       transformer=PixArtTransformer2DModel(**TINY_PIXART))
        m.transformer = TokenSpy(8)
        m(**_inputs(m), phase=phase)
        seen[name] = m.transformer.seen
    assert seen["plain"] == (2, 2, TINY_PIXART["caption_channels"])
    assert seen["phase"] == (2, 3, TINY_PIXART["caption_channels"])


def test_pixart_adaln_adds_the_phase_into_the_timestep_path():
    from diffusers import PixArtTransformer2DModel
    m = backbones.PixArtWorldModel(num_actions=3, context_frames=2, noise_buckets=4, action_dropout=0.0,
                                   grad_ckpt=False, action_inject="adaln", phase_buckets=PHASE_BUCKETS,
                                   transformer=PixArtTransformer2DModel(**TINY_PIXART))
    # zero-init under adaln, exactly as the action and bucket tables are, so step 0 is the pretrained model
    assert torch.equal(m.phase_embedder.weight, torch.zeros_like(m.phase_embedder.weight))
    assert m.phase_embedder.embedding_dim == m.bucket_embedder.embedding_dim


def test_sd35_gets_a_third_context_token_and_leaves_the_pooled_slot_alone():
    S = pytest.importorskip("diffusers").SD3Transformer2DModel

    def tiny():
        # a fresh transformer each time: the wrapper inflates the patch projection in place, so one
        # instance cannot be handed to two wrappers
        return S(sample_size=8, patch_size=2, in_channels=16, num_layers=1, attention_head_dim=8,
                 num_attention_heads=2, joint_attention_dim=16, caption_projection_dim=16,
                 pooled_projection_dim=8, out_channels=16, pos_embed_max_size=32)
    m = backbones.SD35WorldModel(num_actions=3, context_frames=2, noise_buckets=4, action_dropout=0.0,
                                 grad_ckpt=False, latent_channels=16, transformer=tiny(),
                                 phase_buckets=PHASE_BUCKETS)
    act, bucket, phase = torch.tensor([1, 2]), torch.tensor([0, 3]), torch.zeros(2, dtype=torch.long)
    three, pooled = m.conditioning(act, bucket, phase)
    assert three.shape == (2, 3, 16)
    # the phase is appended after the action and bucket tokens, not inserted between them
    assert torch.equal(three[:, 0], m.action_embedder(act))
    assert torch.equal(three[:, 1], m.bucket_embedder(bucket))
    assert torch.equal(three[:, 2], m.phase_embedder(phase))
    # the pooled slot carries only the action, as it did before phase conditioning existed
    assert torch.equal(pooled, m.pooled_action(act) + m.pooled_base)
    plain = backbones.SD35WorldModel(num_actions=3, context_frames=2, noise_buckets=4, action_dropout=0.0,
                                     grad_ckpt=False, latent_channels=16, transformer=tiny())
    assert plain.conditioning(act, bucket)[0].shape == (2, 2, 16)


# ---------------------------------------------------------------------------------------
# --action-history: GameNGen's executed-control token sequence
# ---------------------------------------------------------------------------------------

BITS, HIST = 6, 2        # the tiny models here use context_frames 2, so the history is 2


def _controls(n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n, HIST, BITS), generator=g).float()


def test_the_control_embedder_positions_the_tokens_and_checks_its_input_shape():
    e = backbones.ControlHistoryEmbedder(BITS, 16, HIST)
    out = e(_controls())
    assert out.shape == (2, HIST, 16)
    assert e.pos.shape == (1, HIST, 16) and e.pos.std() > 0      # learned positions, small init
    with pytest.raises(ValueError, match="expected controls of shape"):
        e(torch.zeros(2, HIST + 1, BITS))
    with pytest.raises(ValueError, match="expected controls of shape"):
        e(torch.zeros(2, BITS))


def test_identical_controls_at_different_positions_give_different_tokens():
    """Which is the whole point of the position table: order has to be recoverable."""
    e = backbones.ControlHistoryEmbedder(BITS, 16, HIST)
    same = torch.ones(1, HIST, BITS)
    out = e(same)
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_the_noise_bucket_token_carries_no_position():
    """The position table spans the L control tokens only, so the bucket token is unpositioned."""
    e = backbones.ControlHistoryEmbedder(BITS, 16, HIST)
    assert e.pos.shape[1] == HIST


def test_the_unet_keeps_the_bucket_in_class_labels_and_sends_only_controls(monkeypatch):
    """Decision (Rohan, Sep 20): the bucket stays where it already is, so the sequence is L tokens."""
    from diffusers import UNet2DConditionModel
    tiny = dict(sample_size=8, block_out_channels=(8, 8), layers_per_block=1, in_channels=4,
                out_channels=4, cross_attention_dim=16, attention_head_dim=2, norm_num_groups=8,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"))
    monkeypatch.setattr(UNet2DConditionModel, "from_pretrained",
                        classmethod(lambda cls, *a, **k: cls(**tiny, num_class_embeds=k.get("num_class_embeds", 4))))
    m = backbones.UNetWorldModel(num_actions=3, context_frames=HIST, noise_buckets=4, action_dropout=0.0,
                                 grad_ckpt=False, action_history=HIST, control_bits=BITS)
    m.unet = TokenSpy(4)
    kw = _inputs(m)
    kw["action"] = _controls()
    m(**kw)
    assert m.unet.seen == (2, HIST, 16)


def test_pixart_concatenates_the_controls_then_the_bucket():
    from diffusers import PixArtTransformer2DModel
    m = backbones.PixArtWorldModel(num_actions=3, context_frames=HIST, noise_buckets=4, action_dropout=0.0,
                                   grad_ckpt=False, action_history=HIST, control_bits=BITS,
                                   transformer=PixArtTransformer2DModel(**TINY_PIXART))
    m.transformer = TokenSpy(8)
    kw = _inputs(m)
    kw["action"] = _controls()
    m(**kw)
    assert m.transformer.seen == (2, HIST + 1, TINY_PIXART["caption_channels"])


def test_pixart_adaln_refuses_action_history():
    """`adaln` removes the cross-attention pathway the token sequence needs."""
    from diffusers import PixArtTransformer2DModel
    with pytest.raises(NotImplementedError, match="action-history"):
        backbones.PixArtWorldModel(num_actions=3, context_frames=HIST, noise_buckets=4, action_dropout=0.0,
                                   grad_ckpt=False, action_inject="adaln", action_history=HIST,
                                   control_bits=BITS, transformer=PixArtTransformer2DModel(**TINY_PIXART))


def test_sd35_pools_only_the_newest_control():
    S = pytest.importorskip("diffusers").SD3Transformer2DModel
    tr = S(sample_size=8, patch_size=2, in_channels=16, num_layers=1, attention_head_dim=8,
           num_attention_heads=2, joint_attention_dim=16, caption_projection_dim=16,
           pooled_projection_dim=8, out_channels=16, pos_embed_max_size=32)
    m = backbones.SD35WorldModel(num_actions=3, context_frames=HIST, noise_buckets=4, action_dropout=0.0,
                                 grad_ckpt=False, latent_channels=16, transformer=tr,
                                 action_history=HIST, control_bits=BITS)
    controls, bucket = _controls(), torch.tensor([0, 3])
    tokens, pooled = m.conditioning(controls, bucket)
    assert tokens.shape == (2, HIST + 1, 16)
    assert torch.equal(tokens[:, HIST], m.bucket_embedder(bucket))          # bucket last
    # the pooled slot is the NEWEST control's embedding, never an average of the sequence
    newest = m.control_history(controls)[:, -1]
    assert torch.equal(tokens[:, HIST - 1], newest)
    assert torch.equal(pooled, m.pooled_control(newest) + m.pooled_base)
    assert torch.equal(m.pooled_control.weight, torch.zeros_like(m.pooled_control.weight))


def test_the_dit_documents_that_it_averages_because_it_has_no_cross_attention():
    m = backbones.DiTWorldModel(num_actions=3, context_frames=HIST, noise_buckets=4, model_name="DiT-S/2",
                                action_dropout=0.0, grad_ckpt=False, action_history=HIST, control_bits=BITS)
    kw = _inputs(m)
    kw["action"] = _controls()
    with torch.no_grad():
        assert torch.isfinite(_wake(m).eval()(**kw)).all()


def test_action_dropout_is_refused_with_action_history():
    """A `[B]` keep mask silently broadcasting onto `[B, L]` control tokens is the trap this closes."""
    m = backbones.DiTWorldModel(num_actions=3, context_frames=HIST, noise_buckets=4, model_name="DiT-S/2",
                                action_dropout=0.1, grad_ckpt=False, action_history=HIST, control_bits=BITS)
    m.train()
    kw = _inputs(m)
    kw["action"] = _controls()
    with pytest.raises(ValueError, match="action-dropout"):
        m(**kw)


def test_action_history_needs_the_control_width():
    with pytest.raises(ValueError, match="button-vector width"):
        backbones.DiTWorldModel(num_actions=3, context_frames=HIST, noise_buckets=4, model_name="DiT-S/2",
                                action_dropout=0.0, grad_ckpt=False, action_history=HIST, control_bits=0)


def test_action_history_zero_attaches_nothing():
    off = backbones.DiTWorldModel(num_actions=3, context_frames=HIST, noise_buckets=4, model_name="DiT-S/2",
                                  action_dropout=0.0, grad_ckpt=False)
    assert not any("control_history" in k for k in off.state_dict())
    assert backbones.phase_vec(off, None) is None


def test_the_legacy_action_table_is_not_created_when_history_is_on(monkeypatch):
    """A trainable-but-unused parameter makes DDP fail on the second iteration unless unused-parameter
    detection is on, which costs throughput; so the replaced tables are never built."""
    from diffusers import PixArtTransformer2DModel, UNet2DConditionModel
    tiny_unet = dict(sample_size=8, block_out_channels=(8, 8), layers_per_block=1, in_channels=4,
                     out_channels=4, cross_attention_dim=16, attention_head_dim=2, norm_num_groups=8,
                     down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                     up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"))
    monkeypatch.setattr(UNet2DConditionModel, "from_pretrained",
                        classmethod(lambda cls, *a, **k: cls(**tiny_unet, num_class_embeds=k.get("num_class_embeds", 4))))
    S = pytest.importorskip("diffusers").SD3Transformer2DModel

    def sd3():
        return S(sample_size=8, patch_size=2, in_channels=16, num_layers=1, attention_head_dim=8,
                 num_attention_heads=2, joint_attention_dim=16, caption_projection_dim=16,
                 pooled_projection_dim=8, out_channels=16, pos_embed_max_size=32)
    common = dict(num_actions=3, context_frames=HIST, noise_buckets=4, action_dropout=0.0, grad_ckpt=False)
    builders = {
        "unet": lambda h: backbones.UNetWorldModel(**common, action_history=h, control_bits=BITS if h else 0),
        "pixart": lambda h: backbones.PixArtWorldModel(**common, action_history=h, control_bits=BITS if h else 0,
                                                       transformer=PixArtTransformer2DModel(**TINY_PIXART)),
        "sd35": lambda h: backbones.SD35WorldModel(**common, latent_channels=16, transformer=sd3(),
                                                   action_history=h, control_bits=BITS if h else 0),
    }
    for name, build in builders.items():
        off, on = build(0), build(HIST)
        assert any("action_embedder" in k for k in off.state_dict()), name
        assert not any("action_embedder" in k for k in on.state_dict()), f"{name} kept an unused action table"
        assert any("control_history" in k for k in on.state_dict()), name
        if name == "sd35":
            assert not any("pooled_action" in k for k in on.state_dict())
            assert any("pooled_control" in k for k in on.state_dict())
        # the default path still has every table it always had
        assert all(p.requires_grad for p in off.parameters())


def test_the_dit_freezes_its_own_class_table_instead(monkeypatch):
    """The DiT's table belongs to the pretrained DiT module and cannot be skipped, so it is frozen."""
    on = backbones.DiTWorldModel(num_actions=3, context_frames=HIST, noise_buckets=4, model_name="DiT-S/2",
                                 action_dropout=0.0, grad_ckpt=False, action_history=HIST, control_bits=BITS)
    assert not any(p.requires_grad for p in on.dit.y_embedder.parameters())
    off = backbones.DiTWorldModel(num_actions=3, context_frames=HIST, noise_buckets=4, model_name="DiT-S/2",
                                  action_dropout=0.0, grad_ckpt=False)
    assert all(p.requires_grad for p in off.dit.y_embedder.parameters())


def test_the_fit_check_control_width_comes_from_the_recorded_button_card():
    card = json.load(open(os.path.join(REPO, "docs", "cards", "arnold", "buttons.json")))
    assert train_wm.fit_check_control_bits() == len(card["available_buttons"]) == 19


def test_the_fit_check_windows_supply_every_column_the_model_needs():
    """A fit check under --phase-conditioning used to crash: `SyntheticWindows` always returned
    three tensors, so the phase-conditioned model refused the batch and nothing was measured."""
    plain = train_wm.SyntheticWindows(4, 2, 3, 4)
    assert len(plain[0]) == 3
    phase = train_wm.SyntheticWindows(4, 2, 3, 4, phase_buckets=PHASE_BUCKETS)
    ctx, tgt, act, ph = phase[0]
    assert 0 <= int(ph) < PHASE_BUCKETS
    assert train_wm.unpack_batch(phase[0])[3] is not None
    hist = train_wm.SyntheticWindows(4, 2, 3, 4, action_history=2, control_bits=19)
    assert hist[0][2].shape == (2, 19) and hist[0][2].dtype == torch.float32
    both = train_wm.SyntheticWindows(4, 2, 3, 4, action_history=2, control_bits=19,
                                     phase_buckets=PHASE_BUCKETS)
    assert len(both[0]) == 4 and both[0][2].shape == (2, 19)
    # the first two draws are unchanged, so an existing fit check's windows do not move
    assert torch.equal(plain[3][0], phase[3][0]) and torch.equal(plain[3][1], phase[3][1])


def test_the_fit_check_builds_a_phase_conditioned_model_from_the_flags():
    a = train_wm.build_parser().parse_args(
        ["--backbone", "dit", "--fit-check", "2", "--tic-stride", "1", "--phase-conditioning"])
    ds, val, ids = train_wm.build_loaders(a, 4)
    assert val is None and ids is None
    assert len(ds[0]) == 4, "the fit-check dataset must supply the phase column"
    b = train_wm.build_parser().parse_args(
        ["--backbone", "dit", "--fit-check", "2", "--tic-stride", "1", "--action-history", "32"])
    assert train_wm.build_loaders(b, 4)[0][0][2].shape == (32, train_wm.fit_check_control_bits())


def test_the_trainer_refuses_the_impossible_action_history_combinations():
    for argv, match in ((["--tic-stride", "4", "--action-history", "32"], "tic-stride 1"),
                        (["--tic-stride", "1", "--action-history", "16", "--context-frames", "32"],
                         "equal --context-frames"),
                        (["--tic-stride", "1", "--action-history", "32", "--action-dropout", "0.1"],
                         "action-dropout")):
        a = train_wm.build_parser().parse_args(["--backbone", "dit", "--fit-check", "1"] + argv)
        with pytest.raises(SystemExit, match=match):
            train_wm.main(a)


# ---------------------------------------------------------------------------------------
# --init-from
# ---------------------------------------------------------------------------------------

class TinyNet(torch.nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.lin = torch.nn.Linear(n, n)
        self.phase_embedder = None

    def forward(self, x):
        return self.lin(x)


def _tiny_ckpt(path, seed, *, with_ema=True, step=90000, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    src = torch.nn.Linear(4, 4)
    sd = {f"lin.{k}": v.detach().to(dtype) for k, v in src.state_dict().items()}
    ck = {"model": sd, "step": step, "args": {"backbone": "dit"}}
    if with_ema:
        ck["ema"] = {k: (v.float() * 0 + 0.5).to(dtype) for k, v in sd.items()}
    torch.save(ck, path)
    return sd


def test_init_from_loads_the_weights_and_the_source_step(tmp_path):
    p = str(tmp_path / "best.pt")
    sd = _tiny_ckpt(p, seed=3)
    m = torch.nn.Sequential()
    m.add_module("lin", torch.nn.Linear(4, 4))
    step, ema = train_wm.load_init_weights(m, p)
    assert step == 90000
    for k, v in sd.items():
        assert torch.equal(m.state_dict()[k], v.float())
    assert set(ema) == set(sd)


def test_init_from_leaves_the_optimizer_state_empty(tmp_path):
    """The difference between --init-from and --resume: no Adam moments, no schedule position."""
    p = str(tmp_path / "recovery.pt")
    _tiny_ckpt(p, seed=4, dtype=torch.float32)
    m = torch.nn.Sequential()
    m.add_module("lin", torch.nn.Linear(4, 4))
    train_wm.load_init_weights(m, p)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    assert opt.state_dict()["state"] == {}
    assert all(g["lr"] == 1e-4 for g in opt.param_groups)


def test_init_from_accepts_all_three_checkpoint_shapes(tmp_path):
    for name, kw in (("best.pt", dict(with_ema=False)), ("0090000.pt", dict(dtype=torch.float32)),
                     ("snap_0090000.pt", {})):
        p = str(tmp_path / name)
        _tiny_ckpt(p, seed=5, **kw)
        m = torch.nn.Sequential()
        m.add_module("lin", torch.nn.Linear(4, 4))
        assert train_wm.load_init_weights(m, p)[0] == 90000


def test_init_from_refuses_a_checkpoint_for_another_architecture(tmp_path):
    p = str(tmp_path / "other.pt")
    _tiny_ckpt(p, seed=6)
    m = torch.nn.Sequential()
    m.add_module("lin", torch.nn.Linear(8, 8))
    with pytest.raises(SystemExit, match="does not match this model"):
        train_wm.load_init_weights(m, p)


def test_init_from_refuses_something_that_is_not_our_checkpoint(tmp_path):
    p = str(tmp_path / "junk.pt")
    torch.save({"state_dict": {}}, p)
    with pytest.raises(SystemExit, match="no 'model' state dict"):
        train_wm.load_init_weights(torch.nn.Linear(4, 4), p)


def test_init_from_allows_exactly_the_fresh_phase_table(tmp_path):
    """A stride-4 checkpoint has no tics_since_decision table; that one gap is reported, not fatal."""
    p = str(tmp_path / "nophase.pt")
    _tiny_ckpt(p, seed=7)
    m = TinyNet()
    m.phase_embedder = torch.nn.Embedding(PHASE_BUCKETS, 4)
    assert train_wm.load_init_weights(m, p)[0] == 90000


def test_init_from_and_resume_are_mutually_exclusive(tmp_path):
    a = train_wm.build_parser().parse_args(["--backbone", "dit", "--fit-check", "1",
                                            "--resume", "x.pt", "--init-from", "y.pt"])
    with pytest.raises(SystemExit, match="pick one"):
        train_wm.main(a)


# ---------------------------------------------------------------------------------------
# episode selection
# ---------------------------------------------------------------------------------------

def _args(**kw):
    argv = ["--backbone", "dit"]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    return train_wm.build_parser().parse_args(argv)


def test_episode_ids_without_validation_ids_is_refused():
    with pytest.raises(SystemExit, match="val-episode-ids"):
        train_wm.select_episodes(_args(episode_ids="0:10"))


def test_episode_ids_overlapping_validation_are_refused():
    with pytest.raises(ValueError, match="overlap"):
        train_wm.select_episodes(_args(episode_ids="0:10", val_episode_ids="5:12"))


def test_a_training_range_that_reaches_into_the_dense_val_range_is_refused():
    with pytest.raises(ValueError, match="overlap"):
        train_wm.select_episodes(_args(episode_ids="0:6500", val_episode_ids="7000:7100",
                                       dense_segment="arenas"))


def test_the_decided_dense_ranges_pass(tmp_path):
    d = str(tmp_path / "lat")
    for ep in list(range(4)) + [6000, 6001]:
        write_pertic_episode(d, ep, held_actions([1] * 10))
    a = _args(episode_ids="0:4", val_episode_ids="6000:6002", dense_segment="arenas", latents_dir=d)
    assert train_wm.select_episodes(a) == ([0, 1, 2, 3], [6000, 6001])


def test_max_episodes_takes_the_encoded_prefix(tmp_path):
    d = str(tmp_path / "prefix")
    for ep in range(6):
        write_pertic_episode(d, ep, held_actions([1] * 10))
    # at max_episodes 3 the validation range lies entirely past the encoded prefix, which has to
    # fail loudly rather than leave the run without a checkpoint-selection signal
    with pytest.raises(SystemExit, match="validation"):
        train_wm.select_episodes(_args(episode_ids="0:4", val_episode_ids="4:6", latents_dir=d,
                                       max_episodes=3))
    assert train_wm.select_episodes(_args(episode_ids="0:4", val_episode_ids="4:6", latents_dir=d,
                                          max_episodes=5)) == ([0, 1, 2, 3], [4])
    assert train_wm.select_episodes(_args(episode_ids="0:4", val_episode_ids="4:6",
                                          latents_dir=d)) == ([0, 1, 2, 3], [4, 5])


def test_a_partly_encoded_corpus_is_refused_without_the_flag(tmp_path):
    """A multi-day run must not start on a corpus still being written: the episode set would then
    depend on when the run happened to start."""
    d = str(tmp_path / "half")
    for ep in range(3):
        write_pertic_episode(d, ep, held_actions([1] * 10))
    # episodes 0, 1, 2 are encoded: the training range is complete, the validation range is not
    with pytest.raises(SystemExit, match="still being written"):
        train_wm.select_episodes(_args(episode_ids="0:2", val_episode_ids="2:5", latents_dir=d))
    got = train_wm.select_episodes(_args(episode_ids="0:2", val_episode_ids="2:5", latents_dir=d,
                                         allow_partial=True))
    assert got == ([0, 1], [2])


def test_max_episodes_is_itself_a_deliberate_prefix(tmp_path):
    """--max-episodes says "use the prefix" on purpose, so it does not also need --allow-partial."""
    d = str(tmp_path / "prefix2")
    for ep in range(6):
        write_pertic_episode(d, ep, held_actions([1] * 10))
    assert train_wm.select_episodes(_args(episode_ids="0:4", val_episode_ids="4:6", latents_dir=d,
                                          max_episodes=5)) == ([0, 1, 2, 3], [4])


def test_a_resume_is_pinned_to_the_episodes_the_run_started_with(tmp_path):
    """`limit_to_encoded` resolves against what is encoded NOW, so a resume after more episodes
    finished encoding would silently train on a larger set."""
    d = str(tmp_path / "grow")
    for ep in range(3):
        write_pertic_episode(d, ep, held_actions([1] * 10))
    out = str(tmp_path / "run")
    os.makedirs(out)
    json.dump({"episodes": [0, 1], "val_episodes": [2], "num_episodes": 2},
              open(os.path.join(out, train_wm.EPISODES_FILE), "w"))
    a = _args(episode_ids="0:2", val_episode_ids="2:3", latents_dir=d, results_dir=out, resume="x.pt")
    assert train_wm.pin_episodes(a, [0, 1, 2], [2]) == ([0, 1], [2])
    # a first launch (no --resume) writes nothing and passes the resolved lists through
    b = _args(episode_ids="0:2", val_episode_ids="2:3", latents_dir=d, results_dir=str(tmp_path / "fresh"))
    assert train_wm.pin_episodes(b, [0, 1], [2]) == ([0, 1], [2])


def test_a_resume_refuses_when_a_pinned_episode_has_disappeared(tmp_path):
    d = str(tmp_path / "shrunk")
    write_pertic_episode(d, 0, held_actions([1] * 10))
    out = str(tmp_path / "run2")
    os.makedirs(out)
    json.dump({"episodes": [0, 1], "val_episodes": [2]},
              open(os.path.join(out, train_wm.EPISODES_FILE), "w"))
    a = _args(episode_ids="0:2", val_episode_ids="2:3", latents_dir=d, results_dir=out, resume="x.pt")
    with pytest.raises(SystemExit, match="no longer in the latent directories"):
        train_wm.pin_episodes(a, [0], [])


def test_an_old_format_episodes_file_is_left_alone(tmp_path):
    """The stride-4 rows' `train_episodes.json` has no val list; a resume there must not change."""
    out = str(tmp_path / "old")
    os.makedirs(out)
    json.dump({"episodes": [0, 1], "num_episodes": 2, "train_fraction": 1.0},
              open(os.path.join(out, train_wm.EPISODES_FILE), "w"))
    a = _args(results_dir=out, resume="x.pt")
    assert train_wm.pin_episodes(a, [5, 6], [7]) == ([5, 6], [7])


# ---------------------------------------------------------------------------------------
# end to end: five updates on a per-tic corpus
# ---------------------------------------------------------------------------------------

EPISODES, TICS = 10, 40


@pytest.fixture(scope="module")
def pertic_corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("pertic")
    rng = np.random.RandomState(0)
    for ep in range(EPISODES):
        write_pertic_episode(str(d), ep, rng.randint(0, 3, TICS))
    return str(d)


@pytest.fixture(scope="module")
def stride4_corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("stride4")
    rng = np.random.RandomState(1)
    for ep in range(EPISODES):
        write_stride4_episode(str(d), ep, rng.randint(0, 3, TICS))
    json.dump({"train": list(range(8)), "val": [8, 9]}, open(os.path.join(str(d), "split.json"), "w"))
    return str(d)


@pytest.fixture
def tiny_hub(monkeypatch):
    from diffusers import PixArtTransformer2DModel as P
    monkeypatch.setattr(P, "from_pretrained", classmethod(lambda cls, *a, **k: cls(**TINY_PIXART)))
    monkeypatch.setattr(P, "load_config", classmethod(lambda cls, *a, **k: dict(TINY_PIXART)))
    monkeypatch.setenv("ACCELERATE_USE_CPU", "1")


def _run(out_dir, extra, latents, split=None, ids=None):
    argv = ["--backbone", "pixart", "--warm-start", backbones.PIXART_DEFAULT,
            "--latents-dir", latents, "--results-dir", out_dir,
            "--num-actions", "3", "--noise-buckets", "4", "--context-frames", "32",
            "--per-gpu-batch", "2", "--global-batch", "2", "--steps", "5", "--warmup", "2",
            "--val-every", "5", "--val-windows", "2", "--ckpt-every", "5", "--ema-every", "1",
            "--num-workers", "0", "--action-dropout", "0.0", "--seed", "0"]
    argv += ["--split", split] if split else ["--episode-ids", ids[0], "--val-episode-ids", ids[1]]
    train_wm.main(train_wm.build_parser().parse_args(argv + extra))
    return json.load(open(os.path.join(out_dir, "config.json")))


def test_a_per_tic_run_trains_and_records_its_data_contract(pertic_corpus, tiny_hub, tmp_path):
    out = str(tmp_path / "nexttic")
    cfg = _run(out, ["--tic-stride", "1"], pertic_corpus, ids=("0:8", "8:10"))
    assert cfg["tic_stride"] == 1 and cfg["dataset_class"] == "TicWindowDataset"
    assert cfg["resolved_phase_buckets"] == 0 and cfg["init_from"] is None
    events = [json.loads(ln) for ln in open(os.path.join(out, "log.jsonl"))]
    assert events[0]["event"] == "start" and events[0]["tic_stride"] == 1
    assert events[-1]["event"] == "end" and events[-1]["step"] == 5
    assert [e for e in events if e["event"] == "val"], "no validation ran"
    assert all(np.isfinite(e["val_loss"]) for e in events if e["event"] == "val")


def test_a_per_tic_run_with_phase_conditioning_trains(pertic_corpus, tiny_hub, tmp_path):
    out = str(tmp_path / "phase")
    cfg = _run(out, ["--tic-stride", "1", "--phase-conditioning"], pertic_corpus, ids=("0:8", "8:10"))
    assert cfg["resolved_phase_buckets"] == PHASE_BUCKETS
    ck = torch.load(os.path.join(out, "best.pt"), map_location="cpu", weights_only=False)
    assert any("phase_embedder" in k for k in ck["model"])
    assert ck["args"]["phase_conditioning"] is True


def test_the_stride4_run_is_unchanged_and_says_so(stride4_corpus, tiny_hub, tmp_path):
    out = str(tmp_path / "stride4")
    cfg = _run(out, ["--require-verified-transitions"], stride4_corpus,
               split=os.path.join(stride4_corpus, "split.json"))
    assert cfg["tic_stride"] == 4 and cfg["dataset_class"] == "LatentWindowDataset"
    assert cfg["resolved_phase_buckets"] == 0
    ck = torch.load(os.path.join(out, "best.pt"), map_location="cpu", weights_only=False)
    assert not any("phase_embedder" in k for k in ck["model"])


def test_a_run_can_start_from_another_runs_checkpoint(pertic_corpus, tiny_hub, tmp_path):
    """The stride-4 -> next-tic handoff, which is what --init-from exists for."""
    src = str(tmp_path / "src")
    _run(src, ["--tic-stride", "1"], pertic_corpus, ids=("0:8", "8:10"))
    out = str(tmp_path / "dst")
    cfg = _run(out, ["--tic-stride", "1", "--init-from", os.path.join(src, "best.pt")],
               pertic_corpus, ids=("0:8", "8:10"))
    assert cfg["init_from"].endswith("best.pt") and cfg["init_from_step"] == 5
    events = [json.loads(ln) for ln in open(os.path.join(out, "log.jsonl"))]
    assert events[-1]["step"] == 5, "the step counter must restart, not continue from the source"
    assert events[0]["init_from_step"] == 5


def test_phase_conditioning_is_refused_at_stride_four():
    a = train_wm.build_parser().parse_args(["--backbone", "dit", "--fit-check", "1", "--phase-conditioning"])
    with pytest.raises(SystemExit, match="tic-stride 1"):
        train_wm.main(a)


def test_the_seeded_validation_corruption_keeps_its_seven_element_layout():
    """An existing run's validation loss must not move because a phase column now exists."""
    class Three(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return torch.zeros(4, 2, 2), torch.zeros(2, 2, 2), torch.tensor(1)

    class Four(Three):
        def __getitem__(self, i):
            return super().__getitem__(i) + (torch.tensor(2),)

    assert len(train_wm.SeededCorruption(Three(), 0.7, 1000)[0]) == 7
    assert len(train_wm.SeededCorruption(Four(), 0.7, 1000)[0]) == 8
    three = train_wm.SeededCorruption(Three(), 0.7, 1000)[0]
    four = train_wm.SeededCorruption(Four(), 0.7, 1000)[0]
    for a, b in zip(three, four[:7]):
        assert torch.equal(torch.as_tensor(a), torch.as_tensor(b))
    assert int(four[7]) == 2


def test_unpack_batch_reads_the_optional_fourth_column():
    a, b, c, d = train_wm.unpack_batch((1, 2, 3))
    assert (a, b, c, d) == (1, 2, 3, None)
    assert train_wm.unpack_batch((1, 2, 3, 4)) == (1, 2, 3, 4)
