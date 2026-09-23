"""The rest of the review's pre-launch list (7.1): real-window contract, smoke probes, resume, readback.

Astra's review of the gates found them short of 7.1 in five ways, each now a gate:

  * an exact inventory of the VAL corpus (ids, rows, tics, split file), not only the training one;
  * `check_emitted_windows.py`: on real training windows, the loader's newest control is row r-1 and
    changing buttons[r] leaves the emitted sample unchanged;
  * `smoke_probe.py`: after the 300-step smoke, the control MLP, position table, inflated context
    convolution and SD 3.5 `pooled_control` moved and have finite nonzero gradients, and flipping the
    newest control changes the output under fixed noise;
  * readback of live AND EMA weights at horizon 1 AND 4;
  * a 10-update `--resume` of the smoke run.

The probes run on tiny PixArt and SD 3 transformers on the CPU; the gate wiring through `DRY=1`.

    python -m pytest paper/fixtures/test_gate_probes.py -q
"""
import os
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

import backbones  # noqa: E402
import check_emitted_windows as cew  # noqa: E402
import gate_certificate as gc  # noqa: E402
import smoke_probe as sp  # noqa: E402
from pertic_fixtures import held_actions, write_pertic_episode  # noqa: E402

GATES = os.path.join(REPO, "scripts", "cluster", "gates.sh")
CTX, BITS = 2, 19
TINY_PIXART = {"num_attention_heads": 2, "attention_head_dim": 8, "in_channels": 4, "out_channels": 8,
               "num_layers": 2, "caption_channels": 32, "sample_size": 64, "patch_size": 2,
               "cross_attention_dim": 16, "use_additional_conditions": False, "norm_num_groups": 2}
TINY_SD3 = {"sample_size": 128, "patch_size": 2, "in_channels": 16, "out_channels": 16, "num_layers": 2,
            "attention_head_dim": 8, "num_attention_heads": 2, "joint_attention_dim": 32,
            "caption_projection_dim": 16, "pooled_projection_dim": 24, "pos_embed_max_size": 96,
            "dual_attention_layers": (0,), "qk_norm": "rms_norm"}


def corpus(path, ids, channels=4, T=24, seed=0):
    """Per-tic episodes with random executed controls, so every control row is distinguishable."""
    rng = np.random.RandomState(seed)
    for ep in ids:
        btns = ["".join(str(int(b)) for b in rng.randint(0, 2, BITS)) for _ in range(T)]
        write_pertic_episode(str(path), ep, held_actions([0, 1, 2, 0, 1, 2]), buttons=btns, channels=channels)
    return str(path)


# ---------------------------------------------------------------------------------------
# emitted windows on the real corpus
# ---------------------------------------------------------------------------------------

def test_the_loader_obeys_the_contract_on_real_windows(tmp_path):
    d = corpus(tmp_path / "lat", (0, 1))
    rep = cew.check(d, [0, 1], context_frames=4, windows=16)
    assert rep["ok"], rep["problems"]
    assert rep["windows_checked"] == 16


def test_an_off_by_one_control_history_is_caught(tmp_path, monkeypatch):
    """A loader that handed the model rows s+1 .. r (so the newest is the control chosen AFTER the
    target) passes every shape check; this gate must not."""
    from doom_data import TicWindowDataset
    d = corpus(tmp_path / "lat", (0,))

    def shifted(self, slot, start, offset=0):
        controls = self.episodes[slot][7]
        return torch.from_numpy(controls[start + offset + 1:start + offset + 1 + self.L].copy())

    monkeypatch.setattr(TicWindowDataset, "control_history", shifted)
    rep = cew.check(d, [0], context_frames=4, windows=8)
    assert not rep["ok"]
    text = " ".join(rep["problems"])
    assert "control history is not buttons of rows" in text
    assert "changing buttons[" in text and "changed the sample" in text


def test_a_target_read_from_the_wrong_row_is_caught(tmp_path, monkeypatch):
    from doom_data import TicWindowDataset
    d = corpus(tmp_path / "lat", (0,))
    real = TicWindowDataset.__getitem__

    def late(self, idx):
        ctx, tgt, a = real(self, idx)
        slot, start = self.locate(idx)
        lat = self.episodes[slot][1]
        return ctx, torch.from_numpy(np.asarray(lat[start + self.L - 1], dtype=np.float32)), a

    monkeypatch.setattr(TicWindowDataset, "__getitem__", late)
    rep = cew.check(d, [0], context_frames=4, windows=8)
    assert not rep["ok"] and any("target is not" in p for p in rep["problems"])


def test_the_windows_cli_exit_code(tmp_path):
    d = corpus(tmp_path / "lat", (0,))
    a = cew.build_parser().parse_args(["--latents-dir", d, "--episodes", "0:1", "--windows", "4",
                                       "--context-frames", "4", "--out", str(tmp_path / "r.json")])
    assert cew.main(a) == 0 and (tmp_path / "r.json").is_file()


# ---------------------------------------------------------------------------------------
# smoke probes
# ---------------------------------------------------------------------------------------

def pixart_model():
    from diffusers import PixArtTransformer2DModel
    torch.manual_seed(0)
    return backbones.PixArtWorldModel(num_actions=3, context_frames=CTX, noise_buckets=4, action_dropout=0.0,
                                      grad_ckpt=False, action_history=CTX, control_bits=BITS,
                                      transformer=PixArtTransformer2DModel(**TINY_PIXART))


@pytest.fixture
def tiny_sd3(monkeypatch):
    from diffusers import SD3Transformer2DModel as S
    monkeypatch.setattr(S, "from_pretrained", classmethod(lambda cls, *a, **k: cls(**TINY_SD3)))


def sd35_model():
    torch.manual_seed(0)
    return backbones.build_model("sd35", 3, CTX, 4, grad_ckpt=False, warm_start=backbones.SD35_DEFAULT,
                                 action_dropout=0.0, latent_channels=16, action_history=CTX, control_bits=BITS)


def trained(model, channels, skip=()):
    """What a smoke leaves: EMA = the step-0 weights, live = those weights after some updates."""
    ema = {n: p.detach().clone().float() for n, p in model.named_parameters()}
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            grp = sp.group_of(n, tuple(p.shape), channels, CTX)
            if grp is None or grp in skip:
                continue
            if grp == "context_conv":
                p[:, :channels * CTX] += 0.01 * torch.randn(p[:, :channels * CTX].shape, generator=g)
            else:
                p += 0.01 * torch.randn(p.shape, generator=g)
    return {"model": {k: v.detach().clone() for k, v in model.state_dict().items()}, "ema": ema, "step": 300,
            "args": {"action_history": CTX, "objective": "v"}}


def batch(tmp_path, channels):
    d = corpus(tmp_path / f"lat{channels}", (0,), channels=channels)
    return sp.real_batch(d, [0], CTX, channels, CTX, batch=2)


def test_a_trained_pixart_smoke_passes(tmp_path):
    m = pixart_model()
    ck = trained(m, 4)
    rep = sp.probe(ck, m, batch(tmp_path, 4), 4, CTX, "pixart")
    assert rep["ok"], rep["problems"]
    assert set(rep["groups"]) == {"control_mlp", "control_pos", "context_conv"}
    assert rep["probe"]["newest_control_sensitivity"] > 0


def test_a_trained_sd35_smoke_passes_and_needs_its_pooled_control(tmp_path, tiny_sd3):
    m = sd35_model()
    ck = trained(m, 16)
    rep = sp.probe(ck, m, batch(tmp_path, 16), 16, CTX, "sd35")
    assert rep["ok"], rep["problems"]
    assert "pooled_control" in rep["groups"] and rep["probe"]["grad_pooled_control"] > 0
    m2 = sd35_model()
    ck2 = trained(m2, 16, skip=("pooled_control",))
    rep2 = sp.probe(ck2, m2, batch(tmp_path, 16), 16, CTX, "sd35")
    assert not rep2["ok"] and any("pooled_control: still at its zero initialisation" in p for p in rep2["problems"])


def test_a_position_table_that_never_moved_fails(tmp_path):
    m = pixart_model()
    ck = trained(m, 4, skip=("control_pos",))
    rep = sp.probe(ck, m, batch(tmp_path, 4), 4, CTX, "pixart")
    assert not rep["ok"] and any("control_pos: did not move" in p for p in rep["problems"])


def test_a_context_conv_still_at_zero_fails(tmp_path):
    m = pixart_model()
    ck = trained(m, 4, skip=("context_conv",))
    rep = sp.probe(ck, m, batch(tmp_path, 4), 4, CTX, "pixart")
    assert not rep["ok"] and any("context_conv: still at its zero initialisation" in p for p in rep["problems"])


def test_dead_controls_fail_on_gradient_and_sensitivity(tmp_path):
    """A control MLP whose output layer is zero: the loss runs, the model ignores every control."""
    m = pixart_model()
    ck = trained(m, 4)
    with torch.no_grad():
        m.control_history.mlp[2].weight.zero_()
        m.control_history.mlp[2].bias.zero_()
    rep = sp.probe(ck, m, batch(tmp_path, 4), 4, CTX, "pixart")
    text = " ".join(rep["problems"])
    assert not rep["ok"]
    assert "control_mlp: zero gradient" in text
    assert "flipping the newest control does not change the output" in text


def test_non_finite_weights_fail(tmp_path):
    m = pixart_model()
    ck = trained(m, 4)
    k = [n for n in ck["model"] if n.startswith("control_history.mlp.0")][0]
    ck["model"][k] = ck["model"][k].clone()
    ck["model"][k][0, 0] = float("nan")
    rep = sp.probe(ck, m, batch(tmp_path, 4), 4, CTX, "pixart")
    assert not rep["ok"] and any("non-finite" in p for p in rep["problems"])


def test_the_probe_cli_reads_a_checkpoint_the_trainer_would_write(tmp_path, monkeypatch):
    from diffusers import PixArtTransformer2DModel as P
    monkeypatch.setattr(P, "from_pretrained", classmethod(lambda cls, *a, **k: cls(**TINY_PIXART)))
    monkeypatch.setattr(P, "load_config", classmethod(lambda cls, *a, **k: dict(TINY_PIXART)))
    m = pixart_model()
    ck = trained(m, 4)
    ck["args"].update({"action_dropout": 0.0, "tic_stride": 1, "resolved_control_bits": BITS})
    path = tmp_path / "0000300.pt"
    torch.save(ck, str(path))
    d = corpus(tmp_path / "val", (6000,))
    a = sp.build_parser().parse_args(["--ckpt", str(path), "--backbone", "pixart", "--pixart-path",
                                      backbones.PIXART_DEFAULT, "--latents-dir", d, "--episodes", "6000:6001",
                                      "--latent-channels", "4", "--context-frames", str(CTX), "--num-actions", "3",
                                      "--noise-buckets", "4", "--batch", "2", "--device", "cpu",
                                      "--out", str(tmp_path / "probe.json")])
    assert sp.main(a) == 0
    assert (tmp_path / "probe.json").is_file()


def test_the_probe_is_appended_to_the_runs_wandb_eval_run(tmp_path, monkeypatch):
    """`--wandb-run` logs the report at the step in the checkpoint's filename (0000300.pt)."""
    from diffusers import PixArtTransformer2DModel as P
    from wandb_stub import inits, logged, stub_wandb
    monkeypatch.setattr(P, "from_pretrained", classmethod(lambda cls, *a, **k: cls(**TINY_PIXART)))
    monkeypatch.setattr(P, "load_config", classmethod(lambda cls, *a, **k: dict(TINY_PIXART)))
    wb = stub_wandb()
    monkeypatch.setitem(sys.modules, "wandb", wb)
    m = pixart_model()
    ck = trained(m, 4)
    ck["args"].update({"action_dropout": 0.0, "tic_stride": 1, "resolved_control_bits": BITS})
    path = tmp_path / "0000300.pt"
    torch.save(ck, str(path))
    d = corpus(tmp_path / "val", (6000,))
    a = sp.build_parser().parse_args(["--ckpt", str(path), "--backbone", "pixart", "--pixart-path",
                                      backbones.PIXART_DEFAULT, "--latents-dir", d, "--episodes", "6000:6001",
                                      "--latent-channels", "4", "--context-frames", str(CTX), "--num-actions", "3",
                                      "--noise-buckets", "4", "--batch", "2", "--device", "cpu",
                                      "--out", str(tmp_path / "probe.json"), "--wandb-run", "040-unet-nexttic"])
    assert sp.main(a) == 0
    (kw,) = inits(wb)
    assert kw["id"] == "040-unet-nexttic-eval" and kw["dir"] == str(tmp_path / ".wandb")
    (row,) = logged(wb)
    assert row["step"] == 300
    assert "eval/probe/probe/loss" in row and "eval/probe/probe/newest_control_sensitivity" in row


def test_the_probe_groups_name_the_real_modules():
    """The names the probe looks for are the ones backbones.py defines."""
    src = open(os.path.join(REPO, "backbones.py")).read()
    assert "self.mlp = nn.Sequential(" in src and "self.pos = nn.Parameter(" in src
    assert "module.control_history = ControlHistoryEmbedder(" in src
    assert "self.pooled_control = nn.Linear(" in src


# ---------------------------------------------------------------------------------------
# the gates run all of it
# ---------------------------------------------------------------------------------------

def dry(tmp_path, **env):
    e = {**os.environ, "DRY": "1", "DOOM_ROOT": str(tmp_path), **env}
    p = subprocess.run(["bash", GATES], capture_output=True, text=True, env=e)
    assert p.returncode == 0, p.stderr
    return p.stdout.splitlines()


def test_the_val_inventory_is_exact(tmp_path):
    lines = [ln for ln in dry(tmp_path, VAES="sd15,sd35") if ln.startswith("DRY gate1c val inventory")]
    assert len(lines) == 2
    for ln in lines:
        assert "--expect-ids 6000:6100" in ln and "--raw-tics" in ln and "--split-file" in ln
        assert "split_val.json" in ln and "--check-only" in ln


def test_the_emitted_window_check_reads_the_training_corpus(tmp_path):
    lines = [ln for ln in dry(tmp_path, VAES="sd15,sd35") if "check_emitted_windows.py" in ln]
    assert len(lines) == 2
    assert any("latents_arnold_dense_pertic/arenas" in ln and "--latent-channels 4" in ln for ln in lines)
    assert any("latents_arnold_dense_pertic_sd35/arenas" in ln and "--latent-channels 16" in ln for ln in lines)
    assert all("--episodes 0:2000" in ln for ln in lines)


def test_the_probes_and_the_resume_follow_the_smoke(tmp_path):
    lines = dry(tmp_path)
    smoke = min(i for i, ln in enumerate(lines) if "--steps 300" in ln)
    probes = [i for i, ln in enumerate(lines) if "smoke_probe.py" in ln]
    resume = [i for i, ln in enumerate(lines) if "--steps 310" in ln]
    readback = min(i for i, ln in enumerate(lines) if "eval_tf.py" in ln)
    assert len(probes) == 2 and len(resume) == 2
    assert smoke < min(probes) and max(probes) < min(resume) and max(resume) < readback
    for i in resume:
        assert "--resume" in lines[i] and "0000300.pt" in lines[i] and "--ckpt-every 10" in lines[i]
        last = lines[i].rsplit("--results-dir ", 1)[1].split()[0]
        assert last.startswith(f"{tmp_path}/results_smoke"), "the resume must stay in the smoke directory"
    sd35 = [lines[i] for i in probes if "--backbone sd35" in lines[i]][0]
    assert "0000300.pt" in sd35 and "--latent-channels 16" in sd35


def test_the_readback_covers_live_and_ema_at_both_horizons(tmp_path):
    lines = [ln for ln in dry(tmp_path) if "eval_tf.py" in ln and "--backbone unet" in ln]
    assert len(lines) == 4
    combos = {("--use-ema" in ln, ln.split("--horizon-tics ")[1].split()[0]) for ln in lines}
    assert combos == {(False, "1"), (False, "4"), (True, "1"), (True, "4")}


def test_the_certificate_requires_every_new_gate():
    for g in ("1c inventory val", "1e emitted windows", "4b smoke probes", "4c resume"):
        assert g in gc.REQUIRED_GATES
