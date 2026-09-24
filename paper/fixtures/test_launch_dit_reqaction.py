"""The next two Spiderman launches: a DiT-XL/2 row and a U-Net conditioned on the requested action id.

`launch_nexttic.sh` knew `unet | sd35 | pixart`, one fixed run name per backbone, and the gates knew
two backbones. The two rows these tests pin, both on the 4-channel corpus with the 040 recipe:

  043-dit-nexttic             DiT-XL/2 from the local ImageNet checkpoint the stride-4 DiT rows
                              started from ($D/weights/DiT-XL-2-256x256.pt).
  044-unet-nexttic-reqaction  the 040 U-Net with ACTION_HISTORY=0: one token for Arnold's requested
                              action id instead of 32 executed-control tokens.

Everything here is a dry run, a stub run against a throwaway root, or a tiny model on the CPU:
nothing trains or touches a GPU.

    python -m pytest paper/fixtures/test_launch_dit_reqaction.py -q
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import gate_certificate as gc  # noqa: E402
from test_nexttic_defects3 import U, _launch_dry  # noqa: E402

SCRIPTS = os.path.join(REPO, "scripts", "spiderman")


# ---------------------------------------------------------------------------------------
# 1. the dit backbone in launch_nexttic.sh
# ---------------------------------------------------------------------------------------

def test_dit_starts_from_the_local_checkpoint_the_stride4_dit_rows_used(tmp_path):
    out = _launch_dry(tmp_path, "dit")
    assert f"--backbone dit --latent-channels 4 --warm-start {tmp_path}/weights/DiT-XL-2-256x256.pt " in out
    for older in ("launch_aligned_spiderman.sh", "launch_seed1.sh", "side_queue2.sh"):
        assert "--warm-start $D/weights/DiT-XL-2-256x256.pt" in open(os.path.join(SCRIPTS, older)).read(), older


def test_dit_trains_on_the_four_channel_corpus_under_its_own_run(tmp_path):
    out = _launch_dry(tmp_path, "dit")
    assert f"--latents-dir {tmp_path}/latents_arnold_dense_pertic/arenas " in out
    assert f"--val-latents-dir {tmp_path}/latents_arnold_dense_pertic_eval/val " in out
    assert f"--results-dir {tmp_path}/results_spiderman/043-dit-nexttic " in out
    assert f">> {tmp_path}/logs/train_043-dit-nexttic.log" in out
    assert "_sd35" not in out


def test_dit_carries_the_040_recipe_flag_for_flag(tmp_path):
    """Only what names the backbone and the run differs from the certified U-Net command: the stride-4
    DiT rows passed no DiT-only flag beyond the warm start, so there is nothing else to carry."""
    unet = gc.flag_pairs(_launch_dry(tmp_path, "unet", CERT_QUERY="1", PY_UNET=U))
    dit = gc.flag_pairs(_launch_dry(tmp_path, "dit", CERT_QUERY="1", PY_UNET=U))
    assert {p.split()[0] for p in set(unet) ^ set(dit)} == {"--backbone", "--warm-start", "--results-dir"}
    assert dit[0] == U, "the DiT runs under the 4-channel interpreter"


def test_dit_takes_the_action_history_and_checkpointing_knobs(tmp_path):
    assert "--action-history 32 " in _launch_dry(tmp_path, "dit")
    assert "--action-history 0 " in _launch_dry(tmp_path, "dit", ACTION_HISTORY="0")
    assert "--grad-ckpt" not in _launch_dry(tmp_path, "dit")
    assert _launch_dry(tmp_path, "dit", GRAD_CKPT="1").count("--grad-ckpt") == 1


def test_dit_builds_from_a_local_checkpoint_with_the_executed_control_history(tmp_path, monkeypatch):
    """`--backbone dit --warm-start <local .pt> --action-history 32` over 19 control bits is a supported
    path of build_model: the ImageNet weights load, the control embedder is attached, the class table
    it replaces is frozen. DiT-S/2 stands in for DiT-XL/2: the same code at 1/20 of the parameters."""
    import backbones
    import train_wm
    small = backbones.DiT_models["DiT-S/2"]
    imagenet = small(input_size=32, in_channels=4, num_classes=1000, learn_sigma=True)
    ck = tmp_path / "DiT-XL-2-256x256.pt"
    torch.save(imagenet.state_dict(), ck)
    monkeypatch.setitem(backbones.DiT_models, "DiT-XL/2", small)
    m = backbones.build_model("dit", 29, 32, 10, grad_ckpt=False, warm_start=str(ck), action_dropout=0.0,
                              action_history=32, control_bits=19)
    assert (m.control_history.length, m.control_history.bits) == (32, 19)
    assert not any(p.requires_grad for p in m.dit.y_embedder.parameters())
    w = m.dit.x_embedder.proj.weight
    assert torch.equal(w[:, -4:], imagenet.x_embedder.proj.weight) and not w[:, :-4].any(), \
        "the ImageNet kernel belongs on the noisy target, zeros on the 32 context latents"
    assert torch.equal(m.dit.blocks[0].attn.qkv.weight, imagenet.blocks[0].attn.qkv.weight)
    out = m(torch.randn(2, 4, 32, 40), torch.full((2,), 300), torch.randint(0, 2, (2, 32, 19)).float(),
            torch.randn(2, 128, 32, 40), torch.zeros(2, dtype=torch.long))
    assert out.shape == (2, 4, 32, 40)
    # and the certified command parses into exactly that request
    cmd = _launch_dry(tmp_path, "dit", CERT_QUERY="1").split("train_wm.py", 1)[1].split()
    a = train_wm.build_parser().parse_args(cmd)
    assert (a.backbone, a.warm_start, a.action_history, a.tic_stride, a.action_dropout) == \
        ("dit", f"{tmp_path}/weights/DiT-XL-2-256x256.pt", 32, 1, 0.0)


def test_the_dit_sees_which_controls_occurred_but_not_their_order():
    """The DiT has no cross-attention, so its control tokens are averaged into the adaLN vector
    (backbones.DiTWorldModel). The MLP acts per token and the positions are added before the mean,
    so the mean is a bag of the controls: reversing them leaves the output unchanged while flipping
    the newest changes it. The launcher header says so; this keeps the header true."""
    import backbones
    torch.manual_seed(0)
    m = backbones.DiTWorldModel(num_actions=29, context_frames=4, noise_buckets=10, model_name="DiT-S/2",
                                action_dropout=0.0, grad_ckpt=False, action_history=4, control_bits=19)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():   # adaLN-Zero and the zero output head hide every conditioning signal at init
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    m.eval()
    x, ctx = torch.randn(2, 4, 32, 40, generator=g), torch.randn(2, 16, 32, 40, generator=g)
    t, bucket = torch.full((2,), 500), torch.zeros(2, dtype=torch.long)
    c = torch.randint(0, 2, (2, 4, 19), generator=g).float()
    c[:, -1] = 1 - c[:, 0]                              # the newest differs from the oldest
    flipped = c.clone()
    flipped[:, -1] = 1 - flipped[:, -1]
    with torch.no_grad():
        base = m(x, t, c, ctx, bucket)
        reordered = m(x, t, c.flip(1), ctx, bucket)
        changed = m(x, t, flipped, ctx, bucket)
    assert (base - reordered).abs().max() < 1e-5, "the DiT now sees the order; update the launcher header"
    assert (base - changed).abs().max() > 1e-3
    head = open(os.path.join(SCRIPTS, "launch_nexttic.sh")).read().split("\nset -u", 1)[0]
    assert "not their order" in " ".join(head.replace("#", " ").split())


def test_a_certified_dit_launch_starts_in_its_own_session(tmp_path):
    from test_launch_pin import certify, launch, root_for_launch
    root, sha, bindir = root_for_launch(tmp_path)
    certify(tmp_path, root, bindir, backbones=("dit",))
    p, started = launch(tmp_path, root, bindir, backbone="dit")
    assert p.returncode == 0, p.stderr
    assert len(started) == 1 and "-s train-dit-nexttic " in started[0]
    assert f"--results-dir {root}/results_spiderman/043-dit-nexttic " in started[0]
    p, started = launch(tmp_path, root, bindir, backbone="unet")
    assert p.returncode != 0 and started == [], "a DiT certificate certified a U-Net launch"
