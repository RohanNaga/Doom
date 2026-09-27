"""The 16-channel (SD 3.5) path of the decoder tune, end to end on CPU.

The 2026-09-26 fit checks (60 updates, lr 1e-5, MSE only, frozen encoder, validation on episodes
6000 to 6099) moved the SD 3.5 decoder's validation LPIPS from 0.045 to 0.213 while its PSNR rose
0.7 dB. The SD 1 decoder, under the same 60 updates, moved from 0.0945 to 0.0965. These tests rule
the code path in or out as the cause. They build a tiny AutoencoderKL with SD 3.5's latent contract
(16 channels, scale 1.5305, shift 0.0609, no quant convs) and pin five things.

  * At lr 0, the launcher's own SD 3.5 fit recipe reads the same validation metrics before, during
    and after training (channels-last, streamed frames, held-out validation directory, MSE only,
    reported LPIPS). It also saves the stock weights bit for bit. So "before" and "after" come from
    one code path, and only the weights can move them.
  * The tune never applies the scale or the shift: it decodes the encoder's raw posterior mean. A
    stored SD 3.5 latent, `(z - 0.0609) * 1.5305` in fp16 as encode_parquet.py writes it and
    denormalised as eval_tf.py reads it, decodes to the picture the tune scores. Each misplaced
    shift decodes to a different picture. With SD 1's zero shift, every misplacement is the right
    answer, which is why the SD 1 path cannot expose a shift bug.
  * The validation target and the reconstruction are [-1, 1] RGB on the 240 real rows. The tune
    feeds the encoder exactly the tensor encode_parquet.py fed it. PSNR is the [-1, 1] one (peak to
    peak 2), and the HUD crop is the last 32 real rows, never the padding.
  * A tuned 16-channel decoder saves, reloads, and passes the contract check under
    rescore_tuned_decoder.sh's own eval_tf.py flags. It then decodes exactly as it did before the save.
  * Channels-last and the fused AdamW kernel treat a 16-channel conv_in exactly as a 4-channel one.

    python -m pytest paper/fixtures/test_decoder_sd35_path.py -q
"""
import copy
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

from diffusers.models import AutoencoderKL  # noqa: E402

import doomdit_utils  # noqa: E402
import encode_parquet  # noqa: E402
import eval_tf  # noqa: E402
import finetune_decoder  # noqa: E402
from doomdit_utils import build_vae, denormalize_latents, normalize_latents  # noqa: E402
from finetune_decoder import HUD_ROWS, evaluate, save_vae, to_tensor, trainable_decoder_params  # noqa: E402
from test_decoder_mse_launcher import calls_to, launch  # noqa: E402
from test_decoder_validation import episodes, fake_frame, fake_readers  # noqa: E402
from test_rescore_tuned_decoder import dry, root_with_runs  # noqa: E402

SD35 = {"latent_channels": 16, "scaling_factor": 1.5305, "shift_factor": 0.0609}
SD1 = {"latent_channels": 4, "scaling_factor": 0.18215, "shift_factor": None}
VAL_IDS = list(range(6000, 6004))


def tiny(contract, seed=0):
    """A two-level AutoencoderKL under a real latent contract. SD 3.5's has no quant convs, SD 1's has both."""
    torch.manual_seed(seed)
    quant = contract["shift_factor"] is None
    return AutoencoderKL(in_channels=3, out_channels=3,
                         down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2,
                         block_out_channels=(8, 8), layers_per_block=1, norm_num_groups=8, sample_size=32,
                         use_quant_conv=quant, use_post_quant_conv=quant, **contract)


class RecordingLPIPS(torch.nn.Module):
    """Stands in for `lpips.LPIPS`. It records every pair it is shown and returns a gradient distance
    per frame, so both range and sharpness move it."""

    def __init__(self, net="alex", verbose=False):
        super().__init__()
        self.seen = []

    def forward(self, a, b):
        self.seen.append((a.detach().clone(), b.detach().clone()))
        grad = lambda t: torch.cat([(t[..., 1:, :] - t[..., :-1, :]).flatten(1),  # noqa: E731
                                    (t[..., 1:] - t[..., :-1]).flatten(1)], 1)
        return (grad(a) - grad(b)).abs().mean(1).view(-1, 1, 1, 1)


def install_lpips(monkeypatch):
    """Make `import lpips` inside finetune_decoder.main return the recorder. Returns the list of instances."""
    made = []

    def build(net="alex", verbose=False):
        made.append(RecordingLPIPS(net, verbose))
        return made[-1]
    monkeypatch.setitem(sys.modules, "lpips", types.SimpleNamespace(LPIPS=build))
    return made


def stored_latents(vae, frames, contract):
    """The corpus latents of these frames, exactly as encode_parquet.py writes them (fp16, normalised)."""
    lat = encode_parquet.encode_batch(vae, frames, "cpu", torch.float32,
                                      scale=contract["scaling_factor"], shift=contract["shift_factor"])
    return torch.from_numpy(lat)


# ---------------------------------------------------------------------------------------
# the launcher's SD 3.5 fit at lr 0
# ---------------------------------------------------------------------------------------

def test_at_lr_0_the_sd35_fit_reads_one_code_path_before_and_after(tmp_path, monkeypatch):
    # the finetune_decoder.py argv decoder_mse.sh builds for `FIT=1 SPACE=sd35 ... 0 16 60`
    _, calls, _ = launch(tmp_path / "launcher", "0", "16", "60", SPACE="sd35", FIT="1")
    [tune] = calls_to(calls, "finetune_decoder.py")
    a = finetune_decoder.build_parser().parse_args(tune[2:])
    assert (a.latent_channels, a.scaling_factor, a.shift_factor) == (16, 1.5305, 0.0609)
    assert a.vae_subfolder == "vae" and a.channels_last and a.lpips_weight == 0 and a.report_lpips
    assert a.lr == 1e-5 and a.mse_rows == 240 and a.stride == 1

    # the same recipe on a tiny SD 3.5-shaped autoencoder laid out as the hub repo is (<id>/vae), at lr 0
    stock = tiny(SD35)
    stock.save_pretrained(str(tmp_path / "sd35" / "vae"))
    arenas = episodes(str(tmp_path / "arenas"), list(range(4)) + VAL_IDS)
    fake_readers(monkeypatch)
    monkeypatch.setattr(finetune_decoder, "build_vae", build_vae)       # the real contract check
    made = install_lpips(monkeypatch)
    a.__dict__.update(vae_id=str(tmp_path / "sd35"), cache_dir=None, lr=0.0, max_steps=3, batch_size=2,
                      val_frames=len(VAL_IDS), val_dir=arenas, val_ids="6000:6004", stream_dir=arenas,
                      stream_ids="0:4", frame_cache="", workers=0, val_every=2, device="cpu",
                      out_dir=str(tmp_path / "run"))
    finetune_decoder.main(a)
    m = json.load(open(tmp_path / "run" / "metrics.json"))

    assert m["latent_contract"] == SD35 and m["steps"] == 3
    assert set(m["before"]) == {"psnr", "psnr_hud", "lpips", "lpips_hud"}
    # before, the mid-run read, the epoch-end read and after: one number each, however the model is laid out
    assert [e["step"] for e in m["history"]] == [2, 3]
    for read in [m["after"]] + [{k: v for k, v in e.items() if k != "step"} for e in m["history"]]:
        assert read == pytest.approx(m["before"], rel=0, abs=1e-6)
    saved = AutoencoderKL.from_pretrained(str(tmp_path / "run" / "vae"))
    assert all(torch.equal(v, stock.state_dict()[k]) for k, v in saved.state_dict().items())
    assert saved.config.shift_factor == 0.0609 and saved.post_quant_conv is None

    # what the validation LPIPS read: [-1, 1] RGB, 240 real rows, then the last 32 of them as the HUD
    [lp] = made
    frames = np.stack([fake_frame(e, 0) for e in VAL_IDS])
    target = to_tensor(frames, "cpu")[:, :, :240]
    assert len(lp.seen) == 2 * 4                                         # full and HUD, four reads
    for full, hud in zip(lp.seen[0::2], lp.seen[1::2]):
        assert torch.equal(full[0], target) and full[1].shape == target.shape
        assert torch.equal(hud[0], target[:, :, -HUD_ROWS:]) and torch.equal(hud[1], full[1][:, :, -HUD_ROWS:])
        assert full[1].min() >= -1 and full[1].max() <= 1 and target.min() < -0.9 and target.max() > 0.9

    # the picture the tune scored after training is the one a stored SD 3.5 latent decodes to in eval_tf.py
    after = lp.seen[-2][1] * 0.5 + 0.5
    with torch.no_grad():
        z = stock.encode(to_tensor(frames, "cpu")).latent_dist.mean
        exact = eval_tf.decode(stock, normalize_latents(z, 1.5305, 0.0609), 1.5305, 0.0609)
        stored = eval_tf.decode(stock, stored_latents(stock, frames, SD35), 1.5305, 0.0609)
    assert (exact - after).abs().max() < 1e-5
    assert (stored - after).abs().max() < 2e-3                          # fp16 storage only


# ---------------------------------------------------------------------------------------
# the shift, the range and the channel order
# ---------------------------------------------------------------------------------------

MISPLACED = {"shift added before the division": lambda s, sc, sh: (s + (sh or 0)) / sc,
             "shift subtracted": lambda s, sc, sh: s / sc - (sh or 0),
             "shift dropped": lambda s, sc, sh: s / sc}


def test_a_misplaced_shift_decodes_to_a_different_picture_and_sd1_cannot_show_it():
    vae = tiny(SD35).eval()
    frames = np.stack([fake_frame(6000 + i, 3) for i in range(3)])
    s = stored_latents(vae, frames, SD35).float()
    with torch.no_grad():
        z = vae.encode(to_tensor(frames, "cpu")).latent_dist.mean
        assert (denormalize_latents(s, 1.5305, 0.0609) - z).abs().max() < 1e-3   # the inverse, to fp16
        right = (vae.decode(denormalize_latents(s, 1.5305, 0.0609)).sample - vae.decode(z).sample).abs().max()
        for name, inv in MISPLACED.items():
            wrong = (vae.decode(inv(s, 1.5305, 0.0609)).sample - vae.decode(z).sample).abs().max()
            assert wrong > 20 * max(float(right), 1e-4), name
    # SD 1 stores no shift, so every misplacement is the correct inverse and its path passes regardless
    s1 = torch.randn(2, 4, 32, 40)
    for name, inv in MISPLACED.items():
        assert torch.equal(inv(s1, 0.18215, None), denormalize_latents(s1, 0.18215, None)), name


def test_validation_psnr_is_the_minus_one_to_one_rgb_one_on_the_real_rows():
    vae = tiny(SD35).eval()
    frames = np.stack([fake_frame(6000 + i, 5) for i in range(3)])
    frames[:, :120, :, 0], frames[:, :120, :, 2] = 255, 0              # a red top half, so RGB order shows
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() * (2 / 255) - 1
    assert torch.allclose(to_tensor(frames, "cpu")[:, :, :240], x, atol=1e-6)
    assert torch.equal(to_tensor(frames, "cpu")[:, 0, :120], torch.ones(3, 120, 320))
    # the tune's encoder input is the corpus encoder's input, padding included
    assert torch.equal(to_tensor(frames, "cpu"), encode_parquet.to_input(frames, "cpu"))
    assert torch.equal(to_tensor(frames, "cpu")[:, :, 240:], torch.zeros(3, 3, 16, 320))

    lp = RecordingLPIPS()
    m = evaluate(vae, frames, "cpu", lp)
    with torch.no_grad():
        y = vae.decode(vae.encode(torch.nn.functional.pad(x, (0, 0, 0, 16))).latent_dist.mean).sample
    y = y.clamp(-1, 1)[:, :, :240]
    psnr = lambda a, b: (10 * torch.log10(4 / ((a - b) ** 2).flatten(1).mean(1))).mean()  # noqa: E731
    assert m["psnr"] == pytest.approx(float(psnr(x, y)), abs=1e-4)
    assert m["psnr_hud"] == pytest.approx(float(psnr(x[:, :, 208:], y[:, :, 208:])), abs=1e-4)
    assert m["lpips"] == pytest.approx(float(lp(x, y).mean()), abs=1e-6)


# ---------------------------------------------------------------------------------------
# the tuned decoder, saved and read back by the rescore
# ---------------------------------------------------------------------------------------

def test_a_tuned_16_channel_decoder_reloads_the_way_the_rescore_reads_it(tmp_path):
    root = root_with_runs(tmp_path)
    [argv] = [ln[6:] for ln in dry(tmp_path, root) if ln[1:3] == ["042-sd35-nexttic", "h1"]]
    r = eval_tf.build_parser().parse_args(argv)
    assert r.vae_path == f"{root}/vae_decoder_sd35_mse/vae" and r.vae_subfolder == ""
    assert (r.latent_channels, r.latent_scale, r.latent_shift) == (16, 1.5305, 0.0609)

    live = tiny(SD35)
    with torch.no_grad():                                               # a decoder the tune has moved
        for p in trainable_decoder_params(live):
            p.add_(0.01 * torch.randn_like(p))
    live.to(memory_format=torch.channels_last)
    out = save_vae(live, str(tmp_path / "vae_decoder_sd35_mse"), channels_last=True)
    s = stored_latents(tiny(SD35), np.stack([fake_frame(6000, 0)]), SD35)
    with torch.no_grad():
        want = eval_tf.decode(live.eval(), s, 1.5305, 0.0609)
        for path in (out, os.path.dirname(out)):                        # <out>/vae and <out> name one decoder
            got = build_vae(path, r.vae_subfolder, "cpu", None, latent_channels=r.latent_channels,
                            scaling_factor=r.latent_scale, shift_factor=r.latent_shift)
            assert doomdit_utils.latent_contract(got) == SD35 and got.quant_conv is None
            assert all(torch.equal(v, live.state_dict()[k]) for k, v in got.state_dict().items())
            assert (eval_tf.decode(got, s, 1.5305, 0.0609) - want).abs().max() < 1e-6
    with pytest.raises(SystemExit, match="contract"):                   # a read declaring another shift stops here
        build_vae(out, "", "cpu", None, latent_channels=16, scaling_factor=1.5305, shift_factor=0.0)


# ---------------------------------------------------------------------------------------
# channels-last and the fused kernel on conv_in
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("contract", [SD1, SD35], ids=["4ch", "16ch"])
def test_channels_last_and_the_fused_kernel_treat_conv_in_alike(contract):
    """NCHW with the plain AdamW against NHWC with the fused one: the same losses and the same weights.

    The fused kernel is the CPU one here. It pairs a parameter with its gradient and moments by memory, so
    it is only right while all of them share the parameter's strides. Autograd gives every gradient its
    parameter's strides and the moments are allocated like the parameter; both are checked below.
    """
    ref = tiny(contract)
    nhwc = copy.deepcopy(ref).to(memory_format=torch.channels_last)
    conv_in = nhwc.decoder.conv_in.weight
    assert conv_in.shape[1] == contract["latent_channels"] and not conv_in.is_contiguous()
    p_ref, p_nhwc = trainable_decoder_params(ref), trainable_decoder_params(nhwc)
    assert any(p is conv_in for p in p_nhwc) and not any(p.requires_grad for p in nhwc.encoder.parameters())
    try:
        opt_nhwc = torch.optim.AdamW(p_nhwc, lr=1e-3, weight_decay=0.0, fused=True)
    except RuntimeError as e:                                           # torch < 2.4 has no fused CPU kernel
        pytest.skip(str(e))
    opt_ref = torch.optim.AdamW(p_ref, lr=1e-3, weight_decay=0.0)
    start = conv_in.detach().clone()
    for step in range(3):
        x = to_tensor([fake_frame(step, i) for i in range(2)], "cpu")
        losses = []
        for vae, opt in ((ref, opt_ref), (nhwc, opt_nhwc)):
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                z = vae.encode(x.contiguous(memory_format=torch.channels_last) if vae is nhwc else x).latent_dist.mean
            loss = torch.mean((vae.decode(z).sample[:, :, :240] - x[:, :, :240]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_decoder_params(vae), 1.0)
            opt.step()
            losses.append(loss.item())
        assert losses[0] == pytest.approx(losses[1], rel=1e-5)
        assert conv_in.grad.stride() == conv_in.stride()
        assert all(opt_nhwc.state[conv_in][k].stride() == conv_in.stride() for k in ("exp_avg", "exp_avg_sq"))
    assert not torch.equal(conv_in, start)                              # the step moved conv_in
    for k, v in nhwc.state_dict().items():
        assert torch.allclose(v, ref.state_dict()[k], atol=1e-6), k
