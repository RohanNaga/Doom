"""The SD 3.5 Medium row's launcher and evaluation suite, gated on the CPU before any GPU time.

`scripts/spiderman/launch_sd35.sh` and `after_sd35.sh` are the only two files that decide what the
headline row actually trains and is scored with, and neither can be run here. So the launcher grows
a `DRY=1` mode that prints the command it would hand tmux, and these checks read that string: the
shared recipe, the 16-channel flags, the spike guard, and the two-card accelerate form. Every flag
the printed command uses is then checked against `train_wm.py --help`, so a renamed flag fails here
rather than at 03:00 on the server.

The last check runs `eval_tf.py` end to end on the CPU at 16 channels with a small random
transformer and a small random autoencoder, which is the one part of the evaluation suite that is
new code rather than a new flag value.

    python -m pytest paper/fixtures/test_sd35_row.py -q
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import backbones  # noqa: E402

LAUNCH = os.path.join(REPO, "scripts", "spiderman", "launch_sd35.sh")
AFTER = os.path.join(REPO, "scripts", "spiderman", "after_sd35.sh")
FIT = os.path.join(REPO, "scripts", "spiderman", "fit_sd35.sh")
RUN = "035-sd35-l32-aligned"
D = "/sata2/data/rnagabhi/doom"

# the recipe every aligned row shares, exactly as launch_pixart.sh / launch_unidiffuser.sh pin it
SHARED_RECIPE = ("--context-frames 32", "--num-actions 29", "--global-batch 32", "--lr 5e-5",
                 "--warmup 2000", "--clip 1.0", "--steps 90000", "--seed 0", "--action-dropout 0.0",
                 "--require-verified-transitions", "--val-every 1000", "--val-windows 1024",
                 "--ckpt-every 5000", "--keep-last 2")


def expanded(path):
    """The script's text with the path variables it defines at the top substituted in.

    The scripts write `$R/eval_tf_$S`, not the literal path, so a check that wants to read the real
    directory has to resolve `D`, `RUN`, `R` and `L` the way bash would. Substituting the longest
    names first keeps `$RUN` from being eaten by `$R`.
    """
    body = open(path).read()
    env = {}
    for name in ("D", "RUN", "R", "L"):
        m = re.search(rf"^{name}=(\S+)$", body, re.M)
        if m:
            env[name] = m.group(1)
    for _ in range(3):        # R is defined in terms of D, L in terms of D
        for name in sorted(env, key=len, reverse=True):
            value = env[name]
            for form in (f"${{{name}}}", f"${name}"):
                env = {k: v.replace(form, value) for k, v in env.items()}
                body = body.replace(form, value)
    return body


def dry(gpu, **env):
    """The command the launcher would hand tmux, with no side effects on this machine."""
    e = {**os.environ, "DRY": "1", **{k: str(v) for k, v in env.items()}}
    r = subprocess.run(["bash", LAUNCH, gpu], capture_output=True, text=True, env=e, cwd=REPO)
    return r


# ---- the scripts parse and are executable -----------------------------------------------------

@pytest.mark.parametrize("script", [LAUNCH, AFTER, FIT])
def test_the_scripts_are_valid_bash_and_executable(script):
    assert subprocess.run(["bash", "-n", script], capture_output=True, text=True).returncode == 0
    assert os.access(script, os.X_OK), f"{script} is not executable"


# ---- the launcher -----------------------------------------------------------------------------

def test_the_dry_run_prints_the_command_and_touches_nothing():
    r = dry("3")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "train_wm.py" in r.stdout
    assert "tmux new-session" not in r.stdout, "DRY must print the inner command, not the tmux wrapper call"
    assert not os.path.exists(D), "the dry run created the server's data root on this machine"


def test_the_launcher_pins_the_shared_recipe():
    cmd = dry("3").stdout
    for flag in SHARED_RECIPE:
        assert flag in cmd, flag
    assert f"--results-dir {D}/results_spiderman/{RUN}" in cmd


def test_the_launcher_carries_the_sixteen_channel_corpus_and_backbone():
    cmd = dry("3").stdout
    assert "--backbone sd35" in cmd
    assert "--warm-start stabilityai/stable-diffusion-3.5-medium" in cmd
    assert "--latent-channels 16" in cmd
    assert f"--latents-dir {D}/latents_arnold_sd35" in cmd
    assert f"--split {D}/split_arnold.json" in cmd


def test_the_launcher_turns_on_activation_checkpointing_and_the_spike_guard():
    cmd = dry("3").stdout
    assert "--grad-ckpt" in cmd
    assert "--skip-grad-norm 5" in cmd, "the guard that saved the UniDiffuser row"


def test_the_spike_guard_is_recorded_as_a_deviation_in_the_results_directory():
    body = expanded(LAUNCH)
    assert f"{D}/results_spiderman/{RUN}/DEVIATIONS.md" in body
    assert "skip-grad-norm" in body.split("DEVIATIONS.md", 1)[1][:800]


def test_a_single_card_runs_the_plain_interpreter_at_the_requested_micro_batch():
    cmd = dry("3").stdout
    assert "wanenc/bin/python train_wm.py" in cmd, "diffusers 0.40 lives in ~/wanenc, not the doom env"
    assert "accelerate" not in cmd
    assert "--per-gpu-batch 16" in cmd
    assert "CUDA_VISIBLE_DEVICES=3" in cmd
    assert "--per-gpu-batch 8" in dry("3", MB=8).stdout


def test_two_cards_launch_under_accelerate_with_no_accumulation():
    cmd = dry("2,3").stdout
    assert "accelerate launch --num_processes 2 --mixed_precision bf16 train_wm.py" in cmd
    assert "--per-gpu-batch 16" in cmd, "16 x 2 cards = the global batch of 32, so accum stays 1"
    assert "--global-batch 32" in cmd
    assert "CUDA_VISIBLE_DEVICES=2,3" in cmd


def test_a_micro_batch_that_cannot_reach_the_global_batch_is_refused():
    r = dry("2,3", MB=32)
    assert r.returncode != 0
    assert "32" in (r.stdout + r.stderr)


def test_the_extra_passthrough_reaches_the_trainer():
    assert "--optim adamw8bit" in dry("3", EXTRA="--optim adamw8bit").stdout


def test_the_launcher_is_idempotent_and_resumes():
    body = expanded(LAUNCH)
    assert "tmux has-session -t train-sd35" in body
    assert r'\"event\": \"end\"' in body, "a finished row must not be relaunched"
    assert "--resume" in body
    assert "tmux new-session -d -s train-sd35" in body
    assert f"{D}/logs/train_sd35.log" in body
    assert "HF_HUB_OFFLINE=0" in body, "the SD 3.5 repo is gated and may still need the hub"
    assert "TMPDIR=" in dry("3").stdout, "TMPDIR has to be inside the tmux string, not only exported"


def test_every_flag_the_launcher_passes_exists_in_the_trainer():
    cmd = dry("2,3", EXTRA="--optim adamw8bit").stdout
    flags = set(re.findall(r"(?<![\w-])--[a-z0-9][a-z0-9-]+", cmd.split("train_wm.py", 1)[1]))
    help_text = subprocess.run([sys.executable, os.path.join(REPO, "train_wm.py"), "--help"],
                               capture_output=True, text=True, cwd=REPO)
    assert help_text.returncode == 0, help_text.stderr[-2000:]
    unknown = sorted(f for f in flags if f not in help_text.stdout)
    assert not unknown, f"train_wm.py does not take {unknown}"


# ---- the evaluation suite ---------------------------------------------------------------------

def test_the_evaluators_take_the_sixteen_channel_flags():
    for script, flags in ((("eval_tf.py"), ("--latent-channels", "--sd35-path", "--latent-shift", "--vae-subfolder")),
                          (("rollout_eval.py"), ("--latent-channels", "--sd35-path", "--latent-shift",
                                                 "--idm-reencode-vae"))):
        r = subprocess.run([sys.executable, os.path.join(REPO, script), "--help"],
                           capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, r.stderr[-2000:]
        for f in flags:
            assert f in r.stdout, f"{script} has no {f}"


def test_the_evaluation_suite_scores_the_sixteen_channel_row_on_every_corpus():
    body = expanded(AFTER)
    assert "--latent-channels 16" in body and "--sd35-path" in body
    assert "--latent-scale 1.5305" in body and "--latent-shift 0.0609" in body, \
        "the SD 3.5 autoencoder's own (z - shift) * scale, not the SD 1.x 0.18215"
    for corpus in ("seen", "unseen", "unseen2"):
        assert f"{D}/latents_arnold_eval_sd35/{corpus}" in body or "eval_sd35/$S" in body, corpus
    assert f"{D}/raw_arnold_eval" in body
    assert "--use-ema" in body
    assert "--num-rollouts 256" in body and "--horizon 64" in body
    assert "--frames $F" in body and "for F in 16 32" in body
    assert "AFTER_SD35_DONE" in body
    assert f"{D}/logs/{RUN}_rollout.log" in body


def test_the_evaluation_suite_prefers_the_tuned_decoder_and_falls_back_to_the_stock_one():
    body = expanded(AFTER)
    assert f"{D}/vae_decoder_sd35_lpips/vae" in body
    assert "stabilityai/stable-diffusion-3.5-medium" in body, "the stock fallback"
    assert "decoder_used" in body, "which decoder ran has to be written down next to the numbers"


def test_the_evaluation_suite_round_trips_the_rollout_before_the_idm():
    body = expanded(AFTER)
    assert "--idm-reencode-vae" in body
    assert "sd-vae-ft-mse" in body, "the encoder the IDM's latents came from"
    assert "IDM" in body and "caveat" in body.lower(), "the reference caveat must be stated in the script"


def test_the_corpus_splits_are_read_from_beside_the_sixteen_channel_latents():
    body = expanded(AFTER)
    assert "latents_arnold_eval_sd35/split_" in body


# ---- the fit sweep ----------------------------------------------------------------------------

def test_the_fit_sweep_needs_no_corpus_and_no_split():
    """`--fit-check` builds `SyntheticWindows` and returns before the split is read
    (train_wm.py:74-76) and the split md5 is skipped too (train_wm.py:237), so the sweep can run
    while the 16-channel encode is still in progress. A SPLIT knob would be a flag that does
    nothing, so there is none."""
    import train_wm
    body = expanded(FIT)
    assert "--split" not in body and "--latents-dir" not in body
    assert "--fit-check 200" in body
    args = train_wm.build_parser().parse_args(
        ["--backbone", "sd35", "--fit-check", "200", "--per-gpu-batch", "16", "--global-batch", "32",
         "--context-frames", "32", "--num-actions", "29", "--split", "/does/not/exist.json",
         "--latents-dir", "/does/not/exist"])
    ds, val, ids = train_wm.build_loaders(args, 16)
    assert val is None and ids is None
    assert ds[0][0].shape == (512, 32, 40) and ds[0][1].shape == (16, 32, 40)


# ---- eval_tf.py at 16 channels on the CPU -----------------------------------------------------

TINY_SD3 = dict(sample_size=128, patch_size=2, in_channels=16, out_channels=16, num_layers=2,
                attention_head_dim=8, num_attention_heads=2, joint_attention_dim=32,
                caption_projection_dim=16, pooled_projection_dim=24, pos_embed_max_size=96,
                dual_attention_layers=(0,), qk_norm="rms_norm")


@pytest.fixture
def tiny_sd35_hub(monkeypatch):
    """Serve a small SD3 transformer wherever the 2.2B checkpoint would be pulled from."""
    from diffusers import SD3Transformer2DModel as S
    monkeypatch.setattr(S, "from_pretrained", classmethod(lambda cls, *a, **k: cls(**TINY_SD3)))
    monkeypatch.setenv("ACCELERATE_USE_CPU", "1")


@pytest.fixture(autouse=True)
def lpips_available(monkeypatch):
    """A stand-in where the package is absent, so the 16-channel path runs rather than skipping."""
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


def test_eval_tf_samples_and_decodes_a_sixteen_channel_window_on_the_cpu(tiny_sd35_hub, tmp_path):
    import eval_tf
    from diffusers.models import AutoencoderKL

    torch.manual_seed(0)
    latents = tmp_path / "latents"
    latents.mkdir()
    rng = np.random.RandomState(0)
    frames, ctx = 8, 4
    for ep in (0, 1):
        np.save(latents / f"ep_{ep:05d}_latents.npy", rng.randn(frames, 16, 32, 40).astype(np.float16))
        np.savez(latents / f"ep_{ep:05d}_meta.npz", action=rng.randint(0, 3, frames).astype(np.int64),
                 tic=(np.arange(frames) * 4).astype(np.int64), map_id=np.full(frames, 1, np.int64),
                 chain_id=np.zeros(frames, np.int64))
    split = tmp_path / "split.json"
    json.dump({"train": [0], "val": [1]}, open(split, "w"))

    vae_dir = tmp_path / "vae"
    AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                  up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(4, 4, 4, 4),
                  layers_per_block=1, norm_num_groups=2, sample_size=256, latent_channels=16,
                  scaling_factor=1.5305, shift_factor=0.0609).eval().save_pretrained(str(vae_dir))

    model = backbones.build_model("sd35", 3, ctx, 4, grad_ckpt=False, warm_start=backbones.SD35_DEFAULT,
                                  action_dropout=0.0, latent_channels=16)
    ckpt = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "step": 5,
                "args": {"action_dropout": 0.0, "objective": "v", "latent_channels": 16}}, str(ckpt))

    out = tmp_path / "eval_tf_seen"
    eval_tf.main(eval_tf.build_parser().parse_args(
        ["--ckpt", str(ckpt), "--backbone", "sd35", "--latent-channels", "16",
         "--sd35-path", backbones.SD35_DEFAULT, "--vae-path", str(vae_dir),
         "--latent-scale", "1.5305", "--latent-shift", "0.0609",
         "--latents-dir", str(latents), "--split", str(split), "--subset", "val",
         "--context-frames", str(ctx), "--num-actions", "3", "--noise-buckets", "4",
         "--num-windows", "2", "--batch-size", "2", "--steps", "2", "--save-images", "0",
         "--out-dir", str(out)]))

    metrics = json.load(open(out / "metrics.json"))
    assert metrics["config"]["resolved_latent_channels"] == 16
    assert metrics["psnr_dec"]["n"] == 2
    assert np.isfinite(metrics["psnr_dec"]["mean"])
    # no parquet directory was given, so the raw-frame columns are absent rather than fabricated
    assert metrics.get("psnr_raw") is None
