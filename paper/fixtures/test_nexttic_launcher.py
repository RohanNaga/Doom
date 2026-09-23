"""The next-tic launchers, and the model-free action-alignment check that gates their launch.

Every launcher here is exercised only through its `DRY=1` path with `DOOM_ROOT` pointed at a
throwaway directory, which is the rule `test_launcher_safety.py` enforces: on Sep 21 2026 a test
that ran `encode_pertic.sh` for real started two multi-hour GPU encodes.

    python -m pytest paper/fixtures/test_nexttic_launcher.py -q
"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

SCRIPTS = os.path.join(REPO, "scripts", "spiderman")
LAUNCH = os.path.join(SCRIPTS, "launch_nexttic.sh")
AFTER = os.path.join(SCRIPTS, "after_nexttic.sh")
ENCODE = os.path.join(SCRIPTS, "encode_nexttic.sh")
BACKBONES = ("unet", "sd35", "pixart")


def dry(script, args=(), **env):
    """Run a launcher with its side effects disabled and its data root thrown away."""
    e = {**os.environ, "DRY": "1", "DOOM_ROOT": env.pop("root", "/tmp/nexttic-dry-root"), **env}
    r = subprocess.run(["bash", script, *args], capture_output=True, text=True, env=e)
    assert r.returncode == 0, f"{os.path.basename(script)} DRY exited {r.returncode}: {r.stderr}"
    return r.stdout


def dry_fail(script, args=(), **env):
    e = {**os.environ, "DRY": "1", "DOOM_ROOT": "/tmp/nexttic-dry-root", **env}
    return subprocess.run(["bash", script, *args], capture_output=True, text=True, env=e)


# ---------------------------------------------------------------------------------------
# launch_nexttic.sh
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("script", [LAUNCH, AFTER, ENCODE], ids=os.path.basename)
def test_the_launchers_parse(script):
    assert subprocess.run(["bash", "-n", script], capture_output=True).returncode == 0


@pytest.mark.parametrize("backbone", BACKBONES)
def test_dry_touches_nothing_and_prints_the_command(backbone, tmp_path):
    out = dry(LAUNCH, ["2", backbone], root=str(tmp_path))
    assert any(ln.startswith("DRY ") for ln in out.splitlines()), out
    assert list(tmp_path.iterdir()) == [], "the launcher wrote under the data root under DRY=1"
    assert str(tmp_path) in out, "DOOM_ROOT never reached the command"
    assert "/sata2/data/rnagabhi" not in out


@pytest.mark.parametrize("backbone", BACKBONES)
def test_every_row_predicts_the_next_tic_on_the_dense_corpus(backbone, tmp_path):
    out = dry(LAUNCH, ["2", backbone], root=str(tmp_path))
    assert "--tic-stride 1" in out
    assert "latents_arnold_dense_pertic" in out
    assert "--episode-ids 0:2000" in out and "--val-episode-ids 6000:6100" in out
    assert "--dense-segment arenas" in out, "the held-out-range guard is not armed"
    assert "--context-frames 32" in out


@pytest.mark.parametrize("backbone", BACKBONES)
def test_the_recipe_is_the_one_the_stride4_rows_used(backbone, tmp_path):
    out = dry(LAUNCH, ["2", backbone], root=str(tmp_path))
    for flag in ("--global-batch 32", "--lr 5e-5", "--warmup 2000", "--clip 1.0",
                 "--noise-aug-max 0.7", "--noise-buckets 10", "--ema-decay 0.9999",
                 "--action-dropout 0.0", "--objective v", "--seed 0"):
        assert flag in out, f"{backbone}: {flag} missing"


@pytest.mark.parametrize("backbone", BACKBONES)
def test_a_hand_stop_at_any_snapshot_leaves_something_to_evaluate(backbone, tmp_path):
    """The run is stopped at the freeze, not at --steps, so snapshots must be kept locally."""
    out = dry(LAUNCH, ["2", backbone], root=str(tmp_path))
    assert "--snapshot-every 10000" in out and "--local-snapshots" in out
    assert "--steps 400000" in out


def test_only_sd35_gets_gradient_checkpointing_and_the_spike_guard(tmp_path):
    flags = {b: dry(LAUNCH, ["2", b], root=str(tmp_path)) for b in BACKBONES}
    assert "--grad-ckpt" in flags["sd35"] and "--skip-grad-norm 5" in flags["sd35"]
    for b in ("unet", "pixart"):
        assert "--grad-ckpt" not in flags[b], f"{b} must not pay the 30% checkpointing cost"
        assert "--skip-grad-norm" not in flags[b]


def test_sd35_uses_its_own_sixteen_channel_corpus(tmp_path):
    out = dry(LAUNCH, ["2", "sd35"], root=str(tmp_path))
    assert "--latent-channels 16" in out and "latents_arnold_dense_pertic_sd35" in out
    four = dry(LAUNCH, ["2", "unet"], root=str(tmp_path))
    assert "--latent-channels 4" in four
    assert "latents_arnold_dense_pertic_sd35" not in four


@pytest.mark.parametrize("backbone", BACKBONES)
def test_the_default_start_is_the_public_pretrained_warm_start(backbone, tmp_path):
    """Rohan, Sep 20 2026: purity of the warm-start comparison. INIT is the opt-in."""
    out = dry(LAUNCH, ["2", backbone], root=str(tmp_path))
    assert "--init-from" not in out
    assert "--warm-start" in out
    with_init = dry(LAUNCH, ["2", backbone], root=str(tmp_path), INIT="/x/best.pt")
    assert "--init-from /x/best.pt" in with_init


@pytest.mark.parametrize("backbone", BACKBONES)
def test_action_history_defaults_to_thirty_two_executed_controls(backbone, tmp_path):
    out = dry(LAUNCH, ["2", backbone], root=str(tmp_path))
    assert "--action-history 32" in out
    off = dry(LAUNCH, ["2", backbone], root=str(tmp_path), ACTION_HISTORY="0")
    assert "--action-history 0" in off


def test_an_action_history_that_does_not_match_the_context_is_refused(tmp_path):
    r = dry_fail(LAUNCH, ["2", "unet"], ACTION_HISTORY="16")
    assert r.returncode != 0 and "must be 0 or equal" in r.stderr


def test_phase_conditioning_is_off_unless_asked_for(tmp_path):
    out = dry(LAUNCH, ["2", "unet"], root=str(tmp_path))
    assert "--phase-conditioning" not in out
    on = dry(LAUNCH, ["2", "unet"], root=str(tmp_path), PHASE="1")
    assert "--phase-conditioning" in on


def test_the_knobs_reach_the_command(tmp_path):
    out = dry(LAUNCH, ["2", "unet"], root=str(tmp_path), STEPS="50000", TRAIN_IDS="0:500",
              VAL_IDS="6500:6600")
    assert "--steps 50000" in out and "--episode-ids 0:500" in out
    assert "--val-episode-ids 6500:6600" in out


def test_a_micro_batch_that_cannot_reach_the_global_batch_is_refused():
    r = dry_fail(LAUNCH, ["2", "unet"], MB="7")
    assert r.returncode != 0 and "global batch" in r.stderr


@pytest.mark.parametrize("backbone", BACKBONES)
def test_every_backbone_fills_the_card_with_no_accumulation(backbone, tmp_path):
    """CLAUDE.md, Rohan Sep 17 2026: the micro-batch IS the global batch."""
    out = dry(LAUNCH, ["2", backbone], root=str(tmp_path))
    assert "--per-gpu-batch 32" in out and "--global-batch 32" in out


def test_a_micro_batch_that_would_accumulate_is_refused_by_name():
    r = dry_fail(LAUNCH, ["2", "unet"], MB="16")
    assert r.returncode != 0
    assert "accumulation of 2" in r.stderr and "fill-the-card" in r.stderr
    ok = dry(LAUNCH, ["2", "unet"], MB="16", ALLOW_ACCUM="1")
    assert "--per-gpu-batch 16" in ok


def test_two_cards_split_the_global_batch_without_accumulation(tmp_path):
    out = dry(LAUNCH, ["2,3", "unet"], root=str(tmp_path), MB="16")
    assert "--num_processes 2" in out and "--per-gpu-batch 16" in out


def test_gradient_checkpointing_can_be_turned_on_for_the_four_channel_rows(tmp_path):
    for b in ("unet", "pixart"):
        assert "--grad-ckpt" not in dry(LAUNCH, ["2", b], root=str(tmp_path))
        assert "--grad-ckpt" in dry(LAUNCH, ["2", b], root=str(tmp_path), GRAD_CKPT="1")
    # sd35 already has it; the knob must not double it
    out = dry(LAUNCH, ["2", "sd35"], root=str(tmp_path), GRAD_CKPT="1")
    assert out.count("--grad-ckpt") == 1


def test_fit_mode_measures_this_exact_configuration(tmp_path):
    out = dry(LAUNCH, ["2", "unet"], root=str(tmp_path), FIT="20")
    assert "--fit-check 20" in out and "--per-gpu-batch 32" in out and "--tic-stride 1" in out
    assert "tmux" not in out, "FIT must not launch a training session"


def test_the_loader_gets_enough_workers_for_a_corpus_that_misses_the_page_cache(tmp_path):
    out = dry(LAUNCH, ["2", "unet"], root=str(tmp_path))
    assert "--num-workers 12" in out
    assert "--num-workers 24" in dry(LAUNCH, ["2", "unet"], root=str(tmp_path), WORKERS="24")


def test_a_half_encoded_corpus_is_refused_unless_asked_for(tmp_path):
    out = dry(LAUNCH, ["2", "unet"], root=str(tmp_path))
    assert "--allow-partial" not in out
    assert "--allow-partial" in dry(LAUNCH, ["2", "unet"], root=str(tmp_path), ALLOW_PARTIAL="1")


def test_an_unknown_backbone_is_refused():
    r = dry_fail(LAUNCH, ["2", "wan"])
    assert r.returncode != 0 and "unknown backbone" in r.stderr


# ---------------------------------------------------------------------------------------
# after_nexttic.sh
# ---------------------------------------------------------------------------------------

def test_the_evaluation_script_scores_at_tic_spacing_and_at_equal_game_time(tmp_path):
    out = dry(AFTER, ["3", "unet"], root=str(tmp_path))
    assert "--tic-stride 1" in out
    assert "--horizon-tics 4" in out, "no equal-game-time row against the stride-4 models"
    assert "--horizon-tics 1" in out


def test_the_evaluation_script_covers_the_dense_and_the_original_corpora(tmp_path):
    """Validation is its own stage; the sealed corpora are scored in another invocation, after
    --select (docs/REVIEW_2026-09-22.md H2)."""
    assert "eval_tf val " in dry(AFTER, ["3", "unet"], root=str(tmp_path))
    out = dry(AFTER, ["3", "unet"], root=str(tmp_path), CORPORA="test arenas_678 seen unseen unseen2")
    for corpus in ("test", "arenas_678", "seen", "unseen", "unseen2"):
        assert f"eval_tf {corpus} " in out, f"{corpus} is not scored"


def test_the_rollout_horizon_is_given_in_tics(tmp_path):
    out = dry(AFTER, ["3", "unet"], root=str(tmp_path), CORPORA="test")
    assert "--horizon 256" in out, "256 tics is the 64 decision steps the stride-4 rows roll out"


def test_the_tuned_decoder_is_tested_with_two_separate_file_tests(tmp_path):
    """One `ls` of two paths succeeds when only one exists; that is how the SD 3.5 row silently
    scored against the stock decoder on Sep 20 2026."""
    with open(AFTER) as f:
        text = f.read()
    assert '[ -f "$TUNED/diffusion_pytorch_model.safetensors" ] || [ -f "$TUNED/diffusion_pytorch_model.bin" ]' in text
    assert "ls $TUNED" not in text and "ls \"$TUNED\"" not in text


def test_the_evaluation_script_reports_which_decoder_ran(tmp_path):
    out = dry(AFTER, ["3", "sd35"], root=str(tmp_path))
    assert "decoder:" in out and ("tuned" in out or "stock" in out)


# ---------------------------------------------------------------------------------------
# encode_nexttic.sh
# ---------------------------------------------------------------------------------------

def test_the_encoder_launcher_uses_the_decided_id_ranges(tmp_path):
    out = dry(ENCODE, root=str(tmp_path), CORPUS="all")
    split = json.load(open(os.path.join(REPO, "release", "dense_split.json")))["next_tic_runs"]
    for tag, ids in (("train", split["train_ids"]), ("val", split["val_ids"]),
                     ("test", split["test_ids"]), ("unseen", split["unseen_ids"])):
        line = [ln for ln in out.splitlines() if ln.startswith(f"DRY {tag} ")]
        assert line, f"{tag} not launched"
        assert f"--episode-ids {ids}" in line[0], f"{tag}: {line[0]}"


def test_the_encoder_launcher_keeps_every_tic_and_marks_the_decisions(tmp_path):
    out = dry(ENCODE, root=str(tmp_path), CORPUS="val")
    assert "--every-tic" in out and "--stride 4" in out


def test_the_encoder_launcher_can_write_the_sixteen_channel_corpus(tmp_path):
    out = dry(ENCODE, root=str(tmp_path), CORPUS="val", VAE="sd35")
    assert "--latent-channels 16" in out
    assert "--scaling-factor 1.5305" in out and "--shift-factor 0.0609" in out
    assert "stable-diffusion-3.5-medium" in out and "latents_arnold_dense_pertic_eval_sd35" in out


def test_the_encoder_launcher_passes_the_shard_through(tmp_path):
    out = dry(ENCODE, root=str(tmp_path), CORPUS="train", SHARD="1/4")
    assert "--shard 1/4" in out
    plain = dry(ENCODE, root=str(tmp_path), CORPUS="train")
    assert "--shard" not in plain


def test_the_encoder_launcher_is_evaluation_first_by_default(tmp_path):
    """260 held-out episodes gate the launch; the 2,000 training episodes take days.

    The shared canonical table comes first, before any corpus: every corpus is passed the same file
    as `--canonical`, so there is nothing to encode until it exists."""
    out = dry(ENCODE, root=str(tmp_path))
    tags = [ln.split()[1] for ln in out.splitlines() if ln.startswith("DRY ")]
    assert tags == ["canonical", "val", "test", "unseen"]


def test_an_unknown_corpus_or_vae_is_refused():
    assert dry_fail(ENCODE, CORPUS="banana").returncode != 0
    assert dry_fail(ENCODE, VAE="banana").returncode != 0
