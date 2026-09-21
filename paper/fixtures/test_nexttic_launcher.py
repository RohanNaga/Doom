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

import numpy as np
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
    out = dry(LAUNCH, ["2", "unet"], root=str(tmp_path), STEPS="50000", TRAIN_IDS="0:500", MB="8")
    assert "--steps 50000" in out and "--episode-ids 0:500" in out and "--per-gpu-batch 8" in out


def test_a_micro_batch_that_cannot_reach_the_global_batch_is_refused():
    r = dry_fail(LAUNCH, ["2", "unet"], MB="7")
    assert r.returncode != 0 and "global batch" in r.stderr


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
    out = dry(AFTER, ["3", "unet"], root=str(tmp_path))
    for corpus in ("val", "test", "arenas_678", "seen", "unseen", "unseen2"):
        assert f"eval_tf {corpus} " in out, f"{corpus} is not scored"


def test_the_rollout_horizon_is_given_in_tics(tmp_path):
    out = dry(AFTER, ["3", "unet"], root=str(tmp_path))
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
    """260 held-out episodes gate the launch; the 2,000 training episodes take days."""
    out = dry(ENCODE, root=str(tmp_path))
    tags = [ln.split()[1] for ln in out.splitlines() if ln.startswith("DRY ")]
    assert tags == ["val", "test", "unseen"]


def test_an_unknown_corpus_or_vae_is_refused():
    assert dry_fail(ENCODE, CORPUS="banana").returncode != 0
    assert dry_fail(ENCODE, VAE="banana").returncode != 0


# ---------------------------------------------------------------------------------------
# check_action_alignment.py
# ---------------------------------------------------------------------------------------

import check_action_alignment as caa  # noqa: E402

TURN_LEFT_BITS = "001000000"
TURN_RIGHT_BITS = "000100000"
FORWARD_BITS = "100000000"


def _turning_episode(shift=0, n=48, step=5.0):
    """A synthetic recording in which the control on row t turns the agent between t and t+1.

    `shift` injects an error of that many tics into the CONTROLS, which is exactly the mistake the
    script has to detect: the state is untouched, so a correct script must still name shift 0 the
    truth for `shift=0` and blame the injected offset otherwise.
    """
    intent = np.zeros(n, dtype=int)
    rng = np.random.RandomState(0)
    t = 0
    while t < n:                                       # runs of 4 tics, the action repeat
        intent[t:t + 4] = rng.choice([-1, 1])
        t += 4
    angle = np.zeros(n)
    for t in range(n - 1):
        angle[t + 1] = angle[t] + step * intent[t]     # row t's control moves t -> t+1
    controls = np.zeros((n, 9), dtype=np.float32)
    src = np.roll(intent, shift)
    controls[src > 0, caa.TURN_LEFT] = 1.0
    controls[src < 0, caa.TURN_RIGHT] = 1.0
    return controls, angle % 360.0


def test_the_unshifted_control_wins_on_a_correctly_aligned_episode():
    controls, angle = _turning_episode(shift=0)
    sc = caa.alignment_scores(controls, angle)
    acc = {s: v["accuracy"] for s, v in sc["yaw"].items()}
    assert acc["0"] == 1.0, acc
    assert acc["0"] > acc["-1"] and acc["0"] > acc["1"], acc
    assert caa.verdict(sc)["yaw"]["verdict"] == "aligned at shift 0"


@pytest.mark.parametrize("shift", [-1, 1])
def test_an_injected_off_by_one_is_named_and_signed(shift):
    """The negative controls: an error of one tic must show as a neighbour beating shift 0."""
    controls, angle = _turning_episode(shift=shift)
    sc = caa.alignment_scores(controls, angle)
    v = caa.verdict(sc)["yaw"]
    assert v["verdict"].startswith("OFF BY"), v
    # rolling the controls by s means the control that produced realised[t] now sits at row t+s
    assert v["best"] == str(shift), v
    assert sc["yaw"]["0"]["accuracy"] < sc["yaw"][str(shift)]["accuracy"]


def test_all_three_shifts_are_always_printed():
    controls, angle = _turning_episode()
    sc = caa.alignment_scores(controls, angle)
    assert sorted(sc["yaw"]) == ["-1", "0", "1"]
    assert all(sc["yaw"][s]["rows"] > 0 for s in sc["yaw"])


def test_a_signal_free_episode_is_called_inconclusive_not_aligned():
    """No turning at all: the script must refuse to certify the alignment."""
    n = 40
    controls = np.zeros((n, 9), dtype=np.float32)
    controls[:, caa.MOVE_FORWARD] = 1.0
    sc = caa.alignment_scores(controls, np.zeros(n))
    v = caa.verdict(sc)["yaw"]
    assert v["best"] is None or v["verdict"].startswith("inconclusive"), v


def test_the_yaw_difference_wraps_at_three_sixty():
    assert abs(caa.wrap_deg(359.0 - 1.0) - (-2.0)) < 1e-9
    assert abs(caa.wrap_deg(1.0 - 359.0) - 2.0) < 1e-9


def test_boundaries_are_the_rows_next_to_a_change_on_either_side():
    """Both sides, or an off-by-one of +1 tic scores 1.0 alongside shift 0 and hides."""
    intent = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    assert caa.boundaries(intent).tolist() == [True, False, False, True, True, False, False, True]


def test_position_alignment_is_scored_the_same_way():
    n, step = 40, 10.0
    rng = np.random.RandomState(1)
    intent = np.repeat(rng.choice([-1, 1], n // 4), 4)
    angle = np.zeros(n)                                 # facing +x throughout
    px = np.zeros(n)
    for t in range(n - 1):
        px[t + 1] = px[t] + step * intent[t]
    controls = np.zeros((n, 9), dtype=np.float32)
    controls[intent > 0, caa.MOVE_FORWARD] = 1.0
    controls[intent < 0, caa.MOVE_BACKWARD] = 1.0
    sc = caa.alignment_scores(controls, angle, px, np.zeros(n))
    assert sc["position"]["0"]["accuracy"] == 1.0
    assert caa.verdict(sc)["position"]["verdict"] == "aligned at shift 0"


def test_the_checker_runs_off_a_per_tic_latent_directory(tmp_path):
    """So it can be run on the server where the latents are, without the 1.7 TiB of parquet."""
    from pertic_fixtures import write_pertic_episode
    controls, angle = _turning_episode()
    btns = ["".join(str(int(v)) for v in row) for row in controls]
    d = str(tmp_path / "lat")
    write_pertic_episode(d, 0, np.zeros(len(btns), dtype=np.int64), buttons=btns)
    # the encoder stores `angle`, so add it the way a real sidecar has it
    meta_path = os.path.join(d, "ep_00000_meta.npz")
    meta = dict(np.load(meta_path))
    meta["angle"] = angle
    np.savez(meta_path, **meta)
    per = caa.from_latents(d, episodes=1)
    assert len(per) == 1
    assert caa.verdict(per[0][1])["yaw"]["verdict"] == "aligned at shift 0"
    assert caa.pool(per)["yaw"]["0"]["accuracy"] == 1.0


def test_the_parser_requires_exactly_one_source():
    with pytest.raises(SystemExit):
        caa.build_parser().parse_args([])
    with pytest.raises(SystemExit):
        caa.build_parser().parse_args(["--parquet", "a", "--latents-dir", "b"])
    assert caa.build_parser().parse_args(["--latents-dir", "b"]).episodes == 1
