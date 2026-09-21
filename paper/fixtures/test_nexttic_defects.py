"""Regression tests for the nine defects Astra found in the first next-tic build (Sep 21 2026).

One test per defect, each reproducing the specific way the original code was wrong rather than only
asserting the fixed behaviour. Defect (8), the embedding init-order regression, has its own file
(`test_backbone_init_parity.py`) because it needs the merge base's code to compare against.

    python -m pytest paper/fixtures/test_nexttic_defects.py -q
"""
import json
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

from pertic_fixtures import held_actions, write_pertic_episode  # noqa: E402

import check_action_alignment as caa  # noqa: E402
import eval_tf  # noqa: E402
import make_dense_eval_splits  # noqa: E402
import rollout_eval  # noqa: E402
import train_wm  # noqa: E402

SCRIPTS = os.path.join(REPO, "scripts", "spiderman")
BITS = 9


# ---------------------------------------------------------------------------------------
# (1) the resolved control width has to reach the checkpoint
# ---------------------------------------------------------------------------------------

def test_the_trainer_writes_the_resolved_control_width_not_the_flag(tmp_path):
    """`--control-bits` defaults to 0 and the width comes from the corpus, so a checkpoint storing
    `vars(args)` recorded 0 and both evaluators then built a width-0 control embedder."""
    d = str(tmp_path / "lat")
    rng = np.random.RandomState(0)
    for ep in range(3):
        btns = ["".join(str(int(b)) for b in rng.randint(0, 2, BITS)) for _ in range(40)]
        write_pertic_episode(d, ep, held_actions([0, 1, 2] * 4)[:40], buttons=btns)
    out = str(tmp_path / "run")
    from diffusers import PixArtTransformer2DModel as P
    tiny = dict(num_attention_heads=2, attention_head_dim=8, in_channels=4, out_channels=8,
                num_layers=2, caption_channels=32, sample_size=64, patch_size=2,
                cross_attention_dim=16, use_additional_conditions=False, norm_num_groups=2)
    real_from, real_cfg = P.from_pretrained, P.load_config
    P.from_pretrained = classmethod(lambda cls, *a, **k: cls(**tiny))
    P.load_config = classmethod(lambda cls, *a, **k: dict(tiny))
    try:
        train_wm.main(train_wm.build_parser().parse_args(
            ["--backbone", "pixart", "--warm-start", "PixArt-alpha/PixArt-XL-2-512x512",
             "--latents-dir", d, "--results-dir", out, "--tic-stride", "1", "--action-history", "8",
             "--context-frames", "8", "--episode-ids", "0:2", "--val-episode-ids", "2:3",
             "--num-actions", "3", "--noise-buckets", "4", "--per-gpu-batch", "2", "--global-batch", "2",
             "--steps", "2", "--warmup", "1", "--val-every", "2", "--val-windows", "2",
             "--ckpt-every", "2", "--ema-every", "1", "--num-workers", "0", "--action-dropout", "0.0"]))
    finally:
        P.from_pretrained, P.load_config = real_from, real_cfg

    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["resolved_control_bits"] == BITS
    assert cfg["control_bits"] == BITS, "vars(args) still records the unresolved flag"
    ck = torch.load(os.path.join(out, "best.pt"), map_location="cpu", weights_only=False)
    assert ck["args"]["control_bits"] == BITS
    # and both evaluators rebuild the right width from it
    ns = _ns(ckpt="x.pt")
    assert eval_tf.checkpoint_interface(ck, ns)["control_bits"] == BITS


def _ns(**kw):
    import types
    ns = types.SimpleNamespace(ckpt="x.pt", tic_stride=None, action_history=None)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_a_checkpoint_with_history_but_no_width_is_refused():
    """Rather than silently building an embedder that reads zero input features."""
    ck = {"args": {"tic_stride": 1, "action_history": 32, "control_bits": 0}}
    with pytest.raises(SystemExit, match="no button-vector width"):
        eval_tf.checkpoint_interface(ck, _ns())


def test_the_resolved_key_wins_over_the_flag():
    ck = {"args": {"tic_stride": 1, "action_history": 32, "control_bits": 0,
                   "resolved_control_bits": 19}}
    assert eval_tf.checkpoint_interface(ck, _ns())["control_bits"] == 19


# ---------------------------------------------------------------------------------------
# (2) every flag a launcher passes must exist in the target script's parser
# ---------------------------------------------------------------------------------------

def _dry(script, args, **env):
    e = {**os.environ, "DRY": "1", "DOOM_ROOT": "/tmp/nexttic-defects-root", **env}
    r = subprocess.run(["bash", os.path.join(SCRIPTS, script), *args],
                       capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _flags_of(module):
    p = module.build_parser()
    out = set()
    for a in p._actions:
        out |= set(a.option_strings)
    return out


@pytest.mark.parametrize("backbone", ["unet", "sd35", "pixart"])
def test_every_flag_after_nexttic_passes_exists_in_the_target_parser(backbone):
    """`--parquet-dir` was being handed to rollout_eval.py, which has no such flag and exits 2."""
    known = {"eval_tf.py": _flags_of(eval_tf), "rollout_eval.py": _flags_of(rollout_eval)}
    import fvd
    known["fvd.py"] = _flags_of(fvd)
    for line in _dry("after_nexttic.sh", ["3", backbone]).splitlines():
        parts = line.split()
        target = next((p for p in parts if p in known), None)
        if target is None:
            continue
        for tok in parts[parts.index(target) + 1:]:
            if tok.startswith("--"):
                assert tok in known[target], f"{target} has no {tok} (line: {line})"


def test_the_rollout_call_gets_no_parquet_dir():
    out = _dry("after_nexttic.sh", ["3", "unet"])
    roll = [ln for ln in out.splitlines() if "rollout_eval.py" in ln]
    assert roll and all("--parquet-dir" not in ln for ln in roll)
    tf = [ln for ln in out.splitlines() if "eval_tf.py" in ln]
    assert tf and all("--parquet-dir" in ln for ln in tf), "eval_tf still needs the raw frames"


def test_one_subset_key_is_used_everywhere():
    out = _dry("after_nexttic.sh", ["3", "unet"])
    for ln in out.splitlines():
        if "eval_tf.py" in ln or "rollout_eval.py" in ln:
            assert f"--subset {make_dense_eval_splits.SUBSET}" in ln, ln


# ---------------------------------------------------------------------------------------
# (3) the split files the evaluation script requires have to be created
# ---------------------------------------------------------------------------------------

def test_the_eval_split_is_written_from_the_encoded_episodes(tmp_path):
    d = str(tmp_path / "val")
    for ep in (6000, 6001, 6002):
        write_pertic_episode(d, ep, held_actions([0, 1]))
    path, split = make_dense_eval_splits.build(d, "val")
    assert path.endswith("split_val.json") and os.path.isfile(path)
    assert split[make_dense_eval_splits.SUBSET] == [6000, 6001, 6002]
    assert split["meta"]["episode_range"] == "6000:6003"


def test_every_corpus_uses_the_same_subset_key(tmp_path):
    for name, ids in (("val", (6000, 6001)), ("test", (7000, 7001)), ("arenas_678", (0, 1))):
        d = str(tmp_path / name)
        for ep in ids:
            write_pertic_episode(d, ep, held_actions([0, 1]))
        _, split = make_dense_eval_splits.build(d, name)
        assert make_dense_eval_splits.SUBSET in split, name
        assert split[make_dense_eval_splits.SUBSET] == list(ids)


def test_the_evaluation_script_reads_the_files_the_encoder_writes():
    """The names have to line up, or after_nexttic.sh waits on a file nobody creates."""
    out = _dry("after_nexttic.sh", ["3", "unet"])
    for name in ("split_val.json", "split_test.json", "split_arenas_678.json"):
        assert name in out, name
    enc = _dry("encode_nexttic.sh", [], CORPUS="evals")
    assert "make_dense_eval_splits.py" in open(os.path.join(SCRIPTS, "encode_nexttic.sh")).read()
    assert "DRY val" in enc


# ---------------------------------------------------------------------------------------
# (4) rollout noise has to be per window, not from the global generator
# ---------------------------------------------------------------------------------------

def test_rollout_noise_is_keyed_by_window_and_step():
    """A global generator makes the noise depend on batch size and step count, so two runs of the
    same rollout are not paired. Keyed streams give window (ep, start) step h the same noise always."""
    from eval_tf import window_noise
    shape = (2, 4, 8, 10)
    seed, chunk = 0, [(11, 0, 5), (12, 0, 9)]
    keys = [seed * 1_000_003 + ep * 7919 + s * 97 + 0 for ep, _, s in chunk]
    a = window_noise(shape, keys)
    # the same window in a batch of one gets the same noise
    assert torch.equal(window_noise((1, 4, 8, 10), [keys[1]])[0], a[1])
    # and a different step gets different noise
    later = [seed * 1_000_003 + ep * 7919 + s * 97 + 1 for ep, _, s in chunk]
    assert not torch.allclose(window_noise(shape, later)[0], a[0])


def test_the_rollout_passes_its_own_noise_to_the_sampler():
    src = open(os.path.join(REPO, "rollout_eval.py")).read()
    call = src[src.index("x = diffusion.ddim_sample"):]
    call = call[:call.index("\n")]
    assert "noise=noise" in call, "ddim_sample is still drawing from the global generator"


# ---------------------------------------------------------------------------------------
# (5) the IDM needs exactly four tics between judged frames
# ---------------------------------------------------------------------------------------

def _decision_mask(n, positions):
    m = np.zeros(n, dtype=bool)
    m[list(positions)] = True
    return m


def test_an_eight_tic_gap_between_decision_tics_is_dropped(tmp_path, monkeypatch):
    """The real case: an anti-stuck override between two accepted decisions leaves rows 48 and 56
    flagged, eight tics apart, and the IDM's positional table is for four."""
    H, L, K = 24, 8, 8
    # decision tics at 3, 7, 11, then a gap to 19 (eight tics), then 23
    k = np.flatnonzero(_decision_mask(H, [3, 7, 11, 19, 23]))
    step_ok = np.diff(k) == 4
    runs, start = [], 0
    for i, ok in enumerate(list(step_ok) + [False]):
        if not ok:
            runs.append((start, i + 1)); start = i + 1
    a, b = max(runs, key=lambda r: r[1] - r[0])
    kept = k[a:b]
    assert kept.tolist() == [3, 7, 11], "the eight-tic pair must not survive"
    assert np.all(np.diff(kept) == 4)
    assert L == 8 and K == 8


def test_the_idm_spacing_rule_is_in_the_code():
    src = open(os.path.join(REPO, "rollout_eval.py")).read()
    assert "np.diff(k) == 4" in src
    assert "EIGHT tics apart" in src, "the reason has to be written down next to the rule"


# ---------------------------------------------------------------------------------------
# (6) translation is a diagnostic, never a veto
# ---------------------------------------------------------------------------------------

def _yaw_episode(reps=40, step=5.0, episode=0):
    intent = np.tile(np.repeat(np.array([0, 1, 0, -1]), 4), reps)
    n = len(intent)
    angle = np.zeros(n)
    for t in range(n - 1):
        angle[t + 1] = angle[t] + step * intent[t]
    controls = np.zeros((n, BITS), dtype=np.float32)
    controls[intent > 0, caa.TURN_LEFT] = 1.0
    controls[intent < 0, caa.TURN_RIGHT] = 1.0
    return controls, angle % 360.0, np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.int64)


def test_only_yaw_gates():
    assert caa.GATE_AXIS == "yaw"
    src = open(os.path.join(REPO, "check_action_alignment.py")).read()
    assert "acceleration and friction" in src, "the reason has to be written down"


def _momentum_episode():
    """Yaw aligned at shift 0, translation lagging by a tic because velocity carries over."""
    controls, angle, tic, deaths = _yaw_episode(reps=10)
    n = len(angle)
    fwd = np.tile(np.repeat(np.array([1, 1, 0, 0]), 4), n // 16)[:n]
    controls = controls.copy()
    controls[fwd > 0, caa.MOVE_FORWARD] = 1.0
    px = np.zeros(n)
    v = 0.0
    for t in range(n - 1):
        v = 0.5 * v + 10.0 * fwd[max(t - 1, 0)]            # last tic's control drives this step
        px[t + 1] = px[t] + v
    return controls, angle, tic, deaths, px


def test_a_momentum_driven_translation_score_does_not_fail_the_gate(tmp_path):
    """Displacement between t and t+1 reflects momentum built before t, so the control one tic
    earlier legitimately explains it better. That must not report a correct corpus as misaligned."""
    controls, angle, tic, deaths, px = _momentum_episode()
    sc = caa.alignment_scores(controls, angle, tic, deaths, px, np.zeros(len(px)))
    # translation really does prefer a neighbour here, which is the whole point
    assert sc["position"]["shifts"]["-1"] > sc["position"]["shifts"]["0"]
    # four episodes, so the episode bootstrap has something to resample
    d = str(tmp_path / "mom")
    for ep in range(4):
        _write_latents(d, controls, angle, tic, deaths, px, episode=ep)
    code = caa.main(caa.build_parser().parse_args(
        ["--latents-dir", d, "--episodes", "4", "--min-rows", "8", "--bootstrap", "200",
         "--min-episodes", "4", "--min-per-class", "1"]))
    assert code == caa.EXIT_ALIGNED, "translation vetoed a correctly aligned corpus"


def _write_latents(d, controls, angle, tic, deaths, px=None, episode=0):
    btns = ["".join(str(int(v)) for v in row) for row in controls]
    write_pertic_episode(d, episode, np.zeros(len(btns), dtype=np.int64), buttons=btns,
                         tics=tic, deaths=deaths)
    meta_path = os.path.join(d, f"ep_{episode:05d}_meta.npz")
    meta = dict(np.load(meta_path))
    meta["angle"] = angle
    if px is not None:
        meta["pos_x"] = px
        meta["pos_y"] = np.zeros(len(px))
    np.savez(meta_path, **meta)


# ---------------------------------------------------------------------------------------
# (7) an empty verdict is inconclusive, not aligned
# ---------------------------------------------------------------------------------------

def test_no_episodes_is_inconclusive_and_nonzero(tmp_path):
    """`--episodes 0` produced empty verdicts, and `any(...)` over nothing is False, so the gate
    returned EXIT_ALIGNED having measured nothing at all."""
    d = str(tmp_path / "none")
    _write_latents(d, *_yaw_episode())
    code = caa.main(caa.build_parser().parse_args(
        ["--latents-dir", d, "--episodes", "0", "--min-rows", "8", "--bootstrap", "100"]))
    assert code == caa.EXIT_INCONCLUSIVE != caa.EXIT_ALIGNED


def test_an_empty_score_is_inconclusive():
    assert caa.verdict({"rows": 0, "shifts": {}, "per_class": {}})["verdict"] == "inconclusive"


def test_too_few_episodes_is_inconclusive():
    """The bootstrap resamples episodes, so one episode cannot bound the margin however many rows."""
    sc = caa.alignment_scores(*_yaw_episode())["yaw"]
    v = caa.verdict(sc, min_rows=8, draws=100, min_per_class=1)
    assert v["verdict"] == "inconclusive" and "episode(s) is below the floor" in v["reason"]


def test_a_thin_motion_class_is_inconclusive():
    sc = caa.alignment_scores(*_yaw_episode(reps=40))["yaw"]
    v = caa.verdict(sc, min_rows=8, draws=100, min_episodes=1, min_per_class=10_000)
    assert v["verdict"] == "inconclusive" and "fewer than" in v["reason"]


def test_the_bootstrap_resamples_episodes():
    """One episode gives no interval at all, rather than a spuriously tight one."""
    sc = caa.alignment_scores(*_yaw_episode())["yaw"]
    assert not np.isfinite(caa.bootstrap_margin_low(sc, draws=50))
    per = [(e, caa.alignment_scores(*_yaw_episode(), episode=e)) for e in range(4)]
    pooled = caa.merge(per)["yaw"]
    assert pooled["episodes"] == 4
    assert np.isfinite(caa.bootstrap_margin_low(pooled, draws=50))


def test_the_default_thresholds_are_the_tightened_ones():
    a = caa.build_parser().parse_args(["--latents-dir", "x"])
    assert a.min_yaw == caa.MIN_YAW_DEG == 0.25
    assert a.min_episodes == 20 and a.min_per_class == 100
    assert a.min_rows == 1000 and a.min_accuracy == 0.95 and a.margin == 0.20


# ---------------------------------------------------------------------------------------
# (9) the encoder's exit code, not the logging pipeline's
# ---------------------------------------------------------------------------------------

def test_the_encoder_launcher_reports_the_encoders_own_status():
    """`$?` after `echo ... | tee` is the pipeline's status, so a failed encode logged exit 0 and the
    launcher went on to the next corpus as though nothing had happened."""
    src = open(os.path.join(SCRIPTS, "encode_nexttic.sh")).read()
    assert "local RC=$?" in src
    assert "ENCODE_NEXTTIC_FAILED" in src
    assert "exit=$RC" in src and "exit=$? files=" not in src
    assert "exit $RC" in src


def test_the_encoder_launcher_still_succeeds_under_dry():
    out = _dry("encode_nexttic.sh", [], CORPUS="val")
    assert "ENCODE_NEXTTIC_DONE" in out and "ENCODE_NEXTTIC_FAILED" not in out


def test_a_failing_corpus_makes_the_launcher_exit_nonzero(tmp_path):
    """A real run with a data root that has no recordings: every corpus fails, and so must the script."""
    env = {**os.environ, "DOOM_ROOT": str(tmp_path), "CORPUS": "val",
           "PY": "/bin/false", "REPO": REPO}
    r = subprocess.run(["bash", os.path.join(SCRIPTS, "encode_nexttic.sh")],
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0
    assert "ENCODE_NEXTTIC_DONE" not in r.stdout
