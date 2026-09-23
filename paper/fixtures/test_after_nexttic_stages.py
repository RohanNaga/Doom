"""The evaluation launcher picks one checkpoint, honours RESCORE, and fails loudly.

Four defects, all of which produce a number rather than a crash:

  * `LAST=$(ls $R/[0-9]*.pt $R/snap_*.pt | sort | tail -1)` sorts lexicographically across two
    families, so `snap_0290000.pt` beats `0300000.pt`: whenever any snapshot exists, every numbered
    recovery checkpoint loses to it. Live came from `best.pt`, so the live/EMA pair was unpaired.
  * `[ ! -f $NPZ ] || [ "${RESCORE:-0}" != 1 ]` is true when the file is missing AND true when
    RESCORE is not 1, so the default reran existing rollouts and RESCORE=1 skipped them. The
    teacher-forced passes ignored RESCORE entirely.
  * the script waited for the trainer's `event=end`, which a manually stopped run never writes.
  * `AFTER_NEXTTIC_DONE` was printed unconditionally, so a failed scoring pass announced a finished
    evaluation.

The launcher is run FOR REAL against a throwaway root with a stub python, as the other launcher
tests do; the defects are in bash and the DRY path never reaches them.

    python -m pytest paper/fixtures/test_after_nexttic_stages.py -q
"""
import os
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import pick_checkpoint  # noqa: E402

AFTER = os.path.join(REPO, "scripts", "spiderman", "after_nexttic.sh")
RUN = "040-unet-nexttic"

STUB_PY = '''#!/usr/bin/env python3
"""Stands in for the real evaluation scripts: records its argv and writes the output file.

`select_checkpoint.py` and `decoder_provenance.py` are run for real (STUB_REAL_PY, STUB_REPO) when
the test names them: the selection and the decoder's claim are what those tests are about.
`make_dense_eval_splits.py --check-only` fails for the
corpora named in STUB_CHECK_FAIL. The checkpoint hash is STUB_SHA, or a name-derived fake.
"""
import json, os, re, sys
script = os.path.basename(sys.argv[1])
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
if script in ("select_checkpoint.py", "decoder_provenance.py") and os.environ.get("STUB_REAL_PY"):
    real = os.path.join(os.environ["STUB_REPO"], script)
    os.execv(os.environ["STUB_REAL_PY"], [os.environ["STUB_REAL_PY"], real] + sys.argv[2:])
def arg(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None
def step_of(path):
    if os.path.basename(path) == "best.pt":
        return 41000
    m = re.search(r"(\\d+)\\.pt$", path)
    return int(m.group(1)) if m else 0
if script == "pick_checkpoint.py":
    path = arg("--ckpt") or os.environ["STUB_PICK"]
    sha = os.environ.get("STUB_SHA", "sha-" + os.path.basename(path))
    print("%s %d %d %s" % (path, step_of(path), os.path.basename(path) != "best.pt", sha))
    sys.exit(int(os.environ.get("STUB_PICK_RC", "0")))
if script == "make_dense_eval_splits.py":
    corpus = os.path.basename(arg("--latents-dir") or "")
    sys.exit(1 if corpus in os.environ.get("STUB_CHECK_FAIL", "").split(",") else 0)
if script == "eval_tf.py":
    out = sys.argv[sys.argv.index("--out-dir") + 1]
    os.makedirs(out, exist_ok=True)
    ck, ema = arg("--ckpt"), "--use-ema" in sys.argv
    scores = json.loads(os.environ.get("STUB_SCORES") or "{}")
    psnr, lp = scores.get("%s|%s" % (os.path.basename(ck), "ema" if ema else "live"), [21.0, 0.3])
    fail_on = os.environ.get("STUB_TF_FAIL_ON")
    if fail_on and out.endswith(fail_on):
        sys.exit(1)
    if not os.environ.get("STUB_TF_NO_OUTPUT"):
        with open(os.path.join(out, "metrics.json"), "w") as f:
            json.dump({"psnr": 21.0, "psnr_raw": {"mean": psnr, "n": 8}, "lpips_raw": {"mean": lp, "n": 8},
                       "config": {"ckpt": ck, "use_ema": ema, "step": step_of(ck)}}, f)
    sys.exit(int(os.environ.get("STUB_TF_RC", "0")))
if script == "rollout_eval.py":
    if "--rollout" in sys.argv:
        with open(sys.argv[sys.argv.index("--out") + 1], "w") as f:
            f.write("rollouts")
    else:
        out = sys.argv[sys.argv.index("--out-dir") + 1]
        os.makedirs(out, exist_ok=True)
        # the real scoring pass writes rollout_eval.SCORE_FILE, which the harness passes in; the
        # launcher has to gate on that name and not on one the stage never produces
        with open(os.path.join(out, os.environ["STUB_SCORE_FILE"]), "w") as f:
            json.dump({"idm": 0.5}, f)
    sys.exit(int(os.environ.get("STUB_ROLL_RC", "0")))
sys.exit(0)
'''


def _root(tmp_path, checkpoints=(("snap_0290000.pt", 290000, True),), best=True, end=False):
    """A throwaway data root holding one run directory and the corpora the launcher looks for."""
    root = tmp_path / "root"
    r = root / "results_spiderman" / RUN
    r.mkdir(parents=True)
    (root / "repo").mkdir(parents=True)
    for c in ("val", "test", "arenas_678"):
        (root / "latents_arnold_dense_pertic_eval" / c).mkdir(parents=True)
    for name, step, has_ema in checkpoints:
        ck = {"model": {"w": torch.zeros(4)}, "step": step}
        if has_ema:
            ck["ema"] = {"w": torch.zeros(4)}
        torch.save(ck, str(r / name))
    if best:
        torch.save({"model": {"w": torch.zeros(4)}, "step": 41000, "val_loss": 0.2}, str(r / "best.pt"))
    if end:
        (r / "log.jsonl").write_text('{"event": "end"}\n')
    return root, r


def _run(tmp_path, root, r, timeout=90, **env):
    py = tmp_path / "stubpy"
    py.write_text(STUB_PY)
    py.chmod(0o755)
    log = tmp_path / "stub.log"
    if log.exists():
        log.unlink()            # one log per run: the calls of the previous run are not this run's
    import rollout_eval
    e = {**os.environ, "DOOM_ROOT": str(root), "PY": str(py), "PY_SD35": str(py),
         "STUB_LOG": str(log), "STUB_PICK": str(r / "snap_0290000.pt"),
         "STUB_SCORE_FILE": rollout_eval.SCORE_FILE,
         "CORPORA": "val", "NUM_WINDOWS": "8", **env}
    proc = subprocess.run(["bash", AFTER, "0", "unet"], capture_output=True, text=True,
                          env=e, timeout=timeout)
    calls = log.read_text().splitlines() if log.exists() else []
    return proc, calls


def _tf_calls(calls):
    return [c for c in calls if os.path.basename(c.split()[0]) == "eval_tf.py"]


def _calls_to(calls, script):
    return [c for c in calls if os.path.basename(c.split()[0]) == script]


# ---------------------------------------------------------------------------------------
# one checkpoint, chosen by its stored step
# ---------------------------------------------------------------------------------------

def test_a_newer_recovery_checkpoint_beats_an_older_snapshot(tmp_path):
    """The reproduction: lexicographically `snap_0290000.pt` sorts after `0300000.pt`."""
    root, r = _root(tmp_path, checkpoints=(("snap_0290000.pt", 290000, True),
                                           ("0300000.pt", 300000, True)))
    names = sorted(os.path.basename(p) for p in pick_checkpoint.candidates(str(r)))
    assert names[-1] == "snap_0290000.pt", "the old lexicographic order is not what it was"
    got = pick_checkpoint.pick(str(r))
    assert os.path.basename(got["path"]) == "0300000.pt" and got["step"] == 300000


def test_the_step_is_read_from_the_file_not_the_name(tmp_path):
    root, r = _root(tmp_path, checkpoints=(("snap_0000001.pt", 777000, True),))
    assert pick_checkpoint.pick(str(r))["step"] == 777000


def test_only_a_checkpoint_carrying_an_ema_can_serve_both_variants(tmp_path):
    root, r = _root(tmp_path, checkpoints=(("snap_0290000.pt", 290000, False),))
    with pytest.raises(SystemExit, match="carries an EMA"):
        pick_checkpoint.pick(str(r), require_ema=True)
    assert pick_checkpoint.pick(str(r))["has_ema"] is False


def test_a_named_step_and_a_named_path_are_both_selectable(tmp_path):
    root, r = _root(tmp_path, checkpoints=(("snap_0290000.pt", 290000, True),
                                           ("0300000.pt", 300000, True)))
    assert pick_checkpoint.pick(str(r), step=290000)["step"] == 290000
    with pytest.raises(SystemExit, match="step 12345"):
        pick_checkpoint.pick(str(r), step=12345)
    named = str(r / "0300000.pt")
    assert pick_checkpoint.pick(ckpt=named)["path"] == named
    with pytest.raises(SystemExit, match="no checkpoint at"):
        pick_checkpoint.pick(ckpt=str(r / "nope.pt"))


def test_no_checkpoint_at_all_is_an_error(tmp_path):
    root, r = _root(tmp_path, checkpoints=())
    with pytest.raises(SystemExit, match="no numbered or snapshot checkpoint"):
        pick_checkpoint.pick(str(r))


def test_the_launcher_no_longer_sorts_checkpoint_names():
    code = [ln for ln in open(AFTER).read().splitlines() if not ln.lstrip().startswith("#")]
    assert not any("sort | tail -1" in ln for ln in code), "the lexicographic selection is back"
    assert any("pick_checkpoint.py" in ln for ln in code)


def test_live_and_ema_come_from_the_same_file(tmp_path):
    root, r = _root(tmp_path)
    proc, calls = _run(tmp_path, root, r, CKPT=str(r / "snap_0290000.pt"))
    assert proc.returncode == 0, proc.stderr
    by_out = {c.split("--out-dir")[1].split()[0]: c for c in _tf_calls(calls)}
    live = by_out[str(r / "eval_tf_val")]
    ema = by_out[str(r / "eval_tf_val_ema")]

    def ckpt_of(c):
        t = c.split()
        return t[t.index("--ckpt") + 1]

    assert ckpt_of(live) == ckpt_of(ema) == str(r / "snap_0290000.pt")
    assert "--use-ema" in ema and "--use-ema" not in live


def test_best_is_kept_as_a_separately_labelled_variant(tmp_path):
    root, r = _root(tmp_path)
    proc, calls = _run(tmp_path, root, r, CKPT=str(r / "snap_0290000.pt"))
    best = [c for c in _tf_calls(calls) if c.endswith("eval_tf_val_best")]
    assert best and str(r / "best.pt") in best[0]
    proc, calls = _run(tmp_path, root, r, CKPT=str(r / "snap_0290000.pt"), BEST="0", RESCORE="1")
    assert not [c for c in _tf_calls(calls) if c.endswith("eval_tf_val_best")]


def test_the_scored_checkpoint_is_recorded(tmp_path):
    root, r = _root(tmp_path)
    proc, _ = _run(tmp_path, root, r, CKPT=str(r / "snap_0290000.pt"))
    assert proc.returncode == 0
    text = (r / "scored_checkpoint.txt").read_text()
    assert "step=290000" in text and "snap_0290000.pt" in text


# ---------------------------------------------------------------------------------------
# RESCORE, one rule for every stage
# ---------------------------------------------------------------------------------------

def test_by_default_an_existing_score_is_not_recomputed(tmp_path):
    root, r = _root(tmp_path)
    first, calls1 = _run(tmp_path, root, r, CKPT=str(r / "snap_0290000.pt"))
    assert first.returncode == 0 and _tf_calls(calls1)
    second, calls2 = _run(tmp_path, root, r, CKPT=str(r / "snap_0290000.pt"))
    assert second.returncode == 0
    assert _tf_calls(calls2) == [], _tf_calls(calls2)
    assert _calls_to(calls2, "rollout_eval.py") == [], calls2


def test_rescore_recomputes_every_validation_stage(tmp_path):
    """The inverted test: the DEFAULT reran existing rollouts and RESCORE=1 skipped them. The
    validation stage no longer rolls out at all (test rollouts belong to the sealed stage, see
    test_eval_stages.py), so what RESCORE must redo here is every teacher-forced pass."""
    root, r = _root(tmp_path)
    _run(tmp_path, root, r, CKPT=str(r / "snap_0290000.pt"))
    again, calls = _run(tmp_path, root, r, CKPT=str(r / "snap_0290000.pt"), RESCORE="1")
    assert again.returncode == 0
    assert len(_tf_calls(calls)) == 4, _tf_calls(calls)        # live, ema, best, h4
    assert _calls_to(calls, "rollout_eval.py") == [], "the validation stage rolled out on test"


def test_the_teacher_forced_pass_honours_rescore(tmp_path):
    """`eval_tf` skipped on `[ -f metrics.json ]` alone and never looked at RESCORE. A result is now
    reused only under its own key, and RESCORE=1 redoes it anyway."""
    root, r = _root(tmp_path)
    _run(tmp_path, root, r, CKPT=str(r / "snap_0290000.pt"))
    assert (r / "eval_tf_val" / "metrics.json.key").is_file()
    _, without = _run(tmp_path, root, r, CKPT=str(r / "snap_0290000.pt"))
    assert not [c for c in _tf_calls(without) if c.endswith("eval_tf_val")]
    _, with_flag = _run(tmp_path, root, r, CKPT=str(r / "snap_0290000.pt"), RESCORE="1")
    assert [c for c in _tf_calls(with_flag) if c.endswith("eval_tf_val")]


def _score_output_name():
    """The file `rollout_eval.py --score` writes, read out of that module rather than hardcoded."""
    import rollout_eval
    src = open(os.path.join(REPO, "rollout_eval.py")).read()
    assert "os.path.join(args.out_dir, SCORE_FILE)" in src, \
        "the scoring pass no longer names its output through SCORE_FILE"
    return rollout_eval.SCORE_FILE


def test_the_score_stage_is_gated_on_the_file_it_actually_writes():
    """The defect: the gate tested `rollout_metrics_test/metrics.json`, which this stage has never
    written, so `should_run` was always true and the whole scoring pass -- 256 decodes -- reran on
    every invocation, RESCORE or not. The behaviour is exercised in test_eval_stages.py."""
    name = _score_output_name()
    text = open(AFTER).read()
    assert 'M=$R/rollout_metrics_$S' in text
    assert f'run_or_skip "$S" rollout_score "$M/{name}" "$KEY"' in text, name
    assert 'rollout_metrics_test/metrics.json' not in text, "the wrong filename is back"


def test_the_rescore_condition_is_the_intended_one():
    text = open(AFTER).read()
    assert '[ "$RESCORE" = 1 ] && return 0' in text
    assert '[ "${RESCORE:-0}" != 1 ]' not in text, "the inverted test is back"


# ---------------------------------------------------------------------------------------
# when to run, and what DONE means
# ---------------------------------------------------------------------------------------

def test_a_named_checkpoint_does_not_wait_for_an_end_event(tmp_path):
    """A manually stopped run never writes `event=end`; the old script waited for ever."""
    root, r = _root(tmp_path, end=False)
    proc, calls = _run(tmp_path, root, r, timeout=60, CKPT=str(r / "snap_0290000.pt"))
    assert proc.returncode == 0 and _tf_calls(calls)
    assert "not waiting for event=end" in proc.stdout


def test_a_named_step_also_does_not_wait(tmp_path):
    root, r = _root(tmp_path, end=False)
    proc, calls = _run(tmp_path, root, r, timeout=60, STEP="290000")
    assert proc.returncode == 0 and _tf_calls(calls)
    pick = _calls_to(calls, "pick_checkpoint.py")
    assert pick and "--step 290000" in pick[0]


def test_a_failed_scoring_stage_is_not_reported_as_done(tmp_path):
    root, r = _root(tmp_path)
    proc, calls = _run(tmp_path, root, r, CKPT=str(r / "snap_0290000.pt"), STUB_TF_RC="1")
    assert proc.returncode != 0
    assert "AFTER_NEXTTIC_FAILED" in proc.stdout + proc.stderr
    assert "AFTER_NEXTTIC_DONE" not in proc.stdout
    assert "AFTER_NEXTTIC_STAGE_FAILED" in proc.stderr


def test_a_missing_checkpoint_stops_the_evaluation(tmp_path):
    root, r = _root(tmp_path)
    proc, _ = _run(tmp_path, root, r, CKPT=str(r / "snap_0290000.pt"), STUB_PICK_RC="1")
    assert proc.returncode == 3
    assert "no checkpoint to score" in proc.stderr


def test_a_clean_run_says_which_step_it_scored(tmp_path):
    root, r = _root(tmp_path)
    proc, _ = _run(tmp_path, root, r, CKPT=str(r / "snap_0290000.pt"))
    assert "AFTER_NEXTTIC_DONE 040-unet-nexttic step=290000" in proc.stdout


def test_the_dry_output_names_every_stage(tmp_path):
    root, r = _root(tmp_path)
    e = {**os.environ, "DRY": "1", "DOOM_ROOT": str(root)}
    proc = subprocess.run(["bash", AFTER, "0", "unet"], capture_output=True, text=True, env=e)
    assert proc.returncode == 0, proc.stderr
    assert "DRY pick" in proc.stdout, proc.stdout
    for variant in ("val ", "val_ema ", "val_best ", "val_h4 "):
        assert f"DRY eval_tf {variant}" in proc.stdout, variant
    assert "DRY score" not in proc.stdout, "the validation stage must not score rollouts"
    sealed = subprocess.run(["bash", AFTER, "0", "unet"], capture_output=True, text=True,
                            env={**e, "CORPORA": "test"})
    assert sealed.returncode == 0, sealed.stderr
    for tag in ("DRY require", "DRY rollout", "DRY score"):
        assert tag in sealed.stdout, sealed.stdout
