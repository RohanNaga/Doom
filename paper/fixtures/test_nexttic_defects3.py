"""Astra's third review of the launch gates and the evaluator (2026-09-23), and the launch knobs.

  1. `after_nexttic.sh` took an ACTIVE evaluator for an interrupted one: two simultaneous sealed
     invocations both passed the seal, the second as "re-entered", and test was scored twice. One
     evaluator per run now holds an exclusive process lock for as long as it scores; the persistent
     seal stays separate.

Everything here is a dry run or a stub run: nothing encodes, trains or touches a GPU.

    python -m pytest paper/fixtures/test_nexttic_defects3.py -q
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from test_after_nexttic_stages import AFTER, STUB_PY, _tf_calls  # noqa: E402
from test_eval_stages import root_with, selected  # noqa: E402

# ---------------------------------------------------------------------------------------
# 1. one evaluator per run: an active scorer is not an interrupted one
# ---------------------------------------------------------------------------------------

# Wraps the evaluation stub: every eval_tf.py call first marks that scoring has begun, then holds
# the card for STUB_TF_SLEEP seconds, so a second invocation arrives while the first is scoring.
SLOW_PY = '''#!/usr/bin/env python3
import os, sys, time
if os.path.basename(sys.argv[1]) == "eval_tf.py":
    open(os.environ["STUB_BUSY"], "a").write("%d\\n" % os.getpid())
    time.sleep(float(os.environ.get("STUB_TF_SLEEP", "0")))
os.execv(os.environ["STUB_INNER"], [os.environ["STUB_INNER"]] + sys.argv[1:])
'''


def _slow_env(tmp_path, root, r, tag, **env):
    import rollout_eval
    inner = tmp_path / "stubpy"
    inner.write_text(STUB_PY)
    inner.chmod(0o755)
    slow = tmp_path / "slowpy"
    slow.write_text(SLOW_PY)
    slow.chmod(0o755)
    return {**os.environ, "DOOM_ROOT": str(root), "PY": str(slow), "PY_SD35": str(slow),
            "STUB_INNER": str(inner), "STUB_LOG": str(tmp_path / f"stub_{tag}.log"),
            "STUB_BUSY": str(tmp_path / "busy"), "STUB_PICK": str(r / "0300000.pt"),
            "STUB_SCORE_FILE": rollout_eval.SCORE_FILE, "STUB_REAL_PY": sys.executable, "STUB_REPO": REPO,
            "NUM_WINDOWS": "8", "SELECT_WINDOWS": "4", **env}


def _calls(tmp_path, tag):
    p = tmp_path / f"stub_{tag}.log"
    return p.read_text().splitlines() if p.exists() else []


def _outs(calls):
    return [c.split("--out-dir")[1].split()[0] for c in _tf_calls(calls)]


def _two_at_once(tmp_path, root, r, *args, **env):
    """Start evaluator A, wait until it is inside a scoring call, then run evaluator B to completion."""
    a = subprocess.Popen(["bash", AFTER, "0", "unet", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=_slow_env(tmp_path, root, r, "a", STUB_TF_SLEEP="4", **env))
    busy = tmp_path / "busy"
    t0 = time.time()
    while not busy.exists() and time.time() - t0 < 60 and a.poll() is None:
        time.sleep(0.1)
    assert busy.exists(), "evaluator A never reached a scoring call"
    b = subprocess.run(["bash", AFTER, "0", "unet", *args], capture_output=True, text=True, timeout=120,
                       env=_slow_env(tmp_path, root, r, "b", **env))
    out, err = a.communicate(timeout=180)
    return (a.returncode, out, err), b


def test_two_simultaneous_sealed_invocations_score_test_once(tmp_path):
    """Astra's reproduction: both exited 0 and test was scored by both."""
    root, r = root_with(tmp_path)
    proc, _, sel = selected(tmp_path, root, r)
    assert proc.returncode == 0 and sel, proc.stderr
    (a_rc, _, a_err), b = _two_at_once(tmp_path, root, r, CORPORA="test")
    assert a_rc == 0, a_err
    assert b.returncode != 0, "the second evaluator ran alongside the first"
    assert "AFTER_NEXTTIC_BUSY" in b.stderr and "after_nexttic.lock" in b.stderr, b.stderr
    assert _tf_calls(_calls(tmp_path, "b")) == [], "the second evaluator scored test"
    assert sorted(_outs(_calls(tmp_path, "a"))) == sorted([str(r / "eval_tf_test"), str(r / "eval_tf_test_h4")])
    assert (r / "sealed" / "test" / "complete").is_file()
    # the seal saw one entry, never a re-entry by a process that was not interrupted
    assert "re-entered" not in (r / "sealed" / "test" / "seal").read_text()


def test_a_selection_cannot_overlap_another_evaluation_of_the_run(tmp_path):
    """The lock is per run and covers every stage: a selection made while another evaluation of the
    run is scoring (validation, another selection, a sealed corpus) is refused rather than raced."""
    root, r = root_with(tmp_path)
    (a_rc, _, a_err), b = _two_at_once(tmp_path, root, r, "--select")
    assert a_rc == 0, a_err
    assert b.returncode != 0 and "AFTER_NEXTTIC_BUSY" in b.stderr, b.stderr
    assert _tf_calls(_calls(tmp_path, "b")) == []


def test_the_lock_is_released_when_the_evaluator_exits(tmp_path):
    """A lock that outlived its holder would read as permanently busy; the kernel drops a flock when
    the last descriptor closes, so the next invocation finishes an interrupted seal."""
    root, r = root_with(tmp_path)
    selected(tmp_path, root, r)
    first = subprocess.run(["bash", AFTER, "0", "unet"], capture_output=True, text=True, timeout=120,
                           env=_slow_env(tmp_path, root, r, "c", CORPORA="test", STUB_TF_FAIL_ON="eval_tf_test_h4"))
    assert first.returncode != 0 and "AFTER_NEXTTIC_BUSY" not in first.stderr
    again = subprocess.run(["bash", AFTER, "0", "unet"], capture_output=True, text=True, timeout=120,
                           env=_slow_env(tmp_path, root, r, "d", CORPORA="test"))
    assert again.returncode == 0, again.stderr
    assert _outs(_calls(tmp_path, "d")) == [str(r / "eval_tf_test_h4")], \
        "the interrupted seal was not finished by the next evaluator"
