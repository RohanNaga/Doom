"""What launches is what the gates certified: the training corpus is audited and the commit is pinned.

The 2026-09-22 review (H3) found three gaps between the gates and the launch:

  * gate 1 audited sidecars against the raw recordings for val only; the 2,000 training episodes,
    written by two different encoder versions, were never audited;
  * nothing checked that a stored latent array has as many rows as its sidecar, and at the same tics
    as the recording, over the training corpus;
  * `launch_nexttic.sh` ran `git pull` AFTER the gates, and a failed pull did not stop the launch,
    so the code that trained could differ from the code that passed.

The launcher now refuses to start unless `$D/GATES_COMMIT` names the commit its checkout is at, with
no tracked change. These tests run it for real against a throwaway root with a throwaway git
checkout and a fake `tmux` that only records what it was asked to do.

    python -m pytest paper/fixtures/test_launch_pin.py -q
"""
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import make_dense_eval_splits as mdes  # noqa: E402

LAUNCH = os.path.join(REPO, "scripts", "spiderman", "launch_nexttic.sh")
AFTER = os.path.join(REPO, "scripts", "spiderman", "after_nexttic.sh")
GATES = os.path.join(REPO, "scripts", "cluster", "gates.sh")

FAKE_TMUX = """#!/bin/bash
echo "tmux $*" >> "$TMUX_LOG"
[ "$1" = has-session ] && exit 1
exit 0
"""


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True).stdout.strip()


def checkout(path):
    """A throwaway git repository with one commit, standing in for `$D/repo`."""
    path.mkdir(parents=True)
    git(path, "init", "-q")
    git(path, "config", "user.email", "t@t")
    git(path, "config", "user.name", "t")
    (path / "train_wm.py").write_text("# placeholder\n")
    git(path, "add", "train_wm.py")
    git(path, "commit", "-q", "-m", "one")
    return git(path, "rev-parse", "HEAD")


def root_for_launch(tmp_path):
    root = tmp_path / "root"
    sha = checkout(root / "repo")
    (root / "latents_arnold_dense_pertic" / "arenas").mkdir(parents=True)
    (root / "latents_arnold_dense_pertic_eval" / "val").mkdir(parents=True)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "tmux").write_text(FAKE_TMUX)
    (bindir / "tmux").chmod(0o755)
    return root, sha, bindir


def launch(tmp_path, root, bindir, **env):
    log = tmp_path / "tmux.log"
    if log.exists():
        log.unlink()
    e = {**os.environ, "DOOM_ROOT": str(root), "PY": sys.executable, "TMUX_LOG": str(log),
         "PATH": f"{bindir}:{os.environ['PATH']}", **env}
    p = subprocess.run(["bash", LAUNCH, "0", "unet"], capture_output=True, text=True, env=e, timeout=60)
    calls = log.read_text().splitlines() if log.exists() else []
    return p, [c for c in calls if c.startswith("tmux new-session")]


def code_lines(path):
    with open(path) as f:
        return [ln for ln in f.read().splitlines() if not ln.lstrip().startswith("#")]


# ---------------------------------------------------------------------------------------
# nothing changes the checkout after the gates
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("script", [LAUNCH, AFTER], ids=os.path.basename)
def test_no_launcher_pulls_code(script):
    assert not [ln for ln in code_lines(script) if "git pull" in ln], \
        f"{os.path.basename(script)} changes the checkout it runs"


def test_without_a_gate_receipt_nothing_launches(tmp_path):
    root, sha, bindir = root_for_launch(tmp_path)
    p, started = launch(tmp_path, root, bindir)
    assert p.returncode != 0 and "gate receipt" in p.stderr
    assert started == []


def test_a_receipt_for_another_commit_is_refused(tmp_path):
    root, sha, bindir = root_for_launch(tmp_path)
    (root / "GATES_COMMIT").write_text("0" * 40 + " 2026-09-22T00:00:00 spaces=sd15\n")
    p, started = launch(tmp_path, root, bindir)
    assert p.returncode != 0 and "certified" in p.stderr and sha in p.stderr
    assert started == []


def test_the_certified_commit_launches_and_is_logged(tmp_path):
    root, sha, bindir = root_for_launch(tmp_path)
    (root / "GATES_COMMIT").write_text(f"{sha} 2026-09-22T00:00:00 spaces=sd15\n")
    p, started = launch(tmp_path, root, bindir)
    assert p.returncode == 0, p.stderr
    assert len(started) == 1 and "train-unet-nexttic" in started[0]
    log = (root / "logs" / "resumes.log").read_text()
    assert f"commit {sha}" in log and "certified" in log


def test_a_tracked_change_after_the_gates_is_refused(tmp_path):
    root, sha, bindir = root_for_launch(tmp_path)
    (root / "GATES_COMMIT").write_text(f"{sha} 2026-09-22T00:00:00\n")
    (root / "repo" / "train_wm.py").write_text("# edited after the gates\n")
    p, started = launch(tmp_path, root, bindir)
    assert p.returncode != 0 and "differ" in p.stderr
    assert started == []


def test_an_ungated_launch_must_be_asked_for_and_is_recorded(tmp_path):
    root, sha, bindir = root_for_launch(tmp_path)
    p, started = launch(tmp_path, root, bindir, ALLOW_UNGATED="1")
    assert p.returncode == 0, p.stderr
    assert started
    assert "UNGATED" in (root / "logs" / "resumes.log").read_text()


def test_the_gates_own_smoke_needs_no_receipt(tmp_path):
    root, sha, bindir = root_for_launch(tmp_path)
    p, started = launch(tmp_path, root, bindir, GATE_RUN="1")
    assert p.returncode == 0, p.stderr
    assert started and "gate run" in (root / "logs" / "resumes.log").read_text()


# ---------------------------------------------------------------------------------------
# the gates audit the training corpus and certify one commit
# ---------------------------------------------------------------------------------------

def gates_dry(tmp_path, **env):
    e = {**os.environ, "DRY": "1", "DOOM_ROOT": str(tmp_path), **env}
    p = subprocess.run(["bash", GATES], capture_output=True, text=True, env=e)
    assert p.returncode == 0, p.stderr
    return p.stdout.splitlines()


def test_the_training_corpus_sidecars_are_audited_in_both_spaces(tmp_path):
    lines = gates_dry(tmp_path, VAES="sd15,sd35")

    def latents(ln):
        t = ln.split()
        return t[t.index("--latents-dir") + 1]

    audits = [ln for ln in lines if "--audit-only" in ln and latents(ln).endswith("/arenas")]
    assert len(audits) == 2, audits
    assert any("latents_arnold_dense_pertic/arenas" in ln for ln in audits)
    assert any("latents_arnold_dense_pertic_sd35/arenas" in ln for ln in audits)
    for ln in audits:
        assert "--episodes 2000" in ln and f"--audit-parquet-dir {tmp_path}/raw_arnold_dense/arenas" in ln


def test_every_training_episode_is_checked_for_rows_and_tics(tmp_path):
    lines = gates_dry(tmp_path, VAES="sd15,sd35")
    inv = [ln for ln in lines if "make_dense_eval_splits.py" in ln and "--raw-tics" in ln]
    assert len(inv) == 2, inv
    for ln in inv:
        assert "--check-only" in ln and "--expect-ids 0:2000" in ln and "--sample 0" in ln
        assert f"--raw-tics {tmp_path}/raw_arnold_dense/arenas" in ln


def test_the_new_gates_run_before_the_alignment_gate(tmp_path):
    lines = gates_dry(tmp_path)
    first = {k: min(i for i, ln in enumerate(lines) if k in ln)
             for k in ("gate0 pin", "--raw-tics", "--min-accuracy", "--fit-check")}
    assert first["gate0 pin"] < first["--raw-tics"] < first["--min-accuracy"] < first["--fit-check"]


def test_the_gates_certify_a_commit_and_hand_the_launcher_a_receipt(tmp_path):
    lines = gates_dry(tmp_path)
    text = "\n".join(lines)
    assert f"{tmp_path}/GATES_COMMIT" in text
    assert "DRY gate0 pin" in text
    # the launcher is run by the gates as a gate run, before any receipt exists
    src = open(GATES).read()
    assert "GATE_RUN=1" in src


def test_a_dirty_checkout_fails_gate_zero_and_revokes_the_old_receipt(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    sha = checkout(root / "repo")
    (root / "GATES_COMMIT").write_text(f"{sha} old\n")
    (root / "repo" / "train_wm.py").write_text("# dirty\n")
    e = {**os.environ, "DOOM_ROOT": str(root), "PY": "/bin/echo", "REPO": str(root / "repo"),
         "LAUNCH": LAUNCH}
    p = subprocess.run(["bash", GATES], capture_output=True, text=True, env=e, timeout=60)
    assert p.returncode != 0 and "GATE_FAILED 0 pin" in p.stderr
    assert not (root / "GATES_COMMIT").exists(), "a failed gate run left a receipt standing"


def test_gates_refuse_when_the_launcher_would_run_a_different_checkout(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    checkout(root / "repo")
    other = tmp_path / "other"
    checkout(other)
    git(other, "commit", "-q", "--allow-empty", "-m", "two")
    e = {**os.environ, "DOOM_ROOT": str(root), "PY": "/bin/echo", "REPO": str(other), "LAUNCH": LAUNCH}
    p = subprocess.run(["bash", GATES], capture_output=True, text=True, env=e, timeout=60)
    assert p.returncode != 0 and "GATE_FAILED 0 pin" in p.stderr and "launcher runs" in p.stderr


# ---------------------------------------------------------------------------------------
# rows and tics, per episode
# ---------------------------------------------------------------------------------------

def _episode(tmp_path, ep, tics_side, tics_raw):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pertic_fixtures import write_pertic_episode
    lat = str(tmp_path / "lat")
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    write_pertic_episode(lat, ep, [0] * len(tics_side), tics=np.asarray(tics_side))
    pq.write_table(pa.table({"tic": pa.array(tics_raw, pa.int32())}), str(raw / f"ep_{ep:05d}.parquet"))
    return lat, str(raw)


def test_raw_tics_that_match_pass(tmp_path):
    lat, raw = _episode(tmp_path, 3, list(range(10)), list(range(10)))
    assert mdes.validate(lat, [3], sample=0, raw_tics=raw)["ok"]


def test_a_sidecar_shorter_than_its_recording_is_caught(tmp_path):
    """A per-tic corpus keeps every row, so a sidecar with fewer rows than the recording lost some."""
    lat, raw = _episode(tmp_path, 3, list(range(9)), list(range(10)))
    rep = mdes.validate(lat, [3], sample=0, raw_tics=raw)
    assert not rep["ok"] and any("tic" in p for p in rep["problems"])


def test_a_sidecar_off_by_one_tic_is_caught(tmp_path):
    lat, raw = _episode(tmp_path, 3, list(range(1, 11)), list(range(10)))
    rep = mdes.validate(lat, [3], sample=0, raw_tics=raw)
    assert not rep["ok"] and any("tic" in p for p in rep["problems"])


def test_a_missing_recording_is_caught(tmp_path):
    lat, raw = _episode(tmp_path, 3, list(range(10)), list(range(10)))
    os.remove(os.path.join(raw, "ep_00003.parquet"))
    rep = mdes.validate(lat, [3], sample=0, raw_tics=raw)
    assert not rep["ok"]


def test_the_cli_takes_the_raw_directory():
    a = mdes.build_parser().parse_args(["--latents-dir", "x", "--raw-tics", "/raw"])
    assert a.raw_tics == "/raw"
