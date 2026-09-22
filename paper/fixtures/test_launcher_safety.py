"""No test in this directory may be able to start real work on any machine.

On Sep 21 2026 `pytest paper/fixtures` was run on Spiderman and started two real multi-hour GPU
encodes, because `test_encode_pertic_launcher.py` executed `scripts/spiderman/encode_pertic.sh` with
no environment override. Off the server the launcher exits early for want of `/sata2`, so the test
looked harmless everywhere it had ever been run; on the server it was a launch button.

The rule this file enforces: a test may run a `scripts/*/*.sh` launcher for real only with
`DRY=1` (print the command, touch nothing) **and** a data root pointed somewhere disposable
(`DOOM_ROOT`). Parsing it with `bash -n`, or asking a Python script for `--help`, is always fine.

This is a guard over the other test files' source, so it keeps holding for launcher tests nobody has
written yet.

    python -m pytest paper/fixtures/test_launcher_safety.py -q
"""
import glob
import os
import re
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# every launcher in the repo, whichever machine it targets: `scripts/cluster/*.sh` drives a rented
# node the same way `scripts/spiderman/*.sh` drives the lab server, and a test that ran one of them
# for real would start work on whatever box pytest happens to be on.
LAUNCHERS = sorted(glob.glob(os.path.join(REPO, "scripts", "*", "*.sh")))
TEST_FILES = sorted(glob.glob(os.path.join(HERE, "test_*.py")))

# A launcher that only prints or parses cannot start work, so these need no DRY knob.
NO_SIDE_EFFECTS = {"null_prompt.sh"}


def subprocess_calls(source):
    """Every `subprocess.run(` call in a source file, as the text of its argument list."""
    out = []
    for m in re.finditer(r"subprocess\.run\(", source):
        i = m.end() - 1
        depth, j = 0, i
        while j < len(source):
            if source[j] == "(":
                depth += 1
            elif source[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append(source[i:j + 1])
    return out


def runs_a_shell_script(call):
    """Does this call execute a shell script, rather than parse it or ask for --help?"""
    if '"bash"' not in call and "'bash'" not in call:
        return False
    if '"-n"' in call or "'-n'" in call:      # bash -n only parses
        return False
    if '"-c"' in call or "'-c'" in call:      # an inline script written by the test itself
        return False
    return True


@pytest.mark.parametrize("path", TEST_FILES, ids=[os.path.basename(p) for p in TEST_FILES])
def test_no_test_executes_a_launcher_that_could_touch_real_data(path):
    """Either the launcher is stopped before its side effects (DRY=1), or it is pointed at a
    throwaway data root where there is nothing to act on. One of the two, every time."""
    with open(path) as f:
        source = f.read()
    for call in subprocess_calls(source):
        if not runs_a_shell_script(call):
            continue
        context = call + _enclosing_helper(source, call)
        assert "DRY" in context or "DOOM_ROOT" in context or "tmp_path" in context, (
            f"{os.path.basename(path)} runs a launcher for real against the default data root, so "
            f"on a machine where it exists this test starts real work:\n{call}")


def _enclosing_helper(source, call):
    """The function body the call sits in, so a `dry()` wrapper counts for its callers."""
    idx = source.index(call)
    start = source.rfind("\ndef ", 0, idx)
    end = source.find("\ndef ", idx)
    return source[start if start >= 0 else 0:end if end > 0 else len(source)]


@pytest.mark.parametrize("path", LAUNCHERS, ids=[os.path.basename(p) for p in LAUNCHERS])
def test_every_launcher_parses(path):
    assert subprocess.run(["bash", "-n", path], capture_output=True, text=True).returncode == 0


@pytest.mark.parametrize(
    "path",
    [p for p in LAUNCHERS if os.path.basename(p) in
     {"encode_pertic.sh", "encode_dense.sh", "record_dense.sh", "launch_sd35.sh",
      "setup_node.sh", "fetch_dataset.sh", "encode_all.sh", "gates.sh", "launch_runs.sh",
      "status.sh"}],
    ids=lambda p: os.path.basename(p))
def test_the_launchers_a_test_touches_have_a_dry_knob(path):
    with open(path) as f:
        text = f.read()
    assert re.search(r'\[\s*"?\$\{?DRY', text) or "DRY:-0" in text, \
        f"{os.path.basename(path)} has no DRY knob, so no test can exercise it safely"


@pytest.mark.parametrize("path", [p for p in LAUNCHERS if os.path.basename(p) in
                                  {"encode_pertic.sh", "encode_dense.sh", "record_dense.sh",
                                   "setup_node.sh", "fetch_dataset.sh", "encode_all.sh",
                                   "gates.sh", "launch_runs.sh", "status.sh"}],
                         ids=lambda p: os.path.basename(p))
def test_dry_touches_nothing_and_prints_a_command(path, tmp_path):
    """The knob has to actually work, not merely be mentioned."""
    env = {**os.environ, "DRY": "1", "DOOM_ROOT": str(tmp_path),
           "SEGMENT": "arenas", "CORPUS": "seen", "MODE": "decisions"}
    r = subprocess.run(["bash", path], capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"{os.path.basename(path)} DRY exited {r.returncode}: {r.stderr}"
    assert any(ln.startswith("DRY ") for ln in r.stdout.splitlines()), r.stdout
    assert list(tmp_path.iterdir()) == [], f"{os.path.basename(path)} wrote under the data root"


def test_the_data_root_override_is_honoured_by_those_launchers(tmp_path):
    """DOOM_ROOT must reach the printed command, or the override is decorative."""
    for name in ("encode_pertic.sh", "encode_dense.sh", "record_dense.sh"):
        path = os.path.join(REPO, "scripts", "spiderman", name)
        env = {**os.environ, "DRY": "1", "DOOM_ROOT": str(tmp_path),
               "SEGMENT": "arenas", "CORPUS": "seen", "MODE": "decisions"}
        r = subprocess.run(["bash", path], capture_output=True, text=True, env=env)
        assert str(tmp_path) in r.stdout, f"{name} ignored DOOM_ROOT:\n{r.stdout}"
        assert "/sata2/data/rnagabhi" not in r.stdout, f"{name} still points at the real root"


def test_this_guard_would_have_caught_the_original_bug():
    """The exact call that started the two encodes on Spiderman, as it was written."""
    bad = 'subprocess.run(["bash", SCRIPT], capture_output=True, text=True)'
    assert runs_a_shell_script(bad)
    assert not any(k in bad for k in ("DRY", "DOOM_ROOT", "tmp_path"))
    # and the forms that are fine
    for good in ('subprocess.run(["bash", "-n", SCRIPT])',
                 'subprocess.run(["bash", "-c", script], capture_output=True)',
                 'subprocess.run([sys.executable, script, "--help"])'):
        assert not runs_a_shell_script(good), good
