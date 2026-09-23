"""Astra's re-verification of the third round (2026-09-23): three narrower defects.

  8. `gates.sh revoke_mine` answered a failed revocation of one backbone with `rm -f $CERT`, deleting
     every other backbone's entry: a failed sd35 revocation took a standing U-Net certificate with it.
     A revocation now retries under the gates' own interpreter fallbacks and, if it still fails,
     stops the gates before any gate runs and leaves the file untouched.

Everything here is a dry run or a stub run: nothing encodes, trains or touches a GPU.

    python -m pytest paper/fixtures/test_nexttic_defects4.py -q
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from test_launch_pin import certify, root_for_launch  # noqa: E402
from test_nexttic_defects3 import GATE_KNOBS  # noqa: E402

GATES = os.path.join(REPO, "scripts", "cluster", "gates.sh")
LAUNCH = os.path.join(REPO, "scripts", "spiderman", "launch_nexttic.sh")


def _clean_env(**env):
    e = {k: v for k, v in os.environ.items() if k not in GATE_KNOBS}
    e.update(env)
    return e


# ---------------------------------------------------------------------------------------
# 8. a failed revocation never deletes another backbone's entry
# ---------------------------------------------------------------------------------------

def _both_certified(tmp_path):
    root, sha, bindir = root_for_launch(tmp_path)
    certify(tmp_path, root, bindir, backbones=("unet", "sd35"))
    return root


def _sd35_gates(root, **env):
    """A real (not DRY) staggered SD 3.5 gate run; the stand-in interpreters fail every gate command."""
    e = _clean_env(DOOM_ROOT=str(root), REPO=str(root / "repo"), LAUNCH=LAUNCH, VAES="sd35", SMOKE_BBS="sd35",
                   GATES_RUN_ID="t8", **env)
    return subprocess.run(["bash", GATES], capture_output=True, text=True, env=e, timeout=60)


def test_a_revocation_that_fails_under_the_backbones_interpreter_retries_and_keeps_the_others(tmp_path):
    """Astra's reproduction: PY_SD35 cannot run gate_certificate.py, the revocation of sd35 failed,
    and the whole certificate went, U-Net entry included."""
    root = _both_certified(tmp_path)
    p = _sd35_gates(root, PY_SD35="/usr/bin/false", PY_UNET=sys.executable)
    assert p.returncode != 0 and "GATE_FAILED 1 sidecar audit" in p.stderr, p.stderr
    cert = root / "GATES_CERT.json"
    assert cert.is_file(), "a failed sd35 revocation deleted the certificate"
    assert set(json.loads(cert.read_text())["backbones"]) == {"unet"}, \
        "sd35 must be revoked (by a fallback interpreter) and the U-Net entry kept"


def test_when_no_interpreter_can_revoke_the_gates_stop_before_any_gate_and_touch_nothing(tmp_path):
    root = _both_certified(tmp_path)
    cert = root / "GATES_CERT.json"
    before = cert.read_bytes()
    p = _sd35_gates(root, PY_SD35="/usr/bin/false", PY_UNET="/usr/bin/false")
    assert p.returncode != 0 and "GATE_FAILED 0 revoke" in p.stderr, p.stderr
    assert "sd35" in p.stderr and "untouched" in p.stderr
    assert cert.read_bytes() == before, "the certificate changed although nothing could be revoked"
    assert "gate 0: pin" not in p.stdout, "a gate ran after the revocation failed"
    assert not (root / "logs" / "gates_results_t8.jsonl").exists()


def test_the_header_no_longer_promises_to_revoke_everything():
    src = open(GATES).read()
    assert "revoke everything" not in src and 'rm -f "$CERT"; echo "could not revoke' not in src
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code if 'rm -f "$CERT"' in ln], "the gates still delete the whole certificate somewhere"
