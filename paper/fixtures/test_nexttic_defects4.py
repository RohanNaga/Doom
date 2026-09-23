"""Astra's re-verification of the third round (2026-09-23): three narrower defects.

  8. `gates.sh revoke_mine` answered a failed revocation of one backbone with `rm -f $CERT`, deleting
     every other backbone's entry: a failed sd35 revocation took a standing U-Net certificate with it.
     A revocation now retries under the gates' own interpreter fallbacks and, if it still fails,
     stops the gates before any gate runs and leaves the file untouched.
  9. The readback ran eval_tf.py with every card visible, so it took GPU 0 (another user's on
     Spiderman) while the smoke used the backbone's card. Every GPU step of the gates now runs under
     CUDA_VISIBLE_DEVICES=<its card> and addresses it as cuda:0.

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


# ---------------------------------------------------------------------------------------
# 9. every GPU step runs on its own card: CUDA_VISIBLE_DEVICES=<card>, addressed as cuda:0
# ---------------------------------------------------------------------------------------

# records the device view and the arguments of whatever gate command it stands in for
RECORDING_PY = '''#!/usr/bin/env python3
import json, os, sys
with open(os.environ["RECORD"], "a") as f:
    f.write(json.dumps({"cvd": os.environ.get("CUDA_VISIBLE_DEVICES"), "argv": sys.argv[1:]}) + "\\n")
'''


def _gates_dry(tmp_path, **env):
    e = _clean_env(DRY="1", DOOM_ROOT=str(tmp_path), UNET_GPU="1", SD35_GPU="2", **env)
    e.pop("CUDA_VISIBLE_DEVICES", None)
    p = subprocess.run(["bash", GATES], capture_output=True, text=True, env=e, timeout=60)
    assert p.returncode == 0, p.stderr
    return p.stdout.splitlines()


def _gpu_steps(lines):
    """(label, card the step belongs to, command) for every GPU step the gates run directly."""
    out = []
    for ln in lines:
        t = ln.split()
        if ln.startswith("DRY gate1d latent alignment "):
            out.append((" ".join(t[:6]), "2" if t[4] == "sd35" else "1", " ".join(t[6:])))
        elif ln.startswith("DRY gate4b probes "):
            out.append((" ".join(t[:4]), "2" if t[3] == "sd35" else "1", " ".join(t[4:])))
        elif ln.startswith("DRY gate5 readback "):
            out.append((" ".join(t[:6]), "2" if t[3] == "sd35" else "1", " ".join(t[6:])))
    return out


def test_every_gpu_step_runs_under_its_cards_device_view(tmp_path):
    steps = _gpu_steps(_gates_dry(tmp_path))
    assert len(steps) == 4 + 2 + 8, [s[0] for s in steps]
    for label, card, cmd in steps:
        assert cmd.startswith(f"env CUDA_VISIBLE_DEVICES={card} "), (label, cmd[:80])
        # inside that view the card is cuda:0; naming its physical index would miss it
        assert "--device cuda:1" not in cmd and "--device cuda:2" not in cmd, (label, cmd)
        if "--device" in cmd:
            assert "--device cuda:0" in cmd, (label, cmd)


def test_the_readback_runs_on_the_backbones_card(tmp_path):
    """Astra's finding: eval_tf.py picks `cuda`, i.e. the first visible card, and the readback ran with
    every card visible, so it landed on GPU 0 (another user's on Spiderman). The printed command is run
    here exactly as gates.sh runs it, with word splitting, under a recording stand-in interpreter."""
    rec = tmp_path / "rec.py"
    rec.write_text(RECORDING_PY)
    rec.chmod(0o755)
    record = tmp_path / "record.jsonl"
    steps = _gpu_steps(_gates_dry(tmp_path, PY_UNET=str(rec), PY_SD35=str(rec)))
    for label, card, cmd in steps:
        if record.exists():
            record.unlink()
        e = _clean_env(GATE_CMD=cmd, RECORD=str(record))
        e.pop("CUDA_VISIBLE_DEVICES", None)
        p = subprocess.run(["bash", "-c", "$GATE_CMD"], capture_output=True, text=True, env=e, timeout=30)
        assert p.returncode == 0, (label, p.stderr)
        got = json.loads(record.read_text().splitlines()[-1])
        assert got["cvd"] == card, (label, got["cvd"])
        assert os.path.basename(got["argv"][0]) in ("eval_tf.py", "smoke_probe.py", "check_latent_alignment.py")


def test_the_header_states_the_device_convention():
    head = open(GATES).read().split("\nset -u", 1)[0]
    assert "CUDA_VISIBLE_DEVICES" in head and "cuda:0" in head
