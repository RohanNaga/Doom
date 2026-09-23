"""What launches is what the gates certified: the training corpus is audited and the commit is pinned.

The 2026-09-22 review (H3) found three gaps between the gates and the launch:

  * gate 1 audited sidecars against the raw recordings for val only; the 2,000 training episodes,
    written by two different encoder versions, were never audited;
  * nothing checked that a stored latent array has as many rows as its sidecar, and at the same tics
    as the recording, over the training corpus;
  * `launch_nexttic.sh` ran `git pull` AFTER the gates, and a failed pull did not stop the launch,
    so the code that trained could differ from the code that passed.

A first fix pinned only the commit (`$D/GATES_COMMIT`); Astra's review of it showed that a changed
recipe and an SD 3.5-only gate run still certified a U-Net launch. The launcher now refuses to start
unless `$D/GATES_CERT.json` has an entry for THIS backbone whose resolved command, clean commit,
corpus fingerprints, encoder records and gate results all match (`gate_certificate.py`). These tests
run it for real against a throwaway root with a throwaway git checkout, a small per-tic corpus and a
fake `tmux` that only records what it was asked to do.

    python -m pytest paper/fixtures/test_launch_pin.py -q
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import gate_certificate as gc  # noqa: E402
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


TRAIN_IDS, VAL_IDS = "0:2", "6000:6002"


def corpus_dir(path, ids, space="sd15"):
    """Two per-tic episodes and the encoder's own record, in the layout the launcher trains on."""
    from pertic_fixtures import held_actions, write_pertic_episode
    for ep in ids:
        write_pertic_episode(str(path), ep, held_actions([0] * 4))
    (path / "encode_meta_00.json").write_text(json.dumps({
        "git": "abc123", "vae_id": "stabilityai/sd-vae-ft-mse", "vae_subfolder": "",
        "scaling_factor_applied": 0.18215, "shift_factor_applied": None,
        "args": {"batch_size": 64, "dtype": "bf16"}}))


def root_for_launch(tmp_path):
    root = tmp_path / "root"
    sha = checkout(root / "repo")
    for sub, ids in (("latents_arnold_dense_pertic/arenas", (0, 1)),
                     ("latents_arnold_dense_pertic_eval/val", (6000, 6001)),
                     ("latents_arnold_dense_pertic_sd35/arenas", (0, 1)),
                     ("latents_arnold_dense_pertic_eval_sd35/val", (6000, 6001))):
        corpus_dir(root / sub, ids)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "tmux").write_text(FAKE_TMUX)
    (bindir / "tmux").chmod(0o755)
    return root, sha, bindir


def launcher_env(root, bindir, log, **env):
    return {**os.environ, "DOOM_ROOT": str(root), "PY": sys.executable, "PY_SD35": sys.executable,
            "TMUX_LOG": str(log), "PATH": f"{bindir}:{os.environ['PATH']}",
            "TRAIN_IDS": TRAIN_IDS, "VAL_IDS": VAL_IDS, **env}


def launch(tmp_path, root, bindir, backbone="unet", **env):
    log = tmp_path / "tmux.log"
    if log.exists():
        log.unlink()
    e = launcher_env(root, bindir, log, **env)
    p = subprocess.run(["bash", LAUNCH, "0", backbone], capture_output=True, text=True, env=e, timeout=120)
    calls = log.read_text().splitlines() if log.exists() else []
    return p, [c for c in calls if c.startswith("tmux new-session")]


def cert_command(tmp_path, root, bindir, backbone="unet", **env):
    """What the gates record: the launcher's own CERT_QUERY output."""
    e = launcher_env(root, bindir, tmp_path / "q.log", CERT_QUERY="1", **env)
    p = subprocess.run(["bash", LAUNCH, "0", backbone], capture_output=True, text=True, env=e, timeout=60)
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def certify(tmp_path, root, bindir, backbones=("unet",), drop_gate=None, failed_gate=None, **env):
    """Write GATES_CERT.json as the gates would, with every required gate passed unless told otherwise."""
    results = tmp_path / "gates_results.jsonl"
    rows = []
    for g in gc.REQUIRED_GATES:
        if g == drop_gate:
            continue
        scope = "all" if g in ("0 pin", "2 alignment") else None
        for bb in backbones:
            sc = scope or (f"space:{'sd35' if bb == 'sd35' else 'sd15'}" if g.startswith("1") else f"bb:{bb}")
            rows.append({"gate": g, "scope": sc, "status": "failed" if g == failed_gate else "ok", "detail": ""})
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for bb in backbones:
        space = "sd35" if bb == "sd35" else "sd15"
        suf = "_sd35" if bb == "sd35" else ""
        now = gc.identity(bb, cert_command(tmp_path, root, bindir, bb, **env), env.get("INIT", ""),
                          str(root / "repo"), str(root / f"latents_arnold_dense_pertic{suf}/arenas"), TRAIN_IDS,
                          str(root / f"latents_arnold_dense_pertic_eval{suf}/val"), VAL_IDS)
        gc.write(str(root / "GATES_CERT.json"), bb, space, "0", now, str(results))


def code_lines(path):
    with open(path) as f:
        return [ln for ln in f.read().splitlines() if not ln.lstrip().startswith("#")]


# ---------------------------------------------------------------------------------------
# nothing changes the checkout after the gates, and the certificate pins the whole launch
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("script", [LAUNCH, AFTER], ids=os.path.basename)
def test_no_launcher_pulls_code(script):
    assert not [ln for ln in code_lines(script) if "git pull" in ln], \
        f"{os.path.basename(script)} changes the checkout it runs"


def test_without_a_certificate_nothing_launches(tmp_path):
    root, sha, bindir = root_for_launch(tmp_path)
    p, started = launch(tmp_path, root, bindir)
    assert p.returncode != 0 and "no gate certificate" in p.stderr
    assert started == []


def test_the_certified_launch_starts_and_is_logged(tmp_path):
    root, sha, bindir = root_for_launch(tmp_path)
    certify(tmp_path, root, bindir)
    p, started = launch(tmp_path, root, bindir)
    assert p.returncode == 0, p.stderr
    assert len(started) == 1 and "train-unet-nexttic" in started[0]
    log = (root / "logs" / "resumes.log").read_text()
    assert f"commit {sha}" in log and "certified" in log


def test_an_sd35_only_certificate_does_not_certify_a_unet_launch(tmp_path):
    root, sha, bindir = root_for_launch(tmp_path)
    certify(tmp_path, root, bindir, backbones=("sd35",))
    p, started = launch(tmp_path, root, bindir, backbone="unet")
    assert p.returncode != 0 and "certifies no unet launch" in p.stderr
    assert started == []
    p, started = launch(tmp_path, root, bindir, backbone="sd35")
    assert p.returncode == 0, p.stderr


@pytest.mark.parametrize("change", [{"STEPS": "50000"}, {"MB": "16", "ALLOW_ACCUM": "1"}, {"WORKERS": "3"},
                                    {"ACTION_HISTORY": "0"}, {"PHASE": "1"}, {"EXTRA": "--lr 1e-4"}])
def test_a_changed_recipe_is_refused(tmp_path, change):
    """The commit-only receipt accepted every one of these."""
    root, sha, bindir = root_for_launch(tmp_path)
    certify(tmp_path, root, bindir)
    p, started = launch(tmp_path, root, bindir, **change)
    assert p.returncode != 0 and "launch command differs" in p.stderr, p.stderr
    assert started == []


def test_a_changed_training_corpus_is_refused(tmp_path):
    root, sha, bindir = root_for_launch(tmp_path)
    certify(tmp_path, root, bindir)
    meta = root / "latents_arnold_dense_pertic/arenas/ep_00001_meta.npz"
    m = dict(np.load(meta))
    m["buttons"] = np.array(["010000000"] * len(m["buttons"]))
    np.savez(str(meta), **m)
    p, started = launch(tmp_path, root, bindir)
    assert p.returncode != 0 and "train corpus fingerprint" in p.stderr
    assert started == []


def test_a_changed_encoder_record_is_refused(tmp_path):
    root, sha, bindir = root_for_launch(tmp_path)
    certify(tmp_path, root, bindir)
    f = root / "latents_arnold_dense_pertic_eval/val/encode_meta_00.json"
    m = json.loads(f.read_text())
    m["git"] = "def456"
    f.write_text(json.dumps(m))
    p, started = launch(tmp_path, root, bindir)
    assert p.returncode != 0 and "val corpus encoder records" in p.stderr
    assert started == []


def test_a_new_commit_or_a_tracked_change_is_refused(tmp_path):
    root, sha, bindir = root_for_launch(tmp_path)
    certify(tmp_path, root, bindir)
    (root / "repo" / "train_wm.py").write_text("# edited after the gates\n")
    p, started = launch(tmp_path, root, bindir)
    assert p.returncode != 0 and "tracked files differ" in p.stderr
    git(root / "repo", "commit", "-q", "-am", "two")
    p, started = launch(tmp_path, root, bindir)
    assert p.returncode != 0 and "the gates certified" in p.stderr
    assert started == []


@pytest.mark.parametrize("how", ["dropped", "failed"])
def test_a_missing_or_failed_gate_cannot_be_certified_or_launched(tmp_path, how):
    root, sha, bindir = root_for_launch(tmp_path)
    kw = {"drop_gate": "5 readback"} if how == "dropped" else {"failed_gate": "5 readback"}
    with pytest.raises(SystemExit, match="5 readback"):
        certify(tmp_path, root, bindir, **kw)
    # and a certificate edited to carry one is refused at launch
    certify(tmp_path, root, bindir)
    cert = json.loads((root / "GATES_CERT.json").read_text())
    gates = cert["backbones"]["unet"]["gates"]
    cert["backbones"]["unet"]["gates"] = ([g for g in gates if g["gate"] != "5 readback"] if how == "dropped"
                                          else [{**g, "status": "failed"} if g["gate"] == "5 readback" else g
                                                for g in gates])
    (root / "GATES_CERT.json").write_text(json.dumps(cert))
    p, started = launch(tmp_path, root, bindir)
    assert p.returncode != 0 and "5 readback" in p.stderr
    assert started == []


def test_a_resume_needs_the_same_certificate_and_may_drop_init_from(tmp_path):
    root, sha, bindir = root_for_launch(tmp_path)
    certify(tmp_path, root, bindir, INIT="/x/best.pt")
    r = root / "results_spiderman" / "040-unet-nexttic"
    r.mkdir(parents=True)
    (r / "log.jsonl").write_text('{"event": "start"}\n')
    (r / "0005000.pt").write_bytes(b"ck")
    p, started = launch(tmp_path, root, bindir)
    assert p.returncode == 0, p.stderr
    assert "--resume" in started[0] and "--init-from" not in started[0]
    p, started = launch(tmp_path, root, bindir, STEPS="7")
    assert p.returncode != 0 and "launch command differs" in p.stderr


def test_an_ungated_launch_must_be_asked_for_and_is_recorded(tmp_path):
    root, sha, bindir = root_for_launch(tmp_path)
    p, started = launch(tmp_path, root, bindir, ALLOW_UNGATED="1")
    assert p.returncode == 0, p.stderr
    assert started
    assert "UNGATED" in (root / "logs" / "resumes.log").read_text()


def test_the_gates_own_runs_need_no_certificate(tmp_path):
    root, sha, bindir = root_for_launch(tmp_path)
    p, started = launch(tmp_path, root, bindir, GATE_RUN="1")
    assert p.returncode == 0, p.stderr
    assert started and "gate run" in (root / "logs" / "resumes.log").read_text()


def test_cert_query_prints_one_normalised_command_and_touches_nothing(tmp_path):
    root, sha, bindir = root_for_launch(tmp_path)
    before = sorted(p.name for p in root.iterdir())
    cmd = cert_command(tmp_path, root, bindir)
    assert "\n" not in cmd and "  " not in cmd and "train_wm.py --backbone unet" in cmd
    assert "--resume" not in cmd and "CUDA_VISIBLE_DEVICES" not in cmd
    assert sorted(p.name for p in root.iterdir()) == before


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


def test_the_gates_write_a_per_backbone_certificate_of_the_production_launch(tmp_path):
    lines = gates_dry(tmp_path, SMOKE_BBS="unet sd35")
    text = "\n".join(lines)
    assert "DRY gate0 pin" in text and f"revoke {tmp_path}/GATES_CERT.json" in text
    certs = [ln for ln in lines if ln.startswith("DRY certificate ") and "gate_certificate.py write" in ln]
    assert len(certs) == 2, certs
    for bb, space in (("unet", "sd15"), ("sd35", "sd35")):
        ln = [c for c in certs if f"--backbone {bb} " in c][0]
        assert f"--space {space}" in ln and f"--cert {tmp_path}/GATES_CERT.json" in ln
        assert "STEPS=400000" in ln and "MB=32" in ln, "the certificate must pin the PRODUCTION command"
    src = open(GATES).read()
    assert "GATE_RUN=1" in src and "CERT_QUERY=1" in src
    assert "GATES_COMMIT" not in src


def test_every_gate_records_a_result_for_the_certificate():
    """Every gate the certificate requires is recorded somewhere in the gate script."""
    src = open(GATES).read()
    recorded = set(re.findall(r'record "([^"$]+)', src)) | {
        f"{m} {c}" for m in re.findall(r'record "([^"$]+) \$C"', src) for c in ("train", "val")}
    for g in gc.REQUIRED_GATES:
        assert g in recorded, f"gate {g!r} never records its result, so no certificate can be written"


def test_a_dirty_checkout_fails_gate_zero_and_revokes_the_old_certificate(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    checkout(root / "repo")
    (root / "GATES_CERT.json").write_text('{"backbones": {"unet": {}}}')
    (root / "repo" / "train_wm.py").write_text("# dirty\n")
    e = {**os.environ, "DOOM_ROOT": str(root), "PY": "/bin/echo", "REPO": str(root / "repo"),
         "LAUNCH": LAUNCH}
    p = subprocess.run(["bash", GATES], capture_output=True, text=True, env=e, timeout=60)
    assert p.returncode != 0 and "GATE_FAILED 0 pin" in p.stderr
    assert not (root / "GATES_CERT.json").exists(), "a failed gate run left a certificate standing"


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
