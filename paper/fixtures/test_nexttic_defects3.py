"""Astra's third review of the launch gates and the evaluator (2026-09-23), and the launch knobs.

  1. `after_nexttic.sh` took an ACTIVE evaluator for an interrupted one: two simultaneous sealed
     invocations both passed the seal, the second as "re-entered", and test was scored twice. One
     evaluator per run now holds an exclusive process lock for as long as it scores; the persistent
     seal stays separate.
  2. The gates compared latent contracts within each directory only: a train corpus at scale 1 and a
     val corpus at scale 2 each passed. Train and val of one space must now share one contract, and
     it must be the space's own (the backbone's channels, its autoencoder and normalisation).
  3. The smoke dropped the gates' micro-batch and worker count and replaced the operator's EXTRA, so
     it trained at lr 5e-5 and 12 workers while the certificate pinned lr 0.1 and 4 workers. The fit,
     smoke, resume and certificate now resolve from one set of production arguments.

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


# ---------------------------------------------------------------------------------------
# 2. one latent contract per space: train equals val, and both equal the backbone's
# ---------------------------------------------------------------------------------------

GATES = os.path.join(REPO, "scripts", "cluster", "gates.sh")


def space_corpus(tmp_path, name, eps, scale=0.18215, shift=None, vae_id="stabilityai/sd-vae-ft-mse",
                 subfolder="", channels=4):
    """A one-shard corpus the stub autoencoder really encoded at `scale` and `shift`, whose shard log
    records exactly that contract: on its own it passes every per-shard check."""
    import json

    import numpy as np
    import torch
    from encode_parquet import encode_batch
    from test_latent_alignment import BATCH, T, StubVAE, frame, write_recording
    raw, lat = tmp_path / "raw", tmp_path / name
    raw.mkdir(parents=True, exist_ok=True)
    lat.mkdir(parents=True)
    lines = []
    for ep in eps:
        write_recording(str(raw / f"ep_{ep:05d}.parquet"), ep)
        frames = np.stack([frame(ep * 1000 + t) for t in range(T)])
        z = np.concatenate([encode_batch(StubVAE(), frames[a:a + BATCH], "cpu", torch.float32, False, scale, shift)
                            for a in range(0, T, BATCH)])
        np.save(str(lat / f"ep_{ep:05d}_latents.npy"), z)
        np.savez(str(lat / f"ep_{ep:05d}_meta.npz"), tic=np.arange(T, dtype=np.int64),
                 episode_id=np.full(T, ep, dtype=np.int64))
        lines.append(json.dumps({"episode": f"ep_{ep:05d}", "frames": T}))
    (lat / "episodes_00.jsonl").write_text("\n".join(lines) + "\n")
    (lat / "encode_meta_00.json").write_text(json.dumps({
        "vae_id": vae_id, "vae_subfolder": subfolder, "scaling_factor_applied": scale,
        "shift_factor_applied": shift, "every_tic": True, "legacy": False,
        "latent_contract": {"latent_channels": channels, "scaling_factor": scale, "shift_factor": shift},
        "args": {"batch_size": BATCH, "dtype": "fp32"}}))
    return str(lat), str(raw)


def _check(lat, raw, **kw):
    import check_latent_alignment as cla
    from test_latent_alignment import StubVAE
    return cla.check(lat, raw, device="cpu", vae=StubVAE(), **kw)


def test_train_and_val_under_different_scales_pass_alone_and_fail_as_one_space(tmp_path):
    """Astra's reproduction: train at scale 1 and val at scale 2 each passed their own directory."""
    train, raw = space_corpus(tmp_path, "train", (0, 1), scale=1.0, vae_id="stub")
    val, _ = space_corpus(tmp_path, "val", (6000, 6001), scale=2.0, vae_id="stub")
    assert _check(train, raw)["ok"] and _check(val, raw)["ok"], "the attack corpora must pass on their own"
    for mine, peer in ((train, val), (val, train)):
        rep = _check(mine, raw, contract_peer=peer)
        assert not rep["ok"]
        assert any("different scale" in p for p in rep["contract_problems"]), rep["contract_problems"]


def test_a_space_whose_contract_is_not_its_backbones_fails(tmp_path):
    """Equal train and val are not enough: both must be the space's own contract."""
    train, raw = space_corpus(tmp_path, "train", (0, 1), scale=1.0)
    val, _ = space_corpus(tmp_path, "val", (6000, 6001), scale=1.0)
    rep = _check(train, raw, space="sd15", contract_peer=val)
    assert not rep["ok"] and rep["contract_problems"] == [p for p in rep["contract_problems"] if "sd15" in p]
    assert any("scale" in p for p in rep["contract_problems"]), rep["contract_problems"]
    # a 4-channel sd-vae-ft-mse corpus filed as the SD 3.5 space fails on every key the space fixes
    good, raw = space_corpus(tmp_path / "b", "train", (0, 1))
    probs = _check(good, raw, space="sd35")["contract_problems"]
    for k in ("vae_id", "vae_subfolder", "scale", "shift", "channels"):
        assert any(f"{k} is" in p for p in probs), (k, probs)


def test_the_expected_contract_passes_on_train_and_val(tmp_path):
    train, raw = space_corpus(tmp_path, "train", (0, 1))
    val, _ = space_corpus(tmp_path, "val", (6000, 6001))
    for mine, peer in ((train, val), (val, train)):
        rep = _check(mine, raw, space="sd15", contract_peer=peer)
        assert rep["ok"], rep["contract_problems"]


def test_unrecorded_channels_are_measured_from_the_arrays(tmp_path):
    import json
    train, raw = space_corpus(tmp_path, "train", (0, 1))
    meta = os.path.join(train, "encode_meta_00.json")
    m = json.load(open(meta))
    m.pop("latent_contract")
    json.dump(m, open(meta, "w"))
    assert _check(train, raw, space="sd15")["ok"]
    assert any("channels is [4]" in p for p in _check(train, raw, space="sd35")["contract_problems"])


def test_a_peer_without_shard_logs_fails(tmp_path):
    train, raw = space_corpus(tmp_path, "train", (0, 1))
    empty = tmp_path / "empty"
    empty.mkdir()
    rep = _check(train, raw, contract_peer=str(empty))
    assert not rep["ok"] and any("contract is unknown" in p for p in rep["contract_problems"])


def test_the_expected_contracts_are_the_encoders_and_the_evaluators():
    """The numbers come from the modules that own them, and agree with what the encoder launcher
    writes and what the evaluator and the gates decode with."""
    import check_latent_alignment as cla
    import doomdit_utils
    import verify_sd35
    from backbones import BACKBONE_LATENT_CHANNELS, SD35_DEFAULT
    c = cla.space_contracts()
    assert c["sd15"] == {"vae_id": doomdit_utils.VAE_NAME, "vae_subfolder": "", "scale": doomdit_utils.LATENT_SCALE,
                         "shift": None, "channels": BACKBONE_LATENT_CHANNELS["unet"]}
    assert c["sd15"]["vae_id"] == "stabilityai/sd-vae-ft-mse" and c["sd15"]["scale"] == 0.18215
    vae = verify_sd35.SD35_VAE
    assert c["sd35"] == {"vae_id": SD35_DEFAULT, "vae_subfolder": "vae", "scale": vae["scaling_factor"],
                         "shift": vae["shift_factor"], "channels": BACKBONE_LATENT_CHANNELS["sd35"]}
    assert (c["sd35"]["scale"], c["sd35"]["shift"], c["sd35"]["channels"]) == (1.5305, 0.0609, 16)
    enc = open(os.path.join(REPO, "scripts", "spiderman", "encode_nexttic.sh")).read()
    assert (f"--vae-id {SD35_DEFAULT} --vae-subfolder vae --latent-channels 16" in enc
            and "--scaling-factor 1.5305 --shift-factor 0.0609" in enc)
    gates = open(GATES).read()
    assert "--latent-scale 1.5305 --latent-shift 0.0609" in gates


def test_the_gates_check_each_corpus_against_its_space_and_its_peer(tmp_path):
    e = {**os.environ, "DRY": "1", "DOOM_ROOT": str(tmp_path), "VAES": "sd15,sd35"}
    p = subprocess.run(["bash", GATES], capture_output=True, text=True, env=e)
    assert p.returncode == 0, p.stderr
    lines = [ln for ln in p.stdout.splitlines() if "check_latent_alignment.py" in ln]
    assert len(lines) == 4, lines

    def arg(ln, name):
        t = ln.split()
        return t[t.index(name) + 1]

    pairs = {}
    for ln in lines:
        space = arg(ln, "--space")
        assert space in ("sd15", "sd35")
        assert ("_sd35/" in arg(ln, "--latents-dir")) == (space == "sd35"), ln
        pairs[arg(ln, "--latents-dir")] = (space, arg(ln, "--contract-peer"))
    for d, (space, peer) in pairs.items():
        assert peer != d and pairs[peer] == (space, d), (d, peer)


# ---------------------------------------------------------------------------------------
# 3. the fit, the smoke, the resume and the certificate resolve from one set of production arguments
# ---------------------------------------------------------------------------------------

SMOKE_FLAGS = {"--steps", "--results-dir", "--val-every", "--val-windows", "--ckpt-every", "--snapshot-every",
               "--local-snapshots", "--keep-last"}
# what an operator may have exported that gates.sh reads; each test sets its own
GATE_KNOBS = ("WORKERS", "MB", "MB_UNET", "MB_SD35", "STEPS", "LAUNCH_STEPS", "EXTRA", "PY", "PY_UNET", "PY_SD35",
              "RUN_REPO", "REPO", "ALLOW_ACCUM", "GATES_RUN_ID")


def _gates_dry(tmp_path, **env):
    e = {k: v for k, v in os.environ.items() if k not in GATE_KNOBS}
    e.update({"DRY": "1", "DOOM_ROOT": str(tmp_path), "SMOKE_BBS": "unet sd35", **env})
    p = subprocess.run(["bash", GATES], capture_output=True, text=True, env=e, timeout=60)
    assert p.returncode == 0, p.stderr
    return p.stdout.splitlines()


def _launch_line(lines, marker):
    """The train_wm.py line the launcher printed right after the line starting with `marker`."""
    i = next(i for i, ln in enumerate(lines) if ln.startswith(marker))
    return next(ln for ln in lines[i + 1:] if "train_wm.py" in ln)


def _args(line):
    return line.split("train_wm.py", 1)[1].split(">>")[0]


def _interp(line):
    return line.split("train_wm.py", 1)[0].split()[-1]


def _cert_line(lines, bb):
    return next(ln for ln in lines if ln.startswith(f"DRY certificate command {bb} "))


def _gates_workers():
    n = int(subprocess.run(["getconf", "_NPROCESSORS_ONLN"], capture_output=True, text=True).stdout.strip() or 16)
    return min(16, max(4, n // 4))


def _flag_names(pairs):
    return {p.split()[0] for p in pairs}


def test_the_smoke_carries_the_operators_extra_workers_and_micro_batch(tmp_path):
    """Astra's reproduction: the certificate resolved `--lr 0.1` and the gates' own worker count and
    micro-batch while the smoke ran the launcher's defaults, lr 5e-5 and 12 workers."""
    w = _gates_workers()
    lines = _gates_dry(tmp_path, EXTRA="--lr 0.1", MB_UNET="16", ALLOW_ACCUM="1")
    for marker in ("DRY gate3 fit unet", "DRY gate4 smoke unet", "DRY gate4c resume unet"):
        cmd = _args(_launch_line(lines, marker)) + " "
        assert "--lr 0.1 " in cmd, (marker, cmd)
        assert f"--num-workers {w} " in cmd and "--per-gpu-batch 16 " in cmd, (marker, cmd)
    # the smoke's own overrides come AFTER the operator's, so argparse gives the smoke its settings
    smoke = _args(_launch_line(lines, "DRY gate4 smoke unet"))
    assert smoke.index("--lr 0.1") < smoke.index("--val-windows 128")


def test_smoke_resume_fit_and_certificate_differ_only_in_smoke_settings(tmp_path):
    import gate_certificate as gc
    lines = _gates_dry(tmp_path, EXTRA="--lr 0.1 --clip 0.5", WORKERS="7")
    for bb in ("unet", "sd35"):
        cert_line = _cert_line(lines, bb)
        cert = gc.flag_pairs(_args(cert_line))
        assert "--lr 0.1" in cert and "--num-workers 7" in cert and "--steps 400000" in cert, cert
        for marker, allowed in ((f"DRY gate4 smoke {bb}", SMOKE_FLAGS),
                                (f"DRY gate4c resume {bb}", SMOKE_FLAGS | {"--resume"}),
                                (f"DRY gate3 fit {bb}", {"--results-dir", "--fit-check"})):
            line = _launch_line(lines, marker)
            got = gc.flag_pairs(_args(line))
            only_here, only_cert = set(got) - set(cert), set(cert) - set(got)
            assert _flag_names(only_here) <= allowed and _flag_names(only_cert) <= allowed, \
                (marker, only_here, only_cert)
            assert _interp(line) == _interp(cert_line), marker
