"""The next two Spiderman launches: a DiT-XL/2 row and a U-Net conditioned on the requested action id.

`launch_nexttic.sh` knew `unet | sd35 | pixart`, one fixed run name per backbone, and the gates knew
two backbones. The two rows these tests pin, both on the 4-channel corpus with the 040 recipe:

  043-dit-nexttic             DiT-XL/2 from the local ImageNet checkpoint the stride-4 DiT rows
                              started from ($D/weights/DiT-XL-2-256x256.pt).
  044-unet-nexttic-reqaction  the 040 U-Net with ACTION_HISTORY=0: one token for Arnold's requested
                              action id instead of 32 executed-control tokens.

Everything here is a dry run, a stub run against a throwaway root, or a tiny model on the CPU:
nothing trains or touches a GPU.

    python -m pytest paper/fixtures/test_launch_dit_reqaction.py -q
"""
import json
import os
import re
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import gate_certificate as gc  # noqa: E402
from test_nexttic_defects3 import GATE_KNOBS, S, U, _launch_dry  # noqa: E402

SCRIPTS = os.path.join(REPO, "scripts", "spiderman")


# ---------------------------------------------------------------------------------------
# 1. the dit backbone in launch_nexttic.sh
# ---------------------------------------------------------------------------------------

def test_dit_starts_from_the_local_checkpoint_the_stride4_dit_rows_used(tmp_path):
    out = _launch_dry(tmp_path, "dit")
    assert f"--backbone dit --latent-channels 4 --warm-start {tmp_path}/weights/DiT-XL-2-256x256.pt " in out
    for older in ("launch_aligned_spiderman.sh", "launch_seed1.sh", "side_queue2.sh"):
        assert "--warm-start $D/weights/DiT-XL-2-256x256.pt" in open(os.path.join(SCRIPTS, older)).read(), older


def test_dit_trains_on_the_four_channel_corpus_under_its_own_run(tmp_path):
    out = _launch_dry(tmp_path, "dit")
    assert f"--latents-dir {tmp_path}/latents_arnold_dense_pertic/arenas " in out
    assert f"--val-latents-dir {tmp_path}/latents_arnold_dense_pertic_eval/val " in out
    assert f"--results-dir {tmp_path}/results_spiderman/043-dit-nexttic " in out
    assert f">> {tmp_path}/logs/train_043-dit-nexttic.log" in out
    assert "_sd35" not in out


def test_dit_carries_the_040_recipe_flag_for_flag(tmp_path):
    """Only what names the backbone and the run differs from the certified U-Net command: the stride-4
    DiT rows passed no DiT-only flag beyond the warm start, so there is nothing else to carry."""
    unet = gc.flag_pairs(_launch_dry(tmp_path, "unet", CERT_QUERY="1", PY_UNET=U))
    dit = gc.flag_pairs(_launch_dry(tmp_path, "dit", CERT_QUERY="1", PY_UNET=U))
    assert {p.split()[0] for p in set(unet) ^ set(dit)} == {"--backbone", "--warm-start", "--results-dir"}
    assert dit[0] == U, "the DiT runs under the 4-channel interpreter"


def test_dit_takes_the_action_history_and_checkpointing_knobs(tmp_path):
    assert "--action-history 32 " in _launch_dry(tmp_path, "dit")
    assert "--action-history 0 " in _launch_dry(tmp_path, "dit", ACTION_HISTORY="0")
    assert "--grad-ckpt" not in _launch_dry(tmp_path, "dit")
    assert _launch_dry(tmp_path, "dit", GRAD_CKPT="1").count("--grad-ckpt") == 1


def test_dit_builds_from_a_local_checkpoint_with_the_executed_control_history(tmp_path, monkeypatch):
    """`--backbone dit --warm-start <local .pt> --action-history 32` over 19 control bits is a supported
    path of build_model: the ImageNet weights load, the control embedder is attached, the class table
    it replaces is frozen. DiT-S/2 stands in for DiT-XL/2: the same code at 1/20 of the parameters."""
    import backbones
    import train_wm
    small = backbones.DiT_models["DiT-S/2"]
    imagenet = small(input_size=32, in_channels=4, num_classes=1000, learn_sigma=True)
    ck = tmp_path / "DiT-XL-2-256x256.pt"
    torch.save(imagenet.state_dict(), ck)
    monkeypatch.setitem(backbones.DiT_models, "DiT-XL/2", small)
    m = backbones.build_model("dit", 29, 32, 10, grad_ckpt=False, warm_start=str(ck), action_dropout=0.0,
                              action_history=32, control_bits=19)
    assert (m.control_history.length, m.control_history.bits) == (32, 19)
    assert not any(p.requires_grad for p in m.dit.y_embedder.parameters())
    w = m.dit.x_embedder.proj.weight
    assert torch.equal(w[:, -4:], imagenet.x_embedder.proj.weight) and not w[:, :-4].any(), \
        "the ImageNet kernel belongs on the noisy target, zeros on the 32 context latents"
    assert torch.equal(m.dit.blocks[0].attn.qkv.weight, imagenet.blocks[0].attn.qkv.weight)
    out = m(torch.randn(2, 4, 32, 40), torch.full((2,), 300), torch.randint(0, 2, (2, 32, 19)).float(),
            torch.randn(2, 128, 32, 40), torch.zeros(2, dtype=torch.long))
    assert out.shape == (2, 4, 32, 40)
    # and the certified command parses into exactly that request
    cmd = _launch_dry(tmp_path, "dit", CERT_QUERY="1").split("train_wm.py", 1)[1].split()
    a = train_wm.build_parser().parse_args(cmd)
    assert (a.backbone, a.warm_start, a.action_history, a.tic_stride, a.action_dropout) == \
        ("dit", f"{tmp_path}/weights/DiT-XL-2-256x256.pt", 32, 1, 0.0)


def test_the_dit_sees_which_controls_occurred_but_not_their_order():
    """The DiT has no cross-attention, so its control tokens are averaged into the adaLN vector
    (backbones.DiTWorldModel). The MLP acts per token and the positions are added before the mean,
    so the mean is a bag of the controls: reversing them leaves the output unchanged while flipping
    the newest changes it. The launcher header says so; this keeps the header true."""
    import backbones
    torch.manual_seed(0)
    m = backbones.DiTWorldModel(num_actions=29, context_frames=4, noise_buckets=10, model_name="DiT-S/2",
                                action_dropout=0.0, grad_ckpt=False, action_history=4, control_bits=19)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():   # adaLN-Zero and the zero output head hide every conditioning signal at init
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    m.eval()
    x, ctx = torch.randn(2, 4, 32, 40, generator=g), torch.randn(2, 16, 32, 40, generator=g)
    t, bucket = torch.full((2,), 500), torch.zeros(2, dtype=torch.long)
    c = torch.randint(0, 2, (2, 4, 19), generator=g).float()
    c[:, -1] = 1 - c[:, 0]                              # the newest differs from the oldest
    flipped = c.clone()
    flipped[:, -1] = 1 - flipped[:, -1]
    with torch.no_grad():
        base = m(x, t, c, ctx, bucket)
        reordered = m(x, t, c.flip(1), ctx, bucket)
        changed = m(x, t, flipped, ctx, bucket)
    assert (base - reordered).abs().max() < 1e-5, "the DiT now sees the order; update the launcher header"
    assert (base - changed).abs().max() > 1e-3
    head = open(os.path.join(SCRIPTS, "launch_nexttic.sh")).read().split("\nset -u", 1)[0]
    assert "not their order" in " ".join(head.replace("#", " ").split())


def test_a_certified_dit_launch_starts_in_its_own_session(tmp_path):
    from test_launch_pin import certify, launch, root_for_launch
    root, sha, bindir = root_for_launch(tmp_path)
    certify(tmp_path, root, bindir, backbones=("dit",))
    p, started = launch(tmp_path, root, bindir, backbone="dit")
    assert p.returncode == 0, p.stderr
    assert len(started) == 1 and "-s train-dit-nexttic " in started[0]
    assert f"--results-dir {root}/results_spiderman/043-dit-nexttic " in started[0]
    p, started = launch(tmp_path, root, bindir, backbone="unet")
    assert p.returncode != 0 and started == [], "a DiT certificate certified a U-Net launch"


# ---------------------------------------------------------------------------------------
# 2. RUN_NAME: one name for the run everywhere it is named, and a session derived from it
# ---------------------------------------------------------------------------------------

LAUNCH = os.path.join(SCRIPTS, "launch_nexttic.sh")
GATES = os.path.join(REPO, "scripts", "cluster", "gates.sh")
LAUNCH_RUNS = os.path.join(REPO, "scripts", "cluster", "launch_runs.sh")
STATUS = os.path.join(REPO, "scripts", "cluster", "status.sh")
REQ = {"RUN_NAME": "044-unet-nexttic-reqaction", "ACTION_HISTORY": "0"}
DEFAULT_RUNS = {"unet": ("040-unet-nexttic", "train-unet-nexttic"),
                "pixart": ("041-pixart-nexttic", "train-pixart-nexttic"),
                "sd35": ("042-sd35-nexttic", "train-sd35-nexttic"),
                "dit": ("043-dit-nexttic", "train-dit-nexttic")}

# records every call; `has-session` answers yes only for $LIVE_SESSION, a run already training
LIVE_TMUX = """#!/bin/bash
echo "tmux $*" >> "$TMUX_LOG"
if [ "$1" = has-session ]; then [ -n "${LIVE_SESSION:-}" ] && [ "$3" = "$LIVE_SESSION" ] && exit 0; exit 1; fi
exit 0
"""


def _clean(**env):
    e = {k: v for k, v in os.environ.items() if k not in GATE_KNOBS}
    e.update(env)
    return e


def _query(backbone, **env):
    """launch_nexttic.sh RUN_QUERY=1: `<run> <session>`."""
    e = _clean(DRY="1", DOOM_ROOT="/nonexistent-run-query", RUN_QUERY="1", **env)
    return subprocess.run(["bash", LAUNCH, "0", backbone], capture_output=True, text=True, env=e, timeout=60)


def test_the_default_runs_and_sessions_are_the_ones_the_live_runs_use():
    for bb, (run, session) in DEFAULT_RUNS.items():
        p = _query(bb)
        assert p.returncode == 0 and p.stdout.split() == [run, session], (bb, p.stdout, p.stderr)
    p = _query("unet", **REQ)
    assert p.stdout.split() == ["044-unet-nexttic-reqaction", "train-unet-nexttic-reqaction"]


def test_run_name_replaces_the_default_everywhere_the_run_is_named(tmp_path):
    for extra in ({}, {"FIT": "20"}, {"CERT_QUERY": "1"}):
        out = _launch_dry(tmp_path, "unet", **REQ, **extra)
        assert "044-unet-nexttic-reqaction" in out and "040-unet-nexttic" not in out, (extra, out)
        assert "--action-history 0 " in out, extra
    out = _launch_dry(tmp_path, "unet", **REQ)
    assert f"--results-dir {tmp_path}/results_spiderman/044-unet-nexttic-reqaction " in out
    assert f">> {tmp_path}/logs/train_044-unet-nexttic-reqaction.log" in out
    fit = _launch_dry(tmp_path, "unet", FIT="20", **REQ)
    assert f"--results-dir {tmp_path}/results_spiderman/044-unet-nexttic-reqaction/fitcheck " in fit


def test_a_run_name_that_is_not_a_plain_name_or_is_a_backbone_is_refused():
    for bad in ("a b", "x.y", "a:b", "-x", "../x", "unet", "sd35", "dit"):
        p = _query("unet", RUN_NAME=bad)
        assert p.returncode != 0 and "RUN_NAME" in p.stderr, (bad, p.stdout)


def test_a_named_run_launches_beside_the_live_session_of_its_backbone(tmp_path):
    """044 must start while train-unet-nexttic (040) is alive. Under the per-backbone session name the
    launcher answered `alive` and started nothing. Both U-Net runs hold certificates in one file."""
    from test_launch_pin import certify, launch, root_for_launch
    root, sha, bindir = root_for_launch(tmp_path)
    (bindir / "tmux").write_text(LIVE_TMUX)
    certify(tmp_path, root, bindir)
    certify(tmp_path, root, bindir, **REQ)
    held = json.loads((root / "GATES_CERT.json").read_text())["backbones"]
    assert set(held) == {"040-unet-nexttic", "044-unet-nexttic-reqaction"}
    p, started = launch(tmp_path, root, bindir, LIVE_SESSION="train-unet-nexttic", **REQ)
    assert p.returncode == 0 and len(started) == 1, p.stderr
    assert "-s train-unet-nexttic-reqaction " in started[0] and "--action-history 0 " in started[0]
    assert f"--results-dir {root}/results_spiderman/044-unet-nexttic-reqaction " in started[0]
    assert "044-unet-nexttic-reqaction" in (root / "logs" / "resumes.log").read_text()
    p, started = launch(tmp_path, root, bindir, LIVE_SESSION="train-unet-nexttic")
    assert p.returncode == 0 and started == [] and "040-unet-nexttic alive" in p.stdout


def test_each_run_is_checked_against_its_own_entry(tmp_path):
    from test_launch_pin import certify, launch, root_for_launch
    root, sha, bindir = root_for_launch(tmp_path)
    certify(tmp_path, root, bindir, **REQ)
    p, started = launch(tmp_path, root, bindir, RUN_NAME=REQ["RUN_NAME"])      # without ACTION_HISTORY=0
    assert p.returncode != 0 and "launch command differs" in p.stderr and started == []
    p, started = launch(tmp_path, root, bindir)                                # the default run
    assert p.returncode != 0 and "certifies no unet launch named 040-unet-nexttic" in p.stderr
    p, started = launch(tmp_path, root, bindir, backbone="dit", **REQ)         # its name, another backbone
    assert p.returncode != 0 and "not a dit one" in p.stderr and started == []


def _legacy(root):
    """GATES_CERT.json as the code before run keys wrote it for the live 040 and 042 runs."""
    def entry(bb, run):
        return {"backbone": bb, "command": f"/x/python train_wm.py --backbone {bb} --results-dir /r/{run}",
                "init_from": "", "commit": "a" * 40, "clean": True, "space": "sd35" if bb == "sd35" else "sd15",
                "gpu": "1", "gates": [{"gate": "0 pin", "scope": "all", "status": "ok", "detail": "x"}],
                "certified_at": "2026-09-23T09:00:00+0000"}
    cert = {"backbones": {"unet": entry("unet", "040-unet-nexttic"), "sd35": entry("sd35", "042-sd35-nexttic")},
            "written_by": "scripts/cluster/gates.sh"}
    (root / "GATES_CERT.json").write_text(json.dumps(cert, indent=1))
    return cert["backbones"]


def test_run_keyed_writes_and_revocations_leave_the_legacy_entries_as_they_were(tmp_path):
    from test_launch_pin import certify, root_for_launch
    root, sha, bindir = root_for_launch(tmp_path)
    legacy = _legacy(root)
    cert = str(root / "GATES_CERT.json")
    certify(tmp_path, root, bindir, backbones=("dit",))
    certify(tmp_path, root, bindir, **REQ)
    held = json.loads(open(cert).read())["backbones"]
    assert set(held) == {"unet", "sd35", "043-dit-nexttic", "044-unet-nexttic-reqaction"}
    assert {k: held[k] for k in ("unet", "sd35")} == legacy
    for bb, run in (("dit", "043-dit-nexttic"), ("unet", "044-unet-nexttic-reqaction")):
        gc.main(gc.build_parser().parse_args(["revoke", "--cert", cert, "--backbone", bb, "--run", run]))
    assert json.loads(open(cert).read())["backbones"] == legacy, "the file must stand while an entry is left"
    for bb in ("unet", "sd35"):
        with pytest.raises(SystemExit, match="not a run name"):
            gc.main(gc.build_parser().parse_args(["revoke", "--cert", cert, "--backbone", bb, "--run", bb]))
    assert json.loads(open(cert).read())["backbones"] == legacy
    assert set(gc.LEGACY_KEYS) == set(__import__("backbones").BACKBONES)


def test_a_legacy_entry_certifies_no_launch_of_this_code(tmp_path):
    from test_launch_pin import launch, root_for_launch
    root, sha, bindir = root_for_launch(tmp_path)
    _legacy(root)
    p, started = launch(tmp_path, root, bindir)
    assert p.returncode != 0 and "certifies no unet launch named 040-unet-nexttic" in p.stderr and started == []


def _functions(path, names):
    """The bash source of the named functions of a script, one-liners included."""
    out, grab = [], False
    for ln in open(path).read().splitlines():
        if not grab and any(ln.startswith(f"{n}()") for n in names):
            if ln.rstrip().endswith("}"):
                out.append(ln)
                continue
            grab = True
        if grab:
            out.append(ln)
            if ln == "}":
                grab = False
    return "\n".join(out)


def test_the_gates_wait_on_the_session_the_launcher_named(tmp_path):
    """gates.sh waits for the smoke's tmux session, and after SMOKE_TIMEOUT kills it: it must be the
    run's own session, never the live run of the same backbone."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "tmux").write_text(LIVE_TMUX)
    (bindir / "tmux").chmod(0o755)
    log = tmp_path / "tmux.log"
    funcs = _functions(GATES, ("run_query", "run_name", "session_name", "wait_session"))
    script = 'gate_fail() { echo "GATE_FAILED $*" >&2; exit 1; }\n' + funcs + '\nwait_session "$BB" "4 smoke"\n'
    for bb, env, want in (("unet", {}, "train-unet-nexttic"), ("sd35", {}, "train-sd35-nexttic"),
                          ("dit", {}, "train-dit-nexttic"), ("unet", REQ, "train-unet-nexttic-reqaction")):
        if log.exists():
            log.unlink()
        e = _clean(LAUNCH=LAUNCH, BB=bb, SMOKE_TIMEOUT="60", TMUX_LOG=str(log),
                   PATH=f"{bindir}:{os.environ['PATH']}", **env)
        p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=e, timeout=60)
        assert p.returncode == 0, p.stderr
        assert log.read_text().split() == ["tmux", "has-session", "-t", want], (bb, env, log.read_text())


def test_no_script_names_a_session_after_the_backbone():
    for path in (LAUNCH, GATES, LAUNCH_RUNS, STATUS):
        code = [ln for ln in open(path).read().splitlines() if not ln.lstrip().startswith("#")]
        for pat in ("train-$BACKBONE", "train-$BB-", "train-$1-", "cut -d- -f2)-nexttic"):
            assert not [ln for ln in code if pat in ln], (os.path.basename(path), pat)


def test_the_gates_write_and_revoke_the_entry_of_the_run(tmp_path):
    from test_nexttic_defects3 import _gates_dry
    lines = _gates_dry(tmp_path)
    for bb, (run, _) in (("unet", DEFAULT_RUNS["unet"]), ("sd35", DEFAULT_RUNS["sd35"])):
        write = next(ln for ln in lines if "gate_certificate.py write" in ln and f"--backbone {bb} " in ln)
        assert f"--run {run} " in write, write
        revoke = next(ln for ln in lines if ln.startswith(f"DRY gate0 revoke {bb} "))
        assert f"--backbone {bb} --run {run} " in revoke, revoke


def _launch_runs(tmp_path, **env):
    e = _clean(DRY="1", DOOM_ROOT=str(tmp_path), **env)
    return subprocess.run(["bash", LAUNCH_RUNS], capture_output=True, text=True, env=e, timeout=60)


def test_launch_runs_names_each_rows_session_from_the_launcher(tmp_path):
    p = _launch_runs(tmp_path)
    assert p.returncode == 0, p.stderr
    sessions = [ln.split()[3] for ln in p.stdout.splitlines() if ln.startswith("DRY tmux session ")]
    assert sessions == ["train-unet-nexttic", "train-sd35-nexttic"]
    p = _launch_runs(tmp_path, ONLY="unet", **REQ)
    assert p.returncode == 0, p.stderr
    assert "DRY tmux session train-unet-nexttic-reqaction " in p.stdout
    assert f"{tmp_path}/results_spiderman/044-unet-nexttic-reqaction " in p.stdout
    p = _launch_runs(tmp_path, **REQ)
    assert p.returncode != 0 and "ONLY" in p.stderr, "one RUN_NAME reached both rows"


def test_status_asks_tmux_for_each_runs_own_session(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "tmux").write_text(LIVE_TMUX)
    (bindir / "tmux").chmod(0o755)
    log = tmp_path / "tmux.log"
    runs = {"040-unet-nexttic": ("unet", {}), "044-unet-nexttic-reqaction": ("unet", REQ),
            "043-dit-nexttic": ("dit", {}), "042-sd35-nexttic": ("sd35", {})}
    e = _clean(DOOM_ROOT=str(tmp_path), RUNS=" ".join(runs), TMUX_LOG=str(log), PATH=f"{bindir}:{os.environ['PATH']}")
    p = subprocess.run(["bash", STATUS], capture_output=True, text=True, env=e, timeout=60)
    assert p.returncode == 0, p.stderr
    asked = [ln.split()[-1] for ln in log.read_text().splitlines() if "has-session" in ln]
    assert asked == [_query(bb, **env).stdout.split()[1] for bb, env in runs.values()], asked


# ---------------------------------------------------------------------------------------
# 3. the gates certify one named run of any of the four backbones
# ---------------------------------------------------------------------------------------

def _runs(lines):
    return [ln for ln in lines if ln.startswith("DRY ") and re.search(r"\.py\b", ln)]


def test_a_dit_gate_run_uses_the_dit_card_micro_batch_and_the_four_channel_interpreter(tmp_path):
    from test_nexttic_defects3 import _gates_dry, _launch_line, _operator
    lines = _gates_dry(tmp_path, SMOKE_BBS="dit", VAES="sd15", UNET_GPU="1", DIT_GPU="3", MB_DIT="16",
                       ALLOW_ACCUM="1", PY_UNET=U, PY_SD35=S)
    text = "\n".join(lines)
    for marker in ("DRY gate3 fit dit", "DRY gate4 smoke dit", "DRY gate4c resume dit"):
        ln = _launch_line(lines, marker)
        assert "CUDA_VISIBLE_DEVICES=3 " in ln and "--per-gpu-batch 16 " in ln and "--backbone dit " in ln, marker
        assert "--action-history 32 " in ln, marker
    gpu = [ln for ln in lines if ln.startswith(("DRY gate1d", "DRY gate4b", "DRY gate5"))]
    assert len(gpu) == 2 + 1 + 4 and all("env CUDA_VISIBLE_DEVICES=3 " in ln for ln in gpu), gpu
    runs = _runs(lines)
    assert runs and all(U in ln and S not in ln for ln in runs), [ln for ln in runs if U not in ln][:3]
    assert f"{tmp_path}/results_spiderman/043-dit-nexttic/fitcheck/log.jsonl" in text
    write = next(ln for ln in lines if "gate_certificate.py write" in ln)
    assert "--backbone dit --run 043-dit-nexttic --space sd15 --gpu 3 " in write
    cd, env, args = _operator(lines, "dit")
    assert args == ["3", "dit"] and env["MB"] == "16" and env["PY_UNET"] == U and env["ACTION_HISTORY"] == "32"
    assert "RUN_NAME" not in env and "PY_SD35" not in env
    assert "DRY launch check dit: resolves to the certified command" in text


def test_a_pixart_gate_run_uses_its_own_card_and_rebuilds_pixart_for_the_readback(tmp_path):
    from test_nexttic_defects3 import _gates_dry, _launch_line, _operator
    lines = _gates_dry(tmp_path, SMOKE_BBS="pixart", VAES="sd15", UNET_GPU="1", PIXART_GPU="4", MB_PIXART="32")
    fit = _launch_line(lines, "DRY gate3 fit pixart")
    assert "CUDA_VISIBLE_DEVICES=4 " in fit and "--action-inject token" in fit
    reads = [ln for ln in lines if ln.startswith("DRY gate5 readback pixart")]
    assert len(reads) == 4 and all("--pixart-path PixArt-alpha/PixArt-XL-2-512x512" in ln for ln in reads)
    assert _operator(lines, "pixart")[2] == ["4", "pixart"]
    assert "DRY launch check pixart: resolves to the certified command" in "\n".join(lines)
    write = next(ln for ln in lines if "gate_certificate.py write" in ln)
    assert "--backbone pixart --run 041-pixart-nexttic --space sd15 --gpu 4 " in write


def test_the_pixart_gate_run_is_the_unet_gate_run_with_pixarts_names(tmp_path):
    """Every gate the 040 U-Net passed, the 041 PixArt passes the same way: the two DRY transcripts
    agree line for line once the backbone's name, run, warm start and evaluator source are swapped,
    and the PixArt row's one extra trainer flag (`--action-inject token`, the injection every PixArt
    row trained with) is set aside."""
    from test_nexttic_defects3 import _gates_dry
    knobs = {"VAES": "sd15", "UNET_GPU": "1", "PIXART_GPU": "1", "GATES_RUN_ID": "same", "PY_UNET": U}
    unet = _gates_dry(tmp_path, SMOKE_BBS="unet", **knobs)
    pixart = _gates_dry(tmp_path, SMOKE_BBS="pixart", **knobs)
    swap = (("--pixart-path PixArt-alpha/PixArt-XL-2-512x512", "--sd-path CompVis/stable-diffusion-v1-4"),
            ("--warm-start PixArt-alpha/PixArt-XL-2-512x512", "--warm-start CompVis/stable-diffusion-v1-4"),
            (" --action-inject token", ""), ("041-pixart-nexttic", "040-unet-nexttic"),
            ("train-pixart-nexttic", "train-unet-nexttic"), ("pixart", "unet"))

    def plain(ln):   # pytest names the directory after this test; the empty BB_FLAGS leaves double spaces
        return " ".join(ln.replace(str(tmp_path), "<root>").split())

    def as_unet(ln):
        ln = plain(ln)
        for a, b in swap:
            ln = ln.replace(a, b)
        return ln
    body = [ln for ln in pixart if not ln.startswith("DRY gates ")]
    assert [as_unet(ln) for ln in body] == [plain(ln) for ln in unet if not ln.startswith("DRY gates ")]
    assert sum("--action-inject token" in ln for ln in body) == 4, "fit, smoke, resume and certificate"


def test_the_new_card_and_micro_batch_knobs_default_to_the_unets_card_and_32(tmp_path):
    from test_nexttic_defects3 import _gates_dry, _operator
    for bb in ("dit", "pixart"):
        lines = _gates_dry(tmp_path, SMOKE_BBS=bb, VAES="sd15", UNET_GPU="5")
        cd, env, args = _operator(lines, bb)
        assert args == ["5", bb] and env["MB"] == "32", (bb, env, args)


def test_the_reqaction_gate_run_certifies_exactly_that_launch(tmp_path):
    from test_nexttic_defects3 import _args, _cert_line, _gates_dry, _launch_line, _operator
    lines = _gates_dry(tmp_path, SMOKE_BBS="unet", VAES="sd15", **REQ)
    text = "\n".join(lines)
    assert "040-unet-nexttic" not in text, [ln for ln in lines if "040-unet-nexttic" in ln][:3]
    for marker in ("DRY gate3 fit unet", "DRY gate4 smoke unet", "DRY gate4c resume unet"):
        assert "--action-history 0 " in _launch_line(lines, marker), marker
    assert f"{tmp_path}/results_spiderman/044-unet-nexttic-reqaction/fitcheck/log.jsonl" in text
    assert f"{tmp_path}/results_spiderman/044-unet-nexttic-reqaction/log.jsonl exists" in text
    cert = _args(_cert_line(lines, "unet")) + " "
    assert "--action-history 0 " in cert
    assert f"--results-dir {tmp_path}/results_spiderman/044-unet-nexttic-reqaction " in cert
    write = next(ln for ln in lines if "gate_certificate.py write" in ln)
    assert "--run 044-unet-nexttic-reqaction " in write
    assert "ACTION_HISTORY=0 RUN_NAME=044-unet-nexttic-reqaction" in write
    revoke = next(ln for ln in lines if ln.startswith("DRY gate0 revoke unet "))
    assert "--run 044-unet-nexttic-reqaction " in revoke
    cd, env, args = _operator(lines, "unet")
    assert env["RUN_NAME"] == "044-unet-nexttic-reqaction" and env["ACTION_HISTORY"] == "0"
    assert "DRY launch check unet: resolves to the certified command" in text
    head = next(ln for ln in lines if ln.startswith("DRY gates "))
    assert "runs=044-unet-nexttic-reqaction " in head and "action_history=0" in head


def test_a_run_name_for_two_backbones_or_an_unknown_backbone_is_refused(tmp_path):
    for env, why in (({"SMOKE_BBS": "unet sd35", **REQ}, "RUN_NAME"), ({"SMOKE_BBS": "unet wan"}, "unknown backbone"),
                     ({"SMOKE_BBS": "unet", "RUN_NAME": "unet"}, "names no run")):
        e = _clean(DRY="1", DOOM_ROOT=str(tmp_path), **env)
        p = subprocess.run(["bash", GATES], capture_output=True, text=True, env=e, timeout=60)
        assert p.returncode != 0 and "GATE_FAILED preflight" in p.stderr and why in p.stderr, (env, p.stderr)


def _gate_results(root, run_id, backbone):
    """Every required gate passed, in the scopes a gate run records for one 4-channel backbone."""
    rows = []
    for g in gc.REQUIRED_GATES:
        scope = "all" if g in ("0 pin", "2 alignment") else ("space:sd15" if g.startswith("1") else f"bb:{backbone}")
        rows.append({"gate": g, "scope": scope, "status": "ok", "detail": ""})
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / f"gates_results_{run_id}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.mark.parametrize("backbone,env,run", [("dit", {}, "043-dit-nexttic"),
                                              ("unet", REQ, "044-unet-nexttic-reqaction"),
                                              ("pixart", {}, "041-pixart-nexttic")])
def test_the_printed_certificate_and_launch_work_beside_the_live_entries(tmp_path, backbone, env, run):
    """End to end on a throwaway root holding the live runs' entries as the older code wrote them: run
    the certificate write the gates print, then the launch they print (fake tmux). It launches exactly
    that run, the live entries are untouched, and a failed gate run of it revokes only its own entry."""
    import shlex
    from test_launch_pin import TRAIN_IDS, VAL_IDS, root_for_launch
    from test_nexttic_defects3 import _cert_line, _gates_dry, _operator
    root, sha, bindir = root_for_launch(tmp_path)
    legacy = _legacy(root)
    cert = root / "GATES_CERT.json"
    knobs = {"RUN_REPO": str(root / "repo"), "UNET_GPU": "1", "DIT_GPU": "1", "PIXART_GPU": "1",
             "PY_UNET": sys.executable, "PY_SD35": sys.executable, "TRAIN_IDS": TRAIN_IDS, "VAL_IDS": VAL_IDS,
             "LAUNCH": LAUNCH,
             "SMOKE_BBS": backbone, "VAES": "sd15", "GATES_RUN_ID": "e2e", **env}
    lines = _gates_dry(root, **knobs)
    _gate_results(root, "e2e", backbone)
    printed = next(ln for ln in lines if ln.startswith(f"DRY certificate {backbone} ") and " write " in ln)
    write = shlex.split(printed[len(f"DRY certificate {backbone} "):].split(' "<the certificate command', 1)[0])
    command = _cert_line(lines, backbone)[len(f"DRY certificate command {backbone} "):]
    p = subprocess.run([*write, command], capture_output=True, text=True, timeout=120, cwd=REPO)
    assert p.returncode == 0, p.stderr + p.stdout
    held = json.loads(cert.read_text())["backbones"]
    assert set(held) == {"unet", "sd35", run} and {k: held[k] for k in ("unet", "sd35")} == legacy
    assert held[run]["backbone"] == backbone and held[run]["run"] == run
    cd, op, args = _operator(lines, backbone)
    log = tmp_path / "tmux.log"
    base = {"PATH": f"{bindir}:{os.environ['PATH']}", "HOME": str(tmp_path / "home"), "TMUX_LOG": str(log)}
    p = subprocess.run(["bash", LAUNCH, *args], capture_output=True, text=True, env={**base, **op}, timeout=120)
    assert p.returncode == 0, p.stderr
    started = [ln for ln in log.read_text().splitlines() if "new-session" in ln]
    assert len(started) == 1 and f"--results-dir {root}/results_spiderman/{run} " in started[0]
    assert f"-s {_query(backbone, **env).stdout.split()[1]} " in started[0]
    # a later gate run of the same run fails at gate 0 and revokes that run's entry, nothing else
    (root / "repo" / "train_wm.py").write_text("# dirty\n")
    e = _clean(DOOM_ROOT=str(root), REPO=str(root / "repo"), **{k: v for k, v in knobs.items() if k != "GATES_RUN_ID"})
    p = subprocess.run(["bash", GATES], capture_output=True, text=True, env=e, timeout=120)
    assert p.returncode != 0 and "GATE_FAILED 0 pin" in p.stderr, p.stderr
    assert json.loads(cert.read_text())["backbones"] == legacy


# ---------------------------------------------------------------------------------------
# 4. the invocations the headers and the runbook give are the ones that certify the three rows
# ---------------------------------------------------------------------------------------

RUNBOOK = os.path.join(REPO, ".claude", "analyses", "launch-runbook-2026-09-23.md")
# the Spiderman knobs every one of the three gate runs sets
SEP24 = {"DOOM_ROOT": "/sata2/data/rnagabhi/doom", "RUN_REPO": "/sata2/data/rnagabhi/doom/repo_launch2",
         "PY_UNET": "$HOME/miniconda3/envs/doom/bin/python", "PY_SD35": "$HOME/wanenc/bin/python",
         "LAUNCH_STEPS": "200000", "WORKERS": "12", "EVAL_EVERY": "5000", "EVAL_DEVICE": "cuda:3", "VAES": "sd15"}
# run: (backbone, action history, its card knob, its micro-batch knob), PixArt first as the next launch
ROWS = {"041-pixart-nexttic": ("pixart", "32", "PIXART_GPU", "MB_PIXART"),
        "043-dit-nexttic": ("dit", "32", "DIT_GPU", "MB_DIT"),
        "044-unet-nexttic-reqaction": ("unet", "0", "UNET_GPU", "MB_UNET")}


def _flat(text):
    """A text's lines joined, comment marks and line continuations removed."""
    return " ".join(ln.strip().lstrip("#").strip().rstrip("\\").strip() for ln in text.splitlines())


def _gate_invocations(path):
    """{NAME: VALUE} of every documented `cd .../repo_launch2 && ... bash scripts/cluster/gates.sh`, in order."""
    import shlex
    found = re.findall(r"cd /sata2/data/rnagabhi/doom/repo_launch2 && (.*?) bash scripts/cluster/gates\.sh",
                       _flat(open(path).read()))
    return [dict(t.split("=", 1) for t in shlex.split(inv)) for inv in found]


def _dry_as_documented(tmp_path, env):
    """The documented gate run under DRY against a throwaway root, printed as if on the documented one,
    with $HOME standing for a fixed path."""
    from test_nexttic_defects3 import _gates_dry
    run = {k: v.replace("$HOME", "/HOME") for k, v in env.items()}
    run["DOOM_ROOT"] = str(tmp_path)
    return [ln.replace(str(tmp_path), env["DOOM_ROOT"]) for ln in _gates_dry(tmp_path, **run)]


@pytest.mark.parametrize("path", [GATES, RUNBOOK], ids=os.path.basename)
def test_the_documented_gate_runs_certify_the_three_new_rows(tmp_path, path):
    from test_nexttic_defects3 import _operator
    found = _gate_invocations(path)
    assert len(found) == 3, found
    got = []
    for env in found:
        assert {k: env.get(k) for k in SEP24} == SEP24, env
        lines = _dry_as_documented(tmp_path, env)
        run = next(ln for ln in lines if ln.startswith("DRY gates ")).split("runs=")[1].split()[0]
        bb, history, card, mb = ROWS[run]
        assert env["SMOKE_BBS"] == bb and env[card] == "1" and env[mb] == "32", env
        cd, op, args = _operator(lines, bb)
        assert args == ["1", bb] and op["ACTION_HISTORY"] == history and (op["STEPS"], op["MB"]) == ("200000", "32")
        assert f"DRY launch check {bb}: resolves to the certified command" in "\n".join(lines)
        got.append(run)
    assert got == list(ROWS), "PixArt, the next launch, comes first"


def test_the_launch_header_shows_the_launches_the_documented_gate_runs_print(tmp_path):
    head = _flat(open(LAUNCH).read().split("\nset -u", 1)[0])
    shown = re.findall(r"(cd /sata2/data/rnagabhi/doom/repo_launch2 && .*? "
                       r"bash scripts/spiderman/launch_nexttic\.sh \d+ \w+)", head)
    assert len(shown) == 3, shown
    printed = []
    for env in _gate_invocations(GATES):
        bb = env["SMOKE_BBS"]
        ln = next(ln for ln in _dry_as_documented(tmp_path, env) if ln.startswith(f"DRY launch {bb} "))
        printed.append(ln[len(f"DRY launch {bb} "):].replace("/HOME/", "$HOME/"))
    assert [" ".join(s.split()) for s in shown] == printed


def test_launch_runs_points_the_three_new_rows_to_their_gate_runs():
    head = open(LAUNCH_RUNS).read().split("\nset -u", 1)[0]
    assert all(run in head for run in ROWS) and "GATES_LAUNCH" in head
