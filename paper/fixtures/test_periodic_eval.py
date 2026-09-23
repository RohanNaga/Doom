"""The trainer's periodic evaluation (`--eval-every`): one detached read per multiple, never a stall.

At every multiple of `--eval-every`, right after the checkpoint for that step is on disk, train_wm.py
starts ONE detached process that runs eval_tf.py live and EMA at horizons 1 and 4 and then
smoke_probe.py, and logs `eval_launched` (or `eval_skipped` with the reason). Every process here is a
stub: nothing is spawned, and nothing touches a GPU or the network.

    python -m pytest paper/fixtures/test_periodic_eval.py -q
"""
import json
import os
import shlex
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

os.environ.setdefault("ACCELERATE_USE_CPU", "1")

import periodic_eval as pe  # noqa: E402
import train_wm  # noqa: E402


def parse(*flags):
    return train_wm.build_parser().parse_args(["--backbone", "unet", *flags])


# ---------------------------------------------------------------------------------------
# the cadence is checked when the command line is parsed
# ---------------------------------------------------------------------------------------

def test_the_defaults_leave_it_off():
    a = parse()
    assert a.eval_every == 0 and a.eval_windows == 512 and a.eval_steps == 10
    assert a.eval_device is None and a.eval_latents_dir == "" and a.eval_split == "" and a.eval_parquet_dir == ""


@pytest.mark.parametrize("flags", [
    ("--eval-every", "5000", "--ckpt-every", "5000"),
    ("--eval-every", "10000", "--ckpt-every", "5000"),
    ("--eval-every", "10000", "--ckpt-every", "3000", "--snapshot-every", "10000", "--local-snapshots",
     "--val-every", "1000"),
    ("--eval-every", "5000", "--ckpt-every", "5000", "--eval-device", "cuda:3"),
    ("--eval-every", "5000", "--ckpt-every", "5000", "--eval-device", "cpu"),
])
def test_a_cadence_that_lands_on_a_written_checkpoint_is_accepted(flags):
    assert parse(*flags).eval_every > 0


@pytest.mark.parametrize("flags,why", [
    (("--eval-every", "-5"), "--eval-every"),
    (("--eval-every", "3000", "--ckpt-every", "5000", "--snapshot-every", "10000", "--local-snapshots"),
     "multiple"),
    (("--eval-every", "10000", "--ckpt-every", "3000", "--snapshot-every", "10000"), "--local-snapshots"),
    (("--eval-every", "5000", "--ckpt-every", "5000", "--keep-last", "0"), "--keep-last"),
    (("--eval-every", "5000", "--ckpt-every", "5000", "--eval-device", "gpu3"), "--eval-device"),
])
def test_a_cadence_with_no_checkpoint_to_read_is_refused_at_parse_time(flags, why, capsys):
    with pytest.raises(SystemExit):
        parse(*flags)
    assert why in capsys.readouterr().err


def test_the_help_says_what_the_default_device_costs_and_what_production_sets():
    text = " ".join(train_wm.build_parser().format_help().split())
    assert "stalls training" in text
    assert "EVAL_EVERY=5000" in text and "EVAL_DEVICE=cuda:3" in text


# ---------------------------------------------------------------------------------------
# what one launch starts
# ---------------------------------------------------------------------------------------

class FakeProc:
    def __init__(self, pid, code=None):
        self.pid, self.code = pid, code

    def poll(self):
        return self.code


class Spawner:
    """Stands in for subprocess.Popen: records each launch; `alive` decides what poll() returns."""

    def __init__(self, alive=True, fail=None):
        self.calls, self.alive, self.fail = [], alive, fail

    def __call__(self, argv, **kw):
        if self.fail:
            raise self.fail
        self.calls.append((argv, kw))
        return FakeProc(4000 + len(self.calls), None if self.alive else 1)


def run_args(tmp_path, *flags, backbone="unet"):
    a = train_wm.build_parser().parse_args(
        ["--backbone", backbone, "--results-dir", str(tmp_path / "040-unet-nexttic"), "--tic-stride", "1",
         "--action-history", "32", "--context-frames", "32", "--num-actions", "29", "--noise-buckets", "10",
         "--warm-start", {"unet": "CompVis/stable-diffusion-v1-4",
                          "sd35": "stabilityai/stable-diffusion-3.5-medium"}[backbone],
         "--hf-cache", "/d/hf/hub", "--latents-dir", "/d/lat/arenas", "--val-latents-dir", "/d/lat_eval/val",
         "--val-episode-ids", "6000:6100", "--episode-ids", "0:2000", "--ckpt-every", "5000",
         "--snapshot-every", "10000", "--local-snapshots", "--val-every", "1000", "--eval-every", "5000",
         "--eval-parquet-dir", "/d/raw/arenas", *flags])
    os.makedirs(a.results_dir, exist_ok=True)
    return a


def make_ckpts(args, *names):
    for n in names:
        open(os.path.join(args.results_dir, n), "wb").close()


class Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, **kw):
        self.events.append(kw)


def evaluator(tmp_path, monkeypatch, spawner, *flags, backbone="unet", wandb_run="040-unet-nexttic",
              device="cuda:0", visible="1", split=True):
    import torch
    args = run_args(tmp_path, *flags, backbone=backbone)
    if split:
        monkeypatch.setattr(pe, "default_split", lambda d: str(tmp_path / "split_val.json"))
        (tmp_path / "split_val.json").write_text(json.dumps({"val": [6000, 6001]}))
    monkeypatch.setattr(pe, "_popen", spawner)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setenv("WANDB_SERVICE", "unix-socket-of-the-trainers-wandb-service")
    rec = Recorder()
    ev = pe.PeriodicEval(args, 4 if backbone == "unet" else 16, torch.device(device), rec, wandb_run=wandb_run,
                         python="/env/bin/python")
    return args, ev, rec


def script_commands(spawner):
    """The command lines of the launched script, one argv per evaluator call."""
    argv, _ = spawner.calls[-1]
    assert argv[0] == "bash"
    lines = open(argv[1]).read().splitlines()
    return [shlex.split(ln) for ln in lines if "eval_tf.py" in ln or "smoke_probe.py" in ln]


def flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_only_multiples_are_due(tmp_path, monkeypatch):
    _, ev, _ = evaluator(tmp_path, monkeypatch, Spawner())
    assert [s for s in range(1, 20001) if ev.due(s)] == [5000, 10000, 15000, 20000]


def test_a_recovery_step_reads_the_recovery_checkpoint_and_a_snapshot_step_the_snapshot(tmp_path, monkeypatch):
    sp = Spawner(alive=False)
    args, ev, rec = evaluator(tmp_path, monkeypatch, sp)
    make_ckpts(args, "0005000.pt", "0010000.pt", "snap_0010000.pt")
    ev.launch(5000)
    (ck,) = {flag(c, "--ckpt") for c in script_commands(sp)}
    assert ck == os.path.join(args.results_dir, "eval_0005000", "0005000.pt")
    assert os.path.samefile(ck, os.path.join(args.results_dir, "0005000.pt"))
    ev.launch(10000)
    (ck,) = {flag(c, "--ckpt") for c in script_commands(sp)}
    assert ck == os.path.join(args.results_dir, "eval_0010000", "snap_0010000.pt")
    assert os.path.samefile(ck, os.path.join(args.results_dir, "snap_0010000.pt"))
    assert [e["event"] for e in rec.events if e["event"] != "eval_finished"] == ["eval_launched", "eval_launched"]


def script_lines(spawner):
    argv, _ = spawner.calls[-1]
    return open(argv[1]).read().splitlines()


def test_the_read_holds_a_hard_link_that_pruning_cannot_remove(tmp_path, monkeypatch):
    """Pruning a recovery checkpoint under a read that is still on it raised FileNotFoundError. The
    read now opens a hard link in its own directory, which the trainer's pruning (top-level
    `NNNNNNN.pt` only) never lists, and the wrapper removes the link when the read is done."""
    sp = Spawner()
    args, ev, _ = evaluator(tmp_path, monkeypatch, sp)
    make_ckpts(args, "0005000.pt")
    with open(os.path.join(args.results_dir, "0005000.pt"), "wb") as f:
        f.write(b"weights")
    ev.launch(5000)
    link = os.path.join(args.results_dir, "eval_0005000", "0005000.pt")
    os.remove(os.path.join(args.results_dir, "0005000.pt"))          # what the trainer's pruning does
    with open(link, "rb") as f:
        assert f.read() == b"weights"
    lines = script_lines(sp)
    last_read = max(i for i, ln in enumerate(lines) if "eval_tf.py" in ln or "smoke_probe.py" in ln)
    rm = [i for i, ln in enumerate(lines) if ln.startswith("rm -f") and link in ln]
    assert rm and rm[0] > last_read, "the link must outlive every read and then be removed"


def test_a_link_that_cannot_be_made_is_a_copy_made_before_the_reads(tmp_path, monkeypatch):
    """The copy runs in the detached wrapper, not on the training thread: a recovery checkpoint of the
    SD 3.5 row is 35 GB, and copying it inline would stall training for minutes."""
    def no_links(src, dst):
        raise OSError("hard links not supported")

    monkeypatch.setattr(pe.os, "link", no_links)
    sp = Spawner()
    args, ev, _ = evaluator(tmp_path, monkeypatch, sp)
    make_ckpts(args, "0005000.pt")
    ev.launch(5000)
    src = os.path.join(args.results_dir, "0005000.pt")
    dst = os.path.join(args.results_dir, "eval_0005000", "0005000.pt")
    assert {flag(c, "--ckpt") for c in script_commands(sp)} == {dst}
    lines = script_lines(sp)
    cp = [i for i, ln in enumerate(lines) if ln.startswith("cp ") and src in ln and dst in ln]
    first_read = min(i for i, ln in enumerate(lines) if "eval_tf.py" in ln)
    assert cp and cp[0] < first_read
    assert any(ln.startswith("rm -f") and dst in ln for ln in lines)


def test_a_snapshot_that_cannot_be_linked_is_read_in_place(tmp_path, monkeypatch):
    """Snapshots are never pruned, so they need no copy."""
    def no_links(src, dst):
        raise OSError("hard links not supported")

    monkeypatch.setattr(pe.os, "link", no_links)
    sp = Spawner()
    args, ev, _ = evaluator(tmp_path, monkeypatch, sp)
    make_ckpts(args, "0010000.pt", "snap_0010000.pt")
    ev.launch(10000)
    snap = os.path.join(args.results_dir, "snap_0010000.pt")
    assert {flag(c, "--ckpt") for c in script_commands(sp)} == {snap}
    assert not any(ln.startswith(("cp ", "rm -f")) for ln in script_lines(sp))


def test_one_detached_process_runs_four_reads_then_the_probe(tmp_path, monkeypatch):
    sp = Spawner()
    args, ev, rec = evaluator(tmp_path, monkeypatch, sp)
    make_ckpts(args, "0005000.pt")
    assert ev.launch(5000) == 4001
    (argv, kw), = sp.calls
    assert kw["start_new_session"] is True, "the child must survive the trainer's own session"
    out = os.path.join(args.results_dir, "eval_0005000")
    assert kw["stdout"].name == os.path.join(out, "launch.log")
    assert kw["env"]["CUDA_VISIBLE_DEVICES"] == "1", "the default device is the trainer's own card"
    assert "WANDB_SERVICE" not in kw["env"], "the child must not ride on the trainer's W&B service"
    cmds = script_commands(sp)
    tf, probe = cmds[:4], cmds[4]
    assert len(cmds) == 5 and all(os.path.basename(c[1]) == "eval_tf.py" for c in tf)
    assert os.path.basename(probe[1]) == "smoke_probe.py" and all(c[0] == "/env/bin/python" for c in cmds)
    reads = {(("--use-ema" in c), flag(c, "--horizon-tics")): flag(c, "--out-dir") for c in tf}
    assert reads == {(False, "1"): os.path.join(out, "tf_live_h1"), (True, "1"): os.path.join(out, "tf_ema_h1"),
                     (False, "4"): os.path.join(out, "tf_live_h4"), (True, "4"): os.path.join(out, "tf_ema_h4")}
    for c in tf:
        assert flag(c, "--latents-dir") == "/d/lat_eval/val" and flag(c, "--split") == str(tmp_path / "split_val.json")
        assert flag(c, "--parquet-dir") == "/d/raw/arenas" and flag(c, "--subset") == "val"
        assert flag(c, "--num-windows") == "512" and flag(c, "--steps") == "10"
        assert flag(c, "--backbone") == "unet" and flag(c, "--latent-channels") == "4"
        assert flag(c, "--sd-path") == "CompVis/stable-diffusion-v1-4" and flag(c, "--hf-cache") == "/d/hf/hub"
        assert flag(c, "--context-frames") == "32" and flag(c, "--num-actions") == "29"
        assert flag(c, "--wandb-run") == "040-unet-nexttic-eval" and flag(c, "--wandb-step") == "5000"
    assert flag(probe, "--out") == os.path.join(out, "probe.json") and flag(probe, "--device") == "cuda:0"
    assert flag(probe, "--episodes") == "6000:6100" and flag(probe, "--latents-dir") == "/d/lat_eval/val"
    assert flag(probe, "--wandb-run") == "040-unet-nexttic-eval" and flag(probe, "--wandb-step") == "5000"
    (e,) = rec.events
    assert e["event"] == "eval_launched" and e["step"] == 5000 and e["pid"] == 4001


@pytest.mark.parametrize("backbone", ["unet", "sd35"])
def test_every_generated_command_parses_under_its_own_evaluator(tmp_path, monkeypatch, backbone):
    """A typo in a generated flag would only surface hours into a run, as an argparse error in launch.log."""
    import eval_tf
    import smoke_probe
    sp = Spawner()
    args, ev, _ = evaluator(tmp_path, monkeypatch, sp, backbone=backbone)
    make_ckpts(args, "0005000.pt")
    ev.launch(5000)
    for c in script_commands(sp):
        parser = eval_tf.build_parser() if c[1].endswith("eval_tf.py") else smoke_probe.build_parser()
        a = parser.parse_args(c[2:])
        assert a.wandb_step == 5000 and a.backbone == backbone


def test_sd35_reads_decode_through_its_own_autoencoder(tmp_path, monkeypatch):
    sp = Spawner()
    args, ev, _ = evaluator(tmp_path, monkeypatch, sp, backbone="sd35")
    make_ckpts(args, "0005000.pt")
    ev.launch(5000)
    for c in script_commands(sp)[:4]:
        assert flag(c, "--latent-channels") == "16" and flag(c, "--sd35-path") == "stabilityai/stable-diffusion-3.5-medium"
        assert flag(c, "--vae-path") == "stabilityai/stable-diffusion-3.5-medium" and flag(c, "--vae-subfolder") == "vae"
        assert float(flag(c, "--latent-scale")) == 1.5305 and float(flag(c, "--latent-shift")) == 0.0609


@pytest.mark.parametrize("spec,visible,device", [("cuda:3", "3", "cuda:0"), ("cpu", "", "cpu")])
def test_the_eval_device_picks_the_childs_card(tmp_path, monkeypatch, spec, visible, device):
    sp = Spawner()
    args, ev, _ = evaluator(tmp_path, monkeypatch, sp, "--eval-device", spec)
    make_ckpts(args, "0005000.pt")
    ev.launch(5000)
    assert sp.calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == visible
    assert flag(script_commands(sp)[4], "--device") == device


def test_without_wandb_the_reads_are_not_logged_there(tmp_path, monkeypatch):
    sp = Spawner()
    args, ev, _ = evaluator(tmp_path, monkeypatch, sp, wandb_run=None)
    make_ckpts(args, "0005000.pt")
    ev.launch(5000)
    assert not any("--wandb-run" in c for c in script_commands(sp))


def test_a_second_read_is_skipped_while_the_first_is_alive(tmp_path, monkeypatch):
    sp = Spawner(alive=True)
    args, ev, rec = evaluator(tmp_path, monkeypatch, sp)
    make_ckpts(args, "0005000.pt", "0010000.pt", "snap_0010000.pt")
    assert ev.launch(5000) == 4001
    assert ev.launch(10000) is None
    assert len(sp.calls) == 1
    skip = rec.events[-1]
    assert skip["event"] == "eval_skipped" and skip["step"] == 10000 and "4001" in skip["reason"]
    sp.alive = False
    ev.proc = FakeProc(4001, 0)                     # the first read has finished
    assert ev.launch(10000) == 4002


def test_a_read_left_running_by_a_previous_trainer_process_is_seen(tmp_path, monkeypatch):
    """After a restart the Popen handle is gone; the pid file still names the live read."""
    sp = Spawner()
    args, ev, rec = evaluator(tmp_path, monkeypatch, sp)
    make_ckpts(args, "0005000.pt")
    with open(os.path.join(args.results_dir, pe.PID_FILE), "w") as f:
        json.dump({"pid": os.getpid(), "step": 0, "script": ""}, f)    # this process: certainly alive
    assert ev.launch(5000) is None and sp.calls == []
    assert rec.events[-1]["event"] == "eval_skipped"


@pytest.mark.parametrize("problem", ["no_checkpoint", "spawn_fails", "no_split"])
def test_nothing_about_a_read_can_raise_into_the_trainer(tmp_path, monkeypatch, problem):
    sp = Spawner(fail=OSError("fork failed") if problem == "spawn_fails" else None)
    args, ev, rec = evaluator(tmp_path, monkeypatch, sp, split=problem != "no_split")
    if problem != "no_checkpoint":
        make_ckpts(args, "0005000.pt")
    if problem == "no_split":
        monkeypatch.setattr(pe, "default_split", lambda d: str(tmp_path / "missing_split.json"))
    assert ev.launch(5000) is None
    (e,) = rec.events
    assert e["event"] == "eval_skipped" and e["step"] == 5000 and e["reason"]


# ---------------------------------------------------------------------------------------
# inside a real (tiny, CPU) training run
# ---------------------------------------------------------------------------------------

TINY_PIXART = dict(num_attention_heads=2, attention_head_dim=8, in_channels=4, out_channels=8, num_layers=2,
                   caption_channels=32, sample_size=64, patch_size=2, cross_attention_dim=16,
                   use_additional_conditions=False, norm_num_groups=2)


@pytest.fixture
def tiny_pixart(monkeypatch):
    from diffusers import PixArtTransformer2DModel as P
    monkeypatch.setattr(P, "from_pretrained", classmethod(lambda cls, *a, **k: cls(**TINY_PIXART)))
    monkeypatch.setattr(P, "load_config", classmethod(lambda cls, *a, **k: dict(TINY_PIXART)))
    monkeypatch.setenv("ACCELERATE_USE_CPU", "1")
    monkeypatch.setitem(sys.modules, "wandb", None)          # no W&B in these runs


def train_tiny(tmp_path, *flags):
    import numpy as np
    from pertic_fixtures import held_actions, write_pertic_episode
    d = str(tmp_path / "lat")
    rng = np.random.RandomState(0)
    for ep in range(3):
        btns = ["".join(str(int(b)) for b in rng.randint(0, 2, 19)) for _ in range(40)]
        write_pertic_episode(d, ep, held_actions([0, 1, 2] * 4)[:40], buttons=btns)
    split = tmp_path / "split_lat.json"
    split.write_text(json.dumps({"val": [2]}))
    out = str(tmp_path / "run")
    train_wm.main(train_wm.build_parser().parse_args(
        ["--backbone", "pixart", "--warm-start", "PixArt-alpha/PixArt-XL-2-512x512",
         "--latents-dir", d, "--results-dir", out, "--tic-stride", "1", "--action-history", "8",
         "--context-frames", "8", "--episode-ids", "0:2", "--val-episode-ids", "2:3",
         "--num-actions", "3", "--noise-buckets", "4", "--per-gpu-batch", "2", "--global-batch", "2",
         "--steps", "6", "--warmup", "1", "--log-every", "1", "--val-every", "2", "--val-windows", "2",
         "--ckpt-every", "2", "--ema-every", "1", "--num-workers", "0", "--action-dropout", "0.0",
         "--eval-every", "2", "--eval-split", str(split), *flags]))
    with open(os.path.join(out, "log.jsonl")) as f:
        return out, [json.loads(ln) for ln in f if ln.strip()]


def test_the_trainer_launches_at_each_multiple_after_its_checkpoint(tiny_pixart, tmp_path, monkeypatch):
    seen = []

    def spawn(argv, **kw):
        # the checkpoint the read names must already be complete on disk when the read starts
        ck = [c for c in open(argv[1]).read().split() if c.endswith(".pt")][0]
        seen.append((os.path.isfile(ck), os.path.basename(ck), kw["env"].get("CUDA_VISIBLE_DEVICES")))
        return FakeProc(5000 + len(seen), 0)

    monkeypatch.setattr(pe, "_popen", spawn)
    out, events = train_tiny(tmp_path)
    launched = [e for e in events if e["event"] == "eval_launched"]
    assert [e["step"] for e in launched] == [2, 4, 6]
    assert seen == [(True, "0000002.pt", ""), (True, "0000004.pt", ""), (True, "0000006.pt", "")]
    assert events[-1]["event"] == "end"


def test_the_trainers_pruning_leaves_every_reads_checkpoint_in_place(tiny_pixart, tmp_path, monkeypatch):
    """With --keep-last 2 the trainer deletes 0000002.pt at step 6. Every read's own link survives
    that (the stub reads never run, so no wrapper has removed its link yet)."""
    import torch
    monkeypatch.setattr(pe, "_popen", lambda argv, **kw: FakeProc(8000, 0))
    out, events = train_tiny(tmp_path, "--keep-last", "2")
    assert not os.path.exists(os.path.join(out, "0000002.pt")), "the fixture must actually prune"
    for step in (2, 4, 6):
        link = os.path.join(out, f"eval_{step:07d}", f"{step:07d}.pt")
        assert torch.load(link, map_location="cpu", weights_only=False)["step"] == step


def test_the_trainer_skips_while_a_read_is_alive(tiny_pixart, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pe, "_popen", lambda argv, **kw: calls.append(argv) or FakeProc(6000 + len(calls), None))
    out, events = train_tiny(tmp_path)
    kinds = [(e["event"], e["step"]) for e in events if e["event"].startswith("eval_")]
    assert kinds == [("eval_launched", 2), ("eval_skipped", 4), ("eval_skipped", 6)]
    assert len(calls) == 1 and events[-1]["event"] == "end"


@pytest.mark.parametrize("how", ["spawn_raises", "child_exits_nonzero"])
def test_a_failing_read_never_affects_the_trainer(tiny_pixart, tmp_path, monkeypatch, how):
    def spawn(argv, **kw):
        if how == "spawn_raises":
            raise OSError("the eval could not start")
        return FakeProc(7000, 2)                            # started, and already failed

    monkeypatch.setattr(pe, "_popen", spawn)
    out, events = train_tiny(tmp_path)
    kinds = [e["event"] for e in events if e["event"].startswith("eval_")]
    assert kinds == (["eval_skipped"] * 3 if how == "spawn_raises" else ["eval_launched"] * 3)
    assert events[-1]["event"] == "end" and events[-1]["step"] == 6
    assert os.path.isfile(os.path.join(out, "best.pt")) and os.path.isfile(os.path.join(out, "0000006.pt"))
