"""Native Weights & Biases logging (`wandb_log.py`): the trainer's and evaluators' live curves.

Everything here runs without the `wandb` package: a stub module is injected into `sys.modules`,
records every call, and never reaches the network. The series the native logger produces are
compared against `tools/wandb_tail.py` (the sidecar) on the same input, because a native run and a
sidecar-tailed run must plot on one set of panels.

    python -m pytest paper/fixtures/test_wandb_log.py -q
"""
import importlib.util
import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import wandb_log  # noqa: E402
from wandb_stub import FakeRun, inits, logged, stub_wandb  # noqa: E402

SIDECAR = os.path.join(REPO, "tools", "wandb_tail.py")


def load_sidecar():
    spec = importlib.util.spec_from_file_location("wandb_tail", SIDECAR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def wb(monkeypatch):
    mod = stub_wandb()
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return mod


T0 = 1_758_000_000.0


def sample_events():
    """Event lines shaped exactly like the ones train_wm.py's `log()` writes, `time` included."""
    return [
        {"event": "start", "backbone": "unet", "params": 865_000_000, "world": 1, "accum": 1,
         "latent_channels": 4, "per_gpu_batch": 32, "global_batch": 32, "objective": "v", "tic_stride": 1,
         "dataset_class": "TicWindowDataset", "phase_buckets": 0, "action_history": 32, "control_bits": 19,
         "dataset_summary": {"episodes": 2000, "windows": 9_000_000, "candidate_windows": 9_100_000},
         "init_from": None, "init_from_step": None, "train_fraction": 1.0, "train_episodes": 2000,
         "buffers_outside_ema": 0, "time": T0},
        {"event": "update_ratio", "step": 100, "x_embedder.proj.weight": 1e-4, "time": T0 + 60},
        {"event": "train", "step": 100, "loss": 0.31, "lr": 2.5e-6, "steps_per_s": 1.7, "peak_mem_gb": 38.2,
         "grad_norm": 0.9, "grad_norm_max": 2.1, "clip_frac": 0.12, "nonfinite_loss": 0, "skipped_updates": 0,
         "time": T0 + 61},
        {"event": "skipped_update", "step": 150, "micro": 151, "grad_norm": 9.0, "threshold": 5.0,
         "skipped_total": 1, "time": T0 + 90},
        {"event": "train", "step": 1000, "loss": 0.21, "lr": 2.5e-5, "steps_per_s": 1.8, "peak_mem_gb": 38.4,
         "grad_norm": 0.5, "grad_norm_max": 1.4, "clip_frac": 0.03, "nonfinite_loss": 1, "skipped_updates": 1,
         "time": T0 + 560},
        {"event": "val", "step": 1000, "val_loss": 0.24, "val_loss_by_t_quartile": [0.05, 0.18, 0.3, 0.43],
         "excursion": False, "time": T0 + 600},
        {"event": "train", "step": 2000, "loss": 0.2, "lr": 5e-5, "steps_per_s": 1.8, "peak_mem_gb": 38.4,
         "grad_norm": 0.4, "grad_norm_max": 1.0, "clip_frac": 0.0, "nonfinite_loss": 0, "skipped_updates": 1,
         "time": T0 + 1120},
        {"event": "val", "step": 2000, "val_loss": 0.3, "val_loss_by_t_quartile": [0.06, 0.2, 0.35, 0.6],
         "excursion": True, "time": T0 + 1160},
        {"event": "end", "step": 2000, "time": T0 + 1200},
    ]


EVAL_TF_METRICS = {
    "psnr_dec": {"mean": 24.1, "sem": 0.1, "n": 512},
    "psnr_raw": {"mean": 21.49, "sem": 0.12, "n": 512},
    "lpips_raw": {"mean": 0.264, "sem": 0.003, "n": 512},
    "persist_psnr_raw": {"mean": 21.57, "sem": 0.11, "n": 512},
    "persist_lpips_raw": {"mean": 0.203, "sem": 0.002, "n": 512},
    "per_map": {"1": {"psnr_dec": 24.0, "lpips_dec": 0.2}},
    "sampling_frames_per_s": 9.5,
    "config": {"ckpt": "snap_0010000.pt", "steps": 10},
}

PROBE_REPORT = {
    "step": 10000, "groups": {"control_mlp": 4, "context_conv": 1},
    "updates": {"control_mlp": {"relative_update": 3e-3, "zero_init_norm": None, "finite": True}},
    "probe": {"loss": 0.2, "grad_control_mlp": 0.01, "newest_control_sensitivity": 0.05},
    "problems": [], "ok": True,
}


# ---------------------------------------------------------------------------------------
# no-op unless enabled
# ---------------------------------------------------------------------------------------

def test_a_disabled_logger_never_imports_wandb(monkeypatch):
    boom = types.ModuleType("wandb")
    boom.init = lambda **kw: pytest.fail("a disabled logger opened a W&B run")
    monkeypatch.setitem(sys.modules, "wandb", boom)
    lg = wandb_log.RunLogger(enabled=False, name="040-unet-nexttic", results_dir="/nonexistent")
    assert not lg.active
    for e in sample_events():
        assert lg.log_event(e) is None
    assert lg.log_eval(10000, "live_h1", EVAL_TF_METRICS) is None
    lg.close()


def test_a_logger_without_a_run_name_stays_off(wb):
    lg = wandb_log.RunLogger(enabled=True, name="", results_dir="/nonexistent")
    assert not lg.active and wb.calls == []


# ---------------------------------------------------------------------------------------
# how the run is opened
# ---------------------------------------------------------------------------------------

def test_the_run_resumes_by_name_under_a_custom_step_axis(wb, tmp_path):
    lg = wandb_log.RunLogger(enabled=True, name="040-unet-nexttic", project="doomdit-nexttic", entity="me",
                             config={"lr": 5e-5, "git": "abc"}, results_dir=str(tmp_path))
    assert lg.active and lg.flush(5), "the run is opened on the logger's own thread"
    (kind, _, kw), *rest = wb.calls
    assert kind == "init"
    assert kw["project"] == "doomdit-nexttic" and kw["entity"] == "me"
    assert kw["id"] == "040-unet-nexttic" and kw["name"] == "040-unet-nexttic"
    assert kw["group"] == "040-unet-nexttic", "the trainer's run anchors the group its evaluations join"
    assert kw["resume"] == "allow"
    assert kw["config"] == {"lr": 5e-5, "git": "abc"}
    assert kw["dir"] == str(tmp_path / ".wandb") and os.path.isdir(kw["dir"])
    assert kw["settings"]["console"] == "off", "W&B must not wrap the trainer's stdout"
    assert kw["settings"]["finish_timeout"] == wandb_log.FINISH_TIMEOUT, "W&B's own finish must be bounded too"
    assert ("define_metric", ("step",), {}) in rest
    assert ("define_metric", ("*",), {"step_metric": "step"}) in rest


def test_rows_carry_their_step_and_never_pass_step_to_log(wb, tmp_path):
    """`run.log(step=...)` drops any row at a step below the last one; an evaluation read of an
    earlier checkpoint arrives after later training steps, so the step travels in the row."""
    lg = wandb_log.RunLogger(enabled=True, name="r", results_dir=str(tmp_path))
    for e in sample_events():
        lg.log_event(e)
    logs = [(row, k) for kind, row, k in wb.calls if kind == "log"]
    assert logs
    for row, k in logs:
        assert "step" not in k and isinstance(row["step"], int)


# ---------------------------------------------------------------------------------------
# the series are the sidecar's
# ---------------------------------------------------------------------------------------

def run_sidecar(run_dir, monkeypatch):
    """Tail `run_dir` once with tools/wandb_tail.py under a fresh stub; the rows it logged."""
    side = stub_wandb()
    monkeypatch.setitem(sys.modules, "wandb", side)
    monkeypatch.setattr(sys, "argv", ["wandb_tail.py", "--run-dir", str(run_dir), "--once"])
    load_sidecar().main()
    return logged(side)


def test_trainer_events_give_the_sidecars_series_and_values(tmp_path, monkeypatch):
    run_dir = tmp_path / "040-unet-nexttic"
    run_dir.mkdir()
    with open(run_dir / "log.jsonl", "w") as f:
        for e in sample_events():
            f.write(json.dumps(e) + "\n")
    expected = run_sidecar(run_dir, monkeypatch)
    assert len(expected) == 5, "three train and two val events"

    native = stub_wandb()
    monkeypatch.setitem(sys.modules, "wandb", native)
    lg = wandb_log.RunLogger(enabled=True, name="040-unet-nexttic", results_dir=str(tmp_path / "native"))
    for e in sample_events():
        lg.log_event(e)
    assert logged(native) == expected
    assert [c[0] for c in native.calls].count("finish") == 1, "the end event closes the run"


def test_every_listed_series_is_produced(wb, tmp_path):
    lg = wandb_log.RunLogger(enabled=True, name="r", results_dir=str(tmp_path))
    for e in sample_events():
        lg.log_event(e)
    keys = set().union(*logged(wb))
    for k in ("train/loss", "train/lr", "train/grad_norm", "train/grad_norm_max", "train/clip_frac",
              "train/steps_per_s", "train/peak_mem_gb", "train/nonfinite_loss", "train/skipped_updates",
              "val/loss", "val/loss_q1", "val/loss_q2", "val/loss_q3", "val/loss_q4", "val/excursion",
              "val/loss_minus_train", "time/wall_hours", "time/steps_per_hour", "data/examples_seen",
              "data/game_hours_seen", "data/epochs"):
        assert k in keys, k


def test_the_derived_series_are_right(wb, tmp_path):
    lg = wandb_log.RunLogger(enabled=True, name="r", results_dir=str(tmp_path))
    rows = [lg.log_event(e) for e in sample_events()]
    train_1000 = rows[4]
    assert train_1000["time/wall_hours"] == pytest.approx(560 / 3600)
    assert train_1000["time/steps_per_hour"] == pytest.approx(1.8 * 3600)
    assert train_1000["data/examples_seen"] == 32_000
    assert train_1000["data/game_hours_seen"] == pytest.approx(32_000 / 35 / 3600)
    assert train_1000["data/epochs"] == pytest.approx(32_000 / 9_000_000)
    val_1000 = rows[5]
    assert val_1000["val/loss_minus_train"] == pytest.approx(0.24 - 0.21)
    assert val_1000["val/excursion"] == 0 and rows[7]["val/excursion"] == 1
    assert rows[0] is None and rows[1] is None and rows[3] is None, "start, update_ratio, skipped_update add no row"


def test_evaluator_reads_give_the_sidecars_series(tmp_path, monkeypatch):
    """The sidecar reads the steward's JSONs from `steward_<step>/`; an evaluator logging natively
    hands the same dict to `log_eval` and must land on the same series."""
    run_dir = tmp_path / "run"
    (run_dir / "steward_10000" / "eval_tf_val_live_h1").mkdir(parents=True)
    (run_dir / "steward_10000" / "eval_tf_val_ema_h4").mkdir(parents=True)
    json.dump(EVAL_TF_METRICS, open(run_dir / "steward_10000" / "eval_tf_val_live_h1" / "metrics.json", "w"))
    json.dump(EVAL_TF_METRICS, open(run_dir / "steward_10000" / "eval_tf_val_ema_h4" / "metrics.json", "w"))
    json.dump(PROBE_REPORT, open(run_dir / "steward_10000" / "probe.json", "w"))
    expected = run_sidecar(run_dir, monkeypatch)
    assert len(expected) == 3

    native = stub_wandb()
    monkeypatch.setitem(sys.modules, "wandb", native)
    lg = wandb_log.RunLogger(enabled=True, name="run-eval", results_dir=str(tmp_path / "native"))
    for tag, m in (("ema_h4", EVAL_TF_METRICS), ("live_h1", EVAL_TF_METRICS), ("probe", PROBE_REPORT)):
        lg.log_eval(10000, tag, m)
    assert lg.close()
    key = lambda row: sorted(row)   # noqa: E731
    assert sorted(logged(native), key=key) == sorted(expected, key=key)
    live = [r for r in logged(native) if "eval/live_h1/psnr" in r][0]
    assert live["eval/live_h1/psnr_over_persistence"] == pytest.approx(21.49 - 21.57)
    assert live["eval/live_h1/lpips"] == 0.264 and live["eval/live_h1/persist_psnr"] == 21.57


def test_any_other_evaluator_is_flattened_generically(wb, tmp_path):
    lg = wandb_log.RunLogger(enabled=True, name="r-eval", results_dir=str(tmp_path))
    row = lg.log_eval(5000, "rollout", {"psnr@4": 20.1, "lpips@4": 0.3, "psnr": [1.0] * 64,
                                        "decoder": {"name": "sd-vae-ft-mse"}, "scored_at": [4, 32]})
    assert row == {"eval/rollout/psnr@4": 20.1, "eval/rollout/lpips@4": 0.3,
                   "eval/rollout/scored_at/0": 4, "eval/rollout/scored_at/1": 32}
    assert lg.close()
    assert logged(wb) == [{"step": 5000, **row}]


# ---------------------------------------------------------------------------------------
# a W&B failure never reaches the caller
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("fail", ["init", "log", "finish"])
def test_a_wandb_failure_disables_the_logger_and_is_reported_once(fail, tmp_path, monkeypatch, capsys):
    mod = stub_wandb(fail=fail)
    monkeypatch.setitem(sys.modules, "wandb", mod)
    lg = wandb_log.RunLogger(enabled=True, name="r", results_dir=str(tmp_path))
    for e in sample_events() + sample_events():
        lg.log_event(e)                            # must not raise
    lg.log_eval(10000, "live_h1", EVAL_TF_METRICS)
    lg.close()
    assert not lg.active
    err = [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]
    assert len(err) == 1 and "wandb" in err[0].lower(), err
    if fail == "log":
        assert [c[0] for c in mod.calls].count("log") == 0


def test_a_missing_wandb_package_is_one_line_and_a_no_op(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "wandb", None)      # `import wandb` raises ImportError
    lg = wandb_log.RunLogger(enabled=True, name="r", results_dir=str(tmp_path))
    assert not lg.active
    for e in sample_events():
        assert lg.log_event(e) is None
    err = [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]
    assert len(err) == 1 and "wandb" in err[0].lower(), err


def test_the_logger_closes_once_on_end_and_ignores_later_events(wb, tmp_path):
    lg = wandb_log.RunLogger(enabled=True, name="r", results_dir=str(tmp_path))
    for e in sample_events():
        lg.log_event(e)
    n = len(logged(wb))
    lg.log_event({"event": "train", "step": 3000, "loss": 0.1, "time": T0 + 2000})
    lg.close()
    assert len(logged(wb)) == n and [c[0] for c in wb.calls].count("finish") == 1


# ---------------------------------------------------------------------------------------
# W&B never blocks the caller: its own thread, a bounded queue, a bounded shutdown
# ---------------------------------------------------------------------------------------

def train_events(n, first=100):
    return [{"event": "train", "step": first + 100 * i, "loss": 0.3, "lr": 5e-5, "steps_per_s": 1.8,
             "time": T0 + i} for i in range(n)]


def timed(fn):
    import time
    t = time.monotonic()
    fn()
    return time.monotonic() - t


def test_a_hanging_init_never_blocks_the_caller(tmp_path, monkeypatch):
    import threading
    gate = threading.Event()
    mod = stub_wandb(gate={"init": gate})
    monkeypatch.setitem(sys.modules, "wandb", mod)
    lg = None

    def open_and_log():
        nonlocal lg
        lg = wandb_log.RunLogger(enabled=True, name="r", results_dir=str(tmp_path))
        for e in [sample_events()[0]] + train_events(50):
            lg.log_event(e)

    assert timed(open_and_log) < 0.5, "wandb.init ran on the caller's thread"
    assert logged(mod) == []
    gate.set()
    assert lg.close()
    assert [r["step"] for r in logged(mod)] == [100 + 100 * i for i in range(50)], "rows arrive in order"


def test_a_slow_log_never_blocks_the_caller(tmp_path, monkeypatch):
    mod = stub_wandb(delay={"log": 0.05})
    monkeypatch.setitem(sys.modules, "wandb", mod)
    lg = wandb_log.RunLogger(enabled=True, name="r", results_dir=str(tmp_path))
    assert timed(lambda: [lg.log_event(e) for e in train_events(40)]) < 0.5, "run.log ran on the caller's thread"
    assert lg.close() and len(logged(mod)) == 40


def test_a_full_queue_drops_the_oldest_rows_and_counts_them(tmp_path, monkeypatch, capsys):
    import threading
    gate = threading.Event()
    mod = stub_wandb(gate={"init": gate})
    monkeypatch.setitem(sys.modules, "wandb", mod)
    lg = wandb_log.RunLogger(enabled=True, name="r", results_dir=str(tmp_path), queue_rows=5)
    assert timed(lambda: [lg.log_event(e) for e in train_events(20)]) < 0.5
    gate.set()
    assert lg.close()
    assert [r["step"] for r in logged(mod)] == [1600, 1700, 1800, 1900, 2000], "the NEWEST rows survive"
    err = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("wandb:")]
    assert len(err) == 2 and "5 rows" in err[0] and "15" in err[1], err


def test_a_hanging_finish_is_abandoned_and_the_process_left_free_to_exit(tmp_path, monkeypatch, capsys):
    """The SDK's finish is unbounded by default and its atexit teardown waits for the service
    process. After the bounded wait the logger gives up, says so once, and unregisters that
    teardown, so the trainer's process can exit and release its GPU."""
    import threading
    mod = stub_wandb(gate={"finish": threading.Event()})
    monkeypatch.setitem(sys.modules, "wandb", mod)
    monkeypatch.setattr(wandb_log, "FINISH_TIMEOUT", 0.2)
    monkeypatch.setattr(wandb_log, "CLOSE_MARGIN", 0.1)
    lg = wandb_log.RunLogger(enabled=True, name="r", results_dir=str(tmp_path))
    for e in train_events(3):
        lg.log_event(e)
    assert timed(lambda: lg.log_event({"event": "end", "step": 300, "time": T0 + 9})) < 2.0
    assert not lg.active and mod.released, "the SDK's exit-time teardown is still registered"
    err = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("wandb:")]
    assert len(err) == 1 and "free to exit" in err[0], err
    (kw,) = inits(mod)
    assert kw["settings"]["finish_timeout"] == 0.2


def test_a_trainer_whose_wandb_hangs_trains_and_exits_on_time(tiny_pixart, tmp_path, monkeypatch, capsys):
    import threading
    mod = stub_wandb(gate={"init": threading.Event()})     # W&B never comes up
    monkeypatch.setitem(sys.modules, "wandb", mod)
    monkeypatch.setattr(wandb_log, "FINISH_TIMEOUT", 0.2)
    monkeypatch.setattr(wandb_log, "CLOSE_MARGIN", 0.1)
    out = train_tiny(tmp_path)
    events = jsonl(os.path.join(out, "log.jsonl"))
    assert events[-1]["event"] == "end" and os.path.isfile(os.path.join(out, "best.pt"))
    assert logged(mod) == [] and mod.released
    assert any("free to exit" in ln for ln in capsys.readouterr().err.splitlines())


# ---------------------------------------------------------------------------------------
# the checkpoint's step and the evaluation run
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("path,step", [("results/040/snap_0010000.pt", 10000), ("0010000.pt", 10000),
                                       ("/a/b/0000300.pt", 300), ("snap_0000000.pt", 0),
                                       ("best.pt", None), ("snap_final.pt", None), ("", None), (None, None)])
def test_the_step_is_read_from_the_checkpoint_filename(path, step):
    assert wandb_log.step_from_checkpoint(path) == step


def test_an_explicit_step_wins_and_the_recorded_step_is_the_fallback():
    assert wandb_log.resolve_eval_step(7, "snap_0010000.pt", 9) == 7
    assert wandb_log.resolve_eval_step(None, "snap_0010000.pt", 9) == 10000
    assert wandb_log.resolve_eval_step(None, "best.pt", 12000) == 12000
    assert wandb_log.resolve_eval_step(None, "best.pt", "?") is None


def test_evaluations_go_to_their_own_run_in_the_trainers_group():
    """W&B: "Unexpected results will occur if multiple processes use the same id concurrently", so
    an evaluator never writes to the live trainer's run; it writes `<run>-eval` in group `<run>`."""
    assert wandb_log.eval_run_names("040-unet-nexttic") == ("040-unet-nexttic-eval", "040-unet-nexttic")
    assert wandb_log.eval_run_names("040-unet-nexttic-eval") == ("040-unet-nexttic-eval", "040-unet-nexttic")


# ---------------------------------------------------------------------------------------
# train_wm.py: on by default, off under --no-wandb and in fit checks, fed by the one event writer
# ---------------------------------------------------------------------------------------

os.environ.setdefault("ACCELERATE_USE_CPU", "1")
TINY_PIXART = dict(num_attention_heads=2, attention_head_dim=8, in_channels=4, out_channels=8, num_layers=2,
                   caption_channels=32, sample_size=64, patch_size=2, cross_attention_dim=16,
                   use_additional_conditions=False, norm_num_groups=2)
BITS = 19


@pytest.fixture
def tiny_pixart(monkeypatch):
    """Serve a tiny transformer wherever PixArt would pull the 611M checkpoint, so the trainer runs on the CPU."""
    from diffusers import PixArtTransformer2DModel as P
    monkeypatch.setattr(P, "from_pretrained", classmethod(lambda cls, *a, **k: cls(**TINY_PIXART)))
    monkeypatch.setattr(P, "load_config", classmethod(lambda cls, *a, **k: dict(TINY_PIXART)))
    monkeypatch.setenv("ACCELERATE_USE_CPU", "1")


def train_tiny(tmp_path, *flags, run="040-tiny-nexttic"):
    """Four updates of a tiny next-tic PixArt on three synthetic per-tic episodes; the results dir."""
    import numpy as np
    import train_wm
    from pertic_fixtures import held_actions, write_pertic_episode
    d = str(tmp_path / "lat")
    rng = np.random.RandomState(0)
    for ep in range(3):
        btns = ["".join(str(int(b)) for b in rng.randint(0, 2, BITS)) for _ in range(40)]
        write_pertic_episode(d, ep, held_actions([0, 1, 2] * 4)[:40], buttons=btns)
    out = str(tmp_path / run)
    train_wm.main(train_wm.build_parser().parse_args(
        ["--backbone", "pixart", "--warm-start", "PixArt-alpha/PixArt-XL-2-512x512",
         "--latents-dir", d, "--results-dir", out, "--tic-stride", "1", "--action-history", "8",
         "--context-frames", "8", "--episode-ids", "0:2", "--val-episode-ids", "2:3",
         "--num-actions", "3", "--noise-buckets", "4", "--per-gpu-batch", "2", "--global-batch", "2",
         "--steps", "4", "--warmup", "1", "--log-every", "1", "--val-every", "2", "--val-windows", "2",
         "--ckpt-every", "4", "--ema-every", "1", "--num-workers", "0", "--action-dropout", "0.0", *flags]))
    return out


def jsonl(path):
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def test_the_trainer_flags_default_to_streaming():
    import train_wm
    a = train_wm.build_parser().parse_args(["--backbone", "dit"])
    assert a.wandb is True and a.wandb_project == "doomdit-nexttic" and a.wandb_entity is None
    assert train_wm.build_parser().parse_args(["--backbone", "dit", "--no-wandb"]).wandb is False


def test_the_trainer_streams_every_event_after_writing_it(tiny_pixart, tmp_path, monkeypatch):
    mod = stub_wandb()
    out = str(tmp_path / "040-tiny-nexttic")
    seen_on_disk = []

    class OrderedRun(FakeRun):
        def log(self, row, **k):
            # the line for this event must already be in log.jsonl when W&B is handed it
            kind = "val" if "val/loss" in row else "train"
            on_disk = {(e["event"], e.get("step")) for e in jsonl(os.path.join(out, "log.jsonl"))}
            seen_on_disk.append((kind, row["step"]) in on_disk)
            super().log(row, **k)

    real_init = mod.init
    mod.init = lambda **kw: (real_init(**kw), OrderedRun(mod.calls))[1]
    monkeypatch.setitem(sys.modules, "wandb", mod)
    train_tiny(tmp_path)

    (kind, _, kw), = [c for c in mod.calls if c[0] == "init"]
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert kw["id"] == kw["name"] == kw["group"] == "040-tiny-nexttic"
    assert kw["resume"] == "allow" and kw["dir"] == os.path.join(out, ".wandb")
    assert kw["config"] == cfg and cfg["git"], "the W&B config is config.json, git hash included"
    assert cfg["wandb"] is True and cfg["wandb_run"] == "040-tiny-nexttic"
    events = jsonl(os.path.join(out, "log.jsonl"))
    want = [(e["event"], e["step"]) for e in events if e["event"] in ("train", "val")]
    got = [("val" if "val/loss" in r else "train", r["step"]) for r in logged(mod)]
    assert got == want and len(want) == 6, "four train and two val events"
    assert seen_on_disk and all(seen_on_disk), "a row reached W&B before its line was in log.jsonl"
    assert [c[0] for c in mod.calls].count("finish") == 1, "the end event closes the run"


def test_no_wandb_opts_out_and_says_so_in_config(tiny_pixart, tmp_path, monkeypatch):
    boom = types.ModuleType("wandb")
    boom.init = lambda **kw: pytest.fail("--no-wandb opened a W&B run")
    monkeypatch.setitem(sys.modules, "wandb", boom)
    out = train_tiny(tmp_path, "--no-wandb")
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["wandb"] is False and cfg["wandb_run"] is None
    assert not os.path.exists(os.path.join(out, ".wandb"))


def test_a_fit_check_never_logs(tiny_pixart, tmp_path, monkeypatch):
    import train_wm
    boom = types.ModuleType("wandb")
    boom.init = lambda **kw: pytest.fail("a fit check opened a W&B run")
    monkeypatch.setitem(sys.modules, "wandb", boom)
    out = str(tmp_path / "fit")
    train_wm.main(train_wm.build_parser().parse_args(
        ["--backbone", "pixart", "--warm-start", "PixArt-alpha/PixArt-XL-2-512x512", "--results-dir", out,
         "--fit-check", "2", "--context-frames", "2", "--num-actions", "3", "--noise-buckets", "4",
         "--per-gpu-batch", "2", "--global-batch", "2", "--num-workers", "0", "--action-dropout", "0.0"]))
    assert any(e["event"] == "fit_check" for e in jsonl(os.path.join(out, "log.jsonl")))
    assert json.load(open(os.path.join(out, "config.json")))["wandb_run"] is None


def test_a_wandb_outage_mid_run_does_not_stop_training(tiny_pixart, tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "wandb", stub_wandb(fail="log"))
    out = train_tiny(tmp_path)
    events = jsonl(os.path.join(out, "log.jsonl"))
    assert events[-1]["event"] == "end" and events[-1]["step"] == 4
    assert os.path.isfile(os.path.join(out, "best.pt"))
    err = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("wandb:")]
    assert len(err) == 1, err


def test_a_real_training_run_streams_exactly_what_the_sidecar_would_replay(tiny_pixart, tmp_path, monkeypatch):
    """End to end: the rows the trainer streamed natively equal, series for series and value for
    value, the rows the sidecar logs when it replays that same run's log.jsonl afterwards."""
    native = stub_wandb()
    monkeypatch.setitem(sys.modules, "wandb", native)
    out = train_tiny(tmp_path)
    expected = run_sidecar(out, monkeypatch)
    assert len(expected) == 6, "four train and two val events"
    assert logged(native) == expected


# ---------------------------------------------------------------------------------------
# the evaluators: --wandb-run appends a summary to `<run>-eval` at the checkpoint's step
# ---------------------------------------------------------------------------------------

def evaluator_parsers():
    import eval_tf
    import rollout_eval
    import smoke_probe
    return {
        "eval_tf": (eval_tf.build_parser(), ["--ckpt", "c.pt", "--backbone", "unet", "--latents-dir", "l",
                                             "--split", "s.json", "--out-dir", "o"]),
        "rollout_eval": (rollout_eval.build_parser(), ["--score", "--rollouts", "r.npz", "--out-dir", "o"]),
        "smoke_probe": (smoke_probe.build_parser(), ["--ckpt", "c.pt", "--backbone", "unet", "--latents-dir", "l",
                                                     "--episodes", "6000:6100"]),
    }


@pytest.mark.parametrize("name", ["eval_tf", "rollout_eval", "smoke_probe"])
def test_every_evaluator_takes_the_wandb_flags_and_is_off_by_default(name):
    p, base = evaluator_parsers()[name]
    a = p.parse_args(base)
    assert a.wandb_run == "" and a.wandb_project == "doomdit-nexttic" and a.wandb_entity is None
    assert a.wandb_step is None
    a = p.parse_args(base + ["--wandb-run", "040-unet-nexttic", "--wandb-project", "p", "--wandb-entity", "e",
                             "--wandb-step", "12000"])
    assert (a.wandb_run, a.wandb_project, a.wandb_entity, a.wandb_step) == ("040-unet-nexttic", "p", "e", 12000)


def eval_args(**kw):
    return types.SimpleNamespace(**{"wandb_run": "040-unet-nexttic", "wandb_project": "doomdit-nexttic",
                                    "wandb_entity": None, "wandb_step": None, **kw})


def test_an_evaluation_is_logged_to_the_eval_run_at_the_checkpoints_step(wb, tmp_path):
    row = wandb_log.log_evaluation(eval_args(), "live_h1", EVAL_TF_METRICS, ckpt="r/snap_0010000.pt",
                                   recorded_step=9999, out_dir=str(tmp_path))
    (kw,) = inits(wb)
    assert kw["id"] == kw["name"] == "040-unet-nexttic-eval" and kw["group"] == "040-unet-nexttic"
    assert kw["resume"] == "allow" and kw["dir"] == str(tmp_path / ".wandb")
    assert logged(wb) == [{"step": 10000, **row}] and "eval/live_h1/psnr" in row
    assert [c[0] for c in wb.calls].count("finish") == 1


def test_the_explicit_step_and_the_recorded_step(wb, tmp_path):
    wandb_log.log_evaluation(eval_args(wandb_step=4000), "probe", PROBE_REPORT, ckpt="r/0010000.pt",
                             out_dir=str(tmp_path))
    wandb_log.log_evaluation(eval_args(), "probe", PROBE_REPORT, ckpt="r/best.pt", recorded_step=12000,
                             out_dir=str(tmp_path))
    assert [r["step"] for r in logged(wb)] == [4000, 12000]


def test_an_unknown_step_logs_nothing_and_says_so(wb, tmp_path, capsys):
    assert wandb_log.log_evaluation(eval_args(), "probe", PROBE_REPORT, ckpt="best.pt", recorded_step="?",
                                    out_dir=str(tmp_path)) is None
    assert inits(wb) == []
    assert "--wandb-step" in capsys.readouterr().err


def test_without_wandb_run_nothing_is_imported(monkeypatch, tmp_path):
    boom = types.ModuleType("wandb")
    boom.init = lambda **kw: pytest.fail("an evaluator without --wandb-run opened a W&B run")
    monkeypatch.setitem(sys.modules, "wandb", boom)
    assert wandb_log.log_evaluation(eval_args(wandb_run=""), "live_h1", EVAL_TF_METRICS,
                                    ckpt="snap_0010000.pt", out_dir=str(tmp_path)) is None


def test_an_evaluation_survives_a_wandb_outage(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "wandb", stub_wandb(fail="init"))
    # the checkpoint path must lie under tmp_path: its directory is where the writer lock goes
    assert wandb_log.log_evaluation(eval_args(), "live_h1", EVAL_TF_METRICS,
                                    ckpt=str(tmp_path / "snap_0010000.pt"), out_dir=str(tmp_path)) is None
    assert len([ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("wandb:")]) == 1


@pytest.mark.parametrize("use_ema,horizon,tag", [(False, 1, "live_h1"), (True, 1, "ema_h1"),
                                                 (False, 4, "live_h4"), (True, 4, "ema_h4")])
def test_eval_tf_tags_its_read_by_weights_and_horizon(use_ema, horizon, tag):
    import eval_tf
    assert eval_tf.wandb_tag(types.SimpleNamespace(use_ema=use_ema, horizon_tics=horizon)) == tag


# ---------------------------------------------------------------------------------------
# one writer per evaluation run: a lock per (project, run id) in the training run's directory
# ---------------------------------------------------------------------------------------

def lock_of(run_dir, run_id="040-unet-nexttic-eval", project="doomdit-nexttic"):
    return wandb_log.WriterLock(wandb_log.lock_path(str(run_dir), project, run_id))


def test_the_periodic_read_and_a_standalone_evaluator_share_one_lock(tmp_path):
    r = str(tmp_path / "040-unet-nexttic")
    os.makedirs(f"{r}/eval_0005000")
    assert wandb_log.run_dir_of(f"{r}/eval_0005000/0005000.pt") == r, "the periodic read's own link"
    assert wandb_log.run_dir_of(f"{r}/snap_0010000.pt") == r, "a steward's read of the run's snapshot"
    assert wandb_log.run_dir_of(None, f"{r}/rollout_metrics") == f"{r}/rollout_metrics"
    # a rollout scored on another machine records a checkpoint path that does not exist here
    assert wandb_log.run_dir_of("/elsewhere/run/snap_0010000.pt", f"{r}/rollout_metrics") == f"{r}/rollout_metrics"
    assert wandb_log.lock_path(r, "p", "a-eval") != wandb_log.lock_path(r, "q", "a-eval")


def test_the_lock_is_held_for_the_whole_session(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    mod = stub_wandb()
    held = []
    real_init = mod.init

    def init(**kw):
        other = lock_of(run_dir)
        held.append(not other.acquire(0))      # someone (the evaluator) holds it while W&B is open
        other.release()
        return real_init(**kw)

    mod.init = init
    monkeypatch.setitem(sys.modules, "wandb", mod)
    assert wandb_log.log_evaluation(eval_args(), "probe", PROBE_REPORT, ckpt=str(run_dir / "0010000.pt"),
                                    out_dir=str(tmp_path / "out"))
    assert held == [True]
    after = lock_of(run_dir)
    assert after.acquire(0), "the lock must be released once the session has finished"
    after.release()


def test_a_second_writer_waits_then_takes_its_own_id_in_the_same_group(wb, tmp_path, monkeypatch, capsys):
    import time
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    first = lock_of(run_dir)
    assert first.acquire(0)                                   # the periodic read is writing
    monkeypatch.setattr(wandb_log, "LOCK_WAIT", 0.3)
    t = time.monotonic()
    wandb_log.log_evaluation(eval_args(), "probe", PROBE_REPORT, ckpt=str(run_dir / "snap_0010000.pt"),
                             out_dir=str(tmp_path / "out"))
    assert time.monotonic() - t >= 0.3, "it must wait before giving up on the shared id"
    first.release()
    (kw,) = inits(wb)
    own = f"040-unet-nexttic-eval-{os.getpid()}"
    assert kw["id"] == kw["name"] == own and kw["group"] == "040-unet-nexttic"
    assert own in capsys.readouterr().err


def test_a_writer_that_gets_the_lock_in_time_uses_the_shared_id(wb, tmp_path, monkeypatch):
    import threading
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    first = lock_of(run_dir)
    assert first.acquire(0)
    threading.Timer(0.3, first.release).start()
    monkeypatch.setattr(wandb_log, "LOCK_WAIT", 10.0)
    wandb_log.log_evaluation(eval_args(), "probe", PROBE_REPORT, ckpt=str(run_dir / "snap_0010000.pt"),
                             out_dir=str(tmp_path / "out"))
    (kw,) = inits(wb)
    assert kw["id"] == "040-unet-nexttic-eval"


def test_a_session_that_does_not_finish_keeps_the_lock_until_the_process_exits(tmp_path, monkeypatch):
    """Releasing the lock while W&B is still finishing would let a second writer open the same id."""
    import threading
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setitem(sys.modules, "wandb", stub_wandb(gate={"finish": threading.Event()}))
    monkeypatch.setattr(wandb_log, "FINISH_TIMEOUT", 0.1)
    monkeypatch.setattr(wandb_log, "CLOSE_MARGIN", 0.1)
    wandb_log.log_evaluation(eval_args(), "probe", PROBE_REPORT, ckpt=str(run_dir / "snap_0010000.pt"),
                             out_dir=str(tmp_path / "out"))
    other = lock_of(run_dir)
    assert not other.acquire(0)
