"""Stream a training run's `log.jsonl` (and the steward's evaluation reads) to Weights & Biases.

A sidecar, not a trainer change: the launch certificate pins the trainer's code, so a live run cannot
gain a `--wandb` flag mid-run. This process tails the run directory instead and re-logs every event
at its training step, so the W&B run is a faithful copy of `log.jsonl` and survives restarts of
either process (`wandb.init(id=..., resume="allow")`; the last logged step is kept in W&B's summary
and in a local state file, and only newer steps are sent).

    python tools/wandb_tail.py --run-dir $D/results_spiderman/040-unet-nexttic \\
        --project doomdit-nexttic --name 040-unet-nexttic [--once]

Logged at `step`: every `train` event's loss, lr, grad_norm, grad_norm_max, clip_frac, steps_per_s,
peak_mem_gb, nonfinite_loss, skipped_updates; every `val` event's val_loss and its four t-quartiles;
and each `steward_<step>/eval_tf_val_<live|ema>_h<H>/metrics.json` the steward writes (raw PSNR and
LPIPS with the persistence floor beside them). `--once` processes what is there and exits;
otherwise it polls every `--interval` seconds until the run's `end` event has been logged.
"""
import argparse
import glob
import json
import os
import time


def read_events(path, start_offset):
    """New complete lines of a jsonl file from a byte offset: (events, new_offset)."""
    events = []
    with open(path, "rb") as f:
        f.seek(start_offset)
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.endswith(b"\n"):        # a partial line still being written
                f.seek(pos)
                break
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return events, f.tell()


TRAIN_KEYS = ("loss", "lr", "grad_norm", "grad_norm_max", "clip_frac", "steps_per_s", "peak_mem_gb",
              "nonfinite_loss", "skipped_updates", "data_wait_frac")


def train_row(e):
    row = {f"train/{k}": e[k] for k in TRAIN_KEYS if isinstance(e.get(k), (int, float))}
    return row


def val_row(e):
    row = {}
    if isinstance(e.get("val_loss"), (int, float)):
        row["val/loss"] = e["val_loss"]
    q = e.get("val_loss_by_t_quartile")
    if isinstance(q, list):
        for i, v in enumerate(q):
            if isinstance(v, (int, float)):
                row[f"val/loss_q{i + 1}"] = v
    if "excursion" in e:
        row["val/excursion"] = int(bool(e["excursion"]))
    return row


def mean_of(m, key):
    v = m.get(key)
    if isinstance(v, dict):
        v = v.get("mean")
    return v if isinstance(v, (int, float)) else None


def steward_rows(run_dir, seen):
    """(step, row) for every steward metrics.json not yet sent."""
    out = []
    for path in sorted(glob.glob(os.path.join(run_dir, "steward_*", "eval_tf_val_*", "metrics.json"))):
        if path in seen:
            continue
        parts = path.split(os.sep)
        try:
            step = int(parts[-3].split("_")[1])
            tag = parts[-2].replace("eval_tf_val_", "")      # live_h1, ema_h4, ...
        except (IndexError, ValueError):
            continue
        with open(path) as f:
            m = json.load(f)
        row = {}
        for src, dst in (("psnr_raw", "psnr"), ("lpips_raw", "lpips"), ("persist_psnr_raw", "persist_psnr"),
                         ("persist_lpips_raw", "persist_lpips"), ("recon_psnr_raw", "recon_psnr")):
            v = mean_of(m, src)
            if v is not None:
                row[f"eval/{tag}/{dst}"] = v
        if row:
            out.append((step, row, path))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--project", default="doomdit-nexttic")
    p.add_argument("--entity", default=None)
    p.add_argument("--name", default=None, help="W&B run name and id (default: the run directory's name)")
    p.add_argument("--interval", type=float, default=30.0)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()

    import wandb

    name = a.name or os.path.basename(os.path.normpath(a.run_dir))
    log_path = os.path.join(a.run_dir, "log.jsonl")
    state_path = os.path.join(a.run_dir, ".wandb_tail_state.json")
    config = {}
    cfg_path = os.path.join(a.run_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            config = json.load(f)
    run = wandb.init(project=a.project, entity=a.entity, id=name, name=name, resume="allow", config=config,
                     dir=os.path.join(a.run_dir, ".wandb"), settings=wandb.Settings(_disable_stats=True))
    run.define_metric("*", step_metric="step")

    state = {"offset": 0, "last_step": -1, "steward_seen": []}
    if os.path.isfile(state_path):
        with open(state_path) as f:
            state.update(json.load(f))
    seen = set(state["steward_seen"])
    ended = False
    while True:
        if os.path.isfile(log_path):
            events, state["offset"] = read_events(log_path, state["offset"])
            for e in events:
                kind, step = e.get("event"), e.get("step")
                if not isinstance(step, int):
                    continue
                row = train_row(e) if kind == "train" else val_row(e) if kind == "val" else {}
                if kind == "end":
                    ended = True
                if row and step > state["last_step"] - 1:
                    run.log({"step": step, **row}, step=step)
                    state["last_step"] = max(state["last_step"], step)
        for step, row, path in steward_rows(a.run_dir, seen):
            run.log({"step": step, **row}, step=step)
            seen.add(path)
        state["steward_seen"] = sorted(seen)
        tmp = state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, state_path)
        if a.once or ended:
            break
        time.sleep(a.interval)
    run.finish()


if __name__ == "__main__":
    main()
