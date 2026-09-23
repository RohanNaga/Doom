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
each `steward_<step>/eval_tf_val_<live|ema>_h<H>/metrics.json` the steward writes (raw PSNR and
LPIPS with the persistence floor beside them, and PSNR minus persistence), and every other JSON under
`steward_<step>/` (probe, rollouts, sweeps) flattened generically. `--once` processes what is there and exits;
otherwise it polls every `--interval` seconds until the run's `end` event has been logged.
"""
import argparse
import glob
import json
import os
import re
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


def train_row(e, t0=None, global_batch=32, windows=None, tics_per_s=35.0):
    """The trainer's own numbers plus the derived time and data series Rohan asked for.

    `time/wall_hours` is the wall clock since the run's start event, `time/steps_per_hour` the
    smoothed rate the trainer reports, `data/examples_seen` step x global batch, `data/game_hours_seen`
    those examples in game time at 35 tics/s, and `data/epochs` examples over the corpus's window
    count when the start event recorded it. All share `step` as the x axis, so W&B plots any of
    them against any other (loss against wall clock, PSNR against game hours seen).
    """
    row = {f"train/{k}": e[k] for k in TRAIN_KEYS if isinstance(e.get(k), (int, float))}
    step = e.get("step")
    if t0 is not None and isinstance(e.get("time"), (int, float)):
        row["time/wall_hours"] = (e["time"] - t0) / 3600.0
    if isinstance(e.get("steps_per_s"), (int, float)):
        row["time/steps_per_hour"] = e["steps_per_s"] * 3600.0
    if isinstance(step, int):
        seen = step * global_batch
        row["data/examples_seen"] = seen
        row["data/game_hours_seen"] = seen / tics_per_s / 3600.0
        if windows:
            row["data/epochs"] = seen / windows
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


def flatten(prefix, obj, out, depth=0):
    """Every numeric leaf of a JSON object as `prefix/key`; dicts with a `mean` collapse to it."""
    if isinstance(obj, dict) and isinstance(obj.get("mean"), (int, float)):
        out[prefix] = obj["mean"]
        return
    if isinstance(obj, dict) and depth < 3:
        for k, v in obj.items():
            flatten(f"{prefix}/{k}", v, out, depth + 1)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = obj
    elif isinstance(obj, list) and obj and all(isinstance(v, (int, float)) for v in obj) and len(obj) <= 16:
        for i, v in enumerate(obj):
            out[f"{prefix}/{i}"] = v


def steward_rows(run_dir, seen):
    """(step, row, path) for every JSON the steward wrote under `steward_<step>/` and not yet sent.

    `eval_tf_val_<live|ema>_h<H>/metrics.json` becomes `eval/<tag>/<metric>` (raw PSNR and LPIPS with
    the persistence and reconstruction references beside them); every other JSON (`probe.json`,
    rollout metrics, sweeps) is flattened generically under `eval/<relative path>/...`, so a new
    steward artifact is plotted without a code change.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(run_dir, "steward_*", "**", "*.json"), recursive=True)):
        if path in seen:
            continue
        rel = os.path.relpath(path, run_dir).split(os.sep)
        try:
            step = int(rel[0].split("_")[1])
        except (IndexError, ValueError):
            continue
        try:
            with open(path) as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        row = {}
        m_tag = re.match(r"^(?:eval_tf_val_|tf_)((?:live|ema)_h\d+)$", rel[1]) if len(rel) == 3 else None
        if m_tag and rel[2] == "metrics.json":
            tag = m_tag.group(1)                          # live_h1, ema_h4, ...
            for src, dst in (("psnr_raw", "psnr"), ("lpips_raw", "lpips"), ("persist_psnr_raw", "persist_psnr"),
                             ("persist_lpips_raw", "persist_lpips"), ("recon_psnr_raw", "recon_psnr"),
                             ("recon_lpips_raw", "recon_lpips")):
                v = mean_of(m, src)
                if v is not None:
                    row[f"eval/{tag}/{dst}"] = v
            if f"eval/{tag}/psnr" in row:
                pf = row.get(f"eval/{tag}/persist_psnr")
                if pf is not None:
                    row[f"eval/{tag}/psnr_over_persistence"] = row[f"eval/{tag}/psnr"] - pf
        else:
            flatten("eval/" + "/".join(rel[1:]).replace(".json", ""), m, row)
        if row:
            out.append((step, row, path))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--project", default="doomdit-nexttic")
    p.add_argument("--entity", default=None)
    p.add_argument("--name", default=None, help="W&B run name and id (default: the run directory's name)")
    p.add_argument("--id", default=None, help="W&B run id when it must differ from the name (a deleted id cannot be reused)")
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
    run = wandb.init(project=a.project, entity=a.entity, id=a.id or name, name=name, resume="allow", config=config,
                     dir=os.path.join(a.run_dir, ".wandb"), settings=wandb.Settings(_disable_stats=True))
    run.define_metric("step")
    run.define_metric("*", step_metric="step")

    state = {"offset": 0, "last_step": -1, "steward_seen": []}
    if os.path.isfile(state_path):
        with open(state_path) as f:
            state.update(json.load(f))
    seen = set(state["steward_seen"])
    t0, gb, windows, last_train_loss = state.get("t0"), state.get("gb") or 32, state.get("windows"), None
    ended = False
    while True:
        if os.path.isfile(log_path):
            events, state["offset"] = read_events(log_path, state["offset"])
            for e in events:
                kind, step = e.get("event"), e.get("step")
                if kind == "start":
                    t0 = e.get("time") if isinstance(e.get("time"), (int, float)) else t0
                    gb = e.get("global_batch") if isinstance(e.get("global_batch"), int) else gb
                    ds = e.get("dataset_summary") or {}
                    windows = ds.get("windows") if isinstance(ds.get("windows"), int) else windows
                    state.update({"t0": t0, "gb": gb, "windows": windows})
                    continue
                if not isinstance(step, int):
                    continue
                row = train_row(e, t0, gb, windows) if kind == "train" else val_row(e) if kind == "val" else {}
                if kind == "val" and isinstance(e.get("val_loss"), (int, float)) and last_train_loss is not None:
                    row["val/loss_minus_train"] = e["val_loss"] - last_train_loss
                if kind == "train" and isinstance(e.get("loss"), (int, float)):
                    last_train_loss = e["loss"]
                if kind == "end":
                    ended = True
                if row and step > state["last_step"] - 1:
                    run.log({"step": step, **row})
                    state["last_step"] = max(state["last_step"], step)
        for step, row, path in steward_rows(a.run_dir, seen):
            run.log({"step": step, **row})
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
