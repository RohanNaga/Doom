#!/bin/bash
# One screen of truth about the two rows: step, rate, loss, last validation, disk, cards.
#
#   usage: [DOOM_ROOT=..] [RUNS="040-unet-nexttic 042-sd35-nexttic"] [DRY=1] status.sh
#
# Everything comes from each run's own `log.jsonl`, which the trainer appends to: `event=train`
# every `--log-every` updates (step, loss, lr, steps_per_s, peak_mem_gb, grad_norm,
# skipped_updates) and `event=val` every `--val-every` (step, val_loss). `steps_per_s` there is an
# average since the loop started, including validation and checkpoint time, so it is the number to
# plan a budget with, not the instantaneous rate.
#
# Read-only. It starts nothing, stops nothing and writes nothing.
set -u
D=${DOOM_ROOT:-/data/doom}
DRY=${DRY:-0}
RUNS=${RUNS:-"040-unet-nexttic 042-sd35-nexttic"}
PY=${PY:-$D/env/bin/python}
[ -x "$PY" ] || PY=$(command -v python3 || true)

if [ "$DRY" = 1 ]; then
  for R in $RUNS; do echo "DRY status $R $D/results_spiderman/$R/log.jsonl"; done
  echo "DRY status disk df -h $D"
  echo "DRY status gpu nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader"
  exit 0
fi

[ -n "$PY" ] || { echo "no python to parse the run logs with" >&2; exit 1; }

for R in $RUNS; do
  LOG=$D/results_spiderman/$R/log.jsonl
  # train-<run> without its numeric prefix, the session launch_nexttic.sh gives the run
  if [[ $R =~ ^[0-9]+-(.+)$ ]]; then SESSION=train-${BASH_REMATCH[1]}; else SESSION=train-$R; fi
  ALIVE=no
  command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION" 2>/dev/null && ALIVE=yes
  if [ ! -f "$LOG" ]; then
    printf '%-18s not started (no %s)\n' "$R" "$LOG"
    continue
  fi
  "$PY" - "$R" "$LOG" "$ALIVE" <<'PY'
import json, os, sys, time
run, log, alive = sys.argv[1], sys.argv[2], sys.argv[3]
train = val = None
for line in open(log):
    try:
        r = json.loads(line)
    except ValueError:                      # a torn last line while the trainer is writing
        continue
    if r.get("event") == "train":
        train = r
    elif r.get("event") == "val":
        val = r
age = int(time.time() - os.path.getmtime(log))
bits = [f"{run:<18}"]
if train:
    bits.append(f"step={train['step']}")
    bits.append(f"updates_per_s={train.get('steps_per_s', 0):.3f}")
    bits.append(f"loss={train.get('loss')}")
    bits.append(f"lr={train.get('lr')}")
    bits.append(f"peak_mem_gb={train.get('peak_mem_gb')}")
    bits.append(f"skipped={train.get('skipped_updates')}")
else:
    bits.append("no training line yet")
bits.append(f"val={val['val_loss']}@{val['step']}" if val else "val=none")
bits.append(f"log_age={age}s")
bits.append(f"alive={alive}")
print(" ".join(bits))
PY
  SZ=$(du -sh "$(dirname "$LOG")" 2>/dev/null | cut -f1)
  printf '%-18s results %s\n' "" "${SZ:-?}"
done

echo "disk:"
df -h "$D" 2>/dev/null | tail -2
echo "gpu:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
else
  echo "  no nvidia-smi on this host"
fi
