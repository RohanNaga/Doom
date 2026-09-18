#!/bin/bash
# Run knob-grid cells one after another on one GPU, inside tmux session grid-<gpu>.
# A cell whose log already carries CELL_DONE is skipped, so re-issuing the same command after a
# crash, a reboot or a partial night picks up where it stopped.
#
#   grid_queue.sh <gpu> <cell> [<cell> ...]
#   grid_queue.sh 2 base30k data-1_2 scratch
#
# Cell names are the case table at the top of launch_cell.sh. Attach with `tmux attach -t grid-2`,
# follow with `tail -f /sata2/data/rnagabhi/doom/logs/grid_queue_2.log`.
# Pull the repo before starting a queue, not during one: every cell of a grid has to run the same code.
set -u
GPU=${1:?gpu}; shift
CELLS="$*"
[ -n "$CELLS" ] || { echo "usage: grid_queue.sh <gpu> <cell> [<cell> ...]" >&2; exit 2; }
D=/sata2/data/rnagabhi/doom
HERE=$(cd "$(dirname "$0")" && pwd)
QLOG=$D/logs/grid_queue_${GPU}.log
mkdir -p "$D/logs" "$D/tmp/tmpdir"

if [ -z "${GRID_IN_TMUX:-}" ]; then
  tmux has-session -t "grid-$GPU" 2>/dev/null && { echo "grid-$GPU is already running; nothing queued"; exit 0; }
  # TMPDIR goes inside the tmux command string: tmux starts commands in the server's environment,
  # so an exported TMPDIR never reaches the training process (the Sep 16 ENOSPC stall)
  tmux new-session -d -s "grid-$GPU" \
    "GRID_IN_TMUX=1 TMPDIR=$D/tmp/tmpdir GRID_FIT_CHECK=${GRID_FIT_CHECK:-} GRID_EMA_ROLLOUTS=${GRID_EMA_ROLLOUTS:-} PY=${PY:-} bash $HERE/grid_queue.sh $GPU $CELLS >> $QLOG 2>&1"
  echo "queued on gpu $GPU in tmux grid-$GPU: $CELLS"
  echo "log: $QLOG"
  exit 0
fi

echo "$(date -Iseconds) queue start gpu $GPU git $(git -C "$D/repo" rev-parse --short HEAD) cells: $CELLS"
for CELL in $CELLS; do
  if grep -q "CELL_DONE $CELL" "$D/logs/grid_${CELL}.log" 2>/dev/null; then
    echo "$(date -Iseconds) skip $CELL (CELL_DONE present)"
    continue
  fi
  echo "$(date -Iseconds) --- $CELL ---"
  bash "$HERE/launch_cell.sh" "$GPU" "$CELL"
  RC=$?
  echo "$(date -Iseconds) $CELL finished with exit $RC"
done
echo "GRID_QUEUE_DONE gpu $GPU"
