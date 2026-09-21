#!/bin/bash
# Picks the card up the moment the job in front gives it back, so no GPU-hour is spent waiting on a
# poll. Runs only the lever work that needs no decision in between: the step-count sweep (E1), then
# the fit checks that measure which micro-batch fills the card for the MSE decoder tune (E2). The
# tune itself is launched by hand afterwards with the batch those checks measured, because picking
# it automatically would mean guessing which OOM was the real ceiling.
#
#   usage: levers_queue.sh <gpu> [session-to-wait-for]
set -u
GPU=${1:?gpu}; WAIT=${2:-after-sd35-fix}
D=/sata2/data/rnagabhi/doom
REPO=${REPO:-$D/tmp/levers/repo}
LOG=$D/logs/035-sd35-l32-aligned_rollout.log

echo "$(date -u) waiting for tmux session $WAIT to exit"
while tmux has-session -t "$WAIT" 2>/dev/null; do sleep 60; done
echo "$(date -u) $WAIT gone, AFTER_SD35_DONE count now $(grep -c AFTER_SD35_DONE $LOG 2>/dev/null)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

REPO=$REPO bash $REPO/scripts/spiderman/steps_sweep.sh "$GPU"
for MB in 16 24 32 48 64; do
  FIT=1 REPO=$REPO bash $REPO/scripts/spiderman/decoder_mse.sh "$GPU" $MB 60 || echo "fit mb$MB failed"
done
echo LEVERS_QUEUE_DONE
