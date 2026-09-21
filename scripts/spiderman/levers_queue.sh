#!/bin/bash
# Runs the two lever experiments end to end on one card, inside a hard deadline, without needing
# anyone to be watching. GPU 3 has to be handed back at 07:30 EDT on Sep 21 2026, the card is
# shared with a docker job from another project, and the six per-tic encode shards keep the box
# loaded enough that ssh itself times out, so every choice in here is made from a measurement on
# the machine rather than from a number typed in from outside.
#
#   1. wait for the job in front to release the card (never past the hard stop)
#   2. E1, the DDIM step sweep, which self-guards on its own deadline
#   3. the fit ladder for E2, sized to the memory actually free next to the other job
#   4. pick the micro-batch and the update budget from those fits (tools/pick_fit.py)
#   5. the MSE-only decoder tune until TUNE_END, then the ceiling measurement of every checkpoint
#
#   usage: levers_queue.sh <gpu> <wait-session> <tune-end-utc> <hard-stop-utc>
#     levers_queue.sh 3 after-sd35-fix 2026-09-21T10:30 2026-09-21T11:30
#
# The two stamps are UTC because the server clock is UTC; 10:30 and 11:30 UTC are 06:30 and 07:30
# EDT. Leaving an hour between them is deliberate: if time runs short the presentation budget
# shrinks and the measurement does not.
set -u
GPU=${1:?gpu}; WAIT=${2:?session to wait for}; TUNE_END=${3:?tune end, UTC}; HARD=${4:?hard stop, UTC}
D=/sata2/data/rnagabhi/doom
REPO=${REPO:-$D/tmp/levers/repo}
PY=${PY:-$HOME/miniconda3/envs/doom/bin/python}
T_TUNE=$(date -u -d "$TUNE_END" +%s); T_HARD=$(date -u -d "$HARD" +%s)
say() { echo "$(date -u +%H:%M:%S) $*"; }
left() { echo $(( T_HARD - $(date +%s) )); }

say "tune ends $TUNE_END UTC ($T_TUNE), hard stop $HARD UTC ($T_HARD), $(left)s left"
while tmux has-session -t "$WAIT" 2>/dev/null; do
  [ "$(left)" -le 0 ] && { say "hard stop reached while still waiting for $WAIT"; exit 0; }
  sleep 60
done
say "$WAIT gone"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

# What is actually free on this card, less 2 GB of headroom for the neighbour job's fluctuation.
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
MAXGB=$(awk -v m="$FREE_MIB" 'BEGIN{printf "%.1f", m/1024 - 2}')
say "card $GPU has ${FREE_MIB} MiB free, budgeting ${MAXGB} GB per fit"

say "E1: step sweep"
E1_DEADLINE=$T_TUNE REPO=$REPO bash $REPO/scripts/spiderman/steps_sweep.sh "$GPU"

say "E2: fit ladder"
for MB in 16 24 32 40; do
  [ "$(( T_TUNE - $(date +%s) ))" -le 1800 ] && { say "skipping fit mb$MB, under 30 min of tune time left"; break; }
  FIT=1 REPO=$REPO bash $REPO/scripts/spiderman/decoder_mse.sh "$GPU" $MB 60 || say "fit mb$MB did not finish"
done

SECS=$(( T_TUNE - $(date +%s) ))
if [ "$SECS" -lt 600 ]; then
  say "only ${SECS}s of tune time left; skipping the tune and scoring what exists"
else
  PICK=$($PY $REPO/tools/pick_fit.py --dir $D/tmp/levers --seconds-left "$SECS" --max-gb "$MAXGB")
  echo "$PICK"
  # grep failing leaves an empty string, and `eval ""` succeeds, so the fallback is chosen on the
  # string being empty rather than on an exit status.
  ASSIGN=$(echo "$PICK" | grep '^MB=' || true)
  if [ -n "$ASSIGN" ]; then
    eval "$ASSIGN"
  else
    say "no usable fit measured; falling back to the incumbent recipe's micro 16"
    MB=16; MAX_STEPS=$(( SECS / 4 ))
  fi
  HOURS=$(awk -v s="$SECS" 'BEGIN{printf "%.3f", s/3600}')
  say "E2: tune at micro $MB for $MAX_STEPS updates / ${HOURS}h"
  REPO=$REPO bash $REPO/scripts/spiderman/decoder_mse.sh "$GPU" "$MB" "$MAX_STEPS" "$HOURS"
fi

say LEVERS_QUEUE_DONE
