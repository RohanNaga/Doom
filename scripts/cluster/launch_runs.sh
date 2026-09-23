#!/bin/bash
# Zero to a usable node, step 5 of 5: the two next-tic rows, each on its own card, in tmux.
#
#   usage: [DOOM_ROOT=..] [UNET_GPU=0] [SD35_GPU=1] [MB_UNET=32] [MB_SD35=32] [WORKERS=..] \
#          [STEPS=400000] [PY_UNET=..] [PY_SD35=..] [PY=..] [RUN_REPO=$D/repo] [ONLY=unet|sd35] [DRY=1] \
#          launch_runs.sh
#
# PY_UNET runs the U-Net row and PY_SD35 the SD 3.5 row, each falling back to PY and then to the
# node's one env, $D/env/bin/python; they must be the interpreters gates.sh certified. RUN_REPO is
# the checkout the runs execute from (default $D/repo) and must be the one gates.sh certified.
#
# ONE CARD PER ROW, and that is the recipe's ceiling, not the card's. The global batch is 32 with
# no gradient accumulation, so `launch_nexttic.sh` requires micro-batch x cards = 32: a second card
# for a row would mean micro-batch 16 on each, which fills neither. On an 80 GB H100 the largest
# micro-batch this recipe can use is therefore 32, measured by `gates.sh` at about 36 GB for SD 3.5
# with checkpointing and less for the U-Net. The remaining six cards cannot be spent on these two
# rows without changing the global batch, which is a recipe change and is not ours to make here.
# What they CAN take is more rows (a PixArt third row, a second seed) or the evaluation passes,
# each a separate decision.
#
# WORKERS is sized from the node's cores: a per-tic corpus does not fit the page cache and the
# loader is seek-bound, so each run gets about a quarter of the cores, clamped to 4..16. Two runs
# share the box, so half the cores stay for the encoders, the evaluators and the OS.
#
# Stopping: `tmux kill-session -t train-unet-nexttic`. Never `pkill -f` on a pattern from the
# command line; the tmux server carries the first session's whole command string in its own
# arguments and a pattern match kills every session on the machine.
#
# Resuming: run this again. The launcher resumes from the newest recovery checkpoint in the run
# directory and refuses to start a second copy of a live session.
#
# DRY=1 prints both launch commands and touches nothing.
set -u
D=${DOOM_ROOT:-/data/doom}
DRY=${DRY:-0}
UNET_GPU=${UNET_GPU:-0}
SD35_GPU=${SD35_GPU:-1}
MB_UNET=${MB_UNET:-32}
MB_SD35=${MB_SD35:-32}
STEPS=${STEPS:-400000}
ONLY=${ONLY:-}
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
LAUNCH=$REPO/scripts/spiderman/launch_nexttic.sh
PY_UNET=${PY_UNET:-${PY:-$D/env/bin/python}}
PY_SD35=${PY_SD35:-${PY:-$D/env/bin/python}}
RUN_REPO=${RUN_REPO:-$D/repo}

CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 16)
DEFAULT_WORKERS=$(( CORES / 4 ))
[ "$DEFAULT_WORKERS" -lt 4 ] && DEFAULT_WORKERS=4
[ "$DEFAULT_WORKERS" -gt 16 ] && DEFAULT_WORKERS=16
WORKERS=${WORKERS:-$DEFAULT_WORKERS}

die() { echo "LAUNCH_RUNS_FAILED $*" >&2; exit 1; }
[ "$UNET_GPU" = "$SD35_GPU" ] && die "both rows were given the same card ($UNET_GPU); they must not overlap"

mb()      { [ "$1" = sd35 ] && echo "$MB_SD35" || echo "$MB_UNET"; }
py()      { [ "$1" = sd35 ] && echo "$PY_SD35" || echo "$PY_UNET"; }
gpu()     { [ "$1" = sd35 ] && echo "$SD35_GPU" || echo "$UNET_GPU"; }
session() { echo "train-$1-nexttic"; }

launcher() {  # launcher <backbone>
  env DRY=$DRY DOOM_ROOT=$D RUN_REPO="$RUN_REPO" PY_UNET="$PY_UNET" PY_SD35="$PY_SD35" MB="$(mb "$1")" WORKERS="$WORKERS" STEPS="$STEPS" \
    bash "$LAUNCH" "$(gpu "$1")" "$1"
}

ROWS=${ONLY:-"unet sd35"}

if [ "$DRY" = 1 ]; then
  echo "DRY launch_runs root=$D unet_gpu=$UNET_GPU sd35_gpu=$SD35_GPU workers=$WORKERS cores=$CORES steps=$STEPS"
  echo "DRY note global batch 32 with no accumulation means micro-batch x cards = 32, so each row takes one card at micro-batch 32; the spare cards cannot join a row without changing the recipe"
  for BB in $ROWS; do
    echo "DRY tmux session $(session "$BB") on gpu $(gpu "$BB") micro-batch $(mb "$BB")"
    launcher "$BB"
  done
  echo "DRY watch scripts/cluster/status.sh reads $D/results_spiderman/*/log.jsonl"
  exit 0
fi

for BB in $ROWS; do
  [ -x "$(py "$BB")" ] || die "no interpreter for $BB at $(py "$BB") (run setup_node.sh first)"
done
[ -f "$LAUNCH" ] || die "no launch_nexttic.sh at $LAUNCH"
command -v tmux >/dev/null 2>&1 || die "no tmux on this node (apt-get install tmux)"

RC=0
for BB in $ROWS; do
  echo "$(date -Iseconds) launching $BB on gpu $(gpu "$BB") (micro-batch $(mb "$BB"), workers $WORKERS)"
  launcher "$BB" || { RC=$?; echo "LAUNCH_RUNS_FAILED $BB rc=$RC" >&2; }
done
[ $RC -eq 0 ] || exit $RC
echo "LAUNCH_RUNS_DONE $(date -Iseconds); watch with scripts/cluster/status.sh, stop with tmux kill-session -t $(session unet)"
