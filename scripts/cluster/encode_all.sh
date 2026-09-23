#!/bin/bash
# Zero to a usable node, step 3 of 5: both per-tic latent corpora, across the whole box.
#
#   usage: [GPUS=0,1,2,3,4,5,6,7] [VAES=sd15,sd35] [EVAL_GPU=<first GPU>] [THREADS=4] [WORKERS=0] \
#          [BATCH=64] [SKIP_EVALS=1] [SKIP_TRAIN=1] [CHECK=1] [DRY=1] [DOOM_ROOT=..] encode_all.sh
#
# This is a driver around `scripts/spiderman/encode_nexttic.sh`, not a second encoder: every
# command that touches a frame is that launcher's, so a corpus built here is the corpus the
# Spiderman rows were built from. What this adds is the box: the evaluation corpora on one card,
# then the 2,000 training episodes sharded across all of them, both latent spaces, per-shard logs,
# the sidecar audit, the split publish and an inventory.
#
# ONE canonical control table, and it is the DATASET's. `canonical_controls.json` ships in
# `RohanNaga/doom-dense-arnold` and is a property of the recording, not of the autoencoder: the
# modal executed button vector per action id, built once from training ids 0:2000. It decides which
# rows are verified decisions, so two corpora built under two tables have incomparable decision
# masks. It is never rebuilt here. `encode_nexttic.sh` would rebuild it on demand, and six shards
# each scanning `action` and `buttons` of every episode (~10 GB per process) is what the Sep 20
# Spiderman host-memory outage looks like. CHECK=1 runs the preflight alone: it refuses if the
# table, the raw segments or the repo are not where they must be, and writes nothing.
#
# Order. Evaluation corpora first, on a single card: 260 episodes against 2,000, and the launch
# gates need val before anything else. Training shards then take every card, including that one.
# The two passes do not overlap, so no card is double-booked.
#
# Cost, from the A6000 measurement of 66 fps for the 4-channel encoder: about 42 card-hours for the
# 10.07M training tics and about 5.5 for the 260 evaluation episodes, divided by the number of
# cards. The 16-channel SD 3.5 pass has no measured rate; budget at least as much.
#
# DRY=1 prints every command, including the ones the inner launcher would issue, and touches
# nothing.
set -u
D=${DOOM_ROOT:-/data/doom}
DRY=${DRY:-0}
CHECK=${CHECK:-0}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
VAES=${VAES:-sd15,sd35}
THREADS=${THREADS:-4}          # per PART C: 4 decode threads, 0 process workers, then measure
WORKERS=${WORKERS:-0}
BATCH=${BATCH:-64}             # part of the numerical provenance of the corpus; do not vary
SKIP_EVALS=${SKIP_EVALS:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ENC=$REPO/scripts/spiderman/encode_nexttic.sh
PY=${PY:-$D/env/bin/python}
CANON=${CANON:-$D/raw_arnold_dense/canonical_controls.json}
RAW=$D/raw_arnold_dense
TRAIN_IDS=${TRAIN_IDS:-0:2000}
VAL_IDS=${VAL_IDS:-6000:6100}
TEST_IDS=${TEST_IDS:-7000:7100}
# the unseen range comes from release/dense_split.json (60:120 since 2026-09-22; see its `history`)
# shellcheck source=../dense_ids.sh
. "$(dirname "${BASH_SOURCE[0]}")/../dense_ids.sh"
UNSEEN_IDS=${UNSEEN_IDS:-$(dense_ids unseen_ids)}
[ -n "$UNSEEN_IDS" ] || { echo "ENCODE_ALL_FAILED no unseen range: set UNSEEN_IDS or fix release/dense_split.json" >&2; exit 2; }

IFS=, read -r -a CARDS <<< "$GPUS"
IFS=, read -r -a SPACES <<< "$VAES"
NSHARD=${#CARDS[@]}
EVAL_GPU=${EVAL_GPU:-${CARDS[0]}}
die() { echo "ENCODE_ALL_FAILED $*" >&2; exit 1; }

# the inner launcher's environment, identical for every call
inner() {   # inner <vae> <gpu> <corpus> [shard]
  DRY=$DRY DOOM_ROOT=$D REPO=$REPO PY=$PY PY_SD35=$PY CANON=$CANON \
  THREADS=$THREADS WORKERS=$WORKERS BATCH=$BATCH \
  TRAIN_IDS=$TRAIN_IDS VAL_IDS=$VAL_IDS TEST_IDS=$TEST_IDS UNSEEN_IDS=$UNSEEN_IDS \
  VAE=$1 GPU=$2 CORPUS=$3 SHARD=${4:-} bash "$ENC"
}

suffix() { [ "$1" = sd35 ] && echo _sd35 || echo ""; }
channels() { [ "$1" = sd35 ] && echo 16 || echo 4; }

audit_cmd() {   # audit_cmd <vae>: the sidecar audit of the val corpus against the raw parquet
  echo "$PY $REPO/check_action_alignment.py --audit-only" \
       "--latents-dir $D/latents_arnold_dense_pertic_eval$(suffix "$1")/val" \
       "--audit-parquet-dir $RAW/arenas --episodes 100 --audit-rows 100000" \
       "--canonical $CANON --seed 0"
}
split_cmd() {   # split_cmd <vae> <corpus> <ids>; the unseen corpus is also refused if it holds a worker-first episode
  local WF=""
  [ "$2" = arenas_678 ] && WF=" --refuse-worker-first $RAW/arenas_678"
  echo "$PY $REPO/make_dense_eval_splits.py" \
       "--latents-dir $D/latents_arnold_dense_pertic_eval$(suffix "$1")/$2" \
       "--expect-ids $3 --latent-channels $(channels "$1")$WF"
}
eval_dirs() {   # eval_dirs <vae>: the three evaluation corpora of one latent space
  echo "val $VAL_IDS test $TEST_IDS arenas_678 $UNSEEN_IDS"
}

# --- preflight ---------------------------------------------------------------------------
preflight() {
  [ -f "$ENC" ] || die "no encode_nexttic.sh at $ENC (set REPO=/path/to/checkout)"
  [ -x "$PY" ] || die "no interpreter at $PY (run setup_node.sh first)"
  [ -s "$CANON" ] || die "no canonical control table at $CANON: fetch_dataset.sh downloads it with the dataset, and it must never be rebuilt per shard"
  [ -d "$RAW/arenas" ] || die "no raw episodes at $RAW/arenas (run fetch_dataset.sh first)"
  [ -d "$RAW/arenas_678" ] || die "no raw episodes at $RAW/arenas_678 (run fetch_dataset.sh first)"
}
if [ "$CHECK" = 1 ]; then
  preflight
  echo "ENCODE_ALL_PREFLIGHT_OK canonical=$CANON raw=$RAW cards=$NSHARD spaces=${SPACES[*]}"
  exit 0
fi

if [ "$DRY" = 1 ]; then
  echo "DRY encode_all root=$D spaces=${SPACES[*]} cards=${CARDS[*]} shards=$NSHARD eval_gpu=$EVAL_GPU"
  echo "DRY canonical $CANON (shipped with the dataset, shared by both latent spaces, never rebuilt)"
  for V in "${SPACES[@]}"; do
    [ "$SKIP_EVALS" = 1 ] || {
      echo "DRY evals $V gpu=$EVAL_GPU log=$D/logs/encode_${V}_evals.log"
      inner "$V" "$EVAL_GPU" evals
    }
    [ "$SKIP_TRAIN" = 1 ] || {
      for I in "${!CARDS[@]}"; do
        echo "DRY shard $V $I/$NSHARD gpu=${CARDS[$I]} log=$D/logs/encode_${V}_train_shard${I}.log"
        inner "$V" "${CARDS[$I]}" train "$I/$NSHARD"
      done
    }
  done
  for V in "${SPACES[@]}"; do
    echo "DRY audit $V $(audit_cmd "$V")"
  done
  for V in "${SPACES[@]}"; do
    set -- $(eval_dirs "$V")
    while [ $# -gt 0 ]; do echo "DRY splits $V $(split_cmd "$V" "$1" "$2")"; shift 2; done
  done
  for V in "${SPACES[@]}"; do
    echo "DRY inventory $V $D/latents_arnold_dense_pertic$(suffix "$V")/arenas" \
         "$D/latents_arnold_dense_pertic_eval$(suffix "$V")/val" \
         "$D/latents_arnold_dense_pertic_eval$(suffix "$V")/test" \
         "$D/latents_arnold_dense_pertic_eval$(suffix "$V")/arenas_678"
  done
  exit 0
fi

preflight
mkdir -p "$D/logs" || die "cannot write $D/logs"
T0=$(date +%s)
RC=0

for V in "${SPACES[@]}"; do
  if [ "$SKIP_EVALS" != 1 ]; then
    echo "$(date -Iseconds) $V: val, test and arenas_678 on gpu $EVAL_GPU"
    inner "$V" "$EVAL_GPU" evals >> "$D/logs/encode_${V}_evals.log" 2>&1 \
      || { RC=$?; echo "ENCODE_ALL_FAILED $V evals rc=$RC (see $D/logs/encode_${V}_evals.log)" >&2; break; }
  fi
  if [ "$SKIP_TRAIN" != 1 ]; then
    echo "$(date -Iseconds) $V: training ids $TRAIN_IDS in $NSHARD shards on gpus ${CARDS[*]}"
    PIDS=()
    for I in "${!CARDS[@]}"; do
      inner "$V" "${CARDS[$I]}" train "$I/$NSHARD" >> "$D/logs/encode_${V}_train_shard${I}.log" 2>&1 &
      PIDS+=($!)
    done
    for I in "${!PIDS[@]}"; do
      wait "${PIDS[$I]}" || { RC=$?; echo "ENCODE_ALL_FAILED $V train shard $I rc=$RC (see $D/logs/encode_${V}_train_shard${I}.log)" >&2; }
    done
    [ $RC -eq 0 ] || break
  fi
done

# --- the audit and the split publish -------------------------------------------------------
if [ $RC -eq 0 ]; then
  for V in "${SPACES[@]}"; do
    echo "$(date -Iseconds) $V: sidecar audit of the val corpus against the raw parquet"
    # shellcheck disable=SC2046
    $(audit_cmd "$V") > "$D/logs/audit_${V}_val.json" 2>&1 \
      || { RC=$?; echo "ENCODE_ALL_FAILED $V sidecar audit rc=$RC (see $D/logs/audit_${V}_val.json)" >&2; break; }
    tail -3 "$D/logs/audit_${V}_val.json"
  done
fi
if [ $RC -eq 0 ]; then
  for V in "${SPACES[@]}"; do
    set -- $(eval_dirs "$V")
    while [ $# -gt 0 ]; do
      # shellcheck disable=SC2046
      $(split_cmd "$V" "$1" "$2") \
        || { RC=$?; echo "ENCODE_ALL_FAILED $V split $1 rc=$RC" >&2; break; }
      shift 2
    done
  done
fi

# --- inventory ------------------------------------------------------------------------------
pairs() {   # pairs <dir>: episodes with BOTH a latent file and a sidecar; a latent alone is not one
  local dir=$1
  [ -d "$dir" ] || { echo 0; return; }
  comm -12 <(find "$dir" -maxdepth 1 -name 'ep_*_latents.npy' -exec basename {} _latents.npy \; | sort) \
           <(find "$dir" -maxdepth 1 -name 'ep_*_meta.npz' -exec basename {} _meta.npz \; | sort) | wc -l | tr -d ' '
}
TOTAL=0
echo "corpus inventory:"
for V in "${SPACES[@]}"; do
  S=$(suffix "$V")
  for SPEC in "train $D/latents_arnold_dense_pertic$S/arenas" \
              "val $D/latents_arnold_dense_pertic_eval$S/val" \
              "test $D/latents_arnold_dense_pertic_eval$S/test" \
              "arenas_678 $D/latents_arnold_dense_pertic_eval$S/arenas_678"; do
    set -- $SPEC
    B=$(du -sb "$2" 2>/dev/null | cut -f1); B=${B:-0}
    TOTAL=$(( TOTAL + B ))
    printf '  %-6s %-10s episodes=%-5s bytes=%s\n' "$V" "$1" "$(pairs "$2")" "$B"
  done
done
ELAPSED=$(( $(date +%s) - T0 ))
awk -v b="$TOTAL" -v s="$ELAPSED" 'BEGIN { printf "total_bytes=%d (%.1f GiB) elapsed=%ds\n", b, b / 1073741824, s }'
[ $RC -eq 0 ] || exit $RC
echo "ENCODE_ALL_DONE spaces=${SPACES[*]} shards=$NSHARD $(date -Iseconds)"
