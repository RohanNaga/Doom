#!/bin/bash
# Per-tic latents of the corpora we already train and evaluate on, for the coming stride-1
# experiment. The open GameNGen reproduction stores consecutive tics and repeats each action for
# four of them (confirmed from its data: 44,616 of 44,616 action runs start on a tic divisible by 4,
# runs of exactly 4), so a stride-1 recipe needs per-tic latents of the same footage.
#
#   usage: [CORPUS=all|train|seen|unseen|unseen2] [GPU=1] [THREADS=24] [BATCH=64] encode_pertic.sh
#
# Same VAE, same scaling and the same latent contract as the existing stride-4 directories, written
# to NEW directories so nothing existing is touched:
#
#   raw_arnold                -> latents_arnold_pertic              4.21M tics,  48.8 GiB
#   raw_arnold_eval/seen      -> latents_arnold_eval_pertic/seen    0.30M tics,   3.4 GiB
#   raw_arnold_eval/unseen    -> latents_arnold_eval_pertic/unseen  0.10M tics,   1.2 GiB
#   raw_arnold_eval/unseen2   -> latents_arnold_eval_pertic/unseen2 0.68M tics,   7.9 GiB
#                                                            total  5.29M tics,  61.3 GiB
#
# At the measured 66 frames/s that is about 22 card-hours, so it does not finish in one sitting; it
# is resume-safe per episode, so relaunching continues.
#
# Every corpus reuses `latents_arnold_aligned/canonical_controls.json`. All four stride-4 directories
# were built from that one table (verified identical, md5 983b99adf439, 29 entries), and the table
# decides which rows count as decisions, so reusing it is what makes the `is_decision` rows of the
# output reproduce the stride-4 corpus exactly. Check that with:
#
#   python verify_corpus.py --new $D/latents_arnold_pertic --ref $D/latents_arnold_aligned \
#       --latent-channels 4 --ref-latent-channels 4 --reduce-decisions
#
# DRY=1 prints the command each corpus would be given and stops before every side effect, so a test
# can check the wiring without starting a real encode. DOOM_ROOT repoints the data root for the same
# reason. Both default to the real thing. A launcher test that runs this without DRY=1 starts a
# multi-hour GPU job on any machine where /sata2 exists, which is what happened on Sep 21 2026.
set -u
D=${DOOM_ROOT:-/sata2/data/rnagabhi/doom}
DRY=${DRY:-0}
CORPUS=${CORPUS:-all}
GPU=${GPU:-1}
THREADS=${THREADS:-24}          # threads for PNG decode; NOT the bottleneck, see below
# Measured Sep 21 2026 (.claude/analyses/nexttic-design-2026-09-20.md section 9): one thread decodes
# 5,000 frames/s of 320x240 PNG and a parquet read costs 0.0025 ms per frame, against the observed
# 66 frames/s = 15 ms per frame. Decoding is three orders of magnitude off being the limit, so this
# knob buys nothing above a handful of threads; the remaining suspects are the VAE forward at the
# chosen batch size and reading 1.7 TiB off /sata2.
# The micro-batch is NOT a free choice here, it is part of reproducing the reference corpus. Under
# bf16 autocast cuDNN picks its algorithm from the batch shape, so the same frame encodes slightly
# differently at a different batch size. Measured on 5 episodes against latents_arnold_eval/seen,
# over 31.5M latent values: at batch 16, the batch its reference was built with, 99.62% are exactly
# bit-identical (mean |diff| 1.45e-05, below one float16 step at these magnitudes); at batch 64 only
# 60.3% are, and 35.5% differ by more than 1e-3. So each corpus is encoded at the batch size its own
# stride-4 reference used: 64 for the training corpus (reencode_aligned.sh) and 16 for the
# evaluation corpora (record_eval_corpus.sh:16).
BATCH_TRAIN=${BATCH_TRAIN:-64}
BATCH_EVAL=${BATCH_EVAL:-16}
PY=${PY:-$HOME/miniconda3/envs/doom/bin/python}
CANON=$D/latents_arnold_aligned/canonical_controls.json
# Run the encoder from the checkout this script lives in. It used to point at a scratch copy under
# $D/tmp/night, which meant anyone running this from any checkout silently executed one agent's
# personal file -- and two agents doing so at once put two writers on one output directory
# (Sep 21 2026). REPO can still override it for a machine whose repo cannot be updated.
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ENC=$REPO/encode_parquet.py
[ "$DRY" = 1 ] || [ -f "$ENC" ] || { echo "no encode_parquet.py at $ENC (set REPO=/path/to/checkout)" >&2; exit 2; }
export TMPDIR=$D/tmp/tmpdir HF_HOME=$D/hf
[ "$DRY" = 1 ] || mkdir -p $TMPDIR $D/logs

[ "$DRY" = 1 ] || [ -f "$CANON" ] || { echo "missing canonical table $CANON" >&2; exit 2; }

one() {   # one <in-dir> <out-dir> <tag> <batch>
  # built once, so what DRY prints is exactly what would be run
  local CMD=($PY "$ENC" --in-dir "$1" --out-dir "$2" --every-tic --stride 4 --canonical "$CANON"
             --batch-size "$4" --decode-threads "$THREADS" --device "cuda:$GPU" --dtype bf16
             --cache-dir "$D/hf/hub" --decode-check 16)
  if [ "$DRY" = 1 ]; then
    echo "DRY $3 ${CMD[*]}"
    return 0
  fi
  [ -d "$1" ] || { echo "no such corpus directory: $1" >&2; return 2; }
  mkdir -p "$2"
  # one writer per output directory: two concurrent encodes of the same corpus race on each episode
  if ! mkdir "$2/.lock" 2>/dev/null; then
    echo "$2 is already being encoded (remove $2/.lock if that is stale)" >&2; return 3
  fi
  trap 'rmdir "$2/.lock" 2>/dev/null' RETURN
  echo "$(date -Iseconds) encode-pertic $3 in=$1 out=$2 gpu=$GPU batch=$4 threads=$THREADS repo=$REPO git=$(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null || echo '?')" | tee -a "$2/ENCODE_LOG.txt"
  nice -n 5 "${CMD[@]}"
  echo "$(date -Iseconds) done $3 exit=$? files=$(ls "$2"/ep_*_latents.npy 2>/dev/null | wc -l)" | tee -a "$2/ENCODE_LOG.txt"
}

case $CORPUS in
  train)   one $D/raw_arnold $D/latents_arnold_pertic train $BATCH_TRAIN ;;
  seen|unseen|unseen2)
           one $D/raw_arnold_eval/$CORPUS $D/latents_arnold_eval_pertic/$CORPUS $CORPUS $BATCH_EVAL ;;
  all)     # evaluation corpora first: they are small, and the equivalence check needs only a few episodes
           for S in seen unseen unseen2; do
             one $D/raw_arnold_eval/$S $D/latents_arnold_eval_pertic/$S $S $BATCH_EVAL
           done
           one $D/raw_arnold $D/latents_arnold_pertic train $BATCH_TRAIN ;;
  *) echo "unknown CORPUS=$CORPUS (all|train|seen|unseen|unseen2)" >&2; exit 2 ;;
esac
echo "ENCODE_PERTIC_DONE $(date -Iseconds)"
