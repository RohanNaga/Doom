#!/bin/bash
# Per-tic latents of the DENSE corpus, for the next-tic rows. Every tic is a training target, so
# `encode_parquet.py --every-tic` keeps every row and additionally marks the rows the stride-4
# corpus would have kept (`is_decision`, `chain_id`), which is what keeps a stride-1 result
# comparable with a stride-4 one.
#
#   usage: [CORPUS=train|val|test|unseen|evals|all] [VAE=sd15|sd35] [GPU=1] [SHARD=i/n] \
#          [THREADS=24] [WORKERS=0] [DRY=1] [DOOM_ROOT=..] encode_nexttic.sh
#
# The id ranges are `release/dense_split.json`, fixed by Rohan on Sep 20 2026 before anything was
# scored. `record_arnold.py:322-323` assigns map_ids[episode_id % len(map_ids)], so every range
# below is exactly map-balanced:
#
#   CORPUS   source                              ids          episodes  out
#   train    raw_arnold_dense/arenas             0:2000        500/arena latents_arnold_dense_pertic/arenas
#   val      raw_arnold_dense/arenas             6000:6100     25/arena  latents_arnold_dense_pertic_eval/val
#   test     raw_arnold_dense/arenas             7000:7100     25/arena  latents_arnold_dense_pertic_eval/test
#   unseen   raw_arnold_dense/arenas_678         0:60          20/arena  latents_arnold_dense_pertic_eval/arenas_678
#
# `--episode-ids A:B` is what makes this possible without a symlink farm: the encoder filters by the
# id in the `ep_XXXXX.parquet` filename, so the held-out latents come straight out of the same
# 8,000-episode directory the training latents do. Sharding happens AFTER the id filter, so
# `SHARD=0/4` and `SHARD=1/4` of one corpus are disjoint and several cards can fill one output
# directory. The lock directory carries the shard, for the same reason.
#
# VAE=sd35 writes the 16-channel corpus with SD 3.5's own autoencoder, scale 1.5305 shift 0.0609,
# exactly as encode_sd35.sh does; its output directories get the `_sd35` suffix. A row must be
# trained and evaluated in one latent space, so the two sets are never mixed.
#
# Batch size is NOT a free choice: under bf16 autocast cuDNN picks its algorithm from the batch
# shape, so the same frame encodes slightly differently at a different batch size. This corpus has
# no stride-4 reference to reproduce, so 64 is used throughout and recorded in encode_meta.
#
# DRY=1 prints the command each corpus would be given and stops before every side effect;
# DOOM_ROOT repoints the data root. Both default to the real thing.
set -u
D=${DOOM_ROOT:-/sata2/data/rnagabhi/doom}
DRY=${DRY:-0}
CORPUS=${CORPUS:-evals}
VAE=${VAE:-sd15}
GPU=${GPU:-1}
THREADS=${THREADS:-24}
WORKERS=${WORKERS:-0}           # --decode-workers: processes for PNG decode (0 = the thread pool)
BATCH=${BATCH:-64}
SHARD=${SHARD:-}
PY=${PY:-$HOME/miniconda3/envs/doom/bin/python}
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ENC=$REPO/encode_parquet.py

case $VAE in
  sd15) VAE_FLAGS=""; SUF="" ;;
  sd35)
    VAE_FLAGS="--vae-id stabilityai/stable-diffusion-3.5-medium --vae-subfolder vae --latent-channels 16 \
 --scaling-factor 1.5305 --shift-factor 0.0609"
    SUF="_sd35"; PY=${PY_SD35:-$HOME/wanenc/bin/python} ;;
  *) echo "unknown VAE=$VAE (sd15|sd35)" >&2; exit 2 ;;
esac

[ "$DRY" = 1 ] || [ -f "$ENC" ] || { echo "no encode_parquet.py at $ENC (set REPO=/path/to/checkout)" >&2; exit 2; }
export TMPDIR=$D/tmp/tmpdir HF_HOME=$D/hf
[ "$DRY" = 1 ] || mkdir -p $TMPDIR $D/logs

one() {   # one <in-dir> <out-dir> <tag> <episode-ids>
  local CMD=($PY "$ENC" --in-dir "$1" --out-dir "$2" --every-tic --stride 4 --episode-ids "$4"
             --batch-size "$BATCH" --decode-threads "$THREADS" --decode-workers "$WORKERS"
             --device "cuda:$GPU" --dtype bf16 --cache-dir "$D/hf/hub" --decode-check 16 $VAE_FLAGS)
  [ -n "$SHARD" ] && CMD+=(--shard "$SHARD")
  if [ "$DRY" = 1 ]; then
    echo "DRY $3 ${CMD[*]}"
    return 0
  fi
  [ -d "$1" ] || { echo "no such corpus directory: $1" >&2; return 2; }
  mkdir -p "$2"
  # one writer per (output directory, shard): two encodes of the same shard race on each episode,
  # but two different shards of one corpus are disjoint by construction and may share the directory
  local LOCK="$2/.lock${SHARD:+.shard$(echo "$SHARD" | tr / of)}"
  if ! mkdir "$LOCK" 2>/dev/null; then
    echo "$LOCK exists: this shard is already being encoded (remove it if that encode died)" >&2; return 3
  fi
  # The lock cleanup runs in a SUBSHELL with an EXIT trap, not in a RETURN trap on this function.
  # A RETURN trap is not cleared when the function returns, so it fired again when the caller
  # (`val`, `train`, ...) returned -- by which time `LOCK` was out of scope, and under `set -u`
  # that killed the shell with exit 127 after a SUCCESSFUL encode, before ENCODE_NEXTTIC_DONE and
  # before the remaining corpora were touched.
  (
    trap 'rmdir "$LOCK" 2>/dev/null' EXIT
    echo "$(date -Iseconds) encode-nexttic $3 in=$1 out=$2 ids=$4 vae=$VAE gpu=$GPU batch=$BATCH threads=$THREADS workers=$WORKERS shard=${SHARD:-none} repo=$REPO git=$(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null || echo '?')" | tee -a "$2/ENCODE_LOG.txt"
    # capture the ENCODER's status: `$?` after an `echo | tee` is the pipeline's, so a failed encode
    # would have been logged as exit 0 and the caller would have gone on to the next corpus
    nice -n 5 "${CMD[@]}"
    RC=$?
    echo "$(date -Iseconds) done $3 exit=$RC files=$(ls "$2"/ep_*_latents.npy 2>/dev/null | wc -l)" | tee -a "$2/ENCODE_LOG.txt"
    [ $RC -eq 0 ] || exit $RC
    # the evaluation corpora need their split file before after_nexttic.sh can score them; the path
    # is derived by make_dense_eval_splits.py from the latents directory, so the writer and the
    # reader cannot disagree about where it goes
    [ "$3" = train ] || "$PY" "$REPO/make_dense_eval_splits.py" --latents-dir "$2" || exit $?
  )
  local RC=$?
  if [ $RC -ne 0 ]; then
    echo "ENCODE_NEXTTIC_FAILED $3 $RC" >&2
    return $RC
  fi
}

A=$D/raw_arnold_dense/arenas
B=$D/raw_arnold_dense/arenas_678
OT=$D/latents_arnold_dense_pertic${SUF}/arenas
OE=$D/latents_arnold_dense_pertic_eval${SUF}

train()  { one $A $OT train "${TRAIN_IDS:-0:2000}"; }
val()    { one $A $OE/val val "${VAL_IDS:-6000:6100}"; }
test_()  { one $A $OE/test test "${TEST_IDS:-7000:7100}"; }
unseen() { one $B $OE/arenas_678 unseen "${UNSEEN_IDS:-0:60}"; }

RC=0
run() { "$@" || RC=$?; }      # keep going through the other corpora, but remember the failure
case $CORPUS in
  train)  run train ;;
  val)    run val ;;
  test)   run test_ ;;
  unseen) run unseen ;;
  # the evaluation corpora first: 260 episodes against 2,000, and the launch gate needs them
  evals)  run val; run test_; run unseen ;;
  all)    run val; run test_; run unseen; run train ;;
  *) echo "unknown CORPUS=$CORPUS (train|val|test|unseen|evals|all)" >&2; exit 2 ;;
esac
if [ $RC -ne 0 ]; then
  echo "ENCODE_NEXTTIC_FAILED $RC $(date -Iseconds)" >&2
  exit $RC
fi
echo "ENCODE_NEXTTIC_DONE $(date -Iseconds)"
