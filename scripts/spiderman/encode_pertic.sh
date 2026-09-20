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
set -u
D=/sata2/data/rnagabhi/doom
CORPUS=${CORPUS:-all}
GPU=${GPU:-1}
BATCH=${BATCH:-64}
THREADS=${THREADS:-24}          # the encoder is PNG-decode bound, so threads matter and batch does not
PY=${PY:-$HOME/miniconda3/envs/doom/bin/python}
CANON=$D/latents_arnold_aligned/canonical_controls.json
export TMPDIR=$D/tmp/tmpdir HF_HOME=$D/hf
mkdir -p $TMPDIR $D/logs

[ -f "$CANON" ] || { echo "missing canonical table $CANON" >&2; exit 2; }

one() {   # one <in-dir> <out-dir> <tag>
  [ -d "$1" ] || { echo "no such corpus directory: $1" >&2; return 2; }
  mkdir -p "$2"
  echo "$(date -Iseconds) encode-pertic $3 in=$1 out=$2 gpu=$GPU batch=$BATCH threads=$THREADS git=$(cd $D/repo && git rev-parse --short HEAD)" | tee -a "$2/ENCODE_LOG.txt"
  nice -n 5 $PY $D/tmp/night/repo_test/encode_parquet.py --in-dir "$1" --out-dir "$2" \
    --every-tic --stride 4 --canonical $CANON \
    --batch-size $BATCH --decode-threads $THREADS --device cuda:$GPU --dtype bf16 \
    --cache-dir $D/hf/hub --decode-check 16
  echo "$(date -Iseconds) done $3 exit=$? files=$(ls "$2"/ep_*_latents.npy 2>/dev/null | wc -l)" | tee -a "$2/ENCODE_LOG.txt"
}

case $CORPUS in
  train)   one $D/raw_arnold $D/latents_arnold_pertic train ;;
  seen|unseen|unseen2)
           one $D/raw_arnold_eval/$CORPUS $D/latents_arnold_eval_pertic/$CORPUS $CORPUS ;;
  all)     # evaluation corpora first: they are small, and the equivalence check needs only a few episodes
           for S in seen unseen unseen2; do
             one $D/raw_arnold_eval/$S $D/latents_arnold_eval_pertic/$S $S
           done
           one $D/raw_arnold $D/latents_arnold_pertic train ;;
  *) echo "unknown CORPUS=$CORPUS (all|train|seen|unseen|unseen2)" >&2; exit 2 ;;
esac
echo "ENCODE_PERTIC_DONE $(date -Iseconds)"
