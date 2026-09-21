#!/bin/bash
# Encode a dense recording segment to SD 1.x 4-channel latents, in the layout every finished row
# trains in: (T, 4, 32, 40) float16 per episode, a meta.npz sidecar beside it, resume-safe per
# episode. Documentation of the corpus itself: release/DENSE_CORPUS.md.
#
#   usage: [MODE=decisions|every-tic] [GPU=1] [BATCH=..] SEGMENT=arenas encode_dense.sh
#
# MODE is the one real choice, and it is a disk decision before it is a research one. The live
# latent contract is (4, 32, 40) fp16, measured at 12.4 KB per frame on disk including the sidecar
# (latents_arnold_aligned: 11 GB for 850 x 1116 frames). For the `arenas` segment, 8,000 episodes of
# about 5,100 tics:
#
#   decisions  (stride 4)   10.2M frames    118 GiB   one frame per agent decision, what the rows train on
#   every-tic  (stride 1)   40.8M frames    473 GiB   readable at any stride later, four times the cost
#
# `decisions` also runs --align-decisions against the main corpus's canonical_controls.json, so a
# decision here means what it means in latents_arnold_aligned and the two corpora are poolable.
# `every-tic` cannot align: alignment selects decision rows, which is the opposite of keeping them all.
#
# DRY=1 prints the command and stops before every side effect; DOOM_ROOT repoints the data root.
# Both default to the real thing, and both exist so a test can check the wiring without starting a
# real multi-hour encode on any machine where /sata2 happens to exist.
set -u
D=${DOOM_ROOT:-/sata2/data/rnagabhi/doom}
DRY=${DRY:-0}
SEGMENT=${SEGMENT:?set SEGMENT (arenas|arenas_678|dm_simple)}
MODE=${MODE:-decisions}
GPU=${GPU:-1}
BATCH=${BATCH:-64}
THREADS=${THREADS:-8}
PY=${PY:-$HOME/miniconda3/envs/doom/bin/python}
IN=$D/raw_arnold_dense/$SEGMENT
OUT=${OUT:-$D/latents_arnold_dense/$SEGMENT}
export TMPDIR=$D/tmp/tmpdir HF_HOME=$D/hf

case $MODE in
  decisions) STRIDE=4; EXTRA=(--align-decisions --canonical $D/latents_arnold_aligned/canonical_controls.json) ;;
  every-tic) STRIDE=1; EXTRA=() ;;
  *) echo "unknown MODE=$MODE (decisions|every-tic)" >&2; exit 2 ;;
esac

CMD=($PY $D/repo/encode_parquet.py --in-dir $IN --out-dir $OUT --stride $STRIDE ${EXTRA[@]+"${EXTRA[@]}"}
     --batch-size $BATCH --decode-threads $THREADS --device cuda:$GPU --dtype bf16
     --cache-dir $D/hf/hub --decode-check 16)
if [ "$DRY" = 1 ]; then
  echo "DRY $SEGMENT mode=$MODE stride=$STRIDE ${CMD[*]}"
  exit 0
fi

# validate before creating anything, so a typo leaves no directories behind
[ -d "$IN" ] || { echo "no such segment directory: $IN" >&2; exit 2; }
mkdir -p $TMPDIR $OUT $D/logs
EPISODES=$(ls $IN/ep_*.parquet 2>/dev/null | wc -l)
echo "$(date -Iseconds) encode segment=$SEGMENT mode=$MODE stride=$STRIDE episodes=$EPISODES gpu=$GPU batch=$BATCH git=$(cd $D/repo && git rev-parse --short HEAD)" | tee -a $OUT/ENCODE_LOG.txt

nice -n 5 "${CMD[@]}"
echo "$(date -Iseconds) done segment=$SEGMENT exit=$? files=$(ls $OUT/ep_*_latents.npy 2>/dev/null | wc -l) bytes=$(du -sb $OUT | cut -f1)" | tee -a $OUT/ENCODE_LOG.txt
