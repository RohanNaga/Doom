#!/bin/bash
# Re-encode the Doom corpus in SD 3.5 Medium's 16-channel latent space.
#   usage: encode_sd35.sh <gpu> [shards] [train|eval|both]
#   e.g.   BS=32 bash scripts/spiderman/encode_sd35.sh 3 6 train
#          BS=32 bash scripts/spiderman/encode_sd35.sh 3 2 eval
#
# Writes, mirroring the 4-channel corpus exactly:
#   train  $D/latents_arnold_sd35/                            from $D/raw_arnold  (850 episodes, 41 GB)
#   eval   $D/latents_arnold_eval_sd35/{seen,unseen,unseen2}   from $D/raw_arnold_eval/*
# Same per-episode layout (ep_XXXXX_latents.npy + _meta.npz), same chains, same decision
# alignment; only the autoencoder and the channel count change.
#
# The two phases are separate commands because they compete for a card with the row that trains
# on the output: the training corpus is on the critical path and the evaluation corpora are not,
# so they are usually encoded later, with fewer shards, or on a different GPU.
#
# The canonical control table is *taken from* $D/latents_arnold_aligned rather than recomputed,
# for both phases. `transitions.canonical_table` is the modal control vector per action id over
# the corpus, so recomputing it would give the same answer, but reusing the file makes the two
# corpora's chains identical by construction instead of by coincidence, which is what
# `verify_corpus.py` compares. The evaluation corpora are far too small to derive their own.
#
# Throughput comes from the number of shards, not the batch: measured encoder throughput on an
# A6000 is flat at 41.5 to 43.2 frames/s from batch 8 to batch 48, and one shard saturates the
# card's compute, so a shard's batch only sets its memory footprint (about 6.1 GB reserved at
# batch 32). Six shards measured 13 frames/s each, 78 aggregate, on a card shared with one other
# job. Pick the shard count from the free memory and how much of the card you may take.
#
# HF_HOME is deliberately left unset: it relocates the token file, and SD 3.5 is a gated repo
# whose token has to sit at its default path. HF_HUB_OFFLINE=0 so the first shard can fetch the
# autoencoder into $D/hf/hub (the root filesystem is full; nothing may land in the default cache).
set -u
GPU=${1:?gpu}
SHARDS=${2:-2}
MODE=${3:-both}
D=/sata2/data/rnagabhi/doom
PY=${PY:-$HOME/miniconda3/envs/doom/bin/python}
BS=${BS:-16}
VAE=stabilityai/stable-diffusion-3.5-medium
CAN=$D/latents_arnold_aligned/canonical_controls.json
OUT=$D/latents_arnold_sd35
EVAL_OUT=$D/latents_arnold_eval_sd35
MARK=$D/tmp/enc_sd35
LATENT="--vae-id $VAE --vae-subfolder vae --cache-dir $D/hf/hub --latent-channels 16 --scaling-factor 1.5305 --shift-factor 0.0609"
COMMON="--align-decisions --canonical $CAN --stride 4 --batch-size $BS --device cuda:0 --decode-threads 8"
CORPORA="seen unseen unseen2"

mkdir -p "$D/tmp/tmpdir" "$D/logs" "$OUT" "$EVAL_OUT" "$MARK"
[ -f "$CAN" ] || { echo "no canonical table at $CAN"; exit 1; }
cd $D/repo || exit 1
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i "$GPU"

if [ "$MODE" = train ] || [ "$MODE" = both ]; then
  for i in $(seq 0 $((SHARDS - 1))); do
    rm -f "$MARK/shard$i.done"
    DC=""; [ "$i" = 0 ] && DC="--decode-check 16"
    tmux new-session -d -s "enc-sd35-$i" \
      "cd $D/repo && TMPDIR=$D/tmp/tmpdir HF_HUB_OFFLINE=0 CUDA_VISIBLE_DEVICES=$GPU $PY encode_parquet.py \
         --in-dir $D/raw_arnold --out-dir $OUT --shard $i --num-shards $SHARDS $LATENT $COMMON $DC \
         > $D/logs/encode_sd35_$i.log 2>&1; \
       echo \"\$(date -Iseconds) TRAIN shard $i exit \$?\" >> $D/logs/encode_sd35.log; touch $MARK/shard$i.done"
  done
fi

if [ "$MODE" = eval ]; then
  # The split files are static, so they are copied now rather than after the encode: a consumer
  # that finds a corpus directory always finds the split that names its episodes.
  for c in $CORPORA; do
    mkdir -p "$EVAL_OUT/$c"
    cp "$D/latents_arnold_eval/split_$c.json" "$EVAL_OUT/split_$c.json"
  done
  for i in $(seq 0 $((SHARDS - 1))); do
    rm -f "$MARK/eval$i.done"
    tmux new-session -d -s "enc-sd35-eval-$i" \
      "cd $D/repo; for c in $CORPORA; do \
         TMPDIR=$D/tmp/tmpdir HF_HUB_OFFLINE=0 CUDA_VISIBLE_DEVICES=$GPU $PY encode_parquet.py \
           --in-dir $D/raw_arnold_eval/\$c --out-dir $EVAL_OUT/\$c --shard $i --num-shards $SHARDS $LATENT $COMMON \
           > $D/logs/encode_sd35_eval_\${c}_$i.log 2>&1; \
         echo \"\$(date -Iseconds) EVAL \$c shard $i exit \$?\" >> $D/logs/encode_sd35.log; \
       done; touch $MARK/eval$i.done"
  done
fi

if [ "$MODE" = both ]; then
  # One process per corpus, started only once every training shard is finished.
  tmux new-session -d -s "enc-sd35-eval" \
    "while [ \$(ls $MARK/shard*.done 2>/dev/null | wc -l) -lt $SHARDS ]; do sleep 60; done; \
     for c in $CORPORA; do \
       TMPDIR=$D/tmp/tmpdir HF_HUB_OFFLINE=0 CUDA_VISIBLE_DEVICES=$GPU $PY $D/repo/encode_parquet.py \
         --in-dir $D/raw_arnold_eval/\$c --out-dir $EVAL_OUT/\$c $LATENT $COMMON \
         > $D/logs/encode_sd35_eval_\$c.log 2>&1; \
       echo \"\$(date -Iseconds) EVAL \$c exit \$?\" >> $D/logs/encode_sd35.log; \
       cp $D/latents_arnold_eval/split_\$c.json $EVAL_OUT/split_\$c.json; \
     done; echo \"\$(date -Iseconds) ENCODE_SD35_DONE\" >> $D/logs/encode_sd35.log; touch $MARK/eval.done"
fi

tmux ls
echo "log: $D/logs/encode_sd35.log, per shard $D/logs/encode_sd35_<shard>.log"
