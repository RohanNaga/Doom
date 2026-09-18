#!/bin/bash
# Re-encode the whole Doom corpus in SD 3.5 Medium's 16-channel latent space.
#   usage: encode_sd35.sh <gpu> [shards]      e.g.  BS=24 bash scripts/spiderman/encode_sd35.sh 3 2
#
# Writes, mirroring the 4-channel corpus exactly:
#   $D/latents_arnold_sd35/                  from $D/raw_arnold        (851 episodes, ~43 GB)
#   $D/latents_arnold_eval_sd35/{seen,unseen,unseen2}  from $D/raw_arnold_eval/*
# Same per-episode layout (ep_XXXXX_latents.npy + _meta.npz), same chains, same decision
# alignment; only the autoencoder and the channel count change.
#
# The canonical control table is *taken from* $D/latents_arnold_aligned rather than recomputed,
# for both the training and the evaluation corpora. `transitions.canonical_table` is the modal
# control vector per action id over the corpus, so recomputing it would give the same answer,
# but reusing the file makes the two corpora's chains identical by construction instead of by
# coincidence, which is what the verification compares.
#
# GPU 3 is shared with a decoder tune holding about 28 GB of the 48 GB card, so the encoder has
# to stay under 15 GB. The encoder is the only part of the autoencoder that runs (about 34 M
# parameters) and activations dominate, so the batch size is the whole budget: set BS from the
# measurement, not from a guess. Two shards on one card overlap one shard's parquet decode with
# the other's GPU work; each shard costs its own copy of the budget, so BS is per shard.
#
# HF_HOME is deliberately left unset: it relocates the token file, and SD 3.5 is a gated repo
# whose token has to sit at its default path. HF_HUB_OFFLINE=0 so the first shard can fetch the
# autoencoder into $D/hf/hub (the root filesystem is full; nothing may land in the default cache).
set -u
GPU=${1:?gpu}
SHARDS=${2:-2}
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

mkdir -p "$D/tmp/tmpdir" "$D/logs" "$OUT" "$EVAL_OUT" "$MARK"
[ -f "$CAN" ] || { echo "no canonical table at $CAN"; exit 1; }
cd $D/repo || exit 1
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i "$GPU"

for i in $(seq 0 $((SHARDS - 1))); do
  rm -f "$MARK/shard$i.done"
  DC=""; [ "$i" = 0 ] && DC="--decode-check 16"
  tmux new-session -d -s "enc-sd35-$i" \
    "cd $D/repo && TMPDIR=$D/tmp/tmpdir HF_HUB_OFFLINE=0 CUDA_VISIBLE_DEVICES=$GPU $PY encode_parquet.py \
       --in-dir $D/raw_arnold --out-dir $OUT --shard $i --num-shards $SHARDS $LATENT $COMMON $DC \
       > $D/logs/encode_sd35_$i.log 2>&1; \
     echo \"\$(date -Iseconds) TRAIN shard $i exit \$?\" >> $D/logs/encode_sd35.log; touch $MARK/shard$i.done"
done

# The evaluation corpora are a few hundred episodes in total and wait for the training corpus
# rather than competing with it for the same 15 GB.
tmux new-session -d -s "enc-sd35-eval" \
  "while [ \$(ls $MARK/shard*.done 2>/dev/null | wc -l) -lt $SHARDS ]; do sleep 60; done; \
   for c in seen unseen unseen2; do \
     TMPDIR=$D/tmp/tmpdir HF_HUB_OFFLINE=0 CUDA_VISIBLE_DEVICES=$GPU $PY $D/repo/encode_parquet.py \
       --in-dir $D/raw_arnold_eval/\$c --out-dir $EVAL_OUT/\$c $LATENT $COMMON \
       > $D/logs/encode_sd35_eval_\$c.log 2>&1; \
     echo \"\$(date -Iseconds) EVAL \$c exit \$?\" >> $D/logs/encode_sd35.log; \
     cp $D/latents_arnold_eval/split_\$c.json $EVAL_OUT/split_\$c.json; \
   done; echo \"\$(date -Iseconds) ENCODE_SD35_DONE\" >> $D/logs/encode_sd35.log; touch $MARK/eval.done"

tmux ls
echo "log: $D/logs/encode_sd35.log, per shard $D/logs/encode_sd35_<shard>.log"
