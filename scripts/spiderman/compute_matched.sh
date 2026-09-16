#!/bin/bash
# Compute-matched short runs: each image-pretrained row trained for 2,500 updates of batch 32 (80k samples, the video row's
# sample budget) under the identical recipe, so the video row can be compared at matched samples. Sequential on one gpu.
# Usage: compute_matched.sh <gpu>. Runs 060-dit-cm2500, 061-unet-cm2500, 062-pixart-cm2500; skips finished ones.
GPU=${1:?gpu}; D=/sata2/data/rnagabhi/doom; T=TMPDIR=/sata2/data/rnagabhi/doom/tmp/tmpdir
COMMON="--context-frames 32 --num-actions 29 --global-batch 32 --lr 5e-5 --warmup 2000 --clip 1.0 --steps 2500 --seed 0 --action-dropout 0.0 --require-verified-transitions --latents-dir $D/latents_arnold_aligned --split $D/split_arnold.json --val-every 500 --val-windows 1024 --ckpt-every 2500 --keep-last 1 --num-workers 4"
for spec in "060-dit-cm2500 dit --per-gpu-batch 32 --warm-start $D/weights/DiT-XL-2-256x256.pt" "061-unet-cm2500 unet --per-gpu-batch 16 --hf-cache $D/hf/hub" "062-pixart-cm2500 pixart --per-gpu-batch 32 --warm-start PixArt-alpha/PixArt-XL-2-512x512 --hf-cache $D/hf/hub"; do
  set -- $spec; RUN=$1; BB=$2; shift 2; R=$D/results_spiderman/$RUN
  [ -f $R/log.jsonl ] && grep -q "\"event\": \"end\"" $R/log.jsonl && { echo "$RUN finished"; continue; }
  echo "$(date -Iseconds) compute-matched $RUN on gpu $GPU" >> $D/logs/resumes.log
  cd $D/repo && env $T HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU ~/miniconda3/envs/doom/bin/python train_wm.py --backbone $BB "$@" --results-dir $R $COMMON >> $D/logs/train_cm.log 2>&1
  echo "$RUN exit $?" >> $D/logs/compute_matched.log
done
echo CM_DONE >> $D/logs/compute_matched.log
