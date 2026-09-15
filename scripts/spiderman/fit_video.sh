#!/bin/bash
# SkyReels row fit checks on one gpu: L16 flow, L8 flow, L16 vp-v; 20 updates each on the smoke episode. Usage: fit_video.sh <gpu>
GPU=${1:?gpu}; D=/sata2/data/rnagabhi/doom; cd $D/repo
for cfg in "16 flow" "8 flow" "16 vp-v"; do set -- $cfg; L=$1; OBJ=$2
  CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=0 ~/wanenc/bin/python train_video.py --fit-check 20 --context-frames $L --per-gpu-batch 1 --global-batch 8 --objective $OBJ --null-prompt none --latents-dir $D/tmp/wan_smoke --split $D/split_arnold.json --hf-cache $D/hf/hub --num-workers 2 --results-dir $D/tmp/fit_video_${L}_${OBJ} > $D/logs/fit_video_${L}_${OBJ}.log 2>&1
  echo "FIT $L $OBJ exit $?" >> $D/logs/fit_video.log
done
echo FIT_VIDEO_DONE >> $D/logs/fit_video.log
