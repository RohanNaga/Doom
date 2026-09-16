#!/bin/bash
# throughput probes for the video row on one gpu: (a) microbatch 8 with checkpointing, no accumulation; (b) microbatch 2 without checkpointing, accumulation 4. Usage: fit_video2.sh <gpu>
GPU=${1:?gpu}; D=/sata2/data/rnagabhi/doom; cd $D/repo; export TMPDIR=$D/tmp/tmpdir
for cfg in "8 --grad-ckpt" "2 --no-grad-ckpt" "4 --no-grad-ckpt"; do set -- $cfg; MB=$1; CK=$2
  CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=0 ~/wanenc/bin/python train_video.py --fit-check 12 --context-frames 8 --per-gpu-batch $MB --global-batch 8 $CK --objective flow --null-prompt none --latents-dir $D/tmp/wan_smoke --split $D/split_arnold.json --hf-cache $D/hf/hub --num-workers 2 --results-dir $D/tmp/fit_video2_${MB}${CK} > $D/logs/fit_video2_${MB}${CK}.log 2>&1
  echo "FIT2 mb=$MB $CK exit $? $(grep fit_check $D/logs/fit_video2_${MB}${CK}.log | tail -1 | cut -c1-200)" >> $D/logs/fit_video2.log
done
echo FIT2_DONE >> $D/logs/fit_video2.log
