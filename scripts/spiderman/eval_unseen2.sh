#!/bin/bash
# Zero-shot map-transfer scoring: teacher-forced evaluation of every finished row on the curated unseen2 corpus (13 maps). Usage: eval_unseen2.sh <gpu>
D=/sata2/data/rnagabhi/doom; GPU=${1:?gpu}; PY=~/miniconda3/envs/doom/bin/python; export TMPDIR=$D/tmp/tmpdir; mkdir -p $TMPDIR
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1
for spec in "030-dit-l32-aligned dit" "031-unet-l32-aligned unet" "033-pixart-l32-aligned pixart" "032-dit-l32-aligned-seed1 dit"; do set -- $spec; RUN=$1; BB=$2; R=$D/results_spiderman/$RUN
  [ -f $R/eval_tf_unseen2/metrics.json ] && { echo "$RUN done"; continue; }
  cd $D/repo && $PY eval_tf.py --backbone $BB --ckpt $R/best.pt --vae-path $D/vae_decoder_arnold_lpips/vae --hf-cache $D/hf/hub --context-frames 32 --num-actions 29 --latents-dir $D/latents_arnold_eval/unseen2 --parquet-dir $D/raw_arnold_eval/unseen2 --split $D/latents_arnold_eval/split_unseen2.json --subset val --num-windows 2048 --batch-size 16 --steps 50 --out-dir $R/eval_tf_unseen2 > $D/logs/${RUN}_eval_tf_unseen2.log 2>&1
  echo "$RUN exit $?" >> $D/logs/eval_unseen2.log
done
echo UNSEEN2_EVAL_DONE >> $D/logs/eval_unseen2.log
