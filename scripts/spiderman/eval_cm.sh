#!/bin/bash
# Teacher-forced seen-map evaluation of the compute-matched 2.5k-update rows (80k samples), for the sample-matched comparison with the video row. Usage: eval_cm.sh <gpu>
D=/sata2/data/rnagabhi/doom; GPU=${1:?gpu}; PY=~/miniconda3/envs/doom/bin/python; export TMPDIR=$D/tmp/tmpdir; mkdir -p $TMPDIR
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1
for spec in "060-dit-cm2500 dit" "061-unet-cm2500 unet" "062-pixart-cm2500 pixart"; do set -- $spec; RUN=$1; BB=$2; R=$D/results_spiderman/$RUN
  [ -f $R/eval_tf_seen/metrics.json ] && { echo "$RUN done"; continue; }
  cd $D/repo && $PY eval_tf.py --backbone $BB --ckpt $R/best.pt --vae-path $D/vae_decoder_arnold_lpips/vae --hf-cache $D/hf/hub --context-frames 32 --num-actions 29 --latents-dir $D/latents_arnold_eval/seen --parquet-dir $D/raw_arnold_eval/seen --split $D/latents_arnold_eval/split_seen.json --subset val --num-windows 2048 --batch-size 16 --steps 50 --out-dir $R/eval_tf_seen > $D/logs/${RUN}_eval_tf_seen.log 2>&1
  echo "$RUN exit $?" >> $D/logs/eval_cm.log
done
echo CM_EVAL_DONE >> $D/logs/eval_cm.log
