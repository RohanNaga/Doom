#!/bin/bash
# E1: how many DDIM steps the teacher-forced numbers actually need.
#
# GameNGen's Table 3 (2,048 teacher-forced frames) reads 25.47 dB at 1 step, 31.91 at 2, 32.58 at
# 4, 32.55 at 8 and 32.19 at 64, and they ship the 4-step model. We score every row at 50. If 4 to
# 8 steps match 50 on our footage then every evaluation in the paper costs six to twelve times less
# than it does today, and the rollout harness can be run at horizons we cannot currently afford.
#
# Nothing here is a new measurement protocol: the arguments are the ones after_run2.sh scored each
# row with, and only --num-windows (512, to fit 24 runs in the hour) and --steps differ. 50 steps
# runs first on every row and corpus so the reproduction against the stored 2,048-window numbers is
# the first thing on disk.
#
# E1_DEADLINE, a UTC epoch second, stops the sweep between runs rather than in the middle of one,
# so the card is free for the tune behind it even if the sweep started late.
#
#   usage: [E1_DEADLINE=<epoch>] steps_sweep.sh <gpu>
set -u
GPU=${1:?gpu}
D=/sata2/data/rnagabhi/doom
REPO=${REPO:-$D/tmp/levers/repo}
OUT=$D/results_spiderman/levers_2026-09-20/e1-steps
PY=${PY:-$HOME/miniconda3/envs/doom/bin/python}
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TMPDIR=$D/tmp/tmpdir TORCH_HOME=$D/tmp/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $TMPDIR $TORCH_HOME $OUT $D/logs
cd $REPO

COMMON="--vae-path $D/vae_decoder_arnold_lpips/vae --hf-cache $D/hf/hub --context-frames 32 --num-actions 29"
for ROW in unet:031-unet-l32-aligned pixart:033-pixart-l32-aligned; do
  BB=${ROW%%:*}; RUN=${ROW#*:}
  for S in seen unseen; do
    for N in 50 4 8 2 16 1; do
      O=$OUT/${BB}_${S}_s${N}
      if [ -f $O/metrics.json ]; then echo "$(date -u) skip $BB $S steps=$N"; continue; fi
      if [ -n "${E1_DEADLINE:-}" ] && [ "$(date +%s)" -ge "${E1_DEADLINE}" ]; then
        echo "$(date -u) E1 deadline reached, stopping before $BB $S steps=$N"
        echo E1_STEPS_DEADLINE; exit 0
      fi
      nice -n 15 $PY eval_tf.py --backbone $BB --ckpt $D/results_spiderman/$RUN/best.pt $COMMON \
        --latents-dir $D/latents_arnold_eval/$S --parquet-dir $D/raw_arnold_eval/$S \
        --split $D/latents_arnold_eval/split_$S.json --subset val \
        --num-windows 512 --batch-size 16 --steps $N --seed 0 --save-images 0 \
        --out-dir $O > $D/logs/e1_${BB}_${S}_s${N}.log 2>&1
      echo "$(date -u) $BB $S steps=$N exit $?"
    done
  done
done
echo E1_STEPS_DONE
