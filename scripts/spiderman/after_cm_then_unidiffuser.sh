#!/bin/bash
# Waits for the compute-matched runs to finish on this gpu, downloads UniDiffuser, runs a 20-step fit check (batch 32, then 16x2 on OOM),
# runs in the ~/wanenc env (diffusers 0.40 + transformers): the doom env has no transformers and diffusers 0.31 gates UniDiffuserModel on it
# and launches the full 034 row with the configuration that fit. Usage: after_cm_then_unidiffuser.sh <gpu>
D=/sata2/data/rnagabhi/doom; GPU=${1:?gpu}; PY=~/wanenc/bin/python; export TMPDIR=$D/tmp/tmpdir; mkdir -p $TMPDIR
until grep -q CM_DONE $D/logs/compute_matched.log 2>/dev/null; do sleep 300; done
cd $D/repo && git pull -q
HF_HUB_OFFLINE=0 $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('thu-ml/unidiffuser-v1', allow_patterns=['unet/*'], cache_dir='$D/hf/hub'))" >> $D/logs/fit_unidiffuser.log 2>&1
COMMON="--context-frames 32 --num-actions 29 --global-batch 32 --lr 5e-5 --warmup 2000 --clip 1.0 --seed 0 --action-dropout 0.0 --warm-start thu-ml/unidiffuser-v1 --hf-cache $D/hf/hub --latents-dir $D/latents_arnold_aligned --split $D/split_arnold.json --num-workers 4"
for MB in 32 16; do
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU $PY train_wm.py --backbone unidiffuser --fit-check 20 --per-gpu-batch $MB $COMMON --results-dir $D/tmp/fit_unidiffuser_$MB > $D/logs/fit_unidiffuser_$MB.log 2>&1
  if grep -q '"event": "fit_check"' $D/logs/fit_unidiffuser_$MB.log; then echo "FIT_UNI mb=$MB $(grep fit_check $D/logs/fit_unidiffuser_$MB.log | tail -1 | cut -c1-200)" >> $D/logs/fit_unidiffuser.log; MB=$MB bash $D/launch_unidiffuser.sh $GPU >> $D/logs/resumes.log 2>&1; echo "UNI_LAUNCHED mb=$MB" >> $D/logs/fit_unidiffuser.log; exit 0; fi
  echo "FIT_UNI mb=$MB failed" >> $D/logs/fit_unidiffuser.log
done
echo UNI_FIT_FAILED >> $D/logs/fit_unidiffuser.log
