#!/bin/bash
# Side work on the spare card while the main pair trains: context sweep on corrected data, label-noise ablation, K=2 IDM.
D=/sata2/data/rnagabhi/doom; PY=~/miniconda3/envs/doom/bin/python; cd $D/repo && git pull -q
export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1
COMMON="--num-actions 29 --global-batch 32 --lr 5e-5 --warmup 2000 --clip 1.0 --steps 5000 --seed 0 --action-dropout 0.0 --require-verified-transitions --latents-dir $D/latents_arnold_aligned --split $D/split_arnold.json --val-every 1000 --val-windows 1024 --ckpt-every 5000 --snapshot-every 5000 --keep-last 0 --num-workers 4 --grad-ckpt"
for L in 8 16; do
  R=$D/results_spiderman/040-dit-ctx$L-aligned
  [ -f $R/log.jsonl ] && ! grep -q "\"event\": \"end\"" $R/log.jsonl && rm -rf $R
  grep -q "\"event\": \"end\"" $R/log.jsonl 2>/dev/null && continue
  $PY train_wm.py --backbone dit --context-frames $L --per-gpu-batch 32 --warm-start $D/weights/DiT-XL-2-256x256.pt --results-dir $R $COMMON > $D/logs/sweep_ctx${L}_aligned.log 2>&1
done
echo SWEEP_DONE >> $D/logs/side_queue.log
until [ -f $D/results_superman/010-dit-l32/final/best.pt ] && [ -f $D/results_superman/011-unet-l32/final/best.pt ]; do sleep 120; done
bash $D/eval_ckpt.sh $D/results_spiderman/ablation_gridlabels_dit75k dit $D/results_superman/010-dit-l32/final/best.pt 1
bash $D/eval_ckpt.sh $D/results_spiderman/ablation_gridlabels_unet20k unet $D/results_superman/011-unet-l32/final/best.pt 1
echo ABLATION_DONE >> $D/logs/side_queue.log
$PY train_idm.py --latents-dir $D/latents_arnold_aligned --split $D/split_arnold.json --buttons $D/repo/docs/cards/arnold/buttons.json --out $D/results_spiderman/idm_aligned_k2 --steps 8000 --window 2 --batch-size 64 > $D/logs/idm_k2.log 2>&1
echo IDM_K2_DONE >> $D/logs/side_queue.log
