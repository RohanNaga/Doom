#!/bin/bash
# Resume a corrected run from its newest recovery checkpoint if its tmux session died before the end event.
# Usage: resume_run.sh <run> <backbone> <gpu>. Safe to call repeatedly: does nothing while the run is alive or finished.
RUN=$1; BB=$2; GPU=$3; D=/sata2/data/rnagabhi/doom; R=$D/results_spiderman/$RUN
SESSION=$([ "$BB" = dit ] && echo train-dit || echo train-unet)
tmux has-session -t $SESSION 2>/dev/null && { echo "$RUN alive"; exit 0; }
grep -q "\"event\": \"end\"" $R/log.jsonl 2>/dev/null && { echo "$RUN finished"; exit 0; }
CK=$(ls $R/[0-9]*.pt 2>/dev/null | sort | tail -1)
[ -z "$CK" ] && { echo "$RUN dead with no recovery checkpoint"; exit 1; }
COMMON="--context-frames 32 --num-actions 29 --global-batch 32 --lr 5e-5 --warmup 2000 --clip 1.0 --steps 90000 --seed 0 --action-dropout 0.0 --require-verified-transitions --latents-dir $D/latents_arnold_aligned --split $D/split_arnold.json --val-every 1000 --val-windows 1024 --ckpt-every 5000 --snapshot-every 5000 --keep-last 2 --num-workers 4 --grad-ckpt --resume $CK"
if [ "$BB" = dit ]; then ARGS="--backbone dit --per-gpu-batch 32 --warm-start $D/weights/DiT-XL-2-256x256.pt --results-dir $R"; else ARGS="--backbone unet --per-gpu-batch 16 --hf-cache $D/hf/hub --results-dir $R"; fi
echo "$(date -Iseconds) resuming $RUN from $CK on gpu $GPU" >> $D/logs/resumes.log
tmux new-session -d -s $SESSION "cd $D/repo && HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU ~/miniconda3/envs/doom/bin/python train_wm.py $ARGS $COMMON >> $D/logs/train_${BB}_aligned.log 2>&1"
echo "$RUN resumed from $CK"
