#!/bin/bash
# Second DiT seed on the corrected data, identical recipe, --seed 1. Runs once; safe to call repeatedly. Usage: launch_seed1.sh <gpu>
GPU=${1:?gpu}; D=/sata2/data/rnagabhi/doom; R=$D/results_spiderman/032-dit-l32-aligned-seed1
tmux has-session -t train-dit-seed1 2>/dev/null && { echo "seed1 alive"; exit 0; }
[ -f $R/log.jsonl ] && grep -q "\"event\": \"end\"" $R/log.jsonl && { echo "seed1 finished"; exit 0; }
[ -f $R/log.jsonl ] && { CK=$(ls $R/[0-9]*.pt 2>/dev/null | sort | tail -1); RES=${CK:+--resume $CK}; } || RES=""
cd $D/repo && git pull -q
COMMON="--context-frames 32 --num-actions 29 --global-batch 32 --lr 5e-5 --warmup 2000 --clip 1.0 --steps 90000 --seed 1 --action-dropout 0.0 --require-verified-transitions --latents-dir $D/latents_arnold_aligned --split $D/split_arnold.json --val-every 1000 --val-windows 1024 --ckpt-every 5000 --snapshot-every 5000 --keep-last 2 --num-workers 4 --grad-ckpt"
echo "$(date -Iseconds) launching seed1 on gpu $GPU $RES" >> $D/logs/resumes.log
tmux new-session -d -s train-dit-seed1 "cd $D/repo && CUDA_VISIBLE_DEVICES=$GPU ~/miniconda3/envs/doom/bin/python train_wm.py --backbone dit --per-gpu-batch 32 --warm-start $D/weights/DiT-XL-2-256x256.pt --results-dir $R $COMMON $RES >> $D/logs/train_dit_seed1.log 2>&1"
echo "seed1 launched on gpu $GPU"
