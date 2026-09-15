#!/bin/bash
# PixArt-alpha 512 third row on the corrected data, identical recipe to 030/031/032. Runs once; safe to call repeatedly. Usage: launch_pixart.sh <gpu>
GPU=${1:?gpu}; D=/sata2/data/rnagabhi/doom; R=$D/results_spiderman/033-pixart-l32-aligned
tmux has-session -t train-pixart 2>/dev/null && { echo "pixart alive"; exit 0; }
[ -f $R/log.jsonl ] && grep -q "\"event\": \"end\"" $R/log.jsonl && { echo "pixart finished"; exit 0; }
[ -f $R/log.jsonl ] && { CK=$(ls $R/[0-9]*.pt 2>/dev/null | sort | tail -1); RES=${CK:+--resume $CK}; } || RES=""
[ -d $D/hf/hub/models--PixArt-alpha--PixArt-XL-2-512x512/snapshots ] || { echo "pixart weights missing in $D/hf/hub"; exit 1; }
cd $D/repo && git pull -q
COMMON="--context-frames 32 --num-actions 29 --global-batch 32 --lr 5e-5 --warmup 2000 --clip 1.0 --steps 90000 --seed 0 --action-dropout 0.0 --require-verified-transitions --latents-dir $D/latents_arnold_aligned --split $D/split_arnold.json --val-every 1000 --val-windows 1024 --ckpt-every 5000 --snapshot-every 5000 --keep-last 2 --num-workers 4"
echo "$(date -Iseconds) launching pixart on gpu $GPU $RES" >> $D/logs/resumes.log
tmux new-session -d -s train-pixart "cd $D/repo && HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU ~/miniconda3/envs/doom/bin/python train_wm.py --backbone pixart --per-gpu-batch 32 --warm-start PixArt-alpha/PixArt-XL-2-512x512 --hf-cache $D/hf/hub --results-dir $R $COMMON $RES >> $D/logs/train_pixart.log 2>&1"
echo "pixart launched on gpu $GPU"
