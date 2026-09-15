#!/bin/bash
# NOT LAUNCHED. Corrected paired runs on verified-transition latents, one A6000 each, results on the data drive.
# Shared recipe (Astra round 3, pending Rohan): lr 5e-5, warmup 2000, global batch 32, L=32, action dropout 0, 90k steps.
# Usage: bash launch_aligned_spiderman.sh <dit_gpu> <unet_gpu>
DG=${1:?dit gpu}; UG=${2:?unet gpu}
D=/sata2/data/rnagabhi/doom; cd $D/repo && git pull -q
COMMON="--context-frames 32 --num-actions 29 --global-batch 32 --lr 5e-5 --warmup 2000 --clip 1.0 --steps 90000 --seed 0 --action-dropout 0.0 --require-verified-transitions --latents-dir $D/latents_arnold_aligned --split $D/split_arnold.json --val-every 1000 --val-windows 1024 --ckpt-every 5000 --snapshot-every 5000 --keep-last 2 --num-workers 4"
tmux new-session -d -s train-dit "cd $D/repo && CUDA_VISIBLE_DEVICES=$DG ~/miniconda3/envs/doom/bin/python train_wm.py --backbone dit --per-gpu-batch 32 --warm-start $D/weights/DiT-XL-2-256x256.pt --results-dir $D/results_spiderman/030-dit-l32-aligned $COMMON > $D/logs/train_dit_aligned.log 2>&1"
tmux new-session -d -s train-unet "cd $D/repo && HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$UG ~/miniconda3/envs/doom/bin/python train_wm.py --backbone unet --per-gpu-batch 16 --hf-cache $D/hf/hub --results-dir $D/results_spiderman/031-unet-l32-aligned $COMMON > $D/logs/train_unet_aligned.log 2>&1"
tmux ls
