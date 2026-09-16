#!/bin/bash
# SkyReels-V2 DF 1.3B video-prior row (050), agreed design of Sep 14 night. Runs once; safe to call repeatedly; resumes from the last recovery checkpoint.
# Usage: launch_video.sh <gpu> <context_frames> <objective>
GPU=${1:?gpu}; L=${2:?context}; OBJ=${3:-flow}; D=/sata2/data/rnagabhi/doom; R=$D/results_spiderman/050-skyreels-l${L}-${OBJ}
export TMPDIR=/sata2/data/rnagabhi/doom/tmp/tmpdir; mkdir -p $TMPDIR
tmux has-session -t train-video 2>/dev/null && { echo "video alive"; exit 0; }
[ -f $R/log.jsonl ] && grep -q "\"event\": \"end\"" $R/log.jsonl && { echo "video finished"; exit 0; }
[ -f $R/log.jsonl ] && { CK=$(ls $R/[0-9]*.pt 2>/dev/null | sort | tail -1); RES=${CK:+--resume $CK}; } || RES=""
[ -f $D/weights/skyreels_null_prompt.pt ] || { echo "null prompt missing"; exit 1; }
MB=${MB:-8}; CKPT=${CKPT:-1}; CK=$([ "$CKPT" = 1 ] && echo --grad-ckpt || echo --no-grad-ckpt)
COMMON="--context-frames $L --num-actions 29 --global-batch 8 --per-gpu-batch $MB $CK --lr 2e-5 --warmup 500 --clip 1.0 --steps 10000 --seed 0 --action-dropout 0.0 --objective $OBJ --null-prompt $D/weights/skyreels_null_prompt.pt --latents-dir $D/latents_arnold_wan --split $D/split_arnold.json --hf-cache $D/hf/hub --val-every 500 --val-windows 512 --ckpt-every 1000 --snapshot-every 5000 --keep-last 2 --num-workers 4"
echo "$(date -Iseconds) launching video row on gpu $GPU L=$L $OBJ microbatch $MB ckpt $CKPT $RES" >> $D/logs/resumes.log
tmux new-session -d -s train-video "cd $D/repo && TMPDIR=/sata2/data/rnagabhi/doom/tmp/tmpdir HF_HUB_OFFLINE=0 CUDA_VISIBLE_DEVICES=$GPU $([ "${GPU//[^,]/}" ] && echo "~/wanenc/bin/accelerate launch --num_processes $(( $(echo $GPU | tr -cd , | wc -c) + 1 )) --mixed_precision bf16" || echo ~/wanenc/bin/python) train_video.py --results-dir $R $COMMON $RES >> $D/logs/train_video.log 2>&1"
echo "video launched on gpu $GPU L=$L $OBJ"
