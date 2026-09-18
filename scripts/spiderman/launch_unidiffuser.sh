#!/bin/bash
# UniDiffuser v1 row (LAION-scale transformer in the SD latent space), identical recipe to 030/031/032/033. Runs once; safe to call repeatedly. Usage: [MB=32] [LR=..] [EXTRA='--skip-grad-norm 5'] launch_unidiffuser.sh <gpu>
GPU=${1:?gpu}; D=/sata2/data/rnagabhi/doom; R=$D/results_spiderman/034-unidiffuser-l32-aligned
export TMPDIR=/sata2/data/rnagabhi/doom/tmp/tmpdir; mkdir -p $TMPDIR
tmux has-session -t train-unidiffuser 2>/dev/null && { echo "unidiffuser alive"; exit 0; }
[ -f $R/log.jsonl ] && grep -q "\"event\": \"end\"" $R/log.jsonl && { echo "unidiffuser finished"; exit 0; }
[ -f $R/log.jsonl ] && { CK=$(ls $R/[0-9]*.pt 2>/dev/null | sort | tail -1); RES=${CK:+--resume $CK}; } || RES=""
[ -d $D/hf/hub/models--thu-ml--unidiffuser-v1/snapshots ] || { echo "unidiffuser weights missing in $D/hf/hub"; exit 1; }
cd $D/repo && git pull -q
COMMON="--context-frames 32 --num-actions 29 --global-batch 32 --lr ${LR:-5e-5} --warmup 2000 --clip 1.0 --steps 90000 --seed 0 --action-dropout 0.0 --require-verified-transitions --latents-dir $D/latents_arnold_aligned --split $D/split_arnold.json --val-every 1000 --val-windows 1024 --ckpt-every 5000 --snapshot-every 5000 --keep-last 2 --num-workers 4 ${EXTRA:-}"
echo "$(date -Iseconds) launching unidiffuser on gpu $GPU $RES" >> $D/logs/resumes.log
tmux new-session -d -s train-unidiffuser "cd $D/repo && TMPDIR=/sata2/data/rnagabhi/doom/tmp/tmpdir HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU ~/wanenc/bin/python train_wm.py --backbone unidiffuser --per-gpu-batch ${MB:-32} --warm-start thu-ml/unidiffuser-v1 --hf-cache $D/hf/hub --results-dir $R $COMMON $RES >> $D/logs/train_unidiffuser.log 2>&1"
echo "unidiffuser launched on gpu $GPU"
