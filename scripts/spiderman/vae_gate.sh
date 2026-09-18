#!/bin/bash
# 16-channel VAE ceiling gate (RESEARCH_CONTEXT, Sep 17 11:45 EDT): fine-tune a candidate
# autoencoder's decoder under the recipe that produced our tuned SD 1.x decoder (encoder
# frozen, MSE + 0.1 LPIPS, 50k cached training frames, 100k presentations), then score it,
# its stock decoder and the tuned SD decoder on the same development set and the same
# windows eval_tf.py reports, with a paired episode-level bootstrap.
#
# usage: vae_gate.sh <gpu> <name> <vae_id> <subfolder> <latent_ch> <scaling> <shift>
#   vae_gate.sh 3 flux  Alpha-VLLM/Lumina-Image-2.0             vae 16 0.3611 0.1159
#   vae_gate.sh 3 sd35  stabilityai/stable-diffusion-3.5-medium vae 16 1.5305 0.0609
#
# HF_HOME is deliberately left alone: it relocates the token file, and the gated SD 3.5 repo
# needs the token at its default path. Every download is pinned under $D with --cache-dir
# because the root filesystem is full.
#
# A single-pass batch 32 measured 45.3 GB of this card's 47.4 GB and OOMed, so the effective
# batch of 32 is reached as micro 16 x accum 2 with an NHWC decoder (about 28 GB). The recipe
# numbers that matter, 100k presentations at effective batch 32 and lr 1e-5, are unchanged.
set -u
D=/sata2/data/rnagabhi/doom
GPU=${1:?gpu}; NAME=${2:?name}; VAE=${3:?vae id}; SUB=${4:?subfolder}; CH=${5:?latent channels}; SCALE=${6:?scaling}; SHIFT=${7:?shift}
PY=~/miniconda3/envs/doom/bin/python
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=0 TMPDIR=$D/tmp/tmpdir TORCH_HOME=$D/tmp/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $TMPDIR $TORCH_HOME $D/logs $D/frame_cache $D/results_spiderman/vae_gate_$NAME
cd $D/repo

echo "=== $(date -u) tune $NAME decoder ($VAE/$SUB, ${CH}ch)"
$PY finetune_decoder.py \
  --vae-id "$VAE" --vae-subfolder "$SUB" --cache-dir $D/hf/hub \
  --latent-channels $CH --scaling-factor $SCALE --shift-factor $SHIFT \
  --in-dir $D/raw_arnold --split $D/split_arnold.json --out-dir $D/vae_decoder_${NAME}_lpips \
  --frame-cache $D/frame_cache --train-frames 50000 --val-frames 2000 --stride 4 \
  --epochs 2 --batch-size 16 --accum 2 --channels-last --lr 1e-5 --lpips-weight 0.1 --val-every 500 --device cuda:0
echo "=== tune exit $?"

echo "=== $(date -u) score $NAME"
$PY vae_gate_score.py \
  --decoder stock_sd= \
  --decoder tuned_sd=$D/vae_decoder_arnold_lpips/vae \
  --decoder stock_${CH}ch="$VAE#$SUB" \
  --decoder tuned_${CH}ch=$D/vae_decoder_${NAME}_lpips/vae \
  --baseline tuned_sd --cache-dir $D/hf/hub \
  --dev-in-dir $D/raw_arnold --dev-split $D/split_arnold.json --dev-frames 2000 \
  --frame-cache $D/frame_cache --stride 4 \
  --corpus seen=$D/latents_arnold_eval/seen,$D/raw_arnold_eval/seen,$D/latents_arnold_eval/split_seen.json \
  --corpus unseen=$D/latents_arnold_eval/unseen,$D/raw_arnold_eval/unseen,$D/latents_arnold_eval/split_unseen.json \
  --corpus unseen2=$D/latents_arnold_eval/unseen2,$D/raw_arnold_eval/unseen2,$D/latents_arnold_eval/split_unseen2.json \
  --subset val --num-windows 2048 --context-frames 32 --resamples 1000 --seed 0 \
  --batch-size 32 --device cuda:0 --out-dir $D/results_spiderman/vae_gate_$NAME
echo "=== score exit $?"
echo "=== $(date -u) VAE_GATE_${NAME}_DONE"
