#!/bin/bash
# The matched perceptual decoder tune of the paper: MSE + 0.1 LPIPS, decoder only, and its headline gate.
# decoder_mse.sh is the GameNGen-recipe (MSE-only, --lpips-weight 0) launcher; this file is the exact command pair that produced
# vae_decoder_sd1x_mse_lpips on Spiderman (Sep 27 2026), and the one the sd35 row runs with.
#
# sd1 run (vae_decoder_sd1x_mse_lpips):
#   code: repo_tools 0dc9d85 (finetune_decoder.py, vae_gate_score.py at that commit)
#   train data: streamed raw frames of raw_arnold_dense/arenas, episode ids 0:2000 = 1133 episodes, maps 2-5, 400,059 frames
#   val data: 2000 frames of raw_arnold_dense/arenas episodes 6000:6100 (frame cache arenas_valids_6000-6099_100ep_48117814_2000_s1.npy)
#   steps: 3486 = 14,400 s x (60 updates / 247.8 s), from a 60-step fit at the same settings; micro-batch 24, no accumulation, lr 1e-5,
#          channels-last, peak 38.3 GB on an A6000 (GPU 1); 4.0 h wall clock, hourly checkpoints vae_h1..vae_h3 kept
#   started 2026-09-27 06:22Z, finished 09:10Z; gate 09:10-09:25Z
#   gate: results_spiderman/levers_2026-09-20/e2-decoder-sd1-lpips/metrics.json (stock_sd baseline, 1000 bootstrap resamples):
#         dev +4.33 dB PSNR, -0.039 LPIPS; unseen +3.22 dB
# sd35 run (vae_decoder_sd35_mse_lpips): same recipe, 4880 steps (60 updates / 177.0 s), started 2026-09-27 14:04Z,
#   gate out dir results_spiderman/levers_2026-09-20/e2-decoder-sd35-lpips.
#
# usage: bash scripts/spiderman/decoder_mse_lpips.sh [sd1|sd35]   (tune, then gate; run from anywhere)
set -e
SPACE=${1:-sd1}
D=/sata2/data/rnagabhi/doom
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} HF_HUB_OFFLINE=1 TMPDIR=$D/tmp/tmpdir TORCH_HOME=$D/tmp/torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $D/repo_tools
COMMON="--val-dir $D/raw_arnold_dense/arenas --val-ids 6000:6100 --frame-cache $D/frame_cache --val-frames 2000 --stride 1 \
  --stream-dir $D/raw_arnold_dense/arenas --stream-ids 0:2000 --stream-frames 400000 --stream-episodes 2000 --stream-buffer 8192 --workers 8 --seed 0 \
  --max-hours 4.0 --ckpt-every-hours 1 --batch-size 24 --accum 1 --channels-last --lr 1e-5 --lpips-weight 0.1 --report-lpips --val-every 2000 --device cuda:0"
GATE_COMMON="--cache-dir $D/hf/hub --dev-in-dir $D/raw_arnold --dev-split $D/split_arnold.json --dev-frames 2000 --frame-cache $D/frame_cache --stride 4 \
  --corpus seen=$D/latents_arnold_eval/seen,$D/raw_arnold_eval/seen,$D/latents_arnold_eval/split_seen.json \
  --corpus unseen=$D/latents_arnold_eval/unseen,$D/raw_arnold_eval/unseen,$D/latents_arnold_eval/split_unseen.json \
  --corpus unseen2=$D/latents_arnold_eval/unseen2,$D/raw_arnold_eval/unseen2,$D/latents_arnold_eval/split_unseen2.json \
  --subset val --num-windows 2048 --context-frames 32 --resamples 1000 --seed 0 --batch-size 32 --device cuda:0"
case $SPACE in
  sd1)  PY=$HOME/miniconda3/envs/doom/bin/python; OUT=$D/vae_decoder_sd1x_mse_lpips; STEPS=3486
        VAE="--vae-id stabilityai/sd-vae-ft-mse --latent-channels 4 --scaling-factor 0.18215"; STOCK="stock_sd="; BASE=stock_sd ;;
  sd35) PY=$HOME/wanenc/bin/python; OUT=$D/vae_decoder_sd35_mse_lpips; STEPS=4880
        VAE="--vae-id stabilityai/stable-diffusion-3.5-medium --vae-subfolder vae --cache-dir $D/hf/hub --latent-channels 16 --scaling-factor 1.5305 --shift-factor 0.0609"
        STOCK="stock_sd35=stabilityai/stable-diffusion-3.5-medium#vae"; BASE=stock_sd35 ;;
  *) echo "usage: $0 [sd1|sd35]"; exit 2 ;;
esac
$PY finetune_decoder.py $VAE $COMMON --out-dir $OUT --max-steps $STEPS
$PY vae_gate_score.py --decoder $STOCK --decoder mse_lpips_final=$OUT/vae --baseline $BASE $GATE_COMMON \
  --out-dir $D/results_spiderman/levers_2026-09-20/e2-decoder-$SPACE-lpips
