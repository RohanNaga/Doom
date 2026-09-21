#!/bin/bash
# E2: how high the 4-channel reconstruction ceiling goes under GameNGen's own decoder recipe.
#
# Our tuned decoder (vae_decoder_arnold_lpips) saw 50k frames of the 17-map corpus for 100k
# presentations at effective batch 32 with MSE + 0.1 LPIPS, and gives a ceiling of 28.61 dB / 0.051
# LPIPS on the seen corpus and 26.36 / 0.070 on the unseen one. GameNGen trains the decoder with
# MSE alone at batch 2,048 for as many steps as the denoiser, and its footage's stock ceiling is
# only 0.7 dB above ours, so a much higher ceiling should be reachable here. This run changes three
# things at once, deliberately, because that is the recipe being tested: MSE only, frames streamed
# uniformly from the dense four-arena corpus instead of a 50k cached sample, and a card-filling
# batch for four GPU-hours instead of 2.5.
#
# The validation frames stay the cached 2,000 of the 17-map corpus, so the loss curve is comparable
# to the earlier tunes, and an hourly checkpoint turns the run into a curve of ceiling against
# presentations rather than one number.
#
#   usage: decoder_mse.sh <gpu> <micro-batch> <max-steps> [hours]
#   fit:   FIT=1 decoder_mse.sh <gpu> <micro-batch> 60
set -u
GPU=${1:?gpu}; MB=${2:?micro batch}; STEPS=${3:?max steps}; HOURS=${4:-4.0}
D=/sata2/data/rnagabhi/doom
REPO=${REPO:-$D/tmp/levers/repo}
PY=${PY:-$HOME/miniconda3/envs/doom/bin/python}
OUT=${OUT:-$D/vae_decoder_sd1x_mse}
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TMPDIR=$D/tmp/tmpdir TORCH_HOME=$D/tmp/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $TMPDIR $TORCH_HOME $OUT $D/logs
cd $REPO

# A fit check indexes 40 episodes so it starts in seconds; the real run indexes 2,000 of the 8,036
# so that 400k frames come from a wide spread of episodes across all four arenas. Measured on the
# real corpus: 1,614 row groups holding 400,209 frames from 1,115 episodes, indexed in 1.6 s, and
# eight workers deliver 4,335 frames/s, two orders of magnitude above what the card consumes, so
# the data path is not the bottleneck and more workers only cost the other jobs their cores.
EPS=2000; VAL=2000; HRS="--max-hours $HOURS --ckpt-every-hours 1"
[ "${FIT:-0}" = 1 ] && { EPS=40; VAL=64; HRS=""; OUT=$D/tmp/levers/fit_mb$MB; mkdir -p $OUT; }

echo "=== $(date -u) mse decoder tune, micro $MB, $STEPS steps, ${HOURS}h, out $OUT"
nice -n 15 $PY finetune_decoder.py \
  --in-dir $D/raw_arnold --split $D/split_arnold.json --frame-cache $D/frame_cache \
  --val-frames $VAL --stride 4 --out-dir $OUT \
  --stream-dir $D/raw_arnold_dense/arenas --stream-frames 400000 --stream-episodes $EPS \
  --stream-buffer 8192 --workers 8 --seed 0 \
  --max-steps $STEPS $HRS --batch-size $MB --accum 1 --channels-last \
  --lr 1e-5 --lpips-weight 0 --report-lpips --val-every 2000 --device cuda:0
echo "=== tune exit $?"
[ "${FIT:-0}" = 1 ] && { echo "=== FIT_DONE mb$MB"; exit 0; }

# Ceiling of the stock decoder, the incumbent tuned one and every hourly checkpoint, on exactly the
# frames the stored ceilings were measured on (vae_gate.sh's own arguments).
DEC="--decoder stock_sd= --decoder tuned_sd_lpips=$D/vae_decoder_arnold_lpips/vae"
for H in $OUT/vae_h*; do
  [ -d "$H" ] && DEC="$DEC --decoder mse_$(basename $H | sed s/vae_//)=$H"
done
DEC="$DEC --decoder mse_final=$OUT/vae"

echo "=== $(date -u) score the ceilings"
nice -n 15 $PY vae_gate_score.py $DEC --baseline tuned_sd_lpips --cache-dir $D/hf/hub \
  --dev-in-dir $D/raw_arnold --dev-split $D/split_arnold.json --dev-frames 2000 \
  --frame-cache $D/frame_cache --stride 4 \
  --corpus seen=$D/latents_arnold_eval/seen,$D/raw_arnold_eval/seen,$D/latents_arnold_eval/split_seen.json \
  --corpus unseen=$D/latents_arnold_eval/unseen,$D/raw_arnold_eval/unseen,$D/latents_arnold_eval/split_unseen.json \
  --corpus unseen2=$D/latents_arnold_eval/unseen2,$D/raw_arnold_eval/unseen2,$D/latents_arnold_eval/split_unseen2.json \
  --subset val --num-windows 2048 --context-frames 32 --resamples 1000 --seed 0 \
  --batch-size 32 --device cuda:0 --out-dir $D/results_spiderman/levers_2026-09-20/e2-decoder
echo "=== score exit $?"
echo "=== $(date -u) E2_DECODER_DONE"
