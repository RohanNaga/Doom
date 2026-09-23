#!/bin/bash
# E2: how high the 4-channel reconstruction ceiling goes under GameNGen's own decoder recipe.
#
# Our tuned decoder (vae_decoder_arnold_lpips) saw 50k frames of the 17-map corpus for 100k
# presentations at effective batch 32 with MSE + 0.1 LPIPS, and gives a ceiling of 28.61 dB / 0.051
# LPIPS on the seen corpus and 26.36 / 0.070 on the unseen one. GameNGen trains the decoder with
# MSE alone at batch 2,048 for as many steps as the denoiser, and its footage's stock ceiling is
# only 0.7 dB above ours, so a much higher ceiling should be reachable here. This run changes three
# things at once, deliberately, because that is the recipe being tested: MSE only, frames streamed
# uniformly from the training ids of the dense four-arena corpus instead of a 50k cached sample, and a card-filling
# batch for four GPU-hours instead of 2.5.
#
# The validation frames stay the cached 2,000 of the 17-map corpus, so the loss curve is comparable
# to the earlier tunes, and an hourly checkpoint turns the run into a curve of ceiling against
# presentations rather than one number.
#
#   usage: [TRAIN_IDS=0:2000] decoder_mse.sh <gpu> <micro-batch> <max-steps> [hours]
#   fit:   FIT=1 decoder_mse.sh <gpu> <micro-batch> 60
#
# TRAINING IDS ONLY (docs/REVIEW_2026-09-22.md H4). The stream used to sample row groups from every
# file in raw_arnold_dense/arenas, validation 6000:7000 and test 7000:8000 included, while recording
# `split_subset: "train"`. `--stream-ids $TRAIN_IDS` (default 0:2000, the next-tic runs' own training
# prefix, so the decoder sees no episode the dynamics models did not; 0:6000 is the whole train range) now
# restricts it, `finetune_decoder.py` refuses ids that reach into val or test, and the exact episode
# ids and row groups used are written to provenance.json beside the decoder. The validation frames
# are still the cached 17-map sample: they only measure the decoder and never train it.
set -u
GPU=${1:?gpu}; MB=${2:?micro batch}; STEPS=${3:?max steps}; HOURS=${4:-4.0}
TRAIN_IDS=${TRAIN_IDS:-0:2000}
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
  --stream-dir $D/raw_arnold_dense/arenas --stream-ids "$TRAIN_IDS" --stream-frames 400000 --stream-episodes $EPS \
  --stream-buffer 8192 --workers 8 --seed 0 \
  --max-steps $STEPS $HRS --batch-size $MB --accum 1 --channels-last \
  --lr 1e-5 --lpips-weight 0 --report-lpips --val-every 2000 --device cuda:0
echo "=== tune exit $?"
[ "${FIT:-0}" = 1 ] && { echo "=== FIT_DONE mb$MB"; exit 0; }

# Ceilings, on exactly the frames the stored ceilings were measured on (vae_gate.sh's own
# arguments), in TWO passes. `vae_gate_score.py` writes its metrics.json only when it has scored
# every decoder it was given, so one pass over seven decoders that gets killed at the hand-back
# deadline would leave nothing at all. The headline comparison -- stock, the incumbent, and the
# finished tune -- therefore goes first and on its own, and the hourly curve follows as a second
# pass that can be lost without costing the answer. Both passes carry the baseline so each has its
# own paired bootstrap.
R=$D/results_spiderman/levers_2026-09-20
GATE="--baseline tuned_sd_lpips --cache-dir $D/hf/hub \
  --dev-in-dir $D/raw_arnold --dev-split $D/split_arnold.json --dev-frames 2000 \
  --frame-cache $D/frame_cache --stride 4 \
  --corpus seen=$D/latents_arnold_eval/seen,$D/raw_arnold_eval/seen,$D/latents_arnold_eval/split_seen.json \
  --corpus unseen=$D/latents_arnold_eval/unseen,$D/raw_arnold_eval/unseen,$D/latents_arnold_eval/split_unseen.json \
  --corpus unseen2=$D/latents_arnold_eval/unseen2,$D/raw_arnold_eval/unseen2,$D/latents_arnold_eval/split_unseen2.json \
  --subset val --num-windows 2048 --context-frames 32 --resamples 1000 --seed 0 \
  --batch-size 32 --device cuda:0"
INCUMBENT="--decoder tuned_sd_lpips=$D/vae_decoder_arnold_lpips/vae"

echo "=== $(date -u) score the headline ceilings"
nice -n 15 $PY vae_gate_score.py --decoder stock_sd= $INCUMBENT --decoder mse_final=$OUT/vae \
  $GATE --out-dir $R/e2-decoder
echo "=== headline score exit $?"

CURVE=""
for H in $OUT/vae_h*; do
  [ -d "$H" ] && CURVE="$CURVE --decoder mse_$(basename $H | sed s/vae_//)=$H"
done
if [ -n "$CURVE" ]; then
  echo "=== $(date -u) score the hourly curve"
  nice -n 15 $PY vae_gate_score.py $INCUMBENT $CURVE $GATE --out-dir $R/e2-decoder-curve
  echo "=== curve score exit $?"
fi
echo "=== $(date -u) E2_DECODER_DONE"
