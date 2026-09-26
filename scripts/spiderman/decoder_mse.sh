#!/bin/bash
# E2: how high the reconstruction ceiling goes under GameNGen's own decoder recipe, per latent space.
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
# HELD-OUT VALIDATION (Astra's review, 2026-09-26, section 5). The validation frames are 2,000 frames
# of episodes 6000 to 6099 of raw_arnold_dense/arenas (VAL_IDS, half-open like every id range here):
# the four training arenas' held-out validation episodes, the ones the next-tic rows are read on.
# They are drawn from every tic (--stride 1), because those rows predict every tic, and cached under
# a name of their own that hashes the id list. They used to be the cached 2,000 frames of the 17-map
# corpus, whose maps 1 and 9 to 15 are unseen for the next-tic rows, so reading the hourly
# checkpoints on them was a checkpoint choice made on unseen maps. No choice is made now: the decoder
# is the TERMINAL checkpoint, $OUT/vae, and the hourly checkpoints ($OUT/vae_h<N>) turn the run into
# a curve of ceiling against presentations rather than one number. metrics.json records which of them
# scored best on validation (`checkpoint_selection`), beside the terminal one; nothing reads it to pick.
#
#   usage: [SPACE=sd1|sd35] [TRAIN_IDS=0:2000] [VAL_IDS=6000:6100] [OUT=dir] [PY=..] [REPO=..] \
#          [CURVE=1] [DRY=1] [DOOM_ROOT=..] decoder_mse.sh <gpu> <micro-batch> <max-steps> [hours]
#   fit:   FIT=1 [SPACE=..] decoder_mse.sh <gpu> <micro-batch> 60
#
# THE LATENT SPACE (Astra's review, 2026-09-26, section 5). SPACE names the autoencoder whose decoder
# is tuned, and everything that follows from it. finetune_decoder.py is given the autoencoder's
# identity and its latent contract explicitly and asserts the contract against the loaded config, so
# a tune can no longer run on sd-vae-ft-mse while its output directory says SD 3.5, which is what
# this launcher did before it passed any of these flags.
#
#   sd1  (default)  stabilityai/sd-vae-ft-mse, 4 channels, scale 0.18215, no shift: the decoder of the
#                   U-Net and PixArt rows. ~/miniconda3/envs/doom; out $D/vae_decoder_sd1x_mse. Loads
#                   from the default Hugging Face cache, where every earlier SD 1.x tune found it.
#   sd35            stabilityai/stable-diffusion-3.5-medium, subfolder vae, 16 channels, scale 1.5305,
#                   shift 0.0609: the SD 3.5 row's decoder. ~/wanenc (the SD 3.5 row's interpreter);
#                   out $D/vae_decoder_sd35_mse. Loads from $D/hf/hub, where vae_gate.sh put it.
#
# PY and OUT override the space's interpreter and output directory. DRY=1 prints the commands and
# stops before every side effect; DOOM_ROOT repoints the data root.
#
# EXIT CODES. Each stage's exit code is checked, not only echoed: a failed tune, a tune that exits 0
# without leaving weights in $OUT/vae, a failed headline score or a failed curve ends the launcher with
# that stage's code and an E2_DECODER_FAILED line, and E2_DECODER_DONE (FIT_DONE for a fit) is printed
# only when every stage asked for succeeded. It used to echo each code and carry on, so a crashed tune
# still went on to the gate and ended in E2_DECODER_DONE with exit 0.
#
# TRAINING IDS ONLY (docs/REVIEW_2026-09-22.md H4). The stream used to sample row groups from every
# file in raw_arnold_dense/arenas, validation 6000:7000 and test 7000:8000 included, while recording
# `split_subset: "train"`. `--stream-ids $TRAIN_IDS` (default 0:2000, the next-tic runs' own training
# prefix, so the decoder sees no episode the dynamics models did not; 0:6000 is the whole train range) now
# restricts it, `finetune_decoder.py` refuses ids that reach into val or test, and the exact episode
# ids and row groups used are written to provenance.json beside the decoder. The validation frames
# only measure the decoder and never train it, and finetune_decoder.py refuses validation ids outside
# the arenas validation range (6000:7000).
set -u
set -o pipefail
GPU=${1:?gpu}; MB=${2:?micro batch}; STEPS=${3:?max steps}; HOURS=${4:-4.0}
SPACE=${SPACE:-sd1}
TRAIN_IDS=${TRAIN_IDS:-0:2000}
VAL_IDS=${VAL_IDS:-6000:6100}
DRY=${DRY:-0}
D=${DOOM_ROOT:-/sata2/data/rnagabhi/doom}
REPO=${REPO:-$D/tmp/levers/repo}
case $SPACE in
  sd1)
    PY=${PY:-$HOME/miniconda3/envs/doom/bin/python}; OUT=${OUT:-$D/vae_decoder_sd1x_mse}
    STOCK=stock_sd; STOCK_PATH=""        # vae_gate_score.py: an empty path is sd-vae-ft-mse (load_vae)
    VAE=(--vae-id stabilityai/sd-vae-ft-mse --latent-channels 4 --scaling-factor 0.18215) ;;
  sd35)
    PY=${PY:-$HOME/wanenc/bin/python}; OUT=${OUT:-$D/vae_decoder_sd35_mse}
    STOCK=stock_sd35; STOCK_PATH="stabilityai/stable-diffusion-3.5-medium#vae"
    VAE=(--vae-id stabilityai/stable-diffusion-3.5-medium --vae-subfolder vae --cache-dir "$D/hf/hub"
         --latent-channels 16 --scaling-factor 1.5305 --shift-factor 0.0609) ;;
  *) echo "unknown SPACE '$SPACE' (sd1 | sd35)" >&2; exit 2 ;;
esac

# A fit check indexes 40 episodes so it starts in seconds; the real run indexes 2,000 of the 8,036
# so that 400k frames come from a wide spread of episodes across all four arenas. Measured on the
# real corpus: 1,614 row groups holding 400,209 frames from 1,115 episodes, indexed in 1.6 s, and
# eight workers deliver 4,335 frames/s, two orders of magnitude above what the card consumes, so
# the data path is not the bottleneck and more workers only cost the other jobs their cores.
EPS=2000; VAL=2000; HRS=(--max-hours "$HOURS" --ckpt-every-hours 1)
[ "${FIT:-0}" = 1 ] && { EPS=40; VAL=64; HRS=(); OUT=$D/tmp/levers/fit_${SPACE}_mb$MB; }

TUNE=("$PY" finetune_decoder.py "${VAE[@]}"
  --val-dir "$D/raw_arnold_dense/arenas" --val-ids "$VAL_IDS" --frame-cache "$D/frame_cache"
  --val-frames "$VAL" --stride 1 --out-dir "$OUT"
  --stream-dir "$D/raw_arnold_dense/arenas" --stream-ids "$TRAIN_IDS" --stream-frames 400000 --stream-episodes "$EPS"
  --stream-buffer 8192 --workers 8 --seed 0
  --max-steps "$STEPS" ${HRS[@]+"${HRS[@]}"} --batch-size "$MB" --accum 1 --channels-last
  --lr 1e-5 --lpips-weight 0 --report-lpips --val-every 2000 --device cuda:0)

# Ceilings, STOCK AGAINST TUNED in this space (Astra's review, 2026-09-26, section 5): the paired
# bootstrap subtracts this space's stock decoder (stock_sd, sd-vae-ft-mse; stock_sd35, SD 3.5's own
# autoencoder), where it used to subtract the earlier LPIPS-tuned SD 1.x decoder, on exactly the frames
# the stored ceilings were measured on (vae_gate.sh's own arguments). `vae_gate_score.py` writes its
# metrics.json only when it has scored every decoder it was given, so the headline pair goes first and
# on its own. CURVE=1 adds a second pass over the hourly checkpoints against the same baseline, into its
# own directory. It is off by default: scoring every hourly checkpoint on the seen and unseen corpora
# invites choosing among them there, and the curve of ceiling against presentations is already in
# metrics.json, read on the held-out validation episodes.
R=$D/results_spiderman/levers_2026-09-20
GATE=(--baseline "$STOCK" --cache-dir "$D/hf/hub"
  --dev-in-dir "$D/raw_arnold" --dev-split "$D/split_arnold.json" --dev-frames 2000
  --frame-cache "$D/frame_cache" --stride 4
  --corpus "seen=$D/latents_arnold_eval/seen,$D/raw_arnold_eval/seen,$D/latents_arnold_eval/split_seen.json"
  --corpus "unseen=$D/latents_arnold_eval/unseen,$D/raw_arnold_eval/unseen,$D/latents_arnold_eval/split_unseen.json"
  --corpus "unseen2=$D/latents_arnold_eval/unseen2,$D/raw_arnold_eval/unseen2,$D/latents_arnold_eval/split_unseen2.json"
  --subset val --num-windows 2048 --context-frames 32 --resamples 1000 --seed 0
  --batch-size 32 --device cuda:0)
HEADLINE=("$PY" vae_gate_score.py --decoder "$STOCK=$STOCK_PATH" --decoder "mse_final=$OUT/vae"
  "${GATE[@]}" --out-dir "$R/e2-decoder-$SPACE")

if [ "$DRY" = 1 ]; then
  echo "DRY tune cd $REPO && CUDA_VISIBLE_DEVICES=$GPU ${TUNE[*]}"
  [ "${FIT:-0}" = 1 ] || echo "DRY headline ${HEADLINE[*]}"
  [ "${FIT:-0}" != 1 ] && [ "${CURVE:-0}" = 1 ] && echo "DRY curve (after the tune) $PY vae_gate_score.py" \
    "--decoder $STOCK=$STOCK_PATH --decoder mse_h<N>=$OUT/vae_h<N> ... ${GATE[*]} --out-dir $R/e2-decoder-curve-$SPACE"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TMPDIR=$D/tmp/tmpdir TORCH_HOME=$D/tmp/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$TORCH_HOME" "$OUT" "$D/logs"
cd "$REPO" || exit 1
die() {   # die <exit code> <what failed>
  echo "=== $(date -u) E2_DECODER_FAILED ($SPACE): $2"
  exit "$1"
}

echo "=== $(date -u) mse decoder tune, space $SPACE, micro $MB, $STEPS steps, ${HOURS}h, out $OUT"
nice -n 15 "${TUNE[@]}"
RC=$?
echo "=== tune exit $RC"
[ "$RC" -eq 0 ] || die "$RC" "the tune exited $RC"
[ -f "$OUT/vae/diffusion_pytorch_model.safetensors" ] || die 1 "the tune exited 0 but left no weights in $OUT/vae"
[ "${FIT:-0}" = 1 ] && { echo "=== FIT_DONE $SPACE mb$MB"; exit 0; }

echo "=== $(date -u) score the headline ceilings"
nice -n 15 "${HEADLINE[@]}"
RC=$?
echo "=== headline score exit $RC"
[ "$RC" -eq 0 ] || die "$RC" "the headline score exited $RC"

HOURLY=()
for H in "$OUT"/vae_h*; do
  [ -d "$H" ] && HOURLY+=(--decoder "mse_$(basename "$H" | sed s/vae_//)=$H")
done
if [ "${CURVE:-0}" = 1 ] && [ ${#HOURLY[@]} -gt 0 ]; then
  echo "=== $(date -u) score the hourly curve"
  nice -n 15 "$PY" vae_gate_score.py --decoder "$STOCK=$STOCK_PATH" "${HOURLY[@]}" "${GATE[@]}" \
    --out-dir "$R/e2-decoder-curve-$SPACE"
  RC=$?
  echo "=== curve score exit $RC"
  [ "$RC" -eq 0 ] || die "$RC" "the curve score exited $RC (the headline in $R/e2-decoder-$SPACE is complete)"
fi
echo "=== $(date -u) E2_DECODER_DONE"
