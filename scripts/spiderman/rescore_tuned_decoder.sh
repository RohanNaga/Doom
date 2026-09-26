#!/bin/bash
# The tuned-decoder column: three rows' EMA weights rescored with the MSE-tuned decoders, where only
# the decoder changes (Astra's review, 2026-09-26, section 5, item 5).
#
#   usage: [UNET_STEP=200000] [SD35_STEP=<newest snapshot>] [PIXART_STEP=<newest snapshot>] \
#          [SD1_DECODER=$D/vae_decoder_sd1x_mse/vae] [SD35_DECODER=$D/vae_decoder_sd35_mse/vae] \
#          [PY_UNET=..] [PY_SD35=..] [UNET_REPO=..] [SD35_REPO=..] [PIXART_REPO=..] [FORCE=1] \
#          [DRY=1] [DOOM_ROOT=..] rescore_tuned_decoder.sh <gpu>
#
# Six reads, eval_tf.py on 512 val windows at horizons 1 and 4, of
#
#   040-unet-nexttic    snap_0200000.pt (the final read)  SD 1.x tuned decoder   $UNET_REPO   (repo_launch)
#   042-sd35-nexttic    the newest snap_*.pt              SD 3.5 tuned decoder   $SD35_REPO   (repo_launch)
#   041-pixart-nexttic  the newest snap_*.pt              SD 1.x tuned decoder   $PIXART_REPO (repo_launch2)
#
# into $D/results_spiderman/<run>/steward_<step>/tf_ema_h<K>_tuneddec/. Each command is the gate-5
# readback (scripts/cluster/gates.sh readback_cmd) with --use-ema and --num-windows 512, i.e. the
# command behind every stock-decoder EMA read of these rows, with only the decoder flags of
# eval_tf.py:417-420 changed:
#
#   SD 1.x rows   --vae-path $SD1_DECODER --latent-scale 0.18215                      (stock: the defaults)
#   SD 3.5 row    --vae-path $SD35_DECODER --latent-scale 1.5305 --latent-shift 0.0609
#                 (stock: --vae-path stabilityai/stable-diffusion-3.5-medium --vae-subfolder vae and the
#                 same scale and shift; the tuned decoder is a plain AutoencoderKL directory, so it has no
#                 subfolder)
#
# The scale and shift are the ones the corpus was encoded with: the tune froze the encoder, so the
# latents are the stock encoder's whatever decoder reads them. The windows, the noise, the sampler and
# the weights are the stock read's, so each tuned read pairs with the stock read of the same step: for
# the U-Net and SD 3.5, steward_<step>/tf_ema_h<K>; for PixArt, the trainer's eval_<step>/tf_ema_h<K>.
# Each row runs from the checkout its stock reads came from, so the sampling code is the same too.
#
# The tuned decoders are the TERMINAL checkpoints of decoder_mse.sh (<out>/vae); their validation
# frames are episodes 6000 to 6099, the very episodes scored here, and no checkpoint was chosen on them.
#
# No `--wandb-run`, on purpose: eval_tf.py would append these reads to `<run>-eval` under the same
# `eval/ema_h<K>` keys as the stock reads at the same step, and overwrite them.
#
# A read whose metrics.json exists is skipped (FORCE=1 reruns it). A missing checkpoint or decoder, or a
# failed read, fails only that read; the script runs the rest and exits 1 if any failed. DRY=1 prints the
# six commands and stops before every side effect; DOOM_ROOT repoints the data root.
set -u
set -o pipefail
GPU=${1:?gpu}
D=${DOOM_ROOT:-/sata2/data/rnagabhi/doom}
DRY=${DRY:-0}
FORCE=${FORCE:-0}
SD1_DECODER=${SD1_DECODER:-$D/vae_decoder_sd1x_mse/vae}
SD35_DECODER=${SD35_DECODER:-$D/vae_decoder_sd35_mse/vae}
PY_UNET=${PY_UNET:-$HOME/miniconda3/envs/doom/bin/python}
PY_SD35=${PY_SD35:-$HOME/wanenc/bin/python}
UNET_REPO=${UNET_REPO:-$D/repo_launch}
SD35_REPO=${SD35_REPO:-$D/repo_launch}
PIXART_REPO=${PIXART_REPO:-$D/repo_launch2}
FAILED=0

fail() { echo "$(date -Iseconds) FAILED $*" >&2; FAILED=1; }

newest_snapshot_step() {   # newest_snapshot_step <results dir>: the step of its newest snap_*.pt, or nothing
  local F
  F=$(ls "$1"/snap_[0-9]*.pt 2>/dev/null | sort | tail -1)
  [ -n "$F" ] || return 1
  F=$(basename "$F" .pt); echo $((10#${F#snap_}))
}

has_weights() {   # has_weights <decoder dir>: the tuned decoder finished saving
  [ -f "$1/diffusion_pytorch_model.safetensors" ] || [ -f "$1/diffusion_pytorch_model.bin" ]
}

# row <run> <backbone> <step or empty for the newest snapshot> <repo> <python> <decoder dir> <scale> [shift]
row() {
  local RUN=$1 BB=$2 STEP=$3 REPO=$4 PY=$5 DEC=$6 SCALE=$7 SHIFT=${8:-}
  local R=$D/results_spiderman/$RUN CH=4 S="" SRC K CK OUT
  case $BB in
    unet)   SRC=(--sd-path CompVis/stable-diffusion-v1-4) ;;
    pixart) SRC=(--pixart-path PixArt-alpha/PixArt-XL-2-512x512) ;;
    sd35)   SRC=(--sd35-path stabilityai/stable-diffusion-3.5-medium); CH=16; S=_sd35 ;;
  esac
  if [ -z "$STEP" ] && ! STEP=$(newest_snapshot_step "$R"); then
    # a dry run on a machine without the runs still shows the command, with the step left open
    [ "$DRY" = 1 ] || { fail "$RUN: no snap_*.pt under $R"; return 0; }
    STEP="<newest>"
  fi
  if [ "$STEP" = "<newest>" ]; then CK=$R/snap_$STEP.pt; else CK=$R/snap_$(printf '%07d' "$STEP").pt; fi
  local DECODER=(--vae-path "$DEC" --latent-scale "$SCALE")
  [ -n "$SHIFT" ] && DECODER+=(--latent-shift "$SHIFT")
  for K in 1 4; do
    OUT=$R/steward_$STEP/tf_ema_h${K}_tuneddec
    local CMD=("$PY" "$REPO/eval_tf.py" --backbone "$BB" --latent-channels "$CH"
      --ckpt "$CK" --use-ema --tic-stride 1 --horizon-tics "$K"
      --latents-dir "$D/latents_arnold_dense_pertic_eval$S/val"
      --split "$D/latents_arnold_dense_pertic_eval$S/split_val.json" --subset val
      --parquet-dir "$D/raw_arnold_dense/arenas" --num-windows 512 --batch-size 16 --steps 10
      --context-frames 32 --num-actions 29 --hf-cache "$D/hf/hub" "${SRC[@]}" "${DECODER[@]}"
      --out-dir "$OUT")
    if [ "$DRY" = 1 ]; then
      echo "DRY $RUN h$K CUDA_VISIBLE_DEVICES=$GPU ${CMD[*]}"
      continue
    fi
    if [ -f "$OUT/metrics.json" ] && [ "$FORCE" != 1 ]; then
      echo "$(date -Iseconds) $RUN h$K already scored in $OUT (FORCE=1 reruns it)"; continue
    fi
    [ -f "$CK" ] || { fail "$RUN h$K: no checkpoint $CK"; continue; }
    has_weights "$DEC" || { fail "$RUN h$K: no decoder weights in $DEC (has decoder_mse.sh finished?)"; continue; }
    mkdir -p "$OUT"
    echo "$(date -Iseconds) start $RUN h$K step $STEP -> $OUT"
    CUDA_VISIBLE_DEVICES=$GPU "${CMD[@]}" > "$OUT/eval_tf.log" 2>&1 < /dev/null
    local E=$?
    if [ $E -ne 0 ] || [ ! -f "$OUT/metrics.json" ]; then fail "$RUN h$K: eval_tf.py exit $E (see $OUT/eval_tf.log)"
    else echo "$(date -Iseconds) done $RUN h$K"; fi
  done
}

[ "$DRY" = 1 ] || { export TMPDIR=$D/tmp/tmpdir; mkdir -p "$TMPDIR"; }
row 040-unet-nexttic   unet   "${UNET_STEP:-200000}" "$UNET_REPO"   "$PY_UNET" "$SD1_DECODER"  0.18215
row 042-sd35-nexttic   sd35   "${SD35_STEP:-}"       "$SD35_REPO"   "$PY_SD35" "$SD35_DECODER" 1.5305 0.0609
row 041-pixart-nexttic pixart "${PIXART_STEP:-}"     "$PIXART_REPO" "$PY_UNET" "$SD1_DECODER"  0.18215
[ "$DRY" = 1 ] && exit 0
if [ "$FAILED" = 1 ]; then echo "$(date -Iseconds) RESCORE_FAILED" >&2; exit 1; fi
echo "$(date -Iseconds) RESCORE_DONE"
