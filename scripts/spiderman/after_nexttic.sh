#!/bin/bash
# Evaluation suite for a next-tic row. Same protocol after_sd35.sh runs for the stride-4 rows, at
# tic spacing, plus the two things a next-tic number needs to be readable:
#
#   --tic-stride 1      the per-tic window set and the per-tic copy-last floor
#   --horizon-tics 4    four tics rolled forward from real context, scored at the 4th frame, so the
#                       number is at EQUAL GAME TIME with a stride-4 model's single step (114 ms)
#
#   usage: [PY=..] [NOWAIT=1] [CORPORA=".."] [RESCORE=1] [DRY=1] [DOOM_ROOT=..] \
#          after_nexttic.sh <gpu> <unet | sd35 | pixart>
#
# Corpora, in the order they are scored. The first three are the dense corpus's own held-out sets
# from release/dense_split.json; the last three are the original 17-map evaluation corpora, kept as
# additional per-tic rows and SKIPPED CLEANLY when their per-tic latents do not exist yet:
#
#   val        arenas ids 6000-6099   25/arena, the in-training validation corpus
#   test       arenas ids 7000-7099   25/arena, the sealed seen-map row
#   arenas_678 ids 0-59               20/arena, the unseen-map row
#   seen / unseen / unseen2           the stride-4 rows' corpora, re-encoded per tic
#
# Rollouts are given in TICS: horizon 256 = the 64 decision steps today's rows roll out, i.e. the
# same 7.31 seconds of game time. FVD clips are written at both every tic and every fourth tic, so
# an FVD against a stride-4 row compares clips of the same game duration.
set -u
GPU=${1:?gpu}
BACKBONE=${2:?backbone (unet | sd35 | pixart)}
D=${DOOM_ROOT:-/sata2/data/rnagabhi/doom}
DRY=${DRY:-0}
HORIZON=${HORIZON:-256}
NUM_WINDOWS=${NUM_WINDOWS:-2048}
CORPORA=${CORPORA:-"val test arenas_678 seen unseen unseen2"}

case $BACKBONE in
  unet)   RUN=040-unet-nexttic;   CH=4;  BB="--backbone unet --sd-path CompVis/stable-diffusion-v1-4" ;;
  pixart) RUN=041-pixart-nexttic; CH=4;  BB="--backbone pixart --pixart-path PixArt-alpha/PixArt-XL-2-512x512" ;;
  sd35)   RUN=042-sd35-nexttic;   CH=16; BB="--backbone sd35 --sd35-path stabilityai/stable-diffusion-3.5-medium" ;;
  *) echo "unknown backbone '$BACKBONE' (unet | sd35 | pixart)" >&2; exit 2 ;;
esac
R=$D/results_spiderman/$RUN
PY=${PY:-$HOME/miniconda3/envs/doom/bin/python}
[ "$CH" = 16 ] && PY=${PY_SD35:-$HOME/wanenc/bin/python}

# per-tic latents and raw parquet per corpus; the dense ones and the original ones live apart
if [ "$CH" = 16 ]; then LE=$D/latents_arnold_dense_pertic_eval_sd35; LO=$D/latents_arnold_eval_pertic_sd35
else LE=$D/latents_arnold_dense_pertic_eval; LO=$D/latents_arnold_eval_pertic; fi
corpus_args() {
  case $1 in
    val|test)   echo "--latents-dir $LE/$1 --parquet-dir $D/raw_arnold_dense/arenas --split $LE/split_$1.json" ;;
    arenas_678) echo "--latents-dir $LE/arenas_678 --parquet-dir $D/raw_arnold_dense/arenas_678 --split $LE/split_arenas_678.json" ;;
    seen|unseen|unseen2) echo "--latents-dir $LO/$1 --parquet-dir $D/raw_arnold_eval/$1 --split $LO/split_$1.json" ;;
  esac
}

# The tuned decoder, or the stock one. Tested with one `[ -f a ] || [ -f b ]` per file: a single
# `ls` of two paths reports success when only one exists, which is how the SD 3.5 row silently
# scored against the stock decoder on Sep 20 2026.
if [ "$CH" = 16 ]; then TUNED=$D/vae_decoder_sd35_lpips/vae; STOCK="--vae-path stabilityai/stable-diffusion-3.5-medium --vae-subfolder vae"
else TUNED=$D/vae_decoder_arnold_lpips/vae; STOCK=""; fi
if [ -f "$TUNED/diffusion_pytorch_model.safetensors" ] || [ -f "$TUNED/diffusion_pytorch_model.bin" ]; then
  VAE="--vae-path $TUNED"; USED="tuned: $TUNED"
else
  VAE="$STOCK"; USED="stock fallback (the tuned decoder at $TUNED has no weights)"
fi
SCALE=""
[ "$CH" = 16 ] && SCALE="--latent-scale 1.5305 --latent-shift 0.0609"

# the conditioning interface is read from the checkpoint, never re-specified here: a mismatch
# between training and evaluation would produce a number nobody could interpret
COMMON="$BB --latent-channels $CH $VAE $SCALE --hf-cache $D/hf/hub --tic-stride 1 \
 --context-frames 32 --num-actions 29"

if [ "$DRY" = 1 ]; then
  echo "DRY $RUN decoder: $USED"
  for S in $CORPORA; do echo "DRY eval_tf $S $PY eval_tf.py $COMMON $(corpus_args $S) --ckpt $R/best.pt --horizon-tics 1 --out-dir $R/eval_tf_$S"; done
  echo "DRY eval_tf_h4 $PY eval_tf.py $COMMON $(corpus_args val) --ckpt $R/best.pt --horizon-tics 4 --out-dir $R/eval_tf_val_h4"
  echo "DRY rollout $PY rollout_eval.py --rollout $COMMON --ckpt $R/best.pt $(corpus_args test) --num-rollouts 256 --horizon $HORIZON --out $R/rollouts_test.npz"
  exit 0
fi

export TMPDIR=$D/tmp/tmpdir; mkdir -p $TMPDIR $R $D/logs
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
[ "${NOWAIT:-0}" = 1 ] || until grep -q "\"event\": \"end\"" $R/log.jsonl 2>/dev/null; do sleep 300; done
cd $D/repo && git pull -q
echo "$(date -Iseconds) decoder_used $USED" | tee -a $R/decoder_used.txt

# newest snapshot or recovery checkpoint carries the EMA; best.pt never does
LAST=$(ls $R/[0-9]*.pt $R/snap_*.pt 2>/dev/null | sort | tail -1)

TF="--subset val --num-windows $NUM_WINDOWS --batch-size 16 --steps 50"
for S in $CORPORA; do
  ARGS=$(corpus_args $S)
  LAT=$(echo "$ARGS" | sed -n 's/.*--latents-dir \([^ ]*\).*/\1/p')
  [ -d "$LAT" ] || { echo "$RUN skipping $S: no per-tic latents at $LAT" | tee -a $D/logs/${RUN}_eval.log; continue; }
  for VARIANT in "" "_ema"; do
    CK=$R/best.pt; EMA=""
    [ -n "$VARIANT" ] && { CK=$LAST; EMA="--use-ema"; }
    [ -n "$CK" ] || continue
    OUT=$R/eval_tf_${S}${VARIANT}
    [ -f "$OUT/metrics.json" ] && { echo "$RUN eval_tf ${S}${VARIANT} already scored" >> $D/logs/${RUN}_eval.log; continue; }
    $PY eval_tf.py $COMMON $ARGS $TF --ckpt "$CK" $EMA --horizon-tics 1 --out-dir "$OUT" \
      > $D/logs/${RUN}_eval_tf_${S}${VARIANT}.log 2>&1
    echo "$RUN eval_tf ${S}${VARIANT} exit $?" >> $D/logs/${RUN}_eval.log
  done
  # equal game time against the stride-4 rows' single decision step: four tics rolled forward
  OUT4=$R/eval_tf_${S}_h4
  if [ ! -f "$OUT4/metrics.json" ]; then
    $PY eval_tf.py $COMMON $ARGS $TF --ckpt $R/best.pt --horizon-tics 4 --out-dir "$OUT4" \
      > $D/logs/${RUN}_eval_tf_${S}_h4.log 2>&1
    echo "$RUN eval_tf ${S}_h4 exit $?" >> $D/logs/${RUN}_eval.log
  fi
done

# rollouts on the sealed seen-map corpus, horizon in tics
ROLL_ARGS=$(corpus_args test)
ROLL_LAT=$(echo "$ROLL_ARGS" | sed -n 's/.*--latents-dir \([^ ]*\).*/\1/p')
if [ -d "$ROLL_LAT" ]; then
  NPZ=$R/rollouts_test.npz
  if [ ! -f "$NPZ" ] || [ "${RESCORE:-0}" != 1 ]; then
    $PY rollout_eval.py --rollout $COMMON --ckpt $R/best.pt $ROLL_ARGS \
      --subset val --num-rollouts 256 --horizon $HORIZON --batch-size 16 --steps 50 --out "$NPZ" \
      > $D/logs/${RUN}_rollout.log 2>&1
    echo "$RUN rollout exit $?" >> $D/logs/${RUN}_rollout_status.log
  fi
  IDM_ENC=""
  [ "$CH" = 16 ] && IDM_ENC="--idm-reencode-vae stabilityai/sd-vae-ft-mse"
  $PY rollout_eval.py --score --rollouts "$NPZ" --idm $D/results_spiderman/idm_aligned/idm.pt $IDM_ENC \
    $VAE $SCALE --hf-cache $D/hf/hub --out-dir $R/rollout_metrics_test --save-clips 256 \
    >> $D/logs/${RUN}_rollout.log 2>&1
  # FVD on both clip spacings: every tic, and every fourth tic so the clip covers the same game
  # time as a stride-4 row's clip of the same frame count
  for CLIPS in clips_u8.npz clips_u8_stride4.npz; do
    [ -f "$R/rollout_metrics_test/$CLIPS" ] || continue
    for F in 16 32; do
      $PY fvd.py --clips $R/rollout_metrics_test/$CLIPS --frames $F --i3d $D/weights/i3d_torchscript.pt \
        --out $R/rollout_metrics_test/fvd${F}_${CLIPS%.npz}.json >> $D/logs/${RUN}_rollout.log 2>&1
    done
  done
fi
echo "AFTER_NEXTTIC_DONE $RUN ($USED)" >> $D/logs/${RUN}_rollout.log
