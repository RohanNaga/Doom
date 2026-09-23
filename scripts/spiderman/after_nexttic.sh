#!/bin/bash
# Evaluation suite for a next-tic row. Same protocol after_sd35.sh runs for the stride-4 rows, at
# tic spacing, plus the two things a next-tic number needs to be readable:
#
#   --tic-stride 1      the per-tic window set and the per-tic copy-last floor
#   --horizon-tics 4    four tics rolled forward from real context, scored at the 4th frame, so the
#                       number is at EQUAL GAME TIME with a stride-4 model's single step (114 ms)
#
#   usage: [PY=..] [NOWAIT=1] [CKPT=path] [STEP=n] [CORPORA=".."] [RESCORE=1] [BEST=0] \
#          [DRY=1] [DOOM_ROOT=..] after_nexttic.sh <gpu> <unet | sd35 | pixart>
#
# Which weights. ONE checkpoint is chosen by its STORED step (pick_checkpoint.py) and BOTH the live
# and the EMA number come from that file, so the comparison is paired. `best.pt` is scored too, as
# the separately labelled `_best` variant, because it is a selection result on validation loss and
# not the same weights; BEST=0 drops it.
#
# When to run. With no CKPT or STEP it waits for the trainer's `event=end` line. A manually stopped
# run never writes one, so CKPT=<path> or STEP=<n> evaluates that snapshot immediately and does not
# wait at all. NOWAIT=1 still means "the GPU is free, start now" on the latest checkpoint.
#
# RESCORE=1 reruns every stage whose output already exists; without it every stage with an output
# is skipped. Both rules apply to the teacher-forced passes and to the rollout alike.
#
# Corpora, in the order they are scored. The first three are the dense corpus's own held-out sets
# from release/dense_split.json; the last three are the original 17-map evaluation corpora, kept as
# additional per-tic rows and SKIPPED CLEANLY when their per-tic latents do not exist yet:
#
#   val        arenas ids 6000-6099   25/arena, the in-training validation corpus
#   test       arenas ids 7000-7099   25/arena, the sealed seen-map row
#   arenas_678 ids 60-119             20/arena, the unseen-map row (UNSEEN_IDS; read from the json,
#                                     0:60 until 2026-09-22, replaced before any score: see its history)
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
CKPT=${CKPT:-}
STEP=${STEP:-}
BEST=${BEST:-1}
RESCORE=${RESCORE:-0}
# shellcheck source=../dense_ids.sh
. "$(dirname "${BASH_SOURCE[0]}")/../dense_ids.sh"
UNSEEN_IDS=${UNSEEN_IDS:-$(dense_ids unseen_ids)}
RC=0
fail() { RC=1; echo "AFTER_NEXTTIC_STAGE_FAILED $*" >&2; }
# every stage runs when its output is absent, or when RESCORE=1 says to redo it. The old test was
# `[ ! -f $NPZ ] || [ "$RESCORE" != 1 ]`, which is true whenever the file is missing AND true
# whenever RESCORE is not 1: the default reran existing rollouts and RESCORE=1 skipped them.
should_run() { [ ! -e "$1" ] || [ "$RESCORE" = 1 ]; }

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
#
# Two argument sets per corpus, because the two scripts take different flags. `corpus_latents` is
# the intersection both accept; `corpus_tf` adds `--parquet-dir`, which eval_tf.py takes and
# `rollout_eval.py --rollout` does not (it reads latents only). `rollout_eval.py --score` takes it
# too, and needs it: the raw frames are the primary reference for the drift curve and for FVD.
corpus_latents() {
  case $1 in
    val|test)            echo "--latents-dir $LE/$1 --split $LE/split_$1.json" ;;
    arenas_678)          echo "--latents-dir $LE/arenas_678 --split $LE/split_arenas_678.json" ;;
    seen|unseen|unseen2) echo "--latents-dir $LO/$1 --split $LO/split_$1.json" ;;
  esac
}
corpus_parquet() {
  case $1 in
    val|test)            echo "$D/raw_arnold_dense/arenas" ;;
    arenas_678)          echo "$D/raw_arnold_dense/arenas_678" ;;
    seen|unseen|unseen2) echo "$D/raw_arnold_eval/$1" ;;
  esac
}
corpus_tf() { echo "$(corpus_latents "$1") --parquet-dir $(corpus_parquet "$1")"; }
# every split file written by make_dense_eval_splits.py uses one subset key
SUBSET=val

# The tuned decoder, or the stock one. Tested with one `[ -f a ] || [ -f b ]` per file: a single
# `ls` of two paths reports success when only one exists, which is how the SD 3.5 row silently
# scored against the stock decoder on Sep 20 2026.
if [ "$CH" = 16 ]; then TUNED=$D/vae_decoder_sd35_lpips/vae; STOCK="--vae-path stabilityai/stable-diffusion-3.5-medium --vae-subfolder vae"
else TUNED=$D/vae_decoder_arnold_lpips/vae; STOCK=""; fi
if [ -f "$TUNED/diffusion_pytorch_model.safetensors" ] || [ -f "$TUNED/diffusion_pytorch_model.bin" ]; then
  VAE="--vae-path $TUNED"; USED="tuned: $TUNED"; DEC_PATH=$TUNED; DEC_KIND=tuned
else
  VAE="$STOCK"; USED="stock fallback (the tuned decoder at $TUNED has no weights)"
  DEC_PATH=${STOCK:-stock}; DEC_KIND=stock
fi
# finetune_decoder.py writes metrics.json one level above the `vae` directory, with a `provenance`
# block naming the corpus the decoder was tuned on and the loss it was tuned under. A score is only
# readable next to that: a decoder that saw arenas 6-8 defeats an unseen-map claim whatever the
# denoiser was initialised from, and the tuned-vs-stock fallback above is silent otherwise.
DEC_METRICS=$(dirname "$TUNED")/metrics.json
prov() {   # prov <score-dir>: say which decoder and which checkpoint produced the numbers in it
  mkdir -p "$1" || return 0
  {
    echo "decoder_kind=$DEC_KIND"
    echo "decoder_path=$DEC_PATH"
    echo "decoder_metrics=$DEC_METRICS"
    echo "scored_checkpoint=${PICK:-?} step=${PICK_STEP:-?}"
    echo "recorded=$(date -Iseconds)"
  } > "$1/decoder_provenance.txt"
  [ -f "$DEC_METRICS" ] && cp "$DEC_METRICS" "$1/decoder_metrics.json"
  return 0
}
SCALE=""
[ "$CH" = 16 ] && SCALE="--latent-scale 1.5305 --latent-shift 0.0609"

# the conditioning interface is read from the checkpoint, never re-specified here: a mismatch
# between training and evaluation would produce a number nobody could interpret
COMMON="$BB --latent-channels $CH $VAE $SCALE --hf-cache $D/hf/hub --tic-stride 1 \
 --context-frames 32 --num-actions 29"

if [ "$DRY" = 1 ]; then
  echo "DRY $RUN decoder: $USED"
  echo "DRY pick $PY pick_checkpoint.py --results-dir $R --require-ema ${STEP:+--step $STEP} ${CKPT:+--ckpt $CKPT}"
  for S in $CORPORA; do
    for VARIANT in "" "_ema" "_best"; do
      echo "DRY eval_tf ${S}${VARIANT} $PY eval_tf.py $COMMON $(corpus_tf $S) --subset $SUBSET --ckpt PICKED --horizon-tics 1 --out-dir $R/eval_tf_${S}${VARIANT}"
    done
    echo "DRY eval_tf ${S}_h4 $PY eval_tf.py $COMMON $(corpus_tf $S) --subset $SUBSET --ckpt PICKED --horizon-tics 4 --out-dir $R/eval_tf_${S}_h4"
  done
  echo "DRY rollout $PY rollout_eval.py --rollout $COMMON --ckpt PICKED $(corpus_latents test) --subset $SUBSET --num-rollouts 256 --horizon $HORIZON --out $R/rollouts_test.npz"
  echo "DRY score $PY rollout_eval.py --score --rollouts $R/rollouts_test.npz --parquet-dir $(corpus_parquet test) --clip-frames 128 --out-dir $R/rollout_metrics_test"
  exit 0
fi

export TMPDIR=$D/tmp/tmpdir; mkdir -p $TMPDIR $R $D/logs
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
# A named checkpoint is an explicit "score this now": a manually stopped run never writes
# `event=end`, and waiting for one hangs for ever.
if [ -n "$CKPT" ] || [ -n "$STEP" ]; then
  echo "$(date -Iseconds) evaluating a named checkpoint (${CKPT:-step $STEP}); not waiting for event=end"
elif [ "${NOWAIT:-0}" != 1 ]; then
  until grep -q "\"event\": \"end\"" $R/log.jsonl 2>/dev/null; do sleep 300; done
fi
cd $D/repo && git pull -q
echo "$(date -Iseconds) decoder_used $USED" | tee -a $R/decoder_used.txt

# ONE checkpoint, chosen by its STORED step, carrying both the live weights and the EMA. A
# lexicographic `ls ... | sort | tail -1` put every snap_* ahead of every numbered recovery file.
PICK_LINE=$("$PY" "$D/repo/pick_checkpoint.py" --results-dir "$R" --require-ema \
  ${STEP:+--step "$STEP"} ${CKPT:+--ckpt "$CKPT"}) || {
  echo "AFTER_NEXTTIC_FAILED $RUN: no checkpoint to score in $R" >&2; exit 3; }
read -r PICK PICK_STEP PICK_EMA <<<"$PICK_LINE"
echo "$(date -Iseconds) checkpoint $PICK step=$PICK_STEP ema=$PICK_EMA decoder=$USED" | tee -a $R/scored_checkpoint.txt

TF="--subset $SUBSET --num-windows $NUM_WINDOWS --batch-size 16 --steps 50"
for S in $CORPORA; do
  ARGS=$(corpus_tf $S)
  LAT=$LE/$S; case $S in seen|unseen|unseen2) LAT=$LO/$S ;; esac
  [ -d "$LAT" ] || { echo "$RUN skipping $S: no per-tic latents at $LAT" | tee -a $D/logs/${RUN}_eval.log; continue; }
  for VARIANT in "" "_ema" "_best"; do
    CK=$PICK; EMA=""
    case $VARIANT in
      _ema)  EMA="--use-ema" ;;
      # a separately labelled SELECTION result, not the same weights: best.pt is the minimum of the
      # noisy-context validation loss at whatever step reached it
      _best) [ "$BEST" = 1 ] || continue; CK=$R/best.pt; [ -f "$CK" ] || continue ;;
    esac
    OUT=$R/eval_tf_${S}${VARIANT}
    should_run "$OUT/metrics.json" || { echo "$RUN eval_tf ${S}${VARIANT} already scored (RESCORE=1 to redo)" >> $D/logs/${RUN}_eval.log; continue; }
    $PY eval_tf.py $COMMON $ARGS $TF --ckpt "$CK" $EMA --horizon-tics 1 --out-dir "$OUT" \
      > $D/logs/${RUN}_eval_tf_${S}${VARIANT}.log 2>&1; E=$?
    echo "$RUN eval_tf ${S}${VARIANT} ckpt=$CK exit $E" >> $D/logs/${RUN}_eval.log
    prov "$OUT"
    [ $E -eq 0 ] || fail "eval_tf ${S}${VARIANT} exit $E"
  done
  # equal game time against the stride-4 rows' single decision step: four tics rolled forward
  OUT4=$R/eval_tf_${S}_h4
  if should_run "$OUT4/metrics.json"; then
    $PY eval_tf.py $COMMON $ARGS $TF --ckpt "$PICK" --horizon-tics 4 --out-dir "$OUT4" \
      > $D/logs/${RUN}_eval_tf_${S}_h4.log 2>&1; E=$?
    echo "$RUN eval_tf ${S}_h4 ckpt=$PICK exit $E" >> $D/logs/${RUN}_eval.log
    prov "$OUT4"
    [ $E -eq 0 ] || fail "eval_tf ${S}_h4 exit $E"
  fi
done

# rollouts on the sealed seen-map corpus, horizon in tics
ROLL_ARGS=$(corpus_latents test)
ROLL_LAT=$LE/test
if [ -d "$ROLL_LAT" ]; then
  NPZ=$R/rollouts_test.npz
  if should_run "$NPZ"; then
    $PY rollout_eval.py --rollout $COMMON --ckpt "$PICK" $ROLL_ARGS \
      --subset $SUBSET --num-rollouts 256 --horizon $HORIZON --batch-size 16 --steps 50 --out "$NPZ" \
      > $D/logs/${RUN}_rollout.log 2>&1; E=$?
    echo "$RUN rollout ckpt=$PICK exit $E" >> $D/logs/${RUN}_rollout_status.log
    [ $E -eq 0 ] || fail "rollout exit $E"
  fi
  IDM_ENC=""
  [ "$CH" = 16 ] && IDM_ENC="--idm-reencode-vae stabilityai/sd-vae-ft-mse"
  # drift.json, not metrics.json: rollout_eval.py --score writes `rollout_eval.SCORE_FILE` and has
  # never written metrics.json, so gating on that name made the test always true and reran the whole
  # scoring pass (and its 256 decodes) on every invocation
  if should_run "$R/rollout_metrics_test/drift.json"; then
    $PY rollout_eval.py --score --rollouts "$NPZ" --idm $D/results_spiderman/idm_aligned/idm.pt $IDM_ENC \
      $VAE $SCALE --hf-cache $D/hf/hub --parquet-dir "$(corpus_parquet test)" \
      --out-dir $R/rollout_metrics_test --save-clips 256 --clip-frames 128 \
      >> $D/logs/${RUN}_rollout.log 2>&1 || fail "rollout score"
  fi
  prov "$R/rollout_metrics_test"
  # FVD on both clip spacings: every tic, and every fourth tic so the clip covers the same game
  # time as a stride-4 row's clip of the same frame count
  # the raw-reference clips first: those are the FVD numbers to report, and the decoded-ground-truth
  # ones are kept for continuity with the stride-4 rows
  for CLIPS in clips_u8_raw.npz clips_u8_raw_stride4.npz clips_u8.npz clips_u8_stride4.npz; do
    [ -f "$R/rollout_metrics_test/$CLIPS" ] || continue
    for F in 16 32; do
      OUTF=$R/rollout_metrics_test/fvd${F}_${CLIPS%.npz}.json
      should_run "$OUTF" || continue
      $PY fvd.py --clips $R/rollout_metrics_test/$CLIPS --frames $F --i3d $D/weights/i3d_torchscript.pt \
        --out "$OUTF" >> $D/logs/${RUN}_rollout.log 2>&1 || fail "fvd${F} ${CLIPS}"
    done
  done
fi
# DONE means every stage that ran exited 0. It used to be printed unconditionally, so a failed
# scoring pass still announced a finished evaluation.
if [ $RC -ne 0 ]; then
  echo "AFTER_NEXTTIC_FAILED $RUN rc=$RC ($USED)" | tee -a $D/logs/${RUN}_rollout.log >&2
  exit $RC
fi
echo "AFTER_NEXTTIC_DONE $RUN step=$PICK_STEP ($USED)" | tee -a $D/logs/${RUN}_rollout.log
