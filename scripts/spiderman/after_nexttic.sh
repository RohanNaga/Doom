#!/bin/bash
# Evaluation suite for a next-tic row, in three stages that cannot be run out of order.
#
#   usage: [PY=..] [NOWAIT=1] [CKPT=path] [STEP=n] [CORPORA=".."] [RESCORE=1] [BEST=0] \
#          [SELECT_WINDOWS=512] [SPACING=linear] [SEED=0] [FORCE_SELECT=1] [FORCE_TEST=1] [DRY=1] [DOOM_ROOT=..] \
#          after_nexttic.sh <gpu> <unet | sd35 | pixart> [--select]
#
# The three stages (docs/REVIEW_2026-09-22.md, H2). The old script scored validation, test and
# unseen in one unattended pass, so nothing kept a test number from informing which checkpoint or
# variant was reported, and `CORPORA=val` still rolled out and scored test.
#
#   1 validation   CORPORA=val (the default): one checkpoint picked by STORED step
#                  (pick_checkpoint.py), live and EMA from that same file, `best.pt` as the separately
#                  labelled `_best` variant, and the four-tic horizon. Nothing sealed is touched.
#   2 selection    `--select` (or SELECT=1): the last stable checkpoint and up to two snapshots
#                  before it (select_checkpoint.py), each scored on validation live and EMA on
#                  SELECT_WINDOWS windows, then the preregistered rule (raw PSNR with an LPIPS check,
#                  `select_checkpoint.RULE`) writes `$R/selection.json`. Refused once any test score
#                  exists, because a selection made after seeing test is selection on test. The
#                  selection also PINS the sealed stage's whole scoring configuration: decoder path
#                  and content hash, sampler steps and SPACING, NUM_WINDOWS and SEED, horizons, and
#                  the content identity of every corpus present (score_identity.py); the sealed
#                  stage verifies it (`select_checkpoint.py verify`) and refuses any difference.
#   3 sealed       CORPORA naming test, arenas_678 or the stride-4 corpora: refused without
#                  `selection.json`, and scores ONLY the selected checkpoint and variant. Each corpus
#                  is SEALED at first access, before any score on it exists (`$R/sealed/<corpus>/`,
#                  an atomic mkdir, then one book entry per stage); a completed stage is never
#                  recomputed, RESCORE or not, an interrupted one may only rerun under its original
#                  key, and a complete corpus is refused, all unless FORCE_TEST=1, which the seal
#                  records. `$R/test_scored_at` keeps a readable log. Test rollouts happen here and
#                  only when CORPORA names test.
#                  arenas_678 is refused with a decoder that cannot back an unseen-map claim
#                  (decoder_provenance.py; every decoder tuned before 2026-09-22 is one) unless
#                  UNSEEN_CLAIM=dynamics-only asks for the weaker label, which is then recorded.
#
# A CORPORA that mixes val with a sealed corpus is refused: the two stages score different
# checkpoints by construction. CKPT and STEP apply to the validation stage only.
#
#   --tic-stride 1      the per-tic window set and the per-tic copy-last floor
#   --horizon-tics 4    four tics rolled forward from real context, scored at the 4th frame, so the
#                       number is at EQUAL GAME TIME with a stride-4 model's single step (114 ms)
#
# Caching. Every stage's output is keyed by what produced it: checkpoint path, stored step and
# SHA-256 (eval_identity.py), variant, the corpus by CONTENT (split file bytes and latent
# fingerprint, score_identity.py; never the split's pathname), the exact window or rollout manifest
# the evaluator will draw, horizon, window count, seed, sampler steps and spacing, and decoder. The key sits beside the result as `<file>.key`; a stage reruns when its result is absent,
# when the key differs, or when RESCORE=1. A result computed for one checkpoint can therefore never be
# relabelled with another's name, which the old existence-only test allowed.
#
# Corpus integrity. Before a corpus is scored, `make_dense_eval_splits.py --check-only` must confirm
# that its encoded episodes are exactly the expected ids and that its split file lists exactly those;
# a partial or wrong-range corpus is refused instead of silently scored on the episodes that exist.
#
# When to run. With no CKPT or STEP the validation stage waits for the trainer's `event=end` line. A
# manually stopped run never writes one, so CKPT=<path> or STEP=<n> evaluates that snapshot
# immediately. NOWAIT=1 means "the GPU is free, start now" on the latest checkpoint.
#
# Corpora. The first three are the dense corpus's own held-out sets from release/dense_split.json;
# the last three are the original 17-map evaluation corpora, kept as additional per-tic rows and
# SKIPPED CLEANLY when their per-tic latents do not exist yet:
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
SELECT=${SELECT:-0}
[ "${3:-}" = --select ] && SELECT=1
D=${DOOM_ROOT:-/sata2/data/rnagabhi/doom}
DRY=${DRY:-0}
HORIZON=${HORIZON:-256}
NUM_WINDOWS=${NUM_WINDOWS:-2048}
SELECT_WINDOWS=${SELECT_WINDOWS:-512}
CORPORA=${CORPORA:-val}
CKPT=${CKPT:-}
STEP=${STEP:-}
BEST=${BEST:-1}
RESCORE=${RESCORE:-0}
FORCE_SELECT=${FORCE_SELECT:-0}
FORCE_TEST=${FORCE_TEST:-0}
STEPS=50
SPACING=${SPACING:-linear}      # --timestep-spacing: which trained timesteps the sampler visits (timestep_spacing.py)
SEED=${SEED:-0}                 # the window draw and the sampler's noise keys
# shellcheck source=../dense_ids.sh
. "$(dirname "${BASH_SOURCE[0]}")/../dense_ids.sh"
VAL_IDS=${VAL_IDS:-$(dense_ids val_ids)}
TEST_IDS=${TEST_IDS:-$(dense_ids test_ids)}
UNSEEN_IDS=${UNSEEN_IDS:-$(dense_ids unseen_ids)}
RC=0
fail() { RC=1; echo "AFTER_NEXTTIC_STAGE_FAILED $*" >&2; }

# which stage this invocation is
OPEN=""; SEALED=""
for S in $CORPORA; do
  case $S in
    val) OPEN="$OPEN $S" ;;
    test|arenas_678|seen|unseen|unseen2) SEALED="$SEALED $S" ;;
    *) echo "unknown corpus '$S' (val | test arenas_678 seen unseen unseen2)" >&2; exit 2 ;;
  esac
done
if [ "$SELECT" = 1 ]; then STAGE=select
elif [ -n "$OPEN" ] && [ -n "$SEALED" ]; then
  echo "CORPORA='$CORPORA' mixes val with sealed corpora. Score val (the default), then --select, then" >&2
  echo "the sealed corpora in their own invocation: they score the SELECTED checkpoint, not a picked one." >&2
  exit 2
elif [ -n "$SEALED" ]; then STAGE=sealed
else STAGE=val; fi
if [ "$STAGE" != val ] && { [ -n "$CKPT" ] || [ -n "$STEP" ]; }; then
  echo "CKPT/STEP name a checkpoint for the validation stage only; the $STAGE stage takes its" >&2
  echo "checkpoints from the run directory and selection.json" >&2
  exit 2
fi

case $BACKBONE in
  unet)   RUN=040-unet-nexttic;   CH=4;  BB="--backbone unet --sd-path CompVis/stable-diffusion-v1-4" ;;
  pixart) RUN=041-pixart-nexttic; CH=4;  BB="--backbone pixart --pixart-path PixArt-alpha/PixArt-XL-2-512x512" ;;
  sd35)   RUN=042-sd35-nexttic;   CH=16; BB="--backbone sd35 --sd35-path stabilityai/stable-diffusion-3.5-medium" ;;
  *) echo "unknown backbone '$BACKBONE' (unet | sd35 | pixart)" >&2; exit 2 ;;
esac
R=$D/results_spiderman/$RUN
SEL=$R/selection.json
SCORED=$R/test_scored_at
LOG=$D/logs/${RUN}_eval.log
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
corpus_dir() {
  case $1 in
    val|test|arenas_678) echo "$LE/$1" ;;
    seen|unseen|unseen2) echo "$LO/$1" ;;
  esac
}
corpus_split() { echo "$(dirname "$(corpus_dir "$1")")/split_$1.json"; }
corpus_latents() { echo "--latents-dir $(corpus_dir "$1") --split $(corpus_split "$1")"; }
corpus_parquet() {
  case $1 in
    val|test)            echo "$D/raw_arnold_dense/arenas" ;;
    arenas_678)          echo "$D/raw_arnold_dense/arenas_678" ;;
    seen|unseen|unseen2) echo "$D/raw_arnold_eval/$1" ;;
  esac
}
corpus_tf() { echo "$(corpus_latents "$1") --parquet-dir $(corpus_parquet "$1")"; }
expected_ids() {   # the dense corpora have a declared id range; the stride-4 corpora are checked against their own split
  case $1 in
    val) echo "$VAL_IDS" ;;
    test) echo "$TEST_IDS" ;;
    arenas_678) echo "$UNSEEN_IDS" ;;
  esac
}
# every split file written by make_dense_eval_splits.py uses one subset key
SUBSET=val

# The tuned decoder, or the stock one. Tested with one `[ -f a ] || [ -f b ]` per file: a single
# `ls` of two paths reports success when only one exists, which is how the SD 3.5 row silently
# scored against the stock decoder on Sep 20 2026.
if [ "$CH" = 16 ]; then TUNED=$D/vae_decoder_sd35_lpips/vae; STOCK="--vae-path stabilityai/stable-diffusion-3.5-medium --vae-subfolder vae"
  STOCK_ID=stabilityai/stable-diffusion-3.5-medium
else TUNED=$D/vae_decoder_arnold_lpips/vae; STOCK=""; STOCK_ID=stock; fi
if [ -f "$TUNED/diffusion_pytorch_model.safetensors" ] || [ -f "$TUNED/diffusion_pytorch_model.bin" ]; then
  VAE="--vae-path $TUNED"; USED="tuned: $TUNED"; DEC_PATH=$TUNED; DEC_KIND=tuned
else
  VAE="$STOCK"; USED="stock fallback (the tuned decoder at $TUNED has no weights)"
  DEC_PATH=$STOCK_ID; DEC_KIND=stock
fi
DEC_TAG="$DEC_KIND:$DEC_PATH"
# The decoder's registry name, content hash and provenance (decoder_provenance.py; the decoders
# tuned before 2026-09-22 are `provenance: unknown, corpus: all maps` in release/decoder_registry.json).
# Filled in once the interpreter may run; every score directory records it.
DEC_INFO=unrecorded
DEC_ID=""
# the unseen-map claim a score of arenas_678 may make with this decoder: `system` when the decoder
# never saw the unseen maps or a held-out episode, `dynamics-only` when UNSEEN_CLAIM says so
CLAIM=""
# finetune_decoder.py writes metrics.json one level above the `vae` directory, with a `provenance`
# block naming the corpus the decoder was tuned on and the loss it was tuned under. A score is only
# readable next to that: a decoder that saw arenas 6-8 defeats an unseen-map claim whatever the
# denoiser was initialised from, and the tuned-vs-stock fallback above is silent otherwise.
DEC_METRICS=$(dirname "$TUNED")/metrics.json
prov() {   # prov <score-dir> <key>: say which decoder and which checkpoint produced the numbers in it
  mkdir -p "$1" || return 0
  {
    echo "decoder_kind=$DEC_KIND"
    echo "decoder_path=$DEC_PATH"
    echo "decoder=$DEC_INFO"
    echo "decoder_metrics=$DEC_METRICS"
    [ -n "$CLAIM" ] && echo "unseen_claim=$CLAIM"
    echo "score_key=$2"
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

# --- the cache key --------------------------------------------------------------------------
key_of() {   # key_of <ckpt> <step> <sha256> <variant> <corpus> <horizon> <windows> <tf|rollout>
  # the corpus by CONTENT (split file bytes and latent fingerprint) and the exact window manifest,
  # never the split's pathname: two files at one path can list different episodes
  local CID WM
  CID=$(corpus_id "$5") && [ -n "$CID" ] || return 1
  WM=$(window_manifest "$8" "$5" "$6" "$7") && [ -n "$WM" ] || return 1
  echo "ckpt=$1 step=$2 sha256=$3 variant=$4 corpus=$5 [$CID] [$WM] horizon=$6" \
       "windows=$7 seed=$SEED sampler=ddim$STEPS spacing=$SPACING decoder=$DEC_TAG"
}
window_manifest() {   # window_manifest <tf|rollout> <corpus> <horizon> <count>: what that evaluator will score
  "$PY" "$D/repo/score_identity.py" windows --kind "$1" --latents-dir "$(corpus_dir "$2")" \
    --split "$(corpus_split "$2")" --num "$4" --seed "$SEED" --context-frames 32 --horizon "$3" \
    --latent-channels "$CH" < /dev/null
}
# A stage runs when its result is absent, when the result was produced under a different key, or
# when RESCORE=1. The key sits beside the result as `<file>.key`.
should_run() {   # should_run <result file> <key>
  [ "$RESCORE" = 1 ] && return 0
  [ -e "$1" ] || return 0
  [ "$(cat "$1.key" 2>/dev/null)" = "$2" ] && return 1
  echo "$RUN $1 was produced under a different key; rescoring" >> "$LOG"
  return 0
}
stamp() { printf '%s\n' "$2" > "$1.key"; }   # stamp <result file> <key>
forget() { rm -f "$1" "$1.key"; }            # a failed rerun must not leave the old result looking current

# --- sealing ------------------------------------------------------------------------------------
# A sealed corpus is sealed at FIRST ACCESS: `seal_open` creates `$R/sealed/<corpus>/` with an atomic
# `mkdir` and writes the seal (time, selection, checkpoint) BEFORE any score on it is computed, so a
# run that dies half way has still opened the seal, a second process racing it is refused, and no
# selection can follow it. Every sealed stage is then booked in that directory: `<stage>.started` with
# its key before it runs, `<stage>.done` with its key after it succeeded. On a later invocation a
# stage that is done is never recomputed, RESCORE or not; a stage that started and did not finish may
# rerun only under the key it started with; a different key on either is refused. FORCE_TEST=1 is the
# one override, and it is written into the seal's log. `complete` marks a corpus whose every stage is
# done, which is what "scored once" means.
SEALS=$R/sealed
seal_open() {   # seal_open <corpus>: 0 if the corpus may be (re)entered now, else fails and returns 1
  local SD=$SEALS/$1
  mkdir -p "$SEALS" || { fail "cannot write $SEALS"; return 1; }
  if mkdir "$SD" 2>/dev/null; then
    echo "sealed $(date -Iseconds) corpus=$1 ckpt=$PICK step=$PICK_STEP sha256=$PICK_SHA variant=$PICK_VARIANT selection_sha256=$SEL_SHA" > "$SD/seal"
    return 0
  fi
  if [ "$FORCE_TEST" = 1 ]; then
    echo "FORCED $(date -Iseconds) FORCE_TEST=1 reopened $1 (selection_sha256=$SEL_SHA)" >> "$SD/seal"
    rm -f "$SD/complete"
    return 0
  fi
  [ -f "$SD/seal" ] || { fail "$1: $SD exists without a seal record; another scoring of it may be running"; return 1; }
  if ! grep -q "selection_sha256=$SEL_SHA\$" "$SD/seal"; then
    fail "$1 was sealed under a different selection ($(head -1 "$SD/seal")); FORCE_TEST=1 overrides"
    return 1
  fi
  if [ -f "$SD/complete" ]; then
    fail "$1 was already scored once (see $SD and $SCORED); FORCE_TEST=1 scores it again, and says so there"
    return 1
  fi
  echo "$(date -Iseconds) re-entered to finish its incomplete stages" >> "$SD/seal"
  return 0
}
sealed_should_run() {   # sealed_should_run <corpus> <stage id> <key>: 0 run, 1 skip (kept or refused)
  local SD=$SEALS/$1
  if [ "$FORCE_TEST" = 1 ]; then printf '%s\n' "$3" > "$SD/$2.started"; rm -f "$SD/$2.done"; return 0; fi
  if [ -f "$SD/$2.done" ]; then
    if [ "$(cat "$SD/$2.done")" = "$3" ]; then
      echo "$RUN sealed $1 $2 already complete; kept (RESCORE does not reopen a sealed stage)" >> "$LOG"
    else
      fail "sealed $1 $2 completed under a different configuration; refusing to recompute it (FORCE_TEST=1 overrides)"
    fi
    return 1
  fi
  if [ -f "$SD/$2.started" ] && [ "$(cat "$SD/$2.started")" != "$3" ]; then
    fail "sealed $1 $2 started under a different configuration; refusing to rerun it under this one (FORCE_TEST=1 overrides)"
    return 1
  fi
  printf '%s\n' "$3" > "$SD/$2.started"
  return 0
}
sealed_done() { printf '%s\n' "$3" > "$SEALS/$1/$2.done"; }   # sealed_done <corpus> <stage id> <key>
# run_or_skip <corpus> <stage id> <result file> <key>: the sealed rule for sealed corpora, the key rule otherwise
run_or_skip() {
  if [ "$STAGE" = sealed ]; then sealed_should_run "$1" "$2" "$4"; else should_run "$3" "$4"; fi
}
book() { [ "$STAGE" = sealed ] && sealed_done "$1" "$2" "$3"; return 0; }   # book <corpus> <stage id> <key>

tf() {   # tf <corpus> <out-dir> <ckpt> <step> <sha256> <live|ema> <horizon> <windows>
  local S=$1 OUT=$2 CK=$3 ST=$4 SHA=$5 V=$6 K=$7 N=$8 EMA="" KEY E
  [ "$V" = ema ] && EMA="--use-ema"
  if [ "$DRY" = 1 ]; then
    local LABEL; LABEL=$(basename "$OUT"); LABEL=${LABEL#eval_tf_}
    echo "DRY eval_tf $LABEL $PY eval_tf.py $COMMON $(corpus_tf "$S") --subset $SUBSET --num-windows $N --batch-size 16 --steps $STEPS --timestep-spacing $SPACING --seed $SEED --ckpt $CK $EMA --horizon-tics $K --out-dir $OUT"
    return 0
  fi
  KEY=$(key_of "$CK" "$ST" "$SHA" "$V" "$S" "$K" "$N" tf) \
    || { fail "eval_tf $(basename "$OUT"): the corpus identity or window manifest could not be computed"; return 0; }
  if ! run_or_skip "$S" "$(basename "$OUT")" "$OUT/metrics.json" "$KEY"; then
    echo "$RUN eval_tf $(basename "$OUT") not run: already scored under this key, or refused" >> "$LOG"
    return 0
  fi
  forget "$OUT/metrics.json"
  $PY eval_tf.py $COMMON $(corpus_tf "$S") --subset $SUBSET --num-windows "$N" --batch-size 16 \
    --steps $STEPS --timestep-spacing "$SPACING" --seed "$SEED" --ckpt "$CK" $EMA --horizon-tics "$K" --out-dir "$OUT" \
    > "$D/logs/${RUN}_$(basename "$OUT").log" 2>&1 < /dev/null; E=$?
  echo "$RUN eval_tf $(basename "$OUT") ckpt=$CK variant=$V exit $E" >> "$LOG"
  if [ $E -eq 0 ]; then stamp "$OUT/metrics.json" "$KEY"; prov "$OUT" "$KEY"; book "$S" "$(basename "$OUT")" "$KEY"
  else fail "eval_tf $(basename "$OUT") exit $E"; fi
}

corpus_ok() {   # corpus_ok <corpus>: present, and exactly the expected episodes, or it is not scored
  local S=$1 LAT IDS
  LAT=$(corpus_dir "$S"); IDS=$(expected_ids "$S")
  local CMD=("$PY" "$D/repo/make_dense_eval_splits.py" --latents-dir "$LAT" --split-file "$(corpus_split "$S")"
             --check-only --sample 0)
  [ -n "$IDS" ] && CMD+=(--expect-ids "$IDS")
  if [ "$DRY" = 1 ]; then echo "DRY check $S ${CMD[*]}"; return 0; fi
  [ -d "$LAT" ] || { echo "$RUN skipping $S: no per-tic latents at $LAT" | tee -a "$LOG"; return 1; }
  if ! "${CMD[@]}" > "$D/logs/${RUN}_corpus_$S.json" 2>&1 < /dev/null; then
    fail "corpus $S: the encoded episodes or the split file are not the expected set${IDS:+ ($IDS)}; see $D/logs/${RUN}_corpus_$S.json"
    return 1
  fi
}

# --- the scoring configuration a selection pins ------------------------------------------------
corpus_id() {   # corpus_id <corpus>: split file contents, latent fingerprint and size (score_identity.py)
  "$PY" "$D/repo/score_identity.py" corpus --latents-dir "$(corpus_dir "$1")" --split "$(corpus_split "$1")" < /dev/null
}
build_pins() {   # build_pins <corpus...>: PINS=(--pin key=value ...), what a sealed stage will score with
  PINS=(--pin "decoder_path=$DEC_PATH" --pin "decoder_identity=$DEC_ID" --pin "sampler_steps=$STEPS"
        --pin "spacing=$SPACING" --pin "num_windows=$NUM_WINDOWS" --pin "seed=$SEED" --pin "horizon_tics=1,4"
        --pin "rollout_horizon=$HORIZON" --pin "rollouts=256")
  local S ID
  for S in "$@"; do
    ID=$(corpus_id "$S") || { fail "corpus $S: its identity could not be computed"; return 1; }
    PINS+=(--pin "corpus.$S=$ID")
  done
}
present_corpora() {   # the corpora that exist now: every one of them is pinned at selection
  local S
  for S in val test arenas_678 seen unseen unseen2; do [ -d "$(corpus_dir "$S")" ] && echo "$S"; done
}

rollout() {   # rollout <corpus> <ckpt> <step> <sha256> <live|ema>: roll out, score, FVD
  local S=$1 CK=$2 ST=$3 SHA=$4 V=$5 EMA="" KEY NPZ M E IDM_ENC=""
  [ "$V" = ema ] && EMA="--use-ema"
  [ "$CH" = 16 ] && IDM_ENC="--idm-reencode-vae stabilityai/sd-vae-ft-mse"
  NPZ=$R/rollouts_$S.npz; M=$R/rollout_metrics_$S
  if [ "$DRY" = 1 ]; then
    echo "DRY rollout $PY rollout_eval.py --rollout $COMMON --ckpt $CK $EMA $(corpus_latents "$S") --subset $SUBSET --num-rollouts 256 --horizon $HORIZON --timestep-spacing $SPACING --seed $SEED --out $NPZ"
    echo "DRY score $PY rollout_eval.py --score --rollouts $NPZ --parquet-dir $(corpus_parquet "$S") --clip-frames 128 --out-dir $M"
    return 0
  fi
  KEY=$(key_of "$CK" "$ST" "$SHA" "$V" "$S" "$HORIZON" 256 rollout) \
    || { fail "rollout $S: the corpus identity or rollout manifest could not be computed"; return 1; }
  if run_or_skip "$S" rollout "$NPZ" "$KEY"; then
    forget "$NPZ"; rm -rf "$M"      # everything downstream of a rollout belongs to that rollout
    $PY rollout_eval.py --rollout $COMMON --ckpt "$CK" $EMA $(corpus_latents "$S") \
      --subset $SUBSET --num-rollouts 256 --horizon "$HORIZON" --batch-size 16 --steps $STEPS --timestep-spacing "$SPACING" --seed "$SEED" --out "$NPZ" \
      > "$D/logs/${RUN}_rollout.log" 2>&1 < /dev/null; E=$?
    echo "$RUN rollout $S ckpt=$CK variant=$V exit $E" >> "$D/logs/${RUN}_rollout_status.log"
    [ $E -eq 0 ] || { fail "rollout $S exit $E"; return 1; }
    stamp "$NPZ" "$KEY"; book "$S" rollout "$KEY"
  fi
  [ -f "$NPZ" ] || { fail "rollout $S: no rollouts at $NPZ to score"; return 1; }
  # drift.json, not metrics.json: rollout_eval.py --score writes `rollout_eval.SCORE_FILE`
  if run_or_skip "$S" rollout_score "$M/drift.json" "$KEY"; then
    forget "$M/drift.json"
    $PY rollout_eval.py --score --rollouts "$NPZ" --idm "$D/results_spiderman/idm_aligned/idm.pt" $IDM_ENC \
      $VAE $SCALE --hf-cache "$D/hf/hub" --parquet-dir "$(corpus_parquet "$S")" \
      --out-dir "$M" --save-clips 256 --clip-frames 128 \
      >> "$D/logs/${RUN}_rollout.log" 2>&1 < /dev/null || { fail "rollout score $S"; return 1; }
    stamp "$M/drift.json" "$KEY"; book "$S" rollout_score "$KEY"
    [ "$STAGE" = sealed ] || rm -f "$M"/fvd*.json "$M"/fvd*.json.key
  fi
  prov "$M" "$KEY"
  # FVD on both clip spacings: every tic, and every fourth tic so the clip covers the same game
  # time as a stride-4 row's clip of the same frame count. The raw-reference clips first: those are
  # the FVD numbers to report; the decoded-ground-truth ones are kept for continuity.
  local CLIPS F OUTF
  for CLIPS in clips_u8_raw.npz clips_u8_raw_stride4.npz clips_u8.npz clips_u8_stride4.npz; do
    [ -f "$M/$CLIPS" ] || continue
    for F in 16 32; do
      OUTF=$M/fvd${F}_${CLIPS%.npz}.json
      run_or_skip "$S" "fvd${F}_${CLIPS%.npz}" "$OUTF" "$KEY" || continue
      forget "$OUTF"
      if $PY fvd.py --clips "$M/$CLIPS" --frames $F --i3d "$D/weights/i3d_torchscript.pt" \
           --out "$OUTF" >> "$D/logs/${RUN}_rollout.log" 2>&1 < /dev/null; then
        stamp "$OUTF" "$KEY"; book "$S" "fvd${F}_${CLIPS%.npz}" "$KEY"
      else fail "fvd${F} ${CLIPS}"; fi
    done
  done
}

# --- DRY: print the plan of this stage and stop ------------------------------------------
if [ "$DRY" = 1 ]; then
  echo "DRY $RUN stage=$STAGE corpora=${CORPORA} decoder: $USED"
  case $STAGE in
    val)
      echo "DRY pick $PY pick_checkpoint.py --results-dir $R --require-ema --hash ${STEP:+--step $STEP} ${CKPT:+--ckpt $CKPT}"
      for S in $OPEN; do
        corpus_ok "$S"
        tf "$S" "$R/eval_tf_$S" PICKED STEP SHA live 1 "$NUM_WINDOWS"
        tf "$S" "$R/eval_tf_${S}_ema" PICKED STEP SHA ema 1 "$NUM_WINDOWS"
        [ "$BEST" = 1 ] && tf "$S" "$R/eval_tf_${S}_best" "$R/best.pt" STEP SHA live 1 "$NUM_WINDOWS"
        tf "$S" "$R/eval_tf_${S}_h4" PICKED STEP SHA live 4 "$NUM_WINDOWS"
      done ;;
    select)
      echo "DRY refuse if $SEALS or $SCORED exists (a selection after any sealed access is selection on test)"
      corpus_ok val
      echo "DRY candidates $PY select_checkpoint.py candidates --results-dir $R > $R/select/candidates.txt"
      tf val "$R/select/CANDIDATE/eval_tf_val" CANDIDATE STEP SHA live 1 "$SELECT_WINDOWS"
      tf val "$R/select/CANDIDATE/eval_tf_val_ema" CANDIDATE STEP SHA ema 1 "$SELECT_WINDOWS"
      echo "DRY choose $PY select_checkpoint.py choose --results-dir $R --candidates $R/select/candidates.txt --scores-dir $R/select --out $SEL --pin decoder_path=$DEC_PATH --pin decoder_identity=<sha> --pin sampler_steps=$STEPS --pin spacing=$SPACING --pin num_windows=$NUM_WINDOWS --pin seed=$SEED --pin horizon_tics=1,4 --pin rollout_horizon=$HORIZON --pin rollouts=256 --pin corpus.<every present corpus>=<score_identity.py corpus>" ;;
    sealed)
      echo "DRY require $SEL, else refuse; $PY select_checkpoint.py show --selection $SEL"
      echo "DRY verify $PY select_checkpoint.py verify --selection $SEL <the same pins, for$SEALED>; refuse on any difference, before any seal opens"
      for S in $SEALED; do
        echo "DRY seal $S at first access: mkdir $SEALS/$S, then book every stage; refuse a complete or differently keyed stage (FORCE_TEST=1 overrides)"
        corpus_ok "$S"
        tf "$S" "$R/eval_tf_$S" SELECTED STEP SHA VARIANT 1 "$NUM_WINDOWS"
        tf "$S" "$R/eval_tf_${S}_h4" SELECTED STEP SHA VARIANT 4 "$NUM_WINDOWS"
        [ "$S" = test ] && rollout "$S" SELECTED STEP SHA VARIANT
      done ;;
  esac
  exit 0
fi

export TMPDIR=$D/tmp/tmpdir; mkdir -p "$TMPDIR" "$R" "$D/logs"
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
# No `git pull`: this is the checkout the training run and the gates use, and pulling here changed
# the code under a live run (docs/REVIEW_2026-09-22.md H3). The commit is recorded instead.
cd "$D/repo" || { echo "AFTER_NEXTTIC_FAILED $RUN: no checkout at $D/repo" >&2; exit 2; }
echo "$(date -Iseconds) evaluating with code at $(git rev-parse HEAD 2>/dev/null || echo unversioned)" >> "$LOG"
DEC_INFO=$("$PY" "$D/repo/decoder_provenance.py" show "$DEC_PATH" < /dev/null 2>/dev/null) || DEC_INFO=""
DEC_INFO=${DEC_INFO:-unrecorded}
# the key carries the decoder's content hash, so re-tuned weights under the same path rescore
DEC_ID=$(echo "$DEC_INFO" | tr ' ' '\n' | sed -n 's/^identity=//p')
DEC_TAG="$DEC_TAG:$DEC_ID"
echo "$(date -Iseconds) decoder_used $USED" | tee -a "$R/decoder_used.txt"

# --- stage 1: validation ----------------------------------------------------------------------
if [ "$STAGE" = val ]; then
  # A named checkpoint is an explicit "score this now": a manually stopped run never writes
  # `event=end`, and waiting for one hangs for ever.
  if [ -n "$CKPT" ] || [ -n "$STEP" ]; then
    echo "$(date -Iseconds) evaluating a named checkpoint (${CKPT:-step $STEP}); not waiting for event=end"
  elif [ "${NOWAIT:-0}" != 1 ]; then
    until grep -q "\"event\": \"end\"" "$R/log.jsonl" 2>/dev/null; do sleep 300; done
  fi
  # ONE checkpoint, chosen by its STORED step, carrying both the live weights and the EMA. A
  # lexicographic `ls ... | sort | tail -1` put every snap_* ahead of every numbered recovery file.
  PICK_LINE=$("$PY" "$D/repo/pick_checkpoint.py" --results-dir "$R" --require-ema --hash \
    ${STEP:+--step "$STEP"} ${CKPT:+--ckpt "$CKPT"} < /dev/null) || {
    echo "AFTER_NEXTTIC_FAILED $RUN: no checkpoint to score in $R" >&2; exit 3; }
  read -r PICK PICK_STEP PICK_EMA PICK_SHA <<<"$PICK_LINE"
  echo "$(date -Iseconds) checkpoint $PICK step=$PICK_STEP ema=$PICK_EMA sha256=${PICK_SHA:-?} decoder=$USED" | tee -a "$R/scored_checkpoint.txt"
  for S in $OPEN; do
    corpus_ok "$S" || continue
    tf "$S" "$R/eval_tf_$S" "$PICK" "$PICK_STEP" "${PICK_SHA:-?}" live 1 "$NUM_WINDOWS"
    tf "$S" "$R/eval_tf_${S}_ema" "$PICK" "$PICK_STEP" "${PICK_SHA:-?}" ema 1 "$NUM_WINDOWS"
    # a separately labelled SELECTION result, not the same weights: best.pt is the minimum of the
    # noisy-context validation loss at whatever step reached it
    if [ "$BEST" = 1 ] && [ -f "$R/best.pt" ]; then
      if BEST_LINE=$("$PY" "$D/repo/pick_checkpoint.py" --ckpt "$R/best.pt" --hash < /dev/null); then
        read -r BCK BSTEP _ BSHA <<<"$BEST_LINE"
        tf "$S" "$R/eval_tf_${S}_best" "$BCK" "$BSTEP" "${BSHA:-?}" live 1 "$NUM_WINDOWS"
      else fail "best.pt could not be read"; fi
    fi
    # equal game time against the stride-4 rows' single decision step: four tics rolled forward
    tf "$S" "$R/eval_tf_${S}_h4" "$PICK" "$PICK_STEP" "${PICK_SHA:-?}" live 4 "$NUM_WINDOWS"
  done
  DONE_STEP=$PICK_STEP
fi

# --- stage 2: selection on validation --------------------------------------------------------
if [ "$STAGE" = select ]; then
  if [ -e "$SCORED" ] || [ -d "$SEALS" ]; then
    echo "AFTER_NEXTTIC_FAILED $RUN: a sealed corpus has been opened ($SEALS, $SCORED), so a test score may exist; a selection made now would be a selection on test" >&2
    exit 4
  fi
  if [ -e "$SEL" ] && [ "$FORCE_SELECT" != 1 ]; then
    echo "AFTER_NEXTTIC_FAILED $RUN: $SEL exists; a selection is made once (FORCE_SELECT=1 redoes it, before any test score only)" >&2
    exit 4
  fi
  corpus_ok val || { echo "AFTER_NEXTTIC_FAILED $RUN: the val corpus cannot be scored" >&2; exit 4; }
  mkdir -p "$R/select"
  "$PY" "$D/repo/select_checkpoint.py" candidates --results-dir "$R" > "$R/select/candidates.txt" < /dev/null \
    || { echo "AFTER_NEXTTIC_FAILED $RUN: no candidate checkpoint in $R" >&2; exit 3; }
  CANDS=()
  while IFS= read -r LINE; do [ -n "$LINE" ] && CANDS+=("$LINE"); done < "$R/select/candidates.txt"
  # `${CANDS[@]+...}`: bash before 4.4 calls an empty array unbound under `set -u`
  for LINE in ${CANDS[@]+"${CANDS[@]}"}; do
    read -r CK ST SHA NAME <<<"$LINE"
    tf val "$R/select/$NAME/eval_tf_val" "$CK" "$ST" "$SHA" live 1 "$SELECT_WINDOWS"
    tf val "$R/select/$NAME/eval_tf_val_ema" "$CK" "$ST" "$SHA" ema 1 "$SELECT_WINDOWS"
  done
  if [ $RC -ne 0 ]; then
    echo "AFTER_NEXTTIC_FAILED $RUN: a candidate could not be scored; nothing was selected" >&2; exit $RC
  fi
  FORCE=""; [ "$FORCE_SELECT" = 1 ] && FORCE=--force
  # the whole scoring configuration of the sealed stage is frozen here, before any test score:
  # decoder path and hash, sampler steps and spacing, window count and seed, horizons, and the
  # content identity of every corpus that exists now (test and unseen included)
  # shellcheck disable=SC2046
  build_pins $(present_corpora) || { echo "AFTER_NEXTTIC_FAILED $RUN: the scoring configuration could not be pinned" >&2; exit 4; }
  CHOSE=$("$PY" "$D/repo/select_checkpoint.py" choose --results-dir "$R" --candidates "$R/select/candidates.txt" \
    --scores-dir "$R/select" --out "$SEL" --meta "windows=$SELECT_WINDOWS" --meta "sampler=ddim$STEPS" \
    --meta "decoder=$DEC_TAG" --meta "corpus=val:$VAL_IDS" "${PINS[@]}" $FORCE < /dev/null) \
    || { echo "AFTER_NEXTTIC_FAILED $RUN: selection refused (see above)" >&2; exit 4; }
  echo "$CHOSE" | tee -a "$LOG"
  echo "AFTER_NEXTTIC_SELECTED $RUN $SEL"
  exit 0
fi

# --- stage 3: the sealed corpora, once, with the selected checkpoint ----------------------------
if [ "$STAGE" = sealed ]; then
  [ -f "$SEL" ] || { echo "AFTER_NEXTTIC_FAILED $RUN: no $SEL. Run the validation stage, then --select; the sealed corpora score only the selected checkpoint" >&2; exit 4; }
  SEL_LINE=$("$PY" "$D/repo/select_checkpoint.py" show --selection "$SEL" < /dev/null) \
    || { echo "AFTER_NEXTTIC_FAILED $RUN: $SEL does not name a checkpoint that is still on disk unchanged" >&2; exit 4; }
  read -r PICK PICK_STEP PICK_SHA PICK_VARIANT SEL_SHA <<<"$SEL_LINE"
  # the configuration about to be used must be the one pinned at selection, for every corpus named
  # shellcheck disable=SC2086
  build_pins $SEALED || { echo "AFTER_NEXTTIC_FAILED $RUN: the scoring configuration could not be computed" >&2; exit 4; }
  if ! WHY=$("$PY" "$D/repo/select_checkpoint.py" verify --selection "$SEL" "${PINS[@]}" < /dev/null 2>&1); then
    echo "AFTER_NEXTTIC_FAILED $RUN: $WHY" >&2
    exit 4
  fi
  echo "$(date -Iseconds) selected $PICK step=$PICK_STEP variant=$PICK_VARIANT sha256=$PICK_SHA selection=$SEL_SHA" | tee -a "$R/scored_checkpoint.txt"
  for S in $SEALED; do
    if [ -f "$SEALS/$S/complete" ] && [ "$FORCE_TEST" != 1 ]; then
      fail "$S was already scored once (see $SEALS/$S and $SCORED); FORCE_TEST=1 scores it again, and says so there"
      continue
    fi
    corpus_ok "$S" || continue
    CLAIM=""
    if [ "$S" = arenas_678 ]; then
      # an unseen-map number is about the whole system: a decoder that saw arenas 6-8, a held-out
      # episode, or whose training frames are unknown cannot back it (docs/REVIEW_2026-09-22.md H4)
      if "$PY" "$D/repo/decoder_provenance.py" check "$DEC_PATH" --claim unseen-map \
           > "$D/logs/${RUN}_decoder_claim.txt" 2>&1 < /dev/null; then CLAIM=system
      elif [ "${UNSEEN_CLAIM:-}" = dynamics-only ]; then CLAIM=dynamics-only
      else
        fail "arenas_678: this decoder cannot back an unseen-map claim ($(head -c 300 "$D/logs/${RUN}_decoder_claim.txt")). UNSEEN_CLAIM=dynamics-only scores it as 'unseen by the dynamics model' and records that label"
        continue
      fi
    fi
    # the seal opens here, before the first score on this corpus exists
    seal_open "$S" || continue
    echo "$S started $(date -Iseconds) ckpt=$PICK step=$PICK_STEP sha256=$PICK_SHA variant=$PICK_VARIANT selection_sha256=$SEL_SHA${CLAIM:+ unseen_claim=$CLAIM}" >> "$SCORED"
    BEFORE=$RC; RC=0
    tf "$S" "$R/eval_tf_$S" "$PICK" "$PICK_STEP" "$PICK_SHA" "$PICK_VARIANT" 1 "$NUM_WINDOWS"
    tf "$S" "$R/eval_tf_${S}_h4" "$PICK" "$PICK_STEP" "$PICK_SHA" "$PICK_VARIANT" 4 "$NUM_WINDOWS"
    # rollouts on the sealed seen-map corpus, horizon in tics
    [ "$S" = test ] && rollout "$S" "$PICK" "$PICK_STEP" "$PICK_SHA" "$PICK_VARIANT"
    if [ $RC -eq 0 ]; then
      echo "complete $(date -Iseconds) forced=$FORCE_TEST" > "$SEALS/$S/complete"
      echo "$S done $(date -Iseconds) ckpt=$PICK step=$PICK_STEP sha256=$PICK_SHA variant=$PICK_VARIANT selection_sha256=$SEL_SHA forced=$FORCE_TEST" >> "$SCORED"
    fi
    [ $BEFORE -ne 0 ] && RC=$BEFORE
  done
  DONE_STEP=$PICK_STEP
fi

# DONE means every stage that ran exited 0. It used to be printed unconditionally, so a failed
# scoring pass still announced a finished evaluation.
if [ $RC -ne 0 ]; then
  echo "AFTER_NEXTTIC_FAILED $RUN rc=$RC ($USED)" | tee -a "$D/logs/${RUN}_rollout.log" >&2
  exit $RC
fi
echo "AFTER_NEXTTIC_DONE $RUN step=$DONE_STEP stage=$STAGE ($USED)" | tee -a "$D/logs/${RUN}_rollout.log"
