#!/bin/bash
# Score every map of the distance study with the UNCHANGED eval_tf.py: one directory per map and horizon.
#
#   usage: [CKPT=path] [RUN=name] [SPACE=sd1|sd35] [MAPS="val:2 unseen2"] [STEPS=10] [HORIZONS="1 4"] \
#          [NUM_WINDOWS=256] [SEED=0] [SPLITS=dir] [OUT=dir] [RESCORE=1] [PY_UNET=..] [PY_SD35=..] [PY=..] \
#          [RUN_REPO=$D/repo] [DRY=1] [DOOM_ROOT=..] \
#          score_distance_maps.sh <gpu> <unet | sd35 | pixart>
#
# Design: .claude/analyses/distance-study-design-2026-09-24.md, section 6 item 2. The map list is the
# split files `distance_study.py splits` wrote (SPLITS, default $RUN_REPO/results/distance_study/splits,
# one `split_<set>_map<NN>.json` per map); MAPS narrows it to `set:map` pairs or whole sets. Each map is
# scored with the memo's settings:
#
#   --num-windows 256 --use-ema --num-workers 0   the same 256 windows per map for every row
#                                                 (eval_tf.draw_windows), EMA weights, no workers
#   --steps $STEPS (default 10)                   the paper's sampler step count, as a knob
#   stock decoder                                 no decoder has seen an evaluation map; SD 3.5 needs its
#                                                 own autoencoder's flags (gates.sh readback_cmd)
#   horizons 1 and 4, as separate stages          every map at one tic before any map at four, so the
#                                                 primary outcome finishes first when the card is short
#
# No `--wandb-run`, on purpose: eval_tf.py appends a read to `<run>-eval` at the checkpoint's step as
# `eval/ema_h<H>/...` (wandb_log.py:507-510), so thirty per-map reads at one step would overwrite the
# training run's headline `eval/ema_h1` series with whichever map was scored last.
#
# Resumable. A map is complete when its directory holds metrics.json and a score_key.txt equal to the
# key this invocation would write: checkpoint path, stored step and SHA-256 (pick_checkpoint.py), EMA,
# space, the split file's SHA-256, horizon, windows, seed, sampler steps and decoder. Complete maps are
# skipped; a map scored under a DIFFERENT key is refused rather than overwritten (RESCORE=1 replaces
# it); a failed map leaves no key, so the next invocation retries exactly the failures. One scorer per
# output directory at a time (flock on $OUT/score.lock). Output:
#   OUT (default $D/results_spiderman/distance_study/$RUN/<checkpoint>_ema_ddim$STEPS)/<set>_map<NN>_h<K>/
#
# Which corpora. The study scores val and arenas_678 (dense) and seen, unseen, unseen2 (the seeded
# corpus). `test` is sealed for the headline table (after_nexttic.sh stage 3) and is refused here.
# after_nexttic.sh also treats arenas_678, seen, unseen and unseen2 as sealed (after_nexttic.sh:121);
# this pass reads them with one named checkpoint and books nothing in $R/sealed, which is a decision
# for Rohan, not something this script settles.
#
# The interpreter follows the other launchers: PY_UNET for the 4-channel rows, PY_SD35 for SD 3.5,
# each falling back to PY and then to this host's env. RUN_REPO is the checkout whose eval_tf.py runs.
set -u
GPU=${1:?gpu}
BACKBONE=${2:?backbone (unet | sd35 | pixart)}
D=${DOOM_ROOT:-/sata2/data/rnagabhi/doom}
DRY=${DRY:-0}
STEPS=${STEPS:-10}
HORIZONS=${HORIZONS:-1 4}
NUM_WINDOWS=${NUM_WINDOWS:-256}
SEED=${SEED:-0}
RESCORE=${RESCORE:-0}
MAPS=${MAPS:-}
CKPT=${CKPT:-}
RUN_REPO=${RUN_REPO:-$D/repo}
SPLITS=${SPLITS:-$RUN_REPO/results/distance_study/splits}

case $BACKBONE in
  unet)   DEF_RUN=040-unet-nexttic;   CH=4;  BB=(--backbone unet --sd-path CompVis/stable-diffusion-v1-4) ;;
  pixart) DEF_RUN=041-pixart-nexttic; CH=4;  BB=(--backbone pixart --pixart-path PixArt-alpha/PixArt-XL-2-512x512) ;;
  sd35)   DEF_RUN=042-sd35-nexttic;   CH=16; BB=(--backbone sd35 --sd35-path stabilityai/stable-diffusion-3.5-medium) ;;
  *) echo "unknown backbone '$BACKBONE' (unet | sd35 | pixart)" >&2; exit 2 ;;
esac
RUN=${RUN:-$DEF_RUN}
R=$D/results_spiderman/$RUN
if [ "$CH" = 16 ]; then SPACE=${SPACE:-sd35}; else SPACE=${SPACE:-sd1}; fi
case $SPACE in
  sd1)  SPACE_CH=4;  SUFFIX="" ;;
  sd35) SPACE_CH=16; SUFFIX=_sd35 ;;
  *) echo "unknown SPACE '$SPACE' (sd1 | sd35)" >&2; exit 2 ;;
esac
if [ "$SPACE_CH" != "$CH" ]; then
  echo "SPACE=$SPACE holds $SPACE_CH-channel latents; the $BACKBONE row reads $CH" >&2; exit 2
fi
if [ "$CH" = 16 ]; then PY=${PY_SD35:-${PY:-$HOME/wanenc/bin/python}}
else PY=${PY_UNET:-${PY:-$HOME/miniconda3/envs/doom/bin/python}}; fi
# the stock decoders: eval_tf.py's default (stabilityai/sd-vae-ft-mse) for the 4-channel rows, SD 3.5's
# own autoencoder with its scale and shift for the 16-channel row
if [ "$CH" = 16 ]; then
  DECODER=(--vae-path stabilityai/stable-diffusion-3.5-medium --vae-subfolder vae --latent-scale 1.5305 --latent-shift 0.0609)
  DEC_TAG=stock:stabilityai/stable-diffusion-3.5-medium
else
  DECODER=()
  DEC_TAG=stock:stabilityai/sd-vae-ft-mse
fi

latents_dir() {
  case $1 in
    val|arenas_678) echo "$D/latents_arnold_dense_pertic_eval$SUFFIX/$1" ;;
    seen|unseen|unseen2) echo "$D/latents_arnold_eval_pertic$SUFFIX/$1" ;;
    *) return 1 ;;
  esac
}
raw_dir() {
  case $1 in
    val) echo "$D/raw_arnold_dense/arenas" ;;
    arenas_678) echo "$D/raw_arnold_dense/arenas_678" ;;
    seen|unseen|unseen2) echo "$D/raw_arnold_eval/$1" ;;
    *) return 1 ;;
  esac
}
sha_of() {
  if command -v sha256sum > /dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# --- the maps: one split file each, narrowed by MAPS ------------------------------------------------
# Every split file must name a corpus the study scores; `test` is refused even when MAPS leaves it out.
CORPORA="val arenas_678 seen unseen unseen2"
for F in "$SPLITS"/split_*_map*.json; do
  [ -e "$F" ] || continue
  B=$(basename "$F" .json); B=${B#split_}; S=${B%_map*}
  if [ "$S" = test ]; then
    echo "$F names the sealed test corpus; the distance study never scores it" >&2; exit 2
  fi
  latents_dir "$S" > /dev/null || { echo "$F: unknown corpus '$S' ($CORPORA)" >&2; exit 2; }
done
# in corpus order, the dense corpora first, then by map: what finishes first if the card runs short
LIST=()
for S in $CORPORA; do
  for F in "$SPLITS"/split_"$S"_map*.json; do
    [ -e "$F" ] || continue
    B=$(basename "$F" .json); M=${B##*_map}
    if [ -n "$MAPS" ]; then
      case " $MAPS " in *" $S:$((10#$M)) "*|*" $S "*) ;; *) continue ;; esac
    fi
    LIST+=("$S $M $F")
  done
done
if [ ${#LIST[@]} -eq 0 ]; then
  WHICH=""; [ -n "$MAPS" ] && WHICH=" matching MAPS=\"$MAPS\""
  echo "no split files to score in $SPLITS$WHICH (run distance_study.py splits)" >&2; exit 2
fi

# --- the checkpoint -----------------------------------------------------------------------------------
if [ -n "$CKPT" ]; then PICK=("$PY" "$RUN_REPO/pick_checkpoint.py" --ckpt "$CKPT" --hash)
else PICK=("$PY" "$RUN_REPO/pick_checkpoint.py" --results-dir "$R" --require-ema --hash); fi
if [ "$DRY" = 1 ]; then
  CK=${CKPT:-PICKED}; CK_STEP=STEP; CK_SHA=SHA
  echo "DRY pick ${PICK[*]}"
else
  PICK_LINE=$("${PICK[@]}" < /dev/null) || { echo "SCORE_DISTANCE_FAILED $RUN: no checkpoint (${PICK[*]})" >&2; exit 3; }
  read -r CK CK_STEP CK_EMA CK_SHA <<<"$PICK_LINE"
  if [ "$CK_EMA" != 1 ]; then
    echo "SCORE_DISTANCE_FAILED $RUN: $CK carries no EMA weights, and the study scores EMA" >&2; exit 3
  fi
fi
OUT=${OUT:-$D/results_spiderman/distance_study/$RUN/$(basename "$CK" .pt)_ema_ddim$STEPS}
LOG=$D/logs/distance_${RUN}_${SPACE}.log

RC=0
fail() { RC=1; echo "SCORE_DISTANCE_MAP_FAILED $*" >&2; [ "$DRY" = 1 ] || echo "$(date -Iseconds) FAILED $*" >> "$LOG"; }

missing_episodes() {   # missing_episodes <split> <latents dir>: split episodes without both files there
  "$PY" -c 'import json, os, sys
s = json.load(open(sys.argv[1]))
d = sys.argv[2]
print(" ".join(str(e) for e in s["val"] if not all(os.path.isfile(os.path.join(d, "ep_%05d_%s" % (e, x)))
                                                   for x in ("latents.npy", "meta.npz"))))' "$1" "$2" < /dev/null
}

score() {   # score <set> <map NN> <split> <horizon>
  local S=$1 M=$2 F=$3 K=$4 DIR LAT RAW KEY MISS E FPS
  DIR=$OUT/${S}_map${M}_h$K
  LAT=$(latents_dir "$S"); RAW=$(raw_dir "$S")
  local CMD=("$PY" "$RUN_REPO/eval_tf.py" "${BB[@]}" --latent-channels "$CH" --ckpt "$CK" --use-ema
             --tic-stride 1 --horizon-tics "$K" --latents-dir "$LAT" --split "$F" --subset val --parquet-dir "$RAW"
             --num-windows "$NUM_WINDOWS" --batch-size 16 --steps "$STEPS" --seed "$SEED" --context-frames 32
             --num-actions 29 --num-workers 0 --hf-cache "$D/hf/hub" ${DECODER[@]+"${DECODER[@]}"} --out-dir "$DIR")
  if [ "$DRY" = 1 ]; then
    echo "DRY check the episodes of $F exist in $LAT"
    echo "DRY eval_tf ${S}_map${M}_h$K ${CMD[*]}"
    return 0
  fi
  KEY="ckpt=$CK step=$CK_STEP sha256=$CK_SHA variant=ema space=$SPACE split_sha256=$(sha_of "$F") horizon=$K"
  KEY="$KEY windows=$NUM_WINDOWS seed=$SEED sampler=ddim$STEPS decoder=$DEC_TAG latents=$LAT raw=$RAW"
  if [ -f "$DIR/metrics.json" ] && [ "$RESCORE" != 1 ]; then
    if [ "$(cat "$DIR/score_key.txt" 2>/dev/null)" = "$KEY" ]; then
      echo "$(date -Iseconds) kept ${S}_map${M}_h$K (complete under this key)" >> "$LOG"; return 0
    fi
    fail "${S}_map${M}_h$K: $DIR holds a score made under a different key; RESCORE=1 replaces it"; return 0
  fi
  MISS=$(missing_episodes "$F" "$LAT") || { fail "${S}_map${M}_h$K: cannot read $F"; return 0; }
  if [ -n "$MISS" ]; then fail "${S}_map${M}_h$K: $LAT lacks episode(s) $MISS of $F"; return 0; fi
  mkdir -p "$DIR" && rm -f "$DIR/metrics.json" "$DIR/score_key.txt"
  echo "$(date -Iseconds) start ${S}_map${M}_h$K" >> "$LOG"
  "${CMD[@]}" > "$DIR/eval_tf.log" 2>&1 < /dev/null; E=$?
  if [ $E -ne 0 ] || [ ! -f "$DIR/metrics.json" ]; then fail "${S}_map${M}_h$K: eval_tf.py exit $E (see $DIR/eval_tf.log)"; return 0; fi
  printf '%s\n' "$KEY" > "$DIR/score_key.txt"
  { echo "checkout=$RUN_REPO code=$CODE"; echo "interpreter=$PY"; echo "decoder=$DEC_TAG"; echo "score_key=$KEY"
    echo "recorded=$(date -Iseconds)"; } > "$DIR/provenance.txt"
  FPS=$("$PY" -c 'import json, sys; print(round(json.load(open(sys.argv[1]))["sampling_frames_per_s"], 2))' \
        "$DIR/metrics.json" < /dev/null 2>/dev/null)
  echo "$(date -Iseconds) done ${S}_map${M}_h$K frames_per_s=${FPS:-?}" | tee -a "$LOG"
}

if [ "$DRY" = 1 ]; then
  echo "DRY $RUN space=$SPACE ckpt=$CK steps=$STEPS horizons='$HORIZONS' windows=$NUM_WINDOWS decoder=$DEC_TAG"
  echo "DRY checkout $RUN_REPO interpreter $PY out $OUT maps ${#LIST[@]}"
  echo "DRY lock $OUT/score.lock exclusively; a second scorer of the same output exits 5"
else
  export TMPDIR=$D/tmp/tmpdir
  mkdir -p "$TMPDIR" "$OUT" "$D/logs" || { echo "SCORE_DISTANCE_FAILED $RUN: cannot write under $D" >&2; exit 2; }
  export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
  exec 9>>"$OUT/score.lock" || { echo "SCORE_DISTANCE_FAILED $RUN: cannot open $OUT/score.lock" >&2; exit 2; }
  if ! perl -MFcntl=:flock -e 'open(my $fh, ">&=", 9) or exit 2; exit(flock($fh, LOCK_EX | LOCK_NB) ? 0 : 1)'; then
    echo "SCORE_DISTANCE_BUSY $RUN: another scorer holds $OUT/score.lock" >&2; exit 5
  fi
  cd "$RUN_REPO" || { echo "SCORE_DISTANCE_FAILED $RUN: no checkout at $RUN_REPO" >&2; exit 2; }
  CODE=$(git rev-parse HEAD 2>/dev/null || echo unversioned)
  echo "$(date -Iseconds) scoring ${#LIST[@]} map(s) of $RUN ckpt=$CK step=$CK_STEP sha256=$CK_SHA space=$SPACE" \
       "steps=$STEPS horizons='$HORIZONS' code=$CODE out=$OUT" | tee -a "$LOG"
fi

# stage by horizon: every map at one tic, then every map at four
for K in $HORIZONS; do
  for ITEM in "${LIST[@]}"; do
    read -r S M F <<<"$ITEM"
    score "$S" "$M" "$F" "$K"
  done
done

[ "$DRY" = 1 ] && exit 0
if [ $RC -ne 0 ]; then
  echo "SCORE_DISTANCE_FAILED $RUN (see $LOG)" | tee -a "$LOG" >&2; exit $RC
fi
echo "SCORE_DISTANCE_DONE $RUN space=$SPACE out=$OUT" | tee -a "$LOG"
