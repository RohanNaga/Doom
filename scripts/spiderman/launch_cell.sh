#!/bin/bash
# One cell of the twelve-cell knob grid: 30k updates of the base PixArt-alpha recipe with exactly
# one knob turned, then the evaluation suite after_run2.sh runs, at the cell's context length.
#
# Base recipe (identical for every cell): PixArt-alpha 512 warm start, context 32, global batch 32
# as one micro-batch of 32 with no accumulation and no gradient checkpointing (the Sep 14 fit check
# measured 26.41 GB and 0.68 updates/s that way), lr 5e-5, warmup 2000, action dropout 0, verified
# transitions, v-prediction, seed 0, validation every 1k, EMA kept in every recovery checkpoint.
#
# Resume-safe. Re-running a cell: training resumes from its newest recovery checkpoint, every
# evaluation stage whose output file already exists is skipped, and a cell that already wrote
# CELL_DONE exits immediately. A lock directory keeps two queues off the same cell.
#
# Runs in the foreground: grid_queue.sh is the thing that owns a tmux session.
#   launch_cell.sh <gpu> <cell>
#   GRID_FIT_CHECK=100 launch_cell.sh 2 base30k     # throughput and memory sweep first (fill-the-card rule)
#   GRID_EMA_ROLLOUTS=1 launch_cell.sh 2 base30k    # also roll out the EMA weights (doubles rollout hours)
set -u
GPU=${1:?gpu}; CELL=${2:?cell}
D=/sata2/data/rnagabhi/doom
PY=${PY:-~/miniconda3/envs/doom/bin/python}
PIXART=PixArt-alpha/PixArt-XL-2-512x512
R=$D/results_spiderman/grid/$CELL
LOG=$D/logs/grid_${CELL}.log
export TMPDIR=$D/tmp/tmpdir
mkdir -p "$TMPDIR" "$R" "$D/logs"

# ---- the cell table: one knob per cell, appended after the base flags so it overrides them ----
CTX=32; KNOB=""
case $CELL in
  base30k)    ;;                                     # 1. the reference cell, no change
  data-1_8)   KNOB="--train-fraction 0.125" ;;       # 2. an eighth of the training episodes
  data-1_4)   KNOB="--train-fraction 0.25" ;;        # 3.
  data-1_2)   KNOB="--train-fraction 0.5" ;;         # 4. (the three are nested, see select_train_episodes)
  scratch)    KNOB="--warm-start none" ;;            # 5. PixArt architecture, random init
  eps)        KNOB="--objective eps" ;;              # 6. epsilon prediction, same betas and sampler
  ctx8)       CTX=8 ;;                               # 7.
  ctx16)      CTX=16 ;;                              # 8.
  noaug)      KNOB="--noise-aug-max 0.0" ;;          # 9. context-noise augmentation off
  adaln)      KNOB="--action-inject adaln" ;;        # 10. action and bucket added into adaLN-single
  lr1e-4)     KNOB="--lr 1e-4" ;;                    # 11.
  lr2.5e-5)   KNOB="--lr 2.5e-5" ;;                  # 12.
  *) echo "unknown cell '$CELL'; see the case table in $0" >&2; exit 2 ;;
esac

note() { echo "$(date -Iseconds) $*" >> "$LOG"; }   # never inline $? next to a command substitution

grep -q "CELL_DONE $CELL" "$LOG" 2>/dev/null && { echo "$CELL already complete"; exit 0; }
mkdir "$R/.lock" 2>/dev/null || { echo "$CELL is locked by another queue ($R/.lock); remove it if that queue died" >&2; exit 3; }
trap 'rmdir "$R/.lock" 2>/dev/null' EXIT

export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1
cd "$D/repo" || exit 4
GITSHA=$(git rev-parse --short HEAD)
note "cell $CELL gpu $GPU ctx $CTX knob '${KNOB:-none}' git $GITSHA"

BASE="--backbone pixart --warm-start $PIXART --hf-cache $D/hf/hub \
 --context-frames $CTX --num-actions 29 --noise-buckets 10 --noise-aug-max 0.7 \
 --per-gpu-batch 32 --global-batch 32 --lr 5e-5 --warmup 2000 --clip 1.0 --steps 30000 \
 --objective v --train-fraction 1.0 --action-inject token --action-dropout 0.0 --seed 0 \
 --require-verified-transitions --latents-dir $D/latents_arnold_aligned --split $D/split_arnold.json \
 --val-every 1000 --val-windows 1024 --ckpt-every 5000 --snapshot-every 5000 --keep-last 2 --num-workers 4 \
 --results-dir $R"

# ---- optional fit check: updates/s and peak/reserved memory for this cell's configuration -----
if [ -n "${GRID_FIT_CHECK:-}" ]; then
  $PY train_wm.py $BASE $KNOB --fit-check "$GRID_FIT_CHECK" --results-dir "$R/fitcheck" >> "$D/logs/grid_fitcheck_${CELL}.log" 2>&1
  note "$CELL fit check exit $?"
fi

# ---- train ------------------------------------------------------------------------------------
if grep -q '"event": "end"' "$R/log.jsonl" 2>/dev/null; then
  note "$CELL training already at 30k"
else
  CK=$(ls "$R"/[0-9]*.pt 2>/dev/null | sort | tail -1)
  RES=${CK:+--resume $CK}
  note "$CELL training ${RES:-from the warm start}"
  $PY train_wm.py $BASE $KNOB $RES >> "$D/logs/grid_train_${CELL}.log" 2>&1
  RC=$?
  note "$CELL training exit $RC"
  [ $RC -eq 0 ] || exit $RC
fi
[ -f "$R/best.pt" ] || { note "$CELL has no best.pt; nothing to evaluate"; exit 5; }

# ---- evaluation, exactly the suite after_run2.sh runs, at this cell's context length ----------
COMMON="--backbone pixart --pixart-path $PIXART --vae-path $D/vae_decoder_arnold_lpips/vae \
 --hf-cache $D/hf/hub --context-frames $CTX --num-actions 29"
TF="--subset val --num-windows 2048 --batch-size 16 --steps 50"
# corpus -> (latents, parquet, split); unseen2 is the 13-map transfer corpus from eval_unseen2.sh
corpus_args() {
  case $1 in
    seen)    echo "--latents-dir $D/latents_arnold_eval/seen --parquet-dir $D/raw_arnold_eval/seen --split $D/latents_arnold_eval/split_seen.json" ;;
    unseen)  echo "--latents-dir $D/latents_arnold_eval/unseen --parquet-dir $D/raw_arnold_eval/unseen --split $D/latents_arnold_eval/split_unseen.json" ;;
    unseen2) echo "--latents-dir $D/latents_arnold_eval/unseen2 --parquet-dir $D/raw_arnold_eval/unseen2 --split $D/latents_arnold_eval/split_unseen2.json" ;;
  esac
}

# EMA weights live only in the recovery checkpoints, so the EMA rows are scored on the newest one
LAST=$(ls "$R"/[0-9]*.pt 2>/dev/null | sort | tail -1)

for S in seen unseen unseen2; do
  if [ -f "$R/eval_tf_$S/metrics.json" ]; then
    note "$CELL eval_tf_$S already scored"
  else
    $PY eval_tf.py $COMMON $(corpus_args $S) $TF --ckpt "$R/best.pt" --out-dir "$R/eval_tf_$S" \
      >> "$D/logs/grid_${CELL}_eval_tf_$S.log" 2>&1
    note "$CELL eval_tf_$S exit $?"
  fi
  [ -n "$LAST" ] || continue
  if [ -f "$R/eval_tf_${S}_ema/metrics.json" ]; then
    note "$CELL eval_tf_${S}_ema already scored"
  else
    $PY eval_tf.py $COMMON $(corpus_args $S) $TF --ckpt "$LAST" --use-ema --out-dir "$R/eval_tf_${S}_ema" \
      >> "$D/logs/grid_${CELL}_eval_tf_${S}_ema.log" 2>&1
    note "$CELL eval_tf_${S}_ema exit $?"
  fi
done

# 256 rollouts to horizon 64 on the seen corpus, then drift/IDM scoring and FVD 16/32
roll() {   # roll <suffix> <ckpt> [--use-ema]
  local SUF=$1 CKPT=$2; shift 2
  local NPZ=$R/rollouts_seen${SUF}.npz MDIR=$R/rollout_metrics_seen${SUF}
  if [ ! -f "$NPZ" ]; then
    $PY rollout_eval.py --rollout $COMMON --ckpt "$CKPT" "$@" \
      --latents-dir $D/latents_arnold_eval/seen --split $D/latents_arnold_eval/split_seen.json \
      --subset val --num-rollouts 256 --horizon 64 --batch-size 16 --steps 50 --out "$NPZ" \
      >> "$D/logs/grid_${CELL}_rollout${SUF}.log" 2>&1
    note "$CELL rollout${SUF} exit $?"
  fi
  if [ ! -f "$MDIR/drift.json" ]; then
    $PY rollout_eval.py --score --rollouts "$NPZ" --idm $D/results_spiderman/idm_aligned/idm.pt \
      --vae-path $D/vae_decoder_arnold_lpips/vae --out-dir "$MDIR" --save-clips 256 \
      >> "$D/logs/grid_${CELL}_rollout${SUF}.log" 2>&1
    note "$CELL rollout score${SUF} exit $?"
  fi
  for F in 16 32; do
    if [ ! -f "$MDIR/fvd$F.json" ]; then
      $PY fvd.py --clips "$MDIR/clips_u8.npz" --frames $F --i3d $D/weights/i3d_torchscript.pt \
        --out "$MDIR/fvd$F.json" >> "$D/logs/grid_${CELL}_rollout${SUF}.log" 2>&1
      note "$CELL fvd$F${SUF} exit $?"
    fi
  done
}
roll "" "$R/best.pt"
[ -n "${GRID_EMA_ROLLOUTS:-}" ] && [ -n "$LAST" ] && roll "_ema" "$LAST" --use-ema

echo "CELL_DONE $CELL" >> "$LOG"
