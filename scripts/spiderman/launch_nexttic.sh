#!/bin/bash
# NEXT-TIC rows on the dense corpus: the model predicts the frame 1 tic (28.6 ms) ahead instead of
# 1 agent decision (4 tics, 114 ms) ahead. GameNGen's spacing, on 2,000 dense episodes.
#
#   usage: [MB=..] [STEPS=400000] [TRAIN_IDS=0:2000] [ACTION_HISTORY=32] [INIT=..] [PHASE=1] \
#          [DRY=1] [DOOM_ROOT=..] launch_nexttic.sh <gpu | gpu,gpu> <unet | sd35 | pixart>
#
# What differs from the stride-4 rows (030-035) and what does not:
#
#   --tic-stride 1        the ONLY change to what the model learns. The sample shapes are identical
#                         (32 context latents channel-stacked, one target, one action), so every
#                         backbone is unmodified and the recipe below is the one those rows used.
#   context 32 tics       0.91 s of game time, against 3.66 s at stride 4. GameNGen uses 64 tics and
#                         reports 32 vs 64 as 0.05 dB, so 32 is the cheap end of a flat curve.
#   the dense corpus      latents_arnold_dense_pertic/arenas, training ids from $TRAIN_IDS only.
#                         Validation is a SEPARATE corpus (ids 6000-6099) so the in-training
#                         checkpoint-selection signal is held out; ids are checked against
#                         release/dense_split.json and the run refuses to start if they overlap.
#   the action            ACTION_HISTORY=32 (the default) is GameNGen's own conditioning: one token
#                         per context tic carrying the EXECUTED button vector of that tic, oldest
#                         first, the newest being the control applied from the last context frame
#                         into the target (`buttons[r-32:r]` for target row r; `buttons[r]` is
#                         chosen after the target is observed and is never included). The executed
#                         vector rather than Arnold's requested action id, because a per-tic corpus
#                         keeps the anti-stuck-override tics on which the two disagree.
#                         ACTION_HISTORY=0 falls back to the single action id of the stride-4 rows.
#
# Recipe, identical for every backbone (Rohan, Sep 20 2026): global batch 32, lr 5e-5 constant after
# 2,000 warmup, clip 1.0, fused AdamW, bf16 autocast, fp32 CPU EMA at 0.9999, context-noise
# augmentation max 0.7 with 10 buckets, no action dropout, v-prediction, seed 0, validation every
# 1k on 1,024 windows, recovery checkpoints every 5k keeping the last two.
#
# STOPPING BY HAND IS EXPECTED. The run is stopped at the numbers freeze, not at --steps, so
# --snapshot-every 10000 --local-snapshots writes a compact bf16 weight snapshot (live and EMA) every
# 10k that is never pruned: a stop at any point leaves snap_*.pt to evaluate, and best.pt is written
# at every validation improvement regardless.
#
# Starting weights are the row's PUBLIC PRETRAINED weights, as every other row in the paper (Rohan,
# Sep 20 2026: purity of the warm-start comparison). INIT=<checkpoint.pt> instead starts from one of
# our own finished runs' weights with a fresh optimizer and step 0, which is a different experiment
# and is recorded in config.json as init_from.
#
# PHASE=1 adds tics_since_decision conditioning (the target tic's position inside the 4-tic held
# action run). Off by default: it is an extra conditioning signal no prior work uses, so it belongs
# in an ablation, not in the row the paper reports.
#
# DRY=1 prints the command and stops before every side effect; DOOM_ROOT repoints the data root.
set -u
GPU=${1:?gpu}
BACKBONE=${2:?backbone (unet | sd35 | pixart)}
D=${DOOM_ROOT:-/sata2/data/rnagabhi/doom}
DRY=${DRY:-0}
STEPS=${STEPS:-400000}
TRAIN_IDS=${TRAIN_IDS:-0:2000}
VAL_IDS=${VAL_IDS:-6000:6100}
CTX=${CTX:-32}
ACTION_HISTORY=${ACTION_HISTORY:-32}
GLOBAL=32
L=$D/latents_arnold_dense_pertic/arenas
LVAL=$D/latents_arnold_dense_pertic_eval/val
NP=$(( $(echo "$GPU" | tr -cd , | wc -c) + 1 ))

# per-backbone: warm start, default micro-batch, interpreter, and the two SD 3.5 deviations
case $BACKBONE in
  unet)
    RUN=040-unet-nexttic; WARM=CompVis/stable-diffusion-v1-4; CH=4
    MB=${MB:-16}; PY=${PY:-$HOME/miniconda3/envs/doom/bin/python}; BB_FLAGS="" ;;
  pixart)
    RUN=041-pixart-nexttic; WARM=PixArt-alpha/PixArt-XL-2-512x512; CH=4
    MB=${MB:-32}; PY=${PY:-$HOME/miniconda3/envs/doom/bin/python}; BB_FLAGS="--action-inject token" ;;
  sd35)
    # the 16-channel corpus, its own latent directory, and the two deviations this row already carries
    RUN=042-sd35-nexttic; WARM=stabilityai/stable-diffusion-3.5-medium; CH=16
    MB=${MB:-16}; PY=${PY:-$HOME/wanenc/bin/python}
    BB_FLAGS="--grad-ckpt --skip-grad-norm 5 --skip-grad-after 3000"
    L=$D/latents_arnold_dense_pertic_sd35/arenas
    LVAL=$D/latents_arnold_dense_pertic_eval_sd35/val ;;
  *) echo "unknown backbone '$BACKBONE' (unet | sd35 | pixart)" >&2; exit 2 ;;
esac
R=$D/results_spiderman/$RUN

if [ $(( GLOBAL % (MB * NP) )) -ne 0 ]; then
  echo "per-GPU batch $MB on $NP card(s) cannot reach the global batch of $GLOBAL" >&2; exit 1
fi
ACCELERATE=${ACCELERATE:-"$PY -m accelerate.commands.launch"}
if [ "$NP" -gt 1 ]; then
  LAUNCHER="$ACCELERATE --num_processes $NP --mixed_precision bf16"
else
  LAUNCHER="$PY"
fi

# resume from the newest recovery checkpoint; empty on a first launch and on any machine without $R
[ -f $R/log.jsonl ] && { CK=$(ls $R/[0-9]*.pt 2>/dev/null | sort | tail -1); RES=${CK:+--resume $CK}; } || RES=""
# INIT is a first-launch-only knob: once a recovery checkpoint exists, resuming wins
INITF=""
[ -n "${INIT:-}" ] && [ -z "$RES" ] && INITF="--init-from $INIT"
PHASEF=""
[ "${PHASE:-0}" = 1 ] && PHASEF="--phase-conditioning"
if [ "$ACTION_HISTORY" != 0 ] && [ "$ACTION_HISTORY" != "$CTX" ]; then
  echo "ACTION_HISTORY=$ACTION_HISTORY must be 0 or equal CTX=$CTX (one control per context tic)" >&2; exit 1
fi

COMMON="--tic-stride 1 --action-history $ACTION_HISTORY --context-frames $CTX --num-actions 29 --noise-buckets 10 --noise-aug-max 0.7 \
 --global-batch $GLOBAL --per-gpu-batch $MB --lr 5e-5 --warmup 2000 --clip 1.0 --steps $STEPS \
 --objective v --action-dropout 0.0 --ema-every 8 --ema-decay 0.9999 --seed 0 \
 --latents-dir $L --val-latents-dir $LVAL --episode-ids $TRAIN_IDS --val-episode-ids $VAL_IDS \
 --dense-segment arenas --val-every 1000 --val-windows 1024 --ckpt-every 5000 \
 --snapshot-every 10000 --local-snapshots --keep-last 2 --num-workers 4 \
 $BB_FLAGS $PHASEF ${EXTRA:-}"
CMD="cd $D/repo && TMPDIR=$D/tmp/tmpdir CUDA_VISIBLE_DEVICES=$GPU $LAUNCHER train_wm.py \
 --backbone $BACKBONE --latent-channels $CH --warm-start $WARM --hf-cache $D/hf/hub \
 --results-dir $R $COMMON $INITF $RES >> $D/logs/train_${RUN}.log 2>&1"

[ "$DRY" = 1 ] && { echo "DRY $RUN $CMD"; exit 0; }

tmux has-session -t train-$BACKBONE-nexttic 2>/dev/null && { echo "$RUN alive"; exit 0; }
[ -f $R/log.jsonl ] && grep -q "\"event\": \"end\"" $R/log.jsonl && { echo "$RUN finished"; exit 0; }
[ -d $L ] || { echo "per-tic training latents missing at $L (run encode_nexttic.sh)" >&2; exit 1; }
[ -d $LVAL ] || { echo "per-tic validation latents missing at $LVAL (run encode_nexttic.sh CORPUS=val)" >&2; exit 1; }

export TMPDIR=$D/tmp/tmpdir; mkdir -p $TMPDIR $R $D/logs
cd $D/repo && git pull -q
echo "$(date -Iseconds) launching $RUN on gpu $GPU (world $NP, micro $MB, ids $TRAIN_IDS, steps $STEPS) ${INITF:-pretrained warm start} $RES" >> $D/logs/resumes.log
tmux new-session -d -s train-$BACKBONE-nexttic "$CMD"
echo "$RUN launched on gpu $GPU (world $NP, per-gpu batch $MB)"
