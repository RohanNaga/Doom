#!/bin/bash
# NEXT-TIC rows on the dense corpus: the model predicts the frame 1 tic (28.6 ms) ahead instead of
# 1 agent decision (4 tics, 114 ms) ahead, on 2,000 dense episodes. Per-tic spacing is our inference
# about GameNGen (action repeat 4 in its App. A.5, evaluation on "35FPS data" in A.6); its training
# stride is not stated.
#
#   usage: [MB=32] [WORKERS=12] [STEPS=400000] [TRAIN_IDS=0:2000] [ACTION_HISTORY=32] [INIT=..] \
#          [PHASE=1] [GRAD_CKPT=1] [FIT=20] [ALLOW_ACCUM=1] [ALLOW_PARTIAL=1] [GATE_RUN=1] \
#          [ALLOW_UNGATED=1] [DRY=1] [DOOM_ROOT=..] launch_nexttic.sh <gpu | gpu,gpu> <unet | sd35 | pixart>
#
# A launch refuses unless $D/GATES_COMMIT (written by scripts/cluster/gates.sh) names the commit
# $D/repo is at, with no tracked change; see "PIN WHAT LAUNCHES" below.
#
# FILL THE CARD (CLAUDE.md, Rohan Sep 17 2026). Every job uses the whole card it holds: the micro-batch
# IS the global batch of 32 and there is no gradient accumulation. The launcher refuses MB * cards != 32
# rather than quietly accumulating; ALLOW_ACCUM=1 overrides it and says so in the resume log. The SD 3.5
# row measured 36.4 GB at micro-batch 32 with --grad-ckpt, so 32 is known to fit there; GRAD_CKPT=1 turns
# checkpointing on for the U-Net or PixArt if a fit check says 32 does not fit without it.
#
# FIT=<steps> runs `--fit-check <steps>` with these exact arguments instead of launching, and prints
# updates/s and peak allocated and reserved memory. Run it before every multi-hour launch; that is the
# throughput sweep the fill-the-card rule asks for.
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
#   the action            ACTION_HISTORY=32 (the default) follows GameNGen's idea of one learned token
#                         per past action (its §3.2; vocabulary, embedding and positions undisclosed);
#                         the 19-bit executed-control MLP and learned positions here are ours. One token
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
MB=${MB:-32}                    # fill the card: the micro-batch IS the global batch
WORKERS=${WORKERS:-12}          # a per-tic corpus does not fit the page cache; the loader is seek-bound
GLOBAL=32
L=$D/latents_arnold_dense_pertic/arenas
LVAL=$D/latents_arnold_dense_pertic_eval/val
NP=$(( $(echo "$GPU" | tr -cd , | wc -c) + 1 ))

# per-backbone: warm start, default micro-batch, interpreter, and the two SD 3.5 deviations
case $BACKBONE in
  unet)
    RUN=040-unet-nexttic; WARM=CompVis/stable-diffusion-v1-4; CH=4
    PY=${PY:-$HOME/miniconda3/envs/doom/bin/python}; BB_FLAGS="" ;;
  pixart)
    RUN=041-pixart-nexttic; WARM=PixArt-alpha/PixArt-XL-2-512x512; CH=4
    PY=${PY:-$HOME/miniconda3/envs/doom/bin/python}; BB_FLAGS="--action-inject token" ;;
  sd35)
    # the 16-channel corpus, its own latent directory, and the two deviations this row already carries
    RUN=042-sd35-nexttic; WARM=stabilityai/stable-diffusion-3.5-medium; CH=16
    PY=${PY:-$HOME/wanenc/bin/python}
    BB_FLAGS="--grad-ckpt --skip-grad-norm 5 --skip-grad-after 3000"
    L=$D/latents_arnold_dense_pertic_sd35/arenas
    LVAL=$D/latents_arnold_dense_pertic_eval_sd35/val ;;
  *) echo "unknown backbone '$BACKBONE' (unet | sd35 | pixart)" >&2; exit 2 ;;
esac
R=$D/results_spiderman/$RUN

if [ $(( GLOBAL % (MB * NP) )) -ne 0 ]; then
  echo "per-GPU batch $MB on $NP card(s) cannot reach the global batch of $GLOBAL" >&2; exit 1
fi
if [ $(( MB * NP )) -ne "$GLOBAL" ] && [ "${ALLOW_ACCUM:-0}" != 1 ]; then
  echo "MB=$MB on $NP card(s) is a global batch of $(( MB * NP )), so reaching $GLOBAL needs gradient" >&2
  echo "accumulation of $(( GLOBAL / (MB * NP) )). The fill-the-card rule (CLAUDE.md) forbids accumulation:" >&2
  echo "use the largest micro-batch that fits, with GRAD_CKPT=1 if it does not fit otherwise. Measure it" >&2
  echo "with FIT=20 first. Set ALLOW_ACCUM=1 to override deliberately." >&2
  exit 1
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
CKPTF=""
[ "${GRAD_CKPT:-0}" = 1 ] && [ "$BACKBONE" != sd35 ] && CKPTF="--grad-ckpt"
PARTIALF=""
[ "${ALLOW_PARTIAL:-0}" = 1 ] && PARTIALF="--allow-partial"
if [ "$ACTION_HISTORY" != 0 ] && [ "$ACTION_HISTORY" != "$CTX" ]; then
  echo "ACTION_HISTORY=$ACTION_HISTORY must be 0 or equal CTX=$CTX (one control per context tic)" >&2; exit 1
fi

COMMON="--tic-stride 1 --action-history $ACTION_HISTORY --context-frames $CTX --num-actions 29 --noise-buckets 10 --noise-aug-max 0.7 \
 --global-batch $GLOBAL --per-gpu-batch $MB --lr 5e-5 --warmup 2000 --clip 1.0 --steps $STEPS \
 --objective v --action-dropout 0.0 --ema-every 8 --ema-decay 0.9999 --seed 0 \
 --latents-dir $L --val-latents-dir $LVAL --episode-ids $TRAIN_IDS --val-episode-ids $VAL_IDS \
 --dense-segment arenas --val-every 1000 --val-windows 1024 --ckpt-every 5000 \
 --snapshot-every 10000 --local-snapshots --keep-last 2 --num-workers $WORKERS \
 $BB_FLAGS $PHASEF $CKPTF $PARTIALF ${EXTRA:-}"
CMD="cd $D/repo && TMPDIR=$D/tmp/tmpdir CUDA_VISIBLE_DEVICES=$GPU $LAUNCHER train_wm.py \
 --backbone $BACKBONE --latent-channels $CH --warm-start $WARM --hf-cache $D/hf/hub \
 --results-dir $R $COMMON $INITF $RES >> $D/logs/train_${RUN}.log 2>&1"

# FIT replaces the launch with a throughput and memory measurement of this exact configuration
if [ -n "${FIT:-}" ]; then
  FITCMD="cd $D/repo && TMPDIR=$D/tmp/tmpdir CUDA_VISIBLE_DEVICES=$GPU $PY train_wm.py \
 --backbone $BACKBONE --latent-channels $CH --warm-start $WARM --hf-cache $D/hf/hub \
 --results-dir $R/fitcheck --fit-check $FIT $COMMON"
  [ "$DRY" = 1 ] && { echo "DRY $RUN fit $FITCMD"; exit 0; }
  mkdir -p $D/tmp/tmpdir $R/fitcheck
  eval "$FITCMD"
  exit $?
fi

[ "$DRY" = 1 ] && { echo "DRY $RUN $CMD"; exit 0; }

tmux has-session -t train-$BACKBONE-nexttic 2>/dev/null && { echo "$RUN alive"; exit 0; }
[ -f $R/log.jsonl ] && grep -q "\"event\": \"end\"" $R/log.jsonl && { echo "$RUN finished"; exit 0; }
[ -d $L ] || { echo "per-tic training latents missing at $L (run encode_nexttic.sh)" >&2; exit 1; }
[ -d $LVAL ] || { echo "per-tic validation latents missing at $LVAL (run encode_nexttic.sh CORPUS=val)" >&2; exit 1; }

# PIN WHAT LAUNCHES (docs/REVIEW_2026-09-22.md H3). This used to `git pull` here, after the gates
# had passed, and a failed pull did not stop the launch, so the code that trained could differ from
# the code the gates certified. Nothing here changes the checkout now. `scripts/cluster/gates.sh`
# writes `$D/GATES_COMMIT` naming the commit it certified, and the launch refuses unless the
# checkout it runs ($D/repo) is at that commit with no tracked change. GATE_RUN=1 marks the gates'
# own smoke run, which comes before any receipt; ALLOW_UNGATED=1 launches anyway and says so in
# resumes.log. A resume goes through the same check, so the code cannot change mid-run either.
RECEIPT=${GATES_RECEIPT:-$D/GATES_COMMIT}
HEAD_SHA=$(git -C "$D/repo" rev-parse HEAD 2>/dev/null) || HEAD_SHA=""
refuse() { echo "$RUN not launched: $*" >&2; exit 1; }
if [ "${GATE_RUN:-0}" = 1 ]; then
  PIN="gate run (the gates are certifying ${HEAD_SHA:-an unversioned checkout})"
elif [ "${ALLOW_UNGATED:-0}" = 1 ]; then
  PIN="UNGATED (ALLOW_UNGATED=1)"
else
  [ -n "$HEAD_SHA" ] || refuse "$D/repo is not a git checkout, so what launches cannot be pinned"
  [ -s "$RECEIPT" ] || refuse "no gate receipt at $RECEIPT: run scripts/cluster/gates.sh on this checkout first (ALLOW_UNGATED=1 overrides, and is recorded)"
  CERT=$(awk '{print $1; exit}' "$RECEIPT")
  [ "$CERT" = "$HEAD_SHA" ] || refuse "$D/repo is at $HEAD_SHA but the gates certified $CERT; check out $CERT or rerun the gates"
  git -C "$D/repo" diff --quiet HEAD -- 2>/dev/null \
    || refuse "tracked files in $D/repo differ from $HEAD_SHA, so this is not the code the gates certified"
  PIN="certified by $RECEIPT"
fi

export TMPDIR=$D/tmp/tmpdir; mkdir -p $TMPDIR $R $D/logs
echo "$(date -Iseconds) launching $RUN on gpu $GPU (world $NP, micro $MB, ids $TRAIN_IDS, steps $STEPS) ${INITF:-pretrained warm start} $RES commit ${HEAD_SHA:-?} $PIN" >> $D/logs/resumes.log
tmux new-session -d -s train-$BACKBONE-nexttic "$CMD"
echo "$RUN launched on gpu $GPU (world $NP, per-gpu batch $MB)"
