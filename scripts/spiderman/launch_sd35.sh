#!/bin/bash
# SD 3.5 Medium headline row (035-sd35-l32-aligned), the 16-channel-latent entry in the backbone
# comparison. Same recipe as 030/031/032/033/034: context 32, global batch 32, lr 5e-5, warmup
# 2000, clip 1.0, 90k updates, no action dropout, verified transitions, seed 0, validation every
# 1k on 1,024 windows, recovery checkpoints every 5k keeping the last two, fp32 CPU EMA.
# Runs once; safe to call repeatedly; resumes from the newest recovery checkpoint.
#
#   usage: [MB=16] [PY=..] [EXTRA=..] [DRY=1] launch_sd35.sh <gpu | gpu,gpu>
#
# Two deviations from the shared recipe, both stated in $R/DEVIATIONS.md and both to be stated in
# the paper:
#
#   --grad-ckpt        the 2.246B transformer needs about 44 to 46 GB allocated at batch 32 with
#                      activation checkpointing on (fit_sd35.sh measures it); without it the
#                      arithmetic puts the row over the 48 GB card. The other rows train without.
#   --skip-grad-norm 5 --skip-grad-after 3000 the PaLM-style spike guard that rescued the UniDiffuser row after its two
#                      gradient excursions. It skips the optimizer step, the schedule and the EMA
#                      when the pre-clip gradient norm exceeds 5, about 20x a healthy norm.
#
# Interpreter: SD3Transformer2DModel needs diffusers >= 0.29. The `doom` env is on 0.31 and ~/wanenc
# on 0.40 with transformers 4.57; verify_sd35.py's --real gates were run on the server under 0.40,
# so ~/wanenc is what this row is launched with and what its numbers should be read against.
#
# One card runs the plain interpreter and lets the trainer derive the accumulation from the global
# batch. Two cards run under accelerate, exactly as launch_video.sh does, with per-GPU 16 so that
# 16 x 2 = the global batch of 32 and no accumulation happens at all.
#
# HF_HUB_OFFLINE=0: the SD 3.5 repo is gated, the token sits at its default path (HF_HOME is left
# alone, as in vae_gate.sh) and the 4.2 GB transformer may still have to be fetched into $D/hf/hub.
set -u
GPU=${1:?gpu}
D=/sata2/data/rnagabhi/doom
R=$D/results_spiderman/035-sd35-l32-aligned
PY=${PY:-$HOME/wanenc/bin/python}
ACCELERATE=${ACCELERATE:-"$PY -m accelerate.commands.launch"}  # the wanenc env has the accelerate package but no CLI entry point
MB=${MB:-16}
GLOBAL=32
NP=$(( $(echo "$GPU" | tr -cd , | wc -c) + 1 ))

if [ $(( GLOBAL % (MB * NP) )) -ne 0 ]; then
  echo "per-GPU batch $MB on $NP card(s) cannot reach the global batch of $GLOBAL" >&2; exit 1
fi
if [ "$NP" -gt 1 ]; then
  LAUNCHER="$ACCELERATE launch --num_processes $NP --mixed_precision bf16"
else
  LAUNCHER="$PY"
fi

# resume from the newest recovery checkpoint; empty on a first launch (and on any machine without $R)
[ -f $R/log.jsonl ] && { CK=$(ls $R/[0-9]*.pt 2>/dev/null | sort | tail -1); RES=${CK:+--resume $CK}; } || RES=""

COMMON="--context-frames 32 --num-actions 29 --global-batch $GLOBAL --lr 5e-5 --warmup 2000 --clip 1.0 --steps 90000 --seed 0 --action-dropout 0.0 --require-verified-transitions --latents-dir $D/latents_arnold_sd35 --split $D/split_arnold.json --val-every 1000 --val-windows 1024 --ckpt-every 5000 --snapshot-every 5000 --keep-last 2 --num-workers 4 --grad-ckpt --skip-grad-norm 5 --skip-grad-after 3000 ${EXTRA:-}"
CMD="cd $D/repo && TMPDIR=$D/tmp/tmpdir HF_HUB_OFFLINE=0 CUDA_VISIBLE_DEVICES=$GPU $LAUNCHER train_wm.py --backbone sd35 --per-gpu-batch $MB --latent-channels 16 --warm-start stabilityai/stable-diffusion-3.5-medium --hf-cache $D/hf/hub --results-dir $R $COMMON $RES >> $D/logs/train_sd35.log 2>&1"

# DRY prints what tmux would be given and stops before every side effect, so the command can be
# read on a laptop. It deliberately skips the guards below, which need the server's filesystem.
[ "${DRY:-0}" = 1 ] && { echo "$CMD"; exit 0; }

tmux has-session -t train-sd35 2>/dev/null && { echo "sd35 alive"; exit 0; }
[ -f $R/log.jsonl ] && grep -q "\"event\": \"end\"" $R/log.jsonl && { echo "sd35 finished"; exit 0; }
[ -d $D/latents_arnold_sd35 ] || { echo "16-channel corpus missing at $D/latents_arnold_sd35"; exit 1; }

export TMPDIR=$D/tmp/tmpdir; mkdir -p $TMPDIR $R $D/logs
cd $D/repo && git pull -q
cat > $R/DEVIATIONS.md <<'MD'
# Recipe deviations on 035-sd35-l32-aligned

Two flags differ from the recipe the other aligned rows (030-034) train under. Both must appear in
the paper's description of this row.

1. `--grad-ckpt` (activation checkpointing). The 2.246B-parameter MMDiT-X does not fit the 48 GB
   card at global batch 32 without it. Costs roughly 30% throughput; changes no gradient.
2. `--skip-grad-norm 5 --skip-grad-after 3000` (spike guard). The optimizer step, the learning-rate schedule and the EMA
   are skipped on any update whose pre-clip gradient norm exceeds 5. This is the guard added for
   the UniDiffuser row after its excursions at updates 8,500 and 34,600; 5 is about 20x a healthy
   norm and above the benign 1.5 spikes. The trainer counts the skips in `skipped_updates`, and
   that count belongs in the paper next to this row.

Anything else about this row that differs from 030-034 is a property of the backbone (16-channel
latents, its own autoencoder and therefore its own VAE ceiling), not of the recipe.
MD
echo "$(date -Iseconds) launching sd35 on gpu $GPU (world $NP, micro $MB) $RES" >> $D/logs/resumes.log
tmux new-session -d -s train-sd35 "$CMD"
echo "sd35 launched on gpu $GPU (world $NP, per-gpu batch $MB)"
