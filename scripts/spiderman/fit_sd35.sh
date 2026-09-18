#!/bin/bash
# SD 3.5 Medium row: which per-GPU configuration to train it in. Usage: fit_sd35.sh <gpu>
#
# Global batch stays 32, the shared recipe, so "fill the card" means no gradient accumulation and
# no activation checkpointing whenever they fit. Three 200-update fit checks on synthetic windows
# (no corpus needed), in preference order:
#
#   b32_nockpt          per-GPU 32, accum 1, checkpointing off   <- fastest if it fits
#   b32_ckpt            per-GPU 32, accum 1, checkpointing on    <- trades ~30% speed for activations
#   b16_nockpt_accum2   per-GPU 16, accum 2, checkpointing off   <- fallback, two passes per update
#
# Every run logs updates/s and both peak memory numbers from the trainer's `fit_check` event into
# $D/logs/fit_sd35.log, and the script ends by printing the fastest configuration whose *reserved*
# peak stays under $BUDGET GB (44 by default: an A6000 is 48 GB and the card is shared). Reserved,
# not allocated, is the number that decides whether a configuration fits.
#
# fused AdamW (the trainer picks it on cuda), bf16 autocast with fp32 master weights and SDPA
# attention (diffusers' JointAttnProcessor2_0) are already the defaults; nothing here turns them on.
# torch.compile is deliberately NOT used: see the note at the bottom of this file.
#
# The transformer is a gated repo. HF_HUB_OFFLINE=0 on the first run so it downloads (about 4.2 GB)
# into $D/hf/hub; the token has to sit at its default path, so HF_HOME is left alone, as in
# vae_gate.sh. Set PY= to override the interpreter.
set -u
GPU=${1:?gpu}
D=/sata2/data/rnagabhi/doom
PY=${PY:-$HOME/miniconda3/envs/doom/bin/python}
BUDGET=${BUDGET:-44}
LOG=$D/logs/fit_sd35.log
export TMPDIR=$D/tmp/tmpdir HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
mkdir -p $TMPDIR $D/logs
cd $D/repo || exit 1

$PY -c 'import diffusers; from diffusers import SD3Transformer2DModel; print("diffusers", diffusers.__version__)' \
  || { echo "no SD3Transformer2DModel in this interpreter ($PY); SD3 needs diffusers >= 0.29, try PY=~/wanenc/bin/python" | tee -a $LOG; exit 1; }

for cfg in "b32_nockpt 32 0" "b32_ckpt 32 1" "b16_nockpt_accum2 16 0"; do
  set -- $cfg; NAME=$1; B=$2; CK=$3
  [ "$CK" = 1 ] && CKFLAG=--grad-ckpt || CKFLAG=""
  R=$D/tmp/fit_sd35_$NAME
  rm -rf "$R"
  echo "=== $(date -u) $NAME per-gpu $B ckpt $CK"
  CUDA_VISIBLE_DEVICES=$GPU $PY train_wm.py --backbone sd35 --fit-check 200 \
    --per-gpu-batch "$B" --global-batch 32 --context-frames 32 --num-actions 29 \
    --warm-start stabilityai/stable-diffusion-3.5-medium --hf-cache $D/hf/hub \
    --lr 5e-5 --warmup 2000 --clip 1.0 --action-dropout 0.0 --seed 0 \
    --num-workers 2 --results-dir "$R" $CKFLAG > $D/logs/fit_sd35_$NAME.log 2>&1
  code=$?
  line=$(grep '"event": "fit_check"' "$R/log.jsonl" 2>/dev/null | tail -1)
  echo "$(date -Iseconds) $NAME exit=$code ${line:-NO_FIT_CHECK_EVENT (see $D/logs/fit_sd35_$NAME.log)}" >> $LOG
  echo "$NAME exit=$code ${line:-no fit_check event}"
done

$PY - "$LOG" "$BUDGET" <<'PICK'
import json, sys
log, budget = sys.argv[1], float(sys.argv[2])
order = ["b32_nockpt", "b32_ckpt", "b16_nockpt_accum2"]
rows = {}
for raw in open(log):
    for name in order:
        if f" {name} exit=" in raw and "{" in raw:
            try:
                rows[name] = json.loads(raw[raw.index("{"):])
            except ValueError:
                pass
if not rows:
    print("NO USABLE FIT CHECK: see the per-config logs"); raise SystemExit(1)
print(f"\n{'config':20} {'updates/s':>10} {'alloc GB':>9} {'reserved GB':>12} {'accum':>6} {'ckpt':>5}")
fits = []
for name in order:
    r = rows.get(name)
    if not r:
        print(f"{name:20} {'FAILED':>10}"); continue
    res = r.get("peak_reserved_gb", r.get("peak_mem_gb", 0))
    print(f"{name:20} {r['steps_per_s']:10.3f} {r.get('peak_mem_gb', 0):9.2f} {res:12.2f} {r.get('accum', '?'):>6} {str(r.get('grad_ckpt')):>5}")
    if res and res <= budget:
        fits.append((r["steps_per_s"], name, r))
if not fits:
    print(f"\nNOTHING FITS UNDER {budget} GB: the row needs a smaller per-GPU batch or 8-bit Adam"); raise SystemExit(1)
sps, name, r = max(fits)
print(f"\nCHOOSE {name}: {sps:.3f} updates/s, {r.get('peak_reserved_gb')} GB reserved, "
      f"{90000 / sps / 3600:.1f} h for 90k updates, {30000 / sps / 3600:.1f} h for 30k")
PICK
echo "=== $(date -u) FIT_SD35_DONE"

# torch.compile: not enabled, and not safe to switch on casually here. `torch.compile(module)`
# returns an OptimizedModule whose state dict is re-prefixed `_orig_mod.`, which breaks the strict
# checkpoint load and the parameter-name EMA keying in train_wm.py. The in-place form,
# `model.transformer.compile()`, leaves the state dict alone and is the only variant worth trying,
# but it still recompiles per input shape (training batch, validation batch, sampler batch) and
# interacts with activation checkpointing, so it belongs in its own timed experiment, not in the
# configuration that the headline row is launched with.
