#!/usr/bin/env bash
# Superman: split the fresh arena set, fit-check the LoRA trainer once, then run the U-Net step curve
# on the 13 unseen arenas across seven A4000s (arenas 8, 16, 6, 7 first). Training only; scoring runs
# later from the saved checkpoints with the tuned decoder. Run inside tmux:
#   tmux new -d -s sm-lora 'bash ~/Doom/lora_arenas13.sh 2>&1 | tee -a ~/Doom/logs/lora_arenas13.log'
# Gates before launch (checked here): the encode marker, the snapshot, the SD 1.4 cache, W&B login.
set -euo pipefail
H=/home/rohan/Doom
REPO=${REPO:-$H/repo_main}
PY=/home/rohan/miniconda3/envs/Doom/bin/python
LAT=$H/data/latents_arnold_eval_v2_pertic/arenas13
MANIFEST=$H/data/raw_arnold_eval_v2/arenas13/manifest.json
SNAP=$H/data/results_040/snap_0200000.pt
HF=$H/hf
SPLITS=$H/results/adapt_splits
RUNS=$H/results/adapt
GPUS=${GPUS:-"1 2 3 4 5 6 7"}
MAPS=${MAPS:-"8 16 6 7 17 15 12 14 11 13 10 1 9"}     # pilot four first, then near to far by the frozen D
SEED=${SEED:-0}
K=${K:-8}                                              # adaptation episodes for the step curve
PER_GPU=${PER_GPU:-8}                                  # micro-batch on a 16 GB card; the fit check decides
EXTRA=${EXTRA:-"--allow-accumulation --grad-ckpt"}     # global batch 32 stays the recipe
WANDB=${WANDB:-}                                       # set to --no-wandb only for a throwaway
ts() { date -Is; }
mkdir -p "$SPLITS" "$RUNS" "$H/logs"

echo "$(ts) gates"
grep -q ENCODE_ARENAS13_SD1_DONE "$H/logs/encode_arenas13_sd1.log" || { echo "encode not done"; exit 2; }
[ -f "$SNAP" ] || { echo "snapshot missing: $SNAP"; exit 2; }
[ -d "$HF/hub/models--CompVis--stable-diffusion-v1-4" ] || { echo "SD 1.4 cache missing under $HF/hub"; exit 2; }
[ -n "$WANDB" ] || grep -q api.wandb.ai ~/.netrc 2>/dev/null || { echo "no W&B login on this host (run wandb login) or set WANDB=--no-wandb"; exit 2; }
cd "$REPO"; echo "$(ts) repo $(git rev-parse --short HEAD)"

echo "$(ts) splits (seed $SEED) for maps: $MAPS"
for m in $MAPS; do
  f=$SPLITS/split_adapt_arenas13_map$(printf %02d $m)_seed$SEED.json
  [ -f "$f" ] || $PY adapt_split.py --latents-dir "$LAT" --manifest "$MANIFEST" --map $m --seed $SEED --out-dir "$SPLITS" \
    > "$H/logs/split_map$m.log" 2>&1 || { echo "split failed for map $m"; cat "$H/logs/split_map$m.log" | tail -5; exit 3; }
done
ls "$SPLITS" | head -20

FIRST=$(echo $MAPS | awk '{print $1}'); FGPU=$(echo $GPUS | awk '{print $1}')
echo "$(ts) fit check on GPU $FGPU, map $FIRST, per-gpu $PER_GPU $EXTRA"
CUDA_VISIBLE_DEVICES=$FGPU $PY adapt_wm.py --source "$SNAP" --adapt-split "$SPLITS/split_adapt_arenas13_map$(printf %02d $FIRST)_seed$SEED.json" \
  --latents-dir "$LAT" --results-dir "$RUNS/fitcheck_map$FIRST" --per-gpu-batch $PER_GPU --global-batch 32 $EXTRA \
  --adapt-episodes-k $K --warm-start CompVis/stable-diffusion-v1-4 --hf-cache "$HF" --seed $SEED --fit-check 20 --no-wandb \
  2>&1 | tee "$H/logs/fitcheck.log" | tail -25
grep -i "parity\|updates/s\|peak\|steps/s\|memory" "$H/logs/fitcheck.log" | head -10 || true
[ "${FIT_ONLY:-0}" = "1" ] && { echo "$(ts) FIT_ONLY set; stopping after the fit check"; exit 0; }

echo "$(ts) launching runs"
i=0; gpus=($GPUS); n=${#gpus[@]}
for m in $MAPS; do
  g=${gpus[$((i % n))]}
  name=unet200k_arenas13_map$(printf %02d $m)_r16_k${K}_s$SEED
  tmux new -d -s "lora-map$m" "cd $REPO && CUDA_VISIBLE_DEVICES=$g $PY adapt_wm.py --source $SNAP --adapt-split $SPLITS/split_adapt_arenas13_map$(printf %02d $m)_seed$SEED.json --latents-dir $LAT --results-dir $RUNS/$name --per-gpu-batch $PER_GPU --global-batch 32 $EXTRA --adapt-episodes-k $K --warm-start CompVis/stable-diffusion-v1-4 --hf-cache $HF --seed $SEED $WANDB 2>&1 | tee $H/logs/$name.log; echo LORA_RUN_DONE map=$m >> $H/logs/lora_arenas13.log"
  echo "$(ts) map $m on GPU $g (tmux lora-map$m)"
  i=$((i+1))
  # seven at a time: wait for a free slot before launching the eighth and later
  while [ "$(tmux ls 2>/dev/null | grep -c '^lora-map')" -ge "$n" ]; do sleep 60; done
done
echo "$(ts) all runs launched; waiting"
while [ "$(tmux ls 2>/dev/null | grep -c '^lora-map')" -gt 0 ]; do sleep 120; done
echo "$(ts) LORA_ARENAS13_TRAINING_DONE"
