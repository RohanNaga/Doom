#!/usr/bin/env bash
# Superman side of the fresh arena set: wait for Spiderman's recording to finish, pull the kept
# episodes, encode them into SD 1 every-tic latents on seven A4000s, verify, and push the latents
# back to Spiderman. Idempotent: every step skips what already exists. Run inside tmux on Superman:
#   tmux new -d -s sm-encode-v2 'bash ~/Doom/encode_arenas13_sd1.sh 2>&1 | tee -a ~/Doom/logs/encode_arenas13_sd1.log'
set -euo pipefail

SPIDER=rnagabhi@128.2.204.110
D=/sata2/data/rnagabhi/doom                      # Spiderman
RAW_REMOTE=$D/raw_arnold_eval_v2/arenas13
LOG_REMOTE=$RAW_REMOTE/RECORDING_LOG.txt
CANON_REMOTE=$D/latents_arnold_dense_pertic/canonical_controls.json
LAT_REMOTE=$D/latents_arnold_eval_v2_pertic/arenas13

H=/home/rohan/Doom                               # Superman
RAW=$H/data/raw_arnold_eval_v2/arenas13
LAT=$H/data/latents_arnold_eval_v2_pertic/arenas13
CANON=$H/data/canonical_controls.json
REPO=$H/repo                                     # pinned 190c125
PY=/home/rohan/miniconda3/envs/Doom/bin/python
GPUS=${GPUS:-"1 2 3 4 5 6 7"}                    # GPU 0 belongs to another user tonight
NSHARD=$(echo $GPUS | wc -w)
FLAGS=${FLAGS:-"--every-tic --stride 4 --batch-size 64 --decode-threads 6 --decode-workers 0 --dtype bf16 --decode-check 16 --episode-ids 16:328"}
EXPECT=${EXPECT:-312}

ts() { date -Is; }
mkdir -p "$RAW" "$LAT" "$H/logs"

echo "$(ts) waiting for RECORD_EVAL_V2_DONE on Spiderman"
until ssh -o BatchMode=yes -o ConnectTimeout=15 "$SPIDER" "grep -q RECORD_EVAL_V2_DONE $LOG_REMOTE" 2>/dev/null; do sleep 120; done
echo "$(ts) recording done; pulling kept episodes"

# The set-aside first episodes live under _first_episode_discarded/ and are excluded.
rsync -a --info=progress2 --exclude '_first_episode_discarded/' --exclude 'arnold_dump/' \
  "$SPIDER:$RAW_REMOTE/" "$RAW/"
rsync -a "$SPIDER:$CANON_REMOTE" "$CANON"
NRAW=$(ls "$RAW"/ep_*.parquet | wc -l)
echo "$(ts) pulled $NRAW parquet episodes (expect $EXPECT)"
[ "$NRAW" -eq "$EXPECT" ] || { echo "episode count mismatch"; exit 2; }

echo "$(ts) encoding on GPUs: $GPUS ($NSHARD shards)"
i=0
for g in $GPUS; do
  ( cd "$REPO" && $PY encode_parquet.py --in-dir "$RAW" --out-dir "$LAT" --canonical "$CANON" \
      --shard $i --num-shards $NSHARD --device cuda:$g $FLAGS \
      > "$H/logs/encode_arenas13_sd1_shard$i.log" 2>&1 ) &
  i=$((i+1))
done
wait
NLAT=$(ls "$LAT"/ep_*_latents.npy 2>/dev/null | wc -l)
NMETA=$(ls "$LAT"/ep_*_meta.npz 2>/dev/null | wc -l)
echo "$(ts) encoded: $NLAT latents, $NMETA sidecars (expect $EXPECT each)"
[ "$NLAT" -eq "$EXPECT" ] && [ "$NMETA" -eq "$EXPECT" ] || { echo "encode incomplete"; exit 3; }
grep -h "decode-check\|PSNR" "$H"/logs/encode_arenas13_sd1_shard*.log | head -20 || true

# Cross-host check: Spiderman encodes ep_00016 on its own card into $D/tmp/enc_check; compare if present.
if ssh -o BatchMode=yes "$SPIDER" "test -f $D/tmp/enc_check/ep_00016_latents.npy"; then
  rsync -a "$SPIDER:$D/tmp/enc_check/ep_00016_latents.npy" "$H/logs/spiderman_ep_00016_latents.npy"
  $PY - <<PYEOF
import numpy as np
a=np.load("$LAT/ep_00016_latents.npy").astype(np.float32); b=np.load("$H/logs/spiderman_ep_00016_latents.npy").astype(np.float32)
print("cross-host check ep_00016: shapes", a.shape, b.shape, "max|diff|", float(np.abs(a-b).max()), "rms diff / rms", float(np.sqrt(((a-b)**2).mean())/np.sqrt((b**2).mean())))
PYEOF
else
  echo "cross-host check: Spiderman reference not present yet (compare later)"
fi

echo "$(ts) pushing latents to Spiderman $LAT_REMOTE"
ssh -o BatchMode=yes "$SPIDER" "mkdir -p $LAT_REMOTE"
rsync -a --info=progress2 "$LAT/" "$SPIDER:$LAT_REMOTE/"
echo "$(ts) ENCODE_ARENAS13_SD1_DONE latents=$NLAT"
