#!/bin/bash
# Dense 4-map training corpus (Sep 19 2026). A SEPARATE corpus from raw_arnold (the 17-map, 850-episode corpus):
# its own directory, its own corpus id, its own seeds. Purpose: test whether data density per map, not architecture,
# explains the gap to GameNGen-style results (tens of millions of frames on about five levels).
#   maps      $MAPS, default 3, 10, 12, 13  (chosen from the per-map seen evaluation: both the U-Net and PixArt rank
#             maps identically, so difficulty is a property of the map; these span it: map 10 easiest 23.8 dB / 0.198
#             LPIPS, maps 3 and 12 mid 22.9 to 23.0 dB, map 13 hard 20.0 dB / 0.331 LPIPS, and they differ visually:
#             outdoor brick, library wood, green tech)
#   episodes  8000 (2000 per map, 40x the density of raw_arnold's 50 per map), 150 game-seconds each
#   seeds     a stable hash of (corpus id, episode id); independent of worker count and of MODE, so the corpus is
#             reproducible from the corpus id alone
#   MODE      pertic   every tic stored lossless, what release/DENSE4_CORPUS.md documents (default)
#             decision one row per agent decision, the only rows training ever reads; several times faster and a
#                      quarter of the bytes. Not tic-for-tic the same rollout: the engine runs each 4-tic skip in one
#                      call, so a death inside a skip is seen up to 3 tics later than per tic and the episode diverges
#                      from there. Same agent, same seeds, same distribution; a different draw from it.
#   usage     MODE=decision WORKERS=32 MAPS=3,10,12,13 record_dense4.sh [workers] [episodes]   resume-safe
D=/sata2/data/rnagabhi/doom; cd $D; export TMPDIR=$D/tmp/tmpdir OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${1:-${WORKERS:-32}}; N=${2:-8000}; MAPS=${MAPS:-3,10,12,13}; MODE=${MODE:-pertic}
# decision mode writes its own directory and its own corpus id: one directory must hold one row semantics,
# because `encode_parquet.py` takes `stored_tic_stride` from the first file it reads.
CID=arnold-train-dense4-v1; OUT=$D/raw_arnold_dense4; FAST=()
if [ "$MODE" = decision ]; then OUT=$D/raw_arnold_dense4d; CID=arnold-train-dense4d-v1; FAST=(--decision-only); fi
# PNG level, lossless at every setting: 6 is what the per-tic corpus was recorded at, and 1 is the
# measured choice for decision mode (1.5x faster to record, 18% more bytes; see release/DENSE4_CORPUS.md)
LEVEL=${PNG_LEVEL:-$([ "$MODE" = decision ] && echo 1 || echo 6)}
ARN=(--frame_skip 4 --action_combinations "move_fb+move_lr;turn_lr;attack" --network_type dqn_rnn --recurrence lstm --n_rec_layers 1 --hist_size 4 --remember 1 --labels_mapping "" --game_features "target,enemy" --bucket_size "[10, 1]" --dropout 0.5 --speed on --crouch off --scenario deathmatch --wad full_deathmatch --n_bots 8 --reload $D/Arnold/pretrained/vizdoom_2017_track2.pth --evaluate 1 --visualize 0 --gpu_id -1)
mkdir -p $OUT $D/logs/rec_dense4
echo "$(date -Iseconds) start mode=$MODE workers=$W episodes=$N maps=$MAPS corpus=$CID png=$LEVEL git=$(cd $D/repo && git rev-parse --short HEAD)" >> $OUT/RECORDING_LOG.txt
for w in $(seq 0 $((W - 1))); do
  PYTHONHASHSEED=0 nice -n 10 ~/miniconda3/envs/doom/bin/python repo/record_arnold.py --arnold-dir $D/Arnold --out-dir $OUT --map-ids $MAPS --episodes $N --episode-time 150 --worker-id $w --num-workers $W --compress-level $LEVEL "${FAST[@]}" --corpus-id $CID -- "${ARN[@]}" > $D/logs/rec_dense4/worker_$w.log 2>&1 &
done
wait
echo "$(date -Iseconds) done mode=$MODE files=$(ls $OUT/*.parquet 2>/dev/null | wc -l) bytes=$(du -sb $OUT | cut -f1)" >> $OUT/RECORDING_LOG.txt
