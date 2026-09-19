#!/bin/bash
# Dense 4-map training corpus (Sep 19 2026). A SEPARATE corpus from raw_arnold (the 17-map, 850-episode corpus):
# its own directory, its own corpus id, its own seeds. Purpose: test whether data density per map, not architecture,
# explains the gap to GameNGen-style results (tens of millions of frames on about five levels).
#   maps      3, 10, 12, 13  (chosen from the per-map seen evaluation: both the U-Net and PixArt rank maps identically,
#             so difficulty is a property of the map; these span it: map 10 easiest 23.8 dB / 0.198 LPIPS, maps 3 and 12
#             mid 22.9 to 23.0 dB, map 13 hard 20.0 dB / 0.331 LPIPS, and they differ visually: outdoor brick, library wood, green tech)
#   episodes  8000 (2000 per map, 40x the density of raw_arnold's 50 per map), 150 game-seconds each, every tic stored lossless
#   seeds     a stable hash of (corpus id, episode id); independent of worker count, so the corpus is bit-for-bit reproducible
#   usage     record_dense4.sh [workers=48] [episodes=8000]      resume-safe: finished episodes are skipped
D=/sata2/data/rnagabhi/doom; cd $D; export TMPDIR=$D/tmp/tmpdir OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${1:-48}; N=${2:-8000}; OUT=$D/raw_arnold_dense4; CID=arnold-train-dense4-v1; MAPS=3,10,12,13
ARN=(--frame_skip 4 --action_combinations "move_fb+move_lr;turn_lr;attack" --network_type dqn_rnn --recurrence lstm --n_rec_layers 1 --hist_size 4 --remember 1 --labels_mapping "" --game_features "target,enemy" --bucket_size "[10, 1]" --dropout 0.5 --speed on --crouch off --scenario deathmatch --wad full_deathmatch --n_bots 8 --reload $D/Arnold/pretrained/vizdoom_2017_track2.pth --evaluate 1 --visualize 0 --gpu_id -1)
mkdir -p $OUT $D/logs/rec_dense4
echo "$(date -Iseconds) start workers=$W episodes=$N maps=$MAPS corpus=$CID git=$(cd $D/repo && git rev-parse --short HEAD)" >> $OUT/RECORDING_LOG.txt
for w in $(seq 0 $((W - 1))); do
  PYTHONHASHSEED=0 nice -n 10 ~/miniconda3/envs/doom/bin/python repo/record_arnold.py --arnold-dir $D/Arnold --out-dir $OUT --map-ids $MAPS --episodes $N --episode-time 150 --worker-id $w --num-workers $W --compress-level 6 --corpus-id $CID -- "${ARN[@]}" > $D/logs/rec_dense4/worker_$w.log 2>&1 &
done
wait
echo "$(date -Iseconds) done files=$(ls $OUT/*.parquet 2>/dev/null | wc -l) bytes=$(du -sb $OUT | cut -f1)" >> $OUT/RECORDING_LOG.txt
