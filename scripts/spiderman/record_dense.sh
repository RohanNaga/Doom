#!/bin/bash
# Dense corpus, EVERY tic stored lossless. Separate from raw_arnold (the original 17-map corpus).
# Started Sep 19 2026. Full documentation: release/DENSE_CORPUS.md.
#
#   usage: SEGMENT=arenas record_dense.sh [workers=32] [episodes=8000]
#
#   SEGMENT=arenas      maps 2,3,4,5 of full_deathmatch   arnold-train-dense-v1      raw_arnold_dense/arenas
#   SEGMENT=arenas_678  maps 6,7,8 of full_deathmatch     arnold-dense-arenas678-v1  raw_arnold_dense/arenas_678
#   SEGMENT=dm_simple   deathmatch_simple MAP01 as 101    arnold-dense-dmsimple-v1   raw_arnold_dense/dm_simple
#
# Maps 2 to 5 are the Arnold paper's training arenas and 6 to 8 its test arenas (Arnold README:
# --map_ids_train "2,3,4,5" --map_ids_test "6,7,8"). They were fixed before looking at any model score.
# WHAT TO TRAIN OR TEST ON IS DECIDED LATER; this script only records. deathmatch_simple is the only named
# map in prior Doom world-model work, which is what would make a number on it externally comparable.
#
# Seeds are a stable hash of (corpus id, episode id), independent of worker count, and every segment has its
# own corpus id, so no two segments and no evaluation episode ever share a seed. Resume-safe: an episode whose
# parquet already exists is skipped, so a relaunch continues a corpus and may use a newer build of the recorder.
#
# This replaces record_dense4.sh, which recorded maps 3,10,12,13. Those four were picked by reading our own
# per-map evaluation scores, which selects the training set on the outcome being measured; Rohan rejected it
# on Sep 19 2026. 134 episodes of that aborted corpus remain in $D/raw_arnold_dense4 and are unused.
# DRY=1 prints one worker's command and stops before every side effect; DOOM_ROOT repoints the data
# root. Both default to the real thing, so a test can check the wiring without starting 32 recorders.
D=${DOOM_ROOT:-/sata2/data/rnagabhi/doom}; DRY=${DRY:-0}
[ "$DRY" = 1 ] || cd $D
export TMPDIR=$D/tmp/tmpdir OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${1:-32}; N=${2:-8000}; SEGMENT=${SEGMENT:-arenas}; WAD=full_deathmatch; EXTRA=()
case $SEGMENT in
  arenas)     MAPS=2,3,4,5; CID=arnold-train-dense-v1;     OUT=$D/raw_arnold_dense/arenas ;;
  arenas_678) MAPS=6,7,8;   CID=arnold-dense-arenas678-v1; OUT=$D/raw_arnold_dense/arenas_678 ;;
  dm_simple)
    MAPS=1; CID=arnold-dense-dmsimple-v1; OUT=$D/raw_arnold_dense/dm_simple; WAD=deathmatch_simple
    # MAP01 of deathmatch_simple is map 1 to the engine, which is full_deathmatch's arena 1, so the stored
    # label is offset to 101 and the episode metadata carries the WAD name and the engine's own map id.
    EXTRA=(--map-id-offset 100 --init-game-command "pukename change_difficulty 5")
    # The engine blocker is fixed (Sep 20 2026): --zdoom-bots never reached Arnold, because the recorder
    # injected it with a functools.partial whose keywords Arnold's own call site overrode. A 60-game-second
    # episode now records 9 deaths and 2 frags where the control records none. See release/DENSE_CORPUS.md.
    # The gate stays: whether to spend the disk and the days on this segment is a corpus decision, and one
    # short episode is not a corpus-scale check.
    EXTRA+=(--zdoom-bots)
    if [ "${DM_SIMPLE_OK:-0}" != 1 ]; then
      echo "SEGMENT=dm_simple is gated, not broken. Arnold now fights on this map (--zdoom-bots works as of" >&2
      echo "Sep 20 2026). Record a few full 150-second episodes, confirm deaths and frags are non-zero in all" >&2
      echo "of them, then set DM_SIMPLE_OK=1. See release/DENSE_CORPUS.md." >&2
      exit 3
    fi ;;
  *) echo "unknown SEGMENT=$SEGMENT (arenas|arenas_678|dm_simple)" >&2; exit 2 ;;
esac
ARN=(--frame_skip 4 --action_combinations "move_fb+move_lr;turn_lr;attack" --network_type dqn_rnn --recurrence lstm --n_rec_layers 1 --hist_size 4 --remember 1 --labels_mapping "" --game_features "target,enemy" --bucket_size "[10, 1]" --dropout 0.5 --speed on --crouch off --scenario deathmatch --wad $WAD --n_bots 8 --reload $D/Arnold/pretrained/vizdoom_2017_track2.pth --evaluate 1 --visualize 0 --gpu_id -1)
# One array, used both to print under DRY and to run for real, so the two cannot drift. `env` carries
# PYTHONHASHSEED because a VAR=value prefix cannot live inside an array, and the array preserves the
# quoting of the Arnold arguments, one of which contains a semicolon.
worker_args() {   # worker_args <worker-id> -> WCMD
  WCMD=(env PYTHONHASHSEED=0 nice -n 10 ~/miniconda3/envs/doom/bin/python repo/record_arnold.py
        --arnold-dir $D/Arnold --out-dir $OUT --map-ids $MAPS --episodes $N --episode-time 150
        --worker-id "$1" --num-workers $W --compress-level 6 "${EXTRA[@]}" --corpus-id $CID --
        "${ARN[@]}")
}

if [ "$DRY" = 1 ]; then
  worker_args 0
  echo "DRY segment=$SEGMENT workers=$W episodes=$N maps=$MAPS corpus=$CID wad=$WAD out=$OUT"
  echo "DRY worker0 ${WCMD[*]}"
  exit 0
fi

mkdir -p $OUT $D/logs/rec_dense
echo "$(date -Iseconds) start segment=$SEGMENT workers=$W episodes=$N maps=$MAPS corpus=$CID mode=per-tic png=6 git=$(cd $D/repo && git rev-parse --short HEAD)" >> $OUT/RECORDING_LOG.txt
for w in $(seq 0 $((W - 1))); do
  worker_args "$w"
  "${WCMD[@]}" > $D/logs/rec_dense/${SEGMENT}_worker_$w.log 2>&1 &
done
wait
echo "$(date -Iseconds) done segment=$SEGMENT files=$(ls $OUT/*.parquet 2>/dev/null | wc -l) bytes=$(du -sb $OUT | cut -f1)" >> $OUT/RECORDING_LOG.txt
