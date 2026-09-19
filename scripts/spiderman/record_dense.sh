#!/bin/bash
# Dense training corpus (Sep 19 2026). A SEPARATE corpus from raw_arnold (the 17-map, 850-episode corpus):
# its own directories, its own corpus ids, its own seeds. Purpose: test whether data density per map, not
# architecture, explains the gap to GameNGen-style results (tens of millions of frames on about five levels).
#
#   TARGET=arenas        maps 2,3,4,5 of full_deathmatch             -> raw_arnold_dense/arenas
#   TARGET=dmsimple      deathmatch_simple MAP01, map_id 101         -> raw_arnold_dense/dmsimple
#   TARGET=arenas678     maps 6,7,8 of full_deathmatch, 1,000 per map -> raw_arnold_dense/arenas_678
#                        (recorded on Superman and streamed here; see raw_arnold_dense/arenas_678/README.txt)
#   TARGET=test-dmsimple eval   deathmatch_simple MAP01, map_id 101  -> raw_arnold_dense_eval/dmsimple
#
# Maps. Arenas 2,3,4,5 and 6,7,8 are the agent paper's own published train/test split (Arnold README:
# --map_ids_train "2,3,4,5" --map_ids_test "6,7,8"), so the split is the agent authors' and not ours.
# deathmatch_simple is the only named map in prior Doom world-model work; both open GameNGen
# reproductions use it, which is what makes a number on it comparable to anything outside this repo.
# Recording decides nothing about training: which segments train and which test is chosen later
# (Rohan, Sep 19 2026). Each segment has its own folder and corpus id so any split stays possible.
#
# This replaces an earlier plan (maps 3,10,12,13 of full_deathmatch, corpus arnold-train-dense4-v1).
# Those four were picked by reading our own per-map evaluation scores, which selects the training set on
# the outcome being measured; Rohan rejected it on Sep 19 2026. 134 episodes of that aborted corpus remain
# in $D/raw_arnold_dense4 and are unused. Nothing downstream should read that directory.
#
# Knobs: WORKERS=32  EPISODES_PER_MAP (2000 arenas, 1000 arenas678 and dmsimple, 20 test-dmsimple)  MAPS  MODE=pertic  PNG_LEVEL=6
#        N_BOTS=8  EPISODE_TIME=150.  Positional [workers] [total episodes] still override.
# Resume-safe: an episode whose parquet already exists is skipped, so a relaunch continues a corpus.
#
#   TARGET=arenas bash scripts/spiderman/record_dense.sh
D=/sata2/data/rnagabhi/doom; cd $D; export TMPDIR=$D/tmp/tmpdir OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TARGET=${TARGET:-arenas}; MODE=${MODE:-pertic}; W=${1:-${WORKERS:-32}}
EPISODE_TIME=${EPISODE_TIME:-150}; N_BOTS=${N_BOTS:-8}; LEVEL=${PNG_LEVEL:-6}
WAD=full_deathmatch; OFFSET=0; CMD=(); EPM=${EPISODES_PER_MAP:-2000}
case $TARGET in
  arenas)        MAPS=${MAPS:-2,3,4,5}; CID=arnold-train-dense-v1;          OUT=$D/raw_arnold_dense/arenas ;;
  dmsimple)      MAPS=${MAPS:-1};       CID=arnold-dense-dmsimple-v1;       OUT=$D/raw_arnold_dense/dmsimple;      EPM=${EPISODES_PER_MAP:-1000} ;;
  arenas678)     MAPS=${MAPS:-6,7,8};   CID=arnold-dense-arenas678-v1;      OUT=$D/raw_arnold_dense/arenas_678;    EPM=${EPISODES_PER_MAP:-1000} ;;
  test-dmsimple) MAPS=${MAPS:-1};       CID=arnold-eval-dense-dmsimple-v1;  OUT=$D/raw_arnold_dense_eval/dmsimple; EPM=${EPISODES_PER_MAP:-20} ;;
  *) echo "unknown TARGET=$TARGET (arenas|arenas678|dmsimple|test-dmsimple)" >&2; exit 2 ;;
esac
if [ "${TARGET#test-}" = dmsimple ] || [ "$TARGET" = dmsimple ]; then
  # deathmatch_simple's MAP01 is map 1 to the engine, which is full_deathmatch's arena 1; offset the stored
  # label so the two directories can be merged at encode time.
  WAD=deathmatch_simple; OFFSET=100; CMD=(--zdoom-bots)
  # BLOCKED, Sep 19 2026: under Arnold this map records with no opponent at all. Measured over a full
  # 150 game-second episode: 0 kills, 0 deaths, frags 0, health flat at 100, ammo flat at 50, with Arnold's
  # scripted marines AND with --zdoom-bots. `pukename change_difficulty 5` changes nothing (no monster ever
  # appears, with or without -deathmatch), so the ACS curriculum is not what populates it. The map itself is
  # fine: driven straight from ViZDoom with the reproduction's own cfg and DOOM_ENV_WITH_BOTS_ARGS plus
  # addbot, 4 to 5 opponents are visible at once and DeadDoomPlayer labels appear throughout. So the gap is
  # between Arnold's game setup and the reproduction's, and it is not yet found. Recording this target now
  # would produce thousands of empty episodes, so it refuses until someone clears it.
  if [ "${DMSIMPLE_OK:-0}" != 1 ]; then
    echo "TARGET=$TARGET refuses to run: Arnold records deathmatch_simple with no opponents (see the comment" >&2
    echo "above and release/DENSE_CORPUS.md). Set DMSIMPLE_OK=1 once opponents are verified to appear." >&2
    exit 3
  fi
fi
N=${2:-$((EPM * $(echo $MAPS | tr ',' '\n' | wc -l)))}
FAST=(); [ "$MODE" = decision ] && FAST=(--decision-only)
ARN=(--frame_skip 4 --action_combinations "move_fb+move_lr;turn_lr;attack" --network_type dqn_rnn --recurrence lstm --n_rec_layers 1 --hist_size 4 --remember 1 --labels_mapping "" --game_features "target,enemy" --bucket_size "[10, 1]" --dropout 0.5 --speed on --crouch off --scenario deathmatch --wad $WAD --n_bots $N_BOTS --reload $D/Arnold/pretrained/vizdoom_2017_track2.pth --evaluate 1 --visualize 0 --gpu_id -1)
mkdir -p $OUT $D/logs/rec_dense
echo "$(date -Iseconds) start target=$TARGET mode=$MODE workers=$W episodes=$N maps=$MAPS wad=$WAD offset=$OFFSET png=$LEVEL corpus=$CID git=$(cd $D/repo && git rev-parse --short HEAD)" >> $OUT/RECORDING_LOG.txt
for w in $(seq 0 $((W - 1))); do
  PYTHONHASHSEED=0 nice -n 10 ~/miniconda3/envs/doom/bin/python repo/record_arnold.py --arnold-dir $D/Arnold --out-dir $OUT --map-ids $MAPS --episodes $N --episode-time $EPISODE_TIME --worker-id $w --num-workers $W --compress-level $LEVEL --map-id-offset $OFFSET "${CMD[@]}" "${FAST[@]}" --corpus-id $CID -- "${ARN[@]}" > $D/logs/rec_dense/${TARGET}_worker_$w.log 2>&1 &
done
wait
echo "$(date -Iseconds) done target=$TARGET files=$(ls $OUT/*.parquet 2>/dev/null | wc -l) bytes=$(du -sb $OUT | cut -f1)" >> $OUT/RECORDING_LOG.txt
