#!/bin/bash
# Zero to a usable node, step 2 of 5: the public dense Arnold corpus, subset first.
#
#   usage: [DOOM_ROOT=/data/doom] [FULL=1] [JOBS=8] [CHUNK=400] [LIST=1] [HF_REPO=..] \
#          [DRY=1] fetch_dataset.sh
#
# `RohanNaga/doom-dense-arnold` holds 8,000 episodes of arenas 2 to 5 under `arenas/` and 3,000 of
# arenas 6 to 8 under `arenas_678/`, about 480 GB in all. The two next-tic runs need 2,260 of them:
#
#   train   arenas      0:2000     2,000 episodes, 500 per arena
#   val     arenas      6000:6100    100 episodes,  25 per arena
#   test    arenas      7000:7100    100 episodes,  25 per arena
#   unseen  arenas_678  0:60          60 episodes,  20 per arena
#
# The ranges are `release/dense_split.json`, fixed before anything was scored, and they are exactly
# map-balanced because `record_arnold.py:322` assigns `map_ids[episode_id % len(map_ids)]`. The
# download list is therefore built as EXACT file names, not as prefix globs: `arenas_678/ep_000??`
# would have pulled 100 episodes for the 60 the unseen corpus is, and a 40-episode surplus in an
# evaluation corpus is a different corpus.
#
# Resumable. `hf download` into a `--local-dir` skips files it already has, so re-running after an
# interruption continues; nothing here forces a re-download. The md5 pass then re-verifies
# everything present, which is what makes a resumed transfer trustworthy.
#
# Every downloaded episode is checked against the dataset's own `md5_arenas.txt` /
# `md5_arenas_678.txt`, which are `md5sum`-format manifests of the whole corpus
# (`verify_dense.py:145`). We filter each manifest down to the files actually present and verify
# those in parallel. A mismatch lists the files and stops the script: a silently corrupt episode
# becomes a silently wrong latent, and nothing downstream would catch it.
#
# FULL=1 adds the rest of both segments after the subset is verified, for the multi-pass training
# budget the workshop runs do not reach. Budget about 480 GB and hours of transfer.
#
# DRY=1 prints what would run and touches nothing; LIST=1 additionally prints every file in the
# download list, which is the auditable form of the ranges above.
set -u
D=${DOOM_ROOT:-/data/doom}
DRY=${DRY:-0}
LIST=${LIST:-0}
FULL=${FULL:-0}
JOBS=${JOBS:-8}
CHUNK=${CHUNK:-400}          # files per `hf download` invocation; keeps one argv list sane
HF_REPO=${HF_REPO:-RohanNaga/doom-dense-arnold}
HF=${HF:-$D/env/bin/hf}
MD5SUM=${MD5SUM:-md5sum}
OUT=$D/raw_arnold_dense
TRAIN_IDS=${TRAIN_IDS:-0:2000}
VAL_IDS=${VAL_IDS:-6000:6100}
TEST_IDS=${TEST_IDS:-7000:7100}
UNSEEN_IDS=${UNSEEN_IDS:-0:60}
META="dense_split.json canonical_controls.json md5_arenas.txt md5_arenas_678.txt README.md arenas/README.md arenas_678/README.md"

die() { echo "FETCH_DATASET_FAILED $*" >&2; exit 1; }

# --- the exact download list -----------------------------------------------------------
ARENAS=()
A678=()
add() {   # add <subdir> <lo:hi> <array name>
  local dir=$1 lo=${2%%:*} hi=${2##*:} i
  for ((i = lo; i < hi; i++)); do
    printf -v F '%s/ep_%05d.parquet' "$dir" "$i"
    if [ "$3" = ARENAS ]; then ARENAS+=("$F"); else A678+=("$F"); fi
  done
}
plan() {  # plan <label> <subdir> <lo:hi> <first> <last> <count>
  echo "DRY plan $1 $2 $3 files=$6 first=$4 last=$5"
}
add arenas "$TRAIN_IDS" ARENAS
add arenas "$VAL_IDS" ARENAS
add arenas "$TEST_IDS" ARENAS
add arenas_678 "$UNSEEN_IDS" A678
FILES=("${ARENAS[@]}" "${A678[@]}")

count() { local lo=${1%%:*} hi=${1##*:}; echo $(( hi - lo )); }
name()  { printf '%s/ep_%05d.parquet' "$1" "$2"; }

if [ "$DRY" = 1 ]; then
  echo "DRY fetch repo=$HF_REPO root=$D out=$OUT files=${#FILES[@]}"
  for SPEC in "train arenas $TRAIN_IDS" "val arenas $VAL_IDS" "test arenas $TEST_IDS" \
              "unseen arenas_678 $UNSEEN_IDS"; do
    set -- $SPEC
    plan "$1" "$2" "$3" "$(name "$2" "${3%%:*}")" "$(name "$2" "$(( ${3##*:} - 1 ))")" "$(count "$3")"
  done
  if [ "$LIST" = 1 ]; then for F in "${FILES[@]}"; do echo "DRY file $F"; done; fi
  echo "DRY meta $HF download $HF_REPO --repo-type dataset --local-dir $OUT --include $META"
  echo "DRY download $HF download $HF_REPO --repo-type dataset --local-dir $OUT --include <${#FILES[@]} exact names, ${CHUNK} per call, first ${FILES[0]}>"
  echo "DRY verify arenas ${#ARENAS[@]} file(s) against $OUT/md5_arenas.txt (jobs=$JOBS)"
  echo "DRY verify arenas_678 ${#A678[@]} file(s) against $OUT/md5_arenas_678.txt (jobs=$JOBS)"
  if [ "$FULL" = 1 ]; then
    echo "DRY full $HF download $HF_REPO --repo-type dataset --local-dir $OUT --include arenas/*.parquet --include arenas_678/*.parquet"
  else
    echo "DRY full skipped (FULL=1 also fetches the remaining ~8,740 episodes, about 480 GB)"
  fi
  echo "DRY report bytes/elapsed/mb_per_s of $OUT"
  exit 0
fi

command -v "$HF" >/dev/null 2>&1 || [ -x "$HF" ] || die "no hf client at $HF (run setup_node.sh first)"
command -v "$MD5SUM" >/dev/null 2>&1 || die "no $MD5SUM on this node"
mkdir -p "$OUT" || die "cannot write under $OUT"

bytes() { du -sb "$1" 2>/dev/null | cut -f1; }
T0=$(date +%s)
B0=$(bytes "$OUT"); B0=${B0:-0}

# --- metadata first: the encode cannot start without the canonical table -----------------
# shellcheck disable=SC2086
"$HF" download "$HF_REPO" --repo-type dataset --local-dir "$OUT" --include $META \
  || die "metadata download failed"
for F in dense_split.json canonical_controls.json md5_arenas.txt md5_arenas_678.txt; do
  [ -s "$OUT/$F" ] || die "$F is missing from the download; the dataset layout changed"
done

# --- episodes, in chunks -----------------------------------------------------------------
N=${#FILES[@]}
for ((i = 0; i < N; i += CHUNK)); do
  echo "$(date -Iseconds) downloading files $i..$(( i + CHUNK > N ? N : i + CHUNK ))/$N"
  "$HF" download "$HF_REPO" --repo-type dataset --local-dir "$OUT" \
    --include "${FILES[@]:i:CHUNK}" || die "download of chunk at $i failed (re-run to resume)"
done

if [ "$FULL" = 1 ]; then
  echo "$(date -Iseconds) FULL=1: the remaining episodes of both segments"
  "$HF" download "$HF_REPO" --repo-type dataset --local-dir "$OUT" \
    --include "arenas/*.parquet" --include "arenas_678/*.parquet" || die "full download failed"
fi

# --- md5, in parallel, against the dataset's own manifests --------------------------------
verify() {   # verify <segment> <manifest>
  local seg=$1 man=$2 dir=$OUT/$1 tmp rc want have
  [ -s "$man" ] || { echo "no manifest at $man" >&2; return 1; }
  [ -d "$dir" ] || { echo "no directory at $dir" >&2; return 1; }
  tmp=$(mktemp -d) || return 1
  ls -1 "$dir" | grep -E '^ep_[0-9]+\.parquet$' | sort > "$tmp/present"
  want=$(wc -l < "$tmp/present" | tr -d ' ')
  # the manifest covers the whole segment; verify the files we actually hold
  awk 'NR==FNR { p[$0] = 1; next } ($2 in p) { print }' "$tmp/present" "$man" > "$tmp/filtered"
  have=$(wc -l < "$tmp/filtered" | tr -d ' ')
  if [ "$have" != "$want" ]; then
    echo "$seg: $(( want - have )) downloaded file(s) are absent from $(basename "$man"):" >&2
    awk '{ print $2 }' "$man" | sort > "$tmp/known"
    comm -23 "$tmp/present" "$tmp/known" | head -20 >&2
    rm -rf "$tmp"; return 1
  fi
  split -l 100 "$tmp/filtered" "$tmp/chunk."
  # one md5sum -c per chunk, $JOBS at a time; --quiet prints only the failures
  find "$tmp" -name 'chunk.*' -print0 |
    xargs -0 -P "$JOBS" -I{} bash -c 'cd "$1" && '"$MD5SUM"' -c --quiet "$2"' _ "$dir" {} \
    > "$tmp/out" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || grep -qE 'FAILED|No such file' "$tmp/out"; then
    echo "MD5 MISMATCHES in $seg ($want file(s) checked):" >&2
    grep -E 'FAILED|No such file' "$tmp/out" >&2 || cat "$tmp/out" >&2
    rm -rf "$tmp"; return 1
  fi
  echo "md5 ok: $want file(s) in $seg"
  rm -rf "$tmp"
}

verify arenas "$OUT/md5_arenas.txt" || die "arenas md5 verification failed"
verify arenas_678 "$OUT/md5_arenas_678.txt" || die "arenas_678 md5 verification failed"

# --- what it cost -------------------------------------------------------------------------
B1=$(bytes "$OUT"); B1=${B1:-0}
ELAPSED=$(( $(date +%s) - T0 ))
awk -v b0="$B0" -v b1="$B1" -v s="$ELAPSED" 'BEGIN {
  d = b1 - b0;
  printf "FETCH_DATASET_DONE total_bytes=%d downloaded_bytes=%d (%.1f GiB) elapsed=%ds mb_per_s=%.1f\n",
         b1, d, d / 1073741824, s, (s > 0 ? d / 1048576 / s : 0);
}'
echo "episodes: arenas $(ls -1 "$OUT/arenas" 2>/dev/null | grep -c '^ep_.*\.parquet$'), arenas_678 $(ls -1 "$OUT/arenas_678" 2>/dev/null | grep -c '^ep_.*\.parquet$')"
