# Sourced, not run: the next-tic id ranges, read from `release/dense_split.json` so no launcher
# carries its own copy of a range.
#
#   . "$(dirname "${BASH_SOURCE[0]}")/../dense_ids.sh"; UNSEEN_IDS=${UNSEEN_IDS:-$(dense_ids unseen_ids)}
#
# The unseen subset was 0:60 until 2026-09-22 and was replaced by 60:120 before any model was
# scored (the json's `history` says why). Four scripts hardcoded 0:60; one of them would have kept
# encoding the old subset after the json changed. Reading the json is what makes the file the only
# place a range is written down.
#
# Plain sed rather than python, because the DRY paths of the cluster scripts must run on a node
# that has no interpreter yet. `test_unseen_subset.py` checks that this parse returns exactly what
# `json.load` returns, so a reformatted json cannot silently change the answer.
#
# The json is the one in THIS checkout (beside this file), not under `$REPO`: a launcher's REPO can
# point at a different encoder checkout, and the ranges must come from the scripts' own commit.
# DENSE_SPLIT overrides the path.
dense_ids() {   # dense_ids <next_tic_runs key>: prints A:B, or fails with a message on stderr
  local file=${DENSE_SPLIT:-$(dirname "${BASH_SOURCE[0]}")/../release/dense_split.json} v
  v=$(sed -n "s/^ *\"$1\": *\"\\([0-9][0-9]*:[0-9][0-9]*\\)\",\\{0,1\\} *\$/\\1/p" "$file" 2>/dev/null | head -1)
  if [ -z "$v" ]; then
    echo "no next_tic_runs.$1 range in $file" >&2
    return 1
  fi
  echo "$v"
}
