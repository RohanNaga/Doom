#!/bin/bash
# Mirror the small result files (json, jsonl, csv, plus the audit's per_rollout.npz) of the paper's runs from
# Spiderman's results_spiderman into a local directory that make_figures.py / make_tables.py read.
# Weights, latents, rollouts_seen.npz and clips are never pulled.
#
#   PERSEVE_SERVER_PASSWORD=... bash paper/sync_results.sh [local_dir]     (default paper/results_mirror; needs CMU VPN)
#
# Read-only against the server: rsync only copies remote -> local.
set -euo pipefail
HOST=rnagabhi@128.2.204.110
REMOTE=/sata2/data/rnagabhi/doom/results_spiderman
LOCAL=${1:-$(cd "$(dirname "$0")" && pwd)/results_mirror}
: "${PERSEVE_SERVER_PASSWORD:?set PERSEVE_SERVER_PASSWORD}"

RUNS=(030-dit-l32-aligned 032-dit-l32-aligned-seed1 031-unet-l32-aligned 033-pixart-l32-aligned
      050-skyreels-l8-flow 060-dit-cm2500 061-unet-cm2500 062-pixart-cm2500
      idm_aligned idm_aligned_k2)

FILTER=(--include='*/' --include='*.json' --include='*.jsonl' --include='*.csv' --include='audit/per_rollout.npz' --exclude='*')
RSYNC=(sshpass -p "$PERSEVE_SERVER_PASSWORD" rsync -az --prune-empty-dirs -e "ssh -o StrictHostKeyChecking=accept-new")

mkdir -p "$LOCAL"
for run in "${RUNS[@]}"; do
  echo "== $run"
  "${RSYNC[@]}" "${FILTER[@]}" "$HOST:$REMOTE/$run/" "$LOCAL/$run/" || echo "   (missing on server or transfer failed: $run)"
done

# the grid-label ablation dirs are matched by pattern on the server side
echo "== ablation_gridlabels_*"
"${RSYNC[@]}" "${FILTER[@]}" "$HOST:$REMOTE/ablation_gridlabels_*" "$LOCAL/" || echo "   (no ablation_gridlabels_* dirs)"

# the audit summary lives outside results_spiderman
echo "== tmp/audit/summary.json"
mkdir -p "$LOCAL/tmp/audit"
"${RSYNC[@]}" "$HOST:/sata2/data/rnagabhi/doom/tmp/audit/summary.json" "$LOCAL/tmp/audit/summary.json" || echo "   (no audit summary)"

echo "mirror at $LOCAL:"; du -sh "$LOCAL"
