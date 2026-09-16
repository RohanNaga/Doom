#!/bin/bash
# waits for PixArt's evaluation to finish, then runs the compute-matched short runs on the same gpu
D=/sata2/data/rnagabhi/doom; GPU=${1:?gpu}
until grep -q AFTER_RUN_DONE $D/logs/033-pixart-l32-aligned_rollout.log 2>/dev/null; do sleep 300; done
bash $D/compute_matched.sh $GPU
