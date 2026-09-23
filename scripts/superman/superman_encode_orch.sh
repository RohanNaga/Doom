#!/bin/bash
# The Superman encode schedule of Sep 22-24 2026, run in tmux `enc-orch`:
#
#   phase 1  SD 3.5 (16 ch): val, test, arenas_678 60:120, then train 0:2000; then the eval split files
#   phase 2  sd-vae-ft-mse (4 ch): train 2000:6000
#   phase 3  SD 3.5 (16 ch): train 2000:6000
#
# Cards (Rohan, Sep 22): GPUs 1-7 until 08:45 EDT Sep 23, with GPUs 1 and 2 drained from 08:38 and
# killed at 08:45; GPUs 2-7 from 09:00; GPU 0 is never used. Each phase writes one queue per corpus
# holding the ids Spiderman lacks, keeps one worker per allowed card until the queues are empty, then
# re-lists and repeats, because a worker killed mid-batch loses the ids it had popped. A phase ends
# with a row-count audit (latent rows == sidecar rows == raw parquet rows) run on Spiderman's CPU.
# Spiderman's GPUs are never used. Its /sata2/data must keep FLOOR GB free: a phase does not start
# below it, the workers stop below it, and the orchestrator then stops and says so.
#
#   usage: [START_PHASE=1] [FLOOR=500] superman_encode_orch.sh
set -u
B=/home/rohan/Doom/sd35enc
WR=$B/superman_stream_encode.sh
QD=$B/queues
LOG=/home/rohan/Doom/logs/enc-orch.log
R=rnagabhi@128.2.204.110
SD=/sata2/data/rnagabhi/doom
FLOOR=${FLOOR:-500}
START_PHASE=${START_PHASE:-1}
SSH="ssh -i $HOME/.ssh/id_ed25519_doom -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=60 \
 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm/sm-orch -o ControlPersist=12h"
T_DRAIN=$(date -d "2026-09-23 08:38" +%s)
T_KILL=$(date -d "2026-09-23 08:45" +%s)
T_SIX=$(date -d "2026-09-23 09:00" +%s)

log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }
remote() { $SSH $R "$@"; }

allowed() {     # the cards our workers may hold right now
  local now; now=$(date +%s)
  if [ "$now" -lt "$T_DRAIN" ]; then echo "7 6 5 4 3 2 1"
  elif [ "$now" -lt "$T_SIX" ]; then echo "7 6 5 4 3"
  else echo "7 6 5 4 3 2"; fi
}

running() { tmux has-session -t "enc-$VAE-$1" 2>/dev/null; }

enforce() {     # drain, then kill, a worker on a card that is no longer allowed
  local g now; now=$(date +%s)
  for g in 1 2 3 4 5 6 7; do
    running "$g" || continue
    grep -qw "$g" <<< "$(allowed)" && continue
    if [ "$now" -ge "$T_KILL" ]; then
      tmux kill-session -t "enc-$VAE-$g"; log "killed enc-$VAE-$g: card $g not allowed now"
    elif [ ! -e "/home/rohan/Doom/stream/$g/STOP" ]; then
      touch "/home/rohan/Doom/stream/$g/STOP"; log "draining enc-$VAE-$g: card $g not allowed from $(date -d @"$T_KILL" +%H:%M)"
    fi
  done
}

spiderman_free() { remote "df --output=avail -BG /sata2/data | tail -1" | tr -dc 0-9; }

audit() {       # audit <latents dir> <raw dir> <channels>
  remote "cd $SD/tmp && \$HOME/wanenc/bin/python rowcount_check.py $1 $2 $3" 2>&1 | head -25 | tee -a "$LOG"
}

phase() {       # phase <vae> <train ids> <corpora...>
  VAE=$1; local tids=$2; shift 2
  local corpora="$*" csv round c g
  csv=$(tr ' ' , <<< "$corpora")
  mkdir -p "$QD"
  declare -A launched fails
  for round in 1 2 3; do
    local total=0 counts=""
    for c in $corpora; do
      VAE=$VAE TRAIN_IDS=$tids LIST=1 bash "$WR" 9 0/1 "$c" > "$QD/${VAE}_$c.txt.new" \
        || { log "FATAL listing $VAE $c failed"; exit 5; }
      mv "$QD/${VAE}_$c.txt.new" "$QD/${VAE}_$c.txt"
      total=$((total + $(wc -l < "$QD/${VAE}_$c.txt")))
      counts="$counts $c=$(wc -l < "$QD/${VAE}_$c.txt")"
    done
    log "PHASE $VAE train=$tids round $round: $total missing ($counts )"
    [ "$total" -eq 0 ] && break
    local free; free=$(spiderman_free)
    if [ -z "$free" ] || [ "$free" -lt "$FLOOR" ]; then
      log "FATAL Spiderman /sata2/data has ${free:-?}G free < ${FLOOR}G; not starting $VAE round $round"; exit 9
    fi
    while :; do
      enforce
      local left=0 any=0
      for c in $corpora; do left=$((left + $(wc -l < "$QD/${VAE}_$c.txt"))); done
      for g in 1 2 3 4 5 6 7; do running "$g" && any=1; done
      if [ "$left" -gt 0 ]; then
        for g in $(allowed); do
          running "$g" && continue
          # a worker that keeps dying within 5 minutes of launch is broken, not unlucky
          if [ -n "${launched[$g]:-}" ] && [ $(( $(date +%s) - ${launched[$g]} )) -ge 300 ]; then
            fails[$g]=0
          elif [ -n "${launched[$g]:-}" ]; then
            fails[$g]=$(( ${fails[$g]:-0} + 1 ))
            if [ "${fails[$g]}" -ge 3 ]; then
              [ "${fails[$g]}" -eq 3 ] && log "card $g: worker died three times within 5 min; not relaunching (see enc-$VAE-$g.log)"
              continue
            fi
          fi
          tmux new-session -d -s "enc-$VAE-$g" \
            "VAE=$VAE TRAIN_IDS=$tids QUEUE_DIR=$QD REMOTE_MIN_FREE_GB=$FLOOR bash $WR $g 0/1 $csv"
          launched[$g]=$(date +%s)
          any=1
          log "launched enc-$VAE-$g ($left queued)"
        done
      fi
      [ "$left" -eq 0 ] && [ "$any" -eq 0 ] && break
      free=$(spiderman_free)
      if [ -n "$free" ] && [ "$free" -lt "$FLOOR" ]; then
        log "Spiderman /sata2/data at ${free}G < ${FLOOR}G: workers stop before their next batch"
        while :; do any=0; for g in 1 2 3 4 5 6 7; do running "$g" && any=1; done; [ $any = 0 ] && break; sleep 60; done
        log "FATAL phase $VAE train=$tids stopped at the Spiderman space floor"; exit 9
      fi
      sleep 60
    done
  done
  local miss=0
  for c in $corpora; do
    miss=$((miss + $(VAE=$VAE TRAIN_IDS=$tids LIST=1 bash "$WR" 9 0/1 "$c" | wc -l)))
  done
  if [ "$miss" -eq 0 ]; then log "PHASE_DONE $VAE train=$tids corpora=$csv"
  else log "PHASE_INCOMPLETE $VAE train=$tids: $miss episodes still missing after 3 rounds"; fi
}

log "START orchestrator phase>=$START_PHASE floor=${FLOOR}G"
if [ "$START_PHASE" -le 1 ]; then
  phase sd35 0:2000 val test unseen train
  E=$SD/latents_arnold_dense_pertic_eval_sd35
  for spec in "val 6000:6100" "test 7000:7100" "arenas_678 60:120"; do
    set -- $spec
    remote "cd $SD/repo && \$HOME/wanenc/bin/python make_dense_eval_splits.py --latents-dir $E/$1 --expect-ids $2" 2>&1 \
      | tail -2 | tee -a "$LOG"
  done
  audit "$SD/latents_arnold_dense_pertic_sd35/arenas" "$SD/raw_arnold_dense/arenas" 16
  audit "$E/val" "$SD/raw_arnold_dense/arenas" 16
  audit "$E/test" "$SD/raw_arnold_dense/arenas" 16
  audit "$E/arenas_678" "$SD/raw_arnold_dense/arenas_678" 16
  log "MILESTONE sd35 0:2000 and eval corpora complete"
fi
if [ "$START_PHASE" -le 2 ]; then
  phase sd15 2000:6000 train
  audit "$SD/latents_arnold_dense_pertic/arenas" "$SD/raw_arnold_dense/arenas" 4
  log "MILESTONE sd15 2000:6000 complete"
fi
if [ "$START_PHASE" -le 3 ]; then
  phase sd35 2000:6000 train
  audit "$SD/latents_arnold_dense_pertic_sd35/arenas" "$SD/raw_arnold_dense/arenas" 16
  log "MILESTONE sd35 2000:6000 complete"
fi
log "ORCH_DONE"
