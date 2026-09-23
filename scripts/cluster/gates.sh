#!/bin/bash
# Zero to a usable node, step 4 of 5: the launch gates, in order, stopping at the first failure.
#
#   usage: [DOOM_ROOT=..] [UNET_GPU=0] [SD35_GPU=1] [VAES=sd15,sd35] [SMOKE_BBS="unet sd35"] \
#          [FIT=20] [STEPS=300] [WINDOWS=64] [SMOKE_TIMEOUT=7200] [EXPLAIN=<rc>] [DRY=1] gates.sh
#
# The order is the launch protocol's (`.claude/analyses/astra-prelaunch-audit-2026-09-21.md`, part
# C). Each gate answers one question, and a failure stops here rather than being carried into a
# multi-day run:
#
#   0 pin                which code is being certified? The commit of $REPO, which must be the
#                        commit of $D/repo (the checkout the launcher runs) with no tracked change.
#                        Any old certificate is revoked first; a new one is written only at GATES_GO.
#   1 sidecar audit      do the encoded sidecars equal the raw parquet, tic for tic, in both
#                        latent spaces? `check_action_alignment.py --audit-only`, which is the
#                        audit alone: the yaw scorer can return inconclusive for physics reasons
#                        and would otherwise hide a clean zero-mismatch audit behind exit 2.
#                        Val (100 episodes, 100,000 rows) and, since the 2026-09-22 review (H3),
#                        the TRAINING corpus too: TRAIN_AUDIT_EPISODES (all 2,000) with
#                        TRAIN_AUDIT_ROWS sampled rows each, zero mismatches required. Train shard
#                        0 was written by the old encoder and normalised afterwards, shard 1 by the
#                        new one, and only val had ever been audited.
#   1c inventory         does every training episode's latent array have as many rows as its
#                        sidecar, at exactly the recording's tics? `make_dense_eval_splits.py
#                        --check-only --expect-ids $TRAIN_IDS --raw-tics`: shapes and one int column
#                        per episode, no frames, over all of them.
#   1d latent alignment  do the stored latents belong to the rows their sidecars describe?
#                        `check_latent_alignment.py`, per encode shard of train and val: re-encode
#                        a full batch and the tail batch of ALIGN_EPISODES episodes with the
#                        shard's recorded encoder settings and compare (MAE <= 5e-3, p99 <= 2e-2;
#                        bit identity reported, not required), then decode the stored rows against
#                        the raw frames with rows shifted -4/-1/+1/+4, all of which the true
#                        alignment must beat by 3 dB. Every shard log must be present and every
#                        shard under one latent contract, and since Astra's third review that
#                        contract (autoencoder, subfolder, scale, shift, channels) must equal the
#                        other corpus of the space (`--contract-peer`, train against val) AND the
#                        space's own (`--space`: sd-vae-ft-mse, 0.18215, no shift, 4 channels; SD 3.5
#                        Medium's `vae`, 1.5305, 0.0609, 16 channels): train at scale 1 and val at
#                        scale 2 each passed their own directory. The tolerances are cross-host calibrated
#                        on ONE 4-channel episode: read the printed MAE, p99 and margin of the SD
#                        3.5 shards before trusting a pass there.
#   2 alignment gate     is the control that produced the motion stored on the row the trainer
#                        reads? Yaw-gated, shifts -1/0/+1, on val 6000:6100. Exit 0 aligned, 2
#                        misaligned, 3 inconclusive. Exit 3 is NOT approval.
#   3 fit check          does the launch configuration fit the card, at what rate, with accum 1?
#                        Run through the launcher, so it is the exact configuration that launches.
#   4 300-step smoke     does the real corpus train, validate, checkpoint and snapshot? Into a
#                        throwaway results directory, never the production one.
#   4b smoke probes      did the smoke train the conditioning? `smoke_probe.py` on the recovery
#                        checkpoint: the control MLP, position table, inflated context conv and
#                        SD 3.5 pooled_control moved and have finite nonzero gradients on a real
#                        val batch, and flipping the newest control changes the output (fixed noise).
#   4c resume            does a 10-update `--resume` of the smoke continue it (optimizer, EMA, step)?
#   5 evaluator readback can `eval_tf.py --tic-stride 1` load that snapshot and score raw frames,
#                        live AND EMA, at horizon 1 AND 4?
#
# Review 7.1 (docs/REVIEW_2026-09-22.md) also asks for, and these run with gates 1c and 1e: an exact
# inventory of the VAL corpus (ids, rows, tics and split file), and an emitted-window check on real
# training samples (`check_emitted_windows.py`: newest control = row r-1, and changing buttons[r]
# leaves the sample unchanged).
#
# One set of production arguments. The fit, the smoke, the resume and the certificate all go through
# `launch_nexttic.sh` under the same production settings (`prod_env`: DOOM_ROOT, the interpreter,
# MB_UNET / MB_SD35 as MB, WORKERS, LAUNCH_STEPS as STEPS, TRAIN_IDS, VAL_IDS and the operator's
# EXTRA). The fit adds only FIT; the smoke and its resume change only STEPS and append their own
# `smoke_extra` after the operator's EXTRA (a results directory, validation, checkpoint and snapshot
# cadence, and the resume's `--resume`), so argparse's last-value rule gives them those settings and
# every other flag is the production one. Astra's third review reproduced a smoke at lr 5e-5 and 12
# workers certified as lr 0.1 and 4 workers, because the smoke dropped them.
#
# The certificate. Every gate that passes appends a result to $D/logs/gates_results.jsonl, scoped to
# the whole run, one latent space, or one backbone. At GATES_GO, after checking that the checkout did
# not move while the gates ran, `gate_certificate.py write` records for each smoked backbone: its
# resolved PRODUCTION launch command (launch_nexttic.sh CERT_QUERY=1 under `prod_env`, the values
# launch_runs.sh passes), the commit, the training and validation
# corpus fingerprints, the encoder records and every gate result that applies to it, into
# $D/GATES_CERT.json. `launch_nexttic.sh` recomputes all of it for the backbone it launches and
# refuses on any difference; the gates' own fit, smoke and resume runs pass GATE_RUN=1 because they
# come before the certificate exists. A backbone whose latent space was not gated cannot be written.
#
# What this gate set does NOT do. The audit's 200-update fit and the 10,000-window loader index
# check the protocol also asks for are not here: FIT=20 is what the task specifies, and 20 updates
# exercise memory and a rate estimate but not the EMA/Adam-state cadence. Treat the fit number as
# a sizing estimate, not a certificate, and read the smoke as the real-data evidence.
#
# The smoke runs through `launch_nexttic.sh`, so it appends to `$D/logs/train_<run>.log` and
# `$D/logs/resumes.log` and creates the production results directory empty. The training itself
# goes to `$SMOKE_DIR`, because a 300-step `event=end` marker in the production directory would
# make the evaluator think the real run had finished.
#
# DRY=1 prints every gate command, including the ones the inner launcher would issue, and touches
# nothing. EXPLAIN=<rc> prints how an alignment exit code is read and stops.
set -u
D=${DOOM_ROOT:-/data/doom}
DRY=${DRY:-0}
VAES=${VAES:-sd15,sd35}
SMOKE_BBS=${SMOKE_BBS:-unet sd35}
UNET_GPU=${UNET_GPU:-0}
SD35_GPU=${SD35_GPU:-1}
FIT=${FIT:-20}
STEPS=${STEPS:-300}
WINDOWS=${WINDOWS:-64}
SMOKE_TIMEOUT=${SMOKE_TIMEOUT:-7200}
SMOKE_DIR=${SMOKE_DIR:-$D/results_smoke}
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
LAUNCH=${LAUNCH:-$REPO/scripts/spiderman/launch_nexttic.sh}
PY=${PY:-$D/env/bin/python}
RAW=$D/raw_arnold_dense
VAL_IDS=${VAL_IDS:-6000:6100}
TRAIN_IDS=${TRAIN_IDS:-0:2000}
TRAIN_AUDIT_EPISODES=${TRAIN_AUDIT_EPISODES:-2000}
TRAIN_AUDIT_ROWS=${TRAIN_AUDIT_ROWS:-500}
ALIGN_EPISODES=${ALIGN_EPISODES:-2}   # episodes per encode shard for the stored-latent alignment gate
EMIT_WINDOWS=${EMIT_WINDOWS:-256}     # real training windows the emitted-window check reads
CERT=$D/GATES_CERT.json
RESULTS=$D/logs/gates_results.jsonl
RUN_REPO=$D/repo              # the checkout launch_nexttic.sh runs (`cd $D/repo`)
# the production launch the certificate pins, with launch_runs.sh's own defaults
LAUNCH_STEPS=${LAUNCH_STEPS:-400000}
MB_UNET=${MB_UNET:-32}
MB_SD35=${MB_SD35:-32}
CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 16)
DEFAULT_WORKERS=$(( CORES / 4 ))
[ "$DEFAULT_WORKERS" -lt 4 ] && DEFAULT_WORKERS=4
[ "$DEFAULT_WORKERS" -gt 16 ] && DEFAULT_WORKERS=16
WORKERS=${WORKERS:-$DEFAULT_WORKERS}
PROD_EXTRA=${EXTRA:-}         # the operator's extra trainer flags: part of the production command
IFS=, read -r -a SPACES <<< "$VAES"

gate_fail() { echo "GATE_FAILED $1: ${*:2}" >&2; exit "${RC_FAIL:-1}"; }

explain() {   # how check_action_alignment.py's exit code is read
  case "$1" in
    0) echo "alignment exit 0: aligned. Yaw shift 0 leads both neighbours by the margin and the sidecar audit is clean." ;;
    2) echo "alignment exit 2: misaligned. Either a sidecar disagrees with the raw parquet, or the control that produced the motion is not on the row the trainer reads. STOP: repair the corpus, do not launch." ;;
    3) echo "alignment exit 3: inconclusive. Too few scored boundary rows, too few episodes, or no yaw verdict at all. STOP: this is not approval, and certifying nothing must not read as a pass." ;;
    *) echo "alignment exit $1: the gate itself failed to run. STOP and read the traceback." ;;
  esac
}
[ -n "${EXPLAIN:-}" ] && { explain "$EXPLAIN"; exit 0; }

suffix()   { [ "$1" = sd35 ] && echo _sd35 || echo ""; }
bb_space() { [ "$1" = sd35 ] && echo sd35 || echo sd15; }
bb_gpu()   { [ "$1" = sd35 ] && echo "$SD35_GPU" || echo "$UNET_GPU"; }
bb_chan()  { [ "$1" = sd35 ] && echo 16 || echo 4; }
run_name() { [ "$1" = sd35 ] && echo 042-sd35-nexttic || echo 040-unet-nexttic; }

audit_cmd() {   # audit_cmd <space>
  echo "$PY $REPO/check_action_alignment.py --audit-only" \
       "--latents-dir $D/latents_arnold_dense_pertic_eval$(suffix "$1")/val" \
       "--audit-parquet-dir $RAW/arenas --episodes 100 --audit-rows 100000" \
       "--canonical $RAW/canonical_controls.json --seed 0"
}
train_audit_cmd() {   # train_audit_cmd <space>: the same audit over the training corpus
  echo "$PY $REPO/check_action_alignment.py --audit-only" \
       "--latents-dir $D/latents_arnold_dense_pertic$(suffix "$1")/arenas" \
       "--audit-parquet-dir $RAW/arenas --episodes $TRAIN_AUDIT_EPISODES --audit-rows $TRAIN_AUDIT_ROWS" \
       "--canonical $RAW/canonical_controls.json --seed 0"
}
inventory_cmd() {   # inventory_cmd <space>: rows and tics of every training episode
  echo "$PY $REPO/make_dense_eval_splits.py --check-only" \
       "--latents-dir $D/latents_arnold_dense_pertic$(suffix "$1")/arenas" \
       "--expect-ids $TRAIN_IDS --sample 0 --raw-tics $RAW/arenas"
}
space_gpu() { [ "$1" = sd35 ] && echo "$SD35_GPU" || echo "$UNET_GPU"; }
latent_align_cmd() {   # latent_align_cmd <space> <train|val>: re-encode and shifted-decode, per shard
  local S TRAIN VAL DIR PEER
  S=$(suffix "$1")
  TRAIN=$D/latents_arnold_dense_pertic$S/arenas; VAL=$D/latents_arnold_dense_pertic_eval$S/val
  DIR=$VAL; PEER=$TRAIN
  [ "$2" = train ] && { DIR=$TRAIN; PEER=$VAL; }
  # --space and --contract-peer: one latent contract for the space, the backbone's, in train AND val
  echo "$PY $REPO/check_latent_alignment.py --latents-dir $DIR --parquet-dir $RAW/arenas" \
       "--space $1 --contract-peer $PEER" \
       "--episodes-per-shard $ALIGN_EPISODES --device cuda:$(space_gpu "$1") --cache-dir $D/hf/hub" \
       "--out $D/logs/gate1d_latent_align_$1_$2.json"
}
pin_of() { git -C "$1" rev-parse HEAD 2>/dev/null; }
record() {   # record <gate> <all | space:V | bb:B> <detail>: one passed gate, for the certificate
  local detail=${3//\"/}; detail=${detail//\\/}
  printf '{"gate": "%s", "scope": "%s", "status": "ok", "detail": "%s"}\n' "$1" "$2" "$detail" >> "$RESULTS"
}
bb_mb() { [ "$1" = sd35 ] && echo "$MB_SD35" || echo "$MB_UNET"; }
prod_env() {   # prod_env <backbone>: PROD_ENV=(NAME=VALUE ...), the production launch every gate launch starts from
  PROD_ENV=(DOOM_ROOT="$D" PY="$PY" PY_SD35="$PY" MB="$(bb_mb "$1")" WORKERS="$WORKERS" STEPS="$LAUNCH_STEPS"
        TRAIN_IDS="$TRAIN_IDS" VAL_IDS="$VAL_IDS" EXTRA="$PROD_EXTRA")
}
cert_cmd() {   # cert_cmd <backbone>: the production command, as the launcher resolves it
  prod_env "$1"
  env "${PROD_ENV[@]}" CERT_QUERY=1 DRY=0 bash "$LAUNCH" "$(bb_gpu "$1")" "$1"
}
bb_latents() {   # bb_latents <backbone> <train|val>
  local S; S=$(suffix "$(bb_space "$1")")
  if [ "$2" = train ]; then echo "$D/latents_arnold_dense_pertic$S/arenas"; else echo "$D/latents_arnold_dense_pertic_eval$S/val"; fi
}
cert_write_cmd() {   # cert_write_cmd <backbone> <command>
  echo "$PY $REPO/gate_certificate.py write --cert $CERT --backbone $1 --space $(bb_space "$1") --gpu $(bb_gpu "$1")" \
       "--repo $RUN_REPO --train-latents $(bb_latents "$1" train) --train-ids $TRAIN_IDS" \
       "--val-latents $(bb_latents "$1" val) --val-ids $VAL_IDS --results $RESULTS --command"
}
align_cmd() {   # align_cmd <space>: the yaw gate at the protocol's thresholds
  echo "$PY $REPO/check_action_alignment.py" \
       "--latents-dir $D/latents_arnold_dense_pertic_eval$(suffix "$1")/val" \
       "--audit-parquet-dir $RAW/arenas --episodes 100 --audit-rows 100000" \
       "--min-yaw 0.25 --min-move 0.05 --min-rows 1000 --min-episodes 20 --min-per-class 100" \
       "--min-accuracy 0.95 --margin 0.20 --bootstrap 1000 --seed 0"
}
smoke_extra() { # smoke_extra <backbone>: the operator's EXTRA, then the smoke's operational overrides
  echo "${PROD_EXTRA:+$PROD_EXTRA }--results-dir $SMOKE_DIR/$1 --val-every 100 --val-windows 128 --ckpt-every $STEPS" \
       "--snapshot-every $STEPS --local-snapshots --keep-last 1"
}
resume_extra() { # resume_extra <backbone>: the smoke's EXTRA, then resume its checkpoint for 10 updates
  echo "$(smoke_extra "$1") --resume $SMOKE_DIR/$1/$(printf '%07d' "$STEPS").pt --ckpt-every 10"
}
bb_paths() {    # bb_paths <backbone>: where the evaluator and the probe build the backbone from
  case $1 in
    sd35) echo "--sd35-path stabilityai/stable-diffusion-3.5-medium" ;;
    unet) echo "--sd-path CompVis/stable-diffusion-v1-4" ;;
  esac
}
readback_cmd() {  # readback_cmd <backbone> <live|ema> <horizon>
  local S; S=$(suffix "$(bb_space "$1")")
  local EXTRA EMA=""
  EXTRA=$(bb_paths "$1")
  [ "$1" = sd35 ] && EXTRA="$EXTRA --vae-path stabilityai/stable-diffusion-3.5-medium --vae-subfolder vae --latent-scale 1.5305 --latent-shift 0.0609"
  [ "$2" = ema ] && EMA=" --use-ema"
  echo "$PY $REPO/eval_tf.py --backbone $1 --latent-channels $(bb_chan "$1")" \
       "--ckpt $SMOKE_DIR/$1/snap_$(printf '%07d' "$STEPS").pt$EMA --tic-stride 1 --horizon-tics $3" \
       "--latents-dir $D/latents_arnold_dense_pertic_eval$S/val" \
       "--split $D/latents_arnold_dense_pertic_eval$S/split_val.json --subset val" \
       "--parquet-dir $RAW/arenas --num-windows $WINDOWS --batch-size 16 --steps 10" \
       "--context-frames 32 --num-actions 29 --hf-cache $D/hf/hub $EXTRA" \
       "--out-dir $SMOKE_DIR/$1/eval_tf_val_$2_h$3"
}
probe_cmd() {   # probe_cmd <backbone>: the conditioning pathways of the smoke's recovery checkpoint
  local S; S=$(suffix "$(bb_space "$1")")
  echo "$PY $REPO/smoke_probe.py --ckpt $SMOKE_DIR/$1/$(printf '%07d' "$STEPS").pt --backbone $1" \
       "--latent-channels $(bb_chan "$1") --latents-dir $D/latents_arnold_dense_pertic_eval$S/val" \
       "--episodes $VAL_IDS --device cuda:$(bb_gpu "$1") --hf-cache $D/hf/hub $(bb_paths "$1")" \
       "--out $D/logs/gate4b_probe_$1.json"
}
val_inventory_cmd() {   # val_inventory_cmd <space>: exactly the val ids, rows, tics and split file
  local S; S=$(suffix "$1")
  echo "$PY $REPO/make_dense_eval_splits.py --check-only" \
       "--latents-dir $D/latents_arnold_dense_pertic_eval$S/val" \
       "--expect-ids $VAL_IDS --sample 0 --raw-tics $RAW/arenas" \
       "--split-file $D/latents_arnold_dense_pertic_eval$S/split_val.json"
}
windows_cmd() {   # windows_cmd <space>: the emitted-window contract on real training samples
  echo "$PY $REPO/check_emitted_windows.py --latents-dir $D/latents_arnold_dense_pertic$(suffix "$1")/arenas" \
       "--episodes $TRAIN_IDS --windows $EMIT_WINDOWS --context-frames 32" \
       "--latent-channels $( [ "$1" = sd35 ] && echo 16 || echo 4) --out $D/logs/gate1e_windows_$1.json"
}
wait_session() {   # wait_session <backbone> <gate label>: until the tmux session ends, or the timeout
  local SESSION=train-$1-nexttic WAITED=0
  while tmux has-session -t "$SESSION" 2>/dev/null; do
    sleep 20; WAITED=$(( WAITED + 20 ))
    if [ "$WAITED" -ge "$SMOKE_TIMEOUT" ]; then
      tmux kill-session -t "$SESSION"
      gate_fail "$2 ($1)" "still running after ${SMOKE_TIMEOUT}s; killed $SESSION"
    fi
  done
}
launcher() {   # launcher <backbone> <overrides of the production settings as NAME=VALUE ...>
  local BB=$1; shift
  prod_env "$BB"
  env "${PROD_ENV[@]}" DRY="$DRY" GATE_RUN=1 "$@" bash "$LAUNCH" "$(bb_gpu "$BB")" "$BB"
}

if [ "$DRY" = 1 ]; then
  echo "DRY gates root=$D spaces=${SPACES[*]} smoke=$SMOKE_BBS unet_gpu=$UNET_GPU sd35_gpu=$SD35_GPU"
  echo "DRY gate0 pin: revoke $CERT; certify git -C $REPO rev-parse HEAD, which must equal $RUN_REPO's HEAD with no tracked change"
  for V in "${SPACES[@]}"; do echo "DRY gate1 audit $V $(audit_cmd "$V")"; done
  for V in "${SPACES[@]}"; do echo "DRY gate1 train audit $V $(train_audit_cmd "$V")"; done
  for V in "${SPACES[@]}"; do echo "DRY gate1c inventory $V $(inventory_cmd "$V")"; done
  for V in "${SPACES[@]}"; do echo "DRY gate1c val inventory $V $(val_inventory_cmd "$V")"; done
  for V in "${SPACES[@]}"; do
    for C in train val; do echo "DRY gate1d latent alignment $V $C $(latent_align_cmd "$V" "$C")"; done
  done
  for V in "${SPACES[@]}"; do echo "DRY gate1e emitted windows $V $(windows_cmd "$V")"; done
  echo "DRY gate2 alignment ${SPACES[0]} val $VAL_IDS $(align_cmd "${SPACES[0]}")"
  explain 0; explain 2; explain 3
  for BB in $SMOKE_BBS; do
    echo "DRY gate3 fit $BB"
    launcher "$BB" FIT="$FIT"
    echo "DRY gate3 record steps_per_s and peak_mem_gb from $D/results_spiderman/$(run_name "$BB")/fitcheck/log.jsonl (accum must be 1)"
  done
  for BB in $SMOKE_BBS; do
    echo "DRY gate4 smoke $BB"
    launcher "$BB" STEPS="$STEPS" EXTRA="$(smoke_extra "$BB")"
    echo "DRY gate4 expect $SMOKE_DIR/$BB/$(printf '%07d' "$STEPS").pt and $SMOKE_DIR/$BB/snap_$(printf '%07d' "$STEPS").pt"
    echo "DRY gate4 refuse if $D/results_spiderman/$(run_name "$BB")/log.jsonl exists: the launcher resumes the newest recovery checkpoint of the production run, and the smoke would continue it into the smoke directory"
  done
  for BB in $SMOKE_BBS; do echo "DRY gate4b probes $BB $(probe_cmd "$BB")"; done
  for BB in $SMOKE_BBS; do
    echo "DRY gate4c resume $BB for 10 updates"
    launcher "$BB" STEPS="$(( STEPS + 10 ))" EXTRA="$(resume_extra "$BB")"
    echo "DRY gate4c expect 'resumed weights from $SMOKE_DIR/$BB/$(printf '%07d' "$STEPS").pt at step $STEPS with optimizer', an end event at step $(( STEPS + 10 )) and $SMOKE_DIR/$BB/$(printf '%07d' "$(( STEPS + 10 ))").pt"
  done
  for BB in $SMOKE_BBS; do
    for V in live ema; do for H in 1 4; do echo "DRY gate5 readback $BB $V h$H $(readback_cmd "$BB" "$V" "$H")"; done; done
  done
  echo "DRY summary GATES_GO with the fit rates, the smoke checkpoints and the readback PSNR"
  for BB in $SMOKE_BBS; do
    echo "DRY certificate command $BB $(cert_cmd "$BB")"
    echo "DRY certificate $BB $(cert_write_cmd "$BB") \"<the certificate command above: STEPS=$LAUNCH_STEPS MB=$(bb_mb "$BB") WORKERS=$WORKERS${PROD_EXTRA:+ EXTRA=$PROD_EXTRA}>\""
  done
  echo "DRY certificate written to $CERT only after checking the checkout did not move"
  exit 0
fi

[ -x "$PY" ] || gate_fail preflight "no interpreter at $PY (run setup_node.sh first)"
[ -f "$LAUNCH" ] || gate_fail preflight "no launch_nexttic.sh at $LAUNCH"
mkdir -p "$SMOKE_DIR" "$D/logs" || gate_fail preflight "cannot write under $D"
REPORT=$D/GATES.txt
: > "$REPORT"
say() { echo "$*" | tee -a "$REPORT"; }

# --- gate 0: pin the commit being certified ------------------------------------------------
# revoked first: a gate run that fails or is interrupted must not leave an older certificate
# standing for whatever the checkout is now
rm -f "$CERT"
: > "$RESULTS"
say "$(date -Iseconds) gate 0: pin"
COMMIT=$(pin_of "$REPO") || gate_fail "0 pin" "$REPO is not a git checkout, so the gates cannot say which code they certify"
RUN_COMMIT=$(pin_of "$RUN_REPO") || RUN_COMMIT="none"
[ "$RUN_COMMIT" = "$COMMIT" ] \
  || gate_fail "0 pin" "the launcher runs $RUN_REPO at $RUN_COMMIT but the gates run $REPO at $COMMIT"
git -C "$REPO" diff --quiet HEAD -- \
  || gate_fail "0 pin" "tracked files in $REPO differ from $COMMIT; commit or discard them, the gates certify a commit"
say "  certifying $COMMIT ($REPO)"
record "0 pin" all "$COMMIT"

# --- gate 1: the sidecar audit ---------------------------------------------------------
for V in "${SPACES[@]}"; do
  say "$(date -Iseconds) gate 1: sidecar audit, $V"
  # shellcheck disable=SC2046
  $(audit_cmd "$V") > "$D/logs/gate1_audit_$V.json" 2>&1 \
    || gate_fail "1 sidecar audit ($V)" "the encoded sidecars disagree with the raw parquet; see $D/logs/gate1_audit_$V.json"
  say "  ok: $(grep -o '"mismatches": [0-9]*' "$D/logs/gate1_audit_$V.json" | head -1), $(grep -o '"rows_checked": [0-9]*' "$D/logs/gate1_audit_$V.json" | head -1)"
  record "1 sidecar audit val" "space:$V" "$(grep -o '"rows_checked": [0-9]*' "$D/logs/gate1_audit_$V.json" | head -1)"
done
for V in "${SPACES[@]}"; do
  say "$(date -Iseconds) gate 1: sidecar audit of the TRAINING corpus, $V ($TRAIN_AUDIT_EPISODES episodes, $TRAIN_AUDIT_ROWS rows each)"
  # shellcheck disable=SC2046
  $(train_audit_cmd "$V") > "$D/logs/gate1_train_audit_$V.json" 2>&1 \
    || gate_fail "1 sidecar audit (train, $V)" "the training sidecars disagree with the raw parquet; see $D/logs/gate1_train_audit_$V.json"
  say "  ok: $(grep -o '"episodes": [0-9]*' "$D/logs/gate1_train_audit_$V.json" | head -1), $(grep -o '"mismatches": [0-9]*' "$D/logs/gate1_train_audit_$V.json" | head -1), $(grep -o '"rows_checked": [0-9]*' "$D/logs/gate1_train_audit_$V.json" | head -1)"
  record "1 sidecar audit train" "space:$V" "$(grep -o '"rows_checked": [0-9]*' "$D/logs/gate1_train_audit_$V.json" | head -1)"
done

# --- gate 1c: every training episode's rows and tics ---------------------------------------
for V in "${SPACES[@]}"; do
  say "$(date -Iseconds) gate 1c: rows and tics of every training episode, $V ($TRAIN_IDS)"
  # shellcheck disable=SC2046
  $(inventory_cmd "$V") > "$D/logs/gate1c_inventory_$V.json" 2>&1 \
    || gate_fail "1c inventory ($V)" "a training episode is missing, orphaned, or its latents, sidecar and recording disagree on rows or tics; see $D/logs/gate1c_inventory_$V.json"
  say "  ok: $(grep -c '^  [0-9]' "$D/logs/gate1c_inventory_$V.json" 2>/dev/null || echo '?') ids listed, no problems"
  record "1c inventory train" "space:$V" "$TRAIN_IDS"
  say "$(date -Iseconds) gate 1c: exact val inventory, $V ($VAL_IDS, rows, tics and split file)"
  # shellcheck disable=SC2046
  $(val_inventory_cmd "$V") > "$D/logs/gate1c_val_inventory_$V.json" 2>&1 \
    || gate_fail "1c inventory val ($V)" "the val corpus is not exactly $VAL_IDS with rows and tics matching the recording and its split file; see $D/logs/gate1c_val_inventory_$V.json"
  record "1c inventory val" "space:$V" "$VAL_IDS"
done

# --- gate 1d: do the stored latents belong to their sidecar rows? ----------------------------
# per encode shard: re-encode sampled rows with the shard's recorded settings and compare, then
# decode the stored rows against the raw frames with -4/-1/+1/+4 shifted negative controls
for V in "${SPACES[@]}"; do
  for C in train val; do
    say "$(date -Iseconds) gate 1d: stored-latent alignment, $V $C ($ALIGN_EPISODES episode(s) per shard)"
    # shellcheck disable=SC2046
    $(latent_align_cmd "$V" "$C") > "$D/logs/gate1d_latent_align_${V}_$C.log" 2>&1 \
      || gate_fail "1d latent alignment ($V $C)" "a shard's stored latents do not reproduce, a shifted alignment scores as well as the true one, or the corpus is not under the $V contract its peer and backbone share; see $D/logs/gate1d_latent_align_${V}_$C.json"
    say "  $(grep '^shard ' "$D/logs/gate1d_latent_align_${V}_$C.log" | tr '\n' ';')"
    record "1d latent alignment $C" "space:$V" "$(grep '^shard ' "$D/logs/gate1d_latent_align_${V}_$C.log" | tr '\n' ';')"
  done
done

# --- gate 1e: the windows the training loader emits from the real corpus -------------------------
for V in "${SPACES[@]}"; do
  say "$(date -Iseconds) gate 1e: emitted windows, $V ($EMIT_WINDOWS real training windows)"
  # shellcheck disable=SC2046
  $(windows_cmd "$V") > "$D/logs/gate1e_windows_$V.log" 2>&1 \
    || gate_fail "1e emitted windows ($V)" "a real window broke the causal contract; see $D/logs/gate1e_windows_$V.json"
  say "  $(grep '^EMITTED_WINDOWS' "$D/logs/gate1e_windows_$V.log")"
  record "1e emitted windows" "space:$V" "$(grep '^EMITTED_WINDOWS' "$D/logs/gate1e_windows_$V.log")"
done

# --- gate 2: the alignment gate on val ---------------------------------------------------
say "$(date -Iseconds) gate 2: alignment on val $VAL_IDS (${SPACES[0]})"
# shellcheck disable=SC2046
$(align_cmd "${SPACES[0]}") > "$D/logs/gate2_alignment.json" 2>&1
A_RC=$?
say "  $(explain $A_RC)"
[ $A_RC -eq 0 ] || gate_fail "2 alignment" "exit $A_RC; see $D/logs/gate2_alignment.json"
record "2 alignment" all "exit 0 on val $VAL_IDS (${SPACES[0]})"

# --- gate 3: the fit check, through the launcher -----------------------------------------
for BB in $SMOKE_BBS; do
  say "$(date -Iseconds) gate 3: fit check, $BB, $FIT updates on gpu $(bb_gpu "$BB")"
  launcher "$BB" FIT="$FIT" > "$D/logs/gate3_fit_$BB.log" 2>&1 \
    || gate_fail "3 fit check ($BB)" "the launch configuration did not run; see $D/logs/gate3_fit_$BB.log"
  FITLOG=$D/results_spiderman/$(run_name "$BB")/fitcheck/log.jsonl
  NUMS=$("$PY" - "$FITLOG" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if '"fit_check"' in l]
if not rows:
    print("NO_FIT_CHECK_EVENT"); raise SystemExit(1)
r = rows[-1]
print(f"steps_per_s={r['steps_per_s']:.3f} peak_mem_gb={r['peak_mem_gb']} "
      f"peak_reserved_gb={r.get('peak_reserved_gb')} global_batch={r['global_batch']} accum={r['accum']}")
raise SystemExit(0 if r['accum'] == 1 and r['global_batch'] == 32 else 2)
PY
) || gate_fail "3 fit check ($BB)" "no usable fit_check event, or accumulation is not 1: $NUMS ($FITLOG)"
  say "  $BB $NUMS"
  record "3 fit" "bb:$BB" "$NUMS"
done

# --- gate 4: the 300-step real-data smoke -------------------------------------------------
for BB in $SMOKE_BBS; do
  SD=$SMOKE_DIR/$BB
  SESSION=train-$BB-nexttic
  PROD=$D/results_spiderman/$(run_name "$BB")
  # the launcher derives --resume from the PRODUCTION directory, so a smoke run after a launch
  # would carry that run's weights and optimizer into the smoke: gate before launching
  [ -f "$PROD/log.jsonl" ] \
    && gate_fail "4 smoke ($BB)" "$PROD has already trained; the smoke would resume it. Stop and archive that run, or skip this gate deliberately."
  tmux has-session -t "$SESSION" 2>/dev/null \
    && gate_fail "4 smoke ($BB)" "tmux session $SESSION is already alive; stop it with tmux kill-session -t $SESSION"
  rm -rf "$SD"; mkdir -p "$SD"
  say "$(date -Iseconds) gate 4: $STEPS-step smoke, $BB, into $SD"
  launcher "$BB" STEPS="$STEPS" EXTRA="$(smoke_extra "$BB")" \
    || gate_fail "4 smoke ($BB)" "the launcher refused to start the smoke"
  wait_session "$BB" "4 smoke"
  CK=$SD/$(printf '%07d' "$STEPS").pt
  SNAP=$SD/snap_$(printf '%07d' "$STEPS").pt
  [ -f "$CK" ] || gate_fail "4 smoke ($BB)" "no recovery checkpoint at $CK (see $D/logs/train_$(run_name "$BB").log)"
  [ -f "$SNAP" ] || gate_fail "4 smoke ($BB)" "no snapshot at $SNAP"
  grep -q '"event": "end"' "$SD/log.jsonl" 2>/dev/null \
    || gate_fail "4 smoke ($BB)" "the smoke never reached its end event; the run died mid-way"
  say "  ok: $(du -h "$CK" | cut -f1) recovery checkpoint and $(du -h "$SNAP" | cut -f1) snapshot"
  record "4 smoke" "bb:$BB" "$STEPS steps"
done

# --- gate 4b: did the smoke train the conditioning pathways? ------------------------------------
for BB in $SMOKE_BBS; do
  say "$(date -Iseconds) gate 4b: smoke probes, $BB"
  # shellcheck disable=SC2046
  $(probe_cmd "$BB") > "$D/logs/gate4b_probe_$BB.log" 2>&1 \
    || gate_fail "4b smoke probes ($BB)" "$(grep '^SMOKE_PROBE' "$D/logs/gate4b_probe_$BB.log" | head -1); see $D/logs/gate4b_probe_$BB.json"
  record "4b smoke probes" "bb:$BB" "$(grep '^SMOKE_PROBE' "$D/logs/gate4b_probe_$BB.log" | head -1)"
done

# --- gate 4c: a 10-update resume of the smoke ---------------------------------------------------
for BB in $SMOKE_BBS; do
  SD=$SMOKE_DIR/$BB
  NEXT=$(( STEPS + 10 ))
  TRAINLOG=$D/logs/train_$(run_name "$BB").log
  say "$(date -Iseconds) gate 4c: resume the $BB smoke from step $STEPS for 10 updates"
  launcher "$BB" STEPS="$NEXT" EXTRA="$(resume_extra "$BB")" \
    || gate_fail "4c resume ($BB)" "the launcher refused to resume the smoke"
  wait_session "$BB" "4c resume"
  grep -q "resumed weights from $SD/$(printf '%07d' "$STEPS").pt at step $STEPS with optimizer" "$TRAINLOG" \
    || gate_fail "4c resume ($BB)" "the resume did not restore the optimizer, scheduler and EMA from step $STEPS; see $TRAINLOG"
  [ -f "$SD/$(printf '%07d' "$NEXT").pt" ] || gate_fail "4c resume ($BB)" "no checkpoint at step $NEXT"
  LAST_END=$("$PY" -c "import json,sys; e=[json.loads(l) for l in open(sys.argv[1]) if '\"event\": \"end\"' in l]; print(e[-1]['step'] if e else -1)" "$SD/log.jsonl")
  [ "$LAST_END" = "$NEXT" ] || gate_fail "4c resume ($BB)" "the resumed run ended at step $LAST_END, not $NEXT"
  record "4c resume" "bb:$BB" "$STEPS -> $NEXT"
done

# --- gate 5: the evaluator readback, live and EMA, one tic and four ------------------------------
for BB in $SMOKE_BBS; do
 ALL=""
 for V in live ema; do for H in 1 4; do
  say "$(date -Iseconds) gate 5: eval_tf readback of the $BB snapshot, $V weights, horizon $H, on $WINDOWS val windows"
  # shellcheck disable=SC2046
  $(readback_cmd "$BB" "$V" "$H") > "$D/logs/gate5_readback_${BB}_${V}_h$H.log" 2>&1 \
    || gate_fail "5 readback ($BB $V h$H)" "eval_tf.py could not score the snapshot; see $D/logs/gate5_readback_${BB}_${V}_h$H.log"
  M=$SMOKE_DIR/$BB/eval_tf_val_${V}_h$H/metrics.json
  NUMS=$("$PY" - "$M" <<'PY'
import json, math, sys
m = json.load(open(sys.argv[1]))
def num(k):
    v = m.get(k)
    return v if isinstance(v, (int, float)) else (v or {}).get("mean") if isinstance(v, dict) else None
got = {k: num(k) for k in ("psnr_raw", "lpips_raw", "persist_psnr_raw", "windows", "n")}
bad = [k for k in ("psnr_raw",) if got[k] is None or not math.isfinite(got[k])]
print(" ".join(f"{k}={v}" for k, v in got.items() if v is not None))
raise SystemExit(1 if bad else 0)
PY
) || gate_fail "5 readback ($BB $V h$H)" "no finite raw PSNR in $M ($NUMS)"
  say "  $BB $V h$H $NUMS"
  ALL="$ALL $V/h$H: $NUMS;"
 done; done
 record "5 readback" "bb:$BB" "$ALL"
done

# the certificate: only if the checkout is still the commit gate 0 pinned, with no tracked change
[ "$(pin_of "$REPO")" = "$COMMIT" ] && [ "$(pin_of "$RUN_REPO")" = "$COMMIT" ] && git -C "$REPO" diff --quiet HEAD -- \
  || gate_fail "0 pin" "the checkout moved while the gates ran (certified $COMMIT); rerun the gates"
for BB in $SMOKE_BBS; do
  PROD_CMD=$(cert_cmd "$BB") || gate_fail "certificate ($BB)" "the launcher could not resolve the production command"
  # shellcheck disable=SC2046
  $(cert_write_cmd "$BB") "$PROD_CMD" >> "$REPORT" 2>&1 \
    || { rm -f "$CERT"; gate_fail "certificate ($BB)" "gate_certificate.py refused to certify $BB; see $REPORT"; }
done
say "GATES_GO $(date -Iseconds) commit=$COMMIT root=$D spaces=${SPACES[*]} smoked=$SMOKE_BBS certificate=$CERT"
say "the gates say the corpus, the configuration and the evaluator agree; the training numbers are still unmeasured"
