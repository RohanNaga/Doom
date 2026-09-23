#!/bin/bash
# Streamed SD 3.5 per-tic encode of the dense corpus on one Superman A4000.
#
#   usage: [K=4] [THREADS=6] [DRY=1] [DST_ROOT=..] [TRAIN_IDS=..] [VAL_IDS=..] [TEST_IDS=..] [UNSEEN_IDS=..] \
#          superman_stream_encode.sh <gpu> <i/n> <train|val|test|unseen|evals|all>
#
# The raw episodes live on Spiderman and Superman has no room for them, so each worker pulls K
# parquets at a time into /home/rohan/Doom/stream/<gpu>/, runs encode_parquet.py on exactly those
# files with the arguments `VAE=sd35 encode_nexttic.sh` gives it, and pushes the latents and
# sidecars into the directory that launcher would have written. The next batch is pulled while the
# current one encodes, and each batch ships while the next one encodes, so the card never waits on
# the network.
#
#   corpus  source on Spiderman           ids         destination on Spiderman
#   train   raw_arnold_dense/arenas       0:2000      latents_arnold_dense_pertic_sd35/arenas
#   val     raw_arnold_dense/arenas       6000:6100   latents_arnold_dense_pertic_eval_sd35/val
#   test    raw_arnold_dense/arenas       7000:7100   latents_arnold_dense_pertic_eval_sd35/test
#   unseen  raw_arnold_dense/arenas_678   60:120      latents_arnold_dense_pertic_eval_sd35/arenas_678
#
# Unseen is 60:120, not the launcher's old 0:60 default: 51 of ids 0:60 are worker-first episodes in
# which Arnold's weapon switches execute (review of Sep 22, finding H1).
#
# Sharding is by position in the id range, ids[i::n], so shards of one n are disjoint. Every episode
# whose latents already exist on Spiderman is skipped, which is what makes a restart, a change of n
# or a Spiderman shard working the same corpus safe; the batch is re-checked just before it encodes.
#
# Batch size 64 and bf16 are not free choices: cuDNN picks its algorithm from the batch shape under
# autocast, so the same frame encodes differently at another batch size (encode_nexttic.sh).
#
# A shipped episode is verified before it becomes visible: the latents travel under a `.smpart`
# name, which `ep_*_latents.npy` globs do not match, their md5 and the sidecar's are checked on
# Spiderman, and only then is the part renamed. The summary lines go to `episodes_<10+i>.jsonl` and
# the run record to `encode_meta_<10+i>.json` (Spiderman's own shards own 00..09), so the directory
# holds what encode_nexttic.sh would have produced plus one file pair per Superman shard.
#
# Stop one worker after its current batch with `touch /home/rohan/Doom/stream/<gpu>/STOP`, or at
# once with `tmux kill-session -t enc-sd35-<gpu>`; either way a relaunch resumes.
set -u
GPU=${1:?gpu}
SHARD=${2:?shard i/n}
CORPUS=${3:-train}
I=${SHARD%/*}
N=${SHARD#*/}
[ "$I" -ge 0 ] 2>/dev/null && [ "$I" -lt "$N" ] || { echo "bad shard $SHARD" >&2; exit 2; }
K=${K:-4}                       # episodes per encoder process: amortises torch import and VAE load
THREADS=${THREADS:-6}           # PNG decode threads; 48 cores over 8 workers. Output bytes do not depend on it
DRY=${DRY:-0}
MIN_FREE_GB=${MIN_FREE_GB:-60}
B=/home/rohan/Doom/sd35enc
REPO=$B/repo
PY=${PY:-$HOME/miniconda3/envs/doomenc/bin/python}
CANON=$B/canonical_controls.json
HUB=$B/hf/hub
W=/home/rohan/Doom/stream/$GPU
STATE=$B/state
LOG=/home/rohan/Doom/logs/enc-sd35-$GPU.log
ELOG=/home/rohan/Doom/logs/enc-sd35-$GPU.encoder.log
R=rnagabhi@128.2.204.110
SD=/sata2/data/rnagabhi/doom
DST_ROOT=${DST_ROOT:-$SD}
NN=$(printf %02d $((10 + I)))
# one TCP connection per worker: every ssh and rsync below rides this worker's control socket
SOCK=$HOME/.ssh/cm/sm-enc-$GPU
SSH="ssh -i $HOME/.ssh/id_ed25519_doom -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=60 \
 -o ServerAliveCountMax=5 -o ControlMaster=auto -o ControlPath=$SOCK -o ControlPersist=12h"
VAE_FLAGS=(--vae-id stabilityai/stable-diffusion-3.5-medium --vae-subfolder vae --latent-channels 16
           --scaling-factor 1.5305 --shift-factor 0.0609)
# cuda:K must be nvidia-smi's K; the CUDA default order is fastest-first
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

log() { echo "$(date -Iseconds) gpu=$GPU shard=$SHARD $*" | tee -a "$LOG"; }
remote() { $SSH $R "$@"; }

spec() {        # sets SRC, IDS, DST for one corpus
  case $1 in
    train)  SRC=$SD/raw_arnold_dense/arenas;     IDS=${TRAIN_IDS:-0:2000};    DST=$DST_ROOT/latents_arnold_dense_pertic_sd35/arenas ;;
    val)    SRC=$SD/raw_arnold_dense/arenas;     IDS=${VAL_IDS:-6000:6100};   DST=$DST_ROOT/latents_arnold_dense_pertic_eval_sd35/val ;;
    test)   SRC=$SD/raw_arnold_dense/arenas;     IDS=${TEST_IDS:-7000:7100};  DST=$DST_ROOT/latents_arnold_dense_pertic_eval_sd35/test ;;
    unseen) SRC=$SD/raw_arnold_dense/arenas_678; IDS=${UNSEEN_IDS:-60:120};   DST=$DST_ROOT/latents_arnold_dense_pertic_eval_sd35/arenas_678 ;;
    *) echo "unknown corpus $1 (train|val|test|unseen|evals|all)" >&2; exit 2 ;;
  esac
}

case $CORPUS in
  evals) CORPORA="val test unseen" ;;
  all)   CORPORA="train val test unseen" ;;
  *)     CORPORA=$CORPUS ;;
esac

name() { printf 'ep_%05d' "$1"; }

fetch() {       # fetch <dir> <ids...>: pull those parquets from $SRC; writes <dir>.rc
  local d=$1; shift
  rm -rf "$d"; mkdir -p "$d"
  for id in "$@"; do echo "$(name "$id").parquet"; done > "$d.list"
  rsync -a --files-from="$d.list" -e "$SSH" "$R:$SRC/" "$d/" > "$d.rsync.log" 2>&1
  echo $? > "$d.rc"
}

not_done() {    # print the ids among "$@" whose latents are not on Spiderman yet
  local names="" id
  for id in "$@"; do names="$names $(name "$id")_latents.npy"; done
  local have
  have=$(remote "cd $DST 2>/dev/null && ls $names 2>/dev/null") || true
  for id in "$@"; do
    grep -qx "$(name "$id")_latents.npy" <<< "$have" || echo "$id"
  done
}

patch_meta() {  # record where and from which code this shard's latents were made
  "$PY" - "$1" "$B/repo_git.txt" "$SHARD" "$GPU" <<'P'
import json, sys
path, gitfile, shard, gpu = sys.argv[1:5]
m = json.load(open(path))
head = open(gitfile).read().split()[0]
if m.get("git") in (None, "", "?"):
    m["git"] = head
m["encoded_on"] = {
    "host": "superman", "gpu": "NVIDIA RTX A4000", "gpu_index": int(gpu),
    "wrapper": "scripts/superman/superman_stream_encode.sh", "wrapper_shard": shard,
    "git_source": "HEAD of Spiderman's $D/repo when it was copied; Superman's copy has no .git",
    "note": "args.in_dir and args.out_dir are this worker's local streaming directories; each "
            "encoder process saw one batch of episodes, and the summary lines of every batch are "
            "gathered into the matching episodes_NN.jsonl"}
json.dump(m, open(path, "w"), indent=1)
P
}

wait_ship() {    # wait for the ship in flight (encode_corpus's $sp), and stop the worker if it failed
  [ -n "${sp:-}" ] || return 0
  wait "$sp"
  sp=""
  if [ "$(cat "$src_rc" 2>/dev/null)" != 0 ]; then
    log "FATAL ship failed twice ($src_rc); its local copies are kept"
    exit 7
  fi
}

ship_batch() {  # ship_batch <out> <shipdir> <b/nb> <corpus> <encode_s> <episode names...>; runs in the background
  local out=$1 sh=$2 tag=$3 corpus=$4 dt=$5; shift 5
  local acc=$STATE/episodes_${corpus}_$NN.jsonl n frames=0 ts=$SECONDS
  rm -rf "$sh"; mkdir -p "$sh"
  for n in "$@"; do
    ln "$out/${n}_latents.npy" "$sh/${n}_latents.npy.smpart"
    ln "$out/${n}_meta.npz" "$sh/${n}_meta.npz"
    frames=$((frames + $(grep "\"episode\": \"$n\"" "$out/episodes_00.jsonl" | head -1 | sed 's/.*"frames": \([0-9]*\).*/\1/')))
  done
  (cd "$sh" && md5sum -- * > "$sh/.sums_sm$GPU")
  local parts="" ok="" try
  for n in "$@"; do parts="$parts ${n}_latents.npy.smpart"; done
  for try in 1 2; do
    rsync -a -e "$SSH" "$sh/" "$R:$DST/" >> "$sh.rsync.log" 2>&1 \
      && ok=$(remote "cd $DST && md5sum -c --quiet .sums_sm$GPU && for f in $parts; do \
           t=\${f%.smpart}; if [ -e \$t ]; then rm -f \$f; echo DUP \$t; else mv \$f \$t; fi; done; \
           rm -f .sums_sm$GPU; echo SHIP_OK")
    grep -q SHIP_OK <<< "$ok" && break
    log "ship attempt $try failed for batch $tag: $(tail -2 "$sh.rsync.log" | tr '\n' ' ') $ok"
    [ "$try" = 1 ] && sleep 120
  done
  if ! grep -q SHIP_OK <<< "$ok"; then echo 7 > "$sh.rc"; return 7; fi
  for n in "$@"; do
    grep -q "DUP ${n}_latents.npy" <<< "$ok" && { log "dup $n: Spiderman wrote it first, kept theirs"; continue; }
    grep "\"episode\": \"$n\"" "$out/episodes_00.jsonl" | head -1 >> "$acc"
    echo "$n" >> "$W/nship_$corpus"
  done
  [ -s "$acc" ] && rsync -a -e "$SSH" "$acc" "$R:$DST/episodes_$NN.jsonl" >> "$sh.rsync.log" 2>&1
  if [ ! -e "$W/meta_sent_$corpus" ] && [ -s "$out/encode_meta_00.json" ]; then
    patch_meta "$out/encode_meta_00.json" \
      && rsync -a -e "$SSH" "$out/encode_meta_00.json" "$R:$DST/encode_meta_$NN.json" >> "$sh.rsync.log" 2>&1 \
      && rsync -a --ignore-existing -e "$SSH" "$out/canonical_controls.json" "$R:$DST/canonical_controls.json" >> "$sh.rsync.log" 2>&1 \
      && touch "$W/meta_sent_$corpus"
  fi
  log "batch $tag $corpus shipped=$# frames=$frames encode_s=$dt fps=$(( dt > 0 ? frames / dt : 0 )) ship_s=$((SECONDS - ts)) eps=$*"
  rm -rf "$out" "$sh"
  echo 0 > "$sh.rc"
}

encode_corpus() {
  local corpus=$1
  spec "$corpus"
  local A=${IDS%:*} Bnd=${IDS#*:} mine=() id
  for ((id = A; id < Bnd; id++)); do
    (( (id - A) % N == I )) && mine+=("$id")
  done
  mapfile -t todo < <(not_done "${mine[@]}")
  log "corpus=$corpus ids=$IDS mine=${#mine[@]} todo=${#todo[@]} src=$SRC dst=$DST k=$K"
  [ ${#todo[@]} -eq 0 ] && { log "CORPUS_DONE $corpus (nothing to do)"; return 0; }
  if [ "$DRY" = 1 ]; then
    echo "DRY first batch: ${todo[*]:0:$K}"
    echo "DRY encoder: $PY $REPO/encode_parquet.py --in-dir $W/in0 --out-dir $W/out --every-tic --stride 4" \
         "--canonical $CANON --batch-size 64 --decode-threads $THREADS --decode-workers 0 --device cuda:$GPU" \
         "--dtype bf16 --cache-dir $HUB --decode-check 16 ${VAE_FLAGS[*]}"
    echo "DRY ship to $R:$DST as ep_*_latents.npy + ep_*_meta.npz, summary episodes_$NN.jsonl, meta encode_meta_$NN.json"
    return 0
  fi
  remote "mkdir -p $DST" || { log "FATAL cannot reach Spiderman"; exit 5; }
  rm -f "$W/meta_sent_$corpus" "$W/nship_$corpus"
  local nb=$(( (${#todo[@]} + K - 1) / K )) b slot=0 fpid="" sp="" src_rc="" fails=0
  fetch "$W/in0" "${todo[@]:0:$K}"
  for ((b = 0; b < nb; b++)); do
    if [ -e "$W/STOP" ]; then
      log "STOP file found, exiting before batch $b of $corpus"
      wait_ship; [ -n "$fpid" ] && wait "$fpid"; exit 0
    fi
    local free
    free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
    if [ "$free" -lt "$MIN_FREE_GB" ]; then
      log "FATAL disk ${free}G free < ${MIN_FREE_GB}G"
      wait_ship; [ -n "$fpid" ] && wait "$fpid"; exit 6
    fi
    local IN=$W/in$slot NEXT=$W/in$((1 - slot)) OUT=$W/out$slot
    [ -n "$fpid" ] && wait "$fpid"
    fpid=""
    if [ "$(cat "$IN.rc")" != 0 ]; then
      log "fetch rc=$(cat "$IN.rc") for batch $b: $(tail -2 "$IN.rsync.log" | tr '\n' ' ') -- retrying once in 120 s"
      sleep 120
      local ids_again=()
      mapfile -t ids_again < <(sed -E 's/^ep_0*([0-9]+)\.parquet$/\1/' "$IN.list")
      fetch "$IN" "${ids_again[@]}"
      [ "$(cat "$IN.rc")" != 0 ] && log "fetch failed twice (rc=$(cat "$IN.rc")); encoding what arrived"
    fi
    # prefetch the next batch while this one encodes
    if (( b + 1 < nb )); then
      fetch "$NEXT" "${todo[@]:$(( (b + 1) * K )):$K}" &
      fpid=$!
    fi
    # a Spiderman shard may have finished some of these since the listing
    local batch=() p
    for p in "$IN"/ep_*.parquet; do [ -e "$p" ] && batch+=("$(basename "$p" .parquet | sed 's/^ep_0*//;s/^$/0/')"); done
    local still=()
    [ ${#batch[@]} -gt 0 ] && mapfile -t still < <(not_done "${batch[@]}")
    for id in "${batch[@]}"; do
      printf '%s\n' "${still[@]}" | grep -qx "$id" || { rm -f "$IN/$(name "$id").parquet"; log "skip $(name "$id"): already on Spiderman"; }
    done
    # every slot flip waits for the ship in flight, so the batch after next never clears an OUT still shipping
    if [ ${#still[@]} -eq 0 ]; then rm -rf "$IN"; wait_ship; slot=$((1 - slot)); continue; fi
    # OUT was last shipped two batches ago, and that ship was waited for before the previous one began
    rm -rf "$OUT"; mkdir -p "$OUT"
    local t0=$SECONDS
    (cd "$REPO" && nice -n 5 "$PY" "$REPO/encode_parquet.py" --in-dir "$IN" --out-dir "$OUT" --every-tic --stride 4 \
       --canonical "$CANON" --batch-size 64 --decode-threads "$THREADS" --decode-workers 0 \
       --device "cuda:$GPU" --dtype bf16 --cache-dir "$HUB" --decode-check 16 "${VAE_FLAGS[@]}") >> "$ELOG" 2>&1
    local rc=$? dt=$((SECONDS - t0))
    rm -rf "$IN"
    [ $rc -ne 0 ] && log "encoder exit $rc on batch $b ($(printf '%s ' "${still[@]}")); see $ELOG"
    # what the encoder completed: latents, sidecar and summary line all present
    local complete=() n
    for id in "${still[@]}"; do
      n=$(name "$id")
      if [ -s "$OUT/${n}_latents.npy" ] && [ -s "$OUT/${n}_meta.npz" ] \
         && grep -q "\"episode\": \"$n\"" "$OUT/episodes_00.jsonl" 2>/dev/null; then
        complete+=("$n")
      else
        log "encode incomplete for $n, not shipped"
      fi
    done
    if [ ${#complete[@]} -eq 0 ]; then
      # an encoder that fails every batch (OOM, a broken env) must not churn through the corpus
      fails=$((fails + 1))
      [ $fails -ge 3 ] && { log "FATAL three consecutive batches completed nothing; last encoder exit $rc"; wait_ship; exit 8; }
      wait_ship; slot=$((1 - slot)); continue
    fi
    fails=0
    # one ship in flight at a time: it overlaps the next batch's encode, never another ship
    wait_ship
    src_rc=$W/ship$slot.rc
    rm -f "$src_rc"
    ship_batch "$OUT" "$W/ship$slot" "$((b + 1))/$nb" "$corpus" "$dt" "${complete[@]}" &
    sp=$!
    slot=$((1 - slot))
  done
  wait_ship
  [ -n "$fpid" ] && wait "$fpid"
  local nship; nship=$(cat "$W/nship_$corpus" 2>/dev/null | wc -l)
  # CORPUS_DONE only when every episode this shard owned is now on Spiderman
  local left; left=$(not_done "${todo[@]}" | wc -l)
  if [ "$left" -eq 0 ]; then log "CORPUS_DONE $corpus shipped=$nship todo=${#todo[@]}"
  else log "CORPUS_INCOMPLETE $corpus shipped=$nship todo=${#todo[@]} missing=$left (relaunch to retry)"; fi
}

mkdir -p "$W" "$STATE" "$(dirname "$LOG")"
exec 9> "$W/.lock"
flock -n 9 || { echo "another worker holds $W/.lock" >&2; exit 3; }
rm -rf "$W"/in0 "$W"/in1 "$W"/out0 "$W"/out1 "$W"/ship0 "$W"/ship1 "$W"/*.rc "$W"/STOP
[ "$DRY" = 1 ] || [ -x "$PY" ] || { echo "no python at $PY" >&2; exit 2; }
log "START corpora=\"$CORPORA\" k=$K threads=$THREADS repo=$REPO git=$(head -c 12 "$B/repo_git.txt" 2>/dev/null) dst_root=$DST_ROOT dry=$DRY"
T0=$SECONDS
for c in $CORPORA; do encode_corpus "$c"; done
log "SHARD_DONE corpora=\"$CORPORA\" wall_s=$((SECONDS - T0))"
