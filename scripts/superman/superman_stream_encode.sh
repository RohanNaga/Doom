#!/bin/bash
# Streamed per-tic encode of the dense corpus on one Superman A4000, in either latent space.
#
#   usage: [VAE=sd35|sd15] [QUEUE_DIR=..] [K=4] [THREADS=6] [DRY=1] [LIST=1] [DST_ROOT=..]
#          [REMOTE_MIN_FREE_GB=0] [TRAIN_IDS=..] [VAL_IDS=..] [TEST_IDS=..] [UNSEEN_IDS=..] \
#          superman_stream_encode.sh <gpu> <i/n> <train|val|test|unseen|evals|all[,...]>
#
# The raw episodes live on Spiderman and Superman has no room for them, so each worker pulls K
# parquets at a time into /home/rohan/Doom/stream/<gpu>/, runs encode_parquet.py on exactly those
# files with the arguments `encode_nexttic.sh` gives it for the same VAE, and pushes the latents and
# sidecars into the directory that launcher would have written. The next batch is pulled while the
# current one encodes, and each batch ships while the next one encodes, so the card never waits on
# the network.
#
#   corpus  source on Spiderman           ids         destination on Spiderman (SUF: _sd35 or empty)
#   train   raw_arnold_dense/arenas       0:2000      latents_arnold_dense_pertic${SUF}/arenas
#   val     raw_arnold_dense/arenas       6000:6100   latents_arnold_dense_pertic_eval${SUF}/val
#   test    raw_arnold_dense/arenas       7000:7100   latents_arnold_dense_pertic_eval${SUF}/test
#   unseen  raw_arnold_dense/arenas_678   60:120      latents_arnold_dense_pertic_eval${SUF}/arenas_678
#
# VAE=sd35 is SD 3.5 Medium's autoencoder, 16 channels, scale 1.5305, shift 0.0609; VAE=sd15 is the
# default sd-vae-ft-mse, 4 channels, scale 0.18215. Unseen is 60:120, not the launcher's old 0:60
# default: 51 of ids 0:60 are worker-first episodes in which Arnold's weapon switches execute
# (review of Sep 22, finding H1).
#
# Two ways to split the work. Static: `<i/n>` takes ids[i::n] of each range, so shards of one n are
# disjoint. Queue: with QUEUE_DIR set, every worker pops K ids at a time, under flock, from
# $QUEUE_DIR/<vae>_<corpus>.txt (one id per line, written by the orchestrator from what Spiderman
# lacks), so workers can join or leave at any time and the ids stay disjoint; `<i/n>` is ignored.
# Either way an episode whose latents already exist on Spiderman is skipped, and the batch is
# re-checked just before it encodes. Ids a killed worker had popped are not lost for good: the
# orchestrator re-lists what is missing and queues it again.
#
# Batch size 64 and bf16 are not free choices: cuDNN picks its algorithm from the batch shape under
# autocast, so the same frame encodes differently at another batch size (encode_nexttic.sh).
#
# A shipped episode is verified before it becomes visible: the latents travel under a `.smpart`
# name, which `ep_*_latents.npy` globs do not match, their md5 and the sidecar's are checked on
# Spiderman, and only then is the part renamed. The summary lines go to `episodes_NN.jsonl` and the
# run record to `encode_meta_NN.json` with NN = 10 + i (static) or 17 - gpu (queue, the same number
# GPUs 7..2 had as static shards 0..5); Spiderman's own shards own 00..09.
#
# LIST=1 prints the ids of each corpus whose latents Spiderman lacks, one per line, and exits.
# REMOTE_MIN_FREE_GB stops the worker before a batch when Spiderman's /sata2/data has less free.
#
# Stop one worker after its current batch with `touch /home/rohan/Doom/stream/<gpu>/STOP`, or at
# once with `tmux kill-session -t enc-<vae>-<gpu>`; either way a relaunch resumes.
set -u
GPU=${1:?gpu}
SHARD=${2:?shard i/n}
CORPUS=${3:-train}
I=${SHARD%/*}
N=${SHARD#*/}
[ "$I" -ge 0 ] 2>/dev/null && [ "$I" -lt "$N" ] || { echo "bad shard $SHARD" >&2; exit 2; }
VAE=${VAE:-sd35}
QUEUE_DIR=${QUEUE_DIR:-}
K=${K:-4}                       # episodes per encoder process: amortises torch import and VAE load
THREADS=${THREADS:-6}           # PNG decode threads; 48 cores over 8 workers. Output bytes do not depend on it
DRY=${DRY:-0}
LIST=${LIST:-0}
MIN_FREE_GB=${MIN_FREE_GB:-60}
REMOTE_MIN_FREE_GB=${REMOTE_MIN_FREE_GB:-0}
B=/home/rohan/Doom/sd35enc
REPO=$B/repo
PY=${PY:-$HOME/miniconda3/envs/doomenc/bin/python}
CANON=$B/canonical_controls.json
HUB=$B/hf/hub
W=/home/rohan/Doom/stream/$GPU
STATE=$B/state
LOG=/home/rohan/Doom/logs/enc-$VAE-$GPU.log
ELOG=/home/rohan/Doom/logs/enc-$VAE-$GPU.encoder.log
R=rnagabhi@128.2.204.110
SD=/sata2/data/rnagabhi/doom
DST_ROOT=${DST_ROOT:-$SD}
if [ -n "$QUEUE_DIR" ]; then NN=$(printf %02d $((17 - GPU))); else NN=$(printf %02d $((10 + I))); fi
# one TCP connection per worker: every ssh and rsync below rides this worker's control socket
SOCK=$HOME/.ssh/cm/sm-enc-$GPU
SSH="ssh -i $HOME/.ssh/id_ed25519_doom -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=60 \
 -o ServerAliveCountMax=5 -o ControlMaster=auto -o ControlPath=$SOCK -o ControlPersist=12h"
case $VAE in
  sd35) VAE_FLAGS=(--vae-id stabilityai/stable-diffusion-3.5-medium --vae-subfolder vae --latent-channels 16
                   --scaling-factor 1.5305 --shift-factor 0.0609)
        SUF=_sd35 ;;
  sd15) VAE_FLAGS=(); SUF="" ;;      # encode_parquet's default: sd-vae-ft-mse, 4 channels, 0.18215
  *) echo "unknown VAE=$VAE (sd35|sd15)" >&2; exit 2 ;;
esac
# cuda:K must be nvidia-smi's K (the CUDA default order is fastest-first). HF_HOME puts the default
# cache, which is where the sd-vae-ft-mse load looks, beside the SD 3.5 one; both are offline copies
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HOME=$B/hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

SPLIT="shard=$SHARD"; [ -n "$QUEUE_DIR" ] && SPLIT=queue
log() { echo "$(date -Iseconds) gpu=$GPU vae=$VAE $SPLIT $*" | tee -a "$LOG"; }
remote() { $SSH $R "$@"; }

spec() {        # sets SRC, IDS, DST for one corpus
  case $1 in
    train)  SRC=$SD/raw_arnold_dense/arenas;     IDS=${TRAIN_IDS:-0:2000};  DST=$DST_ROOT/latents_arnold_dense_pertic$SUF/arenas ;;
    val)    SRC=$SD/raw_arnold_dense/arenas;     IDS=${VAL_IDS:-6000:6100}; DST=$DST_ROOT/latents_arnold_dense_pertic_eval$SUF/val ;;
    test)   SRC=$SD/raw_arnold_dense/arenas;     IDS=${TEST_IDS:-7000:7100}; DST=$DST_ROOT/latents_arnold_dense_pertic_eval$SUF/test ;;
    unseen) SRC=$SD/raw_arnold_dense/arenas_678; IDS=${UNSEEN_IDS:-60:120}; DST=$DST_ROOT/latents_arnold_dense_pertic_eval$SUF/arenas_678 ;;
    *) echo "unknown corpus $1 (train|val|test|unseen|evals|all)" >&2; exit 2 ;;
  esac
}

# a comma list runs its corpora in order, e.g. evals,train puts the held-out sets first
CORPORA=""
for c in ${CORPUS//,/ }; do
  case $c in
    evals) CORPORA="$CORPORA val test unseen" ;;
    all)   CORPORA="$CORPORA train val test unseen" ;;
    *)     spec "$c"; CORPORA="$CORPORA $c" ;;
  esac
done

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
  # an episode uploaded to the Hub and then deleted on Spiderman (ids 2000:6000) is listed in the
  # directory's uploaded_manifest.jsonl, and is done just as much as one whose file is still there
  have=$(remote "cd $DST 2>/dev/null && { ls $names 2>/dev/null; grep -ho 'ep_[0-9]*_latents.npy' uploaded_manifest.jsonl 2>/dev/null; }") || true
  for id in "$@"; do
    grep -qx "$(name "$id")_latents.npy" <<< "$have" || echo "$id"
  done
}

missing_in_range() {    # every id of $IDS whose latents Spiderman lacks, from one directory listing
  local have
  have=$(remote "ls $DST 2>/dev/null | grep '_latents.npy\$'; grep -ho 'ep_[0-9]*_latents.npy' $DST/uploaded_manifest.jsonl 2>/dev/null") || true
  local A=${IDS%:*} Bnd=${IDS#*:} id
  for ((id = A; id < Bnd; id++)); do
    grep -qx "$(name "$id")_latents.npy" <<< "$have" || echo "$id"
  done
}

remote_space_ok() {     # REMOTE_MIN_FREE_GB guard on Spiderman's data disk
  [ "$REMOTE_MIN_FREE_GB" -gt 0 ] || return 0
  local f
  f=$(remote "df --output=avail -BG /sata2/data | tail -1" | tr -dc 0-9)
  if [ -n "$f" ] && [ "$f" -lt "$REMOTE_MIN_FREE_GB" ]; then
    log "FATAL Spiderman /sata2/data has ${f}G free < ${REMOTE_MIN_FREE_GB}G"
    return 1
  fi
}

patch_meta() {  # record where and from which code this shard's latents were made
  "$PY" - "$1" "$B/repo_git.txt" "$SHARD" "$GPU" "${QUEUE_DIR:-}" \
    "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "$GPU" 2>/dev/null)" <<'P'
import json, platform, sys
import diffusers, numpy, torch
path, gitfile, shard, gpu, queue, driver = sys.argv[1:7]
m = json.load(open(path))
head = open(gitfile).read().split()[0]
if m.get("git") in (None, "", "?"):
    m["git"] = head
m["encoded_on"] = {
    "host": "superman", "gpu": "NVIDIA RTX A4000", "gpu_index": int(gpu),
    "wrapper": "scripts/superman/superman_stream_encode.sh",
    "wrapper_split": f"queue {queue}" if queue else f"static shard {shard}",
    "git_source": "HEAD of Spiderman's $D/repo when it was copied; Superman's copy has no .git",
    "note": "args.in_dir and args.out_dir are this worker's local streaming directories; each "
            "encoder process saw one batch of episodes, and the summary lines of every batch are "
            "gathered into the matching episodes_NN.jsonl",
    # the encoder environment: bf16 latents differ at rounding level between environments
    "env": {"conda_env": "doomenc", "python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "diffusers": diffusers.__version__, "numpy": numpy.__version__, "nvidia_driver": driver}}
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

ship_batch() {  # ship_batch <out> <shipdir> <tag> <corpus> <encode_s> <episode names...>; runs in the background
  local out=$1 sh=$2 tag=$3 corpus=$4 dt=$5; shift 5
  local acc=$STATE/episodes_${VAE}_${corpus}_$NN.jsonl n frames=0 ts=$SECONDS
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
    grep -q "DUP ${n}_latents.npy" <<< "$ok" && { log "dup $n: already on Spiderman, kept that copy"; continue; }
    grep "\"episode\": \"$n\"" "$out/episodes_00.jsonl" | head -1 >> "$acc"
    echo "$n" >> "$W/nship_${VAE}_$corpus"
  done
  [ -s "$acc" ] && rsync -a -e "$SSH" "$acc" "$R:$DST/episodes_$NN.jsonl" >> "$sh.rsync.log" 2>&1
  if [ ! -e "$W/meta_sent_${VAE}_$corpus" ] && [ -s "$out/encode_meta_00.json" ]; then
    patch_meta "$out/encode_meta_00.json" \
      && rsync -a -e "$SSH" "$out/encode_meta_00.json" "$R:$DST/encode_meta_$NN.json" >> "$sh.rsync.log" 2>&1 \
      && rsync -a --ignore-existing -e "$SSH" "$out/canonical_controls.json" "$R:$DST/canonical_controls.json" >> "$sh.rsync.log" 2>&1 \
      && touch "$W/meta_sent_${VAE}_$corpus"
  fi
  log "batch $tag $corpus shipped=$# frames=$frames encode_s=$dt fps=$(( dt > 0 ? frames / dt : 0 )) ship_s=$((SECONDS - ts)) eps=$*"
  rm -rf "$out" "$sh"
  echo 0 > "$sh.rc"
}

next_batch() {  # NB = the next <= K ids: the static list's next slice, or popped from the shared queue
  NB=()
  if [ -n "$q" ]; then
    mapfile -t NB < <(flock "$q.lock" sh -c "head -n $K '$q'; sed -i '1,${K}d' '$q'")
  else
    NB=("${todo[@]:$pos:$K}")
    pos=$((pos + K))
  fi
}

encode_corpus() {
  local corpus=$1
  spec "$corpus"
  local q="" todo=() pos=0 nb="?"
  if [ -n "$QUEUE_DIR" ]; then
    q=$QUEUE_DIR/${VAE}_${corpus}.txt
    touch "$q"
    log "corpus=$corpus queue=$q left=$(wc -l < "$q") src=$SRC dst=$DST k=$K"
  else
    local A=${IDS%:*} Bnd=${IDS#*:} mine=() id
    for ((id = A; id < Bnd; id++)); do
      (( (id - A) % N == I )) && mine+=("$id")
    done
    mapfile -t todo < <(not_done "${mine[@]}")
    nb=$(( (${#todo[@]} + K - 1) / K ))
    log "corpus=$corpus ids=$IDS mine=${#mine[@]} todo=${#todo[@]} src=$SRC dst=$DST k=$K"
    [ ${#todo[@]} -eq 0 ] && { log "CORPUS_DONE $corpus (nothing to do)"; return 0; }
  fi
  if [ "$DRY" = 1 ]; then
    if [ -n "$q" ]; then echo "DRY first batch (not popped): $(head -n "$K" "$q" | tr '\n' ' ')"
    else echo "DRY first batch: ${todo[*]:0:$K}"; fi
    echo "DRY encoder: $PY $REPO/encode_parquet.py --in-dir $W/in0 --out-dir $W/out0 --every-tic --stride 4" \
         "--canonical $CANON --batch-size 64 --decode-threads $THREADS --decode-workers 0 --device cuda:$GPU" \
         "--dtype bf16 --cache-dir $HUB --decode-check 16 ${VAE_FLAGS[*]}"
    echo "DRY ship to $R:$DST as ep_*_latents.npy + ep_*_meta.npz, summary episodes_$NN.jsonl, meta encode_meta_$NN.json"
    return 0
  fi
  remote "mkdir -p $DST" || { log "FATAL cannot reach Spiderman"; exit 5; }
  rm -f "$W/meta_sent_${VAE}_$corpus" "$W/nship_${VAE}_$corpus"
  local b slot=0 fpid="" sp="" src_rc="" fails=0 cur=() nxt=()
  next_batch; cur=("${NB[@]}")
  [ ${#cur[@]} -gt 0 ] && fetch "$W/in0" "${cur[@]}"
  for ((b = 0; ; b++)); do
    [ "$b" -gt 0 ] && cur=("${nxt[@]}")
    [ ${#cur[@]} -eq 0 ] && break
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
    remote_space_ok || { wait_ship; [ -n "$fpid" ] && wait "$fpid"; exit 9; }
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
    next_batch; nxt=("${NB[@]}")
    if [ ${#nxt[@]} -gt 0 ]; then
      fetch "$NEXT" "${nxt[@]}" &
      fpid=$!
    fi
    local tag="$((b + 1))/$nb"
    [ -n "$q" ] && tag="$((b + 1)) queue_left=$(wc -l < "$q")"
    # another writer may have finished some of these since they were listed
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
    ship_batch "$OUT" "$W/ship$slot" "$tag" "$corpus" "$dt" "${complete[@]}" &
    sp=$!
    slot=$((1 - slot))
  done
  wait_ship
  [ -n "$fpid" ] && wait "$fpid"
  local nship; nship=$(cat "$W/nship_${VAE}_$corpus" 2>/dev/null | wc -l)
  if [ -n "$q" ]; then
    log "QUEUE_EMPTY $corpus shipped=$nship"
    return 0
  fi
  # CORPUS_DONE only when every episode this shard owned is now on Spiderman
  local left; left=$(not_done "${todo[@]}" | wc -l)
  if [ "$left" -eq 0 ]; then log "CORPUS_DONE $corpus shipped=$nship todo=${#todo[@]}"
  else log "CORPUS_INCOMPLETE $corpus shipped=$nship todo=${#todo[@]} missing=$left (relaunch to retry)"; fi
}

if [ "$LIST" = 1 ]; then
  for c in $CORPORA; do spec "$c"; missing_in_range; done
  exit 0
fi
mkdir -p "$W" "$STATE" "$(dirname "$LOG")"
exec 9> "$W/.lock"
flock -n 9 || { echo "another worker holds $W/.lock" >&2; exit 3; }
rm -rf "$W"/in0 "$W"/in1 "$W"/out0 "$W"/out1 "$W"/ship0 "$W"/ship1 "$W"/*.rc "$W"/STOP
[ "$DRY" = 1 ] || [ -x "$PY" ] || { echo "no python at $PY" >&2; exit 2; }
log "START corpora=\"$CORPORA\" k=$K threads=$THREADS nn=$NN repo=$REPO git=$(head -c 12 "$B/repo_git.txt" 2>/dev/null) dst_root=$DST_ROOT dry=$DRY"
T0=$SECONDS
for c in $CORPORA; do encode_corpus "$c"; done
log "SHARD_DONE corpora=\"$CORPORA\" wall_s=$((SECONDS - T0))"
