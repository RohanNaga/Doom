# Rented node: zero to the two next-tic runs

Five scripts take a freshly rented GPU box from nothing to `040-unet-nexttic` and
`042-sd35-nexttic` training, using only the public repo `github.com/RohanNaga/Doom` and the public
dataset `RohanNaga/doom-dense-arnold`. Nothing here needs Spiderman, Superman, the CMU VPN or any
private artifact.

Every script takes `DOOM_ROOT` (default `/data/doom`, which must be the local NVMe) and `DRY=1`,
which prints every command it would run and touches nothing. Read the `DRY=1` output before each
real invocation; that is what it is for.

## What the node must have (confirm before renting)

These are assumptions this directory makes and cannot check for you:

| Assumption | Why it matters |
|---|---|
| Ubuntu 22.04 or 24.04, root or sudo, `git`, `tmux`, `curl` present | `launch_runs.sh` refuses without `tmux`; the rest is stock |
| NVIDIA driver **>= 550** | the pinned torch is a cu124 wheel; an older driver refuses to load it |
| Python **3.10 or 3.11** available as `python3.10` / `python3.11` | `requirements.txt` is pinned for those two; `PY_BIN=` overrides the search |
| Local NVMe mounted at `DOOM_ROOT` with **>= 1.5 TB** free | raw subset ~540 GB, 4-channel latents ~110 GB, 16-channel latents ~430 GB, checkpoints ~40 GB. The 1.081 TB figure is the pre-launch audit's; the margin is for caches and evaluation artifacts |
| **>= 32 physical cores** and **>= 200 GB RAM** | two loaders at 8-12 workers each, plus a recovery checkpoint that serializes the full fp32 model and optimizer on the host |
| 8x H100 80 GB, at least 2 free for the two rows | the rest are idle under this recipe; see "one card per row" below |

The `hf` client downloads over HTTPS; no private token is needed for either repo.

## The sequence

```bash
# 0. clone somewhere to get these scripts (the setup script then makes the real checkout)
git clone https://github.com/RohanNaga/Doom.git /tmp/doom-bootstrap
cd /tmp/doom-bootstrap

# 1. venv, pinned wheels, pinned checkout, driver and CUDA check, NODE.json
DOOM_ROOT=/data/doom COMMIT=<full sha of main> bash scripts/cluster/setup_node.sh

# from here on use the checkout the node just made
cd /data/doom/repo

# 2. the 2,260 episodes the runs need, md5-verified against the dataset's own manifests
DOOM_ROOT=/data/doom bash scripts/cluster/fetch_dataset.sh
#    FULL=1 adds the remaining ~8,740 episodes (roughly 2.1 TB); the workshop runs do not need them

# 3. both latent corpora: evaluation corpora on one card, then 2,000 training episodes on all eight
DOOM_ROOT=/data/doom GPUS=0,1,2,3,4,5,6,7 VAES=sd15,sd35 bash scripts/cluster/encode_all.sh

# 4. the launch gates, in order, stopping at the first failure
DOOM_ROOT=/data/doom UNET_GPU=0 SD35_GPU=1 bash scripts/cluster/gates.sh

# 5. launch, then watch
DOOM_ROOT=/data/doom UNET_GPU=0 SD35_GPU=1 bash scripts/cluster/launch_runs.sh
DOOM_ROOT=/data/doom bash scripts/cluster/status.sh
```

`gates.sh` must print `GATES_GO` before step 5. It runs, in order: the sidecar audit of both latent
spaces against the raw parquet, the yaw alignment gate on val 6000:6100 (exit 2 misaligned, exit 3
inconclusive, and **exit 3 is not approval**), a `FIT=20` fit check of each backbone through the
real launcher, a 300-step real-data smoke into a throwaway results directory, and an
`eval_tf.py --tic-stride 1` readback of that smoke's snapshot on 64 val windows.

## Expected durations

The only measured numbers we have are A6000 numbers. **Everything in the H100 column is an
estimate: 2x the A6000 rate, with a 1.5x to 3x range.** No next-tic run has been timed on any card,
and the 16-channel encoder has never been timed at all. Re-measure with `gates.sh` (the fit check
prints `steps_per_s` and `peak_mem_gb`) before planning a budget from these.

| Stage | A6000, measured | H100, estimate |
|---|---|---|
| fetch the 2,260-episode subset (~540 GB) | — | 1.5-6 h, network-bound, not GPU-bound |
| encode sd15 evaluation corpora (260 episodes, 1 card) | 5.5 card-h at 66 fps | ~2.8 h |
| encode sd15 training corpus (10.07M tics, 8 cards) | 42.4 card-h | ~2.7 h wall |
| encode sd35 corpora (16 channels, 8 cards) | never measured | ~4-8 h wall, budget more |
| gates (audit, alignment, 2 fit checks, 2 smokes, 2 readbacks) | — | 1-2 h |
| U-Net, 100k updates | 0.8-1.0 updates/s -> 28-35 h | 14-17 h |
| SD 3.5, 100k updates | 0.455 updates/s at 36.4 GB -> 61 h | ~31 h |
| U-Net, one pass over the corpus (314,688 updates) | 87-109 h | 44-55 h |

## One card per row

`launch_nexttic.sh` enforces micro-batch x cards = 32, because the recipe's global batch is 32 and
the fill-the-card rule forbids gradient accumulation. On an 80 GB H100 that makes **32 the largest
usable micro-batch**, and a second card for the same row would mean 16 on each, which fills
neither. So each row takes one card and six cards sit idle under this recipe. Spending them means
either more rows (a PixArt third row, a second seed, the evaluation passes) or a different global
batch, and the global batch is a recipe decision that is not ours to take here. `launch_runs.sh`
prints this as a note rather than quietly setting `ALLOW_ACCUM=1`.

## Stopping, resuming, copying back

```bash
# stop a row (never pkill -f: the tmux server carries the first session's command line in its own
# arguments, and a pattern match kills every session on the machine)
tmux kill-session -t train-unet-nexttic
tmux kill-session -t train-sd35-nexttic

# resume: the same command. The launcher picks up the newest recovery checkpoint in the run
# directory, refuses to start a second copy of a live session, and records the resume in
# $DOOM_ROOT/logs/resumes.log
DOOM_ROOT=/data/doom bash scripts/cluster/launch_runs.sh

# resume one row only
DOOM_ROOT=/data/doom ONLY=unet bash scripts/cluster/launch_runs.sh
```

Recovery checkpoints are written every 5k updates (`--keep-last 2`) and compact bf16 snapshots
every 10k, never pruned. A run stopped by hand therefore always leaves a `snap_*.pt` to evaluate.

Copying checkpoints back to Spiderman is a **manual step**, from a machine that can reach it (CMU
VPN, one connection, no retry loops):

```bash
# snapshots and logs only: the recovery checkpoints are fp32 and large
rsync -avP --include='snap_*.pt' --include='best.pt' --include='*.json' --include='*.jsonl' \
      --exclude='*' \
      /data/doom/results_spiderman/040-unet-nexttic/ \
      rnagabhi@128.2.204.110:/sata2/data/rnagabhi/doom/results_spiderman/040-unet-nexttic/
```

Do the same for `042-sd35-nexttic`. Pull rather than push if the rented node has no route to
Spiderman: run the `rsync` from Spiderman with the node as the source.

## Things that will bite

- **The canonical control table is the dataset's.** `encode_all.sh` passes
  `raw_arnold_dense/canonical_controls.json` to every shard and both latent spaces and refuses to
  start without it. `encode_nexttic.sh` would otherwise rebuild it per shard, scanning `action` and
  `buttons` of every episode (~10 GB per process); six shards doing that at once is what the
  Sep 20 2026 Spiderman host-memory outage looks like.
- **Pin `COMMIT` to a sha.** `launch_nexttic.sh` runs `git pull -q` of its own before launching. On
  a detached HEAD at a sha that is a no-op; on `main` it can move the code between the smoke gate
  and the launch.
- **The smoke touches production paths.** It trains into `$DOOM_ROOT/results_smoke/<backbone>`, but
  it goes through the real launcher, so it appends to `$DOOM_ROOT/logs/train_<run>.log` and
  `resumes.log` and creates the production results directory empty. The results directory itself
  stays clean of a 300-step `event=end` marker, which is what would confuse the evaluator.
- **`FIT=20` is a sizing estimate, not a certificate.** The pre-launch audit asks for 200 updates,
  to exercise the EMA cadence and the Adam state allocation. The 300-step smoke is the real-data
  evidence; read the fit number as memory and a rough rate only.
- **One venv for both rows.** On Spiderman the U-Net ran under diffusers 0.31 and SD 3.5 under
  0.40. Here both use the single pin in `requirements.txt`. If bit-comparability with the
  already-published SD 3.5 numbers matters, move the pin and rerun the gates.
