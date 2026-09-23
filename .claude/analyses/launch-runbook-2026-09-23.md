# Launch runbook, Wed Sep 23 2026 (next-tic rows on Spiderman)

Written 23:45 EDT Sep 22 so any session can execute the morning without re-deriving it. `$D=/sata2/data/rnagabhi/doom`, repo `$D/repo` on main, Python `~/miniconda3/envs/doom/bin/python` (4-channel) and `~/wanenc/bin/python` (SD 3.5). Server clock is UTC.

## Preconditions (check, do not assume)
- 4-channel training latents: 2,000 files in `$D/latents_arnold_dense_pertic/arenas`; sidecars normalised (`--normalize-sidecars` run by the steward after both shards finished; `normalize_final.txt`), audit `audit_train.txt` exit 0.
- 4-channel eval latents: val 100, test 100, `arenas_678` 60 for ids 60:120 (`enc_678_60_120.log` ends `ENC678B_DONE exit 0`); split files `split_val.json`, `split_test.json`, `split_arenas_678.json` regenerated for the new unseen ids by the review-fixes launcher.
- 16-channel latents (Superman shards + Spiderman GPU 1 evals): `$D/latents_arnold_dense_pertic_sd35/arenas` 2,000 files, `_eval_sd35/{val,test,arenas_678}` 100/100/60; ep-6000 cross-host diff in `$D/logs/sd35_superman_vs_spiderman.log`.
- Review fixes merged on main and pulled on the server (`git -C $D/repo log --oneline -1`), BUT never `git pull` while an encoder of that repo is running: encoders on Spiderman (`enc-sd35-gpu1`, `enc-678b`, `enc-train1`) must have exited first, or run the gates from a second checkout `$D/repo_launch` (clone main there) and launch from it.
- GPUs: Spiderman GPU 1 free by 07:30 (U-Net), GPU 2 free when `enc-train1` ends (SD 3.5). Superman: seven cards until 09:00, six after.

## Gates, per latent space (stop at the first failure and fix; do not launch on a failed gate)
All of the gates below run in order from one command, `scripts/cluster/gates.sh`, which stops at the first failure and writes `$D/GATES_CERT.json` only at GATES_GO. On Spiderman (updated Sep 23 after Astra's third review; `repo_launch` is a clean clone of main, used because an encoder still runs from `$D/repo`; the gates must run FROM the checkout they certify, so `cd` there first):

```
cd /sata2/data/rnagabhi/doom/repo_launch && DOOM_ROOT=/sata2/data/rnagabhi/doom \
  RUN_REPO=/sata2/data/rnagabhi/doom/repo_launch UNET_GPU=1 SD35_GPU=2 LAUNCH_STEPS=400000 \
  MB_UNET=32 MB_SD35=32 WORKERS=12 PY_UNET=$HOME/miniconda3/envs/doom/bin/python \
  PY_SD35=$HOME/wanenc/bin/python bash scripts/cluster/gates.sh
```

Staggered this morning: prefix `VAES=sd15 SMOKE_BBS=unet` as soon as the 4-channel corpus is complete, and `VAES=sd35 SMOKE_BBS=sd35` when the 16-channel one is. Each run revokes and rewrites only its own backbone's certificate entry and keeps its own results file (`$D/logs/gates_results_<run>.jsonl`), so the second run leaves the training U-Net's entry, and its resumes, alone. `DRY=1` first prints every command without running anything. PY_UNET runs every sd15 and U-Net command, PY_SD35 every sd35 and SD 3.5 command (including the latent alignment that re-encodes with the SD 3.5 autoencoder). The fit, smoke, resume and certificate all resolve from these same production settings plus any EXTRA, so do not change them between the gates and the launch. At GATES_GO the script prints one `GATES_LAUNCH <backbone>:` line per certified backbone; that line is the launch (step 6), already checked to resolve to the certified command in an empty environment.

1. Sidecar audit vs raw: `python check_action_alignment.py --latents-dir <train dir> --audit-parquet-dir $D/raw_arnold_dense/arenas --audit-only --episodes 100 --audit-rows 100000 --canonical $D/latents_arnold_dense_pertic/canonical_controls.json --seed 0` (zero mismatches; exit 0).
2. Latent alignment: `python check_latent_alignment.py --latents-dir <train dir> --parquet-dir $D/raw_arnold_dense/arenas --space <sd15|sd35> --contract-peer <val dir> --device cuda:<free gpu> ...` (added by review-fixes; per shard; unshifted must beat the -4/-1/+1/+4 controls; train and val must share the space's own contract: sd-vae-ft-mse, 0.18215, no shift, 4 channels, or SD 3.5's `vae`, 1.5305, 0.0609, 16 channels).
3. Alignment gate on val: `python check_action_alignment.py --latents-dir <val dir> --episodes 100 --canonical ... --seed 0` (exit 0 = aligned at shift 0; nonzero blocks).
4. Fit check: `FIT=20 bash scripts/spiderman/launch_nexttic.sh <gpu> <unet|sd35>` (peak memory, updates/s; MB=32, no accumulation; sd35 with `--grad-ckpt`).
5. Smoke: gates.sh gate 4, 300 steps through `launch_nexttic.sh` with GATE_RUN=1 into `$D/results_smoke/<backbone>` (must write a recovery checkpoint and a snapshot), then `eval_tf.py --tic-stride 1 --horizon-tics 1` readback on 64 val windows (live and EMA), then a 10-update `--resume` of the smoke run.
6. Launch: paste the two `GATES_LAUNCH` lines gates.sh printed (each is `cd $RUN_REPO && DOOM_ROOT=.. RUN_REPO=.. PY_UNET|PY_SD35=.. MB=32 WORKERS=12 STEPS=400000 TRAIN_IDS=0:2000 VAL_IDS=6000:6100 [EXTRA=..] bash scripts/spiderman/launch_nexttic.sh <1|2> <unet|sd35>`). A bare `launch_nexttic.sh 1 unet` resolves a different command (default interpreter, `$D/repo`) and the certificate check refuses it. Record the commit hash, the commands and the gate outputs in `RESEARCH_CONTEXT.md`.
7. Steward: opus-monitor with the review's first-day list (section 7.2 of `docs/REVIEW_2026-09-22.md`): val loss every 1k against the planning bands, grad norm and clip fraction, skips, control-MLP update ratios, updates/s, data wait, memory, disk; at 10k and 20k the 512-window val read (live and EMA from the same file, raw floor and reconstruction beside), the control-sensitivity probe and the directional check; stop rules as in the audit.

## Mirrors
- Latents to `RohanNaga/doom-dense-arnold-latents` (`sd15/arenas`, `sd35/...`) as each set completes (from Spiderman, `hf upload`).
- Checkpoints and snapshots of both runs to `RohanNaga/doomdit-nexttic` at each 10k snapshot (a steward task).

## If something fails
- Encoder mismatch between hosts: keep the training corpus from one environment; the ep-6000 diff decides whether mixing is allowed.
- A gate fails on the 16-channel set only: launch the U-Net first; SD 3.5 waits for the fix.
- Resume identity refusal: never override the corpus manifest; fix the corpus or start a new run directory.
