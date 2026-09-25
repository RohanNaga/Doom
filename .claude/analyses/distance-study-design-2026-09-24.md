# Distance study: does quality fall with distance from the training footage? (design, 2026-09-24)

Status: the study is go (RC 2026-09-24 22:20); Astra reviews this design on Sep 26. Prototype `distance_study.py`, tests `paper/fixtures/test_distance_study.py`. Maps: 30 distinct (training arenas 2 to 5 as held-out episodes; arenas 1 and 6 to 17; 13 curated campaign maps), about 370 episodes. Target sentence: "across N maps, gain over persistence falls with distance from the nearest training map (partial Spearman ρ, 95% CI)".

## 1. What transfers from the NVIDIA method

That method treats a driving test as a cloud of per-frame ego latents. It adds time as a coordinate, priced so a 15 s shift costs about 10 percent of latent change. It weights frames by latent motion, with the quietest half keeping a quarter of the weight, and compares tests by sliced Wasserstein. Validation: close pairs agree on 95 percent of verdicts, random pairs on 79, re-runs on 99.

**Transfers.** The cloud of frames, which needs no frame correspondence. Motion weighting, for a new reason: persistence scores still frames almost perfectly, so the model's gain over it sits in moving frames. Sliced Wasserstein, which takes weights exactly and scales to 10^5 points. The per-model rule, for space (c).

**Does not transfer.**
- *Alignment.* NVIDIA's two windows started at the same trigger. A map and the corpus share no clock, so time is dropped.
- *Symmetry.* The corpus mixes four maps, and Wasserstein to a mixture charges a seen map for not resembling the other three. In 1-D with training maps at 0, 2, 4 and 6, a held-out draw of map "0" has W₂² = 14 to the pool, while an unseen map at 3 has only 5. So the primary distance goes to the nearest training map (`test_the_nearest_training_map_does_not_rank_a_seen_map_beyond_an_unseen_one`).
- *Latent space.* The ego latent is a decision representation that discards appearance irrelevant to the plan; a VAE latent is a reconstruction code that keeps textures. VAE distance measures how different footage looks, not how different it is to the model.
- *Validation.* Here quality is the dependent variable, so validating on it would be circular. The distance gets outcome-free checks and is frozen before any per-map score is read.

## 2. The distance

**Primary.** `D(m) = min_k SW₂(μ_m, ν_k)` over training maps k ∈ {2, 3, 4, 5}.
- *Cloud:* each map is 4 episodes × 250 target-eligible frames (at least 32 tics into a life), the population the scored windows come from; 4 is the smallest per-map count (maps 1 and 9 to 15). Maps with more episodes average ten random 4-episode subsets. Sizes are equal because the finite-sample floor depends on n.
- *Features:* SD 1.x latents with the bottom padding rows removed (`encode_parquet.py:161`), 5×5-pooled to 192 dimensions (`pool_latents`).
- *Weights:* motion ||z_t − z_(t−1)|| at tic rate within each life (`latent_motion`). Weights are motion plus a closed-form floor giving the quietest half exactly 25 percent, per cloud (`motion_weights`).
- *p = 2:* W₂² is the least mean squared displacement between clouds, in the units of the latent MSE the model trains on.
- *Directions:* 1,000 from seed 0, shared by every distance (common random numbers). Measured: about 2 s per distance on a laptop CPU, under 2 percent spread across seeds.
- *No time coordinate:* windows are drawn uniformly and every episode lasts 150 s, so within-episode time is uniform on both sides. It would add only noise and a free price. The temporal structure the model uses, its 32-tic context, enters through the weights and space (c).

**Secondaries (pre-declared):** pooled distance, uniform weights (`quiet_share=0.5`), best mixture of training maps.

**Reference.** 50 seeded episodes per map from training ids 0:2000, 250 frames each: 12,500 per map, 50,000 in all. Stratified by map (equal, as in training) and by motion decile within each episode (25 frames each). That allocation is proportional, so it adds no bias, and it pins the loud tail that carries the weight. A disjoint second draw gives the train-versus-train floor.

**Validation before scoring.**
- (i) The four validation maps sit inside the floor band.
- (ii) Disjoint episode subsets of a map agree within 10 percent, and the seeded corpus's copies of maps 2 to 8 land where the dense corpus's do. Maps whose distance lies inside the floor band are exempt: there the distance is draw noise and no relative tolerance can hold.
- (iii) The two reference draws rank the maps with Spearman ≥ 0.95.
- (iv) Arenas 6 to 8 (same WAD) sit nearer than the campaign maps.
- (v) |Spearman(D, Kish n_eff)| < 0.4. Motion weights shrink n_eff and raise the floor (synthetic 1,000-frame clouds: 0.060 uniform against 0.084 weighted at n_eff 440); if the check fails, the uniform arm becomes primary.

Then commit the distances with their code hash.

**Feature spaces.**
- *(a) Pixels,* RGB at 20×15×3: model-free and decoder-free, one axis for all rows. It measures colour and layout shift. It is the most exposed to projection concentration: random directions in 900 dimensions mostly miss a low-dimensional difference.
- *(b) Pooled VAE latents* (primary): SD 1.x at 192 dimensions, SD 3.5 at 768. The encoder is fixed, so the axis belongs to no trained row. Kendall τ between the two encoders' rankings says whether distance is a property of the footage or of the encoder.
- *(c) The model's context features:* per row, the mean mid-block token for real context at a fixed low noise level. The ego-latent analogue, but partly circular, since failure can make features atypical. Exploratory, GPU forward passes only.

## 3. Outcomes, confounds, statistics

**Primary outcome:** per-map gain over persistence at one tic, Δ_m = mean(`psnr_raw` − `persist_psnr_raw`) from `eval_tf.py`. Per window this is the log persistence-to-model MSE ratio, so motion level cancels to first order. Secondaries: LPIPS gain, four-tic gain, then 64-tic rollout gain over copy-seed.

Raw PSNR misleads. In the first run, the U-Net's campaign-map PSNR (21.15) sat near its seen-map value (21.36). Over the decoded-copy baseline, though, it gained +0.23 dB on campaign maps, against +1.95 seen and +0.64 on arenas 16 to 17 (dossier 3.2).

**Confounds.**
- *Motion:* Δ, plus a partial correlation on persistence PSNR.
- *Windows:* 256 per map from `eval_tf.draw_windows`, identical across rows.
- *Deaths:* arenas 6 to 8 average 7.1 to 11.7 lives per episode, against 3.2 to 8.6 on training arenas. Report lives and valid-window fraction per map.
- *Decoder:* stock decoders only, so no decoder has seen an evaluation map. Add a decoder-free outcome (latent MSE ratio to copy-last), since the VAE ceiling varies with texture.
- *Control regime:* weapon selects execute only in a recorder's first episode (k = 0). Measure k with `transitions.button_width_report` and stratify.
- *Two clusters:* 13 campaign maps (another WAD) versus 17 arenas. A gap between clusters alone can inflate ρ, so report ρ within each.

**Map level (n = 30, primary).**
- *Correlation:* Spearman ρ(D, Δ), CI from a case bootstrap over maps (10,000 draws, episodes resampled within maps), permutation p-value.
- *Partial on motion:* rank residuals on persistence PSNR.
- *Leave-one-map-out:* the sign must hold in all 30 refits.
- *Distance error:* 200 episode redraws. If D's SD is under 10 percent of its spread across maps, attenuation is below 1 percent.
- *Power:* |ρ| ≥ 0.36 at n = 30, 0.49 within the 17 arenas and 0.56 within the 13 campaign maps.

**Episode level (n ≈ 370, secondary).** Episodes nest in maps, so report a within-map ρ (map-demeaned ranks, permutation within maps) and a mixed model Δ_e ~ D_e + persistence_e + lives_e + (1 | map).

**One primary test:** SD 1.x space, U-Net 200k EMA, one-tic PSNR gain, partial on motion. Everything else is secondary.

## 4. Alternatives

- **Fréchet distance:** assumes one Gaussian, but map frames are multimodal and a 768-dimensional covariance from 1,000 correlated frames is badly conditioned. Robustness check only.
- **MMD:** unbiased at small n, but depends on the kernel bandwidth and handles weights awkwardly.
- **Nearest-training-frame distance:** the most direct "seen this before" measure, and a good secondary if memorisation drives failure. It ignores density.
- **Classifier two-sample test:** AUC saturates near 1 for every other-WAD map, which erases the dose-response.

Sliced Wasserstein stays primary: a true metric with exact weights, no Gaussian or kernel choice, and per-direction error that does not grow with dimension, and it is Rohan's validated method. Pooling limits projection concentration; the nearest-map construction handles symmetry.

## 5. What would support the claim

**Supports** if the primary partial ρ is negative with a CI excluding zero, and the sign holds:
- in every leave-one-out refit;
- within both clusters;
- for every row;
- in pixel space, with map rankings agreeing between (a) and (b) (τ ≥ 0.6).

**Does not support** if any of these holds:
- only the cluster gap exists (seen versus unseen, restated);
- the relation vanishes under motion control;
- it appears only in raw PSNR;
- the distance fails validation;
- the rows disagree.

Never claimed: causation, or maps beyond these arenas and curated campaign maps.

**Figure (one column).** x = D, y = one-tic gain over persistence (dB), one marker per map, shape by group, colour by row, 95% bootstrap bars on both axes. A grey band marks the train-versus-train floor, a dashed line persistence, and each row's partial ρ is printed. An appendix table lists D, n_eff, lives and gains per map.

## 6. Implementation plan

1. **`distance_study.py`** (core committed) gains three subcommands, all CPU.
   - `clouds`: numpy mmap reads of `ep_*_latents.npy` and `_meta.npz`, cut into lives, with motion and stratified draws. Writes `results/distance_study/clouds/<space>/<set>.npz`, about 180,000 frames (136 MB for SD 1.x, 545 MB for SD 3.5).
   - `splits`: per-map split files.
   - `distances`: writes `distances_<space>.json` and `per_episode_<space>.csv`.

   Run on Spiderman beside the roughly 10 GB of training latents, in one tmux session. Timing: about 30 minutes for map distances (ten subsets per map), 40 for episodes, about 2 hours for the bootstrap.
2. **`scripts/spiderman/score_distance_maps.sh`** (`DRY`, `DOOM_ROOT`). It runs `eval_tf.py` unmodified per map with `--num-windows 256 --use-ema --num-workers 0`, horizons 1 and 4, the paper's step count and the stock decoder. It is resumable.
   - *No `--wandb-run`:* 30 per-map reads at one step would overwrite the headline `eval/ema_h1` series (`wandb_log.py:507-510`).
   - *GPU time:* the U-Net needs 30 × 256 × 5 = 38,400 frames at about 20 frames/s (4.0 measured at 50 steps, so about 20 at 10), about 1.3 hours with model loads. SD 3.5 is estimated at twice that. The 3-hour GPU 3 budget covers the U-Net at both horizons and SD 3.5 at one tic; SD 3.5's four-tic reads follow when the card allows. Measure frames/s on the first map.
3. **`paper/make_distance_figure.py`**: the figure, the stats JSON, and the appendix table.

**Order, with a first figure by Sep 25 evening.**
- Tonight (running): `enc-eval-maps` encodes all 210 seeded-corpus episodes per tic on GPU 3, 4-channel first (about 3 hours), then SD 3.5 (about 5). Then run the k and button reports.
- Sep 25 morning: clouds, distances, validation, freeze.
- Sep 25 afternoon: score the U-Net 200k EMA (about 1 hour).
- Sep 25 evening: figure v1.
- Sep 26: SD 3.5 (provisional until its 200k on Sep 27), pixels, and PixArt at 200k.
- Sep 27 to 28: space (c) if time allows, then the final figure.

**For Rohan.**
1. Agree the single pre-declared primary test before any per-map score is read.
2. A GPU 3 slot for the scoring pass after the encode drains.

The seeded corpus's 60 seen-map episodes (maps 1 to 15, 4 each) are in tonight's encode. For the new rows, maps 1 and 9 to 15 are unseen, which adds 8 points in the arena cluster (hence n = 30); maps 2 to 8 serve as a cross-corpus replication, not as extra points.

## 7. How to run

On Spiderman, in this order. Steps 1 to 5 are CPU only and run beside the latents; step 6 needs GPU 3; step 7 is CPU. Run each long step inside its own tmux session and stop it with `tmux kill-session -t <name>`, never with a pattern kill.

**0. A clean checkout.** `$D/repo` is the live runs' checkout and is never pulled (`after_nexttic.sh:481`), so the study runs from a clean clone of main taken after this work merges, as `repo_launch2` was.

```bash
D=/sata2/data/rnagabhi/doom
PY=$HOME/miniconda3/envs/doom/bin/python
R=$D/repo_distance
git clone -q https://github.com/RohanNaga/Doom.git $R && git -C $R rev-parse HEAD
cd $R
O=$R/results/distance_study
L1=$D/latents_arnold_dense_pertic;       E1=$D/latents_arnold_dense_pertic_eval;       S1=$D/latents_arnold_eval_pertic
L35=$D/latents_arnold_dense_pertic_sd35; E35=$D/latents_arnold_dense_pertic_eval_sd35; S35=$D/latents_arnold_eval_pertic_sd35
```

**1. Smoke run** (a few minutes; two episodes per map, 100 directions, a throwaway root):

```bash
$PY distance_study.py clouds --space sd1 --out $D/tmp/distance_smoke --limit 2 \
  --reference $L1/arenas --reference-ids 0:2000 --draw 1 --eval val=$E1/val --eval unseen2=$S1/unseen2
$PY distance_study.py clouds --space sd1 --out $D/tmp/distance_smoke --limit 2 \
  --reference $L1/arenas --reference-ids 0:2000 --draw 2
$PY distance_study.py distances --space sd1 --out $D/tmp/distance_smoke --projections 100 --bootstrap 5
```

**2. Clouds.** SD 1.x draws the frames: 50 seeded episodes per training map from ids 0:2000 (draw 1 the reference, draw 2 the disjoint floor draw), and 250 frames per episode, 25 per motion decile. SD 3.5 and pixels are read at exactly those episodes and tics (`--frames-from`), so the three spaces differ only in the features. About 1.3 GB written in all (SD 1.x about 140 MB, SD 3.5 about 550 MB, pixels about 650 MB).

```bash
tmux new -d -s distance-clouds "cd $R && ( set -e
$PY distance_study.py clouds --space sd1 --out $O --reference $L1/arenas --reference-ids 0:2000 --draw 1 \
  --eval val=$E1/val --eval arenas_678=$E1/arenas_678 --eval seen=$S1/seen --eval unseen=$S1/unseen --eval unseen2=$S1/unseen2
$PY distance_study.py clouds --space sd1 --out $O --reference $L1/arenas --reference-ids 0:2000 --draw 2
$PY distance_study.py clouds --space sd35 --out $O --frames-from $O/clouds/sd1 --reference $L35/arenas --draw 1 \
  --eval val=$E35/val --eval arenas_678=$E35/arenas_678 --eval seen=$S35/seen --eval unseen=$S35/unseen --eval unseen2=$S35/unseen2
$PY distance_study.py clouds --space sd35 --out $O --frames-from $O/clouds/sd1 --reference $L35/arenas --draw 2
$PY distance_study.py clouds --space pixels --out $O --frames-from $O/clouds/sd1 --reference $D/raw_arnold_dense/arenas --draw 1 \
  --eval val=$D/raw_arnold_dense/arenas --eval arenas_678=$D/raw_arnold_dense/arenas_678 \
  --eval seen=$D/raw_arnold_eval/seen --eval unseen=$D/raw_arnold_eval/unseen --eval unseen2=$D/raw_arnold_eval/unseen2
$PY distance_study.py clouds --space pixels --out $O --frames-from $O/clouds/sd1 --reference $D/raw_arnold_dense/arenas --draw 2
) >> $D/logs/distance_clouds.log 2>&1"
```

**3. Per-map split files** for the scoring pass (one per primary map; the seeded corpus's copies of maps 2 to 8 are replications and get none):

```bash
$PY distance_study.py splits --out $O --eval val=$E1/val --eval arenas_678=$E1/arenas_678 \
  --eval seen=$S1/seen --eval unseen=$S1/unseen --eval unseen2=$S1/unseen2
```

**4. Distances, validation and bootstrap**, one space at a time (measured on a laptop CPU: about 25 minutes for the map, episode and floor distances, 20 for the best mixture and 50 for 200 bootstrap redraws per space):

```bash
tmux new -d -s distance-sw "cd $R && for S in sd1 sd35 pixels; do $PY distance_study.py distances --space \$S --out $O --bootstrap 200; done >> $D/logs/distance_sw.log 2>&1"
```

Read `checks` and `primary_arm` in `$O/distances_sd1.json` before going on: check (v) decides whether the motion or the uniform arm is primary.

**5. Freeze the distances before any per-map score exists.** Copy the small outputs back and commit them; each JSON already carries the commit and the SHA-256 of `distance_study.py`.

```bash
# on the laptop, in the repo (Spiderman is rnagabhi@128.2.204.110; connect as the run-server skill says)
rsync -a rnagabhi@128.2.204.110:/sata2/data/rnagabhi/doom/repo_distance/results/distance_study/ results/distance_study/ \
  --include='distances_*.json' --include='per_episode_*.csv' --include='bootstrap_*.npz' --include='splits/***' --exclude='*'
git add -f results/distance_study && git commit -m "freeze the distance study distances"
```

**6. Score the maps on GPU 3** (check `nvidia-smi` first). The U-Net 200k EMA at one and four tics, then SD 3.5's latest snapshot at one tic, then its four-tic reads when the card allows. Every read is resumable, so the same command continues after an interruption. `DRY=1` in front of any of these prints every command instead.

```bash
SD35=$(ls $D/results_spiderman/042-sd35-nexttic/snap_*.pt | tail -1)
tmux new -d -s distance-score "cd $R && (
CUDA_VISIBLE_DEVICES=3 RUN_REPO=$R CKPT=$D/results_spiderman/040-unet-nexttic/snap_0200000.pt bash scripts/spiderman/score_distance_maps.sh 3 unet
CUDA_VISIBLE_DEVICES=3 RUN_REPO=$R CKPT=$SD35 HORIZONS=1 bash scripts/spiderman/score_distance_maps.sh 3 sd35
CUDA_VISIBLE_DEVICES=3 RUN_REPO=$R CKPT=$SD35 HORIZONS=4 bash scripts/spiderman/score_distance_maps.sh 3 sd35
) >> $D/logs/distance_score.log 2>&1"
tail -f $D/logs/distance_040-unet-nexttic_sd1.log     # frames/s of the first map is printed as it finishes
```

Outputs land in `$D/results_spiderman/distance_study/<run>/<snapshot>_ema_ddim10/<set>_map<NN>_h<K>/`. The scorer never passes `--wandb-run` (section 6, item 2).

**7. Statistics, appendix table and figure** (needs matplotlib in the interpreter; otherwise run it on the laptop from a mirror of the two score directories):

```bash
$PY paper/make_distance_figure.py --distances $O/distances_sd1.json \
  --row unet=$D/results_spiderman/distance_study/040-unet-nexttic/snap_0200000_ema_ddim10 \
  --row sd35=$D/results_spiderman/distance_study/042-sd35-nexttic/$(basename $SD35 .pt)_ema_ddim10 \
  --space pixels=$O/distances_pixels.json --space sd35=$O/distances_sd35.json --out $O/figure
```

It writes `stats.json` (the primary partial Spearman with its bootstrap CI and permutation p, leave-one-map-out, within-cluster, per-row, per-space and secondary reads, and the section 5 verdict), `distance_table.md`, and `distance_gain.{png,svg}` with a dark variant `distance_gain_dark.{png,svg}`.
