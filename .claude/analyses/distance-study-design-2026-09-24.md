# Distance study: does quality fall with distance from the training footage? (design, 2026-09-24)

Status: proposal. Prototype `distance_study.py`, tests `paper/fixtures/test_distance_study.py`. GPU scoring waits for Rohan's yes. Target sentence: "across N maps, gain over persistence falls with distance from the nearest training map (partial Spearman ρ, 95% CI)".

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
- *Cloud:* each map is 10 episodes × 250 target-eligible frames (at least 32 tics into a life), the population the scored windows come from. Maps with more episodes average five random 10-episode subsets. Sizes are equal because the finite-sample floor depends on n.
- *Features:* SD 1.x latents with the bottom padding rows removed (`encode_parquet.py:161`), 5×5-pooled to 192 dimensions (`pool_latents`).
- *Weights:* motion ||z_t − z_(t−1)|| at tic rate within each life (`latent_motion`). Weights are motion plus a closed-form floor giving the quietest half exactly 25 percent, per cloud (`motion_weights`).
- *p = 2:* W₂² is the least mean squared displacement between clouds, in the units of the latent MSE the model trains on.
- *Directions:* 1,000 from seed 0, shared by every distance (common random numbers). Measured: 1.6 s per distance on a laptop CPU, 2 percent spread across seeds.
- *No time coordinate:* windows are drawn uniformly and every episode lasts 150 s, so within-episode time is uniform on both sides. It would add only noise and a free price. The temporal structure the model uses, its 32-tic context, enters through the weights and space (c).

**Secondaries (pre-declared):** pooled distance, uniform weights (`quiet_share=0.5`), best mixture of training maps.

**Reference.** 50 seeded episodes per map from training ids 0:2000, 250 frames each: 12,500 per map, 50,000 in all. Stratified by map (equal, as in training) and by motion decile within each episode (25 frames each). That allocation is proportional, so it adds no bias, and it pins the loud tail that carries the weight. A disjoint second draw gives the train-versus-train floor.

**Validation before scoring.**
- (i) The four validation maps sit inside the floor band.
- (ii) Disjoint episode subsets of a map agree within 10 percent.
- (iii) The two reference draws rank the maps with Spearman ≥ 0.95.
- (iv) Arenas 6 to 8 (same WAD) sit nearer than the campaign maps.
- (v) |Spearman(D, Kish n_eff)| < 0.4. Motion weights shrink n_eff and raise the floor (synthetic: 0.040 uniform against 0.059 weighted at n_eff 976); if the check fails, the uniform arm becomes primary.

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
- *Two clusters:* 13 campaign maps versus 9 arenas. A gap between clusters alone can inflate ρ, so report ρ within each.

**Map level (n ≈ 22, primary).**
- *Correlation:* Spearman ρ(D, Δ), CI from a case bootstrap over maps (10,000 draws, episodes resampled within maps), permutation p-value.
- *Partial on motion:* rank residuals on persistence PSNR.
- *Leave-one-map-out:* the sign must hold in all 22 refits.
- *Distance error:* 200 episode redraws. If D's SD is under 10 percent of its spread across maps, attenuation is below 1 percent.
- *Power:* |ρ| ≥ 0.43 at n = 22, and 0.56 within 13 maps.

**Episode level (n ≈ 310, secondary).** Episodes nest in maps, so report a within-map ρ (map-demeaned ranks, permutation within maps) and a mixed model Δ_e ~ D_e + persistence_e + lives_e + (1 | map).

**One primary test:** SD 1.x space, U-Net 200k EMA, one-tic PSNR gain, partial on motion. Everything else is secondary.

## 4. Alternatives

- **Fréchet distance:** assumes one Gaussian, but map frames are multimodal and a 768-dimensional covariance from 2,500 frames is biased. Robustness check only.
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

   Run on Spiderman beside the roughly 10 GB of training latents, in one tmux session. Timing: 3 minutes for map distances, 30 for episodes, about 2 hours for the bootstrap.
2. **`scripts/spiderman/score_distance_maps.sh`** (`DRY`, `DOOM_ROOT`). It runs `eval_tf.py` unmodified per map with `--num-windows 256 --use-ema --num-workers 0`, horizons 1 and 4, the paper's step count and the stock decoder. It is resumable.
   - *No `--wandb-run`:* 22 per-map reads at one step would overwrite the headline `eval/ema_h1` series (`wandb_log.py:507-510`).
   - *GPU time:* the U-Net needs 28,160 frames at about 20 frames/s (4.0 measured at 50 steps, so about 20 at 10), roughly 1 hour. SD 3.5 needs about 2. That is 3 hours on GPU 3, between steward reads.
3. **`paper/make_distance_figure.py`**: the figure, the stats JSON, and the appendix table.

**Order, with a first figure by Sep 25 evening.**
- Tonight: per-tic encode of unseen2 and arenas 16 to 17 (SD 1.x first), then the k and button reports.
- Sep 25 morning: clouds, distances, validation, freeze.
- Sep 25 afternoon: score the U-Net 200k EMA (about 1 hour).
- Sep 25 evening: figure v1.
- Sep 26: SD 3.5 (provisional until its 200k on Sep 27), pixels, and PixArt at 200k.
- Sep 27 to 28: space (c) if time allows, then the final figure.

**For Rohan.**
1. The go-ahead for GPU 3.
2. The primary test.
3. Whether to also encode the seeded corpus's 60 seen-map episodes (maps 1 to 15, 40 percent more encode). For the new rows, maps 1 and 9 to 15 are unseen, which adds 8 points between the clusters (n = 30, threshold 0.36) and replicates maps 2 to 8.
