# Astra round 1: rethinking the distance (2026-09-26)

Codex thread id: `01a0dfe1-861e-7733-8fad-32431548551e` (gpt-6-astra, reasoning effort high, workspace-write sandbox, one turn). No follow-up turn was needed: Astra named a single first candidate without hedging.
Brief: `.claude/analyses/astra-brief-distance-rethink-2026-09-26.md`. Astra's scratch artifacts (in `/tmp`, ephemeral): `/tmp/astra_distance_rethink.py`, `/tmp/astra_distance_rethink_results.json` (with input hashes), `/tmp/astra_distance_maps.csv` (joined 30-map table), `/tmp/astra_transition_audit.py` (proposed server audit, tested on synthetic data only).

Conventions in Astra's numbers: Spearman across equally weighted maps. Partial correlations are Pearson correlations between rank residuals after regressing on the controls. Sources: D from `results/distance_study/distances_sd1.json` (primary records). Scores from `results/distance_study/scores/040-unet-nexttic_snap_0200000_ema_ddim10/*_h{1,4}/metrics.json`. Motion, lives and validity from `results/distance_study/per_episode_sd1.csv`. Gain means `psnr_raw.mean - persist_psnr_raw.mean`.

## 1. Diagnosis

**Summary.** Within the 13 unseen arenas, gain over persistence mostly tracks how much the footage moves. D carries no ordering signal, and it carries none for absolute latent error either once motion is controlled. The pooled -0.73 does not establish an ordering within the unseen arenas.

### Recomputed base relation

| Maps | n | rho(D, h1 gain) | partial, controlling persistence PSNR |
|---|---:|---:|---:|
| All | 30 | -0.440 | -0.734 |
| Unseen arenas | 13 | -0.016 | -0.119 |
| Campaign | 13 | +0.214 | -0.378 |

- If arena 7 (the largest D) is dropped, the unseen rho is -0.035 and the partial is +0.031, so no hidden ordering is being masked by it.
- Across all 30 maps, controlling persistence, training membership and family together leaves a partial of -0.423. Astra therefore would not say the pooled association lives *entirely* in the two contrasts, only that it does not establish the within-unseen-arena relation.
- A 10,000-draw map bootstrap (seed 20260926) gives the unseen rho a 95% interval of [-0.664, +0.616]. The null is wide. It is not an equivalence result.

### What within-arena gain reflects (unseen arenas, n = 13)

| Relation | rho |
|---|---:|
| stored latent motion (`motion_mean`) vs gain | **+0.835** |
| persistence PSNR vs gain | -0.769 |
| persistence PSNR vs model PSNR | +0.945 |
| motion vs model latent MSE | +0.786 |
| lives per episode vs gain | -0.113 |
| valid-window fraction vs gain | +0.137 |
| D vs LPIPS gain | -0.126 |
| D vs model latent MSE | -0.275 |
| D vs model latent MSE, partial on motion | +0.018 |

- Motion vs gain stays at partial +0.652 after controlling persistence PSNR. So the motion signal is more than the arithmetic coupling inside `gain = model - persistence`. Latent change is still a proxy. It also responds to texture, lighting, animation and HUD change, not only to camera motion.
- Arena 8 (D 0.1149, model 22.279 dB, gain -0.922 dB, latent MSE 0.2164) and arena 15 (D 0.1265, 21.142 dB, gain +1.295 dB, latent MSE 0.2303). Arena 15 is *worse* in absolute terms but benefits more relative to copy-last. A positive gain does not mean the transitions are better covered.
- An OLS fit of model PSNR on persistence PSNR across unseen arenas has slope 0.639 and R^2 0.843. Gain therefore falls about 0.36 dB per dB of persistence. This describes the data; it is not a causal estimate.
- **Decision-phase composition is negligible.** Re-weighting phases 0 to 3 equally changes unseen gain by at most 0.0236 dB, and D vs adjusted gain is +0.033.
- **Deaths:** the map-level lives and validity correlations are weak. That does not rule out effects from time since respawn or from excluding windows that cross a life boundary.
- **Decoder ceiling:** no unseen arena's VAE reconstruction PSNR falls below raw persistence. That happens only on campaign maps 19, 25 and 26. Unseen reconstruction PSNR spans 22.690 to 24.632 dB. D vs (reconstruction minus model PSNR) is +0.093. The decoder matters for campaign maps but does not explain the arena null.
- **h4 does not rescue D:** rho(D, h4 gain) is -0.154, and -0.174 partial on h4 persistence.
- **Could not check locally:** the executed-control distribution and window composition. The local score tree has zero `per_window.csv` files and no h1 copy-latent-MSE summaries, so per-window motion matching, weapon switches and the latent ratio are all unresolved.

### Corrections to the brief (Astra's, confirmed by me)

- D's feature space is 192-dimensional pooled SD 1.x latents: 4x32x40 with the padding rows removed, then average-pooled in blocks to 4x6x8. It is not the flattened 4x16x20 latent. Confirmed in the `pool_latents` docstring in `distance_study.py`.
- D averages the distance over the prescribed four-episode subsets rather than transporting one cloud built from all episodes. The distances JSON has `subsets: 10` and `subset_episodes`.
- The scored checkpoint takes a 32-tic history of executed controls, not only the newest control.

### Is gain the wrong outcome?

- **Keep gain** as the answer to "does prediction beat copying". It does not remove intrinsic difficulty, and it is not the outcome a coverage distance should be validated on.
- For adaptation, Astra proposes a primary skill statistic built as a **ratio of aggregate errors**: R_m = sum_i w_i e_i / sum_i w_i b_i. Here e_i is the model's latent MSE on window i, b_i is the copy-last latent MSE, and the weighting is fixed across maps and budgets. Report absolute latent MSE and the denominator next to it.
- This differs from the mean of per-window ratios that `eval_tf.py` exports today. That mean is dominated by near-static windows. Exactly static subsets get a separate absolute-error report.
- For diagnosing transfer, also report latent error that is **standardized by motion and control**:
  - Bin motion with thresholds taken from the source data, and add a few strata of executed controls.
  - Compare maps under common stratum weights.
  - Report missing overlap explicitly instead of extrapolating.
- Context motion is the prospective control. Realized next-frame motion is only a retrospective diagnostic.
- LPIPS stays as a perceptual outcome. It should not be picked just to rescue D. Persistence baselines have precedent in arXiv:1605.08104 (PredNet) and arXiv:1511.05440 (Mathieu et al.).

## 2. Ranked candidate distances

### 1 (first candidate): directed coverage of transition windows by the training corpus

The question it asks: does the training footage contain transitions like the ones this map requires?

- **Per-window features:**
  - State: pooled z_t.
  - Context dynamics: pooled z_t - z_{t-l} for l in {1, 4, 16, 31}.
  - Innovation: pooled z_{t+1} - z_t.
  - Controls: the newest executed control, each button's frequency over the context, and each button's switching rate (57 features).
- **Projection and weighting:** fixed Gaussian projections (seed 0) take state, dynamics and innovation to 32, 64 and 32 dimensions. The controls stay raw. Each block is centered on training data and scaled by its source RMS pairwise distance, and the four blocks get equal total weight. These are design choices, not fitted values.
- **Score:** D_cov(m) = mean over target windows of the mean distance to the 3 nearest source windows, each from a distinct source episode. The search runs over the union of the four training maps. Directed coverage does not penalize a target for training modes it never uses, so the per-map minimum is no longer needed.
- **Sampling floor and diagnostics:** compute the same score on disjoint held-out source episodes. Export ablations with state only and with dynamics plus controls only, examples of the retrieved nearest neighbors, and scores per motion and control stratum.
- **Captures:** appearance, recent trajectory shape, the size and direction of the transition, and which transitions go with which controls. It separates footage whose frame marginals match but whose temporal pairing differs.
- **Does not capture:**
  - Equality of the conditional transition law. It measures support, so a familiar-looking neighbor can hide a wrong transition probability.
  - Rare supported transitions: finite sampling can make them look novel.
  - The full model input: a compressed history is not the whole 32-tic context.
  - Appearance: VAE differences still leak in.
  - LoRA optimization cost is not guaranteed to follow coverage.
- **Motivation:** OTDD (arXiv:2002.02923) motivates putting task information into dataset distances, but it does not validate this particular score.

### 2: sliced Wasserstein on temporal features, stratified by action

- Take the dynamics and innovation blocks plus control-history summaries and drop absolute z_t.
- Stratify by executed turn direction and by attack. In each stratum, compare the target to the pooled source with fixed 256-direction sliced Wasserstein.
- Aggregate with frozen source stratum weights on the common support. Report action-frequency differences and unsupported mass separately rather than folding them into one scalar.
- **Captures:** the temporal-behavior shift that remains after coarse control matching. It is cheap and reuses most of `distance_study.py`.
- **Does not capture:** it penalizes harmless redistribution among transitions that are already covered, loses state-dependent geometry, and latent differences still depend on texture.
- **Why it ranks second:** adaptation probably depends more on missing support than on frequency mismatch.

**Deferred:** distances measured through the model itself, such as Fisher or Task2Vec-style embeddings (arXiv:1902.03545). A faithful version for this diffusion model is its own study.

## 3. Validation protocol (before either distance becomes the adaptation x axis)

- Do not tune feature weights or the representation to maximize correlation on these 13 arenas.
- **Check that the score measures what it claims to:**
  - Source calibration: disjoint source episodes should score close. Exclude neighbors from the same episode.
  - Sampling stability: repeat the source reference draws and the episode subsamples, and bootstrap over episodes rather than windows.
  - Temporal sensitivity: shuffle innovations against contexts while keeping the marginals. The score must rise.
  - Control sensitivity: shuffle controls against transitions while keeping control frequencies. The score must rise.
  - Motion sensitivity: measure how much reweighting by motion alone moves the score.
  - Representation sensitivity: if the state-only ablation recovers nearly the whole score, the new distance is still an appearance distance.
  - Qualitative check: inspect the retrieved source transitions for near and far target windows.
- **Then test predictive value** on unseen arenas beyond motion and zero-shot skill, using prediction on held-out maps and reporting uncertainty. With 13 maps, elaborate regressions will be unstable.
- **Freeze before any adaptation result is seen:** the feature recipe, references, weights, sampling rule, outcome and analysis. A distance developed on zero-shot data can still get an honest prospective test on adaptation cost. Its development correlation cannot become confirmatory evidence.
- **Adaptation analysis:** treat maps that never cross the skill target as right-censored, and note that a checkpoint grid only locates a crossing within an interval. Zero-shot skill and simple motion statistics enter as competing predictors. The distance must add information beyond how far the model starts from the target.

## 4. Server request and cost (Astra's planning estimates, not benchmarks)

**Audit (needs no new inference if the per-window latent errors survive on the server).** Copy `/tmp/astra_transition_audit.py` to the server's `/tmp` and run:

```bash
python /tmp/astra_transition_audit.py \
  --scores /sata2/data/rnagabhi/doom/results_spiderman/distance_study/040-unet-nexttic/snap_0200000_ema_ddim10 \
  --out /tmp/astra_transition_audit.csv
```

- The script reads each score file's latents-dir config, joins windows by episode and start row, and checks tic continuity and life and map boundaries.
- It exports per window: copy-last latent MSE, latent ratios, context-motion statistics, the newest executed buttons and their context frequencies, control switching, time since respawn, the original scores and the window identities.
- Supervisor note: the `--scores` path is Astra's inference and is unverified. The metrics config puts the checkpoint under `/sata2/data/rnagabhi/doom/results_spiderman/` on Spiderman, but where the per-window score rows live must be confirmed before running. The script has passed synthetic checks only.

**Extraction for the candidate distances:**
- Source: the frozen source manifests (50 episodes per training map) plus the disjoint reference draw.
- Target: all 342 primary evaluation episodes, 112 of them from unseen arenas.
- Sampling: 250 valid transition windows per episode. That gives 85,500 target descriptors and 100,000 source and calibration descriptors.
- Weights: weight each window by the episode's eligible-window count divided by its draw count, and also export estimates with equal weight per episode.
- Outputs: feature and config manifests, per-window coverage components, per-map distances, episode-bootstrap intervals and nearest-neighbor identities. The frozen D outputs are never overwritten.
- Storage: about 1 GB of new scratch, streaming the latents rather than copying them. Benchmark on a few episodes first.

| Work | CPU core-hours | When |
|---|---:|---|
| Join existing scores to latent and control windows | 0.1 to 0.5 | Sunday |
| Shared temporal-feature extraction | 2 to 6 | Sunday |
| Candidate 1: coverage, two references, first diagnostics | +4 to 16 | exploratory result Sunday, validated read Tuesday |
| Candidate 2: stratified SW with episode uncertainty | +2 to 8 | exploratory result Sunday, read Tuesday |
| Policy replication or a distance measured through the model | separate pilot | October |

## 5. Proposed claim for the Sep 30 draft (Astra's wording)

> Across 30 maps, the frozen appearance-based distance is associated with lower PSNR gain over persistence after controlling for persistence PSNR (partial Spearman rho = -0.734). However, this association does not establish an ordering of unseen arenas: among the 13 unseen arenas, the corresponding correlations are rho = -0.016 and partial rho = -0.119. The distance therefore characterizes broad differences among training, unseen-arena, and campaign footage, but has not demonstrated predictive ordering within unseen arenas. We report the failed prespecified validation gate and do not treat this distance as a validated predictor of adaptation cost.

- Astra prefers "broad group differences" over "separates families", because the groups overlap.
- It limits the within-family statement to the unseen arenas and does not generalize it to every family. The campaign maps show partial -0.378 on n = 13, which is not a clean null.

## 6. Supervisor rechecks

I recomputed these independently with my own script. It joins the primary records of `distances_sd1.json` to the h1 `metrics.json` files and uses NumPy rank correlations. I did not reuse Astra's code.

| Number | Astra | Recheck |
|---|---:|---:|
| rho(D, gain), all 30 | -0.440 | -0.440 |
| partial rho(D, gain \| persistence), all 30 | -0.734 | -0.734 |
| rho(D, gain), unseen 13 | -0.016 | -0.016 |
| partial, unseen 13 | -0.119 | -0.119 |
| rho(persistence, gain), unseen | -0.769 | -0.769 |
| rho(persistence, model PSNR), unseen | +0.945 | +0.945 |
| rho(motion_mean, gain), unseen, episode means over each map's primary episodes | +0.835 | +0.835 |
| rho(motion_mean, latent MSE), unseen | +0.786 | +0.786 |
| arena 8 gain (22.2792 - 23.2008) | -0.922 | -0.922 |

- The brief's -0.78 for persistence vs gain is -0.769 on recomputation.
- I also confirmed the local score tree contains no `per_window*` files, so Astra's "could not check" list is accurate.
- Not rechecked: the bootstrap interval, the phase re-weighting, the decoder-reference numbers and the OLS slope.
