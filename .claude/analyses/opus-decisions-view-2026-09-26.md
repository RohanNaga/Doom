# Opus view: five open decisions for the paper and the adaptation study (2026-09-26)

Positions A to E were written before reading `astra-adaptation-review-2026-09-26.md`. "(recomputed)" means I derived the number from `results/distance_study/figure_unet_h1/distance_table.md` or from the code.

## Recommendations

- **A. Metric:** we did not invent it. It is a log-form skill score against a persistence reference. Print absolute model, copy-last and reconstruction rows in the table, and plot the paired difference. Cost: about 1 h of writing.
- **B. Decision 7:** use decoder-free latent skill S = −10·mean log10(R), with cost set at 25, 50 and 75 percent of the pooled training-map S. LPIPS is secondary; R ≤ 0.90 is a sensitivity. Freeze the formula tonight. Cost: about 30 min of CPU.
- **C. Decision 5:** run a full fine-tune of arena 7 on Monday only on an idle card, starting from EMA weights, with the pilot's split and grid. Otherwise October. Cost: about 2 A4000-hours plus about 1 h to add the flag.
- **D. Decision 6:** the Sep 30 paper carries only the four-map pilot, run seed-major. The 13-arena curve goes to October whatever it shows. Cost: about 6 A6000-hours by Sunday night.
- **E. Campaign arm:** keep maps 20, 32, 30 and 23 for October as a never-pooled out-of-family prediction test. Record them after the disk deletion. It costs nothing now.

## A. Is "gain over persistence" invented?

**Copy-last as a row.** Mathieu 2016 (Table 2), Finn 2016, PredNet 2017 (Table 2, KITTI to unseen CalTech), ContextVP, SDC-Net, Villegas 2019 (copy-last beats every model on Human3.6M) and ThinkJEPA 2026 (latent persistence) all report it. OccWorld Table 1 and DINO-Foresight Table 2 print three rows: reconstruction, copy-last and model (`distance-study-literature` Q4, `related-work-map` §1).

**A difference or a normalised score.**
- Genie's ΔtPSNR is a difference of two PSNRs. Its reference is random actions, so it measures controllability; cite it for the form only.
- Blitzer subtracts in-domain accuracy, OTDD divides by target-only error, and Procgen and Atari normalise per game.
- Meteorology's skill score is 1 − MSE_model/MSE_ref (Murphy 1988). Whether persistence is Murphy's reference is **VERIFY**.

**Why it matters.** The mean per-window PSNR difference equals the mean of 10·log10(MSE_persist/MSE_model), a log skill score. The LPIPS difference is Blitzer's subtraction. Persistence spans 18.8 to 27.2 dB across maps (recomputed), so absolute PSNR mostly measures how static a map is. Pooled correlations can vanish without per-target normalisation: in Transferring GANs the ρ is 0.8 within a target and −0.07 pooled (`lit-transferability` §D).

**Present both.**
- Table 1: absolute PSNR and LPIPS for the model, copy-last and true-frame reconstruction (OccWorld layout).
- Figure 1 and every statistic: the paired difference with an episode bootstrap.
- No new name. One sentence: "the paired difference PSNR(model) − PSNR(copy-last) on the same windows, 10·log10 of the MSE ratio: a skill score against a persistence reference (PredNet; Villegas et al.), of the same form as Genie's ΔtPSNR."

## B. Decision 7

**Facts (recomputed).**
- The training-map gain is +0.93 dB and +0.027 LPIPS, so half is 0.46 dB and 0.013.
- In PSNR, four of 13 arenas already exceed half at step 0: 15, 13, 16 and 17 (+0.53 to +1.30). That is a floor effect.
- In LPIPS every arena sits at −0.063 to −0.163, so reaching +0.013 invites censoring.
- The decoder falls below persistence only on campaign maps.
- `eval_tf.py` averages per-window ratios arithmetically (lines 369 to 373), so near-static windows dominate.
- R ≤ 0.90 is 0.46 dB of latent skill. It is anchored to pixels, not to the training-map latent skill, which is unknown until Sunday; the cut is unreachable if that skill is small and trivial if it is large.

**Options.**
- (1) The page's half-gain: floor in PSNR, censoring in LPIPS.
- (2) Astra's R ≤ 0.90: decoder-free, but unanchored.
- (3) Recommended: S in dB, the latent twin of the PSNR gain. Cost is the first grid budget that reaches 25, 50 or 75 percent of the pooled training-map S. The headline is 50 percent, pre-registered to move to 75 percent if four or more arenas already exceed 50 percent at step 0.

Also report S and LPIPS at 2,000 updates and the area under the curve (Taylor-Stone §2.1; Neyshabur §3.1). The Sep 30 figure shows curves only, so nothing waits on this.

## C. Decision 5

**For.** AVID (adapter 23.8 against full 25.8 PSNR) and AdAM §3 show adapters losing on far targets. Arena 7 (D 0.282) is the only far arena, so its cost may measure adapter capacity rather than distance. The run is cheap: 2,000 U-Net updates take about 21 min on an A6000. It also gives the forgetting contrast XEWorld invites.

**Against.** It is one map and one seed. `train_wm.py` initialises from live weights, so it needs a flag reviewed under deadline.

**Recommendation.** Run it Monday only if the pilot's arena-7 curve exists, the flag passes review and a card is idle. Use lr 5e-5, the same grid and the forgetting guard. Otherwise October.

## D. Decision 6

**Cards.**
- Encoding (about 4.5 A6000-hours) and the pilot both need Spiderman GPU 1 after 14:00 Sunday.
- GPU 2 owes SD 3.5's 200k reads.
- 13 curves need about 19.5 A6000-hours, and Superman has no copy route.

**Science.** Within the arenas, zero-shot gain ignores D (Spearman 0.01). Arena 7 carries 66 percent of D's variance, so power is about 40 percent. A Monday cost line would likely be flat and underpowered, would pre-empt October, and does not fit in 4 pages.

**Recommendation.**
- The pilot on 8, 16, 6 and 7 (corpus A) runs seed-major: seed 0 lands about 20:00 Sunday, seed 1 runs overnight. If encoding takes GPU 1 first, the draft carries a placeholder.
- Declare now that the paper reports the pilot only, which closes the include-if-good fork.
- A spare Monday card does the step-0 rescore on corpus B (about 2 A4000-hours).

## E. Campaign arm

**For.** The near pair 20 and 32 (D 0.140 and 0.143) and the far pair 30 and 23 (0.338 and 0.371) are the only way to separate family from distance. They were chosen on D alone. They are also the four campaign maps with the smallest per-episode D spread: 0.029 to 0.073, against 0.10 to 0.32 elsewhere (`opus-dataset-view`). So coverage is least confounded there. None of them is 19, 25 or 26, but their reconstruction margins are **VERIFY**.

**Against.** The footage is near-static, and four points cannot fit a slope.

**Recommendation.**
- Pre-register the test: fit cost on D over the arenas, then ask whether the campaign maps fall in its prediction interval.
- Record 80 episodes (7 min, 24 GB raw, 1.3 GPU-h to encode) after the 404 GB deletion, not tonight: 225 GB is free and it fills about 20 GB/h.
- Dropping the arm saves about 9 card-hours and loses the only family test.

## Dataset section

**Adapt and held-out suffice.** A validation set chooses hyperparameters, checkpoints or stopping points. Here the grid is fixed, nothing stops early, and settings and thresholds are frozen from the pilot on corpus A, which is disjoint from corpus B. The held-out episodes are therefore the test set.

**Pre-register two choices,** or they become hidden validation. First, which weights are scored. Second, any retuning after corpus B is read, which must use a declared carve-out of 2 of the 12 adapt episodes. The training-map validation ids 6000:6100 measure forgetting: an outcome, not a selection set.

**Numbers per target arena.** Pretraining used 500 episodes per training map (2,000 in all) and 200k updates of batch 32 (6.4M samples, about 1.6M per map).

| Budget | One training map's footage | All pretraining footage | Pretraining updates |
|---|---|---|---|
| 1 episode (2.5 min of play) | 0.2% | 0.05% | |
| 6 episodes, the step curve (15 min) | 1.2% | 0.3% | |
| 12 episodes, the adapt pool (30 min) | 2.4% | 0.6% | |
| 250 / 1,000 / 2,000 updates | | | 0.125% / 0.5% / 1% |

At 2,000 updates, the 64k samples are 4% of one map's share. They pass over the 6-episode set (about 29k windows, from arena 8's 95,567 over 20 episodes) about 2.2 times. Pretraining covered about 0.67 of its roughly 9.6M windows (**VERIFY** the training-map window count).

For the reader: 15 minutes of footage and 1 percent of the pretraining updates.

## Where I differ from Astra

**Where Astra is right.**
- Genie is cited for its form only.
- Reconstruction is a reference, not a proven ceiling.
- Live weights should be the primary score. An EMA at 0.999 still weights the step-0 model by 0.78 at 250 updates and 0.61 at 500, so it lags exactly the points that set cost. I revise my draft here.

**Where I differ.**
- **B:** an absolute R ≤ 0.90 is unanchored until the training-map latent skill is known, so I fix a fraction of that skill. Ratio-of-sums (Astra) and mean-log (mine) both avoid the mean-of-ratios problem.
- **C:** Astra drops the Monday run because full tuning is not a guaranteed upper bound (DiffFit). I keep it conditional as a capacity check on the only far arena, not as a ceiling.
- **Split:** Astra's 6/2/2 needs validation only for its "validation-selected crossing". With a fixed grid and frozen thresholds, 12/8 needs none.
