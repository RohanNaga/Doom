# Astra review: reference baseline and per-map adaptation study (2026-09-26)

**Thread id:** `01a0dfa4-ec22-7f91-9c5d-7298c494c132` (gpt-6-astra, reasoning high, workspace-write sandbox with network, 2 turns). Round 1 was unanchored: the brief `.claude/analyses/astra-brief-adaptation-2026-09-26.md` verbatim, design page withheld. Round 2 showed `docs/lora_adaptation_design_2026-09-26.html`. Astra changed no repository file and had no lab-server access.

Everything in sections 1 to 3 is Astra's position, condensed in its terms. Section 5 lists what the supervisor checked.

## 1. Round 1, question 1: the reference baseline

**Answer.** Keep copy-last persistence in the headline figure, but present it as a *minimum prediction reference*, not a definition of a good world model. No paper sets a standard threshold for "reasonably good". Each paper picks the reference that fits its claim.

**Literature Astra checked in primary PDFs:**

| Paper | Reference it uses | What it implies here |
|---|---|---|
| PredNet, arXiv 1605.08104, Table 2 | Copy Last Frame, KITTI to CalTech transfer | Direct precedent for persistence on unseen-domain prediction |
| OccWorld, arXiv 2311.16038, Table 1 | Copy&Paste plus reconstruction at t=0 plus forecasters | Precedent for showing reconstruction, persistence and prediction together |
| GameNGen, arXiv 2408.14837, §5.1 | Absolute PSNR/LPIPS, FVD, human eval | Measures fidelity; does not show gain over a trivial predictor |
| Genie, arXiv 2402.15391 | ΔPSNR = inferred-action PSNR minus random-action PSNR | Measures controllability, **not** gain over persistence |
| Dreamer 4, arXiv 2509.24527, §4.3 | Normalised between no-action and all-action training | Action-label transfer; targets were in unlabeled training video, unlike unseen Doom maps |
| Pathdreamer, arXiv 2105.08756, §4.1 | Nearest-neighbour interpolation filling reprojection holes | A geometric reference, **not** nearest-training-example retrieval |

**Headline figure.** Per-map LPIPS gain over raw copy-last, sorted by frozen D, with training membership and family marked. Beside it: a PSNR panel with the reconstruction reference drawn relative to persistence, a latent-persistence panel once measured, and absolute scores in the table.

**What must sit beside persistence:** (1) true-frame reconstruction, (2) control response (copying scores well while ignoring actions), (3) short autoregressive evaluation, (4) motion-stratified results with strata fixed from real footage.

**Rejected alternatives.** An in-domain average as denominator is wrong because another map is another prediction problem. An all-map model would be an informative comparator but is not available and is not a theoretical upper bound. Action shuffling does not substitute for a trained no-action model. Transition retrieval from training data is a good October baseline. FVD is supplementary.

**Correction to the gap explainer.** Encoder-then-decoder reconstruction is *not* a mathematical ceiling on every predicted latent, because the encoder need not produce the pixel-optimal latent for that decoder. Maps 19, 25 and 26 (reconstruction below raw persistence) are therefore not proven unwinnable. They show that raw-pixel persistence crossing is a poor universal adaptation target. Likewise the 0.19 dB flat-ceiling gap does not bound what decoder adaptation could recover.

**Adaptation-curve outcome.** Aggregate latent skill against latent persistence on the fixed window set W_m:

R_m(s,n) = Σ‖ẑ − z‖² / Σ‖z_last − z‖², S = 1 − R (zero means persistence).

Use a ratio of sums, not the mean of per-window ratios, which near-static windows can dominate (the current `eval_tf.py:53` helper computes per-window ratios). Score the unpadded latent region. Plot zero-shot S_m(0) with every curve. Report raw PSNR/LPIPS gains, the gain over the unchanged model, 4-tic autoregressive skill, and the control and retention reads beside it.

**Astra's recomputation from the 30 per-map U-Net 200k EMA h1 files:** training maps +0.923 dB and +0.027 LPIPS over persistence. Unseen maps −0.536 dB, 9/26 PSNR wins, 0/26 LPIPS wins. VAE reconstruction 23.907 vs 23.714. Partial Spearman −0.7337 (controlling for persistence PSNR). It falls to −0.4227 after adding training membership and family, and to −0.4313 on unseen maps only with persistence and family controls. On the 18 adaptation-eligible maps it is −0.4524, with a map-only bootstrap of roughly [−0.76, +0.19]. That diagnostic recomputation is not the original estimator. Astra's conclusion: the pooled −0.73 is not evidence that D ranks difficulty *among unseen maps*.

## 2. Round 1, question 2: Astra's own design

**Question, sharpened.** Under a fixed adaptation recipe, does D predict the budget needed to reach a stated prediction quality, *beyond* what motion content and zero-shot error already predict? A correlation between D and final improvement answers a different question.

**Eligible maps (18).** Five arenas: 6, 7, 8 (20 episodes each), 16 and 17 (10 each). Thirteen campaign maps with 10 each: 18, 19, 20, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32. This corrects the brief, which said 18 ten-episode maps plus 3 arenas.

**Pilot maps.** Arena 8, campaign 31, campaign 23, chosen by D (Astra gave about 0.115, 0.210, 0.371) and not by outcome.

**Adapter recipe (primary arm):**
- Initialize from the 200k **EMA** weights. `load_init_weights` loads `ck["model"]`, so the adapter path must select EMA explicitly and reproduce the zero-shot prediction at step 0.
- Rank-16, alpha-16 LoRA on Q/K/V/out of self- and cross-attention, zero-initialized, no dropout (Vista, arXiv 2405.17398, App. C.3, uses rank 16).
- Frozen: the whole backbone, including input/output convs, norms, timestep/noise embeddings, the **entire control encoder with positions**, and the VAE.
- Fused AdamW, LR 1e-4, 200-update warmup, then constant; weight decay 0, clip 1.0, **global batch 32**, bf16. LR and warmup are engineering choices, not literature-proven.
- Score live adapters over the EMA base; do not inherit a 0.9999 adapter EMA. Keep 10-step eta-0 DDIM scoring (the current score config; the "ancestral DDPM" prose is stale for these reads).
- Sensitivity arm on all three pilot maps: rank 64 plus map-specific input-projection and control-encoder deltas at LR 1e-5, parameter count reported separately.
- DiffFit (arXiv 2304.06648, Table 1): LoRA rank 8/16 mean FID 81.25/81.31, full tuning 16.59, DiffFit 15.39. That is a reason to test capacity, not proof that this recipe fails.

**Data and split.** Per map, a frozen permutation (seed 926) into **6 adapt / 2 validation / 2 test** episodes. On arenas 6 to 8, the other 10 episodes are held for replication. Nested adaptation subsets of 1, 2, 4 and 6 episodes. The split stays fixed across optimizer seeds 0, 1 and 2. Freeze a new 256-window test manifest balanced over the 2 test episodes, with provenance and paired noise, and rescore zero-shot on exactly those windows.

**Grids:**

| Experiment | Episodes | Update checkpoints | Seeds |
|---|---|---|---|
| Pilot (3 maps) | 6 | 0, 100, 250, 500, 1000, 2000 | 0, 1 |
| October step curves | 6 | the above plus 4000 | 0, 1, 2 |
| October data curves | 1, 2, 4, 6 | fixed 2000 | 0, 1, 2 |

Each data-budget run starts fresh from the base. Every point saves latent skill, both MSE terms, h1/h4 pixel metrics and the control read. At zero-shot, at the validation-selected crossing and at the terminal checkpoint, 16 windows also get 64-tic rollouts. The bootstrap resamples episodes, not windows.

**Cost:**
- *Step cost:* the first update budget at which R ≤ 0.90, with 6 episodes.
- *Data cost:* the smallest episode budget reaching R ≤ 0.90 at 2000 updates.
- *Compute cost:* GPU-hours, trainable parameters and storage.
- 0.90 is a proposed margin. R ≤ 1 and R ≤ 0.8 are pre-registered sensitivities.
- A map already over threshold has zero cost. Non-crossers are right-censored, and crossings between checkpoints are interval-censored.
- Keep a threshold-free area under skill against log updates, plus terminal skill.

**Guarded crossing:**
- Source-map mean latent error at most 5% above base, with any single map over 10% flagged.
- Turn-response success at most 5 points below base.
- Target LPIPS at most 0.01 worse than zero-shot.
- No new non-finite or absorbing rollouts.
- Retention is measured with the adapter **enabled** on the source maps.
- Replay, if needed, is a separate fixed recipe with its cost charged, never added only to hard maps.

**Statistics (the unit is the map):**
1. Primary: D against budget-capped step cost, controlling for log latent-persistence error and family, with censoring flags.
2. Incremental test: add zero-shot latent error, since D may only predict a worse starting point.
3. Threshold-free sensitivity: the same analysis on integrated skill and final skill.
4. Leave-one-map-out prediction against family/motion and family/motion/zero-shot baselines.

Use a map-level bootstrap and a covariate-aware permutation. Use censoring-aware concordance, never Spearman after imputing 4000 for non-crossers. Keep SD1 D as the axis; pixel and SD3.5 distances are robustness checks only. LEEP (arXiv 2002.12462) and Westny et al. (arXiv 2606.30777, whose Wasserstein measure trails KL in its Table 2) mean D must be compared with simpler predictors.

**Transductive caveat.** Frozen D was computed from footage that includes the future test episodes. That supports a benchmark association, not "predict cost after collecting one episode". October needs a prospective D from a fixed acquisition prefix, charged to the data budget.

**Reviewer objections:**
- XEWorld (arXiv 2608.05799) already combines distance, few-shot adaptation and forgetting (r = 0.812, permutation p = 0.075 on 5 robots; 25/50/75-episode adaptation). The contribution must be the many-target, fixed-recipe, persistence-referenced cost study.
- AVID (arXiv 2410.12822, Table 1): the adapter reaches 23.8 PSNR against 25.8 for full tuning, but AVID wins on action error, so no method is a universal upper bound.
- AdaWorld (arXiv 2503.18938) already reports step and sample curves: 800 updates at batch 32, 100 samples per action. Episode count alone ignores action coverage.
- DiffFit: a flat rank-16 curve shows adapter failure, not an unadaptable map.
- Does D add anything beyond initial error, motion and family?

**Oct 1 versus October.** The Oct 1 draft needs the persistence/reconstruction presentation, the latent ratio, the motion audit, control calibration and final scoring, and it must stand without adaptation. An optional pilot figure must show all three pilot maps, including failures, with 2 seeds and the zero-shot reference, and no distance law from 3 points. The proposed cap is 48 A4000 GPU-hours including evaluation (not measured). October gets the 18-map, 3-seed grids (about 540k updates), ablations, split replication, prospective D, PixArt next, and SD3.5 when a 48 GB card is free.

## 3. Round 2: decision-by-decision comparison with the design page

| # | Decision | Page | Astra | Astra's change before Sunday |
|---|---|---|---|---|
| 1 | Per-map split | 6/4 (10-ep), 12/8 (20-ep); held-out = existing 256 draw restricted to held-out episodes | 6/2/2 everywhere, fresh frozen 256-window manifest | Use 6/2/2 and rescore zero-shot on it. 6/4 is defensible only if nothing is chosen on target validation. Unequal 6 vs 12 adapt episodes confounds step cost with data. Restricting the old draw keeps about 102 windows (256 × 0.4), not 256, and `eval_tf.py:276` redraws indices over the selected dataset |
| 2 | Pilot maps | 17, 24, 26, 23 | 8, 31, 23 | Prefers 8/31/23. The page's four are fine if the primary outcome is latent skill. Map 17 already clears the +0.5 dB target (+0.528). Map 26's reconstruction sits 2.161 dB below persistence and map 24's only 0.144 above, so a PSNR crossing there is hard to read |
| 3 | Rank | r16, ablation r4 and r64 on 2 maps, about 6 card-h | r16 plus an r64 sensitivity | **Agrees on r16.** Adopts the page's rule of holding non-LoRA parts fixed across ranks. Run r16 vs r64 on 2 maps if time allows; defer r4. The 6 h figure is unverified |
| 4 | Trained in full beside LoRA | Control MLP, positions, input projection, noise embedding at 1e-4 | All frozen in the primary arm; expanded arm separate at 1e-5 | Freeze them in the primary U-Net arm. Name the expanded arm separately and define "input projection" (context channels only, or the whole conv). The page's "under 1 percent" is false for PixArt: the control MLP plus positions at width 4096 hold 16,994,304 params, about 2.8% of about 612M. Live adapters over the EMA base are primary; the page's 0.999 EMA is a diagnostic. Pre-register the policy |
| 5 | Full fine-tune reference | October, one map, 1000 updates, optionally Monday (about 1 h) | Not under this round's constraints | Drop the Monday option. Full tuning is not a guaranteed upper bound (DiffFit 15.39 beats full tuning's 16.59), and one map at one budget cannot show "LoRA recovers X" |
| 6 | Backbones / maps for the draft | U-Net 4-map pilot as a preliminary figure; optionally all 18 by Monday | 3-map pilot with 2 seeds; everything else October | **Agrees: U-Net only in the deadline paper.** Commit only the pilot, and never expand because the result looks good. Keeps 3 seeds in October and a fixed 2000-update data curve (the page uses 1000; either is fine if fixed and labeled). The schedule is the Sunday advisor draft, not "Sep 30" |
| 7 | Cost target | Half the training-map gain, "about +0.5 dB, +0.02 LPIPS", revisable after the pilot | Aggregate latent R ≤ 0.90, pixel gains and guards alongside | Freeze the threshold now; a pilot-chosen threshold makes the pilot maps development data. The recomputed half-gain is 0.4617 dB and **0.0134** LPIPS, not 0.02. Define step, data and compute cost, censoring, zero-cost maps, guards at the crossing, and a threshold-free summary. Add motion and zero-shot error as covariates, not only family |
| 8 | Go/no-go | Go, Superman only | Conditional go, 48 card-h cap including eval | Requires global batch 32 with no accumulation (use a multi-GPU group if 16 GB forces micro-batch 8), explicit EMA init with a step-0 equivalence check, verified trainable-param lists and adapter save/reload, frozen splits/manifests/threshold, measured train and score throughput, working W&B, and source retention with the adapter on |

**Other page claims Astra flags as wrong (round 2):**
1. Line 23, "the rest are facts": fit, runtime, staging and card availability are unverified, and line 26 is a hypothesis.
2. Line 53: the restricted draw gives about 102 windows, not 256.
3. Line 59: "under 1 percent" is false for PixArt (16.99M in the control path alone).
4. Line 61: Astra says "2× pretraining LR is the usual convention" lacks support. *Supervisor note: this misreads the page; see section 5.*
5. Lines 63 to 65: batch 8 on an A4000 without accumulation gives 16k examples in 2000 updates (about 3.2 passes over 5k windows). The page's "about 13 passes" needs batch 32 (64k).
6. Lines 75 to 86: 30 to 40 min per 1000 updates gives 60 to 80 min to 2000 updates. Add 4 evaluation points at 6 + 9 min each (60 min) for 120 to 140 min per map-seed, so the 4-map, 2-seed pilot is 16 to 18.7 card-h, not about 12, before retention reads and rollouts.
7. Lines 28 and 114: distance ranks are not uniformly 0.86 to 0.90; pixels vs SD3.5 is 0.7704.
8. Line 54: arenas 8/6/7 are not "where the model already does best". Their PSNR gains are −0.922, −0.313 and +0.052, and all three lose in LPIPS.
9. Line 93: +0.02 LPIPS is not half the training gain (0.0134).
10. Line 58: frozen weights do not preserve the pretrained function after adaptation. Zero-init LoRA preserves it only at step 0.

Omission: the page does not state the transductive-D caveat.

**Question 1, final (Astra's words, condensed):** Headline reference is raw copy-last persistence, with LPIPS gain as the leading per-map panel (PredNet, arXiv 1605.08104). Beside it must sit the true-frame reconstruction reference, decoder-free latent skill, and calibrated control and rollout diagnostics, following OccWorld's reconstruction/persistence/prediction layout (arXiv 2311.16038). Persistence is a minimum prediction reference, not evidence of a useful simulator.

## 4. Disagreements, both positions

| Topic | Design page | Astra |
|---|---|---|
| Split | 6/4 and 12/8, old draw restricted | 6/2/2 everywhere, new manifest, extra arena episodes for replication |
| Pilot maps | 17, 24, 26, 23 | 8, 31, 23 |
| Control path, input projection, noise embedding | Trained in full with LoRA | Frozen in the primary arm; tested as a separate expanded arm |
| Adapter weights scored | Live and 0.999 EMA both | Live primary, EMA diagnostic, pre-registered |
| Cost target | Half the training-map pixel gain, revisable after the pilot | Latent R ≤ 0.90, frozen now, with censoring and a threshold-free summary |
| Full-FT reference | Optional Monday run | Not before October |
| Batch | Fill the card, micro-batch about 8 on an A4000, no accumulation | Global batch 32 fixed; multi-GPU group if needed |
| Rank ablation | r4 and r64 | r64 only for now |
| Data-curve budget | 1000 updates | 2000 updates |
| Seeds in October | 1 per point, pilot decides | 3 per map |
| Statistics | Spearman of cost on D, bootstrap over maps, family control | Adds motion and zero-shot-error covariates, an incremental test, censoring-aware concordance and leave-one-map-out prediction |

The batch disagreement is also a conflict between two standing rules. "Fill the card, no accumulation" on a 16 GB A4000 gives micro-batch about 8. "The recipe's global batch stays fixed" gives 32. CLAUDE.md says the global batch stays fixed, which supports Astra. Only Rohan can settle whether a 4-card DDP group per adapter run is acceptable on Superman.

## 5. Supervisor verification

Checked against files and recomputed. Confirmed unless stated otherwise.

- **Per-map gains** (recomputed from `results/distance_study/scores/040-unet-nexttic_snap_0200000_ema_ddim10/*_h1/metrics.json`, gain = `psnr_raw − persist_psnr_raw`):
  - Training maps 2 to 5: mean +0.923418 dB and +0.026839 LPIPS. Half is 0.4617 dB and 0.0134 LPIPS, so the page's "+0.02 LPIPS" is wrong.
  - Unseen maps: 9/26 PSNR wins and 0/26 LPIPS wins. VAE reconstruction 23.907 vs 23.714.
  - Map 17 +0.528; 24 −2.445; 26 −3.981; 23 +0.172; 8 −0.922; 6 −0.313; 7 +0.052.
  - VAE minus persistence: 26 −2.161; 24 +0.144. Maps with reconstruction below persistence: exactly 19, 25, 26.
- **Partial Spearman −0.7337** matches `figure_unet_h1/stats.json`. The −0.4227, −0.4313 and −0.4524 variants were not recomputed.
- **Distance rank agreement**, recomputed from the three `distances_*.json` files over 30 maps: SD1–pixels 0.8625, SD1–SD3.5 0.9026, pixels–SD3.5 0.7704. The page's "0.86 to 0.90" holds only for pairs that include SD1.
- **Eligible maps:** confirmed from `splits/index.json` and the `cluster` field. Maps 16 and 17 are arenas with 10 episodes; 6 to 8 are arenas with 20; the 13 campaign maps have 10. The brief's "18 campaign/other-WAD plus 3 arenas" is wrong.
- **EMA init trap:** confirmed. `train_wm.py:465` loads `ck["model"]` into the live model; `ck["ema"]` only seeds the EMA shadow (`train_wm.py:700`). An adapter run started with `--init-from` would train on the non-EMA weights.
- **Per-window ratio:** confirmed. `eval_tf.py:53` `latent_mse_ratio` is elementwise per window, so the reported number is a mean of ratios. `eval_tf.py:276` draws indices over the loaded split.
- **PixArt control-path size:** confirmed from `backbones.py:172-174` and `:474` (width = `caption_channels`). At 4096: 19·4096+4096, plus 4096²+4096, plus 32·4096 = **16,994,304**. The ~612M backbone total and the 4096 config value were taken from Astra and the HF config, not re-downloaded.
- **Compute arithmetic:** confirmed from page lines 74 to 79. 30 to 40 min per 1000 updates plus 4 × (6 + 9) min of scoring gives 120 to 140 min per map-seed, and the pilot is 16 to 18.7 card-h, not 12. The page's time estimate also assumes batch 32 at A6000/3 rates while the batch row says micro-batch 8, so the table is internally inconsistent.

**Where Astra was wrong or imprecise:**
- **Map 8 D.** Astra gave about 0.115; `distances_sd1.json` gives **0.1195**. Maps 31 (0.2104) and 23 (0.3709) are right. The slip does not change the pilot's spread.
- **Round-2 correction 4 misreads the page.** Page line 61 says 1e-4 "is the absolute default of the diffusers, cloneofsimo, GameFactory and DreamGen LoRA recipes (the cited world-model recipes use 5 to 10 times their pretraining rate; ours is 2 times 5e-5)". The page does not claim 2× is a convention. Astra's Vista (5×) and DiffFit (10×) evidence agrees with the page's own parenthetical.

**Not verified by the supervisor:**
- The arXiv table values (DiffFit 81.25/81.31/16.59/15.39, AVID 23.8/25.8, AdaWorld 800 updates, XEWorld r = 0.812 and p = 0.075, Westny Table 2) come from Astra's PDF reads and were not checked here. XEWorld 2608.05799 and Westny 2606.30777 postdate the supervisor's knowledge.
- The 48 card-hour pilot cap and the 540k-update October total are Astra's proposals, not measurements.

**Also noticed (not raised by Astra):**
- Page line 65 says "about 13 passes over one episode's windows". With 6 adaptation episodes (about 30k windows), 2000 × 32 = 64k examples is about 2.1 passes over the adaptation set, which is the more relevant exposure figure.
- The brief says the training maps have 4 evaluation episodes each. `splits/index.json` lists 25 `val` episodes per training map (maps 2 to 5), and the 4-episode rows are the seeded `seen` replications. This matters for sizing the forgetting guard.
