# Independent statistics and figure review — Astra, 2026-09-27

1. **Claim:** describe these held-out arenas; a “typical new arena” inference requires an explicit within-WAD exchangeability assumption.
2. **Body figures:** IQM learning curve plus cumulative half-gap attainment; put individual curves, profiles and predictor scatter in the appendix.
3. **Statistics:** preserve arena/episode pairing, retain censored targets, and distinguish measured wins from equivalence.
4. **Runs:** finish the queued grid and seed checks; figure repair requires no additional training.
5. **Pitfalls:** distance colouring, threshold coupling, coarse-grid ties and unqualified budget precision currently overstate the evidence.
6. **Whole paper:** use matched comparisons, explicit references and readable small multiples; separate truth for different maps.

## 1. What is the sampling unit?

Say: **“Across the held-out arenas of this WAD, adaptation improves next-tic prediction, with heterogeneous recovery budgets.”** This is a benchmark distribution: n=13 arenas, one WAD, one backbone-training realization. Within an arena, n=8 held-out episodes, not 256 independent windows. Training-seed variability is separate. If these are the complete held-out arena set, their descriptive count has no arena-sampling error; arena bootstrap intervals describe sensitivity to an assumed target mixture, not guaranteed generalization to new WADs.

[Agarwal et al., 2108.13264](https://arxiv.org/abs/2108.13264), Figures 7/10 and Appendix A.22, distinguish score profiles, IQM curves, and resampling tasks/runs. Here episodes cannot substitute for training runs. [Taylor–Stone, JMLR 10 (2009)](https://jmlr.org/papers/v10/taylor09a.html) distinguish jumpstart, asymptote, area and time-to-threshold. [Kaplan–Meier, JASA (1958)](https://doi.org/10.1080/01621459.1958.10501452) supplies the censoring framework; our event is **first observed grid attainment**, not an observed continuous training-time crossing.

[AdaWorld, 2503.18938](https://arxiv.org/abs/2503.18938), Figure 6, facets PSNR-versus-update curves by environment and data budget. [XEWorld, 2608.05799](https://arxiv.org/abs/2608.05799), Figure 4/Table 4, separates target recovery and seen-robot forgetting. [Meta-Dataset, 1903.03096](https://arxiv.org/abs/1903.03096), Figure 2 and appendix shots curves, conditions performance on support size and evaluation dataset. Follow that conditional language. [Vista, 2405.17398](https://arxiv.org/abs/2405.17398) demonstrates cross-dataset transfer; it is not evidence for per-target adaptation-cost statistics.

## 2. Exact adaptation layout

Use the planned **Figure 4**, two equal-width panels: `[a: attained quality | b: recovery budget]`.

**4a:** x=updates, linear from zero to the common budget; an early-budget inset resolves rapid changes. y=scene A, dB. Faint grey arena trajectories and measured-point dots; thick blue IQM; translucent pointwise nested-bootstrap band; dashed, directly labelled home reference. Remove distance colours. Caption: **“Adaptation raises prediction quality across held-out arenas, while substantial differences between arenas remain.”** State live adapters, tuned decoder, support size, one training seed and bootstrap units.

**4b:** x=updates on the same linear scale; y=“fraction first reaching half gap”, 0–1. Plot the right-continuous step function 1−KM, with pointwise band, median guide and an at-risk row. Label the terminal count and “four censored”; censor marks do not create an attainment jump. Caption: **“Nine arenas first reach their own half-gap target within the measured budget; four remain unresolved.”** Do not interpolate crossings or extrapolate the tail.

**Appendix:**

- Profiles: x=absolute A threshold, dB; y=fraction of arenas with A≥threshold; directly labelled curves for 0/500/4k updates. Caption: **“Fixed-budget performance profiles expose the complete target distribution without arena-specific thresholds.”**
- Small multiples: arena-ID order, shared axes, A and M rows, individual half-gap/home and M=0 references, paired episode intervals; no smoothed curves. Caption: **“Arena-level trajectories distinguish uncertain threshold crossings from persistent deficits.”**
- Predictor scatter: x=Δ=S_home−S0, dB; y=A4k, dB; label every arena, show paired episode uncertainty, no fitted prediction claim. This simple centering gives rho=−0.896 [A,H]; it is not the separately proposed motion-calibrated deficit. Caption: **“Zero-shot skill is associated with endpoint quality within this benchmark.”**

## 3. Recomputed statistics and wording

All intervals below are nominal 95% percentile intervals under the stated resampling assumptions, conditional on frozen home and the trained models.

| Updates | 0 | 250 | 500 | 1k | 2k | 4k |
|---|---:|---:|---:|---:|---:|---:|
| IQM A, dB [A] | 2.032 | 3.219 | 3.404 | 3.503 | 3.665 | 3.786 |
| Cumulative crossings /13 [A,H] | 0 | 4 | 5 | 7 | 9 | 9 |
| At risk just before read | 13 | 13 | 9 | 8 | 6 | 4 |

IQM uses `scipy.stats.trim_mean(x,.25)`: discard three values from each tail, average seven. Across-arena endpoint intervals are [1.676,2.384] at zero and [3.513,4.260] at 4k; nested arena/episode intervals are [1.682,2.373] and [3.519,4.265]. The band measures uncertainty in the aggregate, not the spread of individual arenas.

Exact headline replacements:

- **“Nine of thirteen arenas cross by 4k, five by 500; the estimated attainment fraction at 4k is 69.2% (arena-bootstrap interval 46.2–92.3%).”** [A,H] Nested resampling gives the same rounded terminal interval. Censored IDs: 1,8,12,16.
- **“The median first-observed budget is 1k updates; its bootstrap interval runs from 250 to beyond 4k.”** [A,H] Both resampling schemes leave the upper endpoint unidentified. Never calculate the median only among successful arenas; use the survival median, not IQM of budgets.
- **“Median A increases from 2.036 to 3.799 dB; the median paired gain is 1.768 dB (nested interval 1.545–2.272).”** [A] Marginal nested median intervals are [1.722,2.424] and [3.450,4.376]. Retain medians for “typical arena”; use IQM for the learning curve. Difference of medians is not median paired gain.
- **“At 4k, eleven arenas have lower mean scene LPIPS than raw persistence; median M=−0.0215 (arena interval −0.0404 to −0.0148).”** [M] The win fraction’s arena interval is 61.5–100%; nested interval 69.2–100%. Episode intervals for the two positive-margin arenas include zero; that does not establish equivalence. Remove “tie” without a prespecified equivalence margin.
- **“On four selected arenas, one episode recovers 74.8–101.1% of the sixteen-episode gain at 4k using the stock decoder.”** [L] In arena order 7/8/12/16, shares are 90.56/101.10/74.77/93.67%; endpoint differences are −0.2314/+0.0112/−0.2528/−0.0473 dB. Thus “within 0.25” needs “about 0.25”. Paired episode intervals are [−0.440,−0.030]/[−0.049,+0.072]/[−0.532,−0.038]/[−0.086,−0.012]. Selected support sets and arenas do not justify a population data-efficiency interval.
- **“Exploratory S0–A4k rank association is 0.896 (arena interval 0.656–0.966; nested 0.566–0.951).”** [A] Splitting sorted held-out episodes alternately between predictor and endpoint gives 0.885 and 0.791 after swapping halves. The descriptive budget association is −0.607, with all censored observations tied above crossings; it is not a censoring-adjusted correlation with true adaptation time.

Profiles [A], at thresholds 3/4/5 dB: zero-shot counts are 0/0/0; 500-update counts 11/3/0; 4k counts 13/4/1. At 4 dB the endpoint fraction is 30.8%, arena interval 7.7–53.8%. Boundary bootstrap intervals can degenerate; they cannot certify universal success.

## 4. What to finish

The denser grid resolves early ties; 8k extends follow-up rather than proving an asymptote. Promote only a complete, consistently scored grid, including guards; do not splice runs. Plot both seeds separately on their tested arenas; the current stock endpoint differences span absolute 0.0195–0.0575 dB [L], with identical adaptation episodes. That checks optimizer randomness, not support-set randomness. The nested data ladder belongs in the appendix with its fixed update budget.

**Additional training requested: 0 GPU-hours.** Finish existing commitments. Arena 7’s logged start-to-end duration is 1.097 GPU-hours for 4k [C]; linear planning estimates are 28.53 GPU-hours for thirteen 8k runs and 8.78 for four seed repeats, excluding subsequent evaluation. Full fine-tuning needs its own measured cost; LoRA timing cannot price it. Missing tuned ladder/seeds and complete guards limit the claims until their reads arrive.

## 5. Reviewer objections addressed

The current D-coloured spaghetti and terciles emphasize a predictor that does not order recovery; aggregate bands plus visible individuals answer the actual question. Half-gap targets depend on A0, and S0 correlates with initial difficulty: avoiding an explicit A0–cost correlation does not remove coupling. Absolute profiles supply a threshold-independent check. Shared WAD and source checkpoint limit independence; no bootstrap manufactures new families or training seeds. Four crossings at 250 are resolution ties, not identical adaptation speeds. Table 3 must not mix pooled medians with a selected arena’s comparator, label GPU-hours per arena, or equate censored budgets with failures forever.

## 6. Every other data figure and table

**Rollouts, Figure 2.** [GameNGen, 2408.14837](https://arxiv.org/abs/2408.14837), Figures 4/8, aligns trajectory rows; [DIAMOND, 2405.12399](https://arxiv.org/abs/2405.12399), Figures 3/10, labels sampled times and method rows; [Genie, 2402.15391](https://arxiv.org/abs/2402.15391), Figure 1, branches trajectories from a prompt. [Oasis’s project page](https://oasis-model.github.io/) uses animated galleries, not a paper-style quantitative rollout grid. Use one held-left-turn example in the body, forward in appendix; proposed columns t=0,4,8,16,32 tics. Separate map-2 truth/model from arena-7 truth/zero-shot/adapted/copy-last: different maps cannot share truth. Use `paper/figures/figure1/manifest.json` entries by `map,pattern`; tuned adapter exports remain pending. Caption: **“Matched control sequences reveal appearance drift and the effect of adaptation.”** Omit per-frame PSNR clutter; put aligned per-tic curves in appendix, computed only against matching executed controls.

**Family step, Figure 3a.** XEWorld Figure 5 uses labelled robot points, appearance/workspace panels, linear axes and fitted lines. Borrow labels, not its regression: use three aligned backbone facets, linear D versus latent skill, training/unseen marker shapes, and a vertical train–train D-floor band labelled with recording provenance. No connections between backbones or across maps; no fitted discontinuity. Caption: **“Training and held-out arenas occupy different regions of distance–skill space.”** Recompute the floor on matching footage before presenting it as a fresh-set calibration.

**Paired decoder panel, Figure 3b.** [Weissgerber et al., PLOS Biology (2015), Figure 2](https://doi.org/10.1371/journal.pbio.1002128) shows paired points and differences because separate group summaries hide pairing. Use horizontal dumbbells: arena rows sorted by frozen U-Net S0, shared across backbone facets; open=stock, filled=tuned; A and M in separate columns, M=0 line. Directly label arena IDs. Full grid in appendix; U-Net pair in body. Current SD3.5 tuned rows are absent: leave them absent, never synthesize pairs. Caption: **“Decoding identical predictions isolates the decoder’s contribution.”**

**Training panel.** GameNGen Figure 13 uses log-update PSNR; [DreamerV3, 2301.04104](https://arxiv.org/abs/2301.04104), Figure 6, uses return versus environment steps. DIAMOND’s main aggregates and drift plots are not PSNR training curves. Use log positive updates, backbone colour, solid EMA/dashed live, raw-persistence horizontal reference, unsmoothed measured points; ring the exact table checkpoint and label its update. Keep LPIPS in an aligned appendix panel, not a second y-axis. Caption: **“Held-out next-tic quality improves with training; marked reads supply the reported comparison.”**

**Decoder images.** GameNGen Figure 12 shows stock/tuned/truth columns with HUD artifacts visible. Use those columns, identical latent input, full frame above matched scene/HUD zooms, identical enlargement and crop boxes. Caption: **“Decoder tuning changes rendering while holding the latent fixed.”** Do not imply a predicted-latent example establishes reconstruction quality.

**Distance null.** Body sentence plus Figure 3a; appendix scatter and gate-result table, reporting failed checks and uncertainty. XEWorld plots associations; AdaWorld’s failure gallery and Oasis’s limitations demonstrate failures, not formal null tests. None licenses interpreting nonsignificance as equivalence.

**Tables.** GameNGen reports PSNR/LPIPS ablations; DIAMOND includes reference agents and aggregate uncertainty; Genie separates controllability from image quality. Table 1: model/checkpoint, next-tic PSNR/LPIPS, rollout endpoint, directional; retain raw persistence/copy-seed and decoder-reconstruction references. Table 2: model/decoder, home-or-unseen, A/M/G, directional; remove unmeasured horizon columns. Table 3: method/support, trainable parameters/GPU-hours, endpoint A/M, crossed count/median budget, forgetting/directional. All method comparisons use identical arenas. Bold only the best comparable estimate, never imply significance; no significance stars from window-level tests.

## Reproduction key

Computed with `/opt/miniconda3/envs/PERSEVE/bin/python`.

**[A]** `results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl`, maps 01,06–17, `weights=live`, steps 0/250/500/1000/2000/4000, `heldout_A_tuned`, `heldout_latent_skill`. Match the builder’s `load_runs`: select the evaluation fingerprint carrying tuned scores with most steps, latest on ties; latest row per step. Resolve each row’s `heldout_per_window` beneath its local run directory. Exclude `dup_latent`/`dup_raw`; A=`scene_psnr_dec_tuned−scene_copy_psnr_dec_tuned`; S=`−10log10(latent_mse/copy_latent_mse)`; average by episode. All retained episodes have equal window counts.

**[H]** `results/home_unet_tuned/metrics.json`, means `scene_psnr_dec_tuned−scene_copy_psnr_dec_tuned`=5.059250826; same-directory CSV supplies S_home. Freeze H; crossing=min measured step with A≥(A0+H)/2; otherwise right-censor at 4000.

**[M]** `results/fresh_rescore/adapt4000_live_tuned/map??/per_window.csv`, same maps; `scene_lpips_raw_tuned−scene_persist_lpips_raw`, same duplicate exclusions.

**[L]** same adaptation path pattern, k=1/2/4/8/16 on maps 07/08/12/16 and k8,s1 on 06/07/08/16; live `heldout_A_stock`, steps 0/4000. Ladder share=(A_k1,4k−A_k8,0)/(A_k16,4k−A_k8,0). Ladder intervals resample paired episode differences, RNG 927, 10,000 draws sequentially in arena order 7/8/12/16. **[C]** map07 k8,s0 `log.jsonl`, last `end.time−start.time`, divided by 3600.

Bootstrap: 10,000 draws, `default_rng(927)`, arenas ascending; sample indices `(10000,13)`, preserving whole curves. Nested variant restarts the RNG, draws those arena indices then episode indices `(10000,13,8)` per selected occurrence, episodes ascending, paired across budgets/metrics. Recalculate targets and statistics; percentile endpoints use NumPy’s default interpolation, except censored medians use `inverted_cdf` and infinity placeholders. These are pointwise bands, not simultaneous bands.

Citations were checked against primary pages; requested figure assets/PDF pages were fetched and visually inspected. Kaplan–Meier metadata was verified through Crossref because the publisher blocked retrieval. No other reviewer views were read. No SSH, commits, or modifications to other repository files were made.
