1. **Usefulness:** No distance reliably orders adaptation; coverage has a weak incremental signal worth retesting.
2. **Axis:** Use zero-shot latent deficit for the predictive scatter; keep updates on the learning curves.
3. **Adaptation:** Replace unconditional proximity with a calibrated, control-conditioned innovation transfer gap.
4. **September:** Retain descriptive distribution shift; withdraw predictive-distance claims and disclose failed gates.
5. **October:** Freeze the metric, then predict costs on new distance-stratified generated arenas.

**Reproduction [R].** Ran `/opt/miniconda3/envs/PERSEVE/bin/python tools/distance_outcome_corr.py`; extended independently with `/opt/miniconda3/envs/PERSEVE/bin/python /tmp/astra_distance_review.py`. Inputs: `results/transition_distance/compare_fresh_sd1.csv` columns `D,coverage,G`, their `_sd` columns, and ablations; `results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl`, live/stock rows, fields `heldout_A_stock`, `heldout_latent_skill`, `heldout_latent_mse`; `results/fresh_rescore/unet200k_ema/map??/metrics.json`, `persist_psnr_raw.mean` and `context_motion.mean`. All statistics below are [R]; proposed budgets are estimates.

**1. Does any distance order the arenas?**

Across the 13 arenas, entries are Spearman / partial-on-persistence / partial-on-context-motion; partials correlate residualized ranks. A0/A4 are scene advantages; S0 is latent skill. C is the observed cost tier, with noncrossers tied above all observed crossings.

| Predictor | A0 | S0 | C | A4 |
|---|---|---|---|---|
| D | −.24/−.23/−.17 | −.05/−.05/−.02 | −.27/−.27/−.18 | .06/.06/.03 |
| Coverage | −.17/−.14/−.25 | −.01/−.10/−.03 | −.22/−.19/−.34 | .12/.03/.14 |
| G | .02/−.05/.18 | .14/.26/.21 | −.01/−.12/.20 | .02/.14/−.03 |

Using home 4.138 and the brief's half-gap rule, arenas 1/8/12/16 are right-censored. True uncensored-cost Spearman is unidentified; the table describes observed tiers, not imputed 8,000-step crossings. Concordance on 63 identifiable pairs, with larger distance predicting slower crossing, is .381/.397/.508 for D/coverage/G.

Leave-one-arena-out OLS RMSE, with intercept, versus the other arenas' mean:

| Predictor | A0, dB | S0, dB | min(cost,4000), thousands | A4, dB |
|---|---:|---:|---:|---:|
| Mean | .494 | .280 | 1.760 | .551 |
| D | .461 | .290 | 1.912 | .573 |
| Coverage | .533 | .311 | 1.839 | .610 |
| G | .512 | .322 | 2.046 | .681 |

Restricted cost is fully observed despite censoring. D's modest A0 improvement disappears without arena 7: .528 versus mean .443. No standalone distance earns a predictive claim. Median bootstrap SDs for D/coverage/G are .00216/.00242/.00110; precise measurement is not predictive validity.

For the pre-registered incremental question, operationalized as zero-shot latent MSE plus motion, restricted-cost RMSE is 1.840; adding D/coverage/G gives 2.322/1.684/1.792. Coverage's partial cost rho is −.391, opposite the intended direction; rank-residual Freedman–Lane p=.236 (9,999 permutations, seed 0). Endpoint RMSE improves .460→.428 with coverage, but partial rho=.206, p=.545. These exploratory improvements deserve reporting, not “adds nothing” or a passed gate. This small, single-seed study cannot establish equivalence.

**2. What should the x axis be?**

Choose home latent skill minus S0. S0 has rho=.890 with A4 and −.648 with C; its standalone RMSE is .269 dB and 1.327 thousand updates, respectively. Decoded deficit 4.138−A0 has cost rho only .176. Keep decoded advantage as the outcome and the existing crossing rule. A reader can obtain S0 from one evaluation pass before adaptation; D/coverage/G also require target recordings. Independent diagnostic episodes are preferable because A0 enters the crossing threshold. This is an exploratory model-specific predictor, not a dataset distance.

**3. How should the metric be adapted?**

Coverage is reproducible (memory-half rho=.989, `coverage_sd1.json`, `points[].half_a/half_b`), yet innovation shuffling raises it only 0.7–1.8%; ablation outcome correlations stay below .36 in magnitude. Median `state_share`=.355 is not an additive attribution: ablations retrieve different neighbors. In `transfer_gap_sd1.json`, source control fallback spans 12–78%; every unseen `L_train` is positive, meaning worse than persistence. G also uses 16 own episodes against adapters' eight. It compares weak predictors; this does not refute transition-based distance.

Propose **G\***: retrieve using scene state, lagged dynamics and the complete executed-control history; predict next-latent innovation with local ridge regression, replacing unconditional neighbor averaging. Score source-minus-own prediction MSE divided by aggregate persistence MSE, within source-defined motion/control strata. Never silently drop controls; report unsupported mass separately. Use equal episode weights, with motion weighting only as sensitivity.

Use training episodes as source memory; validation only calibrates, with train-plus-validation an explicit sensitivity. Match source/own memory sizes and own memory to the actual adaptation episodes. Compare 64 versus 128 nonoverlapping windows/episode before freezing. Fit scaling, neighborhoods and regularization on source-only episode folds; require improvement over persistence, temporal/control shuffle sensitivity, memory stability and inspection of retrieved transitions. Freeze representation, sampling, source checkpoint, adapter recipe and outcomes before prospective testing on new maps; these arenas are now development data. Budget: 4–16 CPU-hours, no GPU inference with cached latents; unbenchmarked.

Direction follows Mensink (2103.13318) and Westny (2606.30777); task-aware comparison follows OTDD (2002.02923) and s-OTDD (2501.18901). Neither those results nor transfer-exponent theory (2002.04747) guarantees optimization cost.

**4. What should the paper say on Sep 30?**

“Frame distance separates fresh arenas from the source resampling floor; the earlier pooled association mixes training/unseen and arena/campaign contrasts and does not establish ordering within unseen arenas. The original pre-registered gate failed repeatability and family-order checks; fresh transition coverage and transfer gap also fail complete training/unseen separation and provide no validated adaptation-cost predictor. We therefore withdraw the predictive-distance claim and report the model's zero-shot diagnostic exploratorily.”

Audit qualification: `results/distance_v2/distances_sd1.json` has no validation-map D; its `.0934` maximum is `floor.motion.values`, below fresh `maps[].D` minimum `.1179`, not a newly measured validation separation.

**5. October: the settling experiment.**

Generate 48 arenas, varying geometry and textures independently; select across frozen distance and motion strata before outcomes. Reserve 16 for calibration, 32 sealed; run two adaptation seeds to 8,000 updates with fixed episodes/recipe, dense early reads, and independent diagnostic/evaluation recordings. Predict restricted cost and endpoint against mean and S0-plus-motion baselines; bootstrap maps, retain censoring, require improvement on sealed maps. This tests within-domain transferability rather than importing XEWorld's appearance finding (2608.05799) or LEEP's convergence evidence (2002.12462).

Training estimate: 48×2×8,000×.993 seconds/update ≈212 A4000-hours; .993 is the median `(end.time−start.time)/4000` from the seed-0 `log.jsonl` files [R]. Budget another 60 GPU-hours for encoding/evaluation, subject to a pilot benchmark.
