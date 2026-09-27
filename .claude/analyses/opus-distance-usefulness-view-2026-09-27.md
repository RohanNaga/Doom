# Opus view: is the distance useful, and how to adapt it (2026-09-27)

Recomputed from the repo files; the main table reproduces. Covariates: scene persistence PSNR, `context_motion`. The censored arenas share one censoring time, so a tied top rank is exact.

## Answers

1. No distance orders the arenas on any outcome; the zero-shot latent skill S0 does (A4000 ρ +0.89, cost −0.65), and D survives only as a family separator.
2. The x axis is S0 (or the deficit from home, 2.96 dB minus S0): one forward pass over footage that adaptation needs anyway.
3. Stop building map-level footage distances; make the scenario a window: stratified per-window coverage tested against per-window zero-shot skill, about one CPU day.
4. Claim D as a family separator with the failed ordering criterion stated, give coverage and G one sentence, and put the arena ordering in the model's zero-shot skill.
5. A retexture-by-geometry factorial plus distance-stratified generated arenas, about 80 A4000-hours.

## 1. Ordering

| | A0 | S0 | cost τ (conc.) | A4000 | gain | LOO cost / A4000 | partial on cost given S0 + motion (perm p) |
|---|---|---|---|---|---|---|---|
| D | −0.24 | −0.05 | −0.21 (0.38) | +0.06 | +0.42 | 1.01 / 1.08 | −0.29 (0.31) |
| coverage | −0.17 | −0.01 | −0.19 (0.40) | +0.12 | +0.26 | 1.11 / 1.23 | −0.55 (0.05) |
| G | +0.02 | +0.14 | +0.01 (0.51) | +0.02 | +0.12 | 1.51 / 1.53 | +0.54 (0.06) |
| S0 | +0.74 | | −0.47 (0.76) | +0.89 | +0.38 | 0.62 / 0.24 | |

Concordance: share of pairs where the lower distance (higher S0) crosses first. LOO: leave-one-arena-out error over predict-the-mean; log2 cost, censored at 8,000.

Covariates rescue nothing (D against cost −0.06 given both; interval [−0.77, +0.37]). The pre-registered incremental test fails for D: it moves the S0-plus-motion baseline's LOO from 0.46 to 0.48. Coverage and G reach p ≈ 0.05 in opposite directions (less covered crosses faster; larger G crosses slower) among twelve tests; neither is a finding. The one hint is that farther arenas recover more (D against gain +0.42, [−0.17, +0.83]; +0.27 without arena 7). S0 strengthens under both covariates (−0.81 cost, +0.93 A4000; intervals [−0.87, −0.20] and [+0.58, +0.98]). The ordering is real: PixArt's zero-shot A0 ranks the arenas like the U-Net's (+0.95), and D misses it (−0.37).

Why coverage and G fail, from their own columns:
- **Floor.** Coverage leaves 12 of 13 arenas at or below the training maps' held-out maximum; `without_state` leaves all 13. State alone almost separates (11 of 13). G has 13 and 14 inside, 11 and 17 within a bootstrap SD (the numbers file says four inside).
- **They measure motion.** Coverage tracks persistence PSNR at −0.69, G at +0.63, hence their mutual −0.53: small innovations sit in the memory's dense core.
- **Coverage barely sees transitions.** Shuffling innovations adds 0.004 to 0.010 against a 0.087 range.
- **G's predictor has no skill.** The kNN predictor is worse than copy-last even with training memory on training-map episodes (L_train +0.006 to +0.017 log10), while the world model's zero-shot skill is 1.14 to 1.96 dB (2.96 home).

## 2. The x axis

S0. A distance needs the same target footage and saves only one forward pass, worthless when you are about to fine-tune. Ben-David 2010 bounds target error by source error plus divergence plus λ; S0 measures the target error itself, as LEEP (2002.12462) does for classifiers. Decoded A0 cannot replace it (LOO on cost 1.23). Caption caveats: S0 predicts the endpoint because arenas keep their rank (S0 against S4000 +0.85), not the gain (LOO 1.06), and resampling episodes moves the cost 2 to 8 times on five arenas.

## 3. Adapting the metric

The NVIDIA recipe assumed appearance was causal; here one WAD's textures vary mainly between families (D's job), the pooled latent cannot express transitions, and cost depends on target sample complexity too (Hanneke-Kpotufe 2002.04747).

v3 keeps the per-scenario design with the window as the scenario:
- **Features and strata.** State plus innovation; control class and motion decile become strata, not metric blocks, because those blocks concentrate distances.
- **Direction and memory.** Target covered by source (Mensink 2103.13318, Westny 2606.30777), training episodes only, floor from training maps' held-out windows in the same stratum.
- **Windows and score.** The scored windows themselves, log(d_k / stratum floor), under fixed training stratum weights.
- **Validation before freezing.** Family floor 4 of 4 against 13 of 13; the shuffle moves the score by more than one arena SD; split-half reliability at least 0.8; window level, log latent ratio ~ v3 + log copy-last error + arena fixed effect over about 3,800 windows with p < 0.01; only then map level.

Cost: half a day of code, 2 to 3 CPU hours. If the window test fails, stop; U-Net-feature coverage needs the model like S0, so it only attributes.

## 4. Paper text for Sep 30

"A per-frame latent distance to the training footage separates the four training maps from all 13 unseen arenas (training floor at most 0.093; arenas 0.118 to 0.270, stable across episode subsets) but does not order the arenas: it tracks neither zero-shot skill, adaptation cost nor the adapted endpoint (|ρ| ≤ 0.27, n = 13), failing the criterion we fixed before the adaptation runs, and two transition-level variants frozen in advance do not even separate the families. The ordering is shared by two backbones (ρ = 0.95) and carried by the model's zero-shot latent skill, which predicts the adapted endpoint (ρ = 0.89) and the half-gap cost (ρ = −0.65) from one forward pass."

## 5. October

- **Appearance arm.** Each training map at three graded Freedoom retexture levels, with geometry and bots fixed.
- **Geometry arm.** Twelve Obsidian arenas with training textures, chosen in D and v3 terciles before scoring.
- **Pre-registered prediction.** Appearance shift lowers S0 but recovers fast (arena 7: lowest A0 0.98, crosses at 250, pan kept while textures drift to grey brick), and geometry sets the cost. A distance earns its place only if it predicts cost within the geometry arm.
- **Cost.** Two seeds; 576 episodes (about 135 GB raw, so Spiderman, raw deleted after encoding), about 10 A6000-hours to encode, 80 A4000-hours to adapt and score. Retexture tooling (omgifol) VERIFY.
