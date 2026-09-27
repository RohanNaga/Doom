# Fresh-set distances against the adaptation outcomes (2026-09-27, 05:35 EDT)

Inputs: `results/transition_distance/compare_fresh_sd1.csv` (distance chain on Superman, commit 4ee0c5c, 04:57; coverage k = 3 distinct episodes, transfer gap k = 16, 1,000 bootstraps, memory = the four training maps' training episodes, targets = their validation episodes and the 13 fresh arenas), `results/distance_v2/` (frozen frame distance D recomputed on the fresh 24-episode set), and the 13 seed-0 adaptation curves `results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl` (live weights, stock decoder, scene crop). Script: `tools/distance_outcome_corr.py`. Home 4.138 dB; cost = first grid step with A >= (A0 + home) / 2; censored runs set to 8,000 for the rank correlation.

## The table

| arena | D | coverage | G | A0 | A4000 | gain | cost | skill0 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.172 | 0.551 | 0.038 | 1.66 | 2.71 | 1.05 | censored | 1.31 |
| 6 | 0.198 | 0.574 | 0.014 | 1.08 | 2.61 | 1.53 | 4000 | 1.14 |
| 7 | 0.270 | 0.602 | 0.020 | 0.98 | 3.37 | 2.39 | 250 | 1.42 |
| 8 | 0.118 | 0.541 | 0.019 | 1.84 | 2.85 | 1.01 | censored | 1.36 |
| 9 | 0.184 | 0.533 | 0.025 | 2.43 | 4.20 | 1.77 | 250 | 1.86 |
| 10 | 0.155 | 0.533 | 0.022 | 1.79 | 3.72 | 1.92 | 250 | 1.73 |
| 11 | 0.156 | 0.613 | 0.013 | 2.01 | 3.75 | 1.74 | 250 | 1.80 |
| 12 | 0.138 | 0.526 | 0.018 | 1.56 | 2.51 | 0.95 | censored | 1.21 |
| 13 | 0.154 | 0.583 | 0.009 | 1.95 | 3.18 | 1.23 | 2000 | 1.42 |
| 14 | 0.147 | 0.551 | 0.011 | 1.63 | 2.99 | 1.36 | 2000 | 1.25 |
| 15 | 0.127 | 0.549 | 0.014 | 2.63 | 3.90 | 1.27 | 500 | 1.96 |
| 16 | 0.171 | 0.559 | 0.015 | 2.25 | 2.97 | 0.72 | censored | 1.68 |
| 17 | 0.130 | 0.573 | 0.013 | 1.59 | 3.14 | 1.54 | 2000 | 1.52 |

## Spearman with the outcomes (n = 13)

| predictor | A0 | A4000 | gain | cost | skill0 |
|---|---|---|---|---|---|
| frame distance D | -0.24 | +0.06 | +0.42 | -0.27 | -0.05 |
| directed coverage | -0.17 | +0.12 | +0.26 | -0.22 | -0.01 |
| transfer gap G | +0.02 | +0.02 | +0.12 | -0.01 | +0.14 |
| coverage, state block only | -0.25 | +0.13 | +0.38 | -0.36 | +0.08 |
| zero-shot latent skill | +0.74 | **+0.89** | +0.38 | **-0.65** | 1 |

No p below 0.15 in the first four rows; skill0 against A4000 p = 5e-5, against cost p = 0.02.

## The family floor (the pre-registered gate, on the fresh set)

- D: every training map's frozen D (max 0.093) lies below every unseen arena's fresh D (min 0.118, arena 8). Passes.
- Coverage: training maps' validation episodes 0.524 to 0.609; unseen arenas 0.526 to 0.613. Fails: a held-out episode of a training map is as uncovered by the training transitions as an arena the model never saw.
- G: training 0.005 to 0.012; unseen 0.009 to 0.038. Fails (arenas 13 and 14 sit inside the training range; 11 and 17 are above it by less than one bootstrap sd).
- Coverage and G disagree with each other on the unseen arenas (Spearman -0.53) and with D (+0.37, +0.36).

## What it says

The two transition-space candidates are worse than the frame distance at the one thing the frame distance does (separating the training family from the unseen family), and none of the three orders the thirteen unseen arenas on any outcome. The quantity that orders them is the model's own zero-shot latent skill on the arena, which needs one evaluation pass and no training: it predicts the endpoint (rho 0.89) and the crossing budget (rho -0.65). The distance section of the paper therefore reports D as a family separator with the failed ordering gate stated, drops coverage and G to one sentence (tried, do not separate families), and puts the zero-shot skill on the x axis of the adaptation figure.
