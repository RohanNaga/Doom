# Thirteen arena adaptation curves (U-Net 200k EMA source, LoRA r16 alpha 16, 8 episodes, stock decoder, live weights, 256 fresh held-out windows), corrected 2026-09-27 00:55 EDT

A = scene-crop (rows 0 to 207) decoded advantage over copy-last, dB, `heldout_A_stock`. Home measured the same way on the training maps' 512 validation windows with the same scorer: scene 4.138 dB (full frame 3.675). Half-gap line = (A0 + 4.138)/2. Cost = first grid step with A >= line; censored if none by 4,000. The earlier version of this file compared scene A against a full-frame home of 3.60, which inflated the crossings (12 of 13) and the above-home count (4).

| arena | D | A0 | A250 | A500 | A1000 | A2000 | A4000 | gain | line | cost | skill0 | skill4k | lpips0 | lpips4k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.115 | 1.84 | 2.48 | 2.48 | 2.58 | 2.82 | 2.85 | +1.01 | 2.99 | censored | 1.36 | 2.04 | 0.229 | 0.171 |
| 15 | 0.127 | 2.63 | 3.30 | 3.62 | 3.61 | 3.80 | 3.90 | +1.27 | 3.38 | 500 | 1.96 | 3.02 | 0.281 | 0.168 |
| 12 | 0.128 | 1.56 | 2.23 | 2.40 | 2.41 | 2.46 | 2.51 | +0.95 | 2.85 | censored | 1.21 | 2.09 | 0.228 | 0.154 |
| 17 | 0.130 | 1.59 | 2.62 | 2.72 | 2.82 | 3.04 | 3.14 | +1.54 | 2.87 | 2000 | 1.52 | 2.64 | 0.333 | 0.177 |
| 14 | 0.146 | 1.63 | 2.56 | 2.68 | 2.80 | 2.94 | 2.99 | +1.36 | 2.88 | 2000 | 1.25 | 2.01 | 0.258 | 0.178 |
| 11 | 0.150 | 2.01 | 3.23 | 3.39 | 3.53 | 3.60 | 3.75 | +1.74 | 3.07 | 250 | 1.80 | 2.72 | 0.228 | 0.140 |
| 13 | 0.156 | 1.95 | 2.70 | 2.88 | 2.96 | 3.08 | 3.18 | +1.23 | 3.05 | 2000 | 1.42 | 2.36 | 0.296 | 0.196 |
| 10 | 0.162 | 1.79 | 3.45 | 3.52 | 3.57 | 3.65 | 3.72 | +1.92 | 2.97 | 250 | 1.73 | 2.51 | 0.181 | 0.127 |
| 16 | 0.164 | 2.25 | 2.75 | 2.86 | 2.86 | 2.87 | 2.97 | +0.72 | 3.19 | censored | 1.68 | 2.44 | 0.334 | 0.170 |
| 1 | 0.179 | 1.66 | 2.28 | 2.42 | 2.54 | 2.62 | 2.71 | +1.05 | 2.90 | censored | 1.31 | 2.16 | 0.324 | 0.229 |
| 9 | 0.180 | 2.43 | 3.73 | 3.91 | 4.02 | 4.01 | 4.20 | +1.77 | 3.29 | 250 | 1.86 | 2.78 | 0.203 | 0.138 |
| 6 | 0.188 | 1.08 | 2.30 | 2.34 | 2.42 | 2.52 | 2.61 | +1.53 | 2.61 | 4000 | 1.14 | 2.23 | 0.249 | 0.158 |
| 7 | 0.282 | 0.98 | 2.94 | 3.09 | 3.16 | 3.29 | 3.37 | +2.39 | 2.56 | 250 | 1.42 | 2.13 | 0.234 | 0.158 |

Spearman over 13 arenas (tie-aware): D vs A0 -0.25; D vs A4000 +0.07; D vs gain +0.45; D vs cost (censored as 8000) -0.27; zero-shot latent skill vs A4000 +0.89; zero-shot latent skill vs cost -0.65; A0 vs cost -0.18
crossings: 9 of 13 by 4,000 (5 by 500); above the scene home at 4,000: 1; median A4000 3.14; median gain +1.36; mean gain 2,000 to 4,000 +0.092
