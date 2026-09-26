# Decision B at the pixel level: the three views and the synthesis (2026-09-26, 19:45 EDT)

Sources: `main-cost-target-view-2026-09-26.md`, `astra-cost-target-view-2026-09-26.md` (thread 01a0e00b-2be6-7820-9c60-7c5ccf1909c0), `opus-cost-target-view-2026-09-26.md`. Numbers rechecked by the supervisors against the frozen h1 metrics. All three: tuned SD 1 decoder frozen during adaptation, stock decoder as the check; scene-only (rows 0 to 207) primary, full frame as the check; cost = first grid budget crossing the line; never-crossing maps right-censored above 4,000; the whole curve, the value at 4,000 and the area under the curve reported beside the crossing.

## The three options, as they converge

| | What is measured per window | The line | In words | Who |
|---|---|---|---|---|
| **A. Decoded advantage over copy-last** | PSNR(decoded prediction, decoded true frame) − PSNR(decoded copy-last, decoded true frame); Genie's ΔPSNR form (2402.15391) | the training maps' own advantage, 3.60 dB today (56 percent of copy-last's squared error removed); milestones at 25 and 50 percent of it | how many updates until the model's edge over copying, as rendered, matches what it has at home | all three (main at half, Astra and Opus at full) |
| **B. Perceptual, against raw persistence** | LPIPS(decoded prediction, true frame) − LPIPS(raw last frame, true frame) | the training maps' margin, −0.027, with zero drawn as "ties copy-last" | how many updates until the model looks better than copying, by the margin it has at home | Astra (headline), Opus (second panel) |
| **C. Gap to the decoder's best render** | PSNR(decoded true latent, true frame) − PSNR(decoded prediction, true frame) = `vae_psnr − psnr_raw` | the training maps' gap, 1.55 dB (band 1.36 to 1.84) | how close the model gets to the best frame its decoder can draw, compared with how close it gets at home | Opus (headline), Astra (diagnostic) |

Facts that decide between them (recomputed):
- A is the only quantity that does not track static footage across the 13 arenas (Spearman with persistence PSNR −0.05); C tracks it at −0.80 and the raw gain at −0.77. A cost built on C or on raw gain partly measures motion.
- A and C both have every arena outside the training band at step 0 (A: 0.72 to 2.68 dB against 3.60; C: 1.96 to 4.00 against 1.55). Half the raw gain (+0.46 dB) is already exceeded by four arenas, which kills the raw-gain target.
- B has room to win (the decoder's LPIPS floor 0.08 to 0.12 sits below persistence 0.15 to 0.26 on every map) but the line is far: arenas need to cut LPIPS by 0.09 to 0.19, so heavy censoring at 4,000 is likely; it is the honest panel for the blur story (4 of 4 training maps win, 13 of 13 arenas lose).
- The HUD cancels in C by construction and is removed from A and B by the scene crop.
- Main's option 2 (fraction of the gap to in-domain absolute PSNR closed) is dropped: absolute PSNR is dominated by footage motion and capped by the ceiling, so the "gap" is ill-defined on calm arenas.

## Synthesis (recommended to Rohan)

- **Headline curve and cost: A**, decoded advantage over copy-last, tuned decoder, scene-only. The line is the training maps' own advantage; the reported crossings are at 25, 50 and 100 percent of it, with 50 percent the headline cost (it is where the in-domain anchor and Astra's "a fixed cut relative to copy-last" meet: half the home advantage is a fixed cut, chosen from in-domain data). Immune to static footage, every arena starts outside, Genie-form, the decoder Rohan wants involved is on both sides.
- **Second panel: B**, the perceptual margin against raw persistence, same decoder, same crop, with the zero line. It is the half of the headline the reader will ask about and the check that the model is not merely blurring its way up A.
- **Table column: C**, the gap to the decoder's best render, because it is the plain "how far from as good as it gets" number and the rendering leg of the decomposition; not a cost axis, because it tracks motion.
- **Latent skill**: out of the figures; one appendix column for the LoRA-versus-full-fine-tune comparison only, because it is the one read the decoder cannot move, and never across backbones (all three agree).
- **Freezing**: the lines depend only on step-0 base-model predictions on the training maps through the tuned decoder, so they are frozen from the saved latents the moment the tuned SD 1 decoder is gated (about 02:30), before any adapted score exists (Opus). The stock-decoder lines are frozen tonight from the fresh-set rescore.

## Where the three still differ

Astra would headline B (perceptual quality reaching in-domain level plus margin); Opus would headline C; main A. The synthesis takes A for the cost axis on the motion-independence fact, and keeps B and C where each is strongest. Full versus half of the home advantage as "the" cost: report both; 50 percent is the headline crossing, 100 percent the aspirational line on the same plot.
