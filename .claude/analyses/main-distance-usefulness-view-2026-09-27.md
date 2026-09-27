# Is the distance useful? The main session's view (Fable 5.1, 2026-09-27 05:50 EDT)

Written before reading the Opus and Astra memos of today. Numbers from `tools/distance_outcome_corr.py` and its partial-correlation extension (scratchpad `dist_partial.py`, to be folded into the tool), inputs in `.claude/analyses/distance-fresh-numbers-2026-09-27.md`.

## One-line answers

1. No pre-computed distance orders the 13 arenas on any outcome (|rho| <= 0.42, p >= 0.15, leave-one-out worse than predicting the mean); the model's own zero-shot latent skill does (rho +0.89 with A at 4,000, -0.65 with the crossing budget, partial on persistence and motion unchanged).
2. The x axis of the adaptation figure is the zero-shot latent skill on the new map, which a reader computes with one evaluation pass and no training; the frame distance D appears only as the family separator in the distance figure.
3. Adapt the metric by changing what it measures, not its features: score how well the *model* already predicts the target's transitions (a model-in-the-loop distance, which is what the zero-shot skill is), or, if it must be model-free, a coverage of transitions by the training set's *prediction residuals*; concrete next version below, about 3 A4000-hours.
4. Sep 30 text: D separates the training family from every unseen arena and does not order the arenas; coverage and G, tried as pre-registered candidates, do not even separate the families; the zero-shot skill predicts the adaptation endpoint and budget.
5. October: distance-stratified generated arenas (retexture and re-lay-out the training maps at controlled steps) to test whether any pre-computed distance predicts cost when the shift is constructed rather than sampled; about 40 A4000-hours.

## 1. What orders the arenas

Spearman against the outcomes, n = 13, censored budgets set to 8,000:

| predictor | A0 | A4000 | gain | cost |
|---|---|---|---|---|
| D | -0.24 | +0.06 | +0.42 | -0.27 |
| coverage | -0.17 | +0.12 | +0.26 | -0.22 |
| G | +0.02 | +0.02 | +0.12 | -0.01 |
| zero-shot skill | +0.74 | +0.89 | +0.38 | -0.65 |

Partialling out the scene persistence PSNR or the context motion moves none of the distance rows by more than 0.15 and leaves the skill row at +0.75 / +0.91 / -0.74 (motion) and +0.76 / +0.89 / -0.64 (persistence). Leave-one-arena-out RMSE of a linear fit against predicting the mean: D 0.57 vs 0.55 on A4000 (worse), coverage 0.61 (worse), G 0.68 (worse); skill 0.27 vs 0.55 (halves the error). The incremental test fails for all three distances: partial on the skill, D adds +0.23 on A4000 and -0.40 on cost, coverage +0.27 / -0.30, G -0.23 / +0.11, none with p below 0.15 at n = 13. The one weak signal is D against the gain (+0.42, +0.48 partial on skill): farther arenas gain a little more because they start lower, which is the same information as the zero-shot deficit.

The family floor: D passes (training maps' frozen D at most 0.093, fresh unseen minimum 0.118); coverage (training 0.52 to 0.61, unseen 0.53 to 0.61) and G (training 0.005 to 0.012, unseen 0.009 to 0.038) fail, with four unseen arenas inside the training range on G. Coverage's own ablations explain why: the state block carries only 28 to 42 percent of the score and shuffling the transitions or the controls changes it by 0.01, so the metric reads episode idiosyncrasy, not the map. A held-out episode of a training map is as far from the training transitions as an unseen arena.

## 2. The x axis

The adaptation figure wants an axis a reader can compute before deciding whether to adapt. The zero-shot skill needs one evaluation of the frozen model on the new map's held-out windows (about eight minutes on an A4000) and it predicts where the curve ends and how fast it gets there; D needs the same latents and predicts neither. The figure therefore puts the zero-shot skill on the x axis (arenas as points, ranked), with A0, A4000 and the crossing budget as the y quantities, and keeps D for the two-panel distance figure that shows the family gap and the flat ordering. Not the zero-shot deficit in decoded dB, which mixes in the decoder floor; the latent skill is decoder-free and gives the stronger correlation.

## 3. How to adapt the metric

The internship metric asked "how much of the test set's appearance does the training set cover". The transition version asked the same of transitions and failed the family floor, because in a fixed-engine game the transitions of every map live on the same manifold and the nearest neighbour of a window is always in its own episode. What differs between maps is not whether a transition has a neighbour but whether the model's prediction rule transfers, and that is a property of the model, not of the two datasets. So the honest adaptation is model-in-the-loop: the distance is the frozen model's per-window latent skill on the target, and its "coverage" form is the fraction of target windows whose skill exceeds the training maps' validation median. This is what the zero-shot skill already measures; the next version makes it a proper distance by (a) scoring on the target's adaptation episodes rather than the held-out ones, so it is available before any held-out data is used, (b) averaging over three sampler seeds, and (c) reporting it beside D with the same bootstrap. Cost: 13 arenas times about eight minutes, three seeds, about 5 A4000-hours, all offline. If Rohan wants a model-free metric kept for the appearance axis, keep D as it is; do not spend more on transition coverage.

## 4. What the paper says

"A frame-level appearance distance to the nearest training map separates the training maps from every unseen arena (Figure 2, left) but does not order the arenas by zero-shot advantage, endpoint, or adaptation budget (Spearman |rho| < 0.45, n = 13), and two transition-level alternatives we pre-registered, a directed coverage of transition windows and a nearest-neighbour transfer gap, fail to separate even the families. The quantity that predicts how far and how fast an adapter climbs is the frozen model's own zero-shot latent skill on the arena (rho = 0.89 with the advantage at 4,000 updates, -0.65 with the crossing budget), which needs one evaluation pass and no training."

## 5. October

Generated arenas at controlled distance: take map 2, retexture it in five steps (swap k of its texture classes for classes of arena 7) and re-lay it out in three steps (mirror, rotate, recombine rooms), record 24 episodes each, and run the same LoRA ladder. Fifteen arenas with a known construction axis let the paper ask whether D or the skill tracks the construction step, which is the only clean test of "pre-computed distance predicts cost". About 40 A4000-hours of adaptation plus a day of recording.
