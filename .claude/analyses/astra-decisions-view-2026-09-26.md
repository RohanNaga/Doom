# Astra's final position on the five open adaptation decisions (Sep 26 2026)

Codex thread `01a0dfa4-ec22-7f91-9c5d-7298c494c132` (gpt-6-astra, reasoning high), round 3 of the adaptation-design thread, two turns, read-only. Inputs: `RESEARCH_CONTEXT.md` section 0 and the three dataset views (`main-`, `astra-`, `opus-dataset-view-2026-09-26.md`). Positions are Astra's words, condensed. The numbers were checked by the supervising session (see the last section).

## Positions

- **(A) Metric.** Show both: headline the difference from persistence, and put absolute model, persistence and reconstruction scores in the table. This is a baseline comparison, not a new metric. Precedent: PredNet (arXiv:1605.08104, Table 2, "Copy Last Frame") and OccWorld (arXiv:2311.16038, Table 1, Copy&Paste plus reconstruction references), both confirmed online. Astra found no established *named* "gain over persistence" metric, so label the plotted quantities plainly as "model minus persistence PSNR" and "persistence minus model LPIPS". Keep reconstruction, latent error and control response beside the figure so that a positive gain is not read as enough for simulator quality.
- **(B) Decision 7.** Keep R <= 0.90 as the primary cost target, with half the training-map gain as secondary and 25/75 percent as sensitivity checks. R is the model's summed latent MSE divided by persistence's summed latent MSE, so a 10 percent error cut means the same thing on every map, whereas a fraction of another map population's gain is not directly comparable. The 0.90 threshold is a proposed practical value, not a literature standard. Freeze the choice before adapting: Sunday's 30-map read gives starting points and the behaviour of the denominator. It must not be used to pick the endpoint that happens to produce crossings.
- **(C) Decision 5.** Run the full fine-tune on arena 7 in October, not Monday. A free card is not a reason to add a training recipe, a fit check and a retention evaluation while the final tables are being made. Call it a *comparator*, not an upper bound: in DiffFit (arXiv:2304.06648, Table 1) DiffFit reaches mean FID 15.39 against 16.59 for full tuning.
- **(D) Decision 6.** Commit to the four-map preliminary figure. Treat the 13-arena curve by Monday night as a conditional stretch goal. Arithmetic, assuming two seeds and 1.5 A6000-hours per map per seed: the pilot is 4 x 2 x 1.5 = 12 card-hours and the full study is 13 x 2 x 1.5 = 39. Encoding on GPU 1 from Sunday 14:00 takes about 4.5 h, so data is ready around 18:30. GPU 1 then finishes 4 runs by Monday 00:30. After that, two cards finish the pilot's remaining 4 runs by 03:30, or the full study's remaining 22 runs by about 17:00 Monday, before any overhead. On Superman, 26 runs on 6 cards take 5 waves, which is 7.5 r hours after the copy, where r is the A4000/A6000 runtime ratio. r, the copy time and the scoring cost are unverified.
- **(E) Campaign arm.** Keep campaign maps 20, 32, 30 and 23 as a separate October stress arm that is never pooled with the arenas. They are the two lowest-D and two highest-D campaign targets, chosen by frozen D rather than by adaptation outcomes. Report coverage explicitly and make no claim of whole-map competence.

## Split question (F)

Adapt/held-out is enough. A separate target validation split is not needed when nothing is selected on held-out outcomes, so the 12/8 split stands. If LR, rank, thresholds, checkpoint policy or reporting rules change after results are inspected, a development set or a new test set is needed.

## Nice numbers (G)

Assumptions: adaptation uses global batch 32 on the 0/250/500/1,000/2,000 step grid (`RESEARCH_CONTEXT.md` line 20). Pretraining is 200k updates of batch 32 (line 16). 2,000 training episodes over 4 maps gives 500 per map.

| Adaptation budget | Fraction |
|---|---|
| 1 / 2 / 4 / 6 / 12 episodes | 0.2 / 0.4 / 0.8 / 1.2 / 2.4 % of one training map's 500 episodes |
| all 20 recorded episodes (incl. held-out) | 4 % of one training map's episodes |
| 250 / 500 / 1,000 / 2,000 updates | 0.125 / 0.25 / 0.5 / 1 % of pretraining updates |
| 2,000 x 32 = 64,000 window presentations | 1 % of 6.4M total; 4 % of the ~1.6M one training map received under balanced sampling |

Headline phrasing: "six episodes (1.2 percent of a training map's data) and 2,000 updates (1 percent of pretraining)". The update fraction counts presentations, not FLOPs: LoRA with a frozen backbone costs less per update, so this is not a claim of 1 percent of compute.

## Numbers checked, and errors found

Reproduced locally by the supervisor:
- **Fractions in (G)**: recomputed and all match.
- **Schedule in (D)**: 18:30 plus 4 x 1.5 h is 00:30. The remaining 4 runs on 2 cards take 2 waves, ending 03:30. The remaining 22 runs on 2 cards take 11 waves (16.5 h), ending 17:00 Monday. Matches.
- **Section 0 says the decoded copy is "about 1.4 dB higher" than raw persistence. This is backwards.** Mean of `persist_psnr_raw - copy_psnr_raw` over the 30 primary maps is **+1.363 dB** (range -0.08 to +4.04). So raw persistence is the higher one. Source: `results/distance_study/scores/040-unet-nexttic_snap_0200000_ema_ddim10/*_h1/metrics.json` with `distances_sd1.json`.
- **Section 0 and the Opus view give the unseen-arena Spearman as +0.01, partial -0.09. The exact values are -0.0165 and partial -0.1188** (n = 13, controlling for `persist_psnr_raw`). The null reading is unchanged.
- **The Opus view's latent budget ("about 5 GB, 12 MB per episode") is wrong for the every-tic recipe.** Every-tic training uses 150 s x 35 Hz = 5,250 tics, at (4, 32, 40) float16 = 10,240 B per tic. That is 53.8 MB per episode and 19.4 GB for 360 episodes (`encode_parquet.py --every-tic`, `doom_data.py` `LATENT_SHAPE_V2`). The 12 MB figure matches a stride-4 corpus (1,312 frames, 13.4 MB).
- **DiffFit "LoRA FID 81 vs 16.6 full" in section 0 is correct.** According to Astra, Table 1 gives LoRA-R8 81.25, R16 81.31, full 16.59 and DiffFit 15.39. Not checked offline.

Other inconsistencies reported by Astra (not reproduced by the supervisor):
- In section 0, 66.2 percent is arena 7's share of the sum of squared distance deviations, not of the distance range.
- Cross-space agreement is 0.8625 for SD1 vs pixels and 0.9026 for SD1 vs SD3.5, but only 0.7704 for pixels vs SD3.5.
- Main view lines 5 to 9: check (ii) tested five unseen arenas, not thirteen. A six-episode pool cannot supply twelve adapt episodes, and the fresh cohort makes that point moot.
- Section 0's "pilot on existing episodes" is a different protocol from the fresh-cohort study and must be labelled exploratory.
- Section 0's "accumulation" option for the fit check conflicts with the no-accumulation rule.
- A flat true-frame reconstruction alone does not show that "the loss is in predicted latents"; that attribution still needs the planned paired stock-versus-tuned decode.
