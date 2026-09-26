# Opus view: how to build the adaptation dataset (2026-09-26)

Independent memo for `dataset-brief-adaptation-2026-09-26.md`; Astra's dataset view unread. "(recomputed)" marks numbers I derived from `results/distance_study/per_episode_sd1.csv`, `splits/index.json` and `figure_unet_h1/distance_table.md`.

## The fact that drives everything

Within the 13 unseen arenas (1, 6 to 17), zero-shot gain over persistence has no relation to D: Spearman 0.01, partial on persistence PSNR -0.09 (recomputed). Dropping arena 7 leaves 12 arenas in D 0.115 to 0.188 (SD 0.023) with Spearman 0.00. The 30-map partial of -0.73 comes from the validation maps near D 0.03 to 0.07 and from the campaign family, not from variation among arenas. Over the 26 unseen maps the partial is -0.56; within the campaign maps it is -0.39 (recomputed). An arenas-only study would most likely return a flat cost line that cannot separate "D does not predict cost" from "the arenas barely differ in D".

The second fact is the coverage asymmetry Rohan points at. Per-episode D within a map spans at most 0.062 on any arena (arena 6), and 0.037 or less on the other 12. On campaign maps it spans 0.10 to 0.32 on 9 of 13 (map 26: 0.138 to 0.462; map 19: 0.128 to 0.436) (recomputed). The exceptions are 23 (span 0.029), 20 (0.067), 30 (0.068) and 32 (0.073).

## 1. Target maps

**Recommendation.** Primary set: the 13 unseen arenas plus the campaign maps that passed check (ii) before any score existed: 19, 23, 25, 30, 32 (D 0.182, 0.371, 0.187, 0.338, 0.143). That is 18 maps. Treat family as a covariate. The other eight campaign maps are an appendix sensitivity only. For October, add a controlled far arm in the arena family (question 4).

**Why.** Arenas only removes Rohan's confound and also the range the claim needs. Check (ii) was frozen before any score and asks whether a map's D depends on which episodes were drawn, so it is the honest filter. Map 19 passes it despite a 0.31 per-episode span, so it is the weakest member; report the set with and without it. The filter admits the only two maps beyond D 0.29 other than arena 7 (23 and 30), both with bootstrap SD of 0.005 to 0.006. Maps 19 and 25 are among the three whose decoder reconstruction scores below persistence, so their PSNR target may be unreachable for decoder reasons. Their cost must come from the decoder-free latent ratio R (design decision 7), PSNR crossings right-censored, not dropped.

## 2. Episodes per map and recording conditions

**Recommendation.** Record a fresh corpus B of 20 episodes on every target map, in one recorder invocation, and use it for both adaptation and evaluation. Keep the existing corpus A for the frozen D and the published zero-shot points. Every map gets the same count.

**"Same data quality" operationally.** One recorder commit and WAD hash (the curated campaign PWAD with its manifest), one Arnold checkpoint, 8 bots at the same skill, 150 s episodes recorded every tic, seeds from a pre-declared range disjoint from every earlier corpus, and the same encoder commit with `--align-decisions` and canonical controls. Discard each worker's first episode by design. Arnold's weapon-select requests execute only there (dossier section 2.3), which is why `arenas_678` ids 0:60 were replaced. Whether `seen`, `unseen` and `unseen2` contain worker-first episodes is unverified (VERIFY), one more reason to record B fresh. The only exclusion is the agent-pinned rule that dropped maps 21 and 27, written before recording as a threshold on explored 128-unit cells. Deaths, motion and validity are covariates, never filters.

**Why 20.** Arenas hold 4, 10 or 20 episodes today, so an episode axis would mean different things per map. Precedent for the adapt budget is 10 to 13 trajectories (DreamGen), 20 episodes (TD-MPC2), 25 to 75 (XEWorld) (`lit-adapters`, `related-work-map`). Twenty covers a log-spaced data curve plus a larger held-out set.

## 3. Per-map split and held-out windows

**Recommendation.** 8 adapt and 12 held-out on every map, drawn once by a seeded permutation and written to split files before any run. Adaptation subsets are nested (1 in 2 in 4 in 8). The steps curve uses all 8. Held-out: 384 windows, 32 per held-out episode, uniform over target-eligible positions (context and target inside one life, at least 32 tics into it), one seed. The same windows serve every checkpoint, rank, seed and backbone, and the step-0 zero-shot point is rescored on them.

**Why uniform.** Held-out precision must not vary with family, or wider campaign CIs push crossings toward censoring and fake a D effect. Stratifying by episode stops one episode dominating. Nothing is chosen on target data here (fixed grid, fixed recipe); if anything is, carve 4 of the 12 into target validation, as Astra's review does with 6/2/2.

**The price.** Campaign intervals stay wide (today's 10-episode CIs reach 4.4 dB on maps 19 and 24). That noise is reported as a property of the map, not fixed with extra episodes.

## 4. Is the distance range wide enough?

**Recommendation.** No. Eighteen maps detect a Spearman of about -0.62 or stronger at 80 percent power and alpha 0.05. The approximate requirement is 14 maps for rho -0.7, 20 for -0.6 and 31 for -0.5 (Fisher z with the 1.06 Spearman factor; my arithmetic). The zero-shot partial on the 18 maps with 10 or more episodes was about -0.45. Among arenas, map 7 alone carries the range above 0.19.

**What to do (October).** Build 8 arena-family maps in D 0.20 to 0.35, selected on D measured from 4 screening episodes each. Selecting on the predictor, unlike the outcome, is legitimate. My first choice is re-texturing unseen arenas: the same geometry and deathmatch play, with wall and floor textures swapped at graded fractions for textures from the campaign IWAD. That moves D through appearance alone, holds coverage and behaviour fixed, and breaks the collinearity of "far" with "campaign". The VAE latent keeps textures (dossier 7.1), so D should move (VERIFY on screening episodes). Obsidian arenas are the second choice. They follow MultiGen (Arnold as agent, RC 09-19) but form a third family, and whether Obsidian emits deathmatch starts Arnold can use is unverified (VERIFY).

## 5. What a reviewer would use to dismiss the curve, and the fixes

- **Leakage between D and the split.** Frozen D used every episode of each map, including future held-out ones (Astra's transductive caveat). Corpus B removes it: D comes from A, adaptation and scoring from B, and no episode is in both. Recompute D on B's adapt episodes as a secondary, prospective x axis, never on held-out.
- **Coverage confound.** Cost on a large map may measure unvisited area, not distance. Recorded positions already give explored cells. For each map, compute the fraction of held-out target windows whose agent cell was visited in the adapt episodes. Report the partial of cost on D given coverage and persistence PSNR, beside the directed D variant `lit-transferability` recommends pre-registering.
- **Family collinearity.** Fit cost against D on the arenas and predict the campaign maps out of family. Report whether they fall on the line.
- **Selection on outcome.** Freeze the map set, split seed, window manifest, cost definition and D before any adapted score. The Sunday pilot runs on corpus A, so it cannot shape B; keep its numbers out of the headline.
- **Agent behaviour and leverage.** Arenas have 3 to 12 deaths per episode (map 1: 22, validity 0.85); campaign maps about 1. Keep persistence PSNR as the control, log deaths, and report leave-one-map-out naming arena 7, 23 and 30.

## Budget

- **Primary corpus B.** 18 maps × 20 = 360 episodes, plus 32 discarded worker-first episodes, is about 35 minutes on Spiderman at 670 per hour. Raw frames are about 98 GB against 225 GB free. Adapt raw frames (36 GB) can go after encoding is verified; held-out raw frames (54 GB) stay because PSNR scores raw targets. SD 1.x encoding takes about 6 A6000-hours (360 at about 1 minute each), or about 3 hours on two cards. Superman needs only latents (about 5 GB, my estimate at 12 MB per episode) and the 384 target frames per map. SD 3.5 encoding, needed only for its map subset, is unmeasured (VERIFY).
- **Appendix campaign maps (optional).** 160 episodes: 15 minutes, 40 GB raw, 2.7 encoding hours.
- **October far arm.** About 20 candidate maps × 4 screening episodes = 80, then 8 selected × 20 = 160. That is 240 episodes, about 60 GB raw and about 4 encoding hours.

## The measurement that would show this wrong

Rescore the U-Net zero-shot on corpus B's arena held-out windows (13 maps × 384 windows, about 2 A4000-hours, before any adaptation). My case beyond arenas rests on the within-arena null (partial -0.09 on today's 4- to 20-episode sets). If the within-arena partial of gain on D given persistence PSNR comes out at -0.5 or stronger with a bootstrap interval excluding zero, the arenas carry a usable D signal, Rohan's arena-only design suffices for the headline, and the campaign and far arms become optional.
