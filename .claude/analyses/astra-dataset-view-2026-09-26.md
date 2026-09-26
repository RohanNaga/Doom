# Astra's view: how to build the adaptation dataset (2026-09-26)

Codex thread: `01a0dfc8-2870-7b23-a706-2d8fa2e4ffcb` (gpt-6-astra, reasoning effort high, workspace-write sandbox, one turn, no files changed by Astra).
Prompt: `.claude/analyses/dataset-brief-adaptation-2026-09-26.md`, verbatim, plus a note to reply as a memo and not write files. Astra gave one recommendation per question, so no follow-up turn was needed.

## Headline

All 13 unseen arenas are the primary study. Each gets 20 fresh episodes under one pinned protocol: 12 available for adaptation and 8 held out. The existing footage stays reserved for the frozen D. Four campaign maps are recorded as a separately reported stress test and are never pooled with the arenas to widen the distance range.

Astra found two things in the files that shape this:
- Check (ii) never tested 8 of the 13 arenas. Maps 1 and 9 to 15 have 4 episodes, so they get one cloud and no disjoint pairs. Only arenas 6, 7, 8, 16 and 17 were tested. The brief's statement that check (ii) "passes on every arena" is too strong.
- Recording conditions depend on worker history. Arnold's weapon-select requests execute only in a recorder worker's first episode (k = 0). Matching "Arnold, 8 bots, seeded recorder" alone does not match behaviour.

## Per question (Astra's words, condensed)

**1. Targets.** Arenas 1 and 6 to 17 form the primary set. It is a complete, externally defined set, so inclusion cannot depend on model performance. Astra agrees with arenas first but rejects "campaign maps will be hard to learn" as the reason, because difficulty is part of the question. The defensible reason is that campaign geometry changes exploration, encounters and revisitation under a deathmatch agent. For October, add campaign maps 20, 32, 30 and 23, the two lowest and two highest frozen D, as a separate diagnostic. That choice uses D, not outcomes, and the near campaign maps separate a family effect from distance. For Sunday, use arenas 8, 16, 6 and 7 (D about 0.115, 0.164, 0.188, 0.282) as preliminary curves with no predictive-distance claim. The estimand is "adaptation to the state distribution Arnold visits on an unseen map". It is not whole-map competence, and exchangeability needs a stable sampling process, not full geometric coverage.

**2. Counts and quality.** Record 20 new usable episodes per target under one pinned protocol, and leave the historical archive as it is. Fresh data keeps adaptation and evaluation independent of the D footage and of earlier model choices, and it makes conditions uniform. "Same quality" means identical Arnold checkpoint, code and policy settings; engine, Freedoom and WAD hashes; opponent implementation, 8 bots, difficulty, HUD and rendering; 150 game-seconds, respawn handling, 4-tic decisions, every-tic recording; distinct predetermined seeds for Python, NumPy, Torch and ViZDoom; the same encoder, scaling, executed-control representation and valid-window rules. Exclude episodes only for technical faults, never for low motion, many deaths or poor exploration. Use the post-first-episode regime for every counted episode, with a documented warm-up start per worker and an audit of executed controls. A silent weapon-switch fix would be a new policy shift. Match the generating process, not death counts or motion histograms, and report deaths, valid windows, motion, stationary time, encounters and coverage per map.

**3. Split.** A seeded whole-episode permutation fixed before any scoring gives 12 adaptation-pool episodes and 8 held-out episodes on every map. Six episodes drive the updates curve. Nested subsets of 1, 2, 4, 6 and 12 episodes at a fixed update budget drive the data curve, each run starting from the same pretrained checkpoint. The other disjoint six-episode subset is an October sensitivity check. The same 8 held-out episodes serve every budget, seed and backbone. Draw 32 windows per held-out episode (256 per map), sampled at random within temporal strata from valid windows, never crossing episode or respawn boundaries, with a fixed window manifest and fixed inference seeds. Rescore step zero on these windows instead of reusing historical zero-shot scores. The natural window distribution is primary, and motion-stratified results are a diagnostic. Average paired model-minus-persistence gains within episodes, then across episodes, and bootstrap over episodes. Target-map held-out episodes never touch hyperparameter selection.

**4. Distance range.** The range is enough for an exploratory arena result but not enough to assume "D predicts cost". The 13 arenas span 0.1149 to 0.2819, but only 0.1149 to 0.1875 without arena 7, which carries 66% of the total squared deviation of D. More episodes improve each map's point but add no distant maps. A Fisher-z approximation gives about 41% power for rho = 0.5 at n = 13, which is a scale check, not a power analysis; the real analysis needs simulation from pilot variance. Report the arena result with map-level uncertainty and leave-one-map-out, including dropping arena 7. Do not add training maps as near targets. For October, prefer more comparable arenas over pooling campaign maps. A bounded Obsidian run would screen 24 fixed generator seeds with 4 distance-only episodes each, pick up to 12 by preregistered near, middle and far strata (random within strata), add 4 more distance episodes before freezing D, then record the 20-episode study cohort. This is conditional on comparable behaviour: Arnold's scripted opponents depend on WAD-specific scripts, and switching to ZDoom bots would be a policy change. If Obsidian cannot give a populated high-D arm, keep the claim exploratory. Campaign points cannot fill a gap in arena support.

**5. Reviewer objections.** *Coverage:* for each budget, measure the held-out visitation mass that falls outside the adaptation set's position-and-heading bins, plus a feature-space version, and plot coverage against episodes. *Distance reuse:* old footage for D, fresh footage for adaptation and evaluation. The frozen D is an average over 10 fixed-size 4-episode clouds, not one cloud of all episodes; keep that estimator. A fresh-cohort D is only a declared robustness check. *Selection:* keep every primary arena, flat or degrading curves included, and publish exclusions and Obsidian screening. The dropped campaign maps 21 and 27 restrict the population to maps the agent can navigate. *Behaviour:* audit executed controls and actual opponent presence, and report valid and unique windows next to episodes; map 1 gives fewer usable windows per episode. *Thresholds:* freeze success thresholds and checkpoint grids before adapting; report full curves and fixed-budget gains, and treat non-crossing maps as censored. On campaign maps 19, 25 and 26 the decoder scores below persistence, so a raw-PSNR crossing there is hard to read; keep latent-space evaluation.

## Budget (Astra's table, from 670 episodes/h, 250 MB raw/episode, 1 A6000-minute per SD1 episode, 51 MB SD1 latents/episode)

| Collection | New episodes | Recording | Raw + SD1 | SD1 encoding |
|---|---:|---:|---:|---:|
| Sunday's four arenas | 80 | 7.2 min | 24.1 GB | 1.3 GPU-h |
| All 13 arenas (includes pilot) | 260 | 23.3 min | 78.3 GB | 4.3 GPU-h |
| Four campaign diagnostics | 80 | 7.2 min | 24.1 GB | 1.3 GPU-h |
| Recommended total | 340 | 30.4 min | 102.4 GB | 5.7 GPU-h |

Reserve about 125 GB for everything, including metadata and warm-up recordings, which leaves about 100 GB of Spiderman's 225 GB. The conditional Obsidian extension (96 screening, 48 extra distance and 240 study episodes) adds about 116 GB and 6.4 GPU-h and needs its own storage or an archive step. SD 3.5 latents need separate encoding and about 4 times the storage.

## Disconfirming measurement

Preregister a coverage test. Across random six-episode adaptation subsets, measure the held-out visitation mass that lands in 128-unit position-and-heading bins the subset never visited. If the median unsupported mass exceeds 20% on at least 4 of the 13 arenas, reject the assumption that six episodes represent an arena. Astra calls the 20% a proposed practical threshold, not an established constant. The 12-episode arm then tests whether more footage closes the gap.

## Supervisor checks against the files

Recomputed from `results/distance_study/distances_sd1.json` (primary-role entries) and `results/distance_study/figure_unet_h1/distance_table.md`:
- Arena D range 0.1149 to 0.2819, and 0.1149 to 0.1875 without arena 7: **confirmed**.
- Arena 7's share of the squared deviation of D over the 13 arenas: **0.662, confirmed**.
- Check (ii) never tested arenas 1 and 9 to 15: **confirmed**. They have `subsets: 1`, no disjoint pairs, and do not appear under `checks.ii.subsets`.
- Sunday arena distances (8: 0.115, 16: 0.164, 6: 0.188, 7: 0.282) and the campaign picks (20: 0.140, 32: 0.143, 30: 0.338, 23: 0.371): **confirmed**.
- D uses 4-episode clouds averaged over 10 subsets, not one pooled cloud: **confirmed** (`config.episodes_per_cloud = 4`, `subsets = 10`; `distance_study.py:1183`). The brief's phrase "all episodes of the map" describes `D_pooled`, not the frozen D.
- Worker-first weapon regime: **confirmed** (`buttons_report.py:1-16`, `docs/RESEARCHER_DOSSIER.md:273,285-287`, `docs/REVIEW_2026-09-22.md` H1).
- Map 1 gives fewer usable windows: **confirmed** (valid fraction 0.85 against 0.93 to 0.99 for the other arenas).
- Arithmetic: 5,000 x 4 x 32 x 40 x 2 bytes = 51.2 MB; 340 x 301.2 MB = 102.4 GB; 340 min = 5.7 h; Fisher-z power for rho 0.5 at n 13 is about 0.41. **All confirmed.**

I found no wrong claim by Astra. Two statements in the brief are slightly off:
- The brief says campaign D spans 0.13 to 0.38. The frozen primary values span 0.140 to 0.371.
- The brief says check (ii) passes on every arena. It passes on the five arenas that were tested, and eight were not tested.

Not verified by Astra or by me: raw trajectories, the actual difficulty and opponent settings, spatial coverage, current free disk, and live recording throughput.
