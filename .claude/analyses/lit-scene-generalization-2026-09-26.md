# Unseen-scene, level and map generalisation: benchmarks and studies (2026-09-26)

Scope: the benchmark and study slice for the final framing (30 Doom maps, zero-shot against copy-last-frame, then LoRA adaptation curves per map with a training-to-map distance). This file does not repeat `distance-study-literature-2026-09-25.md`. Where a paper is already there (GameNGen, DIAMOND, Genie, NWM, Vista, GAIA-2, MultiGen, SCOPE, PredNet, Procgen), only the new columns are added: what counted as unseen, how many, and the verdict against our design. Transferability estimation and adapter methods belong to other readers. AVID and AdaWorld appear here only because they are the closest unseen-environment evaluations of a world model.

Method: every number was read from the arXiv PDF text (`pdftotext`) on 2026-09-26. The one exception is Anand et al.'s venue, taken from the iclr.cc poster listing title. **VERIFY** marks anything not confirmed that way.

"Same as ours" means a learned simulator scored per held-out 3D map against ground truth. "Partly" means one of those three is missing: learned simulator, held-out map, or ground-truth scoring. "Different" means an agent return or policy score, or appearance-only shift.

## Bottom line

- **Agents.** Unseen-level evaluation is the default in RL generalisation benchmarks, and in Doom it goes back to 2016. The ViZDoom competition's Track 2 scored bots on 3 (2016) and 5 (2017) unseen maps. Lample and Chaplot used 10 train and 3 test maps. Neural Map, SPTM, EgoMap and the affordance-map paper all hold out mazes or maps. Every one of these scores an agent, not a simulator.
- **Learned simulators.** Held-out-scene evaluation of a world model against ground truth exists in three places:
  - 2D procedural levels: AVID, 1,024 trajectories from CoinRun levels 10,000 to 11,000.
  - Indoor navigation: Pathdreamer, 783 trajectories in unseen Matterport3D buildings.
  - Driving: dataset-level holdouts (Vista on Waymo, GAIA-2 geofences, NWM with one unknown environment).
  
  None of them reports a per-scene score or a scene-level spread.
- **Game world models.** Recent game world models treat "unseen" as held-out trajectories on training maps (GameNGen), as image-prompted new scenes with no ground truth (GameFactory, SCOPE, WorldMark, Vid2World, Genie), or as training biomes (Matrix-Game). No Doom world model found evaluates on held-out maps. MultiGen trains on 100 generated maps but never says whether its Table 1 maps are held out.
- **Adaptation curves.** These exist for world models on unseen environments at environment granularity: AdaWorld (4 environments, PSNR against fine-tune steps and samples) and AVID (1 held-out game, performance against dataset size). Nobody runs them per map across many maps or sets them against a distance.
- **Kirk et al.'s survey** (JAIR 76, 2023, arXiv 2111.09794) states that very little work applies model-based RL to zero-shot generalisation benchmarks. It names Anand et al. (2021) as the first to test how standard model-based methods generalise.

## A. RL level-generalisation benchmarks (Procgen, Crafter, NetHack, Minecraft)

- **Procgen** (Cobbe et al., ICML 2020, 1912.01588). The default protocol trains on 500 levels (hard) or 200 (easy) and tests on the full level distribution (§2, Fig. 4 and Fig. 13). Fig. 2 plots test return against training-set size per game. The metric is episodic return, normalised per game. Verdict: different (agent return).
- **Anand et al., MuZero on Procgen** (ICLR 2022, 2111.01587). This is the one study where a learned model sits inside the agent on held-out levels. It trains on 10, 100, 500 or unlimited levels and always evaluates on "the infinite test split" (§3; Fig. 5 and Fig. A.6). The metric is mean-normalised test return, 0.50 for MuZero (§4.1). The model is judged only through return, apart from qualitative reconstructions (Fig. 4). Verdict: different. It is still the best citation for "more training levels close the gap".
- **Imagined Autocurricula** (preprint, 2509.13341). It trains a world model on 200 Procgen levels per game and runs agents in imagination. Agents are then evaluated on "unseen test levels (201-∞)" in 7 games (§4.1). The metric is return (Table 1). Verdict: different.
- **Crafter** (Hafner, ICLR 2022, 2109.06780). It generates "a unique world for every episode" (Fig. 3), so every evaluation episode is an unseen world. There is no train and test split. The metric is the score, a geometric mean of achievement success rates. Verdict: different (agent; every world is new but nothing is held out).
- **DreamerV3** (2301.04104; Nature 2025, **VERIFY** from the journal page). It runs ProcGen at hard difficulty on "the unlimited level set" (Methods), so training and test levels come from the same distribution. It reports return only. Verdict: different.
- **NetHack Learning Environment** (NeurIPS 2020, 2006.13760). It trains on restricted seed sets and tests on 100 held-out seeds. The gap narrows at 1,000 or more training seeds (§3.1, Fig. 4). The metric is in-game score. Verdict: different. Its "gap against number of training seeds" figure matches our planned four-map training limitation.
- **MineDojo** (NeurIPS 2022, 2206.08853). It tests zero-shot visual generalisation to 27 scenarios of unseen weather, lighting and terrain. The metric is success rate over 3 seeds and 200 episodes, with relative degradation (Table 3). Verdict: different (policy; appearance shift).
- **MineRL** (IJCAI 2019, 1907.13440) is a demonstration dataset paper. It contains no level-generalisation study for learned dynamics (text searched).

## B. Visual and dynamics shift for model-based RL (DMC family)

These separate **appearance shift**, where pixels change and dynamics stay fixed, from **dynamics shift**, where physics parameters change. Our maps change both geometry (dynamics of what is visible) and texture (appearance), so neither family matches cleanly.

- **DMC-GB** (Hansen and Wang, SODA, 2011.13389; ICRA 2021 **VERIFY**). Its test distributions are random colours and natural-video backgrounds. It reports return for model-free agents. Appearance only.
- **Distracting Control Suite** (Stone et al., 2101.02722). The distractions are colour, camera pose and background. For backgrounds, training uses the first *b* ≤ 60 DAVIS-2017 training videos and generalisation is tested on the 30 DAVIS test-split videos. Performance on unseen videos "increases and then levels off" as *b* grows (Fig. 5d). Only SAC and QT-Opt are evaluated. Appearance only.
- **TIA** (ICML 2021, 2106.15612) and **DreamerPro** (2110.14565, venue **VERIFY**). Both are Dreamer variants on DMC with Kinetics natural-video backgrounds. They use separate train and test video sets, which tests "unseen distractions" (DreamerPro §5). The metric is return. Appearance only.
- **RePo** (NeurIPS 2023, 2309.00082). It trains on undistracted backgrounds and adapts the encoder at test time to distracted DMC, Maniskill with Matterport backgrounds, and a real TurtleBot with TVs (§4.1, Fig. 5). This is a precedent for adapting a world model to a shifted scene while freezing its dynamics. Only the visual encoder is adapted, and success is measured by return.
- **CaDM** (ICML 2020, 2005.06800) and **T-MCL** (NeurIPS 2020, 2010.13303). Both train dynamics models on a range of masses, lengths and similar parameters and test outside that range. They report return (CaDM Table 1) and state prediction error against mass, with the training range shaded (CaDM Fig. 6b). This is dynamics shift on a scalar parameter, where "distance" is literally the distance outside the training range. It is a conceptual precedent for a distance axis, but on low-dimensional states.
- **CARL** (NeurIPS 2021 workshop, 2110.02102) formalises the same thing as contextual MDPs with train and test context sets.

## C. Video prediction and world models on unseen scenes

- **PredNet** (ICLR 2017, 1605.08104) trains on KITTI and tests on CalTech Pedestrian against Copy Last Frame. This is one cross-dataset domain; details are in the 09-25 file.
- **Finn et al. 2016** (1605.07157) and **Ebert et al. 2017** (CoRL, 1710.05268) test on robot pushing with seen and novel objects. Ebert's Table 1 covers 20 object and goal configurations. The metric is final distance to goal (a planning score), not frame error. Verdict: different (objects, not scenes).
- **RoboNet** (CoRL 2019, 1910.11215) holds out viewpoints and backgrounds zero-shot (Table 2, Fig. 3). It also holds out whole robots and fine-tunes on them: Kuka and Franka with 400 trajectories each, Baxter with 300 (Tables 3 to 5). The metric is planning success rate. Verdict: partly. It pretrains, holds out a domain and fine-tunes on a small target set, but reports one data size per robot and no curve.
- **Pathdreamer** (ICCV 2021, 2105.08756) evaluates on R2R Val-Unseen, 783 trajectories in "Matterport3D environments not seen in training", against Val-Seen with 340 (§4). Horizons run 1 to 6 panoramas. Metrics are semantic mIOU (Fig. 5a) and FID (Fig. 5b, Table 2). It includes a non-learned **Nearest Neighbor** reprojection baseline, the closest analogue to a persistence reference in a 3D unseen-scene world-model evaluation. The number of Val-Unseen buildings is not stated in the PDF; the R2R standard is 11 (**VERIFY** against Anderson et al. 2018). Verdict: partly. It has held-out 3D scenes, ground truth and a copy-style baseline, but no per-building scores and no adaptation.
- **AVID** (2410.12822; listed at ICLR 2025, **VERIFY** main track or workshop) pretrains a pixel video-diffusion model on 15 of 16 Procgen games, excluding CoinRun (§4.1). It then adapts on 100, 500 or 2,500 CoinRun levels and tests on 1,024 ten-step trajectories from levels "sampled uniformly at random between level 10000 and 11000" (App. B). Metrics are action error ratio, FVD, PSNR and LPIPS (§4.2). Fig. 4c plots normalised performance against dataset size. Verdict: partly, and the closest on both halves. The held-out game has held-out levels, ground truth and a data-size adaptation curve. But it is one 2D game, levels are pooled into one score, and there is no persistence reference or distance.
  - By my arithmetic, 1,024 uniform draws from 1,000 levels touch about 640 distinct levels. The paper does not state the count.
- **AdaWorld** (ICML 2025, 2503.18938) pretrains on video plus 1,016 Retro and Procgen environments. It adapts to four environments "not included" in pretraining: Habitat, Minecraft, DMLab and nuScenes. Evaluation uses 100 samples per action and 800 fine-tune steps, scored by PSNR and LPIPS (Table 2). Fig. 6 gives PSNR curves against fine-tune steps and against sample count for Minecraft and nuScenes. Verdict: partly. Adaptation curves on unseen environments exist, but with 4 environments, no persistence reference and no distance.

## D. Game world models and "unseen" levels, maps or worlds

| Paper | Venue, year, arXiv | What counted as unseen | How many | Metric | Where | vs ours |
|---|---|---|---|---|---|---|
| GameNGen | ICLR 2025, 2408.14837 | held-out *trajectories*, not maps; OOD only via hand-edited frames | 2,048 trajectories, 5 levels | PSNR 29.43, LPIPS 0.249 teacher-forced | §5.1; App. A.4, Figs. 14–15 | different |
| DIAMOND CS:GO | NeurIPS 2024, 2405.12399 | none; single map (Dust II), qualitative drift in rarely visited areas | 1 map | FID, FVD, LPIPS | §5 | different |
| Genie | 2402.15391 (ICML 2024 **VERIFY**) | OOD image prompts; "unseen RL environment" images | qualitative; CoinRun policy score | FVD, ΔtPSNR; BC score | Fig. 14–15 | different |
| Vid2World | ICLR 2026, 2505.14357 | trained on CS:GO, zero-shot Valorant | qualitative | none | Fig. 18 | different |
| MineWorld | tech report, 2504.08388 | random clip split of VPT (not scene-disjoint) | 1k test clips | video metrics, human | §4 | different |
| Matrix-Game | tech report, 2506.18701 | none: its 8 evaluation biomes are the 8 biomes of its balanced training set | 8 biomes | GameWorld Score (8 dims) | §4.2 text; §6.3, Fig. 9 | different |
| GameFactory | ICCV 2025, 2501.08325 | open-domain scenes generated from text; Minecraft-trained actions | prompt count **VERIFY** | Cam, Flow, CLIP, FID, FVD, Dom | Table 3 | different (no ground truth) |
| SCOPE | 2605.23345 | 4 unseen scene styles, first frames from an image generator | 50 clips per style | JEPA sim., LPIPS, flow, smoothness vs in-distribution row | Table 3 | partly (no ground truth, n = 4) |
| WorldMark | benchmark, 2604.21686 | 50 scenes (25 photoreal, 25 stylised), image-prompted | 500 cases | benchmark suite | "Test Conditions" | different (no ground truth) |
| MultiGen (Doom) | 2603.06679 v2, venue **VERIFY** | not stated; trained on 100 Obsidian-generated maps | 100 train maps; test maps **VERIFY** | SSIM, PSNR, LPIPS by rollout segment | Table 1, §4.1–4.3 | partly at best |
| PlayGen (Doom, Mario) | 2412.00887, venue **VERIFY** | none; agent spawned at random positions on "the map" | 1 map | playability metrics, PSNR | §4, App. A.2 | different |
| GameGAN (VizDoom) | CVPR 2020, 2005.12126 | none; Take Cover scenario after Ha and Schmidhuber | 1 scenario | qualitative, human | §4 | different |
| Chiappa et al. (DMLab 3D mazes) | ICLR 2017, 1704.02254 | "randomly generated 3D mazes"; test set of 1,100 episodes; whether test mazes are disjoint is **VERIFY** | 1,100 test episodes | qualitative frames at 1–200 steps | §4, Fig. 9, App. B.3 | partly if disjoint |
| MIRA | 2607.05352 | none; "three fixed maps" (car-soccer game) | 3 maps | FD-type distances | Limitations | different |
| GameWAM (ViZDoom) | 2608.26200 | four-map ViZDoom suite scored by agent reward | 4 scenarios × 50 episodes | reward | Fig. 4 | different (agent) |

Oasis, Genie 2 and Genie 3 have no papers; this file makes no claims about them.

## E. Doom and ViZDoom generalisation studies (agents)

| Paper | Venue, arXiv | Unseen unit | Train / test | Metric | Where |
|---|---|---|---|---|---|
| ViZDoom Competitions (Wydmuch et al.) | 1809.03470 (IEEE ToG **VERIFY**) | organiser-made maps unseen by entrants | Track 2: 3 maps (2016), 5 maps (2017) | frags, deaths | Track 2 text; Fig. 5 (2017 maps); result tables |
| Lample and Chaplot, Arnold | AAAI 2017, 1609.05521 | deathmatch maps, textures randomised in training | 10 / 3 | kills, K/D | Table 2 |
| Dosovitskiy and Koltun, DFP | ICLR 2017, 1611.01779 | disjoint texture sets | train × test table | frags | Table 2 (09-25 file) |
| Pathak et al., ICM | ICML 2017, 1705.05363 | MyWayHome train map to a test map with different textures | 1 / 1 (textures also differ) | exploration coverage, success | §4.3, Fig. 4 |
| Neural Map | 1702.08360 (venue **VERIFY**) | Doom mazes | 1 train map / 6 unseen mazes; 1,000 held-out 2D mazes | % goals found in 1,000 episodes | Table 2 |
| SPTM | ICLR 2018, 1803.00653 | maze layouts | 1 layout × 400 variants / 3 val / 7 test | success rate | §4.1, Fig. 4 |
| Beeching et al., RL on a budget | 1904.01806 (venue **VERIFY**) | scenario configurations | 16–1,024 train / 64 held out | return vs training-set size | Fig. 2 |
| EgoMap | 2002.02286 (venue **VERIFY**) | scenario configurations | 256 / 64 per scenario | return | Table 2 |
| Qi et al., affordance maps | ICLR 2020, 2001.02364 | Oblige-generated maps | 60 / 15 (hazard-dense vs sparse) | navigation trials, damage | §4 |

The pattern is clear. The Doom community built unseen-map evaluation for policies and memories (up to 1,000 held-out mazes for Neural Map in 2D and 64 configurations for EgoMap). It uses the same generator family as MultiGen (Oblige; Obsidian is its successor, **VERIFY** lineage). It never used unseen maps to test a learned simulator.

## F. Driving and navigation world models

| Paper | Venue, arXiv | What counted as unseen | How many | Metric | Where | vs ours |
|---|---|---|---|---|---|---|
| Vista | NeurIPS 2024, 2405.17398 | whole dataset (Waymo) unseen in training; nuScenes val drives | 1,500 Waymo cases (reward); 60-scene human study over 4 datasets | FID, FVD, 2AFC, action-control reward | Table 2, Figs. 7–8, 10 | partly (dataset-level) |
| GAIA-2 | 2503.20523 | geofenced regions excluded from training | n = 1,024 samples | validation loss, FDD, FID, FVMD | §3, Fig. 13 | partly (one pooled holdout) |
| DriveDreamer | ECCV 2024 **VERIFY**, 2309.09777 | nuScenes validation drives (same cities) | 150 validation videos | FID, FVD | Table 2 | different (scenes, not geographies) |
| Cosmos | 2501.03575 | "50 unseen test videos" for tokenizer/decoder evaluation only | 50 | reconstruction metrics | Tab. 15 area | different |
| NWM | CVPR 2025, 2412.03572 | one unknown environment (Go Stanford) | 1 | LPIPS, DreamSim, PSNR @4 s | Table 4 | partly (n = 1) |

## Answers

**(1) The largest number of held-out scenes any world-model paper evaluated on.**
- **AVID** is the largest with ground-truth scoring of a learned simulator. It uses 1,024 trajectories drawn from CoinRun levels 10,000 to 11,000, all procedurally generated levels never used in adaptation, inside a game excluded from pretraining. That is up to 1,000 held-out levels, about 640 distinct in expectation (my arithmetic, not stated). The scores are pooled into one FVD, PSNR, LPIPS and action-error number (App. B, §4.2).
- **Pathdreamer** is the largest in 3D: 783 trajectories in held-out Matterport3D buildings (11 by the R2R standard, **VERIFY**). It is pooled and compared against a Nearest Neighbor reprojection baseline (Fig. 5, Table 2).
- **Driving** models evaluate on at most 150 held-out drives (DriveDreamer) or one held-out dataset or geography (Vista, GAIA-2).
- **Agent-return evaluations** go much larger (MuZero on Procgen's infinite split; NetHack with 100 seeds; Neural Map with 1,000 2D mazes). They never score the model's predictions per scene.
- **Nobody reports a per-scene score.** Every paper pools. Our 30 per-map scores with confidence intervals have no precedent in this slice.

**(2) Whether any Doom or ViZDoom paper evaluated a learned simulator on unseen maps.**
No verified instance.
- GameNGen holds out trajectories, not maps.
- PlayGen, GameGAN and World Models use one map or scenario.
- GameWAM scores an agent.
- MultiGen trains on 100 generated maps and scores SSIM, PSNR and LPIPS, but never states whether the Table 1 maps are held out. Its limitation paragraph says only that appearance "may not generalize to styles far outside the collected VizDoom trajectories" (Limitations paragraph). If an author or reviewer says MultiGen did this, the answer is that the paper does not say so.
- The Doom unseen-map literature (competition Track 2, Arnold, Neural Map, SPTM, EgoMap, affordance maps) is entirely about agents. It supplies the precedent for the split, not for the measurement.

**(3) Verdict: partly done, and new in the combination we claim.**
- **Done:**
  - unseen-level splits for agents, including in Doom;
  - pooled held-out-scene evaluation of world models (AVID, Pathdreamer, Vista, GAIA-2);
  - adaptation curves of world models on a handful of unseen environments (AdaWorld with 4 environments, AVID with 1 game, RoboNet with 3 robots at a fixed data size);
  - a copy-style baseline on unseen 3D scenes (Pathdreamer's Nearest Neighbor).
- **New:**
  - a learned simulator for a 3D game scored per held-out map on about 30 maps;
  - the same maps carrying a persistence reference, a distance to training, and per-map LoRA adaptation curves;
  - a Doom world model evaluated on maps it never trained on at all.
- **Framing consequence.** Claim the benchmark and the per-map adaptation study as the contribution. Cite AVID and AdaWorld as the nearest adaptation work, and Pathdreamer as the nearest persistence-referenced unseen-scene evaluation. Cite the ViZDoom agent literature for the practice of holding out maps. Do not claim to be first to test world models on unseen levels.

## Three facts most likely to change the decision

1. **AVID already combines a held-out game and held-out levels with a data-size adaptation curve** (Fig. 4c, App. B). It is 2D, uses one game and pools its levels. A reviewer who knows it will ask what per-map granularity and the distance add. The answer has to be in the introduction.
2. **AdaWorld (ICML 2025) plots PSNR against fine-tune steps and samples on unseen environments** (Fig. 6, Table 2 at 800 steps). The adaptation-curve shape is taken, so novelty rests on per-map resolution, the persistence-referenced outcome and the distance predictor.
3. **No Doom paper evaluates a learned simulator on held-out maps, but MultiGen is ambiguous.** It trains on 100 Obsidian maps with a GameNGen baseline. Its authors could plausibly claim held-out maps in a later version. Check v3 or the camera-ready before submission.

## Sources (PDF text read 2026-09-26)

arXiv: 1912.01588, 2111.01587, 2509.13341, 2109.06780, 2301.04104, 2006.13760, 2206.08853, 1907.13440, 2011.13389, 2101.02722, 2106.15612, 2110.14565, 2309.00082, 2005.06800, 2010.13303, 2110.02102, 1910.11215, 1710.05268, 2105.08756, 2410.12822, 2503.18938, 2408.14837, 2405.12399, 2402.15391, 2505.14357, 2504.08388, 2506.18701, 2501.08325, 2605.23345, 2604.21686, 2603.06679, 2412.00887, 2005.12126, 1704.02254, 2607.05352, 2608.26200, 1809.03470, 1609.05521, 1705.05363, 1702.08360, 1803.00653, 1904.01806, 2002.02286, 2001.02364, 2405.17398, 2503.20523, 2309.09777, 2501.03575, 2412.03572, 2111.09794 (Kirk et al., JAIR). Checked and found out of scope: 2609.09418 (Valerant), 2605.02528, 2606.07687, 2607.27599. Venue for 2111.01587 from the iclr.cc 2022 poster listing; AVID's ICLR 2025 listing seen only as a search-result title (**VERIFY**).
