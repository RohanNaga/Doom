# Survey, Sep 8, 2026: data-collection agents, Doom datasets, related work

Four Opus-agent web surveys, condensed. Verified facts only; each item has a URL in the source reports (this chat, Sep 8). Purpose: choose the data-collection agent and position the paper.

## 1. Data-collection agents with paper-backed checkpoints

| agent | paper | checkpoint | maps | notes |
|---|---|---|---|---|
| Arnold (Lample and Chaplot, AAAI 2017, arXiv 1609.05521) | yes | 5 `.pth` in github.com/glample/Arnold (2017-era torch pickle, no LICENSE file) | multi-map deathmatch WAD (train maps 2 to 5, test 6 to 8), ships own freedoom2.wad, texture randomization, full action set incl. doors | used by MultiGen (Mar 2026). Native RES_400X225, net input 108x60, LSTM DRQN; eval loop iterates `set_doom_map`. |
| Sample Factory 2 (Petrenko et al., ICML 2020) | yes | HF hub, edbeeching/*, 12 scenarios x 3 seeds (2022) | one fixed map per agent | trained at 128x72; 320x240 needs a code patch (`custom_resolution` kwarg); pins gymnasium<1.0, numpy<2. Used by Yunncheng/gamewam-vizdoom (79.7M frames, 640x480 H.264, Sep 2026). |
| GameNGen PPO (Valevski et al. 2024) | yes | not released | real Doom levels (count not stated; eval on 5) | reward table in appendix A.5 (see section 3). |
| Stiegler gameNgen-repro PPO | no paper | `best_model.zip` in repo, Apache-2.0 | one arena `deathmatch_simple.wad` | 6 buttons, 18 combos, no USE button (cannot open doors). Used by p(doom), Keerthana April data, six other unrefereed HF datasets. |
| LevDoom, COOM, vizdoom_ppo_rnd | papers / repos | no weights | scenario or difficulty variants | would need training. |

No public agent plays original Doom E1M1 to E4M9 or Freedoom2 MAP01 to MAP32. ViZDoom 1.3.0 (Feb 2026) ships `doom2.cfg` / `freedoom2.cfg` level configs with map-exit reward; 1.2.4 does not (hand-written cfg with `set_doom_map` works).

## 2. gameNgen-repro ecosystem

- All 15 forks diffed: none extended to more maps. Only fidelity fork: Masao-Taketani/GameNGen (pad 320x256, 64-frame context, collect during PPO training, still one arena, JPEG). Multi-scenario attempt: ReverseZoom2151/gamengen-v2 rotates five stock scenarios, no releases.
- Derived HF datasets, all `deathmatch_simple` 320x240 JPEG q85 unless noted: arnaudstiegler (4 sets), johnrobinsn/ViZDoom-500 (regenerated), charrywhite/VizDoom_withpos (Mar 2026, adds pose and enemy state), dokster/vizdoom-5lvls (per-frame bot difficulty), P-H-B-D-a16z/* (incl. random-action and `basic` scenario), Masao-Taketani inference slices, p-doom/doom-dataset (lossless raw uint8 but 60x80, CC0; generation code on unmerged branch `vizdoom-dataset`, 100 parallel envs, Lanczos downscale, metadata says `env: coinrun`).
- Outside the lineage: Yunncheng/gamewam-vizdoom (Sep 8, 2026: 79.7M frames, 640x480 H.264, four combat scenarios, Sample Factory APPO, Apache-2.0); lucrbrtv/doom-e1-gameplay (42.8k frames of real Doom E1, human, 320x240, public domain); invocation02/RandomDoomSamples-110M (random policy, PNG).
- Gap confirmed: no public lossless, per-tic, pose-annotated Doom set at usable resolution; no public multi-map set at all (MultiGen released nothing).

## 3. Reward structures

GameNGen appendix A.5 (verbatim terms): player hit -100; death -5,000; enemy hit +300; kill +1,000; item/weapon pickup +100; secret +500; new area +20 x (1 + 0.5 x L1 distance); health delta x10; armor delta x10; ammo 10 x max(0, delta) + min(0, delta). Actions held 4 frames plus boosted repeat probability. Agent obs 160x120 frame plus in-game map, last 32 actions; 8 parallel games, 50M steps; ALL training trajectories recorded from the random init onward (the skill mixture is the diversity source); 70M examples used. Stated limitation: agent does not explore all locations; beats random policy clearly only in the mid-distance bucket.

Stiegler / Kieliger: frag +1; damage +0.01; movement +0.0005 per unit beyond 3, -0.0025 if still; ammo +0.02 / -0.01; health +0.02 / -0.01; armor +0.01; ACS bot-difficulty curriculum 0 to 5; no exploration term.

MultiGen: no agent trained, no reward; Arnold over 100 Obsidian maps. PlayGen (arXiv 2412.00887): no reward; random spawns, random-action probability mixed with any agent, action repeats, offline k-means rebalancing; explicitly criticizes GameNGen's coverage. Arnold: navigation net gets a distance-travelled bonus to stop circling. Sample Factory deathmatch shaping has no distance term. RND-based explorers exist (vizdoom_ppo_rnd) but report that extrinsic exploration rewards cause circling.

Cheap coverage levers without RL training: record at several difficulty levels; mix a random-action probability with action repeats; randomize spawns; vary the map set (MultiGen).

## 4. Wetzstein group and closest related work

Group arc: Long-Context State-Space Video World Models (arXiv 2505.20171, ICCV 2025; matched 200M-param SSM vs transformer comparison on Memory Maze and TECO, the template for a single-variable architecture study) -> Video World Models with Long-term Spatial Memory (2506.05284, NeurIPS 2025; DiT + 3D point-cloud memory, view-recall metric) -> BAgger (2512.12080, CVPR 2026; rollout drift fix without a teacher) -> MultiGen (2603.06679, 2026; Doom, U-Net observation model, editable map memory, multiplayer, no backbone comparison, nothing released).

Must-read before the method section: Echo-Memory (2606.09803, Jun 2026): controlled study varying only memory with backbone fixed, code released; proves the single-variable paper is publishable, leaves the backbone axis open. Nano World Models (2605.23993, May 2026): minimalist open codebase with checkpoints ablating parameterization, scale, action injection; candidate harness. EDELINE (2502.00466): diffusion world model with SSM memory evaluated on ViZDoom. WorldPack (2512.02473): context packing, LoopNav revisit metric. DreamForge-World 0.1 (2606.30292): 480p at 14 FPS on one RTX 4090, the small-compute reference point. Benchmarks: iWorld-Bench (2605.03941, action following and memory metrics), WorldRoamBench (2606.31672, per-frame action metric), STEVO-Bench (2603.13215, state evolution while unobserved). Model as a Game (2503.21172): numerical consistency of HUD counters as a metric. Multiplayer world models with representation autoencoders (2607.05352): 5B Rocket League, the industrial counterpart to MultiGen.

Negative result across four searches: no paper runs a matched-compute, matched-data DiT vs U-Net comparison in a video or world model. The gap is real and survives only if the comparison is genuinely matched and honestly small-scale.
