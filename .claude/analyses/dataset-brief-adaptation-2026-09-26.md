# Brief: how should the adaptation dataset be built? (2026-09-26, 19:00 EDT)

You are one of three independent reviewers (the main session, Astra, and an Opus 5.5 reader) asked for a separate conclusion on one question. Do not look for the others' answers; reach your own from the files and state it with reasons. Rohan (the project lead) will read the three side by side.

## The question

The paper's next study adapts a world model pretrained on four Doom deathmatch arenas to each unseen map with a small adapter (LoRA on frozen backbone weights) trained on a few episodes of that map, and plots adaptation curves (gain over persistence against LoRA updates, and against episodes) with a frozen training-to-map distance D on the x axis. Zero-shot scores already exist on 30 maps.

How should the dataset for that study be constructed? Specifically:

1. Which maps are targets: the 13 unseen deathmatch arenas only, the 13 campaign maps too, or some other set? Rohan's current view: arenas only, because the campaign maps are large single-player levels that a few episodes cannot cover and "will be really hard to learn out of distribution".
2. How many episodes per target map, and should every target map be brought to the same count and the same recording conditions? What does "the same data quality" mean operationally here (agent, seed policy, episode length, bots, difficulty)?
3. The per-map split: how many episodes adapt, how many are held out, and should the held-out count be uniform across maps? How should the held-out evaluation windows be drawn?
4. Whether the distance range of the chosen set is wide enough to test "D predicts cost", and what to do if it is not (for example generating more arenas with the Obsidian map generator, as MultiGen did, or keeping a few campaign maps as a far arm).
5. Anything about the dataset that would let a reviewer dismiss the curve (coverage confounds, selection on outcome, leakage between the distance computation and the adaptation split, the agent's behaviour).

Give a concrete recommendation for each, the reasoning, the recording and encoding budget it implies, and one measurement that would show your recommendation is wrong.

## Facts (all verifiable in the repo)

- Maps. Arnold's `full_deathmatch.wad` holds MAP01 to MAP17 only. Training maps are arenas 2, 3, 4, 5 (Arnold's own training maps); Arnold's test maps are 6, 7, 8. The 13 campaign maps (18 to 20, 22 to 26, 28 to 32) are Freedoom Phase 2 single-player levels loaded from the campaign IWAD with monsters and some pickups removed (2,430 things removed, geometry byte-identical); maps 21 and 27 were dropped because the agent got pinned. All maps use Freedoom assets, the same engine, the same agent (Arnold, a public deathmatch bot, with 8 bots), the same 320×240 frame with HUD, every tic recorded. `RESEARCH_CONTEXT.md` entries 2026-09-16 15:40 and 2026-09-19 17:00.
- Episodes per map today (one episode is 150 game-seconds, about 5,000 one-tic windows): arenas 6, 7, 8: 20 each; arenas 16, 17: 10 each; arenas 1, 9 to 15: 4 each; the 13 campaign maps: 10 each; training maps: 25 held-out validation episodes each (and 500 trained on). Per-map windows, deaths per episode, validity fraction and distance are in `results/distance_study/per_episode_sd1.csv` and `results/distance_study/splits/*.json`.
- Footage character. Arena episodes: 3 to 12 deaths per episode (map 1: 22), one-tic persistence PSNR 19 to 23 dB (violent motion), explored area 89 to 321 cells of 128 units per episode. Campaign episodes: about 1 death, persistence 20 to 27 dB (near-static wandering), three maps (19, 25, 26) where the decoder's reconstruction of the true frame scores below persistence.
- Distance. D is the motion-weighted sliced Wasserstein distance between a map's per-frame latent cloud (250 frames per episode, stratified by motion decile, all episodes of the map) and the nearest training map's cloud, frozen on Sep 25 (`results/distance_study/distances_sd1.json`). Arenas span 0.12 to 0.29 (only arena 7 above 0.19); campaign maps span 0.13 to 0.38. Check (ii), agreement of disjoint 4-episode subsets of a map within 10 percent, passes on every arena and fails on 8 of 13 campaign maps (subset medians 0.11 to 0.29). Check (iv), "same WAD arenas nearer than every campaign map", fails (arena 7 at 0.28). `docs/RESEARCHER_DOSSIER.md` section on the distance study; `results/distance_study/figure_unet_h1/distance_table.md`.
- Zero-shot result (U-Net 200k EMA, one tic): the model beats persistence on all four training maps and loses on 17 of 26 unseen maps in PSNR and 26 of 26 in LPIPS; partial Spearman of gain on D controlling for persistence PSNR is -0.73 [-0.84, -0.32] on 30 maps; on the 18 maps with 10 or more episodes, controlling for persistence and family, about -0.45 with a bootstrap interval spanning zero. Within the 13 campaign maps alone, -0.38 [-0.79, +0.35].
- Recording throughput (measured Sep 19): about 670 episodes per hour on Spiderman with 32 CPU workers, 194 per hour on Superman with 16; raw frames about 250 MB per episode; SD 1 latent encoding about a minute per episode on an A6000. Spiderman's data volume has about 225 GB free. New episodes would use the same seeded recorder and corpus layout as the existing evaluation corpus.
- Compute and time: the four-map U-Net pilot runs Sunday Sep 27; the paper draft goes to the advisor Sunday evening with the pilot as a preliminary figure; the deadline is Thursday Oct 1 07:59 EDT; the full curve is an October claim. No backbone retraining.
- Design page with the current proposal: `docs/lora_adaptation_design_2026-09-26.html`. Literature: `.claude/analyses/related-work-map-2026-09-26.md`, `lit-scene-generalization-2026-09-26.md`, `lit-transferability-2026-09-26.md`, `lit-adapters-2026-09-26.md`. Astra's earlier design review: `.claude/analyses/astra-adaptation-review-2026-09-26.md` (read it only after forming your own view on the dataset question; it did not address the dataset composition).

## Claims that could be false

- That six episodes cover an arena well enough that adapt and held-out episodes are exchangeable (check ii passing is evidence, not proof).
- That the campaign maps' difficulty is coverage rather than a real, harder shift worth measuring.
- That 13 arenas over a distance range of 0.12 to 0.29 give enough power to test the cost relation.
- That generating new arenas with Obsidian would produce maps whose footage is comparable to Arnold's arenas.

Answer as a memo with a recommendation per numbered question, the budget, and the disconfirming measurement.
