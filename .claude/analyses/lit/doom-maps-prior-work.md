# Which Doom maps prior world models actually used (2026-09-19)

Evidence for choosing the maps of our dense corpus. Every fact carries a URL; anything not traceable to raw page or code text is marked **VERIFY**. Extends `doom-world-models.md` and `doom-baselines-2026-09-17.md`, and corrects one claim in the former (see gameNgen-repro).

## Summary table

| Work | Maps / scenario | Assets | Mode | Agent | Released? |
|---|---|---|---|---|---|
| GameNGen | "5 different levels", **never named**; no WAD named | not stated | not stated; reward has "Secret found", "New area" | PPO, 10M steps, action held 4 frames | nothing |
| gameNgen-repro | **one** map, `deathmatch_simple.wad`; "lvl5" = ACS *difficulty* | Freedoom (ViZDoom default) | `-deathmatch -host 1`, 8 bots | PPO, in-repo ckpt | yes, Apache-2.0 |
| repro (Taketani) | same `deathmatch_simple.wad` | same | same | same fork | yes |
| MultiGen | 100 Obsidian maps + **one unnamed ViZDoom deathmatch map** | not stated | 1 ego vs 4 identical agents | **Arnold** (ref [40]) | no statement |
| PlayGen | **names no scenario, map or WAD** | not stated | not stated | random + other agents | no (Doom is a TODO) |
| DIAMOND (ref. point) | CS:GO **Dust II**, not Doom | n/a | human play, 16 Hz | none | dataset public |
| Arnold (our agent) | `full_deathmatch.wad`, 17 maps; train 2-5, test 6-8 | `freedoom2.wad` in repo | deathmatch, 8 bots | DRQN + game features | yes, no LICENSE file |

## Per-work facts

**GameNGen (arXiv 2408.14837).** The level identity is genuinely absent. Across the full HTML (paper + appendix) the string "map" occurs exactly once, as the agent's input: "downscaled versions of the frame images and in-game map, each at resolution 160x120" (https://arxiv.org/html/2408.14837v1). Levels appear only as counts ("2048 trajectories taken in 5 different levels"). No WAD, no IWAD, no `E1M*`/`MAP0*`, no difficulty, no bot count. What *is* stated: the Vizdoom environment; "All image frames ... are at a resolution of 320x240 padded to 320x256"; "we apply each agent action for 4 frames and additionally artificially increase the probability of repeating the previous action". The reward function (A.3) includes "Secret found: 500 points" and "New area: 20 * (1 + 0.5 * L1 distance) points" — campaign constructs, not deathmatch-arena ones, so the 5 levels are very likely single-player campaign maps with monsters. **VERIFY**: that inference is mine, not stated. The project page names nothing and announces no release (https://gamengen.github.io/).

**gameNgen-repro — correction to our existing card.** Our card says "scenario `deathmatch_simple`, level 5, frameskip 4". "Level 5" is **not a map**: `train_ppo_parallel.py` runs an ACS difficulty curriculum via `send_game_command(f"pukename change_difficulty {self.level}")` with `initial_level=1, max_level=5`, and `load_model_generate_dataset.py` pins `"initial_level": 5, "max_level": 5` — the top difficulty of one map. Frame skip is also indirect: the recording env sets `"frame_skip": 1` while the script defines `ACTION_REPEAT = 4` ("To replicate frame_skip in the environment"), so every tic is stored with each action held 4 tics — exactly our per-tic scheme. `deathmatch_simple.cfg` sets `screen_resolution = RES_320X240`, `render_hud = true`, `episode_timeout = 5250`, 6 buttons, and no `doom_map`. Bots: `"n_bots": 8`, with `DOOM_ENV_WITH_BOTS_ARGS = -host 1 -deathmatch +sv_forcerespawn 1 +sv_respawnprotect 1 ...`. Repo Apache-2.0 (https://github.com/arnaudstiegler/gameNgen-repro). Masao-Taketani/GameNGen is a fork shipping the identical WAD.

**MultiGen (arXiv 2603.06679).** Level design: "we generate gameplay sequences on 100 procedurally generated maps with randomized structure. We create these maps using the Obsidian map generator [50]" (https://obsidian-level-maker.github.io/). Multiplayer: "we collect simulated Doom deathmatch sequences in which one pre-trained agent [40] plays against four identical agents ... For this multiplayer study, we train and evaluate on a single map." Reference [40] is "G. Lample and D. S. Chaplot (2018) Playing FPS games with deep reinforcement learning" — **MultiGen's agent is Arnold, the agent we already record with**. The deathmatch map is not named; no map or data release is stated (https://arxiv.org/html/2603.06679v1).

**PlayGen (arXiv 2412.00887).** "we collect 900M frames of transitions based on the ViZDoom ... then sample 200M balanced transitions"; "we place the agent at a random position on the map at the beginning of each sample"; all frames 128x128. No scenario file, WAD, map name, bot count or frame skip appears. The repo ships only Mario inference; "model weight of DOOM" and "training module and dataset" are unticked TODOs (https://github.com/GreatX3/Playable-Game-Generation).

**DIAMOND (reference point).** FPS setting is CS:GO on Dust II, 5.5M frames of human play at 16 Hz — no Doom map at all (arXiv 2405.12399).

**ViZDoom standard scenarios.** `scenarios/` ships paired `.wad`/`.cfg` for basic(+audio/notifications/simpler/rocket), cig (+`cig_with_unknown.wad`), deadly_corridor, deathmatch, defend_the_center, defend_the_line, health_gathering(+supreme), multi_deathmatch, multi_duel, my_way_home, predict_position, take_cover, plus `doom.cfg`/`doom2.cfg`/`freedoom1.cfg`/`freedoom2.cfg` IWAD configs (https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios). Each scenario `.cfg` sets `doom_scenario_path` and no `doom_map`, so each is effectively a one-map WAD; `deathmatch.cfg` sets `doom_skill = 3` and a 4200-tic timeout, `cig.cfg` (the competition map) a 12-minute timeout and `ASYNC_PLAYER`.

**Arnold (our agent).** README: "A package with 17 selected maps that can be used for training and evaluation"; `resources/freedoom2.wad` "DOOM resources file (containing all textures)"; `resources/scenarios/full_deathmatch.wad` "Scenario containing all deathmatch maps" (plus `deathmatch_rockets.wad`, `deathmatch_shotgun.wad`, `defend_the_center.wad`, `health_gathering.wad`). Defaults: `--freedoom "true"`, `--frame_skip 4`, `--n_bots 8`, `--map_ids_train "2,3,4,5"`, `--map_ids_test "6,7,8"`, `--randomize_textures "true"`, `doom_skill=2` (`src/doom/game.py`). `run.sh` gives the competition entries: Track 1 = `deathmatch_rockets` map 1 vs built-in bots; Track 2 = `full_deathmatch` maps 2 and 3; shotgun agent = `deathmatch_shotgun` map 7 (https://github.com/glample/Arnold). Outside deathmatch it ships only `defend_the_center` and `health_gathering` policies — it has no monster-campaign policy.

## Licensing

- **Freedoom** is BSD 3-clause (https://github.com/freedoom/freedoom/blob/master/COPYING.adoc). Frames rendered from Freedoom assets are redistributable with attribution.
- **ViZDoom** code is MIT and explicitly cannot ship id assets: "Unfortunately, we cannot distribute ViZDoom with original Doom graphics. If you own original Doom and Doom 2 games, you can replace Freedoom graphics by placing `doom2.wad` into your working directory" (https://github.com/Farama-Foundation/ViZDoom#readme). So the ViZDoom default render — including the repro's — is Freedoom.
- **Original `doom.wad`/`doom2.wad`/shareware `doom1.wad`** are id Software IP; shareware terms cover the unmodified archive, not derived rendered frames. **VERIFY**: no prior work states a position on this.
- **What prior work did**: GameNGen, MultiGen and PlayGen say nothing about assets or licensing and release no frames. gameNgen-repro publishes 2.47M rendered frames on the Hub under Apache-2.0 code with no asset statement, relying implicitly on ViZDoom's Freedoom default. Arnold ships `freedoom2.wad` but has **no LICENSE file**, and states "Some of the maps and wad files have been borrowed from the ViZDoom git repository" — `full_deathmatch.wad` is of mixed, partly unstated provenance. **VERIFY** its per-map provenance before redistributing the WAD itself; redistributing *frames* is separate and safer.

## Candidate map sets for our dense corpus

No prior Doom world model names a map we could copy: GameNGen's 5 levels and MultiGen's deathmatch map are unnamed and unreleased, and PlayGen names nothing. The only *named, shipped, reproducible* Doom world-model map that exists is `deathmatch_simple.wad`. So "match a prior paper" can only mean matching a regime, with one exception.

**A — keep Arnold arenas 3, 10, 12, 13 (current plan). Recommended.** (a) Regime match is strong but indirect: MultiGen's multiplayer data is Arnold on a ViZDoom deathmatch map, so "same agent, same regime" is citable; no map identity is shared. (b) Agent fit: perfect — Arnold was trained on these maps with 8 bots at frame skip 4. (c) Freedoom assets, clean for frames; the WAD's provenance is the soft spot.

**B — add `deathmatch_simple.wad` as a fifth map. Recommended as an addition.** (a) The only map where an external comparison is possible: both open GameNGen reproductions and 2.47M public frames sit on it. (b) Agent fit: bot deathmatch, 6 buttons, 320x240, HUD on — Arnold's native regime; we would only pin the ACS difficulty at 5, as the repro does. (c) Apache-2.0 WAD, Freedoom assets. Costs one map's recording.

**C — ViZDoom stock scenarios. Not recommended, except `cig.wad` as one held-out unseen map.** (a) Best name recognition, and `cig.wad` is the competition map, but no prior *world model* used any of them. (b) Agent fit is poor: short-timeout monster-spawn arenas, and the `defend_the_center` class is a turn-in-place task with almost no layout to learn. (c) Cleanest license of all: MIT WADs, Freedoom assets.
