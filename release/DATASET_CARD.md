---
license: cc-by-4.0
pretty_name: "Doom (Freedoom) world-model corpus: lossless 320x240 Arnold deathmatch, every engine tic"
task_categories:
- video-classification
- reinforcement-learning
- image-to-image
tags:
- doom
- vizdoom
- freedoom
- world-model
- game-engine
- action-conditioned
- video-prediction
size_categories:
- 1M<n<10M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# Doom world-model corpus: 850 lossless Arnold deathmatch episodes on 17 Freedoom-rendered maps, every engine tic

Every engine tic of ViZDoom deathmatch play at 320x240 RGB with the HUD, weapon and crosshair rendered, stored as lossless PNG bytes together with the action applied at that tic, the full button vector, health, ammo, kills, deaths, frags, and the player's position and angle. Recorded for the DoomDiT paper (a warm-start study of small-budget latent diffusion world models on Doom) and released as the benchmark it rests on: a training corpus, an episode-level split, three seeded evaluation corpora, and the definition of the verified transitions every reported number uses.

Facts below are tied to the recorder (`record_arnold.py`), the card generator (`docs/cards/arnold/`), and the project log (`RESEARCH_CONTEXT.md`) in the [DoomDiT repository](https://github.com/RohanNaga/Doom). Anything not yet checked against the server copy is marked `<!-- VERIFY -->`.

## Why this exists

The public Doom world-model datasets are JPEG-compressed (quality 85 caps the reachable PSNR at about 31 dB and the HUD rows at 26 dB; `arnaudstiegler/vizdoom-500-episodes-skipframe-4-lvl5`), downscaled to 60x80 (`p-doom/doom-dataset`), or single-arena. GameNGen and MultiGen released no data. This set is lossless at GameNGen's resolution, spans 17 deathmatch maps, carries the button vector and player pose, and is rendered from Freedoom assets, so it can be redistributed.

## Size

| | |
|---|---|
| Episodes | 850 (50 on each of 17 maps) |
| Tics (rows) | 4,212,561 (33.4 hours of play at 35 tics/s) |
| Episode length | 4,272 to 5,247 tics (mean 4,956); 150 game-seconds minus the engine's start-up tics |
| Decision frames | 1,053,453 on the tic-modulo-4 grid; 1,047,351 verified transitions (99.4%) in 7,296 chains (see below) |
| Frame bytes | 51.3 KB per PNG on average; about 202 GB in total |
| Shards | `data/train-XXXXX.parquet`, 227 shards; whole episodes are appended until a shard holds at least 15,000 rows (3 to 4 episodes, about 0.9 GB), so no episode is split across shards <!-- VERIFY shard count on the server: hf_release_arnold/data -->

Per-map tables, the action histogram, and the 64x64 coverage maps are under `docs/cards/arnold/` (`stats.json`, `DATASET_STATS.md`, `coverage_map_XX.png`). Player coverage of visited cells ranges from 38% (map 16) to 89% (map 15).

## Recording

- Engine: ViZDoom 1.2.4 on the Freedoom Phase 2 IWAD (`freedoom2.wad`, the copy shipped with the Arnold repository) with Arnold's `full_deathmatch.wad` as the scenario PWAD, which supplies the 17 deathmatch arenas MAP01 to MAP17. Geometry of maps 1 to 17 is therefore Arnold's; every texture, sprite and sound is Freedoom's. Deathmatch with respawn against 8 built-in bots, 150 game-seconds per episode.
- Player: the Arnold agent (Lample and Chaplot, AAAI 2017; the agent MultiGen used for its Doom data), `vizdoom_2017_track2` checkpoint, unmodified, run through its own entry point with its decision preamble intact (favourite-weapon switch, anti-stuck manual control). It decides every 4 tics from 29 discrete actions and the engine holds the buttons in between. Ported to torch 2.x with a three-line patch (`third_party/arnold-torch2.patch`).
- Render: `RES_320X240`, RGB24, HUD on, weapon on, crosshair on (`meta/buttons.json`, field `screen`).
- 35 tics per second; one row per tic; PNG compress level 6; parquet written uncompressed with row groups of 256 rows.
- Workers: 20 CPU processes, each recording every 20th episode id; map of episode `e` is `1 + (e mod 17)`.

## Row semantics

`frame` is the screen at `tic`. `action` and `buttons` are what is applied from this tic to the next. Game variables are read at the same tic. `deaths` increments on the respawn row, which is how a life boundary is found.

Two things to know before slicing:

1. The agent's 4-tic decision phase is not the tic-modulo-4 grid. A death cuts a decision short and the next decision starts at the respawn row, so 62.2% of decisions start off the grid and a naive stride-4 slice straddles two actions in 39.1% of 32-frame windows. Use the verified-transition procedure below, not `tic % 4 == 0`, when you need one action per frame pair.
2. The anti-stuck override: when the agent has not moved forward for 30 decisions or not turned for 60, Arnold's preamble applies a fixed "turn right + speed (+ forward)" button vector for 40 tics. These rows keep the agent's requested `action` id but carry the override's `buttons`, so `buttons` is the executed control and `action` the requested one. They are 0.17% of tics; weapon-switch bits raise the fraction of rows whose `buttons` differ from the action's canonical vector to about 9%.

### Verified transitions and chain ids

`transitions.py` in the repository defines the training and evaluation unit. A transition from row `s` to row `s + 4` is accepted when both rows are inside one continuous life, the executed control bits (the first 9 entries of `buttons`) are constant over rows `s .. s+3`, and they equal the canonical vector of the requested action (`meta/canonical_controls.json`, the modal control bits per action id over the whole corpus; overrides are therefore excluded rather than mislabelled). Consecutive accepted sources 4 tics apart form a chain; a context window or rollout never crosses a chain boundary. On this corpus: 1,047,351 verified transitions, 7,296 chains over 7,178 lives, median chain 108 transitions (p10 39, p90 277); encoding every source plus each chain's final target gives 1,054,647 frames. Chain ids are not stored in the parquet; they are a deterministic function of `action`, `buttons` and `deaths` plus the canonical table, and `transitions.valid_transitions` recomputes them in a few seconds per episode. `meta/transition_stats.json` holds the survival statistics. <!-- VERIFY path raw_arnold/card/transition_stats.json on the server -->

## Parquet schema

`data/train-*.parquet` (converted for the Hub by `release/hf_upload.py`): the `frame` column is the `datasets` `Image` feature so the viewer renders it; integer columns are widened to `int64`, floats to `float32`.

| Column | Type (as recorded / on the Hub) | Meaning (source: `record_arnold.py`) |
|---|---|---|
| `episode_id` | int32 / int64 | Episode index, 0 to 849; unique within a corpus |
| `map_id` | int8 / int64 | `full_deathmatch.wad` map, 1 to 17 |
| `tic` | int32 / int64 | Engine tic within the episode, from 0, one row per tic |
| `action` | int16 / int64 | Arnold action id, 0 to 28, the combination requested for this tic (see `meta/buttons.json`) |
| `buttons` | string | 19-character 0/1 string over `available_buttons` in `meta/buttons.json` order: MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, MOVE_LEFT, MOVE_RIGHT, ATTACK, SPEED, CROUCH, SELECT_WEAPON0 to SELECT_WEAPON9. The executed control at this tic |
| `health` | int16 / int64 | `GameVariable.HEALTH` |
| `ammo` | int16 / int64 | `GameVariable.SELECTED_WEAPON_AMMO` |
| `kills` | int16 / int64 | `GameVariable.KILLCOUNT`; counts monster kills, which do not occur in deathmatch, so it is 0 throughout. Use `frags` |
| `deaths` | int16 / int64 | `GameVariable.DEATHCOUNT`; increments on the respawn row |
| `frags` | int16 / int64 | `GameVariable.FRAGCOUNT` (bot kills minus suicides) |
| `pos_x`, `pos_y` | float32 | `POSITION_X`, `POSITION_Y` in map units |
| `angle` | float32 | `ANGLE`, degrees |
| `frame` | binary / Image | PNG bytes of the 320x240x3 screen buffer (HUD included) |

The raw per-episode files written by the recorder (`ep_XXXXX.parquet`, kept as the `eval/` corpora, see below) carry one extra thing: parquet schema metadata under the key `doomdit_episode`, a JSON record `{seed_scheme, corpus_id, episode_id, map_id, seeds: {python, numpy, torch, vizdoom}}`. The Hub conversion does not preserve schema metadata, so for the training shards the same provenance is in `meta/provenance/train/worker_XX.jsonl` (one JSON line per episode: `episode_id, map_id, tics, seconds, png_bytes_mean, kills, deaths, suicides, frags`, plus the seed record for seeded corpora).

Action vocabulary: 29 combinations from `move_fb+move_lr;turn_lr;attack` (`meta/buttons.json`). The corpus is imbalanced: the most frequent action is 24.2% of tics (id 9, 1,021,011 rows) and the majority class over decisions is 23.2%; three actions occur fewer than 400 times.

## Evaluation corpora

Every number in the paper is computed on separate seeded corpora that touched no fitted component (world models, decoder, and IDM judge were all fit on the training split). They are stored as the recorder wrote them, one `ep_XXXXX.parquet` per episode with the `doomdit_episode` metadata intact, because the evaluation scripts (`eval_tf.py --parquet-dir`) read raw frames by episode and tic.

| Corpus | Path | Corpus id (seed input) | Episodes | Maps |
|---|---|---|---|---|
| Seen maps | `eval/seen/` | `arnold-eval-seen-v1` | 60 (4 per map) | 1 to 15, the training maps |
| Unseen maps | `eval/unseen/` | `arnold-eval-unseen-v1` | 20 (10 per map) | 16 and 17, held out from training entirely; same design family as 1 to 15 |
| Unseen2, curated | `eval/unseen2/` | `doomdit-eval-unseen2-curated-v1` | 130 (10 per map) <!-- VERIFY final count; recording on 2026-09-16 --> | Freedoom Phase 2 MAP18, 19, 20, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32 |

Seen and unseen corpora share every setting with the training corpus (Arnold checkpoint, 8 bots, 150 game-seconds, render settings, frame skip 4) and were encoded with the training corpus's canonical table so the transition filter is identical.

**Unseen2 curation, stated in full.** Arnold's `full_deathmatch.wad` holds only MAP01 to MAP17; maps 18 to 32 come from the `freedoom2.wad` IWAD itself, which is single-player content. Run raw, those maps crash the unmodified agent (its game-variable asserts require at most one weapon per slot, and its health embedding covers at most 109 health) and, unlike the training arenas, spawn monsters in a deathmatch game. To make them comparable to the training maps, a PWAD was built that copies every lump of `full_deathmatch.wad` byte-identical and appends MAP18 to MAP32 from the IWAD with the following THINGS removed, and nothing else changed:

- all monsters (22 types, 2,058 things across the 13 kept maps; the training arenas spawn none in deathmatch);
- backpack, chainsaw, super shotgun (weapon-slot and ammo asserts in Arnold's `game.py`);
- soulsphere, megasphere, health bonus (health above 100 overflows Arnold's health embedding).

Kept: deathmatch starts, medikit, stimpack, armor bonus, megaarmor, berserk, chaingun, rocket launcher, plasma rifle, BFG. Totals removed: 2,430 things. MAP21 and MAP27 were dropped before any model was scored because the agent stays inside a 32-unit box for whole episodes there (0 frags, near-static frames, measured over 2 to 3 seeds), which would inflate pixel metrics. The full record (kept maps, every removed thing with type, position and reason, excluded maps, totals) is `eval/unseen2/curation_manifest.json`. The curated PWAD (`full_deathmatch_unseen2.wad`) is required to re-record the corpus; whether it ships in this repository is a licensing decision, see below. <!-- VERIFY: WAD path is provisional (tmp/unseen2/ on Spiderman) and its redistribution is undecided -->

Transfer to maps 16 and 17 is "new arena, same design family"; transfer to unseen2 is "new family, rendered with the same assets", and the paper reports them separately. Per-map agent statistics for unseen2 (frags, deaths, coverage) are in its provenance records. <!-- VERIFY: add per-map unseen2 statistics to docs/cards once the recording finishes -->

## Splits

`splits/split_arnold.json` (`doom_data.make_split_by_map`, seed 0): maps 16 and 17 held out entirely as `unseen_map` (100 episodes); on maps 1 to 15, 10% of episodes per map are `val` (75) and the rest `train` (675). Fields: `train`, `val`, `unseen_map` (lists of episode ids), `meta` (`holdout_maps`, `holdout_frac`, `seed`, `episodes_per_map`). `val` monitors training (checkpoint selection by held-out velocity loss); reported numbers use the evaluation corpora above, whose split files (`splits/split_eval_*.json`) mark every episode as reporting-only.

## Reproducibility

Seed scheme `doomdit-episode-v1` (`record_arnold.episode_seeds`): for each RNG stream in (`python`, `numpy`, `torch`, `vizdoom`) the seed is the first 4 bytes, little-endian, of `sha256(json.dumps([scheme, corpus_id, episode_id, stream]))`. It is a pure function of corpus id and episode id: independent of worker count, process id, or time. ViZDoom is seeded on the instance Arnold constructs, before `init`; the anti-stuck counters are reset per episode. Verified on 2026-09-13: one 60-second episode recorded twice under corpus `smoke-seed-v1` gave identical tics, actions, buttons, game variables and frame hashes (1,980 tics).

Which corpora this covers: the three evaluation corpora were recorded with the seeded recorder and can be re-recorded bit for bit from `(corpus_id, episode_id)`, `meta/buttons.json`, the Arnold checkpoint, the WADs, and the recorder at the commit in their provenance records. **The 850-episode training corpus was recorded on 2026-09-08 to 09, before the seeding existed** (repository commits `64083f9`, `72555c3` vs. seeding in `8f8546c`), so it is reproducible in distribution but not tic for tic; its provenance records carry no seeds.

To re-record an evaluation episode:

```bash
python record_arnold.py --arnold-dir Arnold --out-dir out --map-ids 1-15 --episodes 60 --episode-time 150 \
    --worker-id 0 --num-workers 1 --corpus-id arnold-eval-seen-v1 -- \
    --frame_skip 4 --action_combinations "move_fb+move_lr;turn_lr;attack" --evaluate 1 ...   # Arnold's own flags
```
<!-- VERIFY the full Arnold flag list against record_eval_corpus.sh on Spiderman before publishing it -->

## Licensing

- Freedoom Phase 2 assets (`freedoom2.wad`): BSD licence (The Freedoom Project). This is what makes the frames redistributable. <!-- VERIFY the Freedoom release version used and the exact licence text shipped in the Arnold copy -->
- ViZDoom: MIT licence. The engine it wraps derives from ZDoom, whose components carry their own licences (GPL v3 for ZDoom since 2.4; older parts under the Doom Source Licence and BUILD licence). <!-- VERIFY current ViZDoom 1.2.4 LICENSE text -->
- Arnold agent and `full_deathmatch.wad` (github.com/glample/Arnold): the repository ships no licence file. This corpus does not redistribute the agent's weights. The map geometry of maps 1 to 17 is Arnold's; the frames are renderings of it. Whether the curated `full_deathmatch_unseen2.wad`, which embeds those lumps byte-identical, ships here is an open decision; the fallback is to ship `curation_manifest.json` and the build script so anyone with the Arnold WAD can rebuild it. <!-- VERIFY: Rohan's decision -->
- The recordings themselves (frames, actions, game variables, provenance) are released under CC BY 4.0.

## Intended use

Training and evaluating action-conditioned world models, video prediction, and inverse-dynamics models on a fixed, redistributable Doom benchmark with a stated train/eval separation and persistence references (copy-last, copy-seed). The paper's evaluation protocol, split, verified-transition filter and judges are in the DoomDiT repository and the companion model repository.

## Limitations

- One policy: the 2017 Arnold checkpoint, trained on deathmatch maps 2 to 5. It fights and moves purposefully but does not explore like a person; its behaviour on maps it never saw is a behaviour shift as well as a scene shift. Its action set has no USE button, so it never opens doors by hand.
- One game mode (deathmatch against 8 bots, 150 game-seconds). No campaign, cooperative, or human play.
- Class imbalance: the majority action is 23.2% of decisions; three actions occur fewer than 400 times in 4.2M tics.
- 0.6% of 32-frame windows contain a death (a respawn cut), and 62% of decisions start off the tic-modulo-4 grid; use the verified-transition filter.
- `kills` is zero throughout (monster kills only); `frags` is the deathmatch score.
- The training corpus is not tic-for-tic reproducible (see above); the evaluation corpora are.
- Freedoom assets, not id Software's art: models trained here do not transfer their appearance to commercial Doom.

## Citation

```bibtex
@misc{doomdit2026,
  title  = {DoomDiT: a warm-start study of small-budget latent diffusion world models on Doom},   % VERIFY final title
  author = {Nagabhirava, Rohan and Chirumamilla, Keerthana},
  year   = {2026},
  note   = {Dataset and weights: https://huggingface.co/<hf-user>}   % VERIFY once the repositories exist
}
```

Please also cite Kempka et al. (2016) for ViZDoom, Lample and Chaplot (2017) for the Arnold agent, and the Freedoom project for the assets.
