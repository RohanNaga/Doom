---
license: cc-by-4.0
task_categories:
- video-classification
- reinforcement-learning
tags:
- doom
- vizdoom
- world-model
- game-engine
- action-conditioned
size_categories:
- 1M<n<10M
---

# Doom deathmatch recordings, 320x240, lossless, every engine tic (Arnold agent, 17 maps)

Every engine tic of ViZDoom deathmatch play at 320x240 RGB with the HUD, weapon and crosshair rendered, stored as lossless PNG bytes with the action applied at that tic, the full button state, health, ammo, kills, deaths, frags, and the player's position and angle. Recorded for the DoomDiT world-model paper (controlled DiT vs U-Net comparison under one recipe).

## Why this exists

The public Doom world-model datasets are JPEG-compressed (quality 85 caps ground truth at about 31 dB PSNR, and the HUD rows at 26 dB), downscaled to 60x80, or single-arena. GameNGen's data was never released. This set is lossless at GameNGen's resolution, spans 17 deathmatch maps, and carries player pose for memory-conditioned models.

## Size

850 episodes, 4,212,561 tics (33.4 hours of play), 50 episodes on each of 17 maps, 51 KB per frame, about 202 GB. Episode length 4,272 to 5,247 tics (150 game-seconds minus the engine's start-up tics). Per-map coverage and action histograms are in the DoomDiT repository under `docs/cards/arnold/`.

## Recording

- Engine: ViZDoom 1.2.4, Freedoom2 assets (shipped with the Arnold repo), `full_deathmatch.wad` maps 1 to 17, 8 built-in bots, deathmatch with respawn, 150 game-seconds per episode.
- Player: the Arnold agent (Lample and Chaplot, AAAI 2017), competition track-2 checkpoint, unmodified and run with its own decision preamble (favourite-weapon switch, anti-stuck turn). It decides every 4 tics; the engine holds the action in between. Ported to torch 2.x with a three-line patch (in the DoomDiT repo).
- Render: `RES_320X240`, RGB24, HUD on, weapon on, crosshair on, decals and particles off.
- 35 tics per second. One row per tic.

## Row semantics

`frame` is the screen at `tic`. `action` (0 to 28, Arnold's combination id) and `buttons` (a 0/1 string over the buttons listed below) are what is applied from this tic to the next. Game variables are read at the same tic. To reproduce the agent's decision rate use every 4th row.

Buttons and action space:

```
{{BUTTONS}}
```

## Files

`data/train-XXXXX.parquet`, one row per tic, columns: episode_id, map_id, tic, action, buttons, health, ammo, kills, deaths, frags, pos_x, pos_y, angle, frame (image). Episodes are contiguous within a shard and never split across shards.

## Splits used in the paper

Episode-level, seeded: 10% of episodes on each of maps 1 to 15 held out for validation; maps 16 and 17 held out entirely as unseen maps. The split file ships with the DoomDiT repository.

## Limitations

Agent play is not human play: Arnold fights and moves purposefully but does not explore like a person, and it never uses doors' `USE` action beyond what its combination set allows. One scenario type (deathmatch); no cooperative or single-player campaign data. Freedoom assets, not the commercial Doom art.

## Citation

DoomDiT paper (2026), and Lample and Chaplot (2017) for the agent, Kempka et al. (2016) for ViZDoom.
