# Dense corpus

A second recording, kept separate from the original 17-map corpus (`raw_arnold`, 850 episodes). Started Sep 19 2026. **Every tic is stored, lossless.** It exists to test data density per map directly: our models beat "repeat the last frame" by about 2 dB on training maps, while GameNGen reports 29.4 dB with roughly 100x more frames per map on about five levels.

**What to train on and what to test on is not decided by this recording.** Three segments are recorded; the split is chosen later, from the segments that exist.

## Segments

| Segment | Maps | Corpus id | Directory | Recorded on |
|---|---|---|---|---|
| `arenas` | 2, 3, 4, 5 of `full_deathmatch` | `arnold-train-dense-v1` | `raw_arnold_dense/arenas` | Spiderman |
| `arenas_678` | 6, 7, 8 of `full_deathmatch` | `arnold-dense-arenas678-v1` | `raw_arnold_dense/arenas_678` | Superman |
| `dm_simple` | `deathmatch_simple` MAP01, stored as `map_id` 101 | `arnold-dense-dmsimple-v1` | `raw_arnold_dense/dm_simple` | not recorded — blocked, see below |

**Why these maps.** Arenas 2 to 5 are the Arnold paper's training maps and 6 to 8 its test maps (Arnold's README: `--map_ids_train "2,3,4,5" --map_ids_test "6,7,8"`). The split is the agent authors' and was fixed **before looking at any model score of ours**. `deathmatch_simple` is the only named map in prior Doom world-model work — both open GameNGen reproductions use it — which is what would make a number on it comparable to anything published.

An earlier version of this corpus used arenas 3, 10, 12, 13, chosen by reading our own per-map evaluation scores. That selects the training set on the outcome being measured, and Rohan rejected it on Sep 19 2026. 134 episodes of that aborted corpus remain in `$D/raw_arnold_dense4` under corpus id `arnold-train-dense4-v1` and are unused; nothing downstream should read that directory.

## What a file contains

One parquet per episode, `ep_XXXXX.parquet`, 150 game-seconds, about 5,100 tics (5,250 is the timeout; deaths cost a few), about 275 MB.

| column | type | meaning |
|---|---|---|
| `episode_id`, `map_id` | int32, int8 | constant down the episode; `map_id` 101 is `deathmatch_simple` MAP01 |
| `tic` | int32 | strictly increasing, one row per tic |
| `action`, `buttons` | int16, string | Arnold's action id, and the 0/1 button vector actually applied, from this tic to the next |
| `health`, `ammo`, `kills`, `deaths`, `frags` | int16 | read at the same tic; `deaths` increments on the respawn row, which is what cuts a life |
| `pos_x`, `pos_y`, `angle` | float32 | same tic |
| `frame` | binary | lossless PNG, 320x240 RGB, HUD, weapon and crosshair on |

**Seeds.** Every RNG stream (python, numpy, torch, vizdoom) is seeded from a SHA-256 of `(scheme, corpus id, episode id, stream)` — no worker id, no pid, no clock — so the corpus is reproducible from the corpus id alone and is independent of the worker count. Each segment has its own corpus id, so no two segments and no evaluation episode ever share a seed.

**Episode metadata** lives in the parquet schema under `doomdit_episode`: seed scheme, corpus id, episode id, stored map id, and the seeds. Four further keys appear only when the option that needs them was used — `engine_map_id`, `map_id_offset` and `wad` under `--map-id-offset`, `init_game_commands` under `--init-game-command`, `bots` under `--zdoom-bots`, `stored_tic_stride` under `--decision-only`. A default per-tic run therefore writes byte-for-byte what the recorder wrote before those options existed, so a segment already being recorded can be resumed by a newer build. Readers must default a missing key: offset 0, no commands, stride 1.

## Recording

`SEGMENT=arenas bash scripts/spiderman/record_dense.sh [workers=32] [episodes=8000]`, resume-safe: an episode whose parquet exists is skipped. 32 workers at `nice 10`, `OMP_NUM_THREADS=1`, PNG level 6. About 670 episodes/h at 32 workers on Spiderman.

**PNG level.** Lossless at every setting; the level only trades encode time for bytes. One per-tic episode, one worker at `nice 19`, the same seeded episode each time (2,019 tics):

| level | wall | tics/s | KB per frame | parquet |
|---|---|---|---|---|
| 6 | 67 s | 30 | 54.7 | 112.2 MB |
| 3 | 44 s | 46 | 59.5 | 122.0 MB |
| 1 | 38 s | 52 | 64.2 | 131.5 MB |

Level 3 records **1.52x faster for 8.8% more bytes**, level 1 1.76x faster for 17.3% more. Over 8,000 episodes that is roughly 12.2 h and 2.33 TB at level 6 against 7.9 h and 2.54 TB at level 3. The level 6 row matches what the live job actually does (31 to 34 tics/s per worker), which is the check that the projection is not fantasy. **The default stays 6** so a relaunch reproduces the job now running; mixing levels within a segment is safe, because every level decodes to the same array.

## `deathmatch_simple` does not work under Arnold yet

`SEGMENT=dm_simple` refuses to run without `DM_SIMPLE_OK=1`, so that nobody produces thousands of empty episodes. Measured over full Arnold episodes: 0 kills, 0 deaths, 0 frags, health flat at 100, ammo flat at 50 — with Arnold's scripted marines and again with `--zdoom-bots`.

- **`pukename change_difficulty 5` is not the fix.** Driving the map straight from ViZDoom with the reproduction's own `deathmatch_simple.cfg`, with and without `-deathmatch`, and with and without the command, no monster ever spawns: the only non-player objects are pickups (`Medikit`, `ShellBox`, `Clip`, `Shotgun`) and `BulletPuff`. The flag `--init-game-command` exists and works, and the segment sends the command, but it changes nothing here.
- **The map itself is fine.** With the reproduction's `DOOM_ENV_WITH_BOTS_ARGS` plus `removebots` and eight `addbot` calls, 4 to 5 opponents are visible at once and `DeadDoomPlayer` labels appear throughout. The WAD is byte-identical to the reproduction's copy (md5 `421bc939f1015c25d09a9bbcc6d073f4`).
- **So the gap is Arnold's game setup, and it has not been found.** Arnold's deathmatch scenario hardcodes `use_scripted_marines=True` (`src/doom/scenarios/deathmatch.py:106`), whose ACS script lives in `full_deathmatch.wad` and silently adds nobody on another WAD. `--zdoom-bots` switches Arnold to `addbot`, which is what works in the bare probe, but under Arnold it still yields no engagement.

Next step: diff Arnold's argument list against `DOOM_ENV_WITH_BOTS_ARGS` item by item (Arnold adds `+sv_noautoaim`, `+sv_spawnfarthest`, `+freelook`, `+sv_cheats 1` and `set_doom_skill`; the reproduction adds `+viz_nocheat 0`, `+cl_run 1`, `+sv_nocrouch 1`, `+sv_noexit 1`), or enable the labels buffer in a recorded episode and count `DoomPlayer` directly. If it cannot be made to work, the map is exploration-only and that must be stated wherever a number from it is reported.

## How to verify a segment

```
python verify_dense.py --dir $D/raw_arnold_dense/arenas --expect-maps 2,3,4,5 \
    --expect-corpus-id arnold-train-dense-v1 --min-tics 4800 --sample-frames 4 \
    --manifest $D/raw_arnold_dense/arenas/md5.txt --out $D/raw_arnold_dense/arenas/verify.json
```

It reports episode and tic counts per map, mean/min/max tics per episode, lives per map, bytes per episode, and the corpus ids present; it exits non-zero and names the file for a short or empty episode, a non-increasing `tic`, a reused episode id, two episodes sharing seeds, an unexpected map or corpus id, a column missing or of the wrong type, and two row semantics mixed into one directory. `--manifest` writes the md5 manifest in `md5sum` format. It needs no GPU, no ViZDoom and no torch, and can be run on a segment while it is still growing. `verify_corpus.py` is the different, later check: it compares two *latent* directories after re-encoding.

## Decision-only recording: measured, kept, never the default

`record_arnold.py --decision-only` stores one row per agent decision and advances the engine with `make_action(buttons, 4)`, recording `stored_tic_stride: 4`; `encode_parquet.py`, `transition_stats.py` and `transitions.py` take the stride from there. **It is not used for this corpus** — every recording stores every tic — and it is kept only as an opt-in.

Measured at 8 workers on 150 game-second episodes: 2.01x faster (60 against 26 tics/s per worker) and 4.05x fewer bytes. It is not the same rollout: two seeded episodes recorded both ways diverge within a few game-seconds, and a no-bots control diverges too, so the cause is not a death landing inside a skip — ViZDoom does not step identically under `make_action(buttons, 4)` and four `make_action(buttons, 1)` calls. Each mode is reproducible on its own terms (the same episode recorded twice in one mode is byte-identical), so the stepping mode is a third input to the trajectory alongside the corpus id and the episode id.

**Not yet done.** Latent encoding, the train/validation split file, and the md5 manifests for the finished segments.
