# Dense corpus

A second training corpus, kept separate from the original 17-map corpus (`raw_arnold`, 850 episodes). Started Sep 19 2026. It tests data density per map directly: our models beat "repeat the last frame" by about 2 dB on training maps, while GameNGen reports 29.4 dB with roughly 100x more frames per map on about five levels.

| | Original corpus | Dense corpus |
|---|---|---|
| Server directory | `/sata2/data/rnagabhi/doom/raw_arnold` | `raw_arnold_dense/arenas`, `raw_arnold_dense/arenas_678`, `raw_arnold_dense/dmsimple` |
| Corpus id | (recorded before seeding existed; not seed-reproducible) | one per segment, see the table below (seed = stable hash of corpus id and episode id) |
| Maps | Arnold deathmatch arenas 1 to 17 (16 and 17 held out) | arenas 2, 3, 4, 5 and arenas 6, 7, 8 of `full_deathmatch`, plus `deathmatch_simple` MAP01 (blocked, see below) |
| Train or test | fixed episode split | **not decided by the recording**: every segment is recorded densely and the split is chosen later (Rohan, Sep 19 2026) |
| Episodes | 850, about 50 per map | 2,000 per map on arenas 2 to 5, 1,000 per map on arenas 6 to 8, 150 game-seconds each (about 5,100 tics, 270 MB), every tic stored |
| Agent, bots, assets | Arnold (Lample and Chaplot 2017), 8 bots, Freedoom assets, frame skip 4 | identical |
| Format | one parquet per episode: `episode_id, map_id, tic, action, buttons, health, ammo, kills, deaths, frags, pos_x, pos_y, angle, frame` (lossless PNG, 320x240 RGB with HUD) | identical |

**Why these maps.** Arenas 2, 3, 4, 5 and the test set 6, 7, 8 are the agent authors' own published split (Arnold's README: `--map_ids_train "2,3,4,5" --map_ids_test "6,7,8"`), so the split is theirs and not ours. `deathmatch_simple` is the only named map in prior Doom world-model work — both open GameNGen reproductions use it — which is what would make a number on it comparable to anything outside this repo.

**Why not the maps in the first plan.** An earlier version of this corpus used arenas 3, 10, 12, 13, chosen by reading our own per-map evaluation scores and spanning them. That selects the training set on the outcome being measured, and Rohan rejected it on Sep 19 2026. 134 episodes of the aborted corpus remain in `$D/raw_arnold_dense4` under corpus id `arnold-train-dense4-v1` and are unused; nothing downstream should read that directory.

## Recording

`TARGET=arenas bash scripts/spiderman/record_dense.sh` on Spiderman; resume-safe, so a relaunch skips finished episodes and continues a corpus. Knobs: `WORKERS` (32), `EPISODES_PER_MAP` (2,000 for `arenas`, 1,000 for `arenas678` and `dmsimple`, 20 for `test-dmsimple`), `MAPS`, `MODE` (`pertic`), `PNG_LEVEL` (6), `N_BOTS` (8), `EPISODE_TIME` (150).

| TARGET | maps | corpus id | output |
|---|---|---|---|
| `arenas` | 2,3,4,5 of `full_deathmatch` | `arnold-train-dense-v1` | `raw_arnold_dense/arenas` |
| `dmsimple` | `deathmatch_simple` MAP01, stored as `map_id` 101, 1,000 episodes | `arnold-dense-dmsimple-v1` | `raw_arnold_dense/dmsimple` |
| `arenas678` | 6,7,8 of `full_deathmatch`, 1,000 per map; recorded on Superman (16 workers) and streamed to Spiderman by rsync | `arnold-dense-arenas678-v1` | `raw_arnold_dense/arenas_678` |
| `test-dmsimple` | `deathmatch_simple`, 20 episodes | `arnold-eval-dense-dmsimple-v1` | `raw_arnold_dense_eval/dmsimple` |

`deathmatch_simple`'s only map is MAP01, which is `full_deathmatch`'s arena 1 as far as the engine is concerned, so `--map-id-offset 100` stores it as 101 and the two directories stay unambiguous when they are merged at encode time. The episode metadata carries the WAD name, the engine's own map id and any console commands sent at spawn, so the label is decodable and not merely unique. Readers must tolerate a metadata key being absent: a corpus recorded before a key existed does not carry it.

**PNG level.** Lossless at every setting; the level only trades encode time for bytes. One worker, map 3, 60 game-seconds, the same seeded episode every time (2,019 tics, 10 kills, 2 deaths), per tic, at `nice 19` on a loaded machine:

| level | wall | tics/s per worker | KB per frame | parquet |
|---|---|---|---|---|
| 6 | 67 s | 30 | 54.7 | 112.2 MB |
| 3 | 44 s | 46 | 59.5 | 122.0 MB |
| 1 | 38 s | 52 | 64.2 | 131.5 MB |

Level 1 records **1.76x faster** than level 6 for 17.3% more bytes; level 3 is 1.52x faster for 8.8% more. Projected over the full 8,000-episode corpus (42M tics) at 32 workers, assuming throughput scales with the per-worker rate:

| level | 32-worker total | 8,000 episodes | corpus size |
|---|---|---|---|
| 6 | about 960 tics/s | 12.2 h | 2.33 TB |
| 3 | about 1,470 tics/s | 7.9 h | 2.54 TB |
| 1 | about 1,660 tics/s | 7.0 h | 2.74 TB |

The level 6 figure matches what the live 32-worker job actually does (31 to 34 tics/s per worker, and the machine's measured ceiling of 900 to 1,000 tics/s), which is the check that the projection is not fantasy. **The default stays 6** so a relaunch reproduces the job now running; `PNG_LEVEL=1` is the measured speed-up, and mixing levels within a corpus is safe because every level decodes to the same array.

## `deathmatch_simple` does not work under Arnold yet

Recording this map is **blocked**, and `TARGET=dmsimple` refuses to run without `DMSIMPLE_OK=1` so that nobody produces thousands of empty episodes.

Measured over a full 150 game-second Arnold episode: 0 kills, 0 deaths, 0 frags, health flat at 100, ammo flat at 50 — with Arnold's scripted marines, and again with `--zdoom-bots`. No opponent ever appears.

- `pukename change_difficulty 5` is not the mechanism. Driving the map straight from ViZDoom with the reproduction's own `deathmatch_simple.cfg`, with and without `-deathmatch`, and with and without the command, no monster ever spawns: the only non-player objects are pickups (`Medikit`, `ShellBox`, `Clip`, `Shotgun`) and `BulletPuff`. The command is kept available as `--init-game-command`, but it changes nothing here.
- The map itself is fine. With the reproduction's `DOOM_ENV_WITH_BOTS_ARGS` plus `removebots` and eight `addbot` calls, 4 to 5 opponents are visible at once and `DeadDoomPlayer` labels appear throughout the episode. The WAD is byte-identical to the reproduction's copy (md5 `421bc939f1015c25d09a9bbcc6d073f4`).
- So the gap is between Arnold's game setup and the reproduction's, and it has not been found. Arnold's deathmatch scenario hardcodes `use_scripted_marines=True` (`src/doom/scenarios/deathmatch.py:106`), whose ACS script lives in `full_deathmatch.wad` and silently adds nobody on another WAD; `--zdoom-bots` switches Arnold to `addbot`, which is what works in the bare probe, but under Arnold it still yields no engagement.

Next step for whoever picks this up: diff Arnold's argument list against `DOOM_ENV_WITH_BOTS_ARGS` item by item (Arnold adds `+sv_noautoaim`, `+sv_spawnfarthest`, `+freelook`, `+sv_cheats 1` and `set_doom_skill`; the reproduction adds `+viz_nocheat 0`, `+cl_run 1`, `+sv_nocrouch 1`, `+sv_noexit 1`), or enable the labels buffer in a recorded episode and count `DoomPlayer` directly. If it cannot be made to work, the map is exploration-only and that must be stated wherever a number from it is reported.

## Decision-only recording: measured, kept, unused

`record_arnold.py --decision-only` stores one row per agent decision instead of one per tic, advancing the engine with `make_action(buttons, 4)`, and records `stored_tic_stride: 4` in its metadata; `encode_parquet.py`, `transition_stats.py` and `transitions.py` all take the stride from there. It is **not used for this corpus**, because Rohan may train on every tic rather than only on decision frames. It is kept as a working option.

Measured at 8 workers on 150 game-second episodes: 2.01x faster wall clock (60 against 26 tics/s per worker, 15.1 against 6.6 decisions/s) and 4.05x fewer bytes (64 MB against 260 MB per episode).

It is not the same rollout. Two seeded episodes recorded both ways diverge within a few game-seconds, and a no-bots control diverges too, so the cause is not a death landing inside a skip: ViZDoom does not step identically under `make_action(buttons, 4)` and four `make_action(buttons, 1)` calls. Each mode is reproducible on its own terms — the same episode recorded twice in the same mode is byte-identical — so the stepping mode is a third input to the trajectory alongside the corpus id and the episode id. The yield that training consumes does carry over: 99.7% of per-tic decision frames become verified transitions against 99.2% in decision mode.

**Not yet done.** Latent encoding, the train/validation split file, and the md5 manifest.
