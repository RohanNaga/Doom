# Dense corpus

A second recording, kept separate from the original 17-map corpus (`raw_arnold`, 850 episodes). Started Sep 19 2026. **Every tic is stored, lossless.** It exists to test data density per map directly: our models beat "repeat the last frame" by about 2 dB on training maps, while GameNGen reports 29.4 dB with roughly 100x more frames per map on about five levels.

**What to train on and what to test on is not decided by this recording.** Three segments are recorded; the split is chosen later, from the segments that exist.

## Segments

| Segment | Maps | Corpus id | Directory | Recorded on |
|---|---|---|---|---|
| `arenas` | 2, 3, 4, 5 of `full_deathmatch` | `arnold-train-dense-v1` | `raw_arnold_dense/arenas` | Spiderman |
| `arenas_678` | 6, 7, 8 of `full_deathmatch` | `arnold-dense-arenas678-v1` | `raw_arnold_dense/arenas_678` | Superman |
| `dm_simple` | `deathmatch_simple` MAP01, stored as `map_id` 101 | `arnold-dense-dmsimple-v1` | `raw_arnold_dense/dm_simple` | not recorded — engine blocker fixed Sep 20, still gated, see below |

**Why these maps.** Arenas 2 to 5 are the Arnold paper's training maps and 6 to 8 its test maps (Arnold's README: `--map_ids_train "2,3,4,5" --map_ids_test "6,7,8"`). The split is the agent authors' and was fixed **before looking at any model score of ours**. `deathmatch_simple` is the only named map in prior Doom world-model work — both open GameNGen reproductions use it — which is what would make a number on it comparable to anything published.

**The unseen scoring subset is `arenas_678` ids 60:120** (`release/dense_split.json`, 20 per map). It was declared as 0:60 on 2026-09-21 and replaced on 2026-09-22 before any model was scored on it. `arenas_678` was started three times (16, 20, then 16 workers), and every restart gave every worker a fresh first episode (k = 0), the only kind in which Arnold's weapon-select requests execute (see the `buttons` section below). Measured from the raw `buttons` column, 51 of ids 0:60 are k = 0 and none of 60:120 is. Validation and test hold no k = 0 episode, so 0:60 would have tested map transfer and a control-regime shift at once. `make_dense_eval_splits.py --refuse-worker-first` refuses to publish an unseen corpus that holds one.

An earlier version of this corpus used arenas 3, 10, 12, 13, chosen by reading our own per-map evaluation scores. That selects the training set on the outcome being measured, and Rohan rejected it on Sep 19 2026. 134 episodes of that aborted corpus remain in `$D/raw_arnold_dense4` under corpus id `arnold-train-dense4-v1` and are unused; nothing downstream should read that directory.

## What a file contains

One parquet per episode, `ep_XXXXX.parquet`, 150 game-seconds, about 5,100 tics (5,250 is the timeout; deaths cost a few), about 275 MB.

| column | type | meaning |
|---|---|---|
| `episode_id`, `map_id` | int32, int8 | constant down the episode; `map_id` 101 is `deathmatch_simple` MAP01 |
| `tic` | int32 | strictly increasing, one row per tic |
| `action`, `buttons` | int16, string | Arnold's action id, and its **requested** control list rendered as 0/1, from this tic to the next. 9 characters normally, longer when Arnold appends a weapon-select press; the engine executes only the first 19 entries, so the executed control is `buttons[:19]` right-padded with 0 (see below) |
| `health`, `ammo`, `kills`, `deaths`, `frags` | int16 | read at the same tic; `deaths` increments on the respawn row, which is what cuts a life |
| `pos_x`, `pos_y`, `angle` | float32 | same tic |
| `frame` | binary | lossless PNG, 320x240 RGB, HUD, weapon and crosshair on |

**The `buttons` column is variable width.** It is Arnold's requested control list, not a fixed vector: 9 characters on about 95% of rows, 12 to 17 when its favourite-weapon block appends a weapon-select press that still lands inside the engine's 19 buttons, and 112 to 2,506 on about 5% of rows, with a single extra `1` far past index 18. ViZDoom's `setAction` reads `actions[i]` only for `i < availableButtons.size()` (19) and zero-fills the rest (ViZDoom 1.2.4 `src/lib/ViZDoomGame.cpp:147-178`), so the executed control is `buttons[:19].ljust(19, "0")` and a trailing `1` beyond index 18 was never executed.

The cause is in Arnold: `add_buttons` re-appends the ten `SELECT_WEAPON%i` names to the shared `available_buttons` list on every `Game.start()` (`src/doom/actions.py:197-199`, `game.py:485`) while ViZDoom deduplicates its own list, so `SELECT_WEAPONj` maps to `9 + 10*k + j` with k zero-based (`k = starts - 1`). `record_arnold.py:167` starts the game once per recorded episode, so k is the worker's episode count so far and only a worker's first episode (k = 0) can execute a requested switch. This is a behaviour artefact of the recorded agent, not of the world-model data. `release/DATASET_CARD.md` has the full account, including the engine and Arnold references and the bounded negative finding over the four sources that were searched.

Rows longer than 19 and rows that lost a switch are counted separately (`rows_over_executed` against `unexecuted_switch_rows`): an anti-stuck row can be 2,503 characters with nothing requested out there. An empty `buttons` cell is refused rather than padded, because the recorder always renders at least Arnold's nine entries.

`encode_parquet.py` stores the normalised 19-character executed vector in the sidecar, plus `buttons_raw_len` and `switch_requested_index` so the request is recoverable; the parquet is never altered. `encode_parquet.py --normalize-sidecars` repairs an already-encoded corpus in place. `buttons_report.py` prints the per-episode width distribution and the executed/unexecuted switch counts.

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

## `deathmatch_simple`: found and fixed (Sep 20 2026)

**`--zdoom-bots` was never reaching Arnold.** It was not a game argument, a cvar, a skill level or the ACS script. The recorder does not build Arnold's `Game`; Arnold's `deathmatch.py` does, and it passes every choice as an explicit keyword, including `use_scripted_marines=True` at `src/doom/scenarios/deathmatch.py:106`. The recorder injected its own settings by rebinding that module's `Game` to a `functools.partial`, and partial keywords are **defaults a call site overrides**: `partial(Game, use_scripted_marines=False)(use_scripted_marines=True)` builds a Game with scripted marines. So every `--zdoom-bots` run silently kept Arnold's ACS marines, whose script lives in `full_deathmatch.wad` and adds nobody on another WAD. `screen_resolution` was the only injected setting that ever took effect, and only because that one line of Arnold's call site is commented out.

`record_arnold.py` now applies its settings *after* the caller's with `forced_game`, covered by `paper/fixtures/test_recorder_game_kwargs.py`.

**Proof.** Two 60-game-second Arnold episodes on `deathmatch_simple`, same corpus id so both draw the same seeds, same flags (`--zdoom-bots --map-id-offset 100 --init-game-command "pukename change_difficulty 5"`, `--n_bots 8`), one process at a time. The only difference is how the recorder injects its keywords:

| | rows / tics | kills | deaths | frags | health | ammo |
|---|---|---|---|---|---|---|
| `forced_game` (fixed) | 1746 | 0 | **9** | **2** | 5 to 100, 17 changes | 21 to 50, 43 changes |
| `functools.partial` (control) | 2097 | 0 | 0 | 0 | flat 100 | flat 50 |

The control reproduces the previously recorded symptom exactly — 0 kills, 0 deaths, 0 frags, health flat at 100, ammo flat at 50 — which is what makes the mechanism above the explanation for all of it rather than one more suspect. The fixed episode ends early because the agent died nine times in sixty game-seconds; `kills` stays 0 in both because `KILLCOUNT` counts monsters, and a bot killed in deathmatch scores `FRAGCOUNT`.

Two earlier findings still stand and are now explained: `pukename change_difficulty 5` changes nothing here (driving the map straight from ViZDoom with the reproduction's own `deathmatch_simple.cfg`, with and without `-deathmatch` and with and without the command, no monster ever spawns; the only non-player objects are pickups and `BulletPuff`), and the map itself was never the problem (with the reproduction's `DOOM_ENV_WITH_BOTS_ARGS` plus `removebots` and eight `addbot` calls, 4 to 5 opponents are visible at once; the WAD is byte-identical to the reproduction's copy, md5 `421bc939f1015c25d09a9bbcc6d073f4`).

**Still gated.** `SEGMENT=dm_simple` continues to refuse without `DM_SIMPLE_OK=1`. The engine blocker is gone, but whether to spend the disk and the days on this segment is a corpus decision, not a bug fix, and one 60-second episode is not a corpus-scale check. Before lifting it, record a handful of full 150-second episodes and confirm deaths and frags are non-zero across all of them.

## How to verify a segment

```
python verify_dense.py --dir $D/raw_arnold_dense/arenas --expect-maps 2,3,4,5 \
    --expect-corpus-id arnold-train-dense-v1 --min-tics 4600 --sample-frames 4 \
    --manifest $D/raw_arnold_dense/arenas/md5.txt --out $D/raw_arnold_dense/arenas/verify.json
```

**`--min-tics` is a death budget in disguise, so do not set it tight.** Episode length is determined
entirely by how often the agent died: fitting rows against lives over 40 episodes of `arenas_678`
gives `rows = -39.00 x lives + 5285.9` with R-squared 1.0000 and a worst residual of 3.9 tics. Every
death costs exactly 39 tics of un-recorded respawn. So `--min-tics 4800` is really "at most about 12
deaths", which flagged 188 of 1,466 `arenas_678` episodes as short when nothing was wrong with them —
maps 6 to 8 are Arnold's *test* arenas and the agent dies more there (8.6, 11.7 and 7.1 lives per
episode against 3.2 to 8.6 on the training arenas). Use 4600, below the lowest observed length, and
read the tic distribution in the report rather than trusting a flat floor.

It reports episode and tic counts per map, mean/min/max tics per episode, lives per map, bytes per episode, and the corpus ids present; it exits non-zero and names the file for a short or empty episode, a non-increasing `tic`, a reused episode id, two episodes sharing seeds, an unexpected map or corpus id, a column missing or of the wrong type, and two row semantics mixed into one directory. `--manifest` writes the md5 manifest in `md5sum` format. It needs no GPU, no ViZDoom and no torch, and can be run on a segment while it is still growing. `verify_corpus.py` is the different, later check: it compares two *latent* directories after re-encoding.

## Decision-only recording: measured, kept, never the default

`record_arnold.py --decision-only` stores one row per agent decision and advances the engine with `make_action(buttons, 4)`, recording `stored_tic_stride: 4`; `encode_parquet.py`, `transition_stats.py` and `transitions.py` take the stride from there. **It is not used for this corpus** — every recording stores every tic — and it is kept only as an opt-in.

Measured at 8 workers on 150 game-second episodes: 2.01x faster (60 against 26 tics/s per worker) and 4.05x fewer bytes. It is not the same rollout: two seeded episodes recorded both ways diverge within a few game-seconds, and a no-bots control diverges too, so the cause is not a death landing inside a skip — ViZDoom does not step identically under `make_action(buttons, 4)` and four `make_action(buttons, 1)` calls. Each mode is reproducible on its own terms (the same episode recorded twice in one mode is byte-identical), so the stepping mode is a third input to the trajectory alongside the corpus id and the episode id.

## What each segment measured (Sep 20 2026)

`arenas` finished at 06:30 UTC; `verify_dense.py` passes with **zero problems**. `arenas_678` was
still recording and has no manifest, because a manifest of a growing directory is stale as written.

| | `arenas` | `arenas_678` (partial) |
|---|---|---|
| episodes | **8,000** (2,000 per map, exactly) | 1,466 of 3,000 (491 / 487 / 488) |
| tics | 40,285,059 | 7,225,168 |
| tics per episode, mean / min / max | 5,036 / 4,662 / 5,247 | 4,928 / 4,642 / 5,169 |
| bytes per episode | 238 MB | 259 MB |
| total | **1.732 TiB** | 254 GB |
| `verify_dense.py` | **ok**, 0 problems | ok apart from `--min-tics`, see above |
| md5 manifest | `arenas/md5.txt`, 8,000 lines | not yet (still recording) |

Model-free statistics, 200-episode sample per segment, from `persistence_stats.py --per-map`.
Persistence PSNR is the floor a world model has to beat; the gap-4 column is the decision spacing
the models actually predict at.

| map | gap 1 | gap 2 | **gap 4** | gap 8 | near-static | cells, unioned | cells per episode | lives per episode |
|---|---|---|---|---|---|---|---|---|
| 2 | 21.90 | 20.62 | **19.55** | 18.82 | 1.65% | 872 | 227 | 5.67 |
| 3 | 22.03 | 20.62 | **18.94** | 17.92 | 2.29% | 619 | 289 | 8.08 |
| 4 | 23.05 | 21.55 | **20.18** | 19.31 | 0.23% | 1,339 | 327 | 3.21 |
| 5 | 20.34 | 18.81 | **17.57** | 16.81 | 0.54% | 626 | 286 | 8.64 |
| 6 | 20.83 | 19.49 | **18.25** | 17.46 | 1.18% | 657 | 184 | 8.62 |
| 7 | 20.11 | 18.21 | **16.76** | 15.88 | 0.74% | 339 | 156 | 11.70 |
| 8 | 23.10 | 21.53 | **20.15** | 19.14 | 3.65% | 508 | 237 | 7.11 |

**The maps are not equally hard, and the spread is large.** At the decision spacing the floor runs
from 16.76 dB (map 7) to 20.18 dB (map 4), a 3.4 dB range — bigger than the gap between any two of
our model rows. Any per-map number has to be read against its own map's floor, and a train/test
split across these maps changes the apparent difficulty on its own. Map 4 is the easiest and the most
explored (1,339 cells); map 7 is the hardest, the least explored (339 cells) and the deadliest
(11.7 lives per episode).

Pooled over the four training arenas the floor at gap 4 is 19.02 dB, which lands within 0.1 dB of the
18.94 dB measured on the separate evaluation corpus — the cross-check that the instrument agrees
across corpora.

**Not yet done.** Latent encoding (see `scripts/spiderman/encode_dense.sh`; every tic would be
467 GiB, decision frames 117 GiB), the train/validation split file, and the md5 manifest for
`arenas_678` once its recorder drains.
