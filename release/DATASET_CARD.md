---
license: cc-by-4.0
pretty_name: DoomDiT dense Arnold deathmatch recordings
size_categories:
  - 10M<n<100M
task_categories:
  - image-to-image
tags:
  - world-model
  - doom
  - vizdoom
  - video-prediction
  - game-engine
---

# DoomDiT dense Arnold recordings

Lossless, per-tic recordings of the Arnold agent (Lample and Chaplot, AAAI 2017) playing ViZDoom deathmatch
with 8 bots on Freedoom assets, made for training action-conditioned world models. Every engine tic
(35 per second) is stored with the executed control vector, so the data can be used at any frame stride.
Recorded September 2026 at CMU for the DoomDiT project (Rohan Nagabhirava, Keerthana Chirumamilla).

## What is here

| Folder | Maps | Episodes | Tics | Bytes | Purpose |
|---|---|---|---|---|---|
| `arenas/` | `full_deathmatch.wad` arenas 2, 3, 4, 5 (the Arnold paper's training maps), 2,000 per map | 8,000 | 40.3M | 1.90 TB | training, validation and seen-map test |
| `arenas_678/` | arenas 6, 7, 8 (the Arnold paper's test maps), 1,000 per map | 3,000 | 14.8M | 0.78 TB | unseen-map evaluation |
| `arnold/` | arenas 1 to 17, about 50 per map | 850 | 4.2M | 0.20 TB | the original 17-map corpus the first DoomDiT rows trained on |
| `arnold_eval/{seen,unseen,unseen2}/` | seen: the 15 training maps of `arnold/`; unseen: arenas 16, 17; unseen2: 13 curated maps | 60 / 20 / 130 | | 50 GB | the seeded evaluation corpora every reported DoomDiT number was scored on |

Every episode is one parquet file, `ep_XXXXX.parquet`, 150 game-seconds (about 5,000 tics, about 260 MB),
with the episode's provenance in the parquet schema metadata (key `doomdit_episode`: corpus id, episode id,
map id, WAD, seeds, kills, deaths). Episode ids in `arenas/` cycle through the four maps
(`map = [2,3,4,5][id % 4]`), so every id range is map-balanced.

## Fixed split (`dense_split.json`)

Fixed by episode id BEFORE any model was trained on this data, and never to be moved:

| Split | `arenas/` ids | Episodes |
|---|---|---|
| train | 0 to 5999 | 6,000 |
| val | 6000 to 6999 | 1,000 |
| test | 7000 to 7999 | 1,000 |
| unseen maps | `arenas_678/` (all; never trained on) | 3,000 |

A training run may use a prefix of the train split (the first DoomDiT next-tic runs use a subset because of compute);
that never changes the split. Split by episode, never by frame.

The first next-tic runs score subsets: val `arenas/` 6000 to 6099, test 7000 to 7099, and unseen `arenas_678/`
**60 to 119** (20 per map; `next_tic_runs.unseen_ids` and `segments.arenas_678.ranges.unseen` in the json).
The unseen subset was declared as 0 to 59 on 2026-09-21 and replaced on 2026-09-22, before any model was scored
on it: 51 of those 60 are worker-first episodes (k = 0, see the weapon-select section below), a control regime that
validation and test never contain. Ids 60 to 119 hold none. The json's `history` records the measurement.

## Columns (one row per tic)

`tic` (int, engine tic inside the episode), `action` (int, Arnold's requested action id, 0 to 28), `buttons` (string of
0/1, Arnold's REQUESTED control list, one character per entry in `buttons.json` order; **9 characters normally,
longer when Arnold appends a weapon-select press**, and the engine executes only the first 19 entries, so the
executed control is `buttons[:19]` right-padded with 0 — see the next section), `health`, `ammo`,
`kills`, `deaths`, `frags`, `pos_x`, `pos_y`, `angle`, `frame` (PNG bytes, 320x240 RGB, HUD on, lossless).

The executed control can differ from the canonical vector of `action` during Arnold's anti-stuck overrides, so it,
not the action id, is the ground-truth control.

Row semantics: row `i` holds the frame at tic `i` and the control applied FROM tic `i` TO tic `i+1`
(the recorder stores, then steps). A world model predicting frame `i+1` from frames up to `i` is conditioned on
`buttons[i]`. Arnold chooses a new action every 4 tics and holds it, so controls repeat in runs of 4;
a new life starts wherever `deaths` increments (the tic counter keeps running through a death).

## Variable-width `buttons`, and the weapon switches Arnold asked for but never got

`buttons` is what Arnold **requested**, not a fixed-width vector. `record_arnold.py:255` renders the control list its
`decision_buttons` built and hands the *same list* to `make_action` at line 273, so the string is exactly the
submitted control and nothing about it is lost or reordered.

**Widths.** About 95% of rows are 9 characters (MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, MOVE_LEFT,
MOVE_RIGHT, ATTACK, SPEED, CROUCH). 12 to 17 characters appear when Arnold's favourite-weapon block appends
`[False] * mapping["SELECT_WEAPON%i"] + [True]` and the resulting index still falls inside the engine's 19 buttons.
About 5% of rows are 112 to 2,506 characters, with a single extra `1` far past index 18.

**Executed control.** ViZDoom's `setAction` loops over `availableButtons.size()`, which is 19
(the nine above plus SELECT_WEAPON0 to SELECT_WEAPON9), reads `actions[i]` only for `i < 19`, zero-fills anything
missing, and never reads anything beyond; no error is raised (ViZDoom 1.2.4, `src/lib/ViZDoomGame.cpp:147-178`).
`advance_action` takes no vector. So the control the engine applied at a tic is exactly

    executed = buttons[:19].ljust(19, "0")

for a non-empty string (an empty `buttons` cell is refused rather than padded: the recorder always renders at least
Arnold's nine entries, so an empty one means the row stored no control at all). A trailing `1` beyond index 18 is
**not** a weapon switch: it was never executed. `buttons.json` is read out of the engine itself
(`record_arnold.py:319-350`) and lists exactly those 19 buttons; no `_vizdoom.cfg` exists in the recorder's
directories that could change the list.

**Two statistics, not one.** A row longer than 19 characters and a row that lost a weapon switch are different
things, and reporting either as the other hides cases. An anti-stuck row can be 2,503 characters with no `1` past
index 18, so nothing was requested out there and nothing was lost; a 14-character row can carry a switch at index
13 that the engine *did* perform. `buttons_report.py` and the encoder's per-episode summary therefore count
`rows_over_executed` (raw length > 19) and `unexecuted_switch_rows` (the highest `1` past index 8 sits at 19 or
beyond) separately, alongside `executed_switch_rows`.

**Mechanism.** Arnold's `add_buttons` appends the ten `SELECT_WEAPON%i` names to the *shared* `available_buttons`
list on **every** `Game.start()` (`src/doom/actions.py:197-199`, `game.py:485`) and its mapping keeps the last index,
while ViZDoom deduplicates its own button list (`ViZDoomGame.cpp:367-372`) and therefore still has 19. After k starts
in one recorder process, `SELECT_WEAPONj` maps to `9 + 10*k + j`, where **k is zero-based: `k = starts - 1`**, so
the 250th `Game.start()` is k = 249 and puts `SELECT_WEAPON0` at index 2,499 and `SELECT_WEAPON9` at 2,508.
`record_arnold.py:167` calls `game.start(...)` once per recorded episode, so **k is that worker's episode count so
far**: only a worker's first episode (k = 0, one start behind it) can put a switch inside the engine's 19 buttons
— `9 + 0 + j <= 18` for every weapon — and in every later episode the requested
switch was silently dropped. Consequence, and it is a property of the **recorded agent's behaviour**, not of the
world-model data: in those episodes Arnold did not switch weapons on request. It still fires, and weapon changes
from pickups still happen. `buttons_report.py` measures this per episode, including a check that k is constant
inside one episode (the growth is per `Game.start()`, so a within-episode change would falsify the explanation).

**Published Arnold has the same defect.** Its own weapon block (`game.py:625-680`) builds the same oversized list
and passes it to ViZDoom. Its published evaluation calls `start()` once per map (`deathmatch.py:148-153`), so k
stays small there and the effect is mostly invisible.

**No prior report was found in the searched sources.** Searched on 2026-09-22:

- all 19 issues and pull requests on the Arnold repository, and their 40 comments;
- the helper code of its 106 forks;
- the ViZDoom issue tracker for reports about action-vector length;
- MultiGen, which used Arnold as its data collector.

None of them mentions this. That is a bounded negative finding over those four sources, not a proof that nobody
has hit it: mailing lists, private forks, papers that did not publish collection code, and anything after that
date were not searched.

**What the encoder stores.** `encode_parquet.py` writes the normalised 19-character executed vector into
`ep_XXXXX_meta.npz` as a fixed `<U19` column, and keeps the request beside it as `buttons_raw_len` (the raw
string's length) and `switch_requested_index` (the index of the highest `1` past index 8, or -1), so an unexecuted
switch stays recoverable from the sidecar without the parquet. **The raw parquet column is never altered.** A corpus
encoded before this is repaired in place, with no re-encoding, by `encode_parquet.py --normalize-sidecars`.

## Files

- `dense_split.json`: the split above.
- `index_arenas.parquet`, `index_arenas_678.parquet`: one row per episode (id, map, tics, kills, deaths, bytes, md5).
- `md5_*.txt`: md5 of every parquet file, for verification after download.
- `canonical_controls.json`: the modal executed button vector per action id over train ids 0:2000, used to mark
  agent-decision rows (`is_decision`) when the corpus is encoded.
- `buttons.json` in each folder: the game's button list in the order the `buttons` string uses.
- `worker_*.jsonl`, `RECORDING_LOG.txt`: per-episode recording statistics and the recorder's launch log.
- `README_*.txt`: the recorder's own notes for each segment.

## Loading

```python
import pyarrow.parquet as pq, json, io
from PIL import Image
t = pq.read_table("arenas/ep_00000.parquet")
meta = json.loads(pq.read_schema("arenas/ep_00000.parquet").metadata[b"doomdit_episode"])
frame0 = Image.open(io.BytesIO(t["frame"][0].as_py()))
```

`huggingface_hub`: `hf download RohanNaga/doom-dense-arnold --repo-type dataset --include "arenas/ep_00*.parquet"`
fetches training ids 0 to 999; check every file against `md5_arenas.txt`.

## Provenance and reproducibility

Recorder, encoder and training code: https://github.com/RohanNaga/Doom (`record_arnold.py`; `release/DENSE_CORPUS.md`
documents the corpus). Every episode is seeded by a stable hash of (corpus id, episode id), so a recording is
bit-for-bit reproducible with the same Arnold checkpoint (`vizdoom_2017_track2.pth`), ViZDoom build and WADs.
`arenas/` was recorded on one machine (32 workers) and `arenas_678/` on another and streamed; both by the same
recorder at git `103b15e` or later. Assets are Freedoom (BSD licence); the scenario WADs are Arnold's.

The persistence (copy-last-frame) PSNR of this footage, measured on `arnold_eval/seen`: 21.6 dB at a 1-tic gap,
20.2 at 2, 18.9 at 4, 18.1 at 8. Report it beside any prediction PSNR.

## Citation

DoomDiT workshop paper, 2026 (to appear). Until then cite this dataset by its URL and the repository above.
