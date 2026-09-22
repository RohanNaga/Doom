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
| unseen maps | `arenas_678/` (all) | 3,000 |

A training run may use a prefix of the train split (the first DoomDiT next-tic runs use a subset because of compute);
that never changes the split. Split by episode, never by frame.

## Columns (one row per tic)

`tic` (int, engine tic inside the episode), `action` (int, Arnold's requested action id, 0 to 28), `buttons` (string of
0/1, the EXECUTED button vector, one character per entry of `buttons.json`; this can differ from the canonical
vector of `action` during Arnold's anti-stuck overrides, so it is the ground-truth control), `health`, `ammo`,
`kills`, `deaths`, `frags`, `pos_x`, `pos_y`, `angle`, `frame` (PNG bytes, 320x240 RGB, HUD on, lossless).

Row semantics: row `i` holds the frame at tic `i` and the control applied FROM tic `i` TO tic `i+1`
(the recorder stores, then steps). A world model predicting frame `i+1` from frames up to `i` is conditioned on
`buttons[i]`. Arnold chooses a new action every 4 tics and holds it, so controls repeat in runs of 4;
a new life starts wherever `deaths` increments (the tic counter keeps running through a death).

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
