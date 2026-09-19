# Dense 4-map corpus (`arnold-train-dense4-v1`)

A second training corpus, kept separate from the original 17-map corpus (`raw_arnold`, 850 episodes). Started Sep 19 2026.

| | Original corpus | Dense 4-map corpus |
|---|---|---|
| Server directory | `/sata2/data/rnagabhi/doom/raw_arnold` | `/sata2/data/rnagabhi/doom/raw_arnold_dense4` |
| Corpus id | (recorded before seeding existed; not seed-reproducible) | `arnold-train-dense4-v1` (seed = stable hash of corpus id and episode id; bit-for-bit reproducible) |
| Maps | Arnold deathmatch arenas 1 to 17 (16 and 17 held out) | arenas 3, 10, 12, 13 |
| Episodes | 850, about 50 per map | 8,000 planned, 2,000 per map (40x the density) |
| Length | 150 game-seconds, 5,250 tics, every tic stored | same |
| Agent, bots, assets | Arnold (Lample and Chaplot 2017), 8 bots, Freedoom assets, frame skip 4 | identical |
| Format | one parquet per episode: `episode_id, map_id, tic, action, buttons, health, ammo, kills, deaths, frags, pos_x, pos_y, angle, frame` (lossless PNG, 320x240 RGB with HUD) | identical |
| Size | 202 GB | about 275 MB per episode, about 2.2 TB at 8,000 |

**Why it exists.** Our models beat "repeat the last frame" by about 2 dB on training maps, while GameNGen reports 29.4 dB with roughly 100x more frames per map on about five levels. This corpus tests data density directly. Planned comparisons: (1) 4 maps at 1M decision frames against the original 17 maps at 1M frames, same recipe and budget (density against diversity); (2) the same 4 maps at 10x and more (scale).

**Why these maps.** In the seen-map evaluation the U-Net and PixArt rank the 15 training maps identically, so difficulty is a property of the map. The four span it and differ visually: map 10 (easiest, 23.8 dB / LPIPS 0.198), maps 3 and 12 (middle, 22.9 to 23.0 dB; outdoor brick, library wood), map 13 (hard, 20.0 dB / LPIPS 0.331; green tech). Their mean is slightly easier than the 15-map mean (22.4 against about 21.7 dB), which must be stated next to any number from this corpus.

**Evaluation.** Unchanged and shared with the original corpus: the seeded seen-map evaluation episodes on these four maps (a subset of `raw_arnold_eval/seen`), plus the 2 unseen arenas and the 13 curated unseen maps. No dense-corpus episode shares a seed with any evaluation episode, because the corpus ids differ.

**Reproduce.** `bash scripts/spiderman/record_dense4.sh [workers] [episodes]` on Spiderman; resume-safe. Measured throughput: one worker records about 36 tics per second, so one episode takes about 146 seconds per worker and throughput scales with the worker count. `RECORDING_LOG.txt` in the corpus directory records every start and finish with the git revision.

**Not yet done.** Latent encoding (decision frames only, as for the original corpus), the train/validation split file, and the md5 manifest.
