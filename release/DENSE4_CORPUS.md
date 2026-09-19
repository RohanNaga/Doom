# Dense 4-map corpus (`arnold-train-dense4-v1`)

A second training corpus, kept separate from the original 17-map corpus (`raw_arnold`, 850 episodes). Started Sep 19 2026.

| | Original corpus | Dense 4-map corpus |
|---|---|---|
| Server directory | `/sata2/data/rnagabhi/doom/raw_arnold` | `/sata2/data/rnagabhi/doom/raw_arnold_dense4` |
| Corpus id | (recorded before seeding existed; not seed-reproducible) | `arnold-train-dense4-v1` (seed = stable hash of corpus id and episode id; bit-for-bit reproducible) |
| Maps | Arnold deathmatch arenas 1 to 17 (16 and 17 held out) | arenas 3, 10, 12, 13 |
| Episodes | 850, about 50 per map | 8,000 planned, 2,000 per map (40x the density) |
| Length | 150 game-seconds, 5,250 tics, every tic stored | same per tic; in decision mode the same 150 game-seconds with one row per decision, about 1,243 rows |
| Agent, bots, assets | Arnold (Lample and Chaplot 2017), 8 bots, Freedoom assets, frame skip 4 | identical |
| Format | one parquet per episode: `episode_id, map_id, tic, action, buttons, health, ammo, kills, deaths, frags, pos_x, pos_y, angle, frame` (lossless PNG, 320x240 RGB with HUD) | identical |
| Size | 202 GB | about 275 MB per episode, about 2.2 TB at 8,000 |

**Why it exists.** Our models beat "repeat the last frame" by about 2 dB on training maps, while GameNGen reports 29.4 dB with roughly 100x more frames per map on about five levels. This corpus tests data density directly. Planned comparisons: (1) 4 maps at 1M decision frames against the original 17 maps at 1M frames, same recipe and budget (density against diversity); (2) the same 4 maps at 10x and more (scale).

**Why these maps.** In the seen-map evaluation the U-Net and PixArt rank the 15 training maps identically, so difficulty is a property of the map. The four span it and differ visually: map 10 (easiest, 23.8 dB / LPIPS 0.198), maps 3 and 12 (middle, 22.9 to 23.0 dB; outdoor brick, library wood), map 13 (hard, 20.0 dB / LPIPS 0.331; green tech). Their mean is slightly easier than the 15-map mean (22.4 against about 21.7 dB), which must be stated next to any number from this corpus.

**Evaluation.** Unchanged and shared with the original corpus: the seeded seen-map evaluation episodes on these four maps (a subset of `raw_arnold_eval/seen`), plus the 2 unseen arenas and the 13 curated unseen maps. No dense-corpus episode shares a seed with any evaluation episode, because the corpus ids differ.

**Reproduce.** `MODE=decision MAPS=3,10,12,13 bash scripts/spiderman/record_dense4.sh [workers] [episodes]` on Spiderman; resume-safe, `WORKERS` defaults to 32, `MODE` to `pertic`. `RECORDING_LOG.txt` in the corpus directory records every start and finish with the git revision and the mode.

## Two recording modes

Only every fourth tic ever becomes a training frame: the agent decides every 4 tics and `encode_parquet.py --align-decisions` keeps one frame per decision. `record_arnold.py --decision-only` therefore stops rendering the other three. It advances the engine with the agent's own frame skip, `make_action(buttons, 4)`, and stores one row per decision; the file records `stored_tic_stride: 4` in its parquet metadata and every reader takes the stride from there. Per-tic recording is unchanged and stays the default.

Decision mode writes `raw_arnold_dense4d` under corpus id `arnold-train-dense4d-v1`, not the per-tic directory: one directory must hold one row semantics, because the encoder reads `stored_tic_stride` once per corpus.

**Speed and size.** One worker, map 3, 60 game-seconds, the same seeded episode every time (497 rows, 1,988 tics). PNG is lossless at every level; the level only trades encode time for bytes.

| mode, PNG level | wall | tics/s | decisions/s | KB per frame | parquet |
|---|---|---|---|---|---|
| decision-only, 1 | 21 s | 93 | 23 | 60.8 | 30.9 MB |
| decision-only, 3 | 25 s | 79 | 20 | 56.1 | 28.6 MB |
| decision-only, 6 | 32 s | 63 | 16 | 51.7 | 26.3 MB |
| per-tic, 6 | 70 s | 28 | 7 | 52.8 | 107.0 MB |

Level 1 records 1.52x faster than level 6 for 17.6% more bytes, so decision mode defaults to level 1 and per-tic keeps level 6. Override with `PNG_LEVEL`.

**Throughput.** 8 workers, one 150 game-second episode each, PNG level 6, Spiderman shared and at load 61 at the time, so these are conservative.

| | per-tic | decision-only |
|---|---|---|
| wall for 8 episodes | 207 s | 103 s |
| tics/s per worker | 26 (24 to 30) | 60 (51 to 63) |
| decisions/s per worker | 6.6 | 15.1 |
| rows per episode | about 4,960 | about 1,243 |
| bytes per episode | 260 MB | 64 MB |

Projected at 32 workers, scaling by worker count; the range allows for up to 1.5x per-worker degradation from 8 workers to 32. Decision mode at level 1 is a further 1.5x faster and 18% larger.

| | 2,000 episodes | 8,000 episodes |
|---|---|---|
| per-tic, level 6 | 3.6 to 5.4 h, 520 GB | 14.4 to 21.6 h, 2.08 TB |
| decision-only, level 6 | 1.8 to 2.7 h, 128 GB | 7.2 to 10.8 h, 512 GB |
| decision-only, level 1 | 1.2 to 1.8 h, 150 GB | 4.8 to 7.2 h, 600 GB |

## Is the fast mode the same corpus?

No, and it does not need to be. Two seeded episodes per map pair were recorded both ways (same corpus id, same episode ids, maps 3 and 10, 60 game-seconds) and compared decision by decision, with the PNGs decoded and the arrays compared.

| | 8 bots | no bots (control) |
|---|---|---|
| `tic` column agrees at every compared decision | 495/495, 497/497 | 516/516, 515/515 |
| first decision that differs | 18 (tic 72), 5 (tic 20) | 35 (tic 140), 11 (tic 44) |
| frames bit-identical at decision tics | 4.0%, 14.3% | 9.3%, 2.3% |
| verified transitions (per-tic vs decision) | 1014 vs 984 | 1028 vs 1027 |
| chains | 5 vs 8 | 4 vs 4 |
| frames to encode | 1019 vs 992 | 1032 vs 1031 |
| canonical control table | differs | identical |

The episodes diverge within a few game-seconds, and the no-bots control diverges too, so the cause is not a death landing inside a skip: ViZDoom does not step identically under `make_action(buttons, 4)` and four `make_action(buttons, 1)` calls. Each mode is nevertheless reproducible on its own terms. The same episode recorded twice in the same mode is byte-identical (md5 `aa8af139...` per-tic, `8fd6fa98...` decision-only), so the stepping mode is simply a third input to the trajectory alongside the corpus id and the episode id. **The fast mode is a different but equally valid draw from the same process**: same agent, same weights, same maps, same bots, same seeds, same action distribution.

What does carry over is the yield, which is what training consumes: 99.7% of per-tic decision frames become verified transitions against 99.2% in decision mode (97.9% against 99.6% in the control). The canonical control table differs only by how anti-stuck overrides are weighted: an override runs 40 tics, so per tic it votes 40 times for one decision and decision mode votes once. Decision mode is the less biased vote, and `paper/fixtures/test_decision_only.py` pins both behaviours.

**Not yet done.** Latent encoding (decision frames only, as for the original corpus), the train/validation split file, and the md5 manifest.
