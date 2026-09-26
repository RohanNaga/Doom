# DoomDiT: researcher's dossier

A reading companion and experiment ledger for Rohan Nagabhirava. It teaches the project from the documents: what is learned, why each evaluation is defined the way it is, what every completed experiment establishes and does not, the decisions and their reasons, and where our evidence sits beside GameNGen, DIAMOND, MultiGen, PlayGen, Oasis and the open reproductions.

**Cutoff:** Sep 26 2026, 12:00 EDT: main at `2cdff5a`, plus the PixArt 150k and 155k periodic reads that main relayed for the 12:00 EDT log entry. Every number is stated once, in its final corrected form. Where two sources disagree, the later one is used and the disagreement is named once.

**Source tags.** `[RC 09-26 11:30]` is the entry of that date and time (EDT) in section 7 of `RESEARCH_CONTEXT.md`; `[RC §0]` is its section 0. `[rates]` is `results/sd35_stability/collapse_rates_50k_130k.md`. `[stats]` is `results/distance_study/figure_unet_h1/stats.json`, and `[dist-sd1]`, `[dist-pix]`, `[dist-sd35]` are `results/distance_study/distances_{sd1,pixels,sd35}.json`. `[memo]` is `.claude/analyses/distance-study-design-2026-09-24.md` and `[lit]` is `.claude/analyses/distance-study-literature-2026-09-25.md`. `[G]`, `[M]`, `[D]`, `[P]`, `[DF]`, `[SF]`, `[O]` are GameNGen (arXiv 2408.14837v2), MultiGen (2603.06679v2), DIAMOND (2405.12399v2), PlayGen (2412.00887v1), Diffusion Forcing (2407.01392v4), Self Forcing (2506.08009v2) and Open Oasis. "Derived" marks arithmetic done for this dossier from cited numbers. `$D` is `/sata2/data/rnagabhi/doom` on Spiderman.

---

# 0. State of the project, Sep 26

*A reader can stop after this section.*

**The paper.** CoRL 2026 PhysWM workshop, 4 pages, deadline Sep 30 AoE. Authors Rohan Nagabhirava and Keerthana Chirumamilla. Draft to Changliu Sunday Sep 27 evening, with SD 3.5 numbers marked provisional; final tables Sep 28 to 29. [RC §0; RC 09-25 13:25]

## 0.1 The spine

1. **A matched three-backbone comparison.** SD 1.4 U-Net (4-channel latents), SD 3.5 Medium (a DiT, 16-channel latents) and PixArt-alpha (a DiT, 4-channel latents) train as Doom world models under one recipe on one public dataset: every tic (35 Hz), 32 tics of context, the executed-control history, v-prediction, 200k updates of batch 32. The rows land within about 0.7 dB of each other at one tic, so the backbone table is the main table but not the whole contribution. The findings came from how we evaluate.
2. **Persistence-referenced evaluation.** Every score sits beside copy-last-frame persistence on the same windows. At 35 Hz consecutive frames are nearly identical, so persistence is a strong baseline and a raw PSNR means little without it. All three rows beat it on held-out episodes of the training maps. On maps the model never trained on, the U-Net loses to it at one tic on 17 of 26 maps. None of the 2024 to 2026 game and driving world models we checked reports such a reference; 2014 to 2019 video prediction routinely did. [lit Q4]
3. **The map-distance generalization result.** Each evaluation map's footage becomes a motion-weighted cloud of frame latents, and its distance is the sliced Wasserstein distance to the nearest training map. Over 30 maps the U-Net's gain over persistence falls with that distance: partial Spearman −0.73, 95% CI [−0.84, −0.32] at one tic, −0.80 [−0.88, −0.46] at four tics, permutation p = 0.0001, the same sign in all 30 leave-one-map-out refits. It replicates on pixel distances (−0.66) and on SD 3.5 latent distances (−0.73). It lives between maps, not between episodes of one map (ρ = +0.09, p = 0.22). The test and the distances were frozen before any map was scored. The pre-declared gate still prints "does not support", because two distance sanity checks fail; that ruling is open. [stats; section 7]
4. **The closed-loop stability finding.** Under autoregression, SD 3.5's live weights intermittently fall into absorbing states: one latent channel's mean is captured at a fixed value while the latent's overall scale stays normal, and the frames go blank. The fp32 EMA weights fall in less often. Over 50k to 130k, a frame drops under 10 dB in 16.7 percent of live rollouts (48 of 288) and 7.1 percent of EMA rollouts (25 of 352); the rollout ends under 12 dB in 12.8 against 2.6 percent, and ends with channel 13 captured in 8.3 against 1.7 percent. [rates] The live-versus-EMA gap is concentrated at 50k and 70k; on matched windows late in training the two weight sets fail at the same rate (section 6.4). The paper states rates per weight set and per training window, not "the EMA never fails".
5. **The directional check.** On real turning windows, swapping TURN_LEFT and TURN_RIGHT in the newest control reverses the predicted camera motion in 87 percent of windows for the U-Net at 200k and 84 percent for the SD 3.5 EMA (100k and 130k), with about the true magnitude. Neither model learned plain persistence. [RC 09-25 02:10; RC 09-26 11:30]

## 0.2 Current numbers

Validation episodes of the four training maps (ids 6000:6100), 512 teacher-forced windows, 10-step DDIM, stock decoders, raw target frames. "Gain" is model PSNR minus persistence PSNR on the same windows. Rollouts are 16 of 256 tics from the same episodes; copy-seed holds the last real context frame for the whole rollout.

| Row, weights | 1 tic PSNR / LPIPS | Gain | 4 tics PSNR / LPIPS | Gain | 256-tic rollout PSNR | Minus copy-seed |
|---|---|---:|---|---:|---|---:|
| Persistence (copy last frame) | 21.57 / 0.203 | 0 | 19.21 / 0.354 | 0 | copy-seed 17.52 | 0 |
| U-Net 200k EMA (final) | 22.49 / 0.177 | +0.92 | 21.33 / 0.224 | +2.11 | 17.81 | +0.29 |
| U-Net 200k live | 22.29 / 0.186 | +0.72 | 20.97 / 0.244 | +1.76 | 15.87 | −1.65 |
| SD 3.5 130k EMA | 23.21 / 0.135 | +1.64 | 21.52 / 0.199 | +2.31 | 17.70; seed-1 windows 17.25 | +0.18; −0.21 |
| SD 3.5 130k live | not recorded (120k: 22.95 / 0.146) | | not recorded (120k: 21.24 / 0.222) | | 17.86 | +0.34 |
| PixArt 155k EMA | 22.41 / 0.184 | +0.84 | 21.24 / 0.235 | +2.02 | no rollout read yet | |
| PixArt 155k live | 22.30 / 0.189 | +0.73 | 20.94 / 0.250 | +1.73 | | |

Sources: U-Net [RC 09-24 18:40]; SD 3.5 [RC 09-26 10:50, 11:30, 05:40]; PixArt [RC 09-26 12:00, read from `results_spiderman/041-pixart-nexttic/eval_0155000/*/metrics.json`]. The seed-1 rollouts draw different validation windows, whose own copy-seed is 17.46. SD 3.5 passed the U-Net's final LPIPS at 40k and both final U-Net numbers at 50k (22.56 / 0.163). [RC 09-25 23:15] The SD 3.5 130k gains and the U-Net live gains are derived.

Two caveats travel with every row. SD 3.5 decodes through a better autoencoder (about 27.5 against 23.5 dB reconstruction on validation frames), so part of its lead is rendering, not dynamics. And none of these windows come from maps the models did not train on; section 7 holds that evidence.

## 0.3 Runs and ETAs

| Run | Card (Spiderman) | State at cutoff | 200k |
|---|---|---|---|
| `040-unet-nexttic` | GPU 1, since given to PixArt | Stopped at 200k on Sep 24 17:09 EDT; last validation loss 0.1509 | Done |
| `041-pixart-nexttic` | GPU 1 | 160k at 12:00; 1.0 to 1.36 updates/s | Tonight, Sep 26; final read follows (supersedes "early Sunday" of RC 09-25 23:15) |
| `042-sd35-nexttic` | GPU 2 | 133.4k at 11:30; 0.50 to 0.54 updates/s | Sep 27, about 21:30 to 23:00 EDT (21:30 from RC §0; 23:00 derived from 133.4k at 0.52 updates/s) |
| Steward 7 reads | GPU 3 | Every 5k: teacher-forced reads, two-seed EMA rollouts, probe_v2, directional; EMA seed-2 backfill on surviving snapshots; PixArt 150k rollout read running | |

Every run streams to W&B project `doomdit-nexttic`. `/sata2` had 265 GB free and `/home` 10 GB at 10:15. [RC 09-26 10:30, 11:30, 12:00; RC §0]

## 0.4 Open decisions (Rohan's)

1. **The verdict gate** of the distance study. Two of its five distance checks fail (section 7.3). Main's position: check (iv) is an assumption about the maps, not a check of the measurement, and check (ii) is a subset tolerance stricter than the precision the test relies on. The gate was not changed after the data. Astra reviews the question today at 13:28. [RC 09-25 11:30; `.claude/analyses/astra-brief-2026-09-26.md`]
2. **Which SD 3.5 weights to release.** The matched table stays at 200k for every row. Choosing the released SD 3.5 checkpoint by validation-rollout stability instead is legitimate, because the rollouts use validation windows, not test. [RC 09-26 09:00]
3. **The post-training experiment** on GPU 1 after PixArt: four 3k-step arms against the absorbing states (section 6.6). Astra reviews the design at 13:28. [RC 09-26 10:30]
4. **Deletions** on `/sata2`: `hf_release_arnold` (202 GB) and `raw_arnold` (202 GB), both verified on the Hub, and `latents_arnold_sd35` (41 GB), 445 GB in all. An armed ladder deletes the first two if free space reaches 150 GB. [RC 09-25 13:15, 22:15]
5. **GPU 2 after SD 3.5 stops.** Proposal: a 6,000-episode U-Net row. It needs a 250 GB re-download of the 4-channel 2000:6000 set, a hard-linked 6,000-episode directory and a gate run. [RC §0; RC 09-24 22:50]
6. **Smaller:** the sealed-corpora rule for the distance study's fixed-checkpoint reads (section 2.5); a word to hengjinz about GPU 1; `ServerAliveInterval 15` and `ServerAliveCountMax 2` in the multiplexed SSH block (section 10.3).

A new analytical point for the stability claim came out of writing this dossier (section 6.4). The late rise of the pooled EMA event rate coincides with a harder window set joining the reads at 105k. On the standing seed-0 windows the EMA rate is flat: 3.1 percent at 55k to 100k, 4.2 percent at 105k to 130k. A trend test needs the same windows at early and late checkpoints.

