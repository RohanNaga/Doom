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

---

# 1. The task, and why PSNR needs a persistence reference

## 1.1 What is being learned

DoomDiT learns the next frame of the game given recent frames and the controls the engine executed. Everything happens in an autoencoder's latent space: an encoder E maps each 320×240 frame (padded to 320×256) to a latent z, the model predicts the next latent, and a decoder renders it for display and scoring.

For target row r the contract is p(z_r | z_{r−32:r}, u_{r−32:r}). Here u_t is the executed control that takes the engine from frame t toward frame t+1, so the last conditioning control u_{r−1} is the one that produces the target. The control u_r is future information and never enters. Our recorder stores each frame with its *outgoing* control before stepping the engine; other datasets store the *incoming* control, so copying their array offsets would silently change the causal task. [`record_arnold.py:255–273`; `doom_data.py:725–746`]

Two ways of running the model answer different questions:

- **Teacher forcing** feeds real past frames as context and scores one prediction. It measures one-step fidelity. "h1" and "h4" below mean predicting one or four tics past the last real frame.
- **Autoregressive rollout** feeds each prediction back as context, with the recorded controls, for 256 tics (about 7.3 game-seconds). It measures whether errors compound. The engine supplies neither frames nor state during a rollout.

The model is a visual simulator with finite memory. It does not run Doom's geometry, collision, ammunition or damage rules; it infers what it needs from pixels and controls. So several failure modes are distinct and no single scalar captures them: a plausible image can ignore the control, a smooth rollout can freeze the camera, and a correct movement can score badly because an enemy did something else. That is why the evaluation is a battery (section 4): teacher-forced fidelity, rollouts against copy-seed, a directional control test, and per-map scores tied to distance from the training footage.

## 1.2 PSNR, and why it needs a task definition

PSNR is 10·log10(MAX²/MSE) for one image pair; we average it over images. It depends on the reference image, the temporal gap, the crop, the sampler and the aggregation, so a PSNR without those is not a result. [`eval_tf.py:34–43,314–320`]

**Persistence** predicts the next frame by copying the last real frame. Its PSNR measures how much the footage changes over the gap, which is a property of the data, not of any model. It is a baseline, not a lower bound: a poor model scores below it. It moves a lot with the gap. On a 20,000-pair sample of our arenas 2 to 5 it scores 21.78 dB at one tic, 20.35 at two, 19.02 at four and 18.19 at eight; arena 5 alone runs 20.34 at one tic and arena 4 23.05. [`results/night_2026-09-19/q2-verify-dense/stats_arenas.json`] The open GameNGen-reproduction footage is easier still: 23.79 dB at one tic against 21.59 on our seen footage, sampled the same way. [`results/night_2026-09-19/q4-repro-footage/TABLE.md:3–22`] So PSNRs from different gaps or different footage are scores on different tasks.

**Gain over persistence** is model PSNR minus persistence PSNR on the same windows. Per window it equals 10·log10(MSE_persistence / MSE_model), so the footage's motion level cancels to first order and what remains is what the model adds over copying. It has the algebraic form of Genie's ΔtPSNR, which differences two PSNRs against a counterfactual predictor. [lit Q5]

**Three references appear in our metrics files, and the suffix names the target, not the prediction:**

| Field | Prediction | Target | Use |
|---|---|---|---|
| `persist_psnr_raw` | raw last frame | raw next frame | the decoder-free persistence reference; the only one used for gains |
| `copy_psnr_raw` | decoded last latent, D(E(I_{t−1})) | raw next frame | carries the autoencoder's reconstruction error; historical |
| reconstruction | decoded target latent, D(E(I_t)) | raw target | the "VAE ceiling"; a diagnostic of rendering error, not a bound on what a predictor can score |

The decoded copy scores lower than raw persistence, by 1.36 dB on average over the 30 evaluation maps and by up to 4.04 dB on static campaign maps (derived from the frozen per-map metrics). Using it as the reference inflates gains, unevenly across maps. The distance study's first look made exactly that mistake (section 7.6). [`eval_tf.py:278–303`]

**Rollout references.** Copy-seed holds the last real context frame for every horizon; beating it means the rollout is closer to the truth than freezing the game. Copy-last uses the real frame one tic before each target, so it reads the truth and is not an open-loop competitor; it only shows local difficulty. [review M4 in `docs/REVIEW_2026-09-22.md`]

## 1.3 What "stronger than GameNGen" can and cannot mean

GameNGen reports 29.43 dB and 0.249 LPIPS, teacher-forced, on 2,048 held-out trajectory samples from five levels, sampled with four DDIM steps. [G §§3.3, 5.1] Its footage, weights and training code are not public, so we cannot score it on our windows or ours on its windows. Matching its step count leaves the data, gap, autoencoder, decoder and sampling of windows different. Subtracting each dataset's persistence improves interpretation but does not create a common benchmark. [G §5.1; RC 09-22 09:15]

The class-project headline, 26.04 dB and 0.153 LPIPS, was measured on training segments, and the released checkpoint saw all 500 episodes, so it has no honest held-out number. [RC 09-02] The rebuilt project therefore claims what it can control: an open, reproducible recipe and dataset; a matched comparison on it; and an evaluation that exposes persistence and decoder effects that a headline PSNR hides.

---

# 2. The data

## 2.1 Recording: Arnold, every tic

The footage comes from ViZDoom with Freedoom assets, played by Arnold, a public pretrained deathmatch agent (Lample and Chaplot), against bots, with the HUD and crosshair visible. `record_arnold.py` stores every engine tic losslessly: the frame, then the control it is about to send, then the engine steps. Episodes run 150 game-seconds nominally. The engine does not render the respawn interval, so each death removes about 39 stored tics; a fit gives a baseline near 5,286 rows per episode. The minimum-length filter was lowered from 4,800 to 4,600 rows so that it would not reject episodes merely for having more deaths. [`release/DENSE_CORPUS.md:85–94`]

**Why Arnold.** A released agent and an inspectable engine interface make the pipeline reproducible end to end. The cost is a different behaviour distribution from GameNGen, whose PPO agent was recorded throughout its learning, weak and strong phases included. Arnold is one fixed policy plus scripted anti-stuck overrides. Map diversity does not substitute for diversity of competence. [G §3.1; RC 09-09]

**Why every tic.** The first corrected rows (the "stride-four" rows, section 5.5) predicted verified decision frames four tics apart. Two findings moved the project to every tic on Sep 20. Persistence depends strongly on the gap (section 1.2), so stride-four PSNRs answer a different, harder question than any consecutive-frame model. And a global four-tic grid is unsafe for Arnold: deaths and anti-stuck behaviour shift its action phase, so only 11,427 of 30,841 action-run starts in a sample fell on the global phase, against 44,616 of 44,616 in the reproduction footage. [Q4 `TABLE.md:47–71`; RC 09-20 21:45] A next-tic window only needs consecutive tics, an unchanged death counter and an unchanged map; it keeps held actions and override transitions instead of rejecting them. [`doom_data.py:465–514,591–746`]

## 2.2 The corpus and the split

The dense corpus uses Arnold's own published map split: training arenas 2 to 5 and test arenas 6 to 8. An earlier idea of training on arenas 3, 10, 12 and 13, chosen from favourable per-map scores, was rejected because it would make the dataset depend on the outcome. [`release/DENSE_CORPUS.md:9–17`]

| Pool | Content |
|---|---|
| `arenas` | 8,000 episodes, 2,000 per arena 2 to 5; 40,285,059 tics, about 319.7 recorded hours at 35 Hz |
| `arenas_678` | 3,000 episodes, 1,000 per arena 6 to 8; about 14.8M tics; never trained on |

The split is by episode, in half-open id ranges, and frozen in `release/dense_split.json`:

| Purpose | Ids | Episodes |
|---|---|---|
| Training pool | `arenas` 0:6000 | 6,000 (1,500 per arena) |
| Validation pool | `arenas` 6000:7000 | 1,000 |
| Test pool | `arenas` 7000:8000 | 1,000 |
| **Trained on by every row** | `arenas` 0:2000 | 2,000 (500 per arena); 10,070,779 rows |
| **Validation subset used for all reads** | `arenas` 6000:6100 | 100 (25 per arena); 503,864 rows |
| Sealed test subset | `arenas` 7000:7100 | 100; 503,620 rows; never scored |
| Unseen scoring subset | `arenas_678` 60:120 | 60 (20 per arena) |

Of the U-Net's 10,006,779 candidate training windows, 9,661,754 are valid; the 3.45 percent excluded cross a life boundary. [RC 09-23 06:30] At batch 32 one pass is about 300k updates, so 200k updates see about two thirds of one pass (derived). The 2000:6000 episodes were encoded later on Superman in both latent spaces, in a separate tree so the certified training directories stayed unchanged, and uploaded to their own Hub folders. The whole dataset is public on Hugging Face. [RC 09-24 22:50; RC §0]

**Why the unseen subset moved to ids 60:120.** Arnold's weapon-select requests execute only in each recorder worker's first episode (section 2.3). `arenas_678` was started three times, so its ids 0:60 held 51 worker-first episodes, against none in validation and test and 32 (1.6 percent) in the training prefix. Scoring 0:60 would have mixed map transfer with a control regime the models barely saw. Ids 60:120 hold none. The replacement was declared on Sep 22, before any model was scored on either set. [`docs/REVIEW_2026-09-22.md` H1]

## 2.3 Requested versus executed controls: the mechanism and the repair

The raw `buttons` field is the control list Arnold *requested*. It is not what the engine executed, for two reasons.

**The mechanism.** Arnold's action builder exposes nine controls: forward, backward, turn left, turn right, move left, move right, attack, speed, crouch. Its `add_buttons` appends ten weapon-selection names to the same shared list *at every game start*, and the recorder starts a game per episode, so the requested list grows by ten each episode in a worker. ViZDoom deduplicates buttons at registration, so the engine's vector stays 19 wide; it reads the first 19 entries of a longer request, zero-fills a shorter one, and ignores the tail. [Arnold `actions.py:197–199,298`, `game.py:485`; ViZDoom 1.2.4 `ViZDoomGame.cpp:151–158,367–372`] The executed control is therefore

```python
executed = requested[:19].ljust(19, "0")
```

A weapon request sits at index 9 + 10k + j, where j is the weapon slot and k is the number of *previous* game starts in that worker. Only k = 0, the worker's first episode, places it inside the executed range. In later episodes Arnold's weapon requests are silently ignored, while movement and firing still execute and pickups can still change weapons. The dataset records what the engine did under the controls it consumed; it is not a record of Arnold with the bug fixed. [`transitions.py:32,41–89`; `release/DATASET_CARD.md:98–108`]

**The full-corpus count** (8,000 episodes, 40,285,059 rows): raw request widths reach 2,507 characters; 11.0 percent of rows are wider than 19; 3,286 rows carry an executed weapon switch, all in the first episode of each of the 32 workers; 4,424,593 switch requests went unexecuted; the maximum inferred prior-start count is 249, which is 8,000 episodes over 32 workers, exactly as the mechanism predicts. [RC 09-22 13:00] Requested and executed controls agree on 89 percent of tics. Almost all of the rest are dropped weapon requests, and only 0.15 percent are true anti-stuck overrides. [RC 09-24 18:40]

**The repair.** Encoded sidecars store the fixed-width executed string plus `buttons_raw_len` and `switch_requested_index`; the raw parquet stays untouched so the original request is preserved. The repair checks row counts, tics and normalised controls against the raw data, writes atomically and does not re-encode images. The loader refuses wide old sidecars rather than loading a huge control matrix. Two wrong fixes were rejected: remapping an overflow bit into an executed slot (claims a switch the engine ignored) and keeping only the nine base controls (drops real executed switches). [`transitions.py:82–149`; `doom_data.py:562–588`; RC 09-22 11:30]

**Consequence for the experiment plan.** A planned ablation conditioning on requested controls was dropped: with 89 percent identity and 0.15 percent true overrides it would most likely be null. [RC 09-24 18:40]

## 2.4 The evaluation corpus and how its maps were fixed

The distance study (section 7) scores 30 maps, 342 episodes in all:

| Group | Maps | Episodes per map | Source |
|---|---|---:|---|
| Training arenas, held-out episodes | 2, 3, 4, 5 | 25 | validation ids 6000:6100 |
| Same-WAD test arenas | 6, 7, 8 | 20 | `arenas_678` 60:120 |
| Other arenas | 1, 9 to 15 (`seen`) | 4 | seeded corpus |
| Other arenas | 16, 17 (`unseen`) | 10 | seeded corpus |
| Campaign maps (another WAD) | 18 to 20, 22 to 26, 28 to 32 (`unseen2`) | 10 | seeded corpus |

That is 17 arena maps and 13 campaign maps. The seeded evaluation corpus holds 210 episodes (60 `seen` on maps 1 to 15, 20 `unseen`, 130 `unseen2`), all encoded per tic in both latent spaces with the pinned encoder. Its copies of maps 2 to 8 replicate the distance measurement rather than adding points. "Seen" is a historical name: those arenas trained the April model, not any next-tic row. [RC §0; RC 09-25 09:40; dist-sd1 `maps`]

**How map selection was fixed.** The arena split is Arnold's. The campaign maps were curated on Sep 16 for a deathmatch-style experiment: monsters and some pickups removed, maps where the agent got stuck excluded. That is a disclosed, curated distribution, not a random sample of Doom maps, and it was fixed before any next-tic result existed. [RC 09-16 15:40, 17:10]

## 2.5 The seal

Test data is sealed so that no choice is made after looking at it. The evaluator seals the test corpus at first access, pins the whole scoring configuration at selection time, and treats `arenas_678`, `seen`, `unseen` and `unseen2` as sealed reporting corpora. [RC 09-23 00:50; `after_nexttic.sh`] The distance scorer read those corpora once, with the U-Net's final 200k EMA weights, so no checkpoint selection could happen; it refuses the test corpus and wrote nothing into the seal registry `$R/sealed`. Whether such fixed-checkpoint reads count as the corpus's single scoring is Rohan's open ruling. [RC 09-25 01:25, 09:40]

The historical decoders were fine-tuned on frames that included held-out maps, so all next-tic reads use the stock decoders, which have seen no Doom map. [RC 09-21 14:30]

---

# 3. The recipe

All three rows share one recipe and differ only in backbone and latent space. [RC 09-21 01:30; RC 09-23 06:30; RC 09-24 20:35]

| | SD 1.4 U-Net | SD 3.5 Medium | PixArt-alpha 512 |
|---|---|---|---|
| Run | `040-unet-nexttic` | `042-sd35-nexttic` | `041-pixart-nexttic` |
| Architecture | convolutional U-Net | MMDiT (joint-attention transformer) | DiT with cross-attention |
| Latent space | SD 1.x, 4 channels | SD 3.5, 16 channels | SD 1.x, 4 channels |
| Parameters | 860.5M | 2,271.7M | 628M |
| Control tokens enter via | cross-attention, width 768 | joint attention, width 4,096, plus a 2,048-d pooled slot from the newest control | cross-attention tokens (`--action-inject token`), width 4,096 |
| Memory at batch 32 | 23.3 GB, no checkpointing | 36.8 GB, gradient checkpointing | 26.8 GB, no checkpointing |
| Throughput alone | 1.78 to 1.83 updates/s | 0.50 to 0.55 updates/s | about 1.4 updates/s |
| Launched | Sep 23 06:28 EDT | Sep 23 09:04 EDT | Sep 24 20:32 EDT |

Shared settings: public pretrained weights (never our earlier Doom checkpoints, which would add in-domain exposure and the old stride); training ids 0:2000; fused AdamW at 5e-5 after 2,000 warmup updates, weight decay 0, clip norm 1.0; global batch 32 on one card with no gradient accumulation; bf16 autocast with fp32 parameters and EMA; seed 0; action dropout 0; validation loss every 1,000 updates on 1,024 fixed windows, overall and by timestep quartile; recovery checkpoints every 5k and snapshots every 10k. The U-Net fills only half of its 48 GB A6000, but a larger micro-batch would change the recipe's global batch, so the fill-the-card rule does not apply. SD 3.5 is compute-bound (GPU at 100 percent, CPU 93 percent idle), and gradient checkpointing costs it about a third of its speed. [RC 09-23 06:30; RC 09-24 21:05]

## 3.1 Channel-stacked latents

The 32 context latents are concatenated with the noisy target latent along the channel axis, so time is not a separate token axis; the backbone sees one tall image-like tensor. The pretrained input projection is inflated with zero-initialised weights for the new context channels, which preserves the pretrained function on the target at initialisation. The new control embeddings still change downstream activations, so the network does not start as exactly the pretrained model. [`backbones.py:241–259,344–388`] This is GameNGen's visual-context design, with 32 frames instead of 64; GameNGen's own ablation gains only 0.05 dB from 32 to 64 frames. [G Table 2] At 35 Hz, 32 tics span 0.914 s nominally (0.886 s between the oldest and newest frame), against 3.66 s for the stride-four rows, so long-range memory is weaker by design.

## 3.2 The two latent spaces

The SD 1.x autoencoder (`sd-vae-ft-mse`, 8× downsampling) maps a padded 320×256 frame to 4×32×40 with scale 0.18215. The SD 3.5 autoencoder gives 16×32×40 with shift 0.0609 and scale 1.5305. Padding rows are cropped before every decode so only the 240 real rows are scored. [`encode_parquet.py:155–189`; `backbones.py:34–74`] The 16-channel space reconstructs our validation frames at about 27.5 dB against 23.5 dB for the 4-channel space (stock decoders; section 3.7's alignment gate). That gap belongs to the SD 3.5 row as a system: its lead mixes a better representation with its backbone. A Flux autoencoder, also 16 channels, is not interchangeable, because its scale and shift differ.

## 3.3 Controls as tokens

Each of the 32 executed 19-bit controls passes through a shared MLP, gets a learned position embedding, and enters the backbone as a token. This adopts GameNGen's history-token idea without claiming its undisclosed vocabulary or encoding. [`backbones.py:153–217,771–790`] The ImageNet DiT was not used as a next-tic row partly because its conditioning path averages the control embeddings before adaLN (adaptive layer normalisation, where conditioning sets per-channel scale and shift); an average of embeddings with additive positions cannot see the order of the controls. [`backbones.py:153–181,290–307`]

## 3.4 v-prediction and the schedule

The network predicts the *velocity* v = √ᾱ_t·ε − √(1−ᾱ_t)·x₀, a mix of the noise ε and the clean latent x₀ whose weighting changes with the noise level ᾱ_t. Velocity targets keep the regression well-scaled at both ends of the noise range, which is why GameNGen switched to it. The loss is an unweighted MSE with t drawn uniformly. [`diffusion_v.py:38–102`; G §3.2]

The schedule is 1,000 linear betas from 1e-4 to 0.02, kept from the earlier rows for continuity. It is not either backbone's native schedule. SD 1.4 uses a "scaled-linear" schedule with terminal ᾱ about 4.7e-3 against our 4.0e-5, so 273 of our timesteps are noisier than anything SD 1.4 saw in pretraining. SD 3.5 was pretrained with rectified flow, a different interpolation and target altogether. The paper discloses the mismatch; no schedule ablation was run, and the schedule is hard-coded at eight call sites, so changing it would be a new training experiment. [RC 09-21 14:30; `.claude/analyses/astra-prelaunch-audit-2026-09-21.md`]

## 3.5 Noise augmentation and its bucket-zero subtlety

Noise augmentation corrupts the *context* during training so that the model's own imperfect predictions look less foreign when they are fed back during a rollout. Our form is c̃ = √(1−q)·c + √q·ε, with one level q per example drawn uniformly below 0.7 and shared across its 32 context frames. The level is quantised into ten buckets of width 0.07, and the bucket id is embedded so the model knows how noisy its context is. At q = 0.7 the signal and noise coefficients are 0.548 and 0.837. GameNGen states the 0.7 maximum and ten buckets but not the formula; the Stiegler reproduction adds noise instead of blending. [`diffusion_v.py:135–151`; G §3.2.1]

**The subtlety.** Inference uses clean context with bucket 0. But bucket 0 covers q in [0, 0.07), and a continuous uniform draw hits exactly q = 0 with probability zero, so clean context is a boundary case the model never trained on exactly. A proposal to give exactly-clean context 10 percent of training mass was withdrawn: it would change the training distribution and the bucket semantics relative to the earlier rows, and it is not a disclosed GameNGen detail. [RC 09-21 14:30]

**Why augmentation stayed.** GameNGen's no-augmentation model diverges within 10 to 20 generated frames. [G §5.2.2, Figure 7] Our stride-four "noaug" cell instead improved teacher-forced fidelity and action accuracy at a cost in FVD (section 5.5). Both can be true; the recipe kept augmentation because the next-tic rows run much longer rollouts. Section 6 shows its limit: it does not prevent the absorbing states, whose errors are structured shifts of a channel's mean rather than the isotropic noise it trains on.

## 3.6 fp32 EMA

An EMA (exponential moving average) keeps a second copy of the weights, θ_EMA ← d·θ_EMA + (1−d)·θ, which averages the live weights over recent training. The trainer applies decay 0.9999 per update in steps of eight (d = 0.9999⁸ every eighth update), a time constant of about 10,000 updates. The arithmetic is fp32 because a bf16 update with weight 1e-4 underflows and freezes the average, which was the class project's bug. [`train_wm.py:419–422,819–820`]

Early in training the EMA still contains the public start weights: a share of 0.9999^s, about 0.61 at 5k, 0.37 at 10k and 0.14 at 20k. So the EMA scores badly at first (U-Net EMA 11.76 dB at 5k against live 21.15) and passes the live weights once the start has washed out, by 40k for the U-Net. [RC 09-23 11:30] Every read scores live and EMA weights from the same saved file. Reported samples in the paper use the EMA.

## 3.7 Launch gates and the certificate

Nothing trains until `scripts/cluster/gates.sh` passes on the machine and checkout that will run it. The gates, in order: sidecar controls against the raw recordings; training inventory; latent alignment; emitted training windows; yaw alignment of controls against camera motion on validation; a 20-update memory fit; a 300-step smoke with checkpoints; conditioning probes; a 10-update resume that restores optimizer, scheduler, EMA and random state; and a readback of the smoke snapshot through the evaluator. Passing prints `GATES_GO` and a `GATES_LAUNCH` line; the certificate `$D/GATES_CERT.json` records the resolved command, commit, data fingerprints and encoder records per backbone. The launcher refuses any start or resume that differs from its certified entry. [RC 09-23 00:50 to 04:40; `.claude/analyses/launch-runbook-2026-09-23.md`]

**Latent alignment** is the gate that catches an off-by-one between frames and controls. It re-encodes stored frames and compares them with the stored latents (MAE at most 5e-3, p99 at most 2e-2), then decodes the stored latents against raw frames unshifted and shifted by ±1 and ±4 rows; the unshifted score must win by at least 2 dB. The 2 dB floor was set after a correctly aligned, bit-identical 4-channel validation shard cleared only 2.91 dB: at one tic, a neighbouring frame is almost as good a match on slow footage, so the margin is bounded by per-tic motion. The MAE and p99 tolerances test misalignment independently. SD 3.5's latents were encoded on Superman's A4000s and re-encoded by the gate on Spiderman's A6000s; the cross-host differences were rounding-sized (largest MAE 19 percent of tolerance). All sampled shards passed, with margins 2.91 to 4.52 dB in the 4-channel space and 4.85 to 6.85 dB in the 16-channel space. These are sampled checks, not a proof for every row. [`check_latent_alignment.py:22–73`; `$D/GATES.txt`; RC 09-23 04:05, 04:40]

**How the gates got here.** Four review rounds on Sep 22 to 23 closed 21 defects: the independent review's seven findings, then Astra's eight, three and three reproduced defects. Examples: the launcher accepted an SD 3.5-only certificate for a U-Net launch; a smoke at lr 5e-5 certified a launch at lr 0.1; two concurrent evaluators could both score the sealed test set. A separate launch blocker surfaced only because a 300-step pre-smoke ran on real data: one memory map per episode against Spiderman's soft limit of 1,024 open files, fixed by raising the limit before loading. [RC 09-23 00:20 to 04:40, 03:00; `doom_data.py:198`] The rows launched from a second checkout, `$D/repo_launch` at `35258f3`, because encoders were still running from `$D/repo`; PixArt later launched from `repo_launch2` at `f5386f1`. The gates certify the training contract, not model quality: the 300-step U-Net readback scored 18.38 dB against 21.92 persistence.

## 3.8 Native W&B

Every run streams training loss, validation loss (overall and by quartile), gradient norm, throughput, memory and every evaluation read to W&B project `doomdit-nexttic`. The trainer does this by default through `wandb_log.RunLogger`; only the gates' throwaway runs pass `--no-wandb`. `periodic_eval.py` gives the trainer its own reads every N steps on another card. [RC 09-23 19:45; RC 09-24 21:55] The U-Net and SD 3.5 rows are pinned to `35258f3`, which predates native logging, so a sidecar, `tools/wandb_tail.py`, replays their `log.jsonl` and steward files. PixArt, launched before the init fix below, also streams through a sidecar (`041-pixart-nexttic-tail`). The fix: `wandb.init` starts subprocesses on the calling thread while the trainer forks DataLoader workers, and a worker forked in that window held a pipe open and hung the init; `11540db` moves setup before the fork. [RC 09-24 21:55]

