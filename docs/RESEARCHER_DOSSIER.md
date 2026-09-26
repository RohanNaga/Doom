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

