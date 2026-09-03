# DoomDiT requirement sheet: CoRL 2026 workshop paper

> Draft 1, Sep 2, 2026.
> Primary deadline: CoRL 2026 PhysWM workshop, **Sep 30 AoE, 4 pages**. Secondary: "Do Robots Need World Models?" (date TBD), same material.
> Each requirement has an id, a priority (P0 must ship, P1 should, P2 stretch), and a verification line. A requirement is done only when its verification line has been run.

## 0. What the April report established, and what has to be redone

The April report (`doomdit.pdf`) compared DiT-XL/2 against an SD 1.4 U-Net on 500 VizDoom episodes at 160x120 and reported 25.81 dB / 0.15 LPIPS (DiT) vs 24.90 / 0.19 (U-Net) vs GameNGen's 29.43 / 0.249. Four things about that setup must change before the numbers can go in a paper:

1. **Source frames are lossy.** The Hugging Face dataset was written with JPEG quality 85 (`compress_image` in arnaudstiegler/gameNgen-repro). Every PSNR we report is against a compressed target.
2. **Resolution is a quarter of GameNGen's.** GameNGen trains and evaluates at 320x240 (padded to 320x256). PSNR at 160x120 is not comparable to PSNR at 320x240, so the 29.43 row in Table 1 was never a like-for-like number.
3. **One map.** The dataset is one custom deathmatch scenario (`deathmatch_simple.wad`, six buttons, 18 legal button combinations) with a PPO agent. GameNGen evaluates on five levels; MultiGen (Mar 2026) trains on 100 generated maps.
4. **The evaluation set was training data**, and the two baselines were not matched: the U-Net used GameNGen-style noise augmentation and 20k steps, the DiT used no augmentation and 87k steps on all 500 episodes. See `RESEARCH_CONTEXT.md` section 0.

Also worth knowing for every decision below: the dataset stores every engine tic (35 per second) with the agent's action held for four tics. A 4-frame context is therefore 0.11 s of game time, and the next-frame target is 29 ms away from the last context frame. That makes a copy-last-frame baseline strong and inflates PSNR. Section 1 fixes this with a frame stride.

## 1. Data

| id | P | requirement | verification |
|---|---|---|---|
| R1.1 | P0 | Regenerate the dataset ourselves with the `gameNgen-repro` ViZDoom pipeline (PPO agent `deathmatch_simple/best_model.zip` is in that repo). Record **lossless** frames (PNG or raw uint8), 320x240 RGB24, HUD on, crosshair off, weapon on, plus per-tic action id, health, ammo, player x/y/angle, map id, episode id, tic id. | `data/DATASET.md` lists counts per map; a 100-frame sample decodes bit-exact from storage. |
| R1.2 | P0 | **At least three maps.** `deathmatch_simple` plus two Freedoom2 maps loaded through `doom_map` (MAP01, MAP02 or similar), same button set. If the PPO agent does not traverse a new map, fall back to a scripted explorer (forward with random turns, attack on sight) and say so in the data card. | Coverage heatmap of player x/y per map covers at least 50% of walkable cells; each map has equal episode counts. |
| R1.3 | P0 | **Frame stride.** Store every tic, but the training and evaluation unit is one frame per agent decision (stride 4, 8.75 frames per second). Context and target are consecutive decision frames. Rationale: matches how the agent acts and makes the prediction task non-trivial. | `doom_data.py` exposes `stride`; copy-last PSNR on the val split is reported next to every model number. |
| R1.4 | P0 | **Size:** at least 5M raw tics (about 1.2M decision frames) across the three maps, so per-map data is not smaller than the April set. Generation is cheap (a 150 s episode is 5,250 tics). | Row counts in `DATASET.md`. |
| R1.5 | P0 | **Splits by episode**, 10% held out on every map, fixed seed, committed as `data/split.json`. No window from a held-out episode ever appears in training. | `doom_data.make_split` output committed; the training log prints the split hash. |
| R1.6 | P1 | One **held-out map** (train on three, test on a fourth) to report generalization across levels. | Separate row in the results table, labeled unseen map. |
| R1.7 | P1 | Data card: action histogram, health distribution, episode-length distribution, coverage maps, generation config, git hash. | `data/DATASET.md` committed with figures. |

## 2. Latents and the VAE

| id | P | requirement | verification |
|---|---|---|---|
| R2.1 | P0 | **Encode at 320x240, padded to 320x256**, with `stabilityai/sd-vae-ft-mse`, posterior mean, scale 0.18215. Latent is (4, 32, 40); at patch 2 that is 320 tokens (four times the April model). No resize step anywhere in the pipeline. | `encode_episodes.py --height 240 --width 320 --pad-to 256`; `encode_meta.json` records the settings; latent shape asserted. |
| R2.2 | P0 | **Decoder fine-tuning for the HUD** (GameNGen section 3.2.2; `finetune_autoencoder.py` in gameNgen-repro): freeze the encoder, train only the decoder with MSE against lossless target frames on the training split. The encoder stays frozen so latents and the world model are unaffected. | Report VAE reconstruction PSNR/LPIPS before and after on the val split, full frame and HUD crop (bottom 32 rows). Acceptance: HUD-crop PSNR up by at least 2 dB, full-frame not worse. |
| R2.3 | P0 | **VAE ceiling in every table.** Reconstruction PSNR/LPIPS of ground-truth frames through the (fine-tuned) VAE is the upper bound any latent model can reach; print it as a row. | `eval_metrics.py` `vae_psnr` / `vae_lpips` columns. |
| R2.4 | P1 | Storage plan: fp16 latents are 10 KB per frame at (4, 32, 40); 1.2M decision frames is 13 GB. Keep raw frames only for held-out episodes. | `du -sh data/` within the budget in section 5. |

## 3. Models

| id | P | requirement | verification |
|---|---|---|---|
| R3.1 | P0 | **Matched pair.** Same VAE, same latents, same 4-frame stride-4 context by channel concatenation, same single-action conditioning, same noise-augmentation setting, same optimizer, batch, steps, and seed count. Any difference between the two rows must be the backbone. | A one-page "matched settings" table in the paper; the two training configs diff only in the model field. |
| R3.2 | P0 | **Backbones:** DiT-XL/2 (673M) vs the SD 1.4 U-Net (860M), both warm-started, as the main row, because that is what both communities actually use. If time allows, a parameter-matched from-scratch pair as a second row. | Both configs committed under `train_options/`. |
| R3.3 | P0 | **Noise augmentation on the context** (GameNGen: noise level up to 0.7, 10 buckets) for **both** backbones, since the autoregressive metrics in section 4 are meaningless without it. | Flag in both trainers; ablation row with it off for the DiT (P1). |
| R3.4 | P0 | **Fit check at 320 tokens** on a 16 GB A4000: gradient checkpointing on, per-GPU batch 4 with accumulation to global 32, bf16, EMA in fp32. Run a 200-step smoke test and record steps/s before any long run is launched. | Smoke-test log committed; steps/s feeds the schedule in section 5. |
| R3.5 | P1 | Longer context (8 or 16 decision frames) for the DiT only, as an ablation toward GameNGen's 64. | One extra row, same steps. |
| R3.6 | P0 | State the sampler honestly: respaced ancestral DDPM at N steps, or DDIM; report N and wall-clock per frame. | `eval_metrics.py --sampler`. |

## 4. Evaluation

| id | P | requirement | verification |
|---|---|---|---|
| R4.1 | P0 | **Teacher-forced PSNR and LPIPS** on held-out episodes, at least 2,048 windows spread over all maps (GameNGen: 2,048 trajectories over 5 levels). Score against the **raw lossless frame**, not the VAE-decoded latent. State the LPIPS backbone (AlexNet; also report VGG once). | `eval_metrics.py --subset val --num-windows 2048`, `metrics.json` committed. |
| R4.2 | P0 | **Baselines in the same table:** copy-last-frame, VAE ceiling, U-Net, DiT. A model that does not beat copy-last is not a result. | Rows present. |
| R4.3 | P0 | **Autoregressive metrics** (design in `.claude/analyses/autoregressive-eval-design.md`): drift curves (PSNR/LPIPS vs horizon to 64 decision frames), FVD at 16 and 32 frames with N >= 256 clips, action-following accuracy with a VizDoom inverse-dynamics model and its ceiling on real data. | `rollout_eval.py` outputs per model; figure in the paper. |
| R4.4 | P1 | **HUD metric.** PSNR/LPIPS on the HUD crop, and health-readout consistency: read the health digits from generated and real frames with a template matcher and report agreement over rollouts. | Column in `metrics.json`; matcher accuracy on real frames >= 99%. |
| R4.5 | P0 | **Quality bar**, written down before the runs: (a) must: DiT beats the matched U-Net on LPIPS and on drift at horizon 32, and both beat copy-last; (b) should: LPIPS <= 0.25 at 320x240 on held-out episodes, GameNGen's reported level; (c) stretch: PSNR >= 29 dB. GameNGen used 70M frames and 64-frame context, so (c) is not expected at 72 GPU-hours and the paper says so. | Table compared against this list in the final review. |
| R4.6 | P1 | Per-map breakdown of every metric, plus the unseen-map row from R1.6. | Table columns. |
| R4.7 | P0 | Every reported number has a seed, git hash, config, and checkpoint md5; weights go to a GitHub release. | `results/<run>/` and `WEIGHTS.md`. |
| R4.8 | P2 | Human study (GameNGen: 10 raters, 130 clips, 1.6 s and 3.2 s). Out of scope for 4 pages unless the numbers are close to GameNGen. | none |

## 5. Compute, storage, schedule

Reference point: DiT-XL/2 at 80 tokens ran at 1.62 steps/s (global batch 32, 4 A4000). At 320 tokens the attention and MLP cost is about 4x and checkpointing adds about 30%, so expect **0.3 steps/s**, or roughly 3.5 days for 90k steps on four GPUs. That is the number R3.4 must confirm or replace.

| id | P | requirement | verification |
|---|---|---|---|
| R5.1 | P0 | Superman budget: latents 13 GB, raw val frames 4 GB, two runs with three rolling checkpoints plus best 21 GB, decoder fine-tune 2 GB, eval outputs 2 GB: **about 45 GB**. Superman had 98 GB free on Sep 1. Needs the go-ahead from Rohan and, if others' usage grows, a move of the raw data to Spiderman. | `df -h` before and after; recorded in the run log. |
| R5.2 | P0 | GPU plan: at most 6 of 8 A4000s. DiT on four, U-Net on two, or sequential. | `nvidia-smi` snapshot at launch in the run log. |
| R5.3 | P0 | **Go/no-go dates** for Sep 30: data regenerated and encoded by **Sep 9**; decoder fine-tuned and fit check done by **Sep 11**; both final runs launched by **Sep 13**; runs finished and evaluated by **Sep 23**; numbers frozen **Sep 26**. If the fit check gives under 0.2 steps/s, cut steps to 50k for both models rather than slipping the launch. | Dates checked off in `RESEARCH_CONTEXT.md` section 7. |
| R5.4 | P1 | If 320x240 cannot finish in time, the fallback is 256x192 (latent 32x24 padded to 32x24, 192 tokens), stated as such. 160x120 is not an option for a paper that cites GameNGen numbers. | Decision logged. |

## 6. Paper deliverables

| id | P | requirement |
|---|---|---|
| R6.1 | P0 | Claim: under matched data, latents, conditioning, augmentation and compute, a DiT backbone beats a U-Net on perceptual and autoregressive metrics for an action-conditioned game world model, at 72 GPU-hours on 16 GB cards. Nothing about "first". |
| R6.2 | P0 | Table 1: copy-last, VAE ceiling, U-Net, DiT; PSNR, LPIPS, PSNR at horizon 16 and 32, FVD16/32, IDM accuracy; per-map rows in the appendix if the venue allows one. |
| R6.3 | P0 | Figure 1: drift curves. Figure 2: rollout strips with HUD visible, before and after decoder fine-tuning. |
| R6.4 | P0 | Related work must cite and position against GameNGen, DIAMOND, Oasis, MultiGen, Vid2World, Matrix-Game 2.0/3.0, Hunyuan-GameCraft-2, Genie 3, and the 2026 benchmarks (MIND, WBench, PlayWorld) as the reason autoregressive and action-following metrics are in the paper. |
| R6.5 | P0 | Limitations: not real time, fixed short context, single-action conditioning, three maps, no memory. |

## 7. Recent work that sets the bar (checked Sep 2, 2026)

Models on Doom or close cousins:

- **GameNGen** (Aug 2024): SD 1.4 U-Net, 320x240, 64-frame context, noise augmentation, decoder fine-tuning for the HUD, 70M frames from a PPO agent, 2,048 eval trajectories over 5 levels; 29.43 dB / 0.249 LPIPS; 20 FPS with 4 DDIM steps. Still the reference protocol.
- **MultiGen** (Mar 2026, arXiv 2603.06679): Doom via ViZDoom on 100 Obsidian-generated maps, U-Net diffusion with an explicit external memory of map geometry and player pose, 32-frame context, 20 FPS on one A100; over 10M frames. Reports SSIM 0.418 / PSNR 19.32 / LPIPS 0.453 on its level-conditioned protocol against GameNGen 0.405 / 18.77 / 0.471. Shows the field has moved to memory and multi-map; our multi-map data (R1.2) and unseen-map row (R1.6) are the minimum answer.
- **Vid2World** (2025, revised Mar 2026): turns a video diffusion model into an interactive world model, evaluated on 5.5M CS:GO frames against DIAMOND with FID and FVD. Confirms FVD is expected.
- **DIAMOND** (NeurIPS 2024): Atari and CS:GO, U-Net, RL inside the model.
- **Matrix-Game 2.0** (Aug 2025) and **3.0** (Apr 2026): open-source real-time streaming DiT world models, 1.6B distilled and 6.5B, 25 to 40 FPS on H100s, GameWorld Score. **Hunyuan-GameCraft-2** (Nov 2025), **Genie 3** (Aug 2025, 720p at 24 FPS): the large-scale end. We do not compete on scale; we cite them for the DiT-as-default trend and for the metrics.
- **Scalable Generative Game Engine** (Jan 2026, arXiv 2602.00608): the "resolution wall" argument; useful citation for why 320x240 at 16 GB is a real constraint.

Benchmarks and evaluation papers to borrow protocol from:

- **MIND** (Feb 2026, arXiv 2602.08025): memory consistency and action control, closed-loop revisits.
- **WBench** (May 2026, arXiv 2605.25874): multi-turn interaction adherence, consistency, physical compliance.
- **PlayWorld** (Aug 2026, arXiv 2608.13552): agent players over long-horizon objectives; translation and rotation pass rates for action controllability (from WorldMark). Our IDM action-accuracy metric is the small-scale cousin.
- **Towards Interactive Video World Modeling** survey (Jun 2026, arXiv 2606.01164) and **How Should World Models Be Evaluated for Embodied Decision-Making?** (Jun 2026, arXiv 2606.15032): cite for the shift from per-frame PSNR toward controllability and consistency.

What this means for us: per-frame PSNR/LPIPS alone will read as 2024. The paper needs drift curves, FVD, and an action-following number, plus a multi-map data story, to be taken seriously in 2026. Those are R4.3 and R1.2, and they are P0.

## 8. Open decisions

1. Confirm Sep 30 PhysWM as the primary target; "Do Robots Need World Models?" as the second.
2. Resolution: 320x240 (R2.1) with the fit check deciding step count, or the 256x192 fallback (R5.4).
3. Baseline pair: warm-started SD 1.4 U-Net vs warm-started DiT-XL/2 as the main row (R3.2), or parameter-matched from scratch.
4. Frame stride 4 (R1.3) versus keeping every tic.
5. Maps: which two Freedoom2 maps, and whether a fourth is held out (R1.6).
6. Disk: approve about 45 GB on Superman (R5.1).
