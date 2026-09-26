# DoomDiT: researcher's dossier for Rohan Nagabhirava

This document is a reading companion, experiment ledger, and record of the current research design. It distinguishes what a paper states, what our code implements, what our experiments measured, and what we have merely proposed. It is not a claim that the original GameNGen experiment has been reproduced.

**Source cutoff:** the local main-branch snapshot `5ac42c5a6a80ad538c25db75a4ced385ab9d7c54`, including the **2026-09-22 11:30 EDT** entry of `RESEARCH_CONTEXT.md`. Main changed during preparation. The final account includes the executed-control repair that arrived during that change. Analysis documents were read from `.claude/worktrees/vibrant-ritchie-6f6280/.claude/analyses/` when that worktree had a copy, as requested. No new training, server inspection, or measurement was performed for this dossier.

**Length:** approximately 22,106 whitespace-delimited words, including table notation; **37 estimated pages at 600 words per page**. Markdown has no fixed pagination. This is an editorial length estimate, not a measured PDF page count.

## How to use the sources

A citation at the end of a table caption applies to every numerical cell in that table unless a row supplies a different source. A dated `RC` citation means the corresponding entry in section 7 of `RESEARCH_CONTEXT.md`, not a claim that an underlying result artifact is present locally. `A/` means the preferred worktree's `.claude/analyses/` directory. File-and-line citations describe the snapshot above. They may move when the repository changes.

“Not stated” means the paper does not disclose the requested information. “Not locally recovered” means our records refer to an experiment but the reviewed checkout does not contain its numerical result. “Derived” identifies arithmetic from cited quantities. “Estimate” identifies an approximate reading or extrapolation. “Forecast” identifies a prediction made before measurement. These categories must remain separate in the manuscript.

| Key | Primary source and version consulted | Main use |
|---|---|---|
| G | [GameNGen, arXiv:2408.14837v2](https://arxiv.org/pdf/2408.14837v2), ICLR 2025 version | Original recipe, evaluation, ablations, appendices |
| M | [MultiGen, arXiv:2603.06679v2](https://arxiv.org/pdf/2603.06679v2) | Geometry memory, multiplayer, context ablation |
| D | [DIAMOND, arXiv:2405.12399v2](https://arxiv.org/pdf/2405.12399v2) | Pixel diffusion, EDM, Atari and CS:GO |
| P | [PlayGen, arXiv:2412.00887v1](https://arxiv.org/pdf/2412.00887v1) | Recurrent DiT, Diffusion Forcing, action-aware evaluation |
| DF | [Diffusion Forcing, arXiv:2407.01392v4](https://arxiv.org/pdf/2407.01392v4) | Independent noise levels across sequence tokens |
| SF | [Self Forcing, arXiv:2506.08009v2](https://arxiv.org/pdf/2506.08009v2) | Training on autoregressive self-generated histories |
| O | [Open Oasis](https://github.com/etched-ai/open-oasis/tree/f59deef2c019c212bd0c5a3a5b986a51f3701847) | Released inference model and limits of available training disclosure |
| RC | `RESEARCH_CONTEXT.md`, sections 0–7, especially the dated decision log | Project decisions and completed-result summaries |

The primary papers take precedence over the literature cards when they disagree. In particular, MultiGen's current paper reports PSNR and a context-length ablation. PlayGen explicitly uses Diffusion Forcing. GameNGen's inference frame rate does not specify its training stride. Our historical “copy-last” metric is not always raw-frame persistence. These distinctions affect the interpretation of the whole comparison. [M Tables 1, 3; P §3.3 and Figure 2; G §§3.3, 4.2 and Appendix A.6; `eval_tf.py:278–303`.]

**Addenda and reading order (added 2026-09-25).** Sections 3.12 and 4.10 (Sep 23) and sections 3.13, 3.14, 4.11 and 5.9 (Sep 25) are appended after section 8, in that order, so that no earlier section changed; the rendered HTML places each in its part. The Sep 25 sections cut off at main `614a7a8`, which includes the 2026-09-25 23:00 EDT entry of `RESEARCH_CONTEXT.md`. A reader new to the project should start with section 0 of `RESEARCH_CONTEXT.md`, rewritten for Sep 25 at 19:30 EDT, which states the paper's spine, the runs, the dataset and what is open. Then read its section 7 log from Sep 23 onward, newest first; entries after 19:30 can revise section 0. Then read sections 3.12 to 3.14 of this dossier for the next-tic results and the distance study, sections 4.10 and 4.11 for the decisions behind them, and section 5.9 for the literature on distance and persistence baselines. Then read the two memos, `.claude/analyses/distance-study-design-2026-09-24.md` and `.claude/analyses/distance-study-literature-2026-09-25.md`. Sections 1 to 8 remain the background: the stride-four rows, the recipe comparison, the GameNGen reading and the repository map. Section 8.3 ends with pointers to the parts of the paper a co-author can take up now.

# 1. The problem and the goal

## 1.1 What is being learned

DoomDiT learns a conditional distribution over the next visual observation of a game. The conditioning variables are recent observations and the controls that connect them to the target observation. The dynamics model operates on autoencoder latents. A decoder converts a predicted latent into an image for display and evaluation. During autoregressive use, the model's predicted latent enters the next context window. The real engine supplies neither the next image nor a corrected hidden state during that rollout. [G §2; `train_wm.py:732–738`; `rollout_eval.py:94–205`.]

Let (I_t) denote an observed image and (z_t=E(I_t)) its normalized encoder output. Let (u_t) denote the executed control taking the engine from observation (t) toward observation (t+1). The next-tic contract is

\[
p_\theta(z_r\mid z_{r-L:r},u_{r-L:r}).
\]

The final conditioning control is (u_{r-1}). The control (u_r) is future information for this prediction and must not enter it. This notation follows our recorder, which stores a frame and its outgoing control before stepping the engine. Other datasets can store incoming controls instead. Copying their array offsets would change the causal task. [`record_arnold.py:255–273`; `doom_data.py:725–746`; RC 2026-09-21 01:30.]

The model is a visual simulator with finite visual memory. It does not execute DOOM's geometry, collision, ammunition, or damage rules symbolically. It must infer relevant state from images and controls. A visually plausible image can still violate the requested action. A temporally smooth rollout can still freeze the camera or forget the map. A correct movement can still receive poor pixel agreement if another valid stochastic event occurs. These are distinct failure modes, so a single scalar cannot characterize the simulator. [G §6; D §§5.2, 6; M §§1, 3; P §4.]

## 1.2 Why PSNR needs a task definition

For images scaled to a common range, PSNR is a logarithmic transform of mean squared pixel error. Its interpretation depends on the reference image, temporal gap, spatial resolution, crop, sampling method, and aggregation rule. The repository averages per-image PSNR; this is not identical to converting a pooled MSE into PSNR. The choice matters when easy static frames and difficult motion frames coexist. [`eval_tf.py:34–43,314–320`.]

A raw persistence predictor returns the last observed raw frame unchanged. Its score measures how difficult the selected temporal transition is for that particular footage. It is a baseline, not a mathematical lower bound: a poor model can score below it. Our dense-data audits show that changing the frame gap changes this baseline substantially. The Arnold seen-footage audit gives **21.59 dB at a one-tic gap** and **18.94 dB at a four-tic gap** on the same sampling instrument. Those are different prediction problems before any model is trained. [Q4: `results/night_2026-09-19/q4-repro-footage/TABLE.md:3–22`.]

The autoencoder reconstruction reference is (D(E(I_t))) compared with (I_t). We have called this the “VAE ceiling.” That name is useful operationally but is not a theorem. An encoder's chosen reconstruction is not necessarily the decoder output with the smallest pixel error for that image. A dynamics predictor can also favor a conditional mean that improves MSE while losing detail. The reference diagnoses representation and rendering error; it must not be described as an absolute bound on all possible latent predictions. This corrects the stronger wording in `REQUIREMENTS.md:38` and Q4's discussion. [`encode_parquet.py:165–189`; `eval_tf.py:278–303`; `REQUIREMENTS.md:38`; Q4 `TABLE.md:36–40`.]

A decoded persistence predictor returns (D(E(I_{t-1}))). It includes autoencoder distortion, unlike raw persistence. Historical `copy_psnr_raw` means that this decoded copy is scored against a raw target. The suffix identifies the reference, not the origin of the prediction. The newer `persist_psnr_raw` compares raw last frame with raw target and is the decoder-independent baseline required for next-tic reporting. [`eval_tf.py:278–303`; RC 2026-09-21 01:30.]

The distinction is equally important for autoregressive scores. Our completed stride-four rollouts were scored against decoded ground-truth latents. Their teacher-forced headline scores were scored against raw frames. A rollout score at the initial prediction can therefore exceed the teacher-forced headline without showing a better model. The new raw-frame rollout path changes the reference and requires a new table label. [RC 2026-09-13 22:16, 2026-09-21 14:30; `rollout_eval.py:289–403`.]

## 1.3 What “stronger than GameNGen” can honestly mean

GameNGen reports **29.43 dB**, **0.249 LPIPS**, and real-time generation using **four DDIM steps**. Its teacher-forced evaluation uses **2,048 held-out trajectory samples from five levels**. Its exact footage, trained world-model weights, and full original training implementation are not available in the project record. We cannot run that original model through our evaluation harness or score our model on its private benchmark. [G §§3.3, 5.1; RC 2026-09-20 21:45; `A/lit/doom-baselines-2026-09-17.md`.]

A larger PSNR on another dataset is not evidence of superiority. Matching GameNGen's sampler step count removes one difference but leaves the data distribution, prediction interval, autoencoder, and evaluation sample construction different. Subtracting each dataset's persistence score improves interpretation but does not create a common benchmark. The residual can still depend on motion, stochasticity, map coverage, and rendering. [G §5.1 and Appendix A.6; Q4 `TABLE.md:16–45`; RC 2026-09-22 09:15.]

The defensible ambition is to produce an openly reproducible recipe with useful visual fidelity, action responsiveness, and rollout stability under a stated compute budget. A stronger empirical claim can concern a controlled comparison on our released benchmark: for example, a pretrained backbone and recipe that outperform another public adaptation at equal training exposure, with separate compute accounting. It can also concern a better measurement protocol that exposes persistence and decoder effects hidden by a headline PSNR. Neither claim requires declaring victory over an inaccessible original experiment. [RC 2026-09-14 20:00, 2026-09-16 contribution review, 2026-09-21 14:30.]

The paper's current center is therefore a small-compute adaptation study and an open dataset and benchmark. The completed backbone comparison is evidence about particular initializations and adaptation pathways. It is not a clean architectural experiment: the warm starts differ in pretraining data, objective, parameter count, and conditioning interfaces. The next-tic runs test a revised data and conditioning recipe. Their results must remain separate from the completed stride-four benchmark. [RC 2026-09-14 20:00, 2026-09-21 01:30.]

# 2. Recipes side by side

## 2.1 Comparison table

“Stride-four” below means the finished corrected Arnold runs, not the April project. “Next-tic” means the design approved on **2026-09-21**, with the executed-control normalization completed on **2026-09-22**. Five backbone families finished the former recipe; the additional DiT seed is a repeat within one family. [RC 2026-09-21 01:30, 2026-09-22 11:30.]

| Ingredient | GameNGen | DoomDiT stride-four: finished families | DoomDiT next-tic: approved design | MultiGen | DIAMOND | PlayGen |
|---|---|---|---|---|---|---|
| Environment and agent | DOOM; PPO agent trained during collection; all stages of agent learning contribute footage. Exact WAD and map names not stated. [G §§3.1, 4.1; A.5.] | ViZDoom, Freedoom assets, Arnold deathmatch policy, eight bots, HUD and crosshair. [RC 2026-09-14 10:20.] | Same Arnold environment; fixed published Arnold arena split. [RC 2026-09-21 01:30.] | ViZDoom; pretrained Doom agent; Obsidian-generated maps; multiplayer agents on a separate deathmatch map. [M §§4.1, 5.2.] | Atari with learned actor; CS:GO demonstration extension. [D §§4, 6.] | Doom and Mario; hybrid random/expert collection, action repetition, trajectory balancing. [P §§3.2, 4.] |
| Data volume | 70M frames; training episodes and hours not stated. Five levels belong to the reported evaluation description; full collection map count not stated. [G §§4.2, 5.1.] | 850 episodes across 17 arenas, 4,212,561 raw tics (33.43 recorded hours, derived at 35 Hz); 1,054,647 verified encoded frames across the corpus; 675 training, 75 validation, 100 original unseen-map episodes. Separate reporting footage. [RC 2026-09-09, 2026-09-13 22:30.] | Full arenas pool: 8,000 episodes, 40,285,059 tics; unseen arenas: 3,000 episodes, about 14.8M tics. Approved training prefix: 2,000 episodes, about 10.07M raw tics (about 79.9 recorded hours, derived at 35 Hz). [RC 2026-09-20 08:30, 21:45; 2026-09-21 01:30.] | More than 10M frames across 100 generated maps; another more-than-10M multiplayer collection. Hours and episode counts not stated. [M §§4.1, 5.2.] | Atari: 100k environment interactions per game/run. CS:GO: 5.5M frames, 95 h total; 5M/87 h training and 0.5M/8 h test. [D §6; Appendix E.] | Doom: 900M collected frames, 200M after balancing. Mario: 236M collected, 50M after balancing. Hours, map counts, episodes not stated. [P §4.] |
| Frame spacing and action repeat | PPO actions repeat for four frames, with an increased probability of repeating the previous action. Exact diffusion-training stride not stated. Appendix evaluation also uses 35 FPS. [G A.5–A.6.] | Verified transitions four tics apart; engine clock 35 Hz; model rate 8.75 predictions per game-second, derived. Decision phase is not globally fixed. [RC 2026-09-09; `transitions.py:163`.] | One tic per target, 35 predictions per game-second; Arnold usually repeats controls for four tics, with overrides and life boundaries. [`release/DATASET_CARD.md:18–20,62–65`.] | Training stride and action-repeat count not stated. Inference is about 20 FPS. [M §5.3.] | Atari frameskip four; CS:GO footage 16 Hz. [D Appendix E; §6.] | Action repetition is stated; numerical repeat and stored-frame spacing not stated. [P §3.2.] |
| Resolution and representation | 320×240 padded to 320×256; SD latent diffusion. Derived latent shape 4×32×40 from the named autoencoder. [G §§3.2, 4.2.] | Same image dimensions; C4 SD latents for DiT/U-Net/PixArt/UniDiffuser; C16 SD 3.5 latents for that row. [`doom_data.py:191`; `backbones.py:34–74`.] | Same dimensions and separate C4/C16 corpora. Decoder crop removes padded rows. [`encode_parquet.py:155–189`.] | Exact resolution, VAE, latent shape and scaling not stated. [M §§3–5.] | Pixel-space Atari 64×64. CS:GO dynamics 56×30, upsampled to 280×150. [D §§4.1, 6.] | 128×128 images; own VAE, latent 4×32×32. [P §4; Appendix A Table 4.] |
| Backbone and initialization | SD 1.4 U-Net, all denoising weights unfrozen. Adapted parameter count not stated. [G §3.2.] | ImageNet DiT-XL/2; SD 1.4 U-Net; PixArt-alpha 512; UniDiffuser U-ViT; SD 3.5 Medium. Approximate sizes: 673M, 860M, 612M, 1B, and 2.246B respectively; wrapper counts differ. [RC 2026-09-14 22:30, 2026-09-18 07:30; `backbones.py:261–791`.] | Public SD 1.4 and SD 3.5 Medium starts; PixArt-alpha optional third row. No initialization from our completed world models. [RC 2026-09-21 01:30.] | U-Net; exact size and public initialization not stated. [M §3.] | EDM U-Net, about 4M Atari parameters; CS:GO system 381M including 51M upsampler. [D §§4.1, 6.] | Recurrent DiT, 131M; VAE 55M. Public pretrained initialization not stated. [P §3.3; Appendix A Table 4.] |
| Decoder handling | Encoder frozen; decoder separately fine-tuned with MSE for image/HUD quality. [G §3.2.2.] | Tuned frozen-encoder decoder, usually MSE + 0.1 LPIPS; own C16 decoder for SD 3.5. Existing tunes saw maps outside dynamics training. [RC 2026-09-15–18, 2026-09-21 14:30.] | Final decoder deferred. An honest unseen-map pipeline needs a training-map-only tune or an explicit dynamics-only qualification. [RC 2026-09-21 14:30.] | Decoder training details not stated. [M §3.] | No latent decoder for Atari; learned upsampler for CS:GO. [D §6.] | VAE trained with reconstruction, LPIPS and KL, then frozen for dynamics. [P §3.3; Appendix A.] |
| Visual context | 64 past frames concatenated along channels. Exact training seconds not stated. If consecutive native tics, nominal span is 64/35 ≈ 1.83 s, an inference. [G §3.2; A.6.] | 32 decision frames along channels; nominal span 128/35 ≈ 3.66 s, derived. [RC 2026-09-13 22:16.] | 32 consecutive tics along channels; nominal span 32/35 ≈ 0.914 s, derived. [RC 2026-09-21 01:30.] | Main model context 32; Table 3 tests 2–32. Frame-to-seconds conversion not stated. [M §6, Table 3.] | Four prior frames and actions for Atari, channels for observations. CS:GO uses the same model family; separate context disclosure and Atari context duration in seconds are not stated. [D Appendix E; §6.] | Recurrent hidden memory, not an equivalent fixed raw-frame stack; training sequence length 36; context duration in seconds not stated. [P Figure 2; Appendix A Table 4.] |
| Action pathway | Embedding token per past action through cross-attention; 64-action history. Exact action vocabulary and positional encoding not stated. [G §3.2.] | One of 29 requested action IDs, most recent source action. DiT adaLN; U-Net cross-attention token; transformer caption/text-token replacements; SD 3.5 also pooled conditioning. [`backbones.py:261–791`.] | 32 executed 19-bit controls, MLP plus learned positions; U-Net cross-attention; SD 3.5 joint-attention tokens with newest-control pooled vector. [`backbones.py:153–217,771–790`; RC 2026-09-21 01:30.] | Action embedding through cross-attention; equations emphasize current action, diagram depicts history. Exact count not stated. [M §3, Figure 2.] | Action and noise embeddings through adaptive group normalization; four-action history. [D Appendix E.] | Action and noise embeddings condition transformer through cross-attention. [P §3.3, Figure 2.] |
| Context corruption and stability | Gaussian noise augmentation up to 0.7, ten learned buckets; inference noise can be adjusted. Exact corruption formula not stated. [G §3.2.1.] | VP corruption with q uniform below 0.7, ten buckets, one q per context sample; clean inference by default. [`diffusion_v.py:135–151`.] | Same corruption retained; proposed exactly-clean training mass withdrawn. [RC 2026-09-21 14:30.] | Context noise plus explicit geometry/pose memory and history guidance; maximum, buckets and guidance coefficient not stated. [M §§3.1–3.4.] | EDM preconditioning; few-step sampling. No claim that this is the GameNGen context-corruption recipe. [D §§3, 5; Appendix C.] | Diffusion Forcing with recurrent state; balanced data and long-tail sampling. [P §§3.2–3.3.] |
| Diffusion objective and schedule | Velocity MSE; “linear” schedule referenced to latent diffusion; exact beta endpoints and implementation not stated. [G §3.2.] | VP velocity MSE, 1,000 linearly spaced betas from 1e-4 to 0.02; no learned variance head in this trainer. [`diffusion_v.py:38–102`.] | Same schedule retained despite SD scaled-linear and SD 3.5 flow mismatch. [RC 2026-09-21 14:30.] | Velocity objective; exact schedule not stated. [M §3.] | EDM denoising objective with continuous sigma and preconditioning; not our VP velocity objective. [D §3; Appendix C.] | Predicts clean latent with Diffusion Forcing; 1,000 diffusion steps and sigmoid schedule. [P §3.3; Appendix A Table 4.] |
| Optimizer and learning rate | Adafactor, 2e-5 constant, weight decay zero, clip norm 1. [G §4.2.] | Nominal fused AdamW, 5e-5 after 2,000 warmup updates, weight decay zero, clip norm 1; UniDiffuser rescue deviations in §4.7. [RC 2026-09-13 22:16; `train_wm.py:577–890`.] | Same; global batch fixed rather than increased with device count. [RC 2026-09-21 01:30.] | Optimizer, LR, decay, clipping and warmup not stated. [M §§3–6.] | Atari AdamW 1e-4, epsilon 1e-8, dynamics decay 0.01. Separate full CS:GO settings not stated. [D Appendix E.] | Dynamics LR 1e-4; VAE LR 4.5e-6. Optimizer, warmup and EMA not stated. [P Appendix A Table 4.] |
| Batch, updates and EMA | Batch 128, 700k updates; warmup and EMA not stated. Decoder batch 2,048, other training parameters stated to be identical. [G §4.2.] | Global batch 32; 90k-update budgets; CPU fp32 EMA decay 0.9999, updated every eight optimizer steps with adjusted decay. [RC 2026-09-13; `train_wm.py:419–422,577–890`.] | Batch 32 without accumulation; two-device training means 2×16. Train to budget/freeze; fixed final step not yet measured. [RC 2026-09-21 01:30, 2026-09-22 09:45.] | Batch, updates and EMA not stated. | Atari batch 32, 1,000 epochs × 400 dynamics updates = 400k updates, derived. EMA not stated. [D Appendix E.] | Dynamics batch 8, VAE batch 32; total updates and EMA not stated. [P Appendix A Table 4.] |
| Sampler | DDIM, normally four steps; distilled single-step variant also tested. [G §3.3, Table 1; A.6.] | DDIM 50, eta zero for reported corrected rows. April model instead used ancestral respaced DDPM. [`diffusion_v.py:104–132`; RC 2026-09-02, 2026-09-13.] | Number of steps deferred; same trained objective must be respected. [RC 2026-09-21 01:30.] | Exact sampler and number of steps not stated. [M §5.3.] | Three Euler evaluations for dynamics; CS:GO upsampler uses ten stochastic steps. [D §§5.2, 6.] | Step sweep 4/8/16; main Doom table matches eight-step row, an inference. Sampler algorithm not clearly stated. [P Tables 1–2.] |
| Classifier-free guidance | Observation dropout 0.1; observation CFG 1.5. Action CFG did not improve quality. [G §§3.3, 4.2.] | No observation dropout. Launched action dropout zero. Observation CFG unavailable. [RC 2026-09-13 22:16, 2026-09-21 14:30.] | Same limitation; post-hoc sampler selection cannot manufacture a trained unconditional branch. [RC 2026-09-21 14:30.] | History guidance stated; exact dropout training and coefficient not stated. [M §3.4.] | Comparable observation-CFG recipe not stated. | Comparable observation-CFG recipe not stated. |
| Evaluation and references | 2,048 teacher-forced held-out samples from five levels; 512 rollout samples; PSNR, LPIPS, FVD and human comparison. Explicit held-out-map split not stated. [G §5.1.] | 2,048 TF windows per corpus; 256×64 seen rollouts; raw TF references, decoded rollout references; IDM and FVD. Reporting seen/unseen/unseen2 corpora separate from fitting. [RC 2026-09-13, 2026-09-16.] | Fixed episode subsets, raw persistence and raw rollout targets; exact reporting windows, horizons, decoder and sampler deferred. [RC 2026-09-21 01:30, 14:30.] | Level-conditioned and multiplayer experiments; Table 1 PSNR/SSIM/LPIPS, Table 2 opponent coherence, Table 3 context ablation. Exact evaluation sample counts not stated. [M §§4–6.] | Main outcome is Atari return; CS:GO extension primarily qualitative. No directly comparable Doom PSNR benchmark. [D §§4–6.] | 600 test trajectories, one real seed frame then autoregression; PSNR, LPIPS, FID, FVD, action accuracy/probability difference. [P §4.] |
| Compute and wall clock | 128 TPU v5e for training; total wall clock not stated. Single TPU v5 inference about 20 FPS at four steps. [G §§3.3, 4.2.] | A6000/A4000 history with different throughput and checkpointing; not a single verified total GPU-hour figure. [RC 2026-09-09, 2026-09-14–18.] | Forecast: eight H100s briefly for encoding, then two H100s for 48 h, about $650; optional A6000 NVLink pair for PixArt. [RC 2026-09-22 09:45.] | Training compute not stated; about 20 FPS on one A100 per player. [M §5.3.] | Atari 2.9 days per run on one RTX 4090; about 12 GB memory. CS:GO 12 days on one RTX 4090; inference 10 Hz on RTX 3090. [D §§4.1, 6.] | Training compute not stated; Doom 20/10/5 FPS at 4/8/16 steps on RTX 2060. [P Table 2.] |
| Release | Original training footage, code and weights unavailable in our record; paper/project videos available. [RC 2026-09-20; `A/lit/doom-world-models.md`.] | Repository, experiment artifacts and historical release checkpoint; some final metrics survive only in RC summaries. [RC §§2–7.] | Dataset upload underway; first priority batch confirmed uploaded, later batches pending at cutoff. [RC 2026-09-22 11:30.] | Release of complete recipe/data/weights not stated in paper. | Code, trained agents and playable world-model materials linked by paper. [D §1 and project link.] | Repository available; current README marks Doom inference weights and training data as unfinished, unlike Mario inference. [PlayGen repository README:68–73.] |

## 2.2 What each ingredient comparison actually establishes

**Environment and agent.** We matched the use of agent-generated gameplay, not the original distribution. GameNGen trains its own PPO agent and retains trajectories throughout learning. Arnold is a public pretrained deathmatch policy with different objectives, opponents, maps and assets. Its footage is not a substitute for private GameNGen footage. MultiGen is closer to our use of a pretrained Doom agent, but its explicit map memory changes the information available to the model. [G §§3.1, 4.1; M §4.1; RC 2026-09-09.]

**Data volume.** A raw frame count, a valid-window count and a count of training presentations answer different questions. Our verified stride-four corpus contains **1,054,647 encoded frames across all splits**, not that many independent training examples. A context window cannot cross a rejected transition or life boundary. The next-tic prefix contains about **10.07M raw tics**, but its actual valid-window count must be computed after continuity and split checks. At batch **32**, a nominal **90k**-update run presents **2.88M** windows, derived. These presentations can repeat the same underlying footage. [RC 2026-09-13 22:30, 2026-09-21 01:30; `doom_data.py:260–340,465–514`.]

**Frame spacing and action repeat.** We deliberately moved from verified decision intervals to every recorded tic. Repeating a control does not imply dropping intermediate observations. GameNGen's action-repeat statement therefore cannot establish its training stride. Its **20 FPS** generation rate is a hardware throughput statement. We can compare our next-tic task with the hypothesis of consecutive-tic GameNGen training, but must label that hypothesis. [G §3.3, A.5–A.6; RC 2026-09-21 01:30.]

**Resolution and latent space.** We matched GameNGen's padded image geometry in the rebuilt pipeline. The image is normalized before zero padding; zero in this normalized space corresponds to mid-gray, despite a “black” comment in the encoder. The C4 contract is scaled by **0.18215**. SD 3.5 uses **16 channels**, shift **0.0609**, and scale **1.5305**. A Flux autoencoder is not interchangeable merely because it has the same channel count: its scale and shift differ. [`encode_parquet.py:155–170`; `backbones.py:34–74`; RC 2026-09-17 11:45 and 2026-09-18 14:50.]

**Backbone and initialization.** The completed rows test usable public starts under a shared adaptation recipe. They do not isolate convolution versus attention. DiT's ImageNet prior, the SD U-Net's text-image prior, PixArt's text-image prior, UniDiffuser's joint modeling prior, and SD 3.5's flow prior are different treatments. Zero-inflating the context channels preserves the pretrained target projection at initialization, but does not prove the entire conditional network initially computes the old model's function. New conditioning embeddings can change downstream activations. [`backbones.py:241–259,344–388,675–790`; RC 2026-09-14 20:00; `A/astra-prelaunch-audit-2026-09-21.md`.]

**Decoder.** We matched the frozen-encoder, separately tuned decoder strategy. We deliberately added a perceptual term after an MSE-only tune improved PSNR while degrading LPIPS. The decoder affects rendered metrics without changing the latent rollout feedback. That makes post-training decoder selection technically possible. It does not excuse test-map leakage or selection on test metrics. [G §3.2.2; RC 2026-09-15, 2026-09-21 14:30; `finetune_decoder.py:49–64,275–426`.]

**Context.** The old and new recipes have the same frame count but different physical memory spans. Their nominal spans are **3.66 s** and **0.914 s**, respectively, derived from **32** observations at gaps **4/35 s** and **1/35 s**. The elapsed time between oldest and newest observed frame is instead **31** intervals: approximately **3.54 s** and **0.886 s**, derived. Every context comparison should state which convention it uses. [RC 2026-09-21 01:30; `release/DATASET_CARD.md:20`.]

**Action conditioning.** The completed models used only the latest scalar action. The new design adopts GameNGen's history-token idea while representing executed controls rather than assuming an identical action vocabulary. Each control vector passes through a shared MLP and receives a learned position embedding. For the U-Net these tokens enter cross-attention. For SD 3.5 they enter joint attention, with the latest control also furnishing the pooled slot. These are implementation choices for public backbones, not claims about GameNGen's undisclosed embedding internals. [`backbones.py:153–217,379–388,771–790`; G §3.2.]

**Context noise.** Our implementation is (\tilde c=\sqrt{1-q}\,c+\sqrt q\,\epsilon\), with one sampled (q\) shared across the context of an example. The maximum **0.7** is a variance fraction in this formula, not an additive noise standard deviation. At that endpoint the signal and noise coefficients are approximately **0.548** and **0.837**, derived. GameNGen states a maximum and bucket count but does not state this exact formula. The Stiegler reproduction's additive formulation is also different. [`diffusion_v.py:135–151`; G §3.2.1; `A/lit/rollout-stability-training.md`.]

**Objective and schedule.** Our target is (v=\sqrt{\bar\alpha_t}\epsilon-\sqrt{1-\bar\alpha_t}x_0\), with an unweighted MSE and uniformly sampled training time. The linear beta schedule differs from SD 1.4's native scaled-linear schedule and from SD 3.5's rectified flow. Retaining the shared schedule preserves continuity with the finished rows. It does not establish that this schedule is optimal for either warm start. [ `diffusion_v.py:38–102`; RC 2026-09-21 14:30.]

**Optimization.** We changed GameNGen's Adafactor and constant start to fused AdamW with warmup. The shared learning rate was a reviewed adaptation decision after the earlier DiT excursion. A larger relative update in a modulation matrix was a diagnostic, not proof of the excursion's cause. The fair statement is that this was a common recipe chosen for a bounded comparison, with probes and non-finite-update handling. [G §4.2; RC 2026-09-10, 2026-09-13 22:16.]

**Batch, updates and EMA.** Equal update counts at equal global batch match exposure, not FLOPs or wall clock. Model size, recomputation, communication and VAE channels still change cost. Our EMA arithmetic is fp32 because the old bf16 update could underflow. The new snapshot protocol supports live and EMA evaluation at the same numeric checkpoint. Comparing a validation-selected live `best.pt` with a final EMA is descriptive, not a controlled EMA ablation. [`train_wm.py:419–422`; RC 2026-09-02, 2026-09-21 14:30.]

**Sampler.** The rebuilt recipe uses DDIM; the old April implementation used ancestral DDPM despite misleading DDIM comments. The completed DDIM step sweep shows that fewer steps can increase PSNR while making LPIPS much worse. Sampler steps are therefore part of the experiment identity. Matching the original's small step count is useful, but it does not make different datasets directly comparable. [RC 2026-09-02, 2026-09-22 09:15; G Table 1.]

**Guidance.** GameNGen's observation CFG requires a model trained on dropped observations. Our completed and approved recipes do not contain that training branch. A numerical guidance coefficient cannot be added afterward with the same meaning. Action dropout is also distinct from observation dropout. The parser's generic default is less authoritative than the launcher's explicit zero action dropout. [G §§3.3, 4.2; RC 2026-09-13 22:16, 2026-09-21 14:30.]

**Evaluation.** We matched the broad use of teacher-forced fidelity, autoregressive clips and distributional video metrics. We added an inverse-dynamics judge and explicit map splits. We could not match the original windows, levels, decoder or human study. A seen-map episode holdout is different from unseen-map evaluation. A decoder exposed to a held-out map weakens a pipeline-wide unseen-map claim even when the dynamics model never saw that map. [G §5.1; RC 2026-09-21 14:30.]

**Compute.** Hardware names and batch sizes are insufficient for a compute-normalized comparison. The paper needs measured peak memory, updates per second, card-hours, and inference latency under each final configuration. Historical short fit checks sometimes shared a card with other jobs; their absolute rates are not portable estimates. The H100 plan remains a forecast until a clean throughput sweep measures it. [RC 2026-09-14 15:40, 2026-09-22 09:45.]

**Release.** Public weights alone do not make a benchmark reproducible. The release must identify raw recording hashes, split membership, encoder identity, decoder training set, action semantics, checkpoint identity and evaluation windows. Our first priority dataset batch is recorded as uploaded, but that is not evidence that all promised folders and metadata are already public. The dataset card's license remains subject to Rohan's confirmation in the decision log. [RC 2026-09-22 09:15, 11:30; `release/DATASET_CARD.md`.]

# 3. Results so far

## 3.1 Evaluation namespaces and rules for reading the tables

The original corrected corpus and the dense corpus are separate datasets. The historical reporting corpus contains **60 seen-map episodes**, **20 unseen-map episodes**, and **130 unseen2 episodes**. The first group covers the original dynamics-training arenas; the second covers arenas **16–17**; the third covers **13 curated campaign maps**. The main corrected training split contains **675 episodes**, with **75** for validation and **100** original unseen-map episodes. Reporting episodes are separately seeded. The original training recording predates the fully deterministic recorder and cannot retrospectively acquire its reproducibility guarantee. [RC 2026-09-13 22:16–22:30; 2026-09-16 unseen2 entries.]

The unseen2 maps were curated for a deathmatch-style experiment. Monsters and selected pickups were removed, and maps where the agent became stuck were excluded. This is a disclosed curated distribution, not a random sample of all unseen DOOM maps. The historical decoder also saw maps beyond the dynamics training set. “Unseen by the dynamics model” is the supported label until a training-map-only decoder is used. [RC 2026-09-16 15:40, 17:10, 2026-09-21 14:30; `A/lit/doom-maps-prior-work.md`.]

The following tables use PSNR in dB, with higher better, and LPIPS with lower better. FVD is lower better. IDM top-1 is higher better but is interpreted against a real-data reference and a majority-class baseline. Validation velocity loss is lower better only within an unchanged latent and corruption contract. A C16 loss and a C4 loss are not calibrated measures of the same difficulty. [`diffusion_v.py:91–102`; `rollout_eval.py:409–514`; RC 2026-09-13 22:30.]

A dash below means not locally recovered, not zero. Values reconstructed from RC summaries retain the precision recorded there. This dossier does not manufacture missing raw artifacts or uncertainty intervals.

## 3.2 Finished stride-four teacher-forced rows

**Protocol:** historical reporting seen/unseen/unseen2 corpora; **2,048 teacher-forced windows per corpus**, seed **0**, **50-step DDIM**, eta **0**, raw target frames, tuned C4 decoder for the C4 families. Rows use live validation-selected weights after training budgets of **90k updates**. The exact selected checkpoint step is not locally recovered for every family. The U-Net and PixArt checkpoints used in the later Q1 sweep are identified as **87k** and **89k** respectively. Validation loss is the logged training-run endpoint, not a common raw-image metric. Sources: RC 2026-09-16 09:40, 14:30, 19:40; 2026-09-18 14:50, 16:35; Q1 `provenance.json` files; `eval_tf.py:194–335`.

| Backbone / repeat | Seen PSNR / LPIPS | Unseen PSNR / LPIPS | Unseen2 PSNR / LPIPS | Final validation loss |
|---|---:|---:|---:|---:|
| DiT-XL/2, seed 0 | 21.06 / 0.311 | 19.52 / 0.450 | 21.43 / 0.413 | 0.2139 |
| DiT-XL/2, seed 1 | 21.21 / 0.307 | 19.59 / 0.442 | 21.59 / 0.404 | 0.2139 |
| SD 1.4 U-Net | 21.36 / 0.270 | 19.14 / 0.446 | 21.15 / 0.412 | 0.2043 |
| PixArt-alpha | 21.35 / 0.272 | 19.39 / 0.434 | 21.29 / 0.411 | 0.2031; best 0.2030 |
| UniDiffuser U-ViT | 21.11 / 0.293 | 19.25 / 0.454 | 21.25 / 0.419 | 0.2048 |
| Decoded-copy baseline | 19.41 / — | 18.50 / — | 20.92 / — | N/A |
| Tuned C4 reconstruction reference | 28.61 / 0.051 | 26.36 / 0.070 | 28.89 / 0.058 | N/A |

The original hope that the ImageNet-initialized DiT would win the matched recipe did not materialize. The U-Net and PixArt have better seen-map perceptual fidelity. The DiT has competitive or higher PSNR on some unseen distributions, but that does not by itself establish better dynamics. UniDiffuser did not turn a larger public transformer into a clear improvement. These are useful negative results about adaptation under this budget. They are not evidence that all transformers are worse than U-Nets. [RC 2026-09-14 20:00; 2026-09-16 contribution review.]

**Historical EMA comparison:** same seen reporting corpus, **2,048 windows**, **50-step DDIM**, tuned C4 decoder, raw target. These are final-recovery EMA scores beside historical validation-selected live scores, so checkpoint selection is not controlled. Source: RC 2026-09-16 09:40, 14:30, 19:40; 2026-09-18 14:50, 16:35.

| Row | Live PSNR / LPIPS | EMA PSNR / LPIPS |
|---|---:|---:|
| DiT seed 0 | 21.06 / 0.311 | 21.06 / 0.306 |
| DiT seed 1 | 21.21 / 0.307 | 21.10 / 0.304 |
| U-Net | 21.36 / 0.270 | 21.67 / 0.250 |
| PixArt | 21.35 / 0.272 | 21.60 / 0.255 |
| UniDiffuser | 21.11 / 0.293 | 21.34 / 0.281 |

This motivated reporting both weight variants. It did not justify silently replacing every live row by its best-looking EMA score. The approved next-tic evaluator pairs live and EMA from the same numeric-step checkpoint and leaves selection to validation. [RC 2026-09-21 14:30.]

## 3.3 Finished stride-four autoregressive rows

**Protocol:** historical seen reporting corpus; **256 rollouts**, **64 predicted decision frames**, **50-step DDIM**, clean context at inference, tuned C4 decoder, decoded ground-truth references. Live weights follow the historical checkpoint selection described above. Horizons are decision-frame counts; the last horizon spans about **7.31 game-seconds**, derived as **64×4/35**. FVD columns use clip lengths **16** and **32**. Sources: RC 2026-09-16 09:40, 14:30; 2026-09-18 14:50, including PixArt's 2026-09-16 14:30 entry.

| Row | PSNR h8 | PSNR h16 | PSNR h32 | PSNR h64 | LPIPS h64 | IDM top-1 | FVD16 / FVD32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DiT seed 0 | 19.66 | 18.93 | 18.51 | 18.30 | 0.553 | 0.492 | 232 / 481 |
| DiT seed 1 | 19.86 | 19.09 | 18.65 | 17.68 | 0.550 | 0.476 | 198 / 407 |
| U-Net | 19.55 | 18.66 | 17.73 | 16.03 | 0.566 | 0.509 | 184 / 356 |
| PixArt | 19.47 | 18.66 | 18.03 | 17.18 | 0.584 | 0.492 | 211 / 472 |
| UniDiffuser | — | — | — | 17.00 | 0.551 | 0.472 | 206 / 423 |
| Real-sequence IDM reference | N/A | N/A | N/A | N/A | N/A | 0.847 | N/A |
| Majority-class IDM reference | N/A | N/A | N/A | N/A | N/A | 0.232 | N/A |

The U-Net's lower long-horizon PSNR coexists with better FVD and higher action accuracy than the first DiT seed. This combination prompted a motion audit rather than a claim that the higher-PSNR model simulated longer. The second DiT seed also changed the apparent long-horizon advantage. A single seed's drift curve was not a stable architectural result. [RC 2026-09-16 motion/contribution entries.]

**Motion audit protocol:** the paired historical **256×64** seen rollouts above, **50-step DDIM**, tuned C4 decoded references. Optical-flow and frame-difference ratios compare generated motion with real motion under that rendering path. Sources: RC 2026-09-16 14:30; `A/` contribution and motion analyses cited in RC.

| Row | Flow ratio h8 / h16 / h32 / h64 | Frame-difference energy ratio h8 / h16 / h32 / h64 | Closer to seed than target at h64 |
|---|---|---|---:|
| DiT seed 0 | 0.75 / 0.71 / 0.68 / 0.58 | 0.83 / 0.82 / 0.87 / 0.75 | 59% |
| DiT seed 1 | 0.78 / 0.74 / 0.75 / 0.73 | 0.77 / 0.79 / 0.81 / 0.76 | 51% |
| U-Net | 0.85 / 0.81 / 0.78 / 0.75 | 0.89 / 0.91 / 0.97 / 0.95 | 35% |

The copy-seed reference at the final horizon scored **17.86 dB / 0.510 LPIPS** in this audit. Its perceptual score was better than every model in that comparison. The first DiT seed's smaller motion was therefore a plausible contributor to its PSNR advantage. Blur controls did not materially reproduce that advantage, so the evidence points toward reduced motion rather than only spatial smoothing. This is a diagnosis of these outputs, not a proof that every DiT rollout freezes. [RC 2026-09-16 14:30.]

Episode-bootstrap uncertainty was also computed for selected teacher-forced comparisons. Seen PSNR was **21.06 [20.74, 21.38]** for DiT seed zero and **21.36 [21.01, 21.73]** for the U-Net. Seen LPIPS was **0.311 [0.301, 0.320]** versus **0.270 [0.261, 0.280]**. These intervals support the perceptual difference more clearly than the small PSNR difference. Episode resampling over the unseen corpus remains conditional on its particular maps; it is not uncertainty over arbitrary unseen-map draws. [RC 2026-09-16 10:00.]

The learned judge is itself imperfect. The corrected windowed IDM achieved **0.829 top-1**, **0.548 macro recall**, and **0.849 movement accuracy** with its longer input window. Its shorter-window variant achieved **0.701**, **0.406**, and **0.727** respectively. The corresponding majority reference is around **0.23**. A model can exploit visual regularities that help this judge without implementing the engine correctly. Its real-data score should be called a reference, not a mathematical ceiling. [RC 2026-09-13 22:30; 2026-09-14 K=2 IDM entry.]

## 3.4 SD 3.5 final live and EMA

**Protocol:** historical reporting seen/unseen/unseen2 corpora, **2,048 TF windows per corpus**, **50-step DDIM**, final **90k-update** checkpoint, own tuned C16 decoder, raw TF references. The SD 3.5 decoded-copy values differ from the C4 values because the rendering path differs. Source: RC 2026-09-22 09:15.

| Weights / reference | Seen PSNR / LPIPS | Unseen PSNR / LPIPS | Unseen2 PSNR / LPIPS |
|---|---:|---:|---:|
| Live | 21.21 / 0.263 | 19.24 / 0.422 | 21.17 / 0.402 |
| EMA | 21.51 / 0.242 | 19.37 / 0.395 | 21.48 / 0.381 |
| Own decoded-copy reference | 19.29 / — | 18.28 / — | 20.79 / — |

**Rollout protocol:** historical seen reporting rollouts, **256×64**, **50-step DDIM**, live final **90k** checkpoint, tuned C16 decoder and its decoded ground-truth reference. Source: RC 2026-09-22 09:15; historical suite specification in RC 2026-09-13 22:16.

| PSNR h64 | IDM top-1 | FVD16 / FVD32 | Validation velocity loss |
|---:|---:|---:|---|
| 17.77 | 0.49 | 202 / 458 | Approximately 0.110 in the status/audit record; exact endpoint not locally recovered |

For this row, the C4-trained IDM requires a decode/re-encode bridge. Both generated and real clips must traverse it, and the corresponding real reference must be reported. Comparing its raw action-accuracy number with a native-C4 judge score without that qualification is not controlled. [`rollout_eval.py:409–417`.]

An automatic evaluator initially selected the stock decoder because its file-existence test required mutually alternative weight files to coexist. The preserved stock-decoder evaluation gave seen **20.19/0.279** live and **20.42/0.257** EMA, rollout PSNR **17.50**, and FVD **210/468**. These are not an earlier model checkpoint's performance. They are a different decoder applied to the finished model. The launcher was repaired and the tuned evaluation above supersedes that rendering path. [RC 2026-09-20 21:45; `scripts/spiderman/after_sd35.sh:45–60`.]

SD 3.5 supplies the best final perceptual scores in the recorded comparison, especially with EMA. It does not have the highest raw PSNR in every column. Its better autoencoder and larger backbone are both part of the system treatment. The result does not isolate a benefit of its transformer architecture or native pretraining objective. [RC 2026-09-22 09:15; §3.6 below.]

## 3.5 Recipe grid: completed, incomplete, and missing records

**Protocol:** PixArt recipe grid on the corrected stride-four corpus; **30k-update budgets**, global batch **32**, historical reporting **2,048 TF windows per corpus**, **50-step DDIM**, tuned C4 decoder, raw TF targets; **256×64** seen rollouts with decoded references. Unless labeled otherwise, scores are live. Sources: RC 2026-09-19 09:40, 13:30; 2026-09-20 08:30; `scripts/spiderman/launch_cell.sh:1–147`. Completion of an unarchived result is not inferred from the launch script.

| Cell | Seen PSNR / LPIPS | Unseen PSNR / LPIPS | Unseen2 PSNR / LPIPS | h64 PSNR / LPIPS | IDM | FVD16 / FVD32 | Validation loss / status |
|---|---:|---:|---:|---:|---:|---:|---|
| base30k | 20.98 / 0.311 | 19.41 / 0.436 | 21.41 / 0.409 | 17.87 / 0.559 | 0.478 | 186 / 379 | 0.2161 |
| scratch | 20.77 / 0.349 | 19.44 / 0.467 | 21.57 / 0.409 | 17.60 / 0.558 | 0.473 | 345 / 893 | 0.2474 |
| noaug | 21.27 / 0.289 | 19.51 / 0.410 | 21.67 / 0.385 | 17.92 / — | 0.546 | 219 / 469 | Exact endpoint not locally recovered |
| data-1_4 | — | — | — | — | — | — | Named as completed in the dossier request; numerical artifact not locally recovered |
| ctx8 | — | — | — | — | — | — | Named as completed in the dossier request; numerical artifact not locally recovered |
| ctx16 | — | — | — | — | — | — | Stopped at about 27.5k; 25k checkpoint retained; not a completed 30k row |

The base cell's EMA seen score was **20.86/0.317**, below its live score under those metrics. Earlier notes round its FVD to **187/380**; the later noaug comparison records **186/379**, which is the precision used above. This small discrepancy should be resolved from the original metrics before a camera-ready table. [RC 2026-09-19 09:40 and 2026-09-20 08:30.]

The scratch comparison supports retaining public initialization under this budget. Similar PSNR in some columns does not erase worse perceptual fidelity and much worse FVD. It is a comparison of one bounded scratch recipe, not proof that training from scratch cannot work with more data or compute. [RC 2026-09-19 grid results.]

The noaug result is more nuanced than the original assumption that augmentation is unconditionally necessary. Removing augmentation improved teacher-forced fidelity and IDM at this budget. It barely changed final-horizon PSNR, while worsening FVD. This is a measurable tradeoff. It does not establish behavior at longer physical horizons or larger next-tic training budgets. Retaining augmentation for the new runs was a conservative design decision, not a statement that the ablation favored it on every metric. [RC 2026-09-20 08:30; 2026-09-21 14:30.]

The quarter-data cell changes episode coverage, not simply the number of updates. The context cell changes both visible history and input structure. Their missing numerical records are a genuine dossier gap that must be closed by locating the original artifacts. There is no responsible interpolation from the base, scratch or noaug rows. [`doom_data.py:229–249`; `scripts/spiderman/launch_cell.sh`; local `results/levers_2026-09-20/` inventory.]

## 3.6 VAE reconstruction gates and decoder choices

**Protocol:** standalone reconstruction, not world-model sampling; no diffusion steps or dynamics checkpoint. Paired raw target frames from the listed historical corpora; **2,048 frames** for reporting sets and a **2,000-frame** development gate. Frozen encoder; tuned decoder as named. Metrics are PSNR / LPIPS / HUD PSNR. Sources: RC 2026-09-09 12:30, 2026-09-18 00:30, 14:50; `finetune_decoder.py:259–273`.

| Corpus | Tuned SD C4 | Tuned SD 3.5 C16 |
|---|---:|---:|
| Seen | 28.61 / 0.0509 / 31.79 | 31.68 / 0.0247 / 34.60 |
| Unseen | 26.35 / 0.0699 / 30.36 | 29.62 / 0.0311 / 33.63 |
| Unseen2 | 28.89 / 0.0581 / 30.83 | 31.94 / 0.0263 / 34.06 |
| Development | 28.34 / 0.0508 / 32.09 | 31.42 / 0.0246 / 34.73 |

The C16 autoencoder passed the gate of a material PSNR improvement without a perceptual or HUD regression. The recorded criterion was at least **1 dB**, with paired bootstrap intervals excluding no improvement. This justified evaluating a different latent representation as a system row. It did not make that row a same-latent backbone comparison. [RC 2026-09-17 11:45 and 2026-09-18 14:50.]

**Decoder development protocol:** reconstruction of the same development target set, no diffusion sampling; frozen encoder, named decoder treatment. Sources: RC 2026-09-09 05:00, 12:30; 2026-09-18 00:30, 07:30.

| Decoder treatment | PSNR | LPIPS | HUD PSNR | Interpretation |
|---|---:|---:|---:|---|
| SD C4 stock | 23.69 | 0.095 | 17.93 | Poor HUD reconstruction |
| SD C4 MSE-only tune | 29.11 | 0.276 | 34.68 | Better pixel/HUD error, worse perceptual fidelity |
| SD C4 MSE + 0.1 LPIPS tune | 28.34 | 0.051 | 32.09 | Adopted compromise for historical rows |
| Flux C16 stock | 27.46 | 0.030 | 21.46 | Different latent normalization |
| Flux C16 tuned development result | 32.44 | 0.018 | 35.65 | Promising measured gate; usable saved tuned weights were lost |

The Flux save failed because non-contiguous tensors were passed to the serialization path. A later rerun was interrupted. The development metric is evidence that the tune ran, not evidence that a corresponding usable release checkpoint exists. The SD 3.5 stock seen reconstruction was **26.47 dB**, with HUD PSNR about **20.4 dB**, emphasizing how much decoder tuning changes the system. [RC 2026-09-18 07:30, 09:45, 14:50.]

The historical decoder training used **50,000 frames** for **two epochs**, or **100,000 frame presentations**, derived. The recipe used learning rate **1e-5** with **200** warmup updates. The newer implementation crops the loss to the **240 real image rows**. Previously the padded **16 of 256 rows**, or **6.25%** of the image height, contributed an easy gray reconstruction target. These are recipe differences that must accompany comparisons with GameNGen's much larger decoder training. [RC 2026-09-09 12:30, 2026-09-18 14:50, 2026-09-21 14:30; `finetune_decoder.py:275–426`.]

## 3.7 Persistence by temporal gap

**Protocol:** raw-frame persistence only; no model checkpoint, no denoising steps, no decoder. Each dense segment audit samples **200 episodes** and **100 anchors per episode**, giving **20,000 frame pairs** for each gap. Per-map episode counts differ because this was a sample. Values are mean PSNR. Sources: `results/night_2026-09-19/q2-verify-dense/stats_arenas.json:1` and `stats_arenas_678.json:1`, including their `per_map` fields.

| Arena / aggregate | Gap 1 tic | Gap 2 tics | Gap 4 tics | Gap 8 tics |
|---|---:|---:|---:|---:|
| Arena 2 | 21.90 | 20.62 | 19.55 | 18.82 |
| Arena 3 | 22.03 | 20.62 | 18.94 | 17.92 |
| Arena 4 | 23.05 | 21.55 | 20.18 | 19.31 |
| Arena 5 | 20.34 | 18.81 | 17.57 | 16.81 |
| Sampled arenas 2–5 aggregate | 21.78 | 20.35 | 19.02 | 18.19 |
| Arena 6 | 20.83 | 19.49 | 18.25 | 17.46 |
| Arena 7 | 20.11 | 18.21 | 16.76 | 15.88 |
| Arena 8 | 23.10 | 21.53 | 20.15 | 19.14 |
| Sampled arenas 6–8 aggregate | 21.44 | 19.84 | 18.48 | 17.58 |

This table changed the interpretation of “harder unseen maps.” Some unseen arenas have higher persistence than some training arenas. An aggregate PSNR can rise merely because its map mixture has less visual change. The final benchmark should show per-map scores and the same per-map baselines, then state whether its aggregate weights maps, episodes, or windows equally. [Q2 sources above; `eval_tf.py:319–320`.]

## 3.8 Open-reproduction footage: floor and reconstruction reference

**Protocol:** **40 episodes**, **20,000 sampled raw frame pairs per corpus**, seed **0**, **320×240** frames; no world-model checkpoint or denoising. Reconstruction uses the same stock `sd-vae-ft-mse` autoencoder on **20,000 frames** per corpus. Error bars below are the reported standard errors, not episode-bootstrap confidence intervals. Sources: Q4 `TABLE.md:3–22`; `stats_repro.json:1`; `stats_ours_seen.json:1`.

| Measurement | Reproduction-directory footage | Arnold seen footage |
|---|---:|---:|
| Raw persistence, gap 1 | 23.79 ± 0.05 | 21.59 ± 0.06 |
| Raw persistence, gap 2 | 22.65 ± 0.04 | 20.20 ± 0.04 |
| Raw persistence, gap 4 | 21.23 ± 0.02 | 18.94 ± 0.03 |
| Raw persistence, gap 8 | 20.25 ± 0.02 | 18.12 ± 0.03 |
| Near-static share, gap-1 PSNR above 35 dB | 0.81% | 1.47% |
| Stock reconstruction PSNR | 24.375 ± 0.004 | 23.643 ± 0.007 |
| Stock reconstruction LPIPS | 0.0952 | 0.0974 |

The reproduction-directory footage is easier under persistence at every tested gap. This is a property of those recorded pixels, not a measurement of the private GameNGen dataset. The difference must not be inserted as an exact correction to GameNGen's score. The “headroom” obtained by subtracting these logarithmic scores is a descriptive gap, not a decomposition of independent error sources. [Q4 `TABLE.md:16–45`.]

There is an unresolved provenance conflict. Q4 identifies its directory with the public Stiegler dataset and describes PNG files transcoded from JPEG. The project continuity-data record instead describes a freshly recorded, lossless corpus with **1,000 episodes** and about **4.95M tics**. The public dataset name itself contains **500 episodes**. The existing JSON records the server path, not enough immutable provenance to decide which description is correct. PNG magic bytes establish the container format, not whether its pixels were previously JPEG compressed. The floor measurements remain measurements of the local directory; their attribution requires raw metadata and hashes. [Q4 `TABLE.md:7–8,73–78`; `REQUIREMENTS.md:27`; RC 2026-09-09–10 continuity-corpus entries.]

The action-run audit also shows why our old global decision grid was unsafe. In the reproduction-directory sample, **44,616 of 44,616** action-run starts fell on the global four-tic phase. In the Arnold sample, only **11,427 of 30,841** did. Deaths and anti-stuck behavior move Arnold's phase. A global modulo filter is not a substitute for transition verification. [Q4 `TABLE.md:47–71`.]

## 3.9 Inference context-noise sweep

**Protocol:** Q1 seen subset, **128 rollouts×64 decision frames**, seed **0**, **50-step DDIM**, tuned C4 decoder, decoded targets. U-Net live checkpoint **87k**; PixArt live checkpoint **89k**. The subset and historical random-number batching differ from the full **256**-rollout result, so the baseline is an approximate reproduction of that result rather than an identical rerun. Sources: Q1 `q1-infer-noise/table.md:1–27` and corresponding `provenance.json`.

| Row | Inference q | PSNR h1 / h8 / h16 / h32 / h64 | LPIPS h64 | IDM | FVD16 / FVD32 |
|---|---:|---|---:|---:|---:|
| PixArt | 0 | 22.06 / 19.42 / 18.44 / 18.21 / 17.12 | 0.591 | 0.493 | 317.4 / 662.0 |
| PixArt | 0.035 | 22.07 / 19.30 / 18.27 / 18.03 / 16.96 | 0.578 | 0.492 | — |
| PixArt | 0.05 | 22.05 / 19.23 / 18.39 / 18.11 / 17.03 | 0.587 | 0.494 | — |
| PixArt | 0.1 | 22.08 / 19.37 / 18.30 / 18.10 / 17.14 | 0.585 | 0.490 | — |
| PixArt | 0.2 | 22.00 / 19.30 / 18.46 / 17.97 / 17.15 | 0.591 | 0.485 | 351.0 / 713.9 |
| PixArt | 0.3 | 21.90 / 19.40 / 18.31 / 17.98 / 17.07 | 0.594 | 0.464 | — |
| U-Net | 0 | 22.06 / 19.37 / 18.71 / 17.76 / 16.09 | 0.580 | 0.511 | 310.6 / 523.1 |
| U-Net | 0.035 | 22.26 / 19.54 / 18.19 / 17.04 / 15.64 | 0.570 | 0.496 | — |
| U-Net | 0.05 | 22.25 / 19.40 / 18.19 / 17.20 / 15.82 | 0.569 | 0.495 | — |
| U-Net | 0.1 | 22.21 / 19.35 / 18.16 / 17.13 / 15.49 | 0.590 | 0.492 | — |
| U-Net | 0.2 | 22.15 / 19.11 / 18.14 / 17.42 / 15.89 | 0.568 | 0.498 | — |
| U-Net | 0.3 | 22.06 / 19.08 / 18.30 / 17.44 / 15.79 | 0.574 | 0.487 | — |
| Copy seed | N/A | 20.11 / 18.27 / 18.31 / 18.31 / 17.82 | — | N/A | N/A |

The sweep did not rescue long-horizon fidelity. It supports keeping clean inference context for these checkpoints. The log's description of IDM as “monotonically” falling is too strong: both rows have small upward fluctuations. FVD worsens in the measured PixArt comparison, but most noise levels have no FVD measurement. The table supports “no demonstrated benefit,” not a complete monotonic response law. [RC 2026-09-20 08:30; Q1 table above.]

**Teacher-forced companion protocol:** paired **512 windows** per seen/unseen corpus, same checkpoints and tuned C4 decoder, **50-step DDIM**, raw target. Sources: `results/night_2026-09-19/q1-tf-noise/table.md:1` and per-cell metrics.

| Row / corpus | q=0 PSNR / LPIPS | q=0.035 PSNR / LPIPS | q=0.07 PSNR / LPIPS |
|---|---:|---:|---:|
| PixArt seen | 21.06 / 0.2801 | 21.12 / 0.2780 | 21.16 / 0.2790 |
| PixArt unseen | 19.42 / 0.4305 | 19.41 / 0.4362 | 19.46 / 0.4397 |
| U-Net seen | 21.12 / 0.2776 | 21.10 / 0.2782 | 21.12 / 0.2793 |
| U-Net unseen | 19.18 / 0.4449 | 19.16 / 0.4480 | 19.21 / 0.4511 |

The largest seen PixArt PSNR gain was **0.10 dB**, with paired standard error about **0.06 dB**. Its unseen LPIPS worsened at both nonzero levels. The companion table does not supply an episode-level multiple-comparison analysis. These are diagnostics, not a license to choose a test-optimal noise level. [Q1 `q1-tf-noise/table.md`; `paired_table.py`.]

## 3.10 DDIM step sweep and SD 3.5 budget curve

**Step-sweep protocol:** every step count used uniform-in-t timestep spacing over the trained linear betas, which is `--timestep-spacing linear` in `eval_tf.py` (the only spacing that existed then; `trailing` and `karras` were added on 2026-09-22 for the M1 spacing sweep, and none of the numbers here used them). U-Net and PixArt, historical reporting windows, **512 windows** per evaluated setting; historical live checkpoints and tuned C4 decoder, raw TF references. Exact per-cell checkpoint/window manifests and numerical rows are not locally recovered. The surviving summary compares **1/2/4** denoising steps with **50**. Source: RC 2026-09-22 09:15; `results/levers_2026-09-20/` contains plotting/table builders rather than a complete numeric sweep archive.

| Surviving statement | Status |
|---|---|
| Low-step PSNR increases by about 1.15 dB relative to 50 steps | Approximate summary, not a recovered per-step matrix |
| Low-step LPIPS is about 0.45–0.58 versus about 0.28 | Approximate summary showing a large perceptual regression |
| Exact result for each backbone at each step count | Recovered from the server on 2026-09-22 (below) |
| Exact common-window comparison against GameNGen | Unavailable because its original benchmark is private |

**Recovered matrix** (read from `results_spiderman/levers_2026-09-20/e1-steps/*/metrics.json` on Spiderman, 2026-09-22 08:40 EDT; 512 windows, seed 0, live weights, tuned C4 decoder, raw targets; PSNR / LPIPS):

| Steps | U-Net seen | U-Net unseen | PixArt seen | PixArt unseen |
|---:|---|---|---|---|
| 1 | 22.26 / 0.580 | 20.67 / 0.707 | 22.26 / 0.583 | 20.74 / 0.706 |
| 2 | 22.28 / 0.569 | 20.68 / 0.697 | 22.28 / 0.577 | 20.75 / 0.701 |
| 4 | 22.26 / 0.447 | 20.55 / 0.612 | 22.28 / 0.462 | 20.72 / 0.616 |
| 8 | 21.90 / 0.324 | 20.10 / 0.503 | 21.89 / 0.331 | 20.36 / 0.503 |
| 16 | 21.54 / 0.284 | 19.71 / 0.456 | 21.51 / 0.288 | 19.92 / 0.449 |
| 50 | 21.12 / 0.278 | 19.18 / 0.445 | 21.06 / 0.280 | 19.42 / 0.430 |

The 50-step rows reproduce the stored 512-window numbers of these rows. A few-step sample lies near the conditional mean of the next frame, which is blurred: that wins PSNR and loses LPIPS. GameNGen's step table (Table 3 of arXiv v2 in the review's reading; reproduced under "Sampling table protocol" in section 5.1) sweeps its ordinary sampler: 25.47 dB / 0.255 LPIPS at 1 step, 32.58 / 0.198 at 4, and flat after that (32.19 / 0.197 at 64); the distilled single-step model (31.10 / 0.208) is a separate row, not the sweep. Its LPIPS improves from 1 to 4 steps, and ours moves the same way over the same range (U-Net seen 0.580 to 0.447, PixArt seen 0.583 to 0.462, table above), so the LPIPS direction is not where the two differ. The real difference is that our PSNR is flat from 1 to 4 steps and then falls, where GameNGen's rises by 7 dB and then holds, and our LPIPS keeps improving all the way to 50 steps, where GameNGen's is flat after 4. Only the step count matches GameNGen's 4-step headline setting: GameNGen samples with DDIM and observation guidance 1.5 (§3.3.1), and our 4 steps are spaced uniformly in t over a linear beta schedule that puts two of the four network calls at essentially pure noise (`docs/REVIEW_2026-09-22.md` M1). Our 4-step column is therefore a sampler-specific point, not a like-for-like reproduction of GameNGen's setting, until the spacing sweep in M1 is run.

This result changed the reporting rule: every PSNR needs its denoising-step count and a perceptual metric. A blurred prediction can win MSE while losing visual fidelity. The correct next action is to recover the sweep artifacts and choose a sampler on validation, not to promote the highest-PSNR setting. [RC 2026-09-22 09:15.]

**SD 3.5 budget protocol:** **512 TF windows** per seen/unseen corpus, **50-step DDIM**, tuned C16 decoder, raw targets, live numeric-step snapshots. Copy references are **19.01 dB** seen and **18.32 dB** unseen for this window set. Source: RC 2026-09-20 21:45 and preceding budget entries.

| Training update | Seen PSNR | Unseen PSNR | LPIPS where recorded |
|---:|---:|---:|---|
| 40k | 20.14 | 18.45 | Seen 0.308; unseen 0.430 |
| 50k | 20.69 | 19.13 | — |
| 60k | 20.85 | 19.25 | — |
| 70k | 21.14 | 19.45 | — |

The final **90k** result in §3.4 uses **2,048 windows**, so it is not a paired endpoint for this curve. The intermediate curve established that this large row was still improving late in the allotted training. It motivated examining compute and exposure before dismissing its warm start. It did not prove that continued training would beat another model at equal card-hours. [RC 2026-09-20 21:45, 2026-09-22 09:15.]

## 3.11 Other completed checks that constrain interpretation

**Short adaptation pilot:** corrected-data seen reporting set, **2,048 TF windows**, **50-step DDIM**, tuned C4 decoder, live **2,500-update** models, batch **32**, warmup **250**. Sources: RC 2026-09-16 21:40.

| Start | PSNR / LPIPS | Validation loss |
|---|---:|---:|
| ImageNet DiT | 19.20 / 0.471 | 0.2605 |
| SD U-Net | 19.67 / 0.411 | 0.2438 |
| PixArt | 19.89 / 0.412 | 0.2462 |

These models each saw **80,000 presentations**, derived, but did not consume equal compute. The early advantage of the image-text starts was consistent with the final comparison. It did not establish a causal pretraining-data explanation. [Same pilot source.]

**Corrected context-loss pilot:** validation velocity loss on the trainer's fixed windows, no decoder or diffusion sampling; DiT checkpoints at **5k updates**. Source: RC 2026-09-14 20:30.

| Context frames | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| Corrected-label validation loss | 0.2631 | 0.2561 | 0.2532 | 0.2565 | 0.2535 |
| Earlier noisy-label run | 0.2602 | 0.2583 | 0.2532 | 0.2525 | 0.2519 |

The corrected pilot was not monotonic beyond the shorter contexts. It weakened the claim that merely extending channel-concatenated history would solve the DiT gap. Its short budget and loss-only evaluation leave longer-run perceptual effects open. [RC 2026-09-14 20:30.]

A video-pretrained SkyReels/Wan-family pilot was also run. It used a different temporally compressed VAE, native flow objective, context **8**, **10k updates**, and global batch **8**. Its **512-window**, **50-step native-flow** seen score was **20.11/0.305**, versus decoded-copy **18.85** and reconstruction **24.83/0.122**; unseen was **18.80/0.328**, versus copy **17.75** and reconstruction **23.08/0.144**. The recorded final validation loss was **0.1734**, with best **0.1731** at **9.5k**. These are a separate system pilot, not another row in the same-latent DDIM table; the native-flow sampler must not be mislabeled DDIM. Autoregressive evaluation was not completed in the reviewed record. [RC 2026-09-17 09:45, 2026-09-18 00:30; `A/lit/backbone-candidates-video.md`; `A/astra-nexttic-2026-09-20.md`.]

The April checkpoint remains a historical reproduction artifact. Its reported **26.04 dB / 0.153 LPIPS** came from training segments, and its released weights saw all **500 episodes**. An older report also uses **25.81/0.15**, while the old U-Net comparison appears as **24.90/0.19** or **24.60/0.198** in different project materials. The U-Net code and weights were lost. These figures cannot become honest held-out comparisons through relabeling. [RC 2026-09-02 and sections 2–3; `A/DOOMDIT_AGENT_CONTEXT.md`; `A/doomdit-full-recap-2026-07.md`.]

A later April-checkpoint continuity test used **12 episodes**, **512 windows**, **160×120** rendering, an approximately **87.2k-update** checkpoint, and **50-step respaced ancestral DDPM**, scored against decoded targets. It obtained **27.43 ± 0.15 dB / 0.117 ± 0.003 LPIPS**, versus copy **26.24/0.141**. This validates a narrow legacy evaluation path. It is not comparable to the rebuilt raw-frame benchmark or GameNGen's reported resolution. [RC 2026-09-09 05:00.]

**Encoder equivalence check:** five episodes, **24,558 raw rows**, **6,158 decision frames**, **48 chains**, and **6,110 transitions**; no model sampling or decoder evaluation. Metadata matched, but the stored latents were not bit-identical when batching changed. At encode batch **16**, mean absolute difference was **1.4528e-5**, maximum **0.31348**, exact-element agreement **99.6166%**. At batch **64**, those values were **0.0013632**, **0.56046**, and **60.3233%**. Both strict reports set `ok: false` against tolerance **0.0009765625**. Source: Q3 `q3a_equivalence_b16.json:1`, `q3a_equivalence_b64.json:1`, `q3a_diff_distribution.json:1`.

This distinction matters: metadata equivalence passed, while the strict numerical gate did not. “Close on average” is not “byte-identical.” The later producer-prefetch byte-identity claim concerns a different controlled implementation comparison and does not erase this batching result. Encoder dtype, batch size and library version belong in corpus provenance. [Q3 reports; RC 2026-09-22 09:45.]

**Deathmatch-simple engine probe:** same seeded **60-game-second** episode configuration, no world model or decoder. Fixed keyword injection produced **1,746 rows**, **9 deaths**, **2 frags**, health **5–100**, ammo **21–50**, and **43 ammo changes**. The old partial-default injection produced **2,097 rows**, no deaths or frags, and constant health **100** and ammo **50**. `kills` remained zero in both because monster kills and deathmatch frags are different engine counters. Source: `release/DENSE_CORPUS.md:58–75`; Q5 `probe_episodes.json:1`.

The probe resolved an environment-configuration bug. It did not complete a deathmatch-simple benchmark corpus. That segment remained explicitly gated pending longer checks and a compute/storage decision. [Same source.]

# 4. Decisions, evidence, and rejected alternatives

## 4.1 Decision ledger

The dates below identify decisions or observations in RC. An earlier proposal is not retroactively an adopted design. The latest explicit decision takes precedence over stale requirements and the older planning sections.

| Date | Decision or change | Evidence | Alternative rejected or deferred |
|---|---|---|---|
| 2026-09-02 | Rebuild an honest held-out experiment and evaluation pipeline. | April headline used training segments; old U-Net artifacts unavailable. | Presenting the class-project comparison as a new held-out result. [RC 2026-09-02; sections 2–3.] |
| 2026-09-08–09 | Use Arnold-generated lossless footage and retain every recorded tic. | Public pretrained agent, multi-arena coverage, inspectable controls and engine metadata. | Depending on inaccessible original GameNGen footage or treating the reproduction's policy as the same agent. [RC 2026-09-09; `release/DENSE_CORPUS.md:19–42`.] |
| 2026-09-09–10 | Verify actual control intervals before using decision-frame windows. | Death/respawn and anti-stuck behavior break a global modulo grid. | `tic % 4 == 0` as the definition of a decision. [RC 2026-09-09–10; `transitions.py:163` onward.] |
| 2026-09-13 22:16 | Shared corrected recipe: LR 5e-5, warmup 2,000, batch 32, context 32, action dropout zero. | Real-data fit checks, corrected corpus, reviewed adaptation diagnostics. | Backbone-specific LR multipliers without a controlled reason. [RC entry.] |
| 2026-09-14 20:00 | Reframe the initial pair as a shared adaptation study; add PixArt and another DiT seed. | U-Net's persistent loss/perceptual advantage; differences in pretrained priors. | A predetermined claim that DiT wins; training a much larger from-scratch model as a quick rescue. [RC entry.] |
| 2026-09-16 14:30 | Audit persistence and motion before interpreting rollout PSNR. | DiT's higher long-horizon PSNR coexists with weaker motion, FVD and action metrics. | Calling higher PSNR “more stable dynamics” without a motion check. [RC entry.] |
| 2026-09-17 00:20 and 06:20 | Rescue UniDiffuser with logged rollback, reduced LR and a spike guard. | Gradient excursions, validation degradation and collapsed attention-update ratios. | Silent reseeding, hiding failed compute, or claiming an unchanged common recipe. [RC entries.] |
| 2026-09-17–18 | Gate a C16 system row on decoder reconstruction before spending training compute. | Paired SD 3.5 decoder gains on all evaluated corpora. | Assuming a new VAE helps because of a generic benchmark, or attributing its benefit to the backbone. [RC 2026-09-17 11:45; 2026-09-18 14:50.] |
| 2026-09-19 | Record dense coverage on Arnold's published training arenas 2–5 and test arenas 6–8. | Prior split fixed independently of our model scores. | Arenas 3, 10, 12, 13 chosen from favorable per-map outcomes; 134 aborted episodes unused. [`release/DENSE_CORPUS.md:9–17`.] |
| 2026-09-20 21:45 | Move the next experiment to every tic. | Measured gap-dependent persistence and incomplete comparability with GameNGen. | Continuing to treat stride-four PSNR as directly comparable with a likely denser original task. [RC entry.] |
| 2026-09-21 01:30 | Start new U-Net and SD 3.5 runs from public pretrained weights. | Clean separation between the revised recipe and prior in-domain adaptation. | Warm-starting from completed Doom checkpoints, which changes exposure and initialization history. [RC entry.] |
| 2026-09-21 01:30 | Use context 32, executed-control history, and the fixed dense episode split. | Causal row audit; GameNGen history-token mechanism; bounded memory and compute. | Latest-action-only labels, using the target row's outgoing action, or crossing life boundaries. [RC entry.] |
| 2026-09-21 14:30 | Keep the shared velocity schedule and context augmentation. | Existing rows use it; no schedule ablation; inference-noise sweep unsuccessful; original stabilization evidence. | Late unmeasured schedule changes or the proposed exactly-clean training atom. [RC entry.] |
| 2026-09-22 09:45 | Normalize raw requested control strings to the engine's executed width. | Arnold list-growth mechanism and ViZDoom source prove truncation/zero-fill semantics. | Treating raw width as model width or remapping an unexecuted overflow weapon bit into an executed slot. [RC entry; §4.3.] |
| 2026-09-22 11:30 | Apply reviewed sidecar repair and preserve request diagnostics. | Repaired encoded sidecars, raw-to-sidecar checks, fixed-width loader checks. | Re-encoding unchanged images or modifying the raw parquet to hide the original request. [RC entry.] |
| 2026-09-22 09:45 | Encode in a short H100 burst, then train on a smaller rental. | Measured encoder bottleneck and provider/storage analysis. | Earlier plan to hold the full node for several days. Forecast only until rented and measured. [RC entry; `A/h100-providers-2026-09-22.md`.] |

## 4.2 Arnold, dense coverage, and the split

Arnold was chosen because a released agent and inspectable engine interface make the data pipeline reproducible. That choice introduced a different behavior distribution from GameNGen's learning PPO agent. GameNGen includes weak and strong agent behavior over training; Arnold recordings primarily reflect a fixed policy, scripted overrides and the engine behavior actually reached. Diversity of maps does not substitute for diversity of policy competence. [G §3.1; RC 2026-09-09; `record_arnold.py:1–426`.]

The original corpus spread a limited amount of footage across many arenas. The dense corpus increases repeated coverage of a fixed set. The completed `arenas` pool contains **40,285,059 raw tics** across **8,000 episodes**, with **2,000 episodes per map**. At **35 Hz**, this is approximately **319.72 recorded hours**, derived from stored tics rather than nominal timeout. The unseen pool contains about **14.8M tics**, or about **117.46 recorded hours**, an estimate from the rounded count. Neither figure is the number of valid context-target examples. [Q2 `verify_arenas.json:1`; RC 2026-09-20 08:30, 21:45; `release/DATASET_CARD.md:20,25–30`.]

The nominal episode duration is **150 game-seconds**, but stored rows are reduced by unrecorded respawn intervals. The dense verification analysis fitted about **39 missing tics per death**, with a baseline near **5,286 rows**. A minimum-length threshold therefore acts partly as a death-count filter. Lowering that threshold from **4,800** to **4,600** avoided rejecting otherwise valid difficult episodes simply because the agent died more often. The reported fit does not imply that arbitrary shorter recordings are valid. [`release/DENSE_CORPUS.md:85–94`.]

The fixed split is by episode, with half-open ranges. The full pool is distinct from the initially approved run prefix. [`release/dense_split.json:1`; RC 2026-09-21 01:30.]

| Purpose | Segment and IDs | Count and map balance | Status |
|---|---|---|---|
| Full training pool | `arenas` 0:6000 | 6,000 episodes, 1,500 per arena | Frozen pool |
| Full validation pool | `arenas` 6000:7000 | 1,000 episodes, 250 per arena | Frozen pool |
| Full seen-test pool | `arenas` 7000:8000 | 1,000 episodes, 250 per arena | Frozen pool |
| Approved initial training prefix | `arenas` 0:2000 | 2,000 episodes, 500 per arena | Same prefix for each next-tic backbone |
| Initial validation subset | `arenas` 6000:6100 | 100 episodes, 25 per arena | Launch/evaluation subset |
| Initial seen-test subset | `arenas` 7000:7100 | 100 episodes, 25 per arena | Reporting subset |
| Unseen scoring subset | `arenas_678` 60:120 | 60 episodes, 20 per arena, zero worker-first episodes | Reporting subset since 2026-09-22 |
| Withdrawn unseen subset | `arenas_678` 0:60 | 51 of 60 worker-first (k = 0) | Declared 2026-09-21, replaced 2026-09-22 before any score; secondary set only, reported with its k strata |
| Full unseen pool | `arenas_678` all | 3,000 episodes, 1,000 per arena | Never trained on; JSON operational `unseen` range names only the scoring subset |

**Why the unseen subset moved (2026-09-22).** Arnold's weapon-select requests execute only in a recorder process's first episode (k = 0; section 4.3). The review measured k from the raw `buttons` column of ids 0:60 and found 51 worker-first episodes (ids 0 to 24 and 26 to 51), because `arenas_678` was started three times and every restart made a fresh k = 0 episode per worker. Validation and test hold none and the training prefix only 32 (1.6%), so scoring on 0:60 would have mixed map transfer with a control regime the model barely saw. Ids 60:120 hold no k = 0 episode. No model had been scored on either set, so the replacement is not outcome selection. [`release/dense_split.json` `history`; `docs/REVIEW_2026-09-22.md` H1.]

The initial prefix has approximately **10.07M raw tics**. Dividing by global batch **32** gives roughly **315k optimizer updates** for a nominal pass, before valid-window exclusions and repeated sampling are accounted for. The statement in the log that more data cannot help below a pass is too categorical. Additional data can change diversity even when not all examples are consumed. The narrower defensible rationale was storage, encoding cost, and matching the data distribution between rows under a fixed budget. [RC 2026-09-21 01:30.]

The full split can remain fixed while the training prefix expands. Such an expansion changes the experiment and must be logged. The later mention of using all **6,000 training episodes** on H100s was an option, not a replacement of the approved **2,000-episode** prefix at the source cutoff. [RC 2026-09-22 09:15; `release/dense_split.json:33–46`.]

The rejected score-selected maps matter methodologically. Choosing training arenas because preliminary model scores look favorable makes dataset construction depend on the outcome. Using Arnold's published train/test map split removes that particular selection mechanism. It does not make the benchmark representative of every DOOM level. [RC 2026-09-19; `release/DENSE_CORPUS.md:15–17`.]

## 4.3 The executed-control finding, from mechanism to repair

The raw `action` field is Arnold's requested discrete policy action. The raw `buttons` field is the requested list passed to the engine. Neither name alone establishes what the engine executes. Anti-stuck overrides can make the list differ from the canonical action ID. An additional Arnold bookkeeping defect makes the list wider than the engine's button set. [`record_arnold.py:255–273`; `transitions.py:25–38`; `release/DATASET_CARD.md:53–108`.]

The base action builder exposes **nine controls**: forward, backward, turn left, turn right, move left, move right, attack, speed and crouch. Arnold's `add_buttons` appends **ten weapon-selection names** to the same shared list on every game start. It then builds a name-to-index dictionary, so later duplicate names overwrite earlier indices. The relevant source is [Arnold `actions.py:197–199`](https://github.com/glample/Arnold/blob/86af06d2fdb35c4bf552ecacfe8fe6ac1abd8cd4/src/doom/actions.py#L197), [`actions.py:298`](https://github.com/glample/Arnold/blob/86af06d2fdb35c4bf552ecacfe8fe6ac1abd8cd4/src/doom/actions.py#L298), and [`game.py:485`](https://github.com/glample/Arnold/blob/86af06d2fdb35c4bf552ecacfe8fe6ac1abd8cd4/src/doom/game.py#L485). The recorder starts a game for each recorded episode. [`record_arnold.py:167`; `release/DATASET_CARD.md:98–108`.]

ViZDoom deduplicates available buttons when they are registered. Its engine-side vector therefore remains **19 entries** wide. In ViZDoom **1.2.4**, [`ViZDoomGame.cpp:367–372`](https://github.com/Farama-Foundation/ViZDoom/blob/1.2.4/src/lib/ViZDoomGame.cpp#L367) rejects duplicate button registration. [`ViZDoomGame.cpp:151–158`](https://github.com/Farama-Foundation/ViZDoom/blob/1.2.4/src/lib/ViZDoomGame.cpp#L151) iterates only over the engine's available buttons, reads a supplied entry when present, and zero-fills a missing entry. It never reads the excess tail. These source lines establish the control semantics independently of any statistical correlation. [Same upstream sources; `transitions.py:28–32`.]

For a nonempty binary request string, the executed vector is therefore:

```python
executed = requested[:19].ljust(19, "0")
```

The constant and operation are implemented in `transitions.py:32,54–89`. A request bit beyond the executed range must be discarded, not wrapped or remapped. Remapping it would claim the agent executed a weapon switch that the engine ignored. A shorter string is padded because the engine zero-fills missing controls. An empty string is refused because the recorder always supplies a base control list; treating missing data as a genuine no-op would hide corruption. [`transitions.py:54–79`.]

The weapon index formula is **9 + 10k + j**, where (j) is the weapon slot and (k) is the number of *previous* starts in that worker. Thus (k=0) denotes the first recorded episode, after its first start has already occurred. The earlier log's “after k starts” wording was off by this indexing convention; the repaired helper documents it explicitly. Only the first worker episode can place these requested weapon bits inside the engine's range. [`transitions.py:41–51`; `release/DATASET_CARD.md:98–108`; RC 2026-09-22 09:45.]

This is a behavior artifact of the recording agent. In later worker episodes, its explicit weapon-selection requests are ignored. Movement and firing controls still execute. Weapon changes caused by pickups can still occur. The dataset remains a record of what the engine displayed under the controls it actually consumed. It is not a record of the behavior Arnold would have produced with that bookkeeping bug fixed. [`release/DATASET_CARD.md:103–108`.]

The repair preserves the distinction. Encoded sidecars store fixed-width executed strings and also retain `buttons_raw_len` and `switch_requested_index`. The raw parquet remains unchanged. The repair compares row counts, tics and normalized controls against raw metadata, writes atomically, and does not re-encode images. Wide old sidecars are refused with a repair instruction rather than silently loaded into a huge control matrix. [`transitions.py:82–108`; `doom_data.py:562–588`; RC 2026-09-22 11:30.]

The initial repaired sample covered **40 raw episodes**. It found maximum raw width **27**, **3.9%** of rows wider than the executed vector, **3,286 executed switch rows**, **7,818 unexecuted switch rows**, and no inferred within-episode start-count growth. Earlier broader observations included strings up to **2,506 characters**. These summaries cover different portions of the corpus and should not be treated as contradictory maxima. The full-corpus report finished at 13:00 EDT: over all 8,000 episodes (40,285,059 rows) the raw maximum width is **2,507**, **11.0%** of rows are wider than the executed vector, executed switch rows number **3,286** (all inside the first episode of each of the 32 recorder workers, ids 0 to 31), unexecuted switch requests **4,424,593**, no episode shows within-episode growth, and the maximum inferred prior-start count is **249** (8,000 episodes over 32 workers), which is exactly what the mechanism predicts. [RC 2026-09-22 09:45, 11:30, 13:00.]

“Rows wider than the engine” and “rows with an ignored switch” are different statistics. An anti-stuck row can have an overlong all-zero tail. A short row can contain a real executed switch. The repaired report keeps these counts separate. Keeping only the first **nine** controls would also be wrong for the new model: it would discard actual weapon-selection inputs in the episodes where they were executed. The old canonical action table continues to use those base controls for its historical purpose; it is not the new conditioning representation. [`transitions.py:111–149`; `release/DENSE_CORPUS.md:34–38`.]

## 4.4 Every tic and the causal history contract

The new dataset validity rule is consecutive tics, unchanged death counter, and unchanged map. It does not require a verified decision chain. That is deliberate: next-tic prediction must retain transitions during held actions and anti-stuck execution rather than reject them because they do not match the requested scalar action. [`doom_data.py:465–514,591–746`.]

For target row (r), the approved history is `latents[r-32:r]` and `buttons[r-32:r]`. The target is `latents[r]`. The final control in that slice produces the target. This is the opposite row convention from the open reproduction's post-step stored frame. A shifted action sequence can look plausible because controls persist across multiple tics, so an alignment test needs turning transitions where the shift is identifiable. [RC 2026-09-21 01:30; `record_arnold.py:264–273`; `doom_data.py:725–746`.]

The alignment gate compares candidate shifts on the same row set and uses yaw changes to test directionality. An inconclusive result is not a pass. The launch review found an earlier test that accepted an injected off-by-one error, so the gate was tightened. The real-data audit must also compare normalized sidecar controls with normalized raw controls. A successful synthetic test cannot establish correct alignment in the production corpus. [RC 2026-09-21 01:30, 2026-09-22 11:30.]

GameNGen-style action history means adopting a token for each relevant control interval. It does not imply matching its unknown action vocabulary or positional encoding. Our U-Net history width is **768**; PixArt and SD 3.5 token widths are **4,096**. SD 3.5 also uses a **2,048-dimensional** pooled conditioning slot derived from the newest control. These widths come from the public backbone interfaces. [`backbones.py:153–217,344–388,390–497,726–790`.]

The generic DiT history path averages embeddings before adaLN. With a separable additive positional embedding, that average does not retain action ordering: averaging the fixed position terms adds a constant independent of which control occupied which position. The path is available for compatibility and fit checks, but it is not evidence that DiT has the same order-sensitive history interface as the selected cross-attention rows. [`backbones.py:153–181,290–307`; `A/astra-prelaunch-audit-2026-09-21.md`.]

## 4.5 Public initialization, context, batch, and schedule

Public initialization was chosen to keep the revised experiment's starting exposure auditable. Initializing from our own finished world models would add in-domain training and inherit the old stride and action representation. It might be a useful future efficiency experiment, but it is a separate treatment. [RC 2026-09-21 01:30.]

Context **32** was retained as a bounded compute choice. GameNGen's frozen-decoder context ablation changes PSNR only from **22.31** at context **32** to **22.36** at context **64** under its ablation protocol. That supports testing a smaller context, but does not prove the missing memory is irrelevant to long rollouts. The new physical span is shorter than the old one, so revisit and state-memory failures remain open. [G Table 2; RC 2026-09-21 01:30.]

Global batch remains **32**. On a single device this is a micro-batch of **32** if it fits. On a distributed pair it is **16 per device**. Gradient accumulation is not the default means of filling the global batch. Checkpointing is enabled only where memory requires it. Fused AdamW and bf16 autocast are retained, while parameters, gradients, optimizer state and EMA arithmetic are handled in fp32 where specified by the trainer. [`scripts/spiderman/launch_nexttic.sh:95–133`; `train_wm.py:577–890`; RC 2026-09-22 09:45.]

The schedule stays VP velocity with **1,000** linear betas from **1e-4** to **0.02**. SD 1.4's native schedule instead squares a linear interpolation between square roots of **0.00085** and **0.012**. The audit calculates terminal cumulative alpha of about **4.0e-5** for ours versus **4.7e-3** for the SD schedule, and identifies **273** of our training timesteps below the latter's terminal signal level. These are schedule calculations, not an observed quality improvement. [RC 2026-09-21 14:30; `diffusion_v.py:38–47`; `A/astra-prelaunch-audit-2026-09-21.md`.]

SD 3.5's pretrained rectified-flow path differs more fundamentally. Matching a signal-to-noise ratio does not make its interpolation coefficients or velocity target equal to the VP definitions. A timestep remapping alone is not a proven conversion. The decision to keep the shared VP recipe accepts this mismatch and requires the paper to disclose it. No schedule ablation was completed. [RC 2026-09-21 14:30; `backbones.py:675–724`; `diffusion_v.py:59–87`.]

The schedule is hard-coded rather than stored as a complete checkpoint-owned configuration. The audit counted **eight call sites**. Changing it after training would silently change the model's interpretation unless every consumer respected the original schedule. The practical deferral is to leave it fixed now and treat any future schedule comparison as a new training experiment. [RC 2026-09-21 14:30.]

## 4.6 Augmentation retained; clean-bucket proposal withdrawn

The approved context corruption samples a continuous level up to **0.7** and encodes it in **ten buckets**. Bucket zero covers the lowest interval, approximately **[0, 0.07)**. Exactly clean context is a boundary of that interval, not a positive-probability training atom. The noise tensor is sampled elementwise, while the level is shared across the context sample. [`diffusion_v.py:135–151`.]

The proposed change was to assign **10%** probability to exactly clean context. The review withdrew it because it would change the training distribution and bucket semantics relative to the finished recipe, while not being a faithful replication of a disclosed GameNGen detail. The code's current inference use of clean context and bucket zero is therefore retained with an acknowledged boundary mismatch. [RC 2026-09-21 14:30.]

GameNGen's no-augmentation experiment demonstrates rapid autoregressive degradation in its setting. Our noaug cell demonstrates a local teacher-forced and IDM benefit, with a FVD cost. These observations can both be true. The decision to keep augmentation rests on uncertainty about longer rollouts and transfer to the denser task, not on ignoring the local ablation. [G §3.2.1, §5.2.2, Figure 7; RC 2026-09-20 08:30.]

Diffusion Forcing and Self Forcing were deferred. They are changes to the training problem and architecture, not names for our existing bucketed context noise. A separate experiment should isolate one of them after the next-tic baseline is measured. [G §6; DF §3; SF §3; RC 2026-09-20 21:45.]

## 4.7 Deviations within the completed “shared” recipe

UniDiffuser did not finish under an unchanged common optimizer schedule. Its first excursion occurred near **8.5k updates**. The run restored the **5k** checkpoint and reduced LR from **5e-5** to **2.5e-5**. A later excursion near **34.6k** led to restoration from **30k**, LR **1e-5**, and a pre-clip gradient-norm skip threshold of **5**. The final segment recorded **10 skipped updates** over its last **60k** updates. Failed and rolled-back compute belongs in its cost accounting. [RC 2026-09-17 00:20, 06:20; 2026-09-18 07:30.]

The attention-update collapse motivated an attention-saturation hypothesis. It did not establish that mechanism causally. The lowest timestep quartile was the lowest-noise quartile; an earlier interpretation reversed it. Gradient clipping bounds the supplied gradient norm, but does not directly bound Adam's preconditioned parameter displacement. These distinctions matter when discussing why the rescue worked. [RC 2026-09-17 06:20.]

SD 3.5 also carries a spike guard and required gradient checkpointing. Its initial guard threshold was applied too early and skipped valid large adaptation gradients, preventing progress. The revised launcher activates the threshold only after **3,000 updates**. A two-device fit attempt with per-device batch **16** ran out of memory at the first optimizer step, so the finished row continued on one card. This is why a forward/backward memory probe is not enough to certify a distributed optimizer configuration. [RC 2026-09-18 14:50, 16:35; `scripts/spiderman/launch_nexttic.sh:84–90`.]

## 4.8 Deferred choices and why they need not block training

| Deferred choice | Why it can follow dynamics training | Boundary that must not be crossed |
|---|---|---|
| Decoder objective and final tuned decoder | Encoder and latent normalization remain fixed; predicted latents are fed back directly. | Tune on training-map data only for a pipeline-wide unseen-map claim; identify the decoder in every result. [G §3.2.2; RC 2026-09-21 14:30.] |
| DDIM step count and stochasticity | These select an inference trajectory for the fixed learned objective. | Keep the trained schedule; select on validation; use paired windows/noise; do not add unsupported observation CFG. [`diffusion_v.py:104–132`; RC 2026-09-21 14:30.] |
| Final evaluation window count and physical horizons | They do not alter trained parameters. | Freeze protocol before inspecting test outcomes; reserve enough compute; distinguish raw and decoded references. [RC 2026-09-21 14:30.] |
| Live versus EMA reporting | Both can be saved during training. | Compare from the same checkpoint; do not choose each row's best test outcome independently. [RC 2026-09-21 14:30.] |
| Optional PixArt next-tic row | It uses the same split and declared recipe. | Do not make its addition depend on favorable test scores; measure its own compute. [RC 2026-09-21 01:30, 2026-09-22 09:45.] |

Deferring evaluation design is technically safe for optimization but statistically safe only while the test remains sealed. Existing historical corpora have already informed many decisions, so they are development evidence rather than untouched final tests for a new claim. The dense fixed test split provides a chance to make the next evaluation protocol prospective. [RC 2026-09-20–21; `release/dense_split.json:1`.]

The evaluator should compare equal physical horizons across stride changes. For example, **64** stride-four predictions and **256** next-tic predictions both cover **256 engine tics**, about **7.31 s**, derived. Running **256 rollouts × 256 predictions × 50 denoising steps** requires **3,276,800 denoiser evaluations** before decoder, FVD and judge costs, derived. This is why evaluation time is part of the compute plan. [RC 2026-09-21 14:30; `release/DATASET_CARD.md:20`; `diffusion_v.py:104–132`.]

## 4.9 Compute, storage, and launch meaning

The latest agreed rental shape is an encoding burst on **eight H100s for about one to two hours**, followed by **two H100s for 48 hours**, with an approximate **$650** total budget. These are forecasts, not invoices or completed compute. The earlier idea of keeping the full node for about **five days** was superseded in the later entry. [RC 2026-09-22 09:15, 09:45; `A/h100-providers-2026-09-22.md`.]

The measured SD VAE encoder ceiling was about **96 frames/s** on an A6000, with **99.7% GPU busy**. The record attributes the bottleneck to memory-bound GroupNorm and elementwise work. Producer prefetch reached about **97%** of that ceiling in its test. The proposed H100 improvement of about **4×**, and about **seven card-hours per latent space** for the initial training prefix, are forecasts based on memory bandwidth, not measured H100 results. [RC 2026-09-22 09:45.]

Raw upload throughput was measured at **123 MB/s**, with no gain from increasing concurrent streams in that test. The first approximately **620 GB** priority batch was recorded as uploaded at **10:01 EDT**. Later batches were still uploading at the cutoff. Planned completion times must not be rewritten as actual completion. [RC 2026-09-22 09:15, 11:30.]

The optional third row can use an A6000 NVLink pair, listed as device pairs **0+1** or **2+3**. “Free” in the planning note means no rental bill, not zero compute or unrestricted access to shared devices. Each row must preserve global batch **32**. If a row spans a pair, it uses **2×16**; the total rented device count does not imply that both primary rows each receive a pair simultaneously. The provider card and actual allocation must resolve scheduling. [RC 2026-09-22 09:45.]

Latent storage should be calculated from the representation. A C4 fp16 frame at **4×32×40** occupies **10,240 bytes**, derived. The full **40,285,059-tic** arenas pool would therefore require about **412.52 GB**, or **384.19 GiB**, of pure C4 tensor payload before headers and metadata. C16 multiplies that payload by **four**, derived. A raw-data byte total, a latent payload total and a filesystem allocation are different budgets. [`doom_data.py:191–193`; `encode_parquet.py:165–170`; Q2 `verify_arenas.json:1`.]

The launcher must demonstrate actual distributed execution. In the reviewed Spiderman script, the multi-device command supplies `--num_processes`, while the fit path invokes the Python trainer directly. Installed Accelerate configuration can affect the first command, and the second is not a distributed throughput test. A passing single-process fit cannot certify a multi-process training allocation. This is an operational check to perform, not a code change made for this dossier. [`scripts/spiderman/launch_nexttic.sh:105–110,138–146`.]

The approved launch sequence is a raw-control audit, alignment gate on validation, full optimizer fit check, short real-data smoke, checkpoint readback through the evaluator, then launch and an initial monitored period. The audit proposed **300 smoke updates**, recovery saves during that smoke, and a **30-minute** initial watch. Those gates protect the training contract; they do not guarantee the final research result. [RC 2026-09-21 14:30.]

Recovery checkpoints retain optimizer and EMA state, but current resumption is not bit-exact continuation of the original data stream. The code reseeds and reshuffles after resume. Corpus fingerprints sample latent-file regions and hash sidecar content rather than reading every latent byte. Both are useful safeguards, but neither should be described as full bitwise reproducibility of an interrupted trajectory. [`train_wm.py:284–381,577–731`; RC 2026-09-10 14:00.]

The outage explanation remains unresolved. One entry proposed memory pressure from repeated metadata scans; another used the host's installed RAM to reject that explanation. Installed capacity alone does not establish available memory at failure. The oversized Unicode control columns further complicate the earlier estimate. Root-disk pressure was another candidate. Console logs, kernel OOM records and process memory measurements are needed before assigning a cause. [RC 2026-09-21 14:30; 2026-09-22 09:15, 09:45.]

# 5. Reading guides

## 5.1 GameNGen: read the recipe and the evidence separately

### Abstract and introduction

The central claim is that a diffusion model can simulate a complex interactive game at playable speed while retaining visual quality. The paper is not primarily an architectural comparison. Its evidence combines teacher-forced fidelity, short distributional video metrics, human discrimination, and extended qualitative interaction. Our paper should not compress that evidence into its best-known PSNR. [G Abstract, §1, §5.]

The original setting is a neural simulator whose training observations come from an agent. Our setting adds a reproducibility question: which parts of that recipe remain useful with public starts, much smaller local compute, and fully specified data and evaluation? “Small compute” must be quantified from actual card-hours and exposure, not inherited from the class project's budget. [G §1; RC 2026-09-14 20:00, 2026-09-17 11:45.]

### Section 2: task formulation

Check which action produces the observation being predicted. The paper's notation around past/current actions is not a safe array-index specification for another recorder. Translate its conditional distribution into our pre-step row convention before comparing implementations. The simulator receives action inputs during autoregression, but it does not receive ground-truth observations after the seed context. Teacher-forced and autoregressive tests therefore answer different questions. [G §2; `record_arnold.py:264–273`; `rollout_eval.py:72–81`.]

The most useful comparison is conditional information. Does a model receive only pixels and controls, or also map geometry, pose, reward, state variables, or future observations? MultiGen's memory is additional information. Our stored positions and health are audit/evaluation metadata unless an explicit experiment feeds them to the model. Their presence in parquet is not evidence of conditioning leakage. [G §2; M §3; `doom_data.py:734–746`.]

### Section 3.1: collecting experience

GameNGen records the learning agent's trajectories rather than only the final expert's behavior. This creates a changing competence distribution. Read it against PlayGen's explicit balance of expert and random trajectories, and against our fixed Arnold policy. The right question is whether rare events, deaths, turns and poor decisions are represented, not just whether there are many frames. [G §3.1; P §3.2; `release/DATASET_CARD.md:18–21`.]

Do not substitute the diffusion dataset size for the PPO training budget. The paper states **50M environment steps** for the agent and a sampled **70M-example** diffusion training set. These are different counters. Earlier cards that imply a smaller PPO budget should defer to the paper. [G §§4.1–4.2.]

### Section 3.2: adapting latent diffusion

Read the input construction literally. Past latents enter through channel concatenation. Past actions enter through cross-attention embeddings. The model is not a native video transformer. Its pretrained image denoiser is repurposed for conditional next-frame generation. Our completed channel-stacked backbones share the visual-context idea, while the new runs also adopt the history-token action pathway. [G §3.2; `backbones.py:236–259,344–388`.]

The paper switches to velocity prediction and says the noise schedule is linear, referring to latent diffusion. It does not provide a reproducible beta array or endpoint pair. The phrase does not establish equality with our literal linear betas, and it does not license silently importing SD 1.4's scheduler configuration as a paper-stated fact. Record both the paper's statement and the unresolved implementation detail. [G §3.2; `diffusion_v.py:38–47`; RC 2026-09-21 14:30.]

### Section 3.2.1: noise augmentation

The mechanism is to train the next-frame predictor on corrupted observations so that imperfect self-generated context is less surprising at inference. The paper supplies a maximum noise level and discrete bucket conditioning. It does not spell out enough to establish that its corruption is our VP blend, the reproduction's additive blend, or another exact parameterization. [G §3.2.1; §4.2.]

The paper reports divergence without augmentation after approximately **10–20 generated frames**. Its controlled ablation uses **200k training steps**, with rollout evidence in Figure 7. Our noaug cell used a shorter budget and a different environment. Read both as evidence under their own conditions. Neither result should erase the other. [G §3.2.1, §5.2.2; RC 2026-09-20 08:30.]

### Section 3.2.2: decoder fine-tuning

The paper freezes the encoder and tunes the decoder with MSE. The purpose includes making small HUD details readable. Because latent predictions feed the next dynamics step directly, changing the display decoder does not change the latent recursion. This is the core reason our decoder choice can remain deferred after dynamics launch. [G §3.2.2; `rollout_eval.py:94–205`.]

Do not read its frozen-decoder context ablation as an isolated decoder-treatment experiment against the headline. The context ablation has a different training budget and evaluation set. Subtracting its best PSNR from the main headline combines multiple changes. Our own before/after reconstruction gate is a more direct decoder measurement, though it still does not predict the exact gain on generated latents. [G §5.1, §5.2.1; §3.6 above.]

### Section 3.3: efficient sampling

Separate latency from simulated time. The paper reports about **10 ms per U-Net call**, **40 ms** for its usual denoising sequence, and **50 ms** total including the autoencoder, giving **20 FPS**. These are implementation throughput quantities. They do not imply that adjacent training targets are separated by one twentieth of a game-second. [G §3.3.]

The observation guidance scale is **1.5**, and training drops observations with probability **0.1**. The paper says action guidance did not improve quality. Our no-observation-dropout models cannot claim to reproduce that guidance configuration. The relevant comparison is a separately trained observation-dropout ablation, not a sampler flag on the existing checkpoint. [G §§3.3, 4.2; RC 2026-09-21 14:30.]

**Sampling table protocol:** original paper, **2,048 teacher-forced generated frames**, **35 FPS data**, default trained model budget **700k updates**, tuned decoder, unless the paper marks the distilled variant. The table's uncertainty values are the paper's reported errors. This is not our benchmark. Source: G Table 1 and Appendix A.6.

| Sampling variant | PSNR | LPIPS |
|---|---:|---:|
| Distilled single step | 31.10 ± 0.098 | 0.208 ± 0.002 |
| Ordinary 1 step | 25.47 ± 0.098 | 0.255 ± 0.002 |
| 2 steps | 31.91 ± 0.104 | 0.205 ± 0.002 |
| 4 steps | 32.58 ± 0.108 | 0.198 ± 0.002 |
| 8 steps | 32.55 ± 0.110 | 0.196 ± 0.002 |
| 16 steps | 32.44 ± 0.110 | 0.196 ± 0.002 |
| 32 steps | 32.32 ± 0.110 | 0.196 ± 0.002 |
| 64 steps | 32.19 ± 0.110 | 0.197 ± 0.002 |

The **32.58 dB** entry and the headline **29.43 dB** are different reported evaluations. The paper does not provide a single controlled explanation that lets us convert one into the other. Keep both with their local protocols. Do not replace the headline by the larger number, or use their difference as a decoder or stride effect. [G Table 1, §5.1, Appendix A.6.]

### Section 4.1 and Appendix A.5: agent details

The PPO agent uses both a rendered frame and map input at **160×120**, plus its last **32 actions**. The CNN feature size is **512**, with separate two-layer actor and critic heads. Collection uses **eight parallel games**, rollout buffers of **512 steps**, discount **0.99**, entropy coefficient **0.1**, batch **64**, **ten epochs**, and LR **1e-4**. These are the agent's parameters, not the diffusion model's. [G §4.1.]

The action-repeat and shaped-reward details explain the footage distribution. Actions are held for **four frames**, and repeating the prior action is encouraged. Reward terms include enemy hits **300**, kills **1,000**, pickups **100**, secrets **500**, deaths **−5,000**, and player hits **−100**, alongside exploration, health, armor and ammunition terms. The exact prior-action repetition probability is not stated. Compare the resulting state coverage with Arnold rather than borrowing these numbers into our model recipe. [G Appendix A.5.]

### Section 4.2: diffusion training details

The original model uses Adafactor at **2e-5**, batch **128**, clip norm **1**, zero weight decay, and **700k updates**, on **128 TPU v5e devices**. This is **89.6M training presentations**, derived. Wall-clock training time, EMA and warmup are not stated. These omissions prevent a precise original total-compute comparison. [G §4.2.]

Decoder batch is **2,048**, and the paper says other training parameters are identical. Reading that as **700k decoder updates** implies about **1.434B decoder-frame presentations**, derived. That is an interpretation of the shared-parameters statement rather than a separately enumerated decoder-step experiment. State the distinction if using this exposure calculation. Our historical decoder tune's **100k presentations** is directly recorded and far smaller. [G §4.2; RC 2026-09-18 14:50.]

### Section 5.1: quantitative and human evaluation

The headline teacher-forced result uses **2,048 samples from held-out trajectories across five levels**. The paper does not identify a disjoint train-map/test-map split. Calling it a demonstrated unseen-map result is unsupported; calling its exact map overlap proven is also stronger than the disclosure. “Held-out trajectories; map-disjointness not stated” is the precise description. [G §5.1.]

Its FVD results are **114.02** for **16-frame** clips and **186.23** for **32-frame** clips, using **512 rollout samples**. The paper describes those horizons as **0.8 s** and **1.6 s** in its playback setting. Our old horizons represent different game-time intervals, and FVD also depends on sample count and preprocessing. Side-by-side numbers require these captions. [G §5.1.]

The human study uses **ten raters** and **130 clips**, with clip durations **1.6 s** and **3.2 s**. Raters chose the real clip about **58%** and **60%** of the time. A further long-play test uses **150 paired clips** drawn after minutes of interaction and yields about chance discrimination. This is evidence about those clips and raters. It does not prove mechanical equivalence or that experts cannot distinguish the simulator; the authors acknowledge their own ability to do so. [G §5.1.]

### Section 5.2.1: context ablation with frozen decoder

**Protocol:** original paper, **8,912 test examples from five levels**, **200k training updates**, frozen decoder. The default sampler is the paper's usual **four-step DDIM** unless otherwise specified. Source: G §5.2.1, Table 2; default sampler in §3.3.

| Context frames | PSNR | LPIPS |
|---:|---:|---:|
| 1 | 20.94 | 0.358 |
| 2 | 22.03 | 0.304 |
| 4 | 22.26 | 0.298 |
| 8 | 22.26 | 0.296 |
| 16 | 22.28 | 0.296 |
| 32 | 22.31 | 0.296 |
| 64 | 22.36 | 0.295 |

The largest gain is near the shortest contexts. The final doubling adds **0.05 dB** and improves LPIPS by **0.001**, derived from the table, without a reported uncertainty interval for that difference. This supports a bounded context choice. It does not show that history beyond the recent frames is unnecessary for persistent map state. The authors themselves discuss limited memory. [G §5.2.1, §6.]

### Sections 5.2.2–5.2.3: augmentation and collection policy

Read Figure 7 as an autoregressive stability intervention. Its no-augmentation model is not our noaug checkpoint. Compare physical horizons, rendering, data and training budget before claiming replication or contradiction. [G §5.2.2.]

The agent-versus-random experiment uses **700k updates** and decoder tuning, evaluated on **2,048 human-play samples**. Teacher-forced PSNR is **25.06** for the agent-trained model and **24.42** for the random-trained model. After **three seconds**, the reported values are **19.02** and **16.84**. The separate curated difficulty table uses **456 examples**: **112 easy**, **112 medium**, and **232 hard**. These protocols test collection-policy coverage and should not be mixed into the headline benchmark. [G §5.2.3, Table 3.]

### Section 6: limitations and future work

The model has limited memory and can depart from engine rules. The paper names Diffusion Forcing as future work rather than its implemented training method. This is relevant to our deferred stability experiments: adding it would be a new method, not completing an omitted flag from the GameNGen recipe. [G §6.]

### Appendix A.3, Figure 13: data and training exposure

Figure 13 varies the training set over **1M, 5M, 10M and 70M examples**, evaluates **2,048 unseen test trajectories**, and uses a logarithmic training-step axis. At batch **128**, **10k steps** corresponds to **1.28M presentations**, derived. Read both dataset size and presentation count rather than interpreting a curve as “one epoch.” [G Appendix A.3.]

The project note visually estimates the **1M** curve near **24.7 dB at 5k steps** and a peak around **25.9 dB near 20k steps**. These are plot-reading estimates, not tabulated paper values. The latter step count corresponds to **2.56M presentations**, derived. The figure shows the small dataset declining well before the largest training budget; the note that curves separate only after **100k steps** is too strong. [G Figure 13; RC 2026-09-20 21:45.]

Plotting our results relative to measured persistence is useful for explaining why similar model improvements can sit on different absolute PSNR scales. It cannot establish “no unexplained modeling gap” to the original experiment because the original persistence baseline was not measured, and floor-subtracted PSNR does not control motion, stochasticity, frame spacing, decoder or sampler; the claim stays out of the paper (`docs/REVIEW_2026-09-22.md` H5). The **23.79 dB** proxy belongs to reproduction-directory footage with unresolved provenance. Its use is an illustrative comparison, not a substitute measurement. [G Figure 13; Q4 `TABLE.md:16,43–45`; RC 2026-09-20 21:45.]

The decoder, temporal gap and footage changes are not independent additive PSNR effects. The log's rough decomposition of the headline gap into these terms should remain a heuristic hypothesis. A proper decomposition requires controlled interventions on the same windows, rendering path and checkpoint family. [G §§5.1–5.2; Q4; RC 2026-09-20 21:45.]

The appendix also selects illustrative checkpoints by lowest test loss for some qualitative demonstrations. Our model-selection protocol should instead use validation. A practice reported in the source paper is not automatically the right practice for the new benchmark. [G Appendix A.3; RC 2026-09-21 14:30.]

### Remaining appendices

The frame-editing demonstrations include retraining with random initial positions. They are not a held-out-map benchmark. The single-step distillation experiment trains for **1,000 steps**, batch **128**, using guidance **1.5**, and reports about **50 FPS**. The Chrome Dino extension uses **2,000 episodes**, context **32**, resolution **256×512**, and **3,000 training steps**; concatenating episodes allows restart behavior unlike our within-life windows. Keep these as distinct demonstrations with distinct contracts. [G Appendices A.4, A.6, A.10.]

## 5.2 MultiGen

Start with the external state supplied to the model. MultiGen stores geometry and pose in an explicit memory. It ray-traces a depth signal, converts it to a disparity-like representation, and conditions a U-Net alongside visual history. This is a method for persistent scene information, not merely a longer context window. Our next-tic model has no equivalent map memory. [M §3.]

The current paper's quantitative disclosure is richer than some local cards imply. **Table 1** reports SSIM, PSNR and LPIPS, including separate earlier and later trajectory portions. **Table 3** reports a context-length ablation. Neither table makes its “GameNGen” baseline the unreleased original model: it is the authors' implemented memory-free comparison in their setting. [M §§4–6.]

**MultiGen level-conditioned comparison:** source-defined evaluation trajectories, sample count not stated, training-update count not stated, decoder treatment and denoising-step count not stated. Columns are overall / earlier **1–128** / later **128–256** portions as labeled by the paper; do not infer a common protocol with our horizons. Source: M Table 1.

| Method | PSNR, overall / early / late | LPIPS, overall / early / late |
|---|---|---|
| IP-Adapter baseline | 18.74 / 20.30 / 17.19 | 0.488 / 0.397 / 0.578 |
| ControlNet baseline | 18.51 / 19.58 / 17.45 | 0.524 / 0.453 / 0.596 |
| GameNGen-style baseline | 18.77 / 20.23 / 17.33 | 0.471 / 0.379 / 0.562 |
| MultiGen | 19.32 / 20.06 / 18.59 | 0.453 / 0.400 / 0.505 |

The later-horizon improvement is relevant to our memory question. It does not quantify what explicit memory would contribute on our dataset because the inputs and distribution differ. The multiplayer experiment further tests cross-view opponent consistency. Its VLM-based opponent metric is a distinct judge from our action classifier. [M §§4–5, Tables 1–2.]

**Context ablation:** paper's ablation set; count, exact checkpoint updates, decoder and sampling steps not stated. Source: M §6, Table 3.

| Context | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| PSNR | 27.6 | 29.5 | 29.8 | 29.8 | 30.0 |
| LPIPS | 0.121 | 0.097 | 0.094 | 0.093 | 0.089 |
| SSIM | 0.709 | 0.775 | 0.783 | 0.782 | 0.789 |

Do not merge this table's much larger PSNR with Table 1. Their evaluation conditions are not specified as the same. Also distinguish explanatory illustrations from generated samples: the paper's death illustration is not itself evidence that the displayed frames came from the model. [M §6, Table 3; Figure 4 caption.]

## 5.3 DIAMOND

Read the EDM preconditioning argument before treating DIAMOND as evidence against our velocity objective. Its key controlled comparison is between an epsilon-prediction DDPM baseline and EDM on fixed expert Breakout footage. It is not a comparison between EDM and our pretrained VP-velocity adaptation. Pixel-space denoising also removes the VAE bottleneck that shapes our decoder analysis. [D §§3, 5.1; Appendix C.]

The EDM formulation uses data scale **0.5** and samples log noise with mean **−0.4** and standard deviation **1.2**. Its skip, input and output coefficients normalize the denoising task across noise scales. These quantities have a different meaning from our beta endpoints and context-noise level. [D Appendix C.]

The sampling discussion is especially relevant to our low-step blur result. DIAMOND observes that a single denoising step can behave like a conditional mean when futures are multimodal. It uses **three Euler steps** for the final dynamics model. The shared lesson is that few-step PSNR gains do not guarantee perceptual or interactive quality. It is not a claim that the same optimal step count transfers to latent DOOM prediction. [D §5.2.]

Separate Atari from CS:GO. The Atari study trains and evaluates an agent inside the learned world model, so return is central. The CS:GO extension uses a much larger visual model and an upsampler on human gameplay from Dust II. Its **5M training frames** and **87 training hours**, at **16 Hz**, are useful scale references. It does not provide a directly comparable Doom PSNR/LPIPS table. [D §§4, 6.]

The qualitative CS:GO failures include forgotten geometry and invalid repeated jumping. Those are concrete examples of partial observability and dataset-support limits. Our action-conditional image fidelity needs analogous checks for turns, ammunition, health, pickups, deaths and revisits. [D §6.]

## 5.4 PlayGen

PlayGen already applies a DiT-based generative engine to Doom. This removes any defensible “first Doom DiT” framing. Its recurrent hidden state and Diffusion Forcing are structurally different from our channel-stacked image backbones. Read Figure 2 and the training sequence specification together. [P §3.3, Figure 2; Appendix A.]

Its data pipeline is part of the method. Random and expert policies are mixed, trajectory clusters are balanced with a nonnegative least-squares procedure, and high-loss samples receive additional attention. The resulting **200M Doom training frames** are selected from **900M collected frames**. Neither quantity should be mistaken for the number of presentations used in training; the total update count is not stated. [P §§3.2, 4.]

**PlayGen Doom rollout table:** **600 test trajectories**, one real seed observation then autoregression, **128×128** images, frozen learned VAE. Checkpoint training-step count not stated. The principal table matches the **eight-step** sampling row in the step ablation, which is an inference. Source: P §4, Tables 1–2.

| Horizon | PSNR | LPIPS | FVD | Action accuracy |
|---:|---:|---:|---:|---:|
| 1 | 23.81 | 0.165 | — | — |
| 16 | 21.28 | 0.253 | 390.30 | — |
| 32 | 20.41 | 0.285 | 622.51 | 0.858 |
| 64 | 20.03 | 0.311 | 711.09 | 0.851 |
| 128 | 19.24 | 0.346 | 730.29 | 0.848 |
| 1024 | 17.25 | 0.472 | 2176.94 | 0.822 |

Its action metric uses a video-action model over a temporal clip, not our identical judge or action vocabulary. Probability difference compares the classifier's most likely action with the recorded action's probability; it is not sufficient alone because a nearly uniform prediction can have a small gap. Accuracy and a real-data reference remain useful companions. The abstract's small mechanics-decline phrasing should not replace the explicit Doom table values. [P §4 and Table 1.]

Its sampling ablation reports Doom at horizon **32**: **four steps** give **20.74 dB / 0.289 LPIPS / 1156.66 FVD**, **eight** give **20.41 / 0.285 / 622.51**, and **sixteen** give **20.40 / 0.282 / 1030.68**. The corresponding RTX 2060 rates are **20/10/5 FPS**. This independently reinforces the need to keep quality metrics and latency together rather than optimize PSNR alone. [P Table 2.]

The public repository is not a complete released Doom training benchmark. Its current checklist marks Mario inference and weights as available while Doom weights and training data remain unchecked. Cite the paper's measurements as paper results, and the repository's actual available artifacts as release status. [PlayGen repository README:68–73.]

## 5.5 Oasis

The released Open Oasis checkpoint is a **500M-parameter** reduced model with inference code, a transformer autoencoder checkpoint, and action-conditional autoregressive generation. It is not a release of the full training corpus and complete training recipe. The exact optimizer, training exposure, episode split and dataset composition needed for a matched recipe are not stated in the release README. [O README:1–46.]

Read the released generation code alongside the stabilization discussion in our literature card. Context re-noising during sampling is related to our inference-noise question, but its noise schedule, architecture and training distribution differ. Our unsuccessful scalar context-noise sweep is evidence about our checkpoints, not a refutation of every stabilization method used in Oasis. Conversely, a successful Oasis demo is not proof that adding a similarly named flag will repair our rollouts. [`A/lit/rollout-stability-training.md`; Q1 results.]

Oasis is useful as an open inference reference and a reminder that a world model may rely on a different tokenizer and temporal architecture. Adapting its weights would create a system comparison with new representation and pretraining assumptions. It is not a drop-in replacement for the SD-latent backbone table. [O; `A/lit/backbone-candidates-video.md`.]

## 5.6 Diffusion Forcing

Diffusion Forcing independently noises different sequence tokens and trains a causal model across those noise configurations. Sampling can choose a schedule over both sequence time and denoising time. Earlier tokens can be treated as more certain while distant future tokens remain uncertain. A recurrent latent state carries information across the sequence in the paper's causal implementation. [DF §§3.2–3.4.]

Our context corruption instead chooses a shared level for a stacked history and supervises a single target. It does not train a joint sequence over independently noised tokens. Calling it Diffusion Forcing would conflate two different objectives. The meaningful adaptation question is whether independently corrupted temporal tokens and causal state improve long rollouts enough to justify architectural and training changes. [DF §3.2; `diffusion_v.py:91–102,135–151`.]

The video experiments use Minecraft and DMLab with matched recurrent baselines and demonstrate qualitative rollouts beyond training length, including examples around **1,000 frames**. The paper's theoretical statements concern its probabilistic training construction; they are not guarantees of correct game mechanics or indefinite error-free simulation. A Doom experiment still needs raw persistence, motion, action and event checks. [DF §4.1; §3.2 and theoretical appendices.]

Diffusion Forcing does not automatically expose training to the exact distribution of the model's fully self-generated histories. That distinction leads to Self Forcing. [SF §§2–3.]

## 5.7 Self Forcing

Self Forcing generates autoregressive training sequences from the model itself, then applies a video-level distribution-matching objective. Its implementation combines a few-step generator, KV caching, stochastic gradient truncation and losses such as DMD, SiD or GAN matching. It targets exposure bias directly by aligning training with the history distribution encountered at inference. [SF §§3.2–3.4.]

The distinction from scheduled corruption is operational. The generated context may contain structured mistakes, not only Gaussian perturbations of a real frame. The method must manage the cost of self-rollout and the gradients through it. Adding a clean bucket to our current augmentation is not Self Forcing. [SF §3.2; `diffusion_v.py:135–151`.]

The paper reports convergence for a DMD experiment in about **1.5 hours on 64 H100s**. That is a post-training experiment on a pretrained video system, not the full cost of building the base model. Its reported single-H100 speed is about **17 FPS** in the consulted version. Neither number should be presented as a direct cost forecast for our image-stack architectures. [SF §4 and efficiency discussion; Abstract.]

A bounded Doom follow-up would start from a measured next-tic checkpoint, preserve its data split, and compare a carefully specified self-rollout post-training budget against equal additional ordinary training. It would need a fixed reward or distributional objective and safeguards against improving image realism while weakening action control. This is a proposed experiment, not an adopted launch plan.

## 5.8 The open reproductions

### Arnaud Stiegler's `gameNgen-repro`

The repository releases diffusion checkpoints and small/large recorded datasets, provides a PPO collection script, and includes autoregressive inference. It explicitly says its inference is not optimized to the original's **20 FPS**. Its README's full-training command uses batch **12**, LR **5e-5**, **three epochs**, cosine scheduling, gradient checkpointing and **18 data-loader workers**. These are reproduction settings, not GameNGen's original Adafactor recipe. [Stiegler [README](https://github.com/arnaudstiegler/gameNgen-repro):3,12–21,62–76.]

The implementation uses a shorter context and differs in input projection, padding and context-corruption details. The literature audit records context **nine**, unpadded **320×240** images with **4×30×40** latents, and a newly initialized input convolution. Its additive context noise is not our variance-preserving blend. These differences make it a useful public engineering baseline, not an exact restoration of the original method. [`A/lit/doom-world-models.md:65–80`; `A/lit/doom-baselines-2026-09-17.md`; `A/lit/frame-spacing-prior-work.md`.]

Its recorder stores post-step observations with the incoming action. Our recorder stores pre-step observations with outgoing controls. The difference is a causal indexing issue, not a cosmetic schema difference. Its dataset name's skip-frame wording describes action repetition; the recorded rows can still contain every tic. The `lvl5` suffix refers to a difficulty setting, not proof of multiple map identities. [Q4 `TABLE.md:47–71`; `A/lit/doom-maps-prior-work.md`; `A/lit/frame-spacing-prior-work.md`.]

No matched published PSNR/LPIPS benchmark from this reproduction was recovered. We may evaluate its public weights on an explicitly defined compatible dataset, but we should not assign its footage or results to the original GameNGen authors. The local Q4 provenance conflict must be resolved before naming the measured directory as the public dataset. [Stiegler README; Q4 `TABLE.md:7–8,73–78`.]

### Masao Taketani's `GameNGen`

This reproduction builds on Stiegler's work and documents several corrections: context **64**, padding from **320×240** to **320×256**, prevention of cross-episode windows, precomputed latent datasets, distributed synchronization of action embeddings, latent-space observation dropout, and adjustable inference context noise. These are valuable implementation references for failure modes in a practical reproduction. [Taketani [README](https://github.com/Masao-Taketani/GameNGen):35–58.]

The author tried the original LR **2e-5** and Adafactor, then reverted because they performed worse in that implementation. The checklist does not mark **four-step** inference as completed. The README says training used **one or two A100 80 GB GPUs** and warns that default data collection may require roughly **5 TB**, explicitly an approximate recollection. It also states that visual quality remains far from the original. [Taketani README:9–19,40–48.]

The released decoder and diffusion weights are useful. The public pixel and latent datasets are intended for inference; the full training collection must be generated locally. This is more open than the original experiment, but not a downloadable complete original training corpus. Evaluate it under its own provenance and contract. [Taketani README:60–72.]

# 6. Open questions and experiments that would settle them

The experiments below are proposals unless an RC decision explicitly approves them. They should be selected by scientific value and available compute, not all launched automatically.

| Question | What is currently known | Experiment or evidence that would settle it | Required interpretation |
|---|---|---|---|
| Does next-tic training improve dynamics, or mainly raise the persistence floor? | The temporal gap strongly changes raw persistence; next-tic final metrics are not yet available. [Q2/Q4; RC 2026-09-21.] | Compare fixed checkpoints on equal physical horizons, with raw persistence, copy seed, motion, LPIPS, action metrics and event checks. Where possible use the same raw trajectories at each stride. | A larger absolute PSNR alone is insufficient. |
| Does executed history help beyond the latest executed control? | The old scalar ID is wrong during overrides; the new history is causally aligned. [`doom_data.py:725–746`.] | Train a latest-executed-control ablation against full executed history with the same prefix, objective and exposure. Separate turning phases and override transitions. | This isolates history from the correction of mislabeled controls. |
| How much does lost weapon switching limit coverage? | The engine ignores overflow requests; request diagnostics are preserved. [RC 2026-09-22 11:30.] | Complete the full-corpus switch audit; compare weapon/ammo event coverage by worker episode position; record a small corrected-agent corpus under a new namespace. | Corrected-agent footage is a new distribution, not a repair of old pixels. |
| Is the decoder responsible for a large part of the apparent system ranking? | Reconstruction quality differs substantially; old decoders saw held-out maps. [RC 2026-09-18 14:50.] | Tune training-map-only decoders; render identical saved predicted latents through stock, MSE-only and perceptual decoders; score identical raw windows. | Separate dynamics, rendering, and test-map exposure. |
| Does native-schedule adaptation beat the shared VP recipe? | Native SD and SD 3.5 schedules differ; no controlled schedule ablation exists. [RC 2026-09-21 14:30.] | Train a bounded native-schedule row from the same public checkpoint, with an explicit target and sampler contract, matched data and exposure. | A sampler-only change cannot answer a training-objective question. |
| Is augmentation's FVD benefit worth its local fidelity cost? | The noaug grid improves TF and IDM but worsens FVD at its budget. [RC 2026-09-20 08:30.] | Repeat on next-tic data at matched checkpoints and equal physical rollout horizons; include clean/noisy-context stress tests. | Retain all metrics; do not select the winner by a single favorable one. |
| Would an exactly-clean training atom help? | The proposal was withdrawn; clean inference is the low-bucket boundary. [RC 2026-09-21 14:30.] | A separately named mixture-distribution ablation with explicit bucket semantics and identical evaluation. | It is a new training recipe, not a correction to the finished rows. |
| Can observation CFG help? | Existing models lack observation-dropout training. [RC 2026-09-21 14:30.] | Train an observation-dropout branch, sweep guidance on validation with paired noise, and assess both fidelity and control. | Do not call action guidance or arbitrary context subtraction equivalent. |
| Which sampler is useful interactively? | Low-step PSNR can reward blur; exact local sweep matrix is missing. [RC 2026-09-22 09:15.] | Recover artifacts; run a validation sweep with PSNR, LPIPS, FVD, IDM and measured end-to-end latency. | Report quality-latency tradeoffs, not only best PSNR. |
| Does SD 3.5 outperform at equal compute? | It has strong perceptual scores but different representation, size and throughput. [RC 2026-09-22 09:15.] | Plot common-window metrics against presentations and measured card-hours, including failed/rolled-back work. | It is a system-efficiency comparison, not a pure architecture claim. |
| Is long-context memory the limiting factor? | Channel stacking has a bounded history; MultiGen uses explicit geometry memory. [M §3; G §6.] | Use controlled out-and-back trajectories and revisit events; compare a longer temporal model or explicit memory under a separate information contract. | Pixel fidelity near the current view does not establish remembered geometry. |
| Can a learned action judge be trusted across VAEs? | The C16 row needs a rendering/re-encoding bridge. [`rollout_eval.py:409–417`.] | Score every row through the same image-based judge pipeline with its paired real reference; report class recall and override coverage. | Judge accuracy is a proxy, and its domain shift must be measured. |
| Are all eligible IDM subsequences scored? | Current code finds a longest valid run before scoring windows. [`rollout_eval.py:442–468`.] | Construct a rollout with multiple disjoint eligible runs and audit numerator/denominator accounting; decide whether the intended protocol includes all runs. | The log's “all valid windows” wording may mean all windows of the selected run, not all eligible runs. |
| Does Self Forcing offer a better use of extra compute than ordinary continuation? | It addresses generated-history exposure but adds substantial machinery. [SF §3.] | Compare a fixed post-training budget against equal-cost ordinary continuation, preserving split and test protocol. | Account for teacher/critic cost and action fidelity. |
| Can the benchmark include deathmatch_simple? | Engine keyword bug is fixed; only a bounded probe is validated. [`release/DENSE_CORPUS.md:58–75`.] | Record full-length episodes with confirmed opponents, deaths, frags and control semantics; then decide corpus allocation. | The probe is not a finished comparable benchmark. |
| What caused the server outage? | Competing memory and disk explanations remain. [RC 2026-09-21–22.] | Inspect kernel/console logs and historical process/resource records. | Do not infer causality from installed RAM or one plausible code path. |
| Are the release and license complete? | Priority data batch uploaded; later batches and license confirmation pending at cutoff. [RC 2026-09-22 11:30.] | Verify manifests and public file inventory; confirm asset provenance and the chosen license with Rohan. | A staged card is not proof of complete publication. |

## 6.1 Minimum final evaluation contract

Before examining dense-test scores, freeze the episode list and exact window identities. A window should be recoverable by corpus, episode, source tic and target tic, rather than only an integer position in a dataset whose filtering may change. Save the raw-target retrieval contract and reject approximate tic matches. [`eval_tf.py:124–192`; `doom_data.py:711–746`.]

Choose physical horizons explicitly. Save sampler name, denoising steps, eta, guidance branches, inference context noise and the random seed derivation. The evaluator's keyed noise improves pairing across batching changes, but identical seeds do not create identical random tensors across different latent shapes. Cross-VAE comparisons remain paired by observations rather than by a shared latent noise realization. [`eval_tf.py:124–164,246–269`.]

Save the model step, live/EMA choice, model hash, encoder and decoder hashes, and decoder training split. Render raw persistence and autoencoder reconstruction on the same windows. For autoregression, retain copy-seed and motion measurements alongside pixel and distributional metrics. Report map-level results before the aggregate so the mixture is visible. [RC 2026-09-21 14:30; `eval_tf.py:278–335`; `rollout_eval.py:289–532`.]

Use episode-level resampling for uncertainty when windows overlap within episodes. Distinguish uncertainty conditional on a fixed map set from generalization over new maps. FVD needs its clip count, frame sampling and feature extractor stated. If sample counts differ, numerical FVD differences can reflect the estimator as well as the models. [G §5.1; RC 2026-09-16 bootstrap and rollout entries.]

The existing stop rules are operational safeguards, not forecasts of a publishable score. The audit proposes inspection after repeated non-finite skips or sustained validation excursions, and a diagnostic hold if a checkpoint remains below persistence without improvement. Do not turn a forecast into a success threshold or repeatedly restart until a favorable seed appears. [RC 2026-09-21 14:30.]

## 6.2 Source inconsistencies requiring resolution

| Issue | What this dossier does | What a human or original artifact must resolve |
|---|---|---|
| `data-1_4` and `ctx8` are called completed in the request but no numeric results were found locally. | Keeps explicit missing rows. | Locate the completed cell directories, checkpoint identities, window manifests and metrics. |
| DDIM sweep survives as an approximate summary, not a complete numeric matrix. | Reports only the surviving summary. | Recover per-step, per-backbone metrics and exact protocol. |
| Q4 calls local reproduction footage a public JPEG transcode; other records call the continuity corpus fresh lossless recordings. | Attributes measurements to the local directory and flags provenance. | Inspect corpus metadata, source hashes and collection logs. |
| Dense full unseen pool versus JSON operational unseen range. | Distinguishes the full pool from the initial evaluation subset. | Confirm whether release consumers should interpret `ranges.unseen` as the benchmark subset or full split. |
| Decoder provenance conflicts: RC 2026-09-16 15:10 says seen-map training frames; RC 2026-09-21 14:30 says installed decoders saw all maps. | Uses the later warning and qualifies historical unseen-map claims. | Recover exact decoder episode lists/hashes or train a properly restricted decoder. |
| Some completed results exist only in RC; base-grid FVD has slightly different rounded summaries. | Uses the latest dated values and marks missing precision. | Reconcile against original `metrics.json` and checkpoint hashes before publication. |
| Claimed all-window IDM scoring versus longest-run selection in code. | Describes the implemented restriction. | Decide intended denominator and verify it with the original evaluation artifacts. |
| Outage cause is asserted differently across entries. | Leaves cause unresolved. | Use system logs rather than retrospective capacity arguments. |
| Dataset license remains marked for confirmation. | Reports the staged choice and pending confirmation. | Confirm rights/provenance and final release terms. |

Other inconsistencies can be resolved from the sources without a new experiment. MultiGen does report PSNR and a context ablation. PlayGen does use Diffusion Forcing. GameNGen does not state its diffusion-training stride. The repaired weapon-index formula uses a zero-based count of prior starts. The short “compute-matched” pilots actually match presentations. The historical “VAE ceiling” is a reconstruction reference. These corrections should propagate into manuscript text when it is next edited; no other file was changed for this dossier.

# 7. Glossary

| Term | Meaning in this dossier |
|---|---|
| Action-conditioned world model | A learned predictor of future observations given recent observations and controls. It is not necessarily an exact implementation of engine state transitions. |
| Engine tic | The simulation's discrete time unit. This corpus uses 35 tics per game-second. [`release/DATASET_CARD.md:20`.] |
| Stored frame / raw tic | A recorded observation row. Missing respawn intervals mean stored row counts need not equal nominal timeout times engine rate. |
| Decision frame | A frame associated with an agent decision interval. Historical verified windows use intervals of four tics. [`transitions.py:163` onward.] |
| Frame stride | Temporal spacing between observations or prediction targets. It is distinct from control repetition and generation throughput. |
| Action repeat | Holding a requested or executed control across multiple engine steps. It does not imply that intermediate images were not stored. |
| Requested action ID | Arnold's policy-level discrete choice. It may differ from the control list after scripted overrides. |
| Requested control list | The raw `buttons` string submitted to ViZDoom. It may exceed the engine's accepted width. |
| Executed control vector | The nonempty request's first 19 binary entries, right-padded, under the verified engine semantics. [`transitions.py:54–89`.] |
| Incoming / outgoing row convention | Whether a row's action produced its image or is applied after its image. Our recorder uses outgoing controls. |
| Verified chain | A sequence of accepted fixed-duration transitions with consistent executed controls and no life-boundary crossing. |
| Next-tic window | Consecutive raw-tic latents and executed controls ending at the next target; it need not follow a decision chain. |
| Teacher forcing | Supplying real prior observations when predicting the next observation. |
| Autoregressive rollout | Feeding generated observations or latents into subsequent predictions. |
| Exposure bias | The mismatch between real histories used in training and generated histories encountered at inference. |
| Persistence | Copying the last real observation to predict the next real observation. “Floor” is a baseline nickname, not a bound. |
| Decoded copy | Copying the last encoded latent and decoding it. This includes representation/rendering error. |
| Copy seed | Holding the last seed observation fixed for every future horizon. It diagnoses apparent quality from lack of motion. |
| Autoencoder reconstruction reference | Decode the encoder's representation of the true target and compare it with the raw target. Often called the VAE ceiling, but not a strict optimum. |
| Latent normalization | The scale and optional shift converting encoder outputs into the denoiser's expected representation. |
| C4 / C16 | Four-channel and sixteen-channel latent representations in the image-autoencoder rows. [`backbones.py:34–74`.] |
| Channel stacking | Concatenating history frames along channels before spatial processing. Time is not a separate token axis. |
| Cross-attention | Image queries attend to conditioning tokens, such as action history. |
| Joint attention | Image and conditioning tokens participate in a shared attention mechanism, as adapted in SD 3.5. |
| adaLN | Adaptive layer normalization whose modulation depends on conditioning variables. |
| Velocity prediction | Predicting the VP target combining clean signal and Gaussian noise with time-dependent coefficients. [`diffusion_v.py:63–65`.] |
| Epsilon prediction | Predicting the Gaussian noise added to a clean target. It is a different target from velocity. |
| Rectified flow | Learning a velocity along a chosen interpolation between data and noise; its native path differs from the VP schedule here. |
| EDM | A denoising formulation with sigma-dependent preconditioning and noise sampling; DIAMOND uses it in pixel space. [D §3; Appendix C.] |
| Context-noise bucket | A learned discrete label for the sampled corruption level of the observation history. |
| Classifier-free guidance | Combining conditional and appropriately trained condition-dropped predictions at inference. |
| DDPM / DDIM | Different diffusion sampling constructions. The April model's ancestral sampler and the rebuilt DDIM sampler must be named separately. |
| EMA | Exponential moving average of model parameters. It is a distinct weight set whose update arithmetic and checkpoint identity matter. |
| Global batch | Examples contributing to one optimizer update across all devices and accumulation steps. |
| Micro-batch | Examples processed by a device in one forward/backward pass. |
| Presentation | One example consumed during optimization, including repeated visits to the same source footage. |
| Equal exposure | Matching presentations or updates at fixed global batch. It does not match compute. |
| Equal compute | Matching a declared cost measure, such as measured card-hours or FLOPs, including relevant overhead and failed work. |
| PSNR | Logarithmic pixel-error measure. It is sensitive to motion, averaging, rendering and target choice. |
| LPIPS | Learned perceptual image distance; model choice and preprocessing belong in its protocol. |
| FVD | Fréchet distance between video-feature distributions; clip length, sampling and sample count affect the estimate. |
| IDM | Inverse-dynamics model used to infer controls from observation sequences. Its accuracy is a proxy for action fidelity. |
| HUD | Heads-up display, including health/ammunition information whose small symbols are sensitive to decoder quality. |
| Seen map / unseen map | Whether a map appeared in the specified component's training data. State whether this refers to dynamics, decoder, agent, or the whole pipeline. |
| Validation / test | Validation supports choices; test supports final reporting after those choices are frozen. Repeatedly consulted test footage becomes development evidence. |
| Diffusion Forcing | Sequence training with independent noise levels across tokens and flexible temporal denoising schedules. [DF §3.] |
| Self Forcing | Post-training on autoregressively self-generated histories with distribution-matching objectives. [SF §3.] |
| Snapshot / recovery checkpoint | A snapshot primarily supports evaluation; a recovery checkpoint retains the optimizer and other state needed to continue training. |

# 8. Repository and analysis map

## 8.1 Where to answer an implementation question

Paths are relative to the repository root. The analysis paths in the next table resolve through the preferred worktree when a copy exists. The dated RC log is authoritative for adoption status; code is authoritative for current behavior; metrics artifacts are authoritative for a measured value when they exist.

| Question | File or directory | What to inspect |
|---|---|---|
| What are we doing now, and which proposal was accepted? | `RESEARCH_CONTEXT.md`, sections 0 and 7 | Read dated entries newest first; older sections retain superseded plans. |
| What did the class project actually implement? | `A/DOOMDIT_AGENT_CONTEXT.md` | Legacy latent geometry, action convention, sampler, EMA bug and training-set headline. |
| How did the project evolve into a paper? | `A/doomdit-full-recap-2026-07.md` | Historical experiments, missing artifacts, early venue/contribution thinking. |
| What were the initial paper requirements? | `REQUIREMENTS.md` | Useful checklist; stride, context, map scope and desired winner are partly superseded. Its reconstruction “upper bound” language is too strong. |
| What is the dense corpus and why these maps? | `release/DENSE_CORPUS.md` | Recording schema, rejected score-selected maps, engine probe, verification, decision-only caveat. |
| What is meant to be public? | `release/DATASET_CARD.md` | Dataset folders, schema, seeds, control semantics, behavior artifact, split and license. |
| What is the immutable dense split? | `release/dense_split.json` | Full seen-map pools and initial next-tic prefixes/subsets. Distinguish the initial unseen range from its full pool. |
| How are observations and controls recorded? | `record_arnold.py` | Seed derivation, `forced_game`, pre-step row emission, engine button inventory, episode metadata. |
| What does the engine actually consume? | `transitions.py` | `normalize_buttons`, raw-width/switch diagnostics, verified transition construction. |
| How are raw frames encoded? | `encode_parquet.py` | Posterior mean, padding, scale/shift, every-tic mode, producer prefetch, canonical table, sidecar repair. |
| What is a legal training window? | `doom_data.py` | Legacy versus corrected versus next-tic datasets; map/episode splits; continuity/death checks; history slicing. |
| How do the pretrained backbones receive history? | `backbones.py` | Input inflation, action embeddings, history positions, U-Net/PixArt/UniDiffuser/SD 3.5 pathways. |
| What is the exact diffusion target and sampler? | `diffusion_v.py` | Beta construction, velocity/epsilon conversions, DDIM update, context-noise formula. |
| What does one training update do? | `train_wm.py` | Optimization, warmup, precision, corruption, validation seeds, skip guards, EMA, corpus identity, save/resume. |
| What does teacher-forced evaluation measure? | `eval_tf.py` | Raw versus decoded references, raw persistence, horizon selection, keyed noise, per-window/per-map output. |
| What does rollout evaluation feed back and score? | `rollout_eval.py` | Latent recursion, control history, streamed clips, raw targets, copy seed, IDM bridge and eligible-run selection. |
| How is the decoder trained? | `finetune_decoder.py` | Frozen encoder, MSE/perceptual objectives, real-row crop, frame selection, serialization and provenance. |
| What exactly launches a next-tic row? | `scripts/spiderman/launch_nexttic.sh` | Public source, global/per-device batch, action history, skip guard, paths, recovery and fit command. |
| What defines a recipe-grid cell? | `scripts/spiderman/launch_cell.sh` | Base recipe and the single named change; a definition is not evidence the cell completed. |
| Why were initial SD 3.5 metrics wrong? | `scripts/spiderman/after_sd35.sh` | Decoder detection and rescore handling; stock outputs remain a separate rendering treatment. |
| What supports cluster execution? | `scripts/cluster/` | Setup, data fetch/checksums, encoding, gates, launch and status tools merged during the source window. Deployment success is separate from local test success. |
| What happened overnight? | `results/night_2026-09-19/LOG.md` | Queue history and result locations; compare claims against each raw report. |
| Did inference corruption help? | `results/night_2026-09-19/q1-infer-noise/` and `q1-tf-noise/` | Tables, drift/FVD metrics, provenance and paired-window records. |
| Is the dense recording complete and how hard are its maps? | `results/night_2026-09-19/q2-verify-dense/` | Full arenas verification, manifests, sampled map statistics; some unseen verification files are earlier partial inventories. |
| Are decision and every-tic encodes identical? | `results/night_2026-09-19/q3a-pertic/` | Metadata match and failed strict latent-value equivalence at different batch sizes. |
| How does reproduction-directory footage differ? | `results/night_2026-09-19/q4-repro-footage/` | Raw persistence and stock reconstruction metrics; provenance ambiguity remains. |
| Did the deathmatch-simple fix work? | `results/night_2026-09-19/q5-dm-simple/` | Bounded same-seed engine probe, not a full corpus. |
| Where are the sampler/decoder lever results? | `results/levers_2026-09-20/` | Locally present builders do not supply the full requested numeric result matrix. |
| Can the paper-row manifest be used uncritically? | `paper/rows.json:46–56` | Check placeholder flags and run IDs against RC; the reviewed manifest contains stale entries, including the UniDiffuser run ID. |

## 8.2 Which analysis answers which research question?

| Analysis under `A/` | Main question | Reading caution |
|---|---|---|
| `lit/doom-world-models.md` | What do GameNGen, MultiGen, DIAMOND, PlayGen and the first reproduction do? | Primary paper versions supersede incomplete cards. |
| `lit/doom-baselines-2026-09-17.md` | Is there a reproducible external Doom baseline to compare against? | Separate original paper results from another author's implementation and available weights. |
| `lit/doom-maps-prior-work.md` | Which works identify their maps, WADs or held-out-map splits? | A difficulty suffix is not a map count; missing disclosure is “not stated.” |
| `lit/frame-spacing-prior-work.md` | What is a frame, decision, action repeat, or inference FPS in each source? | Do not convert throughput into training stride; distinguish estimates from disclosed rates. |
| `lit/rollout-stability-training.md` | How do context noise, Diffusion Forcing, Self Forcing and inference stabilization differ? | Similar motivation does not mean identical objectives or drop-in code. |
| `lit/backbone-candidates-image.md` | Which public image starts fit each latent space and memory budget? | Approximate parameter and memory estimates need measured fit checks; license/gating affects availability. |
| `lit/backbone-candidates-video.md` | Which video starts offer temporal priors, and what tokenizer changes do they require? | Temporal compression and decoder quality can dominate a per-frame comparison. |
| `astra-nexttic-2026-09-20.md` | What did the next-tic review find about alignment, labels, data and alternatives? | Read successive review rounds; an early recommendation may be withdrawn later. |
| `nexttic-design-2026-09-20.md` | What is the proposed dataset/model/evaluator contract? | Use current code and the final RC decision where implementation evolved. |
| `astra-prelaunch-audit-2026-09-21.md` | What must be fixed before launch, and which recipe choices are intentional mismatches? | Distinguish closed implementation issues, remaining operational gates, and forecasts. |
| `h100-providers-2026-09-22.md` | How should encoding, storage, transfer and training be allocated? | Prices, availability and throughput projections are time-dependent estimates, not measurements of our completed run. |

## 8.3 Reading order for the next paper session

Read GameNGen's task formulation and method first, keeping the recipe table open. Read its implementation section before interpreting the headline metrics. Then read the context, augmentation and collection-policy ablations, followed by Figure 13 and the sampling appendix. Mark each claim as stated, inferred, or missing before comparing it with our ledger.

Read MultiGen for persistent state and additional conditioning information. Read DIAMOND for denoising formulation and the meaning of few-step samples. Read PlayGen for prior Doom transformers, data balancing and action-aware evaluation. Read Diffusion Forcing and Self Forcing only after the current next-tic contract is clear, so their training changes remain distinguishable from our existing augmentation. Use the reproductions to inspect practical engineering choices and public artifacts, while retaining their distance from the original experiment.

For manuscript preparation, recover the missing grid and sampler artifacts before copying those tables into the paper. Freeze dense-test protocol before selecting test results. Preserve the distinctions between a dataset fact, a measured model result, a causal explanation, and a forecast throughout the write-up.

**For a new co-author (added 2026-09-25).** Read in the order section 0 of `RESEARCH_CONTEXT.md` gives: that section, then its log from Sep 23; then this dossier's sections 3.12 to 3.14, 4.10 and 4.11, and 5.9; then the distance-study design and literature memos. Sections 3.1, 1.2 and 4.4 explain the reading rules, the persistence references and the causal contract that every number above depends on.

**What Keerthana can pick up.** Section 0 of `RESEARCH_CONTEXT.md` records the Sep 25 division of labour. Keerthana takes the related work, the dataset and method sections, the PhysWM template with its page budget, and the Figure 1 candidates. Rohan holds the results, the tables, the generalization figure, the scoring and the server. Section 0 cites `.claude/analyses/weekly-update-2026-09-25.md` for the division, but that draft to Changliu states the plan to Sep 30 without assigning work. The rows below point to where each piece starts.

| Piece | Where to start | State at the cutoff |
|---|---|---|
| Related work | Section 5.9 and its citation table; `.claude/analyses/distance-study-literature-2026-09-25.md`; the reading guides of sections 5.1 to 5.8 | Sources read and checked against the PDFs; paragraph not written |
| Dataset section | `release/DATASET_CARD.md`, `release/DENSE_CORPUS.md` and `release/dense_split.json`; sections 4.2 to 4.4; the map list in section 3.14 | Dataset public on Hugging Face; 6,000 training episodes encoded in both latent spaces; 210 seeded evaluation episodes on 30 maps |
| Method section | The recipe and gate paragraphs of section 3.12; section 4.4; the design as run in section 3.14 and the design memo | Recipe fixed and identical across the three rows; the paper's step count and decoder not yet fixed |
| PhysWM template and page budget | `paper/main.tex:1–8`: 4 pages plus references; no CoRL template checked in; the CoRL text block is narrower, so the draft runs about 15 percent longer there | The skeleton carries the Sep 16 stride-four framing and needs the next-tic framing |
| Figure 1 candidates: a rollout strip, W&B curves | Rollout arrays kept on every full read since Sep 24 (section 3.13); the W&B project `doomdit-nexttic` with the native and sidecar runs (section 4.11) | None chosen |

## 3.12 Sep 23: the next-tic launches and the first curves

This section and section 4.10 were appended after section 8 on 2026-09-23, so that no existing section changed while the dossier was being read. This section belongs to part 3 and follows the reading rules of section 3.1. Its source cutoff is main at `7f3b016`, which includes the 2026-09-23 11:30 EDT entry of `RESEARCH_CONTEXT.md`, together with the run steward's last report at 11:49 EDT. `$D` means `/sata2/data/rnagabhi/doom` on Spiderman. Every model number below was read from the named server file by the run steward (`run-steward-unet`) or by the main session during the launch day. This addendum copies their recorded output and contacted no server. The Weights & Biases project `rohannaga04-rapideye/doomdit-nexttic` replays the same `log.jsonl` events and steward JSON files, so it mirrors these sources rather than adding a measurement. “Not recorded” marks a value that exists on the server but that no local record captured before the cutoff.

**Launch protocol:** both rows ran their gates and their production launch from the second checkout `$D/repo_launch` at commit `35258f3`, under the launch certificate `$D/GATES_CERT.json`. The gate runs were staggered by latent space, because the 16-channel corpus finished about three hours after the 4-channel one. Times are UTC as logged, with EDT in brackets. Sources: the `start` event of `$D/results_spiderman/<run>/log.jsonl`; `$D/logs/resumes.log`; `$D/GATES.txt`; RC 2026-09-23 06:30 and 09:10.

| Row | Run directory | Card and tmux session | Gate run | Production launch | Interpreter | Parameters | Training windows |
|---|---|---|---|---|---|---:|---|
| SD 1.4 U-Net | `$D/results_spiderman/040-unet-nexttic` | Spiderman GPU 1, `train-unet-nexttic` | 10:12:11 to 10:27:57, `$D/logs/gates_unet_20260923T1012.log` | 10:28:41 (06:28) | `~/miniconda3/envs/doom/bin/python` | 860,532,932 | 9,661,754 of 10,006,779 candidates; 345,025 (3.45%) excluded |
| SD 3.5 Medium | `$D/results_spiderman/042-sd35-nexttic` | Spiderman GPU 2, `train-sd35-nexttic` | 12:25:59 to 13:03:14, `$D/logs/gates_sd35_20260923T1225.log` | 13:04:38 (09:04) | `~/wanenc/bin/python` | 2,271,745,216 | Not recorded |

The certified launch commands are the `GATES_LAUNCH` lines that each gate run printed at GATES_GO. Before printing a line, `gates.sh` checks that it resolves to the certified command in an empty environment. A bare `launch_nexttic.sh 1 unet` resolves to a different interpreter and checkout, and the certificate check refuses it. [RC 2026-09-23 02:55, 04:05; `.claude/analyses/launch-runbook-2026-09-23.md`, gate section, step 6.]

```
cd /sata2/data/rnagabhi/doom/repo_launch && DOOM_ROOT=/sata2/data/rnagabhi/doom RUN_REPO=/sata2/data/rnagabhi/doom/repo_launch PY_UNET=/home/rnagabhi/miniconda3/envs/doom/bin/python MB=32 WORKERS=12 STEPS=400000 TRAIN_IDS=0:2000 VAL_IDS=6000:6100 bash scripts/spiderman/launch_nexttic.sh 1 unet
cd /sata2/data/rnagabhi/doom/repo_launch && DOOM_ROOT=/sata2/data/rnagabhi/doom RUN_REPO=/sata2/data/rnagabhi/doom/repo_launch PY_SD35=/home/rnagabhi/wanenc/bin/python MB=32 WORKERS=12 STEPS=400000 TRAIN_IDS=0:2000 VAL_IDS=6000:6100 bash scripts/spiderman/launch_nexttic.sh 2 sd35
```

Both rows use the approved next-tic recipe. Each starts from public weights, SD 1.4 or SD 3.5 Medium, and trains on `arenas` ids 0:2000 with validation on ids 6000:6100. The global batch is 32 on one card, with no gradient accumulation. The learning rate is 5e-5 after 2,000 warmup updates, with clip norm 1.0. The fp32 EMA uses decay 0.9999 and is updated every eight steps. Context is 32 tics with 32 executed-control tokens, the objective is v-prediction, and context noise reaches 0.7 over ten buckets. Action dropout is zero and the seed is 0. Validation runs every 1,000 updates on 1,024 fixed windows. Recovery checkpoints come every 5,000 updates and snapshots every 10,000. `STEPS=400000` is a ceiling, not a planned endpoint: the main session stops the rows at the numbers freeze. The U-Net's resolved trainer command in the certificate carries exactly these flags. The SD 3.5 entry of the certificate was not captured locally. [RC 2026-09-23 06:30; `$D/GATES_CERT.json`, `backbones.unet.command`; `.claude/analyses/run-steward-brief-2026-09-23.md`, stop and hold rules.]

The certificate constrains every later start. `launch_nexttic.sh` refuses a launch or resume whose resolved command, commit, data fingerprints or encoder records differ from the certified entry, unless `ALLOW_UNGATED=1` is set and logged. Both production lines in `resumes.log` end with `certified by /sata2/data/rnagabhi/doom/GATES_CERT.json`. Each gate run also left a line for its 310-step resume check, marked `gate run`, at 10:21:59 for the U-Net and 12:48:46 for SD 3.5. [RC 2026-09-23 00:20 (H3), 00:50; `$D/logs/resumes.log`.]

The 10k and 20k U-Net snapshots and the run's `config.json` are mirrored to the public Hugging Face model repository `RohanNaga/doomdit-nexttic` under `040-unet-nexttic/`. The repository listing shows 3,442,605,807 bytes for each snapshot, the size on disk. [The steward's `hf` listings at the 10k and 20k reads; RC 2026-09-23 11:30.]

**Gate protocol:** `scripts/cluster/gates.sh` at `35258f3` stops at the first failure and certifies only the backbone it smoked. Gate 5 scores the 300-step smoke snapshot on 64 validation windows with 10-step DDIM and the stock `sd-vae-ft-mse` decoder, which is the evaluator used for the reads below with fewer windows. GATES_GO means that every listed gate passed. Where a gate's output line was not captured locally, the cell says “passed” on that basis and marks the value as not recorded. h1 and h4 denote horizons of one and four tics. Sources: `$D/GATES.txt`; `$D/logs/gates_unet_20260923T1012.log`; `$D/logs/gates_sd35_20260923T1225.log`; RC 2026-09-23 06:30 and 09:10.

| Gate | SD 1.4 U-Net, 4-channel space | SD 3.5 Medium, 16-channel space |
|---|---|---|
| Corpus before the gates | 0:2000 complete at 05:22 EDT, tail shards 07 and 08 merged without duplicates; normalisation rewrote 250 of 2,000 sidecars; 100-episode raw audit found 0 mismatches over 504,921 rows | Superman finished at 08:24 EDT; row-count audits of train (2,000 episodes, 10,070,779 rows), val (503,864), test (503,620) and unseen (295,984) found 0 mismatches |
| 1: validation sidecars against raw | 0 mismatches in 503,864 rows | 0 mismatches in 503,864 rows |
| 1: training sidecars against raw | 2,000 episodes at 500 rows each: 0 mismatches in 1,000,000 rows | 2,000 episodes at 500 rows each: 0 mismatches in 1,000,000 rows |
| 1c: training inventory, rows and tics of every episode | ok, “6000 ids listed, no problems” | ok, “6000 ids listed, no problems” |
| 1c: exact validation inventory and split file | ok | Passed; line not recorded |
| 1d: stored-latent alignment | Same host: six training shards bit-identical on re-encode, margins 3.21 to 4.52 dB; validation shard 2.91 dB | Cross-host; see the next table |
| 1e: emitted windows | `EMITTED_WINDOWS_OK`, 256 real training windows, 0 violations | Passed; line not recorded |
| 2: yaw alignment on validation | Aligned: shift 0 leads both neighbours by the margin, sidecar audit clean | Passed; line not recorded |
| 3: fit, 20 updates | 1.139 updates/s; 23.3 GB allocated, 23.88 GB reserved; accumulation 1; no checkpointing | 0.377 updates/s; 36.83 GB allocated, 38.19 GB reserved; accumulation 1; gradient checkpointing |
| 4: 300-step smoke | 13 GB recovery checkpoint, 3.3 GB snapshot | 35 GB recovery checkpoint, 8.9 GB snapshot |
| 4b: conditioning probes | Passed; values not recorded | Passed; values not recorded |
| 4c: 10-update resume | Resumed to step 310; optimizer, scheduler, EMA and RNG restored | Resumed to step 310; passed |
| 5: readback, PSNR / LPIPS (persistence PSNR) | Live h1 18.38 / 0.618 (21.92); live h4 18.27 / 0.621 (19.60); EMA h1 10.31 / 0.951; EMA h4 10.14 / 0.956 | Passed for live and EMA at h1 and h4; values not recorded |
| Result | GATES_GO at 10:27:57 UTC | GATES_GO at 13:03:14 UTC |

This table establishes that both corpora match their raw recordings in every audited row, and that the stored latents line up with their sidecars on every sampled shard. It establishes that both configurations fit on one card, train for 300 steps, resume with their optimizer, scheduler, EMA and random state, and can be scored by the evaluator. It does not establish model quality. The 300-step readbacks sit far below persistence, 18.38 against 21.92 dB for the U-Net at h1. The EMA readbacks score the public start, since 0.9999 raised to the 300th power leaves 97% of the start weights in the EMA (derived). The audits read 500 rows per training episode and the alignment gate two episodes per shard, so they bound systematic misalignment rather than checking every row. The fit is a memory certificate more than a speed measurement: from `log.jsonl` timestamps the U-Net later ran at 1.78 to 1.83 updates/s alone, against 1.139 in the 20-update fit. The U-Net holds 23.3 GB of a 48 GB A6000, so the recipe's fixed global batch of 32 leaves about half the card empty. A larger micro-batch would change the recipe, so the fill-the-card lever does not apply to this row. [RC 2026-09-23 02:55, 03:00, 06:30, 09:10; `CLAUDE.md`, Server section.]

**Alignment protocol:** `check_latent_alignment.py` at `35258f3`, two episodes per encoder shard log. It re-encodes stored frames with the recorded settings and compares them with the stored latents: mean absolute difference (MAE), 99th-percentile absolute difference (p99) and the fraction of bit-identical values. It then decodes the stored latents and scores `vae_psnr` against the raw frames, with the rows unshifted and shifted by −4, −1, +1 and +4. The margin is the unshifted score minus the best shifted score. The thresholds at `35258f3` are MAE at most 5e-3, p99 at most 2e-2, a margin of at least 2 dB, every shard log present and one latent contract per space. The 4-channel latents were encoded on Spiderman, the host that ran the gate. The 16-channel latents were encoded on Superman's A4000s with cuDNN 9.10 and re-encoded by the gate on Spiderman's A6000 with cuDNN 9.24. Sources: `$D/GATES.txt` for sd15; the locally captured, truncated excerpt of `$D/logs/gates_sd35_20260923T1225.log` for sd35; the 06:15 EDT validation pre-run output reported in RC 2026-09-23 06:30; thresholds in `check_latent_alignment.py:22–53,70–73` and RC 2026-09-23 04:05, 04:40.

| Space, split, shard | MAE | p99 | Bit-identical | `vae_psnr` unshifted | Best shifted (shift) | Margin (dB) |
|---|---:|---:|---:|---:|---:|---:|
| sd15 train 00 | 0 | 0 | 100.0% | 23.56 | 19.04 (−1) | 4.52 |
| sd15 train 01 | 0 | 0 | 100.0% | 23.45 | 19.29 (+1) | 4.16 |
| sd15 train 03 | 0 | 0 | 100.0% | 24.37 | 21.16 (−1) | 3.21 |
| sd15 train 05 | 0 | 0 | 100.0% | 23.60 | 19.95 (+1) | 3.65 |
| sd15 train 07 | 0 | 0 | 100.0% | 22.99 | 18.66 (−1) | 4.33 |
| sd15 train 08 | 0 | 0 | 100.0% | 23.30 | 19.72 (+1) | 3.58 |
| sd15 val 00 | 0 | 0 | 100.0% | 25.46 | 22.55 (−1) | 2.91 |
| sd35 train 10 | 8.85e-4 | 0.0117 | 70.4% | 27.53 | 22.21 (+1) | 5.32 |
| sd35 train 11 | 8.76e-4 | 0.0117 | Not recorded | Not recorded | Not recorded | Not recorded |
| sd35 train 12 to 16 | Not recorded | Not recorded | Not recorded | Not recorded | Not recorded | Not recorded |
| sd35 val 10 | 9.04e-4 | 0.0117 | 69.7% | 27.80 | 22.95 (−1) | 4.85 |
| sd35 val 11 | 7.82e-4 | 0.00684 | Not recorded | Not recorded | Not recorded | Not recorded |
| sd35 val 14, pre-run | 9.27e-4 | 0.0117 | 69.1% | 26.56 | 19.97 (−1) | 6.59 |
| sd35 val 15, pre-run | 8.04e-4 | 0.00684 | 72.5% | 26.00 | 19.15 (+1) | 6.85 |

The cross-host tolerance was calibrated earlier that night on one 4-channel episode encoded on both hosts. That comparison gave 67% bit-identical values, an RMS difference of 2.5e-3 (0.3% of the latent standard deviation), a p99 difference of 0.011 and decoded PSNR within 0.001 dB. The SD 3.5 gate was the first check of that tolerance on the 16-channel corpus. [RC 2026-09-23 00:50; `check_latent_alignment.py:41–47`.]

Two log summaries disagree with the gate lines, and the gate lines take precedence. The 06:30 entry gives the U-Net training margins as 4.2 to 4.5 dB, while `$D/GATES.txt` records 3.21 to 4.52 dB, with three shards below 4 dB. The 09:10 entry gives the SD 3.5 margins as 4.9 to 5.3 dB for training and 6.6 to 6.9 dB for validation. The captured gate lines show 5.32 dB on training shard 10 and 4.85 dB on validation shard 10. The values 6.59 and 6.85 dB belong to the 06:15 EDT pre-run on validation shards 14 and 15. The same entry's MAE range of 8.9e-4 to 9.0e-4 also excludes the recorded 8.76e-4 of training shard 11 and 7.82e-4 of validation shard 11. Every shard passed, since GATES_GO requires it. Every local capture of the full per-shard SD 3.5 lines was truncated; the complete record is in `$D/GATES.txt`.

This table establishes that on every sampled shard in both spaces the unshifted rows reconstruct the raw frames better than any shifted control. No sampled shard's latents are therefore off by one or four rows against their sidecars. The cross-host SD 3.5 differences are rounding-sized: the largest recorded MAE is 19% of its tolerance and the largest p99 is 59% of its tolerance (derived from 9.27e-4 against 5e-3 and 0.0117 against 2e-2). The SD 3.5 margins are larger than the U-Net's, which follows from its better reconstruction, about 27.5 against 23.5 dB unshifted, while a shifted row stays near the one-tic persistence level. That last point is an inference from the table, not a separate measurement. The check does not establish alignment for every episode. Astra's acceptance of the thresholds called them sampled checks, not proof. [RC 2026-09-23 04:40.]

**Validation protocol:** each trainer's own 1,024 fixed windows from ids 6000:6100, scored every 1,000 updates as velocity loss overall and by timestep quartile, lowest timestep first. The planning bands are the pre-launch audit's forecasts as restated in the review and the steward brief; they are not thresholds. A C16 loss and a C4 loss are not calibrated measures of the same difficulty (section 3.1). Sources: `val` events in `$D/results_spiderman/040-unet-nexttic/log.jsonl` and `$D/results_spiderman/042-sd35-nexttic/log.jsonl`; bands from `docs/REVIEW_2026-09-22.md` §7.2 and `.claude/analyses/run-steward-brief-2026-09-23.md`.

| Row | Update | Validation loss | By timestep quartile, low to high | Planning band |
|---|---:|---:|---|---:|
| U-Net | 1k | 0.2517 | 0.3662 / 0.2209 / 0.2116 / 0.2096 | 0.30 |
| U-Net | 5k | 0.2059 | 0.3178 / 0.1745 / 0.1692 / 0.1634 | 0.24 |
| U-Net | 10k | 0.1948 | 0.3036 / 0.1631 / 0.1597 / 0.1539 | 0.225 |
| U-Net | 20k | 0.1837 | 0.2885 / 0.1515 / 0.1506 / 0.1455 | 0.21 |
| U-Net | 22k, latest recorded | 0.1826 | 0.2859 / 0.1504 / 0.1496 / 0.1456 | None set |
| SD 3.5 | 1k | 0.1408 | 0.2701 / 0.0980 / 0.0972 / 0.0992 | 0.19 |
| SD 3.5 | 2k | 0.1359 | 0.2801 / 0.0898 / 0.0869 / 0.0882 | None set |
| SD 3.5 | 3k, latest recorded | 0.1243 | 0.2533 / 0.0829 / 0.0805 / 0.0817 | None set |
| SD 3.5 | 5k / 10k / 20k | Not reached at the cutoff | Not reached | 0.14 / 0.128 / 0.118 |

Through the cutoff neither run logged a non-finite loss, a skipped update or a validation excursion. The U-Net's gradient norm was about 0.14 to 0.15 from 17k to 22k updates, with clip fraction 0. SD 3.5 clipped occasionally in its first updates (clip fraction 0.16 at update 550 and 0.02 at 2,050) and not in the last records, where its gradient norm was 0.28 at 3,600 updates. From `log.jsonl` timestamps the U-Net ran at 1.78 to 1.83 updates/s alone on GPU 1. It ran at 0.79 to 0.99 while the 5k read shared the card for 50 minutes, and at 1.42 to 1.45 during the 10k read and the SD 3.5 gates. The logged `steps_per_s` field is a slow running average that hides these swings. SD 3.5 ran at 0.414 to 0.416 updates/s by timestamps and 0.418 to 0.43 by the logged average. At 11:30 EDT the U-Net stood at 22,250 updates and SD 3.5 at 3,600. [`train` events of both `log.jsonl` files as read by the steward; RC 2026-09-23 09:10, 11:30.]

This table establishes that both rows descended without an excursion and ran ahead of the audit's forecast. The U-Net's 20k loss of 0.1837 sits 0.026 below its band (derived). SD 3.5 at 3k, 0.1243, is already below the band set for 10k, 0.128. The SD 3.5 lowest-timestep quartile rose once, from 0.2701 at 1k to 0.2801 at 2k, and fell to 0.2533 at 3k; the steward flagged the rise and cleared the flag at 3k. The table does not rank the rows, because the two losses measure targets in different latent spaces. It does not predict pixel quality; the teacher-forced reads below are the pixel evidence. At 20k the U-Net had consumed 640,000 windows, about 6.6% of one pass over its 9,661,754 (derived), so every statement in this section concerns the start of training.

**Teacher-forced read protocol:** `eval_tf.py --tic-stride 1` at `35258f3`, run from `$D/repo_launch` with the run's interpreter. Horizons h1 and h4; 512 windows from validation ids 6000:6100 (`$D/latents_arnold_dense_pertic_eval/val/split_val.json`, seed 0), identical at every step; 10-step DDIM with linear spacing, eta 0, clean context; stock `sd-vae-ft-mse` decoder; raw target frames. Live and EMA weights come from the same saved file: the 5k recovery checkpoint `0005000.pt`, because no snapshot exists at 5k, and the snapshots `snap_0010000.pt` and `snap_0020000.pt`. Persistence scores the raw last frame against the raw target. Decoded copy scores the decoded last latent against the raw target. Reconstruction scores the decoded target latent against the raw target. PSNR is in dB. Sources: `$D/results_spiderman/040-unet-nexttic/steward_<step>/tf_<live|ema>_h<1|4>/metrics.json`; the read script `$D/tmp/steward/steward_eval.sh`.

| Update, file | Weights | Horizon | PSNR | LPIPS | PSNR minus persistence | Persistence PSNR / LPIPS | Decoded-copy PSNR | Reconstruction PSNR / LPIPS |
|---|---|---|---:|---:|---:|---|---:|---|
| 5k, `0005000.pt` | Live | h1 | 21.148 | 0.295 | −0.418 | 21.566 / 0.203 | 20.296 | 23.927 / 0.094 |
| 5k, `0005000.pt` | Live | h4 | 19.724 | 0.388 | +0.511 | 19.213 / 0.354 | 18.598 | 23.913 / 0.094 |
| 5k, `0005000.pt` | EMA | h1 | 11.755 | 0.909 | −9.810 | 21.566 / 0.203 | 20.296 | 23.927 / 0.094 |
| 5k, `0005000.pt` | EMA | h4 | 11.171 | 0.933 | −8.042 | 19.213 / 0.354 | 18.598 | 23.913 / 0.094 |
| 10k, `snap_0010000.pt` | Live | h1 | 21.493 | 0.264 | −0.073 | 21.566 / 0.203 | 20.296 | 23.927 / 0.094 |
| 10k, `snap_0010000.pt` | Live | h4 | 20.222 | 0.347 | +1.009 | 19.213 / 0.354 | 18.598 | 23.913 / 0.094 |
| 10k, `snap_0010000.pt` | EMA | h1 | 14.646 | 0.557 | −6.920 | 21.566 / 0.203 | 20.296 | 23.927 / 0.094 |
| 10k, `snap_0010000.pt` | EMA | h4 | 12.899 | 0.788 | −6.314 | 19.213 / 0.354 | 18.598 | 23.913 / 0.094 |
| 20k, `snap_0020000.pt` | Live | h1 | 21.836 | 0.237 | +0.270 | 21.566 / 0.203 | 20.296 | 23.927 / 0.094 |
| 20k, `snap_0020000.pt` | Live | h4 | 20.522 | 0.311 | +1.309 | 19.213 / 0.354 | 18.598 | 23.913 / 0.094 |
| 20k, `snap_0020000.pt` | EMA | h1 | 20.068 | 0.270 | −1.497 | 21.566 / 0.203 | 20.296 | 23.927 / 0.094 |
| 20k, `snap_0020000.pt` | EMA | h4 | 18.114 | 0.365 | −1.099 | 19.213 / 0.354 | 18.598 | 23.913 / 0.094 |

This table establishes three things about the live weights on validation windows at 10 denoising steps. At one tic, live PSNR rose from 21.148 to 21.493 to 21.836 dB and crossed persistence, 21.566 dB, between 10k and 20k. At one tic, live LPIPS improved from 0.295 to 0.237 but stayed worse than persistence's 0.203 at every read, so the model does not yet beat copying the last frame perceptually. At four tics, where persistence falls to 19.213 dB, live PSNR beat it at every read, by 0.511, 1.009 and 1.309 dB, and live LPIPS beat it from 10k on (0.347 and 0.311 against 0.354). The steward's hold rule does not apply: it fires at 0.3 dB or more below persistence at 20k with under 0.1 dB of improvement since 10k, and the 20k gap is +0.270 dB after an improvement of 0.343 dB (derived). [`.claude/analyses/run-steward-brief-2026-09-23.md`, stop and hold rules.]

The table does not establish a result under the paper's protocol. The reads use 10 steps, and the 10k sweep below shows the 10-step read scoring 0.36 dB above the 50-step read at h1 (21.493 against 21.134, derived). If that offset held at 20k, the 20k live h1 score would sit about 0.09 dB below persistence at 50 steps; that is an estimate, not a measurement. The stock decoder is not the decoder the paper will use. All windows come from seen-map validation episodes, and no test or unseen-map window has been scored for either row. The metrics files hold only an independent-window standard error: 0.095 dB for live h1 PSNR at 5k and 0.166 dB for persistence. The 512 windows share 100 episodes, so those errors understate the uncertainty, and no paired episode-bootstrap interval exists yet. The 10k gap of −0.073 dB is smaller than either standard error. [`steward_5000/tf_live_h1/metrics.json`, `psnr_raw.sem` and `persist_psnr_raw.sem`.]

The EMA trails by design. The trainer applies decay 0.9999 per update in steps of eight (`train_wm.py:819–820`), so after s updates the public start weights still make up 0.9999 raised to the power s of the EMA. That share is about 0.61 at 5k, 0.37 at 10k and 0.14 at 20k (derived). A weight average of the public start and the adapted model need not predict frames at all early in training, which is consistent with the 11.755 dB EMA h1 score at 5k. The EMA gained 8.3 dB at h1 between 5k and 20k and still trails live by 1.77 dB at h1 and 2.41 dB at h4 (derived). The historical stride-four rows put the EMA ahead of live at 90k (section 3.2). These early reads say nothing about the final ordering, which the preregistered validation selection decides: the highest raw PSNR among candidates whose LPIPS is within 0.01 of the best, with live and EMA paired on 512 windows. [RC 2026-09-23 00:20.]

**Rollout protocol:** `rollout_eval.py --rollout` and then `--score` at `35258f3`. Sixteen rollouts of 256 tics each from validation ids 6000:6100, seed 0, identical at every step; 10-step DDIM with linear spacing and clean context; the recorded executed controls; each prediction fed back as context. Scoring uses the stock decoder against raw frames keyed by recorded tic, at 4, 32, 64, 128 and 256 tics. 256 tics are about 7.31 game-seconds (derived). Copy-seed holds the last real seed frame for every horizon. Copy-last uses the real frame one tic before each target, so it measures local difficulty and is not an open-loop competitor (review M4). The reference rows come from the 5k file; the copy-seed and copy-last PSNR values and the copy-seed LPIPS values printed from the 10k and 20k files are identical to them. The 5k EMA rollout was run but its scores were not recorded. Sources: `$D/results_spiderman/040-unet-nexttic/steward_<step>/rollout_<live|ema>_metrics/drift.json`; `$D/tmp/steward/steward_eval.sh`.

Raw PSNR in dB:

| Tics | Live 5k | Live 10k | Live 20k | EMA 10k | EMA 20k | Copy-seed | Copy-last | Reconstruction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 19.66 | 20.07 | 20.39 | 12.81 | 18.28 | 18.93 | 21.53 | 23.81 |
| 32 | 17.55 | 18.23 | 18.72 | 13.07 | 17.14 | 17.47 | 20.60 | 23.67 |
| 64 | 17.21 | 18.22 | 17.34 | 12.98 | 16.16 | 17.60 | 21.16 | 23.74 |
| 128 | 16.79 | 17.86 | 18.04 | 12.92 | 15.36 | 17.80 | 22.24 | 23.85 |
| 256 | 16.22 | 17.35 | 17.76 | 12.75 | 15.17 | 17.52 | 21.30 | 23.82 |

Raw LPIPS:

| Tics | Live 5k | Live 10k | Live 20k | EMA 10k | EMA 20k | Copy-seed | Copy-last | Reconstruction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.347 | 0.316 | 0.274 | 0.771 | 0.324 | 0.317 | 0.175 | 0.092 |
| 32 | 0.546 | 0.527 | 0.496 | 0.828 | 0.533 | 0.487 | 0.223 | 0.097 |
| 64 | 0.579 | 0.568 | 0.535 | 0.844 | 0.614 | 0.505 | 0.211 | 0.104 |
| 128 | 0.575 | 0.620 | 0.578 | 0.844 | 0.648 | 0.502 | 0.185 | 0.106 |
| 256 | 0.625 | 0.620 | 0.609 | 0.870 | 0.674 | 0.520 | 0.226 | 0.102 |

These tables establish that the live U-Net's open-loop rollouts beat holding the seed frame in PSNR at short horizons. The lead at 4 tics grew from 0.73 dB at 5k to 1.47 dB at 20k, and the lead at 32 tics from 0.08 to 1.25 dB (derived). At 20k the live rollout also led copy-seed at 128 and 256 tics by 0.24 dB, but trailed it at 64 tics by 0.26 dB. The same 64-tic cell read 18.22 dB at 10k and 17.34 dB at 20k on identical windows. With 16 rollouts and no recorded interval, the ordering beyond 32 tics is not resolved. In LPIPS the model matched or beat copy-seed only at 4 tics and only from 10k: level at 10k (0.316 against 0.317) and ahead at 20k (0.274), after trailing at 5k (0.347). At 32 tics it was slightly worse at 20k (0.496 against 0.487), and from 64 tics on it was worse at every read, 0.609 against 0.520 at 256 tics at 20k. The stride-four rows showed the same pattern, with copy-seed holding the best perceptual score at the last horizon (section 3.3).

The tables do not establish that the model simulates motion. A model that learned persistence and blurred toward a static frame could also beat copy-seed in PSNR while losing to it in LPIPS. The motion ratio that would separate the two is not implemented (section 4.10). Copy-last scores 20.60 to 22.24 dB only because it reads the real previous frame. Rollouts need the seed and the whole horizon inside one life; on the review's 12 local episodes that rule excluded 34% of 256-tic candidates against 3.7% at one tic, so these rollouts describe surviving stretches (review M4). The EMA rollouts sit below copy-seed at every horizon at 10k and 20k, as their teacher-forced lag predicts. Sampling is 10-step, the decoder is stock, and all 16 rollouts come from validation episodes.

**Probe protocol:** `smoke_probe.py` at `35258f3` on the recovery checkpoint of each step (`0005000.pt`, `0010000.pt`, `0020000.pt`), with a batch of four validation windows from ids 6000:6100 drawn with seed 0. For each conditioning group it reports the probe's `relative_update`: the largest, over the group's tensors, of the norm of live weights minus EMA weights divided by the norm of the live weights. For the inflated input convolution it also reports the norm of the context input channels, which start at zero. Sensitivity is the largest absolute change in the model's output when every bit of the newest executed control is flipped, at fixed noise, context and timestep t = 500. The probe loss is the v-loss at that fixed noise and timestep. Sources: `$D/results_spiderman/040-unet-nexttic/steward_<step>/probe.json`; definitions in `smoke_probe.py:84–115,117–167,183–190`.

| Update | Control MLP | Control positions | Context convolution | Context-channel norm | Newest-control sensitivity | Probe loss | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5k | 0.088 | 0.496 | 0.144 | 1.30 | 1.99 | 0.142 | `SMOKE_PROBE_OK` |
| 10k | 0.068 | 0.447 | 0.163 | 1.93 | 2.15 | 0.132 | `SMOKE_PROBE_OK` |
| 20k | 0.041 | 0.281 | 0.147 | 2.75 | 2.50 | Not recorded | `SMOKE_PROBE_OK` |

This table establishes that the conditioning pathways stayed finite and kept moving through 20k updates. The output depends on the newest executed control, increasingly so: the sensitivity rose from 1.99 to 2.15 to 2.50. The context channels of the inflated input convolution, zero at initialization, grew in norm from 1.30 to 2.75. It does not establish that the dependence has the right sign. The sensitivity is a maximum over four windows and every latent element, so a larger value is not by itself better control. The directional check that would test the sign is not implemented (section 4.10). The relative live-to-EMA distance is not a per-update ratio. Its fall for the control MLP, from 0.088 to 0.041, and for the positions, from 0.496 to 0.281, is what an EMA catching up produces, and it does not show that those pathways stopped learning. The trainer's own `update_ratio` events do measure single-update changes (`train_wm.py:814–818`), but their fixed tensor list omits the control MLP and the positions, a gap the review named. [`docs/REVIEW_2026-09-22.md` §7.2.]

**Step-sweep protocol:** `eval_tf.py` at h1 on the same 512 validation windows, live weights of `snap_0010000.pt`, seed 0, DDIM with eta 0, stock decoder, raw targets. Step counts 4, 8, 16 and 50, each with `--timestep-spacing linear` and `trailing` (review M1). The 10-step row is the linear read from the teacher-forced table. Frames per second is the evaluator's `sampling_frames_per_s`, measured while the card also trained the U-Net. Persistence on these windows is 21.566 dB / 0.203 LPIPS. Sources: `$D/results_spiderman/040-unet-nexttic/steward_10000/sweep_<linear|trailing>_s<steps>/metrics.json`.

| Steps | Linear PSNR / LPIPS | Trailing PSNR / LPIPS | Linear minus persistence (dB) | Trailing minus persistence (dB) | Frames/s, linear / trailing |
|---:|---|---|---:|---:|---|
| 4 | 21.567 / 0.376 | 21.586 / 0.346 | +0.001 | +0.020 | 26.25 / 26.32 |
| 8 | 21.552 / 0.282 | 21.525 / 0.275 | −0.014 | −0.040 | 13.43 / 13.57 |
| 10 | 21.493 / 0.264 | Not run | −0.073 | Not run | Not recorded |
| 16 | 21.371 / 0.239 | 21.365 / 0.239 | −0.195 | −0.201 | 6.51 / 6.58 |
| 50 | 21.134 / 0.221 | 21.143 / 0.222 | −0.432 | −0.422 | 2.07 / 2.09 |

This table establishes a perception-distortion trade at one tic for this checkpoint. Under either spacing, each increase in step count lowered PSNR, from about 21.57 to 21.13 dB, and improved LPIPS, from 0.376 to 0.221. A few-step sample lies nearer the conditional mean of the next frame, which is blurred; that wins mean squared error and loses perceptual fidelity (section 3.10). Spacing mattered only at 4 steps, where trailing improved LPIPS by 0.030. From 8 steps on, the two spacings agree within 0.03 dB and 0.007 LPIPS (derived). Step count moved LPIPS four to five times as much as spacing did: by 0.124 to 0.155 between 4 and 50 steps, against 0.030 between spacings at 4 steps (derived). Persistence, at 0.203, beat every setting in LPIPS. The 10-step reads sit between the 8- and 16-step points, so they flatter PSNR against a 50-step protocol by about 0.36 dB. The paper protocol must therefore fix one step count before scoring. [RC 2026-09-23 11:30.]

The table does not establish the best sampler for rollouts, because the sweep is teacher-forced at one tic only. It does not show how the trade moves with training; the repeat at 50k is planned. The frames-per-second column is a relative cost on a shared card, not a latency measurement. [RC 2026-09-23 11:30.]

**SD 3.5 reads.** No SD 3.5 read existed at the cutoff. Its 5k recovery checkpoint was due at about 12:30 EDT by the steward's estimate. Its reads run on GPU 3 under the 11:30 decision, because a read on a training card costs updates, and because the SD 3.5 training job already holds 39.5 GB of GPU 2 while the evaluator's footprint is unmeasured (section 4.10). Its validation losses and planning band are in the validation table above. [RC 2026-09-23 09:10, 11:30.]

Taken together, the first curves show that both rows train stably under the certified contract. They show that the U-Net crossed the one-tic persistence floor in PSNR at 10 steps between 10k and 20k updates on seen-map validation windows, and led persistence by 1.31 dB at four tics at 20k. They do not yet show which row is better, whether either beats persistence under a 50-step protocol, whether controls move the image in the right direction, or how either row transfers to unseen maps.

## 4.10 Sep 23: decisions of the launch night

This section was appended with section 3.12; it belongs to part 4 and follows the ledger form of section 4.1. Times are EDT on 2026-09-23 and name the `RESEARCH_CONTEXT.md` entry that records each decision. A second time names the entry that revised or accepted it. Commits are on main. Astra is the Codex reviewer named in the RC entries.

| Time | Decision | Evidence | Alternative rejected | Where recorded |
|---|---|---|---|---|
| 00:20 | Round one: close the independent review's seven findings before any launch. H1 moves the unseen scoring subset to `arenas_678` ids 60:120. H2 adds a validation-only selection step and seals the test and unseen corpora for a single scoring. H3 audits training sidecars and latent alignment in the gates, drops the post-gate `git pull` and refuses an uncertified checkout. H4 tunes decoders on training ids only, with a provenance registry. H5, M1 and L1 correct the GameNGen text, add `--timestep-spacing` and correct the dataset card. | The review's verdict was training GO and evaluation NO-GO. 51 of the 60 old unseen ids were worker-first episodes, measured from raw buttons. The suite passed 1,147 tests. | Launching on the training GO alone while the evaluation pipeline stayed NO-GO. | RC 00:20; `docs/REVIEW_2026-09-22.md`, Verdict and §1; merges `41f977c`, `d18bea8` |
| 00:50, merged 02:55 | Round two: close Astra's eight confirmed defects. The certificate verifies the resolved command, backbone, data fingerprints and encoder revisions. The alignment gate requires every shard log, one contract and finite values, and replaces bit identity with calibrated MAE, p99 and a shift margin. The gates gain the rest of review 7.1. Test is sealed at first access, the whole scoring configuration is pinned at selection, caches are keyed by split contents, and the decoder tune defaults to training ids. | Astra reproduced each defect. For example, the launcher accepted an SD 3.5-only certificate for a U-Net launch, and a one-ULP fp16 perturbation, which is harmless rounding, failed the bit-identity floor (MAE 2.4e-4, 0% identical). After the fixes all six of its attacks failed correctly. The suite passed 1,215 tests. | Launching behind gates shown to accept a changed recipe. | RC 00:50, 02:55; commits `d2d6260` to `ebe1bec`; merge `f525782` |
| 02:55, merged 04:05 | Round three: close three narrower defects and add three launch controls. An exclusive per-run lock separates a live evaluator from an interrupted one. One latent contract per space must hold across train, val and the backbone. One production-argument set feeds the fit, smoke, resume and certificate. The additions are per-backbone interpreters (`PY_UNET`, `PY_SD35`), `RUN_REPO`, and a printed `GATES_LAUNCH` line. | Astra's reproductions: two concurrent evaluators both scored test; a train scale of 1 against a val scale of 2 passed; a smoke at lr 5e-5 with 12 workers certified a launch at lr 0.1 with 4. The `doom` environment's diffusers 0.31 is too old for SD 3.5; `wanenc` has 0.40. The suite passed 1,240 tests. | Certifying from arguments the smoke never ran; one interpreter for both latent spaces. | RC 02:55, 04:05; `paper/fixtures/test_nexttic_defects3.py`; commits `42b4a03`, `b063103`, `6fd4635`, `48ff3cd`, `29d5981`, `e3f4045`, `dbaede3`; merge `1747d46` |
| 04:05, merged 04:40 | Round four: close the last three defects. A failed revocation is retried and never deletes another backbone's entry. Every GPU step of the gates runs on its own card as `cuda:0`. `after_nexttic.sh` scores from `RUN_REPO` with the backbone's interpreter and records both. Astra then gave GO for launching behind the gates and GO for paper numbers scored from `repo_launch`. | Astra's fourth pass confirmed all three in code; the readbacks had run on GPU 0, another user's card. The suite passed 1,251 tests. Astra's verdict was CPU-level, so the GPU gates still had to pass. | Launching on the round-three code. | RC 04:05, 04:40; `paper/fixtures/test_nexttic_defects4.py`; commits `32aef39`, `3bfc245`, `261bdb4`, `1132afe`; merge `35258f3` |
| 03:00 | Fix the launch blocker that the real-data pre-smoke found. `doom_data.ensure_open_file_budget` raises the soft open-file limit to the episode count plus headroom before the first memory map, in all three dataset constructors; forked loader workers inherit it. | The 300-step pre-smoke on ids 0:1000 died with `OSError: Too many open files`. `TicWindowDataset` keeps one memory map per episode, and Spiderman's soft limit is 1,024 (hard 1,048,576), so 2,000 episodes could never load. The rerun passed at 1.55 updates/s and 23.3 GB, with recovery checkpoint, snapshot, probe, readback and a resume to step 310. | Going straight to the real gates, which would have failed on the full corpus at about 05:30. | RC 02:55, 03:00; `doom_data.py:198`; commit `1313e22` |
| 04:05, accepted 04:40 | Set the shifted-control margin floor to 2 dB (`MIN_SHIFT_MARGIN_DB`), together with MAE at most 5e-3, p99 at most 2e-2 and complete shard coverage. | The real 4-channel validation pre-run re-encoded bit-identically (MAE 0) yet cleared by only 2.91 dB: unshifted 25.46 dB, ±1 rows 22.55 and 22.53 dB, ±4 rows 20.6 and 20.5 dB. The one-tic copy floor is 21.6 dB, so per-tic motion bounds the margin on slow footage. Astra accepted the set, noting that these are sampled checks, not proof. | Keeping 3 dB, which failed a correctly aligned corpus; bit identity, which rejects harmless rounding; the review's absolute `vae_psnr` thresholds, which Astra rated an alert rather than a certificate. | RC 04:05, 04:40; review §8; `check_latent_alignment.py:47–53,73`; commit `1d9442d` |
| 02:55 | Gate and launch from a second checkout, `$D/repo_launch`, named by `RUN_REPO`, while the encoders keep running from `$D/repo`. | An encoder still ran from `$D/repo`, and the runbook forbids pulling a repository under a running encoder. The gates must run from the checkout they certify, and a resume refuses a changed checkout. `repo_launch` was cloned at `f525782` and stood at `35258f3` for both gate runs. | Pulling `$D/repo` under the encoders; holding the gates until every encoder had exited. | RC 02:55, 04:40; runbook, preconditions and gate section; commits `29d5981`, `261bdb4` |
| 04:05, hardened 04:40 | Revoke per backbone. Each gate run revokes and rewrites only its own certificate entry and keeps its own results file. Gate 0 stops with the file untouched if no interpreter can revoke. | The gates ran staggered: the 4-channel space at 06:12 and the 16-channel space at 08:25, while the U-Net trained. The launcher refuses a resume that differs from its entry, so deleting the U-Net entry would have blocked its resumes. Astra's fourth pass found that a failed revocation deleted the whole file. | One certificate rewritten by every gate run; revocation by deleting the file. | RC 04:05, 04:40; runbook, gate section; commits `dbaede3`, `32aef39` |
| 03:00 | Split the rest of 4-channel shard 1 over two cards. `enc-train1` was stopped with `tmux kill-session`, leaving no orphan. `enc-tailA` took odd ids 1327 to 1661 on GPU 2 and `enc-tailB` odd ids 1663 to 1999 on GPU 1, from the same `$D/repo` code into staging directories, merged into `arenas/` as shards 07 and 08. | 337 odd ids were left at 70 episodes/h, a finish near 07:40. The tails, 168 and 169 episodes, merged at 05:22 without duplicates. The gate re-encoded shards 07 and 08 bit-identically, with margins 4.33 and 3.58 dB. | Letting one encoder finish near 07:40, which would have completed the corpus about 2.3 hours later (derived from 07:40 against 05:22). | RC 03:00, 06:30; `$D/GATES.txt` |
| 06:30 | Encode Superman's 2000:6000 sets under a separate tree, `$D/ext_2000_6000/latents_arnold_dense_pertic{,_sd35}/arenas`. The 0:2000 directories stay frozen for the paper rows and are merged later, deliberately. The uploader reads the new tree and publishes it to its own Hub folders, `sd15/arenas_2000_6000` and `sd35/arenas_2000_6000`. | `gate_certificate.encoder_records` pins every `encode_meta_*.json` in the training directory. Phase 2 would have written new shard logs into the same `arenas/` directories, so every later resume would have refused. The first orchestrator was stopped (supervisor only) and `superman_encode_orch2.sh` took over. | Writing 2000:6000 into the certified directories, the original plan; sharing Hub folders, where Superman's shard logs would overwrite those of the 0:2000 sets. | RC 06:30, 11:30; steward brief, rules; commit `0a168c2` |
| 03:00 | Regenerate the evaluation split files. The 4-channel `split_arenas_678.json` was moved aside as `.ids0_60.old.json` and rebuilt with `--refuse-worker-first`. The SD 3.5 evaluation directory received its val, test and `arenas_678` split files. The canonical control table was copied to `raw_arnold_dense/`, where the gates read it. | The old file still named ids 0:60 while the directory held 60:120, the replacement declared before any score. The SD 3.5 directory had no split files. | Scoring from the stale 0:60 list, 51 of whose 60 episodes are worker-first. | RC 00:50, 03:00; review H1; `release/dense_split.json` history |
| 09:10, revised 11:30 | Evaluation cadence. At 09:10: the full read set at 10k snapshots and a light live-h1 read at the 5k checkpoints, on the run's own card. At 11:30 (Rohan): reads move to GPU 3, which is checked idle before each read and never held between reads; the full set runs at every 5k checkpoint and snapshot, and the sampler sweep at 10k and 50k. | A full read on the shared card cost about 2,500 updates. The U-Net ran at 1.78 to 1.83 updates/s alone and 0.79 to 0.99 while the 5k read shared GPU 1 for 50 minutes, by `log.jsonl` timestamps. SD 3.5 holds 39.5 GB of GPU 2, and its evaluator's footprint is unmeasured. | Reads on the training card, as the steward brief first specified; holding GPU 3 between reads on a shared server. | RC 09:10, 11:30; steward brief, 5k/10k/20k section |
| 09:10, extended 11:30 | W&B. At 09:10: every run streams live to W&B, and the pinned rows get `tools/wandb_tail.py` sidecars (tmux `wb-unet` and `wb-sd35`, environment `~/wbtail`, checkout `$D/repo_tools`) that replay `log.jsonl` and every steward JSON to `doomdit-nexttic`. At 11:30: W&B is on by default in the trainer, with `--no-wandb` only for the gates' fit, smoke and resume; evaluators log to a grouped `<run>-eval` run; the sidecar serves only runs pinned before that code lands. | Rohan asked to watch loss and evaluation curves live. The certificate pins the trainer, so the running rows could not gain a flag. W&B forbids two writers on one run id and cannot reuse a deleted id, so the U-Net sidecar writes run id `040-unet-nexttic-r1`. `wandb` is not yet installed in the `doom` and `wanenc` environments, and that install is deferred while the runs are live. | Restarting the certified runs with a trainer flag; installing `wandb` into the training environments mid-run. | RC 09:10, 11:30; `CLAUDE.md`, Experiment tracking; `tools/wandb_tail.py:162`; commits `3538604`, `d5989d8`; worker `wandb-native` |
| 11:30 | After Spiderman's `/home` filled, keep every cache and upload path on `/sata2`. The uploader restarted from `$D/repo_tools` with its caches there, and uploads use `HF_XET_CACHE` on `/sata2`. Only our regenerable caches were pruned, 18 GB. | `/home` (3.4 TB, shared) reached 0 bytes free at about 10:50. Both runs write only to `/sata2` and kept training. The steward's 20k upload failed once in the xet cache and succeeded with the cache moved. The remaining usage is Rohan's other projects (`~/data` 498 GB, `~/perseve` 258 GB) and other users. | Deleting or moving another project's data without Rohan. | RC 11:30 |
| 09:10, open | Launch without two instruments from review 7.2 and report them as pending at every read: the directional check (swap TURN_LEFT and TURN_RIGHT on turning windows and require the predicted frame to shift the opposite way) and the rollout motion ratio. | The certified code at `35258f3` has neither. TURN_LEFT appears only in the recorder, encoder, IDM and alignment scripts, and `rollout_eval.py` scores copy-seed but no motion ratio. `motion_audit.py` is not called by `after_nexttic.sh` and targets the old stride-four decoded layout (review M4). Both are evaluation instruments, not training contracts, and the review settled that direction is assessed from 10k, not at 300 steps. | Holding both launches until the instruments existed; letting the steward improvise them. | RC 09:10; review §7.2, §8 and M4; steward brief, 5k/10k/20k section, step 3 |

The total of 21 defects closed over the four rounds comes from the main session's 08:50 EDT status to Rohan; the RC entries do not state the total. They account for it as the review's seven findings in round one, then Astra's eight, three and three confirmed defects in rounds two to four (derived: 7 + 8 + 3 + 3). The open-file blocker is counted separately. The code reviews did not catch it: it depends on 2,000 real episodes against the host's soft limit, and it surfaced because a 300-step pre-smoke ran on real data before the gates. [RC 2026-09-23 00:20, 00:50, 02:55, 03:00, 04:05.]

The margin floor was lowered after the pre-run failed it, so its justification has to rest on the measurement rather than on the outcome. The failing shard was bit-identical on re-encode, which rules out a row misalignment independently of the margin. A real misalignment also moves the MAE by orders of magnitude, and the MAE and p99 tolerances test that separately. On the certified corpora the lowest margin was the 4-channel validation shard's 2.91 dB. [RC 2026-09-23 04:05; `check_latent_alignment.py:36–38,47–53`; `$D/GATES.txt`.]

Two of the launch night's gaps remain open at the cutoff, and both concern evaluation, not training. The directional check matters most. The probe in section 3.12 shows that the output depends on the newest control, but not that a left turn moves the image left. Without the motion ratio, a rollout that beats copy-seed in PSNR cannot be told apart from one that drifts toward a smoothed static frame. The main session's 08:50 EDT status offered to write both before the 20k read, and no decision on that offer was recorded before the cutoff. [RC 2026-09-23 09:10; main session status, 08:50 EDT.]

## 3.13 Sep 24 to 25: the three rows to 100k, the U-Net final, and the closed-loop findings

Sections 3.13, 3.14, 4.11 and 5.9 were appended after section 4.10 on 2026-09-25, so that no existing section changed. This section belongs to part 3 and follows the reading rules of section 3.1 and the protocols of section 3.12. Its source cutoff is main at `614a7a8`, which includes the 2026-09-25 23:00 EDT entry of `RESEARCH_CONTEXT.md` and its section 0 as rewritten at 19:30 EDT that day. It covers every entry from 2026-09-23 16:30 EDT to that one, and it applies the corrections of the 23:15 EDT entry, which reached main after this branch was cut (`f331107`). The SD 3.5 110k read was running at the cutoff, so the series below ends at 105k. The run stewards (`run-steward-2`, then stewards 6 and 7 and their automations `stw6-arrive`, `stw6-queue` and `stw7-dirwait`) or the main session read each model number on Spiderman. This addendum copies the number from the RC entry named in brackets and contacted no server. Where two entries disagree, the later one is used and the disagreement is stated. “Not recorded” marks a read that ran but whose value no RC entry states. `$D` means `/sata2/data/rnagabhi/doom` on Spiderman, as in section 3.12.

**Read protocol:** unless a table says otherwise, every teacher-forced read repeats the section 3.12 protocol. It uses 512 windows from validation ids 6000:6100, seed 0, identical at every step and for every row. Sampling is 10-step DDIM with linear spacing, eta 0 and clean context. Each latent space uses its stock decoder, `sd-vae-ft-mse` for the 4-channel rows and the SD 3.5 autoencoder's own decoder for SD 3.5. Targets are raw frames. Rollouts are 16 of 256 tics from the same validation ids, seed 0 unless stated, scored against raw frames at 4, 32, 64, 128 and 256 tics. A full read is the teacher-forced set at h1 and h4 for live and EMA weights, the rollouts and the probe. A light read is a one-tic teacher-forced read only, which takes about five minutes; the recorded light reads report the EMA. [RC 2026-09-23 19:00, 19:45, 21:30; 2026-09-24 11:30; `.claude/analyses/run-steward-brief-2026-09-23.md`, 5k/10k/20k section, step 1.]

**Reference numbers:** the windows are identical for all three rows, and persistence scores the raw last frame against the raw target, so the same references serve every row. One-tic persistence is 21.566 dB and 0.203 LPIPS. Four-tic persistence is 19.213 dB and 0.354 LPIPS. The entries round these to 21.57 / 0.203 and 19.21 / 0.354. Copy-seed, which holds the last real seed frame, scores 18.93 / 17.47 / 17.60 / 17.80 / 17.52 dB at 4 / 32 / 64 / 128 / 256 tics, with LPIPS 0.317 / 0.487 / 0.505 / 0.502 / 0.520. Every gain below is a difference against these numbers. A gain the entry states is copied. A gain marked derived is computed here from the rounded PSNR and can differ by 0.01 dB from an unrounded computation. [Section 3.12, teacher-forced and rollout tables; RC 2026-09-24 10:30, 2026-09-25 01:40.]

**The U-Net stopped at 200k.** Rohan set 200k as the matched step count for every row at 16:20 EDT on 2026-09-24, because the U-Net had saturated. Its EMA read 22.42 to 22.46 dB and 0.182 to 0.179 LPIPS at h1 from 150k to 180k, and its validation loss had stayed in a 0.153 to 0.155 band since 151k. A server-side waiter stopped `train-unet-nexttic` at 17:09 EDT after step 200150, once `snap_0200000.pt`, `0200000.pt` and step 200100 were on disk, and recorded the reason in `resumes.log`. The run logged no skipped update. Its last validation loss was 0.1509. [RC 2026-09-24 16:20, 17:45.]

The U-Net validation loss fell from 0.1713 at 48k and 0.1707 at 50k to 0.1672 at 64k to 66k, 0.1648 at 79k and 0.1544 at 154k, where it flattened (0.1543 from 154k to 157k, 0.154 at 160k). The run held 1.80 updates/s at 154.9k with no skips, excursions or relaunches. The steward of the night of Sep 23 was killed by the harness watchdog at about 01:50 EDT and its replacement started at 09:33, so no read ran in that gap and the recovery checkpoints 105k to 135k were pruned before anyone read them. The snapshots 100k to 150k survived. The later backfill made the read series unbroken from 5k to 200k: full reads at every snapshot the stewards held, and light reads at the others. [RC 2026-09-23 16:30, 19:00, 21:30; 2026-09-24 09:45, 10:30, 11:30, 21:55.]

| Update | Read | Live h1 PSNR / LPIPS | EMA h1 PSNR / LPIPS | EMA h1 gain (dB) | Live h4 PSNR / LPIPS | EMA h4 PSNR / LPIPS | EMA h4 gain (dB) | Source (RC) |
|---:|---|---|---|---:|---|---|---:|---|
| 30k | Full | 21.83 / 0.226 | Level with live; not recorded | Not recorded | 20.52 / 0.301 | Not recorded | Not recorded | 09-23 16:30 |
| 40k | Full | 21.88 / 0.218 | 21.95 / 0.212 | +0.38 | 20.44 / 0.295 | 20.64 / 0.274 | +1.43 | 09-23 16:30 |
| 45k | Full | 21.97 / 0.218 | 22.03 / 0.208 | +0.46 (derived) | 20.68 / 0.286 | 20.72 / 0.269 | +1.51 (derived) | 09-23 19:00 |
| 50k | Full | 21.97 / 0.215 | 22.08 / 0.205 | +0.51 | 20.62 / 0.286 | 20.81 / 0.263 | +1.59 | 09-23 19:00 |
| 55k | Light | Not run | 22.12 / 0.202 | +0.56 | Not run | Not run | Not run | 09-23 19:45, 21:30 |
| 60k | Light | Not run | 22.17 / 0.200 | +0.60 (derived) | Not run | Not run | Not run | 09-23 21:30 |
| 65k | Light | Not run | 22.20 / 0.198 | +0.63 (derived) | Not run | Not run | Not run | 09-23 21:30 |
| 70k | Full | Not recorded | 22.22 / 0.197 | +0.65 (derived) | Not recorded | 20.99 / 0.251 | +1.78 (derived) | 09-23 21:30 |
| 75k | Light | Not run | 22.25 / 0.195 | +0.68 (derived) | Not run | Not run | Not run | 09-23 21:30 |
| 90k | Full | Not recorded | 22.30 / 0.191 | +0.73 (derived) | Not recorded | 21.14 / 0.242 | +1.93 (derived) | 09-24 09:45 |
| 110k | Light | Not run | Not recorded | +0.79 | Not run | Not run | Not run | 09-24 21:55 |
| 150k | Full | 22.27 / 0.197 | 22.42 / 0.182 | +0.86 | 20.96 / 0.257 | 21.24 / 0.232 | +2.03 | 09-24 10:30 |
| 160k to 175k | Full and light | Not recorded | 22.45 / 0.180, flat | +0.88 (derived) | Not recorded | 21.27 / 0.228, flat | +2.06 (derived) | 09-24 15:15 |
| 180k | Full | Not recorded | 22.46 / 0.179 | +0.89 (derived) | Not recorded | Not recorded | Not recorded | 09-24 15:55 |
| 190k | Full | Not recorded | 22.47 / 0.178 | +0.90 (derived) | Not recorded | 21.31 / 0.225 | +2.10 | 09-24 17:45 |
| 195k | Light | Not run | 22.48 / 0.177 | +0.91 (derived) | Not run | Not run | Not run | 09-24 17:45 |
| 200k | Full | 22.29 / 0.186 | 22.49 / 0.177 | +0.92 | 20.97 / 0.244 | 21.33 / 0.224 | +2.11 | 09-24 18:40 |

This table establishes that the U-Net's EMA passed the live weights by 40k and then led them at every full read, reversing the early order of section 3.12. The EMA crossed one-tic persistence in LPIPS between 50k (0.205) and 55k (0.202), after crossing it in PSNR by 40k. Its one-tic gain then rose slowly, by 0.41 dB from 50k to 200k, while its four-tic gain rose by 0.52 dB (derived). From 160k on the EMA moved by 0.04 dB at h1 and 0.06 dB at h4 (derived). The 200k EMA read, 22.49 dB and 0.177 at h1 and 21.33 dB and 0.224 at h4, is the row's final teacher-forced result at 10 steps on seen-map validation windows.

The table does not establish the paper-protocol number, which needs a fixed step count and the chosen decoder (see the sweep below and section 3.12). All windows come from the four training maps. The row's behaviour on the 26 other maps is in section 3.14, and it differs sharply.

Rollout PSNR in dB at 200k, 16 rollouts of 256 tics:

| Tics | EMA 200k | Live 200k | Copy-seed | EMA minus copy-seed (derived) |
|---:|---:|---:|---:|---:|
| 4 | 21.28 | 20.75 | 18.93 | +2.35 |
| 32 | 19.35 | 18.61 | 17.47 | +1.88 |
| 64 | 18.24 | 17.84 | 17.60 | +0.64 |
| 128 | 18.28 | 18.16 | 17.80 | +0.48 |
| 256 | 17.81 | 15.87 | 17.52 | +0.29 |

The EMA rollout history reached this point in steps. At 50k the EMA's LPIPS beat copy-seed at 32 and 64 tics for the first time (0.413 against 0.487 and 0.484 against 0.505). At 70k the EMA sat at or above copy-seed PSNR at every horizon to 256 tics (17.71 against 17.52) and beat it on LPIPS to 128 tics. At 150k it beat copy-seed in PSNR through 128 tics (18.92 against 17.80) and trailed by 0.16 dB at 256 (17.36 against 17.52), with the same shape in LPIPS. At 180k it led by 0.60 dB at 256 tics with LPIPS level. At 200k it led at every horizon. [RC 2026-09-23 19:00, 21:30; 2026-09-24 10:30, 15:55, 18:40.]

The live rollouts behaved differently. The 256-tic live point read 17.8, 15.4 and 16.4 dB at 20k, 30k and 40k. It fell to 14.75 dB at 180k after 18.18 dB at 170k, where the live rollout had led copy-seed at every horizon. At 200k it read 15.87 dB. Scored per rollout at 200k, 4 of 16 live rollouts end below the line 3 dB under the EMA mean at 256 tics, two of them under 8 dB and none blank. The EMA has 2 of 16 below that line, the lowest at 13.6 dB. Across the full reads from 160k to 200k the live 256-tic point varies by 3.4 dB from checkpoint to checkpoint and the EMA's by 0.9 dB. [RC 2026-09-23 16:30; 2026-09-24 15:15, 15:55, 18:40; the line's definition in 2026-09-25 01:40.]

These results establish that at 200k the released EMA weights of the U-Net hold a PSNR lead over copy-seed for 256 tics on these 16 validation rollouts. The lead shrinks from 2.35 dB at 4 tics to 0.29 dB at 256. They also establish that the live weights are the less stable closed-loop generator, by per-rollout count and by read-to-read variance. They do not establish a perceptual lead at 256 tics at 200k, because the 200k EMA rollout LPIPS is not recorded. With 16 rollouts and no interval, a 0.29 dB lead at 256 tics is not resolved, and the 160k-to-200k spread of the EMA point, 0.9 dB, is three times that lead (derived).

**Sampler sweep:** live weights, h1, the same 512 windows, DDIM eta 0, linear spacing unless the cell names trailing; the 10k column is section 3.12's. The 100k sweep ran, and its entry records only that the pattern held and that 4-step PSNR had saturated by 100k. Cells with one value were recorded without a spacing split. Persistence is 21.566 dB / 0.203. Sources: RC 2026-09-23 19:00 (50k); 2026-09-24 10:30 (150k), 15:15 (100k), 18:40 (200k).

| Steps | 10k PSNR / LPIPS | 50k PSNR / LPIPS | 150k linear / trailing | 200k linear / trailing |
|---:|---|---|---|---|
| 4 | 21.567 / 0.376 | 22.03 / 0.317 (trailing LPIPS 0.285) | 22.28 / 0.285; 22.31 / 0.258 | 22.35 / 0.272; 22.37 / 0.245 |
| 8 | 21.552 / 0.282 | 22.01 / 0.231 | 22.30 / 0.210; 22.29 / 0.204 | 22.34 / 0.199; 22.32 / 0.193 |
| 16 | 21.371 / 0.239 | 21.86 / 0.196 | 22.18 / 0.180 | 22.20 / 0.171 |
| 50 | 21.134 / 0.221 | 21.65 / 0.184 | 22.02 / 0.168 | 22.04 / 0.162 |

This table establishes that the perception-distortion trade of section 3.12 persisted through training. At every checkpoint more steps lowered PSNR and improved LPIPS. From 50k on, trailing spacing matched linear from 8 steps and helped only at 4 steps. The step count at which the model beats persistence in LPIPS fell as training went on: 16 steps at 50k (0.196), 16 steps at 150k, and 8 steps at 200k. The 150k entry adds 8 steps with trailing spacing, where the rounded values are 0.204 against 0.203, so that crossing rests on unrounded values. At 200k the 8-step linear point, 22.34 dB and 0.199, beats persistence on both metrics. Every cell at 150k is 0.15 to 0.3 dB and 0.016 to 0.03 LPIPS better than at 50k, and the perceptual gain from 10k to 50k was about 0.04 at every step count. At 200k the 10-step read flatters PSNR against 50 steps by about 0.25 dB (22.29 against 22.04, derived), less than the 0.36 dB of section 3.12 at 10k. [RC 2026-09-23 19:00; 2026-09-24 10:30, 18:40.]

The sweep does not establish the best sampler for rollouts, because it is teacher-forced at one tic, and it covers the U-Net only. The SD 3.5 sweeps planned at 50k and 100k are not recorded in the log through the cutoff.

**SD 3.5 from 5k to 105k.** SD 3.5's validation loss fell from 0.1115 at 10k to 0.1048 at 20k, 0.0973 at 42k, 0.0950 at 52k, 0.0926 at 65k to 67.5k, 0.0913 at 77.7k, 0.0897 at 86.5k and 0.0879 at 103.5k. It ran at 0.50 to 0.55 updates/s whenever its card and host were not contended. A throughput check on 2026-09-24 found GPU 2 at 100 percent utilisation, 0.53 updates/s median, the CPU 93 percent idle, no I/O wait and the data disk 13 percent busy. The run is therefore compute-bound, with gradient checkpointing costing about a third, and only a change to the certified run would speed it up. On the evening of 2026-09-25 other users' CPU-heavy jobs raised the host load to 70 on 64 cores and cut SD 3.5 to 0.32 updates/s; the load fell to 17 at about 22:45 EDT and the run returned to 0.50 to 0.54 updates/s. The reads at 20k, 25k, 30k, 35k, 45k and 60k ran, but their teacher-forced values are not recorded. [RC 2026-09-23 16:30, 19:00, 21:30; 2026-09-24 09:45, 15:15, 21:05, 22:50; 2026-09-25 04:30, 09:20, 19:00, 22:15, 23:00.]

| Update | Live h1 PSNR / LPIPS | EMA h1 PSNR / LPIPS | EMA h1 gain (dB) | Live h4 PSNR / LPIPS | EMA h4 PSNR / LPIPS | EMA h4 gain (dB) | Source (RC) |
|---:|---|---|---:|---|---|---:|---|
| 5k | 21.30 / 0.259 | Not recorded | Not recorded | 19.73 / 0.381 | Not recorded | Not recorded | 09-23 19:00 |
| 10k | 21.60 / 0.220 | 15.6 dB, warming | Not recorded | 19.88 / 0.335 | Not recorded | Not recorded | 09-23 19:00 |
| 15k | 21.91 / 0.211 | 20.4 dB, warming | Not recorded | 20.32 / 0.317 | Not recorded | Not recorded | 09-23 21:30 |
| 40k | 22.37 / 0.173 | 22.33 / 0.173 | +0.76 (derived) | 20.67 / 0.264 | Not recorded | Not recorded | 09-24 11:30 |
| 50k | 22.51 / 0.170 | 22.56 / 0.163 | +0.99 | 20.77 / 0.265 | 20.88 / 0.245 | +1.67 (derived) | 09-24 15:15 |
| 55k | 22.58 / 0.165 | 22.64 / 0.159 | +1.07 | 20.86 / 0.253 | 20.99 / 0.239 | +1.77 | 09-24 17:45 |
| 65k | Not recorded | 22.77 / 0.154 | +1.21 | Not recorded | 21.07 / 0.231 | +1.86 (derived) | 09-24 23:30 |
| 70k | 22.63 / 0.161 | 22.83 / 0.152 | +1.27 | 20.60 / 0.256 | 21.15 / 0.226 | +1.94 | 09-25 01:40 |
| 75k | 22.73 / 0.154 | 22.88 / 0.150 | +1.31 | 20.96 / 0.238 | 21.19 / 0.223 | +1.98 | 09-25 04:30 |
| 80k | 22.70 / 0.156 | 22.92 / 0.148 | +1.35 | 20.90 / 0.239 | 21.25 / 0.219 | +2.04 | 09-25 09:20 |
| 85k | 22.83 / 0.157 | 22.96 / 0.146 | +1.39 | 21.16 / 0.233 | 21.30 / 0.216 | +2.09 | 09-25 10:05 |
| 90k | 22.78 / 0.149 | 23.00 / 0.144 | +1.43 | 21.02 / 0.228 | 21.32 / 0.214 | +2.11 | 09-25 13:45 |
| 95k | 22.83 / 0.150 | 23.04 / 0.142 | +1.47 | 21.10 / 0.230 | 21.35 / 0.212 | +2.14 | 09-25 18:40 |
| 100k | Not recorded | 23.06 / 0.142 | +1.49 | Not recorded | 21.39 / 0.210 | +2.18 | 09-25 18:40, 19:00 |
| 105k | Not recorded | 23.07 / 0.141 | +1.51 | Not recorded | 21.41 / 0.208 | +2.20 | 09-25 21:00 |

The 40k entry states +0.80 dB, which is the live gain (22.37 against 21.566); the EMA gain in the table is derived. This table establishes that SD 3.5's EMA teacher-forced quality improved or held at every recorded read from 50k to 105k, at both horizons, while the live-weight rollouts below collapsed twice. The live weights did not improve monotonically: their one-tic PSNR dipped at 80k and 90k and their four-tic PSNR at 70k, 80k and 90k (derived). By 40k SD 3.5 matched the U-Net's 150k one-tic PSNR (22.33 to 22.37 against 22.42) with better LPIPS (0.173 against 0.182), at about a quarter of the updates. At 50k its one-tic EMA read, 22.56 dB / 0.163 and a gain of +0.99 dB, already beat the U-Net's final 22.49 dB / 0.177 and +0.92 dB on both metrics (derived). Section 0 of `RESEARCH_CONTEXT.md` first dated that passing to 40k, which holds for LPIPS (0.173 against 0.177) but not for PSNR (22.33 EMA and 22.37 live against 22.49); the 23:15 entry corrects it to LPIPS at 40k and both numbers at 50k. Its four-tic EMA gain passed the U-Net's final +2.11 dB at 95k (+2.14). At 105k the EMA read 23.07 dB / 0.141 at h1 and 21.41 dB / 0.208 at h4, the best of the series, with gains of +1.51 and +2.20 dB. Its EMA led its live weights at every read from 50k on where both were recorded. [RC 2026-09-24 11:30, 17:45; 2026-09-25 21:00, 23:15; RC section 0.]

The table does not rank the rows under the paper's protocol. SD 3.5 decodes through a 16-channel autoencoder that reconstructs the validation frames better than the 4-channel one (about 27.5 against 23.5 dB unshifted in the gate of section 3.12), so part of its lead can be decoder quality rather than dynamics. The comparison at matched steps waits for SD 3.5 at 200k. All windows are seen-map validation windows.

Rollout PSNR in dB, 16 rollouts of 256 tics, seed 0 unless stated. The 40k and 45k rows record the 256-tic live point only. Sources: the RC entry of each update in the table above; seed-1 reruns from RC 2026-09-24 15:55 and 2026-09-25 02:30; 105k from RC 2026-09-25 21:00 and 22:15. The 105k seed-1 EMA rerun drew different validation windows and is scored against its own copy-seed row.

| Update | Weights | 4 | 32 | 64 | 128 | 256 |
|---:|---|---:|---:|---:|---:|---:|
| Reference | Copy-seed | 18.93 | 17.47 | 17.60 | 17.80 | 17.52 |
| 40k | Live | Not recorded | Not recorded | Not recorded | Not recorded | 17.12 |
| 45k | Live | Not recorded | Not recorded | Not recorded | Not recorded | 17.00 |
| 50k | Live, seed 0 | 20.74 | 17.24 | 14.71 | 13.38 | 8.98 |
| 50k | Live, seed 1 | 20.16 | 17.93 | 15.30 | 14.60 | 10.13 |
| 50k | EMA | 20.45 | 18.51 | 18.15 | 17.66 | 17.82 |
| 55k | Live | 20.45 | 18.89 | 18.47 | 18.09 | 17.21 |
| 55k | EMA | 20.42 | 18.46 | 18.08 | 18.23 | 16.82 |
| 65k | EMA | 20.64 | 18.77 | 18.39 | 17.97 | 17.54 |
| 70k | Live, seed 0 | 20.72 | 15.20 | 15.29 | 13.69 | 11.11 |
| 70k | Live, seed 1 | 20.57 | 15.62 | 14.68 | 12.96 | 10.95 |
| 70k | EMA | 20.71 | 18.85 | 18.42 | 17.89 | 16.84 |
| 75k | Live | 21.30 | 18.55 | 17.43 | 17.85 | 17.63 |
| 75k | EMA | 20.92 | 18.77 | 18.02 | 18.31 | 18.54 |
| 80k | Live | 20.75 | 18.67 | 17.77 | 18.13 | 18.26 |
| 80k | EMA | 21.15 | 18.94 | 18.43 | 18.65 | 17.94 |
| 85k | Live | 21.28 | 18.42 | 17.66 | 17.45 | 16.99 |
| 85k | EMA | 21.02 | 19.05 | 18.44 | 18.10 | 17.44 |
| 90k | Live | 21.38 | 18.78 | 18.35 | 18.51 | 17.71 |
| 90k | EMA | 21.17 | 19.05 | 18.05 | 17.96 | 17.56 |
| 95k | Live | 21.14 | 19.00 | 17.18 | 18.27 | 17.72 |
| 95k | EMA | 21.26 | 18.77 | 18.18 | 18.54 | 17.55 |
| 100k | Live | 21.06 | 18.46 | 18.36 | 17.90 | 17.36 |
| 100k | EMA | 21.36 | 19.00 | 18.35 | 17.95 | 17.60 |
| 105k | Live | 20.84 | 19.07 | 18.60 | 19.08 | 17.52 |
| 105k | EMA, seed 0 | 21.43 | 18.85 | 18.06 | 18.36 | 16.86 |
| 105k | Copy-seed, seed-1 windows | 19.11 | 18.07 | 17.94 | 17.50 | 17.46 |
| 105k | EMA, seed 1 | 21.26 | 19.93 | 19.23 | 17.31 | 16.49 |

The EMA's 256-tic point ran 18.54, 17.94, 17.44, 17.56, 17.55 and 17.60 dB from 75k to 100k and 16.86 dB at 105k. It sat 0.08 dB under copy-seed at 85k, 0.03 to 0.09 dB above it from 90k to 100k (the 100k entry states +0.09 from unrounded values) and 0.66 dB under it at 105k, while teacher-forced quality kept improving. The 105k point is 0.58 dB under the lowest earlier EMA point of the series, 17.44 at 85k (derived). At 256 tics the 75k LPIPS was 0.611 for live, 0.542 for the EMA and 0.520 for copy-seed, so the best EMA rollout by PSNR still trailed holding the seed frame perceptually at the last horizon. The 65k live rollout is not recorded; the 01:40 entry puts the live 256-tic point at 17.2 to 17.8 dB across 55k to 65k. [RC 2026-09-25 01:40, 04:30, 10:05, 13:45, 19:00, 21:00.]

**The intermittent live collapse and channel 13.** At 50k the live weights produced an absorbing blank-frame failure under autoregression. On seed 1, 8 of 16 live rollouts ended at 2.3 to 3.9 dB at tic 256, a blank or saturated image. Their onsets were spread over tic 4 (one rollout), 36, 43, 95, 135, 225 and two past 128, on maps 2 to 5. Once a rollout reached about 3 dB it never recovered, and 7 of 16 crossed the line 3 dB under the EMA mean and never came back. The decoded latents collapsed the same way, so the latents themselves diverged from about 32 tics. The 50k EMA produced no collapse, and the 45k live rollout had read 17.00 dB at 256 tics. The run logged no NaN, no skipped update and a falling validation loss, so no hold rule applied. [RC 2026-09-24 15:15, 15:55.]

The collapse did not recur at 55k, 60k or 65k, and it returned at 70k. On seed 0, 15 of 16 live rollouts crossed the EMA-minus-3-dB line, 11 were below it at 256 tics, and 11 ended between 5.8 and 10.1 dB with onsets from tic 1 to 137 across maps 2 to 5. The live LPIPS at 256 tics was 0.70 and the decoded PSNR 11.2 dB, so the latents left the data range. Seed 1 reproduced it: 15 of 16 crossed and 10 of 16 were under 10 dB at 256 tics. On the same windows 13 EMA rollouts crossed the line at some tic, none stayed down, and the EMA minimum was 9.7 dB. No live rollout was captured at a fixed point at 75k, 80k, 85k, 90k, 95k, 100k or 105k, although 95k and 100k came close (table below). [RC 2026-09-25 01:40, 02:30, 04:30.]

The per-tic latent statistics of the collapsed rollouts (`stw6_latstats.py`, `steward_70000/latstats_*.json`) located the failure in one channel. The RMS of the predicted latents stayed inside the ground-truth range, 0.98 to 1.19, in almost every collapsed rollout, so the latent neither exploded nor vanished. SD 3.5 latent channel 13, whose per-frame mean sits at about +0.45 in the data (range −0.085 to 0.70), dropped instead to one of two fixed values shared across windows and maps. It sat at −1.39 to −1.50 in 17 of the 22 collapsed rollouts over both seeds, and at −2.60 to −2.73 in 4, the ones with RMS 1.35 to 1.48. Those are the blank frames at 5.8 to 10 dB. On seed 0 channel 13 left its range at the PSNR drop (tics 16/16, 88/90, 137/137, 30/31 and 80/85). On seed 1 six rollouts dropped at tic 1 while channel 13 left later, at tics 2 to 21 or 160 to 208. The channel capture is therefore the end state and not always the trigger. The two EMA reference windows kept channel 13 at or above −0.44 and ended near +0.4. The 21:00 entry corrects the 02:30 reading that this held for the EMA generally: EMA rollouts at 70k dipped under −1 (minimum −1.12) and recovered by tic 256. [RC 2026-09-25 02:30, 21:00.]

| Update | Live channel-13 minimum | Live rollouts | EMA channel-13 minimum | EMA rollouts (number ending under −0.9) |
|---:|---|---|---|---|
| 70k | −1.39 to −1.50 and −2.60 to −2.73 (fixed points) | Collapsed, both seeds | −1.12 (the two reference windows at or above −0.44) | No collapse; dipped under −1 and recovered by tic 256 (0) |
| 75k | Not recorded | No rollout under 10 dB | −0.94 | No rollout under 10 dB (0) |
| 80k | −0.73 | None under 10 dB | −0.54 | None under 10 dB (0) |
| 85k | −1.19; five rollouts under −0.9 for 4 to 30 tics, four recovered | No capture | −1.16; one rollout under −0.9 from tic 104, ending at −1.00 and 13.3 dB | First EMA excursion of that size (1) |
| 90k | −0.96, recovered | No frame under 10 dB | −1.08 in the same map-4 window, under −0.9 for tics 92 to 157, recovered to −0.52 | No frame under 10 dB (0) |
| 95k | −1.26; three rollouts under −1.0, one a 4.5 dB near-blank frame at tics 80 to 83 that recovered to 21.5 dB, one ending at −0.89 and 13.3 dB | Touched the absorbing state and recovered | −0.58 | Clear (0) |
| 100k | Rollout 15 (map 5) under −0.9 from tic 183, at −1.04 with a 10.2 dB frame at 256 | Closest approach to the fixed point since 70k | −1.01 in the recurring map-4 window, recovered to 19.3 dB | Clear (0) |
| 105k | −0.37 | Clean | −1.29; rollout 5 (map 4, the recurring window) under −0.9 from tic 108, ending at −1.12 and 13.4 dB; rollout 2 (map 5) under −0.9 from tic 175, ending at −0.97 and 12.2 dB | No blank frames; the first EMA rollouts to end in the zone (2) |

Sources: RC 2026-09-25 02:30 (70k live), 09:20 (80k), 10:05 (85k), 13:45 (90k), 18:40 (95k), 19:00 (100k) and 21:00 (105k, and the EMA minimum and end-count series from 70k to 105k, which corrects the 02:30 entry's EMA value at 70k). The 75k live channel-13 minimum is not recorded.

These reads establish that SD 3.5's live weights pass through periods in which closed-loop generation falls into an absorbing state in one latent channel, while teacher-forced quality and validation loss keep improving (validation loss reached a new low of 0.0917 at 72k, straight after the 70k collapse). They establish that the fp32 EMA at decay 0.9999 never reached the fixed values at any read from 50k to 105k. It made excursions under −0.9 at 70k, 75k, 85k, 90k, 100k and 105k, most often in one map-4 window, and recovered each time until 105k, where two EMA rollouts ended in that zone without blank frames (next paragraphs). The recorded reading is that noise augmentation at 0.7 over ten buckets does not cover the absorbing state and that the EMA keeps the released model out of it. The U-Net's live weights show a milder form of the same live-versus-EMA gap (earlier in this section). [RC 2026-09-24 15:55; 2026-09-25 01:40, 02:30.]

They do not establish why the live weights enter the state, why it is intermittent, or that the EMA will stay clear to 200k. They do not show that the EMA is a general stabiliser, since only two rows and one EMA setting were observed. The candidate follow-up recorded on 2026-09-24 is a short rollout-consistency post-training that would test whether the fixed point can be removed from the live weights. The paper will report EMA rollouts and state the live-versus-EMA result with the per-rollout counts, and rollout arrays are now kept on every full read. [RC 2026-09-24 15:55.]

**The 105k EMA long-horizon dip and the per-channel mean-shift family.** At 105k the EMA set its teacher-forced bests, 23.07 dB / 0.141 at h1 and 21.41 dB / 0.208 at h4, while its 256-tic rollout point fell to 16.86 dB, 0.66 dB under copy-seed. Two of 16 EMA rollouts ended in the channel-13 excursion zone, the first time for the EMA (table above), and neither produced blank frames. The live weights were clean at 105k, with a channel-13 minimum of −0.37, and held copy-seed at 256 tics (17.52 dB). The probe_v2 means were 0.16 / 0.18 / 0.18 and the directional check read 0.836 for the EMA and 0.820 for the live weights. The pairing rule therefore raised no flag, and the new item was the EMA's ending state. [RC 2026-09-25 21:00.]

A seed-1 EMA rerun at 105k, on different validation windows, reproduced the weakness. It read 21.26 / 19.93 / 19.23 / 17.31 / 16.49 dB at 4 / 32 / 64 / 128 / 256 tics against its own copy-seed of 19.11 / 18.07 / 17.94 / 17.50 / 17.46. The EMA was therefore under copy-seed at 256 tics on both seeds, by 0.66 dB on seed 0 and 0.97 dB on seed 1. On seed 1, 12 of 16 rollouts crossed the steward's 3 dB line at some tic, 3 were below it at 256 tics (maps 4, 5 and 4), and three had frames under 10 dB. One rollout sat in the channel-13 zone for 77 tics and recovered, and none ended there. [RC 2026-09-25 22:15.]

Seed-1 rollout 8 (map 5, episode 6011) ended at 7.1 dB without the channel-13 signature, so its latent statistics were read. Its RMS stayed at 1.00 to 1.17 against a data range of 1.00 to 1.59, so the latent kept its scale. Channel 15's per-frame mean stayed inside its data range through tic 192 and then fell to 0.38 by tic 256, against a range of 1.00 to 1.94, after an earlier 100-tic excursion that had recovered. Rollout 7 (map 4) shifted in channel 13 and rollout 10 (map 4) in channel 8. The recorded reading is that the EMA's long-horizon failures at 105k are per-channel mean shifts with the RMS in scale, in channels 13, 15 and 8. The absorbing state is a family of such shifts, not one channel. [RC 2026-09-25 22:40; `steward_105000/latstats_ema_seed1.json`.]

The first proposal was to track the stability claim on every channel, counting the rollouts whose last frame has any channel mean outside the ground-truth 0.5 to 99.5 percentile band. The steward calibrated that count on the kept rollouts of every read from 55k to 105k before adopting it. The table gives, for each read, the rollouts out of 16 that end outside the band in any channel and in at least three channels.

| Update | Live: any / at least three | EMA: any / at least three |
|---:|---|---|
| 55k | 8 / 8 | 6 / 5 |
| 60k | 5 / 2 | 7 / 5 |
| 65k | 4 / 2 | 5 / 3 |
| 70k | 12 / 12 | 7 / 4 |
| 75k | 5 / 3 | 7 / 5 |
| 80k | 8 / 3 | 5 / 3 |
| 85k | 8 / 4 | 8 / 3 |
| 90k | 8 / 3 | 3 / 1 |
| 95k | 8 / 3 | 4 / 3 |
| 100k | 4 / 3 | 4 / 3 |
| 105k | 6 / 4 | 8 / 5 |

Source: RC 2026-09-25 23:00; `$D/tmp/steward/stw7_endstate.py`.

The EMA sits at 3 to 8 rollouts in every read, and only the 70k live collapse (12 / 12) stands out. The band therefore catches ordinary scene drift, and the all-channel count is too noisy to carry a stability claim. From 110k each read reports, for both weight sets, the 256-tic PSNR against copy-seed, the channel-13 end count (under −0.9) and minimum, and the number of rollouts with any frame under 10 dB, with the all-channel count as context. [RC 2026-09-25 22:40, 23:00.]

These reads establish that at 105k the SD 3.5 EMA is weaker past about 128 tics on two seeds and two window sets, while its teacher-forced quality and directional check are at or near their best. Maps 4 and 5 carry most of the failures. They establish that the EMA's long-horizon failures are per-channel mean shifts with the latent scale intact, unlike the live 70k collapse, which captured channel 13 at a fixed value and produced blank frames.

They do not establish a trend. One read on two seeds cannot separate a checkpoint-level fluctuation from a drift, and the EMA's 256-tic point already moved by 1.1 dB between 75k and 100k (derived). The 110k and 115k reads decide it, and the 110k read was running at the cutoff. The log records the consequence if it is a trend: the paper's claim becomes “the EMA delays the closed-loop failure” instead of “the EMA stays clear”, and the released SD 3.5 checkpoint should be chosen by the rollout series, not by step count. [RC 2026-09-25 21:00, 22:15.]

**Probe sensitivity episodes, and why the maximum misled.** `smoke_probe.py` reports newest-control sensitivity as the largest absolute element of the change in the v-prediction when all 19 bits of the newest executed control are flipped, at fixed noise, context and timestep t = 500, on one batch of four windows. An all-bits-flipped control is not a control the recorder can produce, so the probe measures the response to an off-manifold token, and a one-element maximum is a tail statistic. Commit `caf58bb` added the mean and 99th-percentile fields (called probe_v2 below) on 2026-09-25, and the steward reran the probe on 65k, 70k and 75k. From 80k probe_v2 joined every read. [RC 2026-09-25 04:30, 05:00.]

| Update | Probe maximum, seeds 0 / 1 / 2 | probe_v2 mean, seeds 0 / 1 / 2 | Control-MLP update ratio | Source (RC) |
|---:|---|---|---:|---|
| 40k | 1.98 (one seed) | Not run | Not recorded | 09-24 11:30 |
| 55k | 2.05 (one seed) | Not run | Not recorded | 09-24 23:30 |
| 60k | 2.28 / 2.21 / 2.21 | 0.14, typical of three seeds (p99 0.86) | 0.081 | 09-24 23:30; 09-25 05:00 |
| 65k | 9.13 / 8.30 / 8.48 | Not recorded | 0.155 | 09-24 23:30 |
| 70k | 3.00 / 3.44 / 3.24 | 0.24 to 0.51 (typical 0.36, p99 1.5) | 0.095 | 09-25 01:40, 05:00, 13:45 |
| 75k | 71.0 / 63.9 / 17.2 | 6.9 / 7.6 / 3.2 (p99 64.5 / 55.6 / 14.3) | Ordinary; not recorded | 09-25 04:30, 05:00 |
| 80k | 17.5 / 8.6 / 16.9 | 2.2 / 1.0 / 1.5 | Not recorded | 09-25 09:20 |
| 85k | Not recorded | 2.4 / 3.4 / 2.4 | Not recorded | 09-25 10:05 |
| 90k | 4.3 / 4.5 / 4.2 | 0.63 / 0.78 / 0.63 | 0.082 | 09-25 13:45 |
| 95k | 51 / 46 / 22 | 7.2 / 6.0 / 3.6 | 0.146 | 09-25 18:40 |
| 100k | Not recorded | 0.19 / 0.57 / 0.20 | 0.088 | 09-25 18:40 |
| 105k | Not recorded | 0.16 / 0.18 / 0.18, the lowest since 60k | Not recorded | 09-25 21:00 |

The first episode looked like a learning event. Between 60k and 65k the maximum rose 3.8 times on three seeds, the control-MLP update ratio doubled from 0.081 to 0.155, the other ratios and the per-seed probe loss did not change, validation loss was flat at 0.0926 and no gradient spiked. At 70k the maximum fell back to about 3, so the 65k jump did not persist. At 75k it reached 71.0 on one seed. The probe_v2 fields showed that at 75k the whole difference field moved, with the mean 15 to 30 times its 70k value, rather than one element. The recorded interpretation is that the flipped token is off the data manifold, so the jump is an extrapolation property of the control embedding. It is watched, not acted on, while real control swaps, teacher-forced reads and rollouts stay normal, and the rule flags a future probe jump only if it pairs with a directional drop or a live collapse. [RC 2026-09-24 23:30; 2026-09-25 01:40, 04:30, 05:00.]

At 95k the watched pairing occurred: the highest probe_v2 means of the series coincided with brief live channel-13 excursions. The directional check, the arbiter, showed no drop (EMA 0.836, live 0.828), so no flag was raised. The steward also recorded a pattern for analysis after the runs: probe_v2 was high at the reads at odd multiples of 5k (65k, 75k, 85k, 95k) and low at the reads at multiples of 10k (60k, 70k, 90k, 100k; 80k the exception), and both kinds were probed on recovery files. The maximum misled because it answered a narrower question than the one asked. It said that the response to an impossible control is heavy-tailed at some checkpoints. It did not say that real controls act 20 times more strongly. The directional check below, which swaps real controls, did not drop at any read where the probe rose. The alternation between odd and even multiples of 5k is recorded, not explained, and it did not hold at 105k, an odd multiple whose probe_v2 means were the lowest since 60k (derived from the 21:00 entry). [RC 2026-09-25 05:00, 18:40, 21:00.]

**Directional check protocol:** `directional_check.py` at `999f6b2`, 22 tests, merged at 00:45 EDT on 2026-09-25. It draws turning windows from the validation split whose newest executed control holds exactly one of TURN_LEFT and TURN_RIGHT and no strafe, with the next four tics inside one life. It predicts each window twice from the same noise with the two turn bits swapped in the newest control only, decodes, and measures the horizontal shift of each frame against the decoded last context frame. The shift comes from normalised cross-correlation over ±32 px on rows 48 to 120 of a central crop, sub-pixel by a parabola fit. A left turn slides the scene right, which is a positive shift, and shifts under 1 px count as no motion. `ref_frac` is the fraction of windows in which the ground-truth next frame moves the way the control says; it checks the estimator and the sign convention and is read first. `correct_frac` is the fraction in which the predicted shift reverses under the swap. The motion ratio is the closed-loop four-tic ratio of predicted to true motion from review 7.2, on the same windows; persistence scores 0 by construction. Each run uses 64 windows per direction and takes about three minutes on GPU 3. The requested-action row is refused, since it has no control token to swap. [RC 2026-09-25 00:45, 02:10.]

| Row, update | Weights | correct_frac (left / right) | Recorded-control median shift, left / right (px) | Swapped median shift, left / right (px) | Sign match | Motion ratio (per step) | Source (RC) |
|---|---|---|---|---|---:|---|---|
| U-Net 200k | EMA | 0.867 (0.859 / 0.875) | +19.1 / −19.1 | −19.8 / +22.1 | 0.92 | 0.893 (0.964 / 0.876 / 0.867 / 0.864) | 09-25 02:10 |
| SD 3.5 70k | EMA | 0.805 (0.797 / 0.813) | +17.8 / −18.3; ground truth +18.2 | −18.8 / +21.1 | 0.90 | 0.938 (0.972 / 0.927 / 0.937 / 0.917) | 09-25 02:10 |
| SD 3.5 75k | EMA | 0.8125 (0.797 / 0.828) | +17.8 / −18.8; ground truth +18.2 / −19.1 | −18.6 / +21.2 | Not recorded | 0.941 | 09-25 05:00 |
| SD 3.5 75k | Live | 0.859 (0.844 / 0.875) | +19.4 / −19.0 | −19.6 / +22.0 | Not recorded | 0.920 | 09-25 05:00 |
| SD 3.5 85k | EMA | 0.828 | Not recorded | Not recorded | Not recorded | Not recorded | 09-25 10:40 |
| SD 3.5 90k | EMA / live | 0.836 / 0.828 | Not recorded | Not recorded | Not recorded | Not recorded | 09-25 18:20 |
| SD 3.5 95k | EMA / live | 0.836 / 0.828 | Not recorded | Not recorded | Not recorded | EMA 0.94 | 09-25 18:20, 18:40 |
| SD 3.5 100k | EMA / live | 0.844 / 0.836 | Not recorded | Not recorded | Not recorded | EMA 0.94, live 0.96 | 09-25 19:00 |
| SD 3.5 105k | EMA / live | 0.836 / 0.820 | Not recorded | Not recorded | Not recorded | Not recorded | 09-25 21:00 |

The estimator check passed before any model was read. The ground-truth next frame moved the way the control implies in 0.91 to 0.92 of windows on decoded frames and 0.90 on raw frames for both rows, and 0.914 decoded and 0.898 raw at 75k. The shift estimator and the left-turn-positive convention therefore hold. [RC 2026-09-25 02:10, 05:00, 18:20.]

This table establishes that both rows turn the way the executed control says, with about the true magnitude, and that swapping the turn reverses the predicted motion. At the median the recorded-control shifts sit within 1.2 px of the ground-truth medians recorded at 75k (+18.2 px left, −19.1 px right), and the swapped shifts reverse sign. Neither row learned persistence. This answers the open gap of section 4.10: the probe showed that the output depends on the newest control, and the directional check shows that the dependence has the right sign. The U-Net reversed under the swap a little more often (0.867 against SD 3.5's 0.805 at 70k), and SD 3.5 kept more of the true motion over four tics (0.938 against 0.893). The SD 3.5 EMA value rose from 0.805 at 70k to 0.844 at 100k, the best of the series, and read 0.836 at 105k.

The table does not resolve those orderings. With 128 windows per run, one window moves `correct_frac` by 0.008, and the whole SD 3.5 range from 0.805 to 0.844 is five windows. A binomial standard error at 0.83 over 128 windows is about 0.033, and that of a difference between two such runs about 0.047 (derived). The 0.039 rise from 70k to 100k and the 0.062 gap between the rows are therefore each under 1.5 standard errors of a difference; this ignores that the runs share windows. The check covers turning windows on seen-map validation episodes only, and the motion ratio covers four tics only.

**PixArt-alpha from launch to 106.8k.** The third row, `041-pixart-nexttic`, launched on GPU 1 at 20:32 EDT on 2026-09-24 from `repo_launch2` at commit `f5386f1`. It is the first row on the new code, with native W&B logging and periodic reads on another card. Its gates ran from 20:13 to 20:31 EDT and all passed: audits with 0 mismatches, latent alignment, fit at 1.141 updates/s and 26.8 GB allocated (27.7 GB reserved, no checkpointing), the 300-step smoke (9.4 GB recovery checkpoint, 2.4 GB snapshot), probes, resume and four readbacks. The printed `GATES_LAUNCH pixart` line carries `MB=32 WORKERS=12 STEPS=200000 ACTION_HISTORY=32 EVAL_EVERY=5000 EVAL_DEVICE=cuda:3`. The model has 628 M parameters and receives controls through `--action-inject token`. It trains on the same 2,000 training and 100 validation episodes, and ran at 1.37 updates/s and 28.8 GB at step 100. Its config records `wandb: true` and periodic reads every 5k on `cuda:3` into `eval_<step>/`. [RC 2026-09-24 20:35.]

| Update | Live h1 PSNR / LPIPS | EMA h1 | Live h4 PSNR / LPIPS | Status and notes | Source (RC) |
|---:|---|---|---|---|---|
| 5k | 20.91 / 0.305 | Warming | 19.10 / 0.410 | ok; 22 minutes | 09-24 22:20 |
| 10k | 21.52 / 0.282 | Not recorded | Not recorded | ok; level with the U-Net at 10k | 09-24 23:30 |
| 20k | 21.76 / 0.237 (+0.19) | Not recorded | 20.34 / 0.320 | ok; probe sensitivity 1.56 | 09-25 01:40 |
| 25k | 21.79 / 0.229 | 21.48, nearly warmed | 20.50 / 0.311 | ok | 09-25 02:30 |
| 30k | +0.33 dB gain | +0.14 dB gain | +1.30 dB gain | ok | 09-25 04:30 |
| 65k | Not recorded | Not recorded | Not recorded | Both h4 labels lost to OOM | 09-25 11:45 |
| 70k | Lost | Not recorded | Not recorded | `tf_live_h1`, `tf_ema_h4` and the probe lost to OOM; recovered by hand | 09-25 11:45, 18:40 |
| 70k to 95k | Not recorded | Gain +0.62 to +0.71 | EMA gain +1.73 to +1.86 | 75k to 95k all ok | 09-25 18:40 |
| 95k | Not recorded | Gain +0.71 | EMA gain +1.86 | ok | RC section 0 |
| 105k | Not recorded | Not recorded | Not recorded | ok | 09-25 21:00 |

PixArt's validation loss fell monotonically from 0.196 at 11.2k to 0.1836 at 24k, 0.1758 at 39k, 0.1685 at 61.9k and 0.1610 at 97k. From about 12:40 EDT on 2026-09-25 another user's `extract.py --gpu 1` shared its card, and it slowed from about 1.40 to 0.66 to 0.85 updates/s, about 57 percent of normal. It was back at 1.24 to 1.38 updates/s at 99.5k by 19:00, when that job eased. The CPU load from other users' jobs that evening cut it to 0.6 updates/s from 0.95 at 22:15, and it returned to 0.9 to 1.36 updates/s after the load fell at about 22:45. It stood at 106.8k at 21:00. Its probe_v2 series stopped when the steward's 70k and 75k holds were released for disk space (section 4.11); it had been flat from 45k to 55k. [RC 2026-09-24 22:50; 2026-09-25 01:40, 02:30, 04:30, 09:20, 12:55, 13:15, 13:25, 19:00, 21:00, 22:15, 23:00.]

This table establishes that PixArt trains stably under the matched recipe and tracks the U-Net's curve. It was level with it at 10k and had an EMA one-tic gain of +0.62 to +0.71 dB from 70k to 95k against the U-Net's +0.65 to +0.73 dB at 70k to 90k (section above). Its periodic reads come from the trainer's own evaluator on another card, so they are not the steward's full reads. The log records only their teacher-forced values and the 20k probe; no PixArt rollout or directional check is recorded. Nothing here compares PixArt with SD 3.5 or ranks the three rows. That waits for 200k.

**What the numbers license.** At 10 sampling steps on seen-map validation windows, all three rows beat one-tic and four-tic persistence in PSNR with their EMA weights. The U-Net and SD 3.5 EMA weights also beat it in LPIPS at one tic. The U-Net's EMA at 200k stays above copy-seed in PSNR for 256 tics on 16 rollouts. SD 3.5's EMA did so from 90k to 100k and fell under it at 105k on two seeds. SD 3.5's live weights have an intermittent absorbing state in which latent channel 13 is captured at a fixed value and the frames go blank; its EMA never reached that state, and its own long-horizon failures at 105k are per-channel mean shifts without blank frames. Both measured rows turn the right way under a control swap.

They do not license a ranking of the rows before the matched 200k reads, a result under the paper's step count and decoder, any statement about unseen maps (section 3.14 shows the U-Net losing to persistence on most of them), a perceptual rollout lead at 256 tics, a claim that the SD 3.5 EMA stays stable to 200k, or a causal account of either failure. The 16-rollout reads carry no intervals, and the 5k-series teacher-forced reads carry only independent-window standard errors that understate episode-level uncertainty (section 3.12).

## 3.14 The map-distance generalization study

This section belongs to part 3 and follows section 3.13's conventions. It reports the study Rohan approved at 22:20 EDT on 2026-09-24, as designed in `.claude/analyses/distance-study-design-2026-09-24.md` and as run on Spiderman from 01:25 to 18:20 EDT on 2026-09-25. The frozen files are `results/distance_study/distances_sd1.json`, `distances_pixels.json`, their per-episode CSVs and bootstrap arrays, the per-map score directories under `results/distance_study/scores/040-unet-nexttic_snap_0200000_ema_ddim10/`, and `results/distance_study/figure_unet_h1/` (`stats.json`, `distance_table.md` and the light and dark figures). For this addendum the point statistics were recomputed from the frozen files: the partial and raw Spearman coefficients, the four-tic and LPIPS partials, the pixel-space partial and the rank agreement between spaces all match `stats.json`. The bootstrap intervals and permutation p values are copied, not rerun. [RC 2026-09-24 22:20; 2026-09-25 01:25, 12:00, 14:00, 18:20.]

**Question.** Changliu suggested measuring the distributional distance between each evaluation map's footage and the training footage, and correlating it with the model's quality drop on that map. The study reuses Rohan's NVIDIA method, which treats a test as a weighted cloud of per-frame latents, weights frames by motion and compares clouds by sliced Wasserstein distance. The target sentence is: across N maps, gain over persistence falls with distance from the nearest training map (partial Spearman ρ with a 95% CI). [RC 2026-09-24 22:20; design memo, status line and section 1.]

**Maps.** The 30 primary maps are the four training arenas 2 to 5 as held-out validation episodes; arenas 6 to 8 (`arenas_678`, the training WAD, never trained on); arenas 16 and 17 (`unseen`); the seeded corpus's maps 1 and 9 to 15 (`seen`, which were training maps of the April model but were never trained on by the next-tic rows); and 13 curated campaign maps, 18 to 20, 22 to 26 and 28 to 32 (`unseen2`). That gives 17 arena maps and 13 campaign maps. The seeded corpus's copies of maps 2 to 8 are replications of the distance, not extra points, so the distance file holds 37 map points. The evaluation-map encode (`enc-eval-maps`, log `$D/logs/encode_eval_maps.log`) encoded all 210 seeded episodes per tic in both latent spaces on GPU 3 with the pinned encoder, `$D/repo` at `190c125`, the same canonical table and settings as the training corpora. [RC 2026-09-24 22:20; 2026-09-25 09:40; design memo, section 6, last paragraph; `distances_sd1.json`, `maps`.]

**Distance as run.** Each map's cloud is 4 episodes of 250 target-eligible frames each, at least 32 tics into a life, which is the population the scored windows come from. Four is the smallest per-map episode count, so maps with more episodes average ten random 4-episode subsets, and every cloud has the same size because the finite-sample floor depends on it. The features are SD 1.x latents with the padding rows removed, 5×5-pooled to 192 dimensions. Each frame is weighted by its latent motion, the norm of z at tic t minus z at tic t−1 within a life, plus a closed-form floor that gives the quietest half of the frames exactly a quarter of the weight (`quiet_share` 0.25). The distance is sliced W2 over 1,000 directions from seed 0, shared by every distance, with no time coordinate. The reference is 50 seeded episodes per training map from ids 0:2000, 250 frames each, 25 per motion decile, which is 12,500 frames per map. A disjoint second draw of the same size gives the train-versus-train floor. These are the two reference draws. [`distances_sd1.json`, `config` and `references`; design memo, section 2.]

The primary distance goes to the nearest training map, D(m) = min over k in {2, 3, 4, 5} of SW2(cloud of m, reference of k). The pooled corpus would charge a held-out training map for not resembling the other three. In the memo's one-dimensional example, with training maps at 0, 2, 4 and 6, a held-out draw of map 0 has squared W2 of 14 to the pool while an unseen map at 3 has only 5. The floor band is the range of 40 single-subset train-versus-train distances: 0.022 to 0.093 in the SD 1.x space (mean 0.044) and 0.009 to 0.046 in pixels. Distance uncertainty comes from 200 episode redraws. The pixel space (RGB at 20×15×3) reads exactly the same episodes and tics as the SD 1.x clouds through `--frames-from`, so the spaces differ only in their features. The SD 3.5 latent space waits on its encode pass. [Design memo, sections 1 and 2; `distances_sd1.json` and `distances_pixels.json`, `floor.motion`; RC 2026-09-25 01:25, 14:00.]

**Outcome and statistics.** The outcome is each map's gain over persistence at one tic, the mean of `psnr_raw − persist_psnr_raw` over 256 windows drawn by `eval_tf.draw_windows`. Per window this difference is ten times the log of the persistence-to-model MSE ratio, so motion level cancels to first order. The scored model is the U-Net's 200k EMA. Those are the final weights, so no selection among checkpoints is possible. Scoring uses 10-step DDIM, the stock `sd-vae-ft-mse` decoder, seed 0 and horizons of one and four tics, through `scripts/spiderman/score_distance_maps.sh` from the clean checkout `$D/repo_distance`, at `6a33311` for 32 of the 60 map-horizon reads and at `ca03bad`, which adds only the per-map guard hook, for the other 28. Each map directory holds `metrics.json`, `score_key.txt` (checkpoint SHA-256, variant, split hash, horizon, windows, seed, sampler and decoder) and `provenance.txt`. The covariate is the map's mean persistence PSNR, as a motion proxy. The primary statistic is the partial Spearman coefficient of gain on D after rank residuals on persistence PSNR, with a 95% CI from 10,000 case-bootstrap draws over maps (episodes resampled within maps) and a p value from 10,000 permutations. Leave-one-map-out, within-cluster coefficients, the LPIPS and four-tic gains, an episode-within-map coefficient and the pixel space are secondary. [`stats.json`, `primary`; `scores/.../unseen2_map23_h1/score_key.txt` and `provenance.txt`; design memo, section 3.]

**Pre-declared verdict.** The claim is supported if the primary coefficient is negative with a CI that excludes zero and the sign holds in every leave-one-out refit, within both clusters, for every row and in pixel space (with the two spaces' rankings agreeing at Kendall τ of at least 0.6). The claim is not supported if only the cluster gap exists, if the relation vanishes under motion control, if it appears only in raw PSNR, if the distance fails validation, or if the rows disagree. Causation and maps beyond these arenas and curated campaign maps are never claimed. `paper/make_distance_figure.py` implements the rule, and its `distance_validation` condition is the distances file's `checks.all_pass`. [Design memo, section 5; RC 2026-09-25 11:30; `stats.json`, `verdict`.]

**Pre-registration and freezing.** Commits are on main; times are EDT on 2026-09-25 unless stated.

| Time | Commit | Event |
|---|---|---|
| 09-24 22:52 to 23:30 | `e3df659`, merged at `b3d87bc` | Weighted sliced Wasserstein core and the design memo; suite 1,443 |
| 00:35 to 00:45 | `809f637`, `d8a68cc`, `2f904e9` | `clouds`, `splits` and `distances` subcommands; the resumable per-map scorer; the figure, statistics and appendix-table script |
| 01:19 | `6a33311` | Check (ii) stops judging maps inside the floor band, before any per-map score existed |
| 01:25 | None | The primary test is put to Rohan with the sealed-corpora question and the GPU 3 slot |
| 08:55 (created) and 09:13 (committed) | `01de829` | SD 1.x distances frozen with the code commit and SHA-256 of `distance_study.py`, before any per-map score existed |
| 09:40 | None | The primary test is kept as the memo declares it, under Rohan's “continue with everything”; per-map scoring starts on GPU 3 |
| 11:42 | `7d186dd` | U-Net one-tic scores and the 30-map statistics frozen |
| 13:10 (created) and 13:34 (committed) | `02136c8` | Pixel-space distances frozen with the refreshed statistics |
| 18:14 | `d8e07b8` | U-Net four-tic scores frozen with the refreshed `stats.json` |

Sources: `git log -- results/distance_study distance_study.py paper/make_distance_figure.py scripts/spiderman/score_distance_maps.sh`; the `created` and `code` fields of both distances files; RC 2026-09-24 23:30 and 2026-09-25 01:25, 09:40, 12:00, 14:00, 18:20.

The primary SD 1.x distances were computed and committed before any per-map score existed, and the gate was not changed after the scores were read. The pixel distances were computed at 13:10 EDT, after the one-tic scores were frozen, because the thread-capped chain ran the pixel space second (section 4.11). Their code is unchanged: both files record the same `distance_study.py` SHA-256, `f027fff…`, with `distance_study_modified` false. The pixel arm is therefore blind in its code and parameters but not in time (derived from the two files). “Nothing is booked in the seal” also applies: the scorer read the sealed evaluation corpora once with a fixed checkpoint and wrote nothing to `$R/sealed`, pending Rohan's ruling (section 4.11). [RC 2026-09-25 01:25, 09:40, 11:30.]

**Validation checks before scoring.** Motion arm, the primary arm; values from `checks` in each distances file.

| Check | What it tests | SD 1.x value | Pixel value | Result |
|---|---|---|---|---|
| (i) | The four validation maps sit inside the floor band | Band 0.022 to 0.093; maps 5 / 3 / 2 / 4 at 0.026 / 0.034 / 0.065 / 0.069 | Band 0.009 to 0.046; maps at 0.014 to 0.037 | Pass in both |
| (ii) | Disjoint 4-episode subsets agree within 10% (median over five pairs), and replications land on their primary point; floor-band maps are reported, not judged | Eight campaign maps exceed 10%: 18 0.288, 20 0.202, 26 0.200, 29 0.196, 24 0.195, 22 0.133, 28 0.124, 31 0.108; replications of maps 6, 7 and 8 agree within 0.5%, 0.6% and 4.0% | More maps exceed 10%, up to 0.482 on map 24; replications of 6 to 8 within 0.8% to 8.2% | Fail in both |
| (iii) | The two reference draws rank the maps alike, Spearman at least 0.95 | 0.996 | 0.980 | Pass in both |
| (iv) | Arenas 6 to 8 (training WAD) sit nearer than every campaign map | Max same-WAD 0.282 (map 7) against min campaign 0.140 (map 20); means 0.195 against 0.218 | Max same-WAD 0.147 against min campaign 0.041; means 0.083 against 0.063 | Fail in both |
| (v) | Absolute Spearman of D with Kish effective sample size under 0.4 | −0.022 | 0.112 | Pass; motion arm primary |
| Attenuation | Median bootstrap SD of D over the SD of D across maps; under 0.1 means attenuation under 1% | 0.089 | 0.139 | SD 1.x under 1%; pixels not |

The 09:40 entry counts nine campaign maps failing check (ii) at a median of 11 to 29 percent. The frozen file lists eight (medians 0.108 to 0.288), and the file is used here. The four validation maps disagree across subsets by more (medians 0.146 to 0.604), but they lie inside the floor band, where D is draw noise and no relative tolerance can hold; the toy study had shown 6 to 20 percent pair differences on them. That exemption was committed at 01:19, before any score. [RC 2026-09-25 01:25, 09:40; `distances_sd1.json`, `checks.ii`.]

The recorded reading of the two failures is that they are findings, not measurement faults. Check (iv) fails because the same WAD is not a proxy for closeness in latent space: arena 7's distance to its nearest training map, 0.282, exceeds that of eleven of the 13 campaign maps (derived from the per-map table). Check (ii) fails because four-episode subsets of short, static campaign episodes are noisy, which the ten-subset mean and the 200-redraw bootstrap already absorb; the SD 1.x attenuation ratio of 0.089 says map-level D is precise enough that a correlation with it is attenuated by under 1 percent. Both readings are the main session's, made after the checks failed and before the gate question was ruled. [RC 2026-09-25 09:40, 11:30.]

**Results, U-Net 200k EMA.** All 30 maps at both horizons; values from `results/distance_study/figure_unet_h1/stats.json` as refreshed at 18:14 (`d8e07b8`). A p value of 0.0001 is 1/10,001, the smallest value 10,000 permutations can give.

| Read | n | Spearman ρ | 95% CI | Permutation p |
|---|---:|---:|---|---:|
| Primary: partial on persistence PSNR, one-tic PSNR gain, SD 1.x | 30 maps | −0.734 | [−0.840, −0.320] | 0.0001 |
| Raw (no motion control), one-tic gain | 30 maps | −0.440 | [−0.695, −0.025] | Not computed |
| Raw, one-tic model PSNR instead of gain | 30 maps | −0.369 | [−0.654, +0.079] | Not computed |
| Leave-one-map-out refits of the primary | 30 refits | −0.773 to −0.696 | 30 of 30 negative | Not applicable |
| Within arenas, partial | 17 maps | −0.574 | [−0.856, −0.025] | 0.017 |
| Within campaign maps, partial | 13 maps | −0.378 | [−0.790, +0.354] | 0.200 |
| LPIPS gain, partial | 30 maps | −0.589 | [−0.806, −0.193] | 0.001 |
| Four-tic PSNR gain, partial | 30 maps | −0.795 | [−0.875, −0.458] | 0.0001 |
| Episodes within maps | 342 episodes in 30 maps | +0.088 | Not computed | 0.218 |
| Pixel space, partial | 30 maps | −0.657 | Not computed | Not computed |
| Rank agreement of D, pixels against SD 1.x | 30 maps | Spearman 0.863; Kendall τ 0.701 | Not applicable | Not applicable |

The same file's per-row block reports a second bootstrap of the primary, [−0.841, −0.323]. The entries' intervals differ in the third decimal because the statistics were rerun as maps arrived: 25 maps gave −0.76 [−0.86, −0.36] at 11:30, 30 maps gave −0.734 [−0.842, −0.325] at 12:00, and the refreshed file gives [−0.840, −0.320]. The later file is used. The pixel-to-SD 1.x Spearman of 0.863 is stated as 0.86 in the 14:00 entry and was recomputed here from the two distances files. [RC 2026-09-25 11:30, 12:00, 14:00, 18:20.]

The verdict conditions are negative with a CI excluding zero, pass; leave-one-out sign, pass; within both clusters, pass; every row, pass; pixel space, pass; distance validation, fail. `make_distance_figure.py` therefore prints the pre-declared “does not support”. None of the four diagnostics fired: the relation does not vanish under motion control, does not appear only in raw PSNR, is not only the cluster gap, and no rows disagree. [`stats.json`, `verdict`.]

This table establishes that across these 30 maps the U-Net's one-tic gain over persistence falls with SD 1.x nearest-training-map distance after controlling for persistence PSNR, and that the four-tic gain falls with it more strongly. The sign survives every single-map deletion and holds within each cluster. The relation is stronger with motion controlled than without it (−0.734 against −0.440), and weaker for raw PSNR, whose CI crosses zero, so it lives in what the model adds over persistence rather than in map difficulty. The episode-within-map coefficient of +0.088 places the relation between maps, not between episodes of one map. The pixel space ranks the maps much as the SD 1.x space does and gives a coefficient of the same sign, so the result is not an artifact of one encoder's latent space.

The table does not establish a supported claim under the pre-declared rule, because the distance validation gate failed (the open question below). Three conditions pass more weakly than their names suggest. “Within both clusters” is a sign test, and the campaign coefficient's CI spans zero. “Every row” holds trivially with one row scored. The pixel coefficient has no interval, and the pixel distance's attenuation ratio, 0.139, is above the 0.1 bound. Nothing here speaks to causation or to maps outside these arenas and curated campaign maps.

**Per-map table.** The table below is `results/distance_study/figure_unet_h1/distance_table.md` verbatim, including its caption line. D is the motion-weighted sliced W2 to the nearest training map in the SD 1.x space, ± its bootstrap SD. Nearest is the training map attaining the minimum. n_eff is the Kish effective sample size of the motion-weighted cloud of 1,000 frames. Lives is lives per episode, and Valid the valid-window fraction. Persistence is the map's mean `persist_psnr_raw` at one tic. The gains are in dB with the one-tic 95% bootstrap interval. The LPIPS gain is persistence LPIPS minus model LPIPS, so positive is better. Every map was scored on 256 windows.

Distance D (sliced W2 to the nearest training map, sd1 space, motion weights; ± bootstrap SD), and gain over persistence per map. Primary: partial Spearman −0.73 [−0.84, −0.32] over 30 maps (unet, one tic).

| Map | Cluster | D | Nearest | n_eff | Lives | Valid | Persistence (dB) | unet gain h1 (dB) | unet LPIPS gain | unet gain h4 (dB) |
|---|---|---|---|---|---|---|---|---|---|---|
| val/5 | arena | 0.026 ± 0.005 | 5 | 978 | 8.3 | 0.95 | 19.76 | +1.79 [+1.58, +1.98] | +0.029 | +1.90 |
| val/3 | arena | 0.034 ± 0.007 | 3 | 970 | 8.7 | 0.95 | 21.37 | +0.72 [+0.42, +1.01] | +0.006 | +2.21 |
| val/2 | arena | 0.065 ± 0.019 | 2 | 979 | 5.5 | 0.97 | 21.92 | +0.51 [-0.05, +0.92] | +0.016 | +1.29 |
| val/4 | arena | 0.069 ± 0.026 | 4 | 988 | 2.8 | 0.99 | 22.70 | +0.68 [+0.33, +1.07] | +0.056 | +2.27 |
| arenas_678/8 | arena | 0.115 ± 0.005 | 4 | 970 | 7.2 | 0.96 | 23.20 | -0.92 [-1.28, -0.58] | -0.077 | -0.20 |
| seen/15 | arena | 0.127 ± 0.004 | 3 | 989 | 5.0 | 0.97 | 19.85 | +1.30 [+1.03, +1.45] | -0.079 | +0.49 |
| seen/12 | arena | 0.128 ± 0.008 | 3 | 984 | 5.0 | 0.97 | 21.69 | -0.31 [-0.56, +0.02] | -0.104 | +0.19 |
| unseen/17 | arena | 0.130 ± 0.001 | 3 | 987 | 4.8 | 0.98 | 19.37 | +0.53 [+0.41, +0.65] | -0.155 | +0.49 |
| unseen2/20 | campaign | 0.140 ± 0.016 | 3 | 990 | 1.3 | 1.00 | 22.12 | -0.13 [-0.65, +0.40] | -0.129 | +0.24 |
| unseen2/32 | campaign | 0.143 ± 0.007 | 2 | 994 | 1.4 | 1.00 | 23.55 | -0.68 [-0.83, -0.52] | -0.144 | -0.49 |
| seen/14 | arena | 0.146 ± 0.002 | 3 | 985 | 11.2 | 0.93 | 21.75 | -0.28 [-0.34, -0.18] | -0.109 | -0.27 |
| seen/11 | arena | 0.150 ± 0.002 | 2 | 978 | 11.8 | 0.93 | 20.83 | +0.35 [+0.10, +0.63] | -0.080 | +0.38 |
| seen/13 | arena | 0.156 ± 0.005 | 3 | 978 | 10.2 | 0.94 | 19.37 | +0.79 [+0.33, +1.26] | -0.125 | +0.51 |
| seen/10 | arena | 0.162 ± 0.002 | 3 | 982 | 8.5 | 0.95 | 22.60 | -0.36 [-0.73, -0.06] | -0.069 | +0.88 |
| unseen/16 | arena | 0.164 ± 0.005 | 3 | 992 | 5.4 | 0.97 | 20.03 | +0.77 [+0.65, +0.87] | -0.163 | +0.48 |
| unseen2/28 | campaign | 0.165 ± 0.025 | 4 | 994 | 1.3 | 1.00 | 24.20 | -1.83 [-2.39, -1.06] | -0.161 | -1.96 |
| seen/1 | arena | 0.179 ± 0.001 | 3 | 984 | 21.8 | 0.85 | 21.93 | -0.23 [-0.55, +0.09] | -0.138 | -0.11 |
| seen/9 | arena | 0.180 ± 0.002 | 3 | 988 | 4.5 | 0.98 | 22.58 | -0.06 [-0.45, +0.39] | -0.063 | +0.53 |
| unseen2/24 | campaign | 0.181 ± 0.023 | 3 | 934 | 1.0 | 1.00 | 23.98 | -2.44 [-4.84, -0.36] | -0.107 | -2.58 |
| unseen2/18 | campaign | 0.182 ± 0.040 | 3 | 950 | 1.0 | 1.00 | 22.60 | -1.02 [-2.50, +0.25] | -0.155 | -1.78 |
| unseen2/19 | campaign | 0.182 ± 0.039 | 4 | 946 | 1.0 | 1.00 | 24.64 | -2.35 [-4.75, -0.31] | -0.109 | -2.62 |
| unseen2/25 | campaign | 0.187 ± 0.009 | 4 | 995 | 2.5 | 0.99 | 25.76 | -2.40 [-2.94, -1.65] | -0.130 | -1.84 |
| arenas_678/6 | arena | 0.188 ± 0.010 | 3 | 970 | 8.3 | 0.95 | 20.72 | -0.31 [-1.23, +0.21] | -0.122 | -0.42 |
| unseen2/31 | campaign | 0.210 ± 0.023 | 3 | 980 | 1.0 | 1.00 | 21.66 | -0.19 [-0.98, +0.57] | -0.099 | -0.06 |
| unseen2/22 | campaign | 0.220 ± 0.035 | 3 | 983 | 2.1 | 0.99 | 19.63 | +0.03 [-0.85, +0.96] | -0.159 | -0.34 |
| unseen2/29 | campaign | 0.235 ± 0.033 | 3 | 988 | 6.4 | 0.96 | 18.81 | +0.33 [-0.43, +1.04] | -0.173 | -0.41 |
| unseen2/26 | campaign | 0.273 ± 0.073 | 4 | 985 | 1.3 | 1.00 | 27.23 | -3.98 [-5.40, -2.48] | -0.120 | -3.06 |
| arenas_678/7 | arena | 0.282 ± 0.005 | 3 | 978 | 11.6 | 0.93 | 19.53 | +0.05 [-0.34, +0.38] | -0.084 | -0.85 |
| unseen2/30 | campaign | 0.338 ± 0.006 | 4 | 951 | 1.0 | 1.00 | 20.51 | -0.75 [-1.90, +0.27] | -0.165 | -0.82 |
| unseen2/23 | campaign | 0.371 ± 0.005 | 3 | 994 | 1.1 | 1.00 | 19.60 | +0.17 [+0.10, +0.27] | -0.212 | -0.69 |

This table establishes where the U-Net gains and where it loses. On the four validation maps it gains +0.51 to +1.79 dB at one tic and +1.29 to +2.27 dB at four tics. On the 26 other maps its one-tic gain is negative on 17 and positive on 9: seeded map 15 (+1.30), seeded 13 (+0.79), arena 16 (+0.77), arena 17 (+0.53), seeded 11 (+0.35), campaign 29 (+0.33), campaign 23 (+0.17), arena 7 (+0.05) and campaign 22 (+0.03). Its LPIPS gain is negative on all 26, from −0.063 to −0.212. At four tics 17 of the 26 are again negative (all counts derived from the table). So at one tic on unseen maps the U-Net row mostly loses to copying the last frame on both metrics. It loses most on the campaign maps whose seeded footage is nearly static: the five worst one-tic gains, maps 26, 24, 25, 19 and 28 at −3.98 to −1.83 dB, have persistence of 23.98 to 27.23 dB, and ten of the 13 campaign maps average 1.0 to 1.4 lives per episode. Whether the agent is stuck on those maps needs a look at the frames, which has not been done. [RC 2026-09-25 11:30, 18:20; `distance_table.md`.]

The table does not establish the size of any single map's gain beyond its interval. Several one-tic intervals are wide (campaign 26 [−5.40, −2.48], 24 [−4.84, −0.36], 19 [−4.75, −0.31]), and 256 windows is a small sample of a map. The distance file lists 25 episodes per validation map, 20 per arena 6 to 8, 10 per arena 16 and 17 and per campaign map, and 4 per seeded map 1 and 9 to 15.

**The superseded first look, and the reference lesson.** At 09:50 the main session read the first 16 scored maps against `copy_psnr_raw` and reported gains of +0.09 to +2.43 dB and a raw Spearman of −0.88. That column scores the decoded last latent against the raw target, so it carries the decoder's reconstruction error, and it is not the pre-registered outcome. The pre-registered outcome uses `persist_psnr_raw`, the raw last frame. The 11:30 entry marked the first look superseded and put its gains about 1.4 dB too high. Recomputed from the 30 frozen one-tic `metrics.json` files, `persist_psnr_raw` exceeds `copy_psnr_raw` by 1.36 dB on average, and by −0.08 dB (campaign 23) to +4.04 dB (campaign 26) map by map (derived). The gap is largest on the static campaign maps, so the wrong reference did not shift every map by a constant. It moved the maps against each other, and it turned most unseen-map losses into apparent gains. This is section 1.2's rule applied to a new study: the suffix `_raw` names the reference, not the prediction, and only `persist_psnr_raw` is decoder-free. [RC 2026-09-25 09:50, 11:30; `scores/.../*_h1/metrics.json`.]

**The seen-sidecar incident.** At 10:10 the scorer failed on seeded maps 9 to 15 in `doom_data.check_sidecar_buttons_dtype`, because their buttons column was up to `<U36`, wider than the 19-button executed control. The audit found that `latents_arnold_eval_pertic/seen/ep_00000` to `ep_00027` had been written on Sep 21 by encoder commit `7f7d0b1`, with buttons stored as Arnold's raw request strings of varying width. Even their `<U9` rows carry different button semantics. The 32 `seen` episodes written on Sep 25, and every `unseen` and `unseen2` episode, are `<U19` from the pinned encoder `190c125`; the Sep 24 encode had skipped the 28 because they existed. The 56 old files were moved to `$D/_aborted_safe_to_delete/seen_old28_sep21/` and re-encoded on GPU 3 (`$D/tmp/reencode_seen.sh`, marker `RE_ENCODE_SEEN_OK`). [RC 2026-09-25 10:10.]

The re-encoded latents equal the old ones. For `ep_00000` and `ep_00003` the shapes and dtypes match, the correlation is 0.99999 and the mean relative difference is 0.2 percent, which is bf16 nondeterminism through the same autoencoder. Only the sidecars were wrong, so the SD 1.x distances frozen at `01de829` stand. Seeded map 1 was different. Its old sidecars had narrow strings that passed the dtype check with the wrong button semantics, so its first score (gain 0.84 dB, in the first look) had fed the model wrong control tokens. Its outputs were removed and it was rescored with maps 9 to 15 from the re-encoded sidecars. The frozen table's seeded map 1 reads −0.23 dB. The lesson is that a dtype check catches only wide strings, and that skip-if-exists encoding can mix encoder versions inside one directory. [RC 2026-09-25 10:10, 10:40, 11:30.]

**The open verdict-gate question.** The 11:30 entry states it in these words: “Open for Rohan and Astra: whether (iv), an assumption about the maps rather than a check of the measurement, and (ii), a subset-level tolerance stricter than the bootstrap precision the test uses, belong in the gate; I did not change the gate after seeing the data.” Section 0 of `RESEARCH_CONTEXT.md`, rewritten at 19:30 EDT on 2026-09-25, restates it: the gate prints “does not support” because two distance sanity checks failed, “same-WAD arenas not all nearer than campaign maps; 4-episode subsets of nine campaign maps disagree by more than 10 percent”, and “Rohan rules, Astra reviews Sep 26 13:30 EDT.” The frozen distances file lists eight campaign maps failing check (ii), not nine (validation table above). At the 12:00 and 18:20 entries every other verdict condition passes and only `distance_validation` is false. Astra's Codex credits were exhausted on Sep 23 until Sep 26, so the review cannot run earlier. Section 0 also leaves open for Rohan the sealed-corpora rule for the study's fixed-checkpoint reads. [RC 2026-09-23 19:45; 2026-09-25 11:30, 12:00, 18:20; RC section 0.]

**What remains.** SD 3.5 is scored only at its final 200k weights, and PixArt at its 200k, so the “every row” condition will test three rows. The SD 3.5 latent-space distances wait on the SD 3.5 evaluation-map encode, which stood at 48 of 130 `unseen2` episodes at 14:00 because the encoder is the lowest GPU 3 tenant. The decoder-free outcome (latent MSE ratio to copy-last), the episode-level mixed model, stratification by control regime and the model-feature space (c) are not implemented. [RC 2026-09-25 01:25, 09:40, 14:00; design memo, sections 2 and 6.]

Taken together, the study shows that for the U-Net's final EMA weights the gain over persistence at one and four tics falls with nearest-training-map latent distance across 30 maps, with motion controlled, in two feature spaces, and with no map carrying the result. It shows that the row loses to persistence on most unseen maps. Under the rule declared before scoring, the verdict is “does not support” until the gate question is ruled. The study does not yet show that the relation holds for the other two rows.


## 4.11 Sep 24 to 25: decisions

This section was appended with sections 3.13 and 3.14; it belongs to part 4 and follows the ledger form of sections 4.1 and 4.10. Times are EDT and name the `RESEARCH_CONTEXT.md` entry that records each decision, on 2026-09-25 unless a date is given. A second time names the entry that revised, implemented or confirmed it. Commits are on main. Astra is the Codex reviewer, whose credits ran out on Sep 23 and reset at 13:28 EDT on Sep 26.

| Time | Decision | Evidence | Alternative rejected | Where recorded |
|---|---|---|---|---|
| 09-24 16:20 (Rohan), done 17:45 | Stop the U-Net at 200k and make 200k the matched step count for every row. SD 3.5 runs on to 200k. A server-side waiter stops the run once the 200k files and step 200100 are on disk and records the reason in `resumes.log`. | The U-Net had saturated: EMA 22.42 to 22.46 dB and 0.182 to 0.179 LPIPS at h1 from 150k to 180k, validation loss in a 0.153 to 0.155 band since 151k. It stopped at 17:09 after step 200150 with validation loss 0.1509 and no skips. | Running on toward the 400k ceiling (`STEPS=400000`), which the Sep 23 deck estimated for about 02:00 on Sep 26. | RC 09-24 16:20, 17:45; RC 09-23 13:15 |
| 09-24 18:40 and 20:35 (Rohan) | PixArt-alpha (Chen et al., ICLR 2024, `PixArt-alpha/PixArt-XL-2-512x512`) is the third row, `041-pixart-nexttic` on GPU 1 to 200k at the matched recipe, gated from `repo_launch2` with periodic reads on `cuda:3`. | It is a text-to-image DiT like the other two backbones and was the best transformer of the first run. The worker generalising the launcher recorded that the ImageNet DiT averages its control tokens without cross-attention, so it cannot see their order. The launcher gained a `dit` entry, a `RUN_NAME` override with the certificate keyed by run name, per-backbone GPU and micro-batch, and `ACTION_HISTORY` in the launch line; the live `unet` and `sd35` entries at `35258f3` stayed byte-equal through a gate run (suite 1,423). | DiT-XL/2 from ImageNet as the third row, which stays Rohan's call before any DiT row runs. | RC 09-24 17:45, 18:40, 20:35 |
| 09-24 18:40 (Rohan) | Drop the requested-action ablation (`ACTION_HISTORY=0`). Keep the requested-versus-executed finding as data hygiene plus the audit. | Requested and executed controls are 89 percent identical. Almost all of the rest are dropped weapon-select requests, a recorder defect that acts as label noise in the condition. Only 0.15 percent are true overrides, so the ablation would likely be null. | Spending GPU 1 on the ablation instead of a third backbone. | RC 09-24 16:20, 18:40, 20:35 |
| 09-24 22:20 (Rohan) | Run the map-distance generalization study (section 3.14): motion-weighted per-map latent clouds, nearest-training-map sliced Wasserstein, gain over persistence as the outcome, one pre-declared test. An Opus design worker writes the memo and a CPU prototype while Astra is out of credits; Astra reviews on Sep 26. | Changliu suggested it, and it reuses Rohan's validated NVIDIA method. Only a 28-episode fragment of the seeded evaluation corpus had been encoded, so `enc-eval-maps` encoded all 210 episodes per tic in both spaces on GPU 3 with the pinned encoder. | As the primary distance: the pooled corpus, which charges a held-out training map for not resembling the other three; Fréchet distance (one Gaussian, ill-conditioned covariance); MMD (bandwidth, awkward weights); nearest-training-frame distance (ignores density); a classifier two-sample test (AUC saturates near 1). | RC 09-24 22:20, 23:30; design memo, sections 1, 2 and 4 |
| 01:25, kept 09:40 | The primary test is the partial Spearman of one-tic gain over `persist_psnr_raw` on nearest-map SD 1.x distance, controlling for persistence PSNR, over the 30 primary maps, U-Net 200k EMA, with a case-bootstrap CI, permutation p and leave-one-map-out. SD 1.x distances are frozen before any per-map score. Check (ii) stops judging maps inside the floor band (`6a33311`, 01:19). | Rohan's “continue with everything” covered it; the choice is reversible and scores can be discarded. Inside the floor band D is draw noise, and the toy study saw 6 to 20 percent subset differences on the validation maps. | Choosing the test, the space or the arm after the scores; judging floor-band maps against a relative tolerance. | RC 01:25, 09:40; design memo, section 3 |
| 01:25, open at 09:40 | Score the sealed evaluation corpora once with a fixed checkpoint and book nothing in `$R/sealed`, leaving Rohan free to rule the other way. | `after_nexttic.sh` treats `arenas_678`, `seen`, `unseen` and `unseen2` as sealed. The scorer refuses the test corpus and checkpoints without EMA, and it reads with the final weights, so no selection is possible. | Booking the distance reads in the seal before Rohan rules. | RC 01:25, 09:40 |
| 09-23 19:45 and 09-24 21:55 | Native W&B logging is on by default in the trainer (`wandb_log.RunLogger`, `--no-wandb` only for the gates' fit, smoke and resume), and `periodic_eval.py` gives the trainer `--eval-every N` on `--eval-device`. Every launch after 21:55 on Sep 24 carries the init fix. | Merged at `c1f8fc8` (suite 1,381) after four rounds on 21 files and 3,067 lines. Astra confirmed rounds one to three by reproduction; its credits ran out before the round-four verdict, so main reviewed round four against the two reproductions. The live PixArt run's init then hung: `wandb.init` starts subprocesses on the calling thread while the trainer forks its DataLoader workers, and a worker forked in that window holds the pipe open. `11540db` (suite 1,424) moves setup before the fork, disables git probing and refuses subprocesses on the logger thread; verified live on Spiderman with eight forking workers. | Sidecar-only curves; restarting the pinned rows with a new flag. The two pinned rows keep `35258f3` and their `tools/wandb_tail.py` sidecars, and PixArt's curves come from the sidecar `041-pixart-nexttic-tail`. | RC 09-23 19:45; 09-24 21:05, 21:55 |
| 09-24 21:05 to 11:45 | GPU 3 tenancy order: PixArt's periodic reads, then SD 3.5 steward reads, then the distance scorer, then the evaluation-map encoder. The encoder wrapper (v3) pauses while any `results_spiderman/*/eval_*/child.pid` is alive and yields to steward read sessions; the scorer runs each map behind `MAP_GUARD` (`ca03bad`) and `$D/tmp/gpu3_guard.sh` (14 GB free and no periodic read); the steward starts a read only with 25 GB free, one at a time. | PixArt's `eval_0065000` lost both h4 labels and `eval_0070000` lost `tf_live_h1`, `tf_ema_h4` and the probe to OOM, while the card held the SD 3.5 encoder (19.6 GB), the scorer (14.8 GB), another user (4.6 GB) and a steward read. The trainer's periodic reads have no guard and no retry. PixArt lost no labels after the rule. | First-come sharing of GPU 3. | RC 09-24 21:05; 00:45, 02:10, 10:40, 11:45, 18:40 |
| 10:20 | Any CPU-heavy study job on Spiderman runs with `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` at 16 and at nice 19. The steward watches wall time per 50 steps and reports a trainer under 90 percent of its rate. | The pixel-space distance job ran 64 BLAS threads (load 128 on 64 cores) and cut SD 3.5 from 0.50 to about 0.27 updates/s and PixArt from 1.38 to about 0.9 for about 50 minutes. The earlier SD 1.x distance step likely did the same unnoticed. SD 3.5's 200k slipped by about 25 minutes. | Uncapped CPU jobs beside the trainers. | RC 10:20 |
| 13:15, eased 13:25 | Disk deletion ladder on `/sata2`. Rohan was paged with the commands for the four deletions he reserved for himself: `hf_release_arnold` (202 GB) and `raw_arnold` (202 GB), both verified on the Hub, `latents_arnold_sd35` (41 GB) and `_aborted_safe_to_delete` (39 GB). If he does not answer, main deletes `_aborted_safe_to_delete` at 300 GB free and `hf_release_arnold` and `raw_arnold` at 150 GB. The steward deletes nothing and reports every 15 minutes. PixArt's 70k and 75k holds were released. | Another user's job filled `/sata2` at about 185 GB/h from 12:42 (536 GB free at 12:40, 450 at 13:01), and none of our processes wrote. `train_wm.py` raises if a local checkpoint write fails, so a full disk kills either run at its next save. The rate fell back to 35 to 40 GB/h at 13:05, and `/sata2` had 336 GB free at 19:00. At 22:04 it had 292 GB free at 22 GB/h, so main took the first step at 22:15 and deleted `_aborted_safe_to_delete` (39 GB: the Sep 19 aborted 134-episode recording, the stale 0:60 `arenas_678` latents and the `seen_old28` sidecars, all regenerable and used by nothing, per its README). The 150 GB step stands unless Rohan runs it first. | Deleting anything of Rohan's before he answers; letting a run die on a failed save. | RC 13:15, 13:25, 18:20, 19:00, 22:15 |
| 12:55 | Leave two host issues to Rohan: the other user's job on PixArt's card, and `/home`. The Doom runs write only to `/sata2`, with the Hugging Face cache and `TMPDIR` there, and the steward checks that nothing of ours writes under `~`. | From about 12:40 another user's `extract.py --gpu 1` held GPU 1 at 100 percent and cut PixArt from 1.40 to 0.67 to 0.84 updates/s and SD 3.5 from 0.52 to 0.40 to 0.44 (host load 55 on 64 cores). `/home` (3.4 TB shared, holding the conda envs, `~/.cache` and the W&B config) was 100 percent used, with 19 GB free at 12:55 and 14.2 GB at 19:00. Our share is 822 GB of older data (`~/data` 498 GB, `~/perseve` 258 GB, `.cache` 23 GB, `miniconda3` 24 GB). The entry names asking the other user to move and freeing `/home` as Rohan's calls. | Contacting the other user; deleting other projects' data. | RC 12:55, 13:25, 19:00 |
| 22:15 and 23:00, conditional | If the 110k read confirms the 105k EMA weakness past about 128 tics as a trend, the paper's claim becomes “the EMA delays the closed-loop failure”, and the released SD 3.5 checkpoint is chosen by the rollout series, not by step count. From 110k every read reports, for both weight sets, the 256-tic PSNR against copy-seed, the channel-13 end count and minimum, and the rollouts with any frame under 10 dB, with the all-channel end-state count as context. | The 105k EMA fell 0.66 dB (seed 0) and 0.97 dB (seed 1) under copy-seed at 256 tics while its teacher-forced reads were the best of the row; its failures are per-channel mean shifts in channels 13, 15 and 8. The all-channel count put the EMA at 3 to 8 of 16 at every read from 55k to 105k, so it catches ordinary scene drift. | Keeping “the EMA stays clear” as the claim; tracking channel 13 alone; the all-channel count as the headline. | RC 21:00, 22:15, 22:40, 23:00 |
| 09-24 15:55; 05:00 | Report EMA rollouts in the paper, and state the live-versus-EMA stability result with per-rollout counts; keep rollout arrays on every full read. Watch the probe's off-manifold response and flag it only if a probe jump pairs with a directional drop or a live collapse. | The SD 3.5 50k and 70k live collapses and the channel-13 fixed point (section 3.13); real control swaps normal at 75k while the probe maximum jumped. | Treating the probe maximum as a control-strength measurement. | RC 09-24 15:55; 05:00, 18:40 |

**ETAs for the two remaining rows.** The 18:40 entry gives SD 3.5's 200k at about 21:30 EDT on Sep 27, at 0.527 updates/s, and PixArt's at about 00:50 EDT on Sep 27, at 0.93 updates/s with GPU 1 still shared. Section 0, rewritten at 19:30, restates them as about Sep 27 21:30 and about Sep 27 01:00. These supersede the earlier estimates: Sep 27 19:00 for SD 3.5 (Sep 24 16:20), Sep 26 13:00 for PixArt at launch (Sep 24 20:35), and about Sep 28 01:00 and Sep 27 evening at the slowest point (13:25). The rates kept moving after 19:00. Other users' jobs pushed the host load to 70 on 64 cores around 22:00 and cut SD 3.5 to 0.32 and PixArt to 0.6 updates/s, with SD 3.5's 200k slipping toward Sep 28 midday if that persisted; the load fell to 17 at about 22:45, and at 23:00 SD 3.5 ran at 0.50 to 0.54 and PixArt at 0.9 to 1.36. No entry after 19:30 restates the ETAs. SD 3.5 stood at 107.3k and PixArt at 106.8k at 21:00. [RC 09-24 16:20, 20:35; 13:25, 18:40, section 0, 21:00, 22:15, 23:00.]

**Other standing arrangements.** Stewards wait in loops that print every four minutes, after the harness watchdog killed a silent steward on the night of Sep 23. `stw6-arrive` holds and queues each 5k checkpoint automatically, which let the 95k and 100k reads run while steward 7 was stopped by the session limit from about 13:45 to 18:15. At 18:22 a stale multiplexed SSH master hung the steward's polls for 15 minutes; it was closed with `ssh -O exit`, and one fresh connection succeeded, so this was not an edge block. Superman finished both 2000:6000 sets at 21:30 on Sep 24 (4,000 episodes each, row-count audits with 0 mismatches), uploaded to `sd15/arenas_2000_6000` and `sd35/arenas_2000_6000`. The 6,000-episode U-Net row is the candidate for GPU 2 after SD 3.5 stops, pending Rohan; it needs a re-download of the 4-channel set (about 250 GB), a hard-linked 6,000-episode directory and a gate run. [RC 09-24 09:45, 17:45, 22:50; 04:30, 18:20, 19:00.]

**What remains for the paper.** The 110k and 115k SD 3.5 reads decide whether the EMA's 105k rollout weakness is a trend, and with it the stability claim and the choice of the released SD 3.5 checkpoint. SD 3.5 and PixArt each need their 200k full reads and then the distance study's per-map scoring at 200k, so that the verdict's “every row” condition tests three rows. The SD 3.5 latent-space distances need the SD 3.5 evaluation-map encode to finish. Rohan rules on the verdict gate (section 3.14) and on the sealed-corpora booking, and Astra reviews the gate and the distance-study design on Sep 26 at 13:30 EDT, when its credits return. The paper protocol must fix one step count and one decoder before any final table (sections 3.12 and 3.13). The paper timeline set on Sep 24 at 16:20 was a draft to Changliu on Saturday night, Sep 26, with provisional numbers, final tables on Sep 28 to 29, and submission on Sep 30 AoE. The later entry at 13:25 on Sep 25 and the weekly-update draft move the full draft to Sunday evening, Sep 27, with SD 3.5 numbers marked provisional where its run has not reached 200k; the later plan is used. `paper/main.tex` is still the Sep 16 skeleton built around the stride-four rows, and no CoRL template is checked in. [RC 09-23 19:45; 09-24 16:20, 22:20; 09:40, 11:30, 13:25, 22:15, section 0; `.claude/analyses/weekly-update-2026-09-25.md`, Plan; `paper/main.tex:1–8`.]

## 5.9 Literature on distance versus generalization and persistence baselines

This section was appended with sections 3.13, 3.14 and 4.11; it belongs to part 5. It condenses `.claude/analyses/distance-study-literature-2026-09-25.md` (commit `414e6ee`), written by an Opus reader on 2026-09-25 from the papers' arXiv, proceedings or ACL Anthology PDFs. The memo marks as unverified anything it could not confirm from the PDF text, and this section keeps those marks. It answers two questions for the paper: who has correlated a train-to-domain distance with a performance drop, and who has used copy-last-frame as a reference. [RC 2026-09-25 09:50; memo, scope line.]

**Distance against performance.** The study's shape, one distance per domain against one normalised performance number per domain, is standard in domain adaptation and transfer learning. Ben-David et al. bound target error by source error plus a divergence term (Machine Learning 2010, Theorem 2), and Blitzer, Dredze and Pereira plot proxy A-distance against adaptation loss for 6 domain pairs (ACL 2007, Figure 3). Adaptation loss subtracts in-domain accuracy, which is the same move as our gain over persistence. OTDD (Alvarez-Melis and Fusi, NeurIPS 2020, arXiv 2002.02923) is the nearest method and reporting match: an optimal-transport distance between datasets against the relative drop in target error, with ρ and p in each panel, from −0.59 (n = 16) to −0.85 (n = 11); whether ρ is Pearson or Spearman is unverified. Cui et al. (CVPR 2018, arXiv 1806.06193) use a weighted EMD between feature clouds, structurally our construction, without a statistic. Deng and Zheng (CVPR 2021, arXiv 2007.02915) report Spearman ρ of about −0.91 between Fréchet distance to the training features and accuracy, on synthetic sample sets whose plotted count is unverified. [Memo, Q1.]

Distances have also lost. Guillory et al. (ICCV 2021, arXiv 2107.03315) find that Fréchet distance, MMD and proxy A-distance predict accuracy worse than an average-confidence baseline. Mayilvahanan et al. (ICLR 2024, arXiv 2310.09562) find that matching train-test similarity leaves CLIP's out-of-distribution performance high. Miller et al. (ICML 2021) and Baek et al. (NeurIPS 2022) show that in-distribution performance alone predicts out-of-distribution performance tightly across many models, which a study with two or three rows cannot use. These are the precedents for reporting a weak or null coefficient. [Memo, Q1.]

**World models on unseen scenes.** No paper found correlates a train-to-domain distance with a generative world model's or video predictor's quality across held-out scenes. World-model papers report unseen-domain quality as a table over 1 to 8 domains: NWM compares one unknown environment with one known (CVPR 2025, Table 4) and names drift toward training data as a failure mode; SCOPE reports 4 unseen styles; Matrix-Game reports 8 Minecraft biomes; GAIA-2 holds out geographies with no distance axis. GameNGen has no held-out-map evaluation, and DIAMOND trains on one CS:GO map. MultiGen trains Doom on 100 procedurally generated maps; whether its Table 1 maps are held out is unverified. The Doom precedent for disjoint test textures is Dosovitskiy and Koltun (ICLR 2017, Table 2). No driving paper found correlates a latent scene distance with planner or world-model performance, so the NVIDIA method has no public counterpart in this search. [Memo, Q2.]

**Copy-last-frame as a reference.** Rohan doubted that recent work uses it. The doubt holds for recent world models and fails for 2014 to 2019 video prediction. Ranzato et al. (2014), Mathieu, Couprie and LeCun (ICLR 2016, “Last input”, scored also on moving areas only), Finn, Goodfellow and Levine (NeurIPS 2016), PredNet (Lotter et al., ICLR 2017, “Copy Last Frame”, trained on KITTI and tested on unseen CalTech Pedestrian), ContextVP (ECCV 2018), SDC-Net (ECCV 2018, where CopyLast beats two models on YouTube-8M) and Villegas et al. (NeurIPS 2019, where copy-last beats every model on Human3.6M) all report it. The memo found it in none of the 2024 to 2026 game and driving world models it checked (GameNGen, DIAMOND, Genie, NWM, Vista, GAIA, Matrix-Game and others), with one 2026 latent-space exception, ThinkJEPA. Two observations support reporting it. Matrix-Game 2.0 attributes Oasis's higher consistency scores to static frames after collapse, which a persistence reference would expose. Villegas et al. state the rationale directly (App. A.2.2): per-frame evaluations are unreliable when much of a video does not move. [Memo, Q4.]

Our gain has the algebraic form of Genie's ΔtPSNR (arXiv 2402.15391): a difference of two PSNRs, which is ten times the log of an MSE ratio. The two differ only in the reference prediction, a frozen copy of the last frame for us and random-action generation for Genie. [Memo, Q5.]

**Where the study is standard and where it is not.** Standard: one distance per domain against a normalised outcome (Blitzer, OTDD, Cui, Deng and Zheng); optimal-transport distances on learned features, of which sliced Wasserstein is a cheaper member that takes weights exactly; per-domain normalisation; copy-last-frame as the reference and motion-restricted scoring (Mathieu Table 2, PredNet, Villegas 2019). Not found elsewhere: distance against quality for a generative world model across held-out maps; the minimum over training domains as the distance (closest: Mayilvahanan's per-sample nearest neighbour); motion weighting inside the distance; a partial rank correlation with bootstrap CI, leave-one-out and within-cluster coefficients; and a distance frozen before outcomes are read. No study found runs a partial correlation. Our n of 30 exceeds Blitzer's 6 and OTDD's 11 to 16 and is near Cui's 35. [Memo, Q3 and Q6.]

Citations for the paper's related-work paragraph, in the memo's order of priority:

| Use | Paper | What to cite it for | Where |
|---|---|---|---|
| Nearest method and reporting match | Alvarez-Melis and Fusi, NeurIPS 2020, arXiv 2002.02923 (OTDD) | OT distance between datasets against difficulty-normalised transfer, ρ and p per panel | Figs. 6–7 |
| Nearest “distance predicts drop” claim | Deng and Zheng, CVPR 2021, arXiv 2007.02915 | Fréchet distance to training features against accuracy, Spearman about −0.91; reason to show Fréchet only as a robustness check | Fig. 2 |
| Persistence reference on an unseen domain | Lotter et al., ICLR 2017, arXiv 1605.08104 (PredNet) | KITTI-trained predictor scored on unseen CalTech against Copy Last Frame | Table 2, Fig. 7 |
| Rationale for the persistence reference | Villegas et al., NeurIPS 2019, arXiv 1911.01655 | Copy-last beats every model on Human3.6M; static video breaks per-frame metrics | Appendix Figs. 8–10, A.2.2 |
| Normalised-outcome scatter | Blitzer, Dredze and Pereira, ACL 2007, P07-1056 | Proxy A-distance against adaptation loss, labelled points | Fig. 3 |
| Motion as the confound | Mathieu, Couprie and LeCun, ICLR 2016, arXiv 1511.05440 | Scores restricted to moving areas; “Last input” baseline | Table 2 |
| Form of the gain | Genie, arXiv 2402.15391 | ΔtPSNR as a difference of PSNRs against a counterfactual predictor | Evaluation section |
| Null framing | Guillory et al., ICCV 2021, arXiv 2107.03315; Mayilvahanan et al., ICLR 2024, arXiv 2310.09562 | Distances losing to confidence baselines; similarity not explaining OOD performance | Fig. 3; Fig. 3 and Table 1 |

The memo's conventions for the figure are one labelled marker per map, ρ with its CI and p in the panel corner, error bars from repeated draws, and the two clusters reported separately, with Pearson or R² in an appendix for readers from the accuracy-on-the-line work. The frozen figure already has one marker per map, shaped by cluster, 95% bootstrap bars on both axes, the train-versus-train floor band, the persistence line, and ρ with its CI in the legend. It does not yet label the maps or print p. [Memo, Q6; `results/distance_study/figure_unet_h1/distance_gain.png`.]
