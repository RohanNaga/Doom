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

**Step-sweep protocol:** U-Net and PixArt, historical reporting windows, **512 windows** per evaluated setting; historical live checkpoints and tuned C4 decoder, raw TF references. Exact per-cell checkpoint/window manifests and numerical rows are not locally recovered. The surviving summary compares **1/2/4** denoising steps with **50**. Source: RC 2026-09-22 09:15; `results/levers_2026-09-20/` contains plotting/table builders rather than a complete numeric sweep archive.

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

The 50-step rows reproduce the stored 512-window numbers of these rows. A few-step sample lies near the conditional mean of the next frame, which is blurred: that wins PSNR and loses LPIPS. GameNGen's Table 3 shows PSNR and LPIPS both improving from 1 to 4 steps, the opposite LPIPS pattern; its sampler is a different, distilled configuration, so the pattern need not transfer. GameNGen reports its headline at 4 steps, so our 4-step column is the like-for-like sampler setting for that comparison.

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
| Initial unseen subset | `arenas_678` 0:60 | 60 episodes, 20 per arena | Reporting subset |
| Full unseen pool | `arenas_678` all | 3,000 episodes, 1,000 per arena | Available pool; JSON operational `unseen` range names only the initial subset |

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

Plotting our results relative to measured persistence is useful for explaining why similar model improvements can sit on different absolute PSNR scales. It cannot establish “no unexplained modeling gap” to the original experiment because the original persistence baseline was not measured. The **23.79 dB** proxy belongs to reproduction-directory footage with unresolved provenance. Its use is an illustrative comparison, not a substitute measurement. [G Figure 13; Q4 `TABLE.md:16,43–45`; RC 2026-09-20 21:45.]

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
