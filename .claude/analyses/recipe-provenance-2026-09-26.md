# Recipe provenance: every training and evaluation hyperparameter against its precedent (2026-09-26)

Purpose: let the paper say, for each setting, "we follow X" or "ours, because Y", and never claim an ablation that was not run. Our values come from `train_wm.py` defaults as overridden by `scripts/spiderman/launch_nexttic.sh` (the certified command), `diffusion_v.py`, `backbones.py`, `adapt_wm.py`, `lora.py`, `finetune_decoder.py` with `scripts/spiderman/decoder_mse.sh`, `eval_tf.py`, `rollout_eval.py`, `distance_study.py`, `docs/RESEARCHER_DOSSIER.md` §3 to §4, and `RESEARCH_CONTEXT.md` (log dates quoted as RC mm-dd hh:mm). Every precedent value was read by me on 2026-09-26 from the arXiv PDF text (pdftotext), not from the memos, unless marked VERIFY.

Key: **F** follows the cited paper exactly; **D** differs from its nearest precedent; **O** ours, no precedent found.

Sources (short name, URL): GameNGen https://arxiv.org/abs/2408.14837 · DiT https://arxiv.org/abs/2212.09748 · PixArt https://arxiv.org/abs/2310.00426 · SD3 https://arxiv.org/abs/2403.03206 · ProgDist (Salimans and Ho) https://arxiv.org/abs/2202.00512 · DDIM https://arxiv.org/abs/2010.02502 · LoRA https://arxiv.org/abs/2106.09685 · Vista https://arxiv.org/abs/2405.17398 · DiffFit https://arxiv.org/abs/2304.06648 · Biderman https://arxiv.org/abs/2405.09673 · AdaWorld https://arxiv.org/abs/2503.18938 · Taylor-Stone https://www.jmlr.org/papers/volume10/taylor09a/taylor09a.pdf · PredNet https://arxiv.org/abs/1605.08104 · Genie https://arxiv.org/abs/2402.15391 · OccWorld https://arxiv.org/abs/2311.16038 · LDM https://arxiv.org/abs/2112.10752 · InstructPix2Pix (IP2P) https://arxiv.org/abs/2211.09800 · XEWorld https://arxiv.org/abs/2608.05799 · SD 1.4 configs https://huggingface.co/CompVis/stable-diffusion-v1-4 · PixArt scheduler config https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512 · sd-vae-ft-mse card https://huggingface.co/stabilityai/sd-vae-ft-mse · diffusers LoRA script https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py

| # | Parameter | Ours | Nearest precedent: where, value | = ? | Reason if not equal (log date) | Sentence for the paper |
|---|---|---|---|---|---|---|
| 1 | U-Net warm start | SD 1.4 U-Net, every parameter trained | GameNGen §4.2: SD 1.4, "unfreezing all U-Net parameters" | F | | "The U-Net row starts from Stable Diffusion 1.4 with all parameters trained, as in GameNGen." |
| 2 | Transformer warm starts | PixArt-alpha XL-2-512; SD 3.5 Medium (MMDiT-X) | none as a world model; SD 3.5 Medium has no paper (SD3 describes the family: VERIFY) | O | public packages in the SD latent space and a 16-channel space; "PUBLIC pretrained starting weights (purity)" (RC 09-21 01:30; PixArt chosen RC 09-14 22:30, SD 3.5 RC 09-17 11:45) | "We add two public transformer packages, PixArt-alpha and SD 3.5 Medium, each from its released weights." |
| 3 | Context input and its init | 32 context latents channel-stacked with the noisy target; new input channels zero-initialised | GameNGen §3.2 (concatenate in the channel dimension); IP2P §3.2 (weights on new input channels "initialized to zero") | F | | "Context latents are concatenated along channels (GameNGen) with zero-initialised input weights (InstructPix2Pix), so step 0 is the pretrained model." |
| 4 | PixArt output head | epsilon half of the 8-channel learned-sigma head, retrained to velocity | none | O | `backbones.py` design (RC 09-14 22:30) | state plainly |
| 5 | SD 1.x latent space | frozen KL-f8 encoder (sd-vae-ft-mse; only its decoder differs from SD 1.4's), 4×32×40, scale 0.18215 | GameNGen §3.2.2 (SD 1.4 autoencoder); SD 1.4 `vae/config.json` scaling_factor 0.18215; LDM §4.3.2 rescaling by component std; ft-mse card: "only the decoder part was finetuned" | F | | "The U-Net and PixArt rows use Stable Diffusion's KL-f8 latents, scaled by 0.18215." |
| 6 | SD 3.5 latent space | 16 channels, (z − 0.0609) × 1.5305 | SD3 §5.2.1 (d = 16), App. B.2 (normalise by mean and std) | F | the two constants are from the gated model config as recorded in `backbones.py` (read Sep 17): VERIFY | "The SD 3.5 row uses its own 16-channel autoencoder and normalisation." |
| 7 | Frame padding | 320×240 padded to 320×256, cropped before scoring | GameNGen §4.2: "320x240 padded to 320x256" | F | | cite GameNGen |
| 8 | Context length | 32 tics (0.91 s) | GameNGen §4.2: 64; Table 2: 32 gives 22.31 dB, 64 gives 22.36 | D | "32 is the cheap end of a flat curve" (launcher; RC 09-20 21:45: "32 vs 64 is 0.05 dB") | "We use 32 context frames; GameNGen's ablation finds 32 within 0.05 dB of its 64." |
| 9 | Prediction spacing | every tic, 35 Hz | GameNGen states action repeat 4 (App. A.5) and scores "35FPS data" (App. A.6); its training stride is not stated | O | inferred from GameNGen (RC 09-19 20:00 correction; decision RC 09-20 21:45, 09-21 01:30) | "We predict every 35 Hz tic, which we read as GameNGen's setting; it does not state its stride." |
| 10 | Control pathway (U-Net) | one token per context tic through the cross-attention that carried text | GameNGen §3.2: action embedding "into a single token", replacing text cross-attention | F | | cite GameNGen |
| 11 | Control token content | executed 19-bit button vector, two-layer MLP, learned positions | GameNGen: vocabulary and encoding undisclosed | O | Arnold's anti-stuck overrides make the requested id a wrong label (RC 09-21 01:30; normalisation RC 09-22 11:30) | state plainly |
| 12 | Control-to-frame convention | controls `buttons[r-32:r]`, newest carries the last context frame into target r | none; the open GameNGen reproduction uses the opposite row convention (dossier §9.3) | O | recorder stores a row then steps (RC 09-21 01:30) | state plainly |
| 13 | Transformer control pathways | PixArt: tokens through the caption projection; SD 3.5: joint-attention tokens plus a pooled slot from the newest control | none | O | RC 09-14 22:30, RC 09-18 00:30 | state plainly |
| 14 | Context or action dropout, CFG | none (dropout 0, no guidance) | GameNGen §4.2: context dropped with p 0.1; §3.3.1: CFG 1.5 on observations | D | action dropout 0 (RC 09-13 22:16); "observation CFG unavailable" (RC 09-21 14:30) | "Unlike GameNGen we train without context dropout, so we sample without guidance." |
| 15 | Objective | v-prediction, unweighted MSE | GameNGen §3.2 Eq. 1; ProgDist §4 (v ≡ αε − σx) | F | | cite both |
| 16 | Noise schedule | linear β 1e-4 to 0.02, T = 1,000 | DiT §4 ("tmax = 1000 linear variance schedule ranging from 1×10−4 to 2×10−2", from ADM); PixArt's released scheduler config is the same; SD 1.4 is scaled-linear 0.00085 to 0.012 (config); SD 3.5 is rectified flow (SD3 §2) | F (DiT) | native for PixArt, mismatched for SD 1.4 and SD 3.5; kept for continuity, no ablation (RC 09-21 14:30) | "We use DiT's linear schedule for all rows; it is PixArt's native schedule and differs from SD 1.4's and SD 3.5's; we did not ablate it." |
| 17 | Timestep sampling | t uniform over 1,000 | GameNGen Eq. 1: t ∼ U(0, 1) | F | | |
| 18 | Noise augmentation level | uniform up to 0.7, 10 buckets, bucket embedded | GameNGen §3.2.1, §4.2: "maximal noise level of 0.7, with 10 embedding buckets" | F | | cite GameNGen |
| 19 | Noise augmentation form | c̃ = √(1−q)·c + √q·ε | GameNGen: "adding a varying amount of Gaussian noise", formula not stated | O | "at most one [form] matches GameNGen" (RC 09-19 20:15) | state the formula |
| 20 | Inference context noise | 0 (bucket 0) | GameNGen §3.2.1: level "can be controlled", value not stated | O | stride-4 sweep 0.035 to 0.3 moved PSNR@64 under 2 SE (RC 09-20 08:30) | "We condition on clean context at inference." |
| 21 | Optimiser | AdamW, fused, default betas | DiT §4: AdamW; GameNGen §4.2: Adafactor | F (DiT) | | |
| 22 | Learning rate | 5e-5, constant after warmup | GameNGen 2e-5 (§4.2); DiT 1e-4 (§4) | D | shared rate between the two published values after the DiT excursion at 1e-4 (Astra round 3, RC 09-10 14:00; Rohan RC 09-13 22:16); cells lr1e-4 and lr2.5e-5 were queued (RC 09-18) and never ran | "We use 5e-5, between GameNGen's 2e-5 and DiT's 1e-4; we did not tune it." |
| 23 | Warmup | 2,000 linear | DiT §4: "did not find learning rate warmup ... necessary"; GameNGen: not stated | O | RC 09-10 14:00, RC 09-13 22:16 | state plainly |
| 24 | Batch | 32, one card, no accumulation | GameNGen 128; DiT 256 | D | one 48 GB card per row; April recipe; RC 09-21 01:30 | "Global batch 32 on one GPU." |
| 25 | Weight decay | 0 | GameNGen §4.2 "without weight decay"; DiT §4 "no weight decay" | F | | |
| 26 | Gradient clipping | 1.0 | GameNGen §4.2 | F | | |
| 27 | EMA | 0.9999, fp32 (applied as 0.9999⁸ every 8 updates); reported numbers use EMA | DiT §4: EMA decay 0.9999, "All results reported use the EMA model" | F | fp32 because bf16 freezes the average (dossier §3.6) | cite DiT |
| 28 | Updates | 200k | GameNGen 700k (§4.2); its ablations 200k (§5.2.1) | D | U-Net saturated 150k to 180k; matched count for all rows (RC 09-24 16:20) | "200k updates, the budget of GameNGen's ablations, at a quarter of its batch." |
| 29 | SD 3.5 deviations | gradient checkpointing; skip update if pre-clip norm > 5 after update 3,000 | none | O | guard added after the UniDiffuser excursion (RC 09-17 00:20); armed only after 3,000 updates because post-inflation norms are 12 to 21 (RC 09-18 09:45, 14:50) | state as deviations |
| 30 | Sampler | DDIM, η = 0 | GameNGen §3.3.1 (DDIM); DDIM §4.1 (σ = 0 deterministic) | F | | |
| 31 | Sampling steps | 10, linear spacing | GameNGen 4 with CFG 1.5 (§3.3.2, Table 1); DDIM App. D.2 linear spacing | D | a declared point on the perception-distortion sweep (RC 09-23 11:30; dossier §4.2) | "Ten DDIM steps; fewer raise PSNR and worsen LPIPS (appendix sweep)." |
| 32 | Teacher-forced windows | 512 from val ids 6000:6100, seed 0 | GameNGen Table 1: 2,048 frames; Table 2: 8,912 examples | O | trainer `--eval-windows 512` | state plainly |
| 33 | Distance-study windows | 256 per map | none | O | `score_distance_maps.sh` | state plainly |
| 34 | Adaptation windows | 32 per held-out episode × 8 episodes | none | O | RC 09-26 20:10 | state plainly |
| 35 | Rollouts | 16 × 256 tics against copy-seed | GameNGen Fig. 6: 64 steps | O | dossier §4.3 | state plainly |
| 36 | Persistence reference | copy-last on raw frames, beside the decoder ceiling | PredNet §3.2 Table 2 ("Copy Last Frame", unseen CalTech); OccWorld §4.3 Table 1 (Copy&Paste, 0 s row = reconstruction) | F | raw rather than decoded copy is ours (RC 09-25 11:30) | cite both |
| 37 | Decoder tune: what trains | decoder only, encoder frozen, MSE only | GameNGen §3.2.2 | F | | cite GameNGen |
| 38 | Decoder tune: budget | AdamW 1e-5, 200 warmup then linear decay to 5%, 4 h, 400k frames of train ids 0:2000, micro-batch set at launch (VERIFY value) | GameNGen §4.2: batch 2,048, "other training parameters identical to those of the denoiser" | D | 4 A6000-hours per space (`decoder_mse.sh`; RC 09-26 16:20) | "A four-GPU-hour MSE decoder tune, far below GameNGen's." |
| 39 | Decoder loss rows | 240 real rows (padding excluded) | none | O | 6.25% of the old loss was padding (RC 09-21 14:30) | |
| 40 | LoRA rank | 16 | Vista App. C.3: "The rank of LoRA is set to 16" | F | | cite Vista |
| 41 | LoRA alpha | 16 (scale 1) | LoRA §4.1: scale α/r, α set "to the first r we try"; Biderman §4.7 recommends α = 2r | F (LoRA) | | |
| 42 | LoRA dropout | 0 | LoRA Table 11 (DART 0.0; E2E 0.1) | F | | |
| 43 | LoRA targets | q, k, v, out of every attention | Vista App. C.3 (all attention blocks); LoRA Table 5; Biderman §4.7: attention-only underperforms MLP or all | F (Vista) | MLP is a flag | |
| 44 | Trained in full beside LoRA | control MLP and positions, input projection, noise-bucket embedding | DiffFit §3.2 (bias, norm, class embedding, γ); Vista App. C.3 (new projections) | D | DiffFit Table 1: LoRA r16 mean FID 81.31 vs full 16.59 (design decision 4, RC 09-26 20:10) | "Following DiffFit and Vista, conditioning parameters train in full." |
| 45 | LoRA lr | 1e-4 constant | Vista 5e-5; DiffFit ×10 its full-FT 1e-4 (§4.1); diffusers script default 1e-4 (code) | D | tooling default (design page Sep 26) | "1e-4, the diffusers default." |
| 46 | LoRA warmup | 100 | diffusers default 500 (code) | O | | |
| 47 | Adaptation EMA | 0.999 fp32; live weights primary | none | O | at 0.999 the EMA still weights step 0 at 0.78 after 250 updates (design page) | |
| 48 | Step grid | 0 to 4,000 | AdaWorld §3.2: 800 steps | O | `4ef7673`, RC 09-26 20:10 | |
| 49 | Adaptation batch | 32 | AdaWorld §3.2: batch 32 | F | Superman launch uses micro-batch 8 × 4 accumulation (`lora_arenas13.sh`) | |
| 50 | Adaptation data | 8 episodes (ladder 1 to 16), 8 held out | AdaWorld: 100 samples per action | O | design decisions 1 to 2 | |
| 51 | Source weights | EMA tensors | none | O | zero-shot scores used EMA | |
| 52 | Full fine-tune lr | 2e-5 | GameNGen §4.2: 2e-5 with Adafactor at batch 128 (pretraining); Biderman §4.7: LoRA best lr about 10× full | D | Rohan Sep 26; branch `worktree-agent-acd5c0d7bca9d6bbf` commit `1edd9ee` only | "Full fine-tuning at GameNGen's 2e-5, five times below the LoRA rate." |
| 53 | Cost metric form | PSNR(model) − PSNR(copy-last), decoded, scene rows | Genie §3 Metrics: ΔtPSNR = PSNR(x, x̂) − PSNR(x, x̂′), random-action counterfactual | D | copy-last replaces random actions (RC 09-26 19:45) | "A Genie-style PSNR difference against copy-last." |
| 54 | Cost statistic | updates to threshold at several thresholds, plus AUC | Taylor-Stone §2: time to threshold; "a range of thresholds"; area under the curve | F | | cite Taylor and Stone |
| 55 | Threshold level | training maps' own advantage; 25/50/100%, 50% headline | none | O | RC 09-26 19:45 | |
| 56 | Distance D | motion-weighted sliced Wasserstein (1,000 projections) from the map's SD 1.x latent cloud to the nearest training map | XEWorld §3.3: cosine distance on visual features; OT family (OTDD, Cui EMD) per memo: VERIFY | O | frozen Sep 25 (RC 09-25 09:40) | state plainly |

## Counts

23 parameters follow a cited paper exactly (F), 11 differ from their nearest precedent (D), 22 are ours with no precedent found (O); 56 in all. Two F rows carry caveats: row 6's constants come from a gated config (VERIFY), and row 16 follows DiT but is not the native schedule of two of our three warm starts.

## Ours, with no precedent: state these plainly

- Transformer warm starts (PixArt-alpha, SD 3.5 Medium) and PixArt's head surgery.
- Every-tic spacing: inferred from GameNGen, never stated there.
- The executed 19-bit control, its MLP and positions, the row convention, and the PixArt and SD 3.5 control pathways.
- The variance-preserving corruption formula and clean-context inference.
- Warmup 2,000 (DiT uses none).
- SD 3.5's gradient checkpointing and spike guard.
- Window counts: 512, 256 per map, 32 per held-out episode, 16 rollouts of 256 tics.
- Decoder loss on the 240 real rows.
- LoRA warmup 100, adaptation EMA 0.999, EMA source weights, step grid to 4,000, 8 adaptation episodes.
- The cost threshold level.
- The distance D.

## Ablations that exist, and on which recipe (claim no others)

- Noise augmentation off, from scratch: PixArt, stride 4, 30k updates (RC 09-20 08:30, 09-19 13:30). Not the next-tic recipe.
- Context 2 to 32: ImageNet DiT, stride 4, 5k updates, not monotone (RC 09-14 20:30). ctx16 grid cell stopped at 27.5k (RC 09-21 01:30).
- Inference context noise: stride-4 U-Net and PixArt rollouts (RC 09-20 08:30).
- Sampler steps 4 to 50: next-tic U-Net 200k, teacher-forced only (dossier §4.2).
- Not ablated under any recipe: lr, warmup, batch, schedule, EMA decay, updates, control encoding, decoder budget, and any LoRA setting (rank, alpha, targets, lr).
