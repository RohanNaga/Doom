# Adaptation, decomposition and layers: literature check of the Sep 26 discussion (2026-09-26)

Scope: the eight steps of Rohan and Keerthana's Sep 26 discussion (RESEARCH_CONTEXT.md entries 16:50, 18:30 and 19:15, plus main's step 8). This memo does not repeat `distance-study-literature-2026-09-25.md`, which covers distance against generalization and copy-last baselines; where a paper appears there, this memo points to it.

Method: every number, figure and table below was read from the arXiv PDF text (pdftotext) on 2026-09-26. Venues come from the arXiv metadata comment field, the PDF header, or the proceedings footer. **VERIFY** marks a venue or claim not confirmed that way. Where only an abstract or a secondary source supports a claim, the row says so.

## Bottom line

| Step | Verdict | Closest prior work |
|---|---|---|
| 1. Unseen-map evaluation is rare | Partly done. Held-out domains appear in 1 to 5 domain tables; never 30 maps, never against persistence | XEWorld (2608.05799), Dreamer 4 (2509.24527), AVID Coinrun (2410.12822) |
| 2. Appearance versus dynamics | Done in adjacent fields, and the answer matches Rohan's hypothesis (models fail on appearance) | XEWorld, Kang et al. ICML 2025 (2411.02385), Distracting Control Suite |
| 3. Rendering ceiling beside model score, copy-last in latent space | Done in occupancy and semantic forecasting; not in game video world models | OccWorld Table 1, DOME Table 2, DINO-Foresight Table 2, Kang App. B.3 |
| 4. Cheap adaptation without retraining the backbone | Done in general; decoder-only adaptation to an unseen game map not found | iVideoGPT tokenizer-only adaptation, AVID, GameFactory, DreamGen, AdaWorld |
| 5. Middle layers carry richer features | Done for LLMs and diffusion; "Let's Verify" does not support the middle-layer claim | Layer by Layer (ICML 2025), DoLa, REPA, DDAE, AC3D, VPP |
| 6. Where conditioning acts (timestep, block) | Done for text and camera conditioning; not for game actions | AC3D (CVPR 2025), guidance interval (NeurIPS 2024), 2512.22175 |
| 7. EMA versus live weights in closed loop | New. No paper compares EMA and live weights in rollouts | EDM2 (EMA length matters for FID), DIAMOND Fig. 3 (sampler drift) |
| 8. Adaptation curves predicted by distance | Partly. Every ingredient exists; the combination over about 30 targets does not | XEWorld §5.3, Hernandez et al. 2021, LEEP §5.5, Gao and Chaudhari 2021 |

The three facts most likely to change decisions:

1. **XEWorld (arXiv 2608.05799, Aug 2026) already makes the step-2 and part of the step-8 argument in robotics.** Across five robots (leave-one-out), appearance distance to the training fleet predicts held-out LPIPS (Pearson r = 0.812, permutation p = 0.075), and kinematic distance does not (r = 0.549, p = 0.392). A full-DiT fine-tune on 25 target episodes cuts LPIPS by 30 to 32 percent and raises LPIPS on a seen robot by 69 percent (§5.3, Table 4, Fig. 4, App. Table 14 area). This is the closest competitor for step 8 and must be cited.
2. **The rendering-versus-prediction split has a template.** OccWorld (Table 1) and DOME (Table 2) print the tokenizer's reconstruction score and a Copy&Paste row beside the forecasting scores. DINO-Foresight (NeurIPS 2025, Table 2) prints Oracle 77.0, Copy Last 54.7 and prediction 71.8 mIoU in DINO feature space. Our three-column presentation (ceiling, persistence, model) is established practice in those fields, and new for game world models.
3. **The LLM analogy rests on the wrong paper.** "Let's Verify Step by Step" trains a separate reward model on step labels (PRM 78.2 percent against ORM 72.4 percent, best-of-1860, Fig. 3). It says nothing about middle layers. The middle-layer evidence is Layer by Layer (up to 16 percent over the final layer) and DoLa. For diffusion, AC3D shows the camera signal (our yaw) is linearly decodable best in middle DiT blocks and is decided in the first few percent of denoising, which supports Astra's "control pathway by noise level" axis rather than a depth-only axis.

## Step 1. Held-out scenes, levels or maps

Yesterday's memo (Q2, Q5) covers GameNGen, DIAMOND, Genie, NWM, Vista, GAIA-2, Cosmos, Matrix-Game, MineWorld, MultiGen, SCOPE and Procgen. New rows only:

| Paper | Venue | Year | arXiv | What they did | Metric | Where | Same as ours? |
|---|---|---|---|---|---|---|---|
| XEWorld (Chen et al.) | **VERIFY** (arXiv) | 2026 | 2608.05799 | Hold out robot embodiments in byte-identical RoboTwin scenes; 3-train/2-held-out split and 5-fold leave-one-embodiment-out | PSNR, SSIM, LPIPS plus robot-mask IoU, keypoint PCK, object-trajectory error | §3, Table 1, Fig. 5 | Partly: held-out domain with a distance, n = 5, no persistence reference |
| Dreamer 4 (Hafner et al.) | **VERIFY** (arXiv) | 2025 | 2509.24527 | Actions given for Minecraft Overworld only; action-conditioned generation evaluated on Nether and End | PSNR, SSIM of 16-step generations with 320 context frames, normalized between no-action and all-action models: 76% PSNR, 80% SSIM | §4.3, Fig. 7 right | Partly: held-out region, but its videos were in training; without them, "poor generation scores" (text only) |
| AVID (Rigter et al.) | **VERIFY** (arXiv) | 2024 | 2410.12822 | Pretrain on 15 Procgen games, adapt to the held-out 16th (Coinrun) | FVD, FID, SSIM, LPIPS, PSNR, action error ratio | Table 1 | Partly: held-out game, but adapted, not zero-shot |
| Vid2World (Huang et al.) | ICLR 2026 | 2025 | 2505.14357 | Trained on CS:GO only, rolled out zero-shot on Valorant | Qualitative only; "visual quality degrades much faster" | App., Fig. 18 | Different: qualitative |
| GameFactory (Yu et al.) | ICCV 2025 | 2025 | 2501.08325 | Minecraft actions transferred to open-domain generated scenes | Camera error, flow, domain similarity, CLIP, FID, FVD | Table 3 | Different: no ground-truth target scenes |
| Anand et al. (MuZero) | **VERIFY** (preprint) | 2021 | 2111.01587 | MuZero on Procgen, 500 training levels, zero-shot on unseen levels | Normalized return: 0.50 to 0.63 (contrastive), 0.64 (SPR) | Fig. 3, §5 | Different: agent return, not prediction quality |
| DreamerV3 (Hafner et al.) | **VERIFY** (Nature 2025) | 2023 | 2301.04104 | ProcGen at hard difficulty with the unlimited level set | Return | App. (ProcGen setup) | Different: no train/test level split |
| Kang et al. (physical law) | ICML 2025 | 2024 | 2411.02385 | DiT video models on 2D physics, in- versus out-of-distribution velocities | Velocity error parsed from video: DiT-L, 3M videos, ID 0.012, OOD 0.427 | §3.2, Fig. 3 | Partly: ID versus OOD in one simulator |
| ACWM-Phys (Xue et al.) | **VERIFY** (arXiv) | 2026 | 2605.08567 | Eight simulated environments with InD and OoD splits (e.g. unseen workspace regions) | MSE and perceptual metrics | Table 1, §5 | Partly: controlled OoD shifts, no persistence |

**Verdict: partly done.** Held-out evaluation exists, but always on 1 to 8 domains, usually with FVD or a normalized score, and never against copy-last. "Almost only in-domain" overstates it after XEWorld and Dreamer 4. Say instead: "world models are rarely evaluated on held-out scenes, and then on a handful of domains without a persistence reference." Cite XEWorld and Dreamer 4 beside yesterday's NWM, Vista and SCOPE.

## Step 2. Appearance shift versus dynamics shift

| Paper | Venue | Year | arXiv | What they did | Metric | Where | Same as ours? |
|---|---|---|---|---|---|---|---|
| XEWorld | **VERIFY** | 2026 | 2608.05799 | Appearance distance (cosine on reference renders) against kinematic distance (workspace Chamfer) | Pearson r with held-out LPIPS: 0.812 vs 0.549 | §5.4, Fig. 5, App. | Partly: the same question, robots not maps; concludes models are "2D visual pattern matchers" |
| Kang et al. | ICML 2025 | 2024 | 2411.02385 | Paired attribute conflicts in training data | Which attribute the generation copies; order color > size > velocity > shape; "case-based" generalization | §5, Figs. 8, 9 | Partly: appearance dominates retrieval of training cases |
| ACWM-Phys | **VERIFY** | 2026 | 2605.08567 | OoD splits by physics regime | OoD drop; authors conclude reliance on "appearance statistics" | §5, Table 2 | Partly |
| Distracting Control Suite (Stone et al.) | **VERIFY** (arXiv) | 2021 | 2101.02722 | DMC with color, background video and camera distractions at graded strength | Return | Abstract, §3 | Different: model-free RL policies |
| DMC-GB / SODA (Hansen and Wang) | **VERIFY** (ICRA 2021) | 2020 | 2011.13389 | DMControl Generalization Benchmark: random colors and video backgrounds at test time | Return | §5 | Different: policies |
| DMC Remastered (Grigsby and Qi) | **VERIFY** (arXiv) | 2020 | 2010.06740 | Randomly generated graphics for DMC | Return | abstract | Different: policies |
| TIA (Fu et al.) | ICML 2021 | 2021 | 2106.15612 | Dreamer on Cheetah Run with and without complex backgrounds, three model sizes | Return; performance "much worse" with background, recovers with capacity | Fig. 1 | Partly: shows world-model capacity spent on appearance |
| ContextWM (Wu et al.) | NeurIPS 2023 | 2023 | 2305.18499 | Separate context encoder (appearance) from latent dynamics | Downstream MBRL sample efficiency | §3 | Different: architecture, not evaluation |
| CaDM (Lee et al.) | ICML 2020 | 2020 | 2005.06800 | Dynamics shifts (e.g. body mass) with fixed appearance | Return, prediction error | Fig. 1, §5 | Opposite axis: dynamics-only shift |
| Schneider et al. | NeurIPS 2024 | 2024 | 2411.10175 | Model-based agents with pretrained visual encoders; OOD set is a held-out 20% of colors and sizes | Return ID and OOD | §3 (setup) | Partly |
| Yeom et al. | **VERIFY** | 2026 | 2606.07687 | Probe action recoverability under appearance perturbations | Inverse-dynamics R² | Table 2 | Partly |

**Verdict: done in adjacent fields, and the answer agrees with Rohan.** Where the two are separated, generation error tracks appearance distance and not kinematic distance (XEWorld), and video models copy training cases by color first (Kang). Two cautions for our framing. First, Doom maps change geometry as well as textures, so "same engine" is not a pure appearance shift; XEWorld held scenes byte-identical to isolate its factor, and we cannot. Second, XEWorld found global LPIPS dominated by background pixels (97 percent of the gap closed globally, 51 to 57 percent in the robot region), which argues for a motion-weighted or region metric beside our global PSNR. Cite XEWorld and Kang as the precedents, TIA for world-model capacity spent on background, and the Distracting Control Suite as the benchmark lineage. Keep the 19:15 wording ("latent prediction error on unfamiliar footage", not "shared physics").

## Step 3. Rendering ceiling versus latent prediction error

| Paper | Venue | Year | arXiv | What they did | Metric | Where | Same as ours? |
|---|---|---|---|---|---|---|---|
| OccWorld (Zheng et al.) | **VERIFY** (ECCV 2024) | 2023 | 2311.16038 | 0 s column is the tokenizer reconstruction (66.38 mIoU); Copy&Paste row copies the current occupancy | mIoU, IoU at 1 to 3 s: Copy&Paste average 11.33, OccWorld-O 17.14 | Table 1, §4.3 | Same shape: ceiling, persistence and model in one table |
| DOME (Gu et al.) | **VERIFY** (arXiv) | 2024 | 2410.10429 | "Recon." column and Copy&Paste row beside forecasting | mIoU, IoU | Tables 1, 2 | Same shape |
| DINO-Foresight (Karypidis et al.) | NeurIPS 2025 | 2024 | 2412.11673 | Forecast DINO features; Oracle uses the true future frame, Copy-Last the last context frame, plus Vista decoded and re-encoded | Segmentation mIoU (short term, DINOv2): Oracle 77.0, Copy Last 54.7, prediction 71.8 | Table 2, §4 | Same idea in feature space, not a pixel decoder |
| Luc et al. | ICCV 2017 | 2017 | 1703.07684 | Future segmentation with "Copy last input" and a Dilation10 oracle on the true future frame | mIoU | Tables 1, 2 | Same idea, 2017 |
| Kang et al. | ICML 2025 | 2024 | 2411.02385 | VAE reconstruction error of ground-truth videos compared with ground-truth error and OOD error | Velocity error: reconstruction close to ground truth, both an order of magnitude below OOD | App. B.3, Table 3 | Same: the ceiling is flat, so the error sits in prediction |
| GameNGen | ICLR 2025 | 2024 | 2408.14837 | Fine-tunes only the SD 1.4 decoder with MSE on game frames; autoregression unaffected | Qualitative only | §3.3, App. A.2 Fig. 12 | Different: no reconstruction number (text searched for "reconstruct") |
| DIAMOND | NeurIPS 2024 | 2024 | 2405.12399 | Pixel-space diffusion, no autoencoder | none | — | Not applicable |
| Genie | **VERIFY** (ICML 2024) | 2024 | 2402.15391 | Tokenizer ablation | PSNR of tokenizer variants | Table 3 (per yesterday) | Partly |

**Verdict: done in occupancy and semantic forecasting; new for game video world models.** Our per-map ceiling (23.91 dB training maps, 23.71 unseen) with persistence and model side by side follows OccWorld Table 1 and DINO-Foresight Table 2 exactly, and Kang's App. B.3 makes the same "flat ceiling, so the error is prediction" argument in one sentence. What nobody reports is the ceiling per held-out domain against a distance. Cite OccWorld and DINO-Foresight for the table layout, Kang for the argument, and GameNGen to say the Doom precedent tuned the decoder without reporting its ceiling. Astra's caveat stands: a ceiling on true frames does not bound the decoder on off-manifold predicted latents, and none of these papers tests that either.

## Step 4. Cheap adaptation without retraining the backbone

| Paper | Venue | Year | arXiv | What is trained, what is frozen | Data | Result | Where | Same as ours? |
|---|---|---|---|---|---|---|---|---|
| iVideoGPT (Wu et al.) | NeurIPS 2024 | 2024 | 2405.15223 | Tokenizer only; transformer frozen; also full fine-tune | Unseen BAIR; 100 or 1,000 trajectories | Tokenizer-only adaptation gives "similar perceptual quality" to full fine-tuning; pretraining helps only at 100 to 1,000 trajectories | §3 "Tokenizer adaptation", Fig. 8, Fig. 9a | Closest to decoder-only adaptation, but its tokenizer is also the encoder |
| GameNGen | ICLR 2025 | 2024 | 2408.14837 | Decoder only, MSE | Training data | Qualitative | §3.3 | Same operation, not a new domain |
| AVID | **VERIFY** | 2024 | 2410.12822 | Adapter with a learned mask on a frozen, weight-inaccessible model | Coinrun 100k, 500k, 2.5M | Coinrun500k FVD: AVID 71M 23.1, ControlNet 71M 18.5, full action fine-tune 97M 14.1, base 204.0 | Table 1, Fig. 4 | Partly |
| GameFactory | ICCV 2025 | 2025 | 2501.08325 | LoRA (rank 128) for game style, then action module with base and LoRA frozen; LoRA removed at inference | GF-Minecraft | Scene-generalization table | §5.2, Fig. 6, Table 3 | Partly: LoRA as a removable appearance adapter |
| DreamGen (Jang et al.) | **VERIFY** | 2025 | 2505.12705 | LoRA rank 4 on WAN 2.1 | 5 to 200 epochs per dataset | Notes "the optimal amount of fine-tuning ... differs" per model and data pair, unexplained | §2.1, App. hyperparameters | Partly |
| AdaWorld (Gao et al.) | ICML 2025 | 2025 | 2503.18938 | Whole model, 800 steps | 100 samples per action, 4 unseen environments | Habitat PSNR 23.58 against 20.34 action-agnostic pretraining | Table 2, Fig. 6 | Partly: adaptation curves in steps and samples |
| XEWorld | **VERIFY** | 2026 | 2608.05799 | Full DiT (Wan2.2-TI2V-5B), encoders frozen | 25, 50, 75 episodes | LPIPS −30 to −32% at 25 episodes, about −40% at 75; seen-robot LPIPS +69% | §5.3, Table 4, Fig. 4 | Partly: adaptation plus forgetting |
| RoboNet (Dasari et al.) | CoRL 2019 | 2019 | 1910.11215 | Fine-tune of a video predictor pretrained on RoboNet | 300 to 400 trajectories of a held-out robot | Beats training from scratch on all three robots (task success) | §5 text, Fig. 4 | Partly |
| DiffFit (Xie et al.) | **VERIFY** (arXiv tech report) | 2023 | 2304.06648 | Bias and scale factors of DiT only, 0.12% of parameters | 8 downstream datasets | FID 3.02 on ImageNet 512 | Fig. 1, Table 1 | Partly: PEFT for DiT domains |
| Video Adapter (Yang et al.) | **VERIFY** | 2023 | 2306.01872 | Small video model composed with a black-box large one | Task-specific data | Uses as few as 1.25% of the parameters | abstract | Partly |
| LoRA Learns Less and Forgets Less (Biderman et al.) | TMLR 2024 | 2024 | 2405.09673 | LoRA against full fine-tuning on code and math | 100K prompts; 20B tokens | LoRA "substantially underperforms" full fine-tuning and forgets less | abstract | Design input for step 8 |

**Verdict: done in general; decoder-only adaptation to an unseen map in a game world model not found.** iVideoGPT is the precedent for "adapt the autoencoder, keep the dynamics frozen" and reports a few-shot curve. The Sep 26 decision to drop the unseen-map decoder arm is consistent with the literature: nobody claims decoder adaptation as learning a new environment, and GameNGen used it only for HUD fidelity.

## Step 5. Middle layers, the LLM analogy, and diffusion features

LLM side:

| Paper | Venue | Year | arXiv | Finding | Where | Bears on the claim? |
|---|---|---|---|---|---|---|
| Let's Verify Step by Step (Lightman et al.) | **VERIFY** (ICLR 2024) | 2023 | 2305.20050 | Separate PRM predicts each step's correctness after the step's last token; PRM 78.2%, ORM 72.4%, majority vote 69.6% of 500 MATH problems (best-of-1860) | §2, Fig. 3 | No: process supervision, not layers |
| Layer by Layer (Skean et al.) | ICML 2025 | 2025 | 2502.02013 | Intermediate layers beat the final layer by up to 16% on 32 MTEB tasks | Fig. 1, §1 | Yes, for embeddings |
| DoLa (Chuang et al.) | ICLR 2024 | 2023 | 2309.03883 | Contrast a dynamically chosen early ("premature") layer with the final layer at decoding; +12 to 17 absolute points on TruthfulQA | abstract, §2, Fig. 3 | Yes: reading an earlier layer into the output helps |
| Tuned lens (Belrose et al.) | **VERIFY** (arXiv) | 2023 | 2303.08112 | Affine probe per block; logit lens "often brittle" | §1, §2 | Method |
| LayerSkip (Elhoushi et al.) | ACL 2024 | 2024 | 2404.16710 | Layer dropout plus early-exit loss; 1.34 to 2.16× speedups | abstract | Early exit, speed not quality |
| HSRM (Li and Zhu) | EMNLP 2026 | 2026 | 2608.30841 | Verifier on frozen generator hidden states; best layers in the upper portion; final layer competitive and default | §5 layer ablation, Fig. 3 | Partly: not middle |
| SWIFT (2505.12225) | KDD 2026 | 2025 | 2505.12225 | Linear probe of step correctness about 80% at every layer of Llama-3.1-8B | §4.1, Fig. 2 | Partly: flat across depth |

Diffusion and video side:

| Paper | Venue | Year | arXiv | Finding | Where | Same as ours? |
|---|---|---|---|---|---|---|
| DDAE (Xiang et al.) | ICCV 2023 | 2023 | 2303.09769 | Best linear-probe features lie in the middle of the U-Net up-sampling stage at small noise; the approach also transfers to latent DiT | Fig. 2, §1 | Partly |
| DIFT (Tang et al.) | NeurIPS 2023 | 2023 | 2306.03881 | Correspondence features depend on timestep and up-block (e.g. SD t = 261, block 1 for semantic) | §3, App. | Partly |
| Diffusion Hyperfeatures (Luo et al.) | NeurIPS 2023 | 2023 | 2305.14334 | Learned mixing over all layers and timesteps beats hand-picked ones | Fig. 1, §3 | Partly |
| REPA (Yu et al.) | ICLR 2025 | 2024 | 2410.06940 | Aligning an early block (layer 6 or 8) to DINOv2 is best; over 17.5× faster SiT training | §4, Table 2 | Yes: an auxiliary loss on an intermediate layer improves generation |
| Δ-DiT (Chen et al.) | **VERIFY** (arXiv) | 2024 | 2406.01125 | Front DiT blocks carry outline, rear blocks detail | §3, Fig. 3 | Partly |
| P2 weighting (Choi et al.) | CVPR 2022 | 2022 | 2204.00227 | Content forms at high noise, imperceptible detail at low noise | §3 | Timestep axis |
| eDiff-I (Balaji et al.) | **VERIFY** (arXiv) | 2022 | 2211.01324 | Text conditioning drives early denoising and is "almost entirely ignored" late | Figs. 3, 4 | Timestep axis |
| DeepCache (Ma et al.) | **VERIFY** | 2023 | 2312.00858 | High-level U-Net features barely change across adjacent steps and can be cached | §3 | Redundancy, not quality |
| AC3D (Bahmani et al.) | CVPR 2025 | 2024 | 2411.18673 | Linear probe of camera pose peaks in middle DiT blocks; condition only the first 8 of 32 blocks; camera motion is low-frequency and formed in the first few percent of steps | Fig. 4b, Fig. 5, §4 ablation | Closest: camera is our yaw |
| Video Prediction Policy (Hu et al.) | ICML 2025 | 2024 | 2412.14803 | Policy reads aggregated up-sampling features from one forward pass; replacing them with final-layer features drops CALVIN length 4.33 to 3.60 | §5.2 ablation | Yes: intermediate video-diffusion features improve control |
| Yeom et al. | **VERIFY** | 2026 | 2606.07687 | V-JEPA 2 per-layer action R² peaks at layer 14 (0.51), drops to 0.39 at layer 22 | Fig. 3 | Partly |
| DINO-Foresight | NeurIPS 2025 | 2024 | 2412.11673 | Intermediate features of the forecaster improve downstream tasks | App. A.2 (text) | Partly |

**Verdict: the general claim is done; the specific proposal is not.** Middle layers often beat the final layer for readout (Layer by Layer, DDAE, AC3D, Yeom), and reading an earlier layer into the output can help (DoLa, VPP; REPA at training time). "Let's Verify" should not be cited for it; cite it only if the paper discusses process-style verification. No paper adds a middle-layer head to an action-conditioned world model to improve prediction or control fidelity. AC3D is the closest and cuts against a depth-only story: camera information is formed early in denoising and early in depth, then consumed by later blocks. Cite AC3D, DDAE and REPA for diffusion, Layer by Layer and DoLa for LLMs, and keep the October future-work sentence.

## Step 6. Where conditioning acts

| Paper | Venue | Year | arXiv | What they did | Result | Where | Same as ours? |
|---|---|---|---|---|---|---|---|
| Guidance interval (Kynkäänniemi et al.) | NeurIPS 2024 | 2024 | 2404.07724 | Apply CFG only in a noise window | Harmful at high noise, unnecessary at low; ImageNet-512 FID 1.81 to 1.40 | abstract, Fig. 2 | Method template for a timestep sweep |
| eDiff-I | **VERIFY** | 2022 | 2211.01324 | Switch the prompt after a fixed fraction of steps | Early steps set text-aligned content | Fig. 4 | Timestep-restricted swap |
| Prompt-to-Prompt (Hertz et al.) | **VERIFY** (ICLR 2023) | 2022 | 2208.01626 | Inject source cross-attention for τ steps | Composition is set early; more steps, more fidelity to source | Fig. 6, §3 | Timestep-restricted swap |
| MagicMix (Liew et al.) | **VERIFY** (arXiv) | 2022 | 2210.16056 | Layout from early steps, content from later | Qualitative | §3 | Partly |
| Motion encoding in video timesteps (Baherwani et al.) | **VERIFY** (arXiv) | 2025 | 2512.22175 | Inject new conditions over timestep ranges and measure appearance change against motion preservation | Early motion-dominant regime, later appearance-dominant, across architectures | abstract, §4 | Closest method to Astra's t-sweep |
| AC3D | CVPR 2025 | 2024 | 2411.18673 | Restrict camera conditioning to early (low-frequency) steps and first 8 blocks | About 15% better visual fidelity, about 30% better camera following; conditioning all 32 blocks costs about 10% quality | §1, §4 ablation | Closest for control signals |
| Basu et al. (LocoGen) | ICML 2024 | 2024 | 2405.01008 | Causal tracing and interventions over U-Net layers | Attributes localize to a few cross-attention layers; for SD-XL and DeepFloyd plain causal tracing fails | abstract | Activation patching in diffusion |
| Staniszewski et al. | ICLR 2025 | 2025 | 2502.09935 | Attention activation patching including joint-attention (SD3-style) models | Under 1% of parameters, all attention, control text rendering; LoRA on those layers improves it | abstract | Patching in MMDiT, relevant to SD 3.5 routes |
| Surkov et al. (SAE on SDXL Turbo) | **VERIFY** | 2024 | 2410.22366 | Sparse autoencoders on U-Net blocks | Blocks specialize into composition, detail, style; features act causally | §1, App. H | Partly |
| CoCo (Shi et al.) | **VERIFY** | 2026 | 2608.04653 | Inverse-action, zero-action and mirrored-scene consistency for action world models | Action Response Consistency, Drift Energy | §3, tables | Our directional check's nearest relative; no layer or timestep attribution |
| Twin Rollouts (Ma et al.) | **VERIFY** | 2026 | 2608.08982 | Noise-coupled factual and counterfactual action branches | Spatiotemporal locality against simulator forks | abstract | Counterfactual action evaluation |

**Verdict: done for text and camera, new for game actions.** Timestep-restricted swaps and activation patching are standard in text-to-image, and AC3D and 2512.22175 do it for camera and motion in video DiTs. No paper found attributes a game action's effect to timesteps or blocks of a world model. Astra's optional t-sweep would be the first such measurement for a game world model; frame it as AC3D's analysis applied to a discrete action. For SD 3.5, cite Staniszewski et al. for patching joint attention.

## Step 7. EMA versus live weights and closed-loop collapse

| Paper | Venue | Year | arXiv | Finding | Where | Same as ours? |
|---|---|---|---|---|---|---|
| EDM2 (Karras et al.) | **VERIFY** (CVPR 2024) | 2023 | 2312.02696 | Post-hoc EMA; FID depends strongly on EMA length, the optimum narrows with better configs, and interacts with guidance | §3, Fig. 5 | Partly: one-shot image quality, no rollouts |
| Morales-Brotons et al. | TMLR 2024 | 2024 | 2411.18704 | EMA models generalize better, with better prediction consistency, calibration and robustness to label noise | abstract | Partly: classifiers |
| Persistent Robot World Models (Bardhan et al.) | ECCV 2026 (per arXiv comment) | 2026 | 2603.25685 | EMA only on the RL post-training policy and reference; differences at most about 0.3 dB | Tables 10, 11 | Different: not base-model EMA |
| DIAMOND | NeurIPS 2024 | 2024 | 2405.12399 | DDPM with few denoising steps drifts out of distribution; EDM parameterization stays stable even at one step | §5, Fig. 3 | Relevant: our sampler is respaced ancestral DDPM |
| BAgger (Po et al.) | **VERIFY** | 2025 | 2512.12080 | Long rollouts show progressive over-saturation, over-smoothing, near-static late frames (Diffusion Forcing, Self Forcing) | §5 | Partly: saturation drift, a channel-level symptom |
| Chen, Zhang, Wang | **VERIFY** | 2026 | 2607.27036 | Effective rank of hidden states collapses at the onset of drift; more data does not help | abstract, §3 | Partly: an internal marker of collapse |
| Frozen Flows Forget (Chen et al.) | **VERIFY** | 2026 | 2609.28414 | Latent world model loses motion; pixel L1 rewards stillness, and the lowest-L1 variant is the most static | §1 | Partly: static absorbing state favored by the loss |

**Verdict: new.** No paper found compares EMA and live weights for closed-loop stability of a world model; EMA is either assumed (reported samples from EMA) or studied for one-shot FID. Our observation (EMA reduces but does not remove collapse; channel-13 end state) is a new empirical note. The drift literature supplies context: saturation drift (BAgger), rank collapse (2607.27036), and static solutions favored by pixel losses (Frozen Flows). DIAMOND Fig. 3 is directly relevant because it attributes rollout drift to the DDPM sampler at few steps, which a reviewer may raise for our respaced ancestral DDPM. Cite EDM2 for EMA sensitivity and DIAMOND for the sampler caveat; present the EMA result as an observation, not a claim of novelty over exposure-bias work.

## Step 8. Adaptation curves predicted by the training-to-map distance

The proposal: adapt the four-map model to each unseen map with LoRA (or another small adapter), plot gain over persistence against LoRA steps and against episodes, and test whether the frozen distance predicts how much adaptation each map needs.

### 8a. Predicting fine-tuning outcome before training

| Paper | Venue | Year | arXiv | Predictor | Predicted quantity | Evaluation | Where | Relation |
|---|---|---|---|---|---|---|---|---|
| OTDD (Alvarez-Melis and Fusi) | NeurIPS 2020 | 2020 | 2002.02923 | OT dataset distance | Relative error drop from pretraining | ρ and p per panel, n = 11 to 16 | Figs. 6, 7 (yesterday) | Distance predicts fine-tuned outcome |
| s-OTDD (Nguyen et al.) | ICML 2025 | 2025 | 2501.18901 | Sliced OT dataset distance | Performance gap = full-target accuracy minus adapted accuracy | ρ = 0.40 on *NIST (OTDD exact about the same) | §4.2, Fig. 4 | Sliced Wasserstein, like our D; also states OTDD reported Spearman only (§4.1), which answers yesterday's open question secondhand |
| OTCE (Tan et al.) | CVPR 2021 | 2021 | 2103.13843 | Wasserstein domain difference plus conditional-entropy task difference | Transfer accuracy | correlation | §3 | Splits domain from task difference |
| LEEP (Nguyen et al.) | ICML 2020 | 2020 | 2002.12462 | Expected empirical prediction of the source classifier | Transfer accuracy and **convergence speed** relative to a from-scratch reference | Pearson r above 0.94 in the main setting | §5.1, §5.5, Fig. 4 | Closest to "predict the adaptation curve" |
| LogME (You et al.) | ICML 2021 | 2021 | 2102.11005 | Maximum label evidence | Fine-tuned accuracy ranking | Weighted Kendall τ; up to 3000× faster than fine-tuning | §4 | Model selection |
| Gao and Chaudhari | **VERIFY** (ICML 2021) | 2020 | 2011.00613 | Coupled transfer distance | Fine-tuning distance = weight-trajectory length until 95% of final validation accuracy, called the "gold standard" | Mantel-style r with p; r = 0.428, p = 0.13 (Fig. 2a) | §5.2, Figs. 2, 3 | Defines adaptation cost as the ground truth, as we would |
| Achille et al. | **VERIFY** (arXiv) | 2019 | 1904.03292 | Asymmetric task distance | A lower bound on the cost of transfer | Theory plus correlations | §1 | Theory for "distance predicts cost" |
| Task2Vec | ICCV 2019 | 2019 | 1902.03545 | Fisher embedding | Model selection | — | yesterday | Weak in Gao and Chaudhari's test (r = 0.03) |

### 8b. Few-sample adaptation of world, video and diffusion models

See the step-4 table (iVideoGPT, AVID, GameFactory, DreamGen, AdaWorld, XEWorld, RoboNet, DiffFit) plus:

| Paper | Venue | Year | arXiv | Trained / frozen | Samples | Result | Where |
|---|---|---|---|---|---|---|---|
| Transferring GANs (Wang et al.) | ECCV 2018 | 2018 | 1805.01677 | Full fine-tune from several sources | 1,000 target images and up | Pretrained GAN needs 2 to 5× fewer images for the same score; source choice matters; density beats diversity | Tables 2, 4 | 
| AdAM (Zhao et al.) | NeurIPS 2022 | 2022 | 2210.16559 | Kernel modulation of a frozen generator | 10-shot | Measures source-target proximity with FID and LPIPS; methods tuned for close domains fail on distant ones | Fig. 2, Table 2 |
| Zhu et al. | **VERIFY** (arXiv) | 2022 | 2211.03264 | Fine-tuned DDPMs | 10-shot | Even unrelated sources converge faster than scratch | App., Table 12 |
| Oh et al. | NeurIPS 2022 | 2022 | 2202.01339 | Supervised versus self-supervised pretraining | 5-way 1- and 5-shot, 8 targets | Separates domain similarity (exp(−α·EMD), α = 0.01) from target-intrinsic few-shot difficulty | §3, Fig. 1 |
| BSCD-FSL (Guo et al.) | ECCV 2020 | 2019 | 1912.07200 | Meta-learning and fine-tuning | few-shot | Accuracy of all methods tracks similarity to natural images, over 4 targets ranked by 3 qualitative criteria | abstract, Fig. 1 |

### 8c. Adaptation cost against distance, quantified

| Paper | Venue | Year | arXiv | Curve | n pairs | Where |
|---|---|---|---|---|---|---|
| Hernandez et al., Scaling Laws for Transfer | **VERIFY** (arXiv) | 2021 | 2102.01293 | Effective data transferred D_T = k·D_F^α·N^β; α is read as "directed proximity"; text to Python α = 0.18, mixed text and code 0.096, β = 0.38 | 2 source distributions | eq. 1.1, Table 1 |
| Mikami et al. | **VERIFY** (preprint) | 2021 | 2108.11018 | Synthetic-to-real error as a power law in pretraining data with a transfer-gap floor C that shrinks with model size | tasks **VERIFY** | abstract, §4 |
| Isik et al. | ICLR 2025 | 2024 | 2402.04177 | Downstream scaling is monotone only when pretraining and fine-tuning distributions align | translation pairs | abstract |
| XEWorld | **VERIFY** | 2026 | 2608.05799 | LPIPS against 25/50/75 adaptation episodes for a "near" (Piper) and a "far" (Franka) robot, with forgetting on UR5 | 2 targets for curves, 5 for zero-shot distance | Table 4, Fig. 4 |
| LEEP | ICML 2020 | 2020 | 2002.12462 | Fine-tuned accuracy minus from-scratch reference over epochs, grouped by score | many targets | §5.5, Fig. 4 |
| Xie et al.; Lin et al. (robotics) | see yesterday | | 2307.03659; 2410.18647 | Generalization gap against shift radius; power laws in number of environments | | yesterday Q2 |

**Verdict for step 8: partly done, and new in this domain as a combination.**

- In general, "distance predicts transfer" and "a score predicts convergence speed" are established (OTDD, s-OTDD, OTCE, LEEP, Gao and Chaudhari). Hernandez et al. already write proximity into the exponent of the adaptation curve, from two sources.
- In world models, adaptation curves exist (AdaWorld Fig. 6, iVideoGPT Fig. 9a, XEWorld Table 4, AVID dataset sizes), but over 1 to 4 targets, none predicted from a pre-computed distance. XEWorld is the closest: its two held-out robots differ in appearance distance, and it measures episodes-to-recovery and forgetting. It fine-tunes the full DiT, and its distance analysis covers zero-shot error only.
- DreamGen states the unexplained version of our question: the right amount of fine-tuning differs per model and data pair.

What would make ours distinct, in order of strength:

1. **Many targets.** About 30 maps against XEWorld's 2 adaptation targets and Hernandez's 2 sources. Choose the adapted maps (8 to 10, as the 18:30 entry planned) to span D, and freeze D before any adaptation run, as with the zero-shot study.
2. **Predict curve parameters, not a point.** Fit each map's curve (gain over persistence against episodes, and against steps) and regress a parameter on D. Candidate parameters: episodes to reach parity with persistence, gain at a fixed budget, or Hernandez-style exponent. LEEP's "fine-tuned minus reference over time" and Gao and Chaudhari's "trajectory length until 95 percent of final accuracy" are the two published definitions of adaptation cost to borrow.
3. **A persistence-referenced outcome and the rendering ceiling.** No adaptation curve found uses copy-last as the reference. Report the latent ratio beside PSNR so decoder effects cannot masquerade as adaptation.
4. **Forgetting on the four training maps.** XEWorld found +69 percent LPIPS on a seen robot after 25 episodes. Measure it per map; LoRA should forget less (Biderman et al.), which is itself a result.
5. **Adapter capacity as a second cost axis.** Biderman et al. report LoRA underperforming full fine-tuning on distant domains. If far maps need higher rank, rank at parity is a second distance-dependent cost; include full fine-tuning at one or two maps as the ceiling arm.
6. **Separate map difficulty from distance.** Oh et al. show target-intrinsic difficulty confounds similarity. Our per-map persistence PSNR and the zero-shot gain already play that role; keep the partial correlation on persistence.

Risks the literature flags: global metrics hide the part that matters (XEWorld's region LPIPS), so keep a motion-weighted read; the correlation from 8 to 10 maps will be weak (s-OTDD reports ρ = 0.40 with many more pairs; Gao and Chaudhari r = 0.43, p = 0.13), so pre-register the statistic and report intervals.

Contribution sentence the literature supports: "We measure what it takes for a Doom world model trained on four maps to cross to each of N unseen maps, and show that a distance computed before adaptation predicts the episodes needed to beat persistence." Cite XEWorld, Hernandez et al., LEEP and Gao and Chaudhari as the closest work, OTDD and s-OTDD for the distance family, and AdaWorld and iVideoGPT for world-model adaptation.

## What to cite, by section of the paper

- Evaluation on held-out maps: XEWorld, Dreamer 4, AVID; yesterday's NWM, Vista, SCOPE, PredNet.
- Appearance versus dynamics: XEWorld, Kang et al. ICML 2025, TIA, Distracting Control Suite.
- Rendering ceiling: OccWorld, DINO-Foresight, Kang App. B.3; GameNGen for decoder fine-tuning without a ceiling.
- Control attribution and layers: AC3D, guidance interval, 2512.22175, Staniszewski et al., DDAE, REPA; Layer by Layer and DoLa only if the LLM analogy appears.
- Closed loop: EDM2, DIAMOND Fig. 3, BAgger, 2607.27036.
- Adaptation (October): XEWorld, Hernandez et al., LEEP, Gao and Chaudhari, OTDD, s-OTDD, AdaWorld, iVideoGPT, Biderman et al.

## Sources read (arXiv PDFs, 2026-09-26)

2305.20050, 2303.08112, 2404.16710, 2309.03883, 2502.02013, 2608.30841, 2505.12225, 2306.03881, 2305.14334, 2410.06940, 2312.00858, 2211.01324, 2204.00227, 2303.09769, 2406.01125, 2412.14803, 2411.18673, 2506.17220, 2410.10802, 2512.22175, 2404.07724, 2208.01626, 2210.16056, 2210.04885, 2405.01008, 2502.09935, 2410.22366, 2509.24527, 2608.05799, 2505.14357, 2111.01587, 2301.04104, 2501.08325, 2605.08567, 2411.02385, 2503.18938, 2011.13389, 2101.02722, 2010.06740, 2305.18499, 2206.15477, 2106.15612, 2005.06800, 2307.10224, 2411.10175, 2606.07687, 2608.09926, 2311.16038, 2410.10429, 2405.20337, 2412.11673, 1703.07684, 2408.14837, 2405.12399, 2410.12822, 2306.01872, 2304.06648, 2502.07825, 2203.13880, 2405.15223, 2505.12705, 2501.03575, 1910.11215, 2310.16828, 2312.02696, 2411.18704, 2603.25685, 2512.12080, 2607.27036, 2609.28414, 2504.12626, 2608.04653, 2608.08982, 2002.12462, 2102.11005, 2103.13843, 2011.00613, 1904.03292, 2102.01293, 2108.11018, 2402.04177, 1912.07200, 2202.01339, 1805.01677, 2205.03805, 2210.16559, 2211.03264, 2501.18901, 2405.09673. Titles and venue comments were checked against the arXiv API for every ID.
