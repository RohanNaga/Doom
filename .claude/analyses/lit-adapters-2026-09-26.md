# Adapters and few-sample adaptation of generative world models: literature (2026-09-26)

Scope: prior work for the LoRA adaptation study in `docs/lora_adaptation_design_2026-09-26.html`. The design pretrains on four Doom maps, adds LoRA to frozen backbone weights, adapts to an unseen map from a few episodes, and plots adaptation curves against steps and episodes. A training-to-map distance is meant to predict the cost. This file covers parameter-efficient and few-sample adaptation only. Distance-versus-performance work is in `distance-study-literature-2026-09-25.md` and is not repeated here. Transferability estimation and scene-generalisation benchmarks belong to other readers.

Method: every number below was read from the arXiv PDF text, arXiv HTML, a proceedings PDF, or raw repository source on 2026-09-26. Where a figure is an image, I rendered the page and read the axes (AdaWorld Fig. 6). **VERIFY** marks a claim I could not confirm from a primary source, usually a venue.

## Bottom line

- LoRA on a frozen diffusion backbone is the default recipe for image and video diffusion. Rank 16 with alpha equal to rank is a documented default for a driving world model (Vista) and for NVIDIA's own world-model LoRA (Cosmos-Predict2). Our rank-16 choice is citable.
- Adaptation *curves* for world models exist, but only a few. AdaWorld (ICML 2025) plots PSNR against fine-tuning steps at several sample counts on unseen environments. AVID (RLC 2025) plots performance against adapter size and target dataset size on a held-out Procgen game. iVideoGPT (NeurIPS 2024) plots FVD against target trajectories. None uses LoRA, and none puts a distance on the x axis.
- No paper found adapts a game world model to an unseen *level of the same game* with a small adapter. The nearest are AVID (a trained adapter on a frozen model, adapted to the held-out game Coinrun) and GameFactory (LoRA used for game *style*, rank 128). The claim is **partly new**: the method is standard, but the per-map cost curve across many unseen maps, set against a distance, is not in this slice of the literature.
- Three results argue against a LoRA-only answer. DiffFit reports LoRA r = 8 and 16 failing badly on DiT-XL/2 transfer. AVID and iVideoGPT find full fine-tuning at or above every adapter. SCOPE finds a frozen backbone worse than end-to-end training for an FPS model. These support Decision 4 (train the embeddings in full) and Decision 5 (a full fine-tune reference).

## A. Adapters on diffusion U-Nets and DiTs

| Paper | Venue, arXiv | Frozen / trained | Target data | Metric and result | Where | vs ours |
|---|---|---|---|---|---|---|
| LoRA, Hu et al. | ICLR 2022 (venue from Vista's bibliography; arXiv comment silent), 2106.09685 | Frozen transformer; ΔW = BA on W_q, W_v in most experiments, scaled by α/r | GPT-3, GLUE | Rank 1 suffices for {W_q, W_v} on WikiSQL and MultiNLI; α set "to the first r we try" and not tuned | §4.1, §7.1–7.2, Tables 5–6 | different domain; origin of α = r |
| cloneofsimo/lora | GitHub, Dec 2022 | Frozen SD U-Net; LoRA on CrossAttention, Attention, GEGLU linears; default r = 4 | 9 to 21 subject images | Advises lr about 1e-4 for LoRA against about 1e-6 for full DreamBooth; example uses rank 16 on 21 images | README; `lora_diffusion/lora.py` | practice origin |
| DreamBooth, Ruiz et al. | CVPR 2023, 2208.12242 | *Full* fine-tune of all layers, prior-preservation loss | 3 to 5 images | About 1,000 iterations, lr 1e-5 (Imagen) and 5e-6 (SD) | §3, §4 | different (full FT, subject not domain) |
| ControlNet, Zhang et al. | ICCV 2023 (CVF), 2302.05543 | SD locked; trainable copy of 12 encoder blocks and middle block, zero-convolution links | 1k to 3m pairs | "Sudden convergence" near step 6,133 (Fig. 4); training does not collapse at 1k images (Fig. 10, qualitative) | Figs. 3, 4, 10 | adapter precedent; the data-size figure is qualitative |
| T2I-Adapter, Mou et al. | AAAI 2024 **VERIFY**; arXiv says "Tech Report", 2302.08453 | SD fixed; external adapter of about 77M parameters (18M and 5M variants) | Condition-image pairs | Condition fidelity | Figs. 3, 12 | adapter precedent |
| AnimateDiff, Guo et al. | ICLR 2024 (PDF header), 2307.04725 | Image layers frozen; "domain adapter" = LoRA on the T2I model; MotionLoRA = LoRA on the motion module's self-attention | MotionLoRA: 20 to 50 reference videos, 2,000 iterations, about 30 MB | Fig. 7: rank 2 (about 1M) vs rank 128 (about 36M) comparable; N = 5 videos degrades, N = 50 works (qualitative) | §4.1, §4.3, Fig. 7 | partly: LoRA on a frozen video model with a small-N ablation, but no curve |
| DiffFit, Xie et al. | ICCV 2023 (CVF), 2304.06648 | DiT-XL/2 (ImageNet 256) frozen except bias, norm, class embedding and new scale factors γ, 0.12% of parameters | 8 fine-grained datasets, 24k iterations, batch 256 | Mean FID over 8 sets: full FT 16.59, BitFit 16.82, DiffFit 15.39, **LoRA-R8 81.25, LoRA-R16 81.31**; best lr is 10× pretraining | Table 1, Table 4d, **Fig. 6 (FID every 15k iterations, 4 datasets)** | **same backbone family; an adaptation curve; LoRA reported failing** |
| LoRA Learns Less and Forgets Less, Biderman et al. | TMLR 2024, 2405.09673 | LoRA vs full FT on an LLM | Code and math | LoRA underperforms full FT on target but forgets less off-target | abstract | backs our forgetting guard |

**Notes.**
- The DiffFit LoRA row is the most decision-relevant item in this section. On the DiT-XL/2 we descend from, rank-8 and rank-16 LoRA trail full fine-tuning by roughly 65 FID on average. BitFit (biases only) matches full fine-tuning. The authors attribute the gap to LoRA being weaker on vision tasks (§4.2). They do not rule out an implementation cause, and later practice (Vista, Cosmos, the diffusers DiT scripts) uses LoRA on DiTs successfully. The safe reading is that the parts our design trains in full matter: control MLP, positions, input projection and noise embedding. These are DiffFit's kind of parameters. A LoRA-only run would be exposed to DiffFit's failure mode.
- DiffFit's learning-rate search (Table 4d) found 10× the pretraining rate best, with instability above that. Vista's LoRA phase used 5× its pretraining rate (5e-5 against 1e-5). Our design calls "twice the pretraining rate" the usual LoRA convention. I found no source for 2×. The documented choices are an absolute 1e-4 (diffusers, cloneofsimo) or 5× to 10× pretraining.

**Video-diffusion LoRA tooling.** None of the CogVideoX (ICLR 2025, 2408.06072), Wan (2503.20314) or HunyuanVideo papers reports its own LoRA recipe. The public LoRA practice lives in training scripts:
- diffusers `train_cogvideox_lora.py`: rank 128, alpha 128, lr 1e-4, targets to_q, to_k, to_v, to_out.0.
- DiffSynth-Studio (Wan 2.1): rank 32, alpha = rank, lr 1e-4, targets q, k, v, o, ffn.0, ffn.2.
- musubi-tuner (HunyuanVideo): network_dim 32, lr 2e-4.

## B. World-model and simulator post-training

| Paper | Venue, arXiv | Frozen / trained (rank, targets) | Target samples | Metric and gain | Where | vs ours |
|---|---|---|---|---|---|---|
| Vista, Gao et al. | NeurIPS 2024, 2405.17398 | Phase 2: pretrained SVD-based U-Net **frozen**; **LoRA rank 16** plus new action projections in all attention blocks; lr 5e-5, batch 8, 120k iterations; then unfrozen for 10k at high resolution | OpenDV-YouTube plus nuScenes, 1:1 | App. D.5, Fig. 16: projections alone on a frozen U-Net fail to learn control; adding LoRA is "essential" (30k-iteration comparison) | §3.2, App. C.3, D.5 | **same recipe (frozen backbone, LoRA r16, new condition path)**; the target is a new control, not a new domain |
| Cosmos-Predict1, NVIDIA | tech report, 2501.03575 | Post-training for camera control, robotics and driving; the text says "fine-tune" and names no frozen parts (full fine-tune **VERIFY**) | 1X: about 12,000 episodes; Bridge: about 20,000; DL3DV-10K | Human preference against VideoLDM (Fig. 24); camera metrics (Table 22) | §6 | different: full post-training, large target sets, no curve |
| Cosmos-Predict2 LoRA docs | GitHub `nvidia-cosmos/cosmos-predict2` | Default **rank 16, alpha 16**, targets q, k, v, output_proj, mlp.layer1, mlp.layer2 for 2B; 32 for 14B; lr presets 2^-12 to 2^-9; 1,000 to 2,000 iterations | Examples start from 4 videos | none reported | `documentations/post-training_video2world_lora.md` | recipe source |
| GameFactory, Yu et al. | ICCV 2025 Highlight, 2501.08325 | Phase 1: LoRA (rank 128, lr 1e-4) fits Minecraft style on an open-domain video model; Phase 2: base and LoRA frozen, action module trained (lr 1e-5); LoRA removed at inference | 70 h GF-Minecraft | Open-domain control: multi-phase Flow 54.13, FID 121.18 against one-phase 76.02, 167.79 | §4.2, Fig. 6, Table 3 | partly: LoRA as a game-domain adapter, but used to *absorb* style, not to reach a new level |
| Micro-World, AMD | ROCm blog, 5 Feb 2026 (not peer reviewed) | Wan 2.1 frozen; game-style LoRA, then action module; "original model parameters remain frozen during both stages" | over 6,000 clips of 81 frames | qualitative | blog | same pattern as GameFactory |
| AdaWorld, Gao et al. | ICML 2025, 2503.18938 | Latent-action pretrained world model (SVD-based); **full fine-tune**, pretrained weights at 0.1× lr (5e-5 × 0.1), batch 32, 800 steps | 100 samples per action (Habitat, Minecraft, DMLab); 100 trajectories (nuScenes); curves at 50 to 200 per action and 100 to 300 trajectories | Table 2 at 800 steps, Minecraft PSNR 21.59 vs 19.44 action-agnostic | **Fig. 6: PSNR vs steps 0 to 800, one panel per sample count** | **partly: the closest adaptation-curve figure; full FT, unseen environments rather than levels, no persistence reference** |
| AVID, Rigter et al. | RLC 2025 (RLJ PDF), 2410.12822 | Pretrained model untouched (weights not even accessed); separate 3D U-Net adapter (22M or 71M) outputs a mask and a correction to the frozen model's noise prediction | Pretrained on 15 Procgen games; adapted to held-out **Coinrun** with 100k, 500k or 2.5M steps from 100, 500 or 2,500 levels; tested on levels 10,000 to 11,000 | Coinrun500k PSNR: base 17.3, AVID-71M 23.8, ControlNet-71M 25.5, **full action FT 25.8** | Table 1; **Fig. 4b (vs adapter size), Fig. 4c (vs dataset size)** | **partly: frozen game model plus small adapter on an unseen game, evaluated on unseen levels, with a data-size curve** |
| Video Adapter (Probabilistic Adaptation), Yang et al. | ICLR 2024 **VERIFY**, 2306.01872 | Large model black-box; a small domain model (0.07 to 2.8B) composed as a product of experts | Bridge, Ego4D | Bridge FVD: pretrained 350.1, small 186.8, small + pretrained 177.4 | Table 1 | partly: adapter on a frozen video prior, no curve |
| UniSim, Yang et al. | ICLR 2024, 2310.06114 | Trained from scratch on a mixture; no parameter-efficient adaptation of the simulator | — | Policies trained in simulation transfer | §4 | different |
| DreamGen, Jang et al. | **VERIFY** venue, 2505.12705 | Wan 2.1 with **LoRA rank 4, alpha 4, lr 1e-4**, chosen "to mitigate forgetting" | 10 to 13 real trajectories per task; tens of epochs | Downstream policy success | §2.1, App. D | partly: LoRA world-model adaptation to a new robot |
| Dexterous World Models | **VERIFY**, 2512.17907 | CogVideoX-Fun 5B frozen; LoRA rank 64 plus image projection trained | synthetic plus real interaction video | — | §4 implementation (HTML) | recipe point |
| Vid2World | ICLR 2026, 2505.14357 | DynamiCrafter fine-tuned in full into a causal action-conditioned model | RT-1, CS:GO | — | §3 to 4 | different |
| Hunyuan-GameCraft | **VERIFY**, 2506.17201 | HunyuanVideo fine-tuned on 1M+ clips from over 100 AAA games | large | — | §3 | different (large-data full FT) |
| SCOPE | **VERIFY**, 2605.23345 | FPS world model; ablation of a frozen backbone vs two-stage vs end-to-end | CrossFPS, 69k clips | FVD 775.4 frozen, 732.1 two-stage, 690.3 end-to-end | Table 2 | caution: freezing costs quality in an FPS model |
| GameNGen | ICLR 2025, 2408.14837 | All SD 1.4 U-Net parameters unfrozen, lr 2e-5 | 5 levels, no new-level adaptation | — | §4.2 | our pretraining recipe; no adaptation |

No technical report for Genie 2, Genie 3 or Oasis turned up in the searches, and none of those models has a published adaptation recipe (**VERIFY**: blog posts only). I found no GameNGen reproduction fine-tuned to new Doom levels. MultiGen (2603.06679) trains on 100 maps and does not adapt.

## C. Model-based RL adaptation with a pretrained world model

| Paper | Venue, arXiv | What is transferred / frozen | Target samples | Result | Where | vs ours |
|---|---|---|---|---|---|---|
| TD-MPC2, Hansen et al. | ICLR 2024, 2310.16828 | 19M multitask model pretrained on 70 tasks; **full** fine-tune on 10 held-out tasks | 20k environment steps (20 DMControl or 100 Meta-World episodes) | About 2× over scratch | Fig. 8, App. E | partly: few-episode curve, full FT, reward not frames |
| APV, Seo et al. | ICML 2022, 2203.13880 | Action-free video model pretrained on RLBench; action-conditional model stacked on top; ablations freeze the representation model | Meta-World online | Learning curves vs environment steps; frozen vs all-trained lines | Figs. 3, 7b–c | partly |
| iVideoGPT, Wu et al. | NeurIPS 2024, 2405.15223 | Full model including tokenizer; the authors found this "more effective than parameter-efficient fine-tuning" | BAIR: 100, 1,000 trajectories, or full | Pretraining matters only when target data is scarce; 1,000 action-conditioned trajectories give FVD 82.3 | §3.3, **Fig. 9a (FVD vs target data and strategy)** | partly: data curve; warns against PEFT |
| ReDRAW, Lanier et al. | **VERIFY** venue, 2504.02252 | DreamerV3-style world model **frozen**; small latent-dynamics residual MLP trained | 4×10^4 offline target transitions; 10k causes overfitting | Residual avoids overfitting for 3M updates, while unfrozen fine-tuning overfits | Figs. 2–3 | **partly: frozen world model plus small adapter, with a data-size study** (sim-to-real, state RL) |
| Walker et al. | ICML (year **VERIFY**), 2302.04009 | Pretrained model components carried or reinitialized | Crafter, RoboDesk, Meta-World | Model-based transfer beats model-free | Fig. 1 | different (new reward, same world) |
| Continual-Dreamer, Kessler et al. | CoLLAs 2023, 2211.15944 | Replay-based continual world model | Minigrid, Minihack | Less forgetting with selective replay | abstract | different; adapter-free |
| CoLA-World | **VERIFY**, 2510.26433 | Latent-action world model fully fine-tuned; 2-layer MLP maps real actions | LIBERO (unseen) | PSNR, LPIPS, FVD | Tables 1, 5 | partly, like AdaWorld |

## D. Papers that report an adaptation curve

| Paper | x axis | y axis | Adapter or full | Curve shape reported |
|---|---|---|---|---|
| AdaWorld Fig. 6 | fine-tuning steps 0 to 800; one panel per sample count | PSNR | full (0.1× lr on pretrained weights) | Most of the gain by 200 steps; the action-agnostic baseline stays flat |
| AVID Fig. 4b, 4c | adapter parameters; target dataset size | normalized mean of all metrics | adapter vs ControlNet vs full | Methods keep their order as data grows |
| DiffFit Fig. 6 | iterations (every 15k) | FID | several PEFT methods and full | DiffFit and full converge at similar rates; VPT slowest |
| iVideoGPT Fig. 9a | target trajectories | FVD | full vs scratch | Pretraining helps at 100 and 1,000 trajectories, not at full data |
| TD-MPC2 Fig. 8 | environment steps to 20k | normalized score | full | 2× over scratch |
| ReDRAW Figs. 2–3 | updates; dataset size | return | frozen plus residual vs fine-tune | Fine-tune overfits; the residual holds |
| AnimateDiff Fig. 7, ControlNet Fig. 10 | reference videos 5, 50, 1,000; images 1k, 50k, 3m | qualitative samples | LoRA; adapter | No numbers |

None of these reports a crossing point (steps or samples to reach a target) as the summary statistic. None plots cost against a domain distance. None normalizes by a persistence reference.

## (1) Recipe defaults for LoRA on diffusion backbones

| Source | Backbone | Rank | Alpha | Targets | LR |
|---|---|---|---|---|---|
| LoRA paper (2106.09685) | GPT-3, RoBERTa | 1 to 8 | first r tried (α = r) | W_q, W_v | task-tuned |
| diffusers `train_text_to_image_lora.py` | SD U-Net | 4 | = rank | to_q, to_k, to_v, to_out.0 | 1e-4 |
| diffusers `train_dreambooth_lora_sd3.py` | SD3 MMDiT | 4 | = rank | attention q, k, v, out plus the joint-stream add_* projections | 1e-4 |
| diffusers `train_dreambooth_lora_flux.py` | Flux DiT | 4 | 4 | attention plus feed-forward | 1e-4 |
| diffusers `train_cogvideox_lora.py` | CogVideoX DiT | 128 | 128 | to_q, to_k, to_v, to_out.0 | 1e-4 |
| cloneofsimo/lora | SD U-Net | 4 (examples 1 to 16) | scale 1.0 | attention plus GEGLU | about 1e-4 |
| **Vista** (2405.17398, App. C.3) | SVD U-Net world model | **16** | not stated | all attention blocks | 5e-5 (5× pretrain) |
| **Cosmos-Predict2 docs** | 2B DiT world model | **16** (8 to 32) | **16** | q, k, v, output_proj, mlp.layer1, mlp.layer2 | 2^-12 to 2^-9 |
| DiffSynth-Studio | Wan 2.1 | 32 | = rank | q, k, v, o, ffn.0, ffn.2 | 1e-4 |
| musubi-tuner | HunyuanVideo | 32 | — | — | 2e-4 |
| GameFactory | open-domain video DiT | 128 | — | — | 1e-4 |
| DreamGen | Wan 2.1 | 4 | 4 | — | 1e-4 |
| Dexterous WM | CogVideoX-Fun 5B | 64 | — | DiT | — |
| AnimateDiff MotionLoRA | motion module | 2 to 128 ablated | — | self-attention | — |

Reading:
- **Rank 16 with alpha 16** is the stated default of the only world-model LoRA recipes found (Vista, Cosmos-Predict2 2B). Cite those two.
- **Alpha equal to rank** (scale 1) is the diffusers and DiffSynth default and matches the LoRA paper's practice.
- **Targets.** Image-era scripts put LoRA on attention only. DiT and video world-model recipes (Cosmos, DiffSynth, Flux) add the MLP. Our attention-only choice follows the SD convention. Adding the MLP is the stronger precedent for a DiT and a cheap ablation.
- **Learning rate.** 1e-4 is the near-universal absolute default. Relative to pretraining, the sources say 5× (Vista) or 10× (DiffFit), not 2×.

## (2) Has anyone adapted a game world model to an unseen level with a small adapter?

Not in the form we propose. The pieces exist separately:
- **AVID** trains a 22M or 71M adapter beside a frozen Procgen video model and adapts it to a held-out game (Coinrun), tested on unseen levels, at three data sizes. It is an output-space adapter, not LoRA. It adapts to a new game, not a new level of a pretraining game, and it reports no step curve.
- **AdaWorld** adapts to unseen environments (Minecraft, DMLab, Habitat) with 50 to 200 samples per action and plots PSNR against steps. It fine-tunes the whole model.
- **GameFactory** and **Micro-World** put LoRA on a frozen video model to absorb a game's style. They then drop or merge it, so the adapter carries the domain shift, but in the opposite direction to ours.
- **Vista** is the closest recipe (frozen world model, LoRA rank 16, new conditioning path). Its target is a new control signal, not a new scene.

No Doom, GameNGen-lineage or FPS paper found adapts to a held-out map with an adapter.

## (3) Verdict: partly

The method (LoRA on a frozen diffusion world model with a few episodes) is **done**: Vista, DreamGen, GameFactory and the Cosmos docs all use it. Adaptation curves against steps and samples for a world model are **partly done**: AdaWorld, AVID and iVideoGPT have them, with full fine-tuning or non-LoRA adapters, across 1 to 4 target domains.

What is **new** in this slice:
1. The unseen target is a level of the pretraining game rather than a new game.
2. Cost is defined as the steps or episodes needed to reach a persistence-normalized gain, per map, across 18 maps.
3. That cost is set against a frozen training-to-map distance.
4. Control retention (the directional check) and forgetting (the training-map gain) are guarded at each point.

Write the related-work claim as "adapter adaptation of world models is established; its cost as a function of domain distance has not been measured." Do not claim the adapter recipe itself.

## Facts most likely to change the design

1. **DiffFit Table 1:** LoRA r8 and r16 on DiT-XL/2 give mean FID 81 against 16.6 for full fine-tuning and 16.8 for BitFit. Keep Decision 4 (train the embeddings in full) and bring Decision 5's full fine-tune reference forward. Without it, a flat curve on far maps cannot be separated from a LoRA ceiling.
2. **Full fine-tuning is the upper bound in every head-to-head found.** AVID Coinrun gives PSNR 25.8 for full FT against 23.8 for the adapter. iVideoGPT prefers full FT over PEFT. SCOPE's frozen backbone loses 85 FVD. A rank ablation (4, 16, 64) plus one full-FT point per pilot map makes the curve interpretable.
3. **Learning rate and targets.** The cited precedents use 5× to 10× the pretraining rate, or an absolute 1e-4, and the DiT world-model recipes put LoRA on the MLP as well as attention. The design's "2× is the convention" line needs a different citation or a change. Our 1e-4 is 2× our 5e-5 pretraining rate, which happens to equal the common absolute default.

## Sources

arXiv PDFs or HTML read: 2106.09685, 2208.12242, 2302.05543, 2302.08453, 2307.04725, 2304.06648, 2408.06072 (meta only), 2501.03575, 2405.17398, 2501.08325, 2503.18938 (page 7 rendered), 2410.12822, 2306.01872, 2310.06114, 2505.12705, 2512.17907 (HTML), 2505.14357, 2502.07825, 2506.17201, 2605.23345, 2408.14837, 2310.16828, 2203.13880, 2405.15223, 2504.02252, 2302.04009 (HTML), 2510.26433, 2506.07280; abstracts of 2405.09673 and 2211.15944.

Other primary pages:
- AVID in RLJ: rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_64.pdf
- CVF open-access pages for ControlNet and DiffFit (ICCV 2023)
- Micro-World: rocm.blogs.amd.com/artificial-intelligence/micro-world
- Raw source files:
  - huggingface/diffusers `examples/{text_to_image,dreambooth,cogvideo}` LoRA scripts
  - cloneofsimo/lora `README.md` and `lora_diffusion/lora.py`
  - nvidia-cosmos/cosmos-predict2 `documentations/post-training_video2world_lora.md` and `cosmos_nemo_assets_lora.py`
  - nvidia-cosmos/cosmos-predict2.5 LoRA config
  - modelscope/DiffSynth-Studio `diffsynth/diffusion/parsers.py` and the Wan2.1-T2V-1.3B LoRA script
  - kohya-ss/musubi-tuner `docs/hunyuan_video.md`
