# DoomDiT — Full Project Recap (July 15, 2026)

> Compiled to bring the project back into working memory before pushing toward a workshop/conference paper.
> Sources: final presentation PDF, repo + full git history, training logs, session-transcript search, and fresh web research (literature + venues), all as of 2026-07-15.

---

## 1. What the project is

**DoomDiT: A Diffusion Transformer World Model for DOOM** — CMU 18-789 Deep Generative Modeling final project (Rohan Nagabhirava & Keerthana Chirumamilla). Repo: `github.com/RohanNaga/Doom`, forked from fast-DiT.

**Thesis:** GameNGen (Google, Aug 2024) proved playable neural game simulation of DOOM with a U-Net (SD 1.4, ~860M params, PSNR 29.43). DiTs have since replaced U-Nets as SOTA for image/video generation. DoomDiT is the first DiT-based world model for DOOM, built to enable a *direct U-Net vs DiT backbone comparison on identical game data* — same VAE, same conditioning scheme, same data; backbone is the only variable.

**Headline results (teacher-forced, next-frame prediction, 20k-step checkpoints):**

| Model | PSNR ↑ | LPIPS ↓ |
|---|---|---|
| GameNGen (paper) | 29.43 dB | 0.249 |
| U-Net baseline (ours) | 24.60 dB | 0.198 |
| **DoomDiT DiT-XL/2 (ours)** | **26.04 dB** | **0.153** |

- DiT beats the matched U-Net by +1.44 dB PSNR and 23% LPIPS — the controlled comparison is the real contribution.
- DiT beats GameNGen by 39% on LPIPS while trailing on PSNR — but that's a *cross-paper* comparison (different data/agent/protocol), only usable with caveats.
- Qualitative: correct DOOM palette, wall/floor/ceiling structure matches GT; fine textures slightly blurry.

---

## 2. Technical setup (exact, from code)

### Architecture (`models.py`, `trainDoom.py`)
- **DiT-XL/2**: depth 28, hidden 1152, heads 16, patch 2 → **673,869,344 params**.
- `input_size=(16, 20)`: DOOM frames at 160×120 → SD-VAE ÷8 → 20 wide × 15 tall latent, H padded 15→16 for patch-2 divisibility → **80 tokens**. Padding stripped (`[:, :, :15, :]`) before every VAE decode.
- **Context via channel-concat**: 4 past frames × 4 latent channels = 16 context channels + 4 noisy-target channels → `in_channels=20`; `pred_channels=4`, `learn_sigma=True` → 8 output channels. Chosen over token-concat to keep token count/cost tiny.
- **Action conditioning**: ImageNet class-label `LabelEmbedder` repurposed for **18 VizDoom actions**; `c = t_emb + action_emb` drives adaLN-Zero in every block. 10% label dropout retained (CFG-capable via `forward_with_cfg`; CFG never used in reported results). "**Design A**": conditions on the **single most-recent action only** (`actions[idx][-1]`) — the stored (N,5) action window is otherwise unused.
- 2D sin-cos positional embedding generalized to rectangular `(gh, gw)` grids.
- **Warm-start** from ImageNet DiT-XL/2-256: `x_embedder.proj.weight` inflated 4→20 channels (first 4 pretrained, 16 context channels random); `pos_embed` (grid mismatch) and `y_embedder` (1000→18) skipped. Effect was dramatic: ~20× loss drop in first 1,000 steps. `--resume-from` (DoomDiT ckpt) takes precedence over `--ckpt` warm-start.

### Data pipeline
- Source: HuggingFace `arnaudstiegler/vizdoom-500-episodes-skipframe-4-lvl5` (500 VizDoom episodes, skipframe-4, level 5).
- Frames 320×240 → 160×120 → **SD-VAE-ft-mse** encode → latents (4, 15, 20), scale 0.18215, fp16.
- ⚠️ The frame→latent encoding script is **Keerthana's pipeline, NOT in this repo** (`extract_features.py` is unmodified upstream ImageNet code). Repo consumes per-episode `ep_XXXX_latents.npy` / `ep_XXXX_actions.npy`.
- `build_dataset.py`: sliding window (4 context → predict 5th) → consolidated mmap arrays:
  - `context_latents.npy` (N, 4, 4, 15, 20) fp16 — 23.7 GB
  - `target_latents.npy` (N, 4, 15, 20) fp16 — 5.9 GB
  - `context_actions.npy` (N, 5) int64 — 99 MB, columns `[a_i..a_{i+4}]`
  - **N = 2,468,229 samples**; atomic writes + disk-space preflight (Superman disk is tight/shared).
- Full latent arrays live only on Superman (`data/`, `results/` gitignored; only `data/debug/` committed).

### Training recipe (`trainDoom.py`)
- 4× RTX A4000 (16 GB) on Superman, DDP via HF `accelerate`, **bf16**, ~1.62 steps/sec.
- Global batch 32 (8/GPU), fused AdamW lr=1e-4 (avoids ~2.7 GB unfused sqrt buffer), wd 0, linear warmup 500 steps, grad clip 1.0.
- Diffusion: 1000-step DDPM (linear), `learn_sigma=True`; in-training sampling at 50-step respaced DDIM.
- **EMA 0.9999 stored bf16, updated in fp32** — a real bug was found & fixed: bf16 `add_(alpha=1e-4)` underflows (7-bit mantissa), silently freezing EMA at init (`9936b36`). Policy since: always save the live model too; reported samples use the live model, not EMA.
- Memory stack for 673M on 16 GB: fused AdamW, bf16 EMA, CPU-resident VAE (~335 MB saved), mmap dataset + low num_workers (16× RSS fix), `empty_cache()` after sampling, per-segment sampling under bf16 autocast, weights→CPU before bf16 cast on save, EMA-only rotating checkpoints (~1.35 GB, `--keep-last 3`) + `best.pt` by smoothed train loss. Gradient checkpointing available but **off** (80 tokens = pure overhead; `models.py:185`). *(Presentation says grad-ckpt on every block — reconcile before the paper.)*
- ~72 GPU-hours, 18h wall-clock.

### Training trajectory
- **104,700 cumulative steps**; loss 1.14 → **0.051** (~20×). Milestones: 1k → 0.13, 5k → 0.07, 50k → 0.055, 104k → 0.051; still descending ~0.005/10k.
- Archived best run `results/002-DiT-XL-2/` (2026-04-21→22, ~23h): resumed from run 004's best.pt @ step 4400; **best.pt = step 87,200, loss 0.0481** (verified in log). Prior runs 004/012 existed on Superman, never committed.
- In-training eval: 10 evenly-spaced segments × 8 consecutive frames, PNG per segment per 1000 steps → 91 sample dirs (incl. ground_truth) committed.

### Evaluation & inference code
- `eval_checkpoint.py` — resamples the fixed eval set at 250 DDIM steps with the correct ft-mse VAE. ⚠️ **Broken as committed** (verified): assigns `state` but then reads undefined `ema_state` → NameError on load.
- `rollout_video.py` — **autoregressive rollout**: seed 4 real frames, DDIM-sample next latent, slide window, repeat; ground-truth action sequence; outputs GIF + PNG grid. **No quantitative autoregressive metrics ever computed.**
- `sanity_check.py` — pre-flight shape/loss check.
- Inference: ~5 s per 8 frames at 50 steps — far from real-time (GameNGen: 20 FPS at 4 steps).

### Weights & artifacts
- GitHub release **`002-DiT-XL-2-best-90k`**: `best.pt` (2.6 GB, step 87,200) + `0090000.pt` (2.6 GB); `bash download_weights.sh` reassembles 1.3 GB chunks; md5s in `WEIGHTS.md`. Checkpoint format `{ema (bf16), model (bf16), args, step, loss}`.

---

## 3. Repo provenance & timeline

- Forked from **fast-DiT**; `train.py`, `extract_features.py`, `sample.py`, `sample_ddp.py`, `train_options/*`, `run_DiT.ipynb`, `visuals/` are unmodified upstream ImageNet code. **README.md is still the upstream fast-DiT README.**
- 33 commits: Init → fast-DiT → `doom training` adaptation (Apr 20) → **four "out of memory" commits** (16 GB fight; resolved by mmap/num_workers/atomic-write refactors) → EMA-underflow fix + checkpoint/memory polish (Apr 21) → final run 002 (Apr 21–22) → rollout script → archive + weights release (Jul 15).

## 4. Gaps between current state and a publishable paper

1. **U-Net baseline code is NOT in this repo and NOT in any session transcript** — likely on Superman or in Keerthana's code. Must be recovered or reimplemented.
2. **PSNR/LPIPS metric code is NOT in the repo** (verified: zero grep hits in any branch). The scripts producing the headline numbers must be recovered/rebuilt.
3. **All reported metrics are teacher-forced single-step** — no autoregressive FVD, no drift-vs-horizon, no action-following accuracy. This is the #1 reviewer rejection risk in 2026 (see §6.3).
4. GameNGen comparison is cross-paper, not controlled — caveat carefully.
5. Single run, no seeds/variance; "best" selected by smoothed *training* loss (no val split).
6. Only 1 of 5 stored actions used; 18-class action semantics undocumented in-repo.
7. `eval_checkpoint.py` NameError bug (`ema_state`).
8. Presentation/code mismatch on gradient checkpointing.
9. README still upstream; frame-encoding pipeline lives outside the repo.
10. 50-step inference; distillation untouched — real-time is now table stakes in the field.

---

## 5. Work history (session-transcript search)

**Key finding: the CCD session-transcript store contains no DoomDiT work sessions** (searched 100 sessions incl. archived, every project term). The sessions that built this project predate/were never captured by the transcript store; only a second-hand mention exists in a *Lego_Manipulation* session (references to "the DOOM DiT world model … end-of-sem meeting" and a stray tmux session for "the Doom video gen project"). History above (§2–3) was reconstructed from the unusually well-commented repo itself.

**Not recoverable from this machine:** U-Net baseline training code/checkpoints, the PSNR/LPIPS eval harness, autoregressive-drift discussion, end-of-sem meeting notes. **To recover: check Superman (`rohan@128.2.204.116`) filesystem and Keerthana's code/notebooks.**

---

## 6. Related work & state of the art (through July 2026)

### 6.1 Direct prior work (must-cite)

| Work | Date/Venue | What it is | Numbers relevant to us |
|---|---|---|---|
| **GameNGen** (Valevski et al., [2408.14837](https://arxiv.org/abs/2408.14837)) | ICLR 2025 | SD1.4 U-Net (~860M), 64-frame history, **noise-augmented context** vs drift | PSNR 29.43, LPIPS 0.249, FVD 114 (16f)/186 (32f), 20 FPS @ 4 DDIM steps |
| **DIAMOND** (Alonso et al., [2405.12399](https://arxiv.org/abs/2405.12399)) | NeurIPS 2024 | Pixel-space EDM U-Net WM for RL; also CS:GO | Atari-100k HNS 1.46 |
| **Diffusion Forcing** (Chen et al., [2407.01392](https://arxiv.org/abs/2407.01392)) | NeurIPS 2024 | Per-token noise levels; stable beyond-horizon rollouts | The mechanism DoomDiT did *not* use — a design axis to discuss |
| **Oasis** (Decart/Etched, Oct–Nov 2024, open `open-oasis` 500M) | blog | ViT VAE + DiT + Diffusion Forcing, Minecraft | ~20 FPS 360p on H100; famous drift on turn-around — kills any generic "first DiT WM" claim |
| **Genie 1/2/3** (DeepMind) | ICML 2024 / Dec 2024 / **Aug 2025** | Latent actions → playable 3D → 720p 24 FPS real-time, ~1 min consistency | Closed, no papers for 2/3; the frontier bar we're not competing with |
| **WHAM/Muse** (Microsoft, Nature Feb 2025) | Nature | 1.6B discrete-token AR (VQGAN+transformer), Bleeding Edge, >1B image-action pairs | The discrete-token alternative |
| **MineWorld** (Microsoft, [2504.08388](https://arxiv.org/abs/2504.08388)) | Apr 2025 | Open real-time AR transformer Minecraft WM | 4–7 FPS; **introduced IDM action-accuracy metric** — adopt it |
| **Matrix-Game 1/2/3** (Skywork, [2506.18701](https://arxiv.org/abs/2506.18701), [2508.13009](https://arxiv.org/abs/2508.13009), 3.0 in 2026) | 2025–26 | 17B DiT + **GameWorld Score benchmark**; 2.0: 1.3B, 6-frame KV-cache, 25 FPS 720p; 3.0: 5B, 40 FPS, minute-long consistency | kb/mouse acc ≥0.95 (vs Oasis 0.86/0.56); current open real-time SOTA lineage |
| **GameFactory** (Kwai, [2501.08325](https://arxiv.org/abs/2501.08325)) | ICCV 2025 | Decouples game-style from action control → scene-generalizable control | **The must-cite for any cross-game/transfer framing** |
| **Hunyuan-GameCraft 1/2** (Tencent, [2506.17201](https://arxiv.org/abs/2506.17201), [2511.23429](https://arxiv.org/abs/2511.23429)) | 2025 | kb+mouse unified as camera space; 1M+ recordings, 100+ AAA games; v2 adds language control | Distilled real-time |
| **Cosmos** (NVIDIA, [2501.03575](https://arxiv.org/abs/2501.03575)) | Jan 2025 | World-foundation-model platform (diffusion + AR variants), ~20M h | Physical-AI reference platform |
| **V-JEPA 2 / 2-AC** (Meta, [2506.09985](https://arxiv.org/abs/2506.09985)) | Jun 2025 | Non-generative latent JEPA WM; AC variant plans zero-shot on a Franka | The "predict-in-latent, don't render" camp |

(Note: "Lucid-v1" could not be verified as a real citation — confirm before citing.)

### 6.2 What's new 2025 → mid-2026

- **Real-time via distillation is the biggest shift**: CausVid ([2412.07772](https://arxiv.org/abs/2412.07772)), **Self-Forcing** ([2506.08009](https://arxiv.org/abs/2506.08009), AR self-rollout with KV-cache during training, ~16 FPS 480p H100), Causal Forcing/++ (ICML 2026, [2602.02214](https://arxiv.org/abs/2602.02214), [2605.15141](https://arxiv.org/abs/2605.15141)), **Next-Frame Diffusion** ([2506.01380](https://arxiv.org/abs/2506.01380) — a **310M AR DiT, 30+ FPS on A100; closest architectural cousin to DoomDiT, must cite & differentiate**). 50-step DDPM at 5s/8 frames is off the current map.
- **Long-horizon memory/drift**: KV-cache-as-memory (WorldMem [2504.12369](https://arxiv.org/abs/2504.12369), RELIC [2512.04040](https://arxiv.org/abs/2512.04040), WorldPack [2512.02473](https://arxiv.org/abs/2512.02473)), state-space WMs ([2505.20171](https://arxiv.org/abs/2505.20171)). Our fixed 4-frame channel-concat context is a known weak point — frame honestly.
- **Multi-game/cross-domain transfer** is active and credible (GameFactory, Cosmos, GameCraft) — but **weight-level analysis of what transfers is largely unexplored** (our opening).
- **Metrics matured**: IDM action accuracy, GameWorld Score (8 dims), WorldSimBench, MIND ([2602.08025](https://arxiv.org/abs/2602.08025)).
- **Small-compute niche now exists**: DreamForge-World 0.1 ([2606.30292](https://arxiv.org/abs/2606.30292), 480p 14–15 FPS on one 4090), NVIDIA SANA-WM (2.6B DiT, one GPU, May 2026). This is our sub-community.
- **DOOM specifically is under-served post-GameNGen** — momentum went to Minecraft/AAA. Genuine opening.

### 6.3 Evaluation bar for a credible 2026 paper

Mandatory: **autoregressive FVD at multiple rollout lengths**, **drift/quality-vs-horizon curves**, **IDM-based action-following accuracy** (train a lightweight VizDoom IDM), compute/throughput reporting (FPS, steps, GPU). Strongly recommended: small human study (real-vs-generated forced choice, GameNGen-style). Teacher-forced PSNR/LPIPS alone reads as 2023-era — the most likely rejection point.

### 6.4 Positioning verdict

- **"First DiT world model" (unqualified): dead** — Oasis, Matrix-Game, NFD, SANA-WM etc. DiT is now the default backbone.
- **"First DiT world model for DOOM": technically true, weak as a headline** — usable as a supporting one-liner.
- **"Controlled U-Net vs DiT ablation under matched data/params/compute": genuinely rare and valuable — LEAD WITH THIS.** Cross-paper backbone comparisons are confounded everywhere; a matched-recipe head-to-head is the result the field lacks.

**Recommended paper package:**
1. **Framing 1 (spine):** controlled backbone ablation under matched compute — differentiate vs GameNGen (uncontrolled) and NFD (no U-Net control).
2. **Framing 2 (spine):** small-compute reproducibility — ~72 GPU-hours on 4×16 GB, full recipe + weights; differentiate vs DreamForge/SANA-WM/MineWorld on lowest budget + DOOM.
3. **Framing 3 (supporting ablation):** ImageNet **image**-model warm-start (cheaper, understudied vs video pretraining) — from-scratch vs warm-start ablation is cheap to add.
4. **Framing 4 (future/preliminary → conference extension):** cross-game transfer with weight-level analysis (which layers move: adaLN action pathway vs spatial blocks) — differentiate vs GameFactory on mechanistic analysis at compact compute.

---

## 7. Publication venues & deadlines (as of 2026-07-15)

### Workshops (Tier 1 — recommended)
- **NeurIPS 2026 workshops — best near-term target.** Workshop list being announced now (acceptances went out Jul 11, 2026); contribution deadline **~Aug 29, 2026**; workshops Dec 11–12 (Sydney + satellites); non-archival, 4–9 pages. Watch for the *Embodied World Models for Decision Making* recurrence or any world-models/neural-game-engines workshop. **~6 weeks away.**
- **CoRL 2026 workshops** (Austin, Nov 9–12; CFPs Aug–Sept, deadlines ~Sept–Oct) — natural home for the robotics pivot; non-archival.
- **ICLR 2027 workshops** (~Feb 2027 deadlines) — the recurring *World Models: Understanding, Modelling and Scaling* workshop (2nd edition at ICLR 2026, >1,500 attendees) is likely to have a 3rd edition; good fallback.
- **AAAI-27 workshops** (Montréal, Feb 16–23, 2027; deadlines ~Nov 2026) — moderate fit (game AI / RL).
- Closed this cycle: CVPR 2026 workshops, IEEE CoG 2026, AIIDE 2026 (Jun 19 deadline passed), MILA World Modeling workshop (Feb 2026).

### Full papers (Tier 2/3)
| Venue | Deadline | Fit |
|---|---|---|
| **ICLR 2027** | **abstract Sept 19 / paper Sept 24, 2026** (notif. Jan 22, 2027; conf. Apr 24–28, Brazil) | **Best fit** for the controlled backbone study |
| AAAI-27 | Jul 21/28, 2026 (~2 weeks!) | Only if a draft already existed — skip |
| AAMAS 2027 | Oct 10, 2026 (Hanoi, May 3–7, 2027) | Reframe as agent-environment/simulator |
| CVPR 2027 | ~Nov 15, 2026 (predicted; Seattle, Jun 2027) | Vision/perceptual framing |
| ICRA 2027 | **Sept 15, 2026** (conf. May 24–28, 2027) | For the robotics version |
| IEEE Trans. on Games | rolling (double-anonymous; ~4–8 mo review) | Durable archival home for the extended study |

### Recommended tiered plan
1. **Now → ~Aug 29:** NeurIPS 2026 world-models workshop paper (4–8 pp, non-archival): controlled DiT-vs-U-Net + autoregressive FVD/drift + IDM action accuracy. Requires: rebuild metric harness, recover/retrain U-Net baseline, run rollout evals.
2. **Sept 24:** if the extra experiments land (scaling across DiT sizes, warm-start ablation, generalization axis), harden into **ICLR 2027**; else AAMAS (Oct 10) / CVPR (Nov).
3. **Anytime:** IEEE ToG for the definitive extended version.

---

## 8. Robotics extension: world models for pick-trajectory planning/rejection

### Landscape (2025–26)
- **V-JEPA 2-AC (Meta)** — closest analogue and **#1 to differentiate**: 300M action-conditioned latent predictor fine-tuned on 62 h unlabeled Droid data; latent MPC/CEM planning; zero-shot Franka reach ~100%, pick-and-place 60–80%, ~16 s/action.
- **UniSim** (ICLR 2024 outstanding) — action-in/video-out universal simulator; conceptual parent.
- **NVIDIA Cosmos + GR00T-Dreams/DreamGen** — world model for *offline synthetic data generation* (IDM extracts "neural trajectories"), NOT online candidate rejection — a contrast, not a competitor.
- **UWM** (TRI, [2504.02792](https://arxiv.org/abs/2504.02792)) — one diffusion transformer as policy/forward/inverse/video model (+20% success).
- **VPP** (ICML 2025, [2412.14803](https://arxiv.org/abs/2412.14803)) — video-prediction representations for control (+18.6% CALVIN, +31.6% real dexterous).
- **World-model-as-critic line** (WMPO [2511.09515](https://arxiv.org/abs/2511.09515), dWorldEval [2604.22152](https://arxiv.org/abs/2604.22152), survey [2606.00113](https://arxiv.org/abs/2606.00113)) — fastest-moving competitor; **grasping-specific rejection is not yet crowded**.

### Translation of DoomDiT
The primitive matches: action-conditioned latent DiT predicting next frames from history + action. Swap (DOOM frames, discrete action) → (RGB-D/wrist-cam history, candidate end-effector waypoint/gripper action). For each candidate pick from a grasp sampler (e.g. Contact-GraspNet), roll forward in latent space, score with a learned success/feasibility head (or goal-latent distance), **reject bad candidates**, execute the best. `rollout_video.py` logic ports almost directly.

### Minimal publishable version
- **Simulation-first**: ManiSkill3 or RLBench pick-and-place; a few hundred–thousand trajectories *including deliberate failures* (failures teach rejection); ground-truth success labels for the head + privileged eval. No physical arm needed for the workshop version. (Stretch: Franka/UR5 + RealSense, ~50–100 h of picks.)
- **Claim shape:** "Action-conditioned latent DiT world model rejects infeasible grasp candidates at planning time, improving pick success X% over the sampler and no-WM baseline at Y ms/candidate."
- **Venues:** CoRL 2026 workshop (~Sept–Oct) best first target; **ICRA 2027 (Sept 15, 2026)** if sim results are solid by early Sept; AAMAS 2027 (Oct 10) as planning/agents framing.
- **Differentiation:** vs V-JEPA 2-AC — generative/inspectable rollouts (interpretable failure diagnosis) + explicit candidate-rejection critic framing + grasp-failure taxonomy; vs GR00T-Dreams — online verification, not offline data gen; vs WMPO/dWorldEval — grasp feasibility task + DiT backbone + latency budget.

---

## 9. Candidate directions (user's stated interests — recap only, not acted on)

- Turn DoomDiT into a **workshop paper** this summer (stated to professor).
- **Multi-game training** (DOOM ↔ Minecraft, order effects) with a game flag; probe weights for shared "physics/input" circuitry; cross-domain transfer and faster learning. (Nearest prior: GameFactory — differentiate via weight-level analysis; see §6.4 Framing 4.)
- **Robotic pick trajectory planning/rejection** using the same recipe (§8).

## 10. Immediate action items implied by this recap

1. **Recover or rebuild the U-Net baseline + PSNR/LPIPS harness** (check Superman + Keerthana's code first; else reimplement — needed for any submission).
2. **Build the autoregressive eval suite**: FVD vs rollout length, drift curves, VizDoom IDM for action-following accuracy. (`rollout_video.py` is the starting point.)
3. Fix `eval_checkpoint.py` NameError; rewrite README as a DoomDiT README; commit/import the VAE-encoding pipeline.
4. Check the just-announced **NeurIPS 2026 workshop list** for the world-models workshop and its exact deadline (~Aug 29).
5. Decide the paper package (recommended: Framings 1+2 spine, 3 as ablation, 4 as future work) and scope the ~6-week experiment plan.
