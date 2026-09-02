# DoomDiT — Agent Context Brief

Machine-oriented project brief. Dense facts only. For the narrative version see `doomdit-full-recap-2026-07.md`. Compiled 2026-07-15; all repo facts verified against code/log on that date.

> Superseded on 2026-09-01 (see `RESEARCH_CONTEXT.md` sections 0, 6, 7, 8): Superman no longer holds any Doom artifact; `eval_checkpoint.py` is fixed; sampling is respaced ancestral DDPM not DDIM; the headline was scored on training data; new modules `doom_data.py`, `doomdit_utils.py`, `encode_episodes.py`, `eval_metrics.py`.

## Identity

- Project: **DoomDiT** — action-conditioned latent diffusion (DiT-XL/2) world model that predicts the next VizDoom frame from 4 past frames + 1 discrete action.
- Repo: `github.com/RohanNaga/Doom`, local `/Users/rohan/Documents/Github/Doom`. Fork of `chuanyangjin/fast-DiT`.
- Authors: Rohan Nagabhirava, Keerthana Chirumamilla. Origin: CMU 18-789 final project (Apr 2026).
- Goal now: NeurIPS 2026 world-models workshop paper (deadline ~2026-08-29), then ICLR 2027 (abstract 2026-09-19, paper 2026-09-24).
- Training host: "Superman" — `rohan@128.2.204.116` (sshpass), 8× RTX A4000 16 GB, tight shared disk. Training used 4 GPUs.

## Model (models.py, trainDoom.py)

- DiT-XL/2: depth 28, hidden 1152, heads 16, patch 2 → 673,869,344 params.
- Latent space: SD-VAE-ft-mse (`stabilityai/sd-vae-ft-mse`), scale 0.18215. Frame 160×120 RGB → latent (4, 15, 20). Height padded 15→16 → `input_size=(16,20)` → 80 tokens (patch 2). Strip pad with `[:, :, :15, :]` before every decode.
- Conditioning:
  - Context: 4 past-frame latents channel-concatenated → 16 ch + 4 noisy target ch = `in_channels=20`. Model predicts 4 ch (`pred_channels=4`; `learn_sigma=True` → 8 out ch).
  - Action: single most-recent action `actions[idx][-1]` ("Design A"; dataset stores 5-action window, 4 unused), `num_classes=18` (VizDoom full action set), fed through repurposed `LabelEmbedder`; `c = t_emb + y_emb` → adaLN-Zero. 10% dropout retained; CFG available (`forward_with_cfg`) but unused in results.
  - Positional embedding: 2D sin-cos generalized to rectangular grids.
- Warm-start from ImageNet DiT-XL/2-256: copy shared weights; inflate `x_embedder.proj.weight` 4→20 ch (first 4 pretrained, rest random init); skip `pos_embed` and `y_embedder`. Result: ~20× loss drop in first 1k steps. `--resume-from` (DoomDiT ckpt) overrides `--ckpt` (warm-start).
- Gradient checkpointing: implemented (`models.py:265`), opt-in, default OFF (80 tokens → pure overhead). NOTE: final presentation claims it was on; code default is off — reconcile before publishing.

## Data

- Source: HF `arnaudstiegler/vizdoom-500-episodes-skipframe-4-lvl5` (500 episodes, skipframe 4).
- Pipeline: frames 320×240 → 160×120 → VAE encode fp16 per episode (`ep_XXXX_latents.npy`, `ep_XXXX_actions.npy`). **This encoding script is Keerthana's, NOT in repo** (`extract_features.py` is unmodified upstream). 
- `build_dataset.py` → sliding window (4→5th) → mmap arrays in `data/full/` (Superman only; gitignored):
  - `context_latents.npy` (N,4,4,15,20) fp16, 23.7 GB
  - `target_latents.npy` (N,4,15,20) fp16, 5.9 GB
  - `context_actions.npy` (N,5) int64 (cols a_i..a_{i+4})
  - N = 2,468,229. Atomic writes + disk preflight.
- In-repo: only `data/debug/*_debug.npy`.

## Training run of record

- Experiment `002-DiT-XL-2` (results/002-DiT-XL-2/, log + 91 sample dirs committed). 2026-04-21→22, ~23 h, 4×A4000, DDP via accelerate, bf16.
- Batch 32 global (8/GPU), fused AdamW lr 1e-4 wd 0, warmup 500 linear, grad clip 1.0. DDPM 1000 linear, train-time eval DDIM 50.
- Resumed from run 004 best.pt (step 4400; run 004/012 never committed). Ran to 90,200 steps @ ~1.62 steps/s. **best.pt = step 87,200, smoothed loss 0.0481.**
- Cumulative across runs: ~104.7k steps, loss 1.14 → 0.051. ~72 GPU-hours total.
- EMA 0.9999: stored bf16, math MUST be fp32 (bf16 `add_` with alpha=1e-4 underflows → EMA freezes; fixed in `9936b36`). Reported samples use LIVE model, not EMA.
- Weights: GitHub release `002-DiT-XL-2-best-90k` → `best.pt` + `0090000.pt` (2.6 GB each, bf16). Fetch: `bash download_weights.sh`. md5s in WEIGHTS.md. Ckpt keys: `{ema, model, args, step, loss}`.

## Results (from final presentation; harness NOT in repo)

Teacher-forced next-frame, 20k-step checkpoints:

| model | PSNR dB ↑ | LPIPS ↓ |
|---|---|---|
| GameNGen (paper, cross-paper ref) | 29.43 | 0.249 |
| U-Net baseline (ours, matched) | 24.60 | 0.198 |
| DiT-XL/2 (ours) | 26.04 | 0.153 |

Inference: ~5 s / 8 frames @ 50 DDIM steps (not real-time).

## File map (repo root)

- `trainDoom.py` — main training (DOOM-specific). `models.py` — DiT with context/action mods.
- `build_dataset.py`, `inspect_dataset.py`, `sanity_check.py` — data tooling.
- `eval_checkpoint.py` — re-sample eval segments @ 250 DDIM. **BUG: reads undefined `ema_state` (should be `state`) → NameError on load. Fix before use.**
- `rollout_video.py` — autoregressive rollout → GIF/PNG (seed 4 real frames, slide window, GT actions).
- `download_weights.sh`, `WEIGHTS.md` — weights fetch.
- Upstream-unmodified (ignore for DOOM work): `train.py`, `extract_features.py`, `sample.py`, `sample_ddp.py`, `train_options/`, `run_DiT.ipynb`, `visuals/`, `README.md` (still fast-DiT's).

## Known gaps / TODO (paper-blocking first)

1. U-Net baseline code+ckpts: NOT in repo, NOT in any session transcript. Check Superman + Keerthana; else reimplement matched-params U-Net.
2. PSNR/LPIPS harness: NOT in repo (verified zero grep hits). Rebuild.
3. No autoregressive metrics exist. Required for 2026 credibility: FVD vs rollout length, drift-vs-horizon curves, IDM action-following accuracy (train small VizDoom IDM). Teacher-forced-only = likely rejection.
4. No val split; best.pt chosen by smoothed TRAIN loss. Single seed.
5. `eval_checkpoint.py` NameError (item above).
6. README rewrite; import VAE-encoding pipeline into repo.
7. Real-time: consider few-step distillation (Self-Forcing / CausVid style) or scope as future work.

## Positioning (decided)

- LEAD: controlled U-Net vs DiT ablation under matched data/params/compute (rare in field; cross-paper comparisons all confounded). 
- SUPPORT: small-compute reproducibility (~72 GPU-h, 16 GB cards, full recipe+weights); ImageNet image-model warm-start ablation (cheap to add).
- DO NOT claim "first DiT world model" (Oasis Nov 2024, Matrix-Game, NFD, SANA-WM). "First DiT WM for DOOM" = footnote only.
- FUTURE: cross-game transfer w/ weight-level analysis (vs GameFactory); robotics grasp-candidate rejection (vs V-JEPA 2-AC, WMPO; sim-first in ManiSkill3/RLBench; CoRL 2026 WS ~Sept–Oct or ICRA 2027 = 2026-09-15).

## Key citations (arXiv IDs)

GameNGen 2408.14837 · DIAMOND 2405.12399 · Diffusion Forcing 2407.01392 · Genie 2402.15391 · MineWorld 2504.08388 (IDM metric) · Matrix-Game 2506.18701 / 2508.13009 (GameWorld Score) · GameFactory 2501.08325 · Hunyuan-GameCraft 2506.17201 / 2511.23429 · Cosmos 2501.03575 · V-JEPA 2 2506.09985 · NFD 2506.01380 (closest cousin: 310M AR DiT 30+ FPS) · Self-Forcing 2506.08009 · CausVid 2412.07772 · WorldMem 2504.12369 · UWM 2504.02792 · VPP 2412.14803 · WMPO 2511.09515 · WM-manipulation survey 2606.00113. Oasis/Genie-2/3/WHAM: blog/Nature, no arXiv. "Lucid-v1": unverified, do not cite.

## Venue clock (from 2026-07-15)

- NeurIPS 2026 workshops: submissions ~2026-08-29 (list announced now; watch Embodied World Models recurrence). Non-archival.
- ICRA 2027 full: 2026-09-15. ICLR 2027: 2026-09-19/24. AAMAS 2027: 2026-10-10. CVPR 2027: ~2026-11-15. CoRL 2026 WS: ~Sept–Oct. ICLR 2027 WS: ~Feb 2027. IEEE ToG: rolling.

## Operational notes for agents

- 16 GB VRAM budget shaped everything: bf16, fused AdamW, CPU VAE, mmap data + low num_workers, per-segment sampling, empty_cache after sampling, bf16 ckpts cast on CPU. Respect these when modifying training.
- EMA math in fp32 always (bf16 underflow bug). Prefer live-model weights for eval/rollout (EMA-freeze history).
- Sample dirs: `results/002-DiT-XL-2/samples/step_XXXXXXX/segment_XX.png` + `ground_truth/`.
- Full recap narrative: `doomdit-full-recap-2026-07.md` (same dir as this file; intended repo home `.claude/analyses/`).
