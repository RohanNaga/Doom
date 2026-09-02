# DoomDiT — Research Guide

**Project**: DiT-XL/2 action-conditioned latent diffusion world model for DOOM (CMU 18-789 final project, Rohan Nagabhirava and Keerthana Chirumamilla), now being turned into a workshop paper.
**Repo**: `github.com/RohanNaga/Doom`, forked from fast-DiT. Rohan is the sole author of commits here.

> **IMPORTANT**: Before any work, read `RESEARCH_CONTEXT.md` in the project root (current state, paper plan, deadlines), then `.claude/analyses/DOOMDIT_AGENT_CONTEXT.md` (dense technical brief) and `.claude/analyses/doomdit-full-recap-2026-07.md` (full recap with literature and venues). `IDEAS.md` holds speculative directions.

## Working style

Discussion first for anything beyond a small fix: state the research question, present more than one approach with tradeoffs, agree on the experiment design, then implement in reviewed chunks. Verify results by running them, not by asserting. Single-line lowercase imperative commit messages, no attribution lines.

## Server

Training and evaluation run on **Superman** (`rohan@128.2.204.116`, 8× RTX A4000 16 GB, shared). Rules: leave at least 2 GPUs free, pick the highest-numbered free GPUs first, keep 250 GB disk free, check `nvidia-smi` before launching. The April project directory `/home/rohan/Doom/Doom/` was deleted before Sep 2026 (disk cleanup); recreate under `/home/rohan/Doom/` by cloning the repo. The `Doom` conda env survives. Disk was at 98 GB free on Sep 1, 2026, below the 250 GB rule, so every byte written there needs a stated budget. Requires CMU VPN. Auth via `sshpass -p "$PERSEVE_SERVER_PASSWORD"`.

**Always invoke the `/run-server` skill before any SSH, SCP, or remote command.** The skill in `.claude/skills/run-server/` was copied from the Lego repo; for this project only the Superman section applies.

## Standing facts that shape code changes

- 16 GB VRAM budget drove everything: bf16, fused AdamW, CPU-resident VAE, mmap dataset with low `num_workers`, per-segment sampling, `empty_cache()` after sampling, weights moved to CPU before the bf16 cast on save.
- EMA math must be fp32 (bf16 `add_` with alpha 1e-4 underflows and freezes the EMA). Reported samples use the live model, not EMA.
- Latents are (4, 15, 20); height is padded to 16 for patch-2 and stripped with `[:, :, :15, :]` before every decode.
- Only the most recent action conditions the model ("Design A"); the stored 5-action window is otherwise unused.
- The U-Net baseline code and checkpoints are **not in this repo** and were wiped from Superman. `encode_episodes.py` (frame-to-latent) and `eval_metrics.py` (PSNR/LPIPS) are the Sep 2026 rebuilds; `doom_data.py` replaces the consolidated arrays with per-episode indexing and holds the episode split.
- Sampling everywhere in this repo is *respaced ancestral DDPM* via `p_sample_loop` (learned sigma), not DDIM, even where comments say DDIM. `ddim_sample_loop` exists and is exposed as `--sampler ddim` in the new harness.
- The 26.04 dB / 0.153 headline was measured on training data (the 10x8 in-training segments); the released checkpoint saw all 500 episodes, so it has no honest held-out number.
- `train.py`, `extract_features.py`, `sample.py`, `sample_ddp.py`, `train_options/`, `run_DiT.ipynb`, `visuals/` are unmodified upstream fast-DiT. `README.md` is still upstream's.

## Experiment tracking

Every run records git hash, full args, seed, environment, loss curve, eval metrics, and sample artifacts under `results/<run>/`. Weights go to a GitHub release with md5s in `WEIGHTS.md`.

## Session hygiene (how this stays current for the whole semester)

- One long-lived working chat per repo. Spin off a separate chat (task chip) for any experiment or build that will run for hours, and have it report back.
- **Before a session ends, or whenever a decision, result, or direction change happens, update `RESEARCH_CONTEXT.md`**: section 0 (where we are going now), the "what changed" log with the date, and open decisions. Commit it. Memory gets a one-line pointer only.
- When a chat gets long or stale, start a new one from `RESEARCH_CONTEXT.md`; do not carry context by re-reading old chats.
- The Friday update to Changliu is generated from the "what changed" entries of that week, tagged by fall goal.

