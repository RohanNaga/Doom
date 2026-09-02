# DoomDiT — Research Context

> Updated 2026-09-01 (evening, after the kickoff session). Entry point for every new chat in this repo.
> Read order: this file → `.claude/analyses/DOOMDIT_AGENT_CONTEXT.md` (dense technical brief, verified against code July 15) → `.claude/analyses/doomdit-full-recap-2026-07.md` (full recap: literature through July 2026, venues, robotics extension) → `IDEAS.md` (scenario-latent skill files).

## 0. Where we are going now (as of Sep 1, 2026)

**Target: CoRL 2026 PhysWM workshop, Sep 30 AoE, 4 pages** (working assumption from Rohan's Sep 1 lean; confirm at the Sep 2 1-on-1). The Sep 2 and Sep 5 NeurIPS workshops are out: the Superman project directory is gone and nothing can be re-measured in four days. ICLR 2027 (Sep 24) needs the full experiment set and is not reachable either. Section 0c is the experiment list written backwards from Sep 30.

**The paper** is the controlled U-Net vs DiT world-model comparison under matched data and compute, plus the small-compute recipe (section 3). Two facts found on Sep 1 change what "the final run" means:

1. The released checkpoint trained on all 500 episodes, and the 26.04 dB / 0.153 headline was scored on the training-time eval segments, which are training data. There is no honest held-out number for it. **Both backbones must be retrained on a 450-episode split**; the final DiT run is not optional.
2. Superman holds nothing from April except the `Doom` conda env and logs. Latents, runs 004/012, and any U-Net artifacts were deleted (disk cleanup; home is now LEGO and perseve work). Weights are safe on the GitHub release. The source dataset is 48 GB of JPEG parquet on Hugging Face; `encode_episodes.py` (new) rebuilds per-episode latents from it in a few GPU-hours.

**Sep 1 state of the tooling:** `eval_checkpoint.py` fixed; `doom_data.py` (per-episode dataset, episode split, legacy-segment mapping) verified equal to the old consolidated layout; `eval_metrics.py` (PSNR, LPIPS, copy-last baseline, VAE ceiling, respaced-DDPM or DDIM) runs end to end on CPU with a random DiT-S/2; `encode_episodes.py` unit-tested. Nothing has run on Superman yet: three decisions in section 6 gate that.

**Why it matters beyond the paper:** the same action-conditioned latent DiT is the starting point for fall goal 3, a world model for LEGO trajectory decisions, and Changliu's May 4 suggestion of action rollouts for long-horizon planning plus sim-switching at contact-rich points using the perseve Isaac Sim pipeline. `doom_data.py` and `doomdit_utils.py` are written to be reused there.

**Standing rhythm:** weekly 1-on-1 with Changliu every Wednesday 1:00 to 1:30 PM from Sep 2 to Dec 9; Friday Slack update tagged by goal.

## 0b. How we got here (condensed)

DoomDiT is an action-conditioned latent diffusion world model: DiT-XL/2 (673M params) predicts the next VizDoom frame from 4 past frame latents plus one discrete action, trained on Superman (4× A4000 16 GB, about 72 GPU-hours) as the CMU 18-789 final project with Keerthana Chirumamilla. Best checkpoint is step 87,200, loss 0.0481, released as GitHub tag `002-DiT-XL-2-best-90k`. Headline teacher-forced numbers: DiT 26.04 dB PSNR / 0.153 LPIPS vs a matched U-Net baseline 24.60 / 0.198 (GameNGen reports 29.43 / 0.249, but that is a cross-paper comparison). The defensible contribution is the **controlled U-Net vs DiT comparison under matched data and compute plus the small-compute recipe**, not "first DiT world model" (Oasis, Matrix-Game, NFD, SANA-WM already exist). Nothing has been trained or evaluated since April 22. The fall goal is a workshop paper by mid October.

## 0c. Experiment list, backwards from Sep 30 (CoRL PhysWM, 4 pages)

Compute anchor: DiT-XL/2 ran at 1.62 steps/s at global batch 32 on 4 A4000s, so 90k steps is about 15.5 h; the U-Net is assumed comparable. Six GPUs are usable at most (GPUs 0 and 3 are occupied by others on Sep 1), so DiT on four and U-Net on two, or the two runs back to back.

| Dates | Must be true by the end | Runs on Superman |
|---|---|---|
| Sep 30 (Wed) | Submitted on OpenReview. | none |
| Sep 27 to 29 | Figures final (drift curves, rollout strip, table), related work, limitations (teacher-forced-only history, fixed 4-frame context, no real time). | none |
| Sep 24 to 26 | 4-page draft written; **numbers frozen Sep 26**. | re-runs only |
| Sep 20 to 23 | Full eval on DiT-final, U-Net-final, DiT-from-scratch: val PSNR/LPIPS (`eval_metrics.py`), drift curves and FVD16/32 (`rollout_eval.py`), IDM action accuracy. Human-study is out of scope. | eval, 1 to 2 GPUs |
| Sep 12 to 19 | **Final runs launched by Sep 15 (DiT) and Sep 16 (U-Net)**: 450-episode split, identical recipe, at least 90k steps each. Warm-start ablation: DiT-XL/2 from scratch for 20k steps vs warm-started for 20k steps (about 3.5 h each). | 4 + 2 GPUs, 3 days |
| Sep 5 to 11 | U-Net baseline implemented and smoke-tested (design decision in section 6); IDM trained on real pairs (`train_idm.py`); `rollout_eval.py` written per the design doc; harness reproduces the April headline on the legacy segments with `best.pt` (accept a small delta from re-encoded latents); split committed as `data/split.json`. | encode 3 to 5 h, IDM 20 min, harness runs |
| Sep 2 to 4 | Decisions in section 6 taken; repo cloned to Superman; `lpips` installed in the `Doom` env; `best.pt` fetched from the release; latents re-encoded from Hugging Face (or the Drive zip if Keerthana re-shares it); Keerthana message sent. | encode job |

Fallback if the final runs slip past Sep 19: report the released checkpoint on the legacy segments with the training-data caveat stated in the text, keep the U-Net row from April as "reported in the class project, not reproduced", and lead with the autoregressive metrics on the DiT. That is a weaker paper; the point of launching by Sep 15 is not to need it.

Disk budget on Superman (98 GB free on Sep 1, other users hold the rest): latents 5.9 GB, raw frames for ten held-out episodes about 2.8 GB, released weights 5.2 GB, parquet shards streamed one at a time (0.5 GB), two final runs with `--keep-last 3` plus `best.pt` about 21 GB, IDM and eval outputs under 2 GB. **About 37 GB total**, leaving roughly 60 GB free. The 250 GB rule cannot be met by us because the shortfall is other people's data; this needs Rohan's explicit go-ahead.

## 1. Where work happens

| What | Where |
|---|---|
| Repo | `~/Documents/Github/Doom` on `main` (`github.com/RohanNaga/Doom`). Open chats here. Two worktrees under `.claude/worktrees/` are leftovers from July; `IDEAS.md` has been copied to the main checkout. |
| Server | Superman `rohan@128.2.204.116`. **The April project directory is gone** (verified Sep 1: no `Doom/` under home, no latents, no runs 004/012, no U-Net artifacts); the `Doom` conda env (torch 2.11, diffusers 0.37, no `lpips`) and `~/logs/doom-*.log` survive. Disk 98 GB free of 3.5 TB. GPUs 0 and 3 in use by others. Leave 2 GPUs free. Use the `/run-server` skill. |
| Weights | GitHub release `002-DiT-XL-2-best-90k` (`best.pt`, `0090000.pt`, 2.6 GB each, bf16). `bash download_weights.sh`. |
| Collaborator | Keerthana Chirumamilla holds the frame-to-latent encoding script, the U-Net baseline, the PSNR/LPIPS harness, and the Drive copy of `full_latents.zip` (5.43 GB; the link now needs a Google login). Draft ask: `.claude/analyses/keerthana-ask-2026-09-01.md`. |

## 2. What is missing for a paper (in priority order)

1. **U-Net baseline code and checkpoints**: not in the repo, not in any transcript. Recover from Superman or Keerthana, or re-train a matched-params U-Net from the documented recipe.
2. **PSNR/LPIPS harness**: rebuilt Sep 1 as `eval_metrics.py`; still has to reproduce the April numbers on real latents.
3. **Autoregressive metrics**: none exist. Design in `.claude/analyses/autoregressive-eval-design.md` (drift curves, FVD16/32, IDM accuracy; one rollout engine feeding all three). Teacher-forced-only evaluation is the most likely rejection reason.
4. **No validation split, single seed**; `best.pt` was picked by smoothed train loss and trained on every episode. Split policy now lives in `doom_data.make_split` (episode-level, seeded, 10% held out); the released checkpoint cannot be scored on it honestly.
5. README is still upstream fast-DiT; presentation claims gradient checkpointing was on but the code default is off; every comment says "DDIM" but sampling is respaced ancestral DDPM (`p_sample_loop`).
6. Inference is 50-step DDIM at about 5 s per 8 frames. Real time is out of scope for the workshop paper; state it as future work (Self-Forcing / CausVid style distillation).

## 3. Paper plan (decided July 15, reaffirmed Aug 21)

- **Spine 1**: controlled backbone ablation, U-Net vs DiT, same VAE, same conditioning, same data, same compute.
- **Spine 2**: small-compute reproducibility, about 72 GPU-hours on 16 GB cards, full recipe and weights released.
- **Supporting ablation**: ImageNet image-model warm start vs from scratch (cheap; warm start gave a 20× loss drop in the first 1,000 steps).
- **Future work**: cross-game transfer with weight-level analysis (which layers move: adaLN action pathway vs spatial blocks), differentiated from GameFactory; scenario-latent skill files (`IDEAS.md`); robotics grasp-candidate rejection (recap section 8).
- Must-cite and differentiate: GameNGen, DIAMOND, Diffusion Forcing, Oasis, Next-Frame Diffusion (310M AR DiT, closest cousin), MineWorld (IDM metric), Matrix-Game (GameWorld Score), GameFactory. arXiv IDs in the agent context brief.

## 4. Fall 2026 goal and timeline (submitted Aug 21)

Goal 1 of 4: *DoomDiT world model to a final result, workshop paper by mid October. Evaluate the current checkpoint vs the U-Net / GameNGen baseline, plan and run the final larger model and dataset run, write up as a controlled U-Net vs DiT comparison at small compute. Stretch: one world model learning multiple scenarios or games with a game flag.*

| Block | Doom items |
|---|---|
| Aug 24 to Sep 11 | Evaluate current checkpoint vs baseline, find gaps, size the final run. |
| Sep 14 to Oct 2 | Launch the final run, start the workshop paper. |
| Oct 5 to 16 | Workshop paper submitted. |
| Oct 19 to 30 | Stretch: multi-game conditioning experiment. |

Rohan's Sep 1 framing: get the workshop paper together in the next three to four weeks. Risk stated in the plan: if the final run needs more compute or time than Superman allows, fall back to the current checkpoint plus the new metrics.

## 5. Venues (checked 2026-09-01)

The originally targeted general NeurIPS 2026 workshop date (Aug 29) has passed. Live options:

| Venue | Deadline | Fit | Notes |
|---|---|---|---|
| NeurIPS 2026 WS "Robot Learning with World Models" | **Sep 2, 2026 AoE** | high | 2 to 4 page short papers accepted; non-archival; notification Sep 25. Only reachable with a short paper on existing results. |
| NeurIPS 2026 WS "World Models in Physical AI" | **Sep 5, 2026 AoE** (extended) | high | up to 8 pages, non-archival, OpenReview, notification Sep 29. Topics include generative simulation and evaluation. Reachable with the current checkpoint plus a rebuilt metric harness if U-Net numbers can be reproduced in 4 days. |
| NeurIPS 2026 WS "Foundation Models for Temporal Systems" | Sep 15, 2026 AoE | medium | forecasting / simulation framing. |
| ICRA 2027 | Sep 15, 2026 | low for Doom alone | only with the robotics extension. |
| ICLR 2027 main | abstract Sep 19, paper Sep 24, 2026 | medium | needs the full experiment set (autoregressive metrics, baseline, ablations). |
| CoRL 2026 WS "Bringing Physics Simulation and World Models Together for Robotic Manipulation" | **Sep 30, 2026 AoE** | high for Doom and for the Lego world-model direction | up to 4 pages, non-archival, OpenReview (`PhysWM`), Nov 12 in Austin. Topics include world models for planning and decision making, learning dynamics from video, sim-to-real. Closest match to the mid-October plan. |
| CoRL 2026 WS "Do Robots Need World Models?" | not checked | medium | debate format; site do-robots-need-world-models.github.io. |
| AAMAS 2027 | Oct 10, 2026 | low | agent / simulator framing. |
| IEEE Transactions on Games | rolling | archival home for the extended version | |

The mid-October target in the semester plan does not match any of the NeurIPS workshop dates, which all close by Sep 15. Decision needed: sprint for Sep 5 with existing results, or aim at the CoRL PhysWM workshop (Sep 30, 4 pages, the one that fits the three-to-four-week plan), or ICLR 2027 (Sep 24) with the full experiment set.

## 6. Open decisions (owner: Rohan)

1. **Deadline**: Sep 30 CoRL PhysWM (assumed) vs ICLR 2027 Sep 24 vs slipping to a later workshop. Decides everything in section 0c.
2. **Disk go-ahead**: about 37 GB on a Superman disk that is already at 98 GB free. The 250 GB rule is unmeetable by us; proceed or find another machine (Spiderman has 3 TB free but 4 A6000s shared with perseve).
3. **Data path**: re-encode from Hugging Face with `encode_episodes.py` (48 GB streamed, 3 to 5 GPU-hours, exact match to Keerthana's latents not guaranteed) vs waiting for the Drive zip. Recommendation: start the re-encode now, swap in the zip if it arrives, because the final runs need latents either way and the re-encode makes the repo self-contained.
4. **Matched U-Net design**: match parameter count (about 670M, a 2D U-Net with the same 20-in / 8-out channel contract and adaLN-style action+time conditioning) or match training FLOPs; warm-start from SD 1.4 or not. Recommendation: parameter-matched, both backbones from scratch for the main row, ImageNet warm start as a DiT-only supporting row. The April comparison's warm-start status is unknown until Keerthana answers.
5. **Action offset**: training used the action stored on the target frame's row (`ACTION_OFFSET = 4` in `doom_data.py`). Whether that is the causally right action depends on the dataset's logging convention. Keep it for reproduction; decide before the final runs.
6. Whether the multi-game stretch belongs in this paper at all (recommendation: future work).

## 7. What changed (log; newest first)

- **2026-09-01** Kickoff session. Superman inventory: April project directory deleted, only conda env and logs remain; disk 98 GB free. Google Drive latent links now require login. Fixed `eval_checkpoint.py` (`ema_state`) and moved checkpoint/VAE loading into `doomdit_utils.py`. Added `doom_data.py` (per-episode dataset, episode split, legacy-segment mapping; verified equal to `build_dataset.py` layout), `encode_episodes.py` (Hugging Face parquet to per-episode latents, restartable, unit-tested), `eval_metrics.py` (PSNR, LPIPS, latent MSE, copy-last baseline, VAE ceiling; CPU end-to-end test passed). Wrote the Keerthana ask and the autoregressive eval design. Findings that change the plan: headline was scored on training data and the checkpoint saw all episodes, so both backbones must be retrained on the split; sampling is respaced DDPM not DDIM. Deadline working assumption: CoRL PhysWM Sep 30. Tagged: goal 1.

## 8. Technical facts most likely to matter (full detail in the agent context brief)

- Latents from `stabilityai/sd-vae-ft-mse`, frames 160×120 → (4, 15, 20), height padded to 16, 80 tokens at patch 2.
- Context by channel concatenation: 4 past latents (16 ch) + noisy target (4 ch) → `in_channels=20`; `learn_sigma=True`.
- Action via repurposed `LabelEmbedder` (18 VizDoom actions), `c = t_emb + action_emb` → adaLN-Zero; 10% dropout retained; CFG available but unused.
- Data: HF `arnaudstiegler/vizdoom-500-episodes-skipframe-4-lvl5`: 97 parquet shards, 48 GB, 2,470,229 rows of (episode_id, JPEG frame, action, health, step_id), rows grouped by episode. 500 episodes minus 4 windows each gives the N = 2,468,229 of the old consolidated arrays. `doom_data.EpisodeWindowDataset` indexes per-episode files directly; `build_dataset.py` is kept only for the legacy layout.
- Recipe: global batch 32, fused AdamW lr 1e-4, wd 0, warmup 500, grad clip 1.0, bf16, DDPM 1000 linear; in-training and eval sampling is respaced ancestral DDPM at 50 steps via `p_sample_loop` (the code and docs call it DDIM; `--sampler ddim` in `eval_metrics.py` is the true DDIM); EMA 0.9999 in fp32 math.
- Companion projects: `~/Documents/Github/perseve` (adversarial SDG) and `~/Documents/Github/Lego_Manipulation` (LEGO seg + pose). The fall plan's goal 3 carries this world model into LEGO trajectory decisions using the Isaac Sim pipeline from perseve.
