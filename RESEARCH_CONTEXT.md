# DoomDiT — Research Context

> Updated 2026-09-01. Entry point for every new chat in this repo.
> Read order: this file → `.claude/analyses/DOOMDIT_AGENT_CONTEXT.md` (dense technical brief, verified against code July 15) → `.claude/analyses/doomdit-full-recap-2026-07.md` (full recap: literature through July 2026, venues, robotics extension) → `IDEAS.md` (scenario-latent skill files).

## 0. Where we are going now (as of Sep 1, 2026)

**The objective:** get the DoomDiT workshop paper together in the next three to four weeks. The paper is the controlled U-Net vs DiT world-model comparison under matched data and compute, plus the small-compute recipe. The stated fall target is mid October; the deadline that actually fits is the **CoRL 2026 PhysWM workshop, Sep 30 (4 pages)**, with the NeurIPS "World Models in Physical AI" workshop (Sep 5, 8 pages) only reachable as a sprint on existing numbers. Rohan is leaning toward Sep 30 CoRL as of Sep 1; confirm in the Sep 2 1-on-1.

**What has to happen, in order:** recover or re-train the U-Net baseline and rebuild the PSNR/LPIPS harness so the headline reproduces → define a held-out episode split → add autoregressive metrics (FVD vs rollout length, drift curves, IDM action accuracy) → size the final larger run from the gap analysis and launch it only if it finishes before the deadline → write the 4-page version. Multi-game conditioning stays a stretch or future work.

**Why it matters beyond the paper:** the same action-conditioned latent DiT is the starting point for fall goal 3, a world model for LEGO trajectory decisions, and Changliu's May 4 suggestion of action rollouts for long-horizon planning plus sim-switching at contact-rich points using the perseve Isaac Sim pipeline. Keep the code reusable for that.

**Standing rhythm:** weekly 1-on-1 with Changliu every Wednesday 1:00 to 1:30 PM from Sep 2 to Dec 9; Friday Slack update tagged by goal.

## 0b. How we got here (condensed)

DoomDiT is an action-conditioned latent diffusion world model: DiT-XL/2 (673M params) predicts the next VizDoom frame from 4 past frame latents plus one discrete action, trained on Superman (4× A4000 16 GB, about 72 GPU-hours) as the CMU 18-789 final project with Keerthana Chirumamilla. Best checkpoint is step 87,200, loss 0.0481, released as GitHub tag `002-DiT-XL-2-best-90k`. Headline teacher-forced numbers: DiT 26.04 dB PSNR / 0.153 LPIPS vs a matched U-Net baseline 24.60 / 0.198 (GameNGen reports 29.43 / 0.249, but that is a cross-paper comparison). The defensible contribution is the **controlled U-Net vs DiT comparison under matched data and compute plus the small-compute recipe**, not "first DiT world model" (Oasis, Matrix-Game, NFD, SANA-WM already exist). Nothing has been trained or evaluated since April 22. The fall goal is a workshop paper by mid October.

## 1. Where work happens

| What | Where |
|---|---|
| Repo | `~/Documents/Github/Doom` on `main` (`github.com/RohanNaga/Doom`). Open chats here. Two worktrees under `.claude/worktrees/` are leftovers from July; `IDEAS.md` has been copied to the main checkout. |
| Server | Superman `rohan@128.2.204.116`, project `/home/rohan/Doom/Doom/`. Full latent arrays (`data/full/`, 30 GB) and runs 004/012 live only there. Leave 2 GPUs free, keep 250 GB disk free. Use the `/run-server` skill. |
| Weights | GitHub release `002-DiT-XL-2-best-90k` (`best.pt`, `0090000.pt`, 2.6 GB each, bf16). `bash download_weights.sh`. |
| Collaborator | Keerthana Chirumamilla holds the frame-to-latent encoding script, and probably the U-Net baseline and the PSNR/LPIPS harness. None of the three is in this repo. |

## 2. What is missing for a paper (in priority order)

1. **U-Net baseline code and checkpoints**: not in the repo, not in any transcript. Recover from Superman or Keerthana, or re-train a matched-params U-Net from the documented recipe.
2. **PSNR/LPIPS harness**: not in the repo. Rebuild (short script over the fixed eval segments).
3. **Autoregressive metrics**: none exist. A 2026 reviewer expects FVD vs rollout length, drift-vs-horizon curves, and IDM action-following accuracy (train a small VizDoom inverse dynamics model). `rollout_video.py` is the starting point. Teacher-forced-only evaluation is the most likely rejection reason.
4. **No validation split, single seed**; `best.pt` was picked by smoothed train loss.
5. `eval_checkpoint.py` NameError (`ema_state`); README is still upstream fast-DiT; presentation claims gradient checkpointing was on but the code default is off.
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

1. Which deadline (section 5). This decides whether the "final larger run" happens before or after the paper.
2. Re-train the U-Net baseline from the recipe, or spend the first days recovering Keerthana's code.
3. Evaluation set: define a held-out episode split now, since none exists.
4. Whether the multi-game stretch belongs in this paper at all (recommendation: future work).

## 7. First-session checklist

1. Record the Aug 26 1-on-1 outcome regarding Doom as lab work.
2. Pick the deadline (section 5) and write the experiment list backwards from it.
3. On Superman: inventory `/home/rohan/Doom/Doom/`, runs 004/012, any U-Net directories, any metric scripts. Ask Keerthana for the encoding script, U-Net code, and harness.
4. Fix `eval_checkpoint.py`; define the held-out split; rebuild PSNR/LPIPS; run the current checkpoint through it to reproduce 26.04 / 0.153.
5. Build the autoregressive suite: FVD vs rollout length, drift curves, VizDoom IDM.
6. Size the final run from the gap analysis; launch only if it finishes before the chosen deadline.
7. Rewrite `README.md` as a DoomDiT README; commit the encoding pipeline.

## 8. Technical facts most likely to matter (full detail in the agent context brief)

- Latents from `stabilityai/sd-vae-ft-mse`, frames 160×120 → (4, 15, 20), height padded to 16, 80 tokens at patch 2.
- Context by channel concatenation: 4 past latents (16 ch) + noisy target (4 ch) → `in_channels=20`; `learn_sigma=True`.
- Action via repurposed `LabelEmbedder` (18 VizDoom actions), `c = t_emb + action_emb` → adaLN-Zero; 10% dropout retained; CFG available but unused.
- Data: HF `arnaudstiegler/vizdoom-500-episodes-skipframe-4-lvl5`; `build_dataset.py` makes N = 2,468,229 sliding-window samples (mmap, Superman only).
- Recipe: global batch 32, fused AdamW lr 1e-4, wd 0, warmup 500, grad clip 1.0, bf16, DDPM 1000 linear, in-training DDIM 50; EMA 0.9999 in fp32 math.
- Companion projects: `~/Documents/Github/perseve` (adversarial SDG) and `~/Documents/Github/Lego_Manipulation` (LEGO seg + pose). The fall plan's goal 3 carries this world model into LEGO trajectory decisions using the Isaac Sim pipeline from perseve.
