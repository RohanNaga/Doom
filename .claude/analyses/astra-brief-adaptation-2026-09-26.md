# Brief for Astra: the reference baseline and the map-adaptation study (2026-09-26, 17:45 EDT)

You are a second research engineer on this project. Read the artifacts below yourself; do not trust the summaries in this brief where a file can be recomputed. Propose your own design first. A design page written by the main session exists; it is deliberately withheld from this first round and will be shown in the second round for comparison.

## The problem

We have three action-conditioned latent diffusion world models for Doom (a Stable Diffusion 1.4 U-Net, PixArt-alpha DiT, Stable Diffusion 3.5 medium MMDiT), each trained under one recipe on four Doom maps (32 tics of channel-stacked context latents plus a noisy target latent, executed 19-button control tokens, v-prediction, 200k updates of batch 32). They are evaluated on 30 maps: the four training maps (held-out episodes), and 26 maps the models never saw, spanning same-WAD arenas, the commercial campaign and other WADs. Every map carries a copy-last-frame persistence reference on raw frames, the stock decoder's reconstruction ceiling, and a frozen training-to-map distance D (motion-weighted sliced Wasserstein between per-map latent clouds, nearest training map).

Findings so far (U-Net 200k EMA, one-tic teacher-forced): the model beats persistence on all four training maps (+0.5 to +1.8 dB PSNR) and loses to it on 17 of 26 unseen maps in PSNR and 26 of 26 in LPIPS. The deficit correlates with D (partial Spearman of gain over persistence on D, controlling for persistence PSNR: -0.73, bootstrap CI [-0.84, -0.32], n = 30), but two pre-registered checks failed, so the paper calls it qualified evidence. The decoder's reconstruction ceiling is flat across maps (23.9 vs 23.7 dB), so the loss sits in the predicted latents, not the rendering. The turn response (directional check under a turn swap) survives on unseen maps at 0.79 to 0.87.

The workshop paper (CoRL 2026 PhysWM, 4 pages, deadline Oct 1 07:59 EDT) is an evaluation paper led by this persistence sign flip. The October extension, and possibly a preliminary figure in this paper, is the question: what does it take to cross the shift to an unseen map, and does D predict that cost? The proposal on the table is per-map adaptation curves (a small adapter on frozen backbone weights, a few episodes of the target map) with D on the x axis.

## Two questions for this round

**Question 1, the reference baseline.** Rohan asks: what do other papers cite as the baseline, and should ours really be "gain over persistence"? Is copy-last-frame the right reference for "reasonably good", or should it be something else (an in-domain model, a model trained on all maps, a no-action model, a nearest-neighbour retrieval, the reconstruction ceiling, an absolute FVD)? Answer from the literature you can verify and from the numbers on disk. Say what reference you would put in the headline figure, what reference you would use as the outcome of an adaptation curve, and why. If you think persistence is right but incomplete, say what must sit beside it.

**Question 2, the adaptation study.** Design the study you would run to test "D predicts the cost of crossing to an unseen map" under the constraints below. Be concrete: what is adapted and what is frozen, rank or capacity, learning rate, steps, data per map, the per-map split of episodes, which maps, the step and data grids, the outcome per curve point, the definition of "cost", guards against forgetting and against loss of control response, the seeds, the statistics on 18 maps, and what a reviewer who knows XEWorld, AVID, AdaWorld and DiffFit would object to. Rank the pieces by what must be in the Oct 1 paper versus October.

## Constraints

- No backbone retraining (compute and time). Small adapters, decoder tunes, inference analyses and new evaluations are allowed.
- Compute: Superman, 8 RTX A4000 16 GB, at most 6 usable, Sunday to Tuesday; Spiderman's A6000s are booked by the SD 3.5 run (200k around Mon 00:30 EDT), the PixArt final read, decoder tunes and the 30-map scoring queue. The SD 3.5 model (16-channel latents) fits only a 48 GB card.
- Data per unseen map: 10 episodes (18 campaign and other-WAD maps) or 20 episodes (three same-WAD arenas), about 5,000 tics each at 35 Hz; eight seeded maps have only 4 episodes and cannot be split. The training maps have 4 evaluation episodes each.
- Evaluation is in latent space (rollouts never decode); the stock decoder is used only for pixel metrics. Evaluation windows: a fixed 256-window draw per map.
- Default recipes unless something clearly better is agreed. Every training run must stream to W&B.
- The draft goes to the advisor Sunday evening; final tables Monday and Tuesday; Wednesday is polish; the deadline is Thursday 07:59 EDT.

## Artifacts (absolute paths, all under /Users/rohan/Documents/Github/Doom/.claude/worktrees/vibrant-ritchie-6f6280)

- `RESEARCH_CONTEXT.md`: section 0 is the current state; the dated log entries of 2026-09-26 (16:50, 18:30, 19:15, 19:50) hold the framing decisions.
- Literature memos written today by Opus readers, every number from a primary source, VERIFY marks where not:
  - `.claude/analyses/adaptation-literature-2026-09-26.md` (eight steps of the framing; XEWorld arXiv 2608.05799 as the closest competitor; the occupancy-forecasting template for ceiling / copy-last / model tables)
  - `.claude/analyses/lit-adapters-2026-09-26.md` (LoRA recipes on diffusion world models; DiffFit Table 1 LoRA failure on DiT-XL/2; Vista, Cosmos-Predict2 rank 16; AVID, AdaWorld curves)
  - `.claude/analyses/lit-scene-generalization-2026-09-26.md` (unseen-scene benchmarks; nobody reports per-scene scores; Pathdreamer's nearest-neighbour baseline)
  - `.claude/analyses/distance-study-literature-2026-09-25.md` (copy-last baselines 2015 to 2019, Genie's delta-PSNR, per-domain normalisation precedents)
  - `.claude/analyses/lit-transferability-2026-09-26.md` (transferability estimation; may still be in progress when you start; read it if present)
- Framing reviews: `.claude/analyses/framing-review-opus-2026-09-26.md`, `.claude/analyses/framing-review-astra-2026-09-26.md`, `.claude/analyses/astra-layers-review-2026-09-26.md` (your own earlier threads).
- Distance study design and results: `.claude/analyses/distance-study-design-2026-09-24.md`, `results/distance_study/` (distances JSON and CSV, `figure_unet_h1/stats.json` and `distance_table.md`, per-map scores under `scores/040-unet-nexttic_snap_0200000_ema_ddim10/`).
- Code: `train_wm.py` (trainer), `backbones.py` (the three wrappers; the control path is around lines 780 to 800), `eval_tf.py` (teacher-forced metrics including `persist_psnr_raw`, `vae_psnr`, the copy-last latent MSE ratio), `rollout_eval.py`, `directional_check.py`, `distance_study.py`, `paper/make_distance_figure.py`, `paper/main.tex` (the draft).
- The gap explainer page source: `docs/gap_explainer_2026-09-26.html`.

## Claims that could be false

- That copy-last-frame is the right reference on maps where campaign footage is near-static (three maps have the decoder ceiling below raw persistence in PSNR).
- That a per-map "gain over persistence" is comparable across maps of different motion content.
- That LoRA on attention projections with a fully trained control path recovers most of what full fine-tuning would (DiffFit says LoRA fails on DiT-XL/2 transfer; AVID and iVideoGPT find full fine-tuning above every adapter).
- That 6 adaptation episodes of one map are enough to move the model above persistence on that map.
- That the frozen D (computed in SD 1.x latent space on Sep 25) is the right x axis; the map order agrees across three spaces at rank 0.86 to 0.90.

## What has been tried or decided

- Decoder-only adaptation to unseen maps was considered and dropped: the ceiling is flat, so it would recover at most about 0.2 dB and would blur the decomposition.
- Middle-layer post-training (an LLM-style "physics in the middle layers" idea) was reviewed and deferred to October; the layer axis is wrong for these backbones.
- No adaptation run has been launched yet. The trainer has no adapter path yet.

## How to check things

- `python -m pytest paper/fixtures -q` runs the local tests (do not run anything on the lab servers; you have no access to them).
- The per-map numbers are in the CSVs and JSON under `results/distance_study/`; recompute correlations from them rather than quoting the memos.

Answer in a written memo of your own, with the design you would run first and the baseline you would put in the headline figure, each with the sources you relied on. Then the second round will show you the existing design page and ask where you differ.
