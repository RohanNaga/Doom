# Brief for Astra, Sep 16 2026 22:20 EDT: can this be a "how to train a small-budget world model" paper, and which single improvement do we run?

Thread: 01a0aace-5994-7892-abd4-086da83f6907 (round 9). Rohan reads the answer Thursday morning.

## Rohan's ask tonight (verbatim intent)
"The framing is okay, but still not great. It would be nice to frame it as: you want to train a world model at a smaller compute size? Here's the best methodology. We need to think through what type of methodology we want to come up with based on all these runs. Maybe we can take one of these and make it better, improve it somehow." He also relayed his partner's objection: the U-Net has more parameters than the DiT; could parameter count explain the gain?

Propose your own answer first. I have a view but am holding it back for this exchange.

## What exists (read, do not trust my summary)
- /Users/rohan/Documents/Github/Doom/RESEARCH_CONTEXT.md section 0 (framing decided Sep 16, rows, audits, timeline) and section 7 entries from Sep 14 on.
- Round 7 and 8 of this thread: .claude/analyses/astra-contribution-2026-09-16.md, red team .claude/analyses/red-team-2026-09-16.md.
- Evidence cards: .claude/analyses/lit/*.md (doom-world-models, dit-world-models-a/b, backbones-and-warm-starts, video-warm-starts). Literature artifact fbcfe1db.
- Recipe as built: RESEARCH_CONTEXT.md "As built, Sep 14" paragraph (v-prediction, linear betas, 32-latent channel-stacked context, context-noise augmentation U(0,0.7) with 10 buckets, action dropout 0, no CFG, AdamW 5e-5 warmup 2k batch 32, 90k updates, fp32 EMA 0.9999, DDIM 50).
- Numbers (live weights, seed 0, teacher-forced PSNR/LPIPS seen; unseen-2; unseen-13 curated; rollout PSNR@64; FVD16/32; final val v-loss):
  DiT-XL/2 ImageNet 675M: 21.06/0.311; 19.52/0.450; 21.43/0.413; 18.30; 232/481; 0.2139 (seed 1: 21.21/0.307; 19.59/0.442; 21.59/0.404; 17.68; 198/407; 0.2139)
  SD1.4 U-Net 860M: 21.36/0.270 (EMA 21.67/0.250); 19.14/0.446; 21.15/0.412; 16.03; 184/356; 0.2043
  PixArt-alpha 512 611.6M: 21.35/0.272 (EMA 21.60/0.255); 19.39/0.434; 21.29/0.411; 17.18; 211/472; 0.2031
  copy-last 19.41 / 18.50 / 20.92; copy-seed at h64 17.86/0.510; VAE ceiling 28.61/0.051, 26.36/0.070, 28.89/0.058.
  Compute-matched 80k samples (2.5k updates, warmup 250): DiT 19.20/0.471 (below copy-last), U-Net 19.67/0.411, PixArt 19.89/0.412.
  Motion audit: DiTs under-move (flow ratio at h64 0.58/0.73 vs U-Net 0.75).
  In flight: UniDiffuser v1 952M (034, lr 2.5e-5 after a restore at 5k; val 0.2325 at 12k, ends about Fri 01:00 EDT on GPU 2); SkyReels-V2 DF 1.3B video prior on Wan latents (050, flow objective, L=8, val 0.1803 at 4.5k of 10k, ends about Thu 10:15 EDT on GPU 3). The two rows are not on the same latent space or metric scale as the SD rows.
- Ablations we do NOT have: no recipe-component ablation (v vs eps, context noise on/off, context length, action injection style, EMA on/off is measured only at eval). We have the grid-labels ablation, compute-matched, seed 1 for DiT only.

## Constraints
- CoRL 2026 PhysWM workshop, 4 pages, deadline Sep 30 AoE; internal draft Sep 20, final Sep 28. Everything written must be honest and literature-supported; no "first DiT for Doom", no "beats GameNGen".
- Compute: Spiderman A6000s, GPU 3 free from Thu about 10:15 EDT, GPU 2 free from Fri about 01:00 EDT; GPU 1 shared and unreliable. One 90k run of the SD rows takes about 30 h on one card (0.83 to 0.9 updates/s). Evaluations of 034 and 050 need about 6 to 8 card-hours each. So between Thu noon and Sep 27 we have roughly 2 card-weeks minus evaluation, i.e. at most 4 to 6 full 90k runs, realistically 2 to 3 if we want them evaluated and written up.
- Rohan decides; he wants a proposal to read Thursday morning, not a launched run.

## Questions, in order
1. Parameter count. U-Net 860M vs DiT 675M vs PixArt 611.6M vs UniDiffuser 952M. Does the data let us rule out "the U-Net wins because it is bigger"? Which of our numbers is the cleanest answer to the partner, and what is the smallest additional experiment that would close it (there is no larger adaLN DiT in the SD 1.x latent space; a from-scratch matched-params DiT is not a warm start).
2. The methodology framing. Given only the rows above, can we honestly write "here is the best methodology for a single-environment world model at about 4 card-days", or is that a recipe paper that needs ablations we cannot afford by Sep 28? If the latter, what is the closest honest version (for example "which decision matters most at this budget: the initialisation; here is the evidence and the recipe"), and what would it take to move from "one decision" to "a methodology"?
3. Make one row better. Rank the three cheapest interventions on the strongest SD-latent row (U-Net or PixArt) by expected gain per card-day, with the paper that supports each and the metric you expect to move (teacher-forced LPIPS, rollout PSNR@64, FVD, motion ratio). Candidates I am aware of, do not limit yourself: continuation to 180k updates; Oasis-style context noise at sampling (inference only, no training); REPA alignment loss; history-noise probabilities as in Matrix-Game; EMA as the reported weights; DDIM step sweep; lower lr on the U-Net; larger batch by reclaiming VRAM. Say which one you would launch Thursday on GPU 3 and why.
4. Anything in the plan you would drop to make room.

Reply with numbered answers; mark every claim that depends on a paper with the card or citation, and every claim that depends on our data with the file you read.
