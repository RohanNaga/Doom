# Astra (gpt-6-astra) round 7 reply — paper contribution

- Thread id: 01a0aace-5994-7892-abd4-086da83f6907 (round-6 thread 01a0a160-b1ab-7f93-8bc4-86ddd74a0d07 had expired; this is a fresh session with the round-7 preamble)
- Date: 2026-09-16
- Sandbox: read-only, model_reasoning_effort high
- Prompt: `.claude/analyses/astra-brief-2026-09-16-contribution.md` verbatim

---

**My answer: we can make a distinct contribution in 14 days by delivering a reproducible Doom simulation benchmark and explaining a concrete evaluation failure: apparent long-horizon fidelity can survive even when predictions offer little advantage over persistence.** I do not see a credible path to claiming a better game simulator than the prior systems within this budget.

That is a worthwhile workshop contribution if we finish the diagnosis and release. Another model row alone would not establish it.

**1. My three candidate framings, ranked**

| Rank | Framing | What is new and useful | What we still need |
|---|---|---|---|
| **1** | **A reproducible Freedoom benchmark that separates image fidelity, persistence and motion** | A reusable corpus and common evaluation that expose how model rankings change when trivial persistence and motion are considered together. | The motion audit below; downloadable data, checkpoints and split manifests; a verified reproduction from a clean checkout. |
| **2** | **Which pretrained model package works under a small adaptation budget?** | A shared-recipe comparison revealing a tradeoff: U-Net wins seen perceptual quality, while DiT retains higher long-rollout and unseen-map PSNR. | Finished PixArt evaluation, consistent live/EMA policy, measured compute accounting, and careful separation of the video pilot. |
| **3** | **Generalization to unfamiliar Doom layouts under limited training** | A broader, explicit held-out-map protocol could establish where the packages transfer differently. | More held-out maps, fresh training, map-level uncertainty and exclusion of those maps from every fitted component, including the decoder and IDM. |

For **#1**, the strongest existing evidence is more specific than "metrics disagree":

- At horizon 64, DiT's PSNR advantage over **copy-seed** is only **+0.44 dB for seed 0 and −0.18 dB for seed 1**.
- Copy-seed's LPIPS is **0.510**, better than all three completed models' **0.550–0.566**.
- Thus, beating the U-Net's rollout PSNR does not establish useful simulation. It also does **not yet establish that DiT freezes**.

The novelty boundary matters:

- [GameNGen](https://arxiv.org/abs/2408.14837) already established real-time neural Doom simulation.
- [MultiGen](https://arxiv.org/abs/2603.06679) already addresses multiple layouts, explicit memory and multiplayer.
- **[PlayGen](https://arxiv.org/abs/2412.00887) already uses a DiT for Doom**, evaluates rollouts through 1,024 frames, and introduces ActAcc/ProbDiff. Our older "first DiT for Doom" language is wrong.
- DIAMOND already demonstrates open world models and downstream RL utility; Oasis already provides a public transformer game-world-model precedent.

Consequently, neither multiple maps, action metrics, openness nor transformers constitute the contribution individually. **The contribution is the particular reusable benchmark and the finding it enables.** A reviewer could reasonably call that new; claiming a new simulation method would invite justified criticism.

For **#2**, call these *pretrained packages under a shared recipe*. Initialization, architecture and conditioning interfaces differ. Equal updates and examples do not mean equal compute or isolate an architectural cause.

For **#3**, the current two-map result is promising but narrow. Episode bootstrap quantifies uncertainty within those maps; it cannot establish generalization across a population of maps.

**2. The single additional experiment: a motion-and-persistence audit**

The research question is:

> **Does DiT retain higher long-horizon PSNR by preserving the required motion, or by changing less than the real scene?**

I would use the existing paired **256 × 64 rollouts**, both DiT seeds, U-Net, real trajectories and copy-seed. Add PixArt when its already-scheduled evaluation finishes.

The bounded design:

- Measure motion by horizon using both consecutive-frame change and a fixed optical-flow estimator, excluding the HUD.
- Calibrate against real clips and VAE reconstructions: flicker can inflate frame differences, and blur can affect estimated flow.
- Stratify by **ground-truth motion** and action category; inspect signed camera motion for turning actions. Keep stationary/collision cases separate.
- Plot motion preservation alongside **improvement over copy-seed**, for PSNR and LPIPS, with paired episode-bootstrap intervals.

If the PSNR lead accompanies suppressed motion, we have a concrete failure diagnosis. If motion remains comparable, we reject that explanation and retain the measured tradeoff. Neither outcome alone certifies correct dynamics or causal action control.

I would cap this at **one A6000-day of additional work**, including any necessary rollout regeneration; that is a proposed budget, not measured runtime. Target the result for the September 20 draft.

Against each alternative:

- **Train on 8 maps/test on 9:** strongest runner-up, but two 30k runs introduce training-duration uncertainty and require a fully clean split; starting from the existing Doom checkpoints would leak newly held-out maps.
- **Compute/data-matched curve:** useful for efficiency claims, but it does not resolve whether the long-horizon advantage represents useful dynamics; reuse saved checkpoints if cheap.
- **Cosmos-Predict2.5:** another adaptation with multiple changed variables and integration risk, while the first video-prior experiment remains unfinished.
- **Small human study:** realism judgments would add evidence about appearance, but would not resolve persistence versus correct motion.

There is no scientific requirement to consume all six remaining A6000-days.

**3. What to stop doing or leave out**

- **Stop searching for a universal DiT win.** Replace the winner-required requirements with a question whose answer can favor either model.
- **Stop adding model families after SkyReels.** Finish that row, but label it exploratory: **80k versus 2.88M target presentations is a 36× exposure difference**, alongside different VAE, context and objective.
- **Drop cross-paper parity claims.** Our U-Net EMA LPIPS of 0.250 and GameNGen's 0.249 do not demonstrate equal quality on different datasets and protocols.
- **Avoid "stable dynamics," "superior generalization," and "pretraining explains the gap."** Those interpretations exceed the current evidence.
- **Keep persistence in the main paper.** I revise my round-6 recommendation to put that reference in the appendix: it now changes the interpretation of the headline.
- Keep historical misaligned-action runs, detailed engineering chronology and extensive checkpoint tables out of the four-page narrative.

The main paper needs one compact results table, one figure joining horizon/persistence/motion, and enough release detail for another lab to use the benchmark.

**4. The claim and title I would submit**

**Claim:** "We introduce a reproducible multi-map Freedoom benchmark and show that, under a shared adaptation recipe, higher long-horizon PSNR can coexist with perceptual error worse than a frozen seed frame, motivating persistence- and motion-aware evaluation of neural game simulators."

**Title:** *DoomDiT: A Reproducible Study of Fidelity and Persistence in Neural Doom Simulation*

I reviewed the project context and earlier rounds, directly checked GameNGen, MultiGen and PlayGen, and committed the [recommendation and context update](/Users/rohan/Documents/Github/Doom/.claude/analyses/astra-contribution-2026-09-16.md) as `1c3bbfa`. I did not run experiments or verify release completeness; whether reduced motion explains DiT's advantage remains unanswered.
