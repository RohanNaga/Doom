# Related-work map for the final framing (2026-09-26, 18:00 EDT)

One page that consolidates the five literature memos for the decisions on `docs/lora_adaptation_design_2026-09-26.html` and for the related-work section of `paper/main.tex`. Sources: `distance-study-literature-2026-09-25.md`, `adaptation-literature-2026-09-26.md`, `lit-adapters-2026-09-26.md`, `lit-scene-generalization-2026-09-26.md`, `lit-transferability-2026-09-26.md`. Every number below is in one of those files with its table or figure; the two that matter most for the decisions were re-checked by main against the arXiv HTML (DiffFit Table 1; XEWorld §5.3 to 5.4). Items marked VERIFY in the memos remain so here.

## 1. Verdicts on each claim of the paper

| Claim | Verdict | Evidence, closest prior work | What the paper may say |
|---|---|---|---|
| World models are evaluated on maps they never saw | partly done | Held-out scenes exist on 1 to 5 domains: XEWorld (2608.05799, held-out robots), AVID (2410.12822, held-out Procgen game and levels, pooled), AdaWorld (2503.18938, 4 environments), Pathdreamer (2105.08756, unseen Matterport buildings, nearest-neighbour baseline), driving holdouts (Vista, GAIA-2, NWM). No Doom or ViZDoom paper evaluates a learned simulator on unseen maps; MultiGen (2603.06679) is ambiguous. Nobody reports a per-scene score. | "World models are rarely evaluated on scenes they never saw, and then on a handful of domains, pooled, without a persistence reference. We score 30 maps individually." |
| Copy-last-frame persistence as the reference | done 2015 to 2019, dropped since | PredNet (1605.08104, unseen CalTech vs Copy Last Frame), ContextVP, Villegas 2019 (1911.01655, copy-last beats every model on Human3.6M), Pathdreamer's nearest-neighbour reprojection, ThinkJEPA 2026 latent persistence, Genie's delta-PSNR form. Absent from GameNGen, DIAMOND, Oasis, Matrix-Game, GameFactory. | "Following the video-prediction convention, every score is reported against copy-last-frame; Matrix-Game 2.0 shows why: a collapsed static output inflates consistency metrics." |
| Ceiling, persistence and model in one table | done in forecasting, new for game video models | OccWorld Table 1, DOME Table 2, DINO-Foresight Table 2 (oracle 77.0 / copy-last 54.7 / model 71.8 mIoU), Luc 2017. | Cite as the template; the new part is the ceiling per held-out map against distance. |
| Failures are appearance, not dynamics (control response survives) | done adjacent | XEWorld §5.4: appearance distance predicts held-out error (r = 0.812 across five folds), kinematic distance does not (r = 0.549, CI crosses zero); "a 2D visual pattern matcher". MineDojo, DMC-GB, RePo separate appearance from dynamics shift for agents. | Cite XEWorld as the robotics counterpart; our directional check is the game-side measurement. |
| Distance predicts the zero-shot deficit | partly done, and weak precedents | s-OTDD (2501.18901) same distance family, ρ = 0.40 on *NIST; Westny 2026 (2606.30777) latent KL, ρ = 0.811 with CI over 552 pairs; Mensink (2103.13318) nearest-source τ = 0.42 vs EMD 0.20; Guillory and Mayilvahanan as null precedents. | Qualified evidence with the failed gate stated, as already decided. Add the directed variant (below). |
| EMA versus live weights in closed loop | new | Nobody compares them; DIAMOND attributes rollout drift to few-step sampling. | State it; expect the DIAMOND objection. |
| Adaptation curves per map, cost predicted by distance | new as a combination | Ingredients: LEEP Fig. 4 (convergence binned by score), Ben-David Theorem 3 and Hanneke-Kpotufe (target samples needed depend on divergence), Hernandez α and Barnett's transfer gap (fitted after training), AdaWorld Fig. 6 (PSNR vs steps, 4 envs, full FT), AVID Fig. 4c (vs data size, 1 game), XEWorld (few-shot full FT, 25 to 75 episodes, forgetting 69 percent LPIPS spike at M = 1), Westny (forgetting ρ = 0.729). | "Adapter adaptation of world models is established (Vista, DreamGen, GameFactory, Cosmos); its cost as a function of a pre-computed domain distance, per target, has not been measured." |

## 2. The closest competitors, ranked

1. **XEWorld** (Chen et al., arXiv 2608.05799, Aug 2026). Held-out robot embodiments in identical scenes; appearance distance predicts LPIPS, kinematic distance does not; full fine-tuning on 25 to 75 episodes recovers the target and forgets the seen robot; no persistence baseline; 2 held-out robots in the main protocol, 5-fold for the distance analysis. Ours: 26 unseen maps of one game, LoRA on frozen weights, a cost curve, a persistence-referenced outcome, forgetting as a guard with a threshold.
2. **Westny et al.** (arXiv 2606.30777, 2026). Latent-embedding KL across 24 trajectory datasets predicts zero-shot error and forgetting; fine-tuning curves for three sources. Discriminative, many sources for one target, fixed-budget outcome.
3. **AVID** (RLC 2025, 2410.12822) and **AdaWorld** (ICML 2025, 2503.18938). The adaptation-curve precedents for world models: data-size curve on one held-out 2D game; PSNR versus steps on four environments with full fine-tuning. Neither has per-scene resolution, a distance or a persistence reference.
4. **s-OTDD** (ICML 2025) and **OTDD** for the distance family; **Pathdreamer** for the persistence-style baseline on unseen 3D scenes; **DiffFit** (ICCV 2023) for LoRA on DiT-XL/2.

## 3. The reference baseline (Rohan's question)

What the field cites as "reasonably good" depends on the sub-field, and none of the conventions is a fixed absolute:

- Game and driving world models (GameNGen, DIAMOND, Oasis, Matrix-Game, Vista) report absolute PSNR, LPIPS, FVD on held-out trajectories of the training scenes, or human preference. There is no reference model; "good" is the previous paper's number on the same data. This is why those papers cannot be compared across scenes, and why a collapsed static model can score well (Matrix-Game 2.0 on Oasis).
- Video prediction 2015 to 2019 reported copy-last-frame beside every model, and sometimes the model lost (Villegas 2019 on Human3.6M). PredNet did exactly our move: trained on KITTI, tested on an unseen dataset, against copy-last.
- Forecasting papers (OccWorld, DOME, DINO-Foresight) print three rows: oracle or tokenizer ceiling, copy-last, model.
- Transfer and domain-adaptation papers normalise per target: in-domain accuracy (Blitzer), target-only error (OTDD), per-game normalisation (Procgen, Atari); Genie's delta-PSNR is a difference of two PSNRs against a counterfactual reference.

So persistence is the right reference for the question we ask, "does the model add anything on this map", and it is the only reference that is defined identically on every map with no training. It is not a "reasonably good" bar; it is the zero line. What must sit beside it, and already does in the design: the decoder's reconstruction ceiling (the upper bound the same autoencoder allows), and the decoder-free latent ratio (so the decoder cannot inflate or deflate the comparison). An in-domain reference, a model trained on the target map, would be the "good" bar; we cannot train it for 26 maps, and the training-map gain (+0.5 to +1.8 dB) is the honest stand-in, which is why the adaptation target is a fraction of it. The one caveat the reviewers will raise: on near-static campaign footage persistence is strong and three maps have the ceiling below persistence in PSNR, so the headline leads with LPIPS and the latent ratio, and the motion content of each map is reported.

## 4. What the literature says about each design decision

| Decision | Literature input | Effect on the recommendation |
|---|---|---|
| 1. Per-map split (6/4, 12/8) | XEWorld adapts on 25 to 75 episodes; AdaWorld on 50 to 200 samples per action; AVID on 100 to 2,500 levels. Our 6 episodes of 5,000 tics is in range by window count. | keep |
| 2. Pilot maps (17, 24, 26, 23) | Full fine-tuning is the upper bound on distant targets (AdAM §3; AVID Table 1: full 25.8 vs adapter 23.8 PSNR); far maps are where LoRA capacity confounds distance. | keep; put the full-fine-tune reference on map 23 or 26 |
| 3. Rank 16, ablation 4 and 64 | Rank 16, alpha 16 is the stated default of the only world-model LoRA recipes found (Vista App. C.3; Cosmos-Predict2 docs). Rank scaling is "generally ineffective" for LoRA in LLM fine-tuning (Zhang 2402.17193); AnimateDiff rank 2 vs 128 comparable. | keep rank 16; the ablation is low-yield, October |
| 4. Trained in full beside LoRA | DiffFit Table 1: on DiT-XL/2, LoRA r8 and r16 give mean FID 81 vs 16.6 full and 16.8 bias-only; the embeddings and biases carry the transfer. Vista: projections alone on a frozen U-Net fail, LoRA plus projections works. | keep (control MLP, positions, input projection, noise embedding); add the MLP to the LoRA targets as the DiT precedent (Cosmos, DiffSynth, Flux) or state why attention-only |
| 5. Full-fine-tune reference | AVID, iVideoGPT, SCOPE all find full fine-tuning above adapters; AdAM shows knowledge-preserving methods lose on far targets. | bring forward to one far map on Monday if any card is free; otherwise October, but say so in the paper |
| 6. Backbones and maps for Sep 30 | AdaWorld and AVID already own the curve shape; XEWorld owns "adapt and forget". Novelty rests on per-map resolution and the distance axis, which need many maps. | the four-map U-Net pilot is a preliminary figure; the 18-map curve is the October claim. A run curve beats a proposal, but not at the cost of the Sunday draft |
| 7. Cost target (half the training-map gain) | Taylor and Stone: time-to-threshold depends on the arbitrary threshold; report several thresholds or a time-vs-threshold curve plus area under the curve. Neyshabur: speed and final gain respond to distance differently. | report cost at 25, 50, 75 percent of the training-map gain, the gain at 2,000 steps, and the area under the curve; the headline uses 50 percent |
| 8. Go for the pilot | No blocker in the literature. | go, with two additions below |

Two additions the memos argue for, both cheap:

- **A directed distance variant, pre-registered now.** Symmetric OT or Wasserstein predicts transfer worse than a directed "is the target covered by the source" measure in two independent studies (Mensink Table 10: EMD τ = 0.20 vs nearest-source τ = 0.42; Westny Table 2: Wasserstein ρ ≈ 0.48 vs KL ≈ 0.81). Our D is symmetric sliced Wasserstein. Define the motion-weighted mean distance from each unseen-map latent to its nearest training-map latent as a secondary x axis, on the frozen clouds, before any adaptation result is read. CPU only.
- **Forgetting plotted against D**, not only guarded (WTE Fig. 3, Westny Fig. 7, XEWorld Fig. 4). The design already measures it at 2,000 steps.

Corrections to the design page text: the learning-rate line "twice the pretraining rate is the usual LoRA convention" has no source; the precedents are an absolute 1e-4 (diffusers, cloneofsimo, GameFactory, DreamGen) or 5× to 10× pretraining (Vista, DiffFit). Keep 1e-4 and cite the absolute default. "Dropout 0" and "alpha equal to rank" follow the diffusers and DiffSynth defaults and the LoRA paper's practice.

## 5. Citations by paper section

- Persistence reference: PredNet (1605.08104), Villegas 2019 (1911.01655), Mathieu 2016 for motion-area evaluation, Genie (2402.15391) for the delta-PSNR form, Matrix-Game 2.0 for the static-collapse case.
- Ceiling / copy-last / model table: OccWorld (2311.16038), DINO-Foresight (2412.11673), Luc 2017 (1703.07684).
- Unseen-scene evaluation: XEWorld (2608.05799), AVID (2410.12822), AdaWorld (2503.18938), Pathdreamer (2105.08756), GameNGen (2408.14837) for the trajectory-only holdout; ViZDoom competition (1809.03470) and Lample-Chaplot (1609.05521) for the map-holdout practice among agents.
- Distance: OTDD, s-OTDD (2501.18901), Mensink (2103.13318), Westny (2606.30777), Guillory (2107.03315) and Mayilvahanan for the null precedent.
- Adaptation: Vista (2405.17398), Cosmos-Predict2 docs, DiffFit (2304.06648), LoRA (2106.09685), Biderman "LoRA learns less and forgets less" (2405.09673), AdAM (2210.16559), Taylor-Stone 2009, Ben-David 2010 Theorem 3, Hanneke-Kpotufe (2002.04747), Hernandez (2102.01293), LEEP (2002.12462).
- Closed loop: DIAMOND (2405.12399) for the sampling-drift attribution.

## 6. Still to verify before submission

XEWorld venue (arXiv only); MultiGen's test maps; AVID's ICLR listing (the RLJ PDF is the primary); Westny and s-OTDD venues as stated; T2I-Adapter, Video Adapter and DreamGen venues; Pathdreamer's unseen-building count. None affects a decision.
