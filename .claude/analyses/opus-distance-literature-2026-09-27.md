# Distance literature and method check (Opus 5.5 high, 2026-09-27)

All numbers were recomputed from `paired_windows.csv`, `adapt_summary.json` (reproduced exactly), `results/fresh_rescore/*`, `compare_fresh_sd1.csv`, `distances_sd1.json` and `distance_table.md`, with the tuned decoder, live weights and seed 0 unless stated.

## One-line answers

1. **Literature.** Distances predict transfer when the shift range is wide (OTDD, Westny), when difficulty is fixed by construction (XEWorld, Deng), or when the distance lives in the model's representation (LEEP, Westny). We have none of the three, and with a free label the model-side score reduces to the zero-shot error S0.
2. **Diagnosis.** The binding constraint is construct, not noise. The arenas are ordered by intrinsic predictability. It survives adaptation, is shared by three backbones and is invisible to an appearance distance. n = 13 and the grid only limit moderate effects and the cost outcome.
3. **Distance that shows the shift.** Draw D's family step against all three backbones' zero-shot skill: every arena sits below every training map, for 3 of 3 backbones. Add the motion-matched skill deficit Δ as the model-side panel. It passes the floor and orders the arenas but needs the model.
4. **Method.** Make A4000 and the area under the curve (AUC) primary, and report cost with Harrell's C. Add arena-bootstrap CIs, power and outcome reliability. More seeds, episodes, rates or new distances on these arenas are not worth it.
5. **October.** Use constructed shifts with difficulty held fixed: retextured training maps, layout-controlled generated arenas and other-WAD maps. Score zero-shot on about 30 maps and adapt 24, for about 80 GPU-hours.

## 1. Literature

| Method (arXiv) | Measures | Shown to predict | Model? | Fits us? |
|---|---|---|---|---|
| OTDD 2002.02923; s-OTDD 2501.18901 (ICML 2025) | OT between labelled datasets | fine-tune accuracy gap across different datasets (*NIST ρ 0.40 to 0.66, p ≈ 0.07) | no | D is the unlabelled sliced cousin; the evidence uses wide ranges |
| LEEP 2002.12462, LogME 2102.11005, NCE 1908.08142, TransRate 2106.09362, H-score (ICIP 2019, **VERIFY**) | source model's fit to target labels | fine-tuned accuracy; LEEP Fig. 4 convergence speed | yes | S0 is this, with the next frame as the label |
| Ben-David 2010; proxy A-distance 1505.07818 | domain-classifier error | a bound: source error + divergence + λ | classifier | λ, the joint error, is what varies between arenas |
| FVD 1812.01717 (venue **VERIFY**); FVD content bias 2404.12391 (CVPR 2024); FD-DINOv2 2306.04675 (NeurIPS 2023); JEDi (V-JEPA MMD) 2410.05203 (venue **VERIFY**) | feature-distribution distance | generation quality; FD predicts accuracy on synthetic shifts (Deng 2007.02915) but loses to confidence baselines on natural ones (Guillory 2107.03315) | feature net | appearance-weighted; pixel D and latent D already agree at 0.86 to 0.90 |
| Accuracy-on-the-line 2107.04649; agreement-on-the-line 2206.13089; ATC 2201.04234; disagreement 2106.13799 | OOD performance from model behaviour | OOD accuracy; difficulty shared across models | yes | our cross-backbone agreement (A0 ρ 0.95 PixArt, 0.81 SD 3.5) is this |
| DARC 2006.13916; PAD 2007.04309 (tests include ViZDoom); Sim2Real SRCC 1912.06321; bisimulation 2006.10742 | transition classifiers, test-time adaptation, sim-real rank agreement, behavioural distance | reward correction, adaptation gain, SRCC 0.18 | mixed | none predicts adaptation cost from a distance |
| World-model error as novelty score 2310.08731 (ICML 2025); MOPO 2005.13239 | predicted-vs-observed misalignment; ensemble spread | novelty; offline-RL penalty | yes | Δ is this at map level |
| XEWorld 2608.05799 | appearance vs kinematic distance, held-out robots in identical scenes | held-out error (appearance r = 0.81, kinematic not) | no | scenes fix difficulty; our arenas do not |
| Westny 2606.30777 (ECCV 2026) | directed KL on learned scene embeddings, 24 datasets | zero-shot ρ 0.81; forgetting 0.73; Wasserstein 0.45 to 0.49 | embedding | wide range; basis of candidate 3 |
| Kang 2411.02385 (ICML 2025) | video-model generalisation priority | colour > size > velocity > shape | none | appearance dominates |
| DrivingGen 2601.01528 (ICLR 2026) | scenario coverage as a benchmark property | nothing per scenario | none | no driving paper found where coverage predicts per-scenario error or cost (search not exhaustive) |

I checked the ids and unmarked venues against the arXiv API today. Quoted statistics come from the verified memos `lit-transferability`, `distance-study-literature` and `related-work-map`.

## 2. Diagnosis

| Constraint | Evidence | Binding? |
|---|---|---|
| Noise | D's subset SD is 0.0037 against a between-arena SD of 0.040. Outcome reliability under held-out-episode resampling is 0.93 to 0.97. Seed spread at 4,000 is 0.02 to 0.06 dB against an arena SD of 0.53 (stock decoder). | no |
| Half-gap coupling | A0 vs cost −0.19 | no |
| Grid | At 25% of the gap all 13 cross at 250; at 75%, 10 of 13 are censored; crossings flip in 42 to 54% of resamples on arenas 6, 11, 14 and 17. On the grid-free outcomes D is still null (A4000 +0.10, AUC +0.03). | cost only |
| n = 13 | Power is 0.37 at a true ρ of 0.5. D vs A4000 +0.10, arena-bootstrap CI [−0.49, +0.66]. | limits exclusion |
| Construct | Arena rank persists (S0 vs S4000 0.85) and is shared across backbones. It is unrelated to motion (+0.09), persistence (−0.09) and D (−0.05). S0 vs A4000 +0.90 [0.66, 0.97], partial on motion and persistence +0.95, permutation p < 0.001. | **yes** |

The outcome mixes intrinsic difficulty with shift, and difficulty dominates. The gain A4000 − A0 subtracts the arena's own ceiling, and it is the only outcome where D has the expected sign: +0.46 [−0.21, +0.89], partial +0.26 (p 0.43), +0.31 without arena 7. Fixing difficulty while the shift varies would loosen the constraint.

## 3. Candidates

| Rank | Candidate | Definition | Model? | Gates (computed) | Cost | Prior it orders arenas |
|---|---|---|---|---|---|---|
| 1 | **D family step × three backbones** | x: D per map (training maps' validation D, arenas' fresh D); y: zero-shot latent skill for U-Net, PixArt, SD 3.5 | no | Floor 0.026 to 0.069 vs 0.118 to 0.270. All 13 arenas lie below every training map's skill for all three backbones (U-Net 1.14 to 1.96 vs 2.55 to 3.56). D vs S0 within arenas −0.05. | CPU minutes | ~0 (measured) |
| 2 | **Δ, motion-matched skill deficit** | per window: home skill in the same copy-last-error decile minus the zero-shot skill, averaged per map; floor from each training map scored against the other three | yes | Floor passes: U-Net −0.62 to +0.46 vs 0.89 to 1.86; PixArt −0.57 to 0.35 vs 0.73 to 1.70; SD 3.5 −0.76 to 0.55 vs 1.32 to 3.01. Vs A4000 −0.90; cost +0.65 (p 0.017); fraction closed −0.63; cross-backbone 0.92 and 0.88. S0 re-scaled (ρ −0.89), vs D +0.02. | done | high here by construction; ~0.7 prospectively |
| 3 | Directed Gaussian KL on U-Net mid-block features (Westny recipe) | target windows' feature Gaussian vs training memory, frozen before any read | yes, no outcome | four pre-registered gates | half a day of code + 1 to 2 A4000-h | 0.2 to 0.25 |

A post-hoc exceedance variant of Δ does better still: the share of windows below the home 10th percentile is 0.04 to 0.19 on training maps against 0.23 to 0.81 on arenas, with A4000 −0.96. It was chosen after seeing the data, so it is a sensitivity row only.

Candidate 1 serves Rohan's sentence best. Suggested text: "The shift appears as a step in both the footage and the models. Every arena lies farther from the training footage than any training map's held-out episodes, and every arena sits below every training map in zero-shot skill for all three backbones. Within the step, the model's motion-matched deficit, not the footage distance, orders the arenas." Δ is the second panel.

## 4. Method changes by Tuesday

- **Do (CPU).**
  - Make A4000 and AUC the primary outcomes (Taylor and Stone's threshold-free advice), with gain and fraction of the gap closed beside them.
  - Keep cost as secondary and handle censoring with Harrell's C instead of setting it to 8,000. S0 scores 0.73, A0 0.56 and D 0.36, which is below chance.
  - Put an arena-bootstrap CI on every ρ and add the power sentence.
  - State outcome reliability and seed spread, so the null reads as a finding.
  - Build the candidate 1 figure.
- **Optional (about 5 A4000-hours, only if a card idles).** Add checkpoints at 25, 50 and 100 updates. They would break the four-way tie at 250 on cost but cannot change the distance conclusion.
- **Not worth it.**
  - More seeds.
  - More episodes: k = 1 against k = 16 differs by at most 0.25 dB.
  - Rate or budget sweeps (done).
  - New distances on these 13 arenas, which are now development data.
  - FVD, DINO or JEDi distances.
  - Full fine-tunes on every arena.

## 5. October

- **Appearance arm.** Four training maps × three retexture levels, with geometry, bots and spawns fixed. Textures are edited with omgifol (github.com/devinacker/omgifol, a Python library for WAD maps and texture lumps). This is XEWorld's control. Predicted result: D orders deficit and gain.
- **Layout arm.** Twelve Obsidian arenas with training textures, stratified by D and Δ terciles before recording.
- **Range arm.** Six to ten maps from other WADs.
- **Cost.** Zero-shot on all ~30 maps takes about 8 A4000-minutes each and gives n ≈ 30 for the D-vs-deficit test. Adapting 24 maps with two seeds costs about 55 A4000-hours plus 15 hours of scoring. Encoding takes about 8 A6000-hours. The raw recordings are about 170 GB (0.24 GB per episode × 720), deleted after encoding.
- **Before any read.** Freeze D, Δ (on the adaptation episodes) and candidate 3, and seal the test maps.
