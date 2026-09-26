thread id: 01a0df4e-313f-7930-89ad-552ac64e9399

# Framing review

Paths refer to this worktree. Runtime estimates are planning arithmetic, not measurements. The supervisor's updated schedule supersedes stale RC timing.

## 1. Frame and cheap levers

**Yes: the decoder is the right first component to adapt for a cheap rendering diagnostic, but it cannot repair latent dynamics.**

Ask what transfers when the map changes; do not assume physics transfers. Identical engine rules do not imply identical observable transitions: geometry, collisions, occlusion, enemies and visitation change. Turning establishes local control response. Current U-Net reversal is 0.867 with motion ratio 0.893, on training-map validation windows only (`docs/RESEARCHER_DOSSIER.md:660,667`).

Call the frame **diagnostic separation of control response, latent prediction and rendering**. Latent MSE mixes dynamics, appearance and encoder distortion. Decoder and prediction errors interact; subtracting reconstruction PSNR cannot allocate the gap additively. Hold predicted latents fixed across decoders; separate adaptation and evaluation episodes, and label target-map adaptation separately from zero-shot transfer.

Failed turning with a valid estimator motivates a control-path adapter; preserved turning with poor latent prediction motivates a context/input-projection adapter. Small parameter counts do not guarantee cheap backpropagation. A denoising-step/spacing sweep is cheaper but already shows a perception–distortion tradeoff (`docs/RESEARCHER_DOSSIER.md:615–627`). Preserve the trained noise schedule. Observation CFG lacks its trained dropout branch (`docs/RESEARCHER_DOSSIER.md:1221`). A middle-layer skip has no stronger current justification.

## 2. Strongest deadline paper

**One-sentence contribution:** “An open multi-map Doom evaluation shows how gains over persistence change under map shift, and uses control interventions, latent errors and paired rendering to distinguish failures that image fidelity alone conflates.”

U-Net gains on the **4 training maps** are **+0.51 to +1.79 dB** and **+0.006 to +0.056 LPIPS gain** (`results/distance_study/figure_unet_h1_amended/stats.json`, `maps[training=true].rows.unet.{gain,lpips_gain}`). LPIPS gain means persistence minus model (`paper/make_distance_figure.py:229`). Thus LPIPS changes from **4/4 positive to 26/26 negative**; PSNR is negative on **17/26 unseen maps** (same JSON, `maps[training=false].rows.unet`). These are four in-distribution maps versus different unseen maps, **not a matched comparison**. Other-backbone replication remains pending (`verdict.rows_missing`).

Distance supports association, not prediction: pooled rho **−0.734**, unseen-only family-adjusted **−0.431 [−0.628,+0.158]** (`results/distance_study/figure_unet_h1_amended/stats.json`, `primary.partial_spearman`, `amended_assessment.structural_controls.unseen_only_family`). Preserve the failed gate; use the authoritative rerun's interval because the local primary bootstrap lacks per-window files (`RESEARCH_CONTEXT.md:191`).

**Figures:** per-map persistence gains as the main figure, with distance/structural controls in a companion panel; a diagnostic figure combining control response, latent ratio and paired rendering. If diagnostics remain incomplete, substitute the existing rollout strip with paired checkpoint rates.

**Tables:** backbone packages with persistence/reconstruction references, short-horizon and rollout metrics, compute and uncertainty; compact diagnostics by map group. Packages are not an architecture ablation. Target-tuned columns require separate held-out episodes.

**Cut:** “first multi-map Doom,” novel persistence, validated distance predictor, training chronology, probe oscillations, layer localization and collapse repair. Prior precedents: `.claude/analyses/distance-study-literature-2026-09-25.md:79,129–140`. Sunday's draft must stand without target-map adaptation; Monday fills final results.

## 3. Marginal decisions, ranked by paper value per GPU-hour

Directional checks, latent-ratio scoring and training-map decoder tunes are **already running tonight**: sunk commitments, not additional choices. Do not charge them again. No backbone retraining.

**(a) Before Sunday's evening draft — one A6000 in gaps.**

- **First: synthesis, 0 GPU-hours.** Consume in-flight outputs, build the sign-flip figure, reconcile provenance and compute available uncertainty on CPU. Complete the draft with final-model gaps explicitly marked.
- **Second: the paired-rendering experiment in Q5, at most 1.8 marginal GPU-hours.** This is a conservative U-Net full-pass allowance (`RESEARCH_CONTEXT.md:237`), not measured rerender speed. Time the first map; fix a balanced window count that fits before reading outcomes. Reuse predictions where retained. If no slot exists, retain the stock-decoder result and defer this experiment.

**(b) Monday — two cards; SD3.5 final weights arrive Monday 00:30 EDT (supervisor's schedule).**

- **Third: final cross-backbone map scoring, approximately 5.4 GPU-hours**, split PixArt/SD3.5 across cards after their required final evaluations. Derivation: U-Net-like PixArt assumed 1.8 hours (`RESEARCH_CONTEXT.md:237`), SD3.5 twice that (`.claude/analyses/distance-study-design-2026-09-24.md:108`). These estimates exclude already-booked final rollout/validation reads; those take precedence. Do not rerun completed U-Net scoring.

**(c) Deferred unless evidence and time justify reopening.**

- **Fourth: unseen-map decoder adaptation — no-go for Sunday's four-map proposal.** Reopen Monday only if the new training-map-tuned decoder leaves a mean unseen-versus-seen reconstruction deficit **at least 2.2 dB**, measured on identically sampled, preselected maps, and final tables are complete. This is a proposed threshold anchored to historical **28.6−26.4=2.2 dB**, not a validated cutoff; stock SD1 reconstruction was about **23.5 dB** (`.claude/analyses/astra-brief-2026-09-26.md:30`). Those historical populations/recipes are not directly comparable. Require a confirmed **5.8 GPU-hour** slot: **4** tuning (`RESEARCH_CONTEXT.md:190`) plus **1.8** scoring reserve (`RESEARCH_CONTEXT.md:237`). Otherwise defer to October. Rohan's authorization remains pending.

Layer adapters, schedule searches and collapse rescue stay deferred.

## 4. October paper

Use frozen backbones with controlled interventions: retexture identical geometry/trajectories, then vary geometry while matching appearance/control coverage. Compare decoder, input and control adapters at matched data/compute, with held-out adaptation episodes and evaluation maps. Include collision and revisit tests; layer probes require causal interventions across denoising times.

Conditional stronger claim: “Controlled shifts identify rendering versus latent-transition failures, enabling targeted low-cost adaptation without backbone retraining.” Distance needs out-of-sample prediction beyond family/membership baselines. October workshop alternatives remain unverified in `RESEARCH_CONTEXT.md:136–152`.

## 5. Hostile reviewer and ONE experiment

Attack: **“Your transfer deficit is decoder loss on static footage.”** Run **paired decoder replacement on identical U-Net predictions**, budget **1.8 GPU-hours**, shared with Q3, not additional. Primary endpoint: change in per-map LPIPS gain when stock decoding is replaced by the training-map-tuned decoder, within fixed real-motion strata. Reuse tonight's decoder tune and latent-ratio output; reuse directional results as interpretation, not another experiment. Bootstrap episodes. A residual perceptual deficit alongside latent error blunts the decoder-only explanation; disappearance revises the conclusion. Neither outcome proves physics transfer.

## Where I disagree with the Sep 26 afternoon review

I broadly agree with its contribution sentence, but prefer diagnostic separation over reliability as the final clause. I disagree with its distance-scatter Figure 1: the observed persistence sign flip is clearer than the structurally confounded association (`.claude/analyses/astra-review-2026-09-26.md:88–97`).

Its page budget is **0.45/0.8/1.5/0.8/0.45** for motivation, methods, transfer, closed-loop and limitations (same file, line 90). Keep the other allocations; share the closed-loop allocation with diagnostics rather than reserving it entirely for stability. Matched events are **23/256 versus 9/256**, but **11/240 versus 8/240** excluding the collapse checkpoint (same file, lines 69–74; `results/sd35_stability/collapse_rates_50k_130k.md:9–47`). The inspected strip establishes striking blanking, not trajectory correctness. Paired rendering deserves more than its proposed incidental column (afternoon review, line 119); clamp rescue deserves less priority.
