# Astra's view: one cohesive evaluation (2026-09-26)

Codex thread `01a0dff9-48f1-72c2-a680-61313c7ca444`. Model gpt-6-astra, reasoning effort high, sandbox workspace-write, 2 turns: the brief, plus one follow-up to pin a single primary quantity for question 1. Brief: `.claude/analyses/brief-evaluation-cohesion-2026-09-26.md`. No server access. Astra recomputed everything from the 17 frozen `*_h1/metrics.json` files under `results/distance_study/scores/040-unet-nexttic_snap_0200000_ema_ddim10/`. The supervisor (Opus 5.5) rechecked the numbers marked below.

## One-line answers

1. **Primary quantity.** (a) In-domain three-backbone table: **tuned-decoder raw PSNR** against the true raw frame, with raw persistence as the baseline. (b) Unseen arenas: **latent skill S**, decoder-free, per backbone. (c) Adaptation curves: **latent skill S** (decision B), with cost defined as the first budget that reaches 50% of that backbone's training-map S. This answer comes from the follow-up; the first pass gave "tuned raw PSNR/LPIPS" for (a).
2. **Decoder policy.** Score both the stock and the tuned decoder on the same predictions. Tuning changes every decoded metric, including both sides of the decoded comparison and possibly their order. It cannot change latent skill, the latent ratio or raw persistence.
3. **Sunday rescore.** Keep per-window identities, latent errors, every model/copy/reconstruction pixel comparison under both decoders, and the predicted latents, so later analysis needs no new sampling.
4. **Presentation.** One table that separates prediction skill, matched rendering and delivered image quality, with reconstruction and raw persistence as explicit reference rows.
5. **Headline qualification.** The decoded PSNR advantage shrinks off the training maps. Neither a mathematical "floor cancellation" nor a purely decoder-caused raw deficit is established.

Definition used throughout: S = -10 * mean_w log10( ||z_hat - z*||^2 / ||z_last - z*||^2 ). Copy-last scores S = 0, and positive S means the model beats copy-last. Aggregation: average windows within an episode, then episodes within a map, then weight maps equally. Uncertainty comes from a paired bootstrap over episodes. Exactly static windows (copy-last error 0, so the ratio is undefined) are flagged, counted, and reported apart.

## Why these choices (Astra's reasoning, condensed)

- **(a) uses raw PSNR, not S.** SD1 and SD3.5 latents live in different representations, and dividing by persistence does not make their geometries comparable. PSNR puts all three backbones on one pixel scale and matches the MSE objective of the decoder fine-tune. S goes beside it as a diagnostic.
- **(b) and (c) use S** because both are within-backbone comparisons, where S isolates prediction from the renderer. Decision B already fixes S for the adaptation curves. Astra adds 25% and 75% sensitivity thresholds, zero-cost crossings, and censoring of arenas that never cross. An absolute-ratio sensitivity must say whether it uses the geometric or the arithmetic mean, because the two differ.
- **Tuned decoder as the deployed renderer, stock kept as a paired sensitivity.** Tuning only on training-map frames is a fair zero-shot transfer of the whole package, provided no unseen frame selects a checkpoint or hyperparameter. `decoder_mse.sh` declares training ids `0:2000`, validation `6000:6100` and terminal-checkpoint selection. The artifact's actual provenance still needs checking.
- **What the ceiling shows.** Measure each decoder's reconstruction on every arena. If tuning improves reconstruction less on the unseen arenas, that gap is the renderer's transfer loss. A flat reconstruction score alone does not prove that the remaining degradation is prediction error, because the decoder can behave differently on predicted latents than on true ones.
- **"Ceiling" is shorthand, not a bound.** The raw residual equals the decoded-prediction residual plus the reconstruction residual. Their squared error carries a cross-term, so the two legs cannot be added or subtracted as independent error budgets.

## Table layout

Two group panels, training maps and unseen arenas, repeated once per latent space (SD1, SD3.5):

| Row | Latent | Matched target: stock / tuned | Raw target: stock / tuned |
|---|---|---|---|
| True-latent reconstruction | error 0 | identity | reconstruction reference |
| Copy-last latent | S = 0 | PSNR/LPIPS | PSNR/LPIPS |
| Model | S | PSNR/LPIPS and paired gains | PSNR/LPIPS |
| Raw persistence | n/a | n/a | one decoder-invariant reference |

Precedents: oracle/copy/model reporting in OccWorld (2311.16038) and DINO-Foresight (2412.11673); persistence baselines in PredNet (1605.08104); decoder fine-tuning in GameNGen (2408.14837).

Proposed paper paragraph (Astra): "We evaluate prediction, rendering, and final image quality on one fixed arena protocol. Persistence-normalized latent skill measures improvement before decoding; matched-decoder comparisons test its visible consequences; raw-frame scores measure delivered quality against raw persistence. Paired stock and training-map-tuned decoders expose rendering sensitivity. Frozen U-Net results show positive matched-decoder PSNR gains on every unseen arena, but substantially smaller gains than on training maps; the fresh evaluation tests whether decoder-free skill reproduces this pattern."

## Sunday rescore columns (per window, both decoders per latent space)

- Identity: corpus, map, episode, start/target ids, horizon, controls, decision phase, death/reset flags, sampling seed, split/checkpoint/encoder/decoder hashes, live or EMA, adaptation budget. Use an episode-balanced manifest, not the current global-window draw.
- Latent: model and copy latent MSE, ratio, log-ratio, static/invalid flags, directional results.
- Under the stock decoder and the terminal tuned decoder: MSE/PSNR/LPIPS for model vs decoded target, decoded copy vs decoded target, model vs raw target, decoded copy vs raw target, and decoded target vs raw target (the reconstruction).
- Decoder-free raw persistence MSE/PSNR/LPIPS, plus full-frame, gameplay-only and HUD variants.
- New columns: `copy_lpips_dec` and decoded-copy raw LPIPS.
- Save predicted latents. Apply the same record at every reported short horizon and every rollout step, including copy-seed rollout baselines. The existing decoder-launcher gate sets do not satisfy the single-dataset rule.

## Checks before trusting the reframed headline

1. The decoded target definition. Astra cited scorer commit `6a33311f`. **Rechecked:** at that commit `eval_tf.py:311` sets `pred_img, gt_img, last_img = dec(pred), dec(tgt), dec(ctx[:, -latent_channels:])`, and lines 319-320 score `psnr_dec` and `copy_psnr_dec` against `gt_img`. Both sides are therefore measured against the decoded true frame.
2. Temporal alignment, encoder identity, latent scaling, cropping (the 15-row strip) and clipping.
3. Duplicate or static windows, and episode weighting. The frozen unseen corpora hold 4, 10 or 20 episodes per map, against 25 on the training maps.
4. Whether decoder smoothing, HUD stability, motion or recording batch explains the gains. Compare gameplay-only against HUD results, and paired stock against tuned outputs.
5. Whether the decoded comparison flatters the model. The decoder may render smooth predicted latents more kindly than a sharp copied latent.

## Astra's recomputed numbers (U-Net 200k EMA, DDIM10, stock SD1 decoder, 256 windows per map)

| Quantity | Training maps 2-5 | Unseen arenas (13) |
|---|---:|---:|
| Raw PSNR gain, per-map range | +0.508 to +1.785 | -0.922 to +1.295 |
| Decoded PSNR gain, per-map range | +3.205 to +4.264 | +0.723 to +2.681 |
| Decoded gain, mean over maps | 3.598 | 1.509 |
| Reconstruction PSNR, mean over maps | 23.907 | 23.616 |
| Model raw LPIPS, range | 0.157 to 0.195 | 0.222 to 0.370 |
| Persistence raw LPIPS, range | 0.188 to 0.224 | 0.152 to 0.225 |

Further values: the decoded advantage falls 2.090 dB on average. Training-map raw LPIPS means are 0.1799 (model) against 0.2068 (persistence). Astra calls this numerically better, but the frozen means alone cannot show significance, and the brief called it a tie. Reconstruction LPIPS means are 0.0940 (training) and 0.0974 (unseen). Spearman correlation between unseen raw gain and persistence PSNR is -0.769. OLS slopes of model PSNR on persistence PSNR are 0.582 (training) and 0.639 (unseen). Astra warns that these correlations are not causal, partly because the gain itself subtracts persistence.

## Supervisor recheck (independent script over the same 17 files, taking `[key]["mean"]`)

- Raw gain ranges: training +0.5079 to +1.7853, unseen -0.9216 (map 8) to +1.2951 (map 15). **Match.**
- Decoded gain ranges: training +3.2050 to +4.2637, unseen +0.7233 (map 6) to +2.6811 (map 15). Means 3.5984 and 1.5085. **Match.**
- Reconstruction PSNR means 23.9070 / 23.6156; raw LPIPS means 0.1799 model / 0.2068 persistence on training maps; reconstruction LPIPS 0.0940 / 0.0974. **Match.**
- Unseen Spearman(raw gain, persistence PSNR) = -0.7692. **Matches Astra.** The brief's "-0.78" is slightly off and should read -0.77.
- Map 2 raw gain: 22.4286 - 21.9207 = 0.5079. **Match.**

## What Astra could not do

The frozen files carry no per-window rows, no copy-last latent error (the denominator of S), no fresh-set scores and no tuned-decoder outputs. So no uncertainty estimate was possible, and the latent-skill headline itself is unverified. No repository files were changed and no server was touched.
