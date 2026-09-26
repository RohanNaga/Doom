# Opus view: pixel-level adaptation target (2026-09-26)

Numbers are recomputed from the 30 frozen `040-unet-...ddim10/*_h1/metrics.json` files (stock decoder, AlexNet LPIPS), with definitions from `eval_tf.py` lines 300 to 345. All four earlier objections reproduce. "Tracks persistence" is Spearman against raw persistence PSNR over the 13 unseen arenas. Notation: x is the raw target frame, x_last the raw last context frame, z the true latent, z_last the last context latent, ẑ the prediction, D the decoder. Scene-only means rows 0 to 207.

## Option 1: ceiling gap against the true frame

- **y per window:** PSNR(D(z), x) − PSNR(D(ẑ), x), which equals `vae_psnr − psnr_raw`. Scene-only. The per-map value is the mean over windows.
- **Target:** the training-map mean with the same decoder. On the stock decoder that is 1.55 dB (band 1.36 to 1.84).
- **In words:** the model sits as close to the best frame its decoder can render as it does at home.
- **Zero-shot:** all 13 arenas sit outside the band (1.96 to 4.00) and need 0.41 to 2.45 dB. No arena starts past the line.
- **HUD:** it cancels. `hud_psnr_raw` and `hud_vae_psnr` differ by less than 0.2 dB on every map.
- **Failure modes:**
  - It tracks persistence at −0.80: dynamic arenas leave larger gaps. Fix: also freeze a persistence-matched line, reweighting training windows to each arena's persistence distribution (`per_window.csv`).
  - PSNR rewards blur.
- **Decoder:** tuned SD 1 first, stock second. The per-arena ceiling absorbs what the tuned decoder loses off-map.

## Option 2: decoded skill over decoded copy-last

- **y per window:** PSNR(D(ẑ), D(z)) − PSNR(D(z_last), D(z)), which equals `psnr_dec − copy_psnr_dec`. This is Genie's ΔPSNR form (2402.15391) with PredNet's reference (1605.08104).
- **Target:** the training-map mean, 3.60 dB (band 3.21 to 4.26). That means removing 56% of copy-last's squared error.
- **In words:** the model removes as large a share of copy-last's error as it does at home.
- **Zero-shot:** 0.72 to 2.68 dB, so arenas need 0.92 to 2.88 dB.
- **Strength:** the only candidate that ignores static footage (−0.05).
- **Failure modes:**
  - The truth it scores against is D(z), not x, so it reads as a latent number in pixel clothing.
  - PSNR rewards blur.
  - Decoded copy-last reproduces the HUD, so use scene-only.

## Option 3: perceptual copy-last cut (Astra's cut, in LPIPS)

- **y per window:** LPIPS(D(ẑ), x) − LPIPS(x_last, x), scene-only.
- **Target:** the training-map mean, −0.027 (band −0.056 to −0.006). Zero ("ties copy-last") is drawn as a reference.
- **Why LPIPS:** the decoder's LPIPS floor (0.075 to 0.124) sits below persistence (0.15 to 0.26) on every map. In PSNR the decoder decides the cut in advance: the ceiling falls below persistence on maps 19, 25 and 26.
- **Zero-shot:** the model loses on 13 of 13 arenas (+0.063 to +0.163) and wins on 4 of 4 training maps. Arenas need 0.09 to 0.19. This panel shows the blur that PSNR hides.
- **Failure modes:**
  - It tracks persistence at −0.58.
  - Heavy censoring at 4,000 updates is likely.

## Cost, freezing and compute (all options)

- **Cost:** the first budget in {0, 250, 500, 1k, 2k, 4k} at which the per-arena mean reaches the line and stays there. An arena that never reaches it is right-censored at ">4,000".
- **Summary across arenas:** the fraction of the 13 that have reached the line at each budget, a Kaplan-Meier-style reach curve. For the distance correlation, interpolate budget on a log scale and tie censored arenas at the last rank.
- **Forgetting guard:** the training maps' y stays inside the frozen band.
- **Freezing:** freeze the formulas now. Each decoder's numeric line comes from step-0 base-model predictions on the training-map windows. That line is independent of any adapted score, so the tuned decoder's line can be frozen at 02:30 from saved latents, before the first adapted checkpoint is scored.
- **Compute:** DDIM-10 sampling of 13 × 6 × 256 windows dominates; decodes and LPIPS are small, and a second decoder needs no resampling.

## Recommendation

**Headline: Option 1**, scene-only, tuned SD 1 decoder; stock decoder and the persistence-matched line in the table.
- It is scored against the true frame.
- It is read through the decoder, but each arena's own decoder floor is removed.
- It is anchored to in-domain skill, and every arena starts outside the in-domain band.
- It follows OccWorld's ceiling, copy-last and model table (2311.16038) and GameNGen's decoder tuning (2408.14837). AdaWorld (2503.18938) plots raw PSNR against steps; the in-domain anchor is new.

**Second panel: Option 3.** It carries the result PSNR cannot: the model blurs off-map, and adaptation has to fix that.

**Option 2 goes in the table, not a panel.** It is the motion-free check on Option 1. If the two rank arena costs differently, motion is driving Option 1, and the paper says so.

**Latent number: drop it from the figures and keep one appendix column.** The column is latent MSE over copy-last latent MSE, used only for the within-backbone comparison of LoRA against the full fine-tune.
- Both methods share the decoder, so pixel metrics already rank them. The column is the only read independent of tonight's decoder tuning: if the stock and tuned decoders rank the two methods differently, it settles which ranking belongs to the dynamics model.
- It costs nothing and never crosses backbones (SD 1 and SD 3.5 latents are not comparable).
