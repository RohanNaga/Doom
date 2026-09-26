# Astra's view: the adaptation-cost target at the pixel level (2026-09-26)

Codex thread `01a0e00b-2be6-7820-9c60-7c5ccf1909c0`, model gpt-6-astra, reasoning high, sandbox workspace-write, one turn (no follow-up needed: three options, one recommendation). Brief: `.claude/analyses/brief-cost-target-pixels-2026-09-26.md`. Astra recomputed from the frozen h1 files only; no server access, no repo edits.

**Recommendation.** Headline: option 1 (raw-frame LPIPS toward the training maps' quality, with a persistence margin). Second panel: option 2 (decoded-copy-relative PSNR skill). Option 3 (distance below reconstruction) is diagnostic only. Latent number: drop it from the main panels; keep it in an appendix as a diagnostic of decoder-induced change, not as adaptation success.

## Shared settings (all three options)

- Decoder: the SD 1 decoder tuned on training-map frames only, frozen through adaptation, as the primary. Rescore the same predicted latents with the stock decoder as a sensitivity (GameNGen 2408.14837 motivates the tuning, but a higher reconstruction PSNR does not guarantee a better LPIPS).
- Crop: scene-only (rows 0 to 207) as the primary, full frame as a sensitivity. Decode the whole image first, then crop.
- Notation: x = true raw frame, p = previous raw frame (persistence), a = D(predicted latent), r = D(true latent), c = D(previous latent). e(u,v) = max(MSE, 1e-10). L = AlexNet LPIPS. A bar is the mean over a map's 256 windows. "ID" is the equal-weight mean of the four training maps' validation sets under the **unadapted** model.
- Cost: the first observed budget in {0, 250, 500, 1,000, 2,000, 4,000} that crosses the line. Step 0 counts. Maps that never cross are right-censored above 4,000. No interpolation. Show reversals, episode-bootstrap CIs and forgetting separately. Freeze formulas, decoder checkpoints and calibrated lines before any adapted score exists (AdaWorld 2503.18938 for step-based adaptation curves).

## Option 1: raw perceptual quality with a persistence margin (headline)

- Formula: per window l_i = L(a_i, x_i), b_i = L(p_i, x_i). y = mean l over the map at budget k. Target T_m = min{ l_ID, mean b_m - g_ID }, with g_ID = b_ID - l_ID. Lower is better. If calibration gives no positive ID margin, the target is unsupported.
- Reference: raw persistence plus the training maps' own absolute LPIPS.
- Meaning: "reach training-map perceptual quality while beating copy-last by at least the margin the model has on training maps."
- Why: it expands the fixed cut relative to copy-last without letting a large gain on hard footage pass as good absolute quality. LPIPS exposes the averaging blur that PSNR rewards.
- Failure modes: LPIPS is texture-sensitive and only an approximation of perception; decoder bias enters directly. Static footage can make the margin unattainable, which should be reported as censoring, not fixed by weakening the line. The scene crop removes the HUD but also stops scoring HUD fidelity.
- Compute: one prediction decode plus one LPIPS per window; persistence scores cached. No reconstruction decode needed.

## Option 2: decoded-copy-relative pixel skill (second panel)

- Formula: s_i = 10 log10( e(c_i, r_i) / e(a_i, r_i) ). y = mean s over the map. Target T = s_ID (full ID skill). Higher is better. This is a fixed multiplicative cut against decoded copy-last, calibrated to in-domain skill.
- Reference: decoded copy-last, both scored against the decoded true frame.
- Meaning: "recover the training-map advantage over copying the previous frame, as seen through the same decoder."
- Failure modes: decoded targets hide detail the decoder cannot represent, so it measures agreement between rendered images, not raw-frame fidelity. Reconstruction error does not cancel algebraically. Near-duplicate windows make the ratio sensitive to the numerical floor (report it stratified by duplicate flag). Blur can raise this score while worsening LPIPS. Tuning the decoder changes the geometry of the comparison.
- Compute: one prediction decode per window; decoded targets and previous frames cached; MSE arithmetic is negligible. Precedents: PredNet 1605.08104 (copy-last comparison), Genie 2402.15391 (delta-PSNR, with a different counterfactual).

## Option 3: distance below the reconstruction reference (diagnostic)

- Formula: q_i = 10 log10( e(a_i, x_i) / e(r_i, x_i) ). y = mean q over the map. Target T = q_ID. Lower is better.
- Reference: the decoder's reconstruction of the true frame.
- Meaning: "get as close to the decoder's reconstruction quality as the model gets on training maps."
- Failure modes: a worse reconstruction makes the target easier and a better tuned decoder makes it harder. The reconstruction is not a mathematical ceiling (cross-terms; predictions can beat it). The HUD distorts full-frame normalization. With no persistence term, static footage and blur remain confounds.
- Compute: prediction decode plus cached target reconstructions and MSEs. OccWorld 2311.16038 motivates showing reconstruction, persistence and prediction together, not this exact normalization.

## Latent number

Astra drops it from the main panels and keeps it as an appendix diagnostic of decoder-induced change only. This conflicts with the frozen decision 7 in `RESEARCH_CONTEXT.md` (latent skill S as the primary, 50 percent of pooled training-map S). Rohan has to settle that.

## Numbers rechecked (stock decoder, full frame, frozen h1 means)

Recomputed with my own script over val_map02..05 (ID) and the 13 unseen arenas (map01, 06 to 17). All of Astra's values match to six decimals.

| Quantity (keys) | Astra | Recheck |
|---|---|---|
| l_ID, mean `lpips_raw` | 0.179946 | 0.179946 |
| g_ID, mean `persist_lpips_raw - lpips_raw` | 0.026839 | 0.026839 |
| s_ID, mean `psnr_dec - copy_psnr_dec` | 3.598405 dB (ratio 0.436676) | 3.598405 dB (0.436676) |
| unseen decoded gain range | 0.723 to 2.681 dB | 0.723 to 2.681 dB |
| q, `vae_psnr - psnr_raw`, ID / unseen | 1.548 / 2.479 dB | 1.548 / 2.479 dB |
| half the raw training gain | 0.461709 dB, crossed by 13, 15, 16, 17 | 0.461709, same four |
| Spearman(raw gain, `persist_psnr_raw`), unseen | -0.769231 | -0.769231 |
| unseen LPIPS loss vs persistence | 0.062984 to 0.163497 | same |

## Supervisor's observations (mine, not Astra's)

- Option 1's line is steep on today's numbers. Per-arena targets T_m run 0.125 (arena 10) to 0.180, against current `lpips_raw` of 0.222 to 0.370, so every arena must cut LPIPS by 0.07 to 0.19 before it crosses. The line is above the stock decoder's own LPIPS floor on every arena (`vae_lpips` 0.079 to 0.122), so crossing is possible in principle, but expect heavy censoring at 4,000 updates.
- A brief fact is slightly off. Unseen persistence LPIPS spans 0.152 (arena 10) to 0.225, not 0.18 to 0.23. Arena 10 is the one map where the persistence-margin branch, not the ID-quality branch, sets the line.
- Astra could not compute the tuned-decoder or scene-only thresholds, HUD error shares, duplicate sensitivity or any crossing costs from the aggregate files. Those need tonight's scorer outputs.
