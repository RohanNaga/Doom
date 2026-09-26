# Opus view: one cohesive evaluation (2026-09-26)

Recomputed from `results/distance_study/scores/040-unet-nexttic_snap_0200000_ema_ddim10/*/metrics.json` (U-Net 200k EMA, stock decoder, 256 windows/map; scorer 6a33311, pixel code identical to HEAD). Unseen arenas = maps 1, 6 to 17; training = 2 to 5.

## One-line answers

1. **Primary quantity:** one skill score, 10·mean log10(copy error / model error) per window in dB; latent space is primary for (b) and (c), tuned-decoder raw PSNR plus LPIPS against raw persistence for (a), and the other spaces are columns of the same table.
2. **Tuned decoder:** it moves every decoded and raw column and the ceiling, never latent skill or raw persistence; use tuned for pixel columns and keep the stock ceiling and stock decoded skill as the map-agnostic control; the scene-only ceiling drop on unseen arenas is the rendering leg.
3. **Sunday rescore:** per window, errors as MSE (not only PSNR) for latent and all four decoders, full frame and scene crop, `copy_lpips_dec`, exact-duplicate flags, and the predicted latents saved, at h1 and h4.
4. **Story:** skill over copy-last shrinks off the training maps in every space; in MSE terms it stays positive at one tic; in perceptual terms it flips, and that flip is not the decoder's floor.
5. **Most likely wrong:** the "LPIPS is a decoder statement" line; the HUD carries a large share of the stock floor; "no sign change" holds at h1 only.

## Recomputed numbers (h1 unless stated)

| quantity | training (4) | unseen arenas (13) |
|---|---|---|
| decoded gain `psnr_dec − copy_psnr_dec` | +3.21 to +4.26, mean 3.60 | +0.72 to +2.68, mean 1.51 |
| same decoder, raw target `psnr_raw − copy_psnr_raw` | +1.90 to +2.43, mean 2.16 | +0.67 to +2.00, mean 1.13 |
| raw gain `psnr_raw − persist_psnr_raw` | +0.51 to +1.79, mean 0.92 | −0.92 to +1.30, mean 0.10 |
| stock ceiling `vae_psnr` | 23.38 to 24.74 | 22.69 to 24.63 |
| LPIPS model / persistence / ceiling (means) | 0.180 / 0.207 / 0.094 | 0.300 / 0.194 / 0.097 |
| LPIPS model − persistence | −0.056 to −0.006 (4/4 better) | +0.063 to +0.163 (13/13 worse) |
| h4 decoded gain | +3.22 to +4.98 | −0.76 to +1.93 (6, 7 negative) |
| h4 LPIPS model − persistence | −0.171 to −0.096 | +0.031 to +0.121 |

Within the 13 arenas, Spearman against persistence PSNR: raw gain −0.77, decoded gain −0.05, same-decoder raw-target gain −0.16. Model raw PSNR = 0.64 × persistence + 7.7 (r 0.92). The brief's PSNR ranges reproduce; the PSNR reframe holds at h1.

## Q5: checks that change the reading

**LPIPS is not floored by the decoder.** The stock decoder's own LPIPS (0.079 to 0.122) is far below persistence (0.152 to 0.225) on every arena, so there is room to beat copy-last. The model does on all four training maps and loses on all 13 arenas. Even against the decoded truth (floor removed), model `lpips_dec` (0.179 to 0.332) exceeds raw persistence LPIPS on 13 of 13 arenas. Likely a blurred average under uncertainty, which PSNR rewards and LPIPS penalises. `copy_lpips_dec` would close this; it is not in the files.

**The HUD is a large share of the stock floor.** The stock decoder renders the 32-row status bar at 17.7 to 18.1 dB (`hud_vae_psnr`); persistence copies it almost exactly (`persist_hud_psnr_raw` 88 to 96 dB, most windows at the clamp). From means, the HUD carries roughly 39 to 68 percent of the ceiling's MSE and 20 to 50 percent of the model's raw MSE. So the tuned ceiling (GameNGen 2408.14837 recipe) rises largely through a map-invariant region, and a full-frame ceiling understates the rendering leg. Report scene-only (rows 0 to 207) beside full frame.

**The 100 dB clamp.** `psnr()` clamps MSE at 1e-10, so an exact duplicate scores 100 dB; the HUD column shows it happens. For full frames, the SEM bound admits at least one clamped persistence window on campaign maps 19, 24, 26. `latent_mse_ratio` drops exact copies as NaN while PSNR columns keep them at 100 dB. One rule for all columns: flag, exclude everywhere, report the count.

**Decoded comparison is not flattering on its face.** The decoded copy scores 0.7 dB (training) and 0.8 dB (arenas) above raw persistence, so it is the harder reference. Absolute `psnr_dec` (3.4 and 2.2 dB above `psnr_raw`, shared decoder errors) is meaningless; only the paired difference is valid. The same-decoder raw-target gain avoids the decoded target and is still positive on 13 of 13 arenas.

**Window composition.** Arenas 1 and 9 to 15 have 4 episodes each, so the files' SEMs understate uncertainty; the fresh 8 × 32 set fixes it only with an episode bootstrap. Validity exclusions run 2 to 15 percent (map 1), report them.

## Q1 reasoning

(a) Raw frames are the only space common to SD 1 and SD 3.5 backbones; the stock ceiling is only 1.9 to 3.6 dB above persistence (floor-bound), the tuned one about 7 dB (April 29.1). S is comparable within a backbone only.
(b) Unseen: lead with S, corroborated by decoded-stock skill, because the stock decoder never saw Doom (ceiling means 23.91 vs 23.62). Tuned raw PSNR and LPIPS are the system columns. S and decoded skill share one functional form and should rank maps alike; if not, that is a finding.
(c) Adaptation: S (decision 7) with LPIPS against persistence as the guard, since that is the column that flips.

## Q2 reasoning

Moves: `vae_*`, `psnr_raw`, `lpips_raw`, `copy_psnr_raw`, `psnr_dec`, `copy_psnr_dec`, `lpips_dec`; decoded gain second-order. Invariant: `latent_mse`, `copy_latent_mse`, `latent_mse_ratio`, `persist_*`. A decoder tuned on the training-map frames the dynamics model saw, then frozen (terminal checkpoint, never selected: `finetune_decoder.py:316`), obeys the same zero-shot constraint. Fair renderer for a system claim, unfair instrument for a dynamics claim. Rendering leg = (tuned scene ceiling, training minus unseen) minus the same for stock

## Q3: per-window columns

- **Identity:** episode, map, start, target tic, action, `tics_since_decision`, validity reason, duplicate flags.
- **Latent:** `latent_mse`, `copy_latent_mse`.
- **Per decoder (SD1 stock/tuned, SD3.5 stock/tuned, each in its own space):** MSE of model, copy and ceiling against the decoded and the raw target, full and scene; LPIPS of model, copy (new `copy_lpips_dec`) and ceiling; HUD MSE.
- **Decoder-free:** persistence MSE and LPIPS, full and scene.
- **Saved:** predicted latents (fp16) per window, so a later decoder never resamples.

## Q4: story and table

We score each prediction by its skill over copy-last: 10·mean log10 of copy error over model error, in dB (PredNet 1605.08104 used copy-last as the reference). We read it in three spaces that peel off the renderer:
- latent, decoder-free, as DINO-Foresight 2412.11673 scores forecasts in feature space against copy-last and oracle rows (VERIFY);
- decoded, prediction and copy through one decoder, as OccWorld 2311.16038 reports forecasting beside its tokenizer's reconstruction bound (VERIFY);
- raw frames against persistence, with the decoder's ceiling beside it.

Skill is positive in every space on the training arenas and shrinks on unseen arenas. In MSE terms it stays positive at one tic; raw sign changes track motion and the ceiling. Perceptually the model loses to copy-last on every unseen arena.

Table layout: two groups (training, unseen arenas) with backbone as the row. Columns:
- S;
- decoded skill, stock and tuned;
- raw ΔPSNR vs persistence (tuned);
- LPIPS model / persistence (tuned).

A second small table holds the ceiling (stock and tuned, full and scene) and persistence per group. The brief's ceiling/copy/model rows leave empty latent cells.
