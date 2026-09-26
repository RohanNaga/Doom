# Decision: one cohesive evaluation (three views converged, 2026-09-26, 19:30 EDT)

Sources: `main-cohesion-view-2026-09-26.md`, `astra-cohesion-view-2026-09-26.md` (thread 01a0dff9-48f1-72c2-a680-61313c7ca444), `opus-cohesion-view-2026-09-26.md`; every number rechecked by at least two of the three against the frozen `*_h1/metrics.json` files.

## Where the three agree

1. **Two scales, two jobs.** Latent skill S = 10 · mean over windows of log10(copy-last latent MSE / model latent MSE), in dB, is the primary quantity for every comparison inside one backbone: the unseen-arena comparison and the adaptation curves (decision B). Pixels through the tuned decoder against raw persistence (PSNR and LPIPS) are the primary for the in-domain table that compares the three backbones, because SD 1 and SD 3.5 latents are different spaces and raw frames are the only space they share.
2. **The tuned decoder** (training-map frames only, frozen) moves every decoded and raw column and the ceiling; it cannot move the latent errors, their ratio, raw persistence, the directional check or the rollouts. Score both decoders on the same predictions, once; the tuned decoder renders the paper's pixel columns, the stock decoder stays beside them as the control that never saw Doom.
3. **The Sunday rescore saves per-window rows** so no table is rerun: latent MSE of model and copy-last; pixel MSE (not only PSNR) and LPIPS of model, copy-last and reconstruction under each decoder of the space, stock and tuned; full frame and scene-only (rows 0 to 207, HUD excluded); `copy_lpips_dec` (new); duplicate-window flags with one exclusion rule for every column; the window's episode, start tic, motion and phase; the predicted latents in fp16; at one tic and four tics; windows balanced per episode (32 per held-out episode); confidence intervals by bootstrap over episodes.
4. **The headline, corrected twice tonight.** The model's skill over copy-last shrinks off the training maps in every space. In squared error it stays positive at one tic on all 13 unseen arenas (decoded advantage +0.7 to +2.7 dB against +3.2 to +4.3 on the training maps); the raw-PSNR sign changes on unseen arenas track how static the footage is (Spearman -0.77 with persistence PSNR), not the map shift. Perceptually it flips: in LPIPS the model beats persistence on 4 of 4 training maps and loses on 13 of 13 unseen arenas, and still loses when scored against the decoded true frame, so that flip is not the decoder floor; the likely mechanism is blur under uncertainty, which PSNR rewards and LPIPS penalises. At four tics the decoded advantage also goes negative on arenas 6 and 7.
5. **The decoded comparison does not flatter the model** (Opus: the decoded copy scores 0.7 to 0.8 dB above raw persistence, so it is the harder reference; Astra's caution stays as a sentence, not a finding). **The ceiling is a reference row, not an error budget** (Astra: the raw error has a cross-term).
6. **The HUD** carries 39 to 68 percent of the stock ceiling's error and 20 to 50 percent of the model's raw error because persistence copies it exactly and the stock decoder renders it at about 18 dB. Scene-only columns are required beside full-frame ones; the tuned ceiling will rise mostly in the HUD, so the rendering leg is read on the scene crop.

## Where they differed, and the resolution

- Main had "LPIPS is a decoder statement"; Opus showed it is not (the stock decoder's own LPIPS is below persistence's). Resolved: the perceptual flip is real and goes in the paper as the second half of the headline.
- Main and Opus had the ceiling / copy-last / model row layout; Opus prefers rows per backbone within two groups and a separate reference table because the latent column has no ceiling row. Resolved: Opus's layout for the main table, the reference table beside it.
- Astra wanted PSNR and LPIPS both as primaries for the in-domain table; pinned to PSNR with LPIPS beside it, both against raw persistence.

## The paragraph the paper can use

A world model trained on four deathmatch arenas predicts the next latent better than copying the last one on every arena of the WAD: by 3 to 4 dB on its training maps and by 1 to 3 dB on the 13 arenas it never saw, so its advantage lives in the latents and shrinks off the training maps. Rendered through the decoder and scored perceptually, the picture is harsher: the model beats copy-last persistence on the training maps and loses to it on every unseen arena, because under uncertainty it predicts a blurred average. Neither effect is the decoder's reconstruction floor, which is flat across maps and sits mostly in the HUD; a decoder tuned on training-map frames raises that floor everywhere and does not close the latent gap. Raw-PSNR comparisons against persistence flip sign according to how static a map's footage is, and are reported with the reconstruction ceiling beside them rather than used to compare maps.

## What this changes in the pipeline

- Scorer additions before the Sunday rescore: `copy_lpips_dec`; scene-only metrics; duplicate flags and one exclusion rule; `--decoder` path so the tuned decoder scores the same predictions; save predicted latents fp16 and per-window rows always; four-tic rows. Worker task.
- The one page and section 0 updated tonight; the draft's section 4 follows the paragraph above.
