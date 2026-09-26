# Brief: the adaptation-cost target, at the pixel level (2026-09-26, 19:35 EDT)

Three independent reviewers (main, Astra, Opus 5.5); the others' answers are withheld. Rohan will pick from the three memos. He asks for the **top three options, not a list**, each with what it measures, its failure modes on our data, and the one you recommend.

## Rohan's position

"The decoder should be involved. Everything should be at the pixel level. What is convincing is how good we can get to in-domain data. I kind of like Astra's alternative (a fixed cut relative to copy-last), but we need to expand on this and see what makes sense."

So: define the y axis of the adaptation curve and the target line **in pixel space**, through a decoder, and anchored to what the model achieves on its training maps.

## Facts (all in the frozen files; recompute what you use)

- Per map, one tic, U-Net 200k EMA, 256 windows: `results/distance_study/scores/040-unet-nexttic_snap_0200000_ema_ddim10/*_h1/metrics.json`. Keys: `psnr_raw` (decoded prediction vs true frame), `persist_psnr_raw` (raw copy-last frame vs true frame), `psnr_dec` and `copy_psnr_dec` (decoded prediction and decoded copy-last latent, both vs the decoded true frame), `vae_psnr` (reconstruction ceiling), `lpips_raw`, `persist_lpips_raw`, `lpips_dec`, `vae_lpips`, HUD variants. Definitions in `eval_tf.py` around lines 300 to 345.
- Training maps (val_map02..05): raw gain over persistence +0.51 to +1.79 dB; decoded gain +3.2 to +4.3; raw LPIPS model 0.16 to 0.20 vs persistence 0.19 to 0.22; ceiling 23.4 to 24.7 dB.
- 13 unseen arenas: raw gain -0.92 (arena 8) to +1.30 (arena 15), tracking persistence PSNR at Spearman -0.77 (static footage loses); decoded gain +0.72 to +2.68 on all 13; LPIPS model 0.25 to 0.37 vs persistence 0.18 to 0.23 (loses on 13 of 13, also against the decoded true frame); ceiling 22.7 to 24.6.
- The HUD (bottom 32 rows) carries 39 to 68 percent of the stock decoder's reconstruction error; persistence copies it exactly. Scene-only (rows 0 to 207) metrics are being added to the scorer tonight, with `copy_lpips_dec`, pixel MSEs, duplicate flags, a `--decoder` path so a tuned decoder scores the same predictions, and saved predicted latents.
- Two decoders are being fine-tuned tonight on training-map frames only (GameNGen recipe, MSE, frozen encoder): SD 1 (for the U-Net and PixArt rows) done about 02:30, SD 3.5 about 07:00 Sunday. In April the same recipe raised the reconstruction ceiling from 23.7 to 29.1 dB on training-map frames. Their ceilings on unseen arenas are unknown until they run.
- Adaptation study: LoRA on the frozen U-Net per unseen arena, 8 adaptation episodes for the step curve, checkpoints at 0, 250, 500, 1,000, 2,000, 4,000 updates, scored on 256 fixed held-out windows (32 per held-out episode); the training maps' validation windows scored at each checkpoint too (forgetting guard). Full fine-tune comparators on the same grid Monday. Zero-shot numbers, distance and curves all recomputed on a fresh 24-episode set per arena (recording now). The target must be frozen before the first adapted score exists.
- Earlier objections to pixel targets (verify them): half the training-map raw PSNR gain (+0.46 dB) is already exceeded at step 0 by arenas 15, 13, 16, 17; a raw LPIPS target of +0.013 is below persistence on every arena by 0.06 to 0.16; the raw gain tracks how static the footage is, not the map shift; the decoder's floor is flat across maps and mostly HUD.

## The question

Give the **three best pixel-level definitions** of (y axis, target line) for the adaptation curves, anchored to in-domain performance, through a decoder. For each: the exact formula per window and per map; which decoder (stock, tuned on training maps, or both); which reference (raw persistence, decoded copy-last, reconstruction ceiling, the training maps' own score); full frame or scene-only; what "cost" is (first budget reaching the line; censoring); what it means in words a reader gets in one sentence; the failure modes on our data (floors, static footage, HUD, decoder bias, blur rewarded by PSNR and punished by LPIPS); and its cost in compute. Then say which one you recommend as the headline, which as the second panel, and whether a latent-space number should still appear beside them for the within-backbone comparison (Rohan is sceptical; argue it or drop it). Keep the memo under 900 words. Cite by arXiv id where a precedent matters (Genie delta-PSNR 2402.15391, PredNet 1605.08104, OccWorld 2311.16038, GameNGen 2408.14837, AdaWorld 2503.18938).
