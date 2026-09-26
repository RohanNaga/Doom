# Brief: one cohesive evaluation. Decoder tuning, latent skill, decoded comparison, raw persistence (2026-09-26, 19:30 EDT)

You are one of three independent reviewers (main session, Astra, Opus 5.5). Reach your own conclusion from the files; the others' answers are withheld from you. Rohan will read the three side by side and wants them to converge on one design.

## What changed tonight

The paper's headline was "the model beats copy-last persistence on the four training maps and loses to it on unseen maps". Verified tonight from the frozen one-tic scores (`results/distance_study/scores/040-unet-nexttic_snap_0200000_ema_ddim10/*_h1/metrics.json`, U-Net 200k EMA, 256 windows per map):

- Against raw-frame persistence (`psnr_raw` minus `persist_psnr_raw`): +0.5 to +1.8 dB on the training maps, -0.9 to +1.3 dB on the 13 unseen arenas, tracking each arena's persistence PSNR at Spearman -0.78 (static footage loses, violent footage wins), not the map shift.
- In the decoded comparison (`psnr_dec` minus `copy_psnr_dec`: the model's decoded prediction and the decoded copy-last latent, both scored against the decoded true frame, so the decoder's floor cancels): +3.2 to +4.3 dB on the training maps, +0.7 to +2.7 dB on every unseen arena. The advantage shrinks by about 2 dB; it does not change sign.
- The stock decoder's reconstruction ceiling (`vae_psnr`, decoding the true latent) is 22.7 to 24.7 dB and flat across maps; the model's raw PSNR sits about 0.6 × persistence + constant in every group, which is the floor showing through.
- LPIPS against raw frames: the model is at 0.16 to 0.20 on training maps against persistence 0.19 to 0.22 (a tie), and 0.25 to 0.37 against 0.18 to 0.23 on unseen arenas; the stock decoder's own reconstruction LPIPS is about 0.09. `copy_lpips_dec` is not in the frozen files (the scorer does not emit it yet).
- The decoder-free latent quantities exist per window in the current scorer (`latent_mse`, `copy_latent_mse`, `latent_mse_ratio` in `eval_tf.py` around line 337) and a 30-map rescore with them is queued; the frozen files predate them.

Tonight two decoders are being fine-tuned on training-map frames only (GameNGen recipe, MSE, frozen encoder, `finetune_decoder.py`, `scripts/spiderman/decoder_mse.sh`, validation on ids 6000:6099): one for the SD 1 latent space (U-Net and PixArt rows), one for SD 3.5. In the April project the same recipe raised the reconstruction ceiling from 23.7 to 29.1 dB on training-map frames. Rohan asks whether every pixel number should be recomputed with the tuned decoder, and how the tuned decoder, the stock decoder, the decoded comparison, the latent skill and the raw-persistence reference fold into one cohesive evaluation rather than four competing tables.

## The paper (base document: `RESEARCH_CONTEXT.md` section 0; the one page: `docs/lora_adaptation_design_2026-09-26.html`)

Single-dataset rule: the 17 deathmatch arenas (training maps 2 to 5 on 25 validation episodes each; the 13 unseen arenas on a fresh 24-episode set being recorded now, 16 adapt / 8 held out, 32 fixed windows per held-out episode); every number in the paper is recomputed on that set. Contents per Rohan: the three backbones' in-domain results, the evaluations, the rollouts, a rollout-strip figure, and the 13-arena zero-shot and LoRA adaptation curves. Adaptation cost is measured on a decoder-free latent skill (decision B on the one page). Deadline Thu Oct 1 07:59 EDT; draft to the advisor Sunday evening.

## Questions

1. **What is the primary quantity** for (a) the in-domain table, (b) the unseen-arena comparison, (c) the adaptation curves, and why. Options on the table: latent skill (decoder-free, model latent error against copy-last latent error, aggregated as mean log ratio); the decoded comparison (same decoder both sides); raw pixels against the true frame with the raw persistence row and the reconstruction ceiling beside it; each with the stock or the tuned decoder.
2. **What the tuned decoder changes and what it cannot change.** Which numbers move (raw PSNR, raw LPIPS, the ceiling, the decoded comparison?) and which are invariant (latent skill, latent ratio). Should the paper's pixel columns use the tuned decoder (trained on training maps only, frozen), the stock decoder, or both? Is a decoder tuned on training-map frames a fair renderer for unseen arenas, and what does its ceiling on unseen arenas tell us (the rendering leg of the decomposition)?
3. **What the Sunday rescore must compute** on the fresh set so that no table has to be rerun: list the columns (per window, so any aggregation can be recomputed) and the decoders.
4. **The cohesive story in one paragraph** the paper can use, and the one table layout that carries it (rows: reconstruction ceiling, copy-last, model; columns: latent, decoded-stock, decoded-tuned, raw; groups: training maps, unseen arenas), or a better layout if you have one. Cite precedents by arXiv id (OccWorld 2311.16038, DINO-Foresight 2412.11673, PredNet 1605.08104, GameNGen 2408.14837 for its decoder fine-tuning).
5. **What could still be wrong**: name the checks you would run on the frozen files before trusting the reframed headline (for example whether `psnr_dec` scores against the decoded true frame or the raw frame; whether the decoded comparison flatters the model; window composition).

Write a memo of at most 1,200 words with a one-line answer per question at the top. Recompute anything you rely on from the files; do not trust this brief's numbers.
