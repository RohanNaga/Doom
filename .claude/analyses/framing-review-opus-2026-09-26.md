# Framing review (Opus, independent), Sep 26 2026, evening

Reviewer stance: a professor in world models and generative video, formed before reading `astra-review-2026-09-26.md`, then compared (last section). Sources read per the brief: RESEARCH_CONTEXT.md section 0 and every entry from 09-25 23:15 to 09-26 17:10; dossier sections 0, 1, 4, 6, 7, 9, 10; the literature memo; `results/distance_study/figure_unet_h1_amended/stats.json`; `results/sd35_stability/collapse_rates_50k_130k.md`; the 70k strip; the layers brief. One new zero-GPU analysis was derived from the frozen per-map metrics files (`results/distance_study/scores/040-unet-nexttic_snap_0200000_ema_ddim10/*_h1/metrics.json`, fields `vae_psnr`, `vae_lpips`, `psnr_raw`, `lpips_raw`, `persist_psnr_raw`, `persist_lpips_raw`, `latent_mse`). Nothing was run on a server.

## The finding that changes the plan

The rendering leg of the proposed decomposition is already measured, and it is nearly zero. `vae_psnr` is the stock decoder's reconstruction of the true target latent against the raw frame, i.e. the rendering ceiling per map.

| U-Net 200k EMA, h1, stock decoder | Training maps (4) | Unseen maps (26) |
|---|---:|---:|
| Reconstruction ceiling PSNR / LPIPS | 23.91 / 0.094 | 23.71 / 0.097 |
| Model PSNR gain over raw persistence | +0.92 | -0.54 |
| Ceiling minus model PSNR (error beyond rendering) | 1.55 | 2.41 |
| Model LPIPS (persistence LPIPS) | 0.180 (0.207) | 0.326 (0.201) |
| Latent MSE | 0.168 | 0.236 |

Across the 30 maps the ceiling does not track distance (Spearman with D -0.12, p 0.53; LPIPS ceiling -0.09, p 0.65), while the error beyond the ceiling does (Spearman +0.47, p 0.009; partial on persistence +0.53) and model LPIPS does (+0.59, p 0.0006). All derived here from the files named above.

So the stock SD 1.x autoencoder renders unseen maps as well as training maps. The unseen-map loss lives in the predicted latent. A decoder tuned on an unseen map's own footage can therefore recover at most about 0.2 dB of genuine rendering loss. Anything beyond that would be the decoder learning to clean up this backbone's prediction errors on that map, which is a form of adaptation to the predictor, not rendering, and it would be confounded exactly where the paper wants a clean attribution.

A second fact from the same files: on three campaign maps the ceiling sits below raw persistence (map 19: 23.76 vs 24.64; map 25: 25.27 vs 25.76; map 26: 25.07 vs 27.23), and on map 24 it clears it by 0.14 dB. On those maps no model decoded through this autoencoder can beat copy-last in PSNR, however good its latent. Three to four of the "17 of 26 below persistence" maps are unwinnable by construction.

## (1) Is the frame right, and what is the cheap lever

The decomposition is the right instinct and the right paper shape. Three corrections.

- **Rendering is answered; cancel Sunday's four-map decoder adaptation.** Report the ceiling as a column and the paragraph above. Keep the training-map decoder tunes only if they are already running and cost nothing that PixArt's 30-map scoring needs; they are a secondary column at most.
- **"Physics transfers" must be narrowed to what the directional check measures: the sign and magnitude of camera yaw under a turn swap.** It does not measure forward motion, strafing, collisions, doors, lifts, height changes or enemy behaviour, and campaign maps contain map-specific dynamics that arenas lack. Say "the control-to-yaw response transfers", not "physics transfers". Per map, report `ref_frac` (the estimator's own validity on ground truth) and the number of eligible turn windows; static campaign footage with 1.0 to 1.4 lives per episode may have too few clean turns or textureless views, and a map with `ref_frac` below about 0.8 cannot be scored (VERIFY once the 30-map run lands).
- **"Appearance" is also too narrow a word for the latent leg.** D is computed on VAE latents, so it cannot separate texture novelty from geometry novelty. Call the leg "latent prediction error on unfamiliar footage" and let D describe it.

The cheap lever, if one is wanted, acts on the predicted latent, not the decoder. Inference-time options are weak here: observation guidance needs observation dropout in training (absent), context noise at inference already failed (dossier 5.5), and step count trades PSNR for LPIPS without touching the map gap. The lever that tests the hypothesis directly is a short fine-tune of a small part of the backbone on a few episodes of the unseen map (LoRA on attention, or the input projection plus normalisation/adaLN parameters), scored on held-out episodes of the same map, with the directional check as a guard. If the gain returns and yaw stays, the gap was learnable map-specific content. That is the October experiment (section 4), not a Sep 30 one.

On Keerthana's middle-layer idea: for a diffusion model, the more informative axis is the denoising timestep, not depth. Coarse layout and motion are decided at high noise, texture at low noise. A zero-training probe: in the directional check, apply the swapped control only in the first k of the 10 DDIM steps, then only in the last k. Where the turn gets decided tells you where "physics" lives. It fits the discussion or October, not the Sunday draft.

## (2) The strongest honest four-page paper for Sep 30

**Contribution sentence.** "On an open 30-map Doom benchmark, latent-diffusion world models trained on four maps beat copy-last-frame persistence on held-out episodes of those maps but not on maps they never saw; the loss sits in the predicted latents rather than the decoder, grows with a footage distance to the training maps, and leaves the turn response intact, and one-step fidelity does not predict closed-loop stability." The yaw clause is conditional on tonight's 30-map directional run; if it fails on unseen maps, the clause flips to "and the turn response degrades with it", which is also a result.

**Figures (three).**
1. Per-map decomposition, maps sorted by D: model gain over persistence, reconstruction-ceiling headroom (ceiling minus persistence), training maps filled, unseen open, one-tic and four-tic. This merges the persistence result and the distance result into one picture, and it shows the unwinnable maps honestly.
2. The distance scatter with the qualified statistics printed (pooled -0.734 [-0.84, -0.32]; with training indicator -0.558; with family -0.423, CI crossing zero), three rows if PixArt and SD 3.5 are scored by Monday night, else U-Net plus "replication pending". If page space forces a choice, fold this into figure 1 as a small inset.
3. The 70k strip, cropped to one window (map 5 shows both the blue and grey fixed points), beside the matched-window event counts.

**Tables (two).** Table 1: three rows at 200k EMA, h1 and h4 PSNR/LPIPS with gains, 256-tic minus copy-seed, directional `correct_frac`, and reconstruction ceiling per row (the 27.5 vs 23.5 dB autoencoder gap must sit beside SD 3.5's lead). Table 2: the decomposition by cluster (training / unseen arenas / campaign): ceiling, model, persistence, latent copy-last ratio, directional.

**Cut:** probe_v2 and the oscillation (one sentence at most), the sampler sweep (appendix), the gate-check table (appendix; one sentence in the text that the pre-declared gate failed), the post-training arms, the late-training trend, the middle-layer idea (one line of future work), training chronology, novelty claims about persistence itself.

## (3) Weekend GPU-hours, ranked by effect on the paper per hour

Throughput from the frozen U-Net metrics: about 7 to 14 windows/s at h1 and about 2 at h4, so one 30-map pass is roughly 15 min at h1 and 70 min at h4 of sampling plus loading (derived; SD 3.5 is slower, VERIFY).

1. **0 GPU-h: look at the campaign-map footage GIFs** (already rendered, `$D/tmp/videos/`). If the agent is stuck against walls on maps 19, 24 to 26 and 28, the paper must say so; it explains both the high persistence and the losses.
2. **About 0.5 GPU-h each: tonight's decoder-free latent ratio (MSE model / MSE copy-last latent) on 30 maps** and **the 30-map directional check with per-map `ref_frac`**. These are the two legs of the decomposition that are not yet measured. Highest value per hour.
3. **About 1.5 GPU-h: PixArt 200k EMA through the 30 maps at h1 and h4** as soon as it stops. Cross-backbone replication is the single largest gap in the distance claim.
4. **About 3 to 5 GPU-h (VERIFY): SD 3.5 200k EMA through the 30 maps, h1 first**, Monday after 00:30. h1 alone is enough for the draft; h4 if time allows.
5. **0 GPU-h: add the ceiling analysis above to `make_distance_figure.py` outputs** and a per-map "unwinnable" flag (ceiling below persistence).
6. **Only if 2 to 4 are done: a two-map adaptation pilot** (unseen arenas 16 and 17, which have 10 episodes each: fine-tune a LoRA or the input projection for about 2k updates on 5 episodes, score the other 5, directional as guard). About 1 to 2 GPU-h per map. Gives the October paper its first point; do not put it in the Sep 30 text unless both maps agree.

Drop: unseen-map decoder adaptation (about 4 GPU-h, answers a question the files already answer), the post-training arms, the 6,000-episode U-Net row.

## (4) What to defer to October, and what that paper claims

October claim: "Distance to the training footage predicts not only the zero-shot loss of a game world model on a new map, but also how much footage it takes to adapt, and the adaptation lives in a small part of the backbone while the control response is preserved." Experiments: adaptation curves (0, 1, 2, 5 episodes) on 8 to 10 unseen maps spanning D, small-part fine-tunes only (LoRA, input projection, adaLN), all three rows; the timestep-localisation probe of where the control acts; a directional test extended to forward motion and strafe; the inference-only channel-mean rescue and a structured-offset augmentation fine-tune for the absorbing states; more evaluation maps (n = 30 with three clusters is what caps the distance statistics today).

## (5) The hostile reviewer's first attack, and the experiment that blunts it

First attack: "Below persistence on unseen maps is an artifact of static footage plus a lossy autoencoder. On maps 19, 25 and 26 your own decoder cannot reconstruct the true frame as well as copying the last one, the agent appears stuck (1.0 to 1.4 lives per episode, persistence up to 27.2 dB), and the distance correlation is carried by the training/unseen/campaign split." The second half is already conceded in the amended assessment. The first half is new and currently undefended in the text.

The one experiment: **tonight's decoder-free latent ratio on all 30 maps.** It removes the autoencoder from both sides of the comparison. If the ratio exceeds 1 on most unseen maps and rises with D, the headline stands without the decoder; if it does not, the "below persistence" claim must be restated as a rendered-frame result with the ceiling shown. Report it beside the PSNR gain in figure 1, and flag the unwinnable maps.

## Comparison with Astra's review (read after the above)

Agree: persistence and transfer as the joint lead with the three rows as a systems layer; the qualified distance wording with the failed gate kept; stock decoders primary; paired stock/tuned decoding as a column, not a contribution; matched-window stability counts (live 23/256 vs EMA 9/256 on seed 0, 55k to 130k; 11/240 vs 8/240 without 70k; 4/96 each late); no post-training arms before Wednesday.

Differ or add:
- Astra's review predates the physics/appearance frame, so it never asked whether the rendering leg is open. The per-map ceiling shows it is closed, which kills the Sunday decoder adaptation that the frame spawned.
- Astra did not flag that the decoder ceiling falls below raw persistence on three campaign maps. That weakens the "17 of 26" count Astra calls the clearest finding; it stays clear in LPIPS (26 of 26, where the ceiling is 0.097 against persistence 0.201), so lead the persistence claim with LPIPS and the latent ratio, and give PSNR with the ceiling beside it.
- Astra puts the distance scatter as Figure 1. I would lead with the per-map decomposition sorted by D, because after the structural controls the scatter is qualified evidence and the per-map picture carries both findings.
- Astra's proposed inference-only channel-mean rescue is sound as a mechanism test; I agree it is discussion material for Sep 30.
