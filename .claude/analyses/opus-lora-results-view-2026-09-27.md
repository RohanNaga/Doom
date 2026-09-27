# Opus view: the first 13 LoRA adaptation curves (2026-09-27)

Read: `results/adapt/*/scores.jsonl`, all 78 per-window CSVs (256 windows, 8 held-out episodes each), `log.jsonl`, `config.json`, the frozen distance table, and the home values in `results/distance_study/scores/040-unet-nexttic_snap_0200000_ema_ddim10/val_map0{2..5}_h1/metrics.json`. Bootstraps resample held-out episodes.

## One-line answers

1. Every arena gains 0.72 to 2.39 dB, a median 69% of it by 250 updates, then about +0.1 dB per doubling, still rising at 4,000. "12 of 13 cross" compares a scene-crop A with a full-frame home; like for like it is **9 of 13**. D orders nothing; zero-shot latent skill orders most of it.
2. No overfitting (held-out loss tracks training loss within 0.01 and falls on all 13). The adapter under-fits, so test a larger step, **lr 5e-4**, before more updates or episodes.
3. Fix the home line first. Then the tuned rescore, the forgetting guard, and seed 1 on boundary arenas.
4. Base-SD LoRA: October, one arena, as a floor. Step-0 A > 0 on 13 of 13 already carries the Sep 30 claim.
5. Sep 30: 13 U-Net arenas at the settled recipe, seed 1 on four arenas, the guards, full fine-tuning on two. The rest is October.

## 1. Curves

**Definition mismatch.** `heldout_A_stock` is the scene crop (rows 0 to 207; `score_adapt.py` OUTCOMES "A"). `heldout_A_full_stock` is the full frame. Home 3.60 is the full-frame mean of `psnr_dec − copy_psnr_dec` on maps 2 to 5 (3.228, 3.205, 4.264, 3.697; stock sd-vae-ft-mse, DDIM 10). Scene A exceeds full A by 0.12 to 0.57 dB, so the CURVES table flatters every crossing. The repo holds no scene-crop home.

| arena | D | A0 scene/full | A4000 scene/full | share by 250 | 2k→4k [episode CI] | cost mixed / full |
|---|---|---|---|---|---|---|
| 8 | .115 | 1.84/1.46 | 2.85/2.39 | .63 | +.04 [.01,.06] | 2000 / cens |
| 15 | .127 | 2.63/2.37 | 3.90/3.57 | .53 | +.09 [.03,.15] | 250 / 250 |
| 12 | .128 | 1.56/1.26 | 2.51/2.13 | .71 | +.05 [−.01,.10] | cens / cens |
| 17 | .130 | 1.59/1.41 | 3.14/2.92 | .67 | +.10 [.07,.12] | 250 / 500 |
| 14 | .146 | 1.63/1.38 | 2.99/2.63 | .68 | +.05 [.00,.10] | 500 / 2000 |
| 11 | .150 | 2.01/1.81 | 3.75/3.47 | .70 | +.15 [.12,.17] | 250 / 250 |
| 13 | .156 | 1.95/1.76 | 3.18/2.94 | .61 | +.10 [.06,.13] | 500 / 1000 |
| 10 | .162 | 1.79/1.58 | 3.72/3.21 | .86 | +.06 [.03,.09] | 250 / 250 |
| 16 | .164 | 2.25/1.94 | 2.97/2.66 | .69 | +.10 [.06,.14] | 4000 / cens |
| 1 | .179 | 1.66/1.27 | 2.71/2.32 | .58 | +.09 [.07,.12] | 4000 / cens |
| 9 | .180 | 2.43/2.10 | 4.20/3.63 | .74 | +.19 [.15,.24] | 250 / 250 |
| 6 | .188 | 1.08/0.92 | 2.61/2.38 | .79 | +.10 [.07,.12] | 500 / 2000 |
| 7 | .282 | 0.98/0.86 | 3.37/3.10 | .82 | +.08 [.05,.10] | 250 / 250 |

Above home at 4,000: 4 mixed, 1 full/full (arena 9, 3.63). Median A4000: 3.14 scene, 2.92 full. Two dips, both −0.01 dB. Latent skill rises (1.14–1.96 to 2.01–3.02 dB) and decoded LPIPS falls (0.181–0.334 to 0.127–0.229) at every step on every arena, so A is not rising by blur.

**The crossing is fragile.** With flat curves, resampling episodes moves the crossing 2 to 8 times on arenas 1, 6, 13, 14 and 17. Only 7, 9, 10 and 11 cross at 250 in 99% of resamples under both crops. Report A at fixed budgets and the area under the curve beside the crossing.

**Broad gains.** 88 to 98% of windows improve. All 8 held-out episodes improve on every arena (smallest +0.56 dB). The top 10% of windows carry 18 to 30% of the gain. Windows losing to copy-last fall from 3–27% to 1–7%.

**Ordering (Spearman, n = 13: p < 0.05 needs |ρ| ≥ 0.56; ρ = 0.45 has a 95% CI of −0.13 to 0.80).**
- D: A4000 +0.07, gain +0.45, cost −0.14 mixed and −0.20 full.
- Zero-shot latent skill: A4000 +0.89, cost −0.69 mixed and −0.71 full.
- A0: A4000 +0.58, gain −0.18.
- Motion: share by 250 −0.70, LPIPS drop +0.91. Persistence PSNR: gain −0.04.

The table's D-cost +0.12 and A0-cost +0.07 do not reproduce (I get −0.14 and −0.17, censored at 8,000). Adaptation lifts arenas by similar amounts and keeps the zero-shot order.

## 2. Recipe

- **No overfitting.** Held-out diffusion loss (256 held-out windows) falls at every checkpoint on 13 of 13, still dropping 0.0015 to 0.0062 from 2,000 to 4,000, and sits −0.006 to +0.009 from training loss at 4,000. ρ(held-out loss, A) = −1.00 on 11 arenas. Arithmetic and geometric latent ratios move together. That is after 3.3 passes over 38.6k windows.
- **Under-fitting.** Training loss ends at 0.15 to 0.20 (pretraining 0.15), gradient norm 0.048 to 0.060, no clipping. The adapter does not memorise its 8 episodes, so 16 episodes should not help. More updates buy about 0.1 dB per doubling.
- **Test lr 5e-4.** Apply it to all trained parameters with the same warmup. Biderman (2405.09673) puts LoRA's best rate about 10× full fine-tuning's, DiffFit (2304.06648) at 10× pretraining's, and our pretraining ran at 5e-5. Alpha = 2r would speed only the LoRA factors; MLP LoRA (Biderman §4.7) is the second test. Run arenas **12 and 16** (the flattest) plus the current recipe at seed 1 on both as the noise reference: 4 runs, 5 A4000-hours, 75 minutes on 4 cards. Adopt if A4000 rises on both by more than the seed spread.
- **Throughput.** Peak 6.17 of 16 GB with gradient checkpointing on, 1.0 update/s, 66 minutes per run plus 77 s scoring per checkpoint. Turn checkpointing off (speed-up VERIFY).

## 3. Before the battery (by value, then cost)

1. Scene-crop home, stock and tuned (`trainmap_A_*`): minutes; every crossing depends on it.
2. Tuned rescore with B and C: about 1.7 GPU-hours.
3. Forgetting guard at 0 and 4,000: about 2 A4000-hours (VERIFY window count); the LoRA-forgets-less contrast (Biderman; XEWorld 2608.05799).
4. Seed 1 on arenas 1, 8, 12, 16: 5 A4000-hours.
5. Directional guard at 4,000: cheap.
6. EMA scoring: 1.7 A4000-hours; at 0.999 it lags early, appendix only.
7. Data ladder: 20 A4000-hours, October.

To look at: A against log updates with both lines and crops; decoded frames at steps 0, 250 and 4,000 beside truth and copy-last on arenas 12 and 16 (flat) and 7 and 10 (fast).

## 4. Base SD 1.4 LoRA

It trains the control path, context projection and noise embedding from zero on a model that never saw Doom, so it mixes pretraining's value with a cold start, and its result is predictable. Step-0 A > 0 on 13 of 13 already shows the adapter builds on pretraining. One arena (7) in October, called a floor; the standard comparator is training from scratch (TD-MPC2, 2310.16828).

## 5. Battery

About 1.25 A4000-hours per U-Net LoRA run. Order: (1) home fix and rescore; (2) lr test, 5 h; (3) 13 arenas at the settled recipe, 16 h, 4 h wall on 6 cards (reuse seed 0 if lr fails); (4) guards, 3 h; (5) seed 1 on 1, 8, 12, 16, 5 h; (6) full fine-tuning on arenas 12 and 7 at 2e-5 on an A6000 (AdamW state for 860M parameters exceeds 16 GB; hours VERIFY). Steps 1 to 6 go into the Sep 30 draft. October: seeds on all 13, the ladder, PixArt and SD 3.5 rows, MLP LoRA, the base-SD floor.
