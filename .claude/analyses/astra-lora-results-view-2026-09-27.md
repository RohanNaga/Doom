1. Broad adaptation succeeds, but the mislabeled crop inflates recovery: 9/13 full-frame half-gap crossings, versus 12/13 in the supplied table.
2. Keep the recipe provisionally; first test only a longer, 8,000-update budget on arenas 12 and 16.
3. Prioritize matched-crop/raw evaluation and retention guards, then EMA, replication, data scaling, and full fine-tuning.
4. Base-SD LoRA belongs in October: it tests the complete pretraining initialization, not isolated transferable dynamics.
5. September needs a guarded U-Net pilot; October needs replicated adaptation, data ladders, and backbone/comparator checks.

**1. Curves and audit.** Recomputed all 78 CSVs, paired by episode/start, against every run’s JSONL/config; aggregate discrepancies are below 1e-12. `heldout_A_stock` is **scene-only**, confirmed by `score_adapt.py`; full-frame is `heldout_A_full_stock`. Home recomputed from `results/distance_study/scores/040-unet-nexttic_snap_0200000_ema_ddim10/val_map0[2-5]_h1/metrics.json` is 3.598405 dB, full-frame. Below, preserve the declared 3.60 threshold: S/F costs use (respective A₀+3.60)/2. Scene costs reproduce the table but mix crops and are provisional until scene-home is scored. Gain/window columns concern S. C means right-censored at 4,000.

|Arena|Scene A₀→A₄₀₀₀|Full A₀→A₄₀₀₀|Cost S/F|Gain by 250|ΔS 2k→4k|Latent skill₀→₄ₖ|LPIPS_dec₀→₄ₖ|Improved windows|ρ(loss,S)|
|---|---|---|---|---|---|---|---|---|---|
|1|1.66→2.71|1.27→2.32|4000/C|58%|0.092|1.31→2.16|0.324→0.229|249/256|-0.9|
|6|1.08→2.61|0.92→2.38|500/2000|79%|0.096|1.14→2.23|0.249→0.158|250/256|-0.9|
|7|0.98→3.37|0.86→3.10|250/250|82%|0.078|1.42→2.13|0.234→0.158|246/256|-0.9|
|8|1.84→2.85|1.46→2.39|2000/C|63%|0.036|1.36→2.04|0.229→0.171|230/256|-0.9|
|9|2.43→4.20|2.10→3.63|250/250|74%|0.193|1.86→2.78|0.203→0.138|249/256|-0.7|
|10|1.79→3.72|1.58→3.21|250/250|86%|0.063|1.73→2.51|0.181→0.127|250/256|-0.9|
|11|2.01→3.75|1.81→3.47|250/250|70%|0.147|1.80→2.72|0.228→0.140|245/256|-0.9|
|12|1.56→2.51|1.26→2.13|C/C|71%|0.046|1.21→2.09|0.228→0.154|225/256|-1.0|
|13|1.95→3.18|1.76→2.94|500/1000|61%|0.096|1.42→2.36|0.296→0.196|236/256|-0.9|
|14|1.63→2.99|1.38→2.63|500/2000|68%|0.050|1.25→2.01|0.258→0.178|240/256|-0.9|
|15|2.63→3.90|2.37→3.57|250/250|53%|0.093|1.96→3.02|0.281→0.168|245/256|-1.0|
|16|2.25→2.97|1.94→2.66|4000/C|69%|0.103|1.68→2.44|0.334→0.170|238/256|-0.9|
|17|1.59→3.14|1.41→2.92|250/500|67%|0.096|1.52→2.64|0.333→0.177|247/256|-1.0|

Full-frame non-crossers are 1/8/12/16; only 9 reaches home. Scene scores exceed 3.60 on 9/10/11/15; median endpoint is 3.136. Every arena gains mostly before 250; thereafter all rise slowly, especially 9/11; 8/12/14 approach plateaus. Arena 16 stalls around 500–2,000 then improves again. Paired episode bootstrap (20,000 draws, seed 20260927) gives positive late-gain 95% intervals everywhere except 12: +0.046 [−0.014,+0.105]. This is residual improvement, not evidence that doubling compute doubles gains.

Gains are broad: every arena improves in all eight episode means; window medians are +0.613–2.097 dB. The best 26 windows contribute only 19–31% of net gain. Crossings are noisy: arena 6 clears its scene half-gap at 500 by only 0.001 dB; episode intervals for threshold margins on 6/17/16 include zero.

Recomputed Spearman: D versus scene A₀/A₄ₖ/gain = −0.253/+0.071/+0.445; A₀ versus gain = −0.181. D–gain falls to +0.294 excluding arena 7; full-frame D–gain is +0.407. Gain versus fresh context-motion, fresh decoded persistence PSNR, and frozen raw persistence PSNR = −0.484/−0.154/−0.039. Frozen persistence uses the older recording set. The table’s artificial 8,000-for-censor coding gives D–cost −0.137 and A₀–cost −0.175, contradicting its printed correlations; do not interpret these imputations as survival analysis. With thirteen maps, ties, censoring, and multiple exploratory comparisons, there is no convincing distance–cost result.

**2. Recipe.** Table loss correlations use the preceding 250-update training-loss mean at each nonzero grid point (only five observations per correlation). All arenas show falling loss with rising held-out A; held-out v-loss and arithmetic latent ratios decrease at every checkpoint. Arena 9’s tiny intermediate A dip is insufficient evidence of overfitting. Eight episodes work, but sufficiency is untested: 128,000 sampled windows represent only 3.20–4.33 passes through eligible data.

First change: **4,000→8,000 updates**, otherwise identical, on plateauing 12 and delayed-response 16. Fresh reruns cost approximately 4.41 A4000-hours because the trainer prohibits resume. Score 4k/6k/8k, live/EMA, paired episode intervals, A with its matching home crop, raw perceptual margin, and both guards. Choose using validation episodes carved from unused adaptation data; these inspected held-out curves are exploratory. Retain 4k if additional gains are negligible or guards worsen.

AdaWorld’s short saturation (arXiv:2503.18938) does not establish our stopping budget; Vista and DiffFit use different tasks/budgets (2405.17398;2304.06648). Rank, MLP targets, alpha=2r, and higher learning rate could change capacity or speed; Biderman’s advice is from language models (2405.09673). Test these separately only after the extension; lowering LR or lengthening warmup has little support here. EMA deserves scoring, not presumed superiority. A two-arena 4k alternative costs about 2.21 A4000-hours plus scoring, subject to capacity-dependent throughput.

**3. Measurement priorities and costs.** Logged training costs total 14.34 A4000-hours, mean 1.103/run, with peak allocation 6.17 GB. Benchmark checkpointing off before further runs. Median sampling throughput is 9.60 frames/s; allocate **0.03 A4000-hours per 256-window evaluation** including loading/decoding, an estimate, not a measured end-to-end rate.

Priority order:

- Correct crops/home and add raw persistence, raw PSNR, ceiling, scene/full perceptual margins: 78 reads ≈2.34 hours, excluding raw-I/O uncertainty. Decoded LPIPS already beats decoded copy on all thirteen at 4k; this does not establish a raw-persistence win.
- Forgetting on 512 pooled training-map windows at every checkpoint: ≈4.68 hours; directional counterfactuals at every checkpoint: provisionally another 4.68. These are required to claim retained control/knowledge (XEWorld, 2608.05799).
- EMA at all checkpoints: ≈2.34 hours; second seed on pilot 8/16/6/7: 4.41 training hours plus evaluation.
- Nested 1/2/4/8/16 ladder on those four: sixteen additional seed-0 runs, ≈17.65 training hours; then full-FT on 12/16 to distinguish an adapter ceiling. Reserve 8 A6000-hours for those two comparators; benchmark first, since no comparator throughput is available.

Inspect episode-banded A/loss/LPIPS curves, per-arena guard tables, and fixed median/worst/turning/death windows showing raw truth, persistence, reconstruction, zero-shot, live, and EMA. Add short matched rollouts before claiming simulator improvement.

**4. Base LoRA.** October: one arena with identical episodes, sampler, adapter and budget, but SD1.4 initialization. Its untrained controls/context and objective mismatch make it a system-initialization control. Failure supports the value of four-map pretraining as a package; it cannot isolate geometry, control, or “physics.” Reserve one 4k run, ≈1.10 A4000-hours plus evaluation; a negative short-budget result is not an asymptotic limit.

**5. Battery.** September: existing 13 arenas×1 seed×U-Net, zero-shot/live/EMA comparisons; corrected metrics/guards and pilot four×second seed; extension diagnostic next. Keep adaptation preliminary and the existing three-backbone zero-shot table separate.

October: U-Net 13×2 seeds =26 runs, ≈28.68 A4000-hours total at 4k; pilot four×2 seeds×PixArt/SD3.5 =16 runs, reserve 64 A6000-hours. Full-FT four×2 seeds×U-Net =8 runs, reserve 32 A6000-hours. U-Net ladder adds 32 non-eight-episode runs across two seeds, ≈35.30 A4000-hours. Transformer/full-FT reservations assume four hours/run and require benchmarking; evaluations are additional. If 8k wins, double training budgets. Run metric repair/guards, EMA, extension, replication, ladder/full-FT, then backbone expansion.

All requested score/log/config mirrors were present. Raw scores, guards, EMA scores, and decoded image artifacts were unavailable; none was inferred or generated remotely.
