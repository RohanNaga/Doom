# Fable's view of the first LoRA adaptation results (2026-09-27)

Numbers recomputed from `results/adapt/*/scores.jsonl`, the per-window CSVs and `log.jsonl` (13 runs, grid complete, 256 windows each). Brief: `brief-lora-results-2026-09-27.md`.

## One-line answers

1. **Curves.** All 13 arenas gain (+0.72 to +2.39 dB, mean +1.42); 12 cross the half-gap line, 7 at the first grid point; 69 percent of the gain lands by 250 updates, but the curves still climb (+0.09 dB mean from 2,000 to 4,000). D predicts nothing (|rho| ≤ 0.45); the zero-shot latent skill predicts the endpoint (rho +0.89 with A4000, −0.69 with cost).
2. **Recipe.** Underfitting, not capacity: training loss equals held-out loss within 0.009 at 4,000 on every arena and A rises with it. Keep rank, alpha and parts; grid to 8,000 with 50/100/150 added; the one two-arena test is lr 3e-4.
3. **Before the battery.** Recompute the home line on the scene crop (it decides the crossing count), then forgetting guard, EMA read, directional guard, a second seed; ladder and full fine-tune are October.
4. **Base LoRA.** Not for Sep 30; one arena in October, reported as updates-to-reach-A0.
5. **Battery.** Sep 30: U-Net seed 0 to 8,000 with guards plus a second seed on four arenas (about 45 A4000-hours); October: seeds, ladder, full FT, two transformer backbones.

## 1. The curves

| arena | D | A0 | A250 | A4000 | line | cost | gain by 250 | +2k→4k | skill 0→4k | LPIPS 0→4k |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.115 | 1.84 | 2.48 | 2.85 | 2.72 | 2000 | 0.63 | +0.04 | 1.36→2.04 | 0.229→0.171 |
| 15 | 0.127 | 2.63 | 3.30 | 3.90 | 3.12 | 250 | 0.53 | +0.09 | 1.96→3.02 | 0.281→0.168 |
| 12 | 0.128 | 1.56 | 2.23 | 2.51 | 2.58 | cens. | 0.71 | +0.05 | 1.21→2.09 | 0.228→0.154 |
| 17 | 0.130 | 1.59 | 2.62 | 3.14 | 2.60 | 250 | 0.67 | +0.10 | 1.52→2.64 | 0.333→0.177 |
| 14 | 0.146 | 1.63 | 2.56 | 2.99 | 2.61 | 500 | 0.68 | +0.05 | 1.25→2.01 | 0.258→0.178 |
| 11 | 0.150 | 2.01 | 3.23 | 3.75 | 2.80 | 250 | 0.70 | +0.15 | 1.80→2.72 | 0.228→0.140 |
| 13 | 0.156 | 1.95 | 2.70 | 3.18 | 2.78 | 500 | 0.61 | +0.10 | 1.42→2.36 | 0.296→0.196 |
| 10 | 0.162 | 1.79 | 3.45 | 3.72 | 2.70 | 250 | 0.86 | +0.06 | 1.73→2.51 | 0.181→0.127 |
| 16 | 0.164 | 2.25 | 2.75 | 2.97 | 2.93 | 4000 | 0.69 | +0.10 | 1.68→2.44 | 0.334→0.170 |
| 1 | 0.179 | 1.66 | 2.28 | 2.71 | 2.63 | 4000 | 0.58 | +0.09 | 1.31→2.16 | 0.324→0.229 |
| 9 | 0.180 | 2.43 | 3.73 | 4.20 | 3.02 | 250 | 0.74 | +0.19 | 1.86→2.78 | 0.203→0.138 |
| 6 | 0.188 | 1.08 | 2.30 | 2.61 | 2.34 | 500 | 0.79 | +0.10 | 1.14→2.23 | 0.249→0.158 |
| 7 | 0.282 | 0.98 | 2.94 | 3.37 | 2.29 | 250 | 0.82 | +0.08 | 1.42→2.13 | 0.234→0.158 |

"gain by 250" is the fraction of the 0→4,000 gain. The table's values and costs reproduce; median A4000 3.14; four above 3.60.

**Shape.** 69 percent of the gain by 250 (53 to 86), 85 percent by 1,000, then about 0.1 dB per doubling that has not stopped: 2,000→4,000 exceeds +0.05 dB on 11 of 13, 56 to 80 percent of windows rise over that interval, 5 to 8 of 8 held-out episodes are up. The two decreases (−0.007, −0.005) are noise. Latent skill (+0.68 to +1.09 dB), decoded LPIPS (−0.05 to −0.16) and the arithmetic latent ratio move with A on every arena.

**Breadth.** 88 to 98 percent of windows improve 0→4,000; the top decile carries 18 to 30 percent of the total gain (10 would be uniform); the trimmed mean is within 0.16 dB of the mean; all 8 held-out episodes are positive on all 13 arenas (episode-bootstrap intervals at least 0.67 dB above zero). Gains sit where there is motion: on the half of windows with lower copy-last PSNR, 2.98 versus 1.80 dB on arena 7 and 1.42 versus 0.48 on arena 12.

**Ordering (n = 13; |rho| ≥ 0.56 for p < 0.05).** D: −0.25 with A0, +0.07 with A4000, +0.45 with gain, −0.14 with cost (the table's +0.12 differs in sign; both null). Motion and frozen raw persistence PSNR: nothing beyond |0.48|; motion tracks the LPIPS drop at +0.91. The zero-shot latent skill orders the endpoint: +0.89 with A4000, −0.69 with cost; A0 ranks A4000 at +0.58. Cost is decided in the first 250 updates: rho(cost, gain by 250 minus required gain) = −0.93. Arena 7 is not the rule's artefact: its required gain (1.31 dB) and its 250-update gain (1.96) are both the largest.

## 2. Is this the right recipe?

- **No overfitting.** Training loss and held-out v-loss agree within ±0.009 at 4,000 on all 13 arenas (3.3 passes over about 38,600 windows); over the 65 grid intervals ΔA moves against Δheld-out-loss at rho −0.65. Across arenas loss drop does not rank the gain (−0.40, n.s.).
- **4,000 is short for the endpoint** (still +0.1 dB per doubling) and **250 is coarse for the cost** (7 of 13 cross at the first point). Grid 0/50/100/150/250/500/1k/2k/4k/8k; a checkpoint costs about a minute to score; two arenas to 16,000.
- **8 episodes:** no sign of shortage; only the ladder separates a data-limited tail from an optimisation-limited one (AdaWorld flattens by 200 to 800 full-fine-tune steps, 2503.18938; we flatten at 250 and keep drifting).
- **Rank, alpha, MLP:** with train = held-out loss, capacity is not binding; DiffFit's LoRA failure (2304.06648) is already covered by the fully trained control path. Rank ablation stays October.
- **The one test: lr 3e-4**, alpha 16, on arena 12 (censored) and 16 (slowest), grid to 8,000. DiffFit found 10× the pretraining rate best; Vista used 5× (2405.17398); we run 2×. Prediction if optimisation-limited: A4000 up ≥ 0.2 dB and both cross by 500; if not, the ladder is next. About 4.6 A4000-hours. Alpha = 2r (2405.09673) scales only the LoRA part; lr also moves the fully trained parts.
- **EMA:** at 0.999 it lags about 1,000 updates; expect lower reads at 250 and 500. Score the saved tensors (5 minutes per run); keep live as the cost axis.
- **Compute.** 1.10 h per arena on an A4000 at 1.0 update/s, peak 6.2 of 16 GB with gradient checkpointing on: the card is not filled; checkpointing off should give about 1.3×.
- **Provenance flag.** `score_adapt.py` computes A on the scene crop (rows 0 to 207); the brief and table call the 3.60 home line "full frame". Scene A exceeds full-frame A by 0.12 to 0.57 dB here. If 3.60 is full-frame, the like-for-like reading (A_full against 3.60) gives 9 crossings and 1 arena above home, not 12 and 4. Recompute the home line on the scene crop before any figure.

## 3. Before the battery, in order

1. Scene-crop home line (CPU, minutes).
2. Forgetting guard at every checkpoint, all 13 (about 1.5 A4000-hours): the claim is "adapts without forgetting" (2405.09673).
3. EMA read (free) and directional guard at 0 and 4,000.
4. Sunday's raw-PSNR and margin-B columns: does the 13-of-13 perceptual loss close.
5. Second seed on 8, 12, 16, 7 (about 5 A4000-hours): late gains are 0.05 to 0.19 dB, so seed spread must be known before "still rising" is stated.
6. Ladder 1/2/4/8/16 on the same four (about 21) and full fine-tune on 12 and 7: October unless the lr test says data-limited.

With my own eyes: one panel per arena with A, skill, LPIPS and held-out loss on a log-step axis; the per-window gain histogram; decoded frames at 0 and 4,000 for arena 12's worst windows and arena 7's best.

## 4. Base LoRA

Raw SD 1.4 with an untrained control path starts far below copy-last; its curve measures how fast a conditioning pathway can be learned, not what the four-map pretraining adds, and a same-budget comparison is trivially won. The useful form is the updates the base needs to reach the four-map model's A0: one arena, about 3 A4000-hours plus a flag in `adapt_wm.py`. October, appendix.

## 5. The battery

**Sep 30 (U-Net, seed 0 done):** tonight the lr test (4.6 h); EMA and forgetting scoring of the 13 runs (2 h); all 13 extended to 8,000 with early points, checkpointing off (about 24 A4000-hours, 4 h wall on six cards); seed 1 on 8, 12, 16, 7 (5 h). Report per arena the curve, A at 4,000 and 8,000, the 50-percent crossing, area under the curve, forgetting.

**October:** seeds 1 and 2 on all 13 (about 70 A4000-hours at 8,000); ladder on four arenas (21); full fine-tune on 12 and 7 (the ceiling, AVID 2410.12822); PixArt and SD 3.5 adaptation on the 13 arenas (2 to 3× per run, A6000s); rank 4/16/64 on two arenas; the base-LoRA arena. Order: lr test and the free scorings first, then the 8,000 extension, then seeds, then the ladder.
