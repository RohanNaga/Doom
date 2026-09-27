# The first adaptation results: three readings and the decision (2026-09-27, 01:00 EDT)

Sources: `opus-lora-results-view-2026-09-27.md`, `fable-lora-results-view-2026-09-27.md`, `astra-lora-results-view-2026-09-27.md` (thread 01a0e10b-8da2-7ee2-ac6e-9fd148b2c084); data `results/adapt/` (13 runs, `scores.jsonl`, per-window CSVs, `log.jsonl`); the corrected table `results/adapt/CURVES_13_arenas_2026-09-27.md`. All three read the curves independently; all three recomputed the numbers from the files.

## The corrected result (scene crop on both sides; home = 4.138 dB on the training maps' 512 validation windows with the same scorer)

- All 13 arenas gain, +0.72 to +2.39 dB (median +1.36); 88 to 98 percent of windows improve and all 8 held-out episodes improve on every arena; the top decile of windows carries only 18 to 30 percent of the gain. Latent skill rises on all 13 and decoded LPIPS falls on all 13, so the advantage is better prediction, not kinder rendering.
- 9 of 13 arenas cross their half-gap line by 4,000 updates, 5 of them by 500; arenas 8, 12, 16 and 1 are censored. One arena (9) ends above the home line; median A at 4,000 is 3.14 against a home of 4.14.
- About 69 percent of the gain lands by 250 updates, and the curves are still rising slowly at 4,000 (+0.09 dB from 2,000 to 4,000 on average, positive on 13 of 13 by the figure builder's recomputation, with 56 to 80 percent of windows still improving over that interval).
- The per-frame distance D predicts nothing (Spearman with A0 -0.25, A4000 +0.07, gain +0.45, cost -0.27). The zero-shot latent skill predicts the endpoint (+0.89 with A4000, -0.65 with cost). The far arena (7) crosses at 250.
- Held-out v-loss falls at every checkpoint on all 13 arenas and ends within 0.01 of the training loss; gradient clipping never fired: the adapter is under-fitting, not overfitting or short of data.
- The crossing budget is noisy where the curves are flat: resampling the held-out episodes moves it by a factor of 2 to 8 on five arenas; only arenas 7, 9, 10 and 11 cross at 250 in at least 99 percent of resamples. The paper reports A at fixed budgets and the area under the curve beside the crossing.

## Where the three agree

Keep rank 16, alpha 16, the fully trained parts and the source weights; the recipe is sound and the regime is under-fitting. Fix the home line to the same crop before any figure (done: 4.138). Do not reach for more data first. The base-SD 1.4 LoRA control belongs in October, one arena, labelled a floor (Fable: report it as the updates raw SD 1.4 needs to reach the four-map model's zero-shot A0, since a same-budget comparison is trivially won). Before the battery: the tuned-decoder rescore with the perceptual margin and the ceiling gap, the forgetting and control guards, a second seed on the boundary arenas, the EMA read. Turn gradient checkpointing off when it fits (peak 6.2 of 16 GB) for speed, once a fit check confirms the memory.

## Where they differ, and what is running to settle it

- **First recipe change:** Opus, lr 5e-4 (10x the pretraining rate, per Biderman and DiffFit); Fable, lr 3e-4 with the grid extended to 8,000 and early points 50/100/150 for cost resolution; Astra, the 8,000-update budget alone, everything else unchanged. Resolution: all three run tonight on arenas 12 (censored) and 16 (slowest crosser), against their seed-1 runs as the noise reference: lr 5e-4 at the 4,000 grid, lr 3e-4 at the 4,000 grid, and the base rate on the extended grid 0/50/100/150/250/500/1,000/2,000/4,000/8,000. Adopt a change if A at 4,000 rises on both arenas by more than the seed-to-seed spread and the guards do not worsen; if nothing moves, the tail is data-limited and the ladder becomes the priority.
- **Scope for Sep 30:** Opus, the 13 arenas at the settled recipe plus seed 1 on four arenas, the guards, and full fine-tuning on two arenas on an A6000; Fable, the same with the grid extended to 8,000; Astra, a guarded U-Net pilot with replication in October. Resolution: the 13-arena figure at the settled recipe with seed 1 on the four pilot arenas and the guards is the Sep 30 content; the recipe test decides whether the settled recipe is the current one or a higher rate; full fine-tuning on 12 and 7 Monday on Spiderman.

## Also running tonight (Superman, seven cards, `~/Doom/lora_battery2.sh` and the three test scripts, all done by about 05:00)

Seed 1 on arenas 8, 16, 6, 7; the data ladder k = 1, 2, 4, 16 on arenas 7 and 12; lr 5e-4 and lr 3e-4 on 12 and 16; the 8,000-update grid on 12 and 16. Scoring in the morning on the freed cards (stock decoder there; the tuned-decoder and raw-frame columns on Spiderman).

## What the paper says, as of tonight

With eight episodes and one percent of the pretraining budget, a rank-16 adapter on the frozen four-map model recovers a median +1.4 dB of decoded advantage over copy-last on arenas it never saw, closes half the gap to in-domain performance on 9 of 13, and improves every held-out episode of every arena; the gain is broad across windows and comes with better latent prediction and better perceptual quality. The per-frame distance predicts none of it; the model's own zero-shot latent skill on the arena predicts most of it. Numbers on the tuned decoder, with guards, follow the Sunday rescore.
