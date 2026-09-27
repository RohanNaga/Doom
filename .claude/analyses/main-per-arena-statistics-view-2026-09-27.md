# Per-arena adaptation: how to present it and what to claim (main session, Fable 5.1, 2026-09-27 10:40 EDT)

Written before reading the Opus and Astra views. Numbers from `tools/arena_stats.py` on the tuned rows (`paper/tables/tuned/adapt_summary.json`, 13 arenas, seed 0, live weights, home 5.06 dB, 4k grid; 10,000 arena-bootstrap draws).

## One-line answers

1. The unit is the arena; the population is "a new arena of this game"; the paper claims a distribution over 13 draws (what fraction of arenas, the interquartile mean, the median budget) with arena-bootstrap intervals, and per-arena facts only with their episode intervals. Never "the model adapts in 1k updates".
2. Body: (a) the survival curve, fraction of arenas past their half-gap line against updates, censored, with its arena-bootstrap band and the per-arena crossing ticks; (b) the interquartile-mean learning curve with a stratified arena band and the 13 per-arena lines faint behind it, home dashed. Appendix: the per-arena small multiples with episode bands, the performance profiles at 0 and 4k, the endpoint-against-deficit scatter moves to the distance figure.
3. Every headline number gets an arena-bootstrap interval; medians of budgets keep their censoring; the interquartile mean replaces the mean where a centre is quoted; the exact sentences are below and all are computable today.
4. The 8k grid gives the survival curve its early points (50, 100, 150) and its tail; the seeds give the per-arena band a second source of spread; the ladder is one appendix panel. Nothing else needs running before Tuesday.
5. The reviewer's objections (the line depends on A0, arenas share a WAD, one seed, coarse grid, ties at 250) are answered by the survival curve with its interval, by stating the population explicitly, by the seed panel, and by the 8k grid; the interval widths are the honest part of the story.

## 1. Unit, population, claim

Thirteen adapters are thirteen independent experiments on thirteen draws of a target; they share the source model and the WAD, which is exactly the population the paper is about ("a new arena of this game"), and the text must say so in the protocol sentence. Two things follow. First, the aggregate statistics resample arenas, not windows: the arena bootstrap is the only thing that says how far "9 of 13" would move on a fresh draw of arenas, and it moves a lot: the crossed fraction at 4k is 0.69 with a 95 percent arena-bootstrap interval of [0.46, 0.92]. Second, per-arena statements (arena 7 crosses at 250) carry the episode interval, because the only randomness inside an arena is which eight episodes were held out. The literature has settled this for exactly this structure: Agarwal et al. (2108.13264, rliable) for "one score per task, few tasks" recommend the interquartile mean with a stratified bootstrap over tasks, performance profiles as the full picture, and never the mean or the median alone; Taylor and Stone (JMLR 2009) name the three transfer metrics we use (time to threshold, jumpstart, area under the curve) and warn that the threshold choice is the researcher's; time-to-event analysis (Kaplan–Meier) is the standard way to report "time to threshold" when some units never reach it in the observation window, and it is how the censored arenas belong in the plot instead of a footnote.

The claim the paper is entitled to: "on 13 unseen arenas of one WAD, rank-16 adapters trained on 8 episodes close half of the arena's gap to home within 4k updates on 69 percent of arenas [46, 92], and the interquartile mean of the decoded advantage rises from 2.0 to 3.8 dB [3.5, 4.3]". Not entitled to: any statement about arenas of other games, or about the budget of one specific arena without its own interval.

## 2. The figures

**Figure 4a, survival.** x: updates on the log axis with a break for 0, grid 0/50/100/150/250/500/1k/2k/4k/8k once the rerun lands; y: fraction of arenas whose $A$ has reached its half-gap line, a step function, censored at the last grid point (the curve stops short of 1 and the caption says four arenas never cross). Band: the 95 percent arena-bootstrap interval of the fraction at each step (today: 0.31 at 250, 0.38 at 500, 0.54 at 1k, 0.69 at 2k and 4k). Ticks on the top edge: one per arena at its crossing step, labelled with the arena number, censored arenas as open ticks at the right edge. A second, thinner curve for the home-line crossing (1 of 13). Caption first sentence: "Given a new arena, how many updates until the adapter closes half of the gap to home: 5 of 13 by 500, 9 by 4k, 4 never within the grid."

**Figure 4b, the learning curve across arenas.** Same x; y: $A$ in dB. Thirteen per-arena lines in light grey behind; the interquartile mean across arenas as the heavy line with its stratified arena-bootstrap band (today: 2.03 [1.70, 2.36] at 0, 3.22 [2.99, 3.65] at 250, 3.79 [3.51, 4.25] at 4k); home dashed; the half-gap lines dropped from this panel (they belong to 4a). Caption first sentence: "The decoded advantage over copy-last, interquartile mean over arenas with a 95 percent arena-bootstrap band, rises 1.8 dB, most of it by 250 updates."

**Appendix.** The 13 small multiples (each arena's curve with its episode-bootstrap band, its half-gap line and crossing) so a reader can check any single arena; the performance profile at 0 and 4k ($P(A \ge \tau)$ against $\tau$: at 4k every arena is above 3.0 dB, 69 percent above 3.5, 31 percent above 4.0, one above home); the seed panel; the ladder. The endpoint-against-deficit scatter is the distance figure's second panel, not this figure's.

**Table 3** keeps its rows but every aggregate cell carries the arena-bootstrap interval in brackets and the median budget states its censoring: "1k [250, 2k]; 4 of 13 censored".

## 3. The statistics, sentence by sentence

- "9 of 13 cross by 4k" → "9 of 13 arenas (69 percent, arena-bootstrap 95 percent interval 46 to 92) close half of their gap by 4k; 4 by 250, 5 by 500." Recipe: per-arena crossing step from `A[step] >= (A0 + home)/2`, resample 13 arenas with replacement 10,000 times, percentiles of the fraction.
- "median budget 1k" → "median budget 1k updates (arena-bootstrap interval 250 to 2k; the median is censored in 7 percent of resamples)". Recipe: same resampling of the crossing steps with censored arenas as +infinity; report the interval of finite medians and the censored share.
- "median A rises 2.04 to 3.80" → "the interquartile mean of $A$ rises from 2.0 [1.7, 2.4] to 3.8 dB [3.5, 4.3]" (stratified bootstrap of the IQM across arenas; median 2.04 to 3.80 beside it in the appendix table).
- "11 of 13 tie or win in LPIPS" → "11 of 13 (arena-bootstrap 62 to 100 percent)"; recipe as for the crossings on the sign of $M$ at 4k.
- "one episode gives most of the gain" → "on the four ladder arenas, one episode keeps 75 to 101 percent of the sixteen-episode gain (stock decoder)"; n = 4, no interval, stated as a check.
- "rho 0.90" → "Spearman 0.90 (arena-bootstrap 0.68 to 0.97) with the endpoint"; recipe: resample arenas, recompute Spearman.
- Per-arena crossings keep the episode bootstrap already in the builder (`checked_interval`), and the appendix small multiples show it.

All of this runs in `tools/arena_stats.py` today; the builder gains a `--arena-bootstrap` pass that writes the intervals into `adapt_summary.json` and the two new panels.

## 4. What the reruns add

The 8k grid puts the first crossings where they happen (50 to 250) instead of piling five arenas on the first grid point, and shows the tail flat, so the survival curve's shape at both ends is real. The seed-1 runs on the 8k grid give the per-arena band a second component (seed on top of episodes) for four arenas; the appendix panel shows both. The ladder is one appendix panel. Nothing else to run.

## 5. Pitfalls and answers

- *The half-gap line depends on $A_0$.* State it, report fixed-budget $A$ and AUC beside the crossing (the IQM curve is that), and never correlate the budget with $A_0$.
- *Arenas share a WAD.* Say so in the population sentence; the arena bootstrap is over that population, and the October constructed shifts are the test beyond it.
- *One seed.* The seed panel and the statement that seed spread (0.02 to 0.06 dB) is a tenth of the between-arena spread.
- *Coarse grid, ties at 250.* The 8k grid's early points; until then the survival curve draws the tie as one step and the caption says it.
- *Small n.* Every interval is wide and shown; that is the finding, not a weakness to hide.

## 6. The other data figures

- **Rollouts (Figure 2).** GameNGen and DIAMOND show frames at fixed tics with no numbers; the reader wants the frames. Keep two controls (turn shows control, forward shows drift), eight tics (1, 4, 8, 12, 16, 20, 24, 32), rows truth / in-domain / unseen / unseen adapted / copy-last, per-tic scene PSNR as small grey numbers under the row (not a sparkline), tuned decoder.
- **Family step (3a).** Linear x for D (its range is 0.03 to 0.27), the training floor as a light band from 0.02 to 0.09, three backbones as three markers per map, no connecting lines, training maps shaded; XEWorld draws distance against error as a scatter with a fitted line, which is exactly what we must not draw. Caption: "a step, not a slope".
- **Per-arena zero-shot, stock and tuned (3b).** A slope chart per arena (stock to tuned as a short segment) is the paired-data form Weissgerber recommends over bars; sort arenas by zero-shot skill (the quantity that orders them), not by D; $A$ and $M$ as two rows of the same panel; the adapter's tuned value as a third marker on the $A$ row.
- **Training curves.** Linear x in thousands of updates (log hides the plateau), EMA solid and live dotted, persistence dashed, the read the paper uses marked with a filled dot; LPIPS in the appendix version only; DreamerV3 and GameNGen show loss or score against steps with no bands, we have one seed too, so no band.
- **Decoder pair.** Raw frame, stock, tuned in one row with the scene crop as a zoom inset on the region that differs most; GameNGen shows exactly this (its Figure on decoder fine-tuning shows HUD crops).
- **The distance null.** One panel (3a) and one sentence; the failed gate in the appendix table. Negative results in these papers are sentences, not figures; ours earns a panel only because the step itself is a finding.
- **Tables.** Reference rows first (persistence, decoder ceiling), then models; no bold on "best" where rows are within noise; intervals where a claim rests on them; the fewest columns: Table 1 params, one-tic PSNR/LPIPS, four-tic PSNR, rollout, directional; Table 2 ceiling, $A$, $M$, $G$, directional; Table 3 the five rows it has.
