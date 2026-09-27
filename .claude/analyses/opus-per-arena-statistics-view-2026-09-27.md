# Per-arena statistics view (Opus 5.5, Sep 27 2026)

Brief: `brief-per-arena-statistics-2026-09-27.md`. All numbers recomputed from the 13 headline tuned rows (`results/adapt/unet200k_arenas13_map??_r16_k8_s0`, live, per-window files the rows name, duplicates dropped, each file's mean checked against its row to 1e-6), `results/fresh_rescore/*_tuned` for M and G, `results/home_unet_tuned` for home. Points reproduce `adapt_summary.json`. Bootstraps: 10,000 draws (two-level 5,000), percentile intervals.

## One-line answers

1. **Claim:** the unit is the arena (n = 13, the WAD's whole unseen set); per-arena statements rest on 8 episodes and one seed; headline claims are distribution statements (IQM, quartiles, fraction of arenas by budget) with arena-bootstrap intervals.
2. **Figures:** body = IQM learning curve with arena band and faint per-arena lines, plus a Kaplan–Meier "fraction of arenas past the half-gap line" curve; profile, small multiples and endpoint-vs-skill go to the appendix.
3. **Statistics:** the median budget is not identified (arena bootstrap: 250 to past 4k); two sentences are wrong as written: LPIPS is 11 wins and 2 ties, and S0 predicts the endpoint, not the gain.
4. **Runs:** seed 1 on the other nine arenas at 8k (about 22 GPU-hours) and a 13-read cross-arena control (under 1 GPU-hour).
5. **Pitfalls:** all four censored arenas miss by 0.08 to 0.15 dB; home's own ±0.3 dB interval moves the count from 9 to 13; crossings move under episode resampling on 7 arenas and under seed on 1 of 4.
6. **Other figures:** Fig 3b as a paired dot plot sorted by A0, not D; Fig 2a's y label is wrong; the distance null is a sentence plus an appendix panel; tables get reference rows and intervals, not bold.

## Computed values (13 arenas, tuned, scene crop)

| quantity | point | arena bootstrap | two-level (arena, then episode) | episode only |
|---|---|---|---|---|
| A0 IQM (median) | 2.03 (2.04) | [1.67, 2.38] | [1.68, 2.37] | [1.95, 2.12] |
| A4k IQM (median; IQR) | 3.79 (3.80; 3.44–4.31) | [3.52, 4.26] | [3.51, 4.24] | [3.71, 3.88] |
| gain IQM (median; range) | 1.87 (1.77; 1.11–2.79) | [1.60, 2.21] | [1.59, 2.20] | [1.81, 1.95] |
| gap closed at 4k, IQM (median; IQR) | 0.61 (0.58; 0.47–0.73) | [0.51, 0.74] | [0.51, 0.73] | [0.59, 0.63] |
| past half-gap by 4k / by 500 | 9 / 5 | 6–12 / 2–9 | 6–12 / 2–9 | 8–11 / 5–7 |
| median budget 1k | | 250 7%, 500 12%, 1k 42%, 2k 32%, >4k 7% | | 1k 58%, 2k 38% |
| Spearman S0–A4k | 0.90 (perm. p 1e-4) | [0.68, 0.97] | [0.59, 0.95] | [0.76, 0.92] |
| Spearman S0–budget (censored = 8k) | −0.61 | [−0.86, −0.11] | [−0.82, −0.04] | [−0.66, −0.38] |

- **KM fraction past the line** at 250 / 500 / 1k / 2k / 4k: 0.31 / 0.38 / 0.54 / 0.69 / 0.69. Greenwood 95% bands: [0.13, 0.63], [0.18, 0.69], [0.30, 0.81], [0.45, 0.91] and [0.45, 0.91]. All censoring happens at 4k, so KM equals the empirical fraction.
- **IQM curve** at 0 / 250 / 500 / 1k / 2k / 4k: 2.03, 3.22, 3.40, 3.50, 3.67, 3.79. Arena bands: [1.67, 2.38], [2.99, 3.66], [3.14, 3.88], [3.25, 3.96], [3.41, 4.12], [3.52, 4.26]. AUC IQM 3.55 [3.30, 4.01].
- **Profile and fixed threshold:** at 4k the fraction of arenas with A ≥ 3.0 / 3.5 / 4.0 / 5.0 dB is 1.00 / 0.69 / 0.31 / 0.08. A fixed 3.5 dB threshold also gives 9 of 13.
- **Censored arenas at 4k** sit below their line by 0.08 (arena 1), 0.11 (8), 0.10 (12) and 0.15 dB (16).
- **Home** is 5.06 dB [4.75, 5.39] (99 episodes). At 4.75, 13 of 13 cross by 4k and 7 by 500.
- **Crossing step under episode resampling:**
  - Stable (≥95% of draws) on arenas 7, 9, 10, 15 and 16.
  - Mostly censored on 1, 8 and 12 (11%, 5% and 14% cross at 4k).
  - Moves on 6 (1k 59%), 11 (250 47% / 500 52%), 13 (2k 76%), 14 (2k 54%, censored 10%) and 17 (1k 44% / 2k 42%).
- **Gain timing:** 69% of the gain lands by 250 (range 54–84%). From 2k to 4k A gains +0.05 to +0.31 dB, positive on 13 of 13.
- **M at 4k:**
  - Median −0.021. Eleven arenas have an episode CI below 0. Arenas 1 (+0.001 [−0.010, 0.011]) and 6 (+0.008 [−0.001, 0.016]) are ties.
  - Zero-shot, all 13 CIs lie above 0 (median +0.081). M falls on 13 of 13 (median −0.096).
  - Two-level: the count below 0 is 9–13, and the median M is [−0.041, −0.012].
  - Median G falls from 4.96 dB zero-shot to 3.44 dB at 4k.
- **What S0 carries:** Spearman of S0 with A0 is 0.78 and with the gain 0.18. The partial Spearman of S0 with A4k given A0 is 0.85. A0 against A4k is 0.61.
- **Ladder** (stock decoder):
  - One episode gives 0.91 / 1.01 / 0.75 / 0.94 of the 16-episode gain on arenas 7 / 8 / 12 / 16.
  - k1 minus k16, with paired CIs: −0.23 [−0.43, −0.03], +0.01 [−0.05, 0.07], −0.25 [−0.53, −0.03], −0.05 [−0.09, −0.01].
  - Arena 12's gap is 0.253 dB, so the diff's "within 0.25 dB" should read 0.26.
- **Seeds** (stock decoder): |s1 − s0| is 0.02–0.06 dB at 4k but reaches 0.16 dB at 1k (arena 8). Arena 6 crosses at 4k with seed 0 and is censored with seed 1.

**Recipe.** For each arena and step, read `heldout_per_window` from the run directory and drop the `dup_*` windows. Set A_w = `scene_psnr_dec_tuned − scene_copy_psnr_dec_tuned` and sum by `episode`, giving a 13 × 8 × 6 array of sums and counts. The three bootstraps differ in what they resample:
- **Arena:** draw 13 arenas.
- **Two-level:** draw arenas, then 8 episodes within each, using the same indices at every step so A0 and At stay paired.
- **Episode:** keep the arenas and resample only episodes.

Recompute the line, the crossings, the IQM (`trim_mean(x, 0.25)`) and the ranks inside every draw. The other columns:
- M = `scene_lpips_raw_tuned − scene_persist_lpips_raw`.
- S0 = the per-window mean of −10 log10(`latent_mse`/`copy_latent_mse`).

The scratchpad scripts (`stats.py`, `stats2.py`, `stats3.py`) import `make_adapt_figures.py`. They amount to about 150 lines to fold into the builder.

## 1. Unit of analysis and the claim

There are three levels of variation: arenas (13), episodes (8 × 32 windows) and seeds (1, or 2 on four arenas). For levels, the two-level interval is barely wider than the arena interval. Between-arena spread is therefore the story; episode noise matters only for threshold events.

rliable's stratified bootstrap (Agarwal et al., NeurIPS 2021, 2108.13264) resamples runs *within* tasks and holds the tasks fixed. It supports "on these 13 arenas", which is our episode-only column. "A new arena" requires resampling arenas, the hierarchical bootstrap of Saravanan, Berman and Sober (NBDT 2020, 2007.07797). The 13 are the WAD's whole census and share textures and bots, so the arena bootstrap is an idealisation to "arenas like these". Never generalise past one WAD.

Wording by claim type:
- **Typical arena:** "across the 13 unseen arenas the IQM of A rises from 2.0 to 3.8 dB (95% arena-bootstrap CI 3.5–4.3)".
- **Distribution:** "69% of arenas (95% CI 45–91%) close half their gap within 4k updates".
- **Per arena:** "arena 7 crosses at 250 in every episode resample".

Taylor and Stone (JMLR 10:1633–1685, 2009, Sec. 2, checked in the PDF) define jumpstart, asymptotic performance, total reward (AUC), transfer ratio and time to threshold. They warn that time to threshold needs a "potentially arbitrary" performance level. That supports making AUC and A at the budget primary and the cost secondary.

The world-model papers set no precedent here:
- XEWorld (2608.05799) reports each robot separately ("never average"). Its distance figure has five points with r on the panel and no intervals.
- AdaWorld (2503.18938) and Vista (2405.17398) report per dataset or pooled, with no per-domain adaptation curves.

We are setting the reporting standard, not following one.

## 2. The figures

Body, Figure 4: two panels on one shared x axis. The axis is log-scaled adapter updates with 0 on a broken stub, grid 0/50/100/150/250/500/1k/2k/4k/8k.

```
(a) A (dB)                                (b) fraction of arenas past half-gap line
 5 ─ ─ ─ ─ ─ ─ ─ ─ ─ home 5.06            1.0 ┤           ┌──── 9/13 at 4k
   │   grey: 13 arenas                         │      ┌────┘ ░ Greenwood band
   │   ████ IQM + arena band; black: full FT   │  ┌───┘  ┄┄ A ≥ 3.5 dB
 2 ┤                                      0.0 ┤──┘              × censored
   0 ‖ 50 250 1k 4k 8k                         0 ‖ 50 250 1k 4k 8k
```

- **(a) caption:** "Every unseen arena gains from rank-16 adapters, most of it within 250 updates: A against updates, interquartile mean over 13 arenas with a 95% arena-bootstrap band, each arena in grey." Directly label "home" and "IQM". Do not colour by D.
- **(b) caption:** "Fraction of the 13 arenas whose A has closed half its zero-shot gap to home (Kaplan–Meier; 95% Greenwood band; four censored at the last grid point); dashed: the fixed threshold A ≥ 3.5 dB." The dashed curve shows that the count does not hinge on the A0-dependent line.

Appendix:
- **(c)** The performance profile: fraction of arenas with A ≥ τ, one line per budget.
- **(d)** Thirteen small multiples with episode bands and the crossing.
- **(e)** Endpoint against S0 or Δ. It predicts the endpoint through arena difficulty, and its ρ with the gain is 0.18, so it belongs outside the adaptation panel. If Rohan keeps Figure 4b for Δ, (b) takes its place and Δ goes to the text.

## 3. Headline numbers

- **"9 of 13 by 4k":** use the arena bootstrap and KM. Censoring is the stop at the last grid point.
  - Sentence: "9 of 13 arenas (69%, 95% CI 45–91%) close half their gap within 4k updates; the other four end 0.08–0.15 dB short."
- **"Median budget 1k":** the median is not identified. It ranges from 250 to past 4k across arenas and is 1k or 2k under episode noise. Point to the curve instead.
  - Sentence: "half the arenas are past their line by 1k updates (Figure 4b)."
- **"2.04 to 3.80":** make the IQM the centre, with an interval, and add the closure fraction.
  - Sentence: "the IQM of A rises from 2.0 to 3.8 dB (CI 3.5–4.3); the median arena closes 58% of its gap (quartiles 47–73%)."
- **"11 of 13 tie or win":**
  - Sentence: "at 4k scene LPIPS is below raw persistence on 11 of 13 arenas (episode CI excludes zero) and indistinguishable on two; zero-shot it was above on all 13."
- **"One episode gives most of the gain":** stock decoder, four arenas, paired episode CI.
  - Sentence: "one episode yields 75–100% of the sixteen-episode gain, 0.01–0.25 dB below sixteen, on four arenas."
- **"ρ 0.90":**
  - Sentence: "S0 ranks A at 4k (Spearman 0.90, 95% CI 0.68–0.97, exploratory) but not the gain (0.18): adapters lift arenas by similar amounts and hard arenas stay hard."
  - The budget correlation (−0.61) has a two-level interval that reaches −0.04. Keep it out of the abstract.

## 4. What the 8k grid, seeds and ladder add; runs

- **8k grid:**
  - The 50/100/150 points break the four-way tie at 250, which is interval-censored at (0, 250]. Draw the KM steps at the grid points. Turnbull's estimator (JRSS-B 38:290–295, 1976) formalises this but changes nothing at this density.
  - Censoring moves to 8k. The arena 12 and 16 tests (+0.06 and +0.02 dB from 4k to 8k) predict the censored four stay censored; say so.
- **Seeds:** four arenas give a seed component. Seed 1 on all 13 at 8k lets the bootstrap resample seeds within arenas, which rliable calls runs, and guards against the arena-6 flip. Cost: 2.2 GPU-hours training per arena plus scoring, about 22 GPU-hours for nine arenas, about 3.5 h on seven cards.
- **Ladder:** one appendix panel of A at 4k against episodes (log2 x), four lines, with the k1-minus-k16 interval printed.
- **Cross-arena control:** score each arena's 4k adapter on one other arena, 13 reads, about 40 GPU-minutes (VERIFY per-read time).
  - If the off-arena gain nears the on-arena gain, the adapter learned the fresh recording protocol or generic arena statistics rather than the arena.
  - Reviewers will ask, and no current file answers it.

## 5. Pitfalls in Figure 3 and Table 3

1. **The line uses A0, so A0 noise moves the target.** The two-level bootstrap recomputes the line in every draw. The fixed 3.5 dB threshold gives the same 9 of 13. Report closure as a continuous number beside the crossing.
2. **Home has its own error.** At the low end of home's interval all 13 cross. State home's CI, or resample home episodes in each draw.
3. **Near-miss censoring.** The four censored arenas miss by 0.08–0.15 dB, so 9 versus 13 is a threshold artefact.
4. **Dependent arenas.** All come from one WAD with one bot set; write "13 arenas of one WAD".
5. **One seed, crossings that flip.** Seven arenas move under episode resampling and arena 6 flips under seed. Per-arena costs need the episode-stability share.
6. **Coarse grid.** Handled by the 8k grid, as above.
7. **Colour by D and the D-tercile figure.** D has |ρ| ≤ 0.46 with every outcome, so neither should encode anything. Drop the terciles and draw the arenas in grey.
8. **Table 3.** Replace the median row with IQM [CI] and a row "past line by 4k: 9/13 [45–91%]". Move the per-arena costs to the appendix.

## 6. Every data figure

- **Figure 2 rollouts:**
  - GameNGen (2408.14837) shows frame columns with rows of context, truth and prediction (Figs 8–11), and every 10th of 50 frames for drift (Fig. 4). DIAMOND (2405.12399) shows consecutive frames per condition. Genie (2402.15391) shows rows per prompt. Oasis has only carousels (oasis-model.github.io).
  - None prints per-frame metrics under the frames.
  - Design: rows truth, in-domain, unseen zero-shot, unseen adapted, copy-last; columns at tics 1, 2, 4, 8, 16, 32. Add a narrow right-hand sparkline of scene PSNR against tic per model row, with copy-last dashed.
  - Controls: use turn-left alone unless forward carries a claim.
- **3a family step:**
  - Linear D axis with the floor as a grey band (0.02–0.09). Training maps filled, arenas open, backbones dodged horizontally, no connecting lines and no fit line: the claim is a step.
  - XEWorld prints r on a five-point panel (verified). Its fit line I could not confirm (VERIFY).
- **3b stock and tuned:**
  - Weissgerber et al. (PLOS Biol 13:e1002128, 2015) say to show individual points and make the pairing visible.
  - Use a dot plot with a stock-to-tuned segment per arena, A and M side by side, sorted by tuned A0. Sorting by D implies an order that does not exist.
  - Put the U-Net in the body and PixArt and SD 3.5 in the appendix.
  - The current Fig 2a's y label reads A0 while it also plots the 4k adapter; relabel it or split zero-shot from adapted.
- **Training curves:**
  - Linear x (0–200k), EMA solid and live thin, persistence labelled at 21.57, the 200k read marked with its value.
  - LPIPS goes in a second panel on the same x, not on a twin axis.
  - GameNGen Fig. 13 shows PSNR against steps without bands. DreamerV3 (2301.04104) puts per-task curves in the supplement; its axis and band details I could not open (VERIFY).
- **Decoder pair:** follow GameNGen Fig. 12 (frozen decoder, tuned decoder, truth, zoomed crops). Show raw, stock and tuned full frames plus one 3× scene inset with per-crop PSNR and LPIPS.
- **Distance null:**
  - In the body, a sentence and a Table 2 footnote with ρ and its CI.
  - In the appendix, one panel of D against A0 and A4k with ρ [arena CI] and the failed gate.
  - None of the reference papers draws a null.
- **Tables 1–3:**
  - Copy-last and the decoder ceiling as reference rows (the OccWorld and DINO-Foresight form).
  - Episode-bootstrap ± per cell, as GameNGen gives ± per cell. Arena-bootstrap CIs on aggregate rows, as DIAMOND reports stratified-bootstrap aggregates.
  - Bold only where intervals separate the rows; no stars.
  - Table 3 keeps only updates, IQM A4k [CI], closure, the past-line fraction, forgetting and GPU-hours.

## Citations

Opened at the primary source:
- Agarwal et al. (ar5iv: IQM, stratified bootstrap and profile definitions)
- Taylor and Stone (JMLR PDF)
- XEWorld (HTML)
- AdaWorld, Vista, GameNGen and Genie
- DIAMOND (stratified-bootstrap aggregates; profiles in Fig. 7)
- Weissgerber
- the Oasis page

Bibliographic record only:
- Kaplan and Meier (JASA 53:457–481, 1958)
- Turnbull (1976)
- Saravanan et al. (2020)
