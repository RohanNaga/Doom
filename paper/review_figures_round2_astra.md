# Figure review, round 2 — Astra

Independent review of commit `4e0223f51104e4f312f124d547b65ae95ac83a2a`, 2026-09-27. Steps 1 and 2 below were written before opening `paper/review_figures_round2_fable.md`. Comments are proposed edits for the joint review, not changes made to the figures.

## Scope and checks

Read the figure standard, the per-arena statistics decision, the metrics/axis survey, Sunday-diff items 13–14, and both round-one reviews. The adopted statistics decision supersedes the old Figure 4 endpoint-scatter specification; the terminology decisions supersede “copy-last,” “home,” and “live” as printed names. The M sign remains a choice, not an already-approved change. I have not reinstated obsolete requirements for 8k or full-fine-tune curves: the supplied adaptation summary ends at 4k.

Rendered all seven requested PDFs with `pdftoppm -r 200 -png` and viewed the PNGs. Scratch files are exclusively in `/tmp/astra_round2_png/`. Also viewed greyscale renders of Figure 4, paired zero-shot, family-step, and training curves. Inspected drawing code, PDF text, `pdfinfo`, `pdffonts`, the three requested numerical sources, rollout composition sidecar, and training-curve inputs/sidecar. No figure-generating script was executed.

| Asset | Native PDF size (inches) | Placement finding |
|---|---:|---|
| Figure 2 rollouts | 5.5 × 1.6565 | Within 1.75-inch ceiling; actual frames are only 0.2877 inches wide. |
| Figure 4 adaptation | 5.5 × 1.75 | 0.15 inches above the standard's ceiling; disclosed by the lead in round one. |
| Figure 4a gap-share alternative | 2.75 × 1.75 | Alternative standalone panel, not a full Figure 4. |
| Figure 3b paired zero-shot | 3.3 × 1.9 | Larger than the standard's 2.4-inch share of a 1.8-inch-high Figure 3. |
| Family-step asset | 5.5 × 1.8 | Full-width export, not the standard's 2.0-inch Figure 3a share. |
| Deficit asset | 2.7 × 1.6 | Appropriate as an appendix panel at native size. |
| Figure 1b training curves | 1.5 × 1.4 | Correct revised standalone width; missing panel letter. |

All seven have embedded Arial TrueType fonts and no Type 3 fonts. There are no empty statistical panels, boxed legends, grids, or decorative fills; grey reference bands and statistical uncertainty bands carry data. D is not encoded by colour. The 3.0/3.2-point markers in paired/deficit plots violate the prescribed 3.5–4 points despite satisfying the font-floor guard. Several annotations use the permitted 6-point minimum rather than the preferred 6.5 points.

`main.tex` and `FIGURES.md` still describe older assets and captions. In particular, the supplied family-step and paired plots cannot simply be scaled into the old Figure 3 shares: their 6.5-point labels would become about 2.36 and 4.73 points. Final paper placement, the 100%-zoom assembled-paper check, and a physical print are **not verified**. The untracked `main.pdf` is not a frozen assembled version of this set. Caption requests below are edits for the paper owner, not claims that these captions already accompany the assets.

### Numerical audit

| Check | Verified value / implication |
|---|---|
| Figure 4a IQM at 0, 250, 4k | 2.031664, 3.219360, 3.786261 dB. Endpoint nested 95% interval [3.515161, 4.266438]. Plot agrees. |
| Training reference | Adaptation summary: 5.059251 dB, episode interval [4.753803, 5.379118], 512 windows representing 99 episodes. |
| Figure 4b attainment | At 0/250/500/1k/2k/4k: 0/4/5/7/9/9 of 13; risk counts immediately before read: 13/13/9/8/6/4. Endpoint interval [0.461538, 0.923077]. |
| Rugs | 250: 7,9,10,11; 500: 15; 1k: 6,17; 2k: 13,14; censored at 4k: 1,8,12,16. All positions/IDs agree. |
| Fixed threshold | 3.5-dB attainment differs from half-gap attainment only at 1k: 6/13 versus 7/13. Its interval differs at other budgets too; the extra dotted bounds are real, not redundant decoration. |
| Gap-share alternative | IQM at 250/1k/4k: 0.406734 / 0.514169 / 0.606694; endpoint interval [0.514034, 0.733594]. Denominators are positive, 2.180674–3.990322 dB. This is the IQM of individual shares, not a ratio of IQMs. |
| Paired U-Net, arena 7 | Stock → fine-tuned: A 0.978914 → 1.070758; current-sign M 0.087809 → 0.060474. |
| Paired U-Net, arena 9 | A 2.433864 → 2.877794; M 0.054946 → 0.026294. |
| Paired U-Net, arena 12 | A 1.560292 → 1.789841; M 0.096060 → 0.068503. Positive current M is worse than raw persistence. |
| Paired provenance | U-Net/PixArt have stock/fine-tuned pairs; SD 3.5 has only stock. The high diamonds are 4k adapters, not zero-shot models. The paired training U-Net A is 5.059690, slightly different from adaptation's 5.059251; these separate reads agree to two decimals and must retain their own provenance. |
| Family-step medians, training → unseen | U-Net 2.846638 → 1.423416; PixArt 2.825988 → 1.514359; SD 3.5 3.837780 → 1.463931 dB. All six horizontal segments agree. |
| Family-step examples | Map 2: D 0.064858, U-Net S0 2.987836; arena 7: D 0.270254, S0 1.423416; arena 12: D 0.137594, S0 1.206084. Floor band [0.022325, 0.093373] is an empirical min–max range, not a 95% interval. |
| Deficit examples, (Δ, A4k) | Arena 10 (0.924276, 4.823288); 9 (1.148027, 5.109124); 7 (1.531769, 3.863947); 12 (1.856923, 3.328473). Positions agree. Across 13 arenas, Spearman ρ = −0.895604; the JSON gives a p-value, not a correlation interval. |
| Training curves | Persistence 21.565691 dB; endpoint raw gains U-Net 200k +0.923128, PixArt 200k +0.913920, SD 3.5 joined 155k +1.746058, isolated provisional 170k +1.791102. These are full-frame raw-reference gains, not A. |
| Rollout numbers | All 36 printed scene-PSNR numbers agree with the manifest to their displayed one decimal. Examples: training left tic 1 28.896 → 28.9; arena 7 adapted forward tic 8 23.951 → 24.0; adapted left tic 32 12.983 → 13.0. |

For each of the four displayed rollout windows, the true scene image equals rows 0–207 of its full-frame source. Every model tic-0 image equals the corresponding **decoded** context image, pixel-for-pixel, but differs from the **raw** context image. This is a presentation/reference distinction, not evidence of a wrong source tic. The underlying recorder frame/latent archive is not locally verified; the manifest and exported images establish the available provenance.

Displayed windows: map 2 left, episode 6000, start row 837, last-context game tic 868; map 2 forward, episode 6000, start row 225, tic 256; arena 7 left, episode 41, start row 269, tic 300; arena 7 forward, episode 41, start row 209, tic 240. `start_row` is the start of the 32-frame context, not the last-context tic.

### Round-one disposition

Figure 4's censor positions, separate per-arena rug ticks, fully labelled risk budgets, 3.75-point IQM diamonds, right-end persistence label, and named arena endpoints are applied. Fixed-threshold bounds and a complete dashed curve are present, but their visual identification still needs work. The IQM expansion belongs in the caption, as the reviewers agreed. The 1.75-inch height exception was disclosed, not resolved.

Training curves now use matched raw-persistence subtraction, EMA-only body lines, the correct palette and shapes, direct labels, the 1.5-inch width, and `figstyle` imports. The isolated 170k read and clipped early values are recorded correctly in the sidecar. The requested “b” is still absent; coincident open markers remain difficult in greyscale. Captions and uncertainty disclosure are not delivered by a sidecar alone. Method and appendix-arena/profile assets are outside this round's requested set; I do not certify their round-one fixes here.

## Step 1 — independent proposed edits

### Figure 2 — rollouts (5 edits)

1. **R2.astra.1 — left label gutter:** Shorten the adapted row label to **“LoRA 4k”**, moving “8 adaptation episodes” to the caption, and reclaim the resulting gutter for the frames. The current two-line prose label consumes much of the usable width, leaving 0.288-inch images whose texture differences are difficult to inspect. Preserve 2-point frame gutters and the native 5.5-inch width. Do not stretch pixel aspect ratios. **Basis:** standard §§0,2,4 Figure 2; checklist 3,16.

2. **R2.astra.2 — time header:** Identify each tic-0 column as **“0*”**, with “*last context frame; raw persistence copies this frame at every future tic” in the caption. Bare 0/1/2… does not explain where the missing persistence row went. **Basis:** §4 Figure 2; checklist 10,16–17; Sunday terminology item 13.

3. **R2.astra.3 — tic-0 image cells:** Show the same **raw** last-context image in every row of a given map/control's tic-0 column. Currently truth shows raw context while model rows show its decoded reconstruction; only the former is the raw-persistence prediction. Keep future model frames fine-tuned-decoded. **Basis:** §4 Figure 2's last-context/persistence column; checklist 17; axis survey's distinction between raw and decoded persistence.

4. **R2.astra.4 — metric annotation key:** Add one caption definition, **“Numbers below model frames are scene PSNR against the raw true frame (dB).”** The naked values are otherwise easy to confuse with the paper's decoded ΔPSNR. **Basis:** §§2–3; checklist 8,14; survey reference warning.

5. **R2.astra.5 — Figure 2 inclusion/caption:** Install this asset at native width with a finding-first caption that says adaptation improves the displayed unseen-arena rollout at most shown tics but does not eliminate drift; do not claim uniform restoration. In the left-turn example at tic 32, adapted PSNR is **13.0**, versus zero-shot **14.5** and raw persistence **15.1** dB. Include the four episode/start-row/context-tic identities above, the actual window-selection rule, U-Net 200k EMA versus non-EMA 4k rank-16 adapter, eight adaptation episodes, fine-tuned SD 1 decoder, scene crop, 32-tic autoregression after 32 real context tics, and manifest sampler settings (10 steps, seed 0). If the selection rule is unavailable, disclose that instead of inventing a seeded rule. **Basis:** §§3–4; checklist 1,5,7–8.

### Figure 4 — adaptation in dB (5 edits)

6. **R2.astra.6 — panel b, fixed-threshold curve:** Draw the 3.5-dB curve as a **neutral dark dashed line above the blue solid curve**, with neutral dotted bounds and a matching direct label. Currently both central curves and the extra bounds use the same blue, and the coincident sections hide the dashed curve almost completely. Retain true coordinates, including the 1k separation; do not offset data to manufacture separation. **Basis:** §§1–2; checklist 4,8–9; round-one Fable 1/Astra 6.

7. **R2.astra.7 — panel b, attainment wording:** Replace “fraction of arenas crossed” with **“fraction reaching threshold”** and the curve label “half-gap line” with **“half-gap threshold.”** This curve is a fraction attaining a target by a budget, not the target line itself. **Basis:** §3 clarity; checklist 5,14; Sunday item 13.

8. **R2.astra.8 — panel b, rug/layout:** Compact the rug and its excess vertical separation from the 0–1 plotting region sufficiently to restore the **5.5 × 1.6-inch** budget, keeping four readable lanes, all 13 ticks, actual 4k censor coordinates, and fonts at least 6 points. Redraw at that size rather than scaling the 1.75-inch PDF. **Basis:** §§0,2,4 Figure 4; checklist 1,3,13. The taller export is a known round-one exception, not an unreported new bug.

9. **R2.astra.9 — panel a, arena-9 endpoint label:** Anchor the leader for “9” to the actual **5.109124-dB endpoint**, while moving only its text above the reference band. `iqm_panel` substitutes `home + 0.22` as the anchor for near-reference endpoints; that detaches the label from the data it identifies. **Basis:** §2 direct labels; checklist 7,15.

10. **R2.astra.10 — Figure 4 inclusion/caption:** Replace the obsolete per-arena/D-colour caption with the IQM-and-attainment caption: IQM **2.03 → 3.79 dB**, 4k interval **[3.52,4.27]**, and **9/13 [0.46,0.92]** reaching their threshold by 4k. Define the interquartile mean and per-arena target `(A0 + Atrain)/2`; state 13 arenas from this WAD, eight adaptation and eight held-out episodes per arena, 32 scheduled windows per held-out episode with duplicate handling, seed 0, U-Net 200k EMA initialization/non-EMA adapters, fine-tuned SD 1 decoder, one tic, scene rows, and decoded prediction/persistence/truth for A. Identify the filled half-gap band, fixed-threshold bounds, and training-reference episode band separately. State that the 10,000-draw nested bootstrap pairs episodes across budgets and holds Atrain fixed, and that half a gap in dB is not half the error. **Basis:** §§3–4; checklist 1,5,8; adopted statistics decision and Sunday item 14.

### Figure 4a — gap-share alternative (2 edits; conditional on retaining it)

11. **R2.astra.11 — horizontal 0.5 reference:** Rename “half-gap line” to **“half of in-distribution gap”**; retain the formula, zero-shot zero, and training-reference one. Zero on this transformed axis is **not persistence**, so do not replace its correct “zero-shot” label with a persistence label. **Basis:** checklist 10,14 applied to the actual quantity; Sunday items 13–14.

12. **R2.astra.12 — alternative caption:** State that the line is the **IQM of per-arena shares**, 0.607 **[0.514,0.734]** at 4k, with the same paired nested resampling and frozen Atrain as Figure 4. Say “share of the gap in dB,” not “fraction of prediction error removed,” and explicitly allow shares above one (arena 9 is 1.023). The line at one is a normalized fixed reference, not an uncertainty-free estimate of training performance. **Basis:** §3; checklist 7–8; survey §6. Do not derive this band by transforming only the dB IQM endpoints.

### Figure 3b — per-arena zero-shot (5 edits)

13. **R2.astra.13 — upper-panel diamonds:** Remove the **4k adapter** series from this zero-shot figure; it belongs in Figure 4. The unlabeled diamonds dominate the apparent zero-shot PSNR and do not have a corresponding M series below. **Basis:** §4 Figure 3 explicitly excludes adapters; checklist 5,9.

14. **R2.astra.14 — plotted quantities, conditional on choice (b):** Replace A/M with **scene PSNR (dB, ↑)** and **scene LPIPS (↓)** against the raw true frame, adding a neutral persistence marker on the same windows beside each arena's model marks. Use `scene_psnr_raw[_tuned]`, `scene_lpips_raw[_tuned]` and the raw `scene_persist_*` fields with the same duplicate mask. Do not obtain raw PSNR by adding raw persistence to decoded A; their references differ. **Basis:** checklist 7–8,14; Sunday item 14/raw-reference survey; the explicitly requested round-two alternative. This is a proposed revision of the standard's A/M choice, not a claim that the standard already demands absolute axes.

15. **R2.astra.15 — body layout:** Return the body panel to the standard's **U-Net-only, row-per-map** layout with two metric columns, four individual training maps shaded, and 13 unseen arenas in the adopted S0 order. Move the three-backbone paired comparison to the appendix; Figure 3a and Table 2 already carry that comparison. This exposes all 17 maps instead of replacing four with a pooled training column and makes room for persistence without crowding the current 3.3-inch categorical plot further. Redraw for the assigned slot; never shrink this PDF to 2.4 inches. **Basis:** §§0,2,4 Figure 3; checklist 1,3–4,13; statistics decision's S0 ordering.

16. **R2.astra.16 — markers:** Increase paired model markers from **3.0 to 3.5–4 points**, retaining open stock/filled fine-tuned markers and thin grey pairing segments. Allocate space through the simplified layout rather than shrinking marks to fit more models. **Basis:** §§1–2; checklist 4,9,13.

17. **R2.astra.17 — Figure 3b caption/key:** Add the metric reference, open/filled decoder key, neutral-persistence key, episode-bootstrap interval definition, map/episode/window counts, one-tic horizon, scene rows, checkpoint and EMA status to the final caption; name SD 3.5's absent fine-tuned read if the full comparison is retained in the appendix. The training-map aggregate in the existing PDF must be called a pooled read if it survives there. Start with the verified raw-metric finding after reaggregation, rather than mechanically copying a decoded-A claim. **Basis:** §§1,3; checklist 5,7–9. Numerical conversion and its intervals remain implementation work, not something this review has already drawn.

### Family-step asset — intended Figure 3a (4 edits)

18. **R2.astra.18 — y-axis name:** Replace “zero-shot latent skill S0 (dB)” with **“latent ΔPSNR vs persistence (dB)”**, defining S0 and the mean per-window log error ratio once in the caption. Keep the adopted lower-case frame-distance d consistently with Sunday item 12; there is no reason to revert it just because the older standard says D. **Basis:** checklist 14; Sunday items 12–14 and the axis survey.

19. **R2.astra.19 — coincident backbone marks:** Use distinguishable open/filled-independent shape outlines or explicit point callouts at the overlapping U-Net/PixArt locations so **both exact-coordinate points remain visible**; do not jitter the numerical d coordinates. Their training median segments differ by only 0.021 dB and many markers overpaint one another in greyscale. One concrete option is a thin larger square outline around a smaller circular centre, with that redundant backbone encoding used consistently in this panel and its key. **Basis:** §§1–2; checklist 4,9; show every point when n ≤ 30. Because decoder state is not a variable in a latent plot, explain any outline convention rather than implying a stock/tuned pair.

20. **R2.astra.20 — per-map uncertainty:** Add 95% episode-bootstrap intervals for each map's S0 from its per-window data; if deferred, explicitly caption these as point estimates without intervals. `family_step.json` contains means and counts but no S0 intervals, so “smaller than markers” is unsupported. **Basis:** §2 uncertainty; checklist 8.

21. **R2.astra.21 — placement/caption:** Give this plot an explicitly allocated native-size slot, or redraw a simplified panel for the final Figure 3 composition with its **“a”** letter; do not shrink the 5.5-inch asset into a 2-inch slot. Caption the six horizontal segments as family medians and the shaded d floor as the empirical range **[0.0223,0.0934]**; state 4 training maps plus 13 unseen arenas, scoring counts after exclusions, checkpoints/weights, one tic and latent-space evaluation. Describe the within-unseen correlations (approximately −0.05/−0.15/−0.17) as no detected ordering, not proof of a flat population relation. **Basis:** §§0,3–4; checklist 1,5,8,13. A latent metric has no decoder/crop stage; say that rather than labelling it fine-tuned-decoder PSNR.

### Deficit asset — appendix (3 edits)

22. **R2.astra.22 — clustered arena labels:** Replace the grouped “7, 13” and “1, 6, 12” annotations with separate IDs and short neutral leaders to each exact marker, offsetting **text only**. These are nearby but unequal observations, not tied points: arenas 1/6/12 have A4k **3.366/3.376/3.328**. The current shared labels make their individual intervals difficult to identify. **Basis:** §2 show every point; checklist 7,13,15.

23. **R2.astra.23 — diamonds:** Increase the hard-coded **3.2-point** diamonds to the standard **3.75 points**. **Basis:** §§1–2; checklist 9,13.

24. **R2.astra.24 — appendix caption:** Define Δ as the **motion-matched latent skill deficit**, with reference skill conditioned on training-map persistence-error deciles, rather than letting readers assume `Atrain − A0`. Identify y as decoded scene A at 4k, the 13 U-Net adapters, seed/non-EMA/decoder settings, y intervals as episode-bootstrap intervals, and the grey training-reference band. Report ρ **−0.896** with an arena-bootstrap interval once computed, or explicitly say that its interval is unavailable; the supplied JSON contains only ρ/p/n. Keep this scatter in the appendix under the adopted statistics decision. **Basis:** §§3–4; checklist 5,7–8,14; statistics decision §§1–2.

### Figure 1b — training curves (4 edits)

25. **R2.astra.25 — axis/reference terminology:** Replace “gain over raw copy-last (dB)” with **“ΔPSNR vs raw persistence (dB)”** and rename the zero-line label **“persistence.”** Retain subtraction of the matched raw baseline and do not relabel the result A. **Basis:** checklist 10,14; Sunday items 13–14.

26. **R2.astra.26 — overlapping U-Net/PixArt marks:** Stagger their marker-bearing **existing measured reads**, e.g. mark U-Net at alternating available checkpoints between PixArt's marked checkpoints, keeping the actual x/y values and both continuous curves. At 50/100/150/200k, the square covers most of the circle; the greyscale rendering still fails to expose the two trajectories clearly despite the correct colours/shapes. Keep the standard sizes and end labels. **Basis:** §§1–2; checklist 4,9; incomplete visual outcome of R1.astra.20.

27. **R2.astra.27 — panel letter:** Add **8-point bold lowercase “b”** at the top left when placed in Figure 1. It is absent in both the rendered PDF and the draw function. **Basis:** §§2,4 Figure 1; checklist 13; R1.astra.22.

28. **R2.astra.28 — inclusion/caption:** Attach the round-one disclosure to the actual paper caption: EMA means on 512 training-map validation windows, represented episodes, stock decoders, raw/full-frame scoring, one tic, provisional SD 3.5 170k isolated point versus 155k connected endpoint, U-Net/PixArt 200k, early values clipped below −1 dB, and **no interval estimates in these curve files**. Give the sampler/seed from the underlying evaluation protocol. The 170k point is not a second decoder condition; every marker is open because all these curves use stock decoders. **Basis:** §§0,3–4; checklist 1,5,8; R1.astra.23–24. The sidecar records the facts but does not communicate them to a paper reader.

Independent edit counts: **28 total** — rollout 5; adaptation 5; gap-share alternative 2; paired zero-shot 5; family-step 4; deficit 3; training curves 4.

## Step 2 — the two choices

**(a) Choose dB for Figure 4a; keep gap share as the alternative/appendix.** The dB IQM with its nested band shows the actual learning trajectory and its remaining separation from the training reference, while panel b already answers the per-arena half-gap attainment question. Together these are complementary; normalizing panel a makes both panels primarily about the same threshold construction and erases the initial arena differences. The share plot is valid here (all denominators are comfortably positive) and is a useful continuous companion to the fragile 9/13 count, but 0.607 is neither 60.7% less prediction error nor 60.7% of arenas succeeding. Report that number and interval in the text/appendix, keep Atrain uncertainty distinct from the frozen-reference nested band, and explain that half a dB gap is a geometric midpoint of error ratios. This retains the adopted statistics decision without pretending the alternative is statistically wrong.

**(b) Choose absolute scene PSNR and LPIPS against the raw frame for the body per-arena panel, with raw persistence beside each arena's model markers.** It gives the reader familiar units, an explicit difficulty baseline, and one common truth reference; PSNR ↑ and LPIPS ↓ communicate their senses without a bespoke M sign. The current paired plot silently changes from decoded truth for A to raw truth for M, so even a sign flip would solve only half its reading burden. Retain decoded A and a clearly defined paired LPIPS difference in the table/appendix to separate decoder effects from prediction error; absolute raw PSNR alone cannot establish that attribution, and baseline differences do not constitute a significance test. If paired M is retained there, I favour `M = LPIPS(persistence) − LPIPS(model)`, labelled **“LPIPS reduction vs persistence (↑)”**, with every value, interval endpoint and definition transformed consistently. Do not flip just the axis. The raw alternatives require reaggregation from raw scene fields, not algebra on decoded A; those fields exist locally (for U-Net arena 7, fine-tuned raw scene PSNR 20.508 versus persistence 19.152 dB, LPIPS 0.2774 versus 0.2170).

**Independence checkpoint:** Steps 1 and 2 are complete in this file. Fable's round-two review has not yet been opened. The comparison below is to be appended after that read, without revising this independent list to match it.

## On Fable's list

Read only after the independent section was saved (SHA-256 of that file: `248b2edc172739a343a5b4fab356f34e494b8e73d37a22e6a1b00cfbd43a27da`). Fable cites `39ada73`; a scoped git diff confirms the seven reviewed PDFs are unchanged between that commit and the requested `4e0223f`, so this is not a figure-version disagreement.

1. **DISAGREE** — Two stacked five-row control blocks with approximately 0.7-inch-wide frames would greatly exceed the 1.75-inch body height; retain the prescribed tic grid and reclaim the label gutter first, with larger strips available in the appendix.

2. **DISAGREE** — Substituting fight/strafe because they look brighter changes the held-control comparison and its selection rule; retain matched controls/windows unless a documented selection rule justifies a replacement, and do not brighten scientific images selectively.

3. **AGREE WITH CHANGE** — The grey 6-point numbers already exist; add the caption definition with **raw true frame** explicitly, distinguishing these scene PSNR values from decoded A (R2.astra.4).

4. **AGREE WITH CHANGE** — Use the compact **“LoRA 4k”** row label and put eight adaptation episodes in the caption; adding “(adapter)” to the existing prose widens the gutter further (R2.astra.1).

5. **AGREE WITH CHANGE** — Qualify the finding as improved appearance/fidelity at most displayed tics with remaining drift; uniform restoration is contradicted by the left-turn tic-32 example (13.0 adapted versus 14.5 zero-shot, R2.astra.5).

6. **AGREE WITH CHANGE** — Preserve one tick per arena and the separate tie lanes, and recheck the actual future grid; neither an 8k endpoint nor additional early reads guarantees that the first-reaching budgets will separate.

7. **DISAGREE** — The four-page layout needs a resolved height now; compact the rug and redraw at 1.6 inches rather than assuming a later grid will remove the overflow (R2.astra.8).

8. **DISAGREE** — Keep dB in the body: it preserves achieved quality and the initial deficits, complementing panel b's threshold attainment; the valid gap-share view belongs in the appendix/text (Step 2a).

9. **AGREE WITH CHANGE** — Supply a compact encoding key for the series actually retained, with stock/fine-tuned status; remove the adapter from a zero-shot panel, and keep the three-backbone key with the appendix comparison if using the U-Net-only body layout (R2.astra.13,15,17).

10. **AGREE** — Call the existing training-map point a pooled read in the caption; it must not be mistaken for four independently displayed maps (R2.astra.17).

11. **AGREE WITH CHANGE** — If paired M remains, use **“LPIPS reduction vs persistence (↑)”** and define persistence minus model, flipping values and interval endpoints consistently; “dB-free” is an unfamiliar label and does not specify the subtraction (Step 2b).

12. **AGREE WITH CHANGE** — State the observed separation of map means, but replace “d orders nothing” with **“no detected association within the 13 unseen arenas”**; the small-sample correlations do not prove a population null (R2.astra.21).

13. **AGREE WITH CHANGE** — Say the training medians **nearly** overlap (U-Net 2.846638 versus PixArt 2.825988 dB), not exactly; keep their true coordinates and make both series identifiable without a numerical offset (R2.astra.19).

14. **AGREE WITH CHANGE** — These points are not coincident: label their separate positions with leaders, and put **ρ = −0.896** with its interval/status in the caption rather than adding a lone statistic inside the plot (R2.astra.22,24).

15. **AGREE** — Rename both occurrences to persistence, retaining the explicit raw-reference qualifier on ΔPSNR (R2.astra.25).

16. **AGREE** — Put the isolated provisional SD 3.5 170k read and the 155k connected endpoint in the caption (R2.astra.28).

Fable verdict counts: **3 AGREE / 9 AGREE WITH CHANGE / 4 DISAGREE** (16 items).

Only this review file was written in the repository. Rendered PNGs are outside it; no figure, script, paper source, research-context file, or input data was edited. No commit, SSH, or remote command was made. Existing unrelated working-tree changes were left alone. Remaining checks are final native-size paper assembly and a physical print; source recorder archives, new bootstrap intervals, and an absolute-form Figure 3b were not regenerated as part of this read-only figure review.
