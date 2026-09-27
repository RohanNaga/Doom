# Round 1 figure review — Astra

Independent review, 2026-09-27. Step 1 was written to this file before opening Fable's review. Identifiers below follow FIGURE_STANDARDS §7; each numbered item proposes an edit, not an implementation.

## Scope, precedence, and verification

Read `paper/FIGURE_STANDARDS.md`, the September 27 per-arena statistics decision, and `paper/SESSION_2026-09-27.md`, together with the repository research context. The adopted statistics decision supersedes the older Figure 4 specification: keep the IQM learning curve and cumulative attainment in the body, and put the endpoint scatter in the appendix. Faint neutral per-arena traces are appropriate behind this aggregate. Do not restore the obsolete body scatter or colour arenas by D. The supplied tuned summary is a **4k**, live-weight, seed-0 result; do not invent 8k or full-fine-tune curves to satisfy an older specification.

I viewed all five figures as images: the three supplied tuned PNGs, and temporary rasters of `fig1_method.pdf` and `fig1_curves.pdf` made in `/tmp`. I also inspected their PDF text, page sizes, embedded fonts, and drawing sources. I viewed temporary greyscale renders of the training curves and performance profiles. No figure was assessed only through code/text. The method and training PNGs are absent; this did not prevent visual review. The standards require a rendered review, but do not explicitly require a PNG deliverable; the training script nevertheless advertises both formats.

PDF sizes: Figure 4 **5.5 × 1.6 in**; method **5.5 × 2.75 in**; training curves **1.7 × 1.15 in**; appendix arenas **5.5 × 3.0 in**; profiles **2.7 × 1.6 in**. All fonts are embedded Type 1 or TrueType, with no Type 3. The method uses Times/Nimbus Roman and Computer Modern, contrary to the sans requirement; the other four use Arial. The current `main.tex` and `appendix.tex` do not include these five assets, and the existing paper PDF does not supply their final placement/captions. Consequently, checklist items 1 and 3 cannot pass at the assembled-paper level. I did not physically print anything or regenerate the paper.

### Numerical audit

Reference: `paper/tables/tuned/adapt_summary.json`; training curves additionally checked against `results/training_curves/{unet,pixart,sd35}.json`. The latter are stock-decoder, raw-frame PSNR reads and must not be compared numerically with tuned, scene-only decoded advantage A.

| Figure/quantity | File value and visual check |
|---|---|
| Figure 4a, IQM at 0 / 250 / 4k | 2.031664 / 3.219360 / 3.786261 dB; the plotted points agree. Endpoint nested 95% interval [3.515161, 4.266438]. |
| Training-map reference | 5.059251 dB, episode interval [4.753803, 5.379118]; agrees with the line/band. The summary interval rounds to [4.75, 5.38], not the decision memo's historical [4.75, 5.39]. No figure prints those interval endpoints. |
| Figure 4b, half-gap attainment | At 0/250/500/1k/2k/4k: 0/4/5/7/9/9 of 13; at-risk counts 13/13/9/8/6/4 are correct **immediately before** each read. Endpoint 9/13 = 0.692308, interval [0.461538, 0.923077]. |
| Crossing rug | 250: 7,9,10,11; 500: 15; 1k: 6,17; 2k: 13,14; censored: 1,8,12,16. Arena numbers/groupings agree. The censored marks' x position does not: see R1.astra.1. |
| Fixed-threshold curve | Threshold 3.5 dB is correct. At 1k, 6/13 versus half-gap 7/13; both finish at 9/13. The visible short dotted separation agrees. |
| Appendix arena checks | Arena 7: 1.070313 → 3.863947, half-gap 3.064782, first crossing 250. Arena 9: 2.877916 → 5.109124, half-gap 3.968584, first crossing 250. Arena 12: 1.789354 → 3.328473, half-gap 3.424302, censored at 4k. Curves and crossing marks agree. |
| Profiles | At threshold 3.5: 0/13, 5/13, 9/13 for 0/500/4k. At threshold 4.0: 0/13, 3/13, 4/13. At the training reference: 0/13, 0/13, 1/13. Plots agree to rendering precision. |
| Training curves | Persistence 21.565691; EMA U-Net 200k 22.488819, PixArt 200k 22.479611, SD 3.5 155k 23.311750 and provisional 170k 23.356793 dB. Plotted endpoints agree. Subtracting raw persistence gives +0.923128, +0.913920, and +1.791102 dB at the latest reads. |

No incorrect printed metric value was found. The incorrect censoring coordinate and ambiguous episode denominators are specified below. I did not independently verify every architectural parameter count. The 24 unseen episodes are consistent with the current split (16 available for adaptation, eight held out), despite the older research plan's 20-episode proposal.

## Step 1 — independent proposed edits

### Figure 4 — tuned adaptation (7 items)

1. **R1.astra.1 — panel b, right-hand censoring rug:** Place the four censored arenas at the actual last observation, **4k**, with a short “censored at 4k” label. `crossing_rug` currently places the open marker at `4000 * 1.32 = 5280`, a budget absent from the data. Offset labels in display coordinates if needed; do not encode the label spacing as extra training. **Basis:** statistics decision §2 (censored at last grid point); checklist 7, 14.

2. **R1.astra.2 — panel b, risk row:** Give every displayed risk count an explicit matching budget, including **500 and 2k**; use a separately headed risk table if six axis labels would crowd the plot. Currently “9” and “6” sit under unlabelled minor ticks. Label the row “at risk before read” so the 13 under 250 is not mistaken for the post-crossing count. **Basis:** statistics decision §2; checklist 14 and readable type, §2.

3. **R1.astra.3 — panel b, top rug:** Draw one crossing/censor tick per arena, using separated lanes at tied budgets. The implementation draws one mark per *group* and stacks arena numbers underneath, so the current six marks do not implement the requested 13 per-arena ticks. Keep arena IDs neutral and keep ticks clear of the interval band. **Basis:** statistics decision §2; checklist 6 and §2's show-every-point rule.

4. **R1.astra.4 — panel a, IQM markers:** Increase diamonds from the hard-coded **2.8 pt** to **3.5–4 pt** at the existing 5.5-inch PDF width. **Basis:** §2 marker sizes; checklist 9, 13.

5. **R1.astra.5 — panel a, zero reference:** Move the “copy-last” label to the zero line's **right end**. Keep the existing black 0.6-pt reference. **Basis:** encoding table, copy-last row; checklist 10.

6. **R1.astra.6 — panel b, fixed-threshold uncertainty:** Supply the 3.5-dB attainment curve's own nested-bootstrap interval, either as distinguishable thin bounds or explicitly in the caption at the reported budgets; say that the current filled band belongs only to the half-gap curve. `attainment_fixed.ci` exists and differs from the half-gap interval (e.g. at 500, [0.154, 0.615] versus [0.154, 0.692]). **Basis:** statistics decision §§1,3 (every aggregate carries an interval); checklist 8.

7. **R1.astra.7 — Figure 4 inclusion/caption:** Add this exact tuned PDF at 5.5 × 1.6 in with a finding-first caption, e.g. “Most improvement arrives by 250 updates, while nine of thirteen arenas reach their half-gap target by 4k.” Define IQM and `(A0 + A_training)/2`; state 13 arenas from this WAD, eight adaptation episodes per arena, eight held-out episodes × 32 scheduled windows with duplicate handling, U-Net 200k EMA initialization, live adapter weights, seed 0, tuned SD 1 decoder, one tic, scene rows 0–207, pointwise nested arena-then-episode 95% bands with the training reference held fixed, and the separate training-reference episode band. Report 9/13 with [0.46, 0.92]. Restrict population language to these arenas/the explicit within-WAD assumption. The old caption describing D colours and full fine-tuning cannot accompany this asset. **Basis:** §§0,3; checklist 1,3,5,8; statistics decision §§1–3.

### Figure 1a — method overview (10 items)

8. **R1.astra.8 — entire diagram/layout:** Redraw the diagram as the **3.9-inch left share of the 5.5 × 1.9-inch combined Figure 1**, with one left-to-right reading line; reserve the right 1.5-inch share for the training panel. Replace the four internal lettered stages with one 8-pt bold lowercase “a”. Do not shrink the current 5.5 × 2.75-inch diagram to fit: its 7-pt text would fall below 6 pt. **Basis:** §4 Figure 1; checklist 1,13.

9. **R1.astra.9 — all text/math:** Set actual node text to Helvetica/Arial sans (declaring `\sfdefault` without selecting `\sffamily` is insufficient), and use matching sans math. **Basis:** §2; checklist 13; `pdffonts` currently reports Roman/CM families.

10. **R1.astra.10 — colour mapping:** Use blue `#0072B2` only for the U-Net outline, vermillion `#D55E00` for PixArt, and green `#009E73` for SD 3.5; use neutral labels/outlines for data groups, decoder and control plumbing. Remove the current blue=training and orange=unseen mapping and the three identically blue backbone boxes. **Basis:** §1 encoding table and §4 Figure 1; checklist 9,12.

11. **R1.astra.11 — containers and arrows:** Replace the rounded, shaded stage tiles and rounded/fill-tinted boxes with square, unfilled outlines and whitespace; remove the hatched HUD decoration, and make flow arrows straight black 0.6-pt segments with no sub-0.5-pt strokes. **Basis:** §§2,4,5; checklist 18.

12. **R1.astra.12 — data/output thumbnails:** Replace the numbered map tiles and empty schematic frame rectangles with real Doom frame thumbnails, including separate stacks labelled “4 training arenas” and “13 unseen arenas”; retain the actual 19-button stream as the conditioning input. Crop scene thumbnails to rows 0–207. **Basis:** §4 Figure 1; checklist 6,16,18.

13. **R1.astra.13 — adapter connection:** Attach the rank-16 adapter branch specifically to the **U-Net** box. The current dashed arrow leaves the enclosing three-backbone group and implies the same adaptation experiment for all three. **Basis:** §4 Figure 1, “rank-16 adapter on the U-Net only”; checklist 5,7.

14. **R1.astra.14 — predicted-latent path:** Put the sampler explicitly between the denoiser's v prediction and the next latent, e.g. a short “10-step DDIM” arrow label. The current direct `v̂ → ẑ` flow omits the iterative conversion described only in the evaluation paragraph. **Basis:** §4 Figure 1's data flow; checklist 5,7.

15. **R1.astra.15 — episode-count labels:** Change “4 training maps, 500 ep.” to “500 episodes per training map” (2,000 total), and “13 unseen arenas, 24 ep.” to “24 episodes per unseen arena” (16 adaptation pool + eight held out). The current bare numbers can be read as group totals; `main.tex:113–114` and `adapt_split.py` establish the denominators. Move these details to the caption if the redesigned diagram has no space. **Basis:** checklist 7,8; §3.

16. **R1.astra.16 — evaluation endpoint:** Replace the long A/M/G/S0 block with the specified **A, M, directional** labels; put definitions and auxiliary G/S0 quantities in the caption/text. The directional check is currently absent from the diagram. **Basis:** §4 Figure 1; checklist 5,13,15.

17. **R1.astra.17 — shared Figure 1 caption:** Add a caption stating that the paper builds a 17-arena benchmark, three world models under one recipe, a decoder tune, and a U-Net adapter path; then state that the latest available validation reads of all three backbones exceed copy-last. Keep recipe minutiae and checkpoint/provisional status here rather than in stage paragraphs or legends. **Basis:** §§3,4 Figure 1; checklist 5,8. Training-panel-specific provenance is in R1.astra.24.

### Figure 1b — training curves (8 items)

18. **R1.astra.18 — y quantity/reference:** Plot **PSNR minus each series' matched raw-frame persistence PSNR**, instead of absolute PSNR, with a black 0.6-pt zero line labelled “copy-last” at the right. Label this “gain over raw copy-last (dB)”; do **not** call it the paper's decoded advantage A unless matched decoded-reference reads are supplied. The available JSON supports raw gain only. **Basis:** §4 Figure 1b; checklist 7,10,14; metric definitions in `paper/FIGURES.md`.

19. **R1.astra.19 — weights/series:** Keep one EMA trajectory per backbone in this small body panel; move the live-weight comparison to the appendix. The current six near-overlapping curves spend the limited panel area on an additional question. **Basis:** §4 Figure 1b, “one line per backbone”; checklist 5.

20. **R1.astra.20 — series encodings:** Re-export with the exact standard backbone colours and circle/square/up-triangle markers at 3.5–4 pt on selected reads. The existing PDF uses the older palette and continuous lines without backbone markers; its nearly coincident U-Net/PixArt traces cannot be followed separately in greyscale. **Basis:** §§1,2; checklist 4,9.

21. **R1.astra.21 — legend/direct labels:** Replace the 5-pt two-column legend with 6.5-pt right-end labels, offset with short leaders where the U-Net and PixArt endpoints nearly coincide; raise the 5.5-pt reference annotation to 6.5 pt. **Basis:** §4 Figure 1b (no legend); §2; checklist 13,15.

22. **R1.astra.22 — canvas/panel label:** Draw at the specified **1.5-inch panel width**, within the combined 1.9-inch height budget, and add the 8-pt bold lowercase “b”. The current 1.7 × 1.15-inch standalone panel is not its agreed slot. Keep x linear and labelled from 0 to 200k. **Basis:** §§0,4 Figure 1; checklist 1,13,14.

23. **R1.astra.23 — missing uncertainty:** Add 95% episode-bootstrap intervals from the underlying validation windows. The supplied curve JSON has only means, so obtaining those intervals is a data dependency, not grounds to claim they are “smaller than the markers.” Until available, explicitly identify the plotted means as lacking interval estimates. **Basis:** §2 uncertainty; checklist 8.

24. **R1.astra.24 — training-panel caption/provisional point:** State the 512 training-map validation windows, represented episodes, EMA weights, stock decoders, raw/full-frame scoring, one tic, ten DDIM steps and seed 0; identify the isolated SD 3.5 **170k** point as a provisional fresh-rescore read, with its joined series ending at **155k**, and the other two ending at 200k. Explain any omitted low early EMA values (5k PSNRs 11.76, 10.79, 6.69) rather than silently cropping their descent below the plot. Do not imply all three completed 200k. **Basis:** §3, §4 Figure 1; checklist 7,8.

25. **R1.astra.25 — reproducible export:** Replace `tools/training_curves.py`'s imports of removed style symbols (`INK`, `MARKERS`, `SERIES`, `new_figure`, `save`, etc.) from `make_adapt_figures` with the `figstyle` API, then export the reviewed PDF and its advertised PNG together. Static inspection finds these imported names are not defined by the current adaptation module. The current 5/5.5-pt text also conflicts with `figstyle.save`'s minimum-size guard, so a simple rerun is not a complete repair. **Basis:** §6 build checklist and §5 refusal of invalid outputs. I did not execute a figure rebuild or write either asset.

### Appendix — per-arena panels (4 items)

26. **R1.astra.26 — shared update label:** Change the bottom “updates” labels to a shared **“adapter updates (log scale)”** label, preserving the separate step-0 segment. **Basis:** §2; checklist 14.

27. **R1.astra.27 — uncertainty/reference layering:** Draw the blue episode bands **above** the grey training-reference band, retaining translucency and placing the mean curve on top. Currently `fill_between` has the default lower z-order and `training_line` paints an opaque grey band over it, concealing interval portions for arenas 9 and 10. For example, arena 10 at 4k has CI **[4.538150, 5.201536]**, whose upper portion should remain visible inside the grey band. PDF vector inspection confirms the full interval exists but is overpainted. **Basis:** §2 uncertainty; checklist 7,8.

28. **R1.astra.28 — measured checkpoints:** Add standard adapter diamonds at the measured update budgets on every arena curve, keeping the distinct vertical first-crossing ticks. This separates sampled reads from interpolation across the log axis and matches the adapter encoding. **Basis:** §§1,2; checklist 9.

29. **R1.astra.29 — appendix inclusion/caption:** Include this tuned panel at its native 5.5 × 3.0 in with a finding-first caption describing heterogeneous gains and non-crossers. State the same model/data/scoring provenance as Figure 4, identify blue bands as per-arena 95% episode bootstrap intervals and grey as the training-reference interval, and name arenas 1/8/12/16 as censored **at 4k** (absence of a crossing tick is not missing data). **Basis:** §§0,3; checklist 1,3,5,8; statistics decision §2.

### Appendix — performance profiles (2 items)

30. **R1.astra.30 — threshold axis/reference:** Extend the A-threshold axis to **0 dB** and add a vertical black 0.6-pt line labelled “copy-last”. All three profiles equal one there; the present minimum near 0.77 dB omits the required reference even though A is on the x axis. **Basis:** §1 copy-last encoding; checklist 10.

31. **R1.astra.31 — appendix inclusion/caption:** Include the 2.7 × 1.6-inch tuned PDF at scale 1 with a finding-first caption such as “Adaptation shifts the arena distribution upward, but only one arena exceeds the training-map reference at 4k.” Define the curves as the fraction of the 13 arena means satisfying `A >= τ`, identify 0/500/4k budgets and their pointwise nested 95% bands, and supply the model/episode/seed/decoder/horizon/crop provenance from Figure 4. State that blue shade encodes update budget here. **Basis:** §§0,1,3; checklist 1,3,5,8. Direct labels and shade order remain distinguishable in the greyscale render; no extra categorical palette is needed.

### Checklist disposition after independent inspection

- **Build/page (1–4):** embedded-font check passes all five; Figure 4 has the correct standalone page size. Figure 1 sizes fail. Final inclusion, 100%-zoom paper inspection and physical print remain unverified because the assets are not in the assembled paper. Training-curve greyscale separation fails; profiles remain readable in the greyscale render.
- **Content (5–8):** all statistical panels have data; method frame rectangles are schematic substitutes for the required real images. Adaptation values/IDs/counts pass the audit above, with censoring positioned at a nonexistent budget. Captions and uncertainty qualifications need the edits above.
- **Encoding (9–12):** adaptation/reference colours largely conform; Figure 1's old palette does not. D is not encoded in any of these five reviewed figures. Profiles omit the copy-last threshold; the method omits the directional endpoint.
- **Type/axes (13–15):** Arial and round ticks generally pass in the adaptation assets; the method family and training-panel sub-6-pt text fail. Appendix update axes omit “log scale.” No statistical panel has gridlines or a boxed legend; the training panel still has the prohibited internal legend.
- **Images (16–17):** real-frame/HUD rules apply to the proposed method thumbnails; current schematic frames cannot be pixel-checked. No rollout/tic-0 panel is in this review, so checklist 17 is otherwise not applicable.
- **Decoration (18):** the statistical assets pass; the method's pastel tiles, rounded boxes and hatching fail.

Step 1 complete: **31 items** (Figure 4: 7; method: 10; training curves: 8; appendix arenas: 4; appendix profiles: 2). No other reviewer's list was consulted for these items.

## On Fable's list

Read `paper/review_figures_round1_fable.md` only after the complete Step 1 list above had been saved. Numbers below refer to Fable's original items. Where a verdict changes an edit, that replacement still requires the protocol's joint agreement; this review applies no figure changes.

1. **AGREE WITH CHANGE** — Keep the mandatory 3.5-dB comparison; give it a distinct dashed stroke and a separate “A ≥ 3.5 dB” label, and state that it coincides with half-gap attainment except at 1k. The full curve is already drawn; its overlap is real, and dropping it would contradict the statistics decision. Supply its own uncertainty as in R1.astra.6.

2. **AGREE WITH CHANGE** — Use separate short rug lanes for the four tied arenas at 250 (R1.astra.3), rather than a wide comma-separated label that can collide with the neighbouring 500 group; retain the observed 250 budget and do not assume an unfinished 8k grid will separate the crossings.

3. **AGREE WITH CHANGE** — Expand and define “interquartile mean” in the caption, but keep the short “IQM” end label; the full phrase would consume too much of the 1.6-inch-high two-panel figure. This also avoids repeating a definition in both places (§3).

4. **DISAGREE** — In the reviewed tuned PNG/PDF the label is already above the band's upper edge; `training_line` positions it at the upper CI with `va="bottom"`, and it does not cross the dashed line.

5. **AGREE** — Label arenas 7, 9 and 12 at the ends of their faint neutral traces, with non-overlapping neutral labels; this preserves the aggregate focus while identifying the three named examples (§4 Figure 4).

6. **AGREE WITH CHANGE** — Use the finding-first caption, but accompany 2.03 → 3.79 dB and 9/13 with their nested intervals, call the four “censored at 4k,” and include the model, data, decoder and scoring provenance in R1.astra.7; encodings alone do not satisfy checklist 8.

7. **DISAGREE** — Keep “adapter updates (log scale)”: that is the standard's explicit wording and correctly describes positive budgets, while the visibly separated 0 segment handles the exception.

8. **AGREE WITH CHANGE** — Identify an independent **stock decoder D** input on the rendering/tuning path, rather than drawing a weights arrow from the node currently labelled “frozen VAE encoder E,” which could imply that encoder weights initialize the decoder; keep the EMA-to-adapter link U-Net-specific (R1.astra.13).

9. **AGREE WITH CHANGE** — Remove the empty hatched output rectangles; substitute real decoded examples if the redesigned overview retains output images, otherwise use the concise semantic labels “prediction / decoded truth / decoded copy-last.” Retain real input/map thumbnails as required by §4 Figure 1 (R1.astra.12).

10. **AGREE** — Replace the cryptic concatenation annotation with “context and noisy next latent stacked on channels,” preferably in the caption if the prescribed compact method layout cannot carry it (§4 Figure 1; checklist 5).

11. **AGREE WITH CHANGE** — Adopt the three standard backbone outline colours and neutral plumbing, but replace the map grid with the required two real-thumbnail stacks rather than retaining a light-filled map category (§4 Figure 1; R1.astra.10–12).

12. **AGREE** — Move the agent/bot/duration prose to the caption and parameter counts to Table 1; this is necessary to redraw the method inside its 3.9-inch share without shrinking text below the minimum (§§2–4).

13. **AGREE WITH CHANGE** — Keep the required 0–200k x range and explicitly disclose omitted early EMA values in the caption; changing an absolute-PSNR floor to 20 or 20.5 dB cannot reveal 5k values of 6.69–11.76 dB. Apply the explanation to the gain axis proposed in R1.astra.18 rather than preserving silent clipping.

14. **AGREE WITH CHANGE** — Identify “SD 3.5, 170k, provisional fresh-rescore read” **in the caption**, and explain the isolated/open point there; §3 puts checkpoint/provisional provenance in the caption, and open versus filled already means decoder status elsewhere in the encoding table.

15. **AGREE WITH CHANGE** — Use direct EMA end labels with offsets/leaders for near-coincident endpoints, and move live curves to the appendix; the specified Figure 1b has one line per backbone, so the body caption should not preserve an extra live/EMA comparison (R1.astra.19,21).

16. **DISAGREE** — Subtract the matched persistence baseline as required by §4 Figure 1b; it puts the central comparison directly on the zero axis. Label it raw-copy-last gain, not decoded A, because the available JSON supports only the former (R1.astra.18).

17. **DISAGREE** — Do not add D and S0 to all 13 panel titles: §2 prohibits plot titles, §5 discourages unused value labels, and the statistics decision assigns those relationships to the appendix scatter/table rather than crowding the learning curves.

18. **DISAGREE** — The reviewed tuned profile already has a vertical grey training-reference interval band and dashed line at 5.059 dB; `fig_profiles(..., band=...)` draws both. No edit is needed for this request.

Fable verdicts: **3 AGREE, 10 AGREE WITH CHANGE, 5 DISAGREE**. No figures, scripts, context files or paper sources were changed; no commit or remote command was made. Remaining verification is an owner-supplied assembled paper with these assets and captions, an actual print check, and validation-window uncertainty for Figure 1b.
