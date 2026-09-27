# Figure review, round 0 (Opus reader), 2026-09-27

Reviewed at `ab6c1fa`. The PNGs reviewed are `paper/figures/tuned/*.png`, the stock PDFs `paper/figures/fig3_adaptation_seeds.pdf`, `fig_adapt_ladder.pdf` and `fig_adapt_recipe.pdf` rendered at 300 dpi, and the strips `paper/figures/figure1/train_map02_turn_left_ep6000_s837_strip.png` and `unseen_arena07_turn_left_ep41_s269_strip.png`. I also read the code in `paper/make_adapt_figures.py`, the float layout in `paper/main.tex` and `paper/corl_2026.sty`. The standard these comments apply is `paper/FIGURE_STANDARDS.md`; section numbers like "S2" point into it.

Each comment names one edit. Comments are numbered so the second reviewer can answer "agree" or "disagree" per number.

## 0. What the reference papers' figures actually do (evidence for question 1)

I viewed each figure as a rendered page of the arXiv PDF, not as a summary. The venue comes from the arXiv comments field or the page header.

**GameNGen (2408.14837, ICLR 2025).**
- Overview, Fig 3: full width, about 1.9 in tall. A dashed vertical rule splits "Data Collection via Agent Play" from "Generative Model Training". It uses one robot glyph for the agent, a yellow trapezoid for the denoiser, and small rounded boxes. Frames are blue squares and actions are yellow squares, and the same two colours run through both halves. Annotations are tiny monospace text.
- Rollouts: Fig 5 is a 3 × 7 grid with the columns "past observations (last 4)", an arrow, "prediction" and "ground truth". Column headers appear once at the top, with no row labels.
- Decoder comparison, Fig 12: 5 rows × 3 columns headed "SD v1.4 Autoencoder", "Fine-Tuned Autoencoder" and "Ground Truth". White gutters separate the frames and the HUD is kept, because the HUD is where the decoders differ. Our Figure 3c should be built this way.
- Results, Fig 6: two small default-matplotlib line plots of PSNR and LPIPS over 64 autoregressive steps. Their tick labels are about 4 pt at print size, and neither plot has a baseline or a reference line.
- What works: the column headers and the gutters.
- What does not work: the Fig 6 plots are unreadable at print size, and they give the reader nothing to compare against.

**DIAMOND (2405.12399, NeurIPS 2024 spotlight).**
- Rollouts, Fig 3: two grid panels (DDPM and EDM). Rows are the number of denoising steps n = 10, 5, 3, 1, labelled at the left. Columns are t = 0, 5, 10, 50, 100, 500, 1000, which is a log time schedule. That schedule lets 1000 steps fit in seven columns.
- Fig 4: two rows labelled n=1 and n=3, and six columns t=0 to 5.
- Results, Fig 2: an rliable-style interval plot, one colour per method with DIAMOND in blue, wrapped beside the text at about 2.3 in, with very small fonts.
- Fig 8 (appendix): pixel drift on a log time axis. DDPM variants share one green family and EDM variants one blue family, with the line style encoding n and shaded standard deviations. The title sits inside the plot in monospace.
- What works: the log-spaced columns, and one hue family per method with line style for the level.
- What does not work: the in-plot title, and tick fonts under 5 pt.

**Genie (2402.15391, preprint; ICML 2024 per the authors' site, VERIFY).**
- Overview, Fig 3: flat boxes filled in three saturated colours (blue video tokenizer, yellow latent action model, red dynamics model). Tokens are drawn as bars and real game frames are stacked at both ends. It is simple and legible.
- Scaling, Fig 9: three panels at full width. A sequential magma-like ramp encodes model size, an ordered quantity, from dark to light.
- Behaviour cloning, Fig 15: plotly styling with the titles "Easy" and "Hard" inside the plot. Random and oracle baselines are dashed horizontal lines in their own colours.
- Rollouts, Figs 11 and 13: pixel-font headers ("Prompt", "Generated") with gamepad glyphs. Frame borders are coloured by latent action, and the caption itself is set in coloured text.
- What works: the ordered ramp in Fig 9, and baselines drawn as reference lines in Fig 15.
- What does not work: the pixel fonts, the glyphs and the coloured caption text, which Nature's figure specification lists under "avoid coloured text".

**Genie 2 (blog, 2024-12-04) and Genie 3 (blog, 2025-08-05).** Neither has a paper. The pages are video galleries. Genie 2 has one architecture flowchart, and Genie 3 has a comparison table with GameNGen, Genie 2 and Veo. I read both pages through a fetched summary only, so **VERIFY** the visuals. There is nothing on these pages to imitate for a quantitative figure.

**Oasis (Decart and Etched, blog, 2024-10-31; oasis-model.github.io).** I opened the raw HTML and the two images.
- Architecture (`arch_new.png`): flat, with one accent blue. It uses all-caps monospace labels, real Minecraft frames, and WASD keycaps as the only glyphs. It looks designed, not generated.
- Throughput chart (`speed.png`): a bar chart with rounded bar tops and a bold title inside the chart. Oasis is black and the baselines are light grey. The grey-against-black emphasis works; the rounded bars and the in-chart title are marketing style.

**Vista (2405.17398, NeurIPS 2024).**
- Pipeline, Fig 3: full width and very busy. It has a flame glyph for "trainable", a stop sign, traffic-sign icons, a video-platform-style red play button, rounded tinted boxes, dashed rounded regions, and six or more colours. This is the decorated style that now reads as generated.
- Rollouts, Fig 5: rows are models, labelled by rotated text at the left, and columns are time. Red boxes and overlaid text ("corrupted", "misaligned") point at failures.
- Fig 6: a timeline of blue lines above the frames shows each prior work's horizon, a direct-labelled comparison of length.
- Results: Fig 7 is grouped bars with a value label on every bar. Fig 8 is horizontal bars with the labels written inside the bars in a blue-to-purple gradient.
- What works: the annotations on the failures and the horizon timeline.
- What does not work: the icons, the value labels on every bar, and the gradient bars.

**GAIA-1 (2309.17080, technical report).**
- Architecture, Fig 2: outline-only trapezoids and boxes with small monospace labels. Two accents (teal and magenta-orange) mark input and output tokens, and token columns are drawn as stacked cells. It is disciplined and reads as authored.
- Actions, Fig 13: a "context + conditioning" column in which the commanded path is drawn as an orange ribbon on the context frame, followed by generated frames with "+1 s" stamped in each corner. A thin rule and a header strip ("CONTEXT + CONDITIONING | GENERATED FRAMES") separate the groups, and a separator line isolates the last row.
- What works: drawing the control on the image, and time stamps inside frames.

**XEWorld (2608.05799, preprint, Aug 2026).**
- Teaser, Fig 1: three pastel rounded panels (blue, pink and green) headed in a handwriting-style font: "THE PROBE", "THE QUESTION", "THE ANSWER". It has robot renders, a wordmark logo, and four miniature result charts whose text is unreadable at print size. Each mini chart carries its finding as its title ("Adaptation helps but causes forgetting"). This is the clearest example in the set of the infographic look (S5).
- Fig 2: small multiples, three columns of interventions by two rows of robots. It uses grouped bars in red and teal families with a value label on every bar.
- Fig 5: two scatter panels of LPIPS against distance with five direct-labelled points, a fitted line, and r in the corner.
- What works: the direct labels on the Fig 5 points.
- What does not work: a fitted line through n = 5, and the unreadable mini charts in Fig 1.

**DreamerV3 (2301.04104; published in Nature 2025, VERIFY volume).**
- Fig 1a: eight small-multiple bar panels, one per domain. The baselines are light grey, PPO is teal and Dreamer is blue. Labels are written vertically inside the bars.
- Fig 1b: a learning curve for the Minecraft diamond task with max and mean lines, plus item images at the milestones.
- Rollouts, Fig 4: row labels "True" and "Model" rotated at the left, the headers "Context Input | Open Loop Prediction", and T = 0, 5, 10 up to 50 under the bottom row. Two environments are stacked, each with its own True/Model pair. Our Figure 2 should follow this layout.
- Ablations, Fig 6: lowercase bold panel letters, three ticks per axis, and an ordered blue ramp for model size and replay ratio with the legend beneath.
- What works: almost everything. Dreamer keeps its blue across every figure.

**Cosmos (2501.03575, technical report).**
- Platform, Fig 4: five plain boxes inside a frame, a minimal diagram.
- Tokenizers, Fig 8: two scatters of compression rate against PSNR on a log x axis. Every point has a different colour that encodes nothing, and every point is text-labelled, so the labels crowd. The titles are inside the plots.
- Physics rollouts, Fig 20: pairs of rows labelled "Simulated" and "WFM" at the left. Each group gets a bold scenario header, the bottom edge reads t = 0 (conditioning), 11, 22, 32, and thin white gutters separate the frames. It is clean.
- What does not work: the rainbow of per-point colours in Fig 8.

**DreamGen (2505.12705, CoRL 2025, PMLR 305:5170-5194).**
- Overview, Fig 2: four "Step N" rounded pills in beige, green, lavender and peach, with a clip-art camera, a globe and a neural-network glyph. It is the pastel-zone style.
- Scaling, Fig 4: two panels of success rate against neural-trajectory count. Three coloured series carry shaded bands and a value label at every point, the legend sits inside the plot, the axis labels are bold, and a strip of task thumbnails sits beneath.
- What does not work: the value labels, which clutter the lines.

**DINO-Foresight (2412.11673, NeurIPS 2025).**
- Overview, Fig 1: trapezoid encoders with flame and snowflake glyphs for trained and frozen, and dashed regions for "At Train-Time" and "At Test-Time". The glyph convention is now common, and it is one of the tells listed in S5.
- The qualitative figures are grids with oracle columns.

**Ctrl-World (2510.10125, ICLR 2026 per the page header; added as a recent robotics world-model paper).**
- Fig 6: frame borders coloured by source (blue initial observation, yellow real rollout, green world-model rollout) with a legend strip above.
- Fig 7: scatter of world-model success against real success, with a regression line, its equation printed in the plot, an oracle diagonal, and fonts under 5 pt.

**What none of them do.**
1. Draw a persistence or copy-last reference in any rollout grid or any error-over-steps plot. This covers GameNGen Figs 4, 6 and 7, DIAMOND Fig 8, DreamerV3 Fig 4 and Cosmos Fig 20.
2. Show a per-scene result with intervals for held-out scenes. XEWorld comes closest, with five robots and no intervals.
3. Use more than five rows in a rollout grid in the body.
4. Order a categorical axis by a covariate and leave the reader to infer a slope. Where a distance matters, XEWorld puts it on a numeric axis.

**What works across them.** The rollout grids that read best (DreamerV3 Fig 4, Cosmos Fig 20, GameNGen Fig 12, GAIA-1 Fig 13) have left row labels, time on one edge and thin gutters, and nothing else. The results plots that read best (DreamerV3 Figs 1 and 6) use one emphasis colour against grey, or one ramp for an ordered quantity, with three ticks. The overviews that read as authored (GameNGen Fig 3, GAIA-1 Fig 2, Oasis) use outlines, one or two accents and real frames.

## 1. Global comments (apply to every data figure from `make_adapt_figures.py`)

1. **Type is too small for a 10 pt page.** `style()` sets labels at 6 pt, ticks at 5.5 pt, annotations at 5 pt and legends at 5.5 pt. The ladder legend is 4.5 pt (`fontsize=4.5`, line 949) and the arena labels in `fig_skill` are 4 pt (lines 997 and 1002), which is below Nature's 5 pt floor. Set labels to 7 pt, ticks and annotations to 6.5 pt and the floor to 6 pt (S2). The current type is small because the slots are small (1.7 × 1.15 in), so this comment depends on comment 3.
2. **The palette is a brand-neutral placeholder, not a colour-blind-safe set.** `SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")` puts an orange (`#eb6834`, PixArt) next to an amber (`#eda100`, LoRA), and the two converge under deuteranopia. Replace `SERIES` and `MARKERS` with the encoding table in S1, keyed by entity name rather than by slot order.
3. **The slots are too small for the content.** `main.tex` puts Figures 2a and 2b at 0.49 of a 0.64-linewidth minipage (about 1.7 in each) and Figure 3 in a 0.33-linewidth minipage (1.8 in), all 1.15 in tall. Move to the S4 sizes: Figure 3 at full width × 1.8 in, Figure 4 at full width × 1.6 in. Update `FIG2_SIZE`, `FIG3_SIZE` and `WIDE_SIZE` to match, and delete the shared minipage float.
4. **One glyph means two things across figures.** In fig2a the orange square is PixArt and the amber diamond is "U-Net + LoRA". In the ladder, seeds and recipe figures the amber diamond is arena 7, the orange square is arena 12, the blue circle is arena 8 and the green triangle is arena 16, all through `arena_styles()`. Arena identity must never use backbone colours or markers. Use dark grey lines with direct labels (S1, "Named arenas").
5. **D is mapped to colour in `fig_curves`, `fig_terciles` and `fig_skill` (`d_colours`)**, while the paper's text says D orders nothing among the 13 arenas. A colour ramp invites the reader to look for an ordering. Remove D from colour everywhere. Where a ramp is wanted, key it to Δ, the quantity Figure 4b puts on its x axis.
6. **Replace "home" with "training maps"** in `home_line()` and every `ax.text(..., "home")`. The paper's prose uses "training maps", and "home" is our lab word.
7. **Y ticks fall on odd steps** (0, 1.5, 3.0, 4.5 in fig2a and fig3; 0.00, 0.08, 0.16 in fig2b) because of `MaxNLocator(4)`. Use `MultipleLocator(1)` for A in dB and `MultipleLocator(0.05)` for M.
8. **Confidence intervals are hidden.** In `fig_zero_shot` the CI line is 0.6 pt, drawn at zorder 2 under a 2.8 pt marker with a white edge, so most intervals are shorter than the marker and invisible (fig2a arenas 8, 15 and 13). Draw the CI above the marker in the series colour, or state in the caption that the intervals are smaller than the markers and give their median half-width.
9. **Legends above the axes take 30 to 35 percent of the height** in fig2a, fig2b and the ladder. Replace them with direct labels at the right end of each series, or, for Figure 3, rely on the method figure's three backbone boxes plus one line in the caption.
10. **Provenance appears in a legend** ("SD 3.5 (170k, provisional)", `ROW_NAMES`). Move it to the caption, and keep legend names to the entity only.
11. **Unit and direction cues are missing.** "M_0" has no unit and no direction. Write "M, LPIPS difference (lower is better)", or put a 6.5 pt label "model looks better than copy-last" under the zero line.
12. **Do not ship figures that have no data.** The tuned ladder has one x value, the tuned recipe figure lists four legend entries with no lines, and the tuned seeds figure has an empty right panel (13 to 15 below). Make `save()` refuse a figure where any legend handle has no drawn artist, or any axes has no data.
13. **Fonts are embedded correctly and should stay that way.** `pdffonts` on `tuned/fig3_adaptation_curves.pdf` shows Arial and Arial Italic as embedded CID TrueType. Keep `pdf.fonttype: 42`. The TikZ Figure 1 must use `\sffamily` (Helvetica through the sty's `phv`) so that all four body figures share one family (S2).
14. **Put the story in the figure numbering.** `main.tex` currently calls the strips Figure 1. The plan makes the overview Figure 1 and the rollouts Figure 2. Rename the `\figslot` files and labels in one commit (`fig1_overview.pdf`, `fig2_rollouts.pdf`, `fig3_shift.pdf`, `fig4_adaptation.pdf`) so that the file names match the figure numbers.

## 2. Figure 2a in its current form (`tuned/fig2a_advantage_by_distance.png`, the future Figure 3)

15. **The y label says `A_0`, but the amber diamonds are the 4k adapter**, which is not A_0. Either drop the adapter series from this figure (it belongs in Figure 4, S4) or relabel the axis "A (dB)" and pair each zero-shot U-Net point with its adapter point.
16. **Only one training-map line is drawn**, a blue dashed line at 5.06 dB, which is the U-Net's tuned value. PixArt and SD 3.5 points are therefore read against the U-Net's line. Draw each backbone's own training-map value, or none in this panel. Figure 3a's step design carries that comparison better.
17. **The training maps are absent**, so the step the plan asks for cannot be seen. The docstring says "training maps shaded", but the four maps have no rows in the input. Add maps 2 to 5 from the per-map training read (queued on the review page), or build Figure 3a on a numeric D axis, where the training maps sit inside the floor band.
18. **A categorical axis ordered by D ("arena, by D") shows no D values** and invites a reader to see a slope the text rejects. Following S4, Figure 3a plots D numerically and Figure 3b orders arenas by value.
19. **Markers overlap** (arena 16: circle, square and triangle; arena 6). The dodge width `0.5/len(rows)` gives about 0.12 slot units for four rows. In the Cleveland layout of S4 (arenas as rows, backbones offset vertically by 0.2 of a row) this problem goes away.
20. **SD 3.5 goes through its own stock decoder while the U-Net and PixArt go through the tuned SD 1 decoder**, yet all three share one axis with no mark of the difference. State the decoder per series in the caption, and give SD 3.5 open markers (stock) until its tuned decoder exists, following the S1 rule of open = stock.

## 3. Figure 2b (`tuned/fig2b_margin_by_distance.png`)

21. **The y label `M_0` again covers the adapter series.** Apply the same fix as comment 15.
22. **No tick below zero**, although the adapter points reach −0.05. Set ticks at −0.05, 0, 0.05, 0.10, 0.15.
23. **Label the zero line "copy-last"** at its right end at 6.5 pt. It is the most important line in the panel.
24. **Keep this panel as the M sub-panel of Figure 3b** (Cleveland rows shared with A), not as a stand-alone panel with its own x axis. The two metrics are read per arena side by side.

## 4. Figure 2c (`tuned/fig2c_outcomes_by_skill.png`, the future Figure 4b)

25. **Arena labels are 4 pt and collide** ("7" on "13", "11" on "9", "1" beside "12" and "8"). At 6.5 pt, place the labels with a small offset table written by hand for the 13 points. Automatic placement will not clear the pairs at (1.42, 3.86)/(1.42, 3.80) and (1.80, 0.25k)/(1.86, 0.25k).
26. **Points are coloured by D with no key.** This is the encoding problem of comment 5, now invisible to the reader. Use one colour (U-Net blue) or the Δ ramp described in the caption.
27. **The x axis is S_0**, but the 10:00 decision moved the predictor to the deficit Δ. Rebuild on Δ (dB) as the plan says, and keep S_0 for the appendix.
28. **The right panel's ">4k" row sits one tick above "4k"** as if it were a value. Separate it with an axis break and label it "not crossed by 4k". Better still, drop the right panel from the body: Table 3 carries cost, and S4 gives Figure 4b the endpoint only.
29. **"half-gap budget (updates)" is jargon on an axis.** If the panel stays in the appendix, write "updates to close half the gap" and define the gap in the caption.

## 5. Figure 3 curves (`tuned/fig3_adaptation_curves.png`, the future Figure 4a)

30. **13 lines in one blue ramp cannot be told apart**, and the ramp encodes D (comment 5). Key the ramp to Δ and directly label three arenas at the right end (7, 12, 9). The other ten stay unlabelled as context.
31. **Remove the 13 faint half-gap horizontals** (`ax.axhline(r["half_gap_line"], alpha=0.25)`). They outnumber the curves, cross them, and the reader cannot tell which line belongs to which curve. Crossings belong in Table 3.
32. **Filled and open circles carry meaning (first crossing, censored) that the figure never states**, and the open circles at 4k read as data points. Drop both markers once comment 31 is applied. If crossings stay, mark them with a small vertical tick on the curve and explain it in the caption.
33. **The colour bar takes about 25 percent of a 1.8 in panel.** Delete it. Colour is explained by Figure 4b's x axis (S4).
34. **Extend to the 8k grid** (0, 50, 100, 150, 250, 500, 1k, 2k, 4k, 8k) with the full fine-tune dashed black, as the plan specifies. Ticks at 50, 250, 1k, 8k, with minor ticks unlabelled.
35. **The broken-axis mark "//" is drawn as text over the spine.** Use two spines with a 3 pt gap (`ax.spines['bottom'].set_bounds`) so that step 0 is visibly separate. The current slashes print as a glyph collision at 600 dpi.
36. **The y range starts at 1.2 with ticks at 1.5, 3.0 and 4.5.** Use 1, 2, 3, 4, 5 dB (comment 7).

## 6. Figure 3 terciles (`tuned/fig3_adaptation_terciles.png`)

37. **Drop this figure.** Three translucent bands at alpha 0.12 stack into grey-blue mixtures that are in no legend, and the terciles are of D, the null covariate. The null is stated better as one number (Spearman of D against A at 8k, with its interval) in the Figure 4 caption.

## 7. Seeds, ladder, recipe (appendix)

38. **Tuned seeds figure: the right panel is empty and the left has no arena legend.** Do not build it until seed-1 tuned rows exist (comment 12).
39. **Stock seeds figure: the right panel joins the seed-1 minus seed-0 differences with lines, which suggests a trend in noise.** Draw points only, with the zero line and a grey band at ±0.06 dB labelled "seed spread at 4k".
40. **Stock seeds figure: arena colours reuse backbone colours** (comment 4). Use dark grey with direct labels, solid for seed 0 and dashed for seed 1.
41. **Tuned ladder: one x value (8 episodes)**, and four dotted step-0 lines with no label. Do not ship it (comment 12).
42. **Stock ladder: the x axis is 1, 2, 4, 8, 16 evenly spaced**, which is a log2 axis without saying so. Label it "adaptation episodes (log scale)". Label the dotted step-0 lines "step 0" once, and raise the legend from 4.5 pt to 6.5 pt as direct labels.
43. **Tuned recipe figure: the legend lists lr 3e-4, lr 5e-4, 8k grid and seed 1, but none are drawn.** Do not ship it (comment 12).
44. **Stock recipe figure: the two panels use different y ranges under the same label.** Share y (`sharey=True`), and move "arena 12 (D 0.138)" from an in-plot title to a panel letter plus a caption phrase. The D value is irrelevant here.
45. **Stock recipe figure: the data sit at 500 and 2k, but only 50, 250, 1k, 4k and 8k are labelled.** Label the grid steps actually used, or state "(log scale)" and keep the minor ticks.

## 8. Rollout strips (`figure1/*_strip.png`, the future Figure 2)

46. **No row labels, no time labels, no control label.** Add rotated row labels at the left, the tic numbers above the first row, and a block header naming the held control (S4, Figure 2).
47. **Frames abut with no gutters**, so each frame's HUD bar reads as a horizontal rule between rows. Add 2 pt white gutters, or crop the HUD (comment 49), which removes the false rules entirely.
48. **The copy-last row repeats one frame eight times.** Replace it with a tic-0 column (the last context frame) shared by all rows, as DreamerV3 Fig 4 and GameNGen Fig 5 show their context. This frees a row for the adapter.
49. **In the training map 2 strip, the model and copy-last rows show the stock decoder's wrong HUD digits** ("109 752 ... 05" against the true "139 751 ... 01"). A reader will take these for model errors. Crop to scene rows 0 to 207 as the scored region does, and decode with the tuned SD 1 decoder as the 10:20 plan says.
50. **VERIFY the copy-last frame.** In `train_map02_turn_left_ep6000_s837_strip.png`, the copy-last frame does not match the true tic-1 frame. The wall corner sits at about x = 130 of 320 against x = 270, and the whole frame is darker. The model's tic-1 frame does match the truth. Under a left turn at 35 Hz the previous tic should differ by a few pixels, not 140. Before the figure is used, check that the strip copies `context[-1]` (the last context frame) and not an earlier context frame. The arena 7 strip does not show this offset, so the cause may be specific to the window (a respawn or teleport inside the context) rather than a code bug.
51. **The columns fall at tics 1, 4, 8, 12, 16, 20, 24, 32, an uneven spacing.** Use 1, 2, 4, 8, 16, 32 (six columns, doubling), which shows both the first-tic response and the late drift.
52. **The plan's row list shares one "truth" row between map 2 and arena 7.** Each map needs its own truth row. Use two row groups: map 2 (true, model) and arena 7 (true, zero-shot, adapted 4k). That gives five rows, with the two controls as side-by-side column blocks (S4).
53. **Choose windows by the seeded rule the caption promises**, and put episode and start tic in the caption, not in the image.

## 9. Tables in `main.tex`

54. **Table 1 marks provisional rows in blue text** (`\prov`). Coloured text disappears in greyscale, and Nature lists it as a thing to avoid. Keep the blue for drafts only. The submission build must use a dagger or an "(prov.)" suffix.
55. **Table 1 has no true-frame reference for the directional column.** Add "0.91 on true frames" to the header or the caption, so that 0.867 can be read against a ceiling.
56. **Table 1 labels the persistence row "Persistence / copy-seed" with directional 0.** Split the row label ("copy-last (1 and 4 tics), copy-seed (256 tics)"), and write the directional cell as "n/a" or an en dash: copy-last makes no turn prediction, so a score of 0 is a category error.
57. **Table 2 puts counts in parentheses inside every cell**, which doubles the width. Move the counts to a separate narrow column per metric ("maps > 0"), or into the caption for the unseen rows only.
58. **Table 2 mixes provenance** (pre-fresh-set episodes, stock decoder, full frame). When the fresh-set rows land, add the `\cmidrule` groups for the stock and tuned decoders (S4), and state "scene rows" in the caption.
59. **Table 3 is a transposed table** with metrics as rows and conditions as columns. That is right for three conditions. Keep it, but right-align the numbers and put units in the row labels only.

## 10. Figure 1 (method overview; no file yet)

60. **Before the TikZ worker commits**, check the draft against S5. There should be no icons (the plan's "data, three backbones, decoder tune, arena adaptation, evaluation" invites a robot or joystick glyph), no pastel stage fills, and backbone colours only on the three backbone boxes.
61. **The training-curve side panel should plot the held-out advantage over copy-last against updates, not training loss.** The three backbones' losses are in different latent spaces (4-channel SD 1 against 16-channel SD 3.5) and are not comparable on one axis. The every-5k steward reads already hold A on training-map validation windows.

## 11. Things I could not verify

- The Science figure guide (403 to both curl and WebFetch).
- The Wong 2011 hex values (paywalled). The hex codes in S1 are the widely used Okabe-Ito set and should be checked against Wong's Fig 2 before the palette is frozen.
- The Genie 2 and Genie 3 visuals (read through a fetched summary).
- The DreamerV3 Nature volume and the Genie ICML venue.
- Comment 50 is an observation from the image, not a confirmed bug.


## Lead's response (figure design lead, Opus 5.5, 2026-09-27)

Each entry answers the comment of the same number: "applied" names where, otherwise one line says why not. The shared style is `paper/figstyle.py` (the encoding table, sizes, `refuse_degenerate`); the builder is `paper/make_adapt_figures.py`. Comments 15 to 24 and 46 to 53 are answered in the entries added by the Figure 3b and rollout-composer commits.

1. Applied: `figstyle.style()` sets labels 7 pt, ticks and annotations 6.5 pt; `refuse_degenerate` stops any figure with text under 6 pt, so the 4 and 4.5 pt labels cannot come back.
2. Applied: `figstyle.BACKBONES` is keyed by entity (U-Net `#0072B2` circle, PixArt `#D55E00` square, SD 3.5 `#009E73` triangle, adapter a darker U-Net blue diamond); `SERIES` and `MARKERS` are gone.
3. Applied in the builder (every figure is drawn at its printed size on the 5.5 in page: Figure 4 full width by 1.6 in, appendix panels full or half width); the `main.tex` minipage float belongs to the paper owner, who swaps in `fig4_adaptation.pdf` at `\linewidth` by 1.6 in.
4. Applied: arenas in the seeds, ladder and recipe figures are dark grey lines with the arena number at the right end; no arena uses a backbone colour or marker.
5. Applied: nothing maps D to colour; the per-arena curves use the adapter's blue ramp keyed to the zero-shot skill S0 (the lead's brief; S0 is the quantity the text says orders the arenas), and every other panel is one colour.
6. Applied, and extended by Rohan's direction: the reference reads "training maps (in-distribution)" everywhere, drawn as its 95% episode-bootstrap band (5.06 [4.75, 5.38] dB tuned, 99 episodes, 512 windows) with the dashed point line inside; tables say "training-maps line".
7. Applied: A axes tick every 1 dB from 0, M axes every 0.05.
8. Applied: interval lines are drawn above the markers (zorder 4) and the small multiples carry episode bands.
9. Applied: direct labels at line ends everywhere except the recipe figure, whose five curves lie within 0.1 dB of each other, so it keeps one frameless legend.
10. Applied: no legend or label carries provenance; `ROW_NAMES` is removed.
11. Applied: the M axis reads "M, LPIPS difference (lower is better)" with the zero line labelled "copy-last".
12. Applied: `figstyle.save` refuses an empty panel, a curve at a single x value, a legend entry matching no drawn series and sub-6 pt text; the builder also stops drawing seed, ladder and recipe figures whose runs lack the decoder's rows, and the stale tuned seeds, ladder, recipe and tercile files are deleted.
13. Applied: `pdf.fonttype` 42 stays in `style()`, and the tests assert no Type 3 font in any figure; the TikZ family is the method-figure worker's, checked when that commit reaches main.
14. Partly applied: the builder writes `fig4_adaptation`, and the new figures are `fig3b_zero_shot_paired` and `fig2_rollouts`; renaming the `\figslot` files and labels in `main.tex` is the paper owner's edit.
25. Applied: every arena number is 6.5 pt with a hand offset table for the colliding pairs (7/13, 1/12/14).
26. Applied: one marker and colour (the adapter's diamond) with each arena's episode interval; no D colouring.
27. Not applied here: the lead's brief keeps S0 on this appendix panel; the Δ version is the family-step worker's `fig2e_deficit`.
28. Applied: the budget panel is dropped; the attainment curve in Figure 4b replaces it.
29. Applied with 28: the phrase no longer appears on an axis.
30. Applied: the ramp is keyed to S0 (light is high skill) and arenas 7, 9 and 12 are labelled at their right ends.
31. Applied: the 13 half-gap lines are gone from the curves panel; each arena's line appears only in its own small multiple.
32. Applied: the crossing and censoring markers are gone from the curves panel; crossings are ticks in the small multiples and on Figure 4b.
33. Applied: the colour bar is deleted.
34. Partly applied: the grid follows the rows, so `--headline-variant g8k` switches the headline set and the budget to the 8k grid (ticks 0, 50, 250, 1k, 8k, minor ticks at every other grid step); the full fine-tune line waits for its runs (Monday).
35. Applied: step 0 sits on its own spine segment left of a gap (`figstyle.step_axis`), not under a glyph.
36. Applied: A runs from 0 (copy-last) with ticks at 0, 1, 2, 3, 4, 5 dB.
37. Applied: the tercile figure is no longer built and its files are deleted; the D null belongs in the caption as the Spearman with its interval.
38. Applied: under the tuned decoder no seed-1 rows exist, so no seeds figure is drawn; the stock seeds figure draws only arenas with two scored seeds.
39. Applied: the differences are points only with the zero line; the seed spread at the budget is two dotted lines at plus and minus 0.058 dB, not a grey band, because the grey band means the training maps in the encoding table.
40. Applied: arenas are dark grey, seed 0 solid and seed 1 dashed, labelled at the right ends.
41. Applied: a ladder needs two rungs with the decoder's rows; the tuned ladder is not drawn.
42. Applied: the axis reads "adaptation episodes (log scale)", "step 0" is labelled once, and arena labels are 6.5 pt at the line ends.
43. Applied: the tuned recipe figure is not drawn; the legend check refuses a listed series with no line.
44. Applied: the panels share y, the titles are gone (panel letters a and b, "arena 12" as a small label inside the panel), and D is not shown.
45. Applied: the axis says "(log scale)" and every grid step has a tick, labelled at 0, 50, 250, 1k and 8k.
54. Not mine: `\prov` is the paper's macro in `main.tex`; the generated tables use it only under `--prov`, so the submission build omits the flag.
55. Not mine: Table 1 is hand-set in `main.tex`; for the paper owner.
56. Not mine: Table 1, as above.
57. Not mine: Table 2 is hand-set in `main.tex`; for the paper owner.
58. Not mine: Table 2, as above.
59. Applied to the generated tables: every numeric column is right-aligned (`adapt_cost.tex`, `adapt_perarena.tex`, and `adapt_table3.tex` from the Figure 4 commit).
60. Deferred: the method figure is the other worker's; I check it against section 5 when its commit reaches main.
61. Deferred: the training-curve panel is the other worker's (`tools/training_curves.py`); same check.

### Figure 4 and the statistics decision (beyond round 0)

Built from `per-arena-statistics-decision-2026-09-27.md`: `fig4_adaptation` is (a) the interquartile mean of A across the 13 arenas with a nested bootstrap band (10,000 draws; arenas, then each arena's held-out episodes, paired across steps) over the faint per-arena curves, and (b) cumulative attainment of the half-gap line, one minus Kaplan-Meier, with its band, the at-risk row (13, 13, 9, 8, 6, 4), per-arena crossing ticks with arena numbers (open ticks right of 4k for the four censored arenas) and the fixed 3.5 dB threshold as a thin dotted curve. The appendix adds `figA_adapt_arenas` (13 small multiples with episode bands and a key in the empty slot) and `figA_adapt_profiles` (0, 500 and 4k updates). `adapt_summary.json` gains `across_arenas` and `home_ci`; `adapt_table3.tex` is Table 3's tabular with arena-bootstrap intervals in brackets, the share of the gap closed (0.58 [0.49, 0.74]), the training maps' A with its episode interval (5.06 [4.75, 5.38]) and the median budget with its censoring ("1,000 [250, >4,000]; 4 of 13 censored"). The median-budget interval runs past 4k because 5 percent of the nested resamples have a censored median; the main view's "[250, 2k]" was the interval of the finite medians only. No "tie" wording remains: the summary counts `margin_lower_by_budget` (11) and `margin_within_001_by_budget` (2).

### Comments 15 to 24 (Figure 3b, `fig3b_zero_shot_paired`)

The body panel is now `fig3b_zero_shot_paired` (3.3 x 1.9 in): unseen arenas as columns in the order of the U-Net's zero-shot skill S0 (decision item 6), A on the top row and M on the bottom row, one colour and marker per backbone, stock decoder open and tuned decoder filled on the same windows with a thin grey segment between them, episode intervals, the adapter at 4k as a filled diamond on the A row, and the training maps' reads in a first column under the grey band. `fig2a` and `fig2b` stay as one-decoder appendix variants in the same S0 order.

15. Applied: A at 4k now has its own marker and sits only on the A row of Figure 3b, and the y label no longer says A0; in `fig2a` the axis reads "A (dB)".
16. Applied: each backbone's own training-maps value appears in the training-maps column, in its own colour and fill. I drew these as points rather than the short rules the brief asked for, so the open/filled decoder encoding carries over.
17. Partly applied: the training maps get a column, but it holds the pooled read (`home_<row>/val`), not maps 2 to 5 separately. The per-map training reads are not in `results/fresh_rescore`, and that blocks the per-map rows.
18. Applied: arenas are ordered by S0 on a categorical axis, and D appears nowhere on it.
19. Applied: the backbones sit at fixed offsets inside each arena column, and each stock/tuned pair shares one x, so no two markers of one arena overlap.
20. Applied: SD 3.5 has only a stock read, so it is drawn open and no tuned marker is invented. The caption states the decoder per backbone.
21. Applied: the M row is labelled "M (LPIPS)" with "lower is better", and no adapter M is drawn there.
22. Applied: M ticks are every 0.1 in Figure 3b and every 0.05 in `fig2b`, both including values below zero.
23. Applied: the zero line is labelled "copy-last" on both rows.
24. Applied: M is the bottom row of Figure 3b and shares the arena columns with A.

### Comments 46 to 53 (Figure 2, `tools/compose_rollouts.py`)

The composer reads the steward's export (`results/figure2_rollouts/<map>_<control>/<row>_<decoder>/tic_NN[_scene].png` with `manifest.json`) and writes `paper/figures/fig2_rollouts.pdf` and `.png` at 5.5 in wide. On the synthetic fixture the figure is 1.72 in tall with frames 0.33 in wide. Two controls side by side and seven columns each do not fit the standard's 0.39 in frames inside 5.5 in; `--stack` stacks the two controls instead and doubles the frame width, which is Rohan's open decision 3.

46. Applied: row labels sit at the left, group labels ("map 2 (training)", "arena 7 (unseen)") are rotated beside them, tic numbers run above the first row, and each block carries its held control as a header.
47. Applied: 2 pt white gutters separate the frames, and the scene crop removes the HUD bars.
48. Applied: tic 0 is the last context frame, copy-last's prediction at every tic, so the copy-last row is dropped and a 4 pt gap separates the context column from the predictions.
49. Applied: every frame is the scene crop (rows 0 to 207). Model rows use the tuned decoder (`--decoder`) and true rows use the raw frame (`--truth`).
50. Answered, no data change: the steward confirmed that the row was tic 0 of the same window. The apparent mismatch came from the first column being tic 4 and from the stock decoder's HUD digits.
51. Applied: the columns are tics 0, 1, 2, 4, 8, 16 and 32 (`--tics`).
52. Applied: map 2 and arena 7 each get their own true row, in two row groups (true, U-Net; true, zero-shot, LoRA 4k).
53. Partly applied: `fig2_rollouts.json` records each window's episode, start tic, files and printed PSNR for the caption. The seeded rule that chose the windows is the steward's to state.
