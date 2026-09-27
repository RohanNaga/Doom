# Figure standards for the DoomDiT workshop paper

Status: round 0, 2026-09-27. Owner: the figure owner named in `RESEARCH_CONTEXT.md`. The per-figure review of the current PNGs is `paper/review_figures_round0_opus.md`; that file also holds what each reference paper's figures actually do.

## 0. The page we draw for

- CoRL 2026 is **single column**: `corl_2026.sty` sets `textwidth=5.5in`, `textheight=9in` (lines 87 to 88). No two-column rules apply. Every body figure is either full width (5.5 in) or a `minipage` share of it. There is no "column width".
- Body text is Times (`ptm`, 10 pt); `\small` and `\footnotesize` are 9 pt; `\scriptsize` is 7 pt (sty lines 125 to 229). Captions use the article default, 10 pt.
- Figures are drawn at their printed size and included at scale 1 (`\figslot` already does `keepaspectratio`; the PDF page size must equal the slot).
- Budget: the four body figures and captions take about 8.5 of the 36 body inches. Heights below are ceilings.

## 1. The encoding table (one meaning per colour and marker, in every figure and in the method figure)

The palette is Okabe and Ito's colour-universal-design set as given by Wong (2011). Colour always carries a second cue (marker shape, line style or a direct label), so a greyscale print still reads.

| Entity | Colour | Marker or line | Where it appears |
|---|---|---|---|
| True frames | black `#000000` | row label "true" | Figure 2 rows, Figure 3c |
| Copy-last (persistence) | no colour of its own | the zero line, 0.6 pt black, labelled "copy-last" at its right end; in rollouts the tic-0 column | every plot of A or M; Figure 2 |
| Training maps | grey band `#E8E8E8` behind their points; reference line 0.6 pt dashed `#707070`, labelled "training maps" | band plus line | Figures 3 and 4 |
| U-Net (SD 1.4) | blue `#0072B2` | circle | Figures 1, 3, 4, Tables |
| PixArt-α | vermillion `#D55E00` | square | Figures 1, 3 |
| SD 3.5 Medium | bluish green `#009E73` | upward triangle | Figures 1, 3 |
| U-Net + LoRA adapter | the U-Net's blue family | diamond; curves in a single-hue blue ramp keyed to the zero-shot deficit Δ (light = small Δ, dark = large Δ) | Figures 2 (row), 4 |
| Full fine-tune | black | dashed line (3, 2), 1.0 pt | Figure 4a |
| Stock versus tuned decoder | same hue as the backbone | open marker = stock, filled = tuned, a thin grey segment joins the pair | Figure 3b, appendix |
| Named arenas (7, 9, 12) | never a backbone colour | dark grey line plus a direct text label | Figure 4a, appendix |

Unused: sky blue `#56B4E9`, orange `#E69F00`, yellow `#F0E442` (too light for thin lines), reddish purple `#CC79A7` (spare). An ordered quantity (D, Δ, updates, episodes) gets one hue ramp or position, never a categorical set. An unordered set (backbones) gets categorical colours. A quantity the paper says orders nothing (D among unseen arenas) is never mapped to colour.

## 2. Type, lines, marks

- One sans family in every figure, including the TikZ method figure: Helvetica or Arial (`\sffamily` in TikZ with the sty's `phv`; Arial in matplotlib). Math symbols in figures use the same family's italic (A, M, S, D, Δ), as the matplotlib style already does.
- Sizes at print scale: axis labels 7 pt, tick labels 6.5 pt, annotations and direct labels 6.5 pt, nothing below 6 pt, panel letters 8 pt bold lowercase upright (a, b, c) at the top left. Nature's range is 5 to 7 pt at 89 mm; our 10 pt body text calls for the upper end.
- Lines: data 1.0 pt, reference lines 0.6 pt, axes and ticks 0.6 pt, tick length 2.5 pt, nothing thinner than 0.5 pt. Markers 3.5 to 4 pt, white edge 0.4 pt only where points overlap.
- Axes: left and bottom spines, ticks outward, 3 to 5 ticks at round values (0, 1, 2 dB; −0.05, 0, 0.05), units in parentheses in the label, "(log scale)" when the axis is logarithmic. No gridlines, no plot titles, no boxed legends.
- Uncertainty: 95% episode-bootstrap intervals as thin lines in the series colour, drawn above the marker and longer than it, or stated in the caption as "intervals are smaller than the markers". State n (maps, episodes, windows) in the caption.
- Show every point when n ≤ 30 (13 arenas, 17 maps): dot plots, not bars (Weissgerber et al. 2015).
- Images: no borders, 2 pt white gutters between frames, row labels at the left rotated 90°, time labels above the first row only, the HUD cropped to the scored scene rows (0 to 207) unless the figure is about the HUD.

## 3. Captions

First sentence states the finding in plain words ("The advantage over copy-last drops off the training maps for all three backbones, and among unseen arenas it does not follow D."). Then how to read: what each mark means, n, the interval, the checkpoint, the decoder, one tic or four. No legend text repeated in the caption and no caption text repeated in the figure. Provenance (170k, provisional) lives in the caption, never in the legend.

## 4. Per-figure specifications

### Figure 1: method overview, with a training panel

- Size: full width, 5.5 × 1.9 in. Left 3.9 in the diagram, right 1.5 in the panel (b).
- (a) One reading line, left to right: real Doom frames with the 19-button control stream; "4 training arenas" and "13 unseen arenas" as two stacks of real map thumbnails; the three backbones as outlined boxes in their encoding colours (the diagram's only colour) under one bracket naming the shared recipe; the decoder tune on the rendering path; the rank-16 adapter on the U-Net only; evaluation as three words (A, M, directional). Straight 0.6 pt black arrows, square corners, no fills, no icons.
- (b) Advantage over copy-last on training-map validation windows (dB) against updates (0 to 200k), one line per backbone, the zero line labelled "copy-last". Not validation loss: the backbones predict in different latent spaces, so their losses share no scale.
- Legend: none. The three boxes in (a) are the legend for every later figure; (b) labels its lines at their right ends.
- Caption states what the paper built (the benchmark, one recipe across three backbones, the adapter) and that (b) shows every backbone ends above copy-last at home.
- Mistakes to avoid: icons, pastel zone fills and rounded "stage" pills (section 5); a loss panel across incomparable latent spaces; a diagram that introduces colours other than the three backbones.

### Figure 2: rollouts under held controls

- Size: full width, 5.5 × 1.75 in.
- Layout: two column blocks, one per held control (turn left, forward), each with a tic-0 column (the last context frame, which is copy-last's prediction at every tic) and tics 1, 2, 4, 8, 16, 32 (doubling, as DIAMOND spaces its columns). Frames about 0.39 in wide.
- Rows, in two groups with a 4 pt gap: map 2 (true, model); arena 7 (true, zero-shot, after 4k adapter updates). Copy-last gets no row: six copies of one frame carry no information, so the tic-0 column carries it.
- Labels: row labels at the left ("true", "U-Net", "U-Net + LoRA 4k"), group labels "map 2 (training)" and "arena 7 (unseen)", block headers "held turn left", "held forward", tic numbers above the top row.
- Decoder: the tuned SD 1 decoder for every model row, the raw frame for true rows; scene rows only.
- Caption: finding first (on arena 7 the camera turns the right way while walls and floor drift toward the training maps' grey brick; 4k adapter updates restore the arena's colours), then the window rule, episode and start tic.
- Mistakes to avoid: one "true" row shared by two different maps (the plan's row list implies this; each map needs its own truth); uneven tic spacing; frames abutting with no gutters, so HUD bars read as rules.

### Figure 3: the shift

- Size: full width, 5.5 × 1.8 in. (a) 2.0 in wide, (b) 2.4 in, (c) 1.0 in.
- (a) x: D (unitless, linear, 0 to 0.3). y: zero-shot S or A (dB), whichever the text names. 17 maps × 3 backbones. A grey band marks the train-versus-train floor (0.02 to 0.09), labelled. Per backbone, one horizontal segment at the training-map median and one at the unseen median across its D range: the step drawn as a step. No fitted line.
- (b) Cleveland dot plot for the U-Net: one row per map, the four training maps on top under the grey band, then the 13 arenas sorted by tuned A. Two sub-panels share the y axis: A (dB) and M (LPIPS difference, zero line "copy-last", lower is better). Stock open, tuned filled, grey segment between.
- (c) One latent through the stock and tuned decoders beside the raw frame, stacked, headed "stock", "tuned", "true", HUD kept (the decoders differ most there).
- Caption: finding first (a step at the family boundary, flat beyond it; the tuned decoder raises A but does not close the gap).
- Mistakes to avoid: ordering arenas by D on a categorical axis, which reads as a slope the text denies; one "home" line in one backbone's colour for all three; adapter points in this figure (they belong to Figure 4).

### Figure 4: adaptation

- Size: full width, 5.5 × 1.6 in. (a) 3.3 in, (b) 2.0 in.
- (a) A (dB) against adapter updates, log scale 50 to 8k, with step 0 on a split axis at the left. 13 thin lines in the blue ramp keyed to Δ, the full fine-tune dashed black (per arena, same grid), the training-maps reference line. Three arenas labelled at their right ends (the far arena 7, the censored 12, arena 9 at the line). No per-arena half-gap lines; crossings live in Table 3.
- (b) A at 8k (dB) against the zero-shot deficit Δ (dB), 13 points in the same ramp colours with arena numbers beside them at 6.5 pt, the training-maps line; Spearman ρ with its interval in the caption. No regression line.
- Caption: finding first (most of the gain lands within 250 updates; the endpoint is set by the arena's zero-shot deficit, not by D), then n, weights (live), seed, decoder.
- Mistakes to avoid: a colour bar for D; open and filled markers whose meaning the figure never states; per-arena horizontal lines that outnumber the curves.

### Tables 1 to 3

- `booktabs`, no vertical rules, `\footnotesize`, units and direction in the header ("PSNR (dB) ↑", "LPIPS ↓"), fixed decimals per column (2 for dB, 3 for LPIPS), numbers right-aligned, the persistence row first and labelled as the zero line.
- Table 1: persistence, U-Net, PixArt, SD 3.5; one tic, four tics, 256 tics, directional. Bold nothing the intervals do not separate.
- Table 2: per backbone a training and an unseen row; `\cmidrule` groups "stock decoder" and "tuned decoder" over A and M; counts as "13/13".
- Table 3: censored as ">8k"; medians censored when fewer than 7 cross.
- Mistakes to avoid: mixing full-frame and scene-row numbers in one table; provisional numbers without a mark; a "best" bold that the intervals do not support.

## 5. Machine-made against authored

Sources: Nature's "we avoid" list (decorative icons, drop shadows, patterns, gridlines, text over busy images, coloured text); Rougier et al. rules 5, 6 and 8; PaperBanana's auto-summarised style guide (arXiv 2601.23265, App. F), which names pastel zone fills at 10 to 15 percent opacity and rounded containers as the 2025 look that automated generators copy; SciDraw-Bench's failure taxonomy (arXiv 2606.28406). Pairing each tell with a replacement is my judgement.

| Tell | Replacement |
|---|---|
| Icons and clip art (robot heads, flames and snowflakes for trained and frozen, globes) | Real frames as the data element; the words "frozen" and "trained" |
| Pastel zone fills, rounded "Step 1" pills, drop shadows, gradients, 3D | Outlined square boxes, whitespace to group, one accent per entity |
| Emoji, decorative or curved arrows, arrows that do not carry data flow | Straight 0.6 pt arrows only where data moves |
| Several fonts, handwriting or pixel fonts, bold everywhere | One sans family, two sizes, bold only for panel letters |
| Titles inside plots, boxed legends, a legend that repeats the caption | Panel letters; direct labels at line ends; the caption says how to read |
| Many colours, a rainbow colour per point, colour without meaning | The encoding table; grey for context, colour for the entity under discussion |
| Unlabelled axes, missing units, odd ticks (0, 1.5, 3.0, 4.5) | Label (unit), round ticks |
| Value labels on every bar or point | The axis carries values; label only the points the text names |
| Mini-plots inside a method figure that cannot be read at print size | Results in results figures |
| Legend entries for series that are not drawn; empty panels | The build fails when a series or panel has no data |

## 6. The standing checklist (run before every version)

Build and page
1. The figure PDF's page size equals its slot; it is included at scale 1.
2. `pdffonts` shows only embedded TrueType or Type 1 fonts, no Type 3.
3. Rendered in the paper PDF at 100% zoom and printed once; every label readable.
4. Greyscale print (or a deuteranopia simulation) still separates every series.

Content
5. The figure has one message, and the caption's first sentence states it.
6. Every panel, series and legend entry has data; nothing is a placeholder.
7. Each number drawn matches the table or text that reports it (spot-check three).
8. n, interval, checkpoint, weights (live or EMA), decoder, horizon and scene or full frame are in the caption.

Encoding
9. Colours and markers follow the encoding table; no colour means two things across the paper.
10. Copy-last is the zero line wherever A or M is plotted, labelled once.
11. Training maps are the grey band or the dashed reference line, labelled "training maps".
12. No colour encodes D.

Type and axes
13. One sans family; labels 7 pt; ticks 6.5 pt; nothing under 6 pt; panel letters 8 pt bold lowercase.
14. Axis labels carry units; ticks at round values; log axes say so.
15. No titles inside plots, no gridlines, no boxed legends, no value labels the text does not use.

Images
16. Row and time labels present; gutters between frames; the HUD cropped unless it is the point.
17. The tic-0 frame equals the last context frame (check one pixel row against the source).

Decoration
18. No icons, emoji, pastel zones, rounded pills, shadows, gradients or 3D.

## 7. Review protocol

1. The owner renders the paper PDF and freezes a version tag (`fig-rN`).
2. Two reviewers work independently on that PDF, one Claude reader and Astra, neither seeing the other's list. Each writes numbered comments `R<round>.<reviewer>.<n>`: figure, panel, location, the change, the rule from this file it rests on. Every comment names an edit. "Improve" or "clarify" does not count.
3. The owner merges both lists into one table: comment, reviewer A verdict, reviewer B verdict, owner action.
4. Only comments both reviewers accept are applied. A comment one side rejects goes to Rohan with both reasons, and he rules. Nothing else is changed in that round.
5. Each applied comment cites its commit. The owner reruns the checklist and tags `fig-rN+1`.
6. The round log lives beside `paper/REVIEW_LOG.md`.

## 8. Sources

Checked on 2026-09-27. "Opened" means the raw page or PDF was read, not a summary.

Guidance
- Rougier, Droettboom and Bourne, "Ten simple rules for better figures", PLoS Comput Biol 10(9): e1003833 (2014), https://doi.org/10.1371/journal.pcbi.1003833. Opened: the ten rule texts.
- Weissgerber, Milic, Winham and Garovic, "Beyond bar and line graphs: time for a new data presentation paradigm", PLoS Biol 13(4): e1002128 (2015), https://doi.org/10.1371/journal.pbio.1002128. Opened: abstract and Fig 1 text.
- Nature, "Preparing figures: our specifications", https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/. Opened: 5 to 7 pt, sans, 8 pt bold lowercase panel letters, units in parentheses, accessible palette, the "we avoid" list.
- Nature, "Final submission", https://www.nature.com/nature/for-authors/final-submission. Opened: 89 mm single and 183 mm double column.
- Science, figure guide, https://www.science.org/do/10.5555/page.2385607/full/author_figure_prep_guide_2022.pdf. **VERIFY**: HTTP 403. Search snippets say 5.5 cm per column, Helvetica, 5 pt minimum. Nothing here depends on it.
- Okabe and Ito, "Color Universal Design", https://jfly.uni-koeln.de/color/. Opened: palette rationale, darker blue and orange for thin lines. The RGB values are in an image; the hex codes follow Wong.
- Wong, "Points of view: Color blindness", Nat Methods 8, 441 (2011), https://doi.org/10.1038/nmeth.1618. DOI resolves; paywalled, not opened (**VERIFY** hex codes).
- Crameri, Shephard and Heron, "The misuse of colour in science communication", Nat Commun 11, 5444 (2020), https://doi.org/10.1038/s41467-020-19160-7. Opened: abstract.
- Tufte, *The Visual Display of Quantitative Information* (1983; 2nd ed. 2001) for data-ink and chartjunk; *Envisioning Information* (1990) for small multiples. **Not opened** (edwardtufte.com returned 403); cited through Rougier rule 8, which quotes Tufte.
- arXiv 2601.23265 (PaperBanana), 2606.28406 (SciDraw-Bench) and 2603.16159 (AI-generated figures: policies and guidelines). Opened: the raw arXiv HTML.

Reference papers (figures viewed as rendered PDF pages; details in the review file)
- GameNGen 2408.14837; DIAMOND 2405.12399; Genie 2402.15391; Vista 2405.17398; GAIA-1 2309.17080; XEWorld 2608.05799; DreamerV3 2301.04104; Cosmos 2501.03575; DreamGen 2505.12705 (CoRL 2025, PMLR 305); DINO-Foresight 2412.11673 (NeurIPS 2025); Ctrl-World 2510.10125 (ICLR 2026, the page header). Blogs: Genie 2 (deepmind.google, 2024-12-04) and Genie 3 (2025-08-05), read through a fetched summary, so **VERIFY** their visuals; Oasis (oasis-model.github.io, Decart and Etched), opened: raw HTML and the architecture and throughput images.

The lessons drawn from these papers are in section 0 of the review file. No reference draws a copy-last line in any rollout or error-over-steps figure; that line is what our figures add.
