# Figure 1 candidates (Sep 29 2026)

Three redesigns of the teaser (`fig_teaser_B.pdf`), one reference rebuild of today's figure, and a sheet that shows
them all at their true relative size (`fig1_compare.png`, 200 dpi, every figure 5.5 in wide). Nothing here replaces
`fig_teaser_B.*`; the pick is Rohan's.

| file | concept | printed size | frame width | vs today's height |
|---|---|---|---|---|
| `fig_teaser_B.pdf` (today) | two rows, seven frames, context frames in both blocks | 5.5 x 1.33 in | 0.65 in | |
| `fig1_today_sharp.pdf` | today's figure rebuilt after the resolution fix, nothing else changed | 5.5 x 1.33 in | 0.65 in | same |
| `fig1_vA.pdf` | two rows, five frames, no context, outlines, adaptation arrow | 5.5 x 1.50 in | 0.81 in | +0.17 in |
| `fig1_vB.pdf` | time strip of one action on the unseen map | 5.5 x 1.80 in | 0.72 in | +0.47 in |
| `fig1_vC.pdf` | one moment at hero size, training-map reference at half size | 5.5 x 1.07 in | 1.18 in (reference 0.55 in) | -0.26 in |

Today's PDF is 1.33 in tall; the `\figslot` in `main.tex` allows 1.5 in and keeps the aspect ratio, so it prints at
1.33 in. `fig1_vB` needs that slot raised to 1.8 in, or LaTeX scales it to 83 percent. `fig1_vA` fits the slot as
it is. `fig1_vC` prints at 1.07 in in the same slot.

## The first finding: the frames were embedded at 100 dpi

Today's PDF embeds every 320 x 208 scene crop as a 65 x 43 pixel image (`pdfimages -list`). Matplotlib's PDF backend
resamples any `imshow` with a smoothing interpolation (`lanczos` in `compose_teaser._place`) to the figure's 100
dpi. Most of "the masonry is hard to see" came from this, not from the layout. Commit `8749b26c` switches
`_place` to `interpolation="none"`, which embeds the frames' own pixels (about 490 ppi at today's size), and adds
`test_frames_are_embedded_at_their_own_resolution` to `paper/fixtures/test_compose_teaser.py`. That test failed on
the old code with 95 x 62 pixel embeds and passes now. `fig1_today_sharp.pdf` is today's exact command after that
fix: the same layout, numbers and frames. It is the minimal-change option. The candidates embed native frames as well,
at 272 to 580 ppi. 272 ppi is the hero frames of C, and it is the data's own resolution, so upsampling to 300 would add
nothing.

`tools/compose_rollouts.py` has the same `lanczos` call: `fig2_rollouts.pdf` embeds 70 x 46 pixel frames. That file
is out of this task's scope and was left unchanged.

## What the inspiration figures do

GameNGen (arXiv 2408.14837):
- **Figure 1** is one large game frame and a one-line caption. The picture carries the claim.
- **Figure 4 (auto-regressive drift)** has two rows of five abutting frames, one row per condition, sampled every
  10th frame. The caption explains the rows.
- **Figures 5 and 8 to 11 (predictions against ground truth)** show the context apart from the rollout, and put the
  ground-truth row directly under the prediction row, frame for frame.
- **Figure 12 (decoder fine-tune)** has three columns: frozen, fine-tuned and ground truth, left to right.

MultiGen (arXiv 2603.06679):
- **Figure 1** has two headed blocks joined by a large arrow. Short sequences inside it are joined by small arrows,
  and inset boxes have leader lines.
- **Figure 3** labels its rows on the left and shows abutting frames as a strip over time, with the actions drawn
  under the frames.
- **Figure 4** gives every frame an outline in its row's colour (red for player 1, blue for player 2). Words of the
  same colour in the header sentence tie the text to the outlines. The colour is the only cue.

## The encoding shared by A, B and C

- **Zero-shot frames** have a dashed outline in orange `#E69F00`. **Adapted frames** have a solid outline in the
  adapter's dark blue `#00466E`. Ground truth has no outline.
  - The outline idea is MultiGen Figure 4's. Unlike MultiGen, the dash gives a second cue, so the pair survives
    greyscale: light dashed against dark solid, checked on a greyscale render. Both colours are Okabe-Ito.
  - The outline sits in the gutter, outside the picture, so it stays visible on dark and light frames and hides no
    pixel.
- **Scene PSNR** is a boxed number in each model frame's lower-left corner, next to the frame it scores. A tag of the
  same style reads "scene PSNR (dB)" and serves as the key.
- **All text** is 6.5 pt Arial, embedded TrueType (`pdffonts`), with no Type 3 fonts. Today's figure prints its
  numbers and "scene PSNR (dB)" at 6.0 pt, below this brief's floor.
- **Numbers** are read from the restart manifests (`scene_psnr_vs_truth_raw`) by the builder, which stops if a drawn
  number is missing. The +16 numbers match `fig_teaser_B.json`: forward 23.7, 13.7, 22.4; attack 21.0, 14.3, 21.9.

## Candidates

### A: two rows, no context, adaptation arrow (`fig1_vA.pdf`, 1.50 in)

A keeps today's story whole: both actions, the training-map block and the unseen-map block.
- **What changed:**
  - The two context columns are dropped. They took a quarter of the width and carried no comparison.
  - Frames grow from 0.65 to 0.81 in wide, 1.55 times the area.
  - Zero-shot and adapted get the outlines.
  - PSNR moves into the frame corners.
  - One arrow between the two rows runs from the zero-shot column to the adapted column, labelled "8 episodes, 4k
    updates". The adaptation cost is now in the figure.
- **Inspiration:** MultiGen Figure 4 for the outlines. MultiGen Figure 1 for the arrow joining two states. GameNGen
  Figure 12 for the three columns side by side, one role per column, with ground truth last.
- **What it gives up:** the context frames, so the reader no longer sees where each rollout started, and 0.17 in of
  height against today's printed figure.

### B: time strip on the unseen map (`fig1_vB.pdf`, 1.80 in)

B shows one action (forward, map 7 episode 210 tic 217) as it unfolds.
- **Layout:**
  - Rows are zero-shot, adapted and ground truth.
  - Columns are +1, +2, +4, +8 and +16 tics after the context.
  - One context frame at the left feeds all three rows through a fan of arrows.
- **What it shows:** the zero-shot model loses the map within one to two tics. At +1 the ceiling has already turned
  to stone slabs, at +2 the walls are grey stone (18.2 dB). The adapted model holds the map for all 16 tics (21.1 to
  25.3 dB across the strip).
- **Inspiration:** GameNGen Figure 4 for rows as conditions over a sampled time axis with index labels. GameNGen
  Figures 8 to 11 for the context kept apart and the ground-truth row under the predictions. MultiGen Figure 3 for
  row labels at the left of a strip. MultiGen Figure 1 for small arrows between frames.
- **What it gives up:** the training-map block and the attack row. B does not show that the model is fine in
  distribution. It is also the tallest candidate: +0.47 in, and the `\figslot` height must change.
- **New frames:** the +1, +2, +4 and +8 frames are new to the paper. They come from the same restarted rollout as
  today's +16 frame, which was chosen for its large adaptation gain. No tic was picked by eye, and the strip is the
  fixed doubling sequence plus +1.
- **Height:** B stays under the 1.8 in cap. At 2.0 in (`--height-b 2.0`) its frames would grow to 0.83 in. I did not
  judge that gain clear enough to spend another 0.2 in.

### C: one moment at hero size (`fig1_vC.pdf`, 1.07 in)

C is the forward moment only.
- **Layout:**
  - Zero-shot, adapted and ground truth at 1.18 in wide, 1.8 times today's width and 3.3 times the area.
  - The arrow sits in the gutter between zero-shot and adapted, labelled "8 episodes, 4k updates".
  - At the left, the training-map reference for the same button (map 5 episode 6059 tic 239) shows its prediction
    over its ground truth at half size.
- **What it shows:** at this size the wrong map is plain without a zoom inset. The zero-shot frame is grey stone
  blocks and a cobbled floor, where map 7 has white panels with a blue stripe and a brown floor.
- **Inspiration:** GameNGen Figure 1 for one large frame carrying the claim with little text. MultiGen Figure 1 for
  the arrow between two states and the headed blocks. MultiGen Figure 4 for the outlines. GameNGen Figure 12 for the
  column triad.
- **What it gives up:**
  - The attack row.
  - The context frames.
  - The italic action label. The sidecar keeps `"action": "forward"`.
  - A full-size training-map block: the in-distribution reference is 0.55 in wide, smaller than today's frames.
- **Height:** 0.26 in shorter than today, about two lines of text back on a page that is over its limit.

## Recommendation

**Take C, with the resolution fix in any case.** Rohan's short caption makes one claim: poor zero-shot on an unseen
map, close to ground truth after adaptation. C draws that claim at the largest size of the three, and it is the only
candidate that gives page space back. What C drops mostly did not carry the claim in single frames:
- **The attack row.** The claim that the model keeps its control response rests on the directional score, which
  still-frame thumbnails cannot show in any layout.
- **The training-map block.** C keeps its evidence as the half-size reference.

If Rohan wants both actions in Figure 1, take A. It costs 0.17 in over today.

B is the most informative panel about dynamics: the map is lost within two tics. It would serve better as a Results
or appendix figure than as the teaser, because of its height and because it has no training-map reference.

If none of the three is wanted before the deadline, `fig1_today_sharp.pdf` is a drop-in improvement with no layout
change.

## Not built, and why

- **A zoom inset on the wall texture.** The wrong texture covers the whole zero-shot frame, not one patch. At
  native resolution and C's size it reads without magnification. In A and B an inset would cover a quarter of a
  0.7 to 0.8 in frame.
- **A map-2 reference frame.** The zero-shot frames look like training map 2: dark stone blocks and a cobbled floor.
  A map-2 ground-truth thumbnail beside them would make "it draws the training maps' masonry" visible. Choosing a
  look-alike frame is a selection the paper would have to disclose, so it is left as an idea for Rohan.

## Rebuild

From the repo root, with the restart export at `results/teaser_restart` (the main worktree's copy is read-only input;
pass its absolute path):

    python3 tools/teaser_v2.py --restart-root results/teaser_restart --out-dir paper/figures/candidates \
        --compare "today, as in the paper (fig_teaser_B.pdf)=paper/figures/fig_teaser_B.pdf" \
                  "today, rebuilt with frames at native resolution=paper/figures/candidates/fig1_today_sharp.pdf"

`fig1_today_sharp.pdf` is today's command (`paper/FIGURES.md`, "Figure 1 provenance", plus `--simple`) run after
commit `8749b26c`, with its output `fig_teaser_C_hold16_pick_restart_nopersistence_simple.{pdf,json}` renamed. Its
PNG is a 300 dpi `pdftoppm` render. Each candidate's JSON records its moments, tics, every number drawn, the frame
files, positions and sizes, the encoding, the exact command and the git head.

Tests: `python3 -m pytest paper/fixtures/test_compose_teaser.py paper/fixtures/test_teaser_contact_sheet.py -q` (23
pass: the 22 existing tests plus the resolution test).
