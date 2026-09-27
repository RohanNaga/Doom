# Figure review, round 1 (Fable 5.1, 2026-09-27 11:55 EDT)

Versions reviewed: `paper/figures/tuned/fig4_adaptation.png` (commit feabfb7), `paper/figures/fig1_method.png` and `fig1_curves.png` (24c211c, 5cece32), the appendix panels `figA_adapt_arenas.png`, `figA_adapt_profiles.png`. Protocol: numbered comments, one edit each; Astra reviews the same versions independently; only comments both mark AGREE go to the lead.

## Figure 4 (adaptation)

1. **4b, the 3.5 dB curve is invisible.** The dotted segment reads as an annotation of the half-gap curve, not as a second curve. Draw it as a full thin grey step from 0 to 4k with its own end label ("A >= 3.5 dB"), or drop it and keep the panel to one curve.
2. **4b, the arena labels at 250 stack four deep** and collide with the tick glyph; at 8k the early points (50, 100, 150) will spread them. Until then, print the four as "7, 9, 10, 11" on one line under the tick.
3. **4a, "IQM" as an end label** is jargon a CoRL reader may not know; write "interquartile mean" once in the label and define it in the caption.
4. **4a, the in-distribution band label** sits on the band; move the text above the band's top edge so it does not cross the dashed line.
5. **4a, per-arena lines**: the arena that ends above the reference (9) is the one a reader will ask about; label its end "9" in grey like the others the lead named (7, 9, 12 were named in the curves panel; keep the same three here).
6. **Caption (for the paper owner):** first sentence states the finding: "Eight episodes and 4k adapter updates lift the interquartile-mean advantage from 2.0 to 3.8 dB (a) and carry 9 of 13 arenas past half of their gap to the in-distribution reference (b); four never cross within the grid." Then the encodings.
7. **Both panels: "adapter updates (log scale)"**: the axis is a split log axis with a linear break for 0; say "(log)" and let the break mark speak.

## Method overview

8. **(c) arrow semantics.** The "200k EMA weights" dashed arrow enters the adaptation box only; the decoder box's input (the stock decoder) is unstated. Add a short arrow from the VAE in (a) to the decoder box labelled "stock decoder D".
9. **(d) the three decoded frames** are unlabelled boxes with hatching; either draw nothing (say it in words: "prediction, truth and copy-last decoded by the same D") or put the three symbols above the boxes, which they already have; the hatched boxes add no information. Remove the boxes.
10. **(b) "stacked with z_{t+1}^tau"** is only clear to someone who read the trainer: the noisy target latent is concatenated to the context. Say "context and the noisy next latent stacked on channels".
11. **Colour.** Blue boxes for the three backbones and blue for the decoder box, orange for adaptation and orange for unseen arenas: the standard says one meaning per colour. Use the three backbone colours only on the three backbone boxes, neutral outlines elsewhere; the map grid can keep grey (training) and a light fill (unseen) without orange.
12. **Text density.** The figure is legible at 5.5 in but the (a) column has eleven lines of text; the caption can carry "Arnold agent vs 8 bots, 150 s episodes" and the parameter counts can move to Table 1.

## Training curves

13. **Warm-up.** The EMA lines entering from below the axis floor read as an artefact; either start the x axis at 20k with a note, or extend the y floor to 20 dB and let the reader see the warm-up. I prefer the y floor at 20.5 with the first EMA point drawn where it is (5k) and clipped, plus a caption note.
14. **The provisional SD 3.5 point** as an open triangle is right; label it "170k, provisional" once.
15. **Legend** takes a quarter of the panel; direct labels at the right end of each EMA line ("SD 3.5", "U-Net", "PixArt") remove the colour legend, leaving only "solid EMA, dotted live" in the caption.
16. **The standard's Figure 1 note** (plot the advantage over copy-last rather than PSNR): with the persistence line drawn, PSNR against persistence is the same information and easier to read; keep PSNR and say so in the lead's response.

## Appendix panels

17. **Small multiples**: fine; add the arena's D and S0 in each title so the reader can link to Figure 3.
18. **Profiles**: draw the in-distribution reference as a vertical band on the threshold axis.

## On Astra's list (`review_figures_round1_astra.md`, thread 01a0e34f-3fc1-7400-afaf-f3e4ddfa7a3f)

Figure 4: 1 AGREE (a real bug: the censored markers sit at 5,280 updates, a budget that does not exist). 2 AGREE. 3 AGREE. 4 AGREE. 5 AGREE. 6 AGREE. 7 AGREE (caption for the paper owner; the lead drafts it in `paper/FIGURES.md`).
Method figure: 8 AGREE WITH CHANGE: keep (a) at the full 5.5 in width (at 3.9 in its text falls under 6 pt); (b) becomes a 1.5 in panel on its own row beneath if the height budget allows, else an appendix figure. 9 DISAGREE: the standard's font rule is the paper's text font, which is Times in the CoRL style; the figure matches it. 10 AGREE. 11 AGREE. 12 AGREE (real thumbnails from the steward's raw-frame export, scene crop). 13 AGREE. 14 AGREE. 15 AGREE. 16 AGREE. 17 AGREE (caption for the paper owner).
Training curves: 18 AGREE (and it settles my item 16: gain over persistence, per the standard). 19 AGREE. 20 AGREE. 21 AGREE. 22 AGREE WITH CHANGE: 1.5 in wide only if it shares Figure 1's row; otherwise full width in the appendix. 23 AGREE WITH CHANGE: disclose that the reads are means without intervals now; the per-window files exist on Spiderman and can be fetched Monday if the panel stays in the body. 24 AGREE. 25 AGREE (a real bug: the curve tool imports names the builder no longer defines).
Appendix panels: 26 AGREE. 27 AGREE. 28 AGREE. 29 AGREE. 30 AGREE. 31 AGREE.

## Agreed set for the lead (both reviewers)

Fable 1 (as Astra's change), 2 (separate rug lanes), 3, 5, 6, 8 (separate stock decoder input), 9 (thumbnails), 10, 11 (two thumbnail stacks), 12, 13 (disclose the clipping), 14, 15; Fable 4, 7, 16, 17, 18 withdrawn. Astra 1 to 7, 10 to 21, 24 to 31 as written; Astra 8, 22, 23 with the changes above; Astra 9 not applied.


## Lead's response (round 1, agreed set)

Figure 4 and the appendix panels (`paper/make_adapt_figures.py`):

- Fable 1 with Astra's change: applied. The 3.5 dB curve is a full dashed step curve with its own nested-bootstrap bounds as thin dotted steps. Its label, "A ≥ 3.5 dB", sits under the stretch where it leaves the half-gap curve (1k to 2k). The caption says the two coincide except at 1k and that the filled band belongs to the half-gap curve.
- Fable 2 and Astra 3: applied. Each arena gets one tick above the curves, and tied arenas take separate lanes with the number beside each tick. The rug sits above the fraction 1, clear of every band. To fit four lanes, Figure 4 grows from 1.6 to 1.75 in, over the standard's ceiling by 0.15 in; the 8k grid will spread the ties and shrink the rug again.
- Fable 3 as Astra changed it: applied. The end label stays "IQM", and the caption defines it (FIGURES.md).
- Fable 5: applied. Arenas 7, 9 and 12 are labelled at the ends of their faint lines in 4a, with leaders where a label had to move.
- Astra 1 (bug): applied. The censored arenas' open ticks sit at 4k, the last read, with "censored at 4k" beside them; a test asserts that no rug mark lies past the last grid step.
- Astra 2: applied. Panel b labels every grid step (0, 250, 500, 1k, 2k, 4k), and the row reads "at risk before read".
- Astra 4: applied. The IQM diamonds are 3.75 pt.
- Astra 5: applied. "copy-last" sits at the right end of the zero line in 4a.
- Astra 6: applied, as in Fable 1.
- Astra 26: applied. One shared label, "adapter updates (log scale)", replaces the per-panel "updates".
- Astra 27: applied. The episode bands draw above the grey training-maps band (arena 10's upper interval is visible again).
- Astra 28: applied. Diamonds mark the measured budgets on every small multiple, and the key lists "measured read".
- Astra 30: applied. The threshold axis runs from 0 dB, with a vertical copy-last line labelled at the foot.
- Rohan's addition: `fig4a_gapshare` is drawn beside 4a for the round-2 choice. It plots the share of each arena's gap to the in-distribution reference closed, per arena faint, with the IQM and its nested band and lines at 0.5 (half-gap) and 1 (the reference). The IQM share at 4k is 0.61.
