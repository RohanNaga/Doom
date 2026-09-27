# Figure review, round 2 (Fable 5.1, 2026-09-27 15:05 EDT)

Versions: the design lead's set on main at 39ada73 (`fig2_rollouts`, `tuned/fig4_adaptation` v2, `tuned/fig4a_gapshare`, `tuned/fig3b_zero_shot_paired`, `fig2d_family_step` v2, `fig2e_deficit` v2, `fig1_curves` v2). Numbered, one edit each; Astra reviews independently; agreed items go to the lead.

## Figure 2 (rollouts)
1. Fourteen columns at 5.5 in make each frame 0.33 in wide; unreadable in print. Stack the two controls vertically (turn-left group above forward group) so each frame is about 0.7 in, and cut the tics to 0, 1, 4, 16, 32.
2. The map 2 window is a dark corridor; the reader sees black frames. Use the map 2 fight or strafe window (brighter) for the in-distribution rows, or the forward window only.
3. The per-tic PSNR numbers under the rows are the only numbers on the figure; keep them but grey and 6 pt, and state in the caption that they are scene PSNR against the true frame.
4. Row label "after 8 episodes, 4k updates" is right; add "(adapter)" so the reader knows it is the same model plus the adapter.
5. Caption first sentence: "Under a held control the zero-shot model keeps the camera motion and loses the arena's appearance within four tics; eight episodes of adaptation restore it."

## Figure 4
6. Version 2 is right. One edit: in 4b the rug labels stack four deep at 250; on the 8k grid they will spread, so no change now, but confirm after tonight's rebuild.
7. Height 1.75 in exceeds the standard's 1.6; accept for now, revisit on the 8k grid.
8. The 4a form: I recommend the share-of-gap version in the body (its half-gap line is a horizontal 0.5, the question's answer is the crossing of one line) and the dB version in the appendix; Rohan decides.

## Figure 3b
9. Needs a one-line legend above the panel: U-Net circle, PixArt square, SD 3.5 triangle; open stock, filled fine-tuned; diamond adapter at 4k.
10. The training-maps column shows one pooled point per backbone; label it "pooled" in the caption.
11. If M's sign flips, the lower panel's "lower is better" note goes and the y label becomes "M, LPIPS margin (dB-free)"; otherwise keep.

## Figure 3a (family step)
12. Right as drawn. Caption first sentence: "Every unseen arena lies farther from the training recordings than any training map, and every arena sits below every training map in zero-shot skill for all three backbones; within the arenas, d orders nothing."
13. The U-Net and PixArt training-map medians overlap exactly; say so in the caption rather than offsetting.

## Deficit panel
14. Right as drawn; the label "1, 6, 12" for coincident points is good. Add the Spearman (−0.90) as a small text in the panel's lower left.

## Curve panel
15. The zero line and y label say "copy-last"; the terminology is "persistence" (the second worker's pass).
16. The provisional SD 3.5 open triangle at 170k needs the caption sentence.

## On Astra's list (`review_figures_round2_astra.md`, thread 01a0e372-a2b6-7051-9900-86b2f3716ca8)

Figure 2: 1 AGREE ("LoRA 4k" label; widen the frames). 2 AGREE. 3 AGREE (tic 0 shows the raw context frame in every row). 4 AGREE. 5 AGREE (caption states the tic-32 counterexample).
Figure 4: 6 AGREE. 7 AGREE. 8 AGREE WITH CHANGE: 1.6 in once the rug is compacted; a third variant (plain curves plus a per-arena dot plot, Rohan's request) is being drawn and may replace panel b. 9 AGREE (a bug). 10 AGREE.
4a gap share: 11 AGREE. 12 AGREE.
Figure 3b: 13 AGREE (the adapter's per-arena endpoint lives in Figure 4's dot panel). 14 AGREE (Rohan's absolute form, computed from the raw scene fields). 15 DISAGREE: the three-backbone comparison is the panel's purpose and Table 2's rows; keep three backbones, arena-number order, one band per arena, five marks (persistence bar, three models, no adapter per item 13, so four). 16 AGREE. 17 AGREE.
Family step: 18 AGREE WITH CHANGE: "zero-shot latent skill (dB)" with the definition in the caption; "ΔPSNR" for a latent quantity misleads. 19 AGREE. 20 AGREE (caption as point estimates; the episode intervals are Monday work). 21 AGREE.
Deficit: 22 AGREE. 23 AGREE. 24 AGREE.
Curves: 25 AGREE. 26 AGREE. 27 AGREE. 28 AGREE.

Astra on my list: 1 and 2 DISAGREE (stacking; brighter map 2 window). These two come from Rohan's direction that the frames must be readable and the in-distribution rows should not be a dark corridor, so they are applied with Astra's constraint honoured: the map 2 windows keep matched controls (turn-left and forward) and are re-picked by brightness from a new export rather than swapping in a fight clip; the stacked layout is drawn at whatever height the frames need and the paper owner decides the trade against the 1.6 in ceiling. 7 and 8 DISAGREE: accepted (height 1.6 in now; dB stays in the text). The rest as Astra changed them.

## Agreed set for the lead, round 2

Astra 1 to 14, 16 to 28 (18 with my wording; 8 with the 1.6 in target); Astra 15 not applied. Fable 3 to 6, 9 to 16 as Astra changed them; Fable 1 and 2 applied on Rohan's direction with matched controls and a brightness-picked map 2 window; Fable 7 and 8 withdrawn.


## Lead's response (round 2, agreed set)

Commits on the lead's branch (rebased on dde9a34): Figure 2 `ab6d992`; Figures 3a, 3b, 4 and the deficit panel `93b2c55` and `7073d53`; training curves `1a1d10f`; captions in `paper/FIGURES.md` (this commit).

- Astra 1: applied. The row label reads "LoRA 4k", with the 8 episodes in the caption, and the stacked frames are 0.7 in wide.
- Astra 2 and 3 (bug): applied. Tic 0 is headed "0*" and shows the same raw last-context frame in every row; the sidecar records the source per row, and a test asserts it.
- Astra 4 and 5: applied in the caption, including the tic-32 counterexample (13.0 against 14.5 and 15.1 dB). The arena 7 selection rule is marked VERIFY because the steward's rule is not in the repo.
- Fable 1 and 2 on Rohan's direction: applied. The stacked layout uses tics 0*, 1, 4, 16 and 32 (4.3 x 6.1 in, frames 0.7 in), and map 2 uses the steward's brighter matched-control windows (turn left episode 6008 from row 2208; forward episode 6024 from row 4562, the moving one). A group drawn from one window in every block is drawn once. The side-by-side version stays as `fig2_rollouts_wide`.
- Astra 6: applied. The 3.5 dB curve is a neutral dark dashed line drawn above the blue curve, with neutral dotted bounds and a matching label.
- Astra 7: applied. The labels read "fraction reaching threshold" and "half-gap threshold".
- Astra 8 with the 1.6 in target: applied. Figure 4 is 5.5 x 1.6 in again: the rug keeps four lanes, and one shared update label replaces the two.
- Astra 9 (bug): applied. Arena 9's leader is anchored at its 5.109 dB endpoint; only the text moves above the band (`figstyle.end_labels` takes a label height apart from its anchor).
- Astra 10 and 12: applied in the captions with the listed numbers.
- Astra 11: applied. The 0.5 line of the gap-share panel reads "half of in-distribution gap", and zero stays "zero-shot".
- Astra 13: applied. The adapter's diamonds leave both Figure 3b versions; its endpoint is Figure 4b's dot panel.
- Astra 14: applied. The primary 3b is the absolute form from the raw scene fields (`scene_psnr_raw[_tuned]`, `scene_lpips_raw[_tuned]`, `scene_persist_*`), with the duplicate mask.
- Astra 15: not applied (the agreed set).
- Astra 16: applied. The markers are 3.75 pt.
- Astra 17: applied in the caption, including the pooled training-map read and SD 3.5's missing fine-tuned read.
- Astra 18 with Fable's wording: applied. The y label is "zero-shot latent skill (dB)", with the definition in the caption.
- Astra 19: applied. PixArt-α is a larger open square behind the U-Net's filled circle, at exact coordinates without jitter; the caption says open does not mean a decoder here.
- Astra 20: applied as a caption statement (point estimates, no intervals).
- Astra 21: applied. The panel letter "a" is added, and the caption gives the floor as an empirical range.
- Astra 22: applied. Nearby arenas (7/13/17, 1/6/12) get separate labels stacked beside their cluster, with leaders (`figstyle.label_points(merge=False, leaders=True)`).
- Astra 23: applied. The diamonds are 3.75 pt.
- Astra 24: applied in the caption. rho is -0.896 and no interval has been computed.
- Astra 25: applied. The labels read "ΔPSNR vs raw persistence (dB)" and the zero line "persistence". This file is owned by opus-method-fixes; the coordinator's round-2 list asked for these edits, so they are made in `1a1d10f` and flagged in the report.
- Astra 26: applied. U-Net markers sit at 25k, 75k, 125k and 175k and PixArt-α markers at 50k, 100k, 150k and 200k, all at measured reads.
- Astra 27: applied. The body panel has "b".
- Astra 28: applied in the caption.
- Fable 3 to 6 and 9 to 16, as Astra changed them: applied through the items above and the captions. Fable 4's "(adapter)" gives way to Astra 1's "LoRA 4k"; Fable 14's rho goes to the caption, not the panel.
- Rohan's 4a decision: `fig4_dots` is the primary Figure 4 candidate ((a) dB curves with the median, (b) per-arena share of the gap with the crossing budget above each dot), and the gap-share curve moves to the appendix.
