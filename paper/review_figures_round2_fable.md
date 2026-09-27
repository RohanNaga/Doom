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
