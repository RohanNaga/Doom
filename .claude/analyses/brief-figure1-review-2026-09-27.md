# Brief: Figure 1 (teaser) review, candidate 2 at +8 and +16 tics (Sep 27, 13:50 EDT)

Paper: DoomShift: Efficient Adaptation of World Models to Nearby Domain Shifts (CoRL 2026 PhysWM workshop, 4 pages). Rohan's rule: Figure 1 must be the best example we have, must show an in-domain example beside the unseen one, and must show "a frame, the action, and how the next frame looks different". He sees it only after this review.

## The two candidates (same moments, different horizon)
- /private/tmp/claude-501/-Users-rohan-Documents-Github-Doom--claude-worktrees-vibrant-ritchie-6f6280/bbd290c7-2c43-46b3-bfab-7994575157d8/scratchpad/figpage/img/fig_teaser_C_2_restart.png (+8 tics after a 32-tic ground-truth context; 5.5 x 2.05 in)
- /private/tmp/claude-501/-Users-rohan-Documents-Github-Doom--claude-worktrees-vibrant-ritchie-6f6280/bbd290c7-2c43-46b3-bfab-7994575157d8/scratchpad/figpage/img/fig_teaser_C_2_restart16.png (+16 tics)
Layout: left block "training map 2 (in-domain)": context frame, U-Net rollout frame, ground truth. Right block "unseen arena 7": context, zero-shot rollout, rollout after adapting on 8 episodes (rank-16 LoRA, 4k updates), ground truth. Rows: attack, turn right, forward (the held control). Numbers under frames: scene PSNR of that frame against ground truth. All rollouts restart from ground-truth context (no accumulated drift beyond the shown horizon). Frames rendered by the fine-tuned decoder; ground truth is the raw frame.
Rejected alternative: candidate 1 (episode 288), mean story +5.52 (+8) / +5.25 (+16) against candidate 2's +6.62 / +6.32.

## Numbers (scene PSNR, zero-shot | adapted; in-domain U-Net)
| row | +8 | +16 |
|---|---|---|
| unseen attack | 15.2 \| 20.4 | 15.1 \| 17.1 |
| unseen turn right | 14.2 \| 24.1 | 13.4 \| 23.9 |
| unseen forward | 16.1 \| 20.8 | 13.1 \| 19.5 |
| in-domain attack / turn right / forward | 21.0 / 24.0 / 26.5 | 21.2 / 24.9 / 25.7 |
Persistence (copy-last) on the in-domain rows: 17.1 / 17.1 / 20.2 at +8; 16.6 / 16.9 / 20.3 at +16.

## Questions
1. Which horizon makes the stronger Figure 1 for a reader who sees it for 10 seconds, and why? (Main session's view, withheld until you answer: +16.)
2. Does the figure make the paper's claim visible: control kept (the turn and the forward motion happen), appearance lost (zero-shot renders arena 7 in map 2's grey stone), relearned from 8 episodes? Name any row where it does not.
3. Anything misleading? Check: are the PSNR numbers the right thing to print under frames, is "after 8 episodes" honest given 4k updates, does the in-domain block need its own persistence number, is any header ambiguous.
4. Concrete edits that would make it stronger at the same size (5.5 x 2.05 in), ranked; and the one-sentence caption you would write.
Standards: paper/FIGURE_STANDARDS.md. Sources: tools/compose_teaser.py, tools/teaser_contact_sheet.py (lead's worktree /Users/rohan/Documents/Github/Doom/.claude/worktrees/agent-a05f5df377f900f32). Answer in under 500 words, ranked, no hedging.
