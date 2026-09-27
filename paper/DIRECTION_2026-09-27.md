# What Rohan is trying to do, and the direction he has given (compiled 2026-09-27, 12:00 EDT)

Every agent working on the paper or its figures reads this before editing. It is Rohan's direction in his own terms, compiled by the main session from today's messages; where it conflicts with an older brief, this wins.

## The goal

A four-page CoRL 2026 PhysWM workshop paper ("Crossing the Map Shift") that a world-models researcher reads in ten minutes and comes away with three things: we built a Doom world model (three backbones under one recipe, a matched decoder fine-tune, an adapter path, an open 17-arena benchmark); off its training maps it keeps the controls and loses the appearance in a specific, measurable way; and eight episodes of a new arena and 4k adapter updates fix most of it, with the amount and the speed characterised per arena. The draft goes to Changliu today at 14:30; the deadline is Thursday Oct 1, 07:59 EDT. The paper's quality standard is the best papers in the area (GameNGen, DIAMOND, Genie, Vista, XEWorld), not a class project.

## What Rohan wants the paper to highlight, each with its own figure

1. What we built: a method overview in the role of GameNGen's overview figure (data, three backbones under one recipe, post-training as two boxes: world-model adaptation and decoder fine-tune, evaluation), drawn from the code with a real gameplay frame in the data panel; the backbones' training curves; the decoder with and without fine-tuning. This is a contribution, stated as one, even if the novel part is the measurement.
2. The eye-catching teaser in the role of GameNGen's Figure 1: large frames from one long rollout on an arena the model never saw, at the action moments, captioned with the executed action, and carrying our thesis: the same frame and action shown zero-shot, after eight episodes of adaptation, and true. Fights, not turn-left corridors. Rohan (13:15): Figure 1 must be stronger, it must show an in-domain example beside the unseen one (a real occurrence of the same action on a training map), and it does not come to him again until a selection round over every candidate window and action moment (a scored contact sheet, reviewed by Fable and Astra) has picked the best example we have. Layout B (context frame, action, the frame eight tics later for training map, zero-shot, adapted, true) is the "frame plus action gives the next frame, and how it differs" picture.
3. The shift: a family step (every unseen arena farther from the training recordings than any training map, and below every training map in zero-shot skill for all three backbones; within the arenas the distance orders nothing), and a per-arena panel in absolute PSNR and LPIPS with a persistence marker per arena, arenas in number order, one band per arena, so a reader sees which marks belong to which arena and no sign convention is needed.
4. Adaptation: curves in dB with the median and the in-distribution band, arenas starting at their real levels (the share-of-gap curve hides that by starting every arena at zero), and a per-arena dot plot of the share of the gap closed at the final budget with the crossing budget above each dot. The survival curve with an at-risk row was judged confusing and is out of the body.

## Standing rules from Rohan

- Scientific figures, not design figures: the standard in `paper/FIGURE_STANDARDS.md`, grounded in the reference papers and the published guidance; one meaning per colour and marker; real frames, not schematics; nothing that looks generated.
- Terminology: the field's terms, never ours. "In-distribution" or "training maps", never "home". "Persistence" for the copy-last baseline. "Budget" in updates, not "cost". "Fine-tuned decoder", "non-EMA weights", "control checks", "recordings", "reconstruction upper bound". The decoder keeps the symbol $D$ (the boxes use it); the frame distance is lowercase $d$; the in-distribution advantage is $A_\text{train}$, never $A_\text{ID}$. The full table is item 13 of `paper/diffs/2026-09-27-sunday-adaptation.md`.
- Axes: absolute quantities where a reader expects them (PSNR higher is better, LPIPS lower is better, with the persistence reference drawn), the paired difference only where arenas must share a zero (the adaptation curve). "Advantage" is not an axis label.
- The per-arena framing: the unit is the arena; a claim is a distribution over 13 arenas with an arena-bootstrap interval; per-arena statements carry their episode interval; the "typical new arena of this game" reading is an explicit assumption; nothing beyond the WAD.
- The distance: pre-registered, reported honestly as a family separator that does not order the arenas, with the failed gate stated; the model's own zero-shot skill (or deficit) is what orders them. The distance-variants table (KL, Frechet, MMD, kNN) is for our understanding only, not the paper.
- Reports of "done" are verified: numbers traced to files, Astra truth-checks the text, two reviewers (Fable and Astra) comment on every figure version and only agreed comments are applied; Rohan reviews every version himself and iterates.
- Speed: get the relevant context, then act; do not spend an hour reading for a mechanical edit. A brief names the files that change the edit and a time box. Clearly-right changes are made at once; unclear ones wait for a decision.
- Everything in the repo: page sources, review files, decisions, logs (`RESEARCH_CONTEXT.md`), so nothing lives only in a chat.

## Decisions taken today (do not reopen)

Figure 4a in dB; the share of the gap only in the per-arena dot panel; 3b in absolute form, arena-number order, three backbones, no adapter marks (they live in 4b); "in-distribution" band with its interval instead of a "home" line; EMA only in the body, non-EMA in the appendix; the 8k grid replaces the 4k set tonight as the headline (numbers-only change); the teaser is a fight window on arena 7 held-out episode 210, layout B primary; the recipe stands (learning-rate and grid tests inside seed noise); the cross-arena control and the seed-1 reruns are queued; kills are not in the recordings (data-card fact).

## Open, Rohan's

Figure 2 rollouts: one control in the body or both in the appendix. Whether the ladder goes in the body. The M sign convention if the paired form survives anywhere. Seed 1 on the other nine arenas tonight. The four-tic column of Table 2.
