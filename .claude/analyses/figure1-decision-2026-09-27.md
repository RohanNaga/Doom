# Figure 1 decision (Sep 27 2026, 13:55 EDT)

Brief: `brief-figure1-review-2026-09-27.md`. Astra thread 01a0e3f9-60a8-7c02-a644-f5e7e52c6715. Reviewers: Astra (gpt-6-astra, high), main session (Fable). Rohan sees the figure only after this round (his rule of Sep 27).

**Decision.** Candidate 2 (unseen arena 7, episode 210, start s2888; moments t222 turn right, t134 forward, t162 attack) at +16 tics, in-domain block from map 2 (ep6024 s2568 t29 attack, ep6064 s2312 t34 turn right, ep6008 s712 t78 forward). Both reviewers chose +16 independently: the forward motion produces a visible viewpoint change and the zero-shot turn renders arena 7 in dark training-map masonry, which adaptation restores (turn 13.4 to 23.9 dB, forward 13.1 to 19.5 dB). Candidate 1 (ep288) rejected: mean story +5.25 against +6.32.

**Edits ordered** (lead): rows turn, forward, attack; honest control labels (the selector requires the named button to start at the restart tic and be held 8 tics, other buttons allowed, so the label prints the true held combination and span); adapted header names 8 episodes and 4k updates; one "scene PSNR (dB)" label and persistence under both context frames; "context (t = 0)"; one horizon line; caption says "selected".

**Astra points not taken.** "unseen during pretraining" (our term is training maps); "copies map 2 is not established" (the caption says training-map masonry, not map 2).

**Weak row.** Attack at +16: adapted 17.1 dB, enemy pose disagrees with ground truth; kept as the third row because the appearance recovery is still visible and the paper does not claim recovered enemy dynamics.
