# Verdict (Fable)

## 1. Pick

**A.**

## 2. Why (five sentences)

1. A answers all three axes of its own Section 4 heading, "updates, episodes, parameters": the data-ladder sentence ("one episode already gives most of the gain of sixteen") and the forgetting check (training-map scene PSNR falls a median 0.59 dB at 8k) exist only in A, whereas B's paragraph under the same heading answers updates and parameters and says nothing about episodes beyond "eight", so B leaves the title question one-third unanswered.
2. A keeps the controls that turn "scene, not controls" from a slogan into a demonstration: the HUD-row check (scene rows fall 2.9 dB, map-independent HUD rows 0.2 dB), the per-map ranges (directional 0.73 to 0.86 against 0.79 to 0.91 in distribution; LPIPS 0.229 to 0.401), and the one-in-five frames above LPIPS 0.4 concentrated in maps 16, 17, 13 and 1; B's three finding paragraphs in Section 4 rest on the Table 1 medians alone.
3. A's Section 3 "Data" is internally consistent (500 + 25 episodes per training map, 24 per unseen map, 2,412 episodes, 96 h, 12.1 M frames), while B's "Maps and data" replaces the total with "the release holds 8,000 episodes on the four maps (6,000 train, 1,000 validation, 1,000 test)", which does not match the 500/25 the models actually used and invites "why train on a quarter of it?".
4. A's "Adaptation" paragraph gives the scoring grid per backbone, the budget range (50 to 2k per map, none censored), and the full fine-tune's hardware cost (0.8 h on a 49 GB A6000 against 1.1 h on a 16 GB A4000), which is the cost evidence a paper titled "How much adaptation" needs; B keeps only the 205x parameter ratio and states one grid ("from 50 to 8k") that is wrong for two of the three backbones.
5. B's real advantages are structural (finding-headed results paragraphs, the "Adaptation as a measurement" heading, the one-sentence "excess gap" definition) and can be lifted into A during the length cut, whereas restoring A's evidence into B means rewriting half of Section 4.

## 3. Where B is better (take these across)

- **Section 3, "Measurements", the excess-gap definition:** "A map's *excess gap* is how much further below this bound the model sits there than on the training maps." A uses "excess gap" in the Adaptation paragraph and the Table 2 caption without a standalone definition; add B's sentence to A's Metrics paragraph and A's budget definition becomes one clause.
- **Section 3, heading and first sentence "Adaptation as a measurement. To measure how much adaptation a map needs, we keep the trained model frozen and train a standard adapter":** this is exactly what the professor asked for in item (2). A's paragraph opens "To move a trained model to a new map we keep its weights frozen and train a small adapter", which reads as a method being proposed.
- **Section 4, the three finding-headed paragraphs** ("The shift breaks the scene, not the controls", "The same loss in every backbone, and a step", "The fault is in the model, not the renderer") and B's opening sentence "In domain the three models sit 3.4, 3.4 and 6.5 dB below their decoders' upper bounds; SD 3.5, the best of the three, leaves the most room below its own sharper decoder." A scatters the in-distribution gaps across the Section 3 backbone paragraph; B introduces them once, where they are used. Use B's headings over A's content.
- **Section 4, "How much adaptation: updates, episodes, parameters. Very little. Half of a map's excess gap closes in a median 150 updates":** the two-word answer followed by the budget. A buries the 150-update budget in the middle of a 30-line paragraph.
- **Section 4, "It regains only 48 percent of the lost PSNR, because the unseen maps' upper bound is itself lower (27.3 against 28.6 dB)":** one sentence where A spends two ("Most of the PSNR still missing ... reflects the unseen maps' lower upper bound").
- **Section 5, "so a budget has to be measured, not predicted":** B gives the predictor null result a consequence. A's version is an orphan sentence at the end of Section 4 ("Neither the model's own latent-space score nor a frame distance ... predicts how far a map gets") with no conclusion drawn.
- **Section 1, paragraph 1:** B's "Even MultiGen, one Doom model trained across 100 procedurally generated maps, reports PSNR, SSIM and LPIPS only on its training distribution, with no map held out" is tighter than A's two-sentence version and loses nothing.
- **Section 3, recipe:** B's "(one A6000 per backbone, one A4000 per adapter)" states pretraining compute up front; A never says what the 200k updates ran on.
- **Section 3, Measurements:** B cites three persistence baselines (Mathieu, PredNet, Villegas) where A cites one.

## 4. What a skeptical reviewer would attack in A

- Section 4's adaptation paragraph is a 30-line wall of medians on at least three aggregations (per-map median of ratios, median of per-map values, pooled) with no signposting; the reviewer will compute 27.32 minus 23.60 = 3.7 dB from Table 2 and ask how "a median 3.4 dB again" and "94 percent" arise, and whether the 71/87 full fine-tune comparison uses the same aggregation as the 72.
- Section 1, paragraph 2 asserts the paper's own finding as background without citation ("a model trained on a handful of maps tends to fall back on the appearance it already knows once the surroundings change"), and the decoder-not-renderer result is stated three times (contribution 2, Section 3 "Decoder", Section 4), so a reviewer will say the paper is padded and the "Section 2 repeats Section 1" complaint has moved to Sections 3 and 4 (the backbone paragraph in Section 3 already reports the 0.02 dB / 0.001 in-distribution match and "they lose the same amount on the unseen maps").
- Mostly one seed, one-tic scoring, eight held-out episodes per map, and the forgetting number is measured on the stock decoder while everything else uses the fine-tuned one; the conclusion admits the first two but not the last.

## 5. Anything wrong in either draft

**Draft A**

- `draft_A.tex` line 22: `\title{DoomShift: How Much Adaptation Do World Models Need Under Domain Shift?}` while `draft_A.pdf` shows the new two-line title. The source and the PDF are out of sync; make sure the edited source is the one that was compiled before anyone takes wording from it.
- Section 1, paragraph 2: "Pretraining one model on every map a deployment might meet is costly and never complete, and fine-tuning a separate model per map is prohibitively costly" uses "costly" twice in one sentence; B's version ("is never complete ... prohibitively costly") fixes it.
- Section 3, "Data", first sentence: "We collect the data for this research by running experiments with the Arnold agent" is filler; B's "The Arnold agent plays 150-second ViZDoom deathmatches" says the same in half the words.
- Section 3, "Data": "for 150 seconds" per episode, but 2,412 episodes at 150 s is 100 h, not the stated 96 h (12.1 M frames / 35 per second = 96 h). Episodes average about 143 s, so say "up to 150 seconds" or say why some end early.
- Section 4, last sentence of the adaptation paragraph: "the training maps' scene PSNR against the decoded ground truth falls on all 13 unseen maps at the 8k check (median 0.59 dB; stock decoder), while the directional score rises from 0.81 to 0.84" is ambiguous. It is not clear whose directional score rises (0.81 to 0.84 matches the unseen maps' 0.804, but the sentence is about forgetting on the training maps), nor why this one number uses the stock decoder.
- Section 3, "Three backbones": "architecture turns out not to change what survives the shift ... they lose the same amount on the unseen maps (Table 1)" is a result inside the study design and is restated in Section 4.

**Draft B**

- Section 1, paragraph 2, one sentence with two colons: "... what the model retains, what it loses, and how few episodes and updates recover the rest: in short, a world model moved to a new map keeps playing the game but draws the wrong world, and a small adapter trained on eight episodes draws the map back (Figure 1)." Split it as A does.
- Section 3, "Maps and data": "on the 17 Doom maps it ships with, which we call maps" is a leftover from the arena-to-map rename and reads as a tautology. Delete "which we call maps".
- Section 3, "Maps and data": "The release holds 8,000 episodes on the four maps (6,000 train, 1,000 validation, 1,000 test)" against "we train on 500 episodes of each and score on 25 further validation episodes per map". Either the release figure is wrong or the paper needs one clause saying the models used 500 + 25 of the 2,000 per map and the 1,000 test episodes are unused here. B also drops the total hours and frames, so the reader cannot size the benchmark.
- Section 3, "Adaptation as a measurement": "we score held-out episodes at update counts from 50 to 8k" is wrong for PixArt-alpha (to 4k) and SD 3.5 (250 to 4k). The SD 3.5 caveat surfaces only in the last sentence of Section 4; a reviewer reading Table 2 sees SD 3.5's budget of 250 against the U-Net's 150 and concludes SD 3.5 adapts slower.
- Section 4, paragraph 1: "against 0.885 and 0.892 for the ground-truth frames" does not say which number is training and which is unseen, and never says why a ground-truth frame scores below 1 on a directional check (the same measurement run on the recorded next frame). A avoids this by keeping the numbers in the Table 1 caption only.
- Figure 3 in `draft_B.pdf` is a "pending" placeholder (`figures/raw/raw_row_v2_tall.pdf` missing at compile time), so the B PDF as delivered cannot be reviewed for its main results figure. A compile artifact, but whoever compiles B must fix the path.
- `draft_B.tex` inlines Table 1 instead of `\input{tables/tuned/results_full.tex}`; the table will go stale the next time the generator runs.
- Section 5: "Next are other games" drops A's "other Doom maps and other games"; the nearer step is the one the benchmark supports.

**Both drafts**

- Figure 2, panel (a) "13 unseen arenas" and panel (c) "arena adaptation", and Table 2's first column header "Arenas (by zero-shot gap)", still say "arena" after the text's rename to "map".
- Contribution 3 and Section 4: "recovers 87 percent against the adapter's 71 on four maps". Computed from Table 2's row medians, (0.292 - 0.209) / (0.292 - 0.158) = 62 percent for LoRA and (0.292 - 0.182) / 0.134 = 82 percent for the full fine-tune. The 71 and 87 are per-map medians of the ratio, which is fine, but neither text nor caption says so, and a reviewer who recomputes from the table will call it an error.
- Table 2 caption in both says "the PixArt-alpha and SD 3.5 grids stop at 4k" but not that SD 3.5's grid starts at 250, so SD 3.5's budget of 250 reads as worse than the U-Net's 150 when A's text says the U-Net's budget on the same grid is also 250.
- Table 1 caption: "directional pooled over the turning windows (ground-truth frames 0.885, 0.892)" does not say what a ground-truth directional score is.
