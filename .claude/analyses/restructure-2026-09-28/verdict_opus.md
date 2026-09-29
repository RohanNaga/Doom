# Verdict (opus)

## 1. Pick

B.

## 2. Why

1. B's Results section is split into four bolded findings ("The shift breaks the scene, not the controls", "The same loss in every backbone, and a step", "The fault is in the model, not the renderer", "How much adaptation ... Very little."). These mirror contribution bullets 2 and 3 one to one, so the paper reads as a set of demonstrated results. A puts the same material in two dense blocks: the adaptation paragraph on p.5-6 runs about 25 lines, and the headline number sits in its middle.
2. B names its adapter section "Adaptation as a measurement" and opens it with "To measure how much adaptation a map needs". That answers the professor's second request in the text itself. A's "Adaptation" opens with "To move a trained model to a new map we keep its weights frozen and train a small adapter", which reads as a method description.
3. B keeps findings out of Study design. A states results before Section 4: "architecture turns out not to change what survives the shift" (Three backbones paragraph), and "so the shift lives in the model, not the renderer" (Decoder paragraph). A then states the decoder result a third time in Results. This is the same kind of repetition the professor asked to cut from Section 2.
4. B defines one term, the *excess gap* ("how much further below this bound the model sits there than on the training maps"), and uses it through Results and the Table 2 caption. A drifts between "gap to the upper bound", "extra distance the shift opened" and "excess gap over the in-distribution gap", and it uses "excess gap" in Table 2 without ever defining it in Metrics.
5. B's Conclusion turns the negative predictor result into a claim: "so a budget has to be measured, not predicted". That sentence justifies the benchmark as science. In A, the same result is an orphaned one-sentence paragraph at the end of Results.

## 3. Where A is better (take these across into B)

- **HUD control (A, Results, first paragraph, last sentence):** "the loss sits in the scene, whose rows fall 2.9 dB (25.2 to a median 22.3) while the map-independent HUD rows fall 0.2 dB (29.7 to 29.5)". This is the cleanest within-frame control in the paper. It belongs in B's "The fault is in the model" paragraph.
- **Episodes and forgetting (A, end of the adaptation paragraph):** "Data matters less than the first updates: ... one episode already gives most of the gain of sixteen" and "The adapters trade some in-distribution skill for it: ... (median 0.59 dB ...), while the directional score rises from 0.81 to 0.84."
  - B's heading promises "updates, episodes, parameters", but B never discusses episodes.
  - B's "Parameters matter somewhat more" compares against the data result it deleted, so the sentence now has no referent.
  - B drops forgetting entirely, and a reviewer will ask about it.
- **SD 3.5 grid fairness (A):** "Its median budget is 250 updates, but its grid starts at 250; read on the same grid, the U-Net's median budget is also 250." Without this sentence, B's "crossing half at 250 updates" reads as SD 3.5 being slower than the U-Net's 150.
- **Adapter size per backbone (A, Adaptation):** "25.9M, 4.1 percent, of PixArt-alpha; 40.6M, 1.8 percent, of SD 3.5". B defers this to the supplement. The PixArt adapter trains about 8 times the U-Net's fraction of parameters, so the cross-backbone recovery comparison (72 vs 76 vs 77) needs this number in the body.
- **Full fine-tune detail (A, adaptation paragraph):** "does better on each of the four comparator maps ... (0.20 to 0.29 dB and 0.017 to 0.030 in LPIPS per map)". This is stronger than B's medians only. A is also honest that the full fine-tune takes 0.8 h on an A6000 against the adapter's 1.1 h on an A4000.
- **"Most of" instead of "because" (A):** A's "Most of the PSNR still missing ... (1.6 dB) reflects the unseen maps' lower upper bound" is more accurate than B's causal "because" (see section 5).
- **Tail statistic (A, Results, first paragraph):** "one predicted frame in five has an LPIPS above 0.4, against one in 73 on the training maps". It is a concrete, memorable number worth one line.
- **Motivating sentence (A, Introduction, paragraph 2):** "Yet unchanged dynamics alone do not guarantee a faithful render: pretrained video and world models are typically moved to a new domain only through a further full fine-tune [5] or a dedicated adapter [6]". It motivates the question with citations. Keep the first half. The uncited "tends to fall back on the appearance it already knows" asserts the finding before showing it.

## 4. What a skeptical reviewer would attack in B

- "The same loss in every backbone" and "a step, not a slope" rest on one seed per backbone and on medians over 13 maps, with no intervals. SD 3.5 loses 3.3 dB against 2.8 to 2.9 dB for the other two, so "neither ... 2.6 times the parameters changes what the shift costs" is stated more strongly than the evidence supports.
- "Very little" adaptation is judged only against a full fine-tune that does better on every comparator map and takes similar wall-clock time. Without the episode ladder and the forgetting numbers, B cannot say what "little" is relative to.
- The zero-shot 22.3 dB is only 2.5 dB above copying the last frame (19.8 dB), and all scoring is one tic ahead.

## 5. Errors and unclear passages

**B**

- **Study design, "Maps and data":** "The release holds 8,000 episodes on the four maps (6,000 train, 1,000 validation, 1,000 test)". The same paragraph says "we train on 500 episodes of each", which is 2,000, and A says "In total we collect 2,412 episodes". These counts conflict within B and across the drafts. One of them is wrong, or the text must say that training used a 2,000-episode subset of the release.
- **Same paragraph:** "on the 17 Doom maps it ships with, which we call maps". This is left over from the arena-to-map rename and should be deleted. Figure 2 ("13 unseen arenas", "arena adaptation") and the first column of Table 2 ("Arenas (by zero-shot gap)") in both drafts still say arena.
- **Measurements:** the decoder "rises from 26.9 to 28.4 dB". B drops A's "On 200 validation frames", so 28.4 looks inconsistent with Table 1's 28.55 and the text's 28.6.
- **Results, adaptation paragraph:** "It regains only 48 percent of the lost PSNR, because the unseen maps' upper bound is itself lower (27.3 against 28.6 dB)". The upper-bound difference, 28.55 - 27.32 = 1.23 dB, explains most but not all of the 1.6 dB still missing. Use "mostly because".
- **Same paragraph:** "Parameters matter somewhat more" has no comparison target (see section 3).
- **Introduction, paragraph 2, last sentence:** it chains two colons ("... recover the rest: in short, a world model ..."). A's split into a separate "In short" sentence reads better.
- **Table 1 and Results:** "against 0.885 and 0.892 for the ground-truth frames". Neither draft says which value is which (training vs unseen maps, or one per decoder). The problem is more visible in B because B moves the pair into the body.
- **Build:** Figure 3 renders as a "pending" box in draft_B.pdf because figures/raw/raw_row_v2_tall.pdf was missing at build time. The figure is shared, so this is not a prose criterion, but it must be fixed before sending.

**A**

- **Source and PDF disagree on the title:** draft_A.tex line 17 still carries the old title, "DoomShift: How Much Adaptation Do World Models Need Under Domain Shift?". draft_A.pdf shows the new one. Rebuilding from A's source would revert the change the professor asked for.
- **Introduction, paragraph 2:** "fine-tuning a separate model per map is prohibitively costly". The paper then shows that a per-map full fine-tune takes 0.8 A6000-hours and does better than the adapter, which contradicts this sentence. B's "training a separate model per map" is correct.
- **Adaptation:** "it runs ... at 1.24 updates per second" implies 4000 / 1.24 = 0.90 h for 4k updates. Results says "4k updates take a median 0.8 hours". Reconcile the two numbers, or say the rate is not the median.
- **Adaptation:** "4k updates is the elbow ... captures most of the benefit ... at the smallest budget". 4k is not the smallest budget on the grid. The intended meaning is "the smallest budget that captures most of it".
- **Results, forgetting sentence:** "the training maps' scene PSNR ... falls on all 13 unseen maps at the 8k check". This should read "falls for all 13 adapters". It is also unclear whether "directional score rises from 0.81 to 0.84" refers to the unseen maps or the training maps.
- **Adaptation:** "with the control checks (forgetting, ...)" uses "control" in a paper where "control" means the action input. Call them "retention checks" or similar.
- **Adaptation paragraph:** "half of each map's 8k gain arrives within 100 updates on average" sits beside a median budget of 150 for half the excess gap. These are two different "half" quantities, and a reader will conflate them.

**Both**

- "Closes a median 101 percent" (PixArt-alpha) needs half a clause saying that it ends slightly closer to its bound than in distribution.
