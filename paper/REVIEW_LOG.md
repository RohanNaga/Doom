# Review log: the joint Fable 5.1 and Astra review of the paper skeleton (applied 2026-09-27)

Sources: `paper/review_fable.md` (Fable items F1 to F27, round-two proposals F28 and F29, page cuts c1 to c11, round-two verdicts on Astra's list as "A<n>") and `paper/review_astra.md` (Astra items A1 to A30, round-two proposals A31 and A32, round-two verdicts on Fable's list). Rule (Rohan, via the review brief and the coordinator's instruction of Sep 27): apply only what both sides accept; where both accept a change to the same passage, use the wording the round-two passes named, with the coordinator's rulings on the overlaps; skip anything one side rejected; then anonymise, make the submission build body plus references, and cut to the four-page body.

## Status words

- **applied**: both sides AGREE; the proposer's wording, verbatim.
- **applied-with-the-named-wording**: the other side marked AGREE WITH CHANGE, or both sides changed the same passage; the wording used is named in the row.
- **rejected-by-one-side**: one side marked DISAGREE; not applied.
- **rejected-for-page-budget**: agreed, but removed or superseded by a page cut.

## How overlaps were resolved

Coordinator rulings (Sep 27), applied as given:

1. The per-arena half-gap cost rule takes Fable's compact wording (F7 to F14) plus Astra's censoring clause and "negative is loss" (A2, A4), through cut c3 for the Table 3 caption.
2. The abstract keeps Fable's six-sentence structure with Astra's facts (A7) and Astra's "do not guarantee" verb (A18).
3. The decoded-advantage passage takes Fable's items 3 and 4 and item 5 as amended in Fable's round two (its verdict on A16). Fable's original item 5 is skipped as rejected.
4. XEWorld: Astra's fact (held-out embodiments reported separately, not pooled) in Fable's shorter sentence (cut c4 in the introduction, A25 in Section 5).
5. HUD: Astra's statistic (geometric mean over windows) with "almost exactly (88 to 96 dB)" (Fable's wording on A14 and A15).
6. Introduction sentence 2: Astra's wording (its verdict on F15).
7. Anonymisation: the template's anonymous author form, the original in a `% camera-ready:` comment, and Astra's eight identifying comments treated the same way (A31).
8. `\withappendixfalse` by default, with Fable's `\appref` macro (F29) in place of Astra's `\ifwithappendix` guards (A32).

Where the two round-two passes named different winners and the coordinator gave no ruling, I used the round-two wording that already folds in the other side's stated correction. For the contribution paragraph and the Protocol paragraph, I built the composite the same way the coordinator did for the abstract: one side's structure with the other side's facts. Each such row says so. These rows are the ones to sign off: A5, A6, A8/F2/F12, A10/F14, A12/F16, A15, A17, A22/F20, A26/F22.

## Fable's items

| Item | Passage | Status | Wording used, and why |
|---|---|---|---|
| F1 | abstract sentence 3 | applied-with-the-named-wording | Coordinator ruling 2: Fable's sentence with the U-Net named, plus Astra's explicit metric: "its LPIPS is worse than persistence's on 13 of 13 arenas" |
| F2 | contribution (2) | applied-with-the-named-wording | Fable's text ("on the U-Net row (replication in PixArt and SD 3.5: \tbd{})") inside the contribution composite (see A8); the numbers were then removed by cut c10 |
| F3 | Section 3, ceiling sentence | applied-with-the-named-wording | Coordinator ruling 3: Fable's wording. Astra's round two named A16's "changes less ... consistent with a prediction deficit" |
| F4 | Section 3, the 2 dB sentence | applied | Astra AGREE; verbatim |
| F5 | Section 3, perceptual-flip sentence (original) | rejected-by-one-side | Astra: it mixes model LPIPS against the decoded target with persistence LPIPS against the raw target, and `copy_lpips_dec` is missing. The amended wording is applied under A16 |
| F6 | Section 5, first sentence | rejected-for-page-budget | Superseded by cut c9, which deletes the sentence (Astra's A24 on the same sentence is superseded too) |
| F7 | Section 4, Cost paragraph | applied-with-the-named-wording | Coordinator ruling 1: Fable's three sentences verbatim, with `\appref`; Astra's censoring clause lives in the Table 3 caption (c3) |
| F8 | Table 3 caption and milestone rows | applied-with-the-named-wording | Caption: cut c3 (Fable's compact caption with Astra's censored-median clause and "negative is loss"). Rows: verbatim (Astra: "Fable's two milestone rows win verbatim") |
| F9 | Appendix Table 8 caption, header, rows | applied | Astra AGREE, including the half-gap lines (checked against the unrounded home mean 3.5984 dB); "negative is loss" added under A4 |
| F10 | Figure 2 and Figure 3 captions | applied-with-the-named-wording | Figure 2 deletion verbatim (both). Figure 3: Fable's round-two wording on A5 (see A5) |
| F11 | abstract sentence 5 | applied-with-the-named-wording | Coordinator ruling 2: Fable's slot and "half of each arena's gap", Astra's fact that it is a measurement: "We measure what closing half of each arena's gap to the home advantage costs rank-16 adapters on the frozen backbone: \tbd{} updates on \tbd{} of 13 arenas, at \tbd{} dB of forgetting." |
| F12 | contribution (3) | applied-with-the-named-wording | Contribution composite (see A8): "(3) per-arena adaptation curves, with the cost for rank-16 adapters to close half of each arena's gap to the home advantage (\tbd{}) and a full fine-tune comparator" |
| F13 | Section 4, "What it takes", first clause | applied-with-the-named-wording | Coordinator ruling 1 (Fable's compact half-gap wording). Astra's named replacement for the whole paragraph is A9, which Fable rejected, so the rest of the paragraph stands |
| F14 | conclusion; FIGURES.md `fig:adapt` | applied-with-the-named-wording | Conclusion: Fable's round-two wording on A10, with the U-Net named. Astra named A10 ("remains an open empirical question"); I used Fable's because it is the half-gap wording of ruling 1 and keeps the body's `\tbd` convention, which stands since A9 was rejected. The FIGURES.md part is superseded by A6 |
| F15 | introduction sentence 2 | applied-with-the-named-wording | Coordinator ruling 6: "We need a world model to generalize to unseen levels before using it to plan or train a policy there." |
| F16 | introduction, persistence sentence | applied-with-the-named-wording | Fable's round-two wording on A12: the split, with "each transfer score" and the Genie citation moved to the definition, both of Astra's stated corrections. Astra named A12 |
| F17 | Section 2, backbones sentence split | applied | Astra AGREE; verbatim |
| F18 | Section 2, directional estimator gloss | rejected-by-one-side | Astra: agreement of image motion with the recorded control is a sanity check, not a bound, and the 0.91 needs matched-run provenance |
| F19 | Section 3, within-arena distance clause | applied-with-the-named-wording | Astra's wording: "but within the 13 unseen arenas we do not establish an association between $D$ and $A$ (Spearman \prov{$-0.23$})." |
| F20 | Section 4, Protocol | applied-with-the-named-wording | F20's two sentences; the second sentence as Fable's round two on A22 gives it (position table added); Astra's scope fact folded into the "because" clause ("in DiT-XL/2 image-generation transfer"). Astra named A22 ("We plan ... DiffFit motivates"); composite, needs sign-off |
| F21 | Section 4, "8 of the 16 adaptation episodes" | applied | Astra AGREE; verbatim |
| F22 | Section 5, distances sentence | applied-with-the-named-wording | Fable's round-two wording on A26: the split, with the two directed measures named separately (Astra's correction). Astra named A26 |
| F23 | Section 6, first sentence split | rejected-by-one-side | Astra: "so" derives a power estimate from the concentration of the distance range; power needs a stated test, alternative and sampling model. The original sentence stands |
| F24 | Table 2 caption | applied | Astra AGREE; its last sentence then removed by cut c2 |
| F25 | Table 1 and Table 5 captions | applied-with-the-named-wording | Astra's wording: "Provisional rows: PixArt and SD~3.5\ifdraftmarks{} (blue)\fi; $^\dagger$150k." and "Pre-fresh-set episodes\ifdraftmarks{} (blue)\fi." |
| F26 | Section 3, sampler clause | applied-with-the-named-wording | Astra's wording: "Blur under uncertainty is one possible explanation for the PSNR--LPIPS tradeoff; the mechanism remains untested." The sampler numbers remain in Appendix A |
| F27 | Section 2, decoder sentences | applied-with-the-named-wording | Astra's wording: "Following GameNGen, we plan to tune it with MSE on training-map frames, then freeze it." |
| F28 | author block | applied-with-the-named-wording | Coordinator ruling 7: the template's anonymous form (`Anonymous Author(s)\\ Affiliation\\ Address\\ \texttt{email}`, as `corl_2026.sty` prints it) rather than Fable's "Anonymous Institution / Anonymous City, Country"; the original on a `% camera-ready:` line |
| F29 | `\withappendixfalse` and `\appref` | applied | Coordinator ruling 8; the header comment's last clause changed as F29 asks, merged with A31's second comment |

## Astra's items

| Item | Passage | Status | Wording used, and why |
|---|---|---|---|
| A1 | Section 4, Cost paragraph | applied-with-the-named-wording | Coordinator ruling 1: Fable's F7 wording (Fable: no $C_m(t)$ notation, no four-decimal $H$ in the text) |
| A2 | Table 3 caption | applied-with-the-named-wording | Cut c3 (Fable's round two), which carries Astra's censored-median rule and "negative is loss" |
| A3 | Table 3 rows | applied-with-the-named-wording | Fable's F8 rows (Astra's round two concedes them verbatim; quarter-gap milestone dropped by both) |
| A4 | Appendix Table 8 caption | applied-with-the-named-wording | F9's caption with Astra's sign convention: "forgetting (change in the training maps' $A$, negative is loss)" |
| A5 | Figure 3 caption | applied-with-the-named-wording | Fable's round two: "...; a tick on each curve: its half-gap line; markers: first crossings; open: censored. Black: full fine-tune." Astra's round two also wrote "Black: planned full fine-tune"; Fable holds "planned" is not caption prose. Unresolved word, needs sign-off |
| A6 | FIGURES.md `fig:adapt` | applied-with-the-named-wording | Fable's round-two row: log axis with step 0 as the first labelled tick. Astra's round two wrote "linear axis including step 0"; its stated objection (an ordinary log axis cannot show step 0) is met by the symlog axis. Unresolved axis choice, needs sign-off |
| A7 | abstract | applied-with-the-named-wording | Coordinator ruling 2 (see F1, F11); sentences 1 and 2 unchanged; sentence 4 takes Astra's "has no clear association"; sentence 6 is Fable's round-two sentence with Astra's verb |
| A8 | contribution paragraph | applied-with-the-named-wording | Composite, as ruling 2 for the abstract: Fable's numbered structure (F2, F12) with Astra's facts (U-Net only, replication pending, adaptation as a measurement, "no clear association", "does not guarantee"). Astra named A8's four sentences, Fable its F2 and F12. Needs sign-off |
| A9 | Section 4, "What it takes" | rejected-by-one-side | Fable: every clause already carries `\tbd` and a `% hypothesis:` comment and is rewritten from `score_adapt.py cost` on Sep 28; "Adaptation results are pending" is not submission prose |
| A10 | conclusion | applied-with-the-named-wording | Fable's round-two wording (see F14). Needs sign-off |
| A11 | introduction, rarity sentence | applied-with-the-named-wording | Coordinator ruling 4, through cut c4: "The measurement is rare: we found no Doom world model scored on maps it did not train on (MultiGen does not say~\citep{po2026multigen}), and other held-out domains are an embodiment, a game, an environment or a building (Section~\ref{sec:related})." |
| A12 | introduction, persistence sentence | applied-with-the-named-wording | Fable's round-two wording (see F16). Needs sign-off |
| A13 | Section 2, decoded-advantage definition | applied-with-the-named-wording | Fable's round-two wording: aggregation stated; Genie cited for the paired form, "between inferred and random actions" |
| A14 | Section 2, HUD sentence | applied-with-the-named-wording | Coordinator ruling 5: Fable's round-two sentence, which carries Astra's statistic ("geometric mean over windows") and "almost exactly (88 to 96 dB)"; persistence HUD PSNR rechecked from the 17 h1 files: 87.8 to 96.2 dB. Astra's first clause ("The provisional pixel scores use full frames") is not used: `\prov` marks the numbers |
| A15 | Appendix B paragraph | applied-with-the-named-wording | Coordinator ruling 5: Fable's round-two paragraph (floor mechanism hedged to "consistent with"; HUD statistic stated as a geometric mean). Astra's last sentence ("This is not the fraction of total squared error ...") not used; Fable calls it a reviewer's note. Needs sign-off |
| A16 | Section 3, decoded-advantage passage | applied-with-the-named-wording | Coordinator ruling 3: F4, F3, and F5 as amended: "... and the model, scored against the decoded true frame, still has higher LPIPS than raw persistence (\prov{13 of 13})." (checked: 13 of 13 in the frozen files). Astra's narrower sentence ("reconstruction LPIPS is below raw persistence LPIPS on every arena") is not used |
| A17 | Section 3, distance paragraph | applied-with-the-named-wording | Fable's round two: the heading and floor-band sentence stay; the within-arena clause is Astra's F19 wording; the 30-map sentence becomes "failed its validation gate" first, then "falls to \prov{$-0.42$} with family indicators as covariates ... qualified evidence of a pooled association, not of an ordering" ($-0.423$ checked in `figure_unet_h1_amended/stats.json`). Astra's retitling ("Frame distance and map transfer") not used; the heading is the cohesion decision's wording. Needs sign-off |
| A18 | Section 3, closed-loop first sentence | applied-with-the-named-wording | Fable's round two ("Good one-step scores do not guarantee stable rollouts: ..."), then shortened by cut x4 |
| A19 | Section 3, sampler clause of the closed loop | applied-with-the-named-wording | Fable's round two: "Both share the sampler, so the few-step sampling DIAMOND blames for drift~\citep{alonso2024diamond} is not what differs between them." |
| A20 | Appendix E, latent skill | applied | Fable AGREE; verbatim |
| A21 | Figure 1 caption | applied-with-the-named-wording | Fable's round two: the unobserved result sentence removed, "windows chosen by a seeded rule" added; the stale `% hypothesis:` comment removed with it |
| A22 | Section 4, Protocol | applied-with-the-named-wording | Composite (see F20). Needs sign-off |
| A23 | Section 4, the comparator sentence | rejected-for-page-budget | Fable's round-two wording (a visible `\todo{conditional: ...}`) applied, then removed from the text by cut c1; the conditional is kept in a `% depends` comment, so the rendered sentence reads as before the review. Astra's point (the comparator is pending) is visible only through Table 3's `[tbd]` cells |
| A24 | Section 5, first sentence | rejected-for-page-budget | Superseded by cut c9 (sentence deleted) |
| A25 | Section 5, XEWorld clause | applied-with-the-named-wording | Coordinator ruling 4 with Fable's round two: "XEWorld holds out two robot embodiments, scored separately, finds held-out error tracking an appearance distance, ..."; ", forgetting a seen robot" then removed by cut c5 |
| A26 | Section 5, distances sentence | applied-with-the-named-wording | Fable's round-two wording (see F22); its second sentence then shortened by cut x5. Needs sign-off |
| A27 | Appendix C, transferability sentence | applied-with-the-named-wording | Fable's round two: "In two studies a directed measure predicted transfer better than a symmetric transport distance: nearest-source coverage against EMD ... and Gaussian latent KL against Wasserstein ..." |
| A28 | FIGURES.md HUD row | applied-with-the-named-wording | Fable's round-two row |
| A29 | FIGURES.md `tab:cost` row | applied-with-the-named-wording | Fable's round-two row (half-gap line and home line; Astra's median rule kept) |
| A30 | FIGURES.md `tab:perarena-adapt` row | applied-with-the-named-wording | Fable's round-two row |
| A31 | eight identifying source comments | applied | Coordinator ruling 7: each replaced by an anonymous form, with the original on a `% camera-ready:` line (main.tex: template line, appendix-switch note, `scripts/spiderman` path, "Rohan specified", `eaf565a`; appendix.tex: "Rohan's call", "Spiderman rerun", `42a4688`) |
| A32 | submission build without the appendix | applied-with-the-named-wording | Coordinator ruling 8: Fable's `\appref` (pointers print "the supplement") instead of Astra's `\ifwithappendix` guards |

## Page cuts

Measured on the submission build (`\withappendixfalse`) with today's placeholder figures. Before any cut, the agreed set put the Section 6 heading and its five lines on page 5.

| Cut | Where | Status | Change |
|---|---|---|---|
| c1 | Section 4 | applied | A23's `\todo` moved into a `% depends` comment |
| c2 | Table 2 caption | applied | "The PixArt and SD~3.5 rows take the U-Net's two-row form." deleted |
| c3 | Table 3 caption | applied | Fable's shortened caption (supersedes A2's text) |
| c4 | introduction | applied | Short rarity sentence (supersedes A11's text) |
| c5 | Section 5 | applied | ", forgetting a seen robot" deleted from the XEWorld clause |
| c6 | Section 4 | applied | "so that rendering repair cannot count as adaptation" deleted |
| c7 | Section 3 | applied | "SD~3.5 by the most" and "all" deleted from the in-domain sentence |
| c8 | Section 2 | applied | ", Arnold's own training maps," deleted |
| c9 | Section 5 | applied | First sentence (GameNGen, DIAMOND, MultiGen) deleted; MultiGen moved into c4 |
| c10 | contribution (2) | applied | The numbers the abstract already prints removed from (2) |
| c11 | Section 5 | not applied | Would attach the Doom-agent citations (Arnold, ViZDoom competitions) to "None scores per scene against persistence", which they do not support; not needed after x1 to x5 |
| x1 | Section 5 (related work) | applied | "with data and step curves" deleted from the AVID/AdaWorld clause |
| x2 | Section 5 | applied | "None scores per scene against persistence, and Doom map holdouts are for agents" (a semicolon clause joined) |
| x3 | Section 3 (closed loop) | applied | "11 against 8 without the 70k collapse" becomes "(11 and 8 without 70k)" |
| x4 | Section 3 (closed loop) | applied | "where one latent channel's mean is captured and frames go blank" becomes ", one latent channel's mean captured and frames blank" |
| x5 | Section 5 | applied | "to each of many targets" becomes "to many targets" |

After the cuts: the submission build is 7 pages (body pages 1 to 4, references 5 to 7) and the appendix build is 12 pages (body 1 to 4, references 5 to 7, appendix 8 to 12). Both have no undefined reference or citation and no overfull box. The submission PDF's text has no match for Rohan, Keerthana, Changliu, CMU, Carnegie, Spiderman, Superman, wandb, huggingface or github, and its PDF author field is "Anonymous Submission".

## Counts

- **Fable (29 items):**
  - applied: 6 (F4, F9, F17, F21, F24, F29)
  - applied-with-the-named-wording: 19
  - rejected-by-one-side: 3 (F5, F18, F23)
  - rejected-for-page-budget: 1 (F6)
- **Astra (32 items):**
  - applied: 2 (A20, A31)
  - applied-with-the-named-wording: 27
  - rejected-by-one-side: 1 (A9)
  - rejected-for-page-budget: 2 (A23, A24)
- **Total (61 items):**
  - applied: 8
  - applied-with-the-named-wording: 46
  - rejected-by-one-side: 4
  - rejected-for-page-budget: 3
- **Page cuts:** 15 applied (c1 to c10, x1 to x5), 1 not applied (c11).

## What the review did not settle

- **Section 3's directional sentence is stale.** It still reads "0.79 to 0.87 per arena so far, against 0.87 at home". The U-Net 200k EMA per-map run is now in on 29 of 30 maps (RESEARCH_CONTEXT 2026-09-26 20:15): unseen arenas 0.641 (arena 1) to 0.938 (arena 9); training maps 0.789 to 0.914 per map, while 0.867 is the pooled validation read. Neither review proposed an edit (F18, the only directional item, was rejected), so the text is unchanged. It needs a fix both sides accept.
- **Limitations power sentence (F23 rejected).** The sentence and its "about 40 percent power at $\rho=0.5$" stay. Astra's objection that the test, alternative and sampling model are unstated is unaddressed.
- **Pointers read "the supplement".** The OpenReview form has no supplement field (Fable F29's open decision). If there is no supplement channel, the `\else` branch of `\appref` should read "the extended version".
- **Tests changed with the text.** Cut c9 and item F27 removed the two sentences `paper/fixtures/test_paper_claims.py` pinned verbatim ("GameNGen tunes its decoder with MSE alone"; the 70M/900M/v2 sentence). Those two tests now check the intent instead: any sentence about GameNGen's decoder tune says MSE and not LPIPS, and any stated training-set size names its arXiv version. `paper/fixtures/test_paper_skeleton.py` now builds both versions and checks anonymity, overfull boxes and `\appref`.
- **Unresolved problems from both lists** (Fable P1 to P7; Astra's "Problems without a fix"):
  - venues cited as preprints;
  - the appendix-build pages when real figures replace the placeholders (Figure 2 will need more than 1.15 inches);
  - the fallback if the full fine-tune slips;
  - provenance for the Table 1 directional values (SD 3.5 140k's 0.852 is in the Sep 26 19:50 log entry);
  - missing `copy_lpips_dec`;
  - the frozen inference rule for censored costs.
