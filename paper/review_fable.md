# Review of the paper skeleton, round 1 (Fable 5.1, 2026-09-26)

Scope: `paper/main.tex`, `paper/appendix.tex`, `paper/refs.bib`, `paper/FIGURES.md` at 4b913d4 (skeleton at 976c4b2). Protocol and priorities from `.claude/analyses/paper-review-brief-2026-09-26.md`.

What I checked before proposing anything. Every number in Table 2, the per-arena appendix table, the Section 2 and 3 text and the Section 4 cost paragraph recomputed from `results/distance_study/scores/040-unet-nexttic_snap_0200000_ema_ddim10/*_h{1,4}/metrics.json`: the training-map means (ceiling 23.91, A 3.60 and 3.86, M -0.027, G 1.55), the unseen means (23.62, 1.51 with 13/13, 0.81 with 11/13, 0.105 with 0/13, 2.48), the per-arena A, A at four tics, M, G, persistence, raw gain and D (against `figure_unet_h1/distance_table.md`), the persistence range 19.4 to 23.2, the Spearmans (model PSNR on persistence 0.945, raw gain -0.77, A -0.05, G -0.80, A on D -0.23, raw gain on D -0.02), the floor band and D ranges, the HUD share 39 to 68 percent on the 17 arenas, the decoder LPIPS 0.08 to 0.12 against persistence 0.15 to 0.23, the four-tic sign changes on arenas 6 and 7, the five arenas past half the home line at step 0 (9, 11, 13, 15, 16; 11 and 13 at 1.801 and 1.805 against 1.799), the sampler trade (0.25 dB, 0.024 LPIPS), the collapse counts, 0.999^250 = 0.78, and the 0.91 estimator check (RESEARCH_CONTEXT 2026-09-25 02:10). All match. The build (pdflatex, bibtex, pdflatex, pdflatex on a scratch copy) runs clean: 11 pages, `body-end` on page 4, no undefined citation or reference, no overfull box. Template: `\documentclass{article}`, `\usepackage{corl_2026}` with no option, abstract of six sentences, `\keywords`, natbib `\citep`, appendix after the bibliography, the style's own `color` and `hyperref` supply `\textcolor` and `\nolinkurl`.

Format of each item: file; current text; replacement (compilable); reason.

## A. Truth

**1. `main.tex`, abstract, sentence 3.** The numbers are the U-Net's and the abstract carries no marks, so it currently attributes them to all three backbones.
Current:
```
Decoded through one decoder, the model still beats copy-last on every unseen arena, but its advantage shrinks from \prov{3.2--4.3}~dB at home to \prov{0.7--2.7}~dB and it loses perceptually on \prov{13 of 13}, while its turn response survives.
```
Replacement:
```
Decoded through one decoder, the U-Net row still beats copy-last on every unseen arena, but its advantage shrinks from \prov{3.2--4.3}~dB at home to \prov{0.7--2.7}~dB and it loses perceptually on \prov{13 of 13} arenas, while its turn response survives.
```
Reason: PixArt and SD 3.5 have not been scored off the training maps; the claim must be exactly as wide as the evidence until the rescore lands (then revert to "each row").

**2. `main.tex`, contribution (2).** Same issue in the contribution statement, which has marks available.
Current:
```
(2) the measurement, for three backbones under one recipe, that the decoded advantage shrinks
```
Replacement:
```
(2) the measurement, on the U-Net row (replication in PixArt and SD~3.5: \tbd{}), that the decoded advantage shrinks
```
Reason: the `% hypothesis:` comment marks the replication as pending, but the printed text does not.

**3. `main.tex`, Section 3, "Where the deficit sits", the ceiling sentence.** A flat ceiling shows the decoder reconstructs true latents equally well on every map; it does not show the decoder renders off-distribution predicted latents equally well, and the sentence's own second half says that test is pending.
Current:
```
The rendering does not degrade: the ceiling is \prov{23.91}~dB at home and \prov{23.62}~dB unseen, so the loss sits in the predicted latents; decoding the same predictions with the stock and tuned decoders (\tbd{}) tests a decoder failing on off-distribution latents.
```
Replacement:
```
The rendering of true latents does not degrade: the ceiling is \prov{23.91}~dB at home and \prov{23.62}~dB unseen, so the loss is not the decoder's floor; decoding the same predictions with the stock and tuned decoders (\tbd{}) tests whether it also renders off-distribution latents worse.
```
Reason: the cohesion decision states the decoder floor as a separate layer; "the loss sits in the predicted latents" is the conclusion of the pending paired decoding, not of the ceiling.

**4. `main.tex`, Section 3, the 2 dB sentence.** A is a decoded quantity; calling it "the latent prediction" presumes the decomposition the next sentences argue, and the two means let the reader check the 2 dB against Table 2.
Current:
```
The latent prediction loses about 2~dB of $A$, stays positive on every arena at one tic, and turns negative at four tics on arenas \prov{6 and 7}.
```
Replacement:
```
$A$ drops by about 2~dB (\prov{3.60} to \prov{1.51}), stays positive on every arena at one tic, and turns negative at four tics on arenas \prov{6 and 7}.
```
Reason: names the measured quantity and its two means.

**5. `main.tex`, Section 3, the perceptual-flip sentence.** "Still loses against the decoded true frame" is true (LPIPS of the decoded prediction against the decoded true frame exceeds raw persistence's LPIPS on 13 of 13 arenas, recomputed) but the reader cannot tell what was compared.
Current:
```
Nor is the perceptual flip the decoder's floor: its own LPIPS (\prov{0.08 to 0.12}) sits below persistence's (\prov{0.15 to 0.23}), and the model still loses against the decoded true frame.
```
Replacement:
```
Nor is the perceptual flip the decoder's floor: its own LPIPS (\prov{0.08 to 0.12}) sits below persistence's (\prov{0.15 to 0.23}), and the model still loses to persistence when scored against the decoded true frame (\prov{13 of 13}).
```
Reason: states the comparison and its count; both verified.

**6. `main.tex`, Section 5, first sentence.** The sentence is about where models are scored; GameNGen's training budget and the arXiv-version aside belong in a reviewer's note, not in a four-page paper.
Current:
```
Game world models are scored on held-out trajectories of their training scenes: GameNGen trains for 700k updates at batch 128 on a random 70M-example subset of its agents' play (arXiv v2, \S4.2; v1 reported 900M generated frames)~\citep{valevski2024gamengen}, DIAMOND on one CS:GO map~\citep{alonso2024diamond}, and MultiGen on 100 generated Doom maps without stating whether its test maps are held out~\citep{po2026multigen}.
```
Replacement:
```
Game world models are scored on held-out trajectories of their training scenes: GameNGen on the levels its agent played~\citep{valevski2024gamengen}, DIAMOND on one CS:GO map~\citep{alonso2024diamond}, and MultiGen on 100 generated Doom maps without stating whether its test maps are held out~\citep{po2026multigen}.
```
Reason: keeps the verified claim (trajectory holdout, lit-scene-generalization memo) and drops the off-topic detail.

## B. The 50 percent floor (Section 4 and the cost table)

I agree with the main session's rule, with one addition: print each arena's half-gap line so the reader can check the crossing, and drop the 25 percent milestone, which has no role once the gap convention is used. The per-arena rule is noisier on small gaps (arena 15's gap is 0.92 dB against a per-arena standard error on A of about 0.2 dB), which is a reason to keep the home line as the second, common crossing, not a reason against the rule. Items 7 to 14 are one change in eight places.

**7. `main.tex`, Section 4, the Cost paragraph (both sentences).**
Current:
```
\textbf{Cost.} An arena's cost is the first budget at which its $A$ reaches half of the training maps' own advantage (\prov{3.60}~dB, frozen at step 0); we also report 25 and 100 percent, the value at 4k and the area under the curve~\citep{taylor2009transfer}, and right-censor arenas that never cross.
Every arena starts below the full line (\prov{0.72 to 2.68}~dB); \prov{5} already sit past half of it at step 0 ($M$ curves: Appendix~\ref{app:perarena-adapt}).
```
Replacement:
```
\textbf{Cost.} An arena's headline cost is the first grid budget at which $A$ closes half of its own gap to the training maps' advantage $A_\text{home}$ (\prov{3.60}~dB), that is, reaches $\tfrac12(A_0+A_\text{home})$ with $A_0$ its zero-shot advantage; the second crossing is $A_\text{home}$ itself.
Both lines are frozen from step-0 predictions; we also report the value at 4k and the area under the curve~\citep{taylor2009transfer}, and right-censor arenas that never cross.
Every arena starts below $A_\text{home}$ (\prov{0.72 to 2.68}~dB), but \prov{5} already sit above half of it, so the line is per arena ($M$ curves: Appendix~\ref{app:perarena-adapt}).
```
Reason: the fixed 50 percent line costs zero updates on five arenas today; percent of gap closed gives every arena a well-defined, non-trivial first crossing (zero percent closed at step 0 by construction), and the absolute home line stays as the common second crossing.

**8. `main.tex`, Table 3 caption and the two milestone rows.**
Current (caption):
```
\caption{What crossing costs (U-Net 200k, 8 episodes, live weights; ${>}$4k: censored). Forgetting: change in the training maps' $A$. Per arena: Appendix~\ref{app:perarena-adapt}.}
```
Replacement (caption):
```
\caption{What crossing costs (U-Net 200k, 8 episodes, live weights; ${>}$4k: censored). Half gap, home line: $A$ closes half of the arena's step-0 gap to the training maps' advantage, or reaches it. Forgetting: change in the training maps' $A$. Per arena: Appendix~\ref{app:perarena-adapt}.}
```
Current (rows):
```
Arenas past 25 / 50 / 100\% by 4k & \tbd{} / \tbd{} / \tbd{} of 13 & -- & -- \\
Cost to 50 / 100\% (updates) & \tbd{} / \tbd{} & \tbd{} / \tbd{} & \tbd{} / \tbd{} \\
```
Replacement (rows):
```
Arenas past half gap / home line by 4k & \tbd{} / \tbd{} of 13 & -- & -- \\
Cost to half gap / home line (updates) & \tbd{} / \tbd{} & \tbd{} / \tbd{} & \tbd{} / \tbd{} \\
```
Reason: the table must name the same two crossings as the text.

**9. `appendix.tex`, Table 8 (`tab:perarena-adapt`), caption, header and rows.** Replace the 25 percent column with each arena's half-gap line, computed from the unrounded step-0 A and the 3.60 dB home line (recomputed tonight), so the reader can read the crossing off the curve.
Current caption:
```
\caption{Per unseen arena, sorted by $D$: step-0 decoded advantage, the first grid budget at which $A$ reaches 25, 50 and 100 percent of the training maps' advantage (${>}$4k: censored), $A$ and $M$ at 4,000 updates, forgetting (change in the training maps' $A$) and the directional score at 4,000.}
```
Replacement caption:
```
\caption{Per unseen arena, sorted by $D$: step-0 decoded advantage $A_0$, the half-gap line $\tfrac12(A_0+A_\text{home})$ with $A_\text{home}=\prov{3.60}$~dB, the first grid budget at which $A$ crosses the half-gap line and the home line (${>}$4k: censored), $A$ and $M$ at 4,000 updates, forgetting (change in the training maps' $A$) and the directional score at 4,000.}
```
Current header and rows:
```
Arena & $D$ & $A$ at 0 & 25\% & 50\% & 100\% & $A$ at 4k & $M$ at 4k & Forgetting & Directional \\
\midrule
8 & \prov{0.115} & \prov{+1.29} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
15 & \prov{0.127} & \prov{+2.68} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
12 & \prov{0.128} & \prov{+1.07} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
17 & \prov{0.130} & \prov{+1.30} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
14 & \prov{0.146} & \prov{+1.33} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
11 & \prov{0.150} & \prov{+1.80} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
13 & \prov{0.156} & \prov{+1.80} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
10 & \prov{0.162} & \prov{+1.67} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
16 & \prov{0.164} & \prov{+1.92} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
1 & \prov{0.179} & \prov{+1.09} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
9 & \prov{0.180} & \prov{+2.05} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
6 & \prov{0.188} & \prov{+0.72} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
7 & \prov{0.282} & \prov{+0.88} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
```
Replacement header and rows:
```
Arena & $D$ & $A_0$ & Half-gap line & Half gap & Home line & $A$ at 4k & $M$ at 4k & Forgetting & Directional \\
\midrule
8 & \prov{0.115} & \prov{+1.29} & \prov{2.44} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
15 & \prov{0.127} & \prov{+2.68} & \prov{3.14} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
12 & \prov{0.128} & \prov{+1.07} & \prov{2.33} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
17 & \prov{0.130} & \prov{+1.30} & \prov{2.45} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
14 & \prov{0.146} & \prov{+1.33} & \prov{2.46} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
11 & \prov{0.150} & \prov{+1.80} & \prov{2.70} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
13 & \prov{0.156} & \prov{+1.80} & \prov{2.70} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
10 & \prov{0.162} & \prov{+1.67} & \prov{2.63} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
16 & \prov{0.164} & \prov{+1.92} & \prov{2.76} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
1 & \prov{0.179} & \prov{+1.09} & \prov{2.35} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
9 & \prov{0.180} & \prov{+2.05} & \prov{2.82} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
6 & \prov{0.188} & \prov{+0.72} & \prov{2.16} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
7 & \prov{0.282} & \prov{+0.88} & \prov{2.24} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
```
Reason: the per-arena threshold is the one number the reader needs to check a per-arena crossing; the column count is unchanged so the tabular spec stays `rccccccccc`.

**10. `main.tex`, Figure 3 caption, and `main.tex` Figure 2 caption.** A per-arena threshold cannot be one dotted line.
Current (Figure 3):
```
\caption{$A$ against LoRA updates per unseen arena, coloured by $D$; dashed: home advantage, dotted: half; open: censored. Black: full fine-tune.}
```
Replacement:
```
\caption{$A$ against LoRA updates per unseen arena, coloured by $D$; dashed: home advantage; a tick on each curve: its half-gap line; open: censored. Black: full fine-tune.}
```
Current (Figure 2, inside the caption):
```
(a) $A$; dashed: the training maps' mean, dotted: half. (b) $M$.
```
Replacement:
```
(a) $A$; dashed: the training maps' mean. (b) $M$.
```
Reason: matches item 7; the half line no longer exists as one line.

**11. `main.tex`, abstract, sentence 5.**
Current:
```
Rank-16 adapters on the frozen backbone regain half of the home advantage within \tbd{} updates on \tbd{} of 13 arenas at \tbd{}~dB of forgetting.
```
Replacement:
```
Rank-16 adapters on the frozen backbone close half of each arena's gap to the home advantage within \tbd{} updates on \tbd{} of 13 arenas, at \tbd{}~dB of forgetting.
```
Reason: item 7.

**12. `main.tex`, contribution (3).**
Current:
```
rank-16 adapters regain half of the home advantage within \tbd{} updates on \tbd{} of 13 arenas, against a full fine-tune
```
Replacement:
```
rank-16 adapters close half of the gap to the home advantage within \tbd{} updates on \tbd{} of 13 arenas, against a full fine-tune
```
Reason: item 7.

**13. `main.tex`, Section 4, "What it takes", first clause.**
Current:
```
\tbd{} of 13 arenas reach half of the home advantage within \tbd{} updates and \tbd{} the full advantage by 4k
```
Replacement:
```
\tbd{} of 13 arenas close half of their gap within \tbd{} updates and \tbd{} reach the home line by 4k
```
Reason: item 7.

**14. `main.tex`, conclusion, and `FIGURES.md`, the `fig:adapt` row.**
Current (conclusion):
```
and regains half of the home advantage within \tbd{} adapter updates.
```
Replacement:
```
and closes half of its gap to the home advantage within \tbd{} adapter updates.
```
Current (`FIGURES.md`):
```
with the training-map line and half of it, first crossings marked
```
Replacement:
```
with the training-map line and each arena's half-gap line, first crossings marked
```
Reason: item 7; FIGURES.md is the figure script's specification.

## C. Voice and clarity

**15. `main.tex`, introduction, sentence 2.** The sentence has no working main verb ("The uses ... ask").
Current:
```
The uses a world model is built for, planning or training a policy in a level nobody showed it, ask what it does on a scene it never saw.
```
Replacement:
```
The uses a world model is built for, planning or training a policy on a level nobody showed it, depend on what it does on a scene it never saw.
```
Reason: grammar; "depend on" also states the stake plainly.

**16. `main.tex`, introduction, paragraph 2, last sentence (61 words).**
Current:
```
Video prediction reported a copy-last baseline~\citep{mathieu2016deep,lotter2017prednet,villegas2019fidelity} that game world models dropped; we make persistence, defined on every map without training, the zero line: every score is a paired difference against it on the same windows~\citep{bruce2024genie}, beside the reconstruction ceiling~\citep{zheng2023occworld,karypidis2024dinoforesight}.
```
Replacement:
```
Video prediction reported a copy-last baseline~\citep{mathieu2016deep,lotter2017prednet,villegas2019fidelity}; game world models dropped it.
We make persistence, defined on every map without training, the zero line: every score is a paired difference against it on the same windows~\citep{bruce2024genie}, beside the reconstruction ceiling~\citep{zheng2023occworld,karypidis2024dinoforesight}.
```
Reason: one idea per sentence.

**17. `main.tex`, Section 2, "Three backbones, one recipe", first sentence (56 words).**
Current:
```
(a 2.27B MMDiT with its own 16-channel latents); backbone and latent space change together, so the rows compare pretrained packages.
```
Replacement:
```
(a 2.27B MMDiT with its own 16-channel latents).
Backbone and latent space change together, so the rows compare pretrained packages.
```
Reason: one idea per sentence.

**18. `main.tex`, Section 2, "What we measure", the directional sentence.** "(0.91 on true frames)" is opaque; it is the estimator check (the true next frame moves the way the control says in 0.91 to 0.92 of decoded windows).
Current:
```
The \emph{directional} score is the fraction of turning windows whose predicted image shift reverses when left and right are swapped in the newest control (0.91 on true frames).
```
Replacement:
```
The \emph{directional} score is the fraction of turning windows whose predicted image shift reverses when left and right are swapped in the newest control (the true next frame moves the way the control says in 0.91 of windows, the estimator's bound).
```
Reason: tells the reader what the 0.91 is and why it is printed.

**19. `main.tex`, Section 3, the distance paragraph.** "Orders nothing" claims more than the one correlation reported.
Current:
```
but within the 13 arenas $D$ orders nothing (Spearman with $A$ \prov{$-0.23$}; 0.56 is needed at $n=13$).
```
Replacement:
```
but within the 13 arenas $D$ does not order $A$ (Spearman \prov{$-0.23$}; $\pm0.56$ is significant at $n=13$).
```
Reason: precision; the threshold's meaning is stated.

**20. `main.tex`, Section 4, Protocol, first sentence (62 words, and "trails far behind" is not idiomatic).**
Current:
```
\textbf{Protocol.} For each unseen arena we adapt the U-Net 200k from its EMA weights with rank-16, $\alpha=16$ LoRA~\citep{hu2022lora} on every attention projection of the frozen backbone, training the control MLP, input projection and noise-bucket embedding in full (4.2M parameters, 0.49 percent), since LoRA alone trails full fine-tuning far behind on DiT-XL/2~\citep{xie2023difffit}; Vista uses this rank on a frozen world model~\citep{gao2024vista}.
```
Replacement:
```
\textbf{Protocol.} For each unseen arena we adapt the U-Net 200k from its EMA weights with rank-16, $\alpha=16$ LoRA~\citep{hu2022lora} on every attention projection of the frozen backbone, the rank Vista uses on a frozen world model~\citep{gao2024vista}.
The control MLP, input projection and noise-bucket embedding train in full (4.2M parameters, 0.49 percent), because on DiT-XL/2 LoRA alone falls far short of full fine-tuning~\citep{xie2023difffit}.
```
Reason: two sentences, each with its own citation; the DiffFit fact is as the lit-adapters memo verified it (Table 1: LoRA-R16 FID 81.3 against 16.6 full).

**21. `main.tex`, Section 4, Protocol, second sentence.** Section 2 gives each arena 16 adaptation episodes; Section 4 trains on 8 without saying which.
Current:
```
We train on 8 episodes (learning rate $10^{-4}$, batch 32)
```
Replacement:
```
We train on 8 of the 16 adaptation episodes (learning rate $10^{-4}$, batch 32)
```
Reason: matches `adapt_split.py`'s defaults (16 adapt / 8 held out, step curve at 8; RESEARCH_CONTEXT 2026-09-26 20:10).

**22. `main.tex`, Section 5, last sentence (58 words).**
Current:
```
Dataset distances predict fine-tuned outcomes~\citep{alvarezmelis2020otdd,nguyen2025sotdd}, directed coverage better than symmetric transport~\citep{mensink2021factors,westny2026latent}, but the cost of crossing to each of many targets of one world model has not been measured, nor, that we found, EMA against live weights in closed loop.
```
Replacement:
```
Dataset distances predict fine-tuned outcomes~\citep{alvarezmelis2020otdd,nguyen2025sotdd}, directed coverage better than symmetric transport~\citep{mensink2021factors,westny2026latent}.
We found no measurement of the cost of crossing from one world model to each of many targets, nor of EMA against live weights in closed loop.
```
Reason: two sentences; "that we found" becomes the subject instead of an aside.

**23. `main.tex`, Section 6, first sentence (a semicolon chain of four limits).**
Current:
```
With arena 7 carrying two thirds of the arenas' spread in $D$, the cost-against-distance test has about 40 percent power at $\rho=0.5$; persistence is weak on fight footage; adapted models are scored one tic ahead; each row is one seed, agent and WAD at 10 sampling steps.
```
Replacement:
```
Arena 7 carries two thirds of the arenas' spread in $D$, so the cost-against-distance test has about 40 percent power at $\rho=0.5$.
Persistence is weak on fight footage, adapted models are scored one tic ahead, and each row is one seed, agent and WAD at 10 sampling steps.
```
Reason: the power statement is the limit a reviewer will weigh; it gets its own sentence.

## D. Template and form

**24. `main.tex`, Table 2 caption.** The caption should define its columns and should not rely on a colour the clean build removes.
Current:
```
\caption{Off the training maps: means over maps (in parentheses, maps beating copy-last). Blue: pre-fresh-set episodes, stock decoder, full frame. The PixArt and SD~3.5 rows take the U-Net's two-row form.}
```
Replacement:
```
\caption{Off the training maps: means over maps (in parentheses, maps beating copy-last: $A>0$, $M<0$). U-Net numbers: pre-fresh-set episodes, stock decoder, full frame\ifdraftmarks{} (blue)\fi. The PixArt and SD~3.5 rows take the U-Net's two-row form.}
```
Reason: says what the parentheses count; `\draftmarksfalse` prints the numbers in black, and "Blue:" would then point at nothing.

**25. `main.tex`, Table 1 caption, and `appendix.tex`, Table 5 (`tab:perarena`) caption.** Same colour dependence.
Current (Table 1):
```
Blue: provisional; $^\dagger$150k.}
```
Replacement:
```
\ifdraftmarks Blue: provisional; \fi$^\dagger$150k.}
```
Current (Table 5):
```
Blue: pre-fresh-set episodes. $^\ast$pooled over the four training maps.}
```
Replacement:
```
\ifdraftmarks Blue: pre-fresh-set episodes. \fi$^\ast$pooled over the four training maps.}
```
Reason: as item 24; the conditional is balanced inside the caption and compiles in a moving argument.

**26. `main.tex`, Section 3, the sampler clause (page budget).** A first version of this list pushed Section 6 onto page 5 (scratch build); with the wording above plus this cut and item 27, all 32 replacements applied together compile clean with `body-end` on page 4 and the references starting on page 5. The sampler sweep is already in Appendix A.
Current:
```
We expect blur under uncertainty, which PSNR rewards and LPIPS penalises; the sampler sets part of it (50 steps instead of 10 trade 0.25~dB for 0.024 LPIPS at home).
```
Replacement:
```
We expect blur under uncertainty, which PSNR rewards and LPIPS penalises (the sampler's share: Appendix~\ref{app:protocol}).
```
Reason: the numbers survive in the appendix; the body keeps the mechanism.

**27. `main.tex`, Section 2, the two decoder sentences (page budget).**
Current:
```
Rollouts stay in latent space, so the decoder only renders. GameNGen tunes its decoder with MSE alone, and so do we, once, on training-map frames; it then stays frozen.
```
Replacement:
```
Rollouts stay in latent space, so the decoder only renders; as GameNGen does, we tune it once with MSE on training-map frames, then freeze it.
```
Reason: the same facts in fewer words.

## Problems without a proposed fix

1. **Venues cited as preprints.** `refs.bib` cites Genie, OccWorld, PixArt-alpha, SD3, Mensink and the ViZDoom competitions as arXiv preprints with VERIFY comments (ICML 2024, ECCV 2024, ICLR 2024, ICML 2024, IEEE TPAMI, IEEE Transactions on Games per the memos). DBLP refused my lookups tonight (access denied and rate limits), so I cannot certify them; a reader will notice Genie and SD3 as preprints. Someone with a browser should settle these six before Sep 30; nothing else in the bibliography needs work.
2. **The abstract's second sentence still promises three backbones off the training maps** ("we train three pretrained diffusion backbones on four arenas and ask what they keep, lose and relearn"). If the PixArt and SD 3.5 rescore does not land by the internal cutoff, this sentence and Table 2's two empty rows need a decision: report one row honestly, or hold the sentence.
3. **The full fine-tune comparator is conditional** (Monday if a card idles, else October) but Section 4, Table 3's third column and Appendix E are written as if it exists. There is no fallback wording; if it slips, the column and the appendix paragraph must go and the Protocol sentence "A full fine-tune of arena 7, the farthest, is the comparator" must become a limitation.
4. **Page budget with real figures.** The body ends on page 4 with the three figure slots at 1.0 to 1.15 inches. Figure 2 at a readable size for 17 labelled arenas with intervals will need more height than that; expect the related-work or the sampler sentence to be the first cut once the PDFs exist.
5. **The 30-map test under the single-dataset rule.** Section 0 says the campaign maps stay out of the paper, but Section 3 and Appendix C report the pre-registered 30-map partial Spearman and the five checks on the 30-map set. Reporting a pre-registered result is right; the body should say in one clause that the 30-map set is the pre-registration set and that the paper's tables use the 17 arenas only. Appendix C says it; the body does not.
6. **Directional numbers in Table 1.** The U-Net's 0.867 is the 2026-09-25 02:10 read; RESEARCH_CONTEXT section 0 item 2 says 0.85 for the U-Net 200k EMA and 0.86 for PixArt 155k, and I could not find the read that gives SD 3.5 140k's 0.852 (the 130k read is EMA 0.844, live 0.867). The three values need one provenance line in FIGURES.md.
7. **Figure 1's last caption sentence** ("On the unseen arena the model follows the control but loses texture first") is written before any frames were chosen; it is marked as a hypothesis in a comment and must be rewritten from the frames the seeded rule picks.
