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

## On the other list

Read against `paper/review_astra.md` in full, 2026-09-26 night. Before marking anything I rechecked the facts Astra's items turn on: `lpips_dec` is the decoded prediction against the decoded truth (`eval_tf.py` line 149) and exceeds raw persistence's LPIPS on 13 of 13 unseen arenas, while `copy_lpips_dec` is indeed absent from the legacy h1 files, as Astra says; persistence's HUD PSNR is finite, 87.8 to 96.2 dB across the 17 arenas, so "copies it exactly" needs "almost"; the HUD share as computed from the mean PSNRs is the geometric mean over windows of the per-window share (39.2 to 67.8 percent), as Astra says; `lora.py`'s `control` part is `control_history` plus `action_embedder`, so the position table is inside what the paper calls the control MLP; the lit-transferability memo (Table 10 of Mensink, Table 2 of Westny) confirms that Westny's directed measure is a Gaussian latent KL, not coverage; XEWorld §3.1 says of its two held-out robots "we report their results separately and never average them" (fetched tonight). Counts: AGREE 1, AGREE WITH CHANGE 28, DISAGREE 1.

Where an Astra item and one of mine address the same passage, the winner is stated on the line. Summary of the five shared passages: the half-gap cost rule takes my items 7 to 14 (compact, no $C_m(t)$ notation, no four-decimal $H$) plus Astra's censored-median clause and "negative is loss"; the abstract keeps its six sentences with my items 1 and 11 and the verb of Astra's 18; the decoded-advantage passage takes my items 3, 4 and 5 with item 5 amended to name the comparison the files support; XEWorld takes Astra's fact with a shorter sentence that keeps the rarity claim; the HUD share takes Astra's statistic with the numbers kept in both places.

**A1.** AGREE WITH CHANGE: same rule, my item 7 wording wins (three sentences, no per-arena function notation, no "Provisionally, $H=3.5984$" or "The revised halfway target is" review prose in the paper; the "arena already at $H$" clause has no case today, every arena starts below). Quarter-gap milestone dropped, see A3.

**A2.** AGREE WITH CHANGE: my item 8 caption plus Astra's censored-median rule and sign convention, exactly:
```
\caption{What crossing costs (U-Net 200k from its EMA weights, 8 episodes, live weights scored; ${>}$4k: censored, and a median over the 13 arenas is censored when fewer than 7 cross). Half gap, home line: $A$ closes half of the arena's step-0 gap to the training maps' advantage, or reaches it. Forgetting: change in the training maps' $A$, negative is loss. Per arena: Appendix~\ref{app:perarena-adapt}.}
```

**A3.** AGREE WITH CHANGE: my item 8 rows win (25 percent dropped everywhere, since a percent-of-gap ladder needs no quarter step and the appendix column is better spent on each arena's half-gap line); if Astra wants 25 percent kept, keep it in both tables by adding a column to Table 8 rather than replacing the half-gap line.

**A4.** AGREE WITH CHANGE: my item 9 caption, header and rows win (they print the per-arena threshold), with Astra's sign convention folded in: `forgetting (change in the training maps' $A$, negative is loss)` replaces `forgetting (change in the training maps' $A$)` in my item 9 caption.

**A5.** AGREE WITH CHANGE: my item 10 plus Astra's "markers: first crossings"; "planned" is not caption prose. Exactly:
```
\caption{$A$ against LoRA updates per unseen arena, coloured by $D$; dashed: home advantage; a tick on each curve: its half-gap line; markers: first crossings; open: censored. Black: full fine-tune.}
```

**A6.** AGREE WITH CHANGE: the axis stays logarithmic with step 0 drawn as the first labelled tick (symlog or a broken axis), because the grid 250, 500, 1k, 2k, 4k doubles and a linear axis crushes the first three points into a tenth of the width; the rest of Astra's row stands. Exactly:
```
| `fig:adapt` (Figure 3) | Adaptation curves, one line per unseen arena coloured by D: `A` against LoRA updates (log axis with step 0 as the first labelled tick, grid 0 to 4,000) with the frozen training-map line and each arena's half-gap line (A_0 + home)/2 as a tick on its curve; first crossings marked, non-crossers open at the final budget, the full fine-tune on arena 7 in black. One panel, as Rohan specified; the `M` curves the cost decision called a second panel are Figure 5 in the appendix. Shares one float row with Figure 2 (one third of the width) | `score_adapt.py score` rows (`scores.jsonl`, live weights) from the 13-arena U-Net LoRA runs (`adapt_wm.py`), the home line and each arena's A_0 frozen from step-0 predictions through the tuned SD 1 decoder | `figures/fig3_adaptation_curves.pdf` | pending (runs launch Sep 27 evening; arenas 8, 16, 6, 7 first as the fallback set) |
```

**A7.** AGREE WITH CHANGE: the abstract keeps its six sentences; sentence 3 takes my item 1 (U-Net named), sentence 5 takes my item 11 (half of each arena's gap), sentence 6 takes the verb of A18: `In closed loop, good one-step scores do not guarantee stability, and the EMA reduces collapses without removing them.`; sentences 1, 2 and 4 stay. Astra's sentences 2 and 5 ("currently cover", "We plan to measure") are status reports, and the skeleton's convention holds pending status in `\tbd` and the `% hypothesis:` comments and rewrites the sentence from the numbers when they land (Sep 28), in either direction.

**A8.** AGREE WITH CHANGE: my items 2 and 12 win (they mark the replication and the adaptation result as pending inside the printed text without dropping the numbers that exist); contribution (4) takes A18's verb: `one-step quality does not guarantee closed-loop stability`.

**A9.** DISAGREE: every clause carries `\tbd` and the paragraph's `% hypothesis:` comment, by the writing brief's convention, and is rewritten from `score_adapt.py cost` on Sep 28 whichever way the numbers fall; "Adaptation results are pending" is not submission prose either and would be rewritten just the same, and the fallback if the runs slip is my problem 3, not a rewrite now.

**A10.** AGREE WITH CHANGE: my item 14 plus the U-Net named (my item 1), exactly:
```
Off its training maps the U-Net keeps its turn response and part of its advantage over copying, loses its perceptual advantage, and closes half of its gap to the home advantage within \tbd{} adapter updates.
```

**A11.** AGREE WITH CHANGE: Astra's fact wins ("pooled" is wrong for XEWorld, verified tonight), but the replacement drops the rarity sentence and the Doom absence claim and duplicates Section 5; the shorter fix, exactly:
```
The measurement is rare. Held-out domains in world-model papers are an embodiment, a game, an environment or a building~\citep{chen2026xeworld,rigter2024avid,gao2025adaworld,koh2021pathdreamer}, and we found no Doom world model scored on maps it did not train on.
```

**A12.** AGREE WITH CHANGE: my item 16's split wins, with Astra's two corrections folded in ("every score" overclaims because Table 1 prints absolute PSNR; the Genie citation moves to the definition, A13). Exactly:
```
Video prediction reported a copy-last baseline~\citep{mathieu2016deep,lotter2017prednet,villegas2019fidelity}; game world models dropped it.
We make persistence, defined on every map without training, the zero line: each transfer score is a paired difference against it on the same windows, read beside the reconstruction ceiling~\citep{zheng2023occworld,karypidis2024dinoforesight}.
```

**A13.** AGREE WITH CHANGE: the aggregation belongs here and Genie's $\Delta$PSNR is between inferred and random actions, so cite it as the paired form, not as the same metric. Exactly:
```
The \emph{decoded advantage} $A=\mathrm{PSNR}(D(\hat z),D(z))-\mathrm{PSNR}(D(z_\text{last}),D(z))$ puts the model and copy-last through the same decoder; it is averaged over windows, then equally over maps.
It is a paired difference on the same frame, as Genie's $\Delta$PSNR is between inferred and random actions~\citep{bruce2024genie}.
```

**A14.** AGREE WITH CHANGE: the paper states the final protocol (scene rows) and `\prov` covers today's full-frame numbers, so the first clause stays; Astra is right on "exactly" (persistence's HUD PSNR is 88 to 96 dB, finite) and on the statistic (a geometric mean over windows). My wording wins for the sentence, Astra's statistic wins for the number. Exactly:
```
Pixel quantities use the scene rows: per window the HUD carries \prov{39 to 68} percent of the stock decoder's squared error (geometric mean over windows) and persistence copies it almost exactly (\prov{88 to 96}~dB). We never rank maps on raw PSNR against persistence, which tracks how static the footage is (Spearman \prov{$-0.77$}).
```

**A15.** AGREE WITH CHANGE: the mechanism sentence is hedged rather than deleted ("however good the latent prediction is" was the overclaim), the HUD statistic is stated as what it is, and Astra's last sentence ("This is not the fraction ... requires per-window MSEs") is a reviewer's note, not paper text. Exactly:
```
\textbf{What the variation between arenas follows.} Across the 13 unseen arenas, raw gain over persistence tracks persistence PSNR (Spearman \prov{$-0.77$}) and the model's absolute PSNR tracks it at \prov{0.95}, while $A$ does not (\prov{$-0.05$}), consistent with a floor on the model's pixel error, from the decoder and from blur, that raw persistence lacks. The gap $G$ tracks persistence too (\prov{$-0.80$}), which is why it is a table column and not a cost axis. The 32 HUD rows, which the stock decoder renders at about 18~dB and persistence copies almost exactly (\prov{88 to 96}~dB), carry \prov{39 to 68} percent of the ceiling's squared error per window (geometric mean over windows, from the mean PSNRs) on the 17 arenas.
```

**A16.** AGREE WITH CHANGE: my items 4, 3 and 5 win for the three sentences (they name the measured quantity and its two means, keep the pending paired decoding as the test, and state the flat ceiling rather than "changes less"; "2.09 dB" is over-precise for a provisional mean); Astra is right that `copy_lpips_dec` is not in the files, so item 5 is amended to name the comparison they do support, exactly:
```
Nor is the perceptual flip the decoder's floor: its own LPIPS (\prov{0.08 to 0.12}) sits below persistence's (\prov{0.15 to 0.23}), and the model, scored against the decoded true frame, still has higher LPIPS than raw persistence (\prov{13 of 13}).
```

**A17.** AGREE WITH CHANGE: the heading and the floor-band sentence stay (the family separation is the cohesion decision's headline and the floor band is the evidence for it), the within-arena clause takes my item 19, and the 30-map sentence changes to state the failed gate first and the amended verdict in its own words. Exactly, replacing the third sentence:
```
The pre-registered 30-map test, which also held campaign maps of another WAD, failed its validation gate (Appendix~\ref{app:distance}).
Its partial Spearman of $-0.73$ [$-0.84$, $-0.32$] falls to \prov{$-0.42$} with family indicators as covariates, so we read it as qualified evidence of a pooled association, not of an ordering.
```

**A18.** AGREE WITH CHANGE: the verb is right (one checkpoint pair shows coexistence, not the absence of association) but the specifics stay. Exactly:
```
\textbf{Closed loop.} Good one-step scores do not guarantee stable rollouts: SD~3.5 has the best one-tic numbers, yet at 50k and 70k its live weights fall into an absorbing state where one latent channel's mean is captured and frames go blank (Appendix~\ref{app:closedloop}).
```
The abstract's sentence 6 and contribution (4) take the same verb (A7, A8).

**A19.** AGREE WITH CHANGE: one sentence, stating the sampler as the held factor without denying an interaction. Exactly:
```
Both share the sampler, so the few-step sampling DIAMOND blames for drift~\citep{alonso2024diamond} is not what differs between them.
```

**A20.** AGREE.

**A21.** AGREE WITH CHANGE: same problem as my problem 7; drop the hypothesis sentence and keep it a caption rather than a plan. Exactly:
```
\caption{Rollouts under held controls (turn left, forward, strafe right) on training arena \tbd{} (left) and unseen arena \tbd{} (right), windows chosen by a seeded rule. Rows: truth, U-Net 200k EMA, copy-last; tics 1, 4, 8, 16, 32.}
```

**A22.** AGREE WITH CHANGE: my item 20 wins ("We plan" is not submission prose, and DiffFit's Table 1 is the stated reason for the design choice, which is what "because" says), with the position table added, since `lora.py`'s `control` part trains it. Second sentence exactly:
```
The control MLP with its position table, the input projection and the noise-bucket embedding train in full (4.2M parameters, 0.49 percent), because on DiT-XL/2 LoRA alone falls far short of full fine-tuning~\citep{xie2023difffit}.
```

**A23.** AGREE WITH CHANGE: same as my problem 3; the skeleton marks a pending decision with `\todo`, not with "this run is pending". Exactly:
```
A full fine-tune of arena 7, the farthest, is the comparator\todo{conditional: Monday if a card idles, else drop this sentence, Table 3's third column and the Appendix E clause, and state it as a limitation}.
```

**A24.** AGREE WITH CHANGE: my item 6 wins; it cuts the same training-budget aside but keeps the sentence Section 5 exists to make ("scored on held-out trajectories of their training scenes").

**A25.** AGREE WITH CHANGE: the verified specifics stay (appearance distance, 25 to 75 episodes, forgetting a seen robot); "None scores per scene against persistence" is a claim about these four papers, which were read, not a broad absence claim, and Astra's own reading of XEWorld (per embodiment, not per scene, not against persistence) supports it. The one change, in the first clause: `XEWorld holds out two robot embodiments, scored separately, finds held-out error tracking an appearance distance, ...` with the rest of both sentences unchanged.

**A26.** AGREE WITH CHANGE: my item 22 split wins, with Astra's correction that the two directed measures differ (memo: Mensink's nearest-source coverage against EMD, Westny's Gaussian latent KL against Wasserstein); the sentence stays a comparison, which is what the two tables show, not a causal claim. Exactly:
```
Dataset distances predict fine-tuned outcomes~\citep{alvarezmelis2020otdd,nguyen2025sotdd}; in two studies a directed measure, nearest-source coverage~\citep{mensink2021factors} or latent KL~\citep{westny2026latent}, predicted transfer better than a symmetric transport distance.
We found no measurement of the cost of crossing from one world model to each of many targets, nor of EMA against live weights in closed loop.
```

**A27.** AGREE WITH CHANGE: one sentence naming the two measures; Astra's second sentence repeats what the `\todo` before it already lists (the validation of the chosen distance). Exactly:
```
In two studies a directed measure predicted transfer better than a symmetric transport distance: nearest-source coverage against EMD~\citep{mensink2021factors} and Gaussian latent KL against Wasserstein~\citep{westny2026latent}.
```

**A28.** AGREE WITH CHANGE: the row also covers the appendix use and records the persistence HUD PSNR the "almost exactly" wording rests on. Exactly:
```
| Section 2, Appendix B | HUD share of the ceiling's squared error per window, geometric mean over windows, 39 to 68 %; persistence HUD PSNR 88 to 96 dB | (32/240) × 10^((mean `vae_psnr` − mean `hud_vae_psnr`)/10) per arena, and `persist_hud_psnr_raw`, from the 17 h1 `metrics.json` | provisional |
```

**A29.** AGREE WITH CHANGE: with the naming of A3 (25 percent dropped) and Astra's median rule kept. Exactly:
```
| `tab:cost` (Table 3) | What crossing costs: LoRA over 13 arenas (median), LoRA on arena 7, full fine-tune on arena 7; trained parameters and GPU-hours, arenas past the half-gap line and the home line by 4k, cost to each (first grid crossing), `A`, `M`, `G` at 4,000, forgetting and directional | `score_adapt.py cost` after implementing the per-arena half-gap line (A_0 + home)/2 (right-censored; a median over 13 arenas is censored when fewer than 7 cross, never a median over crossers alone) and the guard rows; parameter counts from `lora.parameter_counts` | table in `main.tex` | trained-parameter row exists; everything else pending (full fine-tune: Monday if a card idles and arena 7's LoRA curve is flat, else October) |
```

**A30.** AGREE WITH CHANGE: same naming. Exactly:
```
| `tab:perarena-adapt` | Per unseen arena: D, step-0 `A`, its half-gap line (A_0 + home)/2, first grid crossing of the half-gap line and the home line, `A` and `M` at 4,000, forgetting, directional | `score_adapt.py` rows and cost | D, step-0 `A` and the half-gap lines provisional; the rest pending |
```

## New proposals (round 2)

**28. Anonymisation for the double-blind submission.** What I checked: `corl_2026.sty` without `[final]` prints "Anonymous Author(s)" (line 430) and sets `pdfauthor={Anonymous Submission}` (line 117); a grep of `main.tex` and `appendix.tex` for Hugging Face, W&B, GitHub, author, university, lab and city strings finds only the author block (lines 45 to 49); the benchmark link already reads "link withheld for review"; `refs.bib` cites no work of the authors; source comments (Rohan, Overleaf, Spiderman paths) never render and the form takes only the PDF. The one rendered risk is a `[final]` build by mistake, so the block itself is replaced and the original kept in a `% camera-ready:` comment.
Current (`main.tex`):
```
\author{
  Rohan Nagabhirava \quad Keerthana Chirumamilla \quad Changliu Liu\\
  Carnegie Mellon University\\
  Pittsburgh, PA, United States
}
```
Replacement:
```
% camera-ready: \author{Rohan Nagabhirava \quad Keerthana Chirumamilla \quad Changliu Liu\\ Carnegie Mellon University\\ Pittsburgh, PA, United States}
\author{
  Anonymous Author(s)\\
  Anonymous Institution\\
  Anonymous City, Country
}
```
Rule for everything still to come: figure PDFs carry no W&B run names, server paths (`/sata2/data/rnagabhi`, Superman, Spiderman) or account names in titles or legends; before the upload, `pdftotext main.pdf - | grep -inE 'nagabhirava|chirumamilla|carnegie|cmu|pittsburgh|rnagabhi|sata2|wandb|huggingface|github'` must return nothing, and the same grep on the two `.tex` files must return only `% camera-ready:` lines. If any identifying string appears later in the text, the same pattern applies: anonymous placeholder in the text, original in a `% camera-ready:` comment on the line above.

**29. The submission build is body plus references.** `\withappendixfalse` becomes the default and the seven body references to the appendix (lines 106, 113, 120, 192, 199, 217, 228, plus any added by A17 and my item 26) go through a macro, so the body-only build has no undefined reference (Astra's packaging note). Reason: the call allows four pages excluding references and says nothing about an appendix, and the OpenReview form takes one PDF; the appendix is kept for the camera-ready (`\withappendixtrue`) or an anonymous supplement.
Current (`main.tex`, line 36):
```
\newif\ifwithappendix \withappendixtrue
```
Replacement:
```
\newif\ifwithappendix \withappendixfalse % submission: body plus references; \withappendixtrue for the camera-ready or an anonymous supplement
\newcommand{\appref}[1]{\ifwithappendix Appendix~\ref{#1}\else the supplement\fi}
```
And each `Appendix~\ref{app:X}` in the body becomes `\appref{app:X}`, for example `(\appref{app:protocol})` at line 106 and `Per arena: \appref{app:perarena-adapt}.` in the Table 3 caption. The header comment (lines 6 to 9) changes its last clause to `set \withappendixtrue for the camera-ready or the supplement build`. The fixture `paper/fixtures/test_paper_skeleton.py` needs no change: the appendix still follows the bibliography and the clean-build test now runs on the submission build. Open decision for Rohan: if the workshop offers no supplement channel, the `\else` branch should read `the extended version` and the appendix becomes the arXiv version's.

## Page budget of the union (measured, round 2)

I built a scratch copy with the union of both lists as marked above (my 27 items, Astra's 30 with the changes given, proposals 28 and 29). It compiles clean through pdflatex, bibtex, pdflatex, pdflatex: no undefined citation or reference, no overfull box, PDF author "Anonymous Submission", no identifying string in the PDF text, the seven body references print "the supplement". But `body-end` lands on page 5: the Astra-derived additions (A2, A13, A14, A16, A17, A23, A26, about one line each) push three lines of Section 3 onto page 4 and Section 6 onto page 5. Note that the style sets `\widowpenalty` and `\clubpenalty` to 10000, so Section 6 (a heading and a five-line paragraph) needs about four free lines before any of it returns to page 4, which is why small cuts do nothing until a threshold. The cuts below are exact, were applied cumulatively and compile; after c1 to c9 the overrun is three lines; c10 and c11 do not yet cross the threshold. The remaining lines come from the figure heights once the PDFs exist (my problem 4), or from a paragraph-level decision by the main session.

**c1.** A23's `\todo{}` disappears when the comparator decision lands (1 line).
**c2.** Table 2 caption: delete ` The PixArt and SD~3.5 rows take the U-Net's two-row form.` (the table shows it).
**c3.** A2 caption shortened; this supersedes the A2 text above:
```
\caption{What crossing costs (U-Net 200k from EMA weights, 8 episodes, live weights; ${>}$4k: censored, medians too when fewer than 7 cross). Half gap, home line: $A$ closes half of its step-0 gap to the home advantage, or reaches it. Forgetting: change in the training maps' $A$, negative is loss. Per arena: \appref{app:perarena-adapt}.}
```
**c4 (with c9).** A11 in its short form, which also removes the intro's duplication of Section 5 that Astra noted; this supersedes the A11 text above:
```
The measurement is rare: we found no Doom world model scored on maps it did not train on (MultiGen does not say~\citep{po2026multigen}), and other held-out domains are an embodiment, a game, an environment or a building (Section~\ref{sec:related}).
```
**c5.** Section 5, XEWorld: `, and fine-tunes on 25 to 75 episodes, forgetting a seen robot~\citep{chen2026xeworld}` becomes `, and fine-tunes on 25 to 75 episodes~\citep{chen2026xeworld}`.
**c6.** Section 4, Protocol: `with the decoder frozen so that rendering repair cannot count as adaptation, and` becomes `with the decoder frozen, and`.
**c7.** Section 3, first sentence: `In domain all three rows beat persistence at one and four tics, within 0.85~dB of each other, and turn the right way (Table~\ref{tab:indomain}).`
**c8.** Section 2, Data: `The models train on arenas 2 to 5 (500 episodes each) and are scored there on 25 held-out episodes per map.`
**c9.** Section 5: delete its first sentence (GameNGen, DIAMOND, MultiGen); the intro's first sentence already cites GameNGen and DIAMOND for trajectory holdout, and MultiGen moves into c4. This supersedes my item 6 and A24 (2 lines).
**c10 (candidate).** Contribution (2) without the numbers the abstract already prints: `(2) the measurement, on the U-Net row (replication in PixArt and SD~3.5: \tbd{}), that the decoded advantage shrinks off the training maps and flips perceptually while the ceiling stays flat and the turn response survives (Figure~\ref{fig:strips});`
**c11 (candidate).** Section 5: `None scores per scene against persistence~\citep{lample2017arnold,wydmuch2018vizdoom}.` (the Doom-agent clause goes; the citations stay).

Filled tables make it worse before better: Table 2's PixArt and SD 3.5 rows add two lines each when they land, and Figure 2 needs more than 1.15 inches for 17 labelled arenas with intervals. The main session should plan the body at 4 pages against those, not against today's placeholders.
