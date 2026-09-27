# Astra review — round 1, independent proposals

Most important: **1** (fix the half-home cost floor), **7** (separate observed U-Net findings from pending experiments in the abstract), and **16** (do not equate decoded advantage with decoder-free latent evidence).

Proposals only. Apply the cost-related items together. Each current block occurs exactly once in its named file. TeX replacements use the existing preamble; Markdown replacements are specifications, not TeX. Percentages, budgets and formulas are proposed definitions; empirical values are audited below. Numbers quoted in “Current” are not endorsements.

1. **Define cost by the gap remaining at step zero**

   File: `paper/main.tex`

   Current:

```tex
\textbf{Cost.} An arena's cost is the first budget at which its $A$ reaches half of the training maps' own advantage (\prov{3.60}~dB, frozen at step 0); we also report 25 and 100 percent, the value at 4k and the area under the curve~\citep{taylor2009transfer}, and right-censor arenas that never cross.
Every arena starts below the full line (\prov{0.72 to 2.68}~dB); \prov{5} already sit past half of it at step 0 ($M$ curves: Appendix~\ref{app:perarena-adapt}).
```

   Replacement:

```tex
\textbf{Cost.} We freeze the training-map mean $H$ and each arena's zero-shot $A_m(0)$ before adapted scores, using the same decoder, crop and evaluation windows throughout.
For $A_m(0)<H$, gap closure is $C_m(t)=[A_m(t)-A_m(0)]/[H-A_m(0)]$.
Cost is the first evaluated budget with $C_m(t)\geq0.5$; we also report quarter-gap and full-home crossings, terminal advantage and area under the curve~\citep{taylor2009transfer}.
Non-crossers are right-censored at the final budget; an arena already at $H$ has zero home-crossing cost and no gap-closure score.
Provisionally, $H=\prov{3.5984}$~dB and arenas \prov{9, 11, 13, 15 and 16} already exceed $H/2$.
The revised halfway target is $[A_m(0)+H]/2$.
```

   Reason: The absolute half-home target gives zero adaptation cost to already-qualified arenas; gap closure fixes that floor without hiding zero-shot differences.

2. **Define the percentages and censoring in the cost caption**

   File: `paper/main.tex`

   Current:

```tex
\caption{What crossing costs (U-Net 200k, 8 episodes, live weights; ${>}$4k: censored). Forgetting: change in the training maps' $A$. Per arena: Appendix~\ref{app:perarena-adapt}.}
```

   Replacement:

```tex
\caption{Adaptation cost (U-Net 200k EMA initialization, 8 episodes, live weights). Percentages denote the zero-shot-to-home gap closed; 100\% reaches home. Costs are first evaluated crossings; ${>}$4k means censored, including the median if fewer than half cross. Forgetting: change in training-map $A$ (negative means loss). Per arena: Appendix~\ref{app:perarena-adapt}.}
```

   Reason: The caption must distinguish gap fractions from fractions of home score and must not silently omit censored arenas from the median.

3. **Label cost-table rows with the new estimand**

   File: `paper/main.tex`

   Current:

```tex
Arenas past 25 / 50 / 100\% by 4k & \tbd{} / \tbd{} / \tbd{} of 13 & -- & -- \\
Cost to 50 / 100\% (updates) & \tbd{} / \tbd{} & \tbd{} / \tbd{} & \tbd{} / \tbd{} \\
```

   Replacement:

```tex
Arenas closing 25 / 50 / 100\% by 4k & \tbd{} / \tbd{} / \tbd{} of 13 & -- & -- \\
Cost to half-gap / home (updates) & \tbd{} / \tbd{} & \tbd{} / \tbd{} & \tbd{} / \tbd{} \\
```

   Reason: Otherwise the table preserves the old definition after the prose changes.

4. **Define the per-arena crossing columns consistently**

   File: `paper/appendix.tex`

   Current:

```tex
\caption{Per unseen arena, sorted by $D$: step-0 decoded advantage, the first grid budget at which $A$ reaches 25, 50 and 100 percent of the training maps' advantage (${>}$4k: censored), $A$ and $M$ at 4,000 updates, forgetting (change in the training maps' $A$) and the directional score at 4,000.}
```

   Replacement:

```tex
\caption{Per unseen arena, sorted by $D$: zero-shot decoded advantage $A_m(0)$ and first evaluated budgets closing 25, 50 and 100 percent of the gap to the frozen training-map mean $H$. The halfway target is $[A_m(0)+H]/2$; 100 percent reaches $H$. ${>}$4k denotes right-censoring. Remaining columns report $A$, $M$, the change in training-map $A$ (negative means forgetting), and the directional score at 4,000 updates.}
```

   Reason: The appendix must use the same thresholds and forgetting sign as the main table.

5. **Show arena-specific halfway targets**

   File: `paper/main.tex`

   Current:

```tex
\caption{$A$ against LoRA updates per unseen arena, coloured by $D$; dashed: home advantage, dotted: half; open: censored. Black: full fine-tune.}
```

   Replacement:

```tex
\caption{$A$ against adapter updates; colour: frame distance $D$. Dashed: home mean $H$; ticks: arena-specific halfway targets $[A_m(0)+H]/2$. Markers: first crossings; open endpoints: censored. Black: planned full fine-tune.}
```

   Reason: A shared horizontal half-gap line is impossible when arenas have different zero-shot scores.

6. **Update the figure specification and retain step zero**

   File: `paper/FIGURES.md`

   Current:

```markdown
| `fig:adapt` (Figure 3) | Adaptation curves, one line per unseen arena coloured by D: `A` against LoRA updates (log axis, grid 0 to 4,000) with the training-map line and half of it, first crossings marked, censored arenas open, full fine-tune on arena 7 in black. One panel, as Rohan specified; the `M` curves the cost decision called a second panel are Figure 5 in the appendix. Shares one float row with Figure 2 (one third of the width) | `score_adapt.py score` rows (`scores.jsonl`, live weights) from the 13-arena U-Net LoRA runs (`adapt_wm.py`), lines frozen from step-0 training-map predictions through the tuned SD 1 decoder | `figures/fig3_adaptation_curves.pdf` | pending (runs launch Sep 27 evening; arenas 8, 16, 6, 7 first as the fallback set) |
```

   Replacement:

```markdown
| `fig:adapt` (Figure 3) | Adaptation curves, one line per unseen arena coloured by D: `A` against LoRA updates (linear axis including step 0) with the frozen training-map line H and arena-specific halfway targets [A_m(0)+H]/2; first evaluated crossings marked, non-crossers open at the final budget, the planned full fine-tune on arena 7 in black. One panel, as Rohan specified; the `M` curves the cost decision called a second panel are Figure 5 in the appendix. Shares one float row with Figure 2 (one third of the width) | `score_adapt.py score` rows (`scores.jsonl`, live weights) from the 13-arena U-Net LoRA runs (`adapt_wm.py`), H and each A_m(0) frozen from step-0 predictions through the tuned SD 1 decoder | `figures/fig3_adaptation_curves.pdf` | pending (runs launch Sep 27 evening; arenas 8, 16, 6, 7 first as the fallback set) |
```

   Reason: The plotting brief must implement per-arena gap targets; an ordinary logarithmic axis cannot display step zero.

7. **Keep the abstract within the evidence and retain six sentences**

   File: `paper/main.tex`

   Current:

```tex
Game world models are scored on held-out trajectories of their training scenes, and absolute PSNR mostly measures how static the footage is.
On an open 17-arena Doom benchmark scored against copy-last-frame persistence, we train three pretrained diffusion backbones on four arenas and ask what they keep, lose and relearn on the other thirteen.
Decoded through one decoder, the model still beats copy-last on every unseen arena, but its advantage shrinks from \prov{3.2--4.3}~dB at home to \prov{0.7--2.7}~dB and it loses perceptually on \prov{13 of 13}, while its turn response survives.
A frame distance to the training footage separates seen from unseen arenas but does not order the unseen ones.
Rank-16 adapters on the frozen backbone regain half of the home advantage within \tbd{} updates on \tbd{} of 13 arenas at \tbd{}~dB of forgetting.
In closed loop, one-step quality does not predict stability, and the EMA reduces collapses without removing them.
```

   Replacement:

```tex
We evaluate Doom world models on 17 arenas against copy-last persistence and autoencoder reconstruction.
We compare three pretrained backbones on four training arenas; unseen-map results currently cover the U-Net.
Its decoded advantage falls from \prov{3.2--4.3}~dB on training arenas to \prov{0.7--2.7}~dB unseen, with worse LPIPS than persistence on \prov{13 of 13} unseen arenas.
Frame distance separates training from unseen arenas, but has no clear association with advantage within the unseen group.
We plan to measure adapter cost by the updates needed to close half of each arena's zero-shot gap to the training-map advantage.
Good one-step scores can coexist with closed-loop collapse, which exponential moving average weights reduce without eliminating.
```

   Reason: The current abstract generalizes U-Net evidence and asserts an unmeasured adaptation result; hidden hypothesis comments do not qualify rendered claims.

8. **Separate completed contributions from planned adaptation**

   File: `paper/main.tex`

   Current:

```tex
We contribute (1) an open 17-arena Doom benchmark (link withheld for review) scored per arena against copy-last beside the reconstruction ceiling, with a directional check and closed-loop rollouts; (2) the measurement, for three backbones under one recipe, that the decoded advantage shrinks from \prov{3.2--4.3}~dB at home to \prov{0.7--2.7}~dB unseen and flips perceptually (\prov{13 of 13} LPIPS losses) while the ceiling stays flat and the turn response survives (Figure~\ref{fig:strips}); (3) per-arena adaptation curves: rank-16 adapters regain half of the home advantage within \tbd{} updates on \tbd{} of 13 arenas, against a full fine-tune; and (4) two negative results: a per-frame distance does not order arenas of one WAD, and one-step quality does not predict closed-loop stability.
```

   Replacement:

```tex
We contribute a per-arena Doom evaluation against persistence and autoencoder reconstruction, with directional checks and closed-loop rollouts.
The U-Net's decoded advantage shrinks on unseen arenas and its LPIPS margin changes sign; replication with the other backbones is pending.
Frame distance has no clear association with advantage within unseen arenas, and good one-step scores can coexist with collapse.
The adapter cost curves and full fine-tune comparison remain planned experiments.
```

   Reason: The contributions must distinguish existing observations from planned replication and adaptation curves.

9. **Make the adaptation-results slot visibly unresolved**

   File: `paper/main.tex`

   Current:

```tex
\textbf{What it takes.} \tbd{} of 13 arenas reach half of the home advantage within \tbd{} updates and \tbd{} the full advantage by 4k (Figure~\ref{fig:adapt}, Table~\ref{tab:cost}); \tbd{} tie copy-last in LPIPS by 4k.
The turn response holds and the training maps lose \tbd{}~dB, consistent with LoRA forgetting less~\citep{biderman2024lora}; the full fine-tune reaches \tbd{}~dB on arena 7 at \tbd{}~dB of forgetting.
Cost tracks the zero-shot advantage (Spearman \tbd{}) more than $D$ (\tbd{}) or the transition distance (\tbd{}), exploratory at $n=13$.
```

   Replacement:

```tex
\textbf{What it takes.} Adaptation results are pending (Figure~\ref{fig:adapt}, Table~\ref{tab:cost}).
Less forgetting with LoRA is a hypothesis motivated by language-model results~\citep{biderman2024lora}; the full fine-tune comparison is pending.
Associations of cost with zero-shot advantage or either distance will be exploratory.
```

   Reason: Magnitude placeholders do not qualify claims that adaptation succeeds, preserves turning, forgets less, or correlates more with a selected predictor.

10. **Leave adaptation recovery unresolved in the conclusion**

   File: `paper/main.tex`

   Current:

```tex
Off its training maps the model keeps its turn response and part of its advantage over copying, loses its perceptual advantage, and regains half of the home advantage within \tbd{} adapter updates.
```

   Replacement:

```tex
On unseen arenas, the U-Net retains a positive one-tic decoded advantage over copying but loses its LPIPS advantage.
Whether adapters close the gap to the training maps remains an open empirical question.
```

   Reason: The conclusion should state the measured zero-shot finding without inventing recovery.

11. **Correct the pooled-holdout claim about XEWorld**

   File: `paper/main.tex`

   Current:

```tex
The measurement is rare. Held-out scenes appear in world-model papers on one to five domains, pooled~\citep{chen2026xeworld,rigter2024avid,gao2025adaworld,koh2021pathdreamer}, and we found no Doom world model scored on maps it did not train on.
```

   Replacement:

```tex
Held-out evaluation covers different shifts: XEWorld tests unseen robot embodiments~\citep{chen2026xeworld}, AVID and AdaWorld adapt to new environments~\citep{rigter2024avid,gao2025adaworld}, and Pathdreamer tests unseen buildings~\citep{koh2021pathdreamer}.
XEWorld reports its held-out embodiments separately; our unit of comparison is an individual Doom arena.
```

   Reason: XEWorld explicitly reports held-out embodiments separately and never averages them; embodiments, games and buildings are not interchangeable scene counts.

12. **Narrow the persistence precedent and remove the all-scores claim**

   File: `paper/main.tex`

   Current:

```tex
Video prediction reported a copy-last baseline~\citep{mathieu2016deep,lotter2017prednet,villegas2019fidelity} that game world models dropped; we make persistence, defined on every map without training, the zero line: every score is a paired difference against it on the same windows~\citep{bruce2024genie}, beside the reconstruction ceiling~\citep{zheng2023occworld,karypidis2024dinoforesight}.
```

   Replacement:

```tex
Following video prediction~\citep{mathieu2016deep,lotter2017prednet,villegas2019fidelity}, we report model quality beside copy-last persistence and paired differences on the same windows.
We also report autoencoder reconstruction, following occupancy and feature forecasting~\citep{zheng2023occworld,karypidis2024dinoforesight}.
```

   Reason: The draft also reports absolute scores, and Genie is an action-counterfactual comparison rather than a persistence precedent.

13. **Define aggregation and distinguish the Genie analogy**

   File: `paper/main.tex`

   Current:

```tex
The \emph{decoded advantage} $A=\mathrm{PSNR}(D(\hat z),D(z))-\mathrm{PSNR}(D(z_\text{last}),D(z))$ puts the model and copy-last through the same decoder, in the form of Genie's $\Delta$PSNR~\citep{bruce2024genie}.
```

   Replacement:

```tex
The \emph{decoded advantage} is $A=\mathrm{PSNR}(D(\hat z),D(z))-\mathrm{PSNR}(D(z_\text{last}),D(z))$, averaged over paired windows and then equally over maps for group summaries.
It compares both predictions against the decoded target; Genie's $\Delta$PSNR instead contrasts inferred and random actions~\citep{bruce2024genie}.
```

   Reason: Name the actual reference and aggregation without implying identity with the Genie controllability metric.

14. **Separate available full-frame scores from the planned scene crop**

   File: `paper/main.tex`

   Current:

```tex
Pixel quantities use the scene rows, since the HUD carries 39 to 68 percent of the stock decoder's error and persistence copies it exactly. We never rank maps on raw PSNR against persistence, which tracks how static the footage is (Spearman \prov{$-0.77$}).
```

   Replacement:

```tex
The provisional pixel scores use full frames and the stock decoder; the planned rescore will report scene-only scores beside them.
We separate the HUD because persistence nearly reproduces it while the stock decoder reconstructs it poorly.
Across unseen arenas, raw PSNR gain correlates with persistence PSNR (Spearman \prov{$-0.77$}), so we report it with the reference scores rather than interpret its sign alone as map transfer.
```

   Reason: Existing scores are full-frame, and finite persistence HUD PSNR does not justify saying the HUD is copied exactly.

15. **Correct the HUD aggregation and the unconditional decoder-floor claim**

   File: `paper/appendix.tex`

   Current:

```tex
\textbf{What the variation between arenas follows.} Across the 13 unseen arenas, raw gain over persistence tracks persistence PSNR (Spearman \prov{$-0.77$}) and the model's absolute PSNR tracks it at \prov{0.95}, while $A$ does not (\prov{$-0.05$}): the model's pixel error has a floor set by the decoder and by blur, and raw persistence has none, so on static footage raw persistence wins however good the latent prediction is. The gap $G$ tracks persistence too (\prov{$-0.80$}), which is why it is a table column and not a cost axis. The 32 HUD rows, which the stock decoder renders at about 18~dB and persistence copies exactly, carry 39 to 68 percent of the ceiling's squared error on the 17 arenas.
```

   Replacement:

```tex
\textbf{What the variation between arenas follows.} Across unseen arenas, raw PSNR gain correlates with persistence PSNR (Spearman \prov{$-0.77$}); absolute model PSNR correlates at \prov{0.95}.
Decoded advantage $A$ has little association with persistence PSNR (\prov{$-0.05$}), whereas $G$ correlates at \prov{$-0.80$}.
These associations show why raw gain alone is a poor measure of transfer; they do not identify a unique cause.
From the mean full-frame and HUD reconstruction PSNRs, the geometric mean of the per-window HUD error fraction is \prov{39--68}\% across arenas.
This is not the fraction of total squared error aggregated over windows; that quantity requires per-window MSEs.
```

   Reason: Exponentiating a difference of mean PSNRs yields a geometric mean of error ratios, and the files do not prove that persistence wins regardless of latent prediction.

16. **Separate decoded performance from decoder-free latent evidence**

   File: `paper/main.tex`

   Current:

```tex
The latent prediction loses about 2~dB of $A$, stays positive on every arena at one tic, and turns negative at four tics on arenas \prov{6 and 7}.
The rendering does not degrade: the ceiling is \prov{23.91}~dB at home and \prov{23.62}~dB unseen, so the loss sits in the predicted latents; decoding the same predictions with the stock and tuned decoders (\tbd{}) tests a decoder failing on off-distribution latents.
Nor is the perceptual flip the decoder's floor: its own LPIPS (\prov{0.08 to 0.12}) sits below persistence's (\prov{0.15 to 0.23}), and the model still loses against the decoded true frame.
```

   Replacement:

```tex
The U-Net's decoded advantage drops by about \prov{2.09}~dB between the group means and remains positive on every unseen arena at one tic.
It becomes negative at four tics on arenas \prov{6 and 7}.
Reconstruction PSNR changes less: \prov{23.91}~dB on training arenas versus \prov{23.62}~dB unseen.
This is consistent with a prediction deficit, but $A$ still depends on the decoder; the latent-skill and paired-decoder checks remain pending.
The perceptual loss is not forced by reconstruction quality alone: reconstruction LPIPS is below raw persistence LPIPS on every arena.
```

   Reason: A nonlinear decoder does not cancel from A, and the supplied files lack decoded-copy LPIPS needed for the stronger perceptual claim.

17. **State the inconclusive association and failed gate plainly**

   File: `paper/main.tex`

   Current:

```tex
\textbf{The distance orders families, not arenas.} Before scoring any map we froze $D$, the sliced Wasserstein-2 distance between the motion-weighted cloud of a map's per-frame SD~1 latents and that of its nearest training map (Appendix~\ref{app:distance}).
The training maps sit inside the train-versus-train floor (\prov{0.03 to 0.07} against 0.02 to 0.09) and every unseen arena outside it (\prov{0.115 to 0.282}), but within the 13 arenas $D$ orders nothing (Spearman with $A$ \prov{$-0.23$}; 0.56 is needed at $n=13$).
The pre-registered 30-map test, which also held campaign maps of another WAD, gave a partial Spearman of $-0.73$ [$-0.84$, $-0.32$] under a failed validation gate, carried by the family contrasts: qualified evidence of a family effect.
```

   Replacement:

```tex
\textbf{Frame distance and map transfer.} Before scoring any map we froze $D$, the sliced Wasserstein-2 distance between the motion-weighted cloud of a map's per-frame SD~1 latents and that of its nearest training map (Appendix~\ref{app:distance}).
The training-map distances are \prov{0.026--0.069}, versus \prov{0.115--0.282} for unseen arenas.
Within the unseen group, the Spearman correlation with $A$ is \prov{$-0.23$}; we do not establish that distance predicts advantage within this group.
The earlier test including campaign maps failed its pre-declared validation gate (Appendix~\ref{app:distance}).
Its pooled association does not establish a useful ordering of unseen arenas.
```

   Reason: A nonsignificant estimate does not establish a null, and a failed gate should not be relabelled evidence of a family effect.

18. **State coexistence of good one-step scores and collapse**

   File: `paper/main.tex`

   Current:

```tex
\textbf{Closed loop.} One-step quality does not predict stability: SD~3.5 has the best one-tic numbers, yet at 50k and 70k its live weights fall into an absorbing state where one latent channel's mean is captured and frames go blank (Appendix~\ref{app:closedloop}).
```

   Replacement:

```tex
\textbf{Closed loop.} Good one-step scores do not guarantee stable rollouts: SD~3.5's live weights can enter a persistent failure in which a latent channel's mean shifts and frames go blank (Appendix~\ref{app:closedloop}).
```

   Reason: The evidence shows coexistence, not the absence of a predictive association across matched checkpoints.

19. **Allow a sampler-by-weights interaction**

   File: `paper/main.tex`

   Current:

```tex
Both share the sampler, so few-step sampling, which DIAMOND blames for drift~\citep{alonso2024diamond}, cannot explain the difference.
```

   Replacement:

```tex
We hold the sampler fixed when comparing live and EMA weights.
This isolates the weight choice under that sampler, but does not rule out an interaction with sampling steps, which affect drift in DIAMOND~\citep{alonso2024diamond}.
```

   Reason: A shared sampler setting cannot show that its approximation error affects the weight sets identically.

20. **Define latent skill as a log ratio and acknowledge static windows**

   File: `paper/appendix.tex`

   Current:

```tex
The latent skill $S=10\,\overline{\log_{10}(\text{copy-last latent MSE}/\text{model latent MSE})}$ is a geometric mean over windows, so near-static windows cannot dominate it, and no decoder touches it.
```

   Replacement:

```tex
The latent skill $S=10\,\overline{\log_{10}(\text{copy-last latent MSE}/\text{model latent MSE})}$ is the mean log error ratio in dB, computed without a decoder.
It gives equal weight to windows on the log scale, but near-zero copy-last error can still make individual terms extreme; zero-error windows require an explicit handling rule.
```

   Reason: S is the logarithm of a geometric mean, and logarithms do not make nearly zero persistence errors harmless.

21. **Remove the result from the pending rollout caption**

   File: `paper/main.tex`

   Current:

```tex
\caption{Rollouts under held controls (turn left, forward, strafe right) on training arena \tbd{} (left) and unseen arena \tbd{} (right). Rows: truth, U-Net 200k EMA, copy-last; tics 1, 4, 8, 16, 32. On the unseen arena the model follows the control but loses texture first.}
```

   Replacement:

```tex
\caption{Planned rollout comparison under held controls (turn left, forward, strafe right) on training arena \tbd{} (left) and unseen arena \tbd{} (right). Rows: truth, U-Net 200k EMA, copy-last; tics 1, 4, 8, 16, 32. Windows will be selected by a seeded rule before visual inspection.}
```

   Reason: The unproduced figure cannot support the asserted order of texture loss and control preservation.

22. **Separate adapter protocol from cross-domain motivation**

   File: `paper/main.tex`

   Current:

```tex
\textbf{Protocol.} For each unseen arena we adapt the U-Net 200k from its EMA weights with rank-16, $\alpha=16$ LoRA~\citep{hu2022lora} on every attention projection of the frozen backbone, training the control MLP, input projection and noise-bucket embedding in full (4.2M parameters, 0.49 percent), since LoRA alone trails full fine-tuning far behind on DiT-XL/2~\citep{xie2023difffit}; Vista uses this rank on a frozen world model~\citep{gao2024vista}.
```

   Replacement:

```tex
\textbf{Protocol.} We plan rank-16, $\alpha=16$ attention LoRA initialized from the U-Net's 200k EMA~\citep{hu2022lora,gao2024vista}.
We also train the control MLP and its position table, input projection and noise-bucket embedding; other backbone weights stay frozen.
DiffFit motivates testing these additional parameters, based on image-generation experiments~\citep{xie2023difffit}.
```

   Reason: Restore the trained position table, shorten the sentence, and treat DiT image-generation findings as motivation rather than proof for a U-Net world model.

23. **Mark the full fine-tune comparator as pending**

   File: `paper/main.tex`

   Current:

```tex
A full fine-tune of arena 7, the farthest, is the comparator.
```

   Replacement:

```tex
We plan a full fine-tune comparator on arena 7, the most distant unseen arena; this run is pending.
```

   Reason: The comparator is conditional in the run plan and must not read as completed.

24. **Spend related-work space on the evaluation distinction**

   File: `paper/main.tex`

   Current:

```tex
Game world models are scored on held-out trajectories of their training scenes: GameNGen trains for 700k updates at batch 128 on a random 70M-example subset of its agents' play (arXiv v2, \S4.2; v1 reported 900M generated frames)~\citep{valevski2024gamengen}, DIAMOND on one CS:GO map~\citep{alonso2024diamond}, and MultiGen on 100 generated Doom maps without stating whether its test maps are held out~\citep{po2026multigen}.
```

   Replacement:

```tex
GameNGen evaluates held-out Doom trajectories~\citep{valevski2024gamengen}, while DIAMOND's CS:GO study uses one map~\citep{alonso2024diamond}.
MultiGen does not clearly state whether its evaluated Doom maps were held out from training~\citep{po2026multigen}.
```

   Reason: Training-update and corpus-size details do not establish the map-holdout distinction and crowd the body.

25. **Position adaptation precedents without equating embodiments with scenes**

   File: `paper/main.tex`

   Current:

```tex
XEWorld holds out robot embodiments, finds held-out error tracking an appearance distance, and fine-tunes on 25 to 75 episodes, forgetting a seen robot~\citep{chen2026xeworld}; AVID and AdaWorld adapt world models to a held-out game or environment with data and step curves~\citep{rigter2024avid,gao2025adaworld}; Pathdreamer scores unseen buildings against a nearest-neighbour reprojection~\citep{koh2021pathdreamer}.
None scores per scene against persistence; Doom map holdouts have been for agents~\citep{lample2017arnold,wydmuch2018vizdoom}.
```

   Replacement:

```tex
XEWorld reports unseen robot embodiments separately and studies appearance distance, adaptation and forgetting~\citep{chen2026xeworld}.
AVID and AdaWorld study adaptation to new environments~\citep{rigter2024avid,gao2025adaworld}; Pathdreamer evaluates unseen buildings against nearest-neighbour reprojection~\citep{koh2021pathdreamer}.
We instead report persistence-referenced scores for individual Doom arenas; map holdouts are established in Doom agent evaluation~\citep{lample2017arnold,wydmuch2018vizdoom}.
```

   Reason: Acknowledge the closest prior study accurately and describe our unit of analysis without a broad absence claim.

26. **Avoid inferring causal superiority of asymmetric distance**

   File: `paper/main.tex`

   Current:

```tex
Dataset distances predict fine-tuned outcomes~\citep{alvarezmelis2020otdd,nguyen2025sotdd}, directed coverage better than symmetric transport~\citep{mensink2021factors,westny2026latent}, but the cost of crossing to each of many targets of one world model has not been measured, nor, that we found, EMA against live weights in closed loop.
```

   Replacement:

```tex
Dataset distances have been studied as predictors of transfer outcomes~\citep{alvarezmelis2020otdd,nguyen2025sotdd}.
Nearest-source coverage~\citep{mensink2021factors} and latent-distribution KL divergence~\citep{westny2026latent} motivate alternative distance measures; neither validates our proposed transition distance.
```

   Reason: The studies compare measures and tasks; they do not isolate asymmetry as the reason for better prediction.

27. **Keep the transferability precedents distinct**

   File: `paper/appendix.tex`

   Current:

```tex
Directed coverage predicted transfer better than symmetric transport in two studies~\citep{mensink2021factors,westny2026latent}.
```

   Replacement:

```tex
Nearest-source coverage predicts transfer in Mensink et al.~\citep{mensink2021factors}, while Westny et al. use KL divergence between latent dataset distributions~\citep{westny2026latent}.
These results motivate a directed alternative but do not validate it for our transition windows.
```

   Reason: Gaussian latent KL is different from the proposed nearest-training-window coverage distance.

28. **Correct the HUD evidence specification**

   File: `paper/FIGURES.md`

   Current:

```markdown
| Section 2 | HUD carries 39 to 68 % of the ceiling's error | recomputed from `hud_vae_psnr` and `vae_psnr` (17 arenas) | provisional |
```

   Replacement:

```markdown
| Section 2 | geometric mean per-window HUD reconstruction-error fraction, 39 to 68 % | (32/240) × 10^((mean `vae_psnr` − mean `hud_vae_psnr`)/10), per arena; this is not the aggregate MSE share | provisional |
```

   Reason: Preserve the statistic actually reconstructable from aggregate JSON.

29. **Require the cost producer to implement the revised rule**

   File: `paper/FIGURES.md`

   Current:

```markdown
| `tab:cost` (Table 3) | What crossing costs: LoRA over 13 arenas (median), LoRA on arena 7, full fine-tune on arena 7; trained parameters and GPU-hours, arenas past 25/50/100 %, cost to 50 % and 100 %, `A`, `M`, `G` at 4,000, forgetting and directional | `score_adapt.py cost` (right-censored) and the guard rows; parameter counts from `lora.parameter_counts` | table in `main.tex` | trained-parameter row exists; everything else pending (full fine-tune: Monday if a card idles and arena 7's LoRA curve is flat, else October) |
```

   Replacement:

```markdown
| `tab:cost` (Table 3) | What crossing costs: LoRA over 13 arenas (median), LoRA on arena 7, full fine-tune on arena 7; trained parameters and GPU-hours, arenas closing 25/50/100 % of their zero-shot-to-home gap, first evaluated cost to half-gap and full home, `A`, `M`, `G` at 4,000, forgetting and directional | `score_adapt.py cost` after implementing gap closure (right-censored; do not take a median over crossers alone) and the guard rows; parameter counts from `lora.parameter_counts` | table in `main.tex` | trained-parameter row exists; everything else pending (full fine-tune: Monday if a card idles and arena 7's LoRA curve is flat, else October) |
```

   Reason: Changing captions is insufficient if the generator still computes absolute fractions of home advantage.

30. **Carry the cost definition into the appendix inventory**

   File: `paper/FIGURES.md`

   Current:

```markdown
| `tab:perarena-adapt` | Per unseen arena: D, step-0 `A`, cost at 25/50/100 %, `A` and `M` at 4,000, forgetting, directional | `score_adapt.py` rows and cost | D and step-0 `A` provisional; the rest pending |
```

   Replacement:

```markdown
| `tab:perarena-adapt` | Per unseen arena: D, step-0 `A`, first evaluated costs to 25/50/100 % of the zero-shot-to-home gap, `A` and `M` at 4,000, forgetting, directional | `score_adapt.py` rows and cost | D and step-0 `A` provisional; the rest pending |
```

   Reason: Prevent the appendix table from silently reverting to absolute home fractions.

## Problems without a fix

- **Final experimental scope is unresolved.** Section 0 calls the workshop adaptation study a pilot, while the skeleton anticipates all unseen arenas and a conditional full fine-tune comparator. The fresh-set sizes, adaptation split and budget have also changed across documents. A manifest and completed run inventory must settle these; this review does not invent completed runs or choose a new design. The abstract and conclusion above remain deliberately provisional.
- **Decoded perceptual evidence is missing.** The supplied arena h1 JSON files contain model `lpips_dec` but lack `copy_lpips_dec`. The raw-reference LPIPS sign flip is verified; the comparison against the decoded true frame is not independently recoverable. Lower reconstruction LPIPS shows that reconstruction quality alone does not force the loss, but does not isolate the decoder response to predicted latents.
- **Direct latent and paired-decoder checks remain pending.** `latent_mse` alone cannot recover copy-last latent skill. The files contain stock-decoder, full-frame results, not the final tuned-decoder scene-crop comparison. Neither a dB gap nor an autoencoder reconstruction is a strict additive error budget or an optimized upper bound over possible decoder inputs.
- **Uncertainty and adaptation inference need a frozen rule.** Aggregate means and SEMs cannot reconstruct paired episode-bootstrap intervals. Define duplicate/static-window handling, area-under-curve integration and budget, and inference with censored costs before results. Since the new threshold depends on zero-shot advantage, correlation of cost with that advantage has mathematical coupling. First observed crossing need not imply sustained recovery. Arenas already at home must be reported separately from the gap-closure denominator.
- **Distance selection remains exploratory.** Selecting a transition distance using these arenas' zero-shot outcomes cannot independently validate its predictive value. Preserve the failed original gate and distinguish design data from evaluation data. The power estimate in the limitations needs its test, alternative and calculation supplied; concentration of distance range in one arena is not itself a power calculation.
- **Directional and closed-loop claims need separate provenance.** The supplied distance-study h1 files contain neither directional scores nor collapse events. Check pooled versus per-arena directional scores, ground-truth reference and magnitude criterion against their dedicated artifacts. Match live/EMA and one-step/rollout checkpoint identities before making a predictive claim.
- **Mechanism remains a hypothesis.** The conditional-mean explanation for fewer sampling steps and blur is not established by the aggregate PSNR/LPIPS sweep. Do not convert that tradeoff into a mechanistic conclusion without another diagnostic.
- **Submission packaging needs a workshop ruling.** The style anonymizes the author block; keywords and natbib usage are present. The current draft compiles with a four-page body. Appendix permission is unresolved by the supplied template; merely disabling the appendix leaves its body references unresolved. Check the final figure-filled build and permitted appendix packaging before submission.
- **Bibliographic limits.** I verified arXiv identities, titles and author lists for entries cited by the main text and appendix, and checked primary passages for disputed usages. I did not independently certify every venue or uncited legacy entry. The SD-family paper is not release-specific provenance for the exact SD 3.5 checkpoint. XEWorld explicitly reports held-out embodiments separately, contradicting the draft's blanket pooling claim.

## Evidence audit and verification notes

Sources: `results/distance_study/figure_unet_h1/distance_table.md` and `results/distance_study/scores/040-unet-nexttic_snap_0200000_ema_ddim10/*_h1/metrics.json`. Select arena IDs 1 through 17; home is 2, 3, 4, 5; unseen is the complement. Each JSON quantity below is its `mean`. Compute per-map `A = psnr_dec - copy_psnr_dec`, `M = lpips_raw - persist_lpips_raw`, `G = vae_psnr - psnr_raw`, and raw gain `psnr_raw - persist_psnr_raw`. Average maps equally. Spearman uses ranks, not printed rounded scores.

| Check | Recomputed value |
|---|---|
| Home A, maps 2 / 3 / 4 / 5 | 3.2279745024 / 3.2050495632 / 4.2636597157 / 3.6969343536 dB |
| Home mean H; old half-home line | 3.5984045337; 1.7992022668 dB |
| Unseen A range; mean; home-minus-unseen mean | 0.7233267874 to 2.6810565963; 1.5085203791; 2.0898841546 dB |
| Arenas above H/2 at step zero | 9, 11, 13, 15, 16 |
| Borderline old-floor checks | arena 11: 1.8006558083; arena 13: 1.8049685508 dB; both exceed the unrounded line |
| New half-gap targets, arenas 6 / 7 / 15 | 2.1608656605 / 2.2379079403 / 3.1397305650 dB, each (A_m(0)+H)/2 |
| Reconstruction means, home / unseen | 23.9070054516 / 23.6155718009 dB |
| M means, home / unseen | -0.0268386983 / +0.1054391207; negative on 4/4 home and positive on 13/13 unseen arenas |
| Reconstruction LPIPS below raw persistence LPIPS | 17/17 arenas |
| Unseen Spearman: D versus A; D versus raw gain | -0.2252747253; -0.0164835165 |
| Unseen Spearman: persistence PSNR versus raw gain / model PSNR / A / G | -0.7692307692 / +0.9450549451 / -0.0494505495 / -0.7967032967 |
| Home D range; unseen D range | 0.026 to 0.069; 0.115 to 0.282 |
| Geometric mean HUD reconstruction-error fraction, arena endpoints | 39.2385409731% (arena 13) to 67.7788852758% (arena 4) |

For the HUD row, compute `(32/240) * 10**((vae_psnr.mean - hud_vae_psnr.mean)/10)` per map. This is the geometric mean of per-window HUD-to-full-frame squared-error fractions when the means use matched windows, not a ratio of summed squared errors. The four-tic sign statement in proposal 16 is available in adjacent h4 JSONs, checked separately rather than inferred from h1.

Build check: every current-text block matched its source exactly once. Applying all proposed TeX replacements in a temporary copy compiled successfully through pdflatex, BibTeX and two further pdflatex passes. The body-end label remains on page 4, with no undefined citations/references or overfull boxes. The repository manuscript and template files were not edited. This checks the current placeholder layout, not the eventual finished figures.

Primary-source passages checked: [XEWorld](https://arxiv.org/html/2608.05799), [Genie](https://arxiv.org/html/2402.15391), [DIAMOND](https://arxiv.org/html/2405.12399), [AVID](https://arxiv.org/html/2410.12822), [AdaWorld](https://arxiv.org/html/2503.18938), [OccWorld](https://arxiv.org/html/2311.16038), [Vista](https://arxiv.org/html/2405.17398), [DiffFit](https://arxiv.org/html/2304.06648), [Biderman](https://arxiv.org/html/2405.09673), and [Westny](https://arxiv.org/html/2606.30777). All cited arXiv abstract metadata resolved. XEWorld says its held-out embodiments are reported separately and never averaged; Genie contrasts inferred-action and random-action generations; OccWorld includes reconstruction and Copy&Paste; DiffFit concerns image generation; Biderman concerns language-model adaptation. These scope boundaries govern the proposed wording.

Additional checks: [PredNet PDF](https://arxiv.org/pdf/1605.08104) confirms the unseen-dataset test and Copy Last Frame baseline; its HTML endpoint was unavailable, but the PDF was readable. [Hu et al.](https://arxiv.org/html/2106.09685) confirms the LoRA method; [Taylor and Stone](https://www.jmlr.org/papers/v10/taylor09a.html) confirms the survey identity. Venue assignments beyond these source checks remain unaudited rather than silently endorsed.


## On the other list

Round 2: I read all 27 Fable proposals (including every multi-part replacement) and all seven unresolved problems. The decisions below supersede conflicting round-1 recommendations without changing that earlier text. Each numbered line corresponds to Fable's proposal number, not an Astra proposal number. For whole-paragraph alternatives, apply the winning paragraph once; do not also apply a conflicting substring edit. Proposal 32 below controls appendix references in every accepted wording.

1. **AGREE WITH CHANGE** — Astra 7 wins for the whole abstract: it limits the unseen results to the U-Net, keeps the metric explicit and does not assert the pending adaptation result. Exact replacement of the abstract prose: ``We evaluate Doom world models on 17 arenas against copy-last persistence and autoencoder reconstruction. We compare three pretrained backbones on four training arenas; unseen-map results currently cover the U-Net. Its decoded advantage falls from \prov{3.2--4.3}~dB on training arenas to \prov{0.7--2.7}~dB unseen, with worse LPIPS than persistence on \prov{13 of 13} unseen arenas. Frame distance separates training from unseen arenas, but has no clear association with advantage within the unseen group. We plan to measure adapter cost by the updates needed to close half of each arena's zero-shot gap to the training-map advantage. Good one-step scores can coexist with closed-loop collapse, which exponential moving average weights reduce without eliminating.``.

2. **AGREE WITH CHANGE** — Astra 8 wins for the whole contribution paragraph, avoiding repeated edits to the same sentence. Exact replacement: ``We contribute a per-arena Doom evaluation against persistence and autoencoder reconstruction, with directional checks and closed-loop rollouts. The U-Net's decoded advantage shrinks on unseen arenas and its LPIPS margin changes sign; replication with the other backbones is pending. Frame distance has no clear association with advantage within unseen arenas, and good one-step scores can coexist with collapse. The adapter cost curves and full fine-tune comparison remain planned experiments.``.

3. **AGREE WITH CHANGE** — Astra 16 wins on the decoder claim: similar reconstruction scores do not establish an invariant rendering floor or identify the whole loss. Exact replacement of Fable's current sentence: ``Reconstruction PSNR changes less: \prov{23.91}~dB on training arenas versus \prov{23.62}~dB unseen. This is consistent with a prediction deficit, but $A$ still depends on the decoder; the latent-skill and paired-decoder checks remain pending.``.

4. **AGREE** — Fable's wording wins over the first two sentences of Astra 16; it names decoded A and exposes both group means. Exact replacement: ``$A$ drops by about 2~dB (\prov{3.60} to \prov{1.51}), stays positive on every arena at one tic, and turns negative at four tics on arenas \prov{6 and 7}.``.

5. **DISAGREE** — The stated 13/13 comparison mixes model LPIPS against a decoded target with persistence LPIPS against a raw target; `copy_lpips_dec` is missing. Astra 16's narrower wording wins: ``The perceptual loss is not forced by reconstruction quality alone: reconstruction LPIPS is below raw persistence LPIPS on every arena.``.

6. **AGREE WITH CHANGE** — Astra 24 wins: MultiGen's training-map count should not be described as its evaluation-map count. Exact replacement: ``GameNGen evaluates held-out Doom trajectories~\citep{valevski2024gamengen}, while DIAMOND's CS:GO study uses one map~\citep{alonso2024diamond}. MultiGen does not clearly state whether its evaluated Doom maps were held out from training~\citep{po2026multigen}.``.

7. **AGREE WITH CHANGE** — Use this merged wording instead of either original: Fable's two milestones and printed per-arena thresholds win; Astra's fixed decoder/crop/windows and nonpositive-gap rule stay. I withdraw the quarter-gap milestone from Astra 1–4 and 29–30. Exact replacement of both Cost sentences: ``\textbf{Cost.} We freeze the training-map mean $A_\text{home}$ and each arena's zero-shot $A_0$ before reading adapted scores, using the same decoder, crop and windows throughout. For $A_0<A_\text{home}$, cost is the first grid budget reaching $\tfrac12(A_0+A_\text{home})$; the second crossing is $A_\text{home}$. We also report the value at 4k and area under the curve~\citep{taylor2009transfer}; non-crossers are right-censored at 4k. An arena already at or above home has zero home-crossing cost and no half-gap cost. Provisionally, $A_\text{home}=\prov{3.60}$~dB and arenas \prov{9, 11, 13, 15 and 16} already exceed half of it.``.

8. **AGREE WITH CHANGE** — Use the merged caption, retaining Astra's censoring and forgetting conventions: ``\caption{Adaptation cost (U-Net 200k EMA initialization, 8 episodes, live weights). Half gap: $\tfrac12(A_0+A_\text{home})$; home: $A_\text{home}$. Costs are first grid crossings; ${>}$4k means censored, including the median if fewer than half cross. Forgetting: change in training-map $A$ (negative means loss). Per arena: Appendix~\ref{app:perarena-adapt}.}`` Fable's two milestone rows win verbatim over Astra 3: ``Arenas past half gap / home line by 4k & \tbd{} / \tbd{} of 13 & -- & -- \\ Cost to half gap / home line (updates) & \tbd{} / \tbd{} & \tbd{} / \tbd{} & \tbd{} / \tbd{} \\``.

9. **AGREE** — Fable's explicit half-gap column wins over Astra 4; every displayed threshold matches recomputation using unrounded home mean 3.5984045336954296 dB, not the printed 3.60. Exact winning caption: ``\caption{Per unseen arena, sorted by $D$: step-0 decoded advantage $A_0$, the half-gap line $\tfrac12(A_0+A_\text{home})$ with $A_\text{home}=\prov{3.60}$~dB, the first grid budget at which $A$ crosses the half-gap line and the home line (${>}$4k: censored), $A$ and $M$ at 4,000 updates, forgetting (change in the training maps' $A$) and the directional score at 4,000.}`` Exact winning header and rows: ``Arena & $D$ & $A_0$ & Half-gap line & Half gap & Home line & $A$ at 4k & $M$ at 4k & Forgetting & Directional \\ \midrule 8 & \prov{0.115} & \prov{+1.29} & \prov{2.44} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\ 15 & \prov{0.127} & \prov{+2.68} & \prov{3.14} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\ 12 & \prov{0.128} & \prov{+1.07} & \prov{2.33} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\ 17 & \prov{0.130} & \prov{+1.30} & \prov{2.45} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\ 14 & \prov{0.146} & \prov{+1.33} & \prov{2.46} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\ 11 & \prov{0.150} & \prov{+1.80} & \prov{2.70} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\ 13 & \prov{0.156} & \prov{+1.80} & \prov{2.70} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\ 10 & \prov{0.162} & \prov{+1.67} & \prov{2.63} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\ 16 & \prov{0.164} & \prov{+1.92} & \prov{2.76} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\ 1 & \prov{0.179} & \prov{+1.09} & \prov{2.35} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\ 9 & \prov{0.180} & \prov{+2.05} & \prov{2.82} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\ 6 & \prov{0.188} & \prov{+0.72} & \prov{2.16} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\ 7 & \prov{0.282} & \prov{+0.88} & \prov{2.24} & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\``.

10. **AGREE WITH CHANGE** — Use Astra 5's explicit marker semantics with Fable's notation; the comparator stays planned. Exact Figure 3 replacement: ``\caption{$A$ against adapter updates; colour: frame distance $D$. Dashed: $A_\text{home}$; ticks: each arena's $\tfrac12(A_0+A_\text{home})$. Markers: first grid crossings; open endpoints: censored. Black: planned full fine-tune.}`` Fable's Figure 2 deletion wins verbatim: ``(a) $A$; dashed: the training maps' mean. (b) $M$.``.

11. **AGREE WITH CHANGE** — Astra 7's planned-study sentence wins over a claim of measured recovery. Exact replacement: ``We plan to measure adapter cost by the updates needed to close half of each arena's zero-shot gap to the training-map advantage.`` This is already included in item 1; apply it once.

12. **AGREE WITH CHANGE** — Astra 8 wins for the whole contribution paragraph, as in item 2; a placeholder count does not make an asserted adaptation result hypothetical. Exact winning replacement: ``We contribute a per-arena Doom evaluation against persistence and autoencoder reconstruction, with directional checks and closed-loop rollouts. The U-Net's decoded advantage shrinks on unseen arenas and its LPIPS margin changes sign; replication with the other backbones is pending. Frame distance has no clear association with advantage within unseen arenas, and good one-step scores can coexist with collapse. The adapter cost curves and full fine-tune comparison remain planned experiments.`` Apply the paragraph once.

13. **AGREE WITH CHANGE** — Astra 9 wins for the whole What-it-takes paragraph, including the later unmeasured preservation and correlation claims. Exact replacement: ``\textbf{What it takes.} Adaptation results are pending (Figure~\ref{fig:adapt}, Table~\ref{tab:cost}). Less forgetting with LoRA is a hypothesis motivated by language-model results~\citep{biderman2024lora}; the full fine-tune comparison is pending. Associations of cost with zero-shot advantage or either distance will be exploratory.``.

14. **AGREE WITH CHANGE** — Astra 10 wins for the whole concluding sentence because recovery remains unmeasured: ``On unseen arenas, the U-Net retains a positive one-tic decoded advantage over copying but loses its LPIPS advantage. Whether adapters close the gap to the training maps remains an open empirical question.`` Astra 6 wins for the entire FIGURES.md `fig:adapt` row, with the agreed notation and a linear axis retaining zero: ``| `fig:adapt` (Figure 3) | Adaptation curves, one line per unseen arena coloured by D: `A` against LoRA updates (linear axis including step 0) with the frozen training-map line A_home and arena-specific halfway targets (A_0+A_home)/2; first evaluated crossings marked, non-crossers open at the final budget, the planned full fine-tune on arena 7 in black. One panel, as Rohan specified; the `M` curves the cost decision called a second panel are Figure 5 in the appendix. Shares one float row with Figure 2 (one third of the width) | `score_adapt.py score` rows (`scores.jsonl`, live weights) from the 13-arena U-Net LoRA runs (`adapt_wm.py`), A_home and each A_0 frozen from step-0 predictions through the tuned SD 1 decoder | `figures/fig3_adaptation_curves.pdf` | pending (runs launch Sep 27 evening; arenas 8, 16, 6, 7 first as the fallback set) |``.

15. **AGREE WITH CHANGE** — Use this shorter first-person wording; neither the original nor Fable's long revision wins: ``We need a world model to generalize to unseen levels before using it to plan or train a policy there.`` The original has a main verb; clarity, not a missing verb, is the issue.

16. **AGREE WITH CHANGE** — Astra 12 wins because merely splitting the sentence preserves the false all-scores claim and the misleading Genie attribution. Exact replacement: ``Following video prediction~\citep{mathieu2016deep,lotter2017prednet,villegas2019fidelity}, we report model quality beside copy-last persistence and paired differences on the same windows. We also report autoencoder reconstruction, following occupancy and feature forecasting~\citep{zheng2023occworld,karypidis2024dinoforesight}.``.

17. **AGREE** — The sentence split is accurate and requires no extra claim; accept Fable's replacement verbatim.

18. **DISAGREE** — Agreement of image motion with recorded controls is an empirical sanity check, not a mathematical bound on a different, swapped-control score; the 0.91 also needs matched-run provenance.

19. **AGREE WITH CHANGE** — Astra 17's uncertainty language wins over a categorical null and an unspecified significance cutoff. Exact replacement of Fable's quoted clause: ``but within the 13 unseen arenas we do not establish an association between $D$ and $A$ (Spearman \prov{$-0.23$}).`` If Astra 17 is applied as a block, its existing equivalent sentence takes precedence; do not duplicate it.

20. **AGREE WITH CHANGE** — Astra 22 wins: preserve the trained position table and treat DiffFit as image-generation motivation. Exact replacement: ``\textbf{Protocol.} We plan rank-16, $\alpha=16$ attention LoRA initialized from the U-Net's 200k EMA~\citep{hu2022lora,gao2024vista}. We also train the control MLP and its position table, input projection and noise-bucket embedding; other backbone weights stay frozen. DiffFit motivates testing these additional parameters, based on image-generation experiments~\citep{xie2023difffit}.``.

21. **AGREE** — The current adapt_split.py defaults confirm the step curve uses 8 of the 16 adaptation episodes; accept Fable's replacement verbatim.

22. **AGREE WITH CHANGE** — Astra 26 wins because splitting the sentence does not fix the conflation of nearest-source coverage with Gaussian latent KL. Exact replacement: ``Dataset distances have been studied as predictors of transfer outcomes~\citep{alvarezmelis2020otdd,nguyen2025sotdd}. Nearest-source coverage~\citep{mensink2021factors} and latent-distribution KL divergence~\citep{westny2026latent} motivate alternative distance measures; neither validates our proposed transition distance.``.

23. **DISAGREE** — The proposed 'so' falsely derives a power estimate from the concentration of distance range; power requires a specified test, alternative and sampling model.

24. **AGREE** — The sign conventions and colour-independent provenance are correct; accept Fable's replacement verbatim.

25. **AGREE WITH CHANGE** — Keep provenance visible when colour is disabled. Exact Table 1 replacement: ``Provisional rows: PixArt and SD~3.5\ifdraftmarks{} (blue)\fi; $^\dagger$150k.}`` Exact appendix per-arena-caption replacement: ``Pre-fresh-set episodes\ifdraftmarks{} (blue)\fi. $^\ast$pooled over the four training maps.}``.

26. **AGREE WITH CHANGE** — Shorten without claiming that sampling identifies a mechanism or pointing to an omitted submission appendix. Exact replacement: ``Blur under uncertainty is one possible explanation for the PSNR--LPIPS tradeoff; the mechanism remains untested.``.

27. **AGREE WITH CHANGE** — Keep the short explanation but identify the decoder tune as planned until its checkpoint and scores exist. Exact replacement: ``Rollouts stay in latent space, so the decoder only renders. Following GameNGen, we plan to tune it with MSE on training-map frames, then freeze it.``.

**HUD wording resolution (Fable's opening audit, not a numbered proposal).** Astra 14–15 and 28 win: the arithmetic matches, but the reported statistic is a geometric mean of per-window error fractions, not an aggregate MSE share. Exact appendix wording: ``From the mean full-frame and HUD reconstruction PSNRs, the geometric mean of the per-window HUD error fraction is \prov{39--68}\% across arenas. This is not the fraction of total squared error aggregated over windows; that quantity requires per-window MSEs.`` The main-text replacement remains Astra 14 exactly: ``The provisional pixel scores use full frames and the stock decoder; the planned rescore will report scene-only scores beside them. We separate the HUD because persistence nearly reproduces it while the stock decoder reconstructs it poorly. Across unseen arenas, raw PSNR gain correlates with persistence PSNR (Spearman \prov{$-0.77$}), so we report it with the reference scores rather than interpret its sign alone as map transfer.``.

**Cost documentation resolution.** Fable's two milestones replace the quarter-gap milestone everywhere: Astra 29's FIGURES.md description should read `arenas closing half of their zero-shot-to-home gap or reaching home, first evaluated costs to half gap and home`; Astra 30's description should read `step-0 A, the per-arena half-gap threshold, and first evaluated costs to half gap and home`. The producer must use unrounded step-zero means. The numerical rows accepted in Fable 9 remain provisional until the fresh-set rescore freezes the actual thresholds. Report arenas already at or above home separately from the half-gap denominator.

**Fable's seven unresolved problems.** P1: venue verification remains incomplete, and metadata verification does not establish every claim's usage. P2: resolved in the winning U-Net-specific abstract and contribution text. P3: the full fine-tune stays explicitly pending, with no fabricated score. P4: final figure readability and page fit still need checking after the agreed set is applied. P5: distinguish the earlier pre-registration set from the arena-only result tables, while retaining the failed-gate disclosure. P6: the directional scores still need matched-artifact provenance. P7: Astra 21 removes the unobserved result from the pending Figure 1 caption.

Counts over the 27 numbered Fable proposals: **5 AGREE, 19 AGREE WITH CHANGE, 3 DISAGREE**. Disagreements: **5** (mixed LPIPS targets), **18** (empirical estimator check is not a bound), **23** (unsupported causal derivation of power).

## Additional required proposals

31. **Anonymise identifying source annotations and retain the style's hidden author block.**

Files: `paper/main.tex`, `paper/appendix.tex`.

I searched both complete files with `rg -n -i` for personal names, account/URL patterns, Hugging Face, W&B, GitHub, university/lab names, host names, source commit identifiers, and self-attribution. There are **no Hugging Face account strings, W&B URLs, GitHub usernames/URLs, or author-affiliation names in the rendered body or appendix**. Do not invent replacements for absent strings. The following are all identifying source annotations found, including host paths and source revisions; preserve each exact original in its requested `% camera-ready:` comment. These comments do not render in the PDF.

File: `paper/main.tex`

Current:

```tex
% Template: corl_2026.sty and corlabbrvnat.bst exactly as Rohan's Overleaf download ships them (f0500f9);
```

Replacement:

```tex
% Template: corl_2026.sty and corlabbrvnat.bst exactly as [anonymous authors’] Overleaf download ships them ([anonymous revision]);
% camera-ready: % Template: corl_2026.sty and corlabbrvnat.bst exactly as Rohan's Overleaf download ships them (f0500f9);
```

File: `paper/main.tex`

Current:

```tex
% body-plus-references submission build if Rohan rules that the appendix may not ride along.
```

Replacement:

```tex
% Default submission: body plus references; appendix reserved for camera-ready or an anonymous supplement by [anonymous authors].
% camera-ready: % body-plus-references submission build if Rohan rules that the appendix may not ride along.
```

File: `paper/main.tex`

Current:

```tex
% depends: PixArt 200k final read (Sep 26 night) and SD 3.5 200k (about Sep 28 00:30) replace the provisional rows; tuned-decoder and scene-only columns from scripts/spiderman/rescore_tuned_decoder.sh
```

Replacement:

```tex
% depends: PixArt 200k final read (Sep 26 night) and SD 3.5 200k (about Sep 28 00:30) replace the provisional rows; tuned-decoder and scene-only columns from scripts/[anonymous-host]/rescore_tuned_decoder.sh
% camera-ready: % depends: PixArt 200k final read (Sep 26 night) and SD 3.5 200k (about Sep 28 00:30) replace the provisional rows; tuned-decoder and scene-only columns from scripts/spiderman/rescore_tuned_decoder.sh
```

File: `paper/main.tex`

Current:

```tex
% Figures 2 and 3 share one float row to fit the page budget; Figure 3 is the single panel Rohan specified (advantage against updates), and the perceptual-margin curves the cost decision called a second panel are Figure 5 in Appendix F.
```

Replacement:

```tex
% Figures 2 and 3 share one float row to fit the page budget; Figure 3 is the single panel [anonymous authors] specified (advantage against updates), and the perceptual-margin curves the cost decision called a second panel are Figure 5 in Appendix F.
% camera-ready: % Figures 2 and 3 share one float row to fit the page budget; Figure 3 is the single panel Rohan specified (advantage against updates), and the perceptual-margin curves the cost decision called a second panel are Figure 5 in Appendix F.
```

File: `paper/main.tex`

Current:

```tex
% adapter facts: lora.py and adapt_wm.py at eaf565a; U-Net trained total 4.21M = 0.49 percent (RESEARCH_CONTEXT 2026-09-26 20:10); the design also trains the control MLP's position table
```

Replacement:

```tex
% adapter facts: lora.py and adapt_wm.py at [anonymous revision]; U-Net trained total 4.21M = 0.49 percent (RESEARCH_CONTEXT 2026-09-26 20:10); the design also trains the control MLP's position table
% camera-ready: % adapter facts: lora.py and adapt_wm.py at eaf565a; U-Net trained total 4.21M = 0.49 percent (RESEARCH_CONTEXT 2026-09-26 20:10); the design also trains the control MLP's position table
```

File: `paper/appendix.tex`

Current:

```tex
% appendix; the OpenReview form takes one PDF. Whether this rides along in the submission PDF is Rohan's call.
```

Replacement:

```tex
% appendix; retained for camera-ready or an anonymous supplement prepared by [anonymous authors].
% camera-ready: % appendix; the OpenReview form takes one PDF. Whether this rides along in the submission PDF is Rohan's call.
```

File: `paper/appendix.tex`

Current:

```tex
% depends: the Spiderman rerun of the structural controls with the per-window files (the local amended CIs resampled without them); the within-unseen partial is recomputed on the fresh set
```

Replacement:

```tex
% depends: the [anonymous host] rerun of the structural controls with the per-window files (the local amended CIs resampled without them); the within-unseen partial is recomputed on the fresh set
% camera-ready: % depends: the Spiderman rerun of the structural controls with the per-window files (the local amended CIs resampled without them); the within-unseen partial is recomputed on the fresh set
```

File: `paper/appendix.tex`

Current:

```tex
% exists: paper/figures/sd35_70k_live_vs_ema_rollout_strip.jpg (tools/collapse_strip.py at 42a4688)
```

Replacement:

```tex
% exists: paper/figures/sd35_70k_live_vs_ema_rollout_strip.jpg (tools/collapse_strip.py at [anonymous revision])
% camera-ready: % exists: paper/figures/sd35_70k_live_vs_ema_rollout_strip.jpg (tools/collapse_strip.py at 42a4688)
```

The author block is already excluded from the rendered submission by `\usepackage{corl_2026}` with neither `final` nor `preprint`. Its exact identifying lines are listed here as the explicit exception requested by the user, with no replacement proposed:

```tex
  Rohan Nagabhirava \quad Keerthana Chirumamilla \quad Changliu Liu\\
  Carnegie Mellon University\\
  Pittsburgh, PA, United States
```

The existing `(link withheld for review)` is already anonymous. First-person `we`/`our` describes the present work and does not identify its authors. Arnold, ViZDoom, Freedoom, Stiegler and Taketani identify cited third-party resources, so those names remain. The generic Hugging Face mention is not an account identifier. Use the anonymous PDF for a supplement: the requested camera-ready comments deliberately retain identities in the private TeX source.

Reason: remove author/host attribution from active annotations without erasing evidence provenance or misidentifying third-party citations as author disclosures; preserve the submission style's existing PDF anonymisation.

32. **Default to body plus references, and suppress references to the omitted appendix.**

File: `paper/main.tex`.

Current:

```tex
\newif\ifwithappendix \withappendixtrue
```

Replacement:

```tex
\newif\ifwithappendix \withappendixfalse
% camera-ready: \withappendixtrue includes appendix.tex after the bibliography.
% Submission PDF: body and references only. Keep appendix.tex for camera-ready
% or a separately built anonymous supplement; do not enable final/preprint for that supplement.
```

Keep the existing `\ifwithappendix ... \input{appendix} ... \fi` block after the bibliography. The user has now decided the submission packaging; this supersedes the earlier unresolved-permission note. Turning the flag off alone would create undefined references. Apply the following exact substring replacements to the present source; the protocol-parenthesis pattern occurs twice and the others once. Apply the same guards to surviving equivalent references in the agreed review edits. A false branch removes only the appendix pointer, not its surrounding scientific statement.

Current:

```tex
 (Appendix~\ref{app:protocol})
```

Replacement:

```tex
\ifwithappendix{} (Appendix~\ref{app:protocol})\fi
```

Current:

```tex
 (Appendix~\ref{app:distance})
```

Replacement:

```tex
\ifwithappendix{} (Appendix~\ref{app:distance})\fi
```

Current:

```tex
 (Appendix~\ref{app:closedloop})
```

Replacement:

```tex
\ifwithappendix{} (Appendix~\ref{app:closedloop})\fi
```

Current:

```tex
; a decoder-free latent skill is in Appendix~\ref{app:skill}
```

Replacement:

```tex
\ifwithappendix; a decoder-free latent skill is in Appendix~\ref{app:skill}\fi
```

Current:

```tex
 ($M$ curves: Appendix~\ref{app:perarena-adapt})
```

Replacement:

```tex
\ifwithappendix{} ($M$ curves: Appendix~\ref{app:perarena-adapt})\fi
```

Current:

```tex
 Per arena: Appendix~\ref{app:perarena-adapt}.
```

Replacement:

```tex
\ifwithappendix{} Per arena: Appendix~\ref{app:perarena-adapt}.\fi
```

The parenthesized distance guard also covers the additional failed-gate pointer introduced by Astra 17; the caption guard covers the agreed cost caption in Fable item 8 above. Fable item 26's changed wording has no appendix pointer. No synthetic appendix labels or frozen page numbers are introduced. Keep the appendix file and its labels intact for the builds that include it. A separate supplement needs its own anonymous PDF build if one is actually submitted; this proposal does not create or promise such a submission.

Reason: enforce the requested body-plus-references submission while retaining a usable appendix source and eliminating dangling cross-references.

Validation of proposals 31–32: all current-text matches were checked; a temporary submission build contains body plus references, keeps `body-end` on page 4, and has no undefined references, overfull boxes, or rendered author names. Re-enabling the guarded appendix in a fresh temporary build also resolves all references. This validates the new packaging edits against the present manuscript, not the still-unapplied merged set of both reviews. The final combined wording and real figures require a fresh page-budget check. No manuscript, template, bibliography or other review file was changed.
