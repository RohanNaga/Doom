# Overleaf diff, Sunday Sep 27 (adaptation, distance, decoder)

Revision 3 (10:10, all three statistics views in, `per-arena-statistics-decision-2026-09-27.md`): "tie" dropped from the LPIPS claim (no equivalence margin), the crossing count carries its arena-bootstrap interval, the ladder gap reads about 0.25 dB; Opus's statistics view corrected the LPIPS count (11 wins, 2 ties), the ladder gap (0.26 dB) and the $S_0$ claim (predicts the endpoint, not the gain). Revision 2 (09:50) after Astra's truth check (`2026-09-27-sunday-adaptation.review_astra.md`, thread 01a0e2f7-c7dc-7e61-8f28-2ace91b00951): every $G$ number moved to the raw-reference definition; PixArt rank agreement 0.96 (tuned); the 2k-to-4k rise given as a range; seed, ladder and recipe claims scoped to the stock-decoder checks; the failed pre-registered gate kept; Astra's five replacement sentences taken with light edits. Still to reconcile in `main.tex` outside this diff: the older "3.60 to 1.51" and MSE-only decoder mentions in section 3 and Table 3's "Full, arena 7" header.

Apply to `main.tex` as it stands in Overleaf. Each item gives the exact current text and the exact replacement. Numbers: tuned decoder (MSE + 0.1 LPIPS) unless marked stock; scene crop; live weights; 256 fresh held-out windows per arena; home = the training maps' 512 validation windows. Sources: `paper/tables/tuned/adapt_cost.tex`, `paper/tables/tuned/adapt_summary.json`, `results/fresh_rescore/*_tuned/map*/metrics.json` (script in the RESEARCH_CONTEXT 2026-09-27 08:05 entry), `.claude/analyses/distance-usefulness-decision-2026-09-27.md`. Final numbers (the arena 6 tuned rescore landed at 08:12: it crosses at 1,000).

Metric conventions the diff assumes (say if you want the other): $A$ = scene PSNR of the rendered prediction minus scene PSNR of the rendered copy-last frame, both against the rendered true frame (unchanged). $M$ = scene LPIPS of the rendered prediction against the **raw** true frame minus scene LPIPS of raw persistence, so both sides are measured against the same raw frame; the decoded-reference version (kinder to the model by about 0.03) goes to the appendix. $G$ = scene PSNR of the decoder's own reconstruction of the true latent minus scene PSNR of the rendered prediction, both against the raw true frame, same decoder (the `score_adapt.py` definition; an earlier draft mixed references and understated $G$ by about 2 dB). $A_\text{home}$ = the source model's own advantage on held-out episodes of its four training maps, the in-distribution reference (5.06 dB, 95 percent episode-bootstrap interval 4.75 to 5.39); figures and tables label it "training maps (in-distribution)" with that band, never "home", and the text says it is a reference, not the arena's own ceiling.

## 1. Abstract, sentence 3 (line 67)

Current:
> Decoded through one decoder, the U-Net still beats copy-last on every unseen arena, but its advantage shrinks from \prov{3.2--4.3}~dB at home to \prov{0.7--2.7}~dB and its LPIPS is worse than persistence's on \prov{13 of 13} arenas, while its turn response survives.

Replace with:
> Decoded through one decoder, the U-Net still beats copy-last on every unseen arena, but its advantage shrinks from 5.1~dB at home to 1.1--2.9~dB (median 2.0) and its LPIPS is worse than persistence's on 13 of 13 arenas, while its turn response survives.

## 2. Abstract, sentence 5 (line 69)

Current:
> We measure what closing half of each arena's gap to the home advantage costs rank-16 adapters on the frozen backbone: \tbd{} updates on \tbd{} of 13 arenas, at \tbd{}~dB of forgetting.

Replace with:
> Rank-16 adapters on the frozen backbone, trained on eight episodes of an arena, close half of its gap to the home advantage within 4k updates on 9 of 13 arenas (five within 500), recover a median 1.8~dB, and turn the perceptual loss into a win on 11 of 13 arenas (the other two within 0.01); one episode already yields most of the gain, and no per-frame or transition-level distance predicts where an arena ends up, while the frozen model's own zero-shot latent skill does.

(If this runs the abstract past six sentences, drop "one episode already yields most of the gain, and".)

## 3. Contributions (3) and (4) (line 96)

Current:
> (3) per-arena adaptation curves, with the cost for rank-16 adapters to close half of each arena's gap to the home advantage (\tbd{}) and a full fine-tune comparator; and (4) two negative results: a per-frame distance has no clear association with the advantage among arenas of one WAD, and one-step quality does not guarantee closed-loop stability.

Replace with:
> (3) per-arena adaptation curves for rank-16 adapters, with the budget to close half of each arena's gap to the home advantage (median 1k updates; 9 of 13 by 4k), a data ladder showing one episode recovers most of the gain, and a full fine-tune comparator (\tbd{}); and (4) two negative results: neither a per-frame distance nor two transition-level distances order the arenas by zero-shot advantage or adaptation budget, whereas the frozen model's zero-shot latent skill does (Spearman 0.90 with the advantage at 4k, exploratory at $n=13$), and one-step quality does not guarantee closed-loop stability.

## 4. Section 3, the decoder lines (lines 179 to 180)

Current:
> The rendering of true latents does not degrade: the ceiling is \prov{23.91}~dB at home and \prov{23.62}~dB unseen, so the loss is not the decoder's floor; decoding the same predictions with the stock and tuned decoders (\tbd{}) tests whether it also renders off-distribution latents worse.
> Nor is the perceptual flip the decoder's floor: its own LPIPS (\prov{0.08 to 0.12}) sits below persistence's (\prov{0.15 to 0.23}), and the model, scored against the decoded true frame, still has higher LPIPS than raw persistence (\prov{13 of 13}).

Replace with:
> The tuned decoder's scene reconstruction PSNR is 28.6~dB at home and 25.2 to 29.7~dB on the unseen arenas (median 27.3), while the gap $G$ between that reconstruction and the rendered prediction is 3.4~dB at home and 3.7 to 6.9~dB unseen (median 5.0). Decoding the same predictions with the stock and the tuned decoder raises $A$ by 0.2~dB zero-shot ($-0.01$ to $+0.51$) and by 0.7~dB after adaptation (0.5 to 1.1), so the decoder's share of the deficit differs between frozen and adapted latents; every column is reported under both decoders (\appref{app:perarena}).
> Nor is the perceptual flip the decoder's floor: the decoder's own reconstruction LPIPS (0.05 to 0.09 scene) sits far below persistence's (0.18 to 0.28), and the rendered prediction, scored against the raw true frame like persistence, still has higher LPIPS on 13 of 13 arenas (margin $M$ +0.03 to +0.17, median +0.08; home $-$0.08).

## 5. Section 3, the distance paragraph (lines 202 to 206)

Current (whole paragraph from "\textbf{The distance orders families, not arenas.}" to "...is the adaptation study's second axis."):

Replace with:
> \textbf{The distance orders families, not arenas.} Before scoring any map we froze $D$, the sliced Wasserstein-2 distance between the motion-weighted cloud of a map's per-frame SD~1 latents and that of its nearest training map (\appref{app:distance}).
> The training maps' held-out episodes sit inside the train-versus-train floor (0.03 to 0.07 against 0.02 to 0.09; measured on the earlier episode set, \appref{app:distance}) and every unseen arena outside it (0.12 to 0.27), so $D$ separates training maps from unseen arenas; within the 13 unseen arenas it does not order them: Spearman with $A$ $-0.22$, with the adaptation budget $-0.35$, with $A$ after adaptation $+0.10$ ($n = 13$, none below $p = 0.2$), and a leave-one-arena-out linear fit is no better than predicting the mean for $A$ after adaptation or the budget.
> Two transition-level distances we registered as alternatives, a directed coverage of transition windows by the training set and a nearest-neighbour transfer gap, fail even that separation: the training maps' held-out episodes overlap the unseen range on both (\appref{app:distance}).
> The pre-registered 30-map test, which pooled campaign maps of another WAD, failed its validation gate (\appref{app:distance}); its partial Spearman of $-0.73$ falls to $-0.42$ with family indicators as covariates, so we read it as a pooled family effect, not an ordering.
> What does order the arenas is the frozen model's own zero-shot latent skill $S_0$, one evaluation pass on the new map's footage: it predicts where an arena ends up rather than how much it gains (exploratory Spearman $+0.90$ with $A$ at 4k, $+0.18$ with the gain, $-0.61$ with the observed crossing budgets, non-crossers tied above all crossings); both persist after adjustment for motion or persistence PSNR, and the U-Net and PixArt zero-shot advantages agree in rank ($+0.96$), so the ordering is shared by two backbones trained on the same latents rather than particular to one.

## 6. Figure 2 and 3 captions (lines 190 and 197)

Figure 2, current:
> \caption{The 17 arenas sorted by $D$, training maps shaded (U-Net 200k EMA, one tic, 95\% intervals). (a) $A$; dashed: the training maps' mean. (b) $M$. $A$ drops off the training maps without following $D$; $M$ changes sign.}

Replace with (figure files come from `figures/tuned/`: `fig2a_advantage_by_distance.pdf`, and `fig2c_outcomes_by_skill.pdf`, left panel only, if you take the skill scatter; otherwise `fig2b_margin_by_distance.pdf`):
> \caption{(a) The 17 arenas sorted by $D$, training maps shaded (U-Net 200k EMA, one tic, tuned decoder, 95\% episode-bootstrap intervals): $A$ drops off the training maps without following $D$. (b) $A$ after 4k adapter updates against the frozen model's zero-shot latent skill $S_0$ on the same arena (Spearman 0.90); dashed: the home advantage.}

Figure 3, current:
> \caption{$A$ against LoRA updates per unseen arena, coloured by $D$; dashed: home advantage; a tick on each curve: its half-gap line; markers: first crossings; open: censored. Black: full fine-tune.}

Replace with:
> \caption{$A$ against adapter updates per unseen arena (tuned decoder, live adapter weights, one training seed; file `figures/tuned/fig3_adaptation_curves.pdf`), coloured by $D$; dashed: the home advantage (5.1~dB); a tick on each curve: its half-gap line; filled marker: first crossing; open: censored at 4k. In the stock-decoder check on arenas 6, 7, 8 and 16 two training seeds differ in $A$ at 4k by 0.02 to 0.06~dB (\appref{app:perarena-adapt}).}

(Drop "Black: full fine-tune" until Monday's comparator exists.)

## 7. Section 4, Protocol (line 226, last sentence) and Cost (lines 231 to 233)

Protocol, current:
> A full fine-tune of arena 7, the farthest, is the comparator.

Replace with:
> A full fine-tune at the GameNGen rate ($2\times10^{-5}$) on every arena is the comparator (\tbd{}: Monday).

Cost, current:
> \textbf{Cost.} An arena's headline cost is the first grid budget at which $A$ closes half of its own gap to the training maps' advantage $A_\text{home}$ (\prov{3.60}~dB), that is, reaches $\tfrac12(A_0+A_\text{home})$ with $A_0$ its zero-shot advantage; the second crossing is $A_\text{home}$ itself.
> Both lines are frozen from step-0 predictions; we also report the value at 4k and the area under the curve~\citep{taylor2009transfer}, and right-censor arenas that never cross.
> Every arena starts below $A_\text{home}$ (\prov{0.72 to 2.68}~dB), but \prov{5} already sit above half of it, so the line is per arena ($M$ curves: \appref{app:perarena-adapt}).

Replace with:
> \textbf{Cost.} An arena's headline cost is the first grid budget at which $A$ closes half of its own gap to the source model's in-distribution advantage on its training maps, $A_\text{home}$ (5.06~dB, interval 4.75 to 5.39), that is, reaches $\tfrac12(A_0+A_\text{home})$ with $A_0$ its zero-shot advantage; the second crossing is $A_\text{home}$ itself.
> Both lines are frozen from step-0 predictions; we also report the value at 4k and the area under the curve~\citep{taylor2009transfer}, and right-censor arenas that never cross.
> Every arena starts below $A_\text{home}$ ($A_0$ 1.07 to 2.88~dB), and because the line depends on $A_0$ we do not correlate the budget with $A_0$ itself ($M$ curves: \appref{app:perarena-adapt}).

## 8. Section 4, What it takes (lines 237 to 239)

Current (three sentences with \tbd):

Replace with:
> \textbf{What it takes.} 9 of 13 arenas close half of their gap within 4k updates (5 within 500; median budget 1k, arena-bootstrap 250 to beyond 4k; arenas 1, 8, 12 and 16 censored), one reaches the home line, and the median $A$ rises from 2.04 to 3.80~dB (Figure~\ref{fig:adapt}, Table~\ref{tab:cost}); on average 69 percent of the gain lands by 250 updates, and every curve still rises between 2k and 4k (+0.05 to +0.31~dB, mean +0.13).
> Adaptation also closes the perceptual gap: at 4k the rendered prediction has lower scene LPIPS than raw persistence on 11 of 13 arenas, the other two within 0.01 ($M$ median $-0.02$ against $+0.08$ zero-shot), and the median gap $G$ to the decoder's own reconstruction falls from 5.0 to 3.4~dB, the home value (3.4).
> Little of the gain needs data: in stock-decoder checks a single adaptation episode finishes within about 0.25~dB of sixteen on arenas 7, 8, 12 and 16 (75 to 101 percent of the sixteen-episode gain, \appref{app:perarena-adapt}), and tripling or quintupling the learning rate or extending training to 8k changes $A$ at 4k by less than 0.06~dB on arenas 12 and 16, inside the two-seed spread of 0.02 to 0.06~dB; on these checks the ceiling an arena reaches looks set by the arena rather than by the budget, the rate or the data.
> The budget tracks the zero-shot latent skill (Spearman $-0.61$) and not $D$ ($-0.35$) or the transition distances, exploratory at $n=13$ with censored budgets; the forgetting and directional guards on all arenas and the full fine-tune comparator are \tbd{} (the guards land with the 8k rerun today, the comparator Monday).

## 9. Table 3 (lines 241 to 257)

Replace the caption's first clause and the body:

Caption, current start: "What crossing costs (U-Net 200k from EMA weights, 8 episodes, live weights; ${>}$4k: censored, medians too when fewer than 7 cross)."
Replace with: "What crossing costs (U-Net 200k from EMA weights, 8 episodes, live weights, tuned decoder, scene crop; ${>}$4k: censored)."

Body rows (keep the header row; replace the five data rows):
```
Parameters trained; GPU-hours & 4.2M (0.49\%); 1.1 per arena & 4.2M; 1.1 & 860M (all); \tbd{} \\
Arenas past half gap / home line by 4k & 9 / 1 of 13 & -- & -- \\
Cost to half gap / home line (updates) & 1k / ${>}$4k & 250 / ${>}$4k & \tbd{} / \tbd{} \\
$A$ (dB) / $M$ / $G$ (dB) at 4k & +3.80 / $-$0.02 / 3.44 & +3.86 / $-$0.03 / 4.65 & \tbd{} / \tbd{} / \tbd{} \\
Forgetting (dB) / directional at 4k & \tbd{} / \tbd{} & \tbd{} / \tbd{} & \tbd{} / \tbd{} \\
```
Arena 7 $M$ and $G$ at 4k: $-0.026$ and 4.65 (`scene_vae_psnr_tuned` minus `scene_psnr_raw_tuned`) from `results/fresh_rescore/adapt4000_live_tuned/map07/metrics.json`. Also change the header's third column to "Full, 13 arenas" once Monday's runs exist. GPU-hours: 4,000 updates at 0.99 s on an A4000 (log.jsonl) is 1.1 h per arena.

## 10. Limitations and conclusion (line 277)

Current:
> Off its training maps the U-Net keeps its turn response and part of its advantage over copying, loses its perceptual advantage, and closes half of its gap to the home advantage within \tbd{} adapter updates.

Replace with:
> Off its training maps the U-Net keeps its turn response and part of its advantage over copying and loses its perceptual advantage; eight episodes and 4k rank-16 adapter updates (1.1 GPU-hours) close half of the gap on 9 of 13 arenas (arena-bootstrap 46 to 92 percent) and restore the perceptual advantage on 11, and in our checks one episode gives most of that.

## Not in this diff (Monday)

Full fine-tune column; forgetting and directional guards at 4k (the overnight scores ran without guards); PixArt and SD 3.5 rows of Table 2 (SD 3.5 provisional read lands this morning, final after the 200k rescore); the sd35 decoder column.

## Appendix additions implied (I will send these as a separate diff once the tuned tables are final)

- `app:perarena`: the per-arena zero-shot table gains a stock/tuned pair of columns for $A$, $M$ and $G$ (source: `results/fresh_rescore/*_tuned/map*/metrics.json`), and one sentence on the two decoders (matched MSE + 0.1 LPIPS, 3,486 steps, gate +4.3 dB / LPIPS $-$0.039; MSE-only GameNGen recipe +5.1 dB / LPIPS +0.19, the blur row).
- `app:distance`: the coverage and transfer-gap definitions, their family-floor failure and their Spearman rows (`results/transition_distance/compare_fresh_sd1.csv`).
- `app:perarena-adapt`: the tuned per-arena table (`paper/tables/tuned/adapt_perarena.tex`), the seed spread, the data ladder figure (`fig_adapt_ladder`) and the recipe figure (`fig_adapt_recipe`).

## 11. Section 2, the decoder sentence (line 121), and the appendix

Current:
> Rollouts stay in latent space, so the decoder only renders. Following GameNGen, we plan to tune it with MSE on training-map frames, then freeze it.

Replace with:
> Rollouts stay in latent space, so the decoder only renders. We fine-tune the SD~1 decoder on training-map frames from their true latents with the encoder frozen, using MSE plus 0.1~LPIPS (GameNGen's MSE-only recipe sharpens the HUD but triples scene LPIPS; \appref{app:perarena}), then freeze it; every pixel number in Sections 3 and 4 uses this decoder unless marked stock.

The appendix's decoder paragraph (appendix.tex line 35) changes the same way; the launcher is `scripts/spiderman/decoder_mse_lpips.sh` (3,486 steps, lr 1e-5, batch 24, 4.0 h on an A6000; gate +4.33 dB, LPIPS $-$0.039 on the dev set).

