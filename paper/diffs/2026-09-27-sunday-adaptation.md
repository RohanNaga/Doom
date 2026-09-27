# Overleaf diff, Sunday Sep 27 (adaptation, distance, decoder)

Apply to `main.tex` as it stands in Overleaf. Each item gives the exact current text and the exact replacement. Numbers: tuned decoder (MSE + 0.1 LPIPS) unless marked stock; scene crop; live weights; 256 fresh held-out windows per arena; home = the training maps' 512 validation windows. Sources: `paper/tables/tuned/adapt_cost.tex`, `paper/tables/tuned/adapt_summary.json`, `results/fresh_rescore/*_tuned/map*/metrics.json` (script in the RESEARCH_CONTEXT 2026-09-27 08:05 entry), `.claude/analyses/distance-usefulness-decision-2026-09-27.md`. Final numbers (the arena 6 tuned rescore landed at 08:12: it crosses at 1,000).

Metric conventions the diff assumes (say if you want the other): $A$ = scene PSNR of the rendered prediction minus scene PSNR of the rendered copy-last frame, both against the rendered true frame (unchanged). $M$ = scene LPIPS of the rendered prediction against the **raw** true frame minus scene LPIPS of raw persistence, so both sides are measured against the same raw frame; the decoded-reference version (kinder to the model by about 0.03) goes to the appendix. $G$ = scene PSNR of the decoder's own reconstruction minus scene PSNR of the prediction, same decoder.

## 1. Abstract, sentence 3 (line 67)

Current:
> Decoded through one decoder, the U-Net still beats copy-last on every unseen arena, but its advantage shrinks from \prov{3.2--4.3}~dB at home to \prov{0.7--2.7}~dB and its LPIPS is worse than persistence's on \prov{13 of 13} arenas, while its turn response survives.

Replace with:
> Decoded through one decoder, the U-Net still beats copy-last on every unseen arena, but its advantage shrinks from 5.1~dB at home to 1.1--2.9~dB (median 2.0) and its LPIPS is worse than persistence's on 13 of 13 arenas, while its turn response survives.

## 2. Abstract, sentence 5 (line 69)

Current:
> We measure what closing half of each arena's gap to the home advantage costs rank-16 adapters on the frozen backbone: \tbd{} updates on \tbd{} of 13 arenas, at \tbd{}~dB of forgetting.

Replace with:
> Rank-16 adapters on the frozen backbone, trained on eight episodes of an arena, close half of its gap to the home advantage within 4k updates on 9 of 13 arenas (five within 500), recover a median 1.8~dB, and turn the perceptual loss into a tie or a win on 11 of 13; one episode already yields most of the gain, and no per-frame or transition-level distance predicts which arenas adapt fastest, while the frozen model's own zero-shot latent skill does.

(If this runs the abstract past six sentences, drop "one episode already yields most of the gain, and".)

## 3. Contributions (3) and (4) (line 96)

Current:
> (3) per-arena adaptation curves, with the cost for rank-16 adapters to close half of each arena's gap to the home advantage (\tbd{}) and a full fine-tune comparator; and (4) two negative results: a per-frame distance has no clear association with the advantage among arenas of one WAD, and one-step quality does not guarantee closed-loop stability.

Replace with:
> (3) per-arena adaptation curves for rank-16 adapters, with the budget to close half of each arena's gap to the home advantage (median 1k updates; 9 of 13 by 4k), a data ladder showing one episode recovers most of the gain, and a full fine-tune comparator (\tbd{}); and (4) two negative results: neither a per-frame distance nor two transition-level distances order the arenas by zero-shot advantage or adaptation budget, whereas the frozen model's zero-shot latent skill does (Spearman 0.90 with the advantage at 4k), and one-step quality does not guarantee closed-loop stability.

## 4. Section 3, the decoder lines (lines 179 to 180)

Current:
> The rendering of true latents does not degrade: the ceiling is \prov{23.91}~dB at home and \prov{23.62}~dB unseen, so the loss is not the decoder's floor; decoding the same predictions with the stock and tuned decoders (\tbd{}) tests whether it also renders off-distribution latents worse.
> Nor is the perceptual flip the decoder's floor: its own LPIPS (\prov{0.08 to 0.12}) sits below persistence's (\prov{0.15 to 0.23}), and the model, scored against the decoded true frame, still has higher LPIPS than raw persistence (\prov{13 of 13}).

Replace with:
> The rendering of true latents does not degrade: the reconstruction ceiling is 28.6~dB at home and 25.2 to 29.7~dB unseen, so the loss is in the predicted latents; the gap to the ceiling $G$ is 1.1~dB at home and 1.8 to 6.0~dB unseen (median 3.6). Decoding the same predictions with the stock and the tuned decoder raises $A$ by 0.2~dB zero-shot ($-0.01$ to $+0.51$) but by 0.7~dB after adaptation (0.5 to 1.1), so the tuned decoder renders adapted latents better than the frozen model's off-distribution ones: the decoder carries a share of the off-map deficit, and we report every column under both decoders (\appref{app:perarena}).
> Nor is the perceptual flip the decoder's floor: the decoder's own reconstruction LPIPS (0.05 to 0.09 scene) sits far below persistence's (0.18 to 0.28), and the rendered prediction, scored against the raw true frame like persistence, still has higher LPIPS on 13 of 13 arenas (margin $M$ +0.03 to +0.17, median +0.08; home $-$0.08).

## 5. Section 3, the distance paragraph (lines 202 to 206)

Current (whole paragraph from "\textbf{The distance orders families, not arenas.}" to "...is the adaptation study's second axis."):

Replace with:
> \textbf{The distance orders families, not arenas.} Before scoring any map we froze $D$, the sliced Wasserstein-2 distance between the motion-weighted cloud of a map's per-frame SD~1 latents and that of its nearest training map (\appref{app:distance}).
> The training maps' held-out episodes sit inside the train-versus-train floor (0.03 to 0.07 against 0.02 to 0.09) and every unseen arena outside it (0.12 to 0.27), so $D$ separates the families; within the 13 unseen arenas it does not order them: Spearman with $A$ $-0.22$, with the adaptation budget $-0.35$, with $A$ after adaptation $+0.10$ ($n = 13$, none below $p = 0.15$), and a leave-one-arena-out fit predicts no better than the mean.
> Two transition-level distances we registered as alternatives, a directed coverage of transition windows by the training set and a nearest-neighbour transfer gap, fail even the family test: the training maps' held-out episodes score inside the unseen range on both (\appref{app:distance}).
> The pre-registered 30-map test, which pooled campaign maps of another WAD, had a partial Spearman of $-0.73$ that falls to $-0.42$ with family indicators as covariates; we read it as a pooled family effect, not an ordering.
> What does order the arenas is the frozen model's own zero-shot latent skill $S_0$, one evaluation pass on the new map's footage: Spearman $+0.90$ with $A$ at 4k and $-0.61$ with the budget, unchanged when motion or persistence PSNR is partialled out, and PixArt's zero-shot $A$ ranks the arenas the same way ($+0.95$), so the ordering is a property of the footage rather than of one backbone.

## 6. Figure 2 and 3 captions (lines 190 and 197)

Figure 2, current:
> \caption{The 17 arenas sorted by $D$, training maps shaded (U-Net 200k EMA, one tic, 95\% intervals). (a) $A$; dashed: the training maps' mean. (b) $M$. $A$ drops off the training maps without following $D$; $M$ changes sign.}

Replace with (and swap the second panel's file to `figures/fig2c_outcomes_by_skill.pdf`, left panel only, if you take the skill scatter; otherwise keep 2b):
> \caption{(a) The 17 arenas sorted by $D$, training maps shaded (U-Net 200k EMA, one tic, tuned decoder, 95\% episode-bootstrap intervals): $A$ drops off the training maps without following $D$. (b) $A$ after 4k adapter updates against the frozen model's zero-shot latent skill $S_0$ on the same arena (Spearman 0.90); dashed: the home advantage.}

Figure 3, current:
> \caption{$A$ against LoRA updates per unseen arena, coloured by $D$; dashed: home advantage; a tick on each curve: its half-gap line; markers: first crossings; open: censored. Black: full fine-tune.}

Replace with:
> \caption{$A$ against adapter updates per unseen arena (tuned decoder, live weights), coloured by $D$; dashed: the home advantage (5.1~dB); a tick on each curve: its half-gap line; filled marker: first crossing; open: censored at 4k. Seed-to-seed spread at 4k is 0.02 to 0.06~dB (\appref{app:perarena-adapt}).}

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
> \textbf{Cost.} An arena's headline cost is the first grid budget at which $A$ closes half of its own gap to the training maps' advantage $A_\text{home}$ (5.06~dB), that is, reaches $\tfrac12(A_0+A_\text{home})$ with $A_0$ its zero-shot advantage; the second crossing is $A_\text{home}$ itself.
> Both lines are frozen from step-0 predictions; we also report the value at 4k and the area under the curve~\citep{taylor2009transfer}, and right-censor arenas that never cross.
> Every arena starts below $A_\text{home}$ ($A_0$ 1.07 to 2.88~dB), and because the line depends on $A_0$ we do not correlate the budget with $A_0$ itself ($M$ curves: \appref{app:perarena-adapt}).

## 8. Section 4, What it takes (lines 237 to 239)

Current (three sentences with \tbd):

Replace with:
> \textbf{What it takes.} 9 of 13 arenas close half of their gap within 4k updates (5 within 500; median budget 1k; arenas 1, 8, 12 and 16 censored), one reaches the home line, and the median $A$ rises from 2.04 to 3.80~dB (Figure~\ref{fig:adapt}, Table~\ref{tab:cost}); 69 percent of the gain lands by 250 updates and the curves still rise by 0.13~dB from 2k to 4k on every arena.
> Adaptation also closes the perceptual gap: at 4k the rendered prediction ties or beats raw persistence in LPIPS on 11 of 13 arenas ($M$ median $-0.02$ against $+0.08$ zero-shot) and the gap to the ceiling falls from 3.6 to 1.4~dB (home 1.1).
> The gain is nearly data-free: on arenas 7, 8, 12 and 16 a single adaptation episode reaches within 0.0 to 0.25~dB of sixteen (\appref{app:perarena-adapt}), and neither a threefold or fivefold learning rate nor 8k updates moves $A$ at 4k beyond the seed-to-seed spread (0.02 to 0.06~dB), so each arena's ceiling is set by the arena, not by the budget, the rate or the data.
> The budget tracks the zero-shot latent skill (Spearman $-0.61$) and not $D$ ($-0.35$) or the transition distances (no better), exploratory at $n=13$; the forgetting and directional guards and the full fine-tune comparator are \tbd{} (Monday).

## 9. Table 3 (lines 241 to 257)

Replace the caption's first clause and the body:

Caption, current start: "What crossing costs (U-Net 200k from EMA weights, 8 episodes, live weights; ${>}$4k: censored, medians too when fewer than 7 cross)."
Replace with: "What crossing costs (U-Net 200k from EMA weights, 8 episodes, live weights, tuned decoder, scene crop; ${>}$4k: censored)."

Body rows (keep the header row; replace the five data rows):
```
Parameters trained; GPU-hours & 4.2M (0.49\%); 1.1 per arena & 4.2M; 1.1 & 860M (all); \tbd{} \\
Arenas past half gap / home line by 4k & 9 / 1 of 13 & -- & -- \\
Cost to half gap / home line (updates) & 1k / ${>}$4k & 250 / ${>}$4k & \tbd{} / \tbd{} \\
$A$ (dB) / $M$ / $G$ (dB) at 4k & +3.80 / $-$0.02 / 1.40 & +3.86 / $-$0.03 / 1.28 & \tbd{} / \tbd{} / \tbd{} \\
Forgetting (dB) / directional at 4k & \tbd{} / \tbd{} & \tbd{} / \tbd{} & \tbd{} / \tbd{} \\
```
Arena 7 $M$ and $G$ at 4k: $-0.026$ and 1.28 from `results/fresh_rescore/adapt4000_live_tuned/map07/metrics.json`. GPU-hours: 4,000 updates at 0.99 s on an A4000 (log.jsonl) is 1.1 h per arena.

## 10. Limitations and conclusion (line 277)

Current:
> Off its training maps the U-Net keeps its turn response and part of its advantage over copying, loses its perceptual advantage, and closes half of its gap to the home advantage within \tbd{} adapter updates.

Replace with:
> Off its training maps the U-Net keeps its turn response and part of its advantage over copying and loses its perceptual advantage; eight episodes and 4k rank-16 adapter updates (1.1 GPU-hours) close half of the gap on 9 of 13 arenas and restore the perceptual tie on 11, and one episode gives most of that.

## Not in this diff (Monday)

Full fine-tune column; forgetting and directional guards at 4k (the overnight scores ran without guards); PixArt and SD 3.5 rows of Table 2 (SD 3.5 provisional read lands this morning, final after the 200k rescore); the sd35 decoder column.

## Appendix additions implied (I will send these as a separate diff once the tuned tables are final)

- `app:perarena`: the per-arena zero-shot table gains a stock/tuned pair of columns for $A$, $M$ and $G$ (source: `results/fresh_rescore/*_tuned/map*/metrics.json`), and one sentence on the two decoders (matched MSE + 0.1 LPIPS, 3,486 steps, gate +4.3 dB / LPIPS $-$0.039; MSE-only GameNGen recipe +5.1 dB / LPIPS +0.19, the blur row).
- `app:distance`: the coverage and transfer-gap definitions, their family-floor failure and their Spearman rows (`results/transition_distance/compare_fresh_sd1.csv`).
- `app:perarena-adapt`: the tuned per-arena table (`paper/tables/tuned/adapt_perarena.tex`), the seed spread, the data ladder figure (`fig_adapt_ladder`) and the recipe figure (`fig_adapt_recipe`).
