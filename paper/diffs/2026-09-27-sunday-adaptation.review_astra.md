# Sunday adaptation diff — Astra truth review

Reviewed locally with `/opt/miniconda3/envs/PERSEVE/bin/python` against `paper/diffs/2026-09-27-sunday-adaptation.md`, the surrounding `paper/main.tex`, and `RESEARCH_CONTEXT.md` section 0 and September 27 entries. No SSH, commits, or source edits.

**Verdict:** the tuned A/cost headline reproduces. The G values use incompatible PSNR reference targets; arena 7’s 1.28 does not reproduce under either convention. The seed/ladder/recipe evidence is stock-only, the tuned cross-backbone correlation rounds to 0.96, and several causal/predictive statements exceed these observations.

## Counting and method

The table inventories every distinct quantitative assertion proposed for the paper, including the preamble and planned appendix additions. Repeated occurrences and alternative rounding are grouped and their locations listed. Counts below mean **table assertions**, not individual digit tokens or repeated occurrences. Editorial section/figure/line numbers, dates, filenames, model names (SD 1 / SD 3.5), the optional six-sentence abstract limit, and superseded text under “Current” are not new empirical assertions. DISPUTED includes wrong values, wrong decoder scope, and unavailable verification; it does not imply every unavailable claim is false.

For each live curve, select rows with non-null `heldout_A_tuned`, keyed by step; stock-only rows are used explicitly for the ablations. Recompute thresholds from the frozen home value, without interpolation. In Spearman calculations, put all four noncrossers in one tied tier above every crossing (8,000 is just a rank sentinel, **not an observed cost**). Cross-map medians are unweighted. Independently subtract named means in the raw rescore files for A and M.

**G must have a common raw target.** `score_adapt.py` explicitly defines its C/G column as `scene_vae_psnr - scene_psnr_raw` (both through the same decoder, both against raw truth). The diff’s G wording omits the reference target. Most of its claimed numbers instead reproduce `scene_vae_psnr_tuned - scene_psnr_dec_tuned`, which subtracts a reconstruction score against raw truth from a prediction score against decoded truth. I report both computations below, but dispute the latter as a “gap to the ceiling.” If mixed references were intended, that must be explicitly defined and its ceiling interpretation withdrawn.

Source abbreviations (all paths relative to repository root):

- **C:** `results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl`; neighboring `config.json`/`log.jsonl` when stated.
- **T:** `paper/tables/tuned/adapt_summary.json` and `paper/tables/tuned/adapt_cost.tex`; stock cross-check: `paper/tables/adapt_summary.json`.
- **Z:** `results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json`.
- **E:** `results/fresh_rescore/adapt4000_live_tuned/map??/metrics.json`.
- **P:** `results/fresh_rescore/pixart200k_ema_tuned/map??/metrics.json`.
- **H:** `results/fresh_rescore/home_unet200k_ema_tuned/val/metrics.json`; **H0:** `results/home_unet_tuned/metrics.json`.
- **L:** `results/adapt/unet200k_arenas13_map{07,08,12,16}_r16_k{1,2,4,8,16}_s0/scores.jsonl`.
- **S:** `results/adapt/unet200k_arenas13_map{06,07,08,16}_r16_k8_s{0,1}/scores.jsonl`.
- **Q:** `results/adapt/unet200k_arenas13_map{12,16}_r16_k8_s0{,_lr3e4,_lr5e4,_g8k}/scores.jsonl`.
- **D:** `results/distance_v2/distances_sd1.json`; **Dold:** `results/distance_study/distances_sd1.json`.
- **X:** `results/transition_distance/compare_fresh_sd1.csv`.
- **O:** `results/distance_study/figure_unet_h1_amended/stats.json`, recomputing from its map records, and the original `figure_unet_h1/stats.json`.

## Number audit

**Audit totals: 65 CONFIRMED; 17 DISPUTED, across 82 distinct quantitative assertions.**

| ID | Diff location | Proposed number/count | Recomputed value / finding | Source | Verdict |
|---|---|---|---|---|---|
| N01 | Preamble; throughout | 256 held-out windows per arena | 256 in all 13 headline certificates and each zero-shot/adapted metric read | C; Z; E | **CONFIRMED** |
| N02 | Preamble | 512 home validation windows | 512; maps 2–5, with 123/137/134/118 windows | H and its per_window.csv | **CONFIRMED** |
| N03 | Preamble; appendix | MSE + 0.1 LPIPS | 0.1 documented in context, but decoder training config/log is absent; not independently verified | RESEARCH_CONTEXT.md, Sep 27 05:20; H.extra_decoders | **DISPUTED** |
| N04 | Preamble | Decoded-reference LPIPS kinder by about 0.03 | Mean raw-minus-decoded margin 0.026092 zero-shot, 0.031149 adapted | Z; E | **CONFIRMED** |
| N05 | Preamble; §§2,3,6–10 | 4k / 4,000 update endpoint and censoring | Last headline grid point 4,000; grid 0/250/500/1,000/2,000/4,000 | C | **CONFIRMED** |
| N06 | Preamble | Arena 6 crosses at 1,000 | A500=2.996892 < 3.064090; A1000=3.129848 >= 3.064090 | C, map06 | **CONFIRMED** |
| N07 | §§1,6,7 | Home A = 5.1 / 5.06 dB | 5.059250826 frozen home; independent fresh home 5.059690252, both round as quoted | H0; H; T | **CONFIRMED** |
| N08 | §§1,7 | Zero-shot A range 1.1–2.9 / 1.07–2.88 | Curves: 1.068928961–2.878576975; fresh metrics: 1.068552922–2.878588539 | C; Z | **CONFIRMED** |
| N09 | §§1,8 | Median A0 = 2.0 / 2.04 | 2.036097892 from curves (fresh independent read 2.034840479) | C; T; Z | **CONFIRMED** |
| N10 | §§1,4 | LPIPS worse on 13 of 13 unseen arenas | 13 strictly positive raw-reference scene margins | Z | **CONFIRMED** |
| N11 | §§2,3,7,10 | Rank 16 | rank=16 in every headline certificate | C config.json | **CONFIRMED** |
| N12 | §§2,9,10 | Eight / 8 adaptation episodes | 8 listed episode IDs in each headline certificate | C config.json | **CONFIRMED** |
| N13 | §§2,3,8–10 | 9 of 13 cross half gap by 4k | 9/13: maps 6,7,9,10,11,13,14,15,17 | C; T | **CONFIRMED** |
| N14 | §§2,8 | Five / 5 by 500 updates | 4 by 250 (7,9,10,11), plus map15 at 500 | C; T | **CONFIRMED** |
| N15 | §2 | Median recovery 1.8 dB | Median paired gain 1.768414170 dB | C | **CONFIRMED** |
| N16 | §§2,8,10 | LPIPS tie or win on 11 of 13 at 4k | 11 strictly negative means; maps 1 and 6 remain positive (+0.000976,+0.007671) | E | **CONFIRMED** |
| N17 | §§2,3,8,10 | One episode gives most of the gain | Stock ladder k=1 retains about 90.6%, 101.1%, 74.8%, 93.7% of k=16 gain on maps 7/8/12/16; no tuned k=1 reads | L, heldout_A_stock | **DISPUTED** |
| N18 | §§3,8,9 | Median half-gap budget 1k / 1,000 | 7th ordered observation crosses at 1,000; four observations are right-censored | C; T | **CONFIRMED** |
| N19 | §§3,5,6 | S0 vs A4k Spearman +0.90 | 0.895604396; p=0.0000348097 | C, heldout_latent_skill / heldout_A_tuned | **CONFIRMED** |
| N20 | §3 | Two transition-level distances | Two candidate columns: coverage and G (transfer gap) | X | **CONFIRMED** |
| N21 | §3 | Two negative results | Two enumerated claims: distance ordering and one-step/closed-loop mismatch; this count does not validate either inference | Diff §3; paper/main.tex closed-loop paragraph | **CONFIRMED** |
| N22 | §4 | Home reconstruction ceiling 28.6 dB | 28.553970795 | H, scene_vae_psnr_tuned.mean | **CONFIRMED** |
| N23 | §4 | Unseen reconstruction ceiling 25.2–29.7 dB | 25.248747557–29.662555285 | Z, scene_vae_psnr_tuned.mean | **CONFIRMED** |
| N24 | §§4,8 | Home G = 1.1 dB | 3.357585488 dB with common raw target; 1.070926586 only with mixed raw/decoded targets | H; score_adapt.py OUTCOME_FIELDS C | **DISPUTED** |
| N25 | §4 | Unseen G = 1.8–6.0 dB | 3.703936914–6.861428615 raw-reference; mixed-reference range is 1.778705083–5.972934246 | Z; score_adapt.py | **DISPUTED** |
| N26 | §§4,8 | Median zero-shot G = 3.6 dB | 4.956499830 raw-reference; 3.562622257 mixed-reference | Z; score_adapt.py | **DISPUTED** |
| N27 | §4 | Decoder raises zero-shot A by 0.2 dB | Across-arena mean 0.220798113; median 0.222126469 | Z, paired tuned-minus-stock A | **CONFIRMED** |
| N28 | §4 | Zero-shot decoder delta −0.01 to +0.51 dB | −0.011285976 to +0.513073497 | Z | **CONFIRMED** |
| N29 | §4 | Decoder raises adapted A by 0.7 dB | Across-arena mean 0.705498321; median 0.660997532 | E, paired tuned-minus-stock A | **CONFIRMED** |
| N30 | §4 | Adapted decoder delta 0.5–1.1 dB | 0.497697953–1.106510781 | E | **CONFIRMED** |
| N31 | §4 | Reconstruction scene LPIPS 0.05–0.09 | 0.053377704–0.090877407 | Z, scene_vae_lpips_tuned.mean | **CONFIRMED** |
| N32 | §4 | Persistence scene LPIPS 0.18–0.28 | 0.181516778–0.275220065 | Z, scene_persist_lpips_raw.mean | **CONFIRMED** |
| N33 | §4 | M range +0.03 to +0.17 | +0.026294222 to +0.172870874 | Z | **CONFIRMED** |
| N34 | §§4,8 | Median M0 +0.08 | +0.080553671 | Z | **CONFIRMED** |
| N35 | §4 | Home M −0.08 | −0.081099200 | H | **CONFIRMED** |
| N36 | §5 | Wasserstein-2 | p=2 in frozen distance configuration | D.config.p | **CONFIRMED** |
| N37 | §5 | Training-map D 0.03–0.07 | 0.026038679–0.069316362, from the older validation read; absent from fresh D file | Dold, motion-arm val rows | **CONFIRMED** |
| N38 | §5 | Train/train floor 0.02–0.09 | 0.022325066–0.093373498 | D.floor.motion.values | **CONFIRMED** |
| N39 | §5 | Unseen D 0.12–0.27 | 0.117886855–0.270253779 | D, primary motion-arm maps | **CONFIRMED** |
| N40 | §§5,8 | n = 13 unseen arenas | 13 complete headline maps, IDs 1 and 6–17 | C; D; X | **CONFIRMED** |
| N41 | §5 | D vs A0 Spearman −0.22 | −0.219780220; p=0.470614851 | C + D | **CONFIRMED** |
| N42 | §§5,8 | D vs budget Spearman −0.35 | −0.354279700; p=0.234959255; censored observations tied above all crossings | C + D | **CONFIRMED** |
| N43 | §5 | D vs adapted A Spearman +0.10 | +0.098901099; p=0.747868304 | C + D | **CONFIRMED** |
| N44 | §5 | None of those three correlations below p=0.15 | p=0.470615, 0.234959, 0.747868; minimum 0.234959 | C + D; scipy.stats.spearmanr | **CONFIRMED** |
| N45 | §5 | Earlier 30-map test | 30 map records; different footage and raw full-frame gain outcome | O.maps | **CONFIRMED** |
| N46 | §5 | Earlier partial Spearman −0.73 | −0.733703977 after residualizing ranks on persistence | O.maps, independently recomputed | **CONFIRMED** |
| N47 | §5 | Earlier partial Spearman −0.42 with family controls | −0.422745054 with persistence, training indicator AND campaign indicator | O.maps, independently recomputed | **CONFIRMED** |
| N48 | §§5,8 | S0 vs budget Spearman −0.61 | −0.606526846; p=0.027966358; tied censoring tiers | C | **CONFIRMED** |
| N49 | §5 | PixArt/U-Net zero-shot A rank correlation +0.95 | Tuned: 0.956043956 → +0.96; stock: 0.950549451 → +0.95 | Z + P, scene_*_dec_tuned vs unsuffixed fields | **DISPUTED** |
| N50 | §6 | 17 arenas | 4 training maps + 13 unseen arenas | H per_window.csv; Z | **CONFIRMED** |
| N51 | §§6,9 | U-Net 200k EMA source | certificate source.step=200000, source.weights=ema; fresh zero-shot config.use_ema=true | C config.json; Z.config | **CONFIRMED** |
| N52 | §6 | One tic | horizon_tics=1 and game_time_tics=1 | Z.config; E.config | **CONFIRMED** |
| N53 | §6 | 95% episode-bootstrap intervals | Percentiles 2.5 and 97.5 of resampled episode pooled means; nominal 95%, not a coverage guarantee | paper/make_adapt_figures.py:314–325; C per-window files | **CONFIRMED** |
| N54 | §§6,8 | Seed-to-seed spread 0.02–0.06 dB at 4k | Stock absolute seed differences: 0.031383/0.019464/0.057473/0.051099 on maps 6/7/8/16; tuned seed-1 reads absent | S, heldout_A_stock | **DISPUTED** |
| N55 | §7 | Full fine-tune lr 2×10^-5 on every arena | 0.00002 is a proposed rate, not a run result; local literature memo attributes it to GameNGen, but primary paper and comparator runs were not available for verification | C contains no full-finetune comparator; .claude/analyses/lit-adapters-2026-09-26.md §4.2 citation | **DISPUTED** |
| N56 | §7 | Half-gap coefficient 1/2 | For each map target=(A0+5.059250826)/2; no step-0 crossing | C; H0 | **CONFIRMED** |
| N57 | §8 | Censored arenas 1, 8, 12, 16 | Exactly these four do not cross by 4,000 | C | **CONFIRMED** |
| N58 | §8; §9 home count | One arena reaches home | Map 9 at 4,000, A=5.109124139 >= 5.059250826 | C | **CONFIRMED** |
| N59 | §§8,9 | Median A4k = 3.80 dB | 3.799065340 in curves; 3.799084213 in independent fresh metrics | C; E | **CONFIRMED** |
| N60 | §8 | 69 percent of gain by 250 | Mean of arena-wise fractions = 69.4922599%; fractions range 53.6812–84.1986% | C | **CONFIRMED** |
| N61 | §8 | +0.13 dB from 2k to 4k on every arena | Mean +0.131221284, but individual gains +0.050697900 to +0.304655608; all 13 positive, not each +0.13 | C | **DISPUTED** |
| N62 | §§8,9 | Median M4k −0.02 | −0.021496872 | E | **CONFIRMED** |
| N63 | §§8,9 | Median G4k 1.4 / 1.40 | 3.441895958 raw-reference; 1.402765960 mixed-reference | E; score_adapt.py | **DISPUTED** |
| N64 | §8 | Ladder arenas 7, 8, 12, 16; 1 vs 16 episodes | All four have k=1/2/4/8/16; certificates list those counts | L config.json | **CONFIRMED** |
| N65 | §8 | One-to-sixteen episode difference 0.0–0.25 dB | Stock signed A16−A1: +0.231417, −0.011196, +0.252777, +0.047252; rounded maximum 0.25, but absent for tuned decoder | L, heldout_A_stock | **DISPUTED** |
| N66 | §8 | Threefold / fivefold learning rates | 0.0003 and 0.0005 vs base 0.0001, on maps 12 and 16 only | Q config.json | **CONFIRMED** |
| N67 | §8 | 8k update recipe test | 8,000 final step, maps 12/16 only; stock A4k→A8k: 2.563468→2.560517 and 2.986512→2.975681 | Q scores.jsonl | **CONFIRMED** |
| N68 | §9 | 4.2M trained parameters | 4,212,544 = 3,188,736 LoRA + 630,528 control + 380,480 input + 12,800 noise | C config.json certificate.counts | **CONFIRMED** |
| N69 | §9 | 0.49 percent trainable | 4,212,544 / 860,532,932 × 100 = 0.489527343%; denominator excludes inserted LoRA | C config.json certificate.counts | **CONFIRMED** |
| N70 | §9 | 860M full model | 863,721,668 total with LoRA − 3,188,736 LoRA = 860,532,932 | C config.json certificate.counts | **CONFIRMED** |
| N71 | §§9,10 | 1.1 GPU-hours per arena | Checkpoint-0→4k elapsed 1.090190–1.124915 h on world=1; median 1.101621 h; excludes encoding/scoring/decoder tune | C log.jsonl and config.json | **CONFIRMED** |
| N72 | §9 note | 0.99 seconds per update | Median checkpoint-0→4k elapsed / 4,000 = 0.991459 s; arena7 0.986303 s | C log.jsonl | **CONFIRMED** |
| N73 | §9 | Median cost half/home = 1k / >4k | Median half cost 1,000; only one home crossing, so median home cost >4,000 | C | **CONFIRMED** |
| N74 | §9 | Arena7 cost half/home = 250 / >4k | Half threshold 3.064782, crossed at 250; no home crossing | C, map07 | **CONFIRMED** |
| N75 | §9 | Arena7 A4k +3.86 | 3.863947392 curves; 3.863700394 fresh read | C map07; E map07 | **CONFIRMED** |
| N76 | §9 and note | Arena7 M4k −0.03 / −0.026 | −0.026093831 | E map07 | **CONFIRMED** |
| N77 | §9 and note | Arena7 G4k 1.28 | 4.649473611 raw-reference; even the diff’s mixed-reference arithmetic gives 3.179991782, not 1.28 | E map07, scene_vae_psnr_tuned minus scene_psnr_raw_tuned (or dec_tuned) | **DISPUTED** |
| N78 | Appendix plan | 3,486 decoder steps | No local decoder training log/config/checkpoint to recompute; context says 3,486 | RESEARCH_CONTEXT.md Sep 27 05:20; referenced results_spiderman tree absent | **DISPUTED** |
| N79 | Appendix plan | Matched decoder gate +4.3 dB | Gate files absent; home full-frame reconstruction improvement is 4.734138, but that is a different evaluation and cannot verify the gate | H; absent decoder gate artifact | **DISPUTED** |
| N80 | Appendix plan | Matched decoder gate LPIPS −0.039 | Gate files absent; no defensible gate recomputation | Absent decoder gate artifact | **DISPUTED** |
| N81 | Appendix plan | MSE-only decoder +5.1 dB | MSE-only gate/metrics absent; context is secondary evidence only | RESEARCH_CONTEXT.md Sep 27 08:05; absent MSE-only gate artifact | **DISPUTED** |
| N82 | Appendix plan | MSE-only decoder LPIPS +0.19 | MSE-only gate/metrics absent; cannot recompute | RESEARCH_CONTEXT.md Sep 27 08:05; absent MSE-only gate artifact | **DISPUTED** |

## Claims and wording the files do not support

1. **“The rendering of true latents does not degrade … so the loss is in the predicted latents.”** The unseen reconstruction ceiling has median 27.3194 and mean 27.4001 dB versus home 28.5540; some arenas are 3.3052 dB below home. The reconstruction LPIPS remains below persistence on every arena, which supports rejecting a simple LPIPS floor explanation, but does not isolate a latent mechanism. Paired decoder changes establish decoder sensitivity of A, not a causal fraction of the off-map deficit: changing the decoder also changes the decoded reference and copy-last baseline in A. Replace “the decoder carries a share” with the direct paired observation.

2. **“No … distance predicts,” “neither … order,” and “S0 … predicts which arenas adapt fastest.”** There are 13 arenas from one WAD, one seed for the complete tuned battery, coarse and censored crossing observations, and exploratory diagnostic selection. The supplied correlations are descriptive evidence, not proof of no association or prospective predictive validation. Endpoint correlation (+0.8956) is not a speed statistic; budget correlation (−0.6065) is the relevant, weaker result. A0 enters the cost threshold, and S0 correlates with A0 (+0.7802); using S0 does not automatically remove coupling. Say “we did not establish an ordering” and “exploratory association.”

3. **“Unchanged when motion or persistence PSNR is partialled out.”** Signs and broad strength survive, but numbers change. Residualized-rank partials of S0 with tuned A4k / budget are **+0.93053 / −0.70671** controlling context motion, and **+0.89464 / −0.60202** controlling persistence PSNR. State “persists after adjustment,” not “unchanged.”

4. **“A leave-one-arena-out fit predicts no better than the mean.”** Outcome, transformation, and censoring treatment are unspecified. Independent OLS with intercept and training-fold mean baseline gives D MSE ratios **0.9261 for tuned A0**, **1.0868 for A4k**, and **0.9885 for cost with the 8k sentinel**. Thus the universal literal statement is false: there is a small A0 improvement, with no validated predictor established. Sentinel-valued cost OLS is only a sensitivity calculation; restricted cost or censored methods are preferable. `tools/distance_outcome_corr.py` itself performs **no** LOO fit or partial correlation; it uses stock A and hard-coded home 4.138. It cannot directly verify the new tuned claims.

5. **“Training maps’ held-out episodes score inside the unseen range on both [transition distances].”** Only **3/4** training-map coverage means and **1/4** transfer-gap means are inside those ranges. Coverage ranges: training **0.523888–0.608878**, unseen **0.526436–0.612669**; transfer-gap ranges: training **0.005416–0.012317**, unseen **0.008735–0.037802**. Conversely, **12/13** unseen coverage means and **2/13** transfer-gap means lie within the training range. The supported statement is **overlap / failure of complete separation**, not all training points inside. “Families” here also needs care: training and unseen arenas are from the same WAD; “training status” is more precise.

6. **Older distance evidence and the failed gate.** Training validation D values come from Dold; D has no validation-map D. Disclose this provenance instead of implying a fresh-set validation rescore. The −0.73 and −0.42 relate to the earlier **raw full-frame prediction-minus-persistence PSNR**, not the tuned A just discussed. The latter controls both training status and campaign membership in addition to persistence. The proposed paragraph deletes the current text’s explicit failed preregistered validation gate; retain that failure. A residual −0.42 does not prove the association is exclusively a family effect. The fresh D file was produced after initial adaptation scoring: distinguish freezing the **definition** from measuring distances before outcomes; file contents alone do not certify preregistration timing.

7. **“Property of the footage rather than of one backbone.”** Agreement between U-Net and PixArt is a useful replication, but they share SD 1 latent space, decoder, data, and recipe. It does not isolate footage causality or establish backbone independence. Tuned rho is +0.9560; +0.9505 is the stock result. State agreement between these two models only. SD 3.5 remains a **170k EMA, stock-decoder provisional** read, not a final 200k/tuned replication; no SD 3.5 number in this diff licenses a stronger claim.

8. **“Nearly data-free,” “one episode gives most of that,” and a ceiling “set by the arena, not by the budget, the rate or the data.”** Only four arenas have the stock data ladder and two have stock recipe/8k tests. One episode is substantial trajectory data. On map12, one versus sixteen episodes retains only about 75% of the measured stock gain; it does not establish the 13-arena tuned crossing or LPIPS counts with one episode. Finite experiments over these settings cannot establish an intrinsic arena ceiling or invariance to data/optimization. The 13 main curves all still improve from 2k to 4k.

9. **Seed and recipe qualifiers.** “0.02–0.06” is the absolute difference of **two training seeds on four arenas, stock decoder**, not an uncertainty interval for all tuned curves. There is no seed-1 read for map12. Stock 4k A changes for lr3e-4/lr5e-4/g8k are **+0.031299/+0.052821/+0.055321** on map12 and **+0.003748/+0.009875/+0.018455** on map16. These are small and below the maximum observed seed difference elsewhere, but not a matched map12 significance test. The g8k runs finish **−0.002951 / −0.010831 dB** below their own 4k values. “8k updates moves A at 4k” should distinguish a changed training schedule from the 8k endpoint. All these reads are stock-only.

10. **Means, ties, and weights.** The 69% number is the mean of per-arena gain fractions; +0.13 is the mean late increment, not the increment on each arena. “Tie or win on 11” is a sign-of-mean count (all 11 actually have negative M), not an equivalence or significance test. Zero-shot source weights are **EMA**, whereas adapter curves use **live** weights initialized from that EMA. The preamble’s blanket “live weights” should not be read as describing the zero-shot base model. Headline results use one adaptation seed; the limited stock seed check does not change that.

11. **Outstanding guards/comparator and surrounding text.** C carries null forgetting/directional guard values; turn preservation after adaptation and forgetting cannot be claimed. The abstract’s surviving turn response refers to the zero-shot model and relies on older directional evidence, not these fresh metric files. Full fine-tuning on every arena is a plan, and the retained Table 3 header still says “Full, arena 7.” The surrounding paper still says MSE-only decoder tuning is planned and repeats old 3.60→1.51 A values; applying this diff alone leaves inconsistencies with MSE+LPIPS and 5.06→2.04. Keep the promised full fine-tune, guard results, and final SD 3.5 rows visibly pending. The 1.1 GPU-hours covers adapter training only.

12. **Figure/data plumbing.** A `_tuned` directory contains both stock and tuned keys: use `_tuned` fields explicitly. T’s `zero_shot` entries for the fresh rows reproduce unsuffixed stock A despite the tuned table; its M/G budget fields are null. Do not use those entries for tuned M/G. The tuned ladder/recipe/seed summaries are empty or seed-0-only, corroborating the missing ablation rescores. The diff leaves figure paths under `figures/`, while the tuned outputs live under `figures/tuned/`; the suggested `fig2c_outcomes_by_skill.pdf` exists, but substituting the path alone does not establish which decoder its plotted values use. Check actual figure provenance and the requested left-panel crop before replacing captions. Home is pooled over 512 windows, whereas `main.tex` says equally over maps: equal-map home is 5.0570593 rather than pooled 5.0596903 (both round to 5.06, no reported crossing changes).

## At most five exact replacement sentences

1. Replace the opening sentence of §4’s decoder replacement:

   > The tuned decoder's scene reconstruction PSNR is 28.6~dB at home and 25.2--29.7~dB on unseen arenas (median 27.3), while the gap $G=\mathrm{PSNR}(D(z),x)-\mathrm{PSNR}(D(\hat z),x)$ is 3.4~dB at home and 3.7--6.9~dB unseen (median 5.0).

   Reason: makes the reference target explicit, corrects G, and removes the unsupported no-degradation inference.

2. Replace §8’s perceptual-gap sentence:

   > At 4k updates, the live adapters have lower mean scene LPIPS than raw persistence on 11 of 13 arenas ($M$ median $-0.02$, versus $+0.08$ zero-shot), and the median raw-reference gap $G$ falls from 5.0 to 3.4~dB (home 3.4).

   Reason: reports the supported sign count and correctly matched reconstruction/prediction scores.

3. Replace §5’s final sentence:

   > Across these 13 arenas, zero-shot latent skill $S_0$ has exploratory Spearman correlations of $+0.90$ with tuned $A$ at 4k and $-0.61$ with the observed crossing tiers, with noncrossers tied above all crossings; both associations persist after adjustment for motion or persistence PSNR, and the U-Net and PixArt zero-shot tuned advantages agree in rank ($+0.96$).

   Reason: corrects the decoder-specific number, preserves censoring, and avoids causal or prospectively validated prediction claims.

4. Replace the Figure 3 seed sentence:

   > In the stock-decoder check on arenas 6, 7, 8 and 16, the two training seeds differ in $A$ at 4k by 0.02--0.06~dB (\appref{app:perarena-adapt}).

   Reason: the complete tuned seed comparison does not exist.

5. Replace §8’s “gain is nearly data-free” sentence:

   > In stock-decoder checks, one episode finishes within about 0.25~dB of sixteen on arenas 7, 8, 12 and 16, while tripling or quintupling the learning rate or extending training to 8k changes $A$ by less than 0.06~dB on arenas 12 and 16; these limited checks do not establish a fixed adaptation ceiling.

   Reason: scopes the ladder/recipe evidence and removes the inference of budget/data independence.

The table and objections identify the remaining required edits; the five-sentence limit prevents supplying a replacement for every affected sentence.

## Unchecked or unavailable evidence

- Decoder training coefficient/step count and gate deltas: secondary context notes exist, but their decoder training logs, gate metrics, and referenced `results_spiderman/levers_2026-09-20/scene_crop/sd1.json` are absent locally. Do not substitute home reconstruction deltas for gate results. The planned appendix also needs explicit full-frame versus scene labels.
- Full-finetune rate attribution: a local literature memo says GameNGen used 2e-5, but no primary-paper verification was performed; no comparator run validates the planned use on every arena.
- Fresh directional survival, forgetting, completed full fine-tuning, final 200k SD 3.5/tuned SD 3.5: absent from the named score artifacts; no remote access attempted.
- Preregistration timing and nominal bootstrap coverage cannot be established by arithmetic on these files. The 95% bootstrap **procedure** is inspectable; statistical coverage is not verified.
- Raw videos/checkpoints were not regenerated. This is an independent arithmetic/provenance review of the saved metrics, curves, run certificates and available per-window records.
