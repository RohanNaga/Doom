# Figures and tables of `paper/main.tex` (and `paper/appendix.tex`)

Status as of 2026-09-27 (after the joint review; `REVIEW_LOG.md`). The submission build (`\withappendixfalse`) is body plus references; the appendix rows below build with `\withappendixtrue`. The quantities, in the paper's names and in the cost-target decision's (`.claude/analyses/cost-target-decision-2026-09-26.md`):

| Paper | Decision memo | Per window |
|---|---|---|
| decoded advantage `A` | A | PSNR(D(ẑ), D(z)) − PSNR(D(z_last), D(z)); `psnr_dec − copy_psnr_dec` in `eval_tf.py` |
| perceptual margin `M` | B | LPIPS(D(ẑ), x) − LPIPS(x_last, x); `lpips_raw − persist_lpips_raw` |
| gap to the ceiling `G` | C | PSNR(D(z), x) − PSNR(D(ẑ), x); `vae_psnr − psnr_raw` |
| latent skill `S` | latent skill | 10 · mean log10(copy-last latent MSE / model latent MSE) |

Status words: **exists** (the file or every number is in the repo today), **provisional** (numbers exist but will be replaced: fresh-set rescore, tuned SD 1 decoder, scene rows, final 200k checkpoints), **pending** (nothing exists yet).
Figures are drawn by `\figslot{file}{width}{height}{label}` in `main.tex`: when `paper/<file>` exists it is included, otherwise a labelled box stands in. Dropping the file into `paper/figures/` under the name below is all a figure needs.

## Body

| Label | What it shows | Produced by | File the slot expects | Status |
|---|---|---|---|---|
| `fig:strips` (Figure 1) | Closed-loop rollouts under held controls (turn left, forward, strafe right), a training arena beside an unseen arena; rows: recorded frames, U-Net 200k EMA rollout, copy-last; columns tics 1, 4, 8, 16, 32 | a strip mode of `tools/rollout_gif.py` or `tools/collapse_strip.py` on the U-Net 200k EMA; windows where the recorded control is held for the shown span, chosen by a seeded rule before looking at frames | `figures/fig1_rollout_strips.pdf` | pending (script and window rule to write; arenas to name) |
| `fig:arenas` (Figure 2) | The 17 arenas sorted by D, training maps shaded; (a) `A` per arena with 95% bootstrap intervals, the training maps' mean and half of it; (b) `M` per arena with zero. Shares one float row with Figure 3 (two-thirds of the width) | a new plot over the per-window rows of the fresh-set rescore and D on the fresh set; a provisional version can be drawn today from `results/distance_study/scores/040-unet-nexttic_snap_0200000_ema_ddim10/*_h1/metrics.json` and `results/distance_study/distances_sd1.json` | `figures/fig2a_advantage_by_distance.pdf`, `figures/fig2b_margin_by_distance.pdf` | provisional data exists; figure pending |
| `fig:adapt` (Figure 3) | Adaptation curves, one line per unseen arena coloured by D: `A` against LoRA updates (log axis with step 0 as the first labelled tick, grid 0 to 4,000) with the frozen training-map line and each arena's half-gap line (A_0 + home)/2 as a tick on its curve; first crossings marked, non-crossers open at the final budget, the full fine-tune on arena 7 in black. One panel, as Rohan specified; the `M` curves the cost decision called a second panel are Figure 5 in the appendix. Shares one float row with Figure 2 (one third of the width) | `score_adapt.py score` rows (`scores.jsonl`, live weights) from the 13-arena U-Net LoRA runs (`adapt_wm.py`), the home line and each arena's A_0 frozen from step-0 predictions through the tuned SD 1 decoder | `figures/fig3_adaptation_curves.pdf` | pending (runs launch Sep 27 evening; arenas 8, 16, 6, 7 first as the fallback set) |
| `tab:indomain` (Table 1) | Three rows in domain: one- and four-tic PSNR / LPIPS against persistence, 256-tic rollout PSNR against copy-seed, directional score | U-Net: dossier 5.1 and 4.5 (200k EMA; frozen reads). PixArt: `RESEARCH_CONTEXT.md` 2026-09-26 12:00 and 12:10 (155k TF, 150k rollout and directional). SD 3.5: 2026-09-26 17:10 (140k) | table in `main.tex` | U-Net exists (stock decoder); PixArt and SD 3.5 provisional until their 200k reads (PixArt Sep 26 night, SD 3.5 about Sep 28 00:30); tuned-decoder and scene-only columns pending (`scripts/spiderman/rescore_tuned_decoder.sh`) |
| `tab:unseen` (Table 2) | Training maps against unseen arenas per row: ceiling, `A` at one and four tics, `M`, `G`, directional; means with the count of maps that beat copy-last. The PixArt and SD 3.5 rows are one placeholder row each today and take the U-Net's two-row form when filled | U-Net: computed from the frozen `*_h{1,4}/metrics.json` (17 arenas); PixArt and SD 3.5: the fresh-set rescore | table in `main.tex` | U-Net provisional (pre-fresh-set episodes, stock decoder, full frame); PixArt and SD 3.5 pending; unseen directional pending (U-Net 30-map directional, 15 of 30 maps done) |
| `tab:cost` (Table 3) | What crossing costs: LoRA over 13 arenas (median), LoRA on arena 7, full fine-tune on arena 7; trained parameters and GPU-hours, arenas past the half-gap line and the home line by 4k, cost to each (first grid crossing), `A`, `M`, `G` at 4,000, forgetting and directional | `score_adapt.py cost` after implementing the per-arena half-gap line (A_0 + home)/2 (right-censored; a median over 13 arenas is censored when fewer than 7 cross, never a median over crossers alone) and the guard rows; parameter counts from `lora.parameter_counts` | table in `main.tex` | trained-parameter row exists; everything else pending (full fine-tune: Monday if a card idles and arena 7's LoRA curve is flat, else October) |

## Appendix

| Label | What it shows | Produced by | Status |
|---|---|---|---|
| `tab:recipe` | Architecture, latent space, parameters, control path, memory, throughput of the three rows | dossier section 3 table; `backbones.py` | exists |
| `tab:perarena` | Per arena (17): D, persistence, ceiling, `A` (1 and 4 tics), `M`, `G`, raw gain, directional | rows generated from the frozen `*_h{1,4}/metrics.json` and `distances_sd1.json` (primary entries), not typed | provisional (fresh-set rescore); directional pending except arena 6 (0.789) |
| `tab:checks` | The five pre-declared validation checks of D and their results | `results/distance_study/distances_sd1.json` checks; `figure_unet_h1_amended/stats.json` amended assessment | exists (on the 30-map set) |
| `tab:collapse` | SD 3.5 collapse events (a)/(b)/(c), live against EMA, same 16 windows, 55k to 130k, with and without 70k | recomputed from `results/sd35_stability/collapse_rates_50k_130k.md` | exists; PixArt 200k rows pending |
| `fig:collapse` | SD 3.5 70k live versus EMA rollout strip, two validation windows, tics 1 to 256 | `tools/collapse_strip.py` at `42a4688` | exists: `paper/figures/sd35_70k_live_vs_ema_rollout_strip.jpg` |
| `tab:perarena-adapt` | Per unseen arena: D, step-0 `A`, its half-gap line (A_0 + home)/2, first grid crossing of the half-gap line and the home line, `A` and `M` at 4,000, forgetting, directional | `score_adapt.py` rows and cost | D, step-0 `A` and the half-gap lines provisional; the rest pending |
| `fig:adapt-margin` (Figure 5) | `M` against LoRA updates, one line per unseen arena coloured by D, with zero and the training maps' margin, censored arenas open, full fine-tune on arena 7 in black | `score_adapt.py score` rows (live weights) | `figures/fig5_adaptation_margin.pdf`; pending with the 13-arena runs |

## Numbers in the text that are not in a table

| Where | Number | Source | Status |
|---|---|---|---|
| Introduction, Section 2 | persistence 19.4 to 23.2 dB on the unseen arenas; model PSNR tracks it at Spearman 0.95; raw gain tracks it at −0.77 | recomputed from the frozen h1 metrics (13 unseen arenas) | provisional |
| Section 2, Appendix B | HUD share of the ceiling's squared error per window, geometric mean over windows, 39 to 68 %; persistence HUD PSNR 88 to 96 dB | (32/240) × 10^((mean `vae_psnr` − mean `hud_vae_psnr`)/10) per arena, and `persist_hud_psnr_raw`, from the 17 h1 `metrics.json` | provisional |
| Section 3 | D ranges, within-arena Spearman of `A` on D −0.23, of raw gain −0.02 | recomputed from the frozen h1 metrics and `distances_sd1.json` | provisional |
| Section 3 | 30-map partial Spearman −0.73 [−0.84, −0.32] | `results/distance_study/figure_unet_h1/stats.json` | exists |
| Section 3 | collapse counts 23 / 256 against 9 / 256, 11 against 8 without 70k | `results/sd35_stability/collapse_rates_50k_130k.md` | exists |
| Appendix A (removed from Section 3 by review item F26) | sampler trade 10 to 50 steps: −0.25 dB, −0.024 LPIPS | dossier 4.2 (U-Net 200k live) | exists |
| Section 4 | cost line 3.60 dB; step-0 `A` 0.72 to 2.68 dB; 5 arenas already past half the line (9, 11, 13, 15, 16) | recomputed from the frozen h1 metrics | provisional; the tuned-decoder, scene-row line replaces it |
| Section 4 | 4.2M trained parameters, 0.49 % | `RESEARCH_CONTEXT.md` 2026-09-26 20:10 (`lora.parameter_counts`) | exists |
