# Truth check of `paper/main.tex` at `a07c415`

Reviewed locally on 2026-09-27; completed 16:29 UTC, within the 20-minute limit. No network, SSH, commits, builds, or edits to the draft. Only this review file was written. The pre-existing `.DS_Store` modification and untracked LaTeX outputs / two shell scripts were left alone.

**Coverage: 190 numerical audit entries: 179 CONFIRMED, 4 DISPUTED, 7 UNCHECKED; 183 checked.** Repeated mentions are grouped with their locations; a range or a fraction is one audit entry, not a count of individual digit tokens. Structural section/figure/table numbers, contribution-list indices, model version names (SD 1.4/3.5), citation years/keys, LaTeX layout lengths, subscripts, and `fp32` precision notation are not experimental quantities. Both numeric tables in the supplied body are checked; “Table 2” below means `tab:unseen` and “Table 3” means `tab:cost`, following the request's names rather than assuming the PDF's automatic numbering.

The important empirical results survive: **9/13 half-gap crossings, median 1,000 updates, median +1.7684 dB recovery, and 11/13 perceptual wins**. The one incorrect confidence endpoint is **5.39 → 5.38**. The error-percentage sentence and the absent map-2 teaser column also need correction. The training-row table caption, common-decoder wording, and unqualified “best one-tic numbers” claim need attention even though their underlying table numbers are correct.

## Calculation conventions

All computations ran as inline Python using `/opt/miniconda3/envs/PERSEVE/bin/python`. For each metrics file, `m(k)` is its stored window mean. For a tuned row, use the **field suffix** `_tuned`, not merely the directory name:

- `A = m(scene_psnr_dec_tuned) - m(scene_copy_psnr_dec_tuned)`.
- `M = m(scene_lpips_raw_tuned) - m(scene_persist_lpips_raw)`.
- `G = m(scene_vae_psnr_tuned) - m(scene_psnr_raw_tuned)`.
- SD 3.5 uses its own stock decoder and the same names without `_tuned`. Persistence is always `scene_persist_*`, never the decoded `scene_copy_*` raw fields.
- Unseen results are medians of the 13 per-arena values, including medians of paired differences. Training results are the **pooled means of 512 validation windows**, not medians of four map means.
- Adaptation uses exact base runs `..._r16_k8_s0/`, excluding the learning-rate and extended-grid variants matched by a loose `s0*` glob; select tuned evaluation records and deduplicate by step. All 78 tuned curve values (13 × 6 steps) agree exactly with the summary. Budgets were recomputed from the first crossing of `(A0 + 5.0592508260160685)/2`, with noncrossings right-censored after 4,000. Censored costs are tied above observed costs for Spearman (the summary uses 8,000, not observed crossings at 8,000).
- Confidence intervals were **looked up**, not newly bootstrapped. The training band is from `home_ci`; aggregate adaptation intervals are nested arena/episode bootstrap intervals with the home point held fixed. This is not an independent bootstrap replication.

The adaptation scorer has median A0 **2.036097892** (2.04), whereas the separate fresh rescore has **2.034840479** (2.03). Both printed roundings have a source; they are not an arithmetic contradiction. Similarly the frozen adaptation home reference is **5.059250826**, versus **5.059690252** in the fresh home rescore; both print 5.06. Keep those estimators/protocols named rather than silently merging them.

## Every numerical claim

`map??` denotes all 13 unseen-arena files; the model/run names identify the exact files in each wildcard. CONFIRMED means the printed precision agrees with the named source under the stated aggregation; qualifications in the value column are part of the verdict.

| Location | Quantity | Draft | Status | Recomputed value / qualification | File / field |
|---|---|---|---|---|---|
| Abstract; §§1–2,5; Fig. 2 | benchmark arenas | 17 | CONFIRMED | 4 training + 13 unseen = 17 | results/family_step/family_step.json; results/distance_v2/distances_sd1.json |
| Abstract; §§1–3; Fig. 2 | backbones | three | CONFIRMED | U-Net, PixArt, SD 3.5 (three rows) | results/family_step/family_step.json |
| Abstract; §§1–2; Fig. 2 | training maps | four; 2 to 5 | CONFIRMED | [2, 3, 4, 5] | results/fresh_rescore/home_unet200k_ema_tuned/val/metrics.json config; release/dense_split.json |
| Abstract; §§1–5; Figs. 3–4; both tables | unseen arenas / sample size | thirteen; 13; n=13; 1, 6 to 17 | CONFIRMED | [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] | results/distance_v2/distances_sd1.json maps; paper/tables/tuned/adapt_summary.json n_arenas |
| Abstract | in-distribution A | 5.1 dB | CONFIRMED | 5.059690252 -> 5.1 | results/fresh_rescore/home_unet200k_ema_tuned/val/metrics.json |
| Abstract | unseen A range | 1.1–2.9 dB | CONFIRMED | 1.068552922–2.878588539 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| Abstract | unseen median A | 2.0 dB | CONFIRMED | 2.034840479493141 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| Abstract; §3; Fig. 3 | U-Net positive decoded advantage / LPIPS loss count | every arena; 13 of 13 | CONFIRMED | 13/13 A>0; 13/13 M>0 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| Abstract; §§1,4; Fig. 2 | LoRA rank | 16 | CONFIRMED | rank=16 in all 13 base configurations | results/adapt/unet200k_arenas13_map??_r16_k8_s0/config.json; results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records) |
| Abstract; §4; Figs. 1,2,4; §5 | adaptation episodes per run | eight; 8 | CONFIRMED | 8 disjoint adaptation episodes in every base run | results/adapt/unet200k_arenas13_map??_r16_k8_s0/config.json certificate.adapt_episodes |
| Abstract; §§1,4; Fig. 4; Table 3 | half-gap crossings by final budget | 9 of 13 by 4k | CONFIRMED | 9/13 by 4000; censored maps 1,8,12,16 | results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records); paper/tables/tuned/adapt_summary.json crossings |
| Abstract | median per-arena recovered advantage | 1.8 dB | CONFIRMED | 1.768414169549942 | results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records); paper/tables/tuned/adapt_summary.json medians.gain |
| Abstract; §4; §5 | perceptual wins after adaptation | 11 of 13 | CONFIRMED | 11/13 M<0 | results/fresh_rescore/adapt4000_live_tuned/map??/metrics.json |
| §1; §4; Fig. 4; Table 3 | median first-crossing budget | 1k | CONFIRMED | 1000 with uncrossed maps ranked beyond 4000 | results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records); paper/tables/tuned/adapt_summary.json medians.cost_half_gap |
| §1; §3 | S0 versus final A Spearman | 0.90; +0.90 | CONFIRMED | 0.8956043956043955 | results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records); paper/tables/tuned/adapt_summary.json spearman.S0_vs_A_budget |
| §1; §3 | transition-level distances | two | CONFIRMED | coverage and transfer-gap G (not the reconstruction-gap G) | results/transition_distance/compare_fresh_sd1.csv |
| Fig. 1 caption | time offset between displayed context and outcome | 8 tics | CONFIRMED | 8 displayed-tic offset; frames at rollout tics 9 and 22 follow displayed true tics 1 and 14; these are continuous closed-loop rollouts, not restarts at the shown true frames | paper/figures/fig_teaser_B.json; tools/compose_teaser.py:365; results/teaser/index.json |
| Fig. 1 caption | unseen arena id | 7 | CONFIRMED | arena 7, held-out episode 210 | paper/figures/fig_teaser_B.json; results/teaser/unseen_arena07_ep210_s2888/manifest.json |
| Fig. 1 caption | adapter snapshot | 4k | CONFIRMED | adapter_0004000.pt | results/teaser/index.json |
| Fig. 1 caption | training-map reference column | one moment on map 2, same in both rows | DISPUTED | No training-map column (0); metadata home=null; PDF has context/zero-shot/after 8 episodes/true only | paper/figures/fig_teaser_B.json; paper/figures/fig_teaser_B.pdf |
| §2 data; Fig. 2 where repeated | bots per episode | 8 | CONFIRMED | 8 bots | scripts/spiderman/record_dense.sh:51; release/DATASET_CARD.md:19 |
| §2 data; Fig. 2 where repeated | episode timeout | 150 s | CONFIRMED | 150 game-seconds | scripts/spiderman/record_dense.sh:57; release/DATASET_CARD.md:32 |
| §2 data; Fig. 2 where repeated | recording rate | 35 Hz | CONFIRMED | 35 engine tics/s | release/DATASET_CARD.md:20 |
| §2 data; Fig. 2 where repeated | frame width | 320 | CONFIRMED | 320 pixels | record_arnold.py:414; release/DATASET_CARD.md:66 |
| §2 data; Fig. 2 where repeated | frame height | 240 | CONFIRMED | 240 pixels | record_arnold.py:414; release/DATASET_CARD.md:66 |
| §2 data; Fig. 2 where repeated | executed control bits | 19 | CONFIRMED | 19; do not confuse with the 29 action-id vocabulary | results/fresh_rescore/home_unet200k_ema_tuned/val/metrics.json config.checkpoint_interface.control_bits |
| §2 data; Fig. 2 where repeated | training episodes per map | 500 | CONFIRMED | 2000 training episode ids / 4 interleaved maps = 500 | release/dense_split.json next_tic_runs; record_arnold.py:323 |
| §2 data; Fig. 2 where repeated | training validation windows | 512 | CONFIRMED | 512 pooled windows; 99 episodes represented in home bootstrap | results/fresh_rescore/home_unet200k_ema_tuned/val/metrics.json; paper/tables/tuned/adapt_summary.json home_ci |
| §2 data; Fig. 2 where repeated | new episodes per unseen arena | 24 | CONFIRMED | 24 on each of 13 maps | results/distance_v2/distances_sd1.json maps[].n_episodes |
| §2 data; Fig. 2 where repeated | available adaptation episodes | 16 | CONFIRMED | 24 total minus 8 held out = 16 available; the headline uses only 8 of these | results/distance_v2/distances_sd1.json; results/adapt/unet200k_arenas13_map??_r16_k8_s0/config.json |
| §2 data; Fig. 2 where repeated | held-out episodes per arena | 8 | CONFIRMED | 8 in every base run | results/adapt/unet200k_arenas13_map??_r16_k8_s0/config.json certificate.held_out |
| §2 data; Fig. 2 where repeated | held-out windows per arena | 256 | CONFIRMED | 256 in every base run | results/adapt/unet200k_arenas13_map??_r16_k8_s0/config.json certificate.held_out_windows; results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| §2 recipe | U-Net parameters | 860M | CONFIRMED | 860,532,932 before LoRA = 863,721,668 total minus 3,188,736 LoRA; approximately 860M | results/adapt/unet200k_arenas13_map??_r16_k8_s0/config.json certificate.counts |
| §2 recipe | PixArt parameters | 628M | UNCHECKED | Only repeated in paper/appendix.tex; no independent local count located | paper/appendix.tex:28; supplied result files do not store a parameter inventory |
| §2 recipe | SD 3.5 parameters | 2.27B | UNCHECKED | Only repeated as 2,271.7M in appendix; no independent local count located | paper/appendix.tex:28; supplied result files do not store a parameter inventory |
| §2 recipe | SD 3.5 latent channels | 16 | CONFIRMED | 16 | results/fresh_rescore/home_sd35_170000/val/metrics.json config.resolved_latent_channels |
| §2 recipe; §4 initialization | training updates / source checkpoint | 200k | CONFIRMED | 200000 achieved for U-Net and PixArt; shared target only for SD 3.5 (latest supplied read 170000) | results/training_curves/{unet,pixart,sd35}.json; results/adapt/unet200k_arenas13_map??_r16_k8_s0/config.json certificate.source.step |
| §2 recipe | global batch | 32 | CONFIRMED | 32 | scripts/spiderman/launch_nexttic.sh recipe / GLOBAL=32; results/adapt/unet200k_arenas13_map??_r16_k8_s0/config.json |
| §2 recipe | EMA decay | 0.9999 | CONFIRMED | 0.9999 for base training (adapter EMA is separately 0.999) | scripts/spiderman/launch_nexttic.sh; train_wm.py |
| §2 recipe | DDIM inference steps | 10 | CONFIRMED | 10; eta=0; evaluator invokes DDIM | results/fresh_rescore/home_unet200k_ema_tuned/val/metrics.json config.steps/eta; eval_tf.py |
| §2 four-tic sentence | evaluation horizon | four tics | CONFIRMED | horizon 4 is the documented separate full-frame stock-decoder evaluation | RESEARCH_CONTEXT.md §0 item 5; scripts/spiderman/launch_nexttic.sh periodic reads |
| §2 four-tic sentence | U-Net four-tic PSNR | 21.33 dB | UNCHECKED | Matches the research log, but no 512-window h4 metrics file exists locally; training_curves files are h1 only | RESEARCH_CONTEXT.md §0 item 5 and Sep 27 01:20 / Sep 26 17:10 entries; results/training_curves/*.json |
| §2 four-tic sentence | PixArt four-tic PSNR | 21.29 dB | UNCHECKED | Matches the research log, but no 512-window h4 metrics file exists locally; training_curves files are h1 only | RESEARCH_CONTEXT.md §0 item 5 and Sep 27 01:20 / Sep 26 17:10 entries; results/training_curves/*.json |
| §2 four-tic sentence | SD 3.5 four-tic PSNR | 21.59 dB | UNCHECKED | Matches the research log, but no 512-window h4 metrics file exists locally; training_curves files are h1 only | RESEARCH_CONTEXT.md §0 item 5 and Sep 27 01:20 / Sep 26 17:10 entries; results/training_curves/*.json |
| §2 four-tic sentence | persistence four-tic PSNR | 19.21 dB | UNCHECKED | Matches the research log, but no 512-window h4 metrics file exists locally; training_curves files are h1 only | RESEARCH_CONTEXT.md §0 item 5 and Sep 27 01:20 / Sep 26 17:10 entries; results/training_curves/*.json |
| §2 four-tic sentence | provisional SD 3.5 checkpoint | 140k | CONFIRMED | 140000 checkpoint exists; its four-tic metric remains unverified | results/training_curves/sd35.json; RESEARCH_CONTEXT.md Sep 26 17:10 |
| §2 decoder recipe | LPIPS objective coefficient | 0.1 | CONFIRMED | --lpips-weight 0.1 | scripts/spiderman/decoder_mse_lpips.sh:26 |
| §2 decoder claim | MSE-only scene LPIPS multiplier | triples (3x) | UNCHECKED | No primary MSE-only decoder validation file located; research log instead gives 0.339/0.092 = 3.685x stock | RESEARCH_CONTEXT.md Sep 27 05:20; no such values in the supplied fresh-rescore files |
| §2 metric definitions | logarithm coefficient/base and zero threshold | 10 log10; M below zero | CONFIRMED | Definitionally consistent with PSNR and LPIPS margin; zero is the persistence tie | paper/main.tex:112–114; eval_tf.py; results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records) |
| §2; Fig. 2 | scene crop endpoints | 0 to 207 | CONFIRMED | 0–207 inclusive, 208 rows | eval_tf.py:111 scene_crop |
| §2 training reference | A_train point estimate | 5.06 dB | CONFIRMED | 5.0592508260160685 | paper/tables/tuned/adapt_summary.json home; results/home_unet_tuned/metrics.json |
| §2 training reference | episode CI lower bound | 4.75 | CONFIRMED | 4.7538025615763955 | paper/tables/tuned/adapt_summary.json home_ci; paper/tables/tuned/adapt_table3.tex |
| §2 training reference | episode CI upper bound | 5.39 | DISPUTED | 5.379118152883 -> 5.38 | paper/tables/tuned/adapt_summary.json home_ci; paper/tables/tuned/adapt_table3.tex |
| Table 2 caption (tab:unseen) | training true directional reference | 0.885 | CONFIRMED | 0.884765625 | results/directional_fresh/summary.json per_map.ref_raw_frac |
| Table 2 caption (tab:unseen) | unseen true directional reference | 0.892 | CONFIRMED | 0.891826923 | results/directional_fresh/summary.json per_map.ref_raw_frac |
| Table 2 caption; Fig. 3 | SD 3.5 snapshot | 170k | CONFIRMED | 170000 EMA; own stock decoder | results/fresh_rescore/home_sd35_170000/val/metrics.json config |
| Table 2 caption; Fig. 3; §5 | scoring horizon | one tic | CONFIRMED | horizon_tics=1 | results/fresh_rescore/home_unet200k_ema_tuned/val/metrics.json config; results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| Table 2 persistence training | PSNR | 20.95 | CONFIRMED | 20.953114351 | results/fresh_rescore/home_unet200k_ema_tuned/val/metrics.json |
| Table 2 persistence training | LPIPS | 0.239 | CONFIRMED | 0.239144956 | results/fresh_rescore/home_unet200k_ema_tuned/val/metrics.json |
| Table 2 persistence training | A | 0 | CONFIRMED | 0 by baseline definition (directional also explicitly 0 per map) | Definitions in paper/main.tex:112–114 |
| Table 2 persistence training | M | 0 | CONFIRMED | 0 by baseline definition (directional also explicitly 0 per map) | Definitions in paper/main.tex:112–114 |
| Table 2 persistence training | Directional | 0 | CONFIRMED | 0 by baseline definition (directional also explicitly 0 per map) | results/directional_fresh/summary.json |
| Table 2 persistence unseen | PSNR | 19.75 | CONFIRMED | 19.751068685 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| Table 2 persistence unseen | LPIPS | 0.222 | CONFIRMED | 0.222331315 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| Table 2 persistence unseen | A | 0 | CONFIRMED | 0 by baseline definition (directional also explicitly 0 per map) | Definitions in paper/main.tex:112–114 |
| Table 2 persistence unseen | M | 0 | CONFIRMED | 0 by baseline definition (directional also explicitly 0 per map) | Definitions in paper/main.tex:112–114 |
| Table 2 persistence unseen | Directional | 0 | CONFIRMED | 0 by baseline definition (directional also explicitly 0 per map) | results/directional_fresh/summary.json |
| Table 2 unet200k_ema_tuned training | PSNR | 25.20 | CONFIRMED | 25.196385307 | results/fresh_rescore/home_unet200k_ema_tuned/val/metrics.json |
| Table 2 unet200k_ema_tuned training | LPIPS | 0.158 | CONFIRMED | 0.158045756 | results/fresh_rescore/home_unet200k_ema_tuned/val/metrics.json |
| Table 2 unet200k_ema_tuned training | A | +5.06 | CONFIRMED | 5.059690252 | results/fresh_rescore/home_unet200k_ema_tuned/val/metrics.json |
| Table 2 unet200k_ema_tuned training | M | -0.081 | CONFIRMED | -0.081099200 | results/fresh_rescore/home_unet200k_ema_tuned/val/metrics.json |
| Table 2 unet200k_ema_tuned training | G | 3.36 | CONFIRMED | 3.357585488 | results/fresh_rescore/home_unet200k_ema_tuned/val/metrics.json |
| Table 2 unet200k_ema_tuned training | Directional | 0.855 | CONFIRMED | 0.855468750 | results/directional_fresh/summary.json |
| Table 2 unet200k_ema_tuned unseen | PSNR | 22.30 | CONFIRMED | 22.297485141 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| Table 2 unet200k_ema_tuned unseen | LPIPS | 0.303 | CONFIRMED | 0.302884985 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| Table 2 unet200k_ema_tuned unseen | A | +2.03 | CONFIRMED | 2.034840479 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| Table 2 unet200k_ema_tuned unseen | M | +0.081 | CONFIRMED | 0.080553671 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| Table 2 unet200k_ema_tuned unseen | G | 4.96 | CONFIRMED | 4.956499830 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| Table 2 unet200k_ema_tuned unseen | Directional | 0.804 | CONFIRMED | 0.804086538 | results/directional_fresh/summary.json |
| Table 2 unet200k_ema_tuned unseen | arenas with A>0 | 13/13 | CONFIRMED | 13/13 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| Table 2 unet200k_ema_tuned unseen | arenas with M<0 | 0/13 | CONFIRMED | 0/13 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| Table 2 pixart200k_ema_tuned training | PSNR | 25.18 | CONFIRMED | 25.183143778 | results/fresh_rescore/home_pixart200k_ema_tuned/val/metrics.json |
| Table 2 pixart200k_ema_tuned training | LPIPS | 0.159 | CONFIRMED | 0.158623156 | results/fresh_rescore/home_pixart200k_ema_tuned/val/metrics.json |
| Table 2 pixart200k_ema_tuned training | A | +5.05 | CONFIRMED | 5.049453914 | results/fresh_rescore/home_pixart200k_ema_tuned/val/metrics.json |
| Table 2 pixart200k_ema_tuned training | M | -0.081 | CONFIRMED | -0.080521800 | results/fresh_rescore/home_pixart200k_ema_tuned/val/metrics.json |
| Table 2 pixart200k_ema_tuned training | G | 3.37 | CONFIRMED | 3.370827017 | results/fresh_rescore/home_pixart200k_ema_tuned/val/metrics.json |
| Table 2 pixart200k_ema_tuned training | Directional | 0.842 | CONFIRMED | 0.841796875 | results/directional_fresh/summary.json |
| Table 2 pixart200k_ema_tuned unseen | PSNR | 22.38 | CONFIRMED | 22.378942885 | results/fresh_rescore/pixart200k_ema_tuned/map??/metrics.json |
| Table 2 pixart200k_ema_tuned unseen | LPIPS | 0.285 | CONFIRMED | 0.285219568 | results/fresh_rescore/pixart200k_ema_tuned/map??/metrics.json |
| Table 2 pixart200k_ema_tuned unseen | A | +2.19 | CONFIRMED | 2.192066729 | results/fresh_rescore/pixart200k_ema_tuned/map??/metrics.json |
| Table 2 pixart200k_ema_tuned unseen | M | +0.062 | CONFIRMED | 0.061665468 | results/fresh_rescore/pixart200k_ema_tuned/map??/metrics.json |
| Table 2 pixart200k_ema_tuned unseen | G | 4.71 | CONFIRMED | 4.712048866 | results/fresh_rescore/pixart200k_ema_tuned/map??/metrics.json |
| Table 2 pixart200k_ema_tuned unseen | Directional | 0.808 | CONFIRMED | 0.808293269 | results/directional_fresh/summary.json |
| Table 2 pixart200k_ema_tuned unseen | arenas with A>0 | 13/13 | CONFIRMED | 13/13 | results/fresh_rescore/pixart200k_ema_tuned/map??/metrics.json |
| Table 2 pixart200k_ema_tuned unseen | arenas with M<0 | 0/13 | CONFIRMED | 0/13 | results/fresh_rescore/pixart200k_ema_tuned/map??/metrics.json |
| Table 2 sd35_170000 training | PSNR | 24.50 | CONFIRMED | 24.502916111 | results/fresh_rescore/home_sd35_170000/val/metrics.json |
| Table 2 sd35_170000 training | LPIPS | 0.146 | CONFIRMED | 0.145535713 | results/fresh_rescore/home_sd35_170000/val/metrics.json |
| Table 2 sd35_170000 training | A | +3.89 | CONFIRMED | 3.887549389 | results/fresh_rescore/home_sd35_170000/val/metrics.json |
| Table 2 sd35_170000 training | M | -0.094 | CONFIRMED | -0.093609242 | results/fresh_rescore/home_sd35_170000/val/metrics.json |
| Table 2 sd35_170000 training | G | 5.48 | CONFIRMED | 5.482597010 | results/fresh_rescore/home_sd35_170000/val/metrics.json |
| Table 2 sd35_170000 training | Directional | 0.850 | CONFIRMED | 0.849609375 | results/directional_fresh/summary.json |
| Table 2 sd35_170000 unseen | PSNR | 21.55 | CONFIRMED | 21.551879948 | results/fresh_rescore/sd35_170000/map??/metrics.json |
| Table 2 sd35_170000 unseen | LPIPS | 0.280 | CONFIRMED | 0.280411766 | results/fresh_rescore/sd35_170000/map??/metrics.json |
| Table 2 sd35_170000 unseen | A | +1.50 | CONFIRMED | 1.498598244 | results/fresh_rescore/sd35_170000/map??/metrics.json |
| Table 2 sd35_170000 unseen | M | +0.059 | CONFIRMED | 0.058770980 | results/fresh_rescore/sd35_170000/map??/metrics.json |
| Table 2 sd35_170000 unseen | G | 6.98 | CONFIRMED | 6.978646930 | results/fresh_rescore/sd35_170000/map??/metrics.json |
| Table 2 sd35_170000 unseen | Directional | 0.793 | CONFIRMED | 0.792668269 | results/directional_fresh/summary.json |
| Table 2 sd35_170000 unseen | arenas with A>0 | 13/13 | CONFIRMED | 13/13 | results/fresh_rescore/sd35_170000/map??/metrics.json |
| Table 2 sd35_170000 unseen | arenas with M<0 | 0/13 | CONFIRMED | 0/13 | results/fresh_rescore/sd35_170000/map??/metrics.json |
| §3 directional | U-Net arenas13 per-map range | 0.73–0.86 | CONFIRMED | 0.726562500–0.859375000 | results/directional_fresh/summary.json |
| §3 directional | U-Net val per-map range | 0.79–0.91 | CONFIRMED | 0.789062500–0.914062500 | results/directional_fresh/summary.json |
| §3 directional | true-frame directional score | 0.89 | CONFIRMED | 0.891826923 unseen raw; 0.884765625 training raw (0.89 refers to unseen) | results/directional_fresh/summary.json |
| §3 deficit | U-Net A training / unseen median | 5.06 to 2.03 | CONFIRMED | 5.059690252 / 2.034840479 | results/fresh_rescore/home_unet200k_ema_tuned/val/metrics.json; results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| §3 deficit | unseen reconstruction PSNR range | 25.2–29.7 | CONFIRMED | 25.248747557–29.662555285 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| §3 deficit | training reconstruction PSNR | 28.6 | CONFIRMED | 28.55397079512477 | results/fresh_rescore/home_unet200k_ema_tuned/val/metrics.json |
| §3 deficit; §4 | training reconstruction gap G | 3.4 | CONFIRMED | 3.3575854878872633 | results/fresh_rescore/home_unet200k_ema_tuned/val/metrics.json |
| §3 deficit; §4 | unseen zero-shot median G | 5.0 | CONFIRMED | 4.956499829888344 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| §3 decoder gains | unet200k_ema_tuned median paired change in A | 0.2 dB | CONFIRMED | 0.222126469; paired mean 0.220798113 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json and unet200k_ema twins |
| §3 decoder gains | adapt4000_live_tuned median paired change in A | 0.7 dB | CONFIRMED | 0.660997532; paired mean 0.705498321 | results/fresh_rescore/adapt4000_live_tuned/map??/metrics.json and adapt4000_live twins |
| §3 perceptual margin | reconstruction LPIPS range | 0.05–0.09 | CONFIRMED | 0.053377704–0.090877407 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| §3 perceptual margin | persistence LPIPS range | 0.18–0.28 | CONFIRMED | 0.181516778–0.275220065 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| §3 perceptual margin | prediction margin M range | +0.03–+0.17 | CONFIRMED | 0.026294222–0.172870874 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| §3 perceptual margin | median unseen M | +0.08 | CONFIRMED | 0.08055367062297591 | results/fresh_rescore/unet200k_ema_tuned/map??/metrics.json |
| §3 perceptual margin | training M | -0.08 | CONFIRMED | -0.0810991997796009 | results/fresh_rescore/home_unet200k_ema_tuned/val/metrics.json |
| Fig. 3(a) | unet200k_ema d–S0 Spearman | -0.05 | CONFIRMED | -0.049450549450549455 | results/family_step/family_step.json |
| Fig. 3(a) | pixart200k_ema d–S0 Spearman | -0.15 | CONFIRMED | -0.14835164835164835 | results/family_step/family_step.json |
| Fig. 3(a) | sd35_170000 d–S0 Spearman | -0.17 | CONFIRMED | -0.17032967032967034 | results/family_step/family_step.json |
| Fig. 3(a) | all backbones strictly separate training/unseen skill | all three | CONFIRMED | All 3 hold; min-training minus max-unseen skill: 0.590017, 0.235212, 0.632370 dB | results/family_step/family_step.json backbones.*.step |
| Fig. 3(b) | raw scene PSNR wins | all 13 for every model | CONFIRMED | 13/13 each; minimum gains U-Net +1.356340, PixArt +1.470609, SD 3.5 +0.349638 dB | results/fresh_rescore/{unet200k_ema_tuned,pixart200k_ema_tuned,sd35_170000}/map??/metrics.json |
| Fig. 3(b) | LPIPS losses | 13/13 for every model | CONFIRMED | 13/13 each; all scene M>0 | results/fresh_rescore/{unet200k_ema_tuned,pixart200k_ema_tuned,sd35_170000}/map??/metrics.json |
| Fig. 3(b) | episode-bootstrap interval level | 95% | CONFIRMED | 2.5th–97.5th percentile of episode-resampled pooled window means (lookup, not rerun) | paper/tables/tuned/adapt_summary.json zero_shot_absolute; paper/make_adapt_figures.py:346 |
| §3 distance definition | Wasserstein order | 2 | CONFIRMED | p=2 | results/distance_v2/distances_sd1.json config.p |
| §3 distance | training-held-out d range | 0.03–0.07 | CONFIRMED | 0.026038679–0.069316362 | results/family_step/family_step.json D_training |
| §3 distance | train-versus-train d floor range | 0.02–0.09 | CONFIRMED | 0.022325066–0.093373498 | results/distance_v2/distances_sd1.json floor.motion.min/max |
| §3 distance | unseen d range | 0.12–0.27 | CONFIRMED | 0.117886855–0.270253779 | results/distance_v2/distances_sd1.json maps[].D |
| §3 distance | D_vs_A0 Spearman | -0.22 | CONFIRMED | -0.21978021978021978 | paper/tables/tuned/adapt_summary.json spearman; results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records) |
| §3 distance | D_vs_cost Spearman | -0.35 | CONFIRMED | -0.3542796999226228 | paper/tables/tuned/adapt_summary.json spearman; results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records) |
| §3 distance | S0_vs_cost Spearman | -0.61 | CONFIRMED | -0.6065268462675303 | paper/tables/tuned/adapt_summary.json spearman; results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records) |
| §3 distance | p-value bound for the two named d correlations | none below p=0.2 | CONFIRMED | p=0.470614851 (A0), 0.234959255 (budget); applies to those two tests, not all possible outcomes | paper/tables/tuned/adapt_summary.json spearman |
| §3 distance | earlier failed study size | 30 maps | CONFIRMED | 30; verdict does not support; distance_validation=false | results/distance_study/figure_unet_h1/stats.json |
| §3 skill | S0 versus per-arena gain Spearman | +0.18 | CONFIRMED | 0.181318681 (p=0.553295009) | paper/tables/tuned/adapt_summary.json across_arenas.spearman_S0_gain; results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records) |
| §3 closed loop | matched rollout windows | 16 | CONFIRMED | 16 per checkpoint | results/sd35_stability/collapse_rates_50k_130k.md (recomputed from rows; excluded extra seeds) |
| §3 closed loop | matched checkpoints | 16 | CONFIRMED | 16, every 5k from 55k through 130k | results/sd35_stability/collapse_rates_50k_130k.md (recomputed from rows; excluded extra seeds) |
| §3 closed loop | first checkpoint | 55k | CONFIRMED | 55000 | results/sd35_stability/collapse_rates_50k_130k.md (recomputed from rows; excluded extra seeds) |
| §3 closed loop | last checkpoint | 130k | CONFIRMED | 130000 | results/sd35_stability/collapse_rates_50k_130k.md (recomputed from rows; excluded extra seeds) |
| §3 closed loop | collapse threshold | frame under 10 dB | CONFIRMED | any frame raw PSNR <10 dB | results/sd35_stability/collapse_rates_50k_130k.md (recomputed from rows; excluded extra seeds) |
| §3 closed loop | non-EMA collapses | 23 of 256 | CONFIRMED | 23/256 seed-0 matched rollouts | results/sd35_stability/collapse_rates_50k_130k.md (recomputed from rows; excluded extra seeds) |
| §3 closed loop | EMA collapses | 9 of 256 | CONFIRMED | 9/256 seed-0 matched rollouts | results/sd35_stability/collapse_rates_50k_130k.md (recomputed from rows; excluded extra seeds) |
| §3 closed loop | non-EMA excluding 70k | 11 | CONFIRMED | 11/240 | results/sd35_stability/collapse_rates_50k_130k.md (recomputed from rows; excluded extra seeds) |
| §3 closed loop | EMA excluding 70k | 8 | CONFIRMED | 8/240 | results/sd35_stability/collapse_rates_50k_130k.md (recomputed from rows; excluded extra seeds) |
| §3 closed loop | excluded checkpoint | 70k | CONFIRMED | 70000; removes 12 live and 1 EMA event | results/sd35_stability/collapse_rates_50k_130k.md (recomputed from rows; excluded extra seeds) |
| §4 protocol | trainable parameters | 4.2M | CONFIRMED | 4,212,544 | results/adapt/unet200k_arenas13_map??_r16_k8_s0/config.json certificate.counts.trainable |
| §4 protocol | trainable fraction | 0.49 percent | CONFIRMED | 0.489527343 percent | results/adapt/unet200k_arenas13_map??_r16_k8_s0/config.json certificate.counts.trainable_fraction |
| §4 protocol; repeated budgets throughout | evaluated grid point | 0 | CONFIRMED | 0 | results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records) grid and step (live weights) |
| §4 protocol; repeated budgets throughout | evaluated grid point | 250 | CONFIRMED | 250 | results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records) grid and step (live weights) |
| §4 protocol; repeated budgets throughout | evaluated grid point | 500 | CONFIRMED | 500 | results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records) grid and step (live weights) |
| §4 protocol; repeated budgets throughout | evaluated grid point | 1k | CONFIRMED | 1000 | results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records) grid and step (live weights) |
| §4 protocol; repeated budgets throughout | evaluated grid point | 2k | CONFIRMED | 2000 | results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records) grid and step (live weights) |
| §4 protocol; repeated budgets throughout | evaluated grid point | 4k | CONFIRMED | 4000 | results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records) grid and step (live weights) |
| §4 budget definition; abstract / §1 / captions | half-gap target | half; 1/2 | CONFIRMED | first grid crossing of (A0 + 5.059250826)/2; four uncrossed maps retained as right-censored | results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records) recomputed; paper/tables/tuned/adapt_summary.json |
| §4 error interpretation | example increase in A | 1.8 dB | CONFIRMED | 10^(-1.8/10)=0.660693448 model-error multiplier | Arithmetic from the definition of A in paper/main.tex:112 |
| §4 error interpretation | fraction of persistence error removed | 34 percent | DISPUTED | 33.930655% of starting MODEL geometric-mean MSE; at A0=2.04, only 21.212519% of persistence geometric-mean MSE | A definition; paper/tables/tuned/adapt_summary.json medians.A0; 10^(-A0/10)*(1-10^(-1.8/10)) |
| §4 error interpretation | alternative percentage | not 28 | DISPUTED | No defined denominator/calculation supports 28%; remove this comparison | paper/main.tex:182; A definition; paper/tables/tuned/adapt_summary.json |
| §4 outcomes | half-gap crossings within 500 | 5 | CONFIRMED | 5 (4 by 250, fifth by 500) | results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records); paper/tables/tuned/adapt_summary.json crossings.half_gap_by_step |
| §4 outcomes; Table 3 | arenas reaching A_train | one; 1 of 13 | CONFIRMED | 1/13, arena 9 | results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records); paper/tables/tuned/adapt_summary.json crossings.home_crossed_arenas |
| §4 outcomes; Fig. 4 | median adaptation-run A0 | 2.04 | CONFIRMED | 2.036097891628742 | results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records); paper/tables/tuned/adapt_summary.json medians.A0 |
| §4 outcomes; Fig. 4; Table 3 | median adaptation-run A4000 | 3.80 | CONFIRMED | 3.799065340310335 | results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records); paper/tables/tuned/adapt_summary.json medians.A_budget |
| §4 outcomes; Fig. 4 | gain share by first nonzero step | 69 percent by 250 | CONFIRMED | 69.4922599%; arithmetic MEAN of the 13 per-arena (A250-A0)/(A4000-A0) ratios | results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records); paper/tables/tuned/adapt_summary.json gain_share_by_first_step |
| §4 outcomes | remaining LPIPS-loss arenas | two | CONFIRMED | 2 (maps 1 and 6) | results/fresh_rescore/adapt4000_live_tuned/map??/metrics.json |
| §4 outcomes | remaining LPIPS margin tolerance | within 0.01 | CONFIRMED | positive margins at most 0.007671148 | results/fresh_rescore/adapt4000_live_tuned/map??/metrics.json |
| §4 outcomes | median adapted G, one decimal | 3.4 | CONFIRMED | 3.4418959580361843 | results/fresh_rescore/adapt4000_live_tuned/map??/metrics.json |
| Table 3 caption | arena-bootstrap confidence level | 95% | CONFIRMED | 95% nested arena-then-episode percentile bootstrap for A and budget; lookup in generated summary | paper/tables/tuned/adapt_summary.json across_arenas.method; paper/tables/tuned/adapt_table3.tex |
| Table 3 aggregate | half-gap count interval lower | 6 | CONFIRMED | 6 | paper/tables/tuned/adapt_summary.json across_arenas; paper/tables/tuned/adapt_table3.tex |
| Table 3 aggregate | half-gap count interval upper | 12 | CONFIRMED | 12 | paper/tables/tuned/adapt_summary.json across_arenas; paper/tables/tuned/adapt_table3.tex |
| Table 3 aggregate | median budget interval lower | 250 | CONFIRMED | 250 | paper/tables/tuned/adapt_summary.json across_arenas; paper/tables/tuned/adapt_table3.tex |
| Table 3 aggregate | median budget interval upper | >4k | CONFIRMED | right-censored beyond 4000; JSON upper endpoint null | paper/tables/tuned/adapt_summary.json across_arenas; paper/tables/tuned/adapt_table3.tex |
| Table 3 aggregate | median A4000 interval lower | 3.45 | CONFIRMED | 3.44881997089833 | paper/tables/tuned/adapt_summary.json across_arenas; paper/tables/tuned/adapt_table3.tex |
| Table 3 aggregate | median A4000 interval upper | 4.39 | CONFIRMED | 4.394810615852475 | paper/tables/tuned/adapt_summary.json across_arenas; paper/tables/tuned/adapt_table3.tex |
| Table 3 aggregate | budget to training-map line | >4k | CONFIRMED | median censored beyond 4000; only arena 9 reaches it | paper/tables/tuned/adapt_summary.json medians.cost_home; results/adapt/unet200k_arenas13_map??_r16_k8_s0/scores.jsonl (exact base runs, tuned evaluation records) |
| Table 3 aggregate | median M4000 | -0.021 | CONFIRMED | -0.021496872011994128 | results/fresh_rescore/adapt4000_live_tuned/map??/metrics.json; paper/tables/tuned/adapt_summary.json |
| Table 3 aggregate | median G4000 | 3.44 | CONFIRMED | 3.4418959580361843 | results/fresh_rescore/adapt4000_live_tuned/map??/metrics.json |
| Table 3 arena 7 | arena id | 7 | CONFIRMED | 7 | paper/tables/tuned/adapt_summary.json per_arena[arena=7]; paper/tables/tuned/adapt_table3.tex |
| Table 3 arena 7 | half-gap budget | 250 | CONFIRMED | 250 | paper/tables/tuned/adapt_summary.json per_arena[arena=7]; paper/tables/tuned/adapt_table3.tex |
| Table 3 arena 7 | budget to training-map line | >4k | CONFIRMED | no crossing by 4000 | paper/tables/tuned/adapt_summary.json per_arena[arena=7]; paper/tables/tuned/adapt_table3.tex |
| Table 3 arena 7 | A4000 | 3.86 | CONFIRMED | 3.8639473915100098 | paper/tables/tuned/adapt_summary.json per_arena[arena=7]; paper/tables/tuned/adapt_table3.tex |
| Table 3 arena 7 | M4000 | -0.026 | CONFIRMED | -0.026093831402249634 | paper/tables/tuned/adapt_summary.json per_arena[arena=7]; paper/tables/tuned/adapt_table3.tex |
| Table 3 arena 7 | G4000 | 4.65 | CONFIRMED | 4.649473611265421 | results/fresh_rescore/adapt4000_live_tuned/map07/metrics.json |
| §5 limitations | WAD / recording agent | one WAD; one agent | CONFIRMED | full_deathmatch / Arnold | scripts/spiderman/record_dense.sh; record_arnold.py |
| §5 limitations | seed coverage | mostly one seed | CONFIRMED | all 13 headline base curves use seed 0; additional seed-1 runs exist for maps 6,7,8,16 | results/adapt/unet200k_arenas13_map??_r16_k8_s0/config.json; results/adapt/unet200k_arenas13_map{06,07,08,16}_r16_k8_s1/config.json |
| §5 conclusion | adapter compute | 1.1 GPU-hours | CONFIRMED | PER ARENA: median 1.102961 GPU-h (range 1.091527–1.125966), 1 GPU each; all 13 total 14.341123 GPU-h | results/adapt/unet200k_arenas13_map??_r16_k8_s0/log.jsonl start/end and world=1 |

## Claims the files do not support, or support only with a qualification

1. **The Table 2 caption says “medians” for both sets.** Its training rows actually match pooled 512-window means. For example U-Net training PSNR is 25.196385 (25.20), whereas the median of its four per-map PSNR means is 24.867707 (24.87); PixArt's corresponding map median is 24.846609 and SD 3.5's is 24.049716. Preserve the current numbers and correct the caption, per the requested training-reference convention. Source: the corresponding `home_<row>/val/per_window.csv` files.
2. **The 34% claim uses the wrong error denominator; “not 28” has no defined calculation.** With fixed persistence, increasing A by 1.8 dB multiplies the model's geometric-mean squared error by 0.660693, a 33.93% reduction from its starting model error. The persistence-normalized reduction also depends on A0: at A0=2.04 it is 0.625173 × 0.339307 = 0.212125, or 21.21%. Averages of PSNR are averages of logs, so none of this establishes a 34% reduction in arithmetic mean pixel MSE. Source: A's definition and `adapt_summary.json`.
3. **Figure 1 describes a training-map column that is not present.** `fig_teaser_B.json` explicitly has `home: null` and four columns, and extracting text from the included PDF confirms those columns. Also, `results/teaser/index.json` identifies a continuous closed-loop rollout after 32 real context tics: `compose_teaser.py` selects predicted tics `t+8` beside true tic `t`, without reconditioning at that displayed true frame. The two selections are t=1 and t=14; the actual predictions are rollout tics 9 and 22. The figure is not evidence for a freshly conditioned eight-tic test or for combat fidelity; its controls also contain additional buttons and change through the rollout. “Keeps the action” is broader than the measured turn-reversal property.
4. **“One fine-tuned decoder” / “use it for every pixel number below” is not universal.** U-Net, PixArt and the adapters use the tuned SD 1 decoder, but SD 3.5 uses its own stock decoder (explicit in config and correctly qualified in Table 2/Figure 3). The Figure 2 caption's claim that prediction, truth and persistence are all rendered applies to A; M uses raw truth and raw persistence, and G uses raw truth. Figure 1's true frames are also raw. Source: `results/fresh_rescore/home_sd35_170000/val/metrics.json`, `results/teaser/index.json`, and the metric definitions.
5. **“SD 3.5 has the best one-tic numbers” needs the stock-decoder, full-frame scope.** The supplied training curves support this comparison at their latest EMA checkpoints: SD 3.5 23.356793 dB / 0.128738, U-Net 22.488819 / 0.177436, PixArt 22.479611 / 0.178150. Under Table 2's displayed decoder choices, SD 3.5 has the best training LPIPS but worse PSNR than U-Net and PixArt (24.50 versus 25.20 and 25.18), and lower A. Sources: `results/training_curves/{sd35,unet,pixart}.json` and the fresh home metrics. Its 200k target has not been reached in the supplied files: 170k is the latest read; the draft does acknowledge this elsewhere.
6. **The control checks are not complete for all 13 adapted arenas.** The tuned summary contains null forgetting and directional results. Searching exact base `scores.jsonl` files finds only arena 7's stock-decoder controls at steps 0 and 4,000: training A falls 4.115690 → 2.957477 (−1.158213 dB on 256 windows), while directional rises 0.773438 → 0.847656. These are not the tuned 512-window reference and cannot fill the advertised all-arena control cells. “We keep ... as control checks” should distinguish planned from completed reads. No full-fine-tune outcome is supplied; the table appropriately has TBD, but the introduction should not count a completed comparator as a result.
7. **“Orders nothing” / “does not order” and “predicts” overstate in-sample correlation evidence if read literally.** The stated correlations are correct, but failure to reject at n=13 is not proof of no association; S0's +0.8956 is an exploratory same-sample association with the final outcome, not validated predictive performance. The two transition measures fail perfect train/unseen separation: coverage ranges 0.523888–0.608878 on training maps and 0.526436–0.612669 unseen (12 unseen arenas below the maximum training value), while transfer-gap G ranges 0.005416–0.012317 and 0.008735–0.037802 (2 below). Their within-arena Spearman values with A0/final A/gain/budget are coverage −0.319/+0.055/+0.220/−0.238 and transfer-gap +0.121/+0.082/+0.055/−0.065, all p>0.28. The sentence “none below p=0.2” is correct for the **two named frame-distance tests**; it is not true of every distance/outcome test, since frame d versus gain has p=0.1173. Sources: `compare_fresh_sd1.csv` and `adapt_summary.json`.
8. **Scope of aggregation needs explicit wording.** “69 percent of the gain” is the mean of per-arena gain fractions, not the fraction inferred by dividing changes in medians (which is about 63.68%). “1.1 GPU-hours” is per arena, not the total for all thirteen (14.3411 GPU-hours). The matched collapse totals reuse 16 windows across 16 checkpoints; 256 is the number of rollout/checkpoint evaluations, not 256 independent trajectories. The bootstrap intervals quantify arena/episode resampling conditional on the fixed reference and this mostly single-seed recipe, not seed uncertainty.
9. **Other broad claims exceed this numeric audit's evidence.** The local results support strong scene-PSNR/persistence association (U-Net unseen Spearman 0.9451), but not a universal statement about what absolute PSNR “mostly measures”; Table 2 and Figure 3 also contain absolute scores, contradicting “every score here is a paired difference against persistence” if read literally. “Reconstruction upper bound” is an empirical decoder-reconstruction reference, not a proved bound over all possible predicted latents. Novelty/literature universals, actual public-release availability, and preregistration/freeze chronology were not independently established from the supplied result snapshots or via network access. The main results do not establish general action fidelity, planning utility, combat fidelity, or post-adaptation closed-loop stability.
10. **Primary-evidence gaps remain explicitly UNCHECKED above:** PixArt/SD 3.5 parameter counts; the four full-frame h4 scores; the “triples scene LPIPS” decoder-ablation claim. The latter is additionally inconsistent in precision with the available research-log account (about 3.7× stock, not exactly 3×); no decoder-ablation metric artifact was found locally. No missing experiment was run, and no uncertainty interval was re-estimated.

## At most five exact replacement sentences

These are proposed text only; `main.tex` was not edited.

1. Replace §2's reference sentence: “Pixel scores use the scene rows 0 to 207, since persistence copies the HUD almost exactly; A_train, the pooled decoded advantage on the training maps' 512 validation windows (5.06 dB, episode-bootstrap interval 4.75 to 5.38), is the in-distribution reference, not an arena's own upper bound.”
2. Replace §4's error-percentage sentence: “Half of the gap in dB is not half of the error: increasing A by 1.8 dB reduces the geometric-mean prediction MSE by 34 percent relative to its starting value.”
3. Replace the Table 2 caption: “Training maps (pooled means over 512 validation windows) and unseen arenas (medians of 13 per-arena means; parentheses count arenas with A>0 or M<0), using scene rows one tic ahead, the fine-tuned SD 1 decoder for U-Net and PixArt and SD 3.5's own stock decoder at a provisional 170k updates; A is scored against rendered truth, PSNR, LPIPS, M and G use raw truth, and directional scores are pooled separately (raw true-frame references 0.885 and 0.892).”
4. Replace the Figure 1 caption: “On unseen arena 7, continuous closed-loop U-Net rollouts before and after 4k adapter updates on eight episodes are shown at tics 9 and 22 beside true frames eight tics earlier and the corresponding future true frames.”
5. Replace §3's closed-loop sentence: “SD 3.5 has the best stock-decoder full-frame one-tic scores among the supplied final EMA reads, yet on 16 matched windows at each of 16 checkpoints from 55k to 130k, a frame under 10 dB occurs in 23 of 256 non-EMA and 9 of 256 EMA rollouts (11 and 8 of 240 without 70k), showing that EMA reduces these failures without eliminating them.”
