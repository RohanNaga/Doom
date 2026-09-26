# Astra pairing review, Sep 26 2026

Thread id: `01a0dec4-0413-78c1-acab-acf0fa2c407f` (model gpt-6-astra, effort high, sandbox workspace-write with network, 3 turns)
Brief: `.claude/analyses/astra-brief-2026-09-26.md`, passed verbatim. Astra read the worktree at bf8ffc1 because local main lagged at 687a9ff and lacked `results/distance_study/`. No repository edits and no server access. Astra's scratch files are in `/tmp/astra-2026-09-26/` (audit.py, round2.py, round3.json). The supervisor's independent recompute scripts were verify1.py to verify3.py in the session scratchpad.

Every number below was recomputed from the frozen files by the supervisor, independently of Astra's scripts. Bootstrap CIs agree to about 0.02 at the endpoints.

## Corrections to the brief and to RESEARCH_CONTEXT.md

- The distance precision ratio (median bootstrap SD of D over the between-map SD) is 0.089 for SD 1.x, 0.139 for pixels and 0.076 for SD 3.5. The brief's "0.08 to 0.09" is wrong for pixels. The RMS ratios are 0.282, 0.249 and 0.254.
- Cross-space rank agreement is 0.863 (SD1/pixels), 0.903 (SD1/SD3.5) and 0.770 (pixels/SD3.5). It is not 0.86 to 0.90 across the board.
- The `gain_h4` partial in stats.json uses one-tic persistence as its covariate (make_distance_figure.py:323-338). With four-tic persistence the value is -0.797, not -0.795.
- RESEARCH_CONTEXT.md:14 says the EMA weights "never fall into" the collapse. That is stale: the EMA has 9 events in 256 on matched windows.
- docs/RESEARCHER_DOSSIER.md section 6.6 (about line 497) says a successful clamp would show that "the content is recoverable". That overclaims. A clamp enforces the channel statistic by construction, so it can only test the mean-shift hypothesis.

## 1. Distance study and the verdict gate

**Astra's independent proposal.** Keep the pre-declared gate and its "does not support" result permanently. Add a dated amended assessment that separates three things: measurement checks (i, iii, v), a scientific expectation (iv, WAD ordering) and outcome evidence. Keep persistence PSNR as the primary covariate and report lives or valid fraction as a labelled sensitivity. Lives and valid fraction correlate at -0.9999, so the analysis needs only one of them.

**Against the brief.** Astra agrees that (iv) is an assumption about the maps: arena 7 has D 0.282, farther than 11 of 13 campaign maps. On (ii), round 1 overreached. Astra said the bootstrap does not resample the reported estimator: D is the mean of 10 subsets (distance_study.py:1214), while each bootstrap draw uses one 4-episode cloud (:1306-1314). I pushed back that a single cloud is noisier than the subset mean. Classical measurement error also biases a correlation toward zero. Astra conceded the direction but not a hard bound. Its robustness checks all hold: controlling n_eff gives -0.751, controlling bootstrap SD -0.729, dropping maps 26/18/19 -0.710, uniform-weight D -0.737, the second reference draw -0.737, and mean per-episode D -0.735. D and n_eff have rank correlation -0.022.

**The problem neither side had seen.** Astra found it in round 1 and quantified it in rounds 2 and 3. The within-arena -0.57 depends on the four training maps: unseen arenas alone give -0.119 (n=13). The structural controls tell the same story:

| Controls beyond persistence PSNR | Partial rho | 95% case-bootstrap CI |
|---|---:|---:|
| none (primary) | -0.734 | [-0.840, -0.320] |
| training-map indicator | -0.558 | [-0.778, -0.138] |
| training-map indicator + family | -0.423 | [-0.694, +0.070] |
| unseen maps only (n=26) | -0.556 | not computed |
| unseen only + family | -0.431 | [-0.694, +0.094] |

With 30 maps, the pooled association cannot be separated from the structure of training maps, arenas and campaign maps.

**Synthesis: a better approach.** Report the pooled association as qualified evidence and never as a pass of the original gate. Proposed paper sentence: "Across 30 evaluated map distributions, the U-Net's one-tic PSNR gain over persistence was negatively associated with visual distance to the nearest training map after adjustment for persistence PSNR (partial Spearman -0.734); this association weakened after excluding training maps, and a family-adjusted unseen-only analysis remained inconclusive." Add the limitation that unseen arenas alone give -0.119. The first reviewer attack is the question "is this just training-map membership plus the arena/campaign split?"

Proposed amended verdict for make_distance_figure.py:405-430:
- Keep `original_verdict` unchanged.
- Add `amended_assessment`, dated 2026-09-26 and marked "specified after observing results", with these conditions:
  - primary CI excludes zero: pre-declared, true;
  - leave-one-map-out same sign: pre-declared, true;
  - both family correlations negative: pre-declared, true;
  - pixel-space agreement: pre-declared, true;
  - checks i, iii and v pass: amended use, true;
  - unseen-only family-adjusted CI upper bound below 0: amended, false;
  - all three backbones complete and negative: amended, pending;
  - the bootstrap covers the full estimator: amended, false.
- Printed string: "qualified evidence for the pooled U-Net association; original gate failed; unseen-only trend inconclusive; cross-backbone replication pending".
- "every_row" currently means "every row supplied", which can be the U-Net alone. Require the expected row list.

**Code and memo changes.**
- make_distance_figure.py:78-84: the permutation is Kennedy-style, correlating ex with permuted ey without refitting. The docstring says Freedman-Lane, so implement it. With the refit there are still 0 exceedances in 10,000 permutations.
- make_distance_figure.py:323-324: h4 is read with 2 bootstrap draws and its persistence is dropped. Keep h4 persistence and run a full bootstrap.
- make_distance_figure.py:342: propagate the outcome bootstrap to h4 and LPIPS.
- distance_study.py:1301-1320: bootstrap the full subset-averaging estimator.
- make_distance_figure.py:231: markers already take their shape from cluster. Add filled markers for training maps and open markers for unseen maps, annotate arena 7, and draw the zero-gain line.
- Design memo line 31: separate the measurement checks from the WAD expectation and document the amendment timing.
- eval_tf.py:312: add per-window copy-last latent MSE, which gives a persistence-normalized outcome that does not depend on the decoder.

## 2. Closed-loop stability

**Astra's independent proposal.** Use paired, checkpoint-specific event risks on identical windows. Treat "frame < 10 dB" as a low-fidelity event, not as a proven blank collapse. Report windows as clusters. State plainly that this is one training run.

**Against the brief.** Astra disagrees with using pooled totals. The brief's 48/288 live versus 25/352 EMA mixes checkpoints and window draws.

Matched seed-0 windows, 55k to 130k, verified against collapse_rates_50k_130k.md:

| Event | Live | EMA |
|---|---:|---:|
| frame < 10 dB | 23/256 | 9/256 |
| end < 12 dB | 17/256 | 4/256 |
| ch13 end state | 14/256 | 4/256 |
| frame < 10 dB, excluding 70k | 11/240 | 8/240 |
| frame < 10 dB at 70k | 12/16 | 1/16 |
| frame < 10 dB, 105k-130k | 4/96 | 4/96 |

The window-cluster bootstrap of live minus EMA gives 5.47 pp [2.73, 7.81], resampling 16 rollout-index clusters. Astra first hedged that rollout indices might not identify windows. I checked the code: windows come from a seeded RandomState over a fixed split (rollout_eval.py:38-75), and noise keys are (seed, episode, start, step) (rollout_eval.py:152-154). So index r is the same window with the same noise at every checkpoint and for both weight sets. Two consequences follow. The effective n is 16 windows, with checkpoints as repeated measures. The seed-1 draw changes windows and noise together.

**Synthesis: same approach from two directions.** Paper text: "Across 16 checkpoints evaluated on the same 16 validation windows, live weights produced a frame below 10 dB in 23/256 checkpoint-window evaluations, versus 9/256 for EMA (paired difference 5.47 pp, window-cluster bootstrap 95% CI 2.73-7.81). The benefit concentrated at 70k: excluding it, 11/240 versus 8/240. EMA also failed, and late matched-window counts were identical. One training run, fixed window bank."

For the release, use 200k EMA and report the per-checkpoint curve. A fresh window draw from the same val episodes is not an untouched holdout, and the test split stays sealed.

**Changes.**
- rollout_eval.py:126 and :129-154: separate the window seed from the noise seed, and write a window manifest plus per-event records (episode, start, map, onset, duration).
- RESEARCH_CONTEXT.md:14: replace the "never fall into" statement.

## 3. Paper spine

**Astra's proposal.** Persistence and transfer should share the lead, with the matched three-row comparison as the systems layer. The rows are not an architecture-causality claim, because the pretrained packages and latent spaces differ. Contribution sentence: "A reproducible multi-map Doom evaluation shows where pretrained world models improve on persistence, how those gains associate with visual distribution shift, and why short-horizon quality does not establish autoregressive reliability."

Page budget: motivation and related work 0.45; data, recipe and evaluation 0.8; persistence and transfer 1.5; closed-loop 0.8; limitations 0.45.
- Figure 1: the distance scatter.
- Figure 2: paired checkpoint event rates plus a 70k live/EMA strip.
- Table 1: three rows at h1 and h4, with persistence and reconstruction references.

Cut architecture-causality claims, the training chronology, the probe oscillation, extra training arms and novelty claims about persistence itself. Persistence has video-prediction precedent; the contribution is what it reveals here.

**Against the brief.** Astra partly disagrees. The brief treats distance as "the strongest quantitative finding", but after the structural controls the persistence result (U-Net below persistence on 17/26 unseen maps in PSNR and 26/26 in LPIPS, verified) is the clearest finding. **Synthesis: a better approach.**

Replacement text Astra proposes for RESEARCH_CONTEXT.md lines 10-13 and 17 (not applied):
- 10: "The matched-recipe comparison anchors an evaluation study; differing pretrained packages and latent spaces prevent an architecture-only interpretation."
- 11: add "At one tic, U-Net 200k EMA falls below it on 17/26 unseen maps in PSNR and 26/26 in LPIPS."
- 12: "pooled -0.734; adding training-map membership and family gives -0.423 [-0.697, +0.050]; the original gate failed; the dated amendment describes qualified pooled evidence, unseen-only inconclusive, other-backbone replication pending."
- 13: the matched stability counts, as in section 2 above.

## 4. Smaller items

**Probe oscillation.** Fourteen reads, a phase chosen after the fact and two exceptions do not establish a period. The trainer has no 10k-period mechanism (train_wm.py:667 is warmup then constant LR; train_wm.py:871-890 saves a checkpoint every 5k). The RC entries of 13:20 and 13:45 show a 200x jump at 135k while real-token and directional reads stay normal, so the real-token probe is the right next step and nothing more is needed.

Astra also found a real seed bug. At smoke_probe.py:188, `probe()` does not forward `--seed` to `gradient_and_sensitivity` (which defaults to seed=0 at :123). The token probe does forward it (:294-295). So the probe v2 "seeds 1/2" varied the batch but not the noise.

**Requested-action ablation.** Astra agrees with dropping it. It could not recompute the 11 percent and 0.15 percent figures from frozen artifacts.

**Synthesis: same approach from two directions.** Change: smoke_probe.py:188, forward `seed`.

## 5. Decoder tuning

**Astra's proposal.** Keep stock decoders primary and tuned decoders secondary. Run the two default MSE tunes, one SD1 decoder shared by the U-Net and PixArt and one SD3.5 decoder. The only extra analysis is paired decoding: the same predicted latents decoded with the stock and the tuned decoder, which requires no extra training arm. Skip the MSE versus LPIPS comparison, crop-specific training and any recipe search.

On the "contribution" question, the paired decoding is a table column and diagnostic, not a contribution. The 4 dB ceiling gap could contribute to the 0.7 dB row gap, but dB errors are not additive, so only paired decoding can say how much.

**Against the brief.** Agrees on presentation. **Synthesis: a problem neither had seen.** The launcher as written would leak and would tune only SD1:
- decoder_mse.sh never passes `--vae-id`, so the SD 3.5 tune would silently run on sd-vae-ft-mse.
- Its validation frames are the cached 2,000 of the 17-map corpus (header lines 13-15, finetune_decoder.py:373). That corpus includes maps 1 and 9-15, which count as unseen for the next-tic rows. Choosing among the hourly checkpoints on it would leak.
- The gate's baseline is `tuned_sd_lpips`, not stock (decoder_mse.sh:66-77).

Minimal changes for tonight (listed, not applied):
1. decoder_mse.sh:49: point validation at a new split containing val ids 6000-6099 of raw_arnold_dense/arenas, with a new frame-cache name (the cache key does not hash the split) and stride 1.
2. decoder_mse.sh:48: pass the VAE identity explicitly, using flags at finetune_decoder.py:568-582.
   - SD1: `--vae-id stabilityai/sd-vae-ft-mse --latent-channels 4 --scaling-factor 0.18215`.
   - SD3.5: `--vae-id stabilityai/stable-diffusion-3.5-medium --vae-subfolder vae --latent-channels 16 --scaling-factor 1.5305 --shift-factor 0.0609`.
3. Take the terminal checkpoint. The hourly checkpoints are diagnostics only.
4. decoder_mse.sh:66-77: run a stock-versus-tuned gate per space (`--baseline stock_sd1` or `stock_sd35`, using `stabilityai/stable-diffusion-3.5-medium#vae`, supported at vae_gate_score.py:131). Drop the hourly unseen-scoring loop.
5. For the tuned columns, run eval_tf.py six times (three rows at h1 and h4) with only the decoder changed: `--vae-path <out>/vae --latent-scale ... [--latent-shift 0.0609]` (eval_tf.py:417-420).
6. Operational fixes:
   - finetune_decoder.py:420: use fused AdamW.
   - finetune_decoder.py: add live logging.
   - decoder_mse.sh:55/79: propagate the exit status.

## 6. Post-training recipe

**Astra's proposal, made before reading main's arms.** Test the mechanism before training anything, with an inference-only rescue at 70k live:
- Reuse the same 16 windows and noise.
- Apply a training-calibrated channel-mean correction.
- Score image-quality events, not the channel statistic that the correction enforces.

This can detect a large effect against the 12/16 baseline, but not a small one: even 0/16 has a one-sided 95% upper bound of 17%. If training ever happens, compare structured-offset augmentation against plain continuation from identical weights and optimizer state. The motivation: the current augmentation is i.i.d. Gaussian (diffusion_v.py:143), which rarely produces a coherent channel-mean offset. Astra would defer self-rollout fine-tuning, because it confounds supervision with trajectory drift.

**Against main's four arms.** Astra disagrees with committing the GPU-day before Wednesday. **Synthesis: a better approach.** Run the rescue only if a normal eval slot exists before Sunday evening; otherwise the repair goes in the discussion section. Change: dossier section 6.6 (about line 497) rewords the clamp arm as a test of the mean-shift hypothesis.

## Decisions for Rohan

1. Framing: persistence and transfer as joint lead, with the qualified distance verdict and the "original gate failed" wording kept in the paper and the stats file.
2. Decoders: approve the two default MSE tunes tonight with the leakage fixes and the terminal checkpoint, or defer the tuned column.
3. Release 200k EMA and skip post-training training arms. The 16-rollout inference-only rescue runs only if a slot already exists.
