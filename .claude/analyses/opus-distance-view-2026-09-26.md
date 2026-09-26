# Opus view: the distance does not order the arenas (2026-09-26)

Third view on `astra-brief-distance-rethink-2026-09-26.md`. Sections 1 to 4 predate reading Astra. Inputs: the frozen U-Net 200k EMA `*_h1`/`*_h4` `metrics.json`, `distances_sd1.json`, `per_episode_sd1.csv`. The repo holds no per-window rows, so this is map level.

## 1. Diagnosis (13 unseen arenas; Spearman, permutation p)

| PSNR gain over persistence vs | ρ | p |
|---|---|---|
| D | −0.02 | 0.97 |
| persistence PSNR | −0.77 | 0.002 |
| per-map motion mean | +0.84 | <0.001 |
| deaths per episode | −0.11 | 0.72 |
| decoder ceiling (`vae_psnr`) | −0.43 | 0.15 |
| ceiling minus persistence | +0.82 | 0.001 |

Partial on D given persistence: −0.12.

**What the variation is: a floor in the metric, not the map shift.** In every group, model PSNR ≈ 0.6 × persistence PSNR + c (slopes 0.58 training, 0.64 arenas, 0.53 campaign). With a slope below 1, gain falls about 0.4 dB per dB of persistence by construction.

The model's pixel error has a floor (decoder reconstruction plus averaging blur; 2.0 to 4.0 dB below the ceiling on every arena). Raw-frame persistence has none, so on static footage it wins however good the latent prediction is. Three reads confirm this:

- **Decoder-matched gain** (`psnr_dec − copy_psnr_dec`) compares the predicted latent and the copy-last latent, both decoded, against the decoded target on the same windows.
  - It does not track persistence (ρ −0.05).
  - It is **positive on all 13 unseen arenas** (+0.72 to +2.68 dB) and on 9 of the 13 campaign maps.
- **LPIPS gain** tracks persistence with the opposite sign (+0.58).
- **A latent-skill proxy**, S̃ = −10 log10(latent_mse / (motion²/5120)), uses cloud motion as a stand-in for copy-last error.
  - It does not track persistence (−0.01).
  - Scores: training 2.72 ± 0.44, unseen arenas 1.50 ± 0.29, campaign 1.22 ± 0.32. The absolute scale is VERIFY; I use ranks only.

With the floor removed, D gives −0.23 (decoder-matched) and −0.26 (S̃) within the arenas. Both are below the 0.56 needed at n = 13.

Both signals vanish once arenas 6, 7 and 8 are dropped (+0.16 and +0.01). Those three come from a separate recording batch (`arenas_678`). On decoder-matched gain they sit below the other ten, 0.96 against 1.67 dB (Mann–Whitney p 0.03). Batch and map are confounded here, and the fresh single-protocol set will separate them. Deaths explain nothing.

**Structure.** The deficit in S̃ is a common offset of about 1.2 (2.72 to 1.50) against a map SD of 0.29: mostly "no memory of this map", shared by every arena. The orderable part is small and may partly be the batch.

**The right outcome** is not PSNR gain over raw persistence, because that mixes in a decoder-floor term set by how static the target is. Blitzer's adaptation loss makes the same move: it subtracts intrinsic difficulty (ACL 2007, per the literature memo).

Use the frozen adaptation metric S = −10 mean log10 R, where R is the per-window latent MSE ratio to copy-last. Then the zero-shot check and the adaptation cost share one outcome. Report decoder-matched gain beside it, and as robustness a motion-conditional excess: the per-map mean residual from E[log model latent error | log copy-last latent error] fitted on training-map validation windows.

**Headline consequence.** "Below persistence on 17 of 26 unseen maps" holds only in raw-pixel PSNR; decoder-matched, the model beats persistence on every unseen arena. Tonight's latent-ratio rescore settles it.

## 2. Candidate distances (CPU; per-tic latents plus executed controls)

The unit is a one-tic transition (z_t, a_t, z_{t+1}) at a scoreable tic. Features:

- s_t = `pool_latents(z_t)`, the state (192 dimensions).
- v_t = `pool_latents(z_{t+1} − z_t)`, the transition (192 dimensions).
- Each block is z-scored on the reference and scaled to unit total variance.
- a_t is one of 18 control classes: turn {L, R, none} × move {fwd, back, none} × attack. The exact `_meta.npz` key is VERIFY.

The reference is about 200k transitions from D's 50 seeded episodes per training map, pooled across maps. Pooling is right because the model trained on the union, and a directed score carries no mixture penalty. Each target map contributes 10k queries, uniform over valid windows.

**1 (first): kNN transfer gap G.**

- *Predictor.* v̂_t is the mean v of the k = 16 nearest same-class reference transitions in (s_t, v_{t−1}) space. If a class has fewer than 500 members, search all classes.
- *Per-window error.* r_t = ‖v̂_t − v_t‖² / ‖v_t‖², the same form as R.
- *Score.* G(m) = mean log10 r_t with training memory, minus mean log10 r_t with memory from the target's own 12 adapt episodes (equal reference size). Both are scored on the 8 held-out episodes.
- *Direction.* Target given training.

Subtracting own-map memory removes intrinsic difficulty, so G measures shift, not hardness: novel geometry revealed on turns, doors, control shift and appearance, each weighted by what it costs a predictor, which is what adaptation buys. It misses the network's invariances (a rotation of new textures is easy for the U-Net but novel to a kNN) and the 32-tic context.

**2: directed coverage C.** From the same search, C(m) = mean_t log(d_k(t) / d̄_k,val).

- d_k(t) is the distance from (s_t, v_t) to its k-th nearest same-class reference transition.
- d̄_k,val is the median of the same distance for held-out training-map transitions (the floor).
- Directed coverage beat symmetric OT for predicting transfer in Mensink (arXiv 2103.13318, Table 10) and Westny (arXiv 2606.30777, Table 2).
- Captures unfamiliar transitions without needing an outcome. It misses whether unfamiliar means hard.
- Secondary: C on (v, a) alone, which removes appearance.

**Validation before freezing.**

1. **Develop** on old episodes against the rescore. Choose k and choose between G and C.
2. **Confirm** on the fresh set's zero-shot S₀. Pre-declared bars:
   - ρ(score, S₀) ≤ −0.5 within the 13 arenas;
   - beats D and mean copy-last error;
   - split-half map reliability ≥ 0.8;
   - training maps sit inside the floor.

   Same maps, new episodes: not an independent-map test.
3. **Powered test** at the window level: log R_w ~ log r_w + log copy_w + (1 | map), over about 3,000 windows.
4. **Freeze** before any 13-arena adaptation score. The four-map pilot is excluded from selection.
5. **In the adaptation study,** report cost against the free zero-shot deficit S_train − S₀ as well. G earns its place only if it adds to that deficit.

**Prior.** Given that offset, the likeliest result is that no distance orders the 13 arenas; the null is publishable if the window-level test is powered.

## 3. Claim for the Sep 30 draft (17-arena single dataset)

> "The distance D from a map's per-frame latent distribution to the nearest training map separates the four training maps (D ≤ 0.07, inside the train-versus-train floor) from all 13 unseen arenas (D ≥ 0.115), but it does not order the unseen arenas: across them, zero-shot skill does not track D (Spearman −0.02 for PSNR gain over persistence, −0.23 for the decoder-matched gain; n = 13). Differences in PSNR gain between unseen arenas follow how static the footage is (ρ = −0.77 with persistence PSNR), an effect of comparing a decoder-limited prediction with raw-frame persistence that disappears in the decoder-matched comparison. We claim D as a seen-versus-unseen separator, not a per-map predictor."

Say "per-frame latent distance", not "appearance" (motion weights). If campaign maps return, add partial −0.73 over 30 maps with the failed gate. Recompute on the fresh set.

## 4. Cost

| Item | Data | CPU | When |
|---|---|---|---|
| S, decoder-matched gain, motion-conditional excess on 30 maps | rescore rows (running) | minutes | Sunday morning |
| G and C, development | 200 reference episodes (10 GB read) plus old episodes | half a day of code; 3 to 4 h on 16 cores | Sunday evening |
| G and C, fresh set | 13 × 20 episodes after the encode | about 2 h | Monday |
| Window-level validation, freeze | fresh rescore rows | under 1 h | Tuesday Sep 29 |

## 5. Where I differ from Astra

- **Diagnosis.** Astra reads the within-arena variation as motion content, since the motion partial given persistence is +0.65. Astra rules out the decoder because no arena's ceiling falls below persistence. I read it as a floor in the model's pixel error: decoder-matched gain and S̃ show no persistence dependence at all. Astra did not compute `psnr_dec − copy_psnr_dec`, which is positive on all 13 arenas.
- **Batch.** Astra did not flag the `arenas_678` recording batch as a confound.
- **Candidates.** Astra ranks directed coverage first; I rank the difficulty-subtracted kNN gap first and coverage second. Astra's features are richer (lags 1, 4, 16, 31; control-history rates) and its shuffle tests belong in validation; I would adopt both.
- **Outcome.** Astra proposes a ratio of summed errors; decision 7's mean of log R already stops static windows dominating, so I would not reopen it.
- **Claim.** Astra keeps 30 maps and "appearance-based"; the single-dataset rule moves it to 17 arenas. Cost: Astra 8 to 30 core-hours, mine about 6; both fit Tuesday.
- **Agreed.** Zero-shot skill enters as a competing predictor of adaptation cost. The null is wide and is not an equivalence result.
