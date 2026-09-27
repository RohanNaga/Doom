# Truth check of the frame-distance explainer

Reviewed 2026-09-27 against `.claude/analyses/distance-explainer-2026-09-27.md` (line numbers below refer to that version). **The section 3 table and all quoted correlations reproduce; several implementation explanations and section 4 rankings need correction.** All eight supplied arXiv IDs resolve to the intended papers. The main citation problems are what the sentences infer from those papers, not incorrect IDs.

Evidence shorthand: **C** = `distance_study.py`; **J** = `results/distance_v2/distances_sd1.json`; **A** = `paper/tables/tuned/adapt_summary.json`. JSON selectors such as `maps[map=8]` mean the list entry selected by its `map` value. Code SHA-256 is `7caf0ca98b3c2f2f4de2a37de433d9d6cc63826831f6a7868312deea237eaf5a`, exactly matching `J.code.distance_study_sha256`; the frozen producer commit is `5e927d3fd7b1d0329ff4a69c6e3fa6c0719eceda`. Thus these implementation checks use the actual frozen code version.

## 1. Section 1: claims against implementation and frozen configuration

| Explainer claim | Verdict | Quoted code / frozen field and implication |
|---|---|---|
| Step 1, L7: SD 1 latent shape is `4 x 32 x 40`, pooled to `4 x 6 x 8 = 192`. | Correct, with omitted preprocessing. | C:582–585 checks `want = (SPACES[space]["channels"], 32, 40)`; C:262 defines SD 1 as `{"channels": 4, "dim": 192}`. C:72–73 sets `VISIBLE_ROWS = 30`, `POOL_BLOCK = 5`; C:215–216 uses `z[:, :, :rows]` and `.mean(axis=(3, 5))`. It removes the bottom two padding rows, averages 5×5 blocks over 30×40 cells, and **keeps the HUD** (C:205); this is not a scene-only representation. |
| Step 1, L7: every recorded tic is represented; no time coordinate is kept. | Qualify the first; second correct. | C:601–610 draws only `eligible_rows(meta, a.context_frames)` and stores `"feats": pool_latents(lat[r])`. C:264 sets the context to 32; C:327–334 requires consecutive tics and a full same-life context. The cloud contains sampled eligible frames, not every recorded tic. C:17 says `There is no time coordinate`; `tic` is stored as metadata (C:611), not projected as a feature. Descriptions of walls/lighting/enemies are interpretations of the VAE representation, not semantic guarantees of the pooling operation. |
| Step 2, L8: 250 frames/episode, 25 per motion decile, four episodes/target cloud. | Correct for the implemented construction and frozen sizes. | C:265–268 sets `FRAMES_PER_EPISODE = 250`, `MOTION_BINS = 10`, `EPISODES_PER_CLOUD = 4`; C:399–400 partitions motion ranks using `np.array_split` and samples each bin with `replace=False`; C:606–607 passes `frames_per_episode // bins`. J: `config.episodes_per_cloud = 4`; all 13 `maps[].frames_per_cloud = 1000.0`, `n_episodes = 24`. The distance JSON does not itself store the cloud-generation `motion_bins` or per-episode draw arguments; those are stored in the cloud metadata by C:741–744, whose NPZ files are not in this local result directory. |
| Step 2, L8: “10 disjoint 4-episode subsets.” | Incorrect. | C:1038–1041 says `n // 2 disjoint PAIRS`; C:1055–1061 draws two disjoint four-episode subsets at a time, allowing episodes to recur in other pairs. J: `config.subsets = 10`, every map has `subsets = 10`, `disjoint_pairs = [[0,1],[2,3],[4,5],[6,7],[8,9]]`. For arena 1, subset 0 is `[26,117,260,312]` and subset 2 is `[143,260,312,325]`: they overlap. Forty mutually distinct episode slots are impossible with its 24 episodes. |
| Step 2, L8: reference maps contribute 12,500 frames from 50 episodes each. | Correct. | J: `references.maps = [2,3,4,5]`; `references.frames_per_map = {"2":12500,"3":12500,"4":12500,"5":12500}` and `episodes_per_map` is 50 for each. C:1105–1109 selects **all** reference frames of each map. C:723–748 constructs and saves the training/evaluation cloud files; C:550–575 selects the seeded reference episodes. |
| Step 2, L8, repeated in §6 L59: “two clouds of the same construction,” “fixed 1,000-frame clouds on both sides,” hence equal finite-sample bias. | Incorrect and internally inconsistent with the preceding reference-size sentence. | C:1189–1191 makes 1,000-frame target subsets; C:1151–1154 retains the full 12,500-frame reference per map. C:252 calls `sliced_wasserstein(x, ref, ...)` without reference downsampling, and `cmd_distances` uses the equivalent batched engine. The floor also uses 1,000 against 12,500. Fixed target sizes standardize comparisons across targets; they do not establish equal bias on both sides or remove distribution-dependent bias. |
| Step 3, L9: motion into each frame plus a floor gives the quietest half a quarter of the mass. | Correct as **at least** a quarter. | C:167 computes `np.linalg.norm(np.diff(z.reshape(...)), axis=1)`; C:605 measures this on the original, unpooled latents. C:194–196 uses `floor = max(0.0, (quiet_share * total - quiet) / (h - quiet_share * n))`, `w = m + floor`, `return w / w.sum()`. J: `config.quiet_share = 0.25`, `primary_arm = "motion"`. The floor is zero when motion alone already gives the quiet half ≥25%; weights are computed separately for each target cloud and each reference. All-zero motion raises an error (C:188–189). The NVIDIA provenance is stated in C:8–9 but cannot independently establish what the internship method did. |
| Step 3, L9: `D_uniform` differs by “at most 0.006,” so weighting “changes nothing.” | Approximately the rounded magnitude, but literally false and overstated. | J: `maps[map=11].D = 0.15599316784118852`, `D_uniform = 0.1621630717042625`: maximum absolute difference **0.006169903863073978**. The rank correlation of the two arms is **0.9945054945**, with arenas 1 and 16 exchanging order. “Small sensitivity” is supported; exact invariance is not. |
| Step 4, L10: 1,000 random unit directions, shared across maps, seed 0. | Correct. | J: `config.projections = 1000`, `direction_seed = 0`, `p = 2`. C:81–82 draws standard-normal columns and divides by their norms; C:1150 creates `dirs` once for the batched study. C:927–942 processes all directions in blocks (`config.block = 50`); the separate `mixture_projections = 100` applies to the mixture search, not D. |
| Step 4, L10: exact one-dimensional transport by sorted quantiles. | Correct, including unequal counts and weights. | C:103–107 specifies the integral over inverse CDFs and merged cumulative-weight breakpoints; C:110–125 sorts projected values and sums `du * abs(qx - qy) ** p`. “Pair quantiles” must mean this weighted quantile integration, not equal-index pairing of 1,000 points with 12,500. |
| Step 4, L10: average squared transport cost, then call that SW2. | Missing the square root. | C:154: `return float(np.mean(cost) ** (1.0 / p))`; the `distances` engine returns `np.sqrt(total / L)` (C:942). With `p=2`, D averages **rooted** SW2 values across subsets, not squared costs. |
| Step 4, L10: SW is an estimator of full Wasserstein; full W at this dimension/sample size must be dominated by the curse, while sliced has guarantees “at this size.” | Misleading / unsupported as stated. | C:132 defines **SW**, a distinct distance based on projected W2 costs; finite directions approximate its spherical integral, not generally full W2. Nadjahi supports conditional dimension-independent sample-complexity results for sliced divergences, not a numerical accuracy guarantee for these temporally dependent, motion-weighted, stratified 1,000-frame clouds. No distribution-specific error bound at n=1,000 is supplied. See citation audit. |
| Step 5, L11: take nearest of four maps for each subset, then average ten subsets. | Correct. | C:1213–1215: `return float(v.min()), train_maps[int(np.argmin(v))]`; C:1222 takes this minimum per cloud; C:1240: `p["D"] = float(np.mean(...["values"]))`. This is **mean of minima**, not minimum of four mean distances. |
| Step 5, L11: `D_pooled` is 0.02–0.07 greater on every arena. | Incorrect. | C:1113–1120 gives each training map equal total mass in the 50,000-frame pooled reference. J gives `D_pooled - D = -0.009890069` for arena 8 and `-0.024233920` for arena 11; the full range is **−0.024233920 to +0.072770700**. Positive differences for arenas 14 and 16 are only +0.009330035 and +0.007442163. A mixture can be closer than each component, so “charges a map” is a motivation, not a universal ordering theorem. |
| Step 5, L11: twelve arenas nearest to map 3, with listed exceptions. | Incorrect count; exceptions largely correct. | J `nearest_counts`: ten arenas (1,6,7,9,10,12,13,14,15,17) have `{"3":10}`; arena 8 has `{"4":10}`, arena 11 `{"2":10}`, arena 16 `{"2":5,"3":5}`. Arena 16's stored `nearest = 2` is a modal tie result, not a unique nearest map (C:1402–1403). |
| Step 6, L12: second disjoint training draw creates 40 floor comparisons, range 0.022–0.093, mean 0.044. | Numerical values correct; comparison construction misstated. | J: `config.reference_draw = 1`, `floor_draw = 2`, `floor_subsets = 10`; `floor.motion.n = 40`, `min = 0.022325065787288417`, `max = 0.09337349802877508`, `mean = 0.044386421011759644`. C:1138–1139 checks episode disjointness between the two draws. C:1197–1207 makes ten four-episode targets per training map from draw 2. **C:1246 uses `nearest(ai,c)` across all four draw-1 reference maps, not a forced comparison to the target's own map.** These are 40 individual cloud distances, not 40 draws of the ten-subset mean D. |
| Step 6, L12: anything in the floor band is “indistinguishable”; validation maps must fall inside it. | Statistical overclaim; validation range has older provenance. | The band is an observed min/max over 40 correlated floor clouds, not a calibrated hypothesis-test acceptance region or a guarantee. In J, `config.sets = ["arenas13"]`, and `checks.i.pass = null` with note `needs the second reference draw and the validation set`: **J contains no validation-map rows**. The quoted 0.026–0.069 is real in the older `results/distance_study/distances_sd1.json`, `checks.i.maps`: map 2 = 0.0648579863, 3 = 0.0336755487, 4 = 0.0693163618, 5 = 0.0260386785. Cite that file explicitly; “as they must” is not justified. |
| Step 6, L12: all unseen arenas have D 0.118–0.270, above the floor. | Correct as an empirical comparison. | J: smallest `maps[].D` = arena 8, **0.11788685535167391**; largest = arena 7, **0.2702537790745706**; each exceeds `floor.motion.max = 0.0933734980`. |
| Step 7, L13: 200 bootstrap draws, averaging ten subsets each, SD 0.002–0.011. | Conflates two bootstrap fields. | J: `config.bootstrap = config.bootstrap_full = 200`. C:1333–1343 draws **one** four-episode cloud with replacement per replicate and records `maps[].bootstrap`; its SD range is **0.0016563661–0.0108319156** (arenas 17/12). C:1082–1084 redraws all 24 target episode IDs with replacement, then samples ten four-position subsets; C:1374–1377 averages their nearest-reference distances into `maps[].bootstrap_full`. Its actual SD range is **0.0008730966–0.0049976337** (arenas 17/12). The full bootstrap reproduces the averaging structure but does not impose D's five-disjoint-pair subset design. |
| Step 7, L13: uncertainty is ten times smaller than spread, therefore failure cannot come from noise. | Too strong. | Across-map sample SD(D) = **0.0396274747**, not its range 0.118–0.270. J: `bootstrap.attenuation_ratio = 0.1079322726`, `attenuation_below_1pct = false`; `bootstrap_full.attenuation_ratio = 0.0545172398`. These are median bootstrap SD / across-map SD. Both bootstrap procedures hold the sampled frames within each episode, reference draw, feature space, and projection directions fixed (C:1336–1340, 1370–1374); neither establishes absence of reference/frame/projection uncertainty, systematic error, or rank uncertainty among close arenas. |

The exact frozen estimator is

\[
D(m)=\frac{1}{10}\sum_{s=1}^{10}\min_{r\in\{2,3,4,5\}}
\sqrt{\frac{1}{1000}\sum_{\ell=1}^{1000}
W_2^2\!\left((\theta_\ell)_\#\widehat\mu_{m,s},(\theta_\ell)_\#\widehat\nu_r\right)},
\]

where each weighted target measure has 1,000 sampled frame points and each weighted reference has 12,500; each measure's weights sum to one.

## 2. Citation identities and whether their use is supported

All eight arXiv IDs were checked live using the [arXiv API batch query](https://export.arxiv.org/api/query?id_list=1902.00434,2003.05783,2002.02923,2501.18901,1706.08500,1812.01717,2608.05799,2606.30777). Title/author records all match. Venue pages were also consulted where linked below. For the two older papers, the confirmed identifiers are publisher DOIs; no arXiv identifier is asserted.

| Citation | Correct identifier, title, venue | Verdict on the citing sentence |
|---|---|---|
| Rabin, Peyré, Delon, Bernot, §1 step 4 | DOI [10.1007/978-3-642-24785-9_37](https://link.springer.com/chapter/10.1007/978-3-642-24785-9_37), **Wasserstein Barycenter and Its Application to Texture Mixing**; SSVM **2011**, LNCS 6667, pp. 435–446; Springer bibliographic publication year **2012**. | Correct historical support for slicing/texture applications, not the n=1,000 assurance. Publisher abstract: “the original Wasserstein metric is replaced by a sliced approximation over 1D distributions.” Retain SSVM 2011 as the conference name, but use 2012 when following the publisher's citation year. |
| Bonneel, Rabin, Peyré, Pfister, §1 step 4 and §5 | DOI [10.1007/s10851-014-0506-3](https://link.springer.com/article/10.1007/s10851-014-0506-3), **Sliced and Radon Wasserstein Barycenters of Measures**; *Journal of Mathematical Imaging and Vision* **51**, 22–45 (**2015**), online publication April 2014. | Appropriate for the SW construction and §5 metric attribution. Abstract explicitly describes “1-D Wasserstein distances along radial projections.” It does not certify this Doom estimator's statistical accuracy. |
| Kolouri et al., §1 step 4 and §5 | [arXiv:1902.00434](https://arxiv.org/abs/1902.00434), **Generalized Sliced Wasserstein Distances**; [NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/f0935e4cd5920aa6c7c996a5ee53a70f-Abstract.html), volume 32. | ID and venue correct. Useful background on ordinary SW and its relationship to the Radon transform; the paper's new contribution is generalized/max-generalized SW. C uses ordinary **linear** projections, not its nonlinear generalizations. Does not support the specific sample-size guarantee. |
| Nadjahi et al., §1 step 4 | [arXiv:2003.05783](https://arxiv.org/abs/2003.05783), **Statistical and Topological Properties of Sliced Probability Divergences**; [NeurIPS 2020](https://papers.nips.cc/paper_files/paper/2020/hash/eefc9e10ebdc4a2333b42b2dbb8f27b6-Abstract.html), volume 33. | Correct paper for statistical properties. Venue abstract says “**under mild conditions**, the sample complexity of a sliced divergence does not depend on the problem dimension.” This supports a qualified motivation, not guaranteed accuracy for weighted, temporally dependent Doom frames or the assertion that full W2 must fail at exactly this sample size. |
| OTDD, Alvarez-Melis and Fusi, §2 L20 and §5 | [arXiv:2002.02923](https://arxiv.org/abs/2002.02923), **Geometric Dataset Distances via Optimal Transport**; [NeurIPS 2020](https://papers.nips.cc/paper_files/paper/2020/hash/f52a7b2610fb4d3f74b4106fb80b233d-Abstract.html), volume 33. | Correct precedent for model-agnostic dataset comparison and empirical transfer-hardness correlations; abstract explicitly makes those claims and allows disjoint label sets. §5's “in the spirit of” is defensible. §2's claim that ordinary SW is simply “its label-free, sample-efficient version” conflates distinct constructions and does not transfer OTDD's empirical findings to D. |
| s-OTDD, §2 L20 | [arXiv:2501.18901](https://arxiv.org/abs/2501.18901), Khai Nguyen, Hai Nguyen, Tuan Pham, Nhat Ho, **Lightspeed Geometric Dataset Distance via Sliced Optimal Transport**; [ICML 2025](https://proceedings.mlr.press/v267/nguyen25g.html), PMLR 267:46162–46177. | Correct ID and precedent for efficient OT dataset comparison. The venue abstract specifies **Moment Transform Projection**, which “maps a label, represented as a distribution over features, to a real number.” Ordinary SW over unlabelled frame features is not this label-aware s-OTDD construction; computational efficiency also does not itself establish finite-sample accuracy. |
| FID, §2 L20 | [arXiv:1706.08500](https://arxiv.org/abs/1706.08500), Heusel et al., **GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium**; *Advances in Neural Information Processing Systems* 30 (**NIPS 2017**), confirmed by API `journal_ref`. | Correct citation/use for Gaussian feature means/covariances; FID operates on Inception features. More precisely, FID fits Gaussian approximations rather than requiring actual data to be Gaussian. FID is itself based on Gaussian W2, so “Wasserstein rather than Fréchet” is an imprecise contrast: the operative distinction is empirical projected distributions versus a Gaussian approximation in a chosen embedding. |
| FVD, §2 L20 | [arXiv:1812.01717](https://arxiv.org/abs/1812.01717), Unterthiner et al., **Towards Accurate Generative Models of Video: A New Metric & Challenges**; verified as an **arXiv preprint (2018)** by the API and [Google Research's publication record](https://research.google/pubs/towards-accurate-generative-models-of-video-a-new-metric-challenges/). | Correct ID/use. [Paper §2, Eqs. 1–2](https://arxiv.org/html/1812.01717v2) explicitly gives the Gaussian means/covariances formula and then uses I3D video features. A conference/workshop venue was **not confirmed** by the inspected primary records; do not invent one. An attempted OpenReview API lookup returned HTTP 403. |
| XEWorld, §2 L18 and §5 | [arXiv:2608.05799](https://arxiv.org/abs/2608.05799), Chen et al., **XEWorld: Can Action-Conditioned World Models Generalize to Unseen Robot Embodiments?**; **arXiv preprint, August 2026** (API has no accepted venue). | Correct appearance-distance description and valid broad analogy. [Appendix C](https://arxiv.org/html/2608.05799v1) says “normalized 16×16 hue-saturation histogram” plus “seven log-transformed Hu moments,” averaged over nine standardized robot-only views, then pairwise cosine distance. It uses each held-out robot's **mean** distance to the other four robots, not minimum SW over map footage; it does not validate this estimator. |
| Westny et al., §2 L21 | [arXiv:2606.30777](https://arxiv.org/abs/2606.30777), **Unveiling Transferability in Trajectory Prediction via Latent Scene Embeddings**; **ECCV 2026**, confirmed by API comment “Accepted to ECCV 2026.” | ID/venue correct; “the same choice … a target is as far as its closest source” is unsupported. [Paper Eq. 8](https://arxiv.org/html/2606.30777v1) defines **pairwise asymmetric KL divergence between Gaussian latent dataset approximations**; Eq. 9 and Fig. 3 explain directional coverage/variability. That is not a definition by minimum over source maps. D's symmetric component SW distances and outer nearest-source aggregation are different choices. |

Additional contextual claims that affect those citations:

| Claim | Verdict / evidence |
|---|---|
| §2 L18, “SD 1 latents, the space every backbone reads.” | False. `backbones.py:52–53` explicitly says SD 3.5 is in a 16-channel latent space; `backbones.py:68` maps `sd35` to 16 and U-Net/PixArt to 4. SD 1 is a common distance representation chosen for this comparison, not every backbone's input space. |
| §2 L20, “MMD needs a kernel bandwidth.” | Too broad: MMD needs a kernel; bandwidth is a hyperparameter of common kernels such as the Gaussian RBF, not of every possible kernel (e.g. a fixed linear kernel). No cited paper establishes a universal bandwidth requirement. |
| §5 L55, “D … does not order them.” | D does produce a numerical ordering; failure to predict an outcome should be stated for that outcome. The nonsignificant D/gain correlation at n=13 is not proof of no association. The cited papers do not establish this local empirical conclusion. |

## 3. Recomputed section 3 table and correlations

Computed using **`/opt/miniconda3/envs/PERSEVE/bin/python`**, reading the unrounded `A.per_arena` fields and deriving `gain = A_budget - A0`. All **13 rows and all displayed numerical/censoring cells** in the explainer match the appropriate rounding. The precision below exposes near-ties hidden by the original table.

| Arena | D | A0 | A at 4,000 | Gain | S0 | First half-gap crossing |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 0.117886855 | 2.036097892 | 3.438343916 | 1.402246024 | 1.361032276 | >4,000 |
| 15 | 0.126808897 | 2.878576975 | 4.559550390 | 1.680973414 | 1.962360821 | 500 |
| 17 | 0.129875140 | 1.645927906 | 3.832807411 | 2.186879504 | 1.518959441 | 1,000 |
| 12 | 0.137593600 | 1.789354086 | 3.328473434 | 1.539119348 | 1.206146041 | >4,000 |
| 14 | 0.146725802 | 1.851713698 | 3.620127868 | 1.768414170 | 1.251516670 | 2,000 |
| 13 | 0.153598523 | 2.121122595 | 3.799065340 | 1.677942745 | 1.421816464 | 2,000 |
| 10 | 0.154972299 | 2.307948589 | 4.823287584 | 2.515338995 | 1.732893636 | 250 |
| 11 | 0.155993168 | 2.276899055 | 4.310293216 | 2.033394162 | 1.803265345 | 250 |
| 16 | 0.170803957 | 2.526544219 | 3.639243430 | 1.112699211 | 1.677311677 | >4,000 |
| 1 | 0.171658075 | 1.838515408 | 3.366173424 | 1.527658015 | 1.305473817 | >4,000 |
| 9 | 0.183636656 | 2.877916414 | 5.109124139 | 2.231207725 | 1.856899874 | 250 |
| 6 | 0.197993324 | 1.068928961 | 3.376384284 | 2.307455324 | 1.136231114 | 1,000 |
| 7 | 0.270253779 | 1.070312791 | 3.863947392 | 2.793634601 | 1.423241653 | 250 |

`A.decoder = "tuned"`, `budget = 4000`, `weights = "live"`, `home = 5.0592508260160685`; all 13 primary rows have `k = 8`, `seed = 0`. `paper/make_adapt_figures.py:18–31` identifies the scene-crop decoded advantage and home reference. `score_adapt.py:27–32` defines latent skill as **−10 × mean(log10(model latent MSE / copy-last latent MSE))**, in dB, not a raw ratio. The cost is each arena's first measured crossing of `A0 + (home - A0)/2`; all runs have the same total 4,000-update budget. Calling the varying crossing cost “budget” obscures this distinction.

For cost correlations, the frozen convention is `A.spearman_cost_censored_as = 8000`: the four censored arenas are tied above all observed costs. Any common value >2,000 yields these ranks, but 8,000 is not an observed crossing time. These are descriptive rank correlations under that censoring convention, not a survival-analysis estimate.

| Correlation | Recomputed Spearman rho | Two-sided SciPy p | n | Explainer verdict |
|---|---:|---:|---:|---|
| D vs A0 | −0.2197802198 | 0.4706148515 | 13 | −0.22 correct |
| D vs A at 4k | +0.0989010989 | 0.7478683036 | 13 | +0.10 correct |
| D vs gain | +0.4560439560 | 0.1172830654 | 13 | +0.46, p=0.12 correct |
| D vs half-gap cost | −0.3542796999 | 0.2349592548 | 13 | −0.35 correct with censoring convention disclosed |
| S0 vs A at 4k | +0.8956043956 | 0.00003480973 | 13 | +0.90 correct |
| S0 vs gain | +0.1813186813 | 0.5532950093 | 13 | +0.18 correct |
| S0 vs half-gap cost | −0.6065268463 | 0.0279663578 | 13 | −0.61 correct with censoring convention disclosed |
| D vs gain, excluding arena 7 | **+0.3076923077** | **0.3305892594** | **12** | **+0.31 correct** |

These p-values are SciPy's usual asymptotic Spearman p-values, not a newly performed permutation test. The decrease on omitting arena 7 establishes sensitivity to that point, not a causal explanation or proof that “everything above them is flat.”

Reproduction command (reads files only; does not rely on stored correlation results):

```sh
/opt/miniconda3/envs/PERSEVE/bin/python - <<'PY'
import json
import numpy as np
from scipy.stats import spearmanr

j = json.load(open('paper/tables/tuned/adapt_summary.json'))
rows = sorted(j['per_arena'], key=lambda r: r['D'])
x = {k: np.array([r[k] for r in rows])
     for k in ('D', 'A0', 'A_budget', 'S0')}
x['gain'] = x['A_budget'] - x['A0']
x['cost'] = np.array([r['cost_half_gap'] if r['cost_half_gap'] is not None
                      else j['spearman_cost_censored_as'] for r in rows])
for r, gain in zip(rows, x['gain']):
    cost = r['cost_half_gap']
    print(r['arena'], *(f'{r[k]:.9f}' for k in ('D', 'A0', 'A_budget')),
          f'{gain:.9f}', f"{r['S0']:.9f}", cost if cost is not None else '>4000')
for a, b in [('D', 'A0'), ('D', 'A_budget'), ('D', 'gain'), ('D', 'cost'),
             ('S0', 'A_budget'), ('S0', 'gain'), ('S0', 'cost')]:
    rho, p = spearmanr(x[a], x[b])
    print(a, b, f'rho={rho:.10f}', f'p={p:.10g}', 'n=', len(rows))
keep = np.array([r['arena'] != 7 for r in rows])
print('without arena 7:', spearmanr(x['D'][keep], x['gain'][keep]), 'n=', keep.sum())
for k in ('A0', 'S0', 'gain'):
    print(k, 'descending:', [r['arena'] for r in
          sorted(rows, key=lambda r: r[k] if k != 'gain' else r['A_budget']-r['A0'],
                 reverse=True)])
PY
```

## 4. Section 4 factual claims and interpretation

| Claim | Verdict | Evidence from unrounded `A.per_arena` |
|---|---|---|
| L47: arena 7 has the lowest A0 but S0 rank 7 of 13. | **Lowest A0 false; S0 rank correct.** | Arena 6 A0 = **1.068928960710764** is below arena 7 = **1.0703127905726433**; both display as 1.07. Arena 7 S0 = **1.4232416525394926**, exactly the median / rank 7 of 13. “Average” is acceptable colloquially as middle-ranked, not a claim that it equals the arithmetic mean. |
| L47: arena 6 has the lowest S0. | **Correct.** | S0 = **1.136231114115652**, minimum over 13; half-gap cost = 1,000. |
| L51: arena 9 has highest S0 and highest A0. | **Both false.** | Arena **15** leads S0 (**1.9623608214153163**, versus arena 9 **1.8568998741287268**) and A0 (**2.8785769753158092**, versus arena 9 **2.8779164142906666**). Arena 9 ranks second on both; the A0 difference disappears at two decimals. |
| L51: arena 9 is above the in-distribution line. | **Correct for A at 4k, not proof of intrinsic predictability.** | `A_budget = 5.109124138951302` versus `home = 5.0592508260160685`; it is the sole arena above home at the final budget. This is an observed adapted advantage over persistence, not an independently measured intrinsic predictability ranking or a significant difference established here. |
| L43 and §5 L55: the two farthest arenas start lowest and “gain most.” | **First correct; second imprecise/false if meaning top two.** | Arenas 6 and 7 are the lowest two by A0. Gain ranking is arena **7** (2.793635), **10** (2.515339), **6** (2.307455): arena 6 is third, not second. |
| L47: arena 7 gains 2.8 dB, crosses at 250, ends mid-pack at 3.86. | **Numbers correct.** | Gain = **2.7936346009373665**, cost = 250, A_budget = **3.86394739151001**, fifth-highest of 13. Mid-pack is loose but reasonable; “exactly where its latent skill said it belonged” overstates a rank-7 predictor versus rank-5 outcome. |
| L47: one episode contains the textures, therefore explains the 2.8 dB recovery. | **Unsupported by this table.** | Primary arena 7 row has **`k = 8`**. It is not a one-episode adaptation result; separate k=1 runs are listed among excluded rows in A's notes because they lack step-0 tuned reads. Neither the A/S comparison nor this table isolates texture learning from other adaptation effects. |
| L47–51: average latent skill plus low decoded advantage diagnoses appearance rather than dynamics; appearance is cheapest to adapt; only arena 7 has a meaningful shift. | **Hypotheses, not established mechanisms.** | Frame latents and latent prediction errors can both encode appearance; the decoder is nonlinear and S is relative to copy-last. The frozen metrics do not separate causal appearance/dynamics contributions or establish adaptation cost by factor. All 13 D values exceed the empirical floor, and arena 6 is also far. L49's explicit “hypothesis” caveat is appropriate and should govern the earlier categorical language too. Visual lava/texture-drift descriptions were not independently checked against figure strips in this audit. |

## 5. Five exact replacement sentences

These are proposed edits only; the original explainer was not changed. Other necessary qualifications are identified in the tables above.

1. **Replace the equal-size/disjoint-subset explanation in §1 step 2 and the corresponding §6 explanation:** “Each target estimate averages ten 1,000-frame clouds, each formed from four episodes with 250 motion-stratified frames per episode, arranged as five disjoint pairs that can overlap across pairs, and compares each cloud with all 12,500 frames from each 50-episode training-map reference.”
2. **Replace §1 step 4's algorithm sentence:** “For each pair of weighted clouds, we project onto 1,000 shared random unit directions with seed 0, integrate the squared difference between their weighted quantile functions exactly on each projection, and take the square root of the mean cost to obtain SW2.”
3. **Replace §1 step 5's pooled-distance and nearest-map numerical claims:** “The pooled-reference distance differs from D by −0.0242 to +0.0728 and is smaller for arenas 8 and 11; ten arenas select map 3 on every subset, arena 8 selects map 4, arena 11 selects map 2, and arena 16 splits five subsets each between maps 2 and 3.”
4. **Replace §1 step 7's bootstrap/precision claims:** “The 200-draw `bootstrap_full` resamples target episodes and averages ten four-episode subsets per draw, giving conditional standard deviations of 0.00087–0.00500, whereas `bootstrap` samples one four-episode cloud per draw and gives 0.00166–0.01083; neither includes variation in the fixed reference cloud, sampled frames within episodes, or projection directions.”
5. **Replace the section 4 ranking assertions:** “Arena 7 has the second-lowest A0 (1.070313 dB, just above arena 6 at 1.068929) and median S0 (1.423242, rank 7 of 13), arena 6 has the lowest S0 (1.136231), and arena 15—not arena 9—has the highest S0 (1.962361) and A0 (2.878577).”

## Verification scope and workspace record

This review checks the code against the frozen JSON, not a fresh distance run over the remote latent corpus: the referenced cloud NPZ files are not present locally. No SSH, distance regeneration, training, or commit was performed. All requested citation identities were confirmed; no accepted venue was found for XEWorld, and no conference/workshop venue was confirmed for FVD. The numerical validation-map range was traced to the older result file, not silently attributed to v2. Causal appearance/dynamics claims, internship-method provenance beyond the code comment, and figure-strip descriptions remain unverified.

Input SHA-256 values at final verification: explainer `d9dc8005906e91877f69b9cf686ae5f5b2fa21676c244a4f589f26434cafba46`; J `f44399e268d6c11de67189d57f3474f9383648e39af8f2a7f90b62ca4ad90197`; A `3c6b26477d0e89a83d72a02f1b9695dc6290bebf9df046075a1067b14d6fa157`. A's content changed during the audit without a write by this reviewer (initial SHA-256 `697ee13ed8a3f7506708499f9a0493aabcd3ca3c1dc39d671c24594a4d762560`); rerunning the reproduction command against the final file returned the same table, correlations, and rankings reported above.

Pre-existing status at start: modified `.DS_Store`; untracked `paper/main.aux`, `paper/main.bbl`, `paper/main.blg`, `paper/main.out`, `paper/main.pdf`, `scripts/spiderman/eval_curve.sh`, and `scripts/spiderman/wait_card.sh`. These were left untouched; the requested review is the only repository file created by this audit.
