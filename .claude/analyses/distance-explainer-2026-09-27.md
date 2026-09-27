# The frame distance D, explained end to end (for Rohan, 2026-09-27 11:20 EDT)

Code: `distance_study.py` (docstring and `motion_weights`, `sliced_wasserstein`, `nearest_reference_distance`); frozen configuration and numbers: `results/distance_v2/distances_sd1.json`; design memo: `.claude/analyses/distance-study-design-2026-09-24.md`. Citations marked (verify) are to be checked by Astra before they enter the paper.

## 1. What D is, step by step

1. **A map is a cloud of frames.** Every recorded tic of a map has an SD 1 latent (4 x 32 x 40). We pool each latent to 192 numbers (4 channels x 6 x 8 blocks) so a frame is a point in a 192-dimensional space that encodes what the frame looks like: walls, floor, lighting, enemies. No time coordinate is kept.
2. **Equal-size clouds.** For a target map we draw 250 frames per episode from 4 episodes (1,000 frames), 25 per motion decile so still and moving frames are both represented, and repeat over 10 disjoint 4-episode subsets. Each training map contributes 12,500 reference frames from 50 episodes. Comparing "a large training corpus" with "a small unseen map" is therefore never done directly: the estimator always compares two clouds of the same construction, and the finite-sample bias that any distribution distance has at 1,000 points is the same on both sides and is calibrated by the floor (step 5).
3. **Motion weights.** Each frame is weighted by how much the latent moved into it, with a floor so the quietest half of a cloud still holds a quarter of the weight. A map's shift should be measured where the game happens, not in the frames where the player stands still; this is the rule from the NVIDIA internship method carried over. `D_uniform` (no weights) is stored beside it and differs by at most 0.006 on the 13 arenas, so the weighting changes nothing here.
4. **Sliced Wasserstein-2 between two clouds.** Project both clouds onto 1,000 random unit directions (the same directions for every map, seed 0), solve the one-dimensional optimal transport exactly on each direction (sort both projected samples and pair quantiles), and average the squared transport cost over directions. This is the sliced Wasserstein distance: the standard cheap, sample-efficient estimator of the Wasserstein distance between two empirical distributions in high dimension (verify: Rabin, Peyre, Delon, Bernot, SSVM 2011; Bonneel et al., JMIV 2015; Kolouri et al., NeurIPS 2019, arXiv 1902.00434; statistical properties in Nadjahi et al., NeurIPS 2020, arXiv 2003.05783). Full Wasserstein in 192 dimensions from 1,000 points would be dominated by the curse of dimensionality; sliced is the form that has finite-sample guarantees at this size.
5. **Nearest training map, not the pooled corpus.** D is the distance to the nearest of the four training maps (the minimum over maps, averaged over the 10 subsets). Distance to the pooled corpus charges a map for not also looking like the other three training maps; `D_pooled` is stored beside D and is 0.02 to 0.07 larger on every arena for that reason. Twelve of the 13 arenas are nearest to map 3, arena 8 to map 4, arena 11 to map 2, arena 16 split between 2 and 3.
6. **The floor.** What is "zero shift" in this estimator? A training map against a second, disjoint draw of its own episodes (`floor_draw 2`), which gives 0.022 to 0.093 over 40 such comparisons (mean 0.044). Any map inside that band is indistinguishable from a resampling of training footage. The training maps' held-out validation episodes score 0.026 to 0.069 (inside the floor, as they must); every unseen arena scores 0.118 to 0.270 (outside).
7. **Uncertainty.** A bootstrap over the map's episodes (200 draws, each averaging 10 subsets as D does) gives a standard deviation of 0.002 to 0.011 per arena, ten times smaller than the spread between arenas (0.118 to 0.270), so D is a precise measurement; whatever it fails to predict, it does not fail from noise.

## 2. Why this distance and not another

- **Model-free and pre-computable.** D needs the VAE and the footage, no world model, so a reader can compute it for a new map before deciding whether to adapt. That is the whole point of a distance as a planning tool, and it is why the model-side quantities (zero-shot skill, the deficit Δ) are reported as a separate thing.
- **In the model's own input space.** The clouds are SD 1 latents, the space every backbone reads, so "far" means far as the model sees it, not as a pixel histogram sees it. XEWorld's appearance distance (2608.05799) uses HSV histograms and silhouette moments; ours is the latent analogue.
- **A distribution distance, not a nearest-neighbour coverage.** The internship method scored per-scenario coverage of a test set by the training set; here maps of one WAD share appearance so densely that coverage saturates (this is what the directed-coverage candidate showed on Sep 27: training maps' held-out episodes are as "uncovered" as unseen arenas). A transport distance between clouds measures how the whole distribution of looks differs, which is the quantity that separates the families.
- **Wasserstein rather than Frechet or MMD.** Frechet distances (FID 1706.08500, FVD 1812.01717) assume Gaussian clouds and reduce to means and covariances; MMD needs a kernel bandwidth. Optimal-transport distances between datasets are the established form for dataset-to-dataset distance in transfer learning (OTDD, Alvarez-Melis and Fusi, NeurIPS 2020, 2002.02923; s-OTDD 2501.18901), and sliced Wasserstein is its label-free, sample-efficient version.
- **Nearest map rather than mixture.** The same choice as the directional dataset distances in Westny et al. (2606.30777): a target is as far as its closest source.

## 3. What D predicts and what it does not (the numbers)

| arena | D | A0 | A at 4k | gain | S0 | budget |
|---|---|---|---|---|---|---|
| 8 | 0.118 | 2.04 | 3.44 | 1.40 | 1.36 | censored |
| 15 | 0.127 | 2.88 | 4.56 | 1.68 | 1.96 | 500 |
| 17 | 0.130 | 1.65 | 3.83 | 2.19 | 1.52 | 1,000 |
| 12 | 0.138 | 1.79 | 3.33 | 1.54 | 1.21 | censored |
| 14 | 0.147 | 1.85 | 3.62 | 1.77 | 1.25 | 2,000 |
| 13 | 0.154 | 2.12 | 3.80 | 1.68 | 1.42 | 2,000 |
| 10 | 0.155 | 2.31 | 4.82 | 2.52 | 1.73 | 250 |
| 11 | 0.156 | 2.28 | 4.31 | 2.03 | 1.80 | 250 |
| 16 | 0.171 | 2.53 | 3.64 | 1.11 | 1.68 | censored |
| 1 | 0.172 | 1.84 | 3.37 | 1.53 | 1.31 | censored |
| 9 | 0.184 | 2.88 | 5.11 | 2.23 | 1.86 | 250 |
| 6 | 0.198 | 1.07 | 3.38 | 2.31 | 1.14 | 1,000 |
| 7 | 0.270 | 1.07 | 3.86 | 2.79 | 1.42 | 250 |

Tuned decoder, scene crop; in-distribution reference 5.06. Spearman of D with A0 −0.22, with A at 4k +0.10, with the gain +0.46 (p 0.12; +0.31 without arena 7), with the budget −0.35. The zero-shot skill S0: +0.90 with A at 4k, +0.18 with the gain, −0.61 with the budget.

**The "trend" is mostly one arena.** Read down the D column: the two farthest arenas (6 at 0.198, 7 at 0.270) do start lowest (A0 1.07 both) and gain the most (2.31, 2.79). Everything above them is flat: arenas at D 0.118 and 0.184 both start at 2.04 and 2.88, and the censored arenas (8, 12, 16, 1) sit at four different distances. So D "works" at the far end and says nothing in the middle. That is what the correlation of +0.46 with the gain, falling to +0.31 without arena 7, means.

## 4. The outlier: why arena 7 starts far and improves fast

Arena 7 is the farthest map by appearance (D 0.270, lava floors and a different wall set; the figure-1 strips show the model's textures drifting toward the training maps' brick). Its zero-shot *decoded* advantage is the lowest of the 13 (1.07 dB), but its zero-shot *latent* skill is average (1.42, rank 7 of 13). That is the signature of an appearance shift rather than a dynamics shift: the model's latent predictions on arena 7 are as good as elsewhere relative to copy-last, but when rendered they pay in pixels for every wrong texture, so the decoded advantage collapses. Appearance is also the cheapest thing for an adapter to learn: one episode of arena 7 contains its textures, so the adapter recovers 2.8 dB and crosses at 250 updates, and ends mid-pack (3.86), exactly where its latent skill said it belonged. Arena 6 (D 0.198, A0 1.07, S0 1.14, the lowest skill) is the other far map, and it is different: it is hard in latent space too, so it starts low for two reasons and crosses only at 1,000.

So the outlier is not a failure of D; it is the one arena where D measures the thing that matters (appearance), because it is the only arena whose appearance shift is large. On the other eleven, appearance shift is small and similar (D 0.118 to 0.184), and what differs between them is how predictable their play is, which no appearance distance can see and which the zero-shot skill measures directly. This reading is a hypothesis with one supporting arena; the October constructed-shift experiment (retexture a training map at several strengths) is the test.

**Arena 9 above the in-distribution line.** It has the highest zero-shot skill (1.86) and the highest A0 (2.88): the most predictable arena of the 13, more predictable than the average training map after adaptation. Nothing about D explains it (0.184, mid-range); its predictability does.

## 5. What the paper says (and cites)

"We measure appearance shift with a frame distance D: the sliced Wasserstein-2 distance (Bonneel et al. 2015; Kolouri et al. 2019) between motion-weighted clouds of a map's per-frame SD 1 latents and those of its nearest training map, calibrated against a resampling floor of the training maps themselves, in the spirit of optimal-transport dataset distances (Alvarez-Melis and Fusi 2020) and of the appearance distance XEWorld reports for held-out robots. D places every unseen arena outside the training floor and does not order them; the two farthest arenas start lowest and gain most, and the rest are ordered by the model's own zero-shot skill."

## 6. The open question you raised, answered as far as it can be

"How can we compare a domain shift between a large corpus and a small unknown target?" By never comparing them at unequal size: fixed 1,000-frame clouds on both sides, the same estimator and the same random directions for every map, the finite-sample bias measured on training-versus-training pairs and reported as the floor, and the target's small size entering only through the width of its episode bootstrap. What the estimator cannot do is turn an appearance distance into a prediction of adaptation cost when the targets differ in something other than appearance; that limitation is the finding, and the model-side deficit Δ is the quantity that fills it.
