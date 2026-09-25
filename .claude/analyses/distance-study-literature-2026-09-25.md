# Distance study: literature survey (2026-09-25)

Scope: prior work for `distance-study-design-2026-09-24.md` (per-map latent clouds, motion-weighted sliced Wasserstein to the nearest training map, outcome = one-tic PSNR gain over copy-last-frame, partial Spearman over about 30 maps). Every figure, table, number and quote below was read from the paper's arXiv PDF, proceedings PDF or ACL Anthology PDF text on 2026-09-25. Anything not confirmed that way says **unverified**. "n" means the number of domain points in the plot or table, not the number of samples.

## Bottom line

- The study's core shape is standard in domain adaptation and transfer learning: one distance per domain against one normalized performance number per domain. The OT-family version exists (OTDD, Cui et al. EMD), and so does the normalized outcome (Blitzer's adaptation loss, OTDD's relative drop in error).
- Nobody found in this survey has done it for a generative world model or video predictor across many held-out maps or scenes. World-model papers report unseen-domain quality as a table by domain with 1 to 8 domains, and never set it against a distance.
- Copy-last-frame was a routine baseline in 2015 to 2019 video prediction (six instances below). It disappears from the 2024 to 2026 game and driving world models checked here, with one 2026 latent-space exception. Rohan's doubt is right for recent world models and wrong for the older video-prediction literature.

## Q1. Distribution distance vs performance drop

**Theory.** Ben-David et al. (Machine Learning 79:151–175, 2010, Theorem 2) bound target error by source error plus an empirical H∆H-divergence, a sample term and λ, the error of the best joint hypothesis. In practice the divergence is the proxy A-distance (PAD), a domain classifier's error (Ganin et al., JMLR 2016, arXiv 1505.07818, §3.2, Fig. 3). Miller et al. (Fig. 1, top left) note that such bounds limit deviation from y = x but predict no specific trend.

**Empirical correlations.**
- *Blitzer, Dredze and Pereira (ACL 2007, P07-1056)* is the canonical scatter. Figure 3 plots proxy A-distance against average adaptation loss for the 6 unordered pairs of 4 Amazon review domains, with labelled points and no statistic. Adaptation loss is the in-domain gold-standard accuracy minus the adapted accuracy (§4, text before Fig. 1), so each target's intrinsic difficulty is subtracted out.
- *Alvarez-Melis and Fusi, OTDD (NeurIPS 2020, arXiv 2002.02923)* is the closest method match: an optimal-transport distance between labelled datasets, correlated with transferability. Transferability is the relative drop in target error from pretraining on the source, compared with target-only training, "to make these numbers comparable across" pairs of different hardness (§6.2 of the NeurIPS version). Each panel prints ρ and a p-value:
  - Fig. 6a: *NIST and USPS pairs, n = 16, ρ = −0.59, p = 0.02, ±1 s.d. over 10 seeds.
  - Fig. 6b: MNIST augmentations to USPS, n = 11, ρ = −0.85, p = 1.0×10⁻³.
  - Fig. 7a: Tiny-ImageNet augmentations to CIFAR-10, ρ = −0.74, p = 9.8×10⁻³.
  - Fig. 7b: BERT-embedded text datasets, ρ = −0.59, p = 3.7×10⁻⁵.

  The text never says whether ρ is Pearson or Spearman (**unverified**).
- *Cui et al. (CVPR 2018, arXiv 1806.06193)* define domain similarity as exp(−γ·EMD), γ = 0.01, where the EMD runs between per-category mean features weighted by each category's share of images (§4.1, Eqns. 2–3). This is a weighted OT distance between two point clouds, structurally our construction. Figure 5 plots transfer accuracy against this similarity: one line per target dataset (7), one marker per source domain (5), no statistic. Food101 breaks the trend, and the authors attribute that to its large target training set.
- *Deng and Zheng (CVPR 2021, arXiv 2007.02915)* measure the Fréchet distance between training features and each "sample set" using the classifier's penultimate layer. Figure 2 reports Spearman ρ ≈ −0.91 against accuracy on digits and on COCO, with a robust linear regression line. The points are synthetic sample sets, meaning transformed copies of a seed set. The meta-sets hold 3,000 (digits) and 1,600 (COCO) sets; the number plotted in Fig. 2 is **unverified**.
- *Achille et al., Task2Vec (ICCV 2019, arXiv 1902.03545)* embed tasks by probe-network Fisher information. Task distance tracks taxonomic distance and embedding norm tracks test error (Fig. 2); an asymmetric distance picks feature extractors (Fig. 3). It predicts transfer choice, not a drop.
- *Mayilvahanan et al. (ICLR 2024, arXiv 2310.09562)* use CLIP-embedding nearest-neighbour similarity between each test image and the training set. Figure 3 (left) is a binned curve, accuracy averaged over 0.05-wide similarity bins. Figure 3 (centre, right) is interventional: pruning the most similar training points ("near-pruning") hurts accuracy far more than random or far pruning. Their headline is a partial null: matching ImageNet's train–test similarity leaves CLIP's OOD performance high (abstract; Table 1).

**Predicting OOD performance, where distances lose.**
- *Guillory et al. (ICCV 2021, arXiv 2107.03315)* compare Fréchet distance, MMD, discriminator AUC and proxy A-distance against confidence-based predictors (AC, DoC) over 6 natural shifts × 12 ImageNet architectures. Calibrated on synthetic shifts, the traditional distances predict accuracy worse than the average-confidence baseline (Fig. 3; MMD's error overlaps it). The supplementary Figures 13 and 14 plot each measure against the accuracy gap, with a best-fit line and Pearson ρ.
- *Miller et al., "Accuracy on the line" (ICML 2021, PMLR 139, arXiv 2107.04649)* plot OOD accuracy against ID accuracy with probit axes. Every point is a model, not a domain. R² of the linear fit runs from 0.881 to 0.997 across pairs, with 39 to 1,060 models each (Table 1). Weak cases are some CIFAR-10-C corruptions and Camelyon17.
- *Baek et al., "Agreement-on-the-line" (NeurIPS 2022, arXiv 2206.13089)* show that ID-vs-OOD agreement between model pairs follows the same line when accuracy does (Fig. 1; slope, bias and R² in Table 1).
- Neither line-based paper uses a distance. Both need many models per shift, and we have 2 to 3 rows. Their relevance is a caution: across models, ID performance alone predicts OOD performance tightly, with no distance involved.

**Generative models.**
- *Kadkhodaie et al. (ICLR 2024 oral, arXiv 2310.02557)*: denoiser train and test PSNR converge as training-set size N reaches 10⁵ (Fig. 1), and models trained on disjoint subsets then produce nearly identical samples (Fig. 2). The axis is N, not distance to a held-out domain.
- *Somepalli et al. (arXiv 2212.03860; CVPR 2023 venue **unverified**)* plot per-class generation-to-training similarity (Fig. 6). This measures memorization, not quality on a new domain.
- *Nalisnick et al. (ICLR 2019, arXiv 1810.09136)*: Glow gives higher likelihood to MNIST than to its own FashionMNIST test set (Fig. 1). Model likelihood is a poor proxy for distance to training, which favours an external feature space.
- No generative-model paper found correlates a train-to-domain distance with generation quality across held-out domains.

| Paper | Field | Distance | Target metric | Statistic | n domains | Baseline / normalization | Where |
|---|---|---|---|---|---|---|---|
| Blitzer+ 2007 ACL | NLP DA | proxy A-distance | adaptation loss | none (labelled scatter) | 6 pairs | in-domain gold standard subtracted | Fig. 3 |
| Ben-David+ 2010 MLJ | DA theory | H∆H-divergence | target error bound | theorem | n/a | source error + λ | Thm 2 |
| Ganin+ 2016 JMLR (1505.07818) | DA | PAD | (PAD per representation) | none | domain pairs | — | Fig. 3 |
| Alvarez-Melis & Fusi 2020 NeurIPS (2002.02923) | transfer | OTDD (OT on features+labels) | relative drop in error vs target-only | ρ (type unverified) + p | 16; 11; ?; ? | target-only training | Figs. 6a,b, 7a,b |
| Cui+ 2018 CVPR (1806.06193) | transfer | exp(−γ·EMD) | fine-tune top-1 | none, line per target | 7 targets × 5 sources | — | Fig. 5, Table 7 |
| Deng & Zheng 2021 CVPR (2007.02915) | accuracy prediction | Fréchet distance (penultimate features) | accuracy | Spearman ≈ −0.91, robust linear fit | synthetic sets (count unverified) | — | Fig. 2 |
| Achille+ 2019 ICCV (1902.03545) | meta-learning | Task2Vec (Fisher) | taxonomic distance; test error | qualitative | iNat/CUB tasks | — | Figs. 2–3 |
| Guillory+ 2021 ICCV (2107.03315) | accuracy prediction | FD, MMD, PAD, disc. AUC vs DoC | accuracy gap | MAE of prediction; Pearson ρ (supp.) | 6 shifts × 12 archs | average confidence (AC) | Figs. 3, 13–14 |
| Miller+ 2021 ICML (2107.04649) | robustness | none (ID accuracy) | OOD accuracy | R² of probit-linear fit | points = models | y = x | Fig. 1, Table 1 |
| Baek+ 2022 NeurIPS (2206.13089) | robustness | none (agreement) | OOD accuracy | slope, bias, R² | points = models | — | Fig. 1, Table 1 |
| Mayilvahanan+ 2024 ICLR (2310.09562) | CLIP OOD | nearest-neighbour CLIP similarity | top-1 | binned mean; pruning intervention | per-sample bins | random pruning | Fig. 3, Table 1 |
| Kadkhodaie+ 2024 ICLR (2310.02557) | diffusion | training-set size N | train vs test PSNR | curves | n/a | — | Figs. 1–2 |

## Q2. Adjacent fields

**Driving.**
- No public paper found correlates a latent distance between driving scenes and the training set with planner or world-model performance. The NVIDIA method in the design doc has no public counterpart in this search.
- *Cosmos (NVIDIA, arXiv 2501.03575)* uses embeddings only for curation: semantic dedup by k-means (k = 10,000) on InternVideo2 embeddings, then within-cluster pairwise distances (§3.5).
- *Alpamayo-R1 (NVIDIA, arXiv 2511.00088)* targets long-tail scenarios. It contains no distance-versus-performance analysis (text searched).
- *GAIA-1 (arXiv 2309.17080)* evaluates scaling on a held-out geofenced validation set by cross-entropy (Fig. 8a).
- *GAIA-2 (arXiv 2503.20523)* holds out whole geographies (§3). It reports FDD, FID and FVMD (§6.2), with no distance axis.
- *Vista (NeurIPS 2024, arXiv 2405.17398)* runs a human 2AFC study over 60 scenes from 4 datasets including unseen Waymo (Fig. 7), and FVD with and without action conditions on nuScenes and Waymo (Fig. 8).
- *Wang et al., "Train in Germany, test in the USA" (arXiv 2005.08139; CVPR 2020 venue **unverified**)* train and test across 5 driving datasets. They trace most of the gap to one diagnosed factor, car-size statistics. This is a confound-first alternative to a scalar distance.

**Robotics.**
- *Xie et al. (arXiv 2307.03659, venue **unverified**)*: generalization gap (train minus test success) per factor (Fig. 5a), and against the radius of camera and table position ranges (Fig. 5b), a binned dose-response on a physical distance with SEM bars.
- *Burns et al. (arXiv 2312.12444, venue **unverified**)*: R² and Spearman ρ between representation metrics and OOD success over 15 models, with separate lines for ViTs and ResNets (§5.2, Fig. 5).
- *Lin et al. (ICLR 2025, arXiv 2410.18647)*: power laws of generalization in the number of training environments and objects, Pearson r (Fig. 5).
- *Gao et al., RADAR (RA-L 2026, arXiv 2603.11426)* retrieve each test task's nearest training examples in a generalist policy's embedding and have a VLM classify the generalization required. It is validated against human labels, not success rate, but it makes our space-(c) argument: policy embeddings see behaviour-relevant change that appearance misses.

**World models on unseen scenes or maps.**
- *GameNGen (ICLR 2025, arXiv 2408.14837)*: teacher-forced PSNR 29.43 and LPIPS 0.249 on 2,048 held-out trajectories from 5 levels (§5.1), FVD at 16 and 32 frames, and a human study. Out-of-distribution play is only qualitative, via edited frames (App. A.4). No held-out-map evaluation.
- *DIAMOND (NeurIPS 2024, arXiv 2405.12399)* is trained on a single map, CS:GO Dust II. Its only unseen-region evidence is qualitative: the model drifts in less-visited areas of that map (§5).
- *Genie (arXiv 2402.15391; ICML 2024 venue **unverified**)* prompts with OOD images (sketches, text-to-image) qualitatively. It reports FVD and ΔtPSNR on held-out data.
- *NWM (CVPR 2025, arXiv 2412.03572)* compares an unknown environment (Go Stanford) with a known one (RECON). It reports LPIPS, DreamSim and PSNR at 4 s, with and without unlabeled Ego4D (Table 4). A named failure mode in unknown environments: outputs drift toward training data (Fig. 10).
- *Matrix-Game (arXiv 2506.18701)*: scores across 8 Minecraft biomes against Oasis and MineWorld (§6.3, Fig. 9).
- *MultiGen (arXiv 2603.06679)*: a Doom model trained on 100 procedurally generated maps (§4.1), with SSIM, PSNR and LPIPS by rollout segment against GameNGen (Table 1). Whether Table 1's maps are held out is **unverified**.
- *SCOPE (arXiv 2605.23345)*: an FPS model with quality on 4 unseen styles against an in-distribution row (Table 3). It notes that the style closest to FPS scenes (sci-fi corridor) reaches near-parity, a one-sentence distance observation with n = 4.
- WorldMem (arXiv 2504.12369), Matrix-Game 2.0 (arXiv 2508.13009), MineWorld (arXiv 2504.08388), DriveDreamer (arXiv 2309.09777) and GameFactory (arXiv 2501.08325) contain no distance analysis (text searched). Genie 2 and 3 and Oasis have no papers.

**Game-level generalization.**
- *Procgen (ICML 2020, arXiv 1912.01588)*: train and test return against the number of training levels, 16 games (Fig. 2), with a per-game normalized return (§2.2).
- *Justesen et al. (NeurIPS 2018 Deep RL workshop, arXiv 1806.10729)*: PCA and DBSCAN over generated and human levels (Fig. 3, §7). Poor transfer to human levels is attributed to the generator covering a different region, qualitatively.
- *Dosovitskiy and Koltun (ICLR 2017, arXiv 1611.01779)*, the Doom precedent: a train-by-test table over D3, D4 and retextured versions with disjoint test textures (Table 2).

| Paper | Field | Distance | Target metric | Statistic | n domains | Baseline | Where |
|---|---|---|---|---|---|---|---|
| Cosmos 2025 (2501.03575) | driving/video WFM | InternVideo2 embedding k-means | (curation only) | — | — | — | §3.5 |
| GAIA-2 2025 (2503.20523) | driving WM | none (geo-holdout) | FDD, FID, FVMD | — | held-out geofences | — | §3, §6.2 |
| Vista 2024 NeurIPS (2405.17398) | driving WM | none (dataset identity) | FVD, human 2AFC, traj. difference | preference % | 4 datasets | action-free vs action-conditioned | Figs. 7–8, Table 2 |
| Wang+ 2020 (2005.08139) | 3D detection | car-size statistics | AP | table | 5 datasets | — | Tables (train × test) |
| Xie+ 2023 (2307.03659) | robot IL | factor shift radius | generalization gap | curves, SEM | 11 factors; radius bins | train success | Fig. 5a–b |
| Burns+ 2023 (2312.12444) | robot IL | representation metrics | OOD success | R², Spearman ρ | 15 models | — | Fig. 5 |
| Lin+ 2025 ICLR (2410.18647) | robot IL | # envs/objects | normalized score | Pearson r, power law | 1–32 envs | — | Fig. 5 |
| RADAR 2026 RA-L (2603.11426) | robot eval | policy-embedding NN retrieval | generalization class | agreement w/ humans | tasks | — | abstract, §III |
| GameNGen 2025 ICLR (2408.14837) | game WM (Doom) | none | PSNR, LPIPS, FVD, human | — | holdout trajectories, 5 levels | none | §5.1, Fig. 6 |
| NWM 2025 CVPR (2412.03572) | nav WM | none | LPIPS, DreamSim, PSNR @4 s | ± s.e. | 1 unknown + 1 known | known env | Table 4, Fig. 10 |
| Matrix-Game 2025 (2506.18701) | game WM | none | quality/control scores | — | 8 biomes | Oasis, MineWorld | §6.3, Fig. 9 |
| MultiGen 2026 (2603.06679) | game WM (Doom) | none | SSIM, PSNR, LPIPS by segment | — | 100 train maps (test maps unverified) | GameNGen | Table 1 |
| SCOPE 2026 (2605.23345) | FPS WM | style category (qualitative) | JEPA, LPIPS, flow, smoothness | — | 4 unseen styles | in-distribution row | Table 3 |
| Procgen 2020 ICML (1912.01588) | RL | # training levels | test return | curves | 16 games | normalized return | Fig. 2, §2.2 |
| Justesen+ 2018 (1806.10729) | RL | PCA+DBSCAN level space | score on human levels | qualitative | 4 games | PCG-level score | Fig. 3, Table 1 |
| Dosovitskiy & Koltun 2017 ICLR (1611.01779) | RL (Doom) | texture set disjointness | frags | train × test table | 4 envs | — | Table 2 |

## Q3. How the relationship was reported

- **Labelled scatter, no statistic:** Blitzer (n = 6), Cui (35 points, one line per target).
- **Scatter with ρ and p in the panel:** OTDD (n = 11 to 16, ±1 s.d. over 10 seeds), the closest template to our figure.
- **Scatter with Spearman and a robust line:** Deng and Zheng.
- **R² or Pearson with a line:** Miller and Baek (points are models), Guillory (Pearson, supplement), Burns (R² and Spearman per architecture family), Lin (log-log Pearson).
- **Binned curve:** Mayilvahanan, Xie Fig. 5b, Procgen.
- **Table by domain:** every world-model paper (NWM 2 environments, SCOPE 4, Vista 4, Matrix-Game 8).

**Confound control.** No study found runs a partial correlation. The controls used are:
- **Per-domain reference normalization:** Blitzer subtracts in-domain accuracy, OTDD divides by target-only error, Procgen and Atari normalize per game. Our gain over persistence is the same move.
- **Stratification:** Cui (per target), Burns (per architecture family), the analogue of our within-cluster ρ.
- **Intervention:** Mayilvahanan prunes by similarity; Xie varies one factor at a time.
- **Masking static content:** Mathieu scores only where optical flow exceeds a threshold, because most UCF101 pixels are still (Table 2). This is the direct precedent for motion weighting.
- **A named factor:** Wang et al. (car size).

Sequence length and scene difficulty are handled only through normalization, never statistically.

## Q4. Copy-last-frame baselines

Rohan's doubt holds for recent world models and fails for 2015 to 2019 video prediction.

| Paper | Name used for the baseline | Metric | Setting | Where |
|---|---|---|---|---|
| Ranzato+ 2014 (arXiv 1412.6604) | "1 frame (copy of previous)"; text: predicting the last frame | perplexity; relative MSE | UCF101 | Table 2, §3 |
| Mathieu, Couprie, LeCun 2016 ICLR (1511.05440) | "Last input" (text: "simple frame copy") | PSNR, SSIM, sharpness, moving areas only | Sports1m-trained, UCF101 test | Table 2; whole-image Tables 4–5 |
| Finn, Goodfellow, Levine 2016 NeurIPS (1605.07157) | text: copying the last observed ground-truth frame | PSNR, SSIM vs time step | robot pushing, seen and novel objects | §5.1, Fig. 3 (legend not extracted from PDF text) |
| Lotter, Kreiman, Cox 2017 ICLR, PredNet (1605.08104) | "Copy Last Frame" | MSE, SSIM; MSE vs steps | rotating faces; **trained on KITTI, tested on unseen CalTech Pedestrian** | Tables 1–2, appendix Fig. 7 |
| Byeon+ 2018 ECCV, ContextVP (1710.08518) | "Copy-Last-Frame" | MSE, PSNR, SSIM | CalTech (PSNR 23.3), Human3.6M (PSNR 32) | results tables |
| Reda+ 2018 ECCV, SDC-Net (1811.00684) | "CopyLast" | L1, L2, PSNR, SSIM | CalTech; YouTube-8M, where CopyLast's PSNR beats BeyondMSE and MCNet | Tables 1–2 |
| Villegas+ 2019 NeurIPS (1911.01655) | "Copy last frame" | PSNR, SSIM, LPIPS vs steps | towel pick, Human3.6M, KITTI; on Human3.6M it beats every model | appendix Figs. 8–10 (Fig. 9 caption) |
| Zhang+ 2026, ThinkJEPA (arXiv 2603.22281) | "Persistence" / copy-last (repeats last latent block) | latent FD/L2, SL1, CD | BAIR, in V-JEPA latent space | Table 3, App. A.3.4 |

**Not found** (text searched for copy, last frame, persistence and static): Oh et al. 2015 (NeurIPS, 1507.08750), whose baselines are an MLP and a no-action feedforward net and whose appendix notes that those two predict nearly the last input frame; VPN (1610.00527); SVG (1802.07687); MCnet (1706.08033); FitVid; MCVD; Chiappa et al.; SAVP; GameGAN; and every 2024 to 2026 world model in Q2 except ThinkJEPA.

Two related observations point the same way:
- Matrix-Game 2.0 attributes Oasis's higher scene-consistency and smoothness scores to Oasis producing static frames after collapse (§5, Minecraft results). A static output inflating a metric is exactly what a persistence reference exposes.
- Villegas et al. 2019 give the rationale in one line: "per-frame evaluations are not reliable when a large portion of a video does not move" (App. A.2.2).

Outside ML, MSE skill scores relative to a reference forecast are standard in meteorology (Murphy 1988, Monthly Weather Review 116:2417). Murphy's references there are climatological. Persistence as the usual short-range reference is **unverified** from that paper.

## Q5. Metrics for unseen-scene evaluation in world models

| Paper | Metrics | Horizon | Gain-over-baseline quantity? |
|---|---|---|---|
| GameNGen | PSNR, LPIPS (teacher-forced one step); PSNR/LPIPS over 64 AR steps; FVD; human | 1 frame; 64 steps; 16/32 frames; 1.6/3.2 s clips | no |
| NWM | LPIPS, DreamSim, PSNR; FID/FVD elsewhere | 4 s (Table 4); 1–16 s (Fig. 4) | in-domain vs +Ego4D rows only |
| Genie | FVD, **ΔtPSNR = PSNR(x, x̂ from inferred actions) − PSNR(x, x̂ from random actions)** | t = 4 | **yes**, a PSNR difference against a counterfactual baseline |
| Vista | FID, FVD, human 2AFC, trajectory difference via inverse dynamics | 25 frames | yes, FVD with vs without action conditioning (Fig. 8) |
| GAIA-2 | FDD, FID, FVMD | video | no |
| SCOPE | JEPA similarity, LPIPS, flow, photometric smoothness | 5 s clips | in-distribution reference row |
| MultiGen | SSIM, PSNR, LPIPS | segments 1–128 and 128–256 | no |
| Matrix-Game | image fidelity, aesthetics, smoothness, keyboard/mouse accuracy, physical consistency | video | no |
| DIAMOND (Atari) | human-normalized score; FID, FVD, LPIPS for CS:GO | episode; video | HNS normalizes by random and human scores |
| Procgen | normalized return | episode | normalized by per-game constants |

Our Δ = PSNR(model) − PSNR(persistence) has the same algebraic form as Genie's ΔtPSNR: a difference of two PSNRs, which equals 10·log₁₀ of an MSE ratio. The two differ only in the reference prediction, a frozen copy of the last frame for us and random-action generation for Genie. Cite Genie for the form and PredNet or Villegas 2019 for the reference.

## Q6. Summary: where our study stands

**Standard.**
- Correlating one distance per domain against one normalized performance number per domain is the Blitzer (2007), OTDD (2020), Cui (2018) and Deng and Zheng (2021) design.
- OT-family distances on learned features are established (OTDD, EMD). Sliced Wasserstein is a cheaper member of the same family and needs no new justification beyond cost and exact weights.
- Normalizing the outcome against a per-domain reference to remove intrinsic difficulty follows Blitzer's adaptation loss and OTDD's relative error drop.
- Copy-last-frame as the reference, and restricting evaluation to motion, are 2016 to 2019 video-prediction practice (Mathieu Table 2, PredNet, Villegas 2019).
- A PSNR difference against a counterfactual predictor is Genie's ΔtPSNR.

**Unusual (nothing found that does it).**
- Distance against quality for a generative world model across held-out maps; world-model papers stop at 1 to 8 domain tables.
- The minimum over training domains as the distance (closest precedent: Mayilvahanan's per-sample nearest neighbour).
- Motion weighting inside the distance (Mathieu masks motion in the metric only).
- Partial rank correlation with bootstrap CI, leave-one-out and within-cluster ρ; prior work prints one ρ and p (OTDD), one Spearman (Deng) or R² (Miller, Burns).
- A distance frozen before outcomes are read.
- n = 30 exceeds Blitzer (6) and OTDD (11 to 16) and is near Cui (35); only synthetic meta-datasets are larger (Deng).

**Conventions to mirror.**
1. **Figure:** one labelled marker per map, ρ with CI and p in the panel corner, y error bars from repeated draws (OTDD, Blitzer). Our planned figure already matches; a robust fitted line is optional (Deng).
2. **Statistic:** Spearman ρ with p (Deng, OTDD), plus our bootstrap CI and permutation p. Put Pearson or R² in the appendix for readers from the accuracy-on-the-line work.
3. **Strata:** report the two clusters separately, as Cui and Burns do.
4. **Confound:** name motion and cite Mathieu (moving-area evaluation) and Villegas 2019 (static background breaks per-frame metrics) for the persistence reference.
5. **Null framing:** Guillory (distances lose to confidence baselines) and Mayilvahanan (similarity does not explain OOD performance) are precedents for reporting a weak ρ.

**Closest prior works for the related-work paragraph.**
1. **Alvarez-Melis and Fusi, NeurIPS 2020 (arXiv 2002.02923):** an OT distance between datasets against difficulty-normalized transfer, with ρ and p per panel (Figs. 6–7). This is the nearest method and reporting match.
2. **Deng and Zheng, CVPR 2021 (arXiv 2007.02915):** Fréchet distance to the training set against accuracy, Spearman ρ ≈ −0.91 (Fig. 2). This is the nearest "distance predicts drop" claim, and the reason to show Fréchet only as a robustness check.
3. **Lotter et al., ICLR 2017 (PredNet, arXiv 1605.08104):** a video predictor trained on KITTI and evaluated on unseen CalTech against "Copy Last Frame" (Table 2, Fig. 7). This is the nearest precedent for persistence-referenced evaluation on an unseen domain. Pair it with Villegas et al. NeurIPS 2019 for the rationale and Blitzer et al. ACL 2007 for the normalized-outcome scatter if space allows.

## Sources (PDFs read)

arXiv: 2002.02923, 1902.03545, 2007.02915, 2107.03315, 2107.04649, 2206.13089, 2310.02557, 1806.06193, 2310.09562, 1505.07818, 2201.04234, 2212.03860, 1810.09136, 1511.05440, 1605.08104, 1507.08750, 1605.07157, 1610.00527, 1802.07687, 1706.08033, 1704.05831, 2106.13195, 2205.09853, 1412.6604, 1811.00684, 1502.04681, 1704.02254, 1911.01655, 1804.01523, 1710.08518, 2005.12126, 2408.14837, 2405.12399, 2402.15391, 2504.12369, 2506.18701, 2508.13009, 2412.03572, 2309.09777, 2309.17080, 2503.20523, 2405.17398, 2501.03575, 2511.00088, 1912.01588, 2006.13760, 1812.02341, 2501.08325, 2504.08388, 2605.23345, 2603.06679, 2506.05284, 2411.04983, 2508.06096, 1806.10729, 2111.09794, 1611.01779, 2603.11426, 2307.03659, 2312.12444, 2410.18647, 2403.05110, 2503.11062, 2504.04419, 2005.08139, 2005.13239, 2603.22281. Also ACL Anthology P07-1056; Ben-David et al. 2010 (alexkulesza.com/pubs/adapt_mlj10.pdf); OTDD NeurIPS proceedings PDF. Venues confirmed from arXiv comments, PDF headers, PMLR, CVF open access or the NeurIPS proceedings, except where marked unverified.
