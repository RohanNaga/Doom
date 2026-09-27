# Citation check, Opus pass (2026-09-27)

Scope: the 29 keys cited in `paper/main.tex` and `paper/appendix.tex` (the 15 uncited bib entries are out of scope). Each key was checked independently of Astra's audit (`citation-check-2026-09-27.md`) against three things: the primary record, a citation count, and the sentence that cites it.

Sources used:
- arXiv API batch (`export.arxiv.org/api/query?id_list=...`) for title, authors, first-version date, comment, journal-ref and DOI.
- Venue records: Crossref `api.crossref.org/works/<doi>`, PMLR pages (v235, v267 index), the NeurIPS proceedings indexes for 2019 and 2024, the NeurIPS 2025 virtual paper list, the ICLR virtual paper lists for 2021, 2022, 2024 and 2025, the ICLR 2016/2017 archive pages, the RLJ paper page, and the ECCV 2026 accepted-papers pages on `eccv.ecva.net`.
- Citation counts: Semantic Scholar batch API (`api.semanticscholar.org/graph/v1/paper/batch`, 2026-09-27). Two keys failed S2 twice. For those two, the OpenAlex title search gives a lower bound, marked "OA"; OpenAlex undercounts arXiv-heavy ML papers by roughly 10x, so it is a floor only.
- Usage: each citing sentence was read against the cited paper's full text (arXiv HTML for post-Dec-2023 papers, arXiv/JMLR PDF via pdftotext otherwise).

DBLP was tried for venue records and refused every request (HTTP 429, then connection resets), so no DBLP record is used.

## Table

| Key | Title | First author | Year | Venue (confirmed record) | ID / DOI | Cites (S2) | Status | Note |
|---|---|---|---|---|---|---|---|---|
| valevski2024gamengen | Diffusion Models Are Real-Time Game Engines | Dani Valevski | 2025 | ICLR 2025 (iclr.cc/virtual/2025/poster/29770) | arXiv:2408.14837 | 300 | VERIFIED | Used at l.52 (Doom in a diffusion model) and l.94 (context noise augmentation, their sec. 3.2.1): both match. |
| alonso2024diamond | Diffusion for World Modeling: Visual Details Matter in Atari | Eloi Alonso | 2024 | NeurIPS 2024 main track (proceedings index) | arXiv:2405.12399 | 336 | VERIFIED | The CS:GO section exists but is marked as added after NeurIPS acceptance, so "Atari and Counter-Strike" describes the arXiv v2, not the proceedings paper. Harmless. |
| he2025matrixgame2 | Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model | Xianglong He | 2025 | arXiv only (no venue in v4, 2026-04-07) | arXiv:2508.13009 | 157 | VERIFIED | Widely cited technical report. Its data are Unreal scenes, GTA5, Minecraft and Temple Run, so "open worlds and many games" fits it. v4 retitled to sentence case; no change needed. |
| po2026multigen | MultiGen: Level-Design for Editable Multiplayer Worlds in Diffusion Game Engines | Ryan Po | 2026 | arXiv only (v2 2026-03-30, no venue) | arXiv:2603.06679 | 9 | MISUSED | l.52 says MultiGen scales the recipe "to open worlds and many games". MultiGen is Doom only. It trains on 100 procedurally generated Obsidian maps and runs ViZDoom deathmatch with the Lample and Chaplot (Arnold) agent against four copies. It is the nearest prior work, and l.52 mischaracterizes it. No held-out map split is stated (no "held-out", "unseen" or "test split" in the text), so l.53 ("none is shown to generalize to a new environment of the same game") survives. The paper should still name MultiGen explicitly there. |
| chen2026xeworld | XEWorld: Can Action-Conditioned World Models Generalize to Unseen Robot Embodiments? | Yixiang Chen | 2026 | arXiv only (v1 2026-08-06, no comment or venue) | arXiv:2608.05799 | 0 | TOO NEW | Posted 7 weeks before today and not vetted. Usage at l.81 (a held-out-embodiment protocol, their sec. 3.1) matches. |
| rigter2024avid | AVID: Adapting Video Diffusion Models to World Models | Marc Rigter | 2025 | RLC 2025; Reinforcement Learning Journal 6:737-764 (rlj.cs.umass.edu/2025/papers/Paper64.html) | arXiv:2410.12822 | 39 | VERIFIED | l.81 "a new game": the base model is pretrained on 15 Procgen games and the adapter is trained on the held-out 16th (Coinrun). Matches. AVID is the paper that trains a small adapter on a frozen pretrained model, so it belongs in l.82 too (see below). |
| gao2025adaworld | AdaWorld: Learning Adaptable World Models with Latent Actions | Shenyuan Gao | 2025 | ICML 2025, PMLR 267:18744-18771 | arXiv:2503.18938 | 118 | MISUSED | l.81 (new environments: Habitat, Minecraft, DMLab and nuScenes, all absent from pretraining) is correct. l.82 says AdaWorld adapts "with a small adapter". It does not: AdaWorld fine-tunes the whole model for 800 steps (pretrained weights at 0.1x learning rate, sec. 3.2.1) after initializing the action embeddings. The paper never mentions LoRA or an adapter. |
| koh2021pathdreamer | Pathdreamer: A World Model for Indoor Navigation | Jing Yu Koh | 2021 | ICCV 2021, pp. 14718-14728 (Crossref 10.1109/ICCV48922.2021.01447) | arXiv:2105.08756 | 148 | VERIFIED | Usage (a new building) matches. A DOI could be added. |
| bruce2024genie | Genie: Generative Interactive Environments | Jake Bruce | 2024 | ICML 2024, PMLR 235:4603-4623 | arXiv:2402.15391 | 829 | MISUSED (+FIX) | l.83 says "Genie reports the same paired ΔPSNR" inside the persistence sentence. Genie's Δ_tPSNR subtracts the PSNR of a rollout driven by *random latent actions* from the PSNR of one driven by inferred actions (sec. 3, Metrics). It is a controllability metric with no copied-frame reference. l.106 ("a paired difference like Genie's ΔPSNR") is accurate. The bib is also stale: it gives arXiv, but the paper appeared at ICML. |
| gao2024vista | Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability | Shenyuan Gao | 2024 | NeurIPS 2024 main track | arXiv:2405.17398 | 389 | MISUSED (l.82 only) | l.100 is correct: Vista freezes the U-Net and adds rank-16 LoRA and projections to all attention blocks (App. C.3). l.82 says a small adapter is how Vista "reach[es] new driving scenes". It is not. Vista's LoRA phase learns action controllability. Its generalization to unseen scenes comes from large-scale pretraining and fine-tuning of SVD, and it is shown zero-shot. |
| lotter2017prednet | Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning | William Lotter | 2017 | ICLR 2017 (conference poster C10, iclr.cc archive) | arXiv:1605.08104 | 1020 | VERIFIED | The "Copy Last Frame" row in the KITTI-to-CalTech tables matches the persistence claim. |
| villegas2019fidelity | High Fidelity Video Prediction with Large Stochastic Recurrent Neural Networks | Ruben Villegas | 2019 | NeurIPS 2019 (proceedings index) | arXiv:1911.01655 | 153 | VERIFIED | "Copy last frame" curves appear in the appendix figures. Matches. |
| mathieu2016deep | Deep Multi-Scale Video Prediction beyond Mean Square Error | Michael Mathieu | 2016 | ICLR 2016 (accepted-main list) | arXiv:1511.05440 | 2028 | VERIFIED | The "Last input" row appears in Tables 1, 4 and 5, and the text says copying the last input wins on static pixels. Matches. Title case differs only cosmetically. |
| zheng2023occworld | OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving | Wenzhao Zheng | 2024 | ECCV 2024, LNCS pp. 55-72 | arXiv:2311.16038; 10.1007/978-3-031-72624-8_4 | 296 | FIX | Usage matches: in Table 1 the 0 s column is reconstruction accuracy, next to the Copy&Paste row. The bib gives arXiv 2023 but the paper is published at ECCV 2024. |
| karypidis2024dinoforesight | DINO-Foresight: Looking into the Future with DINO | Efstathios Karypidis | 2025 | NeurIPS 2025 (neurips.cc/virtual/2025/poster/116713) | arXiv:2412.11673 | 45 | VERIFIED | The Oracle row (task head on the true future features, called "an upper performance bound") and the Copy-Last row are in Table 2. Matches. |
| mensink2021factors | Factors of Influence for Transfer Learning across Diverse Appearance Domains and Task Types | Thomas Mensink | 2022 | IEEE TPAMI 44(12):9298-9314 | arXiv:2103.13318; 10.1109/TPAMI.2021.3129870 | 102 | FIX | Usage matches: in Table 10 the directed inclusion measure (target to closest source) has τ 0.42 and EMD has τ 0.20. The bib still reads "arXiv; accepted to TPAMI", 2021. |
| westny2026latent | Unveiling Transferability in Trajectory Prediction via Latent Scene Embeddings | Theodor Westny | 2026 | ECCV 2026: listed on eccv.ecva.net/virtual/2026/papers.html and /Conferences/2026/AcceptedPapers | arXiv:2606.30777 | 0 | VERIFIED | Posted 2026-06-29, right at the three-month line, but the venue is confirmed on ECVA's own list, so it is not TOO NEW. It proposes a directional KL measure against symmetric ones including Wasserstein (Table 2), which matches the appendix sentence. The table values sit in MathML and were not re-extracted. |
| hu2022lora | LoRA: Low-Rank Adaptation of Large Language Models | Edward J. Hu | 2022 | ICLR 2022 (iclr.cc/virtual/2022/poster/6319) | arXiv:2106.09685 | 23322 | VERIFIED | Method reference. |
| xie2023difffit | DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter-Efficient Fine-Tuning | Enze Xie | 2023 | ICCV 2023, pp. 4207-4216 (Crossref 10.1109/ICCV51070.2023.00390) | arXiv:2304.06648 | 105 | VERIFIED | The IEEE title matches the bib exactly, including "Parameter-Efficient". On usage: DiffFit freezes the backbone and trains the bias terms, norms, scale factors and the class-embedding (condition) module. That is a fair precedent for training the control MLP and embeddings in full, though it is not LoRA. |
| lample2017arnold | Playing FPS Games with Deep Reinforcement Learning | Guillaume Lample | 2017 | AAAI 2017, vol. 31 no. 1 (Crossref 10.1609/aaai.v31i1.10827) | arXiv:1609.05521 | 636 | VERIFIED | "Full deathmatch on unknown maps" (10 train and 3 test maps) matches l.81. l.91 (Arnold's level pack) is a data reference. |
| wydmuch2018vizdoom | ViZDoom Competitions: Playing Doom from Pixels | Marek Wydmuch | 2019 | IEEE Transactions on Games 11(3):248-259 | arXiv:1809.03470; 10.1109/TG.2018.2877047 | 144 | FIX | Usage matches: Track 2 is "Full Deathmatch on Unknown Maps". The bib gives arXiv 2018, but arXiv's own record carries this DOI. |
| kempka2016vizdoom | ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning | Michał Kempka | 2016 | IEEE CIG 2016, pp. 1-8 (Crossref 10.1109/CIG.2016.7860433) | arXiv:1605.02097 | 761 | VERIFIED | Engine reference. A DOI could be added. |
| rombach2022ldm | High-Resolution Image Synthesis with Latent Diffusion Models | Robin Rombach | 2022 | CVPR 2022, pp. 10674-10685 (Crossref 10.1109/CVPR52688.2022.01042) | arXiv:2112.10752 | 27563 | VERIFIED | The SD 1.x U-Net and autoencoder. |
| chen2023pixart | PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis | Junsong Chen | 2024 | ICLR 2024 (iclr.cc/virtual/2024/poster/18231) | arXiv:2310.00426 | 967 | FIX | The ICLR record lists 10 authors and omits Yue Wu; arXiv v3 lists 11. Pick one version and match its author list. |
| esser2024sd3 | Scaling Rectified Flow Transformers for High-Resolution Image Synthesis | Patrick Esser | 2024 | ICML 2024, PMLR 235:12606-12633 | arXiv:2403.03206 | 4751 | FIX | PMLR lists 14 authors. Lacey, Goodwin and Marek appear only on arXiv v1. The paper describes SD 3, not SD 3.5 Medium (MMDiT-X), and l.94 uses it as the family reference, which is acceptable. |
| salimans2022progressive | Progressive Distillation for Fast Sampling of Diffusion Models | Tim Salimans | 2022 | ICLR 2022 (iclr.cc/virtual/2022/poster/6537) | arXiv:2202.00512 | S2 failed twice; OA ≥193 | VERIFIED | Source of the v-prediction parameterization. S2 by ID returned null, and the title search 429'd. Tried: `api.semanticscholar.org/graph/v1/paper/batch` and `/paper/search?query=Progressive+Distillation...`. |
| song2021ddim | Denoising Diffusion Implicit Models | Jiaming Song | 2021 | ICLR 2021 (iclr.cc/virtual/2021/poster/2804) | arXiv:2010.02502 | 13566 | VERIFIED | The repo's `diffusion_v.py` / `eval_tf.py` sample with DDIM, so the l.94 claim matches the code. |
| zhang2018lpips | The Unreasonable Effectiveness of Deep Features as a Perceptual Metric | Richard Zhang | 2018 | CVPR 2018, pp. 586-595 (Crossref 10.1109/CVPR.2018.00068) | arXiv:1801.03924 | 20031 | VERIFIED | Metric reference. |
| taylor2009transfer | Transfer Learning for Reinforcement Learning Domains: A Survey | Matthew E. Taylor | 2009 | JMLR 10:1633-1685 (header of jmlr.org PDF) | jmlr.org/papers/v10/taylor09a.html | S2 429 twice; OA ≥1552 | VERIFIED | Sec. 2.1 defines "time to threshold" as the learning time to reach a pre-specified performance level. It says nothing about right-censoring, yet the cite sits after "right-censored" at l.102. Move it to follow the budget definition. |

Verdict (29 cited keys): **19 VERIFIED, 5 FIX, 4 MISUSED, 1 TOO NEW, 0 METADATA ONLY, 0 NOT FOUND, 0 UNRESOLVED.** Every key is a real paper with correct authors. The two unresolved S2 counts belong to well-known papers and do not affect any verdict.

## Exact bib corrections (old to new)

`bruce2024genie`
- entry type: `@article` to `@inproceedings`
- journal: `{arXiv preprint arXiv:2402.15391}` to removed
- booktitle: absent to `{International Conference on Machine Learning (ICML)}`
- add `series = {PMLR}`, `volume = {235}`, `pages = {4603--4623}`

`zheng2023occworld`
- entry type: `@article` to `@inproceedings`
- journal: `{arXiv preprint arXiv:2311.16038}` to removed
- booktitle: absent to `{European Conference on Computer Vision (ECCV)}`
- year: `{2023}` to `{2024}`
- add `pages = {55--72}`, `doi = {10.1007/978-3-031-72624-8_4}`

`mensink2021factors`
- journal: `{arXiv preprint arXiv:2103.13318; accepted to IEEE Transactions on Pattern Analysis and Machine Intelligence}` to `{IEEE Transactions on Pattern Analysis and Machine Intelligence}`
- year: `{2021}` to `{2022}`
- add `volume = {44}`, `number = {12}`, `pages = {9298--9314}`, `doi = {10.1109/TPAMI.2021.3129870}`

`wydmuch2018vizdoom`
- journal: `{arXiv preprint arXiv:1809.03470}` to `{IEEE Transactions on Games}`
- year: `{2018}` to `{2019}`
- add `volume = {11}`, `number = {3}`, `pages = {248--259}`, `doi = {10.1109/TG.2018.2877047}`

`chen2023pixart` (option A, recommended: cite the venue version)
- entry type: `@article` to `@inproceedings`
- journal: `{arXiv preprint arXiv:2310.00426}` to removed
- booktitle: absent to `{International Conference on Learning Representations (ICLR)}`
- year: `{2023}` to `{2024}`
- author: remove `Wu, Yue and ` (the ICLR record lists 10 authors)
- Option B: keep the arXiv entry and 11 authors unchanged. Either is correct. Mixing the two is not.

`esser2024sd3`
- entry type: `@article` to `@inproceedings`
- journal: `{arXiv preprint arXiv:2403.03206}` to removed
- booktitle: absent to `{International Conference on Machine Learning (ICML)}`
- author: remove `Lacey, Kyle and Goodwin, Alex and Marek, Yannik and ` (PMLR lists 14 authors)
- add `series = {PMLR}`, `volume = {235}`, `pages = {12606--12633}`

Optional, not errors:
- `rigter2024avid`: add `note = {Reinforcement Learning Journal 6:737--764}`.
- `gao2025adaworld`: add `volume = {267}`, `pages = {18744--18771}`.
- DOIs for `koh2021pathdreamer` (10.1109/ICCV48922.2021.01447), `kempka2016vizdoom` (10.1109/CIG.2016.7860433), `rombach2022ldm` (10.1109/CVPR52688.2022.01042), `zhang2018lpips` (10.1109/CVPR.2018.00068), `xie2023difffit` (10.1109/ICCV51070.2023.00390), `lample2017arnold` (10.1609/aaai.v31i1.10827).

No change is needed for the `he2025matrixgame2`, `mathieu2016deep` or `xie2023difffit` titles (see disagreements below).

## Keys to remove, replace or re-sentence

No key needs removal for being fake. The four MISUSED keys need sentence fixes, and one key needs a recency decision.

1. **`po2026multigen`, l.52.** Drop it from the "open worlds and many games" clause, where Matrix-Game 2 alone fits. Describe MultiGen for what it is: one Doom model trained across 100 procedurally generated maps with the same ViZDoom deathmatch plus Arnold setup, conditioned on an editable map memory, and evaluated without a stated held-out map. Then name it in l.53 or l.81 as the closest prior work. A reviewer who knows MultiGen will read the current sentence as either not having read it or hiding the nearest neighbour.
2. **`bruce2024genie`, l.83.** Remove the cite from the persistence sentence, or reword it, for example: "Genie reports a paired ΔPSNR of the same form, against random actions rather than a copied frame." Keep l.106 as it is.
3. **`gao2024vista` and `gao2025adaworld`, l.82.** Rewrite the sentence and add `rigter2024avid`, which does train a small adapter on a frozen pretrained video model and evaluates on a held-out game. For example: "Pretrained video models reach a new domain through a small adapter on a frozen backbone (AVID) or a short full fine-tune (AdaWorld), and Vista adds control to a frozen U-Net with rank-16 LoRA; we measure what that adaptation costs per scene."
4. **`chen2026xeworld`, TOO NEW.** Posted 2026-08-06 with no venue and 0 citations. Either keep it and flag it as concurrent ("concurrent work, arXiv 2026"), or drop it. l.81 still stands on AVID, AdaWorld and Pathdreamer for "new environment" holdouts. I would keep it as concurrent: it is the only held-out-embodiment benchmark in the list, and the sentence only states what its protocol is.
5. **`taylor2009transfer`, l.102.** Move the cite to right after "half of its gap from its zero-shot A_0". Taylor and Stone support time to threshold, not right-censoring.

## Where I disagree with Astra's audit

- **Usage was not checked.** Astra verified metadata only and marked `po2026multigen` UNPUBLISHED and `gao2024vista`, `gao2025adaworld` and `bruce2024genie` VERIFIED or FIX. The four sentence-level mismatches above are the findings most likely to draw a reviewer's objection, and the audit misses all of them.
- **Case-only title FIXes are not errors.** `he2025matrixgame2`: Astra wants the v4 sentence-case title, but the bib matches v1, the style re-cases titles anyway, and brace-protected names already hold. `mathieu2016deep`: the bib's title case matches the ICLR accepted list's capitalization. `xie2023difffit`: Astra cites the CVF HTML slug ("Parameter-efficient"), but the IEEE/Crossref record for the same ICCV paper reads "Parameter-Efficient", which is exactly the bib. I count all three as VERIFIED.
- **Recency.** Astra lists `chen2026xeworld` as UNPUBLISHED without noting that it is seven weeks old. Under the venue's standard it is the one entry not yet vetted.
- **`westny2026latent`.** Astra took ECCV 2026 from the arXiv comment. I confirmed it on ECVA's accepted-papers pages. Same verdict, stronger evidence.
- **Agreements.** Venue updates for Genie, OccWorld, Mensink and Wydmuch, and the PixArt (10 vs 11) and SD3 (14 vs 17) author-list version splits, match what I found in the primary records.
