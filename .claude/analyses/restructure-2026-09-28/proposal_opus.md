# Restructure proposal (Opus): DoomShift workshop paper

Base: `overleaf_draft_2026-09-27.tex`. Numbers: only the brief's final list, or arithmetic on it (differences and ratios of the listed medians). Where a kept Overleaf sentence carries a number that is not in the brief's list, it is marked **[verify]** and has a fallback.

How the tension is resolved, in one line: the paper is a study whose three findings are the headline. DoomShift is the instrument, so it is named first but described in one bullet and one data paragraph. Adaptation is a measurement with three dials (updates, episodes, parameters), which answers the title's question with a number instead of pitching a method. The excitement comes from what the study found, stated as claims: the model keeps the controls and loses the scene; the loss is identical across backbones and is a step; the fault is in the model, not the decoder; and very little adaptation brings the scene back.

---

## 1. Thesis in one sentence

Change only the scene, with the physics and game engine held fixed, and three Doom world models keep their response to the controls but lose the scene's appearance by the same amount, inside the model rather than its decoder, and a small adapter trained on eight episodes with 2 percent of the pretraining updates recovers most of the perceptual loss.

---

## 2. Title

**How Much Adaptation Do World Models Need? A Study of Domain Shifts in Doom** (Changliu's title, verbatim).

Reason: the question is the exciting half and the subtitle is the honest half, and the abstract and the intro's hook answer the question in two words ("very little"), so the paper reads as a result, not a survey. I considered "Scene Shifts" for "Domain Shifts" because it is more exact. I rejected it: "domain shift" is the term reviewers search for, it is her wording, and the first two abstract sentences already say that only the surroundings change. "DoomShift" stays as the benchmark's name in the abstract, bullet 1, Section 3 and the conclusion.

---

## 3. Outline for 4 pages

**How the lines were counted.** A CoRL page holds about 50 lines of body text. Every float is counted at its height (about 7 lines per inch at 10 pt) plus its caption and the float gap. I calibrated the counts against `scenarioF_preview.pdf`, which fills four pages to within about 2 lines. By the same count, Scenario F comes to about 198 lines. This outline comes to about 187, which leaves about 10 lines of reserve for real captions and paragraph gaps. The savings come from the Method section (48 lines in Scenario F, 31 here) and the table caption (9 lines, down to 4). They pay for a longer Results section (26 lines, up to 31) and the intro's hook.

| # | Section | Job | Lines | Carries |
|---|---|---|---|---|
| - | Title, authors, abstract (about 14 lines), keywords | State the question, the finding and the answer, with numbers in blue | 28 | - |
| 1 | Introduction | P1: world models are only scored in domain (6). P2: why a new scene matters, ending on the standing "We therefore explore ..." sentence (8). A two-line hook that gives the answer. Three bullets (20). | 38 | - |
| - | Figure 1 teaser | The finding in one picture: it still fires and turns but paints the training maps' masonry, and adaptation paints the arena back | 14 (1.5 in, 3-line caption, gap) | Figure 1 |
| 2 | Related work | What is held out elsewhere, the novelty sentence, and where the adapter comes from. Nothing the intro already says. | 10 | - |
| 3 | Study design (was "Method") | The instrument: arenas and data (5), three backbones under one recipe (7), measurements including the upper bound, the excess gap, the decoder and the directional score (10), and adaptation as a measurement (7) | 31 | - |
| - | Table 1 (the slim combined table, plus a Budget column) | One row per backbone: in domain, zero-shot, 4k, recovered shares, budget. Two rows compare LoRA with a full fine-tune. | 13 (4-line caption, 7-line body, gap) | Table 1 |
| 4 | Results | Four paragraphs, each titled with its claim: the scene breaks and the controls do not (6); the same loss in every backbone, and it is a step (4); the fault is in the model (3); how much adaptation it takes: updates, episodes, parameters, backbones (14); what we cannot forecast yet (2) | 31 | - |
| - | Figure 3 shift and repair (the 1.6 in body variant) | Every arena loses the scene and every arena recovers | 15 (1.6 in, 3-line caption, gap) | Figure 3 |
| 5 | Conclusion | Scope and limits, next steps, and Rohan's closing vision sentence | 7 | - |
| | **Total** | | **187 of about 200** | |

Placement: Figure 1 at the top of page 2, Table 1 at the top of page 3 or 4, Figure 3 at the top of page 4.

**Cut order if the build overflows.** Each step saves 1 to 2 lines. (1) The hardware clause in the full fine-tune sentence. (2) The data-ladder sentence, keeping only its pointer to the supplementary material. (3) The persistence range, keeping only "a median 19.8 dB". (4) Figure 1's second caption sentence. (5) The "What we cannot forecast" paragraph, folded into the Conclusion as one clause.

**Moves to the supplementary material**
- Figure 2 (method overview, 1.8 in) goes to a new first supplement section, "Study overview". It costs 15 lines, and a method diagram at the front is what makes a paper read as a dataset or method paper, which is the reading Rohan wants to avoid. Figure 1 and Section 3 carry the setup.
- The full Table 1 (six rows, with upper bound and directional columns) goes to the same section. Its SD 3.5 rows must be updated to 25.38 / 0.126 / 31.84 / 0.840 on the training maps and 22.09 / 0.262 / [verify: unseen upper bound] / 0.800 on the unseen arenas. The body keeps its directional numbers as text.
- Table 2 (adaptation cost by arena group).
- The "Training setup" paragraph: A6000 and A4000 memory, updates per second, and per-backbone throughput. Standing rule: no throughput claims in the body.
- From the Decoder paragraph: the argument that the latent space is the contract between encoder, model and decoder, the 3,486 updates, and the result that MSE alone triples LPIPS.
- The 2,412-episode total (96 hours, 12.1M frames), the release split 6,000 / 1,000 / 1,000, and the note that a bf16 EMA underflows.
- The per-arena ranges (20.5 to 24.7 dB, LPIPS 0.229 to 0.401, gap 3.7 to 6.9), the statistic that one frame in five exceeds LPIPS 0.4, the HUD rows falling 29.7 to 29.5 dB, the right-censoring definition, the transformers' adapter sizes and update grids, and the forgetting number (0.59 dB).
- Deleted, because it is stale: the four-tic sentence.

---

## 4. Contributions

**Order.** I keep Changliu's order: benchmark, then finding, then adaptation. The instrument has to be defined before the finding can be read, and the order is what she asked for. The finding still reaches the reader first, in two ways. The two-line hook just before the bullets states it. Bullet 1 is kept to five lines and framed as an instrument ("isolates"), while bullets 2 and 3 carry claim-shaped bold heads and the numbers.

Final text, including the lead-in, ready to paste:

```latex
In short, a world model moved to a new arena keeps playing the game but draws the wrong world, and a small adapter trained on eight episodes draws the arena back (Figure~\ref{fig:teaser}). We make three contributions:
\begin{itemize}\setlength{\itemsep}{0pt}
\item \textbf{DoomShift, a benchmark that isolates a scene shift.} Seventeen Doom arenas run on exactly the same physics and game engine and differ only in their surroundings. We train three diffusion world models, initialized from pretrained diffusion image models, on four arenas and score them on the 13 they never saw, zero-shot and after adaptation, against recorded frames and against each decoder's reconstruction upper bound, which separates the model's error from the renderer's. Recordings, models and adapters are public (link withheld).
\item \textbf{A scene shift breaks appearance, not control.} On the unseen arenas all three models keep their turn response (directional score 0.80 to 0.81, against 0.84 to 0.86 on the training maps) while their perceptual error roughly doubles (LPIPS 0.158 to a median 0.303 for the U-Net) and their scene PSNR drops 2.8 to 3.3~dB. The loss is the same for a U-Net and two transformers from 0.6 to 2.3B parameters, so it follows what the models were exposed to, not their architecture or scale. It is a step, not a slope: every unseen arena is worse in LPIPS than every training map. And it lives in the model, since the decoder renders the unseen arenas nearly as well as the training maps (Table~\ref{tab:main}, Figure~\ref{fig:teaser}).
\item \textbf{How much adaptation it takes: very little.} A standard LoRA adapter on the frozen model (0.49 percent of the U-Net's parameters), trained on eight episodes of the new arena, closes half of the extra distance the shift opened to the upper bound in a median 150 updates. By 4k updates, 2 percent of pretraining and about one A4000 GPU-hour, it closes a median 94 percent of that distance and recovers 72 percent of the lost perceptual quality. The same procedure recovers 76 and 77 percent on PixArt-$\alpha$ and SD~3.5. A full fine-tune with 205 times the trainable parameters recovers 87 percent against the adapter's 71 on four arenas (Table~\ref{tab:main}, Figure~\ref{fig:adapt}).
\end{itemize}
```

Number provenance: directional 0.800 to 0.808 unseen and 0.840 to 0.855 training; PSNR drops 25.20 − 22.30 = 2.90, 25.18 − 22.38 = 2.80 and 25.38 − 22.09 = 3.29; LPIPS ratios 1.92, 1.79 and 2.08 ("roughly doubles"). The full fine-tune figures (87 against 71) are the ones measured on arenas 6, 7, 8 and 16.

---

## 5. Delta against the Overleaf draft

Labels: **KEEP** means verbatim. **CUT** quotes the old text. **REWRITE** quotes the old text and gives the new text. **MOVE** gives the destination.

### Title
REWRITE `DoomShift: Efficient Adaptation of World Models to Domain Shifts` as `How Much Adaptation Do World Models Need?\\A Study of Domain Shifts in Doom`.

### Abstract (every number wrapped in `\prov{}`)
- KEEP S1: "World models have become capable simulators, but they are evaluated almost exclusively on held-out rollouts from the domain they were trained on."
- KEEP S2: "A small domain shift, such as a new arena of the same game, ... though the underlying physics and game engine are exactly the same."
- REWRITE S3. Old: "DoomShift asks how quickly and cheaply a world model can be adapted across such a shift." It reads as a method pitch, which is the reading comment 1 objects to. New: "With DoomShift, a controlled version of this shift in Doom, we measure what it breaks and how much adaptation it takes to repair."
- KEEP S4: "We train three diffusion backbones, initialized from pretrained diffusion image models, on four maps of Doom and score each on in-domain data and on 13 other out-of-distribution maps that it never saw."
- REWRITE S5. Old: "On the unseen maps the U-Net's perceptual error (LPIPS) doubles, ... its turn response survives, and the other two backbones lose the same way." New: "All three keep their response to the controls and lose the scene: the U-Net's perceptual error (LPIPS) doubles, from \prov{0.158} to a median \prov{0.303}, its scene PSNR drops \prov{2.9}~dB, from \prov{25.2} to a median \prov{22.3}~dB, and two transformers of \prov{0.6} and \prov{2.3}B parameters lose as much."
- REWRITE S6. Old: "We then train rank-16 adapters on 8 episodes of each unseen map with 2\% of the original training updates and recover 72\% of the lost perceptual quality and about half of the lost PSNR." New: "Very little adaptation repairs it: rank-16 adapters trained on \prov{8} episodes of each unseen map with \prov{2\%} of the original training updates recover \prov{72\%} of the lost perceptual quality and about \prov{half} of the lost PSNR on the U-Net, and \prov{76\%} and \prov{77\%} of the perceptual loss on the two transformers."
- KEEP the keywords: "World models, Few-shot adaptation".

### 1 Introduction
- P1. KEEP S1 and S2 (GameNGen, DIAMOND, Matrix-Game 2).
- P1. REWRITE S3. Old: "Each of these models is evaluated only on held-out rollouts from the environment it was trained on: the reported score is a held-out trajectory drawn from its own training scenes and none has been shown to generalize to a new environment within the same game." The old version says the same thing twice. New: "Each is evaluated only on held-out rollouts from the environment it was trained on, and none has been shown to generalize to a new environment within the same game."
- P1. REWRITE S4. Old: "This holds even for MultiGen, which trains a single Doom model across 100 procedurally generated maps with a pre-trained agent~\citep{po2026multigen}: it reports PSNR, SSIM and LPIPS only on its training distribution, with no map held out for evaluation." New: "Even MultiGen, one Doom model trained across 100 procedurally generated maps~\citep{po2026multigen}, reports PSNR, SSIM and LPIPS only on its training distribution, with no map held out."
- P2. KEEP S1 ("A learned simulator is useful only where it is faithful, ...") and S2 ("Game worlds let us pose this question with the engine held fixed ...").
- P2. CUT S3: "Yet unchanged dynamics alone do not guarantee a faithful render: pretrained video and world models are typically moved to a new domain only through a further full fine-tune~\citep{gao2025adaworld} or a dedicated adapter on the frozen backbone~\citep{rigter2024avid}, and a model trained on a handful of maps tends to fall back on the appearance it already knows once the surroundings change." There are two reasons. The citations repeat Section 2, which is comment 3. And the last clause states our main finding as something already known ("tends to fall back"), which gives the novelty away before the paper earns it.
- P2. REWRITE S4 (first half). Old: "Pretraining one model on every map a deployment might meet is costly and never complete, and fine-tuning a separate model per map is prohibitively costly." The old version says "costly" twice. New: "Pretraining one model on every map a deployment might meet is never complete, and training a separate model per map is prohibitively costly."
- P2. KEEP the last sentence verbatim (standing decision): "We therefore explore what it takes to close the domain gap ... and how few episodes and updates recover the rest."
- REWRITE the lead-in. Old: "We introduce DoomShift, which addresses these concerns and contributes the following:" New: the hook and the three bullets from Section 4 above.
- Figure 1: KEEP the figure and its caption verbatim (the standing 35-word caption).

### 2 Prior work, renamed "Related work"
REWRITE both paragraphs as one:

```latex
Held-out evaluation of world models exists, but it holds out something other than a map of the same game: a new robot body~\citep{chen2026xeworld}, a new game or environment~\citep{rigter2024avid,gao2025adaworld} or a new building~\citep{koh2021pathdreamer}. Doom maps have been held out, but only to test the agents that play them~\citep{lample2017arnold,wydmuch2018vizdoom}, and game world models that do show new scenes generate them from an image, with no recorded frames to score against~\citep{tong2026scope,bruce2024genie}. To our knowledge, no published game world model has been scored against recorded frames on maps of its own game that were deliberately held out from training. Our adapter is not new: a small adapter on a frozen video model moves it to a new setting in AVID~\citep{rigter2024avid} and adds action control in Vista~\citep{gao2024vista}, and AdaWorld adapts with a short full fine-tune~\citep{gao2025adaworld}, which we run as the comparator. None of them reports what adapting to one new scene costs in episodes and updates, which is what we measure.
```

Old sentences removed:
- CUT: "Most Doom and game world models are trained on a set of maps and tested on maps from that same set." It repeats intro P1.
- REWRITE: "We take our method from two papers that put a small adapter on a frozen pretrained video model." The phrase "our method" is what comment 2 objects to. It becomes "Our adapter is not new".
- CUT: "DoomShift combines what none of these do: maps kept aside from within a single game, with the adaptation budget measured directly." It repeats the novelty sentence and is folded into the last sentence.
- MOVE to Measurements (Section 3): the persistence sentence "Persistence, the last frame copied forward, is the standard reference of video prediction ..." and the reconstruction-upper-bound sentence, with their citations.

### 3 Method, renamed "Study design"
- **Data.** REWRITE the whole paragraph as: "\textbf{Arenas and data.} The Arnold agent~\citep{lample2017arnold} plays 150-second ViZDoom~\citep{kempka2016vizdoom} deathmatches against 8 bots on the 17 Doom maps it ships with, which we call arenas; each has its own layout, textures and lighting. We record every tic (1/35~s): a 320$\times$240 frame and the 19 buttons the engine executed. Four arenas (2 to 5) are the training maps: we train on 500 episodes of each and score on 25 further validation episodes per map (512 windows). The other 13 (1, 6 to 17) are never trained on: we record 24 new episodes on each, 16 for adaptation (8 used here) and 8 held out for scoring (256 windows per arena)."
  - CUT "We split the 17 arenas into two groups."
  - MOVE "In total we collect 2,412 episodes, about 96 hours of play and about 12.1 million frames." to the supplement. It is a dataset-paper sentence.
- **Backbones.** REWRITE both paragraphs as one: "\textbf{Three backbones, one recipe.} We turn three pretrained image diffusion models into next-tic world models: the SD~1.4 U-Net~\citep{rombach2022ldm} (860M parameters), GameNGen's backbone~\citep{valevski2024gamengen}; PixArt-$\alpha$~\citep{chen2023pixart}, a 628M transformer in the same latent space, which changes the architecture at similar scale; and SD~3.5 Medium~\citep{esser2024sd3}, a 2.27B MMDiT with 16-channel latents, which changes the scale. All three share one recipe (\appref{app:overview}): 32 context tics, $v$-prediction~\citep{salimans2022progressive}, context noise augmentation as in GameNGen, 200k updates at batch 32, an fp32 EMA at 0.9999, and 10-step DDIM~\citep{song2021ddim}, which we found to be a strong balance between quality and efficiency."
  - MOVE this result stated inside Method to Results: "architecture turns out not to change what survives the shift: in distribution their scene PSNR differs by 0.02~dB and their LPIPS by 0.001, and they lose the same amount on the unseen maps".
  - CUT "a step count we match across backbones", "(a bf16 EMA underflows and freezes)" (moved to the supplement), and "a widely adopted".
- **Metrics and Decoder.** Merge them into one paragraph. REWRITE: "\textbf{Measurements.} We score the scene rows (0 to 207; the HUD below does not change with the map) of each frame predicted one tic ahead against the raw recorded frame, with PSNR and LPIPS~\citep{zhang2018lpips}. Absolute PSNR alone does not compare arenas, because it rises with how static a recording is: copying the last frame forward, the persistence reference of video prediction~\citep{mathieu2016deep,lotter2017prednet,villegas2019fidelity}, already scores a median 19.8~dB on the unseen arenas (18.5 to 22.5 per arena). We therefore read each score against the \emph{reconstruction upper bound}~\citep{zheng2023occworld,karypidis2024dinoforesight}, the decoder applied to the ground-truth latent, which is the best a prediction rendered through that decoder can score. An arena's \emph{excess gap} is how much further below this bound the model sits there than on the training maps. Each decoder is fine-tuned on training-map frames only (MSE plus 0.1~LPIPS; \appref{app:protocol}), which changes what we can see, not what the model knows. The \emph{directional} score is the fraction of turning windows whose predicted view turns the other way when left and right are swapped in the newest control."
  - CUT "We score every predicted frame against four questions: ... when we mirror the controls?" It costs 3 lines, and the paragraph now answers the four questions directly.
  - MOVE the two decoder sentences that start "Every backbone is trained on latents from its frozen encoder, so the latent space is the contract ..." and "The decoder only renders: ..." to the supplement.
  - MOVE to Results as finding 3: "it reconstructs the unseen arenas nearly as well as the training maps (median 27.3 against 28.6~dB) while the model's prediction loses 2.9~dB, so the shift lives in the model, not the renderer".
  - MOVE to the supplement: "raises the scene reconstruction from 26.9 to 28.4~dB and lowers its LPIPS from 0.092 to 0.060" and "the LPIPS term is there because MSE alone more than triples scene LPIPS (0.092 to 0.339)".
  - CUT as stale: "\prov{SD~3.5 uses its own stock decoder}".
- **Figure 2.** MOVE to the supplement (reason in Section 3).
- **Training setup.** MOVE the whole paragraph to the supplement: "We train each backbone on a single RTX A6000 (49~GB) ... at about 1.0 updates per second."
- **In-distribution performance.** MOVE the numbers to the first Results paragraph and the Table, with SD 3.5 updated to 25.38 / 0.126 and a 6.46 dB gap. CUT as stale: "Four tics ahead they reach 21.33, 21.29 and \prov{21.59}~dB (stock decoders, full frame; \prov{SD~3.5 at 140k})."
- **Post-training.** REWRITE it under a new heading: "\textbf{Adaptation as a measurement.} To measure how much adaptation an arena needs, we keep the trained model frozen and train a standard adapter: rank-16 LoRA~\citep{hu2022lora} on every attention projection, with the control MLP, input projection and noise-bucket embedding trained in full~\citep{xie2023difffit} (4.2M parameters, 0.49 percent of the U-Net; the same components on the transformers, \appref{app:perarena-adapt}). It trains on 8 episodes of the arena, about 1 A4000 GPU-hour for 4k updates, and we score held-out episodes at update counts from 50 to 8k. We report 4k updates, 2 percent of pretraining, everywhere: it captures on average 96 percent of the 8k gain, and 8k serves as the check. An arena's \emph{budget} is the first update count at which it closes half of its excess gap. At the other end of the parameter axis we fine-tune every U-Net parameter on arenas 6, 7, 8 and 16 with the same episodes."
  - CUT "To move a trained model to a new arena we keep its weights frozen and train a small adapter ..." (the method-pitch wording), the grid listing, the control-check listing, "which is why the recipe costs 2 percent of the original training updates", the \tbd comparator clause, and the right-censoring definition (moved to the supplement; no arena is censored).
- **Table 1 (six rows).** MOVE to the supplement. REPLACE it in the body with the slim table (`tables/tuned/results_slim.tex`) under `\label{tab:main}`, with one added column, "Budget" (150, 150, 250$^\dagger$, blank, blank). New caption, 4 lines: "What the shift costs and what adaptation buys. Scene PSNR (dB) and LPIPS one tic ahead against the raw frame, each backbone through its own fine-tuned decoder: in domain (training maps, 512 windows), and zero-shot and after 4k adapter updates on eight episodes (medians over the 13 unseen arenas). Recovered: median per-arena share of the LPIPS rise, the PSNR drop and the excess gap undone at 4k. Budget: median updates to close half the excess gap ($^\dagger$the first point of SD~3.5's grid). Bottom rows: arenas 6, 7, 8 and 16 only."

### 4 Results
- CUT the opener: "We report what a model loses zero-shot, what post-training recovers and with what budget, and what predicts how far an arena gets."
- REWRITE "Zero-shot: where the deficit sits." as: "\textbf{The shift breaks the scene, not the controls.} In domain the three models sit 3.4, 3.4 and 6.5~dB below their decoders' upper bounds; SD~3.5, the best of the three, leaves the most room below its own sharper decoder (Table~\ref{tab:main}). On the 13 unseen arenas all three keep their turn response: the directional score goes from 0.855, 0.842 and 0.840 to 0.804, 0.808 and 0.800, against 0.885 and 0.892 for the ground-truth frames. They lose the scene: the U-Net's scene PSNR falls from 25.2 to a median 22.3~dB and its LPIPS rises from 0.158 to 0.303. On arena 7 it still fires and advances as in the ground truth but paints the training maps' masonry (Figure~\ref{fig:teaser})."
  - CUT "What does a model keep and what does it lose when it meets an arena it never saw?" (the rhetorical question), "(U-Net directional scores 0.73 to 0.86 per arena against 0.79 to 0.91 in distribution ...)" (moved to the supplement), and the per-arena ranges together with "Pooled over the 13 arenas, one predicted frame in five ... below 10 percent." (moved to the supplement).
- NEW paragraph: "\textbf{The same loss in every backbone, and a step.} Scene PSNR drops 2.9, 2.8 and 3.3~dB and LPIPS rises by 0.145, 0.126 and 0.136 for the U-Net, PixArt-$\alpha$ and SD~3.5: neither a transformer in place of the U-Net nor 2.6 times the parameters changes what the shift costs. The loss is a step, not a slope: for all three backbones every unseen arena has a worse LPIPS than every training map (\appref{app:distance})." This replaces the old "The three backbones lose the same things ... not of the architecture or the scale." and "It is a step, not a slope: ...".
- NEW paragraph: "\textbf{The fault is in the model, not the renderer.} The SD~1 decoder, fine-tuned on training maps only, reconstructs the unseen arenas nearly as well as the training maps (27.3 against 28.6~dB) while the prediction loses 2.9~dB, so the shift lives in what the model predicts." This replaces the old "The decoder is not the limit: ... (median 27.3 against 28.6~dB, LPIPS 0.05 to 0.09)."
- CUT the old closing sentence: "What the model keeps is how actions act on the world: ... while the map-independent HUD rows fall 0.2~dB (29.7 to 29.5)." Its first half is now in the first paragraph. The HUD numbers move to the supplement [verify before reuse].
- REWRITE "Post-training: what it takes." as: "\textbf{How much adaptation: updates, episodes, parameters.} Very little. Half of an arena's excess gap closes in a median 150 updates. By 4k updates on eight episodes the U-Net's median scene PSNR rises from 22.3 to 23.6~dB and its LPIPS falls from 0.303 to 0.210 (Figure~\ref{fig:adapt}). Per arena the adapter closes a median 94 percent of the excess gap, at least half on every arena [verify], and undoes a median 72 percent of the LPIPS rise. It regains only 48 percent of the lost PSNR, because the unseen arenas' upper bound is itself lower (27.3 against 28.6~dB). Data matters less than the first updates: on four arenas one episode already gives most of the gain of sixteen [verify] (\appref{app:perarena-adapt}). Parameters matter somewhat more: a full fine-tune of all U-Net parameters, 205 times the adapter's, takes arenas 6, 7, 8 and 16 from 21.71~dB and 0.292 to 23.11~dB and 0.182 at 4k, against the adapter's 22.88~dB and 0.209, recovering 87 rather than 71 percent of the LPIPS rise; it runs on an A6000 (0.83~h per arena) where the adapter fits a 16~GB A4000 (1.10~h). The procedure carries over unchanged: PixArt-$\alpha$ recovers 76 percent of its LPIPS rise and closes a median 101 percent of its excess gap with a median budget of 150 updates, and SD~3.5 recovers 77 percent (0.262 to 0.165) and closes 94 percent, crossing half at 250 updates, the first point of its grid."
  - CUT: "We next adapt a small subset of the U-Net's parameters on each unseen arena and ask what it takes to recover what was lost." "the 8k check adds little (23.7~dB, 0.204)" (Method already gives the 96 percent). The whole sentence "In domain the model sits 3.4~dB below ... (budgets below 4k \tbd{})." (replaced). "(over 100 percent on arenas 8, 9 and 10, ...)" (moved to the supplement). And the \tbd.
  - Fallbacks for the two [verify] clauses: drop "at least half on every arena" and keep the median; for the data ladder, drop the sentence and keep the \appref pointer in the Method.
- REWRITE the forgetting sentence. Old: "The adapters trade some in-distribution skill for it: the training maps' scene PSNR against the decoded ground truth falls on all 13 arenas at the 8k check (median 0.59~dB; stock decoder), while the directional score rises from 0.81 to 0.84." New, merged with the predictor sentence: "\textbf{What we cannot forecast yet.} Neither the model's own latent-space score nor a frame distance between an arena and the training maps predicts how far the arena gets (\appref{app:distance}), so an arena's budget has to be measured, not predicted. Because the backbone stays frozen, removing the adapter restores the in-domain model exactly; the adapted weights lose a little on the training maps (\appref{app:perarena-adapt})."
- Figure 3: use the 1.6 in variant and REWRITE its caption. Old: "On unseen arenas the U-Net loses appearance; eight episodes recover about half in PSNR and most in LPIPS. PSNR (a) and LPIPS (b) ... Fine-tuned decoder, raw frame." New: "The shift and its repair, per unseen arena (U-Net). (a) Scene PSNR and (b) LPIPS zero-shot (open) and after 4k adapter updates on eight episodes (filled), with each arena's reconstruction upper bound (ticks) and the training maps' level (dashed): every arena loses the scene and every arena recovers. (c, d) The same against updates, coloured by zero-shot gap to the upper bound (dark: hard)."
- Table 2: MOVE to the supplement.

### 5 Future work, renamed "Conclusion"
- CUT S1: "On unseen arenas a Doom world model keeps its turn response and its action effects and loses the arena's appearance; eight episodes and 4k adapter updates raise its median scene PSNR from 22.3 to 23.6~dB ... on every arena." It repeats Results and bullet 3 word for word.
- REWRITE S2 and S3 as one sentence: "The claims cover 13 arenas of one game, one agent, one-tic scoring and mostly one seed, and one-tic quality does not guarantee stable rollouts (\appref{app:closedloop})." CUT the stale SD 3.5 collapse counts: "SD~3.5, best at one tic ... not the failure mode".
- REWRITE S4 as: "Next are other games, detecting a shift from the model's own scores so that it can repair itself, closed-loop evaluation of the adapted models, and agents trained inside them."
- KEEP the last sentence verbatim (Rohan's vision): "The pattern we measure, control kept, appearance lost, most of it relearned from a few episodes, is what a deployed world model would need to detect and repair on its own, and DoomShift is a testbed for that loop."

---

## 6. Novelty paragraph

How the finished paper makes the claim: "Game world models have only ever been scored on the scenes they trained on. We hold a game's engine, physics and controls fixed, hold out whole maps of that game, and score three world models against recorded frames there. To our knowledge no published game world model has been scored this way. The controlled shift separates what a world model knows about how the world responds to actions from what it knows about how the world looks, and the two come apart. The response to the controls survives. Appearance is lost, by the same amount for a U-Net and two transformers from 0.6 to 2.3B parameters, as a step at the edge of the training maps, and inside the model rather than its decoder. The study then turns "adapt it" into numbers: half of the excess gap closes in a median 150 updates, and 2 percent of pretraining, 8 episodes and 0.49 percent of the parameters restore the model's in-domain distance to what its decoder can render." The claim is spread over the hook, bullet 2, the last sentence of Related work, and the claim-titled paragraphs in Results. It is never phrased as "first benchmark", which Rohan tried and reverted on Sep 27.

**The skeptical reviewer.** "Of course models degrade out of distribution, and of course LoRA fixes it. Nothing here is surprising." **The answer**, which the paper has to make visible without arguing: degradation is not the claim. The claim is the shape of the degradation, measured under a control nobody had set up. It hits appearance and spares control. It is the same size across architecture and scale. It is a step rather than a slope. It sits in the model rather than the renderer. GameNGen, DIAMOND and MultiGen never hold out a map, so none of this could have been seen. "LoRA works" is not the claim either. The claim is how little it takes, measured along updates, episodes and parameters. AVID, Vista and AdaWorld report no per-scene cost, and that cost is the number a deployment would budget against.

A second objection: "A turn-reversal check is not 'control' and certainly not physics." The answer is that the paper claims only the turn response plus the action effects visible in Figure 1, calls it "response to the controls", and never says the model learned physics. "Physics" appears only to describe the environment, which is exactly the same across arenas. The wording in Sections 4 and 5 already keeps to this.

This framing is also what Changliu's comments buy. They pre-empt the most likely reject line, "the adapter is just LoRA". Once the adapter is a measuring instrument, that line has nothing to attack, and the findings carry the paper.

---

## 7. Where the current-main attempt went right and wrong

**Right**
- It adopted the title, renamed "post-training" to "adaptation" everywhere, and reordered the bullets. These are the cheap, correct responses to comments 1 and 2.
- It extended the adaptation result to all three backbones (76 and 77 percent in the abstract and bullet 3). It also added the sentence explaining that SD 3.5 is best in domain yet furthest below its own decoder's bound. Both turn a single-model anecdote into a finding about world models.
- It moved the persistence and upper-bound citations out of Prior work into Metrics. Scenario F then tightened intro P1 correctly. This proposal keeps both.

**Wrong**
- Condensing Section 2 deleted the paper's only explicit novelty sentence ("To our knowledge, no published game world model has been scored against recorded frames on maps of its own game that were deliberately held out from training"). It also dropped the "our adapter comes from AVID and Vista" attribution, which is the sentence that best answers comment 2. The cut followed "Section 2 repeats Section 1" literally and did not ask which sentences only Section 2 said.
- It relabelled instead of reframing. The abstract still says "DoomShift asks how quickly and cheaply a world model can be adapted". Bullet 1 still reads like a release note (the 6,000 / 1,000 / 1,000 split, "the adaptation path"). The lead-in copies the Slack message ("a benchmark, an empirical finding and an adaptation result"). The Results keep neutral heads ("Zero-shot: where the deficit sits"). So the study framing lives only in the title, and the finding never reads as the headline. It also kept the Overleaf intro clause that presents the finding as already known ("tends to fall back on the appearance it already knows").
- It grew the body before cutting it: per-backbone adapter sizes and grids, adapter throughput on three GPU types, and a 9-line full fine-tune paragraph, which pushed the body to page 7. Scenario F's mechanical trim then removed Rohan's closing vision sentence and the headline sentence of Figure 3's caption, and kept a 9-line table caption. Bullet 3 also says the adapter "trails a full fine-tune ... by only 0.2 to 0.3 dB" and leaves out the larger LPIPS difference (71 against 87 percent). A reviewer who reads the table will call that selective.

---

## 8. Risks

- **Figure 2 leaves the body.** Rohan's Sep 27 figure plan had a method overview in the body, and a reviewer may find the setup harder to picture. The mitigation is Figure 1's column labels, the one-paragraph Study design, and the supplement pointer. If Rohan insists on keeping it, the cost is about 15 lines, paid by the "What we cannot forecast" paragraph, the data-ladder sentence, the full fine-tune hardware clause and about 8 lines of Measurements. That is possible but leaves no reserve.
- **Claim-titled paragraphs can overclaim.** "Follows what the models were exposed to", "lives in the model" and "a step" each rest on three backbones, mostly one seed and 13 arenas. Each head must sit next to its numbers in the same paragraph. The "exposure, not architecture" inference is a reading of three runs, not a causal test. If Changliu balks, soften it to "does not depend on architecture or scale across our three backbones".
- **The measurement framing exposes the weak number.** 48 percent of the lost PSNR sounds poor unless the lower-upper-bound explanation appears in the same sentence. If an edit separates the two, the adaptation result reads weaker than it did in the Overleaf draft.
- **Changliu may feel her benchmark bullet was demoted.** It is still first and still named DoomShift, but it is five lines and framed as an instrument. Say so when sending: "the benchmark is first, described as the controlled shift that makes the finding readable".
- **The line budget is estimated, not built.** The reserve is about 10 lines, calibrated on one PDF. The real Figure 3 caption and the table's added column could eat it. The cut order in Section 3 is the fallback, and nothing in it touches a finding.
- **Numbers outside the brief's list survive in three places:** "at least half on every arena", the one-episode ladder, and "the adapted weights lose a little". Each has a fallback above. The supplement's full Table 1 needs SD 3.5's unseen upper bound, which the brief does not give.
- **Rewriting on deadline day invites drift** between the abstract, bullets, table and text. After the edit, run one number-consistency pass: 0.158 / 0.303, 2.9 dB, 72 / 76 / 77, 150 / 250, 94 / 101, 205×, 87 / 71.
