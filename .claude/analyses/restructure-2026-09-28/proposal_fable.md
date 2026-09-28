# Proposal (Fable): restructure of the DoomShift workshop paper

Base: `overleaf_draft_2026-09-27.tex`. Every delta below is against that file. Numbers come only from the brief's final table; where a sentence carries a number from the draft that the brief does not list, it is marked `[draft number, verify]`.

How the tension is resolved, in one line: Changliu's order and nouns (benchmark, finding, adaptation result) stay exactly as she wrote them; Rohan's excitement goes into the weight (the benchmark bullet is the shortest, four lines; the finding and the measurement get six each), into the bolded heads of bullets 2 and 3, which are the sentences a reviewer will repeat, and into the abstract, which answers the title's question with a number before the reader reaches the contributions. The benchmark is described as the instrument that makes the finding measurable, never as a release.

---

## 1. Thesis in one sentence

A world model that meets a new scene of the same game loses the scene's appearance and keeps its response to the controls, identically across three backbones from 0.6 to 2.3 B parameters, and gets the appearance back from eight episodes and 2% of its training updates on 0.5% of its parameters, with the median map halfway repaired after 150 updates.

Reviewer-length version: "Scene shift breaks appearance, not control, and costs 2% of training to repair."

---

## 2. Title

**How Much Adaptation Do World Models Need? A Study of Domain Shifts in Doom** (Changliu's title, verbatim).

Reason: a question title is the most novelty-forward form available here, because the paper answers it with a number in the abstract's fifth sentence; "A Study" is what Changliu asked for and costs nothing, since the novelty lives in the answer, not in the word "method". "DoomShift" stays as the benchmark's name in the abstract, bullet 1 and Section 3.

If Rohan wants the finding in the title, the only variant I would defend is the subtitle "Scene Shifts in Doom Break Appearance, Not Control"; I do not recommend it, because Changliu proposed the title and Rohan accepted it in writing, and a workshop reviewer reads the abstract anyway.

---

## 3. Outline for 4 pages

A CoRL page is about 50 lines of body text; the budget is 200 lines, floats included, and the scenario F preview shows what that means in practice: F fits exactly four pages with 106 lines of section text, a 16-line abstract, Figure 1, the slim table with an 11-line caption and Figure 3. This outline has 111 lines of section text, a 17-line abstract and a 5-line table caption, so it lands on F's margin (zero slack at the foot of page 4) with the valves below buying six lines. Float geometry: 1.5 in of figure is 9 lines, 1.75 in is 10.5, plus caption and gaps.

| # | Section (new name) | Job | Lines | Carries |
|---|---|---|---|---|
| — | Title block, authors | | 8 | |
| — | Abstract, keywords | Answer the title's question with the number | 17 + 2 | |
| 1 | Introduction | The gap (nobody scores a game world model on held-out maps of its own game), the robotics why, the question, three contributions with the finding as the loudest | 2 + 15 text + 1 lead-in + 17 bullets = 35 | Figure 1 teaser, 1.5 in, 37-word caption unchanged: 12.5 lines |
| 2 | Prior work | One paragraph: what is held out elsewhere, the "to our knowledge" claim, the adapter lineage and that none reports a budget, the two references we score against | 2 + 10 = 12 | |
| 3 | Setup (was "Method") | The fewest lines that make the results reproducible: data and split (6), backbones (4), recipe (5), metrics (9), decoder (4), adaptation and budget rule (9) | 2 + 37 = 39 | Table 1, the slim combined table (5 rows, 5-line caption): 16 lines |
| 4 | Results | Two paragraphs headed by the two findings: "A scene change breaks appearance, not control" (12) and "How much adaptation it takes" (15) | 2 + 27 = 29 | Figure 3, four-panel shift-and-repair, 1.75 in: 15 lines |
| 5 | Conclusion (was "Future work") | Restate the finding in numbers, limitations, the detect-and-repair loop | 2 + 5 = 7 | |
| | **Total** | | **149 text + 43.5 floats = 192.5** | |

Valves, in order, if the compile runs to page 5: Figure 3 at the 1.6 in variant (1 line); drop the predictor sentence at the end of Results (1); drop the data-ladder clause (1); fold the MultiGen sentence into a clause of the previous one (1); trim Section 2 to eight lines by dropping the persistence-and-upper-bound sentence, whose citations then move to the Metrics paragraph (2).

Moves to the supplement:
- Figure 2 (method overview, 1.8 in) with the "Training setup" paragraph (GPU, updates per second; the standing decision is no throughput claims) and the recipe table. It needs a label; the F branch references `app:overview`, which does not exist in `appendix_current.tex`, so add `\label{app:overview}` there or point at `app:protocol`.
- The Overleaf Table 1 (six rows with the upper bound and directional columns) as the full zero-shot table; the body keeps the directional numbers in one sentence of Results.
- Table 2 (adaptation cost by group) as is; the body keeps only the median budgets (150, 150, 250).
- From Section 3: the corpus totals (2,412 episodes, 96 hours, 12.1 M frames), the latent-contract sentence, the decoder's update count and MSE-only ablation, the control checks at 0 and 8k, the right-censoring clause, the sampler sweep.
- From Results: the per-arena ranges, the "one frame in five above 0.4 LPIPS" fact, the forgetting number.
- The supplement's closed-loop and latent-skill sections carry SD 3.5 numbers the brief says are removed from the paper; the body below never references `app:closedloop` or `app:skill`, so those sections can be cut or kept without touching page count. Someone must decide before the supplement is attached.

Nothing new is added anywhere: the full fine-tune numbers replace the draft's `\tbd`, and the SD 3.5 numbers replace the provisional ones.

---

## 4. Contributions

Lead-in (Changliu's nouns, current main's sentence): "This paper makes three contributions: a benchmark, an empirical finding and an adaptation result."

- **DoomShift, a benchmark for a domain shift that leaves the physics and the game engine exactly the same.** Three diffusion backbones, initialized from pretrained diffusion image models, are trained on four maps of Doom and scored on in-domain data and on 13 out-of-distribution maps they never saw, zero-shot and after adaptation, against the raw frame, the reconstruction upper bound of their own decoder and a directional check of the controls; recordings, models and adapters are released (Section~\ref{sec:bench}).

- **The finding: a scene change breaks appearance, not control, and it breaks it the same way at every architecture and scale we tried.** On the unseen maps every backbone's LPIPS roughly doubles and its scene PSNR falls about 3~dB (U-Net median 0.158 to 0.303 and 25.2 to 22.3~dB), a step rather than a slope, since every unseen map scores worse than every training map, while the turn response survives (directional 0.80 to 0.81 against 0.84 to 0.86 in domain) and the decoder reconstructs the unseen maps nearly as well as the training maps; the U-Net, PixArt-$\alpha$ and SD~3.5, from 0.6 to 2.3~B parameters, lose within half a decibel of each other, so the loss follows what the model was exposed to, not its architecture or scale (Table~\ref{tab:unseen}, Figures~\ref{fig:teaser} and~\ref{fig:adapt}).

- **The measurement: 2\% of training on 0.5\% of the parameters repairs it.** A standard rank-16 LoRA adapter on the frozen model, trained on eight episodes of the new map for 4k updates, about one A4000 GPU-hour, recovers 72\% of the lost perceptual quality and about half of the lost PSNR (median LPIPS 0.303 to 0.210, scene PSNR 22.3 to 23.6~dB) and returns the model's gap to the reconstruction upper bound to its in-domain 3.4~dB; the median map is halfway there after 150 updates; a full fine-tune of 205 times as many parameters gains only 0.2 to 0.3~dB more; unchanged, the procedure recovers 76\% on PixArt-$\alpha$ and 77\% on SD~3.5 (Figure~\ref{fig:adapt}, Table~\ref{tab:unseen}).

Order: Changliu's. I considered finding first and rejected it: the finding's sentence needs "13 held-out maps, three backbones" to be parseable, so bullet 1 is doing the reader a service if it is four lines; the dataset-paper reading comes from a long bullet 1 that leads with release counts (the Overleaf and the current main both spend three lines on 6,000/8,000 episodes there), not from the position. The release is now six words in bullet 1 and one sentence in Section 3.

---

## 5. Delta against the Overleaf draft

### Abstract (rewrite sentences 3 to 6; keep 1 and 2)

Keep: "World models have become capable simulators, ..." and "A small domain shift, such as a new arena of the same game, ..., though the underlying physics and game engine are exactly the same." (standing wording).

Cut: "DoomShift asks how quickly and cheaply a world model can be adapted across such a shift." (This is the sentence Changliu's comment 1 is about: it announces an adaptation method.)

New text from sentence 3 on (numbers in `\prov`):

> We ask what such a shift breaks and how much adaptation repairs it. On DoomShift, our benchmark for this question, we train three diffusion backbones, initialized from pretrained diffusion image models, on four maps of Doom and score each on in-domain data and on 13 other out-of-distribution maps that it never saw. On the unseen maps the U-Net's perceptual error (LPIPS) doubles, from \prov{0.158} to a median \prov{0.303}, and its scene PSNR drops \prov{2.9}~dB, from \prov{25.2} to a median \prov{22.3}~dB, while its turn response survives; the other two backbones, from 0.6 to 2.3~B parameters, lose the same way. The loss is cheap to repair: a rank-16 adapter on \prov{0.5\%} of the parameters, trained on 8 episodes of each unseen map for \prov{2\%} of the original training updates, recovers \prov{72\%} of the lost perceptual quality and about \prov{half} of the lost PSNR and returns the model to its in-domain distance from the reconstruction upper bound, with the median map halfway there after \prov{150} updates; unchanged, the same procedure recovers \prov{76\%} and \prov{77\%} on the other two backbones.

17 lines at abstract width. If it runs to 18, drop "On DoomShift, our benchmark for this question," to "We build DoomShift:" and merge with the next clause.

### Introduction

Paragraph 1: keep; shorten sentence 3 to "Each is evaluated only on held-out rollouts from the environment it was trained on, and none has been shown to generalize to a new environment within the same game." (cut "the reported score is a held-out trajectory drawn from its own training scenes and"). Shorten the MultiGen sentence to "Even MultiGen, one Doom model trained across 100 procedurally generated maps~\citep{po2026multigen}, reports PSNR, SSIM and LPIPS only on its training distribution, with no map held out." Six lines.

Paragraph 2: keep all four sentences, including the closing "We therefore explore what it takes to close the domain gap ..." sentence (standing decision). Eight lines. This paragraph is where AVID and AdaWorld are cited; Section 2 stops repeating them.

Lead-in: cut "We introduce DoomShift, which addresses these concerns and contributes the following:" and write "This paper makes three contributions: a benchmark, an empirical finding and an adaptation result."

Bullets: replace all three with Section 4 above. Quoted cuts from the old bullets: "We release every-tic recordings of 6,000 episodes on the four training maps and 24 on each of the 13 unseen maps (this paper trains on 2,000 of them), along with the three trained models, the decoder fine-tune and the adapter path (Figure~\ref{fig:method}; link withheld for review)." (stale count; moves to Section 3 as one sentence); "A post-training recipe that adapts a frozen model to a new arena ..." (the word "recipe" implies a method); "(full fine-tune comparator \tbd{})" (now a number).

Figure 1: keep, caption unchanged.

### Prior work (rewrite to one paragraph, 10 lines)

New text:

> Most Doom and game world models are trained on a set of maps and tested on maps from that same set. Papers that test a world model outside its training data hold out something other than a map of the same game: a new body for the agent~\citep{chen2026xeworld}, a whole new game~\citep{rigter2024avid,gao2025adaworld} or a new building~\citep{koh2021pathdreamer}; in game world models, new scenes are generated from an image and have no recorded frames to score against~\citep{tong2026scope,bruce2024genie}; Doom itself is held out this way only for the agents that play it~\citep{lample2017arnold,wydmuch2018vizdoom}. To our knowledge, no published game world model has been scored against recorded frames on maps of its own game that were deliberately held out from training. Our adapter is the frozen-model-plus-adapter recipe of AVID~\citep{rigter2024avid} and Vista~\citep{gao2024vista}, and our full fine-tune comparator follows AdaWorld~\citep{gao2025adaworld}; none of the three reports what adapting to one new scene costs in episodes and updates, which is the number we measure. Persistence, the last frame copied forward, is the standard reference of video prediction~\citep{mathieu2016deep,lotter2017prednet,villegas2019fidelity}, and the reconstruction upper bound is the standard reference of latent forecasting~\citep{zheng2023occworld,karypidis2024dinoforesight}; we read every score against both.

Cut, with the old sentences: "We take our method from two papers that put a small adapter on a frozen pretrained video model. AVID shows that training a small adapter on top of a frozen model is enough to move it into a new setting~\citep{rigter2024avid}; Vista uses the same frozen-model-plus-adapter idea to add action control rather than move the model to new content~\citep{gao2024vista}. Neither tests this on a map kept aside from the same game, and neither reports the budget adapting needs: how many episodes and how many training updates for one new scene." (three sentences collapsed into one); "AdaWorld solves a similar problem with a short full fine-tune instead of a small adapter~\citep{gao2025adaworld}; our full fine-tune comparator follows the same idea." (folded); "DoomShift combines what none of these do: maps kept aside from within a single game, with the adaptation budget measured directly." (repeats paragraph 2 of the intro; the "to our knowledge" sentence carries the claim); "we use it only to show why absolute PSNR alone does not compare scenes" and "which tells us how much error is the model's fault versus a limit built into its own decoder" (both said again in Metrics).

Kept on purpose: "To our knowledge, no published game world model has been scored against recorded frames on maps of its own game that were deliberately held out from training." The current main dropped it; it is the paper's clearest novelty claim.

### Section 3, renamed "Setup" (label `sec:bench` unchanged)

Reason for the rename: a "Method" section in a paper with no new method is what comment 2 objects to.

**Data** (rewrite, 6 lines):

> The Arnold agent~\citep{lample2017arnold} plays 150-second ViZDoom~\citep{kempka2016vizdoom} deathmatches against 8 bots on the 17 Doom maps it ships with, which we call arenas; each has its own layout, textures and lighting. We record every tic, the engine's 1/35~s step: a 320$\times$240 frame and the 19 buttons the engine executed. Four arenas (2 to 5) are the training maps, with 500 training episodes each and 25 validation episodes each (512 windows). The other 13 (1, 6 to 17) are never trained on: 24 episodes each, 16 for adaptation, of which this paper uses 8, and 8 held out for scoring (256 windows). The release holds 8,000 episodes on the four maps (6,000 train, 1,000 validation, 1,000 test) and the 24 per unseen arena, with the models, decoders and adapters (link withheld for review).

Cut: "We collect the data for this research by running experiments with ..." (rewritten); "In total we collect 2,412 episodes, about 96 hours of play and about 12.1 million frames." (supplement).

**Three backbones, one recipe** (rewrite, 4 lines):

> Our anchor is the SD~1.4 U-Net~\citep{rombach2022ldm} (860M parameters), GameNGen's backbone~\citep{valevski2024gamengen}. To vary the architecture at fixed scale we swap in PixArt-$\alpha$~\citep{chen2023pixart}, a 628M transformer in the same latent space; to vary the scale we use SD~3.5 Medium~\citep{esser2024sd3}, a 2.27B MMDiT with 16-channel latents, whose own decoder has a higher reconstruction upper bound (31.8 against 28.6~dB on the training maps).

Cut: "We pick three pretrained backbones to ask three different questions." and "architecture turns out not to change what survives the shift: in distribution their scene PSNR differs by 0.02~dB and their LPIPS by 0.001, and they lose the same amount on the unseen maps (Table~\ref{tab:unseen})" (a result; Results says it once).

**Recipe** (rewrite, 5 lines):

> We turn all three into next-tic world models under one recipe (\appref{app:protocol}): 32 context tics so the model sees motion, $v$-prediction~\citep{salimans2022progressive}, context noise augmentation as in GameNGen~\citep{valevski2024gamengen} so the model tolerates its own rollout errors, 200k updates at batch size 32 for every backbone, an fp32 EMA at 0.9999, and 10-step DDIM sampling~\citep{song2021ddim}, which we found to be a strong balance between quality and efficiency.

Cut: "(Figure~\ref{fig:method})" (figure to supplement); "(a bf16 EMA underflows and freezes)"; "(sampler sweep in \appref{app:protocol})" (covered by the paragraph's appref).

**Metrics** (rewrite, 9 lines):

> We score the scene rows of every predicted frame (rows 0 to 207; the HUD below them does not change with the map) one tic ahead against the raw ground-truth frame with PSNR and LPIPS~\citep{zhang2018lpips}. Absolute PSNR alone does not compare arenas, because it rises with how static a recording is: the persistence baseline, the last context frame copied forward~\citep{mathieu2016deep}, already scores a median 19.8~dB on the unseen arenas (18.5 to 22.5). We therefore read every arena against two references: the \emph{reconstruction upper bound}, the decoder $D$ applied to the ground-truth latent, which is in practice the best a prediction rendered through $D$ can score (median 27.3~dB on the unseen arenas, 28.6 on the training maps), and the in-distribution level, the same model on the training maps; the \emph{gap} is the upper bound's PSNR minus the prediction's. For control, the \emph{directional} score is the fraction of turning windows whose predicted view turns the other way once we swap left and right in the newest control.

Cut: "We score every predicted frame against four questions: how accurate is it against the raw ground-truth frame, how realistic does it look, how far is it from the best the decoder could possibly render and does it turn the right way when we mirror the controls?" (rhetorical; two lines).

**Training setup**: cut the whole paragraph ("We train each backbone on a single RTX A6000 (49~GB) in bf16 with fused AdamW, reaching 1.8 updates per second for the U-Net, 1.4 for PixArt-$\alpha$ and 0.53 for SD~3.5. We train each adapter on a single RTX A4000 (16~GB), at about 1.0 updates per second."). Supplement, next to Figure 2. The one number the body keeps is "about 1 A4000 GPU-hour per arena" in the Adaptation paragraph.

**Decoder** (rewrite, 4 lines):

> The encoder stays frozen, since its latents are every model's training target; the decoder only renders, so fine-tuning it changes what we can see, not what the model knows. We fine-tune each backbone's decoder on training-map frames paired with their frozen-encoder latents (MSE plus 0.1~LPIPS; \appref{app:protocol}); it never sees an unseen arena. The SD~1 decoder's scene reconstruction rises from 26.9 to 28.4~dB and its LPIPS falls from 0.092 to 0.060, and it reconstructs the unseen arenas nearly as well as the training maps (27.3 against 28.6~dB), so the shift lives in the model, not the renderer.

Cut: "Every backbone is trained on latents from its frozen encoder, so the latent space is the contract between encoder, world model and decoder: changing the encoder would move every training target and invalidate the trained backbones, the adapters and the reconstruction upper bound." (supplement); "for 3,486 updates and gating it on held-out training-map frames"; "the LPIPS term is there because MSE alone more than triples scene LPIPS (0.092 to 0.339)"; "\prov{SD~3.5 uses its own stock decoder}" (stale: every backbone now scores through its own fine-tuned decoder).

**In-distribution performance**: cut the whole paragraph ("On the training maps the three backbones reach a scene PSNR of 25.20, 25.18 and \prov{24.50}~dB ... Four tics ahead they reach 21.33, 21.29 and \prov{21.59}~dB (stock decoders, full frame; \prov{SD~3.5 at 140k})."). The SD 3.5 and four-tic numbers are stale, and Table 1's in-domain columns carry the rest; the "3.4~dB below the upper bound in domain" fact reappears in Results.

**Post-training**, renamed **Adaptation** (rewrite, 9 lines):

> To move a trained model to a new arena we keep its weights frozen and train a small adapter on a few episodes of that arena: rank-16 LoRA~\citep{hu2022lora} on every attention projection~\citep{gao2024vista}, with the control MLP, input projection and noise-bucket embedding trained in full~\citep{xie2023difffit} (4.2M parameters, 0.49 percent of the U-Net). We train on 8 episodes, about 1 A4000 GPU-hour per arena for 4k updates, and score the held-out episodes at 0, 50, 100, 150, 250, 500, 1k, 2k, 4k and 8k updates (SD~3.5 from 250). We report 4k everywhere, 2 percent of the 200k pretraining updates: it captures on average 96 percent of each arena's 8k gain, and 8k serves as the check. An arena's \emph{budget} is the first grid point at which its gap to the upper bound falls to the midpoint between its zero-shot gap and the training maps' 3.4~dB, that is, at which it has closed half of the excess gap the shift opened. As a comparator we fine-tune all 860M parameters of the U-Net on arenas 6, 7, 8 and 16 with the same eight episodes.

Cut: "the U-Net 200k EMA weights" detail (supplement); "with the control checks (forgetting, the scene PSNR on the training maps, and the directional check) at 0 and 8k" (supplement); "4k updates is the elbow of the curve, and we report it everywhere: across the 13 arenas it captures most of the benefit ... which is why the recipe costs 2 percent of the original training updates; 8k serves as the check, and a full fine-tune comparator at $2\times10^{-5}$ is in progress (\tbd{})." (rewritten; the comparator is now a number); "arenas that do not by 8k are right-censored~\citep{taylor2009transfer}" (no arena needs it at the final budgets; supplement).

**Table 1**: replace the six-row zero-shot table with the slim combined table (`results_slim.tex`, five rows) and a five-line caption:

> Scene PSNR (dB) and LPIPS one tic ahead against the raw frame: in domain (four training maps, 512 validation windows), zero-shot on the 13 unseen arenas and after 4k adapter updates on 8 episodes of each (medians over arenas), every backbone through its own fine-tuned decoder. Recovered: the median per-arena share of the LPIPS rise undone, of the lost PSNR regained and of the extra gap to the reconstruction upper bound closed. Last two rows: the U-Net's adapter against a full fine-tune of all its parameters on arenas 6, 7, 8 and 16.

The eleven-line caption in `results_slim_caption.tex` is the single largest saving available (six lines). The old six-row table, with its upper bound and directional columns, goes to the supplement as the full zero-shot table.

### Results

Cut the opening sentence "We report what a model loses zero-shot, what post-training recovers and with what budget, and what predicts how far an arena gets." (the paragraph heads do this).

**Paragraph 1** (rewrite, 12 lines; head renamed from "Zero-shot: where the deficit sits"):

> \textbf{A scene change breaks appearance, not control.} On the unseen arenas the turn response survives: pooled directional scores of 0.804, 0.808 and 0.800 for the U-Net, PixArt-$\alpha$ and SD~3.5 against 0.855, 0.842 and 0.840 on the training maps (ground-truth frames 0.885 and 0.892), and zero-shot on arena 7 the weapon fires and the view advances as in the ground truth (Figure~\ref{fig:teaser}). The appearance does not: the U-Net's scene PSNR falls from 25.2~dB on the training maps to a median 22.3~dB, its LPIPS rises from 0.158 to 0.303 and its gap to the reconstruction upper bound grows from 3.4 to a median 5.0~dB (Figure~\ref{fig:adapt}a, b). The decoder is not the limit: it reconstructs the unseen arenas nearly as well as the training maps (27.3 against 28.6~dB). The shift is a step, not a slope: every unseen arena has a worse LPIPS than every training map, for all three backbones (\appref{app:distance}). And it is the same step for all three: scene PSNR drops 2.9, 2.8 and 3.3~dB and LPIPS rises by 0.145, 0.126 and 0.136 (Table~\ref{tab:unseen}), so a change of architecture at fixed scale and a 2.6-fold increase in scale leave the loss where it was; it follows what the model was exposed to.

Cut: the per-arena ranges "(20.5 to 24.7 per arena)", "(0.229 to 0.401)", "(3.7 to 6.9)", "U-Net directional scores 0.73 to 0.86 per arena against 0.79 to 0.91 in distribution"; "Pooled over the 13 arenas, one predicted frame in five has an LPIPS above 0.4, against one in 73 on the training maps; such frames concentrate in arenas 16, 17, 13 and 1 (32 to 59 percent of their windows), while seven arenas stay below 10 percent." (supplement); "the loss sits in the scene, whose rows fall 2.9~dB (25.2 to a median 22.3) while the map-independent HUD rows fall 0.2~dB (29.7 to 29.5)" (the decoder sentence makes the point; the HUD number is not in the brief's table).

**Figure 3**: keep at 1.75 in, caption shortened to three lines (the F branch's caption is right): "The U-Net per unseen arena, zero-shot (open) and after 4k adapter updates (filled): PSNR (a) and LPIPS (b), with the reconstruction upper bound (ticks) and the training maps (dashed); (c, d) against updates, coloured by zero-shot gap to the upper bound (dark: hard)." Cut from the old caption: "Medians: 2.9~dB lost, 1.3 recovered; LPIPS 0.303 to 0.210. Fine-tuned decoder, raw frame." (the text and Table 1 carry these).

**Paragraph 2** (rewrite, 15 lines; head renamed from "Post-training: what it takes"):

> \textbf{How much adaptation it takes.} Eight episodes and 4k adapter updates raise the U-Net's median scene PSNR from 22.3 to 23.6~dB and lower its median LPIPS from 0.303 to 0.210 (Figure~\ref{fig:adapt}, Table~\ref{tab:unseen}); the 8k check adds little. Read against the upper bound, the repair is complete: in domain the model sits 3.4~dB below its bound, zero-shot a median 5.0~dB, and after 4k updates, 2 percent of its pretraining, a median 3.4~dB again, so per arena the adapter closes a median 94 percent of the extra distance the shift opened, and at least half on every arena `[draft number, verify]`; most of the 1.6~dB still missing against the training maps is the arenas' lower bound (27.3 against 28.6~dB), not the model. In LPIPS it undoes a median 72 percent of each arena's rise (at least 48 percent `[draft number, verify]`) and regains 48 percent of the lost PSNR. It is also fast: the median arena closes half of its excess gap within 150 updates, under 4 percent of the 4k budget. Unchanged, the procedure carries over to the other backbones: PixArt-$\alpha$ undoes 76 percent of its LPIPS rise and closes 101 percent of its excess gap (budget 150 updates), SD~3.5 77 and 94 percent (budget 250, the first point of its grid). A full fine-tune of all 860M U-Net parameters on the same eight episodes, 205 times as many as the adapter trains, does better on the four comparator arenas (median LPIPS 0.182 against 0.209 and PSNR 23.11 against 22.88~dB at 4k; 87 against 71 percent of the LPIPS rise undone) but needs a 49~GB card where the adapter trains on a 16~GB one. Data matters less than the first updates: one adaptation episode already gives most of the gain of sixteen, and the adapters give up a little in-distribution skill for it (\appref{app:perarena-adapt}). Neither the model's own latent-space score nor a frame distance to the training maps predicts how far an arena gets (\appref{app:distance}).

Cut: "(over 100 percent on arenas 8, 9 and 10, whose static recordings already sit near the training maps' PSNR)"; "the training maps' scene PSNR against the decoded ground truth falls on all 13 arenas at the 8k check (median 0.59~dB; stock decoder), while the directional score rises from 0.81 to 0.84" (the 0.59 is not in the brief's table; supplement); "(budgets below 4k \tbd{})" (now 150).

Do not add (the current main did): the per-backbone adapter parameter counts for PixArt and SD 3.5, the A6000 hours for their adapters, "half of each arena's 8k gain arrives within 100 updates", "none censored". None of these is in the brief's table.

**Table 2**: move to the supplement unchanged. The body cites it nowhere; the group budgets (hard, medium, easy) are the one thing lost from the body, see Risks.

### Section 5, renamed "Conclusion" (rewrite, 5 lines)

> On 13 unseen arenas of Doom, three world models from 0.6 to 2.3~B parameters keep their response to the controls and lose the arena's appearance, a step change that the decoder does not explain; eight episodes and 4k adapter updates, 2 percent of pretraining on 0.5 percent of the parameters, put a model back at its in-domain distance from the reconstruction upper bound, and the median arena is halfway there after 150 updates. The claims cover 13 arenas of one game, one agent, one-tic scoring and mostly one seed, and one-tic quality does not guarantee stable rollouts. A deployed world model would need to detect such a shift from its own scores and repair it within the budget measured here; DoomShift is the testbed for that loop.

Cut: "One-tic quality does not guarantee stable rollouts: SD~3.5, best at one tic on the full frame with stock decoders, collapses in 23 non-EMA and 9 EMA rollouts of 256, and the EMA removes checkpoint-specific collapses, not the failure mode (\appref{app:closedloop})." (the brief removes the SD 3.5 closed-loop numbers); "Next are other Doom maps and other games, to see whether the same recipe transfers; detecting a shift from the model's own zero-shot scores so that it repairs itself; closed-loop evaluation of the adapted models; and agents trained inside the adapted model." (folded into the last sentence).

### Draft marks

The SD 3.5 numbers are final, so their `\prov` outside the abstract can go; the abstract keeps `\prov` on its numbers (standing decision) and `\draftmarksfalse` turns them black at submission. Every `\tbd` in the body is gone after this delta; the acknowledgments `\tbd` remains for Rohan.

---

## 6. Novelty paragraph

How the finished paper makes the claim (Section 2's "to our knowledge" sentence plus the two Results heads, in one breath): to our knowledge this is the first time a game world model has been scored against recorded frames on maps of its own game that were held out from training, and the first report of what adapting to one such map costs in episodes, updates and parameters. The novelty is not the adapter, which is standard LoRA, but what the measurement shows and that nobody had it: the shift is a step, not a slope; it is the same step for two architectures and a 2.6-fold change of scale; it leaves the control response and the decoder intact, so it lives in the model's appearance prior; and it is cheap, 2 percent of the training updates from eight episodes on half a percent of the parameters, with the median map halfway repaired after 150 updates and a full fine-tune of 205 times the parameters buying 0.2 to 0.3~dB more.

What a skeptical reviewer says: "This is LoRA on a new Doom level. Of course it works, and of course a bigger image backbone does not help with textures it never saw. Thirteen levels of one game, one seed, one tic ahead, medians over arenas."

The answer, which the paper must state in these words or close to them: the paper claims no method, so "of course it works" is agreement with the measurement, and the measurement was not obvious in advance. Three outcomes were open before the runs and closed the other way: the shift could have degraded the control response (it did not, 0.80 against 0.84 to 0.86), scale could have bought robustness (SD 3.5 loses the most, 3.3~dB), and the repair could have needed the whole network (the full fine-tune gains 0.2 to 0.3~dB for 205 times the parameters and a 49~GB card). The adapter papers this builds on, AVID, Vista and AdaWorld, report no budget, so "an hour on a 16~GB card per new scene" is a number a deployment can plan with and could not read anywhere before. The limits (one game, one agent, one seed, one tic) are stated in the conclusion and are the honest scope of a four-page workshop paper.

---

## 7. Where the current-main attempt (c207bb6) went right and wrong

Right:
- It adopted Changliu's title verbatim and her three nouns as the lead-in ("This paper makes three contributions: a benchmark, an empirical finding and an adaptation result."), and its bullet heads "What a scene change breaks" and "How much adaptation recovers it" are the study framing she asked for; I keep both ideas.
- It condensed Section 2 to one paragraph and removed the sentences that repeat intro paragraph 2 (the AVID and AdaWorld explanations), which is exactly comment 3.
- It renamed post-training to adaptation everywhere, put the full fine-tune into bullet 3 with its numbers, and replaced the stale SD 3.5 numbers and stock-decoder sentence.

Wrong:
- The abstract still opens its third sentence with "DoomShift asks how quickly and cheaply a world model can be adapted across such a shift.", the one sentence comment 1 targets, and bullet 1 still spends three lines on release counts, which is what makes the paper read as a dataset paper. The framing change happened in the bullet heads and nowhere the reader looks first.
- It grew instead of shrinking: per-backbone adapter parameter counts, three adapters' GPU hours, a twelve-line three-backbone adaptation passage in Results and a longer Table 2 caption were added while Section 3 kept the "three questions" paragraph, the in-distribution paragraph and the Training setup paragraph. It is further from four pages than the Overleaf draft, and it uses numbers ("half of each arena's 8k gain within 100 updates", "none censored") that the brief's final table does not list.
- It dropped the paper's clearest novelty claim, "To our knowledge, no published game world model has been scored against recorded frames on maps of its own game that were deliberately held out from training.", and the AVID citation from Section 2, while keeping the sentence that says nothing new ("Held-out tests exist, but ...").

On the F branch, for what fits: its cuts to Data, Recipe, Decoder and Results are the right cuts and I reuse most of them; it references `\appref{app:overview}`, which does not exist in `appendix_current.tex`; its Table 1 caption is eleven lines, the largest saving still on the table; and its three-line Future work loses the closing sentence that ties the finding to the detect-and-repair loop.

---

## 8. Risks

- **The study framing without a loud headline reads as "we fine-tuned LoRA and it worked", which is weaker than the Overleaf draft's method hook.** The whole proposal rests on the abstract's fifth sentence and the two bolded bullet heads carrying the finding; if a later pass softens those to "we study" language, the paper loses its hook without regaining a method. Keep the heads as written.
- **Page fit is on the F branch's margin: zero slack at the foot of page 4.** This text is five lines longer than F in the sections and six lines shorter in the table caption; the abstract is one line longer. The valves in Section 3 buy six lines; if the compile still spills, the next cut is the full fine-tune sentence in Results (two lines), since Table 1's last two rows carry it.
- **Table 2 leaves the body, and with it the per-group budgets** (hard arenas need up to two thousand updates, easy ones about a hundred), which was the "what predicts cost" thread. The body keeps only medians; a reviewer who asks "which maps are hard" is sent to the supplement. If Rohan wants that thread back, it costs three lines and comes out of Section 2.
- **Figure 2 leaves the body, so there is no picture of the pipeline.** Figure 1 shows the three conditions (in domain, zero-shot, adapted) and Figure 3 shows the repair; the recipe paragraph is five lines. For a study paper that is enough, but the reader who wants to reproduce it goes to the supplement.
- **"The same step for all three" rests on three backbones, one seed each, and SD 3.5 lives in a different latent space** (its bound is 31.8 against 28.6~dB, its drop 3.3 against 2.9). The text says drops in dB and LPIPS, not that the models are interchangeable; do not let a later edit say "identical".
- **Two per-arena minima carried from the draft** ("at least half of the excess gap on every arena", "at least 48 percent of the LPIPS rise") are not in the brief's final table. They must be confirmed against the per-arena table before submission or cut; cutting them loses nothing structural.
- **Bullet 1 is still first.** If it grows past four lines in editing, the dataset-paper reading returns; the release stays six words there and one sentence in Setup.
- **The supplement is out of step with the body**: it still holds SD 3.5 closed-loop and latent-skill numbers the brief removes, a `\tbd` in the skill section, and no `app:overview` label. The body above never references those sections, so the risk is a reviewer opening the supplement and finding claims the body withdrew; someone should cut those sections before the supplement is attached.
