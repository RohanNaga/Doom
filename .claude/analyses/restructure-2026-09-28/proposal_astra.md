# 1. Thesis in one sentence

Across three Doom world models, changing the surroundings while keeping the physics and game engine exactly the same separates what transfers from what must be relearned: turn response survives, every unseen arena has worse perceptual error than every training map, and eight episodes with 4k adapter updates recover most of the perceptual loss.

# 2. Title

**What Survives a Scene Change?**  
**A Study of World-Model Adaptation in Doom**

This variant of Changliu's study framing makes the empirical question the headline and adaptation its measurable answer; **DoomShift** remains the benchmark's name in the abstract, first contribution, and protocol.

# 3. Outline for 4 pages

The paper is a study of a failure and its recoverability. The benchmark is the instrument that makes this measurable. Keep Changliu's contribution order without making the dataset the opening argument.

Budget **200 body-line equivalents**, including front matter, headings, captions, float spacing, and figures. These are typeset lines, not TeX source lines. Reserve **188** for content and **12** for reflow. At roughly 50 lines per page, a 1.5-inch image alone occupies about 9 lines and a 1.75-inch image about 10; neither is a free insertion. The following allocations include the caption and float spacing separately from the image height.

| Component | Job | Text/heading lines | Float lines, including caption and spacing | Total |
|---|---|---:|---:|---:|
| Title, authors, abstract, keywords | State the discovery, setting, and measured recovery | 25 | 0 | 25 |
| 1. Introduction, including contributions | Motivate retained versus lost capability; state the three claims | 31 | 14: Figure 1, 1.5-inch teaser, 35-word caption | 45 |
| 2. Prior work | Identify the exact gap and acknowledge established adaptation methods | 8 | 0 | 8 |
| 3. Study protocol | Specify splits, models, scoring, adaptation, and budget definition | 44 | 0 | 44 |
| 4. Results | Establish the split in transfer, quantify recovery, and expose the full-tuning tradeoff | 30 | 13: slim Table 1; 16: original Figure 3 at 1.75 inches | 59 |
| 5. Scope and implications | State limits and the scientific conclusion | 7 | 0 | 7 |
| **Content** | | **145** | **43** | **188** |
| **Reflow allowance** | Citation wrapping, list spacing, float placement | | | **12** |
| **Hard limit** | | | | **200** |

Intended occupancy: page 1 carries front matter and the introduction; page 2 carries the teaser, prior work, and the start of protocol; page 3 finishes protocol and begins results with the table; page 4 carries the quantitative figure, remaining results, and scope. These are flow targets, not forced page breaks. Keep the contributions together if the class permits; do not reproduce scenario F's split of the benchmark bullet onto page 1 and the scientific bullets onto page 2 merely to fill a page.

**Body assets:** retain simplified Figure 1, remove its individual-frame numerical scores, and use the exact caption below. Keep the original four-panel U-Net shift-and-repair figure at 1.75 inches: it shows both endpoints and the update curve, which a study of adaptation needs. It becomes **Figure 2** after the method overview moves. Use one five-row combined table with the three backbone rows and the matched U-Net LoRA/full-tune pair. Its caption is rewritten below; do not carry scenario F's long caption. Do not add the 3.1-inch figure or another backbone panel. Cross-backbone evidence is in the table and directional results.

**Supplement:** move the method overview (original Figure 2, 1.8 inches), full zero-shot table (original Table 1), grouped cost table (original Table 2), detailed per-arena curves, update grids, sampler sweep, decoder training details, recording details, and distance/latent-score analyses. Retain the existing supplement's control/forgetting checks with their scoring protocol clearly separated from the main raw-frame protocol. Do not import its unapproved numerical results into this proposal. Remove the stale SD 3.5 closed-loop material identified in section 5. No new experiments are needed.

The supplied scenario F PDF establishes that the two-figure/one-table arrangement fits four body pages. I also compiled the proposed replacement prose, actual table, captions, author block, and citations in a temporary copy of the CoRL template, reserving the full stated image heights with solid rectangles. That layout check fits four body pages with no overfull boxes. It checks text and float occupancy, not final image legibility or the simplified teaser artwork. The allocations above remain conservative; final asset placement still needs a PDF check.

# 4. Contributions

Use these bullets verbatim, in Changliu's order. The first is short and contains no release inventory.

```tex
\begin{itemize}\setlength{\itemsep}{0pt}
\item \textbf{DoomShift: measuring scene transfer.} A benchmark with four training maps and 13 unseen Doom arenas measures what survives a change of surroundings and what adaptation recovers, while the physics and game engine remain exactly the same.
\item \textbf{A split in what transfers.} Across three diffusion backbones initialized from pretrained diffusion image models, every unseen arena has worse LPIPS than every training map, yet turn response survives. The perceptual separation persists across the backbones; decoder reconstruction shows that rendering limits alone do not explain it.
\item \textbf{Most perceptual loss is recoverable with little adaptation.} Standard adapters trained on eight episodes for 4k updates, 2\% of backbone training, recover median shares of 72\%, 76\%, and 77\% of the LPIPS rise for the U-Net, PixArt-$\alpha$, and SD~3.5. The U-Net uses about 1 A4000 GPU-hour per arena; full fine-tuning recovers more quality with 205 times as many trainable parameters.
\end{itemize}
```

# 5. Delta against the Overleaf draft

**Application rule:** `overleaf_draft_2026-09-27.tex` is the base. Replace each named prose block in full with the corresponding text below; do not append it to the old prose. Preserve author metadata, the existing citation keys used below, `\prov`, and the keywords `World models, Few-shot adaptation`. The contribution list above is part of the introduction, not additional copy. Renumber the remaining body floats normally and replace references to the moved method overview with a supplement reference. The old quotations below identify deletions; their stale numbers are not proposed results.

## Abstract — replace in full

```tex
\begin{abstract}
What survives when a world model encounters new surroundings but the physics and game engine stay exactly the same?
DoomShift studies \prov{three} diffusion backbones initialized from pretrained diffusion image models, trained on \prov{four} Doom maps and evaluated on \prov{13} unseen arenas.
For every backbone, every unseen arena has worse perceptual error than every training map, yet turn response survives.
With \prov{eight} episodes per arena and \prov{4k} adapter updates, \prov{2\%} of backbone training, standard adaptation recovers median shares of \prov{72\%}, \prov{76\%}, and \prov{77\%} of the LPIPS rise for the U-Net, PixArt-$\alpha$, and SD~\prov{3.5}, respectively.
Thus, a scene change exposes a sharp perceptual deficit even when control response transfers, but most of that deficit is recoverable with little adaptation.
\end{abstract}
```

This keeps every abstract quantity blue, including counts. It stops making unmeasured rollout drift the motivation and states the actual all-arena finding. The detailed PSNR/LPIPS levels appear once, in the table.

Cut the old generalization: “A small domain shift, such as a new arena of the same game, a new texture set, a lighting change or a different cast of enemies, makes their outputs drift and lose realism, though the underlying physics and game engine are exactly the same.” No experiment here isolates those interventions or establishes adapted rollout drift. Replace the remaining abstract sentences as a block rather than retaining a second U-Net-only numerical summary.

## Introduction — replace the two opening paragraphs and the contribution lead-in

```tex
Game world models can render familiar environments from pixels and controls~\citep{valevski2024gamengen,alonso2024diamond,he2025matrixgame2}.
But predicting another trajectory in a training scene does not establish what transfers to an unseen scene: even MultiGen's multi-map Doom evaluation does not report deliberately held-out maps~\citep{po2026multigen}.
We find a split in what transfers: turn response survives a scene change across three pretrained backbones, while perceptual error separates every unseen arena from every training map.

For an agent entering a new room or workspace, the question is which capabilities remain usable and how much adaptation the rest requires.
We therefore explore what it takes to close the domain gap that opens when a world model meets a shift in which the underlying physics and game engine are exactly the same and only the surroundings, scenario or scene change: what the model retains, what it loses, and how few episodes and updates recover the rest.

We make three contributions:
```

Insert the three bullets from section 4 immediately afterward. The second paragraph ends with Rohan's exact requested sentence, including the full scope of the question. The first paragraph already delivers the finding before the benchmark bullet appears.

Cut “Pretraining one model on every map a deployment might meet is costly and never complete, and fine-tuning a separate model per map is prohibitively costly.” The measured full-tune comparator does not support calling it prohibitively costly. Move the AVID/AdaWorld comparison out of paragraph 2 and into prior work; the old sentence being displaced is “Yet unchanged dynamics alone do not guarantee a faithful render: pretrained video and world models are typically moved to a new domain only through a further full fine-tune~\citep{gao2025adaworld} or a dedicated adapter on the frozen backbone~\citep{rigter2024avid}, and a model trained on a handful of maps tends to fall back on the appearance it already knows once the surroundings change.”

Remove the old release sentence from contribution 1: “We release every-tic recordings of 6,000 episodes on the four training maps and 24 on each of the 13 unseen maps (this paper trains on 2,000 of them), along with the three trained models, the decoder fine-tune and the adapter path (Figure~\ref{fig:method}; link withheld for review).” Its replacement is the single release sentence in the protocol below. Replace the entire old recipe contribution with contribution 3, including removal of its pending-comparator placeholder. Replace the old characterization bullet with contribution 2, especially its unsupported conclusion “so the loss comes from what the model was exposed to, not from its architecture or scale”.

**Figure 1: keep, simplify, and replace caption.** Remove the per-example PSNR labels and redundant headers; retain training examples, unseen zero-shot/adapted examples, actions, and ground truth. The following is exactly **35 whitespace-delimited words**:

```tex
\caption{On an unseen arena, the U-Net still responds to controls but renders familiar masonry. Eight adaptation episodes improve the scene. Training-map examples appear at left; unseen-map predictions before and after adaptation appear beside ground truth.}
```

This replaces “On an unseen arena the U-Net still moves and fires but renders the training maps' masonry; eight adaptation episodes restore the arena. Left: training maps (in distribution). Right: an unseen map (arena 7), zero-shot and after adaptation.” “Improve” avoids implying complete recovery.

## Prior work — replace both paragraphs with one

```tex
\section{Prior work}
\label{sec:related}
AVID adapts a frozen video model, AdaWorld uses full fine-tuning, and Vista adds control through adaptation~\citep{rigter2024avid,gao2025adaworld,gao2024vista}.
Held-out evaluations also study robot embodiments and buildings~\citep{chen2026xeworld,koh2021pathdreamer}.
DoomShift instead measures retained control response, perceptual loss, and adaptation budgets together on held-out scenes of the same game, extending map holdouts used for Doom agents~\citep{lample2017arnold,wydmuch2018vizdoom}.
Our contribution is this empirical characterization; the adapter uses established techniques.
```

Cut the repeated opener “Most Doom and game world models are trained on a set of maps and tested on maps from that same set.” Cut the sweeping priority sentence “To our knowledge, no published game world model has been scored against recorded frames on maps of its own game that were deliberately held out from training.” The concrete conjunction of measurements is strong enough without a fragile universal first claim. Cut “DoomShift combines what none of these do: maps kept aside from within a single game, with the adaptation budget measured directly.” The replacement states the distinction without pretending the ingredients are new.

Move the persistence and reconstruction-reference explanation into the protocol, where each quantity is defined once. Move the longer AVID/Vista distinctions and the image-conditioned scene-generation discussion into supplementary positioning if retained; do not duplicate them in the introduction. The replacement paragraph is the complete body positioning, not a summary to be followed by the old paragraphs.

## Method — replace in full and rename “Study protocol”

```tex
\section{Study protocol}
\label{sec:bench}
\textbf{DoomShift.} Arnold/ViZDoom deathmatches~\citep{lample2017arnold,kempka2016vizdoom} provide 17 arenas.
We train on 500 episodes per map for maps 2 to 5 and evaluate on 25 validation episodes per map, using 512 windows in total.
Each of the 13 unseen arenas has 24 episodes: 16 reserved for adaptation, of which eight are used, and eight held out for scoring with 256 windows per arena.
We release every-tic recordings on the Hub: 8,000 episodes on the training maps, split 6,000/1,000/1,000 for training, validation, and test, and 24 episodes per unseen arena (link withheld for review).

\textbf{Models.} SD~1.4 U-Net (860M), PixArt-$\alpha$ (628M), and SD~3.5 Medium (2.27B) are initialized from pretrained diffusion image models~\citep{rombach2022ldm,chen2023pixart,esser2024sd3}.
We train each for 200k updates under a shared next-tic recipe with control conditioning, $v$-prediction, and context noise augmentation~\citep{salimans2022progressive,valevski2024gamengen}.
We use 10-step DDIM~\citep{song2021ddim}, a strong balance between quality and efficiency (supplementary sampler sweep).
Encoders stay frozen; decoders are fine-tuned on training-map frames only, with the SD~1 decoder shared by the U-Net and PixArt and SD~3.5 using its own.
These are comparisons of pretrained model packages, not isolated tests of architecture or scale; implementation details are in the supplement.

\textbf{Measurements.} We score one-tic predictions on scene rows 0 to 207 against raw frames using PSNR and LPIPS~\citep{zhang2018lpips}.
Directional score measures reversal of predicted camera motion after swapping the newest left/right turn control.
The reconstruction upper bound is the encoder--decoder reconstruction reference~\citep{zheng2023occworld,karypidis2024dinoforesight}; the gap is its PSNR minus the model's.
It is an empirical reference, not a mathematical bound.
Recovered LPIPS is the reduction in an arena's error divided by its rise above the in-domain error; PSNR recovery analogously divides improvement by the loss from the in-domain level.
Excess-gap recovery divides the gap reduction by the zero-shot gap's excess over the in-domain gap.
We compute recovery within each arena before taking the median.
Persistence copies the last frame~\citep{mathieu2016deep}; its scene PSNR varies from 18.5 to 22.5 dB across unseen arenas, motivating a reconstruction reference alongside absolute PSNR.

\textbf{Adaptation.} We initialize from each backbone's EMA weights and train rank-16 LoRA on every attention projection, updating the control MLP, input projection, and noise-bucket embedding in full while freezing the remaining backbone weights~\citep{hu2022lora,xie2023difffit}.
For the U-Net this trains 4.2M parameters (0.49\%) and costs about 1 A4000 GPU-hour per arena for 4k updates on eight episodes.
We report non-EMA adapter weights at 4k; the U-Net's 8k check shows that 4k captures on average 96\% of the 8k gain.
An arena's budget is the first measured update count closing half its excess reconstruction gap above its backbone's in-domain gap: 3.36, 3.37, and 6.46 dB, respectively.
The supplement gives the grids and per-arena results.
```

This replacement deliberately spends lines on the scoring contract and the distinction between available episodes and used episodes. The release count is not the study's training count. Retain the existing detailed training recipe in the supplement, not as another body paragraph. Restrict the 96%/8k claim to the U-Net: the other backbone grids in the supplied current draft stop at 4k.

Move original Figure 2 to a supplementary “Study overview,” replacing its caption with: “DoomShift protocol: held-out arenas, pretrained backbones, standard adaptation, and evaluation against raw frames and decoder reconstruction references.” Relabel its panel (c) “Adaptation.”

Remove the body `Training setup` paragraph, including “We train each backbone on a single RTX A6000 (49~GB) in bf16 with fused AdamW, reaching 1.8 updates per second for the U-Net, 1.4 for PixArt-$\alpha$ and 0.53 for SD~3.5.” Also remove “We train each adapter on a single RTX A4000 (16~GB), at about 1.0 updates per second.” Do not move throughput claims into the supplement; remove its `Updates per second` row and throughput wording too. Keep optimizer and precision details there.

Move the decoder ablation details to the supplement; the body needs only the scoring contract above and the diagnostic result below. Replace the stale sentence “We use it for every pixel score of the U-Net and PixArt-$\alpha$ unless stated otherwise; \prov{SD~3.5 uses its own stock decoder}.” with the fine-tuned-decoder sentence in `Models`. Move the in-distribution performance paragraph into the combined table. Delete, without relocation, “Four tics ahead they reach 21.33, 21.29 and \prov{21.59}~dB (stock decoders, full frame; \prov{SD~3.5 at 140k}).”

Delete the old data total: “In total we collect 2,412 episodes, about 96 hours of play and about 12.1 million frames.” Replace the full old `Data` paragraph with `DoomShift` above. Move recorder operation details to `Protocol details` in the supplement. Replace the old budget definition that hardcodes the training maps' “3.4~dB” for all backbones with the backbone-specific definition above.

## Results — replace in full

```tex
\section{Results}
\label{sec:results}
\textbf{A scene change separates appearance from turn response.}
Every unseen arena has worse LPIPS than every training map for all three backbones: the shift is a step, not a slope (supplementary per-map scores).
Yet directional scores remain 0.804, 0.808, and 0.800, against 0.855, 0.842, and 0.840 in domain; ground-truth references are 0.892 and 0.885, respectively.
Decoder reconstruction alone does not explain the deficit: the SD~1 decoder reconstructs unseen arenas at 27.32 dB versus 28.55 in domain, while the U-Net falls from 25.20 to 22.30 dB.
Table~\ref{tab:unseen} shows the same perceptual failure across the pretrained packages.

\textbf{Most perceptual loss is quickly recoverable.}
At 4k updates, adapters recover median LPIPS shares of 72\%, 76\%, and 77\%, and PSNR shares of 48\%, 48\%, and 50\% (Table~\ref{tab:unseen}).
The corresponding median shares of excess reconstruction gap closed are 94\%, 101\%, and 94\%; this is relative recovery, not perfect reconstruction.
Median budgets to close half that excess gap are 150, 150, and 250 updates; SD~3.5's grid starts at 250, so these values do not establish a slower adaptation rate.
Figure~\ref{fig:adapt} shows the U-Net's per-arena recovery and update curves.

\textbf{The parameter--quality tradeoff.}
On arenas 6, 7, 8, and 16 with the same eight episodes and 4k updates, full U-Net tuning reaches 23.11 dB / 0.182 LPIPS versus the adapter's 22.88 / 0.209, recovering 87\% versus 71\% of the LPIPS rise.
It trains 205 times as many parameters; measured times are 0.83 hours on an A6000 versus 1.10 hours on an A4000, so this comparison establishes parameter economy, not a wall-clock speedup.
```

**One combined table, exact contents.** Replace original Table 1 in the body with this table. Move its full zero-shot details to the supplement, using only the brief's final values. Move original Table 2 there too. Blank comparator in-domain cells inherit the U-Net reference; dashes in comparator PSNR recovery are intentional because the brief does not supply those shares. Do not copy the unapproved comparator recovery/gap cells from the preview.

```tex
\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{3.2pt}
\caption{Scene PSNR / LPIPS in domain and on unseen arenas before and after adaptation. Unseen scores and recovered shares are medians over 13 arenas; the final pair uses only arenas 6, 7, 8, and 16. Recovery is computed per arena before aggregation.}
\label{tab:unseen}
\begin{tabular}{lccccc}
\toprule
 & In domain & Zero-shot & 4k updates & \multicolumn{2}{c}{Recovered (\%)} \\
Model & PSNR / LPIPS & PSNR / LPIPS & PSNR / LPIPS & LPIPS & PSNR \\
\midrule
SD 1.4 U-Net & 25.20 / .158 & 22.30 / .303 & 23.60 / .210 & 72 & 48 \\
PixArt-$\alpha$ & 25.18 / .159 & 22.38 / .285 & 23.72 / .205 & 76 & 48 \\
SD 3.5 Medium & 25.38 / .126 & 22.09 / .262 & 24.00 / .165 & 77 & 50 \\
\midrule
U-Net LoRA (subset) & & 21.71 / .292 & 22.88 / .209 & 71 & -- \\
U-Net full (subset) & & 21.71 / .292 & 23.11 / .182 & 87 & -- \\
\bottomrule
\end{tabular}
\end{table}
```

**Original Figure 3 becomes body Figure 2.** Keep the 1.75-inch U-Net four-panel asset and `fig:adapt`; replace its long caption with:

```tex
\caption{U-Net shift and recovery: per-arena PSNR and LPIPS before and after 4k updates, and quality versus updates. Open markers: zero-shot; filled: adapted; ticks: reconstruction references; dashed: training maps. Colours group arenas by initial reconstruction gap.}
```

The supplied preview's individual-map curves may be retained; do not derive additional numbers from their axes. The five table rows, body directional scores, and statement of strict per-map separation supply the cross-backbone evidence. Keep the full all-backbone per-map evidence in the supplement already referenced.

Remove the results roadmap “We report what a model loses zero-shot, what post-training recovers and with what budget, and what predicts how far an arena gets.” The new subheads do that work. Cut the tail-statistics sentence “Pooled over the 13 arenas, one predicted frame in five has an LPIPS above 0.4, against one in 73 on the training maps; such frames concentrate in arenas 16, 17, 13 and 1 (32 to 59 percent of their windows), while seven arenas stay below 10 percent.” It competes with the stronger complete separation result and uses numbers outside the brief.

Replace the old causal sentence “The three backbones lose the same things (Table~\ref{tab:unseen}; \appref{app:distance}): the shift looks like a property of the task, not of the architecture or the scale.” with the final sentence of the new first paragraph. Replace “What the model keeps is how actions act on the world: zero-shot on arena 7 the weapon fires and the view advances as in the ground truth (Figure~\ref{fig:teaser}) and the turn response holds; the loss sits in the scene, whose rows fall 2.9~dB (25.2 to a median 22.3) while the map-independent HUD rows fall 0.2~dB (29.7 to 29.5).” with the measured directional result and the qualitative teaser, avoiding a general dynamics claim.

Cut “Most of the PSNR still missing against the training maps (1.6~dB) reflects the arenas' lower upper bound (27.3 against 28.6~dB), not the model.” The new excess-gap recovery sentence states what was measured without subtracting medians to make a causal attribution. Replace the old pending-budget and recovery sentences with the measured recovery/budget paragraph above, retaining the separation between LPIPS recovery, PSNR recovery, and excess-gap recovery.

Move these two sentences out of the body to their existing supplementary checks: “Data matters less than the first updates: on the four arenas of the data ladder, one episode already gives most of the gain of sixteen (\appref{app:perarena-adapt}).” and “The adapters trade some in-distribution skill for it: the training maps' scene PSNR against the decoded ground truth falls on all 13 arenas at the 8k check (median 0.59~dB; stock decoder), while the directional score rises from 0.81 to 0.84.” Do not use these older-protocol values as main-protocol evidence. Preserve the existence of the forgetting check in the scope paragraph below.

Move “Neither the model's own latent-space score nor a frame distance between an arena's recordings and the training maps predicts how far an arena gets (\appref{app:distance}).” to the scope paragraph, using the more measured wording below. This negative result stops future autonomous repair from sounding solved.

## Future work — replace in full and rename “Scope and implications”

```tex
\section{Scope and implications}
\label{sec:limits}
The claims cover 13 arenas of one game, one agent, one-tic scoring, and mostly one seed.
Turn reversal does not establish general dynamics fidelity or stable rollouts; adaptation's forgetting checks are reported separately in the supplement.
Neither latent-space score nor frame distance was shown to predict adaptation outcomes here, so automatic shift detection and repair remain unestablished.
The result is a measurable separation: a world model can retain turn response across a scene change while needing little adaptation to recover most of its perceptual loss.
\label{body-end}
```

Delete the opening numerical recap beginning “On unseen arenas a Doom world model keeps its turn response and its action effects and loses the arena's appearance”; the table and result paragraph already provide those numbers. Delete the unsupported historical sentence in full: “One-tic quality does not guarantee stable rollouts: SD~3.5, best at one tic on the full frame with stock decoders, collapses in 23 non-EMA and 9 EMA rollouts of 256, and the EMA removes checkpoint-specific collapses, not the failure mode (\appref{app:closedloop}).” Do not move it to the supplement.

Cut the shopping list “Next are other Doom maps and other games, to see whether the same recipe transfers; detecting a shift from the model's own zero-shot scores so that it repairs itself; closed-loop evaluation of the adapted models; and agents trained inside the adapted model.” Replace the closing aspiration “The pattern we measure, control kept, appearance lost, most of it relearned from a few episodes, is what a deployed world model would need to detect and repair on its own, and DoomShift is a testbed for that loop.” with the final sentence above. These replacements preserve ambition without promising unmeasured autonomous repair.

**Required supplement correction, using the file actually supplied:** `appendix_current.tex` still contains the `Closed loop` paragraph beginning “In SD~3.5's collapsed rollouts the latent's global scale stays inside the real range”, the `tab:collapse` table, and the `fig:collapse` figure, despite later saying SD 3.5 was not evaluated in closed loop. Remove those three old SD 3.5 blocks. Insert: “SD~3.5 was not evaluated in closed loop in this study.” Keep “$S_0$ was not computed for SD~3.5.” Remove the pending arena-7 adapter/full-tune latent comparison sentence beginning “Because a full fine-tune also moves the input projection”; insert “The adapter/full-tune comparison in this study is reported in raw-frame scene PSNR and LPIPS.” No new latent values should be solicited or invented.

# 6. Novelty paragraph

**Finished-paper novelty, expressed in one paragraph:** DoomShift makes a scene change an empirical test of which parts of a learned simulator transfer and how much repair costs. The physics and game engine are exactly the same, yet every unseen arena is perceptually worse than every training map across three pretrained backbones while turn response survives; the decoder reference shows that rendering limits alone do not explain the deficit. Standard adaptation then recovers most of the perceptual loss from eight episodes and 4k updates. The contribution is the conjunction of a sharp failure, a retained capability, and a measured recovery budget under one scene-holdout protocol. **A skeptical reviewer will say:** domain shift hurts, LoRA helps, and Doom maps are not robots. **The answer:** the paper does not claim any of those ingredients is new or establish robot transfer; it establishes the complete per-map perceptual separation alongside retained turn response, replicates that pattern across pretrained packages, and quantifies both recoverability and the quality sacrificed for parameter economy. This is a scientific characterization with a reusable benchmark, not a new adapter or a release announcement.

The framing history supports this choice: the original architecture contest evolved into an evaluation study as the interesting evidence moved from a winning row to transfer behavior. Do not revive causal pretraining/architecture claims, the older distance-predicts-transfer headline, or historical rollout-collapse claims. The brief's final results supersede that history.

# 7. Where the current-main attempt went right and wrong

**Right:**

- It accepted the substantive correction: a study title, benchmark/finding/adaptation contribution order, and established adaptation in place of an algorithm claim. Keep that direction.
- It completed the three-backbone recovery story and added the full-tune comparator. These make the study stronger than the Overleaf draft's U-Net-only result and pending comparison.
- It reduced duplicated prior work. Scenario F additionally demonstrates a viable asset strategy: method overview and full tables in the supplement, quantitative curve and slim table in the body.

**Wrong:**

- It kept a long release-heavy first contribution while retaining “so the loss follows what the model was exposed to, not its architecture or scale.” The first buries the science; the second exceeds what pretrained-package comparisons identify. The new short benchmark bullet and explicit model-package sentence fix both.
- It advertises full tuning as trailing by “only” a small PSNR difference while the relevant perceptual comparator is better for full tuning. It also leaves a backbone-independent reconstruction-gap threshold. The proposed comparator states LPIPS recovery and differing hardware times; the protocol uses each backbone's own gap.
- The main attempt remains too large; the mechanical trim fits by losing important wording and distinctions. Scenario F drops the requested final sentence of intro paragraph 2 and “a strong balance between quality and efficiency,” keeps throughput details, omits the negative predictor finding, and leaves raw metric medians easy to confuse with medians of recovery fractions. The replacements restore those essentials and spend space on the scientific claim instead of recorder and optimizer detail.

# 8. Risks

- **“What survives” can sound like a proof of learned physics.** The strongest measured retained capability is turn reversal. The title is supported only if the abstract, contribution, and scope continue to say “turn response,” not general action fidelity, dynamics transfer, or stable simulation. The engine is physically unchanged; the experiment does not prove the learned model has recovered its physics.
- **A clean story could hide the cost of repair.** Keep the full-tune LPIPS advantage in the body, preserve the supplement's forgetting checks, and say the small adapter saves trainable parameters. Eight episodes is a demonstrated sufficient setting, not a proven minimum; 4k is the reporting point, not the earliest useful update. The measured threshold budgets answer a different question.
- **Recovery fractions invite false arithmetic.** A reader dividing table medians will not reproduce the reported median per-arena shares. The definition and aggregation sentence must stay. Likewise, excess-gap recovery above complete closure does not mean perfect reconstruction; do not relabel 101% as complete visual recovery.
- **The benchmark can become underspecified if its shrinkage goes too far.** Keep the exact training/validation split, the available-versus-used adaptation split, the held-out scoring windows, raw scene crop, fine-tuned decoder convention, and per-backbone reference. Move recorder mechanics, not the experimental contract.
- **The two-figure choice makes replication less visible than the U-Net illustration.** The three complete backbone rows and directional scores must remain in the body, with a direct supplement pointer for the strict per-map LPIPS separation. Do not substitute a pooled median claim for “every unseen arena.”
- **Page pressure can erase the scientific answer again.** The 12-line allowance is for layout, not additional results. Retain normal type and the stated figure heights. If final wrapping consumes that allowance, the exact first cuts are the last protocol sentence “The supplement gives the grids and per-arena results.” and the results sentence “Table~\ref{tab:unseen} shows the same perceptual failure across the pretrained packages.” Neither removes evidence. Do not cut the comparator, recovery definition, or scope to preserve a method overview.
- **Stale supplementary evidence can undermine an otherwise accurate body.** The supplied appendix contradicts the final brief about SD 3.5 closed-loop results. Its historical collapse blocks and pending latent comparator must be removed before assembling the paper. The proposal supplies no replacement measurements and requires no new experiment.
