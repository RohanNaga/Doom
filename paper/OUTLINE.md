# DoomShift paper outline

**The living copy is the shared doc** https://claude.ai/code/artifact/cef8541f-35bd-4a13-9cc1-f7300e42e00b (Rohan and Keerthana edit there). Structure agreed on the Sep 27 15:09 EDT call: 1 Introduction (world models and games, why they fail to generalize and what we explore, contributions as bullets), 2 Related work, 3 Methodology (data, architectures, aim, training setup, in-distribution performance, post-training recipe), 4 Results (zero-shot deficit, post-training, what predicts the outcome), 5 Conclusion; abstract written last. The file below is the Sep 27 16:30 snapshot before that call and is kept for history.

---


Title: **DoomShift: Efficient Adaptation of World Models to Domain Shifts**. CoRL 2026 PhysWM workshop, 4 pages plus references, deadline Thu Oct 1, 07:59 EDT. Text lives in Overleaf; this outline is the plan the text follows. Status: **final** (numbers settled), **prov** (number may move when a run finishes), **tbd** (not written or not run).

## The story in three lines

1. Game world models are good in the world they were trained in and are only ever tested there.
2. Move one to a new arena of the same game and it keeps its controls but loses the arena's appearance.
3. Eight episodes and a few hundred adapter updates bring most of it back, and DoomShift is the benchmark that measures this per arena.

## Abstract (5 sentences, final wording pending Rohan's read)

| # | Sentence says | Status | Owner |
|---|---|---|---|
| 1 | World models are capable simulators but are tested primarily in the domain they were trained on. | final | |
| 2 | A small domain shift (a new arena of the same game) makes outputs drift and lose realism, though the physics is unchanged. | final | |
| 3 | DoomShift asks how quickly and cheaply a world model adapts across such a shift: 17 arenas, three backbones trained on four, the other thirteen scored zero-shot and after adaptation. | final | |
| 4 | Zero-shot result: advantage over persistence shrinks 5.1 to 2.0 dB, LPIPS worse than copying on 13 of 13, turn response survives. | final (U-Net) | |
| 5 | Adaptation result: 8 episodes and 4k updates close half the gap on 9 of 13 arenas, recover 1.8 dB, beat persistence perceptually on 11 of 13. | prov (8k set lands tonight) | |

## 1. Introduction (bird's-eye; no procedure)

| Paragraph | Says | Cites | Status | Owner |
|---|---|---|---|---|
| P1 What game world models are today | GameNGen (Doom), DIAMOND (Atari, CS), Matrix-Game 2 and MultiGen (open worlds, many games). Each trained and judged in one environment; no Doom world model is scored on an unseen map. | valevski2024gamengen, alonso2024diamond, he2025matrixgame2, po2026multigen | drafted | |
| P2 The problem and the question | A simulator is only useful where it is faithful; an agent that moves to a new room or level needs to know what still holds and what it costs to repair the rest. Games let us hold the engine fixed and change only the environment. Question: what does a world model keep, lose and relearn on a new arena, and at what cost? | none | drafted (Rohan's sentences) | |
| P3 Contributions | (1) few-episode adaptation recipe and per-arena protocol, with the 9 of 13 result; (2) the measured shift (advantage shrinks, perceptual margin flips, turn survives); (3) zero-shot skill predicts the endpoint, no footage distance orders arenas; (4) DoomShift benchmark and released models. | Figures 1 to 4 | drafted | |
| Figure 1 | Two rows (forward, attack), in-domain map beside unseen arena 7, 16 tics after context, zero-shot vs adapted vs ground truth. | | prov (brighter in-domain example from maps 3 to 5 tonight) | |

## 2. Prior work (one paragraph, four sentences)

| Sentence | Says | Cites | Status | Owner |
|---|---|---|---|---|
| 1 | Held-out evaluation elsewhere means a new embodiment, game or building, never a new scene of the same game scored one by one; Doom holdouts exist only for agents. | chen2026xeworld, rigter2024avid, gao2025adaworld, koh2021pathdreamer, lample2017arnold, wydmuch2018vizdoom | drafted | |
| 2 | Vista and AdaWorld adapt pretrained video models to new domains with small adapters; we measure what that costs per scene. | gao2024vista, gao2025adaworld | drafted | |
| 3 | Persistence is video prediction's standard reference; Genie reports the same paired ΔPSNR; game world models dropped it. | mathieu2016deep, lotter2017prednet, villegas2019fidelity, bruce2024genie | drafted (depends on the persistence decision) | |
| 4 | The reconstruction upper bound caps any latent model's score; we read every score beside it. | zheng2023occworld, karypidis2024dinoforesight | drafted | |

## 3. Benchmark and recipe

| Paragraph | Says | Evidence | Status | Owner |
|---|---|---|---|---|
| Data | 17 arenas of the Arnold level pack; ViZDoom deathmatch vs 8 bots, 150 s, every tic, 19 buttons. Four training maps (2 to 5, 500 episodes each), thirteen unseen (24 new episodes each: 16 adapt, 8 held out). | kempka2016vizdoom, lample2017arnold | final | |
| Three backbones, one training recipe | SD 1.4 U-Net, PixArt-α, SD 3.5 Medium under one next-tic recipe (v-prediction, context noise augmentation, 200k updates, EMA, 10-step DDIM). Four-tic in-distribution numbers. | Figure 2 (method overview) | prov (SD 3.5 200k tonight) | |
| Decoder | Fine-tuned SD 1 decoder (MSE + 0.1 LPIPS) for every pixel number; MSE alone triples scene LPIPS. SD 3.5 uses its own fine-tuned decoder. | decoder gate results | prov (SD 3.5 decoder column Monday) | |
| Adaptation (idea) | Keep the weights frozen, train a small adapter on a few episodes of the new arena, score at each update budget; the recipe itself is in Section 5. | | drafted | |
| What we measure | Four questions, four scores: advantage A (better than copying?), perceptual margin M (looks better than copying?), gap G (how far from the decoder's best?), directional (turns the right way?), plus zero-shot latent skill S0. Scene rows only. A_train as the in-distribution reference. | | final wording; **open decision: raw PSNR/LPIPS with persistence as a baseline line instead of A and M** | Rohan |
| Table 1 | Training maps vs unseen medians for the three backbones: PSNR, LPIPS, A, M, G, directional; persistence rows. | results/fresh_rescore | prov (SD 3.5 row at 170k) | |

## 4. What happens off the training maps

| Paragraph | Says | Evidence | Status | Owner |
|---|---|---|---|---|
| Story sentence | Zero-shot, what does a model keep and what does it lose? | | drafted | |
| Where the deficit sits | Turn response survives (directional 0.73 to 0.86 vs 0.79 to 0.91 in distribution). A positive on every arena but 5.06 to median 2.03 dB; G grows 3.4 to 5.0 dB. Perceptual margin flips sign on 13 of 13 and it is not the decoder's floor. | Table 1, Figure 3b | final | |
| Figure 3 | (a) the shift is a step, not a slope: every arena below every training map in S0 for all three backbones; distance d does not order arenas. (b) absolute PSNR and LPIPS per arena with persistence bars. | results/family_step, fresh_rescore | final | |

## 5. Crossing the shift

| Paragraph | Says | Evidence | Status | Owner |
|---|---|---|---|---|
| Story sentence | What does it take to relearn what was lost? | | drafted | |
| Protocol | Rank-16 LoRA on attention projections plus control MLP, input projection, noise embedding (0.49 percent of parameters); 8 episodes; 1.1 GPU-hours per arena; scored at 0/250/500/1k/2k/4k; budget = first grid point at half the gap to A_train, right-censored. Half the gap in dB is 34 percent of the starting error. | | prov (grid becomes 0 to 8k tonight) | |
| What it takes | Median A 2.04 to 3.80 dB; 69 percent of the gain by 250 updates; 9 of 13 close half the gap (median budget 1k, 5 within 500); one reaches A_train; one episode gives most of the gain of sixteen (four arenas). LPIPS below persistence on 11 of 13; G back to 3.4 dB. | Figure 4, Table 2, appendix ladder | prov (8k set) | |
| Figure 4 | (a) median A vs updates with per-arena curves; (b) per-arena dots of gap share with budgets. | results/adapt | prov | |
| Table 2 | Arenas past half gap, budget, A/M/G at 4k, forgetting and directional; columns LoRA median, LoRA arena 7, full fine-tune. | | tbd cells: forgetting, directional, full fine-tune (Monday) | |
| What predicts the outcome | S0 tracks the endpoint (rho 0.90 with A at 4k), not the gain; the frozen distance d separates seen from unseen but orders nothing; two transition distances fail even that. Placed after the adaptation results because it uses them. | results/distance_v2, appendix | final | |

## 6. Limitations and conclusion (three sentences)

| Sentence | Says | Status | Owner |
|---|---|---|---|
| Result | Off its training maps a Doom world model keeps its controls, loses the arena's appearance, and relearns most of it from eight episodes and a few hundred adapter updates. | drafted | |
| Limits | 13 arenas of one level pack, one agent, one-tic scoring, mostly one seed; persistence is weak where the camera moves fast. | final | |
| Closed loop | One-tic quality does not guarantee stable rollouts (SD 3.5 collapses in 23 non-EMA and 9 EMA rollouts of 256; EMA removes checkpoint-specific ones). Moved here from Section 4 so the adaptation bridge is not derailed. | final | |
| Vision | The pattern (control kept, appearance lost, relearned from a few episodes) is what a deployed world model must detect and repair on its own; DoomShift is a testbed for that loop. | final (Rohan's sentence) | |

## Supplement (appendix.tex, off in the submission build)

Protocol details; per-arena zero-shot table; the distance study and its pre-registered test; closed-loop rollouts; skill score; per-arena adaptation curves and the data ladder; public release.

## Open decisions (Rohan)

1. Persistence: keep A and M as headline scores, or report raw PSNR and LPIPS with persistence as one baseline line and define the budget against the reconstruction bound.
2. Page budget: the bird's-eye intro plus Prior work is 13 lines over; pay with Figure 2 to the supplement, or with text cuts, or by folding Prior work into the intro.
3. Ladder figure in body or supplement; seed 1 on more arenas; four-tic column.

## What lands when

| When | What | Touches |
|---|---|---|
| Sun ~16:30 | 8k grid for all 13 arenas (0 to 8k) | Figure 4, Table 2, contribution (1), abstract sentence 5 |
| Sun evening | brighter in-domain example (maps 3 to 5) | Figure 1 |
| Sun 23:00 | SD 3.5 reaches 200k; rescore about 01:00 Mon | Table 1 SD 3.5 rows, four-tic sentence, Figure 3 |
| Mon | full fine-tune comparator; forgetting and directional checks; SD 3.5 decoder column | Table 2, "efficient" in the title |
| Tue | final numbers, Astra recheck, page fit | everything |
