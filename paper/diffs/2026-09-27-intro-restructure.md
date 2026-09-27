# Introduction restructure draft (Sep 27 2026, 15:55 EDT)

Keerthana: the introduction must read bird's-eye, procedure out, prior work before Benchmark and recipe. Drafted against main 17d4828; compiles with 0 errors; costs 13 lines past the four-page body. Decision pending on what pays (Figure 2 to the appendix frees about 12 lines).

## New introduction

```latex
\section{Introduction}
\label{sec:intro}

Game world models now render playable worlds from pixels and controls: GameNGen runs Doom inside a diffusion model~\citep{valevski2024gamengen}, DIAMOND learns Atari and Counter-Strike in pixel space~\citep{alonso2024diamond}, and Matrix-Game~2 and MultiGen scale the recipe to open worlds and many games~\citep{he2025matrixgame2,po2026multigen}.
Each is trained and judged in one environment: its score is a held-out trajectory of its training scenes, and no published Doom world model is scored on a map it never saw.

A learned simulator is only useful where it is faithful, and a robot or an agent that moves to a new room, level or workspace needs to know how much of its world model still holds there and what it costs to make the rest hold.
Game worlds let us ask that question with the engine held fixed and only the environment changed.
So we ask: when a world model meets a new arena of the same game, what does it keep, what does it lose, and how many episodes and updates does it take to relearn the rest?

We contribute (1) a few-episode adaptation recipe, a rank-16 adapter trained on eight episodes of the new arena, and the per-arena protocol that scores it, under which 9 of 13 unseen arenas close half of their gap to the in-distribution advantage within 4k updates (Figure~\ref{fig:adapt}; full fine-tune comparator \tbd{});
(2) the shift itself: off the training maps the decoded advantage shrinks and the perceptual margin changes sign for all three backbones while the turn response survives (Figures~\ref{fig:teaser} and~\ref{fig:shift});
(3) the frozen model's zero-shot latent skill predicts where an arena ends up (Spearman 0.90, exploratory at $n=13$) while no footage distance orders the arenas, and one-tic quality does not guarantee closed-loop stability; and
(4) DoomShift, the 17-arena benchmark scored per arena, released with the three models, the decoder fine-tune and the adapter path (Figure~\ref{fig:method}; link withheld for review).

```

## New section

```latex
\section{Prior work}
\label{sec:prior}

Held-out evaluation of world models has meant a new embodiment~\citep{chen2026xeworld}, a new game or environment~\citep{rigter2024avid,gao2025adaworld} or a new building~\citep{koh2021pathdreamer}, never a new scene of the same game scored one by one; Doom map holdouts exist only for agents~\citep{lample2017arnold,wydmuch2018vizdoom}.
Adapting a pretrained video model to a new domain with a small adapter is how Vista and AdaWorld reach new driving scenes and games~\citep{gao2024vista,gao2025adaworld}; we measure what that adaptation costs per scene.
Persistence, the last frame copied forward, is the standard reference of video prediction~\citep{mathieu2016deep,lotter2017prednet,villegas2019fidelity}, and Genie reports the same paired $\Delta$PSNR~\citep{bruce2024genie}; game world models dropped it, and absolute PSNR then ranks how static a recording is.
The reconstruction upper bound, the decoder applied to the ground-truth latent, caps any latent model's score~\citep{zheng2023occworld,karypidis2024dinoforesight}, and we read every score beside it.

%===============================================================================

```

## Numbers moved into Section 4

- Protocol: "(1.1 A4000 GPU-hours per arena for 4k updates)".
- What it takes: "(median budget 1k [250, >4k]; 5 within 500)" and "on four arenas one episode gives most of the gain (supplement)".
