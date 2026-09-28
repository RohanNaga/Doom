# Restructure brief: DoomShift workshop paper

You are one of three independent reviewers (an Opus 5.5 agent, a Fable agent, and Astra/Codex). Each of you
works alone from this brief and writes one proposal file. Nobody edits the repo, commits, or runs anything on
a server. The main session merges the three proposals into one plan for Rohan to approve.

## The paper

A 4-page workshop paper (CoRL 2026 PhysWM workshop, 4 pages of body plus references, appendix as supplement;
deadline Thu Oct 1 2026 07:59 EDT). Authors: Rohan Nagabhirava, Keerthana Chirumamilla, Changliu Liu (CMU).

Files in this directory (`/private/tmp/claude-501/-Users-rohan-Documents-Github-Doom--claude-worktrees-vibrant-ritchie-6f6280/bbd290c7-2c43-46b3-bfab-7994575157d8/scratchpad/restructure/`):

- `overleaf_draft_2026-09-27.tex`: THE BASE. This is the draft Rohan sent to his professor, Changliu Liu, on Sep 27 at 21:35 and the one her comments refer to. Propose the delta against this file, section by section. (Its SD 3.5 numbers, the 6,000-episode count and the four-tic sentence are stale; use the final numbers below.)
- `main_current_c207bb6.tex`: the repo's current main.tex, where earlier agents already applied a first attempt at her comments (new title, contributions reordered, Section 2 condensed to one paragraph, "post-training" renamed "adaptation"). Rohan is worried these agents changed too much without enough thought. Treat it as one attempt to learn from, not as the answer, and say where it went right or wrong.
- `main_scenarioF_branch.tex` and `scenarioF_preview.pdf`: a mechanical 4-page trim of the current text (figure 2 and the full tables moved to the supplement, a slim table in the body, many sentence cuts). Same status: one attempt, not authoritative. Useful for what fits on 4 pages.
- `appendix_current.tex`: the supplement as it stands.

## The professor's comments (Slack, Sep 28, verbatim)

1. "The current title, 'DoomShift: Efficient Adaptation of World Models to Domain Shifts,' sounds primarily like a new adaptation method, whereas the stronger contribution is studying what fails under a scene change and how much adaptation is needed to recover. A better title maybe: How Much Adaptation Do World Models Need? A Study of Domain Shifts in Doom"
2. "For the contributions, I would suggest re-organize it around three contributions: the benchmark, the empirical finding, and the adaptation result. Since the adapter uses established techniques, I would emphasize what the experiments demonstrate rather than imply a new adaptation algorithm."
3. "Section 2 seems repeating many points that are already discussed in Section 1. So that is a good place for cutting the content"

Rohan replied "Sounds great I will look through and make these changes today". So the comments are accepted in
spirit; the question is how to do them well.

## Rohan's priorities (his words, Sep 28 evening)

- keep the paper strong
- make the novelty claim strong
- as much as we can about the exciting stuff, not about the dataset benchmark
- still thinking about what the professor is doing

Tension to resolve: Changliu leads with "the benchmark" as contribution 1 and reframes the paper as a study;
Rohan wants the paper to read as exciting science, not a dataset paper. Your job is to find the framing that
satisfies both: a study whose findings are the headline, where the benchmark is the instrument, and where the
adaptation result is presented as a measurement (how little it takes) rather than a method.

Earlier standing decisions of Rohan's that still hold (do not fight them): the shift is described as "physics and
game engine exactly the same, only the surroundings change"; three backbones initialized from pretrained
diffusion image models; 13 out-of-distribution maps; "about 1 A4000 GPU-hour"; 4k updates reported everywhere
with 8k as the check; 10-step DDIM as "a strong balance between quality and efficiency"; no throughput/fps
claims; no "level pack"; keywords "World models, Few-shot adaptation"; the abstract's numbers in blue (\prov);
Figure 1 simplified with a 35-word caption; the intro's second paragraph ends with the "We therefore explore
what it takes to close the domain gap ..." sentence.

## Final numbers (all through each backbone's fine-tuned decoder, scene crop rows 0 to 207, raw frame; medians over the 13 unseen arenas)

| Backbone | in-domain PSNR / LPIPS | zero-shot unseen | after 4k adapter updates | recovered LPIPS / PSNR / excess gap | budget (updates) |
|---|---|---|---|---|---|
| SD 1.4 U-Net (860M) | 25.20 / 0.158 | 22.30 / 0.303 | 23.60 / 0.210 | 72% / 48% / 94% | 150 |
| PixArt-alpha (628M) | 25.18 / 0.159 | 22.38 / 0.285 | 23.72 / 0.205 | 76% / 48% / 101% | 150 |
| SD 3.5 Medium (2.27B) | 25.38 / 0.126 | 22.09 / 0.262 | 24.00 / 0.165 | 77% / 50% / 94% | 250 (first point of its grid) |

- Reconstruction upper bound: U-Net and PixArt decoder 28.55 dB training maps, 27.32 unseen; SD 3.5 fine-tuned decoder 31.84 training. In-domain gap to the bound: 3.36 (U-Net), 3.37 (PixArt), 6.46 (SD 3.5).
- Directional score (turn response): training 0.855 / 0.842 / 0.840, unseen 0.804 / 0.808 / 0.800, ground-truth frames 0.885 / 0.892. So the turn response survives the shift on all three.
- Full fine-tune of all U-Net parameters on arenas 6, 7, 8, 16: zero-shot 21.71 / 0.292 to 23.11 / 0.182 at 4k, against the LoRA adapter's 22.88 / 0.209 on the same arenas; LPIPS recovered 87% against 71%; 205 times the trainable parameters; A6000 0.83 h against A4000 1.10 h per arena.
- Adapter: rank-16 LoRA on every attention projection plus control MLP, input projection, noise-bucket embedding: 4.2M parameters, 0.49 percent; 8 adaptation episodes; about 1 A4000 GPU-hour per arena for 4k updates; 4k is 2 percent of the 200k pretraining updates and captures on average 96 percent of the 8k gain.
- Decoder fine-tune (SD 1): scene reconstruction 26.9 to 28.4 dB, LPIPS 0.092 to 0.060; the decoder reconstructs unseen arenas nearly as well as training maps (27.3 vs 28.6), so the shift lives in the model.
- Persistence (copy last frame) 19.8 dB on the unseen arenas (18.5 to 22.5).
- The shift is a step, not a slope: every unseen arena has worse LPIPS than every training map, for all three backbones.
- Data: 17 Arnold/ViZDoom deathmatch arenas; 4 training maps (2 to 5), 500 training episodes each, 25 validation episodes per map (512 windows); 13 unseen arenas, 24 episodes each, 16 for adaptation (8 used), 8 held out (256 windows). Release: 8,000 episodes on the four maps (6,000 train / 1,000 val / 1,000 test) and 24 per unseen map, on the Hub. One-tic scoring, mostly one seed. No SD 3.5 closed-loop or latent-skill numbers exist (removed from the paper).
- Things not shown to predict how far an arena gets: the model's own latent-space score, a frame distance to the training maps.

## What to produce

Write ONE markdown file, `proposal_<yourname>.md`, in this directory (`yourname` = opus, fable, or astra), with these sections, in this order:

1. **Thesis in one sentence.** What the paper claims, in words a reviewer would repeat.
2. **Title.** Changliu's title, or a variant, with a one-line reason. Keep "DoomShift" as the benchmark's name inside the paper regardless.
3. **Outline for 4 pages.** Section list with, for each: its job, its length budget in lines (a CoRL page is about 50 lines of body text; figures and tables count), and which figure/table it carries. Say what moves to the supplement. The current assets: Figure 1 teaser (1.5 in tall), Figure 2 method overview (1.8 in), Figure 3 four-panel shift-and-repair (1.75 in; body variants of 3.1 in and 1.6 in exist), Table 1 (zero-shot, three backbones, 6 rows), Table 2 (adaptation cost by arena group, four blocks), and a slim combined table (in-domain / zero-shot / 4k / recovered shares, five rows).
4. **Contributions.** The three bullets in full, final wording, in Changliu's order (benchmark, empirical finding, adaptation result) unless you argue for a different order and say why. Make the finding the exciting one.
5. **Delta against the Overleaf draft.** Section by section (abstract, intro, prior work, method, results, future work): keep / cut / rewrite / move, with the actual new wording for anything rewritten (not "tighten"). Quote the old sentence when you cut it. Be concrete enough that a writer can apply it without judgment calls.
6. **Novelty paragraph.** How the finished paper makes its novelty claim in one paragraph, and what a skeptical reviewer would say against it, and the answer.
7. **Where the current-main attempt went right and wrong** (three bullets each, at most).
8. **Risks.** What could make this restructure weaker than the Overleaf draft.

Constraints on your proposal: everything must fit 4 pages (count lines honestly); use only the numbers above;
do not propose new experiments (nothing can run before the deadline); keep Rohan's standing decisions; the
paper is not a dataset paper, so the benchmark gets described in the fewest lines that make the results
reproducible, and the release is one sentence.

Work alone. Do not read the other proposals if they appear. Do not edit the repo.
