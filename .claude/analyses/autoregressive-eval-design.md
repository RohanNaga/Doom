# Autoregressive evaluation suite: design (Sep 1, 2026, for review before coding)

Teacher-forced PSNR/LPIPS says how good one denoising step is. Reviewers in 2026
want to know what happens when the model feeds on its own output. Three metrics,
one shared rollout engine.

## Shared rollout engine (`rollout_eval.py`, generalizes `rollout_video.py`)

- Input: a held-out episode, a start index, horizon H, the ground-truth action
  sequence a_{t..t+H}. Seed with 4 real frames, then predict, slide, repeat.
- Batch many rollouts at once (B windows x H steps); each step is one sampler
  call on a (B,16,16,20) context. At 50 respaced steps and ~5 s per 8 frames on
  one A4000, 64 rollouts x 32 frames is about 20 minutes per checkpoint.
- Output: predicted latents (B,H,4,15,20) fp16 on disk plus the matching GT
  latents and actions. Every metric below reads that file, so the expensive
  sampling happens once per checkpoint.
- Sampler and step count are flags (respaced DDPM 50 by default, DDIM as the
  alternative), same as `eval_metrics.py`.

## Metric 1: drift vs horizon (cheapest, most diagnostic)

For each horizon h in 1..H, PSNR and LPIPS of decoded frame h against the GT
frame h, averaged over rollouts, with the copy-last-seed-frame baseline. Also
the latent-space distance. Plot: metric vs h, one curve per model. This is
the "quality-vs-horizon" curve GameNGen and DIAMOND show; it directly exposes
the drift the fixed 4-frame context is expected to cause.

Question to settle: whether to also report drift under *predicted* actions.
No: the model does not produce actions, and GT actions keep the comparison
between backbones controlled.

## Metric 2: FVD vs rollout length

FVD needs an I3D feature extractor over 16-frame (or more) clips at 224x224.
Compute FVD between generated clips and GT clips of the same windows for
lengths 16 and 32 (GameNGen reports 16 and 32). Use the standard
`i3d_torchscript.pt` from the StyleGAN-V / common FVD implementations, resize
120x160 -> 224x224 bilinear. Sample size matters: FVD is biased at small N;
use N >= 256 clips per length and report N. Generated clips include the 4
seed frames? No: score only predicted frames so the seed does not inflate the
number; state this in the paper.

Cost: 256 rollouts x 32 frames ~ 80 minutes per checkpoint on one GPU.

## Metric 3: action-following accuracy via an inverse dynamics model

Train a small IDM on *real* frame pairs from the training episodes:
input (latent_t, latent_{t+1}) -> 18-way action logits. Architecture: a small
conv net on the 8-channel latent stack, ~1M params, cross-entropy, 20 minutes
on one GPU. Validate on held-out real pairs and report its accuracy there:
that number is the ceiling of the metric (MineWorld does the same).

Then run the IDM on every consecutive predicted pair in the rollouts and score
agreement with the conditioning action. Report accuracy overall and per action
class, and as a function of horizon. Under the same action label, the metric
rewards a model whose rollout actually moved the camera / player the way the
action says.

Two subtleties to resolve before coding:
1. The action-offset question (does a_t precede or follow frame t). The IDM
   is trained on real data under whichever convention we pick, so it stays
   consistent with the world model, but the paper must state it.
2. Many VizDoom actions are visually near-indistinguishable over one skipframe-4
   step (e.g. attack vs no-op when nothing is in view). The IDM ceiling will
   reveal that; report top-1 and a "movement-only" collapsed accuracy over
   {forward, back, turn-left, turn-right, strafe-left, strafe-right, other}.

## Table for the paper

| model | PSNR/LPIPS (val, TF) | PSNR@h=8/16/32 | FVD16 / FVD32 | IDM acc (ceiling) |
| U-Net | | | | |
| DiT-XL/2 | | | | |
| DiT-XL/2 from scratch (ablation) | | | | |

Plus figure: drift curves (both backbones) and a rollout strip.

## Order of implementation

1. `rollout_eval.py` (engine + drift curves) - reuses `doom_data`, `doomdit_utils`.
2. `train_idm.py` + IDM scoring inside the same script family.
3. FVD (I3D weights download, clip resize) last; it is the slowest and the
   least informative per GPU-hour.
