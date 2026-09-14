# Brief for Astra, Sep 14 night: a video-pretrained row (Cosmos or otherwise) before Sep 30

## The question
Rohan asks: what if we add a row built on NVIDIA Cosmos, swapping the VAE and whatever else is needed, or on any other video-pretrained model? Propose the best design you can for a video-pretrained third or fourth row of the DoomDiT comparison, or argue that we should not. Give your own approach first; I will put mine on the table after your first pass.

## Where we are
- Paired runs 030 (DiT-XL/2, ImageNet warm start) and 031 (SD 1.4 U-Net) on Spiderman finish Mon Sep 15 (~09:00 and ~19:00 EDT). U-Net leads by ~0.011 held-out v-loss at every matched step. Evaluation fires automatically (`after_run2.sh`), then GPU 2 runs PixArt-alpha 512 (033, identical recipe, ~36 h) and GPU 3 runs the second ImageNet-DiT seed (032, ~30 h).
- Round 4 of our pairing concluded: the gap is pretraining exposure in the SD latent space, not architecture or hyperparameters; every strong transformer world model starts from a video-pretrained checkpoint. The evidence cards from four Opus readers are in `.claude/analyses/lit/` (doom-world-models.md, dit-world-models-a.md, dit-world-models-b.md, backbones-and-warm-starts.md). Read backbones-and-warm-starts.md and the Cosmos / Matrix-Game / Yume / Nano World Models cards first.
- Deadline: CoRL PhysWM workshop, Sep 30 AoE, 4 pages. Draft Sep 20, final Sep 28. Compute: Spiderman 4x A6000 48 GB shared (GPU 1 is crowded by others, ~12 GB free; GPUs 2 and 3 are ours after the current runs), Superman 8x A4000 16 GB (idle for us; can run VAE encoding). No H100s.
- Data: 850 Arnold episodes on 17 Freedoom maps, every tic lossless at 320x240 (202 GB parquet in `/sata2/data/rnagabhi/doom/raw_arnold`), 4.21M tics, 1.05M decision frames; every action is held exactly 4 tics, and current latents are one SD KL-f8 latent (4x32x40) per decision frame. A fresh seeded evaluation corpus (60 seen-map, 20 unseen-map episodes) exists with raw frames. Metrics: teacher-forced PSNR/LPIPS against raw lossless frames, 256x64 rollouts with drift, IDM action agreement, FVD 16/32, copy-last and VAE-ceiling references.
- Code: `backbones.py` (three backbones behind one interface, channel-stacked context, v-prediction DDPM), `train_wm.py`, `eval_tf.py`, `rollout_eval.py`, `diffusion_v.py`. All on main.

## Constraints and claims to test
- A video model changes the latent space (3D causal VAE, e.g. Wan 2.1 VAE 16 ch, 4x temporal, 8x8 spatial) and the objective (flow matching / EDM), so it cannot share the recipe literally. The paper can still compare in pixel space against raw frames with per-row VAE ceilings. Is that an honest comparison, and how should it be framed?
- One latent frame per 4 tics would align exactly with one action per latent frame. Is that the right data layout, or should decision frames be the video?
- Which checkpoint: Cosmos-Predict1 7B (CV8x8x8, EDM), Cosmos-Predict2 2B Video2World (Wan VAE?, action-conditioned sample recipe exists), Cosmos-Predict2.5 2B, Wan2.1-T2V-1.3B, SkyReels-V2 DF or I2V 1.3B (Matrix-Game 2.0's parent), CogVideoX-2B, Open-Sora 1.1 (our VAE, 700M), something else? Weigh: memory on a 48 GB card, diffusers support, published action-conditioning recipe, licence, prior strength, engineering days.
- Context length and conditioning: frames-as-tokens with per-frame action embeddings (additive, Nano World Models priced this at 0 params) vs Matrix-Game's mouse-concat + key cross-attention vs GameFactory. Clean-context concat + mask (Cosmos/Matrix-Game 1.0) vs diffusion forcing (SkyReels DF, Oasis).
- Compute-matching: we cannot give this row 2.9M samples. What is the fair reporting?
- What to drop or reschedule to make a GPU available (the second DiT seed on GPU 3? PixArt on GPU 2?), and what is the fallback if engineering slips past Sep 22.

## What I want back
1. Your proposed design: checkpoint, latent layout, conditioning, objective, context length, batch, steps, expected memory and throughput on one A6000, engineering plan in days, and what the row would let the paper claim.
2. The runner-up and why it lost.
3. The honest answer to whether a VAE swap breaks the paper's control story, and the wording that survives review.
4. Anything you can verify yourself (diffusers class names and versions, model card numbers, licences) with the source quoted. Say "not verified" otherwise.
