# Brief for Astra, Sep 16: what contribution surpasses prior Doom simulation work, with what we have and 14 days

Rohan's question, verbatim in spirit: "Can we create a unique contribution to simulating the game of Doom which surpasses previous people and previous papers, and what does that look like?" Give your own answer first; mine is withheld for this round.

## What we have (all measured, all on main; details in RESEARCH_CONTEXT.md sections 7 and 8)
- Data: 850 Arnold episodes on 17 Freedoom deathmatch maps with 8 bots, every tic lossless (202 GB), 1.05M verified decision frames (one held action per 4 tics), redistributable (Freedoom is BSD). A seeded reproducible evaluation corpus of 60 seen-map and 20 unseen-map episodes never used for any fit. GameNGen and MultiGen released nothing; PlayGen released code but not a comparable corpus.
- Evaluation: teacher-forced PSNR/LPIPS against lossless frames with copy-last and VAE-ceiling references; 256x64 rollouts with drift by horizon, copy-seed persistence reference, blur control (done: blur never raises PSNR for any model), episode-level bootstrap CIs; IDM action judge (82.9% top-1 over 29 actions; 2-frame shortcut reaches 70.1%); FVD16/32. Both DiT seeds and the U-Net were verified to score identical trajectories.
- Finished rows, one recipe (90k updates, batch 32, lr 5e-5, context 32, v-prediction, SD KL-f8 latents): DiT-XL/2 ImageNet warm start (two seeds), SD 1.4 U-Net, PixArt-alpha 512 (evaluating tonight; held-out loss 0.2030, the lowest of all rows). Numbers: DiT s0 21.06 dB / LPIPS 0.311 seen, 19.52 / 0.450 unseen, rollout PSNR@64 18.30, FVD16 232; DiT s1 21.21 / 0.307, 19.59 / 0.442, 17.68, 198; U-Net 21.36 / 0.270 (EMA 21.67 / 0.250), 19.14 / 0.446, 16.03, 184; copy-last 19.41 / 18.50; copy-seed at h64 17.86 dB / 0.510 LPIPS (every model is at or below persistence perceptually at h64); VAE ceiling 28.61 / 26.36. Episode-bootstrap CIs: U-Net's seen LPIPS lead is real; seen PSNR overlaps; DiT's unseen PSNR lead is real on 2 held-out maps; unseen LPIPS mixed; IDM overlaps. Seen-to-unseen degradation DiT 1.54 dB vs U-Net 2.23 dB.
- A video-pretrained row (SkyReels-V2 DF 1.3B, Wan VAE, L=8, flow matching, 10k updates of batch 8 = 80k samples) training now at 0.167 updates/s on GPU 3, done Wed morning; evaluator ready.
- GameNGen for reference: 29.43 dB / 0.249 LPIPS, FVD 114/186, 900M frames, 700k updates at batch 128 on 128 TPU-v5e, id's Doom, PPO agent on levels, nothing released. We are 8 dB below on PSNR and equal on LPIPS.

## Constraints
- CoRL 2026 PhysWM workshop, 4 pages, deadline Sep 30 AoE; draft Sep 20, final Sep 28.
- Compute: Spiderman GPU 2 frees tonight after PixArt's evaluation, GPU 3 after the video row (Wed morning). Superman has 6 usable A4000 16 GB (DiT fits with checkpointing at batch 32; U-Net needs micro-batches) if Rohan commits it. About 6 A6000-days total before Sep 26 realistically.
- Rohan's standing rule: follow published work over open-source-only artifacts; honest claims; he wants the strongest insight and contribution, not the most rows.

## Questions
1. What is the contribution that a PhysWM reviewer would call new relative to GameNGen, MultiGen, PlayGen, DIAMOND and Oasis, given these assets? Rank your top three candidate framings and say what each needs that we do not yet have.
2. Which single additional experiment inside the compute above most raises the paper's value? Candidates I am aware of: a many-held-out-map generalization run (train on 8 maps, test on 9, both backbones, 30k updates), a motion-magnitude test of the DiT's long-horizon "stability", a compute-matched or data-matched curve, a Cosmos-Predict2.5 2B row, a small human study. Argue for one and against the others in a sentence each.
3. What should we stop doing or leave out?
4. The one-sentence claim and the title you would submit.
