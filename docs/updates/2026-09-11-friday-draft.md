# Friday update draft, Sep 11, 2026 (write from the Sep 8 to 11 entries in RESEARCH_CONTEXT.md section 7)

**Goal 1, DoomDiT workshop paper (CoRL 2026 PhysWM, Sep 30).**
- Scoped the paper with Changliu: controlled DiT-XL/2 vs SD 1.4 U-Net comparison under one recipe, GameNGen's evaluation protocol plus autoregressive metrics; MultiGen-style multi-map memory is the follow-up.
- Replaced the unrefereed single-arena agent with Arnold (Lample and Chaplot 2017, the agent MultiGen uses), ported to torch 2. Recorded 850 lossless per-tic episodes on 17 maps at 320x240 with pose and HUD state (202 GB), plus a 1,000-episode continuity set from the old agent. Both to be released on Hugging Face.
- Built and validated the new training and evaluation stack (velocity objective, 32-frame context, noise augmentation, one recipe for both backbones; teacher-forced, rollout drift, FVD, inverse-dynamics action agreement).
- Launched both main runs on Superman Sep 9; DiT finishes about Sep 10, U-Net about Sep 13. Context-length sweep and decoder fine-tuning running alongside.
- Findings so far: the frozen SD VAE reconstructs Doom at only 23.7 dB (17.9 on the HUD), so decoder fine-tuning is essential; JPEG-85 targets in the public datasets cap PSNR at 31.4 dB; the IDM ceiling on real data is 69% top-1 over 29 actions.
- Next week: evaluate both checkpoints, fill tables, first full draft by Sep 19.

**Goal 3 (LEGO world model):** no work this week; the DoomDiT code (`backbones.py`, `train_wm.py`, `rollout_eval.py`) is written to be reused for it.
