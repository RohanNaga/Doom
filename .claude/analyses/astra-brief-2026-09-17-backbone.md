# Brief for Astra, Sep 17 2026 11:30 EDT (round 11): the strongest backbone on the open Doom dataset

Rohan's goal, stated this morning: "train the strongest backbone model on the Doom-specific dataset to show how much better it is than what people have done previously for Doom using U-Nets." He is willing to re-encode the dataset in any VAE and to spend compute. The knob grid (round 9) stays the methodology spine; this is the headline row on top of it. Propose first; I hold my view back.

## Read these (new today, one URL per fact, VERIFY marks where unverified)
- .claude/analyses/lit/backbone-candidates-image.md (SD 3.5 Medium 2B, Lumina-Image 2.0 2.6B, PixArt-Sigma, Sana, Flux, Hunyuan-DiT, CogView4, Z-Image; VAE ceilings; memory wall from our fit checks)
- .claude/analyses/lit/backbone-candidates-video.md (Matrix-Game 2.0, Cosmos-Predict2.5 2B, Wan 2.2 5B, HunyuanVideo 1.5, LTX-2, HY-World 1.5, LingBot-World, Yume, Oasis; decoder fine-tune precedent)
- .claude/analyses/lit/doom-baselines-2026-09-17.md (GameNGen, PlayGen, DIAMOND, MultiGen, Ha and Schmidhuber, open reproductions; how a comparison can be phrased)
- RESEARCH_CONTEXT.md entries dated 2026-09-17 (video row numbers; UniDiffuser restore; Superman)

## Facts that constrain the choice
- Our fine-tuned 4-channel SD decoder reaches 28.61 dB / 0.051 LPIPS on seen Doom maps (26.36 / 0.070 unseen), which equals SD3's generic 16-channel number; the 16-channel ceiling after the same Doom decoder fine-tune is unmeasured.
- Stock Wan VAE ceiling on Doom: 24.83 / 0.122; the video row (SkyReels 1.3B) lands at 20.11 / 0.305 against raw frames on 512 seen windows. Cosmos-Predict2.5 2B uses the same Wan VAE. No published dB gain from domain-tuning a video decoder; decoders trained on encoded latents can be brittle on generated latents.
- Memory: 16 B/param for fp32 master + AdamW dominates; about 2B fits one 49 GB A6000 at batch 32; `train_wm.py --optim` already supports bitsandbytes AdamW8bit, which would lift that to roughly 3B.
- Compute: GPU 3 free now, GPU 2 from Fri 04:00 EDT; UniDiffuser eval needs about 8 card-hours Friday; the knob grid (12 cells at 30k on PixArt, about 120 card-hours) is also queued. Deadline Sep 30 AoE, draft Sep 20 (can slip to Sep 23), final Sep 28. A 2B row at 90k updates is about 60 h plus 8 h evaluation; re-encoding 1.05M frames through a new VAE encoder took about 2.3 h on 4 A6000 shards for the SD VAE (expect similar or 2x).
- No external Doom U-Net number is reproducible: GameNGen released nothing and its 29.43 dB is seen-map teacher-forced; PlayGen (131M DiT, 128x128, own VAE) reports PSNR 20.41 / LPIPS 0.285 at horizon 32; MultiGen's GameNGen reimplementation reports rollout LPIPS 0.442 with no architecture details. The claim must be made against our own matched SD 1.4 U-Net on the released benchmark.

## Questions
1. Backbone and latent space for the headline row: SD 3.5 Medium (2B, 16-ch, gated, community license), Lumina-Image 2.0 (2.6B, 16-ch, Apache 2.0, 8-bit Adam), Cosmos-Predict2.5 2B or Matrix-Game 2.0 (Wan latents, action conditioning built in, per-frame ceiling bounded unless the decoder is tuned), or stay in SD 1.x with the U-Net and put the compute into the grid. Rank with reasons; name the one you would launch and when.
2. The gate: design the cheapest experiment that tells us whether a 16-channel VAE (or a Doom-tuned Wan decoder) actually raises the ceiling on our frames: which VAE, decoder-only fine-tune with MSE + 0.1 LPIPS as we did for SD, how many steps, what number decides go/no-go (our current ceilings 28.61 seen / 26.36 unseen / 28.89 unseen2).
3. Schedule: can the gate, the re-encode, one 90k headline row with evaluation, the UniDiffuser evaluation, and the knob grid all fit on two A6000s (plus Superman for DiT-only cells) by Sep 27? If not, what do you cut, and in what order?
4. The claim sentence a reviewer would accept, given the doom-baselines card.
Under 700 words, numbered, cite the card line or paper for each claim. Do not commit.
