# Astra round 4 brief, Sep 14 2026, 19:30 EDT: the DiT is losing. What, if anything, do we change?

## The measurement
Corrected paired runs, one recipe (v-prediction MSE, linear betas, L=32 context on channels, lr 5e-5 warmup 2000, AdamW wd 0, clip 1.0, global batch 32, action dropout 0, GameNGen context-noise augmentation, bf16 autocast, fp32 master), one A6000 each, 90k updates planned.

Held-out v-loss on 1,024 fixed windows (identical corruption for both):

| step | DiT-XL/2 (675M, ImageNet-256 warm start) | SD 1.4 U-Net (860M, SD warm start) |
|---|---|---|
| 5k | 0.2535 | 0.2377 |
| 10k | 0.2423 | 0.2286 |
| 20k | 0.2324 | 0.2208 |
| 30k | 0.2273 | 0.2159 |
| 38k | 0.2246 | 0.2130 |
| 52k | 0.2206 | (39k: 0.2121) |

Gap about 0.011 from 10k on, not closing; the DiT curve is flattening (0.2222 at 45k to 0.2206 at 52k), the U-Net's still falling. On the fresh test corpus against raw frames at 31k vs 24k: DiT 20.0 dB / LPIPS 0.362, U-Net 20.5 dB / 0.324, copy-last 19.2 dB, VAE ceiling 28.4 dB / 0.051. Diagnostics: no excursions, grad norm mean 0.10 (DiT) / 0.15 (U-Net), clipping never fires, adaLN update ratio 7e-5.

DiT context sweep at 5k updates on corrected data: L=2 0.2631, L=4 0.2561, L=8 0.2532, L=32 0.2535 (L=16 pending). The DiT does not use context beyond 8 at 5k.

Throughput on one A6000 (checkpointing now off): DiT batch 32 about 1.6 updates/s alone, U-Net batch 16x2 about 1.4; shared cards currently halve that. Memory: DiT 24 GB, U-Net 20 GB of 48.

## Asymmetries we accepted and stated
1. Warm start: ImageNet-256 class-conditional (1.3M images) vs Stable Diffusion 1.4 (400M image-text pairs, same VAE latent space, natural images at 512).
2. Parameters: 675M vs 860M.
3. Conditioning pathway: adaLN-Zero (t + action + bucket) vs cross-attention token + class embedding.
4. Learned-variance head disabled on the DiT to share the loss.
5. DiT input projection inflated 4 to 132 channels with zeros on the 128 context channels; U-Net conv_in likewise. Positional embedding rebuilt for 16x20.

## Constraints
- Deadline Sep 30 AoE, 4 pages; draft Sep 20. Today is Sep 14 evening. The current pair finishes Sep 15 (DiT about 13:00 EDT, U-Net Sep 16 morning at current shared rates); evaluation about 4 h each after.
- Compute: Spiderman A6000s, one card per run, shared with other users (no allocation rule); Superman 6 idle A4000 16 GB (needs checkpointing, 0.9 steps/s for the DiT on four cards) now with 329 GB disk free.
- Realistic GPU budget before the draft: about 4 card-days on Spiderman plus Superman.
- Already queued after the DiT finishes: second DiT seed (identical recipe). Rohan is open to replacing it.
- Data: 1.05M decision frames on 17 maps; recorder can add 850 episodes overnight.

## Questions, yours first
Rohan asks: should we use a larger DiT? should we change training parameters? Rank every intervention you would consider by expected gain per GPU-day toward the paper's actual question (what changes when the backbone is swapped under a shared recipe), with the literature that supports it, and say for each what single check would tell us within an hour whether it is worth the day. Candidates I have not evaluated and want your independent take on: a larger DiT (is there any pretrained DiT larger than XL/2 in this latent space? is from-scratch at this budget sane?); a DiT warm-started from natural-image weights in the SD latent space (PixArt-alpha 600M, cross-attention conditioning); DiT-XL/2-512 weights; learning rate (higher for the DiT only, or a schedule); longer training for the DiT (it is faster per step: what does compute-matched look like); EMA-evaluated checkpoints; L=8 instead of 32 for the DiT; parameter-matched from-scratch pair; unfreezing the input projection's zero-init differently; anything in the DiT's adaptation that the U-Net gets for free (e.g., the SD U-Net saw noise-augmented context-like inputs? no; but it saw this VAE's latents at scale).
Also answer: given the gap, is "no DiT advantage at this budget" the honest headline, and what one additional run makes that claim robust rather than weak?
