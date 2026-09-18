# Backbone candidates — open-weight image DiTs for a better-ceiling latent space

Companion to `backbones-and-warm-starts.md` (same schema). Scope: open-weight image diffusion transformers 2023 to mid-2026, under about 3B parameters, warm-startable as a per-frame action-conditioned Doom world model in a latent space with a better reconstruction ceiling than the SD 1.x KL-f8 4-channel VAE. Target: 320x256 frames, one 49 GB A6000, batch 32, 90k updates, bf16 with fp32 master weights, 32 stacked context latents on the channel axis, one discrete action. A URL per fact; "not stated" means the source did not say it; **VERIFY** means untraced.

## The ceiling, first

SD3 trains one autoencoder at three widths (Table 3, https://arxiv.org/html/2403.03206v1): 4 ch — rFID 2.41, PSNR 25.12, SSIM 0.75; 8 ch — 1.56 / 26.40 / 0.79; 16 ch — 1.06 / 28.62 / 0.86. They "thus choose d=16" because "the d=16 autoencoder exhibits better scaling performance in terms of sample FID". So 16 channels buys ~3.5 dB and halves rFID at the same f8 compression.

Our own ceilings for calibration (`RESEARCH_CONTEXT.md`, Sep 9 and Sep 16/17): frozen `sd-vae-ft-mse` 23.7 dB / LPIPS 0.095; decoder fine-tuned 28.61 / 0.051 seen, 26.36 / 0.070 unseen; Wan video VAE 24.83 / 0.122. **Our fine-tuned 4-channel decoder already sits at SD3's generic 16-channel number**, so the honest claim is "16 channels plus the same decoder fine-tune", not 3.5 dB free.

The SD3/Flux VAE is f8, 16 channels, scaling 0.3611, shift 0.1159 (Flux VAE config as republished in Lumina 2.0: https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0/blob/main/vae/config.json). HF's SD3 post calls it "a 16 channel AutoEncoder model that is similar to the one used in Stable Diffusion XL" (https://huggingface.co/blog/sd3) — similar in architecture only: `stabilityai/sdxl-vae` is `latent_channels: 4`, `scaling_factor: 0.13025` (https://huggingface.co/stabilityai/sdxl-vae/blob/main/config.json). Three otherwise plausible candidates are therefore not ceiling upgrades and are left out of the table: PixArt-Sigma XL/2 (0.6B, `PixArtSigmaPipeline`, diffusers ">= 0.28.0", CreativeML Open RAIL++-M, https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-512-MS), Hunyuan-DiT ("2B params", CLIP + mT5 cross-attention, `HunyuanDiTPipeline`, tencent-hunyuan-community, VAE `latent_channels: 4`, https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-Diffusers) and Lumina-Next-SFT ("2B parameters", Next-DiT, Apache 2.0, VAE "stabilityai/sdxl-vae", https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT-diffusers).

Higher compression with wider channels is the other route. DC-AE, ImageNet 256 (Table 2, https://arxiv.org/html/2410.10733v3): f32c32 rFID 0.69 / PSNR 23.85 / LPIPS 0.082 vs a retrained SD-VAE f32c32 at 2.64 / 22.13; f64c128 at 512 reaches 0.22 / 26.15. Not f8c4 comparisons, and 23.85 dB is *below* our fine-tuned decoder. EQ-VAE (https://arxiv.org/abs/2502.09509) is a 5-epoch regularization fine-tune of any existing VAE, claiming better rFID with "no trade-off in reconstruction quality" and 7x faster DiT-XL/2 convergence — an orthogonal add-on, not a backbone.

## Summary table

| model | params | arch / conditioning | VAE | 320x256 grid → tokens | license | diffusers | world-model use |
|---|---|---|---|---|---|---|---|
| SD 3.5 Medium | 2B | MMDiT-X, dual attn in first 12 layers, 3 text encoders, flow matching | 16 ch f8 | 32x40x16, patch 2 → 320 (+text stream) | Stability Community, free under $1M revenue | `StableDiffusion3Pipeline`, gated | none found |
| Lumina-Image 2.0 | 2.6B | Unified Next-DiT, text+image one joint sequence, no cross-attn, Gemma-2-2B | 16 ch f8 (Flux) | 32x40x16, patch 2 → 320 | Apache 2.0 | `Lumina2Pipeline` | none found |
| Sana 1.6B | 1648M, 20 layers, hidden 2240, patch 1 | linear-attention DiT, Gemma2-2B-IT | DC-AE f32c32 | 8x10x32, patch 1 → **80** | Apache 2.0 + Gemma terms | `SanaPipeline`, diffusers from git | **family yes** (SANA-WM) |
| CogView4-6B | 6B | DiT, GLM-4-9B | 16 ch f8, scaling 1.0, shift 0.0 | 32x40x16 → 320 | Apache 2.0 | `CogView4Pipeline` | none found |

Ruled out on size, all open-weight: FLUX.1-schnell 12B, Apache 2.0, `FluxPipeline`, 16 ch f8 (https://huggingface.co/black-forest-labs/FLUX.1-schnell) — **no small Flux exists**, community "lite" checkpoints are 8B+; Qwen-Image 20B, Apache 2.0, `QwenImagePipeline` (https://huggingface.co/Qwen/Qwen-Image); CogView3-Plus 3B, Apache 2.0, `CogView3PlusPipeline`, VAE **VERIFY** (https://huggingface.co/THUDM/CogView3-Plus-3B). SD 3 Medium is SD 3.5 Medium's predecessor: 2B, MMDiT, same license and pipeline, gated, "1 billion images" plus 30M aesthetic and 3M preference (https://huggingface.co/stabilityai/stable-diffusion-3-medium).

## Notes that matter for the port

**SD 3.5 Medium** (https://huggingface.co/stabilityai/stable-diffusion-3.5-medium): "2B params", "MMDiT-X" with "dual attention blocks in the first 12 transformer layers"; "OpenCLIP-ViT/G, CLIP-ViT/L" at 77 tokens plus "T5-xxl" at "77/256 tokens at different stages"; "Progressive training stages: 256 → 512 → 768 → 1024 → 1440 resolution". Depth and width are not on the card; the SD3 paper's rule is "hidden size to 64·d ... and the number of attention heads equal to d", making 2B depth 24 / hidden 1536 / 24 heads — **inferred, VERIFY; the transformer config returns HTTP 401 until the gate is accepted**. The diffusers docs confirm the gate: "the model is gated, before using it with diffusers you first need to go to the ... page, fill in the form and accept the gate" (https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_3). MMDiT's text stream is a full parallel weight set per block, so with no captions we either push two learned action/bucket tokens through it (as for PixArt) or delete it and lose much of the pretrained mass.

**Lumina-Image 2.0** (https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0; https://arxiv.org/abs/2503.21758): "2 billion parameter flow-based diffusion transformer" on the card, "2.6B parameters" in the abstract, Apache 2.0. Transformer config: `in_channels 16`, `num_layers 26`, `num_attention_heads 24`, hidden 2304, `patch_size 2`. Architecturally the friendliest: one joint self-attention stream, nothing to preserve or delete surgically, so action and bucket become two prefix tokens.

**Sana 1.6B** (https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers; https://arxiv.org/abs/2410.10629): "1648M parameters", "DC-AE" with "32x spatial-compressed latent feature encoder"; config `in_channels 32`, `num_layers 20`, `num_attention_heads 70`, `attention_head_dim 32`, `patch_size 1`. f32 is the story: our frame becomes 8x10x32, **80 tokens** instead of 320, 32 context latents become 1024 input channels, and latent storage drops from about 11 GB to about 5.4 GB (estimate). SANA-1.5 scales "from 1.6B to 4.8B parameters" and ships a "memory-efficient 8-bit optimizer" (https://arxiv.org/abs/2501.18427). Only family with a published action-conditioned descendant: SANA-WM, "a 2.6B-parameter diffusion-based world model" with "dual-rate camera conditioning" (https://arxiv.org/abs/2605.15178) — the *video* line, and it swaps in an LTX2 tokenizer.

**CogView4-6B** (https://huggingface.co/THUDM/CogView4-6B) has the right VAE (`latent_channels: 16`, `scaling_factor: 1.0`, `shift_factor: 0.0`) and Apache 2.0, but 6B is out of budget. **Z-Image-Turbo**, Nov 2025, is 6B Apache 2.0, "Scalable Single-Stream DiT (S3-DiT)" where "text, visual semantic tokens, and image VAE tokens are concatenated at the sequence level", "8 NFEs", "within 16G VRAM consumer devices" — inference, not training; VAE not stated **VERIFY** (https://huggingface.co/Tongyi-MAI/Z-Image-Turbo).

## Memory at batch 16 and 32, 40x32 grid, 32 context latents — ESTIMATES

Anchor (`RESEARCH_CONTEXT.md`, Sep 17 00:40): DiT-XL/2 (675M, 28 layers, hidden 1152, 320 tokens) with gradient checkpointing ran per-GPU batch 16 at **12.29 GB peak allocated**. AdamW with fp32 master, two fp32 moments and bf16 weights/grads is ~16 bytes/param, so DiT-XL's states alone are 10.8 GB and **activations at batch 16 are only about 1.5 GB**. Scaling activations by layers x hidden x tokens x batch:

| model | states (16 B/param) | act. b16 | act. b32 | total b32 | fits 49 GB? |
|---|---|---|---|---|---|
| DiT-XL/2 (today) | 10.8 GB | 1.5 (derived) | ~3.0 | ~14 | yes, wide |
| PixArt 0.6B | 9.6 GB | ~1.4 | ~2.8 | ~12 | yes |
| Sana 1.6B (80 tok) | 26.4 GB | ~0.5 | ~1.0 | ~27 | yes, comfortable |
| SD 3.5 Medium 2B | 32.0 GB | ~2.1 | ~4.2 | ~36 | yes, tight |
| Lumina 2.0 2.6B | 41.6 GB | ~2.8 | ~5.6 | ~47 | no usable headroom |
| CogView3-Plus 3B | 48.0 GB | — | — | >48 | no |
| CogView4 / Z-Image 6B | 96.0 GB | — | — | — | no |

Optimizer state, not activations, binds: **fp32 master weights plus AdamW put the single-card ceiling near 2B parameters.** Above that needs 8-bit Adam, bf16 master weights or offload, each a recipe change that breaks comparability with the existing rows. Two further porting costs (estimates): 16-channel re-encoding takes latent storage from about 11 GB to about 43 GB for 1.05M fp16 frames, and the stacked-context patch embedding becomes 528 input channels instead of 132.

## What would change the decision

1. **A measured ceiling on our own frames.** The 3.5 dB is ImageNet-generic. Encode 2,048 held-out Doom frames with the Flux/SD3 VAE, the CogView4 VAE and DC-AE f32c32; score PSNR / LPIPS / HUD PSNR against our fine-tuned 28.61 dB. Under about 1 dB on HUD-heavy frames, the re-encode is not worth it.
2. **Whether the finding survives the latent change.** Our headline is about pretraining exposure *in the target latent space*. A 16-channel space retires the SD 1.4 U-Net and DiT-XL rows as comparators, since nothing else is pretrained there at small scale — this looks like the next paper's axis, not a new row in this one.
3. **The 2B optimizer wall.** If 8-bit Adam or bf16 master weights are acceptable, Lumina-Image 2.0 is the cleanest architectural fit and CogView4 comes into range. If fp32 master is non-negotiable, the field is SD 3.5 Medium, and its gate plus the $1M revenue clause need a decision.
4. **Sana is a different experiment.** 80 tokens means roughly 4x throughput and the only action-conditioned descendant, but DC-AE f32c32's 23.85 dB is below our ceiling — backwards from this goal unless the measured Doom number surprises.
