# Video / interactive-world backbone candidates for a Doom fine-tune

Compiled 2026-09-17. **Extends** `.claude/analyses/lit/video-warm-starts.md`, whose verified facts on
Cosmos-Predict1/2, Wan 2.1, SkyReels-V2, Matrix-Game 2.0, Hunyuan-GameCraft, CogVideoX-2B, LTX-Video and
Open-Sora 1.1 are not repeated. Same method: raw `curl` of arXiv HTML and huggingface/github raw endpoints.
"not stated" = source silent; **est** = my arithmetic. Our shapes: 320x256, 35 fps, a decision every 4 tics,
~1M transitions, 1 to 2 x 49 GB A6000, re-encodable.

## Summary

| Candidate | Params | VAE (T x H x W, ch) | Action cond. built in | Training code | Licence | Full FT on 48 GB? (est) |
|---|---|---|---|---|---|---|
| **Cosmos-Predict2.5-2B** | 2B | Wan-family 4x8x8, 16 | **yes**, `robot/action-cond` | **yes**, `--nproc_per_node=1` at 256x320 | NVIDIA Open Model (gated: auto) | **~33 GB, yes** |
| SkyReels-V2 DF 1.3B (our row) | 1.436B | Wan2.1 4x8x8, 16 | no (we added it) | ours | skywork-license | measured 24.65 GB |
| Matrix-Game 2.0 | 1.8B | Wan2.1 4x8x8, 16 | **yes**, per-block swappable head | **no** | MIT | ~30 GB, yes |
| Wan 2.2 TI2V-5B | 5B (dim 3072, 30 layers) | Wan2.2 **4x16x16, 48** | no | no official; DiffSynth full+LoRA | apache-2.0 | ~78 GB, needs both cards |
| HunyuanVideo-1.5 | 8.3B | own 3D causal **16x sp / 4x T, 32 ch** | no | **yes**, `train.py` + LoRA, Muon | tencent-hunyuan-community | ~127 GB, LoRA only |
| HY-World 1.5 / WorldPlay | 8.3B (HY1.5) or 5B (Wan2.2) | inherited | **yes**, dual (keys + pose) | **yes**, HY1.5 path only | tencent-hy-worldplay-community | 60 GB/GPU at sp=8 stated; no |
| Oasis 500M | 500M | own ViT, **20x spatial, 1x T, 16 ch** | **yes**, 25-dim added to t-embed | **no** | MIT (gated: auto) | trivial, but reimplement |

Ruled out on size (est footprint): LingBot-World v2 14B (CC BY-NC-SA 4.0, inference only; its "1.3b" repo
holds 3.4B of weights — VERIFY), Yume 14B / 5B, LTX-2 dev 19B (~286 GB), Cosmos3-Nano **16B** (~242 GB).

Memory anchor is our own fit check: SkyReels 1.436B at batch 8, L=8, grad ckpt peaked **24.65 GB**, of which
bf16 weights + fp32 master + Adam m/v + grad = 16 B/param = 21.4 GB, leaving ~3.3 GB of activations. Memory
therefore scales near-linearly in parameters and the est column is `16 B/param + 3.3 GB`. At 320x256 an
8x-spatial VAE with patch (1,2,2) gives a 32x40 latent, **320 tokens/frame** (our measured 2,880 over a
9-frame window); a 16x or 32x VAE gives **80**, 4x cheaper and 4x less detail to reconstruct.

## New per-model facts

**Cosmos-Predict2.5-2B — the strongest candidate.** Its diffusers pipeline declares `vae: AutoencoderKLWan`
(`pipelines/cosmos/pipeline_cosmos2_5_predict.py` line 24), so its tokenizer is the **Wan-family 4x8x8
16-channel VAE**, whose ceiling on our frames we measured at 24.83 dB / 0.122. Action post-training
is a published single-GPU recipe: `torchrun --nproc_per_node=1 ... experiment=ac_reason_embeddings_rectified_flow_2b_256_320`,
FSDP, rectified flow, lr 2^-14.5, Bridge 7-D gripper actions
([docs/post-training_video2world_action.md](https://raw.githubusercontent.com/nvidia-cosmos/cosmos-predict2.5/main/docs/post-training_video2world_action.md)).
The experiment name says **256x320, our exact resolution**, though its dataloader override reads
`bridge_13frame_480_640_train` — flag that before budgeting. Paper: 200M curated clips, 2B and 14B
([arXiv 2511.00062](https://arxiv.org/abs/2511.00062)); HF `gated: auto`, click-through not review. The
Predict1/2 action mechanism is `Mlp(in_features=7)` added to the timestep embedding, so swapping in our 29
discrete actions is small. **Neither Cosmos repo publishes a training-memory table**; the
`nproc_per_node=1` claim is unpriced.

**HY-World 1.5 / WorldPlay** is the only candidate where action conditioning *and* a trainer both ship.
Dual action representation ([arXiv 2512.14614 §3.2](https://arxiv.org/html/2512.14614v1)): discrete keys →
positional embedding → **zero-init MLP added into the timestep embedding**; continuous pose → PRoPE in a
second attention path, `Attn1 + zero_init(Attn2)`. Data 320K clips, 53% AAA game recordings. But the README
states **training at sp=8 needs 60 GB per GPU**, and `trainer/README.md` supports only the **8.3B
HunyuanVideo-1.5 path** — the Wan-5B variant is inference-only. Licence excludes EU/UK/KR and §5(b) forbids
using outputs to improve another model.

**The two high-compression backbones.** Wan 2.2 TI2V-5B: `in_dim 48`, dim 3072, 30 layers, patch (1,2,2),
VAE `AutoencoderKLWan` with a 48-length `latents_mean`; no official trainer, DiffSynth provides both.
HunyuanVideo-1.5: VAE `ffactor_spatial: 16`, `ffactor_temporal: 4`, `latent_channels: 32`
([vae/config.json](https://huggingface.co/tencent/HunyuanVideo-1.5/raw/main/vae/config.json)); `train.py`
shipped Dec 5 2025 with FSDP, LoRA and a **required Muon optimizer**. Both quarter our token count and
lower the reconstruction ceiling — the wrong trade for per-frame PSNR.

**Oasis 500M** is the closest design template: a 25-dim action vector through one `nn.Linear` **added to the
timestep embedding** driving adaLN, `max_frames=32`, Diffusion Forcing, spatial-only ViT autoencoder at 20x
with 16 channels. MIT, `generate.py` only — a design to copy, not a checkpoint. **Yume** ships Apache-2.0
training code on a Wan2.2-5B base and a reusable `decode_camera_controls_from_c2w_sequence.py`, but QCM
"are parsed into textual conditions **without introducing new learnable modules**" — keys go through T5, so
there is no action head to inherit. **Inference-only, re-checked today**: Matrix-Game 2.0/3.0, LingBot-World
v1 and v2, Oasis, Hunyuan-GameCraft, GameFactory.

## VAE reconstruction PSNR: what is actually published

**Wan 2.1 and Wan 2.2 publish no numeric VAE PSNR.** arXiv 2503.20314 §4.1.4 gives it only as a scatter plot
(Fig. 7, 200 videos, 25 frames, 720x720); the Wan 2.2 card puts its comparison in `assets/vae.png`. So **our
own 24.83 dB / 0.122 on Doom frames is the only usable Wan-VAE number for our data.** Third-party, on
unrelated sets: Wan2.1 = 40.40 dB (UCF-101), 34.13 dB (DAVIS-720p)
([Flash-VAED](https://arxiv.org/html/2602.19161v2)).

Cosmos is the only complete table ([2501.03575 Table 5](https://arxiv.org/html/2501.03575v1), DAVIS 1080p /
TokenBench): CV4x8x8 **32.80 / 35.45**, CV8x8x8 **30.61 / 34.44**, CV8x16x16 27.60 / 31.61 — frame counts
differ per row (17/49/121). LTX-Video and LTX-2 **report no PSNR at all**. HunyuanVideo
([2412.03603 Table 1](https://arxiv.org/html/2412.03603v2), ImageNet-256 / MCL-JCV) puts itself at
**33.14 / 35.39** vs Cosmos-VAE 30.07 / 32.76 and FLUX-VAE 32.70; HunyuanVideo-1.5 reports nothing. Image
side: sd-vae-ft-mse **24.5 dB** COCO / **27.3 dB** LAION at 256px; SD3 f=8 goes **25.12 (4ch) → 26.40 (8ch)
→ 28.62 (16ch)** ([2403.03206 Table 3](https://arxiv.org/html/2403.03206v1), eval set not stated — VERIFY).

**Cross-table comparison is invalid** — five eval sets, 256px to 1080p, 1 to 121 frames. The one
within-table image-vs-video read (FLUX-VAE 32.70 vs HunyuanVideo 33.14, ImageNet) is **0.4 dB**, not the
4 dB a cross-paper read suggests.

## Decoder fine-tuning on a target domain

**No published dB result exists for tuning a Wan or Cosmos decoder to raise per-frame reconstruction.**
GameNGen §3.2.2 tunes **only the decoder** of the SD 1.4 autoencoder with **MSE alone**, batch 2,048, to fix
"small details and particularly the bottom bar HUD", separately from the U-Net — and reports the gain **only
as a figure** (App. A.2, HUD digits), no dB ([2408.14837](https://arxiv.org/html/2408.14837v2)). It leaves
LPIPS to future work, which is what we did: our 23.7 → 29.1 dB and HUD 17.9 → 34.7 with MSE + 0.1 LPIPS
already exceeds the published precedent.
sd-vae-ft-mse is the only quantified decoder-only precedent: MSE + 0.1 LPIPS, batch 192 on 16 A100s,
**+1.1 dB COCO / +1.3 dB LAION** on broad data. Every Wan/LTX/Cosmos decoder retrain in the literature
optimizes **latency** and *loses* dB: Flash-VAED 40.40 → 37.61 on Wan2.1; Matrix-Game 3.0's MG-LightVAE
33.79 → 31.84. NVIDIA's Cosmos tokenizer recipe tunes the whole tokenizer on 8x H100/80GB, no gain reported.
One community precedent: `spacepxl/Wan2.1-VAE-upscale2x`, **decoder-only, encoder frozen to preserve the
latent space**, L1 + LPIPS + FDL + patchGAN, ~40 h on one RTX 5090, 300k steps at batch 4, no metrics
published by choice. Its warning matters here: a decoder trained on *encoded* latents is brittle on
*generated* latents, so train on degraded ones. DiffSynth maintainers state they have **no VAE trainer** and
could not train away Wan2.1's four-frame jitter
([issue #650](https://github.com/modelscope/DiffSynth-Studio/issues/650)).

## What would change the decision

1. **A measured 2B training footprint.** The 33 GB figure for Cosmos-Predict2.5-2B is arithmetic from our
   own 24.65 GB at 1.436B; NVIDIA publishes nothing. A one-hour fit check at 256x320 settles it, and it
   decides one card versus two.
2. **The Wan-VAE ceiling.** Every Wan-latent candidate inherits the 24.83 dB / 0.122 ceiling we measured,
   3.8 dB below our fine-tuned SD decoder's 28.61. If a decoder-only tune on ~1M Doom frames closes half of
   that, the Wan family becomes competitive on per-frame metrics; if not, per-frame quality caps the video
   row whichever transformer we pick.
3. **Whether we need a pretrained action head at all.** Only HY-WorldPlay ships one with a trainer, at
   60 GB/GPU and under a licence banning use of its outputs to train other models. Matrix-Game 2.0 and Oasis
   both inject actions the way we already do, so the head is cheap to write. If that holds, the choice
   collapses to which checkpoint has the most video pretraining in a latent space we can afford, and
   Cosmos-Predict2.5-2B wins on exposure, licence, single-GPU recipe and resolution.
