---
license: other
license_name: mixed-see-card
license_link: "#licensing"
pipeline_tag: image-to-image
library_name: diffusers
tags:
- doom
- world-model
- action-conditioned
- diffusion
- dit
- video-prediction
datasets:
- <hf-user>/doom-freedoom-arnold-320x240-lossless
base_model:
- facebook/DiT-XL-2-256
- CompVis/stable-diffusion-v1-4
- PixArt-alpha/PixArt-XL-2-512x512
- thu-ml/unidiffuser-v1
- Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers
- stabilityai/sd-vae-ft-mse
---

# DoomDiT warm-start checkpoints: action-conditioned latent diffusion world models for Doom

Five pretrained diffusion checkpoints adapted to the same task under one recipe: predict the next Doom decision frame (latent) from 32 past frames and one discrete action, on the [DoomDiT benchmark corpus](https://huggingface.co/datasets/<hf-user>/doom-freedoom-arnold-320x240-lossless). Also here: the fine-tuned VAE decoder every reported number is decoded through, and the two inverse-dynamics judges that read action following off generated rollouts.

Facts are tied to the code in the [DoomDiT repository](https://github.com/RohanNaga/Doom) (`train_wm.py`, `backbones.py`, `diffusion_v.py`, `finetune_decoder.py`, `train_idm.py`, `eval_tf.py`, `rollout_eval.py`, `scripts/spiderman/*.sh`) and to its `RESEARCH_CONTEXT.md` log. Numbers below are the values logged there on 2026-09-16 and are placeholders until `paper/make_tables.py` regenerates them from the result directories; every such cell is marked `<!-- VERIFY -->`.

## Contents

| Folder | What | Warm start | Params | Status |
|---|---|---|---|---|
| `dit_xl2_seed0/` | DiT-XL/2, seed 0 (run `030-dit-l32-aligned`) | `DiT-XL-2-256x256.pt`, ImageNet-1k class-conditional (Peebles and Xie 2023) | 675M | done, 90k updates |
| `dit_xl2_seed1/` | DiT-XL/2, seed 1 (run `032-dit-l32-aligned-seed1`) | same | 675M | done, 90k updates |
| `unet_sd14/` | Stable Diffusion 1.4 U-Net (run `031-unet-l32-aligned`) | `CompVis/stable-diffusion-v1-4`, `unet` subfolder (LAION) | 860M | done, 90k updates |
| `pixart_alpha512/` | PixArt-alpha 512 transformer (run `033-pixart-l32-aligned`) | `PixArt-alpha/PixArt-XL-2-512x512` | 611.6M (753,664 new) | trained; evaluation running 2026-09-16 <!-- VERIFY --> |
| `unidiffuser_v1/` | UniDiffuser v1 U-ViT (run `040-unidiffuser-l32-aligned`) | `thu-ml/unidiffuser-v1`, `unet` subfolder (LAION-2B) | 952M <!-- VERIFY --> | placeholder: trains 2026-09-17 to 19 |
| `skyreels_v2_df13b/` | SkyReels-V2 DF 1.3B, video-pretrained, own autoencoder (run `050-skyreels-l8-flow`) | `Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers` | 1,435,571,776 (46,080 new) | placeholder: exploratory row, due 2026-09-17 |
| `vae_decoder_lpips/` | `sd-vae-ft-mse` with the decoder fine-tuned on Doom (MSE + 0.1 LPIPS) | `stabilityai/sd-vae-ft-mse` | 83.7M (decoder 49.5M) <!-- VERIFY --> | done |
| `idm_k8/` | Inverse-dynamics judge, window K=8 (`idm_aligned`) | none | 11.4M | done |
| `idm_k2/` | Inverse-dynamics judge, window K=2 (`idm_aligned_k2`) | none | <!-- VERIFY --> | done |
| `legacy_april_dit_xl2/` | April 2026 class-project DiT-XL/2 (GitHub release `002-DiT-XL-2-best-90k`) | ImageNet DiT-XL/2 | 673M | optional; no honest held-out number (trained on all 500 JPEG episodes at 160x120) |

Each run folder holds `best.pt` (bf16 live weights at the best held-out velocity loss, keys `model`, `step`, `val_loss`, `args`), `snap_0090000.pt` where it exists (bf16 live weights plus `ema` at the final step, the source of the EMA rows), `config.json` (the full argument list and git hash), `log.jsonl` (loss, validation, gradient-norm events), and `eval/` (the `metrics.json` and rollout summaries the paper reads). Recovery checkpoints (fp32 weights, optimizer, RNG) are not released.

## Task and input format

One prediction per agent decision (every 4 engine tics, matching GameNGen and Arnold). Frames are padded from 320x240 to 320x256 at the image level and encoded once with the frozen `sd-vae-ft-mse` encoder (posterior mean, scale 0.18215) to `4 x 32 x 40` latents in fp16 (`encode_parquet.py --align-decisions`). The model input is the 32 context latents stacked on channels with the noisy target: 132 channels; the conditioning is the verified action of the transition into the target (29 ids) and a context-noise bucket (10 ids). Training windows come from verified transitions only and never cross a chain boundary (`--require-verified-transitions`).

## Shared recipe (identical for every SD-latent row)

Read from `train_wm.py` defaults and `scripts/spiderman/launch_aligned_spiderman.sh`, `launch_seed1.sh`, `launch_pixart.sh`.

| | |
|---|---|
| Objective | velocity prediction, MSE on `v`; linear betas 1e-4 to 0.02 over 1,000 steps; uniform `t`; learned-variance heads removed (`diffusion_v.py`) |
| Context noise augmentation | Gaussian noise on the context at a level drawn uniformly in [0, 0.7], discretized into 10 buckets and fed as a conditioning id (GameNGen); clean context at inference |
| Action dropout / CFG | 0; no classifier-free guidance |
| Optimizer | AdamW, lr 5e-5, 2,000 warmup steps, weight decay 0, gradient clip 1.0 |
| Batch and length | global batch 32 (DiT 32 x 1 on one A6000; U-Net 16 x 2 accumulation), 90k updates = 2.88M target presentations |
| Precision | bf16 autocast over fp32 master weights; fp32 EMA 0.9999 on the CPU (diagnostic; live weights are the default report) |
| Selection | held-out velocity loss every 1,000 updates on 1,024 fixed windows (`split_arnold.json` `val`); `best.pt` is the minimum |
| Seeds | 0 (seed 1 for `dit_xl2_seed1`) |
| Sampling | DDIM, 50 steps, eta 0 |
| Hardware | one RTX A6000 48 GB per run on Spiderman; DiT 1.26 updates/s with gradient checkpointing (about 20 h), U-Net 0.85 updates/s (about 30 h) |

Conditioning pathway per warm start (`backbones.py`): the input patch/conv projection is inflated with the pretrained kernel on the four target channels and zeros on the 128 context channels, so step 0 equals the pretrained model applied to the noisy target; positional tables are regenerated for the 16x20 token grid.

- DiT-XL/2: action and noise bucket as adaLN-Zero embedding tables added to the timestep embedding; class table skipped; epsilon half of the 8-channel head kept and retrained as velocity.
- SD 1.4 U-Net: action as one cross-attention token; bucket through the class embedding (GameNGen's construction).
- PixArt-alpha 512: action and bucket as two learned 4096-wide tokens through the pretrained caption projection; timestep on PixArt's adaLN-single; `use_additional_conditions=False`; output `sample[:, :4]`.
- UniDiffuser v1: <!-- VERIFY conditioning design once the row is built (two learned text tokens per backbones.py docstring) -->
- SkyReels-V2 DF 1.3B (video row, its own recipe): Wan-VAE latents (16 channels, 4x8x8), 8 decision-frame context, flow-matching objective, batch 8, lr 2e-5, warmup 500, 10k updates (80k presentations, 36x fewer than the other rows), latest action only on the target frame's tokens. Exploratory; not compute- or exposure-matched.

## Fine-tuned VAE decoder

`finetune_decoder.py`: encoder frozen (so every row's latents are unchanged), decoder trained on 50,000 training-split decision frames for 2 epochs at lr 1e-5, batch 16, loss MSE + 0.1 x LPIPS, target the lossless 320x240 frame <!-- VERIFY the launch flags of vae_decoder_arnold_lpips -->. Held-out frames (2,000, training maps): PSNR 23.69 to 28.34 dB, HUD rows (bottom 32) 17.93 to 32.09 dB, LPIPS 0.095 to 0.051, HUD LPIPS 0.060 to 0.0017. On the evaluation corpora the VAE ceiling (true latent decoded) is 28.61 dB / 0.051 on seen maps and 26.36 dB / 0.070 on unseen maps 16 and 17: the decoder was fit on seen-map frames and alone loses 2.25 dB on unseen maps, so unseen numbers should be read relative to that ceiling. Loads as `diffusers.AutoencoderKL.from_pretrained('vae_decoder_lpips')`; pass the folder as `--vae-path` to the evaluators.

## IDM judges

`train_idm.py`, in the structure of VPT's inverse dynamics model as used by MineWorld and Matrix-Game: a shared conv encoder maps each `4 x 32 x 40` latent to a token, a 4-layer transformer (width 128) mixes K tokens bidirectionally, and a head on each adjacent pair emits the K-1 action logits. Trained on real verified windows of the training split, 6,000 steps, batch 64, lr 1e-3 <!-- VERIFY steps used for idm_aligned -->.

| Judge | Top-1 (29 actions) | Movement collapse | Macro recall | Majority baseline | Real reference on the evaluation rollouts |
|---|---|---|---|---|---|
| K=8 (`idm_k8`) | 82.9% | 84.9% | 54.8% | 23.2% | 0.847 |
| K=2 (`idm_k2`) | 70.1% | 72.7% | 40.6% | 22.7% | n/a |

K=2 is the control: most of the action is readable from a single transition, and the 8-frame window adds about 13 points, so the judge is not a long-context artefact. Both judges saw only seen-map latents. The checkpoint stores `model`, `num_actions`, `width`, `window`, `depth`; `metrics.json` the numbers above.

## Metrics (as logged 2026-09-16; placeholders until regenerated) <!-- VERIFY every cell against paper/make_tables.py output -->

Teacher-forced: 32 real context latents and the action in, one DDIM-50 sample out, scored against the raw lossless frame on 2,048 windows per corpus, live weights unless marked EMA. References: copy-last (last context frame) 19.41 dB seen / 18.50 unseen; VAE ceiling 28.61 / 0.051 seen, 26.36 / 0.070 unseen.

| Row | Held-out v-loss | Seen PSNR / LPIPS | Seen EMA | Unseen (16, 17) PSNR / LPIPS |
|---|---|---|---|---|
| DiT-XL/2 seed 0 | 0.2139 | 21.06 / 0.311 | 21.06 / 0.306 | 19.52 / 0.450 |
| DiT-XL/2 seed 1 | 0.2139 | 21.21 / 0.307 | 21.10 / 0.304 | 19.59 / 0.442 |
| SD 1.4 U-Net | 0.2043 | 21.36 / 0.270 | 21.67 / 0.250 | 19.14 / 0.446 |
| PixArt-alpha 512 | n/a | n/a | n/a | n/a |
| UniDiffuser v1 | n/a | n/a | n/a | n/a |
| SkyReels-V2 DF 1.3B | n/a | n/a | n/a | n/a |
| Unseen2 (13 curated maps) | | | | n/a for every row <!-- VERIFY: scored after the corpus lands --> |

Autoregressive: 32 real seed frames and the recorded actions, 64 decision frames fed back, 256 rollouts on the seen corpus, identical trajectories for every row. Copy-seed at horizon 64: 17.86 dB / 0.510 LPIPS.

| Row | PSNR @8 / 16 / 32 / 64 | LPIPS @8 / @64 | IDM top-1 (real 0.847) | FVD16 / FVD32 |
|---|---|---|---|---|
| DiT-XL/2 seed 0 | 19.66 / 18.93 / 18.51 / 18.30 | 0.453 / 0.553 | 0.492 | 232 / 481 |
| DiT-XL/2 seed 1 | 19.86 / 19.09 / 18.65 / 17.68 | 0.436 / 0.550 | 0.476 | 198 / 407 |
| SD 1.4 U-Net | 19.55 / 18.66 / 17.73 / 16.03 | 0.421 / 0.566 | 0.509 | 184 / 356 |
| PixArt-alpha 512 | n/a | n/a | n/a | n/a |

Reading, with its caveats: the U-Net leads on held-out loss, seen-map LPIPS and FVD; the DiTs hold PSNR better at long horizons and lead slightly on unseen-map PSNR; every row is worse than the frozen seed frame on LPIPS by horizon 64, so a long-horizon PSNR lead is not evidence of better dynamics. Two DiT seeds differ by 0.62 dB at horizon 64 and 74 FVD32; the U-Net, PixArt and UniDiffuser rows have one seed each.

## Evaluation commands

Exactly what `scripts/spiderman/after_run2.sh` runs after each training run; substitute the downloaded paths. `latents_arnold_eval/{seen,unseen}` are produced from the dataset's `eval/{seen,unseen}` parquet with `encode_parquet.py --align-decisions --canonical meta/canonical_controls.json`.

```bash
COMMON="--backbone dit --ckpt dit_xl2_seed0/best.pt --vae-path vae_decoder_lpips --context-frames 32 --num-actions 29"
# teacher-forced, raw-frame scoring, seen and unseen corpora
for S in seen unseen; do
  python eval_tf.py $COMMON --latents-dir latents_arnold_eval/$S --parquet-dir eval/$S \
      --split splits/split_eval_$S.json --subset val --num-windows 2048 --batch-size 16 --steps 50 --out-dir eval_tf_$S
done
# EMA weights live in the snapshot, not in best.pt
python eval_tf.py --backbone dit --ckpt dit_xl2_seed0/snap_0090000.pt --use-ema --vae-path vae_decoder_lpips \
    --context-frames 32 --num-actions 29 --latents-dir latents_arnold_eval/seen --parquet-dir eval/seen \
    --split splits/split_eval_seen.json --subset val --num-windows 2048 --steps 50 --out-dir eval_tf_seen_ema
# 256 rollouts x 64 frames, then drift, IDM and FVD
python rollout_eval.py --rollout $COMMON --latents-dir latents_arnold_eval/seen --split splits/split_eval_seen.json \
    --subset val --num-rollouts 256 --horizon 64 --batch-size 16 --steps 50 --out rollouts_seen.npz
python rollout_eval.py --score --rollouts rollouts_seen.npz --idm idm_k8/idm.pt --vae-path vae_decoder_lpips \
    --out-dir rollout_metrics_seen --save-clips 256
for F in 16 32; do python fvd.py --clips rollout_metrics_seen/clips_u8.npz --frames $F --i3d i3d_torchscript.pt --out rollout_metrics_seen/fvd$F.json; done
```

Backbone flags: `--backbone unet --sd-path CompVis/stable-diffusion-v1-4`, `--backbone pixart --pixart-path PixArt-alpha/PixArt-XL-2-512x512`, `--backbone unidiffuser --unidiffuser-path thu-ml/unidiffuser-v1`; the diffusers backbones are instantiated from their source repository before the checkpoint is loaded, so those repositories must be reachable or cached (`--hf-cache`). The video row uses `eval_video.py` with the Wan-VAE latents and its own `idm_real` reference; its IDM number is not comparable to the SD rows. The I3D weights for FVD are the standard `i3d_torchscript.pt` used by StyleGAN-V's FVD implementation <!-- VERIFY source URL -->.

Loading a checkpoint in Python:

```python
import torch
from backbones import build_model
ck = torch.load("dit_xl2_seed0/best.pt", map_location="cpu", weights_only=True)
model = build_model("dit", num_actions=29, context_frames=32, noise_buckets=10, grad_ckpt=False,
                    action_dropout=ck["args"]["action_dropout"])   # 0 for every released row
model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
```

## Intended use and limitations

Research on small-budget action-conditioned world models: warm-start choice, evaluation protocol, transfer to unseen maps. Not a playable game engine: sampling is 50 DDIM steps per frame (about 2 frames/s on one A4000), no real-time distillation. Every model was trained on one policy's deathmatch play on 15 maps for 90k updates; the unseen-map numbers pass through a decoder fit on seen-map frames; the U-Net, PixArt and UniDiffuser rows have a single seed; the SkyReels row is exploratory. Outputs are Freedoom-textured Doom frames and nothing else.

## Licensing

- Trained weights: released under the licence of their warm start, since each checkpoint is a fine-tune. DiT-XL/2: CC BY-NC 4.0 (facebook/DiT), which makes `dit_xl2_seed*` non-commercial. SD 1.4 U-Net: CreativeML Open RAIL-M. PixArt-alpha: Open RAIL++-M <!-- VERIFY -->. UniDiffuser: AGPL-3.0 <!-- VERIFY -->. SkyReels-V2: Apache-2.0 <!-- VERIFY -->. `sd-vae-ft-mse`: MIT.
- IDM judges: trained from scratch on our corpus; CC BY 4.0.
- Training data: Freedoom assets (BSD) rendered by ViZDoom (MIT); see the dataset card.

## Citation

```bibtex
@misc{doomdit2026,
  title  = {DoomDiT: a warm-start study of small-budget latent diffusion world models on Doom},   % VERIFY final title
  author = {Nagabhirava, Rohan and Chirumamilla, Keerthana},
  year   = {2026},
  note   = {Weights: https://huggingface.co/<hf-user>/doomdit-warm-starts}
}
```
