# DoomDiT

A controlled comparison of a Diffusion Transformer (DiT-XL/2) against the Stable Diffusion 1.4 U-Net as the backbone of a GameNGen-style Doom world model, with data, latents, conditioning, objective, and compute held identical between the two. Companion release: a lossless, per-tic, 320x240 ViZDoom deathmatch dataset with pose and HUD state, recorded with the Arnold agent on 17 maps.

Start with `RESEARCH_CONTEXT.md` (where the project is), `REQUIREMENTS.md` (what the paper needs), and `TECHNICAL_PLAN.md` (every design decision and why).

## Pipeline

| stage | script | notes |
|---|---|---|
| record | `record_arnold.py` (Arnold agent, 17 maps), `record_episodes.py` (gameNgen-repro agent, one arena) | every engine tic, PNG bytes in per-episode parquet, action, buttons, health, ammo, kills, deaths, frags, x, y, angle |
| data card | `dataset_stats.py` | counts per map, action histogram, coverage heatmaps |
| encode | `encode_parquet.py` | one frame per agent decision (stride 4), pad 320x240 to 320x256, `sd-vae-ft-mse` encoder, latents (4, 32, 40) fp16 |
| split | `make_split.py` | episode-level, 10% held out per map, maps 16 and 17 held out entirely |
| decoder | `finetune_decoder.py` | GameNGen decoder fine-tune, MSE on lossless frames, HUD-crop metrics before and after |
| train | `train_wm.py --backbone dit|unet` | one recipe: L = 32 stacked context, velocity target, context noise augmentation, AdamW, bf16 autocast, fp32 EMA, held-out v-loss selection; `--fit-check N` measures steps/s and memory on synthetic data |
| evaluate | `eval_tf.py` | teacher-forced PSNR, LPIPS, HUD PSNR, copy-last baseline, VAE ceiling against lossless frames |
| rollouts | `rollout_eval.py --rollout`, `--score` | autoregressive rollouts, drift curves, IDM action agreement, clips for FVD |
| IDM | `train_idm.py` | inverse dynamics model on real latent pairs with its accuracy ceiling |
| figures | `plot_results.py` | drift, IDM-vs-horizon, loss curves |
| release | `release/hf_upload.py`, `release/DATASET_CARD.md` | parquet shards with an image column, Hugging Face upload |

Model code: `backbones.py` (both backbones behind one interface, warm-start inflation), `diffusion_v.py` (velocity DDPM, DDIM, noise augmentation), `models.py` (DiT, from fast-DiT), `doom_data.py` (datasets and splits), `doomdit_utils.py` (checkpoint and VAE loading).

Legacy April 2026 code, kept for the continuity numbers: `trainDoom.py`, `build_dataset.py`, `eval_checkpoint.py`, `eval_metrics.py`, `rollout_video.py`, `encode_episodes.py`; weights on the GitHub release `002-DiT-XL-2-best-90k` (`bash download_weights.sh`).

## Servers

Data is generated, stored, and encoded on Spiderman (`/sata2/data/rnagabhi/doom/`); training and evaluation run on Superman (`/home/rohan/Doom/`). See `CLAUDE.md` and the `/run-server` skill. Launch multi-GPU training with `torchrun --nproc_per_node N train_wm.py ...`.

## Credits

Arnold agent: Lample and Chaplot, "Playing FPS Games with Deep Reinforcement Learning" (AAAI 2017), `github.com/glample/Arnold`, ported to torch 2 with `third_party/arnold-torch2.patch`. Single-arena agent and scenario: `github.com/arnaudstiegler/gameNgen-repro` (Apache-2.0). ViZDoom: Kempka et al., 2016. DiT: Peebles and Xie, 2023 (code from `chuanyangjin/fast-DiT`). GameNGen: Valevski et al., 2024. MultiGen: Po et al., 2026.
