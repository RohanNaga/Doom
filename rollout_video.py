"""
Autoregressive video rollout from a DoomDiT checkpoint.

Starting from a real 4-frame context from the dataset (or user-provided), sample
the next frame via DDIM, slide the window forward, and repeat for --num-frames
steps. Decode all predicted frames via SD-VAE-ft-mse and save as GIF + PNG grid.

Usage:
    python rollout_video.py \\
        --ckpt results/002-DiT-XL-2/checkpoints/best.pt \\
        --num-frames 32 \\
        --seed-index 0 \\
        --out-dir rollout

Outputs in out-dir:
    rollout.gif       animated sequence
    frames.png        grid of all predicted frames
    context.png       grid of the 4 seed context frames (for comparison)
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from PIL import Image

from models import DiT_models
from diffusion import create_diffusion


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load checkpoint ---
    print(f"Loading checkpoint {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["ema"] if args.use_ema and "ema" in ckpt else ckpt.get("model", ckpt.get("ema", ckpt))
    step = ckpt.get("step", "?")
    loss = ckpt.get("loss", "?")
    print(f"  step={step}  loss={loss}  weights={'ema' if args.use_ema else 'live model'}")

    model = DiT_models["DiT-XL/2"](
        input_size=(16, 20),
        in_channels=20,
        pred_channels=4,
        num_classes=18,
    )
    dtype = next(iter(state.values())).dtype
    model.load_state_dict(state, strict=True)
    model = model.to(device).to(dtype).eval()

    print("Loading VAE stabilityai/sd-vae-ft-mse...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("cpu").eval()

    # --- Seed context ---
    # Load 4 real context frames + associated actions to start the rollout.
    # Either pick from the consolidated dataset, or from a user-provided npy.
    if args.seed_npy:
        # Expect shape (4+N, 4, 15, 20) — 4 seed frames then N real subsequent frames
        seed_latents = np.load(args.seed_npy)
        seed_actions = np.load(args.seed_actions_npy) if args.seed_actions_npy else np.zeros((seed_latents.shape[0],), dtype=np.int64)
        context = torch.from_numpy(seed_latents[:4].copy()).float()
        action_sequence = torch.from_numpy(seed_actions[3:3 + args.num_frames].copy()).long()
    else:
        # Pull from the consolidated training data for quick dogfood.
        ctx_all = np.load(f"{args.feature_path}/context_latents.npy", mmap_mode="r")
        act_all = np.load(f"{args.feature_path}/context_actions.npy")
        seed_idx = args.seed_index
        # context_latents at seed_idx has 4 past frames for that sample
        context = torch.from_numpy(ctx_all[seed_idx].copy()).float()  # (4, 4, 15, 20)
        # Action sequence: we roll forward N frames, so we need N target actions.
        # context_actions[i] = [a_{i}, a_{i+1}, a_{i+2}, a_{i+3}, a_{i+4}] — last element is target action.
        # For autoregressive rollout, target action for frame i is act_all[seed_idx + i, -1].
        action_sequence = torch.from_numpy(
            act_all[seed_idx:seed_idx + args.num_frames, -1].copy()
        ).long()

    assert context.shape == (4, 4, 15, 20), f"unexpected seed context shape {context.shape}"
    context = F.pad(context, (0, 0, 0, 1))  # (4, 4, 16, 20)

    # --- Save seed context as PNG for reference ---
    with torch.no_grad():
        seed_img = vae.decode((context[:, :, :15, :].float() / 0.18215).cpu()).sample
    seed_img = (seed_img * 0.5 + 0.5).clamp(0, 1)
    save_image(seed_img, f"{args.out_dir}/context.png", nrow=4)
    print(f"  wrote seed context to {args.out_dir}/context.png")

    # --- Autoregressive rollout ---
    sample_diffusion = create_diffusion(timestep_respacing=str(args.num_sample_steps))
    predicted_latents = []
    context = context.to(device).to(dtype)  # (4, 4, 16, 20)

    print(f"Rolling out {args.num_frames} frames (DDIM {args.num_sample_steps} steps each)...")
    for step_i in range(args.num_frames):
        # Reshape (4 past frames, 4 channels, 16, 20) -> (16, 16, 20) for the model
        ctx_flat = context.reshape(-1, 16, 20).unsqueeze(0)  # (1, 16, 16, 20)
        act = action_sequence[step_i:step_i + 1].to(device)
        noise = torch.randn(1, 4, 16, 20, device=device, dtype=dtype)
        model_kwargs = dict(context=ctx_flat, action=act)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                sampled = sample_diffusion.p_sample_loop(
                    model, noise.shape, noise, clip_denoised=False,
                    model_kwargs=model_kwargs, progress=False, device=device,
                )
        # sampled: (1, 4, 16, 20) — the newly predicted target frame
        predicted = sampled[0].detach()  # (4, 16, 20)
        predicted_latents.append(predicted.cpu())

        # Slide window: drop frame 0, shift 1->0, 2->1, 3->2, append prediction at 3
        context = torch.cat([context[1:], predicted.unsqueeze(0)], dim=0)
        if (step_i + 1) % 4 == 0 or step_i == args.num_frames - 1:
            print(f"  frame {step_i + 1}/{args.num_frames}")

    # --- Decode all predicted frames via VAE ---
    pred_stack = torch.stack(predicted_latents, dim=0)  # (N, 4, 16, 20)
    print(f"Decoding {pred_stack.shape[0]} frames through VAE...")
    with torch.no_grad():
        imgs = vae.decode((pred_stack[:, :, :15, :].float() / 0.18215)).sample
    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)  # (N, 3, 120, 160)

    # PNG grid
    save_image(imgs, f"{args.out_dir}/frames.png", nrow=8)
    print(f"  wrote {args.out_dir}/frames.png")

    # GIF animation
    pil_frames = []
    for i in range(imgs.shape[0]):
        arr = (imgs[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_frames.append(Image.fromarray(arr))
    pil_frames[0].save(
        f"{args.out_dir}/rollout.gif",
        save_all=True, append_images=pil_frames[1:],
        duration=int(1000 / args.fps), loop=0,
    )
    print(f"  wrote {args.out_dir}/rollout.gif ({args.fps} fps)")
    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="rollout")
    parser.add_argument("--num-frames", type=int, default=32,
                        help="How many frames to generate autoregressively.")
    parser.add_argument("--feature-path", type=str, default="data",
                        help="Path with context_latents.npy + context_actions.npy for seeding.")
    parser.add_argument("--seed-index", type=int, default=0,
                        help="Index into context_latents.npy to seed the rollout.")
    parser.add_argument("--seed-npy", type=str, default="",
                        help="Alternative: path to a (4+N, 4, 15, 20) npy for custom seed context.")
    parser.add_argument("--seed-actions-npy", type=str, default="",
                        help="Optional actions npy matching --seed-npy.")
    parser.add_argument("--num-sample-steps", type=int, default=50)
    parser.add_argument("--use-ema", action="store_true",
                        help="Use EMA weights instead of live model.")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    main(args)
