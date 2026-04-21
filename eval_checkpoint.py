"""
Evaluate a trained DoomDiT checkpoint by DDIM-sampling the fixed eval segments
and decoding through the CORRECT VAE (sd-vae-ft-mse per Keerthana's pipeline).

Usage:
    python eval_checkpoint.py --ckpt results/012-DiT-XL-2/checkpoints/best.pt --out-dir eval/best
    python eval_checkpoint.py --ckpt results/012-DiT-XL-2/checkpoints/0020000.pt --out-dir eval/step20k
"""
import argparse
import os

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

from models import DiT_models
from diffusion import create_diffusion


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading checkpoint {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # Prefer live model over EMA when both present. The bf16-EMA bug from earlier
    # taught us that EMA can silently freeze; model weights are ground truth.
    if args.use_ema and "ema" in ckpt:
        state = ckpt["ema"]
        print(f"  using EMA weights")
    elif "model" in ckpt:
        state = ckpt["model"]
        print(f"  using live model weights")
    else:
        state = ckpt
        print(f"  using raw state_dict")
    step = ckpt.get("step", "?")
    loss = ckpt.get("loss", "?")
    print(f"  step={step}  loss={loss}")

    # Build model matching training config
    model = DiT_models["DiT-XL/2"](
        input_size=(16, 20),
        in_channels=20,
        pred_channels=4,
        num_classes=18,
    )
    ema_dtype = next(iter(ema_state.values())).dtype
    print(f"  EMA dtype: {ema_dtype}")
    model.load_state_dict(ema_state, strict=False)
    model = model.to(device).to(ema_dtype).eval()

    # Load the CORRECT VAE (sd-vae-ft-mse, matching Keerthana's encoding pipeline).
    print("Loading VAE stabilityai/sd-vae-ft-mse...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("cpu").eval()

    # Build eval segments matching what trainDoom does: 10 evenly-spaced starting indices,
    # 8 consecutive frames each. Load via the consolidated data file.
    print(f"Loading data from {args.feature_path}/")
    context_np = np.load(f"{args.feature_path}/context_latents.npy", mmap_mode="r")
    target_np = np.load(f"{args.feature_path}/target_latents.npy", mmap_mode="r")
    actions_np = np.load(f"{args.feature_path}/context_actions.npy")

    N = args.num_eval_segments
    S = args.eval_segment_size
    total = context_np.shape[0]
    starts = np.linspace(0, max(total - S, 0), N, dtype=int)

    import torch.nn.functional as F
    ctxs, tgts, acts = [], [], []
    for start in starts:
        for i in range(S):
            idx = int(start) + i
            c = torch.from_numpy(context_np[idx].copy()).float()       # (4, 4, 15, 20)
            t = torch.from_numpy(target_np[idx].copy()).float()        # (4, 15, 20)
            a = torch.tensor(actions_np[idx][-1], dtype=torch.long)
            c = F.pad(c, (0, 0, 0, 1))
            t = F.pad(t, (0, 0, 0, 1))
            c = c.reshape(-1, 16, 20)
            ctxs.append(c); tgts.append(t); acts.append(a)

    eval_ctx = torch.stack(ctxs).to(device).to(ema_dtype)
    eval_tgt = torch.stack(tgts).to(device)
    eval_act = torch.stack(acts).to(device)
    print(f"  eval batch: {eval_ctx.shape[0]} samples ({N} segments x {S} frames)")

    # Save ground truth decoded with the CORRECT VAE.
    gt_dir = os.path.join(args.out_dir, "ground_truth")
    os.makedirs(gt_dir, exist_ok=True)
    with torch.no_grad():
        gt_latent = (eval_tgt[:, :, :15, :].float() / 0.18215).cpu()
        gt_img = vae.decode(gt_latent).sample
    gt_img = (gt_img * 0.5 + 0.5).clamp(0, 1)
    for seg_idx in range(N):
        save_image(gt_img[seg_idx * S:(seg_idx + 1) * S],
                   f"{gt_dir}/segment_{seg_idx:02d}_start{starts[seg_idx]}.png", nrow=S)
    print(f"  wrote {N} GT segments to {gt_dir}/")

    # DDIM sampling with more steps than training used (250 vs training's 50).
    # More steps often helps when training sampling looked noisy.
    print(f"Sampling with {args.num_sample_steps} DDIM steps...")
    sample_diffusion = create_diffusion(timestep_respacing=str(args.num_sample_steps))

    shape = (eval_ctx.shape[0], 4, 16, 20)
    noise = torch.randn(shape, device=device, dtype=ema_dtype)
    model_kwargs = dict(context=eval_ctx, action=eval_act)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=ema_dtype, enabled=ema_dtype != torch.float32):
            samples = sample_diffusion.p_sample_loop(
                model, shape, noise, clip_denoised=False,
                model_kwargs=model_kwargs, progress=True, device=device,
            )
        print(f"  sampled latent stats: mean={samples.float().mean():.3f} "
              f"std={samples.float().std():.3f} "
              f"(target stats: mean={eval_tgt.float().mean():.3f} std={eval_tgt.float().std():.3f})")
        samples = (samples[:, :, :15, :].float() / 0.18215).cpu()
        imgs = vae.decode(samples).sample

    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
    for seg_idx in range(N):
        save_image(imgs[seg_idx * S:(seg_idx + 1) * S],
                   f"{args.out_dir}/segment_{seg_idx:02d}_start{starts[seg_idx]}.png", nrow=S)
    print(f"  wrote {N} sample segments to {args.out_dir}/")
    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--feature-path", type=str, default="data")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num-eval-segments", type=int, default=10)
    parser.add_argument("--eval-segment-size", type=int, default=8)
    parser.add_argument("--num-sample-steps", type=int, default=250)
    args = parser.parse_args()
    main(args)
