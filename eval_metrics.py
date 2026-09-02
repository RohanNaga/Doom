"""
Teacher-forced next-frame metrics (PSNR, LPIPS, latent MSE) for a DoomDiT checkpoint.

Rebuilds the harness behind the 26.04 dB / 0.153 LPIPS headline. For every
evaluation window the model gets the four real context frames and the action,
samples one latent, and the decoded prediction is scored against the decoded
ground-truth latent (and against the raw 160x120 frame when
`ep_XXXX_frames.npy` exists for that episode). A copy-last-frame baseline and
the VAE reconstruction ceiling are reported alongside, because a next-frame
model that does not beat persistence has learned nothing about dynamics.

Subsets:
  val / train   windows drawn from the episodes in --split (honest evaluation)
  legacy        the 10 x 8 consecutive windows trainDoom.py sampled during
                training (training data; only for reproducing the old headline)

Usage:
    python eval_metrics.py --ckpt results/002-DiT-XL-2/checkpoints/best.pt \
        --episodes-dir data/episodes --split data/split.json --subset val \
        --num-windows 512 --out-dir eval/best_val
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from diffusion import create_diffusion
from doom_data import EpisodeWindowDataset, legacy_segment_starts, load_split, sample_eval_windows
from doomdit_utils import decode_latents, load_doomdit, load_vae


def psnr(a, b):
    """Per-image PSNR in dB for tensors in [0, 1], shape (B, 3, H, W)."""
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


class Scorer:
    def __init__(self, lpips_net, device):
        import lpips
        self.lpips = lpips.LPIPS(net=lpips_net, verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, pred, ref):
        pred, ref = pred.to(self.device), ref.to(self.device)
        return psnr(pred, ref).cpu(), self.lpips(pred * 2 - 1, ref * 2 - 1).flatten().cpu()


def sample_batch(model, diffusion, ctx, act, sampler, dtype, device, eta=0.0):
    shape = (ctx.shape[0], 4, 16, 20)
    noise = torch.randn(shape, device=device, dtype=dtype)
    kwargs = dict(context=ctx.to(dtype), action=act)
    # autocast on the actual device: the timestep embedding is built in fp32 inside the
    # model, so bf16 weights need autocast on CPU as well as CUDA.
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype, enabled=(dtype != torch.float32)):
        if sampler == "ddim":
            return diffusion.ddim_sample_loop(model, shape, noise, clip_denoised=False,
                                              model_kwargs=kwargs, device=device, eta=eta)
        return diffusion.p_sample_loop(model, shape, noise, clip_denoised=False,
                                       model_kwargs=kwargs, device=device)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    model, info = load_doomdit(args.ckpt, use_ema=args.use_ema, device=device, model_name=args.model)
    dtype = info["dtype"]
    print(f"loaded {args.ckpt}: {info}")
    vae = load_vae(args.vae_device)
    scorer = Scorer(args.lpips_net, device)
    diffusion = create_diffusion(timestep_respacing=str(args.num_sample_steps))

    # --- evaluation windows ---
    if args.subset == "legacy":
        ds = EpisodeWindowDataset(args.episodes_dir)
        starts = legacy_segment_starts(len(ds), args.legacy_segments, args.legacy_segment_size)
        indices = np.concatenate([np.arange(s, s + args.legacy_segment_size) for s in starts])
    else:
        split = load_split(args.split)
        ds = EpisodeWindowDataset(args.episodes_dir, split[args.subset])
        indices = sample_eval_windows(ds, args.num_windows, seed=args.seed, stride=args.stride)
    print(f"subset={args.subset}: {len(ds.episodes)} episodes, {len(ds):,} windows, evaluating {len(indices)}")
    loader = DataLoader(Subset(ds, indices.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=2)

    # raw frames, if the encoder saved them for these episodes
    frames = {}
    for ep_id, _, _ in ds.episodes:
        fp = os.path.join(args.episodes_dir, f"ep_{ep_id:04d}_frames.npy")
        if os.path.isfile(fp):
            frames[ep_id] = np.load(fp, mmap_mode="r")

    rows, sample_time, n_frames = [], 0.0, 0
    images_saved = 0
    for b, (ctx, tgt, act) in enumerate(loader):
        ctx, tgt, act = ctx.to(device), tgt.to(device), act.to(device)
        t0 = time.time()
        pred = sample_batch(model, diffusion, ctx, act, args.sampler, dtype, device, args.eta)
        if device == "cuda":
            torch.cuda.synchronize()
        sample_time += time.time() - t0
        n_frames += ctx.shape[0]

        pred_img = decode_latents(vae, pred)
        gt_img = decode_latents(vae, tgt)
        last_img = decode_latents(vae, ctx[:, -4:])           # persistence baseline: last context frame
        p_psnr, p_lpips = scorer(pred_img, gt_img)
        c_psnr, c_lpips = scorer(last_img, gt_img)
        lat_mse = ((pred[:, :, :15].float() - tgt[:, :, :15].float()) ** 2).flatten(1).mean(1).cpu()

        batch_idx = indices[b * args.batch_size:(b + 1) * args.batch_size]
        raw_refs = []
        for gi in batch_idx:
            ep_id, start = ds.locate(int(gi))
            raw = frames.get(ep_id)
            raw_refs.append(None if raw is None else torch.from_numpy(np.asarray(raw[start + 4])).permute(2, 0, 1).float() / 255)
        if all(r is not None for r in raw_refs):
            raw_img = torch.stack(raw_refs)
            r_psnr, r_lpips = scorer(pred_img, raw_img)
            v_psnr, v_lpips = scorer(gt_img, raw_img)     # VAE reconstruction ceiling
        else:
            r_psnr = r_lpips = v_psnr = v_lpips = torch.full((ctx.shape[0],), float("nan"))

        for i, gi in enumerate(batch_idx):
            ep_id, start = ds.locate(int(gi))
            rows.append(dict(index=int(gi), episode=ep_id, start=start, action=int(act[i]),
                             psnr=float(p_psnr[i]), lpips=float(p_lpips[i]), latent_mse=float(lat_mse[i]),
                             copy_psnr=float(c_psnr[i]), copy_lpips=float(c_lpips[i]),
                             raw_psnr=float(r_psnr[i]), raw_lpips=float(r_lpips[i]),
                             vae_psnr=float(v_psnr[i]), vae_lpips=float(v_lpips[i])))
        if images_saved < args.save_images:
            grid = torch.cat([last_img.cpu(), gt_img.cpu(), pred_img.cpu()], dim=0)
            save_image(grid, os.path.join(args.out_dir, f"batch_{b:03d}_last_gt_pred.png"), nrow=ctx.shape[0])
            images_saved += 1
        if (b + 1) % 10 == 0:
            print(f"  {len(rows)}/{len(indices)}  psnr={np.mean([r['psnr'] for r in rows]):.2f}", flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- aggregate ---
    def agg(key):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            return None
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "sem": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0, "n": int(len(v))}

    summary = {k: agg(k) for k in ["psnr", "lpips", "latent_mse", "copy_psnr", "copy_lpips",
                                   "raw_psnr", "raw_lpips", "vae_psnr", "vae_lpips"]}
    summary["sampling_frames_per_s"] = n_frames / max(sample_time, 1e-9)
    summary["config"] = {**vars(args), "weights": info["weights"], "dtype": str(dtype),
                         "step": info["step"], "num_windows_evaluated": len(rows)}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(args.out_dir, "per_window.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=1))
    print("DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="DiT-XL/2")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--episodes-dir", default="data/episodes")
    p.add_argument("--split", default="data/split.json")
    p.add_argument("--subset", choices=["val", "train", "legacy"], default="val")
    p.add_argument("--num-windows", type=int, default=512)
    p.add_argument("--stride", type=int, default=8, help="thin random draws so windows are spread out")
    p.add_argument("--legacy-segments", type=int, default=10)
    p.add_argument("--legacy-segment-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm",
                   help="ddpm = respaced ancestral sampling, what trainDoom.py's eval used")
    p.add_argument("--num-sample-steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lpips-net", choices=["alex", "vgg"], default="alex")
    p.add_argument("--vae-device", default="cuda")
    p.add_argument("--save-images", type=int, default=4)
    p.add_argument("--out-dir", required=True)
    main(p.parse_args())
