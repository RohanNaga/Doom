"""
Teacher-forced next-frame metrics for the new stack (velocity backbones, stride-4 latents).

For each held-out window: L real context latents and the action, one DDIM sample, decode
through the (optionally fine-tuned) VAE, score against the raw lossless frame from the
parquet recording and against the VAE-decoded ground truth. Reports PSNR, LPIPS, latent
MSE, HUD-crop PSNR, the copy-last-frame baseline, and the VAE ceiling.

    python eval_tf.py --ckpt results/010-dit-l32/best.pt --backbone dit --latents-dir data/latents_arnold \
        --parquet-dir raw_arnold --split data/split_arnold.json --subset val --num-windows 2048 --out-dir eval/dit_val
"""
import argparse
import csv
import io
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from backbones import build_model
from diffusion_v import VDiffusion, noise_augment
from doom_data import LatentWindowDataset, load_split
from doomdit_utils import LATENT_SCALE, load_vae

HUD_ROWS = 32


def psnr(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


@torch.no_grad()
def decode(vae, z):
    img = vae.decode(z.float() / LATENT_SCALE).sample[:, :, :240]
    return (img * 0.5 + 0.5).clamp(0, 1)


def load_model(args, device):
    model = build_model(args.backbone, args.num_actions, args.context_frames, args.noise_buckets, grad_ckpt=False,
                        warm_start=None if args.backbone == "dit" else args.sd_path, cache_dir=args.hf_cache)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["ema"] if (args.use_ema and "ema" in ck) else ck["model"]
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    return model.to(device).eval(), ck.get("step", "?")


class RawFrames:
    """Lazy per-episode access to raw frames in the parquet recordings."""

    def __init__(self, parquet_dir):
        self.dir, self.cache = parquet_dir, {}

    def get(self, episode_id, tic):
        import pyarrow.parquet as pq
        if episode_id not in self.cache:
            t = pq.read_table(os.path.join(self.dir, f"ep_{episode_id:05d}.parquet"), columns=["tic", "frame"])
            self.cache = {episode_id: (np.array(t["tic"]), t["frame"])}   # keep one episode resident
        tics, frames = self.cache[episode_id]
        i = int(np.searchsorted(tics, tic))
        return np.asarray(Image.open(io.BytesIO(frames[i].as_py())).convert("RGB"), dtype=np.uint8)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    model, step = load_model(args, device)
    vae = load_vae(device) if not args.vae_path else __import__("diffusers").models.AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    import lpips
    lp = lpips.LPIPS(net=args.lpips_net, verbose=False).to(device).eval()
    diffusion = VDiffusion(device=device)

    split = load_split(args.split)
    ds = LatentWindowDataset(args.latents_dir, split[args.subset], args.context_frames)
    rng = np.random.RandomState(args.seed)
    idx = np.sort(rng.choice(len(ds), size=min(args.num_windows, len(ds)), replace=False))
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=2)
    raw = RawFrames(args.parquet_dir) if args.parquet_dir else None
    print(f"{args.subset}: {len(ds.episodes)} episodes, {len(ds):,} windows, evaluating {len(idx)}, step {step}")

    rows, t_sample, n = [], 0.0, 0
    for b, (ctx, tgt, act) in enumerate(loader):
        ctx, tgt, act = ctx.to(device), tgt.to(device), act.to(device)
        bucket = torch.zeros(ctx.shape[0], dtype=torch.long, device=device)
        ctx_in = ctx
        if args.infer_noise > 0:
            lvl = args.infer_noise
            bucket = torch.full_like(bucket, min(int(lvl / args.train_noise_max * args.noise_buckets), args.noise_buckets - 1))
            ctx_in = (1.0 - lvl) ** 0.5 * ctx + lvl ** 0.5 * torch.randn_like(ctx)
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = diffusion.ddim_sample(lambda xt, t: model(xt, t, act, ctx_in, bucket), tgt.shape, steps=args.steps, eta=args.eta, device=device)
        torch.cuda.synchronize() if device == "cuda" else None
        t_sample += time.time() - t0; n += ctx.shape[0]
        pred_img, gt_img, last_img = decode(vae, pred), decode(vae, tgt), decode(vae, ctx[:, -4:])
        lat_mse = ((pred.float() - tgt.float()) ** 2).flatten(1).mean(1)
        for i in range(ctx.shape[0]):
            gi = int(idx[b * args.batch_size + i]); slot, start = ds.locate(gi)
            ep_id, map_id = ds.episodes[slot][0], ds.episodes[slot][3]
            r = dict(index=gi, episode=ep_id, map=map_id, start=start, action=int(act[i]),
                     psnr_dec=float(psnr(pred_img[i:i+1], gt_img[i:i+1])), lpips_dec=float(lp(pred_img[i:i+1] * 2 - 1, gt_img[i:i+1] * 2 - 1).flatten()),
                     copy_psnr_dec=float(psnr(last_img[i:i+1], gt_img[i:i+1])), latent_mse=float(lat_mse[i]),
                     hud_psnr_dec=float(psnr(pred_img[i:i+1, :, -HUD_ROWS:], gt_img[i:i+1, :, -HUD_ROWS:])))
            if raw is not None:
                tic = ds.target_tic(gi)
                if tic is None:
                    tic = (start + args.context_frames) * args.stride
                rf = torch.from_numpy(raw.get(ep_id, tic)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
                r.update(psnr_raw=float(psnr(pred_img[i:i+1], rf)), lpips_raw=float(lp(pred_img[i:i+1] * 2 - 1, rf * 2 - 1).flatten()),
                         copy_psnr_raw=float(psnr(last_img[i:i+1], rf)), vae_psnr=float(psnr(gt_img[i:i+1], rf)),
                         vae_lpips=float(lp(gt_img[i:i+1] * 2 - 1, rf * 2 - 1).flatten()),
                         hud_psnr_raw=float(psnr(pred_img[i:i+1, :, -HUD_ROWS:], rf[:, :, -HUD_ROWS:])),
                         hud_vae_psnr=float(psnr(gt_img[i:i+1, :, -HUD_ROWS:], rf[:, :, -HUD_ROWS:])))
            rows.append(r)
        if b < args.save_images:
            from torchvision.utils import save_image
            save_image(torch.cat([last_img, gt_img, pred_img]).cpu(), os.path.join(args.out_dir, f"batch_{b:03d}_last_gt_pred.png"), nrow=ctx.shape[0])
        if (b + 1) % 10 == 0:
            print(f"  {len(rows)}/{len(idx)} psnr_dec={np.mean([r['psnr_dec'] for r in rows]):.2f}", flush=True)

    keys = [k for k in rows[0] if k not in ("index", "episode", "map", "start", "action")]
    def agg(k):
        v = np.array([r[k] for r in rows], dtype=np.float64); v = v[~np.isnan(v)]
        return {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v))), "n": int(len(v))} if len(v) else None
    summary = {k: agg(k) for k in keys}
    summary["per_map"] = {str(m): {k: float(np.mean([r[k] for r in rows if r["map"] == m])) for k in ("psnr_dec", "lpips_dec")}
                          for m in sorted(set(r["map"] for r in rows))}
    summary["sampling_frames_per_s"] = n / max(t_sample, 1e-9)
    summary["config"] = {**vars(args), "step": step}
    json.dump(summary, open(os.path.join(args.out_dir, "metrics.json"), "w"), indent=1)
    with open(os.path.join(args.out_dir, "per_window.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "per_map")}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--backbone", choices=["dit", "unet"], required=True)
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--context-frames", type=int, default=32)
    p.add_argument("--num-actions", type=int, default=29)
    p.add_argument("--noise-buckets", type=int, default=10)
    p.add_argument("--infer-noise", type=float, default=0.0, help="context noise level at inference (0 = clean)")
    p.add_argument("--train-noise-max", type=float, default=0.7)
    p.add_argument("--latents-dir", required=True)
    p.add_argument("--parquet-dir", default="", help="raw recordings for lossless-frame scoring")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--split", required=True)
    p.add_argument("--subset", default="val", choices=["val", "train", "unseen_map"])
    p.add_argument("--num-windows", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--lpips-net", default="alex")
    p.add_argument("--vae-path", default="", help="fine-tuned VAE directory; default sd-vae-ft-mse")
    p.add_argument("--sd-path", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--save-images", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    main(p.parse_args())
