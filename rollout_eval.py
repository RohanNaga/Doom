"""
Autoregressive rollout engine and the metrics that read it.

Phase 1 (--rollout): for N held-out windows, seed with L real decision-frame latents and the
ground-truth action sequence, predict H frames one at a time feeding predictions back as
context, and write predicted latents, GT latents, and actions to one .npz per checkpoint.
Phase 2 (--score): drift curves (decoded PSNR and LPIPS per horizon, plus copy-seed baseline),
optional IDM action-following accuracy per horizon, and decoded clips for FVD.

    python rollout_eval.py --rollout --ckpt results/010-dit-l32/best.pt --backbone dit \
        --latents-dir data/latents_arnold --split data/split_arnold.json --subset val \
        --num-rollouts 256 --horizon 64 --out results/010-dit-l32/rollouts_val.npz
    python rollout_eval.py --score --rollouts results/010-dit-l32/rollouts_val.npz \
        --idm results/idm/idm.pt --out-dir results/010-dit-l32/rollout_metrics
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from diffusion_v import VDiffusion, noise_augment
from doom_data import list_latent_episodes, load_split
from doomdit_utils import LATENT_SCALE, load_vae


def collect_rollout_windows(latents_dir, episode_ids, L, H, n, seed):
    """Windows with L seed frames and H future frames, spread over episodes."""
    eps = [(ep, np.load(lp, mmap_mode="r"), np.load(mp)) for ep, lp, mp in list_latent_episodes(latents_dir) if ep in set(int(e) for e in episode_ids)]
    eps = [(ep, lat, m) for ep, lat, m in eps if lat.shape[0] >= L + H]
    rng = np.random.RandomState(seed)
    picks = []
    for _ in range(n):
        ep, lat, m = eps[rng.randint(len(eps))]
        s = rng.randint(0, lat.shape[0] - L - H + 1)
        picks.append((ep, int(m["map_id"][0]), s, lat, m))
    return picks


@torch.no_grad()
def do_rollout(args):
    from backbones import build_model
    device = "cuda"
    model = build_model(args.backbone, args.num_actions, args.context_frames, args.noise_buckets, grad_ckpt=False,
                        warm_start=None if args.backbone == "dit" else args.sd_path, cache_dir=args.hf_cache)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["ema"] if (args.use_ema and "ema" in ck) else ck["model"]
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    model = model.to(device).eval()
    diffusion = VDiffusion(device=device)
    split = load_split(args.split)
    picks = collect_rollout_windows(args.latents_dir, split[args.subset], args.context_frames, args.horizon, args.num_rollouts, args.seed)
    L, H, B = args.context_frames, args.horizon, args.batch_size
    torch.manual_seed(args.seed)
    pred_all, gt_all, act_all, meta = [], [], [], []
    t0 = time.time()
    for i in range(0, len(picks), B):
        chunk = picks[i:i + B]
        seed_lat = torch.stack([torch.from_numpy(np.asarray(lat[s:s + L], dtype=np.float32)) for _, _, s, lat, _ in chunk]).to(device)
        gt = np.stack([np.asarray(lat[s + L:s + L + H], dtype=np.float16) for _, _, s, lat, _ in chunk])
        acts = np.stack([m["action"][s + L - 1:s + L - 1 + H].astype(np.int64) for _, _, s, _, m in chunk])
        ctx = seed_lat.reshape(len(chunk), -1, 32, 40)
        preds = []
        for h in range(H):
            act = torch.from_numpy(acts[:, h]).to(device)
            ctx_in, bucket = (ctx, torch.zeros(len(chunk), dtype=torch.long, device=device))
            if args.infer_noise > 0:
                ctx_in, bucket = fixed_noise(ctx, args.infer_noise, args.train_noise_max, args.noise_buckets)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = diffusion.ddim_sample(lambda xt, t: model(xt, t, act, ctx_in, bucket), (len(chunk), 4, 32, 40), steps=args.steps, eta=args.eta, device=device)
            preds.append(x.half().cpu().numpy())
            ctx = torch.cat([ctx[:, 4:], x.float()], dim=1)
        pred_all.append(np.stack(preds, axis=1)); gt_all.append(gt); act_all.append(acts)
        meta += [(ep, mp, s) for ep, mp, s, _, _ in chunk]
        print(f"  {i + len(chunk)}/{len(picks)} rollouts, {(time.time() - t0) / (i + len(chunk)):.1f} s each", flush=True)
    np.savez(args.out, pred=np.concatenate(pred_all), gt=np.concatenate(gt_all), actions=np.concatenate(act_all),
             seed=np.stack([np.asarray(lat[s:s + L], dtype=np.float16) for _, _, s, lat, _ in picks]),
             episode=np.array([m[0] for m in meta]), map=np.array([m[1] for m in meta]), start=np.array([m[2] for m in meta]),
             config=json.dumps({**vars(args), "step": ck.get("step", "?")}))
    print("DONE", args.out)


def fixed_noise(ctx, level, train_max, buckets):
    """Corrupt context at one fixed level, with the bucket id defined on the training scale."""
    b = ctx.shape[0]
    bucket = torch.full((b,), min(int(level / train_max * buckets), buckets - 1), dtype=torch.long, device=ctx.device)
    return (1.0 - level) ** 0.5 * ctx + level ** 0.5 * torch.randn_like(ctx), bucket


def psnr(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


@torch.no_grad()
def do_score(args):
    import lpips
    device = "cuda"
    d = np.load(args.rollouts)
    pred, gt, seed, actions = d["pred"], d["gt"], d["seed"], d["actions"]
    N, H = pred.shape[:2]
    vae = load_vae(device) if not args.vae_path else __import__("diffusers").models.AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def dec(z):
        img = vae.decode(torch.from_numpy(np.asarray(z, dtype=np.float32)).to(device) / LATENT_SCALE).sample[:, :, :240]
        return (img * 0.5 + 0.5).clamp(0, 1)

    psnr_h, lpips_h, copy_h, lat_h = np.zeros(H), np.zeros(H), np.zeros(H), np.zeros(H)
    clips_pred, clips_gt = [], []
    for n in range(N):
        last = dec(seed[n, -1:])
        rp, rg = [], []
        for h0 in range(0, H, args.decode_batch):
            p = dec(pred[n, h0:h0 + args.decode_batch]); g = dec(gt[n, h0:h0 + args.decode_batch])
            k = p.shape[0]
            psnr_h[h0:h0 + k] += psnr(p, g).cpu().numpy()
            lpips_h[h0:h0 + k] += lp(p * 2 - 1, g * 2 - 1).flatten().cpu().numpy()
            copy_h[h0:h0 + k] += psnr(last.expand_as(g), g).cpu().numpy()
            lat_h[h0:h0 + k] += ((pred[n, h0:h0 + k].astype(np.float32) - gt[n, h0:h0 + k].astype(np.float32)) ** 2).reshape(k, -1).mean(1)
            if n < args.save_clips:
                rp.append((p * 255).byte().cpu().numpy()); rg.append((g * 255).byte().cpu().numpy())
        if rp:   # one clip per rollout, all H frames, (H, 3, 240, 320)
            clips_pred.append(np.concatenate(rp)); clips_gt.append(np.concatenate(rg))
        if (n + 1) % 32 == 0:
            print(f"  scored {n + 1}/{N}", flush=True)
    out = {"horizon": list(range(1, H + 1)), "psnr": (psnr_h / N).tolist(), "lpips": (lpips_h / N).tolist(),
           "copy_seed_psnr": (copy_h / N).tolist(), "latent_mse": (lat_h / N).tolist(), "num_rollouts": int(N)}
    for hh in (8, 16, 32, 64):
        if hh <= H:
            out[f"psnr@{hh}"] = float(psnr_h[hh - 1] / N); out[f"lpips@{hh}"] = float(lpips_h[hh - 1] / N)

    if args.idm:
        from train_idm import IDM
        ck = torch.load(args.idm, map_location="cpu", weights_only=False)
        idm = IDM(ck["num_actions"], ck["width"]).to(device).eval(); idm.load_state_dict(ck["model"])
        act2mov = ck["act2mov"]; mov = torch.tensor([act2mov.get(a, -1) for a in range(ck["num_actions"])], device=device)
        top1_h, mov_h = np.zeros(H), np.zeros(H)
        for n in range(N):
            seq = np.concatenate([seed[n, -1:], pred[n]], axis=0).astype(np.float32)   # H+1 frames
            x = torch.from_numpy(np.concatenate([seq[:-1], seq[1:]], axis=1)).to(device)  # (H, 8, 32, 40)
            p = idm(x).argmax(1); y = torch.from_numpy(actions[n]).to(device)
            top1_h += (p == y).cpu().numpy(); mov_h += (mov[p] == mov[y]).cpu().numpy()
        out["idm_top1"] = (top1_h / N).tolist(); out["idm_movement"] = (mov_h / N).tolist()
        out["idm_top1_mean"] = float(top1_h.mean() / N); out["idm_movement_mean"] = float(mov_h.mean() / N)
        out["idm_ceiling_top1"] = ck.get("val_top1"); out["idm_ceiling_movement"] = ck.get("val_movement")
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(args.out_dir, "drift.json"), "w"), indent=1)
    if clips_pred:
        np.savez_compressed(os.path.join(args.out_dir, "clips_u8.npz"), pred=np.stack(clips_pred), gt=np.stack(clips_gt))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rollout", action="store_true"); p.add_argument("--score", action="store_true")
    p.add_argument("--ckpt"); p.add_argument("--backbone", choices=["dit", "unet"]); p.add_argument("--use-ema", action="store_true")
    p.add_argument("--context-frames", type=int, default=32); p.add_argument("--num-actions", type=int, default=29)
    p.add_argument("--noise-buckets", type=int, default=10); p.add_argument("--infer-noise", type=float, default=0.0); p.add_argument("--train-noise-max", type=float, default=0.7)
    p.add_argument("--latents-dir"); p.add_argument("--split"); p.add_argument("--subset", default="val")
    p.add_argument("--num-rollouts", type=int, default=256); p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--steps", type=int, default=50); p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--sd-path", default="CompVis/stable-diffusion-v1-4"); p.add_argument("--hf-cache", default=None)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out")
    p.add_argument("--rollouts"); p.add_argument("--idm", default=""); p.add_argument("--vae-path", default="")
    p.add_argument("--decode-batch", type=int, default=16); p.add_argument("--save-clips", type=int, default=64); p.add_argument("--out-dir")
    a = p.parse_args()
    if a.rollout:
        do_rollout(a)
    if a.score:
        do_score(a)
