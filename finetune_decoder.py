"""
Fine-tune the sd-vae-ft-mse decoder on Doom frames (GameNGen section 3.2.2).

The encoder stays frozen, so the latents both backbones train on are unchanged; only the
decoder learns to render Doom's HUD digits and textures from them. Loss is MSE against the
lossless target frame (GameNGen's choice); `--lpips-weight` adds a perceptual term as the P1
variant. Reports full-frame and HUD-crop (bottom 32 rows) PSNR and LPIPS on held-out frames
before and after, which is the VAE-ceiling row of every results table.

Usage:
    python finetune_decoder.py --in-dir raw_arnold --split split_arnold.json \
        --out-dir vae_decoder_arnold --train-frames 50000 --val-frames 2000 --epochs 2 --device cuda:3
"""
import argparse
import glob
import io
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image

from doomdit_utils import LATENT_SCALE, load_vae

PAD_TO = 256
HUD_ROWS = 32


def sample_frames(parquet_dir, episode_ids, n, stride, seed):
    """Uniformly sample n decision frames across the given episodes. Returns list of (path, row)."""
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    paths = [os.path.join(parquet_dir, f"ep_{e:05d}.parquet") for e in episode_ids]
    paths = [p for p in paths if os.path.exists(p)]
    per = max(1, n // len(paths) + 1)
    out = []
    for p in paths:
        nrows = pq.ParquetFile(p).metadata.num_rows
        rows = [r for r in range(0, nrows, stride)]
        out += [(p, r) for r in rng.sample(rows, min(per, len(rows)))]
    rng.shuffle(out)
    return out[:n]


def load_frames(items):
    import pyarrow.parquet as pq
    by_path = {}
    for p, r in items:
        by_path.setdefault(p, []).append(r)
    frames = {}
    for p, rows in by_path.items():
        col = pq.read_table(p, columns=["frame"])["frame"]
        for r in rows:
            frames[(p, r)] = np.asarray(Image.open(io.BytesIO(col[r].as_py())).convert("RGB"), dtype=np.uint8)
    return [frames[k] for k in items]


def to_tensor(frames_u8, device):
    x = torch.from_numpy(np.stack(frames_u8)).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return torch.nn.functional.pad(x, (0, 0, 0, PAD_TO - x.shape[2]))


def psnr(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(4.0 / mse)      # inputs in [-1, 1]


@torch.no_grad()
def evaluate(vae, frames, device, lpips_fn, bs=32):
    vae.eval()
    tot = {"psnr": [], "psnr_hud": [], "lpips": [], "lpips_hud": []}
    for i in range(0, len(frames), bs):
        x = to_tensor(frames[i:i + bs], device)
        z = vae.encode(x).latent_dist.mean
        y = vae.decode(z).sample.clamp(-1, 1)
        x, y = x[:, :, :240], y[:, :, :240]
        tot["psnr"].append(psnr(x, y).cpu())
        tot["psnr_hud"].append(psnr(x[:, :, -HUD_ROWS:], y[:, :, -HUD_ROWS:]).cpu())
        if lpips_fn is not None:
            tot["lpips"].append(lpips_fn(x, y).flatten().cpu())
            tot["lpips_hud"].append(lpips_fn(x[:, :, -HUD_ROWS:], y[:, :, -HUD_ROWS:]).flatten().cpu())
    return {k: float(torch.cat(v).mean()) for k, v in tot.items() if v}


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    split = json.load(open(args.split))
    train_items = sample_frames(args.in_dir, split["train"], args.train_frames, args.stride, 0)
    val_items = sample_frames(args.in_dir, split["val"], args.val_frames, args.stride, 1)
    print(f"loading {len(train_items)} train and {len(val_items)} val frames", flush=True)
    t0 = time.time()
    train_frames, val_frames = load_frames(train_items), load_frames(val_items)
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    vae = load_vae(device)
    lpips_fn = None
    if args.lpips_weight > 0 or args.report_lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)
    before = evaluate(vae, val_frames, device, lpips_fn)
    print("before:", json.dumps(before), flush=True)

    for p in vae.encoder.parameters():
        p.requires_grad_(False)
    for p in vae.quant_conv.parameters():
        p.requires_grad_(False)
    params = list(vae.decoder.parameters()) + list(vae.post_quant_conv.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = len(train_frames) // args.batch_size
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * max(0.05, 1 - s / max(1, total)))
    step = 0
    for ep in range(args.epochs):
        order = np.random.RandomState(ep).permutation(len(train_frames))
        vae.decoder.train()
        for i in range(steps_per_epoch):
            idx = order[i * args.batch_size:(i + 1) * args.batch_size]
            x = to_tensor([train_frames[j] for j in idx], device)
            with torch.no_grad():
                z = vae.encode(x).latent_dist.mean
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                y = vae.decode(z).sample
            loss = torch.mean((y.float() - x) ** 2)
            if args.lpips_weight > 0:
                loss = loss + args.lpips_weight * lpips_fn(y.float()[:, :, :240].clamp(-1, 1), x[:, :, :240]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                print(f"step {step}/{total} loss {loss.item():.5f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
        mid = evaluate(vae, val_frames, device, lpips_fn)
        print(f"epoch {ep + 1}:", json.dumps(mid), flush=True)
    after = evaluate(vae, val_frames, device, lpips_fn)
    print("after:", json.dumps(after), flush=True)
    vae.save_pretrained(os.path.join(args.out_dir, "vae"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"before": before, "after": after, "args": vars(args), "train_frames": len(train_frames),
                   "val_frames": len(val_frames), "steps": step}, f, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True, help="parquet episodes")
    p.add_argument("--split", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-frames", type=int, default=50000)
    p.add_argument("--val-frames", type=int, default=2000)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lpips-weight", type=float, default=0.0)
    p.add_argument("--report-lpips", action="store_true")
    p.add_argument("--device", default="cuda:3")
    main(p.parse_args())
