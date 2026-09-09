"""
Train a DoomDiT world model, either backbone, under one recipe.

Shared between backbones: stride-4 latents, L context frames channel-stacked, single action at
the last context frame, GameNGen context-noise augmentation with a bucket id, velocity target,
AdamW, bf16 autocast with fp32 master weights, EMA in fp32 on the CPU, held-out latent v-loss
for checkpoint selection. The only thing the --backbone flag changes is the network.

Fit check (no data needed):
    python train_wm.py --backbone dit --fit-check 30 --per-gpu-batch 4 --context-frames 32
Training (single process or under `accelerate launch --multi_gpu`):
    accelerate launch --multi_gpu --num_processes 4 train_wm.py --backbone dit \
        --latents-dir data/latents_arnold --split data/split_arnold.json --results-dir results/010-dit-l32
"""
import argparse
import json
import math
import os
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from backbones import build_model
from diffusion_v import VDiffusion, noise_augment


class SyntheticWindows(Dataset):
    def __init__(self, n, context_frames, num_actions):
        self.n, self.L, self.A = n, context_frames, num_actions

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return (torch.randn(4 * self.L, 32, 40, generator=g), torch.randn(4, 32, 40, generator=g),
                torch.randint(0, self.A, (1,), generator=g)[0])


def build_loaders(args):
    if args.fit_check:
        ds = SyntheticWindows(args.per_gpu_batch * 64, args.context_frames, args.num_actions)
        return ds, None
    from doom_data import LatentWindowDataset, load_split
    split = load_split(args.split)
    train = LatentWindowDataset(args.latents_dir, split["train"], args.context_frames)
    val = LatentWindowDataset(args.latents_dir, split["val"], args.context_frames)
    rng = np.random.RandomState(0)
    val_idx = np.sort(rng.choice(len(val), size=min(args.val_windows, len(val)), replace=False))
    return train, Subset(val, val_idx.tolist())


@torch.no_grad()
def ema_update(ema_params, params, decay):
    for e, p in zip(ema_params, params):
        e.mul_(decay).add_(p.detach().float().cpu(), alpha=1.0 - decay)


def main(args):
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    acc = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=1)
    set_seed(args.seed + acc.process_index)
    device = acc.device
    world = acc.num_processes
    accum = max(1, args.global_batch // (args.per_gpu_batch * world))
    is_main = acc.is_main_process

    os.makedirs(args.results_dir, exist_ok=True)
    log_path = os.path.join(args.results_dir, "log.jsonl")

    def log(**kw):
        if is_main:
            kw["time"] = time.time()
            with open(log_path, "a") as f:
                f.write(json.dumps(kw) + "\n")
            print(json.dumps(kw), flush=True)

    model = build_model(args.backbone, args.num_actions, args.context_frames, args.noise_buckets,
                        grad_ckpt=not args.no_grad_ckpt, warm_start=args.warm_start, cache_dir=args.hf_cache)
    n_params = sum(p.numel() for p in model.parameters())
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, fused=(device.type == "cuda"))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))

    train_ds, val_ds = build_loaders(args)
    loader = DataLoader(train_ds, batch_size=args.per_gpu_batch, shuffle=True, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
    model, opt, loader, sched = acc.prepare(model, opt, loader, sched)
    diffusion = VDiffusion(device=device)
    raw = acc.unwrap_model(model)
    ema = [p.detach().float().cpu().clone() for p in raw.parameters()] if args.ema_every > 0 else None

    if is_main:
        try:
            git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            git = "?"
        with open(os.path.join(args.results_dir, "config.json"), "w") as f:
            json.dump({**vars(args), "git": git, "params": n_params, "world_size": world, "accum": accum}, f, indent=1)
    log(event="start", backbone=args.backbone, params=n_params, world=world, accum=accum,
        per_gpu_batch=args.per_gpu_batch, global_batch=args.per_gpu_batch * world * accum)

    def model_fn(ctx, act, bucket):
        return lambda xt, t: model(xt, t, act, ctx, bucket)

    @torch.no_grad()
    def evaluate():
        if val_ds is None:
            return None
        model.eval()
        vl = DataLoader(val_ds, batch_size=args.per_gpu_batch, shuffle=False, num_workers=2)
        g = torch.Generator(device=device).manual_seed(1234)
        tot, n = torch.zeros((), device=device), 0
        for ctx, tgt, act in vl:
            ctx, tgt, act = ctx.to(device), tgt.to(device), act.to(device)
            ctx_n, bucket = noise_augment(ctx, args.noise_aug_max, args.noise_buckets, generator=g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = diffusion.training_loss(model_fn(ctx_n, act, bucket), tgt, noise=torch.randn(tgt.shape, device=device, generator=g))
            tot += loss.detach() * ctx.shape[0]; n += ctx.shape[0]
        tot = acc.reduce(tot, reduction="sum"); n = acc.reduce(torch.tensor(n, device=device), reduction="sum")
        model.train()
        return (tot / n).item()

    model.train()
    step, t0, tokens = 0, time.time(), 0
    running = []
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    done = False
    max_steps = args.fit_check if args.fit_check else args.steps
    best_val = float("inf")
    while not done:
        for micro, (ctx, tgt, act) in enumerate(loader):
            ctx, tgt, act = ctx.to(device, non_blocking=True), tgt.to(device, non_blocking=True), act.to(device)
            ctx_n, bucket = noise_augment(ctx, args.noise_aug_max, args.noise_buckets)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = diffusion.training_loss(model_fn(ctx_n, act, bucket), tgt) / accum
            acc.backward(loss)
            running.append(loss.item() * accum)
            if (micro + 1) % accum != 0:
                continue
            if args.clip > 0:
                acc.clip_grad_norm_(model.parameters(), args.clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if ema is not None and step % args.ema_every == 0:
                ema_update(ema, raw.parameters(), args.ema_decay ** args.ema_every)
            if step % args.log_every == 0 or step == max_steps:
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
                log(event="train", step=step, loss=float(np.mean(running)), lr=sched.get_last_lr()[0],
                    steps_per_s=step / dt, peak_mem_gb=round(mem, 2))
                running = []
            if not args.fit_check and step % args.val_every == 0 and val_ds is not None:
                v = evaluate()
                log(event="val", step=step, val_loss=v)
                if is_main and v < best_val:
                    best_val = v
                    torch.save({"model": {k: t.to(torch.bfloat16) for k, t in raw.state_dict().items()},
                                "step": step, "val_loss": v, "args": vars(args)}, os.path.join(args.results_dir, "best.pt"))
            if not args.fit_check and is_main and step % args.ckpt_every == 0:
                ck = {"model": {k: t.to(torch.bfloat16) for k, t in raw.state_dict().items()}, "step": step, "args": vars(args)}
                if ema is not None:
                    ck["ema"] = {k: t.to(torch.bfloat16) for k, t in zip(raw.state_dict().keys(), ema)}
                torch.save(ck, os.path.join(args.results_dir, f"{step:07d}.pt"))
                olds = sorted(p for p in os.listdir(args.results_dir) if p.endswith(".pt") and p[0].isdigit())
                for p in olds[:-args.keep_last]:
                    os.remove(os.path.join(args.results_dir, p))
            if step >= max_steps:
                done = True; break
    if args.fit_check:
        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
        log(event="fit_check", backbone=args.backbone, context_frames=args.context_frames, per_gpu_batch=args.per_gpu_batch,
            world=world, steps=step, steps_per_s=step / dt, peak_mem_gb=round(mem, 2), params=n_params, optim=args.optim)
    acc.wait_for_everyone()
    log(event="end", step=step)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["dit", "unet"], required=True)
    p.add_argument("--context-frames", type=int, default=32)
    p.add_argument("--num-actions", type=int, default=29)
    p.add_argument("--noise-buckets", type=int, default=10)
    p.add_argument("--noise-aug-max", type=float, default=0.7)
    p.add_argument("--latents-dir", default="data/latents_arnold")
    p.add_argument("--split", default="data/split_arnold.json")
    p.add_argument("--results-dir", default="results/fit_check")
    p.add_argument("--warm-start", default=None, help="DiT: path to DiT-XL-2-256x256.pt; U-Net: SD 1.4 repo or path")
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--global-batch", type=int, default=32)
    p.add_argument("--per-gpu-batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=90000)
    p.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw")
    p.add_argument("--ema-every", type=int, default=8, help="0 disables the fp32 CPU EMA")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--no-grad-ckpt", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=5000)
    p.add_argument("--val-windows", type=int, default=1024)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--keep-last", type=int, default=2)
    p.add_argument("--fit-check", type=int, default=0, help="run N synthetic steps, report steps/s and memory, exit")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
