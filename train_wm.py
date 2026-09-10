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


class SeededCorruption(Dataset):
    """Wraps a validation dataset so each window's timestep, context noise level and noise, and target noise
    are drawn from a generator seeded by the window's own index: identical across checkpoints, backbones,
    batch sizes, and world sizes."""

    def __init__(self, ds, max_level, num_steps):
        self.ds, self.max_level, self.num_steps = ds, max_level, num_steps

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ctx, tgt, act = self.ds[i]
        g = torch.Generator().manual_seed(1234 + int(i))
        t = torch.randint(0, self.num_steps, (1,), generator=g)[0]
        level = torch.rand((), generator=g) * self.max_level
        ctx_eps = torch.randn(ctx.shape, generator=g, dtype=ctx.dtype)
        tgt_noise = torch.randn(tgt.shape, generator=g, dtype=tgt.dtype)
        return ctx, tgt, act, t, level, ctx_eps, tgt_noise


def build_loaders(args):
    if args.fit_check:
        ds = SyntheticWindows(args.per_gpu_batch * 64, args.context_frames, args.num_actions)
        return ds, None
    from doom_data import LatentWindowDataset, load_split
    split = load_split(args.split)
    train = LatentWindowDataset(args.latents_dir, split["train"], args.context_frames, require_chains=args.require_verified_transitions)
    val = LatentWindowDataset(args.latents_dir, split["val"], args.context_frames, require_chains=args.require_verified_transitions)
    rng = np.random.RandomState(0)
    val_idx = np.sort(rng.choice(len(val), size=min(args.val_windows, len(val)), replace=False))
    return train, Subset(val, val_idx.tolist())


@torch.no_grad()
def ema_update(ema_params, params, decay):
    for e, p in zip(ema_params, params):
        e.mul_(decay).add_(p.detach().float().cpu(), alpha=1.0 - decay)



def save_checkpoint(obj, path, remote=None, keep_local=True):
    """Serialize once, then write to the remote copy of record and, space permitting, to the local path.

    `remote` is "user@host:/dir" reached with the key in REMOTE_SSH; the file lands as <dir>/<run>/<name>
    through an atomic rename. The local write is skipped when the disk has less than 1.5x the file free,
    so a full shared disk degrades to remote-only instead of killing the run. Raises only if no copy was written.
    """
    import io, shutil, subprocess
    buf = io.BytesIO(); torch.save(obj, buf); data = buf.getvalue()
    written = []
    if remote:
        host, rdir = remote.split(":", 1)
        rel = os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))
        rpath = os.path.join(rdir, rel)
        cmd = REMOTE_SSH + [host, f"mkdir -p {os.path.dirname(rpath)} && cat > {rpath}.tmp && mv {rpath}.tmp {rpath}"]
        try:
            r = subprocess.run(cmd, input=data, capture_output=True, timeout=1800)
            if r.returncode == 0:
                written.append(rpath)
            else:
                print(f"remote checkpoint write failed: {r.stderr.decode()[-300:]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"remote checkpoint write timed out after 1800 s: {rpath}", flush=True)
    if keep_local or not written:
        free = shutil.disk_usage(os.path.dirname(path)).free
        if free > 1.5 * len(data) or not written:
            tmp = path + ".tmp"
            try:
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, path); written.append(path)
            except OSError as e:   # a full local disk is survivable once the remote copy exists
                print(f"local checkpoint write failed ({e}); remote copy {'exists' if written else 'MISSING'}", flush=True)
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        else:
            print(f"skipped local checkpoint {path}: {free/2**30:.1f} GB free", flush=True)
    if not written:
        raise RuntimeError(f"could not write checkpoint {path} anywhere")
    return written


def prune_remote(remote, run_name, keep):
    """Keep only the newest `keep` rolling checkpoints (NNNNNNN.pt) of one run on the remote copy of record."""
    host, rdir = remote.split(":", 1)
    d = os.path.join(rdir, run_name)
    cmd = REMOTE_SSH + [host, f"cd {d} && ls [0-9]*.pt 2>/dev/null | sort | head -n -{keep} | xargs -r rm -f"]
    subprocess.run(cmd, capture_output=True, timeout=120)


REMOTE_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"] + (["-i", os.environ["DOOM_SSH_KEY"]] if os.environ.get("DOOM_SSH_KEY") else [])

def main(args):
    from accelerate import Accelerator
    from accelerate.utils import set_seed, InitProcessGroupKwargs
    from datetime import timedelta
    acc = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=1))], mixed_precision="bf16", gradient_accumulation_steps=1)
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
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps(kw) + "\n")
            except OSError as e:   # a full disk must not kill training; stdout and the remote copy still get the line
                print(f"log write failed: {e}", flush=True)
            print(json.dumps(kw), flush=True)

    model = build_model(args.backbone, args.num_actions, args.context_frames, args.noise_buckets,
                        grad_ckpt=not args.no_grad_ckpt, warm_start=args.warm_start, cache_dir=args.hf_cache,
                        action_dropout=args.action_dropout)
    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.float() for k, v in ck["model"].items()}, strict=True)
        start_step = int(ck.get("step", 0))
        print(f"resumed weights from {args.resume} at step {start_step}" + (" with optimizer, scheduler, EMA, and RNG state" if "optimizer" in ck else " (optimizer state reset, warmup restarts)"))
        # every rank reseeds from (seed, rank, step): distinct per rank, reproducible, and independent of which rank
        # wrote the checkpoint. Data order after a resume is a fresh shuffle from this seed (state-exact, order reshuffled).
        set_seed(args.seed + acc.process_index + 7919 * start_step)
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
    model, opt, loader = acc.prepare(model, opt, loader)   # scheduler stays unwrapped: one step per optimizer update
    diffusion = VDiffusion(device=device)
    raw = acc.unwrap_model(model)
    if args.resume and "optimizer" in ck:
        opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
    ema = [p.detach().float().cpu().clone() for p in raw.parameters()] if args.ema_every > 0 else None
    if args.resume and ema is not None and "ema" in ck:
        for e, (k, _) in zip(ema, raw.state_dict().items()):
            if k in ck["ema"] and ck["ema"][k].shape == e.shape:
                e.copy_(ck["ema"][k].float())

    if is_main:
        try:
            git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            git = "?"
        import hashlib
        split_hash = hashlib.md5(open(args.split, "rb").read()).hexdigest() if (not args.fit_check and os.path.exists(args.split)) else None
        with open(os.path.join(args.results_dir, "config.json"), "w") as f:
            json.dump({**vars(args), "git": git, "params": n_params, "world_size": world, "accum": accum,
                       "split_md5": split_hash, "torch": torch.__version__}, f, indent=1)
    log(event="start", backbone=args.backbone, params=n_params, world=world, accum=accum,
        per_gpu_batch=args.per_gpu_batch, global_batch=args.per_gpu_batch * world * accum)

    def model_fn(ctx, act, bucket):
        return lambda xt, t: model(xt, t, act, ctx, bucket)

    @torch.no_grad()
    def evaluate():
        if val_ds is None:
            return None
        model.eval()
        # fixed corruption per window: the same timesteps, context noise, and target noise for every
        # checkpoint and both backbones, independent of batch size and world size
        vl = DataLoader(SeededCorruption(val_ds, args.noise_aug_max, diffusion.num_steps), batch_size=args.per_gpu_batch, shuffle=False, num_workers=2)
        tot = torch.zeros((), device=device); n = torch.zeros((), device=device)
        bins = torch.zeros(4, device=device); bin_n = torch.zeros(4, device=device)
        for ctx, tgt, act, t, level, ctx_eps, tgt_noise in vl:
            ctx, tgt, act, t = ctx.to(device), tgt.to(device), act.to(device), t.to(device)
            ctx_n, bucket = noise_augment(ctx, args.noise_aug_max, args.noise_buckets, level=level.to(device), eps=ctx_eps.to(device))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = diffusion.training_loss(model_fn(ctx_n, act, bucket), tgt, noise=tgt_noise.to(device), t=t, per_sample=True)
            tot += loss.detach().sum(); n += loss.numel()
            q = (t * 4) // diffusion.num_steps
            bins.index_add_(0, q, loss.detach()); bin_n.index_add_(0, q, torch.ones_like(loss))
        tot, n = acc.reduce(tot, reduction="sum"), acc.reduce(n, reduction="sum")
        bins, bin_n = acc.reduce(bins, reduction="sum"), acc.reduce(bin_n, reduction="sum")
        model.train()
        return (tot / n).item(), (bins / bin_n.clamp(min=1)).tolist()

    model.train()
    step, t0, tokens = start_step, time.time(), 0
    running, grad_norms, skipped = [], [], 0
    # probe tensors: input projection, output layer, and adaLN modulation of the first, middle, and last DiT blocks (or U-Net analogues)
    want = ["x_embedder.proj.weight", "final_layer.linear.weight", "blocks.0.adaLN_modulation.1.weight", "blocks.14.adaLN_modulation.1.weight",
            "blocks.27.adaLN_modulation.1.weight", "conv_in.weight", "conv_out.weight", "time_embedding.linear_2.weight", "class_embedding.weight"]
    probe_params = [(n, p) for n, p in raw.named_parameters() if any(n.endswith(w) for w in want)]
    probe_names = [n for n, _ in probe_params]
    probe_prev = {n: p.detach().clone() for n, p in probe_params}
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    done = False
    max_steps = args.fit_check if args.fit_check else args.steps
    best_val = float(ck.get("best_val", float("inf"))) if args.resume else float("inf")
    val_hist = []
    micro = int(ck.get("micro", 0)) if args.resume else 0
    while not done:
        for ctx, tgt, act in loader:
            micro += 1
            ctx, tgt, act = ctx.to(device, non_blocking=True), tgt.to(device, non_blocking=True), act.to(device)
            ctx_n, bucket = noise_augment(ctx, args.noise_aug_max, args.noise_buckets)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = diffusion.training_loss(model_fn(ctx_n, act, bucket), tgt) / accum
            acc.backward(loss)
            running.append(loss.item() * accum)
            if micro % accum != 0:
                continue
            gn = acc.clip_grad_norm_(model.parameters(), args.clip) if args.clip > 0 else torch.zeros(())
            grad_norms.append(float(gn))   # pre-clip norm: the instability diagnostic that the loss alone hides
            if not math.isfinite(float(gn)):
                # gradients are DDP-averaged before clipping, so all ranks agree; skip the update without advancing the schedule
                opt.zero_grad(set_to_none=True); skipped += 1
                if skipped > 20:
                    raise RuntimeError(f"{skipped} non-finite gradient updates; stopping before Adam state is corrupted")
                continue
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if probe_names and step % 100 == 0:
                # actual parameter change over this update relative to weight norm, for a few tensors adaLN and I/O depend on
                with torch.no_grad():
                    ratios = {n: float((p.detach() - probe_prev[n]).norm() / (probe_prev[n].norm() + 1e-12)) for n, p in probe_params}
                    for n, p in probe_params:
                        probe_prev[n].copy_(p.detach())
                log(event="update_ratio", step=step, **ratios)
            if ema is not None and step % args.ema_every == 0:
                ema_update(ema, raw.parameters(), args.ema_decay ** args.ema_every)
            if step % args.log_every == 0 or step == max_steps:
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
                log(event="train", step=step, loss=float(np.mean(running)), lr=sched.get_last_lr()[0],
                    steps_per_s=(step - start_step) / dt, peak_mem_gb=round(mem, 2),
                    grad_norm=float(np.mean(grad_norms)), grad_norm_max=float(np.max(grad_norms)),
                    clip_frac=float(np.mean([g > args.clip for g in grad_norms])) if args.clip > 0 else 0.0,
                    nonfinite_loss=int(sum(not np.isfinite(x) for x in running)), skipped_updates=skipped)
                running, grad_norms = [], []
            if not args.fit_check and step % args.val_every == 0 and val_ds is not None:
                v, vbins = evaluate()
                val_hist.append(v)
                excursion = len(val_hist) > 3 and v > 1.15 * float(np.median(val_hist[-4:-1]))
                log(event="val", step=step, val_loss=v, val_loss_by_t_quartile=vbins, excursion=bool(excursion))
                if is_main and args.remote_results and step % args.snapshot_every == 0:
                    # compact bf16 weights at every validation so an excursion can be located afterwards; never pruned
                    save_checkpoint({"model": {k: t.detach().cpu().to(torch.bfloat16) for k, t in raw.state_dict().items()},
                                     "ema": {k: t.to(torch.bfloat16) for k, t in zip(raw.state_dict().keys(), ema)} if ema is not None else None,
                                     "step": step, "val_loss": v, "args": vars(args)},
                                    os.path.join(args.results_dir, f"snap_{step:07d}.pt"), args.remote_results, keep_local=False)
                if is_main and v < best_val:
                    best_val = v
                    save_checkpoint({"model": {k: t.detach().cpu().to(torch.bfloat16) for k, t in raw.state_dict().items()},
                                     "step": step, "val_loss": v, "args": vars(args)},
                                    os.path.join(args.results_dir, "best.pt"), args.remote_results, keep_local=True)
            if not args.fit_check and is_main and step % args.ckpt_every == 0:
                # prune before writing so the disk never holds keep_last + 1 rolling checkpoints
                olds = sorted(p for p in os.listdir(args.results_dir) if p.endswith(".pt") and p[0].isdigit())
                for p in olds[:-max(0, args.keep_last - 1)] if args.keep_last > 0 else olds:
                    os.remove(os.path.join(args.results_dir, p))
                # recovery checkpoint: fp32 master weights and EMA, optimizer, scheduler, RNG, so --resume reproduces the run
                ck = {"model": {k: t.detach().cpu().float() for k, t in raw.state_dict().items()}, "step": step, "args": vars(args),
                      "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "best_val": best_val, "micro": micro,
                      "rng": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                              "numpy": np.random.get_state()}}
                if ema is not None:
                    ck["ema"] = {k: t.clone() for k, t in zip(raw.state_dict().keys(), ema)}
                # rolling checkpoints live on the remote copy of record; the local copy is kept only when --keep-last > 0
                written = save_checkpoint(ck, os.path.join(args.results_dir, f"{step:07d}.pt"), args.remote_results, keep_local=args.keep_last > 0)
                if args.remote_results and is_main:
                    try:
                        if any(w.startswith(args.remote_results.split(":", 1)[1]) for w in written):
                            prune_remote(args.remote_results, os.path.basename(args.results_dir.rstrip("/")), args.keep_remote)
                        subprocess.run(["rsync", "-a", "-e", " ".join(REMOTE_SSH), "--include=*.json", "--include=*.jsonl", "--exclude=*",
                                        args.results_dir + "/", args.remote_results.rstrip("/") + "/" + os.path.basename(args.results_dir.rstrip("/")) + "/"],
                                       capture_output=True, timeout=600)
                    except (subprocess.TimeoutExpired, OSError) as e:
                        print(f"remote housekeeping failed: {e}", flush=True)
            if not args.fit_check and step % args.ckpt_every == 0:
                acc.wait_for_everyone()   # other ranks wait here instead of inside a collective while rank 0 serializes and uploads
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
    p.add_argument("--keep-last", type=int, default=2, help="rolling checkpoints kept locally; 0 keeps none when --remote-results is set")
    p.add_argument("--keep-remote", type=int, default=2, help="rolling recovery checkpoints kept on --remote-results")
    p.add_argument("--snapshot-every", type=int, default=5000, help="compact bf16 weight snapshot to --remote-results at these validation steps (never pruned)")
    p.add_argument("--action-dropout", type=float, default=0.1, help="fraction of actions replaced by the null id during training (0 disables CFG training)")
    p.add_argument("--require-verified-transitions", action="store_true", help="refuse latents without chain ids and 4-tic spacing")
    p.add_argument("--remote-results", default=None, help="user@host:/dir that receives every checkpoint and log as the copy of record")
    p.add_argument("--fit-check", type=int, default=0, help="run N synthetic steps, report steps/s and memory, exit")
    p.add_argument("--resume", default="", help="checkpoint to resume weights and step from (optimizer state restarts)")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
