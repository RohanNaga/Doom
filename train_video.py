"""
Train the video-pretrained world-model row: SkyReels-V2 Diffusion Forcing 1.3B.

Same trainer skeleton as `train_wm.py` -- its checkpointing, remote copy of record, EMA, skip
policy, update-ratio probes, seeded validation and `log.jsonl` event schema are imported, not
re-implemented -- with the three things this row does differently:

1. The window is a latent VIDEO, (B, 16, L+1, 32, 40), not a channel stack. Context frames are
   the leading frames of the same sequence at their own per-frame timestep, which is what
   diffusion forcing means in this checkpoint.
2. Two objectives. `--objective flow` (default) is the checkpoint's native rectified flow;
   `--objective vp-v` is the shared velocity target of the other three rows on
   `diffusion_v.VDiffusion`'s linear-beta VP schedule. Loss is on the target frame only either
   way.
3. Context corruption is a TIMESTEP, not a bucket. Per window we draw one shared context time
   from U(0, --ctx-noise-max) of the schedule and noise every context frame to it, GameNGen's
   context corruption expressed in the model's own conditioning. The other rows need a separate
   bucket embedding because their context has no timestep of its own; this one does, so there is
   no bucket. `--ctx-noise-max 0` gives an exactly clean context under `flow`; under `vp-v` it
   gives the VP schedule's t=0, which still carries sqrt(beta_0) = 1% noise, the same convention
   the other rows' target uses at t=0. `--ctx-stabilize` fixes that time for inference (the Oasis
   / SkyReels `addnoise_condition` trick, pipeline lines 875-885) and is recorded in the config
   for the evaluator.

Fit check (no data, no checkpoint download, a tiny random transformer):
    python train_video.py --fit-check 20 --tiny --null-prompt none --context-frames 8
Fit check on the real checkpoint:
    python train_video.py --fit-check 20 --context-frames 16 --per-gpu-batch 1 --global-batch 8
Training:
    accelerate launch train_video.py --latents-dir data/latents_wan --split data/split_wan.json \\
        --results-dir results/050-skyreels-l16 --context-frames 16 --lr 2e-5 --warmup 500 \\
        --global-batch 8 --per-gpu-batch 1 --steps 10000
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

from diffusion_v import VDiffusion
from train_wm import REMOTE_SSH, ema_update, prune_remote, save_checkpoint
from video_wm import (FlowMatching, SkyReelsWorldModel, full_finetune_bytes, tiny_config,
                      window_timesteps)

LATENT_CHANNELS = 16
LATENT_HW = (32, 40)


class SyntheticWindows(Dataset):
    """Random windows in the dataset's shape, for `--fit-check` without a latent corpus."""

    def __init__(self, n, context_frames, num_actions):
        self.n, self.L, self.A = n, context_frames, num_actions

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return (torch.randn(self.L, LATENT_CHANNELS, *LATENT_HW, generator=g),
                torch.randn(LATENT_CHANNELS, *LATENT_HW, generator=g),
                torch.randint(0, self.A, (1,), generator=g)[0],
                torch.randint(0, self.A, (self.L,), generator=g))


class SeededCorruption(Dataset):
    """Fix every validation window's randomness to its own index.

    Same contract as `train_wm.SeededCorruption`: the target time, the context time, the context
    noise and the target noise are drawn from a generator seeded by the window index, so the
    validation loss is identical across checkpoints, objectives, batch sizes and world sizes.
    The two uniforms are stored raw and mapped to times by the objective, so `flow` and `vp-v`
    see the same corruption ordering.
    """

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ctx, tgt, act = self.ds[i][:3]
        g = torch.Generator().manual_seed(1234 + int(i))
        return (ctx, tgt, act,
                torch.rand((), generator=g),                              # target time uniform
                torch.rand((), generator=g),                              # context time uniform
                torch.randn(ctx.shape, generator=g, dtype=ctx.dtype),     # context noise
                torch.randn(tgt.shape, generator=g, dtype=tgt.dtype))     # target noise


def build_loaders(args):
    if args.fit_check:
        return SyntheticWindows(args.per_gpu_batch * 64, args.context_frames, args.num_actions), None
    from wan_data import WanWindowDataset, load_split
    split = load_split(args.split)
    train = WanWindowDataset(args.latents_dir, split["train"], args.context_frames,
                             normalize=not args.raw_latents)
    val = WanWindowDataset(args.latents_dir, split["val"], args.context_frames,
                           normalize=not args.raw_latents)
    rng = np.random.RandomState(0)
    idx = np.sort(rng.choice(len(val), size=min(args.val_windows, len(val)), replace=False))
    return train, Subset(val, idx.tolist())


class Objective:
    """Builds the noisy window and the regression target for one objective.

    Returns `(latents, per_frame_timesteps, target)`, where `latents` is (B, 16, L+1, 32, 40)
    with the target frame last, `per_frame_timesteps` is (B, L+1), and `target` is what the model
    must predict for the target frame.
    """

    def __init__(self, kind, flow_shift=1.0, ctx_noise_max=0.3, device="cpu"):
        self.kind = kind
        self.ctx_noise_max = ctx_noise_max
        self.flow = FlowMatching(shift=flow_shift)
        self.vp = VDiffusion(device=device) if kind == "vp-v" else None
        self.num_steps = 1000

    def to(self, device):
        if self.vp is not None:
            self.vp.to(device)
        return self

    def __call__(self, ctx, tgt, u_t=None, u_c=None, ctx_eps=None, tgt_noise=None, ctx_time=None):
        b, L = ctx.shape[0], ctx.shape[1]
        dev = tgt.device
        u_t = torch.rand(b, device=dev) if u_t is None else u_t.to(dev).view(b)
        u_c = torch.rand(b, device=dev) if u_c is None else u_c.to(dev).view(b)
        ctx_eps = torch.randn_like(ctx) if ctx_eps is None else ctx_eps.to(dev)
        tgt_noise = torch.randn_like(tgt) if tgt_noise is None else tgt_noise.to(dev)

        if self.kind == "flow":
            sigma = self.flow.sample_sigma(None, dev, u=u_t)
            sigma_c = (torch.full_like(u_c, ctx_time / self.num_steps) if ctx_time is not None
                       else self.flow.sample_sigma(None, dev, max_sigma=self.ctx_noise_max, u=u_c))
            x_t = self.flow.q_sample(tgt, sigma, tgt_noise)
            target = self.flow.target(tgt, tgt_noise)
            ctx_n = self.flow.q_sample(ctx.flatten(1, 2), sigma_c, ctx_eps.flatten(1, 2)).view_as(ctx)
            t_tgt, t_ctx = self.flow.timestep(sigma), self.flow.timestep(sigma_c)
        else:
            t = (u_t * self.num_steps).long().clamp(max=self.num_steps - 1)
            t_c = (torch.full_like(u_c, float(ctx_time)) if ctx_time is not None
                   else u_c * self.ctx_noise_max * self.num_steps).long().clamp(max=self.num_steps - 1)
            x_t = self.vp.q_sample(tgt, t, tgt_noise)
            target = self.vp.v_target(tgt, t, tgt_noise)
            ctx_n = self.vp.q_sample(ctx.flatten(1, 2), t_c, ctx_eps.flatten(1, 2)).view_as(ctx)
            t_tgt, t_ctx = t.float(), t_c.float()

        # (B, L+1, 16, 32, 40) -> (B, 16, L+1, 32, 40), target last; times in the same frame order
        latents = torch.cat([ctx_n, x_t.unsqueeze(1)], dim=1).transpose(1, 2).contiguous()
        return latents, window_timesteps(t_tgt, t_ctx, L), target


def main(args):
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs, set_seed
    from datetime import timedelta
    acc = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=1))],
                      mixed_precision="bf16", gradient_accumulation_steps=1)
    set_seed(args.seed + acc.process_index)
    device = acc.device
    world = acc.num_processes
    micro_batch = args.per_gpu_batch * world
    assert args.global_batch % micro_batch == 0, \
        f"global batch {args.global_batch} is not a multiple of per-gpu batch x world = {micro_batch}"
    accum = args.global_batch // micro_batch
    is_main = acc.is_main_process

    os.makedirs(args.results_dir, exist_ok=True)
    log_path = os.path.join(args.results_dir, "log.jsonl")

    def log(**kw):
        if is_main:
            kw["time"] = time.time()
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps(kw) + "\n")
            except OSError as e:
                print(f"log write failed: {e}", flush=True)
            print(json.dumps(kw), flush=True)

    model = SkyReelsWorldModel(num_actions=args.num_actions, skyreels_path=args.skyreels_path,
                               null_prompt=args.null_prompt, action_dropout=args.action_dropout,
                               grad_ckpt=args.grad_ckpt, cache_dir=args.hf_cache, fps_id=args.fps_id,
                               tiny=tiny_config() if args.tiny else None)
    start_step = 0
    ck = {}
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.float() for k, v in ck["model"].items()}, strict=True)
        start_step = int(ck.get("step", 0))
        print(f"resumed weights from {args.resume} at step {start_step}"
              + (" with optimizer, scheduler, EMA, and RNG state" if "optimizer" in ck
                 else " (optimizer state reset, warmup restarts)"))
        set_seed(args.seed + acc.process_index + 7919 * start_step)
    n_params = sum(p.numel() for p in model.parameters())
    new_params = model.action_embedder.weight.numel()

    # move explicitly before the optimizer is built: accelerate's prepare did not place the diffusers-loaded
    # model on the fit check of Sep 15 (it trained on the CPU), and a fused AdamW must see CUDA parameters
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                            fused=(device.type == "cuda"))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))

    train_ds, val_ds = build_loaders(args)
    loader = DataLoader(train_ds, batch_size=args.per_gpu_batch, shuffle=True, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
    model, opt, loader = acc.prepare(model, opt, loader)   # scheduler stays unwrapped
    objective = Objective(args.objective, args.flow_shift, args.ctx_noise_max, device=device).to(device)
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
        split_hash = (hashlib.md5(open(args.split, "rb").read()).hexdigest()
                      if (not args.fit_check and os.path.exists(args.split)) else None)
        with open(os.path.join(args.results_dir, "config.json"), "w") as f:
            json.dump({**vars(args), "git": git, "params": n_params, "new_params": new_params,
                       "world_size": world, "accum": accum, "split_md5": split_hash,
                       "tokens_per_window": raw.tokens_per_window(args.context_frames),
                       "optimizer_state_bytes": full_finetune_bytes(n_params),
                       "torch": torch.__version__}, f, indent=1)
    log(event="start", backbone="skyreels-df-1.3b", objective=args.objective, params=n_params, device=str(next(raw.parameters()).device),
        new_params=new_params, world=world, accum=accum, per_gpu_batch=args.per_gpu_batch,
        global_batch=args.per_gpu_batch * world * accum, context_frames=args.context_frames,
        tokens_per_window=raw.tokens_per_window(args.context_frames))

    def step_loss(ctx, tgt, act, **rand):
        latents, times, target = objective(ctx, tgt, **rand)
        pred = model(latents, times, act)
        return ((pred.float() - target.float()) ** 2).flatten(1).mean(1)

    @torch.no_grad()
    def evaluate():
        if val_ds is None:
            return None
        model.eval()
        vl = DataLoader(SeededCorruption(val_ds), batch_size=args.per_gpu_batch, shuffle=False, num_workers=2)
        tot = torch.zeros((), device=device); n = torch.zeros((), device=device)
        bins = torch.zeros(4, device=device); bin_n = torch.zeros(4, device=device)
        for ctx, tgt, act, u_t, u_c, ctx_eps, tgt_noise in vl:
            ctx, tgt, act = ctx.to(device), tgt.to(device), act.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_autocast):
                loss = step_loss(ctx, tgt, act, u_t=u_t, u_c=u_c, ctx_eps=ctx_eps, tgt_noise=tgt_noise)
            tot += loss.detach().sum(); n += loss.numel()
            q = (u_t.to(device) * 4).long().clamp(max=3)
            bins.index_add_(0, q, loss.detach().float()); bin_n.index_add_(0, q, torch.ones_like(loss).float())
        tot, n = acc.reduce(tot, reduction="sum"), acc.reduce(n, reduction="sum")
        bins, bin_n = acc.reduce(bins, reduction="sum"), acc.reduce(bin_n, reduction="sum")
        model.train()
        return (tot / n).item(), (bins / bin_n.clamp(min=1)).tolist()

    model.train()
    step, t0 = start_step, time.time()
    running, grad_norms = [], []
    skipped = int(ck.get("skipped", 0)) if args.resume else 0
    # probe tensors: patch projection, output projection, the action table, and the adaLN
    # modulation table of the first, middle and last block
    last = len(raw.transformer.blocks) - 1
    want = ["patch_embedding.weight", "proj_out.weight", "action_embedder.weight",
            "blocks.0.scale_shift_table", f"blocks.{last // 2}.scale_shift_table",
            f"blocks.{last}.scale_shift_table"]
    probe_params = [(n, p) for n, p in raw.named_parameters() if any(n.endswith(w) for w in want)]
    probe_prev = {n: torch.empty_like(p.detach()) for n, p in probe_params}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    done = False
    max_steps = args.fit_check if args.fit_check else args.steps
    best_val = float(ck.get("best_val", float("inf"))) if args.resume else float("inf")
    val_hist = []
    micro = int(ck.get("micro", 0)) if args.resume else 0
    while not done:
        for batch in loader:
            ctx, tgt, act = batch[0], batch[1], batch[2]
            micro += 1
            ctx, tgt, act = ctx.to(device, non_blocking=True), tgt.to(device, non_blocking=True), act.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_autocast):
                loss = step_loss(ctx, tgt, act).mean() / accum
            acc.backward(loss)
            running.append(loss.item() * accum)
            if micro % accum != 0:
                continue
            gn = acc.clip_grad_norm_(model.parameters(), args.clip if args.clip > 0 else float("inf"))
            grad_norms.append(float(gn))
            if not math.isfinite(float(gn)):
                opt.zero_grad(set_to_none=True); skipped += 1
                log(event="skipped_update", step=step, micro=micro, grad_norm=float(gn), skipped_total=skipped)
                if skipped > 20:
                    raise RuntimeError(f"{skipped} non-finite gradient updates; stopping before Adam state is corrupted")
                continue
            measure = bool(probe_params) and (step + 1) % 100 == 0
            if measure:
                with torch.no_grad():
                    for n, p in probe_params:
                        probe_prev[n].copy_(p.detach())
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if measure:
                with torch.no_grad():
                    ratios = {n: float((p.detach() - probe_prev[n]).norm() / (probe_prev[n].norm() + 1e-12))
                              for n, p in probe_params}
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
                    save_checkpoint({"model": {k: t.detach().cpu().to(torch.bfloat16) for k, t in raw.state_dict().items()},
                                     "ema": {k: t.to(torch.bfloat16) for k, t in zip(raw.state_dict().keys(), ema)} if ema is not None else None,
                                     "step": step, "val_loss": v, "args": vars(args)},
                                    os.path.join(args.results_dir, f"snap_{step:07d}.pt"), args.remote_results,
                                    keep_local=False)
                if is_main and v < best_val:
                    best_val = v
                    save_checkpoint({"model": {k: t.detach().cpu().to(torch.bfloat16) for k, t in raw.state_dict().items()},
                                     "step": step, "val_loss": v, "args": vars(args)},
                                    os.path.join(args.results_dir, "best.pt"), args.remote_results, keep_local=True)
            if not args.fit_check and is_main and step % args.ckpt_every == 0:
                olds = sorted(p for p in os.listdir(args.results_dir) if p.endswith(".pt") and p[0].isdigit())
                for p in (olds[:-max(0, args.keep_last - 1)] if args.keep_last > 0 else olds):
                    os.remove(os.path.join(args.results_dir, p))
                ck = {"model": {k: t.detach().cpu().float() for k, t in raw.state_dict().items()},
                      "step": step, "args": vars(args), "optimizer": opt.state_dict(),
                      "scheduler": sched.state_dict(), "best_val": best_val, "micro": micro, "skipped": skipped,
                      "rng": {"cpu": torch.get_rng_state(),
                              "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                              "numpy": np.random.get_state()}}
                if ema is not None:
                    ck["ema"] = {k: t.clone() for k, t in zip(raw.state_dict().keys(), ema)}
                written = save_checkpoint(ck, os.path.join(args.results_dir, f"{step:07d}.pt"),
                                          args.remote_results, keep_local=args.keep_last > 0)
                if args.remote_results:
                    try:
                        if any(w.startswith(args.remote_results.split(":", 1)[1]) for w in written):
                            prune_remote(args.remote_results, os.path.basename(args.results_dir.rstrip("/")),
                                         args.keep_remote)
                        subprocess.run(["rsync", "-a", "-e", " ".join(REMOTE_SSH), "--include=*.json",
                                        "--include=*.jsonl", "--exclude=*", args.results_dir + "/",
                                        args.remote_results.rstrip("/") + "/"
                                        + os.path.basename(args.results_dir.rstrip("/")) + "/"],
                                       capture_output=True, timeout=600)
                    except (subprocess.TimeoutExpired, OSError) as e:
                        print(f"remote housekeeping failed: {e}", flush=True)
            if not args.fit_check and step % args.ckpt_every == 0:
                acc.wait_for_everyone()
            if step >= max_steps:
                done = True; break
    if args.fit_check:
        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
        log(event="fit_check", backbone="skyreels-df-1.3b", objective=args.objective,
            context_frames=args.context_frames, per_gpu_batch=args.per_gpu_batch, world=world,
            steps=step, steps_per_s=step / dt, peak_mem_gb=round(mem, 2), params=n_params,
            tokens_per_window=raw.tokens_per_window(args.context_frames))
    acc.wait_for_everyone()
    log(event="end", step=step)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # shared with train_wm.py, same meaning
    p.add_argument("--context-frames", type=int, default=16)
    p.add_argument("--num-actions", type=int, default=29)
    p.add_argument("--latents-dir", default="data/latents_wan")
    p.add_argument("--split", default="data/split_wan.json")
    p.add_argument("--results-dir", default="results/fit_check_video")
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--global-batch", type=int, default=8)
    p.add_argument("--per-gpu-batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-dropout", type=float, default=0.0,
                   help="fraction of actions replaced by the null id (the corrected pair used 0)")
    p.add_argument("--ema-every", type=int, default=8, help="0 disables the fp32 CPU EMA")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-windows", type=int, default=512)
    p.add_argument("--ckpt-every", type=int, default=1000,
                   help="a recovery checkpoint here is 10.70 GiB (fp32 weights + fp32 EMA), measured")
    p.add_argument("--snapshot-every", type=int, default=5000,
                   help="bf16 weight+EMA snapshot to --remote-results; 5.35 GiB each and never "
                        "pruned, so 1000 would cost 53 GiB over a 10k-step run")
    p.add_argument("--keep-last", type=int, default=2)
    p.add_argument("--keep-remote", type=int, default=2)
    p.add_argument("--remote-results", default=None,
                   help="user@host:/dir that receives every checkpoint and log as the copy of record")
    p.add_argument("--fit-check", type=int, default=0, help="run N synthetic steps, report steps/s and memory, exit")
    p.add_argument("--resume", default="", help="recovery checkpoint to resume from")
    p.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true", default=True,
                   help="recompute activations in the backward pass; ON by default for this 1.3B row")
    p.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    # this row only
    p.add_argument("--skyreels-path", default="Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers")
    p.add_argument("--null-prompt", default="weights/skyreels_null_prompt.pt",
                   help="precomputed UMT5 null-prompt embedding from make_null_prompt.py; "
                        "'none' substitutes zeros and is only valid for shape gates")
    p.add_argument("--objective", choices=["flow", "vp-v"], default="flow",
                   help="flow: the checkpoint's native rectified flow. vp-v: the shared velocity "
                        "target on diffusion_v's linear-beta VP schedule")
    p.add_argument("--flow-shift", type=float, default=1.0,
                   help="the scheduler's flow shift; the stored config has 1.0, the DF pipeline "
                        "docstring recommends 8.0 for T2V at inference")
    p.add_argument("--ctx-noise-max", type=float, default=0.3,
                   help="context frames share a time drawn from U(0, this) of the schedule. 0 gives "
                        "an exactly clean context under 'flow' (sigma 0); under 'vp-v' it gives the "
                        "VP schedule's own t=0, which still carries sqrt(beta_0) = 1%% noise")
    p.add_argument("--ctx-stabilize", type=float, default=20.0,
                   help="fixed context timestep for inference (SkyReels addnoise_condition / Oasis); "
                        "recorded in config.json for the evaluator, not used in training")
    p.add_argument("--fps-id", type=int, default=1, choices=[0, 1],
                   help="the checkpoint's sample-info id: 0 means 16 fps, 1 means anything else")
    p.add_argument("--raw-latents", action="store_true",
                   help="train in raw Wan encoder space instead of the checkpoint's normalised space")
    p.add_argument("--tiny", action="store_true",
                   help="build a tiny randomly-initialised transformer with the same code path (gates only)")
    p.add_argument("--no-autocast", action="store_true", help="run in fp32 (CPU gates)")
    main(p.parse_args())
