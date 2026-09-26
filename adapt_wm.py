"""
Adapt a pretrained world model to ONE unseen map with a LoRA adapter on the frozen backbone.

The adaptation study (`docs/lora_adaptation_design_2026-09-26.html`): start from a pretrained snapshot, train
a low-rank adapter on the attention projections (`lora.py`) plus a few small parts in full, on a few episodes
of one unseen map (`adapt_split.py`), and save the adapter at a grid of update counts so every point of the
adaptation curve is scored on the same held-out windows (`score_adapt.py`).

    python adapt_wm.py --source $R/040-unet-nexttic/snap_0200000.pt \\
        --adapt-split results/adapt_splits/split_adapt_unseen_map17_seed0.json \\
        --latents-dir $E/latents_arnold_eval_pertic/unseen --hf-cache $HF \\
        --results-dir $R/adapt/unet200k_unseen_map17_s0 --per-gpu-batch 8 --grad-ckpt --allow-accumulation

**Nothing changes but the adapter.** Every recipe setting that defines what the model learns is read from the
SOURCE snapshot's own args, never from this command line: the backbone and its graph, the 32-tic context,
the executed-control history, the objective, the context-noise augmentation level and its buckets, the action
dropout. The loop is the pretraining loop's (`train_wm.py`): the same `noise_augment` call, the same
`VDiffusion.training_loss`, bf16 autocast, gradient clipping at `--clip`, the same non-finite skip, the same
linear warmup through `LambdaLR`, fused AdamW, and the same `SeededCorruption` validation read. It is a
separate script so that `train_wm.py`, which the finished rows were certified against, stays byte-identical.

**The source weights.** `--source-weights ema` (the default) loads the snapshot's EMA tensors INTO THE MODEL
with `doomdit_utils.load_world_model_state`, the loader every zero-shot score was produced with, so the
adapted model starts from exactly the weights the zero-shot scores measured (`train_wm.py --init-from` loads
the live weights and uses the EMA only to seed the EMA copy, which is the wrong start here). At launch the
adapter's output is compared with the frozen model's on `--parity-windows` held-out windows under the same
corruption; with `lora_B` at zero they are identical, the difference is printed in the certificate line, and
anything above `--parity-tol` (default 0, exact) stops the run before it trains.

**What is new.** The LoRA factors and the `--full-parts` train (lr 1e-4 constant after a 100-update warmup,
no weight decay); everything else is frozen. `--lora-rank 0 --full-parts all` is the full fine-tune reference
(design decision 5): no adapter, every parameter trained, the same loop, checkpoints and evaluator. An fp32
EMA of exactly the trained tensors (decay `--ema-decay`, 0.999 by default for runs of a few thousand updates)
lives on the training device, updated every update.
Windows come only from the first k episodes of the split's adapt list: its step-curve rung (8 by default) or
the data-ladder rung `--adapt-episodes-k` names; the live validation curve is the v-loss on the split's
recorded held-out windows.

**Checkpoints** at every step of `--step-grid` (default 0,250,500,1000,2000,4000, log-spaced; 0 is written
before the first update) as `adapter_<step>.pt`: the adapter and the trained parts, live and EMA, in fp32, how
to rebuild them (`adapter_config`), the source snapshot's path, SHA-256 and weights choice, the source run's
args (so every evaluator rebuilds the source graph unchanged), and the certificate. No backbone weight is
written.

**Batch.** `--global-batch` (default 32, the pretraining batch) and `--per-gpu-batch` (the micro-batch) are
separate. Filling the card is the rule (CLAUDE.md): the run is refused unless micro-batch x processes equals
the global batch, and `--allow-accumulation` is the explicit override that reaches it by gradient
accumulation instead. `--grad-ckpt` recomputes activations for the 16 GB cards. The effective batch is in
the certificate line. `--fit-check N` runs N updates (on synthetic windows when no split is given), prints
steps/s and peak memory, and writes nothing else.

**Live curves.** Every event goes to `<results-dir>/log.jsonl` and, as in pretraining, to the W&B run named
after the results directory (project doomdit-nexttic), in group `adapt_<set>_map<NN>` so one map's runs
overlay; `--no-wandb` opts out and a fit check never streams.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import lora
from adapt_split import KIND, WINDOW_KEY, adapt_episodes, git_state, sha256_bytes, windows_in_dataset
from backbones import build_model, resolve_latent_channels
from diffusion_v import VDiffusion, noise_augment
from train_wm import SeededCorruption, SyntheticWindows, save_checkpoint, unpack_batch
from wandb_log import DEFAULT_PROJECT, RunLogger

DEFAULT_GRID = "0,250,500,1000,2000,4000"       # log-spaced until the curve flattens
CKPT_PREFIX = "adapter_"


def parse_grid(spec):
    """"0,250,500" -> [0, 250, 500]: sorted, unique, non-negative, with at least one update after 0."""
    try:
        steps = sorted({int(x) for x in str(spec).split(",") if x.strip()})
    except ValueError:
        raise SystemExit(f"--step-grid takes comma-separated update counts, got {spec!r}")
    if not steps or steps[0] < 0 or steps[-1] < 1:
        raise SystemExit(f"--step-grid {spec!r} needs non-negative update counts and one above 0")
    return steps


def resolve_batch(global_batch, per_gpu_batch, world, allow_accumulation=False):
    """Gradient-accumulation factor for (global batch, micro-batch, processes); refuses what the rule forbids."""
    micro = int(per_gpu_batch) * int(world)
    if micro <= 0 or global_batch % micro:
        raise SystemExit(f"a micro-batch of {per_gpu_batch} on {world} process(es) cannot reach the global batch "
                         f"of {global_batch}")
    accum = global_batch // micro
    if accum > 1 and not allow_accumulation:
        raise SystemExit(f"--per-gpu-batch {per_gpu_batch} on {world} process(es) is a batch of {micro}, so the "
                         f"global batch of {global_batch} needs gradient accumulation of {accum}. The fill-the-card "
                         "rule (CLAUDE.md) forbids accumulation by default: use the largest micro-batch that fits "
                         "(with --grad-ckpt if needed), more processes, a smaller --global-batch stated as a "
                         "deviation, or --allow-accumulation to accumulate deliberately.")
    return accum


def source_recipe(src_args):
    """What the adapted model is and learns, taken from the source run's own args (never from this CLI)."""
    a = src_args or {}
    if not a.get("backbone"):
        raise SystemExit("the source checkpoint records no args; adapt from a train_wm.py snap_*.pt or recovery "
                         "checkpoint")
    phase = int(a.get("phase_buckets", 0) or 0) if a.get("phase_conditioning") else 0
    return {"backbone": a["backbone"],
            "latent_channels": resolve_latent_channels(a["backbone"], a.get("resolved_latent_channels")
                                                       or a.get("latent_channels")),
            "context_frames": int(a.get("context_frames", 32)), "num_actions": int(a.get("num_actions", 29)),
            "noise_buckets": int(a.get("noise_buckets", 10)), "noise_aug_max": float(a.get("noise_aug_max", 0.7)),
            "objective": a.get("objective") or "v", "tic_stride": int(a.get("tic_stride", 4)),
            "action_history": int(a.get("action_history", 0) or 0),
            "control_bits": int(a.get("resolved_control_bits") or a.get("control_bits") or 0),
            "phase_buckets": phase, "action_inject": a.get("action_inject") or "token",
            "action_dropout": float(a.get("action_dropout", 0.1)), "warm_start": a.get("warm_start"),
            "hf_cache": a.get("hf_cache")}


def build_source_model(recipe, grad_ckpt=False, warm_start=None, hf_cache=None):
    """The source run's graph, built as the evaluators build it; its weights come from the snapshot next."""
    return build_model(recipe["backbone"], recipe["num_actions"], recipe["context_frames"], recipe["noise_buckets"],
                       grad_ckpt=grad_ckpt, warm_start=warm_start or recipe["warm_start"],
                       cache_dir=hf_cache or recipe["hf_cache"], action_dropout=recipe["action_dropout"],
                       latent_channels=recipe["latent_channels"], action_inject=recipe["action_inject"],
                       phase_buckets=recipe["phase_buckets"], action_history=recipe["action_history"],
                       control_bits=recipe["control_bits"])


@torch.no_grad()
def ema_update(ema, named, decay):
    """fp32 EMA of the trained tensors, in place: e <- decay * e + (1 - decay) * p (bf16 would underflow)."""
    for n, p in named:
        ema[n].mul_(decay).add_(p.detach().float(), alpha=1.0 - decay)


def window_datasets(recipe, latents_dir, train_ids, held_ids, windows):
    """(training windows over the adaptation episodes, the recorded held-out windows as a Subset)."""
    from doom_data import TicWindowDataset

    def make(ids):
        return TicWindowDataset(latents_dir, ids, recipe["context_frames"], latent_channels=recipe["latent_channels"],
                                with_phase=bool(recipe["phase_buckets"]),
                                phase_buckets=recipe["phase_buckets"] or 5, action_history=recipe["action_history"])
    train, held = make(train_ids), make(held_ids)
    return train, Subset(held, windows_in_dataset(held, windows).tolist())


def corrupted_batch(ds, n, noise_aug_max, num_steps):
    """The first `n` windows of `ds` under `SeededCorruption`'s fixed per-window corruption, stacked."""
    sc = SeededCorruption(ds, noise_aug_max, num_steps)
    items = [sc[i] for i in range(min(n, len(sc)))]
    return [torch.stack([torch.as_tensor(it[j]) for it in items]) for j in range(len(items[0]))]


@torch.no_grad()
def predict(model, diffusion, batch, recipe, device):
    """The model's prediction on a `corrupted_batch`, exactly as validation computes it."""
    ctx, tgt, act, t, level, ctx_eps, tgt_noise = (x.to(device) for x in batch[:7])
    phase = batch[7].to(device) if len(batch) > 7 else None
    ctx_n, bucket = noise_augment(ctx, recipe["noise_aug_max"], recipe["noise_buckets"], level=level, eps=ctx_eps)
    xt = diffusion.q_sample(tgt, t, tgt_noise)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(xt, t, act, ctx_n, bucket) if phase is None else model(xt, t, act, ctx_n, bucket, phase)
    return out.float()


def certificate_line(c):
    """The one greppable line the certificate becomes: provenance, batch, adapter and the step-0 parity."""
    g = c["git"]
    commit = f"{g['commit']}{'+dirty' if g.get('dirty') else ''}"
    fields = [("git", commit), ("seed", c["seed"]), ("map", c["map"]), ("source", c["source"]["path"]),
              ("source_sha256", c["source"]["sha256"]), ("source_step", c["source"]["step"]),
              ("source_weights", c["source"]["weights"]), ("split", c["split"]["path"]),
              ("split_sha256", c["split"]["sha256"]), ("adapt_episodes", len(c["adapt_episodes"])),
              ("held_out_episodes", len(c["held_out"])), ("held_out_windows", c["held_out_windows"]),
              ("micro_batch", c["micro_batch"]), ("world", c["world"]), ("accum", c["accum"]),
              ("effective_batch", c["effective_batch"]), ("lr", f"{c['lr']:g}"), ("warmup", c["warmup"]),
              ("ema_decay", f"{c['ema_decay']:g}"), ("rank", c["rank"]), ("alpha", f"{c['alpha']:g}"),
              ("dropout", f"{c['dropout']:g}"), ("lora_mlp", int(c["lora_mlp"])),
              ("parts", ",".join(c["parts"]) or "none"), ("trainable", c["trainable"]),
              ("grid", ",".join(str(s) for s in c["step_grid"])),
              ("step0_parity_max_abs", f"{c['step0_parity_max_abs']:.3g}"),
              ("step0_parity_windows", c["step0_parity_windows"])]
    return "ADAPT_CERTIFICATE " + " ".join(f"{k}={v}" for k, v in fields)


def main(args):
    """Adapt the source snapshot to one map and write the adapter at every step of the grid."""
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs, set_seed
    from datetime import timedelta
    from eval_identity import sha256_file
    # the same Accelerator settings as train_wm.py: AcceleratorState is a process-wide singleton
    acc = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=1))], mixed_precision="bf16",
                      gradient_accumulation_steps=1)
    set_seed(args.seed + acc.process_index)
    device, world, is_main = acc.device, acc.num_processes, acc.is_main_process
    grid = parse_grid(args.step_grid)
    accum = resolve_batch(args.global_batch, args.per_gpu_batch, world, args.allow_accumulation)
    parts = lora.parse_parts(args.full_parts)
    max_steps = args.fit_check or grid[-1]
    os.makedirs(args.results_dir, exist_ok=True)
    if not args.fit_check and glob.glob(os.path.join(args.results_dir, CKPT_PREFIX + "*.pt")):
        raise SystemExit(f"{args.results_dir} already holds adapter checkpoints; an adaptation is never resumed or "
                         "mixed with another, so pick a new --results-dir")
    log_path = os.path.join(args.results_dir, "log.jsonl")
    wb = RunLogger()

    def log(**kw):
        if is_main:
            kw["time"] = time.time()
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps(kw) + "\n")
            except OSError as e:
                print(f"log write failed: {e}", flush=True)
            print(json.dumps(kw), flush=True)
            wb.log_event(kw)

    # --- the source: its graph from its own args, its EMA (or live) weights -------------------------------
    src_ck = torch.load(args.source, map_location="cpu", weights_only=False)
    recipe = source_recipe(src_ck.get("args"))
    lora.check_backbone(recipe["backbone"])
    if recipe["tic_stride"] != 1:
        raise SystemExit("the unseen-map corpora are per-tic; a stride-4 source cannot be adapted on them")
    model = build_source_model(recipe, args.grad_ckpt, args.warm_start, args.hf_cache)
    lora.load_source_weights(model, src_ck, args.source_weights)
    source = {"path": os.path.abspath(args.source), "weights": args.source_weights, "step": src_ck.get("step"),
              "sha256": sha256_file(args.source)}
    source_args = dict(src_ck["args"])
    del src_ck
    diffusion = VDiffusion(device=device, objective=recipe["objective"])

    # --- the data: adaptation episodes only; the split's recorded held-out windows ------------------------
    split_rec, train_ids, held_ids, corpus, set_name, map_id = None, None, None, "synthetic", None, None
    if args.adapt_split:
        with open(args.adapt_split) as f:
            split = json.load(f)
        meta = split.get("meta", {})
        if meta.get("kind") != KIND:
            raise SystemExit(f"{args.adapt_split} is not an adaptation split (adapt_split.py)")
        if int(meta.get("context_frames", recipe["context_frames"])) != recipe["context_frames"]:
            raise SystemExit(f"the split's windows were drawn for context {meta.get('context_frames')}, the source "
                             f"trains on {recipe['context_frames']}")
        train_ids, held_ids = adapt_episodes(split, args.adapt_episodes_k), list(split["held_out"])
        if set(train_ids) & set(held_ids):
            raise SystemExit("the adaptation and held-out episodes overlap; the split file is corrupt")
        corpus, set_name, map_id = meta.get("corpus"), meta.get("set"), meta.get("map")
        split_rec = {"path": os.path.abspath(args.adapt_split), "sha256": sha256_bytes(args.adapt_split)}
        latents_dir = args.latents_dir or meta.get("latents_dir")
        if not latents_dir:
            raise SystemExit("--latents-dir names the map's latents in the source's latent space")
        train_ds, val_ds = window_datasets(recipe, latents_dir, train_ids, held_ids, split[WINDOW_KEY])
    elif args.fit_check:
        n = args.per_gpu_batch * 64
        train_ds = SyntheticWindows(n, recipe["context_frames"], recipe["num_actions"], recipe["latent_channels"],
                                    recipe["action_history"], recipe["control_bits"], recipe["phase_buckets"])
        val_ds = SyntheticWindows(max(args.parity_windows, 1), recipe["context_frames"], recipe["num_actions"],
                                  recipe["latent_channels"], recipe["action_history"], recipe["control_bits"],
                                  recipe["phase_buckets"])
    else:
        raise SystemExit("--adapt-split names the map to adapt to (only a --fit-check runs without one)")

    # --- the adapter, and the step-0 parity against the frozen model --------------------------------------
    model.to(device).eval()
    parity_batch = corrupted_batch(val_ds, args.parity_windows, recipe["noise_aug_max"], diffusion.num_steps)
    frozen_out = predict(model, diffusion, parity_batch, recipe, device)
    targets = (lora.inject_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout, args.lora_mlp, args.seed)
               if args.lora_rank > 0 else [])
    parity = float((predict(model, diffusion, parity_batch, recipe, device) - frozen_out).abs().max())
    if not parity <= args.parity_tol:
        raise SystemExit(f"step-0 parity failed: the adapter at initialisation moves the prediction by {parity:.3g} "
                         f"(tolerance {args.parity_tol:g}); step 0 would not be the frozen model")
    names = lora.configure_trainable(model, recipe["backbone"], parts)
    counts = lora.parameter_counts(model, recipe["backbone"], parts)
    cfg = lora.adapter_config(recipe["backbone"], args.lora_rank, args.lora_alpha, args.lora_dropout, args.lora_mlp,
                              parts, args.seed, targets)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    ema = {n: p.detach().float().clone() for n, p in named}
    trainable = [p for _, p in named]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd, fused=(device.type == "cuda"))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    extra = {"prefetch_factor": args.prefetch_factor} if args.num_workers > 0 else {}
    loader = DataLoader(train_ds, batch_size=args.per_gpu_batch, shuffle=True, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0, **extra)
    model.train()
    model, opt, loader = acc.prepare(model, opt, loader)
    raw = acc.unwrap_model(model)
    ema_named = [(n, p) for n, p in raw.named_parameters() if n in ema]     # the same Parameter objects

    cert = {"git": git_state(), "seed": args.seed, "map": corpus, "set": set_name, "map_id": map_id,
            "source": source, "split": split_rec or {"path": None, "sha256": None},
            "adapt_episodes": train_ids or [], "adapt_episodes_k": len(train_ids or []), "held_out": held_ids or [],
            "held_out_windows": len(val_ds), "micro_batch": args.per_gpu_batch, "world": world, "accum": accum,
            "effective_batch": args.per_gpu_batch * world * accum, "lr": args.lr, "warmup": args.warmup,
            "ema_decay": args.ema_decay, "rank": args.lora_rank, "alpha": args.lora_alpha,
            "dropout": args.lora_dropout, "lora_mlp": bool(args.lora_mlp), "parts": list(parts),
            "trainable": counts["trainable"], "step_grid": grid, "step0_parity_max_abs": parity,
            "step0_parity_windows": len(parity_batch[0]), "recipe": recipe, "counts": counts, "args": vars(args)}
    line = certificate_line(cert)
    run = os.path.basename(os.path.normpath(args.results_dir))
    group = args.wandb_group or (f"adapt_{corpus}" if corpus else "adapt")
    if is_main:
        cfg_json = {**vars(args), "certificate": cert, "certificate_line": line, "wandb_run": None, "wandb_group": group}
        if args.wandb and not args.fit_check:
            cfg_json["wandb_run"] = run
        if not args.fit_check:
            with open(os.path.join(args.results_dir, "config.json"), "w") as f:
                json.dump(cfg_json, f, indent=1, default=str)
        # opened before the loader forks its workers (wandb_log.py, "Starts no subprocess on its own thread")
        wb = RunLogger(enabled=bool(cfg_json["wandb_run"]), name=run, project=args.wandb_project,
                       entity=args.wandb_entity, config=cfg_json, results_dir=args.results_dir, group=group)
        print(line, flush=True)
    log(event="start", backbone=recipe["backbone"], map=corpus, world=world, accum=accum,
        per_gpu_batch=args.per_gpu_batch, global_batch=args.per_gpu_batch * world * accum,
        dataset_summary=getattr(train_ds, "summary", None), params=counts, step_grid=grid,
        adapt_episodes=train_ids, held_out=held_ids)
    log(event="certificate", line=line, step0_parity_max_abs=parity)

    def model_fn(ctx, act, bucket, phase=None):
        if phase is None:
            return lambda xt, t: model(xt, t, act, ctx, bucket)
        return lambda xt, t: model(xt, t, act, ctx, bucket, phase)

    @torch.no_grad()
    def evaluate():
        """v-loss on the recorded held-out windows under SeededCorruption, overall and by t-quartile."""
        model.eval()
        vl = DataLoader(SeededCorruption(val_ds, recipe["noise_aug_max"], diffusion.num_steps),
                        batch_size=args.per_gpu_batch, shuffle=False, num_workers=0)
        tot = torch.zeros((), device=device); n = torch.zeros((), device=device)
        bins = torch.zeros(4, device=device); bin_n = torch.zeros(4, device=device)
        for batch in vl:
            ctx, tgt, act, t, level, ctx_eps, tgt_noise = (x.to(device) for x in batch[:7])
            phase = batch[7].to(device) if len(batch) > 7 else None
            ctx_n, bucket = noise_augment(ctx, recipe["noise_aug_max"], recipe["noise_buckets"], level=level, eps=ctx_eps)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = diffusion.training_loss(model_fn(ctx_n, act, bucket, phase), tgt, noise=tgt_noise, t=t,
                                               per_sample=True)
            tot += loss.sum(); n += loss.numel()
            q = (t * 4) // diffusion.num_steps
            bins.index_add_(0, q, loss); bin_n.index_add_(0, q, torch.ones_like(loss))
        tot, n = acc.reduce(tot, reduction="sum"), acc.reduce(n, reduction="sum")
        bins, bin_n = acc.reduce(bins, reduction="sum"), acc.reduce(bin_n, reduction="sum")
        model.train()
        return (tot / n).item(), (bins / bin_n.clamp(min=1)).tolist()

    def checkpoint(step, val_loss):
        if is_main:
            ck = lora.adapter_checkpoint(raw, names, ema, cfg, source, source_args, step,
                                         adapt_args=vars(args), certificate=cert, certificate_line=line,
                                         map=corpus, split=split_rec, episodes=train_ids, held_out=held_ids,
                                         counts=counts, recipe=recipe, val_loss=val_loss)
            path = os.path.join(args.results_dir, f"{CKPT_PREFIX}{step:07d}.pt")
            save_checkpoint(ck, path)
            log(event="checkpoint", step=step, path=path, val_loss=val_loss)
        acc.wait_for_everyone()

    def validate(step):
        v, vbins = evaluate()
        log(event="val", step=step, val_loss=v, val_loss_by_t_quartile=vbins, excursion=False)
        return v

    if not args.fit_check and 0 in grid:
        checkpoint(0, validate(0))
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    step, micro, skipped = 0, 0, 0
    running, grad_norms = [], []
    t0 = time.time()
    while step < max_steps:
        for batch in loader:
            micro += 1
            ctx, tgt, act, phase = unpack_batch(batch)
            ctx, tgt, act = ctx.to(device, non_blocking=True), tgt.to(device, non_blocking=True), act.to(device)
            phase = None if phase is None else phase.to(device)
            ctx_n, bucket = noise_augment(ctx, recipe["noise_aug_max"], recipe["noise_buckets"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = diffusion.training_loss(model_fn(ctx_n, act, bucket, phase), tgt) / accum
            acc.backward(loss)
            running.append(loss.item() * accum)
            if micro % accum != 0:
                continue
            gn = acc.clip_grad_norm_(trainable, args.clip if args.clip > 0 else float("inf"))
            grad_norms.append(float(gn))
            if not math.isfinite(float(gn)):
                opt.zero_grad(set_to_none=True); skipped += 1
                log(event="skipped_update", step=step, micro=micro, grad_norm=float(gn), skipped_total=skipped)
                if skipped > 20:
                    raise RuntimeError(f"{skipped} non-finite gradient updates; stopping before Adam state is corrupted")
                continue
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            ema_update(ema, ema_named, args.ema_decay)
            if step % args.log_every == 0 or step == max_steps:
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
                res = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0
                log(event="train", step=step, loss=float(np.mean(running)), lr=sched.get_last_lr()[0],
                    steps_per_s=step / dt, peak_mem_gb=round(mem, 2), peak_reserved_gb=round(res, 2),
                    grad_norm=float(np.mean(grad_norms)), grad_norm_max=float(np.max(grad_norms)),
                    clip_frac=float(np.mean([g > args.clip for g in grad_norms])) if args.clip > 0 else 0.0,
                    nonfinite_loss=int(sum(not np.isfinite(x) for x in running)), skipped_updates=skipped)
                running, grad_norms = [], []
            if not args.fit_check and (step in grid or (args.val_every and step % args.val_every == 0)):
                v = validate(step)
                if step in grid:
                    checkpoint(step, v)
            if step >= max_steps:
                break
    if args.fit_check:
        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
        res = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0
        log(event="fit_check", backbone=recipe["backbone"], per_gpu_batch=args.per_gpu_batch, world=world, accum=accum,
            global_batch=args.per_gpu_batch * world * accum, grad_ckpt=bool(args.grad_ckpt), steps=step,
            steps_per_s=step / dt, peak_mem_gb=round(mem, 2), peak_reserved_gb=round(res, 2),
            trainable=counts["trainable"], data="synthetic" if not args.adapt_split else "map")
    acc.wait_for_everyone()
    log(event="end", step=step)
    return 0


def build_parser():
    """Every adaptation flag; recipe settings are deliberately absent (they come from the source)."""
    p = argparse.ArgumentParser(description="LoRA adaptation of a pretrained world model to one unseen map.")
    p.add_argument("--source", required=True, help="the pretrained snapshot (train_wm.py snap_*.pt or recovery "
                   "checkpoint); its args define the model and the recipe")
    p.add_argument("--source-weights", choices=["ema", "live"], default="ema",
                   help="which of the snapshot's weights the frozen backbone and the parts start from; ema (default) is "
                        "what every zero-shot score used")
    p.add_argument("--adapt-split", default="", help="the map's adaptation split (adapt_split.py)")
    p.add_argument("--adapt-episodes-k", type=int, default=0,
                   help="data ladder: train on the first K episodes of the split's adapt list (0 = the split's "
                        "step-curve rung, 8 by default)")
    p.add_argument("--latents-dir", default="", help="the map's per-tic latent directory, in the source's latent "
                   "space (default: the one the split was written from, which is the SD 1.x space)")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--lora-rank", type=int, default=16,
                   help="0 injects no adapter; with --full-parts all that is the full fine-tune reference (design "
                        "decision 5), whose checkpoints hold every backbone tensor in fp32, live and EMA")
    p.add_argument("--lora-alpha", type=float, default=16.0, help="the LoRA scale is alpha / rank")
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lora-mlp", action="store_true",
                   help="also adapt the feed-forward projections (the DiT-recipe precedent); off by default")
    p.add_argument("--full-parts", default=",".join(lora.DEFAULT_PARTS),
                   help="parts trained in full beside the adapter, comma-separated from "
                        f"{','.join(lora.FULL_PARTS)}; 'none' for LoRA only; 'all' for every backbone parameter")
    p.add_argument("--lr", type=float, default=1e-4, help="constant after the warmup, for the adapter and the parts")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999, help="per update; the EMA is fp32 on the training device")
    p.add_argument("--step-grid", default=DEFAULT_GRID,
                   help="update counts at which the adapter is saved and validated; the last is the run length")
    p.add_argument("--global-batch", type=int, default=32)
    p.add_argument("--per-gpu-batch", type=int, default=32, help="the micro-batch per process")
    p.add_argument("--allow-accumulation", action="store_true",
                   help="reach --global-batch by gradient accumulation when the micro-batch cannot (off by default: "
                        "fill the card)")
    p.add_argument("--grad-ckpt", action="store_true", help="recompute activations in the backward pass (16 GB cards)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--val-every", type=int, default=0,
                   help="extra validation reads every N updates; the grid steps are always validated")
    p.add_argument("--parity-windows", type=int, default=8, help="held-out windows of the step-0 parity check")
    p.add_argument("--parity-tol", type=float, default=0.0,
                   help="largest allowed |adapted - frozen| prediction at initialisation (0 = exact)")
    p.add_argument("--warm-start", default=None,
                   help="where the source graph's architecture is read from (default: the source run's --warm-start); "
                        "the weights always come from --source")
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--fit-check", type=int, default=0,
                   help="run N updates, report steps/s and peak memory, write no checkpoint and stream nothing")
    p.add_argument("--no-wandb", dest="wandb", action="store_false",
                   help="do not stream to Weights & Biases (on by default, as in pretraining)")
    p.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-group", default="", help="default adapt_<set>_map<NN>, so one map's runs overlay")
    p.add_argument("--seed", type=int, default=0, help="the data order and the adapter's initial A matrices")
    return p


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
