# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from accelerate import Accelerator

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from torchvision.utils import save_image


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # Do EMA math in fp32 even if EMA is stored in bf16. bf16 has only 7-bit mantissa:
        # `ema.add_(param, alpha=1e-4)` underflows 1e-4 * param (~1e-4 magnitude) to zero,
        # freezing the EMA at its initial weights. Verified bug. Always compute in fp32.
        ema_p = ema_params[name]
        buf = ema_p.detach().float()
        buf.mul_(decay).add_(param.data.float(), alpha=1 - decay)
        ema_p.copy_(buf.to(ema_p.dtype))


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset(Dataset):
    def __init__(self, context_dir, target_dir, actions_dir):
        # Memory-map the two big latent arrays so all DataLoader worker processes
        # share one page-cached copy rather than loading 23GB+5.9GB into each RSS.
        # Actions (99MB) is small enough to load fully.
        self.context = np.load(context_dir, mmap_mode="r")
        self.target = np.load(target_dir, mmap_mode="r")
        self.actions = np.load(actions_dir)


    def __len__(self):
        return len(self.target)


    def __getitem__(self, idx):
        context = torch.from_numpy(self.context[idx]).float()   # (4, 4, 15, 20)
        target = torch.from_numpy(self.target[idx]).float()     # (4, 15, 20)
        # actions file is (N, 4) or (N, 5) — a sequence of past (and maybe target) actions.
        # Design A: condition on the single most-recent action only. Grab the last element.
        action = torch.tensor(self.actions[idx][-1], dtype=torch.long)
        
        # Pad H from 15 to 16 on the bottom. F.pad format: (W_left, W_right, H_top, H_bottom)
        target = F.pad(target, (0, 0, 0, 1))                    # (4, 16, 20)
        context = F.pad(context, (0, 0, 0, 1))                  # (4, 4, 16, 20)
        
        # Channel-stack context: (4, 4, 16, 20) → (16, 16, 20)
        context = context.reshape(-1, 16, 20)
        
        return context, target, action


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator. bf16 is native on Ampere (A4000 etc.) — no loss-scale overhead, ~1.7x speedup.
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=(16, 20),    # rectangular DOOM latent grid
        in_channels=20,         # 16 context + 4 target after channel concat
        pred_channels=4,        # model predicts the 4-channel target only
        num_classes=args.num_classes,
    )
    model.use_grad_ckpt = args.grad_ckpt  # disabled by default; 80-token DiT doesn't need it
    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)

    # Optional pretrained warm-start. Copies DiT-XL/2 ImageNet weights for shared params;
    # inflates x_embedder from 4→20 input channels (first 4 get pretrained, rest stay random);
    # skips pos_embed (grid mismatch) and y_embedder (class-count mismatch).
    if args.ckpt:
        state_dict = find_model(args.ckpt)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        pretrained_proj = state_dict.pop("x_embedder.proj.weight")  # (D, 4, p, p)
        pretrained_bias = state_dict.pop("x_embedder.proj.bias", None)
        with torch.no_grad():
            model.x_embedder.proj.weight[:, :4].copy_(pretrained_proj.to(device))
            if pretrained_bias is not None:
                model.x_embedder.proj.bias.copy_(pretrained_bias.to(device))
        for k in list(state_dict.keys()):
            if k.startswith("pos_embed") or k.startswith("y_embedder"):
                state_dict.pop(k)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if accelerator.is_main_process:
            logger.info(f"Warm-start from {args.ckpt}: {len(missing)} missing, {len(unexpected)} unexpected keys")

    # Create an EMA copy for inference. Stored in bf16 to save ~1.35GB vs fp32 on a 16GB GPU;
    # the quality impact is negligible because EMA is only read during sampling/eval.
    ema = deepcopy(model).to(device).to(torch.bfloat16 if args.ema_bf16 else torch.float32)
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer. fused=True uses a single CUDA kernel that updates params, moments, and
    # sqrt in-place, avoiding the ~2.7GB intermediate sqrt buffer that unfused multi_tensor_adam
    # allocates during step. Critical on 16GB A4000s running DiT-XL/2.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0, fused=True)
    # Linear warmup from 0 → args.lr over the first args.warmup_steps. Critical for
    # warm-started models: constant LR into a mismatched prior can explode early grads.
    def _lr_lambda(step):
        if args.warmup_steps <= 0:
            return 1.0
        return min(1.0, step / args.warmup_steps)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    # Setup data. Prefer the full-dataset filenames; fall back to the debug files if those
    # don't exist in feature_path (e.g. pointing at data/debug for smoke tests).
    def _pick(full_name, debug_name):
        full_path = os.path.join(args.feature_path, full_name)
        debug_path = os.path.join(args.feature_path, debug_name)
        if os.path.isfile(full_path):
            return full_path
        return debug_path
    context = _pick("context_latents.npy", "context_latents_debug.npy")
    target = _pick("target_latents.npy", "target_latents_debug.npy")
    action = _pick("context_actions.npy", "context_actions_debug.npy")

    dataset = CustomDataset(context, target, action)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    # Setup in-training sampling (main process only). Loads SD-VAE for decoding,
    # pins a fixed eval batch so we see the same frames evolve over training,
    # and builds a faster DDIM diffusion (~50 steps instead of 1000).
    vae = None
    sample_diffusion = None
    eval_ctx = eval_tgt = eval_act = None
    samples_dir = None
    if accelerator.is_main_process and args.sample_every > 0:
        samples_dir = f"{experiment_dir}/samples"
        os.makedirs(samples_dir, exist_ok=True)
        # VAE lives on CPU to save ~335MB of GPU memory on rank 0 (we're tight at 16GB/GPU
        # with batch 16 + grad_ckpt + DDP). Decoding takes a few seconds per event — fine
        # because sampling only runs every --sample-every steps.
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to("cpu")
        vae.eval()
        requires_grad(vae, False)
        sample_diffusion = create_diffusion(timestep_respacing=str(args.num_sample_steps))
        # Comprehensive eval: N segments evenly spaced across the dataset, each S consecutive
        # samples. Stack into one big batch for a single DDIM pass; save each segment as its
        # own PNG so we can visually track different parts of the game simultaneously.
        N = args.num_eval_segments
        S = args.eval_segment_size
        total_samples = len(dataset)
        # linspace start indices so segments don't overlap and cover the full range.
        starts = np.linspace(0, max(total_samples - S, 0), N, dtype=int)
        eval_batches = []
        for start in starts:
            ctxs, tgts, acts = [], [], []
            for i in range(S):
                c, t, a = dataset[int(start) + i]
                ctxs.append(c); tgts.append(t); acts.append(a)
            eval_batches.append((torch.stack(ctxs), torch.stack(tgts), torch.stack(acts)))
        # Concat into one (N*S, ...) batch for a single DDIM pass.
        eval_ctx = torch.cat([b[0] for b in eval_batches], dim=0).to(device)
        eval_tgt = torch.cat([b[1] for b in eval_batches], dim=0).to(device)
        eval_act = torch.cat([b[2] for b in eval_batches], dim=0).to(device)
        # Save ground truth: one PNG per segment so the layout matches sample outputs.
        with torch.no_grad():
            gt_latent = (eval_tgt[:, :, :15, :].float() / 0.18215).cpu()
            gt_img = vae.decode(gt_latent).sample
        gt_img = (gt_img * 0.5 + 0.5).clamp(0, 1)
        gt_dir = f"{samples_dir}/ground_truth"
        os.makedirs(gt_dir, exist_ok=True)
        for seg_idx in range(N):
            seg_imgs = gt_img[seg_idx * S:(seg_idx + 1) * S]
            save_image(seg_imgs, f"{gt_dir}/segment_{seg_idx:02d}_start{starts[seg_idx]}.png", nrow=S)
        logger.info(f"Saved {N} ground-truth segments (each {S} frames) to {gt_dir}/")

    def save_samples(step):
        """DDIM-sample next-frames using the LIVE model in fp32 (not bf16 EMA).
        The live-model weights are the ground truth of training; EMA is kept for export
        but not used for in-training samples (it's bf16-stored and lags the live model).
        Decode through CPU VAE, save one PNG per segment.
        """
        if vae is None:
            return
        sampling_model = accelerator.unwrap_model(model)
        was_training = sampling_model.training
        sampling_model.eval()
        shape = eval_tgt.shape
        noise = torch.randn(shape, device=device, dtype=torch.float32)
        model_kwargs = dict(context=eval_ctx.float(), action=eval_act)
        with torch.no_grad():
            samples = sample_diffusion.p_sample_loop(
                sampling_model, shape, noise, clip_denoised=False,
                model_kwargs=model_kwargs, progress=False, device=device,
            )
            samples = (samples[:, :, :15, :].float() / 0.18215).cpu()
            imgs = vae.decode(samples).sample
        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
        step_dir = f"{samples_dir}/step_{step:07d}"
        os.makedirs(step_dir, exist_ok=True)
        N = args.num_eval_segments
        S = args.eval_segment_size
        for seg_idx in range(N):
            seg_imgs = imgs[seg_idx * S:(seg_idx + 1) * S]
            save_image(seg_imgs, f"{step_dir}/segment_{seg_idx:02d}.png", nrow=S)
        if was_training:
            sampling_model.train()
        del samples, imgs, noise, model_kwargs
        torch.cuda.empty_cache()

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader, lr_scheduler = accelerator.prepare(model, opt, loader, lr_scheduler)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    best_loss = float("inf")  # track best smoothed training loss for best.pt
    
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for context, target, action in loader:
            context = context.to(device)
            target = target.to(device)
            action = action.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (target.shape[0],), device=device)
            model_kwargs = dict(context=context,action=action)
            loss_dict = diffusion.training_losses(model, target, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            if args.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            lr_scheduler.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = accelerator.reduce(avg_loss, reduction="mean")

                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    # Save best EMA checkpoint by smoothed training loss. Persists through rotation.
                    # Skip until after warmup (first ~args.warmup_steps steps) so the initial
                    # loss-plummet doesn't mark a premature "best".
                    if train_steps >= args.warmup_steps and float(avg_loss) < best_loss:
                        best_loss = float(avg_loss)
                        best_path = f"{checkpoint_dir}/best.pt"
                        torch.save({
                            "ema": ema.state_dict(),
                            "model": {k: v.to(torch.bfloat16) if v.is_floating_point() else v
                                  for k, v in accelerator.unwrap_model(model).state_dict().items()},
                            "args": args,
                            "step": train_steps,
                            "loss": best_loss,
                        }, best_path)
                        logger.info(f"New best loss {best_loss:.4f} at step {train_steps} -> {best_path}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint.
            # Default: save only EMA + args (~1.35GB bf16). The EMA is what you use for
            # inference. Optimizer state (5.4GB fp32) is only needed for mid-training resume;
            # the live model copy (2.7GB fp32) duplicates what's in EMA. Full checkpoints
            # only every --full-ckpt-every steps.
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    is_full = args.full_ckpt_every > 0 and train_steps % args.full_ckpt_every == 0
                    # Always save the live model too — bf16 EMA underflow bug taught us that
                    # "ema-only" checkpoints can leave you holding untrained weights if EMA
                    # ever fails silently. Model weights are the ground truth of learning.
                    checkpoint = {
                        "ema": ema.state_dict(),
                        "model": {k: v.to(torch.bfloat16) if v.is_floating_point() else v
                                  for k, v in accelerator.unwrap_model(model).state_dict().items()},
                        "args": args,
                        "step": train_steps,
                    }
                    if is_full:
                        checkpoint["opt"] = opt.state_dict()
                    suffix = "_full" if is_full else ""
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}{suffix}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved {'full' if is_full else 'ema-only'} checkpoint to {checkpoint_path}")

                    # Rotate: keep only the last --keep-last checkpoints of each kind to bound disk.
                    # Never prune best.pt — it's the best-by-loss anchor.
                    if args.keep_last > 0:
                        for kind_suffix in ("", "_full"):
                            existing = sorted(glob(f"{checkpoint_dir}/*{kind_suffix}.pt"))
                            existing = [p for p in existing
                                        if p.endswith("_full.pt") == (kind_suffix == "_full")
                                        and os.path.basename(p) != "best.pt"]
                            for stale in existing[:-args.keep_last]:
                                try:
                                    os.remove(stale)
                                    logger.info(f"Pruned old checkpoint {stale}")
                                except OSError:
                                    pass

            # Decode + save PNG grid from the fixed eval batch.
            if args.sample_every > 0 and train_steps % args.sample_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    save_samples(train_steps)
                    logger.info(f"Saved samples to {samples_dir}/step_{train_steps:07d}/")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=18)  # Vizdoom full action set (0-17)
    parser.add_argument("--ckpt", type=str, default="DiT-XL-2-256x256.pt",
                        help="Pretrained checkpoint for warm-start (empty string to disable)")
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000,
                        help="How often to save an EMA-only checkpoint (~1.35GB each).")
    parser.add_argument("--full-ckpt-every", type=int, default=0,
                        help="How often to save a FULL checkpoint with model+opt state "
                             "(~9.5GB each). Set to 0 to disable full checkpoints entirely.")
    parser.add_argument("--keep-last", type=int, default=3,
                        help="Keep only the N most recent checkpoints of each kind. "
                             "Set to 0 to keep all.")
    parser.add_argument("--mixed-precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"],
                        help="bf16 is native on Ampere (A4000, A100) with no loss-scale overhead")
    parser.add_argument("--grad-ckpt", action="store_true",
                        help="Enable gradient checkpointing. Off by default; at 80 tokens it's overhead.")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm. 0 to disable.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Linear warmup steps. Important for warm-started models.")
    parser.add_argument("--ema-bf16", action="store_true", default=True,
                        help="Store EMA weights in bf16 to save GPU memory. On by default.")
    parser.add_argument("--no-ema-bf16", action="store_false", dest="ema_bf16",
                        help="Force fp32 EMA (needs more memory).")
    parser.add_argument("--sample-every", type=int, default=500,
                        help="Decode + save a PNG grid from EMA every N steps. 0 to disable.")
    parser.add_argument("--num-sample-steps", type=int, default=50,
                        help="DDIM/DDPM sampling steps for in-training visualization.")
    parser.add_argument("--num-eval-segments", type=int, default=10,
                        help="Number of evenly-spaced segments across the dataset to track.")
    parser.add_argument("--eval-segment-size", type=int, default=8,
                        help="Consecutive frames per eval segment.")
    args = parser.parse_args()
    main(args)
