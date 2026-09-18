"""CPU gates for `backbones.SD35WorldModel`, run before the row is given any GPU time.

Five checks, all on the CPU and all reporting a number rather than a pass/fail word:

1. `parity`    the wrapper with zero context channels against the pretrained transformer on the
               same 16-channel target and the same conditioning tensors. Max abs diff must be
               exactly 0: that is what "step 0 equals the pretrained model" means, and it covers
               the inflated patch projection and the cropped positional table together.
2. `grads`     one AdamW step; every parameter the wrapper added (the two joint-attention
               conditioning tables, the pooled action table, the pooled base vector) and the
               context-channel block of the inflated patch projection must receive a finite
               nonzero gradient and move.
3. `overfit`   30 AdamW steps on one fixed random batch under `VDiffusion`'s v-loss; the loss
               must fall, and the report says how many of the 29 steps were non-increasing.
4. `roundtrip` state dict saved and reloaded `strict=True`, plus the trainer's EMA path: the
               fp32 CPU EMA list, cast to bf16 and keyed the way `train_wm.py` keys it, must
               reload into a fresh wrapper with no missing or unexpected keys.
5. `shapes`    rectangular-grid bookkeeping: 16x20 tokens in, (B, 16, 32, 40) out, the positional
               crop offsets, and the parameter count.

By default the checks run on a small randomly initialised `SD3Transformer2DModel` of the same
class and the same structural switches (patch 2, `pos_embed_max_size` 96, 16 in/out channels,
dual attention in the first block). That is enough for every check here, because none of them
depends on the *values* of the pretrained weights: parity is a statement about where the
pretrained kernel was placed, and random non-zero weights test it more sharply than real ones.
`--real` repeats the run against the gated checkpoint itself, which needs an accepted licence
gate and about 4.2 GB of download, and is the form to run on Spiderman's CPU.

    python verify_sd35.py
    python verify_sd35.py --real --sd35-path stabilityai/stable-diffusion-3.5-medium --hf-cache $D/hf/hub
"""
import argparse
import copy
import json
import os
import tempfile

import torch

from backbones import SD35_DEFAULT, SD35WorldModel, stacked_in_channels
from diffusion_v import VDiffusion, noise_augment

# Small but structurally faithful: the switches the wrapper reads (patch size, cropped positional
# table, 16 in/out channels, dual attention, qk norm) are the real ones; only width and depth shrink.
TINY = dict(sample_size=128, patch_size=2, in_channels=16, out_channels=16, num_layers=2,
            attention_head_dim=8, num_attention_heads=2, joint_attention_dim=32,
            caption_projection_dim=16, pooled_projection_dim=24, pos_embed_max_size=96,
            dual_attention_layers=(0,), qk_norm="rms_norm")


def build(args, num_actions, context_frames, noise_buckets, action_dropout=0.0):
    """Return (wrapper, pristine pretrained transformer) sharing the same initial weights."""
    from diffusers import SD3Transformer2DModel
    if args.real:
        pristine = SD3Transformer2DModel.from_pretrained(args.sd35_path, subfolder="transformer",
                                                         cache_dir=args.hf_cache, low_cpu_mem_usage=False)
    else:
        torch.manual_seed(0)
        pristine = SD3Transformer2DModel(**TINY)
    wrapper = SD35WorldModel(num_actions=num_actions, context_frames=context_frames, noise_buckets=noise_buckets,
                             action_dropout=action_dropout, grad_ckpt=False, latent_channels=16,
                             transformer=copy.deepcopy(pristine))
    return wrapper.eval(), pristine.eval()


def batch(wrapper, b=None, latent_hw=(32, 40), seed=0, cover_all_rows=False):
    """A random window batch. `cover_all_rows` sizes it so every action row (including the null
    row at index num_actions) and every noise bucket appears exactly once, which is what lets the
    gradient gate speak for the whole conditioning table rather than the rows one draw happened
    to touch."""
    rows = wrapper.num_actions + 1
    buckets = wrapper.bucket_embedder.num_embeddings
    b = max(rows, buckets) if cover_all_rows else (b or 2)
    g = torch.Generator().manual_seed(seed)
    c = wrapper.latent_channels
    x = torch.randn(b, c, *latent_hw, generator=g)
    ctx = torch.randn(b, c * wrapper.context_frames, *latent_hw, generator=g)
    t = torch.randint(0, 1000, (b,), generator=g)
    action = torch.arange(b) % rows
    bucket = torch.arange(b) % buckets
    return x, ctx, t, action, bucket


@torch.no_grad()
def gate_parity(args):
    wrapper, pristine = build(args, num_actions=4, context_frames=32, noise_buckets=3)
    x, _, t, action, bucket = batch(wrapper, 2)
    tokens, pooled = wrapper.conditioning(action, bucket)
    zero_ctx = torch.zeros(x.shape[0], wrapper.latent_channels * wrapper.context_frames, *x.shape[-2:])
    ours = wrapper(x, t, action, zero_ctx, bucket)
    theirs = pristine(x, encoder_hidden_states=tokens, pooled_projections=pooled, timestep=t).sample
    return {"max_abs_diff": float((ours - theirs).abs().max()), "out_shape": list(ours.shape),
            "in_channels": int(wrapper.transformer.config.in_channels),
            "pretrained_kernel_on_target": bool(torch.equal(
                wrapper.transformer.pos_embed.proj.weight[:, -wrapper.latent_channels:],
                pristine.pos_embed.proj.weight)),
            "context_kernel_max_abs": float(wrapper.transformer.pos_embed.proj.weight[:, :-wrapper.latent_channels].abs().max())}


def new_parameter_names(wrapper):
    """The parameters that did not exist in the pretrained checkpoint."""
    return ["action_embedder.weight", "bucket_embedder.weight", "pooled_action.weight", "pooled_base"]


def gate_grads(args):
    # dropout off and every row present once, so the gate is about the wiring, not about which
    # rows a random draw reached; the null row is fed directly as action id num_actions
    wrapper, _ = build(args, num_actions=4, context_frames=32, noise_buckets=3, action_dropout=0.0)
    wrapper.train()
    torch.manual_seed(0)
    x, ctx, t, action, bucket = batch(wrapper, cover_all_rows=True)
    diffusion = VDiffusion()
    opt = torch.optim.AdamW(wrapper.parameters(), lr=1e-3)
    ctx_n, bucket = noise_augment(ctx, 0.7, wrapper.bucket_embedder.num_embeddings)
    before = {n: p.detach().clone() for n, p in wrapper.named_parameters()}
    loss = diffusion.training_loss(lambda xt, tt: wrapper(xt, tt, action, ctx_n, bucket), x, t=t)
    loss.backward()
    params = dict(wrapper.named_parameters())
    out = {"loss": float(loss.detach()), "batch": int(x.shape[0]), "new_parameters": {}}
    for n in new_parameter_names(wrapper):
        g = params[n].grad
        out["new_parameters"][n] = {"grad_finite": bool(torch.isfinite(g).all()), "grad_abs_max": float(g.abs().max()),
                                    "rows_with_grad": int((g.reshape(g.shape[0], -1).abs().sum(1) > 0).sum()) if g.dim() > 1 else 1,
                                    "rows": int(g.shape[0]) if g.dim() > 1 else 1}
    # the inflated context channels start at zero; they must be trainable, not frozen
    proj = params["transformer.pos_embed.proj.weight"]
    ctx_grad = proj.grad[:, :-wrapper.latent_channels]
    out["context_channels"] = {"grad_finite": bool(torch.isfinite(ctx_grad).all()), "grad_abs_max": float(ctx_grad.abs().max())}
    opt.step()
    moved = {n: float((params[n].detach() - before[n]).abs().max()) for n in new_parameter_names(wrapper)}
    moved["transformer.pos_embed.proj.weight[context]"] = float(
        (proj.detach()[:, :-wrapper.latent_channels] - before["transformer.pos_embed.proj.weight"][:, :-wrapper.latent_channels]).abs().max())
    out["moved_after_one_step"] = moved
    out["all_new_parameters_moved"] = all(v > 0 for v in moved.values())
    out["all_rows_have_grad"] = all(v["rows_with_grad"] == v["rows"] for v in out["new_parameters"].values())
    return out


def gate_overfit(args, steps=30):
    wrapper, _ = build(args, num_actions=4, context_frames=32, noise_buckets=3)
    wrapper.train()
    torch.manual_seed(0)
    x, ctx, t, action, bucket = batch(wrapper, 4)
    ctx_n, bucket = noise_augment(ctx, 0.7, wrapper.bucket_embedder.num_embeddings)
    diffusion = VDiffusion()
    noise = torch.randn_like(x)
    opt = torch.optim.AdamW(wrapper.parameters(), lr=1e-3)
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = diffusion.training_loss(lambda xt, tt: wrapper(xt, tt, action, ctx_n, bucket), x, noise=noise, t=t)
        loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    drops = sum(b <= a for a, b in zip(losses, losses[1:]))
    return {"first": losses[0], "last": losses[-1], "monotone_steps": drops, "of": steps - 1,
            "strictly_monotone": drops == steps - 1, "losses": [round(v, 4) for v in losses]}


def gate_roundtrip(args):
    wrapper, _ = build(args, num_actions=4, context_frames=32, noise_buckets=3)
    fresh, _ = build(args, num_actions=4, context_frames=32, noise_buckets=3)
    sd = {k: v.detach().cpu().to(torch.bfloat16) for k, v in wrapper.state_dict().items()}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ck.pt")
        # the trainer's EMA is a per-parameter fp32 CPU list, keyed on save by parameter name
        ema = [p.detach().float().cpu().clone() for p in wrapper.parameters()]
        names = [n for n, _ in wrapper.named_parameters()]
        torch.save({"model": sd, "ema": {k: t.to(torch.bfloat16) for k, t in zip(names, ema)}}, path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        missing, unexpected = fresh.load_state_dict({k: v.float() for k, v in ck["model"].items()}, strict=True)
        # the EMA covers parameters only, so it loads with strict=False and must leave buffers as the only gap
        ema_missing, ema_unexpected = fresh.load_state_dict({k: v.float() for k, v in ck["ema"].items()}, strict=False)
        buffers = {n for n, _ in wrapper.named_buffers()}
    return {"model_keys": len(sd), "missing": list(missing), "unexpected": list(unexpected),
            "ema_keys": len(ck["ema"]), "ema_unexpected": list(ema_unexpected),
            "ema_missing_are_buffers_only": set(ema_missing) == buffers,
            "persistent_buffers_in_state_dict": sorted(set(sd) - set(names))}


@torch.no_grad()
def gate_shapes(args):
    wrapper, _ = build(args, num_actions=4, context_frames=32, noise_buckets=10)
    cfg = wrapper.transformer.config
    patch = wrapper.transformer.pos_embed
    grid = (32 // cfg.patch_size, 40 // cfg.patch_size)
    top = (cfg.pos_embed_max_size - grid[0]) // 2
    left = (cfg.pos_embed_max_size - grid[1]) // 2
    x, ctx, t, action, bucket = batch(wrapper, 2)
    out = wrapper(x, t, action, ctx, bucket)
    new = {n for n in new_parameter_names(wrapper)}
    return {"token_grid": list(grid), "pos_embed_crop_top_left": [top, left],
            "pos_embed_table": list(patch.pos_embed.shape), "out_shape": list(out.shape),
            "expected_in_channels": stacked_in_channels(16, 32),
            "params_total": sum(p.numel() for p in wrapper.parameters()),
            "params_new": sum(p.numel() for n, p in wrapper.named_parameters() if n in new)}


GATES = {"parity": gate_parity, "grads": gate_grads, "overfit": gate_overfit,
         "roundtrip": gate_roundtrip, "shapes": gate_shapes}


def main(args):
    torch.set_grad_enabled(True)
    results = {}
    for name in (args.gates or list(GATES)):
        results[name] = GATES[name](args)
        print(f"=== {name}\n{json.dumps(results[name], indent=1)}", flush=True)
    verdict = {
        "parity_exact": results.get("parity", {}).get("max_abs_diff") == 0.0,
        "all_new_parameters_move": results.get("grads", {}).get("all_new_parameters_moved"),
        "all_conditioning_rows_have_grad": results.get("grads", {}).get("all_rows_have_grad"),
        "overfit_monotone": results.get("overfit", {}).get("strictly_monotone"),
        "roundtrip_strict": results.get("roundtrip", {}).get("missing") == [] and results.get("roundtrip", {}).get("unexpected") == [],
    }
    print("=== verdict\n" + json.dumps(verdict, indent=1), flush=True)
    if args.out:
        json.dump({"real": args.real, "results": results, "verdict": verdict}, open(args.out, "w"), indent=1)
    return 0 if all(v in (True, None) for v in verdict.values()) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real", action="store_true", help="use the gated SD 3.5 Medium checkpoint instead of a small random config")
    p.add_argument("--sd35-path", default=SD35_DEFAULT)
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--gates", nargs="*", choices=list(GATES), help="subset of gates to run (default: all)")
    p.add_argument("--out", default="", help="write the numbers to this json")
    raise SystemExit(main(p.parse_args()))
