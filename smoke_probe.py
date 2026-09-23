"""Did the 300-step smoke actually train the conditioning pathways? A launch gate (review 7.1 item 3).

A smoke that reaches its end event proves the loop runs. It does not prove that the parts a
next-tic model depends on learn: a control embedder whose gradient is zero, a context convolution
that stays at its zero initialisation, or an SD 3.5 `pooled_control` that never leaves zero all
train "fine" and produce a model that ignores its inputs. On the smoke's recovery checkpoint this
checks, per group:

  control MLP       `control_history.mlp.*`        the executed-control token embedder
  position table    `control_history.pos`          its learned positions
  context conv      the inflated input projection, only its CONTEXT input channels (zero at step 0)
  pooled control    `pooled_control.*` (SD 3.5 only, zero at step 0)

  1. updates        every group moved: live weights differ from the EMA (which starts as a copy of
                    the step-0 weights and trails them), and for the zero-initialised groups the live
                    weights are nonzero. All finite.
  2. gradients      one backward pass of the v-loss on a real validation batch at the smoke's live
                    weights, with fixed noise and timestep: every group's gradient is finite and
                    nonzero.
  3. sensitivity    under fixed noise, context and timestep, flipping every bit of the NEWEST control
                    (the control applied from the last context frame into the target) changes the
                    output; finite and nonzero. Whether it changes it in the right direction is a 10k
                    question (review 7.2), not a 300-step one.

    python smoke_probe.py --ckpt $D/results_smoke/unet/0000300.pt --backbone unet \\
        --latents-dir $D/latents_arnold_dense_pertic_eval/val --episodes 6000:6100 --device cuda:0
"""
import argparse
import json
import math

import numpy as np
import torch

GROUPS = ("control_mlp", "control_pos", "context_conv", "pooled_control")
REL_UPDATE_FLOOR = 1e-6     # relative |live - EMA|; fp32 rounding of an unmoved parameter is ~1e-7


def strip(name):
    for p in ("module.", "_orig_mod."):
        while name.startswith(p):
            name = name[len(p):]
    return name


def group_of(name, shape, latent_channels, context_frames):
    """Which probed group a parameter belongs to, or None.

    The context convolution is found by shape, not by name, because each backbone inflates a
    differently named projection (`unet.conv_in`, PixArt's and SD 3.5's `pos_embed.proj`): it is the
    only 4-D weight taking `latent_channels * (context_frames + 1)` input channels.
    """
    n = strip(name)
    if n.startswith("control_history.mlp."):
        return "control_mlp"
    if n == "control_history.pos":
        return "control_pos"
    if n.startswith("pooled_control."):
        return "pooled_control"
    if len(shape) == 4 and n.endswith(".weight") and shape[1] == latent_channels * (context_frames + 1):
        return "context_conv"
    return None


def groups_in(state, latent_channels, context_frames, expect_pooled=False, expect_controls=True):
    """{group: [parameter names]} of a state dict, with a problem for each expected group that is absent."""
    out, problems = {g: [] for g in GROUPS}, []
    for name, t in state.items():
        g = group_of(name, tuple(t.shape), latent_channels, context_frames)
        if g:
            out[g].append(name)
    want = ["context_conv"] + (["control_mlp", "control_pos"] if expect_controls else []) \
        + (["pooled_control"] if expect_pooled else [])
    for g in want:
        if not out[g]:
            problems.append(f"no {g} parameters in the checkpoint")
    return {g: v for g, v in out.items() if v}, problems


def _norm(t):
    return float(t.double().norm())


def update_report(ck, groups, latent_channels, context_frames):
    """Per group: relative live-vs-EMA movement and, for zero-initialised weights, the live norm."""
    live, ema = ck["model"], ck.get("ema") or {}
    ctx = latent_channels * context_frames
    rep, problems = {}, []
    for g, names in groups.items():
        rel, zero_init_norm, finite = [], None, True
        for n in names:
            w = live[n].float()
            finite &= bool(torch.isfinite(w).all())
            if g == "context_conv":
                zero_init_norm = _norm(w[:, :ctx])
            elif g == "pooled_control":
                zero_init_norm = (zero_init_norm or 0.0) + _norm(w)
            if n in ema:
                e = ema[n].float()
                finite &= bool(torch.isfinite(e).all())
                rel.append(_norm(w - e) / max(_norm(w), 1e-12))
        r = {"relative_update": max(rel) if rel else None, "zero_init_norm": zero_init_norm, "finite": finite}
        rep[g] = r
        if not finite:
            problems.append(f"{g}: non-finite weights")
            continue
        if g in ("context_conv", "pooled_control"):
            if not zero_init_norm or not math.isfinite(zero_init_norm):
                problems.append(f"{g}: still at its zero initialisation after the smoke")
        elif r["relative_update"] is None:
            problems.append(f"{g}: no EMA copy to measure its movement against")
        elif not r["relative_update"] > REL_UPDATE_FLOOR:
            problems.append(f"{g}: did not move (relative |live - EMA| {r['relative_update']:.2e})")
    return rep, problems


def gradient_and_sensitivity(model, groups, batch, objective="v", t_value=500, seed=0):
    """(report, problems): one backward pass at fixed noise and t, and the newest-control flip test."""
    from diffusion_v import VDiffusion
    ctx, tgt, act = batch
    device = next(model.parameters()).device
    ctx, tgt, act = ctx.to(device), tgt.to(device), act.to(device)
    B = tgt.shape[0]
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(tgt.shape, generator=g).to(device)
    t = torch.full((B,), int(t_value), dtype=torch.long, device=device)
    bucket = torch.zeros(B, dtype=torch.long, device=device)
    diff = VDiffusion(device=device, objective=objective)
    named = dict(model.named_parameters())
    model.train()
    model.zero_grad(set_to_none=True)
    loss = diff.training_loss(lambda xt, tt: model(xt, tt, act, ctx, bucket), tgt, noise=noise, t=t)
    loss.backward()
    rep, problems = {"loss": float(loss.detach())}, []
    if not math.isfinite(rep["loss"]):
        problems.append("the probe loss is not finite")
    for grp, names in groups.items():
        norms = []
        for n in names:
            p = named.get(strip(n), named.get(n))
            gr = None if p is None else p.grad
            norms.append(float("nan") if gr is None else _norm(gr))
        tot = float(np.sqrt(np.nansum(np.square(norms)))) if norms else float("nan")
        rep[f"grad_{grp}"] = tot
        if any(not math.isfinite(x) for x in norms):
            problems.append(f"{grp}: a gradient is missing or not finite")
        else:
            # every tensor, not the group total: a dead first layer hides behind a live second one
            dead = [n for n, x in zip(names, norms) if not x > 0]
            if dead:
                problems.append(f"{grp}: zero gradient on {dead[:4]}")
    model.zero_grad(set_to_none=True)
    model.eval()
    if act.ndim == 3:
        with torch.no_grad():
            xt = diff.q_sample(tgt, t, noise)
            a = model(xt, t, act, ctx, bucket).float()
            flipped = act.clone()
            flipped[:, -1] = 1 - flipped[:, -1]
            b = model(xt, t, flipped, ctx, bucket).float()
        s = float((a - b).abs().max())
        rep["newest_control_sensitivity"] = s
        if not math.isfinite(s):
            problems.append("the output under a flipped newest control is not finite")
        elif not s > 0:
            problems.append("flipping the newest control does not change the output")
    return rep, problems


def probe(ck, model, batch, latent_channels, context_frames, backbone):
    """The whole report for one smoke checkpoint; `ok` is False on any problem."""
    trained = (ck.get("args") or {})
    groups, problems = groups_in(ck["model"], latent_channels, context_frames,
                                 expect_pooled=backbone == "sd35",
                                 expect_controls=bool(int(trained.get("action_history") or 0)))
    upd, p1 = update_report(ck, groups, latent_channels, context_frames)
    grad, p2 = gradient_and_sensitivity(model, groups, batch, (trained.get("objective") or "v"))
    problems += p1 + p2
    return {"step": ck.get("step"), "groups": {g: len(v) for g, v in groups.items()}, "updates": upd,
            "probe": grad, "problems": problems, "ok": not problems}


def real_batch(latents_dir, episode_ids, context_frames, latent_channels, action_history, batch=4, seed=0):
    """A few windows of the real corpus, drawn with a fixed seed."""
    from doom_data import TicWindowDataset
    ds = TicWindowDataset(latents_dir, episode_ids, context_frames, latent_channels=latent_channels,
                          action_history=action_history)
    idx = np.random.RandomState(seed).choice(len(ds), size=min(batch, len(ds)), replace=False)
    items = [ds[int(i)] for i in idx]
    return tuple(torch.stack([it[k] for it in items]) for k in range(3))


def main(args):
    import eval_tf
    from backbones import resolve_latent_channels
    from doom_data import parse_episode_ids
    device = args.device if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ns = argparse.Namespace(ckpt=args.ckpt, backbone=args.backbone, num_actions=args.num_actions,
                            context_frames=args.context_frames, noise_buckets=args.noise_buckets,
                            hf_cache=args.hf_cache, use_ema=False, objective="auto", sd_path=args.sd_path,
                            pixart_path=args.pixart_path, unidiffuser_path=args.unidiffuser_path,
                            sd35_path=args.sd35_path, tic_stride=None, action_history=None)
    trained = eval_tf.checkpoint_interface(ck, ns)
    latent_channels = resolve_latent_channels(args.backbone, args.latent_channels)
    model, _, _ = eval_tf.load_model(ns, device, latent_channels, trained)
    model = model.float()
    batch = real_batch(args.latents_dir, parse_episode_ids(args.episodes), args.context_frames, latent_channels,
                       trained["action_history"], args.batch, args.seed)
    rep = probe(ck, model, batch, latent_channels, args.context_frames, args.backbone)
    text = json.dumps(rep, indent=1, default=float)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)
    print(("SMOKE_PROBE_OK " if rep["ok"] else "SMOKE_PROBE_FAILED ") + args.backbone + " "
          + "; ".join(rep["problems"]))
    return 0 if rep["ok"] else 2


def build_parser():
    from backbones import BACKBONES, PIXART_DEFAULT, SD35_DEFAULT, UNIDIFFUSER_DEFAULT
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="the smoke's recovery checkpoint (fp32 live and EMA)")
    p.add_argument("--backbone", choices=list(BACKBONES), required=True)
    p.add_argument("--latents-dir", dest="latents_dir", required=True, help="a real corpus for the probe batch")
    p.add_argument("--episodes", required=True, help="its episode ids, A:B or a comma list")
    p.add_argument("--latent-channels", dest="latent_channels", type=int, default=0)
    p.add_argument("--context-frames", dest="context_frames", type=int, default=32)
    p.add_argument("--num-actions", dest="num_actions", type=int, default=29)
    p.add_argument("--noise-buckets", dest="noise_buckets", type=int, default=10)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--hf-cache", dest="hf_cache", default=None)
    p.add_argument("--sd-path", dest="sd_path", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--pixart-path", dest="pixart_path", default=PIXART_DEFAULT)
    p.add_argument("--unidiffuser-path", dest="unidiffuser_path", default=UNIDIFFUSER_DEFAULT)
    p.add_argument("--sd35-path", dest="sd35_path", default=SD35_DEFAULT)
    p.add_argument("--out", default="")
    return p


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
