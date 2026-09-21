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

from backbones import (BACKBONES, LATENT_HW, PIXART_DEFAULT, SD35_DEFAULT, UNIDIFFUSER_DEFAULT,
                       resolve_latent_channels)
from diffusion_v import VDiffusion, checkpoint_objective
from doom_data import list_latent_episodes, load_split
from doomdit_utils import (LATENT_SCALE, VAE_NAME, build_vae, denormalize_latents, encode_for_idm,
                           load_world_model_state)


def collect_rollout_windows(latents_dir, episode_ids, L, H, n, seed, latent_channels=None, tic_stride=4):
    """Windows with L seed frames and H future frames, spread over episodes.

    At decision spacing (`tic_stride=4`) the window must lie inside one chain of verified
    transitions. At tic spacing the chain test is meaningless and dangerous: `chain_id` is -1 on
    every non-decision tic, so `cid[s] == cid[s + L + H - 1]` is satisfied by two -1 endpoints and
    would accept a rollout that runs straight through a death. Per-tic windows therefore use
    `doom_data.tic_window_starts`: consecutive recorded tics, one continuous life, one map, across
    the whole seed and the whole horizon.
    """
    from doom_data import tic_window_starts
    eps = []
    for ep, lp, mp in list_latent_episodes(latents_dir):
        if ep not in set(int(e) for e in episode_ids):
            continue
        lat, m = np.load(lp, mmap_mode="r"), np.load(mp)
        if latent_channels is not None and lat.shape[1] != latent_channels:
            raise ValueError(f"{lp}: {lat.shape[1]} latent channels, the model wants {latent_channels}")
        T = lat.shape[0]
        if tic_stride == 1:
            if "is_decision" not in m.files:
                raise ValueError(f"{mp}: no is_decision column; a tic-spaced rollout needs a per-tic corpus")
            starts = tic_window_starts(m, L, H).tolist()
        elif "chain_id" in m.files:
            cid = m["chain_id"]; starts = [s for s in range(T - L - H + 1) if cid[s] == cid[s + L + H - 1]]
        else:
            starts = list(range(max(0, T - L - H + 1)))
        if starts:
            eps.append((ep, lat, m, starts))
    if not eps:
        raise ValueError(f"no episode in {latents_dir} has a valid window of {L} seed + {H} future frames")
    rng = np.random.RandomState(seed)
    picks = []
    for _ in range(n):
        ep, lat, m, starts = eps[rng.randint(len(eps))]
        s = int(starts[rng.randint(len(starts))])
        picks.append((ep, int(m["map_id"][0]), s, lat, m))
    return picks


def step_controls(meta_controls, s, L, h):
    """The L executed button vectors conditioning rollout step `h`, oldest first.

    Step h predicts row `s + L + h`, so its controls are rows `s + h .. s + L + h - 1` and the
    newest is row `s + L + h - 1`: the control leaving the last frame in that step's context,
    exactly as in a teacher-forced window. The frame buffer and the control buffer therefore shift
    together by one row per step; letting only the frames shift is the stale-buffer bug.
    """
    return meta_controls[s + h:s + L + h]


def backbone_source(args):
    """Where build_model should pull architecture/weights from for this backbone.

    The DiT is built from local code, so it needs nothing; the diffusers backbones must be
    instantiated from the same repo they were trained from before the checkpoint is loaded.
    """
    return {"dit": None, "unet": args.sd_path, "pixart": args.pixart_path, "unidiffuser": args.unidiffuser_path,
            "sd35": args.sd35_path}[args.backbone]


@torch.no_grad()
def do_rollout(args):
    from backbones import build_model
    device = "cuda"
    C = resolve_latent_channels(args.backbone, args.latent_channels)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # the action table's size depends on the training-time dropout (fast-DiT adds the null row only when it is > 0)
    dropout = ck.get("args", {}).get("action_dropout", 0.1)
    # the adaLN injection cell renames the adaln_single subtree, so the graph has to be rebuilt the way it was trained
    inject = (ck.get("args") or {}).get("action_inject") or "token"
    from eval_tf import checkpoint_interface
    trained = checkpoint_interface(ck, args)
    tic_stride = trained["tic_stride"]
    model = build_model(args.backbone, args.num_actions, args.context_frames, args.noise_buckets, grad_ckpt=False,
                        warm_start=backbone_source(args), cache_dir=args.hf_cache, action_dropout=dropout,
                        latent_channels=C, action_inject=inject, phase_buckets=trained["phase_buckets"],
                        action_history=trained["action_history"], control_bits=trained["control_bits"])
    if args.use_ema and not ck.get("ema"):
        raise SystemExit(f"--use-ema requested but {args.ckpt} carries no EMA weights (use a recovery checkpoint, not best.pt)")
    load_world_model_state(model, ck, args.use_ema)
    model = model.to(device).eval()
    # the parameterization the checkpoint was trained in, so an epsilon cell rolls out as epsilon
    objective = checkpoint_objective(ck, args.objective)
    diffusion = VDiffusion(device=device, objective=objective)
    print(f"rollout: step {ck.get('step', '?')}, objective {objective}, tic stride {tic_stride}, "
          f"horizon {args.horizon} frame(s) = {args.horizon * tic_stride} tic(s) of game time")
    split = load_split(args.split)
    picks = collect_rollout_windows(args.latents_dir, split[args.subset], args.context_frames, args.horizon,
                                    args.num_rollouts, args.seed, latent_channels=C, tic_stride=tic_stride)
    L, H, B = args.context_frames, args.horizon, args.batch_size
    torch.manual_seed(args.seed)
    pred_all, gt_all, act_all, meta, dec_all = [], [], [], [], []
    hist = trained["action_history"]
    t0 = time.time()
    for i in range(0, len(picks), B):
        chunk = picks[i:i + B]
        seed_lat = torch.stack([torch.from_numpy(np.asarray(lat[s:s + L], dtype=np.float32)) for _, _, s, lat, _ in chunk]).to(device)
        gt = np.stack([np.asarray(lat[s + L:s + L + H], dtype=np.float16) for _, _, s, lat, _ in chunk])
        # the action of step h is the one on row s+L-1+h: the control leaving that step's last
        # context frame, the same index a teacher-forced window reads
        acts = np.stack([m["action"][s + L - 1:s + L - 1 + H].astype(np.int64) for _, _, s, _, m in chunk])
        ctl = None
        if hist:
            from doom_data import control_matrix
            ctl = [control_matrix(m["buttons"]) for _, _, _, _, m in chunk]
        ctx = seed_lat.reshape(len(chunk), -1, *LATENT_HW)
        preds = []
        for h in range(H):
            if hist:
                act = torch.from_numpy(np.stack([step_controls(c, s, L, h) for c, (_, _, s, _, _)
                                                 in zip(ctl, chunk)])).to(device)
            else:
                act = torch.from_numpy(acts[:, h]).to(device)
            ctx_in, bucket = (ctx, torch.zeros(len(chunk), dtype=torch.long, device=device))
            if args.infer_noise > 0:
                ctx_in, bucket = fixed_noise(ctx, args.infer_noise, args.train_noise_max, args.noise_buckets)
            ph = None
            if trained["phase_buckets"]:
                from doom_data import tics_since_decision
                ph = torch.from_numpy(np.stack([tics_since_decision(m["is_decision"],
                                                                    trained["phase_buckets"])[s + L + h]
                                                for _, _, s, _, m in chunk])).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = diffusion.ddim_sample(lambda xt, t: model(xt, t, act, ctx_in, bucket, ph), (len(chunk), C, *LATENT_HW), steps=args.steps, eta=args.eta, device=device)
            preds.append(x.half().cpu().numpy())
            ctx = torch.cat([ctx[:, C:], x.float()], dim=1)   # drop the oldest latent, append the prediction
        pred_all.append(np.stack(preds, axis=1)); gt_all.append(gt); act_all.append(acts)
        if tic_stride == 1:
            dec_all.append(np.stack([m["is_decision"][s + L:s + L + H].astype(bool) for _, _, s, _, m in chunk]))
        meta += [(ep, mp, s) for ep, mp, s, _, _ in chunk]
        print(f"  {i + len(chunk)}/{len(picks)} rollouts, {(time.time() - t0) / (i + len(chunk)):.1f} s each", flush=True)
    extra = {}
    if tic_stride == 1:
        # which rolled-out tics are decision tics, so the IDM (trained at 4-tic spacing) can be
        # given the subsequence it expects instead of frames 4x closer together than it ever saw
        extra["decision"] = np.concatenate(dec_all)
        extra["seed_decision"] = np.stack([np.asarray(m["is_decision"][s:s + L], dtype=bool)
                                           for _, _, s, _, m in picks])
    np.savez(args.out, pred=np.concatenate(pred_all), gt=np.concatenate(gt_all), actions=np.concatenate(act_all),
             seed=np.stack([np.asarray(lat[s:s + L], dtype=np.float16) for _, _, s, lat, _ in picks]),
             episode=np.array([m[0] for m in meta]), map=np.array([m[1] for m in meta]), start=np.array([m[2] for m in meta]),
             config=json.dumps({**vars(args), "step": ck.get("step", "?"), "resolved_latent_channels": C,
                                "resolved_objective": objective, "checkpoint_interface": trained,
                                "tic_stride": tic_stride}),
             **extra)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(args.rollouts)
    pred, gt, seed, actions = d["pred"], d["gt"], d["seed"], d["actions"]
    N, H = pred.shape[:2]
    vae = build_vae(args.vae_path, args.vae_subfolder, device, args.hf_cache, latent_channels=int(pred.shape[2]),
                    scaling_factor=args.latent_scale, shift_factor=args.latent_shift)
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def dec(z):
        z = denormalize_latents(torch.from_numpy(np.asarray(z, dtype=np.float32)).to(device), args.latent_scale, args.latent_shift)
        img = vae.decode(z).sample[:, :, :240]
        return (img * 0.5 + 0.5).clamp(0, 1)

    def dec_all(z):
        """`dec` over a whole rollout, in --decode-batch chunks so a 64-frame horizon fits."""
        return torch.cat([dec(z[i:i + args.decode_batch]) for i in range(0, len(z), args.decode_batch)])

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
    cfg = json.loads(str(d["config"])) if "config" in d else {}
    tic_stride = int(cfg.get("tic_stride", 4) or 4)
    # Horizons the rollout is reported at. At tic spacing the defaults are 4, 32, 64, 128, 256 tics,
    # which is the same game time as the stride-4 rows' 1, 8, 16, 32, 64 decision steps, so the two
    # curves can be read off one axis.
    default_at = (4, 32, 64, 128, 256) if tic_stride == 1 else (8, 16, 32, 64)
    at = [int(x) for x in args.score_at.split(",")] if args.score_at else list(default_at)
    out = {"horizon": list(range(1, H + 1)), "psnr": (psnr_h / N).tolist(), "lpips": (lpips_h / N).tolist(),
           "copy_seed_psnr": (copy_h / N).tolist(), "latent_mse": (lat_h / N).tolist(), "num_rollouts": int(N),
           "tic_stride": tic_stride, "game_time_tics": [h * tic_stride for h in range(1, H + 1)],
           "scored_at": at}
    for hh in at:
        if hh <= H:
            out[f"psnr@{hh}"] = float(psnr_h[hh - 1] / N); out[f"lpips@{hh}"] = float(lpips_h[hh - 1] / N)
            out[f"copy_seed_psnr@{hh}"] = float(copy_h[hh - 1] / N)

    if args.idm:
        from train_idm import IDM, movement_probs
        ck = torch.load(args.idm, map_location="cpu", weights_only=False)
        idm = IDM(ck["num_actions"], ck["width"], ck.get("window", 8), ck.get("depth", 4)).to(device).eval(); idm.load_state_dict(ck["model"])
        act2mov = ck["act2mov"]; num_mov = len(ck["classes"])
        mov = torch.tensor([act2mov.get(a, 0) for a in range(ck["num_actions"])], device=device)
        K = idm.window
        # the IDM reads SD KL-f8 latents (its encoder opens with a 4-channel convolution), so a row in
        # another latent space reaches the judge only by decoding with its own decoder and re-encoding here
        sd = None
        channels = int(pred.shape[2])
        if args.idm_reencode_vae:
            sd = build_vae(args.idm_reencode_vae, "", device, args.hf_cache, latent_channels=4)
            print(f"NOTE: idm_* for this row round-trips through {args.vae_path or 'the stock'} decode -> "
                  f"{args.idm_reencode_vae} encode on BOTH the rollout and its real counterpart. The IDM's own "
                  "published accuracy was measured on native SD latents and is not comparable; idm_real_* is "
                  "this row's reference and is the right thing to read idm_top1 against.", flush=True)
        elif channels != 4:
            raise SystemExit(f"the rollout carries {channels}-channel latents but the IDM reads 4-channel SD "
                             "latents; pass --idm-reencode-vae to decode with --vae-path and re-encode first")
        # The IDM was trained on DECISION-frame latents four tics apart (`train_idm.WindowDataset`
        # over the stride-4 corpus, windows inside one chain), and it has a learned positional table
        # for that spacing. Handing it consecutive tics would show it a quarter of the motion it was
        # trained on, so a per-tic rollout is subsampled to its decision tics first, and the label of
        # each judged transition is the action stored at the position of the later decision frame --
        # which, on the grid, is the action held over the whole interval.
        idm_spacing = 4 if tic_stride == 1 else 1
        if tic_stride == 1 and "decision" not in d.files:
            raise SystemExit("this per-tic rollout carries no `decision` mask, so its decision tics cannot be "
                             "identified; re-run --rollout with the current code")

        def judged(n):
            if tic_stride != 1:
                return np.arange(H), seed[n, -(K - 1):]
            k = np.flatnonzero(d["decision"][n])
            sd_mask = np.flatnonzero(d["seed_decision"][n])
            return k, seed[n, sd_mask[-(K - 1):]] if len(sd_mask) >= K - 1 else None
        steps = None
        skipped = 0
        acc = None
        for n in range(N):
            k, seed_frames = judged(n)
            if seed_frames is None or len(k) == 0 or (steps is not None and len(k) != steps):
                skipped += 1
                continue
            steps = len(k)
            if acc is None:
                acc = {kk: np.zeros(steps) for kk in ("top1", "movement", "real_top1", "real_movement")}
            y = torch.from_numpy(actions[n][k]).to(device)
            for tag, frames in (("", pred[n][k]), ("real_", gt[n][k])):
                if sd is None:
                    seq = torch.from_numpy(np.concatenate([seed_frames, frames], axis=0).astype(np.float32)).to(device)
                else:
                    seq = torch.cat([encode_for_idm(sd, dec_all(seed_frames), device, args.decode_batch),
                                     encode_for_idm(sd, dec_all(frames), device, args.decode_batch)])
                logits = idm.predict_sequence(seq)[-steps:]
                acc[tag + "top1"] += (logits.argmax(-1) == y).cpu().numpy()
                acc[tag + "movement"] += (movement_probs(logits, mov, num_mov).argmax(-1) == mov[y]).cpu().numpy()
        scored = N - skipped
        out["idm_spacing_tics"] = idm_spacing * tic_stride if tic_stride != 1 else idm_spacing
        out["idm_rollouts_scored"] = int(scored)
        out["idm_rollouts_skipped"] = int(skipped)
        if acc is None or scored == 0:
            raise SystemExit("no rollout had a usable decision-tic subsequence for the IDM")
        for k, v in acc.items():
            out[f"idm_{k}"] = (v / scored).tolist(); out[f"idm_{k}_mean"] = float(v.mean() / scored)
        out["idm_val_top1"] = ck.get("val_top1"); out["idm_val_movement"] = ck.get("val_movement")
        out["idm_majority_baseline"] = ck.get("val_metrics", {}).get("majority_baseline")
        out["idm_reencode_vae"] = args.idm_reencode_vae or None
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(args.out_dir, "drift.json"), "w"), indent=1)
    if clips_pred:
        np.savez_compressed(os.path.join(args.out_dir, "clips_u8.npz"), pred=np.stack(clips_pred), gt=np.stack(clips_gt))
        if tic_stride == 1:
            # FVD at both spacings. A 16-frame clip of consecutive tics is 0.46 s of game time; a
            # 16-frame clip of every fourth tic is 1.83 s, which is what a stride-4 row's 16-frame
            # clip covers. Comparing FVD across rows needs clips of the same game duration, so both
            # are written and `after_nexttic.sh` scores both.
            np.savez_compressed(os.path.join(args.out_dir, "clips_u8_stride4.npz"),
                                pred=np.stack(clips_pred)[:, 3::4], gt=np.stack(clips_gt)[:, 3::4])
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}, indent=1))


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout", action="store_true"); p.add_argument("--score", action="store_true")
    p.add_argument("--ckpt"); p.add_argument("--backbone", choices=list(BACKBONES)); p.add_argument("--use-ema", action="store_true")
    p.add_argument("--objective", choices=["auto", "v", "eps"], default="auto",
                   help="auto reads the parameterization the checkpoint was trained in (v for every pre-grid row)")
    p.add_argument("--latent-channels", type=int, default=0, help="0 takes the backbone's own (4 for the SD KL-f8 rows, 16 for sd35)")
    p.add_argument("--context-frames", type=int, default=32); p.add_argument("--num-actions", type=int, default=29)
    p.add_argument("--noise-buckets", type=int, default=10); p.add_argument("--infer-noise", type=float, default=0.0); p.add_argument("--train-noise-max", type=float, default=0.7)
    p.add_argument("--latents-dir"); p.add_argument("--split"); p.add_argument("--subset", default="val")
    p.add_argument("--num-rollouts", type=int, default=256)
    p.add_argument("--horizon", type=int, default=64,
                   help="frames to roll out, i.e. TICS for a next-tic model and decisions for a stride-4 one; "
                        "256 tics is the same 7.31 s of game time as 64 decision steps")
    p.add_argument("--tic-stride", type=int, choices=[1, 4], default=None,
                   help="assert the checkpoint's frame spacing; read from the checkpoint otherwise")
    p.add_argument("--action-history", type=int, default=None,
                   help="assert the checkpoint's executed-control history length; read from the checkpoint otherwise")
    p.add_argument("--score-at", default="",
                   help="comma-separated horizons to report, in frames; the default is 4,32,64,128,256 for a "
                        "per-tic rollout and 8,16,32,64 for a decision-spaced one")
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--steps", type=int, default=50); p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--sd-path", default="CompVis/stable-diffusion-v1-4"); p.add_argument("--pixart-path", default=PIXART_DEFAULT)
    p.add_argument("--unidiffuser-path", default=UNIDIFFUSER_DEFAULT); p.add_argument("--sd35-path", default=SD35_DEFAULT)
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out")
    p.add_argument("--rollouts"); p.add_argument("--idm", default=""); p.add_argument("--vae-path", default="")
    p.add_argument("--vae-subfolder", default="", help="subfolder inside --vae-path (e.g. vae for a full pipeline repo)")
    p.add_argument("--latent-scale", type=float, default=LATENT_SCALE, help="scaling_factor the corpus was encoded with")
    p.add_argument("--latent-shift", type=float, default=None, help="shift_factor the corpus was encoded with (SD 3.5: 0.0609)")
    p.add_argument("--idm-reencode-vae", nargs="?", const=VAE_NAME, default="",
                   help="decode the rollout with --vae-path and re-encode with this SD 1.x encoder before the "
                        f"IDM (bare flag = {VAE_NAME}); required when the rollout is not in the IDM's own "
                        "4-channel latent space, as for the 16-channel sd35 row")
    p.add_argument("--decode-batch", type=int, default=16); p.add_argument("--save-clips", type=int, default=64); p.add_argument("--out-dir")
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    if a.rollout:
        do_rollout(a)
    if a.score:
        do_score(a)
