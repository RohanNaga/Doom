"""
Evaluation for the video-pretrained row (SkyReels-V2 DF 1.3B on Wan-VAE latents).

Produces the same numbers, under the same protocol, as `eval_tf.py` and `rollout_eval.py`
produce for the DiT / U-Net / PixArt rows, so the four rows land in one table. The output files
are `metrics.json` + `per_window.csv` (teacher-forced) and `drift.json` + `clips_u8.npz`
(rollout, the latter consumed by `fvd.py`), with the same keys.

Two things make this row different, and both are handled here rather than by changing the
corpus or the metric:

1. **One latent frame is four tics.** A Wan latent frame at index j>=1 of a chain compresses the
   four tics of decision j-1's execution (`encode_wan.py`, scheme `wan-chain-v2`), and its LAST
   decoded tic is the next decision tic — the same frame the SD rows score. So the pixel we
   score is the last of the four frames the target latent decodes to, and the raw reference is
   the parquet frame at that latent frame's `decision_tic`.

2. **The Wan VAE decoder is causal and streaming.** `AutoencoderKLWan._decode`
   (`autoencoder_kl_wan.py:1187-1212`, diffusers 0.40.0) walks latent frames one at a time
   through a persistent `_feat_map` cache, with `first_chunk=True` only for frame 0 (which
   yields one pixel frame instead of four). A latent frame's pixels therefore depend on every
   latent frame before it, and decoding the target frame ALONE is wrong. We decode the whole
   window — context latents followed by the prediction — and keep the last decoded frame.
   `WanStreamDecoder` below is that loop, lifted so we can throw away the 3-in-4 tics we do not
   score instead of materialising `1 + 4F` frames (this is what `encode_wan.WanChainEncoder`
   does for the encoder). `--decode-ctx-frames K` truncates the decoded context to the last K
   latent frames; it is exact once K exceeds the decoder's temporal receptive field, and
   `--check-truncation` measures that receptive field on the corpus at hand instead of
   assuming it.

The copy-last baseline is the last decoded CONTEXT frame, read out of the same decode pass, so
it carries exactly the same VAE and the same cache state as the prediction. The VAE ceiling is
a second pass with the true target latent in place of the prediction.

Sampling follows the training objective, read from the checkpoint's `args`:

* `flow` — the checkpoint's native rectified flow. Euler over the scheduler's own shifted
  sigma grid (`scheduling_unipc_multistep.py:428-441`): `sigmas = linspace(1, 1/1000, n+1)[:-1]`,
  shifted by `flow_shift * s / (1 + (flow_shift - 1) s)`, terminating at 0. Under
  `x_t = (1-sigma) x0 + sigma eps` with the model predicting `eps - x0`, Euler in sigma is
  exact for the probability-flow ODE, and the final step lands on x0.
* `vp-v` — `diffusion_v.VDiffusion.ddim_sample`, the identical sampler the other three rows use.

`--ctx-stabilize` is the fixed context timestep at inference (the SkyReels `addnoise_condition`
trick, `pipeline_skyreels_v2_diffusion_forcing.py:875-885`): the context is blended toward noise
at `sigma = t/1000` under flow, or `q_sample` at VP timestep t under vp-v, and the context
frames' per-frame timesteps are set to t. 0 means a clean context.

The IDM agreement metrics are the one place the corpora cross. The IDM (`train_idm.py`) was
trained on SD latents `(4, 32, 40)` from `encode_parquet.py`, so rollout frames are decoded to
pixels with the Wan VAE and re-encoded with `sd-vae-ft-mse` before `IDM.predict_sequence`; the
real continuation goes through the identical round trip, so the pred/real comparison inside this
row is symmetric. It is NOT symmetric with the other rows, whose latents never make that trip —
see the note printed at the end of a rollout run.

    # teacher-forced
    python eval_video.py --teacher-forced --ckpt results/050-skyreels/best.pt \\
        --latents-dir data/latents_wan_eval/seen --parquet-dir raw_arnold_eval/seen \\
        --split data/split_seen.json --subset val --num-windows 2048 --out-dir eval/skyreels_seen

    # rollout
    python eval_video.py --rollout --ckpt results/050-skyreels/best.pt \\
        --latents-dir data/latents_wan_eval/seen --split data/split_seen.json --subset val \\
        --num-rollouts 256 --horizon 64 --idm results/idm/idm.pt --out-dir eval/skyreels_seen_roll
"""
import argparse
import csv
import io
import json
import os
import time

import numpy as np
import torch

from diffusion_v import VDiffusion
from doomdit_utils import encode_for_idm
from video_wm import SkyReelsWorldModel, tiny_config, window_timesteps
from wan_data import LATENT_SHAPE, WanWindowDataset, load_split

HUD_ROWS = 32
FRAME_H, FRAME_W = 240, 320
PAD_TO = 256                      # encode_parquet.py / encode_wan.py pad 240 -> 256 after [-1,1]
SD_VAE_NAME = "stabilityai/sd-vae-ft-mse"
SD_LATENT_SCALE = 0.18215         # doomdit_utils.LATENT_SCALE
REPEAT = 4                        # tics per decision, and the Wan VAE's temporal factor
WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_SUBFOLDER = "vae"
# checkpoint-recorded training args this evaluator must inherit rather than guess
INHERITED = ("objective", "flow_shift", "context_frames", "num_actions", "fps_id", "raw_latents",
             "skyreels_path", "null_prompt", "hf_cache")


def psnr(a, b):
    """Per-sample PSNR on [0, 1] images; identical to eval_tf.psnr and rollout_eval.psnr."""
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


# --------------------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------------------
class WanCorpus:
    """`WanWindowDataset` plus the per-episode metadata the evaluator needs.

    `WanWindowDataset` deliberately keeps only what training needs (latents and actions), so the
    decision tics, map ids and chain ids are read here from the same `*.meta.json` files rather
    than by widening the trainer's loader.
    """

    def __init__(self, latents_dir, episode_ids, context_frames, normalize=True, verbose=True):
        names = resolve_episodes(latents_dir, episode_ids)
        self.ds = WanWindowDataset(latents_dir, names, context_frames, normalize=normalize,
                                   verbose=verbose)
        self.context_frames = context_frames
        self.meta, self._lat = {}, {}
        for name, lat, _, _ in self.ds.episodes:
            with open(os.path.join(latents_dir, f"{name}.meta.json")) as f:
                m = json.load(f)
            self.meta[name] = {"map_id": int(m.get("map_id", -1)),
                               "decision_tic": np.asarray(m["decision_tic"], dtype=np.int64),
                               "chain_id": np.asarray(m["chain_id"], dtype=np.int64),
                               "is_lone_first": np.asarray(m["is_lone_first"], dtype=np.int64),
                               "action": np.asarray(m["action"], dtype=np.int64)}
            self._lat[name] = lat
        self.latents_mean = self.ds.latents_mean
        self.latents_std = self.ds.latents_std
        self.scheme = self.ds.scheme

    def __len__(self):
        return len(self.ds)

    def locate(self, i):
        """Global window index -> (episode_name, target row in that episode's latent array)."""
        ep_idx, t = (int(v) for v in self.ds.index[i])
        return self.ds.episodes[ep_idx][0], t

    def raw_latents(self, name, lo, hi):
        """Rows [lo, hi) of one episode as float32 in the model's (normalised) space."""
        return self.ds._to_float(self._lat[name][lo:hi])


def _ep_files(latents_dir):
    from wan_data import list_episodes
    return list_episodes(latents_dir)


def resolve_episodes(latents_dir, episode_ids):
    """Map split entries onto the encoded episode names.

    The Wan corpus names episodes by parquet basename (`ep_00123`), while the SD split files
    written by `doom_data.make_split_by_map` hold integer ids. Both spellings resolve, through
    each episode's own `episode` / `episode_id` metadata, so one split file drives both corpora.
    """
    if episode_ids is None:
        return None
    by_name, by_id = {}, {}
    for name, _, meta_path in _ep_files(latents_dir):
        with open(meta_path) as f:
            m = json.load(f)
        by_name[name] = name
        by_name[str(m.get("episode", name))] = name
        if int(m.get("episode_id", -1)) >= 0:
            by_id[int(m["episode_id"])] = name
    out, missing = [], []
    for e in episode_ids:
        key = str(e)
        if key in by_name:
            out.append(by_name[key])
            continue
        try:
            i = int(e)
        except (TypeError, ValueError):
            missing.append(key)
            continue
        if i in by_id:
            out.append(by_id[i])
        elif f"ep_{i:05d}" in by_name:
            out.append(by_name[f"ep_{i:05d}"])
        else:
            missing.append(key)
    if not out:
        raise SystemExit(f"none of the {len(episode_ids)} split episodes are encoded under "
                         f"{latents_dir}; first few requested: {list(episode_ids)[:5]}")
    if missing:
        print(f"WARNING: {len(missing)} split episodes not encoded under {latents_dir} "
              f"(e.g. {missing[:5]})", flush=True)
    return sorted(set(out))


def subset_ids(split, subset):
    if subset in split:
        return split[subset]
    keys = [k for k in split if k != "meta"]
    raise SystemExit(f"--subset {subset!r} is not in the split file; available keys: {keys}")


class RawFrames:
    """Lazy per-episode access to raw frames in the parquet recordings.

    Same contract as `eval_tf.RawFrames`, keyed by the Wan corpus's episode NAME (which is the
    parquet basename) with a fall-back to the SD rows' `ep_%05d` spelling.
    """

    def __init__(self, parquet_dir):
        self.dir, self.cache = parquet_dir, {}

    def _path(self, episode):
        p = os.path.join(self.dir, f"{episode}.parquet")
        if os.path.exists(p):
            return p
        try:
            alt = os.path.join(self.dir, f"ep_{int(episode):05d}.parquet")
        except (TypeError, ValueError):
            raise FileNotFoundError(p)
        if os.path.exists(alt):
            return alt
        raise FileNotFoundError(p)

    def get(self, episode, tic):
        import pyarrow.parquet as pq
        from PIL import Image
        if episode not in self.cache:
            t = pq.read_table(self._path(episode), columns=["tic", "frame"])
            self.cache = {episode: (np.array(t["tic"]), t["frame"])}   # keep one episode resident
        tics, frames = self.cache[episode]
        i = int(np.searchsorted(tics, tic))
        if i >= len(tics) or int(tics[i]) != int(tic):
            raise KeyError(f"episode {episode}: tic {tic} not in the recording")
        return np.asarray(Image.open(io.BytesIO(frames[i].as_py())).convert("RGB"), dtype=np.uint8)


# --------------------------------------------------------------------------------------
# Wan VAE
# --------------------------------------------------------------------------------------
def load_wan_vae(vae_path, hf_cache, device):
    """The Wan 2.1 VAE in fp32 (every diffusers Wan example pins this VAE to fp32)."""
    from diffusers import AutoencoderKLWan
    kw = dict(torch_dtype=torch.float32)
    if vae_path:
        vae = AutoencoderKLWan.from_pretrained(vae_path, **kw)
    else:
        if hf_cache:
            kw["cache_dir"] = hf_cache
        vae = AutoencoderKLWan.from_pretrained(WAN_REPO, subfolder=WAN_SUBFOLDER, **kw)
    return vae.to(device).eval().requires_grad_(False)


class WanStreamDecoder:
    """`AutoencoderKLWan._decode`'s loop, one latent frame at a time.

    Yielding per latent frame instead of concatenating lets the caller keep only the tics it
    scores: a 72-frame rollout window decodes to 285 pixel frames, of which we need 64. The
    computation is identical to `vae.decode(z)` — same `post_quant_conv`, same persistent
    `_feat_map`, same `first_chunk=True` on frame 0, same clamp — and `--self-test` checks that
    against the stock call.
    """

    def __init__(self, vae):
        if getattr(vae, "use_tiling", False):
            raise ValueError("tiling must stay disabled: 320px frames would be tiled and blended")
        if getattr(vae.config, "patch_size", None) is not None:
            raise ValueError("this decoder loop does not implement unpatchify; "
                             f"vae.config.patch_size is {vae.config.patch_size}")
        self.vae = vae

    @torch.no_grad()
    def frames(self, z):
        """z (B, 16, F, 32, 40) raw latents -> yields (B, 3, n_j, 256, 320) in [-1, 1] per frame."""
        vae = self.vae
        vae.clear_cache()
        x = vae.post_quant_conv(z)
        try:
            for i in range(z.shape[2]):
                vae._conv_idx = [0]
                out = vae.decoder(x[:, :, i:i + 1], feat_cache=vae._feat_map,
                                  feat_idx=vae._conv_idx, first_chunk=(i == 0))
                yield torch.clamp(out, -1.0, 1.0)
        finally:
            vae.clear_cache()

    @torch.no_grad()
    def decision_frames(self, z, keep_from=0):
        """Last decoded tic of each latent frame from index `keep_from` on.

        Returns (B, F - keep_from, 3, 240, 320) in [0, 1]. Under `wan-chain-v2` that last tic IS
        the decision tic, which is the frame the SD rows score; the height crop to 240 matches
        `eval_tf.decode`.
        """
        out = []
        for j, chunk in enumerate(self.frames(z)):
            if j >= keep_from:
                out.append((chunk[:, :, -1, :FRAME_H] * 0.5 + 0.5).clamp(0, 1))
        return torch.stack(out, dim=1)


def denormalize(z, mean, std, normalized=True):
    """Model space -> raw Wan encoder space, the inverse of `WanWindowDataset._to_float`."""
    if not normalized:
        return z
    m = torch.as_tensor(mean, device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
    s = torch.as_tensor(std, device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
    return z * s + m


# --------------------------------------------------------------------------------------
# model and samplers
# --------------------------------------------------------------------------------------
def inherit_args(args):
    """Fold the checkpoint's recorded training args into `args` where the CLI stayed silent."""
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False) if args.ckpt else {}
    train = ck.get("args", {}) or {}
    for k in INHERITED:
        if getattr(args, k, None) is None and k in train:
            setattr(args, k, train[k])
    defaults = {"objective": "flow", "flow_shift": 1.0, "context_frames": 16, "num_actions": 29,
                "fps_id": 1, "raw_latents": False,
                "skyreels_path": "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers",
                "null_prompt": "weights/skyreels_null_prompt.pt", "hf_cache": None}
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    args.train_ctx_stabilize = train.get("ctx_stabilize")
    args.train_ctx_noise_max = train.get("ctx_noise_max")
    return ck


def load_model(args, device, ck):
    """Build the wrapper and load the checkpoint. `--use-ema` refuses to fall back silently."""
    model = SkyReelsWorldModel(num_actions=args.num_actions, skyreels_path=args.skyreels_path,
                               null_prompt=args.null_prompt, action_dropout=0.0, grad_ckpt=False,
                               cache_dir=args.hf_cache, fps_id=args.fps_id,
                               tiny=tiny_config() if args.tiny else None)
    if args.use_ema and not ck.get("ema"):
        raise SystemExit(f"--use-ema requested but {args.ckpt} carries no EMA weights "
                         f"(use a recovery checkpoint, not best.pt)")
    state = ck["ema"] if args.use_ema else ck["model"]
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    return model.to(device).eval(), ck.get("step", "?")


def flow_sigmas(steps, shift, num_train_timesteps=1000):
    """The UniPC flow grid this checkpoint's scheduler builds (`scheduling_unipc_multistep.py:428-441`)."""
    s = np.linspace(1.0, 1.0 / num_train_timesteps, steps + 1)[:-1]
    s = shift * s / (1.0 + (shift - 1.0) * s)
    if abs(float(s[0]) - 1.0) < 1e-6:
        s[0] -= 1e-6
    return np.concatenate([s, [0.0]]).astype(np.float64)   # final_sigmas_type "zero"


class Sampler:
    """Deterministic single-frame sampler for whichever objective the checkpoint was trained on."""

    def __init__(self, args, device):
        self.kind = args.objective
        self.steps = args.steps
        self.eta = args.eta
        self.device = device
        self.num_steps = 1000
        self.sigmas = flow_sigmas(args.steps, args.flow_shift) if self.kind == "flow" else None
        self.vp = VDiffusion(device=device) if self.kind == "vp-v" else None
        self.ctx_stabilize = float(args.ctx_stabilize)

    def noise_context(self, ctx, eps):
        """Corrupt the context to the fixed stabilisation time and return (ctx_n, t_ctx)."""
        b = ctx.shape[0]
        t = self.ctx_stabilize
        if self.kind == "flow":
            sigma = t / self.num_steps
            ctx_n = (1.0 - sigma) * ctx + sigma * eps
            t_ctx = torch.full((b,), t, device=ctx.device, dtype=torch.float32)
        else:
            ti = torch.full((b,), int(round(t)), device=ctx.device, dtype=torch.long)
            flat = self.vp.q_sample(ctx.flatten(1, 2), ti, eps.flatten(1, 2))
            ctx_n, t_ctx = flat.view_as(ctx), ti.float()
        return ctx_n, t_ctx

    def model_fn(self, model, ctx_n, t_ctx, action):
        L = ctx_n.shape[1]

        def fn(x, t):
            lat = torch.cat([ctx_n, x.unsqueeze(1)], dim=1).transpose(1, 2).contiguous()
            times = window_timesteps(t.float().view(-1), t_ctx, L)
            return model(lat, times, action)
        return fn

    @torch.no_grad()
    def sample(self, model, ctx_n, t_ctx, action, noise, oracle=None):
        if oracle is not None:      # harness check: a perfect predictor must hit the VAE ceiling
            return oracle
        fn = self.model_fn(model, ctx_n, t_ctx, action)
        if self.kind == "vp-v":
            return self.vp.ddim_sample(fn, noise.shape, steps=self.steps, eta=self.eta,
                                       noise=noise, device=noise.device)
        x = noise.clone()
        for i in range(len(self.sigmas) - 1):
            t = torch.full((x.shape[0],), float(self.sigmas[i]) * self.num_steps,
                           device=x.device, dtype=torch.float32)
            v = fn(x, t).float()
            x = x + float(self.sigmas[i + 1] - self.sigmas[i]) * v
        return x


def window_noise(indices, shape, seed):
    """Per-window noise, seeded by the window's own global index.

    Makes every metric independent of batch size, world size and evaluation order, which is the
    same contract `train_video.SeededCorruption` gives the validation loss.
    """
    out = torch.empty((len(indices),) + tuple(shape))
    for k, i in enumerate(indices):
        g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(i))
        out[k] = torch.randn(shape, generator=g)
    return out


# --------------------------------------------------------------------------------------
# teacher forced
# --------------------------------------------------------------------------------------
@torch.no_grad()
def do_teacher_forced(args, ck):
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    split = load_split(args.split)
    corpus = WanCorpus(args.latents_dir, subset_ids(split, args.subset), args.context_frames,
                       normalize=not args.raw_latents)
    rng = np.random.RandomState(args.seed)
    idx = np.sort(rng.choice(len(corpus), size=min(args.num_windows, len(corpus)), replace=False))
    print(f"{args.subset}: {len(corpus.ds.episodes)} episodes, {len(corpus):,} windows, "
          f"evaluating {len(idx)}, scheme {corpus.scheme}", flush=True)

    # ---- phase 1: sample, transformer only on the device -------------------------------
    model, step = load_model(args, device, ck)
    sampler = Sampler(args, device)
    L = args.context_frames
    preds, t_sample, n = [], 0.0, 0
    for b0 in range(0, len(idx), args.batch_size):
        part = idx[b0:b0 + args.batch_size]
        ctx = torch.stack([corpus.ds[int(i)][0] for i in part]).to(device)
        act = torch.tensor([int(corpus.ds[int(i)][2]) for i in part], device=device)
        ctx_eps = window_noise(part + 2_000_003, ctx.shape[1:], args.seed).to(device)
        x_noise = window_noise(part, LATENT_SHAPE, args.seed).to(device)
        ctx_n, t_ctx = sampler.noise_context(ctx, ctx_eps)
        oracle = torch.stack([corpus.ds[int(i)][1] for i in part]).to(device) if args.oracle else None
        t0 = time.time()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.autocast):
            x = sampler.sample(model, ctx_n, t_ctx, act, x_noise, oracle=oracle)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_sample += time.time() - t0
        n += len(part)
        preds.append(x.float().cpu())
        if (b0 // max(args.batch_size, 1)) % 10 == 0:
            print(f"  sampled {n}/{len(idx)} ({n / max(t_sample, 1e-9):.2f} windows/s)", flush=True)
    preds = torch.cat(preds)
    del model
    free_device(device)

    # ---- phase 2: decode and score, VAEs only on the device -----------------------------
    vae = load_wan_vae(args.wan_vae_path, args.hf_cache, device)
    dec = WanStreamDecoder(vae)
    import lpips
    lp = lpips.LPIPS(net=args.lpips_net, verbose=False).to(device).eval()
    raw = RawFrames(args.parquet_dir) if args.parquet_dir else None
    keep = L if args.decode_ctx_frames <= 0 else min(args.decode_ctx_frames, L)
    mean, std = corpus.latents_mean, corpus.latents_std
    rows, t_decode = [], 0.0
    for b0 in range(0, len(idx), args.decode_batch):
        part = idx[b0:b0 + args.decode_batch]
        ctxs, tgts, metas = [], [], []
        for i in part:
            name, t = corpus.locate(int(i))
            ctxs.append(corpus.raw_latents(name, t - keep, t))
            tgts.append(corpus.raw_latents(name, t, t + 1)[0])
            metas.append((name, t))
        ctx = torch.stack(ctxs).to(device)                              # (b, keep, 16, 32, 40)
        tgt = torch.stack(tgts).to(device)                              # (b, 16, 32, 40)
        pred = preds[b0:b0 + len(part)].to(device)   # phase 1 filled `preds` in `idx` order
        t0 = time.time()
        seq_p = seq(ctx, pred, mean, std, args.raw_latents)
        seq_t = seq(ctx, tgt, mean, std, args.raw_latents)
        got_p = dec.decision_frames(seq_p, keep_from=keep - 1)          # (b, 2, 3, 240, 320)
        got_t = dec.decision_frames(seq_t, keep_from=keep)              # (b, 1, 3, 240, 320)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_decode += time.time() - t0
        last_img, pred_img, gt_img = got_p[:, 0], got_p[:, 1], got_t[:, 0]
        lat_mse = ((pred.float() - tgt.float()) ** 2).flatten(1).mean(1)
        for k, (name, t) in enumerate(metas):
            m = corpus.meta[name]
            r = dict(index=int(part[k]), episode=name, map=m["map_id"], start=int(t - L),
                     action=int(m["action"][t]),
                     psnr_dec=float(psnr(pred_img[k:k + 1], gt_img[k:k + 1])),
                     lpips_dec=float(lp(pred_img[k:k + 1] * 2 - 1, gt_img[k:k + 1] * 2 - 1).flatten()),
                     copy_psnr_dec=float(psnr(last_img[k:k + 1], gt_img[k:k + 1])),
                     latent_mse=float(lat_mse[k]),
                     hud_psnr_dec=float(psnr(pred_img[k:k + 1, :, -HUD_ROWS:], gt_img[k:k + 1, :, -HUD_ROWS:])))
            if raw is not None:
                tic = int(m["decision_tic"][t])
                rf = torch.from_numpy(raw.get(name, tic).copy()).permute(2, 0, 1).float().div(255)[None].to(device)
                r.update(psnr_raw=float(psnr(pred_img[k:k + 1], rf)),
                         lpips_raw=float(lp(pred_img[k:k + 1] * 2 - 1, rf * 2 - 1).flatten()),
                         copy_psnr_raw=float(psnr(last_img[k:k + 1], rf)),
                         vae_psnr=float(psnr(gt_img[k:k + 1], rf)),
                         vae_lpips=float(lp(gt_img[k:k + 1] * 2 - 1, rf * 2 - 1).flatten()),
                         hud_psnr_raw=float(psnr(pred_img[k:k + 1, :, -HUD_ROWS:], rf[:, :, -HUD_ROWS:])),
                         hud_vae_psnr=float(psnr(gt_img[k:k + 1, :, -HUD_ROWS:], rf[:, :, -HUD_ROWS:])))
            rows.append(r)
        if b0 < args.save_images * args.decode_batch:
            from torchvision.utils import save_image
            save_image(torch.cat([last_img, gt_img, pred_img]).cpu(),
                       os.path.join(args.out_dir, f"batch_{b0 // max(args.decode_batch, 1):03d}_last_gt_pred.png"),
                       nrow=len(part))
        if len(rows) % (10 * args.decode_batch) < args.decode_batch:
            print(f"  {len(rows)}/{len(idx)} psnr_dec={np.mean([x['psnr_dec'] for x in rows]):.2f}", flush=True)

    keys = [k for k in rows[0] if k not in ("index", "episode", "map", "start", "action")]

    def agg(k):
        v = np.array([r[k] for r in rows], dtype=np.float64)
        v = v[~np.isnan(v)]
        return ({"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v))), "n": int(len(v))}
                if len(v) else None)

    summary = {k: agg(k) for k in keys}
    summary["per_map"] = {str(m): {k: float(np.mean([r[k] for r in rows if r["map"] == m]))
                                   for k in ("psnr_dec", "lpips_dec")}
                          for m in sorted(set(r["map"] for r in rows))}
    summary["sampling_frames_per_s"] = n / max(t_sample, 1e-9)
    summary["decode_frames_per_s"] = len(rows) / max(t_decode, 1e-9)
    summary["config"] = {**vars(args), "step": step, "scheme": corpus.scheme,
                         "decode_ctx_frames": keep, "per_window_seeds": True,
                         "device": str(device)}
    json.dump(summary, open(os.path.join(args.out_dir, "metrics.json"), "w"), indent=1)
    with open(os.path.join(args.out_dir, "per_window.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "per_map")}, indent=1))
    return summary


def seq(ctx, tgt, mean, std, raw_latents):
    """Context latents plus one target frame as a decodable (B, 16, F, 32, 40) raw-space video."""
    z = torch.cat([ctx, tgt.unsqueeze(1)], dim=1).transpose(1, 2).contiguous()
    return denormalize(z, mean, std, normalized=not raw_latents)


def free_device(device):
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------------------
def collect_rollout_windows(corpus, L, H, n, seed):
    """Seeded (episode, start) picks with L context frames and H future frames inside one chain.

    Mirrors `rollout_eval.collect_rollout_windows`: uniform over episodes, then uniform over the
    legal starts of that episode. The lone leading frame of a chain is excluded from the context,
    the same rule `WanWindowDataset` applies, because its latent statistics come from the
    encoder's single-frame path.
    """
    from wan_data import chain_segments
    eps = []
    for name, lat, _, _ in corpus.ds.episodes:
        m = corpus.meta[name]
        starts = []
        for a, b in chain_segments(m["chain_id"]):
            lo, hi = a + 1 + L, b - H          # t = index of the first predicted frame
            if hi > lo:
                starts += list(range(lo, hi + 1))
        if starts:
            eps.append((name, lat, np.array(starts, dtype=np.int64)))
    if not eps:
        raise SystemExit(f"no chain in {corpus.ds} is long enough for L={L} + H={H}")
    rng = np.random.RandomState(seed)
    picks = []
    for _ in range(n):
        name, lat, starts = eps[rng.randint(len(eps))]
        picks.append((name, int(starts[rng.randint(len(starts))])))
    return picks


@torch.no_grad()
def do_rollout(args, ck):
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    split = load_split(args.split)
    corpus = WanCorpus(args.latents_dir, subset_ids(split, args.subset), args.context_frames,
                       normalize=not args.raw_latents)
    L, H = args.context_frames, args.horizon
    picks = collect_rollout_windows(corpus, L, H, args.num_rollouts, args.seed)
    N = len(picks)
    print(f"{args.subset}: {N} rollouts, horizon {H}, L={L}, scheme {corpus.scheme}", flush=True)

    # ---- phase 1: autoregressive sampling, transformer only ----------------------------
    model, step = load_model(args, device, ck)
    sampler = Sampler(args, device)
    pred = np.lib.format.open_memmap(os.path.join(args.out_dir, "rollout_pred.npy"), mode="w+",
                                     dtype=np.float16, shape=(N, H) + LATENT_SHAPE)
    gt = np.lib.format.open_memmap(os.path.join(args.out_dir, "rollout_gt.npy"), mode="w+",
                                   dtype=np.float16, shape=(N, H) + LATENT_SHAPE)
    seed_lat = np.lib.format.open_memmap(os.path.join(args.out_dir, "rollout_seed.npy"), mode="w+",
                                         dtype=np.float16, shape=(N, L) + LATENT_SHAPE)
    actions = np.zeros((N, H), dtype=np.int64)
    t0, t_step, n_steps = time.time(), 0.0, 0
    for b0 in range(0, N, args.batch_size):
        chunk = picks[b0:b0 + args.batch_size]
        b = len(chunk)
        ctx = torch.stack([corpus.raw_latents(name, t - L, t) for name, t in chunk]).to(device)
        seed_lat[b0:b0 + b] = ctx.cpu().numpy().astype(np.float16)
        for k, (name, t) in enumerate(chunk):
            gt[b0 + k] = corpus.raw_latents(name, t, t + H).numpy().astype(np.float16)
            actions[b0 + k] = corpus.meta[name]["action"][t:t + H]
        for h in range(H):
            act = torch.from_numpy(actions[b0:b0 + b, h]).to(device)
            base = np.array([b0 + k for k in range(b)]) * (H + 1) + h
            ctx_eps = window_noise(base + 3_000_003, ctx.shape[1:], args.seed).to(device)
            x_noise = window_noise(base, LATENT_SHAPE, args.seed).to(device)
            ctx_n, t_ctx = sampler.noise_context(ctx, ctx_eps)
            oracle = torch.from_numpy(np.asarray(gt[b0:b0 + b, h], dtype=np.float32)).to(device) \
                if args.oracle else None
            t1 = time.time()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.autocast):
                x = sampler.sample(model, ctx_n, t_ctx, act, x_noise, oracle=oracle)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_step += time.time() - t1
            n_steps += b
            pred[b0:b0 + b, h] = x.float().cpu().numpy().astype(np.float16)
            ctx = torch.cat([ctx[:, 1:], x.float().unsqueeze(1)], dim=1)
        print(f"  {b0 + b}/{N} rollouts, {(time.time() - t0) / (b0 + b):.1f} s each", flush=True)
    del model
    free_device(device)
    json.dump({"episode": [p[0] for p in picks], "start": [p[1] for p in picks],
               "map": [corpus.meta[p[0]]["map_id"] for p in picks],
               "config": {**vars(args), "step": step, "scheme": corpus.scheme}},
              open(os.path.join(args.out_dir, "rollout_meta.json"), "w"), indent=1)
    np.save(os.path.join(args.out_dir, "rollout_actions.npy"), actions)

    # ---- phase 2: decode, score, IDM ---------------------------------------------------
    out = score_rollouts(args, corpus, picks, pred, gt, seed_lat, actions, device)
    out["sampling_steps_per_s"] = n_steps / max(t_step, 1e-9)
    out["config"] = {**vars(args), "step": step, "scheme": corpus.scheme, "per_window_seeds": True,
                     "device": str(device)}
    json.dump(out, open(os.path.join(args.out_dir, "drift.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if not isinstance(v, (list, dict))}, indent=1))
    print("NOTE: idm_* for this row round-trips through Wan-decode -> SD-encode on BOTH the "
          "predicted and the real branch, because the IDM reads SD latents. The other rows' "
          "latents never make that trip; idm_real_* is this row's own ceiling and is the right "
          "thing to compare against idm_top1.", flush=True)
    return out


@torch.no_grad()
def score_rollouts(args, corpus, picks, pred, gt, seed_lat, actions, device):
    """Drift curves, copy-seed baseline, IDM agreement and the FVD clips."""
    import lpips
    N, H = pred.shape[:2]
    L = args.context_frames
    vae = load_wan_vae(args.wan_vae_path, args.hf_cache, device)
    dec = WanStreamDecoder(vae)
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    mean, std = corpus.latents_mean, corpus.latents_std
    keep = L if args.decode_ctx_frames <= 0 else min(args.decode_ctx_frames, L)

    idm = sd = ick = None
    K = 1
    if args.idm:
        from train_idm import IDM
        ick = torch.load(args.idm, map_location="cpu", weights_only=False)
        idm = IDM(ick["num_actions"], ick["width"], ick.get("window", 8), ick.get("depth", 4)).to(device).eval()
        idm.load_state_dict(ick["model"])
        mov = torch.tensor([ick["act2mov"].get(a, 0) for a in range(ick["num_actions"])], device=device)
        num_mov = len(ick["classes"])
        K = idm.window
        from diffusers.models import AutoencoderKL
        sd = AutoencoderKL.from_pretrained(args.sd_vae_path, cache_dir=args.hf_cache).to(device).eval()
        sd.requires_grad_(False)
        if keep < K - 1:
            print(f"WARNING: only {keep} context latent frames are decoded but the IDM window "
                  f"wants {K - 1} real frames before the first transition", flush=True)

    # one causal decode per branch. `n_pre` context frames are kept out of the same pass: the
    # last of them is the copy-seed reference, and all of them feed the windowed IDM, so nothing
    # is decoded twice and the context pixels are bit-identical between the two branches.
    n_pre = min(keep, max(1, K - 1))
    kf = keep - n_pre

    psnr_h = np.zeros(H); lpips_h = np.zeros(H); copy_h = np.zeros(H); lat_h = np.zeros(H)
    acc_h = {k: np.zeros(H) for k in ("top1", "movement", "real_top1", "real_movement")}
    clips_pred, clips_gt = [], []
    for n in range(N):
        ctx = torch.from_numpy(np.asarray(seed_lat[n, -keep:], dtype=np.float32))[None].to(device)
        p_lat = torch.from_numpy(np.asarray(pred[n], dtype=np.float32))[None].to(device)
        g_lat = torch.from_numpy(np.asarray(gt[n], dtype=np.float32))[None].to(device)
        zp = denormalize(torch.cat([ctx, p_lat], 1).transpose(1, 2).contiguous(), mean, std, not args.raw_latents)
        zg = denormalize(torch.cat([ctx, g_lat], 1).transpose(1, 2).contiguous(), mean, std, not args.raw_latents)
        fp = dec.decision_frames(zp, keep_from=kf)[0]                # (n_pre + H, 3, 240, 320)
        fg = dec.decision_frames(zg, keep_from=kf)[0]
        pre, P, G = fp[:n_pre], fp[n_pre:], fg[n_pre:]
        last = pre[-1:]
        for h0 in range(0, H, args.decode_batch):
            sl = slice(h0, min(h0 + args.decode_batch, H))
            p, g = P[sl], G[sl]
            psnr_h[sl] += psnr(p, g).cpu().numpy()
            lpips_h[sl] += lp(p * 2 - 1, g * 2 - 1).flatten().cpu().numpy()
            copy_h[sl] += psnr(last.expand_as(g), g).cpu().numpy()
        lat_h += ((np.asarray(pred[n], dtype=np.float32) - np.asarray(gt[n], dtype=np.float32)) ** 2
                  ).reshape(H, -1).mean(1)
        if n < args.save_clips:
            clips_pred.append((P * 255).round().byte().cpu().numpy())
            clips_gt.append((G * 255).round().byte().cpu().numpy())
        if idm is not None:
            # the windowed IDM needs K-1 real frames before the first transition it judges, so it
            # sees both sides of every transition; the real branch runs the identical procedure
            y = torch.from_numpy(actions[n]).to(device)
            pre_z = sd_encode(sd, pre[-(K - 1):], device, args.decode_batch)
            for tag, frames in (("", P), ("real_", G)):
                z = torch.cat([pre_z, sd_encode(sd, frames, device, args.decode_batch)])
                logits = idm.predict_sequence(z)[-H:]
                acc_h[tag + "top1"] += (logits.argmax(-1) == y).cpu().numpy()
                acc_h[tag + "movement"] += (movement_of(logits, mov, num_mov) == mov[y]).cpu().numpy()
        if (n + 1) % 32 == 0:
            print(f"  scored {n + 1}/{N}", flush=True)

    out = {"horizon": list(range(1, H + 1)), "psnr": (psnr_h / N).tolist(),
           "lpips": (lpips_h / N).tolist(), "copy_seed_psnr": (copy_h / N).tolist(),
           "latent_mse": (lat_h / N).tolist(), "num_rollouts": int(N)}
    for hh in (8, 16, 32, 64):
        if hh <= H:
            out[f"psnr@{hh}"] = float(psnr_h[hh - 1] / N)
            out[f"lpips@{hh}"] = float(lpips_h[hh - 1] / N)
    if idm is not None:
        for k, v in acc_h.items():
            out[f"idm_{k}"] = (v / N).tolist()
            out[f"idm_{k}_mean"] = float(v.mean() / N)
        out["idm_val_top1"] = ick.get("val_top1")
        out["idm_val_movement"] = ick.get("val_movement")
        out["idm_majority_baseline"] = ick.get("val_metrics", {}).get("majority_baseline")
    if clips_pred:
        np.savez_compressed(os.path.join(args.out_dir, "clips_u8.npz"),
                            pred=np.stack(clips_pred), gt=np.stack(clips_gt))
    return out


def movement_of(logits, mov, num_mov):
    from train_idm import movement_probs
    return movement_probs(logits, mov, num_mov).argmax(-1)


def sd_encode(sd, frames, device, batch):
    """Decoded frames (T, 3, 240, 320) in [0, 1] -> the SD latents the IDM was trained on.

    One line over `doomdit_utils.encode_for_idm`, which `rollout_eval.py --score` uses for the
    same round trip on the 16-channel row; the pad-after-normalisation contract lives there.
    """
    return encode_for_idm(sd, frames, device, batch, PAD_TO, SD_LATENT_SCALE)


# --------------------------------------------------------------------------------------
# self-test on the decoder
# --------------------------------------------------------------------------------------
@torch.no_grad()
def check_truncation(args):
    """Prove the streaming decode equals `vae.decode`, and measure the truncation error.

    Prints, for each K, the max absolute difference between the last decoded frame of a
    K-context decode and of the full-context decode. The K at which that difference reaches
    floating-point noise is the decoder's temporal receptive field, and `--decode-ctx-frames`
    can be set there.
    """
    device = torch.device(args.device)
    vae = load_wan_vae(args.wan_vae_path, args.hf_cache, device)
    dec = WanStreamDecoder(vae)
    F = args.context_frames + 1
    g = torch.Generator().manual_seed(args.seed)
    z = torch.randn((1, LATENT_SHAPE[0], F) + LATENT_SHAPE[1:], generator=g).to(device)
    ref = vae.decode(z).sample
    t0 = time.time()
    mine = torch.cat(list(dec.frames(z)), dim=2)
    dt = time.time() - t0
    print(f"[stream] shapes {tuple(ref.shape)} vs {tuple(mine.shape)}, "
          f"max|diff| {float((ref - mine).abs().max()):.3e}")
    print(f"[stream] {mine.shape[2]} pixel frames from {F} latent frames in {dt:.1f}s "
          f"= {mine.shape[2] / dt:.1f} frames/s on {device} — the throughput the whole "
          f"evaluation budget is set by")
    # causality: appending the target cannot change the context's own decoded frames
    ctx_only = dec.decision_frames(z[:, :, :F - 1], keep_from=F - 2)[:, -1]
    with_tgt = dec.decision_frames(z, keep_from=F - 2)[:, 0]
    print(f"[causal] last context frame with and without the target appended: "
          f"max|diff| {float((ctx_only - with_tgt).abs().max()):.3e} "
          f"(this is the copy-last reference)")
    full = dec.decision_frames(z, keep_from=F - 1)[:, -1]
    grid = sorted({k for k in (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, args.context_frames)
                   if 1 <= k <= args.context_frames})
    for K in grid:
        part = dec.decision_frames(z[:, :, F - 1 - K:], keep_from=K)[:, -1]
        print(f"[truncate] K={K:2d}  max|diff| {float((full - part).abs().max()):.3e}  "
              f"psnr {float(psnr(part, full)):.2f} dB")


# --------------------------------------------------------------------------------------
def main(args):
    if args.teacher_forced == args.rollout and not args.check_truncation:
        raise SystemExit("choose exactly one of --teacher-forced / --rollout (or --check-truncation)")
    ck = inherit_args(args)
    if args.train_ctx_stabilize is not None and float(args.train_ctx_stabilize) != float(args.ctx_stabilize):
        print(f"NOTE: the checkpoint recorded --ctx-stabilize {args.train_ctx_stabilize}, "
              f"this run uses {args.ctx_stabilize}", flush=True)
    if args.objective == "flow" and args.eta != 0:
        print(f"NOTE: --eta {args.eta} is a DDIM knob and is ignored under --objective flow", flush=True)
    if args.check_truncation:
        return check_truncation(args)
    if not (args.latents_dir and args.split and args.ckpt):
        raise SystemExit("--latents-dir, --split and --ckpt are required for --teacher-forced / --rollout")
    if args.teacher_forced:
        return do_teacher_forced(args, ck)
    return do_rollout(args, ck)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-forced", action="store_true")
    p.add_argument("--rollout", action="store_true")
    p.add_argument("--check-truncation", action="store_true",
                   help="decoder self-test: streaming vs stock decode, and the truncation error per K")
    p.add_argument("--ckpt", default="", help="best.pt or a recovery checkpoint from train_video.py")
    p.add_argument("--use-ema", action="store_true",
                   help="use the EMA weights; refuses rather than falling back when there are none")
    # inherited from the checkpoint's args unless given here
    p.add_argument("--objective", choices=["flow", "vp-v"], default=None)
    p.add_argument("--flow-shift", type=float, default=None)
    p.add_argument("--context-frames", type=int, default=None)
    p.add_argument("--num-actions", type=int, default=None)
    p.add_argument("--fps-id", type=int, default=None, choices=[0, 1])
    p.add_argument("--raw-latents", action="store_const", const=True, default=None)
    p.add_argument("--skyreels-path", default=None)
    p.add_argument("--null-prompt", default=None)
    p.add_argument("--hf-cache", default=None)
    # corpus
    p.add_argument("--latents-dir", default="", help="encode_wan.py output directory")
    p.add_argument("--parquet-dir", default="", help="raw recordings for lossless-frame scoring")
    p.add_argument("--split", default="")
    p.add_argument("--subset", default="val")
    # sampling
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0, help="vp-v only; non-zero breaks per-window determinism")
    p.add_argument("--ctx-stabilize", type=float, default=0.0,
                   help="fixed context timestep at inference (SkyReels addnoise_condition); 0 is clean")
    # sizes
    p.add_argument("--num-windows", type=int, default=2048)
    p.add_argument("--num-rollouts", type=int, default=256)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8, help="windows sampled through the transformer at once")
    p.add_argument("--decode-batch", type=int, default=4, help="windows decoded through the Wan VAE at once")
    p.add_argument("--decode-ctx-frames", type=int, default=0,
                   help="decode only the last K context latent frames; 0 (default) decodes them all, "
                        "which is exact. Verify a smaller K with --check-truncation first")
    # scoring
    p.add_argument("--idm", default="", help="idm.pt from train_idm.py; enables the agreement metrics")
    p.add_argument("--sd-vae-path", default=SD_VAE_NAME, help="encoder the IDM's latents came from")
    p.add_argument("--wan-vae-path", default="", help="local Wan vae directory; default pulls the repo")
    p.add_argument("--lpips-net", default="alex")
    p.add_argument("--save-images", type=int, default=3)
    p.add_argument("--save-clips", type=int, default=64)
    # runtime
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--autocast", dest="autocast", action="store_true", default=True)
    p.add_argument("--no-autocast", dest="autocast", action="store_false")
    p.add_argument("--tiny", action="store_true", help="tiny random transformer, for CPU gates")
    p.add_argument("--oracle", action="store_true",
                   help="harness check: substitute the true target latent for the sample. "
                        "psnr_raw must then equal vae_psnr exactly, which is what proves the "
                        "decode-and-score path adds nothing of its own")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    main(p.parse_args())
