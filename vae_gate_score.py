"""
Reconstruction ceiling of several VAE decoders on the frames the paper reports.

Every decoder sees exactly the same frames: the decoder-tune development set (frames of the
training corpus's validation episodes, the set `finetune_decoder.py` validates on) and the
target frames of the windows `eval_tf.py` scores on each reporting corpus. The metric is the
encode/decode round trip against the lossless recording, i.e. the `vae_psnr` / `vae_lpips`
ceiling row of the results tables, plus HUD-crop PSNR. Per-frame values are kept so a paired
episode-level bootstrap can compare a candidate decoder against the incumbent: windows inside
one episode are correlated, so the episode is the resampling unit.

    python vae_gate_score.py \
      --decoder tuned_sd=$D/vae_decoder_arnold_lpips/vae \
      --decoder stock_flux=Alpha-VLLM/Lumina-Image-2.0#vae \
      --decoder tuned_flux=$D/vae_decoder_flux_lpips/vae \
      --dev-in-dir $D/raw_arnold --dev-split $D/split_arnold.json \
      --corpus seen=$D/latents_arnold_eval/seen,$D/raw_arnold_eval/seen,$D/latents_arnold_eval/split_seen.json \
      --baseline tuned_sd --out-dir $D/results_spiderman/vae_gate_flux
"""
import argparse
import io
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from doom_data import LatentWindowDataset, load_split
from doomdit_utils import build_vae, latent_contract
from finetune_decoder import HUD_ROWS, cached_frames, load_frames, psnr, sample_frames, to_tensor


def corpus_targets(latents_dir, parquet_dir, split_path, subset, num_windows, context_frames, seed, stride):
    """(episode_ids, frames_u8) for the target frames of the windows eval_tf.py scores.

    Mirrors eval_tf.main: the same dataset, the same RandomState(seed).choice over its windows
    and the same target tic, so the ceiling is measured on the frames the model rows were
    measured on.
    """
    import pyarrow.parquet as pq
    split = load_split(split_path)
    ds = LatentWindowDataset(latents_dir, split[subset], context_frames)
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(ds), size=min(num_windows, len(ds)), replace=False))
    want = {}
    for gi in idx:
        slot, start = ds.locate(int(gi))
        tic = ds.target_tic(int(gi))
        if tic is None:
            tic = (start + context_frames) * stride
        want.setdefault(ds.episodes[slot][0], []).append(int(tic))
    eps, frames = [], []
    for ep in sorted(want):
        t = pq.read_table(os.path.join(parquet_dir, f"ep_{ep:05d}.parquet"), columns=["tic", "frame"])
        tics, blobs = np.array(t["tic"]), t["frame"]
        for tic in want[ep]:
            i = int(np.searchsorted(tics, tic))
            frames.append(np.asarray(Image.open(io.BytesIO(blobs[i].as_py())).convert("RGB"), dtype=np.uint8))
            eps.append(ep)
    print(f"  {len(frames)} target frames from {len(want)} episodes of {latents_dir}", flush=True)
    return np.array(eps), np.stack(frames)


def dev_targets(in_dir, split_path, n, stride, frame_cache):
    """(episode_ids, frames_u8) for the fine-tune's own validation frames."""
    split = json.load(open(split_path))
    items = sample_frames(in_dir, split["val"], n, stride, 1)
    eps = np.array([int(os.path.basename(p).split("_")[1].split(".")[0]) for p, _ in items])
    tag = f"{os.path.basename(in_dir.rstrip('/'))}_{os.path.basename(split_path)}_val_{n}_s{stride}"
    frames = cached_frames(frame_cache, tag, lambda: load_frames(items))
    return eps, np.asarray(frames)


@torch.no_grad()
def score(vae, frames, device, lpips_fn, bs):
    """Per-frame PSNR, LPIPS and HUD PSNR of the encode/decode round trip, on the 320x240 crop."""
    out = {"psnr": [], "lpips": [], "hud_psnr": []}
    for i in range(0, len(frames), bs):
        x = to_tensor([frames[j] for j in range(i, min(i + bs, len(frames)))], device)
        z = vae.encode(x).latent_dist.mean
        y = vae.decode(z).sample.clamp(-1, 1)
        x, y = x[:, :, :240], y[:, :, :240]
        out["psnr"].append(psnr(x, y).cpu())
        out["hud_psnr"].append(psnr(x[:, :, -HUD_ROWS:], y[:, :, -HUD_ROWS:]).cpu())
        out["lpips"].append(lpips_fn(x, y).flatten().cpu())
    return {k: torch.cat(v).double().numpy() for k, v in out.items()}


def summarise(v):
    return {"mean": float(np.mean(v)), "sem": float(np.std(v, ddof=1) / np.sqrt(len(v))), "n": int(len(v))}


def paired_bootstrap(eps, a, b, resamples, seed):
    """Episode-level bootstrap CI for mean(a - b), the paired candidate-minus-incumbent delta.

    Episodes are resampled with replacement and every frame of a drawn episode is kept, which
    keeps the within-episode correlation of neighbouring windows in the interval.
    """
    d = np.asarray(a) - np.asarray(b)
    groups = [np.flatnonzero(eps == e) for e in np.unique(eps)]
    rng = np.random.RandomState(seed)
    draws = np.empty(resamples)
    for r in range(resamples):
        pick = rng.randint(0, len(groups), len(groups))
        draws[r] = d[np.concatenate([groups[p] for p in pick])].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"delta": float(d.mean()), "ci95": [float(lo), float(hi)], "episodes": len(groups),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    import lpips
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    sets = {}
    if args.dev_in_dir:
        sets["dev"] = dev_targets(args.dev_in_dir, args.dev_split, args.dev_frames, args.stride, args.frame_cache)
    for spec in args.corpus:
        name, rest = spec.split("=", 1)
        lat, parq, split = rest.split(",")
        sets[name] = corpus_targets(lat, parq, split, args.subset, args.num_windows,
                                    args.context_frames, args.seed, args.stride)

    per_frame, contracts, t0 = {}, {}, time.time()
    for spec in args.decoder:
        name, path = spec.split("=", 1)
        vae_id, _, sub = path.partition("#")
        vae = build_vae(vae_id, sub, device, args.cache_dir)
        contracts[name] = latent_contract(vae)
        print(f"{name}: {path} {json.dumps(contracts[name])}", flush=True)
        for s, (eps, frames) in sets.items():
            per_frame.setdefault(s, {})[name] = score(vae, frames, device, lpips_fn, args.batch_size)
            print(f"  {s}: " + json.dumps({k: round(float(np.mean(v)), 4) for k, v in per_frame[s][name].items()}), flush=True)
        del vae
        torch.cuda.empty_cache()

    names = [s.split("=", 1)[0] for s in args.decoder]
    from decoder_provenance import describe
    from doomdit_utils import VAE_NAME
    out = {"decoders": {n: s.split("=", 1)[1] for n, s in zip(names, args.decoder)},
           # which weights each column is, by content hash, and what each was tuned on: a ceiling
           # is only readable next to the decoder's training episodes (docs/REVIEW_2026-09-22.md H4)
           "decoder_provenance": {n: describe(s.split("=", 1)[1].partition("#")[0] or VAE_NAME)
                                  for n, s in zip(names, args.decoder)},
           "latent_contracts": contracts, "baseline": args.baseline, "metrics": {}, "paired": {},
           "config": vars(args), "score_seconds": time.time() - t0,
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device != "cpu" else None}
    for s in sets:
        out["metrics"][s] = {n: {k: summarise(v) for k, v in per_frame[s][n].items()} for n in names}
        out["metrics"][s]["episodes"] = int(len(np.unique(sets[s][0])))
        if args.baseline in names:
            eps = sets[s][0]
            out["paired"][s] = {n: {k: paired_bootstrap(eps, per_frame[s][n][k], per_frame[s][args.baseline][k],
                                                        args.resamples, args.seed)
                                    for k in ("psnr", "lpips", "hud_psnr")}
                                for n in names if n != args.baseline}
    json.dump(out, open(os.path.join(args.out_dir, "metrics.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(args.out_dir, "per_frame.npz"),
                        **{f"{s}__episode": sets[s][0] for s in sets},
                        **{f"{s}__{n}__{k}": v for s in sets for n in names for k, v in per_frame[s][n].items()})
    print(json.dumps({"metrics": out["metrics"], "paired": out["paired"]}, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--decoder", action="append", required=True, help="name=path_or_repo[#subfolder]")
    p.add_argument("--baseline", default="", help="decoder name the paired bootstrap subtracts")
    p.add_argument("--dev-in-dir", default="", help="parquet dir of the training corpus (development set)")
    p.add_argument("--dev-split", default="")
    p.add_argument("--dev-frames", type=int, default=2000)
    p.add_argument("--corpus", action="append", default=[], help="name=latents_dir,parquet_dir,split.json")
    p.add_argument("--subset", default="val")
    p.add_argument("--num-windows", type=int, default=2048)
    p.add_argument("--context-frames", type=int, default=32)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--frame-cache", default="")
    p.add_argument("--cache-dir", default=None, help="Hugging Face cache for repo ids")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", required=True)
    main(p.parse_args())
