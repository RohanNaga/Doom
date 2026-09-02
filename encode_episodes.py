"""
Encode the VizDoom source dataset into per-episode SD-VAE latents.

Rebuilds the frame-to-latent step that produced `ep_XXXX_latents.npy` /
`ep_XXXX_actions.npy` (the original script was never committed). Streams the
Hugging Face parquet shards one at a time so the 48 GB source never sits on disk
at once, decodes the JPEG frames, resizes 320x240 -> 160x120, encodes with
stabilityai/sd-vae-ft-mse, and scales by 0.18215. Output is 5.9 GB for 500
episodes.

Two-phase, restartable:
  phase 1  each shard -> parts/ep_XXXX_shard_SS.npz  (latents, actions, step_ids[, frames])
  phase 2  merge parts per episode, sort by step_id, write ep_XXXX_latents.npy + actions

Usage (on a GPU box):
    python encode_episodes.py --out-dir data/episodes --frames-for data/split.json
    python encode_episodes.py --out-dir data/episodes --merge-only

Known unknowns vs the original pipeline (ask Keerthana; see RESEARCH_CONTEXT.md):
resize filter, posterior mean vs sample. Both are flags here; defaults are
bilinear + mean.
"""
import argparse
import glob
import io
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from doom_data import LATENT_SHAPE, load_split
from doomdit_utils import LATENT_SCALE, load_vae

DATASET = "arnaudstiegler/vizdoom-500-episodes-skipframe-4-lvl5"
NUM_SHARDS = 97
TARGET_HW = (120, 160)
RESAMPLE = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS, "area": Image.BOX, "nearest": Image.NEAREST}


def shard_name(i):
    return f"data/train-{i:05d}-of-{NUM_SHARDS:05d}.parquet"


def decode_frame(jpeg_bytes, resample):
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    if img.size != (TARGET_HW[1], TARGET_HW[0]):
        img = img.resize((TARGET_HW[1], TARGET_HW[0]), resample)
    return np.asarray(img, dtype=np.uint8)  # (120, 160, 3)


@torch.no_grad()
def encode_frames(vae, frames_u8, device, posterior, dtype):
    """(B,120,160,3) uint8 -> (B,4,15,20) float16 scaled latents."""
    x = torch.from_numpy(frames_u8).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=(device == "cuda" and dtype != torch.float32)):
        dist = vae.encode(x).latent_dist
        z = dist.mean if posterior == "mean" else dist.sample()
    z = (z.float() * LATENT_SCALE).cpu().numpy().astype(np.float16)
    assert z.shape[1:] == LATENT_SHAPE, z.shape
    return z


def process_shard(shard_idx, args, vae, device, frame_eps, pool):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    done_marker = os.path.join(args.parts_dir, f"shard_{shard_idx:02d}.done")
    if os.path.exists(done_marker):
        return 0
    t0 = time.time()
    path = hf_hub_download(DATASET, shard_name(shard_idx), repo_type="dataset",
                           cache_dir=args.hf_cache)
    pf = pq.ParquetFile(path)
    buckets = {}  # episode_id -> dict of lists
    resample = RESAMPLE[args.resize]
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    n_rows = 0
    for batch in pf.iter_batches(batch_size=args.batch_size, columns=["episode_id", "frames", "actions", "step_ids"]):
        ep = batch.column("episode_id").to_numpy()
        act = batch.column("actions").to_numpy()
        sid = batch.column("step_ids").to_numpy()
        raw = batch.column("frames").to_pylist()
        frames = np.stack(list(pool.map(lambda b: decode_frame(b, resample), raw)), axis=0)
        lat = encode_frames(vae, frames, device, args.posterior, dtype)
        for e in np.unique(ep):
            m = ep == e
            b = buckets.setdefault(int(e), {"latents": [], "actions": [], "step_ids": [], "frames": []})
            b["latents"].append(lat[m]); b["actions"].append(act[m]); b["step_ids"].append(sid[m])
            if int(e) in frame_eps:
                b["frames"].append(frames[m])
        n_rows += len(ep)
    for e, b in buckets.items():
        out = {"latents": np.concatenate(b["latents"]), "actions": np.concatenate(b["actions"]).astype(np.int64),
               "step_ids": np.concatenate(b["step_ids"]).astype(np.int64)}
        if b["frames"]:
            out["frames"] = np.concatenate(b["frames"])
        np.savez(os.path.join(args.parts_dir, f"ep_{e:04d}_shard_{shard_idx:02d}.npz"), **out)
    if not args.keep_shards:
        os.remove(os.path.realpath(path))
        try:
            os.remove(path)  # the cache symlink
        except OSError:
            pass
    open(done_marker, "w").write(f"{n_rows}\n")
    print(f"shard {shard_idx:02d}: {n_rows} rows, {len(buckets)} episodes, {time.time() - t0:.0f}s", flush=True)
    return n_rows


def merge_parts(args):
    parts = sorted(glob.glob(os.path.join(args.parts_dir, "ep_*_shard_*.npz")))
    by_ep = {}
    for p in parts:
        e = int(os.path.basename(p).split("_")[1])
        by_ep.setdefault(e, []).append(p)
    summary = {}
    for e, paths in sorted(by_ep.items()):
        chunks = [np.load(p) for p in paths]
        sid = np.concatenate([c["step_ids"] for c in chunks])
        order = np.argsort(sid, kind="stable")
        sid = sid[order]
        if not np.array_equal(sid, np.arange(len(sid))):
            raise RuntimeError(f"episode {e}: step_ids not contiguous 0..{len(sid) - 1} (got {sid[:5]}..{sid[-5:]})")
        lat = np.concatenate([c["latents"] for c in chunks])[order]
        act = np.concatenate([c["actions"] for c in chunks])[order]
        np.save(os.path.join(args.out_dir, f"ep_{e:04d}_latents.npy"), lat.astype(np.float16))
        np.save(os.path.join(args.out_dir, f"ep_{e:04d}_actions.npy"), act.astype(np.int64))
        if all("frames" in c for c in chunks):
            fr = np.concatenate([c["frames"] for c in chunks])[order]
            np.save(os.path.join(args.out_dir, f"ep_{e:04d}_frames.npy"), fr.astype(np.uint8))
        summary[e] = int(len(sid))
    total = sum(summary.values())
    print(f"merged {len(summary)} episodes, {total:,} frames, "
          f"{total - 4 * len(summary):,} training windows")
    return summary


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    args.parts_dir = os.path.join(args.out_dir, "parts")
    os.makedirs(args.parts_dir, exist_ok=True)
    frame_eps = set()
    if args.frames_for:
        # Raw frames cost ~58 KB each (2.8 GB per ten episodes), so cap how many
        # episodes keep them; the metric harness uses them only for the VAE ceiling
        # and raw-reference PSNR, which a subset estimates well.
        frame_eps = set(sorted(load_split(args.frames_for)[args.frames_subset])[:args.frames_max_episodes])
        print(f"saving uint8 frames for {len(frame_eps)} '{args.frames_subset}' episodes: {sorted(frame_eps)}")

    if not args.merge_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vae = load_vae(device)
        shards = range(args.shard_start, args.shard_end)
        with ThreadPoolExecutor(args.decode_threads) as pool:
            for s in shards:
                process_shard(s, args, vae, device, frame_eps, pool)

    summary = merge_parts(args)
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git = "?"
    meta = {"dataset": DATASET, "args": vars(args), "git": git, "episodes": summary,
            "latent_scale": LATENT_SCALE, "target_hw": TARGET_HW}
    with open(os.path.join(args.out_dir, "encode_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/episodes")
    p.add_argument("--hf-cache", default=None, help="where shards are downloaded (deleted after use)")
    p.add_argument("--shard-start", type=int, default=0)
    p.add_argument("--shard-end", type=int, default=NUM_SHARDS)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--decode-threads", type=int, default=8)
    p.add_argument("--resize", choices=list(RESAMPLE), default="bilinear")
    p.add_argument("--posterior", choices=["mean", "sample"], default="mean")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--frames-for", default="", help="split.json; also save uint8 frames for one subset")
    p.add_argument("--frames-subset", default="val")
    p.add_argument("--frames-max-episodes", type=int, default=10)
    p.add_argument("--keep-shards", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    main(p.parse_args())
