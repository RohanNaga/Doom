"""
Encode per-episode parquet recordings (from record_episodes.py or record_arnold.py) into
stride-4 latents for training.

For every episode: take one frame per agent decision (tic % stride == 0), pad 320x240 to
320x256 at the image level (GameNGen), encode with the frozen sd-vae-ft-mse encoder
(posterior mean, scale 0.18215), and write
    ep_XXXXX_latents.npy   (T, 4, 32, 40) float16
    ep_XXXXX_meta.npz      action, buttons, health, ammo, kills, deaths, frags, pos_x, pos_y,
                           angle, tic, map_id, episode_id   (all length T)
plus `episodes.json` with per-episode counts and map ids. Restartable: existing outputs skip.

Usage:
    python encode_parquet.py --in-dir /sata2/.../raw_arnold --out-dir /sata2/.../latents_arnold \
        --stride 4 --batch-size 64 --device cuda:1
"""
import argparse
import glob
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from doomdit_utils import LATENT_SCALE, load_vae

PAD_TO = 256
META_COLS = ["action", "buttons", "health", "ammo", "kills", "deaths", "frags", "pos_x", "pos_y", "angle", "tic", "map_id", "episode_id"]


def decode(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)


@torch.no_grad()
def encode_batch(vae, frames_u8, device, dtype):
    x = torch.from_numpy(frames_u8).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    if x.shape[2] < PAD_TO:
        x = torch.nn.functional.pad(x, (0, 0, 0, PAD_TO - x.shape[2]))   # bottom rows, zeros (black)
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32 and x.is_cuda)):
        z = vae.encode(x).latent_dist.mean
    return (z.float() * LATENT_SCALE).cpu().numpy().astype(np.float16)


def encode_episode(path, out_dir, vae, device, dtype, stride, batch_size, pool):
    import pyarrow.parquet as pq
    ep = os.path.basename(path).replace(".parquet", "")
    out_lat = os.path.join(out_dir, f"{ep}_latents.npy")
    if os.path.exists(out_lat):
        return None
    t = pq.read_table(path)
    tic = np.array(t["tic"])
    keep = np.flatnonzero(tic % stride == 0)
    cols = {}
    for c in META_COLS:
        if c in t.schema.names:
            arr = t[c].to_numpy(zero_copy_only=False) if c != "buttons" else np.array(t["buttons"].to_pylist())
            cols[c] = arr[keep]
    frames_col = t["frame"]
    lat = []
    for i in range(0, len(keep), batch_size):
        idx = keep[i:i + batch_size]
        raw = [frames_col[int(j)].as_py() for j in idx]
        frames = np.stack(list(pool.map(decode, raw)))
        lat.append(encode_batch(vae, frames, device, dtype))
    lat = np.concatenate(lat)
    assert lat.shape[1:] == (4, 32, 40), lat.shape
    tmp = out_lat + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, lat)
    os.replace(tmp, out_lat)
    np.savez(os.path.join(out_dir, f"{ep}_meta.npz"), **cols)
    return dict(episode=ep, frames=int(len(keep)), map_id=int(cols["map_id"][0]) if "map_id" in cols else -1)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    vae = load_vae(device)
    paths = sorted(glob.glob(os.path.join(args.in_dir, "ep_*.parquet")))
    if args.shard is not None:
        paths = paths[args.shard::args.num_shards]
    print(f"{len(paths)} episodes on {device}", flush=True)
    summary_path = os.path.join(args.out_dir, f"episodes_{args.shard or 0:02d}.jsonl")
    t0 = time.time(); n_frames = 0
    with ThreadPoolExecutor(args.decode_threads) as pool:
        for k, p in enumerate(paths):
            r = encode_episode(p, args.out_dir, vae, device, dtype, args.stride, args.batch_size, pool)
            if r is None:
                continue
            n_frames += r["frames"]
            with open(summary_path, "a") as f:
                f.write(json.dumps(r) + "\n")
            if (k + 1) % 10 == 0:
                print(f"  {k + 1}/{len(paths)} episodes, {n_frames:,} frames, {n_frames / (time.time() - t0):.0f} frames/s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--decode-threads", type=int, default=8)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    main(p.parse_args())
