"""The model-free floor and the autoencoder ceiling of a raw parquet corpus.

Two numbers bound what any world model trained on a corpus can score, and neither needs a model.

The FLOOR is persistence: copy the frame from `g` tics ago and measure PSNR. It says how fast the
footage moves, and it is the reference a rollout has to beat to have done anything. The Sep 2026
finding that every row falls to or below the frozen-seed persistence reference by horizon 64 is what
makes this the first thing to measure about new footage.

The CEILING is the autoencoder: encode and decode with the frozen VAE the corpus will be trained in
and measure PSNR and LPIPS. No model in that latent space can beat it.

A corpus with a high floor and a high ceiling is easy footage, and a headline PSNR measured on it is
not comparable to one measured on harder footage. That is the whole reason to run this before
spending GPU-hours: `release/DENSE_CORPUS.md` exists because GameNGen reports 29.4 dB and our own
persistence floor sits at 18.9 dB at the decision spacing.

Alongside the floor it reports two things that decide whether a corpus is as big as it looks:
explored floor cells per map, since a corpus can be enormous and still show three corridors, and the
row semantics -- tic spacing and action run lengths -- which is how you check from the data, rather
than from a dataset card, whether rows are per tic and how long an action is held.

The two passes are separate so the cheap one can run while a card is busy:

    python persistence_stats.py --dir $D/raw_arnold_dense/arenas --per-map --episodes 200 \
        --workers 24 --out stats_arenas.json
    python persistence_stats.py --dir $D/raw_stiegler --episodes 40 --anchors 500 \
        --cache-frames frames_repro.npy --out stats_repro.json
    python persistence_stats.py --vae --frames frames_repro.npy --out stats_repro.json
"""
import argparse
import collections
import glob
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

GAPS = (1, 2, 4, 8)
NEAR_STATIC_DB = 35.0       # a 1-tic pair above this is a frame a model can score on by doing nothing
CELL = 64.0                 # Doom map units: a 64x64 floor cell, the original level grid
FRAME_ROWS, FRAME_COLS = 240, 320
PAD_TO = 256                # the encoder pipeline pads 240 rows to 256 and strips them after


def psnr_u8(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-10))


def decode_png(buf):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"), dtype=np.uint8)


def row_semantics(tic, action):
    """Is a row a tic, and how long is an action held? Answered from the columns, not from a card.

    `tic_step` is a histogram of consecutive tic differences: all-ones means every tic is stored.
    `run_lengths` histograms the lengths of constant-action runs, and `run_start_phase` counts what
    each run's start tic is modulo 4, which separates a strict decision grid from one that drifts.
    """
    n = len(tic)
    starts = np.concatenate([[0], np.flatnonzero(np.diff(action)) + 1])
    runs = np.diff(np.concatenate([starts, [n]]))
    return {"tic_step": np.bincount(np.diff(tic).astype(np.int64), minlength=6)[:6],
            "run_lengths": np.bincount(runs.astype(np.int64), minlength=10)[:10],
            "run_start_phase": np.bincount((tic[starts] % 4).astype(np.int64), minlength=4)}


def episode_job(args):
    """Persistence at every gap over `anchors` sampled rows of one episode, and that episode's shape."""
    path, anchors, seed, want_frames = args
    import pyarrow.parquet as pq
    try:
        cols = ["tic", "action", "frame"]
        have = set(pq.read_schema(path).names)
        cols += [c for c in ("map_id", "pos_x", "pos_y") if c in have]
        t = pq.read_table(path, columns=cols)
    except Exception as ex:
        return {"path": os.path.basename(path), "error": f"{type(ex).__name__}: {ex}"}
    n = t.num_rows
    if n <= max(GAPS) + 1:
        return {"path": os.path.basename(path), "error": f"only {n} rows"}
    tic = t["tic"].to_numpy(zero_copy_only=False)
    action = t["action"].to_numpy(zero_copy_only=False)
    cells = set()
    if "pos_x" in t.schema.names:
        x = t["pos_x"].to_numpy(zero_copy_only=False)
        y = t["pos_y"].to_numpy(zero_copy_only=False)
        cells = set(zip(np.floor(x / CELL).astype(np.int64).tolist(),
                        np.floor(y / CELL).astype(np.int64).tolist()))
    frames = t["frame"]
    rng = np.random.RandomState(seed)
    idx = np.unique(rng.randint(0, n - max(GAPS) - 1, size=anchors * 2))[:anchors]
    ps = {g: [] for g in GAPS}
    kept = []
    for i in idx:
        base = decode_png(frames[int(i)].as_py())
        if base.shape != (FRAME_ROWS, FRAME_COLS, 3):
            return {"path": os.path.basename(path), "error": f"frame shape {base.shape}"}
        for g in GAPS:
            ps[g].append(psnr_u8(decode_png(frames[int(i) + g].as_py()), base))
        if want_frames:
            kept.append(base)
    out = {"path": os.path.basename(path), "rows": int(n), "cells": cells,
           "map_id": int(t["map_id"][0].as_py()) if "map_id" in t.schema.names else 0,
           "psnr": {g: ps[g] for g in GAPS}, **row_semantics(tic, action)}
    if want_frames:
        out["frames"] = np.stack(kept)
    return out


def _group(records):
    """Aggregate one group of episode records into the reported block."""
    p = {g: np.concatenate([np.asarray(r["psnr"][g]) for r in records]) for g in GAPS}
    cells = set().union(*(r["cells"] for r in records))
    per_ep = [len(r["cells"]) for r in records]
    out = {"episodes": len(records), "rows": int(sum(r["rows"] for r in records)),
           "pairs": int(len(p[1])),
           "persistence_psnr": {str(g): float(p[g].mean()) for g in GAPS},
           "persistence_psnr_se": {str(g): float(p[g].std(ddof=1) / np.sqrt(len(p[g]))) for g in GAPS},
           "near_static_fraction": float(np.mean(p[1] > NEAR_STATIC_DB))}
    for k in ("tic_step", "run_lengths", "run_start_phase"):
        out[k] = np.sum([r[k] for r in records], axis=0).astype(int).tolist()
    if any(per_ep):
        out.update(explored_cells_union=len(cells),
                   explored_cells_per_episode_mean=float(np.mean(per_ep)),
                   explored_cells_per_episode_min=int(np.min(per_ep)),
                   explored_cells_per_episode_max=int(np.max(per_ep)))
    return out


def do_stats(a):
    paths = sorted(glob.glob(os.path.join(a.dir, "ep_*.parquet")))
    if not paths:
        raise SystemExit(f"no ep_*.parquet in {a.dir}")
    rng = np.random.RandomState(a.seed)
    k = min(a.episodes, len(paths))
    pick = [paths[i] for i in sorted(rng.choice(len(paths), k, replace=False))]
    jobs = [(p, a.anchors, a.seed + i, bool(a.cache_frames)) for i, p in enumerate(pick)]
    got, errors, frames = [], [], []
    with ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(episode_job, jobs):
            if "error" in r:
                errors.append(r)
                continue
            got.append(r)
            if a.cache_frames:
                frames.append(r.pop("frames"))
    if not got:
        raise SystemExit(f"every sampled episode failed: {errors[:3]}")
    out = {"dir": a.dir, "name": a.name or os.path.basename(a.dir.rstrip("/")),
           "episodes_sampled": len(got), "anchors_per_episode": a.anchors,
           "cell_units": CELL, "near_static_threshold_db": NEAR_STATIC_DB,
           "gaps_in_tics": list(GAPS), "n_errors": len(errors), "errors": errors[:20],
           "all": _group(got)}
    if a.per_map:
        by = collections.defaultdict(list)
        for r in got:
            by[str(r["map_id"])].append(r)
        out["per_map"] = {k: _group(v) for k, v in sorted(by.items(), key=lambda kv: int(kv[0]))}
    if a.cache_frames:
        arr = np.concatenate(frames)[:a.frames_cap] if a.frames_cap else np.concatenate(frames)
        np.save(a.cache_frames, arr)
        out["frames_cached"] = {"path": a.cache_frames, "shape": list(arr.shape)}
    write(a.out, out)


def do_vae(a):
    """Encode-decode the cached frames with a frozen VAE and add the ceiling to the same report."""
    import lpips
    import torch
    from diffusers import AutoencoderKL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arr = np.load(a.frames, mmap_mode="r")
    vae = AutoencoderKL.from_pretrained(a.vae_id, cache_dir=a.hf_cache).to(device).eval()
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    ps, ls = [], []
    with torch.no_grad():
        for i in range(0, len(arr), a.batch_size):
            x = torch.from_numpy(np.asarray(arr[i:i + a.batch_size])).to(device)
            x = x.permute(0, 3, 1, 2).float() / 127.5 - 1.0
            xp = torch.nn.functional.pad(x, (0, 0, 0, PAD_TO - x.shape[2]))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                rec = vae.decode(vae.encode(xp).latent_dist.mean).sample
            rec = rec.float()[:, :, :x.shape[2]]
            mse = ((rec - x) ** 2).flatten(1).mean(1).clamp_min(1e-10)
            ps.append((10 * torch.log10(4.0 / mse)).cpu().numpy())     # [-1, 1] range, so peak is 2
            ls.append(lp(rec.clamp(-1, 1), x).flatten().cpu().numpy())
            if (i // a.batch_size) % 20 == 0:
                print(f"  {i + len(x)}/{len(arr)}", flush=True)
    ps, ls = np.concatenate(ps), np.concatenate(ls)
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    out["vae_ceiling"] = {"vae_id": a.vae_id, "frames": int(len(ps)),
                          "psnr": float(ps.mean()), "psnr_se": float(ps.std(ddof=1) / np.sqrt(len(ps))),
                          "lpips": float(ls.mean()), "lpips_se": float(ls.std(ddof=1) / np.sqrt(len(ls)))}
    write(a.out, out)


def write(path, obj):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)
    print(json.dumps(obj, indent=1))


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", action="store_true", help="second pass: the autoencoder ceiling on cached frames")
    p.add_argument("--dir", default="", help="a corpus directory of ep_*.parquet recordings")
    p.add_argument("--out", required=True, help="report JSON; the --vae pass adds to an existing one")
    p.add_argument("--name", default="")
    p.add_argument("--per-map", action="store_true", help="also break every statistic down by map_id")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--anchors", type=int, default=100, help="sampled frame pairs per episode")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-frames", default="", help="save the sampled anchor frames here for --vae")
    p.add_argument("--frames-cap", type=int, default=0, help="keep at most this many cached frames")
    p.add_argument("--frames", default="", help="--vae: the cached frames to round-trip")
    p.add_argument("--vae-id", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    if a.vae:
        do_vae(a)
    else:
        do_stats(a)
