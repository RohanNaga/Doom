"""
Build the training dataset from per-episode files.

Takes a directory of `ep_XXXX_latents.npy` and `ep_XXXX_actions.npy` files,
applies a sliding window of 4 past frames → predict 5th, and writes three
consolidated arrays that CustomDataset can memory-map during training.

Usage:
    python build_dataset.py --episodes-dir data/full --out-dir data

Output (in out-dir):
    context_latents.npy   shape (N, 4, 4, 15, 20)  float16
    target_latents.npy    shape (N, 4, 15, 20)     float16
    context_actions.npy   shape (N, 5)             int64
        columns are [a_i, a_{i+1}, a_{i+2}, a_{i+3}, a_{i+4}] — last column is target action,
        matching what CustomDataset picks via `actions[idx][-1]`.
"""
import argparse
import glob
import os
import re
import shutil

import numpy as np


def _save_atomic(path, arr, required_bytes):
    """Write np.save output to a .tmp path, fsync, then atomically rename.
    Fails loud if there isn't enough free disk for the expected file size.
    """
    out_dir = os.path.dirname(path) or "."
    free = shutil.disk_usage(out_dir).free
    # Require 1GB headroom beyond the file itself to avoid other-user races.
    headroom = 1 << 30
    if free < required_bytes + headroom:
        raise RuntimeError(
            f"Not enough disk at {out_dir}: need ~{(required_bytes + headroom) / 1e9:.1f} GB, "
            f"have {free / 1e9:.2f} GB free. Free space and retry."
        )
    tmp_path = path + ".tmp"
    # Use a file object so np.save doesn't auto-append `.npy` to our tmp_path.
    with open(tmp_path, "wb") as f:
        np.save(f, arr)
    # Verify the file actually has the expected size before renaming.
    actual = os.path.getsize(tmp_path)
    if actual < required_bytes * 0.95:
        os.remove(tmp_path)
        raise RuntimeError(
            f"Truncated write to {tmp_path}: expected ~{required_bytes} bytes, got {actual}. "
            f"Likely ENOSPC — free space and retry."
        )
    os.replace(tmp_path, path)


def main(args):
    latent_files = sorted(glob.glob(os.path.join(args.episodes_dir, "ep_*_latents.npy")))
    assert latent_files, f"no ep_*_latents.npy files found in {args.episodes_dir}"

    # Pair each latent file with its matching actions file.
    pairs = []
    for lat_path in latent_files:
        ep_id_match = re.search(r"ep_(\d+)_latents\.npy$", lat_path)
        assert ep_id_match, f"unexpected filename format: {lat_path}"
        act_path = os.path.join(args.episodes_dir, f"ep_{ep_id_match.group(1)}_actions.npy")
        assert os.path.isfile(act_path), f"missing actions file: {act_path}"
        pairs.append((lat_path, act_path))

    print(f"Found {len(pairs)} episodes.")

    ctx_chunks, tgt_chunks, act_chunks = [], [], []
    total_samples = 0
    for lat_path, act_path in pairs:
        latents = np.load(lat_path)    # (N, 4, 15, 20) float16
        actions = np.load(act_path)    # (N,) int-like
        assert latents.ndim == 4 and latents.shape[1:] == (4, 15, 20), \
            f"unexpected latent shape in {lat_path}: {latents.shape}"
        assert actions.ndim == 1 and actions.shape[0] == latents.shape[0], \
            f"action/latent length mismatch in {lat_path}: {actions.shape} vs {latents.shape}"

        n = latents.shape[0]
        if n < 5:
            print(f"  skipping {lat_path} (only {n} frames, need >=5)")
            continue

        # Sliding window: sample i uses frames [i..i+3] as context, frame i+4 as target.
        num_samples = n - 4
        # context: (num_samples, 4, 4, 15, 20)
        ctx = np.stack([latents[i:i + 4] for i in range(num_samples)], axis=0)
        # target: (num_samples, 4, 15, 20)
        tgt = latents[4:4 + num_samples]
        # actions: (num_samples, 5) — [a_i, a_{i+1}, a_{i+2}, a_{i+3}, a_{i+4}]
        act = np.stack([actions[i:i + 5] for i in range(num_samples)], axis=0).astype(np.int64)

        ctx_chunks.append(ctx)
        tgt_chunks.append(tgt)
        act_chunks.append(act)
        total_samples += num_samples

    print(f"Total training samples: {total_samples:,}")

    ctx_all = np.concatenate(ctx_chunks, axis=0)
    tgt_all = np.concatenate(tgt_chunks, axis=0)
    act_all = np.concatenate(act_chunks, axis=0)

    print(f"  context_latents: {ctx_all.shape} {ctx_all.dtype}")
    print(f"  target_latents:  {tgt_all.shape} {tgt_all.dtype}")
    print(f"  context_actions: {act_all.shape} {act_all.dtype}"
          f"  min={act_all.min()} max={act_all.max()}")

    os.makedirs(args.out_dir, exist_ok=True)
    _save_atomic(os.path.join(args.out_dir, "context_latents.npy"), ctx_all, ctx_all.nbytes)
    _save_atomic(os.path.join(args.out_dir, "target_latents.npy"), tgt_all, tgt_all.nbytes)
    _save_atomic(os.path.join(args.out_dir, "context_actions.npy"), act_all, act_all.nbytes)
    print(f"Saved to {args.out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-dir", type=str, required=True,
                        help="Directory containing ep_XXXX_latents.npy and ep_XXXX_actions.npy")
    parser.add_argument("--out-dir", type=str, default="data",
                        help="Where to write context/target/actions consolidated arrays")
    args = parser.parse_args()
    main(args)
