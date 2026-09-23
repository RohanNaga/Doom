"""Row-count audit: latent rows == sidecar rows == raw parquet rows, for every episode in a latents dir.

usage: rowcount_check.py <latents_dir> <raw_dir> [channels]
"""
import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq

lat_dir, raw_dir = sys.argv[1:3]
channels = int(sys.argv[3]) if len(sys.argv) > 3 else None
bad, n, frames = [], 0, 0
for lat in sorted(glob.glob(os.path.join(lat_dir, "ep_*_latents.npy"))):
    ep = os.path.basename(lat)[:-len("_latents.npy")]
    a = np.load(lat, mmap_mode="r")
    meta = os.path.join(lat_dir, f"{ep}_meta.npz")
    raw = os.path.join(raw_dir, f"{ep}.parquet")
    if not os.path.exists(meta):
        bad.append((ep, "no sidecar")); continue
    m = np.load(meta)
    lens = {k: len(m[k]) for k in m.files}
    rows = pq.read_metadata(raw).num_rows if os.path.exists(raw) else -1
    shape_ok = a.shape[1:] == ((channels, 32, 40) if channels else a.shape[1:]) and a.dtype == np.float16
    if not (shape_ok and len(set(lens.values())) == 1 and a.shape[0] == rows == next(iter(lens.values()))
            and str(m["buttons"].dtype) == "<U19"):
        bad.append((ep, a.shape, str(a.dtype), sorted(set(lens.values())), rows, str(m["buttons"].dtype)))
    n += 1
    frames += a.shape[0]
print(f"{lat_dir}: {n} episodes, {frames} rows, {len(bad)} mismatches")
for b in bad[:20]:
    print("  MISMATCH", b)
