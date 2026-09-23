"""Do the stored latents belong to the rows their sidecars describe? A per-shard launch gate.

The 2026-09-22 review (H3, section 7.1 item 2) found that no gate ever reads a stored latent array
against the recording: the encoder's own decode check round-trips freshly encoded frames of one
episode, and the sidecar audit reads sidecars only. A latent array shifted by a row against its
sidecar, or a shard written under a different encoder contract, would train without complaint.
Train shard 0 was written by the old encoder and shard 1 by the new one, so the check is per shard.

For each shard (from `episodes_NN.jsonl`, or one shard for `--episodes`), for a few sampled episodes:

  1. Re-encode.  Take one full encoder batch at a random batch boundary and the episode's tail batch
     (the short last batch, where cuDNN sees a different shape), read the raw frames at the tics the
     sidecar names, and encode them exactly as `encode_parquet.encode_batch` did, with the settings
     recorded in that shard's `encode_meta_NN.json` (VAE, scale, shift, batch size, dtype). Compare
     with the stored rows: max and mean absolute difference, and the fraction bit-identical.
  2. Decode.  Decode the stored latents through one decoder and score `vae_psnr` against the raw
     frames, unshifted and with the stored rows deliberately shifted by -4, -1, +1 and +4. The
     unshifted alignment must score best. This is an ALERT, not a certificate: a low-motion stretch
     can make neighbours nearly tie, which is why it is one of two tests and why it is per shard.

A shard fails if its mean absolute difference exceeds `--max-abs-diff` (1e-3, the Sep 20 batch-size
measurement's scale), if fewer than `--min-identical` (0.5) of its re-encoded values are
bit-identical at the recorded batch size, or if any shifted control scores at least as well as the
unshifted alignment. Every shard is reported on its own; nothing is only pooled.

Re-encoding on different hardware than the corpus was written on can lower the identical fraction
without anything being wrong; read the mean difference and the shifted controls first in that case.

    python check_latent_alignment.py --latents-dir $D/latents_arnold_dense_pertic/arenas \\
        --parquet-dir $D/raw_arnold_dense/arenas --device cuda:0 --out $D/logs/latent_align_train.json
"""
import argparse
import glob
import io
import json
import os
import re

import numpy as np
import torch

SHIFTS = (-4, -1, 0, 1, 4)
FRAME_ROWS = 240
_SHARD = re.compile(r"episodes_(\d+)\.jsonl$")


def shards_of(latents_dir, episodes=None):
    """{shard name: sorted episode ids}, from the encoder's per-shard `episodes_NN.jsonl` files.

    `--episodes` bypasses the files and names one shard, "given". An episode listed by a shard
    file but absent from the directory is skipped here; the inventory gate is what counts those.
    """
    from doom_data import list_latent_episodes
    present = {ep for ep, _, _ in list_latent_episodes(latents_dir)}
    if episodes is not None:
        return {"given": sorted(int(e) for e in episodes if int(e) in present)}
    out = {}
    for path in sorted(glob.glob(os.path.join(latents_dir, "episodes_*.jsonl"))):
        name = _SHARD.search(os.path.basename(path)).group(1)
        ids = set()
        with open(path) as f:
            for ln in f:
                if ln.strip():
                    ids.add(int(str(json.loads(ln)["episode"]).split("_")[-1]))
        out[name] = sorted(ids & present)
    if not out:
        raise SystemExit(f"no episodes_NN.jsonl in {latents_dir}; pass --episodes to name the episodes")
    return out


def shard_meta(latents_dir, shard):
    """The encoder settings recorded for one shard: `encode_meta_<shard>.json`."""
    name = "00" if shard == "given" else shard
    path = os.path.join(latents_dir, f"encode_meta_{name}.json")
    if not os.path.isfile(path):
        raise SystemExit(f"no {os.path.basename(path)} in {latents_dir}: the settings shard {shard} was "
                         "encoded under are unknown, so its latents cannot be reproduced")
    with open(path) as f:
        m = json.load(f)
    if m.get("legacy"):
        raise SystemExit(f"{path} describes a legacy (April) corpus; this check is for the padded layout")
    a = m.get("args") or {}
    return {"vae_id": m.get("vae_id") or "", "vae_subfolder": m.get("vae_subfolder") or "",
            "scale": m.get("scaling_factor_applied"), "shift": m.get("shift_factor_applied"),
            "batch_size": int(a.get("batch_size") or 64), "dtype": a.get("dtype") or "bf16",
            "every_tic": bool(m.get("every_tic")), "path": path}


def load_encoder(meta, device, cache_dir=None):
    """The autoencoder a shard was encoded with, as `encode_parquet` builds it. Tests replace this."""
    from doomdit_utils import build_vae
    return build_vae(meta["vae_id"], meta["vae_subfolder"], device, cache_dir)


def sample_rows(n_rows, batch_size, rng):
    """Row blocks to re-encode: one full batch at a random boundary, and the tail batch.

    The encoder cut an episode into batches at 0, B, 2B, ...; a block is only reproducible if it is
    re-encoded with the same members, so every block starts on a boundary and keeps its length.
    """
    starts = list(range(0, n_rows, batch_size))
    tail = starts[-1]
    full = [s for s in starts if s + batch_size <= n_rows and s != tail]
    blocks = []
    if full:
        s = int(rng.choice(full))
        blocks.append((s, s + batch_size))
    blocks.append((tail, n_rows))
    return blocks


def raw_frames(parquet_path, tics):
    """uint8 (k, 240, 320, 3) frames at exactly these tics, refusing a tic the recording lacks."""
    import pyarrow.parquet as pq
    from PIL import Image
    t = pq.read_table(parquet_path, columns=["tic", "frame"])
    rec = np.asarray(t["tic"]).astype(np.int64)
    pos = {int(x): i for i, x in enumerate(rec.tolist())}
    missing = [int(x) for x in tics if int(x) not in pos]
    if missing:
        raise KeyError(f"{parquet_path} has no frame at tic(s) {missing[:4]}: the sidecar and the recording disagree")
    col = t["frame"]
    return np.stack([np.asarray(Image.open(io.BytesIO(col[pos[int(x)]].as_py())).convert("RGB"), dtype=np.uint8)
                     for x in tics])


def _psnr01(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


@torch.no_grad()
def decode01(vae, lat, device, scale, shift):
    """Stored latents -> [0, 1] images of the real 240 rows."""
    from doomdit_utils import denormalize_latents
    z = denormalize_latents(torch.from_numpy(np.asarray(lat, dtype=np.float32)).to(device), scale, shift)
    img = vae.decode(z).sample[:, :, :FRAME_ROWS]
    return (img.float() * 0.5 + 0.5).clamp(0, 1)


def check_episode(vae, lat_path, meta_path, parquet_path, meta, device, rng, decoder=None):
    """Per-episode arrays: re-encode differences and vae_psnr per shift, over the sampled blocks."""
    from encode_parquet import encode_batch
    lat = np.load(lat_path, mmap_mode="r")
    with np.load(meta_path) as m:
        tic = np.asarray(m["tic"]).astype(np.int64)
    if len(tic) != lat.shape[0]:
        return {"problem": f"{os.path.basename(lat_path)}: {lat.shape[0]} latents vs {len(tic)} sidecar rows"}
    dtype = torch.bfloat16 if meta["dtype"] == "bf16" else torch.float32
    dec = decoder or vae
    diffs, identical, total = [], 0, 0
    psnr = {s: [] for s in SHIFTS}
    fresh_psnr = []
    rows = 0
    for a, b in sample_rows(lat.shape[0], meta["batch_size"], rng):
        frames = raw_frames(parquet_path, tic[a:b])
        fresh = encode_batch(vae, frames, device, dtype, False, meta["scale"], meta["shift"])
        stored = np.asarray(lat[a:b])
        if fresh.shape != stored.shape:
            return {"problem": f"{os.path.basename(lat_path)}: re-encoded shape {fresh.shape} != stored {stored.shape}"}
        d = np.abs(fresh.astype(np.float32) - stored.astype(np.float32))
        diffs.append(d.reshape(-1))
        identical += int((fresh == stored).sum())
        total += fresh.size
        ref = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255).to(device)
        fresh_psnr += _psnr01(decode01(dec, fresh, device, meta["scale"], meta["shift"]), ref).cpu().tolist()
        lo, hi = max(0, a - max(SHIFTS)), min(lat.shape[0], b + max(SHIFTS))
        imgs = decode01(dec, lat[lo:hi], device, meta["scale"], meta["shift"])
        for s in SHIFTS:
            idx = [i for i in range(a, b) if lo <= i + s < hi]
            if idx:
                got = imgs[[i + s - lo for i in idx]]
                want = ref[[i - a for i in idx]]
                psnr[s] += _psnr01(got, want).cpu().tolist()
        rows += b - a
    d = np.concatenate(diffs)
    return {"rows": rows, "max_abs_diff": float(d.max()), "abs_diff_sum": float(d.sum()), "values": int(d.size),
            "identical": identical, "total": total, "psnr": psnr, "fresh_psnr": fresh_psnr}


def check(latents_dir, parquet_dir, episodes=None, per_shard=2, device="cpu", max_abs_diff=1e-3,
          min_identical=0.5, seed=0, vae=None, decoder=None, cache_dir=None):
    """The per-shard report; `ok` is False if any shard fails."""
    from doom_data import list_latent_episodes
    files = {ep: (lp, mp) for ep, lp, mp in list_latent_episodes(latents_dir)}
    rng = np.random.RandomState(seed)
    report = {"latents_dir": latents_dir, "parquet_dir": parquet_dir, "shifts": list(SHIFTS),
              "max_abs_diff_allowed": max_abs_diff, "min_identical": min_identical, "shards": {}}
    for shard, ids in shards_of(latents_dir, episodes).items():
        meta = shard_meta(latents_dir, shard)
        enc = vae if vae is not None else load_encoder(meta, device, cache_dir)
        picked = sorted(rng.choice(ids, size=min(per_shard, len(ids)), replace=False).tolist()) if ids else []
        problems, rows, dmax, dsum, nval, ident, tot = [], 0, 0.0, 0.0, 0, 0, 0
        psnr = {s: [] for s in SHIFTS}
        fresh_psnr = []
        for ep in picked:
            r = check_episode(enc, *files[ep], os.path.join(parquet_dir, f"ep_{ep:05d}.parquet"), meta, device,
                              rng, decoder)
            if "problem" in r:
                problems.append(r["problem"])
                continue
            rows += r["rows"]
            dmax, dsum, nval = max(dmax, r["max_abs_diff"]), dsum + r["abs_diff_sum"], nval + r["values"]
            ident, tot = ident + r["identical"], tot + r["total"]
            for s in SHIFTS:
                psnr[s] += r["psnr"][s]
            fresh_psnr += r["fresh_psnr"]
        if not picked:
            problems.append(f"shard {shard} has no encoded episode to check")
        mean_diff = dsum / nval if nval else float("nan")
        frac = ident / tot if tot else 0.0
        vae_psnr = {f"{s:+d}" if s else "0": (float(np.mean(v)) if v else None) for s, v in psnr.items()}
        if nval and mean_diff > max_abs_diff:
            problems.append(f"mean |re-encoded - stored| {mean_diff:.3g} > {max_abs_diff:g}")
        if tot and frac < min_identical:
            problems.append(f"only {frac:.1%} of re-encoded values are bit-identical at batch {meta['batch_size']} "
                            f"(need {min_identical:.0%})")
        base = vae_psnr["0"]
        worse = [k for k, v in vae_psnr.items() if k != "0" and v is not None and base is not None and v >= base]
        if worse:
            problems.append(f"shifted stored rows {worse} score vae_psnr at least as high as the unshifted "
                            f"alignment ({base:.2f} dB): the latents may be off by rows against their sidecar")
        report["shards"][shard] = {"episodes": picked, "rows": rows, "encode_meta": meta["path"],
                                   "batch_size": meta["batch_size"], "dtype": meta["dtype"],
                                   "max_abs_diff": dmax, "mean_abs_diff": mean_diff, "fraction_identical": frac,
                                   "vae_psnr": vae_psnr,
                                   "vae_psnr_reencoded": float(np.mean(fresh_psnr)) if fresh_psnr else None,
                                   "problems": problems, "ok": not problems}
    report["ok"] = bool(report["shards"]) and all(s["ok"] for s in report["shards"].values())
    return report


def main(args):
    from doom_data import parse_episode_ids
    decoder = None
    if args.decoder:
        from doomdit_utils import build_vae
        decoder = build_vae(args.decoder, "", args.device, args.cache_dir)
    rep = check(args.latents_dir, args.parquet_dir,
                parse_episode_ids(args.episodes) if args.episodes else None, args.episodes_per_shard,
                args.device, args.max_abs_diff, args.min_identical, args.seed, decoder=decoder,
                cache_dir=args.cache_dir)
    text = json.dumps(rep, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)
    for name, s in rep["shards"].items():
        print(f"shard {name}: {'ok' if s['ok'] else 'FAILED'} mean_abs_diff={s['mean_abs_diff']:.3g} "
              f"identical={s['fraction_identical']:.1%} vae_psnr={s['vae_psnr']}")
    return 0 if rep["ok"] else 2


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--latents-dir", required=True)
    p.add_argument("--parquet-dir", required=True, help="the raw recordings the corpus was encoded from")
    p.add_argument("--episodes", default="", help="check these ids as one shard instead of reading episodes_NN.jsonl")
    p.add_argument("--episodes-per-shard", dest="episodes_per_shard", type=int, default=2)
    p.add_argument("--device", default="cuda:0", help="where to re-encode and decode; cpu works, slowly")
    p.add_argument("--max-abs-diff", dest="max_abs_diff", type=float, default=1e-3,
                   help="largest allowed MEAN absolute difference between re-encoded and stored latents")
    p.add_argument("--min-identical", dest="min_identical", type=float, default=0.5,
                   help="smallest allowed fraction of bit-identical values at the recorded batch size")
    p.add_argument("--decoder", default="", help="decode through this VAE instead of the shard's own")
    p.add_argument("--cache-dir", dest="cache_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="", help="also write the JSON report here")
    return p


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
