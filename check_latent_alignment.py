"""Do the stored latents belong to the rows their sidecars describe? A per-shard launch gate.

The 2026-09-22 review (H3, section 7.1 item 2) found that no gate ever reads a stored latent array
against the recording: the encoder's own decode check round-trips freshly encoded frames of one
episode, and the sidecar audit reads sidecars only. A latent array shifted by a row against its
sidecar, or a shard written under a different encoder contract, would train without complaint.
Train shard 0 was written by the old encoder and shard 1 by the new one, so the check is per shard.

Before anything is sampled, the corpus has to be fully accounted for (Astra's review, 2026-09-23):
every `episodes_NN.jsonl` needs its `encode_meta_NN.json` and the reverse, every encoded episode must
be listed by exactly one shard, and all shards must share one latent contract (VAE id and subfolder,
scale, shift, channels, batch size, dtype). A missing shard log hid a shard of shifted latents, and
shards under two scaling contracts passed, before this. Every tensor and statistic must be finite.

For each shard, for a few sampled episodes (`--episodes` restricts which, inside their own shards):

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
import math
import os
import re

import numpy as np
import torch

SHIFTS = (-4, -1, 0, 1, 4)
FRAME_ROWS = 240
_SHARD = re.compile(r"episodes_(\d+)\.jsonl$")
_META = re.compile(r"encode_meta_(\d+)\.json$")


def shards_of(latents_dir, episodes=None):
    """({shard name: sorted episode ids to sample from}, coverage problems).

    Coverage is complete or the gate fails, because a shard the check cannot see is a shard it
    cannot catch (Astra, 2026-09-23: a missing `episodes_01.jsonl` hid shard 1's shifted latents):

      * every `episodes_NN.jsonl` has its `encode_meta_NN.json` and every `encode_meta_NN.json` its
        `episodes_NN.jsonl`;
      * every encoded episode in the directory is listed by exactly one shard;
      * with `--episodes`, every named episode is encoded and attributed to a shard, and sampling is
        restricted to those episodes inside their own shards.
    """
    from doom_data import list_latent_episodes
    present = {ep for ep, _, _ in list_latent_episodes(latents_dir)}
    problems, listed, owner = [], {}, {}
    for path in sorted(glob.glob(os.path.join(latents_dir, "episodes_*.jsonl"))):
        name = _SHARD.search(os.path.basename(path)).group(1)
        ids = set()
        with open(path) as f:
            for ln in f:
                if ln.strip():
                    ids.add(int(str(json.loads(ln)["episode"]).split("_")[-1]))
        listed[name] = ids
        for ep in ids:
            if ep in owner and owner[ep] != name:
                problems.append(f"episode {ep} is listed by shards {owner[ep]} and {name}")
            owner.setdefault(ep, name)
    metas = {_META.search(os.path.basename(p)).group(1)
             for p in glob.glob(os.path.join(latents_dir, "encode_meta_*.json"))}
    if not listed:
        problems.append(f"no episodes_NN.jsonl in {latents_dir}: no episode can be attributed to a shard")
    for name in sorted(set(listed) - metas):
        problems.append(f"episodes_{name}.jsonl has no encode_meta_{name}.json: its encoder settings are unknown")
    for name in sorted(metas - set(listed)):
        problems.append(f"encode_meta_{name}.json has no episodes_{name}.jsonl: that shard's episodes are unknown")
    unattributed = sorted(present - set(owner))
    if unattributed:
        problems.append(f"{len(unattributed)} encoded episode(s) belong to no shard log: {unattributed[:8]}")
    want = present if episodes is None else {int(e) for e in episodes}
    if episodes is not None:
        gone = sorted(want - present)
        if gone:
            problems.append(f"{len(gone)} requested episode(s) are not encoded: {gone[:8]}")
    return {name: sorted(ids & present & want) for name, ids in listed.items()}, problems


CONTRACT_KEYS = ("vae_id", "vae_subfolder", "scale", "shift", "channels", "batch_size", "dtype", "every_tic")


def shard_meta(latents_dir, shard):
    """The encoder settings recorded for one shard: `encode_meta_<shard>.json`."""
    path = os.path.join(latents_dir, f"encode_meta_{shard}.json")
    if not os.path.isfile(path):
        raise SystemExit(f"no {os.path.basename(path)} in {latents_dir}: the settings shard {shard} was "
                         "encoded under are unknown, so its latents cannot be reproduced")
    with open(path) as f:
        m = json.load(f)
    if m.get("legacy"):
        raise SystemExit(f"{path} describes a legacy (April) corpus; this check is for the padded layout")
    a = m.get("args") or {}
    channels = (m.get("latent_contract") or {}).get("latent_channels") or a.get("latent_channels")
    return {"vae_id": m.get("vae_id") or "", "vae_subfolder": m.get("vae_subfolder") or "",
            "scale": m.get("scaling_factor_applied"), "shift": m.get("shift_factor_applied"),
            "channels": None if channels is None else int(channels),
            "batch_size": int(a.get("batch_size") or 64), "dtype": a.get("dtype") or "bf16",
            "every_tic": bool(m.get("every_tic")), "path": path}


def contract_problems(metas):
    """One problem per contract key on which the shards disagree: one corpus, one latent contract."""
    bad = []
    for k in CONTRACT_KEYS:
        seen = {str(m[k]) for m in metas.values()}
        if len(seen) > 1:
            bad.append(f"shards were encoded under different {k}: "
                       + ", ".join(f"{name}={metas[name][k]!r}" for name in sorted(metas)))
    return bad


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
    """Per-episode arrays: re-encode differences and vae_psnr per shift, over the sampled blocks.

    Every tensor on the way is required to be finite: the stored rows, the re-encoded rows, both
    decodes and every PSNR. A NaN compared with `>` is False, so a sampled NaN used to pass.
    """
    from encode_parquet import encode_batch
    name = os.path.basename(lat_path)
    lat = np.load(lat_path, mmap_mode="r")
    with np.load(meta_path) as m:
        tic = np.asarray(m["tic"]).astype(np.int64)
    if len(tic) != lat.shape[0]:
        return {"problem": f"{name}: {lat.shape[0]} latents vs {len(tic)} sidecar rows"}
    if meta.get("channels") is not None and lat.shape[1] != meta["channels"]:
        return {"problem": f"{name}: {lat.shape[1]} latent channels, the shard's contract says {meta['channels']}"}
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
            return {"problem": f"{name}: re-encoded shape {fresh.shape} != stored {stored.shape}"}
        if not np.isfinite(stored.astype(np.float32)).all():
            return {"problem": f"{name}: non-finite stored latents in rows {a}..{b - 1}"}
        if not np.isfinite(fresh.astype(np.float32)).all():
            return {"problem": f"{name}: the re-encode of rows {a}..{b - 1} is not finite"}
        d = np.abs(fresh.astype(np.float32) - stored.astype(np.float32))
        diffs.append(d.reshape(-1))
        identical += int((fresh == stored).sum())
        total += fresh.size
        ref = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255).to(device)
        fresh_img = decode01(dec, fresh, device, meta["scale"], meta["shift"])
        lo, hi = max(0, a - max(SHIFTS)), min(lat.shape[0], b + max(SHIFTS))
        imgs = decode01(dec, lat[lo:hi], device, meta["scale"], meta["shift"])
        if not (torch.isfinite(fresh_img).all() and torch.isfinite(imgs).all()):
            return {"problem": f"{name}: a decode of rows {a}..{b - 1} is not finite"}
        fresh_psnr += _psnr01(fresh_img, ref).cpu().tolist()
        for s in SHIFTS:
            idx = [i for i in range(a, b) if lo <= i + s < hi]
            if idx:
                got = imgs[[i + s - lo for i in idx]]
                want = ref[[i - a for i in idx]]
                psnr[s] += _psnr01(got, want).cpu().tolist()
        rows += b - a
    return {"rows": rows, "diffs": np.concatenate(diffs), "identical": identical, "total": total,
            "psnr": psnr, "fresh_psnr": fresh_psnr}


def _finite(x):
    return x is not None and math.isfinite(x)


def shard_verdict(diffs, identical, total, psnr, fresh_psnr, max_abs_diff, min_identical):
    """(statistics, problems) of one shard's pooled samples; any non-finite statistic is a problem."""
    problems = []
    mean_diff = float(diffs.mean()) if diffs.size else float("nan")
    max_diff = float(diffs.max()) if diffs.size else float("nan")
    frac = identical / total if total else float("nan")
    vae_psnr = {f"{s:+d}" if s else "0": (float(np.mean(v)) if v else None) for s, v in psnr.items()}
    reenc = float(np.mean(fresh_psnr)) if fresh_psnr else None
    stats = {"max_abs_diff": max_diff, "mean_abs_diff": mean_diff, "fraction_identical": frac,
             "vae_psnr": vae_psnr, "vae_psnr_reencoded": reenc}
    bad = [k for k, v in (("mean_abs_diff", mean_diff), ("max_abs_diff", max_diff), ("fraction_identical", frac),
                          ("vae_psnr_reencoded", reenc), *((f"vae_psnr[{k}]", v) for k, v in vae_psnr.items()))
           if not _finite(v)]
    if bad:
        problems.append(f"non-finite or missing statistics: {bad}")
        return stats, problems
    if mean_diff > max_abs_diff:
        problems.append(f"mean |re-encoded - stored| {mean_diff:.3g} > {max_abs_diff:g}")
    if frac < min_identical:
        problems.append(f"only {frac:.1%} of re-encoded values are bit-identical (need {min_identical:.0%})")
    base = vae_psnr["0"]
    worse = [k for k, v in vae_psnr.items() if k != "0" and v >= base]
    if worse:
        problems.append(f"shifted stored rows {worse} score vae_psnr at least as high as the unshifted "
                        f"alignment ({base:.2f} dB): the latents may be off by rows against their sidecar")
    return stats, problems


def check(latents_dir, parquet_dir, episodes=None, per_shard=2, device="cpu", max_abs_diff=1e-3,
          min_identical=0.5, seed=0, vae=None, decoder=None, cache_dir=None):
    """The per-shard report; `ok` is False if coverage, the contracts or any shard fails."""
    from doom_data import list_latent_episodes
    files = {ep: (lp, mp) for ep, lp, mp in list_latent_episodes(latents_dir)}
    rng = np.random.RandomState(seed)
    shards, coverage = shards_of(latents_dir, episodes)
    metas = {}
    for shard in shards:
        try:
            metas[shard] = shard_meta(latents_dir, shard)
        except SystemExit as e:
            coverage.append(str(e))
    contracts = contract_problems(metas)
    report = {"latents_dir": latents_dir, "parquet_dir": parquet_dir, "shifts": list(SHIFTS),
              "max_abs_diff_allowed": max_abs_diff, "min_identical": min_identical,
              "coverage_problems": coverage, "contract_problems": contracts, "shards": {}}
    for shard, ids in shards.items():
        if shard not in metas:
            continue
        meta = metas[shard]
        enc = vae if vae is not None else load_encoder(meta, device, cache_dir)
        picked = sorted(rng.choice(ids, size=min(per_shard, len(ids)), replace=False).tolist()) if ids else []
        problems, rows, diffs, ident, tot = [], 0, [], 0, 0
        psnr = {s: [] for s in SHIFTS}
        fresh_psnr = []
        for ep in picked:
            r = check_episode(enc, *files[ep], os.path.join(parquet_dir, f"ep_{ep:05d}.parquet"), meta, device,
                              rng, decoder)
            if "problem" in r:
                problems.append(r["problem"])
                continue
            rows += r["rows"]
            diffs.append(r["diffs"])
            ident, tot = ident + r["identical"], tot + r["total"]
            for s in SHIFTS:
                psnr[s] += r["psnr"][s]
            fresh_psnr += r["fresh_psnr"]
        if not picked:
            problems.append(f"shard {shard} has no encoded episode to check")
        stats, verdict = shard_verdict(np.concatenate(diffs) if diffs else np.zeros(0, np.float32), ident, tot,
                                       psnr, fresh_psnr, max_abs_diff, min_identical)
        if diffs:
            problems += verdict
        report["shards"][shard] = {"episodes": picked, "rows": rows, "encode_meta": meta["path"],
                                   "batch_size": meta["batch_size"], "dtype": meta["dtype"], **stats,
                                   "problems": problems, "ok": not problems}
    report["ok"] = (bool(report["shards"]) and not coverage and not contracts
                    and all(s["ok"] for s in report["shards"].values()))
    return report


def fmt(v, spec):
    return format(v, spec) if _finite(v) else str(v)


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
    for p in rep["coverage_problems"] + rep["contract_problems"]:
        print(f"corpus FAILED: {p}")
    for name, s in rep["shards"].items():
        print(f"shard {name}: {'ok' if s['ok'] else 'FAILED'} mean_abs_diff={fmt(s['mean_abs_diff'], '.3g')} "
              f"identical={fmt(s['fraction_identical'], '.1%')} vae_psnr={s['vae_psnr']}")
    return 0 if rep["ok"] else 2


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--latents-dir", required=True)
    p.add_argument("--parquet-dir", required=True, help="the raw recordings the corpus was encoded from")
    p.add_argument("--episodes", default="",
                   help="sample only these ids (A:B or a comma list); each must be encoded and listed by a shard "
                        "log, and is checked under its own shard's settings")
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
