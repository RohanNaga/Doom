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

Within one directory is not enough (Astra, third review, 2026-09-23): a synthetic train corpus at
scale 1 and a val corpus at scale 2 each passed. With `--space sd15|sd35` the directory's contract
must also be the one its latent space's backbone was pretrained in and is evaluated under
(`space_contracts()`: autoencoder, subfolder, scale, shift, channels, read from the modules that own
those numbers), and with `--contract-peer <dir>` it must equal the other corpus of the same space on
every one of those keys. The gates pass both, for train and for val.

For each shard, for a few sampled episodes (`--episodes` restricts which, inside their own shards):

  1. Re-encode.  Take one full encoder batch at a random batch boundary and the episode's tail batch
     (the short last batch, where cuDNN sees a different shape), read the raw frames at the tics the
     sidecar names, and encode them exactly as `encode_parquet.encode_batch` did, with the settings
     recorded in that shard's `encode_meta_NN.json` (VAE, scale, shift, batch size, dtype). Compare
     with the stored rows: mean (MAE), RMS, 99th-percentile and max absolute difference, and the
     fraction bit-identical.
  2. Decode.  Decode the stored latents through one decoder and score `vae_psnr` against the raw
     frames, unshifted and with the stored rows deliberately shifted by -4, -1, +1 and +4. The
     unshifted alignment must beat EVERY shifted control by a margin.

A shard passes only if its MAE is finite and at most `--max-mae`, its 99th-percentile difference is
at most `--max-p99`, and the unshifted `vae_psnr` exceeds the best shifted control by at least
`--min-shift-margin-db`. `--min-identical` defaults to 0: bit identity is reported, not required,
because harmless rounding breaks it (one fp16 ULP on every value gives MAE 2.4e-4 and 0% identical)
while a real misalignment moves the MAE by orders of magnitude and collapses the margin. Every shard
is reported on its own; nothing is only pooled.

Calibration of the defaults. Same host and batch (Sep 20): MAE 1.45e-5. Across hosts (A4000 with
cuDNN 9.10 against A6000 with cuDNN 9.24, measured 2026-09-23 on ONE 4-channel episode): RMS 2.5e-3 of
latent values (0.3% of the latent standard deviation), p99 |diff| 0.011, 67% bit-identical, decoded
PSNR within 0.001 dB. Hence `--max-mae 5e-3` and `--max-p99 2e-2`, which admit that cross-host
rounding. The tolerance rests on one 4-channel episode: re-check it on the 16-channel SD 3.5 corpus at
gate time (the printed MAE, p99 and margin per shard are what to read), and tighten it when the corpus
is re-encoded on the host that wrote it. The 3 dB margin is a floor for a moving scene; a
near-static stretch can make neighbours tie, which fails the shard rather than certifying it.

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
MAX_MAE = 5e-3              # cross-host calibrated; see the module docstring
MAX_P99 = 2e-2
MIN_IDENTICAL = 0.0
MIN_SHIFT_MARGIN_DB = 3.0
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


# the keys that make a latent mean the same thing in two corpora: which autoencoder, how it was
# normalised, how many channels. Batch size and dtype are execution settings and may differ between
# the train and the val encode (each shard is re-encoded under its own), so they are compared within
# a directory only.
NORMALISATION_KEYS = ("vae_id", "vae_subfolder", "scale", "shift", "channels")


def space_contracts():
    """The latent contract of each latent space: the autoencoder the encoder writes under, its
    normalisation `(z - shift) * scale`, and the channel count the space's backbone was pretrained in.

    sd15 is `encode_parquet.py`'s default (sd-vae-ft-mse, `doomdit_utils.LATENT_SCALE`, no shift) and
    the U-Net's 4 channels; sd35 is SD 3.5 Medium's own autoencoder (`verify_sd35.SD35_VAE`) under the
    subfolder `vae`, 16 channels. The evaluator decodes with the same numbers (`--latent-scale 1.5305
    --latent-shift 0.0609` for SD 3.5), so a corpus under any other contract trains and scores wrong.
    """
    from backbones import BACKBONE_LATENT_CHANNELS, SD35_DEFAULT
    from doomdit_utils import LATENT_SCALE, LATENT_SHIFT, VAE_NAME
    from verify_sd35 import SD35_VAE
    return {"sd15": {"vae_id": VAE_NAME, "vae_subfolder": "", "scale": LATENT_SCALE, "shift": LATENT_SHIFT,
                     "channels": BACKBONE_LATENT_CHANNELS["unet"]},
            "sd35": {"vae_id": SD35_DEFAULT, "vae_subfolder": "vae", "scale": SD35_VAE["scaling_factor"],
                     "shift": SD35_VAE["shift_factor"], "channels": BACKBONE_LATENT_CHANNELS["sd35"]}}


def _norm_value(key, v):
    """One contract value in a comparable form: floats rounded, and a shift of 0 the same as none
    (`normalize_latents` treats them alike)."""
    if key in ("scale", "shift"):
        if v in (None, 0, 0.0):
            return None if key == "shift" else v
        return round(float(v), 6)
    if key == "channels":
        return None if v is None else int(v)
    return v or ""


def measured_channels(latents_dir):
    """The channel count of the first stored latent array, for a shard log that does not record it."""
    from doom_data import list_latent_episodes
    for _, lat_path, _ in list_latent_episodes(latents_dir):
        return int(np.load(lat_path, mmap_mode="r").shape[1])
    return None


def normalisation_of(metas, latents_dir):
    """{key: sorted distinct values} over a directory's shards, channels measured where unrecorded."""
    measured = None
    out = {}
    for k in NORMALISATION_KEYS:
        vals = set()
        for m in metas.values():
            v = m.get(k)
            if k == "channels" and v is None:
                measured = measured if measured is not None else measured_channels(latents_dir)
                v = measured
            vals.add(_norm_value(k, v))
        out[k] = sorted(vals, key=repr)
    return out


def expected_problems(metas, latents_dir, space):
    """One problem per normalisation key on which a directory differs from its space's contract."""
    want = space_contracts()[space]
    got = normalisation_of(metas, latents_dir)
    return [f"{latents_dir} is not the {space} latent contract: {k} is {got[k]!r}, the space needs "
            f"{_norm_value(k, want[k])!r}" for k in NORMALISATION_KEYS if got[k] != [_norm_value(k, want[k])]]


def peer_problems(metas, latents_dir, peer_dir):
    """One problem per normalisation key on which this corpus and its peer (the other corpus of the
    same latent space, train against val) disagree; a peer with no readable shard log is a problem."""
    peer = {}
    names = sorted(_META.search(os.path.basename(p)).group(1)
                   for p in glob.glob(os.path.join(peer_dir, "encode_meta_*.json")))
    for name in names:
        try:
            peer[name] = shard_meta(peer_dir, name)
        except SystemExit as e:
            return [f"contract peer {peer_dir}: {e}"]
    if not peer:
        return [f"contract peer {peer_dir} has no encode_meta_NN.json, so its latent contract is unknown"]
    mine, theirs = normalisation_of(metas, latents_dir), normalisation_of(peer, peer_dir)
    return [f"{latents_dir} and its peer {peer_dir} were encoded under different {k}: {mine[k]!r} vs {theirs[k]!r}"
            for k in NORMALISATION_KEYS if mine[k] != theirs[k]]


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


def shard_verdict(diffs, identical, total, psnr, fresh_psnr, max_mae=MAX_MAE, max_p99=MAX_P99,
                  min_identical=MIN_IDENTICAL, min_shift_margin_db=MIN_SHIFT_MARGIN_DB):
    """(statistics, problems) of one shard's pooled samples; any non-finite statistic is a problem."""
    problems = []
    d = diffs.astype(np.float64)
    mae = float(d.mean()) if d.size else float("nan")
    rms = float(np.sqrt((d ** 2).mean())) if d.size else float("nan")
    p99 = float(np.percentile(d, 99)) if d.size else float("nan")
    max_diff = float(d.max()) if d.size else float("nan")
    frac = identical / total if total else float("nan")
    vae_psnr = {f"{s:+d}" if s else "0": (float(np.mean(v)) if v else None) for s, v in psnr.items()}
    reenc = float(np.mean(fresh_psnr)) if fresh_psnr else None
    shifted = [v for k, v in vae_psnr.items() if k != "0"]
    margin = (vae_psnr["0"] - max(shifted)) if all(_finite(v) for v in shifted + [vae_psnr["0"]]) else None
    stats = {"mean_abs_diff": mae, "rms_diff": rms, "p99_abs_diff": p99, "max_abs_diff": max_diff,
             "fraction_identical": frac, "vae_psnr": vae_psnr, "vae_psnr_reencoded": reenc,
             "shift_margin_db": margin}
    bad = [k for k, v in (("mean_abs_diff", mae), ("rms_diff", rms), ("p99_abs_diff", p99),
                          ("max_abs_diff", max_diff), ("fraction_identical", frac), ("vae_psnr_reencoded", reenc),
                          ("shift_margin_db", margin), *((f"vae_psnr[{k}]", v) for k, v in vae_psnr.items()))
           if not _finite(v)]
    if bad:
        problems.append(f"non-finite or missing statistics: {bad}")
        return stats, problems
    if mae > max_mae:
        problems.append(f"mean |re-encoded - stored| {mae:.3g} > --max-mae {max_mae:g}")
    if p99 > max_p99:
        problems.append(f"p99 |re-encoded - stored| {p99:.3g} > --max-p99 {max_p99:g}")
    if frac < min_identical:
        problems.append(f"only {frac:.1%} of re-encoded values are bit-identical (--min-identical {min_identical:.0%})")
    if margin < min_shift_margin_db:
        best = max((k for k in vae_psnr if k != "0"), key=lambda k: vae_psnr[k])
        problems.append(f"the unshifted alignment ({vae_psnr['0']:.2f} dB) beats the best shifted control "
                        f"({best}: {vae_psnr[best]:.2f} dB) by {margin:.2f} dB, under the "
                        f"{min_shift_margin_db:g} dB margin: the latents may be off by rows against their sidecar, "
                        "or the sampled stretch is too static to tell")
    return stats, problems


def check(latents_dir, parquet_dir, episodes=None, per_shard=2, device="cpu", max_mae=MAX_MAE, max_p99=MAX_P99,
          min_identical=MIN_IDENTICAL, min_shift_margin_db=MIN_SHIFT_MARGIN_DB, seed=0, vae=None, decoder=None,
          cache_dir=None, space=None, contract_peer=None):
    """The per-shard report; `ok` is False if coverage, the contracts or any shard fails.

    `space` also requires the latent space's own contract (`space_contracts`), and `contract_peer`
    equality with the other corpus of that space, on every normalisation key.
    """
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
    if metas and space:
        contracts += expected_problems(metas, latents_dir, space)
    if metas and contract_peer:
        contracts += peer_problems(metas, latents_dir, contract_peer)
    report = {"latents_dir": latents_dir, "parquet_dir": parquet_dir, "shifts": list(SHIFTS),
              "space": space, "contract_peer": contract_peer,
              "thresholds": {"max_mae": max_mae, "max_p99": max_p99, "min_identical": min_identical,
                             "min_shift_margin_db": min_shift_margin_db},
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
                                       psnr, fresh_psnr, max_mae, max_p99, min_identical, min_shift_margin_db)
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
                args.device, args.max_mae, args.max_p99, args.min_identical, args.min_shift_margin_db, args.seed,
                decoder=decoder, cache_dir=args.cache_dir, space=args.space or None,
                contract_peer=args.contract_peer or None)
    text = json.dumps(rep, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)
    for p in rep["coverage_problems"] + rep["contract_problems"]:
        print(f"corpus FAILED: {p}")
    t = rep["thresholds"]
    print(f"thresholds: max_mae={t['max_mae']:g} max_p99={t['max_p99']:g} min_identical={t['min_identical']:g} "
          f"min_shift_margin_db={t['min_shift_margin_db']:g} (cross-host calibrated on one 4-channel episode; "
          "re-check on the SD 3.5 corpus)")
    for name, s in rep["shards"].items():
        print(f"shard {name}: {'ok' if s['ok'] else 'FAILED'} mae={fmt(s.get('mean_abs_diff'), '.3g')} "
              f"rms={fmt(s.get('rms_diff'), '.3g')} p99={fmt(s.get('p99_abs_diff'), '.3g')} "
              f"max={fmt(s.get('max_abs_diff'), '.3g')} identical={fmt(s.get('fraction_identical'), '.1%')} "
              f"margin_db={fmt(s.get('shift_margin_db'), '.2f')} vae_psnr={s.get('vae_psnr')}")
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
    p.add_argument("--max-mae", "--max-abs-diff", dest="max_mae", type=float, default=MAX_MAE,
                   help="largest allowed MEAN absolute difference between re-encoded and stored latents "
                        "(cross-host calibrated: 2.5e-3 RMS measured A4000 against A6000; same host 1.45e-5)")
    p.add_argument("--max-p99", dest="max_p99", type=float, default=MAX_P99,
                   help="largest allowed 99th-percentile absolute difference (cross-host measured 0.011)")
    p.add_argument("--min-identical", dest="min_identical", type=float, default=MIN_IDENTICAL,
                   help="smallest allowed fraction of bit-identical values; 0 by default, because one fp16 ULP "
                        "of harmless rounding gives 0%% identical")
    p.add_argument("--min-shift-margin-db", dest="min_shift_margin_db", type=float, default=MIN_SHIFT_MARGIN_DB,
                   help="how far the unshifted alignment's decoded PSNR must exceed every shifted control's")
    p.add_argument("--space", default="", choices=["", "sd15", "sd35"],
                   help="also require this latent space's contract (autoencoder, scale, shift, channels)")
    p.add_argument("--contract-peer", dest="contract_peer", default="",
                   help="the other corpus of the same latent space (train for val, val for train); its "
                        "normalisation contract must equal this one's")
    p.add_argument("--decoder", default="", help="decode through this VAE instead of the shard's own")
    p.add_argument("--cache-dir", dest="cache_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="", help="also write the JSON report here")
    return p


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
