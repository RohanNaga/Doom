"""
Encode per-episode parquet recordings (from record_episodes.py or record_arnold.py) into
stride-4 latents for training.

Both recording layouts are accepted. A per-tic file keeps every tic, so a decision is `stride` rows
apart; a `record_arnold.py --decision-only` file already holds one row per decision and says so with
`stored_tic_stride` in its parquet metadata, so the control interval spans `stride // stored` rows.
The latents layout, the metadata columns and the chain ids are identical either way.

For every episode: take one frame per agent decision (tic % stride == 0), pad 320x240 to
320x256 at the image level (GameNGen), encode with a frozen `AutoencoderKL` encoder (posterior
mean), normalise the latent the way that autoencoder's own pipeline does, and write
    ep_XXXXX_latents.npy   (T, C, 32, 40) float16
    ep_XXXXX_meta.npz      action, buttons, health, ammo, kills, deaths, frags, pos_x, pos_y,
                           angle, tic, map_id, episode_id[, chain_id]   (all length T)
plus `episodes_NN.jsonl` with per-episode counts and map ids, `canonical_controls.json` under
`--align-decisions`, and `encode_meta_NN.json` recording the latent contract the corpus was
written under. Restartable: existing outputs skip.

The default autoencoder is sd-vae-ft-mse, C = 4, `scaling_factor` 0.18215, no shift: the latent
space every finished row trains in. `--vae-id` / `--vae-subfolder` point the identical pipeline
at any other `AutoencoderKL`, which is how the 16-channel SD 3.5 corpus is built. Normalisation
follows the pipelines exactly, `(z - shift_factor) * scaling_factor`, so a latent written here
decodes with `doomdit_utils.denormalize_latents` and nothing else has to know the numbers.

Usage:
    python encode_parquet.py --in-dir /sata2/.../raw_arnold --out-dir /sata2/.../latents_arnold \
        --stride 4 --batch-size 64 --device cuda:1

    python encode_parquet.py --in-dir /sata2/.../raw_arnold --out-dir /sata2/.../latents_arnold_sd35 \
        --vae-id stabilityai/stable-diffusion-3.5-medium --vae-subfolder vae \
        --latent-channels 16 --scaling-factor 1.5305 --shift-factor 0.0609 \
        --align-decisions --decode-check 16 --stride 4 --batch-size 32 --device cuda:3
"""
import argparse
import glob
import io
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from doomdit_utils import LATENT_SCALE, build_vae, denormalize_latents, latent_contract, normalize_latents
from transitions import decision_rows, stored_tic_stride

PAD_TO = 256
LEGACY_HW = (120, 160)      # April pipeline: frames resized to 160x120, latents (4, 15, 20), no padding
FRAME_ROWS = 240            # rows of real picture inside the 256-row padded frame
META_COLS = ["action", "buttons", "health", "ammo", "kills", "deaths", "frags", "pos_x", "pos_y", "angle", "tic", "map_id", "episode_id"]


def decode(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)


def build_decode_pool(decode_threads, decode_workers=0):
    """The pool `encode_episode` maps `decode` over: threads by default, processes on request.

    PIL releases the GIL inside the PNG decoder itself but not around `Image.convert("RGB")` or the
    `np.asarray` copy, so a thread pool saturates well below the core count on 320x240 frames.
    `--decode-workers N` runs the same `decode` in N processes instead. Both pools' `map` preserves
    order, and `decode` is a pure function of the PNG bytes, so the batch the encoder sees -- and
    therefore every latent byte it writes -- is identical either way.
    """
    if decode_workers and decode_workers > 0:
        from concurrent.futures import ProcessPoolExecutor
        return ProcessPoolExecutor(max_workers=int(decode_workers))
    return ThreadPoolExecutor(decode_threads)


def to_input(frames_u8, device, legacy=False):
    """(B, H, W, 3) uint8 -> (B, 3, 256, 320) in [-1, 1], the tensor the encoder sees."""
    x = torch.from_numpy(frames_u8).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    if legacy:
        return torch.nn.functional.interpolate(x, size=LEGACY_HW, mode="bilinear", align_corners=False, antialias=True)
    if x.shape[2] < PAD_TO:
        x = torch.nn.functional.pad(x, (0, 0, 0, PAD_TO - x.shape[2]))   # bottom rows, zeros (black)
    return x


@torch.no_grad()
def encode_batch(vae, frames_u8, device, dtype, legacy=False, scale=LATENT_SCALE, shift=None):
    x = to_input(frames_u8, device, legacy)
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32 and x.is_cuda)):
        z = vae.encode(x).latent_dist.mean
    return normalize_latents(z.float(), scale, shift).cpu().numpy().astype(np.float16)


@torch.no_grad()
def decode_check(vae, frames_u8, device, dtype, legacy, scale, shift):
    """Encode then decode a few frames and report reconstruction PSNR on the real 320x240 picture.

    This is the cheap proof that the normalisation is the right way round: a wrong `shift_factor`
    or a reciprocal `scaling_factor` still produces plausible-looking latents but destroys the
    round trip, and the number here is the same quantity `eval_tf.py` calls the VAE ceiling.
    """
    lat = encode_batch(vae, frames_u8, device, dtype, legacy, scale, shift)
    z = denormalize_latents(torch.from_numpy(lat).float().to(device), scale, shift)
    rec = vae.decode(z).sample[:, :, :FRAME_ROWS]
    ref = to_input(frames_u8, device, legacy)[:, :, :FRAME_ROWS]
    mse = ((rec.float() - ref) ** 2).flatten(1).mean(1).clamp_min(1e-10)   # [-1, 1] scale, peak 2
    psnr = 10 * torch.log10(4.0 / mse)
    return {"frames": int(len(frames_u8)), "latent_shape": list(lat.shape[1:]),
            "psnr_mean": float(psnr.mean()), "psnr_min": float(psnr.min()),
            "latent_abs_mean": float(np.abs(lat.astype(np.float32)).mean()),
            "latent_abs_max": float(np.abs(lat.astype(np.float32)).max())}


def encode_episode(path, out_dir, vae, device, dtype, stride, batch_size, pool, legacy=False, align_decisions=False,
                   latent_channels=4, scale=LATENT_SCALE, shift=None, every_tic=False):
    import pyarrow.parquet as pq
    if every_tic and align_decisions:
        raise ValueError("--every-tic keeps every row and --align-decisions selects a subset; pick one")
    ep = os.path.basename(path).replace(".parquet", "")
    if legacy:
        ep = "ep_%04d" % int(ep.split("_")[1])      # April layout: ep_XXXX_latents.npy + ep_XXXX_actions.npy
    out_lat = os.path.join(out_dir, f"{ep}_latents.npy")
    if os.path.exists(out_lat):
        return None
    t = pq.read_table(path)
    tic = np.array(t["tic"])
    chain_id = None
    is_decision = None
    stored = stored_tic_stride(t.schema.metadata)
    if every_tic:
        if stored > 1:
            raise ValueError(f"--every-tic needs a per-tic recording, but this file has "
                             f"stored_tic_stride {stored}: the tics in between were never rendered")
        # Keep every row, and additionally mark the rows `--align-decisions` would have kept, with
        # their chain ids. Selecting those rows therefore reproduces the stride-4 corpus exactly,
        # which is what makes a stride-1 result comparable with a stride-4 one.
        keep = np.arange(t.num_rows)
        dec, dec_chain = decision_rows(t["action"].to_numpy(zero_copy_only=False),
                                       np.array(t["buttons"].to_pylist()),
                                       t["deaths"].to_numpy(zero_copy_only=False), stride, CANONICAL, stored)
        is_decision = np.zeros(t.num_rows, dtype=bool)
        chain_id = np.full(t.num_rows, -1, dtype=np.int64)   # -1: this tic belongs to no chain
        is_decision[dec] = True
        chain_id[dec] = dec_chain
    elif align_decisions and "buttons" in t.schema.names:
        keep, chain_id = decision_rows(t["action"].to_numpy(zero_copy_only=False), np.array(t["buttons"].to_pylist()),
                                       t["deaths"].to_numpy(zero_copy_only=False), stride, CANONICAL, stored)
        if len(keep) == 0:
            return dict(episode=ep, frames=0, map_id=-1)
    elif stored > 1:
        keep = np.arange(t.num_rows)                 # a decision-only file holds nothing but decision tics
    else:
        keep = np.flatnonzero(tic % stride == 0)
    cols = {}
    for c in META_COLS:
        if c in t.schema.names:
            arr = t[c].to_numpy(zero_copy_only=False) if c != "buttons" else np.array(t["buttons"].to_pylist())
            cols[c] = arr[keep]
    if chain_id is not None:
        cols["chain_id"] = chain_id
    if is_decision is not None:
        cols["is_decision"] = is_decision
    frames_col = t["frame"]
    lat = []
    for i in range(0, len(keep), batch_size):
        idx = keep[i:i + batch_size]
        raw = [frames_col[int(j)].as_py() for j in idx]
        frames = np.stack(list(pool.map(decode, raw)))
        lat.append(encode_batch(vae, frames, device, dtype, legacy, scale, shift))
    lat = np.concatenate(lat)
    want = (latent_channels, 15, 20) if legacy else (latent_channels, 32, 40)
    assert lat.shape[1:] == want, (lat.shape, want)
    tmp = out_lat + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, lat)
    os.replace(tmp, out_lat)
    if legacy:
        np.save(os.path.join(out_dir, f"{ep}_actions.npy"), cols["action"].astype(np.int64))
    else:
        np.savez(os.path.join(out_dir, f"{ep}_meta.npz"), **cols)
    return dict(episode=ep, frames=int(len(keep)), map_id=int(cols["map_id"][0]) if "map_id" in cols else -1)


CANONICAL = None

_EP_PARQUET = re.compile(r"ep_(\d+)\.parquet$")


def episode_id_of(path):
    """The episode id in `ep_XXXXX.parquet`, or None for a name that is not one."""
    m = _EP_PARQUET.search(os.path.basename(path))
    return None if m is None else int(m.group(1))


def parse_shard(spec, num_shards):
    """(shard, num_shards) from `--shard`, which takes either "i" or "i/n".

    The "i/n" form keeps the two numbers on one launcher line, which matters when several cards
    encode disjoint parts of one corpus and a mismatched `--num-shards` would silently make two of
    them write the same episodes.
    """
    if spec in (None, ""):
        return None, int(num_shards)
    s = str(spec)
    if "/" in s:
        i, n = s.split("/", 1)
        i, n = int(i), int(n)
    else:
        i, n = int(s), int(num_shards)
    if n < 1 or not 0 <= i < n:
        raise ValueError(f"--shard {spec!r} is not a shard of {n}: need 0 <= i < n")
    return i, n


def select_paths(in_dir, episode_ids="", max_episodes=0, shard=None, num_shards=1):
    """The parquet files this process encodes, in a deterministic order.

    Three filters, applied in this order and each documented on its flag:

      * `--episode-ids A:B` keeps the episodes whose *id* falls in the half-open range, read from
        the filename. This is what lets the dense corpus's held-out evaluation latents be encoded
        straight out of the 8,000-episode directory, with no symlink farm and no second copy.
      * `--max-episodes N` truncates what is left to the first N, for a smoke run.
      * `--shard i/n` takes every n-th of what is left. Sharding *after* the id filter means shard
        i of a range is a deterministic subset of that range, so several cards can fill one output
        directory without overlapping.
    """
    from doom_data import parse_episode_ids
    paths = sorted(glob.glob(os.path.join(in_dir, "ep_*.parquet")))
    if episode_ids:
        keep = set(parse_episode_ids(episode_ids))
        paths = [p for p in paths if episode_id_of(p) in keep]
    if max_episodes:
        paths = paths[:int(max_episodes)]
    shard, num_shards = parse_shard(shard, num_shards)
    if shard is not None:
        paths = paths[shard::num_shards]
    return paths


def write_meta(args, contract, scale, shift, check, stored=1):
    """Record what a consumer of this directory has to know: the latent contract and the encoder."""
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git = "?"
    meta = {"vae_id": args.vae_id or "stabilityai/sd-vae-ft-mse", "vae_subfolder": args.vae_subfolder,
            "latent_contract": contract, "scaling_factor_applied": scale, "shift_factor_applied": shift,
            "pad_to": PAD_TO, "legacy": bool(args.legacy), "stride": args.stride, "source_stored_tic_stride": stored,
            "align_decisions": bool(args.align_decisions), "every_tic": bool(args.every_tic),
            "row_semantics": "one row per tic" if args.every_tic else f"one row per {args.stride} tics",
            "decode_check": check,
            "git": git, "torch": torch.__version__, "args": vars(args)}
    with open(os.path.join(args.out_dir, f"encode_meta_{args.shard or 0:02d}.json"), "w") as f:
        json.dump(meta, f, indent=1)


def main(args):
    global CANONICAL
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    if args.every_tic and args.align_decisions:
        raise SystemExit("--every-tic keeps every row and --align-decisions selects a subset; pick one")
    # --every-tic also needs the canonical table, because it marks the decision rows as it goes
    if args.align_decisions or args.every_tic:
        # canonical control bits per action id over the whole recording, so every shard filters identically
        import pyarrow.parquet as pq
        from transitions import canonical_table
        acts, btns = [], []
        for p in sorted(glob.glob(os.path.join(args.in_dir, "ep_*.parquet"))):
            t = pq.read_table(p, columns=["action", "buttons"]); acts.append(t["action"].to_numpy(zero_copy_only=False)); btns.append(np.array(t["buttons"].to_pylist()))
        if args.canonical:
            # a small corpus (evaluation set) reuses the main corpus's table so the transition filter is identical
            CANONICAL = {int(k): v for k, v in json.load(open(args.canonical)).items()}
        else:
            CANONICAL = canonical_table(np.concatenate(acts), np.concatenate(btns))
        json.dump(CANONICAL, open(os.path.join(args.out_dir, "canonical_controls.json"), "w"), indent=1)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    vae = build_vae(args.vae_id, args.vae_subfolder, device, args.cache_dir,
                    latent_channels=args.latent_channels, scaling_factor=args.scaling_factor,
                    shift_factor=args.shift_factor)
    contract = latent_contract(vae)
    scale = args.scaling_factor if args.scaling_factor is not None else contract["scaling_factor"]
    shift = args.shift_factor if args.shift_factor is not None else contract["shift_factor"]
    channels = contract["latent_channels"]
    print(f"latent contract: {json.dumps(contract)}; writing (z - {shift or 0}) * {scale}", flush=True)
    # normalise "i/n" to the pair before anything formats the shard index into a filename
    args.shard, args.num_shards = parse_shard(args.shard, args.num_shards)
    paths = select_paths(args.in_dir, args.episode_ids, args.max_episodes, args.shard, args.num_shards)
    stored = 1
    if paths:
        import pyarrow.parquet as pq
        stored = stored_tic_stride(pq.read_schema(paths[0]).metadata)
    print(f"{len(paths)} episodes on {device}, source rows {stored} tic(s) apart", flush=True)
    check = None
    if args.decode_check and paths:
        import pyarrow.parquet as pq
        t = pq.read_table(paths[0], columns=["frame"])
        with build_decode_pool(args.decode_threads, args.decode_workers) as pool:
            frames = np.stack(list(pool.map(decode, [t["frame"][i].as_py() for i in range(min(args.decode_check, t.num_rows))])))
        check = decode_check(vae, frames, device, dtype, args.legacy, scale, shift)
        print(f"decode check on {os.path.basename(paths[0])}: {json.dumps(check)}", flush=True)
    write_meta(args, contract, scale, shift, check, stored)
    summary_path = os.path.join(args.out_dir, f"episodes_{args.shard or 0:02d}.jsonl")
    t0 = time.time(); n_frames = 0
    with build_decode_pool(args.decode_threads, args.decode_workers) as pool:
        for k, p in enumerate(paths):
            r = encode_episode(p, args.out_dir, vae, device, dtype, args.stride, args.batch_size, pool, args.legacy,
                               args.align_decisions, channels, scale, shift, args.every_tic)
            if r is None:
                continue
            n_frames += r["frames"]
            with open(summary_path, "a") as f:
                f.write(json.dumps(r) + "\n")
            if (k + 1) % 10 == 0:
                print(f"  {k + 1}/{len(paths)} episodes, {n_frames:,} frames, {n_frames / (time.time() - t0):.0f} frames/s", flush=True)
    print("DONE", flush=True)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--decode-threads", type=int, default=8)
    p.add_argument("--decode-workers", type=int, default=0,
                   help="decode PNG frames in N separate processes instead of the thread pool. The\n                        batch order and every output byte are identical; this only removes the GIL\n                        contention in Image.convert and the array copy (0 = the thread pool)")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--shard", default=None,
                   help="this process's share of the selected episodes, as \"i\" (with --num-shards) or \"i/n\". "
                        "Sharding happens after --episode-ids, so shard i of a range is a deterministic subset "
                        "of that range and several cards can fill one output directory safely")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--episode-ids", dest="episode_ids", default="",
                   help="encode only these episode ids, as A:B (half-open, like a Python slice) or a comma list, "
                        "read from the ep_XXXXX.parquet filename. This is how the dense corpus's held-out "
                        "evaluation latents are built straight out of the 8,000-episode directory")
    p.add_argument("--episode-range", dest="episode_ids", help=argparse.SUPPRESS)   # alias
    p.add_argument("--canonical", default=None, help="canonical_controls.json from the main corpus, used instead of recomputing")
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--vae-id", default="", help="AutoencoderKL repo or path (default: sd-vae-ft-mse)")
    p.add_argument("--vae-subfolder", default="", help="subfolder inside --vae-id (e.g. vae for a full pipeline repo)")
    p.add_argument("--cache-dir", default=None, help="Hugging Face cache for --vae-id")
    p.add_argument("--latent-channels", type=int, default=None, help="assert the autoencoder's channel count (16 for SD 3.5)")
    p.add_argument("--scaling-factor", type=float, default=None, help="default: the autoencoder config's own")
    p.add_argument("--shift-factor", type=float, default=None, help="default: the autoencoder config's own (SD 3.5: 0.0609)")
    p.add_argument("--decode-check", type=int, default=0, help="round-trip this many frames of the first episode and print PSNR")
    p.add_argument("--legacy", action="store_true", help="April layout: resize to 160x120, latents (4,15,20), ep_XXXX_actions.npy")
    p.add_argument("--align-decisions", action="store_true", help="one frame per reconstructed agent decision instead of every `stride` tics")
    p.add_argument("--every-tic", action="store_true",
                   help="encode every tic, not one frame per decision, and add `is_decision` and `chain_id` "
                        "columns marking the rows --align-decisions would have kept. Selecting those rows "
                        "reproduces the stride-`stride` corpus, so a stride-1 corpus stays comparable with it. "
                        "Needs a per-tic recording; four times the frames and four times the disk")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
