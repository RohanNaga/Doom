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

The canonical control table (`canonical_controls.json`, modal executed bits per action id) decides
which rows count as verified decisions, so every shard and every corpus of one experiment must use
ONE table. Build it once with `--canonical-only --canonical-ids 0:2000` and pass the file to every
other invocation as `--canonical`, which is read before any recording is opened. Sidecars already
written under a different table are repaired in place with `--rebuild-sidecar-masks`, which
recomputes `is_decision`/`chain_id` from the raw parquet and never re-encodes a latent.

Usage:
    python encode_parquet.py --in-dir /sata2/.../raw_arnold --out-dir /sata2/.../latents_arnold \
        --stride 4 --batch-size 64 --device cuda:1

    python encode_parquet.py --in-dir /sata2/.../raw_arnold_dense/arenas \
        --out-dir /sata2/.../canonical --canonical-only --canonical-ids 0:2000

    python encode_parquet.py --in-dir /sata2/.../raw_arnold --out-dir /sata2/.../latents_arnold_sd35 \
        --vae-id stabilityai/stable-diffusion-3.5-medium --vae-subfolder vae \
        --latent-channels 16 --scaling-factor 1.5305 --shift-factor 0.0609 \
        --align-decisions --decode-check 16 --stride 4 --batch-size 32 --device cuda:3
"""
import argparse
import glob
import io
import itertools
import json
import os
import re
import subprocess
import time
from collections import deque
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


PREFETCH_BATCHES = 2        # batches of frames kept ready ahead of the encoder


def batch_stream(frames_col, keep, batch_size, pool, prefetch=PREFETCH_BATCHES):
    """Yield the uint8 frame batches of rows `keep`, prepared `prefetch` batches ahead of the caller.

    Pulling the PNG bytes out of the parquet column, decoding them and stacking them is host work;
    encoding them is GPU work, and done in lockstep the card idles through every batch's decode.
    One producer thread runs the same three steps, in the same order, on the same row indices, so
    the array handed to `encode_batch` -- its rows, their order, their bytes -- is exactly the array
    the serial loop built.

    Measured on an idle A6000 over two 5,000-tic episodes at batch 64, `--every-tic`, bf16: 86 to
    93 frames/s in steady state against a 96 frames/s ceiling set by the VAE forward alone, which
    holds the card 99.7% busy. What is left is the per-episode parquet read and sidecar write, not
    the per-batch decode, so there is nothing further to hide behind the GPU here.

    Batch BOUNDARIES are deliberately untouched. Under bf16 autocast cuDNN picks its convolution
    algorithm from the batch shape, so the same frame encodes slightly differently in a batch of 32
    than in a batch of 64. A corpus is only reproducible against what is already written if every
    batch keeps its size, so this function changes *when* a batch is built and never *what* is in it.
    """
    prefetch = max(1, int(prefetch))

    def build(i):
        idx = keep[i:i + batch_size]
        raw = [frames_col[int(j)].as_py() for j in idx]
        return np.stack(list(pool.map(decode, raw)))

    starts = iter(range(0, len(keep), batch_size))
    with ThreadPoolExecutor(1) as producer:
        pending = deque(producer.submit(build, i) for i in itertools.islice(starts, prefetch))
        for i in starts:
            ready = pending.popleft()
            pending.append(producer.submit(build, i))   # queued BEFORE the caller blocks on the GPU
            yield ready.result()
        while pending:
            yield pending.popleft().result()


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
    lat = [encode_batch(vae, frames, device, dtype, legacy, scale, shift)
           for frames in batch_stream(t["frame"], keep, batch_size, pool)]
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

CANONICAL_FILE = "canonical_controls.json"

_EP_PARQUET = re.compile(r"ep_(\d+)\.parquet$")


def load_canonical(path):
    """A `canonical_controls.json` written by this module or `build_canonical.py`, keyed by int."""
    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}


def write_canonical(table, out_dir, supplied=None, dry_run=False):
    """Record the table this run filtered with, atomically, and only when there is something to write.

    Three rules, each one a way the plain `open(..., "w")` went wrong:

      * a temporary name plus `os.replace`, because a reader that opens the file while a writer is
        still inside `json.dump` sees a truncated table. Two shards auto-building their own table
        into one output directory do exactly that.
      * nothing is written under `--dry-run`: a mode whose whole promise is to change no file must
        not leave a canonical table behind.
      * a supplied `--canonical` that already IS the output path is left alone. Rewriting the shared
        table from itself is at best a no-op and at worst the truncation above, on the one file
        every other shard is reading.

    Returns the path written, or None.
    """
    path = os.path.join(out_dir, CANONICAL_FILE)
    if dry_run:
        return None
    if supplied and os.path.exists(supplied) and os.path.exists(path) and os.path.samefile(supplied, path):
        return None
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(table, f, indent=1)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


def build_canonical(paths, progress_every=0):
    """The canonical control table of `paths`, streamed one episode at a time.

    Reads only the `action` and `buttons` columns, folds them straight into the counters and drops
    both arrays before opening the next file, so peak memory is one episode's rows rather than the
    corpus's. Nothing here retains a per-episode array: at 40M rows the old whole-corpus form cost
    about 10 GB per process, and six shards each doing it was the host-memory risk behind the
    Sep 20 outage.
    """
    import pyarrow.parquet as pq
    from transitions import canonical_from_counters, control_counters, count_controls
    counters = control_counters()
    for k, p in enumerate(paths):
        t = pq.read_table(p, columns=["action", "buttons"])
        count_controls(counters, t["action"].to_numpy(zero_copy_only=False), t["buttons"].to_pylist())
        del t
        if progress_every and (k + 1) % progress_every == 0:
            print(f"  canonical table: {k + 1}/{len(paths)} episodes, {len(counters)} action ids", flush=True)
    return canonical_from_counters(counters)


def resolve_canonical(args, paths):
    """(table, provenance string) for this run, WITHOUT reading a column when one is supplied.

    `--canonical` is checked before any parquet is opened. That ordering is the fix: the table load
    used to sit after an unconditional scan of every episode in `--in-dir`, so passing the shared
    table skipped nothing at all and each of six shards still paid for the whole 8,000-episode
    corpus. When no table is supplied the scan covers only the episodes this run encodes, or
    `--canonical-ids A:B` when a larger, explicitly named set should define it.
    """
    if args.canonical:
        return load_canonical(args.canonical), f"supplied {args.canonical}"
    if args.canonical_ids:
        table_paths = select_paths(args.in_dir, args.canonical_ids)
        if not table_paths:
            raise SystemExit(f"--canonical-ids {args.canonical_ids} selects no episode in {args.in_dir}")
        where = f"streamed over --canonical-ids {args.canonical_ids} ({len(table_paths)} episodes)"
    else:
        table_paths = paths
        if not table_paths:
            raise SystemExit("no episodes selected, so there is nothing to build a canonical table from; "
                             "pass --canonical (preferred: one table shared by every shard) or --canonical-ids")
        where = f"streamed over this run's {len(table_paths)} episodes"
    return build_canonical(table_paths, progress_every=200), where


# the columns the mask repair reads from raw and copies from the sidecar: both sources must agree,
# or the output is masks from one recording beside controls from another
REPAIR_COLUMNS = ("action", "buttons", "deaths")


def _columns_agree(cols, table, name):
    """Is the sidecar's `name` column the raw recording's, after dtype conversion?"""
    if name not in cols or name not in table.schema.names:
        return False
    side = np.asarray(cols[name])
    if name == "buttons":
        return np.array_equal(side.astype(str), np.array(table["buttons"].to_pylist(), dtype=str))
    return np.array_equal(side.astype(np.int64),
                          table[name].to_numpy(zero_copy_only=False).astype(np.int64))


def rebuild_sidecar_masks(paths, out_dir, canonical, stride=4, dry_run=False):
    """Recompute `is_decision` / `chain_id` in sidecars that already exist, from the raw parquet.

    The latents are never touched and the VAE is never built: the masks are a pure function of the
    recording's `action`, `buttons` and `deaths` columns plus the canonical table, so a corpus that
    was encoded under a per-shard table can be brought onto one shared table without re-encoding a
    single frame. Every other sidecar column is copied through unchanged, and the file is replaced
    atomically.

    Only an `--every-tic` sidecar can be repaired this way, and every column the repair DEPENDS on
    is checked against the raw recording first. The masks are recomputed from raw `action`,
    `buttons` and `deaths` while those same columns are copied through from the old sidecar, so a
    sidecar that disagrees with the recording would come out self-inconsistent: masks derived from
    one recording beside controls from another. Row count, recorded tics and those three columns
    must all match, and the episode is listed as refused rather than silently realigned.
    """
    import pyarrow.parquet as pq
    out = {"episodes": 0, "changed": 0, "unchanged": 0, "missing": [], "refused": []}
    for p in paths:
        ep = os.path.basename(p).replace(".parquet", "")
        meta_path = os.path.join(out_dir, f"{ep}_meta.npz")
        if not os.path.exists(meta_path):
            out["missing"].append(ep)
            continue
        t = pq.read_table(p, columns=["action", "buttons", "deaths", "tic"])
        stored = stored_tic_stride(pq.read_schema(p).metadata)
        if stored > 1:
            out["refused"].append(f"{ep}: stored_tic_stride {stored} is not a per-tic recording")
            continue
        with np.load(meta_path) as z:
            cols = {k: z[k] for k in z.files}
        if len(cols.get("tic", ())) != t.num_rows:
            out["refused"].append(f"{ep}: {len(cols.get('tic', ()))} sidecar rows vs {t.num_rows} raw rows")
            continue
        if not np.array_equal(np.asarray(cols["tic"]).astype(np.int64),
                              t["tic"].to_numpy(zero_copy_only=False).astype(np.int64)):
            out["refused"].append(f"{ep}: sidecar tics differ from the raw tics")
            continue
        disagree = [c for c in REPAIR_COLUMNS if not _columns_agree(cols, t, c)]
        if disagree:
            out["refused"].append(f"{ep}: sidecar {disagree} differ from the raw recording, so the repaired "
                                  "masks would not describe the controls beside them")
            continue
        dec, dec_chain = decision_rows(t["action"].to_numpy(zero_copy_only=False),
                                       np.array(t["buttons"].to_pylist()),
                                       t["deaths"].to_numpy(zero_copy_only=False), stride, canonical, stored)
        is_decision = np.zeros(t.num_rows, dtype=bool)
        chain_id = np.full(t.num_rows, -1, dtype=np.int64)
        is_decision[dec] = True
        chain_id[dec] = dec_chain
        out["episodes"] += 1
        same = (np.array_equal(is_decision, np.asarray(cols.get("is_decision", ())).astype(bool))
                and np.array_equal(chain_id, np.asarray(cols.get("chain_id", ())).astype(np.int64)))
        if same:
            out["unchanged"] += 1
            continue
        out["changed"] += 1
        if dry_run:
            continue
        cols["is_decision"] = is_decision
        cols["chain_id"] = chain_id
        tmp = meta_path + ".tmp.npz"       # np.savez appends .npz to a name without one
        np.savez(tmp, **cols)
        os.replace(tmp, meta_path)
    return out


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
    # normalise "i/n" to the pair before anything formats the shard index into a filename
    args.shard, args.num_shards = parse_shard(args.shard, args.num_shards)
    paths = select_paths(args.in_dir, args.episode_ids, args.max_episodes, args.shard, args.num_shards)
    # --every-tic also needs the canonical table, because it marks the decision rows as it goes.
    # This runs BEFORE the VAE is built and before any frame column is touched, and a supplied
    # --canonical costs no parquet read at all.
    if args.align_decisions or args.every_tic or args.canonical_only or args.rebuild_sidecar_masks:
        CANONICAL, where = resolve_canonical(args, paths)
        print(f"canonical table: {len(CANONICAL)} action ids, {where}", flush=True)
        write_canonical(CANONICAL, args.out_dir, args.canonical, dry_run=args.dry_run)
    if args.canonical_only:
        print(f"wrote {os.path.join(args.out_dir, CANONICAL_FILE)}; pass it to every shard as --canonical")
        print("DONE", flush=True)
        return
    if args.rebuild_sidecar_masks:
        # repair masks written under a per-shard table, without re-encoding a single latent
        r = rebuild_sidecar_masks(paths, args.out_dir, CANONICAL, args.stride, args.dry_run)
        print(f"rebuilt sidecar masks: {json.dumps(r)}", flush=True)
        if r["refused"]:
            raise SystemExit(f"{len(r['refused'])} sidecar(s) refused: {r['refused'][:4]}")
        print("DONE", flush=True)
        return
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    vae = build_vae(args.vae_id, args.vae_subfolder, device, args.cache_dir,
                    latent_channels=args.latent_channels, scaling_factor=args.scaling_factor,
                    shift_factor=args.shift_factor)
    contract = latent_contract(vae)
    scale = args.scaling_factor if args.scaling_factor is not None else contract["scaling_factor"]
    shift = args.shift_factor if args.shift_factor is not None else contract["shift_factor"]
    channels = contract["latent_channels"]
    print(f"latent contract: {json.dumps(contract)}; writing (z - {shift or 0}) * {scale}", flush=True)
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
    p.add_argument("--canonical", default=None,
                   help="canonical_controls.json from the main corpus, used instead of building one. It is read "
                        "BEFORE any recording is opened, so passing it costs zero parquet column reads; this is "
                        "how every shard and every corpus filter decisions identically")
    p.add_argument("--canonical-ids", dest="canonical_ids", default="",
                   help="build the canonical table over these episode ids (A:B or a comma list) instead of the "
                        "episodes this run encodes. Use it with --canonical-only to write the one table the "
                        "whole corpus shares")
    p.add_argument("--canonical-only", action="store_true",
                   help="build the canonical table, write it to --out-dir/canonical_controls.json and stop. No "
                        "VAE is loaded and no frame is read")
    p.add_argument("--rebuild-sidecar-masks", dest="rebuild_sidecar_masks", action="store_true",
                   help="recompute is_decision and chain_id in sidecars that already exist, from the raw parquet "
                        "and --canonical, leaving the latents alone. This is the repair for a corpus encoded "
                        "under a per-shard table")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="with --rebuild-sidecar-masks: report what would change and write nothing")
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
