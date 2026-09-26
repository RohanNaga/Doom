"""
Fine-tune a latent-diffusion VAE decoder on Doom frames (GameNGen section 3.2.2).

The encoder stays frozen, so the latents both backbones train on are unchanged; only the
decoder learns to render Doom's HUD digits and textures from them. Loss is MSE against the
lossless target frame (GameNGen's choice); `--lpips-weight` adds a perceptual term as the P1
variant. Reports full-frame and HUD-crop (bottom 32 rows) PSNR and LPIPS on held-out frames
before and after, which is the VAE-ceiling row of every results table.

Defaults reproduce the sd-vae-ft-mse tune exactly. `--vae-id` / `--vae-subfolder` point the
same recipe at any other `AutoencoderKL` (e.g. the 16-channel Flux/SD3 autoencoder) so the
reconstruction ceiling of a candidate latent space is measured under an identical fine-tune.

`--stream-dir` replaces the cached training array with a streamed sample of a corpus too large
to decode into memory, and `--max-steps` / `--max-hours` / `--ckpt-every-hours` turn the run
into a budgeted one that leaves an hourly checkpoint behind. Everything else is unchanged.

`provenance.json`, written beside `metrics.json` and inside every saved decoder directory, lists
the exact episode ids (and maps) the decoder trained on. `decoder_provenance.py` reads it to decide
whether a score made with this decoder may claim unseen maps. A streamed tune of the dense corpus
must name its episodes with `--stream-ids`, inside the segment's train range.

`--val-dir` / `--val-ids` take the validation frames from named held-out episodes instead of the
split's `val` list of `--in-dir`; for a dense segment they must lie inside its validation range
(`validation_episode_ids`). The decoder the run delivers is always the terminal one, `<out-dir>/vae`;
the hourly checkpoint that scores best on the validation frames is recorded in metrics.json beside
it (`checkpoint_selection`), never swapped in for it.

Usage:
    python finetune_decoder.py --in-dir raw_arnold --split split_arnold.json \
        --out-dir vae_decoder_arnold --train-frames 50000 --val-frames 2000 --epochs 2 --device cuda:3

    python finetune_decoder.py --vae-id Alpha-VLLM/Lumina-Image-2.0 --vae-subfolder vae \
        --latent-channels 16 --scaling-factor 0.3611 --shift-factor 0.1159 ...

    python finetune_decoder.py --stream-dir raw_arnold_dense/arenas --stream-ids 0:2000 \
        --stream-frames 400000 --stream-episodes 2000 --val-dir raw_arnold_dense/arenas --val-ids 6000:6100 \
        --workers 8 --max-steps 12000 --max-hours 4 --ckpt-every-hours 1 --lpips-weight 0 ...
"""
import argparse
import glob
import io
import json
import os
import random
import shutil
import time

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image

from doomdit_utils import build_vae, latent_contract

PAD_TO = 256
HUD_ROWS = 32


def trainable_decoder_params(vae):
    """Freeze the encoder side, unfreeze the decoder side. Returns the decoder parameter list.

    The 16-channel Flux/SD3 autoencoder sets `use_quant_conv=False`, so `quant_conv` and
    `post_quant_conv` are None there and must be skipped rather than assumed.
    """
    frozen = [vae.encoder] + ([vae.quant_conv] if vae.quant_conv is not None else [])
    train = [vae.decoder] + ([vae.post_quant_conv] if vae.post_quant_conv is not None else [])
    for m in frozen:
        for p in m.parameters():
            p.requires_grad_(False)
    params = [p for m in train for p in m.parameters()]
    for p in params:
        p.requires_grad_(True)
    return params


def save_vae(vae, out_dir, channels_last=False, name="vae"):
    """Write the autoencoder to `<out_dir>/<name>`, proving it reloads before it replaces the old one.

    `--channels-last` calls `vae.to(memory_format=torch.channels_last)`, which leaves every 4-D
    conv weight strided as NHWC, and `safetensors.torch.save_file` refuses a non-contiguous
    tensor. On 2026-09-18 that raised after `config.json` was written and a finished two-epoch
    Flux-VAE tune was lost, so the weights are packed back into NCHW here, written to a scratch
    directory, reloaded from disk and compared tensor by tensor against the live model. Only
    then does the scratch directory replace the previous checkpoint. The model is left exactly
    as it was found, layout included, so an epoch checkpoint does not disturb the next epoch.

    `name` is the subdirectory, so an hourly checkpoint of a budgeted run can be written next to
    the run's current best without either one overwriting the other.

    Returns the checkpoint directory.
    """
    final, scratch = os.path.join(out_dir, name), os.path.join(out_dir, name + ".saving")
    live = {k: v.detach().cpu().clone() for k, v in vae.state_dict().items()}
    vae.to(memory_format=torch.contiguous_format)
    for t in list(vae.parameters()) + list(vae.buffers()):
        t.data = t.data.contiguous()
    try:
        shutil.rmtree(scratch, ignore_errors=True)
        vae.save_pretrained(scratch)
        reloaded = AutoencoderKL.from_pretrained(scratch).state_dict()
        bad = [k for k in live if k not in reloaded or not torch.equal(reloaded[k], live[k])]
        if bad or set(reloaded) != set(live):
            raise SystemExit(f"saved decoder does not reload identically: {sorted(bad)[:5]} "
                             f"(extra {sorted(set(reloaded) - set(live))[:5]})")
        shutil.rmtree(final, ignore_errors=True)
        os.rename(scratch, final)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        if channels_last:
            vae.to(memory_format=torch.channels_last)
    print(f"  saved and reloaded {len(live)} tensors to {final}", flush=True)
    return final


def sample_frames(parquet_dir, episode_ids, n, stride, seed):
    """Uniformly sample n decision frames across the given episodes. Returns list of (path, row)."""
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    paths = [os.path.join(parquet_dir, f"ep_{e:05d}.parquet") for e in episode_ids]
    paths = [p for p in paths if os.path.exists(p)]
    per = max(1, n // len(paths) + 1)
    out = []
    for p in paths:
        nrows = pq.ParquetFile(p).metadata.num_rows
        rows = [r for r in range(0, nrows, stride)]
        out += [(p, r) for r in rng.sample(rows, min(per, len(rows)))]
    rng.shuffle(out)
    return out[:n]


def load_frames(items):
    import pyarrow.parquet as pq
    by_path = {}
    for p, r in items:
        by_path.setdefault(p, []).append(r)
    frames = {}
    for p, rows in by_path.items():
        col = pq.read_table(p, columns=["frame"])["frame"]
        for r in rows:
            frames[(p, r)] = np.asarray(Image.open(io.BytesIO(col[r].as_py())).convert("RGB"), dtype=np.uint8)
    return [frames[k] for k in items]


def episode_of(path):
    """The id in `ep_XXXXX.parquet`."""
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def sample_row_groups(parquet_dir, n_frames, seed, max_episodes=0, episode_ids=None):
    """Sample whole parquet row groups of a corpus until they hold `n_frames` frames.

    Returns `(groups, n_frames_selected, n_episodes_touched)` where `groups` is a list of
    `(path, row_group_index)`. Row groups rather than rows are the unit because the recorder
    writes `row_group_size=256`: one row group is about 7.7 MB of PNG, so reading a group to
    reach a single random frame would read the corpus tens of times over. `max_episodes` caps
    how many episode files are opened for metadata, which is one seek each.

    `episode_ids` restricts the files to those ids BEFORE anything is drawn. It used to take every
    file in the directory, which for the dense `arenas` segment meant validation and test episodes
    (docs/REVIEW_2026-09-22.md H4). `None` keeps every file, and the draw is then unchanged.
    """
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(parquet_dir, "ep_*.parquet")))
    if episode_ids is not None:
        keep = {int(e) for e in episode_ids}
        files = [p for p in files if episode_of(p) in keep]
    if not files:
        raise SystemExit(f"no ep_*.parquet under {parquet_dir}"
                         + (" among the requested episode ids" if episode_ids is not None else ""))
    rng = random.Random(seed)
    if max_episodes and max_episodes < len(files):
        files = sorted(rng.sample(files, max_episodes))
    groups = []
    for p in files:
        md = pq.ParquetFile(p).metadata
        groups += [(p, g, md.row_group(g).num_rows) for g in range(md.num_row_groups)]
    rng.shuffle(groups)
    picked, total = [], 0
    for path, g, rows in groups:
        if total >= n_frames:
            break
        picked.append((path, g))
        total += rows
    return picked, total, len({p for p, _ in picked})


def stream_episode_ids(stream_dir, spec):
    """The episode ids a streamed tune may draw from, or None for every file of a non-dense corpus.

    A directory named after a segment of `release/dense_split.json` is the dense corpus, and there
    the ids are required and must lie outside that segment's validation and test ranges; the unseen
    segment is refused outright. A decoder that saw held-out episodes or unseen maps cannot back an
    unseen-map claim, and the 2026-09-22 review found the planned dense tune would have done both.
    """
    from doom_data import check_dense_training_ids, load_dense_split, parse_episode_ids
    ids = parse_episode_ids(spec) if spec else None
    split = load_dense_split()
    seg = os.path.basename(os.path.normpath(stream_dir))
    if seg in split["segments"]:
        ranges = split["segments"][seg]["ranges"]
        if "train" not in ranges:
            raise SystemExit(f"{stream_dir} is the unseen segment {seg}: a decoder tuned on it can never back "
                             "an unseen-map claim, so it is not a tuning corpus")
        if ids is None:
            raise SystemExit(f"{stream_dir} is the dense {seg} segment, which holds validation and test "
                             f"episodes: pass --stream-ids (its train range is {ranges['train']})")
        try:
            check_dense_training_ids(split, seg, ids)
        except ValueError as e:
            raise SystemExit(f"--stream-ids {spec}: {e}")
    return ids


def validation_episode_ids(val_dir, spec, train_dir="", train_ids=None):
    """The episode ids `--val-dir` validation frames may come from, checked before anything is read.

    The validation curve is what the hourly checkpoints are read on, so its episodes must be held
    out from everything a row is later scored as unseen on. The 17-map sample this replaces holds
    maps 1 and 9 to 15, which are unseen for the next-tic rows (Astra's review, 2026-09-26,
    section 5). For a segment of `release/dense_split.json` the ids must lie inside its validation
    range, which keeps out training episodes (the decoder would be read on frames it trained on)
    and the sealed test range; a segment with no validation range (the unseen arenas_678) is
    refused. When the training frames come from the same directory, the ids must also be disjoint
    from the training ids, and `train_ids=None` there means every file, so it is refused.
    """
    from doom_data import assert_disjoint, dense_ids, load_dense_split, parse_episode_ids
    try:
        ids = parse_episode_ids(spec)
    except ValueError as e:
        raise SystemExit(f"--val-ids {spec}: {e}")
    if not ids:
        raise SystemExit(f"--val-dir {val_dir} needs --val-ids: the validation episodes are named, never implied")
    split = load_dense_split()
    seg = os.path.basename(os.path.normpath(val_dir))
    if seg in split["segments"]:
        ranges = split["segments"][seg]["ranges"]
        if "val" not in ranges:
            raise SystemExit(f"{val_dir} is the dense {seg} segment, which has no validation range "
                             f"(only {', '.join(ranges)}): a checkpoint read there is read on held-out maps")
        outside = sorted(set(ids) - set(dense_ids(split, seg, "val")))
        if outside:
            raise SystemExit(f"--val-ids {spec}: {len(outside)} id(s) outside the {seg} validation range "
                             f"({ranges['val']}): {outside[:8]}")
    if train_dir and os.path.normpath(train_dir) == os.path.normpath(val_dir):
        if train_ids is None:
            raise SystemExit(f"the training frames come from every file of {val_dir}, the validation "
                             "episodes included: name the training ids")
        try:
            assert_disjoint(train_ids, ids, "training and validation episode ids")
        except ValueError as e:
            raise SystemExit(str(e))
    return ids


def val_cache_tag(val_dir, ids, n, stride):
    """The frame-cache name of a `--val-dir` sample: segment, id bounds, count and a hash of the ids.

    The split-file key (`<in-dir>_<split file>_val_<n>_s<stride>`) names the FILE, so an edited
    split under the same name would read the old frames back; this name changes with every id.
    """
    import hashlib
    ids = sorted(int(e) for e in ids)
    digest = hashlib.sha1(",".join(map(str, ids)).encode()).hexdigest()[:8]
    seg = os.path.basename(os.path.normpath(val_dir))
    return f"{seg}_valids_{ids[0]}-{ids[-1]}_{len(ids)}ep_{digest}_{n}_s{stride}"


def validation_frames(args, split):
    """(frames, record) of the held-out frames every validation pass and hourly checkpoint is read on.

    Without `--val-dir` this is the split's `val` episodes of `--in-dir` under the old cache name,
    unchanged. With it, the ids of `--val-ids` after `validation_episode_ids`, every one of which
    must exist, cached under `val_cache_tag`.
    """
    from doom_data import load_dense_split, parse_episode_ids
    if not args.val_dir:
        tag = f"{split_tag(args)}_val_{args.val_frames}_s{args.stride}"
        frames = cached_frames(args.frame_cache, tag, lambda: load_frames(
            sample_frames(args.in_dir, split["val"], args.val_frames, args.stride, 1)))
        return frames, {"dir": args.in_dir, "split": args.split, "ids": sorted(split["val"])}
    if args.stream_dir:
        try:
            train_ids = parse_episode_ids(args.stream_ids) if args.stream_ids else None
        except ValueError as e:
            raise SystemExit(f"--stream-ids {args.stream_ids}: {e}")
        train_dir = args.stream_dir
    else:
        train_dir, train_ids = args.in_dir, list(split["train"])
    ids = validation_episode_ids(args.val_dir, args.val_ids, train_dir, train_ids)
    missing = [e for e in ids if not os.path.exists(os.path.join(args.val_dir, f"ep_{e:05d}.parquet"))]
    if missing:
        raise SystemExit(f"{len(missing)} validation episode(s) missing from {args.val_dir}: {missing[:8]}; "
                         "the validation set is exactly the ids named")
    tag = val_cache_tag(args.val_dir, ids, args.val_frames, args.stride)
    frames = cached_frames(args.frame_cache, tag, lambda: load_frames(
        sample_frames(args.val_dir, ids, args.val_frames, args.stride, 1)))
    seg = os.path.basename(os.path.normpath(args.val_dir))
    return frames, {"dir": args.val_dir, "segment": seg if seg in load_dense_split()["segments"] else None,
                    "ids": ids, "frames": len(frames), "stride": args.stride, "frame_cache": tag}


def split_tag(args):
    """The cache prefix of a sample drawn with `--in-dir` and `--split`."""
    return f"{os.path.basename(args.in_dir.rstrip('/'))}_{os.path.basename(args.split)}"


def checkpoint_selection(history, step, after):
    """Which saved decoder is the run's decoder: always the terminal one, `<out-dir>/vae`.

    The hourly checkpoints are diagnostics of the ceiling-against-presentations curve. The one that
    scores best on the validation frames is recorded beside the terminal one and never swapped in
    for it: a checkpoint chosen on the validation curve carries that choice into every number later
    scored on those episodes. A tie names the terminal checkpoint.
    """
    terminal = {"dir": "vae", "step": step, **after}
    hourly = [{"dir": f"vae_h{e['hour_ckpt']}", "step": e["step"], "hour": e["hour_ckpt"],
               **{k: v for k, v in e.items() if k not in ("step", "hour_ckpt")}}
              for e in history if "hour_ckpt" in e]
    cands = [terminal] + hourly
    return {"rule": "terminal", "selected": "vae", "terminal": terminal,
            "best_on_validation": max(cands, key=lambda c: c["psnr"]), "candidates": cands}


def _maps_of(paths):
    """Sorted map ids of these recordings, from each file's first `map_id`."""
    import pyarrow.parquet as pq
    maps = set()
    for p in paths:
        col = pq.read_table(p, columns=["map_id"])["map_id"]
        if len(col):
            maps.add(int(col[0].as_py()))
    return sorted(maps)


def training_episodes(corpus_dir, ids, groups=None):
    """One `train_episodes` entry: directory, dense segment (or None), exact ids and maps.

    Maps of a dense segment follow from the id (`record_arnold.py:322-323`); any other corpus is
    read from its files, so a 17-map corpus that contains the unseen arenas says so.
    """
    from doom_data import dense_episode_map, load_dense_split
    split = load_dense_split()
    seg = os.path.basename(os.path.normpath(corpus_dir))
    ids = sorted(int(e) for e in ids)
    if seg in split["segments"]:
        maps = sorted({dense_episode_map(split["segments"][seg]["maps"], e) for e in ids})
    else:
        seg = None
        maps = _maps_of([os.path.join(corpus_dir, f"ep_{e:05d}.parquet") for e in ids])
    entry = {"dir": corpus_dir, "segment": seg, "ids": ids, "maps": maps}
    if groups is not None:
        entry["row_groups"] = [[os.path.basename(p), int(g)] for p, g in groups]
    return entry


def write_provenance(path, record):
    """`provenance.json` in a directory, atomically."""
    tmp = os.path.join(path, f"provenance.json.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(record, f, indent=1)
    os.replace(tmp, os.path.join(path, "provenance.json"))


class RowGroupFrames(torch.utils.data.IterableDataset):
    """One pass over the sampled row groups, shuffled through a per-worker buffer.

    The dense corpus is 1.8 TB, so the cached-array path above cannot be used: 400k frames at
    320x240x3 is 92 GB. Each selected row group is read once and all of its frames are used,
    which keeps the read amplification at 1, and the 256 consecutive tics that a group holds
    are then broken up by a reservoir of undecoded PNG blobs, so a batch of 32 is drawn from
    about `buffer / 256` different groups. The blobs are held compressed (about 30 KB each) and
    decoded on the way out, which is what keeps the buffer inside a worker's memory.
    """

    def __init__(self, groups, buffer=4096, seed=0):
        self.groups, self.buffer, self.seed, self.epoch = list(groups), buffer, seed, 0

    def __iter__(self):
        import pyarrow.parquet as pq
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        rng = random.Random(self.seed * 1000003 + self.epoch * 1009 + wid)
        mine = self.groups[wid::nw]
        rng.shuffle(mine)
        buf = []
        for path, g in mine:
            col = pq.ParquetFile(path).read_row_group(g, columns=["frame"])["frame"]
            for i in range(len(col)):
                buf.append(col[i].as_py())
                if len(buf) >= self.buffer:
                    yield _decode_frame(_pop_random(buf, rng))
        rng.shuffle(buf)
        for blob in buf:
            yield _decode_frame(blob)


def _pop_random(buf, rng):
    """Remove and return a random element in O(1) by swapping it to the end."""
    j = rng.randrange(len(buf))
    buf[j], buf[-1] = buf[-1], buf[j]
    return buf.pop()


def _decode_frame(blob):
    # np.array, not np.asarray: PIL hands back a read-only buffer and torch.from_numpy warns on it.
    return torch.from_numpy(np.array(Image.open(io.BytesIO(blob)).convert("RGB"), dtype=np.uint8))


def stream_batches(dataset, batch_size, workers, max_batches):
    """Yield `max_batches` uint8 batches, starting a fresh shuffled pass whenever one ends."""
    from torch.utils.data import DataLoader
    n = 0
    while n < max_batches:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, drop_last=True,
                            prefetch_factor=4 if workers else None, persistent_workers=False)
        empty = True
        for x in loader:
            empty = False
            yield x
            n += 1
            if n >= max_batches:
                return
        dataset.epoch += 1
        if empty:
            raise SystemExit("streaming dataset yielded no batches")


def cached_frames(cache_dir, tag, build):
    """Decode the parquet JPEGs once and reuse the uint8 array across runs.

    Two decoder tunes on the same frames (one per candidate latent space) must see an
    identical sample, and the JPEG decode costs about five minutes for 50k frames.
    """
    if not cache_dir:
        return build()
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, tag + ".npy")
    if os.path.exists(path):
        print(f"  frame cache hit {path}", flush=True)
        return np.load(path, mmap_mode="r")
    frames = np.stack(build())
    np.save(path, frames)
    return frames


def to_tensor(frames_u8, device):
    """NHWC uint8 frames (a list of arrays, or an already collated tensor) -> padded NCHW [-1, 1]."""
    batch = frames_u8 if torch.is_tensor(frames_u8) else torch.from_numpy(np.stack(frames_u8))
    x = batch.to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return torch.nn.functional.pad(x, (0, 0, 0, PAD_TO - x.shape[2]))


def psnr(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(4.0 / mse)      # inputs in [-1, 1]


@torch.no_grad()
def evaluate(vae, frames, device, lpips_fn, bs=32):
    vae.eval()
    tot = {"psnr": [], "psnr_hud": [], "lpips": [], "lpips_hud": []}
    for i in range(0, len(frames), bs):
        x = to_tensor(frames[i:i + bs], device)
        z = vae.encode(x).latent_dist.mean
        y = vae.decode(z).sample.clamp(-1, 1)
        x, y = x[:, :, :240], y[:, :, :240]
        tot["psnr"].append(psnr(x, y).cpu())
        tot["psnr_hud"].append(psnr(x[:, :, -HUD_ROWS:], y[:, :, -HUD_ROWS:]).cpu())
        if lpips_fn is not None:
            tot["lpips"].append(lpips_fn(x, y).flatten().cpu())
            tot["lpips_hud"].append(lpips_fn(x[:, :, -HUD_ROWS:], y[:, :, -HUD_ROWS:]).flatten().cpu())
    return {k: float(torch.cat(v).mean()) for k, v in tot.items() if v}


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    if not (args.in_dir and args.split) and not (args.stream_dir and args.val_dir):
        raise SystemExit("--in-dir and --split are needed unless the training frames come from --stream-dir "
                         "and the validation frames from --val-dir")
    split = json.load(open(args.split)) if args.split else None
    t0 = time.time()
    # --stream-dir feeds training from a corpus too large to decode into memory; the validation
    # frames are a cached in-memory sample either way, of --in-dir's val split or of --val-dir.
    train_frames = [] if args.stream_dir else cached_frames(
        args.frame_cache, f"{split_tag(args)}_train_{args.train_frames}_s{args.stride}", lambda:
        load_frames(sample_frames(args.in_dir, split["train"], args.train_frames, args.stride, 0)))
    val_frames, val_record = validation_frames(args, split)
    print(f"loaded {len(train_frames)} train and {len(val_frames)} val frames in {time.time() - t0:.0f}s", flush=True)

    stream, stream_info = None, None
    if args.stream_dir:
        if args.max_steps <= 0:
            raise SystemExit("--stream-dir needs --max-steps: the stream has no epoch to end on")
        t0 = time.time()
        allowed = stream_episode_ids(args.stream_dir, args.stream_ids)
        groups, n_frames, n_eps = sample_row_groups(args.stream_dir, args.stream_frames, args.seed,
                                                    args.stream_episodes, allowed)
        stream = RowGroupFrames(groups, args.stream_buffer, args.seed)
        stream_info = {"dir": args.stream_dir, "row_groups": len(groups), "frames": n_frames,
                       "episodes": n_eps, "buffer": args.stream_buffer, "workers": args.workers,
                       "index_seconds": time.time() - t0, "episode_ids_allowed": args.stream_ids or "all"}
        print("stream:", json.dumps(stream_info), flush=True)
    # the exact episodes this decoder trains on: an unseen-map claim is refused for a decoder that saw
    # a held-out episode or an unseen map (decoder_provenance.py), so they are written down, not implied
    if stream is not None:
        train_eps = [training_episodes(args.stream_dir, {episode_of(p) for p, _ in groups}, groups)]
    else:
        present = [e for e in split["train"] if os.path.exists(os.path.join(args.in_dir, f"ep_{e:05d}.parquet"))]
        train_eps = [training_episodes(args.in_dir, present)]
    provenance = {"provenance": "recorded", "recorded_by": "finetune_decoder.py",
                  "corpus": "; ".join(f"{t['dir']} ({len(t['ids'])} episodes, maps {t['maps']})" for t in train_eps),
                  "train_episodes": train_eps,
                  "validation_frames": val_record,
                  "loss": "mse" if args.lpips_weight <= 0 else f"mse + {args.lpips_weight} lpips",
                  "vae_source": args.vae_id or "sd-vae-ft-mse"}

    vae = build_vae(args.vae_id, args.vae_subfolder, device, args.cache_dir,
                    args.latent_channels, args.scaling_factor, args.shift_factor)
    print("latent contract:", json.dumps(latent_contract(vae)), flush=True)
    lpips_fn = None
    if args.lpips_weight > 0 or args.report_lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)
    before = evaluate(vae, val_frames, device, lpips_fn)
    print("before:", json.dumps(before), flush=True)

    params = trainable_decoder_params(vae)
    if args.channels_last:
        # NHWC is what the bf16 tensor cores want; the conv results differ only in the last bits,
        # so this is a throughput switch, not a recipe change. Off by default.
        vae.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    micro, eff = args.batch_size, args.batch_size * args.accum
    if stream is not None:
        # A stream has no epoch, so the budget is the schedule: --max-steps sets the decay horizon
        # and --max-hours is a hard stop that can only end the run early.
        epochs, steps_per_epoch, total = 1, args.max_steps, args.max_steps
    else:
        steps_per_epoch = len(train_frames) // eff
        epochs, total = args.epochs, steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * max(0.05, 1 - s / max(1, total)))
    print(f"{total} updates at effective batch {eff} (micro {micro} x accum {args.accum}), "
          f"{total * eff} presentations", flush=True)
    step, history, t_train = 0, [], time.time()
    batches = stream_batches(stream, micro, args.workers, total * args.accum) if stream is not None else None
    hours_saved, stopped = 0, ""

    def write_metrics(after, selection=None):
        """Record the run so far. Written next to every checkpoint, so a later crash keeps it."""
        with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
            json.dump({"before": before, "after": after, "history": history, "args": vars(args),
                       # what a score measured with this decoder has to state: which frames it was
                       # tuned on and what it was tuned for. An unseen-map claim is about the whole
                       # system, so a decoder that saw arenas 6-8 defeats it whatever the denoiser
                       # was initialised from.
                       # `split_subset` was always "train", including for a streamed tune whose
                       # frames never came from the split; `train_episodes` is the exact record
                       "provenance": {"train_corpus": args.stream_dir or args.in_dir, "split": args.split,
                                      "split_subset": ("none" if args.val_dir else "val (validation frames only)")
                                      if args.stream_dir else "train", "vae_source": args.vae_id or "sd-vae-ft-mse",
                                      "loss": provenance["loss"],
                                      "mse_rows": int(args.mse_rows), "lpips_rows": 240,
                                      "train_episodes": provenance["train_episodes"],
                                      "validation_frames": val_record},
                       "latent_contract": latent_contract(vae), "train_frames": len(train_frames),
                       "val_frames": len(val_frames), "steps": step, "effective_batch": eff,
                       "presentations": step * eff, "stream": stream_info, "stopped": stopped,
                       "train_seconds": time.time() - t_train, "checkpoint_selection": selection,
                       "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device != "cpu" else None},
                      f, indent=1)
        write_provenance(args.out_dir, provenance)

    for ep in range(epochs):
        order = None if stream is not None else np.random.RandomState(ep).permutation(len(train_frames))
        vae.decoder.train()
        for i in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for a in range(args.accum):
                if stream is not None:
                    frames = next(batches)
                else:
                    lo = i * eff + a * micro
                    frames = [train_frames[j] for j in order[lo:lo + micro]]
                x = to_tensor(frames, device)
                if args.channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
                with torch.no_grad():
                    z = vae.encode(x).latent_dist.mean
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                    y = vae.decode(z).sample
                # The MSE term covers the same 240 real picture rows the LPIPS term and every
                # reported metric do. Over all 256 rows, 16/256 = 6.25% of the squared error is
                # the gray padding the encoder added to reach a multiple of 8 -- pixels no metric
                # scores and nothing ever looks at, competing for decoder capacity with the game.
                # `--mse-rows 256` reproduces the old loss for the record.
                rows = min(int(args.mse_rows), x.shape[2])
                loss = torch.mean((y.float()[:, :, :rows] - x[:, :, :rows]) ** 2)
                if args.lpips_weight > 0:
                    loss = loss + args.lpips_weight * lpips_fn(y.float()[:, :, :240].clamp(-1, 1), x[:, :, :240]).mean()
                (loss / args.accum).backward()
                loss_acc += loss.detach().item() / args.accum
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            elapsed = time.time() - t_train
            if step % 100 == 0:
                print(f"step {step}/{total} loss {loss_acc:.5f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{step / elapsed:.3f} upd/s {step * eff / elapsed:.1f} frames/s", flush=True)
            if args.val_every and step % args.val_every == 0 and step < total:
                m = evaluate(vae, val_frames, device, lpips_fn)
                history.append({"step": step, **m})
                print(f"val {step}:", json.dumps(m), flush=True)
                vae.decoder.train()
            # An hourly checkpoint is what makes a budgeted run a curve rather than one number:
            # each one is a point of ceiling against presentations that the gate scorer can read.
            if args.ckpt_every_hours and elapsed >= (hours_saved + 1) * args.ckpt_every_hours * 3600:
                hours_saved += 1
                m = evaluate(vae, val_frames, device, lpips_fn)
                history.append({"step": step, "hour_ckpt": hours_saved, **m})
                print(f"hour {hours_saved} at step {step}:", json.dumps(m), flush=True)
                write_provenance(save_vae(vae, args.out_dir, args.channels_last, f"vae_h{hours_saved}"), provenance)
                write_metrics(m)
                vae.decoder.train()
            if args.max_hours and elapsed >= args.max_hours * 3600:
                stopped = f"max-hours at step {step} of {total}"
                print(stopped, flush=True)
                break
        if stopped:
            break
        mid = evaluate(vae, val_frames, device, lpips_fn)
        history.append({"step": step, **mid})
        print(f"epoch {ep + 1}:", json.dumps(mid), flush=True)
        # Checkpoint every epoch: the tune costs about 2.5 card-hours and the only thing that
        # makes those hours unrecoverable is having no weights on disk when something raises.
        write_provenance(save_vae(vae, args.out_dir, args.channels_last), provenance)
        write_metrics(mid)
    after = evaluate(vae, val_frames, device, lpips_fn)
    print("after:", json.dumps(after), flush=True)
    write_provenance(save_vae(vae, args.out_dir, args.channels_last), provenance)
    selection = checkpoint_selection(history, step, after)
    best = selection["best_on_validation"]
    print(f"selected the terminal decoder vae (step {step}, psnr {after['psnr']:.3f}); best on validation "
          f"{best['dir']} (step {best['step']}, psnr {best['psnr']:.3f}), recorded only", flush=True)
    write_metrics(after, selection)
    print("DONE", flush=True)


def build_parser():
    """Every flag in one place, so a test can read the defaults without a subprocess."""
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="", help="parquet episodes of the cached training sample and, without "
                                                "--val-dir, of the validation frames")
    p.add_argument("--split", default="", help="split file whose train and val ids --in-dir is read with")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-frames", type=int, default=50000)
    p.add_argument("--val-frames", type=int, default=2000)
    p.add_argument("--val-dir", default="", help="parquet corpus the validation frames come from instead of "
                                                 "--in-dir's val split (e.g. raw_arnold_dense/arenas)")
    p.add_argument("--val-ids", dest="val_ids", default="",
                   help="episode ids of --val-dir, as A:B or a comma list; required with it. For a dense segment "
                        "they must lie inside its validation range (arenas: 6000:7000), and the unseen segment "
                        "is refused")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16, help="micro-batch; effective batch is this times --accum")
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lpips-weight", type=float, default=0.0)
    p.add_argument("--mse-rows", dest="mse_rows", type=int, default=240,
                   help="picture rows the MSE term covers. 240 is the real picture, matching the LPIPS term "
                        "and every reported metric; 256 includes the 16 gray padding rows the encoder added, "
                        "which is 6.25%% of the loss spent on pixels nothing scores. Pass 256 only to reproduce "
                        "a tune made before this flag existed")
    p.add_argument("--report-lpips", action="store_true")
    p.add_argument("--val-every", type=int, default=0, help="validate every N updates (0 = epoch ends only)")
    p.add_argument("--channels-last", action="store_true", help="NHWC decoder; throughput only")
    p.add_argument("--frame-cache", default="", help="directory for the decoded uint8 frame sample")
    p.add_argument("--stream-dir", default="", help="parquet corpus to stream training frames from "
                                                    "instead of the cached in-memory sample")
    p.add_argument("--stream-frames", type=int, default=400000, help="distinct frames to sample for the stream")
    p.add_argument("--stream-episodes", type=int, default=0, help="cap on episode files opened (0 = all)")
    p.add_argument("--stream-ids", dest="stream_ids", default="",
                   help="episode ids the stream may draw from, as A:B or a comma list. Required for a dense "
                        "segment directory (arenas: its train range, 0:6000), where it must avoid validation "
                        "and test; the unseen segment is refused. Empty keeps every file of any other corpus")
    p.add_argument("--stream-buffer", type=int, default=4096, help="per-worker shuffle buffer, in frames")
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers for --stream-dir")
    p.add_argument("--max-steps", type=int, default=0, help="update budget and LR decay horizon; required with --stream-dir")
    p.add_argument("--max-hours", type=float, default=0.0, help="wall-clock stop, checked after every update")
    p.add_argument("--ckpt-every-hours", type=float, default=0.0, help="also save <out-dir>/vae_h<N> every N hours")
    p.add_argument("--seed", type=int, default=0, help="stream sampling and shuffle seed")
    p.add_argument("--vae-id", default="", help="AutoencoderKL repo or path (default: sd-vae-ft-mse)")
    p.add_argument("--vae-subfolder", default="")
    p.add_argument("--cache-dir", default=None, help="Hugging Face cache for --vae-id")
    p.add_argument("--latent-channels", type=int, default=None, help="asserted against the VAE config")
    p.add_argument("--scaling-factor", type=float, default=None, help="asserted against the VAE config")
    p.add_argument("--shift-factor", type=float, default=None, help="asserted against the VAE config")
    p.add_argument("--device", default="cuda:3")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
