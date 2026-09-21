"""
Train a DoomDiT world model, any backbone, under one recipe.

Shared between backbones: stride-4 latents, L context frames channel-stacked, single action at
the last context frame, GameNGen context-noise augmentation with a bucket id, velocity target
(or epsilon under --objective eps, which is one cell of the knob grid),
AdamW, bf16 autocast with fp32 master weights, EMA in fp32 on the CPU, held-out latent v-loss
for checkpoint selection. The only thing the --backbone flag changes is the network, and the
only thing --latent-channels changes is the autoencoder the corpus was encoded with (4 for the
SD KL-f8 rows, 16 for sd35's own autoencoder); it defaults to whichever the warm start needs.

Fit check (no data needed):
    python train_wm.py --backbone dit --fit-check 30 --per-gpu-batch 4 --context-frames 32
Training (single process or under `accelerate launch --multi_gpu`):
    accelerate launch --multi_gpu --num_processes 4 train_wm.py --backbone dit \
        --latents-dir data/latents_arnold --split data/split_arnold.json --results-dir results/010-dit-l32
"""
import argparse
import json
import math
import os
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from backbones import ACTION_INJECTIONS, BACKBONES, LATENT_HW, build_model, resolve_latent_channels
from diffusion_v import OBJECTIVES, VDiffusion, noise_augment
from doom_data import PHASE_BUCKETS

ARNOLD_BUTTONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "cards", "arnold", "buttons.json")


def fit_check_control_bits(path=ARNOLD_BUTTONS):
    """Width of the Arnold button vector, read from the recorded card rather than hardcoded.

    A fit check has no corpus to read the width from, and guessing it wrong changes the control
    embedder's input layer and therefore the memory and throughput the check reports. The card says
    19 (9 movement/attack/speed/crouch bits plus 10 weapon-selection bits).
    """
    try:
        with open(path) as f:
            return len(json.load(f)["available_buttons"])
    except (OSError, KeyError):
        return 19


class SyntheticWindows(Dataset):
    def __init__(self, n, context_frames, num_actions, latent_channels=4, action_history=0, control_bits=0,
                 phase_buckets=0):
        self.n, self.L, self.A, self.C = n, context_frames, num_actions, latent_channels
        self.action_history, self.control_bits = action_history, control_bits
        self.phase_buckets = phase_buckets

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # the draw order is context, target, action, and it must stay that way: the fit-check
        # windows are asserted bit-identical to the ones the finished rows' fit checks used
        g = torch.Generator().manual_seed(i)
        ctx = torch.randn(self.C * self.L, *LATENT_HW, generator=g)
        tgt = torch.randn(self.C, *LATENT_HW, generator=g)
        act = (torch.randint(0, 2, (self.action_history, self.control_bits), generator=g).float()
               if self.action_history else torch.randint(0, self.A, (1,), generator=g)[0])
        if self.phase_buckets:
            # a fit check under --phase-conditioning has to supply the fourth column, or the model
            # refuses the batch and the check crashes instead of measuring anything
            return ctx, tgt, act, torch.randint(0, self.phase_buckets, (1,), generator=g)[0]
        return ctx, tgt, act


class SeededCorruption(Dataset):
    """Wraps a validation dataset so each window's timestep, context noise level and noise, and target noise
    are drawn from a generator seeded by the window's own index: identical across checkpoints, backbones,
    batch sizes, and world sizes.

    The phase, when the wrapped dataset supplies one, is appended *after* the corruption tensors, so
    the seven-element layout every existing run validates under is byte-for-byte unchanged.
    """

    def __init__(self, ds, max_level, num_steps):
        self.ds, self.max_level, self.num_steps = ds, max_level, num_steps

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        sample = self.ds[i]
        ctx, tgt, act = sample[0], sample[1], sample[2]
        g = torch.Generator().manual_seed(1234 + int(i))
        t = torch.randint(0, self.num_steps, (1,), generator=g)[0]
        level = torch.rand((), generator=g) * self.max_level
        ctx_eps = torch.randn(ctx.shape, generator=g, dtype=ctx.dtype)
        tgt_noise = torch.randn(tgt.shape, generator=g, dtype=tgt.dtype)
        out = (ctx, tgt, act, t, level, ctx_eps, tgt_noise)
        return out + (sample[3],) if len(sample) > 3 else out


def unpack_batch(batch):
    """(context, target, action, phase or None) from a training batch.

    The stride-4 datasets yield three tensors and the per-tic dataset a fourth,
    `tics_since_decision`, which only `--phase-conditioning` consumes. Reading the fourth
    positionally here is what lets one training loop serve both without touching the old contract.
    """
    return batch[0], batch[1], batch[2], (batch[3] if len(batch) > 3 else None)


def select_episodes(args):
    """(train ids, val ids). Either the split's own lists or the explicit ranges, then the encoded prefix.

    Two routes, and a run states which one it took in config.json:

      * `--split`, as every finished row did: the seeded episode-level split, thinned by
        `--train-fraction`.
      * `--episode-ids A:B --val-episode-ids C:D`, which is what the dense corpus needs: its
        episodes are consecutively numbered and its held-out ranges were fixed in
        `release/dense_split.json` before anything was scored, so a range is the split.

    Either way the two lists are checked disjoint, and `--dense-segment` additionally checks the
    training range against that segment's recorded val and test ranges. A mistyped range is the one
    mistake that silently produces a training-data headline, so it fails here rather than later.
    """
    from doom_data import (assert_disjoint, check_dense_training_ids, limit_to_encoded, load_dense_split,
                           load_split, parse_episode_ids, select_train_episodes)
    if args.episode_ids:
        if not args.val_episode_ids:
            raise SystemExit("--episode-ids names the training episodes explicitly, so --val-episode-ids "
                             "must name the validation ones; there is no split file to fall back on")
        train_ids = parse_episode_ids(args.episode_ids)
        val_ids = parse_episode_ids(args.val_episode_ids)
        if args.dense_segment:
            check_dense_training_ids(load_dense_split(), args.dense_segment, train_ids)
    else:
        split = load_split(args.split)
        train_ids = select_train_episodes(split["train"], args.train_fraction, args.seed)
        val_ids = [int(e) for e in split["val"]]
    assert_disjoint(train_ids, val_ids, "training and validation episode ids")
    if not (args.max_episodes or args.episode_ids):
        return train_ids, val_ids      # the split route as it has always been: the dataset filters
    requested = (len(train_ids), len(val_ids))
    train_ids = limit_to_encoded(args.latents_dir, train_ids, args.max_episodes)
    val_ids = limit_to_encoded(args.val_latents_dir or args.latents_dir, val_ids,
                               args.max_episodes if not args.val_latents_dir else 0)
    if not train_ids:
        raise SystemExit(f"no encoded training episodes in {args.latents_dir} for the requested ids")
    if not val_ids:
        raise SystemExit("no encoded validation episodes for the requested ids; with --max-episodes the "
                         "validation range may lie entirely past the encoded prefix")
    print(f"episodes: {len(train_ids)}/{requested[0]} requested training and {len(val_ids)}/{requested[1]} "
          f"validation episodes are encoded")
    short = (len(train_ids), len(val_ids)) != requested
    if short and not (args.max_episodes or args.allow_partial):
        raise SystemExit(
            f"only {len(train_ids)}/{requested[0]} training and {len(val_ids)}/{requested[1]} validation "
            "episodes of the requested ranges are encoded. A multi-day run must not start on a corpus that "
            "is still being written: the episode set would then depend on when the run happened to start. "
            "Wait for the encode, or pass --allow-partial (or --max-episodes N to fix the prefix on purpose).")
    return train_ids, val_ids


EPISODES_FILE = "train_episodes.json"

# Args that define WHICH EXPERIMENT this is. A resume that changes any of them continues a run under
# a different recipe while keeping its step count, its loss curve and its checkpoint name, so the
# result is a row nobody can describe: a v-prediction checkpoint resumed under `--objective eps`
# trains as eps and saves an eps-labelled checkpoint whose first N steps were velocity.
# The optimization half is here too: a resume that quietly changes the clip threshold, the EMA
# cadence or the spike guard produces one checkpoint whose first N updates followed one recipe and
# the rest another, under one step count and one loss curve. Local execution of the old function
# accepted clip 1.0 -> 999 and EMA decay 0.9999 -> 0.9 without a word.
RESUME_IDENTITY = ("backbone", "latent_channels", "context_frames", "tic_stride", "action_history",
                   "control_bits", "objective", "noise_buckets", "noise_aug_max", "global_batch",
                   "lr", "warm_start", "phase_buckets", "phase_conditioning",
                   "optim", "wd", "warmup", "clip", "ema_every", "ema_decay", "seed",
                   "skip_grad_norm", "skip_grad_after")
# `lr` is the one key a resume is allowed to change, because a diagnosed loss excursion is fixed by
# lowering it; the trainer already rewrites `initial_lr` and logs it as a recipe deviation. It now
# has to be asked for by name, so it cannot happen by forgetting to repeat a flag.
RESUME_OVERRIDABLE = ("lr",)


def parse_resume_override(values):
    """`--resume-override lr=2.5e-5` -> {"lr": 2.5e-5}. Only `lr` is accepted."""
    out = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--resume-override takes key=value, got {item!r}")
        k, v = item.split("=", 1)
        if k not in RESUME_OVERRIDABLE:
            raise SystemExit(f"--resume-override {k} is not allowed; only {list(RESUME_OVERRIDABLE)} may differ "
                             "across a resume, because everything else changes what the run is")
        out[k] = float(v)
    return out


def check_resume_identity(ck, args, overrides):
    """Refuse a resume whose recipe-identity args differ from the checkpoint's.

    `latents_dir` is compared by basename only: the same corpus is mounted at different paths on
    different machines, and a run is legitimately resumed from another checkout. The basename says
    nothing about the CONTENTS of that directory, so it is a cheap early rejection and not the
    corpus check; `check_corpus_identity` compares the manifest once the episode lists are resolved.
    """
    old = ck.get("args") or {}
    if not old:
        return {}
    now = vars(args)
    differ = {}
    for key in RESUME_IDENTITY:
        if key not in old:
            continue                       # a checkpoint from before this flag existed
        a, b = old[key], now.get(key)
        if isinstance(a, float) or isinstance(b, float):
            same = a is not None and b is not None and abs(float(a) - float(b)) < 1e-12
        else:
            same = a == b
        if not same:
            differ[key] = (a, b)
    ao, bo = os.path.basename(str(old.get("latents_dir", "")).rstrip("/")), \
        os.path.basename(str(now.get("latents_dir", "")).rstrip("/"))
    if old.get("latents_dir") is not None and ao != bo:
        differ["latents_dir"] = (ao, bo)
    allowed = {k: v for k, v in differ.items() if k in overrides and abs(overrides[k] - float(v[1])) < 1e-12}
    blocked = {k: v for k, v in differ.items() if k not in allowed}
    if blocked:
        lines = "\n".join(f"  {k}: checkpoint {v[0]!r}, now {v[1]!r}" for k, v in sorted(blocked.items()))
        raise SystemExit(f"--resume {args.resume} was trained under a different recipe:\n{lines}\n"
                         f"Resuming would continue one experiment as another. Only "
                         f"{list(RESUME_OVERRIDABLE)} may differ, and only with --resume-override key=value.")
    for k, (a, b) in allowed.items():
        print(f"resume override {k}: {a!r} -> {b!r} (recipe deviation, stated by --resume-override)")
    return allowed


def pin_episodes(args, train_ids, val_ids, checkpoint=None):
    """Read the episode lists the run started with, or return the freshly resolved ones.

    `limit_to_encoded` resolves the id ranges against what is encoded *now*, so a run resumed after
    more episodes finished encoding would silently train on a larger set than it started with.

    The pin is read from the CHECKPOINT first and only then from the destination results directory:
    a resume into a new directory (a rerun, a copied tree, a different machine) finds no
    `train_episodes.json` there and would re-resolve the list, which is exactly the case the pin
    exists to prevent. Every checkpoint carries `episodes`, so the pin travels with the weights.
    """
    rec = (checkpoint or {}).get("episodes")
    path = os.path.join(args.results_dir, EPISODES_FILE)
    if args.resume and rec is None and os.path.exists(path):
        with open(path) as f:
            rec = json.load(f)
    if args.resume and rec:
        if "val_episodes" not in rec:
            return train_ids, val_ids       # an old-format record from the split route; nothing to pin
        from doom_data import limit_to_encoded
        pinned_train, pinned_val = rec["episodes"], rec["val_episodes"]
        have_t = set(limit_to_encoded(args.latents_dir, pinned_train))
        have_v = set(limit_to_encoded(args.val_latents_dir or args.latents_dir, pinned_val))
        gone = sorted((set(pinned_train) - have_t) | (set(pinned_val) - have_v))
        if gone:
            raise SystemExit(f"this run trained on {len(pinned_train)} + {len(pinned_val)} episodes, and "
                             f"{len(gone)} of them are no longer in the latent directories: {gone[:8]}. "
                             "Resuming would change the training set.")
        print(f"resume pinned to the original {len(pinned_train)} training and {len(pinned_val)} validation episodes")
        return pinned_train, pinned_val
    return train_ids, val_ids


def episode_record(args, train_ids, val_ids):
    """The pin, as it is written to the results directory AND carried in every checkpoint."""
    return {"train_fraction": args.train_fraction, "seed": args.seed,
            "num_episodes": len(train_ids), "episodes": list(train_ids), "val_episodes": list(val_ids)}


CORPUS_KEY = "corpus"
NPY_HEADER_BYTES = 128      # magic, version, header length and the shape/dtype/order dict


def corpus_manifest(latents_dir, episode_ids):
    """Identity of one latent corpus: its episode ids and a fingerprint of their contents.

    Episode ids alone do not identify a corpus. A second directory holding ids 6000..6099 encoded
    by a different autoencoder, at a different batch size or from a different recording passes every
    id check there is, and a differently encoded corpus of the same shape never fails a later shape
    assertion either. The fingerprint closes that: per episode it folds in the id, both files' byte
    sizes, the `.npy` header (shape, dtype, order) and the WHOLE sidecar, which is a few hundred
    kilobytes and carries every metadata column.

    Filesystem mtime is deliberately not part of it. An `rsync -a` copy or a fresh clone of the
    corpus keeps the bytes and must keep the fingerprint; what must change it is a re-encode, a
    mask repair or a truncated file. The latents themselves are 50 MB per episode and are not
    hashed: their header plus their exact size is the affordable proxy, and reading 2,000 sidecars
    costs under a second of a multi-day run.
    """
    import hashlib
    from doom_data import list_latent_episodes
    keep = {int(e) for e in episode_ids}
    h = hashlib.blake2b(digest_size=16)
    eps = []
    for ep, lat_path, meta_path in list_latent_episodes(latents_dir):
        if ep not in keep:
            continue
        eps.append(int(ep))
        h.update(f"{ep}|{os.path.getsize(lat_path)}|{os.path.getsize(meta_path)}|".encode())
        with open(lat_path, "rb") as f:
            h.update(f.read(NPY_HEADER_BYTES))
        with open(meta_path, "rb") as f:
            h.update(f.read())
    return {"dir": os.path.basename(str(latents_dir).rstrip("/")), "episodes": sorted(eps),
            "fingerprint": h.hexdigest()}


def corpus_identity(args, train_ids, val_ids):
    """The manifest pair stored in every checkpoint, for train and for validation."""
    return {"train": corpus_manifest(args.latents_dir, train_ids),
            "val": corpus_manifest(args.val_latents_dir or args.latents_dir, val_ids)}


def check_corpus_identity(ck, corpus):
    """Refuse a resume whose latent corpus is not the one the checkpoint trained on.

    This is the second half of the resume gate. `check_resume_identity` compares the recipe args and
    `pin_episodes` compares the episode ids; neither notices that the directory behind those ids now
    holds different latents. Nothing here is restored: the manifest is only compared.
    """
    old = (ck or {}).get(CORPUS_KEY)
    if not old:
        print("resume: this checkpoint carries no corpus manifest, so only its episode ids are pinned "
              "and the CONTENTS of the latent directories are unverified", flush=True)
        return {}
    differ = {}
    for split in ("train", "val"):
        a, b = old.get(split) or {}, corpus.get(split) or {}
        if a.get("episodes") != b.get("episodes"):
            differ[f"{split} episode ids"] = (f"{len(a.get('episodes') or [])} episodes",
                                              f"{len(b.get('episodes') or [])} episodes")
        elif a.get("fingerprint") != b.get("fingerprint"):
            differ[f"{split} corpus contents"] = (a.get("fingerprint"), b.get("fingerprint"))
    if differ:
        lines = "\n".join(f"  {k}: checkpoint {v[0]}, now {v[1]}" for k, v in sorted(differ.items()))
        raise SystemExit(f"--resume: the latent corpus is not the one this run trained on:\n{lines}\n"
                         "Resuming would continue one experiment on another corpus under the same step "
                         "count and loss curve. Point --latents-dir / --val-latents-dir at the original "
                         "corpus, or start a new run.")
    return old


def build_loaders(args, latent_channels, checkpoint=None):
    """(train dataset, validation subset, training episode ids). The ids are None for a fit check.

    `--train-fraction` thins the *training* episode list only; the validation list and the fixed
    `RandomState(0)` draw of validation windows are the same for every fraction, so a data cell's
    held-out loss is comparable to the full-data cell's.

    `--tic-stride 1` swaps `LatentWindowDataset` (one row per agent decision, four tics apart) for
    `TicWindowDataset` (one row per tic). The sample shapes are identical, so every backbone and the
    whole recipe are unchanged; only the game time between the last context frame and the target
    moves, from 114 ms to 28.6 ms. `--tic-stride 4` is the default and takes the old path untouched.
    """
    if args.fit_check:
        ds = SyntheticWindows(args.per_gpu_batch * 64, args.context_frames, args.num_actions, latent_channels,
                              args.action_history, args.control_bits or fit_check_control_bits(),
                              args.phase_buckets if args.phase_conditioning else 0)
        return ds, None, None
    from doom_data import LatentWindowDataset, TicWindowDataset
    train_ids, val_ids = pin_episodes(args, *select_episodes(args), checkpoint=checkpoint)
    if args.tic_stride == 1:
        def make(d, ids):
            return TicWindowDataset(d, ids, args.context_frames, latent_channels=latent_channels,
                                    with_phase=args.phase_conditioning, phase_buckets=args.phase_buckets,
                                    action_history=args.action_history)
    else:
        def make(d, ids):
            return LatentWindowDataset(d, ids, args.context_frames, latent_channels=latent_channels,
                                       require_chains=args.require_verified_transitions)
    train = make(args.latents_dir, train_ids)
    val = make(args.val_latents_dir or args.latents_dir, val_ids)
    rng = np.random.RandomState(0)
    val_idx = np.sort(rng.choice(len(val), size=min(args.val_windows, len(val)), replace=False))
    return train, Subset(val, val_idx.tolist()), (train_ids, val_ids)


@torch.no_grad()
def ema_update(ema_params, params, decay):
    for e, p in zip(ema_params, params):
        e.mul_(decay).add_(p.detach().float().cpu(), alpha=1.0 - decay)


def ema_keys(raw):
    """Names for the fp32 CPU EMA list, which holds one tensor per *parameter*.

    Earlier code zipped the list against `state_dict().keys()`. Those agree only while a model has
    no persistent buffers: SD 3.5's positional table is one (`pos_embed.pos_embed`), and it sorts
    before the patch projection, so that zip would shift every EMA key by one and silently save an
    EMA whose tensors belong to the wrong weights. Parameter names are the list's own order.
    """
    return [n for n, _ in raw.named_parameters()]


PHASE_PARAM_MARK = "phase_embedder"


def load_init_weights(model, path):
    """Start a NEW run from one of our own checkpoints' weights. Returns (step of the source, ema dict or None).

    Accepts every format `train_wm.py` writes, because all three carry the full state dict under
    `model`: `best.pt` (bf16 weights, no optimizer), a recovery `NNNNNNN.pt` (fp32 weights plus
    optimizer, scheduler, EMA and RNG) and a `snap_*.pt` (bf16 weights and EMA). Only the weights
    and, when present, the EMA are taken. Nothing else is: the step restarts at 0, the optimizer is
    fresh and the warmup runs again, which is the difference between this and `--resume`.

    Key matching is strict, with exactly one exception: the `tics_since_decision` tables that
    `--phase-conditioning` adds have no counterpart in a checkpoint trained without it, so they are
    reported as freshly initialised instead of failing the load. Any other gap is an error, because
    it means the checkpoint belongs to a different architecture and a silent partial load would
    produce a run nobody could interpret.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in ck:
        raise SystemExit(f"--init-from {path} carries no 'model' state dict; is it a train_wm.py checkpoint?")
    try:
        missing, unexpected = model.load_state_dict({k: v.float() for k, v in ck["model"].items()}, strict=False)
    except RuntimeError as e:      # a shape mismatch: same key names, different architecture
        raise SystemExit(f"--init-from {path} does not match this model: {e}") from e
    fresh = [k for k in missing if PHASE_PARAM_MARK in k]
    gap = sorted(set(missing) - set(fresh))
    if gap or unexpected:
        raise SystemExit(f"--init-from {path} does not match this model: missing {gap[:8]}, "
                         f"unexpected {sorted(unexpected)[:8]}. Check --backbone, --context-frames "
                         "and --latent-channels against the source run's config.json.")
    if fresh:
        print(f"--init-from: {len(fresh)} phase-conditioning tensor(s) initialised fresh: {fresh}")
    print(f"initialised weights from {path} (source step {ck.get('step', '?')}); optimizer, step and warmup start over")
    return int(ck.get("step", 0) or 0), ck.get("ema")


def tensor_bytes(obj):
    """Bytes of every tensor reachable in a checkpoint dict: what its file will mostly weigh.

    The local write needs a size before it exists, to decide whether the disk can hold it. Without
    a serialized buffer to measure, the tensor payload is the estimate: `torch.save` writes each
    storage verbatim into a zip archive plus a small pickle, so this is the file size to within a
    few hundred kilobytes.
    """
    total, stack, seen = 0, [obj], set()
    while stack:
        x = stack.pop()
        if torch.is_tensor(x):
            s = x.untyped_storage()
            key = (s.data_ptr(), s.nbytes())
            if key in seen:
                continue
            seen.add(key)
            total += s.nbytes()
        elif isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, (list, tuple, set)):
            stack.extend(x)
    return total


def save_checkpoint(obj, path, remote=None, keep_local=True):
    """Write the checkpoint to the remote copy of record and, space permitting, to the local path.

    `remote` is "user@host:/dir" reached with the key in REMOTE_SSH; the file lands as <dir>/<run>/<name>
    through an atomic rename. The local write is skipped when the disk has less than 1.5x the file free,
    so a full shared disk degrades to remote-only instead of killing the run. Raises only if no copy was written.

    **A local-only save never builds a byte buffer.** `ssh` needs the bytes on stdin, so the remote
    branch still serializes into memory, but with no `remote` the object goes straight to a temporary
    file through `torch.save` and is renamed into place. The buffer was an artifact-sized allocation
    on top of the fp32 model copy and two EMA copies the caller already holds: about 37 GB for an
    SD 3.5 recovery checkpoint, next to which the host has to fit a second training job.

    The object is serialized into an OPEN FILE, not a path: `torch.save` names its zip archive
    after the path it is given, so `torch.save(obj, "0000005.pt.tmp")` would write bytes that
    differ from every checkpoint this code has ever produced. Handed a file object it uses the
    literal name "archive", exactly as the byte buffer did, so a checkpoint saved after this change
    is byte-identical to one saved before it. The rename keeps the file atomic: a failure mid-write
    leaves the previous checkpoint in place and removes the partial temporary.
    """
    import shutil, subprocess
    written = []
    data = None
    if remote:
        import io
        buf = io.BytesIO(); torch.save(obj, buf); data = buf.getvalue()
        host, rdir = remote.split(":", 1)
        rel = os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))
        rpath = os.path.join(rdir, rel)
        cmd = REMOTE_SSH + [host, f"mkdir -p {os.path.dirname(rpath)} && cat > {rpath}.tmp && mv {rpath}.tmp {rpath}"]
        try:
            r = subprocess.run(cmd, input=data, capture_output=True, timeout=1800)
            if r.returncode == 0:
                written.append(rpath)
            else:
                print(f"remote checkpoint write failed: {r.stderr.decode()[-300:]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"remote checkpoint write timed out after 1800 s: {rpath}", flush=True)
    if keep_local or not written:
        size = len(data) if data is not None else tensor_bytes(obj)
        free = shutil.disk_usage(os.path.dirname(path)).free
        if free > 1.5 * size or not written:
            tmp = path + ".tmp"
            try:
                with open(tmp, "wb") as f:
                    if data is not None:
                        f.write(data)
                    else:
                        torch.save(obj, f)
                os.replace(tmp, path); written.append(path)
            except OSError as e:   # a full local disk is survivable once the remote copy exists
                print(f"local checkpoint write failed ({e}); remote copy {'exists' if written else 'MISSING'}", flush=True)
                _unlink(tmp)
            except BaseException:  # a partial file must never be mistaken for a checkpoint
                _unlink(tmp)
                raise
        else:
            print(f"skipped local checkpoint {path}: {free/2**30:.1f} GB free", flush=True)
    if not written:
        raise RuntimeError(f"could not write checkpoint {path} anywhere")
    return written


def _unlink(path):
    try:
        os.remove(path)
    except OSError:
        pass


def prune_remote(remote, run_name, keep):
    """Keep only the newest `keep` rolling checkpoints (NNNNNNN.pt) of one run on the remote copy of record."""
    host, rdir = remote.split(":", 1)
    d = os.path.join(rdir, run_name)
    cmd = REMOTE_SSH + [host, f"cd {d} && ls [0-9]*.pt 2>/dev/null | sort | head -n -{keep} | xargs -r rm -f"]
    subprocess.run(cmd, capture_output=True, timeout=120)


REMOTE_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"] + (["-i", os.environ["DOOM_SSH_KEY"]] if os.environ.get("DOOM_SSH_KEY") else [])

def main(args):
    from accelerate import Accelerator
    from accelerate.utils import set_seed, InitProcessGroupKwargs
    from datetime import timedelta
    acc = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=1))], mixed_precision="bf16", gradient_accumulation_steps=1)
    set_seed(args.seed + acc.process_index)
    device = acc.device
    world = acc.num_processes
    micro_batch = args.per_gpu_batch * world
    assert args.global_batch % micro_batch == 0, f"global batch {args.global_batch} is not a multiple of per-gpu batch x world = {micro_batch}"
    accum = args.global_batch // micro_batch
    is_main = acc.is_main_process

    os.makedirs(args.results_dir, exist_ok=True)
    log_path = os.path.join(args.results_dir, "log.jsonl")

    def log(**kw):
        if is_main:
            kw["time"] = time.time()
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps(kw) + "\n")
            except OSError as e:   # a full disk must not kill training; stdout and the remote copy still get the line
                print(f"log write failed: {e}", flush=True)
            print(json.dumps(kw), flush=True)

    latent_channels = resolve_latent_channels(args.backbone, args.latent_channels)
    phase_buckets = args.phase_buckets if args.phase_conditioning else 0
    if args.phase_conditioning and args.tic_stride != 1:
        raise SystemExit("--phase-conditioning needs per-tic windows (--tic-stride 1); at stride 4 every "
                         "target is a decision tic, so tics_since_decision is 0 for every sample")
    if args.action_history:
        if args.tic_stride != 1:
            raise SystemExit("--action-history conditions on the executed control of each context TIC, so it "
                             "needs --tic-stride 1")
        if args.action_history != args.context_frames:
            raise SystemExit(f"--action-history {args.action_history} must equal --context-frames "
                             f"{args.context_frames}: one executed control per context tic")
        if args.action_dropout > 0:
            raise SystemExit("--action-dropout has no null row to drop to when the conditioning is a button "
                             "vector; set --action-dropout 0")
    # the control embedder's input width is a property of the corpus's button list, so it is read from
    # the corpus rather than assumed (a fit check has no corpus and uses the recorded Arnold width)
    control_bits = args.control_bits
    if args.action_history and not control_bits:
        from doom_data import corpus_control_bits
        control_bits = fit_check_control_bits() if args.fit_check else corpus_control_bits(args.latents_dir)
    # write the RESOLVED width back onto args, because every checkpoint stores `vars(args)` and the
    # evaluators rebuild the control embedder from it; a 0 there builds a width-0 embedder
    args.control_bits = control_bits
    model = build_model(args.backbone, args.num_actions, args.context_frames, args.noise_buckets,
                        grad_ckpt=args.grad_ckpt, warm_start=args.warm_start, cache_dir=args.hf_cache,
                        action_dropout=args.action_dropout, latent_channels=latent_channels,
                        action_inject=args.action_inject, phase_buckets=phase_buckets,
                        action_history=args.action_history, control_bits=control_bits)
    start_step = 0
    if args.resume and args.init_from:
        raise SystemExit("--resume continues a run from its own checkpoint and --init-from starts a new run "
                         "from another run's weights; pick one")
    init_step, init_ema = (None, None)
    if args.init_from:
        init_step, init_ema = load_init_weights(model, args.init_from)
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        check_resume_identity(ck, args, parse_resume_override(args.resume_override))
        model.load_state_dict({k: v.float() for k, v in ck["model"].items()}, strict=True)
        start_step = int(ck.get("step", 0))
        print(f"resumed weights from {args.resume} at step {start_step}" + (" with optimizer, scheduler, EMA, and RNG state" if "optimizer" in ck else " (optimizer state reset, warmup restarts)"))
        # every rank reseeds from (seed, rank, step): distinct per rank, reproducible, and independent of which rank
        # wrote the checkpoint. Data order after a resume is a fresh shuffle from this seed (state-exact, order reshuffled).
        set_seed(args.seed + acc.process_index + 7919 * start_step)
    n_params = sum(p.numel() for p in model.parameters())
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, fused=(device.type == "cuda"))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))

    train_ds, val_ds, episode_lists = build_loaders(args, latent_channels, ck if args.resume else None)
    train_ids = None if episode_lists is None else episode_lists[0]
    # the pin travels with the weights: a resume into a NEW results directory finds no
    # train_episodes.json there and would otherwise re-resolve the episode list
    pin = None if episode_lists is None else episode_record(args, *episode_lists)
    # the corpus manifest travels with the weights too, and a resume is refused if the directories
    # behind the pinned ids no longer hold the latents this run trained on
    corpus = None if episode_lists is None else corpus_identity(args, *episode_lists)
    if args.resume and corpus is not None:
        check_corpus_identity(ck, corpus)
    # more workers than the stride-4 rows used: a per-tic corpus is 125 GB (4-channel) or 410 GB
    # (16-channel), so random window reads miss the page cache and the loader is seek-bound
    extra = {"prefetch_factor": args.prefetch_factor} if args.num_workers > 0 else {}
    loader = DataLoader(train_ds, batch_size=args.per_gpu_batch, shuffle=True, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0, **extra)
    model, opt, loader = acc.prepare(model, opt, loader)   # scheduler stays unwrapped: one step per optimizer update
    diffusion = VDiffusion(device=device, objective=args.objective)
    raw = acc.unwrap_model(model)
    if args.resume and "optimizer" in ck:
        opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        # a resume may deliberately change the learning rate (a diagnosed excursion, logged as a recipe deviation); the restored
        # optimizer state carries the old lr and initial_lr, and LambdaLR scales initial_lr, so both must be overridden
        old_lr = ck["optimizer"]["param_groups"][0].get("initial_lr", ck["optimizer"]["param_groups"][0]["lr"])
        if abs(old_lr - args.lr) > 1e-12:
            for g in opt.param_groups:
                g["initial_lr"] = args.lr; g["lr"] = args.lr * sched.lr_lambdas[0](sched.last_epoch)
            sched.base_lrs = [args.lr for _ in sched.base_lrs]
            print(f"resume overrides learning rate {old_lr:g} -> {args.lr:g} (recipe deviation)")
    # the EMA starts as a copy of the live weights, which after --init-from are already the loaded ones,
    # so "copy live into EMA" needs no extra code; an EMA carried by the source checkpoint overrides it
    ema = [p.detach().float().cpu().clone() for p in raw.parameters()] if args.ema_every > 0 else None
    restore = ck["ema"] if (args.resume and "ema" in ck) else (init_ema if args.init_from else None)
    if ema is not None and restore:
        for e, k in zip(ema, ema_keys(raw)):
            if k in restore and restore[k].shape == e.shape:
                e.copy_(restore[k].float())

    if is_main:
        try:
            git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            git = "?"
        import hashlib
        split_hash = hashlib.md5(open(args.split, "rb").read()).hexdigest() if (not args.fit_check and os.path.exists(args.split)) else None
        with open(os.path.join(args.results_dir, "config.json"), "w") as f:
            json.dump({**vars(args), "git": git, "params": n_params, "world_size": world, "accum": accum,
                       "split_md5": split_hash, "torch": torch.__version__,
                       "resolved_latent_channels": latent_channels,
                       # what a consumer needs to know about the data contract this run trained under:
                       # the frame spacing, the dataset class that produced it, and whether the phase
                       # of the held action was a conditioning signal
                       "tic_stride": args.tic_stride,
                       "dataset_class": type(train_ds).__name__,
                       "resolved_phase_buckets": phase_buckets,
                       "resolved_control_bits": control_bits,
                       "dataset_summary": getattr(train_ds, "summary", None),
                       "init_from": args.init_from or None, "init_from_step": init_step}, f, indent=1)
        if train_ids is not None and not (args.resume and os.path.exists(os.path.join(args.results_dir, EPISODES_FILE))):
            # which episodes this run actually trained on, so a data cell is reproducible from the
            # results directory alone and not only from (split, fraction, seed) -- and so a resume
            # can be pinned to the same set even after more of the corpus finishes encoding
            with open(os.path.join(args.results_dir, EPISODES_FILE), "w") as f:
                json.dump(episode_record(args, train_ids, episode_lists[1]), f, indent=1)
    log(event="start", backbone=args.backbone, params=n_params, world=world, accum=accum, latent_channels=latent_channels,
        per_gpu_batch=args.per_gpu_batch, global_batch=args.per_gpu_batch * world * accum, objective=args.objective,
        tic_stride=args.tic_stride, dataset_class=type(train_ds).__name__, phase_buckets=phase_buckets,
        action_history=args.action_history, control_bits=control_bits,
        # the fraction of candidate windows the per-tic validity contract removed (deaths, tic gaps,
        # map changes), so a "within-life simulation" claim can state it
        dataset_summary=getattr(train_ds, "summary", None),
        init_from=args.init_from or None, init_from_step=init_step,
        train_fraction=args.train_fraction, train_episodes=None if train_ids is None else len(train_ids),
        # state-dict entries the EMA does not cover, i.e. persistent buffers: 0 for every backbone
        # in the SD KL-f8 latent space, 1 for sd35 (its sin-cos positional table)
        buffers_outside_ema=len(raw.state_dict()) - len(ema_keys(raw)))

    def model_fn(ctx, act, bucket, phase=None):
        # the sixth argument is passed only when there is one, so a run without phase conditioning
        # calls the backbones exactly as every finished row did
        if phase is None:
            return lambda xt, t: model(xt, t, act, ctx, bucket)
        return lambda xt, t: model(xt, t, act, ctx, bucket, phase)

    @torch.no_grad()
    def evaluate():
        if val_ds is None:
            return None
        model.eval()
        # fixed corruption per window: the same timesteps, context noise, and target noise for every
        # checkpoint and both backbones, independent of batch size and world size
        vl = DataLoader(SeededCorruption(val_ds, args.noise_aug_max, diffusion.num_steps), batch_size=args.per_gpu_batch, shuffle=False, num_workers=2)
        tot = torch.zeros((), device=device); n = torch.zeros((), device=device)
        bins = torch.zeros(4, device=device); bin_n = torch.zeros(4, device=device)
        for batch in vl:
            ctx, tgt, act, t, level, ctx_eps, tgt_noise = batch[:7]
            phase = batch[7].to(device) if len(batch) > 7 else None
            ctx, tgt, act, t = ctx.to(device), tgt.to(device), act.to(device), t.to(device)
            ctx_n, bucket = noise_augment(ctx, args.noise_aug_max, args.noise_buckets, level=level.to(device), eps=ctx_eps.to(device))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = diffusion.training_loss(model_fn(ctx_n, act, bucket, phase), tgt, noise=tgt_noise.to(device), t=t, per_sample=True)
            tot += loss.detach().sum(); n += loss.numel()
            q = (t * 4) // diffusion.num_steps
            bins.index_add_(0, q, loss.detach()); bin_n.index_add_(0, q, torch.ones_like(loss))
        tot, n = acc.reduce(tot, reduction="sum"), acc.reduce(n, reduction="sum")
        bins, bin_n = acc.reduce(bins, reduction="sum"), acc.reduce(bin_n, reduction="sum")
        model.train()
        return (tot / n).item(), (bins / bin_n.clamp(min=1)).tolist()

    model.train()
    step, t0 = start_step, time.time()
    running, grad_norms = [], []
    skipped = int(ck.get("skipped", 0)) if args.resume else 0
    # probe tensors: input projection, output layer, and adaLN modulation of the first, middle, and last DiT blocks (or U-Net / U-ViT analogues)
    want = ["x_embedder.proj.weight", "final_layer.linear.weight", "blocks.0.adaLN_modulation.1.weight", "blocks.14.adaLN_modulation.1.weight",
            "blocks.27.adaLN_modulation.1.weight", "conv_in.weight", "conv_out.weight", "time_embedding.linear_2.weight", "class_embedding.weight",
            "vae_img_in.proj.weight", "vae_img_out.weight", "transformer_mid_block.attn1.to_q.weight",
            "pos_embed.proj.weight", "proj_out.weight", "context_embedder.weight", "transformer_blocks.11.attn.to_q.weight"]
    probe_params = [(n, p) for n, p in raw.named_parameters() if any(n.endswith(w) for w in want)]
    probe_names = [n for n, _ in probe_params]
    probe_prev = {n: torch.empty_like(p.detach()) for n, p in probe_params}
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    done = False
    max_steps = args.fit_check if args.fit_check else args.steps
    best_val = float(ck.get("best_val", float("inf"))) if args.resume else float("inf")
    val_hist = []
    micro = int(ck.get("micro", 0)) if args.resume else 0
    while not done:
        for batch in loader:
            micro += 1
            ctx, tgt, act, phase = unpack_batch(batch)
            ctx, tgt, act = ctx.to(device, non_blocking=True), tgt.to(device, non_blocking=True), act.to(device)
            phase = None if phase is None else phase.to(device)
            ctx_n, bucket = noise_augment(ctx, args.noise_aug_max, args.noise_buckets)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = diffusion.training_loss(model_fn(ctx_n, act, bucket, phase), tgt) / accum
            acc.backward(loss)
            running.append(loss.item() * accum)
            if micro % accum != 0:
                continue
            gn = acc.clip_grad_norm_(model.parameters(), args.clip if args.clip > 0 else float("inf"))
            grad_norms.append(float(gn))   # pre-clip norm: the instability diagnostic that the loss alone hides
            spike = args.skip_grad_norm > 0 and step >= args.skip_grad_after and float(gn) > args.skip_grad_norm
            if not math.isfinite(float(gn)) or spike:
                # gradients are DDP-averaged before clipping, so all ranks agree; skip the update without advancing the schedule.
                # A finite spike is skipped too when --skip-grad-norm is set: clipping bounds the gradient but not Adam's
                # preconditioned step, and one such step froze the UniDiffuser mid-block attention twice (PaLM-style skip).
                opt.zero_grad(set_to_none=True); skipped += 1
                log(event="skipped_update", step=step, micro=micro, grad_norm=float(gn), threshold=args.skip_grad_norm, skipped_total=skipped)
                if skipped > 20:
                    raise RuntimeError(f"{skipped} non-finite gradient updates; stopping before Adam state is corrupted")
                continue
            measure = bool(probe_names) and (step + 1) % 100 == 0
            if measure:
                with torch.no_grad():
                    for n, p in probe_params:
                        probe_prev[n].copy_(p.detach())
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if measure:
                # actual single-update parameter change relative to weight norm, for tensors adaLN and I/O depend on
                with torch.no_grad():
                    ratios = {n: float((p.detach() - probe_prev[n]).norm() / (probe_prev[n].norm() + 1e-12)) for n, p in probe_params}
                log(event="update_ratio", step=step, **ratios)
            if ema is not None and step % args.ema_every == 0:
                ema_update(ema, raw.parameters(), args.ema_decay ** args.ema_every)
            if step % args.log_every == 0 or step == max_steps:
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
                log(event="train", step=step, loss=float(np.mean(running)), lr=sched.get_last_lr()[0],
                    steps_per_s=(step - start_step) / dt, peak_mem_gb=round(mem, 2),
                    grad_norm=float(np.mean(grad_norms)), grad_norm_max=float(np.max(grad_norms)),
                    clip_frac=float(np.mean([g > args.clip for g in grad_norms])) if args.clip > 0 else 0.0,
                    nonfinite_loss=int(sum(not np.isfinite(x) for x in running)), skipped_updates=skipped)
                running, grad_norms = [], []
            if not args.fit_check and step % args.val_every == 0 and val_ds is not None:
                v, vbins = evaluate()
                val_hist.append(v)
                excursion = len(val_hist) > 3 and v > 1.15 * float(np.median(val_hist[-4:-1]))
                log(event="val", step=step, val_loss=v, val_loss_by_t_quartile=vbins, excursion=bool(excursion))
                if is_main and (args.remote_results or args.local_snapshots) and step % args.snapshot_every == 0:
                    # compact bf16 weights at every validation so an excursion can be located afterwards; never pruned
                    save_checkpoint({"model": {k: t.detach().cpu().to(torch.bfloat16) for k, t in raw.state_dict().items()},
                                     "ema": {k: t.to(torch.bfloat16) for k, t in zip(ema_keys(raw), ema)} if ema is not None else None,
                                     "step": step, "val_loss": v, "args": vars(args), "episodes": pin, CORPUS_KEY: corpus},
                                    os.path.join(args.results_dir, f"snap_{step:07d}.pt"), args.remote_results, keep_local=args.local_snapshots)
                if is_main and v < best_val:
                    best_val = v
                    save_checkpoint({"model": {k: t.detach().cpu().to(torch.bfloat16) for k, t in raw.state_dict().items()},
                                     "step": step, "val_loss": v, "args": vars(args), "episodes": pin, CORPUS_KEY: corpus},
                                    os.path.join(args.results_dir, "best.pt"), args.remote_results, keep_local=True)
            if not args.fit_check and is_main and step % args.ckpt_every == 0:
                # prune before writing so the disk never holds keep_last + 1 rolling checkpoints
                olds = sorted(p for p in os.listdir(args.results_dir) if p.endswith(".pt") and p[0].isdigit())
                for p in olds[:-max(0, args.keep_last - 1)] if args.keep_last > 0 else olds:
                    os.remove(os.path.join(args.results_dir, p))
                # recovery checkpoint: fp32 master weights and EMA, optimizer, scheduler, and the RNG state as a
                # RECORD. `rng` is written and never read back: resume reseeds from (seed, rank, step) instead, and
                # restoring these tensors would not restore the sampler's position in the epoch anyway, so a resume
                # is a documented reshuffle rather than a state-exact continuation (see --resume)
                ck = {"model": {k: t.detach().cpu().float() for k, t in raw.state_dict().items()},
                      "step": step, "args": vars(args), "episodes": pin, CORPUS_KEY: corpus,
                      "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "best_val": best_val, "micro": micro, "skipped": skipped,
                      "rng": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                              "numpy": np.random.get_state()}}
                if ema is not None:
                    ck["ema"] = {k: t.clone() for k, t in zip(ema_keys(raw), ema)}
                # rolling checkpoints live on the remote copy of record; the local copy is kept only when --keep-last > 0
                written = save_checkpoint(ck, os.path.join(args.results_dir, f"{step:07d}.pt"), args.remote_results, keep_local=args.keep_last > 0)
                if args.remote_results and is_main:
                    try:
                        if any(w.startswith(args.remote_results.split(":", 1)[1]) for w in written):
                            prune_remote(args.remote_results, os.path.basename(args.results_dir.rstrip("/")), args.keep_remote)
                        subprocess.run(["rsync", "-a", "-e", " ".join(REMOTE_SSH), "--include=*.json", "--include=*.jsonl", "--exclude=*",
                                        args.results_dir + "/", args.remote_results.rstrip("/") + "/" + os.path.basename(args.results_dir.rstrip("/")) + "/"],
                                       capture_output=True, timeout=600)
                    except (subprocess.TimeoutExpired, OSError) as e:
                        print(f"remote housekeeping failed: {e}", flush=True)
            if not args.fit_check and step % args.ckpt_every == 0:
                acc.wait_for_everyone()   # other ranks wait here instead of inside a collective while rank 0 serializes and uploads
            if step >= max_steps:
                done = True; break
    if args.fit_check:
        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
        # allocated is what the tensors need; reserved is what the caching allocator holds from the
        # card, so reserved is the number a "does this configuration fit" decision has to use
        reserved = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0
        log(event="fit_check", backbone=args.backbone, context_frames=args.context_frames, per_gpu_batch=args.per_gpu_batch,
            world=world, steps=step, steps_per_s=step / dt, peak_mem_gb=round(mem, 2), params=n_params, optim=args.optim,
            latent_channels=latent_channels, grad_ckpt=bool(args.grad_ckpt), accum=accum,
            global_batch=args.per_gpu_batch * world * accum, peak_reserved_gb=round(reserved, 2))
    acc.wait_for_everyone()
    log(event="end", step=step)


def build_parser():
    """Every trainer flag in one place, so a test can construct args without a subprocess."""
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=list(BACKBONES), required=True)
    p.add_argument("--latent-channels", type=int, default=0,
                   help="latent channels of the corpus and the backbone; 0 takes the warm start's own (4 for the SD KL-f8 rows, 16 for sd35)")
    p.add_argument("--context-frames", type=int, default=32)
    p.add_argument("--num-actions", type=int, default=29)
    p.add_argument("--noise-buckets", type=int, default=10)
    p.add_argument("--noise-aug-max", type=float, default=0.7)
    p.add_argument("--objective", choices=list(OBJECTIVES), default="v",
                   help="prediction target: velocity (every finished row) or epsilon, same betas and same sampler; "
                        "recorded in every checkpoint so eval_tf.py and rollout_eval.py sample in the right one")
    p.add_argument("--train-fraction", type=float, default=1.0,
                   help="fraction of the split's TRAINING episodes to use, whole episodes, seeded by --seed and "
                        "nested across fractions; validation and evaluation are untouched (1.0 = every train episode)")
    p.add_argument("--tic-stride", type=int, choices=[1, 4], default=4,
                   help="game time between the frames the model predicts, in ViZDoom tics. 4 (default) is "
                        "one frame per agent decision, the spacing every finished row trained at; 1 selects "
                        "the per-tic dataset, GameNGen's spacing. Sample shapes are identical either way, so "
                        "the backbones and the whole recipe are unchanged")
    p.add_argument("--action-history", type=int, default=0,
                   help="GameNGen's action conditioning: one token per context tic carrying the EXECUTED button "
                        "vector of that tic (the `buttons` column), oldest first, the newest being the control "
                        "applied into the target. Must equal --context-frames. 0 (default) keeps the single "
                        "action-id token every finished row trained with, bit-identically")
    p.add_argument("--control-bits", type=int, default=0,
                   help="width of the executed button vector; 0 reads it from the corpus's own metadata")
    p.add_argument("--phase-conditioning", action="store_true",
                   help="also condition on tics_since_decision, the target tic's position inside the held-action "
                        "run, through the same small-embedding mechanism the noise bucket uses; needs --tic-stride 1")
    p.add_argument("--phase-buckets", type=int, default=PHASE_BUCKETS,
                   help="size of the tics_since_decision table: 4 grid positions plus one bucket for tics with no "
                        "verified decision row within a control interval")
    p.add_argument("--max-episodes", type=int, default=0,
                   help="use only the first N episodes present in --latents-dir, so a run can start on the prefix "
                        "of a corpus that is still being encoded (0 = every encoded episode)")
    p.add_argument("--episode-ids", default="",
                   help="explicit TRAINING episode ids as A:B (half-open, like a Python slice) or a comma list, "
                        "instead of --split's train list; the dense corpus is numbered, so a range is the split")
    p.add_argument("--val-episode-ids", default="",
                   help="explicit VALIDATION episode ids; required with --episode-ids and checked disjoint from it")
    p.add_argument("--val-latents-dir", default="",
                   help="latent directory for validation when it is a separate corpus (the dense val corpus is); "
                        "empty means validate out of --latents-dir")
    p.add_argument("--dense-segment", default="",
                   help="check --episode-ids against this segment's held-out ranges in release/dense_split.json "
                        "(arenas | arenas_678); empty skips the check")
    p.add_argument("--latents-dir", default="data/latents_arnold")
    p.add_argument("--split", default="data/split_arnold.json")
    p.add_argument("--results-dir", default="results/fit_check")
    p.add_argument("--warm-start", default=None,
                   help="DiT: path to DiT-XL-2-256x256.pt; U-Net: SD 1.4 repo or path; PixArt: PixArt-alpha repo or path; "
                        "UniDiffuser: thu-ml/unidiffuser-v1 repo or path; sd35: stabilityai/stable-diffusion-3.5-medium repo or path; "
                        "'none' (dit and pixart) builds the same architecture with a random init")
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--global-batch", type=int, default=32)
    p.add_argument("--per-gpu-batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--skip-grad-norm", type=float, default=0.0, help="skip the optimizer step when the pre-clip gradient norm exceeds this (0 = off; recipe deviation)")
    p.add_argument("--skip-grad-after", type=int, default=0, help="the spike guard is inactive before this many updates: gradient norms of 10 to 20 are normal right after the input layer is inflated, so a guard calibrated on a settled run would skip every early update")
    p.add_argument("--steps", type=int, default=90000)
    p.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw")
    p.add_argument("--ema-every", type=int, default=8, help="0 disables the fp32 CPU EMA")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--grad-ckpt", action="store_true", help="recompute activations in the backward pass (about 30%% slower); needed on 16 GB cards, off by default on the A6000s")
    p.add_argument("--no-grad-ckpt", action="store_true", help=argparse.SUPPRESS)   # former default; kept so old launch lines still parse
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4,
                   help="batches each loader worker reads ahead; a per-tic corpus does not fit the page cache, "
                        "so the loader is seek-bound and this matters more than it did at stride 4")
    p.add_argument("--allow-partial", action="store_true",
                   help="start even though some requested episodes are not encoded yet. Off by default: the "
                        "episode set would otherwise depend on when the run happened to start")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=5000)
    p.add_argument("--val-windows", type=int, default=1024)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--keep-last", type=int, default=2, help="rolling checkpoints kept locally; 0 keeps none when --remote-results is set")
    p.add_argument("--keep-remote", type=int, default=2, help="rolling recovery checkpoints kept on --remote-results")
    p.add_argument("--snapshot-every", type=int, default=5000, help="compact bf16 weight snapshot to --remote-results at these validation steps (never pruned)")
    p.add_argument("--local-snapshots", action="store_true", help="keep the compact bf16 weight snapshots (live and EMA) in the results dir, so intermediate budgets can be evaluated later; never pruned")
    p.add_argument("--action-dropout", type=float, default=0.1, help="fraction of actions replaced by the null id during training (0 disables CFG training)")
    p.add_argument("--action-inject", choices=list(ACTION_INJECTIONS), default="token",
                   help="PixArt only: action and bucket as cross-attention caption tokens (every finished row) or "
                        "added into the timestep/adaLN-single path; recorded in every checkpoint so the evaluators rebuild the right graph")
    p.add_argument("--require-verified-transitions", action="store_true", help="refuse latents without chain ids and 4-tic spacing")
    p.add_argument("--remote-results", default=None, help="user@host:/dir that receives every checkpoint and log as the copy of record")
    p.add_argument("--fit-check", type=int, default=0, help="run N synthetic steps, report steps/s and memory, exit")
    p.add_argument("--resume", default="",
                   help="recovery checkpoint to continue from: weights, step, optimizer, scheduler and EMA. "
                        "What is NOT continued: the saved RNG state is stored in the checkpoint but never "
                        "restored, and neither is the sampler's position in the epoch. Every rank reseeds from "
                        "(seed, rank, step) and the DataLoader draws a fresh permutation, so a resume at half "
                        "an epoch re-presents windows already seen that epoch. Presentations are therefore not "
                        "unique targets after a restart; count restarts when reporting exposure. The recipe args, "
                        "the pinned episode ids and the corpus manifest must all match, and only "
                        "--resume-override lr= may differ")
    p.add_argument("--resume-override", action="append", default=[],
                   help="permit one recipe-identity arg to differ across a resume, as key=value; only lr is "
                        "accepted, for a diagnosed loss excursion, and the change is logged as a deviation")
    p.add_argument("--init-from", default="",
                   help="start a NEW run from one of our own checkpoints' weights (best.pt, a recovery "
                        "NNNNNNN.pt or a snap_*.pt): weights and EMA only, fresh optimizer, step 0, warmup "
                        "again. Mutually exclusive with --resume, and recorded in config.json")
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
