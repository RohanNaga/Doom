"""
Windowed access to the Wan-VAE latent corpus written by `encode_wan.py` (scheme `wan-chain-v2`,
or the older `wan-chain-v1`; both load, and `WanWindowDataset.scheme` reports which).

One latent frame is one decision: latent frame j of a chain is the 4-tic execution of decision
j-1, and latent frame 0 is the lone leading frame the causal Wan encoder needs (see the
`encode_wan` module docstring). Under v2 those four tics are exactly a_{j-1}'s effect span and
the frame's last decoded tic is the next decision tic; under v1 the chunk starts on the
decision tic instead. A sample is therefore

    context  float32 [L, 16, 32, 40]   L consecutive latent frames, oldest first
    target   float32 [16, 32, 40]      the next latent frame
    action   int64   scalar            the action of the decision the target frame executes
    ctx_act  int64   [L]               the actions of the context frames

Unlike `doom_data.EpisodeWindowDataset`, the context keeps a frame axis instead of being
stacked on channels: a video transformer consumes (B, 16, F, 32, 40), so the caller
concatenates context and target along the frame axis rather than unstacking channels.

Two invariants that make the windows honest:

* **A window never crosses a chain.** A chain from `transitions.valid_transitions` is a run of
  decisions spaced exactly 4 tics apart inside one continuous life, so it contains no death,
  no respawn and no interrupted decision. Windows are enumerated per chain.
* **The lone first frame is never a target, and by default never a context frame either.**
  Latent frame 0 of a chain is the only latent built from a single raw frame; every other
  latent compresses four. It is scaffolding demanded by the Wan encoder's `1 + 4k` layout
  (`autoencoder_kl_wan.py:1143`, diffusers 0.40.0), not a decision, and its latent statistics come from a
  different code path (empty feature cache, temporal stride 1 instead of 4). Mixing it into a
  context stack would put a distribution the model never sees at rollout time into training
  contexts, so `allow_lone_context` defaults to False. Set it True only if you also seed
  rollouts from a freshly encoded prefix that includes frame 0; the action stored for that
  frame is -1.

Latents on disk are **raw** encoder posterior means. `normalize=True` (the default) applies the
per-channel `(z - latents_mean) / latents_std` from the checkpoint's `vae/config.json`, which
is the normalisation the Wan pipelines apply before the transformer
(`pipelines/wan/pipeline_wan_i2v.py:440-459`); training in that space is what makes a Wan warm
start meaningful. Set `normalize=False` to train in raw encoder space.
"""
import glob
import json
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset

LATENT_SHAPE = (16, 32, 40)
SCHEMES = ("wan-chain-v2", "wan-chain-v1")
SCHEME = SCHEMES[0]
CONTEXT_FRAMES = 8
LONE_ACTION = -1

_EP_RE = re.compile(r"(.+)\.npy$")


def list_episodes(latents_dir):
    """Sorted [(episode_name, latents_path, meta_path)] for a `encode_wan.py` output directory."""
    out = []
    for lat in sorted(glob.glob(os.path.join(latents_dir, "*.npy"))):
        name = _EP_RE.search(os.path.basename(lat)).group(1)
        meta = os.path.join(latents_dir, f"{name}.meta.json")
        if not os.path.isfile(meta):
            raise FileNotFoundError(f"missing metadata for {lat}: {meta}")
        out.append((name, lat, meta))
    if not out:
        raise FileNotFoundError(f"no *.npy under {latents_dir}")
    return out


def episode_name(e):
    """Canonical episode name for an id from a split file: 7 -> "ep_00007", "7" -> "ep_00007", "ep_00007" unchanged."""
    s = str(e)
    return f"ep_{int(s):05d}" if s.isdigit() else s


def make_split(episode_names, holdout_frac=0.1, seed=0):
    """Episode-level split. Splitting by window would leak: neighbouring windows in one chain
    share all but one context frame."""
    names = sorted(episode_names)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(names))
    n_val = max(1, int(round(holdout_frac * len(names))))
    val = sorted(names[i] for i in perm[:n_val])
    return {"train": sorted(set(names) - set(val)), "val": val,
            "meta": {"holdout_frac": holdout_frac, "seed": seed, "num_episodes": len(names)}}


def save_split(split, path):
    with open(path, "w") as f:
        json.dump(split, f, indent=1)


def load_split(path):
    with open(path) as f:
        return json.load(f)


def chain_segments(chain_id):
    """[(start, stop)) row ranges of each contiguous chain run in an episode's latent array."""
    c = np.asarray(chain_id)
    if len(c) == 0:
        return []
    cuts = np.flatnonzero(np.diff(c) != 0) + 1
    bounds = np.concatenate([[0], cuts, [len(c)]])
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


class WanWindowDataset(Dataset):
    """Sliding windows over `encode_wan.py` latents, confined to single chains.

    Args:
        latents_dir: directory holding `<episode>.npy` and `<episode>.meta.json`. It may hold
            only a subset of `episode_ids` -- ids with no file are skipped and counted, which is
            what lets a pilot encode part of a split and still train against the whole split file.
        episode_ids: episode names to keep (as in `make_split`); None keeps all.
        context_frames: L, the number of context latent frames.
        normalize: apply the checkpoint's `(z - latents_mean) / latents_std`.
        allow_lone_context: permit latent frame 0 of a chain in the context (never as target).
        verbose: print the one-line constructor summary (episode and window counts, skips).

    After construction, `missing_episodes` holds the requested ids that were not on disk and
    `scheme` holds the layout the corpus was encoded with.
    """

    def __init__(self, latents_dir, episode_ids=None, context_frames=CONTEXT_FRAMES,
                 normalize=True, allow_lone_context=False, verbose=True):
        if context_frames < 1:
            raise ValueError("context_frames must be >= 1")
        self.context_frames = context_frames
        self.normalize = normalize
        self.allow_lone_context = allow_lone_context
        # split files carry integer episode ids (split_arnold.json) while encode_wan.py names files by
        # parquet basename (ep_00000); accept both spellings
        keep = None if episode_ids is None else {episode_name(e) for e in episode_ids}

        self.episodes = []          # (name, mmap latents, action array, is_lone array)
        starts, mean, std = [], None, None
        self.scheme, found = None, set()
        for name, lat_path, meta_path in list_episodes(latents_dir):
            if keep is not None and name not in keep:
                continue
            found.add(name)
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("scheme") not in SCHEMES:
                raise ValueError(f"{meta_path}: scheme {meta.get('scheme')!r} not in {SCHEMES}")
            if self.scheme is None:
                self.scheme = meta["scheme"]
            elif meta["scheme"] != self.scheme:
                raise ValueError(f"{meta_path}: scheme {meta['scheme']!r} mixes with {self.scheme!r} "
                                 f"already seen in this directory")
            lat = np.load(lat_path, mmap_mode="r")
            if lat.shape[1:] != LATENT_SHAPE:
                raise ValueError(f"{lat_path}: latent shape {lat.shape[1:]} != {LATENT_SHAPE}")
            action = np.asarray(meta["action"], dtype=np.int64)
            is_lone = np.asarray(meta["is_lone_first"], dtype=np.int64)
            chain_id = np.asarray(meta["chain_id"], dtype=np.int64)
            if not (len(action) == len(is_lone) == len(chain_id) == lat.shape[0]):
                raise ValueError(f"{meta_path}: metadata length does not match {lat.shape[0]} latent frames")
            if mean is None:
                mean = np.asarray(meta["latents_mean"], dtype=np.float32)
                std = np.asarray(meta["latents_std"], dtype=np.float32)
            elif not (np.allclose(mean, meta["latents_mean"]) and np.allclose(std, meta["latents_std"])):
                raise ValueError(f"{meta_path}: latents_mean/std differ from the rest of the corpus")

            ep_idx = len(self.episodes)
            self.episodes.append((name, lat, action, is_lone))
            # earliest legal target in a chain: context must clear the lone frame unless allowed
            floor = context_frames if allow_lone_context else context_frames + 1
            for a, b in chain_segments(chain_id):
                if is_lone[a] != 1:
                    raise ValueError(f"{meta_path}: chain starting at row {a} does not begin with a lone frame")
                if b > a + floor:
                    t = np.arange(a + floor, b, dtype=np.int64)
                    starts.append(np.stack([np.full_like(t, ep_idx), t], axis=1))

        if not self.episodes:
            raise ValueError(f"no episodes selected from {latents_dir}")
        self.missing_episodes = sorted(keep - found) if keep is not None else []
        self.latents_mean = mean.reshape(LATENT_SHAPE[0], 1, 1)
        self.latents_std = std.reshape(LATENT_SHAPE[0], 1, 1)
        self.index = (np.concatenate(starts) if starts
                      else np.zeros((0, 2), dtype=np.int64))
        if verbose:
            miss = (f", {len(self.missing_episodes)} requested ids not encoded yet"
                    if self.missing_episodes else "")
            print(f"WanWindowDataset({latents_dir}): {len(self.episodes)} episodes, "
                  f"{len(self.index)} windows, L={context_frames}, scheme {self.scheme}, "
                  f"normalize={normalize}, allow_lone_context={allow_lone_context}{miss}",
                  flush=True)

    def __len__(self):
        return len(self.index)

    def _to_float(self, z):
        z = np.asarray(z, dtype=np.float32)
        if self.normalize:
            z = (z - self.latents_mean) / self.latents_std
        return torch.from_numpy(z)

    def __getitem__(self, i):
        ep_idx, t = (int(v) for v in self.index[i])
        _, lat, action, _ = self.episodes[ep_idx]
        L = self.context_frames
        context = self._to_float(lat[t - L:t])
        target = self._to_float(lat[t])
        return (context, target,
                torch.tensor(int(action[t]), dtype=torch.long),
                torch.from_numpy(action[t - L:t].copy()))


# `encode_wan.py` documents the loader by this name; both spellings import the same class.
LatentWindowDataset = WanWindowDataset
