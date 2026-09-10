"""
Episode-level data access for DoomDiT.

Replaces the consolidated (N,4,4,15,20) `context_latents.npy` produced by
`build_dataset.py`, which stored every latent four times (once per context slot,
23.7 GB) and threw away episode boundaries. This module indexes the per-episode
`ep_XXXX_latents.npy` / `ep_XXXX_actions.npy` files directly (5.9 GB total) and
keeps episode identity, which is what a held-out split needs.

Sample contract is identical to `trainDoom.CustomDataset`:
    context: (16, 16, 20) float32  four past latents, channel-stacked, H padded 15->16
    target:  (4, 16, 20)  float32  latent of the frame after the context, H padded
    action:  int64                 conditioning action (see ACTION_OFFSET)

Episode files are the ones `build_dataset.py` consumes:
    ep_XXXX_latents.npy  (T, 4, 15, 20) float16, SD-VAE-ft-mse latents * 0.18215
    ep_XXXX_actions.npy  (T,) int, VizDoom action id in [0, 17]
"""
import glob
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CONTEXT_FRAMES = 4
LATENT_SHAPE = (4, 15, 20)
NUM_ACTIONS = 18

# Which stored action conditions the prediction of frame i+4 from frames i..i+3.
# `build_dataset.py` stored actions[i : i+5] and training used the last column,
# i.e. the action recorded on the *target* frame's row. Offset 4 reproduces that
# exactly. Offset 3 would be the action recorded on the last context frame. Which
# one is causally right depends on whether the dataset logs the action taken
# *after* observing a frame or the action that *produced* it; that is still an
# open question (see RESEARCH_CONTEXT.md). Keep 4 until it is resolved so the
# released checkpoint evaluates under the convention it was trained with.
ACTION_OFFSET = 4

_EP_RE = re.compile(r"ep_(\d+)_latents\.npy$")


def list_episodes(episodes_dir):
    """Return sorted [(episode_id:int, latents_path, actions_path)] for a directory.

    Sorting by the zero-padded filename matches the concatenation order used by
    `build_dataset.py`, so global sample indices from the old consolidated arrays
    (and the training-time eval segment starts) can be mapped back to episodes.
    """
    out = []
    for lat in sorted(glob.glob(os.path.join(episodes_dir, "ep_*_latents.npy"))):
        m = _EP_RE.search(lat)
        ep_id = int(m.group(1))
        act = os.path.join(episodes_dir, f"ep_{m.group(1)}_actions.npy")
        if not os.path.isfile(act):
            raise FileNotFoundError(f"missing actions file for {lat}: {act}")
        out.append((ep_id, lat, act))
    if not out:
        raise FileNotFoundError(f"no ep_*_latents.npy under {episodes_dir}")
    return out


def make_split(episode_ids, holdout_frac=0.1, seed=0):
    """Episode-level train/val split.

    Held-out episodes are drawn uniformly at random with a fixed seed, so the
    split is reproducible from (episode_ids, holdout_frac, seed) alone. Splitting
    by episode (not by window) is what makes the split honest: neighbouring
    windows inside one episode share three of four context frames, so a
    window-level split would leak almost every held-out target into training.
    """
    ids = sorted(int(e) for e in episode_ids)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ids))
    n_val = max(1, int(round(holdout_frac * len(ids))))
    val = sorted(ids[i] for i in perm[:n_val])
    train = sorted(set(ids) - set(val))
    return {"train": train, "val": val,
            "meta": {"holdout_frac": holdout_frac, "seed": seed, "num_episodes": len(ids)}}


def save_split(split, path):
    with open(path, "w") as f:
        json.dump(split, f, indent=1)


def load_split(path):
    with open(path) as f:
        return json.load(f)


class EpisodeWindowDataset(Dataset):
    """Sliding-window (4 context -> 1 target) dataset over per-episode latent files.

    Each episode's latents are memory-mapped once and shared across DataLoader
    workers through the page cache, the same trick `CustomDataset` relied on.
    """

    def __init__(self, episodes_dir, episode_ids=None, context_frames=CONTEXT_FRAMES,
                 action_offset=ACTION_OFFSET):
        self.context_frames = context_frames
        self.action_offset = action_offset
        keep = None if episode_ids is None else set(int(e) for e in episode_ids)
        self.episodes = []          # (ep_id, latents_mmap, actions_array)
        self.window_counts = []
        for ep_id, lat_path, act_path in list_episodes(episodes_dir):
            if keep is not None and ep_id not in keep:
                continue
            lat = np.load(lat_path, mmap_mode="r")
            act = np.load(act_path)
            if lat.shape[1:] != LATENT_SHAPE:
                raise ValueError(f"{lat_path}: latent shape {lat.shape[1:]} != {LATENT_SHAPE}")
            if act.shape[0] != lat.shape[0]:
                raise ValueError(f"{lat_path}: {lat.shape[0]} latents vs {act.shape[0]} actions")
            n_windows = lat.shape[0] - context_frames
            if n_windows <= 0:
                continue
            self.episodes.append((ep_id, lat, act.astype(np.int64)))
            self.window_counts.append(n_windows)
        if not self.episodes:
            raise ValueError("no usable episodes after filtering")
        # cumulative offsets: global window index -> (episode slot, local start frame)
        self.offsets = np.concatenate([[0], np.cumsum(self.window_counts)])

    def __len__(self):
        return int(self.offsets[-1])

    def locate(self, idx):
        """Global window index -> (episode_id, local start frame index)."""
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        slot = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        return self.episodes[slot][0], int(idx - self.offsets[slot])

    def raw_window(self, idx):
        """Return (context (4,4,15,20) f32, target (4,15,20) f32, action int, episode_id, start)."""
        slot = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        ep_id, lat, act = self.episodes[slot]
        start = int(idx - self.offsets[slot])
        k = self.context_frames
        context = np.asarray(lat[start:start + k], dtype=np.float32)
        target = np.asarray(lat[start + k], dtype=np.float32)
        action = int(act[start + self.action_offset])
        return context, target, action, ep_id, start

    def __getitem__(self, idx):
        context, target, action, _, _ = self.raw_window(idx)
        context = F.pad(torch.from_numpy(context), (0, 0, 0, 1))   # (4, 4, 16, 20)
        target = F.pad(torch.from_numpy(target), (0, 0, 0, 1))     # (4, 16, 20)
        return context.reshape(-1, 16, 20), target, torch.tensor(action, dtype=torch.long)


def legacy_segment_starts(total_windows, num_segments=10, segment_size=8):
    """The fixed eval segments trainDoom.py and eval_checkpoint.py used.

    Over the full 500-episode set (2,468,229 windows) this yields starts
    0, 274246, ..., 2468221, matching results/002-DiT-XL-2/samples/ground_truth/.
    These segments come from the *training* data; use them only to reproduce the
    original headline numbers, never as a held-out evaluation.
    """
    return np.linspace(0, max(total_windows - segment_size, 0), num_segments, dtype=int)


def sample_eval_windows(dataset, num_windows, seed=0, stride=1):
    """Deterministic evaluation window indices spread over every episode in `dataset`.

    Draws windows uniformly over the concatenated index space so long episodes
    contribute proportionally, then thins by `stride` so consecutive windows do
    not dominate the estimate. Returns sorted unique global indices.
    """
    rng = np.random.RandomState(seed)
    n = len(dataset)
    idx = rng.choice(n, size=min(num_windows * stride, n), replace=False)
    return np.sort(idx[::stride])[:num_windows]


# ---------------------------------------------------------------------------------------
# Stride-4 latent episodes written by encode_parquet.py (Arnold / Stiegler recordings)
# ---------------------------------------------------------------------------------------

LATENT_SHAPE_V2 = (4, 32, 40)


def list_latent_episodes(latents_dir):
    """Sorted [(episode_id:int, latents_path, meta_path)] for encode_parquet.py outputs."""
    out = []
    for lat in sorted(glob.glob(os.path.join(latents_dir, "ep_*_latents.npy"))):
        ep = int(os.path.basename(lat).split("_")[1])
        meta = lat.replace("_latents.npy", "_meta.npz")
        if os.path.isfile(meta):
            out.append((ep, lat, meta))
    if not out:
        raise FileNotFoundError(f"no ep_*_latents.npy under {latents_dir}")
    return out


def make_split_by_map(latents_dir, holdout_maps=(16, 17), holdout_frac=0.1, seed=0):
    """Episode-level split that also holds out whole maps for the unseen-map row."""
    eps = list_latent_episodes(latents_dir)
    by_map = {}
    for ep, _, meta in eps:
        m = int(np.load(meta)["map_id"][0])
        by_map.setdefault(m, []).append(ep)
    rng = np.random.RandomState(seed)
    train, val, unseen = [], [], []
    for m, ids in sorted(by_map.items()):
        ids = sorted(ids)
        if m in holdout_maps:
            unseen += ids; continue
        perm = rng.permutation(len(ids)); n_val = max(1, int(round(holdout_frac * len(ids))))
        val += [ids[i] for i in perm[:n_val]]; train += [ids[i] for i in perm[n_val:]]
    return {"train": sorted(train), "val": sorted(val), "unseen_map": sorted(unseen),
            "meta": {"holdout_maps": list(holdout_maps), "holdout_frac": holdout_frac, "seed": seed,
                     "episodes_per_map": {str(m): len(v) for m, v in sorted(by_map.items())}}}


class LatentWindowDataset(Dataset):
    """L context decision frames -> next decision frame, over encode_parquet.py outputs.

    context: (4L, 32, 40) float32, target: (4, 32, 40) float32, action: int64 recorded at the
    last context frame (the action applied from it to the target).
    """

    def __init__(self, latents_dir, episode_ids=None, context_frames=32, require_chains=False):
        self.L = context_frames
        keep = None if episode_ids is None else set(int(e) for e in episode_ids)
        self.episodes, counts = [], []
        for ep, lat_path, meta_path in list_latent_episodes(latents_dir):
            if keep is not None and ep not in keep:
                continue
            lat = np.load(lat_path, mmap_mode="r")
            meta = np.load(meta_path)
            T = lat.shape[0]
            if require_chains:
                # verified-transition contract: chain ids and real tics present, consecutive frames in a chain 4 tics apart
                assert "chain_id" in meta.files and "tic" in meta.files, f"{ep}: no chain_id/tic; encode with --align-decisions"
                assert len(meta["chain_id"]) == T == len(meta["action"]) == len(meta["tic"]), f"{ep}: metadata length mismatch"
                same = meta["chain_id"][1:] == meta["chain_id"][:-1]
                assert np.all(np.diff(meta["tic"])[same] == 4), f"{ep}: tic spacing inside a chain is not 4"
            if "chain_id" in meta.files:
                # a window of L+1 frames must lie inside one chain of verified transitions
                cid = meta["chain_id"]
                ok = np.array([cid[s] == cid[s + context_frames] for s in range(T - context_frames)], dtype=bool) if T > context_frames else np.zeros(0, bool)
                starts = np.flatnonzero(ok)
            else:
                starts = np.arange(max(0, T - context_frames))
            if len(starts) == 0:
                continue
            tics = meta["tic"] if "tic" in meta.files else None
            self.episodes.append((ep, lat, meta["action"].astype(np.int64), int(meta["map_id"][0]), starts, tics))
            counts.append(len(starts))
        if not self.episodes:
            raise ValueError("no usable episodes")
        self.offsets = np.concatenate([[0], np.cumsum(counts)])

    def __len__(self):
        return int(self.offsets[-1])

    def locate(self, idx):
        """Global index -> (episode slot, start frame index within the episode's latent array)."""
        slot = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        return slot, int(self.episodes[slot][4][idx - self.offsets[slot]])

    def target_tic(self, idx):
        """Recorded tic of the target frame, from the encoder metadata (None for old layouts)."""
        slot, start = self.locate(idx)
        tics = self.episodes[slot][5]
        return None if tics is None else int(tics[start + self.L])

    def __getitem__(self, idx):
        slot, start = self.locate(idx)
        ep, lat, act = self.episodes[slot][:3]
        L = self.L
        ctx = torch.from_numpy(np.asarray(lat[start:start + L], dtype=np.float32)).reshape(-1, 32, 40)
        tgt = torch.from_numpy(np.asarray(lat[start + L], dtype=np.float32))
        return ctx, tgt, torch.tensor(int(act[start + L - 1]), dtype=torch.long)
