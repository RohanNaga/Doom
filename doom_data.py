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
LATENT_HW_V2 = LATENT_SHAPE_V2[1:]


def latent_shape_v2(latent_channels=4):
    """Per-frame latent shape of the stride-4 layout. 4 channels is SD KL-f8, 16 is SD 3.5's autoencoder."""
    return (latent_channels,) + LATENT_HW_V2


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


def select_train_episodes(train_ids, fraction, seed=0):
    """Seeded subset of a split's *training* episode list, by whole episode.

    One permutation under `RandomState(seed)`, then the first `round(fraction * N)` ids. Nesting is
    the point: the 1/8 list is a subset of the 1/4 list, which is a subset of the 1/2 list, so a
    data-scaling ladder differs only in how much data it saw, never in which episodes were drawn.
    Whole episodes rather than windows, for the same reason `make_split` splits by episode:
    neighbouring windows share all but one context frame.

    Validation and evaluation are untouched by construction, because this only ever receives the
    train list. `fraction == 1.0` returns the list unchanged, which is the default recipe.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"train fraction must be in (0, 1], got {fraction}")
    ids = sorted(set(int(e) for e in train_ids))
    if fraction == 1.0:
        return ids
    n = max(1, int(round(fraction * len(ids))))
    perm = np.random.RandomState(seed).permutation(len(ids))
    return sorted(ids[i] for i in perm[:n])


class LatentWindowDataset(Dataset):
    """L context decision frames -> next decision frame, over encode_parquet.py outputs.

    context: (CL, 32, 40) float32, target: (C, 32, 40) float32, action: int64 recorded at the
    last context frame (the action applied from it to the target). C is `latent_channels`: 4 for
    the SD KL-f8 corpus, 16 for a corpus encoded with SD 3.5's autoencoder. Mixing the two would
    silently reshape into the wrong channel count, so every episode's shape is checked here.
    """

    def __init__(self, latents_dir, episode_ids=None, context_frames=32, require_chains=False, latent_channels=4):
        self.L = context_frames
        self.latent_channels = latent_channels
        want = latent_shape_v2(latent_channels)
        keep = None if episode_ids is None else set(int(e) for e in episode_ids)
        self.episodes, counts = [], []
        for ep, lat_path, meta_path in list_latent_episodes(latents_dir):
            if keep is not None and ep not in keep:
                continue
            lat = np.load(lat_path, mmap_mode="r")
            meta = np.load(meta_path)
            if tuple(lat.shape[1:]) != want:
                raise ValueError(f"{lat_path}: latent shape {tuple(lat.shape[1:])} != {want}")
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
        ctx = torch.from_numpy(np.asarray(lat[start:start + L], dtype=np.float32)).reshape(-1, *LATENT_HW_V2)
        tgt = torch.from_numpy(np.asarray(lat[start + L], dtype=np.float32))
        return ctx, tgt, torch.tensor(int(act[start + L - 1]), dtype=torch.long)


# ---------------------------------------------------------------------------------------
# Per-tic latent episodes written by encode_parquet.py --every-tic
# ---------------------------------------------------------------------------------------

# `tics_since_decision` lives in {0, 1, 2, 3} on the recorded 4-tic action-repeat grid, plus one
# bucket meaning "no verified decision row within a control interval before this row". That extra
# bucket exists because `is_decision` marks only the rows whose whole 4-tic run was clean and
# canonical (see `transitions.valid_transitions`): an interrupted decision, an anti-stuck override,
# or the rows before an episode's first accepted decision leave a gap in the grid. Reserving a
# bucket for them keeps the window set identical whether phase conditioning is on or off, instead
# of quietly folding an off-grid tic into phase 3.
PHASE_BUCKETS = 5
PHASE_OFF_GRID = PHASE_BUCKETS - 1


def parse_episode_ids(spec):
    """"A:B" -> [A, A+1, ..., B-1]; a comma list -> those ids.

    Half-open like a Python slice, so `0:2000` is the first 2,000 episode ids and `7900:8000` is
    the last hundred of an 8,000-episode segment, with no off-by-one to argue about at the launcher.
    """
    s = str(spec).strip()
    if not s:
        return []
    if ":" in s:
        a, b = s.split(":", 1)
        lo, hi = int(a), int(b)
        if hi <= lo:
            raise ValueError(f"episode range {spec!r} is empty: B must be greater than A")
        return list(range(lo, hi))
    return sorted({int(x) for x in s.split(",") if x.strip()})


def assert_disjoint(train_ids, eval_ids, what="train and eval episode ids"):
    """Refuse an episode selection that would train and evaluate on the same episode.

    The dense corpus is one directory of consecutively numbered episodes, so a held-out set is a
    range rather than a seeded draw, and a mistyped range is the one mistake that silently produces
    a training-data number. This is the check that makes that impossible rather than unlikely.
    """
    overlap = sorted(set(int(e) for e in train_ids) & set(int(e) for e in eval_ids))
    if overlap:
        raise ValueError(f"{what} overlap on {len(overlap)} episode(s): {overlap[:8]}"
                         + (" ..." if len(overlap) > 8 else ""))
    return True


def limit_to_encoded(latents_dir, episode_ids, max_episodes=0):
    """`episode_ids` restricted to the first `max_episodes` episodes present in `latents_dir`.

    The dense corpus is encoded progressively, so a run has to be able to start on the prefix that
    exists. Truncating the *directory listing* rather than the requested list means the train and
    validation lists shrink against the same prefix, so validation never drifts onto episodes the
    training list was cut off before.
    """
    ids = sorted(ep for ep, _, _ in list_latent_episodes(latents_dir))
    if max_episodes:
        ids = ids[:int(max_episodes)]
    keep = set(ids)
    return sorted(int(e) for e in episode_ids if int(e) in keep)


DENSE_SPLIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "release", "dense_split.json")


def dense_episode_map(maps, episode_id):
    """Which map a dense-corpus episode was recorded on, from its id alone.

    `record_arnold.py` hands worker w the episodes `w, w + W, ...` and picks
    `maps[e % len(maps)]` for each (record_arnold.py:322-323), so the map is a pure function of the
    episode id and independent of the worker count. A contiguous id range whose length is a multiple
    of `len(maps)` is therefore exactly map-balanced, which is what makes "ids 6000 to 6999" a
    legitimate held-out set rather than a set that happens to over-represent one arena.
    """
    maps = list(maps)
    return int(maps[int(episode_id) % len(maps)])


def dense_map_counts(maps, episode_ids):
    """{map_id: number of episodes} for an id list, by the recorder's own rule."""
    out = {}
    for e in episode_ids:
        m = dense_episode_map(maps, e)
        out[m] = out.get(m, 0) + 1
    return out


def load_dense_split(path=DENSE_SPLIT_PATH):
    """The fixed id ranges of the dense corpus (`release/dense_split.json`)."""
    with open(path) as f:
        return json.load(f)


def dense_ids(split, segment, name):
    """The episode ids of one named range of one dense segment."""
    seg = split["segments"][segment]
    return parse_episode_ids(seg["ranges"][name])


def check_dense_training_ids(split, segment, train_ids, forbid=("val", "test")):
    """Refuse a training id selection that reaches into a held-out range of the same segment.

    This is the one guard between a mistyped `TRAIN_IDS` and a headline number measured on training
    data, which is exactly the mistake the April checkpoint's 26.04 dB made.
    """
    seg = split["segments"][segment]
    for name in forbid:
        if name in seg["ranges"]:
            assert_disjoint(train_ids, dense_ids(split, segment, name),
                            f"training ids and the {segment} {name} range ({seg['ranges'][name]})")
    return True


def tics_since_decision(is_decision, buckets=PHASE_BUCKETS):
    """Per-row position inside the held-action run, as a bucket id in [0, buckets).

    Row t gets `t - d(t)` where `d(t)` is the most recent row at or before t with `is_decision`
    true; rows with no such predecessor, or more than `buckets - 1` tics past one, get the last
    bucket (`PHASE_OFF_GRID`). On a clean recording this is exactly 0 on a decision tic and 1, 2, 3
    on the three tics the engine holds the buttons for.
    """
    flag = np.asarray(is_decision).astype(bool)
    last = -1
    out = np.empty(len(flag), dtype=np.int64)
    for t in range(len(flag)):
        if flag[t]:
            last = t
        d = buckets - 1 if last < 0 else t - last
        out[t] = min(d, buckets - 1)
    return out


def tic_window_starts(meta, context_frames, horizon=1):
    """Start rows whose `context_frames + horizon` consecutive rows form one legal per-tic window.

    A window is valid when, over every adjacent pair of rows it spans:

      R1. the rows come from one episode file (true by construction: one file per episode);
      R2. the recorded tics are consecutive, `tic[i+1] - tic[i] == 1`, so the target really is one
          tic (28.6 ms) after the last context row and no unrendered tic hides inside the window;
      R3. `deaths` does not change, which is the life/respawn signal `transitions.life_segments`
          uses (`record_arnold.py` increments `deaths` on the respawn row), so a window never
          straddles a death and the pixels never teleport to a spawn point;
      R4. `map_id` does not change, so a future recording that walks maps inside one file cannot
          put a map load in the middle of a window.

    `chain_id` is deliberately *not* used: it is -1 on every row that is not a verified decision
    source, so it would reject three quarters of a per-tic corpus. The rules above are the same
    physical constraints the verified-transition filter enforces, applied at tic resolution.
    """
    T = len(meta["tic"])
    span = int(context_frames) + int(horizon)   # rows one window covers
    need = span - 1                             # adjacent pairs inside it, all of which must be good
    if T < span:
        return np.zeros(0, dtype=np.int64)
    good = np.diff(np.asarray(meta["tic"]).astype(np.int64)) == 1
    if "deaths" in meta.files:
        good &= np.diff(np.asarray(meta["deaths"]).astype(np.int64)) == 0
    if "map_id" in meta.files:
        good &= np.diff(np.asarray(meta["map_id"]).astype(np.int64)) == 0
    run = np.concatenate([[0], np.cumsum(good.astype(np.int64))])
    ok = (run[need:T] - run[0:T - need]) == need
    return np.flatnonzero(ok).astype(np.int64)


class TicWindowDataset(Dataset):
    """L consecutive tics -> the next tic, over `encode_parquet.py --every-tic` outputs.

    Sample contract, the same shapes `LatentWindowDataset` yields so every backbone is unmodified:

        context (C*L, 32, 40) float32   tics start .. start+L-1, channel-stacked, oldest first
        target  (C, 32, 40)   float32   tic start+L, one tic later
        action  int64                   the action stored on row start+L-1
        phase   int64                   `tics_since_decision` of the target row (only if with_phase)

    **Which action conditions which target.** `record_arnold.py` stores the action id and button
    vector of row t as the ones applied from row t to row t+1 (`transitions` module docstring), so
    the action that carries the last context tic into the target tic is the one on row
    `start + L - 1`. Under the recorded 4-tic action repeat that row's value *is* the held action:
    the id the agent chose on the decision tic whose run covers this transition, written again on
    every tic of the run. It is the same index `LatentWindowDataset` reads, so a next-tic model and
    a next-decision model are conditioned by identical rule and differ only in spacing.

    **Window validity** is `tic_window_starts`: tic continuity, one life, one map. `horizon > 1`
    demands that many contiguous target rows, which is what equal-game-time evaluation needs;
    `__getitem__` still returns only the first target, so `horizon=1` is the training contract and
    `with_horizon=True` is what the evaluator uses to read all of them.
    """

    def __init__(self, latents_dir, episode_ids=None, context_frames=32, latent_channels=4,
                 horizon=1, with_phase=False, with_horizon=False, phase_buckets=PHASE_BUCKETS):
        self.L = int(context_frames)
        self.horizon = int(horizon)
        self.latent_channels = latent_channels
        self.with_phase = bool(with_phase)
        self.with_horizon = bool(with_horizon)
        self.phase_buckets = int(phase_buckets)
        want = latent_shape_v2(latent_channels)
        keep = None if episode_ids is None else set(int(e) for e in episode_ids)
        self.episodes, counts = [], []
        for ep, lat_path, meta_path in list_latent_episodes(latents_dir):
            if keep is not None and ep not in keep:
                continue
            lat = np.load(lat_path, mmap_mode="r")
            meta = np.load(meta_path)
            if tuple(lat.shape[1:]) != want:
                raise ValueError(f"{lat_path}: latent shape {tuple(lat.shape[1:])} != {want}")
            if "is_decision" not in meta.files:
                raise ValueError(f"{meta_path}: no is_decision column; this is a stride-4 corpus, "
                                 "not one written by encode_parquet.py --every-tic")
            if len(meta["tic"]) != lat.shape[0]:
                raise ValueError(f"{lat_path}: {lat.shape[0]} latents vs {len(meta['tic'])} metadata rows")
            starts = tic_window_starts(meta, self.L, self.horizon)
            if len(starts) == 0:
                continue
            phase = tics_since_decision(meta["is_decision"], self.phase_buckets)
            self.episodes.append((ep, lat, meta["action"].astype(np.int64), int(meta["map_id"][0]),
                                  starts, np.asarray(meta["tic"]).astype(np.int64), phase))
            counts.append(len(starts))
        if not self.episodes:
            raise ValueError("no usable per-tic episodes")
        self.offsets = np.concatenate([[0], np.cumsum(counts)])

    def __len__(self):
        return int(self.offsets[-1])

    def locate(self, idx):
        """Global index -> (episode slot, start row of the window inside that episode's arrays)."""
        slot = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        return slot, int(self.episodes[slot][4][idx - self.offsets[slot]])

    def target_tic(self, idx):
        """Recorded tic of the first target row."""
        slot, start = self.locate(idx)
        return int(self.episodes[slot][5][start + self.L])

    def target_phase(self, idx):
        """`tics_since_decision` of the first target row."""
        slot, start = self.locate(idx)
        return int(self.episodes[slot][6][start + self.L])

    def held_action_row(self, start):
        """The row whose action conditions the target at `start + L`: the last context row."""
        return start + self.L - 1

    def __getitem__(self, idx):
        slot, start = self.locate(idx)
        _, lat, act, _, _, _, phase = self.episodes[slot]
        L, H = self.L, self.horizon
        ctx = torch.from_numpy(np.asarray(lat[start:start + L], dtype=np.float32)).reshape(-1, *LATENT_HW_V2)
        if self.with_horizon:
            tgt = torch.from_numpy(np.asarray(lat[start + L:start + L + H], dtype=np.float32))
            a = torch.from_numpy(act[self.held_action_row(start):self.held_action_row(start) + H].copy())
            ph = torch.from_numpy(phase[start + L:start + L + H].copy())
            return ctx, tgt, a, ph
        tgt = torch.from_numpy(np.asarray(lat[start + L], dtype=np.float32))
        a = torch.tensor(int(act[self.held_action_row(start)]), dtype=torch.long)
        if self.with_phase:
            return ctx, tgt, a, torch.tensor(int(phase[start + L]), dtype=torch.long)
        return ctx, tgt, a
