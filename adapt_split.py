"""
Per-map adaptation splits: which episodes of an unseen map adapt the model, which score it, and on which windows.

The adaptation study (`docs/lora_adaptation_design_2026-09-26.html`, decision 1) adapts a model pretrained on
the four training maps to ONE unseen map from a few of its episodes and scores it on that map's other
episodes. This module writes that split, once per (map, seed), before anything trains:

    python adapt_split.py --map-split results/distance_study/splits/split_unseen_map17.json \\
        --latents-dir $E/latents_arnold_eval_pertic/unseen --seed 0 --out-dir results/adapt_splits

The input is the distance study's per-map split file (its `val` list is every episode of the map); the
latent directory is read for the per-tic sidecars only (`tic`, `deaths`, `map_id`), so the same split
serves every latent space of that corpus.

**Episodes.** One seeded permutation of the map's episodes (`make_adapt_split`): the first
`--held-out-episodes` (default 4) are held out, the rest form the ordered ADAPT POOL, whose first
`--adapt-episodes` (default 6) are the adaptation set. Every map gets the same budget, 6 adapt and 4 held
out, whether it has 10 episodes or 20; a 20-episode arena's pool holds 16, and the extra ten are reachable
only through the data grid (`adapt_episodes(split, k)`, the trainer's `--adapt-episodes-k`), which takes
the first k of the pool. Because it is a prefix of one permutation, k = 1, 2, 4, 6 are nested and the
episode curve reuses one split and one held-out set.

**Windows.** A fresh fixed draw of `--windows` (256) one-tic windows from the HELD-OUT episodes only
(`held_out_windows`), seeded by the split seed and recorded in the file as [episode, start row] pairs.
Every checkpoint of every run on this split, step 0 included, is scored on exactly these windows, so the
zero-shot number on them is the step-0 row. The distance study's own draw (256 windows over all of the
map's episodes, `eval_tf.draw_windows` with seed 0) is recorded as well, restricted to the held-out
episodes (`legacy_held_out_windows`, about 100 windows at 4 of 10 episodes): a cross-check against the
frozen zero-shot scores, not the primary outcome. `windows_in_dataset` turns either list back into
dataset indices for `eval_tf.py --windows-file`.

The window rules are `doom_data.tic_window_starts` at horizon 1 (tic continuity, one life, one map) over
the episodes in `doom_data.list_latent_episodes` order, which is exactly how `TicWindowDataset` enumerates
them, so a recorded window is a dataset window by construction; the scorer refuses any that is not.

The file keeps `val` (the held-out episodes) and `train` (the adaptation set) as well, so `eval_tf.py` and
`directional_check.py` read it unchanged with `--subset val`.
"""
import argparse
import hashlib
import json
import os
import subprocess

import numpy as np

DEFAULT_ADAPT = 6
DEFAULT_HELD_OUT = 4
DEFAULT_WINDOWS = 256
CONTEXT_FRAMES = 32
WINDOW_KEY = "held_out_windows"
LEGACY_KEY = "legacy_held_out_windows"
WINDOW_KEYS = (WINDOW_KEY, LEGACY_KEY)
KIND = "adapt_split"
SUBSET = "val"                  # the key eval_tf.py and directional_check.py read (make_dense_eval_splits.SUBSET)
REPO = os.path.dirname(os.path.abspath(__file__))


def make_adapt_split(episodes, seed=0, n_adapt=DEFAULT_ADAPT, n_held_out=DEFAULT_HELD_OUT):
    """{held_out, adapt_pool, adapt} from one seeded permutation of a map's episode ids.

    `held_out` and `adapt` are sorted; `adapt_pool` keeps the permutation order, because its prefixes are
    the data grid. Deterministic in (episode set, seed, n_held_out) and independent of the input order.
    """
    ids = sorted(int(e) for e in episodes)
    if len(set(ids)) != len(ids):
        raise ValueError("the episode list repeats an id")
    n_adapt, n_held_out = int(n_adapt), int(n_held_out)
    if n_adapt < 1 or n_held_out < 1:
        raise ValueError(f"need at least one adaptation and one held-out episode, got {n_adapt} and {n_held_out}")
    if n_adapt + n_held_out > len(ids):
        raise ValueError(f"{n_adapt} adaptation + {n_held_out} held-out episodes exceed the map's {len(ids)}")
    perm = np.random.RandomState(int(seed)).permutation(len(ids))
    held_out = sorted(ids[i] for i in perm[:n_held_out])
    pool = [ids[i] for i in perm[n_held_out:]]
    return {"held_out": held_out, "adapt_pool": pool, "adapt": sorted(pool[:n_adapt])}


def adapt_episodes(split, k=0):
    """The episodes a run adapts on: the split's adaptation set, or the first `k` of its pool (the data grid)."""
    k = int(k or 0)
    if not k:
        return list(split["adapt"])
    pool = list(split["adapt_pool"])
    if k < 0 or k > len(pool):
        raise ValueError(f"--adapt-episodes-k {k}: the adapt pool of this split holds {len(pool)} episodes")
    return sorted(pool[:k])


def window_starts(latents_dir, episodes, context_frames=CONTEXT_FRAMES, horizon=1):
    """[(episode, valid start rows)] in the order `TicWindowDataset` enumerates them; empty episodes dropped."""
    from doom_data import list_latent_episodes, tic_window_starts
    keep = {int(e) for e in episodes}
    out, found = [], set()
    for ep, _, meta_path in list_latent_episodes(latents_dir):
        if ep not in keep:
            continue
        found.add(ep)
        with np.load(meta_path) as meta:
            starts = tic_window_starts(meta, context_frames, horizon)
        if len(starts):
            out.append((int(ep), np.asarray(starts, dtype=np.int64)))
    missing = sorted(keep - found)
    if missing:
        raise ValueError(f"{latents_dir} holds no latents for episode(s) {missing}")
    return out


def draw_windows(n_total, num_windows, seed):
    """`eval_tf.draw_windows`, repeated here so the split writer needs numpy only; a test pins the two equal."""
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_total, size=min(num_windows, n_total), replace=False))


def pairs_at(per_episode, indices):
    """Dataset indices over `per_episode` (the `window_starts` list) -> [[episode, start], ...]."""
    counts = [len(s) for _, s in per_episode]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    out = []
    for i in np.asarray(indices, dtype=np.int64):
        slot = int(np.searchsorted(offsets, i, side="right") - 1)
        ep, starts = per_episode[slot]
        out.append([int(ep), int(starts[i - offsets[slot]])])
    return out


def draw_held_out_windows(latents_dir, held_out, num_windows=DEFAULT_WINDOWS, seed=0, context_frames=CONTEXT_FRAMES):
    """(pairs, windows available): the fresh fixed draw of one-tic windows from the held-out episodes."""
    per = window_starts(latents_dir, held_out, context_frames, 1)
    total = sum(len(s) for _, s in per)
    return pairs_at(per, draw_windows(total, num_windows, seed)), total


def legacy_held_out_windows(latents_dir, map_episodes, held_out, num_windows=DEFAULT_WINDOWS, seed=0,
                            context_frames=CONTEXT_FRAMES):
    """The distance study's draw over the whole map, restricted to the held-out episodes.

    Returns (kept pairs, windows over the map, windows drawn). Scored against the map's own split file,
    the kept windows get the same dataset index, hence the same sampler noise, as in the frozen scores.
    """
    per = window_starts(latents_dir, map_episodes, context_frames, 1)
    total = sum(len(s) for _, s in per)
    drawn = pairs_at(per, draw_windows(total, num_windows, seed))
    keep = {int(e) for e in held_out}
    return [p for p in drawn if p[0] in keep], total, len(drawn)


def windows_in_dataset(ds, windows):
    """Sorted dataset indices of recorded [episode, start] windows; refuses any the dataset does not hold.

    `ds` is a `TicWindowDataset` (or `LatentWindowDataset`): `ds.episodes[slot]` carries the episode id
    first and the valid start rows fifth, and `ds.offsets` the cumulative window counts.
    """
    index = {}
    for slot, ep in enumerate(ds.episodes):
        base = int(ds.offsets[slot])
        for j, s in enumerate(np.asarray(ep[4]).tolist()):
            index[(int(ep[0]), int(s))] = base + j
    out, missing = [], []
    for e, s in windows:
        i = index.get((int(e), int(s)))
        if i is None:
            missing.append((e, s))
        else:
            out.append(i)
    if missing:
        raise ValueError(f"{len(missing)} recorded window(s) are not windows of this dataset, e.g. episode "
                         f"{missing[0][0]} start {missing[0][1]}: the split was written over another corpus, "
                         "another context length or another horizon")
    if len(set(out)) != len(out):
        raise ValueError("the recorded window list repeats a window")
    return np.array(sorted(out), dtype=np.int64)


def load_windows(path, key=WINDOW_KEY):
    """The [episode, start] list stored under `key` of an adaptation split file."""
    if key not in WINDOW_KEYS:
        raise ValueError(f"window key must be one of {WINDOW_KEYS}, got {key!r}")
    with open(path) as f:
        split = json.load(f)
    if split.get("meta", {}).get("kind") != KIND:
        raise ValueError(f"{path} is not an adaptation split file (adapt_split.py)")
    if key not in split:
        raise ValueError(f"{path} records no {key}")
    return split[key]


def sha256_bytes(path):
    """Hex SHA-256 of a small file (a split file), read whole and never cached."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git_state(repo=REPO):
    """{commit, dirty} of the checkout this code runs from; `dirty` counts tracked files with changes."""
    def git(*cmd):
        try:
            r = subprocess.run(["git", "-C", repo, *cmd], capture_output=True, text=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD") or "unversioned",
            "dirty": None if status is None else len([ln for ln in status.splitlines() if ln.strip()])}


def split_filename(set_name, map_id, seed, n_adapt=DEFAULT_ADAPT, n_held_out=DEFAULT_HELD_OUT):
    """`split_adapt_<set>_map<NN>_seed<S>.json`, with `_a<A>h<H>` only when the budget is not the default."""
    tail = "" if (n_adapt, n_held_out) == (DEFAULT_ADAPT, DEFAULT_HELD_OUT) else f"_a{n_adapt}h{n_held_out}"
    return f"split_adapt_{set_name}_map{int(map_id):02d}_seed{int(seed)}{tail}.json"


def build_split(episodes, set_name, map_id, latents_dir, seed=0, n_adapt=DEFAULT_ADAPT, n_held_out=DEFAULT_HELD_OUT,
                num_windows=DEFAULT_WINDOWS, context_frames=CONTEXT_FRAMES, legacy=True, legacy_windows=DEFAULT_WINDOWS,
                legacy_seed=0, source_split=None):
    """The whole split record: episodes, the held-out window draw, the legacy draw and the provenance."""
    ep = make_adapt_split(episodes, seed, n_adapt, n_held_out)
    windows, available = draw_held_out_windows(latents_dir, ep["held_out"], num_windows, seed, context_frames)
    if len(windows) < num_windows:
        print(f"note: the held-out episodes hold only {available} windows, fewer than the {num_windows} asked for")
    meta = {"kind": KIND, "set": set_name, "map": int(map_id), "corpus": f"{set_name}_map{int(map_id):02d}",
            "seed": int(seed), "n_adapt": int(n_adapt), "n_held_out": int(n_held_out),
            "episodes": sorted(int(e) for e in episodes), "num_episodes": len(episodes),
            "context_frames": int(context_frames), "window_horizon": 1, "num_windows": len(windows),
            "window_seed": int(seed), "held_out_windows_available": int(available),
            "latents_dir": os.path.abspath(latents_dir), "subset_key": SUBSET,
            "source_split": os.path.abspath(source_split) if source_split else None,
            "source_split_sha256": sha256_bytes(source_split) if source_split else None,
            "code": git_state(),
            "note": "adaptation split for adapt_wm.py; score with eval_tf.py --split <this file> --subset val "
                    f"--windows-file <this file> --windows-key {WINDOW_KEY}"}
    out = {"train": list(ep["adapt"]), SUBSET: list(ep["held_out"]), "adapt": ep["adapt"],
           "adapt_pool": ep["adapt_pool"], "held_out": ep["held_out"], WINDOW_KEY: windows}
    if legacy:
        kept, total, drawn = legacy_held_out_windows(latents_dir, episodes, ep["held_out"], legacy_windows,
                                                     legacy_seed, context_frames)
        out[LEGACY_KEY] = kept
        meta["legacy"] = {"num_windows": int(legacy_windows), "seed": int(legacy_seed), "map_windows": int(total),
                          "drawn": int(drawn), "kept_in_held_out": len(kept),
                          "score_with_split": meta["source_split"],
                          "note": "the distance study's eval_tf draw over every episode of the map, restricted to "
                                  "the held-out episodes; score it with --split <source split> so the dataset "
                                  "indices, and so the sampler noise, are the frozen scores' own"}
    out["meta"] = meta
    return out


def write_split(split, path, force=False):
    """Write atomically; an existing file with different content is refused unless `force`."""
    text = json.dumps(split, indent=1)
    if os.path.exists(path) and not force:
        with open(path) as f:
            old = json.load(f)
        if {k: v for k, v in old.items() if k != "meta"} != {k: v for k, v in split.items() if k != "meta"}:
            raise SystemExit(f"{path} exists with a different split; checkpoints record its hash, so it is not "
                             "overwritten (pass --force to replace it deliberately)")
        return path, False
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)
    return path, True


def main(argv=None):
    """Write one adaptation split file and print where it went and what it holds."""
    a = build_parser().parse_args(argv)
    source = None
    if a.map_split:
        with open(a.map_split) as f:
            src = json.load(f)
        meta = src.get("meta", {})
        episodes = [int(e) for e in src[SUBSET]]
        set_name = a.set or meta.get("set")
        map_id = a.map if a.map is not None else meta.get("map")
        latents_dir = a.latents_dir or meta.get("latents_dir")
        source = a.map_split
    else:
        from doom_data import parse_episode_ids
        episodes = parse_episode_ids(a.episodes)
        set_name, map_id, latents_dir = a.set, a.map, a.latents_dir
    if not episodes or set_name is None or map_id is None or not latents_dir:
        raise SystemExit("name the map: --map-split FILE, or --episodes with --set, --map and --latents-dir")
    split = build_split(episodes, set_name, map_id, latents_dir, a.seed, a.adapt_episodes, a.held_out_episodes,
                        a.windows, a.context_frames, legacy=bool(source) and not a.no_legacy,
                        legacy_windows=a.legacy_windows, legacy_seed=a.legacy_seed, source_split=source)
    path = a.out or os.path.join(a.out_dir, split_filename(set_name, map_id, a.seed, a.adapt_episodes,
                                                           a.held_out_episodes))
    path, wrote = write_split(split, path, a.force)
    print(json.dumps({"split": path, "written": wrote, "sha256": sha256_bytes(path), "held_out": split["held_out"],
                      "adapt": split["adapt"], "adapt_pool": split["adapt_pool"],
                      "held_out_windows": len(split[WINDOW_KEY]),
                      "legacy_held_out_windows": len(split.get(LEGACY_KEY, []))}))
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="Write a per-map adaptation split (adapt / held-out episodes, scored windows).")
    p.add_argument("--map-split", default="", help="the distance study's per-map split file "
                   "(results/distance_study/splits/split_<set>_map<NN>.json); its val list is the map's episodes")
    p.add_argument("--episodes", default="", help="the map's episode ids (A:B or a comma list) instead of --map-split")
    p.add_argument("--set", default=None, help="corpus name (unseen, unseen2, arenas_678); default from --map-split")
    p.add_argument("--map", type=int, default=None, help="map number; default from --map-split")
    p.add_argument("--latents-dir", default="", help="the corpus's per-tic latent directory (its sidecars are read); "
                   "default: the one --map-split records, which is a Spiderman path")
    p.add_argument("--seed", type=int, default=0, help="seeds the episode permutation and the window draw")
    p.add_argument("--adapt-episodes", type=int, default=DEFAULT_ADAPT,
                   help="adaptation episodes (default 6 on every map, 10 or 20 episodes alike)")
    p.add_argument("--held-out-episodes", type=int, default=DEFAULT_HELD_OUT,
                   help="held-out episodes (default 4 on every map); the rest of the map is the data-grid pool")
    p.add_argument("--windows", type=int, default=DEFAULT_WINDOWS, help="one-tic windows drawn from the held-out episodes")
    p.add_argument("--context-frames", type=int, default=CONTEXT_FRAMES)
    p.add_argument("--legacy-windows", type=int, default=DEFAULT_WINDOWS,
                   help="the distance study's per-map draw size, recorded restricted to the held-out episodes")
    p.add_argument("--legacy-seed", type=int, default=0, help="the distance study's draw seed")
    p.add_argument("--no-legacy", action="store_true", help="do not record the legacy draw")
    p.add_argument("--out-dir", default="results/adapt_splits")
    p.add_argument("--out", default="", help="explicit output path instead of --out-dir/<canonical name>")
    p.add_argument("--force", action="store_true", help="replace an existing split file with a different split")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
