"""
Per-map adaptation splits: which episodes of an unseen map adapt the model, which score it, and on which windows.

The adaptation study (`docs/lora_adaptation_design_2026-09-26.html`, decision 1) adapts a model pretrained on
the four training maps to ONE unseen map from some of its episodes and scores it on that map's other
episodes. This module writes that split, once per (map, seed), before anything trains. For the fresh
24-episode target arenas, one file per map straight from the corpus:

    python adapt_split.py --latents-dir $D/latents_arnold_eval_v2_pertic/arenas13 --manifest <its manifest> \\
        --map all --seed 0 --out-dir results/adapt_splits

**Where the episode ids come from.** Never from a hard-coded count. One of:

  * `--manifest FILE`: the corpus manifest, as JSON or JSON lines. Accepted shapes: a list (or lines) of
    episode records carrying `episode_id` (or `episode`) and `map_id` (or `map`), which is what the
    recorder's `worker_*.jsonl` provenance records carry; `{"episodes": [records]}`; or
    `{"maps": {"<map>": [ids]}}` / `{"<map>": [ids]}`. Every episode it lists for the map must be in the
    latent directory with that map id in its sidecar, or the split is refused.
  * a DIRECTORY LISTING (`--latents-dir` alone): every `ep_*_latents.npy` there, grouped by the `map_id`
    column of its sidecar.
  * `--map-split FILE`: the distance study's per-map split file (its `val` list is every episode of the
    map), which also enables the legacy window draw below.
  * `--episodes A:B|list` with `--map` and `--set`.

`--map` takes one map, a comma list, or `all` (every map the source lists); one file is written per map.

**Episodes.** One seeded permutation of the map's episodes (`make_adapt_split`): the first
`--held-out-episodes` (default 8) are held out; the rest, in permutation order, are the ADAPT LIST, whose
first `--adapt-episodes` (default 16) are the adaptation budget. The data ladder (`--ladder`, default
1,2,4,8,16) takes the first k of that list, so the rungs are nested and every rung shares one held-out set;
the step curve trains on the first `--step-curve-k` (default 8), which is what the trainer uses unless its
`--adapt-episodes-k` names another rung (`adapt_episodes`).

**Windows.** A fixed draw of `--windows-per-episode` (32) one-tic windows from EACH held-out episode (256
per map at 8 held out), seeded by (split seed, episode) so each episode's draw is its own, recorded as
[episode, start row] pairs under `held_out_windows`. Every checkpoint of every run on this split, step 0
included, is scored on exactly these windows, so the zero-shot number on them is the step-0 row. With
`--map-split`, the distance study's own draw (256 windows over all of the map's episodes, `eval_tf.draw_windows`
with seed 0) is recorded too, restricted to the held-out episodes (`legacy_held_out_windows`): a
cross-check against the frozen zero-shot scores, not the primary outcome. `windows_in_dataset` turns either
list back into dataset indices for `eval_tf.py --windows-file`.

The window rules are `doom_data.tic_window_starts` at horizon 1 (tic continuity, one life, one map) over the
episodes in `doom_data.list_latent_episodes` order, which is exactly how `TicWindowDataset` enumerates them,
so a recorded window is a dataset window by construction; the scorer refuses any that is not. Only the
sidecars are read, so one split serves every latent space of a corpus.

The file keeps `val` (the held-out episodes) and `train` (the step-curve episodes) as well, so `eval_tf.py`
and `directional_check.py` read it unchanged with `--subset val`.
"""
import argparse
import hashlib
import json
import os
import subprocess

import numpy as np

DEFAULT_ADAPT = 16
DEFAULT_HELD_OUT = 8
DEFAULT_LADDER = (1, 2, 4, 8, 16)
DEFAULT_STEP_CURVE_K = 8
DEFAULT_WINDOWS_PER_EPISODE = 32
LEGACY_WINDOWS = 256            # the distance study's per-map draw
CONTEXT_FRAMES = 32
WINDOW_KEY = "held_out_windows"
LEGACY_KEY = "legacy_held_out_windows"
WINDOW_KEYS = (WINDOW_KEY, LEGACY_KEY)
KIND = "adapt_split"
SUBSET = "val"                  # the key eval_tf.py and directional_check.py read (make_dense_eval_splits.SUBSET)
REPO = os.path.dirname(os.path.abspath(__file__))


def parse_ladder(spec):
    """"1,2,4,8,16" -> (1, 2, 4, 8, 16): sorted, unique, positive."""
    rungs = sorted({int(x) for x in str(spec).split(",") if str(x).strip()})
    if not rungs or rungs[0] < 1:
        raise ValueError(f"the data ladder needs positive episode counts, got {spec!r}")
    return tuple(rungs)


def make_adapt_split(episodes, seed=0, n_adapt=DEFAULT_ADAPT, n_held_out=DEFAULT_HELD_OUT, ladder=DEFAULT_LADDER,
                     step_curve_k=DEFAULT_STEP_CURVE_K):
    """{held_out, adapt, adapt_pool, ladder, step_curve_k} from one seeded permutation of a map's episode ids.

    `held_out` is sorted. `adapt` (the first `n_adapt` of the pool) and `adapt_pool` (every episode not
    held out) keep the permutation order, because their prefixes are the ladder. Deterministic in
    (episode set, seed, n_held_out) and independent of the input order.
    """
    ids = sorted(int(e) for e in episodes)
    if len(set(ids)) != len(ids):
        raise ValueError("the episode list repeats an id")
    n_adapt, n_held_out, step_curve_k = int(n_adapt), int(n_held_out), int(step_curve_k)
    if n_adapt < 1 or n_held_out < 1:
        raise ValueError(f"need at least one adaptation and one held-out episode, got {n_adapt} and {n_held_out}")
    if n_adapt + n_held_out > len(ids):
        raise ValueError(f"{n_adapt} adaptation + {n_held_out} held-out episodes exceed the map's {len(ids)}")
    ladder = parse_ladder(",".join(str(k) for k in ladder))
    if ladder[-1] > n_adapt or not 1 <= step_curve_k <= n_adapt:
        raise ValueError(f"the ladder {ladder} and the step-curve k {step_curve_k} must lie within the "
                         f"{n_adapt} adaptation episodes")
    perm = np.random.RandomState(int(seed)).permutation(len(ids))
    held_out = sorted(ids[i] for i in perm[:n_held_out])
    pool = [ids[i] for i in perm[n_held_out:]]
    return {"held_out": held_out, "adapt": pool[:n_adapt], "adapt_pool": pool, "ladder": list(ladder),
            "step_curve_k": step_curve_k}


def adapt_episodes(split, k=0):
    """The episodes a run adapts on: the first `k` of the adapt list, `k = 0` meaning the step-curve rung."""
    k = int(k or 0) or int(split.get("step_curve_k") or len(split["adapt"]))
    pool = list(split.get("adapt_pool") or split["adapt"])
    if k < 1 or k > len(pool):
        raise ValueError(f"--adapt-episodes-k {k}: this split's adapt list holds {len(pool)} episodes")
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


def draw_held_out_windows(latents_dir, held_out, per_episode=DEFAULT_WINDOWS_PER_EPISODE, seed=0,
                          context_frames=CONTEXT_FRAMES):
    """([episode, start] pairs, {episode: windows available}): `per_episode` windows from each held-out episode.

    Each episode draws from its own generator, seeded by (seed, episode), so an episode's windows do not
    depend on which other episodes are held out. An episode with fewer valid windows gives all of them.
    """
    per = window_starts(latents_dir, held_out, context_frames, 1)
    pairs, available = [], {}
    for ep, starts in per:
        available[int(ep)] = int(len(starts))
        pick = np.random.RandomState([int(seed), int(ep)]).choice(len(starts), size=min(per_episode, len(starts)),
                                                                   replace=False)
        pairs += [[int(ep), int(s)] for s in np.sort(starts[pick])]
    for ep in held_out:
        available.setdefault(int(ep), 0)
    return pairs, available


def legacy_held_out_windows(latents_dir, map_episodes, held_out, num_windows=LEGACY_WINDOWS, seed=0,
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


# ---------------------------------------------------------------------------------------------
# where a map's episode ids come from: a manifest or the directory itself
# ---------------------------------------------------------------------------------------------

def _record_ids(rec):
    ep = rec.get("episode_id", rec.get("episode"))
    m = rec.get("map_id", rec.get("map"))
    if ep is None or m is None:
        raise ValueError(f"a manifest record needs episode_id (or episode) and map_id (or map): {rec}")
    return int(ep), int(m)


def manifest_maps(path):
    """{map id: sorted episode ids} from a corpus manifest (the shapes are in the module docstring)."""
    with open(path) as f:
        text = f.read()
    try:
        data = json.loads(text)
    except ValueError:
        data = [json.loads(ln) for ln in text.splitlines() if ln.strip()]      # JSON lines
    if isinstance(data, dict) and isinstance(data.get("episodes"), list):
        data = data["episodes"]
    out = {}
    if isinstance(data, list):
        for rec in data:
            ep, m = _record_ids(rec)
            out.setdefault(m, set()).add(ep)
    elif isinstance(data, dict):
        maps = data.get("maps", data)
        for m, ids in maps.items():
            if not str(m).lstrip("-").isdigit() or not isinstance(ids, list):
                raise ValueError(f"{path}: a map entry must be <map number>: [episode ids], got {m!r}")
            out[int(m)] = {int(e) for e in ids}
    else:
        raise ValueError(f"{path}: not a manifest this writer can read")
    return {m: sorted(v) for m, v in out.items()}


def directory_maps(latents_dir):
    """{map id: sorted episode ids} of a latent directory, from each episode's sidecar `map_id` column."""
    from doom_data import list_latent_episodes
    out = {}
    for ep, _, meta_path in list_latent_episodes(latents_dir):
        with np.load(meta_path) as meta:
            ids = np.unique(np.asarray(meta["map_id"]))
        if len(ids) != 1:
            raise ValueError(f"{meta_path}: an episode on {len(ids)} maps ({ids.tolist()}) belongs to no one map")
        out.setdefault(int(ids[0]), []).append(int(ep))
    return {m: sorted(v) for m, v in out.items()}


def check_against_directory(maps, on_disk, latents_dir):
    """Refuse a manifest whose episodes are missing from the directory or recorded on another map there."""
    where = {ep: m for m, eps in on_disk.items() for ep in eps}
    for m, eps in maps.items():
        missing = [e for e in eps if e not in where]
        wrong = [(e, where[e]) for e in eps if e in where and where[e] != m]
        if missing or wrong:
            raise SystemExit(f"the manifest's map {m} disagrees with {latents_dir}: episodes not there {missing[:6]}, "
                             f"episodes whose sidecar names another map {wrong[:6]}")


# ---------------------------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------------------------

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
                ladder=DEFAULT_LADDER, step_curve_k=DEFAULT_STEP_CURVE_K,
                windows_per_episode=DEFAULT_WINDOWS_PER_EPISODE, context_frames=CONTEXT_FRAMES, legacy=False,
                legacy_windows=LEGACY_WINDOWS, legacy_seed=0, source=None, source_kind=None):
    """The whole split record: episodes, the ladder, the held-out window draw, the legacy draw, provenance."""
    ep = make_adapt_split(episodes, seed, n_adapt, n_held_out, ladder, step_curve_k)
    windows, available = draw_held_out_windows(latents_dir, ep["held_out"], windows_per_episode, seed, context_frames)
    short = {e: n for e, n in available.items() if n < windows_per_episode}
    if short:
        print(f"note: held-out episode(s) with fewer than {windows_per_episode} valid windows give all they have: {short}")
    meta = {"kind": KIND, "set": set_name, "map": int(map_id), "corpus": f"{set_name}_map{int(map_id):02d}",
            "seed": int(seed), "n_adapt": int(n_adapt), "n_held_out": int(n_held_out),
            "ladder": ep["ladder"], "step_curve_k": ep["step_curve_k"],
            "episodes": sorted(int(e) for e in episodes), "num_episodes": len(episodes),
            "context_frames": int(context_frames), "window_horizon": 1, "num_windows": len(windows),
            "windows_per_episode": int(windows_per_episode), "window_seed": int(seed),
            "held_out_windows_available": {str(e): n for e, n in sorted(available.items())},
            "latents_dir": os.path.abspath(latents_dir), "subset_key": SUBSET,
            "source": os.path.abspath(source) if source else None, "source_kind": source_kind,
            "source_sha256": sha256_bytes(source) if source else None,
            "code": git_state(),
            "note": "adaptation split for adapt_wm.py; score with eval_tf.py --split <this file> --subset val "
                    f"--windows-file <this file> --windows-key {WINDOW_KEY}"}
    out = {"train": adapt_episodes(ep, 0), SUBSET: list(ep["held_out"]), "held_out": ep["held_out"],
           "adapt": ep["adapt"], "adapt_pool": ep["adapt_pool"], "ladder": ep["ladder"],
           "step_curve_k": ep["step_curve_k"], WINDOW_KEY: windows}
    if legacy:
        kept, total, drawn = legacy_held_out_windows(latents_dir, episodes, ep["held_out"], legacy_windows,
                                                     legacy_seed, context_frames)
        out[LEGACY_KEY] = kept
        meta["legacy"] = {"num_windows": int(legacy_windows), "seed": int(legacy_seed), "map_windows": int(total),
                          "drawn": int(drawn), "kept_in_held_out": len(kept), "score_with_split": meta["source"],
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


def select_maps(spec, available):
    """The maps to write: one number, a comma list, or `all` of the source's maps."""
    s = str(spec or "").strip().lower()
    if s == "all":
        return sorted(available)
    if not s:
        raise SystemExit("--map names the map(s): a number, a comma list, or all")
    want = sorted({int(x) for x in s.split(",") if x.strip()})
    missing = [m for m in want if m not in available]
    if missing:
        raise SystemExit(f"map(s) {missing} are not in the source, which lists {sorted(available)}")
    return want


def sources(a):
    """[(set name, map id, episode ids, latents dir, source path, source kind)] for every map to write."""
    if a.map_split:
        with open(a.map_split) as f:
            src = json.load(f)
        meta = src.get("meta", {})
        m = int(a.map) if a.map not in (None, "") else meta.get("map")
        return [(a.set or meta.get("set"), m, [int(e) for e in src[SUBSET]], a.latents_dir or meta.get("latents_dir"),
                 a.map_split, "distance_study_split")]
    if not a.latents_dir:
        raise SystemExit("--latents-dir is needed: the windows are drawn from its sidecars")
    set_name = a.set or os.path.basename(os.path.normpath(a.latents_dir))
    if a.episodes:
        from doom_data import parse_episode_ids
        if a.map in (None, "") or "," in str(a.map) or str(a.map).lower() == "all":
            raise SystemExit("--episodes names ONE map's episodes; give that map with --map")
        return [(set_name, int(a.map), parse_episode_ids(a.episodes), a.latents_dir, None, "episodes")]
    on_disk = directory_maps(a.latents_dir)
    if a.manifest:
        maps = manifest_maps(a.manifest)
        check_against_directory({m: maps[m] for m in select_maps(a.map, maps)}, on_disk, a.latents_dir)
        kind, path = "manifest", a.manifest
    else:
        maps, kind, path = on_disk, "directory_listing", None
    return [(set_name, m, maps[m], a.latents_dir, path, kind) for m in select_maps(a.map, maps)]


def main(argv=None):
    """Write one adaptation split file per selected map and print where each went and what it holds."""
    a = build_parser().parse_args(argv)
    ladder = parse_ladder(a.ladder)
    todo = sources(a)
    if a.out and len(todo) != 1:
        raise SystemExit("--out names one file; with several maps use --out-dir")
    for set_name, map_id, episodes, latents_dir, source, kind in todo:
        if not episodes or set_name is None or map_id is None or not latents_dir:
            raise SystemExit("name the map and its corpus: see --help for the four sources")
        split = build_split(episodes, set_name, map_id, latents_dir, a.seed, a.adapt_episodes, a.held_out_episodes,
                            ladder, a.step_curve_k, a.windows_per_episode, a.context_frames,
                            legacy=(kind == "distance_study_split") and not a.no_legacy,
                            legacy_windows=a.legacy_windows, legacy_seed=a.legacy_seed, source=source,
                            source_kind=kind)
        path = a.out or os.path.join(a.out_dir, split_filename(set_name, map_id, a.seed, a.adapt_episodes,
                                                               a.held_out_episodes))
        path, wrote = write_split(split, path, a.force)
        print(json.dumps({"split": path, "written": wrote, "sha256": sha256_bytes(path), "map": map_id,
                          "episodes": len(episodes), "held_out": split["held_out"], "adapt": split["adapt"],
                          "step_curve": split["train"], "ladder": split["ladder"],
                          "held_out_windows": len(split[WINDOW_KEY]),
                          "legacy_held_out_windows": len(split.get(LEGACY_KEY, []))}))
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="Write per-map adaptation splits (adapt / held-out episodes, the data "
                                            "ladder, the scored windows).")
    p.add_argument("--latents-dir", default="", help="the corpus's per-tic latent directory; its sidecars give the "
                   "windows and, with no other source, the episodes of each map")
    p.add_argument("--manifest", default="", help="the corpus manifest listing each episode's map (JSON or JSON "
                   "lines; checked against the directory)")
    p.add_argument("--map-split", default="", help="a distance-study per-map split file instead "
                   "(results/distance_study/splits/split_<set>_map<NN>.json); adds the legacy window draw")
    p.add_argument("--episodes", default="", help="one map's episode ids (A:B or a comma list), with --map")
    p.add_argument("--map", default="", help="a map number, a comma list, or all (default from --map-split)")
    p.add_argument("--set", default=None, help="corpus name in the file names (default: the latent directory's "
                   "basename, e.g. arenas13, or the --map-split's set)")
    p.add_argument("--seed", type=int, default=0, help="seeds the episode permutation and the window draw")
    p.add_argument("--adapt-episodes", type=int, default=DEFAULT_ADAPT, help="adaptation episodes per map (default 16)")
    p.add_argument("--held-out-episodes", type=int, default=DEFAULT_HELD_OUT, help="held-out episodes per map (default 8)")
    p.add_argument("--ladder", default=",".join(map(str, DEFAULT_LADDER)),
                   help="the nested data-ladder rungs, each the first k of the adapt list (default 1,2,4,8,16)")
    p.add_argument("--step-curve-k", type=int, default=DEFAULT_STEP_CURVE_K,
                   help="adaptation episodes of the step curve, the trainer's default rung (default 8)")
    p.add_argument("--windows-per-episode", type=int, default=DEFAULT_WINDOWS_PER_EPISODE,
                   help="fixed one-tic windows drawn from each held-out episode (default 32, 256 per map at 8)")
    p.add_argument("--context-frames", type=int, default=CONTEXT_FRAMES)
    p.add_argument("--legacy-windows", type=int, default=LEGACY_WINDOWS,
                   help="the distance study's per-map draw size, recorded restricted to the held-out episodes")
    p.add_argument("--legacy-seed", type=int, default=0, help="the distance study's draw seed")
    p.add_argument("--no-legacy", action="store_true", help="do not record the legacy draw")
    p.add_argument("--out-dir", default="results/adapt_splits")
    p.add_argument("--out", default="", help="explicit output path (one map only) instead of --out-dir/<canonical name>")
    p.add_argument("--force", action="store_true", help="replace an existing split file with a different split")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
