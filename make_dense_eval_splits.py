"""Write the split file an evaluation corpus needs, once every expected episode is validly encoded.

`eval_tf.py` and `rollout_eval.py` both take `--split` and `--subset`, and both read
`split[subset]` as the list of episodes to score. A dense evaluation corpus is a whole held-out
range, so every episode in the directory is the evaluation set and there is nothing to draw: the
split file just names them.

**One subset key for every corpus.** The key is always `val`, whatever the corpus is called, so a
caller never has to remember that the test corpus's episodes live under `test` while the validation
corpus's live under `val`. `after_nexttic.sh` passes `--subset val` everywhere for that reason. The
corpus's real name and its id range are recorded alongside, so the file says what it is.

**It publishes only a complete, valid corpus.** It used to name whatever happened to be in the
directory, so a shard that was still running, or a directory holding the wrong id range, produced an
apparently valid evaluation split of a partial corpus. `--expect-ids` states the range the split must
cover exactly, and every episode is checked as a PAIR: latent shape, every sidecar column present and
as long as the latents, and a finite sample of the latents. A latent with no sidecar is an orphan of
an interrupted write and is reported rather than ignored.

    python make_dense_eval_splits.py --latents-dir $D/latents_arnold_dense_pertic_eval/val \
        --expect-ids 6000:6100
    python make_dense_eval_splits.py --latents-dir $D/latents_arnold_dense_pertic_eval/test \
        --expect-ids 7000:7100
    python make_dense_eval_splits.py --latents-dir $D/latents_arnold_dense_pertic_eval/arenas_678 \
        --expect-ids 60:120 --refuse-worker-first $D/raw_arnold_dense/arenas_678

**The unseen subset must hold no worker-first episode.** Arnold's weapon-select requests execute only
in a recorder process's first episode (k = 0; `release/DATASET_CARD.md`), a control regime that
validation and test never contain. `--refuse-worker-first <raw dir>` reads each expected episode's
raw `buttons` column and refuses to publish if any of them is k = 0. The rule is the one the
2026-09-22 review measured with: `transitions.button_width_report(...)["inferred_starts"]`, k from
the widest request string. An episode with no weapon request at all cannot be classified; it is
reported and allowed, since nothing in it could have executed.
"""
import argparse
import glob
import json
import os

import numpy as np

SUBSET = "val"          # the one key every evaluator is pointed at


def corpus_name(latents_dir):
    """The corpus's name IS its directory basename, so a caller cannot pass a different one.

    The encoder's internal tag for the unseen corpus is `unseen` while its directory is
    `arenas_678`; naming the file after the tag wrote `.../arenas_678/split_unseen.json` while
    `after_nexttic.sh` looked for `.../split_arenas_678.json`. Deriving both from the directory is
    what makes the writer and the reader agree by construction.
    """
    return os.path.basename(os.path.normpath(latents_dir))


def split_path(latents_dir):
    """THE canonical location: `split_<corpus>.json` in the PARENT of the corpus directory.

    The parent, because `after_nexttic.sh` holds one evaluation root (`$LE`) and reads
    `$LE/split_<corpus>.json` for every corpus; one directory of split files beside the corpora.
    """
    return os.path.join(os.path.dirname(os.path.normpath(latents_dir)),
                        f"split_{corpus_name(latents_dir)}.json")


def orphan_latents(latents_dir):
    """Episode ids whose `_latents.npy` exists with no `_meta.npz` beside it.

    `encode_parquet.encode_episode` renames the latents into place BEFORE writing the sidecar, so an
    interrupted encode leaves exactly this. `list_latent_episodes` skips such a pair silently, which
    is why a published split could be short without anything saying so.
    """
    out = []
    for lat in sorted(glob.glob(os.path.join(latents_dir, "ep_*_latents.npy"))):
        if not os.path.isfile(lat.replace("_latents.npy", "_meta.npz")):
            out.append(int(os.path.basename(lat).split("_")[1]))
    return out


def check_episode(lat_path, meta_path, latent_channels=None, sample=8, seed=0):
    """Problems with one (latent, sidecar) pair, as a list of strings; empty means it is usable."""
    from doom_data import TIC_CORPUS_COLUMNS, check_sidecar_buttons_dtype, latent_shape_v2
    bad = []
    lat = np.load(lat_path, mmap_mode="r")
    if lat.ndim != 4:
        return [f"{os.path.basename(lat_path)}: {lat.ndim} dimensions, not (T, C, 32, 40)"]
    want = latent_shape_v2(latent_channels if latent_channels else lat.shape[1])
    if tuple(lat.shape[1:]) != want:
        bad.append(f"{os.path.basename(lat_path)}: latent shape {tuple(lat.shape[1:])} != {want}")
    T = lat.shape[0]
    if T == 0:
        bad.append(f"{os.path.basename(lat_path)}: no frames")
    with np.load(meta_path) as m:
        missing = [c for c in TIC_CORPUS_COLUMNS if c not in m.files]
        if missing:
            bad.append(f"{os.path.basename(meta_path)}: missing columns {missing}")
        for c in TIC_CORPUS_COLUMNS:
            if c in m.files and len(m[c]) != T:
                bad.append(f"{os.path.basename(meta_path)}: {len(m[c])} rows of {c!r} vs {T} latents")
        if "buttons" in m.files:
            # a column wider than the executed control means this sidecar holds Arnold's raw request
            # strings: readable, but 38 to 49 MB per episode, and the trainer refuses it
            try:
                check_sidecar_buttons_dtype(m["buttons"], os.path.basename(meta_path))
            except ValueError as e:
                bad.append(str(e))
    if T and sample:
        rows = np.random.RandomState(seed).choice(T, size=min(int(sample), T), replace=False)
        block = np.asarray(lat[np.sort(rows)], dtype=np.float32)
        if not np.isfinite(block).all():
            bad.append(f"{os.path.basename(lat_path)}: non-finite latents in the sampled rows")
    return bad


def worker_first_episodes(parquet_dir, episode_ids):
    """k per episode from its raw `buttons` column, and which episodes are worker-first (k = 0).

    Returns `{"k": {ep: k or None}, "worker_first": [...], "unclassifiable": [...], "missing": [...]}`.
    Only the `buttons` column is read, so this costs one column projection per recording.
    """
    import pyarrow.parquet as pq
    from transitions import button_width_report
    out = {"k": {}, "worker_first": [], "unclassifiable": [], "missing": []}
    for ep in sorted(int(e) for e in episode_ids):
        path = os.path.join(parquet_dir, f"ep_{ep:05d}.parquet")
        if not os.path.isfile(path):
            out["missing"].append(ep)
            continue
        col = pq.read_table(path, columns=["buttons"])["buttons"].to_pylist()
        k = button_width_report(np.array(col, dtype=object))["inferred_starts"]
        out["k"][ep] = k
        if k is None:
            out["unclassifiable"].append(ep)
        elif k == 0:
            out["worker_first"].append(ep)
    return out


def check_no_worker_first(parquet_dir, episode_ids):
    """Problems, as strings, if any of these episodes is worker-first or has no recording to classify."""
    r = worker_first_episodes(parquet_dir, episode_ids)
    bad = []
    if r["worker_first"]:
        bad.append(f"{len(r['worker_first'])} worker-first episode(s) (k = 0, weapon-select requests can "
                   f"execute): {r['worker_first'][:16]}; the unseen subset must hold none")
    if r["missing"]:
        bad.append(f"{len(r['missing'])} episode(s) have no recording in {parquet_dir} to classify: "
                   f"{r['missing'][:8]}")
    return bad


def check_raw_tics(meta_path, parquet_path):
    """Problems, as strings, if the sidecar's `tic` column is not the recording's, row for row.

    A per-tic corpus keeps every recorded row, so the sidecar (which `check_episode` already holds
    to the latent row count) must list exactly the recording's tics in order. A dropped, duplicated
    or shifted row shows up here without reading a single frame.
    """
    import pyarrow.parquet as pq
    name = os.path.basename(meta_path)
    if not os.path.isfile(parquet_path):
        return [f"{name}: no recording at {parquet_path} to check its tics against"]
    raw = np.asarray(pq.read_table(parquet_path, columns=["tic"])["tic"]).astype(np.int64)
    with np.load(meta_path) as m:
        side = np.asarray(m["tic"]).astype(np.int64) if "tic" in m.files else None
    if side is None:
        return [f"{name}: no tic column"]
    if len(side) != len(raw):
        return [f"{name}: {len(side)} sidecar rows vs {len(raw)} recorded tics in {os.path.basename(parquet_path)}"]
    bad = np.flatnonzero(side != raw)
    if len(bad):
        i = int(bad[0])
        return [f"{name}: tic {int(side[i])} at row {i} but the recording has tic {int(raw[i])} "
                f"({len(bad)} rows differ)"]
    return []


def check_split_file(path, want):
    """Problems, as strings, if the split file at `path` does not list exactly the episodes `want`."""
    try:
        with open(path) as f:
            listed = sorted(int(e) for e in json.load(f)[SUBSET])
    except (OSError, ValueError, KeyError, TypeError) as e:
        return [f"split file {path} is missing or unreadable ({e.__class__.__name__}: {e})"]
    want = sorted(int(e) for e in want)
    if listed == want:
        return []
    extra = sorted(set(listed) - set(want))
    short = sorted(set(want) - set(listed))
    return [f"split file {path} lists {len(listed)} episode(s), not the {len(want)} expected: "
            f"{len(short)} missing {short[:8]}, {len(extra)} extra {extra[:8]}"]


def validate(latents_dir, expect_ids=None, latent_channels=None, sample=8, refuse_worker_first=None,
             split_file=None, raw_tics=None):
    """What this directory holds against what it should hold, as a report with no side effects.

    `refuse_worker_first` is the raw recording directory; with it, every expected episode (every
    usable one when no ids are expected) is classified, and a worker-first one is a problem.

    `split_file` is the published split an evaluator will read. Its `val` list must be exactly the
    expected ids (the usable ones when none are expected): `TicWindowDataset` intersects the split
    with what is encoded, so a stale or short split file would silently score a different episode set.

    `raw_tics` is the raw recording directory of a per-tic corpus; with it every episode's sidecar
    `tic` column must equal the recording's (`check_raw_tics`). This is the launch gate's row-count
    and tic-alignment check over the training corpus: shapes and one int column, no frames.
    """
    from doom_data import list_latent_episodes
    try:
        found = {ep: (lp, mp) for ep, lp, mp in list_latent_episodes(latents_dir)}
    except FileNotFoundError:
        found = {}
    problems, invalid = [], []
    for ep in sorted(found):
        bad = check_episode(*found[ep], latent_channels=latent_channels, sample=sample)
        if raw_tics and not bad:
            bad = check_raw_tics(found[ep][1], os.path.join(raw_tics, f"ep_{ep:05d}.parquet"))
        if bad:
            invalid.append(ep)
            problems += bad
    orphans = orphan_latents(latents_dir)
    if orphans:
        problems.append(f"{len(orphans)} latent file(s) with no sidecar (an interrupted encode): "
                        f"{orphans[:8]}")
    usable = sorted(ep for ep in found if ep not in set(invalid))
    report = {"latents_dir": latents_dir, "found": sorted(found), "usable": usable,
              "invalid": invalid, "orphans": orphans, "missing": [], "unexpected": [],
              "problems": problems}
    if expect_ids is not None:
        want = sorted(int(e) for e in expect_ids)
        report["expected"] = want
        report["missing"] = [e for e in want if e not in set(usable)]
        report["unexpected"] = [e for e in usable if e not in set(want)]
        if report["missing"]:
            problems.append(f"{len(report['missing'])} expected episode(s) are not validly encoded: "
                            f"{report['missing'][:8]}")
        if report["unexpected"]:
            problems.append(f"{len(report['unexpected'])} episode(s) present but not expected: "
                            f"{report['unexpected'][:8]}")
    if refuse_worker_first:
        ids = report.get("expected") or usable
        wf = worker_first_episodes(refuse_worker_first, ids)
        report["worker_first"], report["unclassifiable"] = wf["worker_first"], wf["unclassifiable"]
        problems += check_no_worker_first(refuse_worker_first, ids)
    if split_file:
        problems += check_split_file(split_file, report.get("expected") or usable)
    report["ok"] = not problems
    return report


def build(latents_dir, name=None, out=None, expect_ids=None, latent_channels=None, sample=8,
          refuse_worker_first=None):
    """Write the canonical split file for a COMPLETE, valid corpus and return (path, contents).

    Refuses to publish anything else: a split file is what an evaluator treats as the definition of
    the held-out set, so a partial or wrong-range corpus must not be able to become one.
    """
    report = validate(latents_dir, expect_ids, latent_channels, sample, refuse_worker_first)
    if not report["ok"]:
        raise SystemExit("this corpus cannot be published as an evaluation split:\n  "
                         + "\n  ".join(report["problems"]))
    name = name or corpus_name(latents_dir)
    eps = report["usable"]
    split = {SUBSET: eps, "meta": {"corpus": name, "latents_dir": latents_dir,
                                   "num_episodes": len(eps),
                                   "episode_range": f"{eps[0]}:{eps[-1] + 1}",
                                   "expected_ids": (f"{report['expected'][0]}:{report['expected'][-1] + 1}"
                                                    if report.get("expected") else None),
                                   "subset_key": SUBSET,
                                   "worker_first_checked": bool(refuse_worker_first),
                                   "note": "every encoded episode of a held-out dense range, each checked as a "
                                           "valid latent/sidecar pair; the key is 'val' for every corpus so the "
                                           "evaluators take one --subset"}}
    path = out or split_path(latents_dir)
    # a unique temporary name: two shards finalising one evaluation root shared `<path>.tmp` and
    # could rename each other's half-written file into place
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(split, f, indent=1)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path, split


def main(args):
    from doom_data import parse_episode_ids
    expect = parse_episode_ids(args.expect_ids) if args.expect_ids else None
    if args.check_only:
        report = validate(args.latents_dir, expect, args.latent_channels or None, args.sample,
                          args.refuse_worker_first or None, args.split_file or None, args.raw_tics or None)
        print(json.dumps(report, indent=1))
        return 0 if report["ok"] else 1
    path, split = build(args.latents_dir, args.name or None, args.out or None, expect,
                        args.latent_channels or None, args.sample, args.refuse_worker_first or None)
    print(json.dumps({"wrote": path, "num_episodes": split["meta"]["num_episodes"],
                      "episode_range": split["meta"]["episode_range"], "subset_key": SUBSET,
                      "expected_ids": split["meta"]["expected_ids"]}))
    return 0


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--latents-dir", required=True, help="an encoded per-tic evaluation corpus")
    p.add_argument("--name", default="",
                   help="override the corpus name; by default it is the latents directory's basename, which is "
                        "what keeps the writer's filename and the evaluation script's identical")
    p.add_argument("--out", default="", help="write here instead of the canonical <parent>/split_<corpus>.json")
    p.add_argument("--expect-ids", dest="expect_ids", default="",
                   help="the episode ids this corpus must hold EXACTLY, as A:B or a comma list. Without it the "
                        "split covers whatever is validly encoded, which is how a still-running shard could "
                        "publish a partial evaluation set")
    p.add_argument("--latent-channels", type=int, default=0,
                   help="assert the corpus's channel count (16 for the SD 3.5 corpus); 0 takes the first "
                        "episode's own")
    p.add_argument("--sample", type=int, default=8,
                   help="latent rows per episode read back and checked finite (0 skips the read)")
    p.add_argument("--refuse-worker-first", dest="refuse_worker_first", default="",
                   help="the RAW recording directory of this corpus; refuse to publish if any expected episode is "
                        "a recorder process's first episode (k = 0), the only control regime in which Arnold's "
                        "weapon-select requests execute. Used for the unseen subset")
    p.add_argument("--split-file", dest="split_file", default="",
                   help="with --check-only: also require this published split file to list exactly the expected "
                        "episodes. after_nexttic.sh runs this before scoring any corpus")
    p.add_argument("--raw-tics", dest="raw_tics", default="",
                   help="with --check-only: the raw recording directory of a per-tic corpus; every sidecar's tic "
                        "column must equal its recording's. The launch gate runs this over the training corpus")
    p.add_argument("--check-only", dest="check_only", action="store_true",
                   help="print the validation report and write nothing; exit 1 if the corpus is not publishable")
    return p


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
