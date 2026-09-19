"""Check a dense recording segment and write its md5 manifest.

`verify_corpus.py` compares two *latent* directories after re-encoding. This is the check one step
earlier: a directory of per-episode parquet recordings, before anything is encoded. It answers the
questions that decide whether a segment is finished and usable — how many episodes, how they split
across maps, whether any episode is short, whether the schema and the seed scheme are what the
corpus claims, and whether two episodes collide on an id or a seed.

Nothing here needs a GPU, ViZDoom or torch. Run it on a segment as it grows; it reports on whatever
is finished and never rewrites the recording.

    python verify_dense.py --dir $D/raw_arnold_dense/arenas --expect-maps 2,3,4,5 \
        --expect-corpus-id arnold-train-dense-v1 --manifest $D/raw_arnold_dense/arenas/md5.txt
"""
import argparse
import collections
import glob
import hashlib
import json
import os

import numpy as np

# The recording contract: every episode parquet has these columns, in this order, with these types.
SCHEMA = [("episode_id", "int32"), ("map_id", "int8"), ("tic", "int32"), ("action", "int16"),
          ("buttons", "string"), ("health", "int16"), ("ammo", "int16"), ("kills", "int16"),
          ("deaths", "int16"), ("frags", "int16"), ("pos_x", "float"), ("pos_y", "float"),
          ("angle", "float"), ("frame", "binary")]
EXPECTED_TICS = 5250        # 150 game-seconds at 35 tics/s; deaths cost a few, so this is an upper bound


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_episode(path):
    """Everything about one episode that does not need the frames decoded."""
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path)
    schema, n = f.schema_arrow, f.metadata.num_rows
    meta = json.loads((schema.metadata or {}).get(b"doomdit_episode", b"{}"))
    cols = pq.read_table(path, columns=["episode_id", "map_id", "tic", "deaths"])
    tic = cols["tic"].to_numpy(zero_copy_only=False)
    return dict(
        path=path, rows=n, bytes=os.path.getsize(path),
        names=schema.names, types=[str(t) for t in schema.types],
        episode_id=int(cols["episode_id"][0].as_py()), map_id=int(cols["map_id"][0].as_py()),
        first_tic=int(tic[0]) if n else -1, last_tic=int(tic[-1]) if n else -1,
        monotonic=bool(n and np.all(np.diff(tic) > 0)),
        lives=int(np.count_nonzero(np.diff(cols["deaths"].to_numpy(zero_copy_only=False))) + 1) if n else 0,
        meta=meta)


def verify(directory, expect_maps=None, expect_corpus_id=None, min_tics=0, sample_frames=0):
    """Report on a segment. `problems` is empty exactly when the segment is internally consistent."""
    paths = sorted(glob.glob(os.path.join(directory, "ep_*.parquet")))
    r = dict(directory=directory, episodes=len(paths), problems=[], per_map={}, corpus_ids=[],
             tics=dict(total=0, mean=0.0, min=0, max=0), bytes=0, bytes_per_episode=0,
             stored_tic_stride=[], ok=False)
    if not paths:
        r["problems"].append("no ep_*.parquet in the directory")
        return r
    r["stored_tic_stride"] = set()
    seen_ids, seeds, tics = {}, {}, []
    want_names = [c for c, _ in SCHEMA]
    for p in paths:
        e = read_episode(p)
        base = os.path.basename(p)
        if e["names"] != want_names:
            r["problems"].append(f"{base}: columns {e['names']} not the recording schema")
        else:
            for (name, want), got in zip(SCHEMA, e["types"]):
                if want not in got:
                    r["problems"].append(f"{base}: column {name} is {got}, expected {want}")
        if e["rows"] == 0:
            r["problems"].append(f"{base}: no rows")
            continue
        if not e["monotonic"]:
            r["problems"].append(f"{base}: tic column is not strictly increasing")
        if e["rows"] < min_tics:
            r["problems"].append(f"{base}: {e['rows']} tics, short of the {min_tics} required")
        if e["rows"] > EXPECTED_TICS:
            r["problems"].append(f"{base}: {e['rows']} tics, more than {EXPECTED_TICS} for one episode")
        m = e["meta"]
        cid, eid = m.get("corpus_id"), e["episode_id"]
        if cid and cid not in r["corpus_ids"]:
            r["corpus_ids"].append(cid)
        if expect_corpus_id and cid != expect_corpus_id:
            r["problems"].append(f"{base}: corpus id {cid!r}, expected {expect_corpus_id!r}")
        if eid in seen_ids:
            r["problems"].append(f"{base}: episode id {eid} already used by {seen_ids[eid]}")
        seen_ids[eid] = base
        # two episodes sharing a seed would be the same rollout twice, which silently halves the corpus
        key = json.dumps(m.get("seeds"), sort_keys=True) if m.get("seeds") else None
        if key:
            if key in seeds:
                r["problems"].append(f"{base}: same seeds as {seeds[key]}")
            seeds[key] = base
        r["stored_tic_stride"].add(int(m.get("stored_tic_stride", 1)))
        pm = r["per_map"].setdefault(str(e["map_id"]), dict(episodes=0, tics=0, lives=0, bytes=0))
        pm["episodes"] += 1; pm["tics"] += e["rows"]; pm["lives"] += e["lives"]; pm["bytes"] += e["bytes"]
        r["tics"]["total"] += e["rows"]; r["bytes"] += e["bytes"]
        tics.append(e["rows"])
    t = np.array(tics) if tics else np.zeros(1)
    r["tics"].update(mean=float(t.mean()), min=int(t.min()), max=int(t.max()))
    r["bytes_per_episode"] = r["bytes"] // max(len(tics), 1)
    r["stored_tic_stride"] = sorted(r["stored_tic_stride"])
    if len(r["stored_tic_stride"]) > 1:
        r["problems"].append(f"mixed row semantics in one directory: stored_tic_stride {r['stored_tic_stride']}")
    if expect_maps is not None:
        got = sorted(int(k) for k in r["per_map"])
        if got != sorted(expect_maps):
            r["problems"].append(f"maps {got}, expected {sorted(expect_maps)}")
    if sample_frames:
        r["frame_check"] = check_frames(paths, sample_frames, r["problems"])
    r["ok"] = not r["problems"]
    return r


def check_frames(paths, n, problems):
    """Decode a few PNGs from the first and last episode: cheap proof the frames are real pictures."""
    import io
    import pyarrow.parquet as pq
    from PIL import Image
    shapes = collections.Counter()
    for p in ({paths[0], paths[-1]}):
        t = pq.read_table(p, columns=["frame"])
        for i in np.linspace(0, t.num_rows - 1, min(n, t.num_rows), dtype=int):
            try:
                a = np.asarray(Image.open(io.BytesIO(t["frame"][int(i)].as_py())).convert("RGB"))
            except Exception as ex:
                problems.append(f"{os.path.basename(p)} row {i}: frame does not decode ({ex})")
                continue
            shapes[str(a.shape)] += 1
    return dict(shapes=dict(shapes))


def main(a):
    maps = [int(x) for x in a.expect_maps.split(",")] if a.expect_maps else None
    r = verify(a.dir, maps, a.expect_corpus_id, a.min_tics, a.sample_frames)
    if a.manifest:
        paths = sorted(glob.glob(os.path.join(a.dir, "ep_*.parquet")))
        tmp = a.manifest + ".tmp"
        with open(tmp, "w") as f:
            for p in paths:
                f.write(f"{md5(p)}  {os.path.basename(p)}\n")
        os.replace(tmp, a.manifest)
        r["manifest"] = dict(path=a.manifest, files=len(paths))
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
    print(json.dumps(r, indent=1))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="a segment directory of ep_*.parquet recordings")
    p.add_argument("--expect-maps", default=None, help="comma-separated map ids the segment must contain")
    p.add_argument("--expect-corpus-id", default=None)
    p.add_argument("--min-tics", type=int, default=0, help="flag any episode shorter than this")
    p.add_argument("--sample-frames", type=int, default=0, help="decode this many frames per sampled episode")
    p.add_argument("--manifest", default=None, help="write an md5sum-format manifest here")
    p.add_argument("--out", default=None, help="write the report as JSON here")
    raise SystemExit(main(p.parse_args()))
