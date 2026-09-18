"""
Convert the per-episode parquet recording into Hub-friendly shards (step 1 of the release).

Converts the `frame` binary column to the `datasets` Image feature so the viewer renders
frames, keeps every other column, and appends whole episodes to a shard until it holds at
least --rows-per-shard rows (so no episode is split), writing data/train-XXXXX.parquet under
--work-dir. Restartable at the shard level: existing shards are kept and the conversion
resumes after them. The Hub upload itself, the checksums and the verification are done by
release/upload.py from release/manifest.json, which points at this script's --work-dir.

    python release/hf_upload.py --in-dir /sata2/data/rnagabhi/doom/raw_arnold \
        --work-dir /sata2/data/rnagabhi/doom/hf_release_arnold

The parquet schema metadata of the source files (key `doomdit_episode`, the seed record) is
not carried into the converted shards; the dataset card points readers to the provenance
jsonl files for it.
"""
import argparse
import glob
import os


def convert(in_dir, work_dir, max_rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Dataset, Features, Image, Value
    os.makedirs(os.path.join(work_dir, "data"), exist_ok=True)
    paths = sorted(glob.glob(os.path.join(in_dir, "ep_*.parquet")))
    shard, buf = 0, []
    feats = None
    def flush():
        nonlocal shard, buf, feats
        if not buf:
            return
        out = os.path.join(work_dir, "data", f"train-{shard:05d}.parquet")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            shard += 1; buf = []            # written by an earlier run over the same episode list
            return
        table = pa.concat_tables(buf)
        d = table.to_pydict()
        d["frame"] = [{"bytes": b, "path": None} for b in d.pop("frame")]
        if feats is None:
            feats = Features({k: (Image() if k == "frame" else (Value("string") if k == "buttons" else Value("int64") if table.schema.field(k).type in (pa.int8(), pa.int16(), pa.int32(), pa.int64()) else Value("float32"))) for k in d})
        Dataset.from_dict(d, features=feats).to_parquet(out + ".tmp")
        os.replace(out + ".tmp", out)
        shard += 1; buf = []
    rows = 0
    for p in paths:
        t = pq.read_table(p)
        buf.append(t); rows += t.num_rows
        if rows >= max_rows:
            flush(); rows = 0
    flush()
    return shard


def main(args):
    n = convert(args.in_dir, args.work_dir, args.rows_per_shard)
    print(f"{n} shards written to {args.work_dir}; next: python release/upload.py stage ... (see release/README_RELEASE.md)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True); p.add_argument("--work-dir", required=True)
    p.add_argument("--rows-per-shard", type=int, default=15000, help="flush after the episode that reaches this many rows; about 0.9 GB at 51 KB per frame")
    main(p.parse_args())
