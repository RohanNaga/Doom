"""
Upload a per-episode parquet recording to the Hugging Face Hub as a dataset.

Converts the `frame` binary column to the `datasets` Image feature so the viewer renders
frames, keeps every other column, writes shards under 1 GB into data/, adds the card, and
uploads with `upload_large_folder` (resumable, multi-commit). Needs HF_TOKEN in the
environment; nothing runs without it.

    python release/hf_upload.py --in-dir /sata2/.../raw_arnold --work-dir /sata2/.../hf_release_arnold \
        --repo RohanNaga/doom-arnold-320x240-lossless --card release/DATASET_CARD.md --buttons raw_arnold/buttons.json
"""
import argparse
import glob
import io
import json
import os
import shutil


def convert(in_dir, work_dir, max_rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Dataset, Features, Image, Value
    os.makedirs(os.path.join(work_dir, "data"), exist_ok=True)
    paths = sorted(glob.glob(os.path.join(in_dir, "ep_*.parquet")))
    shard, buf, n_out = 0, [], 0
    feats = None
    def flush():
        nonlocal shard, buf, feats
        if not buf:
            return
        table = pa.concat_tables(buf)
        d = table.to_pydict()
        d["frame"] = [{"bytes": b, "path": None} for b in d.pop("frame")]
        if feats is None:
            feats = Features({k: (Image() if k == "frame" else (Value("string") if k == "buttons" else Value("int64") if table.schema.field(k).type in (pa.int8(), pa.int16(), pa.int32(), pa.int64()) else Value("float32"))) for k in d})
        Dataset.from_dict(d, features=feats).to_parquet(os.path.join(work_dir, "data", f"train-{shard:05d}.parquet"))
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
    card = open(args.card).read()
    if args.buttons:
        card = card.replace("{{BUTTONS}}", json.dumps(json.load(open(args.buttons)), indent=1))
    open(os.path.join(args.work_dir, "README.md"), "w").write(card)
    print(f"{n} shards written to {args.work_dir}")
    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(args.repo, repo_type="dataset", exist_ok=True, private=args.private)
        api.upload_large_folder(repo_id=args.repo, repo_type="dataset", folder_path=args.work_dir)
        print("uploaded", args.repo)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True); p.add_argument("--work-dir", required=True)
    p.add_argument("--repo", required=True); p.add_argument("--card", required=True); p.add_argument("--buttons", default="")
    p.add_argument("--rows-per-shard", type=int, default=15000, help="about 0.75 GB at 50 KB per frame")
    p.add_argument("--upload", action="store_true"); p.add_argument("--private", action="store_true")
    main(p.parse_args())
