"""Write a small split over the episodes a re-encode has finished, so a fit sweep can start early.

A full corpus encode takes hours, but a fit check or a short sweep only needs a few dozen
episodes. This reads the parent split (the one the full corpus will be trained under), keeps
only episodes whose latents *and* metadata are already on disk in `--latents-dir`, and writes
the first `--train` training and `--val` validation episodes of that intersection.

Taking the first N in sorted episode order, rather than a random draw, is deliberate: the file
is regenerated as more episodes land, and a prefix of a sorted list only ever grows, so a sweep
launched from an early copy sees the same episodes as one launched later. The episodes are a
subset of the parent split's own lists, so nothing held out for validation there can leak into
training here.

    python fit_split.py --latents-dir $D/latents_arnold_sd35 --split $D/split_arnold.json \
        --out $D/latents_arnold_sd35/split_fit.json --train 40 --val 8
"""
import argparse
import glob
import json
import os


def encoded_episodes(latents_dir):
    """Episode ids that have both a latents array and its metadata written.

    `encode_parquet.py` writes the latents to a `.tmp` and renames, then writes the meta, so a
    pair of files is the only evidence that an episode finished; a lone `_latents.npy` may be
    from an episode whose metadata write is still in flight.
    """
    out = []
    for lat in glob.glob(os.path.join(latents_dir, "ep_*_latents.npy")):
        if os.path.isfile(lat.replace("_latents.npy", "_meta.npz")):
            out.append(int(os.path.basename(lat).split("_")[1]))
    return sorted(out)


def fit_split(parent, present, n_train, n_val, allow_partial=False):
    """Prefix of the parent split's train/val lists restricted to `present` episode ids."""
    have = set(int(e) for e in present)
    picked = {}
    for key, want in (("train", n_train), ("val", n_val)):
        ids = [e for e in sorted(int(x) for x in parent[key]) if e in have]
        if len(ids) < want and not allow_partial:
            raise SystemExit(f"only {len(ids)} of {want} {key} episodes are encoded so far")
        picked[key] = ids[:want]
    return {"train": picked["train"], "val": picked["val"], "unseen_map": [],
            "meta": {"source_split": None, "latents_dir": None, "encoded_episodes": len(have),
                     "requested": {"train": n_train, "val": n_val},
                     "note": "prefix of the parent split over the episodes encoded so far; "
                             "for fit checks and short sweeps, never for a reported row"}}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--latents-dir", required=True)
    p.add_argument("--split", required=True, help="the full corpus split this is a prefix of")
    p.add_argument("--out", required=True)
    p.add_argument("--train", type=int, default=40)
    p.add_argument("--val", type=int, default=8)
    p.add_argument("--allow-partial", action="store_true", help="write whatever is encoded instead of failing")
    a = p.parse_args()
    with open(a.split) as f:
        parent = json.load(f)
    present = encoded_episodes(a.latents_dir)
    split = fit_split(parent, present, a.train, a.val, a.allow_partial)
    split["meta"]["source_split"] = os.path.abspath(a.split)
    split["meta"]["latents_dir"] = os.path.abspath(a.latents_dir)
    with open(a.out, "w") as f:
        json.dump(split, f, indent=1)
    print(f"{a.out}: {len(split['train'])} train, {len(split['val'])} val "
          f"of {len(present)} episodes encoded so far")
