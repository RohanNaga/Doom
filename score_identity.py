"""The identity of what a score was computed ON: the corpus, its split file, and (later) its windows.

`after_nexttic.sh` pins the whole scoring configuration in `selection.json` before any test score
exists and refuses a sealed stage whose configuration differs (Astra's review, 2026-09-23). The
checkpoint and decoder are pinned by content hash elsewhere (`eval_identity.py`); this module gives
the corpus side:

  corpus   `split=<sha256 of the split file's CONTENTS> fingerprint=<train_wm.corpus_manifest of
           the split's episodes> episodes=<n>`. The split path is not identity: two files at one path
           can list different episodes, and the manifest reaches the latent values themselves.

    python score_identity.py corpus --latents-dir $D/latents_arnold_dense_pertic_eval/test \\
        --split $D/latents_arnold_dense_pertic_eval/split_test.json
"""
import argparse
import hashlib
import json
import os

SUBSET = "val"      # make_dense_eval_splits.SUBSET: every evaluation split file uses this key


def split_digest(split_path):
    """sha256 (16 hex) of the split file's bytes, or `missing`."""
    try:
        with open(split_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return "missing"


def split_ids(split_path, subset=SUBSET):
    try:
        with open(split_path) as f:
            return [int(e) for e in json.load(f)[subset]]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def corpus_identity(latents_dir, split_path, subset=SUBSET):
    """One line naming a corpus by content: split file bytes, latent fingerprint, episode count."""
    from train_wm import corpus_manifest
    ids = split_ids(split_path, subset)
    fp = corpus_manifest(latents_dir, ids)["fingerprint"] if os.path.isdir(latents_dir) else "absent"
    return f"split={split_digest(split_path)} fingerprint={fp} episodes={len(ids)}"


def main(args):
    print(corpus_identity(args.latents_dir, args.split, args.subset))
    return 0


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["corpus"])
    p.add_argument("--latents-dir", dest="latents_dir", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--subset", default=SUBSET)
    return p


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
