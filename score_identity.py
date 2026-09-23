"""The identity of what a score was computed ON: the corpus, its split file, and (later) its windows.

`after_nexttic.sh` pins the whole scoring configuration in `selection.json` before any test score
exists and refuses a sealed stage whose configuration differs (Astra's review, 2026-09-23). The
checkpoint and decoder are pinned by content hash elsewhere (`eval_identity.py`); this module gives
the corpus side:

  corpus   `split=<sha256 of the split file's CONTENTS> fingerprint=<train_wm.corpus_manifest of
           the split's episodes> episodes=<n>`. The split path is not identity: two files at one path
           can list different episodes, and the manifest reaches the latent values themselves.
  windows  `windows=<digest> n=<count>`: the exact (episode, start) pairs `eval_tf.py` or
           `rollout_eval.py --rollout` will score, drawn the way they draw them. Every cached
           score in `after_nexttic.sh` is keyed by the corpus line and this one, so a stale result
           can never be reused for another corpus, split or window set.

    python score_identity.py corpus --latents-dir $D/latents_arnold_dense_pertic_eval/test \\
        --split $D/latents_arnold_dense_pertic_eval/split_test.json
    python score_identity.py windows --kind tf --latents-dir ... --split ... --num 2048 --seed 0 --horizon 4
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


def _digest(rows):
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


def tf_window_manifest(latents_dir, split_path, num_windows, seed, context_frames, horizon, latent_channels,
                       subset=SUBSET):
    """The (episode, start row) pairs `eval_tf.py --tic-stride 1` scores, and their digest.

    Built from the same dataset and the same draw (`eval_tf.draw_windows`), so a change to the corpus,
    the split, the window count, the seed, the context length or the horizon changes it.
    """
    from doom_data import TicWindowDataset
    from eval_tf import draw_windows
    ds = TicWindowDataset(latents_dir, split_ids(split_path, subset), context_frames,
                          latent_channels=latent_channels, horizon=horizon, with_horizon=True)
    rows = []
    for gi in draw_windows(len(ds), num_windows, seed):
        slot, start = ds.locate(int(gi))
        rows.append([int(ds.episodes[slot][0]), int(start)])
    return rows, _digest([horizon, rows])


def rollout_window_manifest(latents_dir, split_path, num_rollouts, seed, context_frames, horizon, latent_channels,
                            subset=SUBSET):
    """The (episode, seed start row) pairs `rollout_eval.py --rollout` rolls out, and their digest."""
    from rollout_eval import collect_rollout_windows
    picks = collect_rollout_windows(latents_dir, split_ids(split_path, subset), context_frames, horizon,
                                    num_rollouts, seed, latent_channels=latent_channels, tic_stride=1)
    rows = [[int(ep), int(s)] for ep, _, s, _, _ in picks]
    return rows, _digest([horizon, rows])


def main(args):
    if args.cmd == "corpus":
        print(corpus_identity(args.latents_dir, args.split, args.subset))
        return 0
    build = tf_window_manifest if args.kind == "tf" else rollout_window_manifest
    rows, digest = build(args.latents_dir, args.split, args.num, args.seed, args.context_frames, args.horizon,
                         args.latent_channels, args.subset)
    print(f"windows={digest} n={len(rows)}")
    return 0


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["corpus", "windows"])
    p.add_argument("--latents-dir", dest="latents_dir", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--subset", default=SUBSET)
    p.add_argument("--kind", choices=["tf", "rollout"], default="tf", help="windows: which evaluator's draw")
    p.add_argument("--num", type=int, default=2048, help="windows: --num-windows (tf) or --num-rollouts (rollout)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--context-frames", dest="context_frames", type=int, default=32)
    p.add_argument("--horizon", type=int, default=1, help="windows: --horizon-tics (tf) or --horizon (rollout)")
    p.add_argument("--latent-channels", dest="latent_channels", type=int, default=4)
    return p


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
