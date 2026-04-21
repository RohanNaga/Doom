"""
Inspect a consolidated dataset directory. Prints shapes, dtypes, ranges, and
estimates training steps/epochs at a given global batch size.

Usage:
    python inspect_dataset.py                          # default: data/
    python inspect_dataset.py --feature-path data/debug
    python inspect_dataset.py --global-batch-size 128
"""
import argparse
import os

import numpy as np


def _pick(feature_path, full_name, debug_name):
    full_path = os.path.join(feature_path, full_name)
    debug_path = os.path.join(feature_path, debug_name)
    return full_path if os.path.isfile(full_path) else debug_path


def main(args):
    ctx_path = _pick(args.feature_path, "context_latents.npy", "context_latents_debug.npy")
    tgt_path = _pick(args.feature_path, "target_latents.npy", "target_latents_debug.npy")
    act_path = _pick(args.feature_path, "context_actions.npy", "context_actions_debug.npy")

    print(f"feature_path: {args.feature_path}")
    for label, p in [("context", ctx_path), ("target", tgt_path), ("actions", act_path)]:
        if not os.path.isfile(p):
            print(f"  MISSING {label}: {p}")
            return
        size_gb = os.path.getsize(p) / 1e9
        print(f"  {label}: {os.path.basename(p)}  ({size_gb:.2f} GB on disk)")

    ctx = np.load(ctx_path, mmap_mode="r")
    tgt = np.load(tgt_path, mmap_mode="r")
    act = np.load(act_path)  # small enough to fully load

    print()
    print(f"context_latents  shape={ctx.shape}  dtype={ctx.dtype}")
    print(f"target_latents   shape={tgt.shape}  dtype={tgt.dtype}")
    print(f"context_actions  shape={act.shape}  dtype={act.dtype}")
    print(f"  action min={act.min()}  max={act.max()}  n_unique={len(np.unique(act))}")
    print(f"  action unique values: {np.unique(act)}")

    # Last column is what CustomDataset conditions on (target action per build_dataset.py).
    if act.ndim == 2:
        last_col = act[:, -1]
        print(f"  target action (col -1): min={last_col.min()} max={last_col.max()} "
              f"n_unique={len(np.unique(last_col))}")

    assert ctx.shape[0] == tgt.shape[0] == act.shape[0], \
        f"sample count mismatch: ctx={ctx.shape[0]} tgt={tgt.shape[0]} act={act.shape[0]}"

    N = ctx.shape[0]
    steps_per_epoch = N // args.global_batch_size
    print()
    print(f"Total samples: {N:,}")
    print(f"At global_batch_size={args.global_batch_size} (drop_last): "
          f"{steps_per_epoch:,} steps/epoch")

    # Rough projections at ~8 steps/sec on 4 A4000s with grad_ckpt.
    steps_per_sec = args.steps_per_sec
    for target_hours in (1, 4, 8):
        total_steps = int(target_hours * 3600 * steps_per_sec)
        epochs = total_steps / max(steps_per_epoch, 1)
        print(f"  ~{target_hours}h @ {steps_per_sec} steps/s → {total_steps:,} steps "
              f"(~{epochs:.1f} epochs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="data")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--steps-per-sec", type=float, default=8.0,
                        help="Estimate of sustained throughput; used for epoch projections only.")
    args = parser.parse_args()
    main(args)
