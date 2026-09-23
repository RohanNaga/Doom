"""Pick ONE checkpoint of a run by its STORED step, so live and EMA are scored from the same file.

`after_nexttic.sh` used to select the EMA checkpoint with
`ls $R/[0-9]*.pt $R/snap_*.pt | sort | tail -1`. That sort is lexicographic across two families, so
`snap_0290000.pt` beats `0300000.pt`: whenever any snapshot exists, EVERY numbered recovery
checkpoint loses to it, including newer ones. Live was read from `best.pt`, selected on validation
loss at some other step, so the live/EMA comparison was systematically unpaired and the two numbers
came from different weights.

The step is read from the checkpoint itself, not from its filename: the name is a convention and the
stored `step` is the fact. `mmap=True` maps the archive instead of materialising its tensors, so
reading `step` out of a 37 GB recovery checkpoint costs a few page faults.

    python pick_checkpoint.py --results-dir results/042-sd35-nexttic
    python pick_checkpoint.py --results-dir results/042-sd35-nexttic --step 290000
    python pick_checkpoint.py --ckpt results/042-sd35-nexttic/snap_0290000.pt

Prints one line, `<path> <step> <has_ema>`, which is what the launcher reads with `read -r`;
`--hash` appends the file's SHA-256. `--json` prints the full record instead. Exit 1 when nothing
matches.
"""
import argparse
import glob
import json
import os


def candidates(results_dir):
    """Every checkpoint of a run that can carry weights: the recovery files and the snapshots."""
    numbered = [p for p in glob.glob(os.path.join(results_dir, "*.pt"))
                if os.path.basename(p)[0].isdigit()]
    return sorted(numbered + glob.glob(os.path.join(results_dir, "snap_*.pt")))


def describe(path):
    """(step, has_ema) of one checkpoint, read from the file and not from its name."""
    import torch
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, ValueError, TypeError):
        # an archive saved without the zip format cannot be mmapped; fall back to a full read
        ck = torch.load(path, map_location="cpu", weights_only=False)
    ema = ck.get("ema")
    return {"path": path, "step": int(ck.get("step") or 0), "has_ema": bool(ema),
            "val_loss": ck.get("val_loss"), "kind": "snapshot" if
            os.path.basename(path).startswith("snap_") else "recovery"}


def pick(results_dir=None, ckpt=None, step=None, require_ema=False):
    """The one checkpoint to score, with a reason when there is none.

    `--ckpt` names a file outright. Otherwise the candidate with the largest stored step wins, or
    the one whose stored step equals `--step`; among equal steps a file that carries an EMA wins,
    because the point of the selection is to score both variants from one set of weights.
    """
    if ckpt:
        if not os.path.isfile(ckpt):
            raise SystemExit(f"no checkpoint at {ckpt}")
        return describe(ckpt)
    found = [describe(p) for p in candidates(results_dir)]
    if step is not None:
        found = [r for r in found if r["step"] == int(step)]
        if not found:
            raise SystemExit(f"no checkpoint in {results_dir} carries step {step}")
    if require_ema:
        with_ema = [r for r in found if r["has_ema"]]
        if not with_ema:
            raise SystemExit(f"no checkpoint in {results_dir} carries an EMA "
                             "(best.pt never does; a snapshot or a recovery checkpoint does)")
        found = with_ema
    if not found:
        raise SystemExit(f"no numbered or snapshot checkpoint in {results_dir}")
    return max(found, key=lambda r: (r["step"], r["has_ema"]))


def main(args):
    r = pick(args.results_dir or None, args.ckpt or None,
             args.step if args.step >= 0 else None, args.require_ema)
    if args.hash:
        from eval_identity import sha256_file
        r["sha256"] = sha256_file(r["path"])
    line = f"{r['path']} {r['step']} {int(r['has_ema'])}" + (f" {r['sha256']}" if args.hash else "")
    print(json.dumps(r, indent=1) if args.json else line)
    return 0


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", dest="results_dir", default="", help="a run directory to choose from")
    p.add_argument("--ckpt", default="", help="score this exact checkpoint instead of choosing one")
    p.add_argument("--step", type=int, default=-1, help="require this stored step")
    p.add_argument("--require-ema", dest="require_ema", action="store_true",
                   help="only consider checkpoints that carry an EMA, so live and EMA come from one file")
    p.add_argument("--hash", action="store_true",
                   help="append the file's SHA-256 (eval_identity.py, cached beside the file): the evaluation "
                        "launcher keys every cached score by it, so a stale score cannot carry a new file's name")
    p.add_argument("--json", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
