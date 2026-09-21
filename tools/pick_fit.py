"""Pick the micro-batch and update budget for a decoder tune from its own fit checks.

Each fit check leaves a `metrics.json` holding the updates it completed, the seconds they took,
the effective batch and the peak memory. This turns those into the two numbers the real run needs:
the micro-batch that moved the most frames per second while staying inside the memory actually
free on the card, and the update budget whose linear decay ends when the card has to be handed
back. A fit check that ran out of memory leaves no `metrics.json`, so it simply does not compete.

Nothing here touches a GPU, which is the point: the choice is made from measurements, and it can
be tested anywhere.

    eval $(python tools/pick_fit.py --dir $D/tmp/levers --seconds-left 11700 --max-gb 22)
"""
import argparse
import glob
import json
import os
import re

MIN_STEPS = 20          # fewer than this is warmup, not a throughput measurement


def read_fits(directory):
    """[{micro, effective_batch, steps, seconds, frames_per_s, peak_gb}] for every fit that finished."""
    out = []
    for p in sorted(glob.glob(os.path.join(directory, "fit_mb*", "metrics.json"))):
        m = re.search(r"fit_mb(\d+)", p)
        if not m:
            continue
        try:
            d = json.load(open(p))
        except (OSError, ValueError):
            continue
        steps, secs = d.get("steps") or 0, d.get("train_seconds") or 0.0
        eff = d.get("effective_batch") or 0
        if steps < MIN_STEPS or secs <= 0 or eff <= 0:
            continue
        out.append({"micro": int(m.group(1)), "effective_batch": eff, "steps": steps,
                    "seconds": secs, "frames_per_s": steps * eff / secs,
                    "peak_gb": d.get("peak_mem_gb")})
    return out


def choose(fits, seconds_left, max_gb=0.0):
    """The fastest fit inside the memory budget, with the update budget for the time left.

    Returns None when nothing qualifies. `max_gb` of 0 means do not filter on memory; a fit with
    no recorded peak is kept either way, because a missing number is not evidence of a breach.
    """
    ok = [f for f in fits if not (max_gb and f["peak_gb"] and f["peak_gb"] > max_gb)]
    if not ok or seconds_left <= 0:
        return None
    best = max(ok, key=lambda f: (f["frames_per_s"], f["micro"]))
    steps = int(best["frames_per_s"] * seconds_left / best["effective_batch"])
    return {**best, "max_steps": max(1, steps), "seconds_left": seconds_left,
            "rejected": [f["micro"] for f in fits if f not in ok]}


def main(args):
    fits = read_fits(args.dir)
    pick = choose(fits, args.seconds_left, args.max_gb)
    for f in fits:
        peak = "?" if f["peak_gb"] is None else f"{f['peak_gb']:.1f}"
        print(f"# fit mb{f['micro']}: {f['frames_per_s']:.1f} frames/s, peak {peak} GB, "
              f"{f['steps']} updates in {f['seconds']:.0f}s", flush=True)
    if pick is None:
        print(f"# no usable fit under {args.max_gb} GB in {args.dir}")
        raise SystemExit(1)
    print(f"# picked mb{pick['micro']} at {pick['frames_per_s']:.1f} frames/s; "
          f"{pick['max_steps']} updates x {pick['effective_batch']} = "
          f"{pick['max_steps'] * pick['effective_batch']} presentations in {args.seconds_left}s"
          + (f"; rejected on memory: {pick['rejected']}" if pick["rejected"] else ""))
    print(f"MB={pick['micro']}; MAX_STEPS={pick['max_steps']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="directory holding the fit_mb*/ checks")
    p.add_argument("--seconds-left", type=int, required=True, help="training seconds the budget allows")
    p.add_argument("--max-gb", type=float, default=0.0, help="peak memory a fit may have used (0 = any)")
    main(p.parse_args())
