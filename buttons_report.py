"""What the recorded `buttons` strings say about Arnold's weapon-select requests, per episode.

The corpus stores Arnold's REQUESTED control list, not the engine's. On about 95% of rows it is the
nine movement and attack entries; the rest carry a weapon-select press, which sits at index
`9 + 10*k + j` for weapon j after k `Game.start()` calls in that recorder process. Arnold's
`add_buttons` re-appends the ten `SELECT_WEAPON%i` names to the SHARED `available_buttons` list on
every start (`src/doom/actions.py:197-199`, `game.py:485`) while ViZDoom deduplicates its own list,
so the list Arnold serialises grows without the engine's growing. `record_arnold.py:167` calls
`game.start(...)` once per recorded episode, so k is the worker's running episode count and only its
FIRST episode (k = 0) can put a switch inside the engine's 19 buttons.

This report measures that claim rather than asserting it:

  * `raw_max_width` and `raw_widths_over_control_bits` are the widths actually in the file;
  * `executed_switch_rows` / `unexecuted_switch_rows` split the switch presses at index 18, which is
    where `ViZDoomGame::setAction` stops reading (`src/lib/ViZDoomGame.cpp:151-158`);
  * `inferred_starts` is the k those widths imply;
  * `within_episode_growth` is True when ONE episode holds two tail widths. The explanation says the
    list grows per `Game.start()`, so that must be false on every episode; if it is true anywhere,
    the explanation is wrong and the row is printed rather than averaged away.

It reads either the raw recordings or, with `--sidecars`, the encoder's `.npz` provenance columns
(`buttons_raw_len`, `switch_requested_index`), which survive the normalisation.

    python buttons_report.py --dir /sata2/.../raw_arnold_dense/arenas --episodes 40
    python buttons_report.py --dir /sata2/.../latents_arnold_dense_pertic/arenas --sidecars
"""
import argparse
import glob
import json
import os

import numpy as np

from transitions import CONTROL_BITS, EXECUTED_BUTTONS, button_width_report, inferred_starts


def _episode_id(path, suffix):
    name = os.path.basename(path)
    try:
        return int(name[len("ep_"):-len(suffix)])
    except ValueError:
        return -1


def from_parquet(path):
    """One episode's width report, read from the raw `buttons` column."""
    import pyarrow.parquet as pq
    raw = pq.read_table(path, columns=["buttons"])["buttons"].to_pylist()
    r = button_width_report(raw)
    r["episode"] = _episode_id(path, ".parquet")
    return r


def from_sidecar(path):
    """The same report rebuilt from the sidecar's provenance columns, with no raw strings needed.

    `buttons_raw_len` and `switch_requested_index` are exactly what the width report reads, so the
    numbers are identical to the ones the raw recording would give.
    """
    with np.load(path) as z:
        if not all(c in z.files for c in ("buttons_raw_len", "switch_requested_index")):
            raise ValueError(f"{path}: no buttons_raw_len / switch_requested_index columns; this sidecar "
                             "predates the executed-control normalisation, so the raw widths are gone")
        lens = np.asarray(z["buttons_raw_len"]).astype(np.int64)
        idx = np.asarray(z["switch_requested_index"]).astype(np.int64)
    over = int((idx >= EXECUTED_BUTTONS).sum())
    tails = sorted({int(n) for n in lens if n > CONTROL_BITS})
    starts = sorted({inferred_starts(n) for n in tails})
    widest = int(lens.max()) if len(lens) else 0
    return {"episode": _episode_id(path, "_meta.npz"), "rows": int(len(lens)),
            "width": EXECUTED_BUTTONS, "raw_max_width": widest, "raw_min_width": int(lens.min()) if len(lens) else 0,
            "rows_over_executed": over, "fraction_over_executed": over / len(lens) if len(lens) else 0.0,
            "executed_switch_rows": int(((idx >= 0) & (idx < EXECUTED_BUTTONS)).sum()),
            "unexecuted_switch_rows": over, "raw_widths_over_control_bits": tails,
            "inferred_starts_seen": starts, "within_episode_growth": len(starts) > 1,
            "inferred_starts": inferred_starts(widest)}


def report(directory, episodes=0, sidecars=False):
    """Per-episode width reports plus the corpus roll-up, in filename order."""
    pattern, reader = ("ep_*_meta.npz", from_sidecar) if sidecars else ("ep_*.parquet", from_parquet)
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if episodes:
        paths = paths[:int(episodes)]
    per = [reader(p) for p in paths]
    starts = [e["inferred_starts"] for e in per if e["inferred_starts"] is not None]
    corpus = {"episodes": len(per), "source": "sidecars" if sidecars else "recordings",
              "rows": sum(e["rows"] for e in per),
              "raw_max_width": max((e["raw_max_width"] for e in per), default=0),
              "rows_over_executed": sum(e["rows_over_executed"] for e in per),
              "executed_switch_rows": sum(e["executed_switch_rows"] for e in per),
              "unexecuted_switch_rows": sum(e["unexecuted_switch_rows"] for e in per),
              "episodes_with_within_episode_growth": sum(1 for e in per if e["within_episode_growth"]),
              "max_inferred_starts": max(starts) if starts else None}
    corpus["fraction_over_executed"] = (corpus["rows_over_executed"] / corpus["rows"]
                                        if corpus["rows"] else 0.0)
    return {"directory": directory, "episodes": per, "corpus": corpus}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", required=True, help="a raw recording directory, or a latents directory with --sidecars")
    p.add_argument("--episodes", type=int, default=0, help="report only the first N episodes (0 = every one)")
    p.add_argument("--sidecars", action="store_true",
                   help="read the encoder's .npz provenance columns instead of the raw parquet")
    return p


def main(args):
    r = report(args.dir, args.episodes, args.sidecars)
    print(json.dumps(r, indent=1, default=float))
    g = r["corpus"]
    if g["episodes_with_within_episode_growth"]:
        print(f"WARNING: {g['episodes_with_within_episode_growth']} episode(s) change their button-list width "
              "inside one episode, which the per-Game.start() explanation does not allow", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
