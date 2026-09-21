"""Is the control on row t the one that produced the motion between frames t and t+1?

A one-tic error in the action conditioning is invisible in every loss curve and every PSNR number:
the model simply learns a slightly blurrier transition. This script answers the question without a
model, from the recorded state alone, so an off-by-one is caught before a multi-day run rather than
after it.

**The test.** `record_arnold.py` stores the control of row t as the one applied from t to t+1, so
the yaw change between frames t and t+1 must be explained by the TURN bits of row t. For every row
we know the recorded `angle`, so we know the realised yaw delta; the control tells us the intended
turn direction. Score a shift s by the fraction of turning rows whose yaw delta has the sign the
control at row t+s predicts, over the rows where the intended direction actually *changes* (the
boundaries), because on a long constant turn every shift looks equally good.

**The negative controls.** Shifts -1 and +1 are scored alongside 0 and all three are printed. A
weak signal shows all three near chance together; an off-by-one shows one neighbour beating 0. That
distinction is the whole point of running this, so the script never reports 0 alone.

Position is scored the same way, from the MOVE_FORWARD/BACKWARD bits against the displacement
projected onto the facing direction.

    python check_action_alignment.py --parquet /sata2/.../raw_arnold_dense/arenas/ep_00000.parquet
    python check_action_alignment.py --latents-dir /sata2/.../latents_arnold_dense_pertic/arenas --episodes 4
"""
import argparse
import json

import numpy as np

# Indices into the recorder's `buttons` string, which is the game's `available_buttons` order.
# `record_arnold.py` builds it from `--action_combinations "move_fb+move_lr;turn_lr;attack"` with
# speed and crouch appended, and `transitions.CONTROL_BITS = 9` names the first nine:
#   0 MOVE_FORWARD 1 MOVE_BACKWARD 2 TURN_LEFT 3 TURN_RIGHT 4 MOVE_LEFT 5 MOVE_RIGHT
#   6 ATTACK 7 SPEED 8 CROUCH
TURN_LEFT, TURN_RIGHT = 2, 3
MOVE_FORWARD, MOVE_BACKWARD = 0, 1
SHIFTS = (-1, 0, 1)


def wrap_deg(d):
    """Signed yaw difference in (-180, 180]; ViZDoom's `angle` is degrees and wraps at 360."""
    return (np.asarray(d, dtype=np.float64) + 180.0) % 360.0 - 180.0


def intended_turn(controls):
    """+1 for a left turn, -1 for a right turn, 0 for neither, per row.

    ViZDoom's yaw increases counter-clockwise, so TURN_LEFT raises `angle`.
    """
    return controls[:, TURN_LEFT].astype(int) - controls[:, TURN_RIGHT].astype(int)


def intended_move(controls):
    """+1 forward, -1 backward, 0 for neither, per row."""
    return controls[:, MOVE_FORWARD].astype(int) - controls[:, MOVE_BACKWARD].astype(int)


def boundaries(intent):
    """Rows whose intended direction differs from EITHER neighbour's.

    On a run of identical controls every shift explains the motion equally well, so only the rows
    next to a change carry information about alignment. Both neighbours matter, and taking only one
    of them is a trap: on a 4-tic action repeat, the rows where the intent differs from the previous
    row are the first tic of each run, and there `intent[t+1] == intent[t]`, so shift +1 scores a
    perfect 1.0 alongside shift 0 and an off-by-one in that direction is invisible. The union makes
    shift 0 the only one that can score 1.0.
    """
    n = len(intent)
    prev = np.empty(n, dtype=bool); prev[0] = True; prev[1:] = intent[1:] != intent[:-1]
    nxt = np.empty(n, dtype=bool); nxt[-1] = True; nxt[:-1] = intent[:-1] != intent[1:]
    return prev | nxt


def score_shift(intent, realised, usable, shift):
    """Fraction of usable boundary rows where the shifted control's sign matches the realised sign.

    `realised[t]` is the change between frames t and t+1. Shift s scores the hypothesis
    "row t+s carries the control that produced realised[t]", so s = 0 is our recorder's convention
    and s = +1 is the open reproduction's (record after stepping).
    """
    n = len(intent)
    idx = np.flatnonzero(usable & boundaries(intent))
    idx = idx[(idx + shift >= 0) & (idx + shift < n)]
    want = intent[idx + shift]
    keep = want != 0
    idx, want = idx[keep], want[keep]
    if len(idx) == 0:
        return {"rows": 0, "accuracy": float("nan")}
    got = np.sign(realised[idx])
    return {"rows": int(len(idx)), "accuracy": float(np.mean(got == want))}


def alignment_scores(controls, angle, pos_x=None, pos_y=None, min_yaw=1.0, min_move=1.0):
    """{"yaw": {shift: score}, "position": {...}} for one episode's recorded state.

    `min_yaw` and `min_move` drop rows whose realised change is too small to have a reliable sign
    (a blocked turn against a wall, a frame where the agent is stuck). They are thresholds on the
    OBSERVED motion only, so they cannot favour one shift over another.
    """
    controls = np.asarray(controls, dtype=np.float32)
    angle = np.asarray(angle, dtype=np.float64)
    n = len(angle)
    d_yaw = np.zeros(n)
    d_yaw[:-1] = wrap_deg(angle[1:] - angle[:-1])
    yaw_usable = np.abs(d_yaw) >= min_yaw
    yaw_usable[-1] = False
    out = {"yaw": {str(s): score_shift(intended_turn(controls), d_yaw, yaw_usable, s) for s in SHIFTS}}
    if pos_x is not None and pos_y is not None:
        px, py = np.asarray(pos_x, dtype=np.float64), np.asarray(pos_y, dtype=np.float64)
        rad = np.deg2rad(angle)
        along = np.zeros(n)
        # displacement projected onto the facing direction of the frame the motion starts from
        along[:-1] = (px[1:] - px[:-1]) * np.cos(rad[:-1]) + (py[1:] - py[:-1]) * np.sin(rad[:-1])
        move_usable = np.abs(along) >= min_move
        move_usable[-1] = False
        out["position"] = {str(s): score_shift(intended_move(controls), along, move_usable, s) for s in SHIFTS}
    return out


def verdict(scores, margin=0.05):
    """Which shift wins, and whether the win is big enough to call the alignment confirmed.

    Three outcomes, and the script prints whichever applies:
      * shift 0 wins by at least `margin`      -> the conditioning row is right
      * a neighbour wins by at least `margin`  -> an off-by-one of that sign
      * nothing wins by `margin`               -> the signal is too weak to decide, not a pass
    """
    out = {}
    for key, per_shift in scores.items():
        acc = {s: v["accuracy"] for s, v in per_shift.items() if v["rows"]}
        if not acc:
            out[key] = {"best": None, "verdict": "no usable rows"}
            continue
        best = max(acc, key=acc.get)
        second = max((a for s, a in acc.items() if s != best), default=float("-inf"))
        clear = acc[best] - second >= margin
        out[key] = {"best": best, "accuracy": acc[best], "margin": acc[best] - second,
                    "verdict": ("aligned at shift 0" if best == "0" and clear else
                                f"OFF BY {best} TICS" if clear else
                                "inconclusive: no shift wins by the margin")}
    return out


def from_parquet(path, min_yaw=1.0, min_move=1.0):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["buttons", "angle", "pos_x", "pos_y"])
    from doom_data import control_matrix
    return alignment_scores(control_matrix(np.array(t["buttons"].to_pylist())),
                            t["angle"].to_numpy(zero_copy_only=False),
                            t["pos_x"].to_numpy(zero_copy_only=False),
                            t["pos_y"].to_numpy(zero_copy_only=False), min_yaw, min_move)


def from_latents(latents_dir, episodes=1, min_yaw=1.0, min_move=1.0):
    """The same check off the encoder's per-tic `.npz` sidecars, so it runs where the latents are."""
    from doom_data import control_matrix, list_latent_episodes
    per = []
    for ep, _, meta_path in list_latent_episodes(latents_dir)[:episodes]:
        m = np.load(meta_path)
        per.append((ep, alignment_scores(control_matrix(m["buttons"]), m["angle"],
                                         m.get("pos_x"), m.get("pos_y"), min_yaw, min_move)))
    return per


def pool(per_episode):
    """Row-weighted mean accuracy per shift over several episodes."""
    out = {}
    for _, sc in per_episode:
        for key, per_shift in sc.items():
            for s, v in per_shift.items():
                if v["rows"]:
                    a = out.setdefault(key, {}).setdefault(s, [0, 0.0])
                    a[0] += v["rows"]; a[1] += v["rows"] * v["accuracy"]
    return {k: {s: {"rows": n, "accuracy": tot / n} for s, (n, tot) in v.items()} for k, v in out.items()}


def main(args):
    if args.parquet:
        scores = from_parquet(args.parquet, args.min_yaw, args.min_move)
    else:
        scores = pool(from_latents(args.latents_dir, args.episodes, args.min_yaw, args.min_move))
    print(json.dumps({"scores": scores, "verdict": verdict(scores, args.margin)}, indent=1))
    bad = [k for k, v in verdict(scores, args.margin).items() if v.get("best") not in (None, "0")]
    return 1 if bad else 0


def build_parser():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--parquet", help="one ep_XXXXX.parquet recording")
    g.add_argument("--latents-dir", help="a per-tic latent directory (reads the .npz sidecars)")
    p.add_argument("--episodes", type=int, default=1, help="episodes to pool over with --latents-dir")
    p.add_argument("--min-yaw", type=float, default=1.0, help="degrees of realised yaw change a row needs to count")
    p.add_argument("--min-move", type=float, default=1.0, help="map units of realised displacement a row needs to count")
    p.add_argument("--margin", type=float, default=0.05, help="accuracy lead a shift needs before the verdict is called")
    return p


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
