"""Is the control on row t the one that produced the motion between frames t and t+1?

A one-tic error in the action conditioning is invisible in every loss curve and every PSNR number:
the model simply learns a slightly blurrier transition. This script answers the question without a
model, from the recorded state alone, so an off-by-one is caught before a multi-day run rather than
after it. It is a launch GATE: it exits non-zero unless the alignment is positively confirmed.

**Sign convention.** Shift s scores the hypothesis "the control that produced the motion between
frames t and t+1 is stored on row t+s". Our recorder stores the row and *then* steps the engine
(`record_arnold.py:226` then `:265`), so s = 0 is our convention and the gate must pick it. The open
GameNGen reproduction steps and then records, i.e. s = +1; a corpus built under that convention, or
a stale control buffer that lags one tic, shows up as a neighbour winning.

**What made a naive version of this useless** (found by Astra, Sep 20 2026, and both bugs are now
regression-tested):

* Dropping the rows where the *shifted* control is a no-op scores each shift on a different row set.
  Arnold's action set produces 4-tic holds that cycle no-turn, left, no-turn, right, and under that
  pattern all three shifts scored a perfect 1.000. No-turn is therefore a CLASS, not a skipped row,
  every shift is scored on one identical row set, and the metric is balanced accuracy over the three
  classes so the no-turn majority cannot carry it.
* "The best shift wins by a margin" is not a pass. A stale-control injection scored 1.000 at shift 0
  on 24 rows and 1.000 at shift +1 on 48 rows, and the verdict came back "inconclusive" with exit
  code 0. A pass now needs all three shifts scored on the same rows, a row-count floor, an absolute
  accuracy floor, a margin over BOTH neighbours, and a bootstrap interval on that margin that
  excludes zero.

Rows are scored only inside one continuous life at unit tic spacing: a respawn teleports the agent,
so a window across it would be scored against motion no control produced.

    python check_action_alignment.py --latents-dir /sata2/.../latents_arnold_dense_pertic_eval/val --episodes 8
    python check_action_alignment.py --parquet /sata2/.../raw_arnold_dense/arenas/ep_00000.parquet
    python check_action_alignment.py --latents-dir ... --audit-parquet-dir /sata2/.../raw_arnold_dense/arenas

Exit codes: 0 aligned at shift 0, 2 misaligned (a neighbour wins), 3 inconclusive (not enough
evidence to certify either way -- which is a failure, not a pass).
"""
import argparse
import json

import numpy as np

# Indices into the recorder's `buttons` string, which is the game's `available_buttons` order
# (`docs/cards/arnold/buttons.json`). `transitions.CONTROL_BITS = 9` names the first nine:
#   0 MOVE_FORWARD 1 MOVE_BACKWARD 2 TURN_LEFT 3 TURN_RIGHT 4 MOVE_LEFT 5 MOVE_RIGHT
#   6 ATTACK 7 SPEED 8 CROUCH
TURN_LEFT, TURN_RIGHT = 2, 3
MOVE_FORWARD, MOVE_BACKWARD = 0, 1
SHIFTS = (-1, 0, 1)
CLASSES = (-1, 0, 1)

EXIT_ALIGNED, EXIT_MISALIGNED, EXIT_INCONCLUSIVE = 0, 2, 3


def wrap_deg(d):
    """Signed yaw difference in (-180, 180]; ViZDoom's `angle` is degrees and wraps at 360."""
    return (np.asarray(d, dtype=np.float64) + 180.0) % 360.0 - 180.0


def control_class(controls, plus, minus):
    """Per row: +1 when the `plus` bit is held alone, -1 for `minus` alone, 0 for neither or both.

    Zero is a class of its own -- "the agent asked for no turn" is a prediction the physics can
    confirm or refute -- and not a row to throw away.
    """
    p = controls[:, plus].astype(int)
    m = controls[:, minus].astype(int)
    return p - m


def realised_class(delta, deadband):
    """Per row: the observed motion class, with a deadband for changes too small to have a sign.

    Derived from the recorded state ALONE, so the row set and the labels are the same for every
    shift. That is what makes the three scores comparable.
    """
    out = np.zeros(len(delta), dtype=int)
    out[delta >= deadband] = 1
    out[delta <= -deadband] = -1
    return out


def usable_rows(tic, deaths, n):
    """Rows t where t-1, t, t+1 all exist, are one tic apart, and lie in one continuous life.

    All three shifts need a defined control, so a row is usable only if its neighbours are too;
    scoring the shifts on different row sets is the bug this function exists to prevent. `deaths`
    increments on the respawn row (`record_arnold.py`), which is the life boundary
    `transitions.life_segments` uses.
    """
    ok = np.zeros(n, dtype=bool)
    if n < 3:
        return ok
    tic = np.asarray(tic, dtype=np.int64)
    deaths = np.zeros(n, dtype=np.int64) if deaths is None else np.asarray(deaths, dtype=np.int64)
    step = np.diff(tic) == 1
    same_life = np.diff(deaths) == 0
    good = step & same_life                       # edge i joins rows i and i+1
    ok[1:-1] = good[:-1] & good[1:]
    return ok


def boundaries(cls, usable):
    """Usable rows whose observed motion class differs from a neighbour's.

    On a run of identical motion every shift explains the frames equally well, so only the rows
    around a change carry information. Computed from the OBSERVED class, so it does not depend on
    which shift is being tested.
    """
    n = len(cls)
    diff_prev = np.zeros(n, dtype=bool); diff_prev[1:] = cls[1:] != cls[:-1]
    diff_next = np.zeros(n, dtype=bool); diff_next[:-1] = cls[:-1] != cls[1:]
    return usable & (diff_prev | diff_next)


def balanced_accuracy(truth, pred):
    """Mean per-class recall over the classes actually present in `truth`.

    Plain accuracy would be carried by whichever class dominates the boundary rows; the question
    here is whether the control explains every kind of motion, including "no turn".
    """
    recalls = []
    for c in CLASSES:
        m = truth == c
        if m.any():
            recalls.append(float((pred[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def score_axis(controls, delta, tic, deaths, plus, minus, deadband):
    """{"rows": n, "per_class": {...}, "shifts": {s: balanced accuracy}} on ONE shared row set.

    The per-shift predictions are gathered here, so everything downstream works on aligned arrays
    of the same length and cannot shift a second time.
    """
    n = len(delta)
    truth = realised_class(delta, deadband)
    rows = np.flatnonzero(boundaries(truth, usable_rows(tic, deaths, n)))
    cls = control_class(controls, plus, minus)
    t = truth[rows]
    pred = {s: cls[rows + s] for s in SHIFTS}
    return {"rows": int(len(rows)),
            "per_class": {str(c): int((t == c).sum()) for c in CLASSES} if len(rows) else {},
            "shifts": {str(s): (balanced_accuracy(t, pred[s]) if len(rows) else float("nan")) for s in SHIFTS},
            "_truth": t, "_pred": pred}


def bootstrap_margin_low(score, alpha=0.05, draws=1000, seed=0):
    """Lower end of a percentile bootstrap interval on (shift 0) minus (the better neighbour).

    Resampling the scored ROWS, because they are the unit of evidence here. A point margin can come
    from a handful of rows; requiring the interval to exclude zero is what makes the gate refuse a
    lucky one.
    """
    truth, pred = score["_truth"], score["_pred"]
    if len(truth) < 2:
        return float("nan")
    rng = np.random.RandomState(seed)
    m = np.empty(draws)
    for i in range(draws):
        r = rng.randint(0, len(truth), len(truth))
        b = {s: balanced_accuracy(truth[r], pred[s][r]) for s in SHIFTS}
        m[i] = b[0] - max(b[-1], b[1])
    return float(np.percentile(m[np.isfinite(m)], 100 * alpha)) if np.isfinite(m).any() else float("nan")


def verdict(score, min_rows=1000, min_accuracy=0.95, margin=0.20, alpha=0.05, draws=1000):
    """Aligned, misaligned or inconclusive, with the reason. Every condition must hold to pass."""
    acc = {int(s): a for s, a in score["shifts"].items()}
    if any(not np.isfinite(a) for a in acc.values()):
        return {"verdict": "inconclusive", "reason": "not every shift could be scored", "best": None,
                "rows": score["rows"]}
    if score["rows"] < min_rows:
        return {"verdict": "inconclusive", "best": int(max(acc, key=acc.get)), "rows": score["rows"],
                "reason": f"{score['rows']} scored rows is below the floor of {min_rows}"}
    best = int(max(acc, key=acc.get))
    neighbour = max(acc[-1], acc[1])
    lead = acc[0] - neighbour
    low = bootstrap_margin_low(score, alpha, draws)
    common = {"best": best, "rows": score["rows"], "accuracy": acc, "lead_over_neighbours": lead,
              "bootstrap_margin_low": low}
    if best != 0 and acc[best] - acc[0] >= margin:
        return {**common, "verdict": "misaligned",
                "reason": f"shift {best:+d} beats shift 0 by {acc[best] - acc[0]:.3f}; the control that "
                          f"produced the motion is stored {abs(best)} row(s) "
                          f"{'later' if best > 0 else 'earlier'} than this code assumes"}
    if acc[0] < min_accuracy:
        return {**common, "verdict": "inconclusive",
                "reason": f"shift 0 balanced accuracy {acc[0]:.3f} is below the floor of {min_accuracy}"}
    if lead < margin:
        return {**common, "verdict": "inconclusive",
                "reason": f"shift 0 leads its neighbours by only {lead:.3f}, below the required {margin}"}
    if not (np.isfinite(low) and low > 0):
        return {**common, "verdict": "inconclusive",
                "reason": f"the bootstrap interval on the margin includes zero (low end {low:.3f})"}
    return {**common, "verdict": "aligned", "reason": "shift 0 wins on every criterion"}


def alignment_scores(controls, angle, tic=None, deaths=None, pos_x=None, pos_y=None,
                     min_yaw=1.0, min_move=1.0):
    """{"yaw": score, "position": score} for one episode's recorded state."""
    controls = np.asarray(controls, dtype=np.float32)
    angle = np.asarray(angle, dtype=np.float64)
    n = len(angle)
    tic = np.arange(n, dtype=np.int64) if tic is None else np.asarray(tic, dtype=np.int64)
    d_yaw = np.zeros(n)
    d_yaw[:-1] = wrap_deg(angle[1:] - angle[:-1])
    out = {"yaw": score_axis(controls, d_yaw, tic, deaths, TURN_LEFT, TURN_RIGHT, min_yaw)}
    if pos_x is not None and pos_y is not None:
        px, py = np.asarray(pos_x, dtype=np.float64), np.asarray(pos_y, dtype=np.float64)
        rad = np.deg2rad(angle)
        along = np.zeros(n)
        # displacement projected onto the facing direction of the frame the motion starts from
        along[:-1] = (px[1:] - px[:-1]) * np.cos(rad[:-1]) + (py[1:] - py[:-1]) * np.sin(rad[:-1])
        out["position"] = score_axis(controls, along, tic, deaths, MOVE_FORWARD, MOVE_BACKWARD, min_move)
    return out


def merge(per_episode):
    """Pool several episodes' scores by concatenating their row sets, not by averaging accuracies."""
    keys = set()
    for _, sc in per_episode:
        keys |= set(sc)
    out = {}
    for key in sorted(keys):
        truth, preds = [], {s: [] for s in SHIFTS}
        for _, sc in per_episode:
            if key not in sc:
                continue
            truth.append(sc[key]["_truth"])
            for sh in SHIFTS:
                preds[sh].append(sc[key]["_pred"][sh])
        if not truth:
            continue
        t = np.concatenate(truth)
        p = {sh: np.concatenate(preds[sh]) for sh in SHIFTS}
        out[key] = {"rows": int(len(t)), "per_class": {str(c): int((t == c).sum()) for c in CLASSES},
                    "shifts": {str(sh): balanced_accuracy(t, p[sh]) for sh in SHIFTS},
                    "_truth": t, "_pred": p}
    return out


def public(scores):
    """The scores without the internal arrays, for printing."""
    return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in scores.items()}


def from_parquet(path, min_yaw=1.0, min_move=1.0):
    import pyarrow.parquet as pq
    from doom_data import control_matrix
    cols = [c for c in ("buttons", "angle", "pos_x", "pos_y", "tic", "deaths")
            if c in pq.read_schema(path).names]
    t = pq.read_table(path, columns=cols)

    def col(name):
        return t[name].to_numpy(zero_copy_only=False) if name in cols else None
    return alignment_scores(control_matrix(np.array(t["buttons"].to_pylist())), col("angle"),
                            col("tic"), col("deaths"), col("pos_x"), col("pos_y"), min_yaw, min_move)


def from_latents(latents_dir, episodes=1, min_yaw=1.0, min_move=1.0):
    """The same check off the encoder's per-tic `.npz` sidecars, so it runs where the latents are."""
    from doom_data import control_matrix, list_latent_episodes
    per = []
    for ep, _, meta_path in list_latent_episodes(latents_dir)[:episodes]:
        m = np.load(meta_path)

        def col(name):
            return m[name] if name in m.files else None
        per.append((ep, alignment_scores(control_matrix(m["buttons"]), m["angle"], col("tic"),
                                         col("deaths"), col("pos_x"), col("pos_y"), min_yaw, min_move)))
    return per


def audit_sidecar(latents_dir, parquet_dir, episodes=4, rows=2000, seed=0):
    """Do the sidecar's button vectors match the raw recording's, row for row, by TIC?

    The conditioning is read from the `.npz` the encoder wrote, so the pre-launch question is not
    only "is the convention right" but "is this file the same data as the recording". Also counts
    how many sampled rows are anti-stuck overrides -- rows whose executed vector is not the
    canonical vector of the requested action id -- because those are exactly the rows on which
    conditioning on the action id instead of the button vector would be wrong.
    """
    import pyarrow.parquet as pq
    from doom_data import list_latent_episodes
    canon = None
    path = f"{latents_dir}/canonical_controls.json"
    try:
        with open(path) as f:
            canon = {int(k): v for k, v in json.load(f).items()}
    except OSError:
        pass
    rng = np.random.RandomState(seed)
    checked = mismatch = overrides = 0
    problems = []
    for ep, _, meta_path in list_latent_episodes(latents_dir)[:episodes]:
        m = np.load(meta_path)
        src = f"{parquet_dir}/ep_{ep:05d}.parquet"
        t = pq.read_table(src, columns=["tic", "buttons", "action"])
        raw_tic = t["tic"].to_numpy(zero_copy_only=False)
        raw_btn = np.array(t["buttons"].to_pylist())
        by_tic = {int(x): i for i, x in enumerate(raw_tic)}
        side_tic = np.asarray(m["tic"]).astype(np.int64)
        take = rng.choice(len(side_tic), size=min(rows, len(side_tic)), replace=False)
        for i in take:
            tic = int(side_tic[i])
            if tic not in by_tic:
                problems.append(f"ep {ep} tic {tic} is in the sidecar but not in {src}")
                mismatch += 1
                continue
            a, b = str(m["buttons"][i]), str(raw_btn[by_tic[tic]])
            checked += 1
            if a != b:
                mismatch += 1
                if len(problems) < 8:
                    problems.append(f"ep {ep} tic {tic}: sidecar {a!r} vs recording {b!r}")
            if canon is not None:
                want = canon.get(int(m["action"][i]))
                if want is not None and not a.startswith(want):
                    overrides += 1
    return {"episodes": min(episodes, len(list_latent_episodes(latents_dir))), "rows_checked": checked,
            "mismatches": mismatch, "anti_stuck_override_rows": overrides,
            "override_fraction": overrides / checked if checked else 0.0,
            "canonical_table": canon is not None, "problems": problems, "ok": mismatch == 0}


def main(args):
    if args.parquet:
        scores = from_parquet(args.parquet, args.min_yaw, args.min_move)
    else:
        scores = merge(from_latents(args.latents_dir, args.episodes, args.min_yaw, args.min_move))
    verdicts = {k: verdict(v, args.min_rows, args.min_accuracy, args.margin, draws=args.bootstrap)
                for k, v in scores.items()}
    report = {"sign_convention": "shift s means the control that produced the motion from frame t to "
                                 "t+1 is stored on row t+s; our recorder is s = 0, the open GameNGen "
                                 "reproduction is s = +1",
              "scores": public(scores), "verdict": verdicts}
    if args.audit_parquet_dir:
        report["sidecar_audit"] = audit_sidecar(args.latents_dir, args.audit_parquet_dir,
                                                args.episodes, args.audit_rows, args.seed)
    print(json.dumps(report, indent=1, default=float))
    if report.get("sidecar_audit") and not report["sidecar_audit"]["ok"]:
        return EXIT_MISALIGNED
    if any(v["verdict"] == "misaligned" for v in verdicts.values()):
        return EXIT_MISALIGNED
    if any(v["verdict"] != "aligned" for v in verdicts.values()):
        return EXIT_INCONCLUSIVE
    return EXIT_ALIGNED


def build_parser():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--parquet", help="one ep_XXXXX.parquet recording")
    g.add_argument("--latents-dir", help="a per-tic latent directory (reads the .npz sidecars)")
    p.add_argument("--episodes", type=int, default=1, help="episodes to pool over with --latents-dir")
    p.add_argument("--min-yaw", type=float, default=1.0, help="degrees of yaw change a row needs to count as a turn")
    p.add_argument("--min-move", type=float, default=1.0, help="map units of displacement a row needs to count as a move")
    p.add_argument("--min-rows", type=int, default=1000, help="scored boundary rows required before a verdict is called")
    p.add_argument("--min-accuracy", type=float, default=0.95, help="balanced accuracy shift 0 must reach")
    p.add_argument("--margin", type=float, default=0.20, help="balanced-accuracy lead shift 0 needs over BOTH neighbours")
    p.add_argument("--bootstrap", type=int, default=1000, help="bootstrap draws for the interval on that lead")
    p.add_argument("--audit-parquet-dir", default="", help="also check the sidecar's buttons against these recordings")
    p.add_argument("--audit-rows", type=int, default=2000, help="rows per episode to check in the sidecar audit")
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
