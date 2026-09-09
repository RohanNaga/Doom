"""
Audit the action labels of a per-tic recording (record_arnold.py output).

Reports (a) decisions whose executed button vector differs from the modal vector for the requested
action id (the anti-stuck override), (b) decision boundaries off the tic%stride grid (after a death
cut a decision short), and (c) stride-4 transitions that contain more than one action or a death.
    python audit_actions.py --in-dir raw_arnold --stride 4 --out raw_arnold/card/action_audit.json
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np


def main(args):
    import pyarrow.parquet as pq
    paths = sorted(glob.glob(os.path.join(args.in_dir, "ep_*.parquet")))
    modal = defaultdict(Counter)
    rows = []
    for p in paths:
        t = pq.read_table(p, columns=["tic", "action", "buttons", "deaths"])
        a = np.array(t["action"]); b = np.array(t["buttons"].to_pylist()); tic = np.array(t["tic"]); d = np.array(t["deaths"])
        rows.append((a, b, tic, d))
        # decision starts: where (action, buttons) changes or tic == 0
        for x, y in zip(a.tolist(), b.tolist()):
            modal[x][y] += 1
    canonical = {x: c.most_common(1)[0][0] for x, c in modal.items()}
    tot = dict(tics=0, decisions=0, override_tics=0, override_decisions=0, offgrid_decisions=0,
               stride_windows=0, mixed_action_windows=0, death_windows=0)
    for a, b, tic, d in rows:
        n = len(a); tot["tics"] += n
        change = np.r_[True, (a[1:] != a[:-1]) | (b[1:] != b[:-1])]
        starts = np.flatnonzero(change)
        tot["decisions"] += len(starts)
        tot["offgrid_decisions"] += int(np.sum(tic[starts] % args.stride != 0))
        ov = np.array([b[i] != canonical[int(a[i])] for i in range(n)])
        tot["override_tics"] += int(ov.sum()); tot["override_decisions"] += int(ov[starts].sum())
        # stride windows: frame k*stride -> (k+1)*stride
        for k in range(0, n - args.stride, args.stride):
            tot["stride_windows"] += 1
            seg_a = a[k:k + args.stride]; seg_b = b[k:k + args.stride]
            if len(set(seg_a.tolist())) > 1 or len(set(seg_b.tolist())) > 1:
                tot["mixed_action_windows"] += 1
            if d[k + args.stride] != d[k]:
                tot["death_windows"] += 1
    out = {**tot, "override_tic_fraction": tot["override_tics"] / tot["tics"], "override_decision_fraction": tot["override_decisions"] / tot["decisions"],
           "offgrid_decision_fraction": tot["offgrid_decisions"] / tot["decisions"], "mixed_action_window_fraction": tot["mixed_action_windows"] / tot["stride_windows"],
           "death_window_fraction": tot["death_windows"] / tot["stride_windows"], "canonical_buttons": canonical}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items() if k != "canonical_buttons"}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--in-dir", required=True); p.add_argument("--stride", type=int, default=4); p.add_argument("--out", required=True)
    main(p.parse_args())
