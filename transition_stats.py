"""Measure how much of a recording survives verified fixed-duration transitions.

Reads per-tic and `record_arnold.py --decision-only` recordings alike: `--repeat` is always in tics,
and a decision-only file's `stored_tic_stride` turns it into a row count.

    python transition_stats.py --in-dir raw_arnold --repeat 4 --out raw_arnold/card/transition_stats.json
"""
import argparse, glob, json, os
import numpy as np
from transitions import canonical_table, valid_transitions, chain_frames, life_segments, stored_tic_stride


def main(a):
    import pyarrow.parquet as pq
    paths = sorted(glob.glob(os.path.join(a.in_dir, "ep_*.parquet")))
    acts, btns = [], []
    per_ep = []
    stored = 1
    for p in paths:
        t = pq.read_table(p, columns=["action", "buttons", "deaths", "map_id"])
        stored = stored_tic_stride(pq.read_schema(p).metadata)
        acts.append(t["action"].to_numpy(zero_copy_only=False)); btns.append(np.array(t["buttons"].to_pylist()))
        per_ep.append((p, t["deaths"].to_numpy(zero_copy_only=False), int(t["map_id"][0].as_py())))
    canon = canonical_table(np.concatenate(acts), np.concatenate(btns))
    repeat = a.repeat // stored
    tot = dict(tics=0, lives=0, stride4_frames=0, transitions=0, transitions_unfiltered=0, chains=0, frames_to_encode=0, windows_L32=0, windows_L4=0)
    chain_lens, per_map = [], {}
    for (p, deaths, m), act, btn in zip(per_ep, acts, btns):
        n = len(act) * stored; tot["tics"] += n; tot["lives"] += len(life_segments(deaths)); tot["stride4_frames"] += (n + 3) // 4
        src, ch = valid_transitions(act, btn, deaths, repeat, canon)
        src_u, _ = valid_transitions(act, btn, deaths, repeat, None)
        tot["transitions"] += len(src); tot["transitions_unfiltered"] += len(src_u)
        rows, cid = chain_frames(src, ch, repeat)
        tot["frames_to_encode"] += len(rows)
        if len(ch):
            _, counts = np.unique(ch, return_counts=True); chain_lens += counts.tolist(); tot["chains"] += len(counts)
            # a chain of n transitions has n+1 frames and yields n-L+1 windows of L context frames plus one target
            tot["windows_L32"] += int(np.sum(np.maximum(0, counts - 32 + 1))); tot["windows_L4"] += int(np.sum(np.maximum(0, counts - 4 + 1)))
        pm = per_map.setdefault(m, dict(tics=0, transitions=0)); pm["tics"] += n; pm["transitions"] += len(src)
    tot["stored_tic_stride"] = stored
    cl = np.array(chain_lens) if chain_lens else np.zeros(1)
    out = {**tot, "transition_fraction_of_stride4": tot["transitions"] / tot["stride4_frames"],
           "override_excluded": tot["transitions_unfiltered"] - tot["transitions"],
           "chain_len_transitions": dict(mean=float(cl.mean()), median=float(np.median(cl)), p10=float(np.percentile(cl, 10)), p90=float(np.percentile(cl, 90)), max=int(cl.max())),
           "per_map": {str(k): v for k, v in sorted(per_map.items())}, "num_actions_seen": len(canon)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_map"}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--in-dir", required=True); p.add_argument("--repeat", type=int, default=4); p.add_argument("--out", required=True)
    main(p.parse_args())
