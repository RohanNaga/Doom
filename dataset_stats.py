"""
Data card statistics for a per-episode parquet recording: counts per map, action histogram,
episode lengths, HUD-state ranges, and coverage heatmaps of player position per map.

    python dataset_stats.py --in-dir raw_arnold --out-dir raw_arnold/card --buttons raw_arnold/buttons.json
Writes card/stats.json, card/DATASET_STATS.md, card/coverage_map_XX.png
"""
import argparse
import glob
import json
import os

import numpy as np


def main(args):
    import pyarrow.parquet as pq
    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.in_dir, "ep_*.parquet")))
    names = json.load(open(args.buttons))["available_buttons"] if args.buttons else None
    per_map, lengths, act_hist, pos = {}, [], {}, {}
    tot = dict(tics=0, kills=0, deaths=0, png_bytes=0)
    health_min = 999
    for p in paths:
        t = pq.read_table(p, columns=[c for c in ["map_id", "tic", "action", "health", "kills", "deaths", "pos_x", "pos_y"] if c in pq.ParquetFile(p).schema.names])
        n = t.num_rows
        m = int(t["map_id"][0].as_py()) if "map_id" in t.schema.names else 0
        per_map.setdefault(m, dict(episodes=0, tics=0, kills=0, deaths=0))
        per_map[m]["episodes"] += 1; per_map[m]["tics"] += n
        if "kills" in t.schema.names:
            per_map[m]["kills"] += int(t["kills"][-1].as_py()); per_map[m]["deaths"] += int(t["deaths"][-1].as_py())
        lengths.append(n); tot["tics"] += n
        a = np.array(t["action"]); u, c = np.unique(a, return_counts=True)
        for k, v in zip(u.tolist(), c.tolist()):
            act_hist[k] = act_hist.get(k, 0) + v
        health_min = min(health_min, int(np.array(t["health"]).min()))
        if "pos_x" in t.schema.names:
            pos.setdefault(m, []).append(np.stack([np.array(t["pos_x"]), np.array(t["pos_y"])], 1)[::args.stride])
        tot["png_bytes"] += os.path.getsize(p)
    stats = dict(episodes=len(paths), tics=tot["tics"], decision_frames=sum((l + args.stride - 1) // args.stride for l in lengths),
                 hours_of_play=tot["tics"] / 35 / 3600, bytes_per_frame=tot["png_bytes"] / max(tot["tics"], 1),
                 episode_len_tics=dict(min=int(min(lengths)), mean=float(np.mean(lengths)), max=int(max(lengths))),
                 per_map={str(k): v for k, v in sorted(per_map.items())}, action_histogram={str(k): v for k, v in sorted(act_hist.items())},
                 health_min=health_min)
    # coverage heatmaps
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        cover = {}
        for m, chunks in sorted(pos.items()):
            xy = np.concatenate(chunks)
            H, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=64)
            cover[str(m)] = float((H > 0).mean())
            plt.figure(figsize=(4, 4)); plt.imshow(np.log1p(H.T), origin="lower", cmap="magma"); plt.axis("off"); plt.title(f"map {m}: {len(xy):,} positions")
            plt.savefig(os.path.join(args.out_dir, f"coverage_map_{m:02d}.png"), dpi=100, bbox_inches="tight"); plt.close()
        stats["coverage_fraction_of_visited_cells_64x64"] = cover
    except Exception as e:
        stats["coverage_error"] = str(e)
    json.dump(stats, open(os.path.join(args.out_dir, "stats.json"), "w"), indent=1)
    lines = [f"# Dataset statistics", "", f"- episodes: {stats['episodes']}", f"- tics: {stats['tics']:,} ({stats['hours_of_play']:.1f} hours of play at 35 tics/s)",
             f"- decision frames (stride {args.stride}): {stats['decision_frames']:,}", f"- bytes per frame (PNG, in parquet): {stats['bytes_per_frame']:.0f}",
             f"- episode length in tics: min {stats['episode_len_tics']['min']}, mean {stats['episode_len_tics']['mean']:.0f}, max {stats['episode_len_tics']['max']}", "",
             "| map | episodes | tics | kills | deaths |", "|---|---|---|---|---|"]
    lines += [f"| {m} | {v['episodes']} | {v['tics']:,} | {v['kills']} | {v['deaths']} |" for m, v in stats["per_map"].items()]
    lines += ["", "Action histogram (id: count):", "", ", ".join(f"{k}: {v:,}" for k, v in stats["action_histogram"].items())]
    open(os.path.join(args.out_dir, "DATASET_STATS.md"), "w").write("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k not in ("per_map", "action_histogram")}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True); p.add_argument("--out-dir", required=True)
    p.add_argument("--buttons", default=""); p.add_argument("--stride", type=int, default=4)
    main(p.parse_args())
