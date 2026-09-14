"""Reduce a server_usage.py snapshot to the small per-interval history document the dashboard charts.
    python tools/usage_compact.py snapshot.json history.json
"""
import json, sys
snap = json.load(open(sys.argv[1]))
out = {"ts": snap["collected_at"], "servers": {}}
for name, sv in snap["servers"].items():
    out["servers"][name] = {
        "gpu_util": [g["util"] for g in sv.get("gpus", [])],
        "gpu_ours_mb": [g.get("ours_mb", 0) for g in sv.get("gpus", [])],
        "gpu_others_mb": [g.get("others_mb", 0) for g in sv.get("gpus", [])],
        "disk_avail_b": {d["mount"]: d["avail_b"] for d in sv.get("disks", [])},
        "runs": {k: {f: v.get(f) for f in ("step", "loss", "val_step", "val_loss", "steps_per_s")} for k, v in sv.get("runs", {}).items()},
    }
json.dump(out, open(sys.argv[2], "w"))
print("history doc", out["ts"], {n: s["runs"] for n, s in out["servers"].items()})
