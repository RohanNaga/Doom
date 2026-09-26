#!/bin/bash
# Collect one lab-server snapshot and print the history rows; the caller writes them to the dashboard db.
S=/private/tmp/claude-501/-Users-rohan-Documents-Github-Doom--claude-worktrees-vibrant-ritchie-6f6280/bbd290c7-2c43-46b3-bfab-7994575157d8/scratchpad
cd /Users/rohan/Documents/Github/Doom/.claude/worktrees/vibrant-ritchie-6f6280 && source ~/.zshrc >/dev/null 2>&1
python3 tools/server_usage.py --out $S/usage_snapshot.json "$@" >/dev/null 2>&1 || { echo "SNAPSHOT_FAILED"; exit 1; }
python3 - <<PY
import json
d=json.load(open("$S/usage_snapshot.json")); t=d["collected_at"]
for name,sv in d["servers"].items():
    disk=(sv.get("disks") or [{}])[0]; sysm=sv.get("system") or {}
    row={"t":t,"server":name,"avail_gb":round(disk.get("avail_b",0)/1024**3,1),"load1":sysm.get("load1"),"ours_gpus":sum(1 for g in sv.get("gpus",[]) if g.get("ours_mb",0)>0)}
    json.dump(row,open(f"$S/hist_{t}_{name}.json","w")); print(f"hist_{t}_{name}.json", row)
d["ts"]=t; json.dump(d,open("$S/usage_latest_ts.json","w")); json.dump(d,open(f"$S/hist_full_{t}.json","w")); print("collected_at", t, "full", f"hist_full_{t}.json")
PY
