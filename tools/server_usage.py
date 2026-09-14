"""Collect one usage snapshot from the lab servers for the live dashboard.

    python tools/server_usage.py --out snapshot.json [--du]

Reads GPUs (per-card memory, utilisation, and the processes on each with their owner), disks, our
directory sizes (only with --du; slow), tmux sessions, and the DoomDiT training logs. Never stores
the password: it reads PERSEVE_SERVER_PASSWORD from the environment exactly like the ssh helpers.
"""
import argparse
import json
import os
import subprocess
import time

SERVERS = {
    "superman": dict(host="rohan@128.2.204.116", disks=["/"], ours=["/home/rohan"],
                     ours_detail=["/home/rohan/Doom", "/home/rohan/lego_project", "/home/rohan/perseve", "/home/rohan/miniconda3", "/home/rohan/.cache", "/home/rohan/data"],
                     logs={}, rule="leave 2 of 8 GPUs free", gpu_note="8x RTX A4000 16 GB"),
    "spiderman": dict(host="rnagabhi@128.2.204.110", disks=["/sata2/data", "/home", "/"],
                      ours=["/sata2/data/rnagabhi", "/home/rnagabhi"],
                      ours_detail=["/sata2/data/rnagabhi/doom/raw_arnold", "/sata2/data/rnagabhi/doom/raw_stiegler", "/sata2/data/rnagabhi/doom/hf_release_arnold",
                                   "/sata2/data/rnagabhi/doom/results_superman", "/sata2/data/rnagabhi/doom/results_spiderman", "/sata2/data/rnagabhi/doom/latents_arnold",
                                   "/sata2/data/rnagabhi/doom/latents_arnold_aligned", "/sata2/data/rnagabhi/doom/raw_arnold_eval", "/home/rnagabhi/.cache", "/home/rnagabhi/miniconda3"],
                      logs={"030-dit-l32-aligned": "/sata2/data/rnagabhi/doom/results_spiderman/030-dit-l32-aligned/log.jsonl",
                            "031-unet-l32-aligned": "/sata2/data/rnagabhi/doom/results_spiderman/031-unet-l32-aligned/log.jsonl"},
                      rule="shared, no posted rule", gpu_note="4x RTX A6000 48 GB"),
}
OUR_USERS = {"rohan", "rnagabhi"}

REMOTE = r'''
set +e
echo "@@date"; date -u +%s
echo "@@gpus"; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo "@@procs"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
echo "@@uuids"; nvidia-smi --query-gpu=index,uuid --format=csv,noheader
echo "@@owners"; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do echo "$p $(ps -o user= -p $p 2>/dev/null) $(ps -o etimes= -p $p 2>/dev/null) $(ps -o args= -p $p 2>/dev/null | cut -c1-70)"; done
echo "@@disks"; df -B1 --output=target,size,used,avail %DISKS% 2>/dev/null | tail -n +2
echo "@@load"; cat /proc/loadavg; nproc; free -b | awk "NR==2{print \$2, \$3, \$7}"
echo "@@tmux"; tmux ls 2>/dev/null | cut -d: -f1
echo "@@ours"; %OURS%
echo "@@detail"; %DETAIL%
echo "@@homes"; %HOMES%
%LOGS%
'''


def ssh(host, script):
    pw = os.environ.get("PERSEVE_SERVER_PASSWORD")
    if not pw:
        raise SystemExit("PERSEVE_SERVER_PASSWORD not set (source ~/.zshrc)")
    r = subprocess.run(["sshpass", "-p", pw, "ssh", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=no", host, "bash -s"],
                       input=script, capture_output=True, text=True, timeout=900)
    return r.stdout, r.returncode


def sections(text):
    out, key = {}, None
    for line in text.splitlines():
        if line.startswith("@@"):
            key = line[2:].strip(); out[key] = []
        elif key is not None:
            out[key].append(line)
    return out


def parse_log(lines):
    """Last train, val, start, and end events of a DoomDiT log.jsonl tail."""
    last = {}
    for l in lines:
        try:
            d = json.loads(l)
        except ValueError:
            continue
        last[d.get("event")] = d
    tr, va = last.get("train"), last.get("val")
    out = {}
    if tr:
        out.update(step=tr.get("step"), loss=tr.get("loss"), lr=tr.get("lr"), steps_per_s=tr.get("steps_per_s"), grad_norm=tr.get("grad_norm"),
                   clip_frac=tr.get("clip_frac"), skipped=tr.get("skipped_updates"), t=tr.get("time"))
    if va:
        out.update(val_step=va.get("step"), val_loss=va.get("val_loss"), excursion=va.get("excursion"), val_by_quartile=va.get("val_loss_by_t_quartile"))
    if last.get("start"):
        out["started"] = last["start"].get("time")
    if last.get("end"):
        out["ended"] = last["end"].get("time")
    return out


def du_map(lines):
    out = {}
    for l in lines:
        p = l.split("\t") if "\t" in l else l.split(None, 1)
        if len(p) == 2 and p[0].strip().isdigit():
            out[p[1].strip()] = int(p[0])
    return out


def collect(name, cfg, du):
    script = REMOTE.replace("%DISKS%", " ".join(cfg["disks"]))
    script = script.replace("%OURS%", " ".join(f'du -xsb {p} 2>/dev/null;' for p in cfg["ours"]) if du else "true")
    script = script.replace("%DETAIL%", " ".join(f'du -xsb {p} 2>/dev/null;' for p in cfg["ours_detail"]) if du else "true")
    script = script.replace("%HOMES%", "du -xsb /home/* 2>/dev/null | sort -n" if (du and name == "superman") else "true")
    script = script.replace("%LOGS%", "\n".join(f'echo "@@log:{k}"; tail -n 400 {v} 2>/dev/null' for k, v in cfg["logs"].items()))
    text, rc = ssh(cfg["host"], script)
    s = sections(text)
    uuid_to_idx = {}
    for l in s.get("uuids", []):
        if "," in l:
            i, u = [x.strip() for x in l.split(",")]; uuid_to_idx[u] = int(i)
    owners = {}
    for l in s.get("owners", []):
        parts = l.split(None, 3)
        if len(parts) >= 2:
            owners[parts[0]] = dict(user=parts[1], etime_s=int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None, cmd=parts[3] if len(parts) > 3 else "")
    gpus = []
    for l in s.get("gpus", []):
        if not l.strip():
            continue
        i, gname, used, total, util = [x.strip() for x in l.split(",")]
        gpus.append(dict(index=int(i), name=gname, mem_used_mb=int(used), mem_total_mb=int(total), util=int(util), procs=[]))
    for l in s.get("procs", []):
        if "," not in l:
            continue
        u, pid, mem = [x.strip() for x in l.split(",")]
        gi = uuid_to_idx.get(u)
        if gi is None or gi >= len(gpus):
            continue
        o = owners.get(pid, {})
        gpus[gi]["procs"].append(dict(pid=int(pid), mem_mb=int(mem), user=o.get("user", "?"), ours=o.get("user") in OUR_USERS, etime_s=o.get("etime_s"), cmd=o.get("cmd", "")))
    for g in gpus:
        g["ours_mb"] = sum(p["mem_mb"] for p in g["procs"] if p["ours"])
        g["others_mb"] = sum(p["mem_mb"] for p in g["procs"] if not p["ours"])
        g["users"] = sorted({p["user"] for p in g["procs"]})
    disks = []
    for l in s.get("disks", []):
        parts = l.split()
        if len(parts) == 4:
            disks.append(dict(mount=parts[0], size_b=int(parts[1]), used_b=int(parts[2]), avail_b=int(parts[3])))
    load = s.get("load", [])
    la = load[0].split() if load else []
    sysinfo = dict(load1=float(la[0]) if la else None, load5=float(la[1]) if len(la) > 1 else None, cores=int(load[1]) if len(load) > 1 else None)
    if len(load) > 2:
        m = load[2].split(); sysinfo.update(mem_total_b=int(m[0]), mem_used_b=int(m[1]), mem_avail_b=int(m[2]))
    return dict(name=name, host=cfg["host"], rule=cfg["rule"], gpu_note=cfg["gpu_note"], ok=rc == 0 and bool(gpus),
                ts=int(s["date"][0]) if s.get("date") else int(time.time()),
                gpus=gpus, disks=disks, system=sysinfo, tmux=[t for t in s.get("tmux", []) if t.strip()],
                ours_b=du_map(s.get("ours", [])), ours_detail_b=du_map(s.get("detail", [])), homes_b=du_map(s.get("homes", [])) if name == "superman" else {},
                runs={k: parse_log(s.get(f"log:{k}", [])) for k in cfg["logs"]})


def main(a):
    snap = dict(collected_at=int(time.time()), servers={})
    for name, cfg in SERVERS.items():
        try:
            snap["servers"][name] = collect(name, cfg, a.du)
        except Exception as e:   # one unreachable server must not blank the dashboard
            snap["servers"][name] = dict(name=name, host=cfg["host"], rule=cfg["rule"], gpu_note=cfg["gpu_note"], ok=False, error=f"{type(e).__name__}: {e}",
                                         ts=int(time.time()), gpus=[], disks=[], system={}, tmux=[], ours_b={}, ours_detail_b={}, homes_b={}, runs={})
    json.dump(snap, open(a.out, "w"))
    for n, sv in snap["servers"].items():
        g = sv.get("gpus", [])
        print(n, "ok" if sv.get("ok") else sv.get("error"), f"{len(g)} gpus, ours on {[x['index'] for x in g if x['ours_mb']]}, disks {[(d['mount'], round(d['avail_b']/2**30)) for d in sv.get('disks', [])]}, runs {list(sv.get('runs', {}))}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--du", action="store_true", help="also measure directory sizes (slow)")
    main(p.parse_args())
