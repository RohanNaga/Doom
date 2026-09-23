"""Upload the dense per-tic latents to RohanNaga/doom-dense-arnold-latents, and free Spiderman's disk.

Runs on Spiderman's CPU in tmux `hf-latents-6000` (Rohan, Sep 23 2026). Every 10 minutes:

  rolling   ids 2000:5999 under $LATENTS_ROLLING_ROOT (default $D/ext_2000_6000, Hub folders
            sd15/arenas_2000_6000 and sd35/arenas_2000_6000): finished pairs
            (`ep_XXXXX_latents.npy` + `_meta.npz`) that pass the row check are uploaded in commits of
            about 100 episodes; every uploaded file's byte size is then read back from the Hub tree API
            and compared, and only then are the LOCAL files deleted. Each episode gets one line in the
            directory's `uploaded_manifest.jsonl` (id, files, sizes, commit), which the Superman encoder
            reads so that a deleted episode is not encoded again. Every deletion is logged.
  once      complete sets, uploaded and verified the same way but never deleted: sd15/arenas ids
            0:1999 (after Spiderman's enc-train1 exits), sd35/arenas ids 0:1999, and the three SD 3.5
            evaluation corpora (after their split files are published).

Hard guards: nothing below id 2000 and nothing in an evaluation directory is ever deleted. Each commit
also carries the directory's encode_meta_*.json, episodes_*.jsonl, canonical_controls.json and a
regenerated PROVENANCE.md stating which ids came from which encoder environment.

    ~/miniconda3/envs/doom/bin/python hf_latents_uploader.py [--once]
"""
import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, get_token

REPO = "RohanNaga/doom-dense-arnold-latents"
D = "/sata2/data/rnagabhi/doom"
RAW = f"{D}/raw_arnold_dense/arenas"
RAW678 = f"{D}/raw_arnold_dense/arenas_678"
LOG = f"{D}/logs/hf_latents_6000.log"
STATE = f"{D}/tmp/hf_latents_state.json"
BATCH = 100
QUIET_S = 1800          # a partial rolling batch goes up once no new episode has arrived for this long
META_PATTERNS = ["encode_meta_*.json", "episodes_*.jsonl", "canonical_controls.json", "PROVENANCE.md"]
EP = re.compile(r"^ep_(\d{5})_latents\.npy$")
FAILED = set()          # (dir, id) row-check failures already logged

TRAIN = {   # the frozen 0:2000 paper-row corpora: uploaded once, never deleted from
    "sd15": dict(dir=f"{D}/latents_arnold_dense_pertic/arenas", repo="sd15/arenas", ch=4, raw=RAW),
    "sd35": dict(dir=f"{D}/latents_arnold_dense_pertic_sd35/arenas", repo="sd35/arenas", ch=16, raw=RAW),
}
# The 2000:6000 sets live under their own tree (Sep 23 2026): the launch certificates pin every shard
# log in the 0:2000 directories, so Superman's phase 2 and 3 write elsewhere, and these are the only
# directories anything is ever deleted from. They go to their own Hub folders too, so their shard
# logs (encode_meta_1N.json from Superman's cards) never overwrite the 0:2000 sets' files.
ROLLING_ROOT = os.environ.get("LATENTS_ROLLING_ROOT", f"{D}/ext_2000_6000")
ROLLING = {
    "sd15": dict(dir=f"{ROLLING_ROOT}/latents_arnold_dense_pertic/arenas", repo="sd15/arenas_2000_6000", ch=4, raw=RAW),
    "sd35": dict(dir=f"{ROLLING_ROOT}/latents_arnold_dense_pertic_sd35/arenas", repo="sd35/arenas_2000_6000", ch=16, raw=RAW),
}
EVAL_ROOT = f"{D}/latents_arnold_dense_pertic_eval_sd35"
EVALS = {
    "val": dict(dir=f"{EVAL_ROOT}/val", repo="sd35/val", ch=16, raw=RAW, ids=(6000, 6100)),
    "test": dict(dir=f"{EVAL_ROOT}/test", repo="sd35/test", ch=16, raw=RAW, ids=(7000, 7100)),
    "arenas_678": dict(dir=f"{EVAL_ROOT}/arenas_678", repo="sd35/arenas_678", ch=16, raw=RAW678, ids=(60, 120)),
}

# Encoder environments, measured on Sep 23 2026. Superman's workers also write theirs into
# encode_meta_NN.json (`encoded_on.env`) from the 2000:6000 sets on; Spiderman's shards did not.
ENVS = {
    "spiderman": "Spiderman, NVIDIA RTX A6000, conda env `doom`: torch 2.14.0+cu130, cuDNN 9.24.0, diffusers 0.31.0 "
                 "(versions read from the environment on Sep 23 2026; not recorded at encode time)",
    "superman": "Superman, NVIDIA RTX A4000, conda env `doomenc`: python 3.10.21, torch 2.14.0+cu126, cuDNN 9.10.2, "
                "diffusers 0.40.0, driver 550.144.03",
}


def log(msg):
    line = f"{datetime.datetime.now().astimezone().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_state():
    try:
        return json.load(open(STATE))
    except (OSError, ValueError):
        return {}


def save_state(st):
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, STATE)


def manifest_ids(d):
    ids = set()
    try:
        for line in open(os.path.join(d, "uploaded_manifest.jsonl")):
            ids.add(json.loads(line)["id"])
    except OSError:
        pass
    return ids


def local_ids(d):
    """Ids whose latents AND sidecar are both on disk (the Superman ship renames the latents last)."""
    out = {}
    for name in os.listdir(d):
        m = EP.match(name)
        if m and os.path.exists(os.path.join(d, f"ep_{m.group(1)}_meta.npz")):
            out[int(m.group(1))] = os.path.getmtime(os.path.join(d, name))
    return out


def row_check(d, raw, i, ch):
    """None if latent rows == sidecar rows == raw parquet rows, shape (T, ch, 32, 40) fp16, buttons <U19."""
    ep = f"ep_{i:05d}"
    try:
        a = np.load(os.path.join(d, f"{ep}_latents.npy"), mmap_mode="r")
        m = np.load(os.path.join(d, f"{ep}_meta.npz"))
        lens = {len(m[k]) for k in m.files}
        rows = pq.read_metadata(os.path.join(raw, f"{ep}.parquet")).num_rows
        ok = (a.dtype == np.float16 and a.shape[1:] == (ch, 32, 40) and lens == {a.shape[0]}
              and a.shape[0] == rows and str(m["buttons"].dtype) == "<U19")
        return None if ok else f"shape {a.shape} {a.dtype} sidecar lens {sorted(lens)} raw rows {rows} buttons {m['buttons'].dtype}"
    except Exception as e:      # an unreadable file is a failed check, never an upload
        return f"{type(e).__name__}: {e}"


def hub_sizes(folder, limit=1000):
    """{path: size} of every file under `folder` in the repo, following the tree API's Link pagination."""
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{folder}?limit={limit}"
    headers = {"Authorization": f"Bearer {get_token()}"}
    sizes = {}
    while url:
        r = requests.get(url, headers=headers, timeout=120)
        r.raise_for_status()
        for e in r.json():
            if e.get("type") == "file":
                sizes[e["path"]] = e.get("lfs", {}).get("size", e.get("size"))
        nxt = r.links.get("next", {}).get("url")
        url = nxt
    return sizes


def provenance(d, title):
    """PROVENANCE.md: which ids of this directory came from which encoder environment."""
    by_nn = {}
    for f in glob.glob(os.path.join(d, "episodes_*.jsonl")):
        nn = int(re.search(r"episodes_(\d+)\.jsonl$", f).group(1))
        for line in open(f):
            try:
                by_nn.setdefault(nn, set()).add(int(json.loads(line)["episode"][3:]))
            except (ValueError, KeyError):
                pass
    env_ids = {"spiderman": set(), "superman": set()}
    for nn, ids in by_nn.items():
        env_ids["superman" if nn >= 10 else "spiderman"] |= ids
    present = set(local_ids(d)) | manifest_ids(d)
    attributed = env_ids["spiderman"] | env_ids["superman"]
    both = env_ids["spiderman"] & env_ids["superman"]

    def ranges(ids):
        ids = sorted(ids)
        out, start = [], None
        for k, i in enumerate(ids):
            if start is None:
                start = i
            if k + 1 == len(ids) or ids[k + 1] != i + 1:
                out.append(f"{start}:{i + 1}" if i > start else f"{start}")
                start = None
        return ", ".join(out) if out else "none"

    metas = []
    for f in sorted(glob.glob(os.path.join(d, "encode_meta_*.json"))):
        try:
            m = json.load(open(f))
        except ValueError:
            continue
        a = m.get("args", {})
        on = m.get("encoded_on", {})
        metas.append(f"| {os.path.basename(f)} | {on.get('host', 'spiderman')} | {on.get('gpu', 'NVIDIA RTX A6000')} "
                     f"| {m.get('torch')} | {on.get('env', {}).get('cudnn', '')} | {on.get('env', {}).get('diffusers', '')} "
                     f"| {(m.get('git') or '?')[:12]} | {a.get('batch_size')} | {a.get('dtype')} | {m.get('vae_id')} |")
    text = [f"# Provenance of `{title}`", "",
            f"Generated {datetime.datetime.now().astimezone().isoformat(timespec='minutes')} from this directory's "
            "`episodes_NN.jsonl` (NN 00-09: Spiderman shards; NN 10-17: Superman workers) and `encode_meta_NN.json`.", "",
            "Latents are bf16-autocast encodes at batch 64. Two encoder environments differ at rounding level: on "
            "ep 0 of the 4-channel set, Superman against Spiderman was 67% bit-identical, RMS difference 0.3% of the "
            "latent std, zero mean, decode PSNR within 0.001 dB. Within one environment encodes are bit-identical "
            "across cards and runs (checked on Superman GPUs 1, 2 and 7).", "",
            "| Environment | Episodes | Ids (half-open ranges A:B) |", "|---|---|---|"]
    for env in ("spiderman", "superman"):
        if env_ids[env]:
            text.append(f"| {ENVS[env]} | {len(env_ids[env])} | {ranges(env_ids[env])} |")
    unattributed = present - attributed
    text += ["", f"Episodes present or uploaded: {len(present)}. Without a summary line (environment unknown): "
             f"{len(unattributed)} ({ranges(unattributed)}). Listed by both environments: {len(both)} ({ranges(both)}).",
             "", "| encode_meta | host | GPU | torch | cuDNN | diffusers | git | batch | dtype | VAE |",
             "|---|---|---|---|---|---|---|---|---|---|"] + metas + [""]
    with open(os.path.join(d, "PROVENANCE.md"), "w") as f:
        f.write("\n".join(text))


def upload_batch(api, spec, ids, delete, label):
    """Upload `ids` of one directory, verify sizes on the Hub, record them, and optionally delete."""
    d, folder = spec["dir"], spec["repo"]
    for i in ids:       # the deletion guard lives next to the only code that deletes
        if delete and not (2000 <= i < 6000 and d in (ROLLING["sd15"]["dir"], ROLLING["sd35"]["dir"])):
            raise SystemExit(f"refusing to delete id {i} in {d}")
    provenance(d, folder)
    names = [n for i in ids for n in (f"ep_{i:05d}_latents.npy", f"ep_{i:05d}_meta.npz")]
    sizes = {n: os.path.getsize(os.path.join(d, n)) for n in names}
    t0 = time.time()
    info = api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=d, path_in_repo=folder,
                             allow_patterns=names + META_PATTERNS,
                             commit_message=f"{label}: {len(ids)} episodes {min(ids)}..{max(ids)}")
    commit = getattr(info, "oid", None) or str(info)
    hub = hub_sizes(folder)
    bad = [n for n in names if hub.get(f"{folder}/{n}") != sizes[n]]
    gb = sum(sizes.values()) / 1e9
    if bad:
        log(f"VERIFY_FAILED {label} commit {commit}: {len(bad)} files differ on the Hub, e.g. {bad[:4]}; nothing deleted")
        return False
    log(f"uploaded {label} {len(ids)} episodes {min(ids)}..{max(ids)} {gb:.1f} GB in {time.time() - t0:.0f} s, "
        f"commit {commit[:12]}, all {len(names)} sizes verified")
    with open(os.path.join(d, "uploaded_manifest.jsonl"), "a") as f:
        for i in ids:
            ep = f"ep_{i:05d}"
            f.write(json.dumps({"id": i, "episode": ep, "latents": f"{ep}_latents.npy", "folder": folder,
                                "files": {n: sizes[n] for n in (f"{ep}_latents.npy", f"{ep}_meta.npz")},
                                "commit": commit, "deleted_local": bool(delete),
                                "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}) + "\n")
    if delete:
        for n in names:
            os.remove(os.path.join(d, n))
            log(f"deleted local {d}/{n} ({sizes[n]} bytes, on the Hub at {folder}/{n}, commit {commit[:12]})")
    return True


def pending(spec, lo, hi):
    """Ids in [lo, hi) on disk, not yet uploaded, passing the row check (failures are logged once each)."""
    d = spec["dir"]
    done = manifest_ids(d)
    have = local_ids(d)
    ok, newest = [], 0
    for i, mt in sorted(have.items()):
        if lo <= i < hi and i not in done:
            err = row_check(d, spec["raw"], i, spec["ch"])
            if err:
                if (d, i) not in FAILED:
                    FAILED.add((d, i))
                    log(f"ROW_CHECK_FAILED {d} ep_{i:05d}: {err}; not uploaded")
                continue
            ok.append(i)
            newest = max(newest, mt)
    return ok, newest


def tmux_running(name):
    return subprocess.run(["tmux", "has-session", "-t", name], capture_output=True).returncode == 0


def finish_deletions(spec):
    """Delete local 2000:5999 pairs the manifest already records as uploaded, verified and deleted.

    Only reachable when the process died between writing the manifest and removing the files."""
    d = spec["dir"]
    try:
        lines = [json.loads(x) for x in open(os.path.join(d, "uploaded_manifest.jsonl"))]
    except OSError:
        return
    for r in lines:
        if not (r.get("deleted_local") and 2000 <= r["id"] < 6000):
            continue
        for n, size in r["files"].items():
            p = os.path.join(d, n)
            if os.path.exists(p) and os.path.getsize(p) == size:
                os.remove(p)
                log(f"deleted local {p} ({size} bytes; manifest commit {r['commit'][:12]}, left by an interrupted cycle)")


def cycle(api, st):
    # rolling uploads with local deletion: training ids 2000:5999 only, from the separate tree
    for space, spec in ROLLING.items():
        if not os.path.isdir(spec["dir"]):
            continue
        finish_deletions(spec)
        ids, newest = pending(spec, 2000, 6000)
        while len(ids) >= BATCH or (ids and time.time() - newest > QUIET_S):
            if not upload_batch(api, spec, ids[:BATCH], True, f"{space} arenas 2000:6000"):
                break
            ids = ids[BATCH:]
    # complete sets, uploaded once and kept
    for space, spec in TRAIN.items():
        key = f"{space}_0_2000"
        if st.get(key) == "done":
            continue
        have = local_ids(spec["dir"])
        if any(i not in have for i in range(2000)):
            continue
        if space == "sd15" and tmux_running("enc-train1"):
            continue
        ids, _ = pending(spec, 0, 2000)
        for k in range(0, len(ids), BATCH):
            if not upload_batch(api, spec, ids[k:k + BATCH], False, f"{space} arenas 0:2000"):
                return
        if not pending(spec, 0, 2000)[0] and len(manifest_ids(spec["dir"]) & set(range(2000))) == 2000:
            st[key] = "done"; save_state(st); log(f"COMPLETE {key}: 2000 episodes on the Hub at {spec['repo']}")
    for name, spec in EVALS.items():
        key = f"sd35_eval_{name}"
        if st.get(key) == "done" or not os.path.exists(f"{EVAL_ROOT}/split_{name}.json"):
            continue
        lo, hi = spec["ids"]
        ids, _ = pending(spec, lo, hi)
        if ids and not upload_batch(api, spec, ids, False, f"sd35 {name}"):
            continue
        if len(manifest_ids(spec["dir"]) & set(range(lo, hi))) == hi - lo:
            api.upload_file(path_or_fileobj=f"{EVAL_ROOT}/split_{name}.json", path_in_repo=f"sd35/split_{name}.json",
                            repo_id=REPO, repo_type="dataset", commit_message=f"sd35 split_{name}.json")
            st[key] = "done"; save_state(st); log(f"COMPLETE {key}: {hi - lo} episodes and split_{name}.json on the Hub")


def main(args):
    api = HfApi()
    log(f"START uploader repo={REPO} once={args.once} as {api.whoami()['name']}")
    while True:
        st = load_state()
        try:
            cycle(api, st)
        except Exception as e:      # network or Hub trouble: log, keep every local file, try again next cycle
            log(f"cycle error {type(e).__name__}: {str(e)[:300]}")
        if args.once:
            break
        free = os.statvfs("/sata2/data")
        log(f"idle; /sata2/data {free.f_bavail * free.f_frsize / 1e9:.0f} GB free")
        time.sleep(600)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true", help="one cycle, then exit")
    main(p.parse_args())
    sys.exit(0)
