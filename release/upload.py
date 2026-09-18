"""
Publish the DoomDiT benchmark to Hugging Face: dataset shards, splits, evaluation corpora,
and checkpoints, driven by release/manifest.json.

Subcommands, in the order they are run (every one accepts --dry-run, which prints the plan
and touches nothing; `stage`, `checksum`, `upload` and `verify` are resumable):

    stage       hardlink (same filesystem) or copy every manifest entry into <staging>/<repo>/,
                plus the card as README.md and the manifest itself
    checksum    md5, sha256, git-blob sha1 and size of every staged file -> checksums.json;
                files already listed with the same size and mtime are not re-read
    upload      create the repos and run upload_large_folder per repo (multi-commit,
                resumes from <staging>/<repo>/.cache/.huggingface); needs HF_TOKEN
    verify      list the remote tree and compare every file's size and hash (LFS sha256, or
                git blob sha1 for small files) against checksums.json; exits 1 on any mismatch
    weights-md  render WEIGHTS.md from the manifest and checksums.json, with placeholders for
                files that do not exist yet

Typical sequence (see release/README_RELEASE.md for the full runbook):

    PY=~/miniconda3/envs/doom/bin/python
    $PY release/upload.py stage     --staging /sata2/data/rnagabhi/doom/hf_staging
    $PY release/upload.py checksum  --staging /sata2/data/rnagabhi/doom/hf_staging --out release/checksums.json
    HF_TOKEN=... $PY release/upload.py upload --staging ... --user <hf-user> --private
    HF_TOKEN=... $PY release/upload.py verify --staging ... --user <hf-user> --checksums release/checksums.json
    $PY release/upload.py weights-md --checksums release/checksums.json --user <hf-user> --out WEIGHTS.md

Hugging Face stores an LFS sha256 per large file and a git blob sha1 per small file, never an
md5, so `verify` checks the hash the Hub exposes and `checksums.json` carries the md5 for
humans (WEIGHTS.md) next to the hashes the Hub can confirm.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DEFAULT_MANIFEST = os.path.join(HERE, "manifest.json")
DEFAULT_CHECKSUMS = os.path.join(HERE, "checksums.json")
CHUNK = 1 << 24


# ----------------------------------------------------------------------------- manifest

def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def resolve_src(src, root):
    """Manifest source -> absolute path. `repo:` is relative to the git checkout, `/` absolute, else under root."""
    if src.startswith("repo:"):
        return os.path.join(REPO_ROOT, src[len("repo:"):])
    if os.path.isabs(src):
        return src
    return os.path.join(root, src)


def expand_entry(entry, root):
    """Yield (abs_src, dst) pairs for one manifest entry; directories are walked recursively."""
    pattern = resolve_src(entry["src"], root)
    dst = entry["dst"]
    matches = sorted(glob.glob(pattern))
    pairs = []
    for m in matches:
        if os.path.isdir(m):
            if not dst.endswith("/"):
                raise ValueError(f"{entry['src']}: a directory source needs a dst ending in '/'")
            for dirpath, _, files in os.walk(m):
                for fn in sorted(files):
                    p = os.path.join(dirpath, fn)
                    pairs.append((p, dst + os.path.relpath(p, m)))
        else:
            if dst.endswith("/"):
                d = dst + os.path.basename(m)
            elif "{parent}" in dst:
                d = dst.replace("{parent}", os.path.basename(os.path.dirname(m)))
            elif len(matches) > 1:
                raise ValueError(f"{entry['src']}: glob matched {len(matches)} files but dst is a single path")
            else:
                d = dst
            pairs.append((m, d))
    return pairs


def plan(manifest, only=None):
    """Resolve every entry. Returns (pairs_by_repo, missing) where missing lists required entries with no match."""
    root = manifest["root"]
    by_repo = {r: [] for r in manifest["repos"]}
    missing, absent_optional = [], []
    for e in manifest["entries"]:
        if only and e["repo"] != only:
            continue
        pairs = expand_entry(e, root)
        if not pairs:
            (missing if e.get("required", True) else absent_optional).append(e)
            continue
        for src, dst in pairs:
            by_repo[e["repo"]].append((src, dst, e))
    return by_repo, missing, absent_optional


def repo_id(manifest, repo, user):
    rid = manifest["repos"][repo]["repo_id"]
    if "{user}" in rid:
        if not user:
            raise SystemExit(f"repo id {rid!r} needs --user")
        rid = rid.replace("{user}", user)
    return rid


# ----------------------------------------------------------------------------- hashing

def hash_file(path):
    """md5, sha256, and git blob sha1 (what the Hub shows for non-LFS files) in one pass."""
    size = os.path.getsize(path)
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    blob = hashlib.sha1(); blob.update(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            md5.update(b); sha256.update(b); blob.update(b)
    return {"size": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest(), "git_sha1": blob.hexdigest()}


def human(n):
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024


# ----------------------------------------------------------------------------- stage

def stage(args):
    manifest = load_manifest(args.manifest)
    for repo, spec in manifest["repos"].items():
        card = resolve_src(spec["card"], manifest["root"])
        if not os.path.exists(card):
            raise SystemExit(f"{repo}: card {card} does not exist")
    by_repo, missing, absent = plan(manifest, args.only)
    for e in absent:
        print(f"  (absent, optional) {e['src']}")
    for e in missing:
        print(f"  MISSING REQUIRED  {e['src']}")
    total_files = total_bytes = 0
    for repo, pairs in by_repo.items():
        if args.only and repo != args.only:
            continue
        dest_root = os.path.join(args.staging, repo)
        n_new = 0
        for src, dst, _ in pairs:
            out = os.path.join(dest_root, dst)
            total_files += 1; total_bytes += os.path.getsize(src)
            if os.path.exists(out) and os.path.getsize(out) == os.path.getsize(src):
                continue
            n_new += 1
            if args.dry_run:
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if os.path.exists(out):
                os.remove(out)
            try:
                os.link(src, out)           # same filesystem: no extra disk
            except OSError:
                shutil.copy2(src, out)      # different filesystem: a full copy
        card = resolve_src(manifest["repos"][repo]["card"], manifest["root"])
        extras = [(card, "README.md"), (args.manifest, "release_manifest.json")]
        for src, dst in extras:
            out = os.path.join(dest_root, dst)
            total_files += 1
            if not args.dry_run:
                os.makedirs(dest_root, exist_ok=True)
                with open(src) as f:
                    text = f.read()
                if args.user:
                    text = text.replace("<hf-user>", args.user)   # the cards name the repos before the account exists
                with open(out, "w") as f:
                    f.write(text)           # small files: always refresh
        print(f"{repo}: {len(pairs)} files planned, {n_new} to stage, +{len(extras)} card/manifest -> {dest_root}")
    print(f"total {total_files} files, {human(total_bytes)}{' (dry run, nothing written)' if args.dry_run else ''}")
    if missing and not args.allow_missing:
        raise SystemExit(f"{len(missing)} required entries are missing; pass --allow-missing to stage anyway")


# ----------------------------------------------------------------------------- checksum

def staged_files(staging, repo):
    root = os.path.join(staging, repo)
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != ".cache"]      # upload_large_folder's state
        for fn in sorted(files):
            p = os.path.join(dirpath, fn)
            yield os.path.relpath(p, root).replace(os.sep, "/"), p


def checksum(args):
    manifest = load_manifest(args.manifest)
    old = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            old = json.load(f).get("files", {})
    by_repo, _, _ = plan(manifest, args.only)
    note_of = {}
    for repo, pairs in by_repo.items():
        for _, dst, e in pairs:
            note_of[f"{repo}/{dst}"] = e
    files = dict(old)
    n_hashed = n_kept = 0; t0 = time.time(); hashed_bytes = 0
    for repo in manifest["repos"]:
        if args.only and repo != args.only:
            continue
        if not os.path.isdir(os.path.join(args.staging, repo)):
            print(f"{repo}: nothing staged"); continue
        for rel, p in staged_files(args.staging, repo):
            key = f"{repo}/{rel}"
            st = os.stat(p)
            prev = old.get(key)
            if prev and prev.get("size") == st.st_size and prev.get("mtime") == int(st.st_mtime):
                n_kept += 1; continue
            if args.dry_run:
                print(f"  would hash {key} ({human(st.st_size)})"); continue
            rec = hash_file(p); rec["mtime"] = int(st.st_mtime)
            e = note_of.get(key, {})
            rec["group"] = e.get("group", "card and manifest" if rel in ("README.md", "release_manifest.json") else "")
            rec["src"] = e.get("src", "")
            files[key] = rec; n_hashed += 1; hashed_bytes += st.st_size
    if not args.dry_run:
        out = {"release": manifest["release"], "generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
               "staging": os.path.abspath(args.staging), "files": dict(sorted(files.items()))}
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, args.out)
    dt = time.time() - t0
    print(f"hashed {n_hashed} files ({human(hashed_bytes)}, {dt:.0f}s), kept {n_kept} unchanged, {len(files)} total -> {args.out}")


# ----------------------------------------------------------------------------- upload / verify

def hf_api():
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; nothing is uploaded or listed without it")
    return HfApi(token=token)


def upload(args):
    manifest = load_manifest(args.manifest)
    for repo, spec in manifest["repos"].items():
        if args.only and repo != args.only:
            continue
        folder = os.path.join(args.staging, repo)
        if not os.path.isdir(folder):
            print(f"{repo}: nothing staged at {folder}, skipped"); continue
        rid = repo_id(manifest, repo, args.user)
        n = sum(1 for _ in staged_files(args.staging, repo))
        size = sum(os.path.getsize(p) for _, p in staged_files(args.staging, repo))
        print(f"{repo}: {n} files, {human(size)} -> {spec['repo_type']} repo {rid} ({'private' if args.private else 'public'})")
        if args.dry_run:
            continue
        api = hf_api()
        api.create_repo(rid, repo_type=spec["repo_type"], exist_ok=True, private=args.private)
        api.upload_large_folder(repo_id=rid, repo_type=spec["repo_type"], folder_path=folder,
                                ignore_patterns=[".cache/**"], print_report=True, num_workers=args.workers)
        print(f"uploaded {rid}")


def verify(args):
    manifest = load_manifest(args.manifest)
    with open(args.checksums) as f:
        local = json.load(f)["files"]
    bad = 0
    for repo, spec in manifest["repos"].items():
        if args.only and repo != args.only:
            continue
        rid = repo_id(manifest, repo, args.user)
        expected = {k[len(repo) + 1:]: v for k, v in local.items() if k.startswith(repo + "/")}
        if args.dry_run:
            print(f"{repo}: would list {rid} and compare {len(expected)} files"); continue
        api = hf_api()
        remote = {}
        for item in api.list_repo_tree(rid, repo_type=spec["repo_type"], recursive=True):
            if getattr(item, "size", None) is None:      # folders
                continue
            lfs = getattr(item, "lfs", None)
            sha256 = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
            remote[item.path] = {"size": item.size, "sha256": sha256, "git_sha1": getattr(item, "blob_id", None)}
        missing = sorted(set(expected) - set(remote))
        extra = sorted(set(remote) - set(expected))
        mismatched = []
        for path, loc in expected.items():
            rem = remote.get(path)
            if rem is None:
                continue
            ok = rem["size"] == loc["size"] and (rem["sha256"] == loc["sha256"] if rem["sha256"] else rem["git_sha1"] == loc["git_sha1"])
            if not ok:
                mismatched.append(path)
        print(f"{repo} ({rid}): {len(expected)} expected, {len(remote)} remote, {len(missing)} missing, {len(mismatched)} mismatched, {len(extra)} extra")
        for p in missing[:50]:
            print(f"  missing    {p}")
        for p in mismatched[:50]:
            print(f"  mismatched {p}")
        for p in extra[:20]:
            print(f"  extra      {p}")
        bad += len(missing) + len(mismatched)
    if bad:
        raise SystemExit(f"{bad} files missing or mismatched")
    if not args.dry_run:
        print("every staged file is on the Hub with the expected size and hash")


# ----------------------------------------------------------------------------- WEIGHTS.md

def weights_md(args):
    manifest = load_manifest(args.manifest)
    files = {}
    if os.path.exists(args.checksums):
        with open(args.checksums) as f:
            files = json.load(f)["files"]
    user = args.user or "<hf-user>"
    ids = {r: manifest["repos"][r]["repo_id"].replace("{user}", user) for r in manifest["repos"]}
    lines = ["# Weights and data", "",
             f"Generated by `python release/upload.py weights-md` from `release/manifest.json` and `release/checksums.json` "
             f"(release `{manifest['release']}`). Rows without an md5 are placeholders for files that did not exist when the "
             "checksums were computed; rerun `checksum` then `weights-md` after they land.", "",
             "| Repo | Hub id |", "|---|---|"]
    lines += [f"| {r} | `{ids[r]}` |" for r in ids]
    lines += ["", "## Download", "", "```python", "from huggingface_hub import snapshot_download",
              f"snapshot_download('{ids['model']}', local_dir='weights')                    # every checkpoint, decoder, judge",
              f"snapshot_download('{ids['dataset']}', repo_type='dataset', local_dir='data',",
              "                  allow_patterns=['splits/*', 'meta/*', 'eval/seen/*'])       # pick what you need; data/ is 202 GB",
              "```", "",
              "Load a world-model checkpoint (bf16 live weights; `snap_*.pt` files also carry `ema`):", "",
              "```python", "import torch", "from backbones import build_model",
              "ck = torch.load('weights/dit_xl2_seed0/best.pt', map_location='cpu', weights_only=True)",
              "model = build_model('dit', num_actions=29, context_frames=32, noise_buckets=10, grad_ckpt=False,",
              "                    action_dropout=ck['args']['action_dropout'])",
              "model.load_state_dict({k: v.float() for k, v in ck['model'].items()})", "```", ""]
    for repo in manifest["repos"]:
        lines += [f"## {repo} repo: `{ids[repo]}`", ""]
        groups = {}
        for e in manifest["entries"]:
            if e["repo"] == repo:
                groups.setdefault(e["group"], []).append(e)
        # files named by an exact entry are not repeated under a directory entry that also covers them
        exact = {f"{repo}/{e['dst']}" for e in manifest["entries"] if e["repo"] == repo and not e["dst"].endswith("/") and "{parent}" not in e["dst"]}
        for g, entries in groups.items():
            lines += [f"### {g}", "", "| Published path | Size | md5 | sha256 (Hub LFS) | Note |", "|---|---|---|---|---|"]
            for e in entries:
                note = e.get("note", "")
                shown = e["dst"].replace("{parent}", "*")
                is_dir = e["dst"].endswith("/")
                # a dst ending in "/" owns everything below it; "{parent}" stands for exactly one path segment
                pat = "^" + re.escape(f"{repo}/{e['dst']}").replace(r"\{parent\}", "[^/]+") + ("" if is_dir else "$")
                matches = sorted((k, v) for k, v in files.items() if re.match(pat, k) and not (is_dir and k in exact))
                if not matches:
                    tag = "placeholder, not yet released" if not e.get("required", True) else "placeholder, not yet checksummed"
                    lines.append(f"| `{shown}` | | *{tag}* | | {note} |")
                elif len(matches) > 8:
                    tot = sum(v["size"] for _, v in matches)
                    lines.append(f"| `{shown}` ({len(matches)} files) | {human(tot)} | per-file md5s in `release/checksums.json` | | {note} |")
                else:
                    for i, (k, v) in enumerate(matches):
                        lines.append(f"| `{k[len(repo) + 1:]}` | {human(v['size'])} | `{v['md5']}` | `{v['sha256'][:16]}...` | {note if i == 0 else ''} |")
            lines.append("")
    lines += ["## Legacy: April 2026 class-project checkpoint", "",
              "The DiT-XL/2 trained for the CMU 18-789 project on the JPEG `arnaudstiegler/vizdoom-500-episodes-skipframe-4-lvl5` set "
              "at 160x120 lives in GitHub release [`002-DiT-XL-2-best-90k`](https://github.com/RohanNaga/Doom/releases/tag/002-DiT-XL-2-best-90k) "
              "(`bash download_weights.sh` reassembles it). It trained on all 500 episodes, so it has no honest held-out number; "
              "its 27.43 dB / 0.117 LPIPS on lossless Stiegler-agent frames (Sep 9) is a same-agent, different-recording score.", "",
              "| File | Size | md5 |", "|---|---|---|",
              "| `results/002-DiT-XL-2/checkpoints/best.pt` (step 87,200) | 2.6 GB | `ff54cab2093665f00d997f3e5fd1dd27` |",
              "| `results/002-DiT-XL-2/checkpoints/0090000.pt` | 2.6 GB | `3f1be8922d4ffdfd8cd698fe9dede922` |", ""]
    text = "\n".join(lines)
    if args.dry_run:
        print(text); return
    with open(args.out, "w") as f:
        f.write(text)
    print(f"wrote {args.out} ({len(lines)} lines)")


# ----------------------------------------------------------------------------- cli

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, staging=True):
        sp.add_argument("--manifest", default=DEFAULT_MANIFEST)
        sp.add_argument("--only", choices=["dataset", "model"], default=None, help="limit to one repo")
        sp.add_argument("--dry-run", action="store_true")
        if staging:
            sp.add_argument("--staging", required=True, help="staging root; one subfolder per repo is created inside it")
        return sp

    s = common(sub.add_parser("stage")); s.add_argument("--allow-missing", action="store_true")
    s.add_argument("--user", default=None, help="Hub namespace written into the cards in place of <hf-user>"); s.set_defaults(fn=stage)
    s = common(sub.add_parser("checksum")); s.add_argument("--out", default=DEFAULT_CHECKSUMS); s.set_defaults(fn=checksum)
    s = common(sub.add_parser("upload")); s.add_argument("--user", default=None, help="Hub namespace that fills {user} in the manifest")
    s.add_argument("--private", action="store_true"); s.add_argument("--workers", type=int, default=4); s.set_defaults(fn=upload)
    s = common(sub.add_parser("verify"), staging=False); s.add_argument("--user", default=None)
    s.add_argument("--checksums", default=DEFAULT_CHECKSUMS); s.set_defaults(fn=verify)
    s = common(sub.add_parser("weights-md"), staging=False); s.add_argument("--user", default=None)
    s.add_argument("--checksums", default=DEFAULT_CHECKSUMS); s.add_argument("--out", default=os.path.join(REPO_ROOT, "WEIGHTS.md")); s.set_defaults(fn=weights_md)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
