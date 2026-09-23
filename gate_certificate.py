"""The launch certificate: what the gates passed, per backbone, and the check that a launch is exactly that.

The first version of the pin (a commit hash in `$D/GATES_COMMIT`) was too weak: it accepted a launch
whose recipe had changed after the gates, and an SD 3.5-only gate run certified a U-Net launch. The
certificate is now a JSON file, written by `scripts/cluster/gates.sh` at GATES_GO, with one entry per
backbone the gates smoked. Each entry records

  * `command`     the resolved launch command, every flag, as `launch_nexttic.sh CERT_QUERY=1` prints
                  it (`--resume` and `--init-from` are left out: they are per-launch, and `init_from`
                  is recorded and compared separately on a first launch);
  * `commit`      `git rev-parse HEAD` of the checkout the launcher runs, which must be clean;
  * `corpus`      `train_wm.corpus_manifest` of the training and validation corpora: episode count, a
                  hash of the id list, and the content fingerprint the trainer itself checks on resume;
  * `encoders`    every `encode_meta_*.json` of both corpora: encoder git, VAE id and subfolder,
                  scale, shift, batch size, dtype;
  * `gates`       the result of every gate that applies to this backbone (its own fit, smoke, probes
                  and readback; its latent space's audits; the corpus-wide ones), each `ok`.

`launch_nexttic.sh` recomputes all of it for THE backbone it is launching and refuses on any
difference, a missing entry, or a gate that is absent or not ok. `ALLOW_UNGATED=1` bypasses the check
and is recorded in `resumes.log`.

    python gate_certificate.py write --cert $D/GATES_CERT.json --backbone unet --space sd15 \\
        --command "<CERT_QUERY output>" --repo $D/repo --train-latents ... --train-ids 0:2000 \\
        --val-latents ... --val-ids 6000:6100 --results $D/logs/gates_results.jsonl
    python gate_certificate.py check --cert $D/GATES_CERT.json --backbone unet --command "..." ...
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import time

# every gate a backbone's certificate must carry, each with status ok
REQUIRED_GATES = ("0 pin", "1 sidecar audit val", "1 sidecar audit train", "1c inventory train",
                  "1d latent alignment train", "1d latent alignment val", "2 alignment",
                  "3 fit", "4 smoke", "5 readback")
ENCODER_FIELDS = ("git", "vae_id", "vae_subfolder", "scaling_factor_applied", "shift_factor_applied")


def git_state(repo):
    """(commit, clean) of a checkout; (None, False) when it is not one."""
    try:
        head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None, False
    clean = subprocess.run(["git", "-C", repo, "diff", "--quiet", "HEAD", "--"], capture_output=True).returncode == 0
    return head, clean


def corpus_record(latents_dir, ids_spec):
    """The trainer's own corpus fingerprint, with the id list reduced to a count and a hash."""
    from doom_data import parse_episode_ids
    from train_wm import corpus_manifest
    ids = parse_episode_ids(ids_spec)
    m = corpus_manifest(latents_dir, ids)
    return {"ids": ids_spec, "num_episodes": len(m["episodes"]),
            "episodes_sha": hashlib.sha256(json.dumps(m["episodes"]).encode()).hexdigest()[:16],
            "fingerprint": m["fingerprint"]}


def encoder_records(latents_dir):
    """What each encode shard of a corpus recorded about its encoder, sorted by shard file."""
    out = []
    for path in sorted(glob.glob(os.path.join(latents_dir, "encode_meta_*.json"))):
        with open(path) as f:
            m = json.load(f)
        a = m.get("args") or {}
        out.append({"file": os.path.basename(path), **{k: m.get(k) for k in ENCODER_FIELDS},
                    "batch_size": a.get("batch_size"), "dtype": a.get("dtype")})
    return out


def identity(backbone, command, init_from, repo, train_latents, train_ids, val_latents, val_ids):
    """Everything about a launch the certificate pins, computed now."""
    commit, clean = git_state(repo)
    return {"backbone": backbone, "command": " ".join(command.split()), "init_from": init_from or "",
            "commit": commit, "clean": clean,
            "corpus": {"train": corpus_record(train_latents, train_ids), "val": corpus_record(val_latents, val_ids)},
            "encoders": {"train": encoder_records(train_latents), "val": encoder_records(val_latents)}}


def gate_results(results_path, backbone, space):
    """The gate results that apply to this backbone: corpus-wide, its latent space's, its own."""
    scopes = {"all", f"space:{space}", f"bb:{backbone}"}
    out = []
    with open(results_path) as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                if r.get("scope") in scopes:
                    out.append(r)
    return out


def gate_problems(gates):
    """Required gates that are missing or not ok, as strings."""
    passed = {}
    for g in gates:
        # a gate recorded more than once (per shard, per corpus) is ok only if every record is
        passed[g["gate"]] = passed.get(g["gate"], True) and g.get("status") == "ok"
    bad = [f"gate {g!r} did not pass" for g in REQUIRED_GATES if g in passed and not passed[g]]
    bad += [f"gate {g!r} has no result for this backbone" for g in REQUIRED_GATES if g not in passed]
    return bad


def flag_pairs(command):
    """A command as its words, with each `--flag value` kept together so a changed value shows."""
    toks, out, i = command.split(), [], 0
    while i < len(toks):
        if toks[i].startswith("--") and i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            out.append(f"{toks[i]} {toks[i + 1]}")
            i += 2
        else:
            out.append(toks[i])
            i += 1
    return out


def command_diff(cert, now):
    """What differs between the certified command and this one, readably; None if they are equal."""
    if cert.split() == now.split():
        return None
    a, b = flag_pairs(cert), flag_pairs(now)
    gone = [t for t in a if t not in b]
    new = [t for t in b if t not in a]
    return (f"launch command differs from the certified one (certified only: {gone[:12]}; now only: {new[:12]}"
            + ("; same flags in a different order or multiplicity" if not gone and not new else "") + ")")


def compare(entry, now, resuming=False):
    """Every difference between a certificate entry and a launch, as strings; empty means certified."""
    bad = []
    d = command_diff(entry.get("command", ""), now["command"])
    if d:
        bad.append(d)
    if not resuming and entry.get("init_from", "") != now["init_from"]:
        bad.append(f"init_from {now['init_from']!r} differs from the certified {entry.get('init_from')!r}")
    if not now["commit"]:
        bad.append("the checkout is not a git repository, so what launches cannot be pinned")
    elif entry.get("commit") != now["commit"]:
        bad.append(f"the checkout is at {now['commit']} but the gates certified {entry.get('commit')}")
    if not now["clean"]:
        bad.append("tracked files differ from the commit, so this is not the certified code")
    for which in ("train", "val"):
        c, n = (entry.get("corpus") or {}).get(which) or {}, now["corpus"][which]
        for k in ("ids", "num_episodes", "episodes_sha", "fingerprint"):
            if c.get(k) != n.get(k):
                bad.append(f"{which} corpus {k} is {n.get(k)!r}, certified {c.get(k)!r}")
        if (entry.get("encoders") or {}).get(which) != now["encoders"][which]:
            bad.append(f"{which} corpus encoder records (encode_meta_*.json) differ from the certified ones")
    bad += gate_problems(entry.get("gates") or [])
    return bad


def load(cert_path):
    try:
        with open(cert_path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"backbones": {}}


def write(cert_path, backbone, space, gpu, now, results_path):
    """Add or replace one backbone's entry, atomically, refusing if its gates did not all pass."""
    gates = gate_results(results_path, backbone, space)
    problems = gate_problems(gates)
    if problems:
        raise SystemExit(f"{backbone} cannot be certified: " + "; ".join(problems))
    if not now["commit"] or not now["clean"]:
        raise SystemExit(f"{backbone} cannot be certified: the checkout is not a clean git commit")
    cert = load(cert_path)
    cert.setdefault("backbones", {})[backbone] = {**now, "space": space, "gpu": gpu, "gates": gates,
                                                  "certified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    cert["written_by"] = "scripts/cluster/gates.sh"
    tmp = f"{cert_path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(cert, f, indent=1)
    os.replace(tmp, cert_path)
    return cert


def main(args):
    now = identity(args.backbone, args.command, args.init_from, args.repo, args.train_latents, args.train_ids,
                   args.val_latents, args.val_ids)
    if args.cmd == "write":
        write(args.cert, args.backbone, args.space, args.gpu, now, args.results)
        print(f"CERTIFIED {args.backbone} commit={now['commit']} into {args.cert}")
        return 0
    if not os.path.isfile(args.cert):
        print(f"no gate certificate at {args.cert}: run scripts/cluster/gates.sh first")
        return 1
    entry = load(args.cert).get("backbones", {}).get(args.backbone)
    if entry is None:
        print(f"{args.cert} certifies no {args.backbone} launch (it has: "
              f"{sorted(load(args.cert).get('backbones', {}))}); run the gates for {args.backbone}")
        return 1
    bad = compare(entry, now, args.resuming)
    if bad:
        print(f"{args.backbone} launch is not the certified one:\n  " + "\n  ".join(bad))
        return 1
    print(f"CERTIFIED {args.backbone} commit={now['commit']} certified_at={entry.get('certified_at')}")
    return 0


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["write", "check"])
    p.add_argument("--cert", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--space", default="", help="write: the backbone's latent space (sd15 or sd35)")
    p.add_argument("--gpu", default="", help="write: recorded for information, not compared")
    p.add_argument("--command", required=True, help="the normalized launch command (launch_nexttic.sh CERT_QUERY=1)")
    p.add_argument("--init-from", dest="init_from", default="")
    p.add_argument("--resuming", action="store_true", help="check: a resume, so init_from is not compared")
    p.add_argument("--repo", required=True, help="the checkout the launcher runs")
    p.add_argument("--train-latents", dest="train_latents", required=True)
    p.add_argument("--train-ids", dest="train_ids", required=True)
    p.add_argument("--val-latents", dest="val_latents", required=True)
    p.add_argument("--val-ids", dest="val_ids", required=True)
    p.add_argument("--results", default="", help="write: the gates' results, one JSON object per line")
    return p


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
