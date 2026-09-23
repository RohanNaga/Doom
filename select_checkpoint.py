"""Choose the checkpoint a run reports, on validation only, by a rule written down before any test score.

The 2026-09-22 review (H2) found that `after_nexttic.sh` scored validation, test and unseen in one
pass, so nothing stopped a test number from informing which checkpoint or variant was reported. The
evaluation is now two stages. This module is the hinge between them:

  1. `candidates`: the last stable checkpoint (the newest file carrying an EMA, by STORED step) and up
     to two snapshots before it. `after_nexttic.sh --select` scores each on validation, live and EMA
     from the same file, so the six numbers are paired.
  2. `choose`: applies `RULE` to those scores and writes `selection.json`, with every candidate's
     numbers, the file hashes, the rule itself and the settings the scores were measured under.
  3. `show`: prints the chosen checkpoint for the test stage, after checking the file on disk still
     has the hash that was selected.

The rule (the audit's "preregistered raw PSNR with LPIPS check",
`.claude/analyses/astra-prelaunch-audit-2026-09-21.md`): among the candidate (checkpoint, variant)
pairs, keep those whose validation `lpips_raw` is within `LPIPS_TOLERANCE` of the best `lpips_raw`,
and take the highest `psnr_raw` among them. Ties go to the lower LPIPS, then the later step, then live
weights. Raw, not decoded: both are measured against the game's own frames.

    python select_checkpoint.py candidates --results-dir R
    python select_checkpoint.py choose --results-dir R --candidates R/select/candidates.txt \\
        --scores-dir R/select --out R/selection.json --meta decoder=sha256:... --meta windows=512
    python select_checkpoint.py show --selection R/selection.json
"""
import argparse
import hashlib
import json
import os
import time

from eval_identity import sha256_file

LPIPS_TOLERANCE = 0.01
PREVIOUS_SNAPSHOTS = 2
VARIANTS = ("live", "ema")
RULE = {
    "name": "raw-psnr-with-lpips-check-v1",
    "metric": "psnr_raw mean on validation, higher is better",
    "check": f"lpips_raw mean within {LPIPS_TOLERANCE} of the best candidate's lpips_raw",
    "tie_break": "lower lpips_raw, then later stored step, then live before ema",
    "candidates": f"the newest checkpoint carrying an EMA and up to {PREVIOUS_SNAPSHOTS} earlier snapshots, "
                  "each scored live and EMA from the same file",
    "declared": "2026-09-22, before any next-tic model was scored (docs/REVIEW_2026-09-22.md H2)",
}


def candidates(results_dir, previous=PREVIOUS_SNAPSHOTS):
    """The last stable checkpoint and up to `previous` earlier snapshots, newest first.

    "Stable" is the newest file carrying an EMA by stored step: the trainer writes every checkpoint
    to a temporary name and renames it into place, so a listed file is complete. Earlier candidates
    are snapshots only, one per stored step.
    """
    from pick_checkpoint import candidates as files, describe, pick
    newest = pick(results_dir, require_ema=True)
    earlier = {}
    for p in files(results_dir):
        if not os.path.basename(p).startswith("snap_"):
            continue
        d = describe(p)
        if d["has_ema"] and d["step"] < newest["step"] and d["step"] not in earlier:
            earlier[d["step"]] = d
    picked = [newest] + [earlier[s] for s in sorted(earlier, reverse=True)[:previous]]
    return [{"path": r["path"], "step": int(r["step"]), "kind": r["kind"],
             "sha256": sha256_file(r["path"])} for r in picked]


def score_dir_name(c):
    """Where a candidate's validation scores live under the scores directory."""
    return f"s{int(c['step']):07d}_{c['sha256'][:12]}"


def read_candidates(path):
    """The `<path> <step> <sha256> [<score dir>]` lines `candidates` printed, as dicts."""
    out = []
    with open(path) as f:
        for ln in f:
            parts = ln.split()
            if parts:
                out.append({"path": parts[0], "step": int(parts[1]), "sha256": parts[2]})
    return out


def _mean(m, key):
    v = m.get(key)
    return float(v["mean"]) if isinstance(v, dict) and v.get("mean") is not None else None


def load_scores(cands, scores_dir):
    """One row per (candidate, variant), read from `<scores_dir>/<name>/eval_tf_val{,_ema}/metrics.json`.

    Each metrics file must say it scored this candidate: the same checkpoint path, the same stored
    step, and the requested variant. A score written for another file is refused, not used.
    """
    rows, problems = [], []
    for c in cands:
        for v in VARIANTS:
            d = os.path.join(scores_dir, score_dir_name(c), "eval_tf_val" + ("_ema" if v == "ema" else ""))
            try:
                with open(os.path.join(d, "metrics.json")) as f:
                    m = json.load(f)
            except (OSError, ValueError) as e:
                problems.append(f"{d}: no readable metrics.json ({e.__class__.__name__})")
                continue
            cfg = m.get("config") or {}
            if cfg.get("ckpt") != c["path"] or bool(cfg.get("use_ema")) != (v == "ema"):
                problems.append(f"{d}: scored {cfg.get('ckpt')} use_ema={cfg.get('use_ema')}, "
                                f"not {c['path']} {v}")
                continue
            if str(cfg.get("step")) != str(c["step"]):
                problems.append(f"{d}: scored stored step {cfg.get('step')}, not {c['step']}")
                continue
            psnr, lp = _mean(m, "psnr_raw"), _mean(m, "lpips_raw")
            if psnr is None or lp is None:
                problems.append(f"{d}: no psnr_raw/lpips_raw (was it scored without --parquet-dir?)")
                continue
            rows.append({**c, "variant": v, "psnr_raw": psnr, "lpips_raw": lp,
                         "windows": (m.get("psnr_raw") or {}).get("n"), "score_dir": d})
    return rows, problems


def choose(rows, tolerance=LPIPS_TOLERANCE):
    """The row `RULE` selects, and the rows that passed the LPIPS check."""
    if not rows:
        raise SystemExit("no scored candidate to choose from")
    best_lpips = min(r["lpips_raw"] for r in rows)
    eligible = [r for r in rows if r["lpips_raw"] <= best_lpips + tolerance]
    chosen = max(eligible, key=lambda r: (r["psnr_raw"], -r["lpips_raw"], r["step"], r["variant"] == "live"))
    return chosen, eligible


def _sha_of_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_selection(results_dir, cands, scores_dir, out, meta=None, force=False):
    """Score, choose and write `selection.json` atomically; refuses to overwrite one without `force`."""
    if os.path.exists(out) and not force:
        raise SystemExit(f"{out} exists: a selection is made once. Pass --force only to redo it "
                         "deliberately, and never after a test score exists")
    rows, problems = load_scores(cands, scores_dir)
    if problems:
        raise SystemExit("cannot select, the validation scores are incomplete or mislabelled:\n  "
                         + "\n  ".join(problems))
    chosen, eligible = choose(rows)
    for c in cands:     # the files must still be the ones that were scored
        now = sha256_file(c["path"])
        if now != c["sha256"]:
            raise SystemExit(f"{c['path']} changed after it was scored ({c['sha256'][:12]} -> {now[:12]})")
    body = {"results_dir": results_dir, "rule": RULE, "lpips_tolerance": LPIPS_TOLERANCE,
            "chosen": {k: chosen[k] for k in ("path", "step", "sha256", "variant", "psnr_raw", "lpips_raw")},
            "eligible": [(r["step"], r["variant"]) for r in eligible],
            "scores": rows, "meta": dict(meta or {}),
            "selected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    body["selection_sha256"] = _sha_of_json({k: v for k, v in body.items() if k != "selected_at"})
    tmp = f"{out}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(body, f, indent=1)
    os.replace(tmp, out)
    return body


def show(selection_path):
    """`<path> <step> <sha256> <variant> <selection_sha256>` of the chosen checkpoint, verified on disk."""
    with open(selection_path) as f:
        s = json.load(f)
    c = s["chosen"]
    if not os.path.isfile(c["path"]):
        raise SystemExit(f"the selected checkpoint {c['path']} is gone")
    now = sha256_file(c["path"])
    if now != c["sha256"]:
        raise SystemExit(f"the selected checkpoint {c['path']} is not the file that was selected "
                         f"({c['sha256'][:12]} selected, {now[:12]} on disk)")
    return f"{c['path']} {c['step']} {c['sha256']} {c['variant']} {s['selection_sha256']}"


def main(args):
    if args.cmd == "candidates":
        for c in candidates(args.results_dir, args.previous):
            print(f"{c['path']} {c['step']} {c['sha256']} {score_dir_name(c)}")
    elif args.cmd == "choose":
        meta = dict(kv.split("=", 1) for kv in args.meta)
        body = write_selection(args.results_dir, read_candidates(args.candidates),
                               args.scores_dir, args.out, meta, args.force)
        c = body["chosen"]
        print(f"SELECTED {c['path']} step={c['step']} variant={c['variant']} "
              f"psnr_raw={c['psnr_raw']:.3f} lpips_raw={c['lpips_raw']:.4f}")
    else:
        print(show(args.selection))
    return 0


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["candidates", "choose", "show"])
    p.add_argument("--results-dir", dest="results_dir", default="")
    p.add_argument("--previous", type=int, default=PREVIOUS_SNAPSHOTS)
    p.add_argument("--candidates", default="", help="the file `candidates` wrote")
    p.add_argument("--scores-dir", dest="scores_dir", default="")
    p.add_argument("--out", default="")
    p.add_argument("--meta", action="append", default=[], help="key=value recorded in selection.json")
    p.add_argument("--force", action="store_true")
    p.add_argument("--selection", default="")
    return p


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
