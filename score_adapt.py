"""
Score every checkpoint of a LoRA adaptation run on its map: one JSON row per (map, step, seed, weights).

    python score_adapt.py score --run-dir $R/adapt/unet200k_unseen_map17_s0 --hf-cache $HF \\
        --parquet-dir $E/raw_arnold_eval/unseen \\
        --trainmap-latents-dir $D/latents_arnold_dense_pertic_eval/val \\
        --trainmap-split $D/latents_arnold_dense_pertic_eval/split_val.json \\
        --trainmap-parquet-dir $D/raw_arnold_dense/arenas

    python score_adapt.py cost --scores $R/adapt/*/scores.jsonl --metric heldout_psnr_gain_paired --target 0.46

`score` reads each `adapter_<step>.pt` of the run (`adapt_wm.py`) with the live adapter and with its EMA
(`--weights`), and runs the existing evaluators unchanged:

  held-out      `eval_tf.py` one tic ahead on the split's recorded held-out windows (`--windows-file`),
                every step including 0, so the step-0 row is the zero-shot number on the same windows.
                Columns: raw PSNR and LPIPS, the persistence floor on the same windows, the gains over it
                (PSNR model minus floor; LPIPS floor minus model, so positive is better for both), both as
                the difference of eval_tf's means and as the mean of the per-window paired differences
                (`_paired`); the decoder-free LATENT SKILL S = -10 x mean_w log10 R_w, where R_w is the
                model's latent MSE over the copy-last latent MSE on window w (a geometric mean, in dB, that
                near-static windows cannot dominate), beside the arithmetic mean of R_w that eval_tf's summary
                reports and the ratio of the mean MSEs; decoded-frame numbers and the decoder's own ceiling
                (`vae_psnr`) too. The per-window rows are kept: eval_tf's `per_window.csv` and the derived
                `paired_windows.csv` (episode, start, R_w, its skill, the paired gains), both named in the row,
                so the paired means and their bootstrap over held-out episodes can be recomputed.
  directional   `directional_check.py` on the held-out episodes at `--guard-steps`: did the control
                response survive adaptation.
  training map  `eval_tf.py` on the training maps' validation set at `--guard-steps`: the forgetting guard,
                the same columns with the prefix `trainmap_`.

Rows are appended to `<run-dir>/scores.jsonl` as each (step, weights) finishes, and a row already there under
the same key (checkpoint hash, weights, windows, sampler, decoder, guard settings) is skipped, so an
interrupted scoring resumes. Both weights are scored at every checkpoint by default and every row names its
own (`weights`: live or ema); where the live adapter and its EMA are the same tensors (step 0 always, checked
by digest) the second row copies the first and says so (`same_tensors_as`). No plotting, no thresholds.

`cost` applies a target afterwards: per (run, map, seed, weights) the first scored step whose metric reaches
it. A curve that never reaches it within the steps scored is RIGHT-CENSORED: `censored: true`, `step: null`,
and `last_step_scored` says how far the curve was followed. It is never reported as reached at the last step.

Every evaluator read is also appended to W&B (`<run>-eval`, group `<run>`, tag `adapt_<weights>`) unless
`--no-wandb`.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time

CKPT_RE = re.compile(r"^adapter_(\d+)\.pt$")
WEIGHTS = ("live", "ema")
EVAL_FIELDS = (("psnr_raw", "psnr"), ("persist_psnr_raw", "persist_psnr"), ("lpips_raw", "lpips"),
               ("persist_lpips_raw", "persist_lpips"), ("latent_mse_ratio", "latent_mse_ratio"),
               ("latent_mse", "latent_mse"), ("copy_latent_mse", "copy_latent_mse"), ("psnr_dec", "psnr_dec"),
               ("lpips_dec", "lpips_dec"), ("copy_psnr_dec", "copy_psnr_dec"), ("vae_psnr", "vae_psnr"),
               ("vae_lpips", "vae_lpips"))
BACKBONE_PATH_FLAG = {"unet": "--sd-path", "pixart": "--pixart-path", "sd35": "--sd35-path"}


def _mean(metrics, key):
    v = metrics.get(key)
    if isinstance(v, dict):
        v = v.get("mean")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def eval_columns(prefix, metrics):
    """One eval_tf summary as flat columns: the means, the gains over persistence and the latent ratio."""
    row = {f"{prefix}_{dst}": _mean(metrics, src) for src, dst in EVAL_FIELDS}
    p, pf = row[f"{prefix}_psnr"], row[f"{prefix}_persist_psnr"]
    lp, lpf = row[f"{prefix}_lpips"], row[f"{prefix}_persist_lpips"]
    lm, cm = row[f"{prefix}_latent_mse"], row[f"{prefix}_copy_latent_mse"]
    row[f"{prefix}_psnr_gain"] = p - pf if p is not None and pf is not None else None
    row[f"{prefix}_lpips_gain"] = lpf - lp if lp is not None and lpf is not None else None
    row[f"{prefix}_latent_ratio_of_means"] = lm / cm if lm is not None and cm else None
    n = metrics.get("psnr_dec")
    row[f"{prefix}_windows"] = n.get("n") if isinstance(n, dict) else None
    return row


PAIRED_FIELDS = ("episode", "start", "index", "latent_mse", "copy_latent_mse", "latent_ratio", "latent_skill",
                 "psnr_gain", "lpips_gain")
WINDOW_COLUMNS = ("latent_skill", "latent_skill_windows", "latent_ratio_undefined", "latent_ratio_mean",
                  "psnr_gain_paired", "lpips_gain_paired", "paired_windows", "episodes", "per_window", "paired")


def _num(row, key):
    v = row.get(key)
    try:
        return float(v) if v not in (None, "") else float("nan")
    except ValueError:
        return float("nan")


def _finite_mean(values):
    import math
    ok = [v for v in values if math.isfinite(v)]
    return (sum(ok) / len(ok) if ok else None), len(ok)


def window_columns(prefix, per_window_csv, paired_csv):
    """Per-window paired quantities from eval_tf's `per_window.csv`, written to `paired_csv` and summarised.

    Per window w: R_w = model latent MSE / copy-last latent MSE, its skill -10 log10 R_w (dB; positive beats
    persistence), and the paired gains over persistence on the same window (PSNR model minus floor, LPIPS
    floor minus model). The row gets the LATENT SKILL S = -10 x mean_w log10 R_w, a geometric mean that a
    few near-static windows (tiny copy-last error, huge R_w) cannot dominate, beside the arithmetic mean of
    R_w that eval_tf's summary reports; and the mean paired gains. Windows where R_w is undefined (copying
    was exact) or zero are left out of S and counted. The per-window file keeps the episode of every window,
    so the paired means and their bootstrap over held-out episodes can be recomputed later.
    """
    import csv
    import math
    with open(per_window_csv) as f:
        rows = list(csv.DictReader(f))
    paired = []
    for r in rows:
        lm, cm = _num(r, "latent_mse"), _num(r, "copy_latent_mse")
        ratio = lm / cm if cm > 0 else float("nan")
        skill = -10.0 * math.log10(ratio) if ratio > 0 and math.isfinite(ratio) else float("nan")
        paired.append({"episode": int(r["episode"]), "start": int(r["start"]), "index": int(r["index"]),
                       "latent_mse": lm, "copy_latent_mse": cm, "latent_ratio": ratio, "latent_skill": skill,
                       "psnr_gain": _num(r, "psnr_raw") - _num(r, "persist_psnr_raw"),
                       "lpips_gain": _num(r, "persist_lpips_raw") - _num(r, "lpips_raw")})
    with open(paired_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(PAIRED_FIELDS))
        w.writeheader()
        w.writerows(paired)
    skill, n_skill = _finite_mean([p["latent_skill"] for p in paired])
    ratio_mean, _ = _finite_mean([p["latent_ratio"] for p in paired if math.isfinite(p["latent_skill"])])
    psnr_gain, n_paired = _finite_mean([p["psnr_gain"] for p in paired])
    lpips_gain, _ = _finite_mean([p["lpips_gain"] for p in paired])
    return {f"{prefix}_latent_skill": skill, f"{prefix}_latent_skill_windows": n_skill,
            f"{prefix}_latent_ratio_undefined": len(paired) - n_skill, f"{prefix}_latent_ratio_mean": ratio_mean,
            f"{prefix}_psnr_gain_paired": psnr_gain, f"{prefix}_lpips_gain_paired": lpips_gain,
            f"{prefix}_paired_windows": n_paired, f"{prefix}_episodes": len({p["episode"] for p in paired}),
            f"{prefix}_per_window": os.path.abspath(per_window_csv), f"{prefix}_paired": os.path.abspath(paired_csv)}


def directional_columns(summary):
    """The directional check's summary as flat columns."""
    motion = summary.get("motion") or {}
    return {"directional_correct_frac": summary.get("correct_frac"), "directional_ref_frac": summary.get("ref_frac"),
            "directional_motion_ratio": motion.get("ratio"), "directional_windows": summary.get("windows")}


EMPTY_GUARDS = {**{f"trainmap_{d}": None for _, d in EVAL_FIELDS},
                **{f"trainmap_{k}": None for k in ("psnr_gain", "lpips_gain", "latent_ratio_of_means", "windows")},
                **{f"trainmap_{k}": None for k in WINDOW_COLUMNS},
                **directional_columns({})}


def run_checkpoints(run_dir):
    """[(step, path)] of a run's adapter checkpoints, by the step in the file name, ascending."""
    out = []
    for p in glob.glob(os.path.join(run_dir, "adapter_*.pt")):
        m = CKPT_RE.match(os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    if not out:
        raise SystemExit(f"no adapter_<step>.pt under {run_dir}")
    return sorted(out)


def parse_steps(spec, steps):
    """The subset of `steps` a guard runs at: "all", "none", or a comma list of steps, "first" and "last"."""
    s = str(spec).strip().lower()
    if s == "all":
        return set(steps)
    if s in ("", "none"):
        return set()
    out = set()
    for tok in s.split(","):
        tok = tok.strip()
        out.add(min(steps) if tok == "first" else max(steps) if tok == "last" else int(tok))
    return out & set(steps)


def adaptation_cost(rows, metric, target, higher_is_better=True):
    """The adaptation cost of one curve (the rows of one run, map, seed and weights) at one target.

    The first scored step whose `metric` reaches `target` (>= when higher is better). A curve that never
    reaches it is right-censored: `step` is None and `censored` True, with `last_step_scored` saying how far
    it was followed; it is never reported as reached at its last step.
    """
    pts = sorted((int(r["step"]), float(r[metric])) for r in rows if r.get(metric) is not None)
    base = {"metric": metric, "target": target, "higher_is_better": bool(higher_is_better),
            "steps_scored": [s for s, _ in pts], "last_step_scored": pts[-1][0] if pts else None}
    for s, v in pts:
        if (v >= target) if higher_is_better else (v <= target):
            return {**base, "reached": True, "censored": False, "step": s, "value": v}
    return {**base, "reached": False, "censored": True, "step": None, "value": None,
            "last_value": pts[-1][1] if pts else None}


def tensor_digest(state):
    """A digest of a {name: tensor} dict, to tell whether the live adapter and its EMA are the same tensors."""
    h = hashlib.blake2b(digest_size=16)
    for n in sorted(state):
        t = state[n].detach().float().contiguous().cpu()
        h.update(n.encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()


def decoder_flags(a):
    flags = []
    if a.vae_path:
        flags += ["--vae-path", a.vae_path]
    if a.vae_subfolder:
        flags += ["--vae-subfolder", a.vae_subfolder]
    flags += ["--latent-scale", str(a.latent_scale)]
    if a.latent_shift is not None:
        flags += ["--latent-shift", str(a.latent_shift)]
    return flags


def model_flags(a, recipe):
    """The flags both evaluators need to rebuild the source graph."""
    bb = recipe["backbone"]
    flags = ["--backbone", bb, "--latent-channels", str(recipe["latent_channels"]),
             "--context-frames", str(recipe["context_frames"]), "--num-actions", str(recipe["num_actions"]),
             "--noise-buckets", str(recipe["noise_buckets"])]
    path = a.backbone_path or recipe.get("warm_start")
    if path and bb in BACKBONE_PATH_FLAG:
        flags += [BACKBONE_PATH_FLAG[bb], path]
    if a.hf_cache:
        flags += ["--hf-cache", a.hf_cache]
    return flags


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def run_eval_tf(argv):
    import eval_tf
    a = eval_tf.build_parser().parse_args(argv)
    eval_tf.main(a)
    with open(os.path.join(a.out_dir, "metrics.json")) as f:
        return json.load(f)


def run_directional(argv):
    import directional_check
    a = directional_check.build_parser().parse_args(argv)
    directional_check.main(a)
    with open(a.out) as f:
        return json.load(f)["summary"]


def cmd_score(a):
    """Score every (checkpoint, weights) of one adaptation run not yet in its scores file."""
    import torch
    from adapt_split import LEGACY_KEY, git_state, sha256_bytes
    from eval_identity import sha256_file
    ckpts = run_checkpoints(a.run_dir)
    steps = [s for s, _ in ckpts]
    first = torch.load(ckpts[0][1], map_location="cpu", weights_only=False)
    recipe, cert = first["recipe"], first["certificate"]
    split = a.split or first["split"]["path"]
    if sha256_bytes(split) != first["split"]["sha256"]:
        raise SystemExit(f"{split} is not the split this run trained on (SHA-256 differs from the certificate)")
    with open(split) as f:
        split_meta = json.load(f)["meta"]
    latents = a.latents_dir or split_meta["latents_dir"]
    eval_split = split
    if a.windows_key == LEGACY_KEY:
        eval_split = a.legacy_split or (split_meta.get("legacy") or {}).get("score_with_split")
        if not eval_split:
            raise SystemExit("the legacy windows are scored against the map's own split; pass --legacy-split")
    guard = parse_steps(a.guard_steps, steps)
    directional_at = guard if a.directional_windows > 0 else set()
    trainmap_at = guard if a.trainmap_latents_dir else set()
    if a.trainmap_latents_dir and not a.trainmap_split:
        raise SystemExit("--trainmap-latents-dir needs --trainmap-split")
    run = os.path.basename(os.path.normpath(a.run_dir))
    out_path = a.out or os.path.join(a.run_dir, "scores.jsonl")
    done = {json.dumps(r["key"], sort_keys=True): r for r in read_rows(out_path)}
    common = model_flags(a, recipe)
    dec = decoder_flags(a)
    decoder_tag = a.vae_path or "stock"
    code = git_state()
    print(f"scoring {run}: {len(ckpts)} checkpoint(s) x {list(a.weights)}, windows {a.windows_key}, guards at "
          f"{sorted(guard)}", flush=True)
    for step, path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        digests = {"live": tensor_digest(ck["adapter"]), "ema": tensor_digest(ck["adapter_ema"])}
        ck_sha = sha256_file(path)
        scored_here = {}
        for w in a.weights:
            key = {"ckpt_sha256": ck_sha, "step": step, "weights": w, "windows_key": a.windows_key,
                   "eval_seed": a.seed, "sampler_steps": a.steps, "decoder": decoder_tag,
                   "directional": a.directional_windows if step in directional_at else 0,
                   "trainmap": [a.trainmap_split, a.trainmap_windows] if step in trainmap_at else None,
                   "parquet": bool(a.parquet_dir)}
            ks = json.dumps(key, sort_keys=True)
            if ks in done:
                scored_here[w] = done[ks]
                continue
            other = next((o for o, r in scored_here.items() if digests[o] == digests[w]
                          and r["key"]["directional"] == key["directional"] and r["key"]["trainmap"] == key["trainmap"]),
                         None)
            base = {"run": run, "map": ck.get("map"), "set": split_meta.get("set"), "map_id": split_meta.get("map"),
                    "step": step, "seed": int(ck["adapt_args"]["seed"]), "weights": w, "backbone": recipe["backbone"],
                    "adapt_episodes": len(ck.get("episodes") or []),
                    "adapt_episodes_k": ck["adapt_args"].get("adapt_episodes_k", 0),
                    "rank": ck["adapter_config"]["rank"], "alpha": ck["adapter_config"]["alpha"],
                    "lora_mlp": ck["adapter_config"]["include_mlp"], "parts": ck["adapter_config"]["parts"],
                    "grid": cert["step_grid"], "source_sha256": ck["source"]["sha256"],
                    "source_weights": ck["source"]["weights"], "ckpt": os.path.abspath(path),
                    "split_sha256": ck["split"]["sha256"], "windows_key": a.windows_key, "eval_seed": a.seed,
                    "sampler_steps": a.steps, "decoder": decoder_tag, "key": key}
            if other is not None:
                src = scored_here[other]
                row = {**{k: v for k, v in src.items() if k not in base}, **base, "same_tensors_as": other,
                       "scored_at": time.time(), "code": code}
            else:
                out_dir = os.path.join(a.run_dir, "scores", f"step{step:07d}_{w}")
                use_ema = ["--use-ema"] if w == "ema" else []
                raw = ["--parquet-dir", a.parquet_dir] if a.parquet_dir else []
                held = run_eval_tf(["--ckpt", path, *common, *dec, *use_ema, *raw, "--latents-dir", latents,
                                    "--split", eval_split, "--subset", "val", "--windows-file", split,
                                    "--windows-key", a.windows_key, "--tic-stride", "1", "--steps", str(a.steps),
                                    "--seed", str(a.seed), "--batch-size", str(a.batch_size),
                                    "--num-workers", str(a.num_workers), "--save-images", "0",
                                    "--out-dir", os.path.join(out_dir, "heldout")])
                row = {**base, **eval_columns("heldout", held), **EMPTY_GUARDS,
                       **window_columns("heldout", os.path.join(out_dir, "heldout", "per_window.csv"),
                                        os.path.join(out_dir, "heldout", "paired_windows.csv"))}
                if step in directional_at:
                    row.update(directional_columns(run_directional(
                        ["--ckpt", path, *common, *dec, *use_ema, *raw, "--latents-dir", latents, "--split", split,
                         "--subset", "val", "--windows", str(a.directional_windows), "--steps", str(a.steps),
                         "--seed", str(a.seed), "--batch-size", str(a.batch_size), "--device", a.device,
                         "--run", run, "--out", os.path.join(out_dir, "directional.json")])))
                if step in trainmap_at:
                    tm_raw = ["--parquet-dir", a.trainmap_parquet_dir] if a.trainmap_parquet_dir else []
                    tm = run_eval_tf(["--ckpt", path, *common, *dec, *use_ema, *tm_raw,
                                      "--latents-dir", a.trainmap_latents_dir, "--split", a.trainmap_split,
                                      "--subset", "val", "--num-windows", str(a.trainmap_windows), "--tic-stride", "1",
                                      "--steps", str(a.steps), "--seed", str(a.seed), "--batch-size", str(a.batch_size),
                                      "--num-workers", str(a.num_workers), "--save-images", "0",
                                      "--out-dir", os.path.join(out_dir, "trainmap")])
                    row.update(eval_columns("trainmap", tm))
                    row.update(window_columns("trainmap", os.path.join(out_dir, "trainmap", "per_window.csv"),
                                              os.path.join(out_dir, "trainmap", "paired_windows.csv")))
                row.update({"out_dir": out_dir, "scored_at": time.time(), "code": code})
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            done[ks] = scored_here[w] = row
            print(f"SCORE_ADAPT {run} step={step} weights={w} heldout_psnr_gain_paired={row['heldout_psnr_gain_paired']} "
                  f"heldout_lpips_gain_paired={row['heldout_lpips_gain_paired']} "
                  f"heldout_latent_skill={row['heldout_latent_skill']} heldout_latent_ratio_mean="
                  f"{row['heldout_latent_ratio_mean']}" + (f" same_tensors_as={other}" if other else ""), flush=True)
            if a.wandb:
                from wandb_log import log_evaluation
                ns = argparse.Namespace(wandb_run=run, wandb_project=a.wandb_project, wandb_entity=a.wandb_entity,
                                        wandb_step=step)
                numeric = {k: v for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                           and k.startswith(("heldout_", "directional_", "trainmap_"))}
                log_evaluation(ns, f"adapt_{w}", numeric, ckpt=path, recorded_step=step, out_dir=a.run_dir)
    print(f"SCORE_ADAPT_DONE {run} rows={out_path}", flush=True)
    return 0


def cmd_cost(a):
    """Apply one target to scored curves: one JSON line per (run, map, seed, weights), censored where unreached."""
    rows = [r for p in a.scores for r in read_rows(p)]
    groups = {}
    for r in rows:
        groups.setdefault((r["run"], r["map"], r["seed"], r["weights"]), []).append(r)
    for (run, m, seed, w), rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        c = adaptation_cost(rs, a.metric, a.target, not a.lower_is_better)
        print(json.dumps({"run": run, "map": m, "seed": seed, "weights": w, **c}))
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="Score LoRA adaptation checkpoints; apply cost targets afterwards.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score every checkpoint of one adaptation run")
    s.add_argument("--run-dir", required=True, help="the adapt_wm.py results directory")
    s.add_argument("--weights", default="live,ema", type=lambda v: tuple(x for x in v.split(",") if x),
                   help="which adapter weights to score, from live,ema")
    s.add_argument("--split", default="", help="the adaptation split (default: the path the run recorded)")
    s.add_argument("--latents-dir", default="", help="the map's latents (default: the split's)")
    s.add_argument("--parquet-dir", default="", help="the map's raw recordings; without them there is no raw PSNR, "
                   "LPIPS or persistence floor, hence no gain")
    s.add_argument("--windows-key", default="held_out_windows", choices=["held_out_windows", "legacy_held_out_windows"],
                   help="the split's own held-out draw, or the distance study's draw restricted to it (cross-check)")
    s.add_argument("--legacy-split", default="", help="the map's distance-study split, for the legacy windows")
    s.add_argument("--guard-steps", default="0,last",
                   help="steps at which the directional check and the training-map read run: all, none, or a list "
                        "of steps with first/last")
    s.add_argument("--directional-windows", type=int, default=128, help="turning windows per direction (0 = skip)")
    s.add_argument("--device", default="cuda:0", help="the directional check's device")
    s.add_argument("--trainmap-latents-dir", default="", help="the training maps' validation latents (forgetting)")
    s.add_argument("--trainmap-split", default="", help="their split file (its val list)")
    s.add_argument("--trainmap-parquet-dir", default="")
    s.add_argument("--trainmap-windows", type=int, default=256)
    s.add_argument("--steps", type=int, default=10, help="sampler steps, the distance study's")
    s.add_argument("--seed", type=int, default=0, help="the evaluators' noise seed")
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--num-workers", type=int, default=0)
    s.add_argument("--vae-path", default="", help="decoder; default the stock sd-vae-ft-mse")
    s.add_argument("--vae-subfolder", default="")
    s.add_argument("--latent-scale", type=float, default=0.18215)
    s.add_argument("--latent-shift", type=float, default=None)
    s.add_argument("--backbone-path", default="", help="where the source architecture is read (default: its warm start)")
    s.add_argument("--hf-cache", default=None)
    s.add_argument("--out", default="", help="rows file (default <run-dir>/scores.jsonl)")
    s.add_argument("--no-wandb", dest="wandb", action="store_false")
    s.add_argument("--wandb-project", default="doomdit-nexttic")
    s.add_argument("--wandb-entity", default=None)
    c = sub.add_parser("cost", help="apply one target to scored curves (right-censored where unreached)")
    c.add_argument("--scores", nargs="+", required=True)
    c.add_argument("--metric", required=True, help="a row column, e.g. heldout_psnr_gain")
    c.add_argument("--target", type=float, required=True)
    c.add_argument("--lower-is-better", action="store_true")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "score":
        bad = [w for w in a.weights if w not in WEIGHTS]
        if bad or not a.weights:
            raise SystemExit(f"--weights takes live and/or ema, got {a.weights}")
    return {"score": cmd_score, "cost": cmd_cost}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
