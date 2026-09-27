"""
Score every checkpoint of a LoRA adaptation run on its map: one JSON row per (map, step, seed, weights, evaluation).

    python score_adapt.py score --run-dir $R/adapt/unet200k_arenas13_map17_r16_k8_s0 --hf-cache $HF \\
        --parquet-dir $E/raw_arnold_eval_v2/arenas13 --decoder tuned=$D/vae_decoder_tuned \\
        --trainmap-latents-dir $D/latents_arnold_dense_pertic_eval/val \\
        --trainmap-split $D/latents_arnold_dense_pertic_eval/split_val.json \\
        --trainmap-parquet-dir $D/raw_arnold_dense/arenas

    python score_adapt.py cost --scores $R/adapt/*/scores.jsonl --decoder tuned --home-value tuned=<training maps' A>

**The outcomes** (`.claude/analyses/cost-target-decision-2026-09-26.md`), per window and paired, on the scene
crop (rows 0 to 207, the frame without the HUD), under the stock decoder always (`_stock`) and under every
`--decoder NAME=PATH` given (`_NAME`, the tuned SD 1 decoder as `tuned`), all from `eval_tf.py`'s per-window
columns:

  A  decoded advantage over copy-last: scene PSNR of the decoded prediction minus scene PSNR of the decoded
     last context frame, both against the decoded true frame (`scene_psnr_dec - scene_copy_psnr_dec`). The
     headline curve and cost axis; higher is better. `A_full` is the same on the full frame, the check.
  B  perceptual margin against raw persistence: scene LPIPS of the decoded prediction minus scene LPIPS of the
     raw last frame, both against the raw true frame (`scene_lpips_raw - scene_persist_lpips_raw`); lower is
     better, zero is "ties copy-last".
  C  gap to the decoder's best render: scene PSNR of the decoded true latent minus scene PSNR of the decoded
     prediction, both against the raw frame (`scene_vae_psnr - scene_psnr_raw`); lower is better. A table
     column, not a cost axis (it tracks footage motion).

Beside them, decoder-free: the LATENT SKILL S = -10 x mean_w log10 R_w, where R_w is the model's latent MSE
over the copy-last latent MSE on window w (a geometric mean in dB), the arithmetic mean of R_w, and the
full-frame raw paired gains over persistence (PSNR model minus floor, LPIPS floor minus model). Every mean
leaves out the windows `eval_tf.py` flags as duplicates (`dup_raw`, `dup_latent`: the target repeats the last
context frame) and reports how many; nothing else is excluded, so a perfect window (R_w = 0) counts in the
arithmetic mean of R_w and only a value that is not finite (its skill, +inf) is left out of a mean, counted.

`score` reads each `adapter_<step>.pt` of the run (`adapt_wm.py`) with the live adapter and with its EMA
(`--weights`, both by default; every row names its own), and runs the existing evaluators unchanged:

  held-out      `eval_tf.py` one tic ahead on the split's recorded held-out windows (`--windows-file`) at every
                step including 0, so the step-0 row is the zero-shot number on the same windows.
  directional   `directional_check.py` on the held-out episodes: did the control response survive adaptation.
  training map  `eval_tf.py` on the training maps' validation windows: the forgetting guard, the same columns
                prefixed `trainmap_`.

The guards run at every grid step by default (`--guard-steps all`), and the training-map inputs are then
required; `--no-guards` (or `--guard-steps none`) is the explicit opt-out.

**Caching and evidence.** A row's `key` holds the checkpoint's SHA-256, the step, the weights, the guard
settings and the whole evaluation configuration: the windows, the split's hash, the latent directory, the
raw recordings, the sampler steps and seed, the batch size, the latent scale and shift, and every decoder's
name, path, subfolder and contents hash (`eval_fingerprint` is the hash of the configuration without the
step, the weights and the guards, so one curve shares one). A row already in `scores.jsonl` under the same key
is skipped, so an interrupted scoring resumes; any change to an input scores again. Every read writes into
`scores/step<N>_<weights>_<hash of the key>/`, so a rescore never touches the per-window files an earlier
row points to. Where the live adapter and its EMA are the same tensors (step 0 always, by digest) the second
row copies the first and says so (`same_tensors_as`). No plotting.

`cost` applies a target afterwards, one curve per (run, map, seed, weights, decoder, evaluation fingerprint):
the first scored step whose metric reaches it. By default the metric is A (`heldout_A`, one curve per
decoder) and the target is 50 percent of `--home-value`, the training maps' own A the caller passes
(`--fractions 0.25,0.5,1` for all three crossings); `--target` gives an absolute line instead. A curve that
never reaches it is RIGHT-CENSORED: `censored: true`, `step: null`, `last_step_scored` the last step with a
finite value; a value that is not finite is never an observation, and two different values at one step of
one curve are an error.

Every evaluator read is also appended to W&B (`<run>-eval`, group `<run>`, tag `adapt_<weights>`) unless
`--no-wandb`.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time

CKPT_RE = re.compile(r"^adapter_(\d+)\.pt$")
WEIGHTS = ("live", "ema")
STOCK = "stock"
EVAL_FIELDS = (("psnr_raw", "psnr"), ("persist_psnr_raw", "persist_psnr"), ("lpips_raw", "lpips"),
               ("persist_lpips_raw", "persist_lpips"), ("latent_mse_ratio", "latent_mse_ratio"),
               ("latent_mse", "latent_mse"), ("copy_latent_mse", "copy_latent_mse"), ("psnr_dec", "psnr_dec"),
               ("lpips_dec", "lpips_dec"), ("copy_psnr_dec", "copy_psnr_dec"), ("vae_psnr", "vae_psnr"),
               ("vae_lpips", "vae_lpips"))
BACKBONE_PATH_FLAG = {"unet": "--sd-path", "pixart": "--pixart-path", "sd35": "--sd35-path"}
DUPLICATE_FLAGS = ("dup_raw", "dup_latent")
# outcome: (model column, reference column, reference is decoder-free); the value is model minus reference
OUTCOMES = {"A": ("scene_psnr_dec", "scene_copy_psnr_dec", False),
            "A_full": ("psnr_dec", "copy_psnr_dec", False),
            "B": ("scene_lpips_raw", "scene_persist_lpips_raw", True),
            "C": ("scene_vae_psnr", "scene_psnr_raw", False)}
LOWER_IS_BETTER = ("B", "C")
DEFAULT_COST_METRIC = "heldout_A"


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


def _num(row, key):
    v = row.get(key)
    try:
        return float(v) if v not in (None, "") else float("nan")
    except ValueError:
        return float("nan")


def _finite_mean(values):
    ok = [v for v in values if math.isfinite(v)]
    return (sum(ok) / len(ok) if ok else None), len(ok)


def decoders_in(header):
    """The decoders a per_window.csv carries: `stock` (unsuffixed columns) and every `--decoder` suffix."""
    names = [STOCK] if "psnr_dec" in header else []
    names += [c[len("psnr_dec_"):] for c in header
              if c.startswith("psnr_dec_") and re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", c[len("psnr_dec_"):])]
    return names


def outcome_value(r, outcome, decoder):
    """One window's outcome under one decoder (NaN where a column is absent)."""
    model, ref, ref_free = OUTCOMES[outcome]
    sfx = "" if decoder == STOCK else f"_{decoder}"
    return _num(r, model + sfx) - _num(r, ref if ref_free else ref + sfx)


def window_columns(prefix, per_window_csv, paired_csv, decoders=None):
    """Per-window paired quantities from eval_tf's `per_window.csv`, written to `paired_csv` and summarised.

    Per window: the outcomes A, A_full, B and C under every decoder (module docstring), R_w = model latent
    MSE over copy-last latent MSE and its skill -10 log10 R_w, and the full-frame paired gains over raw
    persistence. The row gets, per decoder, the mean of each outcome; the LATENT SKILL S (mean skill over
    the windows where it is finite: a perfect window's +inf is counted in `latent_skill_infinite`, not
    averaged); the arithmetic mean of R_w over every window with a finite ratio, ratio 0 included; and the
    mean paired gains. Windows flagged as duplicates are left out of every mean and counted
    (`dup_excluded`). The per-window file keeps the episode of every window, so every paired mean and its
    bootstrap over held-out episodes can be recomputed.
    """
    import csv
    with open(per_window_csv) as f:
        reader = csv.DictReader(f)
        header, rows = list(reader.fieldnames or []), list(reader)
    names = list(decoders) if decoders is not None else decoders_in(header)
    flags = [c for c in DUPLICATE_FLAGS if c in header]
    paired = []
    for r in rows:
        lm, cm = _num(r, "latent_mse"), _num(r, "copy_latent_mse")
        ratio = lm / cm if cm > 0 else float("nan")
        skill = -10.0 * math.log10(ratio) if ratio > 0 and math.isfinite(ratio) else (
            float("inf") if ratio == 0 else float("nan"))
        p = {"episode": int(r["episode"]), "start": int(r["start"]), "index": int(r["index"]),
             "dup": int(any(_num(r, c) == 1 for c in flags)), "latent_mse": lm, "copy_latent_mse": cm,
             "latent_ratio": ratio, "latent_skill": skill,
             "psnr_gain": _num(r, "psnr_raw") - _num(r, "persist_psnr_raw"),
             "lpips_gain": _num(r, "persist_lpips_raw") - _num(r, "lpips_raw")}
        for d in names:
            for o in OUTCOMES:
                p[f"{o}_{d}"] = outcome_value(r, o, d)
        paired.append(p)
    fields = ["episode", "start", "index", "dup", "latent_mse", "copy_latent_mse", "latent_ratio", "latent_skill",
              "psnr_gain", "lpips_gain"] + [f"{o}_{d}" for d in names for o in OUTCOMES]
    with open(paired_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(paired)
    kept = [p for p in paired if not p["dup"]]
    skill, n_skill = _finite_mean([p["latent_skill"] for p in kept])
    ratio_mean, n_ratio = _finite_mean([p["latent_ratio"] for p in kept])
    psnr_gain, n_paired = _finite_mean([p["psnr_gain"] for p in kept])
    lpips_gain, _ = _finite_mean([p["lpips_gain"] for p in kept])
    out = {f"{prefix}_latent_skill": skill, f"{prefix}_latent_skill_windows": n_skill,
           f"{prefix}_latent_skill_infinite": sum(1 for p in kept if p["latent_skill"] == float("inf")),
           f"{prefix}_latent_ratio_mean": ratio_mean, f"{prefix}_latent_ratio_windows": n_ratio,
           f"{prefix}_latent_ratio_undefined": len(kept) - n_ratio,
           f"{prefix}_psnr_gain_paired": psnr_gain, f"{prefix}_lpips_gain_paired": lpips_gain,
           f"{prefix}_paired_windows": n_paired, f"{prefix}_dup_excluded": len(paired) - len(kept),
           f"{prefix}_dup_flags": flags, f"{prefix}_episodes": len({p["episode"] for p in paired}),
           f"{prefix}_decoders": names,
           f"{prefix}_per_window": os.path.abspath(per_window_csv), f"{prefix}_paired": os.path.abspath(paired_csv)}
    for d in names:
        for o in OUTCOMES:
            out[f"{prefix}_{o}_{d}"], n = _finite_mean([p[f"{o}_{d}"] for p in kept])
            if o == "A":
                out[f"{prefix}_outcome_windows_{d}"] = n
    return out


def directional_columns(summary):
    """The directional check's summary as flat columns."""
    motion = summary.get("motion") or {}
    return {"directional_correct_frac": summary.get("correct_frac"), "directional_ref_frac": summary.get("ref_frac"),
            "directional_motion_ratio": motion.get("ratio"), "directional_windows": summary.get("windows")}


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
    """The adaptation cost of ONE curve (the rows of one run, map, seed, weights, decoder and evaluation).

    The first scored step whose `metric` reaches `target` (>= when higher is better). Only finite values are
    observations: a NaN or infinite value is listed under `nonfinite_steps` and never extends the curve. A
    curve that never reaches the target is right-censored: `step` is None and `censored` True, with
    `last_step_scored` the last step with a finite value; it is never reported as reached at its last step.
    Two different values at one step mean two curves were pooled, which is an error.
    """
    by_step, nonfinite = {}, []
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        s, v = int(r["step"]), float(v)
        if not math.isfinite(v):
            nonfinite.append(s)
            continue
        if s in by_step and by_step[s] != v:
            raise ValueError(f"{metric} has two values at step {s} ({by_step[s]} and {v}): these rows are not one "
                             "curve; separate them by decoder and evaluation configuration")
        by_step[s] = v
    pts = sorted(by_step.items())
    base = {"metric": metric, "target": target, "higher_is_better": bool(higher_is_better),
            "steps_scored": [s for s, _ in pts], "last_step_scored": pts[-1][0] if pts else None,
            "nonfinite_steps": sorted(set(nonfinite))}
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


def fingerprint(obj):
    """SHA-256 of a JSON-able object, keys sorted: the identity of a configuration."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def decoder_identity(path, subfolder=""):
    """The contents identity of a decoder (`eval_identity.decoder_identity`), after resolving a local subfolder."""
    from eval_identity import decoder_identity as identity
    p = os.path.join(path, subfolder) if path and subfolder and os.path.isdir(os.path.join(path, subfolder)) else path
    return identity(p)


def decoder_flags(a, extra=True):
    """The decoder flags of the evaluators: the stock decoder and its latent contract, and with `extra` every
    `--decoder` (eval_tf.py only; directional_check.py measures the turn through the stock decoder)."""
    flags = []
    if a.vae_path:
        flags += ["--vae-path", a.vae_path]
    if a.vae_subfolder:
        flags += ["--vae-subfolder", a.vae_subfolder]
    flags += ["--latent-scale", str(a.latent_scale)]
    if a.latent_shift is not None:
        flags += ["--latent-shift", str(a.latent_shift)]
    for spec in (a.decoders or []) if extra else []:
        flags += ["--decoder", spec]
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


def _abs(path):
    return os.path.abspath(path) if path else ""


def evaluation_config(a, recipe, split_sha, eval_split, eval_split_sha, latents):
    """Everything a held-out read's numbers depend on besides the checkpoint, the step and the weights."""
    from eval_tf import parse_decoders
    extra = [{"name": n, "path": _abs(p) if os.path.exists(p) else p, "identity": decoder_identity(p)}
             for n, p in parse_decoders(a.decoders)]
    return {"windows_key": a.windows_key, "split_sha256": split_sha, "eval_split": _abs(eval_split),
            "eval_split_sha256": eval_split_sha, "latents_dir": _abs(latents), "parquet_dir": _abs(a.parquet_dir),
            "eval_seed": a.seed, "sampler_steps": a.steps, "batch_size": a.batch_size,
            "latent_scale": a.latent_scale, "latent_shift": a.latent_shift,
            "stock_decoder": {"path": a.vae_path, "subfolder": a.vae_subfolder,
                              "identity": decoder_identity(a.vae_path, a.vae_subfolder)},
            "decoders": extra, "backbone_path": a.backbone_path or recipe.get("warm_start")}


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
    """Score every (checkpoint, weights) of one adaptation run not yet in its scores file under this configuration."""
    import torch
    from adapt_split import LEGACY_KEY, git_state, sha256_bytes
    from eval_identity import sha256_file
    from eval_tf import parse_decoders
    ckpts = run_checkpoints(a.run_dir)
    steps = [s for s, _ in ckpts]
    first = torch.load(ckpts[0][1], map_location="cpu", weights_only=False)
    recipe, cert = first["recipe"], first["certificate"]
    split = a.split or first["split"]["path"]
    split_sha = sha256_bytes(split)
    if split_sha != first["split"]["sha256"]:
        raise SystemExit(f"{split} is not the split this run trained on (SHA-256 differs from the certificate)")
    with open(split) as f:
        split_meta = json.load(f)["meta"]
    latents = a.latents_dir or split_meta["latents_dir"]
    eval_split = split
    if a.windows_key == LEGACY_KEY:
        eval_split = a.legacy_split or (split_meta.get("legacy") or {}).get("score_with_split")
        if not eval_split:
            raise SystemExit("the legacy windows are scored against the map's own split; pass --legacy-split")
    guard = set() if a.no_guards else parse_steps(a.guard_steps, steps)
    if guard and not (a.trainmap_latents_dir and a.trainmap_split):
        raise SystemExit("the forgetting guard needs the training maps' validation set: pass --trainmap-latents-dir "
                         "and --trainmap-split (and --trainmap-parquet-dir), or opt out with --no-guards")
    directional_at = guard if a.directional_windows > 0 else set()
    trainmap_at = guard
    eval_cfg = evaluation_config(a, recipe, split_sha, eval_split, sha256_bytes(eval_split), latents)
    eval_fp = fingerprint(eval_cfg)
    tm_cfg = {"latents_dir": _abs(a.trainmap_latents_dir), "split": _abs(a.trainmap_split),
              "split_sha256": sha256_bytes(a.trainmap_split) if trainmap_at else None,
              "parquet_dir": _abs(a.trainmap_parquet_dir), "windows": a.trainmap_windows}
    run = os.path.basename(os.path.normpath(a.run_dir))
    out_path = a.out or os.path.join(a.run_dir, "scores.jsonl")
    done = {json.dumps(r["key"], sort_keys=True): r for r in read_rows(out_path)}
    used_dirs = {r.get("out_dir") for r in done.values() if not r.get("same_tensors_as")}
    common = model_flags(a, recipe)
    dec = decoder_flags(a)
    names = [STOCK] + [n for n, _ in parse_decoders(a.decoders)]
    code = git_state()
    print(f"scoring {run}: {len(ckpts)} checkpoint(s) x {list(a.weights)}, windows {a.windows_key}, decoders "
          f"{names}, guards at {sorted(guard)}, evaluation {eval_fp[:12]}", flush=True)
    for step, path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        digests = {"live": tensor_digest(ck["adapter"]), "ema": tensor_digest(ck["adapter_ema"])}
        ck_sha = sha256_file(path)
        scored_here = {}
        for w in a.weights:
            guards = {"directional": ({"windows": a.directional_windows} if step in directional_at else None),
                      "trainmap": (tm_cfg if step in trainmap_at else None)}
            key = {"ckpt_sha256": ck_sha, "step": step, "weights": w, "eval": eval_cfg, "guards": guards}
            ks = json.dumps(key, sort_keys=True)
            if ks in done:
                scored_here[w] = done[ks]
                continue
            other = next((o for o, r in scored_here.items() if digests[o] == digests[w]
                          and r["key"]["guards"] == guards), None)
            base = {"run": run, "map": ck.get("map"), "set": split_meta.get("set"), "map_id": split_meta.get("map"),
                    "step": step, "seed": int(ck["adapt_args"]["seed"]), "weights": w, "backbone": recipe["backbone"],
                    "adapt_episodes": len(ck.get("episodes") or []),
                    "adapt_episodes_k": ck["certificate"].get("adapt_episodes_k"),
                    "rank": ck["adapter_config"]["rank"], "alpha": ck["adapter_config"]["alpha"],
                    "lora_mlp": ck["adapter_config"]["include_mlp"], "parts": ck["adapter_config"]["parts"],
                    "grid": cert["step_grid"], "source_sha256": ck["source"]["sha256"],
                    "source_weights": ck["source"]["weights"], "ckpt": os.path.abspath(path),
                    "split_sha256": ck["split"]["sha256"], "windows_key": a.windows_key, "eval_seed": a.seed,
                    "sampler_steps": a.steps, "decoder": STOCK if not a.vae_path else a.vae_path,
                    "decoders": {STOCK: eval_cfg["stock_decoder"], **{d["name"]: d for d in eval_cfg["decoders"]}},
                    "eval_fingerprint": eval_fp, "key": key}
            if other is not None:
                src = scored_here[other]
                row = {**{k: v for k, v in src.items() if k not in base}, **base, "same_tensors_as": other,
                       "scored_at": time.time(), "code": code}
            else:
                # the key's hash names the directory, so no later configuration writes into this one
                out_dir = os.path.join(a.run_dir, "scores", f"step{step:07d}_{w}_{fingerprint(key)[:16]}")
                if out_dir in used_dirs:
                    raise SystemExit(f"{out_dir} already holds the evidence of an earlier row; refusing to overwrite it")
                use_ema = ["--use-ema"] if w == "ema" else []
                raw = ["--parquet-dir", a.parquet_dir] if a.parquet_dir else []
                held = run_eval_tf(["--ckpt", path, *common, *dec, *use_ema, *raw, "--latents-dir", latents,
                                    "--split", eval_split, "--subset", "val", "--windows-file", split,
                                    "--windows-key", a.windows_key, "--tic-stride", "1", "--steps", str(a.steps),
                                    "--seed", str(a.seed), "--batch-size", str(a.batch_size),
                                    "--num-workers", str(a.num_workers), "--save-images", "0",
                                    "--out-dir", os.path.join(out_dir, "heldout")])
                held_cols = window_columns("heldout", os.path.join(out_dir, "heldout", "per_window.csv"),
                                           os.path.join(out_dir, "heldout", "paired_windows.csv"), names)
                row = {**base, **eval_columns("heldout", held), **held_cols,
                       **{k.replace("heldout_", "trainmap_", 1): None
                          for k in list(eval_columns("heldout", {})) + list(held_cols)},
                       **directional_columns({})}
                if step in directional_at:
                    row.update(directional_columns(run_directional(
                        ["--ckpt", path, *common, *decoder_flags(a, extra=False), *use_ema, *raw, "--latents-dir", latents, "--split", split, "--subset", "val",
                         "--windows", str(a.directional_windows), "--steps", str(a.steps), "--seed", str(a.seed),
                         "--batch-size", str(a.batch_size), "--device", a.device, "--run", run,
                         "--out", os.path.join(out_dir, "directional.json")])))
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
                                              os.path.join(out_dir, "trainmap", "paired_windows.csv"), names))
                row.update({"out_dir": out_dir, "scored_at": time.time(), "code": code})
                used_dirs.add(out_dir)
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            done[ks] = scored_here[w] = row
            outcomes = " ".join(f"heldout_A_{d}={row.get(f'heldout_A_{d}')}" for d in names)
            print(f"SCORE_ADAPT {run} step={step} weights={w} {outcomes} heldout_latent_skill="
                  f"{row['heldout_latent_skill']} evaluation={eval_fp[:12]}"
                  + (f" same_tensors_as={other}" if other else ""), flush=True)
            if a.wandb:
                from wandb_log import log_evaluation
                ns = argparse.Namespace(wandb_run=run, wandb_project=a.wandb_project, wandb_entity=a.wandb_entity,
                                        wandb_step=step)
                numeric = {k: v for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                           and k.startswith(("heldout_", "directional_", "trainmap_"))}
                log_evaluation(ns, f"adapt_{w}", numeric, ckpt=path, recorded_step=step, out_dir=a.run_dir)
    print(f"SCORE_ADAPT_DONE {run} rows={out_path}", flush=True)
    return 0


def lower_is_better(metric, direction="auto"):
    """The direction a target is crossed in: explicit, or by the outcome (B and C and the latent ratio are lower)."""
    if direction != "auto":
        return direction == "lower"
    return bool(re.search(r"(^|_)(B|C)(_|$)", metric)) or "latent_ratio" in metric


def parse_home(values):
    """`--home-value [decoder=]value`, repeatable -> {decoder or None: value}."""
    out = {}
    for v in values or []:
        name, sep, num = str(v).partition("=")
        out[name if sep else None] = float(num if sep else name)
    return out


def curve_series(rows, metric, decoder=None):
    """[(column, decoder)] the metric names: the column itself, or `<metric>_<decoder>` for every decoder rows carry."""
    cols = {k for r in rows for k in r}
    if metric in cols:
        return [(metric, None)]
    names = sorted({d for r in rows for d in (r.get("decoders") or {})})
    out = [(f"{metric}_{d}", d) for d in names if f"{metric}_{d}" in cols and decoder in (None, d)]
    if not out:
        raise SystemExit(f"no column {metric} or {metric}_<decoder> in these rows"
                         + (f" for decoder {decoder}" if decoder else ""))
    return out


def cmd_cost(a):
    """One JSON line per (curve, target): the first step that reaches it, or right-censored."""
    rows = [r for p in a.scores for r in read_rows(p)]
    series = curve_series(rows, a.metric, a.decoder)
    homes = parse_home(a.home_value)
    fractions = [float(x) for x in str(a.fractions).split(",") if x.strip()]
    if a.target is None and not homes:
        raise SystemExit("give the line: --home-value (the training maps' own value of the metric; the cost is "
                         "reached at --fractions of it) or an absolute --target")
    groups = {}
    for column, d in series:
        for r in rows:
            if column not in r:
                continue
            key = (str(r["run"]), str(r["map"]), str(r["seed"]), str(r["weights"]), str(d or r.get("decoder")),
                   str(r.get("eval_fingerprint")), str(r.get("windows_key")), str(r.get("sampler_steps")),
                   str(r.get("eval_seed")), column)
            groups.setdefault(key, []).append(r)
    lower = lower_is_better(a.metric, a.direction)
    decoders = {k[4] for k in groups}
    for key, rs in sorted(groups.items()):
        run, m, seed, w, dec, fp, _, _, _, column = key
        if a.decoder and dec != a.decoder:
            continue
        if a.target is not None:
            lines = [(a.target, None, None)]
        else:
            home = homes.get(dec)
            if home is None and None in homes:
                if len(decoders) > 1 and not a.decoder:
                    raise SystemExit(f"these rows hold curves for decoders {sorted(decoders)}; a bare --home-value "
                                     "cannot be every decoder's: pass <decoder>=<value>, or --decoder")
                home = homes[None]
            if home is None:
                raise SystemExit(f"no --home-value for decoder {dec}; pass {dec}=<value>")
            lines = [(f * home, f, home) for f in fractions]
        for target, fraction, home in lines:
            c = adaptation_cost(rs, column, target, not lower)
            print(json.dumps({"run": rs[0]["run"], "map": rs[0]["map"], "seed": rs[0]["seed"], "weights": w,
                              "decoder": dec, "eval_fingerprint": None if fp == "None" else fp,
                              "fraction": fraction, "home_value": home, **c}))
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="Score LoRA adaptation checkpoints; apply cost targets afterwards.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score every checkpoint of one adaptation run")
    s.add_argument("--run-dir", required=True, help="the adapt_wm.py results directory")
    s.add_argument("--weights", default="live,ema", type=lambda v: tuple(x for x in v.split(",") if x),
                   help="which adapter weights to score, from live,ema (both by default; each row names its own)")
    s.add_argument("--split", default="", help="the adaptation split (default: the path the run recorded)")
    s.add_argument("--latents-dir", default="", help="the map's latents (default: the split's)")
    s.add_argument("--parquet-dir", default="", help="the map's raw recordings; without them there is no B, C, raw "
                   "PSNR, LPIPS or persistence floor")
    s.add_argument("--decoder", dest="decoders", action="append", default=None, metavar="[NAME=]PATH",
                   help="a further decoder, passed to eval_tf.py (repeatable); its outcomes are suffixed _NAME. The "
                        "tuned SD 1 decoder as tuned=PATH; the stock decoder (--vae-path) is always scored as _stock")
    s.add_argument("--windows-key", default="held_out_windows", choices=["held_out_windows", "legacy_held_out_windows"],
                   help="the split's own held-out draw, or the distance study's draw restricted to it (cross-check)")
    s.add_argument("--legacy-split", default="", help="the map's distance-study split, for the legacy windows")
    s.add_argument("--guard-steps", default="all",
                   help="steps at which the directional check and the training-map read run: all (default), none, or "
                        "a list of steps with first/last")
    s.add_argument("--no-guards", action="store_true", help="run neither guard (the explicit opt-out)")
    s.add_argument("--directional-windows", type=int, default=128, help="turning windows per direction (0 = skip)")
    s.add_argument("--device", default="cuda:0", help="the directional check's device")
    s.add_argument("--trainmap-latents-dir", default="", help="the training maps' validation latents (forgetting); "
                   "required with guards")
    s.add_argument("--trainmap-split", default="", help="their split file (its val list); required with guards")
    s.add_argument("--trainmap-parquet-dir", default="")
    s.add_argument("--trainmap-windows", type=int, default=256)
    s.add_argument("--steps", type=int, default=10, help="sampler steps, the distance study's")
    s.add_argument("--seed", type=int, default=0, help="the evaluators' noise seed")
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--num-workers", type=int, default=0)
    s.add_argument("--vae-path", default="", help="the stock decoder; default sd-vae-ft-mse")
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
    c.add_argument("--metric", default=DEFAULT_COST_METRIC,
                   help="a row column, or a prefix whose _<decoder> columns are one curve each (default heldout_A)")
    c.add_argument("--decoder", default=None, help="only this decoder's curves (stock, tuned, ...)")
    c.add_argument("--home-value", action="append", default=[], metavar="[DECODER=]VALUE",
                   help="the training maps' own value of the metric under that decoder; the lines are --fractions of it")
    c.add_argument("--fractions", default="0.5", help="fractions of the home value to cross (default 0.5; e.g. 0.25,0.5,1)")
    c.add_argument("--target", type=float, default=None, help="an absolute line instead of a fraction of --home-value")
    c.add_argument("--direction", choices=["auto", "higher", "lower"], default="auto",
                   help="which way the line is crossed; auto: lower for B, C and the latent ratio, higher otherwise")
    return p


def main(argv=None):
    """Dispatch `score` or `cost`."""
    a = build_parser().parse_args(argv)
    if a.cmd == "score":
        bad = [w for w in a.weights if w not in WEIGHTS]
        if bad or not a.weights:
            raise SystemExit(f"--weights takes live and/or ema, got {a.weights}")
    return {"score": cmd_score, "cost": cmd_cost}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
