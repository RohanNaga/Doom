"""
Any backbone's adaptation runs read into per-arena, per-step scene reads: raw where the data allow, labelled where
they do not. `paper/make_raw_figures.py` turns them into Table 2's per-backbone blocks and the per-backbone panel.

**Runs.** Every directory `<source>_<set>_map<NN>_r<rank>_k<k>_s<seed>[_<variant>]` under the adaptation root with a
`scores.jsonl` (`run_meta`): the backbone from the source's prefix (`unet`, `pixart`, `sd35`), a full fine-tune
when the rank is 0 (`adapt_wm.py --lora-rank 0 --full-parts all`), else a LoRA. Only the eight-episode, seed-0 runs
enter a block; of an arena's runs the 8k grid (`g8k`) is preferred over the base recipe and every other variant (the
recipe tests) is left out.

**The decoder** (`pick_decoder`) is the fine-tuned one (`tuned`) when the rows carry it, else the stock one, named
by the identity the rows record; U-Net and PixArt-alpha share the fine-tuned SD 1 decoder, SD 3.5 has its own.

**A read** (`step_read`), per scored step and live weights, takes the best quantity the run carries across every
row of that step (the guard rows share the curve's configuration and may be scored later) and every held-out
per-window file of that step (`step_files`), in order:

1. `per-window`, raw: a held-out per-window file with the decoder's raw scene columns: the prediction against the
   raw frame (`scene_psnr_raw`, `scene_lpips_raw`) and the reconstruction upper bound (`scene_vae_psnr`), duplicate
   windows left out. `score_adapt.py` writes it to `<run>/scores/step<NNNNNNN>_<weights>_<hash>/heldout/
   per_window.csv` and records the server path in the row; the files the rows name come first (found under the run
   when the recorded path is a server path), then any other held-out file of the step under `scores/`. The
   training-map guard's file beside it (`.../trainmap/per_window.csv`) is never a held-out read;
2. `row`, raw: the row's raw margin B (`heldout_B_<decoder>`) and gap C (`heldout_C_<decoder>`) with the arena's
   reference from a zero-shot read of the same windows (`zero_shot_refs`): PSNR = upper bound - C, LPIPS = raw
   persistence's LPIPS + B. The upper bound and persistence do not depend on the model, only on the decoder and the
   windows;
3. `per-window`, decoded: the decoded-reference columns (`scene_psnr_dec`, `scene_lpips_dec`), the prediction against
   the decoded ground truth, no upper bound;
4. `row`, decoded: PSNR = decoded copy-last's PSNR + A (`heldout_A_<decoder>`), no LPIPS.

**GPU-hours** to a step (`gpu_hours`): the training log's checkpoint time at that step minus its start, times the
number of processes.
"""
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from score_adapt import DUPLICATE_FLAGS, STOCK  # noqa: E402

RUN_RE = re.compile(r"^(?P<source>[a-z0-9_]+?)_(?P<set>arenas\d+)_map(?P<map>\d+)_r(?P<rank>\d+)_k(?P<k>\d+)_s"
                    r"(?P<seed>\d+)(?:_(?P<variant>\w+))?$")
BACKBONE_PREFIXES = (("unet", "unet"), ("pixart", "pixart"), ("sd35", "sd35"))
PREFERRED_VARIANTS = ("g8k", "")                 # the 8k grid over the base recipe; other variants are recipe tests


# ---------------------------------------------------------------------------------------------
# small readers
# ---------------------------------------------------------------------------------------------

def as_float(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def read_windows(path):
    """The rows of one eval_tf `per_window.csv`, duplicate windows left out, numbers as floats where they parse."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [{k: (as_float(v) if as_float(v) is not None else v) for k, v in r.items()} for r in rows
            if not any(as_float(r.get(c)) == 1 for c in DUPLICATE_FLAGS)]


def mean_of(rows, column):
    v = [r[column] for r in rows if isinstance(r.get(column), float) and math.isfinite(r[column])]
    return float(np.mean(v)) if v else None


def run_meta(name):
    """{source, backbone, kind, arena, rank, k, seed, variant} from a run directory's name, or None."""
    m = RUN_RE.match(name)
    if not m:
        return None
    backbone = next((b for prefix, b in BACKBONE_PREFIXES if m["source"].startswith(prefix)), None)
    if backbone is None:
        return None
    rank = int(m["rank"])
    return {"source": m["source"], "backbone": backbone, "kind": "full" if rank == 0 else "lora",
            "arena": int(m["map"]), "rank": rank, "k": int(m["k"]), "seed": int(m["seed"]),
            "variant": m["variant"] or ""}


def pick_decoder(rows):
    """(name, identity): the fine-tuned decoder (`tuned`) when the rows carry it, else the stock one."""
    names = {d for r in rows for d in (r.get("heldout_decoders") or [])}
    name = "tuned" if "tuned" in names else STOCK if STOCK in names or not names else sorted(names)[0]
    identity = next(((r.get("decoders") or {}).get(name, {}).get("identity") for r in rows
                     if (r.get("decoders") or {}).get(name)), None)
    return name, identity


def local_per_window(run_dir, recorded):
    """The per-window file a row names: the recorded path when it exists here, else the same `scores/<step dir>/
    <split>/per_window.csv` under the run directory (the rows record server paths)."""
    if not recorded:
        return None
    if os.path.exists(recorded):
        return recorded
    parts = str(recorded).replace("\\", "/").split("/")
    if len(parts) >= 4 and parts[-4] == "scores":
        local = os.path.join(run_dir, "scores", parts[-3], parts[-2], parts[-1])
        if os.path.exists(local):
            return local
    return None


def step_files(run_dir, rows, step=None, weights="live"):
    """The held-out per-window files one step may read, in preference order and without repeats: those its rows
    name (`rows` latest scored first), then any other `scores/step<NNNNNNN>_<weights>_*/heldout/per_window.csv` of
    the step under the run, newest first."""
    named = [p for r in rows for p in [local_per_window(run_dir, r.get("heldout_per_window"))] if p]
    found = []
    if step is not None:
        pattern = os.path.join(glob.escape(run_dir), "scores", f"step{int(step):07d}_{weights}_*", "heldout",
                               "per_window.csv")
        found = sorted(glob.glob(pattern), key=lambda p: (-os.path.getmtime(p), p))
    out, seen = [], set()
    for p in named + found:
        if os.path.realpath(p) not in seen:
            seen.add(os.path.realpath(p))
            out.append(p)
    return out


# ---------------------------------------------------------------------------------------------
# references and reads
# ---------------------------------------------------------------------------------------------

def zero_shot_refs(per_window_files, sfx):
    """{arena: {upper, upper_lpips, persist_lpips, copy_psnr_dec, psnr, lpips}} from zero-shot per-window files
    ({arena: path}) under the decoder whose columns end in `sfx`: the references a row-only read needs."""
    cols = {"upper": f"scene_vae_psnr{sfx}", "upper_lpips": f"scene_vae_lpips{sfx}",
            "persist_lpips": "scene_persist_lpips_raw", "copy_psnr_dec": f"scene_copy_psnr_dec{sfx}",
            "psnr": f"scene_psnr_raw{sfx}", "lpips": f"scene_lpips_raw{sfx}"}
    out = {}
    for arena, path in per_window_files.items():
        rows = read_windows(path)
        out[arena] = {k: mean_of(rows, c) for k, c in cols.items()}
    return out


def step_read(rows, run_dir, decoder, ref, step=None, weights="live"):
    """One scored step's read (the module docstring's order), or None when the run carries nothing usable. `rows`
    is the step's rows, latest scored first (or one row); with `step` the step's other held-out files count too."""
    rows = [rows] if isinstance(rows, dict) else list(rows)
    sfx = "" if decoder == STOCK else f"_{decoder}"
    ref = ref or {}
    decoded = None
    for path in step_files(run_dir, rows, step, weights):
        windows = read_windows(path)
        header = set(windows[0]) if windows else set()
        if f"scene_psnr_raw{sfx}" in header:
            return {"quantity": "raw", "source": "per-window", "psnr": mean_of(windows, f"scene_psnr_raw{sfx}"),
                    "lpips": mean_of(windows, f"scene_lpips_raw{sfx}"),
                    "upper": mean_of(windows, f"scene_vae_psnr{sfx}") or ref.get("upper"), "path": path}
        if decoded is None and f"scene_psnr_dec{sfx}" in header:
            decoded = {"quantity": "dec", "source": "per-window", "psnr": mean_of(windows, f"scene_psnr_dec{sfx}"),
                       "lpips": mean_of(windows, f"scene_lpips_dec{sfx}"), "upper": None, "path": path}
    for row in rows:
        b, c = as_float(row.get(f"heldout_B_{decoder}")), as_float(row.get(f"heldout_C_{decoder}"))
        if b is not None and c is not None and ref.get("upper") is not None and ref.get("persist_lpips") is not None:
            return {"quantity": "raw", "source": "row", "psnr": ref["upper"] - c, "lpips": ref["persist_lpips"] + b,
                    "upper": ref["upper"]}
    if decoded:
        return decoded
    for row in rows:
        a = as_float(row.get(f"heldout_A_{decoder}"))
        if a is not None and ref.get("copy_psnr_dec") is not None:
            return {"quantity": "dec", "source": "row", "psnr": ref["copy_psnr_dec"] + a, "lpips": None,
                    "upper": None}
    return None


def gpu_hours(events):
    """{step: GPU-hours to its checkpoint} from a training log's events: the checkpoint's time minus the start's,
    times the number of processes (`world` in the start event)."""
    start = next((e for e in events if e.get("event") == "start"), None)
    times = [e["time"] for e in events if isinstance(e.get("time"), (int, float))]
    if not times:
        return {}
    t0 = start["time"] if start and isinstance(start.get("time"), (int, float)) else min(times)
    world = (start or {}).get("world") or 1
    return {str(int(e["step"])): (e["time"] - t0) * world / 3600.0 for e in events
            if e.get("event") == "checkpoint" and isinstance(e.get("time"), (int, float)) and "step" in e}


def _live_rows(path, weights="live"):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if r.get("weights") == weights and "step" in r]


def _by_step(rows, decoder):
    """{step: rows, latest scored first} of the evaluation configuration with the most steps among those carrying
    the decoder's columns (the latest scored on a tie)."""
    column = f"heldout_A_{decoder}"
    groups = {}
    for r in rows:
        groups.setdefault(str(r.get("eval_fingerprint")), []).append(r)
    carrying = [rs for rs in groups.values()
                if any(as_float(r.get(c)) is not None for r in rs
                       for c in (column, f"heldout_B_{decoder}", f"heldout_C_{decoder}"))
                or any(r.get("heldout_per_window") for r in rs)] or list(groups.values())
    best = max(carrying, key=lambda rs: (len({r["step"] for r in rs}), max(str(r.get("scored_at", "")) for r in rs)))
    out = {}
    for r in sorted(best, key=lambda r: str(r.get("scored_at", "")), reverse=True):
        out.setdefault(int(r["step"]), []).append(r)
    return dict(sorted(out.items()))


def load_blocks(adapt_root, refs, k=8, seed=0):
    """{"<backbone>_<kind>": {backbone, kind, decoder, identity, arenas: {arena: {run, decoder, grid, reads,
    gpu_hours}}}} over the eight-episode, seed-0 runs under `adapt_root`; `refs` is {backbone: {decoder: {arena:
    reference}}}."""
    runs = {}
    for name in sorted(os.listdir(adapt_root)) if os.path.isdir(adapt_root) else []:
        meta = run_meta(name)
        path = os.path.join(adapt_root, name, "scores.jsonl")
        if not meta or meta["k"] != k or meta["seed"] != seed or meta["variant"] not in PREFERRED_VARIANTS \
                or not os.path.exists(path):
            continue
        key = (meta["backbone"], meta["kind"], meta["arena"])
        rank = PREFERRED_VARIANTS.index(meta["variant"])
        if key not in runs or rank < runs[key][0]:
            runs[key] = (rank, name, meta)
    blocks = {}
    for (backbone, kind, arena), (_, name, _meta) in sorted(runs.items()):
        run_dir = os.path.join(adapt_root, name)
        rows = _live_rows(os.path.join(run_dir, "scores.jsonl"))
        if not rows:
            continue
        decoder, identity = pick_decoder(rows)
        by_step = _by_step(rows, decoder)
        ref = refs.get(backbone, {}).get(decoder, {}).get(arena)
        reads = {s: rd for s, rs in by_step.items() for rd in [step_read(rs, run_dir, decoder, ref, s)] if rd}
        grid = sorted(int(s) for s in (next(iter(by_step.values()))[0].get("grid") or by_step))
        log_path = os.path.join(run_dir, "log.jsonl")
        hours = {}
        if os.path.exists(log_path):
            with open(log_path) as f:
                hours = gpu_hours([json.loads(line) for line in f if line.strip()])
        block = blocks.setdefault(f"{backbone}_{kind}", {"backbone": backbone, "kind": kind, "decoder": decoder,
                                                         "identity": identity, "arenas": {}})
        if block["decoder"] != decoder:
            block["decoder"] = "mixed"
        block["arenas"][arena] = {"run": name, "decoder": decoder, "grid": grid, "reads": reads, "gpu_hours": hours}
    return blocks
