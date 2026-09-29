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
number of processes. They depend on the card, so every arena records the one it trained on (`run_card`): the server
whose path its certificate's source checkpoint has (Spiderman's `/sata2/` holds RTX A6000s, Superman's `/home/rohan/`
RTX A4000s).
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
                    r"(?P<seed>\d+)(?P<tag>[a-z]+)?(?:_(?P<variant>\w+))?$")
FULLFT_DIR_RE = re.compile(r"^map0*(\d+)$")      # the 8k-grid full fine-tunes sit under results/adapt_fullft_g8k/map<NN>
# a full-grid rerun of one map is preferred by its tag and seed: the seed-0 rerun, then seed 1, then the first seed-0
# run (the coordinator's order for map 15, whose seed-0 run diverged at step 7054 and has nine points)
FULL_GRID_PREFERENCE = (("rerun", 0), ("", 1), ("", 0))
BACKBONE_PREFIXES = (("unet", "unet"), ("pixart", "pixart"), ("sd35", "sd35"))
PREFERRED_VARIANTS = ("g8k", "")                 # the 8k grid over the base recipe; other variants are recipe tests
SERVER_CARDS = (("/sata2/", "A6000"), ("/home/rohan/", "A4000"))    # Spiderman, Superman (CLAUDE.md)
SOURCE_RE = re.compile(r"\bsource=(\S+)")


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
            "tag": m["tag"] or "", "variant": m["variant"] or ""}


def guards(rows):
    """{step: {directional, directional_ref, trainmap_psnr, trainmap_lpips, trainmap_psnr_dec}} from the rows that
    carry the guard reads (the directional check and the training maps' scene PSNR and LPIPS, raw and against the
    decoded ground truth), so the text can cite forgetting and the turn response at 0, 4k and 8k for every
    backbone; steps without a guard are absent."""
    out = {}
    for r in rows:
        if r.get("directional_correct_frac") is None and r.get("trainmap_psnr") is None:
            continue
        out[int(r["step"])] = {"directional": as_float(r.get("directional_correct_frac")),
                               "directional_ref": as_float(r.get("directional_ref_frac")),
                               "trainmap_psnr": as_float(r.get("trainmap_psnr")),
                               "trainmap_lpips": as_float(r.get("trainmap_lpips")),
                               "trainmap_psnr_dec": as_float(r.get("trainmap_psnr_dec"))}
    return dict(sorted(out.items()))


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


def in_distribution_gap(fresh_root, row, decoder):
    """(gap, (psnr, lpips)): the training maps' own gap to the reconstruction upper bound through `decoder` (pooled
    over the validation windows of the zero-shot row's home read, `home_<row>` or its `_tuned` twin) and their raw
    scene PSNR and LPIPS; (None, None) when no home read carries that decoder's columns. The gap is never borrowed
    from another decoder, whose upper bound is a different one."""
    sfx = "" if decoder == STOCK else f"_{decoder}"
    base = row[:-len("_tuned")] if row.endswith("_tuned") else row
    for name in ((f"{base}_tuned", base) if decoder == "tuned" else (base, f"{base}_tuned")):
        found = glob.glob(os.path.join(glob.escape(fresh_root), f"home_{name}", "**", "per_window.csv"),
                          recursive=True)
        if len(found) != 1:
            continue
        windows = read_windows(found[0])
        upper, psnr = mean_of(windows, f"scene_vae_psnr{sfx}"), mean_of(windows, f"scene_psnr_raw{sfx}")
        if upper is not None and psnr is not None:
            return upper - psnr, (psnr, mean_of(windows, f"scene_lpips_raw{sfx}"))
    return None, None


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


def run_card(events):
    """The card a run trained on ("A6000", "A4000"), from the server its certificate's source checkpoint lives on;
    None when the log has no certificate or names no known server."""
    line = next((e.get("line", "") for e in events if e.get("event") == "certificate"), "")
    m = SOURCE_RE.search(line)
    return next((card for prefix, card in SERVER_CARDS if m and m.group(1).startswith(prefix)), None)


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


def run_source(run_dir):
    """(checkpoint file name, weights) a run adapted, from its `config.json` (`source`, `source_weights`); the file
    name alone, since the same checkpoint sits at different paths on the two servers. None without a config."""
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        cfg = json.load(f)
    return (os.path.basename(str(cfg.get("source", ""))), cfg.get("source_weights")) if cfg.get("source") else None


def load_run(run_dir, name, backbone, arena, refs):
    """One run's arena entry {run, decoder, identity, grid, reads, guards, gpu_hours, card, source, grid_source},
    or None when its rows carry nothing usable; `grid_source` says "full" when every step of the run's own grid has
    a read, "partial" otherwise (the caller decides what a partial run stands in for)."""
    path = os.path.join(run_dir, "scores.jsonl")
    if not os.path.exists(path):
        return None
    rows = _live_rows(path)
    if not rows:
        return None
    decoder, identity = pick_decoder(rows)
    by_step = _by_step(rows, decoder)
    ref = refs.get(backbone, {}).get(decoder, {}).get(arena)
    reads = {s: rd for s, rs in by_step.items() for rd in [step_read(rs, run_dir, decoder, ref, s)] if rd}
    grid = sorted(int(s) for s in (next(iter(by_step.values()))[0].get("grid") or by_step))
    log_path = os.path.join(run_dir, "log.jsonl")
    events = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            events = [json.loads(line) for line in f if line.strip()]
    return {"run": name, "decoder": decoder, "identity": identity, "grid": grid, "reads": reads,
            "guards": guards(rows), "gpu_hours": gpu_hours(events), "card": run_card(events),
            "source": run_source(run_dir),
            "grid_source": "full" if all(s in reads for s in grid) else "partial"}


def load_full_grid(root, backbone, refs, k=8):
    """{arena: entry} of the full-grid reruns of `backbone` under `root` (ten reads from 0 to 8k, one run directory
    per map and seed, `load_run`): per map the first run in `FULL_GRID_PREFERENCE` order that has every read of its
    grid, else the run with the most reads (map 15: the seed-0 rerun when it is complete, else the seed-1 run when
    it is, else the nine-point seed-0 run); the entry's `grid_source` says which case, and `candidates` lists every
    run seen with its read count."""
    found = {}
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        meta = run_meta(name)
        if not meta or meta["backbone"] != backbone or meta["kind"] != "lora" or meta["k"] != k or meta["variant"]:
            continue
        entry = load_run(os.path.join(root, name), name, backbone, meta["arena"], refs)
        if entry:
            pref = FULL_GRID_PREFERENCE.index((meta["tag"], meta["seed"])) \
                if (meta["tag"], meta["seed"]) in FULL_GRID_PREFERENCE else len(FULL_GRID_PREFERENCE)
            found.setdefault(meta["arena"], []).append((pref, entry))
    out = {}
    for arena, runs in found.items():
        complete = sorted((p, e) for p, e in runs if e["grid_source"] == "full")
        best = complete[0][1] if complete else max(runs, key=lambda pe: (len(pe[1]["reads"]), -pe[0]))[1]
        out[arena] = {**best, "candidates": [(e["run"], len(e["reads"])) for _, e in sorted(runs, key=lambda pe: pe[0])]}
    return out


def load_fullft_g8k(root, refs):
    """{arena: entry} of the U-Net's full fine-tunes on the 8k grid under `root/map<NN>` (rows at 4k and 8k; the
    step-0 read is borrowed from the LoRA run of the same checkpoint by the caller)."""
    out = {}
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        m = FULLFT_DIR_RE.match(name)
        if not m:
            continue
        entry = load_run(os.path.join(root, name), name, "unet", int(m.group(1)), refs)
        if entry:
            out[int(m.group(1))] = {**entry, "grid_source": "g8k"}
    return out


def load_blocks(adapt_root, refs, k=8, seed=0, full_roots=None, fullft_root=None):
    """{"<backbone>_<kind>": {backbone, kind, decoder, identity, arenas: {arena: {run, decoder, identity, grid,
    reads, guards, gpu_hours, card, source, grid_source}}}} over the eight-episode, seed-0 runs under `adapt_root`
    (the block's decoder "mixed" when its arenas differ; `card` from `run_card`, `source` from `run_source`); `refs`
    is {backbone: {decoder: {arena: reference}}}. With `full_roots` ({backbone: root}) a map's coarse-grid run is
    replaced by its full-grid rerun (`load_full_grid`) whenever the rerun has every read of its grid, or when no
    rerun is complete but the best one has more reads than the coarse run (map 15's nine points); the entry keeps
    `grid_source` ("coarse", "full", "partial") and `coarse_run`, the run it replaced. With `fullft_root` the
    U-Net's full fine-tunes on the 8k grid replace the 4k-only originals for the maps they cover
    (`grid_source` "g8k"; the originals keep "coarse")."""
    runs = {}
    for name in sorted(os.listdir(adapt_root)) if os.path.isdir(adapt_root) else []:
        meta = run_meta(name)
        path = os.path.join(adapt_root, name, "scores.jsonl")
        if not meta or meta["k"] != k or meta["seed"] != seed or meta["tag"] \
                or meta["variant"] not in PREFERRED_VARIANTS or not os.path.exists(path):
            continue
        key = (meta["backbone"], meta["kind"], meta["arena"])
        rank = PREFERRED_VARIANTS.index(meta["variant"])
        if key not in runs or rank < runs[key][0]:
            runs[key] = (rank, name, meta)
    blocks = {}
    for (backbone, kind, arena), (_, name, _meta) in sorted(runs.items()):
        entry = load_run(os.path.join(adapt_root, name), name, backbone, arena, refs)
        if not entry:
            continue
        _place(blocks, backbone, kind, arena, {**entry, "grid_source": "coarse"})
    for backbone, root in (full_roots or {}).items():
        for arena, entry in load_full_grid(root, backbone, refs, k).items():
            block = blocks.get(f"{backbone}_lora", {})
            coarse = block.get("arenas", {}).get(arena)
            if entry["grid_source"] == "full" or coarse is None or len(entry["reads"]) > len(coarse["reads"]):
                _place(blocks, backbone, "lora", arena,
                       {**entry, "coarse_run": coarse["run"] if coarse else None})
    if fullft_root:
        for arena, entry in load_fullft_g8k(fullft_root, refs).items():
            coarse = blocks.get("unet_full", {}).get("arenas", {}).get(arena)
            _place(blocks, "unet", "full", arena, {**entry, "coarse_run": coarse["run"] if coarse else None})
    return blocks


def _place(blocks, backbone, kind, arena, entry):
    block = blocks.setdefault(f"{backbone}_{kind}", {"backbone": backbone, "kind": kind, "decoder": entry["decoder"],
                                                     "identity": entry["identity"], "arenas": {}})
    if block["decoder"] != entry["decoder"]:
        block["decoder"] = "mixed"
    block["arenas"][arena] = entry
