"""
Build every LaTeX table of the paper (booktabs tabulars, one file each, for \\input) from the
result directories.

    python paper/make_tables.py --results-root paper/results_mirror --out paper/tables

Tables:
  main.tex       Table 1: rows of the "main" group; exposure, params, seen and unseen teacher-forced
                 PSNR / LPIPS, rollout PSNR@64 / LPIPS@64, FVD16/32, IDM top-1; copy-last, copy-seed and
                 VAE-ceiling reference rows.
  transfer.tex   Table 2: seen mean, unseen-2 mean, unseen-15 mean (placeholder corpus), seen-to-unseen drop.
  ema.tex        appendix: live versus EMA teacher-forced numbers.
  seeds.tex      appendix: both DiT seeds on every column plus the held-out v-loss.
  cm.tex         appendix: compute-matched 2.5k-update rows.
  grid.tex       appendix: historical grid-label checkpoints.
  horizons.tex   appendix: per-horizon rollout values with 95% CIs where the audit provides them.
  blur.tex       appendix: blur-control sweep grid.
  late_drop.tex  appendix: horizon-32-to-64 late-drop distribution.
  paired.tex     appendix: paired U-Net minus DiT differences from tmp/audit/summary.json.
  idm.tex        appendix: IDM judge quality (K=8 and K=2) and per-row rollout agreement.
  knobs.tex      appendix: the twelve-cell knob grid, ordered by rollout LPIPS@64.

Also writes rows_grid.json next to the tables: one record per knob-grid cell with the knob the
launcher turns, the recipe fields that cell's own config.json recorded, its metrics, and the
LPIPS@64 ranking. A cell whose recorded recipe contradicts its knob is warned about.

Every number is read from the files; a missing artifact renders as n/a with a warning.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperdata as pd  # noqa: E402
from paperdata import NA, fmt, fmt_ci, fmt_params  # noqa: E402

DASH = "--"
HORIZONS = (8, 16, 32, 64)
SIGMAS = (0.0, 0.5, 1.0, 2.0, 4.0)


# ---- LaTeX helpers ----------------------------------------------------------------------------

def tabular(colspec, header, body, note=None):
    """A booktabs tabular. `header` and `body` are lists of cell lists; a body entry equal to
    "midrule" inserts a rule. `note` is appended below the tabular in footnotesize."""
    lines = [f"\\begin{{tabular}}{{{colspec}}}\\toprule"]
    lines += [" & ".join(h) + " \\\\" for h in header]
    lines[-1] += "\\midrule"
    for row in body:
        lines.append("\\midrule" if row == "midrule" else " & ".join(row) + " \\\\")
    lines.append("\\bottomrule\\end{tabular}")
    if note:
        lines.append("\\\\[2pt]{\\footnotesize " + note + "}")
    return "\n".join(lines) + "\n"


def bold_best(cells, values, higher):
    """Wrap the best value's cell in \\textbf; `values` are floats or None aligned with `cells`."""
    present = [(v, i) for i, v in enumerate(values) if v is not None]
    if len(present) < 2:
        return cells
    best = (max if higher else min)(present)[1]
    cells = list(cells)
    cells[best] = f"\\textbf{{{cells[best]}}}"
    return cells


def write(out, name, text):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    open(path, "w").write(text)
    print("wrote", path)


# ---- shared row columns -----------------------------------------------------------------------

MAIN_COLS = (  # (key, higher_is_better, decimals)
    ("seen_psnr", True, 2), ("seen_lpips", False, 3), ("unseen_psnr", True, 2), ("unseen_lpips", False, 3),
    ("psnr64", True, 2), ("lpips64", False, 3), ("fvd16", False, 0), ("fvd32", False, 0), ("idm", True, 3),
)


def row_values(r):
    return {
        "seen_psnr": r.tf_mean("seen", "psnr_raw"), "seen_lpips": r.tf_mean("seen", "lpips_raw"),
        "unseen_psnr": r.tf_mean("unseen", "psnr_raw"), "unseen_lpips": r.tf_mean("unseen", "lpips_raw"),
        "psnr64": r.rollout_at("psnr", 64), "lpips64": r.rollout_at("lpips", 64),
        "fvd16": r.fvd(16), "fvd32": r.fvd(32), "idm": r.idm_top1(),
    }


def model_block(rows, with_meta=True, with_vloss=False):
    """Formatted model rows with per-column bold on the best finished row."""
    vals = [row_values(r) for r in rows]
    cols = {k: [fmt(v[k], nd) for v in vals] for k, _, nd in MAIN_COLS}
    for k, higher, _ in MAIN_COLS:
        cols[k] = bold_best(cols[k], [v[k] for v in vals], higher)
    out = []
    for i, r in enumerate(rows):
        cells = [r.label]
        if with_meta:
            cells += [r.spec.get("exposure", NA), fmt_params(r.params)]
        cells += [cols["seen_psnr"][i], cols["seen_lpips"][i], cols["unseen_psnr"][i], cols["unseen_lpips"][i],
                  cols["psnr64"][i], cols["lpips64"][i], f"{cols['fvd16'][i]} / {cols['fvd32'][i]}",
                  cols["idm"][i] + ("$^{" + r.spec["idm_note"] + "}$" if r.spec.get("idm_note") else "")]
        if with_vloss:
            cells.append(fmt(r.vloss_final, 4))
        out.append(cells)
    return out


def reference_rows(rows, pad):
    """copy-last / copy-seed and VAE-ceiling rows; `pad` is the number of leading meta columns."""
    lead = [DASH] * pad
    copy_seen, copy_unseen = pd.reference(rows, "seen", "copy_psnr_raw"), pd.reference(rows, "unseen", "copy_psnr_raw")
    vae = [pd.reference(rows, c, k) for c in ("seen", "unseen") for k in ("vae_psnr", "vae_lpips")]
    seed_psnr = next((r.copy_seed_psnr(64) for r in rows if r.copy_seed_psnr(64) is not None), None)
    seed_lpips = next((r.copy_seed_lpips(64) for r in rows if r.copy_seed_lpips(64) is not None), None)
    idm_real = next((r.idm_real_top1() for r in rows if r.idm_real_top1() is not None), None)
    if seed_psnr is None:
        pd.warn("copy-seed PSNR@64 reference missing (no drift.json copy_seed_psnr)")
    if seed_lpips is None:
        pd.warn("copy-seed LPIPS@64 reference missing (needs audit.json blur_sweep)")
    return [
        ["copy-last / copy-seed"] + lead + [fmt(copy_seen), DASH, fmt(copy_unseen), DASH, fmt(seed_psnr), fmt(seed_lpips, 3), DASH, DASH],
        ["VAE ceiling"] + lead + [fmt(vae[0]), fmt(vae[1], 3), fmt(vae[2]), fmt(vae[3], 3), DASH, DASH, DASH, fmt(idm_real, 3) + "$^*$"],
    ]


MAIN_HEADER = [
    ["", "", "", "\\multicolumn{2}{c}{Seen maps}", "\\multicolumn{2}{c}{Unseen maps}", "\\multicolumn{2}{c}{Rollout @64}", "", ""],
    ["Start", "Pretraining exposure", "Params", "PSNR$\\uparrow$", "LPIPS$\\downarrow$", "PSNR$\\uparrow$", "LPIPS$\\downarrow$",
     "PSNR$\\uparrow$", "LPIPS$\\downarrow$", "FVD$_{16/32}\\downarrow$", "IDM$\\uparrow$"],
]
MAIN_NOTE = ("$^*$IDM agreement on the real counterparts of the same rollouts. $^\\S$the video row's rollouts pass "
             "through the SD encoder to reach the judge; read against its own real reference. n/a: artifact not "
             "yet produced.")


def table_main(rows):
    body = reference_rows(rows, 2) + ["midrule"] + model_block(rows)
    return tabular("lllcccccccc", MAIN_HEADER, body, MAIN_NOTE)


def table_simple_grid(rows, with_vloss=False):
    header = [["", "\\multicolumn{2}{c}{Seen maps}", "\\multicolumn{2}{c}{Unseen maps}", "\\multicolumn{2}{c}{Rollout @64}", "", ""]
              + (["v-loss"] if with_vloss else []),
              ["Row", "PSNR", "LPIPS", "PSNR", "LPIPS", "PSNR", "LPIPS", "FVD$_{16/32}$", "IDM"] + (["held-out"] if with_vloss else [])]
    body = reference_rows(rows, 0) + ["midrule"] + model_block(rows, with_meta=False, with_vloss=with_vloss)
    return tabular("l" + "c" * (8 + int(with_vloss)), header, body)


def table_transfer(rows):
    header = [["", "\\multicolumn{2}{c}{Training maps (seen)}", "\\multicolumn{2}{c}{Unseen maps 16--17}",
               "\\multicolumn{2}{c}{15 further maps}", "Drop"],
              ["Row", "PSNR [95\\% CI]", "LPIPS", "PSNR [95\\% CI]", "LPIPS", "PSNR", "LPIPS", "PSNR (dB)"]]
    body = []
    for r in rows:
        seen, unseen = r.tf_mean("seen", "psnr_raw"), r.tf_mean("unseen", "psnr_raw")
        drop = None if seen is None or unseen is None else seen - unseen
        body.append([r.label,
                     f"{fmt(seen)} {fmt_ci(r.tf_ci('seen', 'psnr_raw'))}".strip(), fmt(r.tf_mean("seen", "lpips_raw"), 3),
                     f"{fmt(unseen)} {fmt_ci(r.tf_ci('unseen', 'psnr_raw'))}".strip(), fmt(r.tf_mean("unseen", "lpips_raw"), 3),
                     fmt(r.tf_mean("unseen15", "psnr_raw")), fmt(r.tf_mean("unseen15", "lpips_raw"), 3), fmt(drop)])
    note = "Teacher-forced, live weights, against lossless frames. Intervals are episode-level bootstrap 95\\% CIs from the audit. Drop is seen minus unseen PSNR. The 15-map corpus is scored when its evaluation exists."
    return tabular("lccccccc", header, body, note)


def table_ema(rows):
    header = [["", "\\multicolumn{2}{c}{Live}", "\\multicolumn{2}{c}{EMA}", "\\multicolumn{2}{c}{EMA $-$ live}"],
              ["Row", "PSNR", "LPIPS", "PSNR", "LPIPS", "PSNR", "LPIPS"]]
    body = []
    for r in rows:
        lp, ll = r.tf_mean("seen", "psnr_raw"), r.tf_mean("seen", "lpips_raw")
        ep, el = r.tf_mean("seen_ema", "psnr_raw"), r.tf_mean("seen_ema", "lpips_raw")
        dp = None if lp is None or ep is None else ep - lp
        dl = None if ll is None or el is None else el - ll
        body.append([r.label, fmt(lp), fmt(ll, 3), fmt(ep), fmt(el, 3), fmt(dp), fmt(dl, 3)])
    return tabular("lcccccc", header, body, "Seen-map teacher-forced PSNR / LPIPS; EMA weights come from the last recovery checkpoint.")


def table_horizons(rows):
    header = [["", "\\multicolumn{4}{c}{PSNR@h [95\\% CI]}", "\\multicolumn{4}{c}{LPIPS@h [95\\% CI]}"],
              ["Row"] + [f"h={h}" for h in HORIZONS] * 2]
    body = []
    for r in rows:
        cells = [r.label]
        for metric, nd in (("psnr", 2), ("lpips", 3)):
            for h in HORIZONS:
                v = r.rollout_at(metric, h)
                cells.append(f"{fmt(v, nd)} {fmt_ci(r.rollout_ci(metric, h), nd) if v is not None else ''}".strip())
        body.append(cells)
    def first(getter, nd):
        return [fmt(next((getter(r, h) for r in rows if getter(r, h) is not None), None), nd) for h in HORIZONS]
    body.append("midrule")
    body.append(["copy-seed"] + first(lambda r, h: r.copy_seed_psnr(h), 2) + first(lambda r, h: r.copy_seed_lpips(h), 3))
    note = "256 rollouts on seen maps. Intervals: episode bootstrap, the audit's 10k resamples at h=64 and 2k resamples over the audit's per-rollout values at h=8 and 32; h=16 has no per-rollout artifact and no interval. Copy-seed LPIPS exists only at the audited horizons."
    return tabular("lcccccccc", header, body, note)


def table_blur(rows):
    header = [["", "", "\\multicolumn{5}{c}{horizon 8, $\\sigma$ (px)}", "\\multicolumn{5}{c}{horizon 64, $\\sigma$ (px)}"],
              ["Row", "metric"] + [str(s) for s in SIGMAS] * 2]
    body = []
    for metric, nd in (("psnr", 2), ("lpips", 3)):
        for r in rows:
            body.append([r.label, metric.upper()] + [fmt(r.blur(h, s, metric), nd) for h in (8, 64) for s in SIGMAS])
        src = next((r for r in rows if r.blur(8, 0.0, f"copy_seed_{metric}") is not None), None)
        body.append(["copy-seed", metric.upper()] + [fmt(src.blur(h, s, f"copy_seed_{metric}") if src else None, nd) for h in (8, 64) for s in SIGMAS])
        if metric == "psnr":
            body.append("midrule")
    return tabular("llcccccccccc", header, body, "Gaussian blur applied to the predictions only, targets untouched; copy-seed blurred the same way as the persistence reference.")


def table_late_drop(rows):
    header = [["Row", "median $\\Delta$", "p10", "p90", "frac $< -2$ dB", "mean PSNR@64", "trim worst 10\\%: shift"]]
    body = []
    for r in rows:
        ld = pd.get(r.audit, "late_drop") or {}
        body.append([r.label, fmt(ld.get("median_delta")), fmt(ld.get("p10_delta")), fmt(ld.get("p90_delta")),
                     fmt(ld.get("frac_delta_below_-2db"), 2), fmt(ld.get("mean_psnr64")),
                     fmt(pd.get(ld, "drop_worst_10pct", "mean_psnr64_shift_by_psnr64"))])
    return tabular("lcccccc", header, body, "$\\Delta$ is the per-rollout PSNR change from horizon 32 to 64 (dB). The last column is how far the horizon-64 mean moves when the worst 10\\% of rollouts by PSNR@64 are dropped.")


def table_paired(summary, short_of):
    header = [["Pair", "Where", "Metric", "diff [95\\% CI]", "excludes 0"]]
    body = []
    names = {"psnr_h64_s0.0": "PSNR@64", "lpips_h64_s0.0": "LPIPS@64", "idm_top1": "IDM", "psnr_raw": "PSNR", "lpips_raw": "LPIPS",
             "psnr_dec": "PSNR (dec.)", "lpips_dec": "LPIPS (dec.)", "copy_psnr_raw": "copy-last PSNR"}
    for pair, entry in (pd.get(summary, "paired") or {}).items():
        a, _, b = pair.partition("_minus_")
        label = f"{short_of.get(a, a)} $-$ {short_of.get(b, b)}"
        for where in ("tf_seen", "tf_unseen", "rollout"):
            for k, d in (entry.get(where) or {}).items():
                nd = 3 if "lpips" in k or "idm" in k else 2
                body.append([label, where.replace("_", " "), names.get(k, k), f"{fmt(d.get('diff'), nd)} {fmt_ci(d.get('ci95'), nd)}",
                             "yes" if d.get("excludes_zero") else "no"])
                label = ""
    if not body:
        pd.warn("no paired differences (tmp/audit/summary.json missing or without 'paired')")
        body.append([NA, NA, NA, NA, NA])
    return tabular("lllcc", header, body, "Paired episode-bootstrap differences on the shared windows and rollouts.")


def table_idm(rows, idm, idm_k2):
    header = [["Judge / row", "top-1", "macro recall", "movement", "majority"]]
    body = []
    for name, m in (("IDM $K{=}8$ (held-out real windows)", idm), ("IDM $K{=}2$ control", idm_k2)):
        body.append([name, fmt(pd.get(m, "top1"), 3), fmt(pd.get(m, "macro_recall"), 3), fmt(pd.get(m, "movement"), 3), fmt(pd.get(m, "majority_baseline"), 3)])
    body.append("midrule")
    for r in rows:
        body.append([f"{r.label}, rollout agreement", f"{fmt(r.idm_top1(), 3)} {fmt_ci(r.idm_ci(), 3)}".strip(),
                     DASH, fmt(pd.get(r.drift, "idm_movement_mean"), 3), fmt(pd.get(r.drift, "idm_majority_baseline"), 3)])
    real = next((r.idm_real_top1() for r in rows if r.idm_real_top1() is not None), None)
    body.append(["real counterparts of the same rollouts", fmt(real, 3), DASH, DASH, DASH])
    return tabular("lcccc", header, body, "Rollout agreement is the mean over 64 horizons of the top-1 match between the judge's action and the action given to the model.")


# ---- knob grid ------------------------------------------------------------------------------

GRID_METRICS = (  # (name in rows_grid.json, corpus or None, eval_tf key, decimals, higher_is_better)
    ("seen_psnr", "seen", "psnr_raw", 2, True), ("seen_lpips", "seen", "lpips_raw", 3, False),
    ("unseen_psnr", "unseen", "psnr_raw", 2, True), ("unseen_lpips", "unseen", "lpips_raw", 3, False),
    ("unseen2_psnr", "unseen2", "psnr_raw", 2, True), ("unseen2_lpips", "unseen2", "lpips_raw", 3, False),
    ("seen_psnr_ema", "seen_ema", "psnr_raw", 2, True), ("seen_lpips_ema", "seen_ema", "lpips_raw", 3, False),
    ("unseen_psnr_ema", "unseen_ema", "psnr_raw", 2, True), ("unseen_lpips_ema", "unseen_ema", "lpips_raw", 3, False),
    ("unseen2_psnr_ema", "unseen2_ema", "psnr_raw", 2, True), ("unseen2_lpips_ema", "unseen2_ema", "lpips_raw", 3, False),
)


def grid_values(cell):
    """Every number one cell contributes, read from its own result directory."""
    v = {name: cell.tf_mean(corpus, key) for name, corpus, key, _, _ in GRID_METRICS}
    v.update(psnr8=cell.rollout_at("psnr", 8), psnr64=cell.rollout_at("psnr", 64),
             lpips64=cell.rollout_at("lpips", 64), fvd16=cell.fvd(16), fvd32=cell.fvd(32),
             idm=cell.idm_top1(), vloss_final=cell.vloss_final, params=cell.params)
    return v


def grid_records(cells, reg):
    """The `rows_grid.json` payload: one record per cell, its knobs as recorded by training, its
    metrics, and the ranking the grid is read on (rollout LPIPS@64, then PSNR@64)."""
    records = []
    for c in cells:
        knobs = c.knobs
        declared = c.spec.get("knob", "")
        records.append({"key": c.key, "label": c.label, "run": c.run, "exists": c.exists,
                        "knob": declared, "trained": knobs, "metrics": grid_values(c),
                        "knob_confirmed": _knob_confirmed(declared, knobs)})
    ranked = sorted([r for r in records if r["metrics"]["lpips64"] is not None],
                    key=lambda r: (r["metrics"]["lpips64"], -(r["metrics"]["psnr64"] or 0)))
    for i, r in enumerate(ranked, 1):
        r["rank_lpips64"] = i
    return {"results_root": reg.get("_results_root"), "ranked_on": "rollout lpips@64 then psnr@64",
            "cells": records,
            "complete": [r["key"] for r in records if r["metrics"]["lpips64"] is not None],
            "incomplete": [r["key"] for r in records if r["metrics"]["lpips64"] is None]}


def _knob_confirmed(declared, trained):
    """Did the cell train with the flag the launcher says it turns? None when the cell has no config."""
    if not any(v is not None for v in trained.values()):
        return None
    if declared in ("", DASH):
        return True
    flag, _, value = declared.partition(" ")
    key = flag.lstrip("-").replace("-", "_")
    got = trained.get(key)
    if got is None:
        return False
    try:
        return float(got) == float(value)
    except (TypeError, ValueError):
        return str(got) == value


def table_grid(cells):
    """One row per cell, ordered by rollout LPIPS@64: the ranking the grid design reads it on."""
    header = [["", "", "\\multicolumn{2}{c}{Seen (EMA)}", "\\multicolumn{2}{c}{Unseen-2 (EMA)}",
               "\\multicolumn{3}{c}{Rollout}", "", ""],
              ["Cell", "Knob", "PSNR$\\uparrow$", "LPIPS$\\downarrow$", "PSNR$\\uparrow$", "LPIPS$\\downarrow$",
               "PSNR@8$\\uparrow$", "PSNR@64$\\uparrow$", "LPIPS@64$\\downarrow$", "FVD$_{16/32}\\downarrow$", "v-loss$\\downarrow$"]]
    rows = []
    for c in cells:
        v = grid_values(c)
        rows.append((v["lpips64"] if v["lpips64"] is not None else float("inf"), c, v))
    rows.sort(key=lambda r: r[0])
    body = []
    for _, c, v in rows:
        body.append([c.label, "\\texttt{" + c.spec.get("knob", DASH).replace("_", "\\_") + "}",
                     fmt(v["seen_psnr_ema"]), fmt(v["seen_lpips_ema"], 3),
                     fmt(v["unseen2_psnr_ema"]), fmt(v["unseen2_lpips_ema"], 3),
                     fmt(v["psnr8"]), fmt(v["psnr64"]), fmt(v["lpips64"], 3),
                     f"{fmt(v['fvd16'], 0)} / {fmt(v['fvd32'], 0)}", fmt(v["vloss_final"], 4)])
    note = ("Twelve cells, one knob each against the first row, 30k updates of the shared PixArt-$\\alpha$ recipe. "
            "EMA weights come from each cell's last recovery checkpoint; rows are ordered by rollout LPIPS@64. "
            "n/a: that cell's artifact does not exist yet.")
    return tabular("ll" + "c" * 9, header, body, note)


def main(args):
    reg = pd.load_registry(args.rows)
    root = args.results_root
    rows = lambda g: pd.load_rows(root, reg, g)  # noqa: E731
    write(args.out, "main.tex", table_main(rows("main")))
    write(args.out, "transfer.tex", table_transfer(rows("transfer")))
    write(args.out, "ema.tex", table_ema(rows("ema")))
    write(args.out, "seeds.tex", table_simple_grid(rows("seeds"), with_vloss=True))
    write(args.out, "cm.tex", table_simple_grid(rows("cm")))
    write(args.out, "grid.tex", table_simple_grid(rows("grid")))
    write(args.out, "horizons.tex", table_horizons(rows("horizon")))
    write(args.out, "blur.tex", table_blur(rows("blur")))
    write(args.out, "late_drop.tex", table_late_drop(rows("blur")))
    summary = pd.read_json(os.path.join(root, reg["audit_summary"]), "audit summary")
    short_of = {s["run"]: s.get("short", s["label"]) for s in reg["rows"]}
    write(args.out, "paired.tex", table_paired(summary, short_of))
    idm = pd.read_json(os.path.join(root, reg["idm"], "metrics.json"), "IDM metrics")
    idm_k2 = pd.read_json(os.path.join(root, reg["idm_k2"], "metrics.json"), "IDM K=2 metrics")
    write(args.out, "idm.tex", table_idm(rows("main"), idm, idm_k2))
    cells = pd.load_grid(root, reg)
    write(args.out, "knobs.tex", table_grid(cells))
    grid = grid_records(cells, {**reg, "_results_root": root})
    os.makedirs(args.out, exist_ok=True)
    json.dump(grid, open(args.grid_json or os.path.join(args.out, "rows_grid.json"), "w"), indent=1)
    print(f"wrote {args.grid_json or os.path.join(args.out, 'rows_grid.json')}: "
          f"{len(grid['complete'])}/{len(grid['cells'])} cells complete")
    for r in grid["cells"]:
        if r["knob_confirmed"] is False:
            pd.warn(f"cell '{r['key']}' trained with {r['trained']}, which does not match its knob '{r['knob']}'")
    if pd.warn.seen:
        print(f"[paper] {len(pd.warn.seen)} warning(s); see stderr", file=sys.stderr)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True, help="local mirror of results_spiderman (see sync_results.sh)")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables"))
    p.add_argument("--rows", default=pd.ROWS_JSON)
    p.add_argument("--grid-json", default="", help="where to write the collected knob-grid records (default <out>/rows_grid.json)")
    main(p.parse_args())
