"""`paper/make_adapt_figures.py`: the adaptation study's figures, tables and summary from `score_adapt.py` rows.

Every test builds a tiny results tree by hand in the layout the adaptation runs leave: one directory per run
named `<source>_<set>_map<NN>_r<rank>_k<k>_s<seed>[_<variant>]` holding `scores.jsonl` (one row per scored
checkpoint, `heldout_A_<decoder>` the scene-crop decoded advantage) and `eval_tf.py`'s `per_window.csv` under
`scores/step<N>_live_<hash>/heldout/`, where each row's `heldout_per_window` names it by its server path. Two
arenas and three checkpoints (0, 250, 500) with a budget of 500 and a home of 4.0:

  arena 6: A = 1.0, 2.5, 3.0, half-gap line (1.0 + 4.0) / 2 = 2.5, reached exactly at 250 (reaching counts);
  arena 9: A = 2.0, 2.5, 2.9, line 3.0, never reached: right-censored at the budget.

    python -m pytest paper/fixtures/test_make_adapt_figures.py -q
"""
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
sys.path.insert(0, PAPER)
sys.path.insert(0, os.path.dirname(PAPER))

pytest.importorskip("matplotlib")
pytest.importorskip("scipy")
import make_adapt_figures as maf  # noqa: E402

HOME = 4.0
BUDGET = 500
STEPS = (0, 250, 500)
EPISODE_OFFSETS = (-0.3, -0.1, 0.1, 0.3)     # sum to zero, so the pooled window mean is the row's A exactly
WINDOWS_PER_EPISODE = 3
ARENA_A = {6: (1.0, 2.5, 3.0), 9: (2.0, 2.5, 2.9)}
ARENA_D = {6: 0.19, 9: 0.18}


def run_name(arena, k=8, seed=0, variant=""):
    return f"unet200k_arenas13_map{arena:02d}_r16_k{k}_s{seed}" + (f"_{variant}" if variant else "")


def write_per_window(path, a, decoders=("stock",)):
    """eval_tf's per_window.csv for one checkpoint: scene columns whose pooled paired mean is `a[decoder]`."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["index", "episode", "map", "start", "psnr_dec", "copy_psnr_dec", "latent_mse", "copy_latent_mse",
              "dup_latent", "scene_psnr_dec", "scene_copy_psnr_dec", "lpips_dec"]
    for d in decoders:
        if d != "stock":
            fields += [f"psnr_dec_{d}", f"copy_psnr_dec_{d}", f"scene_psnr_dec_{d}", f"scene_copy_psnr_dec_{d}"]
    rows, i = [], 0
    for e, off in enumerate(EPISODE_OFFSETS):
        for w in range(WINDOWS_PER_EPISODE):
            copy = 20.0 + 0.5 * w
            r = {"index": i, "episode": 100 + e, "map": 6, "start": 40 * w, "latent_mse": 0.2, "copy_latent_mse": 0.3,
                 "dup_latent": 0, "lpips_dec": 0.2, "psnr_dec": copy + a["stock"] + off, "copy_psnr_dec": copy,
                 "scene_psnr_dec": copy + a["stock"] + off, "scene_copy_psnr_dec": copy}
            for d in decoders:
                if d != "stock":
                    r.update({f"psnr_dec_{d}": copy + a[d] + off, f"copy_psnr_dec_{d}": copy,
                              f"scene_psnr_dec_{d}": copy + a[d] + off, f"scene_copy_psnr_dec_{d}": copy})
            rows.append(r)
            i += 1
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_run(root, arena, A, k=8, seed=0, variant="", steps=STEPS, tuned=None, skill=None, lpips=None,
              per_window=True, weights="live", extra=None):
    """One adaptation run's `scores.jsonl` (and per-window files) under `root/results/adapt/`."""
    name = run_name(arena, k, seed, variant)
    run_dir = os.path.join(str(root), "results", "adapt", name)
    os.makedirs(run_dir, exist_ok=True)
    skill = skill or [1.0 + 0.5 * i for i in range(len(steps))]
    lpips = lpips or [0.25 - 0.03 * i for i in range(len(steps))]
    decoders = {"stock": {"path": "", "identity": "hub:stock"}}
    if tuned is not None:
        decoders["tuned"] = {"path": "/x/vae_decoder_tuned", "identity": "sha:1"}
    with open(os.path.join(run_dir, "scores.jsonl"), "a") as f:
        for i, s in enumerate(steps):
            step_dir = f"step{s:07d}_{weights}_{i:016x}"
            pw = os.path.join(run_dir, "scores", step_dir, "heldout", "per_window.csv")
            a = {"stock": A[i]}
            if tuned is not None:
                a["tuned"] = tuned[i]
            if per_window:
                write_per_window(pw, a, tuple(decoders))
            row = {"run": name, "map": f"arenas13_map{arena:02d}", "set": "arenas13", "map_id": arena, "step": s,
                   "seed": seed, "weights": weights, "adapt_episodes_k": k, "rank": 16, "grid": list(steps),
                   "decoder": "stock", "decoders": decoders, "eval_fingerprint": "fp0",
                   "scored_at": f"2026-09-27T0{i}:00:00", "heldout_A_stock": A[i], "heldout_A_full_stock": A[i] - 0.4,
                   "heldout_B_stock": None, "heldout_C_stock": None, "heldout_latent_skill": skill[i],
                   "heldout_lpips_dec": lpips[i], "heldout_windows": 12, "trainmap_A_stock": None,
                   "directional_correct_frac": None,
                   "heldout_per_window": f"/home/rohan/Doom/results/adapt/{name}/scores/{step_dir}/heldout/"
                                         "per_window.csv"}
            if tuned is not None:
                row.update({"heldout_A_tuned": tuned[i], "heldout_A_full_tuned": tuned[i] - 0.4,
                            "heldout_B_tuned": None, "heldout_C_tuned": None})
            row.update(extra or {})
            f.write(json.dumps(row) + "\n")
    return run_dir


def write_distances(path):
    maps = [{"key": f"val/{m}", "map": m, "role": "primary", "cluster": "arena", "D": 0.03 + 0.01 * m}
            for m in (2, 3, 4, 5)]
    maps += [{"key": f"seen/{m}", "map": m, "role": "primary", "cluster": "arena", "D": d} for m, d in ARENA_D.items()]
    maps += [{"key": "seen/6", "map": 6, "role": "replication", "cluster": "arena", "D": 0.5}]
    with open(path, "w") as f:
        json.dump({"space": "sd1", "maps": maps}, f)
    return str(path)


def tree(tmp_path, tuned=False):
    for arena, A in ARENA_A.items():
        write_run(tmp_path, arena, A, tuned=[a + 0.3 for a in A] if tuned else None)
    return write_distances(tmp_path / "distances.json")


def cli(tmp_path, distances, *extra):
    out, tables = tmp_path / "figures", tmp_path / "tables"
    argv = ["--runs-glob", str(tmp_path / "results" / "adapt" / "*" / "scores.jsonl"), "--distances", distances,
            "--out-dir", str(out), "--tables-dir", str(tables), "--budget", str(BUDGET),
            "--fresh-root", str(tmp_path / "no_fresh_reads"), "--bootstrap", "200", "--arena-bootstrap", "1000",
            *extra]
    if "--home" not in extra and "--home-json" not in extra:
        argv += ["--home", str(HOME)]
    assert maf.main(argv) == 0
    return out, tables


def summary(tables):
    return json.load(open(os.path.join(tables, "adapt_summary.json")))


# ---------------------------------------------------------------------------------------------
# the crossing rule
# ---------------------------------------------------------------------------------------------

def curve_rows(values, steps=STEPS):
    return [{"step": s, "heldout_A_stock": v} for s, v in zip(steps, values)]


def test_the_half_gap_line_is_halfway_from_the_zero_shot_advantage_to_home():
    assert maf.half_gap_line(1.0, 4.0) == pytest.approx(2.5)
    assert maf.half_gap_line(2.63, 4.138) == pytest.approx(3.384)


def test_the_first_grid_step_that_reaches_the_line_is_the_cost_and_reaching_it_exactly_counts():
    c = maf.crossing(curve_rows((1.0, 2.5, 3.0)), "heldout_A_stock", 2.5, BUDGET)
    assert (c["step"], c["censored"], c["incomplete"]) == (250, False, False)
    assert c["value"] == pytest.approx(2.5)


def test_a_curve_that_never_reaches_the_line_by_the_budget_is_right_censored():
    c = maf.crossing(curve_rows((2.0, 2.5, 2.9)), "heldout_A_stock", 3.0, BUDGET)
    assert c["step"] is None and c["censored"] and not c["incomplete"]
    assert c["last_step"] == 500 and c["last_value"] == pytest.approx(2.9)


def test_a_crossing_after_the_budget_is_censored_at_the_budget():
    rows = curve_rows((2.0, 2.5, 2.9, 3.1), steps=(0, 250, 500, 1000))
    assert maf.crossing(rows, "heldout_A_stock", 3.0, BUDGET)["censored"]
    assert maf.crossing(rows, "heldout_A_stock", 3.0, None)["step"] == 1000


def test_a_curve_not_yet_scored_to_the_budget_is_incomplete_not_censored():
    c = maf.crossing(curve_rows((2.0, 2.5), steps=(0, 250)), "heldout_A_stock", 3.0, BUDGET)
    assert c["incomplete"] and not c["censored"] and c["step"] is None
    # a curve that already crossed is not incomplete, however far it was scored
    assert maf.crossing(curve_rows((2.0, 3.5), steps=(0, 250)), "heldout_A_stock", 3.0, BUDGET)["step"] == 250


def test_step_zero_crosses_only_when_the_zero_shot_advantage_is_already_at_home():
    line = maf.half_gap_line(4.2, HOME)
    assert maf.crossing(curve_rows((4.2, 4.0, 4.1)), "heldout_A_stock", line, BUDGET)["step"] == 0
    line = maf.half_gap_line(3.9, HOME)
    assert maf.crossing(curve_rows((3.9, 4.0, 4.1)), "heldout_A_stock", line, BUDGET)["step"] == 250


def test_a_censored_median_is_censored_and_never_a_median_over_crossers():
    inf = math.inf
    assert maf.median_censored([250, 500, inf]) == 500
    assert maf.median_censored([250, inf, inf]) == inf
    assert maf.median_censored([250, 500, 1000, inf]) == 750
    assert maf.median_censored([250, inf]) == inf
    assert maf.median_censored([]) is None


def test_the_spearman_set_ranks_censored_costs_tied_above_every_crossing():
    assert maf.cost_rank_value(None, BUDGET) == 2 * BUDGET
    assert maf.cost_rank_value(250, BUDGET) == 250
    from scipy.stats import spearmanr
    x, cost = [0.1, 0.2, 0.3, 0.4], [250, None, 500, None]
    r = maf.spearman(x, [maf.cost_rank_value(c, BUDGET) for c in cost])
    assert r["rho"] == pytest.approx(spearmanr(x, [250, 1000, 500, 1000]).statistic)
    assert r["n"] == 4
    assert maf.spearman([1, 2], [2, 1])["rho"] is None      # fewer than three pairs is not a correlation


# ---------------------------------------------------------------------------------------------
# the pipeline on the two-arena tree
# ---------------------------------------------------------------------------------------------

def test_the_summary_counts_come_from_the_rows(tmp_path):
    _, tables = cli(tmp_path, tree(tmp_path))
    s = summary(tables)
    assert s["n_arenas"] == 2 and s["n_complete"] == 2 and s["budget"] == BUDGET
    assert s["home"] == pytest.approx(HOME) and s["decoder"] == "stock"
    c = s["crossings"]
    assert c["half_gap_by_budget"] == 1 and c["half_gap_by_500"] == 1
    assert c["half_gap_by_step"] == {"0": 0, "250": 1, "500": 1}
    assert c["home_by_budget"] == 0 and c["above_home_at_budget"] == 0
    assert c["censored_arenas"] == [9]
    m = s["medians"]
    assert m["A0"] == pytest.approx(1.5) and m["A_budget"] == pytest.approx(2.95)
    assert m["gain"] == pytest.approx((2.0 + 0.9) / 2)
    assert m["cost_half_gap"] == {"value": None, "censored": True, "label": ">500"}
    assert m["cost_home"]["censored"]
    late = s["late_gain"]
    assert late["from"] == 250 and late["to"] == BUDGET
    assert late["mean"] == pytest.approx(((3.0 - 2.5) + (2.9 - 2.5)) / 2) and late["positive"] == 2
    per = {a["arena"]: a for a in s["per_arena"]}
    assert per[6]["cost_half_gap"] == 250 and per[9]["cost_half_gap"] is None and per[9]["censored_half_gap"]
    assert per[6]["half_gap_line"] == pytest.approx(2.5) and per[6]["D"] == pytest.approx(0.19)
    assert per[6]["A"]["500"] == pytest.approx(3.0)
    # the time-averaged advantage over the budget (trapezoid over updates), beside the crossing
    assert per[6]["auc"] == pytest.approx(((1.0 + 2.5) / 2 * 250 + (2.5 + 3.0) / 2 * 250) / 500)
    assert set(s["spearman"]) == {"D_vs_A0", "D_vs_A_budget", "D_vs_gain", "D_vs_cost", "S0_vs_A_budget",
                                  "S0_vs_cost", "A0_vs_cost"}


def test_the_intervals_come_from_the_per_window_files_by_episode(tmp_path):
    _, tables = cli(tmp_path, tree(tmp_path))
    per = {a["arena"]: a for a in summary(tables)["per_arena"]}
    lo, hi = per[6]["A0_ci"]
    assert lo < 1.0 < hi and hi - lo < 1.0
    assert summary(tables)["notes"] == []


def test_a_per_window_file_that_disagrees_with_its_row_is_flagged(tmp_path):
    distances = tree(tmp_path)
    path = os.path.join(tmp_path, "results", "adapt", run_name(9), "scores.jsonl")
    rows = [json.loads(line) for line in open(path)]
    rows[0]["heldout_A_stock"] = 2.2
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    _, tables = cli(tmp_path, distances)
    assert any("per_window" in n and "9" in n for n in summary(tables)["notes"])


def test_the_latex_tables_carry_the_header_and_compile(tmp_path):
    _, tables = cli(tmp_path, tree(tmp_path))
    for name in ("adapt_cost.tex", "adapt_perarena.tex"):
        text = open(os.path.join(tables, name)).read()
        assert re.match(r"% generated by make_adapt_figures\.py from .+ at \d{4}-\d\d-\d\dT", text)
        assert "\\toprule" in text and "\\bottomrule" in text
    cost = open(os.path.join(tables, "adapt_cost.tex")).read()
    assert "${>}$500" in cost and "250" in cost
    # arenas sorted by D: arena 9 (0.18) before arena 6 (0.19)
    assert cost.index("\n9 &") < cost.index("\n6 &")
    per = open(os.path.join(tables, "adapt_perarena.tex")).read()
    assert per.count("\\tbd") >= 6          # M, forgetting and directional are not in these rows yet
    if shutil.which("pdflatex") is None:
        pytest.skip("pdflatex not installed")
    doc = tmp_path / "preview.tex"
    doc.write_text("\\documentclass{article}\\usepackage{booktabs}\\newcommand{\\tbd}{[tbd]}"
                   "\\newcommand{\\prov}[1]{#1}\\begin{document}"
                   f"\\input{{{tables / 'adapt_cost.tex'}}}\n\n\\input{{{tables / 'adapt_perarena.tex'}}}"
                   "\\end{document}")
    proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", str(doc)],
                          capture_output=True, text=True, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert (tmp_path / "preview.pdf").exists()


def test_prov_wraps_every_number_for_the_papers_provisional_marks(tmp_path):
    _, tables = cli(tmp_path, tree(tmp_path), "--prov")
    cost = open(os.path.join(tables, "adapt_cost.tex")).read()
    assert "\\prov{0.180}" in cost and "\\prov{+1.00}" in cost


def mediabox(path):
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", open(path, "rb").read())
    return float(m.group(1)) / 72, float(m.group(2)) / 72


def test_the_figures_are_written_as_pdf_and_png_at_the_slot_size_without_type3_fonts(tmp_path):
    out, _ = cli(tmp_path, tree(tmp_path))
    for stem in ("fig3_adaptation_curves", "fig3_adaptation_skill", "fig4_adaptation",
                 "fig2a_advantage_by_distance"):
        for ext in ("pdf", "png"):
            p = os.path.join(out, f"{stem}.{ext}")
            assert os.path.getsize(p) > 1000, p
        assert b"/Type3" not in open(os.path.join(out, f"{stem}.pdf"), "rb").read()
    w, h = mediabox(os.path.join(out, "fig3_adaptation_curves.pdf"))
    assert w == pytest.approx(maf.FIG3_SIZE[0], abs=0.01) and h == pytest.approx(maf.FIG3_SIZE[1], abs=0.01)
    # nothing to draw yet: no seed, ladder or recipe runs, no raw columns; the D terciles are no longer drawn
    for stem in ("fig3_adaptation_seeds", "fig_adapt_ladder", "fig_adapt_recipe", "fig2b_margin_by_distance",
                 "fig3_adaptation_terciles"):
        assert not os.path.exists(os.path.join(out, f"{stem}.pdf")), stem


def test_seed_ladder_and_recipe_runs_get_their_own_figures_and_summary_entries(tmp_path):
    distances = tree(tmp_path)
    write_run(tmp_path, 6, (1.0, 2.4, 3.2), seed=1)
    write_run(tmp_path, 6, (1.0, 1.8, 2.2), k=2)
    write_run(tmp_path, 6, (1.0, 2.9, 3.6), variant="lr5e4")
    write_run(tmp_path, 9, (2.0, 2.6, 2.8, 2.95, 3.3), variant="g8k", steps=(0, 50, 250, 500, 1000))
    out, tables = cli(tmp_path, distances)
    for stem in ("fig3_adaptation_seeds", "fig_adapt_ladder", "fig_adapt_recipe"):
        assert os.path.getsize(os.path.join(out, f"{stem}.pdf")) > 1000, stem
    s = summary(tables)
    assert s["n_arenas"] == 2                        # the extra runs never enter the headline set
    seeds = {e["arena"]: e for e in s["seeds"]}
    assert seeds[6]["A_budget"] == {"0": pytest.approx(3.0), "1": pytest.approx(3.2)}
    assert seeds[6]["spread_budget"] == pytest.approx(0.2)
    assert s["ladder"]["6"] == {"2": pytest.approx(2.2), "8": pytest.approx(3.0)}
    recipe = {(e["arena"], e["variant"]): e for e in s["recipe"]}
    assert recipe[(6, "lr5e4")]["A_budget"] == pytest.approx(3.6)
    assert recipe[(6, "lr5e4")]["delta_vs_base"] == pytest.approx(0.6)
    assert recipe[(9, "g8k")]["A_budget"] == pytest.approx(2.95) and recipe[(9, "g8k")]["A_last"] == pytest.approx(3.3)
    # the 8k grid reaches arena 9's line after the budget: censored at the budget, reached on the full curve
    assert recipe[(9, "g8k")]["cost_half_gap"] is None and recipe[(9, "g8k")]["cost_half_gap_full_curve"] == 1000


def test_live_rows_are_read_and_ema_rows_ignored_by_default(tmp_path):
    distances = tree(tmp_path)
    write_run(tmp_path, 6, (9.0, 9.0, 9.0), weights="ema")
    _, tables = cli(tmp_path, distances)
    per = {a["arena"]: a for a in summary(tables)["per_arena"]}
    assert per[6]["A0"] == pytest.approx(1.0)


def test_an_incomplete_run_is_reported_and_left_out_of_the_counts(tmp_path):
    distances = tree(tmp_path)
    write_run(tmp_path, 12, (1.5, 2.0), steps=(0, 250))
    with open(distances) as f:
        js = json.load(f)
    js["maps"].append({"key": "seen/12", "map": 12, "role": "primary", "cluster": "arena", "D": 0.128})
    with open(distances, "w") as f:
        json.dump(js, f)
    _, tables = cli(tmp_path, distances)
    s = summary(tables)
    assert s["n_arenas"] == 3 and s["n_complete"] == 2 and s["crossings"]["incomplete_arenas"] == [12]
    assert s["crossings"]["half_gap_by_budget"] == 1


def test_the_tuned_decoder_needs_its_own_home_and_adds_a_column_beside_stock(tmp_path):
    distances = tree(tmp_path, tuned=True)
    with pytest.raises(SystemExit):
        maf.main(["--runs-glob", str(tmp_path / "results" / "adapt" / "*" / "scores.jsonl"), "--distances",
                  distances, "--decoder", "tuned", "--out-dir", str(tmp_path / "f"), "--tables-dir",
                  str(tmp_path / "t"), "--fresh-root", str(tmp_path / "none")])
    _, tables = cli(tmp_path, distances)
    assert "tuned" in open(os.path.join(tables, "adapt_cost.tex")).read()
    _, tables = cli(tmp_path, distances, "--decoder", "tuned", "--home", "4.6")
    s = summary(tables)
    per = {a["arena"]: a for a in s["per_arena"]}
    assert s["decoder"] == "tuned" and per[6]["A0"] == pytest.approx(1.3)
    assert per[6]["half_gap_line"] == pytest.approx((1.3 + 4.6) / 2)
    assert "stock" in open(os.path.join(tables, "adapt_cost.tex")).read()


def test_the_home_json_is_the_scene_advantage_of_an_eval_tf_read(tmp_path):
    p = tmp_path / "metrics.json"
    p.write_text(json.dumps({"scene_psnr_dec": {"mean": 25.0, "n": 512}, "scene_copy_psnr_dec": {"mean": 21.0},
                             "scene_psnr_dec_tuned": {"mean": 26.0}, "scene_copy_psnr_dec_tuned": {"mean": 21.5},
                             "psnr_dec": {"mean": 30.0}, "copy_psnr_dec": {"mean": 20.0}}))
    assert maf.home_from_json(str(p), "stock") == pytest.approx(4.0)
    assert maf.home_from_json(str(p), "tuned") == pytest.approx(4.5)
    # score_adapt leaves duplicate windows out of every mean; the duplicate-free means win where present
    js = json.loads(p.read_text())
    js.update({"scene_psnr_dec_nodup": {"mean": 25.2}, "scene_copy_psnr_dec_nodup": {"mean": 21.0}})
    p.write_text(json.dumps(js))
    assert maf.home_from_json(str(p), "stock") == pytest.approx(4.2)
    distances = tree(tmp_path)
    _, tables = cli(tmp_path, distances, "--home-json", str(p))
    s = summary(tables)
    assert s["home"] == pytest.approx(4.2) and s["home_source"].endswith("metrics.json")


def write_fresh(root, row, arena, scene_pred, scene_copy, lpips_raw=None, persist_lpips=None, name=None):
    d = os.path.join(str(root), row, name or f"arenas13_map{arena:02d}")
    os.makedirs(d, exist_ok=True)
    m = {"scene_psnr_dec": {"mean": scene_pred, "n": 256}, "scene_copy_psnr_dec": {"mean": scene_copy, "n": 256}}
    if lpips_raw is not None:
        m.update({"scene_lpips_raw": {"mean": lpips_raw}, "scene_persist_lpips_raw": {"mean": persist_lpips}})
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(m, f)


def test_a_directory_of_eval_tf_reads_keyed_by_map_is_one_zero_shot_row(tmp_path):
    fresh = tmp_path / "fresh"
    write_fresh(fresh, "pixart", 6, 24.0, 23.0, 0.30, 0.20)
    write_fresh(fresh, "pixart", 9, 24.5, 22.0, name="seen_map09_h1")
    write_fresh(fresh, "pixart", 9, 99.0, 0.0, name="seen_map09_h4")        # four tics: not a zero-shot A
    for m in (2, 3, 4, 5):
        write_fresh(fresh, "pixart", m, 25.0, 21.0 + 0.1 * m, 0.2, 0.25, name=f"val_map{m:02d}")
    row = maf.load_zero_shot_row(str(fresh / "pixart"), "stock")
    assert row[6]["A"] == pytest.approx(1.0) and row[9]["A"] == pytest.approx(2.5)
    assert row[6]["M"] == pytest.approx(0.10) and row[9]["M"] is None
    out, tables = cli(tmp_path, tree(tmp_path), "--fresh-root", str(fresh), "--with-raw")
    assert os.path.getsize(os.path.join(out, "fig2a_advantage_by_distance.pdf")) > 1000
    assert os.path.getsize(os.path.join(out, "fig2b_margin_by_distance.pdf")) > 1000
    zs = summary(tables)["zero_shot"]
    assert zs["pixart"]["maps"]["9"]["A"] == pytest.approx(2.5)
    # the row's own home: the training maps' advantage, pooled over their windows
    assert zs["pixart"]["home"] == pytest.approx(sum(4.0 - 0.1 * m for m in (2, 3, 4, 5)) / 4)
    assert zs["unet"]["source"] == "adaptation step 0"


def test_the_configuration_that_carries_the_decoder_is_read_and_the_choice_noted(tmp_path):
    distances = tree(tmp_path)
    # a partial rescore with the tuned decoder beside the full stock-only scoring, under another fingerprint
    for arena, A in ARENA_A.items():
        write_run(tmp_path, arena, A[:2], steps=STEPS[:2], tuned=[a + 0.3 for a in A[:2]],
                  extra={"eval_fingerprint": "fp1", "scored_at": "2026-09-28T00:00:00"})
    _, tables = cli(tmp_path, distances)
    s = summary(tables)
    assert {a["arena"]: a["A0"] for a in s["per_arena"]} == {9: pytest.approx(2.0), 6: pytest.approx(1.0)}
    assert s["n_complete"] == 2 and any("2 evaluation configurations" in n for n in s["notes"])
    _, tables = cli(tmp_path, distances, "--decoder", "tuned", "--home", "4.6")
    s = summary(tables)
    assert {a["arena"]: a["A0"] for a in s["per_arena"]} == {9: pytest.approx(2.3), 6: pytest.approx(1.3)}
    assert s["n_complete"] == 0 and s["crossings"]["incomplete_arenas"] == [6, 9]


def test_a_decoder_no_row_carries_stops_with_a_message(tmp_path):
    with pytest.raises(SystemExit, match="heldout_A_tuned"):
        cli(tmp_path, tree(tmp_path), "--decoder", "tuned", "--home", "4.6")


# ---------------------------------------------------------------------------------------------
# across arenas: attainment, the interquartile mean, the nested bootstrap, Table 3
# ---------------------------------------------------------------------------------------------

def test_the_interquartile_mean_trims_a_quarter_from_each_end():
    import numpy as np
    assert maf.iqm(np.arange(13.0)) == pytest.approx(6.0)             # keeps 3..9
    assert maf.iqm(np.array([1.0, 3.0])) == pytest.approx(2.0)        # below four values: the mean
    assert maf.iqm(np.array([[0.0, 100.0, 1.0, 2.0, 3.0]]), axis=1)[0] == pytest.approx(2.0)


def test_first_attainment_at_risk_and_profiles():
    import numpy as np
    steps = [0, 250, 500]
    curves = np.array([[1.0, 2.5, 3.0], [2.0, 2.5, 2.9]])
    first = maf.first_step(curves, np.array([2.5, 3.0]), steps)
    assert first.tolist() == [250, math.inf]
    assert maf.at_risk(first.tolist(), steps) == [2, 2, 1]
    assert maf.profile([1.0, 2.0, 3.0], [1.5, 3.0]).tolist() == pytest.approx([2 / 3, 1 / 3])


def test_the_nested_bootstrap_resamples_arenas_then_episodes_paired_across_steps():
    import numpy as np
    point = np.array([[1.0, 2.0], [3.0, 5.0]])
    # arena 0 has two episodes (two windows each) whose means straddle its point; arena 1 has no per-window files
    mats = [(np.array([[0.5, 1.5], [1.0, 3.0]]) * 2, np.array([2.0, 2.0])), None]
    idx, boot = maf.nested_bootstrap(point, mats, 400, 0)
    assert boot.shape == (400, 2, 2)
    assert np.allclose(boot[idx == 1], [3.0, 5.0])                  # no episodes: the point curve
    drew_0 = boot[idx == 0]
    assert set(np.round(drew_0[:, 0], 6)) <= {0.5, 1.0, 1.5}
    # one episode draw serves every step: the step-0 and step-1 means move together
    assert np.allclose(drew_0[:, 1] - 1.0, 2 * (drew_0[:, 0] - 0.5))


def test_the_summary_carries_attainment_iqm_and_gap_share_with_intervals(tmp_path):
    _, tables = cli(tmp_path, tree(tmp_path))
    aa = summary(tables)["across_arenas"]
    assert aa["steps"] == [0, 250, 500] and aa["n"] == 2
    att = aa["attainment_half_gap"]
    assert att["value"] == pytest.approx([0.0, 0.5, 0.5])
    assert att["at_risk"] == [2, 2, 1] and att["crossing_step"] == {"6": 250, "9": None}
    for (lo, hi), v in zip(att["ci"], att["value"]):
        assert lo <= v <= hi
    iqm = aa["iqm_A"]
    assert iqm["value"] == pytest.approx([1.5, 2.5, 2.95])     # two arenas: the IQM is their mean
    assert all(lo <= v <= hi for (lo, hi), v in zip(iqm["ci"], iqm["value"]))
    assert aa["attainment_fixed"]["threshold"] == pytest.approx(3.5)
    share = aa["gap_share"]["per_arena"]
    assert share["6"] == pytest.approx((3.0 - 1.0) / (HOME - 1.0)) and share["9"] == pytest.approx(0.9 / 2.0)
    assert "nested bootstrap" in aa["method"] and aa["episode_resampling_missing"] == []


def test_figure_4_and_the_appendix_panels_are_drawn_at_their_sizes(tmp_path):
    out, _ = cli(tmp_path, tree(tmp_path))
    for stem in ("fig4_adaptation", "figA_adapt_arenas", "figA_adapt_profiles"):
        assert os.path.getsize(os.path.join(out, f"{stem}.png")) > 1000, stem
        assert b"/Type3" not in open(os.path.join(out, f"{stem}.pdf"), "rb").read()
    w, h = mediabox(os.path.join(out, "fig4_adaptation.pdf"))
    assert w == pytest.approx(maf.FIG4_SIZE[0], abs=0.01) and h == pytest.approx(maf.FIG4_SIZE[1], abs=0.01)


def test_table_3_carries_arena_bootstrap_intervals_and_states_the_censoring(tmp_path):
    p = tmp_path / "home" / "metrics.json"
    os.makedirs(p.parent)
    p.write_text(json.dumps({"scene_psnr_dec": {"mean": 25.0}, "scene_copy_psnr_dec": {"mean": 21.0}}))
    write_per_window(str(p.parent / "per_window.csv"), {"stock": HOME})
    _, tables = cli(tmp_path, tree(tmp_path), "--home-json", str(p))
    s = summary(tables)
    assert s["home"] == pytest.approx(HOME) and s["home_ci"]["ci"][0] < HOME < s["home_ci"]["ci"][1]
    t3 = open(os.path.join(tables, "adapt_table3.tex")).read()
    assert "\\toprule" in t3 and "arena bootstrap" in t3
    assert "1 [" in t3 and "of 2" in t3                      # one of two past the half-gap line, with its interval
    assert "1 of 2 censored" in t3                            # the median budget states its censoring
    assert "4.00 [" in t3                                     # home with its episode interval
    cost = open(os.path.join(tables, "adapt_cost.tex")).read()
    assert "IQM" in cost and "censored" in cost


def test_runs_without_the_decoders_rows_draw_no_seed_ladder_or_recipe_figure(tmp_path):
    distances = tree(tmp_path, tuned=True)
    write_run(tmp_path, 6, (1.0, 2.4, 3.2), seed=1)                   # stock only: nothing to draw under tuned
    write_run(tmp_path, 6, (1.0, 1.8, 2.2), k=2)
    write_run(tmp_path, 6, (1.0, 2.9, 3.6), variant="lr5e4")
    out, tables = cli(tmp_path, distances, "--decoder", "tuned", "--home", "4.6")
    for stem in ("fig3_adaptation_seeds", "fig_adapt_ladder", "fig_adapt_recipe"):
        assert not os.path.exists(os.path.join(out, f"{stem}.pdf")), stem
    s = summary(tables)
    assert s["seeds"] == [] and s["ladder"] == {} and s["recipe"] == []


def test_the_headline_variant_flag_promotes_the_8k_grid_and_its_budget(tmp_path):
    distances = tree(tmp_path)
    steps8 = (0, 50, 250, 500, 1000)
    write_run(tmp_path, 6, (1.0, 2.0, 2.6, 3.0, 3.2), variant="g8k", steps=steps8)
    write_run(tmp_path, 9, (2.0, 2.2, 2.5, 2.9, 3.1), variant="g8k", steps=steps8)
    argv = ["--runs-glob", str(tmp_path / "results" / "adapt" / "*" / "scores.jsonl"), "--distances", distances,
            "--out-dir", str(tmp_path / "f"), "--tables-dir", str(tmp_path / "t"), "--home", str(HOME),
            "--fresh-root", str(tmp_path / "none"), "--bootstrap", "100", "--arena-bootstrap", "200",
            "--headline-variant", "g8k"]
    assert maf.main(argv) == 0
    s = summary(tmp_path / "t")
    assert s["budget"] == 1000 and s["headline_variant"] == "g8k"          # the grid's last step, automatically
    assert all(r["run"].endswith("_g8k") for r in s["per_arena"])
    per = {r["arena"]: r for r in s["per_arena"]}
    assert per[6]["cost_half_gap"] == 250 and per[9]["cost_half_gap"] == 1000
    assert s["across_arenas"]["steps"] == list(steps8)
    assert s["recipe"] == []                  # the base-recipe runs anchor the recipe tests; they are not tests
    assert os.path.getsize(tmp_path / "f" / "fig4_adaptation.pdf") > 1000


# ---------------------------------------------------------------------------------------------
# Figure 3b: zero-shot A and M per arena through both decoders
# ---------------------------------------------------------------------------------------------

def write_paired(root, row, arena, a_stock, a_tuned, m_stock, m_tuned, name=None, tuned_twin=True):
    """One eval_tf read carrying stock and tuned columns (a `_tuned` twin) or stock columns only."""
    d = os.path.join(str(root), row + ("_tuned" if tuned_twin else ""), name or f"map{arena:02d}")
    os.makedirs(d, exist_ok=True)
    m = {"scene_psnr_dec": {"mean": 20.0 + a_stock, "n": 256}, "scene_copy_psnr_dec": {"mean": 20.0},
         "scene_lpips_raw": {"mean": 0.2 + m_stock}, "scene_persist_lpips_raw": {"mean": 0.2},
         "scene_psnr_raw": {"mean": 19.0 + a_stock}, "scene_persist_psnr_raw": {"mean": 19.0}}
    if tuned_twin:
        m.update({"scene_psnr_dec_tuned": {"mean": 21.0 + a_tuned}, "scene_copy_psnr_dec_tuned": {"mean": 21.0},
                  "scene_lpips_raw_tuned": {"mean": 0.2 + m_tuned}, "scene_psnr_raw_tuned": {"mean": 19.0 + a_tuned}})
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(m, f)


def test_figure_3b_pairs_the_decoders_per_backbone_in_zero_shot_skill_order(tmp_path):
    distances = write_distances(tmp_path / "distances.json")
    # arena 9 has the higher zero-shot skill, so it comes first although its number is larger
    write_run(tmp_path, 6, ARENA_A[6], skill=[1.0, 1.5, 2.0])
    write_run(tmp_path, 9, ARENA_A[9], skill=[1.8, 2.0, 2.2])
    fresh = tmp_path / "fresh"
    for arena, (a, m) in {6: (1.0, 0.10), 9: (2.0, 0.05)}.items():
        write_paired(fresh, "unet200k_ema", arena, a, a + 0.3, m, m - 0.03)
        write_paired(fresh, "pixart200k_ema", arena, a + 0.2, a + 0.4, m, m - 0.02)
        write_paired(fresh, "sd35_170000", arena, a - 0.5, None, m + 0.01, None, tuned_twin=False)
        write_paired(fresh, "adapt4000_live", arena, a + 1.5, a + 2.0, m - 0.08, m - 0.1)
    write_paired(fresh, "home_unet200k_ema", 0, 4.0, 5.0, -0.05, -0.08, name="val")
    out, tables = cli(tmp_path, distances, "--fresh-root", str(fresh))
    assert os.path.getsize(os.path.join(out, "fig3b_zero_shot_paired.png")) > 1000
    w, h = mediabox(os.path.join(out, "fig3b_zero_shot_paired.pdf"))
    assert w == pytest.approx(maf.FIG3B_SIZE[0], abs=0.01) and h == pytest.approx(maf.FIG3B_SIZE[1], abs=0.01)
    zp = summary(tables)["zero_shot_paired"]
    assert zp["order"] == [9, 6]
    rows = zp["rows"]
    assert set(rows) == {"unet", "pixart", "sd35", "adapter"}
    assert rows["unet"]["stock"]["maps"]["6"]["A"] == pytest.approx(1.0)
    assert rows["unet"]["tuned"]["maps"]["6"]["A"] == pytest.approx(1.3)
    assert rows["unet"]["tuned"]["maps"]["9"]["M"] == pytest.approx(0.02)
    assert rows["unet"]["tuned"]["home"]["A"] == pytest.approx(5.0)
    assert rows["unet"]["stock"]["home"]["A"] == pytest.approx(4.0)
    assert "tuned" not in rows["sd35"]                 # SD 3.5 has no tuned read: no filled marker is invented
    assert rows["sd35"]["stock"]["maps"]["9"]["A"] == pytest.approx(1.5)
    # the absolute form: scene PSNR and LPIPS against the raw frame, persistence beside every read
    ab = summary(tables)["zero_shot_absolute"]
    assert ab["unet"]["tuned"]["maps"]["6"]["psnr"] == pytest.approx(20.3)
    assert ab["unet"]["stock"]["maps"]["6"]["persist_psnr"] == pytest.approx(19.0)
    assert ab["pixart"]["tuned"]["maps"]["9"]["lpips"] == pytest.approx(0.2 + 0.03)
    assert ab["unet"]["tuned"]["home"]["psnr"] == pytest.approx(24.0)
    # the primary sits beside 3a in the paper owner's 3.15 x 1.5 in slot; the appendix's stock twin stays full width
    for stem, size in (("fig3b_zero_shot_absolute", (3.15, 1.5)), ("fig3b_zero_shot_absolute_stock", (5.5, 2.1))):
        assert os.path.getsize(os.path.join(out, f"{stem}.png")) > 1000, stem
        w, h = mediabox(os.path.join(out, f"{stem}.pdf"))
        assert (w, h) == (pytest.approx(size[0], abs=0.01), pytest.approx(size[1], abs=0.01)), stem


def test_the_crossing_rug_puts_every_arena_at_a_measured_budget_one_tick_each():
    import matplotlib.pyplot as plt
    maf.fs.style()
    fig, ax = plt.subplots()
    steps = [0, 250, 500, 1000, 2000, 4000]
    z = maf.step_axis(ax, steps)
    crossing = {"7": 250, "9": 250, "10": 250, "15": 500, "1": None, "8": None}
    stats = {"attainment_half_gap": {"crossing_step": crossing}}
    n0 = len(ax.lines)
    lanes = maf.crossing_rug(ax, stats, z, steps)
    assert lanes == 3                                    # three arenas tie at 250: three lanes
    assert len(ax.lines) - n0 == 6                       # one tick per arena
    xs = [x for ln in ax.lines[n0:] for x in ln.get_xdata()]
    assert max(xs) == 4000 and set(xs) <= {250, 500, 4000}   # censored arenas sit at the last read, not past it
    labels = sorted(t.get_text() for t in ax.texts)
    assert labels == sorted(["7", "9", "10", "15", "1", "8", "censored at 4k"])
    plt.close(fig)


def test_the_gap_share_variant_of_figure_4a_is_drawn(tmp_path):
    out, tables = cli(tmp_path, tree(tmp_path))
    assert os.path.getsize(os.path.join(out, "fig4a_gapshare.png")) > 1000
    share = summary(tables)["across_arenas"]["iqm_gap_share"]["value"]
    assert share[0] == pytest.approx(0.0) and share[-1] == pytest.approx(((3.0 - 1.0) / 3.0 + 0.9 / 2.0) / 2)


def test_the_dot_variant_of_figure_4_prints_at_the_paper_owners_size(tmp_path):
    distances = write_distances(tmp_path / "distances.json")
    write_run(tmp_path, 6, ARENA_A[6], skill=[1.0, 1.5, 2.0])
    write_run(tmp_path, 9, ARENA_A[9], skill=[1.8, 2.0, 2.2])
    out, tables = cli(tmp_path, distances)
    assert os.path.getsize(os.path.join(out, "fig4_dots.png")) > 1000
    w, h = mediabox(os.path.join(out, "fig4_dots.pdf"))
    assert (w, h) == (pytest.approx(5.5, abs=0.01), pytest.approx(1.4, abs=0.01))     # the body slot, unscaled
