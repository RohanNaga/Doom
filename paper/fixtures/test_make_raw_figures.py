"""`paper/make_raw_figures.py`: the persistence-free set (raw scene PSNR and LPIPS, the reconstruction ceiling and
the in-distribution level as the only references), on a synthetic export.

The fixture writes the fresh-rescore layout (`<root>/<row>/map<NN>/per_window.csv`, `home_<row>/val/per_window.csv`)
for two arenas (6, 9) and the four training maps, the adapter's raw-frame reads at 4k and 8k, the two distance
files, and two 8k-grid runs whose per-step per-window files carry the fine-tuned decoder's scene columns.

    python -m pytest paper/fixtures/test_make_raw_figures.py -q
"""
import csv
import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
sys.path.insert(0, PAPER)
sys.path.insert(0, os.path.dirname(PAPER))

pytest.importorskip("matplotlib")
pytest.importorskip("scipy")
import make_raw_figures as mrf  # noqa: E402

TRAINING = (2, 3, 4, 5)
ARENAS = (6, 9)
ALL_13 = (1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
# raw scene PSNR and LPIPS per map: training maps 22 to 23 dB, arenas 18 and 19 dB (the step holds)
ZERO = {6: (18.0, 0.30), 9: (19.0, 0.26)}
CEILING = {6: (26.0, 0.08), 9: (27.0, 0.07), 2: (28.0, 0.06), 3: (28.0, 0.06), 4: (28.0, 0.06), 5: (28.0, 0.06)}
ADAPTED = {4000: {6: (19.0, 0.22), 9: (23.5, 0.19)}, 8000: {6: (22.5, 0.20), 9: (24.0, 0.18)}}
ROWS = {4000: "adapt4000g8k_live_tuned", 8000: "adapt8000_live_tuned"}


def mediabox(path):
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", open(path, "rb").read())
    return float(m.group(1)) / 72, float(m.group(2)) / 72


def write_windows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def windows(map_, psnr, lpips, ceiling, sfx, n=4, dup=None, dec=None):
    """n windows of one map around the given means (offsets sum to zero); window `dup` flagged as a duplicate."""
    out = []
    for i, off in enumerate((-0.3, -0.1, 0.1, 0.3)[:n]):
        r = {"episode": 100 + i, "map": map_, "dup_raw": int(i == dup), "dup_latent": 0,
             f"scene_psnr_raw{sfx}": psnr + off, f"scene_lpips_raw{sfx}": lpips + off / 100,
             f"scene_vae_psnr{sfx}": ceiling[0] + off, f"scene_vae_lpips{sfx}": ceiling[1] + off / 100}
        if dec is not None:
            r.update({"scene_psnr_dec_tuned": dec[0] + off, "scene_lpips_dec_tuned": dec[1] + off / 100})
        out.append(r)
    return out


def export(tmp_path):
    fresh = tmp_path / "fresh"
    for row, shift in (("unet200k_ema", 0.0), ("pixart200k_ema", 0.2), ("sd35_170000", -0.5)):
        sfx = mrf.DECODER_SUFFIX[row]
        for a in ARENAS:
            write_windows(str(fresh / f"{row}{sfx}" / f"map{a:02d}" / "per_window.csv"),
                          windows(a, ZERO[a][0] + shift, ZERO[a][1], CEILING[a], sfx))
        home = [w for m in TRAINING for w in windows(m, 22.0 + 0.3 * (m - 2) + shift, 0.15, CEILING[m], sfx,
                                                     dec=(25.0, 0.10))]
        write_windows(str(fresh / f"home_{row}{sfx}" / "val" / "per_window.csv"), home)
    for step, reads in ADAPTED.items():
        for a, (p, lp) in reads.items():
            write_windows(str(fresh / ROWS[step] / f"map{a:02d}" / "per_window.csv"),
                          windows(a, p, lp, CEILING[a], "_tuned"))
    dist = tmp_path / "distances.json"
    json.dump({"primary_arm": "motion", "floor": {"motion": {"n": 4, "min": 0.02, "max": 0.09}},
               "maps": [{"map": 6, "role": "primary", "cluster": "arena", "D": 0.19},
                        {"map": 9, "role": "primary", "cluster": "arena", "D": 0.18}]}, open(dist, "w"))
    tdist = tmp_path / "training.json"
    json.dump({"maps": [{"map": m, "set": "val", "D": 0.03 + 0.01 * i} for i, m in enumerate(TRAINING)]},
              open(tdist, "w"))
    # the family-step record's zero-shot latent skill S0 per arena (tools/family_step.py)
    json.dump({"backbones": {"unet200k_ema": {"maps": {"6": {"S0": 1.0}, "9": {"S0": 2.0}}}}},
              open(tmp_path / "family_step.json", "w"))
    # the guard reads of the 8k-grid runs: directional and the training maps' decoded-reference scene PSNR at 0, 8k
    adapt = tmp_path / "adapt"
    for a, (psnr0, psnr8, dir0, dir8) in {6: (25.0, 24.0, 0.80, 0.85), 9: (25.0, 24.5, 0.70, 0.75)}.items():
        run = adapt / f"unet200k_arenas13_map{a:02d}_r16_k8_s0_g8k"
        rows = []
        for step, psnr, frac in ((0, psnr0, dir0), (8000, psnr8, dir8)):
            stepdir = f"step{step:07d}_live_guard"
            write_windows(str(run / "scores" / stepdir / "trainmap" / "per_window.csv"),
                          [{"episode": 6000 + i, "map": 2 + i % 4, "dup_latent": 0, "scene_psnr_dec": psnr + off}
                           for i, off in enumerate((-0.3, -0.1, 0.1, 0.3))])
            rows.append({"step": step, "weights": "live", "directional_correct_frac": frac,
                         "trainmap_per_window": f"/home/x/results/adapt/{run.name}/scores/{stepdir}/trainmap/"
                                                "per_window.csv"})
        with open(run / "scores.jsonl", "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    return fresh, dist, tdist


def test_means_by_map_leave_duplicates_out_and_group_by_the_map_column(tmp_path):
    path = str(tmp_path / "pw.csv")
    write_windows(path, windows(2, 20.0, 0.2, (28.0, 0.06), "_tuned", dup=0) +
                  windows(3, 21.0, 0.2, (28.0, 0.06), "_tuned"))
    got = mrf.means_by_map(mrf.read_windows(path), ["scene_psnr_raw_tuned"])
    assert got[3]["scene_psnr_raw_tuned"] == pytest.approx(21.0) and got[3]["n"] == 4
    # map 2 loses its first window (offset -0.3): the mean of offsets -0.1, 0.1 and 0.3
    assert got[2]["scene_psnr_raw_tuned"] == pytest.approx(20.1) and got[2]["n"] == 3


def test_the_half_ceiling_budget_is_the_first_read_past_halfway_and_censored_otherwise():
    # zero-shot 18, ceiling 26: halfway is 22
    assert mrf.half_ceiling_budget(18.0, [(4000, 21.0), (8000, 22.5)], 26.0) == 8000
    assert mrf.half_ceiling_budget(18.0, [(8000, 22.0), (4000, 22.0)], 26.0) == 4000      # reaching counts
    assert mrf.half_ceiling_budget(18.0, [(4000, 20.0), (8000, 21.9)], 26.0) is None


def test_arenas_are_coloured_by_their_group_and_the_key_uses_the_same_three_shades():
    from matplotlib import colors
    groups = [("hard", [7, 11, 14, 15]), ("medium", [8, 9]), ("easy", [16])]
    colour_of, key = mrf.group_colours(groups)
    assert [name for name, _ in key] == ["hard", "medium", "easy"]
    shade = dict(key)
    assert colour_of[7] == colour_of[14] == colour_of[15] == shade["hard"]         # one shade per group, as keyed
    assert colour_of[8] == shade["medium"] and colour_of[16] == shade["easy"]
    lum = {g: sum(colors.to_rgb(c)) for g, c in key}
    assert lum["hard"] < lum["medium"] < lum["easy"]                                # dark = hard


def test_the_budget_rules_set_their_thresholds_in_raw_psnr():
    # zero-shot 18, ceiling 26 (gap 8), the training maps' own gap to the ceiling 3
    assert mrf.budget_threshold("half_ceiling_gap", 18.0, 26.0, 3.0) == pytest.approx(22.0)
    assert mrf.budget_threshold("half_excess_gap", 18.0, 26.0, 3.0) == pytest.approx(26.0 - (8.0 + 3.0) / 2)
    assert mrf.budget_threshold("in_distribution_gap", 18.0, 26.0, 3.0) == pytest.approx(23.0)


def test_the_step_holds_only_when_every_arena_is_on_the_far_side_of_every_training_map():
    assert mrf.step_check([22.0, 23.0], [18.0, 19.0]) == {"holds": True, "margin": pytest.approx(3.0)}
    assert mrf.step_check([22.0, 23.0], [18.0, 22.5])["holds"] is False
    # LPIPS: lower is better, so the arenas must all sit above
    assert mrf.step_check([0.15, 0.16], [0.26, 0.30], higher_is_better=False) == {
        "holds": True, "margin": pytest.approx(0.10)}


def test_arenas_group_by_zero_shot_psnr_four_five_four_and_the_table_lists_them():
    z = {a: 10.0 + i for i, a in enumerate(ALL_13)}          # arena 1 is the hardest, 17 the easiest
    groups = mrf.group_arenas(z)
    assert groups == [("hard", [1, 6, 7, 8]), ("medium", [9, 10, 11, 12, 13]), ("easy", [14, 15, 16, 17])]
    per = {a: {"zero_shot": z[a], "psnr_4k": z[a] + 2.0, "psnr_8k": z[a] + 3.0, "ceiling": z[a] + 8.0,
               "lpips_zero_shot": 0.30, "lpips_4k": 0.22, "lpips_8k": 0.20,
               "budget": 4000 if a < 12 else (8000 if a == 12 else None)} for a in z}
    stats = mrf.group_stats(groups, per)
    assert stats["hard"]["psnr_zero_shot"] == pytest.approx(11.5) and stats["hard"]["psnr_8k"] == pytest.approx(14.5)
    assert stats["hard"]["budget"]["value"] == 4000 and stats["hard"]["censored"] == 0
    assert stats["easy"]["budget"]["censored"] is True and stats["easy"]["censored"] == 4
    assert stats["all"]["n"] == 13 and stats["all"]["censored"] == 5
    tex = mrf.groups_table(stats, comparator=(6, 7, 8, 16), budgets_final=True)
    assert "1, 6, 7, 8" in tex and "9, 10, 11, 12, 13" in tex
    assert "\\tbd{} (6, 7, 8)" in tex and "\\tbd{} (16)" in tex and "\\tbd{} (6, 7, 8, 16)" in tex
    assert "${>}$8k; 4 of 4 censored" in tex
    assert "arena 7" not in tex.lower() and "ceiling" not in tex.lower() and "upper bound" in tex
    # until every grid step has a raw read, the budget cells wait
    assert "${>}$8k" not in mrf.groups_table(stats, budgets_final=False)
    # by the zero-shot gap to the ceiling instead, the largest gap is the hardest
    gap = {a: 20.0 - i for i, a in enumerate(ALL_13)}                    # arena 1 has the largest gap
    assert mrf.group_arenas(gap, higher_is_harder=True)[0] == ("hard", [1, 6, 7, 8])
    # an empty group (fewer than three arenas) prints dashes, not a crash
    two = mrf.group_stats(mrf.group_arenas({6: 1.0, 9: 2.0}), {a: per[a] for a in (6, 9)})
    assert "--" in mrf.groups_table(two)


def test_adapter_rows_are_the_8k_grid_runs_reads_by_step(tmp_path):
    for d in ("adapt4000_live_tuned", "adapt4000_live", "adapt4000g8k_live_tuned", "adapt8000_live_tuned",
              "adapt250g8k_live_tuned", "unet200k_ema_tuned"):
        os.makedirs(tmp_path / d)
    assert mrf.adapter_rows(str(tmp_path)) == {250: "adapt250g8k_live_tuned", 4000: "adapt4000g8k_live_tuned",
                                               8000: "adapt8000_live_tuned"}


def test_the_raw_set_is_drawn_at_its_slot_sizes_and_the_numbers_are_recorded(tmp_path, monkeypatch):
    fresh, dist, tdist = export(tmp_path)
    letters = []
    draw_letter = mrf.fs.panel_letter

    def record_letter(ax, letter, **kw):
        letters.append(letter)
        return draw_letter(ax, letter, **kw)
    monkeypatch.setattr(mrf.fs, "panel_letter", record_letter)
    legends = {}
    save = mrf.fs.save

    def record_legends(fig, out_dir, stem):
        found = list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
        legends[stem] = [t.get_text() for lg in found for t in lg.get_texts()]
        return save(fig, out_dir, stem)
    monkeypatch.setattr(mrf.fs, "save", record_legends)
    out, side, tables = tmp_path / "figs", tmp_path / "raw_summary.json", tmp_path / "tables"
    assert mrf.main(["--fresh-root", str(fresh), "--distances", str(dist), "--training-distances", str(tdist),
                     "--out-dir", str(out), "--summary", str(side), "--tables-dir", str(tables),
                     "--family-step", str(tmp_path / "family_step.json"),
                     "--adapt-glob", str(tmp_path / "adapt" / "*_g8k")]) == 0
    for stem, size in mrf.SIZES.items():
        w, h = mediabox(os.path.join(out, f"{stem}.pdf"))
        assert (w, h) == (pytest.approx(size[0], abs=0.01), pytest.approx(size[1], abs=0.01)), stem
    s = json.load(open(side))
    # the step without persistence: every arena below every training map in PSNR and above in LPIPS, all backbones
    assert all(s["step"][b]["psnr"]["holds"] and s["step"][b]["lpips"]["holds"] for b in mrf.BACKBONES)
    # the budget closes half of the excess gap over the training maps' own gap to the upper bound (28 - 22.45)
    g_id = 28.0 - sum(22.0 + 0.3 * (m - 2) for m in TRAINING) / 4
    ab = s["adaptation"]
    assert ab["in_distribution_gap"] == pytest.approx(g_id)
    assert ab["per_arena"]["6"]["threshold"] == pytest.approx(26.0 - (8.0 + g_id) / 2)       # 19.2: 4k (19.0) short
    assert ab["per_arena"]["6"]["budget"] == 8000 and ab["per_arena"]["9"]["budget"] == 4000
    assert (ab["past_half_by_4k"], ab["past_half_by_8k"]) == (1, 2) and ab["grid_complete"] is False
    assert ab["median_budget"] == {"value": 6000.0, "censored": False, "label": "6000"}
    # arenas group and colour by the zero-shot gap to the upper bound (both gaps are 8 dB here)
    assert s["group_key"] == "zero-shot gap to the reconstruction upper bound (dB)"
    assert s["groups"]["all"]["n"] == 2
    # before and after: in-distribution minus zero-shot, and adapted (8k) minus zero-shot, medians over arenas
    ba = s["before_after"]
    home_psnr = 28.0 - g_id
    assert ba["psnr"]["in_distribution"] == pytest.approx(home_psnr)
    assert ba["psnr"]["median_drop"] == pytest.approx(home_psnr - 18.5)
    assert ba["psnr"]["median_recovery"] == pytest.approx(((22.5 - 18.0) + (24.0 - 19.0)) / 2)
    assert ba["lpips"]["median_recovery"] == pytest.approx(((0.20 - 0.30) + (0.18 - 0.26)) / 2)
    # the raw trajectory of the row's curve panels ends on the filled 8k mark of its before-and-after panel
    assert s["trajectories"]["6"]["steps"] == [0, 4000, 8000]
    assert s["trajectories"]["6"]["psnr"] == [pytest.approx(18.0), pytest.approx(19.0), pytest.approx(22.5)]
    tex = open(tables / "adapt_groups.tex").read()
    assert "\\tbd" in tex and "upper bound" in tex
    # only the merged row's panels are lettered (a to d, in both layouts); the single supplement panels carry none
    assert letters == list("abcd") * 2
    # the appendix's per-arena table: the gap, its threshold, the reads, the guards
    per = s["per_arena_table"]
    assert per["6"]["gap_zero_shot"] == pytest.approx(8.0)
    assert per["6"]["threshold_gap"] == pytest.approx((8.0 + g_id) / 2)
    assert per["6"]["forgetting_dec"] == pytest.approx(-1.0) and per["9"]["forgetting_dec"] == pytest.approx(-0.5)
    assert (per["6"]["directional_0"], per["6"]["directional_8k"]) == (pytest.approx(0.80), pytest.approx(0.85))
    rows = open(tables / "adapt_perarena.tex").read()
    assert "\\tbd" in rows and "$-$1.00" in rows and "0.85" in rows and "decoded" in rows
    # legends: the marks and the colour key on the row (both layouts), the backbones and the band on the step panels
    shown = [g for g in ("hard", "medium", "easy") if s["groups"][g]["n"]]      # two arenas: medium only
    assert shown == ["medium"]
    for stem in ("raw_row", "raw_row_grid"):
        for text in ["zero-shot", "after 8k updates", "reconstruction upper bound",
                     "training maps (in distribution)", "zero-shot gap to the upper bound:"] + shown:
            assert text in legends[stem], (stem, text)
        assert not {"hard", "easy"} & set(legends[stem])               # an empty group gets no swatch
    for stem in ("raw_fig3a_psnr", "raw_fig3a_lpips"):
        assert {"U-Net", "PixArt-α", "SD 3.5", "train vs train $d$"} <= set(legends[stem]), stem
    # the numbers the text waits on: counts and medians with arena-bootstrap intervals, provisional until the grid
    t = s["text_numbers"]
    assert t["provisional"] is True and t["pending"]["forgetting"] and t["pending"]["gain_step"]
    assert t["past_threshold_by_8k"]["count"] == 2 and len(t["past_threshold_by_8k"]["interval"]) == 2
    assert t["past_threshold_by_4k"]["count"] == 1
    assert t["median_gap_8k"]["value"] == pytest.approx(((26.0 - 22.5) + (27.0 - 24.0)) / 2)
    # the share of each arena's 0-to-8k raw gain in place by 4k, averaged: (1/4.5 + 4.5/5) / 2
    assert t["gain_share_by_step"]["4000"] == pytest.approx((1.0 / 4.5 + 4.5 / 5.0) / 2)
    assert set(t["spearman_S0"]) == {"gap_8k", "gain_8k", "budget"}          # two arenas: no rho yet
    assert t["spearman_S0"]["gap_8k"]["n"] == 2
