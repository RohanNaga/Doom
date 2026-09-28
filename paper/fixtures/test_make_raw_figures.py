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
             f"scene_vae_psnr{sfx}": ceiling[0] + off, f"scene_vae_lpips{sfx}": ceiling[1] + off / 100,
             "scene_persist_lpips_raw": 0.20 + off / 100}
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
    # PixArt-alpha LoRA on both arenas: per-window raw reads through the fine-tuned decoder at 0, 250 and 4k
    for a in ARENAS:
        run = adapt / f"pixart200k_arenas13_map{a:02d}_r16_k8_s0"
        rows = []
        for step, lift in ((0, 0.0), (250, 1.0), (4000, 2.0)):
            stepdir = f"step{step:07d}_live_px"
            write_windows(str(run / "scores" / stepdir / "heldout" / "per_window.csv"),
                          windows(a, ZERO[a][0] + 0.2 + lift, ZERO[a][1] - 0.02 * lift, CEILING[a], "_tuned"))
            recorded = f"/home/x/results/adapt/{run.name}/scores/{stepdir}/heldout/per_window.csv"
            rows.append({"step": step, "weights": "live", "grid": [0, 250, 4000],
                         "heldout_decoders": ["stock", "tuned"],
                         "decoders": {"stock": {"identity": "hub:stock"}, "tuned": {"identity": "sha256:sd1tuned"}},
                         "eval_fingerprint": "px", "scored_at": f"t{step}", "heldout_per_window": recorded})
        with open(run / "scores.jsonl", "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")
        with open(run / "log.jsonl", "w") as f:
            f.write(json.dumps({"event": "start", "world": 1, "time": 0.0}) + "\n" +
                    json.dumps({"event": "checkpoint", "step": 4000, "time": 5400.0}) + "\n")
    # the U-Net's full fine-tune on arena 6: its 4k read only (its step-0 checkpoint went before scoring), from the
    # same checkpoint as the U-Net's 8k-grid LoRA runs (whose configs sit beside them)
    for a in ARENAS:
        with open(adapt / f"unet200k_arenas13_map{a:02d}_r16_k8_s0_g8k" / "config.json", "w") as f:
            json.dump({"source": "/home/x/results_040/snap_0200000.pt", "source_weights": "ema"}, f)
    run = adapt / "unet200k_arenas13_map06_r0_k8_s0"
    write_windows(str(run / "scores" / "step0004000_live_ft" / "heldout" / "per_window.csv"),
                  windows(6, 23.0, 0.18, CEILING[6], "_tuned"))
    with open(run / "config.json", "w") as f:
        json.dump({"source": "/sata2/x/040-unet-nexttic/snap_0200000.pt", "source_weights": "ema"}, f)
    with open(run / "scores.jsonl", "w") as f:
        f.write(json.dumps({"step": 4000, "weights": "live", "grid": [0, 4000], "heldout_decoders": ["stock", "tuned"],
                            "decoders": {"stock": {"identity": "hub:stock"}, "tuned": {"identity": "sha256:sd1tuned"}},
                            "eval_fingerprint": "ft", "scored_at": "t4000",
                            "heldout_per_window": "/sata2/x/adapt_fullft/map06/scores/step0004000_live_ft/heldout/"
                                                  "per_window.csv"}) + "\n")
    # PixArt-alpha LoRA on arena 16, still training: its zero-shot read only, far below the others
    run = adapt / "pixart200k_arenas13_map16_r16_k8_s0"
    write_windows(str(run / "scores" / "step0000000_live_px" / "heldout" / "per_window.csv"),
                  windows(16, 12.0, 0.60, CEILING[6], "_tuned"))
    with open(run / "scores.jsonl", "w") as f:
        f.write(json.dumps({"step": 0, "weights": "live", "grid": [0, 250, 4000],
                            "heldout_decoders": ["stock", "tuned"],
                            "decoders": {"stock": {"identity": "hub:stock"}, "tuned": {"identity": "sha256:sd1tuned"}},
                            "eval_fingerprint": "px", "scored_at": "t0",
                            "heldout_per_window": f"/home/x/results/adapt/{run.name}/scores/step0000000_live_px/"
                                                  "heldout/per_window.csv"}) + "\n")
    # SD 3.5 LoRA on arena 6 only, as the overnight runs leave it: every read of its six-step grid through the
    # fine-tuned SD 3.5 decoder (`tuned`), per-window raw files under scores/, while the only SD 3.5 training-map read
    # is the stock decoder's
    run = adapt / "sd35_200k_arenas13_map06_r16_k8_s0"
    sd_grid = [0, 250, 500, 1000, 2000, 4000]
    rows = []
    for step in sd_grid:
        stepdir = f"step{step:07d}_live_sd"
        write_windows(str(run / "scores" / stepdir / "heldout" / "per_window.csv"),
                      windows(6, 17.5 + 2.0 * step / 4000, 0.30 - 0.05 * step / 4000, CEILING[6], "_tuned"))
        rows.append({"step": step, "weights": "live", "grid": sd_grid, "heldout_decoders": ["stock", "tuned"],
                     "decoders": {"stock": {"identity": "hub:sd35"}, "tuned": {"identity": "sha256:sd35tuned"}},
                     "eval_fingerprint": "sd", "scored_at": f"t{step}",
                     "heldout_per_window": f"/sata2/x/adapt/{run.name}/scores/{stepdir}/heldout/per_window.csv"})
    with open(run / "scores.jsonl", "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    # SD 3.5 LoRA on arena 9 as Superman's runs arrive before Monday's rescore: the stock decoder, per-window files
    # against the decoded ground truth only (no raw columns)
    run = adapt / "sd35_200k_arenas13_map09_r16_k8_s0"
    rows = []
    for step in (0, 4000):
        stepdir = f"step{step:07d}_live_sm"
        write_windows(str(run / "scores" / stepdir / "heldout" / "per_window.csv"),
                      [{"episode": 100 + i, "map": 9, "dup_raw": 0, "dup_latent": 0,
                        "scene_psnr_dec": 30.0 + step / 4000 + off, "scene_lpips_dec": 0.10 + off / 100}
                       for i, off in enumerate((-0.3, -0.1, 0.1, 0.3))])
        rows.append({"step": step, "weights": "live", "grid": sd_grid, "heldout_decoders": ["stock"],
                     "decoders": {"stock": {"identity": "hub:sd35"}}, "eval_fingerprint": "sm",
                     "scored_at": f"t{step}",
                     "heldout_per_window": f"/home/x/adapt/{run.name}/scores/{stepdir}/heldout/per_window.csv"})
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


def test_a_group_budget_between_two_grid_reads_prints_both_reads():
    # budgets are reads on a log grid: an even group whose two middle reads differ prints the pair, not their mean
    assert mrf.middle_reads([250, 2000, 1000, 2000]) == [1000, 2000]
    assert mrf.budget_label([1000, 2000]) == "1k--2k"
    assert mrf.budget_label(mrf.middle_reads([150, 150, 100, 2000])) == "150"         # equal middle reads: one
    assert mrf.budget_label(mrf.middle_reads([50, 100, 150])) == "100"                # odd: the middle read
    assert mrf.budget_label(mrf.middle_reads([4000, None, 250, None])) == "4k--${>}$8k"   # censored counts as >8k
    assert mrf.budget_label(mrf.middle_reads([None, None])) == "${>}$8k"


def test_block_labels_count_arenas_and_mark_reads_against_the_decoded_ground_truth():
    # short labels (the table sets at \scriptsize in the body); the arena lists go to the table's comment lines
    full = {"n": 4, "n_wanted": 4, "wanted": [6, 7, 8, 16], "arenas": [6, 7, 8, 16], "quantity": "raw"}
    assert mrf.block_label("U-Net LoRA", full, False) == "U-Net LoRA (6, 7, 8, 16)"
    part = {"n": 1, "n_wanted": 4, "wanted": [6, 7, 8, 16], "arenas": [6], "quantity": "raw"}
    assert mrf.block_label("U-Net full fine-tune", part, False) == "U-Net full fine-tune (1 of 4)"
    some = {"n": 2, "n_wanted": 13, "wanted": list(range(13)), "arenas": [6, 7], "quantity": "dec"}
    assert mrf.block_label("SD 3.5 LoRA", some, True) == "SD 3.5 LoRA (2 of 13)$^\\ddagger$"
    none = {"n": 0, "n_wanted": 13, "wanted": list(range(13)), "arenas": [], "quantity": None}
    assert mrf.block_label("PixArt LoRA", none, True) == "PixArt LoRA (all 13)"
    assert mrf.block_label("PixArt LoRA", {**none, "n": 13, "arenas": list(range(13)), "quantity": "raw"},
                           True) == "PixArt LoRA (all 13)"


def test_a_block_mixing_raw_and_decoded_arenas_reports_the_raw_ones_and_lists_the_rest():
    def arena(p0, p4, quantity):
        reads = {st: {"quantity": quantity, "psnr": p, "lpips": 0.3 - 0.01 * p, "upper": 26.0 if quantity == "raw"
                      else None} for st, p in ((0, p0), (4000, p4))}
        return {"grid": [0, 4000], "reads": reads, "gpu_hours": {}}
    # Spiderman's two arenas read raw; Superman's two only against the decoded ground truth (30 dB and up)
    data = {6: arena(17.0, 19.0, "raw"), 7: arena(18.0, 20.0, "raw"), 8: arena(30.0, 31.0, "dec"),
            9: arena(31.0, 32.0, "dec")}
    b = mrf.block_summary(data, [6, 7, 8, 9], 3.0, "half_excess_gap", 4000)
    assert (b["n"], b["arenas"], b["quantity"], b["decoded_only"]) == (2, [6, 7], "raw", [8, 9])
    assert b["psnr_zero_shot"] == pytest.approx(17.5) and b["psnr_4k"] == pytest.approx(19.5)
    assert mrf.block_label("SD 3.5 LoRA", b, True) == "SD 3.5 LoRA (2 of 4)"                # no dagger
    empty = {"arenas": [], "n": 0, "budget_middle": None, "censored": None, **dict.fromkeys(
        ("psnr_zero_shot", "psnr_4k", "psnr_8k", "ceiling", "lpips_zero_shot", "lpips_4k", "lpips_8k"))}
    tex = mrf.groups_table({g: empty for g in list(mrf.GROUP_NAMES) + ["all"]}, budgets_final=False,
                           blocks=[("SD 3.5 LoRA (2 of 4)", b, "sha256:sd35tuned")])
    assert "arenas 8, 9 read against the decoded ground truth only" in tex
    # with no raw arena the block reports them all, daggered
    only_dec = mrf.block_summary({a: data[a] for a in (8, 9)}, [6, 7, 8, 9], 3.0, "half_excess_gap", 4000)
    assert (only_dec["arenas"], only_dec["quantity"], only_dec["decoded_only"]) == ([8, 9], "dec", [])


def test_an_arena_still_training_stays_out_of_its_block_until_it_has_the_headline_read():
    def arena(reads):
        return {"grid": [0, 250, 4000], "gpu_hours": {},
                "reads": {st: {"quantity": "raw", "psnr": p, "lpips": 0.3, "upper": 26.0} for st, p in reads.items()}}
    # arena 10 has its zero-shot and first reads only: in a median its 17 dB would sit in the zero-shot column only
    data = {6: arena({0: 20.0, 250: 21.0, 4000: 22.0}), 7: arena({0: 21.0, 250: 22.0, 4000: 23.0}),
            10: arena({0: 17.0, 250: 18.0})}
    b = mrf.block_summary(data, [6, 7, 10], 3.0, "half_excess_gap", 4000)
    assert (b["arenas"], b["in_progress"], b["decoded_only"]) == ([6, 7], [10], [])
    assert b["psnr_zero_shot"] == pytest.approx(20.5) and b["psnr_4k"] == pytest.approx(22.5)
    assert b["budget_final"] is True                                     # the two finished arenas have every read
    assert mrf.block_label("PixArt LoRA", b, True) == "PixArt LoRA (2 of 3)"
    empty = {"arenas": [], "n": 0, "budget_middle": None, "censored": None, **dict.fromkeys(
        ("psnr_zero_shot", "psnr_4k", "psnr_8k", "ceiling", "lpips_zero_shot", "lpips_4k", "lpips_8k"))}
    tex = mrf.groups_table({g: empty for g in list(mrf.GROUP_NAMES) + ["all"]}, budgets_final=False,
                           blocks=[("PixArt LoRA (2 of 3)", b, "sha256:sd1tuned")])
    assert "arenas 10 still training (no 4k read yet)" in tex
    assert mrf.finished(data[6], 4000) and not mrf.finished(data[10], 4000)
    assert mrf.finished({"grid": [0, 250], "reads": {0: {}, 250: {}}}, 4000)          # a grid that stops before


def test_the_shared_panel_averages_the_arenas_every_lora_backbone_has_finished():
    sets = {"unet_lora": [1, 6, 7, 8, 16], "pixart_lora": [6, 7, 8, 10], "sd35_lora": [6, 7, 8], "unet_full": [6]}
    assert mrf.shared_arenas(sets) == [6, 7, 8]                  # the full fine-tune does not shrink the set
    assert mrf.shared_arenas({"unet_lora": [6, 7]}) == [6, 7]
    assert mrf.shared_arenas({}) == []
    # a curve is drawn on the shared set only when it has every shared arena
    raw = {6: {0: {"psnr": 20.0, "lpips": 0.3}, 4000: {"psnr": 22.0, "lpips": 0.2}},
           7: {0: {"psnr": 21.0, "lpips": 0.4}, 4000: {"psnr": 24.0, "lpips": 0.1}},
           9: {0: {"psnr": 10.0, "lpips": 0.9}}}
    assert mrf.panel_points(raw, [6, 7]) == {0: (pytest.approx(20.5), pytest.approx(0.35)),
                                             4000: (pytest.approx(23.0), pytest.approx(0.15))}
    assert mrf.panel_points(raw, [6, 8]) is None
    assert mrf.panel_points(raw)[0][0] == pytest.approx(20.0)                   # every arena when no set is given


def write_sd35_200k(fresh, arenas=ARENAS, tuned=True):
    """The SD 3.5 200k zero-shot read as the steward writes it: arena and training-map files carrying the stock and
    (with `tuned`) the fine-tuned SD 3.5 decoder's columns from one pass."""
    for a in arenas:
        rows = windows(a, ZERO[a][0] - 0.4, ZERO[a][1], CEILING[a], "")
        if tuned:
            for r, t in zip(rows, windows(a, ZERO[a][0] - 0.2, ZERO[a][1] - 0.02, CEILING[a], "_tuned")):
                r.update(t)
        write_windows(str(fresh / "sd35_200000" / f"map{a:02d}" / "per_window.csv"), rows)
    home = []
    for m in TRAINING:
        rows = windows(m, 22.0 + 0.3 * (m - 2) - 0.4, 0.15, CEILING[m], "")
        if tuned:
            for r, t in zip(rows, windows(m, 22.0 + 0.3 * (m - 2) - 0.4, 0.12, CEILING[m], "_tuned")):
                r.update(t)
        home += rows
    write_windows(str(fresh / "home_sd35_200000" / "val" / "per_window.csv"), home)


def test_sd35_is_read_at_its_latest_checkpoint_through_its_own_fine_tuned_decoder(tmp_path):
    fresh, _, _ = export(tmp_path)
    fallback = {"unet200k_ema": ("unet200k_ema_tuned", "_tuned"), "pixart200k_ema": ("pixart200k_ema_tuned", "_tuned"),
                "sd35_170000": ("sd35_170000", "")}
    assert mrf.backbone_rows(str(fresh)) == fallback                  # no 200k files: the provisional stock read
    write_sd35_200k(fresh, tuned=False)
    assert mrf.backbone_rows(str(fresh)) == fallback                  # 200k without its decoder's columns: still
    write_sd35_200k(fresh)
    assert mrf.backbone_rows(str(fresh)) == {**{k: v for k, v in fallback.items() if not k.startswith("sd35")},
                                             "sd35_200000": ("sd35_200000", "_tuned")}


def test_the_raw_set_switches_sd35_to_200k_through_its_fine_tuned_decoder(tmp_path):
    fresh, dist, tdist = export(tmp_path)
    write_sd35_200k(fresh)
    out, side, tables = tmp_path / "figs", tmp_path / "raw_summary.json", tmp_path / "tables"
    assert mrf.main(["--fresh-root", str(fresh), "--distances", str(dist), "--training-distances", str(tdist),
                     "--out-dir", str(out), "--summary", str(side), "--tables-dir", str(tables),
                     "--family-step", str(tmp_path / "family_step.json"),
                     "--adapt-glob", str(tmp_path / "adapt" / "*_g8k"), "--adapt-root", str(tmp_path / "adapt")]) == 0
    s = json.load(open(side))
    # the step panels, the zero-shot and in-distribution reads: SD 3.5 at 200k through its fine-tuned decoder
    assert sorted(s["step"]) == ["pixart200k_ema", "sd35_200000", "unet200k_ema"]
    assert s["zero_shot"]["sd35_200000"]["6"]["psnr"] == pytest.approx(ZERO[6][0] - 0.2)          # the tuned column
    assert s["definitions"]["decoders"]["sd35_200000"] == "fine-tuned SD 3.5"
    assert s["sd35_row"] == {"row": "sd35_200000", "decoder": "tuned", "fallback": False}
    # the SD 3.5 block gets its in-distribution gap from the same read, so its budget can be final
    sd35 = s["blocks"]["sd35_lora"]
    assert sd35["g_train_source"] == "sd35_200000" and sd35["g_train"] == pytest.approx(28.0 - (22.45 - 0.4))
    assert sd35["budget_final"] is True


def test_a_full_fine_tune_without_its_step_0_read_borrows_the_lora_one_only_from_the_same_checkpoint():
    lora = {8: {"grid": [0, 4000], "gpu_hours": {}, "source": ("snap_0200000.pt", "ema"),
                "reads": {0: {"quantity": "raw", "psnr": 24.6, "lpips": 0.27, "upper": 29.6}}}}
    full = {8: {"grid": [0, 4000], "gpu_hours": {}, "source": ("snap_0200000.pt", "ema"),
                "reads": {4000: {"quantity": "raw", "psnr": 26.0, "lpips": 0.2, "upper": 29.6}}},
            16: {"grid": [0, 4000], "gpu_hours": {}, "source": ("snap_0150000.pt", "ema"),
                 "reads": {4000: {"quantity": "raw", "psnr": 25.0, "lpips": 0.2, "upper": 26.0}}}}
    lora[16] = {**lora[8], "reads": {0: {**lora[8]["reads"][0], "psnr": 22.4}}}
    out, borrowed = mrf.borrow_zero_shot(full, lora)
    assert borrowed == [8]                                             # 16 names another checkpoint: no borrowing
    assert out[8]["reads"][0]["psnr"] == pytest.approx(24.6) and out[8]["reads"][0]["borrowed"] == "unet_lora"
    assert 0 not in out[16]["reads"] and 0 not in full[8]["reads"]     # the input is left as it was
    b = mrf.block_summary(out, [6, 7, 8, 16], 3.0, "half_excess_gap", 4000)
    assert (b["n"], b["arenas"]) == (1, [8]) and mrf.block_label("U-Net full fine-tune", b, False) == \
        "U-Net full fine-tune (1 of 4)"


def test_a_block_scored_only_at_0_and_4k_prints_its_budget_as_at_most_4k():
    def arena(p4):
        return {"grid": [0, 4000], "gpu_hours": {}, "reads": {0: {"quantity": "raw", "psnr": 20.0, "lpips": 0.3,
                                                                  "upper": 26.0},
                                                              4000: {"quantity": "raw", "psnr": p4, "lpips": 0.2,
                                                                     "upper": 26.0}}}
    # threshold 26 - (6 + 3) / 2 = 21.5: reached by 4k at 22, not at 21
    empty = {"arenas": [], "n": 0, "budget_middle": None, "censored": None, **dict.fromkeys(
        ("psnr_zero_shot", "psnr_4k", "psnr_8k", "ceiling", "lpips_zero_shot", "lpips_4k", "lpips_8k"))}
    stats = {g: empty for g in list(mrf.GROUP_NAMES) + ["all"]}

    def cell(data):
        b = mrf.block_summary(data, [6, 7, 8, 16], 3.0, "half_excess_gap", 4000)
        tex = mrf.groups_table(stats, budgets_final=False, blocks=[("U-Net full fine-tune (x)", b, "sha256:t")])
        row = next(ln for ln in tex.splitlines() if ln.startswith("U-Net full fine-tune"))
        return b, row.split(" & ")[-1].rstrip(" \\"), tex
    b, budget, tex = cell({8: arena(22.0)})
    assert b["coarse_grid"] is True and budget == "${\\le}$4k"
    assert "budget grid 0 and 4k only" in tex
    _, budget, _ = cell({8: arena(21.0)})
    assert budget == "${>}$4k; 1 of 1 censored"                    # not reached by 4k: above 4k, never above 8k
    _, budget, _ = cell({8: arena(22.0), 16: arena(22.0), 6: arena(21.0)})
    assert budget == "${\\le}$4k; 1 of 3 censored"
    # a finer grid keeps the rule's own label
    fine = {8: {**arena(22.0), "grid": [0, 250, 4000]}}
    fine[8]["reads"][250] = {"quantity": "raw", "psnr": 21.8, "lpips": 0.25, "upper": 26.0}
    b, budget, tex = cell(fine)
    assert b["coarse_grid"] is False and budget == "250" and "budget grid 0 and 4k only" not in tex


def test_the_full_fine_tune_panel_waits_for_every_comparator_arena_the_lora_has():
    # the comparator arenas the U-Net LoRA has; drawn only when the full fine-tune has every one of them
    assert mrf.fullft_arenas([1, 6, 7, 8, 9, 16, 17], [6, 8, 16]) is None
    assert mrf.fullft_arenas([1, 6, 7, 8, 9, 16, 17], [6, 7, 8, 16]) == [6, 7, 8, 16]
    assert mrf.fullft_arenas([6, 9], [6]) == [6]                          # the LoRA has only arena 6 of the four
    assert mrf.fullft_arenas([6, 9], []) is None and mrf.fullft_arenas([9], [9]) is None


def test_backbone_end_labels_clear_every_mark_and_the_next_panel(tmp_path, monkeypatch):
    # the shared panel's case: three curves ending within 0.1 dB, two at 4k and the U-Net's at 8k
    kept = {}
    save = mrf.fs.save

    def keep(fig, out_dir, stem):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        px, lx = fig.axes[:2]
        labels = [t.get_window_extent(r) for t in px.texts if t.get_text().endswith(")")]
        marks = [px.transData.transform(p) for ln in px.lines if ln.get_marker() not in (None, "None", "", " ")
                 for p in ln.get_xydata()]
        right = [lx.yaxis.label.get_window_extent(r)] + [t.get_window_extent(r) for t in lx.get_yticklabels()
                                                         if t.get_text()]
        kept[stem] = (labels, marks, right)
        return save(fig, out_dir, stem)
    monkeypatch.setattr(mrf.fs, "save", keep)
    unet = {st: (21.7 + 1.2 * i / 9, 0.29 - 0.09 * i / 9) for i, st in enumerate(mrf.GRID)}
    unet[4000], unet[8000] = (22.90, 0.21), (22.93, 0.20)
    pix = {st: (21.9 + 1.07 * i / 8, 0.28 - 0.08 * i / 8) for i, st in enumerate(mrf.GRID[:-1])}
    sd = {0: (21.2, 0.28), 250: (22.6, 0.19), 500: (22.7, 0.185), 1000: (22.8, 0.18), 2000: (22.85, 0.172),
          4000: (22.94, 0.167)}
    curves = [("unet_lora", "U-Net", unet, 4), ("pixart_lora", "PixArt-$\\alpha$", pix, 4),
              ("sd35_lora", "SD 3.5", sd, 4)]
    mrf.fig_backbones(curves, {}, str(tmp_path), 4000, stem="raw_backbones_shared")
    labels, marks, right = kept["raw_backbones_shared"]
    assert len(labels) == 3
    assert not [(b, m) for b in labels for m in marks if b.contains(*m)]                 # no label on a mark
    assert not [(i, j) for i, a in enumerate(labels) for j, b in enumerate(labels) if i < j and a.overlaps(b)]
    assert not [(a, b) for a in labels for b in right if a.overlaps(b)]                  # clear of panel b


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
    tex = mrf.groups_table(stats, budgets_final=True)
    assert "1, 6, 7, 8" in tex and "9, 10, 11, 12, 13" in tex
    assert "& fine-tune" not in tex and "Full \\\\" not in tex   # the full fine-tune is a row block now, not a column
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
                     "--adapt-glob", str(tmp_path / "adapt" / "*_g8k"), "--adapt-root", str(tmp_path / "adapt")]) == 0
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
    # the headline is 4k updates (Rohan, Sep 27 evening); 8k is the check
    assert s["headline_step"] == 4000
    assert ba["psnr"]["median_recovery"] == pytest.approx(((19.0 - 18.0) + (23.5 - 19.0)) / 2)
    assert ba["lpips"]["median_recovery"] == pytest.approx(((0.22 - 0.30) + (0.19 - 0.26)) / 2)
    assert ba["psnr"]["median_recovery_8k"] == pytest.approx(((22.5 - 18.0) + (24.0 - 19.0)) / 2)
    # the raw trajectory of the row's curve panels ends on the filled 8k mark of its before-and-after panel
    assert s["trajectories"]["6"]["steps"] == [0, 4000, 8000]
    assert s["trajectories"]["6"]["psnr"] == [pytest.approx(18.0), pytest.approx(19.0), pytest.approx(22.5)]
    tex = open(tables / "adapt_groups.tex").read()
    assert "\\tbd" in tex and "upper bound" in tex
    # Table 2's per-backbone blocks at the 4k headline; the full fine-tune column is gone (its rows carry it)
    assert "& fine-tune \\\\" not in tex and "\\tbd{} (6" not in tex
    lines = {line.split(" & ")[0]: line for line in tex.splitlines() if " & " in line}
    assert "U-Net LoRA (1 of 4)" in lines
    # the full fine-tune's arena 6 has its 4k read and the U-Net LoRA's step 0 (the same checkpoint): 1 of 4
    ft = lines["U-Net full fine-tune (1 of 4)"]
    assert " & 18.00 & 23.00 & " in ft and "zero-shot of arenas 6 from the U-Net LoRA's step 0" in tex
    assert ft.endswith(" & ${\\le}$4k \\\\")                          # scored at 0 and 4k only
    px = lines["PixArt-$\\alpha$ LoRA (all 2)"]
    assert " & 20.70 & " in px and px.split(" & ")[3] == "--"                         # 4k median; its grid stops at 4k
    sd = lines["SD 3.5 LoRA (1 of 2)"]
    assert " & 19.50 & " in sd and "\\tbd{}" in sd                                    # budget waits on its grid
    # the arena lists live in the comment lines under the table
    assert "% SD 3.5 LoRA (1 of 2): arenas 6; decoder sha256:sd35tuned" in tex
    assert "% U-Net LoRA (1 of 4): arenas 6; decoder" in tex
    blocks = s["blocks"]
    assert blocks["pixart_lora"]["quantity"] == "raw" and blocks["pixart_lora"]["decoder"] == "tuned"
    assert blocks["pixart_lora"]["budget_final"] is True and blocks["pixart_lora"]["budget_middle"] == [4000]
    assert blocks["pixart_lora"]["gpu_hours_headline"] == pytest.approx(1.5)
    # SD 3.5's own fine-tuned decoder has no training-map read: no in-distribution gap and no budget, never the stock
    # decoder's gap (whose upper bound is another decoder's)
    sd35 = blocks["sd35_lora"]
    assert (sd35["decoder"], sd35["decoder_identity"]) == ("tuned", "sha256:sd35tuned")
    assert sd35["g_train"] is None and sd35["budget_final"] is False and "tuned" in sd35["g_train_note"]
    # Superman's decoded-only arena 9 stays out of the raw medians until it is rescored raw, and the table says so
    assert (sd35["arenas"], sd35["decoded_only"], sd35["quantity"]) == ([6], [9], "raw")
    assert "$^\\ddagger$" not in sd and "arenas 9 read against the decoded ground truth only" in tex
    assert blocks["pixart_lora"]["g_train"] == pytest.approx(28.0 - (22.45 + 0.2)) and not blocks["pixart_lora"].get(
        "g_train_note")
    assert (blocks["unet_full"]["n"], blocks["unet_full"]["zero_shot_borrowed"]) == (1, [6])
    # the every-arena panel: one LoRA curve per backbone; the full fine-tune (a few arenas against medians over
    # every arena) waits for the shared panel
    assert s["backbone_panel"]["drawn"] == ["unet_lora", "pixart_lora", "sd35_lora"]
    assert list(s["backbone_panel"]["not_drawn"]) == ["unet_full"]
    # PixArt's arena 16 is still training: out of its curve, whose every step is then over the same arenas
    assert s["backbone_panel"]["arenas"]["pixart_lora"] == [6, 9]
    assert s["backbone_panel"]["medians"]["pixart_lora"]["0"][0] == pytest.approx((18.2 + 19.2) / 2)
    # the second panel: every curve over the arenas all three LoRA backbones have finished (SD 3.5 has 6 raw only)
    shared = s["backbone_panel_shared"]
    # the full fine-tune has the shared arena (6), so it is drawn there, over the same arenas as the LoRAs
    assert shared["arenas"] == [6] and shared["drawn"] == ["unet_lora", "unet_full", "pixart_lora", "sd35_lora"]
    assert shared["medians"]["unet_full"]["4000"][0] == pytest.approx(23.0)
    assert shared["medians"]["pixart_lora"]["0"][0] == pytest.approx(18.2)
    assert shared["medians"]["unet_lora"]["0"][0] == pytest.approx(18.0)
    assert shared["medians"]["sd35_lora"]["4000"][0] == pytest.approx(19.5)
    # no 200k SD 3.5 files: the provisional 170k read through the stock decoder, recorded as the fallback
    assert s["sd35_row"] == {"row": "sd35_170000", "decoder": "stock", "fallback": True}
    # only the merged row's panels are lettered (a to d, in both layouts); the single supplement panels carry none
    # the row, its grid layout, both backbone panels and the full fine-tune panel
    assert letters == list("abcd") * 2 + list("ab") * 3
    # the full fine-tune panel: the U-Net LoRA against the full fine-tune on the comparator arenas both have (6)
    ft = s["fullft_panel"]
    assert (ft["arenas"], ft["drawn"]) == ([6], ["unet_lora", "unet_full"])
    assert sorted(ft["medians"]["unet_full"]) == ["0", "4000"]                        # two points, 0 borrowed
    assert ft["medians"]["unet_full"]["4000"][0] == pytest.approx(23.0)
    assert ft["medians"]["unet_lora"]["0"][0] == pytest.approx(18.0) == ft["medians"]["unet_full"]["0"][0]
    assert len(ft["medians"]["unet_lora"]) > 2                                       # the LoRA's whole grid
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
        for text in ["zero-shot", "after 4k updates", "reconstruction upper bound",
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
