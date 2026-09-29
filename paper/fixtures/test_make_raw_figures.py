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
    # the U-Net's directional-check reads (tools/directional_check.py) for Table 1's column: the four training maps
    # and the two arenas, arena 9 on half the windows so the pooled read has to weigh by windows
    for m, set_name, n_windows, correct in ((2, "val", 128, 0.8), (3, "val", 128, 0.9), (4, "val", 128, 0.8),
                                            (5, "val", 128, 0.9), (6, "arenas13", 128, 0.7), (9, "arenas13", 64, 0.8)):
        path = tmp_path / "directional" / "unet200k_ema" / f"directional_map{m:02d}_{set_name}_ema.json"
        os.makedirs(path.parent, exist_ok=True)
        json.dump({"summary": {"windows": n_windows, "correct_frac": correct, "ref_raw_frac": 0.9}}, open(path, "w"))
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
                    json.dumps({"event": "certificate", "line": "ADAPT_CERTIFICATE seed=0 source=/sata2/x/041-pixart-"
                                                                "nexttic/snap_0200000.pt world=1"}) + "\n" +
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


def no_full_grid(tmp_path):
    """The full-grid roots pointed at empty folders: the defaults are the repo's real results."""
    return ["--sd35-full-root", str(tmp_path / "none_sd35"), "--pixart-full-root", str(tmp_path / "none_pixart"),
            "--fullft-g8k-root", str(tmp_path / "none_fullft")]


def test_means_by_map_leave_duplicates_out_and_group_by_the_map_column(tmp_path):
    path = str(tmp_path / "pw.csv")
    write_windows(path, windows(2, 20.0, 0.2, (28.0, 0.06), "_tuned", dup=0) +
                  windows(3, 21.0, 0.2, (28.0, 0.06), "_tuned"))
    got = mrf.means_by_map(mrf.read_windows(path), ["scene_psnr_raw_tuned"])
    assert got[3]["scene_psnr_raw_tuned"] == pytest.approx(21.0) and got[3]["n"] == 4
    # map 2 loses its first window (offset -0.3): the mean of offsets -0.1, 0.1 and 0.3
    assert got[2]["scene_psnr_raw_tuned"] == pytest.approx(20.1) and got[2]["n"] == 3


def test_the_budget_is_the_first_read_recovering_half_of_the_lpips_rise_and_censored_otherwise():
    # zero-shot LPIPS 0.30, training maps 0.15: the rise is 0.15, half of it is recovered at 0.225 (Rohan, Sep 29)
    assert mrf.lpips_threshold(0.30, 0.15) == pytest.approx(0.225)
    assert mrf.budget_step([(4000, 0.23), (8000, 0.22)], 0.225) == 8000
    assert mrf.budget_step([(8000, 0.225), (4000, 0.225)], 0.225) == 4000              # reaching counts
    assert mrf.budget_step([(4000, 0.23), (8000, 0.226)], 0.225) is None               # censored
    # a share: recovered over the deficit, None when there is no deficit to recover
    assert mrf.share(1.0, 4.0) == pytest.approx(0.25) and mrf.share(1.0, 0.0) is None and mrf.share(None, 4.0) is None


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
    assert mrf.budget_label(mrf.middle_reads([None, 4000]), last=4000) == "4k--${>}$4k"   # a grid that ends at 4k


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
    b = mrf.block_summary(data, [6, 7, 8, 9], (22.0, 0.15), 4000)
    assert (b["n"], b["arenas"], b["quantity"], b["decoded_only"]) == (2, [6, 7], "raw", [8, 9])
    assert b["psnr_zero_shot"] == pytest.approx(17.5) and b["psnr_4k"] == pytest.approx(19.5)
    assert mrf.block_label("SD 3.5 LoRA", b, True) == "SD 3.5 LoRA (2 of 4)"                # no dagger
    empty = {"arenas": [], "n": 0, "budget_middle": None, "censored": None, **dict.fromkeys(
        ("psnr_zero_shot", "psnr_4k", "psnr_8k", "lpips_zero_shot", "lpips_4k", "lpips_8k", "lpips_share_4k",
         "psnr_share_4k"))}
    tex = mrf.groups_table({g: empty for g in list(mrf.GROUP_NAMES) + ["all"]}, budgets_final=False,
                           blocks=[("SD 3.5 LoRA (2 of 4)", b, "sha256:sd35tuned")])
    assert "maps 8, 9 read against the decoded ground truth only" in tex
    # with no raw arena the block reports them all, daggered
    only_dec = mrf.block_summary({a: data[a] for a in (8, 9)}, [6, 7, 8, 9], (22.0, 0.15), 4000)
    assert (only_dec["arenas"], only_dec["quantity"], only_dec["decoded_only"]) == ([8, 9], "dec", [])


def test_an_arena_still_training_stays_out_of_its_block_until_it_has_the_headline_read():
    def arena(reads):
        return {"grid": [0, 250, 4000], "gpu_hours": {},
                "reads": {st: {"quantity": "raw", "psnr": p, "lpips": 0.3, "upper": 26.0} for st, p in reads.items()}}
    # arena 10 has its zero-shot and first reads only: in a median its 17 dB would sit in the zero-shot column only
    data = {6: arena({0: 20.0, 250: 21.0, 4000: 22.0}), 7: arena({0: 21.0, 250: 22.0, 4000: 23.0}),
            10: arena({0: 17.0, 250: 18.0})}
    b = mrf.block_summary(data, [6, 7, 10], (22.0, 0.15), 4000)
    assert (b["arenas"], b["in_progress"], b["decoded_only"]) == ([6, 7], [10], [])
    assert b["psnr_zero_shot"] == pytest.approx(20.5) and b["psnr_4k"] == pytest.approx(22.5)
    assert b["budget_final"] is True                                     # the two finished arenas have every read
    assert mrf.block_label("PixArt LoRA", b, True) == "PixArt LoRA (2 of 3)"
    empty = {"arenas": [], "n": 0, "budget_middle": None, "censored": None, **dict.fromkeys(
        ("psnr_zero_shot", "psnr_4k", "psnr_8k", "lpips_zero_shot", "lpips_4k", "lpips_8k", "lpips_share_4k",
         "psnr_share_4k"))}
    tex = mrf.groups_table({g: empty for g in list(mrf.GROUP_NAMES) + ["all"]}, budgets_final=False,
                           blocks=[("PixArt LoRA (2 of 3)", b, "sha256:sd1tuned")])
    assert "maps 10 still training (no 4k read yet)" in tex
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
                     "--adapt-glob", str(tmp_path / "adapt" / "*_g8k"), "--adapt-root", str(tmp_path / "adapt"),
                     *no_full_grid(tmp_path)]) == 0
    s = json.load(open(side))
    # the step panels, the zero-shot and in-distribution reads: SD 3.5 at 200k through its fine-tuned decoder
    assert sorted(s["step"]) == ["pixart200k_ema", "sd35_200000", "unet200k_ema"]
    assert s["zero_shot"]["sd35_200000"]["6"]["psnr"] == pytest.approx(ZERO[6][0] - 0.2)          # the tuned column
    assert s["definitions"]["decoders"]["sd35_200000"] == "fine-tuned SD 3.5"
    assert s["sd35_row"] == {"row": "sd35_200000", "decoder": "tuned", "fallback": False}
    # the SD 3.5 block gets its in-distribution level from the same read, so its budget can be final
    sd35 = s["blocks"]["sd35_lora"]
    assert sd35["level_source"] == "sd35_200000" and sd35["level"][0] == pytest.approx(22.45 - 0.4)
    assert sd35["budget_final"] is True


def test_gpu_hours_are_reported_per_card_and_never_as_one_median():
    # SD 3.5's arenas trained on two cards (1.33 h on an A6000, 3.35 h on an A4000): one median would be neither
    def arena(hours, card):
        return {"grid": [0, 4000], "gpu_hours": {} if hours is None else {"4000": hours}, "card": card,
                "reads": {st: {"quantity": "raw", "psnr": 20.0 + st / 4000, "lpips": 0.3, "upper": 26.0}
                          for st in (0, 4000)}}
    data = {6: arena(1.33, "A6000"), 7: arena(1.34, "A6000"), 9: arena(3.38, "A4000"), 11: arena(3.34, "A4000"),
            13: arena(3.35, "A4000"), 16: arena(2.0, None), 17: arena(None, "A6000")}
    b = mrf.block_summary(data, [6, 7, 9, 11, 13, 16, 17], (22.0, 0.15), 4000)
    assert "gpu_hours_headline" not in b
    by = b["gpu_hours_headline_by_card"]
    assert by["A6000"] == {"median": pytest.approx(1.335), "n": 2, "arenas": [6, 7]}     # 17 has no log time
    assert by["A4000"] == {"median": pytest.approx(3.35), "n": 3, "arenas": [9, 11, 13]}
    assert by["unknown"] == {"median": pytest.approx(2.0), "n": 1, "arenas": [16]}
    assert list(by) == ["A4000", "A6000", "unknown"]


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
    b = mrf.block_summary(out, [6, 7, 8, 16], (22.0, 0.15), 4000)
    assert (b["n"], b["arenas"]) == (1, [8]) and mrf.block_label("U-Net full fine-tune", b, False) == \
        "U-Net full fine-tune (1 of 4)"


def test_a_block_scored_only_at_0_and_4k_prints_its_budget_as_at_most_4k():
    def arena(l4):
        return {"grid": [0, 4000], "gpu_hours": {}, "reads": {0: {"quantity": "raw", "psnr": 20.0, "lpips": 0.3},
                                                              4000: {"quantity": "raw", "psnr": 22.0, "lpips": l4}}}
    # training maps at 0.15, zero-shot 0.30: half of the rise is recovered at 0.225, by 4k at 0.22, not at 0.23
    empty = {"arenas": [], "n": 0, "budget_middle": None, "censored": None, **dict.fromkeys(
        ("psnr_zero_shot", "psnr_4k", "psnr_8k", "lpips_zero_shot", "lpips_4k", "lpips_8k", "lpips_share_4k",
         "psnr_share_4k"))}
    stats = {g: empty for g in list(mrf.GROUP_NAMES) + ["all"]}

    def cell(data):
        b = mrf.block_summary(data, [6, 7, 8, 16], (22.0, 0.15), 4000)
        tex = mrf.groups_table(stats, budgets_final=False, blocks=[("U-Net full fine-tune (x)", b, "sha256:t")])
        row = next(ln for ln in tex.splitlines() if ln.startswith("U-Net full fine-tune"))
        return b, row.split(" & ")[-1].rstrip(" \\"), tex
    b, budget, tex = cell({8: arena(0.22)})
    assert b["coarse_grid"] is True and budget == "${\\le}$4k"
    assert "budget grid starts at 4k" in tex
    _, budget, _ = cell({8: arena(0.23)})
    assert budget == "${>}$4k; 1 of 1 censored"                    # not reached by 4k: above 4k, never above 8k
    _, budget, _ = cell({8: arena(0.22), 16: arena(0.22), 6: arena(0.23)})
    assert budget == "${\\le}$4k; 1 of 3 censored"
    # a finer grid keeps the rule's own label, and a crossing at the grid's first read is named
    fine = {8: {**arena(0.22), "grid": [0, 250, 4000]}}
    fine[8]["reads"][250] = {"quantity": "raw", "psnr": 21.8, "lpips": 0.22}
    b, budget, tex = cell(fine)
    assert b["coarse_grid"] is False and budget == "250" and "budget grid starts at 4k" not in tex
    assert b["budget_at_first_read"] == [(8, 250)] and "first point of its grid for maps 8 (250)" in tex
    # a run whose grid stops short still gives a budget when it crosses on the reads it has
    short = {8: {**arena(0.22), "grid": [0, 250, 4000, 8000]}}
    short[8]["reads"][250] = {"quantity": "raw", "psnr": 21.8, "lpips": 0.25}
    b, budget, _ = cell(short)
    assert b["incomplete_grid"] == [8] and b["budget_final"] is True and budget == "4k"


def test_the_full_fine_tune_panel_waits_for_every_comparator_arena_the_lora_has():
    # the comparator arenas the U-Net LoRA has; drawn only when the full fine-tune has every one of them
    assert mrf.fullft_arenas([1, 6, 7, 8, 9, 16, 17], [6, 8, 16]) is None
    assert mrf.fullft_arenas([1, 6, 7, 8, 9, 16, 17], [6, 7, 8, 16]) == [6, 7, 8, 16]
    assert mrf.fullft_arenas([6, 9], [6]) == [6]                          # the LoRA has only arena 6 of the four
    assert mrf.fullft_arenas([6, 9], []) is None and mrf.fullft_arenas([9], [9]) is None


def test_the_full_fine_tune_is_two_bare_points_keyed_by_its_reads(tmp_path, monkeypatch):
    from matplotlib.colors import same_color
    kept = {}
    save = mrf.fs.save

    def keep(fig, out_dir, stem):
        px = fig.axes[0]
        black = [ln for ln in px.lines if same_color(ln.get_color(), mrf.fs.FULL_FINE_TUNE.colour)
                 and ln.get_gid() not in ("ref", "decor")]                   # data, not the axis-break mark
        kept[stem] = (black, [t.get_text() for t in px.texts])
        return save(fig, out_dir, stem)
    monkeypatch.setattr(mrf.fs, "save", keep)
    lora = {st: (21.7 + 0.1 * i, 0.29 - 0.01 * i) for i, st in enumerate(mrf.GRID)}
    full = {0: (21.7, 0.29), 4000: (23.1, 0.18)}
    mrf.fig_backbones([("unet_lora", "U-Net LoRA", lora, 4), ("unet_full", mrf.fullft_label(full), full, None)], {},
                      str(tmp_path), 4000, stem="raw_fullft")
    black, texts = kept["raw_fullft"]
    assert black and all(ln.get_linestyle() == "None" for ln in black)         # points only: nothing between 0 and 4k
    assert sorted(x for ln in black for x, _ in ln.get_xydata()) and "full fine-tune (0, 4k)" in texts
    assert "U-Net LoRA (4)" in texts


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


def slim_block(n, wanted, p0, p4, l0, l4, upper, middle, coarse=False, quantity="raw", final=True, shares=None):
    """A block summary as `slim_table` reads it; `shares` is (lpips, psnr, gap) as fractions, the medians of
    `recovery_shares`."""
    return {"n": n, "n_wanted": len(wanted), "wanted": list(wanted), "arenas": list(wanted)[:n], "quantity": quantity,
            "psnr_zero_shot": p0, "psnr_4k": p4, "lpips_zero_shot": l0, "lpips_4k": l4, "ceiling": upper,
            "budget_middle": middle, "budget_final": final, "censored": 0, "coarse_grid": coarse, "headline": 4000,
            "shares": {k: {"median": v, "min": v, "n": n} for k, v in zip(("lpips", "psnr", "gap"), shares)}
            if shares else None}


def test_recovery_shares_are_per_arena_medians_against_the_training_maps_level():
    data = {6: {"reads": {0: {"psnr": 18.0, "lpips": 0.30, "upper": 26.0}, 4000: {"psnr": 19.0, "lpips": 0.22}}},
            9: {"reads": {0: {"psnr": 19.0, "lpips": 0.26, "upper": 27.0}, 4000: {"psnr": 23.5, "lpips": 0.19}}},
            # an arena that starts past the training maps in PSNR and LPIPS and inside their gap to the upper bound
            7: {"reads": {0: {"psnr": 23.0, "lpips": 0.10, "upper": 27.0}, 4000: {"psnr": 23.4, "lpips": 0.09}}}}
    got = mrf.recovery_shares(data, [6, 9, 7], (22.45, 0.15))
    # map 6 undoes 0.08 of its 0.15 LPIPS rise and regains 1 of its 4.45 dB deficit; map 9: 0.07 of 0.11 and 4.5 of
    # 3.45 (past the training maps: above 1); map 7 has no deficit and stays out of both
    assert got["lpips"] == {"median": pytest.approx((0.08 / 0.15 + 0.07 / 0.11) / 2),
                            "min": pytest.approx(0.08 / 0.15), "n": 2}
    assert got["psnr"] == {"median": pytest.approx((1 / 4.45 + 4.5 / 3.45) / 2), "min": pytest.approx(1 / 4.45),
                           "n": 2}
    assert set(got) == {"lpips", "psnr"}                                    # no share of a gap to an upper bound
    # no training-map level: no share; no headline read: nothing
    assert mrf.recovery_shares(data, [6], None) == {"lpips": None, "psnr": None}
    assert mrf.recovery_shares({6: {"reads": {0: data[6]["reads"][0]}}}, [6], (22.45, 0.15)) == {
        "lpips": None, "psnr": None}
    assert mrf.share_cell(got["lpips"]) == "58" and mrf.share_cell(None) == "--" and mrf.share_cell(0.257) == "26"


def test_the_slim_results_table_has_one_row_per_backbone_and_the_comparison_on_four_arenas():
    four = (6, 7, 8, 16)
    rows = [("SD 1.4 U-Net", slim_block(13, ALL_13, 22.30, 23.60, 0.303, 0.210, 27.32, [150],
                                        shares=(0.72, 0.48, 0.94)), (25.20, 0.158), True),
            ("PixArt-$\\alpha$", slim_block(13, ALL_13, 22.38, 23.72, 0.285, 0.205, 27.32, [150],
                                            shares=(0.76, 0.48, 1.01)), (25.18, 0.159), True),
            ("SD 3.5 Medium", slim_block(8, ALL_13, 21.97, 23.53, 0.260, 0.167, 31.38, [250],
                                         shares=(0.77, 0.50, 0.94)), (25.38, 0.126), True),
            ("U-Net LoRA", slim_block(4, four, 21.71, 22.88, 0.292, 0.209, 26.69, [100, 250],
                                      shares=(0.71, 0.41, 1.23)), None, False),
            ("U-Net full fine-tune", slim_block(4, four, 21.71, 23.11, 0.292, 0.182, 26.69, [4000], coarse=True,
                                                shares=(0.87, 0.46, 1.44)), None, False)]
    tex = mrf.slim_table(rows)
    body = [ln for ln in tex.splitlines() if " & " in ln and not ln.startswith("%")][2:]      # after the two headers
    assert [ln.split(" & ")[0] for ln in body] == ["SD 1.4 U-Net", "PixArt-$\\alpha$", "SD 3.5 Medium (8 of 13)",
                                                   "U-Net LoRA (6, 7, 8, 16)", "U-Net full fine-tune (6, 7, 8, 16)"]
    cells = [ln.rstrip(" \\").split(" & ")[1:] for ln in body]
    # in domain, zero-shot, the headline, then the recovered shares in whole percent (LPIPS, PSNR)
    assert cells[0] == ["25.20", "0.158", "22.30", "0.303", "23.60", "0.210", "72", "48"]
    assert cells[2][-2:] == ["77", "50"]
    # the comparison rows leave the in-domain columns blank: the source model's read, not theirs after adaptation
    assert cells[3] == ["", "", "21.71", "0.292", "22.88", "0.209", "71", "41"]
    assert cells[4] == ["", "", "21.71", "0.292", "23.11", "0.182", "87", "46"]
    assert all(len(c) == 8 for c in cells)
    # no upper bound, gap or budget columns, no group rows, nothing waiting
    assert "Upper" not in tex and "Budget" not in tex and "27.32" not in tex and "Gap" not in tex
    assert not [ln for ln in tex.splitlines() if ln.startswith(("Hard", "Medium", "Easy", "All"))]
    assert "\\tbd" not in tex
    assert tex.count("\\midrule") == 2 and "\\begin{tabular}{lrrrrrrrr}" in tex
    assert "\\multicolumn{2}{c}{Recovered (\\%)}" in tex and "\\cmidrule(lr){8-9}" in tex
    # a block with no map yet waits in every cell after the in-domain ones
    waiting = mrf.slim_table([("SD 3.5 Medium", slim_block(0, ALL_13, None, None, None, None, None, None,
                                                           final=False), (25.38, 0.126), True)])
    assert waiting.splitlines()[-3] == "SD 3.5 Medium & 25.38 & 0.126 & " + " & ".join(["\\tbd{}"] * 6) + " \\\\"
    # the caption names the step, the arena count, the windows and the comparator arenas from the rows themselves
    caption = mrf.slim_caption(rows, 512)
    assert caption.startswith("\\caption{") and caption.rstrip().endswith("}")
    for phrase in ("after 4k adapter updates", "scene crop (rows 0 to 207)", "512 validation windows",
                   "medians over the 13 unseen maps", "four comparator maps 6, 7, 8 and 16", "(PSNR)"):
        assert phrase in caption, phrase
    for phrase in ("(Gap)", "upper bound"):
        assert phrase not in caption, phrase
    for phrase in ():
        assert phrase in caption, phrase


def thirteen_arena_row():
    """The merged row's inputs at the paper's density: 13 arenas on the 8k grid, three LoRA backbones on 8 shared
    arenas (SD 3.5 without the 50 to 150 reads, the U-Net on to 8k), each with its training maps' level."""
    arenas = list(ALL_13)
    zero = {a: {"psnr": 20.5 + 0.35 * i, "lpips": 0.40 - 0.013 * i} for i, a in enumerate(arenas)}
    adapted = {a: {"psnr": z["psnr"] + 1.3, "lpips": z["lpips"] - 0.09} for a, z in zero.items()}
    traj = {a: {"steps": list(mrf.GRID), "psnr": [z["psnr"] + 1.4 * j / 9 for j in range(10)],
                "lpips": [z["lpips"] - 0.1 * j / 9 for j in range(10)]} for a, z in zero.items()}
    groups = mrf.group_arenas({a: -z["psnr"] for a, z in zero.items()})
    shade, key_colours = mrf.group_colours(groups)
    unet = {st: (22.2 + 1.1 * j / 9, 0.29 - 0.09 * j / 9) for j, st in enumerate(mrf.GRID)}
    pix = {st: (22.35 + 1.05 * j / 8, 0.28 - 0.08 * j / 8) for j, st in enumerate(mrf.GRID[:-1])}
    sd = {0: (21.97, 0.26), 250: (23.13, 0.19), 500: (23.18, 0.184), 1000: (23.23, 0.178), 2000: (23.39, 0.172),
          4000: (23.53, 0.167)}
    curves = [("unet_lora", "U-Net", unet, 8), ("pixart_lora", "PixArt-α", pix, 8), ("sd35_lora", "SD 3.5", sd, 8)]
    levels = {"unet_lora": (25.20, 0.158), "pixart_lora": (25.19, 0.159), "sd35_lora": (25.38, 0.126)}
    return (arenas, zero, adapted, {"psnr": 25.2, "lpips": 0.158}, traj, lambda a: shade[a], (7, 16),
            key_colours), (curves, levels)


def capture(monkeypatch):
    """Keep every figure `fs.save` writes, by stem; `kept[stem]` is (figure, renderer) after a fresh draw at the
    figure's own dpi (the save's 600 dpi PNG pass leaves pixel-placed labels at 600 dpi coordinates)."""
    figs = {}
    save = mrf.fs.save

    def keep(fig, out_dir, stem):
        figs[stem] = fig
        return save(fig, out_dir, stem)
    monkeypatch.setattr(mrf.fs, "save", keep)

    class Drawn(dict):
        def __missing__(self, stem):
            fig = figs[stem]
            fig.canvas.draw()
            return fig, fig.canvas.get_renderer()
    return Drawn()


def label_boxes(ax, r):
    """The window boxes of an axes' visible tick labels and axis labels."""
    ticks = [t for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_visible() and t.get_text()]
    labels = [t for t in (ax.xaxis.label, ax.yaxis.label) if t.get_visible() and t.get_text()]
    return [t.get_window_extent(r) for t in ticks + labels]


def test_the_row_with_the_backbones_as_e_and_f_reads_at_print_size(tmp_path, monkeypatch):
    # six panels in one 5.5 in row do not fit (the 13 arena numbers overlap), so (e, f) sit on a line of their own
    # under the Figure 3 row, which keeps its panels where the row puts them
    kept = capture(monkeypatch)
    row, backbones = thirteen_arena_row()
    mrf.fig_row(*row[:7], str(tmp_path), layout="row_backbones", key_colours=row[7], backbones=backbones)
    mrf.fig_row(*row[:7], str(tmp_path), layout="row", key_colours=row[7])
    fig, r = kept["raw_row_backbones"]
    assert (fig.get_figwidth(), fig.get_figheight()) == mrf.SIZES["raw_row_backbones"]
    pa, la, pc, lc, pe, lf = fig.axes[:6]
    alone, r_alone = kept["raw_row"]
    for mine, theirs in zip((pa, la, pc, lc), alone.axes[:4]):
        a, b = mine.get_window_extent(r), theirs.get_window_extent(r_alone)
        assert (a.width, a.height, a.x0) == (pytest.approx(b.width, abs=3), pytest.approx(b.height, abs=3),
                                             pytest.approx(b.x0, abs=3))          # within 3 px (0.03 in)
    assert pe.get_window_extent(r).y1 < pa.get_window_extent(r).y0                 # (e, f) below the row
    # the 13 arena numbers of (a) and (b) each stand clear of their neighbours
    for ax in (pa, la):
        boxes = [t.get_window_extent(r) for t in ax.get_xticklabels() if t.get_text()]
        assert len(boxes) == 13 and not [(i, j) for i, a in enumerate(boxes) for j, b in enumerate(boxes)
                                         if i < j and a.overlaps(b)]
    # no panel's labels run into another panel's labels or plotting area
    axes = [pa, la, pc, lc, pe, lf]
    for i, a in enumerate(axes):
        for j, b in enumerate(axes):
            if i != j:
                hits = [box for box in label_boxes(a, r) if box.overlaps(b.get_window_extent(r))
                        or any(box.overlaps(o) for o in label_boxes(b, r))]
                assert not hits, (i, j)
    # (e, f) carry their own y axes, so medians 0.1 to 0.3 dB apart stay apart (not (a)'s 20 to 26 dB range)
    assert pe.get_ylim() != pa.get_ylim()
    assert pe.get_ylim()[1] - pe.get_ylim()[0] < pa.get_ylim()[1] - pa.get_ylim()[0]
    # both keys fit the page width, and the backbones are named in the key, not at the curves' ends
    page, keys = fig.bbox, mrf.fs.figure_legends(fig)
    texts = [t.get_text() for lg in keys for t in lg.get_texts()]
    assert all(page.x0 <= lg.get_window_extent(r).x0 and lg.get_window_extent(r).x1 <= page.x1 for lg in keys)
    assert {"U-Net", "PixArt-α", "SD 3.5", "hard", "medium", "easy", "after 4k updates"} <= set(texts)
    assert not [t for t in pe.texts + lf.texts if t.get_text() in ("U-Net", "PixArt-α", "SD 3.5")]


def test_the_separate_body_figure_keys_its_marks_instead_of_labelling_the_curves(tmp_path, monkeypatch):
    kept = capture(monkeypatch)
    _, (curves, levels) = thirteen_arena_row()
    mrf.fig_backbones(curves, levels, str(tmp_path), 4000, stem="raw_backbones_body", keyed=True)
    fig, r = kept["raw_backbones_body"]
    assert (fig.get_figwidth(), fig.get_figheight()) == mrf.SIZES["raw_backbones_body"] == (5.5, 1.6)
    texts = [t.get_text() for lg in fig.legends for t in lg.get_texts()]
    assert texts[:3] == ["U-Net", "PixArt-α", "SD 3.5"]
    assert {"after 4k updates", "training maps (in distribution)"} <= set(texts)
    px, lx = fig.axes[:2]
    assert not [t.get_text() for t in px.texts + lx.texts if t.get_gid() != "decor"]   # no end labels, no counts
    assert all(fig.bbox.x0 <= lg.get_window_extent(r).x0 and lg.get_window_extent(r).x1 <= fig.bbox.x1
               for lg in fig.legends)


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
    per = {a: {"zero_shot": z[a], "psnr_4k": z[a] + 2.0, "psnr_8k": z[a] + 3.0,
               "lpips_zero_shot": 0.30, "lpips_4k": 0.22, "lpips_8k": 0.20,
               "lpips_share_4k": 0.5, "psnr_share_4k": 2.0 / (25.0 - z[a]),
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
    assert "arena" not in tex.lower() and "ceiling" not in tex.lower() and "upper bound" not in tex
    assert "half the rise" in tex and "Unseen maps (by zero-shot LPIPS rise)" in tex
    # the shares print as whole percents: the hard group regains 2 of 15 to 12 dB (13 to 17 percent, median 15)
    hard = next(ln for ln in tex.splitlines() if ln.startswith("Hard"))
    assert hard.rstrip(" \\\\").split(" & ")[-3:-1] == ["15", "50"]   # PSNR, then LPIPS
    # until every grid step has a raw read, the budget cells wait
    assert "${>}$8k" not in mrf.groups_table(stats, budgets_final=False)
    # by the zero-shot LPIPS rise over the training maps, the largest rise is the hardest
    rise = {a: 0.20 - 0.01 * i for i, a in enumerate(ALL_13)}            # map 1 has the largest rise
    assert mrf.group_arenas(rise, higher_is_harder=True)[0] == ("hard", [1, 6, 7, 8])
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
        found = mrf.fs.figure_legends(fig) + [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
        legends[stem] = [t.get_text() for lg in found for t in lg.get_texts()]
        return save(fig, out_dir, stem)
    monkeypatch.setattr(mrf.fs, "save", record_legends)
    out, side, tables = tmp_path / "figs", tmp_path / "raw_summary.json", tmp_path / "tables"
    assert mrf.main(["--fresh-root", str(fresh), "--distances", str(dist), "--training-distances", str(tdist),
                     "--out-dir", str(out), "--summary", str(side), "--tables-dir", str(tables),
                     "--family-step", str(tmp_path / "family_step.json"),
                     "--adapt-glob", str(tmp_path / "adapt" / "*_g8k"), "--adapt-root", str(tmp_path / "adapt"),
                     "--directional-roots", str(tmp_path / "directional"), *no_full_grid(tmp_path)]) == 0
    for stem, size in mrf.SIZES.items():
        w, h = mediabox(os.path.join(out, f"{stem}.pdf"))
        assert (w, h) == (pytest.approx(size[0], abs=0.01), pytest.approx(size[1], abs=0.01)), stem
    s = json.load(open(side))
    # the step without persistence: every arena below every training map in PSNR and above in LPIPS, all backbones
    assert all(s["step"][b]["psnr"]["holds"] and s["step"][b]["lpips"]["holds"] for b in mrf.BACKBONES)
    # the budget recovers half of the zero-shot LPIPS rise over the training maps' level (0.150)
    home_psnr = sum(22.0 + 0.3 * (m - 2) for m in TRAINING) / 4
    ab = s["adaptation"]
    assert ab["rule"] == "half_lpips_rise" and ab["in_distribution"] == {"psnr": pytest.approx(home_psnr),
                                                                          "lpips": pytest.approx(0.15)}
    assert ab["per_arena"]["6"]["psnr_deficit"] == pytest.approx(home_psnr - 18.0)
    assert ab["per_arena"]["6"]["lpips_rise"] == pytest.approx(0.15)
    assert ab["per_arena"]["6"]["threshold"] == pytest.approx(0.225)        # 4k (0.22) crosses; map 9: 0.205, 0.19
    assert ab["per_arena"]["6"]["budget"] == 4000 and ab["per_arena"]["9"]["budget"] == 4000
    assert (ab["past_half_by_4k"], ab["past_half_by_8k"]) == (2, 2) and ab["grid_complete"] is False
    assert ab["median_budget"] == {"value": 4000, "censored": False, "label": "4000"}
    # maps group and colour by the zero-shot LPIPS rise over the training maps (map 6: 0.15, map 9: 0.11)
    assert s["group_key"] == "zero-shot LPIPS rise over the training maps"
    assert s["groups"]["all"]["n"] == 2 and s["groups"]["all"]["lpips_share_4k"] == pytest.approx(
        (0.08 / 0.15 + 0.07 / 0.11) / 2)
    assert "upper_bound" not in s["definitions"] and "ceiling_psnr" not in s["zero_shot"]["unet200k_ema"]["6"]
    # before and after: in-distribution minus zero-shot, and adapted (8k) minus zero-shot, medians over maps
    ba = s["before_after"]
    assert "median_upper_bound" not in ba["psnr"]
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
    assert "\\tbd" in tex and "upper bound" not in tex and "Recovered" in tex
    # Table 2's per-backbone blocks at the 4k headline; the full fine-tune column is gone (its rows carry it)
    assert "& fine-tune \\\\" not in tex and "\\tbd{} (6" not in tex
    lines = {line.split(" & ")[0]: line for line in tex.splitlines() if " & " in line}
    assert "U-Net LoRA (1 of 4)" not in lines                          # the adapter row left Table 2 (Sep 29)
    # the full fine-tune's arena 6 has its 4k read and the U-Net LoRA's step 0 (the same checkpoint): 1 of 4
    ft = lines["U-Net full fine-tune (1 of 2)"]      # every map it has, of the fixture's two
    assert " & 18.00 & 23.00 & " in ft and "zero-shot of maps 6 from the U-Net LoRA's step 0" in tex
    assert ft.endswith(" & ${\\le}$4k \\\\")                          # scored at 0 and 4k only
    # the caption's comparison: the adapter's own reads on the same maps, in the table's comment line
    assert "against the adapter's 19.00 dB / 0.220 and 53 percent (LPIPS) / 22 percent (PSNR) recovered on the " \
        "same maps 6" in tex
    assert s["blocks"]["unet_full"]["matched_adapter"]["arenas"] == [6]
    px = lines["PixArt-$\\alpha$ LoRA (all 2)"]
    assert " & 20.70 & " in px and px.split(" & ")[3] == "--"                         # 4k median; its grid stops at 4k
    sd = lines["SD 3.5 LoRA (1 of 2)"]
    assert " & 19.50 & " in sd and "\\tbd{}" in sd                                    # budget waits on its grid
    # the arena lists live in the comment lines under the table
    assert "% SD 3.5 LoRA (1 of 2): maps 6; decoder sha256:sd35tuned" in tex
    assert "% U-Net LoRA (1 of 4)" not in tex                         # its row and comment line are gone
    blocks = s["blocks"]
    assert blocks["pixart_lora"]["quantity"] == "raw" and blocks["pixart_lora"]["decoder"] == "tuned"
    # neither PixArt map recovers half of its LPIPS rise on its 4k grid (0.26 against 0.225, 0.22 against 0.205):
    # censored at the grid's last read
    assert blocks["pixart_lora"]["budget_final"] is True and blocks["pixart_lora"]["budget_middle"] == [None]
    assert blocks["pixart_lora"]["censored"] == 2 and " & ${>}$4k; 2 of 2 censored \\\\" in px
    assert blocks["pixart_lora"]["gpu_hours_headline_by_card"] == {
        "A6000": {"median": pytest.approx(1.5), "n": 2, "arenas": [6, 9]}}
    # SD 3.5's own fine-tuned decoder has no training-map read: no in-distribution level, so no budget and no
    # shares, never the stock decoder's level (another decoder's reads)
    sd35 = blocks["sd35_lora"]
    assert (sd35["decoder"], sd35["decoder_identity"]) == ("tuned", "sha256:sd35tuned")
    assert sd35["level"] is None and sd35["budget_final"] is False and "tuned" in sd35["level_note"]
    assert sd35["shares"] == {"lpips": None, "psnr": None}
    # Superman's decoded-only arena 9 stays out of the raw medians until it is rescored raw, and the table says so
    assert (sd35["arenas"], sd35["decoded_only"], sd35["quantity"]) == ([6], [9], "raw")
    assert "$^\\ddagger$" not in sd and "maps 9 read against the decoded ground truth only" in tex
    assert blocks["pixart_lora"]["level"] == [pytest.approx(22.45 + 0.2), pytest.approx(0.15)]
    assert not blocks["pixart_lora"].get("level_note")
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
    # the slim results table: the U-Net over every arena (6 and 9 here) from the row's own reads, then the blocks
    slim = open(tables / "results_slim.tex").read()
    lines = {line.split(" & ")[0]: line.rstrip(" \\").split(" & ")[1:] for line in slim.splitlines()
             if " & " in line and not line.startswith("%")}
    # in domain, zero-shot and 4k medians over maps 6 and 9, then the recovered shares: map 6 undoes 0.08 of its
    # 0.15 LPIPS rise and regains 1 of its 4.45 dB deficit; map 9: 0.07 of 0.11 and 4.5 of 3.45; medians 58 and 76
    assert lines["SD 1.4 U-Net"] == ["22.45", "0.150", "18.50", "0.280", "21.25", "0.205", "58", "76"]
    assert lines["PixArt-$\\alpha$"][4] == "20.70" and lines["SD 3.5 Medium (1 of 2)"][4] == "19.50"
    # the comparison rows leave the in-domain columns blank; their shares are against the U-Net's level: map 6's
    # full fine-tune (4k 23.0 dB, LPIPS 0.18, from the borrowed 18.0 and 0.30) undoes 0.12 of 0.15 and regains 5 of
    # 4.45 dB
    assert lines["U-Net full fine-tune (1 of 2)"] == ["", "", "18.00", "0.300", "23.00", "0.180", "80", "112"]
    assert lines["U-Net LoRA (1 of 4)"][:3] == ["", "", "18.00"]
    assert s["blocks"]["unet_full"]["shares"]["lpips"] == {"median": pytest.approx(0.8), "min": pytest.approx(0.8),
                                                           "n": 1}
    caption = open(tables / "results_slim_caption.tex").read()
    assert "after 4k adapter updates" in caption and "16 validation windows" in caption      # 4 maps x 4 windows
    assert "medians over the 2 unseen maps" in caption and "four comparator maps 6, 7, 8 and 16" in caption
    # Table 1's copy: the step check's reads plus the directional column, pooled by windows (arena 9 has 64)
    full = open(tables / "results_full.tex").read()
    t1 = [line.rstrip(" \\").split(" & ") for line in full.splitlines() if " & " in line and not line.startswith("%")]
    assert t1[0] == ["Model", "Maps", "PSNR (dB)", "LPIPS", "Directional"]              # no upper bound column
    assert t1[1] == ["SD 1.4 U-Net", "training", "22.45", "0.150", "0.850"]
    assert t1[2] == ["", "unseen", "18.50", "0.280", "0.733"]
    assert t1[3][:2] == ["PixArt-$\\alpha$", "training"] and t1[3][-1] == "--"        # no directional read
    assert t1[5][:2] == ["SD 3.5 Medium", "training"] and t1[6][-1] == "--" and len(t1) == 7
    assert "% SD 1.4 U-Net unseen: directional over 2 maps (6, 9), 192 windows, ground-truth frames 0.900" in full
    assert "% PixArt-$\\alpha$ unseen: no directional read of pixart200k_ema under" in full
    assert s["directional"]["unet200k_ema"]["unseen"] == {"correct": pytest.approx((0.7 * 128 + 0.8 * 64) / 192),
                                                          "reference": pytest.approx(0.9), "maps": [6, 9],
                                                          "windows": 192}
    assert s["directional"]["pixart200k_ema"] == {"training": None, "unseen": None}
    assert any(p["path"].endswith("directional_map09_arenas13_ema.json") for p in s["inputs"])
    # Tables 1 and 2 merged: one row block per backbone; the training row carries the in-distribution reads and
    # Table 1's training directional in the 0 columns, the unseen row Table 1's unseen directional at 0 and the
    # guards' median at 8k (maps 6 and 9: 0.85 and 0.75), the group row the guards' medians at both
    merged = open(tables / "results_merged.tex").read()
    body = [[c.strip() for c in line.rstrip("\\ ").split("&")] for line in merged.splitlines()
            if " & " in line and not line.startswith("%")][2:]                   # after the two header rows
    assert [r[0] for r in body] == ["SD 1.4 U-Net, training maps", "unseen maps (all 2)", "unseen medium (6, 9)",
                                    "full fine-tune (1 of 2)", "PixArt-$\\alpha$, training maps",
                                    "unseen maps (all 2)", "SD 3.5 Medium, training maps", "unseen maps (1 of 2)"]
    assert body[0][1:] == ["22.45", "", "", "0.150", "", "", "0.850", "", "", "", ""]
    assert body[1][1:] == ["18.50", "21.25", "23.25", "0.280", "0.205", "0.190", "0.733", "0.800", "76", "58",
                           "\\tbd{}"]                       # shares PSNR then LPIPS; the budget waits on the grid
    assert body[2][7:9] == ["0.750", "0.800"]
    assert body[3][1:] == ["18.00", "23.00", "--", "0.300", "0.180", "--", "--", "--", "112", "80", "${\\le}$4k"]
    assert "& 0 & 8k & PSNR & LPIPS & half the rise" in merged                      # the score columns' order
    assert body[4][1:8] == ["22.65", "", "", "0.150", "", "", "--"]                 # no directional read: "--"
    assert body[5][1:4] == ["18.70", "20.70", "--"] and body[5][-1] == "${>}$4k; 2 of 2 censored"   # maps 6, 9
    assert body[6][1] == "21.95" and body[7][-1] == "\\tbd{}"
    assert "% SD 3.5 LoRA (1 of 2): maps 6; decoder sha256:sd35tuned" in merged            # the comment lines
    assert "upper bound" not in merged and "arena" not in merged.lower()
    cap = open(tables / "results_merged_caption.tex").read()
    for phrase in ("pooled over 16 windows", "medians over the 2", "terciles of the zero-shot LPIPS rise",
                   "ground truth 0.900 / 0.900", "per map then median", "half the rise"):
        assert phrase in cap, phrase
    assert "8k: over the maps scored there" not in cap                 # every fixture map has its 8k read or none
    # no 200k SD 3.5 files: the provisional 170k read through the stock decoder, recorded as the fallback
    assert s["sd35_row"] == {"row": "sd35_170000", "decoder": "stock", "fallback": True}
    # only the merged row's panels are lettered (a to d, in both layouts); the single supplement panels carry none
    # the row, its grid layout, both backbone panels and the full fine-tune panel, then the two body candidates: the
    # row with the backbones as (e, f) and the separate backbone figure
    # row, row_150, grid; the three backbone panels; the two body candidates; the per-backbone row and its tall form
    assert letters == list("abcd") * 3 + list("ab") * 3 + list("abcdef") + list("ab") + list("abcd") * 2
    # the per-backbone row: every LoRA backbone's per-arena reads with its own upper bound, keyed by backbone
    v2 = s["row_v2"]
    assert v2["stems"] == ["raw_row_v2", "raw_row_v2_tall"]
    assert v2["backbones"] == ["unet_lora", "pixart_lora", "sd35_lora"]
    assert v2["headline"]["unet_lora"]["6"] == [pytest.approx(19.0), pytest.approx(0.22)]      # adapt4000 row
    assert v2["headline"]["pixart_lora"]["6"] == [pytest.approx(20.2), pytest.approx(0.26)]    # its LoRA's 4k read
    assert v2["headline"]["sd35_lora"] == {"6": [pytest.approx(19.5), pytest.approx(0.25)]}    # arena 6 only
    assert v2["levels"]["pixart_lora"] == [pytest.approx(22.65), pytest.approx(0.15)]         # its training maps
    for stem in v2["stems"]:
        assert {"U-Net", "PixArt-α", "SD 3.5", "zero-shot", "after 4k updates"} <= set(legends[stem]), stem
        # no upper bound ticks and no "arena" anywhere in the keys (the figure says "unseen map")
        assert not any("upper bound" in t or "arena" in t for t in legends[stem]), legends[stem]
        assert any(t.startswith("training maps (in distribution)") for t in legends[stem])
    body = s["body_candidates"]
    assert body["arenas"] == [6] and body["drawn"] == ["unet_lora", "pixart_lora", "sd35_lora"]   # the LoRAs only
    assert body["stems"] == ["raw_row_backbones", "raw_backbones_body"]
    for stem in body["stems"]:
        assert {"U-Net", "PixArt-α", "SD 3.5", "after 4k updates"} <= set(legends[stem]), stem
        assert "full fine-tune" not in " ".join(legends[stem])
    # the full fine-tune panel: the U-Net LoRA against the full fine-tune on the comparator arenas both have (6)
    ft = s["fullft_panel"]
    assert (ft["arenas"], ft["drawn"]) == ([6], ["unet_lora", "unet_full"])
    assert sorted(ft["medians"]["unet_full"]) == ["0", "4000"]                        # two points, 0 borrowed
    assert ft["medians"]["unet_full"]["4000"][0] == pytest.approx(23.0)
    assert ft["medians"]["unet_lora"]["0"][0] == pytest.approx(18.0) == ft["medians"]["unet_full"]["0"][0]
    assert len(ft["medians"]["unet_lora"]) > 2                                       # the LoRA's whole grid
    assert ft["labels"] == ["U-Net LoRA (1)", "full fine-tune (0, 4k)"]
    # the appendix's per-map table: the deficit, its threshold, the shares, the reads, the guards
    per = s["per_arena_table"]
    assert per["6"]["psnr_deficit"] == pytest.approx(home_psnr - 18.0) and per["6"]["lpips_rise"] == pytest.approx(0.15)
    assert per["6"]["threshold"] == pytest.approx(0.225)
    assert per["6"]["lpips_share_4k"] == pytest.approx(0.08 / 0.15) and per["9"]["psnr_share_4k"] == pytest.approx(
        4.5 / (home_psnr - 19.0))
    assert per["6"]["forgetting_dec"] == pytest.approx(-1.0) and per["9"]["forgetting_dec"] == pytest.approx(-0.5)
    assert (per["6"]["directional_0"], per["6"]["directional_8k"]) == (pytest.approx(0.80), pytest.approx(0.85))
    rows = open(tables / "adapt_perarena.tex").read()
    assert "\\tbd" in rows and "$-$1.00" in rows and "0.85" in rows and "decoded" in rows
    assert "LPIPS & & " in rows and "& rise & Budget" in rows and "upper" not in rows.lower() and "gap" not in rows.lower()
    # legends: the marks and the colour key on the row (both layouts), the backbones and the band on the step panels
    shown = [g for g in ("hard", "medium", "easy") if s["groups"][g]["n"]]      # two arenas: medium only
    assert shown == ["medium"]
    for stem in ("raw_row", "raw_row_grid"):
        for text in ["zero-shot", "after 4k updates", "training maps (in distribution)",
                     "zero-shot LPIPS rise over the training maps:"] + shown:
            assert text in legends[stem], (stem, text)
        assert not any("upper bound" in t for t in legends[stem]), stem
        assert not {"hard", "easy"} & set(legends[stem])               # an empty group gets no swatch
    for stem in ("raw_fig3a_psnr", "raw_fig3a_lpips"):
        assert {"U-Net", "PixArt-α", "SD 3.5", "train vs train $d$"} <= set(legends[stem]), stem
    # the numbers the text waits on: counts and medians with arena-bootstrap intervals, provisional until the grid
    t = s["text_numbers"]
    assert t["provisional"] is True and t["pending"]["forgetting"] and t["pending"]["gain_step"]
    assert t["past_threshold_by_8k"]["count"] == 2 and len(t["past_threshold_by_8k"]["interval"]) == 2
    assert t["median_lpips_share_4k"]["value"] == pytest.approx((0.08 / 0.15 + 0.07 / 0.11) / 2)
    assert t["median_psnr_share_4k"]["n"] == 2 and t["median_budget"]["value"] == 4000
    assert t["past_threshold_by_4k"]["count"] == 2
    # the share of each arena's 0-to-8k raw gain in place by 4k, averaged: (1/4.5 + 4.5/5) / 2
    assert t["gain_share_by_step"]["4000"] == pytest.approx((1.0 / 4.5 + 4.5 / 5.0) / 2)
    assert set(t["spearman_S0"]) == {"lpips_share_4k", "gain_8k", "budget"}   # two maps: no rho yet
    assert t["spearman_S0"]["lpips_share_4k"]["n"] == 2
