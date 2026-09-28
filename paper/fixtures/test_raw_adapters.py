"""`paper/raw_adapters.py`: any backbone's adaptation runs read into per-arena, per-step scene reads, in raw terms
where the data allow and labelled where they do not.

The fixture writes run directories in the layout the adaptation runs leave (`<run>/scores.jsonl`, `<run>/log.jsonl`,
optionally `<run>/scores/<step dir>/heldout/per_window.csv`) and a fresh-rescore zero-shot read whose per-window file
gives the references a row-only read needs (the upper bound, raw persistence's LPIPS, decoded copy-last's PSNR).

    python -m pytest paper/fixtures/test_raw_adapters.py -q
"""
import csv
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
sys.path.insert(0, PAPER)
sys.path.insert(0, os.path.dirname(PAPER))

import raw_adapters as ra  # noqa: E402

DECODERS = {"stock": {"path": "", "identity": "hub:stock"},
            "tuned": {"name": "tuned", "path": "/x/sd1_mse_lpips", "identity": "sha256:abc"}}


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_run(root, name, rows, log=None):
    d = os.path.join(str(root), name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "scores.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    if log:
        with open(os.path.join(d, "log.jsonl"), "w") as f:
            f.write("\n".join(json.dumps(e) for e in log) + "\n")
    return d


def row(step, decoders=("stock",), grid=(0, 250, 4000), weights="live", pw=None, **fields):
    return {"step": step, "weights": weights, "grid": list(grid), "heldout_decoders": list(decoders),
            "decoders": {d: DECODERS[d] for d in decoders}, "eval_fingerprint": "fp", "scored_at": f"t{step}",
            "heldout_per_window": pw, **fields}


def windows(arena, **cols):
    """Four windows around each column's mean (offsets sum to zero); the first a duplicate the reads leave out."""
    out = []
    for i, off in enumerate((0.0, -0.1, 0.0, 0.1)):
        r = {"episode": 300 + i, "map": arena, "dup_raw": int(i == 0), "dup_latent": 0}
        r.update({c: v + off * (10 if "psnr" in c else 0.1) + (50.0 if i == 0 else 0.0) for c, v in cols.items()})
        out.append(r)
    return out


def test_run_names_parse_into_backbone_kind_and_arena():
    m = ra.run_meta("pixart200k_arenas13_map07_r16_k8_s0")
    assert (m["backbone"], m["kind"], m["arena"], m["k"], m["seed"], m["variant"]) == ("pixart", "lora", 7, 8, 0, "")
    assert ra.run_meta("sd35_200k_arenas13_map11_r16_k8_s0")["backbone"] == "sd35"
    full = ra.run_meta("unet200k_arenas13_map16_r0_k8_s0")
    assert (full["backbone"], full["kind"], full["arena"]) == ("unet", "full", 16)
    assert ra.run_meta("unet200k_arenas13_map07_r16_k8_s0_g8k")["variant"] == "g8k"
    assert ra.run_meta("fitcheck_map8") is None


def test_the_decoder_is_the_fine_tuned_one_when_the_rows_carry_it():
    assert ra.pick_decoder([row(0, decoders=("stock", "tuned"))]) == ("tuned", "sha256:abc")
    assert ra.pick_decoder([row(0, decoders=("stock",))]) == ("stock", "hub:stock")


def test_each_read_takes_the_best_quantity_the_run_carries(tmp_path):
    ref = {"upper": 27.0, "persist_lpips": 0.20, "copy_psnr_dec": 21.0}
    # (1) per-window raw columns beside the run (the recorded path is a server path)
    d = str(tmp_path / "a")
    write_csv(os.path.join(d, "scores", "step0004000_live_aa", "heldout", "per_window.csv"),
              windows(7, scene_psnr_raw_tuned=23.0, scene_lpips_raw_tuned=0.21, scene_vae_psnr_tuned=27.2,
                      scene_psnr_dec_tuned=25.0))
    r = row(4000, decoders=("stock", "tuned"), pw="/home/x/results/adapt/a/scores/step0004000_live_aa/heldout/"
                                                  "per_window.csv")
    got = ra.step_read(r, d, "tuned", ref)
    assert got["quantity"] == "raw" and got["source"] == "per-window"
    assert (got["psnr"], got["lpips"], got["upper"]) == (pytest.approx(23.0), pytest.approx(0.21), pytest.approx(27.2))
    # (2) no per-window file, but the row carries the raw margin and gap: the reference supplies the rest
    got = ra.step_read(row(4000, heldout_B_stock=-0.03, heldout_C_stock=3.5), str(tmp_path / "b"), "stock", ref)
    assert got["quantity"] == "raw" and got["source"] == "row"
    assert (got["psnr"], got["lpips"], got["upper"]) == (pytest.approx(23.5), pytest.approx(0.17), pytest.approx(27.0))
    # (3) only the decoded reference: the prediction against the decoded ground truth, no LPIPS from a row
    got = ra.step_read(row(4000, heldout_A_stock=1.5), str(tmp_path / "c"), "stock", ref)
    assert got["quantity"] == "dec" and got["psnr"] == pytest.approx(22.5) and got["lpips"] is None
    # nothing usable
    assert ra.step_read(row(4000), str(tmp_path / "d"), "stock", ref) is None


def test_gpu_hours_to_a_step_come_from_the_training_log():
    log = [{"event": "start", "world": 1, "time": 1000.0}, {"event": "checkpoint", "step": 0, "time": 1003.0},
           {"event": "checkpoint", "step": 4000, "time": 1000.0 + 3960.0}]
    assert ra.gpu_hours(log) == {"0": pytest.approx(3.0 / 3600), "4000": pytest.approx(1.1)}
    log[0]["world"] = 2
    assert ra.gpu_hours(log)["4000"] == pytest.approx(2.2)


def test_runs_group_into_blocks_with_the_8k_grid_preferred_for_the_u_net(tmp_path):
    ref = {7: {"upper": 27.0, "persist_lpips": 0.20, "copy_psnr_dec": 21.0},
           16: {"upper": 26.0, "persist_lpips": 0.25, "copy_psnr_dec": 20.0}}
    root = tmp_path / "adapt"
    log = [{"event": "start", "world": 1, "time": 0.0}, {"event": "checkpoint", "step": 4000, "time": 3600.0}]
    for arena in (7, 16):
        write_run(root, f"pixart200k_arenas13_map{arena:02d}_r16_k8_s0",
                  [row(s, heldout_B_stock=-0.01 * s / 4000, heldout_C_stock=4.0 - s / 4000) for s in (0, 250, 4000)],
                  log=log)
    write_run(root, "unet200k_arenas13_map16_r0_k8_s0", [row(s, heldout_A_stock=0.1 * s / 250) for s in (0, 250)])
    write_run(root, "unet200k_arenas13_map07_r16_k8_s0", [row(0, heldout_A_stock=1.0)])
    write_run(root, "unet200k_arenas13_map07_r16_k8_s0_g8k", [row(0, heldout_A_stock=2.0)])
    write_run(root, "unet200k_arenas13_map07_r16_k8_s0_lr5e4", [row(0, heldout_A_stock=3.0)])   # a recipe test
    write_run(root, "pixart200k_arenas13_map07_r16_k2_s0", [row(0, heldout_A_stock=3.0)])       # a ladder rung
    blocks = ra.load_blocks(str(root), {"pixart": {"stock": ref}, "unet": {"stock": ref}})
    assert sorted(blocks) == ["pixart_lora", "unet_full", "unet_lora"]
    px = blocks["pixart_lora"]
    assert (px["decoder"], px["identity"], sorted(px["arenas"])) == ("stock", "hub:stock", [7, 16])
    r = px["arenas"][7]
    assert r["grid"] == [0, 250, 4000] and r["gpu_hours"]["4000"] == pytest.approx(1.0)
    assert r["reads"][4000]["quantity"] == "raw" and r["reads"][4000]["psnr"] == pytest.approx(27.0 - 3.0)
    assert blocks["unet_lora"]["arenas"][7]["run"].endswith("_g8k")          # the 8k grid over the base run
    assert blocks["unet_full"]["arenas"][16]["reads"][250]["quantity"] == "dec"
