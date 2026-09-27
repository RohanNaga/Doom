"""`tools/compose_rollouts.py`: Figure 2 composed from a synthetic export in the steward's layout.

The fixture writes `<root>/<map>_<control>/<row>_<decoder>/tic_NN.png` (320 x 240, with a HUD band below row 208)
and `tic_NN_scene.png` (rows 0 to 207) for tics 0 to 32, one `manifest.json` per window with per-tic full-frame and
scene PSNR for every row and copy-last, for map 2 and arena 7 under a held left turn and held forward.

    python -m pytest paper/fixtures/test_compose_rollouts.py -q
"""
import json
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "paper"))

pytest.importorskip("matplotlib")
Image = pytest.importorskip("PIL.Image")
import compose_rollouts as cr  # noqa: E402

ROWS = {"train_map02": ("truth_raw", "truth_stock", "truth_tuned", "unet_stock", "unet_tuned"),
        "unseen_arena07": ("truth_raw", "truth_stock", "truth_tuned", "unet_stock", "unet_tuned", "adapter_stock",
                           "adapter_tuned")}


def synthetic_frame(seed, tic):
    """A 240 x 320 frame: a colour field that drifts with the tic over the scene, a grey HUD band below row 208."""
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, 3)
    img = np.zeros((240, 320, 3), np.uint8)
    xs = np.linspace(0, 1, 320)[None, :, None]
    img[:208] = np.clip(base + 50 * np.sin(6 * xs + 0.2 * tic), 0, 255).astype(np.uint8)
    img[208:] = 90
    return img


def write_export(root, controls=("turn_left", "forward"), manifest_at_root=False, drop=None):
    """The steward's export for the two maps; `drop` names one file to leave out."""
    everything = {}
    for gmap, rows in ROWS.items():
        for control in controls:
            wdir = os.path.join(str(root), f"{gmap}_{control}")
            manifest = {"map": gmap, "control": control, "episode": 41 if "arena" in gmap else 6000,
                        "start": 269, "tics": list(range(33)), "rows": {}}
            for k, row in enumerate(rows):
                rdir = os.path.join(wdir, row)
                os.makedirs(rdir, exist_ok=True)
                for tic in range(33):
                    img = synthetic_frame(hash((gmap, control, row)) % 1000, tic)
                    name = f"tic_{tic:02d}"
                    if drop != f"{gmap}_{control}/{row}/{name}.png":
                        Image.fromarray(img).save(os.path.join(rdir, name + ".png"))
                    if drop != f"{gmap}_{control}/{row}/{name}_scene.png":
                        Image.fromarray(img[:208]).save(os.path.join(rdir, name + "_scene.png"))
                manifest["rows"][row] = {"psnr": [30.0 - 0.3 * t - k for t in range(33)],
                                         "scene_psnr": [29.0 - 0.25 * t - k for t in range(33)]}
            for d in ("raw", "stock", "tuned"):
                manifest[f"copylast_{d}"] = {"psnr": [25.0 - 0.4 * t for t in range(33)],
                                             "scene_psnr": [24.0 - 0.4 * t for t in range(33)]}
            everything[os.path.basename(wdir)] = manifest
            if not manifest_at_root:
                with open(os.path.join(wdir, "manifest.json"), "w") as f:
                    json.dump(manifest, f)
    if manifest_at_root:
        with open(os.path.join(str(root), "manifest.json"), "w") as f:
            json.dump({"windows": everything}, f)
    return str(root)


def mediabox(path):
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", open(path, "rb").read())
    return float(m.group(1)) / 72, float(m.group(2)) / 72


def test_the_composer_draws_both_controls_and_both_maps_at_full_width(tmp_path):
    root = write_export(tmp_path / "export")
    paths, notes = cr.compose(root, str(tmp_path / "out"), tics=cr.WIDE_TICS)
    pdf = str(tmp_path / "out" / "fig2_rollouts.pdf")
    assert pdf in paths and os.path.getsize(tmp_path / "out" / "fig2_rollouts.png") > 5000
    assert b"/Type3" not in open(pdf, "rb").read()
    w, h = mediabox(pdf)
    assert w == pytest.approx(5.5, abs=0.01) and 1.2 < h < 2.0          # the standard's 1.75 in, give or take
    side = json.load(open(tmp_path / "out" / "fig2_rollouts.json"))
    assert notes == [] and side["tics"] == [0, 1, 2, 4, 8, 16, 32] and side["decoder"] == "tuned"
    assert [b["control"] for b in side["blocks"]] == ["turn_left", "forward"]
    arena = side["blocks"][0]["windows"]["unseen_arena07"]
    assert arena["episode"] == 41 and arena["start"] == 269
    # the model rows carry their per-tic scene PSNR from the manifest (tuned rows are 4th and 6th in the fixture)
    expected = {"1": 24.8, "2": 24.5, "4": 24.0, "8": 23.0, "16": 21.0, "32": 17.0}
    assert arena["rows"]["unet_tuned"]["scene_psnr"] == expected
    assert arena["rows"]["adapter_tuned"]["label"] == "LoRA 4k"
    assert arena["rows"]["truth_raw"]["scene_psnr"] == {}                  # true rows get no numbers
    # each map has its own true row
    assert set(side["blocks"][0]["windows"]["train_map02"]["rows"]) == {"truth_raw", "unet_tuned"}


def test_a_scene_crop_is_taken_from_the_full_frame_when_the_crop_file_is_missing(tmp_path):
    root = write_export(tmp_path / "export", drop="train_map02_turn_left/unet_tuned/tic_04_scene.png")
    img = cr.frame(os.path.join(root, "train_map02_turn_left", "unet_tuned"), 4)
    assert img.shape[:2] == (208, 320)
    assert np.array_equal(img, synthetic_frame(hash(("train_map02", "turn_left", "unet_tuned")) % 1000, 4)[:208])


def test_a_missing_frame_stops_the_build_with_its_path(tmp_path):
    root = write_export(tmp_path / "export", drop="unseen_arena07_forward/adapter_tuned/tic_16.png")
    os.remove(os.path.join(root, "unseen_arena07_forward", "adapter_tuned", "tic_16_scene.png"))
    with pytest.raises(SystemExit, match="adapter_tuned/tic_16"):
        cr.compose(root, str(tmp_path / "out"))
    assert not os.path.exists(tmp_path / "out" / "fig2_rollouts.pdf")


def test_a_root_manifest_and_the_stacked_layout(tmp_path):
    root = write_export(tmp_path / "export", manifest_at_root=True)
    cr.compose(root, str(tmp_path / "side"), tics=cr.WIDE_TICS)
    cr.compose(root, str(tmp_path / "stack"), tics=cr.WIDE_TICS, stack=True)
    side = json.load(open(tmp_path / "side" / "fig2_rollouts.json"))
    assert side["blocks"][1]["windows"]["train_map02"]["rows"]["unet_tuned"]["scene_psnr"]["1"] == pytest.approx(24.8)
    _, h_side = mediabox(str(tmp_path / "side" / "fig2_rollouts.pdf"))
    _, h_stack = mediabox(str(tmp_path / "stack" / "fig2_rollouts.pdf"))
    assert h_stack > 2 * h_side          # stacked blocks: each frame about twice as wide and tall
    fw_side = json.load(open(tmp_path / "side" / "fig2_rollouts.json"))["frame_in"][0]
    fw_stack = json.load(open(tmp_path / "stack" / "fig2_rollouts.json"))["frame_in"][0]
    assert fw_stack > 1.9 * fw_side


def test_per_tic_series_are_read_in_every_known_shape():
    assert cr.as_tic_series([10.0, 11.0, 12.0]) == {0: 10.0, 1: 11.0, 2: 12.0}           # tics 0..2
    assert cr.as_tic_series([11.0, 12.0]) == {1: 11.0, 2: 12.0}                          # tics 1..2
    assert cr.as_tic_series({"tic_01": 11.0, "2": 12.0}) == {1: 11.0, 2: 12.0}
    m = {"scene_psnr": {"unet_tuned": [9.0, 8.0, 7.0]}}
    assert cr.per_tic(m, "unet_tuned") == {0: 9.0, 1: 8.0, 2: 7.0}
    assert cr.per_tic({"unet_tuned_scene_psnr": {"1": 5.0}}, "unet_tuned") == {1: 5.0}
    assert cr.per_tic({}, "unet_tuned") == {}


def test_the_stewards_manifest_shape_a_root_list_of_windows_with_per_tic_series(tmp_path):
    root = write_export(tmp_path / "export", manifest_at_root=True)
    side = json.load(open(os.path.join(root, "manifest.json")))
    windows = []
    for name, m in side["windows"].items():
        windows.append({"map": m["map"], "control": m["control"], "episode": m["episode"], "start_row": 837,
                        "dir": f"/sata2/export_v2/figure2_rollouts/{name}", "rows": list(m["rows"]),
                        "per_tic": {**m["rows"], **{k: v for k, v in m.items() if k.startswith("copylast_")}}})
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump({"what": "synthetic", "windows": windows}, f)
    cr.compose(root, str(tmp_path / "out"))
    out = json.load(open(tmp_path / "out" / "fig2_rollouts.json"))
    arena = out["blocks"][0]["windows"]["unseen_arena07"]
    assert arena["start_row"] == 837 and arena["episode"] == 41
    assert arena["rows"]["adapter_tuned"]["scene_psnr"]["1"] == pytest.approx(22.8)


def test_the_tic_zero_column_is_the_raw_context_frame_in_every_row(tmp_path):
    root = write_export(tmp_path / "export")
    cr.compose(root, str(tmp_path / "out"), stack=True, map_controls={"train_map02": "forward"})
    side = json.load(open(tmp_path / "out" / "fig2_rollouts.json"))
    assert side["tics"] == [0, 1, 4, 16, 32]
    # map 2 comes from its forward window until the brighter re-pick lands; stacked, it is drawn once
    assert side["blocks"][0]["windows"]["train_map02"]["dir"] == "train_map02_forward"
    assert "train_map02" not in side["blocks"][1]["windows"] and side["drawn_once"] == ["train_map02"]
    assert side["frame_in"][0] == pytest.approx(cr.STACKED_FRAME_IN)
    rows = side["blocks"][0]["windows"]["unseen_arena07"]["rows"]
    assert {r["tic0"] for r in rows.values()} == {"truth_raw"}         # the model rows too


def test_a_per_block_window_override_and_a_second_root_manifest(tmp_path):
    root = write_export(tmp_path / "export", manifest_at_root=True)
    # a brighter map 2 window under another name, its manifest in a second root file
    import shutil
    shutil.copytree(os.path.join(root, "train_map02_turn_left"), os.path.join(root, "train_map02_turn_left_bright"))
    side = json.load(open(os.path.join(root, "manifest.json")))["windows"]["train_map02_turn_left"]
    entry = {**side, "dir": "/sata2/x/train_map02_turn_left_bright", "start_row": 2208}
    with open(os.path.join(root, "manifest_bright.json"), "w") as f:
        json.dump({"windows": [entry]}, f)
    cr.compose(root, str(tmp_path / "out"), stack=True,
               map_controls={"train_map02:turn_left": "turn_left_bright"})
    out = json.load(open(tmp_path / "out" / "fig2_rollouts.json"))
    left, fwd = out["blocks"]
    assert left["windows"]["train_map02"]["dir"] == "train_map02_turn_left_bright"
    assert left["windows"]["train_map02"]["start_row"] == 2208
    assert fwd["windows"]["train_map02"]["dir"] == "train_map02_forward" and out["drawn_once"] == []
