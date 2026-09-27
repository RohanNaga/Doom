"""`tools/compose_teaser.py`: both teaser layouts from a synthetic export in the steward's layout.

The fixture writes `<root>/<map>_ep<E>_s<S>/{truth_raw,unet_tuned[,adapter_tuned]}/tic_NNN.png` (240 x 320 with a
HUD band) for tics 0 to 40 for one unseen arena and one training map, and a root `manifest.json` whose `windows`
list carries the executed controls per tic (button names), per-tic scene PSNR and a richness score.

    python -m pytest paper/fixtures/test_compose_teaser.py -q
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
import compose_teaser as ct  # noqa: E402

TICS = 41
# the unseen rollout's executed control: forward, then attack, then turn left + attack, then strafe, ...
UNSEEN_CONTROLS = ([["MOVE_FORWARD"]] * 6 + [["ATTACK"]] * 6 + [["TURN_LEFT", "ATTACK"]] * 8 + [["MOVE_RIGHT"]] * 6
                   + [[]] * 5 + [["ATTACK"]] * 10)
HOME_CONTROLS = [["TURN_RIGHT"]] * 10 + [["ATTACK"]] * 10 + [["MOVE_FORWARD"]] * 21


def img(seed, tic):
    rng = np.random.default_rng(seed)
    a = np.zeros((240, 320, 3), np.uint8)
    a[:208] = np.clip(rng.integers(40, 200, 3) + 3 * tic, 0, 255)
    a[208:] = 80
    return a


def write_window(root, name, rows, controls, richness, windows):
    d = os.path.join(str(root), name)
    for k, row in enumerate(rows):
        os.makedirs(os.path.join(d, row), exist_ok=True)
        for t in range(TICS):
            Image.fromarray(img(k + 7 * len(name), t)).save(os.path.join(d, row, f"tic_{t:03d}.png"))
    windows.append({"window": name, "map": name.split("_ep")[0], "episode": 41, "start": 269, "richness": richness,
                    "controls": ["+".join(c) for c in controls],
                    "per_tic": {row: {"scene_psnr": [25.0 - 0.1 * t - k for t in range(TICS)]}
                                for k, row in enumerate(rows)}})


def export(tmp_path, extra_unseen=False):
    root, windows = tmp_path / "teaser", []
    write_window(root, "unseen_arena07_ep41_s269", ("truth_raw", "unet_tuned", "adapter_tuned"), UNSEEN_CONTROLS,
                 0.9, windows)
    if extra_unseen:
        write_window(root, "unseen_arena12_ep9_s100", ("truth_raw", "unet_tuned", "adapter_tuned"), UNSEEN_CONTROLS,
                     0.4, windows)
    write_window(root, "train_map02_ep6024_s1428", ("truth_raw", "unet_tuned"), HOME_CONTROLS, 0.8, windows)
    with open(root / "manifest.json", "w") as f:
        json.dump({"what": "synthetic", "windows": windows}, f)
    return str(root)


def mediabox(path):
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", open(path, "rb").read())
    return float(m.group(1)) / 72, float(m.group(2)) / 72


def test_control_changes_and_their_printed_names():
    ctrl = ct.controls_of({"controls": ["+".join(c) for c in UNSEEN_CONTROLS]})
    assert ct.change_moments(ctrl, 1, 40) == [6, 12, 20, 26, 31]
    assert ct.action_text(ctrl[12]) == "attack + turn left"
    assert ct.action_text(frozenset()) == "no input"
    assert ct.spread(list(range(10)), 4) == [0, 3, 6, 9]


def test_layout_a_picks_the_best_ranked_windows_and_draws_every_moment(tmp_path):
    root = export(tmp_path, extra_unseen=True)
    paths, rec = ct.layout_a(root, str(tmp_path / "out"), moments=4)
    assert rec["window"] == "unseen_arena07_ep41_s269" and rec["home"] == "train_map02_ep6024_s1428"
    assert rec["tics"] == [6, 12, 26, 31]              # four of the five control changes, spread
    assert [c["action"] for c in rec["columns"]] == ["attack", "attack + turn left", "no input", "attack"]
    assert rec["columns"][0]["scene_psnr"] == {"model": pytest.approx(25.0 - 0.6 - 1), "adapted":
                                               pytest.approx(25.0 - 0.6 - 2)}
    assert rec["home_tic"] == 10                        # the training map's first attack, the first column's action
    pdf = str(tmp_path / "out" / "fig_teaser.pdf")
    assert pdf in paths and b"/Type3" not in open(pdf, "rb").read()
    w, _ = mediabox(pdf)
    assert w == pytest.approx(5.5, abs=0.01)


def test_layout_b_one_row_per_action_at_eight_tics(tmp_path):
    root = export(tmp_path)
    paths, rec = ct.layout_b(root, str(tmp_path / "out"), actions=3)
    assert [r["action"] for r in rec["rows"]] == ["attack", "attack + turn left", "move right"]
    assert [r["tic"] for r in rec["rows"]] == [6, 12, 20]
    assert rec["rows"][0]["home_tic"] == 10 and rec["horizon"] == 8
    assert os.path.getsize(tmp_path / "out" / "fig_teaser_actions.png") > 2000


def test_a_missing_frame_stops_with_its_path(tmp_path):
    root = export(tmp_path)
    os.remove(os.path.join(root, "unseen_arena07_ep41_s269", "adapter_tuned", "tic_026.png"))
    with pytest.raises(SystemExit, match="adapter_tuned/tic_026"):
        ct.layout_a(root, str(tmp_path / "out"), moments=4)
