"""`tools/teaser_contact_sheet.py`: the Figure 1 selection round on a synthetic export in the steward's layout.

The fixture writes `<root>/<map>_ep<E>_s<S>/{truth_raw,unet_tuned[,adapter_tuned]}/tic_NNN.png` (240 x 320 with a HUD
band, tics 1 to 40) and a per-window `manifest.json` whose `per_tic` list carries the executed control and the
per-tic scene PSNR of each model row, as `results/teaser/` does.

    python -m pytest paper/fixtures/test_teaser_contact_sheet.py -q
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
import teaser_contact_sheet as tcs  # noqa: E402

TICS = 40


def held(spans):
    """Per-tic controls (tics 1..TICS) from [(first, last, buttons)] spans; 'speed' is always held."""
    ctrl = [["speed"] for _ in range(TICS)]
    for first, last, buttons in spans:
        for t in range(first, last + 1):
            ctrl[t - 1] = ctrl[t - 1] + list(buttons)
    return ctrl


def img(seed, tic, colour=True):
    rng = np.random.default_rng(seed * 1000 + tic)
    a = np.zeros((240, 320, 3), np.uint8)
    base = rng.integers(40, 200, 3) if colour else np.full(3, rng.integers(40, 200))
    a[:208] = base
    a[:208, ::16] = 255 - base                                   # vertical stripes: some edges on screen
    a[208:] = 80
    return a


def write_window(root, name, controls, zs, ad=None, colour=True):
    d = os.path.join(str(root), name)
    rows = ["truth_raw", "unet_tuned"] + (["adapter_tuned"] if ad is not None else [])
    for k, row in enumerate(rows):
        os.makedirs(os.path.join(d, row), exist_ok=True)
        for t in range(1, TICS + 1):
            Image.fromarray(img(k + len(name), t, colour)).save(os.path.join(d, row, f"tic_{t:03d}.png"))
    per = []
    for t in range(1, TICS + 1):
        r = {"tic": t, "control": controls[t - 1], "unet_scene_psnr": zs(t)}
        if ad is not None:
            r["adapter_scene_psnr"] = ad(t)
        per.append(r)
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump({"map": name.split("_ep")[0], "per_tic": per}, f)


# arena A: attack starts at 3 (held 10), turn right at 15 (held 9), forward at 26 (held 8); attack from tic 1 in
# arena B is held from the context, so it is not a start; B's turn left at 12 is held only 5 tics; B's second
# forward at 20 overlaps its turn at 18, so the rows take its first forward at 5
ARENA_A = held([(3, 12, ["attack"]), (15, 23, ["turn right"]), (26, 33, ["forward"])])
ARENA_B = held([(1, 10, ["attack"]), (5, 13, ["forward"]), (12, 16, ["turn left"]),
                (18, 27, ["turn right", "strafe left"]), (20, 28, ["forward"]), (28, 36, ["attack"])])
MAP_2 = held([(2, 11, ["attack"]), (5, 14, ["turn right"]), (20, 29, ["turn right"]), (24, 33, ["forward"])])


def export(tmp_path):
    root = tmp_path / "teaser"
    write_window(root, "unseen_arena07_ep41_s100", ARENA_A, zs=lambda t: 16.0, ad=lambda t: 16.0 + 0.1 * t)
    write_window(root, "unseen_arena07_ep9_s200", ARENA_B, zs=lambda t: 15.0,
                 ad=lambda t: 15.5 + (1.0 if t == 28 else 0.0))       # the overlapping forward at 20 scores best
    write_window(root, "train_map02_ep6008_s712", MAP_2, zs=lambda t: 25.0 - 0.1 * t, colour=False)
    return str(root)


def test_a_moment_is_a_start_held_for_eight_tics_and_never_the_first_tic():
    ctrl = tcs.ct.controls_of({"per_tic": [{"tic": t, "control": c} for t, c in enumerate(ARENA_B, 1)]})
    got = tcs.moments(ctrl, TICS)
    assert ("attack", "attack", 1) not in got                     # held since the context: no visible start
    assert not any(a == "turn left" for a, _, _ in got)           # 5 tics is not a held control
    assert ("turn right", "turn right", 18) in got and ("strafe", "strafe left", 18) in got
    assert ("attack", "attack", 28) in got and ("forward", "forward", 5) in got
    assert ("forward", "forward", 20) in got
    assert all(t + tcs.HORIZON <= TICS for _, _, t in got)


def test_colourfulness_and_edge_density_separate_grey_flat_from_coloured_busy_frames():
    grey = np.full((208, 320, 3), 128, np.uint8)
    red = grey.copy()
    red[:] = (200, 30, 30)
    # two-pixel stripes: every pixel sits next to a transition (a one-pixel checkerboard is above Sobel's passband)
    checker = (((np.arange(320) // 2) % 2) * 255).astype(np.uint8)[None, :, None].repeat(208, 0).repeat(3, 2)
    assert tcs.chroma(grey) == pytest.approx(0.0, abs=0.5) and tcs.chroma(red) > 50
    assert tcs.edge_density(grey) == 0.0 and tcs.edge_density(checker) > 0.9


def test_the_round_scores_every_moment_picks_whole_windows_and_matches_map_2_by_depth(tmp_path):
    root = export(tmp_path)
    out = tmp_path / "out"
    rec = tcs.run(root, str(out), str(out / "review"), n_candidates=2)
    assert os.path.getsize(out / "review" / "teaser_contacts.png") > 10000
    side = json.load(open(out / "review" / "teaser_contacts.json"))
    arena = [m for m in side["moments"] if m["role"] == "unseen"]
    a = next(m for m in arena if m["window"] == "unseen_arena07_ep41_s100" and m["action"] == "turn right")
    assert a["tic"] == 15 and a["depth"] == 23 and a["story"] == pytest.approx(0.1 * 23)
    # within an action the unseen moments are sorted by story, best first
    stories = [m["story"] for m in arena if m["action"] == "attack"]
    assert stories == sorted(stories, reverse=True)
    # candidates are whole windows ranked by their three rows' mean story; A beats B
    picks = rec["candidates"]
    assert [p["window"] for p in picks] == ["unseen_arena07_ep41_s100", "unseen_arena07_ep9_s200"]
    rows = {r["row"]: r for r in picks[0]["rows"]}
    assert rows["attack"]["tic"] == 3 and rows["turn"]["tic"] == 15 and rows["forward"]["tic"] == 26
    # in-domain: map 2's own turn right, the start whose rollout depth is closest to the arena row's (23 -> 28)
    assert rows["turn"]["home"]["tic"] == 20 and rows["turn"]["home"]["button"] == "turn right"
    # a candidate's rows never overlap: their +8 spans are disjoint, so B's forward row is its start at 5
    assert {r["row"]: r["tic"] for r in picks[1]["rows"]} == {"attack": 28, "turn": 18, "forward": 5}
    assert rec["missing_on_map2"] == ["strafe", "turn left"]          # map 2 never starts either
    for n in (1, 2):
        pdf = out / f"fig_teaser_C_{n}.pdf"
        m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", open(pdf, "rb").read())
        w, h = float(m.group(1)) / 72, float(m.group(2)) / 72
        assert w == pytest.approx(5.5, abs=0.01) and h <= 2.2 + 1e-6
