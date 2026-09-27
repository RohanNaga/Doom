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


def img(seed, tic, colour=True, luma=None):
    rng = np.random.default_rng(seed * 1000 + tic)
    a = np.zeros((240, 320, 3), np.uint8)
    base = rng.integers(40, 200, 3) if colour else np.full(3, rng.integers(40, 200))
    if luma is not None:
        base = np.full(3, luma(tic))
    a[:208] = base
    a[:208, ::16] = 255 - base                                   # vertical stripes: some edges on screen
    a[208:] = 80
    return a


def write_window(root, name, controls, zs, ad=None, colour=True, luma=None):
    d = os.path.join(str(root), name)
    rows = ["truth_raw", "unet_tuned"] + (["adapter_tuned"] if ad is not None else [])
    for k, row in enumerate(rows):
        os.makedirs(os.path.join(d, row), exist_ok=True)
        for t in range(1, TICS + 1):
            Image.fromarray(img(k + len(name), t, colour, luma)).save(os.path.join(d, row, f"tic_{t:03d}.png"))
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
MAP_2 = held([(2, 11, ["attack"]), (13, 21, ["turn right"]), (24, 32, ["turn right"])])
MAP_2B = held([(5, 13, ["forward"]), (24, 32, ["forward"])])


def export(tmp_path):
    root = tmp_path / "teaser"
    write_window(root, "unseen_arena07_ep41_s100", ARENA_A, zs=lambda t: 16.0, ad=lambda t: 16.0 + 0.1 * t)
    write_window(root, "unseen_arena07_ep9_s200", ARENA_B, zs=lambda t: 15.0,
                 ad=lambda t: 15.5 + (1.0 if t == 28 else 0.0))       # the overlapping forward at 20 scores best
    # map 2's first turn right (tics 5 to 13) is bright, its second (20 to 28) dark
    write_window(root, "train_map02_ep6008_s712", MAP_2, zs=lambda t: 25.0 - 0.1 * t, colour=False,
                 luma=lambda t: 200 if 13 <= t <= 21 else 50)
    write_window(root, "train_map02_ep6024_s900", MAP_2B, zs=lambda t: 24.0 - 0.1 * t, colour=False)
    return str(root)


def restart_export(tmp_path, entries):
    """The steward's restart layout: <window>_t<T>/{truth_raw,unet_tuned[,adapter_tuned]}/tic_00..16 and a
    manifest whose `scene_psnr_vs_truth_raw` holds each row's series over restart tics 0..16 (tic 0 is the moment
    frame itself, the last ground-truth context)."""
    root = tmp_path / "teaser_restart"
    for name, unseen in entries:
        d = root / name
        rows = ["truth_raw", "unet_tuned"] + (["adapter_tuned"] if unseen else [])
        for k, row in enumerate(rows):
            os.makedirs(d / row, exist_ok=True)
            for t in range(17):
                Image.fromarray(img(90 + k, t)).save(d / row / f"tic_{t:02d}.png")
        series = {"unet_tuned": [30.0 - t for t in range(17)]}
        if unseen:
            series["adapter_tuned"] = [31.0 - t + (2.0 if "ep41" in name else 0.0) + (0.5 if t == 16 else 0.0)
                                       for t in range(17)]
        with open(d / "manifest.json", "w") as f:
            steps = [{"tic": k, "control": ["speed", "forward"] + (["attack"] if k <= 5 else ["turn left"])}
                     for k in range(1, 17)]
            json.dump({"scene_psnr_vs_truth_raw": series, "copylast_scene_psnr": [99.0] + [18.0] * 16,
                       "control_per_tic": steps}, f)
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
    # in-domain: map 2's own turn right; both starts lie within 20 rollout tics of the arena row's depth 23 (13 and
    # 28), and the brighter one wins although the other is closer
    assert rows["turn"]["home"]["tic"] == 13 and rows["turn"]["home"]["button"] == "turn right"
    assert rows["turn"]["home"]["near"] is True
    # forward: the only near start (24, depth 32) reads below the map-2 median, so it is taken and flagged
    assert (rows["forward"]["home"]["window"], rows["forward"]["home"]["tic"]) == ("train_map02_ep6024_s900", 24)
    assert rows["forward"]["home"]["at_floor"] is False
    # B's attack at depth 36 has no map-2 attack within 20 tics (depth 10): the closest is taken and flagged
    b_attack = next(r for r in picks[1]["rows"] if r["row"] == "attack")
    assert b_attack["home"]["tic"] == 2 and b_attack["home"]["near"] is False
    # a candidate's rows never overlap: their +8 spans are disjoint, so B's forward row is its start at 5
    assert {r["row"]: r["tic"] for r in picks[1]["rows"]} == {"attack": 28, "turn": 18, "forward": 5}
    assert rec["missing_on_map2"] == ["strafe", "turn left"]          # map 2 never starts either
    for n in (1, 2):
        pdf = out / f"fig_teaser_C_{n}.pdf"
        m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", open(pdf, "rb").read())
        w, h = float(m.group(1)) / 72, float(m.group(2)) / 72
        assert w == pytest.approx(5.5, abs=0.01) and h <= 2.2 + 1e-6
    # seven frames per row: the in-domain block (context, U-Net, ground truth) and the unseen block (context,
    # zero-shot, adapted, ground truth); without a restart export the +8 frames are headed by their rollout tic
    c1 = json.load(open(out / "fig_teaser_C_1.json"))
    assert c1["mode"] == "rollout" and len(c1["columns"]) == 7 and "rollout tic t+8" in c1["note"]
    assert [c["name"] for c in c1["columns"]][3:] == ["context (t = 0)", "zero-shot", "adapted", "ground truth"]
    assert c1["frame_in"][0] > 0.55 and c1["size_in"][1] <= 2.05 + 1e-6


def test_layout_c_reads_the_restart_export_so_the_plus_8_frame_is_8_tics_after_the_context(tmp_path):
    root = export(tmp_path)
    rec = tcs.run(root, str(tmp_path / "out"), str(tmp_path / "out" / "review"), n_candidates=1)
    rows = rec["candidates"][0]["rows"]
    names = [(f"{r['window']}_t{r['tic']}", True) for r in rows] + \
            [(f"{r['home']['window']}_t{r['home']['tic']}", False) for r in rows]
    restart = restart_export(tmp_path, names[:-1])                 # the last map-2 moment is missing
    with pytest.raises(SystemExit, match=names[-1][0]):
        tcs.ct.layout_c(root, str(tmp_path / "c"), rows, restart_root=restart)
    restart = restart_export(tmp_path, names)
    paths, side = tcs.ct.layout_c(root, str(tmp_path / "c"), rows, restart_root=restart)
    assert side["mode"] == "restart" and "+8 tics after 32 ground-truth context tics" in side["note"]
    first = side["rows"][0]
    assert first["sources"]["zero-shot"].endswith(os.path.join(names[0][0], "unet_tuned", "tic_08.png"))
    assert first["sources"]["context"].endswith(os.path.join(names[0][0], "truth_raw", "tic_00.png"))
    assert first["scene_psnr"] == {"zero-shot": 22.0, "adapted": 25.0, "in-domain": 22.0}
    assert first["story_shown"] == pytest.approx(3.0)


def test_with_a_restart_export_the_candidates_are_redrawn_from_it_and_rescored(tmp_path):
    root = export(tmp_path)
    # the steward restarted arena A's three rows and map 2's attack, second turn right and forward starts only
    a_rows = [("unseen_arena07_ep41_s100_t3", True), ("unseen_arena07_ep41_s100_t15", True),
              ("unseen_arena07_ep41_s100_t26", True)]
    home = [("train_map02_ep6008_s712_t2", False), ("train_map02_ep6008_s712_t24", False),
            ("train_map02_ep6024_s900_t24", False)]
    restart = restart_export(tmp_path, a_rows + home)
    out = tmp_path / "out"
    rec = tcs.run(root, str(out), str(out / "review"), n_candidates=2, restart_root=restart,
                  restart_horizons=(8, 16))
    first, second = rec["candidates"]
    r = first["restart"]["8"]
    assert r["files"][0] == "fig_teaser_C_1_restart.pdf" and r["score_shown"] == pytest.approx(3.0)
    r16 = first["restart"]["16"]                                     # the +16 variant, rescored at +16
    assert r16["files"][0] == "fig_teaser_C_1_restart16.pdf" and r16["score_shown"] == pytest.approx(3.5)
    assert "+16 tics" in json.load(open(out / "fig_teaser_C_1_restart16.json"))["note"]
    # the in-domain start is the restarted one of the same button (turn right at 20, not the unexported 5)
    homes = {row["row"]: (row["home"]["window"], row["home"]["tic"]) for row in r["rows"]}
    assert homes == {"attack": ("train_map02_ep6008_s712", 2), "turn": ("train_map02_ep6008_s712", 24),
                     "forward": ("train_map02_ep6024_s900", 24)}
    assert "unseen_arena07_ep9_s200_t" in second["restart"]["skipped"]      # B was not restarted
    side = json.load(open(out / "fig_teaser_C_1_restart.json"))
    assert side["mode"] == "restart" and "+8 tics" in side["note"]
    # persistence under both context frames, from each moment's copy-last series at +8
    row = side["rows"][0]
    assert row["persistence"] == {"in-domain": 18.0, "unseen": 18.0}


def test_the_in_domain_start_is_the_brightest_near_one_that_reads_at_least_the_map_2_median():
    row = {"depth": 50}
    homes = [{"window": "w", "tic": 1, "depth": 45, "luma": 180.0, "in_domain": 7.0},   # a flash the model misses
             {"window": "w", "tic": 2, "depth": 60, "luma": 90.0, "in_domain": 21.0},
             {"window": "v", "tic": 3, "depth": 52, "luma": 40.0, "in_domain": 24.0},
             {"window": "w", "tic": 40, "depth": 200, "luma": 250.0, "in_domain": 30.0}]  # outside the 20-tic window
    home, flags = tcs.pick_home(homes, row, floor=20.0)
    assert home["tic"] == 2 and flags == {"near": True, "at_floor": True}
    # a plasma flash on the context or ground-truth frame (map 2 is grey stone, chroma 2 to 5) never wins on brightness
    flash = {"window": "u", "tic": 5, "depth": 50, "luma": 200.0, "in_domain": 23.0, "flash": True}
    home, flags = tcs.pick_home(homes + [flash], row, floor=20.0)
    assert home["tic"] == 2
    home, flags = tcs.pick_home(homes[:1], row, floor=20.0)                   # nothing near reads that well
    assert home["tic"] == 1 and flags == {"near": True, "at_floor": False}
    home, flags = tcs.pick_home(homes[3:], row, floor=20.0)                   # nothing near at all: the closest
    assert home["tic"] == 40 and flags == {"near": False, "at_floor": True}
    # a start whose +8 span overlaps one another row already shows is never reused (the same frames twice)
    home, flags = tcs.pick_home(homes, row, floor=20.0, taken=[("w", 9)])
    assert home["tic"] == 3
    assert tcs.pick_home(homes[:2], row, floor=20.0, taken=[("w", 2)]) == (None, None)


def test_a_row_names_what_was_really_held_over_the_steps_shown():
    steps = [{"forward", "turn right", "speed"}] * 7 + [{"forward", "turn left"}] * 4 + [{"forward", "turn right"}] * 5
    assert tcs.ct.held_note(steps, "turn right") == "+forward \u00b7 7/16"      # the run from step 1, then a switch
    steps = [{"attack", "strafe right"}] * 3 + [{"attack", "turn left"}] * 12 + [{"back"}]
    assert tcs.ct.held_note(steps, "attack") == "held 15/16"                    # nothing else held throughout


def test_the_rows_follow_the_requested_order(tmp_path):
    root = export(tmp_path)
    rec = tcs.run(root, str(tmp_path / "out"), str(tmp_path / "out" / "review"), n_candidates=1,
                  row_order=("turn", "forward", "attack"))
    side = json.load(open(tmp_path / "out" / "fig_teaser_C_1.json"))
    assert [r["row"] for r in side["rows"]] == ["turn", "forward", "attack"]
    # the order shown does not change which in-domain start a row gets: those are assigned attack, turn, forward
    plain = tcs.run(root, str(tmp_path / "plain"), str(tmp_path / "plain" / "review"), n_candidates=1)
    homes = {r["row"]: (r["home"]["window"], r["home"]["tic"]) for r in rec["candidates"][0]["rows"]}
    assert homes == {r["row"]: (r["home"]["window"], r["home"]["tic"]) for r in plain["candidates"][0]["rows"]}


def test_a_held_moment_has_its_button_pressed_on_every_restart_step():
    ctrl = [frozenset()] + [frozenset({"attack", "speed"}) if 2 <= t <= 20 else frozenset({"speed"})
                            for t in range(1, 31)]
    got = tcs.held_moments(ctrl, 30, steps=16)
    assert [(b, t, starts) for _, b, t, starts in got] == [("attack", 1, True), ("attack", 2, False),
                                                           ("attack", 3, False), ("attack", 4, False)]


def test_the_hold_round_picks_rows_across_windows_with_a_held_in_domain_start(tmp_path):
    root = export(tmp_path)
    out = tmp_path / "out"
    rec = tcs.run(root, str(out), str(out / "review"), hold_steps=8, row_order=("turn", "forward", "attack"))
    picks = {p["row"]: p for p in rec["hold"]["rows"]}
    # every pick holds its button on all 8 steps after its context; turn is turn right, the only turn map 2 holds
    for p in picks.values():
        for m in p["unseen"] + [p["home"]]:
            assert m["held"] == 8
    assert all(m["button"] == "turn right" for m in picks["turn"]["unseen"] + [picks["turn"]["home"]])
    # the unseen picks are ranked by story at +8, and the two picks of a row never share frames
    stories = [m["story"] for m in picks["attack"]["unseen"]]
    assert stories == sorted(stories, reverse=True)
    a, b = picks["attack"]["unseen"][:2]
    assert a["window"] != b["window"] or abs(a["tic"] - b["tic"]) > 8
    assert rec["hold"]["counts"]["training:turn left"] == 0
    side = json.load(open(out / "fig_teaser_C_hold8.json"))
    assert [r["row"] for r in side["rows"]] == ["turn", "forward", "attack"]
    assert all(r["held"]["unseen"].endswith("8/8") and r["held"]["in-domain"].endswith("8/8") for r in side["rows"])
