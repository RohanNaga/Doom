"""`tools/training_curves.py`: one-tic PSNR against updates for the three backbones, from their curve files.

Every test writes synthetic curve files in the layout the steward exports, one JSON per backbone at
`<curves-dir>/<backbone>.json`, next to a combined export that is not a backbone file and must be ignored:

  unet:   EMA and live reads at 5k, 10k and 20k; the EMA starts far below the axis (its warm-up)
  pixart: EMA reads only, so no dotted line is drawn for it
  sd35:   EMA and live reads, plus one EMA read carrying a note (drawn as an open marker, not joined)

    python -m pytest paper/fixtures/test_training_curves.py -q
"""
import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))

pytest.importorskip("matplotlib")
import training_curves as tc  # noqa: E402

PERSISTENCE = {"psnr": 21.5, "lpips": 0.2}


def read(step, weights, psnr, lpips=0.2, note=None):
    return {"step": step, "weights": weights, "psnr": psnr, "lpips": lpips, "note": note}


CURVES = {
    "unet": {"backbone": "unet", "label": "U-Net (SD 1.4)", "persistence": PERSISTENCE,
             "reads": [read(20000, "ema", 21.9), read(5000, "ema", 11.8), read(10000, "ema", 14.6),
                       read(5000, "live", 21.1), read(10000, "live", 21.5), read(20000, "live", 21.8)]},
    "pixart": {"backbone": "pixart", "label": "PixArt-alpha", "persistence": PERSISTENCE,
               "reads": [read(5000, "ema", 10.8), read(10000, "ema", 18.0), read(20000, "ema", 21.7)]},
    "sd35": {"backbone": "sd35", "label": "SD 3.5 Medium", "persistence": PERSISTENCE, "source": "synthetic",
             "reads": [read(5000, "ema", 6.7), read(10000, "ema", 20.9), read(20000, "ema", 22.6),
                       read(5000, "live", 21.3), read(10000, "live", 22.0), read(20000, "live", 22.4),
                       read(35000, "ema", 23.0, note="provisional read")]},
}


def write_curves(root, curves=CURVES):
    d = root / "training_curves"
    d.mkdir(parents=True, exist_ok=True)
    for name, body in curves.items():
        (d / f"{name}.json").write_text(json.dumps(body))
    # the steward's combined export lives in the same directory and is not a backbone file
    (d / "training_curves_h1.json").write_text(json.dumps({"runs": {"unet": {"reads": []}}}))
    return d


def mediabox(path):
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", open(path, "rb").read())
    return float(m.group(1)) / 72, float(m.group(2)) / 72


def test_each_backbone_is_read_from_its_own_file_in_backbone_order(tmp_path):
    curves = tc.load_curves(str(write_curves(tmp_path)))
    assert [c["backbone"] for c in curves] == ["unet", "pixart", "sd35"]
    assert curves[0]["label"] == "U-Net (SD 1.4)"
    assert curves[0]["persistence"] == {"psnr": 21.5, "lpips": 0.2}


def test_a_missing_backbone_file_is_skipped_and_the_others_drawn(tmp_path):
    d = write_curves(tmp_path, {"unet": CURVES["unet"], "sd35": CURVES["sd35"]})
    assert [c["backbone"] for c in tc.load_curves(str(d))] == ["unet", "sd35"]


def test_a_directory_without_any_backbone_file_stops_with_a_message(tmp_path):
    d = write_curves(tmp_path, {})
    with pytest.raises(SystemExit, match="no curve file"):
        tc.load_curves(str(d))


def test_a_file_whose_backbone_disagrees_with_its_name_is_refused(tmp_path):
    d = write_curves(tmp_path, {"unet": {**CURVES["unet"], "backbone": "pixart"}})
    with pytest.raises(SystemExit, match="unet.json"):
        tc.load_curves(str(d))


@pytest.mark.parametrize("bad", [{"weights": "average"}, {"psnr": None}, {"psnr": float("nan")}, {"step": -5}])
def test_a_malformed_read_is_refused(tmp_path, bad):
    body = json.loads(json.dumps(CURVES["pixart"]))
    body["reads"][0].update(bad)
    with pytest.raises(SystemExit):
        tc.load_curves(str(write_curves(tmp_path, {"pixart": body})))


def test_a_repeated_step_of_one_weight_set_is_refused(tmp_path):
    body = json.loads(json.dumps(CURVES["pixart"]))
    body["reads"].append(read(5000, "ema", 11.0))
    with pytest.raises(SystemExit, match="5000"):
        tc.load_curves(str(write_curves(tmp_path, {"pixart": body})))


def test_series_are_sorted_in_thousands_of_updates_and_noted_reads_are_kept_apart(tmp_path):
    unet, pixart, sd35 = tc.load_curves(str(write_curves(tmp_path)))
    ema = tc.series(unet, "ema")
    assert ema["k_updates"] == [5.0, 10.0, 20.0] and ema["psnr"] == [11.8, 14.6, 21.9]
    assert tc.series(pixart, "live")["k_updates"] == []
    joined, apart = tc.series(sd35, "ema"), tc.series(sd35, "ema", noted=True)
    assert joined["k_updates"] == [5.0, 10.0, 20.0]
    assert apart["k_updates"] == [35.0] and apart["psnr"] == [23.0]


def test_the_psnr_axis_clips_the_ema_warm_up_and_lists_every_clipped_read(tmp_path):
    curves = tc.load_curves(str(write_curves(tmp_path)))
    lo, hi = tc.y_limits(curves)
    # absolute PSNR: the floor sits one dB under the lowest live read (21.1), the top above the highest read (23.0)
    assert lo == pytest.approx(21.1 - tc.FLOOR_BELOW_LIVE) and hi >= 23.0
    clipped = {(r["backbone"], r["step"]) for r in tc.clipped_reads(curves)}
    assert clipped == {("unet", 5000), ("unet", 10000), ("pixart", 5000), ("pixart", 10000), ("sd35", 5000)}
    # the gain over persistence stays available for the record, not drawn
    assert tc.series(curves[0], "ema")["gain"] == pytest.approx([11.8 - 21.5, 14.6 - 21.5, 21.9 - 21.5])


def test_the_figure_is_written_as_pdf_and_png_at_the_panel_size(tmp_path):
    d = write_curves(tmp_path)
    out = tmp_path / "figures"
    assert tc.main(["--curves-dir", str(d), "--out-dir", str(out)]) == 0
    for ext in ("pdf", "png"):
        p = out / f"fig1_curves.{ext}"
        assert p.exists() and p.stat().st_size > 1000, p
    pdf = out / "fig1_curves.pdf"
    assert open(pdf, "rb").read(4) == b"%PDF"
    assert b"/Type3" not in open(pdf, "rb").read()
    w, h = mediabox(pdf)
    assert w == pytest.approx(tc.PANEL_SIZE[0], abs=0.01) and h == pytest.approx(tc.PANEL_SIZE[1], abs=0.01)
    w, h = mediabox(out / "figA_training_curves.pdf")
    assert w == pytest.approx(tc.APPENDIX_SIZE[0], abs=0.01)
    side = json.load(open(out / "fig1_curves.json"))
    assert side["weights"] == ["ema"] and side["noted_reads"][0]["step"] == 35000
    assert {r["backbone"] for r in side["clipped_reads"]} == {"unet", "pixart", "sd35"}


def test_the_drawn_lines_are_ema_solid_live_dotted_in_absolute_psnr_with_no_persistence(tmp_path):
    curves = tc.load_curves(str(write_curves(tmp_path)))
    fig, ax = tc.draw(curves, weights=tc.WEIGHTS, size=tc.APPENDIX_SIZE)
    lines = {ln.get_label(): ln for ln in ax.get_lines()}
    assert lines["U-Net (SD 1.4) EMA"].get_linestyle() == "-" and lines["U-Net (SD 1.4) live"].get_linestyle() == ":"
    assert "PixArt-alpha live" not in lines          # no live reads, no dotted line
    assert lines["U-Net (SD 1.4) EMA"].get_color() == "#0072B2"      # the encoding table's U-Net blue
    assert "persistence" not in lines and "persistence" not in ax.get_ylabel()     # nothing relative to it
    assert list(lines["U-Net (SD 1.4) EMA"].get_ydata()) == pytest.approx([11.8, 14.6, 21.9])     # absolute PSNR
    assert ax.get_xlabel().startswith("updates (thousands)") and "PSNR (dB)" in ax.get_ylabel()
    body_fig, body = tc.draw(curves)                     # the body panel: EMA only
    assert not any(ln.get_label().endswith(" live") for ln in body.get_lines())
