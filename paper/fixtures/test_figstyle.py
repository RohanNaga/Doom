"""`paper/figstyle.py`: the encoding table, print sizes and the refusal of degenerate figures.

    python -m pytest paper/fixtures/test_figstyle.py -q
"""
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

pytest.importorskip("matplotlib")
import figstyle as fs  # noqa: E402


def mediabox(path):
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", open(path, "rb").read())
    return float(m.group(1)) / 72, float(m.group(2)) / 72


def test_the_encoding_table_is_the_standards_okabe_ito_set_keyed_by_entity():
    assert fs.BACKBONES["unet"] == fs.Entity("U-Net", "#0072B2", "o")
    assert fs.BACKBONES["pixart"].colour == "#D55E00" and fs.BACKBONES["pixart"].marker == "s"
    assert fs.BACKBONES["sd35"].colour == "#009E73" and fs.BACKBONES["sd35"].marker == "^"
    assert fs.BACKBONES["adapter"].marker == "D"
    assert (fs.TRAINING_BAND, fs.TRAINING_LINE) == ("#E8E8E8", "#707070")
    # every entity keeps one colour and one marker: no two backbones share either
    assert len({e.colour for e in fs.BACKBONES.values()}) == 4 and len({e.marker for e in fs.BACKBONES.values()}) == 4
    assert [fs.backbone_of(n) for n in ("unet200k_ema_tuned", "pixart200k_ema", "sd35_170000",
                                        "adapt4000_live_tuned")] == ["unet", "pixart", "sd35", "adapter"]


def test_the_adapter_ramp_is_light_at_high_skill_and_dark_at_low_skill():
    from matplotlib.colors import rgb_to_hsv, to_rgb
    light, dark = fs.ramp_colour(3.0, 1.0, 3.0), fs.ramp_colour(1.0, 1.0, 3.0)
    assert rgb_to_hsv(to_rgb(light))[2] > rgb_to_hsv(to_rgb(dark))[2]
    assert fs.ramp_colour(None, 1.0, 3.0) == fs.BACKBONES["adapter"].colour


def test_a_figure_with_data_is_written_at_its_size_with_truetype_fonts(tmp_path):
    fs.style()
    fig, (ax,) = fs.new_figure((fs.TEXT_WIDTH, 1.6))
    ax.plot([0, 1, 2], [1, 2, 3], color=fs.BACKBONES["unet"].colour)
    fs.copy_last_line(ax)
    fs.training_line(ax, 2.5)
    ax.set_xlabel("x (unit)")
    pdf, png = fs.save(fig, str(tmp_path), "ok")
    assert os.path.getsize(png) > 1000
    assert b"/Type3" not in open(pdf, "rb").read()
    w, h = mediabox(pdf)
    assert w == pytest.approx(fs.TEXT_WIDTH, abs=0.01) and h == pytest.approx(1.6, abs=0.01)


def test_an_empty_panel_is_refused_with_its_reason(tmp_path):
    fs.style()
    fig, (ax, ax2) = fs.new_figure((3, 1.5), ncols=2)
    ax.plot([0, 1], [0, 1])
    ax2.axhline(0)                    # a reference line alone is not data
    ax2.set_label("seed differences")
    with pytest.raises(fs.DegenerateFigure, match="seed differences has no data"):
        fs.save(fig, str(tmp_path), "empty")
    assert not os.path.exists(tmp_path / "empty.pdf")


def test_a_curve_at_a_single_x_value_is_refused(tmp_path):
    fs.style()
    fig, (ax,) = fs.new_figure((2, 1.5))
    for v in (1.0, 2.0, 3.0):
        ax.plot([8], [v], marker="o", ls="-")          # one ladder rung per arena: nothing to connect
    with pytest.raises(fs.DegenerateFigure, match="single x value"):
        fs.save(fig, str(tmp_path), "one_x")


def test_a_dot_plot_with_one_point_per_row_is_not_a_single_x_curve(tmp_path):
    fs.style()
    fig, (ax,) = fs.new_figure((2, 1.5))
    ax.plot([1.0, 2.0, 1.5], [0, 1, 2], ls="none", marker="o")
    ax.plot([0.8, 1.2], [0, 0], lw=0.6)                  # an interval whisker
    fs.save(fig, str(tmp_path), "dots")


def test_a_legend_entry_with_no_drawn_series_is_refused(tmp_path):
    fs.style()
    fig, (ax,) = fs.new_figure((2, 1.5))
    ax.plot([0, 1], [0, 1], color="#0072B2", label="base")
    ax.plot([], [], color="#D55E00", label="lr 3e-4")    # listed, never drawn
    ax.legend()
    with pytest.raises(fs.DegenerateFigure, match="lr 3e-4"):
        fs.save(fig, str(tmp_path), "legend")


def test_text_below_the_six_point_floor_is_refused(tmp_path):
    fs.style()
    fig, (ax,) = fs.new_figure((2, 1.5))
    ax.plot([0, 1], [0, 1])
    ax.text(0.5, 0.5, "7", fontsize=4)
    with pytest.raises(fs.DegenerateFigure, match="below the 6 pt floor"):
        fs.save(fig, str(tmp_path), "tiny")


def test_the_step_axis_puts_zero_on_its_own_spine_segment():
    fs.style()
    fig, (ax,) = fs.new_figure((3, 1.5))
    z = fs.step_axis(ax, [0, 250, 500, 1000, 2000, 4000])
    assert z == pytest.approx(100)
    lo, hi = ax.spines["bottom"].get_bounds()
    assert z < lo < 250 and hi >= 4000
    labels = [t.get_text() for t in ax.xaxis.get_ticklabels()]
    assert labels[0] == "0" and "4k" in labels
    z8 = fs.step_axis(ax, [0, 50, 100, 150, 250, 500, 1000, 2000, 4000, 8000])
    assert z8 == pytest.approx(20)
    assert [t.get_text() for t in ax.xaxis.get_ticklabels()] == ["0", "50", "250", "1k", "8k"]
    fs.plt.close(fig)
