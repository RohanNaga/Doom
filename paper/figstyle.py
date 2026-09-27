"""
The paper's figure style: the encoding table of `paper/FIGURE_STANDARDS.md` as constants, print sizes for the 5.5 in
single-column CoRL page, and a save path that refuses a degenerate figure.

    import figstyle as fs
    fs.style()
    fig, (ax,) = fs.new_figure((fs.TEXT_WIDTH, 1.6))
    ax.plot(x, y, color=fs.BACKBONES["unet"].colour, lw=fs.DATA_LW)
    fs.copy_last_line(ax)
    fs.save(fig, out_dir, "fig_name")          # raises fs.DegenerateFigure with the reason instead of writing

Every figure script in `paper/` and `tools/` imports this module rather than setting its own colours or sizes, so a
colour or a marker means one thing across the paper (standard section 1) and every label prints at 7 pt, every
tick at 6.5 pt and nothing below 6 pt (section 2). `save` runs `refuse_degenerate` first: an axes with no data,
a line plot whose data sit at a single x value, a legend entry that matches no drawn series, or text under the
6 pt floor stops the build with a message naming the figure and the axes (section 5, the last row of the table).
"""
import math
import os
from dataclasses import dataclass

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import collections as mcoll  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib import font_manager, ticker, transforms  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.image import AxesImage  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.text import Text  # noqa: E402

# ---------------------------------------------------------------------------------------------
# the page and the type (standard sections 0 and 2)
# ---------------------------------------------------------------------------------------------

TEXT_WIDTH = 5.5                  # in; corl_2026.sty textwidth, single column
LABEL_PT = 7.0                    # axis labels
TICK_PT = 6.5                     # tick labels
ANNOT_PT = 6.5                    # annotations and direct labels
MIN_PT = 6.0                      # nothing prints below this
LETTER_PT = 8.0                   # panel letters, bold lowercase
DATA_LW = 1.0                     # data lines
REF_LW = 0.6                      # reference lines
AXIS_LW = 0.6                     # axes and ticks
TICK_LEN = 2.5
MIN_LW = 0.5
MARKER_SIZE = 3.75                # pt, within the standard's 3.5 to 4
MARKER_EDGE = 0.4                 # white edge where points overlap
PNG_DPI = 600

# ---------------------------------------------------------------------------------------------
# the encoding table (standard section 1): Okabe-Ito as given by Wong (2011)
# ---------------------------------------------------------------------------------------------

BLACK = "#000000"
INK = "#000000"                   # text and axes
CONTEXT_INK = "#404040"           # named arenas, secondary direct labels
FAINT = "#BDBDBD"                 # per-arena context lines behind an aggregate
TRAINING_BAND = "#E8E8E8"         # grey band behind the training maps' points
TRAINING_LINE = "#707070"         # the training maps' reference line, dashed
TRAINING_DASH = (0, (3, 2))
FULL_FT_DASH = (0, (3, 2))        # the full fine-tune: black, dashed, 1.0 pt
SPARE = {"sky": "#56B4E9", "orange": "#E69F00", "yellow": "#F0E442", "purple": "#CC79A7"}  # unused by rule


@dataclass(frozen=True)
class Entity:
    """One row of the encoding table: the name a figure prints, its colour and its marker."""
    label: str
    colour: str
    marker: str


BACKBONES = {
    "unet": Entity("U-Net", "#0072B2", "o"),
    "pixart": Entity("PixArt-α", "#D55E00", "s"),
    "sd35": Entity("SD 3.5", "#009E73", "^"),
    "adapter": Entity("U-Net + LoRA", "#00466E", "D"),     # the U-Net's blue family, darker, diamond
}
TRUE_FRAMES = Entity("true", BLACK, "")
FULL_FINE_TUNE = Entity("full fine-tune", BLACK, "")
BACKBONE_ORDER = ("unet", "pixart", "sd35")

# the single-hue blue ramp of the adapter curves: light = high zero-shot skill (small deficit), dark = low
ADAPTER_RAMP = LinearSegmentedColormap.from_list("adapter_blue", ("#9DC3E2", "#0072B2", "#002B45"))


def backbone_of(row):
    """The encoding-table key of a result row's name: `adapt*` the adapter, `pixart*`, `sd35*`, else the U-Net."""
    row = row.lower()
    if row.startswith("adapt"):
        return "adapter"
    if row.startswith("pixart"):
        return "pixart"
    if row.startswith("sd35"):
        return "sd35"
    return "unet"


def ramp_colour(value, lo, hi):
    """The adapter ramp's colour for `value` on [lo, hi]: light at the high end, dark at the low end."""
    if value is None or not math.isfinite(value) or hi <= lo:
        return BACKBONES["adapter"].colour
    return ADAPTER_RAMP(1.0 - min(1.0, max(0.0, (value - lo) / (hi - lo))))


# ---------------------------------------------------------------------------------------------
# style, figures, labels
# ---------------------------------------------------------------------------------------------

def font_family():
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in ("Arial", "Helvetica", "DejaVu Sans") if f in have), "DejaVu Sans")


def style():
    """rcParams at print size: Arial, 7 pt labels, 6.5 pt ticks, 0.6 pt axes, TrueType embedding (no Type 3)."""
    fam = font_family()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [fam, "DejaVu Sans"], "font.size": ANNOT_PT,
        "axes.labelsize": LABEL_PT, "axes.titlesize": LABEL_PT, "legend.fontsize": ANNOT_PT,
        "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
        "axes.linewidth": AXIS_LW, "xtick.major.width": AXIS_LW, "ytick.major.width": AXIS_LW,
        "xtick.minor.width": MIN_LW, "ytick.minor.width": MIN_LW, "xtick.major.size": TICK_LEN,
        "ytick.major.size": TICK_LEN, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.direction": "out", "ytick.direction": "out", "lines.linewidth": DATA_LW,
        "lines.markersize": MARKER_SIZE, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "text.color": INK, "axes.grid": False, "legend.frameon": False, "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.2, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.labelpad": 2.0,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.titlepad": 2.0,
        "mathtext.fontset": "dejavusans" if fam == "DejaVu Sans" else "custom", "mathtext.rm": fam,
        "mathtext.it": f"{fam}:italic", "mathtext.bf": f"{fam}:bold", "mathtext.sf": fam,
        "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.unicode_minus": True,
    })


def new_figure(size, ncols=1, nrows=1, width_ratios=None, height_ratios=None, sharex=False, sharey=False,
               w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.04):
    """A figure of exactly `size` inches with constrained layout; returns (fig, flat list of axes)."""
    fig, axes = plt.subplots(nrows, ncols, figsize=size, layout="constrained", squeeze=False, sharex=sharex,
                             sharey=sharey, gridspec_kw={"width_ratios": width_ratios, "height_ratios": height_ratios})
    fig.get_layout_engine().set(w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace)
    return fig, [ax for row in axes for ax in row]


def panel_letter(ax, letter, dx=-0.02, dy=0.0):
    """The panel letter, 8 pt bold lowercase upright, at the top left of the axes' tight box."""
    ax.text(dx, 1.0 + dy, letter.lower(), transform=ax.transAxes, fontsize=LETTER_PT, fontweight="bold",
            ha="right", va="bottom", gid="decor")


def direct_label(ax, x, y, text, colour=INK, ha="left", va="center", dx=2.0, dy=0.0, **kw):
    """A 6.5 pt label next to a data point (offset in points), for series labelled at their ends."""
    return ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                       fontsize=ANNOT_PT, color=colour, annotation_clip=False, **kw)


def copy_last_line(ax, orientation="h", label=True, where=1.0, align="right"):
    """The zero line: copy-last (persistence), 0.6 pt black, labelled 'copy-last' at its end once."""
    if orientation == "h":
        ax.axhline(0, color=BLACK, lw=REF_LW, zorder=1.5, gid="ref")
        if label:
            ax.text(where, 0, "copy-last", transform=ax.get_yaxis_transform(), ha=align, va="bottom",
                    fontsize=ANNOT_PT, color=INK, gid="decor")
    else:
        ax.axvline(0, color=BLACK, lw=REF_LW, zorder=1.5, gid="ref")
        if label:
            ax.text(0, where, " copy-last", transform=ax.get_xaxis_transform(), ha="left",
                    va="top" if where >= 0.5 else "bottom", fontsize=ANNOT_PT, color=INK, gid="decor")


def training_line(ax, value, label="training maps", orientation="h", where=0.0, align="left", colour=TRAINING_LINE):
    """The training maps' reference: 0.6 pt dashed grey at `value`, labelled at one end."""
    if orientation == "h":
        ax.axhline(value, color=colour, lw=REF_LW, ls=TRAINING_DASH, zorder=1.4, gid="ref")
        if label:
            ax.text(where, value, (" " if align == "left" else "") + label + (" " if align == "right" else ""),
                    transform=ax.get_yaxis_transform(), ha=align, va="bottom", fontsize=ANNOT_PT,
                    color=TRAINING_LINE, gid="decor")
    else:
        ax.axvline(value, color=colour, lw=REF_LW, ls=TRAINING_DASH, zorder=1.4, gid="ref")
        if label:
            ax.text(value, where, label, transform=ax.get_xaxis_transform(), ha=align, va="bottom",
                    fontsize=ANNOT_PT, color=TRAINING_LINE, rotation=0, gid="decor")


def training_band(ax, lo, hi, orientation="v"):
    """The grey band behind the training maps' points (a span of x when vertical, of y when horizontal)."""
    span = ax.axvspan if orientation == "v" else ax.axhspan
    return span(lo, hi, color=TRAINING_BAND, lw=0, zorder=0, gid="ref")


# ---------------------------------------------------------------------------------------------
# the update axis: log scale with step 0 on its own spine segment (standard, Figure 4)
# ---------------------------------------------------------------------------------------------

def zero_position(steps):
    """Where step 0 sits on the log axis: a factor 2.5 left of the first positive step."""
    pos = sorted(s for s in steps if s > 0)
    return pos[0] / 2.5 if pos else 1.0


def step_label(s):
    return "0" if s == 0 else f"{s // 1000}k" if s >= 1000 and s % 1000 == 0 else f"{s:g}"


def default_step_ticks(steps):
    """The labelled grid steps: 0, then 50, 250, 1k and the last step where the grid has them, else every other."""
    grid = sorted(set(steps) | {0})
    preferred = [s for s in grid if s in (50, 250, 1000, 8000)] + [max(grid)]
    if len([s for s in preferred if s > 0]) >= 3:
        return sorted({0, *preferred})
    pos = [s for s in grid if s > 0]
    return sorted({0, *pos[::2], pos[-1]}) if pos else [0]


def step_axis(ax, steps, label="adapter updates (log scale)", labelled=None, gap=1.07):
    """A log axis of updates with step 0 on a separate spine segment left of a small gap; returns x of step 0.

    `labelled` names the grid steps that get a tick label (default `default_step_ticks`); every other grid step
    gets an unlabelled minor tick. The bottom spine is cut at the geometric midpoint between 0's position and the
    first positive step, so the break reads as two spines, not as a glyph over one.
    """
    z = zero_position(steps)
    grid = sorted(set(steps) | {0})
    ticks = sorted(set(labelled if labelled is not None else default_step_ticks(grid)) | {0})
    ticks = [s for s in ticks if s in grid]
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.FixedLocator([z if s == 0 else s for s in ticks]))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([step_label(s) for s in ticks]))
    ax.xaxis.set_minor_locator(ticker.FixedLocator([s for s in grid if s not in ticks]))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    lo, hi = z / 1.5, max(grid) * 1.5
    ax.set_xlim(lo, hi)
    ax.set_xlabel(label)
    first = min((s for s in grid if s > 0), default=z * 2.5)
    mid = math.sqrt(z * first)
    ax.spines["bottom"].set_bounds(mid * gap, hi)
    under = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.add_line(Line2D([lo, mid / gap], [0, 0], transform=under, lw=AXIS_LW, color=INK, clip_on=False,
                       solid_capstyle="butt", gid="decor"))
    return z


def xs_of(points, z):
    """x positions of (step, value) points on a `step_axis`, with step 0 at `z`."""
    return [z if s == 0 else s for s, _ in points]


# ---------------------------------------------------------------------------------------------
# refusal and saving
# ---------------------------------------------------------------------------------------------

class DegenerateFigure(ValueError):
    """A figure the build must not write: an empty panel, a single x value, a legend entry with no series."""


def _is_data_transform(artist, ax):
    return artist.get_transform() is ax.transData


def _finite_xy(line):
    x = np.asarray(line.get_xdata(), dtype=float)
    y = np.asarray(line.get_ydata(), dtype=float)
    if x.shape != y.shape:
        return np.empty(0), np.empty(0)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def data_lines(ax):
    """The axes' Line2D artists that carry data: data coordinates, at least one finite point, not a reference."""
    out = []
    for ln in ax.lines:
        if ln.get_gid() in ("ref", "decor") or not ln.get_visible() or not _is_data_transform(ln, ax):
            continue
        x, _ = _finite_xy(ln)
        if len(x):
            out.append(ln)
    return out


def data_collections(ax):
    """Collections that carry data: scatter points or filled bands in data coordinates."""
    out = []
    for c in ax.collections:
        if c.get_gid() in ("ref", "decor") or not c.get_visible():
            continue
        if isinstance(c, mcoll.PathCollection):
            off = np.asarray(c.get_offsets(), dtype=float)
            if off.size and np.isfinite(off).all(axis=-1).any() and c.get_offset_transform() is ax.transData:
                out.append(c)
        elif c.get_paths() and _is_data_transform(c, ax):
            out.append(c)
    return out


def has_data(ax):
    return bool(data_lines(ax) or data_collections(ax) or any(isinstance(i, AxesImage) for i in ax.images))


def _same_colour(a, b):
    try:
        return np.allclose(mcolors.to_rgba(a), mcolors.to_rgba(b), atol=0.02)
    except (ValueError, TypeError):
        return False


def _norm_ls(ls):
    return "none" if ls in (None, "", " ", "None", "none") else str(ls)


def _handle_matches(handle, lines, colls):
    """True when a legend handle matches a drawn data series by colour and (for lines) line style and marker."""
    if isinstance(handle, Line2D):
        for ln in lines:
            if not _same_colour(handle.get_color(), ln.get_color()):
                continue
            hl, ll = _norm_ls(handle.get_linestyle()), _norm_ls(ln.get_linestyle())
            if hl != "none" and hl != ll:
                continue
            hm, lm = str(handle.get_marker()), str(ln.get_marker())
            if hl == "none" and hm != lm:
                continue
            return True
        return any(_same_colour(handle.get_color(), fc) for c in colls if isinstance(c, mcoll.PathCollection)
                   for fc in (list(c.get_facecolors()) + list(c.get_edgecolors()))[:4])
    if isinstance(handle, Patch):
        return any(_same_colour(handle.get_facecolor(), fc) for c in colls for fc in list(c.get_facecolors())[:4])
    return True


def refuse_degenerate(fig, stem=""):
    """Raise DegenerateFigure when the figure has an empty panel, a single x value, a legend entry that matches no
    drawn series, or text below the 6 pt floor; axes with gid 'decor' (image or label panels) are exempt."""
    where = f"{stem}: " if stem else ""
    axes = [ax for ax in fig.axes if ax.get_gid() != "decor" and ax.get_visible()]
    if not axes:
        raise DegenerateFigure(f"{where}the figure has no axes")
    lines, colls = [], []
    for i, ax in enumerate(axes):
        name = ax.get_label() or f"axes {i}"
        if not has_data(ax):
            raise DegenerateFigure(f"{where}{name} has no data (an empty panel)")
        curve_x = set()
        for ln in data_lines(ax):
            if _norm_ls(ln.get_linestyle()) != "none":
                curve_x.update(np.round(_finite_xy(ln)[0], 9).tolist())
        if any(_norm_ls(ln.get_linestyle()) != "none" for ln in data_lines(ax)) and len(curve_x) < 2 \
                and not data_collections(ax) and ax.get_gid() != "points":
            raise DegenerateFigure(f"{where}{name}: every line sits at a single x value ({sorted(curve_x)}); "
                                   "a curve needs at least two")
        lines += data_lines(ax)
        colls += data_collections(ax)
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None] + list(fig.legends)
    for leg in legends:
        for handle, text in zip(leg.legend_handles, leg.get_texts()):
            if not _handle_matches(handle, lines, colls):
                raise DegenerateFigure(f"{where}the legend entry {text.get_text()!r} has no drawn series")
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_PT - 1e-6:
            raise DegenerateFigure(f"{where}text {t.get_text()!r} at {t.get_fontsize():g} pt, below the "
                                   f"{MIN_PT:g} pt floor")


def save(fig, out_dir, stem):
    """Refuse a degenerate figure, then write the PDF (no dates, so an unchanged figure rewrites byte-identical)
    and a 600 dpi PNG at the figure's own size; returns the two paths."""
    try:
        refuse_degenerate(fig, stem)
    except DegenerateFigure:
        plt.close(fig)
        raise
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, f"{stem}.pdf"), os.path.join(out_dir, f"{stem}.png")]
    fig.savefig(paths[0], metadata={"CreationDate": None, "Creator": None, "Producer": None})
    fig.savefig(paths[1], dpi=PNG_DPI, metadata={"Software": None})
    plt.close(fig)
    return paths
