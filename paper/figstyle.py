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
TRAINING_LABEL = "training maps (in-distribution)"   # never "home" (Rohan, 2026-09-27)
PERSISTENCE_LABEL = "persistence"   # the copy-last baseline's printed name, the zero line (Rohan, 2026-09-27)
# axis labels (the y-axis survey, opus-metrics-survey-2026-09-27.md): never "advantage" on an axis
A_LABEL = "\u0394PSNR vs persistence (dB)"
A_LABEL_SHORT = "\u0394PSNR vs\npersistence (dB)"
S_LABEL = "latent skill (dB)"
G_LABEL = "gap to reconstruction upper bound (dB)"
# M's sign, one switch: False keeps M = LPIPS(model) - LPIPS(persistence) (lower is better); True flips it so that
# above zero means "beats persistence" on every axis. Rohan decides; the default is the current sign.
M_POSITIVE_IS_BETTER = False


def m_value(v):
    """M as drawn under the `M_POSITIVE_IS_BETTER` switch (None stays None)."""
    return None if v is None else (-v if M_POSITIVE_IS_BETTER else v)


def m_interval(ci):
    """An M interval as drawn under the switch, low end first."""
    return None if not ci else sorted(m_value(x) for x in ci)


def m_label(short=False):
    """M's axis label and its direction."""
    direction = "higher is better" if M_POSITIVE_IS_BETTER else "lower is better"
    return f"$M$, LPIPS difference\n({direction})" if short else f"$M$, LPIPS difference ({direction})"


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


def declutter(values, gap):
    """Label heights for values that must sit at least `gap` apart, in the values' order, each as near its value as
    the others allow (labels pushed apart symmetrically around their cluster's centre)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ys = [float(values[i]) for i in order]
    for _ in range(len(ys) * 4):
        moved = False
        for k in range(1, len(ys)):
            overlap = gap - (ys[k] - ys[k - 1])
            if overlap > 1e-12:
                ys[k - 1] -= overlap / 2
                ys[k] += overlap / 2
                moved = True
        if not moved:
            break
    out = [0.0] * len(values)
    for i, y in zip(order, ys):
        out[i] = y
    return out


def end_labels(ax, points, gap, colour=CONTEXT_INK, dx=3.0, leaders=False, min_shift=None):
    """Direct labels at the right ends of lines: `points` is [(x, y, text)], [(x, y, text, colour)] or
    [(x, y, text, colour, label_y)], the leader anchored at (x, y) and the label placed near `label_y` (default y);
    heights decluttered by `gap` (data units). With `leaders`, a label moved more than `min_shift` (default a third
    of the gap) from its line's end is joined to it by a thin leader."""
    ys = declutter([p[4] if len(p) > 4 and p[4] is not None else p[1] for p in points], gap)
    shifted = transforms.offset_copy(ax.transData, fig=ax.figure, x=dx + (4.0 if leaders else 0.0), y=0,
                                     units="points")
    limit = gap / 3 if min_shift is None else min_shift
    for p, y in zip(points, ys):
        x, y0, text = p[:3]
        c = p[3] if len(p) > 3 else colour
        lead = leaders and abs(y - y0) > limit
        ax.annotate(text, (x, y0), xytext=(x, y), textcoords=shifted, ha="left", va="center",
                    fontsize=ANNOT_PT, color=c, annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": c, "lw": MIN_LW, "shrinkA": 1.0, "shrinkB": 1.5}
                    if lead else None)


# offsets (points) a point label tries in turn: right-above first, then right-below, the same on the left, over
# and under the marker, then one line farther up or down on either side
LABEL_OFFSETS = ((3, 1), (3, -7), (-3, 1), (-3, -7), (0, 4), (0, -10), (3, 7), (-3, 7), (3, -13), (-3, -13))


# with leaders the labels may stand farther off, a thin line back to their marker
LEADER_OFFSETS = ((9, 7), (9, -9), (-9, 7), (-9, -9), (13, 13), (-13, 13), (13, -15), (-13, -15), (0, 13),
                  (0, -15), (18, 2), (-18, 2), (18, -12), (-18, -12))


def label_points(ax, points, fontsize=ANNOT_PT, colour=CONTEXT_INK, marker_radius=1.9, merge=True, leaders=False):
    """Label each (x, y, text) beside its marker at the first offset whose text stays inside the axes and clears
    the texts already on the axes and every other marker; with no clear offset, the first one. With `merge`,
    markers drawn on top of each other (centres closer than a diameter) share one label, their texts joined in
    numeric order; without it every point keeps its own label, and with `leaders` a label may stand farther off
    with a thin leader to its marker (for nearby but unequal observations).

    Call it after the axes' scales and limits are final: it draws the figure once so the constrained layout
    settles, then measures the labels in display space. (Ported from the family-step worker's 404d577.)
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    r = marker_radius * fig.dpi / 72
    marks = [ax.transData.transform((px, py)) for px, py, _ in points]
    taken = [t.get_window_extent(renderer) for t in ax.texts]
    groups, seen = [], set()
    for i in range(len(points)):
        if i not in seen:
            members = [j for j in range(len(points)) if j not in seen and
                       (j == i or (merge and math.dist(marks[i], marks[j]) < 2 * r))]
            seen.update(members)
            groups.append(members)
    arrow = {"arrowstyle": "-", "lw": MIN_LW, "color": colour, "shrinkA": 0.5, "shrinkB": 2.0}

    def clear(bb, members):
        inside = frame.x0 <= bb.x0 and bb.x1 <= frame.x1 and frame.y0 <= bb.y0 and bb.y1 <= frame.y1
        hits = any(bb.x0 - r < mx < bb.x1 + r and bb.y0 - r < my < bb.y1 + r
                   for j, (mx, my) in enumerate(marks) if j not in members)
        return inside and not hits and not any(bb.overlaps(o) for o in taken)

    placed = []
    if leaders and not merge:
        # clusters of markers closer than two diameters: their labels stack beside the cluster, one line each in
        # the markers' vertical order, each with a leader to its own marker
        clusters, done = [], set()
        for i in range(len(points)):
            if i in done:
                continue
            members, grow = {i}, [i]
            while grow:
                k = grow.pop()
                for j in range(len(points)):
                    if j not in members and math.dist(marks[k], marks[j]) < 4 * r:
                        members.add(j)
                        grow.append(j)
            done |= members
            if len(members) > 1:
                clusters.append(sorted(members, key=lambda j: -marks[j][1]))
        line = fontsize * 1.15 * fig.dpi / 72
        for members in clusters:
            cx = sum(marks[j][0] for j in members) / len(members)
            cy = sum(marks[j][1] for j in members) / len(members)
            side = -1 if cx > frame.x0 + 0.72 * frame.width else 1
            for n, j in enumerate(members):
                ty = cy + (len(members) - 1) / 2 * line - n * line
                tx = cx + side * 14 * fig.dpi / 72
                t = ax.annotate(points[j][2], tuple(points[j][:2]),
                                xytext=(tx / fig.bbox.width, ty / fig.bbox.height), textcoords="figure fraction",
                                fontsize=fontsize, color=colour, ha="left" if side > 0 else "right", va="center",
                                arrowprops=arrow, annotation_clip=False)
                taken.append(t.get_window_extent(renderer))
                placed.append(points[j][2])
        stacked = {j for c in clusters for j in c}
        groups = [g for g in groups if not set(g) & stacked]
    for members in groups:
        px, py = points[members[0]][:2]
        offsets = LABEL_OFFSETS + (LEADER_OFFSETS if leaders else ())
        text = ", ".join(sorted((points[j][2] for j in members), key=lambda t: (not t.isdigit(), len(t), t)))
        chosen = None
        for dx, dy in offsets:
            far = leaders and (dx, dy) in LEADER_OFFSETS
            t = ax.annotate(text, (px, py), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                            color=colour, ha="left" if dx > 0 else "right" if dx < 0 else "center",
                            va="center" if far else "baseline", arrowprops=arrow if far else None)
            if clear(t.get_window_extent(renderer), members):
                chosen = t
                break
            t.remove()
        if chosen is None:
            dx, dy = LABEL_OFFSETS[0]
            chosen = ax.annotate(text, (px, py), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                                 color=colour)
        taken.append(chosen.get_window_extent(renderer))
        placed.append(text)
    return placed


def copy_last_line(ax, orientation="h", label=True, where=1.0, align="right"):
    """The zero line: copy-last, 0.6 pt black, labelled with its printed name 'persistence' at its end once."""
    if orientation == "h":
        ax.axhline(0, color=BLACK, lw=REF_LW, zorder=1.5, gid="ref")
        if label:
            ax.text(where, 0, PERSISTENCE_LABEL, transform=ax.get_yaxis_transform(), ha=align, va="bottom",
                    fontsize=ANNOT_PT, color=INK, gid="decor")
    else:
        ax.axvline(0, color=BLACK, lw=REF_LW, zorder=1.5, gid="ref")
        if label:
            ax.text(0, where, " " + PERSISTENCE_LABEL, transform=ax.get_xaxis_transform(), ha="left",
                    va="top" if where >= 0.5 else "bottom", fontsize=ANNOT_PT, color=INK, gid="decor")


def training_line(ax, value, label=TRAINING_LABEL, orientation="h", where=0.0, align="left", colour=TRAINING_LINE,
                  band=None):
    """The training maps' in-distribution reference: its 95% episode interval `band` as a thin grey band, the point
    value as a 0.6 pt dashed grey line inside it, labelled at one end (no label when `label` is None)."""
    span, line = (ax.axhspan, ax.axhline) if orientation == "h" else (ax.axvspan, ax.axvline)
    if band is not None and all(b is not None for b in band):
        span(band[0], band[1], color=TRAINING_BAND, lw=0, zorder=1.3, gid="ref")
    line(value, color=colour, lw=REF_LW, ls=TRAINING_DASH, zorder=1.4, gid="ref")
    if not label:
        return
    if orientation == "h":
        top = band[1] if band is not None and band[1] is not None else value
        ax.text(where, top, (" " if align == "left" else "") + label + (" " if align == "right" else ""),
                transform=ax.get_yaxis_transform(), ha=align, va="bottom", fontsize=ANNOT_PT, color=TRAINING_LINE,
                gid="decor")
    else:
        ax.text(value, where, label, transform=ax.get_xaxis_transform(), ha=align, va="bottom", fontsize=ANNOT_PT,
                color=TRAINING_LINE, gid="decor")


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


def _handle_matches(handle, lines, colls, patches=()):
    """True when a legend handle matches a drawn series by colour and (for lines) line style and marker. `lines`
    include the reference lines, `patches` the drawn rectangles (bands), so a legend may key a reference."""
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
        # a band or bar (a collection or a drawn rectangle), or a colour-key swatch of a drawn line
        return (any(_same_colour(handle.get_facecolor(), fc) for c in colls for fc in list(c.get_facecolors())[:4])
                or any(_same_colour(handle.get_facecolor(), p.get_facecolor()) for p in patches)
                or any(_same_colour(handle.get_facecolor(), ln.get_color()) for ln in lines))
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
    # a legend may also key what the axes draw as references: dashed lines and grey bands (gid "ref")
    refs = [ln for ax in axes for ln in ax.lines if ln.get_gid() == "ref" and ln.get_visible()]
    patches = [p for ax in axes for p in ax.patches if p.get_visible()]
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None] + list(fig.legends)
    for leg in legends:
        for handle, text in zip(leg.legend_handles, leg.get_texts()):
            if not handle.get_visible():         # a text-only entry names the entries after it
                continue
            if not _handle_matches(handle, lines + refs, colls, patches):
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
