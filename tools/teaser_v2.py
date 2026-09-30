"""
Figure 1 candidates, second round: three layouts of the teaser that point the eye at the one contrast the paper
makes, zero-shot against adapted against ground truth on an unseen map, with the training maps as the reference.

    python tools/teaser_v2.py --restart-root results/teaser_restart --out-dir paper/figures/candidates
    python tools/teaser_v2.py ... --compare "today=paper/figures/fig_teaser_B.pdf"     # + the comparison sheet

**Input** (`--restart-root`, the steward's `results/teaser_restart/`): one directory per moment, `<window>_t<T>/`,
with `truth_raw/`, `unet_tuned/` and (on an unseen map) `adapter_tuned/` holding `tic_NN[_scene].png` for restart
tics 0 (the last of 32 ground-truth context tics) to 16, and a `manifest.json` with the per-tic scene PSNR of each
model row against the raw frame (`scene_psnr_vs_truth_raw`). Every number drawn is read from there.

**Rows** (`--rows`): `action:<unseen moment>:<training-map moment>` pairs, default today's Figure 1 (forward on map 7
episode 210 tic 217 with map 5 episode 6059 tic 239; attack on map 7 episode 288 tic 77 with map 3 episode 6037
tic 55). Variants B and C show the first row only.

**Encoding.** Zero-shot frames carry a dashed orange outline (the standard's spare orange, which no other figure
uses) and adapted frames a solid outline in the adapter's dark blue (`figstyle.BACKBONES["adapter"]`); the outline
sits in the gutter, outside the picture, so it reads on dark and light frames alike, and the dash survives
greyscale. Scene PSNR (dB) is a boxed tag in the frame's lower left corner, keyed once by a tag of the same style.

- **A** (`fig1_vA`): the two rows of today's figure without the context frames: training maps (prediction, ground
  truth) and the unseen map (zero-shot, adapted, ground truth), five frames a row; an arrow between the rows from
  zero-shot to adapted names what adaptation cost.
- **B** (`fig1_vB`): one action on the unseen map as a time strip (`--strip-tics`, default +1, +2, +4, +8 and +16
  tics after the context), rows zero-shot, adapted and ground truth, the one context frame feeding all three.
- **C** (`fig1_vC`): one moment at hero size, zero-shot, adapted and ground truth with the adaptation arrow between
  the first two, and the training-map prediction and ground truth of the same button, half size, as the reference.

Output per variant: `<out-dir>/<stem>.pdf`, a 300 dpi `<stem>.png` and `<stem>.json` (moments, tics, every number
drawn, the frame files, sizes and the command). `--compare label=path.pdf ...` also writes `fig1_compare.png`: those
PDFs and the candidates rendered at one dpi, so they stand at their true relative size.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import compose_teaser as ct  # noqa: E402

fs = ct.fs

ZERO_SHOT_COLOUR = fs.SPARE["orange"]
ADAPTED_COLOUR = fs.BACKBONES["adapter"].colour
ZERO_SHOT_DASH = (0, (2.4, 1.2))
BORDER_LW = 1.1                    # pt, drawn outside the picture
GUTTER = 3.5 / 72                  # room for two outlines and a hairline of white between neighbours
LINE = 8 / 72                      # one header line
TOP = 1.5 / 72                     # above the first header line, so no ascender touches the edge
HEAD = TOP + 2 * LINE + 2 / 72     # two header lines and the gap to the frames
PREVIEW_DPI = 300
ADAPTATION_COST = "8 episodes, 4k updates"
ADAPTATION_COST_LINES = "8 episodes,\n4k updates"   # where the label must be narrow
PSNR_KEY = "scene PSNR (dB)"
DEFAULT_ROWS = (("forward", "unseen_arena07_ep210_s2888_t217", "train_map05_ep6059_s4104_t239"),
                ("attack", "unseen_arena07_ep288_s456_t77", "train_map03_ep6037_s2824_t55"))
DEFAULT_HORIZON = 16
DEFAULT_STRIP = (1, 2, 4, 8, 16)
ROLE_STYLE = {"zero-shot": (ZERO_SHOT_COLOUR, ZERO_SHOT_DASH), "adapted": (ADAPTED_COLOUR, "solid")}


# ---------------------------------------------------------------------------------------------
# reading the restart export
# ---------------------------------------------------------------------------------------------

def moment(restart_root, name):
    """One restarted moment `<window>_t<T>`: its directory, window, place name and per-row scene PSNR by tic."""
    d = os.path.join(restart_root, name)
    if not os.path.isdir(d):
        raise SystemExit(f"the restart export has no {d}")
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    window = name.rsplit("_t", 1)[0]
    return {"name": name, "dir": d, "window": window, "place": ct.place_name(window),
            "psnr": {k: ct.per_tic_scene(man, ct.ROWS[k]) for k in ("model", "adapted")},
            "held_all_16": man.get("held_all_16")}


def frame_of(m, row, tic):
    """(pixels, file) of one frame of a moment: `row` is "model", "adapted" or "truth"."""
    row_dir = os.path.join(m["dir"], ct.ROWS[row])
    return ct.frame(row_dir, tic), os.path.relpath(ct.frame_path(row_dir, tic), os.path.dirname(m["dir"]))


def psnr_of(m, row, tic):
    """The scene PSNR the export records for one model frame, or stop: a drawn number must come from the export."""
    v = m["psnr"][row].get(tic)
    if v is None:
        raise SystemExit(f"{m['name']}: no scene PSNR for {ct.ROWS[row]} at tic {tic}")
    return round(v, 3)


# ---------------------------------------------------------------------------------------------
# drawing (inches from the top left, as in compose_teaser)
# ---------------------------------------------------------------------------------------------

class Sheet:
    """A figure of `width` x `height` inches addressed from the top left, with the frame, outline, tag and arrow
    primitives the three variants share, and the record of what was drawn."""

    def __init__(self, width, height):
        fs.style()
        self.w, self.h = width, height
        self.fig = ct._figure(width, height)
        self.frames = []

    def fx(self, x):
        return x / self.w

    def fy(self, y):
        return 1 - y / self.h

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", fs.ANNOT_PT)
        return ct._text(self.fig, self.w, self.h, x, y, s, **kw)

    def frame(self, x, y, fw, fh, m, row, tic, role, psnr=None):
        """One frame with its outline (zero-shot, adapted) and its scene PSNR tag; recorded for the sidecar."""
        img, path = frame_of(m, row, tic)
        ct._place(self.fig, self.w, self.h, x, y, fw, fh, img)
        if role in ROLE_STYLE:
            self.outline(x, y, fw, fh, *ROLE_STYLE[role])
        if psnr is not None:
            self.tag(x + 1.5 / 72, y + fh - 1.5 / 72, f"{psnr:.1f}", ROLE_STYLE.get(role, (fs.CONTEXT_INK,))[0])
        self.frames.append({"moment": m["name"], "role": role, "tic": tic, "file": path, "scene_psnr": psnr,
                            "at_in": [round(x, 3), round(y, 3)], "size_in": [round(fw, 3), round(fh, 3)]})

    def outline(self, x, y, fw, fh, colour, dash):
        """A rectangle just outside the picture, so the frame's own edge pixels stay visible."""
        from matplotlib.patches import Rectangle
        o = BORDER_LW / 2 / 72
        self.fig.add_artist(Rectangle((self.fx(x - o), self.fy(y + fh + o)), (fw + 2 * o) / self.w,
                                      (fh + 2 * o) / self.h, transform=self.fig.transFigure, fill=False,
                                      edgecolor=colour, lw=BORDER_LW, linestyle=dash, joinstyle="miter"))

    def tag(self, x, y, s, colour, va="bottom", ha="left"):
        """A boxed number: white ground, thin outline in the frame's colour, black text."""
        return self.text(x, y, s, ha=ha, va=va, color=fs.INK,
                         bbox={"boxstyle": "square,pad=0.18", "fc": "white", "ec": colour, "lw": 0.6})

    def arrow(self, x0, y0, x1, y1, colour=fs.INK, lw=0.7, head=True):
        self.fig.add_artist(_arrow(self, x0, y0, x1, y1, colour, lw, head))


def _arrow(sheet, x0, y0, x1, y1, colour, lw, head):
    from matplotlib.patches import FancyArrowPatch
    return FancyArrowPatch((sheet.fx(x0), sheet.fy(y0)), (sheet.fx(x1), sheet.fy(y1)),
                           transform=sheet.fig.transFigure, arrowstyle="-|>,head_length=2.6,head_width=1.3"
                           if head else "-", color=colour, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=1)


def save(sheet, out_dir, stem, record):
    """Refuse a degenerate figure, write the PDF (dateless), a 300 dpi PNG and the sidecar; returns the paths."""
    fs.refuse_degenerate(sheet.fig, stem)
    os.makedirs(out_dir, exist_ok=True)
    pdf, png, side = (os.path.join(out_dir, f"{stem}.{e}") for e in ("pdf", "png", "json"))
    sheet.fig.savefig(pdf, metadata={"CreationDate": None, "Creator": None, "Producer": None})
    sheet.fig.savefig(png, dpi=PREVIEW_DPI, metadata={"Software": None})
    import matplotlib.pyplot as plt
    plt.close(sheet.fig)
    record = {**record, "size_in": [sheet.w, round(sheet.h, 3)], "frames": sheet.frames,
              "command": _command(), "git_head": _git_head(),
              "encoding": {"zero-shot": {"colour": ZERO_SHOT_COLOUR, "outline": "dashed"},
                           "adapted": {"colour": ADAPTED_COLOUR, "outline": "solid"},
                           "tags": "scene PSNR (dB) against the raw ground-truth frame, scene rows 0 to 207"}}
    with open(side, "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    return [pdf, png, side]


def _command():
    return "python3 " + " ".join(shlex.quote(a) for a in [os.path.relpath(sys.argv[0], REPO)] + sys.argv[1:])


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def aspect_of(m):
    img, _ = frame_of(m, "truth", 0)
    return img.shape[0] / img.shape[1]


def psnr_key(sheet, x, y, ha="right"):
    """The key to the tags: one tag of the same style naming the metric."""
    sheet.tag(x, y, PSNR_KEY, fs.CONTEXT_INK, va="top", ha=ha)


# ---------------------------------------------------------------------------------------------
# the three variants
# ---------------------------------------------------------------------------------------------

def variant_a(restart_root, rows, out_dir, horizon=DEFAULT_HORIZON, height=1.5, stem="fig1_vA"):
    """Two rows (one per action), five frames each at +`horizon` tics: training maps (prediction, ground truth),
    then the unseen map (zero-shot, adapted, ground truth); the adaptation arrow runs between the rows from the
    zero-shot column to the adapted column. The context frames of today's figure are dropped."""
    ms = [(a, moment(restart_root, u), moment(restart_root, h)) for a, u, h in rows]
    aspect = aspect_of(ms[0][1])
    n = len(ms)
    head = HEAD
    row_gap = 11 / 72
    group_gap = 12 / 72
    label_w = max([ct.text_width(a, style="italic") for a, _, _ in ms] + [ct.text_width(PSNR_KEY)]) + 7 / 72
    fh = (height - head - (n - 1) * row_gap - 0.03) / n
    fw = fh / aspect
    used = label_w + 5 * fw + 3 * GUTTER + group_gap
    if used > fs.TEXT_WIDTH - 0.02:
        raise SystemExit(f"{stem}: {used:.2f} in wide at {height} in tall; lower the height")
    x0 = (fs.TEXT_WIDTH - used) / 2
    xs = [x0 + label_w + i * (fw + GUTTER) + (group_gap - GUTTER if i >= 2 else 0) for i in range(5)]
    s = Sheet(fs.TEXT_WIDTH, height)
    home_places = {h["place"] for _, _, h in ms}
    home_title = ("training maps" if len(home_places) > 1 else home_places.pop()) + " (in distribution)"
    away_title = re.sub(r"^unseen map (\d+)$", r"unseen map (map \1)", ms[0][1]["place"])
    s.text((xs[0] + xs[1] + fw) / 2, TOP, home_title, ha="center", va="top")
    s.text((xs[2] + xs[4] + fw) / 2, TOP, away_title, ha="center", va="top")
    for x, name in zip(xs, ("prediction", ct.TRUTH_LABEL, "zero-shot", "adapted", ct.TRUTH_LABEL)):
        s.text(x + fw / 2, TOP + LINE, name, ha="center", va="top")
    psnr_key(s, xs[0] - 7 / 72, TOP + LINE + 1 / 72)
    record = {"variant": "A", "horizon": horizon, "rows": []}
    for r, (action, u, h) in enumerate(ms):
        y = head + r * (fh + row_gap)
        nums = {"prediction": psnr_of(h, "model", horizon), "zero-shot": psnr_of(u, "model", horizon),
                "adapted": psnr_of(u, "adapted", horizon)}
        s.frame(xs[0], y, fw, fh, h, "model", horizon, "prediction", nums["prediction"])
        s.frame(xs[1], y, fw, fh, h, "truth", horizon, "ground truth")
        s.frame(xs[2], y, fw, fh, u, "model", horizon, "zero-shot", nums["zero-shot"])
        s.frame(xs[3], y, fw, fh, u, "adapted", horizon, "adapted", nums["adapted"])
        s.frame(xs[4], y, fw, fh, u, "truth", horizon, "ground truth")
        s.text(xs[0] - 7 / 72, y + fh / 2, action, ha="right", va="center", style="italic")
        record["rows"].append({"action": action, "unseen": u["name"], "training": h["name"], "scene_psnr": nums,
                               "held_all_16": {"unseen": u["held_all_16"], "training": h["held_all_16"]}})
    # one arrow for both rows: it sits between them, from the zero-shot column's centre to the adapted column's
    ya = head + fh + row_gap / 2
    mid = (xs[2] + fw + xs[3]) / 2
    half = ct.text_width(ADAPTATION_COST) / 2 + 2 / 72
    s.arrow(xs[2] + fw * 0.25, ya, mid - half, ya, head=False)
    s.arrow(mid + half, ya, xs[3] + fw * 0.75, ya)
    s.text(mid, ya, ADAPTATION_COST, ha="center", va="center_baseline")
    record["frame_in"] = [round(fw, 3), round(fh, 3)]
    return save(s, out_dir, stem, record), record


def variant_d(restart_root, rows, out_dir, horizon=DEFAULT_HORIZON, height=1.55, stem="fig1_vD", numbers=True):
    """Two rows (one per action), five frames each at +`horizon` tics: training map (prediction, ground truth), then
    the unseen map (zero-shot, adapted, ground truth). The pictures take the space (Rohan, Sep 29 night): no
    outlines, boxes or arrows, two small header lines, a small italic row label, and the scene PSNR as small text
    inside each predicted frame's corner (`numbers`). The frames are as wide as the text width allows at `height`."""
    import matplotlib.patheffects as pe
    ms = [(a, moment(restart_root, u), moment(restart_root, h)) for a, u, h in rows]
    aspect = aspect_of(ms[0][1])
    n = len(ms)
    gutter, group_gap, row_gap = 2 / 72, 8 / 72, 4 / 72
    head = TOP + 2 * LINE + 1 / 72
    label_w = max(ct.text_width(a, style="italic") for a, _, _ in ms) + 5 / 72
    fw = (fs.TEXT_WIDTH - label_w - 3 * gutter - group_gap) / 5
    fh = fw * aspect
    if head + n * fh + (n - 1) * row_gap > height:            # the height binds: shrink the frames to it
        fh = (height - head - (n - 1) * row_gap) / n
        fw = fh / aspect
    height = head + n * fh + (n - 1) * row_gap + 0.01
    x0 = (fs.TEXT_WIDTH - (label_w + 5 * fw + 3 * gutter + group_gap)) / 2
    xs = [x0 + label_w + i * (fw + gutter) + (group_gap - gutter if i >= 2 else 0) for i in range(5)]
    s = Sheet(fs.TEXT_WIDTH, height)
    home_places = {h["place"] for _, _, h in ms}
    home_title = ("training maps" if len(home_places) > 1 else home_places.pop()) + " (in distribution)"
    away_title = re.sub(r"^unseen map (\d+)$", r"unseen map (map \1)", ms[0][1]["place"])
    s.text((xs[0] + xs[1] + fw) / 2, TOP, "(a) " + home_title, ha="center", va="top")
    s.text((xs[2] + xs[4] + fw) / 2, TOP, "(b) " + away_title, ha="center", va="top")
    for x, name in zip(xs, ("prediction", ct.TRUTH_LABEL, "zero-shot", "adapted", ct.TRUTH_LABEL)):
        s.text(x + fw / 2, TOP + LINE, name, ha="center", va="top")
    record = {"variant": "D", "horizon": horizon, "rows": []}
    for r, (action, u, h) in enumerate(ms):
        y = head + r * (fh + row_gap)
        nums = {"prediction": psnr_of(h, "model", horizon), "zero-shot": psnr_of(u, "model", horizon),
                "adapted": psnr_of(u, "adapted", horizon)}
        for x, m, row, role in ((xs[0], h, "model", "prediction"), (xs[1], h, "truth", "ground truth"),
                                (xs[2], u, "model", "zero-shot"), (xs[3], u, "adapted", "adapted"),
                                (xs[4], u, "truth", "ground truth")):
            img, path = frame_of(m, row, horizon)
            ct._place(s.fig, s.w, s.h, x, y, fw, fh, img)
            v = nums.get(role)
            if numbers and v is not None:
                s.fig.text(s.fx(x + 2.5 / 72), s.fy(y + fh - 2 / 72), f"{v:.1f} dB", ha="left", va="bottom",
                           color="white", fontsize=fs.ANNOT_PT - 0.5, transform=s.fig.transFigure,
                           path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])
            s.frames.append({"moment": m["name"], "role": role, "tic": horizon, "file": path, "scene_psnr": v,
                             "at_in": [round(x, 3), round(y, 3)], "size_in": [round(fw, 3), round(fh, 3)]})
        s.text(xs[0] - 4 / 72, y + fh / 2, action, ha="right", va="center", style="italic")
        record["rows"].append({"action": action, "unseen": u["name"], "training": h["name"], "scene_psnr": nums,
                               "held_all_16": {"unseen": u["held_all_16"], "training": h["held_all_16"]}})
    record["frame_in"] = [round(fw, 3), round(fh, 3)]
    return save(s, out_dir, stem, record), record


def variant_b(restart_root, row, out_dir, strip=DEFAULT_STRIP, height=1.8, stem="fig1_vB"):
    """One action on the unseen map as a time strip: rows zero-shot, adapted and ground truth at `strip` tics after
    the context, the one ground-truth context frame at the left feeding all three rows. No training-map block."""
    action, uname, _ = row
    u = moment(restart_root, uname)
    aspect = aspect_of(u)
    head = HEAD
    labels = ("zero-shot", "adapted", ct.TRUTH_LABEL)
    # context | fan of arrows | row labels | strip: each label sits against its own row
    label_w = max(ct.text_width(t) for t in labels + tuple(ADAPTATION_COST_LINES.split("\n"))) + 5 / 72
    fan = 14 / 72
    fh = (height - head - 2 * GUTTER - 0.03) / 3
    fw = fh / aspect
    k = len(strip)
    used = fw + fan + label_w + k * fw + (k - 1) * GUTTER
    if used > fs.TEXT_WIDTH - 0.02:
        raise SystemExit(f"{stem}: {used:.2f} in wide at {height} in tall; lower the height or drop a tic")
    xc = (fs.TEXT_WIDTH - used) / 2
    xl = xc + fw + fan
    xs = [xl + label_w + i * (fw + GUTTER) for i in range(k)]
    s = Sheet(fs.TEXT_WIDTH, height)
    title = re.sub(r"^unseen map (\d+)$", r"unseen map (map \1)", u["place"])
    s.text((xs[0] + xs[-1] + fw) / 2, TOP, title, ha="center", va="top")
    psnr_key(s, xc, TOP + 1 / 72, ha="left")
    for x, t in zip(xs, strip):
        s.text(x + fw / 2, TOP + LINE, f"+{t} tic" + ("" if t == 1 else "s"), ha="center", va="top")
    ys = [head + r * (fh + GUTTER) for r in range(3)]
    yc = head + (3 * fh + 2 * GUTTER - fh) / 2
    s.frame(xc, yc, fw, fh, u, "truth", 0, "context")
    s.text(xc + fw / 2, yc - 1.5 / 72, "context", ha="center", va="bottom")
    s.text(xc + fw / 2, yc + fh + 1.5 / 72, action, ha="center", va="top", style="italic")
    for y in ys:       # the one context frame starts all three rows
        s.arrow(xc + fw + 2 / 72, yc + fh / 2, xl - 3 / 72, y + fh / 2, colour=fs.CONTEXT_INK, lw=0.5)
    record = {"variant": "B", "action": action, "unseen": u["name"], "tics": list(strip), "scene_psnr": {},
              "held_all_16": u["held_all_16"]}
    for (row, role), y in zip((("model", "zero-shot"), ("adapted", "adapted"), ("truth", "ground truth")), ys):
        nums = []
        for x, t in zip(xs, strip):
            v = psnr_of(u, row, t) if row != "truth" else None
            s.frame(x, y, fw, fh, u, row, t, role, v)
            nums.append(v)
        if row != "truth":
            record["scene_psnr"][role] = dict(zip((f"+{t}" for t in strip), nums))
    s.text(xl, ys[0] + fh / 2, "zero-shot", ha="left", va="center")
    s.text(xl, ys[1] + fh / 2, "adapted", ha="left", va="bottom")
    s.text(xl, ys[1] + fh / 2 + 1 / 72, ADAPTATION_COST_LINES, ha="left", va="top", color=fs.CONTEXT_INK,
           linespacing=1.0)
    s.text(xl, ys[2] + fh / 2, ct.TRUTH_LABEL, ha="left", va="center")
    record["frame_in"] = [round(fw, 3), round(fh, 3)]
    return save(s, out_dir, stem, record), record


def variant_c(restart_root, row, out_dir, horizon=DEFAULT_HORIZON, stem="fig1_vC"):
    """One moment at hero size: zero-shot, adapted and ground truth on the unseen map at +`horizon` tics, the
    adaptation arrow in the gutter between zero-shot and adapted, and at the left the same button on a training
    map at half size (prediction over ground truth) as the in-distribution reference. The height follows from the
    width."""
    action, uname, hname = row
    u, h = moment(restart_root, uname), moment(restart_root, hname)
    aspect = aspect_of(u)
    head = HEAD
    home_title = f"{h['place']} (in distribution)"
    label_w = max(ct.text_width("prediction"), ct.text_width(ct.TRUTH_LABEL)) + 5 / 72
    group_gap = 12 / 72
    arrow_gap = max(ct.text_width(t) for t in ADAPTATION_COST_LINES.split("\n")) + 10 / 72
    # widths: label + mini + group gap + 3 heroes + arrow gap + gutter, the mini column half a hero less a gutter
    fixed = label_w + group_gap + arrow_gap + GUTTER + 0.04
    fw = (fs.TEXT_WIDTH - fixed + GUTTER / (2 * aspect)) / 3.5
    fh = fw * aspect
    mh = (fh - GUTTER) / 2
    mw = mh / aspect
    used = label_w + mw + group_gap + 3 * fw + arrow_gap + GUTTER
    x0 = (fs.TEXT_WIDTH - used) / 2
    xm = x0 + label_w
    xz = xm + mw + group_gap
    xa = xz + fw + arrow_gap
    xg = xa + fw + GUTTER
    height = head + fh + 0.03
    s = Sheet(fs.TEXT_WIDTH, height)
    away_title = re.sub(r"^unseen map (\d+)$", r"unseen map (map \1)", u["place"])
    left = max(0.0, xm + mw / 2 - ct.text_width(home_title) / 2)
    s.text(left, TOP, home_title, ha="left", va="top")
    psnr_key(s, left, TOP + LINE + 1 / 72, ha="left")
    s.text((xz + xg + fw) / 2, TOP, away_title, ha="center", va="top")
    for x, name in ((xz, "zero-shot"), (xa, "adapted"), (xg, ct.TRUTH_LABEL)):
        s.text(x + fw / 2, TOP + LINE, name, ha="center", va="top")
    nums = {"prediction": psnr_of(h, "model", horizon), "zero-shot": psnr_of(u, "model", horizon),
            "adapted": psnr_of(u, "adapted", horizon)}
    s.frame(xm, head, mw, mh, h, "model", horizon, "prediction", nums["prediction"])
    s.frame(xm, head + mh + GUTTER, mw, mh, h, "truth", horizon, "ground truth")
    s.text(xm - 5 / 72, head + mh / 2, "prediction", ha="right", va="center")
    s.text(xm - 5 / 72, head + mh + GUTTER + mh / 2, ct.TRUTH_LABEL, ha="right", va="center")
    s.frame(xz, head, fw, fh, u, "model", horizon, "zero-shot", nums["zero-shot"])
    s.frame(xa, head, fw, fh, u, "adapted", horizon, "adapted", nums["adapted"])
    s.frame(xg, head, fw, fh, u, "truth", horizon, "ground truth")
    ya = head + fh / 2
    s.arrow(xz + fw + 3 / 72, ya, xa - 3 / 72, ya)
    s.text((xz + fw + xa) / 2, ya - 2 / 72, ADAPTATION_COST_LINES, ha="center", va="bottom", linespacing=1.0)
    record = {"variant": "C", "action": action, "horizon": horizon, "unseen": u["name"], "training": h["name"],
              "scene_psnr": nums, "held_all_16": {"unseen": u["held_all_16"], "training": h["held_all_16"]},
              "frame_in": [round(fw, 3), round(fh, 3)], "reference_frame_in": [round(mw, 3), round(mh, 3)]}
    return save(s, out_dir, stem, record), record


# ---------------------------------------------------------------------------------------------
# the comparison sheet
# ---------------------------------------------------------------------------------------------

def compare(entries, path, dpi=200):
    """Render each (label, pdf) at one `dpi` and stack them with their labels and printed sizes, left-aligned on a
    5.5 in ruler, so every figure stands at its true size relative to the others."""
    from matplotlib import font_manager as fm
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family=fs.font_family())), int(dpi * 0.1))
    pad, text_h = int(dpi * 0.12), int(dpi * 0.16)
    shots = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (label, pdf) in enumerate(entries):
            out = os.path.join(tmp, f"p{i}")
            subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", pdf, out], check=True)
            img = Image.open(out + ".png").convert("RGB")
            shots.append((f"{label}   ({img.width / dpi:.2f} x {img.height / dpi:.2f} in)", img))
    width = int(fs.TEXT_WIDTH * dpi) + 2 * pad
    height = sum(text_h + img.height + pad for _, img in shots) + pad
    sheet = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(sheet)
    y = pad
    for label, img in shots:
        d.text((pad, y), label, font=font, fill=(0, 0, 0))
        y += text_h
        d.rectangle([pad - 1, y - 1, pad + int(fs.TEXT_WIDTH * dpi), y + img.height], outline=(200, 200, 200))
        sheet.paste(img, (pad, y))
        y += img.height + pad
    sheet.save(path, optimize=True)
    return path


# ---------------------------------------------------------------------------------------------

def parse_rows(spec):
    """`action:<unseen moment>:<training moment>,...` into row triples."""
    return tuple(tuple(r.split(":")) for r in spec.split(",")) if spec else DEFAULT_ROWS


def main(argv=None):
    p = argparse.ArgumentParser(description="Figure 1 candidates A, B and C from the restart export.")
    p.add_argument("--restart-root", default=os.path.join(REPO, "results", "teaser_restart"))
    p.add_argument("--out-dir", default=os.path.join(REPO, "paper", "figures", "candidates"))
    p.add_argument("--variant", choices=("A", "B", "C", "D", "all"), default="all")
    p.add_argument("--rows", default=None, help="action:<unseen moment>:<training moment>,... (default: today's)")
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="A and C: tics after the context")
    p.add_argument("--strip-tics", default=",".join(map(str, DEFAULT_STRIP)), help="B: the strip's tics")
    p.add_argument("--height-a", type=float, default=1.5)
    p.add_argument("--height-b", type=float, default=1.8)
    p.add_argument("--height-d", type=float, default=1.55)
    p.add_argument("--no-numbers", action="store_true", help="D: no PSNR text inside the frames")
    p.add_argument("--compare", nargs="*", default=None,
                   help="label=path.pdf entries drawn above the candidates on fig1_compare.png")
    a = p.parse_args(argv)
    rows = parse_rows(a.rows)
    written, made = [], []
    if a.variant in ("A", "all"):
        written += variant_a(a.restart_root, rows, a.out_dir, a.horizon, a.height_a)[0]
        made.append(("A: two rows, no context, adaptation arrow", written[-3]))
    if a.variant in ("B", "all"):
        strip = tuple(int(t) for t in a.strip_tics.split(","))
        written += variant_b(a.restart_root, rows[0], a.out_dir, strip, a.height_b)[0]
        made.append(("B: time strip on the unseen map", written[-3]))
    if a.variant in ("C", "all"):
        written += variant_c(a.restart_root, rows[0], a.out_dir, a.horizon)[0]
    if a.variant in ("D", "all"):
        written += variant_d(a.restart_root, rows, a.out_dir, a.horizon, a.height_d, numbers=not a.no_numbers)[0]
        made.append(("C: one moment at hero size", written[-3]))
    if a.compare is not None:
        extra = [tuple(e.split("=", 1)) for e in a.compare]
        written.append(compare(extra + made, os.path.join(a.out_dir, "fig1_compare.png")))
    for path in written:
        print("wrote", os.path.relpath(path, REPO) if path.startswith(REPO) else path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
