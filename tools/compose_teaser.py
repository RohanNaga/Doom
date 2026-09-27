"""
The teaser: one rollout on an unseen arena at the moments the control changes, zero-shot against after adaptation,
with the ground-truth frames beneath and the same action on a training map for reference (GameNGen's Figure 1 in
role).

    python tools/compose_teaser.py                                # layout A from results/teaser, best-ranked window
    python tools/compose_teaser.py --layout B                     # three or four actions, +8 tics, side by side
    python tools/compose_teaser.py --window unseen_arena07_ep41_s269 --home train_map02_ep6000_s837

**Input** (`--root`, the steward's `results/teaser/`): one directory per window, `<map>_ep<E>_s<S>/`, holding
`truth_raw/`, `unet_tuned/` and (on an unseen arena) `adapter_tuned/`, each with `tic_NNN.png` for tic 0 (the last
context frame) onward, and a `manifest.json` in the window directory or at the root (a `windows` list or dict)
with per window its `map`, `episode`, `start`, the executed control per tic (`controls`: a list over tics of
button names, or of strings joined by "+"), per-tic scene PSNR per row, and a `richness` score. Frames are cropped
to the scene rows 0 to 207 unless a `tic_NNN_scene.png` is beside them.

**Layout A** (`fig_teaser`, full width): columns are the moments of the unseen-arena rollout where the executed
control changes (`--moments` of them, spread over the rollout; a manifest `moments` list wins), each headed by the
executed action in italics and its tic; rows are "zero-shot" (the U-Net) and the adapted model, with the ground-truth
frames
beneath as a strip about a third of the frame height; a small first column shows the same model on a training map
at a moment with the first column's action, its truth beneath, as the in-distribution reference.

**Layout B** (`fig_teaser_actions`): one row per action (up to `--actions` distinct actions of the unseen
rollout, or the ones `--pick` names, in that order): the context frame at the tic the action starts, then the frame
8 tics later from the model on a training map (the same action), zero-shot, adapted, and the truth. `--height` fixes
the page height for a slot: the frames shrink to fit it and the block is centred across the width. `--no-reference`
drops the training-map column (and needs no training-map window).

**Layout C** (`layout_c`, driven by `tools/teaser_contact_sheet.py`): one row per chosen action, seven frames:
the in-domain block (the same button starting on a training map: context, U-Net +8, ground truth +8) and the unseen
block (context, zero-shot, adapted and ground truth +8), scene PSNR under every model frame; with `--restart-root`
the +8 frames come from rollouts restarted at each moment, so they are 8 tics after the context shown.

Row labels carry at most one number (`--adapted-label`, default "after 8 episodes"); the budget and GPU-hours go in
the caption. Output: `<out-dir>/<stem>.pdf/.png` through `figstyle.save` and `<stem>.json` with the windows, tics,
actions and per-tic scene PSNR drawn.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "paper"))
import figstyle as fs  # noqa: E402

SCENE_ROWS = 208
WINDOW_RE = re.compile(r"^(?P<map>.+?)_ep(?P<episode>\d+)_s(?P<start>\d+)$")
UNSEEN_HINTS = ("unseen", "arena")
GUTTER = 2 / 72
ROWS = {"model": "unet_tuned", "adapted": "adapter_tuned", "truth": "truth_raw"}
DEFAULT_ADAPTED_LABEL = "after 8 episodes"
TRUTH_LABEL = "ground truth"          # the field's term (Rohan, 2026-09-27)
HORIZON_B = 8


# ---------------------------------------------------------------------------------------------
# reading the export
# ---------------------------------------------------------------------------------------------

def windows(root):
    """{name: directory} of every window directory `<map>_ep<E>_s<S>` under `root`."""
    return {d: os.path.join(root, d) for d in sorted(os.listdir(root))
            if WINDOW_RE.match(d) and os.path.isdir(os.path.join(root, d))}


def is_unseen(name):
    m = WINDOW_RE.match(name)
    return bool(m) and any(h in m["map"] for h in UNSEEN_HINTS) and not m["map"].startswith("train")


def manifest_of(root, wdir):
    """A window's manifest entry: `<window>/manifest.json`, else its entry in `<root>/manifest.json`, else {}."""
    p = os.path.join(wdir, "manifest.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    p = os.path.join(root, "manifest.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        js = json.load(f)
    name = os.path.basename(wdir)
    ws = js.get("windows", js) if isinstance(js, dict) else js
    if isinstance(ws, dict):
        return ws.get(name, {})
    return next((e for e in ws if isinstance(e, dict) and name in (
        e.get("window"), e.get("name"), os.path.basename(str(e.get("dir", "")).rstrip("/")))), {})


def controls_of(manifest):
    """[frozenset of button names] per tic from the manifest's executed controls: a list over tics (of name lists
    or "+"-joined strings), or the steward's `per_tic` list of {tic, control} records (tic 0 then empty)."""
    per = manifest.get("per_tic")
    if isinstance(per, list) and per and isinstance(per[0], dict) and "control" in per[0]:
        n = max(int(r["tic"]) for r in per) + 1
        raw = [[] for _ in range(n)]
        for r in per:
            raw[int(r["tic"])] = r["control"] or []
    else:
        raw = next((manifest[k] for k in ("controls", "executed", "buttons", "actions") if k in manifest), [])
    out = []
    for c in raw:
        if isinstance(c, dict):
            c = c.get("buttons", c.get("names", []))
        names = c.split("+") if isinstance(c, str) else list(c)
        out.append(frozenset(n.strip() for n in names if str(n).strip() and str(n).strip().lower() != "none"))
    return out


HIDDEN_BUTTONS = ("speed",)       # held nearly always; printing it on every column adds nothing
PRIORITY = ("attack", "turn left", "turn right", "forward", "move forward", "strafe left", "strafe right",
            "move left", "move right", "back", "move backward")


def button_names(buttons):
    """A control's printed button names: lowercase words, the always-held run button left out, in priority order."""
    names = {b.lower().replace("_", " ") for b in buttons} - set(HIDDEN_BUTTONS)
    return sorted(names, key=lambda n: (PRIORITY.index(n) if n in PRIORITY else len(PRIORITY), n))


def action_text(buttons, sep=" + "):
    """A control as printed ("no input" when nothing but the run button is held)."""
    names = button_names(buttons)
    return sep.join(names) if names else "no input"


def last_tic(row_dir):
    tics = [int(m.group(1)) for f in os.listdir(row_dir) for m in [re.match(r"tic_(\d+)\.png$", f)] if m]
    return max(tics) if tics else -1


def frame_path(row_dir, tic):
    """The file `frame` reads for one tic: the `_scene` crop, else the full frame (3- or 2-digit tic names)."""
    for digits in (3, 2):
        base = os.path.join(row_dir, f"tic_{tic:0{digits}d}")
        for path in (base + "_scene.png", base + ".png"):
            if os.path.exists(path):
                return path
    raise SystemExit(f"missing frame {os.path.join(row_dir, f'tic_{tic:03d}.png')}")


def place_name(window):
    """'training map 2' or 'unseen arena 7' from a window name."""
    m = WINDOW_RE.match(window)
    digits = re.search(r"(\d+)$", m["map"]) if m else None
    number = str(int(digits.group(1))) if digits else ""
    return (f"unseen arena {number}" if is_unseen(window) else f"training map {number}").strip()


def frame(row_dir, tic):
    """One frame, scene rows only: the `_scene` file, else the full frame cropped to rows 0 to 207."""
    path = frame_path(row_dir, tic)
    img = np.asarray(Image.open(path).convert("RGB"))
    return img if path.endswith("_scene.png") else img[:SCENE_ROWS]


def change_moments(controls, first, last):
    """Tics in [first, last] where the executed control differs from the tic before."""
    return [t for t in range(max(first, 1), min(last, len(controls) - 1) + 1) if controls[t] != controls[t - 1]]


def spread(items, n):
    """Up to n items spread evenly over the list, first and last included."""
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idx))]


def per_tic_scene(manifest, row):
    """{tic: scene PSNR} of one row, from the shapes the steward writes (a `per_tic` list of records with
    `<model>_scene_psnr`, the restart export's `scene_psnr_vs_truth_raw` series per row, or per-row series)."""
    series = manifest.get("scene_psnr_vs_truth_raw")
    if isinstance(series, dict) and isinstance(series.get(row), list):
        return {i: float(x) for i, x in enumerate(series[row]) if isinstance(x, (int, float))}
    per = manifest.get("per_tic")
    if isinstance(per, list) and per and isinstance(per[0], dict):
        key = f"{row.split('_')[0]}_scene_psnr"
        return {int(r["tic"]): float(r[key]) for r in per if isinstance(r.get(key), (int, float))}
    for container in (manifest.get("per_tic"), manifest.get("rows"), manifest):
        if isinstance(container, dict) and isinstance(container.get(row), dict):
            v = container[row].get("scene_psnr")
            if isinstance(v, list):
                return {i: float(x) for i, x in enumerate(v) if isinstance(x, (int, float))}
            if isinstance(v, dict):
                return {int(re.search(r"(\d+)$", str(k)).group(1)): float(x) for k, x in v.items()}
    return {}


def choose(root, window=None, home=None):
    """(unseen window name, home window name): the named ones, else the best-ranked by `richness`."""
    ws = windows(root)
    if not ws:
        raise SystemExit(f"no window directories <map>_ep<E>_s<S> under {root}")

    def rank(name):
        m = manifest_of(root, ws[name])
        r = m.get("richness")
        if isinstance(r, (int, float)):
            return r
        rank_sum = (m.get("score") or {}).get("rank_sum")        # the steward's ranking: lower is better
        return -rank_sum if isinstance(rank_sum, (int, float)) else float("-inf")

    unseen = window or max((n for n in ws if is_unseen(n)), key=rank, default=None)
    if unseen is None or unseen not in ws:
        raise SystemExit(f"no unseen-arena window under {root} (named {window!r})")
    home_name = home or max((n for n in ws if not is_unseen(n)), key=rank, default=None)
    if home is not None and home not in ws:
        raise SystemExit(f"no window {home!r} under {root}")
    return unseen, home_name


def home_moment(home_ctrl, action, first, last):
    """The first tic of the home rollout where `action` starts (or is held), else its first control change."""
    for t in range(max(first, 1), min(last, len(home_ctrl) - 1) + 1):
        if home_ctrl[t] == action and home_ctrl[t - 1] != action:
            return t
    for t in range(first, min(last, len(home_ctrl) - 1) + 1):
        if home_ctrl[t] == action:
            return t
    changes = change_moments(home_ctrl, first, last)
    return changes[0] if changes else first


# ---------------------------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------------------------

def text_width(text, size=fs.ANNOT_PT, **kw):
    """The printed width (in) of one line of text at `size` pt."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(2, 1))
    t = fig.text(0, 0, text, fontsize=size, **kw)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    plt.close(fig)
    return w


def _figure(width, height):
    import matplotlib.pyplot as plt
    return plt.figure(figsize=(width, height))


def _place(fig, width, height, x, y, w, h, img):
    ax = fig.add_axes([x / width, 1 - (y + h) / height, w / width, h / height])
    ax.imshow(img, interpolation="lanczos", aspect="auto")
    ax.set_axis_off()
    return ax


def _text(fig, width, height, x, y, s, **kw):
    fig.text(x / width, 1 - y / height, s, **kw)


def layout_a(root, out_dir, window=None, home=None, moments=7, adapted_label=DEFAULT_ADAPTED_LABEL,
             true_scale=0.4, width=fs.TEXT_WIDTH, stem="fig_teaser"):
    """Layout A; returns the written paths and the sidecar record."""
    unseen, home_name = choose(root, window, home)
    ws = windows(root)
    man = manifest_of(root, ws[unseen])
    ctrl = controls_of(man)
    rows = {k: os.path.join(ws[unseen], v) for k, v in ROWS.items()}
    for d in rows.values():
        if not os.path.isdir(d):
            raise SystemExit(f"{unseen}: missing row {os.path.basename(d)}")
    last = min(last_tic(d) for d in rows.values())
    picks = man.get("moments") or spread(change_moments(ctrl, 1, last), moments)
    if not picks:
        raise SystemExit(f"{unseen}: no control changes in tics 1 to {last}; name --moments in the manifest")
    fs.style()
    sample = frame(rows["model"], picks[0])
    aspect = sample.shape[0] / sample.shape[1]
    label_w = max(text_width(t) for t in ("zero-shot", adapted_label, TRUTH_LABEL)) + 5 / 72
    ref_w = 0.0 if home_name is None else 0.62
    gap_ref = 0.0 if home_name is None else 6 / 72
    fw = (width - label_w - ref_w - gap_ref - (len(picks) - 1) * GUTTER - 0.02) / len(picks)
    fh = fw * aspect
    tw, th = fw * true_scale, fh * true_scale
    lines = max(len(button_names(ctrl[t])) if t < len(ctrl) else 1 for t in picks) or 1
    head = 8 / 72 + lines * fs.ANNOT_PT * 1.05 / 72 + 2 / 72
    height = head + 2 * fh + GUTTER + 2 / 72 + th + 0.04
    fig = _figure(width, height)
    x0 = label_w + ref_w + gap_ref
    record = {"layout": "A", "window": unseen, "home": home_name, "tics": list(picks), "columns": []}
    ps = {k: per_tic_scene(man, v) for k, v in ROWS.items()}
    for i, t in enumerate(picks):
        x = x0 + i * (fw + GUTTER)
        act = action_text(ctrl[t]) if t < len(ctrl) else "?"
        _text(fig, width, height, x + fw / 2, 0.0, f"tic {t}", ha="center", va="top", fontsize=fs.MIN_PT,
              color=fs.CONTEXT_INK)
        _text(fig, width, height, x + fw / 2, head, action_text(ctrl[t], "\n") if t < len(ctrl) else "?",
              ha="center", va="bottom", fontsize=fs.ANNOT_PT, style="italic", linespacing=1.0)
        _place(fig, width, height, x, head, fw, fh, frame(rows["model"], t))
        _place(fig, width, height, x, head + fh + GUTTER, fw, fh, frame(rows["adapted"], t))
        _place(fig, width, height, x + (fw - tw) / 2, head + 2 * fh + GUTTER + 2 / 72, tw, th, frame(rows["truth"], t))
        record["columns"].append({"tic": t, "action": act, "scene_psnr": {k: ps[k].get(t) for k in ("model",
                                                                                                    "adapted")}})
    for y, text in ((head + fh / 2, "zero-shot"), (head + fh + GUTTER + fh / 2, adapted_label),
                    (head + 2 * fh + GUTTER + 2 / 72 + th / 2, TRUTH_LABEL)):
        _text(fig, width, height, label_w - 3 / 72, y, text, ha="right", va="center", fontsize=fs.ANNOT_PT)
    if home_name is not None:
        hman = manifest_of(root, ws[home_name])
        hrows = {k: os.path.join(ws[home_name], ROWS[k]) for k in ("model", "truth")}
        hlast = min(last_tic(d) for d in hrows.values())
        t_home = home_moment(controls_of(hman), ctrl[picks[0]] if picks[0] < len(ctrl) else frozenset(), 1, hlast)
        rw = ref_w
        rh = rw * aspect
        _text(fig, width, height, label_w + rw / 2, 0.0, f"tic {t_home}", ha="center", va="top",
              fontsize=fs.MIN_PT, color=fs.CONTEXT_INK)
        _text(fig, width, height, label_w + rw / 2, head, "training\nmap", ha="center", va="bottom",
              fontsize=fs.ANNOT_PT, linespacing=1.0)
        _place(fig, width, height, label_w, head, rw, rh, frame(hrows["model"], t_home))
        _place(fig, width, height, label_w + (rw - rw * true_scale) / 2, head + 2 * fh + GUTTER + 2 / 72,
               rw * true_scale, rh * true_scale, frame(hrows["truth"], t_home))
        record["home_tic"] = t_home
    paths = fs.save(fig, out_dir, stem)
    return paths + [_sidecar(out_dir, stem, record)], record


def layout_b(root, out_dir, window=None, home=None, actions=4, adapted_label=DEFAULT_ADAPTED_LABEL,
             horizon=HORIZON_B, width=fs.TEXT_WIDTH, stem="fig_teaser_actions", pick=None, slot_height=None,
             reference=True):
    """Layout B; returns the written paths and the sidecar record. `pick` names the rows' actions in order (each
    must start in the rollout with `horizon` tics after it); `slot_height` fixes the page height in inches;
    `reference=False` drops the training-map column."""
    unseen, home_name = choose(root, window, home)
    if not reference:
        home_name = None
    elif home_name is None:
        raise SystemExit("layout B needs a training-map window for its reference column")
    ws = windows(root)
    ctrl = controls_of(manifest_of(root, ws[unseen]))
    rows = {k: os.path.join(ws[unseen], v) for k, v in ROWS.items()}
    last = min(last_tic(d) for d in rows.values()) - horizon
    if reference:
        hctrl = controls_of(manifest_of(root, ws[home_name]))
        hmodel = os.path.join(ws[home_name], ROWS["model"])
        hlast = last_tic(hmodel) - horizon
    chosen, names = [], []
    present = sorted({n for c in ctrl for n in button_names(c)},
                     key=lambda n: (PRIORITY.index(n) if n in PRIORITY else len(PRIORITY), n))
    if pick:
        missing = [n for n in pick if n not in present]
        if missing:
            raise SystemExit(f"{unseen}: the rollout never presses {', '.join(missing)}")
        present, actions = list(pick), len(pick)
    for name in present:
        t = next((t for t in range(1, last + 1) if name in button_names(ctrl[t]) and
                  name not in button_names(ctrl[t - 1])), None)
        if t is not None:
            chosen.append(t)
            names.append(name)
        if len(chosen) == actions:
            break
    if not chosen:
        raise SystemExit(f"{unseen}: no action starts with {horizon} tics after it")
    if pick and len(chosen) < len(pick):
        late = [n for n in pick if n not in names]
        raise SystemExit(f"{unseen}: {', '.join(late)} never starts with {horizon} tics after it")
    fs.style()
    cols = ["context"] + (["training map"] if reference else []) + ["zero-shot", adapted_label, TRUTH_LABEL]
    sample = frame(rows["model"], chosen[0])
    aspect = sample.shape[0] / sample.shape[1]
    label_w = max(text_width(n, style="italic") for n in names) + 5 / 72
    fw = (width - label_w - (len(cols) - 1) * GUTTER - 4 / 72 - 0.02) / len(cols)
    head = 2 * 8 / 72
    n = len(chosen)
    if slot_height is not None:      # the frames shrink until the rows fit the slot
        fw = min(fw, (slot_height - head - (n - 1) * GUTTER - 0.02) / n / aspect)
    fh = fw * aspect
    height = slot_height if slot_height is not None else head + n * fh + (n - 1) * GUTTER + 0.02
    used = label_w + len(cols) * fw + (len(cols) - 1) * GUTTER + 4 / 72
    x0 = max(0.0, (width - used) / 2) if slot_height is not None else 0.0
    fig = _figure(width, height)
    xs = [x0 + label_w + i * (fw + GUTTER) + (4 / 72 if i else 0) for i in range(len(cols))]
    for x, c in zip(xs, cols):
        _text(fig, width, height, x + fw / 2, 0.0, c, ha="center", va="top", fontsize=fs.ANNOT_PT)
    for x in xs[1:]:
        _text(fig, width, height, x + fw / 2, 8 / 72, f"+{horizon} tics", ha="center", va="top", fontsize=fs.MIN_PT,
              color=fs.CONTEXT_INK)
    record = {"layout": "B", "window": unseen, "home": home_name, "horizon": horizon, "columns": cols, "rows": []}
    for r, (t, name) in enumerate(zip(chosen, names)):
        y = head + r * (fh + GUTTER)
        imgs = [frame(rows["truth"], t), frame(rows["model"], t + horizon), frame(rows["adapted"], t + horizon),
                frame(rows["truth"], t + horizon)]
        row = {"tic": t, "action": name, "control": action_text(ctrl[t])}
        if reference:
            th = next((u for u in range(1, hlast + 1) if name in button_names(hctrl[u]) and
                       name not in button_names(hctrl[u - 1])), None) if hlast > 0 else None
            th = th if th is not None else home_moment(hctrl, ctrl[t], 1, hlast)
            imgs.insert(1, frame(hmodel, th + horizon))
            row["home_tic"] = th
        for x, img in zip(xs, imgs):
            _place(fig, width, height, x, y, fw, fh, img)
        _text(fig, width, height, x0 + label_w - 3 / 72, y + fh / 2, name, ha="right", va="center",
              fontsize=fs.ANNOT_PT, style="italic")
        record["rows"].append(row)
    paths = fs.save(fig, out_dir, stem)
    return paths + [_sidecar(out_dir, stem, record)], record


def held_note(steps, button):
    """What the recorded controls did with `button` over the rollout steps shown (a list of button sets, step 1
    first): how many steps from the first it stays pressed, and the other buttons pressed on every one of those
    steps, e.g. "+forward · 7/16"; "held 15/16" when nothing else is held throughout."""
    names = [set(button_names(c)) for c in steps]
    run = next((i for i, c in enumerate(names) if button not in c), len(names))
    co = set.intersection(*names[:run]) - {button} if run else set()
    co = sorted(co, key=lambda n: (PRIORITY.index(n) if n in PRIORITY else len(PRIORITY), n))
    return (f"+{' +'.join(co)} · " if co else "held ") + f"{run}/{len(names)}"


def moment_steps(manifest, first, count):
    """The recorded controls of rollout steps first..first+count-1 from a restart manifest (`control_per_tic`,
    step k produces restart tic k) or a teaser manifest (`per_tic`, by teaser tic); None when absent."""
    per = manifest.get("control_per_tic") or manifest.get("per_tic")
    if not isinstance(per, list):
        return None
    by_tic = {int(r["tic"]): r.get("control") or [] for r in per if isinstance(r, dict) and "tic" in r}
    steps = [by_tic.get(k) for k in range(first, first + count)]
    return None if any(s is None for s in steps) else steps


def layout_c(root, out_dir, rows, adapted_label="adapted", adapted_note="(8 episodes, 4k updates)",
             horizon=HORIZON_B, width=fs.TEXT_WIDTH, max_height=2.05, stem="fig_teaser_C", restart_root=None,
             persistence=True, home_root=None):
    """Layout C (the selection round's form); returns the written paths and the sidecar record. One row per entry
    of `rows` ({"row", "window", "tic", "button", "home": {"window", "tic"}}), in the order given, seven frames: the
    in-domain block (a start of the same button on a training map: its ground-truth context, the U-Net `horizon`
    tics later and the ground truth then) and the unseen block (ground-truth context at the tic the button starts,
    zero-shot, adapted and ground truth `horizon` tics later). One number line under every row: persistence under
    both context frames, scene PSNR under the model frames (labelled once, "scene PSNR (dB)", at the left of the
    first row), and under each ground-truth frame what the recorded controls did (`held_note`).

    With `restart_root` (the steward's `<window>_t<T>/{truth_raw,unet_tuned,adapter_tuned}/tic_00..16` export, 32
    ground-truth tics ending at T as context) every frame is `horizon` tics after the context shown, and a missing
    moment stops the build. Without it the frames come from the one closed-loop rollout per window, so the note
    says "rollout tic t+8" and each context frame carries its rollout tic. `persistence=False` leaves the
    persistence numbers off the context frames (they stay in the sidecar) for a caption that states them once.
    `home_root` is a second export the in-domain windows may come from (the training maps 3 to 5 export)."""
    ws = {**(windows(home_root) if home_root else {}), **windows(root)}
    fs.style()
    restart = restart_root is not None

    def source(window, tic):
        """(directory, context tic, +horizon tic) of one moment in the chosen export."""
        if restart:
            d = os.path.join(restart_root, f"{window}_t{tic}")
            if not os.path.isdir(d):
                raise SystemExit(f"the restart export has no {d}")
            return d, 0, horizon
        return ws[window], tic, tic + horizon

    psnr_label = "scene PSNR (dB)"
    label_w = max([text_width(r["button"], style="italic") for r in rows] +
                  [text_width(psnr_label, size=fs.MIN_PT)]) + 5 / 72
    group_gap = 6 / 72
    ncols = 7
    fixed = label_w + (ncols - 2) * GUTTER + group_gap + 0.02
    head = 4 * 8 / 72
    number_line = 7 / 72
    n = len(rows)
    d0, c0, _ = source(rows[0]["window"], rows[0]["tic"])
    sample = frame(os.path.join(d0, ROWS["truth"]), c0)
    aspect = sample.shape[0] / sample.shape[1]
    fw = (width - fixed) / ncols
    fw = min(fw, (max_height - head - n * number_line - (n - 1) * GUTTER - 0.02) / n / aspect)
    fh = fw * aspect
    height = head + n * (fh + number_line) + (n - 1) * GUTTER + 0.02
    x0 = max(0.0, (width - (fixed - 0.02 + ncols * fw)) / 2)
    xs, x = [], x0 + label_w
    for i in range(ncols):
        xs.append(x)
        x += fw + (group_gap if i == 2 else GUTTER)
    away_place = place_name(rows[0]["window"])
    home_places = [place_name(r["home"]["window"]) for r in rows]
    # rows from different training maps: the header says "training maps" and each in-domain context names its map
    mixed = len(set(home_places)) > 1
    home_place = "training maps" if mixed else home_places[0]
    context = "context (t = 0)"
    columns = [{"block": f"{home_place} (in-domain)", "name": context},
               {"block": f"{home_place} (in-domain)", "name": "U-Net"},
               {"block": f"{home_place} (in-domain)", "name": TRUTH_LABEL},
               {"block": away_place, "name": context},
               {"block": away_place, "name": "zero-shot"},
               {"block": away_place, "name": adapted_label},
               {"block": away_place, "name": TRUTH_LABEL}]
    note = (f"rollout frames at +{horizon} tics after 32 ground-truth context tics" if restart else
            f"frames at rollout tic t+{horizon} of one closed-loop rollout")
    fig = _figure(width, height)
    for lo, hi in ((0, 2), (3, 6)):
        _text(fig, width, height, (xs[lo] + xs[hi] + fw) / 2, 0.0, columns[lo]["block"], ha="center", va="top",
              fontsize=fs.ANNOT_PT)
    _text(fig, width, height, (xs[0] + xs[-1] + fw) / 2, 8 / 72, note, ha="center", va="top", fontsize=fs.MIN_PT,
          color=fs.CONTEXT_INK)
    for xi, col in zip(xs, columns):
        _text(fig, width, height, xi + fw / 2, 16 / 72, col["name"], ha="center", va="top", fontsize=fs.ANNOT_PT)
    if adapted_note:
        _text(fig, width, height, xs[5] + fw / 2, 24 / 72, adapted_note, ha="center", va="top", fontsize=fs.MIN_PT,
              color=fs.CONTEXT_INK)
    record = {"layout": "C", "mode": "restart" if restart else "rollout", "restart_root": restart_root,
              "horizon": horizon, "note": note, "size_in": [width, round(height, 4)],
              "frame_in": [round(fw, 4), round(fh, 4)], "columns": columns, "rows": []}
    for r, row in enumerate(rows):
        y = head + r * (fh + number_line + GUTTER)
        hd, hc, h8 = source(row["home"]["window"], row["home"]["tic"])
        ud, uc, u8 = source(row["window"], row["tic"])
        hman, uman = manifest_of(root, hd), manifest_of(root, ud)
        psnr = {"zero-shot": per_tic_scene(uman, ROWS["model"]).get(u8),
                "adapted": per_tic_scene(uman, ROWS["adapted"]).get(u8),
                "in-domain": per_tic_scene(hman, ROWS["model"]).get(h8)}
        copy_scores = {"in-domain": copy_last(hman, h8), "unseen": copy_last(uman, u8)}
        notes = {"in-domain": context_note(copy_scores["in-domain"] if persistence else None, row["home"]["tic"],
                                           restart, home_places[r].replace("training ", "") if mixed else None),
                 "unseen": context_note(copy_scores["unseen"] if persistence else None, row["tic"], restart)}
        held, step_controls = {}, {}
        for key, man, first in (("in-domain", hman, hc + 1), ("unseen", uman, uc + 1)):
            steps = moment_steps(man, first, horizon)
            held[key] = held_note(steps, row["button"]) if steps else None
            # the row's `control` is the context tic's (t = 0); these are what rollout steps 1..horizon executed
            step_controls[key] = [action_text(s) for s in steps] if steps else None
        cells = [("in-domain context", os.path.join(hd, ROWS["truth"]), hc, notes["in-domain"]),
                 ("in-domain", os.path.join(hd, ROWS["model"]), h8, psnr["in-domain"]),
                 ("in-domain ground truth", os.path.join(hd, ROWS["truth"]), h8, held["in-domain"]),
                 ("context", os.path.join(ud, ROWS["truth"]), uc, notes["unseen"]),
                 ("zero-shot", os.path.join(ud, ROWS["model"]), u8, psnr["zero-shot"]),
                 ("adapted", os.path.join(ud, ROWS["adapted"]), u8, psnr["adapted"]),
                 ("ground truth", os.path.join(ud, ROWS["truth"]), u8, held["unseen"])]
        sources = {}
        for xi, (key, row_dir, tic, value) in zip(xs, cells):
            sources[key] = frame_path(row_dir, tic)
            _place(fig, width, height, xi, y, fw, fh, frame(row_dir, tic))
            if value is not None:
                _text(fig, width, height, xi + fw / 2, y + fh + 0.5 / 72,
                      value if isinstance(value, str) else f"{value:.1f}", ha="center", va="top",
                      fontsize=fs.MIN_PT, color=fs.CONTEXT_INK)
        if r == 0:
            _text(fig, width, height, x0 + label_w - 3 / 72, y + fh + 0.5 / 72, psnr_label, ha="right", va="top",
                  fontsize=fs.MIN_PT, color=fs.CONTEXT_INK)
        _text(fig, width, height, x0 + label_w - 3 / 72, y + fh / 2, row["button"], ha="right", va="center",
              fontsize=fs.ANNOT_PT, style="italic")
        shown = (round(psnr["adapted"] - psnr["zero-shot"], 3)
                 if psnr["adapted"] is not None and psnr["zero-shot"] is not None else None)
        record["rows"].append({**row, "scene_psnr": psnr, "persistence": copy_scores, "context_notes": notes,
                               "held": held, "step_controls": step_controls, "story_shown": shown,
                               "sources": sources})
    paths = fs.save(fig, out_dir, stem)
    return paths + [_sidecar(out_dir, stem, record)], record


def context_note(persistence, tic, restart, place=None):
    """The line under a context frame: persistence's scene PSNR when the export has it, else (in a long rollout)
    the frame's rollout tic; `place` ("map 5") first when the block's rows come from different maps."""
    if persistence is not None:
        note = f"persistence {persistence:.1f}"
    else:
        note = None if restart else f"rollout tic {tic}"
    return " · ".join(s for s in (place, note) if s) or None


def copy_last(manifest, tic):
    """Persistence's scene PSNR at `tic` from the restart export's `copylast_scene_psnr`, else None."""
    series = manifest.get("copylast_scene_psnr")
    if isinstance(series, list) and 0 <= tic < len(series) and isinstance(series[tic], (int, float)):
        return float(series[tic])
    return None


def _sidecar(out_dir, stem, record):
    path = os.path.join(out_dir, f"{stem}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description="Compose the teaser from the steward's export.")
    p.add_argument("--root", default=os.path.join(REPO, "results", "teaser"))
    p.add_argument("--out-dir", default=os.path.join(REPO, "paper", "figures"))
    p.add_argument("--layout", choices=("A", "B", "both"), default="both")
    p.add_argument("--window", default=None, help="the unseen-arena window (default: best richness)")
    p.add_argument("--home", default=None, help="the training-map window (default: best richness)")
    p.add_argument("--moments", type=int, default=7, help="layout A: columns")
    p.add_argument("--actions", type=int, default=4, help="layout B: rows")
    p.add_argument("--pick", default=None, help="layout B: the rows' actions in order, comma-separated")
    p.add_argument("--height", type=float, default=None, help="layout B: the page height of a fixed slot (in)")
    p.add_argument("--no-reference", action="store_true", help="layout B: drop the training-map column")
    p.add_argument("--adapted-label", default=DEFAULT_ADAPTED_LABEL)
    p.add_argument("--stem", default=None, help="output stem (default fig_teaser / fig_teaser_actions)")
    a = p.parse_args(argv)
    written = []
    if a.layout in ("A", "both"):
        written += layout_a(a.root, a.out_dir, a.window, a.home, a.moments, a.adapted_label,
                            stem=a.stem or "fig_teaser")[0]
    if a.layout in ("B", "both"):
        stem_b = (a.stem + "_actions") if a.stem and a.layout == "both" else (a.stem or "fig_teaser_actions")
        pick = [n.strip() for n in a.pick.split(",")] if a.pick else None
        written += layout_b(a.root, a.out_dir, a.window, a.home, a.actions, a.adapted_label, stem=stem_b,
                            pick=pick, slot_height=a.height, reference=not a.no_reference)[0]
    for path in written:
        print("wrote", os.path.relpath(path, REPO) if path.startswith(REPO) else path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
