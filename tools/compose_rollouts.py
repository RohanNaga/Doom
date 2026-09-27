"""
Figure 2, rollouts under held controls, composed from the steward's per-frame export.

    python tools/compose_rollouts.py                                  # results/figure2_rollouts -> paper/figures
    python tools/compose_rollouts.py --stack                          # the two controls stacked, frames twice as wide

**Input** (`--root`, the steward's `export_v2/figure2_rollouts/` mirrored to `results/figure2_rollouts/`): one
directory per window, `<map>_<control>/` (`train_map02_turn_left`, `unseen_arena07_forward`, ...), holding one
directory per row, `<row>_<decoder>/` (`truth_raw`, `truth_stock`, `truth_tuned`, `unet_stock`, `unet_tuned`,
`adapter_stock`, `adapter_tuned`; the adapter only on the unseen arena), each with `tic_NN.png` for tic 0 (the
last context frame: raw for `truth_raw`, the last context latent decoded for the decoder rows) to 32 and the scene
crop `tic_NN_scene.png` (rows 0 to 207) beside every frame; and a `manifest.json` (in the window directory, or one
at the root keyed by window) with each row's per-tic full-frame and scene PSNR and copy-last's as
`copylast_{raw,stock,tuned}` (the steward's shape: a root `manifest.json` whose `windows` list holds per window
its `map`, `control`, `episode`, `start_row`, `dir` and `per_tic: {row: {psnr: [tic 0..32], scene_psnr: [...]}}`).

**Figure** (`paper/FIGURE_STANDARDS.md`, Figure 2): the full 5.5 in width; two column blocks, one per held control,
each a tic-0 context column (copy-last's prediction at every tic, so copy-last needs no row) and tics 1, 2, 4, 8, 16
and 32; rows in two groups, map 2 (true, U-Net) and arena 7 (true, zero-shot, the adapter after 4k updates), each
map with its own true row; the scene crop; the tuned decoder for every model row and the raw frame for true rows;
2 pt white gutters; each model row's per-tic scene PSNR in small grey numbers beneath it. Row labels sit at the
left, group labels left of them, tic numbers above the first row and the held control above each block.

**Output**: `<out-dir>/fig2_rollouts.pdf` and `.png` through `figstyle.save`, and `fig2_rollouts.json` with the
windows (episode, start tic), the files drawn and the PSNR numbers printed, for the caption and the checks. A missing
frame stops the build with its path; a model row with no PSNR in the manifest is drawn without numbers and noted.
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

TICS = (0, 1, 2, 4, 8, 16, 32)
CONTROLS = ("turn_left", "forward")
CONTROL_LABELS = {"turn_left": "held turn left", "turn_right": "held turn right", "forward": "held forward",
                  "strafe": "held strafe", "fight": "fight"}
# (window map, group label, [(row label, row directory stem)]); the decoder suffix is added per row
GROUPS = (("train_map02", "map 2\n(training)", (("true", "truth"), ("U-Net", "unet"))),
          ("unseen_arena07", "arena 7\n(unseen)", (("true", "truth"), ("zero-shot", "unet"), ("LoRA 4k", "adapter"))))
SCENE_ROWS = 208                  # rows 0 to 207 of the frame are the scored scene; the HUD lies below
GUTTER = 2 / 72                   # in, between frames
CONTEXT_GAP = 4 / 72              # in, after the tic-0 column
BLOCK_GAP = 8 / 72                # in, between the two control blocks
ROW_GAP = 2 / 72
GROUP_GAP = 4 / 72
PSNR_LINE = 7.5 / 72              # in, the line of PSNR numbers under a model row
HEADER_LINE = 9 / 72              # in, one header line (control, then tics)
WINDOW_RE = re.compile(r"^(?P<map>(?:train_map|unseen_arena|map|arena)\d+)_(?P<control>.+)$")


def window_dirs(root):
    """{(map, control): directory} for every window directory under `root`."""
    out = {}
    for d in sorted(os.listdir(root)):
        m = WINDOW_RE.match(d)
        if m and os.path.isdir(os.path.join(root, d)):
            out[(m["map"], m["control"])] = os.path.join(root, d)
    return out


def load_manifest(root, window_dir):
    """The window's manifest: `<window>/manifest.json`, else its entry in `<root>/manifest.json`, else {}."""
    p = os.path.join(window_dir, "manifest.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    p = os.path.join(root, "manifest.json")
    if os.path.exists(p):
        with open(p) as f:
            js = json.load(f)
        name = os.path.basename(window_dir)
        windows = js.get("windows", js) if isinstance(js, dict) else js
        if isinstance(windows, dict):
            return windows.get(name, {})
        if isinstance(windows, list):
            return next((e for e in windows if isinstance(e, dict) and name in (
                e.get("window"), os.path.basename(str(e.get("dir", "")).rstrip("/")),
                f"{e.get('map')}_{e.get('control')}")), {})
    return {}


def as_tic_series(v):
    """{tic: value} from a per-tic series: a list over tics 0..N (N+1 entries) or 1..N (N entries), or a dict keyed
    by tic ("1", "tic_01", 1)."""
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            m = re.search(r"(\d+)$", str(k))
            if m and isinstance(x, (int, float)):
                out[int(m.group(1))] = float(x)
        return out
    if isinstance(v, list):
        start = 0 if len(v) in (33, 17, 9, 5) or len(v) % 2 == 1 else 1
        return {i + start: float(x) for i, x in enumerate(v) if isinstance(x, (int, float))}
    return {}


def per_tic(manifest, row, kind="scene_psnr"):
    """A row's per-tic series of `kind` from the manifest, in whichever of the known shapes it is stored."""
    places = []
    for container in (manifest.get("per_tic"), manifest.get("rows"), manifest.get("series"), manifest):
        if isinstance(container, dict) and isinstance(container.get(row), dict):
            places.append(container[row])
    for place in places:
        for k in (kind, f"{kind}_per_tic", f"per_tic_{kind}"):
            if k in place:
                return as_tic_series(place[k])
    for k in (kind, f"per_tic_{kind}"):
        if isinstance(manifest.get(k), dict) and row in manifest[k]:
            return as_tic_series(manifest[k][row])
    return as_tic_series(manifest.get(f"{row}_{kind}"))


def frame(row_dir, tic, scene=True):
    """One frame as an RGB array: the scene crop file, else the full frame cropped to the scene rows."""
    base = os.path.join(row_dir, f"tic_{tic:02d}")
    if scene and os.path.exists(base + "_scene.png"):
        return np.asarray(Image.open(base + "_scene.png").convert("RGB"))
    if not os.path.exists(base + ".png"):
        raise SystemExit(f"missing frame {base}.png (and no {base}_scene.png)")
    img = np.asarray(Image.open(base + ".png").convert("RGB"))
    return img[:SCENE_ROWS] if scene else img


def plan(root, controls, decoder, truth, notes):
    """The blocks and groups to draw: [(control, header, {group map: window dir})] and the group rows."""
    windows = window_dirs(root)
    blocks = []
    for control in controls:
        found = {g[0]: windows.get((g[0], control)) for g in GROUPS}
        missing = [m for m, d in found.items() if d is None]
        if missing:
            raise SystemExit(f"no window directory for {missing} under control {control!r} in {root}")
        blocks.append((control, CONTROL_LABELS.get(control, control.replace("_", " ")), found))
    groups = []
    for gmap, glabel, rows in GROUPS:
        groups.append((gmap, glabel, [(label, f"{stem}_{truth if stem == 'truth' else decoder}", stem != "truth")
                                      for label, stem in rows]))
    return blocks, groups


def compose(root, out_dir, controls=CONTROLS, tics=TICS, decoder="tuned", truth="raw", stack=False,
            width=fs.TEXT_WIDTH, stem="fig2_rollouts"):
    """Compose Figure 2 and write it with its sidecar; returns the written paths and the notes."""
    notes = []
    blocks, groups = plan(root, controls, decoder, truth, notes)
    fs.style()
    import matplotlib.pyplot as plt

    # the frames' aspect from the first true frame
    sample = frame(os.path.join(blocks[0][2][groups[0][0]], groups[0][2][0][1]), tics[0])
    aspect = sample.shape[0] / sample.shape[1]
    measure = plt.figure(figsize=(width, 1))
    renderer = measure.canvas.get_renderer()

    def text_width(s, size):
        t = measure.text(0, 0, s, fontsize=size)
        w = t.get_window_extent(renderer).width / measure.dpi
        t.remove()
        return w

    row_w = max(text_width(label, fs.ANNOT_PT) for _, _, rows in groups for label, _, _ in rows) + 4 / 72
    lines = max(g[1].count("\n") + 1 for g in groups)
    group_w = lines * fs.ANNOT_PT * 1.1 / 72 + 2 / 72          # the group labels are rotated, one column per line
    plt.close(measure)
    ncols = len(tics)
    per_block = ncols - 1
    n_side = 1 if stack else len(blocks)
    left = group_w + row_w
    avail = width - left - 0.02 - (n_side - 1) * BLOCK_GAP - n_side * ((per_block - 1) * GUTTER + CONTEXT_GAP)
    fw = avail / (n_side * ncols)
    fh = fw * aspect

    def group_height(rows):
        return sum(fh + (PSNR_LINE if model else 0.0) for _, _, model in rows) + ROW_GAP * (len(rows) - 1)

    body = sum(group_height(rows) for _, _, rows in groups) + GROUP_GAP * (len(groups) - 1)
    stacks = len(blocks) if stack else 1
    height = stacks * (2 * HEADER_LINE + body) + (stacks - 1) * GROUP_GAP * 2 + 0.02
    fig = plt.figure(figsize=(width, height))
    record = {"root": os.path.relpath(root, REPO) if root.startswith(REPO) else root, "decoder": decoder,
              "truth": truth, "tics": list(tics), "frame_in": [round(fw, 4), round(fh, 4)], "blocks": []}

    def place(x, y, w, h):
        return fig.add_axes([x / width, 1 - (y + h) / height, w / width, h / height])

    def fig_text(x, y, s, **kw):
        fig.text(x / width, 1 - y / height, s, **kw)

    for b, (control, header, found) in enumerate(blocks):
        bx = left + (0 if stack else b * (ncols * fw + (per_block - 1) * GUTTER + CONTEXT_GAP + BLOCK_GAP))
        by = (b * (2 * HEADER_LINE + body + 2 * GROUP_GAP)) if stack else 0.0
        xs = [bx + (0 if i == 0 else fw + CONTEXT_GAP + (i - 1) * (fw + GUTTER)) for i in range(ncols)]
        block_w = xs[-1] + fw - bx
        fig_text(bx + block_w / 2, by, header, ha="center", va="top", fontsize=fs.LABEL_PT)
        for x, tic in zip(xs, tics):
            fig_text(x + fw / 2, by + HEADER_LINE, str(tic), ha="center", va="top", fontsize=fs.TICK_PT)
        y = by + 2 * HEADER_LINE
        entry = {"control": control, "windows": {}}
        for gmap, glabel, rows in groups:
            wdir = found[gmap]
            manifest = load_manifest(root, wdir)
            entry["windows"][gmap] = {"dir": os.path.basename(wdir),
                                      **{k: manifest.get(k) for k in ("episode", "start", "start_tic", "start_row",
                                                                      "tic_00_game_tic", "seed") if k in manifest},
                                      "rows": {}}
            gy = y
            for label, row, model in rows:
                row_dir = os.path.join(wdir, row)
                if not os.path.isdir(row_dir):
                    raise SystemExit(f"missing row directory {row_dir}")
                for x, tic in zip(xs, tics):
                    ax = place(x, y, fw, fh)
                    ax.imshow(frame(row_dir, tic), interpolation="lanczos", aspect="auto")
                    ax.set_axis_off()
                if b == 0 or stack:
                    fig_text(left - 3 / 72, y + fh / 2, label, ha="right", va="center", fontsize=fs.ANNOT_PT)
                shown = {}
                if model:
                    series = per_tic(manifest, row)
                    if not series:
                        notes.append(f"{os.path.basename(wdir)}/{row}: no per-tic scene PSNR in the manifest")
                    for x, tic in zip(xs[1:], tics[1:]):
                        if tic in series:
                            shown[tic] = round(series[tic], 1)
                            fig_text(x + fw / 2, y + fh + 0.5 / 72, f"{series[tic]:.1f}", ha="center", va="top",
                                     fontsize=fs.MIN_PT, color=fs.CONTEXT_INK)
                entry["windows"][gmap]["rows"][row] = {"label": label, "scene_psnr": shown}
                y += fh + (PSNR_LINE if model else 0.0) + ROW_GAP
            y += GROUP_GAP - ROW_GAP
            if b == 0 or stack:
                gh = y - GROUP_GAP - gy
                fig_text(group_w / 2, gy + gh / 2, glabel, ha="center", va="center", rotation=90,
                         fontsize=fs.ANNOT_PT, linespacing=1.0)
        record["blocks"].append(entry)
    paths = fs.save(fig, out_dir, stem)
    side = os.path.join(out_dir, f"{stem}.json")
    with open(side, "w") as f:
        json.dump({**record, "notes": notes}, f, indent=1)
        f.write("\n")
    return paths + [side], notes


def main(argv=None):
    p = argparse.ArgumentParser(description="Compose Figure 2 (rollouts under held controls) from the export.")
    p.add_argument("--root", default=os.path.join(REPO, "results", "figure2_rollouts"))
    p.add_argument("--out-dir", default=os.path.join(REPO, "paper", "figures"))
    p.add_argument("--controls", nargs="+", default=list(CONTROLS))
    p.add_argument("--tics", nargs="+", type=int, default=list(TICS))
    p.add_argument("--decoder", default="tuned", help="the model rows' decoder (row directory suffix)")
    p.add_argument("--truth", default="raw", help="the true rows' source (raw, stock or tuned)")
    p.add_argument("--stack", action="store_true", help="stack the control blocks instead of side by side")
    a = p.parse_args(argv)
    if a.tics[0] != 0:
        raise SystemExit("the first tic column is the tic-0 context frame: --tics must start with 0")
    paths, notes = compose(a.root, a.out_dir, a.controls, tuple(a.tics), a.decoder, a.truth, a.stack)
    for path in paths:
        print("wrote", os.path.relpath(path, REPO) if path.startswith(REPO) else path)
    for n in notes:
        print("note:", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
