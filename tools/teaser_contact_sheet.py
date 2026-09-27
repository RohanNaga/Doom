"""
The Figure 1 selection round: every action moment of every teaser window on one contact sheet, scored, and the
top candidates composed in layout C (`compose_teaser.layout_c`).

    python tools/teaser_contact_sheet.py --root results/teaser            # sheet + fig_teaser_C_1..3

**Input** (`--root`, the steward's `results/teaser/`): one directory per window, `<map>_ep<E>_s<S>/`, with
`truth_raw/`, `unet_tuned/` and, on an unseen arena, `adapter_tuned/` holding `tic_NNN.png` for rollout tics 1 to
256 (tic 1 is the first predicted tic after 32 ground-truth context tics), and a `manifest.json` whose `per_tic`
list gives the executed control and each model row's scene PSNR per tic.

**Moments.** For each action (attack, turn left, turn right, forward, strafe = strafe left or right), a moment is a
tic t >= 2 where that button is pressed and was not at t - 1, held for at least `HOLD` = 8 tics (t to t + 7), with
the frame t + 8 in the export. Tic 1 never counts: its control is the last context tic's, so nothing starts there.
Note that the +8 frames are rollout tic t + 8 of one closed-loop rollout from tic 0, not 8 tics after a context
that ends at t; `depth` = t + 8 records how long the model has been rolling out.

**Scores** (per moment): `story` = adapter minus zero-shot scene PSNR at t + 8 (unseen arenas; how visible the
adaptation is); the ground-truth +8 frame's `chroma` (mean CIELAB chroma C*, how colourful) and `edges` (the share
of pixels whose Sobel luma gradient exceeds 0.1 of full scale, how much is on screen); on a training map the U-Net's
own scene PSNR at t + 8 (`in_domain`).

**Output**: `<review-dir>/teaser_contacts.png`, one sheet sectioned by action; within an action the unseen-arena
moments sorted by story (best first), then the training-map moments sorted by in-domain PSNR; under each thumbnail
its role and scene PSNR, beside each row the window, tic, control and scores; `teaser_contacts.json` with every
moment and the candidates. **Candidates** are whole unseen windows (one rollout per figure, as the direction asks):
for rows attack, a turn (left or right) and forward, each window's best-story start of that action whose +8 span
does not overlap another row's (so no two rows show the same moment); windows ranked by the three rows' mean
story; each row paired with a real start of the same button on a training map: among those within `NEAR_TICS` = 20
rollout tics of the row and reading at least the training maps' median in-domain PSNR, the one with the brighter
ground truth (mean luma of its context and +8 frames; map 2 is dark); see `pick_home` for the fallbacks, which the
sidecar flags (`near`, `at_floor`); no two rows' in-domain starts share frames. The top `--candidates` are composed as
`<out-dir>/fig_teaser_C_<n>.pdf` (`compose_teaser.layout_c`); with `--restart-root` their +8 frames come from the
steward's rollouts restarted at each moment, and `--restart-horizon 8 16` adds the +16 redraw
(`fig_teaser_C_<n>_restart16`), each rescored from the frames it shows. Actions that never start on a training
map are reported.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import compose_teaser as ct  # noqa: E402

HOLD = 8
HORIZON = 8
ACTIONS = (("attack", ("attack",)), ("turn left", ("turn left",)), ("turn right", ("turn right",)),
           ("forward", ("forward", "move forward")),
           ("strafe", ("strafe left", "strafe right", "move left", "move right")))
FIGURE_ROWS = (("attack", ("attack",)), ("turn", ("turn left", "turn right")), ("forward", ("forward",)))
EDGE_THRESHOLD = 0.1
NEAR_TICS = 20                     # an in-domain start this close in rollout depth counts as matched
FLASH_CHROMA = 12.0                # mean CIELAB chroma above this is a weapon flash: map 2's stone sits at 2 to 5
THUMB = (128, 83)                  # px, 0.4 of the 320 x 208 scene crop
TEXT_W = 318
PAD = 4


# ---------------------------------------------------------------------------------------------
# moments and scores
# ---------------------------------------------------------------------------------------------

def moments(controls, last, hold=HOLD, horizon=HORIZON):
    """[(action, button, tic)]: every start of an action's button at tic >= 2, held `hold` tics, with tic + horizon
    at most `last`."""
    names = [set(ct.button_names(c)) for c in controls]
    out = []
    for action, buttons in ACTIONS:
        for b in buttons:
            for t in range(2, min(last - horizon, len(names) - hold) + 1):
                if b in names[t] and b not in names[t - 1] and all(b in names[u] for u in range(t, t + hold)):
                    out.append((action, b, t))
    return out


def held_moments(controls, last, steps):
    """[(action, button, tic, starts)]: every tic t >= 1 whose button is pressed on each of the next `steps` tics
    (restart steps 1..steps, teaser tics t+1..t+steps), with t + steps at most `last`; `starts` when the button is
    not pressed at t itself, so the action begins right after the context frame (a tie-break only)."""
    names = [set(ct.button_names(c)) for c in controls]
    out = []
    for action, buttons in ACTIONS:
        for b in buttons:
            for t in range(1, min(last, len(names) - 1) - steps + 1):
                if all(b in names[u] for u in range(t + 1, t + steps + 1)):
                    out.append((action, b, t, b not in names[t]))
    return out


def chroma(img):
    """Mean CIELAB chroma C* = sqrt(a*^2 + b*^2) of an RGB uint8 image (D65)."""
    c = img[..., :3].astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return float(np.sqrt(a * a + b * b).mean())


def edge_density(img, threshold=EDGE_THRESHOLD):
    """Share of interior pixels whose Sobel luma gradient magnitude (a unit step gives 1) exceeds `threshold`."""
    y = (img[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])) / 255.0
    gx = (y[:-2, 2:] + 2 * y[1:-1, 2:] + y[2:, 2:]) - (y[:-2, :-2] + 2 * y[1:-1, :-2] + y[2:, :-2])
    gy = (y[2:, :-2] + 2 * y[2:, 1:-1] + y[2:, 2:]) - (y[:-2, :-2] + 2 * y[:-2, 1:-1] + y[:-2, 2:])
    return float((np.hypot(gx, gy) / 4.0 > threshold).mean())


def luma(img):
    """Mean Rec. 601 luma (0 to 255) of an RGB image."""
    return float((img[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])).mean())


def window_moments(root, name, wdir, hold_steps=None):
    """Every scored moment of one window: starts held `HOLD` tics scored at +`HORIZON`, or with `hold_steps` the
    tics whose button is pressed on every one of the next `hold_steps` steps, scored at +`hold_steps`."""
    man = ct.manifest_of(root, wdir)
    ctrl = ct.controls_of(man)
    unseen = ct.is_unseen(name)
    last = ct.last_tic(os.path.join(wdir, ct.ROWS["truth"]))
    zs = ct.per_tic_scene(man, ct.ROWS["model"])
    ad = ct.per_tic_scene(man, ct.ROWS["adapted"]) if unseen else {}
    horizon = hold_steps or HORIZON
    found = (list(held_moments(ctrl, last, hold_steps)) if hold_steps else
             [(a, b, t, None) for a, b, t in moments(ctrl, last)])
    out = []
    for action, button, t, starts in found:
        gt = ct.frame(os.path.join(wdir, ct.ROWS["truth"]), t + horizon)
        context = ct.frame(os.path.join(wdir, ct.ROWS["truth"]), t)
        rec = {"window": name, "role": "unseen" if unseen else "training", "action": action, "button": button,
               "tic": t, "depth": t + horizon, "control": ct.action_text(ctrl[t]),
               "zero_shot" if unseen else "in_domain": zs.get(t + horizon),
               "chroma": round(chroma(gt), 2), "edges": round(edge_density(gt), 4),
               "luma": round((luma(context) + luma(gt)) / 2, 2),
               "flash": max(chroma(context), chroma(gt)) > FLASH_CHROMA}
        if hold_steps:
            held = [set(ct.button_names(c)) for c in ctrl[t + 1:t + hold_steps + 1]]
            rec.update({"held": hold_steps, "starts": starts,
                        "co_held": sorted(set.intersection(*held) - {button})})
        if unseen:
            rec["adapted"] = ad.get(t + horizon)
            rec["story"] = (round(rec["adapted"] - rec["zero_shot"], 3)
                            if rec["adapted"] is not None and rec["zero_shot"] is not None else None)
        out.append(rec)
    return out


def sort_key(m):
    order = [a for a, _ in ACTIONS]
    if m["role"] == "unseen":
        return (order.index(m["action"]), 0, -(m["story"] if m["story"] is not None else -1e9), m["window"], m["tic"])
    return (order.index(m["action"]), 1, -(m["in_domain"] if m["in_domain"] is not None else -1e9), m["window"],
            m["tic"])


# ---------------------------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------------------------

def in_domain_floor(all_moments):
    """The training-map moments' median in-domain scene PSNR at +8."""
    return float(np.median([m["in_domain"] for m in all_moments
                            if m["role"] == "training" and m["in_domain"] is not None]))


def overlaps(window, tic, taken):
    """True when (window, tic) shares frames with a taken start: the same window within `HORIZON` tics."""
    return any(window == w and abs(tic - t) <= HORIZON for w, t in taken)


def pick_home(homes, row, floor, taken=()):
    """The in-domain start for one figure row and how it was found, or (None, None) when every start overlaps one
    in `taken` (another row's in-domain start: a row must not repeat another's frames). Among the starts within
    `NEAR_TICS` rollout tics of the row whose in-domain scene PSNR reaches `floor` (the training-map median: a
    typical read, not a failure such as a weapon flash the model misses), the brightest ground truth wins, since
    map 2 is dark; with none reaching the floor, the near start that reads best; with none near, the closest."""
    homes = [h for h in homes if not overlaps(h["window"], h["tic"], taken)]
    if not homes:
        return None, None
    # a weapon flash lights the frame without showing the map; such a start wins only if nothing else is left
    homes = [h for h in homes if not h.get("flash")] or homes
    near = [h for h in homes if abs(h["depth"] - row["depth"]) <= NEAR_TICS]
    typical = [h for h in near if h["in_domain"] is not None and h["in_domain"] >= floor]
    if typical:
        return max(typical, key=lambda h: (h["luma"], -abs(h["depth"] - row["depth"]))), {"near": True,
                                                                                          "at_floor": True}
    if near:
        best = max(near, key=lambda h: (h["in_domain"] if h["in_domain"] is not None else -1e9))
        return best, {"near": True, "at_floor": False}
    home = min(homes, key=lambda h: (abs(h["depth"] - row["depth"]), -(h["in_domain"] or -1e9)))
    return home, {"near": False, "at_floor": home["in_domain"] is not None and home["in_domain"] >= floor}


def candidates(all_moments, n):
    """The top-n whole unseen windows for the figure rows, each row paired with its closest-depth training start."""
    training = [m for m in all_moments if m["role"] == "training"]
    floor = in_domain_floor(all_moments)
    picks = []
    for window in sorted({m["window"] for m in all_moments if m["role"] == "unseen"}):
        mine = [m for m in all_moments if m["window"] == window and m["story"] is not None]
        rows, used, taken = [], set(), []
        for row, buttons in FIGURE_ROWS:
            options = sorted((m for m in mine if m["button"] in buttons
                              and all(abs(m["tic"] - u) > HORIZON for u in used)),
                             key=lambda m: (-m["story"], -m["chroma"], m["tic"]))
            homes = None
            for m in options:
                homes = [h for h in training if h["button"] == m["button"]]
                if homes:
                    break
            home, flags = pick_home(homes or [], m, floor, taken) if options else (None, None)
            if home is None:
                rows = None
                break
            used.add(m["tic"])
            taken.append((home["window"], home["tic"]))
            rows.append({"row": row, "window": window, "tic": m["tic"], "button": m["button"],
                         "control": m["control"],
                         "story": m["story"], "zero_shot": m["zero_shot"], "adapted": m["adapted"],
                         "chroma": m["chroma"], "edges": m["edges"], "depth": m["depth"],
                         "home": {**{k: home[k] for k in ("window", "tic", "button", "control", "depth",
                                                          "in_domain", "luma")}, **flags}})
        if rows:
            score = float(np.mean([r["story"] for r in rows]))
            picks.append({"window": window, "score": round(score, 3), "rows": rows})
    picks.sort(key=lambda p: (-p["score"], p["window"]))
    return picks[:n]


# ---------------------------------------------------------------------------------------------
# the sheet
# ---------------------------------------------------------------------------------------------

def _font(size, italic=False):
    from matplotlib import font_manager as fm
    path = fm.findfont(fm.FontProperties(family="Arial", style="italic" if italic else "normal"))
    return ImageFont.truetype(path, size)


def _thumb(img):
    return Image.fromarray(img).resize(THUMB, Image.LANCZOS)


def _fmt(v, signed=False):
    return "-" if v is None else (f"{v:+.1f}" if signed else f"{v:.1f}")


def draw_sheet(root, ordered, path, per_line=3):
    """The contact sheet as one PNG."""
    ws = ct.windows(root)
    small, body, head = _font(12), _font(13), _font(22)
    cell_w = 4 * (THUMB[0] + PAD) + TEXT_W
    row_h = THUMB[1] + 16 + PAD
    blocks = []                     # (kind, payload): section headers and rows, laid out per_line rows per line
    for action, _ in ACTIONS:
        for role, what in (("unseen", "unseen arena, by story = adapter - zero-shot scene PSNR at +8 (dB)"),
                           ("training", "training map, by the U-Net's own scene PSNR at +8 (dB)")):
            rows = [m for m in ordered if m["action"] == action and m["role"] == role]
            blocks.append(("head", f"{action}  ·  {what}  ·  {len(rows)} moments"))
            blocks.extend(("row", m) for m in rows)
    lines, cur = [], []
    for kind, payload in blocks:
        if kind == "head":
            if cur:
                lines.append(("rows", cur))
                cur = []
            lines.append(("head", payload))
        else:
            cur.append(payload)
            if len(cur) == per_line:
                lines.append(("rows", cur))
                cur = []
    if cur:
        lines.append(("rows", cur))
    width = per_line * (cell_w + 2 * PAD) + 2 * PAD
    height = sum(34 if k == "head" else row_h for k, _ in lines) + 2 * PAD
    sheet = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(sheet)
    y = PAD
    for kind, payload in lines:
        if kind == "head":
            d.rectangle([0, y, width, y + 30], fill=(235, 235, 235))
            d.text((PAD + 4, y + 3), payload, font=head, fill=(0, 0, 0))
            y += 34
            continue
        for i, m in enumerate(payload):
            x = PAD + i * (cell_w + 2 * PAD)
            wdir = ws[m["window"]]
            t, t8 = m["tic"], m["tic"] + HORIZON
            unseen = m["role"] == "unseen"
            cells = [("truth", t, f"context t{t}"), ("model", t8, ("zero-shot " if unseen else "U-Net ")
                                                     + _fmt(m.get("zero_shot", m.get("in_domain"))))]
            if unseen:
                cells.append(("adapted", t8, "adapted " + _fmt(m["adapted"])))
            cells.append(("truth", t8, f"ground truth t{t8}"))
            for k, (row, tic, caption) in enumerate(cells):
                xx = x + k * (THUMB[0] + PAD)
                sheet.paste(_thumb(ct.frame(os.path.join(wdir, ct.ROWS[row]), tic)), (xx, y))
                d.text((xx, y + THUMB[1] + 1), caption, font=small, fill=(60, 60, 60))
            tx = x + 4 * (THUMB[0] + PAD) + 2
            score = (f"story {_fmt(m['story'], signed=True)} dB" if unseen else
                     f"in-domain {_fmt(m['in_domain'])} dB")
            text = [m["window"].replace("unseen_", "").replace("train_", ""), f"tic {t}, depth {t8}: {m['control']}",
                    score, f"chroma {m['chroma']:.1f}   edges {m['edges']:.3f}"]
            for j, line in enumerate(text):
                d.text((tx, y + 2 + j * 17), line, font=body, fill=(0, 0, 0) if j != 2 else (0, 70, 110))
        y += row_h
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet = sheet.convert("P", palette=Image.ADAPTIVE, colors=256)       # a contact sheet, not a print figure
    sheet.save(path, optimize=True)
    return path


# ---------------------------------------------------------------------------------------------
# the round
# ---------------------------------------------------------------------------------------------

def row_picks(found, n=2, steps=HORIZON):
    """The hold round: per figure row, the top-n unseen moments across every window by story (ties: an action that
    starts right after the context, then chroma), no two sharing frames, among the buttons some training map also
    holds; and the row's in-domain moment, a training-map moment of the first pick's button (`pick_home`: the
    brightest at or above the training maps' median in-domain PSNR; depth does not apply, every restart is `steps`
    deep), no two rows' in-domain moments sharing frames."""
    floor = in_domain_floor(found)
    held_home = {m["button"] for m in found if m["role"] == "training"}
    rows, taken = [], []
    for row, buttons in FIGURE_ROWS:
        options = sorted((m for m in found if m["role"] == "unseen" and m["button"] in buttons
                          and m["button"] in held_home and m["story"] is not None),
                         key=lambda m: (-m["story"], not m["starts"], -m["chroma"], m["window"], m["tic"]))
        unseen = []
        for m in options:
            if not any(m["window"] == o["window"] and abs(m["tic"] - o["tic"]) <= steps for o in unseen):
                unseen.append(m)
            if len(unseen) == n:
                break
        if not unseen:
            rows.append({"row": row, "unseen": [], "home": None, "missing": "no held moment in an unseen arena"})
            continue
        homes = [{**h, "depth": steps} for h in found if h["role"] == "training" and h["button"] == unseen[0]["button"]]
        home, flags = pick_home(homes, {"depth": steps}, floor, taken) if homes else (None, None)
        if home is not None:
            taken.append((home["window"], home["tic"]))
        rows.append({"row": row, "unseen": unseen, "home": home and {**home, **flags}})
    return {"floor": round(floor, 3), "rows": rows}


def restart_home(homes, restart_root, steps, taken=()):
    """Among the training-map moments restarted in `restart_root` (`<window>_t<T>`), the one whose U-Net scene PSNR
    beats persistence's by most at +`steps` (ties, to 0.01 dB: the brighter), weapon flashes excluded unless
    nothing else is left, no frames shared with `taken`; None when no candidate was restarted."""
    scored = []
    for h in homes:
        if overlaps(h["window"], h["tic"], taken):
            continue
        path = os.path.join(restart_root, f"{h['window']}_t{h['tic']}", "manifest.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            man = json.load(f)
        model = ct.per_tic_scene(man, ct.ROWS["model"]).get(steps)
        copy = ct.copy_last(man, steps)
        if model is None or copy is None:
            continue
        scored.append({**h, "in_domain_restart": round(model, 3), "persistence_restart": round(copy, 3),
                       "advantage": round(model - copy, 3)})
    scored = [h for h in scored if not h.get("flash")] or scored
    return max(scored, key=lambda h: (round(h["advantage"], 2), h["luma"])) if scored else None


def hold_figure_rows(picks, k=0, chosen=None):
    """Layout C rows from the hold round: the k-th unseen pick of each row with the row's in-domain moment, or with
    `chosen` ([(row, pick number from 1)]) those rows only, in that order."""
    by_row = {p["row"]: p for p in picks["rows"]}
    wanted = chosen or [(p["row"], k + 1) for p in picks["rows"]]
    out = []
    for row, number in wanted:
        p = by_row.get(row)
        if p is None or p["home"] is None or len(p["unseen"]) < number:
            return None
        m = p["unseen"][number - 1]
        out.append({"row": p["row"], "window": m["window"], "tic": m["tic"], "button": m["button"],
                    "control": m["control"], "story": m["story"], "zero_shot": m["zero_shot"],
                    "adapted": m["adapted"], "depth": m["depth"],
                    "home": {k2: p["home"][k2] for k2 in ("window", "tic", "button", "control", "in_domain", "luma")}})
    return out


def restart_rows(rows, restart_root, found, floor):
    """A candidate's rows re-pointed at the restart export: each unseen moment must be there as `<window>_t<T>`, and
    its in-domain start becomes a restarted training-map start of the same button (by `pick_home` among those).
    Returns (rows, None), or (None, the first missing piece)."""
    have = {d for d in os.listdir(restart_root) if os.path.isdir(os.path.join(restart_root, d))}
    out, taken = [], []
    for row in rows:
        name = f"{row['window']}_t{row['tic']}"
        if name not in have:
            return None, name
        homes = [m for m in found if m["role"] == "training" and m["button"] == row["button"]
                 and f"{m['window']}_t{m['tic']}" in have]
        if not homes:
            return None, f"no restarted {row['button']} start on a training map"
        # every restarted rollout is HORIZON tics deep, so depth no longer separates the starts
        home, flags = pick_home([{**h, "depth": HORIZON} for h in homes], {**row, "depth": HORIZON}, floor, taken)
        if home is None:
            return None, f"every restarted {row['button']} start overlaps another row's in-domain start"
        taken.append((home["window"], home["tic"]))
        keep = ("window", "tic", "button", "control", "depth", "in_domain", "luma")
        out.append({**row, "home": {**{k: home[k] for k in keep}, **flags}})
    return out, None


def run(root, out_dir, review_dir, n_candidates=3, restart_root=None, restart_horizons=(HORIZON,),
        row_order=None, hold_steps=None, hold_rows=None, max_height=2.05):
    """Score every moment, draw the sheet, compose the top candidates; returns the sidecar record. With
    `hold_steps` it runs the hold round instead (`run_hold`)."""
    ws = ct.windows(root)
    if not ws:
        raise SystemExit(f"no window directories <map>_ep<E>_s<S> under {root}")
    if hold_steps:
        return run_hold(root, out_dir, review_dir, hold_steps, restart_root, row_order, hold_rows, max_height)
    found = [m for name, wdir in ws.items() for m in window_moments(root, name, wdir)]
    ordered = sorted(found, key=sort_key)
    starts_on_map2 = {m["action"] for m in found if m["role"] == "training"}
    missing = sorted(a for a, _ in ACTIONS if a not in starts_on_map2)
    picks = candidates(found, n_candidates)

    def in_order(rows):
        """The rows in the figure's order (the review round: strongest story first); the in-domain starts are
        assigned in FIGURE_ROWS order before this, so the order shown never changes which start a row gets."""
        return sorted(rows, key=lambda r: list(row_order).index(r["row"])) if row_order else rows
    sheet = draw_sheet(root, ordered, os.path.join(review_dir, "teaser_contacts.png"))
    written = [sheet]
    floor = in_domain_floor(found)
    for i, pick in enumerate(picks, 1):
        paths, _ = ct.layout_c(root, out_dir, in_order(pick["rows"]), stem=f"fig_teaser_C_{i}")
        pick["files"] = [os.path.basename(p) for p in paths]
        written += paths
        if restart_root:
            rows, skipped = restart_rows(pick["rows"], restart_root, found, floor)
            pick["restart"] = {"skipped": skipped}
            for h in restart_horizons if rows else ():
                stem = f"fig_teaser_C_{i}_restart" + ("" if h == HORIZON else str(h))
                paths, side = ct.layout_c(root, out_dir, in_order(rows), stem=stem, restart_root=restart_root,
                                          horizon=h)
                shown = [r["story_shown"] for r in side["rows"] if r["story_shown"] is not None]
                pick["restart"][str(h)] = {"files": [os.path.basename(p) for p in paths], "rows": side["rows"],
                                           "score_shown": round(float(np.mean(shown)), 3) if shown else None}
                written += paths
    rec = {"root": root, "hold": HOLD, "horizon": HORIZON, "edge_threshold": EDGE_THRESHOLD,
           "counts": {f"{role}:{a}": sum(1 for m in found if m["role"] == role and m["action"] == a)
                      for role in ("unseen", "training") for a, _ in ACTIONS},
           "missing_on_map2": missing, "candidates": picks, "moments": ordered}
    side = os.path.join(review_dir, "teaser_contacts.json")
    with open(side, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    rec["written"] = written + [side]
    return rec


def run_hold(root, out_dir, review_dir, steps, restart_root=None, row_order=None, hold_rows=None,
             max_height=2.05):
    """The hold round: every moment whose button is pressed on all `steps` restart steps, counted per button and
    role; the per-row picks (`row_picks`); layout C of the first and second picks at +`steps` from the continuous
    rollouts (`fig_teaser_C_hold<steps>[_alt]`) and, when the restart export holds every moment, from it
    (`..._restart`); with `hold_rows` (["forward:2", "attack:1"]: row and pick number) also those rows only, in
    that order, at most `max_height` tall (`..._pick`). Writes `<review-dir>/teaser_hold<steps>.json`."""
    ws = ct.windows(root)
    found = [m for name, wdir in ws.items() for m in window_moments(root, name, wdir, hold_steps=steps)]
    buttons = ("attack", "turn left", "turn right", "forward", "strafe left", "strafe right")
    counts = {f"{role}:{b}": sum(1 for m in found if m["role"] == role and m["button"] == b)
              for role in ("unseen", "training") for b in buttons}
    picks = row_picks(found, n=2, steps=steps)
    if restart_root and os.path.isdir(restart_root):
        # with restarts on disk the in-domain moment is the restarted one that beats persistence by most at +steps
        taken = []
        for p in picks["rows"]:
            if not p["unseen"]:
                continue
            homes = [m for m in found if m["role"] == "training" and m["button"] == p["unseen"][0]["button"]]
            best = restart_home(homes, restart_root, steps, taken)
            if best is not None:
                p["home_proxy"], p["home"] = p["home"], {**best, "rule": "restarted advantage over persistence"}
                taken.append((best["window"], best["tic"]))
    order = list(row_order) if row_order else [r for r, _ in FIGURE_ROWS]
    written = []
    have = (set(os.listdir(restart_root)) if restart_root and os.path.isdir(restart_root) else set())
    chosen = [(spec.split(":")[0], int(spec.split(":")[1])) for spec in hold_rows] if hold_rows else None
    for k, suffix, subset in ((0, "", None), (1, "_alt", None), (0, "_pick", chosen)):
        if suffix == "_pick" and not chosen:
            continue
        rows = hold_figure_rows(picks, k, subset)
        if rows is None:
            continue
        if not subset:
            rows = sorted(rows, key=lambda r: order.index(r["row"]))
        stem = f"fig_teaser_C_hold{steps}{suffix}"
        height = max_height if subset else 2.05
        written += ct.layout_c(root, out_dir, rows, stem=stem, horizon=steps, max_height=height)[0]
        need = [f"{r['window']}_t{r['tic']}" for r in rows] + [f"{r['home']['window']}_t{r['home']['tic']}"
                                                               for r in rows]
        if restart_root and all(d in have for d in need):
            written += ct.layout_c(root, out_dir, rows, stem=stem + "_restart", horizon=steps,
                                   restart_root=restart_root, max_height=height)[0]
        elif restart_root:
            picks.setdefault("restart_missing", {})[stem] = [d for d in need if d not in have]
    rec = {"root": root, "hold_steps": steps, "counts": counts, **picks}
    side = os.path.join(review_dir, f"teaser_hold{steps}.json")
    os.makedirs(review_dir, exist_ok=True)
    with open(side, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    return {"hold": rec, "written": written + [side], "candidates": [], "missing_on_map2": []}


def main(argv=None):
    p = argparse.ArgumentParser(description="The Figure 1 selection round: contact sheet and candidates.")
    p.add_argument("--root", default=os.path.join(REPO, "results", "teaser"))
    p.add_argument("--out-dir", default=os.path.join(REPO, "paper", "figures"))
    p.add_argument("--review-dir", default=os.path.join(REPO, "paper", "figures", "review"))
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--restart-root", default=None,
                   help="the steward's restarted rollouts, <window>_t<T>/...: the candidates' +8 frames come from it")
    p.add_argument("--row-order", default=None, help="the rows' order, e.g. turn,forward,attack")
    p.add_argument("--hold-steps", type=int, default=None,
                   help="the hold round: moments whose button is pressed on every one of the next N restart steps")
    p.add_argument("--hold-rows", default=None,
                   help="the hold round's chosen figure: row:pick pairs in order, e.g. forward:2,attack:1")
    p.add_argument("--max-height", type=float, default=2.05, help="the chosen figure's page height at most (in)")
    p.add_argument("--restart-horizon", type=int, nargs="+", default=[HORIZON],
                   help="tics after the context for the restarted redraws (the export holds 16); stems gain the "
                        "number past 8, fig_teaser_C_<n>_restart16")
    a = p.parse_args(argv)
    rec = run(a.root, a.out_dir, a.review_dir, a.candidates, a.restart_root, tuple(a.restart_horizon),
              tuple(a.row_order.split(",")) if a.row_order else None, a.hold_steps,
              tuple(a.hold_rows.split(",")) if a.hold_rows else None, a.max_height)
    for path in rec["written"]:
        print("wrote", os.path.relpath(path, REPO) if path.startswith(REPO) else path)
    for i, pick in enumerate(rec["candidates"], 1):
        rows = ", ".join(f"{r['row']} t{r['tic']} ({r['story']:+.2f} dB; map 2 {r['home']['window']} "
                         f"t{r['home']['tic']})" for r in pick["rows"])
        print(f"candidate {i}: {pick['window']} mean story {pick['score']:+.2f} dB: {rows}")
    print("never starts on map 2:", ", ".join(rec["missing_on_map2"]) or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
